"""exp-045-harvest: re-derive one month's ground-segment summaries from its stored points.

The assignment provides for exactly this: "if the pilot changes the rule, re-derive the
summaries from the stored points rather than re-fetching".  The pilot's reviewer repaired
the rule on 2026-09-08 (a first-airborne point with readsb's null altitude is not evidence
of a take-off), after four months had already been harvested under the shipped rule.  The
points are untouched by the change — the repair reads `alt_ft`, which every points file
carries — so this kernel mounts a fetch kernel's output, applies the current rule to it,
and writes the summaries again.  Nothing is re-fetched and no archive is read.

The mounted month is whatever `kernel_sources` names in the metadata; the code takes every
`points_*.parquet` under /kaggle/input.

Outputs, in /kaggle/working:
  segments_<day>.parquet   the summaries under the current rule
  callsigns_<day>.parquet  the ground callsign table, unchanged in definition
  rederive_<day>.json      row counts, the rule hash, and what the rule changed
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

CFG = {"out": "/kaggle/working", "input": "/kaggle/input"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(glob.glob(f"{CFG['input']}/*/points_*.parquet"))
    print(f"rule {RULE_VERSION} {RULE_SHA256}  REQUIRE_ALT={REQUIRE_ALT}  "
          f"{len(files)} points files", flush=True)
    out = CFG["out"]
    total = {"days": 0, "segments": 0, "departures": 0, "dropped_by_require_alt": 0}
    for path in files:
        t0 = time.time()
        day = os.path.basename(path)[7:17]
        pdf = pd.read_parquet(path)
        sdf = all_segments(pdf)
        # what the repair actually changed on this day, priced rather than argued
        raw_dep = int((sdf.ends_airborne | (sdf.airborne_alt_ok != sdf.ends_airborne)).sum())
        dropped = int(((~sdf.ends_airborne) & sdf.kind.ne("departure")
                       & sdf.airborne_alt_ok.eq(False)).sum()) if REQUIRE_ALT else 0
        g = pdf[pdf.on_ground & pdf.in_box]
        cs = (g.dropna(subset=["callsign"])
              .groupby(["airport", "callsign"], observed=True)
              .agg(n_points=("t", "size"), n_icao=("icao", "nunique"),
                   t_first=("t", "min"), t_last=("t", "max")).reset_index())
        files_out = {}
        for name, df in (("segments", sdf), ("callsigns", cs)):
            p = f"{out}/{name}_{day}.parquet"
            df.to_parquet(p, index=False, compression="zstd")
            files_out[name] = {"rows": int(len(df)), "bytes": os.path.getsize(p),
                               "sha256": _sha256(p)}
        rec = {"day": day, "source": path, "rule_sha256": RULE_SHA256,
               "rule_version": RULE_VERSION,
               "require_alt": bool(REQUIRE_ALT),
               "points_rows": int(len(pdf)), "outputs": files_out,
               "departures": int((sdf.kind == "departure").sum()),
               "segments_without_a_real_altitude": int((~sdf.airborne_alt_ok
                                                        & sdf.ends_stationary.notna()).sum()),
               "seconds": round(time.time() - t0, 1)}
        with open(f"{out}/rederive_{day}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        total["days"] += 1
        total["segments"] += len(sdf)
        total["departures"] += rec["departures"]
        print(f"{day}: {len(sdf)} segments, {rec['departures']} departures, "
              f"{rec['seconds']} s", flush=True)
    with open(f"{out}/rederive_session.json", "w") as fh:
        json.dump(total, fh, indent=2)
    print(json.dumps(total), flush=True)
    print("done", flush=True)



"""exp-043-pilot: turn kept trace points into departure ground segments, and match them.

One "departure event" is a ground -> airborne transition inside an aircraft-day
(the `first_airborne` point the kernel keeps).  Its ground segment is the run of
on-ground points before it with no time gap larger than `GAP`.  From that segment:

  obs_wheels_off  the last on-ground point before the transition
  obs_offblock    the first SUSTAINED motion (ground speed above 3 kt held for at
                  least 30 s) after the last stationary period at the stand.

**Which stationary period is "the stand" is the one real decision here, and the
data settled it.**  Measured on days 1-2, the network sees a departing aircraft for
a median of 161 s before this rule's 3 kt onset and 284 s before the 250 m one
(measured, and stored under `stand_stop` in `analysis/sweep.json`; a single
150 s was a stale quotation until round 4): transponders
come on near pushback.  Requiring the assignment's long stand stop (600 s) would
have kept only 3.5 % of the development days' departures (3.6 % of the matched
segments) for a margin the network does not provide -- measured, and
`analysis/sweep.json` carries the 600 s arm.  But a
segment cannot simply be anchored at its first motion either, because an aircraft
received continuously through its turnaround has the PREVIOUS arrival's taxi-in
inside the same segment.  So:

    anchor = the end of the last stationary run of at least TURNAROUND seconds
             (a turnaround at the stand), or the start of the segment if there is
             none (the ordinary case: the trace begins at the stand)
    obs_offblock = the start of the first sustained motion run at or after anchor
    stand_stop_s = the stationary time actually observed immediately before it

`stand_stop_s` is the confidence margin, not a filter: `strict` keeps only the
segments where it reaches `STAND_STOP`, `loose` keeps every motion onset including
the traces that begin already moving.  The two are reported side by side and never
mixed, and `RESULTS.md` reports the accuracy as a function of the margin, because
the margin is the only thing that separates a seen pushback from a missed one.
"""

import numpy as np
import pandas as pd

GAP = 1800.0        # s: a larger hole in reception splits the ground segment
MOVE_KT = 3.0       # ground speed above this is motion (assignments/pilot.md)
HOLD_S = 30.0       # motion must be held this long to count as sustained
STAND_STOP = 60.0   # observed stationary time that makes a motion onset trustworthy
TURNAROUND = 600.0  # a stationary run at least this long is a stand turnaround
REQUIRE_ALT = True  # False reproduces the rule the harvest shipped (see below)
STABLE = True       # stable point ordering; False reproduces the rule the pilot ran to round 4
DISP_M = 250.0      # metres from the stand that count as "the aircraft has left it"


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """`[(start, stop_exclusive, value)]` over a boolean array."""
    if not len(mask):
        return []
    cut = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    edges = [0, *cut.tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


def _moving(gs: np.ndarray, lat: np.ndarray, lon: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Motion per point.  Ground speed where it was received, displacement where
    it was not: a null `gs` is unknown, and calling it stationary would invent a
    stand stop that the network never saw."""
    mv = gs > MOVE_KT
    miss = np.isnan(gs)
    if miss.any() and len(t) > 1:
        # metres per second from the neighbouring positions, 1 kt = 0.5144 m/s
        dx = np.gradient(lon) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
        dy = np.gradient(lat) * 110540.0
        dt = np.gradient(t)
        with np.errstate(invalid="ignore", divide="ignore"):
            spd = np.hypot(dx, dy) / np.where(dt == 0, np.nan, dt) / 0.5144
        mv[miss] = spd[miss] > MOVE_KT
    return np.nan_to_num(mv, nan=0.0).astype(bool)


def _span(t: np.ndarray, a: int, b: int) -> float:
    """Seconds a run covers, counting the step out of it: a single point that is
    followed 60 s later by another moving point has held motion for 60 s."""
    if b - a >= 2:
        return float(t[b - 1] - t[a])
    return float(t[b] - t[a]) if b < len(t) else 0.0


def segments(points: pd.DataFrame) -> pd.DataFrame:
    """One row per departure event seen in `points` (the kernel's slice)."""
    out = []
    # `kind="stable"` is not a preference.  Points that share an (icao, t) exist at
    # the UTC-midnight overlap and a few disagree about `on_ground`; an unstable sort
    # puts them either way round and the ground run splits at a different point.  The
    # harvest measured that as 14 and 17 segments moving between this laptop and the
    # Kaggle image on its two stored days -- the same code, the same points, two
    # answers.  A stable sort keeps the order readsb wrote the trace in, which is the
    # truth about which of two same-second points came first, and gives one answer on
    # every machine.  The pilot ran unstable until round 4 found it; `analysis/
    # divergence.json` prices the change.
    order = "stable" if STABLE else "quicksort"
    for (icao,), g in points.sort_values("t", kind=order).groupby(["icao"], sort=False):
        g = g.reset_index(drop=True)
        t = g["t"].to_numpy("float64")
        grd = g["on_ground"].to_numpy(bool)
        # the kernel marks the first non-ground point after a ground run as
        # `first_airborne`, and readsb writes `null` altitude for a position with
        # no altitude at all.  A null is not evidence of a take-off, so a
        # transition into one is not read as one (the reviewer's repair).
        #
        # `REQUIRE_ALT = False` is the SHIPPED rule the harvest ran, kept here so
        # the divergence can be priced rather than argued.  `analysis/divergence.json`
        # holds the comparison; the harvest's rule is not changed by this file.
        air = g["first_airborne"].to_numpy(bool)
        if REQUIRE_ALT:
            air = air & g["alt_ft"].notna().to_numpy()
        gs = g["gs_kt"].to_numpy("float64")
        lat, lon = g["lat"].to_numpy("float64"), g["lon"].to_numpy("float64")
        for k in np.flatnonzero(air):
            j = k - 1
            if j < 0 or not grd[j]:
                continue
            i = j
            while i > 0 and grd[i - 1] and t[i] - t[i - 1] <= GAP:
                i -= 1
            sl = slice(i, j + 1)
            ts, gss = t[sl], gs[sl]
            mv = _moving(gss, lat[sl], lon[sl], ts)
            runs = _runs(mv)
            sustained = [(a, b) for a, b, v in runs if v and _span(ts, a, b) >= HOLD_S]
            turns = [b for a, b, v in runs if not v and _span(ts, a, b) >= TURNAROUND]
            anchor = turns[-1] - 1 if turns else 0
            after = [a for a, _ in sustained if a >= anchor]
            strict = loose = np.nan
            margin = np.nan
            if after:
                k0 = after[0]
                loose = ts[k0]
                prev = [(a, b) for a, b, v in runs if not v and b <= k0]
                margin = _span(ts, *prev[-1]) if prev else 0.0
                if margin >= STAND_STOP:
                    strict = loose
            # a second, independent onset that never reads the broadcast ground
            # speed: the first time the aircraft is DISP_M metres from where it
            # was standing and never comes back inside that radius.  Ground speed
            # is noisy at a stand (jitter fires a low gate at a random time) while
            # a 50 m displacement is far outside any GPS error.
            disp = np.nan
            if after:
                k0 = after[0]
                st_pts = [k for k in range(anchor, k0) if not mv[k]]
                ref = st_pts if st_pts else [anchor]
                rla = float(np.median(lat[sl][ref]))
                rlo = float(np.median(lon[sl][ref]))
                dx = (lon[sl] - rlo) * 111320.0 * np.cos(np.radians(rla))
                dy = (lat[sl] - rla) * 110540.0
                far = np.hypot(dx, dy) > DISP_M
                far[:anchor] = False
                out_idx = np.flatnonzero(far)
                if len(out_idx):
                    # the LAST re-entry inside the radius is the true departure
                    back = np.flatnonzero(~far[out_idx[0]:]) 
                    k1 = out_idx[0]
                    if len(back):
                        later = np.flatnonzero(far[out_idx[0] + back[-1]:])
                        if len(later):
                            k1 = out_idx[0] + back[-1] + later[0]
                    disp = ts[k1]

            cs = g["callsign"].iloc[sl].dropna()
            out.append({
                "icao": icao,
                "callsign_modal": cs.mode().iat[0] if len(cs) else None,
                "callsign_n": int(cs.nunique()),
                "reg": g["reg"].iloc[j], "type": g["type"].iloc[j],
                "obs_wheels_off": t[j],
                "t_airborne": t[k],
                "seg_start": ts[0], "seg_points": int(j - i + 1),
                "seg_span_s": float(ts[-1] - ts[0]),
                "seg_gap_max_s": float(np.diff(ts).max()) if len(ts) > 1 else 0.0,
                "seg_dt_median_s": float(np.median(np.diff(ts))) if len(ts) > 1 else np.nan,
                "stand_stop_s": margin,
                "turnaround_anchored": bool(turns),
                "obs_offblock_strict": strict,
                "obs_offblock_loose": loose,
                "obs_offblock_disp": disp,
                "n_ground_points": int(grd[sl].sum()),
            })
    return pd.DataFrame(out)


def seg_key(d):
    """The stable identity of a departure segment: which aircraft, at which
    take-off instant.  One definition, used by the matcher and by every probe."""
    return d["icao"].astype(str) + "@" + d["obs_wheels_off"].astype(str)

"""exp-045-harvest: every ground segment, not only the departure ones.

`exp-043-pilot/code/segments.py` is the frozen rule and this file does not touch
it: `all_segments()` calls the pilot's `_runs`, `_moving`, `_span` and its five
constants, and reproduces the pilot's per-segment computation line for line, so a
segment whose kind is `departure` carries exactly the numbers `segments()` gives.
`code/check_rule.py` asserts that equality on the pilot's own stored day before
any kernel runs.

What is added:

* **Arrival and other segments.**  The pilot enumerates ground runs backwards from
  a ground -> airborne transition, so a landing that never departs again inside the
  day is invisible to it.  Here every run of on-ground points (split by a hole
  larger than `GAP`, or by an airborne point) becomes a row, and `kind` says which
  end of the flight it is:

      departure  the run is followed by the kept first-airborne point
      arrival    it is not, it begins at runway speed, and it ends stationary
      other      everything else: a tow, a ground vehicle, a through-taxi, or a
                 run whose two ends were both lost to reception

  `starts_fast` and `ends_stationary` are kept as columns, so a reader can redraw
  the boundary without re-deriving anything.
* **The stand position.**  The median latitude and longitude of the stationary
  points the rule anchored on, which is the position the aircraft pushed back from.
* **The airport.**  Points carry the box they fell in, and segments are cut inside
  `(airport, icao)`, so an aircraft that flies EHAM -> EDDF on the same day gives
  one departure row and one arrival row.
"""

import numpy as np
import pandas as pd


LAND_KT = 30.0      # a run that begins above this speed began as a landing rollout
STOP_KT = 3.0       # a run that ends below this speed ended at a stand


def _segment_row(ts, gss, lat, lon, mv, runs) -> dict:
    """The pilot's rule on one ground run: anchor, onset, margin, stand position."""
    sustained = [(a, b) for a, b, v in runs if v and _span(ts, a, b) >= HOLD_S]
    turns = [b for a, b, v in runs if not v and _span(ts, a, b) >= TURNAROUND]
    anchor = turns[-1] - 1 if turns else 0
    after = [a for a, _ in sustained if a >= anchor]
    strict = loose = disp = margin = np.nan
    stand_lat = stand_lon = np.nan
    if after:
        k0 = after[0]
        loose = ts[k0]
        prev = [(a, b) for a, b, v in runs if not v and b <= k0]
        margin = _span(ts, *prev[-1]) if prev else 0.0
        if margin >= STAND_STOP:
            strict = loose
        st_pts = [k for k in range(anchor, k0) if not mv[k]]
        ref = st_pts if st_pts else [anchor]
        stand_lat = float(np.median(lat[ref]))
        stand_lon = float(np.median(lon[ref]))
        dx = (lon - stand_lon) * 111320.0 * np.cos(np.radians(stand_lat))
        dy = (lat - stand_lat) * 110540.0
        far = np.hypot(dx, dy) > 250.0                      # segments.DISP_M
        far[:anchor] = False
        out_idx = np.flatnonzero(far)
        if len(out_idx):
            back = np.flatnonzero(~far[out_idx[0]:])
            k1 = out_idx[0]
            if len(back):
                later = np.flatnonzero(far[out_idx[0] + back[-1]:])
                if len(later):
                    k1 = out_idx[0] + back[-1] + later[0]
            disp = ts[k1]
    return {
        "obs_offblock_strict": strict,
        "obs_offblock_loose": loose,
        "obs_offblock_disp": disp,
        "stand_stop_s": margin,
        "turnaround_anchored": bool(turns),
        "stand_lat": stand_lat,
        "stand_lon": stand_lon,
    }


def all_segments(points: pd.DataFrame) -> pd.DataFrame:
    """One row per aircraft ground segment in the kernel's slice."""
    out = []
    cols = ["t", "on_ground", "first_airborne", "gs_kt", "lat", "lon", "callsign"]
    # `kind="stable"` is not a preference.  92 of a day's 553,178 points share an
    # (airport, icao, t) with another point, and 22 of those pairs disagree about
    # `on_ground`; an unstable sort puts them either way round, and the ground run
    # splits at a different point.  Measured against the gate kernel
    # (analysis/machine_check.json), that moved 14 of 8,301 segments between this
    # laptop and the Kaggle image -- the same code, the same points, two answers.
    # A stable sort keeps the order readsb wrote the trace in, which is the truth
    # about which of two same-second points came first, and gives one answer on
    # every machine.
    ordered = points.sort_values("t", kind="stable")
    for (airport, icao), g in ordered.groupby(["airport", "icao"], sort=False, observed=True):
        g = g.reset_index(drop=True)
        t = g["t"].to_numpy("float64")
        grd = g["on_ground"].to_numpy(bool)
        air = g["first_airborne"].to_numpy(bool)
        gs = g["gs_kt"].to_numpy("float64")
        lat, lon = g["lat"].to_numpy("float64"), g["lon"].to_numpy("float64")
        # the pilot's reviewer repair (2026-09-08): a first-airborne point with
        # readsb's null altitude is not evidence of a take-off.  It is priced at 6 of
        # 5,395 departure segments and 0.07 s of held-out RMSE over the pilot's week
        # (`exp-043-pilot/analysis/divergence.json`), and it is the committed default,
        # so the harvest ships it.  `airborne_alt_ok` keeps the distinction in the row,
        # so either rule can be read off the summaries without re-deriving them.
        alt = g["alt_ft"].notna().to_numpy()
        gidx = np.flatnonzero(grd)
        if not len(gidx):
            continue
        # a ground run breaks at an airborne point (index jump) or at a hole > GAP
        brk = np.flatnonzero((np.diff(gidx) > 1) | (np.diff(t[gidx]) > GAP)) + 1
        for a, b in zip([0, *brk.tolist()], [*brk.tolist(), len(gidx)]):
            i, j = gidx[a], gidx[b - 1]
            sl = slice(i, j + 1)
            ts, gss = t[sl], gs[sl]
            la, lo = lat[sl], lon[sl]
            mv = _moving(gss, la, lo, ts)
            runs = _runs(mv)
            row = _segment_row(ts, gss, la, lo, mv, runs)
            raw_air = bool(j + 1 < len(t) and air[j + 1])
            alt_ok = bool(raw_air and alt[j + 1])
            ends_air = alt_ok if REQUIRE_ALT else raw_air
            starts_fast = bool(mv[0] and (gss[0] > LAND_KT if np.isfinite(gss[0]) else False))
            ends_stationary = bool(not mv[-1])
            kind = ("departure" if ends_air
                    else "arrival" if (starts_fast and ends_stationary)
                    else "other")
            # where the aircraft was when the network last saw it on the ground, and
            # where it was when it first saw it airborne.  `obs_wheels_off` is only a
            # take-off clock if the last ground point is actually on the runway; when
            # reception fades in the departure queue it is an early clock instead, and
            # these six columns are what tells the two apart per airport.  Added
            # 2026-09-08 after the modelling worker found a 253 s gap at EGLL.
            wheels_gs = float(gss[-1]) if np.isfinite(gss[-1]) else np.nan
            air_lat = air_lon = air_alt = np.nan
            if ends_air:
                air_lat, air_lon = float(lat[j + 1]), float(lon[j + 1])
                a_ft = g["alt_ft"].iloc[j + 1]
                air_alt = float(a_ft) if pd.notna(a_ft) else np.nan
            cs = g["callsign"].iloc[sl].dropna()
            row.update({
                "airport": airport,
                "icao": icao,
                "kind": kind,
                # the two names the assignment asks for, bound to the variants the pilot
                # froze: `obs_offblock` is the 250 m displacement onset (143 s RMSE at
                # EHAM against the block time, versus 184 s for the speed onset,
                # exp-043-pilot/analysis/match.json), and `first_motion` is that speed
                # onset.  Both variants stay in the row, so a later stage can re-choose
                # without re-deriving anything.
                "obs_offblock": row["obs_offblock_disp"],
                "first_motion": row["obs_offblock_loose"],
                "callsign_modal": cs.mode().iat[0] if len(cs) else None,
                "callsign_n": int(cs.nunique()),
                "callsigns_all": "|".join(sorted(cs.unique())) if len(cs) else None,
                "seg_start": float(ts[0]),
                "seg_end": float(ts[-1]),
                "obs_wheels_off": float(ts[-1]) if ends_air else np.nan,
                "wheels_lat": float(la[-1]), "wheels_lon": float(lo[-1]),
                "wheels_gs_kt": wheels_gs,
                "air_lat": air_lat, "air_lon": air_lon, "air_alt_ft": air_alt,
                "t_airborne": float(t[j + 1]) if ends_air else np.nan,
                "seg_points": int(j - i + 1),
                "n_ground_points": int(grd[sl].sum()),
                "seg_span_s": float(ts[-1] - ts[0]),
                "seg_gap_max_s": float(np.diff(ts).max()) if len(ts) > 1 else 0.0,
                "seg_dt_median_s": float(np.median(np.diff(ts))) if len(ts) > 1 else np.nan,
                "starts_fast": starts_fast,
                "ends_stationary": ends_stationary,
                "ends_airborne": ends_air,
                "airborne_alt_ok": alt_ok,
            })
            out.append(row)
    order = ["airport", "icao", "callsign_modal", "callsign_n", "callsigns_all", "kind",
             "seg_start", "seg_end", "obs_offblock", "first_motion",
             "obs_offblock_disp", "obs_offblock_strict",
             "obs_offblock_loose", "stand_stop_s", "turnaround_anchored",
             "obs_wheels_off", "t_airborne", "stand_lat", "stand_lon",
             "wheels_lat", "wheels_lon", "wheels_gs_kt",
             "air_lat", "air_lon", "air_alt_ft",
             "seg_points", "n_ground_points", "seg_span_s", "seg_gap_max_s",
             "seg_dt_median_s", "starts_fast", "ends_stationary", "ends_airborne", "airborne_alt_ok"]
    df = pd.DataFrame(out)
    return df[order] if len(df) else pd.DataFrame(columns=order)

RULE_SHA256 = "d4c59868864523ce6513151c1627ec2ba997fd25d6394cfe2ff03a14d89f01a0"
RULE_VERSION = "repair-2026-09-08+stable-order"


if __name__ == "__main__":
    main()
