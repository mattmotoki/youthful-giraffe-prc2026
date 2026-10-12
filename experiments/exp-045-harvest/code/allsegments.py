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
from __future__ import annotations

import numpy as np
import pandas as pd

from segments import GAP, HOLD_S, REQUIRE_ALT, STAND_STOP, TURNAROUND, _moving, _runs, _span

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
