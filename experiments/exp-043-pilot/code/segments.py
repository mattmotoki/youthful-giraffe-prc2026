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
from __future__ import annotations

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
