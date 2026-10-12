"""The airline-rotation signals: same-airline arrivals at the departure's own stand.

Round 1 of the review refuted §3b's first mechanism.  The arrivals that crowd a
placeholder row's stand are not other aircraft taking the stand: they carry the
DEPARTING flight's own three-letter prefix on 92 % of 1-2 h placeholder rows
against a 21 % base rate at those stands, and the probability that some OTHER
airline arrives during the taxi is the same for placeholder rows as for the
rest.  The stand key is exclusive; what the chain finds on a placeholder row is
the aircraft's own late inbound.

That makes the observable, not the confounder.  The anchor has to change, and
this is the part the review got wrong: the review's indicator is anchored on the
departure's `BLOCK_TIME_UTC_mvt`, which is exactly the column the ranking file
blanks (`docs/input-rule.md`), so it cannot be computed on the 388.  The legal
anchor is `SCHED_TIME_UTC_mvt`, which is never blanked -- and on a placeholder
row the block time IS the schedule, which is what makes the placeholder a
placeholder.  Every column below is computable on a ranking row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: half-width of the window around the scheduled off-block, seconds
W = 1800.0
#: how far the nearest same-airline arrival may be before it stops counting
NEAR = 21600.0

ROT_COLS = ("rot_same_near", "rot_same_before", "rot_same_after", "rot_same_gap",
            "rot_g_same", "rot_pref_base", "rot_n_same_stand")


def _epoch(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).astype("int64").to_numpy("float64") / 1e9


def rotation_signals(masked: pd.DataFrame, keep: pd.Index) -> pd.DataFrame:
    """Per departure in `keep`, the same-airline arrivals at its own stand.

    Legal inputs only: arrivals keep every column, and on the departure side
    this reads `ADEP_mvt`, `STAND_mvt`, `FLIGHT_mvt`, `SCHED_TIME_UTC_mvt` and
    `MVT_TIME_UTC_mvt`.  It never touches a departure's block time or taxi time.
    """
    if "BLOCK_TIME_UTC_mvt" in masked.columns:
        dep_rows = masked["PHASE_mvt"].to_numpy() == "DEP"
        if masked.loc[dep_rows, "BLOCK_TIME_UTC_mvt"].notna().any():
            raise ValueError("rotation_signals wants a ranking-masked frame")

    arr = masked[masked["PHASE_mvt"].to_numpy() == "ARR"]
    arr = arr.dropna(subset=["ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"])
    a_key = list(zip(arr["ADES_mvt"].astype(str), arr["STAND_mvt"].astype(str)))
    a_t = _epoch(arr["BLOCK_TIME_UTC_mvt"])
    a_p = arr["FLIGHT_mvt"].astype(str).str[:3].to_numpy()
    order = np.argsort(a_t, kind="mergesort")

    groups: dict[tuple, list[int]] = {}
    for pos in order:
        groups.setdefault(a_key[pos], []).append(int(pos))

    dep = masked.loc[keep]
    d_key = list(zip(dep["ADEP_mvt"].astype(str), dep["STAND_mvt"].astype(str)))
    d_p = dep["FLIGHT_mvt"].astype(str).str[:3].to_numpy()
    d_sched = _epoch(dep["SCHED_TIME_UTC_mvt"])
    d_off = _epoch(dep["MVT_TIME_UTC_mvt"])

    out = {c: np.full(len(dep), np.nan) for c in ROT_COLS}
    for i, (k, p, s, t) in enumerate(zip(d_key, d_p, d_sched, d_off)):
        g = groups.get(k)
        if g is None or not np.isfinite(s):
            continue
        idx = np.asarray(g, dtype="int64")
        ts, ps = a_t[idx], a_p[idx]
        same = ps == p
        out["rot_pref_base"][i] = float(same.mean())
        out["rot_n_same_stand"][i] = float(same.sum())
        st = ts[same]
        if st.size == 0:
            out["rot_same_near"][i] = 0.0
            out["rot_same_before"][i] = 0.0
            out["rot_same_after"][i] = 0.0
            continue
        lo, mid, hi = np.searchsorted(st, [s - W, s, s + W], side="right")
        out["rot_same_before"][i] = float(mid - lo)
        out["rot_same_after"][i] = float(hi - mid)
        out["rot_same_near"][i] = float(hi - lo > 0)
        j = int(np.argmin(np.abs(st - s)))
        if abs(st[j] - s) <= NEAR:
            out["rot_same_gap"][i] = float(st[j] - s)
        last = np.searchsorted(st, t, side="right")
        if last > 0:
            out["rot_g_same"][i] = float(t - st[last - 1])
    return pd.DataFrame(out, index=dep.index)
