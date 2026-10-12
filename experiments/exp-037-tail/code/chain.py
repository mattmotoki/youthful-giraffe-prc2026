"""The stand chain: for every departure, the previous arrival at its stand.

Key `(airport, stand)`: a departure's airport is `ADEP_mvt`, an arrival's is
`ADES_mvt`, and the stand is the raw `STAND_mvt` string.  An arrival occupies
its stand from its in-block time `BLOCK_TIME_UTC_mvt`, which is never blanked.
A departure stops occupying its stand at its off-block, so the only legal
off-block for a NEIGHBOURING departure is `AOBT_3_flt` (the departure's own
`BLOCK_TIME_UTC_mvt` is blanked in the ranking file and in a masked fold).

Label-free: this module never reads a departure's `BLOCK_TIME_UTC_mvt` or
`TAXITIME_SEC_mvt`, and it raises if it is handed a frame that still carries
them.  It takes a RANKING-MASKED frame.

Per departure, indexed like the departure rows of `masked`:

    apt, stand    the chain key
    t_off         MVT_TIME_UTC_mvt, the take-off
    b_in          the previous arrival's in-block time at the same stand
    g_any         t_off - b_in, the gap in seconds
    n_dep_between other departures whose NM off-block lies in (b_in, t_off]
    g_strict      g_any where n_dep_between == 0
    g_prev2       t_off - the in-block of the arrival BEFORE b_in
    self_aobt_pre the row's OWN AOBT_3_flt is at or before b_in: NM already says
                  the aircraft left the stand before that arrival
    n_arr_stand   arrivals seen at that stand in the frame
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["apt", "stand"]


def _rank(q: pd.DataFrame, on: str, od: pd.DataFrame) -> np.ndarray:
    """Per row of `q`, the number of `od` rows in its group with `t <= q[on]`."""
    left = q.loc[q[on].notna(), KEY + [on, "_row"]].sort_values(on, kind="mergesort")
    j = pd.merge_asof(left, od, left_on=on, right_on="t", by=KEY,
                      direction="backward", allow_exact_matches=True)
    out = np.zeros(len(q), dtype="float64")
    out[j["_row"].to_numpy("int64")] = np.nan_to_num(j["c"].to_numpy("float64"), nan=0.0)
    return out


def stand_chain(masked: pd.DataFrame) -> pd.DataFrame:
    dep = masked[masked["PHASE_mvt"] == "DEP"]
    arr = masked[masked["PHASE_mvt"] == "ARR"]
    if dep["BLOCK_TIME_UTC_mvt"].notna().any() or dep["TAXITIME_SEC_mvt"].notna().any():
        raise ValueError("stand_chain wants a ranking-masked frame: departure labels are present")

    a = pd.DataFrame({
        "apt": arr["ADES_mvt"].to_numpy(),
        "stand": arr["STAND_mvt"].to_numpy(),
        "b_in": arr["BLOCK_TIME_UTC_mvt"].to_numpy("datetime64[ns]"),
    }).dropna().sort_values("b_in", kind="mergesort")

    od = pd.DataFrame({
        "apt": dep["ADEP_mvt"].to_numpy(),
        "stand": dep["STAND_mvt"].to_numpy(),
        "t": dep["AOBT_3_flt"].to_numpy("datetime64[ns]"),
    }).dropna().sort_values("t", kind="mergesort")
    od["c"] = od.groupby(KEY, sort=False).cumcount().to_numpy() + 1.0

    q = pd.DataFrame({
        "apt": dep["ADEP_mvt"].to_numpy(),
        "stand": dep["STAND_mvt"].to_numpy(),
        "t_off": dep["MVT_TIME_UTC_mvt"].to_numpy("datetime64[ns]"),
        "aobt": dep["AOBT_3_flt"].to_numpy("datetime64[ns]"),
    })
    q["_row"] = np.arange(len(q), dtype="int64")

    sel = q["apt"].notna() & q["stand"].notna() & q["t_off"].notna()
    left = q.loc[sel, KEY + ["t_off", "_row"]].sort_values("t_off", kind="mergesort")
    j = pd.merge_asof(left, a, left_on="t_off", right_on="b_in", by=KEY,
                      direction="backward", allow_exact_matches=True)
    b_in = np.full(len(q), np.datetime64("NaT", "ns"))
    b_in[j["_row"].to_numpy("int64")] = j["b_in"].to_numpy("datetime64[ns]")
    q["b_in"] = b_in
    q["g_any"] = (q["t_off"] - q["b_in"]).dt.total_seconds()

    # the arrival before that one: a strictly weaker bound that survives one
    # contaminating arrival (the stand re-occupied while our aircraft taxied)
    left2 = q.loc[q["b_in"].notna(), KEY + ["b_in", "_row"]].sort_values("b_in", kind="mergesort")
    j2 = pd.merge_asof(left2, a.rename(columns={"b_in": "b_in2"}), left_on="b_in", right_on="b_in2",
                       by=KEY, direction="backward", allow_exact_matches=False)
    b_in2 = np.full(len(q), np.datetime64("NaT", "ns"))
    b_in2[j2["_row"].to_numpy("int64")] = j2["b_in2"].to_numpy("datetime64[ns]")
    q["b_in2"] = b_in2
    q["g_prev2"] = (q["t_off"] - q["b_in2"]).dt.total_seconds()

    c_t = _rank(q, "t_off", od)
    c_b = _rank(q.assign(_b=q["b_in"]), "_b", od)
    self_in = ((q["aobt"] > q["b_in"]) & (q["aobt"] <= q["t_off"])).to_numpy()
    n_between = c_t - c_b - self_in.astype("float64")
    n_between[~np.isfinite(q["g_any"].to_numpy("float64"))] = np.nan
    q["n_dep_between"] = n_between
    q["g_strict"] = q["g_any"].where(q["n_dep_between"] == 0)
    q["self_aobt_pre"] = (q["aobt"] <= q["b_in"]).to_numpy()

    n = a.groupby(KEY, sort=False).size().rename("n")
    q["n_arr_stand"] = pd.to_numeric(
        pd.Series(q.set_index(KEY).index.map(n)), errors="coerce").fillna(0).astype("int64").to_numpy()

    q = q.drop(columns=["aobt", "_row"])
    q.index = dep.index
    return q
