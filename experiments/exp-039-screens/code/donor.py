"""The donor search of Q06, used by the census, the accuracy test and the stress.

QUEUE.md's rules, and nothing beyond them:

* `FLIGHT_mvt` is normalised by **uppercase and whitespace only**.
* A donor must agree on the flight identifier and on the DIRECTED leg
  (`ADEP_mvt`, `ADES_mvt`).
* Two donor classes are kept apart: the **destination arrival** of the same leg
  (landing minus take-off in `[0, 12 h]`) and a **duplicate departure**
  (take-off or schedule within +-60 s).  Both windows are conservative search
  restrictions, not label bounds -- neither reads B or y.
* Acceptance needs ONE consistent `FLIGHT_ID_mvt` across the surviving
  NM-carrying donors AND reciprocal uniqueness: no other query in the same
  population may point at the same donor.  Ambiguity is never resolved by
  nearest time.
* The arrival donor's `BLOCK_TIME_UTC_mvt` is the destination in-block clock and
  is never read as the departure's B.  Only the `_flt` (NM) fields and
  `AIRCRAFT_TYPE_mvt` are taken from a donor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ARR_MAX = 12 * 3600.0     # declared arrival landing - take-off window
DUP_TOL = 60.0            # declared duplicate take-off/schedule tolerance
NEAR = 60.0               # "boundary case": within this of a declared bound
NM_FIELDS = ["LOBT_flt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt"]


#: pandas 2.x keeps a parquet `timestamp[us]` column at MICROSECOND resolution,
#: so `astype("int64")` is microseconds, not nanoseconds.  Reading the unit off
#: the dtype is the only conversion that is right for both.
_UNIT = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}


def epoch(s: pd.Series) -> np.ndarray:
    """UTC epoch seconds, whatever datetime resolution the column carries."""
    u = getattr(s.dtype, "unit", "ns")
    out = s.astype("int64").to_numpy("float64") / _UNIT[u]
    out[s.isna().to_numpy()] = np.nan
    return out


def norm_key(s: pd.Series) -> pd.Series:
    """Uppercase and whitespace only -- no other normalisation is permitted."""
    return s.astype("string").str.upper().str.strip()


def table(df: pd.DataFrame) -> pd.DataFrame:
    """The movement table the search reads: keys, legs, clocks, NM presence."""
    t = pd.DataFrame({
        "mid": df["MVT_ID_mvt"].to_numpy("float64"),
        "phase": df["PHASE_mvt"].to_numpy(),
        "key": norm_key(df["FLIGHT_mvt"]).to_numpy(),
        "adep": df["ADEP_mvt"].astype("string").to_numpy(),
        "ades": df["ADES_mvt"].astype("string").to_numpy(),
        "t": epoch(df["MVT_TIME_UTC_mvt"]),
        "sched": epoch(df["SCHED_TIME_UTC_mvt"]),
        "fid": df["FLIGHT_ID_mvt"].to_numpy("float64"),
        "actype": df["AIRCRAFT_TYPE_mvt"].astype("string").to_numpy()})
    for c in NM_FIELDS:
        t[c] = epoch(df[c])
    t["has_nm"] = np.isfinite(t["fid"].to_numpy())
    return t


def candidates(q: pd.DataFrame, tbl: pd.DataFrame) -> pd.DataFrame:
    """Every key+leg agreement between a query and another movement row.

    Returns one row per (query, donor) with the class and the clock gaps, before
    any time filter -- the census counts both sides of the filter.
    """
    left = q[["mid", "key", "adep", "ades", "t", "sched"]].rename(
        columns={"mid": "q_mid", "t": "q_t", "sched": "q_sched"})
    e = left.merge(tbl, on=["key", "adep", "ades"], how="inner",
                   suffixes=("", "_d"))
    e = e[e["mid"] != e["q_mid"]].copy()
    e["dt"] = e["t"] - e["q_t"]
    e["dsched"] = e["sched"] - e["q_sched"]
    e["cls"] = np.where(e["phase"] == "ARR", "arrival", "duplicate")
    ok_arr = (e["cls"] == "arrival") & (e["dt"] >= 0) & (e["dt"] <= ARR_MAX)
    ok_dup = (e["cls"] == "duplicate") & (
        (e["dt"].abs() <= DUP_TOL) | (e["dsched"].abs() <= DUP_TOL))
    e["in_window"] = (ok_arr | ok_dup).to_numpy()
    e["boundary"] = (
        ((e["cls"] == "arrival") & ((e["dt"].abs() <= NEAR)
                                    | ((e["dt"] - ARR_MAX).abs() <= NEAR)))
        | ((e["cls"] == "duplicate")
           & ((e["dt"].abs() - DUP_TOL).abs() <= 5)
           | ((e["dsched"].abs() - DUP_TOL).abs() <= 5))).to_numpy()
    return e


def accept(e: pd.DataFrame) -> pd.DataFrame:
    """One accepted donor id per query, or none.  No nearest-time tie-break.

    A query is accepted when the NM-carrying donors inside the window agree on a
    single `FLIGHT_ID_mvt` and no other query in the population reaches that same
    `FLIGHT_ID_mvt` (reciprocal uniqueness, by id, inside the query population).
    """
    w = e[e["in_window"] & e["has_nm"]].copy()
    if w.empty:
        return pd.DataFrame(columns=["q_mid", "fid", "n_donor", "cls"])
    nfid = w.groupby("q_mid")["fid"].nunique()
    # Reciprocal uniqueness is by accepted FLIGHT_ID, not by donor row: two
    # queries may reach the same NM flight through two different rows, and that
    # is exactly the ambiguity the rule must refuse (Sol round 1, blocking
    # finding 11).  Grouping by `fid` subsumes grouping by `mid`, because two
    # queries that share a donor row share its id.
    deg_fid = w.groupby("fid")["q_mid"].nunique()
    w["donor_shared"] = deg_fid.reindex(w["fid"]).to_numpy() > 1
    shared = w.groupby("q_mid")["donor_shared"].any()
    good = nfid[(nfid == 1) & ~shared.reindex(nfid.index).fillna(True)].index
    a = w[w["q_mid"].isin(good)]
    out = a.groupby("q_mid").agg(fid=("fid", "first"), n_donor=("mid", "size"),
                                 cls=("cls", lambda s: "+".join(sorted(set(s)))))
    keep = ["mid", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "actype", "t"]
    first = a.sort_values(["q_mid", "t"]).groupby("q_mid")[keep].first()
    return out.join(first).reset_index()
