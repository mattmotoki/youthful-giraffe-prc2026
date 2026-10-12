"""A brute-force check of `chain.py` on a synthetic frame and on real data.

`stand_chain` is vectorised (`merge_asof` plus a within-group rank difference),
and two things in it were wrong in a first version: the self-match, and the
index `merge_asof` returns.  This module recomputes `b_in`, `b_in2` and
`n_dep_between` by an O(n^2) loop and asserts they agree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from chain import stand_chain  # noqa: E402

T0 = pd.Timestamp("2025-07-01", tz=None)


def brute(masked: pd.DataFrame) -> pd.DataFrame:
    dep = masked[masked.PHASE_mvt == "DEP"]
    arr = masked[masked.PHASE_mvt == "ARR"]
    a = pd.DataFrame({"k": list(zip(arr.ADES_mvt, arr.STAND_mvt)),
                      "t": arr.BLOCK_TIME_UTC_mvt.to_numpy("datetime64[ns]")}).dropna()
    od = pd.DataFrame({"k": list(zip(dep.ADEP_mvt, dep.STAND_mvt)),
                       "t": dep.AOBT_3_flt.to_numpy("datetime64[ns]"),
                       "id": dep.MVT_ID_mvt.to_numpy("float64")}).dropna()
    rows = []
    for k, t, mid in zip(zip(dep.ADEP_mvt, dep.STAND_mvt),
                         dep.MVT_TIME_UTC_mvt.to_numpy("datetime64[ns]"),
                         dep.MVT_ID_mvt.to_numpy("float64")):
        ta = np.sort(a.t[[x == k for x in a.k]].to_numpy("datetime64[ns]"))
        before = ta[ta <= t]
        b1 = before[-1] if len(before) else np.datetime64("NaT", "ns")
        b2 = np.datetime64("NaT", "ns")
        if len(before) >= 2:
            b2 = before[-2]
        n_between = np.nan
        if not np.isnat(b1):
            sel = np.array([x == k for x in od.k]) & (od.id.to_numpy() != mid)
            td = od.t[sel].to_numpy("datetime64[ns]")
            n_between = float(((td > b1) & (td <= t)).sum())
        rows.append({"b_in": b1, "b_in2": b2, "n_dep_between": n_between})
    return pd.DataFrame(rows, index=dep.index)


def synthetic() -> pd.DataFrame:
    """Six movements on one stand, plus a decoy stand, all times explicit."""
    def row(i, phase, adep, ades, stand, mvt, block, aobt):
        return {"MVT_ID_mvt": float(i), "PHASE_mvt": phase, "ADEP_mvt": adep,
                "ADES_mvt": ades, "STAND_mvt": stand,
                "MVT_TIME_UTC_mvt": T0 + pd.Timedelta(minutes=mvt),
                "BLOCK_TIME_UTC_mvt": (T0 + pd.Timedelta(minutes=block)) if block is not None else pd.NaT,
                "AOBT_3_flt": (T0 + pd.Timedelta(minutes=aobt)) if aobt is not None else pd.NaT,
                "TAXITIME_SEC_mvt": np.nan}
    r = [
        row(1, "ARR", "XXXX", "AAAA", "S1", 0, 10, None),      # in-block 00:10
        row(2, "DEP", "AAAA", "XXXX", "S1", 70, None, 60),     # off 01:00, take-off 01:10
        row(3, "ARR", "XXXX", "AAAA", "S1", 80, 90, None),     # in-block 01:30
        row(4, "DEP", "AAAA", "XXXX", "S1", 200, None, 190),   # off 03:10, take-off 03:20
        row(5, "ARR", "XXXX", "AAAA", "S2", 5, 15, None),      # decoy stand
        row(6, "DEP", "AAAA", "XXXX", "S2", 100, None, 95),
    ]
    return pd.DataFrame(r)


def main() -> None:
    s = synthetic()
    ch = stand_chain(s)
    dep = s[s.PHASE_mvt == "DEP"]
    got = ch.loc[dep.index]
    # row 2: previous arrival is row 1 (00:10); no other departure left S1 in (00:10, 01:10]
    # row 4: previous arrival is row 3 (01:30); departure 2 left S1 at 01:00, BEFORE 01:30,
    #        so n_dep_between is 0 and the second-previous arrival is row 1
    exp_b = [T0 + pd.Timedelta(minutes=10), T0 + pd.Timedelta(minutes=90),
             T0 + pd.Timedelta(minutes=15)]
    assert list(got.b_in) == exp_b, (list(got.b_in), exp_b)
    assert list(got.n_dep_between) == [0.0, 0.0, 0.0], list(got.n_dep_between)
    assert pd.isna(got.b_in2.iloc[0]) and got.b_in2.iloc[1] == T0 + pd.Timedelta(minutes=10)
    assert list(got.g_any) == [3600.0, 6600.0, 5100.0], list(got.g_any)
    print("synthetic: b_in, b_in2, n_dep_between, g_any all as hand-computed")

    # a real slice: one airport, one day, brute-forced
    raw = D.load_month("2025-07")
    day = raw[(raw.MVT_TIME_UTC_mvt >= "2025-07-03") & (raw.MVT_TIME_UTC_mvt < "2025-07-04")
              & ((raw.ADEP_mvt == "LSZH") | (raw.ADES_mvt == "LSZH"))].copy()
    masked, _ = D.mask_like_ranking(day)
    fast = stand_chain(masked)
    slow = brute(masked)
    for c in ("b_in", "b_in2"):
        a1, a2 = fast[c].to_numpy("datetime64[ns]"), slow[c].to_numpy("datetime64[ns]")
        bad = int(((a1 != a2) & ~(pd.isna(a1) & pd.isna(a2))).sum())
        print(f"  {c}: {bad} mismatches of {len(a1)}")
        assert bad == 0
    a1, a2 = fast.n_dep_between.to_numpy(float), slow.n_dep_between.to_numpy(float)
    bad = int((~np.isclose(a1, a2, equal_nan=True)).sum())
    print(f"  n_dep_between: {bad} mismatches of {len(a1)}")
    assert bad == 0
    print("real slice (LSZH, 2025-07-03, %d departures): the vectorised chain "
          "equals the brute-force one" % len(fast))


if __name__ == "__main__":
    main()
