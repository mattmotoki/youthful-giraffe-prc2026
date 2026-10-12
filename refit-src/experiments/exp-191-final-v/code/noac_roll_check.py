"""NOAC and ROLL of my c5 against finallirf's exp-187 functions, on every LIRF departure without an aircraft type.

finallirf's `final_lirf.py` (exp-187) applies the c5 NOAC and ROLL rules on the 2026 rows: mixture rows
900 + lam (v - 900) clipped to the bounds, then the roll value p (86,400 + offset) + (1 - p) ms on the cross rows
with 50,000 <= ms < 86,400 s, with lam, the logistic and the offset of its record.json. This script applies those
rules (the formulas of `final_lirf.run`, its row set, ms, cross flags and bounds from its lirf-notype.parquet) to my
chain's v64 values and compares the result with my c5 values on the same rows. A second pass uses my chain's
bounds (c5_lower, c5_upper of work/w2/base_ranking.parquet) instead of finallirf's, to split a bound difference
from a rule difference.

    python noac_roll_check.py <run> <finallirf out dir> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
NOAC_R, ROLL_LO, DAY = 900.0, 50000.0, 86400.0
run, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rec = json.loads((ref / "record.json").read_text())
t = pd.read_parquet(ref / "lirf-notype.parquet")
b = pd.read_parquet(run / "work/w2/base_ranking.parquet",
                    columns=[ID, "airport", "cls", "month", "v64", "c5", "c5_lower", "c5_upper"])
f = pd.read_parquet(run / "work/feat/ranking.parquet", columns=[ID, "mvt_sched"])
assert np.array_equal(b[ID].to_numpy(), f[ID].to_numpy())
b["ms"] = f.mvt_sched.to_numpy(float)
mine = b[b.airport.eq("LIRF") & b.cls.eq("noac")].set_index(ID)
res = {"finallirf_ranking_sha256": rec["ranking_sha256"], "rows_finallirf": int(len(t)), "rows_mine_noac_lirf": int(len(mine)),
       "row_sets_equal": set(t[ID].astype("int64")) == set(mine.index.astype("int64"))}
x = mine.reindex(t[ID].to_numpy())
assert x.v64.notna().all(), "a finallirf row is not a LIRF noac row of my chain"
ms = t.ms.to_numpy(float)
res["ms_equal"] = bool(np.array_equal(ms, x.ms.to_numpy(float)))


def rules(lo, hi):
    """finallirf's NOAC and ROLL (final_lirf.run) on my v64, with the bounds lo, hi."""
    v = x.v64.to_numpy(float).copy()
    den = np.where(np.abs(ms - NOAC_R) < 1, 1.0, ms - NOAC_R)
    p = (v - NOAC_R) / den
    mix = (p > 0.03) & (p < 0.97) & (ms > NOAC_R + 600)
    v[mix] = NOAC_R + rec["noac"]["lam"] * (v[mix] - NOAC_R)
    v[mix] = np.maximum(60, np.clip(v[mix], lo[mix], hi[mix]))
    roll = t.cross.to_numpy(bool) & (ms >= ROLL_LO) & (ms < DAY)
    z = rec["roll"]["coef"] * ms[roll] / 3600.0 + rec["roll"]["intercept"]
    pr = 1.0 / (1.0 + np.exp(-z))
    val = pr * (DAY + rec["roll"]["offset"]) + (1 - pr) * ms[roll]
    v[roll] = np.maximum(60, np.clip(val, lo[roll], hi[roll]))
    return v, mix, roll


lo, hi = t.lower.to_numpy(float), t.upper.to_numpy(float)
mlo, mhi = x.c5_lower.to_numpy(float), x.c5_upper.to_numpy(float)
v, mix, roll = rules(lo, hi)
v_mine_bounds, _, _ = rules(mlo, mhi)
lam = rec["noac"]["lam"]
c5 = x.c5.to_numpy(float)
d = np.abs(v - c5)
month = t.month.to_numpy()
res.update({"noac_lam": lam, "roll": {k: rec["roll"][k] for k in ("offset", "coef", "intercept")},
            "mixture_rows": {int(m): int((mix & (month == m)).sum()) for m in sorted(set(month))},
            "roll_rows": {int(m): int((roll & (month == m)).sum()) for m in sorted(set(month))},
            "c5_moved_rows": {int(m): int(((c5 != x.v64.to_numpy()) & (month == m)).sum()) for m in sorted(set(month))},
            "equal_rows": int((d == 0).sum()), "max_abs_diff": float(d.max()),
            "rows_over_1e-6": int((d > 1e-6).sum()),
            "with_my_bounds": {"equal_rows": int((v_mine_bounds == c5).sum()),
                               "max_abs_diff": float(np.abs(v_mine_bounds - c5).max())},
            "bounds_differ_rows": {int(m): int((((lo != mlo) & ~(np.isnan(lo) & np.isnan(mlo)))
                                                | ((hi != mhi) & ~(np.isnan(hi) & np.isnan(mhi))))[month == m].sum())
                                   for m in sorted(set(month))},
            "v64_vs_finallirf_v_bounds": {int(m): float(np.abs(x.v64.to_numpy() - t.v_bounds.to_numpy())[month == m].max())
                                          for m in sorted(set(month))}})
k = d > 1e-6
bad = t.loc[k, [ID, "month", "ms", "lower", "upper", "lower_by", "upper_by"]].assign(
    my_lower=mlo[k], my_upper=mhi[k], v64=x.v64.to_numpy()[k], mine=c5[k], theirs=v[k])
res["differing_rows"] = bad.head(20).to_dict("records")
out.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
