"""Write a guarded copy of a finished candidate, and report what it moved.

The guard is a transform of a completed prediction, so a guarded and an
unguarded candidate come from one fit. Bounds are the symmetric 0.01 % tails of
`r = BLOCK - AOBT` over the twelve training months.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import guard  # noqa: E402
import prc.data as D  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
d = pd.concat([D.load_month(m, columns=["PHASE_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]) for m in D.MONTHS])
d = d[d.PHASE_mvt == "DEP"]
r = ((d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()).dropna().to_numpy()
lo, hi = np.quantile(r, [1e-4, 1 - 1e-4])

rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
rk = rk[rk.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
rk["month"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
a = ((rk.MVT_TIME_UTC_mvt - rk.AOBT_3_flt).dt.total_seconds()).to_numpy()

c = pd.read_parquet(ROOT / "submissions" / "candidates" / f"exp-001-retro-{src}.parquet").set_index("MVT_ID_mvt")
y = c[D.TARGET].reindex(rk.index).to_numpy()
touched = np.isfinite(a) & (((a - y) < lo) | ((a - y) > hi))
g = guard.apply(y, a, lo, hi, floor=60.0)

print(f"guard bounds from the twelve training months at [1e-4, 1-1e-4]: [{lo:.0f}, {hi:.0f}]")
print(f"touched {touched.sum()} of {len(y)} ranking rows ({100 * touched.mean():.4f} %)")
for m in sorted(rk.month.unique()):
    mm = (rk.month == m).to_numpy()
    ch = np.abs(g - y)[touched & mm]
    print(f"  {m}: {int((touched & mm).sum()):5d} of {int(mm.sum()):6d} "
          f"({100 * (touched & mm).sum() / mm.sum():.4f} %), mean change {ch.mean() if ch.size else 0:.0f} s")
path = ROOT / "submissions" / "candidates" / f"exp-001-retro-{dst}.parquet"
digest = D.write_submission(pd.Series(g, index=rk.index), path)
print(f"wrote {path.relative_to(ROOT)}  sha256 {digest}")
