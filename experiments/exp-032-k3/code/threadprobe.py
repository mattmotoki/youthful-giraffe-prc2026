"""Is this member's output independent of the LightGBM thread count?

The question matters for one decision only: whether a ranking-side identity
check run at six threads is a valid check of a committed file that was fitted at
`prc.tail_model.DEFAULT_N_JOBS` = 16 (`exp-012-blend2/tail_ranking_v4.py` passes
no `n_jobs`). LightGBM's `deterministic` flag guarantees reproducibility for a
given thread count; it does not promise the same answer at a different one.

One fold, flag off, three threads, against the committed six-thread vector.

    PRC_THREADS=3 python code/threadprobe.py july_loo
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
THREADS = int(os.environ.get("PRC_THREADS", "3"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402
from prc.tail_model import TailModel  # noqa: E402

fold = sys.argv[1] if len(sys.argv) > 1 else "july_loo"
train, valid, y_valid = D.fold_frames(fold)
m = TailModel().fit(train, n_jobs=THREADS, noac_head="v4")
p = m.predict(valid)
y = y_valid.reindex(p.index).to_numpy(float)
got = pd.DataFrame({"MVT_ID_mvt": p["MVT_ID_mvt"].to_numpy(),
                    "y_pred": p["y_pred"].to_numpy()})
ref = pd.read_parquet(EXP / "oof" / f"k3-tail-base-{fold}.parquet")
j = got.merge(ref[["MVT_ID_mvt", "y_pred"]], on="MVT_ID_mvt", suffixes=("", "_6t"))
d = float(np.abs(j["y_pred"].to_numpy() - j["y_pred_6t"].to_numpy()).max())
print(f"{fold}: {THREADS} threads rmse {float(np.sqrt(np.mean((y - got['y_pred'].to_numpy()) ** 2))):.4f}"
      f"   6-thread committed 301.6430-family; max abs row difference {d:.6g}"
      f"   -> thread {'INDEPENDENT' if d == 0.0 else 'DEPENDENT'}")
