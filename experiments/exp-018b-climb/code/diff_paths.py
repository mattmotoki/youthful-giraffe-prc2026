"""Is model2 at one seed the same estimator as exp-018's model.py?

The retention experiment the blend lane wants (s1 vs s3, differing only in
seed count) is only clean if both arms come from one code path. This prints
the diff between the freshly produced one-seed OOF and the committed exp-018
OOF for the same fold.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from prc import data

fold = sys.argv[1] if len(sys.argv) > 1 else "july"
new = pd.read_parquet(f"experiments/exp-018b-climb/oof/s1v-s1-{fold}.parquet")
old = pd.read_parquet(f"experiments/exp-018-fresh/oof/{fold}.parquet")
m = new.merge(old, on="MVT_ID_mvt", suffixes=("_new", "_old"))
assert len(m) == len(old) == len(new), (len(m), len(old), len(new))
assert np.allclose(m.y_true_new, m.y_true_old)
d = (m.y_pred_new - m.y_pred_old).to_numpy()
print(f"fold {fold}: n={len(m)}")
print(f"  max |diff|   {np.abs(d).max():.6g}")
print(f"  mean |diff|  {np.abs(d).mean():.6g}")
print(f"  rows differing by more than 1e-6: {(np.abs(d) > 1e-6).sum()}")
print(f"  RMSE new {data.rmse(m.y_true_new, m.y_pred_new):.4f}   "
      f"old {data.rmse(m.y_true_old, m.y_pred_old):.4f}")
verdict = ("IDENTICAL" if np.abs(d).max() < 1e-6 else
           "SMALL" if np.abs(d).max() < 5.0 else "LARGE")
print(f"  verdict: {verdict}")
