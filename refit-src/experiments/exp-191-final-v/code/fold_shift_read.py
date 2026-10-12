"""Split the change of the held fold G between the ranking-mode pack and a final-mode pack into its two parts:
the fold exposure weights (finalplan design row 3, the four-month cell mix) and the fold values (the weight records that the
replay refits with those weights). Each stage step (v64 against zabd, c5 against v64, L2C against c5) is scored
with the values of tree A or B and the weights of tree A or B. G = sum w ((y - old)^2 - (y - new)^2).

    python fold_shift_read.py <ranking run tree A> <final run tree B> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
A, B, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
COLS = [ID, "month", "airport", "cls", "y", "w", "zabd", "v64", "c5"]


def load(run):
    d = pd.read_parquet(run / "work/w2/base_fold.parquet", columns=COLS)
    d["l2c"] = np.load(run / "work/w2/l2c_fold.npy").astype(float)
    return d


a, b = load(A), load(B)
assert np.array_equal(a[ID].to_numpy(), b[ID].to_numpy()) and np.array_equal(a.y.to_numpy(), b.y.to_numpy())
assert np.isfinite(a.y).all() and np.isfinite(a.w).all() and np.isfinite(b.w).all()
y = a.y.to_numpy(float)
month = a.month.to_numpy()
STEPS = [("v64", "zabd"), ("c5", "v64"), ("l2c", "c5")]


def g(vals, w, new, old, mask=None):
    d = w * ((y - vals[old].to_numpy(float)) ** 2 - (y - vals[new].to_numpy(float)) ** 2)
    return round(float(d.sum() if mask is None else d[mask].sum()), 1)


res = {"rows": len(a), "trees": {"A": str(A), "B": str(B)},
       "weights": {"sum_A": float(a.w.sum()), "sum_B": float(b.w.sum()),
                   "rows_differ": int((a.w != b.w).sum()),
                   "sum_by_month_A": {int(m): round(float(a.w[month == m].sum()), 4) for m in (1, 7)},
                   "sum_by_month_B": {int(m): round(float(b.w[month == m].sum()), 4) for m in (1, 7)},
                   "rows_zero_A": int((a.w == 0).sum()), "rows_zero_B": int((b.w == 0).sum())},
       "values": {c: {"rows_differ": int((a[c] != b[c]).sum()), "max_abs_diff": float(np.abs(a[c] - b[c]).max())}
                  for c in ("zabd", "v64", "c5", "l2c")},
       "g": {}}
wa, wb = a.w.to_numpy(float), b.w.to_numpy(float)
for new, old in STEPS:
    res["g"][f"{new}_vs_{old}"] = {
        "valuesA_weightsA": g(a, wa, new, old), "valuesA_weightsB": g(a, wb, new, old),
        "valuesB_weightsA": g(b, wa, new, old), "valuesB_weightsB": g(b, wb, new, old),
        "valuesB_weightsB_by_month": {int(m): g(b, wb, new, old, month == m) for m in (1, 7)},
        "valuesA_weightsA_by_month": {int(m): g(a, wa, new, old, month == m) for m in (1, 7)}}
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
