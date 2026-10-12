"""Seed-bag curve: RMSE against the number of seeds averaged, per fold.

The per-seed predictions are stored raw as `y_pred_s0..` columns in each arm's
OOF file, so the curve needs no refit: it averages the first k seeds and then
applies the same post-processing the arm reports -- the 60 s floor and the
parameter-free residual guard on the fold's own training-month bounds -- so the
k = n point reproduces the arm's `y_pred` exactly.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-001-retro-v1"))
FOLDS = ("july", "july_loo", "december", "january")
FLOOR = 60.0

import json  # noqa: E402
import guard as guard_mod  # noqa: E402

BOUNDS = json.loads((HERE / "guard_bounds.json").read_text()) if (HERE / "guard_bounds.json").exists() else {}


def post(p, d, fold):
    """The arm's own post-processing: floor, then the residual guard."""
    p = np.maximum(np.asarray(p, "float64"), FLOOR)
    if fold in BOUNDS:
        lo, hi = BOUNDS[fold]
        p = guard_mod.apply(p, d["d_takeoff_aobt"].to_numpy("float64"), lo, hi, floor=FLOOR)
    return p


def curve(name: str, folds=FOLDS) -> pd.DataFrame:
    rows = []
    for f in folds:
        p = HERE / "oof" / f"{name}-{f}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        cols = sorted([c for c in d.columns if c.startswith("y_pred_s")],
                      key=lambda c: int(c.split("s")[-1]))
        y = d["y_true"].to_numpy("float64")
        P = d[cols].to_numpy("float64")
        # Average over ALL subsets of size k, not the first k in fit order. Taking
        # the first k makes the k = 1 point one particular seed's score, and if
        # that seed happens to be the lucky one the curve appears to get worse as
        # seeds are added. Bagging improves on the AVERAGE member, never
        # necessarily on the best one, so the honest curve averages the orderings.
        for k in range(1, len(cols) + 1):
            vals = []
            for idx in itertools.combinations(range(len(cols)), k):
                pk = post(P[:, list(idx)].mean(axis=1), d, f)
                vals.append(float(np.sqrt(((y - pk) ** 2).mean())))
            rows.append({"arm": name, "fold": f, "seeds": k,
                         "rmse": float(np.mean(vals)),
                         "best_subset": float(np.min(vals)),
                         "worst_subset": float(np.max(vals)),
                         "n_subsets": len(vals)})
    t = pd.DataFrame(rows)
    if len(t):
        t["gain_vs_1"] = t.groupby("fold")["rmse"].transform("first") - t["rmse"]
        t["gain_last_seed"] = t.groupby("fold")["rmse"].diff().fillna(0.0) * -1
    return t


if __name__ == "__main__":
    out = pd.concat([curve(n) for n in sys.argv[1:]], ignore_index=True)
    print(out.round(2).to_string(index=False))
    piv = out.pivot_table(index=["arm", "seeds"], columns="fold", values="rmse")
    print("\n" + piv.round(1).to_string())
