"""Repair OOF files whose id column holds a positional index.

`features.month_features` resets the index, so an early version of
`run_ablation` wrote 0..n-1 into `MVT_ID_mvt`. The row order is the departure
order of the month frame and is deterministic, so the true ids are recoverable
exactly; this rewrites them in place and verifies the target column matches.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D

for p in sorted((HERE / "oof").glob("*.parquet")):
    o = pd.read_parquet(p)
    fold = o["fold"].iloc[0]
    raw = D.load_month(D.FOLDS[fold].valid_month, columns=[D.ID, "PHASE_mvt", D.TARGET])
    dep = raw[raw["PHASE_mvt"] == "DEP"]
    if len(dep) != len(o):
        print("SKIP", p.name, "row count differs"); continue
    if (o[D.ID].to_numpy() == dep[D.ID].to_numpy()).all():
        print("ok  ", p.name); continue
    ids = dep[D.ID].to_numpy()
    assert np.allclose(o["y_true"].to_numpy(), dep[D.TARGET].to_numpy(float)), p.name
    o[D.ID] = ids
    o.to_parquet(p, index=False)
    print("fixed", p.name)
