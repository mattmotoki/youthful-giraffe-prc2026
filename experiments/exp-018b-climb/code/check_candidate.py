"""Re-verify a written candidate from disk, and diff it against a reference.

`prc.data.write_submission` already refuses a bad id set, a bad order, a
non-float64 dtype and a non-finite value, so this is not that check again: it
is the check that the file on disk is the file that passed, and a read of how
far it sits from the candidate it is meant to replace.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/check_candidate.py \
      submissions/candidates/exp-018b-s1.parquet [reference.parquet]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from prc import data

path = Path(sys.argv[1])
ref = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "submissions/candidates/exp-018-fresh.parquet")

df = pd.read_parquet(path)
tmpl = data.load_submitting()
ids, vals = df[data.ID].to_numpy(), df[data.TARGET].to_numpy()

print(f"file      {path}")
print(f"sha256    {data.sha256(path)}")
print(f"rows      {len(df)}  (template {len(tmpl)})")
print(f"id order  {'exact' if np.array_equal(ids, tmpl[data.ID].astype('float64').to_numpy()) else 'MISMATCH'}")
print(f"dtype     {df[data.TARGET].dtype}   finite {bool(np.isfinite(vals).all())}")
print(f"min {vals.min():.1f}  median {np.median(vals):.1f}  mean {vals.mean():.1f}  "
      f"p99 {np.percentile(vals, 99):.0f}  max {vals.max():.0f}")
print(f"below 60 s {(vals < 60).sum()}   above 3600 s {(vals > 3600).sum()}")

if ref.exists():
    r = pd.read_parquet(ref)
    assert np.array_equal(r[data.ID].to_numpy(), ids), "reference is a different id order"
    d = vals - r[data.TARGET].to_numpy()
    print(f"\nvs {ref.name}")
    print(f"  rms diff  {np.sqrt(np.mean(d ** 2)):.2f} s   mean {d.mean():+.2f} s   "
          f"max |diff| {np.abs(d).max():.0f} s")
    print(f"  rows moving more than 60 s: {(np.abs(d) > 60).sum()} "
          f"({100 * (np.abs(d) > 60).mean():.2f} %)")
