"""exp-168: load raw challenge files (read-only looks; no fits)."""
import glob
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
      "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]


def train(months=None):
    files = sorted(glob.glob(os.path.join(RAW, "training_2025-*.parquet")))
    if months:
        files = [f for f in files if any(f"training_2025-{m:02d}-01" in f for m in months)]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def ranking():
    return pd.read_parquet(os.path.join(RAW, "ranking.parquet"))


def secs(a, b):
    """a - b in seconds (float, NaN where either is missing)."""
    return (a - b).dt.total_seconds()
