#!/usr/bin/env python3
"""Prove the cached folds are exactly what `prc.data.fold_frames` returns.

COMMON.md requires `fold_frames(fold)` for every measured number. `ablate.py`
assembles folds from the per-month cache instead, because re-reading thirteen
parquet files for each of forty-five fits is wasteful. That is only legitimate
if the two agree exactly, so this checks it rather than asserting it:
same departure ID set, same order, same labels, and the masked columns absent.
Sol round 3, requirement audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prc.data as d  # noqa: E402
from ablate import load_fold  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> None:
    for fold_name in ("january", "december", "july"):
        train_cache, valid_cache = load_fold(fold_name)
        train_ref, valid_ref, y_ref = d.fold_frames(fold_name)

        dep_t = train_ref["PHASE_mvt"] == "DEP"
        dep_v = valid_ref["PHASE_mvt"] == "DEP"

        check(f"[{fold_name}] train departure ids identical",
              np.array_equal(train_cache["MVT_ID_mvt"].to_numpy(),
                             train_ref.loc[dep_t, "MVT_ID_mvt"].to_numpy()),
              f"{len(train_cache):,} rows")
        check(f"[{fold_name}] valid departure ids identical",
              np.array_equal(valid_cache["MVT_ID_mvt"].to_numpy(),
                             valid_ref.loc[dep_v, "MVT_ID_mvt"].to_numpy()),
              f"{len(valid_cache):,} rows")
        check(f"[{fold_name}] train labels identical",
              np.allclose(train_cache["y"].to_numpy(),
                          train_ref.loc[dep_t, d.TARGET].to_numpy(), equal_nan=True))
        check(f"[{fold_name}] valid hidden labels identical",
              np.allclose(valid_cache["y"].to_numpy(),
                          y_ref[dep_v].to_numpy(), equal_nan=True))
        check(f"[{fold_name}] validation frame carries no block time or taxi time",
              valid_ref.loc[dep_v, "BLOCK_TIME_UTC_mvt"].isna().all()
              and valid_ref.loc[dep_v, d.TARGET].isna().all())
        check(f"[{fold_name}] cache exposes neither masked column as a feature",
              not {"BLOCK_TIME_UTC_mvt", d.TARGET}.intersection(valid_cache.columns))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("cached folds are identical to prc.data.fold_frames on all three folds")


if __name__ == "__main__":
    main()
