#!/usr/bin/env python3
"""Copy the complete per-row exp-127b OOF deltas into a corrected fold NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = ("july_loo", "december", "january")
EXPECTED = {"july_loo": 87614, "december": 100515, "january": 69501}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.predictions)
    required = {"MVT_ID_mvt", "fold", "airport", "pred_implied"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["MVT_ID_mvt", "fold"]).any():
        raise ValueError("duplicate ID within fold")
    if not np.isfinite(frame["pred_implied"]).all():
        raise ValueError("non-finite OOF update")
    if set(frame["fold"].unique()) != set(FOLDS):
        raise ValueError(f"unexpected fold set: {sorted(frame['fold'].unique())}")

    arrays: dict[str, np.ndarray] = {}
    for fold in FOLDS:
        part = frame.loc[frame["fold"].eq(fold)]
        if len(part) != EXPECTED[fold]:
            raise ValueError(f"{fold}: expected {EXPECTED[fold]} rows, got {len(part)}")
        arrays[f"{fold}_MVT_ID_mvt"] = part["MVT_ID_mvt"].to_numpy(dtype=np.int64)
        arrays[f"{fold}_delta"] = part["pred_implied"].to_numpy(dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"wrote {args.output} with {len(frame)} OOF rows across {len(FOLDS)} folds")


if __name__ == "__main__":
    main()
