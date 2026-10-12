#!/usr/bin/env python3
"""Kaggle script entry point for the private exp-130 CPU fit."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user


def run_kernel():
    INPUT = Path("/kaggle/input")
    matches = sorted(path for path in INPUT.rglob("fit.py") if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"exact-file lookup for fit.py found {len(matches)} files: {matches}")

    os.environ["PRC_KERNEL_SLUG"] = f"{kaggle_user()}/prc-exp130-weighted-no-clock"
    os.environ["PRC_KERNEL_VERSION"] = "1"
    os.environ["PRC_DATASET_ID"] = f"{kaggle_user()}/prc-exp130-clean-input"
    os.environ["PRC_DATASET_VERSION"] = "1"
    runpy.run_path(str(matches[0]), run_name="__main__")


if __name__ == "__main__":
    run_kernel()
