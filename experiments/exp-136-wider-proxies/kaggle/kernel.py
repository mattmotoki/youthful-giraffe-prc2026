#!/usr/bin/env python3
"""Kaggle script entry point for the private exp-136 CPU fit."""
from __future__ import annotations

import hashlib
import os
import runpy
import sys
from pathlib import Path


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user


def run_kernel():
    INPUT = Path("/kaggle/input")
    if not INPUT.exists():
        raise RuntimeError(f"Kaggle input root {INPUT} does not exist")
    matches = sorted(path for path in INPUT.glob("**/fit_exp136.py") if path.is_file())
    if len(matches) != 1:
        roots = sorted(path.name for path in INPUT.iterdir())
        raise RuntimeError(f"expected one attached fit_exp136.py, found {len(matches)}; "
                           f"input mounts={roots}, matches={[str(path) for path in matches]}")
    source = matches[0]
    sys.path.insert(0, str(source.parent))
    kernel_code = Path(__file__).resolve()
    os.environ["PRC_KERNEL_SLUG"] = f"{kaggle_user()}/prc-exp136-wider-runway-state-proxies"
    os.environ["PRC_KERNEL_VERSION"] = "2"
    os.environ["PRC_DATASET_VERSION"] = "3"
    os.environ["PRC_KERNEL_CODE_SHA256"] = hashlib.sha256(kernel_code.read_bytes()).hexdigest()
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    run_kernel()
