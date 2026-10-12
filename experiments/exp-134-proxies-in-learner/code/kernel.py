#!/usr/bin/env python3
"""Kaggle script entry point for the private exp-134 CPU fit."""
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
    matches = sorted(path for path in INPUT.rglob("fit_exp134.py") if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"expected one private exp-134 fit source; found {len(matches)}")
    source = matches[0]
    sys.path.insert(0, str(source.parent))
    kernel_code = Path(__file__).resolve()
    os.environ["PRC_KERNEL_SLUG"] = f"{kaggle_user()}/prc-exp134-proxy-learner"
    os.environ["PRC_KERNEL_VERSION"] = "2"
    os.environ["PRC_SPEC_COMMIT"] = "441bacb3de357aaf398cde6148974fee7f400d1c"
    os.environ["PRC_DATASET_VERSION"] = "3"
    os.environ["PRC_KERNEL_CODE_SHA256"] = hashlib.sha256(kernel_code.read_bytes()).hexdigest()
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    run_kernel()
