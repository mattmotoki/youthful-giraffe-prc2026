#!/usr/bin/env python3
"""Kaggle entry point for exp-137's private proxy arm."""
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
    matches = sorted(path for path in INPUT.rglob("fit_exp137_arm.py") if path.is_file())
    if len(matches) != 1:
        roots = sorted(path.name for path in INPUT.iterdir()) if INPUT.exists() else []
        raise RuntimeError(f"expected one exp-137 runner, found {len(matches)}; mounts={roots}")
    source = matches[0]
    sys.path.insert(0, str(source.parent))
    kernel_code = Path(__file__).resolve()
    os.environ["PRC_KERNEL_SLUG"] = f"{kaggle_user()}/prc-exp137-with-clock-proxy-arm"
    os.environ["PRC_KERNEL_VERSION"] = "2"
    os.environ["PRC_EXP130_DATASET_VERSION"] = "1"
    os.environ["PRC_EXP136_DATASET_VERSION"] = "4"
    os.environ["PRC_EXP137_DATASET_VERSION"] = "2"
    os.environ["PRC_EXP137_ARM"] = "proxy"
    os.environ["PRC_KERNEL_CODE_SHA256"] = hashlib.sha256(kernel_code.read_bytes()).hexdigest()
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    run_kernel()
