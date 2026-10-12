#!/usr/bin/env python3
"""Build the review notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/00_data_quality_eda.ipynb"

notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# PRC Data Challenge 2026 — initial data-quality EDA

## TL;DR

This notebook checks the official data package before modeling. It verifies file integrity, schemas, row counts, keys, missing values, time coverage, major categories, target distribution, and potential target leakage. Actual-time fields are quarantined from model features until the organizer confirms that they are valid at prediction time."""
    ),
    nbf.v4.new_markdown_cell(
        """## Context and methods

- Source: the caller's private data bucket in the OpenSky S3 service.
- Local data path: `../data/raw`; the data is not committed.
- Outputs: bounded CSV and JSON tables under `../artifacts/eda`; these are not committed.
- Method: Parquet metadata plus DuckDB scans. The script limits DuckDB to eight threads for the shared remote workstation."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import os
import sys
import pandas as pd

default_root = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
ROOT = Path(os.getenv("PRC_REPO_DIR", default_root)).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from profile_data import build_profile

DATA_DIR = Path(os.getenv("PRC_DATA_DIR", ROOT / "data/raw")).resolve()
ARTIFACTS_DIR = Path(os.getenv("PRC_ARTIFACT_DIR", ROOT / "artifacts/eda")).resolve()
DATA_DIR, ARTIFACTS_DIR"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        """summary = build_profile(DATA_DIR, ARTIFACTS_DIR)
summary"""
    ),
    nbf.v4.new_code_cell(
        """manifest = pd.read_csv(ARTIFACTS_DIR / "file_manifest.csv")
manifest[["file", "bytes", "rows", "columns", "row_groups", "sha256"]]"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """shapes = pd.read_csv(ARTIFACTS_DIR / "dataset_shapes.csv")
shapes"""
    ),
    nbf.v4.new_code_cell(
        """nulls = pd.read_csv(ARTIFACTS_DIR / "null_profile.csv")
nulls.groupby("dataset", group_keys=False).head(15)"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "key_checks.csv")"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "time_coverage.csv")"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "major_categories.csv").groupby("column", group_keys=False).head(20)"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "target_profile.csv")"""
    ),
    nbf.v4.new_markdown_cell(
        """### Leakage quarantine

The checks below are diagnostic only. Do not use the listed actual-time fields as model inputs until their availability at the prediction timestamp is confirmed. An exact or near-exact reconstruction of taxi time is evidence of leakage, not a modeling win."""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "leakage_checks.csv")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

After execution, use the results above to set the first validation split, define feature-availability rules, and choose a simple airport-and-time baseline. Keep the raw package and generated artifacts outside Git."""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
