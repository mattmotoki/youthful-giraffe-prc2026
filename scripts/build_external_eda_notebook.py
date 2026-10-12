#!/usr/bin/env python3
"""Build the external-data EDA notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_external_data_eda.ipynb"

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
        """# PRC Data Challenge 2026 — external-data EDA

## TL;DR

This notebook profiles two external sources: the open PRC 2024 flight list with 2022 taxi-out values, and airport-point ERA5 weather for 2024–2025. It measures overlap, shift, weather join coverage, and simple weather associations. It does not claim a score gain. The Open-Meteo weather snapshot is EDA-only until its service terms and the competition prediction cut-off are resolved."""
    ),
    nbf.v4.new_markdown_cell(
        """## Context and methods

### Key assumptions

- The unit of analysis is one departure movement.
- Historical values are used only as priors or aggregates. They are not joined by flight identity.
- Weather is joined by departure airport and scheduled UTC hour for this EDA.
- Both data products use CC BY 4.0 attribution and are recorded in `references/external-data-lock.json`.
- The free Open-Meteo service has separate non-commercial terms. Its snapshot is not approved for the prize model.
- ERA5 is delayed reanalysis. A forward split does not make post-event values available at prediction time.
- Actual ranking-period weather and trajectories remain outside the model until the organizer defines the prediction cut-off."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import os
import sys
import pandas as pd
from IPython.display import Image, display

default_root = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
ROOT = Path(os.getenv("PRC_REPO_DIR", default_root)).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from profile_external_data import build_profile

EXTERNAL_DATA_DIR = Path(os.getenv("PRC_EXTERNAL_DATA_DIR", ROOT / "data/external")).resolve()
OFFICIAL_DATA_DIR = Path(os.getenv("PRC_DATA_DIR", ROOT / "data/raw")).resolve()
ARTIFACTS_DIR = Path(os.getenv("PRC_EXTERNAL_ARTIFACT_DIR", ROOT / "artifacts/eda/external")).resolve()
EXTERNAL_DATA_DIR, OFFICIAL_DATA_DIR, ARTIFACTS_DIR"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        """summary = build_profile(EXTERNAL_DATA_DIR, OFFICIAL_DATA_DIR, ARTIFACTS_DIR)
summary"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "external_file_manifest.csv")"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """historical_shift = pd.read_csv(ARTIFACTS_DIR / "historical_airport_shift.csv")
historical_shift"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "airport_dataset_coverage.csv")"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "historical_aircraft_overlap.csv")"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ARTIFACTS_DIR / "historical_airport_medians.png")))"""
    ),
    nbf.v4.new_code_cell(
        """weather_coverage = pd.read_csv(ARTIFACTS_DIR / "weather_join_coverage.csv")
weather_coverage"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "weather_target_quality.csv")"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ARTIFACTS_DIR / "weather_join_coverage.png")))"""
    ),
    nbf.v4.new_code_cell(
        """weather_effects = pd.read_csv(ARTIFACTS_DIR / "weather_effects.csv")
weather_effects"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ARTIFACTS_DIR / "weather_wet_dry_effect.png")))"""
    ),
    nbf.v4.new_markdown_cell("### Optional full-trajectory sample"),
    nbf.v4.new_code_cell(
        """pd.read_csv(ARTIFACTS_DIR / "trajectory_sample_summary.csv")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

- Keep a source only if coverage and schema checks pass.
- Treat historical and weather relationships as candidate mechanisms, not proof of gain.
- Next, test leakage-safe historical aggregates in the fixed forward-only validation folds.
- Test weather only after the organizer permits post-event reanalysis and the retrieval route is valid for prize use. Otherwise, replace it with forecasts available at the declared cut-off.
- Do not use same-flight external records or post-event values that can reconstruct a hidden target."""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
