#!/usr/bin/env python3
"""Build the direct Copernicus ERA5 EDA notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_copernicus_era5_eda.ipynb"

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
        """# PRC Data Challenge 2026 — direct Copernicus ERA5 EDA

## TL;DR

The two-day EDDF access test returned 48 hourly rows with no duplicates or missing values. All eight physical-range checks passed. Temperature was 6.13–10.07 °C, total precipitation was 20.50 mm, and maximum 10 m wind speed was 9.12 m/s. Rerun this notebook with `PRC_CDS_ERA5_SCOPE=full` after the background download completes. These source checks do not claim a score gain."""
    ),
    nbf.v4.new_markdown_cell(
        """## Context and methods

### Key assumptions

- CDS is the source of record. The ERA5 data license is CC BY 4.0.
- The point product selects the nearest 0.25-degree grid cell.
- Accumulated precipitation and snowfall describe the hour ending at the timestamp.
- The profile checks source quality only. Event-hour reanalysis stays outside the final model until the organizer confirms the prediction cut-off."""
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
from profile_copernicus_era5 import build_profile

SCOPE = os.getenv("PRC_CDS_ERA5_SCOPE", "test")
DATA_DIR = Path(os.getenv("PRC_CDS_ERA5_DATA_DIR", ROOT / "data/external/weather/copernicus_era5")).resolve()
OUTPUT_DIR = Path(os.getenv("PRC_CDS_ERA5_ARTIFACT_DIR", ROOT / "artifacts/eda/copernicus_era5")).resolve()
SCOPE, DATA_DIR, OUTPUT_DIR"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        """summary = build_profile(DATA_DIR, OUTPUT_DIR, SCOPE)
summary"""
    ),
    nbf.v4.new_code_cell(
        """profile_dir = OUTPUT_DIR / SCOPE
weather = pd.read_parquet(profile_dir / "weather.parquet")
weather.head()"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """pd.read_csv(profile_dir / "variable_summary.csv")"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(profile_dir / "airport_summary.csv")"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(profile_dir / "weather_overview.png")))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

- Keep the batch only if all airport-hour, missingness, and physical-range checks pass.
- Use the complete download for forward validation and weather-feature ablations.
- Treat grid-cell weather as a candidate signal. It is not a substitute for direct airport observations.
- Do not use event-hour ranking weather in the prize model until the organizer confirms that the competition uses a retrospective cut-off."""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
