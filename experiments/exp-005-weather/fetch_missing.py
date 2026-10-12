#!/usr/bin/env python3
"""Fetch the CDS ERA5 files missing from the snapshot, in priority order.

Priority: LSZH core (needed for its grid cell), then the two ranking-month
extra files, then 2025 extras, then the 2024 extras that only feed EDA.
Uses the shared download script's request builders and validator so the
manifest records stay identical in shape to the existing ones.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import cdsapi  # noqa: E402
import download_copernicus_era5 as dl  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data" / "external" / "weather" / "copernicus_era5"

MONTHS_ALL = [f"{m:02d}" for m in range(1, 13)]
DAYS_ALL = [f"{d:02d}" for d in range(1, 32)]
EXTRA_PERIODS = {
    "2024": [("2024-01-01", "2024-12-31")],
    "2025": [("2025-01-01", "2025-12-31")],
    "2026_01_07": [("2026-01-01", "2026-01-31"), ("2026-07-01", "2026-07-31")],
}
EXTRA_MONTHS = {"2024": MONTHS_ALL, "2025": MONTHS_ALL, "2026_01_07": ["01", "07"]}

# (airport, kind, label) in the order they should be requested.
PLAN = [
    ("LSZH", "core", None),
    ("LSZH", "extra", "2025"),
    ("LSZH", "extra", "2026_01_07"),
    ("LTFM", "extra", "2026_01_07"),
    ("LSZH", "extra", "2024"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    raw = data_dir / "raw"
    req = data_dir / "requests"
    client = cdsapi.Client(timeout=900, quiet=False, debug=False)
    cells: dict[str, tuple[float, float]] = {}
    records = []
    for airport, kind, label in PLAN:
        if kind == "core":
            lat, lon = dl.AIRPORTS[airport]
            target = raw / f"{airport}_core_2024-01-01_2026-07-31.nc"
            record = dl.retrieve_and_validate(
                client,
                dl.TIME_SERIES_DATASET,
                dl.core_request(lat, lon, "2024-01-01", "2026-07-31"),
                target,
                dl.CORE_VARIABLES,
                [("2024-01-01", "2026-07-31")],
                expected_cell=None,
                request_dir=req,
            )
            cells[airport] = (record["latitude"], record["longitude"])
        else:
            if airport not in cells:
                core = raw / f"{airport}_core_2024-01-01_2026-07-31.nc"
                v = dl.validate_netcdf(core, dl.CORE_VARIABLES, [("2024-01-01", "2026-07-31")])
                cells[airport] = (v["latitude"], v["longitude"])
            target = raw / f"{airport}_extra_{label}.nc"
            record = dl.retrieve_and_validate(
                client,
                dl.PARENT_DATASET,
                dl.extra_request(cells[airport], label[:4], EXTRA_MONTHS[label], DAYS_ALL),
                target,
                dl.EXTRA_VARIABLES,
                EXTRA_PERIODS[label],
                expected_cell=cells[airport],
                request_dir=req,
            )
        records.append(record)
        dl.save_manifest(data_dir, records)
        print(f"done: {airport} {kind} {label}", flush=True)
    print(f"fetch_missing complete: files={len(records)}", flush=True)


if __name__ == "__main__":
    main()
