#!/usr/bin/env python3
"""Download and validate direct Copernicus ERA5 weather for PRC 2026."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cdsapi
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "external" / "weather" / "copernicus_era5"
TIME_SERIES_DATASET = "reanalysis-era5-single-levels-timeseries"
PARENT_DATASET = "reanalysis-era5-single-levels"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
TIME_SERIES_DOI = "10.24381/1cf1ad76"
PARENT_DOI = "10.24381/cds.adbb2d47"

AIRPORTS = {
    "EDDF": (50.026, 8.543),
    "EDDM": (48.354, 11.786),
    "EGLL": (51.478, -0.461),
    "EHAM": (52.309, 4.764),
    "LEBL": (41.297, 2.078),
    "LEMD": (40.494, -3.567),
    "LFPG": (49.013, 2.550),
    "LIRF": (41.804, 12.251),
    "LTAI": (36.899, 30.800),
    "LTFM": (41.275, 28.752),
    "LSZH": (47.465, 8.549),
}

CORE_VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "total_cloud_cover",
)
EXTRA_VARIABLES = ("low_cloud_cover", "snowfall")
VARIABLE_ALIASES = {
    "2m_temperature": ("2m_temperature", "t2m"),
    "2m_dewpoint_temperature": ("2m_dewpoint_temperature", "d2m"),
    "mean_sea_level_pressure": ("mean_sea_level_pressure", "msl"),
    "10m_u_component_of_wind": ("10m_u_component_of_wind", "u10"),
    "10m_v_component_of_wind": ("10m_v_component_of_wind", "v10"),
    "total_precipitation": ("total_precipitation", "tp"),
    "total_cloud_cover": ("total_cloud_cover", "tcc"),
    "low_cloud_cover": ("low_cloud_cover", "lcc"),
    "snowfall": ("snowfall", "sf"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_longitude(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def unique_scalar(dataset: xr.Dataset, names: Iterable[str]) -> float:
    for name in names:
        if name not in dataset.coords and name not in dataset.variables:
            continue
        values = pd.Series(dataset[name].values.ravel()).dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Expected one {name} value, found {len(values)}")
        return float(values[0])
    raise ValueError(f"Missing coordinate; tried {tuple(names)}")


def expected_hours(periods: list[tuple[str, str]]) -> pd.DatetimeIndex:
    indexes = [
        pd.date_range(start, end + " 23:00:00", freq="h", tz="UTC")
        for start, end in periods
    ]
    result = indexes[0]
    for index in indexes[1:]:
        result = result.append(index)
    return result


def validate_netcdf(
    path: Path,
    variables: tuple[str, ...],
    periods: list[tuple[str, str]],
    expected_cell: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty NetCDF file: {path}")
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        time_name = next(
            (name for name in ("valid_time", "time") if name in dataset.coords),
            None,
        )
        if time_name is None:
            raise ValueError(f"Missing time coordinate in {path}")
        timestamps = pd.DatetimeIndex(pd.to_datetime(dataset[time_name].values, utc=True))
        expected = expected_hours(periods)
        if not timestamps.equals(expected):
            raise ValueError(
                f"Time coverage mismatch in {path}: "
                f"actual={len(timestamps)} expected={len(expected)} "
                f"first={timestamps.min()} last={timestamps.max()}"
            )
        if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unsorted timestamps in {path}")

        missing_variables = [
            variable
            for variable in variables
            if not any(alias in dataset.data_vars for alias in VARIABLE_ALIASES[variable])
        ]
        if missing_variables:
            raise ValueError(f"Missing variables in {path}: {missing_variables}")

        latitude = unique_scalar(dataset, ("latitude", "lat"))
        longitude = normalized_longitude(
            unique_scalar(dataset, ("longitude", "lon"))
        )
        if expected_cell is not None and (
            not math.isclose(latitude, expected_cell[0], abs_tol=1e-6)
            or not math.isclose(longitude, expected_cell[1], abs_tol=1e-6)
        ):
            raise ValueError(
                f"Grid-cell mismatch in {path}: "
                f"actual={(latitude, longitude)} expected={expected_cell}"
            )
        return {
            "latitude": latitude,
            "longitude": longitude,
            "hours": len(timestamps),
            "first_hour_utc": timestamps[0].isoformat(),
            "last_hour_utc": timestamps[-1].isoformat(),
            "data_variables": sorted(dataset.data_vars),
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def quarantine_invalid(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantined = path.with_name(path.name + f".invalid-{timestamp}")
    path.replace(quarantined)
    print(f"quarantined invalid file: {quarantined}", flush=True)


def normalize_download(path: Path, archive_path: Path) -> None:
    """Convert a CDS ZIP response into one merged NetCDF file."""
    if not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() == ".nc"
        ]
        if not members:
            raise ValueError(
                f"Expected at least one NetCDF member in {path}, found none"
            )
        extracted = path.with_suffix(path.suffix + ".extracting")
        if extracted.exists():
            quarantine_invalid(extracted)
        if len(members) == 1:
            with archive.open(members[0]) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        else:
            member_paths = []
            for index, member in enumerate(members):
                member_path = path.with_suffix(path.suffix + f".member-{index}.nc")
                if member_path.exists():
                    quarantine_invalid(member_path)
                with archive.open(member) as source, member_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                member_paths.append(member_path)
            datasets = []
            try:
                for member_path in member_paths:
                    with xr.open_dataset(member_path, engine="netcdf4") as dataset:
                        datasets.append(dataset.load())
                merged = xr.merge(datasets, compat="override", join="exact")
                merged.to_netcdf(extracted, engine="netcdf4")
                merged.close()
            finally:
                for dataset in datasets:
                    dataset.close()
                for member_path in member_paths:
                    member_path.unlink(missing_ok=True)
    if archive_path.exists():
        quarantine_invalid(archive_path)
    path.replace(archive_path)
    extracted.replace(path)
    print(
        f"unpacked CDS archive: archive={archive_path} netcdf={path}",
        flush=True,
    )


def retrieve_and_validate(
    client: cdsapi.Client,
    dataset: str,
    request: dict[str, Any],
    target: Path,
    variables: tuple[str, ...],
    periods: list[tuple[str, str]],
    expected_cell: tuple[float, float] | None,
    request_dir: Path,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / (target.stem + ".request.json")
    request_record = {
        "dataset": dataset,
        "request": request,
        "target": str(target),
    }
    write_json(request_path, request_record)
    archive_path = target.with_name(target.name + ".download.zip")

    if target.exists():
        try:
            validation = validate_netcdf(
                target, variables, periods, expected_cell=expected_cell
            )
            print(f"verified existing: {target}", flush=True)
            return build_manifest_record(
                target,
                dataset,
                request_path,
                validation,
                downloaded=False,
                archive_path=archive_path if archive_path.exists() else None,
            )
        except (OSError, ValueError) as error:
            print(f"existing file failed validation: {error}", file=sys.stderr, flush=True)
            quarantine_invalid(target)

    partial = target.with_suffix(target.suffix + ".part")
    validation = None
    if partial.exists():
        try:
            normalize_download(partial, archive_path)
            validation = validate_netcdf(
                partial, variables, periods, expected_cell=expected_cell
            )
            print(f"recovered complete partial download: {partial}", flush=True)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            print(f"partial file failed validation: {error}", file=sys.stderr, flush=True)
            quarantine_invalid(partial)
    if validation is None:
        print(f"request start: dataset={dataset} target={target}", flush=True)
        client.retrieve(dataset, request, str(partial))
        normalize_download(partial, archive_path)
        validation = validate_netcdf(
            partial, variables, periods, expected_cell=expected_cell
        )
    partial.replace(target)
    print(
        f"request complete: target={target} bytes={target.stat().st_size:,} "
        f"hours={validation['hours']}",
        flush=True,
    )
    return build_manifest_record(
        target,
        dataset,
        request_path,
        validation,
        downloaded=True,
        archive_path=archive_path if archive_path.exists() else None,
    )


def build_manifest_record(
    path: Path,
    dataset: str,
    request_path: Path,
    validation: dict[str, Any],
    downloaded: bool,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    record = {
        "dataset": dataset,
        "doi": TIME_SERIES_DOI if dataset == TIME_SERIES_DATASET else PARENT_DOI,
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "file": str(path),
        "request_file": str(request_path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "retrieved_or_verified_utc": utc_now(),
        "downloaded_this_run": downloaded,
        **validation,
    }
    if archive_path is not None:
        record["source_archive"] = str(archive_path)
        record["source_archive_bytes"] = archive_path.stat().st_size
        record["source_archive_sha256"] = sha256(archive_path)
    return record


def core_request(latitude: float, longitude: float, start: str, end: str) -> dict[str, Any]:
    return {
        "variable": list(CORE_VARIABLES),
        "location": {"latitude": latitude, "longitude": longitude},
        "date": [f"{start}/{end}"],
        "data_format": "netcdf",
    }


def extra_request(
    cell: tuple[float, float], year: str, months: list[str], days: list[str]
) -> dict[str, Any]:
    latitude, longitude = cell
    return {
        "product_type": ["reanalysis"],
        "variable": list(EXTRA_VARIABLES),
        "year": [year],
        "month": months,
        "day": days,
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": [latitude, longitude, latitude, longitude],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def save_manifest(data_dir: Path, records: list[dict[str, Any]]) -> None:
    manifest_path = data_dir / "manifest.json"
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {record["file"]: record for record in payload.get("files", [])}
    existing.update({record["file"]: record for record in records})
    write_json(
        manifest_path,
        {
            "created_or_updated_utc": utc_now(),
            "source": "Copernicus Climate Data Store",
            "license": "CC BY 4.0",
            "files": [existing[key] for key in sorted(existing)],
        },
    )


def run_test(client: cdsapi.Client, data_dir: Path) -> None:
    airport = "EDDF"
    latitude, longitude = AIRPORTS[airport]
    periods = [("2024-01-01", "2024-01-02")]
    test_dir = data_dir / "test"
    request_dir = data_dir / "requests"
    core_target = test_dir / "EDDF_core_2024-01-01_2024-01-02.nc"
    core_record = retrieve_and_validate(
        client,
        TIME_SERIES_DATASET,
        core_request(latitude, longitude, periods[0][0], periods[0][1]),
        core_target,
        CORE_VARIABLES,
        periods,
        expected_cell=None,
        request_dir=request_dir,
    )
    cell = (core_record["latitude"], core_record["longitude"])
    extra_target = test_dir / "EDDF_extra_2024-01-01_2024-01-02.nc"
    extra_record = retrieve_and_validate(
        client,
        PARENT_DATASET,
        extra_request(cell, "2024", ["01"], ["01", "02"]),
        extra_target,
        EXTRA_VARIABLES,
        periods,
        expected_cell=cell,
        request_dir=request_dir,
    )
    save_manifest(data_dir, [core_record, extra_record])
    print(
        f"test passed: cell={cell} files=2 "
        f"bytes={core_record['bytes'] + extra_record['bytes']:,}",
        flush=True,
    )


def run_full(client: cdsapi.Client, data_dir: Path, airports: list[str]) -> None:
    raw_dir = data_dir / "raw"
    request_dir = data_dir / "requests"
    core_periods = [("2024-01-01", "2026-07-31")]
    extra_periods = {
        "2024": [("2024-01-01", "2024-12-31")],
        "2025": [("2025-01-01", "2025-12-31")],
        "2026_01_07": [
            ("2026-01-01", "2026-01-31"),
            ("2026-07-01", "2026-07-31"),
        ],
    }
    records: list[dict[str, Any]] = []
    for airport in airports:
        latitude, longitude = AIRPORTS[airport]
        core_target = raw_dir / f"{airport}_core_2024-01-01_2026-07-31.nc"
        core_record = retrieve_and_validate(
            client,
            TIME_SERIES_DATASET,
            core_request(latitude, longitude, "2024-01-01", "2026-07-31"),
            core_target,
            CORE_VARIABLES,
            core_periods,
            expected_cell=None,
            request_dir=request_dir,
        )
        records.append(core_record)
        save_manifest(data_dir, records)
        cell = (core_record["latitude"], core_record["longitude"])

        extra_specs = (
            ("2024", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"], [f"{day:02d}" for day in range(1, 32)]),
            ("2025", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"], [f"{day:02d}" for day in range(1, 32)]),
            ("2026_01_07", ["01", "07"], [f"{day:02d}" for day in range(1, 32)]),
        )
        for label, months, days in extra_specs:
            year = label[:4]
            extra_target = raw_dir / f"{airport}_extra_{label}.nc"
            extra_record = retrieve_and_validate(
                client,
                PARENT_DATASET,
                extra_request(cell, year, months, days),
                extra_target,
                EXTRA_VARIABLES,
                extra_periods[label],
                expected_cell=cell,
                request_dir=request_dir,
            )
            records.append(extra_record)
            save_manifest(data_dir, records)
    print(f"full download passed: airports={len(airports)} files={len(records)}", flush=True)


def parse_airports(value: str) -> list[str]:
    airports = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(airports) - set(AIRPORTS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown airports: {unknown}")
    if not airports:
        raise argparse.ArgumentTypeError("At least one airport is required")
    return airports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "full"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--airports",
        type=parse_airports,
        default=list(AIRPORTS),
        help="Comma-separated airport codes; default: all 11",
    )
    arguments = parser.parse_args()
    data_dir = arguments.data_dir.expanduser().resolve()
    if not (Path.home() / ".cdsapirc").is_file():
        raise FileNotFoundError("Missing $HOME/.cdsapirc")
    client = cdsapi.Client(timeout=900, quiet=False, debug=False)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    if arguments.mode == "test":
        run_test(client, data_dir)
    else:
        run_full(client, data_dir, arguments.airports)


if __name__ == "__main__":
    main()
