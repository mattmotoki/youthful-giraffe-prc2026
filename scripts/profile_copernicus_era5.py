#!/usr/bin/env python3
"""Build a reproducible quality profile for direct Copernicus ERA5 files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from download_copernicus_era5 import (
    AIRPORTS,
    CORE_VARIABLES,
    EXTRA_VARIABLES,
    VARIABLE_ALIASES,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "external" / "weather" / "copernicus_era5"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "eda" / "copernicus_era5"
CORE_OUTPUT_NAMES = {
    "2m_temperature": "temperature_2m_c",
    "2m_dewpoint_temperature": "dew_point_2m_c",
    "mean_sea_level_pressure": "pressure_msl_hpa",
    "10m_u_component_of_wind": "wind_u_10m_ms",
    "10m_v_component_of_wind": "wind_v_10m_ms",
    "total_precipitation": "precipitation_mm",
    "total_cloud_cover": "cloud_cover_pct",
}
EXTRA_OUTPUT_NAMES = {
    "low_cloud_cover": "cloud_cover_low_pct",
    "snowfall": "snowfall_mm_water_equivalent",
}
ANALYSIS_PERIODS = (
    (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:00", tz="UTC")),
    (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-31 23:00", tz="UTC")),
    (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-07-31 23:00", tz="UTC")),
)


def find_time_name(dataset: xr.Dataset) -> str:
    time_name = next(
        (name for name in ("valid_time", "time") if name in dataset.coords),
        None,
    )
    if time_name is None:
        raise ValueError("NetCDF file has no valid_time or time coordinate")
    return time_name


def find_variable(dataset: xr.Dataset, requested_name: str) -> str:
    for alias in VARIABLE_ALIASES[requested_name]:
        if alias in dataset.data_vars:
            return alias
    raise ValueError(f"Missing ERA5 variable: {requested_name}")


def extract_values(
    dataset: xr.Dataset,
    time_name: str,
    requested_name: str,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    source_name = find_variable(dataset, requested_name)
    values = dataset[source_name].to_dataframe(name="value").reset_index()
    values[time_name] = pd.to_datetime(values[time_name], utc=True)
    by_time = values.dropna(subset=["value"]).groupby(time_name)["value"].first()
    return by_time.reindex(timestamps).reset_index(drop=True)


def load_file(path: Path, airport: str, variables: Iterable[str]) -> pd.DataFrame:
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        time_name = find_time_name(dataset)
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(dataset[time_name].values, utc=True),
            name="weather_hour_utc",
        )
        frame = pd.DataFrame(
            {
                "airport": airport,
                "weather_hour_utc": timestamps,
            }
        )
        for variable in variables:
            frame[variable] = extract_values(
                dataset, time_name, variable, timestamps
            ).to_numpy()
    if frame.duplicated(["airport", "weather_hour_utc"]).any():
        raise ValueError(f"Duplicate airport-hour keys in {path}")
    return frame


def transform_units(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns={**CORE_OUTPUT_NAMES, **EXTRA_OUTPUT_NAMES}).copy()
    for column in ("temperature_2m_c", "dew_point_2m_c"):
        if column in result:
            result[column] = result[column] - 273.15
    if "pressure_msl_hpa" in result:
        result["pressure_msl_hpa"] = result["pressure_msl_hpa"] / 100.0
    for column in ("precipitation_mm", "snowfall_mm_water_equivalent"):
        if column in result:
            result[column] = result[column] * 1000.0
    for column in ("cloud_cover_pct", "cloud_cover_low_pct"):
        if column in result:
            result[column] = result[column] * 100.0
    if {"wind_u_10m_ms", "wind_v_10m_ms"} <= set(result):
        result["wind_speed_10m_ms"] = (
            result["wind_u_10m_ms"] ** 2 + result["wind_v_10m_ms"] ** 2
        ) ** 0.5
    return result


def in_analysis_period(timestamp: pd.Series) -> pd.Series:
    selected = pd.Series(False, index=timestamp.index)
    for start, end in ANALYSIS_PERIODS:
        selected |= timestamp.between(start, end)
    return selected


def load_test(data_dir: Path) -> pd.DataFrame:
    core = load_file(
        data_dir / "test" / "EDDF_core_2024-01-01_2024-01-02.nc",
        "EDDF",
        CORE_VARIABLES,
    )
    extra = load_file(
        data_dir / "test" / "EDDF_extra_2024-01-01_2024-01-02.nc",
        "EDDF",
        EXTRA_VARIABLES,
    )
    return transform_units(
        core.merge(extra, on=["airport", "weather_hour_utc"], how="left", validate="one_to_one")
    )


def load_full(data_dir: Path) -> pd.DataFrame:
    frames = []
    raw_dir = data_dir / "raw"
    for airport in AIRPORTS:
        core = load_file(
            raw_dir / f"{airport}_core_2024-01-01_2026-07-31.nc",
            airport,
            CORE_VARIABLES,
        )
        core = core.loc[in_analysis_period(core["weather_hour_utc"])].copy()
        extras = pd.concat(
            [
                load_file(raw_dir / f"{airport}_extra_2024.nc", airport, EXTRA_VARIABLES),
                load_file(raw_dir / f"{airport}_extra_2025.nc", airport, EXTRA_VARIABLES),
                load_file(raw_dir / f"{airport}_extra_2026_01_07.nc", airport, EXTRA_VARIABLES),
            ],
            ignore_index=True,
        )
        frame = core.merge(
            extras,
            on=["airport", "weather_hour_utc"],
            how="left",
            validate="one_to_one",
        )
        frames.append(transform_units(frame))
    result = pd.concat(frames, ignore_index=True)
    expected_rows = 11 * 19_032
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} airport-hours, found {len(result):,}")
    return result


def variable_summary(weather: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in weather.columns
        if column not in {"airport", "weather_hour_utc"}
    ]
    return pd.DataFrame(
        [
            {
                "variable": column,
                "non_null_rows": int(weather[column].notna().sum()),
                "missing_rows": int(weather[column].isna().sum()),
                "minimum": float(weather[column].min()),
                "median": float(weather[column].median()),
                "mean": float(weather[column].mean()),
                "maximum": float(weather[column].max()),
            }
            for column in columns
        ]
    )


def airport_summary(weather: pd.DataFrame) -> pd.DataFrame:
    return (
        weather.groupby("airport", as_index=False)
        .agg(
            hours=("weather_hour_utc", "size"),
            first_hour_utc=("weather_hour_utc", "min"),
            last_hour_utc=("weather_hour_utc", "max"),
            mean_temperature_c=("temperature_2m_c", "mean"),
            total_precipitation_mm=("precipitation_mm", "sum"),
            wet_hours=("precipitation_mm", lambda values: int((values >= 0.1).sum())),
            snow_hours=("snowfall_mm_water_equivalent", lambda values: int((values >= 0.1).sum())),
            maximum_wind_ms=("wind_speed_10m_ms", "max"),
            mean_low_cloud_pct=("cloud_cover_low_pct", "mean"),
        )
        .sort_values("airport")
    )


def validate_physical_ranges(weather: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "temperature_c": weather["temperature_2m_c"].between(-90, 60).all(),
        "dew_point_c": weather["dew_point_2m_c"].between(-100, 50).all(),
        "pressure_hpa": weather["pressure_msl_hpa"].between(850, 1100).all(),
        "precipitation_nonnegative": weather["precipitation_mm"].ge(0).all(),
        "snowfall_nonnegative": weather["snowfall_mm_water_equivalent"].ge(0).all(),
        "total_cloud_pct": weather["cloud_cover_pct"].between(0, 100).all(),
        "low_cloud_pct": weather["cloud_cover_low_pct"].between(0, 100).all(),
        "wind_speed_ms": weather["wind_speed_10m_ms"].between(0, 100).all(),
    }
    return {name: bool(value) for name, value in checks.items()}


def create_overview_chart(weather: pd.DataFrame, output_path: Path) -> None:
    plot = weather.loc[weather["airport"] == sorted(weather["airport"].unique())[0]]
    if len(plot) > 24 * 14:
        plot = plot.iloc[: 24 * 14]
    figure, temperature_axis = plt.subplots(figsize=(11, 5))
    temperature_axis.plot(
        plot["weather_hour_utc"],
        plot["temperature_2m_c"],
        color="#2563EB",
        label="Temperature",
    )
    temperature_axis.set_ylabel("Temperature (°C)", color="#2563EB")
    precipitation_axis = temperature_axis.twinx()
    precipitation_axis.bar(
        plot["weather_hour_utc"],
        plot["precipitation_mm"],
        width=0.025,
        color="#D97706",
        alpha=0.55,
        label="Precipitation",
    )
    precipitation_axis.set_ylabel("Hourly precipitation (mm)", color="#D97706")
    temperature_axis.set_title(
        f"Direct ERA5 sample: {plot['airport'].iloc[0]} temperature and precipitation"
    )
    temperature_axis.set_xlabel("UTC hour")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_profile(data_dir: Path, output_dir: Path, scope: str) -> dict[str, object]:
    weather = load_test(data_dir) if scope == "test" else load_full(data_dir)
    output_dir = output_dir / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(output_dir / "weather.parquet", index=False)
    variables = variable_summary(weather)
    airports = airport_summary(weather)
    variables.to_csv(output_dir / "variable_summary.csv", index=False)
    airports.to_csv(output_dir / "airport_summary.csv", index=False)
    create_overview_chart(weather, output_dir / "weather_overview.png")
    checks = validate_physical_ranges(weather)
    summary: dict[str, object] = {
        "scope": scope,
        "rows": len(weather),
        "airports": sorted(weather["airport"].unique()),
        "first_hour_utc": weather["weather_hour_utc"].min().isoformat(),
        "last_hour_utc": weather["weather_hour_utc"].max().isoformat(),
        "duplicate_airport_hours": int(
            weather.duplicated(["airport", "weather_hour_utc"]).sum()
        ),
        "missing_values": int(weather.isna().sum().sum()),
        "physical_range_checks": checks,
        "all_physical_range_checks_pass": all(checks.values()),
        "interpretation": (
            "This is a source and data-quality profile. It does not prove model gain. "
            "Event-hour ERA5 remains outside the final prize model until the organizer "
            "confirms the prediction cut-off."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("test", "full"), default="full")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    print(
        json.dumps(
            build_profile(
                arguments.data_dir.expanduser().resolve(),
                arguments.output_dir.expanduser().resolve(),
                arguments.scope,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
