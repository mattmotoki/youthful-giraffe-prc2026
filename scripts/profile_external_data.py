#!/usr/bin/env python3
"""Profile historical taxi-out and ERA5 weather data against PRC 2026."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AIRPORTS = (
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LTAI",
    "LTFM",
    "LSZH",
)
WEATHER_RENAME = {
    "time": "weather_hour_utc",
    "temperature_2m (°C)": "temperature_2m_c",
    "relative_humidity_2m (%)": "relative_humidity_2m_pct",
    "dew_point_2m (°C)": "dew_point_2m_c",
    "precipitation (mm)": "precipitation_mm",
    "rain (mm)": "rain_mm",
    "snowfall (cm)": "snowfall_cm",
    "pressure_msl (hPa)": "pressure_msl_hpa",
    "cloud_cover (%)": "cloud_cover_pct",
    "cloud_cover_low (%)": "cloud_cover_low_pct",
    "wind_speed_10m (m/s)": "wind_speed_10m_ms",
    "wind_direction_10m (°)": "wind_direction_10m_deg",
}
EXPECTED_WEATHER_ROWS = 17_544
EXPECTED_WEATHER_FIRST_HOUR = pd.Timestamp("2024-01-01 00:00:00")
EXPECTED_WEATHER_LAST_HOUR = pd.Timestamp("2025-12-31 23:00:00")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_frame(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    frame.to_csv(output_dir / f"{name}.csv", index=False)


def quote_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def verify_raw_manifest(data_dir: Path) -> None:
    manifest_path = ROOT / "references" / "external-data-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files", [])
    if len(records) != 12:
        raise ValueError(f"Expected 12 raw manifest records, found {len(records)}")
    for record in records:
        path = data_dir / record["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Raw file size does not match tracked manifest: {path}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Raw file SHA-256 does not match tracked manifest: {path}")


def load_weather(weather_dir: Path) -> pd.DataFrame:
    frames = []
    expected_paths = {
        weather_dir / f"{airport}_era5_2024-01-01_2025-12-31.csv": airport
        for airport in AIRPORTS
    }
    actual_paths = set(weather_dir.glob("*_era5_*.csv"))
    if actual_paths != set(expected_paths):
        missing = sorted(path.name for path in set(expected_paths) - actual_paths)
        unexpected = sorted(path.name for path in actual_paths - set(expected_paths))
        raise ValueError(
            f"Weather file identity mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for path, airport in sorted(expected_paths.items()):
        frame = pd.read_csv(path, skiprows=3).rename(columns=WEATHER_RENAME)
        if set(WEATHER_RENAME.values()) - set(frame.columns):
            raise ValueError(f"Weather schema mismatch in {path}")
        frame.insert(0, "airport", airport)
        frame["weather_hour_utc"] = pd.to_datetime(
            frame["weather_hour_utc"], utc=True
        ).dt.tz_localize(None)
        if (
            len(frame) != EXPECTED_WEATHER_ROWS
            or frame["weather_hour_utc"].isna().any()
            or frame["weather_hour_utc"].duplicated().any()
            or frame["weather_hour_utc"].iloc[0] != EXPECTED_WEATHER_FIRST_HOUR
            or frame["weather_hour_utc"].iloc[-1] != EXPECTED_WEATHER_LAST_HOUR
            or not frame["weather_hour_utc"].is_monotonic_increasing
            or not frame["weather_hour_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
        ):
            raise ValueError(f"Weather time coverage mismatch in {path}")
        frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    if weather.duplicated(["airport", "weather_hour_utc"]).any():
        raise ValueError("Duplicate airport-hour weather keys")
    return weather


def create_charts(
    airport_shift: pd.DataFrame,
    weather_effects: pd.DataFrame,
    weather_coverage: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    blue = "#2563EB"
    gold = "#D97706"
    ink = "#1F2937"

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        airport_shift["median_2022_sec"],
        airport_shift["median_2025_sec"],
        color=blue,
        edgecolor=ink,
        linewidth=0.6,
        s=70,
    )
    low = min(airport_shift["median_2022_sec"].min(), airport_shift["median_2025_sec"].min())
    high = max(airport_shift["median_2022_sec"].max(), airport_shift["median_2025_sec"].max())
    axis.plot([low, high], [low, high], color=ink, linestyle="--", linewidth=1)
    for row in airport_shift.itertuples(index=False):
        axis.annotate(row.airport, (row.median_2022_sec, row.median_2025_sec), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(
        title="Historical airport median taxi-out time",
        xlabel="PRC 2024 open data: 2022 median (seconds)",
        ylabel="PRC 2026 training data: 2025 median (seconds)",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "historical_airport_medians.png", dpi=160)
    plt.close(figure)

    effect_plot = weather_effects.sort_values("wet_minus_dry_residual_sec")
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = [gold if value >= 0 else blue for value in effect_plot["wet_minus_dry_residual_sec"]]
    axis.barh(effect_plot["airport"], effect_plot["wet_minus_dry_residual_sec"], color=colors)
    axis.axvline(0, color=ink, linewidth=1)
    axis.set(
        title="Wet-hour taxi-out residual by airport",
        xlabel="Wet minus dry mean residual (seconds); 2025 departures",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "weather_wet_dry_effect.png", dpi=160)
    plt.close(figure)

    coverage_plot = weather_coverage.loc[weather_coverage["departures"] > 0].sort_values("match_rate")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(coverage_plot["airport"], coverage_plot["match_rate"] * 100, color=blue)
    axis.set_xlim(0, 100)
    axis.set(
        title="ERA5 join coverage",
        xlabel="Departures matched to airport-hour weather (%)",
        ylabel="Airport",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "weather_join_coverage.png", dpi=160)
    plt.close(figure)


def build_profile(data_dir: Path, official_data_dir: Path, output_dir: Path) -> dict[str, Any]:
    flight_list = data_dir / "prc2024" / "flight_list.csv"
    weather_dir = data_dir / "weather" / "era5"
    training_glob = official_data_dir / "training_*.parquet"
    if not flight_list.is_file():
        raise FileNotFoundError(flight_list)
    training_files = sorted(official_data_dir.glob("training_*.parquet"))
    if len(training_files) != 12:
        raise ValueError(f"Expected 12 training files, found {len(training_files)}")
    ranking_file = official_data_dir / "ranking.parquet"
    if not ranking_file.is_file():
        raise FileNotFoundError(ranking_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    verify_raw_manifest(data_dir)
    weather = load_weather(weather_dir)
    weather_parquet = data_dir / "weather" / "era5_airports_2024_2025.parquet"
    weather.to_parquet(weather_parquet, index=False)

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    connection.execute(
        "CREATE VIEW training AS SELECT * FROM read_parquet("
        f"{quote_literal(training_glob)}, union_by_name=true)"
    )
    connection.execute(
        "CREATE VIEW history AS SELECT * FROM read_csv_auto("
        f"{quote_literal(flight_list)}, header=true, sample_size=-1, "
        "quote=chr(34), escape=chr(34))"
    )
    connection.execute(
        "CREATE VIEW weather AS SELECT * FROM read_parquet("
        f"{quote_literal(weather_parquet)})"
    )
    connection.execute(
        "CREATE VIEW ranking AS SELECT * FROM read_parquet("
        f"{quote_literal(ranking_file)})"
    )
    connection.register("expected_airports", pd.DataFrame({"airport": AIRPORTS}))
    airport_list = ",".join(f"'{airport}'" for airport in AIRPORTS)

    historical_coverage = connection.execute(
        f"""
        SELECT
          adep AS airport,
          COUNT(*) AS rows_2022,
          COUNT(taxiout_time) AS target_rows_2022,
          median(taxiout_time) * 60 AS median_2022_sec,
          AVG(taxiout_time) * 60 AS mean_2022_sec,
          quantile_cont(taxiout_time, 0.90) * 60 AS p90_2022_sec
        FROM history
        WHERE adep IN ({airport_list})
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    current_profile = connection.execute(
        f"""
        SELECT
          ADEP_mvt AS airport,
          COUNT(*) AS rows_2025,
          COUNT(TAXITIME_SEC_mvt) AS target_rows_2025,
          median(TAXITIME_SEC_mvt) AS median_2025_sec,
          AVG(TAXITIME_SEC_mvt) AS mean_2025_sec,
          quantile_cont(TAXITIME_SEC_mvt, 0.90) AS p90_2025_sec
        FROM training
        WHERE PHASE_mvt = 'DEP' AND ADEP_mvt IN ({airport_list})
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    airport_shift = historical_coverage.merge(current_profile, on="airport", how="outer")
    airport_shift["median_shift_sec"] = (
        airport_shift["median_2025_sec"] - airport_shift["median_2022_sec"]
    )
    write_frame(airport_shift, output_dir, "historical_airport_shift")

    airport_dataset_coverage = connection.execute(
        """
        WITH training_counts AS (
          SELECT ADEP_mvt AS airport, COUNT(*) AS training_departures
          FROM training
          WHERE PHASE_mvt = 'DEP'
          GROUP BY 1
        ), ranking_counts AS (
          SELECT ADEP_mvt AS airport, COUNT(*) AS ranking_departures
          FROM ranking
          WHERE PHASE_mvt = 'DEP'
          GROUP BY 1
        )
        SELECT
          expected_airports.airport,
          COALESCE(training_counts.training_departures, 0) AS training_departures,
          COALESCE(ranking_counts.ranking_departures, 0) AS ranking_departures
        FROM expected_airports
        LEFT JOIN training_counts USING (airport)
        LEFT JOIN ranking_counts USING (airport)
        ORDER BY 1
        """
    ).df()
    write_frame(airport_dataset_coverage, output_dir, "airport_dataset_coverage")

    aircraft_overlap = connection.execute(
        f"""
        WITH old_types AS (
          SELECT DISTINCT aircraft_type
          FROM history
          WHERE adep IN ({airport_list}) AND aircraft_type IS NOT NULL
        ), current_rows AS (
          SELECT AIRCRAFT_TYPE_mvt AS aircraft_type, COUNT(*) AS rows
          FROM training
          WHERE PHASE_mvt = 'DEP' AND ADEP_mvt IN ({airport_list})
          GROUP BY 1
        )
        SELECT
          COUNT(*) AS current_types,
          COUNT(*) FILTER (WHERE old_types.aircraft_type IS NOT NULL) AS overlapping_types,
          SUM(current_rows.rows) AS current_rows,
          SUM(current_rows.rows) FILTER (WHERE old_types.aircraft_type IS NOT NULL) AS rows_with_historical_type,
          SUM(current_rows.rows) FILTER (WHERE old_types.aircraft_type IS NOT NULL) / SUM(current_rows.rows) AS row_coverage
        FROM current_rows
        LEFT JOIN old_types USING (aircraft_type)
        """
    ).df()
    write_frame(aircraft_overlap, output_dir, "historical_aircraft_overlap")

    connection.execute(
        """
        CREATE VIEW departure_weather AS
        SELECT
          t.ADEP_mvt AS airport,
          t.TAXITIME_SEC_mvt AS target_sec,
          date_trunc('hour', try_cast(t.SCHED_TIME_UTC_mvt AS TIMESTAMP)) AS scheduled_hour,
          w.* EXCLUDE (airport, weather_hour_utc)
        FROM training t
        LEFT JOIN weather w
          ON t.ADEP_mvt = w.airport
         AND date_trunc('hour', try_cast(t.SCHED_TIME_UTC_mvt AS TIMESTAMP)) = w.weather_hour_utc
        WHERE t.PHASE_mvt = 'DEP'
        """
    )
    weather_coverage = connection.execute(
        """
        WITH counts AS (
          SELECT
            airport,
            COUNT(*) AS departures,
            COUNT(temperature_2m_c) AS matched_departures
          FROM departure_weather
          GROUP BY 1
        )
        SELECT
          expected_airports.airport,
          COALESCE(counts.departures, 0) AS departures,
          COALESCE(counts.matched_departures, 0) AS matched_departures,
          CASE
            WHEN counts.departures > 0
            THEN counts.matched_departures / counts.departures
          END AS match_rate,
          CASE
            WHEN counts.departures IS NULL THEN 'NO_2025_DEPARTURES'
            WHEN counts.matched_departures = counts.departures THEN 'COMPLETE'
            ELSE 'INCOMPLETE'
          END AS status
        FROM expected_airports
        LEFT JOIN counts USING (airport)
        ORDER BY 1
        """
    ).df()
    write_frame(weather_coverage, output_dir, "weather_join_coverage")

    target_quality = connection.execute(
        """
        SELECT
          COUNT(*) AS departure_rows,
          COUNT(target_sec) AS target_rows,
          COUNT(*) FILTER (WHERE target_sec BETWEEN 0 AND 7200) AS plausible_target_rows,
          COUNT(*) FILTER (WHERE target_sec < 0) AS negative_target_rows,
          COUNT(*) FILTER (WHERE target_sec > 7200) AS over_two_hour_target_rows
        FROM departure_weather
        """
    ).df()
    write_frame(target_quality, output_dir, "weather_target_quality")

    weather_effects = connection.execute(
        """
        WITH residuals AS (
          SELECT
            *,
            target_sec - AVG(target_sec) OVER (
              PARTITION BY airport, month(scheduled_hour), hour(scheduled_hour)
            ) AS residual_sec
          FROM departure_weather
          WHERE target_sec BETWEEN 0 AND 7200 AND temperature_2m_c IS NOT NULL
        )
        SELECT
          airport,
          COUNT(*) AS rows,
          COUNT(*) FILTER (WHERE precipitation_mm > 0) AS wet_rows,
          AVG(residual_sec) FILTER (WHERE precipitation_mm > 0) AS wet_mean_residual_sec,
          AVG(residual_sec) FILTER (WHERE precipitation_mm = 0) AS dry_mean_residual_sec,
          AVG(residual_sec) FILTER (WHERE precipitation_mm > 0)
            - AVG(residual_sec) FILTER (WHERE precipitation_mm = 0)
            AS wet_minus_dry_residual_sec,
          corr(residual_sec, wind_speed_10m_ms) AS wind_residual_correlation,
          corr(residual_sec, cloud_cover_low_pct) AS low_cloud_residual_correlation,
          corr(residual_sec, pressure_msl_hpa) AS pressure_residual_correlation
        FROM residuals
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    write_frame(weather_effects, output_dir, "weather_effects")

    source_manifest = pd.DataFrame(
        [
            {
                "file": str(path.relative_to(data_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [flight_list, weather_parquet, *sorted(weather_dir.glob("*.csv"))]
        ]
    )
    write_frame(source_manifest, output_dir, "external_file_manifest")

    trajectory_files = [
        path
        for path in sorted((data_dir / "prc2024" / "trajectories").glob("*.parquet"))
        if path.stat().st_size > 10_000
    ][:1]
    if trajectory_files:
        trajectory_path = trajectory_files[0]
        connection.execute(
            "CREATE VIEW trajectory_sample AS SELECT * FROM read_parquet("
            f"{quote_literal(trajectory_path)})"
        )
        trajectory_schema = connection.execute("DESCRIBE trajectory_sample").df()
        write_frame(trajectory_schema, output_dir, "trajectory_sample_schema")
        trajectory_summary = connection.execute(
            f"""
            SELECT
              {quote_literal(trajectory_path.name)} AS file,
              COUNT(*) AS position_rows,
              COUNT(DISTINCT t.flight_id) AS flights,
              COUNT(*) FILTER (WHERE h.adep IN ({airport_list})) AS challenge_airport_position_rows,
              COUNT(DISTINCT t.flight_id) FILTER (WHERE h.adep IN ({airport_list})) AS challenge_airport_flights
            FROM trajectory_sample t
            LEFT JOIN history h USING (flight_id)
            """
        ).df()
        write_frame(trajectory_summary, output_dir, "trajectory_sample_summary")
    else:
        write_frame(
            pd.DataFrame(
                [{
                    "status": "No complete non-empty trajectory file was present when EDA ran.",
                    "next_step": "Rerun after the background trajectory download completes a sample file.",
                }]
            ),
            output_dir,
            "trajectory_sample_summary",
        )

    create_charts(airport_shift.dropna(), weather_effects.dropna(), weather_coverage, output_dir)

    correlations = {
        "historical_airport_median_correlation": float(
            airport_shift[["median_2022_sec", "median_2025_sec"]].corr().iloc[0, 1]
        ),
        "weather_join_rate": float(
            weather_coverage["matched_departures"].sum()
            / weather_coverage["departures"].sum()
        ),
        "wet_minus_dry_residual_range_sec": [
            float(weather_effects["wet_minus_dry_residual_sec"].min()),
            float(weather_effects["wet_minus_dry_residual_sec"].max()),
        ],
        "maximum_absolute_wind_residual_correlation": float(
            weather_effects["wind_residual_correlation"].abs().max()
        ),
    }
    summary = {
        "as_of": "2026-09-02",
        "historical_flight_rows_all": connection.execute(
            "SELECT COUNT(*) FROM history"
        ).fetchone()[0],
        "historical_departures_at_challenge_airports": int(
            airport_shift["rows_2022"].sum()
        ),
        "historical_airports_covered": int(airport_shift["rows_2022"].notna().sum()),
        "current_departure_airports": int((weather_coverage["departures"] > 0).sum()),
        "missing_current_departure_airports": weather_coverage.loc[
            weather_coverage["departures"] == 0, "airport"
        ].tolist(),
        "ranking_departure_airports": int(
            (airport_dataset_coverage["ranking_departures"] > 0).sum()
        ),
        "missing_ranking_departure_airports": airport_dataset_coverage.loc[
            airport_dataset_coverage["ranking_departures"] == 0, "airport"
        ].tolist(),
        "historical_taxiout_resolution_sec": 60,
        "aircraft_type_row_coverage": float(aircraft_overlap.loc[0, "row_coverage"]),
        "plausible_weather_target_rows": int(target_quality.loc[0, "plausible_target_rows"]),
        **correlations,
        "interpretation": (
            "Historical aggregates require forward-only validation. Weather is "
            "EDA-only until the Open-Meteo service terms and the competition "
            "prediction cut-off are resolved."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    connection.close()
    return summary


def main() -> None:
    data_dir = Path(os.getenv("PRC_EXTERNAL_DATA_DIR", ROOT / "data" / "external"))
    official_data_dir = Path(os.getenv("PRC_DATA_DIR", ROOT / "data" / "raw"))
    output_dir = Path(
        os.getenv("PRC_EXTERNAL_ARTIFACT_DIR", ROOT / "artifacts" / "eda" / "external")
    )
    print(json.dumps(build_profile(data_dir, official_data_dir, output_dir), indent=2))


if __name__ == "__main__":
    main()
