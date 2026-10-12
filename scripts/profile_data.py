#!/usr/bin/env python3
"""Create a bounded, reproducible data-quality profile for PRC 2026."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq


EXPECTED_TRAINING_FILES = 12
KEY_CANDIDATES = ("MVT_ID_mvt", "FLIGHT_ID_mvt")
TIME_CANDIDATES = (
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_flt",
    "AOBT_3_flt",
)
TARGET = "TAXITIME_SEC_mvt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_manifest(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": metadata.num_rows,
                "columns": metadata.num_columns,
                "row_groups": metadata.num_row_groups,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(records).sort_values("file").reset_index(drop=True)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_frame(frame: pd.DataFrame, artifacts_dir: Path, name: str) -> None:
    frame.to_csv(artifacts_dir / f"{name}.csv", index=False)


def describe_view(connection: duckdb.DuckDBPyConnection, view: str) -> pd.DataFrame:
    return connection.execute(f"DESCRIBE {quote_ident(view)}").df()


def null_profile(
    connection: duckdb.DuckDBPyConnection, view: str, columns: list[str]
) -> pd.DataFrame:
    expressions = [
        f"SUM(CASE WHEN {quote_ident(column)} IS NULL THEN 1 ELSE 0 END) AS {quote_ident(column)}"
        for column in columns
    ]
    counts = connection.execute(
        f"SELECT COUNT(*) AS row_count, {', '.join(expressions)} FROM {quote_ident(view)}"
    ).df()
    row_count = int(counts.loc[0, "row_count"])
    records = []
    for column in columns:
        nulls = int(counts.loc[0, column])
        records.append(
            {
                "dataset": view,
                "column": column,
                "null_rows": nulls,
                "null_rate": nulls / row_count if row_count else None,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["null_rate", "column"], ascending=[False, True]
    )


def leakage_checks(
    connection: duckdb.DuckDBPyConnection, columns: set[str]
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    if {TARGET, "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"}.issubset(columns):
        checks.append(
            {
                "candidate": "MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt",
                "sql_expression": (
                    "date_diff('second', try_cast(BLOCK_TIME_UTC_mvt AS TIMESTAMP), "
                    "try_cast(MVT_TIME_UTC_mvt AS TIMESTAMP))"
                ),
            }
        )
    if {TARGET, "MVT_TIME_UTC_mvt", "AOBT_3_flt"}.issubset(columns):
        checks.append(
            {
                "candidate": "MVT_TIME_UTC_mvt - AOBT_3_flt",
                "sql_expression": (
                    "date_diff('second', try_cast(AOBT_3_flt AS TIMESTAMP), "
                    "try_cast(MVT_TIME_UTC_mvt AS TIMESTAMP))"
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    phase_filter = (
        f" AND {quote_ident('PHASE_mvt')} = 'DEP'" if "PHASE_mvt" in columns else ""
    )
    for check in checks:
        expression = check["sql_expression"]
        result = connection.execute(
            f"""
            SELECT
              COUNT(*) FILTER (
                WHERE {quote_ident(TARGET)} IS NOT NULL AND ({expression}) IS NOT NULL
              ) AS comparable_rows,
              AVG(ABS(CAST({quote_ident(TARGET)} AS DOUBLE) - ({expression}))) AS mean_abs_error_sec,
              MAX(ABS(CAST({quote_ident(TARGET)} AS DOUBLE) - ({expression}))) AS max_abs_error_sec,
              AVG(
                CASE WHEN CAST({quote_ident(TARGET)} AS DOUBLE) = ({expression})
                THEN 1.0 ELSE 0.0 END
              ) FILTER (
                WHERE {quote_ident(TARGET)} IS NOT NULL AND ({expression}) IS NOT NULL
              ) AS exact_match_rate
            FROM training
            WHERE 1 = 1 {phase_filter}
            """
        ).fetchone()
        rows.append(
            {
                "candidate": check["candidate"],
                "comparable_rows": result[0],
                "mean_abs_error_sec": result[1],
                "max_abs_error_sec": result[2],
                "exact_match_rate": result[3],
                "status": "QUARANTINE_FROM_FEATURES",
            }
        )
    return pd.DataFrame(rows)


def build_profile(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    training_files = sorted(data_dir.glob("training_*.parquet"))
    ranking_file = data_dir / "ranking.parquet"
    submitting_file = data_dir / "submitting.parquet"
    all_files = training_files + [ranking_file, submitting_file]

    if len(training_files) != EXPECTED_TRAINING_FILES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_FILES} training files, found {len(training_files)}"
        )
    missing = [str(path) for path in all_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected files: {missing}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest = parquet_manifest(all_files)
    write_frame(manifest, artifacts_dir, "file_manifest")

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    training_glob = str(data_dir / "training_*.parquet")
    connection.execute(
        "CREATE VIEW training AS SELECT * FROM read_parquet("
        f"{quote_literal(training_glob)}, union_by_name=true, filename=true)"
    )
    connection.execute(
        "CREATE VIEW ranking AS SELECT * FROM read_parquet("
        f"{quote_literal(str(ranking_file))}, union_by_name=true, filename=true)"
    )
    connection.execute(
        "CREATE VIEW submitting AS SELECT * FROM read_parquet("
        f"{quote_literal(str(submitting_file))}, union_by_name=true, filename=true)"
    )

    schemas: dict[str, pd.DataFrame] = {}
    nulls: list[pd.DataFrame] = []
    row_counts: list[dict[str, Any]] = []
    for view in ("training", "ranking", "submitting"):
        schema = describe_view(connection, view)
        schemas[view] = schema
        write_frame(schema, artifacts_dir, f"schema_{view}")
        columns = schema["column_name"].tolist()
        nulls.append(null_profile(connection, view, columns))
        row_counts.append(
            {
                "dataset": view,
                "rows": connection.execute(
                    f"SELECT COUNT(*) FROM {quote_ident(view)}"
                ).fetchone()[0],
                "columns": len(columns),
            }
        )

    row_counts_frame = pd.DataFrame(row_counts)
    nulls_frame = pd.concat(nulls, ignore_index=True)
    write_frame(row_counts_frame, artifacts_dir, "dataset_shapes")
    write_frame(nulls_frame, artifacts_dir, "null_profile")

    training_columns = set(schemas["training"]["column_name"])
    duplicate_rows: list[dict[str, Any]] = []
    for key in KEY_CANDIDATES:
        if key not in training_columns:
            continue
        total, non_null, distinct_count = connection.execute(
            f"""
            SELECT COUNT(*), COUNT({quote_ident(key)}), COUNT(DISTINCT {quote_ident(key)})
            FROM training
            """
        ).fetchone()
        duplicate_rows.append(
            {
                "key": key,
                "rows": total,
                "non_null_rows": non_null,
                "distinct_non_null": distinct_count,
                "duplicate_non_null_rows": non_null - distinct_count,
            }
        )
    duplicates = pd.DataFrame(duplicate_rows)
    write_frame(duplicates, artifacts_dir, "key_checks")

    time_rows: list[dict[str, Any]] = []
    for column in TIME_CANDIDATES:
        if column not in training_columns:
            continue
        minimum, maximum, parseable = connection.execute(
            f"""
            SELECT
              MIN(try_cast({quote_ident(column)} AS TIMESTAMP)),
              MAX(try_cast({quote_ident(column)} AS TIMESTAMP)),
              COUNT(try_cast({quote_ident(column)} AS TIMESTAMP))
            FROM training
            """
        ).fetchone()
        time_rows.append(
            {
                "column": column,
                "min_timestamp": minimum,
                "max_timestamp": maximum,
                "parseable_rows": parseable,
            }
        )
    time_coverage = pd.DataFrame(time_rows)
    write_frame(time_coverage, artifacts_dir, "time_coverage")

    group_frames: list[pd.DataFrame] = []
    for column in ("PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_RULE_mvt"):
        if column not in training_columns:
            continue
        frame = connection.execute(
            f"""
            SELECT {quote_ident(column)} AS value, COUNT(*) AS rows
            FROM training
            GROUP BY 1
            ORDER BY rows DESC, value
            LIMIT 100
            """
        ).df()
        frame.insert(0, "column", column)
        group_frames.append(frame)
    categories = (
        pd.concat(group_frames, ignore_index=True)
        if group_frames
        else pd.DataFrame(columns=["column", "value", "rows"])
    )
    write_frame(categories, artifacts_dir, "major_categories")

    target_profile = pd.DataFrame()
    if TARGET in training_columns:
        phase_expression = (
            quote_ident("PHASE_mvt") if "PHASE_mvt" in training_columns else "'ALL'"
        )
        target_profile = connection.execute(
            f"""
            SELECT
              CAST({phase_expression} AS VARCHAR) AS phase,
              COUNT(*) AS rows,
              COUNT({quote_ident(TARGET)}) AS non_null_rows,
              AVG(CAST({quote_ident(TARGET)} AS DOUBLE)) AS mean_sec,
              STDDEV_POP(CAST({quote_ident(TARGET)} AS DOUBLE)) AS stddev_sec,
              MIN(CAST({quote_ident(TARGET)} AS DOUBLE)) AS min_sec,
              quantile_cont(CAST({quote_ident(TARGET)} AS DOUBLE), 0.01) AS p01_sec,
              quantile_cont(CAST({quote_ident(TARGET)} AS DOUBLE), 0.50) AS p50_sec,
              quantile_cont(CAST({quote_ident(TARGET)} AS DOUBLE), 0.99) AS p99_sec,
              MAX(CAST({quote_ident(TARGET)} AS DOUBLE)) AS max_sec
            FROM training
            GROUP BY 1
            UNION ALL
            SELECT
              'ALL' AS phase,
              COUNT(*) AS rows,
              COUNT({quote_ident(TARGET)}) AS non_null_rows,
              AVG(CAST({quote_ident(TARGET)} AS DOUBLE)) AS mean_sec,
              STDDEV_POP(CAST({quote_ident(TARGET)} AS DOUBLE)) AS stddev_sec,
              MIN(CAST({quote_ident(TARGET)} AS DOUBLE)) AS min_sec,
              quantile_cont(CAST({quote_ident(TARGET)} AS DOUBLE), 0.01) AS p01_sec,
              quantile_cont(CAST({quote_ident(TARGET)} AS DOUBLE), 0.50) AS p50_sec,
              quantile_cont(CAST({quote_ident(TARGET)} AS DOUBLE), 0.99) AS p99_sec,
              MAX(CAST({quote_ident(TARGET)} AS DOUBLE)) AS max_sec
            FROM training
            """
        ).df()
        write_frame(target_profile, artifacts_dir, "target_profile")

    leakage = leakage_checks(connection, training_columns)
    write_frame(leakage, artifacts_dir, "leakage_checks")

    result = {
        "compressed_bytes": int(manifest["bytes"].sum()),
        "file_count": int(len(manifest)),
        "training_rows": int(
            row_counts_frame.loc[row_counts_frame["dataset"] == "training", "rows"].iloc[0]
        ),
        "ranking_rows": int(
            row_counts_frame.loc[row_counts_frame["dataset"] == "ranking", "rows"].iloc[0]
        ),
        "submitting_rows": int(
            row_counts_frame.loc[row_counts_frame["dataset"] == "submitting", "rows"].iloc[0]
        ),
        "leakage_candidates_quarantined": leakage["candidate"].tolist()
        if not leakage.empty
        else [],
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/eda")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_profile(args.data_dir.resolve(), args.artifacts_dir.resolve())
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
