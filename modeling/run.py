#!/usr/bin/env python3
"""Run the first PRC 2026 baselines and build the review dashboard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "modeling" / "results"
DASHBOARD_PATH = ROOT / "modeling" / "dashboard.html"
TARGET = "TAXITIME_SEC_mvt"
SEED = 20260902

SCHEDULE_COLUMNS = (
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "SCHED_TIME_UTC_mvt",
    "RUNWAY_mvt",
)
DEPARTURE_COLUMNS = (
    "MVT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    TARGET,
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_RULE_mvt",
)
CATEGORICAL_FEATURES = (
    "airport",
    "destination",
    "filed_destination",
    "aircraft_type",
    "runway",
    "stand",
    "market_segment",
    "flight_type",
    "wake_category",
    "operator",
    "flight_rule",
)
NUMERIC_FEATURES = (
    "scheduled_month",
    "scheduled_week",
    "scheduled_day_of_year",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_quarter",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "lobt_minus_schedule_min",
    "iobt_minus_schedule_min",
    "eobt_minus_schedule_min",
    "eobt_minus_iobt_min",
    "lobt_missing",
    "iobt_missing",
    "eobt_missing",
    "scheduled_departures_15m",
    "scheduled_arrivals_15m",
    "scheduled_departures_75m",
    "scheduled_arrivals_75m",
    "scheduled_runway_departures_60m",
)
BASELINE_LEVELS = (
    (("airport", "scheduled_weekday", "scheduled_hour", "scheduled_quarter"), 30),
    (("airport", "scheduled_weekday", "scheduled_hour"), 100),
    (("airport", "scheduled_hour"), 250),
    (("airport",), 1),
)


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


FOLDS = (
    Fold("may", "2025-01-01", "2025-05-01", "2025-05-01", "2025-06-01"),
    Fold("july", "2025-01-01", "2025-07-01", "2025-07-01", "2025-08-01"),
    Fold("december", "2025-01-01", "2025-12-01", "2025-12-01", "2026-01-01"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def training_files() -> list[Path]:
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    if len(files) != 12:
        raise ValueError(f"Expected 12 training files, found {len(files)}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(packages: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_schedule_counts(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow_parts = []
    runway_parts = []
    for path in files:
        frame = pd.read_parquet(path, columns=list(SCHEDULE_COLUMNS))
        frame["scheduled_time"] = pd.to_datetime(
            frame["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce"
        )
        frame = frame.loc[
            frame["scheduled_time"].notna()
            & frame["PHASE_mvt"].isin(["DEP", "ARR"])
        ].copy()
        frame["airport"] = np.where(
            frame["PHASE_mvt"].eq("DEP"),
            frame["ADEP_mvt"],
            frame["ADES_mvt"],
        )
        frame["scheduled_bin_15m"] = frame["scheduled_time"].dt.floor("15min")
        flow_parts.append(
            frame.groupby(
                ["airport", "scheduled_bin_15m", "PHASE_mvt"],
                observed=True,
            )
            .size()
            .rename("rows")
            .reset_index()
        )
        departures = frame.loc[frame["PHASE_mvt"].eq("DEP")].copy()
        departures["runway"] = departures["RUNWAY_mvt"].fillna("__MISSING__").astype(str)
        departures["scheduled_hour_utc"] = departures["scheduled_time"].dt.floor("h")
        runway_parts.append(
            departures.groupby(
                ["airport", "runway", "scheduled_hour_utc"],
                observed=True,
            )
            .size()
            .rename("scheduled_runway_departures_60m")
            .reset_index()
        )
        del frame, departures

    flow = (
        pd.concat(flow_parts, ignore_index=True)
        .groupby(["airport", "scheduled_bin_15m", "PHASE_mvt"], observed=True)["rows"]
        .sum()
        .unstack(fill_value=0)
    )
    for phase in ("DEP", "ARR"):
        if phase not in flow:
            flow[phase] = 0
    dense_parts = []
    for airport, airport_flow in flow.groupby(level="airport", observed=True):
        local = airport_flow.droplevel("airport").sort_index()
        full_index = pd.date_range(local.index.min(), local.index.max(), freq="15min", tz="UTC")
        local = local.reindex(full_index, fill_value=0)
        local.index.name = "scheduled_bin_15m"
        local["scheduled_departures_15m"] = local["DEP"].astype(np.int32)
        local["scheduled_arrivals_15m"] = local["ARR"].astype(np.int32)
        local["scheduled_departures_75m"] = (
            local["DEP"].rolling(5, center=True, min_periods=1).sum().astype(np.int32)
        )
        local["scheduled_arrivals_75m"] = (
            local["ARR"].rolling(5, center=True, min_periods=1).sum().astype(np.int32)
        )
        local = local.drop(columns=["DEP", "ARR"]).reset_index()
        local.insert(0, "airport", airport)
        dense_parts.append(local)
    flow_features = pd.concat(dense_parts, ignore_index=True)
    runway_features = (
        pd.concat(runway_parts, ignore_index=True)
        .groupby(["airport", "runway", "scheduled_hour_utc"], observed=True)[
            "scheduled_runway_departures_60m"
        ]
        .sum()
        .reset_index()
    )
    return flow_features, runway_features


def load_departures(files: list[Path]) -> pd.DataFrame:
    parts = []
    for path in files:
        frame = pd.read_parquet(
            path,
            columns=list(DEPARTURE_COLUMNS),
            filters=[("PHASE_mvt", "==", "DEP")],
        )
        parts.append(frame.loc[frame["PHASE_mvt"].eq("DEP")])
    departures = pd.concat(parts, ignore_index=True)
    departures["scheduled_time"] = pd.to_datetime(
        departures["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce"
    )
    departures = departures.loc[
        departures["scheduled_time"].notna()
        & departures[TARGET].notna()
    ].copy()
    return departures


def add_time_delta(
    frame: pd.DataFrame,
    source: str,
    output: str,
    reference: pd.Series,
) -> None:
    values = pd.to_datetime(frame[source], utc=True, errors="coerce")
    frame[output] = (
        (values - reference).dt.total_seconds().div(60).clip(-720, 720)
    )
    frame[output + "_missing"] = values.isna().astype(np.int8)
    frame[output] = frame[output].fillna(0).astype(np.float32)


def build_features(
    departures: pd.DataFrame,
    flow_features: pd.DataFrame,
    runway_features: pd.DataFrame,
) -> pd.DataFrame:
    frame = departures.copy()
    frame["airport"] = frame["ADEP_mvt"].fillna("__MISSING__").astype(str)
    frame["destination"] = frame["ADES_mvt"].fillna("__MISSING__").astype(str)
    frame["filed_destination"] = frame["ADES_FILED_flt"].fillna("__MISSING__").astype(str)
    frame["aircraft_type"] = frame["AIRCRAFT_TYPE_mvt"].fillna("__MISSING__").astype(str)
    frame["runway"] = frame["RUNWAY_mvt"].fillna("__MISSING__").astype(str)
    frame["stand"] = frame["STAND_mvt"].fillna("__MISSING__").astype(str)
    frame["market_segment"] = frame["MARKET_SEGMENT_flt"].fillna("__MISSING__").astype(str)
    frame["flight_type"] = frame["FLIGHT_TYPE_flt"].fillna("__MISSING__").astype(str)
    frame["wake_category"] = frame["WK_TBL_CAT_flt"].fillna("__MISSING__").astype(str)
    frame["operator"] = frame["AIRCRAFT_OPERATOR_flt"].fillna("__MISSING__").astype(str)
    frame["flight_rule"] = frame["FLIGHT_RULE_mvt"].fillna("__MISSING__").astype(str)

    scheduled = frame["scheduled_time"]
    frame["date"] = scheduled.dt.strftime("%Y-%m-%d")
    frame["scheduled_month"] = scheduled.dt.month.astype(np.int8)
    frame["scheduled_week"] = scheduled.dt.isocalendar().week.astype(np.int16)
    frame["scheduled_day_of_year"] = scheduled.dt.dayofyear.astype(np.int16)
    frame["scheduled_weekday"] = scheduled.dt.weekday.astype(np.int8)
    frame["scheduled_hour"] = scheduled.dt.hour.astype(np.int8)
    frame["scheduled_quarter"] = (scheduled.dt.minute // 15).astype(np.int8)
    frame["hour_sin"] = np.sin(2 * math.pi * (scheduled.dt.hour + scheduled.dt.minute / 60) / 24).astype(np.float32)
    frame["hour_cos"] = np.cos(2 * math.pi * (scheduled.dt.hour + scheduled.dt.minute / 60) / 24).astype(np.float32)
    frame["weekday_sin"] = np.sin(2 * math.pi * frame["scheduled_weekday"] / 7).astype(np.float32)
    frame["weekday_cos"] = np.cos(2 * math.pi * frame["scheduled_weekday"] / 7).astype(np.float32)

    add_time_delta(frame, "LOBT_flt", "lobt_minus_schedule_min", scheduled)
    add_time_delta(frame, "IOBT_flt", "iobt_minus_schedule_min", scheduled)
    add_time_delta(frame, "EOBT_1_flt", "eobt_minus_schedule_min", scheduled)
    iobt = pd.to_datetime(frame["IOBT_flt"], utc=True, errors="coerce")
    eobt = pd.to_datetime(frame["EOBT_1_flt"], utc=True, errors="coerce")
    frame["eobt_minus_iobt_min"] = (
        (eobt - iobt).dt.total_seconds().div(60).clip(-720, 720).fillna(0).astype(np.float32)
    )
    frame.rename(
        columns={
            "lobt_minus_schedule_min_missing": "lobt_missing",
            "iobt_minus_schedule_min_missing": "iobt_missing",
            "eobt_minus_schedule_min_missing": "eobt_missing",
        },
        inplace=True,
    )
    frame["scheduled_bin_15m"] = scheduled.dt.floor("15min")
    frame["scheduled_hour_utc"] = scheduled.dt.floor("h")
    frame = frame.merge(
        flow_features,
        on=["airport", "scheduled_bin_15m"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        runway_features,
        on=["airport", "runway", "scheduled_hour_utc"],
        how="left",
        validate="many_to_one",
    )
    count_columns = [
        "scheduled_departures_15m",
        "scheduled_arrivals_15m",
        "scheduled_departures_75m",
        "scheduled_arrivals_75m",
        "scheduled_runway_departures_60m",
    ]
    frame[count_columns] = frame[count_columns].fillna(0).astype(np.float32)
    return frame


def metric_row(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "rows": int(len(actual)),
        "rmse_sec": float(np.sqrt(np.mean(error**2))),
        "mae_sec": float(np.mean(absolute)),
        "p90_abs_error_sec": float(np.quantile(absolute, 0.90)),
        "bias_sec": float(np.mean(error)),
    }


def hierarchical_median_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    prediction = np.full(len(valid), np.nan, dtype=np.float64)
    for keys, minimum_rows in BASELINE_LEVELS:
        statistics = (
            train.groupby(list(keys), observed=True)[TARGET]
            .agg([("median", "median"), ("rows", "size")])
            .loc[lambda value: value["rows"] >= minimum_rows, "median"]
        )
        if len(keys) == 1:
            candidate = valid[keys[0]].map(statistics).to_numpy(dtype=float)
        else:
            index = pd.MultiIndex.from_frame(valid[list(keys)])
            candidate = statistics.reindex(index).to_numpy(dtype=float)
        missing = np.isnan(prediction) & ~np.isnan(candidate)
        prediction[missing] = candidate[missing]
    prediction[np.isnan(prediction)] = float(train[TARGET].median())
    return prediction


def select_rows(
    frame: pd.DataFrame,
    start: str,
    end: str,
    maximum_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")
    selected = frame.loc[frame["scheduled_time"].between(start_time, end_time, inclusive="left")]
    if maximum_rows is not None and len(selected) > maximum_rows:
        selected = selected.sample(maximum_rows, random_state=seed).sort_values("scheduled_time")
    return selected


def make_oof_rows(
    valid: pd.DataFrame,
    prediction: np.ndarray,
    model: str,
    fold: str,
) -> pd.DataFrame:
    result = valid[["MVT_ID_mvt", "date", "airport", TARGET]].copy()
    result.rename(columns={TARGET: "actual_sec"}, inplace=True)
    result["prediction_sec"] = prediction.astype(np.float32)
    result["error_sec"] = result["prediction_sec"] - result["actual_sec"]
    result["abs_error_sec"] = result["error_sec"].abs()
    result["model"] = model
    result["fold"] = fold
    return result


def train_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    iterations: int,
    threads: int,
    fold: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise RuntimeError(
            "CatBoost is not installed. Run .venv/bin/pip install -r modeling/requirements.lock"
        ) from error

    feature_columns = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.08,
        depth=8,
        l2_leaf_reg=6,
        random_strength=0.5,
        random_seed=SEED,
        thread_count=threads,
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(
        train[feature_columns],
        train[TARGET],
        cat_features=list(CATEGORICAL_FEATURES),
    )
    prediction = model.predict(valid[feature_columns])
    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"catboost_{fold}.cbm")
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(),
            "fold": fold,
        }
    )
    return prediction, importance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one bounded May-fold integration test")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--skip-boosted", action="store_true")
    arguments = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log("load and aggregate scheduled movements")
    files = training_files()
    flow_features, runway_features = build_schedule_counts(files)
    log(f"schedule features: flow={len(flow_features):,} runway={len(runway_features):,}")
    departures = load_departures(files)
    log(f"valid departure targets: {len(departures):,}")
    frame = build_features(departures, flow_features, runway_features)
    del departures, flow_features, runway_features
    gc.collect()

    folds = FOLDS[:1] if arguments.smoke else FOLDS
    iterations = min(arguments.iterations, 30) if arguments.smoke else arguments.iterations
    max_train = 200_000 if arguments.smoke else None
    max_valid = 60_000 if arguments.smoke else None
    oof_parts = []
    importance_parts = []
    fold_metrics = []
    for fold_index, fold in enumerate(folds):
        train = select_rows(frame, fold.train_start, fold.train_end, max_train, SEED + fold_index)
        valid = select_rows(frame, fold.valid_start, fold.valid_end, max_valid, SEED + 100 + fold_index)
        if train.empty or valid.empty:
            raise ValueError(f"Fold {fold.name} is empty: train={len(train)} valid={len(valid)}")
        log(f"fold={fold.name} train={len(train):,} valid={len(valid):,}")
        baseline_prediction = hierarchical_median_predict(train, valid)
        baseline_metrics = metric_row(valid[TARGET].to_numpy(float), baseline_prediction)
        fold_metrics.append({"model": "airport_calendar", "fold": fold.name, **baseline_metrics})
        oof_parts.append(make_oof_rows(valid, baseline_prediction, "airport_calendar", fold.name))
        log(f"fold={fold.name} airport_calendar_rmse={baseline_metrics['rmse_sec']:.2f}")

        if not arguments.skip_boosted:
            boosted_prediction, importance = train_catboost(
                train,
                valid,
                iterations=iterations,
                threads=arguments.threads,
                fold=fold.name,
            )
            boosted_metrics = metric_row(valid[TARGET].to_numpy(float), boosted_prediction)
            fold_metrics.append({"model": "catboost_operational", "fold": fold.name, **boosted_metrics})
            oof_parts.append(
                make_oof_rows(valid, boosted_prediction, "catboost_operational", fold.name)
            )
            importance_parts.append(importance)
            log(f"fold={fold.name} catboost_rmse={boosted_metrics['rmse_sec']:.2f}")
        del train, valid
        gc.collect()

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = RESULTS_DIR / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    importance = (
        pd.concat(importance_parts, ignore_index=True)
        if importance_parts
        else pd.DataFrame(columns=["feature", "importance", "fold"])
    )
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    metadata: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "smoke": arguments.smoke,
        "iterations": iterations,
        "threads": arguments.threads,
        "seed": SEED,
        "training_files": [path.name for path in files],
        "training_file_sha256": {path.name: sha256_file(path) for path in files},
        "training_rows_after_quality_filter": len(frame),
        "folds": [asdict(fold) for fold in folds],
        "fold_metrics": fold_metrics,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "unsafe_fields_excluded": [
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "AOBT_3_flt",
            "ARVT_1_flt",
            "ARVT_3_flt",
            "FLIGHT_ID_mvt",
            "FLIGHT_mvt",
            "CALLSIGN_flt",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(("catboost", "numpy", "pandas", "pyarrow")),
        },
    }
    (RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    sys.path.insert(0, str(ROOT / "modeling"))
    from dashboard import build_dashboard

    build_dashboard(oof, importance, metadata, DASHBOARD_PATH, RESULTS_DIR)
    log(f"wrote {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
