#!/usr/bin/env python3
"""Check attached exp-137 dataset files and schemas against kernel reads."""
from __future__ import annotations
import os

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import pandas as pd
import pyarrow.parquet as pq



def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[1]
EXP130 = Path("/tmp/prc-exp130-clean-input")
EXP136 = Path("/tmp/prc-exp136-input")
EXP137 = Path("/tmp/prc-exp137-input")
ID = "MVT_ID_mvt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def schema(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def require(path: Path, columns: set[str], context: str) -> None:
    actual = schema(path)
    if not columns.issubset(actual):
        raise RuntimeError(f"{context}: missing fields {sorted(columns - actual)}")


def check_exp130() -> dict:
    manifest = json.loads((EXP130 / "input-manifest.json").read_text())
    if manifest.get("ranking_target_read") is not False or manifest.get("weather_used_or_downloaded") is not False:
        raise RuntimeError("exp-130 input manifest does not prove label-safe, weather-free inputs")
    files = [path for path in EXP130.rglob("*") if path.is_file()
             and path.name != "dataset-metadata.json" and path.suffix != ".pyc"
             and "__pycache__" not in path.parts]
    required_files = {
        "fit.py", "input-manifest.json", "preflight.json", "rank_counts.csv",
        "rank_meta_2026-01.parquet", "rank_meta_2026-07.parquet",
        "eval_july_loo.parquet", "eval_december.parquet", "eval_january.parquet",
        "submitting_ids.parquet", "v22_submission.parquet",
    }
    required_files.update(f"month_2025-{month:02d}.parquet" for month in range(1, 13))
    required_files.update({"month_2026-01.parquet", "month_2026-07.parquet"})
    required_files.update({"dataset.py", "features.py", "model.py", "data.py", "lirf_features.py"})
    actual_names = {path.name for path in files}
    if not required_files.issubset(actual_names):
        raise RuntimeError(f"exp-130 package lacks required files: {sorted(required_files-actual_names)}")
    fit = EXP130 / "fit.py"
    if not fit.is_file():
        raise RuntimeError("exp-130 input package lacks fit.py")
    training_schemas = []
    for month in range(1, 13):
        name = f"2025-{month:02d}"
        path = EXP130 / f"month_{name}.parquet"
        require(path, {ID, "month", "y", "ghost", "r_aobt"}, f"exp-130 {name}")
        training_schemas.append(schema(path))
    rank_count = {"month", "airport", "ranking_rows"}
    for name in ("rank_counts.csv",):
        header = set((EXP130 / name).read_text().splitlines()[0].split(","))
        if not rank_count.issubset(header):
            raise RuntimeError(f"exp-130 {name} lacks ranking count fields")
    rank_fields = {ID, "month", "airport", "utc_day", "obs_has_clock", "noac", "rank_class"}
    for month in ("2026-01", "2026-07"):
        path = EXP130 / f"month_{month}.parquet"
        require(path, {ID, "month", "ghost", "r_aobt"}, f"exp-130 {month}")
        if "y" in schema(path):
            raise RuntimeError(f"exp-130 {month} inference frame contains y")
        require(EXP130 / f"rank_meta_{month}.parquet", rank_fields, f"exp-130 rank metadata {month}")
    for fold in ("july_loo", "december", "january"):
        require(EXP130 / f"eval_{fold}.parquet",
                {ID, "y_true", "r12", "airport", "utc_day", "obs_has_clock"},
                f"exp-130 held fold {fold}")
    source = EXP130 / "source"
    for relative in ("exp018/dataset.py", "exp018/features.py", "exp018/model.py",
                     "prc/data.py", "prc/lirf_features.py"):
        if not (source / relative).is_file():
            raise RuntimeError(f"exp-130 input source is missing {relative}")
    spec = importlib.util.spec_from_file_location("exp130_fit_schema_check", fit)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the exp-130 feature schema source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.INPUT = EXP130
    D18, F, M, _ = module.setup_modules()
    train_columns = set.union(*training_schemas) | {
        "clock_current250_row", "clock_implied_row"}
    train_empty = pd.DataFrame({column: pd.Series(dtype="float64")
                                for column in sorted(train_columns)})
    base_feature_counts = {}
    base_feature_columns = None
    for fold in ("july_loo", "december", "january"):
        held_month = {"july_loo": "2025-07", "december": "2025-12",
                      "january": "2025-01"}[fold]
        valid_columns = (schema(EXP130 / f"month_{held_month}.parquet")
                         | {"y_true", "r12", "obs_has_clock",
                            "clock_current250_row", "clock_implied_row"})
        valid_empty = pd.DataFrame({column: pd.Series(dtype="float64")
                                    for column in sorted(valid_columns)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            encoded_train, encoded_targets = D18.add_encodings(train_empty.copy(), [valid_empty])
        _, _, columns, categorical = module.feature_frames(
            encoded_train, encoded_targets[0], F, M)
        if len(columns) != 151 or not {"clock_current250_row", "clock_implied_row"}.issubset(columns):
            raise RuntimeError(f"exp-130 {fold}: expected 151 base-plus-clock features, got {len(columns)}")
        if base_feature_columns is not None and columns != base_feature_columns:
            raise RuntimeError(f"exp-130 {fold}: feature names or order differ from other folds")
        base_feature_columns = columns
        base_feature_counts[fold] = {"features": len(columns), "categorical": len(categorical)}
    rank_schema = schema(EXP130 / "month_2026-01.parquet")
    rank_empty = pd.DataFrame({column: pd.Series(dtype="float64")
                               for column in sorted(rank_schema | {
                                   "clock_current250_row", "clock_implied_row"})})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        encoded_train, encoded_targets = D18.add_encodings(train_empty.copy(), [rank_empty])
    _, _, rank_features, _ = module.feature_frames(encoded_train, encoded_targets[0], F, M)
    if rank_features != base_feature_columns:
        raise RuntimeError("exp-130 ranking feature names or order differ from held-fold features")
    return {"file_count": len(files), "fit_sha256": sha256(fit),
            "base_plus_clock_features": base_feature_counts,
            "ranking_target_read": False, "weather_used_or_downloaded": False}


def check_exp136() -> dict:
    inventory = json.loads((EXP136 / "source_inventory.json").read_text())
    if (inventory.get("ranking_target_read") is not False
            or inventory.get("departure_target_column_read") is not False
            or inventory.get("taxitime_read_for") != "ARR rows only"):
        raise RuntimeError("exp-136 input inventory does not prove label-safe source use")
    expected_files = set(inventory["files"]) | {"source_inventory.json"}
    actual_files = {path.name for path in EXP136.iterdir() if path.is_file()
                    and path.name != "dataset-metadata.json"}
    if actual_files != expected_files:
        raise RuntimeError(f"exp-136 files differ: missing={sorted(expected_files-actual_files)} extra={sorted(actual_files-expected_files)}")
    for name, record in inventory["files"].items():
        if name in {"fit_exp136.py", "proxy_features.py"}:
            # Exp-137 does not execute either exp-136 source file. The exp-136
            # v4 input check pins its current runner separately; the old
            # inventory hash for fit_exp136.py predates the v4 runner update.
            continue
        path = EXP136 / name
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"exp-136 input file hash differs: {name}")
    dep_columns = schema(EXP136 / "proxy_departures.parquet")
    arr_columns = schema(EXP136 / "proxy_arrivals.parquet")
    required_dep = {ID, "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "EOBT_1_flt",
                    "FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "month"}
    required_arr = {ID, "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "month"}
    if not required_dep.issubset(dep_columns) or "TAXITIME_SEC_mvt" in dep_columns:
        raise RuntimeError("exp-136 departure schema has missing inputs or a departure label")
    if not required_arr.issubset(arr_columns):
        raise RuntimeError("exp-136 arrival schema lacks required ARR fields")
    expected = json.loads((EXP136 / "points_expected.json").read_text())
    point_count = sum(len(rows) for rows in expected["months"].values())
    sources = inventory.get("point_source_kernels", [])
    if point_count != 427 or len(sources) != 17:
        raise RuntimeError(f"exp-136 point inventory is {point_count} files from {len(sources)} kernels")
    geometry = set((EXP136 / "stand_geometry.csv").read_text().splitlines()[0].split(","))
    if not {"airport", "stand_name_mvt", "stand_latitude", "stand_longitude"}.issubset(geometry):
        raise RuntimeError("exp-136 stand geometry schema lacks required fields")
    return {"file_count": len(actual_files), "departure_columns": len(dep_columns),
            "arrival_columns": len(arr_columns), "point_files": point_count,
            "point_source_kernels": sources, "inventory_sha256": sha256(EXP136 / "source_inventory.json")}


def check_exp137() -> dict:
    inventory = json.loads((EXP137 / "exp137_source_inventory.json").read_text())
    if inventory.get("ranking_target_read") is not False or inventory.get("weather_used_or_downloaded") is not False:
        raise RuntimeError("exp-137 source inventory does not preserve label safety or weather exclusion")
    spec = ROOT / "SPEC.md"
    if sha256(spec) != inventory.get("spec_sha256"):
        raise RuntimeError("private exp-137 input package uses a different frozen SPEC")
    required_fold = {ID, "fold", "airport", "obs_has_clock", "known_aircraft",
                     "p12", "p22", "delta127", "p23"}
    require(EXP137 / "fold_base_v23.parquet", required_fold, "exp-137 v23 fold base")
    require(EXP137 / "ranking_base_v24.parquet", {ID, "row_order", "p24"}, "exp-137 v24 ranking base")
    base = json.loads((EXP137 / "base_reproduction-r1.json").read_text())
    if base.get("experiment") != "exp-127-clock-precision":
        raise RuntimeError("exp-137 base source is not exp-127")
    for name in ("fit_exp137_arm.py", "fit_exp137_final.py", "proxy_features_exp137.py"):
        if not (EXP137 / "code" / name).is_file():
            raise RuntimeError(f"exp-137 input package lacks {name}")
    for name, record in inventory["files"].items():
        path = EXP137 / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"exp-137 input package hash mismatch: {name}")
    point_sources = json.loads((EXP136 / "source_inventory.json").read_text())["point_source_kernels"]
    expected_datasets = [f"{kaggle_user()}/prc-exp130-clean-input", f"{kaggle_user()}/prc-exp136-input",
                         f"{kaggle_user()}/prc-exp137-input"]
    kernel_checks = {}
    for arm in ("control", "proxy", "sham"):
        folder = ROOT / "kaggle/arm-kernels" / arm
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        code = (folder / "kernel.py").read_text()
        if metadata.get("is_private") is not True or metadata.get("enable_gpu") is not False\
                or metadata.get("enable_tpu") is not False or metadata.get("enable_internet") is not False:
            raise RuntimeError(f"{arm}: kernel must be private CPU with internet disabled")
        if metadata.get("dataset_sources") != expected_datasets or metadata.get("kernel_sources") != point_sources:
            raise RuntimeError(f"{arm}: attached datasets or point-source kernels differ")
        if metadata.get("id") != f"{kaggle_user()}/prc-exp137-with-clock-{arm}-arm" or f'PRC_EXP137_ARM"] = "{arm}"' not in code:
            raise RuntimeError(f"{arm}: kernel slug or single-arm selection differs")
        if any(line not in code for line in (
                'PRC_KERNEL_VERSION"] = "2"',
                'PRC_EXP130_DATASET_VERSION"] = "1"',
                'PRC_EXP136_DATASET_VERSION"] = "4"',
                'PRC_EXP137_DATASET_VERSION"] = "3"')):
            raise RuntimeError(f"{arm}: dataset versions are not pinned in its wrapper")
        kernel_checks[arm] = {"slug": metadata["id"], "private": True,
                              "dataset_sources": len(metadata["dataset_sources"]),
                              "point_source_kernels": len(metadata["kernel_sources"])}
    return {"file_count": len(inventory["files"]), "inventory_sha256": sha256(EXP137 / "exp137_source_inventory.json"),
            "spec_sha256": inventory["spec_sha256"], "base_fold_rmse": inventory["base_fold_rmse"],
            "fold_rows": inventory["base_fold_rows"], "ranking_rows": inventory["ranking_rows"],
            "kernel_checks": kernel_checks}


def main() -> None:
    result = {"exp130": check_exp130(), "exp136": check_exp136(), "exp137": check_exp137()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
