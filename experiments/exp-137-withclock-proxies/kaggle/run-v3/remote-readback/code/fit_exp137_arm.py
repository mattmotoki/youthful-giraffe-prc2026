#!/usr/bin/env python3
"""Run one frozen exp-137 arm in one private Kaggle CPU kernel."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import proxy_features_exp137 as PX


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/exp137-output")
ID = "MVT_ID_mvt"
MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 13))
RANK_MONTHS = ("2026-01", "2026-07")
FOLDS = (("july_loo", "2025-07", 1),
         ("december", "2025-12", 2),
         ("january", "2025-01", 3))
FOLD_MONTH = {fold: month for fold, month, _ in FOLDS}
RANK_MONTH = {"july_loo": "2026-07", "december": "2026-01", "january": "2026-01"}
CLOCK_COLS = ["clock_current250_row", "clock_implied_row"]
ARM_MODE = {"control": None, "proxy": "proxy", "sham": "sham"}
BOOT_DRAWS = 2000
BOOT_SEED = 20260925
MIN_SUPPORT_ROWS = 100
MIN_SUPPORT_DAYS = 10
MISSING_PROXY_ROWS: dict[str, dict] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_file(name: str) -> Path:
    paths = sorted(path for path in INPUT.rglob(name) if path.is_file())
    if len(paths) != 1:
        raise RuntimeError(f"expected one attached {name}, found {len(paths)}: {paths}")
    return paths[0]


def aligned_features(source: pd.DataFrame, rows: pd.DataFrame,
                     source_months: list[str], context: str,
                     columns: list[str]) -> pd.DataFrame:
    row_ids = pd.to_numeric(rows[ID], errors="raise").astype("int64").to_numpy()
    missing = ~pd.Index(row_ids).isin(source.index)
    if missing.any():
        missing_ids = np.sort(row_ids[missing]).astype("<i8", copy=False)
        id_hash = hashlib.sha256(missing_ids.tobytes()).hexdigest()
        record_key = f"{len(source_months)}:{','.join(source_months)}:{id_hash}"
        MISSING_PROXY_ROWS[record_key] = {
            "source_months": list(source_months),
            "row_count": int(len(missing_ids)),
            "sorted_id_sha256": id_hash,
        }
        print(f"{context}: {len(missing_ids)} source IDs are missing; keep rows with NaNs",
              flush=True)
    selected = source.reindex(row_ids)
    if len(selected) != len(rows):
        raise RuntimeError(f"{context}: feature rows do not align with model rows")
    selected.index = rows.index
    return selected[columns].astype("float32")


def base_fit_module():
    path = exact_file("fit.py")
    spec = importlib.util.spec_from_file_location("exp130_fit_for_exp137", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen exp-130 fit source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def build_views(raw_months: dict, fit_months: list[str], fold_index: int, arm: str):
    mode = ARM_MODE[arm]
    if mode is None:
        return {}
    denominators = PX.fit_denominators(raw_months, fit_months)
    seed = 13417 + fold_index
    views = {}
    for month in (*MONTHS, *RANK_MONTHS):
        z, windows = PX.transform_rows(raw_months[month], denominators)
        views[month] = {mode: PX.row_view(z, windows, mode=mode, seed=seed)}
    return views


def month_proxy_features(views: dict, months: list[str], mode: str,
                         rows: pd.DataFrame) -> pd.DataFrame:
    pieces = [views[month][mode][[ID, *PX.Z_COLS]] for month in months]
    features = pd.concat(pieces, ignore_index=True)
    if features[ID].duplicated().any():
        raise RuntimeError(f"{mode}: duplicate proxy IDs across months")
    return aligned_features(features.set_index(ID), rows, months,
                            f"{mode} row features", PX.Z_COLS)


def add_clock_features(month_frames: dict, raw_months: dict) -> None:
    for month, frame in month_frames.items():
        source = raw_months[month]["row_features"].set_index(ID)
        ids = pd.to_numeric(frame[ID], errors="raise").astype("int64").to_numpy()
        selected = source.reindex(ids)
        if len(selected) != len(frame):
            raise RuntimeError(f"{month}: row clock inputs do not align")
        for column in CLOCK_COLS:
            frame[column] = pd.to_numeric(selected[column], errors="coerce").to_numpy("float32")


def load_fold_base() -> tuple[pd.DataFrame, dict]:
    path = exact_file("fold_base_v23.parquet")
    frame = pd.read_parquet(path)
    required = {ID, "fold", "airport", "known_aircraft", "p22", "delta127", "p23"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"fold base is missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["fold", ID]).any():
        raise RuntimeError("fold base IDs are not unique")
    if not np.isfinite(frame[["p22", "delta127", "p23"]].to_numpy("float64")).all():
        raise RuntimeError("fold base contains non-finite values")
    if not np.allclose(frame.p22 + frame.delta127, frame.p23, rtol=0, atol=1e-10):
        raise RuntimeError("fold base differs from v22 plus exp-127 OOF delta")
    if frame.loc[frame.airport.astype(str).eq("LIRF"), "delta127"].abs().max() > 1e-12:
        raise RuntimeError("exp-127 unit-2 delta is not held at LIRF")
    return frame, json.loads(exact_file("base_reproduction-r1.json").read_text())


def evaluation_frame(base, fold: str, month: str, month_frames: dict,
                     fold_base: pd.DataFrame) -> pd.DataFrame:
    valid = month_frames[month].drop(columns=["y"], errors="ignore").copy()
    evaluation = base.eval_frame(fold).rename(
        columns={"airport": "eval_airport", "utc_day": "eval_utc_day"})
    valid = valid.merge(evaluation, on=ID, how="left", validate="one_to_one")
    if valid[["y_true", "r12"]].isna().any().any():
        raise RuntimeError(f"{fold}: saved fold labels or p12 residuals are missing")
    if (not valid["airport"].astype(str).eq(valid["eval_airport"].astype(str)).all()
            or not valid["utc_day"].astype(str).eq(valid["eval_utc_day"].astype(str)).all()):
        raise RuntimeError(f"{fold}: prepared feature metadata differs from saved fold rows")
    valid = valid.drop(columns=["eval_airport", "eval_utc_day"])
    valid["p12"] = valid["y_true"].to_numpy("float64") - valid["r12"].to_numpy("float64")
    fold_values = fold_base.loc[fold_base.fold.eq(fold),
                                [ID, "known_aircraft", "p22", "delta127", "p23"]]
    valid = valid.merge(fold_values, on=ID, how="left", validate="one_to_one", indicator=True)
    if not valid["_merge"].eq("both").all() or not np.isfinite(valid.p23).all():
        raise RuntimeError(f"{fold}: v23 base is missing a prepared row")
    return valid.drop(columns="_merge")


def base_preflight(base, D18, D, month_frames: dict, fold_base: pd.DataFrame,
                   base_reproduction: dict) -> pd.DataFrame:
    records = []
    for fold, month, _ in FOLDS:
        values = fold_base.loc[fold_base.fold.eq(fold)]
        valid = base.eval_frame(fold)[[ID, "y_true", "airport", "utc_day",
                                      "obs_has_clock"]].merge(
            values[[ID, "known_aircraft", "p22", "p23"]],
            on=ID, how="left", validate="one_to_one")
        if valid.p23.isna().any() or valid[ID].duplicated().any():
            raise RuntimeError(f"{fold}: full-fold v23 prediction rows are incomplete")
        rmse = float(np.sqrt(np.mean((valid.y_true.to_numpy("float64")
                                      - valid.p23.to_numpy("float64")) ** 2)))
        expected = float(base_reproduction["folds"][fold]["rmse_s_lirf_held"])
        if abs(rmse - expected) > 1e-6:
            raise RuntimeError(f"{fold}: full-fold v23 RMSE {rmse:.9f} differs from {expected:.9f}")

        with_clock = (valid.obs_has_clock.astype(bool).to_numpy()
                      & valid.known_aircraft.astype(bool).to_numpy())
        if not with_clock.any():
            raise RuntimeError(f"{fold}: observed-clock base rows are empty")
        with_clock_rmse = float(np.sqrt(np.mean(
            (valid.loc[with_clock, "y_true"].to_numpy("float64")
             - valid.loc[with_clock, "p23"].to_numpy("float64")) ** 2)))

        fit_months = [value for value in MONTHS if value != month]
        train_raw = pd.concat([month_frames[value] for value in fit_months], ignore_index=True)
        train_raw = train_raw.loc[D.keep_reachable(train_raw)].copy()
        target = evaluation_frame(base, fold, month, month_frames, fold_base)
        _, (encoded,) = D18.add_encodings(train_raw, [target])
        ordinary = encoded["ghost"].to_numpy(dtype="float64") < 0.5
        clean = ~encoded[ID].astype("int64").isin(D.excluded_ids(fold)).to_numpy()
        with_clock = (encoded["obs_has_clock"].astype(bool).to_numpy()
                      & encoded["known_aircraft"].astype(bool).to_numpy())
        score_mask = ordinary & clean & with_clock
        scored = encoded.loc[score_mask]
        if scored.empty:
            raise RuntimeError(f"{fold}: clean ordinary with-clock baseline rows are empty")
        rmse_class = float(np.sqrt(np.mean((scored.y_true.to_numpy("float64")
                                            - scored.p23.to_numpy("float64")) ** 2)))
        records.append({"fold": fold, "full_fold_rows": int(len(valid)),
                        "v23_full_fold_rmse_s": rmse,
                        "source_rmse_s": expected,
                        "absolute_error_s": abs(rmse - expected),
                        "with_clock_rows": int(with_clock.sum()),
                        "v23_with_clock_rmse_s": with_clock_rmse,
                        "clean_with_clock_rows": int(len(scored)),
                        "clean_with_clock_rmse_s": rmse_class})
        del train_raw, target, encoded, scored
        gc.collect()
    print("all v23 source RMSE checks passed before the first learner fit", flush=True)
    return pd.DataFrame(records)


def rank_maps() -> dict[str, dict[str, int]]:
    maps = {}
    for fold, _, _ in FOLDS:
        month = RANK_MONTH[fold]
        frame = pd.read_parquet(exact_file(f"rank_meta_{month}.parquet"))
        required = {ID, "airport", "rank_class", "noac"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{month}: ranking metadata is missing required columns")
        part = frame.loc[frame.rank_class.eq("with-clock") & ~frame.noac.astype(bool)]
        counts = part.groupby(part.airport.astype(str), observed=True).size()
        maps[fold] = {str(airport): int(count) for airport, count in counts.items()}
        if not maps[fold] or any(count <= 0 for count in maps[fold].values()):
            raise RuntimeError(f"{fold}: with-clock ranking counts are invalid")
    return maps


def prefit_support_counts(base, D, month_frames: dict, fold_base: pd.DataFrame,
                         all_ranks: dict) -> tuple[dict, dict, list[dict]]:
    """Freeze supported airport-fold cells from row metadata before any model fit."""
    ranks = {}
    clean_counts = {}
    support_records = []
    for fold, _, _ in FOLDS:
        valid = evaluation_frame(base, fold, FOLD_MONTH[fold], month_frames, fold_base)
        ordinary = valid["ghost"].to_numpy(dtype="float64") < 0.5
        clean = ~valid[ID].astype("int64").isin(D.excluded_ids(fold)).to_numpy()
        with_clock = (valid["obs_has_clock"].astype(bool).to_numpy()
                      & valid["known_aircraft"].astype(bool).to_numpy())
        score_rows = valid.loc[ordinary & clean & with_clock, ["airport", "utc_day"]].copy()
        score_rows["airport"] = score_rows.airport.astype(str)
        by_airport = score_rows.groupby("airport", observed=True).agg(
            clean_validation_rows=("utc_day", "size"),
            clean_validation_utc_days=("utc_day", "nunique"))
        supported = {}
        clean_counts[fold] = {}
        for airport, ranking_rows in sorted(all_ranks[fold].items()):
            if airport in by_airport.index:
                validation_rows = int(by_airport.loc[airport, "clean_validation_rows"])
                validation_days = int(by_airport.loc[airport, "clean_validation_utc_days"])
            else:
                validation_rows = validation_days = 0
            is_supported = (validation_rows >= MIN_SUPPORT_ROWS
                            and validation_days >= MIN_SUPPORT_DAYS)
            support_records.append({
                "fold": fold, "airport": airport, "rank_month": RANK_MONTH[fold],
                "ranking_rows": int(ranking_rows),
                "clean_validation_rows": validation_rows,
                "clean_validation_utc_days": validation_days,
                "supported": bool(is_supported),
                "support_min_rows": MIN_SUPPORT_ROWS,
                "support_min_utc_days": MIN_SUPPORT_DAYS,
            })
            if is_supported:
                supported[airport] = int(ranking_rows)
                clean_counts[fold][airport] = validation_rows
        if not supported:
            raise RuntimeError(f"{fold}: no ranking airport cells meet the frozen support rule")
        ranks[fold] = supported
    return ranks, clean_counts, support_records


def scoring_features(base, D18, F, M, D, month_frames: dict, fold_base: pd.DataFrame,
                     views: dict, fit_months: list[str], score_folds: tuple[str, ...],
                     arm: str, split_name: str) -> tuple[dict[str, pd.DataFrame], list[dict], list[pd.DataFrame], list[dict]]:
    train_raw = pd.concat([month_frames[month] for month in fit_months], ignore_index=True)
    train_raw = train_raw.loc[D.keep_reachable(train_raw)].copy()
    valid_parts = [evaluation_frame(base, fold, FOLD_MONTH[fold], month_frames, fold_base)
                   for fold in score_folds]
    valid_parts = [part.loc[part["known_aircraft"].astype(bool)]
                       .drop(columns="known_aircraft").copy()
                   for part in valid_parts]
    train_encoded, valid_encoded = D18.add_encodings(train_raw, valid_parts)
    train_ordinary = train_encoded["ghost"].to_numpy(dtype="float64") < 0.5
    train = train_encoded.loc[train_ordinary].reset_index(drop=True)
    valid_ordinary = [part["ghost"].to_numpy(dtype="float64") < 0.5 for part in valid_encoded]
    valid = pd.concat([part.loc[mask].copy() for part, mask in zip(valid_encoded, valid_ordinary)],
                      ignore_index=True)
    if train.empty or valid.empty:
        raise RuntimeError(f"{split_name}: ordinary training or validation rows are empty")

    anchor_train = base.anchor_for(train)
    y_train = train["y"].to_numpy("float64")
    anchor_valid = base.anchor_for(valid)
    if not np.isfinite(anchor_train).all() or not np.isfinite(y_train).all() or not np.isfinite(anchor_valid).all():
        raise RuntimeError(f"{split_name}: target or anchor is not finite")

    helper_cols = ["known_aircraft", "p12", "p22", "delta127", "p23"]
    valid_for_features = valid.drop(columns=helper_cols, errors="ignore")
    base_x_train, base_x_valid, base_columns, categorical = base.feature_frames(
        train, valid_for_features, F, M)
    expected_base = 151
    if len(base_columns) != expected_base or not set(CLOCK_COLS).issubset(base_columns):
        raise RuntimeError(f"{split_name}: expected 151 base-plus-clock features, got {len(base_columns)}")

    mode = ARM_MODE[arm]
    fit_info = []
    importance = []
    if mode is None:
        x_train, x_valid = base_x_train, base_x_valid
    else:
        z_train = month_proxy_features(views, fit_months, mode, train)
        score_months = [FOLD_MONTH[fold] for fold in score_folds]
        z_valid = month_proxy_features(views, score_months, mode, valid)
        x_train = pd.concat([base_x_train.reset_index(drop=True), z_train.reset_index(drop=True)], axis=1)
        x_valid = pd.concat([base_x_valid.reset_index(drop=True), z_valid.reset_index(drop=True)], axis=1)
        if x_train.shape[1] != 172 or x_valid.shape[1] != 172:
            raise RuntimeError(f"{split_name}: expected 172 arm features")

    prediction, record = base.fit_arm(
        split_name, x_train, x_valid, y_train - anchor_train,
        np.ones(len(train), dtype="float32"), categorical, anchor_valid, M, split_name)
    info = {"split": split_name, "arm": arm, **record["info"]}
    fit_info.append(info)
    importance_frame = record["importance"].copy()
    importance_frame["split"] = split_name
    importance_frame["arm"] = arm
    importance.append(importance_frame)

    score_frames = {}
    for fold in score_folds:
        count = int(valid["month"].astype(str).eq(FOLD_MONTH[fold]).sum())
        part_mask = valid["month"].astype(str).eq(FOLD_MONTH[fold]).to_numpy()
        held = valid.loc[part_mask].copy().reset_index(drop=True)
        pred = prediction[part_mask]
        clean = ~held[ID].astype("int64").isin(D.excluded_ids(fold)).to_numpy()
        with_clock = held["obs_has_clock"].astype(bool).to_numpy()
        score_mask = clean & with_clock
        rows = held.loc[score_mask, [ID, "airport", "utc_day", "y_true", "p12", "p22", "p23"]].copy()
        rows["fold"] = fold
        rows["pred_raw"] = pred[score_mask]
        rows["clock_current250_row"] = pd.to_numeric(
            held.loc[score_mask, CLOCK_COLS[0]], errors="coerce").to_numpy("float32")
        rows["clock_implied_row"] = pd.to_numeric(
            held.loc[score_mask, CLOCK_COLS[1]], errors="coerce").to_numpy("float32")
        if rows.empty or not np.isfinite(rows[["y_true", "p22", "p23", "pred_raw"]]).all().all():
            raise RuntimeError(f"{fold}: scored with-clock rows or predictions are invalid")
        score_frames[fold] = rows.reset_index(drop=True)

        if mode is not None:
            z_test = month_proxy_features(views, [FOLD_MONTH[fold]], mode, rows)
            coverage = []
            for column in PX.Z_COLS:
                values = z_test[column].to_numpy("float64")
                coverage.append({"fold": fold, "arm_view": mode, "feature": column,
                                 "rows": int(len(values)),
                                 "finite_rows": int(np.isfinite(values).sum()),
                                 "finite_rate": float(np.isfinite(values).mean())})
            score_frames[fold].attrs["proxy_coverage"] = coverage
        else:
            score_frames[fold].attrs["proxy_coverage"] = []
        del held

    del train_raw, train_encoded, valid_encoded, train, valid, base_x_train, base_x_valid
    if mode is not None:
        del z_train, z_valid, x_train, x_valid
    gc.collect()
    return score_frames, fit_info, importance, [row for frame in score_frames.values()
                                                for row in frame.attrs["proxy_coverage"]]


def row_weights(rows: pd.DataFrame, fold: str, ranks: dict, clean_counts: dict) -> np.ndarray:
    airports = rows.airport.astype(str).to_numpy()
    weights = np.asarray([
        ranks[fold][airport] / clean_counts[fold][airport]
        if airport in ranks[fold] and airport in clean_counts[fold] else 0.0
        for airport in airports
    ], dtype="float64")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError(f"{fold}: invalid clean with-clock ranking weights")
    return weights


def fit_weight(source_rows: list[tuple[str, pd.DataFrame]], fold: str,
               ranks: dict, clean_counts: dict) -> dict:
    numerator = denominator = 0.0
    source_n = 0
    for source_fold, rows in source_rows:
        weights = row_weights(rows, source_fold, ranks, clean_counts)
        delta = rows.pred_raw.to_numpy("float64") - rows.p23.to_numpy("float64")
        target = rows.y_true.to_numpy("float64") - rows.p23.to_numpy("float64")
        good = np.isfinite(delta) & np.isfinite(target) & np.isfinite(weights)
        numerator += float(np.sum(weights[good] * delta[good] * target[good]))
        denominator += float(np.sum(weights[good] * delta[good] ** 2))
        source_n += int(good.sum())
    raw = numerator / denominator if denominator > 0 else 0.0
    return {"fold": fold, "weight": float(np.clip(raw, 0.0, 1.0)),
            "unclipped_weight": float(raw), "numerator": numerator,
            "denominator": denominator, "rows": source_n}


def add_blend(rows: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = rows.copy()
    p23 = out.p23.to_numpy("float64")
    y = out.y_true.to_numpy("float64")
    raw = out.pred_raw.to_numpy("float64")
    blended = p23 + weight * (raw - p23)
    out["weight"] = weight
    out["pred_blend"] = blended
    out["gain_blend"] = (y - p23) ** 2 - (y - blended) ** 2
    return out


def bootstrap_interval(values: np.ndarray, days: np.ndarray, draws: int,
                       rng: np.random.Generator) -> tuple[float, float]:
    unique = np.asarray(sorted(pd.unique(days)), dtype=object)
    if not len(unique):
        raise RuntimeError("airport has no UTC-day bootstrap clusters")
    groups = [values[np.asarray(days == day)] for day in unique]
    draw_ids = rng.integers(0, len(groups), size=(draws, len(groups)))
    # Retain unequal day sizes by sampling days and averaging the selected rows.
    means = np.empty(draws, dtype="float64")
    for i, selected in enumerate(draw_ids):
        sample = np.concatenate([groups[index] for index in selected])
        means[i] = float(np.mean(sample))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def weighted_gain(rows: pd.DataFrame, metric: str, fold: str,
                  ranks: dict, clean_counts: dict) -> float:
    cell_means = rows.assign(_airport=rows.airport.astype(str)).groupby(
        "_airport", sort=True, observed=True)[metric].mean()
    total = sum(ranks[fold].values())
    if set(ranks[fold]) - set(cell_means.index):
        raise RuntimeError(f"{fold}: one or more ranking airports lack clean score rows")
    return float(sum(ranks[fold][airport] * cell_means[airport]
                     for airport in ranks[fold]) / total)


def main(arm: str) -> None:
    if arm not in ARM_MODE:
        raise RuntimeError(f"unknown exp-137 arm {arm!r}")
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_inventory_path = exact_file("exp137_source_inventory.json")
    inventory = json.loads(input_inventory_path.read_text())
    if inventory.get("ranking_target_read") is not False or inventory.get("weather_used_or_downloaded") is not False:
        raise RuntimeError("exp-137 input inventory does not prove label-safe, weather-free inputs")
    source_inventory_path = exact_file("source_inventory.json")
    proxy_inventory = json.loads(source_inventory_path.read_text())
    if (proxy_inventory.get("ranking_target_read") is not False
            or proxy_inventory.get("departure_target_column_read") is not False
            or proxy_inventory.get("taxitime_read_for") != "ARR rows only"):
        raise RuntimeError("exp-136 proxy input inventory does not prove label-safe inputs")
    exp130_inventory = json.loads(exact_file("input-manifest.json").read_text())
    if exp130_inventory.get("ranking_target_read") is not False:
        raise RuntimeError("exp-130 input inventory does not prove ranking targets were withheld")
    preflight = json.loads(exact_file("preflight.json").read_text())
    if preflight.get("v22_baseline_passed") is not True:
        raise RuntimeError("exp-130 v22 preflight did not pass")

    # Build all proxy sources before opening any fit or fold labels.
    raw_months, proxy_audit = PX.build_all(MONTHS, RANK_MONTHS)
    print("all exp-045 point files and label-free proxy source tables passed before fold-label reads",
          flush=True)

    base, base_code_path = base_fit_module()
    fold_base, base_reproduction = load_fold_base()
    D18, F, M, D = base.setup_modules()
    month_frames = {month: base.month_frame(month) for month in (*MONTHS, *RANK_MONTHS)}
    add_clock_features(month_frames, raw_months)
    all_ranks = rank_maps()

    base_checks = base_preflight(base, D18, D, month_frames, fold_base, base_reproduction)
    base_checks.to_csv(OUTPUT / "base_reproduction.csv", index=False)
    ranks, clean_counts, support_records = prefit_support_counts(
        base, D, month_frames, fold_base, all_ranks)
    pd.DataFrame(support_records).to_csv(OUTPUT / "ranking_support_preflight.csv", index=False)
    print("airport-fold support check passed before the first learner fit", flush=True)

    main_scores: dict[str, pd.DataFrame] = {}
    main_fit_records: list[dict] = []
    importance_records: list[pd.DataFrame] = []
    proxy_coverage = []
    for fold, valid_month, fold_index in FOLDS:
        fit_months = [month for month in MONTHS if month != valid_month]
        print(f"main OOF start arm={arm} fold={fold}", flush=True)
        views = build_views(raw_months, fit_months, fold_index, arm)
        score, fit_info, importance, coverage = scoring_features(
            base, D18, F, M, D, month_frames, fold_base, views, fit_months,
            (fold,), arm, f"main_{fold}_{arm}")
        main_scores.update(score)
        main_fit_records.extend(fit_info)
        importance_records.extend(importance)
        proxy_coverage.extend(coverage)
        print(f"main OOF end arm={arm} fold={fold}: {len(score[fold]):,} clean with-clock rows",
              flush=True)
        del views
        gc.collect()

    for fold, rows in main_scores.items():
        cells = rows.groupby(rows.airport.astype(str), observed=True).size().to_dict()
        mismatch = {airport: (int(cells.get(airport, 0)), expected)
                    for airport, expected in clean_counts[fold].items()
                    if int(cells.get(airport, 0)) != expected}
        if mismatch:
            raise RuntimeError(f"{fold}: fitted score rows differ from prefit support counts {mismatch}")

    calibration: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    calibration_fit_records: list[dict] = []
    calibration_importance: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    fold_names = [record[0] for record in FOLDS]
    for i, first in enumerate(fold_names):
        for second in fold_names[i + 1:]:
            pair = (first, second)
            first_index = next(index for name, _, index in FOLDS if name == first)
            pair_months = [FOLD_MONTH[name] for name in pair]
            fit_months = [month for month in MONTHS if month not in pair_months]
            print(f"calibration start arm={arm} folds={first}/{second}", flush=True)
            views = build_views(raw_months, fit_months, first_index, arm)
            score, fit_info, importance, coverage = scoring_features(
                base, D18, F, M, D, month_frames, fold_base, views, fit_months,
                pair, arm, f"cal_{first}_{second}_{arm}")
            calibration[pair] = score
            calibration_fit_records.extend(fit_info)
            calibration_importance.extend(importance)
            for fold, rows in score.items():
                record = rows.copy()
                record["calibration_pair"] = f"{first}/{second}"
                calibration_rows.append(record)
            proxy_coverage.extend(coverage)
            del views
            gc.collect()

    weights = {}
    weight_rows = []
    for held_fold in fold_names:
        sources = [source for source in fold_names if source != held_fold]
        pair = tuple(name for name in fold_names if name in {held_fold, sources[0]})
        pair2 = tuple(name for name in fold_names if name in {held_fold, sources[1]})
        source_rows = []
        for source_fold in sources:
            source_pair = pair if source_fold in pair else pair2
            source_rows.append((source_fold, calibration[source_pair][source_fold]))
        weight = fit_weight(source_rows, held_fold, ranks, clean_counts)
        weight["arm"] = arm
        weights[held_fold] = weight
        weight_rows.append(weight)

    ranking_source = [(fold, main_scores[fold]) for fold in ("january", "july_loo")]
    rank_weight = fit_weight(ranking_source, "ranking_jan_jul", ranks, clean_counts)
    rank_weight["arm"] = arm
    rank_weight["source_folds"] = "january,july_loo"
    weight_rows.append(rank_weight)

    blended = {}
    metric_rows = []
    for fold in fold_names:
        rows = add_blend(main_scores[fold], weights[fold]["weight"])
        blended[fold] = rows
        metrics = {
            "fold": fold, "arm": arm, "rows": int(len(rows)),
            "ranking_rows": int(sum(ranks[fold].values())),
            "weight": float(weights[fold]["weight"]),
            "p23_rmse_s": float(np.sqrt(np.mean((rows.y_true - rows.p23) ** 2))),
            "raw_rmse_s": float(np.sqrt(np.mean((rows.y_true - rows.pred_raw) ** 2))),
            "blend_rmse_s": float(np.sqrt(np.mean((rows.y_true - rows.pred_blend) ** 2))),
            "gain_blend_s2_per_rank_row": weighted_gain(rows, "gain_blend", fold, ranks, clean_counts),
        }
        metric_rows.append(metrics)

    pooled_n = sum(sum(ranks[fold].values()) for fold in ("january", "july_loo"))
    all_rank_rows_by_fold = {fold: int(sum(all_ranks[fold].values())) for fold in fold_names}
    supported_rank_rows_by_fold = {fold: int(sum(ranks[fold].values())) for fold in fold_names}
    ranking_coverage_by_fold = {
        fold: (supported_rank_rows_by_fold[fold] / all_rank_rows_by_fold[fold]
               if all_rank_rows_by_fold[fold] else 0.0)
        for fold in fold_names
    }
    jan_jul_total = all_rank_rows_by_fold["january"] + all_rank_rows_by_fold["july_loo"]
    jan_jul_supported = (supported_rank_rows_by_fold["january"]
                         + supported_rank_rows_by_fold["july_loo"])
    pooled_gain = sum(weighted_gain(blended[fold], "gain_blend", fold, ranks, clean_counts)
                      * sum(ranks[fold].values()) for fold in ("january", "july_loo")) / pooled_n
    rank_weighted_base = sum(
        weighted_gain(blended[fold].assign(_p23_error=(blended[fold].y_true - blended[fold].p23) ** 2),
                      "_p23_error", fold, ranks, clean_counts)
        * sum(ranks[fold].values()) for fold in ("january", "july_loo")) / pooled_n
    result = {
        "experiment": "exp-137-withclock-proxies", "arm": arm,
        "state": "arm_complete_no_gate", "spec_sha256": inventory["spec_sha256"],
        "ranking_target_read": False, "weather_used_or_downloaded": False,
        "fold_metrics": metric_rows,
        "pooled_january_july_gain_s2_per_rank_row": float(pooled_gain),
        "pooled_january_july_p23_mse_s2_per_rank_row": float(rank_weighted_base),
        "ranking_support": support_records,
        "all_rank_rows_by_fold": all_rank_rows_by_fold,
        "supported_rank_rows_by_fold": supported_rank_rows_by_fold,
        "ranking_coverage_by_fold": ranking_coverage_by_fold,
        "jan_jul_ranking_coverage": float(jan_jul_supported / jan_jul_total),
        "weights": weight_rows,
        "elapsed_seconds": time.monotonic() - started,
        "base_reproduction": base_checks.to_dict(orient="records"),
        "verified_point_files": int(proxy_audit["point_files_verified"]),
        "verified_point_bytes": int(proxy_audit["point_bytes_verified"]),
    }
    pd.concat([blended[fold] for fold in fold_names], ignore_index=True).to_parquet(
        OUTPUT / "main_predictions.parquet", index=False, compression="zstd")
    if calibration_rows:
        pd.concat(calibration_rows, ignore_index=True).to_parquet(
            OUTPUT / "calibration_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(weight_rows).to_csv(OUTPUT / "weights.csv", index=False)
    pd.DataFrame(support_records).to_csv(OUTPUT / "ranking_counts.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "metrics.csv", index=False)
    pd.DataFrame(main_fit_records + calibration_fit_records).to_csv(
        OUTPUT / "fit_info.csv", index=False)
    if main_fit_records:
        pd.concat(importance_records + calibration_importance, ignore_index=True).to_csv(
            OUTPUT / "feature_importance.csv", index=False)
    pd.DataFrame(proxy_coverage).drop_duplicates().to_csv(
        OUTPUT / "proxy_coverage.csv", index=False)
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")

    outputs = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    run_manifest = {
        "experiment": "exp-137-withclock-proxies", "arm": arm,
        "state": "arm_complete_no_gate",
        "kernel_slug": os.environ.get("PRC_KERNEL_SLUG", "set-by-kaggle"),
        "kernel_version": os.environ.get("PRC_KERNEL_VERSION", "set-by-kaggle"),
        "dataset_versions": {
            "prc-exp130-clean-input": os.environ.get("PRC_EXP130_DATASET_VERSION", "unknown"),
            "prc-exp136-input": os.environ.get("PRC_EXP136_DATASET_VERSION", "unknown"),
            "prc-exp137-input": os.environ.get("PRC_EXP137_DATASET_VERSION", "unknown"),
        },
        "kernel_code_sha256": os.environ.get("PRC_KERNEL_CODE_SHA256", "unknown"),
        "spec_sha256": inventory["spec_sha256"],
        "input_inventory_sha256": sha256(input_inventory_path),
        "proxy_inventory_sha256": sha256(source_inventory_path),
        "exp130_input_manifest_sha256": sha256(exact_file("input-manifest.json")),
        "code_sha256": {
            "fit_exp137_arm.py": sha256(Path(__file__).resolve()),
            "proxy_features_exp137.py": sha256(Path(PX.__file__).resolve()),
            "exp130_fit.py": sha256(base_code_path),
        },
        "source_hashes": inventory["source_sha256"],
        "proxy_source_hashes": proxy_audit["source_sha256"],
        "folds": FOLDS, "training_months": list(MONTHS),
        "features": {"control": 151, "proxy": 172, "sham": 172},
        "ranking_support": support_records,
        "all_rank_rows_by_fold": all_rank_rows_by_fold,
        "supported_rank_rows_by_fold": supported_rank_rows_by_fold,
        "ranking_coverage_by_fold": ranking_coverage_by_fold,
        "jan_jul_ranking_coverage": float(jan_jul_supported / jan_jul_total),
        "model": {"rounds": 1500, "threads": 6,
                  "seeds": {"lightgbm": 7, "bagging": 8, "feature_fraction": 9,
                            "sham": "13417 + target-fold index"}},
        "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED},
        "base_reproduction": base_checks.to_dict(orient="records"),
        "verified_point_files": int(proxy_audit["point_files_verified"]),
        "verified_point_bytes": int(proxy_audit["point_bytes_verified"]),
        "missing_proxy_source_rows": list(MISSING_PROXY_ROWS.values()),
        "output_sha256": outputs, "elapsed_seconds": time.monotonic() - started,
        "python_version": platform.python_version(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(run_manifest, indent=2, default=float) + "\n")
    print(json.dumps({"state": "arm_complete_no_gate", "arm": arm,
                      "elapsed_seconds": run_manifest["elapsed_seconds"],
                      "metrics": metric_rows}, indent=2), flush=True)


if __name__ == "__main__":
    selected_arm = os.environ.get("PRC_EXP137_ARM")
    if not selected_arm:
        raise RuntimeError("PRC_EXP137_ARM must select exactly one private arm")
    main(selected_arm)
