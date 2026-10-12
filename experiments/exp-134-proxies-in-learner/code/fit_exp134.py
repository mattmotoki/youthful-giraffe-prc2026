#!/usr/bin/env python3
"""One private Kaggle fit for exp-134's frozen row-proxy learner design."""
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

import proxy_features as PX



def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/exp134-output")
ID = "MVT_ID_mvt"
MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 13))
RANK_MONTHS = ("2026-01", "2026-07")
FOLDS = (("july_loo", "2025-07", 1),
         ("december", "2025-12", 2),
         ("january", "2025-01", 3))
FOLD_MONTH = {fold: month for fold, month, _ in FOLDS}
RANK_MONTH = {"july_loo": "2026-07", "december": "2026-01", "january": "2026-01"}
ARM_MODE = {"control": None, "proxy": "proxy", "sham": "sham", "shifted_day": "shifted_day"}
ARM_NAMES = tuple(ARM_MODE)
BOOT_DRAWS = 2000
BOOT_SEED = 20260925
BOOT_METRICS = (
    "gain_blend_control", "gain_blend_proxy", "gain_blend_sham", "gain_blend_shifted_day",
    "proxy_vs_control", "proxy_vs_sham", "proxy_vs_shifted_day",
)
MISSING_PROXY_SOURCE_ROWS: dict[str, dict] = {}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_file(name: str) -> Path:
    return PX.exact_file(name)


def aligned_proxy_features(source: pd.DataFrame, rows: pd.DataFrame,
                           source_months: list[str], context: str) -> pd.DataFrame:
    row_ids = pd.to_numeric(rows[ID], errors="raise").astype("int64").to_numpy()
    missing = ~pd.Index(row_ids).isin(source.index)
    if missing.any():
        missing_ids = np.sort(row_ids[missing]).astype("<i8", copy=False)
        id_hash = hashlib.sha256(missing_ids.tobytes()).hexdigest()
        record_key = f"{len(source_months)}:{','.join(source_months)}:{id_hash}"
        MISSING_PROXY_SOURCE_ROWS[record_key] = {
            "source_months": list(source_months),
            "row_count": int(len(missing_ids)),
            "sorted_id_sha256": id_hash,
        }
        print(f"{context}: proxy source missing for {len(missing_ids)} rows; all 21 values stay missing",
              flush=True)
    selected = source.reindex(row_ids)
    if len(selected) != len(rows):
        raise RuntimeError(f"{context}: proxy rows do not align with model rows")
    selected.index = rows.index
    return selected[PX.Z_COLS].astype("float32")


def base_fit_module():
    path = input_file("fit.py")
    spec = importlib.util.spec_from_file_location("exp130_fit_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen exp-130 fit source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def build_proxy_views(raw_months: dict, fit_months: list[str], all_months: tuple[str, ...],
                      held_seed_index: int) -> dict[str, dict[str, pd.DataFrame]]:
    denominators = PX.fit_denominators(raw_months, fit_months)
    seed = 13417 + held_seed_index
    views = {}
    for month in all_months:
        z, windows = PX.transform_rows(raw_months[month], denominators)
        views[month] = {
            "proxy": PX.row_view(z, windows, mode="proxy", seed=seed),
            "sham": PX.row_view(z, windows, mode="sham", seed=seed),
            "shifted_day": PX.row_view(z, windows, mode="shifted_day", seed=seed),
        }
    return views


def month_feature_z(views: dict[str, dict[str, pd.DataFrame]], months: list[str], mode: str,
                    rows: pd.DataFrame) -> pd.DataFrame:
    pieces = [views[month][mode][[ID, *PX.Z_COLS]] for month in months]
    features = pd.concat(pieces, ignore_index=True)
    if features[ID].duplicated().any():
        raise RuntimeError("proxy feature IDs repeat across requested months")
    indexed = features.set_index(ID)
    return aligned_proxy_features(indexed, rows, months, f"month features {mode}")


def evaluation_features(base, fold: str, month: str, month_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    valid = month_frames[month].drop(columns=["y"], errors="ignore").copy()
    evaluation = base.eval_frame(fold).rename(
        columns={"airport": "eval_airport", "utc_day": "eval_utc_day"})
    valid = valid.merge(evaluation, on=ID, how="left", validate="one_to_one")
    if valid["y_true"].isna().any() or valid["r12"].isna().any():
        raise RuntimeError(f"{fold}: held-fold label or p12 residual is missing")
    if (not valid["airport"].astype(str).eq(valid["eval_airport"].astype(str)).all()
            or not valid["utc_day"].astype(str).eq(valid["eval_utc_day"].astype(str)).all()):
        raise RuntimeError(f"{fold}: saved fold metadata differs from exp-130 input")
    return valid.drop(columns=["eval_airport", "eval_utc_day"])


def model_split(base, D18, F, M, D, month_frames: dict[str, pd.DataFrame],
                views: dict[str, dict[str, pd.DataFrame]], fit_months: list[str],
                score_folds: tuple[str, ...], split_name: str) -> tuple[dict, dict, list[pd.DataFrame]]:
    train_raw = pd.concat([month_frames[m] for m in fit_months], ignore_index=True)
    train_raw = train_raw.loc[D.keep_reachable(train_raw)].copy()
    valid_parts = [evaluation_features(base, fold, FOLD_MONTH[fold], month_frames)
                   for fold in score_folds]
    train_encoded, valid_encoded = D18.add_encodings(train_raw, valid_parts)
    train_ord_mask = train_encoded["ghost"].to_numpy(dtype="float64") < 0.5
    train_ord = train_encoded.loc[train_ord_mask].reset_index(drop=True)
    valid_ord_parts = []
    for valid in valid_encoded:
        ordinary = valid["ghost"].to_numpy(dtype="float64") < 0.5
        valid_ord_parts.append(valid.loc[ordinary].copy())
    valid_ord = pd.concat(valid_ord_parts, ignore_index=True)
    if train_ord.empty or valid_ord.empty:
        raise RuntimeError(f"{split_name}: ordinary learner rows are empty")

    anchor_train = base.anchor_for(train_ord)
    y_train = train_ord["y"].to_numpy("float64")
    if not np.isfinite(anchor_train).all() or not np.isfinite(y_train).all():
        raise RuntimeError(f"{split_name}: a training label or anchor is not finite")
    anchor_valid = base.anchor_for(valid_ord)
    if not np.isfinite(anchor_valid).all():
        raise RuntimeError(f"{split_name}: a validation anchor is not finite")

    base_x_train, base_x_valid, base_columns, categorical = base.feature_frames(
        train_ord, valid_ord, F, M)
    if len(base_columns) != 149:
        raise RuntimeError(f"{split_name}: exp-130 base feature count is {len(base_columns)}, not 149")
    predictions: dict[str, np.ndarray] = {}
    fit_records = []
    for arm, mode in ARM_MODE.items():
        if mode is None:
            x_train, x_valid = base_x_train, base_x_valid
        else:
            z_train = month_feature_z(views, fit_months, mode, train_ord)
            valid_months = [FOLD_MONTH[fold] for fold in score_folds]
            z_valid = month_feature_z(views, valid_months, mode, valid_ord)
            x_train = pd.concat([base_x_train.reset_index(drop=True), z_train.reset_index(drop=True)], axis=1)
            x_valid = pd.concat([base_x_valid.reset_index(drop=True), z_valid.reset_index(drop=True)], axis=1)
            if x_train.shape[1] != 170 or x_valid.shape[1] != 170:
                raise RuntimeError(f"{split_name}/{arm}: expected 170 features")
        prediction, record = base.fit_arm(
            f"{split_name}_{arm}", x_train, x_valid, y_train - anchor_train,
            np.ones(len(train_ord), dtype="float32"), categorical,
            anchor_valid, M, split_name)
        predictions[arm] = prediction
        importance = record["importance"].copy()
        importance["model_split"] = split_name
        importance["arm"] = arm
        fit_records.append({"split": split_name, "arm": arm, **record["info"]})
        fit_records.append(importance)
        if mode is not None:
            del z_train, z_valid, x_train, x_valid
        gc.collect()

    score_frames = {}
    for fold in score_folds:
        part_mask = valid_ord["month"].astype(str).eq(FOLD_MONTH[fold]).to_numpy()
        held = valid_ord.loc[part_mask].copy().reset_index(drop=True)
        score_mask = ~held["obs_has_clock"].astype(bool).to_numpy()
        score_mask &= held["ghost"].to_numpy(dtype="float64") < 0.5
        excluded = set(map(int, D.excluded_ids(fold)))
        score_mask &= ~held[ID].astype("int64").isin(excluded).to_numpy()
        rows = held.loc[score_mask, [ID, "airport", "utc_day", "y_true", "r12", "month"]].copy()
        rows["fold"] = fold
        rows["p12"] = rows["y_true"].to_numpy("float64") - rows["r12"].to_numpy("float64")
        for arm in ARM_NAMES:
            rows[f"pred_{arm}"] = predictions[arm][part_mask][score_mask]
        if rows.empty:
            raise RuntimeError(f"{fold}: clean ordinary no-clock score rows are empty")
        score_frames[fold] = rows.reset_index(drop=True)
    del train_raw, train_encoded, valid_encoded, train_ord, valid_ord, base_x_train, base_x_valid
    gc.collect()
    return score_frames, {"fit_records": fit_records}, []


def rank_maps(base) -> dict[str, dict[str, int]]:
    counts = base.rank_counts()
    maps = {}
    for fold, _, _ in FOLDS:
        part = counts.loc[counts.month.eq(RANK_MONTH[fold])]
        maps[fold] = dict(zip(part.airport.astype(str), part.ranking_rows.astype(int)))
        if not maps[fold] or any(count <= 0 for count in maps[fold].values()):
            raise RuntimeError(f"{fold}: invalid clean ranking counts")
    return maps


def verify_scoring_support(score_frames: dict, ranks: dict) -> dict[str, dict[str, int]]:
    clean_counts = {}
    for fold, rows in score_frames.items():
        values = rows.groupby(rows.airport.astype(str), observed=True).size().to_dict()
        missing = sorted(set(ranks[fold]) - set(values))
        if missing:
            raise RuntimeError(f"{fold}: clean no-clock score lacks ranking support: {missing}")
        clean_counts[fold] = {str(a): int(n) for a, n in values.items()}
    return clean_counts


def row_density_weights(rows: pd.DataFrame, fold: str, ranks: dict,
                        clean_counts: dict) -> np.ndarray:
    airport = rows.airport.astype(str).to_numpy()
    result = np.asarray([
        ranks[fold][a] / clean_counts[fold][a]
        if a in ranks[fold] and a in clean_counts[fold] else 0.0
        for a in airport
    ], dtype="float64")
    if not np.isfinite(result).all() or (result < 0).any():
        raise RuntimeError(f"{fold}: invalid clean ranking row weights")
    return result


def fit_scalar(source_rows: list[tuple[str, pd.DataFrame]], arm: str,
               ranks: dict, clean_counts: dict) -> dict:
    numer = denom = 0.0
    n_rows = 0
    for fold, rows in source_rows:
        weights = row_density_weights(rows, fold, ranks, clean_counts)
        delta = rows[f"pred_{arm}"].to_numpy("float64") - rows.p12.to_numpy("float64")
        target = rows.y_true.to_numpy("float64") - rows.p12.to_numpy("float64")
        good = np.isfinite(delta) & np.isfinite(target) & np.isfinite(weights)
        numer += float(np.sum(weights[good] * delta[good] * target[good]))
        denom += float(np.sum(weights[good] * delta[good] ** 2))
        n_rows += int(good.sum())
    raw = numer / denom if denom > 0 else 0.0
    return {"arm": arm, "weight": float(np.clip(raw, 0.0, 1.0)),
            "unclipped_weight": float(raw), "numerator": numer,
            "denominator": denom, "rows": n_rows}


def attach_blends(score_frames: dict, weights: dict[str, dict[str, dict]]) -> None:
    for fold, rows in score_frames.items():
        for arm in ARM_NAMES:
            w = weights[fold][arm]["weight"]
            rows[f"pred_blend_{arm}"] = rows.p12.to_numpy("float64") + w * (
                rows[f"pred_{arm}"].to_numpy("float64") - rows.p12.to_numpy("float64"))
            rows[f"gain_raw_{arm}"] = (rows.y_true.to_numpy("float64") - rows.p12.to_numpy("float64")) ** 2 - (
                rows.y_true.to_numpy("float64") - rows[f"pred_{arm}"].to_numpy("float64")) ** 2
            rows[f"gain_blend_{arm}"] = (rows.y_true.to_numpy("float64") - rows.p12.to_numpy("float64")) ** 2 - (
                rows.y_true.to_numpy("float64") - rows[f"pred_blend_{arm}"].to_numpy("float64")) ** 2
        loss = {arm: (rows.y_true.to_numpy("float64") - rows[f"pred_blend_{arm}"].to_numpy("float64")) ** 2
                for arm in ARM_NAMES}
        rows["proxy_vs_control"] = loss["control"] - loss["proxy"]
        rows["proxy_vs_sham"] = loss["sham"] - loss["proxy"]
        rows["proxy_vs_shifted_day"] = loss["shifted_day"] - loss["proxy"]


def rank_weighted_mean(rows: pd.DataFrame, metric: str, fold: str,
                       ranks: dict) -> float:
    by_airport = rows.assign(_airport=rows.airport.astype(str)).groupby(
        "_airport", sort=True, observed=True)[metric].mean()
    denominator = sum(ranks[fold].values())
    return float(sum(ranks[fold][airport] * by_airport[airport]
                     for airport in ranks[fold]) / denominator)


def fold_point(rows: pd.DataFrame, fold: str, ranks: dict) -> dict:
    metrics = [f"gain_raw_{arm}" for arm in ARM_NAMES]
    metrics += [f"gain_blend_{arm}" for arm in ARM_NAMES]
    metrics += list(BOOT_METRICS[4:])
    point = {metric: rank_weighted_mean(rows, metric, fold, ranks) for metric in metrics}
    p12_mse = rank_weighted_mean(
        rows.assign(_p12_loss=(rows.y_true - rows.p12) ** 2), "_p12_loss", fold, ranks)
    point["p12_mse_s2_per_rank_row"] = p12_mse
    point["clean_score_rows"] = int(len(rows))
    point["ranking_rows"] = int(sum(ranks[fold].values()))
    return point


def cluster_bootstrap(score_frames: dict, ranks: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(BOOT_SEED)
    draws_by_fold = {}
    points = {}
    for fold, _, _ in FOLDS:
        rows = score_frames[fold]
        total_rank = sum(ranks[fold].values())
        sampled = np.zeros((BOOT_DRAWS, len(BOOT_METRICS)), dtype="float64")
        for airport, rank_n in sorted(ranks[fold].items()):
            cell = rows.loc[rows.airport.astype(str).eq(airport)].copy()
            grouped = cell.groupby("utc_day", sort=True, observed=True)
            day_sums = grouped[list(BOOT_METRICS)].sum().to_numpy("float64")
            day_counts = grouped.size().to_numpy("float64")
            if len(day_sums) == 0:
                raise RuntimeError(f"{fold}/{airport}: no UTC-day clusters")
            draw_ids = rng.integers(0, len(day_sums), size=(BOOT_DRAWS, len(day_sums)))
            sums = day_sums[draw_ids].sum(axis=1)
            counts = day_counts[draw_ids].sum(axis=1)
            sampled += rank_n * (sums / counts[:, None]) / total_rank
        draws_by_fold[fold] = sampled
        points[fold] = {metric: rank_weighted_mean(rows, metric, fold, ranks)
                        for metric in BOOT_METRICS}
    jan, jul = "january", "july_loo"
    jan_n, jul_n = sum(ranks[jan].values()), sum(ranks[jul].values())
    pooled = (draws_by_fold[jan] * jan_n + draws_by_fold[jul] * jul_n) / (jan_n + jul_n)
    pooled_point = {metric: (points[jan][metric] * jan_n + points[jul][metric] * jul_n) / (jan_n + jul_n)
                    for metric in BOOT_METRICS}
    interval_rows = []
    for fold in (*[item[0] for item in FOLDS], "pooled_jan_jul"):
        values = pooled if fold == "pooled_jan_jul" else draws_by_fold[fold]
        p = pooled_point if fold == "pooled_jan_jul" else points[fold]
        for index, metric in enumerate(BOOT_METRICS):
            interval_rows.append({"scope": fold, "metric": metric,
                                  "point": float(p[metric]),
                                  "lower_95": float(np.quantile(values[:, index], 0.025)),
                                  "upper_95": float(np.quantile(values[:, index], 0.975)),
                                  "draws": BOOT_DRAWS, "seed": BOOT_SEED})
    return {"fold_draws": draws_by_fold, "pooled_draws": pooled,
            "fold_points": points, "pooled_point": pooled_point}, interval_rows


def metrics_table(score_frames: dict, ranks: dict) -> pd.DataFrame:
    records = []
    for fold, _, _ in FOLDS:
        record = {"scope": fold, **fold_point(score_frames[fold], fold, ranks)}
        records.append(record)
    jan, jul = "january", "july_loo"
    jan_n, jul_n = sum(ranks[jan].values()), sum(ranks[jul].values())
    pooled_rows = pd.concat([score_frames[jan], score_frames[jul]], ignore_index=True)
    pooled = {}
    metrics = [f"gain_raw_{arm}" for arm in ARM_NAMES]
    metrics += [f"gain_blend_{arm}" for arm in ARM_NAMES]
    metrics += list(BOOT_METRICS[4:])
    for metric in metrics:
        pooled[metric] = (rank_weighted_mean(score_frames[jan], metric, jan, ranks) * jan_n
                          + rank_weighted_mean(score_frames[jul], metric, jul, ranks) * jul_n) / (jan_n + jul_n)
    p12 = (rank_weighted_mean(score_frames[jan].assign(_p12_loss=(score_frames[jan].y_true-score_frames[jan].p12)**2),
                              "_p12_loss", jan, ranks) * jan_n
           + rank_weighted_mean(score_frames[jul].assign(_p12_loss=(score_frames[jul].y_true-score_frames[jul].p12)**2),
                                "_p12_loss", jul, ranks) * jul_n) / (jan_n + jul_n)
    pooled.update({"p12_mse_s2_per_rank_row": p12,
                   "clean_score_rows": len(pooled_rows),
                   "ranking_rows": jan_n + jul_n})
    records.append({"scope": "pooled_jan_jul", **pooled})
    return pd.DataFrame(records)


def tail_table(score_frames: dict) -> pd.DataFrame:
    rows = []
    for fold, scored in score_frames.items():
        tail = scored.y_true.to_numpy("float64") > 3600.0
        for name, mask in (("all", np.ones(len(scored), dtype=bool)),
                           ("y_gt_3600", tail)):
            part = scored.loc[mask]
            record = {"fold": fold, "group": name, "rows": int(len(part)),
                      "row_share": float(len(part) / len(scored)) if len(scored) else 0.0}
            for arm in ARM_NAMES:
                if len(part):
                    record[f"gain_blend_{arm}_s2_per_row"] = float(part[f"gain_blend_{arm}"].mean())
                else:
                    record[f"gain_blend_{arm}_s2_per_row"] = np.nan
            rows.append(record)
    return pd.DataFrame(rows)


def coverage_table(score_frames: dict) -> pd.DataFrame:
    rows = []
    for fold, scored in score_frames.items():
        month = FOLD_MONTH[fold]
        # Coverage is measured on the rows in the frozen score population.
        source = scored.attrs.get("proxy_values")
        if source is None:
            continue
        for column in PX.Z_COLS:
            n = int(np.isfinite(source[column].to_numpy("float64")).sum())
            rows.append({"fold": fold, "month": month, "proxy": column,
                         "clean_no_clock_rows": len(source), "finite_rows": n,
                         "finite_rate": n / len(source) if len(source) else 0.0})
    return pd.DataFrame(rows)


def add_coverage(score_frames: dict, main_views: dict) -> pd.DataFrame:
    records = []
    for fold, rows in score_frames.items():
        month = FOLD_MONTH[fold]
        z = main_views[month]["proxy"].set_index(ID).reindex(rows[ID].to_numpy("int64"))
        for column in PX.Z_COLS:
            n = int(np.isfinite(z[column].to_numpy("float64")).sum())
            records.append({"fold": fold, "month": month, "proxy": column,
                            "clean_no_clock_rows": len(z), "finite_rows": n,
                            "finite_rate": n / len(z) if len(z) else 0.0})
    return pd.DataFrame(records)


def proxy_vs_p12_gate(intervals: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    lookup = {(row.scope, row.metric): row for row in intervals.itertuples(index=False)}
    folds = [item[0] for item in FOLDS]
    positive_points = all(float(metrics.loc[metrics.scope.eq(fold), "gain_blend_proxy"].iat[0]) > 0
                          for fold in folds)
    required_scopes = [*folds, "pooled_jan_jul"]
    p12_clear = all(lookup[(scope, "gain_blend_proxy")].lower_95 > 0
                    for scope in required_scopes)
    sham_clear = lookup[("pooled_jan_jul", "proxy_vs_sham")].lower_95 > 0
    shifted_clear = lookup[("pooled_jan_jul", "proxy_vs_shifted_day")].lower_95 > 0
    point_beats = all(
        float(metrics.loc[metrics.scope.eq(scope), "gain_blend_proxy"].iat[0])
        > float(metrics.loc[metrics.scope.eq(scope), other].iat[0])
        for scope in [*folds, "pooled_jan_jul"]
        for other in ("gain_blend_sham", "gain_blend_shifted_day")
    )
    return {"positive_gain_each_fold": positive_points,
            "proxy_interval_lower_gt_zero_each_fold_and_pool": p12_clear,
            "proxy_point_beats_sham_and_shifted_each_fold_and_pool": point_beats,
            "proxy_minus_sham_pooled_interval_lower_gt_zero": sham_clear,
            "proxy_minus_shifted_day_pooled_interval_lower_gt_zero": shifted_clear,
            "pass": bool(positive_points and p12_clear and point_beats and sham_clear and shifted_clear),
            "verdict": "PASS" if positive_points and p12_clear and point_beats and sham_clear and shifted_clear else "FAIL"}


def final_refit(base, D18, F, M, D, month_frames: dict, raw_months: dict,
                ranks: dict, weights: dict, audits: dict) -> dict:
    train_raw = pd.concat([month_frames[month] for month in MONTHS], ignore_index=True)
    train_raw = train_raw.loc[D.keep_reachable(train_raw)].copy()
    rank_raw = [month_frames[month].drop(columns=["y"], errors="ignore").copy()
                for month in RANK_MONTHS]
    train_encoded, rank_encoded = D18.add_encodings(train_raw, rank_raw)
    train_ordinary = train_encoded["ghost"].to_numpy(dtype="float64") < 0.5
    rank_ord_parts = [frame.loc[frame["ghost"].to_numpy(dtype="float64") < 0.5].copy()
                      for frame in rank_encoded]
    rank_ord = pd.concat(rank_ord_parts, ignore_index=True)
    train_ord = train_encoded.loc[train_ordinary].reset_index(drop=True)

    denoms = PX.fit_denominators(raw_months, list(MONTHS))
    rank_views = {}
    for month in RANK_MONTHS:
        z, windows = PX.transform_rows(raw_months[month], denoms)
        rank_views[month] = {"proxy": PX.row_view(z, windows, mode="proxy", seed=0)}
    # The real training feature tables use the same final 12-month normalizers.
    train_z_source = pd.concat([PX.transform_rows(raw_months[month], denoms)[0][[ID, *PX.Z_COLS]]
                                for month in MONTHS], ignore_index=True).set_index(ID)
    train_z = aligned_proxy_features(train_z_source, train_ord, list(MONTHS), "final training features")
    rank_z_parts = [rank_views[month]["proxy"][[ID, *PX.Z_COLS]] for month in RANK_MONTHS]
    rank_z_source = pd.concat(rank_z_parts, ignore_index=True).set_index(ID)
    rank_z = aligned_proxy_features(rank_z_source, rank_ord, list(RANK_MONTHS), "final ranking features")

    x_train, x_rank, columns, categorical = base.feature_frames(train_ord, rank_ord, F, M)
    if len(columns) != 149:
        raise RuntimeError("final exp-130 base feature count is not 149")
    x_train = pd.concat([x_train.reset_index(drop=True), train_z.reset_index(drop=True)], axis=1)
    x_rank = pd.concat([x_rank.reset_index(drop=True), rank_z.reset_index(drop=True)], axis=1)
    if x_train.shape[1] != 170 or x_rank.shape[1] != 170:
        raise RuntimeError("final proxy learner does not have 170 columns")
    anchor_train = base.anchor_for(train_ord)
    anchor_rank = base.anchor_for(rank_ord)
    y = train_ord["y"].to_numpy("float64")
    if not np.isfinite(anchor_train).all() or not np.isfinite(y).all() or not np.isfinite(anchor_rank).all():
        raise RuntimeError("final fit has a non-finite label or anchor")
    prediction, result = base.fit_arm(
        "final_proxy", x_train, x_rank, y - anchor_train,
        np.ones(len(train_ord), dtype="float32"), categorical, anchor_rank, M, "final_rank")
    pred_by_id = dict(zip(rank_ord[ID].astype("int64"), prediction.astype("float64")))

    rank_meta = pd.concat([pd.read_parquet(input_file(f"rank_meta_{m}.parquet")) for m in RANK_MONTHS],
                          ignore_index=True)
    rank_meta[ID] = pd.to_numeric(rank_meta[ID], errors="raise").astype("int64")
    if rank_meta[ID].duplicated().any():
        raise RuntimeError("ranking metadata IDs are not unique")
    class_by_id = rank_meta.set_index(ID)["rank_class"]
    if class_by_id.index.duplicated().any():
        raise RuntimeError("ranking class IDs are not unique")
    submission_ids = pd.read_parquet(input_file("submitting_ids.parquet"))
    v22 = pd.read_parquet(input_file("v22_submission.parquet"))
    if not np.array_equal(submission_ids[ID].to_numpy("int64"), v22[ID].to_numpy("int64")):
        raise RuntimeError("v22 submission IDs differ from frozen submission order")
    if submission_ids[ID].duplicated().any() or submission_ids.row_order.duplicated().any():
        raise RuntimeError("submitting ID table has duplicate IDs or row order")
    v22_by_id = v22.set_index(ID).p22.astype("float64")
    rank_by_id = rank_meta.set_index(ID)
    submission_classes = class_by_id.reindex(submission_ids[ID].to_numpy("int64"))
    if submission_classes.isna().any():
        raise RuntimeError("submission order includes an ID without a ranking class")
    rank_no_clock = submission_classes.eq("no-clock_ordinary").to_numpy()
    delta = np.zeros(len(submission_ids), dtype="float32")
    w_rank = weights["rank"]["proxy"]["weight"]
    for i, mvt_id in enumerate(submission_ids[ID].to_numpy("int64")):
        if mvt_id not in rank_by_id.index:
            raise RuntimeError(f"submission ID {mvt_id} has no ranking class")
        if rank_by_id.at[mvt_id, "rank_class"] == "no-clock_ordinary":
            if mvt_id not in pred_by_id:
                raise RuntimeError(f"ordinary no-clock ID {mvt_id} has no final learner prediction")
            p12 = float(v22_by_id.loc[mvt_id])
            delta[i] = float(p12 + w_rank * (pred_by_id[mvt_id] - p12) - p12)
    if len(delta) != len(submission_ids) or np.count_nonzero(delta[~rank_no_clock]) != 0:
        raise RuntimeError("ranking delta changed rows outside ordinary no-clock class")
    path = OUTPUT / "ranking_delta.npz"
    np.savez_compressed(path, row_order=submission_ids.row_order.to_numpy("int64"),
                        MVT_ID_mvt=submission_ids[ID].to_numpy("int64"), delta=delta)
    return {"final_fit": result["info"], "w_rank": w_rank,
            "ranking_rows_updated": int(rank_no_clock.sum()),
            "delta_nonzero_rows": int(np.count_nonzero(delta)),
            "delta_sha256": file_hash(path)}


def main() -> None:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_inventory_path = input_file("source_inventory.json")
    source_inventory = json.loads(source_inventory_path.read_text())
    if (source_inventory.get("ranking_target_read") is not False
            or source_inventory.get("departure_target_column_read") is not False
            or source_inventory.get("taxitime_read_for") != "ARR rows only"):
        raise RuntimeError("exp-134 input inventory does not prove label-safe proxy inputs")

    # Build and verify every input-only proxy before opening exp-130 labels.
    raw_months, proxy_audit = PX.build_all(MONTHS, RANK_MONTHS)
    print("all proxy source tables built before any exp-130 label read", flush=True)

    base, base_code_path = base_fit_module()
    base_manifest_path = input_file("input-manifest.json")
    base_manifest = json.loads(base_manifest_path.read_text())
    if base_manifest.get("ranking_target_read") is not False:
        raise RuntimeError("exp-130 input manifest does not prove ranking labels were withheld")
    preflight = json.loads(input_file("preflight.json").read_text())
    if not preflight.get("v22_baseline_passed"):
        raise RuntimeError("exp-130 saved v22 fold baseline did not pass preflight")

    D18, F, M, D = base.setup_modules()
    month_frames = {month: base.month_frame(month) for month in MONTHS}
    month_frames.update({month: base.month_frame(month) for month in RANK_MONTHS})
    ranks = rank_maps(base)

    main_scores = {}
    main_fit_records = []
    for fold, valid_month, fold_index in FOLDS:
        print(f"main OOF start {fold}", flush=True)
        fit_months = [month for month in MONTHS if month != valid_month]
        views = build_proxy_views(raw_months, fit_months, MONTHS, fold_index)
        score, records, _ = model_split(
            base, D18, F, M, D, month_frames, views, fit_months,
            (fold,), f"main_{fold}")
        main_scores.update(score)
        main_fit_records.extend(records["fit_records"])
        del views
        gc.collect()
        print(f"main OOF end {fold}: {len(score[fold]):,} clean no-clock rows", flush=True)

    clean_counts = verify_scoring_support(main_scores, ranks)
    calibration = {}
    calibration_records = []
    fold_names = [entry[0] for entry in FOLDS]
    for i, first in enumerate(fold_names):
        for second in fold_names[i + 1:]:
            pair = (first, second)
            held_index = next(index for fold, _, index in FOLDS if fold == first)
            pair_months = tuple(FOLD_MONTH[fold] for fold in pair)
            fit_months = [month for month in MONTHS if month not in pair_months]
            print(f"cross-fit calibration start {first}/{second}", flush=True)
            views = build_proxy_views(raw_months, fit_months, MONTHS, held_index)
            score, records, _ = model_split(
                base, D18, F, M, D, month_frames, views, fit_months,
                pair, f"cal_{first}_{second}")
            calibration[(first, second)] = score
            calibration_records.extend(records["fit_records"])
            del views
            gc.collect()

    weights = {}
    weight_rows = []
    for fold in fold_names:
        source_folds = [source for source in fold_names if source != fold]
        source_scores = {}
        for source in source_folds:
            pair_key = tuple(sorted((fold, source), key=fold_names.index))
            source_scores[source] = calibration[pair_key][source]
        weights[fold] = {}
        for arm in ARM_NAMES:
            fit_record = fit_scalar([(source, source_scores[source]) for source in source_folds],
                                    arm, ranks, clean_counts)
            weights[fold][arm] = fit_record
            weight_rows.append({"target_fold": fold, "source_folds": ",".join(source_folds), **fit_record})
    pooled_weight_sources = [(fold, main_scores[fold]) for fold in ("january", "july_loo")]
    weights["rank"] = {}
    for arm in ARM_NAMES:
        fit_record = fit_scalar(pooled_weight_sources, arm, ranks, clean_counts)
        weights["rank"][arm] = fit_record
        weight_rows.append({"target_fold": "rank", "source_folds": "january,july_loo", **fit_record})

    attach_blends(main_scores, weights)
    metrics = metrics_table(main_scores, ranks)
    bootstrap, intervals = cluster_bootstrap(main_scores, ranks)
    gate = proxy_vs_p12_gate(pd.DataFrame(intervals), metrics)
    interval_frame = pd.DataFrame(intervals)
    metric_lookup = {(row.scope, row.metric): row for row in interval_frame.itertuples(index=False)}
    weights_frame = pd.DataFrame(weight_rows)
    for fold, rows in main_scores.items():
        rows.to_parquet(OUTPUT / f"predictions_{fold}.parquet", index=False, compression="zstd")
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    interval_frame.to_csv(OUTPUT / "intervals.csv", index=False)
    weights_frame.to_csv(OUTPUT / "weights.csv", index=False)

    main_views_coverage = None
    # The main validation proxy coverage uses the matching held-fold normalizers.
    coverage_records = []
    for fold, valid_month, fold_index in FOLDS:
        fit_months = [month for month in MONTHS if month != valid_month]
        denom = PX.fit_denominators(raw_months, fit_months)
        z, _ = PX.transform_rows(raw_months[valid_month], denom)
        scored = main_scores[fold]
        aligned = z.set_index(ID).reindex(scored[ID].astype("int64").to_numpy())
        for column in PX.Z_COLS:
            n = int(np.isfinite(aligned[column].to_numpy("float64")).sum())
            coverage_records.append({"fold": fold, "month": valid_month, "proxy": column,
                                     "clean_no_clock_rows": len(aligned), "finite_rows": n,
                                     "finite_rate": n / len(aligned) if len(aligned) else 0.0})
        del z, aligned, denom
    pd.DataFrame(coverage_records).to_csv(OUTPUT / "proxy_coverage.csv", index=False)
    tail_table(main_scores).to_csv(OUTPUT / "tail_metrics.csv", index=False)

    pooled_metrics = metrics.loc[metrics.scope.eq("pooled_jan_jul")].iloc[0]
    baseline = float(pooled_metrics.p12_mse_s2_per_rank_row)
    gain = float(pooled_metrics.gain_blend_proxy)
    census = pd.read_csv(input_file("census.csv"))
    sse = pd.to_numeric(census.estimated_ranking_sse_clean_s2, errors="coerce")
    total_sse = float(sse.sum())
    no_clock_sse = float(sse.loc[census["class"].eq("no-clock_ordinary")].sum())
    share = no_clock_sse / total_sse if total_sse > 0 else np.nan
    frac = gain / baseline if baseline > 0 else np.nan
    product = share * frac if np.isfinite(share) and np.isfinite(frac) else np.nan
    price_domain = bool(np.isfinite(product) and 1.0 - product >= 0.0)
    board_price = float(245.0207 * (1.0 - np.sqrt(1.0 - product))) if price_domain else None

    repair = None
    if gate["verdict"] == "FAIL":
        for fold, rows in main_scores.items():
            rows["pred_repair_50_50"] = rows.p12 + 0.5 * (rows.pred_proxy - rows.p12)
            rows["gain_repair_50_50"] = (rows.y_true - rows.p12) ** 2 - (rows.y_true - rows.pred_repair_50_50) ** 2
        repair_by_fold = {fold: rank_weighted_mean(rows, "gain_repair_50_50", fold, ranks)
                          for fold, rows in main_scores.items()}
        jan_n, jul_n = sum(ranks["january"].values()), sum(ranks["july_loo"].values())
        repair_pooled = (repair_by_fold["january"] * jan_n + repair_by_fold["july_loo"] * jul_n) / (jan_n + jul_n)
        repair = {"method": "fixed 50/50 proxy prediction and p12 using saved OOF predictions",
                  "gain_s2_per_rank_row": repair_by_fold,
                  "pooled_january_july_s2_per_rank_row": float(repair_pooled),
                  "additional_fit": False, "changes_gate": False, "ranking_vector": False}
        for fold, rows in main_scores.items():
            rows.to_parquet(OUTPUT / f"predictions_{fold}.parquet", index=False, compression="zstd")

    final_result = None
    if gate["verdict"] == "PASS":
        final_result = final_refit(base, D18, F, M, D, month_frames,
                                   raw_months, ranks, weights, proxy_audit)
    metrics = metrics_table(main_scores, ranks)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    fit_records = main_fit_records + calibration_records
    importance_records = [record for record in fit_records if isinstance(record, pd.DataFrame)]
    fit_info = [record for record in fit_records if isinstance(record, dict)]
    if importance_records:
        pd.concat(importance_records, ignore_index=True).to_csv(
            OUTPUT / "feature_importance.csv", index=False)
    pd.DataFrame(fit_info).to_csv(OUTPUT / "fit_info.csv", index=False)

    results = {
        "experiment": "exp-134-proxies-in-learner",
        "verdict": gate["verdict"],
        "gate": gate,
        "source_ledger_rows": ["exp-133 step 2", "exp-130 step 2"],
        "source_result": "The 21 proxy contrast gained 306 s2 per clean ranking row in July and 118 in January; the earlier window correction failed D107.",
        "exp130_control": "The annual unweighted learner's fixed 50/50 p12 blend was MAYBE; this unit uses an OOF scalar.",
        "cross_fitted_weights": {fold: weights[fold] for fold in fold_names},
        "rank_weight": weights["rank"]["proxy"],
        "metrics": metrics.to_dict(orient="records"),
        "intervals": intervals,
        "supported_no_clock_share_v22_sse": share,
        "pooled_proxy_fractional_mse_gain": frac,
        "board_price_seconds_v23": board_price,
        "price_domain_ok": price_domain,
        "one_repair": repair,
        "final_refit": final_result,
        "proxy_feature_coverage": coverage_records,
        "proxy_build_audit": proxy_audit,
        "rows_without_proxy_source": list(MISSING_PROXY_SOURCE_ROWS.values()),
        "fit_info": fit_info,
        "lightgbm_version": __import__("lightgbm").__version__,
        "python_version": platform.python_version(),
        "ranking_target_read": False,
        "spec_commit": os.environ.get("PRC_SPEC_COMMIT", "441bacb3de357aaf398cde6148974fee7f400d1c"),
        "spec_sha256": source_inventory.get("spec_sha256"),
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    code_path = input_file("fit_exp134.py")
    proxy_path = input_file("proxy_features.py")
    outputs = {str(path.relative_to(OUTPUT)): file_hash(path)
               for path in sorted(OUTPUT.rglob("*")) if path.is_file()}
    run_manifest = {
        "experiment": "exp-134-proxies-in-learner",
        "kernel_slug": os.environ.get("PRC_KERNEL_SLUG", "unknown"),
        "kernel_version": os.environ.get("PRC_KERNEL_VERSION", "unknown"),
        "dataset_sources": {
            "exp130": {"id": f"{kaggle_user()}/prc-exp130-clean-input", "version": "1",
                        "input_manifest_sha256": file_hash(base_manifest_path)},
            "exp134": {"id": f"{kaggle_user()}/prc-exp134-input",
                        "version": os.environ.get("PRC_DATASET_VERSION", "unknown"),
                        "source_inventory_sha256": file_hash(source_inventory_path)},
        },
        "spec_commit": results["spec_commit"],
        "spec_sha256": results["spec_sha256"],
        "code_sha256": {"fit_exp134.py": file_hash(code_path),
                        "proxy_features.py": file_hash(proxy_path),
                        "kernel.py": os.environ.get("PRC_KERNEL_CODE_SHA256", "external manifest")},
        "source_hashes": source_inventory.get("source_sha256", {}),
        "point_manifest_sha256": proxy_audit["source_hashes"]["points_expected.json"],
        "verified_point_files": proxy_audit["point_files_verified"],
        "verified_point_bytes": proxy_audit["point_bytes_verified"],
        "proxy_columns": PX.Z_COLS,
        "oof_folds": FOLDS,
        "rounds": 1500,
        "threads": 6,
        "seeds": {"lightgbm": 7, "bagging": 8, "feature_fraction": 9,
                  "sham": "13417 + target fold index", "bootstrap": BOOT_SEED},
        "output_sha256": outputs,
        "results": {"verdict": gate["verdict"], "board_price_seconds_v23": board_price,
                    "ranking_vector_written": final_result is not None},
        "rows_without_proxy_source": list(MISSING_PROXY_SOURCE_ROWS.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(run_manifest, indent=2, default=float) + "\n")
    print(json.dumps({"verdict": gate["verdict"], "gate": gate,
                      "board_price_seconds_v23": board_price,
                      "final_refit": final_result,
                      "elapsed_seconds": run_manifest["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
