#!/usr/bin/env python3
"""Run the frozen exp-130 fold fits and, only on a pass, make a v22 delta."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/exp130-output")
ID = "MVT_ID_mvt"
RANK_MONTHS = ("2026-01", "2026-07")
FOLDS = (("july_loo", "2025-07", 1),
         ("december", "2025-12", 2),
         ("january", "2025-01", 3))
SEASON = {**{m: "winter" for m in ("2025-10", "2025-11", "2025-12",
                                    "2025-01", "2025-02", "2025-03")},
          **{m: "summer" for m in ("2025-04", "2025-05", "2025-06",
                                    "2025-07", "2025-08", "2025-09")}}
RANK_SEASON = {"2026-01": "winter", "2026-07": "summer"}
REQUIRED_CELLS = {
    ("july_loo", "LIRF"),
    ("january", "LTFM"), ("july_loo", "LTFM"),
    ("january", "LFPG"), ("july_loo", "LFPG"),
    ("january", "EDDF"), ("july_loo", "EDDF"),
}
ARM_NAMES = ("control", "primary", "sham", "placebo")
BASELINE_SEED = 7
BOOT_SEED = 20260925
BOOT_DRAWS = 2000
SEEDS = {
    "lightgbm": 7,
    "bagging": 8,
    "feature_fraction": 9,
    "feature_sham": {fold: 13017 + index for fold, _, index in FOLDS},
    "weight_placebo": {fold: 13007 + index for fold, _, index in FOLDS},
    "bootstrap": BOOT_SEED,
}


def locate_file(name: str) -> Path:
    matches = sorted(p for p in INPUT.rglob(name) if p.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"exact-file lookup for {name!r} found {len(matches)} files: {matches}")
    return matches[0]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_modules():
    dataset_path = locate_file("dataset.py")
    data_path = locate_file("data.py")
    model_path = locate_file("model.py")
    code_path = dataset_path.parent
    prc_path = data_path.parent.parent
    sys.path.insert(0, str(prc_path))
    sys.path.insert(0, str(code_path))
    D18 = importlib.import_module("dataset")
    F = importlib.import_module("features")
    M = importlib.import_module("model")
    D = importlib.import_module("prc.data")
    return D18, F, M, D


def month_frame(month: str) -> pd.DataFrame:
    path = locate_file(f"month_{month}.parquet")
    frame = pd.read_parquet(path)
    if "month" not in frame or not frame["month"].eq(month).all():
        raise RuntimeError(f"wrong month content in {path}")
    if ID not in frame or frame[ID].duplicated().any():
        raise RuntimeError(f"missing or duplicate IDs in {path}")
    return frame


def eval_frame(fold: str) -> pd.DataFrame:
    frame = pd.read_parquet(locate_file(f"eval_{fold}.parquet"))
    if ID not in frame or frame[ID].duplicated().any():
        raise RuntimeError(f"{fold}: missing or duplicate evaluation IDs")
    return frame


def anchor_for(frame: pd.DataFrame) -> np.ndarray:
    anchor = frame["r_aobt"].to_numpy(dtype="float64", copy=True)
    for name in ("te_y_standrwy", "te_y_rwy", "te_y_apt"):
        fallback = frame[name].to_numpy(dtype="float64")
        use = ~np.isfinite(anchor) & np.isfinite(fallback)
        anchor[use] = fallback[use]
    return anchor


def season_of(month: str) -> str:
    try:
        return SEASON[month]
    except KeyError as exc:
        raise RuntimeError(f"no frozen season mapping for {month}") from exc


def normalize_weights(meta: pd.DataFrame, rank_counts: pd.DataFrame,
                      placebo_seed: int | None = None) -> tuple[np.ndarray, dict, dict]:
    """Density-ratio weights p(airport,season) / q(airport,season), mean one."""
    m = meta[["airport", "month"]].copy()
    m["airport"] = m["airport"].astype(str)
    m["season"] = m["month"].map(season_of)
    q_counts = m.groupby(["airport", "season"], observed=True).size().to_dict()
    n_train = len(m)
    q = {key: int(count) / n_train for key, count in q_counts.items()}

    rc = rank_counts.copy()
    rc["season"] = rc["month"].map(RANK_SEASON)
    if rc["season"].isna().any():
        raise RuntimeError("rank count input contains an unexpected month")
    total_rank = int(rc["ranking_rows"].sum())
    p_counts = rc.groupby(["airport", "season"], observed=True)["ranking_rows"].sum().to_dict()
    p = {key: int(count) / total_rank for key, count in p_counts.items()}

    if placebo_seed is not None:
        rng = np.random.default_rng(placebo_seed)
        placebo_p = dict(p)
        for season in ("winter", "summer"):
            airports = sorted(a for (a, s) in q if s == season)
            if len(airports) < 2:
                raise RuntimeError(f"weight placebo needs at least two {season} airports")
            values = np.array([p.get((a, season), 0.0) for a in airports], dtype="float64")
            # Sattolo's cycle guarantees a non-identity permutation.
            order = list(range(len(airports)))
            for i in range(len(order) - 1, 0, -1):
                j = int(rng.integers(0, i))
                order[i], order[j] = order[j], order[i]
            for i, airport in enumerate(airports):
                placebo_p[(airport, season)] = float(values[order[i]])
        p = placebo_p

    unsupported = [key for key, share in p.items() if share > 0 and q.get(key, 0.0) <= 0]
    if unsupported:
        raise RuntimeError(f"positive ranking exposure has no eligible training support: {unsupported}")
    keys = list(zip(m["airport"], m["season"]))
    weights = np.asarray([p.get(key, 0.0) / q[key] for key in keys], dtype="float64")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.mean() <= 0:
        raise RuntimeError("invalid exposure weights")
    weights /= weights.mean()
    return weights.astype("float32"), p, q


def map_lirf_categories(train: pd.DataFrame, valid: pd.DataFrame,
                        columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    valid = valid.copy()
    for column in columns:
        if column not in train or column not in valid:
            raise RuntimeError(f"missing LIRF categorical feature {column}")
        train_text = train[column].astype("string")
        valid_text = valid[column].astype("string")
        values = sorted(train_text.dropna().unique().tolist())
        mapping = {value: index for index, value in enumerate(values)}
        train[column] = train_text.map(mapping).fillna(-1).astype("int32")
        valid[column] = valid_text.map(mapping).fillna(-1).astype("int32")
    return train, valid


def feature_frames(train: pd.DataFrame, valid: pd.DataFrame,
                   F, M, map_categories: bool = True
                   ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    lirf_cats = ["lirf_flight_prefix", "lirf_stand_group"]
    if map_categories:
        train, valid = map_lirf_categories(train, valid, lirf_cats)
    drop = set(M.DROP) | set(M.HIGH_CARD) | {
        "airport", "utc_day", "obs_has_clock", "y_true", "r12", "clean_keep",
    }
    features = [column for column in train.columns if column not in drop]
    if not features or set(features) != {c for c in valid.columns if c not in drop}:
        raise RuntimeError("train and validation feature columns do not match")
    categorical = [c for c in list(F.CAT_COLS) + lirf_cats if c in features]
    if not set(categorical).issubset(features):
        raise RuntimeError("categorical column list is not in the model feature set")

    def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        values = {}
        for column in features:
            series = frame[column]
            if column in categorical:
                codes = pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy()
                if not np.isfinite(codes).all():
                    raise RuntimeError(f"non-finite category codes in {column}")
                values[column] = codes.astype("int32", copy=False)
            else:
                number = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float32")
                if np.isinf(number).any():
                    raise RuntimeError(f"infinite model feature in {column}")
                values[column] = number
        return pd.DataFrame(values, index=frame.index)

    return numeric_frame(train), numeric_frame(valid), features, categorical


def permute_lirf(train_x: pd.DataFrame, valid_x: pd.DataFrame,
                 train_meta: pd.DataFrame, valid_meta: pd.DataFrame,
                 columns: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_sham = train_x.copy()
    valid_sham = valid_x.copy()
    for frame, meta in ((train_sham, train_meta), (valid_sham, valid_meta)):
        group_positions = meta.groupby(["airport", "month"], sort=True, observed=True).indices
        for column in columns:
            values = frame[column].to_numpy(copy=True)
            for positions in group_positions.values():
                if len(positions) > 1:
                    values[positions] = values[positions][rng.permutation(len(positions))]
            frame[column] = values
    return train_sham, valid_sham


def fit_arm(name: str, train_x: pd.DataFrame, valid_x: pd.DataFrame,
            residual: np.ndarray, weights: np.ndarray, categorical: list[str],
            anchor_valid: np.ndarray, M, fold: str) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb

    params = dict(M.MAIN_PARAMS)
    params.update(seed=BASELINE_SEED,
                  bagging_seed=BASELINE_SEED + 1,
                  feature_fraction_seed=BASELINE_SEED + 2)
    if int(params.get("num_threads", 6)) != 6:
        raise RuntimeError("exp-018 CPU thread setting changed")
    print(f"fit start fold={fold} arm={name} rows={len(train_x):,} "
          f"features={train_x.shape[1]}", flush=True)
    started = time.monotonic()
    ds = lgb.Dataset(train_x, label=residual, weight=weights,
                     categorical_feature=categorical, free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=1500)
    correction = booster.predict(valid_x, num_threads=6)
    prediction = np.maximum(30.0, anchor_valid + correction)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{fold} {name}: model emitted a non-finite prediction")
    importance = pd.DataFrame({
        "fold": fold,
        "arm": name,
        "feature": train_x.columns,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    })
    info = {"fit_seconds": time.monotonic() - started,
            "rounds": int(booster.current_iteration()),
            "train_rows": int(len(train_x)),
            "features": int(train_x.shape[1])}
    print(f"fit end fold={fold} arm={name} seconds={info['fit_seconds']:.1f} "
          f"rounds={info['rounds']}", flush=True)
    del booster, ds
    return prediction.astype("float64"), {"info": info, "importance": importance}


def make_prediction_rows(fold: str, frame: pd.DataFrame,
                         predictions: dict[str, np.ndarray], D) -> pd.DataFrame:
    month = dict((name, valid_month) for name, valid_month, _ in FOLDS)[fold]
    rank_month = "2026-07" if month == "2025-07" else "2026-01"
    no_clock = ~frame["obs_has_clock"].astype(bool).to_numpy()
    ordinary = frame["ghost"].to_numpy(dtype="float64") < 0.5
    keep = no_clock & ordinary
    rows = frame.loc[keep, [ID, "airport", "utc_day", "y_true", "r12"]].copy()
    rows["fold"] = fold
    rows["clean"] = ~rows[ID].isin(D.excluded_ids(fold)).to_numpy()
    rows["rank_month"] = rank_month
    rows["p12_pred"] = rows["y_true"].to_numpy("float64") - rows["r12"].to_numpy("float64")
    for arm, values in predictions.items():
        rows[f"pred_{arm}"] = values[keep]
        rows[f"pred_blend_{arm}"] = 0.5 * (values[keep] + rows["p12_pred"].to_numpy())
    if rows.empty:
        raise RuntimeError(f"{fold}: no ordinary no-clock score rows")
    return rows


def prediction_columns(rows: pd.DataFrame) -> dict[str, str]:
    return {
        **{arm: f"pred_{arm}" for arm in ARM_NAMES},
        **{f"blend_{arm}": f"pred_blend_{arm}" for arm in ARM_NAMES},
    }


def rank_counts() -> pd.DataFrame:
    frame = pd.read_csv(locate_file("rank_counts.csv"))
    required = {"month", "airport", "ranking_rows"}
    if not required.issubset(frame.columns):
        raise RuntimeError("rank_counts.csv has the wrong schema")
    frame["month"] = frame["month"].astype(str)
    frame["airport"] = frame["airport"].astype(str)
    if frame["ranking_rows"].le(0).any():
        raise RuntimeError("rank_counts.csv contains a non-positive cell")
    return frame


def cell_metrics(rows: pd.DataFrame, columns: dict[str, str],
                 rank_map: dict[str, int], fold: str, view: str) -> list[dict]:
    view_rows = rows if view == "all" else rows.loc[rows["clean"]]
    results = []
    for airport, rank_n in sorted(rank_map.items()):
        cell = view_rows.loc[view_rows["airport"].astype(str).eq(airport)]
        if cell.empty:
            raise RuntimeError(f"{fold}/{view}/{airport}: no validation support")
        baseline = float(np.mean((cell["y_true"] - cell["p12_pred"]) ** 2))
        common = {"fold": fold, "view": view, "airport": airport,
                  "ranking_rows": int(rank_n), "validation_rows": int(len(cell)),
                  "baseline_p12_mse_s2": baseline}
        for arm, column in columns.items():
            model_mse = float(np.mean((cell["y_true"] - cell[column]) ** 2))
            results.append({**common, "arm": arm, "arm_mse_s2": model_mse,
                            "gain_s2_per_rank_row": baseline - model_mse})
    return results


def weighted_rows(cell_rows: list[dict], fold: str, view: str,
                  pooled: bool = False) -> list[dict]:
    scope = "pooled_jan_jul" if pooled else fold
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in cell_rows:
        grouped.setdefault((row["view"], row["arm"]), []).append(row)
    out = []
    for (rview, arm), rows in grouped.items():
        if rview != view:
            continue
        denominator = sum(row["ranking_rows"] for row in rows)
        baseline = sum(row["ranking_rows"] * row["baseline_p12_mse_s2"] for row in rows) / denominator
        model = sum(row["ranking_rows"] * row["arm_mse_s2"] for row in rows) / denominator
        out.append({"fold": scope, "view": view, "arm": arm,
                    "ranking_rows": denominator,
                    "validation_rows": sum(row["validation_rows"] for row in rows),
                    "baseline_p12_mse_s2": baseline, "arm_mse_s2": model,
                    "gain_s2_per_rank_row": baseline - model})
    return out


def diff_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["airport", "utc_day", "y_true", "p12_pred"]].copy()
    base_loss = (out["y_true"].to_numpy() - out["p12_pred"].to_numpy()) ** 2
    losses = {}
    for arm, col in prediction_columns(rows).items():
        losses[arm] = (rows["y_true"].to_numpy() - rows[col].to_numpy()) ** 2
        out[f"gain_{arm}"] = base_loss - losses[arm]
    losses["control"] = losses["control"]
    out["primary_vs_control"] = losses["control"] - losses["primary"]
    out["primary_vs_sham"] = losses["sham"] - losses["primary"]
    out["primary_vs_placebo"] = losses["placebo"] - losses["primary"]
    return out


BOOT_METRICS = tuple([f"gain_{arm}" for arm in ARM_NAMES]
                     + [f"gain_blend_{arm}" for arm in ARM_NAMES]
                     + ["primary_vs_control", "primary_vs_sham", "primary_vs_placebo"])


def bootstrap_for_view(fold_rows: dict[str, pd.DataFrame],
                       rank_maps: dict[str, dict[str, int]], view: str) -> list[dict]:
    rng = np.random.default_rng(BOOT_SEED)
    sampled: dict[str, np.ndarray] = {
        fold: np.zeros((BOOT_DRAWS, len(BOOT_METRICS)), dtype="float64")
        for fold, _, _ in FOLDS}
    folds_meta = {fold: rows.loc[rows["clean"]] if view == "clean" else rows
                  for fold, rows in fold_rows.items()}
    for fold, _, _ in FOLDS:
        scored = diff_columns(folds_meta[fold])
        denominator = sum(rank_maps[fold].values())
        for airport, rank_n in rank_maps[fold].items():
            cell = scored.loc[scored["airport"].astype(str).eq(airport)]
            if cell.empty:
                raise RuntimeError(f"bootstrap {fold}/{view}/{airport}: no rows")
            grouped = cell.groupby("utc_day", sort=True, observed=True)
            day_sums = grouped[list(BOOT_METRICS)].sum().to_numpy("float64")
            day_counts = grouped.size().to_numpy("float64")
            if len(day_sums) == 0:
                raise RuntimeError(f"bootstrap {fold}/{view}/{airport}: no day clusters")
            draws = rng.integers(0, len(day_sums), size=(BOOT_DRAWS, len(day_sums)))
            sums = day_sums[draws].sum(axis=1)
            counts = day_counts[draws].sum(axis=1)
            means = sums / counts[:, None]
            sampled[fold] += rank_n * means / denominator

    output = []
    for fold, _, _ in FOLDS:
        point_data = diff_columns(folds_meta[fold])
        point = {}
        den = sum(rank_maps[fold].values())
        for airport, rank_n in rank_maps[fold].items():
            cell = point_data.loc[point_data["airport"].astype(str).eq(airport)]
            point[airport] = cell[list(BOOT_METRICS)].mean().to_numpy("float64")
        fold_point = sum(rank_n * point[airport] for airport, rank_n in rank_maps[fold].items()) / den
        for index, metric in enumerate(BOOT_METRICS):
            values = sampled[fold][:, index]
            output.append({"scope": fold, "view": view, "metric": metric,
                           "point": float(fold_point[index]),
                           "lower_95": float(np.quantile(values, 0.025)),
                           "upper_95": float(np.quantile(values, 0.975)),
                           "draws": BOOT_DRAWS, "seed": BOOT_SEED})

    jan, jul = "january", "july_loo"
    jan_total = sum(rank_maps[jan].values())
    jul_total = sum(rank_maps[jul].values())
    pooled_denom = jan_total + jul_total
    pooled_draws = (sampled[jan] * jan_total + sampled[jul] * jul_total) / pooled_denom
    jan_point = {a: diff_columns(folds_meta[jan]).loc[
        diff_columns(folds_meta[jan])["airport"].astype(str).eq(a), list(BOOT_METRICS)
    ].mean().to_numpy("float64") for a in rank_maps[jan]}
    jul_diff = diff_columns(folds_meta[jul])
    jul_point = {a: jul_diff.loc[jul_diff["airport"].astype(str).eq(a),
                                 list(BOOT_METRICS)].mean().to_numpy("float64")
                 for a in rank_maps[jul]}
    pooled_point = (
        sum(n * jan_point[a] for a, n in rank_maps[jan].items())
        + sum(n * jul_point[a] for a, n in rank_maps[jul].items())
    ) / pooled_denom
    for index, metric in enumerate(BOOT_METRICS):
        values = pooled_draws[:, index]
        output.append({"scope": "pooled_jan_jul", "view": view, "metric": metric,
                       "point": float(pooled_point[index]),
                       "lower_95": float(np.quantile(values, 0.025)),
                       "upper_95": float(np.quantile(values, 0.975)),
                       "draws": BOOT_DRAWS, "seed": BOOT_SEED})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def refit_and_write_delta(D18, F, M, D, train_files: list[pd.DataFrame],
                          rank_counts_frame: pd.DataFrame, importance_rows: list[pd.DataFrame]) -> dict:
    train = pd.concat(train_files, ignore_index=True)
    train = train.loc[D.keep_reachable(train)].copy()
    rank_frames = {month: month_frame(month) for month in RANK_MONTHS}
    rank_targets = [rank_frames[m].drop(columns=["y"], errors="ignore") for m in RANK_MONTHS]
    train, rank_targets = D18.add_encodings(train, rank_targets)
    ordinary = train["ghost"].to_numpy(dtype="float64") < 0.5
    rank_counts_frame = rank_counts_frame.copy()
    weights, _, _ = normalize_weights(train.loc[ordinary, ["airport", "month"]],
                                       rank_counts_frame)
    fit_train = train.loc[ordinary].reset_index(drop=True)
    rank_ordinary = [frame.loc[frame["ghost"].to_numpy(dtype="float64") < 0.5].copy()
                     for frame in rank_targets]
    rank_all = pd.concat(rank_ordinary, ignore_index=True)
    fit_train, rank_all = map_lirf_categories(
        fit_train, rank_all, ["lirf_flight_prefix", "lirf_stand_group"])
    anchor_train = anchor_for(fit_train)
    y = fit_train["y"].to_numpy(dtype="float64")
    if not np.isfinite(anchor_train).all() or not np.isfinite(y).all():
        raise RuntimeError("final training labels or anchors are non-finite")
    x_train, x_rank, features, categorical = feature_frames(
        fit_train, rank_all, F, M, map_categories=False)
    params = dict(M.MAIN_PARAMS)
    params.update(seed=BASELINE_SEED, bagging_seed=BASELINE_SEED + 1,
                  feature_fraction_seed=BASELINE_SEED + 2)
    import lightgbm as lgb
    ds = lgb.Dataset(x_train, label=y - anchor_train, weight=weights,
                     categorical_feature=categorical, free_raw_data=True)
    print(f"fit start final primary rows={len(x_train):,} features={len(features)}", flush=True)
    started = time.monotonic()
    booster = lgb.train(params, ds, num_boost_round=1500)
    pred_by_id: dict[int, float] = {}
    for month in RANK_MONTHS:
        selected = rank_all["month"].eq(month).to_numpy()
        target_ordinary = rank_all.loc[selected]
        anchor = anchor_for(target_ordinary)
        if not np.isfinite(anchor).all():
            raise RuntimeError(f"{month}: final ranking anchors are non-finite")
        prediction = np.maximum(30.0, anchor + booster.predict(x_rank.loc[selected], num_threads=6))
        pred_by_id.update({int(i): float(p) for i, p in zip(target_ordinary[ID], prediction)})

    rank_metadata_frames = [pd.read_parquet(locate_file(f"rank_meta_{m}.parquet"))
                            for m in RANK_MONTHS]
    rank_meta = pd.concat(rank_metadata_frames, ignore_index=True)
    rank_meta[ID] = pd.to_numeric(rank_meta[ID], errors="raise").astype("int64")
    ordinary_nc = rank_meta["rank_class"].eq("no-clock_ordinary")
    ids = pd.read_parquet(locate_file("submitting_ids.parquet"))
    v22 = pd.read_parquet(locate_file("v22_submission.parquet"))
    if not np.array_equal(ids[ID].to_numpy("int64"), v22[ID].to_numpy("int64")):
        raise RuntimeError("final v22 IDs differ from the submission order")
    rank_by_id = rank_meta.set_index(ID)
    v22_by_id = v22.set_index(ID)["p22"]
    delta = np.zeros(len(ids), dtype="float32")
    for i, mvt_id in enumerate(ids[ID].to_numpy("int64")):
        if mvt_id not in rank_by_id.index:
            raise RuntimeError(f"submission ID {mvt_id} is absent from the ranking metadata")
        if rank_by_id.at[mvt_id, "rank_class"] == "no-clock_ordinary":
            if mvt_id not in pred_by_id:
                raise RuntimeError(f"missing final prediction for no-clock ID {mvt_id}")
            delta[i] = pred_by_id[mvt_id] - float(v22_by_id.loc[mvt_id])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "ranking_delta.npz",
                        row_order=ids["row_order"].to_numpy("int64"),
                        MVT_ID_mvt=ids[ID].to_numpy("int64"),
                        delta=delta)
    info = {"final_fit_seconds": time.monotonic() - started,
            "final_train_rows": int(len(x_train)),
            "ranking_rows_updated": int(ordinary_nc.sum()),
            "update_nonzero_rows": int(np.count_nonzero(delta)),
            "delta_sha256": file_hash(OUTPUT / "ranking_delta.npz")}
    print(f"final primary refit complete: {info}", flush=True)
    return info


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_manifest_path = locate_file("input-manifest.json")
    input_manifest = json.loads(input_manifest_path.read_text())
    if input_manifest.get("ranking_target_read") is not False:
        raise RuntimeError("input manifest does not prove ranking labels were withheld")
    preflight = json.loads(locate_file("preflight.json").read_text())
    if not preflight.get("v22_baseline_passed"):
        raise RuntimeError("v22 baseline gate was not passed before the fit")
    D18, F, M, D = setup_modules()
    counts = rank_counts()
    census = pd.read_csv(locate_file("census.csv"))
    denominator = pd.to_numeric(census["estimated_ranking_sse_clean_s2"], errors="coerce").sum()
    numerator = pd.to_numeric(
        census.loc[census["class"].eq("no-clock_ordinary"), "estimated_ranking_sse_clean_s2"],
        errors="coerce").sum()
    supported_no_clock_share = float(numerator / denominator)
    rank_maps = {}
    for fold, valid_month, _ in FOLDS:
        target_month = "2026-07" if valid_month == "2025-07" else "2026-01"
        part = counts.loc[counts["month"].eq(target_month)]
        rank_maps[fold] = dict(zip(part["airport"].astype(str),
                                   part["ranking_rows"].astype(int)))

    predictions: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict] = []
    cell_rows: list[dict] = []
    importance_rows: list[pd.DataFrame] = []
    fold_inputs: dict[str, pd.DataFrame] = {}
    training_frames: list[pd.DataFrame] = []
    fit_info: dict[str, dict] = {}

    for fold, valid_month, fold_index in FOLDS:
        print(f"fold preparation start {fold}", flush=True)
        train_months = [m for m in D.MONTHS if m != valid_month]
        raw_train = pd.concat([month_frame(m) for m in train_months], ignore_index=True)
        raw_train = raw_train.loc[D.keep_reachable(raw_train)].copy()
        valid_features = month_frame(valid_month).drop(columns=["y"], errors="ignore")
        raw_train, (valid_features,) = D18.add_encodings(raw_train, [valid_features])
        val_eval = eval_frame(fold).rename(
            columns={"airport": "eval_airport", "utc_day": "eval_utc_day"})
        valid_features = valid_features.merge(val_eval, on=ID, how="left",
                                              validate="one_to_one")
        if valid_features["y_true"].isna().any() or valid_features["r12"].isna().any():
            raise RuntimeError(f"{fold}: validation labels or p12 residuals are missing")
        if (not valid_features["airport"].astype(str).eq(
                valid_features["eval_airport"].astype(str)).all()
                or not valid_features["utc_day"].astype(str).eq(
                    valid_features["eval_utc_day"].astype(str)).all()):
            raise RuntimeError(f"{fold}: feature and baseline row metadata differ")
        valid_features = valid_features.drop(columns=["eval_airport", "eval_utc_day"])
        train_ordinary = raw_train["ghost"].to_numpy(dtype="float64") < 0.5
        valid_ordinary = valid_features["ghost"].to_numpy(dtype="float64") < 0.5
        train = raw_train.loc[train_ordinary].reset_index(drop=True)
        valid = valid_features.loc[valid_ordinary].reset_index(drop=True)
        train_meta = train[["airport", "month"]].copy()
        valid_meta = valid[["airport", "month"]].copy()
        clean_mask = ~valid[ID].isin(D.excluded_ids(fold))
        score_mask = ~valid["obs_has_clock"].astype(bool)
        score_support = valid.loc[score_mask & clean_mask].groupby(
            valid.loc[score_mask & clean_mask, "airport"].astype(str), observed=True).size()
        missing = sorted(set(rank_maps[fold]) - set(score_support.index.astype(str)))
        if missing:
            raise RuntimeError(f"{fold}: clean no-clock score lacks supported ranking cells {missing}")
        required_for_fold = {(required_fold, airport) for required_fold, airport in REQUIRED_CELLS
                             if required_fold == fold}
        available_for_fold = {(fold, airport) for airport in score_support.index.astype(str)}
        if required_for_fold and not required_for_fold.issubset(available_for_fold):
            missing_required = sorted(required_for_fold - available_for_fold)
            raise RuntimeError(f"{fold}: required top score cells are missing: {missing_required}")

        anchor_train = anchor_for(train)
        anchor_valid = anchor_for(valid)
        y = train["y"].to_numpy(dtype="float64")
        if not np.isfinite(anchor_train).all() or not np.isfinite(y).all():
            raise RuntimeError(f"{fold}: training target or anchor is non-finite")
        if not np.isfinite(anchor_valid).all():
            raise RuntimeError(f"{fold}: validation anchor is non-finite")

        primary_weights, p_map, q_map = normalize_weights(train_meta, counts)
        placebo_weights, placebo_map, _ = normalize_weights(
            train_meta, counts, placebo_seed=13007 + fold_index)
        train_x, valid_x, features, categorical = feature_frames(train, valid, F, M)
        train_residual = y - anchor_train
        predictions_fold: dict[str, np.ndarray] = {}
        infos_fold = {}

        for arm in ("control", "primary"):
            weights = np.ones(len(train), dtype="float32") if arm == "control" else primary_weights
            pred, result = fit_arm(arm, train_x, valid_x, train_residual, weights,
                                  categorical, anchor_valid, M, fold)
            predictions_fold[arm] = pred
            infos_fold[arm] = result["info"]
            importance_rows.append(result["importance"])
        sham_train_x, sham_valid_x = permute_lirf(
            train_x, valid_x, train_meta, valid_meta,
            [c for c in features if c.startswith("lirf_") and c not in ("lirf_MVT_ID_mvt",)],
            seed=13017 + fold_index)
        pred, result = fit_arm("sham", sham_train_x, sham_valid_x, train_residual,
                              primary_weights, categorical, anchor_valid, M, fold)
        predictions_fold["sham"] = pred
        infos_fold["sham"] = result["info"]
        importance_rows.append(result["importance"])
        del sham_train_x, sham_valid_x

        pred, result = fit_arm("placebo", train_x, valid_x, train_residual,
                              placebo_weights, categorical, anchor_valid, M, fold)
        predictions_fold["placebo"] = pred
        infos_fold["placebo"] = result["info"]
        importance_rows.append(result["importance"])
        fit_info[fold] = {"arms": infos_fold,
                          "train_rows": int(len(train)),
                          "valid_ordinary_rows": int(len(valid)),
                          "clean_no_clock_rows": int((score_mask & clean_mask).sum()),
                          "ranking_weight_stats": {
                              "primary_mean": float(primary_weights.mean()),
                              "primary_min": float(primary_weights.min()),
                              "primary_max": float(primary_weights.max()),
                              "placebo_mean": float(placebo_weights.mean()),
                              "positive_cells": int(sum(v > 0 for v in p_map.values())),
                              "weight_placebo_cells": int(sum(v > 0 for v in placebo_map.values()))}}
        rows = make_prediction_rows(fold, valid, predictions_fold, D)
        fold_inputs[fold] = rows
        predictions[fold] = rows
        for view in ("all", "clean"):
            current = cell_metrics(rows, prediction_columns(rows), rank_maps[fold], fold, view)
            cell_rows.extend(current)
            metric_rows.extend(weighted_rows(current, fold, view))
        rows.to_parquet(OUTPUT / f"predictions_{fold}.parquet", index=False,
                        compression="zstd")
        print(f"fold complete {fold}: {len(rows):,} ordinary no-clock rows", flush=True)
        del raw_train, train, valid, train_x, valid_x, train_residual, predictions_fold

    for view in ("all", "clean"):
        pooled_cells = []
        for fold in ("january", "july_loo"):
            pooled_cells.extend(cell_metrics(fold_inputs[fold], prediction_columns(fold_inputs[fold]),
                                             rank_maps[fold], fold, view))
        metric_rows.extend(weighted_rows(pooled_cells, "pooled_jan_jul", view, pooled=True))
    write_csv(OUTPUT / "cell_gains.csv", cell_rows)
    write_csv(OUTPUT / "metrics.csv", metric_rows)
    write_csv(OUTPUT / "feature_importance.csv", [row for df in importance_rows
                                                   for row in df.to_dict(orient="records")])

    interval_rows = []
    for view in ("all", "clean"):
        interval_rows.extend(bootstrap_for_view(fold_inputs, rank_maps, view))
    write_csv(OUTPUT / "intervals.csv", interval_rows)

    clean_metrics = [r for r in metric_rows if r["view"] == "clean"]
    primary_points = {r["fold"]: r["gain_s2_per_rank_row"] for r in clean_metrics
                      if r["arm"] == "primary"}
    clean_intervals = [r for r in interval_rows if r["view"] == "clean"]
    interval_lookup = {(r["scope"], r["metric"]): r for r in clean_intervals}
    required_scopes = ["july_loo", "december", "january", "pooled_jan_jul"]
    positive_points = all(primary_points.get(fold, float("nan")) > 0
                          for fold in ("july_loo", "december", "january"))
    p12_clear = all(interval_lookup.get((scope, "gain_primary"), {}).get("lower_95", -np.inf) > 0
                    for scope in required_scopes)
    sham_clear = interval_lookup.get(("pooled_jan_jul", "primary_vs_sham"), {}).get(
        "lower_95", -np.inf) > 0
    placebo_clear = interval_lookup.get(("pooled_jan_jul", "primary_vs_placebo"), {}).get(
        "lower_95", -np.inf) > 0
    gate_pass = bool(positive_points and p12_clear and sham_clear and placebo_clear)
    pooled_primary = next(r for r in clean_metrics
                          if r["fold"] == "pooled_jan_jul" and r["arm"] == "primary")
    frac = float(pooled_primary["gain_s2_per_rank_row"] / pooled_primary["baseline_p12_mse_s2"])
    board_seconds = float(245.0207 * (1.0 - np.sqrt(1.0 - supported_no_clock_share * frac)))
    census = pd.read_csv(locate_file("census.csv"))
    unsupported_with_clock = int(census.loc[
        census["class"].eq("with-clock")
        & census["fold_clean_support_status"].eq("unsupported"), "ranking_rows"].sum())
    repair_points = {r["fold"]: r["gain_s2_per_rank_row"] for r in clean_metrics
                     if r["arm"] == "blend_primary"}
    summary = {
        "experiment": "exp-130-step-2",
        "gate": {
            "positive_primary_gain_all_folds": positive_points,
            "primary_vs_p12_intervals_clear_each_fold_and_pooled": p12_clear,
            "primary_beats_feature_sham_pooled": sham_clear,
            "primary_beats_weight_placebo_pooled": placebo_clear,
            "pass": gate_pass,
            "verdict": "PASS" if gate_pass else "FAIL",
        },
        "clean_primary_gain_s2_per_rank_row": primary_points,
        "clean_primary_gain_interval_s2_per_rank_row": {
            scope: interval_lookup.get((scope, "gain_primary")) for scope in required_scopes},
        "clean_pooled_primary_vs_sham_interval": interval_lookup.get(
            ("pooled_jan_jul", "primary_vs_sham")),
        "clean_pooled_primary_vs_placebo_interval": interval_lookup.get(
            ("pooled_jan_jul", "primary_vs_placebo")),
        "clean_pooled_primary_vs_control_interval": interval_lookup.get(
            ("pooled_jan_jul", "primary_vs_control")),
        "board_price_seconds_supported_projection": board_seconds,
        "clean_no_clock_share_of_supported_v22_sse": supported_no_clock_share,
        "primary_fractional_clean_mse_reduction": frac,
        "unsupported_with_clock_ranking_rows": unsupported_with_clock,
        "one_repair": {
            "method": "fixed 50/50 blend of primary prediction with p12",
            "clean_gain_s2_per_rank_row": repair_points,
            "changes_d107_gate": False,
            "candidate": False,
        },
        "fit_info": fit_info,
        "lightgbm_version": importlib.import_module("lightgbm").__version__,
        "python_version": platform.python_version(),
        "ranking_target_read": False,
        "weather_used_or_downloaded": False,
    }
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"D107 {summary['gate']['verdict']}: {summary['gate']}", flush=True)

    final_info = None
    if gate_pass:
        train_files = [month_frame(m) for m in (*D.MONTHS,)]
        final_info = refit_and_write_delta(D18, F, M, D, train_files, counts, importance_rows)
        summary["final_refit"] = final_info
        (OUTPUT / "results.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    hashes = {str(path.relative_to(OUTPUT)): file_hash(path)
              for path in sorted(OUTPUT.rglob("*")) if path.is_file()}
    run_manifest = {
        "kernel_slug": os.environ.get("PRC_KERNEL_SLUG", "unknown"),
        "kernel_version": os.environ.get("PRC_KERNEL_VERSION", "unknown"),
        "dataset_id": os.environ.get("PRC_DATASET_ID", "unknown"),
        "dataset_version": os.environ.get("PRC_DATASET_VERSION", "unknown"),
        "seed_values": SEEDS,
        "rounds": 1500,
        "input_manifest_sha256": file_hash(input_manifest_path),
        "input_manifest": input_manifest,
        "spec_commit": input_manifest.get("spec_commit"),
        "source_sha256": input_manifest.get("source_sha256"),
        "output_sha256": hashes,
        "results": summary,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(run_manifest, indent=2, default=float) + "\n")
    print(f"outputs ready: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
