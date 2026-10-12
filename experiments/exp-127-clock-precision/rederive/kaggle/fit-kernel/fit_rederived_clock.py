#!/usr/bin/env python3
"""Fit the frozen exp-127b implied off-block residual model on private Kaggle CPU."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
FOLDS = ("july_loo", "december", "january")
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
BOOTSTRAPS = 2000
BOOT_SEED = 12728
BOARD_RMSE = 245.0207
BOARD_WEIGHTS = {"july_loo": 0.5571321275602379, "december": 0.0, "january": 0.4428678724397621}
EXPECTED_SPEC_SHA256 = "afcc0113061ca93bcd6eb66a48581b57081924928d7a2d07efa92d0cf487e75c"
EXPECTED_COVERAGE_SUMMARY_SHA256 = "f2286e2391f5c2b27f8912eab38824438cd248780a16c6fe0b5dc7ded589a467"
EXPECTED_COVERAGE_CSV_SHA256 = "3e643bb7126619b31190cb6ecbc72cdee7950e6ccbc587075f66482d1342ba9a"
EXPECTED_COVERAGE_RECEIPT_SHA256 = "b8d9578f80c7e29325cc09409be4318f72f45e3f5118b5d44b7a34f04c367f4e"
EXPECTED_DATASET_HASHES = {
    "fold_frames.parquet": "fb811f693697ca359050907a5a89d03227e30f11ab0ecaf2d0eadd8a3588f98e",
    "baseline_reproduction.json": "6d2c5eee25f4af3cf021f06afe3ef3d11082a696fe54098b33693077f709d530",
    "fold_cluster_population.csv": "d8eeb9c94b75e79571c626e43810f1e9f7203968fec3b5eb5ffedcdbf92b54f4",
    "ranking_frame.parquet": "3481827b5ce9961c3b4a9567a22fe745f4a7f72c5de0273351be797e11dfb4f0",
    "submitting_ids.csv": "397ebd40c810d3e305b9f634b99c2bfc40e8daa82357b19e27d9965135e5239e",
}
FEATURES = (
    "survey_to_obs_stand_m", "survey_to_runway_km", "survey_is_osm",
    "implied_offblock_to_current250_s",
)
SHAM_FEATURES = (
    "sham_survey_to_obs_stand_m", "sham_survey_to_runway_km", "sham_survey_is_osm",
    "sham_implied_offblock_to_current250_s",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_dataset_file(filename: str) -> Path:
    """Find the attached private dataset by file, independent of Kaggle mount name."""
    if not INPUT_ROOT.exists():
        raise SystemExit(f"Kaggle input root {INPUT_ROOT} does not exist")
    matches = sorted(INPUT_ROOT.glob(f"**/{filename}"))
    if len(matches) != 1:
        roots = sorted(p.name for p in INPUT_ROOT.iterdir())
        raise SystemExit(
            f"expected one attached {filename}, found {len(matches)}; "
            f"input mounts={roots}, matches={[str(p) for p in matches]}"
        )
    return matches[0]


def design(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    geo = frame.loc[:, columns].to_numpy(dtype=np.float64)
    geo[~np.isfinite(geo)] = np.nan
    airport = (
        pd.get_dummies(frame["airport"].astype(str))
        .reindex(columns=AIRPORTS, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    return np.column_stack([geo, airport])


def learner() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_bins=63,
        min_samples_leaf=250,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=12727,
    )


def airport_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby("airport", observed=True)["r_v22"].mean()
    fallback = float(train["r_v22"].mean())
    return test["airport"].map(means).fillna(fallback).to_numpy(dtype=np.float64)


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def cluster_sums(frame: pd.DataFrame, gain: np.ndarray, cluster_population: pd.DataFrame, fold: str) -> np.ndarray:
    rows = frame[["airport", "day"]].copy()
    rows["gain"] = gain
    grouped = rows.groupby(["airport", "day"], sort=True, observed=True)["gain"].sum()
    population = cluster_population.loc[cluster_population["fold"].eq(fold)].copy()
    population["day"] = population["day"].astype(str)
    full_index = pd.MultiIndex.from_frame(population[["airport", "day"]].astype(str))
    values = grouped.reindex(full_index, fill_value=0.0).to_numpy(dtype=np.float64)
    return values


def bootstrap_cluster_gain(frame: pd.DataFrame, gain: np.ndarray, population: pd.DataFrame,
                           fold: str, sample_indices: np.ndarray) -> np.ndarray:
    sums = cluster_sums(frame, gain, population, fold)
    fold_population = population.loc[population["fold"].eq(fold)].copy()
    row_counts = fold_population["fold_rows"].to_numpy(dtype=np.float64)
    denominators = row_counts[sample_indices].sum(axis=1)
    if np.any(denominators <= 0):
        raise SystemExit(f"{fold}: a cluster-bootstrap draw has no fold rows")
    return sums[sample_indices].sum(axis=1) / denominators


def metrics_by_model(predictions: pd.DataFrame, base: dict, cluster_population: pd.DataFrame,
                     bootstrap_indices: dict[str, np.ndarray]) -> tuple[dict, dict]:
    summaries: dict[str, dict[str, dict]] = {}
    draws_by_model: dict[str, dict[str, np.ndarray]] = {}
    models = {"current_head": "pred_current", "airport_only": "pred_airport",
              "implied_clock": "pred_implied", "coordinate_sham": "pred_sham"}

    for model_name, pred_col in models.items():
        summaries[model_name] = {}
        draws_by_model[model_name] = {}
        for fold in FOLDS:
            test = predictions.loc[predictions["fold"].eq(fold)].copy()
            residual = test["r_v22"].to_numpy(dtype=np.float64)
            delta = test[pred_col].to_numpy(dtype=np.float64)
            gain = residual**2 - (residual - delta) ** 2
            n_fold = int(base["folds_by_name"][fold]["rows_all"])
            gain_all = float(gain.sum() / n_fold)
            before_rmse = float(base["folds_by_name"][fold]["rmse_v22_reproduced"])
            summaries[model_name][fold] = {
                "rows_with_clock": int(len(test)),
                "gain_mse_s2_per_all_row": gain_all,
                "gain_mse_s2_per_with_clock_row": float(gain.mean()),
                "v22_rmse_s": before_rmse,
                "updated_rmse_s": float(np.sqrt(max(0.0, before_rmse**2 - gain_all))),
                "with_clock_rmse_before_s": rmse(residual),
                "with_clock_rmse_after_s": rmse(residual - delta),
            }
            draw = bootstrap_cluster_gain(
                test, gain, cluster_population, fold, bootstrap_indices[fold]
            )
            draws_by_model[model_name][fold] = draw
            summaries[model_name][fold]["gain_mse_cluster_bootstrap_95ci"] = [
                float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))
            ]

        july = summaries[model_name]["july_loo"]["gain_mse_s2_per_all_row"]
        january = summaries[model_name]["january"]["gain_mse_s2_per_all_row"]
        pooled = BOARD_WEIGHTS["july_loo"] * july + BOARD_WEIGHTS["january"] * january
        pooled_draw = (BOARD_WEIGHTS["july_loo"] * draws_by_model[model_name]["july_loo"]
                       + BOARD_WEIGHTS["january"] * draws_by_model[model_name]["january"])
        summaries[model_name]["pooled_board_weighted"] = {
            "gain_mse_s2_per_row": float(pooled),
            "gain_mse_cluster_bootstrap_95ci": [
                float(np.quantile(pooled_draw, 0.025)), float(np.quantile(pooled_draw, 0.975))
            ],
        }
        draws_by_model[model_name]["pooled_board_weighted"] = pooled_draw
    return summaries, draws_by_model


def price(summary: dict, board_share: float, board_mse: float) -> dict:
    total_gain = float(summary["pooled_board_weighted"]["gain_mse_s2_per_row"])
    frac = total_gain / (board_share * board_mse) if board_share > 0 and board_mse > 0 else 0.0
    return {
        "share": board_share,
        "frac": frac,
        "share_times_frac": float(total_gain / board_mse),
        "board_seconds": float(BOARD_RMSE * (1.0 - np.sqrt(max(0.0, 1.0 - total_gain / board_mse)))),
    }


def ridge_design(train: pd.DataFrame, frame: pd.DataFrame,
                 columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Ridge form: log distances and standardized implied-clock gap."""
    train_geo = train.loc[:, columns[:2]].to_numpy(dtype=np.float64)
    test_geo = frame.loc[:, columns[:2]].to_numpy(dtype=np.float64)
    train_geo[~np.isfinite(train_geo)] = np.nan
    test_geo[~np.isfinite(test_geo)] = np.nan
    if np.any(train_geo[np.isfinite(train_geo)] < 0) or np.any(test_geo[np.isfinite(test_geo)] < 0):
        raise SystemExit("Ridge distances must be nonnegative")
    train_geo = np.log1p(train_geo)
    test_geo = np.log1p(test_geo)
    medians = np.nanmedian(train_geo, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train_geo = np.where(np.isfinite(train_geo), train_geo, medians)
    test_geo = np.where(np.isfinite(test_geo), test_geo, medians)
    means = train_geo.mean(axis=0)
    scales = train_geo.std(axis=0)
    scales[scales <= 1e-12] = 1.0
    train_geo = (train_geo - means) / scales
    test_geo = (test_geo - means) / scales

    train_source = train[columns[2]].to_numpy(dtype=np.float64)
    test_source = frame[columns[2]].to_numpy(dtype=np.float64)
    train_source[~np.isfinite(train_source)] = np.nan
    test_source[~np.isfinite(test_source)] = np.nan
    source_median = float(np.nanmedian(train_source))
    if not np.isfinite(source_median):
        source_median = 0.0
    train_source = np.where(np.isfinite(train_source), train_source, source_median)
    test_source = np.where(np.isfinite(test_source), test_source, source_median)
    train_clock = train[columns[3]].to_numpy(dtype=np.float64)
    test_clock = frame[columns[3]].to_numpy(dtype=np.float64)
    train_clock[~np.isfinite(train_clock)] = np.nan
    test_clock[~np.isfinite(test_clock)] = np.nan
    clock_median = float(np.nanmedian(train_clock))
    if not np.isfinite(clock_median):
        clock_median = 0.0
    train_clock = np.where(np.isfinite(train_clock), train_clock, clock_median)
    test_clock = np.where(np.isfinite(test_clock), test_clock, clock_median)
    clock_mean = float(train_clock.mean())
    clock_scale = float(train_clock.std())
    if clock_scale <= 1e-12:
        clock_scale = 1.0
    train_clock = (train_clock - clock_mean) / clock_scale
    test_clock = (test_clock - clock_mean) / clock_scale
    train_airport = (
        pd.get_dummies(train["airport"].astype(str))
        .reindex(columns=AIRPORTS, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    test_airport = (
        pd.get_dummies(frame["airport"].astype(str))
        .reindex(columns=AIRPORTS, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    return (
        np.column_stack([train_geo, train_source, train_clock, train_airport]),
        np.column_stack([test_geo, test_source, test_clock, test_airport]),
    )


def fit_oof(frame: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[dict]]:
    """Run the registered outer folds and same-learner sham for one model family."""
    predictions = []
    fit_log = []
    for fold in FOLDS:
        train = frame.loc[~frame["fold"].eq(fold)].reset_index(drop=True)
        test = frame.loc[frame["fold"].eq(fold)].reset_index(drop=True)
        y_train = train["r_v22"].to_numpy(dtype=np.float64)
        y_test = test["r_v22"].to_numpy(dtype=np.float64)

        if kind == "hist_gradient_boosting":
            model = learner()
            model.fit(design(train, FEATURES), y_train)
            pred = model.predict(design(test, FEATURES)).astype(np.float64)
            sham_model = learner()
            sham_model.fit(design(train, SHAM_FEATURES), y_train)
            sham_pred = sham_model.predict(design(test, SHAM_FEATURES)).astype(np.float64)
            fit_log.append({
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "primary_iterations": int(model.n_iter_),
                "sham_iterations": int(sham_model.n_iter_),
            })
        elif kind == "ridge_alpha_10":
            x_train, x_test = ridge_design(train, test, FEATURES)
            model = Ridge(alpha=10.0)
            model.fit(x_train, y_train)
            pred = model.predict(x_test).astype(np.float64)
            sx_train, sx_test = ridge_design(train, test, SHAM_FEATURES)
            sham_model = Ridge(alpha=10.0)
            sham_model.fit(sx_train, y_train)
            sham_pred = sham_model.predict(sx_test).astype(np.float64)
            fit_log.append({
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "ridge_alpha": 10.0,
            })
        else:
            raise ValueError(f"unknown model family {kind}")

        predictions.append(pd.DataFrame({
            "MVT_ID_mvt": test["MVT_ID_mvt"].to_numpy(dtype=np.int64),
            "fold": fold,
            "month": test["month"].astype(str).to_numpy(),
            "airport": test["airport"].astype(str).to_numpy(),
            "day": test["day"].astype(str).to_numpy(),
            "r_v22": y_test,
            "pred_current": np.zeros(len(test), dtype=np.float64),
            "pred_airport": airport_only(train, test),
            "pred_implied": pred,
            "pred_sham": sham_pred,
        }))
    return pd.concat(predictions, ignore_index=True), fit_log


def gate(summary: dict) -> dict:
    geom = summary["implied_clock"]
    sham = summary["coordinate_sham"]
    airport = summary["airport_only"]
    folds_positive = all(geom[f]["gain_mse_s2_per_all_row"] > 0 for f in FOLDS)
    folds_beat_sham = all(
        geom[f]["gain_mse_s2_per_all_row"] > sham[f]["gain_mse_s2_per_all_row"]
        for f in FOLDS
    )
    folds_beat_airport = all(
        geom[f]["gain_mse_s2_per_all_row"] > airport[f]["gain_mse_s2_per_all_row"]
        for f in FOLDS
    )
    pooled = geom["pooled_board_weighted"]
    every_fold_ci_positive = all(
        geom[f]["gain_mse_cluster_bootstrap_95ci"][0] > 0 for f in FOLDS
    )
    return {
        "positive_on_all_folds": bool(folds_positive),
        "beats_coordinate_sham_on_all_folds_and_pooled": bool(
            folds_beat_sham
            and pooled["gain_mse_s2_per_row"] > sham["pooled_board_weighted"]["gain_mse_s2_per_row"]
        ),
        "beats_airport_only_on_all_folds_and_pooled": bool(
            folds_beat_airport
            and pooled["gain_mse_s2_per_row"] > airport["pooled_board_weighted"]["gain_mse_s2_per_row"]
        ),
        "all_fold_and_pooled_95ci_above_zero": bool(
            every_fold_ci_positive and pooled["gain_mse_cluster_bootstrap_95ci"][0] > 0
        ),
    }


def save_fold_outputs(prefix: str, predictions: pd.DataFrame, summaries: dict) -> None:
    pd.DataFrame([
        {"model": model, "fold": fold, **summaries[model][fold]}
        for model in summaries for fold in FOLDS
    ]).to_csv(OUT / f"{prefix}fold_scores.csv", index=False, lineterminator="\n")
    predictions.to_parquet(
        OUT / f"{prefix}fold_predictions.parquet", index=False, compression="zstd"
    )


def ranking_predictions(frame: pd.DataFrame, kind: str, data_dir: Path,
                       clock_features: pd.DataFrame) -> tuple[np.ndarray, dict]:
    ranking_path = data_dir / "ranking_frame.parquet"
    ids_path = data_dir / "submitting_ids.csv"
    ranking = pd.read_parquet(ranking_path)
    submitting_ids = pd.read_csv(ids_path, dtype={"MVT_ID_mvt": "int64"})
    if not np.array_equal(ranking["rank_order"].to_numpy(), np.arange(len(ranking))):
        raise SystemExit("ranking feature frame is not in submitting order")
    if not np.array_equal(
        ranking["MVT_ID_mvt"].to_numpy(dtype=np.int64),
        submitting_ids["MVT_ID_mvt"].to_numpy(dtype=np.int64),
    ):
        raise SystemExit("ranking features do not match submitting IDs in order")
    if ranking["MVT_ID_mvt"].duplicated().any():
        raise SystemExit("ranking feature frame has duplicate IDs")
    clock_rank = clock_features.loc[
        clock_features["scope"].eq("ranking"),
        ["MVT_ID_mvt", "implied_offblock_to_current250_s",
         "sham_implied_offblock_to_current250_s"],
    ].copy()
    if clock_rank["MVT_ID_mvt"].duplicated().any() or len(clock_rank) != len(ranking):
        raise SystemExit("re-derived ranking clock rows do not match the ranking frame")
    ranking = ranking.merge(clock_rank, on="MVT_ID_mvt", how="left", validate="one_to_one", sort=False)
    if not np.array_equal(ranking["rank_order"].to_numpy(), np.arange(len(ranking))):
        raise SystemExit("joining clock inputs changed submitting order")

    usable = (
        ranking["obs_has_clock"].to_numpy(dtype=bool)
        & ~ranking["noac"].to_numpy(dtype=bool)
        & ranking["airport"].astype(str).ne("LFPG").to_numpy()
    )
    update = np.zeros(len(ranking), dtype=np.float64)
    if kind == "hist_gradient_boosting":
        model = learner()
        model.fit(design(frame, FEATURES), frame["r_v22"].to_numpy(dtype=np.float64))
        update[usable] = model.predict(design(ranking.loc[usable], FEATURES)).astype(np.float64)
        fit_info = {"family": kind, "training_rows": int(len(frame)),
                    "iterations": int(model.n_iter_), "ranking_rows_updated": int(usable.sum())}
    elif kind == "ridge_alpha_10":
        x_train, x_rank = ridge_design(frame, ranking.loc[usable], FEATURES)
        model = Ridge(alpha=10.0)
        model.fit(x_train, frame["r_v22"].to_numpy(dtype=np.float64))
        update[usable] = model.predict(x_rank).astype(np.float64)
        fit_info = {"family": kind, "training_rows": int(len(frame)),
                    "alpha": 10.0, "ranking_rows_updated": int(usable.sum())}
    else:
        raise ValueError(f"unknown model family {kind}")

    if not np.isfinite(update).all() or np.any(update[~usable] != 0.0):
        raise SystemExit("ranking update vector is non-finite or moves a held row")
    np.savez_compressed(
        OUT / "ranking_update_vector.npz",
        rank_order=ranking["rank_order"].to_numpy(dtype=np.int64),
        MVT_ID_mvt=ranking["MVT_ID_mvt"].to_numpy(dtype=np.int64),
        delta=update,
    )
    return update, fit_info


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = locate_dataset_file("fold_frames.parquet")
    data_dir = data_path.parent
    clock_path = locate_dataset_file("clock_features.parquet")
    coverage_summary_path = locate_dataset_file("coverage_summary.json")
    coverage_receipt_path = locate_dataset_file("run_receipt.json")
    coverage_csv_path = locate_dataset_file("coverage_by_airport.csv")
    base_path = data_dir / "baseline_reproduction.json"
    cluster_path = data_dir / "fold_cluster_population.csv"
    spec_path = locate_dataset_file("rederive_SPEC.md")
    ranking_path = data_dir / "ranking_frame.parquet"
    submitting_ids_path = data_dir / "submitting_ids.csv"
    for path in (data_path, base_path, cluster_path, ranking_path, submitting_ids_path):
        expected = EXPECTED_DATASET_HASHES[path.name]
        if sha256(path) != expected:
            raise SystemExit(f"{path.name} does not match the frozen fold dataset hash")
    frame = pd.read_parquet(data_path)
    clock_features = pd.read_parquet(clock_path)
    base_record = json.loads(base_path.read_text())
    coverage_summary = json.loads(coverage_summary_path.read_text())
    coverage_receipt = json.loads(coverage_receipt_path.read_text())
    cluster_population = pd.read_csv(cluster_path, dtype={"fold": str, "airport": str, "day": str})
    for path, expected in (
        (spec_path, EXPECTED_SPEC_SHA256),
        (coverage_summary_path, EXPECTED_COVERAGE_SUMMARY_SHA256),
        (coverage_csv_path, EXPECTED_COVERAGE_CSV_SHA256),
        (coverage_receipt_path, EXPECTED_COVERAGE_RECEIPT_SHA256),
    ):
        if sha256(path) != expected:
            raise SystemExit(f"{path.name} does not match the frozen input hash")
    if not coverage_summary.get("coverage_gate_pass", False):
        raise SystemExit("the frozen fit cannot start because implied-clock coverage failed")
    for forbidden in ("y_true", "r12", "r_v22", "target", "label"):
        if forbidden in clock_features.columns:
            raise SystemExit(f"label-free clock features contain forbidden column {forbidden}")
    expected_clock_hash = coverage_receipt["outputs"]["clock_features.parquet"]["sha256"]
    if sha256(clock_path) != expected_clock_hash:
        raise SystemExit("mounted re-derived clock features do not match their coverage receipt")
    if len(frame) != sum(int(row["rows_with_clock_not_noac"]) for row in base_record["folds"]):
        raise SystemExit("prepared row count does not match the frozen baseline record")
    for row in base_record["folds"]:
        observed = frame.loc[frame["fold"].eq(row["fold"])]
        if len(observed) != int(row["rows_with_clock_not_noac"]):
            raise SystemExit(f"{row['fold']}: prepared class count does not match the baseline record")
        if abs(float(row["rmse_v22_expected"]) - float(row["rmse_v22_reproduced"])) > 1e-8:
            raise SystemExit(f"{row['fold']}: the saved v22 baseline did not reproduce")
        clusters = cluster_population.loc[cluster_population["fold"].eq(row["fold"])]
        if int(clusters["fold_rows"].sum()) != int(row["rows_all"]):
            raise SystemExit(f"{row['fold']}: cluster population does not sum to all fold rows")
        if int(clusters["eligible_rows"].sum()) != int(row["rows_with_clock_not_noac"]):
            raise SystemExit(f"{row['fold']}: cluster population does not sum to eligible rows")
    if frame.duplicated(["fold", "MVT_ID_mvt"]).any() or not np.isfinite(frame["r_v22"]).all():
        raise SystemExit("prepared fold frame has duplicate IDs or non-finite residuals")
    fold_clock = clock_features.loc[
        clock_features["scope"].eq("fold"),
        ["MVT_ID_mvt", "fold", "airport", "implied_offblock_to_current250_s",
         "sham_implied_offblock_to_current250_s"],
    ].copy()
    if fold_clock.duplicated(["fold", "MVT_ID_mvt"]).any():
        raise SystemExit("re-derived fold clock rows are not unique")
    frame = frame.merge(
        fold_clock, on=["fold", "MVT_ID_mvt"], how="left", validate="one_to_one",
        suffixes=("", "_clock"), indicator="_clock_match",
    )
    if not frame["_clock_match"].eq("both").all():
        raise SystemExit("re-derived clock rows do not cover every prepared fold key")
    if not frame["airport"].astype(str).eq(frame["airport_clock"].astype(str)).all():
        raise SystemExit("re-derived fold clock airport keys do not match the fold frame")
    frame = frame.drop(columns=["airport_clock", "_clock_match"])
    fold_rows_before_lfpg_exclusion = {
        fold: int(frame["fold"].eq(fold).sum()) for fold in FOLDS
    }
    frame = frame.loc[frame["airport"].astype(str).ne("LFPG")].reset_index(drop=True)
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap_indices = {}
    for fold in FOLDS:
        n_clusters = int(cluster_population["fold"].eq(fold).sum())
        bootstrap_indices[fold] = rng.integers(
            0, n_clusters, size=(BOOTSTRAPS, n_clusters)
        )
    board_scope_sse = 0.0
    fold_record_by_name = {row["fold"]: row for row in base_record["folds"]}
    for fold in FOLDS:
        fold_rows = frame.loc[frame["fold"].eq(fold)]
        all_rows = int(fold_record_by_name[fold]["rows_all"])
        board_scope_sse += BOARD_WEIGHTS[fold] * float(
            np.square(fold_rows["r_v22"].to_numpy(dtype=np.float64)).sum() / all_rows
        )
    base = {
        "folds_by_name": {r["fold"]: r for r in base_record["folds"]},
        "share": float(board_scope_sse / float(base_record["weighted_board_mse"])),
        "board_mse": float(base_record["weighted_board_mse"]),
    }
    predictions, primary_fit_log = fit_oof(frame, "hist_gradient_boosting")
    primary_summaries, _ = metrics_by_model(
        predictions, base, cluster_population, bootstrap_indices
    )
    primary_gate = gate(primary_summaries)
    primary_gate["pass"] = all(primary_gate.values())
    save_fold_outputs("primary_", predictions, primary_summaries)

    repair_summaries = None
    repair_gate = None
    repair_fit_log = None
    selected_kind = "hist_gradient_boosting" if primary_gate["pass"] else None
    selected_summaries = primary_summaries
    selected_predictions = predictions
    if not primary_gate["pass"]:
        failed = [name for name, passed in primary_gate.items() if not passed]
        diagnosis = {
            "before_repair": True,
            "primary_model": "HistGradientBoostingRegressor",
            "primary_gate": primary_gate,
            "failed_gate_items": failed,
            "repair_attempt": "one Ridge(alpha=10) with the frozen distance fields, source flag, implied-clock gap, and airport indicators",
        }
        (OUT / "primary_diagnosis.json").write_text(
            json.dumps(diagnosis, indent=2, allow_nan=False) + "\n"
        )
        repair_predictions, repair_fit_log = fit_oof(frame, "ridge_alpha_10")
        repair_summaries, _ = metrics_by_model(
            repair_predictions, base, cluster_population, bootstrap_indices
        )
        repair_gate = gate(repair_summaries)
        repair_gate["pass"] = all(repair_gate.values())
        save_fold_outputs("repair_", repair_predictions, repair_summaries)
        if repair_gate["pass"]:
            selected_kind = "ridge_alpha_10"
            selected_summaries = repair_summaries
            selected_predictions = repair_predictions

    selected_prices = {
        name: price(selected_summaries[name], base["share"], base["board_mse"])
        for name in ("implied_clock", "coordinate_sham", "airport_only", "current_head")
    }
    ranking_fit_log = None
    ranking_written = False
    if selected_kind is not None:
        if not ranking_path.exists() or not submitting_ids_path.exists():
            raise SystemExit("ranking input is missing after the selected model passed D107")
        _, ranking_fit_log = ranking_predictions(frame, selected_kind, data_dir, clock_features)
        np.savez_compressed(
            OUT / "fold_update_vectors.npz",
            **{
                f"{fold}_MVT_ID_mvt": selected_predictions.loc[
                    selected_predictions["fold"].eq(fold), "MVT_ID_mvt"
                ].to_numpy(dtype=np.int64),
                f"{fold}_delta": selected_predictions.loc[
                    selected_predictions["fold"].eq(fold), "pred_implied"
                ].to_numpy(dtype=np.float64),
            },
        )
        ranking_written = True

    output_paths = sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.parquet"))
    output_paths += sorted(OUT.glob("*.npz")) + sorted(OUT.glob("*.json"))
    output_paths = [p for p in output_paths if p.name not in ("results.json", "run_receipt.json")]

    output = {
        "experiment": "exp-127-clock-precision",
        "spec": "experiments/exp-127-clock-precision/rederive/SPEC.md",
        "input_sha256": {
            "fold_frames.parquet": sha256(data_path),
            "baseline_reproduction.json": sha256(base_path),
            "fold_cluster_population.csv": sha256(cluster_path),
            "rederive_SPEC.md": sha256(spec_path),
            "clock_features.parquet": sha256(clock_path),
            "coverage_summary.json": sha256(coverage_summary_path),
            "coverage_by_airport.csv": sha256(coverage_csv_path),
            "coverage_run_receipt.json": sha256(coverage_receipt_path),
            **({"ranking_frame.parquet": sha256(ranking_path),
                "submitting_ids.csv": sha256(submitting_ids_path)}
               if ranking_path.exists() and submitting_ids_path.exists() else {}),
        },
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "runtime_seconds": float(time.time() - started),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOT_SEED,
        "cluster_bootstrap_denominator": "sum fold_rows in sampled airport-day clusters for each draw",
        "coverage_gate": coverage_summary,
        "fold_rows_before_lfpg_exclusion": fold_rows_before_lfpg_exclusion,
        "fold_rows_scored_after_lfpg_exclusion": {
            fold: int(frame["fold"].eq(fold).sum()) for fold in FOLDS
        },
        "lfpg_excluded_rows": {
            fold: int(fold_rows_before_lfpg_exclusion[fold]
                     - frame["fold"].eq(fold).sum()) for fold in FOLDS
        },
        "primary_fit_log": primary_fit_log,
        "primary_folds": primary_summaries,
        "primary_board_prices": {
            name: price(primary_summaries[name], base["share"], base["board_mse"])
            for name in ("implied_clock", "coordinate_sham", "airport_only", "current_head")
        },
        "primary_d107_gate": primary_gate,
        "primary_diagnosis_sha256": sha256(OUT / "primary_diagnosis.json")
            if (OUT / "primary_diagnosis.json").exists() else None,
        "repair_fit_log": repair_fit_log,
        "repair_folds": repair_summaries,
        "repair_board_prices": ({
            name: price(repair_summaries[name], base["share"], base["board_mse"])
            for name in ("implied_clock", "coordinate_sham", "airport_only", "current_head")
        } if repair_summaries is not None else None),
        "repair_d107_gate": repair_gate,
        "selected_model": selected_kind,
        "selected_folds": selected_summaries,
        "selected_board_prices": selected_prices,
        "ranking_fit": ranking_fit_log,
        "ranking_update_vector_written": ranking_written,
        "ranking_vector_order": "MVT_ID_mvt order of submitting.parquet, checked against supplied ID list",
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    output["output_sha256"] = {
        path.name: sha256(path) for path in output_paths
    }
    output["output_sha256"]["results.json"] = sha256(OUT / "results.json")
    # Include the outputs' hashes in a small separate receipt so results.json keeps
    # its hash stable and can be checked byte for byte after download.
    (OUT / "run_receipt.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
