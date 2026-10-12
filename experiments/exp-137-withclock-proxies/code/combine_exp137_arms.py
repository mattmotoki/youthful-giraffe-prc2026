#!/usr/bin/env python3
"""Join exp-137 arm outputs, verify weights, and apply the frozen gain gate."""
from __future__ import annotations
import os

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd



def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "proxy", "sham")
FOLDS = ("july_loo", "december", "january")
RANK_MONTH = {"july_loo": "2026-07", "december": "2026-01", "january": "2026-01"}
JAN_JUL = ("january", "july_loo")
BOOT_DRAWS = 2000
BOOT_SEED = 20260925
MIN_SUPPORT_ROWS = 100
MIN_SUPPORT_DAYS = 10
GAIN_COLUMNS = tuple(f"gain_{arm}" for arm in ARMS)
PAIR_COLUMNS = ("proxy_minus_control", "proxy_minus_sham")
BOOT_COLUMNS = (*GAIN_COLUMNS, *PAIR_COLUMNS)
ARM_COLUMNS = {arm: {"pred_raw", "weight", "pred_blend", "gain_blend"} for arm in ARMS}
EXPECTED_SPEC_SHA = "71ca0e05d8cacaaeeac26667e3d8f73449e0772df17596a4e13d92b2255ce689"
def expected_kernel_metadata_ids():
    return {arm: f"{kaggle_user()}/prc-exp137-with-clock-{arm}-arm" for arm in ARMS}

def kernel_url_slugs():
    return {
    arm: f"{kaggle_user()}/prc-exp137-with-clock-{arm}-arm" for arm in ARMS
}

EXPECTED_DATASET_VERSIONS = {
    "prc-exp130-clean-input": "1",
    "prc-exp136-input": "4",
    "prc-exp137-input": "3",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_arm(run_dir: Path, arm: str) -> tuple:
    output = run_dir / arm / "output"
    result_path, manifest_path = output / "results.json", output / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{arm}: missing results.json or manifest.json")
    result, manifest = json.loads(result_path.read_text()), json.loads(manifest_path.read_text())
    if result.get("state") != "arm_complete_no_gate" or result.get("arm") != arm:
        raise RuntimeError(f"{arm}: result is not a completed arm-only fit")
    if manifest.get("state") != "arm_complete_no_gate" or manifest.get("arm") != arm:
        raise RuntimeError(f"{arm}: manifest does not match a completed arm-only fit")
    if result.get("spec_sha256") != EXPECTED_SPEC_SHA or manifest.get("spec_sha256") != EXPECTED_SPEC_SHA:
        raise RuntimeError(f"{arm}: output SPEC hash differs from the frozen SPEC")
    if result.get("ranking_target_read") is not False or manifest.get("ranking_target_read") is not False:
        raise RuntimeError(f"{arm}: output does not preserve ranking label isolation")
    if result.get("weather_used_or_downloaded") is not False or manifest.get("weather_used_or_downloaded") is not False:
        raise RuntimeError(f"{arm}: output does not preserve the weather exclusion")
    if manifest.get("kernel_slug") != expected_kernel_metadata_ids()[arm]:
        raise RuntimeError(f"{arm}: output kernel metadata ID differs from the frozen arm")
    if manifest.get("dataset_versions") != EXPECTED_DATASET_VERSIONS:
        raise RuntimeError(f"{arm}: attached dataset versions differ from checked inputs")
    for name, record in manifest.get("output_sha256", {}).items():
        path = output / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"{arm}: output hash mismatch for {name}")
    required = ("main_predictions.parquet", "calibration_predictions.parquet",
                "weights.csv", "ranking_counts.csv", "base_reproduction.csv")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"{arm}: required output files are missing: {missing}")
    main = pd.read_parquet(output / "main_predictions.parquet")
    calibration = pd.read_parquet(output / "calibration_predictions.parquet")
    weights = pd.read_csv(output / "weights.csv")
    ranking = pd.read_csv(output / "ranking_counts.csv")
    return result, manifest, main, calibration, weights, ranking


def validate_counts(arm_data: dict[str, tuple]) -> tuple[dict, pd.DataFrame]:
    first = arm_data[ARMS[0]][5].sort_values(["fold", "airport"]).reset_index(drop=True)
    needed = {"fold", "airport", "rank_month", "ranking_rows", "clean_validation_rows",
              "clean_validation_utc_days", "supported", "support_min_rows",
              "support_min_utc_days"}
    if not needed.issubset(first.columns):
        raise RuntimeError(f"ranking_counts.csv lacks {sorted(needed - set(first.columns))}")
    for arm in ARMS:
        part = arm_data[arm][5].sort_values(["fold", "airport"]).reset_index(drop=True)
        if not first.equals(part):
            raise RuntimeError(f"{arm}: ranking counts differ from the control arm")
    ranks, clean_counts = {}, {}
    for fold in FOLDS:
        part = first.loc[first.fold.astype(str).eq(fold)]
        if part.empty or not part.rank_month.astype(str).eq(RANK_MONTH[fold]).all():
            raise RuntimeError(f"{fold}: ranking count rows have the wrong rank month")
        if part.airport.astype(str).duplicated().any() or part.ranking_rows.le(0).any():
            raise RuntimeError(f"{fold}: ranking counts contain duplicates or non-positive rows")
        supported = part.supported.astype(bool)
        expected_supported = (part.clean_validation_rows.ge(MIN_SUPPORT_ROWS)
                              & part.clean_validation_utc_days.ge(MIN_SUPPORT_DAYS))
        if not supported.equals(expected_supported):
            raise RuntimeError(f"{fold}: saved support flags differ from the frozen minimum")
        if not part.support_min_rows.eq(MIN_SUPPORT_ROWS).all()\
                or not part.support_min_utc_days.eq(MIN_SUPPORT_DAYS).all():
            raise RuntimeError(f"{fold}: saved support thresholds differ from the frozen minimum")
        supported_part = part.loc[supported]
        if supported_part.empty:
            raise RuntimeError(f"{fold}: no ranking airport cells meet the support minimum")
        ranks[fold] = dict(zip(supported_part.airport.astype(str),
                               supported_part.ranking_rows.astype(int)))
        clean_counts[fold] = dict(zip(supported_part.airport.astype(str),
                                      supported_part.clean_validation_rows.astype(int)))
    return {"ranks": ranks, "clean_counts": clean_counts}, first


def check_arm_rows(frame: pd.DataFrame, arm: str, fold: str,
                   ranks: dict, clean_counts: dict,
                   allow_derived_delta127: bool = False) -> None:
    if "delta127" not in frame and allow_derived_delta127\
            and {"p22", "p23"}.issubset(frame.columns):
        frame["delta127"] = frame.p23.astype("float64") - frame.p22.astype("float64")
    required = {"MVT_ID_mvt", "fold", "airport", "utc_day", "y_true", "p12", "p22", "delta127", "p23",
                "pred_raw", "weight", "pred_blend", "gain_blend"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{arm}/{fold}: predictions lack {sorted(required - set(frame.columns))}")
    if frame.MVT_ID_mvt.duplicated().any() or not frame.fold.astype(str).eq(fold).all():
        raise RuntimeError(f"{arm}/{fold}: movement IDs repeat or fold name differs")
    if not np.isfinite(frame[["y_true", "p12", "p22", "delta127", "p23", "pred_raw",
                              "weight", "pred_blend", "gain_blend"]].to_numpy("float64")).all():
        raise RuntimeError(f"{arm}/{fold}: a saved score is non-finite")
    if not np.allclose(frame.p22.to_numpy("float64") + frame.delta127.to_numpy("float64"),
                       frame.p23.to_numpy("float64"), rtol=0, atol=1e-9):
        raise RuntimeError(f"{arm}/{fold}: p23 differs from p22 plus exp-127 delta")
    observed = frame.groupby(frame.airport.astype(str), observed=True).size().to_dict()
    for airport, count in clean_counts[fold].items():
        if observed.get(airport, 0) != count:
            raise RuntimeError(f"{arm}/{fold}/{airport}: clean row count differs from saved counts")
    if set(ranks[fold]) - set(observed):
        raise RuntimeError(f"{arm}/{fold}: one or more ranking airports lack score support")


def join_main(arm_data: dict[str, tuple], count_data: dict) -> dict[str, pd.DataFrame]:
    joined = {}
    ranks, clean_counts = count_data["ranks"], count_data["clean_counts"]
    allow_derived_delta127 = all(
        arm_data[arm][1].get("recovery", {}).get("delta127_reconstruction_allowed") is True
        for arm in ARMS
    )
    for fold in FOLDS:
        parts = {arm: arm_data[arm][2].loc[arm_data[arm][2].fold.astype(str).eq(fold)]
                 .reset_index(drop=True) for arm in ARMS}
        for arm, frame in parts.items():
            check_arm_rows(frame, arm, fold, ranks, clean_counts, allow_derived_delta127)
        ids = parts["control"].MVT_ID_mvt.to_numpy("int64")
        if any(not np.array_equal(parts[arm].MVT_ID_mvt.to_numpy("int64"), ids) for arm in ARMS[1:]):
            raise RuntimeError(f"{fold}: arms have different score IDs or row order")
        arm_specific = set().union(*(ARM_COLUMNS[arm] for arm in ARMS))
        common = [column for column in parts["control"].columns if column not in arm_specific]
        out = parts["control"][common].copy()
        for arm in ARMS:
            part = parts[arm]
            for column in common:
                if not out[column].reset_index(drop=True).equals(part[column].reset_index(drop=True)):
                    raise RuntimeError(f"{arm}/{fold}: shared score field differs: {column}")
            for column in sorted(ARM_COLUMNS[arm]):
                out[f"{column}_{arm}"] = part[column].to_numpy()
        for arm in ARMS:
            saved_weight = float(arm_data[arm][4].loc[
                arm_data[arm][4].fold.astype(str).eq(fold), "weight"].iloc[0])
            w = out[f"weight_{arm}"].to_numpy("float64")
            if not np.allclose(w, saved_weight, rtol=0, atol=1e-12):
                raise RuntimeError(f"{arm}/{fold}: row blend weights differ from weights.csv")
            blend = out.p23.to_numpy("float64") + saved_weight * (
                out[f"pred_raw_{arm}"].to_numpy("float64") - out.p23.to_numpy("float64"))
            if not np.allclose(blend, out[f"pred_blend_{arm}"].to_numpy("float64"), rtol=0, atol=1e-8):
                raise RuntimeError(f"{arm}/{fold}: saved blend does not match its scalar")
        y = out.y_true.to_numpy("float64")
        base_loss = (y - out.p23.to_numpy("float64")) ** 2
        for arm in ARMS:
            pred = out[f"pred_blend_{arm}"].to_numpy("float64")
            gain = base_loss - (y - pred) ** 2
            if not np.allclose(gain, out[f"gain_blend_{arm}"].to_numpy("float64"), rtol=0, atol=1e-7):
                raise RuntimeError(f"{arm}/{fold}: saved gain does not match blended predictions")
            out[f"gain_{arm}"] = gain
        losses = {arm: (y - out[f"pred_blend_{arm}"].to_numpy("float64")) ** 2 for arm in ARMS}
        out["proxy_minus_control"] = losses["control"] - losses["proxy"]
        out["proxy_minus_sham"] = losses["sham"] - losses["proxy"]
        joined[fold] = out
    return joined


def density_weighted_point(rows: pd.DataFrame, metric: str, fold: str, ranks: dict) -> float:
    means = rows.assign(_airport=rows.airport.astype(str)).groupby(
        "_airport", sort=True, observed=True)[metric].mean()
    total = sum(ranks[fold].values())
    return float(sum(rank_n * means[airport] for airport, rank_n in ranks[fold].items()) / total)


def bootstrap(joined: dict[str, pd.DataFrame], ranks: dict) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(BOOT_SEED)
    by_fold = {}
    points = {}
    for fold in FOLDS:
        rows = joined[fold]
        total_rank = sum(ranks[fold].values())
        draws = np.zeros((BOOT_DRAWS, len(BOOT_COLUMNS)), dtype="float64")
        for airport, ranking_rows in sorted(ranks[fold].items()):
            cell = rows.loc[rows.airport.astype(str).eq(airport)]
            groups = cell.groupby("utc_day", sort=True, observed=True)[list(BOOT_COLUMNS)]
            day_sums = groups.sum().to_numpy("float64")
            day_counts = groups.size().to_numpy("float64")
            if not len(day_sums):
                raise RuntimeError(f"{fold}/{airport}: no UTC-day bootstrap clusters")
            sampled = rng.integers(0, len(day_sums), size=(BOOT_DRAWS, len(day_sums)))
            sums = day_sums[sampled].sum(axis=1)
            counts = day_counts[sampled].sum(axis=1)
            draws += ranking_rows * (sums / counts[:, None]) / total_rank
        by_fold[fold] = draws
        points[fold] = {name: density_weighted_point(rows, name, fold, ranks)
                        for name in BOOT_COLUMNS}

    jan_n, jul_n = sum(ranks["january"].values()), sum(ranks["july_loo"].values())
    pooled = (by_fold["january"] * jan_n + by_fold["july_loo"] * jul_n) / (jan_n + jul_n)
    pooled_point = {name: (points["january"][name] * jan_n + points["july_loo"][name] * jul_n)
                    / (jan_n + jul_n) for name in BOOT_COLUMNS}
    records = []
    for scope in (*FOLDS, "pooled_jan_jul"):
        samples = pooled if scope == "pooled_jan_jul" else by_fold[scope]
        summary = pooled_point if scope == "pooled_jan_jul" else points[scope]
        for index, name in enumerate(BOOT_COLUMNS):
            records.append({"scope": scope, "metric": name, "point": float(summary[name]),
                            "lower_95": float(np.quantile(samples[:, index], 0.025)),
                            "upper_95": float(np.quantile(samples[:, index], 0.975)),
                            "draws": BOOT_DRAWS, "seed": BOOT_SEED})
    return pd.DataFrame(records), {"fold_draws": by_fold, "pooled_draws": pooled,
                                    "fold_points": points, "pooled_point": pooled_point}


def fold_metrics(joined: dict[str, pd.DataFrame], ranks: dict) -> pd.DataFrame:
    records = []
    for fold in FOLDS:
        rows = joined[fold]
        rec = {"scope": fold, "score_rows": int(len(rows)),
               "ranking_rows": int(sum(ranks[fold].values())),
               "p23_mse_s2_per_rank_row": float(density_weighted_point(
                   rows.assign(_p23=(rows.y_true - rows.p23) ** 2), "_p23", fold, ranks))}
        for arm in ARMS:
            rec[f"gain_{arm}_s2_per_rank_row"] = density_weighted_point(rows, f"gain_{arm}", fold, ranks)
            rec[f"blend_rmse_{arm}_s"] = float(np.sqrt(max(0.0, rec["p23_mse_s2_per_rank_row"]
                                                                    - rec[f"gain_{arm}_s2_per_rank_row"])))
            rec[f"weight_{arm}"] = float(rows[f"weight_{arm}"].iloc[0])
        rec["proxy_minus_control_s2_per_rank_row"] = density_weighted_point(rows, "proxy_minus_control", fold, ranks)
        rec["proxy_minus_sham_s2_per_rank_row"] = density_weighted_point(rows, "proxy_minus_sham", fold, ranks)
        records.append(rec)

    total_jan_jul = sum(sum(ranks[fold].values()) for fold in JAN_JUL)
    pooled = {"scope": "pooled_jan_jul", "score_rows": sum(len(joined[f]) for f in JAN_JUL),
              "ranking_rows": total_jan_jul}
    for metric in ("p23", *ARMS, "proxy_minus_control", "proxy_minus_sham"):
        if metric == "p23":
            vals = {fold: density_weighted_point(
                joined[fold].assign(_p23=(joined[fold].y_true - joined[fold].p23) ** 2),
                "_p23", fold, ranks) for fold in JAN_JUL}
            pooled["p23_mse_s2_per_rank_row"] = sum(vals[f] * sum(ranks[f].values()) for f in JAN_JUL) / total_jan_jul
        elif metric in ARMS:
            vals = {fold: density_weighted_point(joined[fold], f"gain_{metric}", fold, ranks)
                    for fold in JAN_JUL}
            gain = sum(vals[f] * sum(ranks[f].values()) for f in JAN_JUL) / total_jan_jul
            pooled[f"gain_{metric}_s2_per_rank_row"] = gain
            pooled[f"blend_rmse_{metric}_s"] = float(np.sqrt(max(
                0.0, pooled["p23_mse_s2_per_rank_row"] - gain)))
            pooled[f"weight_{metric}"] = float(pd.concat([joined[f] for f in JAN_JUL])[f"weight_{metric}"].iloc[0])
        else:
            vals = {fold: density_weighted_point(joined[fold], metric, fold, ranks) for fold in JAN_JUL}
            pooled[f"{metric}_s2_per_rank_row"] = sum(vals[f] * sum(ranks[f].values()) for f in JAN_JUL) / total_jan_jul
    records.append(pooled)
    return pd.DataFrame(records)


def tail_metrics(joined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat([joined[fold] for fold in FOLDS], ignore_index=True)
    scopes = [(fold, joined[fold]) for fold in FOLDS]
    scopes.append(("pooled_three_folds", all_rows))
    records = []
    for scope, source in scopes:
        tail = source.loc[source.y_true.gt(3600.0)]
        for airport, cell in tail.groupby(tail.airport.astype(str), sort=True, observed=True):
            values = {"scope": scope, "airport": airport, "tail_rows": int(len(cell)),
                      "v23_rmse_s": float(np.sqrt(np.mean((cell.y_true - cell.p23) ** 2))),
                      "lirf_report_separate": airport == "LIRF"}
            for arm in ARMS:
                values[f"{arm}_rmse_s"] = float(np.sqrt(np.mean(
                    (cell.y_true - cell[f"pred_blend_{arm}"]) ** 2)))
                values[f"{arm}_change_vs_v23_s"] = values[f"{arm}_rmse_s"] - values["v23_rmse_s"]
            records.append(values)
    return pd.DataFrame(records)


def verify_weights(arm_data: dict[str, tuple], joined: dict, ranks: dict,
                   clean_counts: dict) -> pd.DataFrame:
    rows_out = []
    for arm in ARMS:
        calibration = arm_data[arm][3]
        weights = arm_data[arm][4]
        for _, target, _ in ((0, "july_loo", 1), (1, "december", 2), (2, "january", 3)):
            sources = [fold for fold in FOLDS if fold != target]
            relevant = calibration.loc[calibration.fold.astype(str).isin(sources)].copy()
            # Each source fold occurs once in each of its two pair fits. Keep the fit that
            # contains the held fold, which leaves one unique calibration prediction per source.
            source_parts = []
            for source in sources:
                part = relevant.loc[relevant.fold.astype(str).eq(source)].copy()
                expected_pairs = {"july_loo/december", "july_loo/january", "december/january"}
                part = part.loc[part.calibration_pair.astype(str).isin(expected_pairs)]
                if part.empty:
                    raise RuntimeError(f"{arm}/{target}: missing calibration predictions for {source}")
                # There is one calibration learner per unordered pair; select the pair with target.
                pair_key = "/".join(fold for fold in FOLDS if fold in {target, source})
                part = part.loc[part.calibration_pair.astype(str).eq(pair_key)]
                if part.MVT_ID_mvt.duplicated().any():
                    raise RuntimeError(f"{arm}/{target}/{source}: repeated calibration score IDs")
                source_parts.append((source, part))
            numerator = denominator = 0.0
            for source, part in source_parts:
                airport = part.airport.astype(str)
                rank_n = airport.map(ranks[source]).fillna(0).to_numpy("float64")
                clean_n = airport.map(clean_counts[source]).fillna(0).to_numpy("float64")
                weights_by_row = np.divide(rank_n, clean_n, out=np.zeros_like(rank_n), where=clean_n > 0)
                delta = part.pred_raw.to_numpy("float64") - part.p23.to_numpy("float64")
                target_delta = part.y_true.to_numpy("float64") - part.p23.to_numpy("float64")
                numerator += float(np.sum(weights_by_row * delta * target_delta))
                denominator += float(np.sum(weights_by_row * delta * delta))
            raw = numerator / denominator if denominator > 0 else 0.0
            recomputed = float(np.clip(raw, 0.0, 1.0))
            saved_row = weights.loc[weights.fold.astype(str).eq(target)]
            if len(saved_row) != 1 or not np.isclose(recomputed, float(saved_row.weight.iloc[0]), rtol=0, atol=1e-10):
                raise RuntimeError(f"{arm}/{target}: saved cross-fit weight differs from calibration rows")
            rows_out.append({"arm": arm, "target": target, "source_folds": ",".join(sources),
                             "weight": recomputed, "unclipped_weight": raw,
                             "numerator": numerator, "denominator": denominator})

        numerator = denominator = 0.0
        for source in JAN_JUL:
            part = joined[source]
            airport = part.airport.astype(str)
            rank_n = airport.map(ranks[source]).fillna(0).to_numpy("float64")
            clean_n = airport.map(clean_counts[source]).fillna(0).to_numpy("float64")
            weights_by_row = np.divide(rank_n, clean_n, out=np.zeros_like(rank_n), where=clean_n > 0)
            delta = part[f"pred_raw_{arm}"].to_numpy("float64") - part.p23.to_numpy("float64")
            target_delta = part.y_true.to_numpy("float64") - part.p23.to_numpy("float64")
            numerator += float(np.sum(weights_by_row * delta * target_delta))
            denominator += float(np.sum(weights_by_row * delta * delta))
        raw = numerator / denominator if denominator > 0 else 0.0
        recomputed = float(np.clip(raw, 0.0, 1.0))
        saved_row = weights.loc[weights.fold.astype(str).eq("ranking_jan_jul")]
        if len(saved_row) != 1 or not np.isclose(recomputed, float(saved_row.weight.iloc[0]), rtol=0, atol=1e-10):
            raise RuntimeError(f"{arm}: saved Jan/Jul ranking weight differs from main OOF rows")
        rows_out.append({"arm": arm, "target": "ranking_jan_jul", "source_folds": "january,july_loo",
                         "weight": recomputed, "unclipped_weight": raw,
                         "numerator": numerator, "denominator": denominator})
    return pd.DataFrame(rows_out)


def make_gate(metrics: pd.DataFrame, intervals: pd.DataFrame, tails: pd.DataFrame) -> dict:
    metric_map = metrics.set_index("scope")
    interval_map = intervals.set_index(["scope", "metric"])
    scopes = (*FOLDS, "pooled_jan_jul")
    positive = all(float(metric_map.loc[fold, "gain_proxy_s2_per_rank_row"]) > 0 for fold in FOLDS)
    gain_intervals = all(float(interval_map.loc[(scope, "gain_proxy"), "lower_95"]) > 0
                         for scope in (*FOLDS, "pooled_jan_jul"))
    beats_controls = all(
        float(metric_map.loc[scope, "proxy_minus_control_s2_per_rank_row"]
              if scope != "pooled_jan_jul" else metric_map.loc[scope, "proxy_minus_control_s2_per_rank_row"]) > 0
        and float(metric_map.loc[scope, "proxy_minus_sham_s2_per_rank_row"]
                  if scope != "pooled_jan_jul" else metric_map.loc[scope, "proxy_minus_sham_s2_per_rank_row"]) > 0
        for scope in scopes)
    paired_intervals = all(float(interval_map.loc[("pooled_jan_jul", name), "lower_95"]) > 0
                           for name in PAIR_COLUMNS)
    d107_pass = positive and gain_intervals and beats_controls and paired_intervals
    pooled_tail = tails.loc[tails.scope.astype(str).eq("pooled_three_folds")]
    tail_ok = bool(not pooled_tail.empty and (pooled_tail.proxy_change_vs_v23_s <= 1.0).all())
    return {
        "experiment": "exp-137-withclock-proxies", "spec_sha256": EXPECTED_SPEC_SHA,
        "state": "D107_PASS" if d107_pass else "D107_FAIL",
        "d107_pass": bool(d107_pass),
        "score_scope": "supported airport-fold cells only",
        "clauses": {"positive_gain_each_fold": bool(positive),
                    "proxy_gain_lower_95_above_zero_each_fold_and_pool": bool(gain_intervals),
                    "proxy_point_beats_control_and_sham_each_fold_and_pool": bool(beats_controls),
                    "pooled_proxy_minus_control_and_sham_lower_95_above_zero": bool(paired_intervals)},
        "d20_tail": {"pass": tail_ok, "pooled_airports_with_support": int(len(pooled_tail)),
                     "max_proxy_rmse_change_s": (float(pooled_tail.proxy_change_vs_v23_s.max())
                                                  if not pooled_tail.empty else None)},
        "ranking_delta_allowed": bool(d107_pass),
        "ranking_target_read": False, "weather_used_or_downloaded": False,
    }


def board_price(joined: dict, ranks: dict, metrics: pd.DataFrame,
                ranking_counts: pd.DataFrame) -> dict:
    v22_by_fold, v23_by_fold = {}, {}
    for fold in JAN_JUL:
        rows = joined[fold]
        v22_by_fold[fold] = density_weighted_point(
            rows.assign(_loss=(rows.y_true - rows.p22) ** 2), "_loss", fold, ranks)
        v23_by_fold[fold] = density_weighted_point(
            rows.assign(_loss=(rows.y_true - rows.p23) ** 2), "_loss", fold, ranks)
    weights_n = {fold: sum(ranks[fold].values()) for fold in JAN_JUL}
    total = sum(weights_n.values())
    mse_v22 = sum(weights_n[f] * v22_by_fold[f] for f in JAN_JUL) / total
    mse_v23 = sum(weights_n[f] * v23_by_fold[f] for f in JAN_JUL) / total
    pooled = metrics.set_index("scope").loc["pooled_jan_jul"]
    gain = float(pooled.gain_proxy_s2_per_rank_row)
    supported_fraction = gain / mse_v23
    jan_jul_counts = ranking_counts.loc[
        ranking_counts.fold.astype(str).isin(JAN_JUL)]
    supported_rows = int(jan_jul_counts.loc[jan_jul_counts.supported.astype(bool), "ranking_rows"].sum())
    total_rows = int(jan_jul_counts.ranking_rows.sum())
    coverage = supported_rows / total_rows
    fraction = supported_fraction * coverage
    share = 0.23301073748872064 * (mse_v23 / mse_v22) * (245.0207 / 242.8064) ** 2
    domain = 1.0 - share * fraction
    score_reduction = 242.8064 * (1.0 - np.sqrt(domain)) if domain >= 0 else float("nan")
    return {"type": "transfer estimate, not a measured board score", "census_with_clock_share": 0.23301073748872064,
            "v22_mse_by_fold": v22_by_fold, "v23_mse_by_fold": v23_by_fold,
            "ranking_rows_by_fold": weights_n, "pooled_v22_mse_s2_per_rank_row": float(mse_v22),
            "pooled_v23_mse_s2_per_rank_row": float(mse_v23), "v23_over_v22_mse_ratio": float(mse_v23 / mse_v22),
            "v24_adjustment_ratio_squared": float((245.0207 / 242.8064) ** 2),
            "estimated_v24_with_clock_error_share": float(share),
            "proxy_gain_over_v23_s2_per_rank_row": gain,
            "supported_cell_gain_fraction_of_v23_mse": float(supported_fraction),
            "supported_jan_jul_ranking_rows": supported_rows,
            "all_jan_jul_with_clock_ranking_rows": total_rows,
            "supported_jan_jul_ranking_coverage": float(coverage),
            "coverage_adjusted_gain_fraction_assuming_zero_unsupported_gain": float(fraction),
            "square_root_domain": float(domain), "square_root_domain_valid": bool(domain >= 0),
            "estimated_score_reduction_s": float(score_reduction),
            "estimated_score_from_v24_242_8064": float(242.8064 - score_reduction) if domain >= 0 else None}


def run(run_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_data = {arm: read_arm(run_dir, arm) for arm in ARMS}
    allow_derived_delta127 = all(
        arm_data[arm][1].get("recovery", {}).get("delta127_reconstruction_allowed") is True
        for arm in ARMS
    )
    count_data, ranking_counts = validate_counts(arm_data)
    ranks, clean_counts = count_data["ranks"], count_data["clean_counts"]
    for arm in ARMS:
        base = pd.read_csv(run_dir / arm / "output" / "base_reproduction.csv")
        if len(base) != 3 or set(base.fold.astype(str)) != set(FOLDS):
            raise RuntimeError(f"{arm}: base reproduction is incomplete")
        if base.absolute_error_s.max() > 1e-6:
            raise RuntimeError(f"{arm}: source full-fold v23 RMSE differs by more than 1e-6 s")
    joined = join_main(arm_data, count_data)
    weight_table = verify_weights(arm_data, joined, ranks, clean_counts)
    metrics = fold_metrics(joined, ranks)
    intervals, draws = bootstrap(joined, ranks)
    tails = tail_metrics(joined)
    gate = make_gate(metrics, intervals, tails)
    price = board_price(joined, ranks, metrics, ranking_counts)

    diagnostic = []
    if not gate["d107_pass"]:
        for fold in FOLDS:
            rows = joined[fold].copy()
            rows["pred_50"] = rows.p23 + 0.5 * (rows.pred_raw_proxy - rows.p23)
            rows["gain_50"] = (rows.y_true - rows.p23) ** 2 - (rows.y_true - rows.pred_50) ** 2
            diagnostic.append({"scope": fold, "gain_50_s2_per_rank_row": density_weighted_point(
                rows, "gain_50", fold, ranks), "rows": int(len(rows))})
        jan_n, jul_n = sum(ranks["january"].values()), sum(ranks["july_loo"].values())
        total = jan_n + jul_n
        points = {row["scope"]: row["gain_50_s2_per_rank_row"] for row in diagnostic}
        diagnostic.append({"scope": "pooled_jan_jul", "gain_50_s2_per_rank_row":
                           (points["january"] * jan_n + points["july_loo"] * jul_n) / total,
                           "rows": len(joined["january"]) + len(joined["july_loo"])})

    pd.concat([joined[fold] for fold in FOLDS], ignore_index=True).to_parquet(
        output_dir / "predictions.parquet", index=False, compression="zstd")
    base_checks = pd.concat([
        pd.read_csv(run_dir / arm / "output" / "base_reproduction.csv").assign(arm=arm)
        for arm in ARMS
    ], ignore_index=True)
    base_checks.to_csv(output_dir / "base_reproduction.csv", index=False)
    ranking_counts.to_csv(output_dir / "ranking_counts.csv", index=False)
    weight_table.to_csv(output_dir / "verified_weights.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    intervals.to_csv(output_dir / "intervals.csv", index=False)
    tails.to_csv(output_dir / "tail_metrics.csv", index=False)
    if diagnostic:
        pd.DataFrame(diagnostic).to_csv(output_dir / "fixed_50_50_diagnostic.csv", index=False)
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    (output_dir / "board_price.json").write_text(json.dumps(price, indent=2) + "\n")
    if gate["d107_pass"]:
        proxy_ranking_weight = float(arm_data["proxy"][4].loc[
            arm_data["proxy"][4].fold.astype(str).eq("ranking_jan_jul"), "weight"].iloc[0])
        final_weights = {
            "experiment": "exp-137-withclock-proxies",
            "spec_sha256": EXPECTED_SPEC_SHA,
            "d107_pass": True,
            "proxy_ranking_weight": proxy_ranking_weight,
            "weight_scope": "supported January and July airport-fold cells",
            "supported_jan_jul_ranking_coverage": price["supported_jan_jul_ranking_coverage"],
        }
        (output_dir / "final_weights.json").write_text(
            json.dumps(final_weights, indent=2) + "\n")

    input_manifests = {arm: {"kernel_metadata_id": arm_data[arm][1].get("kernel_slug"),
                             "kernel_url_slug": kernel_url_slugs()[arm],
                             "kernel_version": arm_data[arm][1].get("kernel_version"),
                             "dataset_versions": arm_data[arm][1].get("dataset_versions"),
                             "output_sha256": arm_data[arm][1].get("output_sha256"),
                             "manifest_sha256": sha256(run_dir / arm / "output" / "manifest.json")}
                       for arm in ARMS}
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "manifest.json"}
    combined_manifest = {"experiment": "exp-137-withclock-proxies", "state": gate["state"],
                         "spec_sha256": EXPECTED_SPEC_SHA, "ranking_target_read": False,
                         "weather_used_or_downloaded": False, "arms": input_manifests,
                         "recovery": {"delta127_reconstructed_from_saved_p23_minus_p22": allow_derived_delta127},
                         "ranking_support": ranking_counts.to_dict(orient="records"),
                         "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED},
                         "files": files}
    (output_dir / "manifest.json").write_text(json.dumps(combined_manifest, indent=2) + "\n")
    return {"gate": gate, "price": price, "metrics": metrics.to_dict(orient="records"),
            "intervals": intervals.to_dict(orient="records"),
            "tail_metrics": tails.to_dict(orient="records"),
            "fixed_50_50_diagnostic": diagnostic,
            "verified_weights": weight_table.to_dict(orient="records"),
            "arm_manifests": input_manifests, "combined_files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "kaggle/run-v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "kaggle/run-v3/combined")
    args = parser.parse_args()
    result = run(args.run_dir.resolve(), args.output_dir.resolve())
    print(json.dumps({"state": result["gate"]["state"], "d107_pass": result["gate"]["d107_pass"],
                      "price": result["price"], "combined_files": result["combined_files"]}, indent=2))


if __name__ == "__main__":
    main()
