#!/usr/bin/env python3
"""Join frozen exp-136 arm outputs and apply the unchanged frozen gain gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop("PRC_EXP136_ARM", None)
import fit_exp136 as FIT  # noqa: E402


ARMS = FIT.ALL_ARM_NAMES
def kernel_url_slugs():
    return {
    "control": f"{kaggle_user()}/prc-exp136-control-arm",
    "t2": f"{kaggle_user()}/prc-exp136-t2-arm",
    "wide": f"{kaggle_user()}/prc-exp136-wide-arm",
    "sham": f"{kaggle_user()}/prc-exp136-sham-arm",
}

FOLDS = tuple(item[0] for item in FIT.FOLDS)
SCORE_COLUMNS = {
    arm: {
        f"pred_{arm}", f"pred_blend_{arm}",
        f"gain_raw_{arm}", f"gain_blend_{arm}",
    }
    for arm in ARMS
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_arm(run_dir: Path, arm: str) -> tuple[dict, dict, dict, dict]:
    output = run_dir / arm / "output"
    result_path = output / "results.json"
    manifest_path = output / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{arm}: missing results.json or manifest.json in {output}")
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if result.get("state") != "arm_complete_no_gate" or result.get("arm") != arm:
        raise RuntimeError(f"{arm}: output is not a complete arm-only fit")
    if manifest.get("state") != "arm_complete_no_gate" or manifest.get("arm") != arm:
        raise RuntimeError(f"{arm}: manifest does not match its arm")
    if result.get("feature_sha256") != FIT.FEATURE_SHA256:
        raise RuntimeError(f"{arm}: feature design hash differs from the frozen runner")
    if manifest.get("feature_sha256") != FIT.FEATURE_SHA256:
        raise RuntimeError(f"{arm}: manifest feature design hash differs")
    if result.get("spec_sha256") != manifest.get("spec_sha256"):
        raise RuntimeError(f"{arm}: result and manifest SPEC hashes differ")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = output / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{arm}: output hash check failed for {name}")
    missing_folds = [fold for fold in FOLDS
                     if not (output / f"predictions_{fold}.parquet").is_file()]
    if missing_folds:
        raise RuntimeError(f"{arm}: missing fold predictions {missing_folds}")
    return result, manifest, {
        fold: pd.read_parquet(output / f"predictions_{fold}.parquet")
        for fold in FOLDS
    }, {
        "weights": pd.read_csv(output / "weights.csv"),
        "ranking_counts": pd.read_csv(output / "ranking_counts.csv"),
        "coverage": pd.read_csv(output / "proxy_coverage.csv"),
        "fit_info": pd.read_csv(output / "fit_info.csv"),
        "feature_importance": (
            pd.read_csv(output / "feature_importance.csv")
            if (output / "feature_importance.csv").exists() else pd.DataFrame()
        ),
    }


def join_scores(arm_frames: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    joined = {}
    for fold in FOLDS:
        first = arm_frames[ARMS[0]][fold].reset_index(drop=True)
        if first.MVT_ID_mvt.duplicated().any():
            raise RuntimeError(f"{ARMS[0]}/{fold}: duplicate score IDs")
        frame = first[[column for column in first.columns
                       if column not in SCORE_COLUMNS[ARMS[0]]]].copy()
        expected_ids = first.MVT_ID_mvt.to_numpy("int64")
        common_columns = list(frame.columns)
        for arm in ARMS:
            part = arm_frames[arm][fold].reset_index(drop=True)
            if part.MVT_ID_mvt.duplicated().any():
                raise RuntimeError(f"{arm}/{fold}: duplicate score IDs")
            if not np.array_equal(part.MVT_ID_mvt.to_numpy("int64"), expected_ids):
                raise RuntimeError(f"{arm}/{fold}: score IDs or row order differ")
            for column in common_columns:
                left = frame[column].reset_index(drop=True)
                right = part[column].reset_index(drop=True)
                if not left.equals(right):
                    raise RuntimeError(f"{arm}/{fold}: shared score field differs: {column}")
            expected = SCORE_COLUMNS[arm]
            if not expected.issubset(part.columns):
                raise RuntimeError(f"{arm}/{fold}: missing arm score columns")
            for column in sorted(expected):
                frame[column] = part[column].to_numpy()
        joined[fold] = frame
    return joined


def join_weights(arm_details: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    weights = {}
    rows = []
    for arm in ARMS:
        frame = arm_details[arm]["weights"]
        if len(frame) != len(FOLDS) + 1 or not frame.arm.astype(str).eq(arm).all():
            raise RuntimeError(f"{arm}: weight table must have three fold rows and one rank row")
        for row in frame.to_dict(orient="records"):
            target = str(row["target_fold"])
            if target != "rank" and target not in FOLDS:
                raise RuntimeError(f"{arm}: unexpected target fold {target}")
            weights.setdefault(target, {})[arm] = {
                "arm": arm,
                "weight": float(row["weight"]),
                "unclipped_weight": float(row["unclipped_weight"]),
                "numerator": float(row["numerator"]),
                "denominator": float(row["denominator"]),
                "rows": int(row["rows"]),
            }
            rows.append(row)
    return weights, pd.DataFrame(rows)


def join_rank_counts(arm_details: dict[str, dict]) -> dict:
    first = arm_details[ARMS[0]]["ranking_counts"].sort_values(
        ["fold", "airport"], kind="stable").reset_index(drop=True)
    ranks = {}
    for arm in ARMS:
        frame = arm_details[arm]["ranking_counts"].sort_values(
            ["fold", "airport"], kind="stable").reset_index(drop=True)
        if not first.equals(frame):
            raise RuntimeError(f"{arm}: ranking row counts differ from {ARMS[0]}")
    for row in first.itertuples(index=False):
        ranks.setdefault(str(row.fold), {})[str(row.airport)] = int(row.ranking_rows)
    if set(ranks) != set(FOLDS):
        raise RuntimeError("ranking counts do not cover all three folds")
    return ranks


def board_prices(metrics: pd.DataFrame, census_path: Path) -> tuple[float, dict]:
    census = pd.read_csv(census_path)
    required = {"estimated_ranking_sse_clean_s2", "class"}
    if not required.issubset(census.columns):
        raise RuntimeError(f"census is missing columns {sorted(required - set(census.columns))}")
    sse = pd.to_numeric(census.estimated_ranking_sse_clean_s2, errors="coerce")
    total_sse = float(sse.sum())
    no_clock_sse = float(sse.loc[census["class"].eq("no-clock_ordinary")].sum())
    share = no_clock_sse / total_sse if total_sse > 0 else np.nan
    pooled = metrics.loc[metrics.scope.eq("pooled_jan_jul")].iloc[0]
    baseline = float(pooled.p12_mse_s2_per_rank_row)
    control_gain = float(pooled.gain_blend_control)
    control_mse = baseline - control_gain
    v22_board = 245.0207
    v24_board = 242.8064
    class_share_v24 = (
        share * (1.0 - control_gain / baseline) * (v22_board / v24_board) ** 2
        if baseline > 0 and np.isfinite(share) else np.nan
    )
    prices = {}
    for arm in ("t2", "wide"):
        increment = float(pooled[f"{arm}_vs_control"])
        fraction = increment / control_mse if control_mse > 0 else np.nan
        product = class_share_v24 * fraction
        in_domain = bool(np.isfinite(product) and 1.0 - product >= 0.0)
        prices[arm] = {
            "incremental_gain_s2_per_rank_row": increment,
            "estimated_no_clock_share_of_v24_sse": class_share_v24,
            "estimated_fraction_of_v24_sse_reduced": product,
            "board_price_seconds": (
                float(v24_board * (1.0 - np.sqrt(1.0 - product)))
                if in_domain else None
            ),
            "price_domain_ok": in_domain,
        }
    return share, prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="run-v3 directory with one folder per arm")
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arm_results = {}
    arm_manifests = {}
    arm_frames = {}
    arm_details = {}
    for arm in ARMS:
        result, manifest, frames, details = read_arm(args.run_dir, arm)
        arm_results[arm] = result
        arm_manifests[arm] = manifest
        arm_frames[arm] = frames
        arm_details[arm] = details
    spec_hashes = {manifest["spec_sha256"] for manifest in arm_manifests.values()}
    if len(spec_hashes) != 1:
        raise RuntimeError("arm kernels used different frozen SPEC hashes")

    scores = join_scores(arm_frames)
    weights, weights_frame = join_weights(arm_details)
    ranks = join_rank_counts(arm_details)
    clean_counts = FIT.verify_scoring_support(scores, ranks)
    FIT.attach_blends(scores, weights)
    metrics = FIT.metrics_table(scores, ranks)
    _, intervals = FIT.cluster_bootstrap(scores, ranks)
    interval_frame = pd.DataFrame(intervals)
    gate = FIT.wider_proxy_gate(interval_frame, metrics)
    tail = FIT.tail_table(scores)
    share, prices = board_prices(metrics, args.census)
    selected = gate["selected_arm"]
    final_weights = None
    if selected is not None:
        final_weights = {
            "experiment": "exp-136-wider-proxies",
            "state": "gate_pass_final_refit_authorized",
            "selected_arm": selected,
            "spec_sha256": next(iter(spec_hashes)),
            "feature_sha256": FIT.FEATURE_SHA256,
            "gate": gate,
            "rank_weights": {
                "control": float(weights["rank"]["control"]["weight"]),
                selected: float(weights["rank"][selected]["weight"]),
            },
        }
        (args.output_dir / "final_weights.json").write_text(
            json.dumps(final_weights, indent=2) + "\n")

    coverage = arm_details[ARMS[0]]["coverage"]
    for arm in ARMS[1:]:
        if not coverage.equals(arm_details[arm]["coverage"]):
            raise RuntimeError(f"{arm}: proxy coverage differs from {ARMS[0]}")
    fit_info = pd.concat(
        [arm_details[arm]["fit_info"] for arm in ARMS], ignore_index=True)
    feature_importance = pd.concat(
        [arm_details[arm]["feature_importance"] for arm in ARMS], ignore_index=True)

    for fold, frame in scores.items():
        frame.to_parquet(args.output_dir / f"predictions_{fold}.parquet",
                         index=False, compression="zstd")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    interval_frame.to_csv(args.output_dir / "intervals.csv", index=False)
    weights_frame.to_csv(args.output_dir / "weights.csv", index=False)
    tail.to_csv(args.output_dir / "tail_metrics.csv", index=False)
    coverage.to_csv(args.output_dir / "proxy_coverage.csv", index=False)
    fit_info.to_csv(args.output_dir / "fit_info.csv", index=False)
    feature_importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    (args.output_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    pooled = metrics.loc[metrics.scope.eq("pooled_jan_jul")].iloc[0]
    result = {
        "experiment": "exp-136-wider-proxies",
        "state": "gate_complete",
        "verdict": gate["verdict"],
        "gate": gate,
        "selected_arm": selected,
        "source_ledger_rows": ["861"],
        "source_result": "Exp-134 runway-state proxies passed D107 and became v24; row 861 records its measured gains.",
        "spec_sha256": next(iter(spec_hashes)),
        "feature_sha256": FIT.FEATURE_SHA256,
        "cross_fitted_weights": {
            fold: weights[fold] for fold in FOLDS
        },
        "rank_weights": weights["rank"],
        "metrics": metrics.to_dict(orient="records"),
        "intervals": intervals,
        "supported_no_clock_share_v22_sse": share,
        "board_prices_on_v24": prices,
        "selected_board_price_seconds": (
            prices[selected]["board_price_seconds"] if selected else None
        ),
        "final_refit": (
            {"state": "pending", "weights_file": "final_weights.json"}
            if selected else {"state": "not_run_gate_fail"}
        ),
        "v24_score_seconds": 242.8064,
        "control_gain_over_p12_s2_per_rank_row": float(pooled.gain_blend_control),
        "ranking_delta_written": False,
        "proxy_feature_coverage": coverage.to_dict(orient="records"),
        "clean_score_rows_by_fold": {
            fold: len(frame) for fold, frame in scores.items()
        },
        "ranking_rows_by_fold": {
            fold: int(sum(ranks[fold].values())) for fold in FOLDS
        },
        "arm_kernels": {
            arm: {
                "slug": kernel_url_slugs()[arm],
                "wrapper_slug": arm_manifests[arm].get("kernel_slug"),
                "version": arm_manifests[arm].get("kernel_version"),
                "dataset_version": arm_manifests[arm]["dataset_sources"]["exp136"]["version"],
                "manifest_sha256": sha256(args.run_dir / arm / "output" / "manifest.json"),
            }
            for arm in ARMS
        },
        "ranking_target_read": False,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "experiment": "exp-136-wider-proxies",
        "state": "gate_complete",
        "spec_sha256": result["spec_sha256"],
        "feature_sha256": FIT.FEATURE_SHA256,
        "arms": list(ARMS),
        "arm_manifests": {
            arm: arm_manifests[arm] for arm in ARMS
        },
        "output_sha256": output_hashes,
        "clean_score_counts": clean_counts,
        "bootstrap_draws": FIT.BOOT_DRAWS,
        "bootstrap_seed": FIT.BOOT_SEED,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n")
    print(json.dumps({
        "verdict": gate["verdict"],
        "selected_arm": selected,
        "metrics": metrics.to_dict(orient="records"),
        "board_prices_on_v24": prices,
        "outputs": output_hashes,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
