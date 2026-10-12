#!/usr/bin/env python3
"""Fit exp-137 all-year proxy under positive-point rule after its positive point estimate."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

def run_kernel():
    INPUT = Path("/kaggle/input")
    OUTPUT = Path("/kaggle/working/exp137-d132-ranking-output")
    ID = "MVT_ID_mvt"
    MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 13))
    RANK_MONTHS = ("2026-01", "2026-07")
    CLOCK_COLS = ["clock_current250_row", "clock_implied_row"]
    SPEC_SHA256 = "71ca0e05d8cacaaeeac26667e3d8f73449e0772df17596a4e13d92b2255ce689"
    D132_MAIN_COMMIT = "88b8660586c66b2662c8b5dd92103d5d0c45935b"
    def kernel_slug():
        return f"{kaggle_user()}/prc-exp137-with-clock-final-proxy"

    KERNEL_VERSION = "1"
    RANK_WEIGHT = 0.12797310003895088
    SUPPORTED_AIRPORTS_BY_MONTH = {
        "2026-01": ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH"],
        "2026-07": ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"],
    }
    EXPECTED_SUPPORTED_ROWS = {"2026-01": 68716, "2026-07": 104823}


    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()


    def exact_file(name: str) -> Path:
        matches = sorted(path for path in INPUT.rglob(name) if path.is_file())
        if len(matches) != 1:
            raise RuntimeError(f"expected one attached {name}, found {len(matches)}")
        return matches[0]


    # Load the frozen proxy builder from the attached exp-137 input by exact file.
    PX_PATH = exact_file("proxy_features_exp137.py")
    sys.path.insert(0, str(PX_PATH.parent))
    import proxy_features_exp137 as PX


    def base_fit_module():
        path = exact_file("fit.py")
        spec = importlib.util.spec_from_file_location("exp130_fit_for_exp137_final", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import the frozen exp-130 fit source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, path


    def add_clock_features(month_frames: dict, raw_months: dict) -> None:
        for month, frame in month_frames.items():
            source = raw_months[month]["row_features"].set_index(ID)
            ids = pd.to_numeric(frame[ID], errors="raise").astype("int64").to_numpy()
            selected = source.reindex(ids)
            if len(selected) != len(frame):
                raise RuntimeError(f"{month}: row-level clock inputs do not align")
            for column in CLOCK_COLS:
                frame[column] = pd.to_numeric(selected[column], errors="coerce").to_numpy("float32")


    def month_features(views: dict, months: list[str], rows: pd.DataFrame) -> pd.DataFrame:
        parts = [views[month][[ID, *PX.Z_COLS]] for month in months]
        values = pd.concat(parts, ignore_index=True)
        if values[ID].duplicated().any():
            raise RuntimeError("proxy feature IDs repeat across months")
        ids = pd.to_numeric(rows[ID], errors="raise").astype("int64").to_numpy()
        aligned = values.set_index(ID).reindex(ids)
        if len(aligned) != len(rows):
            raise RuntimeError("proxy features do not align with model rows")
        aligned.index = rows.index
        return aligned[PX.Z_COLS].astype("float32")


    def main() -> None:
        started = time.monotonic()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        weight_input = {
            "d107_pass": False,
            "d132_positive_point_override": True,
            "d132_main_commit": D132_MAIN_COMMIT,
            "d132_source_ledger_row": 872,
            "positive_proxy_gain_s2_per_supported_row": 191.3,
            "d20_tail_verdict": "FAIL",
            "spec_sha256": SPEC_SHA256,
            "proxy_ranking_weight": RANK_WEIGHT,
            "ranking_weight_source": "run-v3/combined/verified_weights.csv: proxy, ranking_jan_jul",
            "support_source": "run-v3/manifest.json: supported airport-fold cells",
            "supported_airports_by_month": SUPPORTED_AIRPORTS_BY_MONTH,
            "expected_supported_rows_by_month": EXPECTED_SUPPORTED_ROWS,
        }
        weight_path = OUTPUT / "ranking-weight.json"
        weight_path.write_text(json.dumps(weight_input, indent=2) + "\n")
        if weight_input.get("d107_pass") is not False or weight_input.get("d132_positive_point_override") is not True:
            raise RuntimeError("D132 positive-point override is not recorded")
        if weight_input.get("d132_main_commit") != D132_MAIN_COMMIT or weight_input.get("d132_source_ledger_row") != 872:
            raise RuntimeError("D132 decision or positive-result source changed")
        positive_gain = float(weight_input["positive_proxy_gain_s2_per_supported_row"])
        if not np.isfinite(positive_gain) or positive_gain <= 0.0:
            raise RuntimeError("D132 positive-point condition is not met")
        if weight_input.get("d20_tail_verdict") != "FAIL":
            raise RuntimeError("the prior exp-137 D20 tail verdict changed")
        if weight_input.get("spec_sha256") != SPEC_SHA256:
            raise RuntimeError("ranking weight SPEC hash differs from the frozen SPEC")
        rank_weight = float(weight_input["proxy_ranking_weight"])
        if rank_weight != RANK_WEIGHT or not np.isfinite(rank_weight) or not 0.0 <= rank_weight <= 1.0:
            raise RuntimeError("proxy ranking weight differs from the frozen cross-fitted value")
        if weight_input["supported_airports_by_month"] != SUPPORTED_AIRPORTS_BY_MONTH:
            raise RuntimeError("supported ranking cells differ from exp-137 pre-fit support")
        if weight_input["expected_supported_rows_by_month"] != EXPECTED_SUPPORTED_ROWS:
            raise RuntimeError("supported ranking row counts differ from exp-137 pre-fit support")

        inventory_path = exact_file("exp137_source_inventory.json")
        inventory = json.loads(inventory_path.read_text())
        if inventory.get("ranking_target_read") is not False or inventory.get("weather_used_or_downloaded") is not False:
            raise RuntimeError("exp-137 data inventory does not preserve label isolation and weather exclusion")
        proxy_inventory = json.loads(exact_file("source_inventory.json").read_text())
        if (proxy_inventory.get("ranking_target_read") is not False
                or proxy_inventory.get("departure_target_column_read") is not False
                or proxy_inventory.get("taxitime_read_for") != "ARR rows only"):
            raise RuntimeError("proxy inventory does not preserve the label-safety checks")

        # Rebuild proxy inputs and check the ranking support before reading 2025 labels.
        raw_months, proxy_audit = PX.build_all(MONTHS, RANK_MONTHS)
        base, fit_path = base_fit_module()
        D18, F, M, D = base.setup_modules()
        target_parts = {month: base.month_frame(month) for month in RANK_MONTHS}
        ranking_ids_by_month = {}
        support_audit_by_month = {}
        unsupported_january_ids = set()
        for month in RANK_MONTHS:
            frame = target_parts[month]
            if "y" in frame.columns or "y_true" in frame.columns:
                raise RuntimeError(f"{month}: ranking input frame contains a target label")
            if frame[ID].duplicated().any():
                raise RuntimeError(f"{month}: ranking inference IDs repeat")
            meta = pd.read_parquet(exact_file(f"rank_meta_{month}.parquet"))
            if meta[ID].duplicated().any():
                raise RuntimeError(f"{month}: ranking metadata IDs repeat")
            if month not in SUPPORTED_AIRPORTS_BY_MONTH:
                raise RuntimeError(f"no frozen support map exists for {month}")
            airport = meta.airport.astype(str)
            ordinary_clock = meta.rank_class.astype(str).eq("with-clock") & ~meta.noac.astype(bool)
            supported_clock = ordinary_clock & airport.isin(SUPPORTED_AIRPORTS_BY_MONTH[month])
            supported_ids = set(meta.loc[supported_clock, ID].astype("int64"))
            if len(supported_ids) != EXPECTED_SUPPORTED_ROWS[month]:
                raise RuntimeError(f"{month}: supported with-clock rows differ from the frozen support count")
            ordinary_frame_ids = set(frame.loc[frame.ghost.to_numpy("float64") < 0.5, ID].astype("int64"))
            if not supported_ids.issubset(ordinary_frame_ids):
                raise RuntimeError(f"{month}: supported IDs are missing from ordinary ranking inputs")
            ranking_ids_by_month[month] = supported_ids
            unsupported_rows = meta.loc[ordinary_clock & ~airport.isin(SUPPORTED_AIRPORTS_BY_MONTH[month])]
            support_audit_by_month[month] = {
                "supported_rows": len(supported_ids),
                "supported_airports": list(SUPPORTED_AIRPORTS_BY_MONTH[month]),
                "unsupported_rows_by_airport": {
                    str(airport_name): int(count)
                    for airport_name, count in unsupported_rows.groupby(unsupported_rows.airport.astype(str), observed=True).size().items()
                },
            }
            if month == "2026-01":
                unsupported_january_ids = set(unsupported_rows[ID].astype("int64"))
        if sum(len(ids) for ids in ranking_ids_by_month.values()) != sum(EXPECTED_SUPPORTED_ROWS.values()):
            raise RuntimeError("supported ranking IDs differ from the frozen January/July counts")
        preflight = {
            "ranking_target_read": False,
            "training_labels_read": False,
            "supported_rows_by_month": EXPECTED_SUPPORTED_ROWS,
            "support_audit_by_month": support_audit_by_month,
            "unsupported_january_ids": len(unsupported_january_ids),
            "spec_sha256": SPEC_SHA256,
            "d107_pass": False,
            "d132_positive_point_override": True,
        }
        (OUTPUT / "label-free-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        print("label-free ranking support checks passed before reading 2025 training labels", flush=True)

        month_frames = {month: base.month_frame(month) for month in MONTHS}
        month_frames.update(target_parts)
        add_clock_features(month_frames, raw_months)
        train_raw = pd.concat([month_frames[month] for month in MONTHS], ignore_index=True)
        train_raw = train_raw.loc[D.keep_reachable(train_raw)].copy()
        train_encoded, target_encoded_list = D18.add_encodings(
            train_raw, [target_parts[month] for month in RANK_MONTHS])
        ordinary_train = train_encoded.ghost.to_numpy("float64") < 0.5
        train = train_encoded.loc[ordinary_train].reset_index(drop=True)
        target_encoded = dict(zip(RANK_MONTHS, target_encoded_list))
        valid_parts = []
        expected_ids = []
        for month in RANK_MONTHS:
            part = target_encoded[month]
            ordinary = part.ghost.to_numpy("float64") < 0.5
            month_ids = ranking_ids_by_month[month]
            expected_ids.extend(month_ids)
            with_clock = part[ID].astype("int64").isin(month_ids).to_numpy()
            valid_parts.append(part.loc[ordinary & with_clock].copy())
        valid = pd.concat(valid_parts, ignore_index=True)
        if train.empty or valid.empty:
            raise RuntimeError("all-year ordinary training or supported with-clock ranking rows are empty")

        if len(set(expected_ids)) != len(expected_ids):
            raise RuntimeError("saved ranking metadata contains repeated supported with-clock IDs")
        if set(valid[ID].astype("int64")) != set(expected_ids):
            raise RuntimeError("learner ranking rows differ from the saved supported with-clock class")
        if valid[ID].duplicated().any():
            raise RuntimeError("supported ordinary with-clock ranking IDs repeat")

        anchors_train = base.anchor_for(train)
        target = train.y.to_numpy("float64")
        anchors_valid = base.anchor_for(valid)
        if not np.isfinite(anchors_train).all() or not np.isfinite(target).all() or not np.isfinite(anchors_valid).all():
            raise RuntimeError("all-year training target or anchor is non-finite")

        base_x_train, base_x_valid, base_columns, categorical = base.feature_frames(
            train, valid, F, M)
        if len(base_columns) != 151 or not set(CLOCK_COLS).issubset(base_columns):
            raise RuntimeError(f"expected 151 base and clock features, found {len(base_columns)}")
        denominators = PX.fit_denominators(raw_months, list(MONTHS))
        proxy_views = {}
        for month in (*MONTHS, *RANK_MONTHS):
            z, _ = PX.transform_rows(raw_months[month], denominators)
            proxy_views[month] = z[[ID, *PX.Z_COLS]]
        z_train = month_features(proxy_views, list(MONTHS), train)
        z_valid = month_features(proxy_views, list(RANK_MONTHS), valid)
        x_train = pd.concat([base_x_train.reset_index(drop=True), z_train.reset_index(drop=True)], axis=1)
        x_valid = pd.concat([base_x_valid.reset_index(drop=True), z_valid.reset_index(drop=True)], axis=1)
        if x_train.shape[1] != 172 or x_valid.shape[1] != 172:
            raise RuntimeError("all-year proxy fit must use 172 features")

        prediction, record = base.fit_arm(
            "all_2025_proxy", x_train, x_valid, target - anchors_train,
            np.ones(len(train), dtype="float32"), categorical, anchors_valid, M, "all_2025")
        if not np.isfinite(prediction).all():
            raise RuntimeError("all-year proxy prediction contains a non-finite value")

        fold_base = pd.read_parquet(exact_file("ranking_base_v24.parquet"))
        if fold_base[ID].duplicated().any():
            raise RuntimeError("saved v24 ranking base has duplicate movement IDs")
        if not np.array_equal(fold_base.row_order.to_numpy("int64"), np.arange(len(fold_base), dtype="int64")):
            raise RuntimeError("saved v24 ranking base row_order is not submitting.parquet order")
        submitting = pd.read_parquet(exact_file("submitting_ids.parquet"))
        if submitting[ID].duplicated().any():
            raise RuntimeError("submitting.parquet IDs are not unique")
        submitting = submitting.sort_values("row_order", kind="stable")
        if not np.array_equal(submitting.row_order.to_numpy("int64"), np.arange(len(submitting), dtype="int64")):
            raise RuntimeError("submitting ID row order is incomplete")
        ordered_ids = submitting[ID].astype("int64").to_numpy()
        base_order_ids = fold_base.sort_values("row_order", kind="stable")[ID].astype("int64").to_numpy()
        if not np.array_equal(ordered_ids, base_order_ids):
            raise RuntimeError("saved v24 base IDs differ from submitting.parquet order")
        base_map = fold_base.set_index(ID).p24
        ids = valid[ID].astype("int64").to_numpy()
        if not pd.Index(ids).isin(base_map.index).all():
            raise RuntimeError("v24 base does not contain every ordinary with-clock ranking row")
        base_prediction = base_map.reindex(ids).to_numpy("float64")
        if not np.isfinite(base_prediction).all():
            raise RuntimeError("v24 prediction is non-finite on ordinary with-clock ranking rows")
        delta_values = rank_weight * (prediction - base_prediction)
        delta_by_id = pd.Series(delta_values, index=ids)
        target_mask = pd.Index(ordered_ids).isin(ids)
        if int(target_mask.sum()) != len(ids):
            raise RuntimeError("submitting order has missing or repeated target IDs")
        delta = delta_by_id.reindex(ordered_ids).fillna(0.0).to_numpy("float64")
        if len(delta) != len(ordered_ids) or not np.isfinite(delta).all():
            raise RuntimeError("ranking delta does not cover the v24 row order")
        if int(target_mask.sum()) != sum(EXPECTED_SUPPORTED_ROWS.values()):
            raise RuntimeError("ranking delta support mask differs from the frozen January/July cells")
        if np.any(delta[~target_mask] != 0.0):
            raise RuntimeError("ranking delta is nonzero outside supported ordinary with-clock rows")
        if pd.Index(ordered_ids[delta != 0.0]).isin(unsupported_january_ids).any():
            raise RuntimeError("unsupported January ranking rows have nonzero delta")
        expected_delta = delta_by_id.reindex(ordered_ids[target_mask]).to_numpy("float64")
        if not np.allclose(delta[target_mask], expected_delta, rtol=0, atol=1e-10):
            raise RuntimeError("ranking delta ID alignment differs from learner output")

        delta_path = OUTPUT / "ranking_delta.npz"
        np.savez_compressed(delta_path, MVT_ID_mvt=ordered_ids.astype("int64"), delta=delta)
        output_manifest = {
            "experiment": "exp-137-withclock-proxies", "state": "ranking_delta_written_by_D132",
            "spec_sha256": weight_input["spec_sha256"], "d107_pass": False,
            "fixed_d107_verdict": "FAIL", "d132_positive_point_override": True,
            "fixed_d20_tail_verdict": "FAIL",
            "d132_main_commit": D132_MAIN_COMMIT, "source_ledger_row": 872,
            "source_positive_gain_s2_per_supported_row": positive_gain,
            "ranking_target_read": False, "training_labels_read": True,
            "weather_used_or_downloaded": False,
            "candidate_parquet_created": False, "competition_upload_made": False,
            "kernel_slug": os.environ.get("PRC_KERNEL_SLUG", kernel_slug()),
            "kernel_version": os.environ.get("PRC_KERNEL_VERSION", KERNEL_VERSION),
            "dataset_versions": {
                "prc-exp130-clean-input": os.environ.get("PRC_EXP130_DATASET_VERSION", "1"),
                "prc-exp136-input": os.environ.get("PRC_EXP136_DATASET_VERSION", "4"),
                "prc-exp137-input": os.environ.get("PRC_EXP137_DATASET_VERSION", "3"),
            },
            "rank_weight": rank_weight, "ranking_class_rows": int(target_mask.sum()),
            "supported_rows_by_month": EXPECTED_SUPPORTED_ROWS,
            "support_audit_by_month": support_audit_by_month,
            "unsupported_january_rows": len(unsupported_january_ids),
            "supported_airports_by_month": SUPPORTED_AIRPORTS_BY_MONTH,
            "ranking_rows_moved": int(np.count_nonzero(delta[target_mask])),
            "delta_rms_s": float(np.sqrt(np.mean(delta ** 2))),
            "supported_delta_rms_s": float(np.sqrt(np.mean(delta[target_mask] ** 2))),
            "delta_max_abs_s": float(np.max(np.abs(delta))),
            "delta_sha256": sha256(delta_path), "delta_values_sha256": hashlib.sha256(np.ascontiguousarray(delta.astype("<f8")).tobytes()).hexdigest(),
            "id_order_sha256": hashlib.sha256(np.ascontiguousarray(ordered_ids.astype("<i8")).tobytes()).hexdigest(),
            "delta_bytes": delta_path.stat().st_size,
            "prediction_rows": int(len(ids)), "submitting_rows": int(len(ordered_ids)),
            "training_rows": int(len(train)), "training_months": list(MONTHS),
            "feature_count": int(x_train.shape[1]), "fit_info": record["info"],
            "kernel_code_sha256": sha256(Path(__file__).resolve()),
            "fit_source_sha256": sha256(fit_path), "proxy_source_sha256": sha256(Path(PX.__file__).resolve()),
            "input_inventory_sha256": sha256(inventory_path), "proxy_inventory_sha256": sha256(exact_file("source_inventory.json")),
            "ranking_weight_file_sha256": sha256(weight_path),
            "point_files_verified": int(proxy_audit["point_files_verified"]),
            "point_bytes_verified": int(proxy_audit["point_bytes_verified"]),
            "elapsed_seconds": time.monotonic() - started,
        }
        output_manifest["outputs"] = {
            delta_path.name: {"bytes": delta_path.stat().st_size, "sha256": sha256(delta_path)}
        }
        (OUTPUT / "manifest.json").write_text(json.dumps(output_manifest, indent=2, default=float) + "\n")
        print(json.dumps(output_manifest, indent=2, default=float), flush=True)
        del train_raw, train_encoded, target_encoded, base_x_train, base_x_valid, x_train, x_valid
        gc.collect()


    if __name__ == "__main__":
        os.environ.setdefault("PRC_KERNEL_SLUG", kernel_slug())
        os.environ.setdefault("PRC_KERNEL_VERSION", KERNEL_VERSION)
        os.environ.setdefault("PRC_EXP130_DATASET_VERSION", "1")
        os.environ.setdefault("PRC_EXP136_DATASET_VERSION", "4")
        os.environ.setdefault("PRC_EXP137_DATASET_VERSION", "3")
        main()


if __name__ == "__main__":
    run_kernel()
