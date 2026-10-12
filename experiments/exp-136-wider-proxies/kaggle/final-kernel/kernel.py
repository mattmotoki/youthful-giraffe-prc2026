#!/usr/bin/env python3
"""Run the frozen exp-136 final refit after a frozen gain gate pass."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user


def run_kernel():
    INPUT = Path("/kaggle/input")
    if not INPUT.exists():
        raise RuntimeError(f"Kaggle input root {INPUT} does not exist")
    fit_matches = sorted(path for path in INPUT.glob("**/fit_exp136.py") if path.is_file())
    weight_matches = sorted(path for path in INPUT.glob("**/final_weights.json") if path.is_file())
    if len(fit_matches) != 1 or len(weight_matches) != 1:
        raise RuntimeError(
            f"expected one fit runner and one final weight file; "
            f"fits={[str(path) for path in fit_matches]}, weights={[str(path) for path in weight_matches]}"
        )
    fit_path = fit_matches[0]
    weight_path = weight_matches[0]
    sys.path.insert(0, str(fit_path.parent))
    import fit_exp136 as FIT  # noqa: E402

    OUTPUT = FIT.OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kernel_path = Path(__file__).resolve()
    os.environ["PRC_KERNEL_SLUG"] = f"{kaggle_user()}/prc-exp136-final-refit"
    os.environ["PRC_KERNEL_VERSION"] = "1"
    os.environ["PRC_DATASET_VERSION"] = "4"
    os.environ["PRC_FINAL_WEIGHTS_VERSION"] = "1"
    os.environ["PRC_KERNEL_CODE_SHA256"] = hashlib.sha256(kernel_path.read_bytes()).hexdigest()


    def main() -> None:
        started = time.monotonic()
        weights_source = json.loads(weight_path.read_text())
        inventory_path = FIT.input_file("source_inventory.json")
        inventory = json.loads(inventory_path.read_text())
        if (inventory.get("ranking_target_read") is not False
                or inventory.get("departure_target_column_read") is not False
                or inventory.get("taxitime_read_for") != "ARR rows only"):
            raise RuntimeError("proxy input inventory does not prove label-safe inputs")
        if weights_source.get("state") != "gate_pass_final_refit_authorized":
            raise RuntimeError("final refit is not authorized by the arm gate")
        if (weights_source.get("gate", {}).get("verdict") != "PASS"
                or weights_source.get("gate", {}).get("selected_arm")
                != weights_source.get("selected_arm")):
            raise RuntimeError("final weights do not carry a matching D107 pass")
        if weights_source.get("spec_sha256") != inventory.get("spec_sha256"):
            raise RuntimeError("final weights were made for a different frozen SPEC")
        if weights_source.get("feature_sha256") != FIT.FEATURE_SHA256:
            raise RuntimeError("final weights were made for a different feature design")
        selected_arm = weights_source.get("selected_arm")
        if selected_arm not in ("t2", "wide"):
            raise RuntimeError(f"invalid gated arm for final refit: {selected_arm}")
        rank_weights = weights_source.get("rank_weights", {})
        if not {"control", selected_arm}.issubset(rank_weights):
            raise RuntimeError("final weights lack the control or selected rank blend weight")
        if any(not FIT.np.isfinite(float(rank_weights[arm]))
               or not 0.0 <= float(rank_weights[arm]) <= 1.0
               for arm in ("control", selected_arm)):
            raise RuntimeError("final rank weights are not finite values in [0, 1]")
        weights = {
            "rank": {
                arm: {"weight": float(rank_weights[arm])}
                for arm in ("control", selected_arm)
            }
        }

        raw_months, proxy_audit = FIT.PX.build_all(FIT.MONTHS, FIT.RANK_MONTHS)
        base, base_code_path = FIT.base_fit_module()
        base_manifest_path = FIT.input_file("input-manifest.json")
        base_manifest = json.loads(base_manifest_path.read_text())
        if base_manifest.get("ranking_target_read") is not False:
            raise RuntimeError("exp-130 input manifest does not prove ranking labels were withheld")
        preflight = json.loads(FIT.input_file("preflight.json").read_text())
        if not preflight.get("v22_baseline_passed"):
            raise RuntimeError("exp-130 saved v22 fold baseline did not pass preflight")

        D18, F, M, D = base.setup_modules()
        month_frames = {month: base.month_frame(month) for month in FIT.MONTHS}
        month_frames.update({month: base.month_frame(month) for month in FIT.RANK_MONTHS})
        ranks = FIT.rank_maps(base)
        final_result = FIT.final_refit(
            base, D18, F, M, D, month_frames, raw_months, ranks, weights, selected_arm)

        result = {
            "experiment": "exp-136-wider-proxies",
            "state": "final_refit_complete",
            "source_ledger_rows": ["861"],
            "gate_verdict": "PASS",
            "selected_arm": selected_arm,
            "spec_sha256": inventory.get("spec_sha256"),
            "feature_sha256": FIT.FEATURE_SHA256,
            "rank_weights": weights["rank"],
            "final_refit": final_result,
            "proxy_build_audit": proxy_audit,
            "rows_without_proxy_source": list(FIT.MISSING_PROXY_SOURCE_ROWS.values()),
            "ranking_target_read": False,
            "final_weights_sha256": FIT.file_hash(weight_path),
            "elapsed_seconds": time.monotonic() - started,
        }
        result_path = OUTPUT / "final_results.json"
        result_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
        outputs = {
            str(path.relative_to(OUTPUT)): FIT.file_hash(path)
            for path in sorted(OUTPUT.rglob("*")) if path.is_file()
        }
        manifest = {
            "experiment": "exp-136-wider-proxies",
            "state": "final_refit_complete",
            "kernel_slug": os.environ["PRC_KERNEL_SLUG"],
            "kernel_version": os.environ["PRC_KERNEL_VERSION"],
            "dataset_sources": {
                "exp130": {
                    "id": f"{kaggle_user()}/prc-exp130-clean-input", "version": "1",
                    "input_manifest_sha256": FIT.file_hash(base_manifest_path),
                },
                "exp136": {
                    "id": f"{kaggle_user()}/prc-exp136-input",
                    "version": os.environ["PRC_DATASET_VERSION"],
                    "source_inventory_sha256": FIT.file_hash(inventory_path),
                },
                "weights": {
                    "id": f"{kaggle_user()}/prc-exp136-final-weights",
                    "version": os.environ["PRC_FINAL_WEIGHTS_VERSION"],
                    "file_sha256": FIT.file_hash(weight_path),
                },
            },
            "spec_sha256": inventory.get("spec_sha256"),
            "feature_sha256": FIT.FEATURE_SHA256,
            "code_sha256": {
                "kernel.py": os.environ["PRC_KERNEL_CODE_SHA256"],
                "fit_exp136.py": FIT.file_hash(FIT.input_file("fit_exp136.py")),
                "proxy_features.py": FIT.file_hash(FIT.input_file("proxy_features.py")),
                "exp130_fit.py": FIT.file_hash(base_code_path),
            },
            "point_source_hashes": inventory.get("source_sha256", {}),
            "point_manifest_sha256": proxy_audit["source_hashes"]["points_expected.json"],
            "verified_point_files": proxy_audit["point_files_verified"],
            "verified_point_bytes": proxy_audit["point_bytes_verified"],
            "seeds": {"lightgbm": 7, "bagging": 8, "feature_fraction": 9},
            "output_sha256": outputs,
            "elapsed_seconds": result["elapsed_seconds"],
        }
        (OUTPUT / "final_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=float) + "\n")
        print(json.dumps(result, indent=2, default=float), flush=True)


    if __name__ == "__main__":
        main()


if __name__ == "__main__":
    run_kernel()
