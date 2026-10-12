#!/usr/bin/env python3
"""Recover local arm manifests after Kaggle saved score files but failed packaging."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "kaggle/run-v3"
ARMS = ("control", "proxy", "sham")
SPEC_SHA = "71ca0e05d8cacaaeeac26667e3d8f73449e0772df17596a4e13d92b2255ce689"
DATASET_VERSIONS = {
    "prc-exp130-clean-input": "1",
    "prc-exp136-input": "4",
    "prc-exp137-input": "3",
}
REQUIRED = (
    "main_predictions.parquet",
    "calibration_predictions.parquet",
    "weights.csv",
    "ranking_counts.csv",
    "base_reproduction.csv",
    "metrics.csv",
    "fit_info.csv",
    "feature_importance.csv",
    "proxy_coverage.csv",
    "results.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run_manifest = json.loads((RUN / "manifest.json").read_text())
    recovery = {"experiment": "exp-137-withclock-proxies", "spec_sha256": SPEC_SHA,
                "kernel_outputs_recovered_after_all_arms_terminal": True, "arms": {}}

    for arm in ARMS:
        arm_dir = RUN / arm
        output_dir = arm_dir / "output"
        raw_dir = output_dir / "exp137-output"
        if raw_dir.is_dir():
            for path in sorted(raw_dir.iterdir()):
                target = output_dir / path.name
                if target.exists():
                    raise FileExistsError(f"refuse to overwrite recovered output: {target}")
                shutil.move(str(path), str(target))
            raw_dir.rmdir()

        missing = [name for name in REQUIRED if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{arm}: missing recovered output files: {missing}")
        result = json.loads((output_dir / "results.json").read_text())
        if result.get("experiment") != "exp-137-withclock-proxies" or result.get("arm") != arm:
            raise RuntimeError(f"{arm}: recovered results identify a different experiment or arm")
        if result.get("state") != "arm_complete_no_gate" or result.get("spec_sha256") != SPEC_SHA:
            raise RuntimeError(f"{arm}: recovered result state or SPEC hash does not match")
        if result.get("ranking_target_read") is not False or result.get("weather_used_or_downloaded") is not False:
            raise RuntimeError(f"{arm}: recovered result does not preserve label/weather limits")

        arm_input = run_manifest["arms"][arm]
        if arm_input["status"] != "ERROR" or arm_input["kernel_version"] != 2:
            raise RuntimeError(f"{arm}: expected the recorded v2 post-fit packaging error")
        output_hashes = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name not in {"manifest.json"} and not path.name.endswith(".log")
        }
        manifest = {
            "experiment": "exp-137-withclock-proxies",
            "arm": arm,
            "state": "arm_complete_no_gate",
            "kernel_slug": arm_input["slug"],
            "kernel_version": str(arm_input["kernel_version"]),
            "kernel_code_sha256": arm_input["kernel_code_sha256"],
            "dataset_versions": DATASET_VERSIONS,
            "spec_sha256": SPEC_SHA,
            "ranking_target_read": False,
            "weather_used_or_downloaded": False,
            "output_sha256": output_hashes,
            "recovery": {
                "manifest_reconstructed_locally": True,
                "kernel_terminal_status": "ERROR",
                "error": "KeyError: 'source_sha256' while writing the output manifest",
                "error_log": f"kaggle/run-v3/{arm}/error-logs/{arm}.log",
                "reason": "Kaggle saved all result files before manifest packaging failed; local output files were hash-recorded after all three kernels were terminal.",
                "delta127_reconstruction_allowed": True,
                "delta127_reconstruction": "The saved prediction table has p22 and p23 but omits the redundant delta127 column; combine derives it as p23 - p22 and validates the fold base identity.",
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        recovery["arms"][arm] = {
            "kernel_status": "ERROR",
            "error": "KeyError: 'source_sha256'",
            "result_state": result["state"],
            "result_spec_sha256": result["spec_sha256"],
            "ranking_target_read": result["ranking_target_read"],
            "weather_used_or_downloaded": result["weather_used_or_downloaded"],
            "files": output_hashes,
            "local_manifest_sha256": sha256(output_dir / "manifest.json"),
        }

    (RUN / "artifact-recovery.json").write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": "arm outputs recovered and hash-recorded", "recovery": recovery}, indent=2))


if __name__ == "__main__":
    main()
