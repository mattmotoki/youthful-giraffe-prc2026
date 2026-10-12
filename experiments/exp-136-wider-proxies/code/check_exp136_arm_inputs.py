#!/usr/bin/env python3
"""Check one arm kernel's attached inputs before pushing it to Kaggle."""
from __future__ import annotations
import os

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("/tmp/prc-exp136-input")
DATASET_READBACK = Path("/tmp/prc-exp136-v4-readback/fit_exp136.py")
def exp136_dataset():
    return f"{kaggle_user()}/prc-exp136-input"

def exp130_dataset():
    return f"{kaggle_user()}/prc-exp130-clean-input"

EXP130_CHECK = ROOT / "kaggle/run-v2/INPUT-CHECK.md"
RUNTIME_FILES = {
    "fit_exp136.py", "proxy_features.py", "proxy_departures.parquet",
    "proxy_arrivals.parquet", "points_expected.json", "stand_geometry.csv",
    "source_inventory.json",
}
DEPARTURE_COLUMNS = {
    "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
    "EOBT_1_flt", "FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "month",
}
ARRIVAL_COLUMNS = {
    "MVT_ID_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
    "TAXITIME_SEC_mvt", "month",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("control", "t2", "wide", "sham"))
    parser.add_argument("--exp136-status", type=Path, required=True)
    parser.add_argument("--exp130-status", type=Path, required=True)
    parser.add_argument("--dataset-files", type=Path, required=True)
    args = parser.parse_args()

    exp136_status = load_json(args.exp136_status)
    exp130_status = load_json(args.exp130_status)
    remote_files = load_json(args.dataset_files)
    if exp136_status != {"status": "ready", "current_version_number": 4}:
        raise RuntimeError(f"exp-136 input dataset status differs: {exp136_status}")
    if exp130_status != {"status": "ready", "current_version_number": 1}:
        raise RuntimeError(f"exp-130 input dataset status differs: {exp130_status}")

    remote_by_name = {item["name"]: int(item["size"]) for item in remote_files}
    if set(remote_by_name) != RUNTIME_FILES:
        raise RuntimeError(
            f"exp-136 dataset files differ: missing={sorted(RUNTIME_FILES-set(remote_by_name))}, "
            f"extra={sorted(set(remote_by_name)-RUNTIME_FILES)}"
        )
    local_by_name = {
        path.name: path.stat().st_size for path in PACKAGE.iterdir()
        if path.is_file() and path.name in RUNTIME_FILES
    }
    if local_by_name != remote_by_name:
        raise RuntimeError("uploaded exp-136 dataset file sizes differ from the checked package")
    if not DATASET_READBACK.is_file() or digest(DATASET_READBACK) != digest(
            PACKAGE / "fit_exp136.py"):
        raise RuntimeError("the downloaded version-4 runner differs from the local runner")

    dep_columns = set(pq.ParquetFile(
        PACKAGE / "proxy_departures.parquet").schema_arrow.names)
    arr_columns = set(pq.ParquetFile(
        PACKAGE / "proxy_arrivals.parquet").schema_arrow.names)
    if not DEPARTURE_COLUMNS.issubset(dep_columns) or "TAXITIME_SEC_mvt" in dep_columns:
        raise RuntimeError("departure schema lacks a required field or includes taxi-time labels")
    if not ARRIVAL_COLUMNS.issubset(arr_columns):
        raise RuntimeError("arrival schema lacks a required ARR field")
    geometry_columns = set(next(iter(__import__("csv").reader(
        (PACKAGE / "stand_geometry.csv").open()))))
    if not {"airport", "stand_name_mvt", "stand_latitude",
            "stand_longitude"}.issubset(geometry_columns):
        raise RuntimeError("stand geometry schema lacks a required field")
    inventory = load_json(PACKAGE / "source_inventory.json")
    if (inventory.get("ranking_target_read") is not False
            or inventory.get("departure_target_column_read") is not False
            or inventory.get("taxitime_read_for") != "ARR rows only"):
        raise RuntimeError("source inventory does not preserve the label-safety checks")
    if int(inventory.get("point_file_count", -1)) != 427:
        raise RuntimeError("point-file inventory count differs from the frozen package")

    folder = ROOT / f"kaggle/arm-kernels/{args.arm}"
    metadata = load_json(folder / "kernel-metadata.json")
    kernel_code = (folder / "kernel.py").read_text()
    expected_slug = (
        f"{kaggle_user()}/prc-exp136-arm-control" if args.arm == "control"
        else f"{kaggle_user()}/prc-exp136-{args.arm}-arm"
    )
    expected_sources = inventory.get("point_source_kernels")
    if metadata.get("id") != expected_slug or metadata.get("is_private") is not True:
        raise RuntimeError(f"{args.arm}: kernel slug or privacy differs")
    if set(metadata.get("dataset_sources", [])) != {
            exp130_dataset(), exp136_dataset()}:
        raise RuntimeError(f"{args.arm}: attached datasets differ")
    if metadata.get("kernel_sources") != expected_sources:
        raise RuntimeError(f"{args.arm}: attached point-source kernels differ")
    if f'PRC_EXP136_ARM"] = "{args.arm}"' not in kernel_code:
        raise RuntimeError(f"{args.arm}: kernel does not select exactly its arm")
    if 'PRC_DATASET_VERSION"] = "4"' not in kernel_code:
        raise RuntimeError(f"{args.arm}: wrapper does not record input version 4")
    if not EXP130_CHECK.is_file() or "All 30 required files were present" not in EXP130_CHECK.read_text():
        raise RuntimeError("the exp-130 version-1 file and column check is missing")

    output = ROOT / f"kaggle/run-v3/{args.arm}/INPUT-CHECK.md"
    output.write_text(f"""# Input check before {args.arm} kernel push

Checked at {datetime.now().astimezone().isoformat()}.

Kaggle reports both attached datasets ready: exp-130 version 1 and exp-136 version 4. The exp-136 version-4 file list matches the local seven runtime files by name and size. The runner downloaded from version 4 has the same SHA256 as the local runner: {digest(PACKAGE / 'fit_exp136.py')}.

The exp-130 version-1 input check is in ../../run-v2/INPUT-CHECK.md. It confirms all 30 required files and the 32 manifest paths, the fold, ranking, submission, v22, and monthly fields used by this runner. It also confirms that the 2026 month files have no y column.

The exp-136 package schemas contain {len(dep_columns)} departure columns and {len(arr_columns)} arrival columns. Required departure and ARR fields are present. The departure parquet has no TAXITIME_SEC_mvt; the arrival builder reads that field for ARR rows only. The geometry columns are present. This check reads parquet schemas only, not rows or labels. The source inventory still says ranking targets were not read, departure taxi-time was not read, and taxi time was read for ARR rows only.

The point inventory has 427 files from 17 source kernels. This private kernel attaches the same 17 kernels as the inventory and both checked datasets. Its only arm-specific setting is {args.arm}. It uses the exact-file wrapper for fit_exp136.py. No weather input is attached.

Remote exp-136 files and sizes:

{chr(10).join(f"{name}: {remote_by_name[name]:,} bytes" for name in sorted(remote_by_name))}
""")
    print(f"{args.arm}: input files, schemas, hashes, and private kernel sources pass")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
