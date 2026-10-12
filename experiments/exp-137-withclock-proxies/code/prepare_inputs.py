#!/usr/bin/env python3
"""Build exp-137's small private base-vector and runtime-code dataset."""
from __future__ import annotations

import hashlib
import os
import importlib.util
import io
import json
import shutil
import subprocess
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

MAIN = Path(os.environ.get("PRC_ROOT", Path(__file__).resolve().parents[3])).resolve()
REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments/exp-137-withclock-proxies"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PRC_EXP137_OUT", str(MAIN / "artifacts/rebuild/exp137"))).resolve()
ID = "MVT_ID_mvt"
EXP127_COMMIT = "f03ee873"
EXP127_NPZ = "experiments/exp-127-clock-precision/push/fold_update_vectors-r1.npz"
EXP127_JSON = "release/V23-FOLD-BASE.json"
EXP130_INPUT = Path(os.environ.get("PRC_EXP130_OUT", MAIN / "artifacts/rebuild/exp130"))
V24_PATH = MAIN / "submissions/uploaded/youthful-giraffe_v24.parquet"
SUBMITTING_PATH = MAIN / "data/raw/submitting.parquet"
FOLDS = {
    "july_loo": ("2025-07", "rmse_s_lirf_held"),
    "december": ("2025-12", "rmse_s_lirf_held"),
    "january": ("2025-01", "rmse_s_lirf_held"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(revision: str, path: str) -> bytes:
    return (MAIN / path).read_bytes()


def import_exp130_prepare():
    path = REPO / "experiments/exp-130-ranking-exposure/code/prepare.py"
    spec = importlib.util.spec_from_file_location("exp130_prepare_for_exp137", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen exp-130 preparation source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def build_fold_base(prep, unit2_bytes: bytes, base_reproduction: dict) -> pd.DataFrame:
    from io import BytesIO

    folds = prep.BS.load(with_v12=True)
    v22_delta = np.load(prep.V22_VECTOR)
    unit2 = np.load(BytesIO(unit2_bytes))
    records = []
    for fold, (month, rmse_field) in FOLDS.items():
        frame = folds[fold].copy().reset_index(drop=True)
        ids = pd.to_numeric(frame[ID], errors="raise").astype("int64")
        if ids.isna().any() or ids.duplicated().any():
            raise RuntimeError(f"{fold}: exp-047 fold IDs are missing or repeated")

        clock_path = prep.BLOCK_DIR / f"{month}.parquet"
        block = pd.read_parquet(clock_path, columns=[ID, "obs_has_clock"])
        block[ID] = pd.to_numeric(block[ID], errors="raise").astype("int64")
        movement = prep.D.load_month(month, columns=[ID, "PHASE_mvt", "AIRCRAFT_TYPE_mvt"])
        movement = movement.loc[movement["PHASE_mvt"].eq("DEP"),
                                [ID, "AIRCRAFT_TYPE_mvt"]].copy()
        movement[ID] = pd.to_numeric(movement[ID], errors="raise").astype("int64")
        if block[ID].duplicated().any() or movement[ID].duplicated().any():
            raise RuntimeError(f"{fold}: clock or aircraft inputs have duplicate IDs")
        clock_map = block.set_index(ID)["obs_has_clock"]
        aircraft_map = movement.set_index(ID)["AIRCRAFT_TYPE_mvt"]
        clock = ids.map(clock_map)
        aircraft = ids.map(aircraft_map)
        if clock.isna().any() or aircraft.isna().all():
            raise RuntimeError(f"{fold}: class inputs miss fold IDs")
        with_clock_known_ac = clock.astype(bool).to_numpy() & aircraft.notna().to_numpy()

        d46 = np.asarray(v22_delta[fold], dtype="float64")
        if len(d46) != len(frame) or not np.isfinite(d46).all():
            raise RuntimeError(f"{fold}: v22 fold delta does not align with exp-047 rows")
        p12 = frame["y_true"].to_numpy("float64") - frame["r12"].to_numpy("float64")
        d46_applied = np.where(with_clock_known_ac, d46, 0.0)
        p22 = p12 + d46_applied

        id_key, delta_key = f"{fold}_{ID}", f"{fold}_delta"
        unit_ids = np.asarray(unit2[id_key], dtype="int64")
        unit_delta = np.asarray(unit2[delta_key], dtype="float64")
        if len(unit_ids) != len(unit_delta) or len(np.unique(unit_ids)) != len(unit_ids):
            raise RuntimeError(f"{fold}: exp-127 OOF IDs or deltas are invalid")
        outside = np.setdiff1d(unit_ids, ids.to_numpy(), assume_unique=False)
        if len(outside):
            raise RuntimeError(f"{fold}: exp-127 OOF vector has {len(outside)} unknown IDs")
        delta_map = pd.Series(unit_delta, index=unit_ids)
        delta127 = ids.map(delta_map).fillna(0.0).to_numpy("float64")
        airport = frame["airport"].astype(str).to_numpy()
        delta127[airport == "LIRF"] = 0.0
        p23 = p22 + delta127

        expected = float(base_reproduction["folds"][fold][rmse_field])
        got = float(np.sqrt(np.mean((frame["y_true"].to_numpy("float64") - p23) ** 2)))
        if abs(got - expected) > 1e-6:
            raise RuntimeError(f"{fold}: v23 RMSE {got:.9f} does not match source {expected:.9f}")
        records.append(pd.DataFrame({
            ID: ids.to_numpy(), "fold": fold, "fold_row": np.arange(len(ids), dtype="int64"),
            "airport": airport, "obs_has_clock": clock.astype(bool).to_numpy(),
            "known_aircraft": aircraft.notna().to_numpy(), "p12": p12,
            "d46_applied": d46_applied, "p22": p22, "delta127": delta127, "p23": p23,
        }))
    out = pd.concat(records, ignore_index=True)
    if out.duplicated(["fold", ID]).any() or not np.isfinite(out[["p22", "delta127", "p23"]]).all().all():
        raise RuntimeError("fold base has repeated IDs or non-finite predictions")
    if not np.allclose(out.p22 + out.delta127, out.p23, rtol=0, atol=1e-10):
        raise RuntimeError("the v23 fold base does not equal v22 plus exp-127")
    return out


def build_ranking_base() -> pd.DataFrame:
    submitting = pd.read_parquet(SUBMITTING_PATH, columns=[ID])
    v24 = pd.read_parquet(V24_PATH, columns=[ID, "TAXITIME_SEC_mvt"])
    submitting[ID] = pd.to_numeric(submitting[ID], errors="raise").astype("int64")
    v24[ID] = pd.to_numeric(v24[ID], errors="raise").astype("int64")
    if submitting[ID].duplicated().any() or v24[ID].duplicated().any():
        raise RuntimeError("submitting order or v24 base has duplicate IDs")
    if not np.array_equal(submitting[ID].to_numpy(), v24[ID].to_numpy()):
        raise RuntimeError("the uploaded v24 file does not match submitting.parquet order")
    prediction = pd.to_numeric(v24["TAXITIME_SEC_mvt"], errors="raise").to_numpy("float64")
    if not np.isfinite(prediction).all():
        raise RuntimeError("v24 has a non-finite ranking prediction")
    return pd.DataFrame({"row_order": np.arange(len(v24), dtype="int64"),
                         ID: submitting[ID].to_numpy(), "p24": prediction})


def main() -> None:
    if not (REPO / "experiments/exp-137-withclock-proxies/SPEC.md").is_file():
        raise SystemExit("missing frozen exp-137 SPEC")
    if not EXP130_INPUT.is_dir() or not (EXP130_INPUT / "input-manifest.json").is_file():
        raise SystemExit("missing prepared exp-130 input package; set PRC_EXP130_OUT")
    if not V24_PATH.is_file() or not SUBMITTING_PATH.is_file():
        raise SystemExit("missing saved v24 or submitting order")

    base_reproduction_bytes = git_bytes(EXP127_COMMIT, EXP127_JSON)
    unit2_bytes = git_bytes(EXP127_COMMIT, EXP127_NPZ)
    base_reproduction = json.loads(base_reproduction_bytes)
    if base_reproduction.get("experiment") != "exp-127-clock-precision":
        raise RuntimeError("unexpected exp-127 base reproduction file")

    prep, prep_path = import_exp130_prepare()
    fold_base = build_fold_base(prep, unit2_bytes, base_reproduction)
    rank_base = build_ranking_base()

    code_out = OUT / "code"
    code_out.mkdir(parents=True, exist_ok=True)
    for name in ("fit_exp137_arm.py", "fit_exp137_final.py", "proxy_features_exp137.py"):
        source = EXP / "code" / name
        if not source.is_file():
            raise SystemExit(f"missing exp-137 runtime code: {source}")
        shutil.copy2(source, code_out / name)
    (OUT / "fold_base_v23.parquet").parent.mkdir(parents=True, exist_ok=True)
    fold_base.to_parquet(OUT / "fold_base_v23.parquet", index=False, compression="zstd")
    rank_base.to_parquet(OUT / "ranking_base_v24.parquet", index=False, compression="zstd")
    (OUT / "base_reproduction-r1.json").write_bytes(base_reproduction_bytes)
    (OUT / "fold_update_vectors-r1.npz").write_bytes(unit2_bytes)

    source_hashes = {
        "spec": sha256(EXP / "SPEC.md"),
        "exp127_fold_update_vectors_r1": hashlib.sha256(unit2_bytes).hexdigest(),
        "exp127_base_reproduction_r1": hashlib.sha256(base_reproduction_bytes).hexdigest(),
        "exp130_input_manifest": sha256(EXP130_INPUT / "input-manifest.json"),
        "exp130_prepare_source": sha256(prep_path),
        "v22_fold_vector": sha256(prep.V22_VECTOR),
        "v24_submission": sha256(V24_PATH),
        "submitting_order": sha256(SUBMITTING_PATH),
    }
    records = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {
                "dataset-metadata.json", "exp137_source_inventory.json"}:
            records[path.relative_to(OUT).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
    inventory = {
        "experiment": "exp-137",
        "spec_commit": os.environ.get("PRC_SOURCE_REV", "export:" + sha256(EXP / "SPEC.md")),
        "spec_sha256": source_hashes["spec"],
        "exp127_commit": EXP127_COMMIT,
        "ranking_target_read": False,
        "weather_used_or_downloaded": False,
        "departure_target_column_read": False,
        "taxitime_read_for": "ARR rows only; v24 is a saved prediction file",
        "base_fold_rmse": {
            fold: float(base_reproduction["folds"][fold][field])
            for fold, (_, field) in FOLDS.items()
        },
        "base_fold_rows": {
            fold: int((fold_base.fold == fold).sum()) for fold in FOLDS
        },
        "ranking_rows": int(len(rank_base)),
        "source_sha256": source_hashes,
        "files": records,
    }
    (OUT / "exp137_source_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    metadata = {
        "title": "PRC exp-137 fold base and runtime code",
        "id": f"{kaggle_user()}/prc-exp137-input",
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "experiment": "exp-137",
        "spec_sha256": inventory["spec_sha256"],
        "source_sha256": source_hashes,
        "files": records,
        "base_fold_rmse": inventory["base_fold_rmse"],
        "base_fold_rows": inventory["base_fold_rows"],
        "ranking_rows": inventory["ranking_rows"],
        "inventory_sha256": sha256(OUT / "exp137_source_inventory.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
