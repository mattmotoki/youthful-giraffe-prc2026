#!/usr/bin/env python3
"""Prepare private, label-safe exp-130 fit inputs from local cached features."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MAIN = Path(os.environ.get("PRC_ROOT", Path(__file__).resolve().parents[3])).resolve()
EXP = Path(__file__).resolve().parents[1]
CODE = EXP / "code"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PRC_EXP130_OUT", str(MAIN / "artifacts/rebuild/exp130"))).resolve()
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
CACHE = MAIN / "artifacts/exp-018-fresh/cache"
CLASS_DIR = MAIN / "artifacts/exp-046/class"
BLOCK_DIR = MAIN / "artifacts/exp-047/block"
SUBMISSIONS = MAIN / "submissions/uploaded"
V22_VECTOR = MAIN / "experiments/exp-046-surfacefit/work/d-seasonal-bag10-255-free.npz"
CENSUS_REF = MAIN / "experiments/exp-047-queue/analysis/census-v22.json"
FOLDS = {
    "july_loo": "2025-07",
    "december": "2025-12",
    "january": "2025-01",
}
RANK_MONTHS = ("2026-01", "2026-07")
SIGNAL_COLS = [
    ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "LOBT_flt",
    "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "FLIGHT_mvt",
]

os.environ.pop("PRC_WINDOW", None)
os.environ["PRC_DATA_DIR"] = str(MAIN / "data/raw")
sys.path.insert(0, str(MAIN))
from prc import data as D  # noqa: E402
from prc.lirf_features import VERSION as LIRF_VERSION, lirf_signals  # noqa: E402

sys.path.insert(0, str(MAIN / "experiments/exp-018-fresh/code"))
import dataset as D18  # noqa: E402

sys.path.insert(0, str(MAIN / "experiments/exp-047-queue/code"))
import base as BS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_id(values: pd.Series) -> pd.Series:
    number = pd.to_numeric(values, errors="raise")
    if number.isna().any() or (number != np.floor(number)).any():
        raise SystemExit(f"{ID} contains a missing or non-integer value")
    return number.astype("int64")


def cached_month(month: str, *, include_target: bool = True) -> pd.DataFrame:
    path = CACHE / f"{month}.parquet"
    if not path.is_file():
        raise SystemExit(f"missing exp-018 cache: {path}")
    columns = None
    if not include_target:
        columns = [column for column in pq.read_schema(path).names if column != "y"]
    frame = pd.read_parquet(path, columns=columns)
    if ID not in frame or "month" not in frame:
        raise SystemExit(f"{path} lacks {ID} or month")
    frame[ID] = to_id(frame[ID])
    if frame[ID].duplicated().any():
        raise SystemExit(f"duplicate {ID} in {path}")
    if not frame["month"].eq(month).all():
        raise SystemExit(f"{path} includes a different month")
    return frame


def raw_month(month: str, ranking: pd.DataFrame | None = None) -> pd.DataFrame:
    if month.startswith("2025-"):
        return D.load_month(month, columns=SIGNAL_COLS)
    if ranking is None:
        raise ValueError("ranking frame is required for 2026 months")
    return ranking.loc[ranking["month"].eq(month), SIGNAL_COLS].copy()


def with_lirf(base: pd.DataFrame, raw: pd.DataFrame, month: str) -> pd.DataFrame:
    raw = raw.copy()
    raw[ID] = to_id(raw[ID])
    departure = raw["PHASE_mvt"].eq("DEP")
    if "BLOCK_TIME_UTC_mvt" in raw:
        raw.loc[departure, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    if TARGET in raw:
        raw.loc[departure, TARGET] = np.nan
    dep = raw.loc[raw["PHASE_mvt"].eq("DEP"), [ID, "ADEP_mvt", "MVT_TIME_UTC_mvt"]].copy()
    if dep[ID].duplicated().any():
        raise SystemExit(f"{month}: duplicate departure IDs in raw input")
    dep["airport"] = dep["ADEP_mvt"].astype("string").fillna("").astype(str)
    dep["utc_day"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
    if dep[["airport", "utc_day"]].isna().any().any() or (dep["airport"] == "").any():
        raise SystemExit(f"{month}: missing airport or UTC day")

    signals = lirf_signals(raw)
    if ID not in signals or signals[ID].duplicated().any():
        raise SystemExit(f"{month}: invalid LIRF signal IDs")
    signals = signals.copy()
    signals[ID] = to_id(signals[ID])
    signal_names = [column for column in signals.columns if column != ID]
    signals = signals.rename(columns={column: f"lirf_{column}" for column in signal_names})

    base = base.copy()
    base[ID] = to_id(base[ID])
    if set(base[ID]) != set(dep[ID]) or set(signals[ID]) != set(dep[ID]):
        raise SystemExit(f"{month}: feature, raw and LIRF departure IDs do not match")
    merged = base.merge(dep[[ID, "airport", "utc_day"]], on=ID, how="left",
                        validate="one_to_one")
    merged = merged.merge(signals, on=ID, how="left", validate="one_to_one")
    if len(merged) != len(base) or merged.filter(regex=r"^lirf_").isna().all(axis=None):
        raise SystemExit(f"{month}: LIRF join is incomplete")
    merged["month"] = month
    return merged


def rank_metadata(ranking: pd.DataFrame, submitting: pd.DataFrame,
                  source_paths: list[Path]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    # ranking.parquet contains arrival context as well as submitted departures.
    # Compare the submission template with departures only; keep the full
    # ranking frame for month-local LIRF context features below.
    rank = ranking.loc[ranking["PHASE_mvt"].eq("DEP"),
                       [ID, "month", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                        "AIRCRAFT_TYPE_mvt"]].copy()
    rank[ID] = to_id(rank[ID])
    submitted_ids = to_id(submitting[ID])
    if submitted_ids.duplicated().any() or rank[ID].duplicated().any():
        raise SystemExit("ranking or submitting IDs are not unique")
    if len(rank) != len(submitted_ids) or set(rank[ID]) != set(submitted_ids):
        raise SystemExit("ranking IDs and submitting IDs do not match exactly")
    if not rank["PHASE_mvt"].eq("DEP").all():
        raise SystemExit("submitted IDs include a non-departure row")

    pieces: dict[str, pd.DataFrame] = {}
    for month in RANK_MONTHS:
        part = rank.loc[rank["month"].eq(month)].copy()
        class_path = CLASS_DIR / f"{month}-v25.parquet"
        source_paths.append(class_path)
        classes = pd.read_parquet(class_path,
                                  columns=[ID, "obs_has_clock", "noac", "occupied_by_exp_046"])
        classes[ID] = to_id(classes[ID])
        if classes[ID].duplicated().any():
            raise SystemExit(f"{month}: duplicate IDs in the saved class vector")
        part = part.merge(classes, on=ID, how="left", validate="one_to_one", indicator=True)
        if not part["_merge"].eq("both").all() or part["obs_has_clock"].isna().any():
            raise SystemExit(f"{month}: saved class vector misses ranking rows")
        if not np.array_equal(part["noac"].astype(bool).to_numpy(),
                              part["AIRCRAFT_TYPE_mvt"].isna().to_numpy()):
            raise SystemExit(f"{month}: saved noac class differs from ranking aircraft type")
        if not np.array_equal(part["occupied_by_exp_046"].astype(bool).to_numpy(),
                              (part["obs_has_clock"].astype(bool)
                               & ~part["noac"].astype(bool)).to_numpy()):
            raise SystemExit(f"{month}: saved with-clock class is inconsistent")
        part["rank_class"] = np.where(
            part["noac"].astype(bool), "noac",
            np.where(part["obs_has_clock"].astype(bool), "with-clock", "no-clock_ordinary"))
        part["airport"] = part["ADEP_mvt"].astype("string").fillna("").astype(str)
        part["utc_day"] = pd.to_datetime(part["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
        pieces[month] = part[[ID, "month", "airport", "utc_day", "obs_has_clock",
                              "noac", "rank_class"]].copy()

    all_meta = pd.concat(pieces.values(), ignore_index=True)
    return pieces, all_meta


def main() -> None:
    if D.data_dir().resolve() != (MAIN / "data/raw").resolve():
        raise SystemExit("PRC raw data directory resolved outside the main checkout")
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths: list[Path] = []
    file_records: dict[str, str] = {}

    ranking_path = D.data_dir() / "ranking.parquet"
    submitting_path = D.data_dir() / "submitting.parquet"
    v22_path = SUBMISSIONS / "youthful-giraffe_v22.parquet"
    source_paths.extend([ranking_path, submitting_path, v22_path, V22_VECTOR, CENSUS_REF])
    submitting = pd.read_parquet(submitting_path, columns=[ID])
    if ID not in submitting:
        raise SystemExit("submitting template lacks MVT_ID_mvt")

    rank_cols = sorted(set(SIGNAL_COLS))
    ranking = D.load_ranking(columns=rank_cols)
    rank_parts, rank_all = rank_metadata(ranking, submitting, source_paths)
    expected = pd.read_csv(EXP / "census.csv")
    expected = expected.loc[expected["class"].eq("no-clock_ordinary"),
                            ["ranking_month", "airport", "ranking_rows"]]
    actual = (rank_all.loc[rank_all["rank_class"].eq("no-clock_ordinary")]
              .groupby(["month", "airport"], observed=True).size().rename("ranking_rows")
              .reset_index())
    actual = actual.rename(columns={"month": "ranking_month"})
    check = expected.merge(actual, on=["ranking_month", "airport"], how="outer",
                           suffixes=("_census", "_input"), indicator=True)
    if not check["_merge"].eq("both").all() or not np.array_equal(
            check["ranking_rows_census"].to_numpy(), check["ranking_rows_input"].to_numpy()):
        raise SystemExit("rank no-clock ordinary counts differ from the pushed census")
    rank_counts = actual.rename(columns={"ranking_month": "month"})
    rank_counts.to_csv(OUT / "rank_counts.csv", index=False)
    shutil.copy2(EXP / "census.csv", OUT / "census.csv")

    monthly_paths = []
    ranking_lookup = ranking
    for month in (*D.MONTHS, *RANK_MONTHS):
        cached_path = CACHE / f"{month}.parquet"
        source_paths.append(cached_path)
        base = cached_month(month, include_target=month.startswith("2025-"))
        raw_path = D.month_file(month) if month.startswith("2025-") else ranking_path
        source_paths.append(raw_path)
        raw = raw_month(month, ranking_lookup if month.startswith("2026-") else None)
        enriched = with_lirf(base, raw, month)
        path = OUT / f"month_{month}.parquet"
        enriched.to_parquet(path, index=False, compression="zstd")
        monthly_paths.append(path)
        print(f"prepared {month}: {len(enriched):,} rows, {len(enriched.columns)} columns",
              flush=True)
        del base, raw, enriched

    for month, rank_part in rank_parts.items():
        rank_path_out = OUT / f"rank_meta_{month}.parquet"
        rank_part.to_parquet(rank_path_out, index=False, compression="zstd")
        monthly_paths.append(rank_path_out)

    submitting_ids = to_id(submitting[ID]).rename(ID)
    submitting_order = pd.DataFrame({"row_order": np.arange(len(submitting_ids), dtype="int64"),
                                     ID: submitting_ids.to_numpy()})
    order_path = OUT / "submitting_ids.parquet"
    submitting_order.to_parquet(order_path, index=False, compression="zstd")

    v22 = pd.read_parquet(v22_path, columns=[ID, TARGET])
    v22[ID] = to_id(v22[ID])
    if v22[ID].duplicated().any() or not np.array_equal(v22[ID].to_numpy(), submitting_ids.to_numpy()):
        raise SystemExit("v22 submission IDs differ from submitting.parquet order")
    v22 = v22.rename(columns={TARGET: "p22"})
    v22_path_out = OUT / "v22_submission.parquet"
    v22.to_parquet(v22_path_out, index=False, compression="zstd")
    monthly_paths.extend([order_path, v22_path_out, OUT / "rank_counts.csv"])

    folds = BS.load(with_v12=True)
    v22_delta = np.load(V22_VECTOR)
    census_ref = json.loads(CENSUS_REF.read_text())
    preflight_checks = {}
    for fold, month in FOLDS.items():
        frame = folds[fold]
        block_path = BLOCK_DIR / f"{month}.parquet"
        source_paths.append(block_path)
        block = pd.read_parquet(block_path, columns=[ID, "obs_has_clock"])
        block[ID] = to_id(block[ID])
        movement = D.load_month(month, columns=[ID, "PHASE_mvt", "AIRCRAFT_TYPE_mvt"])
        movement = movement.loc[movement["PHASE_mvt"].eq("DEP"), [ID, "AIRCRAFT_TYPE_mvt"]].copy()
        movement[ID] = to_id(movement[ID])
        if block[ID].duplicated().any() or movement[ID].duplicated().any():
            raise SystemExit(f"{fold}: duplicate IDs in v22 preflight inputs")
        ids = to_id(frame[ID]).to_numpy()
        clock = pd.Series(ids).map(block.set_index(ID)["obs_has_clock"])
        aircraft = pd.Series(ids).map(movement.set_index(ID)["AIRCRAFT_TYPE_mvt"])
        if clock.isna().any() or aircraft.isna().all():
            raise SystemExit(f"{fold}: class vectors do not cover the v22 baseline rows")
        with_clock = clock.to_numpy(bool) & aircraft.notna().to_numpy()
        residual = frame["r12"].to_numpy("float64") - np.where(
            with_clock, np.asarray(v22_delta[fold], dtype="float64"), 0.0)
        got = float(np.sqrt(np.mean(residual * residual)))
        expected_rmse = float(census_ref["fold_totals"][fold]["rmse_v22"])
        error = abs(got - expected_rmse)
        if error > 1e-6:
            raise SystemExit(f"{fold}: v22 RMSE {got:.9f} differs from census {expected_rmse:.9f}")
        preflight_checks[fold] = {"rmse_rebuilt": got, "rmse_reference": expected_rmse,
                                  "absolute_error": error, "rows": int(len(frame))}
    (OUT / "preflight.json").write_text(json.dumps({
        "v22_baseline_passed": True,
        "v22_method": "r12 - d-seasonal on ordinary with-clock rows; r12 on the complement",
        "checks": preflight_checks,
    }, indent=2) + "\n")

    for fold, month in FOLDS.items():
        frame = folds[fold]
        block_path = BLOCK_DIR / f"{month}.parquet"
        source_paths.append(block_path)
        block = pd.read_parquet(block_path, columns=[ID, "obs_has_clock"])
        block[ID] = to_id(block[ID])
        if block[ID].duplicated().any():
            raise SystemExit(f"{fold}: duplicate IDs in saved clock vector")
        raw = D.load_month(month, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
        raw = raw.loc[raw["PHASE_mvt"].eq("DEP"), [ID, "ADEP_mvt", "MVT_TIME_UTC_mvt"]].copy()
        raw[ID] = to_id(raw[ID])
        raw["airport"] = raw["ADEP_mvt"].astype("string").fillna("").astype(str)
        raw["utc_day"] = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
        eval_frame = pd.DataFrame({
            ID: to_id(frame[ID]),
            "y_true": frame["y_true"].to_numpy("float64"),
            "r12": frame["r12"].to_numpy("float64"),
            "airport_base": frame["apt"].astype(str).to_numpy(),
        })
        eval_frame = eval_frame.merge(raw[[ID, "airport", "utc_day"]], on=ID,
                                      how="left", validate="one_to_one")
        eval_frame = eval_frame.merge(block, on=ID, how="left", validate="one_to_one")
        if eval_frame[["airport", "utc_day", "obs_has_clock"]].isna().any().any():
            raise SystemExit(f"{fold}: missing raw timing or class metadata")
        if not eval_frame["airport"].eq(eval_frame["airport_base"]).all():
            raise SystemExit(f"{fold}: base and raw airport labels differ")
        feature_valid = cached_month(month)
        if set(feature_valid[ID]) != set(eval_frame[ID]):
            raise SystemExit(f"{fold}: cached features and saved base IDs differ")
        y_cache = feature_valid.set_index(ID)["y"].reindex(eval_frame[ID]).to_numpy("float64")
        if not np.allclose(y_cache, eval_frame["y_true"].to_numpy("float64"), rtol=0, atol=1e-8):
            raise SystemExit(f"{fold}: cached labels and base labels differ")
        eval_frame = eval_frame.drop(columns="airport_base")
        eval_path = OUT / f"eval_{fold}.parquet"
        eval_frame.to_parquet(eval_path, index=False, compression="zstd")
        monthly_paths.append(eval_path)
        print(f"prepared eval {fold}: {len(eval_frame):,} rows, clean exclusions "
              f"{len(D.excluded_ids(fold))}", flush=True)

    source_map: dict[str, str] = {}
    for path in sorted(set(source_paths)):
        if path.is_file():
            source_map[str(path)] = sha256(path)
        else:
            raise SystemExit(f"missing source file for manifest: {path}")
    source_files = {
        "exp018_dataset.py": MAIN / "experiments/exp-018-fresh/code/dataset.py",
        "exp018_features.py": MAIN / "experiments/exp-018-fresh/code/features.py",
        "exp018_model.py": MAIN / "experiments/exp-018-fresh/code/model.py",
        "prc_data.py": MAIN / "prc/data.py",
        "prc_lirf_features.py": MAIN / "prc/lirf_features.py",
        "exp047_base.py": MAIN / "experiments/exp-047-queue/code/base.py",
        "input_rule.md": MAIN / "docs/input-rule.md",
        "exp130_fit.py": CODE / "fit.py",
        "exp130_prepare.py": Path(__file__),
        "exp130_spec.py": EXP / "SPEC.md",
        "exp130_census_csv": EXP / "census.csv",
        "exp130_census_md": EXP / "CENSUS.md",
        "exp130_census_code": CODE / "census.py",
    }
    for label, path in source_files.items():
        if not path.is_file():
            raise SystemExit(f"missing source file {path}")
        source_map[label] = sha256(path)

    packaged_sources = {
        "source/prc/__init__.py": MAIN / "prc/__init__.py",
        "source/prc/data.py": MAIN / "prc/data.py",
        "source/prc/lirf_features.py": MAIN / "prc/lirf_features.py",
        "source/exp018/dataset.py": MAIN / "experiments/exp-018-fresh/code/dataset.py",
        "source/exp018/features.py": MAIN / "experiments/exp-018-fresh/code/features.py",
        "source/exp018/model.py": MAIN / "experiments/exp-018-fresh/code/model.py",
        "fit.py": CODE / "fit.py",
    }
    for relative, source in packaged_sources.items():
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    file_hashes = {}
    total_bytes = 0
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        total_bytes += path.stat().st_size
        file_hashes[str(path.relative_to(OUT))] = sha256(path)
    if total_bytes >= 2 * 1024**3:
        raise SystemExit(f"prepared local output is {total_bytes:,} bytes, above 2 GiB")
    repository = EXP.parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository,
                                         text=True).strip()
        spec_commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--",
             "experiments/exp-130-ranking-exposure/SPEC.md"],
            cwd=repository, text=True).strip()
    except Exception:
        commit = "unknown"
        spec_commit = "unknown"
    manifest = {
        "experiment": "exp-130",
        "spec_commit": spec_commit,
        "worker_source_commit": commit,
        "lirf_features_version": LIRF_VERSION,
        "ranking_target_read": False,
        "weather_used_or_downloaded": False,
        "seed_values": {
            "lightgbm": 7,
            "bagging": 8,
            "feature_fraction": 9,
            "feature_sham": {fold: 13017 + index
                             for fold, index in zip(FOLDS, (1, 2, 3))},
            "weight_placebo": {fold: 13007 + index
                               for fold, index in zip(FOLDS, (1, 2, 3))},
            "bootstrap": 20260925,
        },
        "source_sha256": source_map,
        "input_sha256": file_hashes,
        "input_total_bytes": total_bytes,
        "ranking_no_clock_ordinary_rows": int(
            rank_all["rank_class"].eq("no-clock_ordinary").sum()),
        "ranking_rows_by_month": {m: int(rank_all["month"].eq(m).sum()) for m in RANK_MONTHS},
    }
    (OUT / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared input package: {total_bytes:,} bytes, {len(file_hashes)} files, "
          f"LIRF v{LIRF_VERSION}, no-clock ordinary "
          f"{manifest['ranking_no_clock_ordinary_rows']:,} rows", flush=True)


if __name__ == "__main__":
    main()
