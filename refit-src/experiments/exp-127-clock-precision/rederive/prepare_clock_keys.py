#!/usr/bin/env python3
"""Prepare target-free movement keys for the exp-127b point kernel."""

from __future__ import annotations

import calendar
import hashlib
import os
import importlib.util
import json
import re
import sys
import zlib
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
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "rederive/kaggle/keys-dataset"
FOLD_FILE = EXP / "work/fold_frames.parquet"
GEOMETRY = EXP / "stand_geometry.csv"
POINT_MANIFEST = MAIN / "experiments/exp-045-harvest/manifest.json"
FOLD_MONTHS = {
    "2025-07": ("july_loo", "training_2025-07-01_2025-08-01.parquet"),
    "2025-12": ("december", "training_2025-12-01_2026-01-01.parquet"),
    "2025-01": ("january", "training_2025-01-01_2025-02-01.parquet"),
}
RANK_MONTHS = {"2026-01", "2026-07"}
TEMPLATE = MAIN / "data/raw/submitting.parquet"
# exp-186 (2026-10-08, desk request b for finaldata): $PRC_RANKING_MODE=final takes the months and the
# template of the final file (prc/data.py RANKING_SPECS["final"], by SHA-256), adds the two harvest kernels
# of the new months, reads the final stand table `stand_geometry-final.csv` (build_stand_geometry.py in the
# same mode) and writes the sibling dataset folder `keys-dataset-final`. The ranking mode keeps every value.
RANKING_MODE = os.environ.get("PRC_RANKING_MODE") or "ranking"
if RANKING_MODE != "ranking":
    sys.path.insert(0, str(MAIN))
    from prc import data as _D
    RANK_MONTHS = set(_D.rank_months(RANKING_MODE))
    TEMPLATE = _D.submitting_file(RANKING_MODE)
    OUT = OUT.with_name(f"{OUT.name}-{RANKING_MODE}")
    GEOMETRY = GEOMETRY.with_name(f"{GEOMETRY.stem}-{RANKING_MODE}{GEOMETRY.suffix}")
LFPG = "LFPG"
def point_kernel_versions():
    versions = {
    f"{kaggle_user()}/prc-harvest-2025-01-b": {"version": 1, "last_run_utc": "2026-09-22T12:34:33.733Z"},
    f"{kaggle_user()}/prc-harvest-2025-07-b": {"version": 1, "last_run_utc": "2026-09-22T12:34:34.187Z"},
    f"{kaggle_user()}/prc-harvest-2025-12": {"version": 1, "last_run_utc": "2026-09-22T12:34:35.410Z"},
    f"{kaggle_user()}/prc-harvest-2026-01-b": {"version": 1, "last_run_utc": "2026-09-22T12:34:35.854Z"},
    f"{kaggle_user()}/prc-harvest-2026-01-gate": {"version": 1, "last_run_utc": "2026-09-22T12:34:37.235Z"},
    f"{kaggle_user()}/prc-harvest-2026-07-b": {"version": 1, "last_run_utc": "2026-09-22T12:34:37.695Z"},
}
    if RANKING_MODE == "final":
        # the harvest kernels of the two new months (finaldata, version 1); their run time is not known here
        for month in ("2026-02", "2026-06"):
            versions[f"{kaggle_user()}/prc-harvest-{month}"] = {"version": 1, "last_run_utc": None}
    return versions



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def unique_stand_map(frame: pd.DataFrame) -> tuple[dict, dict]:
    exact: dict[tuple[str, str], tuple[float, float, str]] = {}
    for (airport, name), rows in frame.groupby(["airport", "stand_name_mvt"], sort=True):
        values = rows[["stand_latitude", "stand_longitude", "stand_source"]].drop_duplicates()
        if pd.notna(name) and len(values) == 1:
            row = values.iloc[0]
            exact[(str(airport), str(name))] = (
                float(row.stand_latitude), float(row.stand_longitude), str(row.stand_source)
            )
    norm: dict[tuple[str, str], list[tuple[str, tuple[float, float, str]]]] = {}
    for (airport, name), value in exact.items():
        norm.setdefault((airport, normalize(name)), []).append((name, value))
    unique = {key: vals[0] for key, vals in norm.items() if key[1] and len(vals) == 1}
    return exact, unique


def add_stands(frame: pd.DataFrame, exact: dict, normalized: dict) -> pd.DataFrame:
    lat, lon, source, key_name = [], [], [], []
    for airport, stand in zip(frame.airport, frame.STAND_mvt):
        if str(airport) == LFPG or pd.isna(stand):
            lat.append(np.nan); lon.append(np.nan); source.append(None); key_name.append(None)
            continue
        hit = exact.get((str(airport), str(stand)))
        matched_name = str(stand) if hit is not None else None
        if hit is None:
            item = normalized.get((str(airport), normalize(stand)))
            hit = item[1] if item is not None else None
            matched_name = item[0] if item is not None else None
        if hit is None:
            lat.append(np.nan); lon.append(np.nan); source.append(None); key_name.append(None)
        else:
            lat.append(hit[0]); lon.append(hit[1]); source.append(hit[2]); key_name.append(matched_name)
    out = frame.copy()
    out["survey_stand_lat"] = np.asarray(lat, dtype="float64")
    out["survey_stand_lon"] = np.asarray(lon, dtype="float64")
    out["survey_source"] = source
    out["survey_stand_key"] = key_name
    return out


def add_sham_stands(frame: pd.DataFrame, stand_source: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sham_stand_lat"] = np.nan
    out["sham_stand_lon"] = np.nan
    out["sham_source"] = None
    eligible = out.scope.eq("fold") & out.survey_stand_key.notna()
    for (airport, month), indices in out.loc[eligible].groupby(["airport", "month"], sort=True).groups.items():
        group = stand_source.loc[stand_source.airport.eq(airport)].copy()
        names = group.stand_name_mvt.astype(str).to_numpy()
        tuples = group[["stand_latitude", "stand_longitude", "stand_source"]].to_numpy(object)
        seed = zlib.crc32(f"12727:{airport}:{month}".encode("utf-8"))
        permuted = tuples[np.random.default_rng(seed).permutation(len(tuples))]
        values = {
            name: (float(value[0]), float(value[1]), str(value[2]))
            for name, value in zip(names, permuted)
        }
        for idx in indices:
            key = out.at[idx, "survey_stand_key"]
            lat, lon, source = values[key]
            out.at[idx, "sham_stand_lat"] = lat
            out.at[idx, "sham_stand_lon"] = lon
            out.at[idx, "sham_source"] = source
    return out


def load_serve():
    sys.path.insert(0, str(MAIN))
    path = MAIN / "experiments/exp-046-surfacefit/code/serve.py"
    spec = importlib.util.spec_from_file_location("exp127b_saved_serve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import saved serving block from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXP = EXP
    return module


def event_seconds(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    unit = getattr(dt.dtype, "unit", "ns")
    if unit not in unit_scale:
        raise SystemExit(f"unsupported timestamp unit: {unit}")
    seconds = dt.astype("int64", copy=False).to_numpy(dtype="float64") / unit_scale[unit]
    seconds[dt.isna().to_numpy()] = np.nan
    return seconds


def validate_current_time_scale(keys: pd.DataFrame) -> None:
    valid = keys.loc[keys.current_250m_s.notna(), ["month", "current_250m_s"]].copy()
    if valid.empty:
        raise SystemExit("no current 250 m clock times are available")
    valid["current_250m_s"] = pd.to_numeric(valid.current_250m_s, errors="coerce")
    for month, rows in valid.groupby("month", sort=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        lo = start.timestamp() - 86400.0
        hi = end.timestamp() + 86400.0
        if not rows.current_250m_s.between(lo, hi, inclusive="both").all():
            raise SystemExit(f"{month}: current 250 m times are not Unix seconds for that month")


def main() -> None:
    if not FOLD_FILE.is_file() or not GEOMETRY.is_file():
        raise SystemExit("unit-1 prepared frame or surveyed stand geometry is missing")
    stand = pd.read_csv(
        GEOMETRY,
        dtype={"airport": str, "stand_name_mvt": str, "runway_end": str},
    )
    stand["source_order"] = stand.stand_source.map(
        {"xplane_gateway": 0, "openstreetmap": 1}
    ).fillna(9)
    stand = stand.sort_values(["source_order", "stand_source", "runway_end"])
    stand = stand.groupby(["airport", "stand_name_mvt"], as_index=False).first()
    stand_exact, stand_norm = unique_stand_map(stand)
    sham_stands = stand[["airport", "stand_name_mvt", "stand_latitude", "stand_longitude", "stand_source"]].copy()

    prepared = pd.read_parquet(FOLD_FILE)
    if prepared.MVT_ID_mvt.duplicated().any():
        raise SystemExit("unit-1 fold frame has duplicate movement IDs")
    fold_parts = []
    source_hashes = {str(FOLD_FILE): sha256(FOLD_FILE), str(GEOMETRY): sha256(GEOMETRY)}
    source_hashes[str(POINT_MANIFEST)] = sha256(POINT_MANIFEST)
    for month, (fold, training_name) in FOLD_MONTHS.items():
        train_path = MAIN / "data/raw" / training_name
        record_path = MAIN / f"experiments/exp-046-surfacefit/analysis/block-{month}.json"
        record = json.loads(record_path.read_text())
        block_path = MAIN / "experiments/exp-046-surfacefit/work" / record["block_cache"]
        source_hashes[str(train_path)] = sha256(train_path)
        source_hashes[str(record_path)] = sha256(record_path)
        source_hashes[str(block_path)] = sha256(block_path)

        fold_rows = prepared.loc[prepared.month.eq(month),
                                 ["MVT_ID_mvt", "fold", "month", "airport", "day", "STAND_mvt"]].copy()
        train = pd.read_parquet(
            train_path,
            columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"],
        )
        train = train.loc[train.PHASE_mvt.eq("DEP"),
                          ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"]].copy()
        train["MVT_ID_mvt"] = train.MVT_ID_mvt.astype("int64")
        train = train.rename(columns={"ADEP_mvt": "training_airport", "STAND_mvt": "training_stand"})
        block = pd.read_parquet(
            block_path,
            columns=["MVT_ID_mvt", "obs_icao", "obs_wheels_off", "obs_taxi", "obs_has_clock"],
        )
        block["MVT_ID_mvt"] = block.MVT_ID_mvt.astype("int64")
        rows = fold_rows.merge(train, on="MVT_ID_mvt", how="left", validate="one_to_one")
        rows = rows.merge(block, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if rows.training_airport.isna().any():
            raise SystemExit(f"{month}: movement rows did not join to the training departures")
        # The prepared fold frame has the airport and stand copied from training;
        # assert these keys before dropping duplicate columns.
        if not rows.airport.eq(rows.training_airport).all():
            raise SystemExit(f"{month}: training airport disagrees with the saved fold")
        if not rows.STAND_mvt.eq(rows.training_stand).all():
            raise SystemExit(f"{month}: training stand disagrees with the saved fold")
        rows = rows.drop(columns=["training_airport", "training_stand"])
        if not rows.obs_has_clock.fillna(False).all() or rows.obs_taxi.isna().any():
            raise SystemExit(f"{month}: prepared fit rows are not all with-clock")
        rows["MVT_ID_mvt"] = rows.MVT_ID_mvt.astype("int64")
        rows["scope"] = "fold"
        rows["event_time_s"] = event_seconds(rows.MVT_TIME_UTC_mvt)
        rows["current_250m_s"] = rows.event_time_s - rows.obs_taxi.astype("float64")
        rows["current_wheels_off_s"] = rows.obs_wheels_off.astype("float64")
        rows["current_has_clock"] = True
        rows["noac"] = False
        rows["rank_order"] = -1
        fold_parts.append(rows)

    # Keep the ranking side in the exact input-template order. The saved serving
    # block supplies the same current observation identity used by the fold builder.
    serve = load_serve()
    ranking, _ = serve.ranking_block()
    template = pd.read_parquet(TEMPLATE, columns=["MVT_ID_mvt"])
    template["MVT_ID_mvt"] = template.MVT_ID_mvt.astype("int64")
    if template.MVT_ID_mvt.duplicated().any():
        raise SystemExit("submitting template has duplicate movement IDs")
    ranking["MVT_ID_mvt"] = ranking.MVT_ID_mvt.astype("int64")
    if ranking.MVT_ID_mvt.duplicated().any() or len(ranking) != len(template):
        raise SystemExit("saved ranking block does not have one row per submitting ID")
    rank = template.reset_index(names="rank_order").merge(
        ranking[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "STAND_mvt", "obs_icao",
                 "obs_wheels_off", "obs_taxi", "obs_has_clock", "noac"]],
        on="MVT_ID_mvt", how="left", validate="one_to_one", sort=False,
    )
    rank["month"] = pd.to_datetime(rank.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m")
    rank = rank.loc[rank.month.isin(RANK_MONTHS)].copy()
    rank["airport"] = rank.ADEP_mvt.astype(str)
    rank["day"] = pd.to_datetime(rank.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    rank["fold"] = None
    rank["scope"] = "ranking"
    rank["event_time_s"] = event_seconds(rank.MVT_TIME_UTC_mvt)
    rank["current_250m_s"] = rank.event_time_s - pd.to_numeric(rank.obs_taxi, errors="coerce")
    rank["current_wheels_off_s"] = pd.to_numeric(rank.obs_wheels_off, errors="coerce")
    rank["current_has_clock"] = rank.obs_has_clock.fillna(False).astype(bool)
    rank["noac"] = rank.noac.fillna(True).astype(bool)
    rank = rank.rename(columns={"rank_order": "rank_order"})
    rank_ids = set(rank.MVT_ID_mvt)
    if len(rank) and not np.array_equal(
        rank.sort_values("rank_order").rank_order.to_numpy(),
        template.reset_index(names="rank_order").loc[
            template.reset_index(names="rank_order").MVT_ID_mvt.isin(rank_ids), "rank_order"
        ].to_numpy(),
    ):
        raise SystemExit("ranking rows lost submitting order")

    common = [
        "MVT_ID_mvt", "scope", "fold", "month", "airport", "day", "rank_order", "event_time_s",
        "current_250m_s", "current_wheels_off_s", "obs_icao", "current_has_clock", "noac",
        "STAND_mvt", "survey_stand_lat", "survey_stand_lon", "survey_source", "survey_stand_key",
        "sham_stand_lat", "sham_stand_lon", "sham_source",
    ]
    fold_keys = add_stands(pd.concat(fold_parts, ignore_index=True), stand_exact, stand_norm)
    fold_keys = add_sham_stands(fold_keys, sham_stands)
    rank_keys = add_stands(rank, stand_exact, stand_norm)
    rank_keys["STAND_mvt"] = rank_keys["STAND_mvt"].astype("string")
    fold_keys["STAND_mvt"] = fold_keys["STAND_mvt"].astype("string")
    rank_keys["sham_stand_lat"] = np.nan
    rank_keys["sham_stand_lon"] = np.nan
    rank_keys["sham_source"] = None
    keys = pd.concat([fold_keys[common], rank_keys[common]], ignore_index=True)
    keys["MVT_ID_mvt"] = keys.MVT_ID_mvt.astype("int64")
    if keys.duplicated(["scope", "MVT_ID_mvt"]).any():
        raise SystemExit("clock key data has duplicate row identities")
    if keys.loc[keys.scope.eq("fold"), "MVT_ID_mvt"].nunique() != len(prepared):
        raise SystemExit("fold key count differs from the prepared eligible frame")
    validate_current_time_scale(keys)
    forbidden = {"y_true", "r12", "r_v22", "target", "label"}
    if forbidden.intersection(keys.columns):
        raise SystemExit("target field found in the label-free key dataset")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "clock_keys.parquet"
    keys.to_parquet(output, index=False, compression="zstd")
    template_path = OUT / "ranking_ids.csv"
    template.to_csv(template_path, index=False, lineterminator="\n")
    sfx = "" if RANKING_MODE == "ranking" else f"-{RANKING_MODE}"
    metadata = {
        "title": "PRC exp-127b clock re-derive keys" + (f" ({RANKING_MODE})" if sfx else ""),
        "id": f"{kaggle_user()}/prc-exp127b-clock-rederive-keys{sfx}",
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    harvest = json.loads(POINT_MANIFEST.read_text())
    point_days: dict[str, list[dict]] = {}
    for month in sorted(set(FOLD_MONTHS) | RANK_MONTHS):
        records = []
        for day, rec in sorted(harvest["months"][month]["days"].items()):
            points = rec.get("outputs", {}).get("points")
            if points is None:
                raise SystemExit(f"{month} {day}: harvest has no saved point output")
            records.append({
                "day": day,
                "file": f"points_{day}.parquet",
                "kernel_slug": f"{kaggle_user()}/" + rec["kernel"],
                "kernel_version": point_kernel_versions()[f"{kaggle_user()}/" + rec["kernel"]]["version"],
                "kernel_last_run_utc": point_kernel_versions()[f"{kaggle_user()}/" + rec["kernel"]]["last_run_utc"],
                "expected_bytes": int(points["bytes"]),
                "expected_rows": int(points["rows"]),
                "sha256": points["sha256"],
            })
        n_days = calendar.monthrange(int(month[:4]), int(month[5:7]))[1]   # 31 for every ranking-mode month
        if len(records) != n_days:
            raise SystemExit(f"{month}: expected {n_days} source point files, found {len(records)}")
        point_days[month] = records
    points_manifest_path = OUT / "points_expected.json"
    points_manifest_path.write_text(json.dumps({
        "source": "experiments/exp-045-harvest/manifest.json",
        "source_manifest_sha256": sha256(POINT_MANIFEST),
        "points_live_on": harvest["points_live_on"],
        "source_kernel_versions": point_kernel_versions(),
        "months": point_days,
        # the coverage kernel (rederive_clock.py) reads its months from this field
        **({"ranking_mode": RANKING_MODE} if RANKING_MODE != "ranking" else {}),
    }, indent=2) + "\n")
    receipt = {
        "experiment": "exp-127b-clock-rederive",
        "label_free": True,
        "rows": int(len(keys)),
        "fold_rows": int(keys.scope.eq("fold").sum()),
        "ranking_rows": int(keys.scope.eq("ranking").sum()),
        "ranking_months": sorted(RANK_MONTHS),
        **({"ranking_mode": RANKING_MODE, "template": str(TEMPLATE)} if RANKING_MODE != "ranking" else {}),
        "lfpg_rows_retained_for_reporting_but_without_coordinates": int(
            keys.airport.eq(LFPG).sum()
        ),
        "fold_eligible_counts": {
            str(k): int(v) for k, v in keys.loc[keys.scope.eq("fold")].groupby("fold").size().items()
        },
        "target_columns_present": False,
        "inputs": {
            str(Path(path).relative_to(MAIN) if Path(path).is_relative_to(MAIN) else path): digest
            for path, digest in source_hashes.items()
        },
        "outputs": {
            "clock_keys.parquet": {"bytes": output.stat().st_size, "sha256": sha256(output)},
            "ranking_ids.csv": {"bytes": template_path.stat().st_size, "sha256": sha256(template_path)},
            "points_expected.json": {"bytes": points_manifest_path.stat().st_size,
                                      "sha256": sha256(points_manifest_path)},
        },
    }
    (OUT / "keys_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
