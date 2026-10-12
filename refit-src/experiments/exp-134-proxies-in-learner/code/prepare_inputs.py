#!/usr/bin/env python3
"""Build the private, label-free exp-134 proxy input package on the Mac."""
from __future__ import annotations

import hashlib
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq



def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

MAIN = Path(os.environ.get("PRC_ROOT", Path(__file__).resolve().parents[3])).resolve()
REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PRC_EXP134_OUT", MAIN / "artifacts/rebuild/exp134")).resolve()
POINT_MANIFEST = MAIN / "experiments/exp-045-harvest/manifest.json"
SPEC = REPO / "experiments/exp-134-proxies-in-learner/SPEC.md"
ID = "MVT_ID_mvt"
MONTHS_2025 = tuple(f"2025-{m:02d}" for m in range(1, 13))
RANK_MONTHS = ("2026-01", "2026-07")
RANKING_PATH = MAIN / "data/raw/ranking.parquet"
GEOMETRY = REPO / "experiments/exp-127-clock-precision/stand_geometry.csv"
# exp-186 (2026-10-08, desk request d for finaldata): $PRC_RANKING_MODE=final takes the months and the ranking
# file of the final file (prc/data.py RANKING_SPECS["final"], accepted by its SHA-256) and names the dataset
# ...-input-final, so a final push leaves the ranking dataset alone; without $PRC_EXP134_OUT it writes to the
# sibling folder exp134-final. The stand table copied below is the exp-127 table of the mode (stand_geometry-final.csv,
# from build_stand_geometry.py in the final mode); fit_exp134.py reads the mode from source_inventory.json. The
# ranking mode keeps every value.
RANKING_MODE = os.environ.get("PRC_RANKING_MODE") or "ranking"
if RANKING_MODE != "ranking":
    sys.path.insert(0, str(MAIN))
    from prc import data as _D
    RANK_MONTHS = _D.rank_months(RANKING_MODE)
    RANKING_PATH = _D.ranking_file(RANKING_MODE)
    GEOMETRY = GEOMETRY.with_name(f"{GEOMETRY.stem}-{RANKING_MODE}{GEOMETRY.suffix}")
    if "PRC_EXP134_OUT" not in os.environ:
        OUT = OUT.with_name(f"{OUT.name}-{RANKING_MODE}")
DEPARTURE_COLUMNS = [
    ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
    "EOBT_1_flt", "FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
]
ARRIVAL_COLUMNS = [
    ID, "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
    "TAXITIME_SEC_mvt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def month_path(month: str) -> Path:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return MAIN / "data/raw" / f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"


def to_ids(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    ids = pd.to_numeric(out[ID], errors="raise")
    if ids.isna().any() or (ids != ids.astype("int64")).any():
        raise SystemExit(f"{label}: invalid movement ID")
    out[ID] = ids.astype("int64")
    if out[ID].duplicated().any():
        raise SystemExit(f"{label}: duplicate movement IDs")
    return out


def departure_rows(path: Path, month: str, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=DEPARTURE_COLUMNS)
    frame = frame.loc[frame["PHASE_mvt"].eq("DEP")].copy()
    event_month = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    frame = frame.loc[event_month.dt.strftime("%Y-%m").eq(month)].copy()
    frame = frame.drop(columns="PHASE_mvt")
    frame = to_ids(frame, label)
    frame["month"] = month
    frame["source"] = label
    return frame


def arrival_rows(path: Path, month: str, label: str, *, ranked: bool) -> pd.DataFrame:
    # The taxi-time field is selected only after an ARR-only parquet filter.
    filters = [("PHASE_mvt", "=", "ARR")]
    frame = pd.read_parquet(path, columns=ARRIVAL_COLUMNS, filters=filters)
    if "PHASE_mvt" in frame:
        frame = frame.loc[frame["PHASE_mvt"].eq("ARR")].copy()
        frame = frame.drop(columns="PHASE_mvt")
    frame = to_ids(frame, label)
    event_month = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    frame = frame.loc[event_month.dt.strftime("%Y-%m").eq(month)].copy()
    frame["month"] = month
    frame["source"] = label
    return frame


def compact_points_manifest() -> tuple[dict, list[str]]:
    manifest = json.loads(POINT_MANIFEST.read_text())
    point_months = (*MONTHS_2025, *RANK_MONTHS)
    expected = {"source": str(POINT_MANIFEST), "source_sha256": sha256(POINT_MANIFEST),
                "months": {}}
    kernels: set[str] = set()
    for month in point_months:
        month_record = manifest["months"].get(month)
        if not month_record:
            raise SystemExit(f"exp-045 point manifest has no month {month}")
        records = []
        for day, row in sorted(month_record.get("days", {}).items()):
            point = row.get("outputs", {}).get("points")
            if point is None:
                raise SystemExit(f"exp-045 point manifest has no point file for {day}")
            slug = f"{kaggle_user()}/" + row["kernel"]
            kernels.add(slug)
            records.append({
                "day": day,
                "file": f"points_{day}.parquet",
                "kernel_slug": slug,
                "bytes": int(point["bytes"]),
                "rows": int(point["rows"]),
                "sha256": point["sha256"],
            })
        if len(records) != pd.Period(month).days_in_month:
            raise SystemExit(f"{month}: expected a point file for every calendar day")
        expected["months"][month] = records
    return expected, sorted(kernels)


def main() -> None:
    if not POINT_MANIFEST.is_file():
        raise SystemExit(f"missing exp-045 point manifest: {POINT_MANIFEST}")
    if not SPEC.is_file():
        raise SystemExit(f"missing frozen exp-134 SPEC: {SPEC}")
    OUT.mkdir(parents=True, exist_ok=True)

    departures = []
    arrivals = []
    source_hashes = {"spec": sha256(SPEC), "exp045_manifest": sha256(POINT_MANIFEST)}
    for month in MONTHS_2025:
        path = month_path(month)
        if not path.is_file():
            raise SystemExit(f"missing raw month input: {path}")
        source_hashes[str(path)] = sha256(path)
        departures.append(departure_rows(path, month, f"training:{month}"))
        arrivals.append(arrival_rows(path, month, f"training:{month}", ranked=False))

    ranking_path = RANKING_PATH
    if not ranking_path.is_file():
        raise SystemExit(f"missing ranking input: {ranking_path}")
    source_hashes[str(ranking_path)] = sha256(ranking_path)
    for month in RANK_MONTHS:
        departures.append(departure_rows(ranking_path, month, f"ranking:{month}"))
        arrivals.append(arrival_rows(ranking_path, month, f"ranking:{month}", ranked=True))

    dep = pd.concat(departures, ignore_index=True)
    arr = pd.concat(arrivals, ignore_index=True)
    if dep[ID].duplicated().any() or arr[ID].duplicated().any():
        raise SystemExit("input movement IDs are not unique across months")
    if "TAXITIME_SEC_mvt" in dep.columns:
        raise SystemExit("departure proxy input contains the departure target column")
    if "TAXITIME_SEC_mvt" not in arr.columns:
        raise SystemExit("ARR-only taxi-in input lacks TAXITIME_SEC_mvt")
    if not arr["TAXITIME_SEC_mvt"].notna().any():
        raise SystemExit("ARR-only taxi-in input has no usable values")

    dep_path = OUT / "proxy_departures.parquet"
    arr_path = OUT / "proxy_arrivals.parquet"
    dep.to_parquet(dep_path, index=False, compression="zstd")
    arr.to_parquet(arr_path, index=False, compression="zstd")

    point_expected, kernels = compact_points_manifest()
    (OUT / "points_expected.json").write_text(
        json.dumps(point_expected, indent=2, sort_keys=True) + "\n"
    )
    geometry_bytes = GEOMETRY.read_bytes()
    geometry_path = OUT / "stand_geometry.csv"
    geometry_path.write_bytes(geometry_bytes)
    source_hashes["prepare_inputs.py"] = sha256(Path(__file__).resolve())

    code_out = OUT / "code"
    code_out.mkdir(exist_ok=True)
    for name in ("fit_exp134.py", "proxy_features.py", "kernel.py"):
        source = Path(__file__).resolve().parent / name
        if not source.is_file():
            raise SystemExit(f"missing exp-134 runtime source: {source}")
        shutil.copy2(source, code_out / name)

    records = {}
    total = 0
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"dataset-metadata.json", "source_inventory.json"}:
            stat = path.stat()
            records[path.relative_to(OUT).as_posix()] = {"bytes": stat.st_size, "sha256": sha256(path)}
            total += stat.st_size
    inventory = {
        "experiment": "exp-134",
        "spec_sha256": sha256(SPEC),
        "spec_commit": os.environ.get("PRC_SOURCE_REV", "export:" + sha256(SPEC)),
        "ranking_target_read": False,
        **({"ranking_mode": RANKING_MODE, "ranking_months": list(RANK_MONTHS)} if RANKING_MODE != "ranking" else {}),
        "departure_target_column_read": False,
        "taxitime_read_for": "ARR rows only",
        "source_sha256": source_hashes,
        "files": records,
        "total_bytes": total,
        "departure_rows_by_month": {
            month: int(dep["month"].eq(month).sum()) for month in (*MONTHS_2025, *RANK_MONTHS)
        },
        "arrival_rows_by_month": {
            month: int(arr["month"].eq(month).sum()) for month in (*MONTHS_2025, *RANK_MONTHS)
        },
        "point_source_kernels": kernels,
        "point_file_count": sum(len(rows) for rows in point_expected["months"].values()),
    }
    metadata = {
        "title": "PRC exp-134 input-only proxies and fit code" + (f" ({RANKING_MODE})" if RANKING_MODE != "ranking" else ""),
        "id": f"{kaggle_user()}/prc-exp134-input" + (f"-{RANKING_MODE}" if RANKING_MODE != "ranking" else ""),
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / "source_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    for name in ("proxy_departures.parquet", "proxy_arrivals.parquet"):
        schema = pq.read_schema(OUT / name)
        if name == "proxy_departures.parquet" and "TAXITIME_SEC_mvt" in schema.names:
            raise SystemExit("departure parquet has a target field")
    print(json.dumps({
        "files": records,
        "departure_rows_by_month": inventory["departure_rows_by_month"],
        "arrival_rows_by_month": inventory["arrival_rows_by_month"],
        "point_source_kernels": kernels,
        "point_file_count": inventory["point_file_count"],
        "total_bytes": total,
        "inventory_sha256": sha256(OUT / "source_inventory.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
