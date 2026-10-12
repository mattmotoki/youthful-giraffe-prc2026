#!/usr/bin/env python3
"""Rebuild the v22 fold baseline and prepare stand-geometry fold frames."""

from __future__ import annotations

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
import pyarrow.parquet as pq


MAIN = Path(os.environ.get("PRC_ROOT", Path(__file__).resolve().parents[2])).resolve()
EXP = Path(__file__).resolve().parent
WORK = EXP / "work"
SUPPORT = EXP / "support"
FOLDS = ("july_loo", "december", "january")
MONTH_OF = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
TRAINING = {
    "2025-07": "training_2025-07-01_2025-08-01.parquet",
    "2025-12": "training_2025-12-01_2026-01-01.parquet",
    "2025-01": "training_2025-01-01_2025-02-01.parquet",
}
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
V22_VECTOR = MAIN / "experiments/exp-046-surfacefit/work/d-seasonal-bag10-255-free.npz"
EXPECTED = MAIN / "experiments/exp-047-queue/analysis/census-v22.json"
EARTH_M = 6_371_008.8
SEED = 12727


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def normalize_runway(value: object) -> str:
    key = normalize(value)
    if key.startswith("RWY"):
        key = key[3:]
    elif key.startswith("RW"):
        key = key[2:]
    return key


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * (EARTH_M / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_queue_base():
    code = MAIN / "experiments/exp-047-queue/code"
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(MAIN))
    spec = importlib.util.spec_from_file_location("exp047_clockfit_base", code / "base.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the saved exp-047 fold loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_map(frame: pd.DataFrame, key: str, value_cols: list[str], group: str):
    """Build an unambiguous exact and normalized lookup for one field."""
    exact = {}
    normalized = {}
    for (airport, name), rows in frame.groupby([group, key], sort=True, dropna=False):
        if pd.isna(name):
            continue
        values = rows[value_cols].drop_duplicates()
        if len(values) == 1:
            exact[(str(airport), str(name))] = tuple(values.iloc[0].tolist())
    for (airport, name), values in exact.items():
        nkey = (airport, normalize(name))
        normalized.setdefault(nkey, []).append((name, values))
    unambiguous = {
        k: vals[0]
        for k, vals in normalized.items()
        if k[1] and len(vals) == 1
    }
    return exact, unambiguous


def resolve_name(airport: str, name: object, exact: dict, normalized: dict):
    if pd.isna(name):
        return None
    name = str(name)
    found = exact.get((airport, name))
    if found is not None:
        return name, found, "exact"
    item = normalized.get((airport, normalize(name)))
    if item is not None:
        return item[0], item[1], "normalized_unique"
    return None


def stable_rng(airport: str, month: str) -> np.random.Generator:
    code = zlib.crc32(f"{SEED}:{airport}:{month}".encode("utf-8"))
    return np.random.default_rng(code)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    SUPPORT.mkdir(parents=True, exist_ok=True)

    # The lane assignment requires this exact saved-v22 fold view before a fit.
    base = load_queue_base()
    folds = base.load(with_v12=True)
    expected = json.loads(EXPECTED.read_text())["fold_totals"]
    vector = np.load(V22_VECTOR)

    geometry_path = EXP / "stand_geometry.csv"
    geometry = pd.read_csv(geometry_path, dtype={"airport": str, "stand_name_mvt": str,
                                                  "runway_end": str})
    if geometry.empty:
        raise SystemExit("stand_geometry.csv is empty")

    # A stand is repeated once per runway end. Select Gateway first, then OSM, and
    # require a single coordinate and source for each resulting stand key.
    source_order = {"xplane_gateway": 0, "openstreetmap": 1}
    geometry["source_order"] = geometry["stand_source"].map(source_order).fillna(9)
    stand_rows = geometry.sort_values(["source_order", "stand_source", "runway_end"])
    stand_source = stand_rows.groupby(["airport", "stand_name_mvt"], as_index=False).first()
    stand_exact, stand_normalized = unique_map(
        stand_source,
        "stand_name_mvt",
        ["stand_latitude", "stand_longitude", "stand_source"],
        "airport",
    )

    # Runway endpoints are independent of the stand key. Refuse an ambiguous
    # normalized designator rather than attach a movement to an arbitrary end.
    runway_rows = geometry[["airport", "runway_end", "runway_end_latitude", "runway_end_longitude"]].drop_duplicates()
    runway_exact, runway_normalized = unique_map(
        runway_rows,
        "runway_end",
        ["runway_end_latitude", "runway_end_longitude"],
        "airport",
    )
    runway_end_names = {
        (str(a), str(name)): (str(name), values)
        for (a, name), values in runway_exact.items()
    }
    runway_by_norm = runway_normalized

    summaries = []
    baseline_rows = []
    feature_rows = []
    per_airport_coverage = []
    cluster_rows = []
    input_hashes: dict[str, str] = {
        str(V22_VECTOR): sha256(V22_VECTOR),
        str(EXPECTED): sha256(EXPECTED),
        str(geometry_path): sha256(geometry_path),
        str(MAIN / "experiments/exp-047-queue/code/base.py"): sha256(MAIN / "experiments/exp-047-queue/code/base.py"),
        str(MAIN / "experiments/exp-047-queue/work/census_v22.py"): sha256(MAIN / "experiments/exp-047-queue/work/census_v22.py"),
    }

    for fold in FOLDS:
        month = MONTH_OF[fold]
        training_path = MAIN / "data/raw" / TRAINING[month]
        record_path = MAIN / f"experiments/exp-046-surfacefit/analysis/block-{month}.json"
        record = json.loads(record_path.read_text())
        feature_path = MAIN / "experiments/exp-046-surfacefit/work" / record["block_cache"]
        class_path = MAIN / f"artifacts/exp-047/block/{month}.parquet"
        geom_path = next((MAIN / f"experiments/exp-046-surfacefit/work/geom-{month}-geom-1-*.parquet").parent.glob(f"geom-{month}-geom-1-*.parquet"), None)
        if geom_path is None:
            raise SystemExit(f"No saved exp-046 geometry cache for {month}")
        for path in (training_path, record_path, feature_path, class_path, geom_path):
            input_hashes[str(path)] = sha256(path)

        training = pq.read_table(
            training_path,
            columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"],
        ).to_pandas()
        training = training.loc[training["PHASE_mvt"].eq("DEP")].copy()
        training["MVT_ID_mvt"] = training["MVT_ID_mvt"].astype("int64")
        if training["MVT_ID_mvt"].duplicated().any():
            raise SystemExit(f"{month}: duplicate departure movement ids")
        training["airport"] = training["ADEP_mvt"].astype(str)

        block = pq.read_table(class_path, columns=["MVT_ID_mvt", "obs_has_clock"]).to_pandas()
        block["MVT_ID_mvt"] = block["MVT_ID_mvt"].astype("int64")
        feat = pq.read_table(feature_path, columns=["MVT_ID_mvt", "stand_lat", "stand_lon", "obs_icao", "obs_wheels_off"]).to_pandas()
        feat["MVT_ID_mvt"] = feat["MVT_ID_mvt"].astype("int64")
        geom = pq.read_table(geom_path, columns=["MVT_ID_mvt", "rwy_lat", "rwy_lon"]).to_pandas()
        geom["MVT_ID_mvt"] = geom["MVT_ID_mvt"].astype("int64")
        if block["MVT_ID_mvt"].duplicated().any() or feat["MVT_ID_mvt"].duplicated().any() or geom["MVT_ID_mvt"].duplicated().any():
            raise SystemExit(f"{month}: a saved feature source has duplicate movement ids")

        pred = folds[fold].copy()
        pred["MVT_ID_mvt"] = pred["MVT_ID_mvt"].astype("int64")
        if len(vector[fold]) != len(pred):
            raise SystemExit(f"{fold}: v22 vector length does not match the fold frame")
        pred["v22_delta"] = vector[fold].astype("float64")
        pred["r12"] = pred["r12"].astype("float64")
        pred["apt"] = pred["apt"].astype(str)
        pred["day"] = pred["day"].astype(str)
        pred = pred[["MVT_ID_mvt", "y_true", "r12", "v22_delta", "apt", "day", "bin"]]

        frame = pred.merge(training[["MVT_ID_mvt", "airport", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]],
                           on="MVT_ID_mvt", how="left", validate="one_to_one")
        frame = frame.merge(block, on="MVT_ID_mvt", how="left", validate="one_to_one")
        frame = frame.merge(feat, on="MVT_ID_mvt", how="left", validate="one_to_one")
        frame = frame.merge(geom, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if frame["airport"].isna().any() or not frame["airport"].eq(frame["apt"]).all():
            raise SystemExit(f"{month}: fold IDs do not align with training airport rows")
        frame["month"] = month
        frame["fold"] = fold
        frame["noac"] = frame["AIRCRAFT_TYPE_mvt"].isna()
        frame["eligible"] = frame["obs_has_clock"].fillna(False).astype(bool) & ~frame["noac"]
        frame["r_v22"] = frame["r12"] - frame["v22_delta"]
        clusters = (
            frame.groupby(["airport", "day"], as_index=False, sort=True)
            .agg(fold_rows=("MVT_ID_mvt", "size"), eligible_rows=("eligible", "sum"))
        )
        clusters.insert(0, "month", month)
        clusters.insert(0, "fold", fold)
        cluster_rows.append(clusters)
        residual_served = np.where(frame["eligible"].to_numpy(), frame["r_v22"].to_numpy(), frame["r12"].to_numpy())
        got_rmse = float(np.sqrt(np.mean(residual_served**2)))
        want_rmse = float(expected[fold]["rmse_v22"])
        if abs(got_rmse - want_rmse) > 1e-8:
            raise SystemExit(f"{fold}: v22 RMSE {got_rmse:.12f} does not reproduce {want_rmse:.12f}")
        baseline_rows.append({
            "fold": fold,
            "month": month,
            "rows_all": int(len(frame)),
            "rows_with_clock_not_noac": int(frame["eligible"].sum()),
            "rmse_v22_expected": want_rmse,
            "rmse_v22_reproduced": got_rmse,
            "max_abs_error_s": abs(got_rmse - want_rmse),
            "weighted_share": float(base.SHARES.get(fold, 0.0)),
            "with_clock_rmse": float(np.sqrt(np.mean(frame.loc[frame["eligible"], "r_v22"]**2))),
        })

        eligible = frame.loc[frame["eligible"]].copy()
        resolved_stands = [
            resolve_name(str(airport), stand, stand_exact, stand_normalized)
            for airport, stand in zip(eligible["airport"], eligible["STAND_mvt"])
        ]
        eligible["survey_stand_key"] = [item[0] if item else None for item in resolved_stands]
        eligible["survey_stand_lat"] = [item[1][0] if item else np.nan for item in resolved_stands]
        eligible["survey_stand_lon"] = [item[1][1] if item else np.nan for item in resolved_stands]
        eligible["survey_source"] = [item[1][2] if item else None for item in resolved_stands]
        eligible["survey_match_rule"] = [item[2] if item else None for item in resolved_stands]

        resolved_runways = []
        for airport, runway in zip(eligible["airport"], eligible["RUNWAY_mvt"]):
            apt = str(airport)
            if pd.isna(runway):
                resolved_runways.append(None)
                continue
            end = runway_end_names.get((apt, str(runway)))
            rule = "exact"
            if end is None:
                end = runway_by_norm.get((apt, normalize_runway(runway)))
                rule = "normalized_unique"
            resolved_runways.append((end[1], rule) if end is not None else None)
        eligible["runway_lat"] = [item[0][0] if item else np.nan for item in resolved_runways]
        eligible["runway_lon"] = [item[0][1] if item else np.nan for item in resolved_runways]
        eligible["runway_match_rule"] = [item[1] if item else None for item in resolved_runways]

        stand_lat = eligible["survey_stand_lat"].to_numpy("float64")
        stand_lon = eligible["survey_stand_lon"].to_numpy("float64")
        survey_ok = np.isfinite(stand_lat) & np.isfinite(stand_lon)
        obs_lat = eligible["stand_lat"].to_numpy("float64")
        obs_lon = eligible["stand_lon"].to_numpy("float64")
        obs_ok = np.isfinite(obs_lat) & np.isfinite(obs_lon)
        wheel_lat = eligible["rwy_lat"].to_numpy("float64")
        wheel_lon = eligible["rwy_lon"].to_numpy("float64")
        wheel_ok = np.isfinite(wheel_lat) & np.isfinite(wheel_lon)
        runway_lat = eligible["runway_lat"].to_numpy("float64")
        runway_lon = eligible["runway_lon"].to_numpy("float64")
        runway_ok = np.isfinite(runway_lat) & np.isfinite(runway_lon)
        eligible["survey_to_obs_stand_m"] = np.where(
            survey_ok & obs_ok,
            haversine_km(stand_lat, stand_lon, obs_lat, obs_lon) * 1000.0,
            np.nan,
        )
        eligible["survey_to_wheels_km"] = np.where(
            survey_ok & wheel_ok,
            haversine_km(stand_lat, stand_lon, wheel_lat, wheel_lon),
            np.nan,
        )
        eligible["survey_to_runway_km"] = np.where(
            survey_ok & runway_ok,
            haversine_km(stand_lat, stand_lon, runway_lat, runway_lon),
            np.nan,
        )
        # Unknown stand matches have an unknown source. Keep that distinction so the
        # learner can separate a missing coordinate from a surveyed Gateway stand.
        eligible["survey_is_osm"] = eligible["survey_source"].map(
            {"xplane_gateway": 0.0, "openstreetmap": 1.0}
        ).astype("float64")

        # The sham permutes the complete stand-coordinate/source tuple within each
        # airport and fold month. It keeps each row's runway and ADS-B observations fixed.
        perm_rows = stand_source[["airport", "stand_name_mvt", "stand_latitude", "stand_longitude", "stand_source"]].copy()
        perm_map = {}
        for airport, group in perm_rows.groupby("airport", sort=True):
            names = group["stand_name_mvt"].astype(str).to_numpy()
            tuples = group[["stand_latitude", "stand_longitude", "stand_source"]].to_numpy(object)
            permuted = tuples[stable_rng(str(airport), month).permutation(len(tuples))]
            perm_map[str(airport)] = {
                name: (float(value[0]), float(value[1]), str(value[2]))
                for name, value in zip(names, permuted)
            }
        assigned_coords = [
            perm_map.get(str(airport), {}).get(str(stand_key)) if pd.notna(stand_key) else None
            for airport, stand_key in zip(eligible["airport"], eligible["survey_stand_key"])
        ]
        eligible["perm_stand_lat"] = [item[0] if item else np.nan for item in assigned_coords]
        eligible["perm_stand_lon"] = [item[1] if item else np.nan for item in assigned_coords]
        eligible["perm_source"] = [item[2] if item else None for item in assigned_coords]
        perm_lat = eligible["perm_stand_lat"].to_numpy("float64")
        perm_lon = eligible["perm_stand_lon"].to_numpy("float64")
        perm_ok = np.isfinite(perm_lat) & np.isfinite(perm_lon)
        eligible["sham_survey_to_obs_stand_m"] = np.where(
            perm_ok & obs_ok,
            haversine_km(perm_lat, perm_lon, obs_lat, obs_lon) * 1000.0,
            np.nan,
        )
        eligible["sham_survey_to_wheels_km"] = np.where(
            perm_ok & wheel_ok,
            haversine_km(perm_lat, perm_lon, wheel_lat, wheel_lon),
            np.nan,
        )
        eligible["sham_survey_to_runway_km"] = np.where(
            perm_ok & runway_ok,
            haversine_km(perm_lat, perm_lon, runway_lat, runway_lon),
            np.nan,
        )
        eligible["sham_survey_is_osm"] = eligible["perm_source"].map(
            {"xplane_gateway": 0.0, "openstreetmap": 1.0}
        ).astype("float64")

        for airport in AIRPORTS:
            rows = eligible.loc[eligible["airport"].eq(airport)]
            if rows.empty:
                continue
            per_airport_coverage.append({
                "fold": fold,
                "month": month,
                "airport": airport,
                "eligible_rows": int(len(rows)),
                "stand_matched_rows": int(rows["survey_stand_key"].notna().sum()),
                "stand_coverage": float(rows["survey_stand_key"].notna().mean()),
                "runway_end_matched_rows": int(rows["runway_match_rule"].notna().sum()),
                "runway_end_coverage": float(rows["runway_match_rule"].notna().mean()),
                "adsb_stand_offset_rows": int(rows["survey_to_obs_stand_m"].notna().sum()),
                "wheels_position_rows": int(rows["survey_to_wheels_km"].notna().sum()),
                "stand_runway_rows": int(rows["survey_to_runway_km"].notna().sum()),
                "all_three_geometry_rows": int(rows[["survey_to_obs_stand_m", "survey_to_wheels_km", "survey_to_runway_km"]].notna().all(axis=1).sum()),
                "osm_rows": int(rows["survey_is_osm"].sum()),
            })

        keep = [
            "MVT_ID_mvt", "fold", "month", "airport", "day", "bin", "STAND_mvt", "RUNWAY_mvt",
            "r_v22", "survey_to_obs_stand_m", "survey_to_wheels_km", "survey_to_runway_km", "survey_is_osm",
            "sham_survey_to_obs_stand_m", "sham_survey_to_wheels_km", "sham_survey_to_runway_km", "sham_survey_is_osm",
        ]
        feature_rows.append(eligible[keep].copy())

    baselines = pd.DataFrame(baseline_rows)
    coverage = pd.DataFrame(per_airport_coverage)
    cluster_population = pd.concat(cluster_rows, ignore_index=True)
    prepared = pd.concat(feature_rows, ignore_index=True)
    weighted_total_mse = 0.0
    weighted_geometry_sse = 0.0
    for row in baseline_rows:
        fold = row["fold"]
        frame = folds[fold]
        ids = frame["MVT_ID_mvt"].astype("int64").to_numpy()
        eligible_ids = set(prepared.loc[prepared["fold"].eq(fold), "MVT_ID_mvt"].astype("int64"))
        mask = np.isin(ids, np.fromiter(eligible_ids, dtype=np.int64, count=len(eligible_ids)))
        resid = frame["r12"].to_numpy("float64") - vector[fold].astype("float64")
        residual_served = resid.copy()
        noac = pd.read_parquet(MAIN / "data/raw" / TRAINING[MONTH_OF[fold]], columns=["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt"])
        noac = noac.loc[noac["PHASE_mvt"].eq("DEP") & noac["AIRCRAFT_TYPE_mvt"].isna(), "MVT_ID_mvt"].astype("int64").to_numpy()
        clock = pd.read_parquet(MAIN / f"artifacts/exp-047/block/{MONTH_OF[fold]}.parquet", columns=["MVT_ID_mvt", "obs_has_clock"])
        clock = clock.set_index("MVT_ID_mvt")["obs_has_clock"].reindex(ids).fillna(False).to_numpy(bool)
        is_noac = np.isin(ids, noac)
        residual_served[~(clock & ~is_noac)] = frame["r12"].to_numpy("float64")[~(clock & ~is_noac)]
        weight = float(base.SHARES.get(fold, 0.0))
        weighted_total_mse += weight * float(np.mean(residual_served**2))
        weighted_geometry_sse += weight * float(np.sum(residual_served[mask] ** 2) / len(frame))
    share = weighted_geometry_sse / weighted_total_mse if weighted_total_mse else 0.0

    coverage.to_csv(SUPPORT / "geometry_feature_coverage.csv", index=False, lineterminator="\n")
    cluster_population.to_csv(SUPPORT / "fold_cluster_population.csv", index=False, lineterminator="\n")
    prepared.to_parquet(WORK / "fold_frames.parquet", index=False, compression="zstd")
    baseline_record = {
        "experiment": "exp-127-clock-precision",
        "method": "exact saved exp-047 base.load(with_v12=True), v22 vector residual r12-d on obs_has_clock and not noac; p12 residual r12 on its complement, as in census_v22.py",
        "expected_from": str(EXPECTED),
        "folds": baseline_rows,
        "board_weights": {f: float(base.SHARES.get(f, 0.0)) for f in FOLDS},
        "eligible_with_clock_board_weighted_v22_sse_share": share,
        "weighted_board_mse": weighted_total_mse,
        "permutation": "stand latitude, longitude and source are permuted across surveyed stands within each airport and fold month; each row's observed stand, wheels point, runway and response stay fixed",
        "fit_inputs": ["survey_to_obs_stand_m", "survey_to_runway_km", "survey_is_osm"],
        "coverage_only_inputs": ["survey_to_wheels_km"],
        "candidate_rows": int(len(prepared)),
        "coverage_csv_sha256": sha256(SUPPORT / "geometry_feature_coverage.csv"),
        "cluster_population_csv_sha256": sha256(SUPPORT / "fold_cluster_population.csv"),
        "airport_day_clusters_by_fold": {
            f: int((cluster_population["fold"] == f).sum()) for f in FOLDS
        },
        "prepared_parquet_sha256": sha256(WORK / "fold_frames.parquet"),
        "prepared_parquet_bytes": (WORK / "fold_frames.parquet").stat().st_size,
        "input_hashes": input_hashes,
    }
    (SUPPORT / "baseline_reproduction.json").write_text(json.dumps(baseline_record, indent=2) + "\n")
    print(json.dumps(baseline_record, indent=2))


if __name__ == "__main__":
    main()
