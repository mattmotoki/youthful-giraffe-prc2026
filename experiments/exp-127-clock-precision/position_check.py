#!/usr/bin/env python3
"""Compare matched survey/OSM stand positions with ADS-B stationary positions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MAIN = Path("/path/to/competitions/prc-data-challenge-2026")
EXP = Path(__file__).resolve().parent
EARTH_RADIUS_M = 6_371_008.8
MONTHS = ("2025-01", "2025-07", "2025-12")
TRAINING_FILES = {
    "2025-01": "training_2025-01-01_2025-02-01.parquet",
    "2025-07": "training_2025-07-01_2025-08-01.parquet",
    "2025-12": "training_2025-12-01_2026-01-01.parquet",
}
SOURCES = ("xplane_gateway", "openstreetmap", "combined")
POPULATIONS = ("all_adsb_positions", "with_clock")
N_PERMUTATIONS = 500
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def haversine_arrays(lat1, lon1, lat2, lon2) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def geometry_maps(path: Path) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    raw = pd.read_csv(path)
    maps = {source: {} for source in SOURCES}
    for (airport, stand), rows in raw.groupby(["airport", "stand_name_mvt"], sort=True):
        # Each stand is repeated once per runway end, but its coordinates must stay fixed.
        by_source = {s: g for s, g in rows.groupby("stand_source")}
        selected = {}
        for source, g in by_source.items():
            coords = g[["stand_latitude", "stand_longitude"]].drop_duplicates().to_numpy(float)
            if len(coords) != 1:
                raise ValueError(f"Stand coordinates vary across runway rows: {airport} {stand} {source}")
            selected[source] = (float(coords[0, 0]), float(coords[0, 1]))
            maps.setdefault(source, {}).setdefault(airport, {})[stand] = selected[source]
        if selected:
            # Gateway wins on overlap. The OSM merger only adds missing stand names.
            maps["combined"].setdefault(airport, {})[stand] = selected.get(
                "xplane_gateway", selected.get("openstreetmap")
            )
    return maps


def load_population() -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    input_hashes = {}
    for month in MONTHS:
        training_path = MAIN / "data/raw" / TRAINING_FILES[month]
        record_path = MAIN / f"experiments/exp-046-surfacefit/analysis/block-{month}.json"
        record = json.loads(record_path.read_text())
        feature_path = MAIN / f"experiments/exp-046-surfacefit/work/{record['block_cache']}"
        class_path = MAIN / f"artifacts/exp-047/block/{month}.parquet"
        rank = pq.read_table(
            training_path,
            columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"],
        ).to_pandas()
        rank = rank.loc[
            rank["PHASE_mvt"].eq("DEP"),
            ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"],
        ]
        if rank["MVT_ID_mvt"].duplicated().any():
            raise ValueError(f"Duplicate movement id in {month} training file")
        feature = pq.read_table(
            feature_path, columns=["MVT_ID_mvt", "stand_lat", "stand_lon"]
        ).to_pandas()
        classes = pq.read_table(
            class_path, columns=["MVT_ID_mvt", "obs_has_clock"]
        ).to_pandas()
        if feature["MVT_ID_mvt"].duplicated().any() or classes["MVT_ID_mvt"].duplicated().any():
            raise ValueError(f"Non-unique movement id in {month} position/class source")
        joined = rank.merge(feature, on="MVT_ID_mvt", how="inner", validate="one_to_one")
        joined = joined.merge(classes, on="MVT_ID_mvt", how="inner", validate="one_to_one")
        joined["month"] = month
        frames.append(joined)
        input_hashes[str(training_path)] = sha256(training_path)
        input_hashes[str(feature_path)] = sha256(feature_path)
        input_hashes[str(class_path)] = sha256(class_path)
        input_hashes[str(record_path)] = sha256(record_path)
    out = pd.concat(frames, ignore_index=True)
    out["stand_lat"] = pd.to_numeric(out["stand_lat"], errors="coerce")
    out["stand_lon"] = pd.to_numeric(out["stand_lon"], errors="coerce")
    out = out.loc[
        out["ADEP_mvt"].notna()
        & out["STAND_mvt"].notna()
        & out["stand_lat"].notna()
        & out["stand_lon"].notna()
        & out["stand_lat"].between(-90, 90)
        & out["stand_lon"].between(-180, 180)
    ].copy()
    out["STAND_mvt"] = out["STAND_mvt"].astype(str)
    out["ADEP_mvt"] = out["ADEP_mvt"].astype(str)
    out["with_clock"] = out["obs_has_clock"].fillna(False).astype(bool) & out["AIRCRAFT_TYPE_mvt"].notna()
    return out, input_hashes


def metrics(rows: pd.DataFrame, positions: dict[str, tuple[float, float]], seed: int) -> dict:
    names = sorted(set(rows["STAND_mvt"]) & set(positions))
    matched = rows.loc[rows["STAND_mvt"].isin(names)].copy()
    if matched.empty:
        return {"n_adsb_rows": len(rows), "n_matched_rows": 0, "n_matched_stands": 0}

    coords = np.asarray([positions[n] for n in names], dtype=np.float64)
    name_to_i = {n: i for i, n in enumerate(names)}
    row_idx = matched["STAND_mvt"].map(name_to_i).to_numpy()
    obs_lat = matched["stand_lat"].to_numpy(dtype=np.float64)
    obs_lon = matched["stand_lon"].to_numpy(dtype=np.float64)
    real = haversine_arrays(obs_lat, obs_lon, coords[row_idx, 0], coords[row_idx, 1])

    rng = np.random.default_rng(seed)
    placebo_median = np.empty(N_PERMUTATIONS, dtype=np.float64)
    placebo_share50 = np.empty(N_PERMUTATIONS, dtype=np.float64)
    placebo_share100 = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for i in range(N_PERMUTATIONS):
        perm = rng.permutation(len(names))
        distance = haversine_arrays(obs_lat, obs_lon, coords[perm[row_idx], 0], coords[perm[row_idx], 1])
        placebo_median[i] = np.median(distance)
        placebo_share50[i] = np.mean(distance <= 50)
        placebo_share100[i] = np.mean(distance <= 100)
    share50 = float(np.mean(real <= 50))
    placebo_share50_q95 = float(np.quantile(placebo_share50, 0.95))
    placebo_share100_q95 = float(np.quantile(placebo_share100, 0.95))
    return {
        "n_adsb_rows": int(len(rows)),
        "n_matched_rows": int(len(matched)),
        "n_matched_stands": int(len(names)),
        "row_coverage": len(matched) / len(rows) if len(rows) else None,
        "real_median_distance_m": float(np.median(real)),
        "real_share_within_50m": share50,
        "real_share_within_100m": float(np.mean(real <= 100)),
        "real_share50_minus_placebo_q95": share50 - placebo_share50_q95,
        "placebo_median_distance_m": float(np.median(placebo_median)),
        "placebo_share_within_50m": float(np.median(placebo_share50)),
        "placebo_share_within_100m": float(np.median(placebo_share100)),
        "placebo_share50_q95": placebo_share50_q95,
        "placebo_share100_q95": placebo_share100_q95,
        "real_share50_above_placebo_q95": bool(share50 > placebo_share50_q95),
        "real_share50_far_above_placebo": bool(share50 - placebo_share50_q95 >= 0.05),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, default=EXP / "stand_geometry.csv")
    parser.add_argument("--output", type=Path, default=EXP / "position_check.csv")
    parser.add_argument("--manifest", type=Path, default=EXP / "position_check_manifest.json")
    args = parser.parse_args()

    data, input_hashes = load_population()
    maps = geometry_maps(args.geometry)
    input_hashes["stand_geometry.csv"] = sha256(args.geometry)
    results = []
    airports = list(AIRPORTS)
    for airport in airports:
        airport_rows = data.loc[data["ADEP_mvt"].eq(airport)]
        populations = {
            "all_adsb_positions": airport_rows,
            "with_clock": airport_rows.loc[airport_rows["with_clock"]],
        }
        for population, pop_rows in populations.items():
            for source in SOURCES:
                positions = maps.get(source, {}).get(airport, {})
                result = metrics(
                    pop_rows,
                    positions,
                    seed=int(hashlib.sha256(f"exp127:{source}:{population}:{airport}".encode()).hexdigest()[:8], 16),
                )
                if result.get("n_matched_rows", 0) == 0:
                    continue
                results.append({"airport": airport, "population": population, "source": source, **result})

    fields = [
        "airport", "population", "source", "n_adsb_rows", "n_matched_rows", "n_matched_stands", "row_coverage",
        "real_median_distance_m", "real_share_within_50m", "real_share_within_100m",
        "real_share50_minus_placebo_q95",
        "placebo_median_distance_m", "placebo_share_within_50m", "placebo_share_within_100m",
        "placebo_share50_q95", "placebo_share100_q95", "real_share50_above_placebo_q95",
        "real_share50_far_above_placebo",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    xplane_all = [r for r in results if r["source"] == "xplane_gateway" and r["population"] == "all_adsb_positions"]
    position_counts = data.groupby("ADEP_mvt").size().to_dict()
    record = {
        "experiment": "exp-127-clock-precision",
        "months": list(MONTHS),
        "input_hashes": input_hashes,
        "position_population": "rows with non-null exp-046 stand_lat/stand_lon and non-null PRC STAND_mvt; second stratum is obs_has_clock and not noac",
        "position_rows_by_airport": {airport: int(position_counts.get(airport, 0)) for airport in AIRPORTS},
        "airports_without_adsb_stand_positions": [airport for airport in AIRPORTS if position_counts.get(airport, 0) == 0],
        "source_populations": list(SOURCES),
        "placebo": f"{N_PERMUTATIONS} deterministic within-airport permutations of stand-to-coordinate assignments for the matched stand names; no coordinates move across airports",
        "far_above_rule": "real share within 50m is far above placebo when it is at least 5 percentage points above the 95th percentile of the placebo share",
        "xplane_airports_above_placebo_q95": [r["airport"] for r in xplane_all if r["real_share50_above_placebo_q95"]],
        "xplane_airports_far_above_placebo": [r["airport"] for r in xplane_all if r["real_share50_far_above_placebo"]],
        "xplane_airport_pass_count": sum(r["real_share50_far_above_placebo"] for r in xplane_all),
        "results": len(results),
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    for row in xplane_all:
        print(
            f"{row['airport']} all: n={row['n_matched_rows']}/{row['n_adsb_rows']} stands={row['n_matched_stands']} "
            f"median={row['real_median_distance_m']:.1f}m <=50={row['real_share_within_50m']:.3f} "
            f"placebo50={row['placebo_share_within_50m']:.3f} q95={row['placebo_share50_q95']:.3f} "
            f"far={row['real_share50_far_above_placebo']}"
        )


if __name__ == "__main__":
    main()
