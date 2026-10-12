#!/usr/bin/env python3
"""Prepare label-free geometry features in submitting.parquet order."""

from __future__ import annotations
import os

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAIN = Path(os.environ.get("PRC_ROOT", Path(__file__).resolve().parents[2])).resolve()
EXP = Path(__file__).resolve().parent
SUPPORT = EXP / "support"
DATASET = EXP / "kaggle/dataset"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
# exp-186 (2026-10-08, desk request b for finaldata): $PRC_RANKING_MODE=final reads the final files of
# prc/data.py (each accepted by its SHA-256) and the final stand table `stand_geometry-final.csv`, and writes
# a `-final` sibling of each output. The ranking mode keeps every path and value.
RANKING_MODE = os.environ.get("PRC_RANKING_MODE") or "ranking"
SFX = "" if RANKING_MODE == "ranking" else f"-{RANKING_MODE}"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    return str(path.relative_to(MAIN)) if path.is_relative_to(MAIN) else str(path)


def main() -> None:
    sys.path.insert(0, str(MAIN))
    if RANKING_MODE == "ranking":
        ranking_source = MAIN / "data/raw/ranking.parquet"
        template_source = MAIN / "data/raw/submitting.parquet"
    else:
        from prc import data as D
        ranking_source, template_source = D.ranking_file(RANKING_MODE), D.submitting_file(RANKING_MODE)
    prep = load_module("exp127_prepare_helpers", EXP / "prepare_fold_frames.py")
    serve = load_module(
        "exp127_saved_ranking_block",
        MAIN / "experiments/exp-046-surfacefit/code/serve.py",
    )
    # ranking_block writes a reach receipt. Direct that single output into this worktree;
    # its matcher, rule and input reads still come from the read-only main checkout.
    serve.EXP = EXP
    (EXP / "analysis").mkdir(parents=True, exist_ok=True)

    geometry_path = EXP / f"stand_geometry{SFX}.csv"
    geometry = pd.read_csv(
        geometry_path,
        dtype={"airport": str, "stand_name_mvt": str, "runway_end": str},
    )
    geometry["source_order"] = geometry["stand_source"].map(
        {"xplane_gateway": 0, "openstreetmap": 1}
    ).fillna(9)
    stand_rows = geometry.sort_values(["source_order", "stand_source", "runway_end"])
    stand_source = stand_rows.groupby(["airport", "stand_name_mvt"], as_index=False).first()
    stand_exact, stand_normalized = prep.unique_map(
        stand_source,
        "stand_name_mvt",
        ["stand_latitude", "stand_longitude", "stand_source"],
        "airport",
    )
    runway_rows = geometry[
        ["airport", "runway_end", "runway_end_latitude", "runway_end_longitude"]
    ].drop_duplicates()
    runway_exact, runway_normalized = prep.unique_map(
        runway_rows,
        "runway_end",
        ["runway_end_latitude", "runway_end_longitude"],
        "airport",
    )

    ranking, ranking_block = serve.ranking_block()
    required = {
        "MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
        "obs_has_clock", "stand_lat", "stand_lon",
    }
    missing = sorted(required.difference(ranking.columns))
    if missing:
        raise SystemExit(f"saved ranking block lacks required columns: {missing}")
    ranking["MVT_ID_mvt"] = ranking["MVT_ID_mvt"].astype("int64")
    if ranking["MVT_ID_mvt"].duplicated().any():
        raise SystemExit("saved ranking block has duplicate movement IDs")

    template = pd.read_parquet(template_source, columns=["MVT_ID_mvt"])
    template["MVT_ID_mvt"] = template["MVT_ID_mvt"].astype("int64")
    if template["MVT_ID_mvt"].duplicated().any():
        raise SystemExit("submitting template has duplicate movement IDs")
    if len(template) != len(ranking) or set(template.MVT_ID_mvt) != set(ranking.MVT_ID_mvt):
        raise SystemExit("ranking departures do not match the submitting template IDs")

    dep = template.reset_index(names="rank_order").merge(
        ranking[
            ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
             "obs_has_clock", "stand_lat", "stand_lon"]
        ],
        on="MVT_ID_mvt",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if dep["ADEP_mvt"].isna().any() or not dep["rank_order"].is_monotonic_increasing:
        raise SystemExit("ranking feature rows lost submitting template order")
    dep["airport"] = dep["ADEP_mvt"].astype(str)
    dep["noac"] = dep["AIRCRAFT_TYPE_mvt"].isna()
    dep["obs_has_clock"] = dep["obs_has_clock"].fillna(False).astype(bool)

    stands = [
        prep.resolve_name(str(airport), stand, stand_exact, stand_normalized)
        for airport, stand in zip(dep.airport, dep.STAND_mvt)
    ]
    dep["survey_stand_lat"] = [item[1][0] if item else np.nan for item in stands]
    dep["survey_stand_lon"] = [item[1][1] if item else np.nan for item in stands]
    dep["survey_source"] = [item[1][2] if item else None for item in stands]

    runway_points = []
    for airport, runway in zip(dep.airport, dep.RUNWAY_mvt):
        if pd.isna(runway):
            runway_points.append(None)
            continue
        apt = str(airport)
        point = runway_exact.get((apt, str(runway)))
        if point is None:
            normalized = runway_normalized.get((apt, prep.normalize_runway(runway)))
            point = normalized[1] if normalized is not None else None
        runway_points.append(point)
    dep["runway_lat"] = [point[0] if point else np.nan for point in runway_points]
    dep["runway_lon"] = [point[1] if point else np.nan for point in runway_points]

    survey_lat = dep["survey_stand_lat"].to_numpy("float64")
    survey_lon = dep["survey_stand_lon"].to_numpy("float64")
    observed_lat = dep["stand_lat"].to_numpy("float64")
    observed_lon = dep["stand_lon"].to_numpy("float64")
    runway_lat = dep["runway_lat"].to_numpy("float64")
    runway_lon = dep["runway_lon"].to_numpy("float64")
    stand_ok = np.isfinite(survey_lat) & np.isfinite(survey_lon)
    observed_ok = np.isfinite(observed_lat) & np.isfinite(observed_lon)
    runway_ok = np.isfinite(runway_lat) & np.isfinite(runway_lon)
    dep["survey_to_obs_stand_m"] = np.where(
        stand_ok & observed_ok,
        prep.haversine_km(survey_lat, survey_lon, observed_lat, observed_lon) * 1000.0,
        np.nan,
    )
    dep["survey_to_runway_km"] = np.where(
        stand_ok & runway_ok,
        prep.haversine_km(survey_lat, survey_lon, runway_lat, runway_lon),
        np.nan,
    )
    dep["survey_is_osm"] = dep["survey_source"].map(
        {"xplane_gateway": 0.0, "openstreetmap": 1.0}
    ).astype("float64")

    output_columns = [
        "rank_order", "MVT_ID_mvt", "airport", "noac", "obs_has_clock",
        "survey_to_obs_stand_m", "survey_to_runway_km", "survey_is_osm",
    ]
    out = dep[output_columns].copy()
    if not np.array_equal(out.rank_order.to_numpy(), np.arange(len(template))):
        raise SystemExit("ranking frame is not in submitting.parquet order")
    if not np.array_equal(out.MVT_ID_mvt.to_numpy(), template.MVT_ID_mvt.to_numpy()):
        raise SystemExit("ranking frame IDs do not match submitting.parquet order")
    if out.duplicated("MVT_ID_mvt").any() or not out.airport.isin(AIRPORTS).all():
        raise SystemExit("ranking frame has duplicate IDs or an unknown airport")

    DATASET.mkdir(parents=True, exist_ok=True)
    SUPPORT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATASET / f"ranking_frame{SFX}.parquet", index=False, compression="zstd")
    template.to_csv(DATASET / f"submitting_ids{SFX}.csv", index=False, lineterminator="\n")
    coverage = (
        out.groupby("airport", sort=True)
        .agg(rows=("MVT_ID_mvt", "size"),
             with_clock=("obs_has_clock", "sum"),
             observed_stand_offset=("survey_to_obs_stand_m", "count"),
             surveyed_runway=("survey_to_runway_km", "count"))
        .reset_index()
    )
    # Separate class counts avoid conflating no-aircraft-type rows with the clock class.
    usable = out.loc[out.obs_has_clock & ~out.noac].groupby("airport").size()
    coverage["with_clock_not_noac"] = coverage.airport.map(usable).fillna(0).astype("int64")
    coverage.to_csv(SUPPORT / f"ranking_feature_coverage{SFX}.csv", index=False, lineterminator="\n")
    summary_sources = {}
    for month, group in ranking.groupby(ranking.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")):
        days = sorted(group.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique().tolist())
        summary_sources[month] = [
            {"airport": airport, "day": day, "path": str(path.relative_to(MAIN)),
             "bytes": int(path.stat().st_size), "sha256": prep.sha256(path)}
            for airport, day, path in serve.B.present_days(list(AIRPORTS), days)
        ]
    rule_record = ranking_block.attrs.get("rule_cover", {})
    record = {
        "rows": int(len(out)),
        "ranking_source": {
            "path": rel(ranking_source),
            "sha256": prep.sha256(ranking_source),
        },
        "submitting_template": {
            "path": rel(template_source),
            "sha256": prep.sha256(template_source),
        },
        "geometry_source": {
            "path": f"experiments/exp-127-clock-precision/stand_geometry{SFX}.csv",
            "sha256": prep.sha256(geometry_path),
        },
        "serving_code": {
            str(path.relative_to(MAIN)): prep.sha256(path)
            for path in (
                MAIN / "experiments/exp-046-surfacefit/code/serve.py",
                MAIN / "experiments/exp-046-surfacefit/code/block.py",
                MAIN / "experiments/exp-046-surfacefit/code/arms.py",
            )
        },
        "serving_rule": rule_record,
        "ranking_block_by_month": ranking_block.attrs.get("built_per_month", {}),
        "summary_files_read_by_month": summary_sources,
        "feature_frame_sha256": prep.sha256(DATASET / f"ranking_frame{SFX}.parquet"),
        "submitting_ids_sha256": prep.sha256(DATASET / f"submitting_ids{SFX}.csv"),
        "feature_builder_sha256": prep.sha256(Path(__file__)),
    }
    (SUPPORT / f"ranking_input_manifest{SFX}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({
        **{k: record[k] for k in ("rows", "ranking_source", "submitting_template",
                                  "serving_rule")},
        "ranking_feature_sha256": record["feature_frame_sha256"],
        "submission_ids_sha256": record["submitting_ids_sha256"],
        "summary_file_counts": {m: len(files) for m, files in summary_sources.items()},
        "coverage": coverage.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
