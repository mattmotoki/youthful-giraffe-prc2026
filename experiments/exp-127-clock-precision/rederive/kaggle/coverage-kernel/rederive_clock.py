#!/usr/bin/env python3
"""Read the exp-045 point outputs and measure surveyed-stand clock events."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Frozen from exp-043/045 so this one-file Kaggle script does not depend on
# auxiliary files that Kaggle's script runner does not mount.
GAP = 1800.0
MOVE_KT = 3.0
HOLD_S = 30.0
STAND_STOP = 60.0
TURNAROUND = 600.0
DISP_M = 250.0


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
MONTHS = ("2025-01", "2025-07", "2025-12", "2026-01", "2026-07")
FOLD_MONTHS = {"2025-07", "2025-12", "2025-01"}
RANK_MONTHS = {"2026-01", "2026-07"}
RADIUSES_M = (30, 60, 100)
LFPG = "LFPG"
MATCH_TOLERANCE_S = 0.51
PUSHBACK_SPEED_KT = 3.0
POINT_COLUMNS = [
    "airport", "icao", "t", "lat", "lon", "gs_kt", "track", "alt_ft",
    "on_ground", "first_airborne", "in_box", "callsign",
]


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if not len(mask):
        return []
    cut = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    edges = [0, *cut.tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


def _moving(gs: np.ndarray, lat: np.ndarray, lon: np.ndarray, t: np.ndarray) -> np.ndarray:
    mv = gs > MOVE_KT
    miss = np.isnan(gs)
    if miss.any() and len(t) > 1:
        dx = np.gradient(lon) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
        dy = np.gradient(lat) * 110540.0
        dt = np.gradient(t)
        with np.errstate(invalid="ignore", divide="ignore"):
            spd = np.hypot(dx, dy) / np.where(dt == 0, np.nan, dt) / 0.5144
        mv[miss] = spd[miss] > MOVE_KT
    return np.nan_to_num(mv, nan=0.0).astype(bool)


def _span(t: np.ndarray, a: int, b: int) -> float:
    if b - a >= 2:
        return float(t[b - 1] - t[a])
    return float(t[b] - t[a]) if b < len(t) else 0.0


def _segment_row(ts: np.ndarray, gss: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 mv: np.ndarray, runs: list[tuple[int, int, bool]]) -> dict:
    sustained = [(a, b) for a, b, is_moving in runs
                 if is_moving and _span(ts, a, b) >= HOLD_S]
    turns = [b for a, b, is_moving in runs
             if not is_moving and _span(ts, a, b) >= TURNAROUND]
    anchor = turns[-1] - 1 if turns else 0
    after = [a for a, _ in sustained if a >= anchor]
    strict = loose = disp = margin = np.nan
    stand_lat = stand_lon = np.nan
    if after:
        k0 = after[0]
        loose = ts[k0]
        prev = [(a, b) for a, b, is_moving in runs if not is_moving and b <= k0]
        margin = _span(ts, *prev[-1]) if prev else 0.0
        if margin >= STAND_STOP:
            strict = loose
        st_pts = [k for k in range(anchor, k0) if not mv[k]]
        ref = st_pts if st_pts else [anchor]
        stand_lat = float(np.median(lat[ref]))
        stand_lon = float(np.median(lon[ref]))
        dx = (lon - stand_lon) * 111320.0 * np.cos(np.radians(stand_lat))
        dy = (lat - stand_lat) * 110540.0
        far = np.hypot(dx, dy) > DISP_M
        far[:anchor] = False
        out_idx = np.flatnonzero(far)
        if len(out_idx):
            back = np.flatnonzero(~far[out_idx[0]:])
            k1 = out_idx[0]
            if len(back):
                later = np.flatnonzero(far[out_idx[0] + back[-1]:])
                if len(later):
                    k1 = out_idx[0] + back[-1] + later[0]
            disp = ts[k1]
    return {
        "obs_offblock_strict": strict,
        "obs_offblock_loose": loose,
        "obs_offblock_disp": disp,
        "stand_stop_s": margin,
        "turnaround_anchored": bool(turns),
        "stand_lat": stand_lat,
        "stand_lon": stand_lon,
    }


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_input(pattern: str) -> Path:
    matches = glob.glob(str(INPUT / "**" / pattern), recursive=True)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {len(matches)}")
    return Path(matches[0])


def haversine_m(lat: np.ndarray, lon: np.ndarray, stand_lat: float, stand_lon: float) -> np.ndarray:
    earth_m = 6_371_008.8
    p1 = np.radians(lat)
    p2 = np.radians(stand_lat)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon, dtype="float64") - stand_lon)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * earth_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def departure_anchor(ts: np.ndarray, moving: np.ndarray, runs: list[tuple[int, int, bool]]) -> int:
    turns = [b for a, b, is_moving in runs
             if not is_moving and _span(ts, a, b) >= TURNAROUND]
    return turns[-1] - 1 if turns else 0


def persistent_exit(distances: np.ndarray, radius_m: float) -> int | None:
    """First observed point outside after the last observed point inside."""
    valid = np.isfinite(distances)
    inside = np.flatnonzero(valid & (distances <= radius_m))
    if not len(inside):
        return None
    last_inside = int(inside[-1])
    after = np.flatnonzero(valid[last_inside + 1:] & (distances[last_inside + 1:] > radius_m))
    if not len(after):
        return None
    return last_inside + 1 + int(after[0])


def clock_features(
    ts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    speed: np.ndarray,
    track: np.ndarray,
    anchor: int,
    current_250m_s: float,
    stand_lat: float,
    stand_lon: float,
    sham_lat: float,
    sham_lon: float,
) -> dict:
    out: dict[str, float] = {}
    if anchor >= len(ts):
        return out
    out["first_ground_time_s"] = float(ts[anchor])
    first_speed = float(speed[anchor]) if np.isfinite(speed[anchor]) else np.nan
    use_fallback = not np.isfinite(first_speed) or first_speed < 1.0
    effective_speed = PUSHBACK_SPEED_KT if use_fallback else first_speed
    for name, ref_lat, ref_lon in (
        ("survey", stand_lat, stand_lon),
        ("sham", sham_lat, sham_lon),
    ):
        if not (np.isfinite(ref_lat) and np.isfinite(ref_lon)):
            continue
        distance = haversine_m(lat[anchor:], lon[anchor:], ref_lat, ref_lon)
        if name == "survey":
            out["first_ground_distance_m"] = float(distance[0])
            out["first_ground_to_current250_s"] = float(current_250m_s - ts[anchor])
            implied_offblock = float(ts[anchor] - distance[0] / (effective_speed * 0.514444))
            out["first_ground_speed_kt"] = first_speed
            out["first_ground_effective_speed_kt"] = float(effective_speed)
            out["first_ground_speed_fallback"] = float(use_fallback)
            out["implied_offblock_s"] = implied_offblock
            out["implied_offblock_to_current250_s"] = float(current_250m_s - implied_offblock)
        else:
            out["sham_first_ground_distance_m"] = float(distance[0])
            out["sham_first_ground_to_current250_s"] = float(current_250m_s - ts[anchor])
            out["sham_implied_offblock_s"] = float(
                ts[anchor] - distance[0] / (effective_speed * 0.514444)
            )
            out["sham_implied_offblock_to_current250_s"] = float(
                current_250m_s - out["sham_implied_offblock_s"]
            )
        for radius in RADIUSES_M:
            idx = persistent_exit(distance, radius)
            prefix = "exit" if name == "survey" else "sham_exit"
            if idx is None:
                continue
            point_idx = anchor + idx
            event_time = float(ts[point_idx])
            out[f"{prefix}{radius}_time_s"] = event_time
            out[f"{prefix}{radius}_gs_kt"] = float(speed[point_idx]) if np.isfinite(speed[point_idx]) else np.nan
            out[f"{prefix}{radius}_track_deg"] = float(track[point_idx]) if np.isfinite(track[point_idx]) else np.nan
            if name == "survey":
                out[f"exit{radius}_to_current250_s"] = float(current_250m_s - event_time)
            else:
                out[f"sham_exit{radius}_to_current250_s"] = float(current_250m_s - event_time)
    return out


def canonical_icao(value: object) -> str:
    return str(value).strip().lower()


def validate_current_time_scale(keys: pd.DataFrame) -> None:
    valid = keys.loc[keys.current_250m_s.notna(), ["month", "current_250m_s"]].copy()
    if valid.empty:
        raise RuntimeError("no current 250 m clock times are available")
    valid["current_250m_s"] = pd.to_numeric(valid.current_250m_s, errors="coerce")
    for month, rows in valid.groupby("month", sort=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        if not rows.current_250m_s.between(
            start.timestamp() - 86400.0, end.timestamp() + 86400.0, inclusive="both"
        ).all():
            raise RuntimeError(f"{month}: current 250 m times are not Unix seconds for that month")


def build_targets(keys: pd.DataFrame):
    eligible = (
        keys.current_has_clock.astype(bool)
        & ~keys.noac.astype(bool)
        & keys.airport.ne(LFPG)
        & keys.survey_stand_lat.notna()
        & keys.survey_stand_lon.notna()
        & keys.obs_icao.notna()
        & keys.current_250m_s.notna()
    )
    target_rows = {}
    lookup: dict[tuple[str, str], list[tuple[float, int]]] = {}
    airport_icaos: dict[str, set[str]] = {}
    for index, row in keys.loc[eligible].iterrows():
        airport = str(row.airport)
        icao = canonical_icao(row.obs_icao)
        if not icao or icao == "nan":
            continue
        target_rows[int(index)] = row
        pair = (airport, icao)
        lookup.setdefault(pair, []).append((float(row.current_250m_s), int(index)))
        airport_icaos.setdefault(airport, set()).add(icao)
    lookup_arrays = {}
    for pair, items in lookup.items():
        items.sort()
        lookup_arrays[pair] = (
            np.asarray([item[0] for item in items], dtype="float64"),
            np.asarray([item[1] for item in items], dtype="int64"),
        )
    return target_rows, lookup_arrays, airport_icaos


def match_target(
    airport: str,
    icao: str,
    onset_s: float,
    wheels_off_s: float,
    lookup: dict,
) -> tuple[int | None, list[int]]:
    pair = (airport, icao)
    entry = lookup.get(pair)
    if entry is None or not np.isfinite(onset_s):
        return None, []
    times, indices = entry
    at = int(np.searchsorted(times, onset_s))
    window = range(max(0, at - 2), min(len(times), at + 2))
    candidates = []
    for j in window:
        error = abs(float(times[j]) - onset_s)
        if error > MATCH_TOLERANCE_S:
            continue
        row_index = int(indices[j])
        row = TARGET_ROWS[row_index]
        expected_wheel = float(row.current_wheels_off_s)
        if np.isfinite(expected_wheel) != np.isfinite(wheels_off_s):
            continue
        if np.isfinite(expected_wheel) and abs(expected_wheel - wheels_off_s) > MATCH_TOLERANCE_S:
            continue
        candidates.append(row_index)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) > 1:
        return None, candidates
    return None, []


def segment_rows(points: pd.DataFrame):
    """Yield the exact point slices and summary used by exp-045 allsegments."""
    ordered = points.sort_values("t", kind="stable")
    for (airport, icao), aircraft in ordered.groupby(["airport", "icao"], sort=False, observed=True):
        aircraft = aircraft.reset_index(drop=True)
        t = aircraft.t.to_numpy("float64")
        ground = aircraft.on_ground.to_numpy(bool)
        first_airborne = aircraft.first_airborne.to_numpy(bool)
        alt_ok = aircraft.alt_ft.notna().to_numpy()
        speed = aircraft.gs_kt.to_numpy("float64")
        lat = aircraft.lat.to_numpy("float64")
        lon = aircraft.lon.to_numpy("float64")
        track = aircraft.track.to_numpy("float64")
        ground_indices = np.flatnonzero(ground)
        if not len(ground_indices):
            continue
        breaks = np.flatnonzero(
            (np.diff(ground_indices) > 1) | (np.diff(t[ground_indices]) > GAP)
        ) + 1
        for a, b in zip([0, *breaks.tolist()], [*breaks.tolist(), len(ground_indices)]):
            i, j = int(ground_indices[a]), int(ground_indices[b - 1])
            sl = slice(i, j + 1)
            ts, gss = t[sl], speed[sl]
            la, lo = lat[sl], lon[sl]
            moving = _moving(gss, la, lo, ts)
            runs = _runs(moving)
            summary = _segment_row(ts, gss, la, lo, moving, runs)
            next_is_air = bool(j + 1 < len(t) and first_airborne[j + 1] and alt_ok[j + 1])
            starts_fast = bool(moving[0] and np.isfinite(gss[0]) and gss[0] > 30.0)
            ends_stationary = bool(not moving[-1])
            kind = "departure" if next_is_air else "arrival" if starts_fast and ends_stationary else "other"
            yield {
                "airport": str(airport),
                "icao": canonical_icao(icao),
                "kind": kind,
                "onset_s": float(summary["obs_offblock_disp"]),
                "wheels_off_s": float(ts[-1]) if next_is_air else np.nan,
                "ts": ts,
                "lat": la,
                "lon": lo,
                "speed": gss,
                "track": track[sl],
                "moving": moving,
                "runs": runs,
                "anchor": departure_anchor(ts, moving, runs),
            }


def output_template(keys: pd.DataFrame) -> pd.DataFrame:
    out = keys[["MVT_ID_mvt", "scope", "fold", "month", "airport", "day", "rank_order",
                "current_has_clock", "noac", "survey_stand_lat", "survey_stand_lon", "survey_source",
                "sham_stand_lat", "sham_stand_lon", "sham_source", "current_250m_s"]].copy()
    for name in (
        "first_ground_time_s", "first_ground_distance_m", "first_ground_to_current250_s",
        "first_ground_speed_kt", "first_ground_effective_speed_kt", "first_ground_speed_fallback",
        "implied_offblock_s", "implied_offblock_to_current250_s",
        "sham_first_ground_distance_m", "sham_first_ground_to_current250_s",
        "sham_implied_offblock_s", "sham_implied_offblock_to_current250_s",
    ):
        out[name] = np.nan
    for prefix in ("exit", "sham_exit"):
        for radius in RADIUSES_M:
            out[f"{prefix}{radius}_time_s"] = np.nan
            out[f"{prefix}{radius}_gs_kt"] = np.nan
            out[f"{prefix}{radius}_track_deg"] = np.nan
            out[f"{prefix}{radius}_to_current250_s"] = np.nan
    out["match_status"] = np.where(
        ~out.current_has_clock.astype(bool), "no_current_clock",
        np.where(out.noac.astype(bool), "no_aircraft_type",
                 np.where(out.airport.eq(LFPG), "excluded_lfpg",
                          np.where(out.survey_stand_lat.isna(), "no_surveyed_position", "no_segment_match"))),
    )
    out["matched_segment_day"] = None
    out["matched_segment_start_s"] = np.nan
    out["matched_segment_end_s"] = np.nan
    out["matched_segment_points"] = np.nan
    out["segment_match_error_s"] = np.nan
    return out


def summarize(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    eligible = features.current_has_clock.astype(bool) & ~features.noac.astype(bool)
    records = []
    for (scope, fold, month, airport), group in features.loc[eligible].groupby(
        ["scope", "fold", "month", "airport"], dropna=False, sort=True
    ):
        excluded = str(airport) == LFPG
        n = int(len(group))
        record = {
            "scope": scope,
            "fold": fold if pd.notna(fold) else "",
            "month": month,
            "airport": airport,
            "out_of_scope": excluded,
            "with_clock_rows": n,
            "survey_position_rows": int(group.survey_stand_lat.notna().sum()),
            "matched_rows": int(group.match_status.eq("matched").sum()),
            "ambiguous_rows": int(group.match_status.eq("ambiguous_multiple_segments").sum()),
            "current_250m_rows": n,
            "current_250m_share": 1.0 if n else np.nan,
        }
        for col, label in (
            ("first_ground_time_s", "first_ground_time"),
            ("implied_offblock_s", "implied_offblock"),
            ("exit30_time_s", "exit30_time"),
            ("exit60_time_s", "exit60_time"),
            ("exit100_time_s", "exit100_time"),
        ):
            count = int(group[col].notna().sum())
            record[f"{label}_rows"] = count
            record[f"{label}_share"] = count / n if n else np.nan
        for radius in RADIUSES_M:
            delay = pd.to_numeric(group[f"exit{radius}_to_current250_s"], errors="coerce").dropna()
            record[f"exit{radius}_to_current250_median_s"] = float(delay.median()) if len(delay) else np.nan
            record[f"exit{radius}_to_current250_p10_s"] = float(delay.quantile(0.10)) if len(delay) else np.nan
            record[f"exit{radius}_to_current250_p90_s"] = float(delay.quantile(0.90)) if len(delay) else np.nan
        delay = pd.to_numeric(group["first_ground_to_current250_s"], errors="coerce").dropna()
        record["first_ground_to_current250_median_s"] = float(delay.median()) if len(delay) else np.nan
        record["first_ground_to_current250_p10_s"] = float(delay.quantile(0.10)) if len(delay) else np.nan
        record["first_ground_to_current250_p90_s"] = float(delay.quantile(0.90)) if len(delay) else np.nan
        lead = pd.to_numeric(group["implied_offblock_to_current250_s"], errors="coerce").dropna()
        record["implied_offblock_lead_median_s"] = float(lead.median()) if len(lead) else np.nan
        record["implied_offblock_lead_p10_s"] = float(lead.quantile(0.10)) if len(lead) else np.nan
        record["implied_offblock_lead_p90_s"] = float(lead.quantile(0.90)) if len(lead) else np.nan
        record["implied_before_current250_share"] = float((lead >= 0).mean()) if len(lead) else np.nan
        speed = pd.to_numeric(group["first_ground_speed_kt"], errors="coerce").dropna()
        record["first_ground_speed_median_kt"] = float(speed.median()) if len(speed) else np.nan
        record["first_ground_speed_fallback_rows"] = int(group["first_ground_speed_fallback"].fillna(0).sum())
        record["first_ground_speed_fallback_share"] = (
            float(group["first_ground_speed_fallback"].fillna(0).mean()) if n else np.nan
        )
        records.append(record)
    table = pd.DataFrame(records)
    fold_rows = table.loc[table.scope.eq("fold") & ~table.out_of_scope]
    time_cols = ["first_ground_time", "implied_offblock", "exit30_time", "exit60_time", "exit100_time"]
    by_fold = {}
    for fold, rows in fold_rows.groupby("fold", sort=True):
        denominator = int(rows.with_clock_rows.sum())
        by_fold[str(fold)] = {col: float(rows[f"{col}_rows"].sum() / denominator) for col in time_cols}
    minimum = min((values["implied_offblock"] for values in by_fold.values()), default=0.0)
    summary = {
        "coverage_denominator": "with-clock fold rows without missing aircraft type, excluding LFPG by assignment",
        "excluded_airport": LFPG,
        "per_fold_coverage": by_fold,
        "minimum_implied_offblock_coverage": minimum,
        "coverage_gate_pass": bool(by_fold and minimum >= 0.5),
        "coverage_gate": "implied off-block time exists on at least half of in-scope with-clock rows in every fold",
        "radius_exit_gate_from_first_scan": "not used for this repair; the 30/60/100 m exits remain label-free diagnostics",
    }
    return table, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    keys_path = unique_input("clock_keys.parquet")
    expected_path = unique_input("points_expected.json")
    keys = pd.read_parquet(keys_path)
    expected = json.loads(expected_path.read_text())
    if "y_true" in keys.columns or "r_v22" in keys.columns or "target" in keys.columns:
        raise RuntimeError("label-free coverage input includes a target")
    validate_current_time_scale(keys)
    if keys.duplicated(["scope", "MVT_ID_mvt"]).any():
        raise RuntimeError("input key rows are not unique")
    global TARGET_ROWS
    TARGET_ROWS, lookup, airport_icaos = build_targets(keys)
    features = output_template(keys)

    expected_days = {
        month: {record["day"]: record for record in expected["months"][month]}
        for month in MONTHS
    }
    point_files = glob.glob(str(INPUT / "**" / "points_*.parquet"), recursive=True)
    files_by_month: dict[str, dict[str, str]] = {month: {} for month in MONTHS}
    for path in point_files:
        name = os.path.basename(path)
        day = name[7:17]
        month = day[:7]
        if month not in files_by_month:
            continue
        if day not in expected_days[month]:
            raise RuntimeError(f"unexpected exp-045 point day {day}: {path}")
        if day in files_by_month[month]:
            raise RuntimeError(f"duplicate exp-045 point output for {day}")
        files_by_month[month][day] = path
    for month in MONTHS:
        missing = sorted(set(expected_days[month]) - set(files_by_month[month]))
        if missing:
            raise RuntimeError(f"{month}: missing {len(missing)} expected point files: {missing[:4]}")

    matches: dict[int, tuple[dict, tuple]] = {}
    ambiguous_ids: set[int] = set()
    stats = {"point_files": [], "point_rows": 0, "target_airport_icao_pairs": 0,
             "segments_seen_after_icao_filter": 0, "segments_with_current_onset": 0,
             "arrival_segments_skipped": 0, "segments_without_a_match": 0,
             "segments_with_multiple_row_matches": 0}
    for month in MONTHS:
        for day, path in sorted(files_by_month[month].items()):
            started = time.time()
            expected_record = expected_days[month][day]
            if os.path.getsize(path) != int(expected_record["expected_bytes"]):
                raise RuntimeError(f"{day}: point file byte size does not match exp-045 manifest")
            source_sha = sha256(path)
            if source_sha != expected_record["sha256"]:
                raise RuntimeError(f"{day}: mounted point file SHA-256 does not match exp-045 manifest")
            raw = pd.read_parquet(path, columns=POINT_COLUMNS)
            if len(raw) != int(expected_record["expected_rows"]):
                raise RuntimeError(f"{day}: {len(raw)} rows, expected {expected_record['expected_rows']}")
            stats["point_rows"] += int(len(raw))
            relevant_airports = airport_icaos
            raw = raw.loc[raw.airport.astype(str).isin(relevant_airports)].copy()
            if len(raw):
                keep = np.zeros(len(raw), dtype=bool)
                airport_values = raw.airport.astype(str).to_numpy()
                for airport, icaos in relevant_airports.items():
                    idx = np.flatnonzero(airport_values == airport)
                    if len(idx):
                        vals = raw.iloc[idx].icao.astype(str).str.strip().str.lower()
                        keep[idx] = vals.isin(icaos).to_numpy()
                raw = raw.loc[keep]
            for segment in segment_rows(raw):
                stats["segments_seen_after_icao_filter"] += 1
                if segment["kind"] == "arrival":
                    stats["arrival_segments_skipped"] += 1
                    continue
                if not np.isfinite(segment["onset_s"]):
                    continue
                stats["segments_with_current_onset"] += 1
                row_index, ambiguous = match_target(
                    segment["airport"], segment["icao"], segment["onset_s"],
                    segment["wheels_off_s"], lookup,
                )
                if ambiguous:
                    stats["segments_with_multiple_row_matches"] += 1
                    ambiguous_ids.update(ambiguous)
                    continue
                if row_index is None:
                    stats["segments_without_a_match"] += 1
                    continue
                row = TARGET_ROWS[row_index]
                coords = [
                    float(row.survey_stand_lat), float(row.survey_stand_lon),
                    float(row.sham_stand_lat), float(row.sham_stand_lon),
                ]
                sampled = clock_features(
                    segment["ts"], segment["lat"], segment["lon"], segment["speed"],
                    segment["track"], segment["anchor"], float(row.current_250m_s),
                    *coords,
                )
                signature = (
                    day, float(segment["ts"][0]), float(segment["ts"][-1]),
                    int(len(segment["ts"])), float(segment["onset_s"]),
                    float(segment["wheels_off_s"]) if np.isfinite(segment["wheels_off_s"]) else None,
                )
                if row_index in matches:
                    if matches[row_index][1] != signature:
                        ambiguous_ids.add(row_index)
                    continue
                matches[row_index] = (sampled, signature)
            stats["point_files"].append({
                "day": day,
                "kernel_slug": expected_record["kernel_slug"],
                "kernel_version": int(expected_record["kernel_version"]),
                "kernel_last_run_utc": expected_record["kernel_last_run_utc"],
                "bytes": int(os.path.getsize(path)),
                "rows": int(expected_record["expected_rows"]),
                "source_sha256": source_sha,
                "seconds": round(time.time() - started, 2),
            })
            print(f"{day}: {len(raw)} filtered points; {time.time() - started:.1f} s", flush=True)

    matched_records = []
    for row_index, (sampled, signature) in matches.items():
        if row_index in ambiguous_ids:
            continue
        record = dict(sampled)
        record.update({
            "matched_segment_day": signature[0],
            "matched_segment_start_s": signature[1],
            "matched_segment_end_s": signature[2],
            "matched_segment_points": signature[3],
            "segment_match_error_s": signature[4] - float(TARGET_ROWS[row_index].current_250m_s),
        })
        matched_records.append((row_index, record))
    if matched_records:
        metrics = pd.DataFrame([row for _, row in matched_records],
                               index=[index for index, _ in matched_records])
        for column in metrics.columns:
            features.loc[metrics.index, column] = metrics[column]
        features.loc[metrics.index, "match_status"] = "matched"
    matched_ids = len(matched_records)
    for row_index in ambiguous_ids:
        features.at[row_index, "match_status"] = "ambiguous_multiple_segments"
        derived_columns = [
            "first_ground_time_s", "first_ground_distance_m", "first_ground_to_current250_s",
            "sham_first_ground_distance_m", "sham_first_ground_to_current250_s",
            "matched_segment_start_s", "matched_segment_end_s", "matched_segment_points",
            "segment_match_error_s",
        ]
        for prefix in ("exit", "sham_exit"):
            for radius in RADIUSES_M:
                derived_columns.extend([
                    f"{prefix}{radius}_time_s", f"{prefix}{radius}_gs_kt",
                    f"{prefix}{radius}_track_deg", f"{prefix}{radius}_to_current250_s",
                ])
        features.loc[row_index, derived_columns] = np.nan
        features.at[row_index, "matched_segment_day"] = None
    stats["target_airport_icao_pairs"] = int(len(lookup))
    stats["matched_target_rows"] = matched_ids
    stats["ambiguous_target_rows"] = int(len(ambiguous_ids))
    table, summary = summarize(features)
    features_path = OUTPUT / "clock_features.parquet"
    coverage_path = OUTPUT / "coverage_by_airport.csv"
    summary_path = OUTPUT / "coverage_summary.json"
    features.to_parquet(features_path, index=False, compression="zstd")
    table.to_csv(coverage_path, index=False, lineterminator="\n")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    receipt = {
        "experiment": "exp-127b-clock-rederive-coverage",
        "kernel_input_dataset": str(keys_path.parent),
        "point_manifest_sha256": sha256(expected_path),
        "point_source_manifest_sha256": expected["source_manifest_sha256"],
        "point_sources": [
            {"kernel_slug": slug, **version}
            for slug, version in sorted(expected["source_kernel_versions"].items())
        ],
        "point_files_read": len(stats["point_files"]),
        "expected_point_files": 155,
        "point_rows_read": stats["point_rows"],
        "target_airport_icao_pairs": stats["target_airport_icao_pairs"],
        "segments_seen_after_icao_filter": stats["segments_seen_after_icao_filter"],
        "segments_with_current_onset": stats["segments_with_current_onset"],
        "arrival_segments_skipped": stats["arrival_segments_skipped"],
        "segments_without_a_match": stats["segments_without_a_match"],
        "segments_with_multiple_row_matches": stats["segments_with_multiple_row_matches"],
        "matched_target_rows": stats["matched_target_rows"],
        "ambiguous_target_rows": stats["ambiguous_target_rows"],
        "coverage": summary,
        "point_file_sources": stats["point_files"],
        "outputs": {
            features_path.name: {"bytes": features_path.stat().st_size, "sha256": sha256(features_path)},
            coverage_path.name: {"bytes": coverage_path.stat().st_size, "sha256": sha256(coverage_path)},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
        },
    }
    receipt_path = OUTPUT / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "point_files_read": receipt["point_files_read"],
        "point_rows_read": receipt["point_rows_read"],
        "matched_target_rows": receipt["matched_target_rows"],
        "ambiguous_target_rows": receipt["ambiguous_target_rows"],
        "coverage": summary,
        "outputs": receipt["outputs"],
        "receipt_sha256": sha256(receipt_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    # The look-up is read only by `match_target`; assigned here once to avoid copying
    # a large table into every candidate function call.
    TARGET_ROWS = {}
    main()
