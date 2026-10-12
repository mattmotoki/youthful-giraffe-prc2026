#!/usr/bin/env python3
"""Build the frozen exp-134 row-level input proxies from private Kaggle inputs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("/kaggle/input")
ID = "MVT_ID_mvt"
POINT_COLUMNS = [
    "airport", "icao", "t", "lat", "lon", "gs_kt", "alt_ft",
    "on_ground", "first_airborne", "callsign",
]
PROXY_BASES = (
    "arrival_mean", "arrival_ratio", "clock_current250", "clock_implied",
    "spacing_short_share", "departure_count_q95", "filed_delay_mean",
)
LAGS = (-1, 0, 1)
WIDE_LAGS = (-1, 0, 1)
T2_LAGS = (-2, 2)
Z_COLS = [f"{name}_l{lag:+d}" for name in PROXY_BASES for lag in LAGS]
T2_COLS = [f"{name}_l{lag:+d}" for name in PROXY_BASES for lag in T2_LAGS]
WIDE_FAMILIES = ("arrival_mean", "arrival_ratio", "departure_count_q95",
                 "spacing_short_share")
WIDE_SCOPES = (("runway", 15), ("runway", 60), ("airport", 15),
               ("airport", 30), ("airport", 60))
WIDE_COLS = [f"{family}_{scope}_w{width}_l{lag:+d}"
             for scope, width in WIDE_SCOPES for family in WIDE_FAMILIES
             for lag in WIDE_LAGS]
EXTRA_COLS = [*T2_COLS, *WIDE_COLS]
ALL_COLS = [*Z_COLS, *EXTRA_COLS]
if (len(Z_COLS), len(T2_COLS), len(WIDE_COLS), len(ALL_COLS)) != (21, 14, 60, 95):
    raise RuntimeError("exp-136 proxy feature counts differ from the frozen design")
RAW_DEP_COLUMNS = [
    ID, "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
    "EOBT_1_flt", "FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
]
RAW_ARR_COLUMNS = [
    ID, "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
    "TAXITIME_SEC_mvt",
]
KEY = ["airport", "runway", "day", "slot"]
SOURCE_KEY = ["airport", "runway", "day", "source_slot"]
ARR_KEY = ["airport", "day", "source_slot"]
FOLD_SEED_INDEX = {"2025-07": 1, "2025-12": 2, "2025-01": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(name: str) -> Path:
    if not INPUT.exists():
        raise RuntimeError(f"Kaggle input root {INPUT} does not exist")
    paths = sorted(path for path in INPUT.glob(f"**/{name}") if path.is_file())
    if len(paths) != 1:
        roots = sorted(path.name for path in INPUT.iterdir())
        raise RuntimeError(f"expected one attached {name}, found {len(paths)}; "
                           f"input mounts={roots}, matches={[str(path) for path in paths]}")
    return paths[0]


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def category(value: object) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    return str(value).strip().upper() or "<NA>"


def epoch_seconds(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    seconds = parsed.astype("float64") / 1e9
    seconds[parsed == np.iinfo("int64").min] = np.nan
    return seconds


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if not len(mask):
        return []
    cuts = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    edges = [0, *cuts.tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


def _moving(gs: np.ndarray, lat: np.ndarray, lon: np.ndarray,
            times: np.ndarray) -> np.ndarray:
    moving = gs > 3.0
    missing = np.isnan(gs)
    if missing.any() and len(times) > 1:
        dx = np.gradient(lon) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
        dy = np.gradient(lat) * 110540.0
        dt = np.gradient(times)
        with np.errstate(invalid="ignore", divide="ignore"):
            speed = np.hypot(dx, dy) / np.where(dt == 0, np.nan, dt) / 0.5144
        moving[missing] = speed[missing] > 3.0
    return np.nan_to_num(moving, nan=0.0).astype(bool)


def _span(times: np.ndarray, start: int, stop: int) -> float:
    if stop - start >= 2:
        return float(times[stop - 1] - times[start])
    return float(times[stop] - times[start]) if stop < len(times) else 0.0


def _persistent_exit(distances: np.ndarray, radius_m: float = 250.0) -> int | None:
    valid = np.isfinite(distances)
    inside = np.flatnonzero(valid & (distances <= radius_m))
    if not len(inside):
        return None
    last_inside = int(inside[-1])
    after = np.flatnonzero(valid[last_inside + 1:] &
                           (distances[last_inside + 1:] > radius_m))
    if not len(after):
        return None
    return last_inside + 1 + int(after[0])


def _haversine_m(lat: float, lon: float, ref_lat: float, ref_lon: float) -> float:
    earth_m = 6_371_008.8
    p1, p2 = np.radians(lat), np.radians(ref_lat)
    dp = p2 - p1
    dl = np.radians(lon - ref_lon)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * earth_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def departure_segments(points: pd.DataFrame):
    """Yield the stable exp-127 ground segments that end at a take-off."""
    ordered = points.sort_values("t", kind="stable")
    for (airport, icao), aircraft in ordered.groupby(
        ["airport", "icao"], sort=False, observed=True
    ):
        aircraft = aircraft.reset_index(drop=True)
        times = aircraft.t.to_numpy("float64")
        ground = aircraft.on_ground.to_numpy(bool)
        first_airborne = aircraft.first_airborne.to_numpy(bool)
        altitude_ok = aircraft.alt_ft.notna().to_numpy()
        speed = aircraft.gs_kt.to_numpy("float64")
        lat = aircraft.lat.to_numpy("float64")
        lon = aircraft.lon.to_numpy("float64")
        ground_indices = np.flatnonzero(ground)
        if not len(ground_indices):
            continue
        breaks = np.flatnonzero(
            (np.diff(ground_indices) > 1)
            | (np.diff(times[ground_indices]) > 1800.0)
        ) + 1
        for a, b in zip([0, *breaks.tolist()], [*breaks.tolist(), len(ground_indices)]):
            first, last = int(ground_indices[a]), int(ground_indices[b - 1])
            if last + 1 >= len(aircraft) or not first_airborne[last + 1] or not altitude_ok[last + 1]:
                continue
            span = slice(first, last + 1)
            ts, gs, la, lo = times[span], speed[span], lat[span], lon[span]
            moving = _moving(gs, la, lo, ts)
            runs = _runs(moving)
            turns = [stop for start, stop, is_moving in runs
                     if not is_moving and _span(ts, start, stop) >= 600.0]
            anchor = turns[-1] - 1 if turns else 0
            sustained = [(start, stop) for start, stop, is_moving in runs
                         if is_moving and _span(ts, start, stop) >= 30.0]
            after = [start for start, _ in sustained if start >= anchor]
            onset_250 = np.nan
            stand_lat = stand_lon = np.nan
            if after:
                first_move = after[0]
                stationary = [i for i in range(anchor, first_move) if not moving[i]]
                refs = stationary if stationary else [anchor]
                stand_lat = float(np.nanmedian(la[refs]))
                stand_lon = float(np.nanmedian(lo[refs]))
                dlat = (la - stand_lat) * 110540.0
                dlon = (lo - stand_lon) * 111320.0 * np.cos(np.radians(stand_lat))
                exit_idx = _persistent_exit(np.hypot(dlat, dlon)[anchor:])
                if exit_idx is not None:
                    onset_250 = float(ts[anchor + exit_idx])
            calls = aircraft.callsign.iloc[span].dropna().astype(str)
            callsign = calls.mode().iat[0] if len(calls) else ""
            yield {
                "airport": str(airport).strip().upper(),
                "icao": str(icao).strip().lower(),
                "flight_key": norm(callsign),
                "wheels_off_s": float(ts[-1]),
                "onset_250_s": onset_250,
                "anchor_s": float(ts[anchor]),
                "anchor_lat": float(la[anchor]),
                "anchor_lon": float(lo[anchor]),
                "anchor_speed_kt": float(gs[anchor]),
            }


def _stand_map(geometry: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    g = geometry.copy()
    g["airport_key"] = g.airport.map(category)
    g["stand_key"] = g.stand_name_mvt.map(norm)
    g["stand_latitude"] = pd.to_numeric(g.stand_latitude, errors="coerce")
    g["stand_longitude"] = pd.to_numeric(g.stand_longitude, errors="coerce")
    g = g.dropna(subset=["stand_latitude", "stand_longitude"])
    records = g.groupby(["airport_key", "stand_key"], sort=True)[
        ["stand_latitude", "stand_longitude"]
    ].agg("median")
    return {(str(airport), str(stand)): (float(row.stand_latitude), float(row.stand_longitude))
            for (airport, stand), row in records.iterrows()}


def _segment_implied(segment: dict, stand: tuple[float, float] | None) -> float:
    if stand is None:
        return np.nan
    lat, lon = segment["anchor_lat"], segment["anchor_lon"]
    if not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan
    distance = _haversine_m(lat, lon, stand[0], stand[1])
    speed = segment["anchor_speed_kt"]
    speed = speed if np.isfinite(speed) and speed >= 1.0 else 3.0
    return float(segment["anchor_s"] - distance / (speed * 0.514444))


def _match_segments(dep: pd.DataFrame, segments: list[dict],
                    stand_positions: dict) -> tuple[pd.DataFrame, dict]:
    """Apply airport/date/flight and nearest-MVT matching with frozen margins."""
    dep = dep.copy().reset_index(drop=True)
    dep["_rowpos"] = np.arange(len(dep), dtype="int64")
    dep["_airport_key"] = dep["ADEP_mvt"].map(category)
    dep["_day_key"] = pd.DatetimeIndex(dep["MVT_TIME_UTC_mvt"]).strftime("%Y-%m-%d")
    dep["_flight_key"] = dep["FLIGHT_mvt"].map(norm)
    dep["_mvt_s"] = epoch_seconds(dep["MVT_TIME_UTC_mvt"])
    candidates: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
    for airport, day, flight, mvt_s, rowpos in zip(
        dep["_airport_key"], dep["_day_key"], dep["_flight_key"],
        dep["_mvt_s"], dep["_rowpos"],
    ):
        key = (airport, day, flight)
        if flight and np.isfinite(mvt_s):
            candidates.setdefault(key, []).append((float(mvt_s), int(rowpos)))
    accepted: dict[int, list[tuple[float, dict]]] = {}
    ambiguous_nearest = 0
    eligible_segments = 0
    for segment in segments:
        key = (segment["airport"],
               pd.Timestamp(segment["wheels_off_s"], unit="s", tz="UTC").strftime("%Y-%m-%d"),
               segment["flight_key"])
        if not key[2]:
            continue
        options = candidates.get(key, [])
        if not options:
            continue
        errors = sorted((abs(mvt_s - segment["wheels_off_s"]), pos)
                        for mvt_s, pos in options)
        if errors[0][0] > 600.0:
            continue
        if len(errors) > 1 and errors[1][0] - errors[0][0] < 120.0:
            ambiguous_nearest += 1
            continue
        eligible_segments += 1
        pos = errors[0][1]
        target = dep.iloc[pos]
        stand_key = (str(target["_airport_key"]), norm(target["STAND_mvt"]))
        stand = stand_positions.get(stand_key)
        implied = _segment_implied(segment, stand)
        current = (float(target["_mvt_s"]) - segment["onset_250_s"]
                   if np.isfinite(segment["onset_250_s"]) else np.nan)
        implied_duration = float(target["_mvt_s"] - implied) if np.isfinite(implied) else np.nan
        record = {"clock_current250": current, "clock_implied": implied_duration,
                  "segment_error_s": float(errors[0][0]),
                  "segment_wheels_off_s": segment["wheels_off_s"]}
        accepted.setdefault(pos, []).append((float(errors[0][0]), record))
    values = {"clock_current250": np.full(len(dep), np.nan, dtype="float64"),
              "clock_implied": np.full(len(dep), np.nan, dtype="float64")}
    duplicate_row_matches = 0
    for pos, rows in accepted.items():
        rows.sort(key=lambda pair: pair[0])
        if len(rows) > 1 and rows[1][0] - rows[0][0] < 120.0:
            duplicate_row_matches += 1
            continue
        values["clock_current250"][pos] = rows[0][1]["clock_current250"]
        values["clock_implied"][pos] = rows[0][1]["clock_implied"]
    dep["clock_current250"] = values["clock_current250"]
    dep["clock_implied"] = values["clock_implied"]
    return dep, {"segments_matched": int(eligible_segments),
                 "ambiguous_segment_nearest": int(ambiguous_nearest),
                 "multiple_segments_per_row": int(duplicate_row_matches),
                 "rows_with_current250": int(np.isfinite(values["clock_current250"]).sum()),
                 "rows_with_implied": int(np.isfinite(values["clock_implied"]).sum())}


def _day_point_files(expected: dict) -> dict[str, Path]:
    by_day = {}
    for path in INPUT.rglob("points_*.parquet"):
        day = path.name[7:17]
        if day in by_day:
            raise RuntimeError(f"duplicate mounted point file for {day}")
        by_day[day] = path
    required = {record["day"]: record
                for month_records in expected["months"].values() for record in month_records}
    if set(by_day) != set(required):
        missing, extra = sorted(set(required) - set(by_day)), sorted(set(by_day) - set(required))
        raise RuntimeError(f"point-day coverage mismatch; missing={missing[:4]} extra={extra[:4]}")
    for day, path in by_day.items():
        spec = required[day]
        if path.stat().st_size != int(spec["bytes"]):
            raise RuntimeError(f"{day}: point-file byte size differs from exp-045 manifest")
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"{day}: point-file SHA-256 differs from exp-045 manifest")
    return by_day


def _window_id(dep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = dep[KEY].drop_duplicates().reset_index(drop=True)
    meta.insert(0, "window_id", np.arange(len(meta), dtype="int32"))
    ids = pd.MultiIndex.from_frame(dep[KEY])
    lookup = pd.MultiIndex.from_frame(meta[KEY])
    codes = lookup.get_indexer(ids)
    if (codes < 0).any():
        raise RuntimeError("a departure row has no source-window key")
    return meta, codes.astype("int32")


def _merge_window_values(meta: pd.DataFrame, lag: int,
                         stats: pd.DataFrame, *, airport_only: bool = False) -> pd.DataFrame:
    source = meta[["window_id", *KEY]].copy()
    source = source.rename(columns={"slot": "source_slot"})
    source["source_slot"] = source["source_slot"].astype("int16") + int(lag)
    source = source.loc[source.source_slot.between(0, 47)].copy()
    keys = ARR_KEY if airport_only else SOURCE_KEY
    stats = stats.rename(columns={"slot": "source_slot"})
    out = source.merge(stats, on=keys, how="left", validate="many_to_one", sort=False)
    return out.set_index("window_id").reindex(meta.window_id.to_numpy())


def _period_values(dep: pd.DataFrame, stats: pd.DataFrame, width: int, lag: int,
                   *, airport_only: bool, value: str) -> tuple[np.ndarray, np.ndarray]:
    """Look up one source-bin statistic for every departure row."""
    period = f"bin_w{width}"
    n_bins = 1440 // width
    bins = dep[period].to_numpy("int16") + int(lag)
    query = pd.DataFrame({
        "_row": np.arange(len(dep), dtype="int64"),
        "airport": dep.airport.astype(str).to_numpy(),
        "day": dep.day.astype(str).to_numpy(),
        "source_bin": bins,
    })
    if not airport_only:
        query["runway"] = dep.runway.astype(str).to_numpy()
    query = query.loc[query.source_bin.between(0, n_bins - 1)].copy()
    keys = ["airport", "day", "source_bin"] if airport_only else [
        "airport", "runway", "day", "source_bin"]
    selected = query.merge(stats, on=keys, how="left", validate="many_to_one", sort=False)
    values = np.full(len(dep), np.nan, dtype="float64")
    hours = np.full(len(dep), np.nan, dtype="float64")
    if len(selected):
        positions = selected["_row"].to_numpy("int64")
        values[positions] = pd.to_numeric(selected[value], errors="coerce").to_numpy("float64")
        hours[positions] = (selected.source_bin.to_numpy("int16") * width // 60).astype("float64")
    return values, hours


def _wide_column(family: str, scope: str, width: int, lag: int) -> str:
    return f"{family}_{scope}_w{width}_l{lag:+d}"


def build_month_raw(month: str, dep: pd.DataFrame, arr: pd.DataFrame,
                    point_files: dict[str, Path], stand_positions: dict) -> dict:
    """Build raw row/window proxies. This function reads no departure label."""
    dep = dep.loc[dep.month.eq(month)].copy().reset_index(drop=True)
    arr = arr.loc[arr.month.eq(month)].copy().reset_index(drop=True)
    dep_time = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    arr_time = pd.to_datetime(arr["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    if dep_time.isna().any():
        raise RuntimeError(f"{month}: departure row has no UTC movement time")
    arr_valid_time = arr_time.notna()
    arr = arr.loc[arr_valid_time].reset_index(drop=True)
    arr_time = arr_time.loc[arr_valid_time].reset_index(drop=True)
    dep["airport"] = dep["ADEP_mvt"].map(category)
    dep["runway"] = dep["RUNWAY_mvt"].map(category)
    dep["day"] = dep_time.dt.strftime("%Y-%m-%d")
    dep["slot"] = (dep_time.dt.hour * 2 + dep_time.dt.minute // 30).astype("int16")
    dep["hour"] = dep_time.dt.hour.astype("int8")
    dep["_mvt_s"] = epoch_seconds(dep_time)
    arr["airport"] = arr["ADES_mvt"].map(category)
    arr["runway"] = arr["RUNWAY_mvt"].map(category)
    arr["day"] = arr_time.dt.strftime("%Y-%m-%d")
    arr["slot"] = (arr_time.dt.hour * 2 + arr_time.dt.minute // 30).astype("int16")
    arr["hour"] = arr_time.dt.hour.astype("int8")
    arr["taxi_in"] = pd.to_numeric(arr["TAXITIME_SEC_mvt"], errors="coerce")
    dep = dep.loc[dep._mvt_s.notna() & dep.day.ne("NaT")].copy().reset_index(drop=True)
    arr = arr.loc[arr.taxi_in.notna() & arr.day.ne("NaT")].copy().reset_index(drop=True)
    for width in (15, 30, 60):
        dep[f"bin_w{width}"] = (
            dep.hour.astype("int16") * 60
            + pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.minute.to_numpy("int16")
        ) // width
        arr[f"bin_w{width}"] = (
            arr.hour.astype("int16") * 60
            + pd.to_datetime(arr.MVT_TIME_UTC_mvt, utc=True).dt.minute.to_numpy("int16")
        ) // width
    if dep[ID].duplicated().any() or dep[ID].isna().any():
        raise RuntimeError(f"{month}: duplicate or missing departure IDs")

    dep_keys = set(zip(dep.airport, dep.day, dep.FLIGHT_mvt.map(norm)))
    segments = []
    day_rows = {day: part for day, part in dep.groupby("day", sort=True)}
    for day, point_path in sorted(point_files.items()):
        if day[:7] != month or day not in day_rows:
            continue
        targets = day_rows[day]
        airports = set(targets.airport.astype(str))
        flights = set(targets.FLIGHT_mvt.map(norm))
        points = pd.read_parquet(point_path, columns=POINT_COLUMNS)
        points["_day"] = pd.to_datetime(points.t, unit="s", utc=True).dt.strftime("%Y-%m-%d")
        points["_flight_key"] = points.callsign.map(norm)
        points = points.loc[
            points.airport.astype(str).str.upper().isin(airports)
            & points._day.eq(day)
            & points._flight_key.isin(flights)
        ].copy()
        if points.empty:
            continue
        for segment in departure_segments(points):
            if not segment["flight_key"]:
                continue
            event_day = pd.Timestamp(segment["wheels_off_s"], unit="s", tz="UTC").strftime("%Y-%m-%d")
            if (segment["airport"], event_day, segment["flight_key"]) in dep_keys:
                segments.append(segment)

    # A segment can occur in adjacent daily extracts. Keep a single stable record.
    unique = {}
    for segment in segments:
        signature = (segment["airport"], segment["flight_key"], round(segment["wheels_off_s"], 1))
        unique.setdefault(signature, segment)
    dep, match_audit = _match_segments(dep, list(unique.values()), stand_positions)

    arr["_key"] = list(zip(arr.airport, arr.runway, arr.day, arr.slot.astype(int)))
    arr_valid = arr.loc[np.isfinite(arr.taxi_in.to_numpy("float64"))].copy()
    arr_runway = (arr_valid.groupby(["airport", "runway", "day", "slot"], sort=False)
                  .taxi_in.mean().rename("arrival_runway_mean").reset_index())
    arr_airport = (arr_valid.groupby(["airport", "day", "slot"], sort=False)
                   .taxi_in.mean().rename("arrival_airport_mean").reset_index())
    arrival_events = arr[["airport", "hour", "taxi_in"]].copy()
    arrival_events["airport"] = arrival_events["airport"].astype("category")

    dep["_key"] = list(zip(dep.airport, dep.runway, dep.day, dep.slot.astype(int)))
    for feature in ("clock_current250", "clock_implied"):
        dep[feature] = pd.to_numeric(dep[feature], errors="coerce")
    clock_stats = {}
    for feature in ("clock_current250", "clock_implied"):
        part = dep.loc[np.isfinite(dep[feature].to_numpy("float64"))]
        if len(part):
            grouped = part.groupby(["airport", "runway", "day", "slot"], sort=False)[feature]
            stats = grouped.agg(["sum", "count"]).reset_index()
        else:
            stats = pd.DataFrame(columns=["airport", "runway", "day", "slot", "sum", "count"])
        stats = stats.rename(columns={"sum": f"{feature}_sum", "count": f"{feature}_count"})
        clock_stats[feature] = stats

    ordered = dep.sort_values(["airport", "runway", "day", "_mvt_s", ID], kind="stable").copy()
    prior = ordered.groupby(["airport", "runway", "day"], sort=False)._mvt_s.shift(1)
    gap = ordered._mvt_s.to_numpy("float64") - prior.to_numpy("float64")
    valid_gap = np.isfinite(gap) & (gap >= 0.0)
    ordered["short90"] = np.where(valid_gap, (gap < 90.0).astype("float64"), np.nan)
    short_stats = (ordered.groupby(["airport", "runway", "day", "slot"], sort=False)
                   .short90.mean().rename("spacing_short_share").reset_index())
    count_stats = (dep.groupby(["airport", "runway", "day", "slot"], sort=False)
                   .size().rename("departure_count").reset_index())
    delay = dep._mvt_s.to_numpy("float64") - epoch_seconds(dep["EOBT_1_flt"])
    dep["filed_delay"] = delay
    delay_stats = (dep.groupby(["airport", "runway", "day", "slot"], sort=False)
                   .filed_delay.mean().rename("filed_delay_mean").reset_index())
    dep_hourly = dep.groupby(["airport", "runway", "day", "hour"], sort=False).size()
    hourly = dep_hourly.rename("count").reset_index()
    for column in ("airport", "runway", "day"):
        hourly[column] = hourly[column].astype("category")

    # Wider/coarser families. Each table is keyed by the UTC bin at its own width.
    wide_stats: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    wide_count_events = []
    for scope, width in WIDE_SCOPES:
        bin_col = f"bin_w{width}"
        airport_only = scope == "airport"
        keys = (["airport", "day", bin_col] if airport_only
                else ["airport", "runway", "day", bin_col])
        arrival_runway = (arr.groupby(keys, sort=False).taxi_in.mean()
                          .rename("arrival_mean").reset_index()
                          .rename(columns={bin_col: "source_bin"}))
        if airport_only:
            arrival_airport = arrival_runway.copy()
        else:
            airport_keys = ["airport", "day", bin_col]
            arrival_airport = (arr.groupby(airport_keys, sort=False).taxi_in.mean()
                               .rename("arrival_mean").reset_index()
                               .rename(columns={bin_col: "source_bin"}))

        group_cols = ["airport", "day"] if airport_only else ["airport", "runway", "day"]
        ordered = dep.sort_values([*group_cols, "_mvt_s", ID], kind="stable").copy()
        prior = ordered.groupby(group_cols, sort=False)._mvt_s.shift(1)
        gap = ordered._mvt_s.to_numpy("float64") - prior.to_numpy("float64")
        valid_gap = np.isfinite(gap) & (gap >= 0.0)
        ordered["short90"] = np.where(valid_gap, (gap < 90.0).astype("float64"), np.nan)
        wide_short_stats = (ordered.groupby([*group_cols, bin_col], sort=False).short90.mean()
                            .rename("spacing_short_share").reset_index()
                            .rename(columns={bin_col: "source_bin"}))
        wide_departure_counts = (dep.groupby([*group_cols, bin_col], sort=False).size()
                                 .rename("departure_count").reset_index()
                                 .rename(columns={bin_col: "source_bin"}))
        wide_stats[(scope, width)] = {
            "arrival_runway": arrival_runway,
            "arrival_airport": arrival_airport,
            "spacing": wide_short_stats,
            "count": wide_departure_counts,
        }
        count_event = wide_departure_counts.copy()
        count_event["scope"] = scope
        count_event["width"] = width
        if airport_only:
            count_event["runway"] = "<ALL>"
        wide_count_events.append(count_event[["scope", "width", "airport", "runway", "departure_count"]])

    meta, row_window = _window_id(dep)
    row_features = pd.DataFrame({ID: dep[ID].to_numpy("int64"), "window_id": row_window})
    row_features["clock_current250_row"] = dep.clock_current250.to_numpy("float64")
    row_features["clock_implied_row"] = dep.clock_implied.to_numpy("float64")
    row_features["airport"] = dep.airport.astype("category").reset_index(drop=True)
    row_features["runway"] = dep.runway.astype("category").reset_index(drop=True)
    row_features["day"] = dep.day.astype("category").reset_index(drop=True)
    row_features["slot"] = dep.slot.to_numpy("int16")
    for width in (15, 30, 60):
        row_features[f"bin_w{width}"] = dep[f"bin_w{width}"].to_numpy("int16")
    row_features["month"] = pd.Categorical([month] * len(row_features))

    arrival_mean = {}
    source_hours = {}
    clock_window = {}
    for lag in (*LAGS, *T2_LAGS):
        arrived = _merge_window_values(meta, lag, arr_runway, airport_only=False)
        airport_arrived = _merge_window_values(meta, lag, arr_airport, airport_only=True)
        vals = arrived.get("arrival_runway_mean", pd.Series(np.nan, index=meta.index)).to_numpy("float64")
        fallback = airport_arrived.get("arrival_airport_mean", pd.Series(np.nan, index=meta.index)).to_numpy("float64")
        vals = np.where(np.isfinite(vals), vals, fallback)
        arrival_mean[lag] = vals[row_window]
        source_hours[lag] = ((meta.slot.to_numpy("int16") + lag) // 2).astype("float64")[row_window]
        stats = _merge_window_values(meta, lag, short_stats, airport_only=False)
        counts = _merge_window_values(meta, lag, count_stats, airport_only=False)
        delays = _merge_window_values(meta, lag, delay_stats, airport_only=False)
        current = {"spacing_short_share": stats.get("spacing_short_share", pd.Series(np.nan, index=meta.index)).to_numpy("float64")[row_window],
                   "departure_count": counts.get("departure_count", pd.Series(np.nan, index=meta.index)).to_numpy("float64")[row_window],
                   "filed_delay_mean": delays.get("filed_delay_mean", pd.Series(np.nan, index=meta.index)).to_numpy("float64")[row_window]}
        for feature in ("clock_current250", "clock_implied"):
            stats = _merge_window_values(meta, lag, clock_stats[feature], airport_only=False)
            current[f"{feature}_sum"] = stats.get(f"{feature}_sum", pd.Series(0.0, index=meta.index)).fillna(0).to_numpy("float64")[row_window]
            current[f"{feature}_count"] = stats.get(f"{feature}_count", pd.Series(0.0, index=meta.index)).fillna(0).to_numpy("float64")[row_window]
        clock_window[lag] = current
        row_features[f"arrival_mean_l{lag:+d}"] = arrival_mean[lag]
        row_features[f"source_hour_l{lag:+d}"] = source_hours[lag]
        row_features[f"spacing_short_share_l{lag:+d}"] = current["spacing_short_share"]
        row_features[f"departure_count_l{lag:+d}"] = current["departure_count"]
        row_features[f"filed_delay_mean_l{lag:+d}"] = current["filed_delay_mean"]
        for feature in ("clock_current250", "clock_implied"):
            total = current[f"{feature}_sum"].copy()
            count = current[f"{feature}_count"].copy()
            if lag == 0:
                own = row_features[f"{feature}_row"].to_numpy("float64")
                own_valid = np.isfinite(own)
                total[own_valid] -= own[own_valid]
                count[own_valid] -= 1.0
            mean = np.divide(total, count, out=np.full(len(total), np.nan), where=count > 0)
            row_features[f"{feature}_l{lag:+d}"] = mean

    for scope, width in WIDE_SCOPES:
        tables = wide_stats[(scope, width)]
        airport_only = scope == "airport"
        for lag in WIDE_LAGS:
            arr_mean, arr_hour = _period_values(
                dep, tables["arrival_runway"], width, lag,
                airport_only=airport_only, value="arrival_mean")
            if not airport_only:
                arr_fallback, _ = _period_values(
                    dep, tables["arrival_airport"], width, lag,
                    airport_only=True, value="arrival_mean")
                arr_mean = np.where(np.isfinite(arr_mean), arr_mean, arr_fallback)
            short, _ = _period_values(dep, tables["spacing"], width, lag,
                                      airport_only=airport_only,
                                      value="spacing_short_share")
            count, _ = _period_values(dep, tables["count"], width, lag,
                                      airport_only=airport_only,
                                      value="departure_count")
            row_features[_wide_column("arrival_mean", scope, width, lag)] = arr_mean
            row_features[_wide_column("arrival_hour", scope, width, lag)] = arr_hour
            row_features[_wide_column("spacing_short_share", scope, width, lag)] = short
            row_features[_wide_column("departure_count", scope, width, lag)] = count

    return {
        "month": month,
        "row_features": row_features,
        "window_meta": meta,
        "arrival_events": arrival_events,
        "hourly_counts": hourly,
        "wide_count_events": pd.concat(wide_count_events, ignore_index=True),
        "match_audit": match_audit,
        "raw_departure_rows": int(len(dep)),
        "raw_arrival_rows": int(len(arr)),
    }


def fit_denominators(raw_months: dict[str, dict], train_months: list[str]) -> dict:
    arrivals = pd.concat([raw_months[m]["arrival_events"] for m in train_months], ignore_index=True)
    arrivals = arrivals.loc[np.isfinite(arrivals.taxi_in.to_numpy("float64"))]
    if arrivals.empty:
        raise RuntimeError("training months have no ARR taxi-in observations")
    arr_ah = arrivals.groupby(["airport", "hour"], sort=False, observed=True).taxi_in.median().to_dict()
    arr_airport = arrivals.groupby("airport", sort=False, observed=True).taxi_in.median().to_dict()
    arr_global = float(arrivals.taxi_in.median())

    hours = pd.concat([raw_months[m]["hourly_counts"] for m in train_months], ignore_index=True)
    if hours.empty:
        raise RuntimeError("training months have no departure hour counts")
    q95_ar = hours.groupby(["airport", "runway"], sort=False, observed=True)["count"].quantile(0.95).to_dict()
    q95_a = hours.groupby("airport", sort=False, observed=True)["count"].quantile(0.95).to_dict()
    q95_global = float(hours["count"].quantile(0.95))
    wide_counts = pd.concat([raw_months[m]["wide_count_events"] for m in train_months],
                            ignore_index=True)
    wide_q95 = {}
    for scope, width in WIDE_SCOPES:
        part = wide_counts.loc[wide_counts.scope.eq(scope) & wide_counts.width.eq(width)]
        if part.empty:
            raise RuntimeError(f"training months have no {scope} {width}-minute departure counts")
        if scope == "runway":
            by_ar = part.groupby(["airport", "runway"], sort=False).departure_count.quantile(0.95).to_dict()
            by_airport = part.groupby("airport", sort=False).departure_count.quantile(0.95).to_dict()
        else:
            by_ar = {}
            by_airport = part.groupby("airport", sort=False).departure_count.quantile(0.95).to_dict()
        wide_q95[(scope, width)] = {
            "airport_runway": by_ar,
            "airport": by_airport,
            "global": float(part.departure_count.quantile(0.95)),
        }
    return {"arrival_ah_median": arr_ah, "arrival_airport_median": arr_airport,
            "arrival_global_median": arr_global, "q95_airport_runway": q95_ar,
            "q95_airport": q95_a, "q95_global": q95_global,
            "wide_q95": wide_q95}


def _mapped(values: list, mapping: dict) -> np.ndarray:
    return np.asarray([mapping.get(tuple(x) if isinstance(x, tuple) else x, np.nan)
                       for x in values], dtype="float64")


def transform_rows(raw: dict, denominators: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = raw["row_features"].copy()
    airports = rows.airport.astype(str).to_numpy()
    runways = rows.runway.astype(str).to_numpy()
    z = pd.DataFrame({ID: rows[ID].to_numpy("int64"), "window_id": rows.window_id.to_numpy("int32")})

    def arrival_ratio(mean: np.ndarray, hour: np.ndarray) -> np.ndarray:
        ah = _mapped([(a, int(h)) if np.isfinite(h) else (a, -999)
                      for a, h in zip(airports, hour)], denominators["arrival_ah_median"])
        ap = _mapped(airports.tolist(), denominators["arrival_airport_median"])
        den = np.where(np.isfinite(ah) & (ah > 0), ah, ap)
        den = np.where(np.isfinite(den) & (den > 0), den,
                       denominators["arrival_global_median"])
        return np.divide(mean, den, out=np.full(len(rows), np.nan),
                         where=np.isfinite(mean) & (den > 0))

    def custom_count_ratio(count: np.ndarray, *, scope: str, width: int) -> np.ndarray:
        table = denominators["wide_q95"][(scope, width)]
        if scope == "runway":
            exact = _mapped(list(zip(airports, runways)), table["airport_runway"])
        else:
            exact = np.full(len(rows), np.nan, dtype="float64")
        airport_q95 = _mapped(airports.tolist(), table["airport"])
        q95 = np.where(np.isfinite(exact) & (exact > 0), exact, airport_q95)
        q95 = np.where(np.isfinite(q95) & (q95 > 0), q95, table["global"])
        return np.divide(count, q95, out=np.full(len(rows), np.nan),
                         where=np.isfinite(count) & (q95 > 0))

    for lag in (*LAGS, *T2_LAGS):
        mean = rows[f"arrival_mean_l{lag:+d}"].to_numpy("float64")
        hour = rows[f"source_hour_l{lag:+d}"].to_numpy("float64")
        ratio = arrival_ratio(mean, hour)
        count = rows[f"departure_count_l{lag:+d}"].to_numpy("float64")
        ar = _mapped(list(zip(airports, runways)), denominators["q95_airport_runway"])
        ap95 = _mapped(airports.tolist(), denominators["q95_airport"])
        q95 = np.where(np.isfinite(ar) & (ar > 0), ar, ap95)
        q95 = np.where(np.isfinite(q95) & (q95 > 0), q95, denominators["q95_global"])
        count_ratio = np.divide(count, q95, out=np.full(len(rows), np.nan), where=np.isfinite(count) & (q95 > 0))
        for family, values in (
            ("arrival_mean", mean),
            ("arrival_ratio", ratio),
            ("clock_current250", rows[f"clock_current250_l{lag:+d}"].to_numpy("float64")),
            ("clock_implied", rows[f"clock_implied_l{lag:+d}"].to_numpy("float64")),
            ("spacing_short_share", rows[f"spacing_short_share_l{lag:+d}"].to_numpy("float64")),
            ("departure_count_q95", count_ratio),
            ("filed_delay_mean", rows[f"filed_delay_mean_l{lag:+d}"].to_numpy("float64")),
        ):
            z[f"{family}_l{lag:+d}"] = np.asarray(values, dtype="float32")

    for scope, width in WIDE_SCOPES:
        for lag in WIDE_LAGS:
            mean = rows[_wide_column("arrival_mean", scope, width, lag)].to_numpy("float64")
            hour = rows[_wide_column("arrival_hour", scope, width, lag)].to_numpy("float64")
            ratio = arrival_ratio(mean, hour)
            count = rows[_wide_column("departure_count", scope, width, lag)].to_numpy("float64")
            count_q95 = custom_count_ratio(count, scope=scope, width=width)
            z[_wide_column("arrival_mean", scope, width, lag)] = mean.astype("float32")
            z[_wide_column("arrival_ratio", scope, width, lag)] = ratio.astype("float32")
            z[_wide_column("departure_count_q95", scope, width, lag)] = count_q95.astype("float32")
            z[_wide_column("spacing_short_share", scope, width, lag)] = rows[
                _wide_column("spacing_short_share", scope, width, lag)
            ].to_numpy("float32")

    z = z[[ID, "window_id", *ALL_COLS]]
    if list(z.columns[2:]) != ALL_COLS:
        raise RuntimeError("exp-136 proxy feature order differs from its frozen 95-column vector")
    windows = raw["window_meta"].copy()
    means = z.groupby("window_id", sort=False)[ALL_COLS].mean()
    windows = windows.set_index("window_id").join(means, how="left").reset_index()
    return z, windows


def sham_windows(windows: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = windows[["window_id", *KEY, *EXTRA_COLS]].copy()
    out[EXTRA_COLS] = out[EXTRA_COLS].astype("float32")
    rng = np.random.default_rng(seed)
    # One permutation moves all 74 added values among windows in each airport-month.
    for _, positions in out.groupby(["airport"], sort=True, observed=True).indices.items():
        ids = np.asarray(positions, dtype="int64")
        if len(ids) > 1:
            out.loc[ids, EXTRA_COLS] = out.loc[rng.permutation(ids), EXTRA_COLS].to_numpy()
    return out[["window_id", *KEY, *EXTRA_COLS]]


def row_view(z: pd.DataFrame, windows: pd.DataFrame, *, mode: str,
             seed: int) -> pd.DataFrame:
    if mode == "proxy":
        return z[[ID, *ALL_COLS]].copy()
    if mode == "sham":
        source = sham_windows(windows, seed)
    else:
        raise ValueError(f"unknown proxy view {mode}")
    base = z[[ID, "window_id", *Z_COLS]].copy()
    return base.merge(source[["window_id", *EXTRA_COLS]], on="window_id",
                      how="left", validate="many_to_one")[[ID, *ALL_COLS]]


def build_all(months: tuple[str, ...], rank_months: tuple[str, ...]) -> tuple[dict, dict]:
    dep_path = exact_file("proxy_departures.parquet")
    arr_path = exact_file("proxy_arrivals.parquet")
    expected_path = exact_file("points_expected.json")
    geometry_path = exact_file("stand_geometry.csv")
    expected = json.loads(expected_path.read_text())
    point_files = _day_point_files(expected)
    dep = pd.read_parquet(dep_path, columns=RAW_DEP_COLUMNS + ["month"])
    arr = pd.read_parquet(arr_path, columns=RAW_ARR_COLUMNS + ["month"])
    dep[ID] = pd.to_numeric(dep[ID], errors="raise").astype("int64")
    arr[ID] = pd.to_numeric(arr[ID], errors="raise").astype("int64")
    if "TAXITIME_SEC_mvt" in dep.columns:
        raise RuntimeError("departure source includes departure taxi-time target")
    if dep[ID].duplicated().any() or arr[ID].duplicated().any():
        raise RuntimeError("private exp-134 movement input has duplicate IDs")
    stands = _stand_map(pd.read_csv(geometry_path))
    raw_months = {}
    audits = {"point_files_verified": len(point_files), "point_bytes_verified": int(sum(p.stat().st_size for p in point_files.values())),
              "source_hashes": {"proxy_departures.parquet": sha256(dep_path),
                                "proxy_arrivals.parquet": sha256(arr_path),
                                "points_expected.json": sha256(expected_path),
                                "stand_geometry.csv": sha256(geometry_path)},
              "matches_by_month": {}}
    for month in (*months, *rank_months):
        raw = build_month_raw(month, dep, arr, point_files, stands)
        raw_months[month] = raw
        audits["matches_by_month"][month] = raw["match_audit"]
        print(f"proxy source built {month}: dep={raw['raw_departure_rows']:,} ",
              f"arr={raw['raw_arrival_rows']:,} matches={raw['match_audit']['segments_matched']:,}",
              flush=True)
    return raw_months, audits


if __name__ == "__main__":
    raise SystemExit("Import build_all() from the exp-134 kernel entry point")
