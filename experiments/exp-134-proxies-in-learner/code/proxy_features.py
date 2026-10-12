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
Z_COLS = [f"{name}_l{lag:+d}" for name in PROXY_BASES for lag in LAGS]
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
    paths = sorted(path for path in INPUT.rglob(name) if path.is_file())
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name}; found {len(paths)}")
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

    meta, row_window = _window_id(dep)
    row_features = pd.DataFrame({ID: dep[ID].to_numpy("int64"), "window_id": row_window})
    row_features["clock_current250_row"] = dep.clock_current250.to_numpy("float64")
    row_features["clock_implied_row"] = dep.clock_implied.to_numpy("float64")
    row_features["airport"] = dep.airport.astype("category").reset_index(drop=True)
    row_features["runway"] = dep.runway.astype("category").reset_index(drop=True)
    row_features["day"] = dep.day.astype("category").reset_index(drop=True)
    row_features["slot"] = dep.slot.to_numpy("int16")
    row_features["month"] = pd.Categorical([month] * len(row_features))

    arrival_mean = {}
    source_hours = {}
    clock_window = {}
    for lag in LAGS:
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

    return {
        "month": month,
        "row_features": row_features,
        "window_meta": meta,
        "arrival_events": arrival_events,
        "hourly_counts": hourly,
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
    return {"arrival_ah_median": arr_ah, "arrival_airport_median": arr_airport,
            "arrival_global_median": arr_global, "q95_airport_runway": q95_ar,
            "q95_airport": q95_a, "q95_global": q95_global}


def _mapped(values: list, mapping: dict) -> np.ndarray:
    return np.asarray([mapping.get(tuple(x) if isinstance(x, tuple) else x, np.nan)
                       for x in values], dtype="float64")


def transform_rows(raw: dict, denominators: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = raw["row_features"].copy()
    airports = rows.airport.astype(str).to_numpy()
    runways = rows.runway.astype(str).to_numpy()
    z = pd.DataFrame({ID: rows[ID].to_numpy("int64"), "window_id": rows.window_id.to_numpy("int32")})
    for lag in LAGS:
        mean = rows[f"arrival_mean_l{lag:+d}"].to_numpy("float64")
        hour = rows[f"source_hour_l{lag:+d}"].to_numpy("float64")
        ah = _mapped([(a, int(h)) if np.isfinite(h) else (a, -999) for a, h in zip(airports, hour)],
                     denominators["arrival_ah_median"])
        ap = _mapped(airports.tolist(), denominators["arrival_airport_median"])
        den = np.where(np.isfinite(ah) & (ah > 0), ah, ap)
        den = np.where(np.isfinite(den) & (den > 0), den, denominators["arrival_global_median"])
        ratio = np.divide(mean, den, out=np.full(len(rows), np.nan), where=np.isfinite(mean) & (den > 0))
        count = rows[f"departure_count_l{lag:+d}"].to_numpy("float64")
        ar = _mapped(list(zip(airports, runways)), denominators["q95_airport_runway"])
        ap95 = _mapped(airports.tolist(), denominators["q95_airport"])
        q95 = np.where(np.isfinite(ar) & (ar > 0), ar, ap95)
        q95 = np.where(np.isfinite(q95) & (q95 > 0), q95, denominators["q95_global"])
        count_ratio = np.divide(count, q95, out=np.full(len(rows), np.nan), where=np.isfinite(count) & (q95 > 0))
        z[f"arrival_mean_l{lag:+d}"] = mean.astype("float32")
        z[f"arrival_ratio_l{lag:+d}"] = ratio.astype("float32")
        z[f"clock_current250_l{lag:+d}"] = rows[f"clock_current250_l{lag:+d}"].to_numpy("float32")
        z[f"clock_implied_l{lag:+d}"] = rows[f"clock_implied_l{lag:+d}"].to_numpy("float32")
        z[f"spacing_short_share_l{lag:+d}"] = rows[f"spacing_short_share_l{lag:+d}"].to_numpy("float32")
        z[f"departure_count_q95_l{lag:+d}"] = count_ratio.astype("float32")
        z[f"filed_delay_mean_l{lag:+d}"] = rows[f"filed_delay_mean_l{lag:+d}"].to_numpy("float32")
    z = z[[ID, "window_id", *Z_COLS]]
    if list(z.columns[2:]) != Z_COLS:
        raise RuntimeError("proxy feature order differs from the frozen 21-column vector")
    windows = raw["window_meta"].copy()
    means = z.groupby("window_id", sort=False)[Z_COLS].mean()
    windows = windows.set_index("window_id").join(means, how="left").reset_index()
    return z, windows


def shifted_windows(windows: pd.DataFrame) -> pd.DataFrame:
    lookup = {tuple(row[k] for k in KEY): row[Z_COLS].to_numpy("float64")
              for _, row in windows.iterrows()}
    values = np.full((len(windows), len(Z_COLS)), np.nan, dtype="float64")
    for i, row in enumerate(windows.itertuples(index=False)):
        airport, runway, day, slot = (str(row.airport), str(row.runway), str(row.day), int(row.slot))
        candidates = []
        for offset in (-7, 7):
            shifted_day = (pd.Timestamp(day) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            candidate = lookup.get((airport, runway, shifted_day, slot))
            if candidate is not None:
                candidates.append(candidate)
        if candidates:
            stack = np.vstack(candidates)
            finite = np.isfinite(stack)
            count = finite.sum(axis=0)
            values[i] = np.divide(np.nansum(stack, axis=0), count,
                                  out=np.full(len(Z_COLS), np.nan), where=count > 0)
    out = windows[["window_id", *KEY]].copy()
    out[Z_COLS] = values.astype("float32")
    return out


def sham_windows(windows: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = windows[["window_id", *KEY, *Z_COLS]].copy()
    out[Z_COLS] = out[Z_COLS].astype("float32")
    rng = np.random.default_rng(seed)
    # One permutation moves the complete 21-value vector among windows in each airport-month.
    for _, positions in out.groupby(["airport"], sort=True, observed=True).indices.items():
        ids = np.asarray(positions, dtype="int64")
        if len(ids) > 1:
            out.loc[ids, Z_COLS] = out.loc[rng.permutation(ids), Z_COLS].to_numpy()
    return out[["window_id", *KEY, *Z_COLS]]


def row_view(z: pd.DataFrame, windows: pd.DataFrame, *, mode: str,
             seed: int) -> pd.DataFrame:
    if mode == "proxy":
        return z[[ID, *Z_COLS]].copy()
    if mode == "shifted_day":
        source = shifted_windows(windows)
    elif mode == "sham":
        source = sham_windows(windows, seed)
    else:
        raise ValueError(f"unknown proxy view {mode}")
    return z[[ID, "window_id"]].merge(source[["window_id", *Z_COLS]],
                                       on="window_id", how="left", validate="many_to_one")


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
