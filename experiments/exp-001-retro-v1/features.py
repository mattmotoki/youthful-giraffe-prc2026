"""Feature frame for the retrospective taxi-out model (exp-001).

One function, `build_features`, serves training months, a ranking-masked
validation month, and each ranking month. It never reads a departure's own
BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt: both are blanked on every departure row
on entry, exactly as the ranking file blanks them, so a leak is structurally
impossible rather than a matter of discipline. Arrival block time and arrival
taxi-in are left intact from the v3 set on, because they are present in the
ranking file and are the only actual surface-time readout the task leaves.

Everything is computed from the single month frame it is given, because the
ranking file holds January and July 2026 as two isolated months.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Airport local time zones for the ten departure airports (plus the extras that
# appear as origins of arrivals; unknown codes fall back to UTC).
AIRPORT_TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
    "LTAI": "Europe/Istanbul",
}

CAT_FEATURES = [
    "airport",
    "airport_stand",
    "airport_runway",
    "actype_mvt",
    "actype_flt",
    "operator",
    "destination",
    "market_segment",
    "flight_type",
    "wake_cat",
    "flight_rule_mvt",
    "flight_rule_flt",
]

#: signals taken from prc/tail_features.py in the v4 set: everything there that
#: my own feature set does not already build.
TAIL_IMPORT = (
    "nonm", "vfr", "lobt_iobt", "eobt_sched", "nonm_day_count",
    "apt_day_med_mvt_aobt", "apt_day_p90_mvt_aobt", "apt_day_long_frac",
    "apt_hour_med_mvt_aobt", "mvt_aobt_vs_day_med",
    "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
    "n_arr_sched_mvt", "gap_arr_after_sched",
    "gap_dep_prev_mvt", "gap_dep_prev_aobt", "gap_dep_next_mvt", "stand_busy_frac_6h",
)

#: Frozen feature-frame contract. exp-008 (diversity) consumes the v2 frame for
#: its CatBoost, XGBoost and MLP siblings, so v2's column list is a published
#: interface: columns may not be renamed, reordered, retyped or removed without
#: a message to that worker through the desk. New columns go in a new version.
FRAME_VERSION = {"v1": 1, "v2": 1, "v3": 1, "v4": 1}

WINDOWS = (600, 1800, 3600)  # 10, 30, 60 minutes
_BIG = 1 << 34  # key stride for the packed (group, time) sort key


def _epoch(s: pd.Series) -> np.ndarray:
    """Seconds since epoch as float64, NaN for NaT.

    The parquet files store microsecond timestamps, so the resolution is pinned
    to nanoseconds before the integer cast rather than assumed.
    """
    v = s.astype("datetime64[ns, UTC]").astype("int64").to_numpy().astype("float64")
    v[s.isna().to_numpy()] = np.nan
    return v / 1e9


def _packed(group: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack (group code, time) into one sortable int64; NaN times give -1."""
    ok = np.isfinite(t)
    out = np.full(t.shape, -1, dtype="int64")
    out[ok] = group[ok].astype("int64") * _BIG + t[ok].astype("int64")
    return out


def _window_counts(ref_group, ref_t, q_group, q_t, window, side):
    """Count reference events in a window around each query time.

    side 'before': [t - window, t).  side 'after': (t, t + window].
    Events with a missing time are ignored; queries with a missing time get NaN.
    """
    ref = _packed(ref_group, ref_t)
    ref = np.sort(ref[ref >= 0])
    base = np.where(np.isfinite(q_t), q_group.astype("int64") * _BIG, 0)
    t = np.where(np.isfinite(q_t), q_t, 0).astype("int64")
    if side == "before":
        lo, hi = base + t - int(window), base + t
        cnt = np.searchsorted(ref, hi, "left") - np.searchsorted(ref, lo, "left")
    else:
        lo, hi = base + t, base + t + int(window)
        cnt = np.searchsorted(ref, hi, "right") - np.searchsorted(ref, lo, "right")
    return np.where(np.isfinite(q_t), cnt, np.nan)


def _between_counts(ref_group, ref_t, q_group, q_lo, q_hi):
    """Count reference events with group-matched time in [lo, hi]."""
    ref = _packed(ref_group, ref_t)
    ref = np.sort(ref[ref >= 0])
    ok = np.isfinite(q_lo) & np.isfinite(q_hi)
    base = np.where(ok, q_group.astype("int64") * _BIG, 0)
    lo = base + np.where(ok, q_lo, 0).astype("int64")
    hi = base + np.where(ok, q_hi, 0).astype("int64")
    cnt = np.searchsorted(ref, hi, "right") - np.searchsorted(ref, lo, "left")
    return np.where(ok, cnt, np.nan)


def _prev_next(ref_group, ref_t, q_group, q_t, side="prev"):
    """Index of the nearest reference event in the same group before/after each query.

    Returns (index into the reference arrays, mask of queries that have one).
    """
    ref = _packed(ref_group, ref_t)
    order = np.argsort(ref, kind="stable")
    rs = ref[order]
    q = _packed(q_group, q_t)
    if side == "prev":
        pos = np.searchsorted(rs, q, "left") - 1
        ok = (pos >= 0) & (q >= 0)
    else:
        pos = np.searchsorted(rs, q, "right")
        ok = (pos < len(rs)) & (q >= 0)
    posc = np.clip(pos, 0, max(len(rs) - 1, 0))
    ok = ok & (len(rs) > 0)
    same = ok & ((rs[posc] // _BIG) == (q // _BIG))
    return order[posc], same


def _take(values, idx, mask):
    out = np.full(mask.shape, np.nan)
    if len(values):
        out[mask] = values[idx[mask]]
    return out


def _window_sum(ref_group, ref_t, ref_v, q_group, q_t, window, side):
    """Count and sum of a reference value over a window, same group only."""
    ref = _packed(ref_group, ref_t)
    keep = ref >= 0
    ref, v = ref[keep], np.nan_to_num(ref_v[keep])
    order = np.argsort(ref, kind="stable")
    ref, v = ref[order], v[order]
    csum = np.concatenate([[0.0], np.cumsum(v)])
    base = np.where(np.isfinite(q_t), q_group.astype("int64") * _BIG, 0)
    t = np.where(np.isfinite(q_t), q_t, 0).astype("int64")
    if side == "before":
        lo, hi = np.searchsorted(ref, base + t - int(window), "left"), np.searchsorted(ref, base + t, "left")
    else:
        lo, hi = np.searchsorted(ref, base + t, "right"), np.searchsorted(ref, base + t + int(window), "right")
    n = (hi - lo).astype("float64")
    tot = csum[hi] - csum[lo]
    n[~np.isfinite(q_t)] = np.nan
    tot[~np.isfinite(q_t)] = np.nan
    return n, tot


def _prev_gap(group: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Seconds since the previous event in the same group (NaN for the first)."""
    packed = _packed(group, t)
    order = np.argsort(packed, kind="stable")
    p = packed[order]
    prev = np.full(p.shape, np.nan)
    same = np.empty(p.shape, dtype=bool)
    same[0] = False
    same[1:] = (p[1:] // _BIG) == (p[:-1] // _BIG)
    prev[1:] = np.where(same[1:], (p[1:] % _BIG) - (p[:-1] % _BIG), np.nan)
    prev[p < 0] = np.nan
    out = np.empty_like(prev)
    out[order] = prev
    return out


def _local_parts(t_utc: pd.Series, airport: pd.Series) -> pd.DataFrame:
    """Local wall-clock hour / weekday / minute-of-day at the departure airport."""
    out = pd.DataFrame(index=t_utc.index, columns=["local_hour", "local_weekday", "local_minute_of_day"], dtype="float64")
    for code, tz in AIRPORT_TZ.items():
        m = (airport == code).to_numpy()
        if not m.any():
            continue
        loc = t_utc[m].dt.tz_convert(ZoneInfo(tz))
        out.loc[m, "local_hour"] = loc.dt.hour.to_numpy(dtype="float64")
        out.loc[m, "local_weekday"] = loc.dt.dayofweek.to_numpy(dtype="float64")
        out.loc[m, "local_minute_of_day"] = (loc.dt.hour * 60 + loc.dt.minute).to_numpy(dtype="float64")
    unknown = (~airport.isin(AIRPORT_TZ)).to_numpy()
    if unknown.any():
        loc = t_utc[unknown]
        out.loc[unknown, "local_hour"] = loc.dt.hour.to_numpy(dtype="float64")
        out.loc[unknown, "local_weekday"] = loc.dt.dayofweek.to_numpy(dtype="float64")
        out.loc[unknown, "local_minute_of_day"] = (loc.dt.hour * 60 + loc.dt.minute).to_numpy(dtype="float64")
    return out


def build_features(frame: pd.DataFrame, version: str = "v1") -> pd.DataFrame:
    """Departure-row features for one month frame (departures and arrivals in).

    The returned frame is indexed like the departure rows of `frame` and carries
    MVT_ID_mvt plus every feature. Departure block time and taxi time are
    dropped before anything is computed.

    version "v2" adds the placeholder-block-time block: the `noac` marker (a
    departure with no AIRCRAFT_TYPE_mvt, whose recorded block time is often the
    schedule rather than an actual off-block), the length of the flight
    designator, and how many other noac departures the airport recorded that
    day — an isolated noac row is a placeholder, a whole day of them is a
    recording outage where the block times may be real.
    """
    df = frame.copy()
    is_dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    # A departure's own block time and taxi time are blanked here, exactly as the
    # ranking file blanks them, so no branch below can read them even by mistake.
    # Arrival block time and taxi-in are NOT blanked: they are present in the
    # ranking file for every arrival and are the only actual surface-time
    # readout the task leaves us (docs/input-rule.md).
    if "BLOCK_TIME_UTC_mvt" in df.columns:
        df.loc[is_dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    if "TAXITIME_SEC_mvt" in df.columns:
        df["TAXITIME_SEC_mvt"] = df["TAXITIME_SEC_mvt"].astype("float64")
        df.loc[is_dep, "TAXITIME_SEC_mvt"] = np.nan

    # Airport of the movement: origin for departures, destination for arrivals.
    apt_all = df["ADEP_mvt"].where(df["PHASE_mvt"] == "DEP", df["ADES_mvt"])
    apt_code_all = pd.factorize(apt_all.astype("string").fillna("\x00NA"))[0]
    t_all = _epoch(df["MVT_TIME_UTC_mvt"])

    dep = df.loc[is_dep]
    apt = apt_all[is_dep]
    apt_code = apt_code_all[is_dep]
    t_dep = t_all[is_dep]
    aobt = _epoch(dep["AOBT_3_flt"])

    out = pd.DataFrame(index=dep.index)
    out["MVT_ID_mvt"] = dep["MVT_ID_mvt"].to_numpy()

    # --- own-flight deltas, seconds ---------------------------------------
    out["d_takeoff_aobt"] = t_dep - aobt
    out["d_takeoff_lobt"] = t_dep - _epoch(dep["LOBT_flt"])
    out["d_takeoff_iobt"] = t_dep - _epoch(dep["IOBT_flt"])
    out["d_takeoff_eobt1"] = t_dep - _epoch(dep["EOBT_1_flt"])
    out["d_takeoff_sched"] = t_dep - _epoch(dep["SCHED_TIME_UTC_mvt"])
    out["d_aobt_lobt"] = aobt - _epoch(dep["LOBT_flt"])
    out["d_aobt_sched"] = aobt - _epoch(dep["SCHED_TIME_UTC_mvt"])
    out["d_lobt_sched"] = _epoch(dep["LOBT_flt"]) - _epoch(dep["SCHED_TIME_UTC_mvt"])
    out["d_eobt1_iobt"] = _epoch(dep["EOBT_1_flt"]) - _epoch(dep["IOBT_flt"])
    out["nm_missing"] = dep["FLIGHT_ID_mvt"].isna().to_numpy().astype("int8")
    out["aobt_missing"] = (~np.isfinite(aobt)).astype("int8")

    # --- categoricals ------------------------------------------------------
    out["airport"] = apt.to_numpy()
    out["airport_stand"] = (apt.astype("string") + "|" + dep["STAND_mvt"].astype("string")).to_numpy()
    out["airport_runway"] = (apt.astype("string") + "|" + dep["RUNWAY_mvt"].astype("string")).to_numpy()
    out["actype_mvt"] = dep["AIRCRAFT_TYPE_mvt"].to_numpy()
    out["actype_flt"] = dep["AIRCRAFT_TYPE_flt"].to_numpy()
    out["operator"] = dep["AIRCRAFT_OPERATOR_flt"].to_numpy()
    out["destination"] = dep["ADES_mvt"].to_numpy()
    out["market_segment"] = dep["MARKET_SEGMENT_flt"].to_numpy()
    out["flight_type"] = dep["FLIGHT_TYPE_flt"].to_numpy()
    out["wake_cat"] = dep["WK_TBL_CAT_flt"].to_numpy()
    out["flight_rule_mvt"] = dep["FLIGHT_RULE_mvt"].to_numpy()
    out["flight_rule_flt"] = dep["FLIGHT_RULE_flt"].to_numpy()

    # --- calendar ----------------------------------------------------------
    parts = _local_parts(dep["MVT_TIME_UTC_mvt"], apt)
    for c in parts.columns:
        out[c] = parts[c].to_numpy()
    out["month_num"] = dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy(dtype="float64")
    out["day_of_month"] = dep["MVT_TIME_UTC_mvt"].dt.day.to_numpy(dtype="float64")

    # --- congestion from the month frame -----------------------------------
    rwy_all = (apt_all.astype("string") + "|" + df["RUNWAY_mvt"].astype("string")).fillna("\x00NA")
    rwy_code_all = pd.factorize(rwy_all)[0]
    dep_mask = is_dep
    arr_mask = ~is_dep

    for w, tag in zip(WINDOWS, ("10m", "30m", "60m")):
        for side in ("before", "after"):
            out[f"dep_takeoffs_{tag}_{side}"] = _window_counts(
                apt_code_all[dep_mask], t_all[dep_mask], apt_code, t_dep, w, side
            )
            out[f"rwy_takeoffs_{tag}_{side}"] = _window_counts(
                rwy_code_all[dep_mask], t_all[dep_mask], rwy_code_all[is_dep], t_dep, w, side
            )
            out[f"arr_landings_{tag}_{side}"] = _window_counts(
                apt_code_all[arr_mask], t_all[arr_mask], apt_code, t_dep, w, side
            )

    # queue ahead: same-airport departures whose off-block falls between this
    # flight's own off-block and its take-off.
    qa = _between_counts(apt_code_all[dep_mask], _epoch(dep["AOBT_3_flt"]), apt_code, aobt, t_dep)
    out["queue_ahead_aobt"] = np.where(aobt <= t_dep, qa - 1.0, np.nan)  # drop self; undefined if AOBT after take-off

    if version in ("v3", "v4"):
        # Seasonality as a cycle rather than a month index. The backward january
        # fold validates on a month the model never trained on, and the ranking
        # set is January and July 2026; a raw month number is extrapolated in
        # both cases, while day-of-year sine and cosine are interpolated.
        doy = dep["MVT_TIME_UTC_mvt"].dt.dayofyear.to_numpy(dtype="float64")
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        arr = ~is_dep
        stand_all = (apt_all.astype("string") + "|" + df["STAND_mvt"].astype("string")).fillna("\x00NA")
        stand_code_all = pd.factorize(stand_all)[0]
        arr_land = t_all[arr]
        arr_block = _epoch(df.loc[arr, "BLOCK_TIME_UTC_mvt"])
        arr_taxiin = df.loc[arr, "TAXITIME_SEC_mvt"].to_numpy(dtype="float64")

        # the aircraft that last occupied this stand: its on-block bounds when a
        # real off-block could have been recorded, and its taxi-in is a live read
        # of how long the surface is taking right now.
        idx, ok = _prev_next(stand_code_all[arr], arr_block, stand_code_all[is_dep], t_dep, "prev")
        out["stand_prev_arr_block_gap"] = t_dep - _take(arr_block, idx, ok)
        out["stand_prev_arr_land_gap"] = t_dep - _take(arr_land, idx, ok)
        out["stand_prev_arr_taxiin"] = _take(arr_taxiin, idx, ok)
        idx, ok = _prev_next(stand_code_all[arr], arr_land, stand_code_all[is_dep], t_dep, "next")
        out["stand_next_arr_land_gap"] = _take(arr_land, idx, ok) - t_dep
        idx, ok = _prev_next(stand_code_all[is_dep], t_dep, stand_code_all[is_dep], t_dep, "prev")
        out["stand_prev_dep_gap"] = t_dep - _take(t_dep, idx, ok)

        # arrival taxi-in around this take-off: the surface state measured on the
        # only movements whose taxi time the ranking file still carries.
        for w, tag in zip(WINDOWS, ("10m", "30m", "60m")):
            for side in ("before", "after"):
                n, tot = _window_sum(apt_code_all[arr], arr_land, arr_taxiin, apt_code, t_dep, w, side)
                out[f"arr_taxiin_mean_{tag}_{side}"] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
            n, tot = _window_sum(rwy_code_all[arr], arr_land, arr_taxiin, rwy_code_all[is_dep], t_dep, w, "before")
            out[f"arr_taxiin_rwy_mean_{tag}"] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)

    if version in ("v2", "v3", "v4"):
        noac = dep["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
        out["noac"] = noac.astype("int8")
        out["flight_len"] = dep["FLIGHT_mvt"].astype("string").str.len().to_numpy(dtype="float64")
        day = (t_dep // 86400.0).astype("int64")
        key = pd.Series(apt_code.astype("int64") * 100000 + day)
        n_day = key.map(key[noac].value_counts()).fillna(0.0).to_numpy(dtype="float64")
        out["noac_day_airport"] = n_day
        out["noac_share_day"] = n_day / key.map(key.value_counts()).to_numpy(dtype="float64")
    out["prev_rwy_gap"] = _prev_gap(rwy_code_all[is_dep], t_dep)
    out["prev_apt_gap"] = _prev_gap(apt_code, t_dep)

    if version == "v4":
        # The tail worker's signals (prc/tail_features.py), imported rather than
        # re-derived: the airport-day aggregates of take-off minus NM off-block
        # are the outage detector my own set has no equivalent of, and the
        # multi-arrival stand history goes deeper than my single previous
        # arrival. Duplicates of features I already build are left out. The
        # already-masked frame is passed, not the caller's, so the guarantee
        # holds whatever the module does internally.
        from prc.tail_features import tail_signals

        ts = tail_signals(df, airports=None).reindex(dep.index)
        for c in TAIL_IMPORT:
            out[f"ts_{c}"] = ts[c].to_numpy(dtype="float64")

    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "MVT_ID_mvt"]
