#!/usr/bin/env python3
"""Cache one departures feature frame per 2025 month (and the two ranking months).

Every month is passed through `prc.data.mask_like_ranking` before any feature is
computed, so the cached columns are exactly what a ranking-file row can supply.
The hidden label is carried alongside in `y`, never used by a feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import prc.data as d  # noqa: E402
import prc.weather_features as wf  # noqa: E402

OUT = ROOT / "artifacts" / "exp-005-weather" / "cache"

CATEGORICALS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]
DERIVED_CATEGORICALS = ["stand_runway"]
FORBIDDEN = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Own-flight deltas, calendar, and the raw categoricals. No masked column."""
    def delta(a: str, b: str) -> pd.Series:
        return (frame[a] - frame[b]).dt.total_seconds()

    out = pd.DataFrame(index=frame.index)
    out["nm_taxi"] = delta("MVT_TIME_UTC_mvt", "AOBT_3_flt")
    out["mvt_sched"] = delta("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    out["mvt_lobt"] = delta("MVT_TIME_UTC_mvt", "LOBT_flt")
    out["mvt_eobt1"] = delta("MVT_TIME_UTC_mvt", "EOBT_1_flt")
    out["aobt_sched"] = delta("AOBT_3_flt", "SCHED_TIME_UTC_mvt")
    out["aobt_lobt"] = delta("AOBT_3_flt", "LOBT_flt")
    out["aobt_eobt1"] = delta("AOBT_3_flt", "EOBT_1_flt")
    out["lobt_iobt"] = delta("LOBT_flt", "IOBT_flt")
    out["lobt_sched"] = delta("LOBT_flt", "SCHED_TIME_UTC_mvt")
    out["arvt3_aobt"] = delta("AOBT_3_flt", "ARVT_3_flt")
    out["has_nm"] = frame["AOBT_3_flt"].notna().astype("float32")
    t = frame["MVT_TIME_UTC_mvt"]
    out["hour"] = t.dt.hour.astype("float32")
    out["dow"] = t.dt.dayofweek.astype("float32")

    # Routed by the desk from exp-002-tail: a null AIRCRAFT_TYPE_mvt ("noac")
    # marks rows whose recorded block time is often a placeholder equal to the
    # schedule. An isolated noac row on a day is a placeholder; a day-wide batch
    # of them is a real feed outage with real block times, so the per-day count
    # at the airport separates the two. Both are computed from this month's
    # frame alone, from columns the ranking file carries.
    noac = frame["AIRCRAFT_TYPE_mvt"].isna()
    out["noac"] = noac.astype("float32")
    day_key = pd.MultiIndex.from_arrays([frame["ADEP_mvt"], t.dt.floor("D")])
    out["noac_day_count"] = noac.groupby(day_key).transform("sum").to_numpy()
    out["flight_len"] = frame["FLIGHT_mvt"].astype("string").str.len().to_numpy()
    return out.astype("float32")


def build_month(month: str, hourly: pd.DataFrame, ranking: pd.DataFrame | None = None) -> pd.DataFrame:
    if ranking is None:
        full = d.load_month(month)
        masked, labels = d.mask_like_ranking(full)
    else:
        masked = ranking[ranking["month"] == month].copy()
        labels = pd.Series(np.nan, index=masked.index)
    for column in FORBIDDEN:
        assert masked.loc[masked["PHASE_mvt"] == "DEP", column].isna().all(), column

    features = pd.concat([base_features(masked), wf.transform(masked, hourly)], axis=1)
    features["MVT_ID_mvt"] = masked["MVT_ID_mvt"].to_numpy()
    features["month"] = month
    # carried for the airport-day cluster bootstrap in analyse.py; never a model input
    features["takeoff_hour_utc"] = masked["MVT_TIME_UTC_mvt"].dt.floor("h").to_numpy()
    for column in CATEGORICALS:
        features[column] = masked[column].astype("string").fillna("__NA__").to_numpy()
    # Routed by the desk from docs/shift-2026.md: stand-to-runway distance is the
    # physics of taxi-out, so the pair carries what neither column carries alone
    # (CDG lost 27R and 09L, so 13.8 % of its 2025 stand-runway routes are gone).
    features["stand_runway"] = (
        features["ADEP_mvt"] + "|" + features["STAND_mvt"] + "|" + features["RUNWAY_mvt"])
    features["y"] = labels.to_numpy()

    departures = features[(masked["PHASE_mvt"] == "DEP").to_numpy()].reset_index(drop=True)
    return departures


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hourly = wf.load_hourly()
    for month in d.MONTHS:
        frame = build_month(month, hourly)
        path = OUT / f"{month}.parquet"
        frame.to_parquet(path, index=False)
        print(f"{month}: rows={len(frame):,} y_null={frame['y'].isna().mean():.4f} "
              f"wx_missing={frame['wx_missing'].mean():.4f} -> {path}", flush=True)
    ranking = d.load_ranking()
    for month in ("2026-01", "2026-07"):
        frame = build_month(month, hourly, ranking=ranking)
        path = OUT / f"{month}.parquet"
        frame.to_parquet(path, index=False)
        print(f"{month}: rows={len(frame):,} wx_missing={frame['wx_missing'].mean():.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
