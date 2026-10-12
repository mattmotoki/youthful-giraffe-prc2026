#!/usr/bin/env python3
"""How much winter is in each month, and what it does to the model's target.

Two questions the ablation cannot answer on its own.

1. Does the validation fold contain the weather the lane exists to measure?
   The scored months are January and July 2026. December 2025 is the winter
   primary fold, so this prints snow and freezing-precipitation exposure per
   airport-month for the candidate folds and for the two ranking months.

2. Does weather move the quantity the model actually has to predict? Once
   AOBT_3_flt is a feature, the learnable part of taxi is the clock offset
   `delta = BLOCK_TIME - AOBT_3`, not the physical taxi duration. If weather
   moved taxi but left delta alone, weather could not help the scored task.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import prc.data as d  # noqa: E402
import prc.weather_features as wf  # noqa: E402

CACHE = ROOT / "artifacts" / "exp-005-weather" / "cache"
WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]


def exposure() -> pd.DataFrame:
    hourly = wf.load_hourly()
    hourly["ym"] = hourly["hour_utc"].dt.strftime("%Y-%m")
    months = ["2025-01", "2025-02", "2025-07", "2025-11", "2025-12", "2026-01", "2026-07"]
    sub = hourly[hourly["ym"].isin(months)]
    return sub.groupby(["ym", "airport"]).agg(
        snow_hours=("snowing", "mean"),
        snow_mm=("snow_mm_we", "sum"),
        freezing_hours=("freezing_precip", "mean"),
        subzero_hours=("subzero", "mean"),
        lowvis_hours=("low_visibility", "mean"),
    )


def target_shift() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["ADEP_mvt", "y", "nm_taxi", "has_nm", "wx_snow_6h_mm",
               "wx_freezing_precip", "month"]
    frame = pd.concat([pd.read_parquet(CACHE / f"{m}.parquet", columns=columns)
                       for m in WINTER_MONTHS], ignore_index=True)
    frame = frame[frame["has_nm"] > 0.5].copy()
    # delta = BLOCK - AOBT_3 = nm_taxi - taxi: what the anchored model predicts.
    frame["delta"] = frame["nm_taxi"] - frame["y"]
    frame["bad_wx"] = ((frame["wx_snow_6h_mm"].fillna(0) > 0.1)
                       | (frame["wx_freezing_precip"].fillna(0) > 0.5))
    overall = frame.groupby("bad_wx").agg(
        n=("y", "size"), taxi_mean=("y", "mean"), taxi_median=("y", "median"),
        delta_mean=("delta", "mean"), delta_sd=("delta", "std"))
    per_airport = frame.groupby(["ADEP_mvt", "bad_wx"]).agg(
        n=("y", "size"), taxi_mean=("y", "mean"),
        delta_mean=("delta", "mean"), delta_sd=("delta", "std")).unstack()
    return overall, per_airport


def main() -> None:
    pd.set_option("display.width", 220, "display.max_rows", 300)
    print("=== 1. winter exposure per airport-month (fraction of hours, mm) ===")
    print(exposure().round(3).to_string())
    overall, per_airport = target_shift()
    print("\n=== 2. effect on taxi and on the model's target, winter 2025 NM rows ===")
    print(overall.round(1).to_string())
    print("\nper airport (False = other hours, True = snow or freezing hours):")
    print(per_airport.round(1).to_string())


if __name__ == "__main__":
    main()
