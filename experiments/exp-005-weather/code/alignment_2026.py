#!/usr/bin/env python3
"""Do the anomalous January 2026 days line up with the January 2026 weather?

`docs/shift-2026.md` found that daily mean take-off minus AOBT in January 2026
sits above the whole 2025 range on 9 days at EDDF and 9 at EHAM. That statistic
is computed from two columns the ranking file carries, so it is observable
without the hidden label. If those days are the snow and freezing days, a
weather feature can tell the model they are different when no statistic fitted
on 2025 alone can, because no 2025 month contains them. Weather is not the ONLY
such input: within-month airport-day aggregates of MVT - AOBT_3 are equally
observable in the ranking file and measure the same disruption directly. An
earlier version of this docstring claimed weather was the only one; withdrawn
(Sol review 2026-09-05, finding 8).

Test: rank each 2026 ranking day inside its airport's 2025 distribution of daily
mean take-off minus AOBT, then compare the weather of the days above the 2025
maximum against the rest of the same month.
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

COLUMNS = ["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]


def daily(frame: pd.DataFrame) -> pd.DataFrame:
    dep = frame[(frame["PHASE_mvt"] == "DEP") & frame["AOBT_3_flt"].notna()].copy()
    dep["nm_taxi"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dep["date"] = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    return dep.groupby(["ADEP_mvt", "date"]).agg(
        n=("nm_taxi", "size"), nm_mean=("nm_taxi", "mean"), nm_std=("nm_taxi", "std"))


def weather_daily() -> pd.DataFrame:
    hourly = wf.load_hourly()
    hourly["date"] = hourly["hour_utc"].dt.floor("D")
    return hourly.groupby(["airport", "date"]).agg(
        snow_mm=("snow_mm_we", "sum"),
        snow_hours=("snowing", "sum"),
        freezing_hours=("freezing_precip", "sum"),
        subzero_hours=("subzero", "sum"),
        lowvis_hours=("low_visibility", "sum"),
        precip_mm=("precip_mm", "sum"),
        wind_max=("wind_speed_ms", "max"),
        temp_min=("temp_c", "min"),
    ).reset_index().rename(columns={"airport": "ADEP_mvt"})


def main() -> None:
    pd.set_option("display.width", 240, "display.max_rows", 400)
    train = d.load_months(d.MONTHS, COLUMNS)
    rank = d.load_ranking(COLUMNS)
    print(f"training departures with NM join: {len(train):,} rows read; "
          f"ranking rows read: {len(rank):,}")

    d2025 = daily(train).reset_index()
    d2026 = daily(rank).reset_index()
    envelope = d2025.groupby("ADEP_mvt")["nm_mean"].agg(["max", "min", "mean",
                                                        lambda s: s.quantile(0.95)])
    envelope.columns = ["max2025", "min2025", "mean2025", "p95_2025"]

    frame = d2026.merge(envelope, on="ADEP_mvt").merge(weather_daily(), on=["ADEP_mvt", "date"])
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    frame["above_2025_max"] = frame["nm_mean"] > frame["max2025"]
    frame["above_2025_p95"] = frame["nm_mean"] > frame["p95_2025"]

    print("\n=== days above the 2025 maximum, per airport and ranking month ===")
    counts = frame.groupby(["ADEP_mvt", "month"])["above_2025_max"].agg(["sum", "size"])
    print(counts.to_string())

    jan = frame[frame["month"] == "2026-01"]
    print("\n=== January 2026: weather of the days above the 2025 maximum vs the rest ===")
    weather_columns = ["snow_mm", "snow_hours", "freezing_hours", "subzero_hours",
                       "lowvis_hours", "precip_mm", "wind_max", "temp_min"]
    summary = jan.groupby("above_2025_max")[weather_columns + ["nm_mean", "nm_std"]].mean()
    print(summary.round(2).to_string())

    print("\n=== the anomalous days themselves (EDDF, EHAM, EDDM, LSZH) ===")
    focus = jan[jan["ADEP_mvt"].isin(["EDDF", "EHAM", "EDDM", "LSZH"]) & jan["above_2025_max"]]
    show = ["ADEP_mvt", "date", "n", "nm_mean", "nm_std", "max2025", "snow_mm",
            "snow_hours", "freezing_hours", "subzero_hours", "lowvis_hours", "wind_max"]
    print(focus[show].sort_values(["ADEP_mvt", "date"]).to_string(index=False,
          float_format=lambda v: f"{v:8.1f}"))

    print("\n=== rank correlation of daily dispersion with daily weather, January 2026 ===")
    rows = []
    for airport, group in jan.groupby("ADEP_mvt"):
        if len(group) < 20:
            continue
        entry = {"airport": airport, "days": len(group)}
        for column in ["snow_mm", "freezing_hours", "subzero_hours", "lowvis_hours", "wind_max"]:
            if group[column].std(skipna=True) and group[column].notna().sum() > 10:
                entry[f"r_mean_{column}"] = group["nm_mean"].corr(group[column], method="spearman")
                entry[f"r_std_{column}"] = group["nm_std"].corr(group[column], method="spearman")
        rows.append(entry)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    out = ROOT / "experiments" / "exp-005-weather" / "logs" / "alignment_2026.csv"
    frame.to_csv(out, index=False)
    print(f"\nper-day table: {out}")


if __name__ == "__main__":
    main()
