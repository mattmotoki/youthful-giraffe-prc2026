#!/usr/bin/env python3
"""Check the weather join against the raw NetCDF, by hand, on real departures.

Three things can silently go wrong and all three would manufacture signal:
the accumulation relabel (a one-hour shift on precipitation), the unit
conversions, and the runway-aligned wind convention. Each is checked here
against values read straight out of the source file.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import prc.data as d  # noqa: E402
import prc.weather_features as wf  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=None)
    args = parser.parse_args()
    month = d.load_month("2025-01")
    masked, _ = d.mask_like_ranking(month)
    hourly = wf.load_hourly()
    features = wf.transform(masked, hourly)

    dep = masked["PHASE_mvt"] == "DEP"
    edd = dep & (masked["ADEP_mvt"] == "EDDM")
    # a departure in an hour with snow on the ground, for a non-trivial check
    snowy = features["wx_snow_mm_we"].fillna(0) > 0.2
    idx = masked.index[edd & snowy][5]
    row = masked.loc[idx]
    takeoff = row["MVT_TIME_UTC_mvt"]
    hour = takeoff.floor("h")
    print(f"probe: EDDM departure {row['MVT_ID_mvt']:.0f} take-off {takeoff} -> hour {hour}\n")

    raw = (args.snapshot.resolve() if args.snapshot else wf._snapshot_dir()) / "raw"
    # the NetCDF time axis is tz-naive UTC; drop the tz to index it
    naive = hour.tz_localize(None)
    stamps = [naive, naive + pd.Timedelta("1h")]
    with xr.open_dataset(raw / "EDDM_core_2024-01-01_2026-07-31.nc") as ds:
        core = ds.sel(valid_time=stamps)
        t2m = float(core["t2m"].values[0])
        u10, v10 = float(core["u10"].values[0]), float(core["v10"].values[0])
        tp_at_h = float(core["tp"].values[0])
        tp_at_h1 = float(core["tp"].values[1])
    with xr.open_dataset(raw / "EDDM_extra_2025.nc") as ds:
        extra = ds.sel(valid_time=stamps)
        sf_at_h1 = float(np.ravel(extra["sf"].values[1])[0])

    got = features.loc[idx]
    check("temperature K->C at the take-off hour",
          np.isclose(got["wx_temp_c"], t2m - 273.15, atol=1e-3),
          f"joined {got['wx_temp_c']:.3f} vs raw {t2m - 273.15:.3f}")
    check("precipitation is the accumulation over [H, H+1), not (H-1, H]",
          np.isclose(got["wx_precip_mm"], tp_at_h1 * 1000.0, atol=1e-4)
          and not np.isclose(tp_at_h1, tp_at_h, atol=1e-9),
          f"joined {got['wx_precip_mm']:.4f} mm; raw stamped H+1 {tp_at_h1 * 1000:.4f}, "
          f"stamped H {tp_at_h * 1000:.4f}")
    check("snowfall uses the same relabel",
          np.isclose(got["wx_snow_mm_we"], sf_at_h1 * 1000.0, atol=1e-4),
          f"joined {got['wx_snow_mm_we']:.4f} mm vs raw H+1 {sf_at_h1 * 1000:.4f}")

    speed = np.hypot(u10, v10)
    direction = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0
    check("wind speed from u,v",
          np.isclose(got["wx_wind_speed_ms"], speed, atol=1e-3),
          f"joined {got['wx_wind_speed_ms']:.3f} vs {speed:.3f} m/s")
    check("wind direction is the direction the wind blows FROM",
          np.isclose(got["wx_wind_dir_deg"], direction, atol=1e-2),
          f"joined {got['wx_wind_dir_deg']:.2f} deg vs {direction:.2f}")

    heading = float(str(row["RUNWAY_mvt"])[:2]) * 10.0
    angle = np.radians(direction - heading)
    check("runway heading from the designator",
          np.isclose(got["wx_runway_heading_deg"], heading),
          f"runway {row['RUNWAY_mvt']} -> {got['wx_runway_heading_deg']:.0f} deg")
    check("headwind = speed * cos(wind_from - heading)",
          np.isclose(got["wx_headwind_ms"], speed * np.cos(angle), atol=1e-3),
          f"joined {got['wx_headwind_ms']:.3f} vs {speed * np.cos(angle):.3f} m/s")
    check("crosswind = speed * sin(wind_from - heading)",
          np.isclose(got["wx_crosswind_ms"], speed * np.sin(angle), atol=1e-3),
          f"joined {got['wx_crosswind_ms']:.3f} vs {speed * np.sin(angle):.3f} m/s")

    # A pure-headwind sanity case: wind straight down the runway is all headwind.
    probe = pd.Series({"wx_wind_speed_ms": 10.0, "wx_wind_dir_deg": 260.0})
    a = np.radians(probe["wx_wind_dir_deg"] - 260.0)
    check("wind straight down runway 26 is pure headwind",
          np.isclose(10.0 * np.cos(a), 10.0) and np.isclose(10.0 * np.sin(a), 0.0, atol=1e-9))

    # The off-block join must use AOBT, not take-off.
    with_nm = masked.index[dep & masked["AOBT_3_flt"].notna()]
    sample = with_nm[(masked.loc[with_nm, "AOBT_3_flt"].dt.floor("h")
                      != masked.loc[with_nm, "MVT_TIME_UTC_mvt"].dt.floor("h"))][0]
    aobt_hour = masked.loc[sample, "AOBT_3_flt"].floor("h")
    airport = masked.loc[sample, "ADEP_mvt"]
    expected = hourly[(hourly["airport"] == airport) & (hourly["hour_utc"] == aobt_hour)]
    check("off-block join uses the AOBT hour, not the take-off hour",
          np.isclose(features.loc[sample, "wxo_temp_c"], float(expected["temp_c"].iloc[0]), atol=1e-3),
          f"{airport} {aobt_hour}")

    # No-NM rows must not silently borrow the take-off hour.
    no_nm = masked.index[dep & masked["AOBT_3_flt"].isna()]
    check("rows with no NM join have no off-block weather",
          bool(features.loc[no_nm, "wxo_temp_c"].isna().all())
          and bool((features.loc[no_nm, "wx_missing_aobt_hour"] == 1).all()),
          f"{len(no_nm)} rows")

    # Row order and coverage.
    check("transform preserves the frame's index and row count",
          features.index.equals(masked.index) and len(features) == len(masked))
    check("every departure at a ranking airport found its take-off hour",
          bool(features.loc[dep, "wx_temp_c"].notna().all()))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all join checks passed")


if __name__ == "__main__":
    main()
