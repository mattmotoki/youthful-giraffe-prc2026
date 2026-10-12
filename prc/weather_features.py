"""Event-hour ERA5 weather at the airport point, for PRC 2026 taxi-out.

Two layers.

`build_hourly()` turns the frozen Copernicus CDS NetCDF snapshot
(`data/external/weather/copernicus_era5/raw`, one core time-series file and
three parent-product files per airport) into one tidy hourly table with one row
per (airport, UTC hour) and every derived airport-level quantity already
computed: unit conversions, wind speed and meteorological direction, rolling
precipitation and snow, and 3-hour changes.

`transform(month_frame)` joins that table to a movement frame's departure rows
at the take-off clock hour and at the actual off-block clock hour, and adds the
per-row quantities that need a flight column: runway-aligned head- and
crosswind, and the take-off-minus-off-block weather change across the taxi.

Time rule. Instantaneous ERA5 variables (2 m temperature and dew point, mean
sea-level pressure, 10 m wind components, total and low cloud cover) are valid
*at* their timestamp. Accumulated variables (total precipitation, snowfall) are
the accumulation over the hour *ending* at their timestamp. The hourly table is
keyed by clock-hour start `H`: instantaneous columns carry the analysis at `H`,
accumulated columns carry the accumulation over `[H, H+1)`, i.e. the raw value
stamped `H+1`. A departure at 13:42 therefore joins to `H = 13:00` for both
kinds and gets the precipitation that actually fell while it was taxiing.

Leakage. Nothing here reads `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. Weather
is a function of (airport, clock hour) only; the flight-dependent inputs are
`MVT_TIME_UTC_mvt`, `AOBT_3_flt` and `RUNWAY_mvt`, all present in the ranking
file (`docs/input-rule.md`). Provenance: `docs/weather-provenance.md`.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 2
"""Behaviour version of the derived hourly table. Pass `version=` to build_hourly
to reach an older behaviour; the default is always the current one.

2 (2026-09-05, Sol review): rolling accumulation windows (`snow_3h_mm`,
  `snow_6h_mm`, `precip_3h_mm`, `precip_6h_mm`) require the FULL window and
  return NaN otherwise. Under version 1 they used `min_periods=1`, so a window
  containing a missing hour returned the sum of the hours that were present --
  the last hour of each ranking month reported a non-null 3/6-hour total built
  from a null accumulation.
1 (2026-09-05, commit 38ef8e3): as version 2 but `min_periods=1` on those four
  columns. This is the ONLY difference; version 1 reproduces 38ef8e3 exactly.

NaN-propagating condition flags (freezing_precip, snowing, subzero,
low_visibility, snow_6h_any) and NaN-propagating `liquid_mm` are NOT version
dependent: they were fixed before 38ef8e3 and no released version reads a
missing input as False.

Consumers: exp-002-tail (ledger row exp-002 it12 measured this module and did
not adopt it), exp-001-retro-v1 (integrating WINTER_CORE). Both should pin
`version=` if they depend on a specific behaviour.
"""

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HOURLY_SHA256 = "8238378ec1f6d66a6c59712993ec54ac72af3f130cb7a8a7754a3fbec369f293"
EXPECTED_HOURLY_BYTES = 15_181_053


def _snapshot_dir() -> Path:
    """The frozen CDS snapshot: this repo's data/external, else the sibling checkout."""
    tail = Path("weather") / "copernicus_era5"
    for base in (ROOT / "data" / "external",
                 ROOT.parent / "prc-data-challenge-2026" / "data" / "external"):
        if (base / tail / "manifest.json").is_file():
            return base / tail
    raise FileNotFoundError("no Copernicus ERA5 snapshot found under data/external")


HOURLY_PATH = ROOT / "artifacts" / "weather" / "era5_hourly.parquet"
RELEASED_HOURLY_PATH = ROOT / "external-data" / "copernicus-era5-v32" / "era5_hourly.parquet"

RANKING_AIRPORTS = (
    "EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
    "LEMD", "LFPG", "LIRF", "LSZH", "LTFM",
)
EXTRA_LABELS = ("2024", "2025", "2026_01_07")

# ERA5 short names as they appear in the retrieved NetCDF files.
CORE_VARS = ("u10", "v10", "d2m", "t2m", "msl", "tcc", "tp")
EXTRA_VARS = ("lcc", "sf")
ACCUMULATED = ("tp", "sf")  # accumulation over the hour ENDING at the stamp


# --------------------------------------------------------------------------
# layer 1: the hourly airport-point table
# --------------------------------------------------------------------------

def _read_point(path: Path, variables: tuple[str, ...]) -> pd.DataFrame:
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4") as ds:
        frame = ds[list(variables)].to_dataframe().reset_index()
    time_col = "valid_time" if "valid_time" in frame.columns else "time"
    frame = frame.rename(columns={time_col: "valid_time"})
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    return frame[["valid_time", *variables]].sort_values("valid_time")


def _raw_airport(airport: str, snapshot: Path) -> pd.DataFrame:
    raw = snapshot / "raw"
    core = _read_point(raw / f"{airport}_core_2024-01-01_2026-07-31.nc", CORE_VARS)
    extras = [
        _read_point(raw / f"{airport}_extra_{label}.nc", EXTRA_VARS)
        for label in EXTRA_LABELS
        if (raw / f"{airport}_extra_{label}.nc").is_file()
    ]
    if extras:
        extra = pd.concat(extras, ignore_index=True).drop_duplicates("valid_time")
        frame = core.merge(extra, on="valid_time", how="left")
    else:
        frame = core.assign(lcc=np.nan, sf=np.nan)
    frame.insert(0, "airport", airport)
    return frame


def _derive(frame: pd.DataFrame, version: int = VERSION) -> pd.DataFrame:
    """Units, relabelled accumulations, wind, rolling windows, 3-hour changes."""
    frame = frame.sort_values("valid_time").reset_index(drop=True)

    # Accumulated variables describe the hour ENDING at valid_time; relabel them
    # to the hour they cover so one clock-hour key serves every column.
    hourly = frame.set_index("valid_time")
    shifted = hourly[list(ACCUMULATED)].shift(-1)  # value stamped H+1 -> row H
    out = pd.DataFrame(index=hourly.index)
    out["temp_c"] = hourly["t2m"] - 273.15
    out["dewpoint_c"] = hourly["d2m"] - 273.15
    out["dewpoint_depression_c"] = out["temp_c"] - out["dewpoint_c"]
    out["msl_hpa"] = hourly["msl"] / 100.0
    u, v = hourly["u10"], hourly["v10"]
    out["wind_speed_ms"] = np.sqrt(u**2 + v**2)
    out["wind_dir_deg"] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    out["cloud_total"] = hourly["tcc"]
    out["cloud_low"] = hourly["lcc"]
    out["precip_mm"] = shifted["tp"] * 1000.0
    out["snow_mm_we"] = shifted["sf"] * 1000.0
    out["liquid_mm"] = (out["precip_mm"] - out["snow_mm_we"]).clip(lower=0.0)

    # Flags and proxies.
    # A flag whose input is missing is unknown, not false: keep it NaN so the
    # trees see "no snowfall retrieved here" as a gap and not as "dry".
    def flag(condition: pd.Series, *sources: pd.Series) -> pd.Series:
        known = sources[0].notna()
        for source in sources[1:]:
            known &= source.notna()
        return condition.astype("float64").where(known)

    out["freezing_precip"] = flag(
        (out["temp_c"] <= 2.0) & (out["precip_mm"] > 0.1), out["temp_c"], out["precip_mm"])
    out["snowing"] = flag(out["snow_mm_we"] > 0.05, out["snow_mm_we"])
    out["cloud_base_proxy_m"] = 125.0 * out["dewpoint_depression_c"].clip(lower=0.0)
    out["low_visibility"] = flag(
        (out["dewpoint_depression_c"] < 2.0) & (out["cloud_low"] > 0.8),
        out["dewpoint_depression_c"], out["cloud_low"])
    out["subzero"] = flag(out["temp_c"] <= 0.0, out["temp_c"])

    # Rolling windows on the relabelled accumulations: the 3 hours ending with
    # (and including) the joined hour.
    # `min_periods=window`: a window containing a missing hour is unknown, not a
    # partial sum. Under min_periods=1 the last snapshot hour reported a non-null
    # rolling total built from a null accumulation. Sol review 2026-09-05, f.6.
    # Convention: the `window` hours ENDING WITH AND INCLUDING the joined clock
    # hour -- accumulation over [H-window+1, H+1), not the hours strictly before H.
    least = 1 if version < 2 else None  # None -> pandas requires the full window
    for window in (3, 6):
        periods = least if least is not None else window
        out[f"snow_{window}h_mm"] = out["snow_mm_we"].rolling(window, min_periods=periods).sum()
        out[f"precip_{window}h_mm"] = out["precip_mm"].rolling(window, min_periods=periods).sum()
    out["snow_6h_any"] = flag(out["snow_6h_mm"] > 0.1, out["snow_6h_mm"])

    # 3-hour changes.
    for column in ("temp_c", "msl_hpa", "wind_speed_ms", "cloud_low", "dewpoint_depression_c"):
        out[f"d3_{column}"] = out[column] - out[column].shift(3)

    out.insert(0, "airport", frame["airport"].to_numpy())
    return out.reset_index().rename(columns={"valid_time": "hour_utc"})


def build_hourly(
    airports: tuple[str, ...] = RANKING_AIRPORTS,
    snapshot: Path | None = None,
    out_path: Path | None = HOURLY_PATH,
    version: int = VERSION,
) -> pd.DataFrame:
    """Tidy hourly table for the ranking airports; writes parquet when asked.

    `version` selects the derived-column behaviour; see the VERSION docstring.
    """
    # Resolve the raw snapshot only on the rebuild route.  Importing this module
    # must work for a release reader who has the verified derived table only.
    snapshot = _snapshot_dir() if snapshot is None else Path(snapshot)
    parts = [_derive(_raw_airport(airport, snapshot), version) for airport in airports]
    table = pd.concat(parts, ignore_index=True)
    # The last hour of the snapshot has no H+1 stamp, so its accumulation is
    # unknown rather than zero; leave it NaN and let the join flag it.
    floats = [c for c in table.columns if c not in ("airport", "hour_utc")]
    table[floats] = table[floats].astype("float32")
    table.attrs["weather_features_version"] = version
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out_path, index=False)
    return table


def _default_hourly_path() -> Path:
    """Prefer the local cache, then the separately licensed release bundle."""
    override = os.environ.get("PRC_ERA5_HOURLY")
    candidates = ([Path(override)] if override else []) + [HOURLY_PATH, RELEASED_HOURLY_PATH]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "exact ERA5 hourly table absent; import the v32 CC BY data bundle "
        "with release/era5_data.py pack"
    )


def load_hourly(path: Path | None = None) -> pd.DataFrame:
    path = _default_hourly_path() if path is None else Path(path)
    if path.stat().st_size != EXPECTED_HOURLY_BYTES:
        raise ValueError(f"ERA5 hourly table has wrong size: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_HOURLY_SHA256:
        raise ValueError(f"ERA5 hourly table has wrong SHA-256: {path}")
    table = pd.read_parquet(path)
    table["hour_utc"] = pd.to_datetime(table["hour_utc"], utc=True)
    return table


# --------------------------------------------------------------------------
# layer 2: the per-departure join
# --------------------------------------------------------------------------

_RUNWAY_RE = re.compile(r"^\s*(\d{1,2})\s*([LCR]?)\s*$")

# Columns joined at the take-off hour (everything) and at the off-block hour.
TAKEOFF_COLUMNS = (
    "temp_c", "dewpoint_c", "dewpoint_depression_c", "msl_hpa",
    "wind_speed_ms", "wind_dir_deg", "cloud_total", "cloud_low",
    "precip_mm", "snow_mm_we", "liquid_mm",
    "freezing_precip", "snowing", "cloud_base_proxy_m", "low_visibility", "subzero",
    "snow_3h_mm", "snow_6h_mm", "precip_3h_mm", "precip_6h_mm", "snow_6h_any",
    "d3_temp_c", "d3_msl_hpa", "d3_wind_speed_ms", "d3_cloud_low",
    "d3_dewpoint_depression_c",
)
AOBT_COLUMNS = (
    "temp_c", "wind_speed_ms", "wind_dir_deg", "precip_mm", "snow_mm_we",
    "cloud_low", "dewpoint_depression_c", "freezing_precip", "snow_3h_mm",
)
# Feature families, for ablation.
FAMILIES = {
    "basic": ["wx_temp_c", "wx_dewpoint_c", "wx_msl_hpa", "wx_wind_speed_ms",
              "wx_wind_dir_deg", "wx_cloud_total"],
    "precip": ["wx_precip_mm", "wx_snow_mm_we", "wx_liquid_mm", "wx_freezing_precip",
               "wx_snowing", "wx_subzero", "wx_snow_3h_mm", "wx_snow_6h_mm",
               "wx_precip_3h_mm", "wx_precip_6h_mm", "wx_snow_6h_any"],
    "runway_wind": ["wx_headwind_ms", "wx_crosswind_ms", "wx_crosswind_abs_ms",
                    "wx_tailwind_ms", "wx_runway_heading_deg"],
    "visibility": ["wx_dewpoint_depression_c", "wx_cloud_low", "wx_cloud_base_proxy_m",
                   "wx_low_visibility"],
    "change": ["wx_d3_temp_c", "wx_d3_msl_hpa", "wx_d3_wind_speed_ms",
               "wx_d3_cloud_low", "wx_d3_dewpoint_depression_c"],
    "aobt": ["wxo_temp_c", "wxo_wind_speed_ms", "wxo_wind_dir_deg", "wxo_precip_mm",
             "wxo_snow_mm_we", "wxo_cloud_low", "wxo_dewpoint_depression_c",
             "wxo_freezing_precip", "wxo_snow_3h_mm", "wx_taxi_d_temp_c",
             "wx_taxi_d_wind_speed_ms", "wx_taxi_precip_mm", "wx_missing_aobt_hour"],
}
ALL_FEATURES = [c for family in FAMILIES.values() for c in family] + ["wx_missing"]

# The delivered feature sets (exp-005-weather RESULTS section 7). Import one of
# these rather than re-typing it; experiments/exp-005-weather/code/ablate.py
# imports them too, so the delivered list and the measured list cannot drift.
#
#   WINTER_CORE      7 columns. The set to integrate. Three tested sets lower
#                    seed-0 overall RMSE on all three folds -- winter_core
#                    (-3.52 / -12.73 / -1.91), precip_vis (15 cols, -4.40 /
#                    -7.23 / -1.27) and aobt (13 cols, -0.18 / -2.31 / -1.75).
#                    Of those, winter_core is the smallest, has much the largest
#                    december gain, and is the only one whose bulk gain was put
#                    through an airport-day cluster bootstrap on all three folds
#                    and excluded zero each time (-4.42 / -2.05 / -2.72 s).
#                    It does NOT dominate the 44-column block: `all` has the
#                    better july bulk (251.50 vs 252.41) and january <=3600
#                    (223.78 vs 228.02). The case is robustness and cost.
#   WINTER_CORE_PLUS 12 columns. Best january overall measured (376.71) but
#                    worse on july (520.28): January-specific models only.
WINTER_CORE = [
    "wx_temp_c",                  # de-icing and ground-crew regime
    "wx_precip_mm",               # rain or snow falling in the taxi hour
    "wx_snow_6h_mm",              # accumulation on the surface, 6 h to the hour
    "wx_freezing_precip",         # temp <= 2 C with precipitation
    "wx_subzero",                 # temp <= 0 C
    "wx_dewpoint_depression_c",   # visibility proxy
    "wx_wind_speed_ms",           # runway-configuration driver
]
WINTER_CORE_PLUS = WINTER_CORE + [
    "wx_cloud_low",               # the other half of the visibility proxy
    "wx_snow_3h_mm",              # recent accumulation, sharper than 6 h
    "wx_crosswind_abs_ms",        # runway-aligned, needs RUNWAY_mvt
    "wx_headwind_ms",             # runway-aligned, needs RUNWAY_mvt
    "wx_msl_hpa",                 # synoptic state
]


def runway_heading(runway: pd.Series) -> pd.Series:
    """Magnetic heading in degrees from a runway designator ('26L' -> 260).

    Non-numeric designators ('NA', 'H' for the LSZH helipad) become NaN.
    """
    digits = runway.astype("string").str.extract(_RUNWAY_RE, expand=True)[0]
    number = pd.to_numeric(digits, errors="coerce")
    number = number.where((number >= 1) & (number <= 36))
    return (number * 10.0).astype("float64")


def transform(
    month_frame: pd.DataFrame,
    hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Weather features for every row of `month_frame`, in its own row order.

    `month_frame` is a movement frame as `prc.data` returns it (masked or not:
    no masked column is read). Arrival rows get the same airport-hour weather
    as departures; callers that only model departures can ignore them.
    """
    if hourly is None:
        hourly = load_hourly()
    frame = month_frame

    takeoff = pd.DataFrame(index=frame.index)
    takeoff["airport"] = frame["ADEP_mvt"].to_numpy()
    takeoff["hour_utc"] = frame["MVT_TIME_UTC_mvt"].dt.floor("h").to_numpy()

    wx = hourly[["airport", "hour_utc", *TAKEOFF_COLUMNS]]
    joined = takeoff.merge(wx, on=["airport", "hour_utc"], how="left", sort=False)
    joined.index = frame.index
    out = joined[list(TAKEOFF_COLUMNS)].add_prefix("wx_")

    offblock = pd.DataFrame(index=frame.index)
    offblock["airport"] = frame["ADEP_mvt"].to_numpy()
    offblock["hour_utc"] = frame["AOBT_3_flt"].dt.floor("h").to_numpy()
    wxo = hourly[["airport", "hour_utc", *AOBT_COLUMNS]]
    joined_o = offblock.merge(wxo, on=["airport", "hour_utc"], how="left", sort=False)
    joined_o.index = frame.index
    out_o = joined_o[list(AOBT_COLUMNS)].add_prefix("wxo_")
    out = pd.concat([out, out_o], axis=1)

    # Runway-aligned wind: meteorological direction is the direction the wind
    # blows FROM, so the angle to the runway heading is (wind_dir - heading).
    heading = runway_heading(frame["RUNWAY_mvt"])
    angle = np.radians(out["wx_wind_dir_deg"] - heading)
    out["wx_runway_heading_deg"] = heading
    out["wx_headwind_ms"] = out["wx_wind_speed_ms"] * np.cos(angle)
    out["wx_crosswind_ms"] = out["wx_wind_speed_ms"] * np.sin(angle)
    out["wx_crosswind_abs_ms"] = out["wx_crosswind_ms"].abs()
    out["wx_tailwind_ms"] = (-out["wx_headwind_ms"]).clip(lower=0.0)

    # Weather change across the taxi itself.
    out["wx_taxi_d_temp_c"] = out["wx_temp_c"] - out["wxo_temp_c"]
    out["wx_taxi_d_wind_speed_ms"] = out["wx_wind_speed_ms"] - out["wxo_wind_speed_ms"]
    out["wx_taxi_precip_mm"] = out[["wx_precip_mm", "wxo_precip_mm"]].max(axis=1)
    out["wx_missing_aobt_hour"] = out["wxo_temp_c"].isna().astype("float32")
    out["wx_missing"] = out["wx_temp_c"].isna().astype("float32")

    return out[ALL_FEATURES].astype("float32")
