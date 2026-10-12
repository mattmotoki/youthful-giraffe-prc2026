"""Paths, hashes and the arm lists of exp-159 (the with-clock head as a direct R4 member).

Runs on the workstation. Everything outside `WORK` is read only.

exp-190 copy (PROVENANCE.md): the type code of the HB3u frame (`x_actype_code`, `x_actype_common`) comes from
the frozen table `actype_table.json` of the released twelve-month frame; nothing else changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP = Path(os.environ.get("PRC_COMP_ROOT", os.environ.get("PRC_COMP_ROOT", "/path/to/competitions")))
WORK = Path(os.environ.get("EXP159_WORK", COMP / "prc-exp159-work"))
SOURCE = WORK / "source"                      # prc/, pilot.py and the frozen rows at exp148-short-route ec40e740
#: exp-186 (final file): each input path can be set by its environment variable; the defaults are the paths of
#: the ranking-mode build.
W148 = Path(os.environ.get("EXP148_WORK", COMP / "prc-exp148-work"))
RAW = Path(os.environ.get("EXP159_RAW", COMP / "prc-data-challenge-2026/data/raw"))
FEATURES = Path(os.environ.get("EXP159_FEATURES", W148 / "inputs/features-r4c.parquet"))
PROXY = Path(os.environ.get("EXP159_PROXY", W148 / "inputs/proxy-features-r4c.parquet"))
OPDI = Path(os.environ.get("EXP159_OPDI", os.path.join(os.environ.get("PRC_OPDI_ROOT", "/path/to/prc-exp155-opdi"), "r2/inputs/complete/opdi-first.parquet")))
#: "9": the nine block months of valid's tables. "12": with April, May and September 2025 (frame12.py)
SET = os.environ.get("EXP159_SET", "9")
if SET not in ("9", "12"):
    raise SystemExit(f"EXP159_SET={SET}: 9 or 12")
_S = "12" if SET == "12" else ""
FRAME = WORK / f"inputs/frame{_S}.parquet"
TRACK = WORK / f"inputs/track{_S}.parquet"
EXTRA = WORK / "inputs/extra.parquet"          # queue counts and slot proxies of valid's R4 tables (extra.py)
WEATHER = WORK / f"inputs/weather{_S}.parquet"  # ERA5 airport-hour weather for the frame's rows (weather.py)
FEATURES5 = WORK / "inputs/features-r5.parquet"  # row features of the three new months (features12.py, built on the Mac)
ERA5 = Path(os.environ.get("EXP159_ERA5", COMP / "prc-exp148-src/artifacts/weather/era5_hourly.parquet"))   # the frozen v22 table (docs/weather-provenance.md)

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "experiments/exp-147-foldserve/code"))
os.environ.setdefault("PRC_VALIDATION_DIR", str(SOURCE / "experiments/exp-146-validation/frozen"))
os.environ.setdefault("PRC_DATA_DIR", str(RAW))

ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
MONTHS_NEW = ("2025-04", "2025-05", "2025-09")
MONTHS_2025 = ("2025-01", "2025-02", "2025-03", "2025-06", "2025-07", "2025-08", "2025-10", "2025-11", "2025-12") \
    + (MONTHS_NEW if SET == "12" else ())
#: exp-186: $PRC_RANKING_MODE selects the 2026 population (prc/data.py RANKING_SPECS). The ranking mode keeps the
#: literal months and file. The final mode reads them from the SOURCE tree's prc/data.py, which must have the
#: ranking modes (VERSION >= 7), and takes the final file from RAW (RAW/final/ or RAW) by its SHA-256.
RANKING_MODE = os.environ.get("PRC_RANKING_MODE") or "ranking"
if RANKING_MODE == "ranking":
    MONTHS_2026 = ("2026-01", "2026-07")
    RANKING = RAW / "ranking.parquet"
    N_DEPARTURES = None
else:
    from prc import data as _D        # noqa: E402  (the SOURCE tree's prc)
    if not hasattr(_D, "rank_months"):
        raise SystemExit(f"PRC_RANKING_MODE={RANKING_MODE}: {_D.__file__} has no ranking modes; "
                         "put prc/data.py of main (VERSION >= 7) into the SOURCE tree")
    MONTHS_2026 = _D.rank_months(RANKING_MODE)
    RANKING = _D.ranking_file(RANKING_MODE)
    N_DEPARTURES = _D.n_ranking(RANKING_MODE)
    if RAW.resolve() not in RANKING.resolve().parents:
        raise SystemExit(f"the ranking file {RANKING} is not under RAW {RAW}; set PRC_DATA_DIR and EXP159_RAW alike")
#: split name -> token (R4-SPLIT-CONTRACT.md and the R2 contract; checked at every fit)
TOKENS = {"janjul": "b84316ca23737218", "jjx-12": "5946e5dc8a0f27f8", "jjx-02": "c844a0fb44a4f203",
          "jjx-06": "9a38a21f5288df37", "jjx-08": "6de56c539521a214", "jjx-03": "74be48796d29ae83",
          "jjx-10": "b0d549b7b3d4a662", "jjx-11": "10e173cdad96e9d8", "final": "8e5ae084dc94d3a1",
          "jjrank": "4abce89100ca3bad", "jjrefit": "59e4f0bf15b227ba",
          # the week-pair folds of exp-147 (tokens as in valid's L-outer-k records): same-month read
          "outer-1": "fa61496d69ce5530", "outer-2": "16a26046e00aca15", "outer-3": "b50a91f1bb387feb",
          "outer-4": "33bc30e2d9af8421", "outer-5": "32cbe4135b301f4a"}

# ---------------------------------------------------------------- input lists
OBS = ["obs_obs_taxi", "obs_obs_taxi_motion", "obs_obs_seg_taxi", "obs_dt_takeoff", "obs_time_to_250m",
       "obs_seen_before_move_s", "obs_stand_stop_s", "obs_turnaround_anchored", "obs_seg_points",
       "obs_n_ground_points", "obs_seg_dt_median_s", "obs_seg_gap_max_s", "obs_seg_span_s",
       "obs_callsign_n", "obs_stand_dist_km", "obs_stand_sin", "obs_stand_cos", "obs_arr_present",
       "obs_arr_taxi_in_s", "obs_arr_stand_occupancy_s", "obs_arr_gap_reception_s", "obs_obs_ambiguous",
       "obs_obs_n_candidates", "obs_match_mode", "obs_clock_ok", "obs_has_wheels_off",
       "obs_matched_on_modal"]
#: the six reception inputs the assignment removes
RECEPTION = ["obs_seg_points", "obs_n_ground_points", "obs_seg_dt_median_s", "obs_seg_gap_max_s",
             "obs_seg_span_s", "obs_arr_gap_reception_s"]
#: side arm A1b: the other inputs that depend on what the receivers saw
RECEPTION_WIDE = ["obs_seen_before_move_s", "obs_callsign_n", "obs_obs_n_candidates", "obs_obs_ambiguous",
                  "obs_matched_on_modal", "obs_has_wheels_off", "obs_clock_ok", "obs_dt_takeoff"]
OPDI_RAW = ["opdi_exit_parking_position_lead_s", "opdi_exit_parking_position_dist_m",
            "opdi_exit_apron_lead_s", "opdi_exit_apron_dist_m",
            "opdi_entry_taxiway_lead_s", "opdi_entry_taxiway_dist_m"]
TABLE_NEW = ["clock_implied_row", "clock_current250_row", "x_onset_minus_aobt", "x_implrow_minus_aobt",
             "x_implrow_minus_onset", *OPDI_RAW, "x_opdi_pp_minus_onset"]
#: the inputs that need exp-137's proxy table (`clock_*_row`)
PROXY_CLOCK = ["clock_implied_row", "clock_current250_row", "x_implrow_minus_aobt", "x_implrow_minus_onset"]
#: track inputs (track.py); the pushback-phase part is filtered by coverage before a fit
TRACK_CLOCK = ["trk_implied_taxi", "trk_implied_lead_s", "x_trkimpl_minus_aobt", "trk_d_first_m",
               "trk_v_first_kt", "trk_survey_to_runway_km", "trk_survey_to_obs_stand_m", "trk_survey_is_osm"]
TRACK_PUSH = ["trk_push_detected", "trk_push_duration_s", "trk_push_length_m", "trk_stop_after_push_s",
              "trk_push_start_lead_s", "trk_move_start_lead_s", "trk_onset_speed_kt", "trk_onset_to_10kt_s",
              "trk_remote_proxy", "trk_heading_change_deg"]


#: SPEC-push: context inputs derived from the frame (label-free codes)
CONTEXT = ["x_airport_code", "x_hour_utc", "x_weekday"]
CATEGORICAL = ["x_actype_code", "x_stand_code"]
#: SPEC-push addendum 7: the type code only for types with this many clock rows of the frame (label-free count);
#: the other types are missing. A4t gave C56X rows at LSZH offsets of thousands of seconds.
#: Addendum 8 uses the same floor for the operator and the runway codes.
ACTYPE_MIN_ROWS = 5000
#: exp-190: the type code table of the released twelve-month frame (frame12 of 2026-10-01, SHA-256 250fb20b...):
#: every type of that frame in code order, and the types at or above the floor. With SET 12 the frame's type codes
#: come from this table in both modes. The released rule (codes and floor counted on the whole frame) would shift
#: the codes and could add types once the frame holds February and June 2026; the table keeps the inputs of the
#: 2025 training rows as released. The ranking mode checks that the rule gives the table's columns.
ACTYPE_TABLE = Path(os.environ.get("EXP190_ACTYPE_TABLE", Path(__file__).with_name("actype_table.json")))
#: every input that fit.py hands to LightGBM as a category
CATEGORICAL_ALL = CATEGORICAL + ["x_actype_common", "x_oper_common", "x_rwy_common", "x_oper_airliner"]
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
QUEUE = ["q_taxi_f", "q_taxi_o"] + [f"q_{t}_n_{k}" for t in ("Nf", "No")
                                    for k in ("anom_cell", "anom_rwy_cell", "anom_rwy_30m", "n_cell", "n_rwy_cell", "n_rwy_30m")]
SLOT = [f"{n}_l{k}" for n in ("arrival_mean", "arrival_ratio", "clock_current250", "clock_implied",
                              "spacing_short_share", "departure_count_q95", "filed_delay_mean") for k in ("-1", "+0", "+1")]
#: big-lever arm A6: ERA5 weather at the airport in the take-off hour (weather.py; no input of the route reads weather)
WX_HOUR = ["wx_" + c for c in (
    "temp_c", "dewpoint_c", "dewpoint_depression_c", "msl_hpa", "wind_speed_ms", "wind_dir_deg", "cloud_total",
    "cloud_low", "precip_mm", "snow_mm_we", "liquid_mm", "freezing_precip", "snowing", "cloud_base_proxy_m",
    "low_visibility", "subzero", "snow_3h_mm", "snow_6h_mm", "precip_3h_mm", "precip_6h_mm", "snow_6h_any",
    "d3_temp_c", "d3_msl_hpa", "d3_wind_speed_ms", "d3_cloud_low", "d3_dewpoint_depression_c")]
WX_RUNWAY = ["wx_headwind_ms", "wx_crosswind_ms", "wx_crosswind_abs_ms", "wx_tailwind_ms"]
WX_ROLL = ["wx_snow_12h_mm", "wx_snow_24h_mm", "wx_precip_24h_mm", "wx_temp_min_12h_c", "wx_subzero_hours_24h",
           "wx_snowing_hours_12h"]
WX = WX_HOUR + WX_RUNWAY + WX_ROLL


def shp_cols(columns) -> list[str]:
    return sorted(c for c in columns if c.startswith("shp_"))


def arm_features(arm: str, columns, track_push: list[str] | None = None) -> list[str]:
    """The explicit input list of an arm. `columns`: the frame's columns."""
    shp = shp_cols(columns)
    if len(shp) != 76:
        raise ValueError(f"{len(shp)} shp_ columns, expected 76")
    a0 = OBS + shp
    a1 = [c for c in a0 if c not in RECEPTION]
    a2 = a1 + TABLE_NEW
    table = {"A0": a0, "A1": a1, "A1b": [c for c in a1 if c not in RECEPTION_WIDE], "A2": a2,
             "A2c": a1 + TABLE_NEW[:5],                       # A2 without OPDI
             "A0n": a0 + TABLE_NEW,                           # the new table inputs with reception kept
             "A3c": a2 + TRACK_CLOCK,                         # A2 + the rebuilt implied clock only
             "A3": a2 + TRACK_CLOCK + (TRACK_PUSH if track_push is None else track_push),
             # A3 with the six reception inputs kept: the local cost of the removal on the final arm
             "A3r": a0 + TABLE_NEW + TRACK_CLOCK + (TRACK_PUSH if track_push is None else track_push)}
    a3 = table["A3"]
    table.update({"A4a": a3 + CONTEXT, "A4b": a3 + CONTEXT + CATEGORICAL,
                  "A5": a3 + CONTEXT + CATEGORICAL + QUEUE + SLOT,
                  "A6": a3 + WX, "A6a": a3 + CONTEXT + WX})
    # A3 without exp-137's two row clocks and their differences: the proxy table has no row of April, May, September
    table["A3p"] = [c for c in a3 if c not in PROXY_CLOCK]
    table["A6p"] = table["A3p"] + WX
    # the two codes of A4b one at a time (A4b loses at EGLL and gains at five airports)
    table["A4t"], table["A4s"] = a3 + ["x_actype_code"], a3 + ["x_stand_code"]
    table["A4u"] = a3 + ["x_actype_common"]                  # A4t with the frequent types only
    # SPEC-push addendum 8: the operator and the departure runway, frequent levels only, on top of A4u
    table["A4w"], table["A4r"] = table["A4u"] + ["x_oper_common"], table["A4u"] + ["x_rwy_common"]
    table["A4z"] = table["A4u"] + ["x_oper_common", "x_rwy_common"]
    table["A4v"] = table["A4u"] + ["x_oper_airliner"]         # the repair arm of A4w: no operator code on a rare type
    if arm not in table:
        raise ValueError(f"unknown arm {arm}")
    out = table[arm]
    if len(set(out)) != len(out):
        raise ValueError(f"{arm}: repeated input")
    return out


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Label-free differences of inputs (the clock - AOBT disagreement and its relatives)."""
    f = frame
    out = pd.DataFrame(index=f.index)
    out["x_onset_minus_aobt"] = f["obs_obs_taxi"] - f["shp_r_aobt"]          # AOBT_3 - onset time
    out["x_implrow_minus_aobt"] = f["clock_implied_row"] - f["shp_r_aobt"]
    out["x_implrow_minus_onset"] = f["clock_implied_row"] - f["obs_obs_taxi"]
    out["x_opdi_pp_minus_onset"] = f["opdi_exit_parking_position_lead_s"] - f["obs_obs_taxi"]
    if "trk_implied_taxi" in f:
        out["x_trkimpl_minus_aobt"] = f["trk_implied_taxi"] - f["shp_r_aobt"]
    # SPEC-push context: codes from the row's own columns; the code table is the whole frame's
    apt = f["airport"].astype(str)
    if not apt.isin(AIRPORTS).all():
        raise ValueError("an airport outside the code table")
    out["x_airport_code"] = apt.map({a: i for i, a in enumerate(AIRPORTS)}).astype("float64")
    t = f[TIME]
    out["x_hour_utc"] = t.dt.hour + t.dt.minute / 60.0
    out["x_weekday"] = t.dt.weekday.astype("float64")
    typ = f["AIRCRAFT_TYPE_mvt"].astype("string")
    out["x_actype_code"] = pd.Series(pd.Categorical(typ, categories=sorted(typ.dropna().unique())).codes,
                                     index=f.index).replace(-1, np.nan).astype("float64")
    out["x_actype_common"] = out["x_actype_code"].where(typ.map(typ.value_counts()).to_numpy("float64") >= ACTYPE_MIN_ROWS)
    if SET == "12":                                       # exp-190: the frozen table of the released frame
        code, common = actype_columns(typ)
        same = (np.array_equal(code, out["x_actype_code"].to_numpy("float64"), equal_nan=True)
                and np.array_equal(common, out["x_actype_common"].to_numpy("float64"), equal_nan=True))
        if RANKING_MODE == "ranking" and not same:
            raise ValueError("ranking mode: the released type rule does not give the frozen type table")
        if not same:
            rule = set(typ[out["x_actype_common"].notna().to_numpy()].unique())
            tab = set(typ[np.isfinite(common)].unique())
            print(f"type table: the released rule on this frame would make {sorted(rule - tab)} frequent and "
                  f"{sorted(tab - rule)} rare; the frozen table is used", flush=True)
        out["x_actype_code"], out["x_actype_common"] = code, common
    if {"CALLSIGN_flt", "FLIGHT_mvt", "RUNWAY_mvt"} <= set(f.columns):
        # operator: the ICAO prefix of the callsign, else of the flight number; runway: airport and departure runway
        oper = f["CALLSIGN_flt"].astype("string").str.extract(r"^([A-Z]{3})\d", expand=False)
        oper = oper.fillna(f["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]{3})\d", expand=False))
        rwy = (apt + "|" + f["RUNWAY_mvt"].astype("string")).where(f["RUNWAY_mvt"].notna())
        # x_oper_airliner: the operator only on rows with a coded type; the floor counts those rows (A4w gave the
        # business-jet rows of one operator, NJE, offsets of thousands of seconds, as the plain type code did)
        airliner = oper.where(out["x_actype_common"].notna())
        for name, v in (("x_oper_common", oper), ("x_rwy_common", rwy), ("x_oper_airliner", airliner)):
            code = pd.Series(pd.Categorical(v, categories=sorted(v.dropna().unique())).codes, index=f.index)
            out[name] = code.replace(-1, np.nan).astype("float64").where(v.map(v.value_counts()).to_numpy("float64") >= ACTYPE_MIN_ROWS)
    stand = (apt + "|" + f["STAND_mvt"].astype("string")).where(f["STAND_mvt"].notna())
    out["x_stand_code"] = pd.Series(pd.Categorical(stand, categories=sorted(stand.dropna().unique())).codes,
                                    index=f.index).replace(-1, np.nan).astype("float64")
    return out


def actype_columns(typ: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(x_actype_code, x_actype_common) of the frozen table; a type outside the table is missing in both."""
    tab = json.loads(ACTYPE_TABLE.read_text())
    if tab["types"] != sorted(tab["types"]) or not set(tab["common"]) <= set(tab["types"]):
        raise ValueError(f"{ACTYPE_TABLE}: types not sorted, or a frequent type outside the table")
    code = typ.astype(object).map({t: float(i) for i, t in enumerate(tab["types"])}).astype("float64").to_numpy()
    common = np.where(typ.isin(tab["common"]).fillna(False).to_numpy(bool), code, np.nan)
    return code, common


# ---------------------------------------------------------------- small helpers
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refuse_existing(*paths: Path) -> None:
    for p in paths:
        if Path(p).exists():
            raise SystemExit(f"{p} exists; write-once")


def write_json(path: Path, rec: dict) -> None:
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")


def list_sha(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def load_frame(with_track: bool = False, with_extra: bool = False, with_weather: bool = False) -> pd.DataFrame:
    f = pd.read_parquet(FRAME)
    if with_weather:
        f = f.merge(pd.read_parquet(WEATHER), on=ID, how="left", validate="one_to_one")
    if with_extra:
        x = pd.read_parquet(EXTRA)
        f = f.merge(x, on=ID, how="left", validate="one_to_one")
    if with_track:
        t = pd.read_parquet(TRACK)
        if t[ID].duplicated().any():
            raise ValueError("track table: repeated ids")
        f = f.merge(t, on=ID, how="left", validate="one_to_one")
    f = pd.concat([f, derived(f)], axis=1)
    return f


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, "float64") - np.asarray(b, "float64")) ** 2)))
