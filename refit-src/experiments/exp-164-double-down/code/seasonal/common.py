"""Paths, hashes and the arm lists of exp-159 (the with-clock head as a direct R4 member).

Runs on the workstation. Everything outside `WORK` is read only.
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
W148 = COMP / "prc-exp148-work"
RAW = COMP / "prc-data-challenge-2026/data/raw"
FEATURES = W148 / "inputs/features-r4c.parquet"
PROXY = W148 / "inputs/proxy-features-r4c.parquet"
OPDI = Path(os.path.join(os.environ.get("PRC_OPDI_ROOT", "/path/to/prc-exp155-opdi"), "r2/inputs/complete/opdi-first.parquet"))
FRAME = WORK / "inputs/frame.parquet"
TRACK = WORK / "inputs/track.parquet"

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "experiments/exp-147-foldserve/code"))
os.environ.setdefault("PRC_VALIDATION_DIR", str(SOURCE / "experiments/exp-146-validation/frozen"))
os.environ.setdefault("PRC_DATA_DIR", str(RAW))

ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
MONTHS_2025 = ("2025-01", "2025-02", "2025-03", "2025-06", "2025-07", "2025-08", "2025-10", "2025-11", "2025-12")
MONTHS_2026 = ("2026-01", "2026-07")
#: split name -> token (R4-SPLIT-CONTRACT.md and the R2 contract; checked at every fit)
TOKENS = {"janjul": "b84316ca23737218", "jjx-12": "5946e5dc8a0f27f8", "jjx-02": "c844a0fb44a4f203",
          "jjx-06": "9a38a21f5288df37", "jjx-08": "6de56c539521a214", "jjx-03": "74be48796d29ae83",
          "jjx-10": "b0d549b7b3d4a662", "jjx-11": "10e173cdad96e9d8", "final": "8e5ae084dc94d3a1",
          "jjrank": "4abce89100ca3bad", "jjrefit": "59e4f0bf15b227ba"}

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
#: track inputs (track.py); the pushback-phase part is filtered by coverage before a fit
TRACK_CLOCK = ["trk_implied_taxi", "trk_implied_lead_s", "x_trkimpl_minus_aobt", "trk_d_first_m",
               "trk_v_first_kt", "trk_survey_to_runway_km", "trk_survey_to_obs_stand_m", "trk_survey_is_osm"]
TRACK_PUSH = ["trk_push_detected", "trk_push_duration_s", "trk_push_length_m", "trk_stop_after_push_s",
              "trk_push_start_lead_s", "trk_move_start_lead_s", "trk_onset_speed_kt", "trk_onset_to_10kt_s",
              "trk_remote_proxy", "trk_heading_change_deg"]


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
             "A3": a2 + TRACK_CLOCK + (TRACK_PUSH if track_push is None else track_push)}
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
    return out


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


def load_frame(with_track: bool = False) -> pd.DataFrame:
    f = pd.read_parquet(FRAME)
    if with_track:
        t = pd.read_parquet(TRACK)
        if t[ID].duplicated().any():
            raise ValueError("track table: repeated ids")
        f = f.merge(t, on=ID, how="left", validate="one_to_one")
    f = pd.concat([f, derived(f)], axis=1)
    return f


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, "float64") - np.asarray(b, "float64")) ** 2)))
