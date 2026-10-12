"""Build train-only encoders and label-free airport-day windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
        "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "month_cat"]
TIMES = [TIME, "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
         "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
OBS = ["obs_obs_taxi", "obs_obs_taxi_motion", "obs_obs_seg_taxi", "obs_time_to_250m",
       "obs_seen_before_move_s", "obs_stand_stop_s", "obs_turnaround_anchored",
       "obs_stand_dist_km", "obs_stand_sin", "obs_stand_cos", "obs_arr_present",
       "obs_arr_stand_occupancy_s", "clock_current250_row", "clock_implied_row",
       "opdi_exit_parking_position_lead_s", "opdi_exit_parking_position_dist_m",
       "opdi_exit_apron_lead_s", "opdi_exit_apron_dist_m",
       "opdi_entry_taxiway_lead_s", "opdi_entry_taxiway_dist_m"]
TRACK = ["trk_implied_taxi", "trk_implied_lead_s", "trk_d_first_m", "trk_v_first_kt",
         "trk_survey_to_runway_km", "trk_survey_to_obs_stand_m", "trk_survey_is_osm",
         "trk_push_detected", "trk_push_duration_s", "trk_push_length_m",
         "trk_stop_after_push_s", "trk_push_start_lead_s", "trk_move_start_lead_s",
         "trk_onset_speed_kt", "trk_onset_to_10kt_s", "trk_remote_proxy", "trk_heading_change_deg"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def numeric(f):
    z = pd.DataFrame(index=f.index)
    for c in TIMES[1:]:
        z["m_minus_" + c] = (f[TIME] - f[c]).dt.total_seconds()
    for a, b in [("AOBT_3_flt", "EOBT_1_flt"), ("AOBT_3_flt", "LOBT_flt"),
                 ("ARVT_3_flt", "ARVT_1_flt"), ("ARVT_1_flt", "EOBT_1_flt"),
                 ("SCHED_TIME_UTC_mvt", "LOBT_flt")]:
        z[a + "_minus_" + b] = (f[a] - f[b]).dt.total_seconds()
    hour = f[TIME].dt.hour + f[TIME].dt.minute / 60
    for name, val, period in [("hour", hour, 24), ("weekday", f[TIME].dt.dayofweek, 7),
                              ("month", f.month, 12)]:
        z[name + "_sin"] = np.sin(2 * np.pi * val / period)
        z[name + "_cos"] = np.cos(2 * np.pi * val / period)
    for c in OBS + TRACK:
        z[c] = f[c].astype(float)
    sec = f[TIME].astype("int64") / 1e9
    z["gap_prev"] = sec.groupby(f.group, sort=False).diff().clip(0, 86400)
    z["gap_next"] = -sec.groupby(f.group, sort=False).diff(-1).clip(-86400, 0)
    z["seconds_in_day"] = sec % 86400
    z["nm_minus_obs"] = z.m_minus_AOBT_3_flt - z.obs_obs_taxi
    return z.replace([np.inf, -np.inf], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--clock", type=Path, required=True)
    ap.add_argument("--track", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise RuntimeError("input directory already exists")
    a.out.mkdir(parents=True)
    sources = {}
    parts = []
    cols = [ID, "PHASE_mvt", "TAXITIME_SEC_mvt"] + TIMES + CATS[:-1]
    for p in sorted(a.raw.glob("training_2025-*.parquet")):
        f = pd.read_parquet(p, columns=cols)
        f = f[f.PHASE_mvt == "DEP"].copy()
        for c in TIMES:
            f[c] = pd.to_datetime(f[c], utc=True)
        f["year"], f["month"] = 2025, f[TIME].dt.month
        parts.append(f)
        sources[str(p)] = sha(p)
        print("raw", p.name, len(f), flush=True)
    f = pd.read_parquet(a.raw / "ranking.parquet", columns=[c for c in cols if c != "TAXITIME_SEC_mvt"])
    f = f[f.PHASE_mvt == "DEP"].copy()
    for c in TIMES:
        f[c] = pd.to_datetime(f[c], utc=True)
    f["TAXITIME_SEC_mvt"] = np.nan
    f["year"], f["month"] = 2026, f[TIME].dt.month
    parts.append(f)
    sources[str(a.raw / "ranking.parquet")] = sha(a.raw / "ranking.parquet")
    f = pd.concat(parts, ignore_index=True)
    f[ID] = f[ID].astype("int64")
    assert not f[ID].duplicated().any()
    for p, cs in [(a.clock, OBS), (a.track, TRACK)]:
        clock = pd.read_parquet(p, columns=[ID] + cs)
        clock[ID] = clock[ID].astype("int64")
        assert not clock[ID].duplicated().any()
        f = f.merge(clock, on=ID, how="left", validate="one_to_one")
        sources[str(p)] = sha(p)
    f["date"] = f[TIME].dt.strftime("%Y-%m-%d")
    f = f.sort_values(["ADEP_mvt", "date", TIME, ID], kind="stable").reset_index(drop=True)
    f["group"] = pd.factorize(pd.MultiIndex.from_frame(f[["ADEP_mvt", "date"]]), sort=False)[0]
    f["month_cat"] = f.month.astype(str)
    train = (f.year == 2025) & ~f.month.isin([1, 7])
    held = (f.year == 2025) & f.month.isin([1, 7])
    assert set(f.loc[train, "month"]) == {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}
    assert set(f.loc[train, "group"]).isdisjoint(set(f.loc[held, "group"]))
    z = numeric(f)
    assert not set(z) & {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y"}
    # Signed log compresses long, legal time differences without using a label.
    val = np.sign(z.to_numpy(float)) * np.log1p(np.abs(z.to_numpy(float)))
    center = np.nanmedian(val[train], axis=0)
    center = np.nan_to_num(center, nan=0.)
    scale = np.nanstd(val[train], axis=0)
    scale = np.maximum(np.nan_to_num(scale, nan=1.), 0.1)
    num = np.concatenate([np.nan_to_num((val - center) / scale, nan=0.).clip(-12, 12),
                          np.isfinite(val).astype(float)], axis=1).astype("float32")
    vocab, cat = {}, []
    for c in CATS:
        s = f[c].astype("string").fillna("__MISSING__")
        counts = s[train].value_counts()
        values = sorted(counts[counts >= 5].index.tolist())
        vocab[c] = values
        cat.append(s.map({v: i + 1 for i, v in enumerate(values)}).fillna(0).to_numpy("int32"))
    cat = np.stack(cat, axis=1)
    nm = (f[TIME] - f.AOBT_3_flt).dt.total_seconds().to_numpy(float)
    # A legal common anchor; the neural output is in seconds above this anchor.
    anchor = np.where(np.isfinite(nm), np.clip(nm, 60., 43200.), 900.).astype("float32")
    y = f.TAXITIME_SEC_mvt.to_numpy("float64")
    meta = f[[ID, "ADEP_mvt", TIME, "year", "month", "date", "group"]].copy()
    meta["aobt"], meta["aircraft"] = f.AOBT_3_flt.notna(), f.AIRCRAFT_TYPE_mvt.notna()
    meta["plan"] = f.LOBT_flt.notna()
    meta.to_parquet(a.out / "rows.parquet", index=False)
    for name, array in [("numeric", num), ("category", cat), ("anchor", anchor), ("labels", y)]:
        np.save(a.out / (name + ".npy"), array, allow_pickle=False)
    rec = {"sources": sources, "numeric": list(z), "category": CATS, "vocab": vocab,
           "center": center.tolist(), "scale": scale.tolist(), "rows": len(f),
           "train_rows": int(train.sum()), "held_rows": int(held.sum()),
           "train_months": sorted(f.loc[train, "month"].unique().tolist()),
           "input_label_columns": [], "source_code_sha256": sha(__file__),
           "files": {p.name: sha(p) for p in sorted(a.out.iterdir())}}
    (a.out / "MANIFEST.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("ready", rec["train_rows"], rec["held_rows"], num.shape, cat.shape, flush=True)


if __name__ == "__main__":
    main()
