"""Shared helpers for exp-004: base feature set, model config, metric reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"


def epoch(s: pd.Series) -> np.ndarray:
    """tz-aware datetime series -> float64 seconds since epoch, NaN for NaT."""
    v = s.to_numpy(dtype="datetime64[ns]")
    out = v.astype("int64").astype("float64")
    out[np.isnat(v)] = np.nan
    return out / 1e9


def airport(df: pd.DataFrame) -> pd.Series:
    """Airport the movement happened at: ADEP for departures, ADES for arrivals."""
    return np.where(df["PHASE_mvt"].to_numpy() == "DEP", df["ADEP_mvt"], df["ADES_mvt"])


BASE_CAT = ["apt", "apt_stand", "apt_rwy", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Own-flight deltas + airport, stand, runway, aircraft type, operator, hour.

    Computed from the (possibly masked) month frame alone; departures only.
    """
    d = frame[frame["PHASE_mvt"] == "DEP"]
    t = epoch(d["MVT_TIME_UTC_mvt"])
    aobt = epoch(d["AOBT_3_flt"])
    lobt = epoch(d["LOBT_flt"])
    eobt = epoch(d["EOBT_1_flt"])
    sched = epoch(d["SCHED_TIME_UTC_mvt"])
    arvt3 = epoch(d["ARVT_3_flt"])
    apt = d["ADEP_mvt"].astype(str)
    out = pd.DataFrame(index=d.index)
    out["dt_to_aobt"] = t - aobt
    out["dt_to_lobt"] = t - lobt
    out["dt_to_eobt"] = t - eobt
    out["dt_to_sched"] = t - sched
    out["dt_aobt_sched"] = aobt - sched
    out["dt_aobt_lobt"] = aobt - lobt
    out["flight_time_est"] = arvt3 - t
    ts = d["MVT_TIME_UTC_mvt"].dt
    out["hour"] = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    out["dow"] = ts.dayofweek.to_numpy()
    out["has_nm"] = (~np.isnan(aobt)).astype(np.int8)
    out["apt"] = apt.to_numpy()
    out["apt_stand"] = (apt + ":" + d["STAND_mvt"].astype(str)).to_numpy()
    out["apt_rwy"] = (apt + ":" + d["RUNWAY_mvt"].astype(str)).to_numpy()
    out["AIRCRAFT_TYPE_mvt"] = d["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    out["AIRCRAFT_OPERATOR_flt"] = d["AIRCRAFT_OPERATOR_flt"].astype(str).to_numpy()
    return out


EXTRA_NUM = ["noac", "flight_len", "noac_day_apt", "sched_min"]


def base_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    """base_features plus the tail worker's noac row-set columns (exp-002 finding 1).

    `noac` (AIRCRAFT_TYPE_mvt null) marks rows whose recorded block time is
    often a placeholder equal to SCHED_TIME; the per-day count of noac rows at
    the airport separates an isolated unenriched record from a bulk feed
    outage. All three are observable in the ranking file.
    """
    out = base_features(frame)
    d = frame[frame["PHASE_mvt"] == "DEP"]
    noac = d["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    out["noac"] = noac.astype(np.int8)
    out["flight_len"] = d["FLIGHT_mvt"].astype(str).str.len().to_numpy()
    day = d["MVT_TIME_UTC_mvt"].dt.floor("D").astype("int64").to_numpy()
    key = pd.DataFrame({"apt": d["ADEP_mvt"].astype(str).to_numpy(), "day": day, "noac": noac})
    out["noac_day_apt"] = key.groupby(["apt", "day"])["noac"].transform("sum").to_numpy()
    ts = d["SCHED_TIME_UTC_mvt"].dt
    out["sched_min"] = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    return out


def metrics(y: np.ndarray, p: np.ndarray, has_nm: np.ndarray,
            noac: np.ndarray | None = None) -> dict:
    """RMSE overall and on the segments COMMON.md requires, plus `non-noac`.

    `non-noac` is the population exp-002's leaderboard arithmetic is stated
    over (all rows whose AIRCRAFT_TYPE_mvt is present, tail included). It is
    not the same as the NM bulk and the two must never be compared to the same
    target band.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e2 = (y - p) ** 2
    bulk = y <= 3600
    tail = ~bulk
    nonm = has_nm == 0
    r = lambda m: float(np.sqrt(e2[m].mean())) if m.sum() else float("nan")
    out = {
        "n": int(len(y)),
        "rmse": r(np.ones(len(y), bool)),
        "rmse_bulk": r(bulk),
        "rmse_tail": r(tail),
        "rmse_nonm": r(nonm),
        "rmse_nm_bulk": r((~nonm) & bulk),
        "n_tail": int(tail.sum()),
        "n_nonm": int(nonm.sum()),
    }
    if noac is not None:
        noac = np.asarray(noac, bool)
        out["rmse_non_noac"] = r(~noac)
        out["rmse_noac"] = r(noac)
        out["n_noac"] = int(noac.sum())
    return out


def fmt(m: dict) -> str:
    s = (f"n={m['n']} rmse={m['rmse']:.1f} bulk={m['rmse_bulk']:.1f} "
         f"tail={m['rmse_tail']:.0f} (n={m['n_tail']}) noNM={m['rmse_nonm']:.0f} (n={m['n_nonm']})")
    if "rmse_non_noac" in m:
        s += f" nonNoac={m['rmse_non_noac']:.1f}"
    return s


CB_PARAMS = dict(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=7,
    verbose=500,
    thread_count=16,
    task_type="GPU",
    devices="1",
    max_ctr_complexity=2,
)


def fit_predict(Xtr, ytr, Xva, cat_cols, params=None):
    from catboost import CatBoostRegressor, Pool
    p = dict(CB_PARAMS)
    if params:
        p.update(params)
    cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols]
    tr = Pool(Xtr, ytr, cat_features=cat_idx)
    va = Pool(Xva, cat_features=cat_idx)
    m = CatBoostRegressor(**p)
    m.fit(tr)
    return m, m.predict(va)
