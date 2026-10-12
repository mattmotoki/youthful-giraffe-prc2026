"""PLAN.md (c) path G: a one-model base from official columns only, as a measured floor for Feb/Jun.

LightGBM (L2 loss, 2 threads) on official departure columns plus same-file traffic counts. Held read: fit on
the ten 2025 months without January and July, score January and July 2025 (same rows as the zabd held set).
With --final it then refits on all twelve months and predicts the February/June 2026 departures of
final_ranking.parquet into <out.parquet> (MVT_ID_mvt, TAXITIME_SEC_mvt). Development read: no review, no seeds bag.
Run: OMP_NUM_THREADS=2 .venv/bin/python -I official_gbdt.py <repo root> [--final <out.parquet>] [--save-held <held.parquet>]
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
root = Path(sys.argv[1]).resolve()
final_out = Path(sys.argv[sys.argv.index("--final") + 1]) if "--final" in sys.argv else None
held_out = Path(sys.argv[sys.argv.index("--save-held") + 1]) if "--save-held" in sys.argv else None
# --clip C: train on labels capped at C s, without the LIRF rows that have no aircraft type (the rule rows).
CLIP = float(sys.argv[sys.argv.index("--clip") + 1]) if "--clip" in sys.argv else None
raw = root / "data" / "raw"
CAT = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADES_mvt",
       "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]


def features(df):
    """Features of the DEP rows of one file (all rows of the file give the traffic counts)."""
    df = df.copy()
    t = df.MVT_TIME_UTC_mvt
    dep = df[df.PHASE_mvt.eq("DEP")].copy()
    ap = np.where(df.PHASE_mvt.eq("DEP"), df.ADEP_mvt, df.ADES_mvt)
    # Movements of the same airport within +-15 min of the take-off (both phases), by a sorted search.
    counts = np.zeros(len(dep))
    for a in np.unique(ap):
        tt = np.sort(t[ap == a].astype("int64").to_numpy() // 10**6)
        m = (dep.ADEP_mvt == a).to_numpy()
        x = dep.loc[m, "MVT_TIME_UTC_mvt"].astype("int64").to_numpy() // 10**6
        counts[m] = np.searchsorted(tt, x + 900) - np.searchsorted(tt, x - 900)
    tm = dep.MVT_TIME_UTC_mvt
    out = pd.DataFrame(index=dep.index)
    for c, ref in (("aobt3", "AOBT_3_flt"), ("eobt1", "EOBT_1_flt"), ("sched", "SCHED_TIME_UTC_mvt"),
                   ("lobt", "LOBT_flt"), ("iobt", "IOBT_flt")):
        out[f"mvt_minus_{c}"] = (tm - dep[ref]).dt.total_seconds()
    out["hour"] = tm.dt.hour + tm.dt.minute / 60
    out["dow"] = tm.dt.dayofweek
    out["traffic_30min"] = counts
    for c in CAT:
        out[c] = dep[c].astype("string").fillna("na")
    return dep[[ID]].join(out), dep[Y].astype(float) if Y in dep else None


def load(path):
    return pd.read_parquet(path)


def month_file(m):
    a, b = f"2025-{m:02d}-01", (f"2025-{m + 1:02d}-01" if m < 12 else "2026-01-01")
    return raw / f"training_{a}_{b}.parquet"


t0 = time.time()
parts = {m: features(load(month_file(m))) for m in range(1, 13)}
cats = {c: pd.CategoricalDtype(sorted(set().union(*[set(parts[m][0][c]) for m in parts]) | {"na"})) for c in CAT}


def matrix(x):
    x = x.drop(columns=[ID]).copy()
    for c in CAT:
        x[c] = x[c].astype(cats[c])
    return x


params = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, num_threads=2,
              verbose=-1, seed=1)
ROUNDS = 1500
tr = [m for m in range(1, 13) if m not in (1, 7)]
Xtr = matrix(pd.concat([parts[m][0] for m in tr]))
ytr = pd.concat([parts[m][1] for m in tr])
Xh = matrix(pd.concat([parts[m][0] for m in (1, 7)]))
yh = pd.concat([parts[m][1] for m in (1, 7)]).to_numpy()
mo = np.r_[np.full(len(parts[1][0]), 1), np.full(len(parts[7][0]), 7)]
if CLIP is not None:
    keep = ~((Xtr["ADEP_mvt"] == "LIRF") & (Xtr["AIRCRAFT_TYPE_mvt"] == "na")).to_numpy()
    Xtr, ytr = Xtr[keep], ytr[keep].clip(upper=CLIP)
booster = lgb.train(params, lgb.Dataset(Xtr, ytr), ROUNDS)
p = np.maximum(booster.predict(Xh, num_threads=2), 60)
e = p - yh
read = {"held_rmse": round(float(np.sqrt(np.mean(e ** 2))), 3),
        "held_rmse_m1": round(float(np.sqrt(np.mean(e[mo == 1] ** 2))), 3),
        "held_rmse_m7": round(float(np.sqrt(np.mean(e[mo == 7] ** 2))), 3),
        "rows": int(len(yh)), "rounds": ROUNDS, "clip": CLIP, "fit_minutes": round((time.time() - t0) / 60, 1)}
print(json.dumps(read), flush=True)
if held_out is not None:
    pd.DataFrame({ID: pd.concat([parts[m][0][ID] for m in (1, 7)]).to_numpy(), "y": yh, "pred": p}).to_parquet(held_out, index=False)
if final_out is not None:
    Xall = matrix(pd.concat([parts[m][0] for m in range(1, 13)]))
    yall = pd.concat([parts[m][1] for m in range(1, 13)])
    full = lgb.train(params, lgb.Dataset(Xall, yall), ROUNDS)
    fx, _ = features(load(raw / "final" / "final_ranking.parquet"))
    tpl = pd.read_parquet(raw / "final" / "final_submitting.parquet")
    fx = fx.set_index(ID).loc[tpl[ID].to_numpy()].reset_index()
    pf = np.maximum(full.predict(matrix(fx), num_threads=2), 60)
    pd.DataFrame({ID: tpl[ID].to_numpy(), Y: pf}).to_parquet(final_out, index=False)
    read["final_written"] = str(final_out)
    read["total_minutes"] = round((time.time() - t0) / 60, 1)
Path(__file__).with_name("official_gbdt.json" if CLIP is None else f"official_gbdt_clip{int(CLIP)}.json").write_text(json.dumps(read, indent=1) + "\n")
print(json.dumps(read), flush=True)
