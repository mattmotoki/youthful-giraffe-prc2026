"""Cross-month residual level 2 on top of c5 (screen).

Fit on held January, predict held July, and the reverse; G against c5 with iso/score.py.
    .venv/bin/python w2/l2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso")); sys.path.insert(0, str(ROOT / "w2"))
import base as B  # noqa: E402
import score as S  # noqa: E402

MEMBERS = ["raw_g", "raw_mlp", "raw_sequence021", "raw_sequence039", "raw_row6", "m_dark18", "raw_route",
           "raw_head", "raw_sequence011", "raw_row011", "st_route", "st_head", "st_primary"]
ROWF = ["hour", "dow", "nm_taxi", "aobt_lobt", "mvt_sched", "mvt_eobt", "mvt_lobt", "lobt_iobt", "sched_lobt",
        "arvt_change", "joined", "vfr", "ap_n300", "ap_nm300", "ap_n900", "ap_nm900", "ap_n1800", "ap_nm1800",
        "rw_n300", "rw_nm300", "rw_n900", "rw_nm900", "rw_n1800", "rw_nm1800", "arr_n600", "arr_n1800",
        "day_nm_mean", "day_nm_p90", "day_nm_long", "day_n", "day_join", "prev_day_same_flight_gap",
        "arr_nm_taxi", "arr_mvt_lobt", "arr_mvt_eobt", "arr_flight_time"]
CATS = ["airport", "cls"]


def frame(d: pd.DataFrame, base_col="c5") -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    b = d[base_col].to_numpy(float)
    X["base"] = b
    for m in MEMBERS:
        X["d_" + m] = d[m].to_numpy(float) - b
    for c in ROWF:
        X[c] = d[c].to_numpy(float)
    X["base_minus_nm"] = b - d["nm_taxi"].to_numpy(float)
    X["lo_gap"] = b - d["c5_lower"].to_numpy(float)
    X["hi_gap"] = d["c5_upper"].to_numpy(float) - b
    for c in CATS:
        X[c] = pd.Categorical(d[c].astype(str))
    return X


def cross_month(d, X, mask, params, rounds, clip=2000.0, base_col="c5"):
    y = d.y.to_numpy(float); w = d.w.to_numpy(float); b = d[base_col].to_numpy(float)
    r = np.clip(y - b, -clip, clip)
    month = d.month.to_numpy()
    corr = np.zeros(len(d))
    for held in (1, 7):
        tr = mask & (month != held) & (w > 0)
        te = mask & (month == held)
        ds = lgb.Dataset(X[tr], r[tr], weight=w[tr] / w[tr].mean(), categorical_feature=CATS, free_raw_data=False)
        m = lgb.train(params, ds, rounds)
        corr[te] = m.predict(X[te])
    return corr


def main():
    d = B.load("fold")
    d["aday"] = S.fold().aday.to_numpy()
    X = frame(d)
    base = d.c5.to_numpy(float); lo = d.c5_lower.to_numpy(float); hi = d.c5_upper.to_numpy(float)
    mask = (d.cls != "noac").to_numpy()
    params = dict(objective="huber", alpha=300.0, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=8,
                  verbose=-1, seed=1)
    for rounds in (300, 1000):
        corr = cross_month(d, X, mask, params, rounds)
        for k in (0.5, 1.0):
            cand = np.maximum(60, np.clip(base + k * corr, lo, hi))
            res = S.score(cand, base=base, d=S.fold())
            print(f"rounds {rounds} k {k}", S.fmt(res), flush=True)
            print("   airport", res["airport"], flush=True)


if __name__ == "__main__":
    main()
