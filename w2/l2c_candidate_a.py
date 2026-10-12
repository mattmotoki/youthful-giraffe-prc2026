"""Step L2C: a nonlinear residual level 2 on clock rows, on top of c5.

LightGBM (Huber 300, 31 leaves, 800 rounds, 9 seeds) on y - c5 with the member deviations, the row
features of w2/l2.py and the s7 deviation d_s7 = s7 - c5 (worker 5 candidate A; s7 = the ORD-S member). Held: fitted on one held month, applied to the other. 2026: fitted on both held
months. Value = max(60, clip(c5 + K * correction, lower, upper)). Scope: clock rows with an NM join;
on 2026 the extension rows and the LIRF January clock rows (no 2025 support) keep c5.

    .venv/bin/python w2/l2c.py [--s7 w2/inputs/s7c.npz] [--out outputs/x.parquet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso")); sys.path.insert(0, str(ROOT / "w2"))
import base as B  # noqa: E402
import finalmode as FM  # noqa: E402
import l2  # noqa: E402
import out as O  # noqa: E402
import score as S  # noqa: E402

K = 0.5
SEEDS = tuple(range(1, 10))
ROUNDS = 800
PARAMS = dict(objective="huber", alpha=300.0, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=int(__import__("os").environ.get("NT", "8")), verbose=-1,
              deterministic=True, force_row_wise=True)


def masks(d: pd.DataFrame, split: str) -> np.ndarray:
    m = (d.cls == "clock").to_numpy() & (d.joined.to_numpy() == 1)
    if split == "ranking":
        lirf_jan = (d.airport.eq("LIRF") & d.month.isin(FM.JAN_SCOPE)).to_numpy()   # season map: Feb as Jan
        m &= ~d.extension.to_numpy(bool) & ~lirf_jan
    return m


def fit_predict(Xtr, r, w, Xte):
    pred = np.zeros(len(Xte))
    for s in SEEDS:
        ds = lgb.Dataset(Xtr, r, weight=w / w.mean(), categorical_feature=l2.CATS, free_raw_data=False)
        pred += lgb.train(dict(PARAMS, seed=s), ds, ROUNDS).predict(Xte) / len(SEEDS)
    return pred


def build(extra_fold: dict | None = None, extra_rank: dict | None = None):
    d = B.load("fold"); r = B.load("ranking")
    d["y"] = S.fold().y.to_numpy()
    Xd = l2.frame(d); Xr = l2.frame(r)
    for name, v in (extra_fold or {}).items():
        Xd[name] = v - d.c5.to_numpy(float); Xr[name] = extra_rank[name] - r.c5.to_numpy(float)
    Xr["airport"] = pd.Categorical(r.airport.astype(str), categories=Xd["airport"].cat.categories)
    Xr["cls"] = pd.Categorical(r.cls.astype(str), categories=Xd["cls"].cat.categories)
    y = d.y.to_numpy(float); w = d.w.to_numpy(float); base = d.c5.to_numpy(float)
    res = np.clip(y - base, -2000, 2000)
    mf = masks(d, "fold"); mr = masks(r, "ranking"); month = d.month.to_numpy()
    corr_f = np.zeros(len(d))
    for held in (1, 7):
        tr = mf & (month != held) & (w > 0); te = mf & (month == held)
        corr_f[te] = fit_predict(Xd[tr], res[tr], w[tr], Xd[te])
    tr = mf & (w > 0)
    corr_r = np.zeros(len(r))
    corr_r[mr] = fit_predict(Xd[tr], res[tr], w[tr], Xr[mr])
    vf = np.maximum(60, np.clip(base + K * corr_f, d.c5_lower.to_numpy(float), d.c5_upper.to_numpy(float)))
    vr = r.c5.to_numpy(float).copy()
    vr[mr] = np.maximum(60, np.clip(vr[mr] + K * corr_r[mr], r.c5_lower.to_numpy(float)[mr], r.c5_upper.to_numpy(float)[mr]))
    return d, r, vf, vr, corr_f, corr_r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path)
    ap.add_argument("--s7", type=Path, default=ROOT / "w2/inputs/s7c.npz", help="s7 member, keys fold and ranking")
    a = ap.parse_args()
    s7 = np.load(a.s7)
    d, r, vf, vr, corr_f, corr_r = build(extra_fold={"d_s7": s7["fold"].astype(float)},
                                         extra_rank={"d_s7": s7["ranking"].astype(float)})
    res = S.score(vf, base=d.c5.to_numpy(float))
    print("held vs c5", S.fmt(res)); print("airport", res["airport"])
    dv = vr - r.c5.to_numpy(float)
    print("2026 changed", int((dv != 0).sum()), "mean change %.2f" % dv[dv != 0].mean(),
          "squared change per row %.2f" % float((dv ** 2).mean()))
    print("2026 mean change by airport-month", json.dumps({f"{k[0]}-{k[1]}": round(float(v), 1) for k, v in
          pd.Series(dv).groupby([r.airport.to_numpy(), r.month.to_numpy()]).mean().items()}))
    np.save(ROOT / "work/w2/l2c_fold.npy", vf); np.save(ROOT / "work/w2/l2c_ranking.npy", vr)
    if a.out:
        print("sha256", O.write(r.MVT_ID_mvt.to_numpy(), vr, a.out), a.out)


if __name__ == "__main__":
    main()
