"""Leave-one-month-out over all 1,507 training noac rows.

The desk's decomposition: the error on a noac row is
`p(1-p)·(mvt_sched - taxi)^2`, so the lever is the SHARPNESS of p, not either
conditional mean. Every variant below is therefore judged on pooled RMSE and on
the calibration/sharpness of its p, not on feature counts.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent))
from threads import THREADS   # caps BLAS at import time; docs/COMPUTE.md compute allocation
import sys
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D

ART = D.ROOT / "artifacts" / "exp-009-noac2"
EXP = D.ROOT / "experiments" / "exp-009-noac2"
NR, NN = 344841, 388
PH_TOL = 300.0
SEEDS = (11, 23, 37, 51, 77)

BASE = ["mvt_sched", "flight_len", "hour", "dow", "minute_of_day",
        "noac_day_count", "nonm_day_count", "n_arr_sched_mvt", "gap_arr_after_sched",
        "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
        "gap_dep_prev_mvt", "gap_dep_next_mvt", "vfr"]          # exp-002's set
NEW = ["noac_day_n", "noac_day_share", "dep_day_n", "arr_within_1h_of_sched",
       "arr_within_3h_of_sched", "flight_n_digits", "flight_n_alpha",
       "flight_repeated_prefix", "mvt_sched_log"]
CATS_BASE = ["apt", "runway"]
CATS_NEW = ["apt", "runway", "stand", "flight_prefix"]

P = dict(objective="regression", learning_rate=0.04, num_leaves=8, n_estimators=300,
         min_child_samples=30, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
         n_jobs=THREADS, verbose=-1, deterministic=True, force_row_wise=True)


def enc(d, feats, cats, levels):
    X = d[[c for c in feats if c not in cats]].copy()
    for c in cats:
        X[c] = pd.Categorical(d[c].astype(str), categories=levels[c])
    return X


def lomo(df, feats, cats, name, mode="offset"):
    """mode: 'offset' (one L2 regressor) or 'mixture' (calibrated p x components)."""
    ys, ps, pp, mons = [], [], [], []
    for m in D.MONTHS:
        te, tr = df[df.month == m], df[df.month != m]
        if not len(te):
            continue
        levels = {c: pd.Index(sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))) for c in cats}
        Xtr, Xte = enc(tr, feats, cats, levels), enc(te, feats, cats, levels)
        ytr = tr.y.to_numpy(float)
        o = tr.mvt_sched.to_numpy(float) - ytr
        ms = te.mvt_sched.to_numpy(float)
        if mode == "offset":
            pred = ms - np.mean([lgb.LGBMRegressor(**dict(P, seed=s, bagging_seed=s,
                                 feature_fraction_seed=s, data_random_seed=s)).fit(
                                 Xtr, o, categorical_feature=cats).predict(Xte) for s in SEEDS], axis=0)
            p_hat = np.full(len(te), np.nan)
        else:
            ph = (np.abs(o) < PH_TOL).astype(int)
            p_hat = np.mean([lgb.LGBMClassifier(**dict(P, objective="binary", seed=s, bagging_seed=s,
                             feature_fraction_seed=s, data_random_seed=s)).fit(
                             Xtr, ph, categorical_feature=cats).predict_proba(Xte)[:, 1] for s in SEEDS], axis=0)
            onp = np.mean([lgb.LGBMRegressor(**dict(P, seed=s, bagging_seed=s,
                           feature_fraction_seed=s, data_random_seed=s)).fit(
                           Xtr[ph == 0], o[ph == 0], categorical_feature=cats).predict(Xte) for s in SEEDS], axis=0)
            pred = ms - (1.0 - p_hat) * onp
        ys.append(te.y.to_numpy(float)); ps.append(pred); pp.append(p_hat)
        mons.append(np.repeat(m, len(te)))
    y, pr, p_hat, mon = (np.concatenate(x) for x in (ys, ps, pp, mons))
    se = (y - pr) ** 2
    row = dict(model=name, mode=mode, n=len(y), rmse=np.sqrt(se.mean()),
               ranking_contrib_s=np.sqrt(NN * se.mean() / NR),
               worst_month=max(np.sqrt(se[mon == m].mean()) for m in np.unique(mon)))
    per = {m: np.sqrt(se[mon == m].mean()) for m in np.unique(mon)}
    return row, per, pd.DataFrame({"month": mon, "y": y, "pred": pr, "p": p_hat})


if __name__ == "__main__":
    df = pd.read_parquet(ART / "noac_train.parquet")
    o_all = df.mvt_sched.to_numpy(float) - df.y.to_numpy(float)
    print(f"{len(df)} noac rows, P(placeholder) = {(np.abs(o_all) < PH_TOL).mean():.3f}\n")
    runs = [
        ("exp-002 baseline (BASE, offset)", BASE, CATS_BASE, "offset"),
        ("exp-002 baseline (BASE, mixture)", BASE, CATS_BASE, "mixture"),
        ("+new observables (offset)", BASE + NEW, CATS_BASE, "offset"),
        ("+new observables (mixture)", BASE + NEW, CATS_BASE, "mixture"),
        ("+new + stand/prefix cats (offset)", BASE + NEW, CATS_NEW, "offset"),
        ("+new + stand/prefix cats (mixture)", BASE + NEW, CATS_NEW, "mixture"),
    ]
    rows, pers, best = [], {}, None
    for name, feats, cats, mode in runs:
        r, per, pred = lomo(df, feats, cats, name, mode)
        rows.append(r); pers[name] = per
        if best is None or r["rmse"] < best[0]:
            best = (r["rmse"], name, pred)
        print(f'{name:38s} {mode:8s} rmse={r["rmse"]:7.1f}  contrib={r["ranking_contrib_s"]:5.1f} s', flush=True)
    res = pd.DataFrame(rows).sort_values("rmse")
    res.to_csv(EXP / "work" / "lomo_variants.csv", index=False)
    print("\n" + res.round(1).to_string(index=False))
    print(f"\nbest: {best[1]}")
    best[2].to_parquet(EXP / "work" / "lomo_best_pred.parquet", index=False)
    print("\nper-month RMSE of the best variant:")
    print(pd.Series(pers[best[1]]).round(0).to_string())
