"""exp-018b: exp-018's model with seed bagging, the repo's modules, and the
four-row training drop.

Everything the fit does is unchanged from `exp-018-fresh/code/model.py` except:

* `seeds` bags n fits of the main head (and of CatBoost) and averages them;
* `extras` selects cached module columns to add to the main head's features;
* `drop_unreached` applies `prc.data.keep_reachable` to the TRAINING rows only;
* the SCHED-anchor classifier is fitted **once per fold on the base feature
  set** and reused across configurations. That makes a module ablation change
  only the main head, which is what it claims to change, and it removes a
  three-minute refit from every arm.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

_F = Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"
sys.path.insert(0, str(_F))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D            # noqa: E402
import features as F           # noqa: E402
import model as M              # noqa: E402
import extras as X             # noqa: E402
from prc import data           # noqa: E402

#: Desk allocation for this lane. Raised 6 -> 10 on 2026-09-06 when the stacker
#: lane closed. One place, so the next change is one line or one env var, and
#: the exact head and the main head can never drift apart on it.
THREADS = int(os.environ.get("PRC_THREADS", "10"))


def attach(train: pd.DataFrame, valid: pd.DataFrame, fold_name: str, kinds):
    """Merge the cached module columns for `kinds` onto the two frames."""
    if not kinds:
        return train, valid, []
    etr, eva = X.build_rank() if fold_name == "rank" else X.build(fold_name)
    cols = []
    for k in kinds:
        cols += X.columns_of(k, etr.columns)
    cols = [c for c in dict.fromkeys(cols) if c in etr.columns and c in eva.columns]
    train = train.merge(etr[["MVT_ID_mvt"] + cols], on="MVT_ID_mvt", how="left")
    valid = valid.merge(eva[["MVT_ID_mvt"] + cols], on="MVT_ID_mvt", how="left")
    return train, valid, cols


def fit_exact_head(train: pd.DataFrame, valid: pd.DataFrame, base_feats, rounds=300, seed=7,
                   threads=THREADS):
    """P(|BLOCK - SCHED| <= 6 s | x), fitted on training months, base features only."""
    yt = train["y"].to_numpy(dtype="float64")
    ms_t = train["ms_sched"].to_numpy(dtype="float64")
    gt = train["ghost"].to_numpy() > 0.5
    exact_t = np.isfinite(ms_t) & (np.abs(yt - ms_t) <= 6)
    cfit = (~gt) & np.isfinite(yt)
    cat = [c for c in F.CAT_COLS if c in base_feats]
    p = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=50,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             num_threads=threads, verbose=-1, seed=seed)
    ds = lgb.Dataset(train.loc[cfit, base_feats], label=exact_t[cfit].astype(float),
                     categorical_feature=cat, free_raw_data=False)
    booster = lgb.train(p, ds, num_boost_round=rounds)
    ms_v = valid["ms_sched"].to_numpy(dtype="float64")
    pe = booster.predict(valid[base_feats])
    return np.where(np.isfinite(ms_v) & (ms_v > 0), pe, 0.0), exact_t


def fit_predict(train, valid, fold_name, *, rounds=1200, clip_z=6000.0, seeds=3,
                extras_kinds=(), drop_unreached=False, main_model="both",
                cat_rounds=3000, blend=0.5, ghost_rounds=300, threads=THREADS,
                p_exact=None, exact_t=None, params=None):
    """Returns (pred, info). `train`/`valid` are already encoded by D.add_encodings."""
    base_feats = [c for c in train.columns if c not in M.DROP + M.HIGH_CARD]
    train, valid, added = attach(train, valid, fold_name, extras_kinds)
    feats = base_feats + added
    cat_cols = [c for c in F.CAT_COLS if c in feats]

    if p_exact is None:
        p_exact, exact_t = fit_exact_head(train, valid, base_feats, threads=threads)

    gt = train["ghost"].to_numpy() > 0.5
    gv = valid["ghost"].to_numpy() > 0.5
    base_t, base_v = M._base(train), M._base(valid)
    yt = train["y"].to_numpy(dtype="float64")
    ms_t = train["ms_sched"].to_numpy(dtype="float64")
    ms_v = valid["ms_sched"].to_numpy(dtype="float64")
    z = yt - base_t

    fit = (~gt) & np.isfinite(z) & (~exact_t)
    if drop_unreached:
        # keep_reachable returns a mask on the frame's own index; the left-merge
        # in attach() preserves row order, so a positional AND is correct
        fit = fit & data.keep_reachable(train).to_numpy()
    n_dropped = int(((~gt) & np.isfinite(z) & (~exact_t)).sum() - fit.sum())
    zt = np.clip(z[fit], -clip_z, clip_z) if clip_z else z[fit]

    Xtr, Xva = train.loc[fit, feats], valid[feats]
    raws = []
    for s in range(seeds):
        p = dict(M.MAIN_PARAMS); p["num_threads"] = threads
        if params:
            p.update(params)
        p["seed"] = 7 + 13 * s; p["bagging_seed"] = p["seed"] + 1
        p["feature_fraction_seed"] = p["seed"] + 2
        ds = lgb.Dataset(Xtr, label=zt, categorical_feature=cat_cols, free_raw_data=False)
        raws.append(lgb.train(p, ds, num_boost_round=rounds).predict(Xva))
    raw = np.mean(raws, axis=0)

    if main_model in ("cat", "both"):
        import catboost as cb
        cidx = [feats.index(c) for c in cat_cols]
        Xc, Xvc = Xtr.copy(), Xva.copy()
        for c in cat_cols:
            Xc[c] = Xc[c].astype("int32"); Xvc[c] = Xvc[c].astype("int32")
        pool = cb.Pool(Xc, label=zt, cat_features=cidx)
        vpool = cb.Pool(Xvc, cat_features=cidx)
        craws = []
        for s in range(seeds):
            cm = cb.CatBoostRegressor(loss_function="RMSE", iterations=cat_rounds,
                                      learning_rate=0.06, depth=8, l2_leaf_reg=6.0,
                                      random_seed=7 + 13 * s, verbose=0,
                                      task_type="GPU", devices="0", max_ctr_complexity=2)
            cm.fit(pool)
            craws.append(cm.predict(vpool))
        rawc = np.mean(craws, axis=0)
        raw = rawc if main_model == "cat" else (1 - blend) * raw + blend * rawc

    pred_main = np.clip(base_v + raw, 30.0, None)
    pred = p_exact * np.nan_to_num(ms_v) + (1 - p_exact) * pred_main

    # ---- ghost head (mixbase), unchanged from exp-018 -----------------------
    gfit = gt & (ms_t > 60) & np.isfinite(yt)
    gpred = gv & np.isfinite(ms_v) & (ms_v > 60)
    if gfit.sum() >= 50 and gpred.sum() > 0:
        w = ms_t[gfit] ** 2; w = w / w.mean()
        gcats = [c for c in M.GHOST_FEATS if c in F.CAT_COLS]
        lab = (yt[gfit] / ms_t[gfit] > 0.5).astype(float)
        gp = dict(M.GHOST_PARAMS); gp["objective"] = "binary"; gp["num_threads"] = threads
        pas = []
        for s in range(seeds):
            gp["seed"] = 7 + 13 * s
            gds = lgb.Dataset(train.loc[gfit, M.GHOST_FEATS], label=lab, weight=w,
                              categorical_feature=gcats, free_raw_data=False)
            pas.append(lgb.train(gp, gds, num_boost_round=ghost_rounds)
                       .predict(valid.loc[gpred, M.GHOST_FEATS]))
        pa = np.mean(pas, axis=0)
        pred[gpred] = pa * ms_v[gpred] + (1 - pa) * base_v[gpred]

    pred = np.clip(pred, 30.0, None)
    return pred, {"feats": feats, "n_added": len(added), "added": added,
                  "n_train": int(fit.sum()), "n_dropped": n_dropped, "seeds": seeds}
