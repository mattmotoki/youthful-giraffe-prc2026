"""exp-018 fresh lineage: the model.

Structure (SPEC.md): the target is decomposed into a *base* that already
carries the bulk of the signal and a learned residual.

  base = MVT_TIME - AOBT_3           when the Network Manager row joins
       = encoded median taxi         otherwise
  z    = TAXITIME_SEC - base         learned by LightGBM (L2)

Departures with a null AIRCRAFT_TYPE_mvt ("ghosts", 0.11 % of the ranking
file, essentially all LIRF) are a different generative process: about half of
them carry BLOCK_TIME == SCHED_TIME, so their label is exactly
MVT_TIME - SCHED_TIME.  They are removed from the main fit and given their own
ratio model, y = rho * (MVT_TIME - SCHED_TIME), fitted with weight ms^2 so the
squared error in the ratio space equals the squared error in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402
import features as F  # noqa: E402
from prc import data  # noqa: E402

DROP = ["y", "MVT_ID_mvt", "month", "hourbin", "ghost"]
#: high-cardinality identifiers.  Their signal is already in the target
#: encodings (te_y_stand, te_off_oper, ...) and in the in-month arrival
#: geometry (stand_taxiin_med); leaving them in as raw LightGBM categoricals
#: costs a large multiple of the fit time for a split search over thousands of
#: levels at every node.
HIGH_CARD = ["c_stand", "c_oper", "c_ades", "c_csgn"]

MAIN_PARAMS = dict(
    objective="l2", learning_rate=0.05, num_leaves=255, min_data_in_leaf=60,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, max_cat_threshold=64, cat_smooth=20.0, cat_l2=10.0,
    num_threads=6, verbose=-1, seed=7,
)
GHOST_PARAMS = dict(
    objective="l2", learning_rate=0.04, num_leaves=15, min_data_in_leaf=20,
    feature_fraction=0.7, bagging_fraction=0.9, bagging_freq=1, lambda_l2=5.0,
    num_threads=6, verbose=-1, seed=7,
)
GHOST_FEATS = ["ms_sched", "hour", "dow", "stand_prev_arr_inblock_dt", "stand_next_arr_inblock_dt",
               "stand_prev_dep_dt", "stand_next_dep_dt", "stand_bound_minus_ms", "stand_bound_over_ms",
               "apt_to_30m_before", "apt_land_30m_before", "arr_taxiin_last30", "n_stand_month",
               "te_y_stand", "te_y_standrwy", "te_y_rwy", "c_rwy", "c_stand", "c_adescty", "c_csgn"]


def _base(df: pd.DataFrame) -> np.ndarray:
    b = df["r_aobt"].to_numpy(dtype="float64").copy()
    fb = df["te_y_standrwy"].to_numpy(dtype="float64")
    fb = np.where(np.isfinite(fb), fb, df["te_y_rwy"].to_numpy(dtype="float64"))
    fb = np.where(np.isfinite(fb), fb, df["te_y_apt"].to_numpy(dtype="float64"))
    return np.where(np.isfinite(b), b, fb)


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, *, rounds: int = 1500,
                clip_z: float | None = None, params: dict | None = None,
                ghost_rounds: int = 300, seed: int | None = None, curve: int = 0,
                y_valid: np.ndarray | None = None, exact_mix: bool = True,
                exact_rounds: int = 400, ghost_mode: str = "ratio", prepared: bool = False,
                cats: str = "lean", exact_label: str = "eq6", exact_weight: str | None = None,
                main_model: str = "lgb", cat_params: dict | None = None, blend: float = 0.5):
    """Fit on `train` (labelled) and predict `valid`. Returns (pred, info).

    Three heads, all fitted on the training months only:

    main    LightGBM on z = y - base over ordinary departures.
    exact   P(BLOCK_TIME == SCHED_TIME | x).  When that holds the label is
            exactly MVT - SCHED, so the final prediction is the posterior
            mixture p*ms + (1-p)*main.  4.7 % of departures are such rows and
            they carry 13.5 % of the squared error of the MVT-AOBT baseline.
    ghost   the null-aircraft-type class gets a ratio model,
            y = rho(x) * (MVT - SCHED), fitted with weight ms^2 so squared
            error in ratio space is squared error in seconds.  A posterior
            mixture was tried here too and is much worse (8308 s against
            4013 s on the july fold): rows whose ratio is near but not exactly
            1 fall on the "not exact" side of the label and are then dragged
            to the class mean, and those are precisely the 80,000 s rows.
    """
    if not prepared:
        train, (valid,) = D.add_encodings(train, [valid])

    drop = list(DROP) + (HIGH_CARD if cats == "lean" else [])
    feats = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in F.CAT_COLS if c in feats]
    p = dict(MAIN_PARAMS)
    if params:
        p.update(params)
    if seed is not None:
        p["seed"] = seed
        p["bagging_seed"] = seed + 1
        p["feature_fraction_seed"] = seed + 2

    gt = train["ghost"].to_numpy() > 0.5
    gv = valid["ghost"].to_numpy() > 0.5
    base_t, base_v = _base(train), _base(valid)
    yt = train["y"].to_numpy(dtype="float64")
    ms_t = train["ms_sched"].to_numpy(dtype="float64")
    ms_v = valid["ms_sched"].to_numpy(dtype="float64")
    z = yt - base_t
    tol = {"eq6": 6.0, "eq60": 60.0, "eq300": 300.0}.get(exact_label)
    if tol is not None:
        exact_t = np.isfinite(ms_t) & (np.abs(yt - ms_t) <= tol)
    else:                                    # "ratio": near one rather than equal
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = yt / ms_t
        exact_t = np.isfinite(rr) & (rr >= 0.92) & (rr <= 1.03)

    # ---- exact head -----------------------------------------------------
    p_exact = np.zeros(len(valid))
    if exact_mix:
        cfit = (~gt) & np.isfinite(yt)
        cp = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  num_threads=p["num_threads"], verbose=-1, seed=p["seed"])
        cw = None
        if exact_weight == "gap2":
            # the cost of a wrong posterior is (ms - main)^2 * (error in p)^2, so
            # weight the classifier by how much the decision is worth in seconds
            gap = np.abs(np.nan_to_num(ms_t) - base_t)[cfit]
            cw = gap ** 2
            cw = cw / cw.mean()
        cds = lgb.Dataset(train.loc[cfit, feats], label=exact_t[cfit].astype(float), weight=cw,
                          categorical_feature=cat_cols, free_raw_data=False)
        cb = lgb.train(cp, cds, num_boost_round=exact_rounds)
        p_exact = cb.predict(valid[feats])
        p_exact = np.where(np.isfinite(ms_v) & (ms_v > 0), p_exact, 0.0)

    # ---- main head ------------------------------------------------------
    fit = (~gt) & np.isfinite(z)
    if exact_mix:
        fit = fit & (~exact_t)
    zt = z[fit]
    if clip_z is not None:
        zt = np.clip(zt, -clip_z, clip_z)
    ds = lgb.Dataset(train.loc[fit, feats], label=zt, categorical_feature=cat_cols, free_raw_data=False)
    if curve and y_valid is not None:
        booster = None
        Xv = valid[feats]
        keep = np.isfinite(y_valid) & (~gv)
        for start in range(0, rounds, curve):
            booster = lgb.train(p, ds, num_boost_round=curve, init_model=booster,
                                keep_training_booster=True)
            pm = np.clip(base_v + booster.predict(Xv), 30.0, None)
            pv = p_exact * np.nan_to_num(ms_v) + (1 - p_exact) * pm
            print(f"    round {start + curve:5d}  rmse_nonghost={np.sqrt(np.mean((y_valid[keep] - pv[keep]) ** 2)):8.2f}"
                  f"  main_only={np.sqrt(np.mean((y_valid[keep] - pm[keep]) ** 2)):8.2f}", flush=True)
    else:
        booster = lgb.train(p, ds, num_boost_round=rounds)
    raw = booster.predict(valid[feats])
    if main_model in ("cat", "both"):
        import catboost as cb
        cpar = dict(loss_function="RMSE", iterations=3000, learning_rate=0.06, depth=8,
                    l2_leaf_reg=6.0, random_seed=p["seed"], verbose=0,
                    task_type="GPU", devices="0", max_ctr_complexity=2)
        if cat_params:
            cpar.update(cat_params)
        cidx = [feats.index(c) for c in cat_cols]
        Xtr = train.loc[fit, feats].copy()
        Xva = valid[feats].copy()
        for c in cat_cols:
            Xtr[c] = Xtr[c].astype("int32")
            Xva[c] = Xva[c].astype("int32")
        pool_tr = cb.Pool(Xtr, label=zt, cat_features=cidx)
        cm = cb.CatBoostRegressor(**cpar)
        cm.fit(pool_tr)
        rawc = cm.predict(cb.Pool(Xva, cat_features=cidx))
        raw = rawc if main_model == "cat" else (1 - blend) * raw + blend * rawc
    pred_main = np.clip(base_v + raw, 30.0, None)
    pred = p_exact * np.nan_to_num(ms_v) + (1 - p_exact) * pred_main

    # ---- ghost head -----------------------------------------------------
    gfit = gt & (ms_t > 60) & np.isfinite(yt)
    gpred = gv & np.isfinite(ms_v) & (ms_v > 60)
    n_ghost = int(gfit.sum())
    if n_ghost >= 50 and gpred.sum() > 0 and ghost_mode != "off":
        w = ms_t[gfit] ** 2
        w = w / w.mean()
        gcats = [c for c in GHOST_FEATS if c in F.CAT_COLS]
        gfeats = GHOST_FEATS
        if ghost_mode == "ratio":
            rho = np.clip(yt[gfit] / ms_t[gfit], 0.0, 1.05)
            gds = lgb.Dataset(train.loc[gfit, GHOST_FEATS], label=rho, weight=w,
                              categorical_feature=gcats, free_raw_data=False)
            gb = lgb.train(GHOST_PARAMS, gds, num_boost_round=ghost_rounds)
            pred[gpred] = np.clip(gb.predict(valid.loc[gpred, GHOST_FEATS]), 0.02, 1.02) * ms_v[gpred]
        elif ghost_mode == "mixbase":
            # classifier on "the ratio is near one" (a soft label, unlike the
            # exact-equality one that fails here), with the encoded median as
            # the other component so a class-B ghost gets a per-row value
            lab = (yt[gfit] / ms_t[gfit] > 0.5).astype(float)
            gp = dict(GHOST_PARAMS); gp["objective"] = "binary"
            gds = lgb.Dataset(train.loc[gfit, GHOST_FEATS], label=lab, weight=w,
                              categorical_feature=gcats, free_raw_data=False)
            gb = lgb.train(gp, gds, num_boost_round=ghost_rounds)
            pa = gb.predict(valid.loc[gpred, GHOST_FEATS])
            pred[gpred] = pa * ms_v[gpred] + (1 - pa) * base_v[gpred]
        else:                                        # posterior mixture
            lab = (np.abs(yt[gfit] - ms_t[gfit]) <= 6).astype(float)
            gp = dict(GHOST_PARAMS); gp["objective"] = "binary"
            gds = lgb.Dataset(train.loc[gfit, GHOST_FEATS], label=lab, weight=w,
                              categorical_feature=gcats, free_raw_data=False)
            gb = lgb.train(gp, gds, num_boost_round=ghost_rounds)
            pa = gb.predict(valid.loc[gpred, GHOST_FEATS])
            mu_b = float(np.average(yt[gfit][lab < 0.5])) if (lab < 0.5).any() else 1000.0
            pred[gpred] = pa * ms_v[gpred] + (1 - pa) * mu_b
    pred = np.clip(pred, 30.0, None)
    return pred, dict(booster=booster, feats=feats, n_train=int(fit.sum()), n_ghost=n_ghost)


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype="float64")
    lo = y <= 3600
    return dict(rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
                rmse_lo=float(np.sqrt(np.mean((y[lo] - pred[lo]) ** 2))),
                rmse_hi=float(np.sqrt(np.mean((y[~lo] - pred[~lo]) ** 2))),
                n=int(len(y)), n_hi=int((~lo).sum()))
