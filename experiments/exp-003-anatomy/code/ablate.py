"""Fixed-LightGBM ablation harness for the bulk (NM join, taxi <= 3600 s).

Frozen model (never tuned): the only thing that changes between rows of the
ablation table is the feature set. Target is the offset D = BLOCK - AOBT_3;
taxi_hat = nm_taxi - D_hat, so RMSE on taxi equals RMSE on D exactly.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import prc.data as d
import prc.stand_features as sf
from features import ART, CAT, BASE_NUM, OCC, NBR, CSN

THREADS = 16
PARAMS = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=0,
              max_cat_threshold=64, cat_smooth=20.0)
ROUNDS = 1200
# Fold definitions come from prc.data.FOLDS, never from a second copy here.
FOLDS = {name: (list(f.train_months), f.valid_month) for name, f in d.FOLDS.items()}
OCC_B = [c + "_b" for c in OCC]
STAT = None  # filled by StandStats.transform column names

SAMEFLIGHT = ["same_flight_arr", "airborne"]
OCC_PREVARR = ["taxi_ub_pm", "taxi_ub_pa", "off_lb_pm", "off_lb_pa",
               "arr_taxi_pa", "arr_turnaround_pa", "arr_td_gap_pa", "same_type_pa"]
OCC_NEXTARR = ["taxi_lb_nm", "taxi_lb_na", "off_ub_na", "off_ub_nm",
               "arr_taxi_na", "same_type_na"]
OCC_NBRDEP = ["dep_prev_aobt_gap", "dep_prev_mvt_gap", "dep_next_aobt_gap", "dep_next_mvt_gap"]
assert set(OCC_PREVARR + OCC_NEXTARR + OCC_NBRDEP) == set(OCC), "OCC split must be exact"

BLOCKS = {
    "base": CAT + BASE_NUM,
    "base_nosf": [c for c in CAT + BASE_NUM if c not in SAMEFLIGHT],
    "sameflight": SAMEFLIGHT,
    "nbr": NBR,
    "csn": CSN,
    "occ": OCC,
    "occ_prevarr": OCC_PREVARR,
    "occ_nextarr": OCC_NEXTARR,
    "occ_nbrdep": OCC_NBRDEP,
    "occb": OCC_B,
    "stats": "STATS",
    "phi": "PHI",
}


def fit_phi(tr, keys=("apt", "apt_op"), n0=200.0):
    """Least-squares phi in `BLOCK - AOBT = phi * (LOBT - AOBT)`, per group.

    Routed from the lirf worker (exp-006). Fitted on training months only,
    shrunk toward the airport value and then the global value with a pseudo-count
    of `n0` rows, so a small operator at a small airport falls back gracefully.
    Superseded by `prc/phi_features.py` when that lands.
    """
    t = tr[tr["has_nm"] & tr["y"].le(3600) & tr["off"].notna()].copy()
    t["s"] = -t["aobt_lobt"]                       # LOBT - AOBT
    t = t[t["s"].abs() > 60]                       # phi is unidentified when s ~ 0
    num = (t["off"] * t["s"]).groupby(t["apt"], observed=True).sum()
    den = (t["s"] ** 2).groupby(t["apt"], observed=True).sum()
    g_num, g_den = float((t["off"] * t["s"]).sum()), float((t["s"] ** 2).sum())
    phi_global = g_num / g_den
    cnt_apt = t.groupby("apt", observed=True).size()
    phi_apt = ((num / den) * cnt_apt + phi_global * n0) / (cnt_apt + n0)
    t["apt_op"] = t["apt"].astype(str) + "|" + t["AIRCRAFT_OPERATOR_flt"].astype(str)
    num2 = (t["off"] * t["s"]).groupby(t["apt_op"], observed=True).sum()
    den2 = (t["s"] ** 2).groupby(t["apt_op"], observed=True).sum()
    cnt2 = t.groupby("apt_op", observed=True).size()
    parent = pd.Series(t.groupby("apt_op", observed=True)["apt"].first()).map(phi_apt)
    phi_apt_op = ((num2 / den2) * cnt2 + parent * n0) / (cnt2 + n0)
    return {"global": phi_global, "apt": phi_apt, "apt_op": phi_apt_op}


def phi_features(df, fit):
    apt = df["apt"].astype(object)
    key = df["apt"].astype(str) + "|" + df["AIRCRAFT_OPERATOR_flt"].astype(str)
    p_apt = apt.map(fit["apt"]).astype(float).fillna(fit["global"])
    p_op = key.map(fit["apt_op"]).astype(float)
    p_op = p_op.where(p_op.notna(), p_apt)
    s = -df["aobt_lobt"]                            # LOBT - AOBT
    out = pd.DataFrame(index=df.index)
    out["phi_apt"] = p_apt.to_numpy()
    out["phi_apt_op"] = p_op.to_numpy()
    out["phi_delta_apt"] = (p_apt * s).to_numpy()   # candidate offset from the airport phi
    out["phi_delta_op"] = (p_op * s).to_numpy()     # candidate offset from the operator phi
    out["phi_taxi_apt"] = (df["nm_taxi"] - p_apt * s).to_numpy()
    out["phi_taxi_op"] = (df["nm_taxi"] - p_op * s).to_numpy()
    return out


def load_fold(name, cache={}):
    if name in cache:
        return cache[name]
    tr_m, va_m = FOLDS[name]
    tr = pd.concat([pd.read_parquet(ART / f"{m}.parquet") for m in tr_m], ignore_index=True)
    va = pd.read_parquet(ART / f"{va_m}.parquet")
    for f in (tr, va):
        f["y"] = f["TAXITIME_SEC_mvt"].astype("float64")
        f["off"] = (f["BLOCK_TIME_UTC_mvt"] - f["AOBT_3_flt"]).dt.total_seconds()
    cache.clear(); cache[name] = (tr, va)
    return tr, va


def add_stats(tr, va):
    """Fit per-stand offset statistics on training months, transform both."""
    st = sf.StandStats(min_count=30).fit(tr.assign(blk_aobt=tr["off"]))
    a = st.transform(tr.assign(blk_aobt=tr["off"]))
    b = st.transform(va.assign(blk_aobt=va["off"]))
    return a, b, list(a.columns)


def prep(df, feats):
    X = df[feats].copy()
    for c in feats:
        if c in CAT:
            X[c] = X[c].astype("category")
    return X


def slices(y, pred, has_nm, unseen=None):
    extra = {}
    if unseen is not None and unseen.any():
        b = has_nm & (y <= 3600)
        extra["rmse_bulk_unseen_stand"] = d.rmse(y[b & unseen], pred[b & unseen])
        extra["n_bulk_unseen_stand"] = int((b & unseen).sum())
    return extra | {
        "rmse_all": d.rmse(y, pred),
        "rmse_le3600": d.rmse(y[y <= 3600], pred[y <= 3600]),
        "rmse_gt3600": d.rmse(y[y > 3600], pred[y > 3600]),
        "rmse_noNM": d.rmse(y[~has_nm], pred[~has_nm]) if (~has_nm).any() else np.nan,
        "rmse_bulk": d.rmse(y[has_nm & (y <= 3600)], pred[has_nm & (y <= 3600)]),
        "n_bulk": int((has_nm & (y <= 3600)).sum()),
    }


def run(name, blocks, folds=("july", "december"), rounds=ROUNDS, save_oof=None, log=None,
        target="offset", clip0=False):
    res = {}
    for fold in folds:
        tr, va = load_fold(fold)
        feats = []
        stat_tr = stat_va = extra_tr = extra_va = None
        for b in blocks:
            if BLOCKS[b] == "STATS":
                stat_tr, stat_va, sc = add_stats(tr, va)
                feats += sc
            elif BLOCKS[b] == "PHI":
                fit = fit_phi(tr)
                a, bb = phi_features(tr, fit), phi_features(va, fit)
                extra_tr = a if extra_tr is None else pd.concat([extra_tr, a], axis=1)
                extra_va = bb if extra_va is None else pd.concat([extra_va, bb], axis=1)
                feats += list(a.columns)
            else:
                feats += BLOCKS[b]
        add_tr = [f for f in (stat_tr, extra_tr) if f is not None]
        add_va = [f for f in (stat_va, extra_va) if f is not None]
        TR = pd.concat([tr] + add_tr, axis=1) if add_tr else tr
        VA = pd.concat([va] + add_va, axis=1) if add_va else va
        mtr = TR["has_nm"] & TR["y"].le(3600) & TR["off"].notna()
        t0 = time.time()
        if target == "taxi":
            label = TR.loc[mtr, "y"]
        elif target == "phi_resid":
            label = TR.loc[mtr, "off"] - TR.loc[mtr, "phi_delta_op"]
        else:
            label = TR.loc[mtr, "off"]
        ds = lgb.Dataset(prep(TR[mtr], feats), label=label)
        model = lgb.train(PARAMS, ds, num_boost_round=rounds)
        raw = model.predict(prep(VA, feats))
        if target == "taxi":
            pred = raw
        elif target == "phi_resid":
            pred = VA["nm_taxi"].to_numpy() - (raw + VA["phi_delta_op"].to_numpy())
        else:
            pred = VA["nm_taxi"].to_numpy() - raw
        # fallback for rows with no NM join: airport median taxi from training
        med = TR.loc[TR["y"].notna()].groupby("apt", observed=True)["y"].median()
        fb = VA["apt"].map(med).fillna(TR["y"].median()).to_numpy()
        pred = np.where(VA["has_nm"].to_numpy(), pred, fb)
        if clip0:
            pred = np.maximum(pred, 0.0)
        y = VA["y"].to_numpy()
        unseen = ~VA["key_raw"].isin(set(TR["key_raw"].unique())).to_numpy()
        s = slices(y, pred, VA["has_nm"].to_numpy(), unseen)
        s["secs"] = round(time.time() - t0)
        s["n_feat"] = len(feats)
        res[fold] = s
        if save_oof:
            outp = d.ROOT / "experiments/exp-003-anatomy/oof" / f"{save_oof}_{fold}.parquet"
            pd.DataFrame({"MVT_ID_mvt": VA["MVT_ID_mvt"], "fold": fold,
                          "y_true": y, "y_pred": pred}).to_parquet(outp, index=False)
        print(f"[{name}] {fold}: bulk={s['rmse_bulk']:.1f} all={s['rmse_all']:.1f} "
              f"le={s['rmse_le3600']:.1f} gt={s['rmse_gt3600']:.0f} noNM={s['rmse_noNM']:.0f} "
              f"unseen={s.get('rmse_bulk_unseen_stand', float('nan')):.0f}"
              f"(n={s.get('n_bulk_unseen_stand', 0)}) ({s['secs']}s, {len(feats)} feat)", flush=True)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({"name": name, "blocks": blocks, "target": target, "res": res}) + "\n")
    return res


if __name__ == "__main__":
    log = d.ROOT / "experiments/exp-003-anatomy/climb.log"
    plan = [
        ("A_base", ["base"]),
        ("B_base+stats", ["base", "stats"]),
        ("C_base+occ", ["base", "occ"]),
        ("D_base+nbr", ["base", "nbr"]),
        ("E_base+stats+occ", ["base", "stats", "occ"]),
        ("F_all", ["base", "stats", "occ", "occb", "nbr"]),
    ]
    only = sys.argv[1:] or None
    for name, blocks in plan:
        if only and name not in only:
            continue
        run(name, blocks, save_oof=name if name == "F_all" else None, log=log)
