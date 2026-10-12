"""exp-002-tail: fixed bulk model + tail mechanisms, one climb-log line each."""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
import prc.data as D

ART = D.ROOT / "artifacts" / "exp-002-tail"
EXP = D.ROOT / "experiments" / "exp-002-tail"
NTHREAD = 16
CATS = ["apt", "stand", "runway"]
NUM = ["hour", "dow", "minute_of_day", "noac", "nonm", "vfr", "flight_len",
       "mvt_sched", "mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt",
       "aobt_lobt", "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
       "noac_day_count", "nonm_day_count"]
STAND = ["gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
         "n_arr_sched_mvt", "gap_arr_after_sched", "gap_dep_prev_mvt",
         "gap_dep_prev_aobt", "gap_dep_next_mvt", "stand_busy_frac_6h"]
# the assignment's fixed bulk model: own-flight deltas + airport, stand, runway, hour
BULK = CATS + ["hour", "dow", "minute_of_day", "mvt_sched", "mvt_lobt", "mvt_iobt",
               "mvt_eobt", "mvt_aobt", "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched"]
TAILF = ["noac", "nonm", "vfr", "flight_len", "aobt_lobt", "noac_day_count", "nonm_day_count"]


def load(months):
    df = pd.concat([pd.read_parquet(ART / f"signals_{m}.parquet") for m in months], ignore_index=True)
    return df


def harmonise(*frames):
    """One category mapping across frames so LightGBM sees consistent codes."""
    for c in CATS:
        cats = pd.Index(sorted(set().union(*[set(f[c].astype(str).unique()) for f in frames])))
        for f in frames:
            f[c] = pd.Categorical(f[c].astype(str), categories=cats)
    return frames


def anchor_base(df):
    """MVT minus the anchor: NM actual off-block when present, else the schedule."""
    return np.where(df.nonm.to_numpy() == 0, df.mvt_aobt.to_numpy(), df.mvt_sched.to_numpy())


def metrics(y, p, df, name):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e2 = (y - p) ** 2
    def r(m):
        return float(np.sqrt(e2[m].mean())) if m.sum() else float("nan")
    tail = y > 3600
    return dict(model=name, n=len(y), rmse=r(np.ones(len(y), bool)), rmse_le3600=r(~tail),
                rmse_gt3600=r(tail), rmse_nonm=r(df.nonm.to_numpy() == 1),
                rmse_noac=r(df.noac.to_numpy() == 1),
                se_share_noac=float(e2[df.noac.to_numpy() == 1].sum() / e2.sum()),
                se_share_gt3600=float(e2[tail].sum() / e2.sum()))


def fit_lgb(tr, va, feats, target_tr, objective="regression", **kw):
    params = dict(objective=objective, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_estimators=1200,
                  n_jobs=NTHREAD, verbose=-1, seed=7)
    params.update(kw)
    m = lgb.LGBMRegressor(**params) if objective != "binary" else lgb.LGBMClassifier(**params)
    m.fit(tr[feats], target_tr, categorical_feature=[c for c in CATS if c in feats])
    return m


# ---------------------------------------------------------------- iterations
def run_fold(fold_name, iters, log_rows):
    fold = D.FOLDS[fold_name]
    tr = load(fold.train_months); va = load([fold.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    yv = va.y.to_numpy(float); ytr = tr.y.to_numpy(float)
    base_tr, base_va = anchor_base(tr), anchor_base(va)
    o_tr = base_tr - ytr                      # BLOCK - anchor
    res = {}

    if "it0" in iters:
        med = tr.groupby("apt", observed=True).y.median()
        p = np.where(np.isnan(va.mvt_aobt.to_numpy()), va.apt.map(med).astype(float).to_numpy(), va.mvt_aobt.to_numpy())
        res["it0-naive"] = (p, metrics(yv, p, va, "it0-naive"))

    if "it1" in iters:
        m = fit_lgb(tr, va, BULK, ytr)
        p = m.predict(va[BULK]); res["it1-bulk-y"] = (p, metrics(yv, p, va, "it1-bulk-y"))

    if "it2" in iters:
        m = fit_lgb(tr, va, BULK, o_tr)
        p = base_va - m.predict(va[BULK]); res["it2-bulk-offset"] = (p, metrics(yv, p, va, "it2-bulk-offset"))

    if "it3" in iters:
        f = BULK + TAILF
        m = fit_lgb(tr, va, f, o_tr)
        p = base_va - m.predict(va[f]); res["it3-+tailfeat"] = (p, metrics(yv, p, va, "it3-+tailfeat"))

    if "it4" in iters:
        f = BULK + TAILF + STAND
        m = fit_lgb(tr, va, f, o_tr)
        p = base_va - m.predict(va[f]); res["it4-+stand"] = (p, metrics(yv, p, va, "it4-+stand"))

    for k, (p, mt) in res.items():
        mt["fold"] = fold_name; log_rows.append(mt)
        np.save(EXP / "work" / f"pred_{fold_name}_{k}.npy", p)
    va[["MVT_ID_mvt", "y", "noac", "nonm", "mvt_sched"]].to_parquet(EXP / "work" / f"va_{fold_name}.parquet")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", default="it0,it1,it2,it3,it4")
    ap.add_argument("--folds", default="july,december,january")
    a = ap.parse_args()
    rows = []
    for f in a.folds.split(","):
        t0 = time.time(); run_fold(f, a.iters.split(","), rows)
        print(f"[{f}] {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)[["fold", "model", "n", "rmse", "rmse_le3600", "rmse_gt3600", "rmse_nonm", "rmse_noac", "se_share_noac", "se_share_gt3600"]]
    print(df.round(3).to_string(index=False))
    df.to_csv(EXP / "work" / "metrics_stage1.csv", index=False)
