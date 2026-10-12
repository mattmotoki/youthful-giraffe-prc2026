"""exp-002-tail: the full climb, one line per mechanism, on all three folds.

Fixed bulk model = LightGBM on own-flight deltas plus airport, stand, runway,
hour (assignment question 3). Every later line changes exactly one mechanism on
top of it, so the reported gain is attributable.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from noac_model import fit_predict as noac_fit_predict

ART = D.ROOT / "artifacts" / "exp-002-tail"
EXP = D.ROOT / "experiments" / "exp-002-tail"
NTHREAD = 16
NR, NN_RANK = 344841, 388
CATS = ["apt", "stand", "runway"]
BULK = CATS + ["hour", "dow", "minute_of_day", "mvt_sched", "mvt_lobt", "mvt_iobt",
               "mvt_eobt", "mvt_aobt", "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched"]
TAILF = ["noac", "nonm", "vfr", "flight_len", "aobt_lobt", "noac_day_count", "nonm_day_count"]
STAND = ["gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
         "n_arr_sched_mvt", "gap_arr_after_sched", "gap_dep_prev_mvt",
         "gap_dep_prev_aobt", "gap_dep_next_mvt", "stand_busy_frac_6h"]
FULL = BULK + TAILF + STAND
BASEP = dict(learning_rate=0.08, num_leaves=63, min_child_samples=60, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, n_estimators=500, n_jobs=NTHREAD,
             verbose=-1, seed=7, max_cat_threshold=64, cat_smooth=20)


def load(months):
    return pd.concat([pd.read_parquet(ART / f"signals_{m}.parquet") for m in months], ignore_index=True)


def harmonise(*fr):
    for c in CATS:
        cats = pd.Index(sorted(set().union(*[set(f[c].astype(str).unique()) for f in fr])))
        for f in fr:
            f[c] = pd.Categorical(f[c].astype(str), categories=cats)
    return fr


def anchor_base(df):
    return np.where(df.nonm.to_numpy() == 0, df.mvt_aobt.to_numpy(), df.mvt_sched.to_numpy())


def reg(tr, feats, target, **kw):
    p = dict(BASEP); p.update(kw)
    m = lgb.LGBMRegressor(objective=p.pop("objective", "regression"), **p)
    m.fit(tr[feats], target, categorical_feature=[c for c in CATS if c in feats])
    return m


def clf(tr, feats, target, **kw):
    p = dict(BASEP); p.update(kw)
    m = lgb.LGBMClassifier(objective="binary", **p)
    m.fit(tr[feats], target, categorical_feature=[c for c in CATS if c in feats])
    return m


def metrics(y, p, df, name, fold):
    y = np.asarray(y, float); p = np.asarray(p, float); e2 = (y - p) ** 2
    tail = y > 3600; nonm = df.nonm.to_numpy() == 1; noac = df.noac.to_numpy() == 1
    r = lambda m: float(np.sqrt(e2[m].mean())) if m.sum() else np.nan
    return dict(fold=fold, model=name, rmse=r(np.ones(len(y), bool)), rmse_le3600=r(~tail),
                rmse_gt3600=r(tail), rmse_nonm=r(nonm), rmse_noac=r(noac),
                noac_contrib_s=float(np.sqrt(NN_RANK * e2[noac].mean() / NR)) if noac.sum() else np.nan,
                se_share_noac=float(e2[noac].sum() / e2.sum()), se_share_gt3600=float(e2[tail].sum() / e2.sum()))


def run(fold_name, rows, oof):
    f = D.FOLDS[fold_name]
    tr = load(f.train_months); va = load([f.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    ytr, yv = tr.y.to_numpy(float), va.y.to_numpy(float)
    b_tr, b_va = anchor_base(tr), anchor_base(va)
    o_tr = b_tr - ytr
    P = {}
    t0 = time.time()
    T = lambda s: print(f"    {s} {time.time()-t0:.0f}s", flush=True)

    med = tr.groupby("apt", observed=True).y.median()
    P["it0-naive"] = np.where(np.isnan(va.mvt_aobt.to_numpy()),
                              va.apt.map(med).astype(float).to_numpy(), va.mvt_aobt.to_numpy())
    P["it1-bulk-target-y"] = reg(tr, BULK, ytr).predict(va[BULK]); T("it1")
    P["it2-bulk-target-offset"] = b_va - reg(tr, BULK, o_tr).predict(va[BULK]); T("it2")
    P["it3-plus-tail-signals"] = b_va - reg(tr, BULK + TAILF, o_tr).predict(va[BULK + TAILF]); T("it3")
    P["it4-plus-stand-occupancy"] = b_va - reg(tr, FULL, o_tr).predict(va[FULL]); T("it4")

    c = clf(tr, FULL, (ytr > 3600).astype(int))
    p_tail = c.predict_proba(va[FULL])[:, 1]
    lo = ytr <= 3600
    y_lo = b_va - reg(tr[lo].reset_index(drop=True), FULL, o_tr[lo]).predict(va[FULL])
    y_hi = b_va - reg(tr[~lo].reset_index(drop=True), FULL, o_tr[~lo],
                      num_leaves=15, n_estimators=300, min_child_samples=10).predict(va[FULL])
    P["it5-twostage-expectation"] = (1 - p_tail) * y_lo + p_tail * y_hi
    P["it5-twostage-hardswitch"] = np.where(p_tail >= 0.5, y_hi, y_lo); T("it5")

    P["it6-huber-bulk"] = b_va - reg(tr, FULL, o_tr, objective="huber", alpha=300.0).predict(va[FULL])
    y_hi_q = b_va - reg(tr[~lo].reset_index(drop=True), FULL, o_tr[~lo], objective="quantile",
                        alpha=0.5, num_leaves=15, n_estimators=300, min_child_samples=10).predict(va[FULL])
    P["it6-twostage-shrunk-tail"] = (1 - p_tail) * y_lo + p_tail * y_hi_q; T("it6")

    NOFLT = [x for x in FULL if x not in ("mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt", "aobt_lobt",
                                          "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
                                          "gap_dep_prev_aobt")]
    nn_tr = tr[tr.nonm == 1].reset_index(drop=True); nn_va = va.nonm.to_numpy() == 1
    m_nn = reg(nn_tr, NOFLT, anchor_base(nn_tr) - nn_tr.y.to_numpy(float),
               num_leaves=31, n_estimators=400, min_child_samples=20)
    sep = P["it4-plus-stand-occupancy"].copy()
    sep[nn_va] = (b_va - m_nn.predict(va[NOFLT]))[nn_va]
    P["it7-separate-nonm-model"] = sep; T("it7")

    na_m = va.noac.to_numpy() == 1
    na_tr = tr[tr.noac == 1].reset_index(drop=True); na_va = va[na_m].reset_index(drop=True)
    if na_m.sum() and len(na_tr) > 30:
        o = noac_fit_predict(na_tr, na_va)
        for k in ("const-mean", "band-mean", "lgb-direct-y", "lgb-direct-offset",
                  "mixture-expectation", "mixture-hardswitch"):
            q = P["it4-plus-stand-occupancy"].copy(); q[na_m] = o[k]; P[f"it8-noac:{k}"] = q
        q = P["it4-plus-stand-occupancy"].copy(); q[na_m] = yv[na_m]; P["it8-noac:ORACLE"] = q
    T("it8")

    for k, p in P.items():
        rows.append(metrics(yv, p, va, k, fold_name))
    bestk = "it8-noac:lgb-direct-offset" if "it8-noac:lgb-direct-offset" in P else "it4-plus-stand-occupancy"
    oof.append(pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold_name,
                             "y_true": yv, "y_pred": P[bestk]}))
    np.savez_compressed(EXP / "work" / f"preds_{fold_name}.npz", **P)


if __name__ == "__main__":
    rows, oof = [], []
    for fd in ("july", "december", "january"):
        t1 = time.time(); print(f"[{fd}]", flush=True); run(fd, rows, oof)
        print(f"[{fd}] done {time.time()-t1:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(EXP / "work" / "metrics_all.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    df.to_csv(EXP / "work" / "metrics_all.csv", index=False)
    # NOT the delivered OOF: this harness measures the early climb lines only.
    # The delivered artifact is written by deliver.py from prc.tail_model.
    pd.concat(oof, ignore_index=True).to_parquet(EXP / "work" / "it8_climb_oof.parquet", index=False)
