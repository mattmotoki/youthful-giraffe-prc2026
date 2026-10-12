"""exp-002-tail stage 2: the tail mechanisms of assignment question 3, and the
RMSE-optimal treatment of question 4, each on top of the fixed bulk model."""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from model import (ART, EXP, CATS, BULK, TAILF, STAND, load, harmonise, anchor_base, metrics, fit_lgb, NTHREAD)
from noac_model import fit_predict as noac_fit_predict, PH_TOL

FULL = BULK + TAILF + STAND
NR, NN_RANK = 344841, 388


def run(fold_name, rows, oof):
    fold = D.FOLDS[fold_name]
    tr = load(fold.train_months); va = load([fold.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    ytr, yv = tr.y.to_numpy(float), va.y.to_numpy(float)
    b_tr, b_va = anchor_base(tr), anchor_base(va)
    o_tr = b_tr - ytr
    P = {}

    # --- it4 reference: one model, all features, missingness flags inside ----
    m4 = fit_lgb(tr, va, FULL, o_tr)
    P["it4-onemodel"] = b_va - m4.predict(va[FULL])

    # --- it5 two-stage: P(tail) x conditional regressors, as an expectation --
    t0 = time.time()
    clf = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=63,
                             min_child_samples=40, n_estimators=600, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, n_jobs=NTHREAD, verbose=-1, seed=5)
    clf.fit(tr[FULL], (ytr > 3600).astype(int), categorical_feature=CATS)
    p_tail = clf.predict_proba(va[FULL])[:, 1]
    lo = ytr <= 3600
    m_lo = fit_lgb(tr[lo].reset_index(drop=True), va, FULL, o_tr[lo])
    m_hi = fit_lgb(tr[~lo].reset_index(drop=True), va, FULL, o_tr[~lo],
                   num_leaves=15, n_estimators=300, min_child_samples=10)
    y_lo = b_va - m_lo.predict(va[FULL]); y_hi = b_va - m_hi.predict(va[FULL])
    P["it5-twostage-expect"] = (1 - p_tail) * y_lo + p_tail * y_hi
    P["it5-twostage-hardswitch"] = np.where(p_tail >= 0.5, y_hi, y_lo)
    print(f"  it5 {time.time()-t0:.0f}s  mean p_tail={p_tail.mean():.5f}", flush=True)

    # --- it6 robust / expectile heads ---------------------------------------
    m_hub = fit_lgb(tr, va, FULL, o_tr, objective="huber", alpha=300.0)
    P["it6-huber-bulk"] = b_va - m_hub.predict(va[FULL])
    m_q90 = fit_lgb(tr[~lo].reset_index(drop=True), va, FULL, o_tr[~lo],
                    objective="quantile", alpha=0.5, num_leaves=15, n_estimators=300, min_child_samples=10)
    y_hi_q = b_va - m_q90.predict(va[FULL])
    P["it6-twostage-shrunktail"] = (1 - p_tail) * y_lo + p_tail * y_hi_q

    # --- it7 separate no-NM model vs missingness flags in one model ----------
    NOFLT = [c for c in FULL if c not in ("mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt",
                                          "aobt_lobt", "lobt_sched", "eobt_sched", "lobt_iobt",
                                          "aobt_sched", "gap_dep_prev_aobt")]
    nn_tr = tr[tr.nonm == 1].reset_index(drop=True); nn_va = va.nonm.to_numpy() == 1
    m_nn = fit_lgb(nn_tr, va, NOFLT, anchor_base(nn_tr) - nn_tr.y.to_numpy(float),
                   num_leaves=31, n_estimators=600, min_child_samples=20)
    sep = P["it4-onemodel"].copy()
    sep[nn_va] = (b_va - m_nn.predict(va[NOFLT]))[nn_va]
    P["it7-separate-nonm"] = sep

    # --- it8 noac sub-model, mixture expectation ----------------------------
    na_tr = tr[tr.noac == 1].reset_index(drop=True); na_va_mask = va.noac.to_numpy() == 1
    na_va = va[na_va_mask].reset_index(drop=True)
    best = P["it4-onemodel"]
    if len(na_va) and len(na_tr) > 30:
        o = noac_fit_predict(na_tr, na_va)
        for k in ("const-mean", "band-mean", "lgb-direct-y", "lgb-direct-offset",
                  "mixture-expectation", "mixture-hardswitch"):
            q = best.copy(); q[na_va_mask] = o[k]
            P[f"it8-noac:{k}"] = q
        q = best.copy(); q[na_va_mask] = yv[na_va_mask]
        P["it8-noac:ORACLE"] = q

    for k, p in P.items():
        mt = metrics(yv, p, va, k); mt["fold"] = fold_name
        e2 = (yv - p) ** 2
        mt["noac_contrib_s"] = float(np.sqrt(NN_RANK * e2[na_va_mask].mean() / NR)) if na_va_mask.sum() else np.nan
        rows.append(mt)
    oof.append(pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold_name,
                             "y_true": yv, "y_pred": P["it8-noac:mixture-expectation"]}))
    np.save(EXP / "work" / f"stage2_preds_{fold_name}.npy", np.vstack([P[k] for k in P]))
    pd.Series(list(P)).to_csv(EXP / "work" / f"stage2_keys_{fold_name}.csv", index=False, header=False)


if __name__ == "__main__":
    rows, oof = [], []
    for f in ("july", "december", "january"):
        t0 = time.time(); run(f, rows, oof); print(f"[{f}] {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)[["fold", "model", "rmse", "rmse_le3600", "rmse_gt3600", "rmse_nonm",
                             "rmse_noac", "noac_contrib_s", "se_share_noac"]]
    print(df.round(2).to_string(index=False))
    df.to_csv(EXP / "work" / "metrics_stage2.csv", index=False)
    (EXP / "oof").mkdir(exist_ok=True)
    pd.concat(oof, ignore_index=True).to_parquet(EXP / "work" / "stage2_oof.parquet", index=False)
