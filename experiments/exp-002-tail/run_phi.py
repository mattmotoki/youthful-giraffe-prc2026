"""it14/it15: lirf's phi on the NM head, as features and as the offset anchor.

phi says BLOCK = AOBT_3 + phi*(LOBT - AOBT_3), so this experiment's NM-head
target `BLOCK - AOBT_3` *is* `phi * (LOBT - AOBT_3)`. Two ways to use it:

  it14  keep the AOBT anchor, hand the model `phi`, `phi_support`, `taxi_hat`
  it15  move the anchor: target `taxi_hat - y`, predict `taxi_hat - resid_hat`

it15 is the finding-8 move applied again — pick the anchor that leaves the
smallest offset. Out of sample, `rms(BLOCK - AOBT - phi*den)` is 400.9 s on July under the pinned
`rows` weighting (392.9 s under `gap2`) against 465.8 s for `rms(BLOCK - AOBT)`,
so the phi anchor looked like the better one by the same argument that made AOBT
beat SCHED for NM rows. It is not: it15 loses to it14 on all three folds, because
`taxi_hat` is a shrunk estimate rather than a measured time. See RESULTS
finding 17.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_model import TailModel, FEATURES, CATS, MAIN_PARAMS, MIN_TAXI, PHI_WEIGHT
from prc.tail_features import tail_signals
from prc.phi_features import PhiModel
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))

EXP = D.ROOT / "experiments" / "exp-002-tail"
NR, NN_RANK = 344841, 388
PHI_F = ["phi", "phi_support", "taxi_hat", "gap", "phi_airport", "taxi_hat_airport",
         "phi_airport_operator", "phi_support_airport_operator", "taxi_hat_airport_operator"]


def metrics(y, p, regime, fold, name):
    y = np.asarray(y, float); p = np.asarray(p, float); e2 = (y - p) ** 2
    tail = y > 3600; nonm = regime != "nm"; noac = regime == "noac"
    r = lambda m: float(np.sqrt(e2[m].mean())) if m.sum() else np.nan
    return dict(fold=fold, model=name, n=len(y), rmse=r(np.ones(len(y), bool)),
                rmse_le3600=r(~tail), rmse_gt3600=r(tail), rmse_nonm=r(nonm), rmse_noac=r(noac),
                noac_contrib_s=float(np.sqrt(NN_RANK * e2[noac].mean() / NR)) if noac.sum() else np.nan)


def phi_join(pm, frame, sig):
    t = pm.transform(frame).set_index("MVT_ID_mvt")
    t = t.reindex(sig["MVT_ID_mvt"].to_numpy() if "MVT_ID_mvt" in sig else frame.loc[sig.index, "MVT_ID_mvt"].to_numpy())
    t.index = sig.index
    return t


def run(fold_name, rows):
    import lightgbm as lgb
    f = D.FOLDS[fold_name]
    train, valid, y_valid = D.fold_frames(fold_name)
    t0 = time.time()
    # `TailModel.fit` now defaults to use_phi=True, so the no-phi baseline must
    # ask for it explicitly; otherwise this script compares it16 against itself.
    base = TailModel().fit(train, n_jobs=16, use_phi=False)
    pred_base = base.predict(valid)
    y = y_valid.reindex(pred_base.index).to_numpy(float)
    regime = pred_base["regime"].to_numpy()
    print(f"  base {time.time()-t0:.0f}s", flush=True)

    sig_tr = pd.concat([tail_signals(g) for _, g in train.groupby("month", sort=True)])
    sig_tr["MVT_ID_mvt"] = train["MVT_ID_mvt"].reindex(sig_tr.index)
    y_tr = train[D.TARGET].reindex(sig_tr.index).astype(float)
    keep = y_tr.notna(); sig_tr, y_tr = sig_tr[keep], y_tr[keep]
    sig_va = tail_signals(valid); sig_va["MVT_ID_mvt"] = valid["MVT_ID_mvt"].reindex(sig_va.index)

    pm = PhiModel(weight=PHI_WEIGHT).fit(train)
    ph_tr, ph_va = phi_join(pm, train, sig_tr), phi_join(pm, valid, sig_va)
    # leakage check: phi transform must not read the masked block time
    assert np.allclose(phi_join(pm, valid, sig_va)["taxi_hat"].to_numpy(dtype=float),
                       ph_va["taxi_hat"].to_numpy(dtype=float), equal_nan=True)
    cols = [c for c in PHI_F if c in ph_tr.columns]
    Xtr = pd.concat([sig_tr, ph_tr[cols]], axis=1)
    Xva = pd.concat([sig_va, ph_va[cols]], axis=1)
    cats = {c: pd.Index(sorted(set(Xtr[c].astype(str)))) for c in CATS}
    def enc(d, feats):
        X = d.loc[:, list(feats)].copy()
        for c in CATS:
            if c in feats: X[c] = pd.Categorical(d[c].astype(str), categories=cats[c])
        return X
    nm_tr = (sig_tr["nonm"].to_numpy() == 0)
    nm_va = regime == "nm"
    P = dict(MAIN_PARAMS, n_jobs=16)

    variants = {}
    # it14: AOBT anchor, phi as features
    F14 = list(FEATURES) + cols
    o = Xtr.loc[nm_tr, "mvt_aobt"].to_numpy(float) - y_tr[nm_tr].to_numpy(float)
    m = lgb.LGBMRegressor(**P).fit(enc(Xtr[nm_tr], F14), o, categorical_feature=list(CATS))
    variants["it14-phi-as-features"] = Xva["mvt_aobt"].to_numpy(float) - m.predict(enc(Xva, F14))
    # it15: phi anchor (taxi_hat), phi still available as features
    r = Xtr.loc[nm_tr, "taxi_hat"].to_numpy(float) - y_tr[nm_tr].to_numpy(float)
    ok = np.isfinite(r)
    m2 = lgb.LGBMRegressor(**P).fit(enc(Xtr[nm_tr][ok], F14), r[ok], categorical_feature=list(CATS))
    variants["it15-phi-as-anchor"] = Xva["taxi_hat"].to_numpy(float) - m2.predict(enc(Xva, F14))
    print(f"  heads {time.time()-t0:.0f}s", flush=True)

    rows.append(metrics(y, pred_base["y_pred"], regime, fold_name, "it13-baseline (use_phi=False)"))
    for k, v in variants.items():
        p = pred_base["y_pred"].to_numpy(float).copy()
        good = np.isfinite(v)
        p[nm_va & good] = np.clip(v[nm_va & good], MIN_TAXI, None)
        rows.append(metrics(y, p, regime, fold_name, k))
        np.save(EXP / "work" / f"phi_{k}_{fold_name}.npy", p)
    # raw phi taxi_hat as a predictor, no model at all
    p = pred_base["y_pred"].to_numpy(float).copy()
    th = Xva["taxi_hat"].to_numpy(float); good = np.isfinite(th)
    p[nm_va & good] = np.clip(th[nm_va & good], MIN_TAXI, None)
    rows.append(metrics(y, p, regime, fold_name, "it15b-raw taxi_hat, no model"))
    print(f"[{fold_name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    rows = []
    for fd in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["july", "december", "january"]):
        run(fd, rows)
        pd.DataFrame(rows).to_csv(EXP / "work" / "metrics_phi.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
