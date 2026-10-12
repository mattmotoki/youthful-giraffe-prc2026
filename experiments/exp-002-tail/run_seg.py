"""it9: one anchor per regime, because a tree cannot reproduce a continuous feature.

Diagnosis from run_all (July segment table):

* The single offset model (it2) cuts the NM-tail RMSE from 4058 s to 2293 s but
  pushes the no-NM rows from 995 s to 2933 s — worse than the naive airport
  median (658 s).
* Cause: the target `BLOCK - anchor`. For NM rows the anchor is `AOBT_3` and the
  offset is small (std 418 s), so a tree only has to learn a small correction.
  For no-NM rows the anchor is `SCHED`, so the offset is `mvt_sched - y`: to
  predict it the tree has to *reproduce a continuous input feature*, which a
  piecewise-constant model cannot do. Squared error in the offset equals squared
  error in y algebraically, but the two are not equally learnable.
* Repair: choose the anchor per regime so the offset is always small, and where
  no such anchor exists predict the taxi time directly.
    - NM rows            -> offset from AOBT_3
    - noac rows          -> offset from SCHED (the placeholder mode puts half of
                            them at offset 0, so the offset really is small)
    - other no-NM rows   -> taxi time directly, no `_flt` columns at all
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from run_all import (ART, EXP, CATS, BULK, TAILF, STAND, FULL, load, harmonise,
                     anchor_base, reg, metrics, NR, NN_RANK)
from noac_model import fit_predict as noac_fit_predict

NOFLT = [x for x in FULL if x not in ("mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt", "aobt_lobt",
                                      "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
                                      "gap_dep_prev_aobt")]


def run(fold_name, rows):
    f = D.FOLDS[fold_name]
    tr = load(f.train_months); va = load([f.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    ytr, yv = tr.y.to_numpy(float), va.y.to_numpy(float)
    nm_tr = tr[tr.nonm == 0].reset_index(drop=True)
    nn_tr = tr[(tr.nonm == 1) & (tr.noac == 0)].reset_index(drop=True)
    na_tr = tr[tr.noac == 1].reset_index(drop=True)
    nn_va = (va.nonm.to_numpy() == 1) & (va.noac.to_numpy() == 0)
    na_va = va.noac.to_numpy() == 1
    P = {}
    t0 = time.time()

    # --- NM head: offset from the NM actual off-block ----------------------
    m_nm = reg(nm_tr, FULL, nm_tr.mvt_aobt.to_numpy(float) - nm_tr.y.to_numpy(float))
    pred_nm = va.mvt_aobt.to_numpy(float) - m_nm.predict(va[FULL])
    print(f"  NM head {time.time()-t0:.0f}s", flush=True)

    # --- no-NM (non-noac) head: taxi time directly, no _flt columns --------
    m_nn_y = reg(nn_tr, NOFLT, nn_tr.y.to_numpy(float), num_leaves=31, n_estimators=500, min_child_samples=20)
    pred_nn_y = m_nn_y.predict(va[NOFLT])
    m_nn_o = reg(nn_tr, NOFLT, nn_tr.mvt_sched.to_numpy(float) - nn_tr.y.to_numpy(float),
                 num_leaves=31, n_estimators=500, min_child_samples=20)
    pred_nn_o = va.mvt_sched.to_numpy(float) - m_nn_o.predict(va[NOFLT])
    med = tr.groupby("apt", observed=True).y.median()
    pred_nn_med = va.apt.map(med).astype(float).to_numpy()
    print(f"  no-NM head {time.time()-t0:.0f}s", flush=True)

    # --- noac head ----------------------------------------------------------
    noac_out = noac_fit_predict(na_tr, va[na_va].reset_index(drop=True)) if na_va.sum() and len(na_tr) > 30 else None

    for nn_tag, pnn in (("y", pred_nn_y), ("offset", pred_nn_o), ("aptmedian", pred_nn_med)):
        p = pred_nm.copy(); p[nn_va] = pnn[nn_va]
        if noac_out is not None:
            for na_tag in ("mixture-expectation", "mixture-hardswitch", "lgb-direct-offset", "const-mean"):
                q = p.copy(); q[na_va] = noac_out[na_tag]
                P[f"it9-seg(nonm:{nn_tag}|noac:{na_tag})"] = np.clip(q, 60.0, None)
            q = p.copy(); q[na_va] = yv[na_va]
            P[f"it9-seg(nonm:{nn_tag}|noac:ORACLE)"] = np.clip(q, 60.0, None)
        else:
            P[f"it9-seg(nonm:{nn_tag})"] = np.clip(p, 60.0, None)
    best = "it9-seg(nonm:y|noac:mixture-expectation)"
    if best in P:
        P[best + "-noclip"] = pred_nm.copy()
        P[best + "-noclip"][nn_va] = pred_nn_y[nn_va]
        P[best + "-noclip"][na_va] = noac_out["mixture-expectation"]
    for k, p in P.items():
        rows.append(metrics(yv, p, va, k, fold_name))
    np.savez_compressed(EXP / "work" / f"seg_preds_{fold_name}.npz", **P)
    pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold_name, "y_true": yv,
                  "y_pred": P[best]}).to_parquet(EXP / "work" / f"seg_oof_{fold_name}.parquet", index=False)
    print(f"[{fold_name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    rows = []
    folds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["july", "december", "january"]
    for fd in folds:
        run(fd, rows)
        pd.DataFrame(rows).to_csv(EXP / "work" / "metrics_seg.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
