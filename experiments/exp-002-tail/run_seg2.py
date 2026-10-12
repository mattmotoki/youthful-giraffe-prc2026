"""it10: train the no-NM head on *all* departures with the `_flt` columns dropped.

Diagnosis from it9 (July): a per-airport median beats the dedicated no-NM model
(658 s vs 895 s on the 3,599 no-NM non-noac rows). Cause: the dedicated model
sees only the ~14k no-NM rows of the training months, which is too few rows per
stand to learn the stand-to-runway taxi geometry, and their 1.6 % tail pulls the
L2 fit up. But whether Network Manager joined a flight is a property of the data
pipeline, not of the taxi process: the physics of stand -> runway -> take-off is
the same for every departure. So the head should be trained on every training
departure with the `_flt` columns removed, which is ~1M rows of the same physics.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from run_all import (EXP, FULL, load, harmonise, reg, metrics)
from run_seg import NOFLT
from noac_model import fit_predict as noac_fit_predict


def run(fold_name, rows):
    f = D.FOLDS[fold_name]
    tr = load(f.train_months); va = load([f.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    yv = va.y.to_numpy(float)
    nm_tr = tr[tr.nonm == 0].reset_index(drop=True)
    nn_va = (va.nonm.to_numpy() == 1) & (va.noac.to_numpy() == 0)
    na_va = va.noac.to_numpy() == 1
    t0 = time.time()

    m_nm = reg(nm_tr, FULL, nm_tr.mvt_aobt.to_numpy(float) - nm_tr.y.to_numpy(float))
    pred_nm = va.mvt_aobt.to_numpy(float) - m_nm.predict(va[FULL])
    # no-NM head trained on EVERY training departure, no _flt columns
    all_tr = tr[tr.noac == 0].reset_index(drop=True)
    m_all = reg(all_tr, NOFLT, all_tr.y.to_numpy(float))
    pred_nn_all = m_all.predict(va[NOFLT])
    med = tr.groupby("apt", observed=True).y.median()
    pred_nn_med = va.apt.map(med).astype(float).to_numpy()
    print(f"  heads {time.time()-t0:.0f}s", flush=True)

    na_tr = tr[tr.noac == 1].reset_index(drop=True)
    noac_out = noac_fit_predict(na_tr, va[na_va].reset_index(drop=True))
    P = {}
    for nn_tag, pnn in (("allrows-noflt", pred_nn_all), ("aptmedian", pred_nn_med)):
        base = pred_nm.copy(); base[nn_va] = pnn[nn_va]
        for na_tag in ("mixture-expectation", "lgb-direct-offset"):
            q = base.copy(); q[na_va] = noac_out[na_tag]
            P[f"it10-seg(nonm:{nn_tag}|noac:{na_tag})"] = np.clip(q, 60.0, None)
        q = base.copy(); q[na_va] = yv[na_va]
        P[f"it10-seg(nonm:{nn_tag}|noac:ORACLE)"] = np.clip(q, 60.0, None)
    # and the same head used for the noac rows too, as a floor comparison
    q = pred_nm.copy(); q[nn_va] = pred_nn_all[nn_va]; q[na_va] = pred_nn_all[na_va]
    P["it10-seg(nonm:allrows-noflt|noac:same-head)"] = np.clip(q, 60.0, None)
    for k, p in P.items():
        rows.append(metrics(yv, p, va, k, fold_name))
    np.savez_compressed(EXP / "work" / f"seg2_preds_{fold_name}.npz", **P)
    best = "it10-seg(nonm:allrows-noflt|noac:mixture-expectation)"
    pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold_name, "y_true": yv,
                  "y_pred": P[best]}).to_parquet(EXP / "work" / f"seg2_oof_{fold_name}.parquet", index=False)
    print(f"[{fold_name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    rows = []
    for fd in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["july", "december", "january"]):
        run(fd, rows)
        pd.DataFrame(rows).to_csv(EXP / "work" / "metrics_seg2.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
