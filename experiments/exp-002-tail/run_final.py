"""it11: the segmented model plus the airport-day taxi-regime signals.

The tail clusters by airport-day: 61 % of January's non-noac tail rows fall on
the 5 % worst airport-days, and the day's own median `MVT - AOBT_3` — observable
for every ranking row — raises the tail rate 3.1x. Neither the noac sub-model
nor any per-flight feature reaches that, so it is added here as the last line.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from run_all import EXP, CATS, BULK, TAILF, STAND, load, harmonise, reg, metrics
from noac_model import fit_predict as noac_fit_predict

DAYF = ["apt_day_med_mvt_aobt", "apt_day_p90_mvt_aobt", "apt_day_long_frac",
        "apt_hour_med_mvt_aobt", "mvt_aobt_vs_day_med"]
FULL2 = BULK + TAILF + STAND + DAYF
NOFLT2 = [x for x in FULL2 if x not in ("mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt", "aobt_lobt",
                                        "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
                                        "gap_dep_prev_aobt", "mvt_aobt_vs_day_med")]


def run(fold_name, rows):
    f = D.FOLDS[fold_name]
    tr = load(f.train_months); va = load([f.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    yv = va.y.to_numpy(float)
    nm_tr = tr[tr.nonm == 0].reset_index(drop=True)
    all_tr = tr[tr.noac == 0].reset_index(drop=True)
    nn_va = (va.nonm.to_numpy() == 1) & (va.noac.to_numpy() == 0)
    na_va = va.noac.to_numpy() == 1
    o_nm = nm_tr.mvt_aobt.to_numpy(float) - nm_tr.y.to_numpy(float)
    t0 = time.time()
    P = {}
    for tag, F, NF in (("no-day", BULK + TAILF + STAND, [x for x in NOFLT2 if x in BULK + TAILF + STAND]),
                       ("with-day", FULL2, NOFLT2)):
        pnm = va.mvt_aobt.to_numpy(float) - reg(nm_tr, F, o_nm).predict(va[F])
        pnn = reg(all_tr, NF, all_tr.y.to_numpy(float)).predict(va[NF])
        base = pnm.copy(); base[nn_va] = pnn[nn_va]
        noac_out = noac_fit_predict(tr[tr.noac == 1].reset_index(drop=True), va[na_va].reset_index(drop=True))
        for na_tag in ("mixture-expectation", "lgb-direct-offset"):
            q = base.copy(); q[na_va] = noac_out[na_tag]
            P[f"it11-{tag}|noac:{na_tag}"] = np.clip(q, 60.0, None)
        q = base.copy(); q[na_va] = yv[na_va]
        P[f"it11-{tag}|noac:ORACLE"] = np.clip(q, 60.0, None)
        print(f"  {tag} {time.time()-t0:.0f}s", flush=True)
    for k, p in P.items():
        rows.append(metrics(yv, p, va, k, fold_name))
    np.savez_compressed(EXP / "work" / f"final_preds_{fold_name}.npz", **P)
    best = "it11-with-day|noac:mixture-expectation"
    pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold_name, "y_true": yv,
                  "y_pred": P[best]}).to_parquet(EXP / "work" / f"final_oof_{fold_name}.parquet", index=False)
    print(f"[{fold_name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    rows = []
    for fd in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["july", "december", "january"]):
        run(fd, rows)
        pd.DataFrame(rows).to_csv(EXP / "work" / "metrics_final.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
