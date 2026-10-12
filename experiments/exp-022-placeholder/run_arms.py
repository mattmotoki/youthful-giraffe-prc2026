"""exp-022 arms. Two fits per (fold, seed); P-oracle and P-hard are derived from
arm P's own predictions, so the bound and the cliff cost no extra fits."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "exp-009-noac2"))
from threads import THREADS, gate
import numpy as np, pandas as pd
import prc.data as D, prc.tail_model as TM
from prc.tail_model import TailModel

EXP = D.ROOT / "experiments/exp-022-placeholder"
VAL = {"july": "2025-07", "january": "2025-01", "july_loo": "2025-07"}
rows, preds = [], []
for fold, mo in VAL.items():
    tr, va, yv = D.fold_frames(fold)
    full = D.load_month(mo).query("PHASE_mvt == 'DEP'").set_index("MVT_ID_mvt")
    for seed in (7, 23, 51):
        got = {}
        for arm, ph in (("0 control", False), ("P mixture", True)):
            gate(); t0 = time.time()
            m = TailModel().fit(tr, n_jobs=THREADS, seed=seed, placeholder=ph)
            p = m.predict(va)
            y = yv.reindex(p.index).astype(float)
            k = y.notna().to_numpy()
            got[arm] = (p, y, k, m)
            print(f"{fold:9s} {arm:11s} seed={seed:3d} rmse={D.rmse(y[k], p.y_pred[k]):7.2f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        p, y, k, m = got["P mixture"]
        ids = p.MVT_ID_mvt.to_numpy()
        b = full["BLOCK_TIME_UTC_mvt"].reindex(ids)
        c = full["SCHED_TIME_UTC_mvt"].reindex(ids)
        truth = ((b.dt.floor("min") == c.dt.floor("min")) & b.notna() & c.notna()).to_numpy()
        pp = p.p_placeholder_pop.to_numpy(float)
        anchor = p.ph_anchor.to_numpy(float)
        ok = np.isfinite(pp) & np.isfinite(anchor)
        variants = {"0 control": got["0 control"][0].y_pred.to_numpy(float),
                    "P mixture": p.y_pred.to_numpy(float)}
        o = p.y_pred.to_numpy(float).copy(); o[ok & truth] = anchor[ok & truth]
        variants["P-oracle"] = o
        h = p.y_pred.to_numpy(float).copy(); h[ok & (pp > 0.5)] = anchor[ok & (pp > 0.5)]
        variants["P-hard"] = h
        for arm, pr in variants.items():
            rows.append(dict(fold=fold, arm=arm, seed=seed,
                             rmse=D.rmse(y[k], pr[k]),
                             rmse_clean=D.rmse_clean(y[k], pr[k], p.MVT_ID_mvt[k], fold)))
            preds.append(pd.DataFrame({"fold": fold, "arm": arm, "seed": seed,
                                       "MVT_ID_mvt": ids[k], "y_true": y.to_numpy()[k],
                                       "y_pred": pr[k]}))
        print(f"    -> oracle {rows[-2]['rmse']:.2f}  hard {rows[-1]['rmse']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(EXP / "work" / "arms.csv", index=False)
        pd.concat(preds, ignore_index=True).to_parquet(EXP / "work" / "preds.parquet", index=False)

df = pd.DataFrame(rows)
print("\n=== mean over seeds ===")
for metric in ("rmse", "rmse_clean"):
    print(f"\n-- {metric} --")
    print(df.groupby(["fold", "arm"])[metric].agg(["mean", "std"]).round(3).unstack(0).to_string())
