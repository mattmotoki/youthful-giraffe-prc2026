"""A1 against the control, SAVING per-row predictions, so january can get the
day-clustered paired bootstrap it needs.

exp-017's first pass reported seed spreads only. july_loo's -1.56 s against a
control seed sd of 0.09 needs no help; january is inside its spread on the clean
metric and a seed sd cannot settle it, because rows within a calendar day are
not independent.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "exp-009-noac2"))
from threads import THREADS, gate
import numpy as np, pandas as pd
import prc.data as D, prc.tail_model as TM
from prc.tail_model import TailModel

EXP = D.ROOT / "experiments/exp-017-labels"
UNREACHED = {r["id"] for rows in D.UNREACHED_ROLLOVER.values() for r in rows}
ARMS = (("0 control", False), ("A1 unreached", True))

rows = []
for fold in ("july", "january", "july_loo"):
    tr, va, yv = D.fold_frames(fold)
    dep = tr[tr["PHASE_mvt"] == "DEP"]
    for name, drop in ARMS:
        keep = ~dep["MVT_ID_mvt"].isin(UNREACHED) if drop else pd.Series(True, index=dep.index)
        for seed in (7, 23, 51):
            gate()
            t0 = time.time()
            m = TailModel().fit(tr, n_jobs=THREADS, seed=seed, keep=keep)
            p = m.predict(va)
            y = yv.reindex(p.index).astype(float)
            k = y.notna().to_numpy()
            rows.append(pd.DataFrame({
                "fold": fold, "arm": name, "seed": seed,
                "MVT_ID_mvt": p.MVT_ID_mvt.to_numpy()[k],
                "y_true": y.to_numpy()[k], "y_pred": p.y_pred.to_numpy()[k]}))
            print(f"{fold:9s} {name:13s} seed={seed:3d} "
                  f"rmse={D.rmse(y[k], p.y_pred[k]):7.2f} {time.time()-t0:.0f}s", flush=True)
            pd.concat(rows, ignore_index=True).to_parquet(EXP / "work" / "interval_preds.parquet",
                                                          index=False)
print("saved", EXP / "work" / "interval_preds.parquet")
