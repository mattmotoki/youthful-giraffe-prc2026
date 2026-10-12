"""Two gates before any arm runs:
  1. placeholder=False must leave the delivered model byte-unchanged (318.767301)
  2. placeholder=True must fit, and its p must be calibrated, not just ranked
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "exp-009-noac2"))
from threads import THREADS, gate
import numpy as np, pandas as pd
import prc.data as D, prc.tail_model as TM
from prc.tail_model import TailModel

gate()
tr, va, yv = D.fold_frames("july")
t0 = time.time()
m0 = TailModel().fit(tr, n_jobs=THREADS)
p0 = m0.predict(va); y = yv.reindex(p0.index).astype(float); k = y.notna()
r0 = D.rmse(y[k], p0.y_pred[k])
print(f"placeholder OFF: july {r0:.6f}  (delivered 318.767301)  "
      f"UNCHANGED={abs(r0-318.767301) < 5e-4}  {time.time()-t0:.0f}s", flush=True)

gate()
t0 = time.time()
m1 = TailModel().fit(tr, n_jobs=THREADS, placeholder=True)
p1 = m1.predict(va)
r1 = D.rmse(y[k], p1.y_pred[k])
print(f"placeholder ON : july {r1:.6f}  (delta {r1-r0:+.3f})  {time.time()-t0:.0f}s")
print(f"  fitted offset {m1.ph_offset:.2f} s, training class rate {m1.ph_rate*100:.2f}%")

# truth on the validation month, for calibration and the oracle
full = D.load_month("2025-07").query("PHASE_mvt == 'DEP'")
sig = TM.signals_by_month(va)
lab = TM.placeholder_label(full.set_index(full.MVT_ID_mvt).reindex(
    p1.MVT_ID_mvt.to_numpy()).reset_index(drop=True), pd.RangeIndex(len(p1)))
truth = lab.to_numpy()
pp = p1.p_placeholder_pop.to_numpy(float)
ok = np.isfinite(pp)
from sklearn.metrics import roc_auc_score
print(f"  n scored {ok.sum():,}  true rate {truth[ok].mean()*100:.2f}%  "
      f"AUC {roc_auc_score(truth[ok], pp[ok]):.4f}")
print("  calibration (decile of p -> observed rate):")
q = pd.qcut(pp[ok], 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"p": pp[ok], "t": truth[ok], "q": q}).groupby("q").agg(
    mean_p=("p", "mean"), obs=("t", "mean"), n=("t", "size"))
print(cal.round(4).to_string())

anchor = p1.ph_anchor.to_numpy(float)
orc = p1.y_pred.to_numpy(float).copy()
sel = ok & truth
orc[sel] = anchor[sel]
print(f"\n  ORACLE on this fit: july {D.rmse(y[k], orc[k]):.3f}  "
      f"(bound {r0 - D.rmse(y[k], orc[k]):+.3f} s)")
hard = p1.y_pred.to_numpy(float).copy()
hsel = ok & (pp > 0.5)
hard[hsel] = anchor[hsel]
print(f"  HARD switch p>0.5 : july {D.rmse(y[k], hard[k]):.3f}  "
      f"(delta vs control {D.rmse(y[k], hard[k]) - r0:+.3f} s, n switched {int(hsel.sum())})")
