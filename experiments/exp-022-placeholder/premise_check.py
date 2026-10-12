"""Reproduce every number in exp-022's SPEC that predates a fit.

Three claims, all measured rather than taken from the brief:
  1. the class's size and resolution (the brief says "to the second")
  2. the residual of using MVT-SCHED as the anchor on that class
  3. the oracle bound, against the delivered model's own july OOF
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "exp-009-noac2"))
from threads import THREADS  # caps BLAS; docs/COMPUTE.md compute allocation
import numpy as np, pandas as pd, prc.data as D

print("=== 1. the class, at two resolutions ===")
for mo in ("2025-01", "2025-07"):
    dep = D.load_month(mo).query("PHASE_mvt == 'DEP'")
    blk, sch = dep.BLOCK_TIME_UTC_mvt, dep.SCHED_TIME_UTC_mvt
    exact = (blk == sch)
    same = (blk.dt.floor("min") == sch.dt.floor("min"))
    print(f"{mo}: n={len(dep):,}  exact-second {exact.mean()*100:5.2f}%  "
          f"same-minute {same.mean()*100:5.2f}%")
    if mo == "2025-07":
        g = pd.DataFrame({"apt": dep.ADEP_mvt, "e": exact, "s": same}).groupby("apt").agg(
            n=("e", "size"), exact_pct=("e", "mean"), same_pct=("s", "mean"))
        g[["exact_pct", "same_pct"]] = (g[["exact_pct", "same_pct"]] * 100).round(2)
        print(g.sort_values("same_pct", ascending=False).to_string())
        jul = dep

blk, sch = jul.BLOCK_TIME_UTC_mvt, jul.SCHED_TIME_UTC_mvt
exact = (blk == sch); same = (blk.dt.floor("min") == sch.dt.floor("min"))
y = jul.TAXITIME_SEC_mvt.astype(float)
ms = (jul.MVT_TIME_UTC_mvt - sch).dt.total_seconds()

print("\n=== 2. is the branch an identity? ===")
print(f"exact-second rows: max |y - (MVT-SCHED)| = {(y[exact]-ms[exact]).abs().max():.6f} s")
r = ms[same] - y[same]
print(f"same-minute rows : delta = BLOCK-SCHED mean {(blk-sch).dt.total_seconds()[same].mean():.2f} s, "
      f"sd {(blk-sch).dt.total_seconds()[same].std():.2f}")
print(f"  RMS of predicting y = MVT-SCHED        : {np.sqrt((r**2).mean()):.2f} s")
print(f"  RMS after subtracting the fitted mean  : {np.sqrt(((r-r.mean())**2).mean()):.2f} s")

print("\n=== 3. the oracle bound, on the delivered model's july OOF ===")
oof = pd.read_parquet(D.ROOT / "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet")
o = oof[oof.fold == "july"]
f = pd.DataFrame({"MVT_ID_mvt": jul.MVT_ID_mvt.to_numpy(), "same": same.to_numpy(),
                  "ms": ms.to_numpy(), "noac": jul.AIRCRAFT_TYPE_mvt.isna().to_numpy()})
j = o.merge(f, on="MVT_ID_mvt", how="inner")
j = j[j.y_true.notna()]
base = D.rmse(j.y_true, j.y_pred)
sel = (j.same & ~j.noac).to_numpy()
sse = ((j.y_true - j.y_pred) ** 2).to_numpy()
print(f"july delivered model      : {base:.3f} on {len(j):,} rows")
print(f"same-minute non-noac rows : {sel.sum():,} ({sel.mean()*100:.2f}%), "
      f"model RMSE {D.rmse(j.y_true[sel], j.y_pred[sel]):.1f} s vs {D.rmse(j.y_true[~sel], j.y_pred[~sel]):.1f} elsewhere")
print(f"their share of july SSE   : {sse[sel].sum()/sse.sum()*100:.2f}%")
for off in (0.0, float(r.mean())):
    p = j.y_pred.to_numpy(float).copy()
    p[sel] = np.clip(j.ms.to_numpy(float)[sel] - off, 60, None)
    print(f"ORACLE, anchor MVT-SCHED-{off:5.1f}: july {D.rmse(j.y_true, p):.3f}  "
          f"(bound {base - D.rmse(j.y_true, p):+.3f} s)")
