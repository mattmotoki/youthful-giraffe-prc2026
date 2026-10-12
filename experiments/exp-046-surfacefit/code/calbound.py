"""In-fold UPPER BOUND on any mean-correction arm inside the writable class.
Fits the cell means on the very rows it scores, so it cannot be beaten by an
honest arm of the same form.  If this is small, the form is dead."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/path/to/competitions/prc-data-challenge-2026")
EXP = ROOT / "experiments/exp-046-surfacefit"
sys.path.insert(0, str(EXP/"code")); sys.path.insert(0, str(ROOT))
import arms as A, base as BA, chainfold as CF, rung511 as R, bagarm as BAG
rule = A.serving_rule()[0]
blk = {}
for f in A.FOLDS:
    m = A.FOLD_MONTH[f]
    b = CF.build_month(m, A.prefix_months_for(m), rule); R.assert_mode3(b,m,"shipped"); blk[m]=b
pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
mem = {s:{f:v.astype('float64') for f,v in np.load(EXP/f"work/d-bundle-placebo{s}-255-free.npz").items()} for s in range(1,11)}
bag = BAG.build_bag(mem, pan, list(range(1,11)))[0]
SH = BA.SHARES
mse0 = sum(SH[f]*float(((pan[f].y_true.to_numpy('float64')-pan[f].pbase.to_numpy('float64')-bag[f])**2).mean()) for f in SH)
print(f"board mse now {mse0:.4f}  rmse {np.sqrt(mse0):.4f} s")

def bound(name, keyfn):
    tot = 0.0
    per = {}
    for f in A.FOLDS:
        p = pan[f]; pb = p.pbase.to_numpy('float64'); d = bag[f]
        y = p.y_true.to_numpy('float64'); e = y - pb - d
        w = d != 0                                   # the rows this lane writes
        k = keyfn(p, pb + d)
        df = pd.DataFrame({"k": k[w], "e": e[w]})
        g = df.groupby("k")["e"]
        sse_drop = float((g.transform("mean")**2).sum())   # SSE removed by the cell means
        per[f] = (int(w.sum()), int(df.k.nunique()), sse_drop/len(p))
        if f in SH: tot += SH[f]*sse_drop/len(p)
    new = mse0 - tot
    print(f"{name:38s} board gain {np.sqrt(mse0)-np.sqrt(new):+.4f} s   "
          f"(cells: " + ", ".join(f"{f[:4]} {per[f][1]}" for f in A.FOLDS) + ")")

def deciles(v, n=10):
    q = np.quantile(v, np.linspace(0,1,n+1)); q[0]-=1; q[-1]+=1
    return np.digitize(v, q[1:-1])
bound("per pred-decile mean",            lambda p,pred: deciles(pred).astype(str))
bound("per airport mean",                lambda p,pred: p.airport.astype(str).to_numpy())
bound("per airport x pred-decile mean",  lambda p,pred: p.airport.astype(str).to_numpy()+"|"+deciles(pred).astype(str))
bound("per airport x pred-vigintile",    lambda p,pred: p.airport.astype(str).to_numpy()+"|"+deciles(pred,20).astype(str))
