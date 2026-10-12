"""covmatch against the control, PAIRED -- the difference with its own interval.

Reading two overlapping 95 % intervals as a difference is not a test.  Both bags are
built on the same ten perturbations over the same head, so the difference is paired
and can be priced directly by the same day-clustered machinery."""
import sys
from pathlib import Path
import numpy as np
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
head = {f: v.astype("float64") for f,v in np.load(EXP/"work/d-bag10-255-free.npz").items()}
cov  = {f: v.astype("float64") for f,v in np.load(EXP/"work/d-covmatch10-255-free.npz").items()}
ctl  = {f: v.astype("float64") for f,v in np.load(EXP/"work/d-kishctl10-255-free.npz").items()}
for name, a, b in (("covmatch over the head", cov, head),
                   ("control  over the head", ctl, head),
                   ("covmatch over the CONTROL (the paired difference)", cov, ctl)):
    pr = R.price(pan, a, b, name)
    bb = pr["board_all_row"]
    print(f"{name:52s} {bb['gain_s']:+9.5f} [{bb['interval']['ci95'][0]:+8.5f},"
          f"{bb['interval']['ci95'][1]:+8.5f}]  every-fold-clear {pr['every_fold_interval_clear']}")
    for f in A.FOLDS:
        r = pr["by_fold"][f]
        print(f"    {f:10s} {r['mse_gain_s2']:+9.2f} [{r['ci95'][0]:+9.2f},{r['ci95'][1]:+9.2f}]")
