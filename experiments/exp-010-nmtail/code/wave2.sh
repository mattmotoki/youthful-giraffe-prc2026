set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH=.:experiments/exp-010-nmtail/code:experiments/exp-003-anatomy/code
export OMP_NUM_THREADS=8
P=.venv/bin/python
$P - <<'PY'
# ablation: what the re-occupation screen is worth inside the head
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'experiments/exp-010-nmtail/code'); sys.path.insert(0,'experiments/exp-003-anatomy/code')
import prc.data as d, prc.nmtail as nt
import run_folds as R
from features import CAT
log = d.ROOT/"experiments/exp-010-nmtail/climb.log"
for fold in ("july_loo","december"):
    tr, va = R.fold_frames(fold)
    tr_nm = tr[tr.has_nm & tr.TAXITIME_SEC_mvt.notna()]
    y = va.TAXITIME_SEC_mvt.to_numpy(float); nm = va.has_nm.to_numpy(); tail = nm & (y>3600)
    for label, feats in [("with reocc", R.FEATURES),
                         ("without reocc", [c for c in R.FEATURES if c not in nt.SIGNALS
                                            and c not in ("reocc_x_screen","reocc_x_screen1h")])]:
        m = nt.NMTailModel(mode="phi", features=feats, categoricals=list(CAT),
                           n_estimators=900, learning_rate=0.04)
        m.fit(tr_nm, tr_nm.TAXITIME_SEC_mvt.to_numpy(float))
        p = m.predict(va)
        r = {"name":"REOCC-"+label.replace(" ","_"), "fold":fold, "mode":"phi",
             "nm_tail": d.rmse(y[tail],p[tail]), "nm_all": d.rmse(y[nm],p[nm]),
             "n_feat": len(feats)}
        print(f"[reocc] {fold} {label:14s} nm_tail={r['nm_tail']:.0f} nm_all={r['nm_all']:.1f} ({len(feats)} feat)", flush=True)
        open(log,"a").write(json.dumps(r)+"\n")
PY
$P experiments/exp-010-nmtail/code/loo.py
