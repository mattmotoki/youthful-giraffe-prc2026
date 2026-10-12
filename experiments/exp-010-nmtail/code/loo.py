"""Leave-one-month-out over all twelve 2025 months, for both parameterisations.

Writes one OOF row per departure with `phi_hat` (weighted regression) and
`p_hat` (block-on-LOBT classifier), so the calibration of `p` and the RMSE of
each parameterisation can be read off the same rows.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-010-nmtail/code")
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import prc.data as d
import prc.nmtail as nt
from run_folds import frame, FEATURES
from features import CAT

OUT = d.ROOT / "experiments/exp-010-nmtail/oof"
OUT.mkdir(parents=True, exist_ok=True)
ROUNDS, LR, LEAVES = 400, 0.05, 63


def main():
    months = {m: frame(m) for m in d.MONTHS}
    rows = []
    for m in d.MONTHS:
        tr = pd.concat([months[k] for k in d.MONTHS if k != m], ignore_index=True)
        va = months[m]
        tr_nm = tr[tr.has_nm & tr.TAXITIME_SEC_mvt.notna()]
        y_tr = tr_nm["TAXITIME_SEC_mvt"].to_numpy(float)
        got = {}
        for mode in ("phi", "p"):
            t0 = time.time()
            mdl = nt.NMTailModel(mode=mode, features=FEATURES, categoricals=list(CAT),
                                 n_estimators=ROUNDS, learning_rate=LR, num_leaves=LEAVES)
            mdl.fit(tr_nm, y_tr)
            got[mode] = mdl.predict_phi(va)
            print(f"{m} {mode}: {round(time.time()-t0)}s", flush=True)
        r = va[["MVT_ID_mvt", "has_nm", "TAXITIME_SEC_mvt", "nm_taxi", "mvt_lobt", "gap", "apt"]].copy()
        r["month"] = m
        r["phi_hat"] = got["phi"]
        r["p_hat"] = got["p"]
        rows.append(r)
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(OUT / "loo_2025.parquet", index=False)
    print("wrote", OUT / "loo_2025.parquet", oof.shape)


if __name__ == "__main__":
    main()
