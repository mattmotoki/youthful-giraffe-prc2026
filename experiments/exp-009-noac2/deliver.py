"""exp-009 delivery: the v2 noac head on the folds, against v1, byte-verified.

Only the noac rows differ between the two heads; the NM and no-NM heads are
frozen (`prc.tail_model._EXP002_SIGNALS`), so every difference below is the
noac head and nothing else.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent))
from threads import THREADS   # caps BLAS at import time; docs/COMPUTE.md compute allocation
import sys, time
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_model import TailModel

EXP = D.ROOT / "experiments" / "exp-009-noac2"
#: Published OOF files are versioned and immutable. v1 is the single-seed noac
#: head (sha 7dacc589...); v2 is the seed-averaged head that Opus round 1 B1
#: forced. Other lanes consume these, so a rerun writes a NEW version rather
#: than editing a published file in place (desk rule, 2026-09-05).
OOF_VERSION = 3
(EXP / "oof").mkdir(exist_ok=True)
NR, NN = 344841, 388
FOLDS = ("july", "july_loo", "december", "january")


def metrics(y, p, reg, fold, name):
    y = np.asarray(y, float); p = np.asarray(p, float); e2 = (y - p) ** 2
    tail = y > 3600; nonm = reg != "nm"; noac = reg == "noac"
    r = lambda m: float(np.sqrt(e2[m].mean())) if m.sum() else np.nan
    return dict(fold=fold, model=name, n=len(y), rmse=r(np.ones(len(y), bool)),
                rmse_le3600=r(~tail), rmse_gt3600=r(tail), rmse_nonm=r(nonm),
                rmse_noac=r(noac), n_noac=int(noac.sum()),
                noac_contrib_s=float(np.sqrt(NN * e2[noac].mean() / NR)) if noac.sum() else np.nan)


def main(out_name=f"exp-009-noac2-oof-v{OOF_VERSION}.parquet"):
    rows, oof = [], []
    for fold in FOLDS:
        t0 = time.time()
        train, valid, y_valid = D.fold_frames(fold)
        preds = {}
        for head in ("v4", "v3"):
            m = TailModel().fit(train, n_jobs=THREADS, noac_head=head)
            preds[head] = m.predict(valid)
        y = y_valid.reindex(preds["v4"].index).to_numpy(float)
        reg = preds["v4"]["regime"].to_numpy()
        for head in ("v4", "v3"):
            rows.append(metrics(y, preds[head]["y_pred"], reg, fold,
                                f"TailModel noac_head={head}"))
        oof.append(pd.DataFrame({"MVT_ID_mvt": preds["v4"]["MVT_ID_mvt"].to_numpy(),
                                 "fold": fold, "y_true": y,
                                 "y_pred": preds["v4"]["y_pred"].to_numpy()}))
        print(f'[{fold}] {time.time()-t0:.0f}s  v4={rows[-2]["rmse"]:.2f}  v3={rows[-1]["rmse"]:.2f}', flush=True)
    out = pd.concat(oof, ignore_index=True)
    path = EXP / "oof" / out_name
    if path.exists() and not out_name.startswith("_"):
        raise SystemExit(f"{path.name} already published; bump OOF_VERSION rather than "
                         f"editing a file other lanes read")
    out.to_parquet(path, index=False)
    m = pd.DataFrame(rows)
    if not out_name.startswith("_"):
        m.to_csv(EXP / "work" / "metrics_delivered.csv", index=False)
        print(m.round(2).to_string(index=False))
    sha = D.sha256(path)
    print(f"\n{len(out)} OOF rows -> {path}\nsha256 {sha}", flush=True)
    return sha


if __name__ == "__main__":
    if "--verify" in sys.argv:
        a = main(); b = main("_verify.parquet")
        (EXP / "oof" / "_verify.parquet").unlink()
        print(f"\nreproducible: {a == b}\n  run 1 {a}\n  run 2 {b}")
        if a != b:
            raise SystemExit("NOT REPRODUCIBLE")
    else:
        main()
