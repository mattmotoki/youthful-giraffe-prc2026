"""Question 3: predict BLOCK − SCHED and subtract, versus predicting taxi directly.

Same features, same rounds, same seed; only the target changes. The prior from
`RESULTS.md` §1 is that this loses: a boosted tree predicts a constant per leaf,
so moving the target from taxi (σ ≈ 690 s on the segment) to `BLOCK − SCHED`
(σ ≈ 4,500 s) makes the quantity being fitted an order of magnitude larger than
the quantity anyone cares about, and the subtraction afterwards does not undo
that. Measured rather than assumed.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
import features as F
from segment import rmse, FOLDS
from run_ablation import month, fit_predict, CONFIGS


def block_sched(month_name: str) -> np.ndarray:
    raw = D.load_month(month_name)
    dep = raw[raw["PHASE_mvt"] == "DEP"]
    return (dep["BLOCK_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()


def run(fold: str, out: dict):
    f = D.FOLDS[fold]
    tr = [month(m) for m in f.train_months]
    Xtr = pd.concat([t[0] for t in tr], ignore_index=True)
    ytr = np.concatenate([t[1] for t in tr])
    btr = np.concatenate([block_sched(m) for m in f.train_months])
    Xva, yva, segva = month(f.valid_month)
    cols = list(F.BASE) + [c for fam in CONFIGS["all"] for c in F.FAMILIES[fam]]
    res = out.setdefault(fold, {})

    t0 = time.time()
    p_direct, _ = fit_predict(Xtr[cols].copy(), ytr, Xva[cols].copy())
    res["target_taxi"] = rmse(yva[segva], p_direct[segva])

    ok = np.isfinite(btr)
    p_block, _ = fit_predict(Xtr[cols][ok].copy(), btr[ok], Xva[cols].copy())
    taxi_hat = Xva["mvt_sched"].to_numpy() - p_block
    res["target_block_sched"] = rmse(yva[segva], taxi_hat[segva])
    res["n"] = int(segva.sum())
    res["secs"] = round(time.time() - t0)
    print(f"[{fold}] taxi={res['target_taxi']:.1f} block_sched={res['target_block_sched']:.1f} "
          f"({res['secs']}s)", flush=True)
    (HERE / "q3.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    out = json.loads((HERE / "q3.json").read_text()) if (HERE / "q3.json").exists() else {}
    for fold in (sys.argv[1:] or ["july_loo", "december"]):
        run(fold, out)
