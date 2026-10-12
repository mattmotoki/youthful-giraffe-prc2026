"""Can ANY number of seeds pass the assignment's 1 s tail clause? No, and here is why.

Arm 1's member gate fails on `y > 3600`: K2's shipped fold reads 2594.06 and the
three-seed average 2599.67, against a 1 s tolerance. The repair the protocol asks
for is the one the diagnosis names -- if the loss is variance, more seeds remove
more of it -- and it can be settled with no GPU at all, because a k-seed average
of exchangeable fits obeys

    MSE(k) = MSE(inf) + V / k

with `V` the per-row variance the seed draw contributes. **That form is an
identity, not a fit**: averaging MSE over every subset of size k of exchangeable
draws gives exactly `B + (A - B)/k`, where `A` is the mean single-draw MSE and
`B` the mean pairwise cross term, so the k=2 agreement below is arithmetic and
not evidence. What IS assumed is exchangeability -- that seed `None`, seed 20 and
seed 33 are draws from one distribution -- and `None` is not exactly
exchangeable, because `--seeds 1` leaves LightGBM's `bagging_seed` and
`feature_fraction_seed` at their defaults while the other two derive them
(SPEC, "the residual asymmetry"). `MSE(inf) = B` is estimated from three draws
and carries that sampling error.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
K2 = ROOT / "experiments/exp-027-step/oof/k2-window-{fold}.parquet"
SEEDS = {20: EXP / "oof/k2s1-{fold}.parquet", 33: EXP / "oof/k2s2-{fold}.parquet"}
CUT = 3600.0


def mse(y, p, m):
    return float(np.mean((p[m] - y[m]) ** 2))


def main() -> None:
    out = {}
    for fold in ("july_loo", "december", "january"):
        k2 = pd.read_parquet(str(K2).format(fold=fold)).set_index("MVT_ID_mvt")
        cols = {"None": k2["y_pred"]}
        for s, tpl in SEEDS.items():
            p = Path(str(tpl).format(fold=fold))
            if p.exists():
                cols[str(s)] = pd.read_parquet(p).set_index("MVT_ID_mvt")["y_pred"]
        if len(cols) < 3:
            print(f"{fold}: only {len(cols)} seed files, skipped")
            continue
        y = k2["y_true"].to_numpy("float64")
        names = list(cols)
        P = {n: cols[n].reindex(k2.index).to_numpy("float64") for n in names}
        tail = y > CUT
        rec = {"n_tail": int(tail.sum()), "shipped_tail_rmse": round(float(np.sqrt(
            mse(y, P["None"], tail))), 4), "by_k": {}}
        for k in (1, 2, 3):
            vals = [mse(y, np.mean([P[n] for n in c], axis=0), tail)
                    for c in combinations(names, k)]
            rec["by_k"][k] = {"mse": float(np.mean(vals)),
                              "rmse": round(float(np.sqrt(np.mean(vals))), 4),
                              "n_subsets": len(vals)}
        m1, m2, m3 = (rec["by_k"][k]["mse"] for k in (1, 2, 3))
        # MSE(k) = Minf + V/k, fitted on k=1 and k=3, then k=2 tests it
        V = (m1 - m3) / (1 - 1 / 3)
        minf = m1 - V
        rec["V"] = round(V, 1)
        rec["mse_inf"] = round(minf, 1)
        rec["rmse_inf"] = round(float(np.sqrt(minf)), 4)
        rec["k2_predicted"] = round(float(np.sqrt(minf + V / 2)), 4)
        rec["k2_measured"] = rec["by_k"][2]["rmse"]
        rec["law_error_s"] = round(rec["k2_measured"] - rec["k2_predicted"], 4)
        rec["limit_vs_shipped"] = round(rec["shipped_tail_rmse"] - rec["rmse_inf"], 4)
        out[fold] = rec
        print(f"=== {fold}  tail rows {rec['n_tail']}")
        print(f"  shipped draw (seed None)          {rec['shipped_tail_rmse']:9.4f}")
        for k in (1, 2, 3):
            print(f"  average of {k} seeds ({rec['by_k'][k]['n_subsets']} subsets)  "
                  f"{rec['by_k'][k]['rmse']:9.4f}")
        print(f"  fitted V {V:.0f}, MSE(inf) {minf:.0f}")
        print(f"  k=2 from the identity {rec['k2_predicted']:.4f}, measured "
              f"{rec['k2_measured']:.4f} (equal by construction, not a test)")
        d = rec["limit_vs_shipped"]
        print(f"  UNLIMITED seeds land at {rec['rmse_inf']:.4f}: "
              f"{abs(d):.4f} s {'BETTER' if d > 0 else 'WORSE'} than the shipped draw"
              + ("   -> the 1 s clause cannot be met at any k"
                 if d < -1 else ""))
    q = EXP / "analysis" / "seed-limit.json"
    q.write_text(json.dumps(out, indent=1))
    print("wrote", q.relative_to(ROOT))


if __name__ == "__main__":
    main()
