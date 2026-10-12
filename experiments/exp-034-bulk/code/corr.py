"""What the bulk member is worth on its own, and how much of it is new.

Two questions, one instrument, because exp-032 measured that the first does not
predict the second: a member 2.0-3.5 s better on its own folds transmitted
+0.0157 s of board.

1. **Level.** The member's RMSE on the bulk slice (`has_nm`, `y <= 3600`, not
   `noac`) beside every member of K2's set on the IDENTICAL rows, so the queue's
   "4.1 s better than the best committed single member" is read against the
   members it must actually join rather than against an inventory row.
2. **Diversity.** The correlation of the member's residual with each of K2's
   seven, on the non-`noac` rows the stacker fits.  `L2-hub8725` was made 3.57 s
   WORSE as a member on purpose to buy 0.0177 of decorrelation
   (`exp-012/run_hub8725_folds.sh`); that exchange rate is the currency, and
   this table prices the bulk member in it.

The bulk member is null off the NM join, so every pair is taken on the rows
where BOTH residuals are finite -- the same rule `N1-nmtail` already needs.

    python corr.py   ->  analysis/member-corr.csv, analysis/member-level.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
sys.path.insert(0, str(ROOT))
import candidate as C  # noqa: E402  registers the member addresses and PARTIAL
import frames as F  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = C.FOLDS
BULK = C.BULK
SET = C.NAMESK2                       # K2's seven, the set the member joins


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    lvl, rows = [], []
    for fold in FOLDS:
        m = F.fold_frame(fold, C.NAMES5 + (C.K2, BULK))
        y = m["y_true"].to_numpy("float64")
        reg = m["regime"].to_numpy()
        nn = reg != "noac"
        # the member's own population: an NM join, a bulk label, not noac
        bulk = np.isfinite(m[BULK].to_numpy("float64")) & (y <= 3600.0) & nn
        for n in SET + (BULK,):
            v = m[n].to_numpy("float64")
            ok = bulk & np.isfinite(v)
            lvl.append({"fold": fold, "member": n, "n": int(ok.sum()),
                        "rmse_bulk": round(float(np.sqrt(np.mean((y[ok] - v[ok]) ** 2))), 4),
                        "coverage_of_bulk": round(float(ok.sum() / bulk.sum()), 4)})
        res = {n: y - m[n].to_numpy("float64") for n in SET + (BULK,)}
        for n in SET:
            rows.append({"fold": fold, "against": n,
                         "rho": round(_rho(res[BULK][nn], res[n][nn]), 4)})
        rest = [r["rho"] for r in rows if r["fold"] == fold]
        rows.append({"fold": fold, "against": "MEAN of the seven",
                     "rho": round(float(np.nanmean(rest)), 4)})

    L = pd.DataFrame(lvl)
    R = pd.DataFrame(rows)
    L.to_csv(EXP / "analysis" / "member-level.csv", index=False)
    R.to_csv(EXP / "analysis" / "member-corr.csv", index=False)

    print("=== member RMSE on the bulk slice (NM join, y<=3600, not noac), identical rows ===")
    p = L.pivot(index="member", columns="fold", values="rmse_bulk")[list(FOLDS)]
    best = p.drop(index=BULK).min()
    p.loc["-- best of K2's seven"] = best
    p.loc[f"-- {BULK} minus that"] = p.loc[BULK] - best
    print(p.round(3).to_string())
    print("\n=== residual correlation of the bulk member, non-noac rows ===")
    print(R.pivot(index="against", columns="fold", values="rho")[list(FOLDS)].to_string())
    print(f"\n-> analysis/member-level.csv, analysis/member-corr.csv")


if __name__ == "__main__":
    main()
