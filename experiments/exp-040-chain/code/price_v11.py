"""Sol round 1, blocking 3: price the predictor the candidate actually is.

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/price_v11.py

`code/regime_gate.py` priced `H'` against **v10**, and that is the number §4c and
§5 quoted.  The file `code/candidate.py` writes is `v11 + (H' - v10)`, so the
residual the chain update is evaluated against is **v11's**, not v10's.  v11 is
`max(v10 + 0.217*(stack-5 - v10), 60)` (`docs/SUBMISSIONS.md` row 11), which is a
different vector, so the alignment `A = mean(r.d)` and every interval change.

This file builds v11's fold analogue from the same two committed sources the
served file was built from, forms the candidate's fold analogue with the same
floor, and prices BOTH readings the record now has to carry:

  increment   the candidate against v11 -- what this file adds to the base it
              sits on, and the reading the advance clause has to be applied to
  composite   the candidate against v10 -- the whole move from the file the
              clause was originally written against

Writes `analysis/price-v11.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
sys.path.insert(0, str(ROOT))
import align as A                                           # noqa: E402
import score as Sc                                          # noqa: E402

FOLDS = A.FOLDS
HELD = ("nm_bulk", "nm_late")
#: `docs/SUBMISSIONS.md` row 11 and exp-038 §2c: the amplitude v11 was served at.
S_V11 = 0.217
FLOOR = 60.0


def vectors() -> dict:
    """Per fold: y, v10, v11, the candidate, and the block index."""
    ref = Sc.reference()          # gates stack-5 and v6 on exp-028's numbers
    out = {}
    for f in FOLDS:
        d = pd.read_parquet(EXP / "packet" / f"q09-{f}.parquet")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        s5 = pd.Series(ref[f]["y_pred_stack5"].to_numpy("float64"),
                       index=ref[f]["MVT_ID_mvt"].to_numpy("float64")).reindex(ids)
        if not np.isfinite(s5.to_numpy()).all():
            raise SystemExit(f"{f}: stack-5 is not defined on every packet row")
        v10 = d["v10 replay"].to_numpy("float64")
        v11 = np.maximum(v10 + S_V11 * (s5.to_numpy() - v10), FLOOR)
        on = d["regime"].isin(list(HELD)).to_numpy()
        u = np.where(on, d["chain d11-noac"].to_numpy("float64") - v10, 0.0)
        cand = np.maximum(v11 + u, FLOOR)
        out[f] = {"y": d["y_true"].to_numpy("float64"), "v10": v10, "v11": v11,
                  "cand": cand, "day": d["day"].to_numpy("int64"),
                  "floor_bound": int((cand != v11 + u).sum()),
                  "v11_floor_bound": int((v11 != v10 + S_V11 * (s5.to_numpy() - v10)).sum())}
    return out


def price(V: dict, base: str, arm: str) -> dict:
    st, parts = {}, {}
    for f in FOLDS:
        y, b, p = V[f]["y"], V[f][base], V[f][arm]
        pr = A.per_row(y, b, p)
        st[f] = A.fold_stats(pr["r"], pr["d"], np.ones(len(y), bool))
        parts[f] = (A.blocks(V[f]["day"], 1), pr["r"] ** 2, (pr["r"] - pr["d"]) ** 2)
    g = A.board_gain(st)
    lo, hi = A.boot_board(parts, 2000)
    out = {"base": base, "arm": arm, "gain_board": g["gain_board"],
           "ci": [lo, hi], "folds": {}}
    for f in FOLDS:
        ci = A.boot_fold(*parts[f])
        w3 = A.boot_fold(A.blocks(V[f]["day"], 3), parts[f][1], parts[f][2])
        out["folds"][f] = {"gain_rmse": st[f]["gain_rmse"], "ci": list(ci),
                           "ci_3day": list(w3), "A": st[f]["A"], "B": st[f]["B"],
                           "scale": st[f]["A"] / st[f]["B"], "clear": bool(ci[0] > 0.0),
                           "clear_3day": bool(w3[0] > 0.0)}
    out["every_fold_clear"] = bool(all(out["folds"][f]["clear"] for f in FOLDS))
    out["every_fold_clear_3day"] = bool(all(out["folds"][f]["clear_3day"] for f in FOLDS))
    return out


def main() -> None:
    V = vectors()
    rec = {"s_v11": S_V11, "floor": FLOOR,
           "floor_bound_rows": {f: V[f]["floor_bound"] for f in FOLDS},
           "increment_vs_v11": price(V, "v11", "cand"),
           "composite_vs_v10": price(V, "v10", "cand"),
           "v11_vs_v10": price(V, "v10", "v11")}
    (EXP / "analysis" / "price-v11.json").write_text(json.dumps(rec, indent=2))
    for k in ("increment_vs_v11", "composite_vs_v10", "v11_vs_v10"):
        r = rec[k]
        print(f"\n[{k}]  board {r['gain_board']:+.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        for f in FOLDS:
            x = r["folds"][f]
            print(f"  {f:9s} {x['gain_rmse']:+.4f} "
                  f"1d [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] "
                  f"3d [{x['ci_3day'][0]:+.4f}, {x['ci_3day'][1]:+.4f}] "
                  f"scale {x['scale']:.3f}  "
                  f"{'CLEAR' if x['clear'] else 'crosses zero'}")
        print(f"  every fold clear: 1-day {r['every_fold_clear']}  "
              f"3-day {r['every_fold_clear_3day']}")


if __name__ == "__main__":
    main()
