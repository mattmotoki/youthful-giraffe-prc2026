"""The repair the slices name: serve the chain member only where it helps.

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/regime_gate.py

§4b measured the arm's paired SSE change by regime and it is not one sign:
`nm_bulk` gains on all three folds, `nonm` LOSES on all three, `nm_late` is
mixed. So the obvious repair is a served regime gate -- the chain member on the
regimes it helps, v10 elsewhere -- which is the same shape of change exp-036
already priced per airport and needs no new fit: every vector is in
`packet/q09-{fold}.parquet`.

**The selection is held out from the rows it scores.** The queue's contract says
"Fit update scales and gates away from their scoring rows" and "Never fit and
score an update weight on the same rows". A regime subset chosen on all three
folds and then scored on all three would be exactly that, so the subset scoring
a fold is chosen on the paired SSE of the OTHER TWO folds only, which is
`align.scale_loo`'s rotation applied to a discrete choice instead of a scalar.
The all-three-fold choice is printed beside it as the in-sample number it is,
and it is not the row that counts.

Writes `analysis/regime-gate.csv` and `analysis/regime-gate.json`.
"""
from __future__ import annotations

import json
import sys
from itertools import chain as ichain, combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
sys.path.insert(0, str(ROOT))
import align as A  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = A.FOLDS
BASE = "v10 replay"
ARM = "chain d11-noac"
#: `noac` is pinned to v10 in this arm already, so it is not a choice; the gate
#: is over the three regimes the arm actually moves.
REGIMES = ("nm_bulk", "nm_late", "nonm")
SUBSETS = [frozenset(s) for r in range(len(REGIMES) + 1)
           for s in combinations(REGIMES, r)]


def load() -> dict:
    return {f: pd.read_parquet(EXP / "packet" / f"q09-{f}.parquet") for f in FOLDS}


def served(d: pd.DataFrame, keep: frozenset) -> np.ndarray:
    """The arm on `keep`'s regimes, v10 everywhere else."""
    on = d["regime"].isin(list(keep)).to_numpy()
    return np.where(on, d[ARM].to_numpy("float64"), d[BASE].to_numpy("float64"))


def sse_change(d: pd.DataFrame, keep: frozenset) -> float:
    """Paired SSE of the gated vector against v10; negative is better."""
    y = d["y_true"].to_numpy("float64")
    b = d[BASE].to_numpy("float64")
    p = served(d, keep)
    return float((((y - p) ** 2) - ((y - b) ** 2)).sum())


def price(data: dict, pick: dict) -> dict:
    """The contract's endpoints for a per-fold choice of regime subset."""
    st, parts = {}, {}
    for f in FOLDS:
        d = data[f]
        y = d["y_true"].to_numpy("float64")
        b = d[BASE].to_numpy("float64")
        p = served(d, pick[f])
        pr = A.per_row(y, b, p)
        st[f] = A.fold_stats(pr["r"], pr["d"], np.ones(len(y), bool))
        blk = A.blocks(d["day"].to_numpy("int64"), 1)
        parts[f] = (blk, pr["r"] ** 2, (pr["r"] - pr["d"]) ** 2)
    g = A.board_gain(st)
    ci = A.boot_board(parts, 2000)
    out = {"gain_board_allrow": g["gain_board"], "ci_lo": ci[0], "ci_hi": ci[1]}
    for f in FOLDS:
        fold_ci = A.boot_fold(*parts[f])
        out[f"gain_rmse_{f}"] = st[f]["gain_rmse"]
        out[f"ci_{f}_lo"], out[f"ci_{f}_hi"] = fold_ci
        out[f"clear_{f}"] = bool(fold_ci[0] > 0.0)
        out[f"A_{f}"] = st[f]["A"]
        out[f"pick_{f}"] = "+".join(sorted(pick[f])) or "(none)"
    out["every_fold_clear"] = bool(all(out[f"clear_{f}"] for f in FOLDS))
    return out


def main() -> None:
    data = load()
    S = {f: {s: sse_change(data[f], s) for s in SUBSETS} for f in FOLDS}

    rows = []
    for s in SUBSETS:
        rows.append({"subset": "+".join(sorted(s)) or "(none)",
                     **{f"sse_change_{f}": S[f][s] for f in FOLDS},
                     "sse_change_total": sum(S[f][s] for f in FOLDS)})
    T = pd.DataFrame(rows).sort_values("sse_change_total")
    print("=== paired SSE change by regime subset (negative = better than v10) ===")
    print(T.round(0).to_string(index=False))

    # the held-out choice: the subset scoring a fold is chosen on the other two
    pick = {f: min(SUBSETS, key=lambda s: sum(S[g][s] for g in FOLDS if g != f))
            for f in FOLDS}
    # and the in-sample choice, printed as what it is
    best_all = min(SUBSETS, key=lambda s: sum(S[f][s] for f in FOLDS))
    insample = {f: best_all for f in FOLDS}

    recs = {"held_out_choice": price(data, pick),
            "in_sample_choice": price(data, insample),
            "ungated (the arm itself)": price(data, {f: frozenset(REGIMES) for f in FOLDS})}
    print("\n=== the gate, priced ===")
    for name, r in recs.items():
        print(f"\n[{name}]")
        print(f"  picks       " + "  ".join(f"{f}={r[f'pick_{f}']}" for f in FOLDS))
        print(f"  all-row board {r['gain_board_allrow']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
        for f in FOLDS:
            print(f"  {f:9s} {r[f'gain_rmse_{f}']:+.4f} "
                  f"[{r[f'ci_{f}_lo']:+.4f}, {r[f'ci_{f}_hi']:+.4f}]"
                  f"  {'CLEAR' if r[f'clear_{f}'] else 'crosses zero'}")
        print(f"  every fold interval clear of zero: {r['every_fold_clear']}")

    T.to_csv(EXP / "analysis" / "regime-gate.csv", index=False)
    (EXP / "analysis" / "regime-gate.json").write_text(json.dumps(
        {"base": BASE, "arm": ARM, "regimes": list(REGIMES),
         "selection": "leave-one-fold-out on paired SSE; the subset scoring a fold "
                      "is chosen on the two folds it is not",
         **recs}, indent=2, default=str))


if __name__ == "__main__":
    main()
