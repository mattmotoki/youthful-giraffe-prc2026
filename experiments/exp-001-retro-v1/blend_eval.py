"""Honest blend weights: choose them on two folds, score the third.

An in-fold optimal weight flatters the blend, so every number here comes from a
rotation — for each fold the per-regime weights are fitted on the union of the
other two and then applied unseen. A fixed 0.5 and a single global rotated
weight are reported alongside, because a blend that only works with per-regime
tuning is a different claim from one that works at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

MINE = sys.argv[1] if len(sys.argv) > 1 else "c6-noac"
FOLDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["july", "december", "january"]
FLOOR = 60.0
pd.set_option("display.width", 220)


def load(fold: str) -> pd.DataFrame | None:
    a = HERE / "oof" / f"{MINE}-{fold}.parquet"
    b = HERE / "oof" / f"tailmodel-{fold}.parquet"
    if not (a.exists() and b.exists()):
        return None
    m = pd.read_parquet(a).set_index("MVT_ID_mvt")
    t = pd.read_parquet(b).set_index("MVT_ID_mvt")
    j = m.join(t[["regime", "y_pred"]].rename(columns={"y_pred": "theirs"}), how="inner")
    j = j[j.y_true.notna() & j.theirs.notna()]
    j["fold"] = fold
    return j.rename(columns={"y_pred": "mine"})


def best_w(g: pd.DataFrame) -> float:
    """Least-squares weight on `theirs`, clipped to [0, 1]."""
    d = (g.theirs - g.mine).to_numpy()
    r = (g.y_true - g.mine).to_numpy()
    den = float((d * d).sum())
    return float(np.clip((r * d).sum() / den, 0.0, 1.0)) if den > 0 else 0.0


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.maximum(np.asarray(p), FLOOR)) ** 2)))


data = {f: d for f in FOLDS if (d := load(f)) is not None}
if not data:
    sys.exit("no paired predictions yet")

rows, weights = [], {}
for held in data:
    other = pd.concat([d for f, d in data.items() if f != held])
    g = data[held]
    w_reg = {r: best_w(sub) for r, sub in other.groupby("regime")}
    w_glob = best_w(other)
    weights[held] = {"per_regime": w_reg, "global": round(w_glob, 3)}
    wv = g.regime.map(w_reg).fillna(w_glob).to_numpy()
    rows.append(dict(
        fold=held, n=len(g),
        mine=rmse(g.y_true, g.mine),
        theirs=rmse(g.y_true, g.theirs),
        fixed_half=rmse(g.y_true, 0.5 * g.theirs + 0.5 * g.mine),
        rotated_global=rmse(g.y_true, w_glob * g.theirs + (1 - w_glob) * g.mine),
        rotated_per_regime=rmse(g.y_true, wv * g.theirs + (1 - wv) * g.mine),
        in_fold_per_regime=rmse(g.y_true, g.regime.map(
            {r: best_w(sub) for r, sub in g.groupby("regime")}).to_numpy() * g.theirs
            + (1 - g.regime.map({r: best_w(sub) for r, sub in g.groupby("regime")}).to_numpy()) * g.mine),
    ))

t = pd.DataFrame(rows)
print("== blend, weights fitted on the other two folds and applied unseen ==")
print(t.round(1).to_string(index=False))
print("\nweights used per held-out fold:")
print(json.dumps(weights, indent=2))

# weights for the shipped model: every fold available, which is what a final
# refit on all data is entitled to use.
allrows = pd.concat(data.values())
final = {r: round(best_w(sub), 3) for r, sub in allrows.groupby("regime")}
final["_global"] = round(best_w(allrows), 3)
print("\nweights for the candidate (fitted on all folds together):")
print(json.dumps(final))
(HERE / "blend_weights.json").write_text(json.dumps(final, indent=2))
