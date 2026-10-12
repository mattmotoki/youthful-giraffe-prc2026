"""exp-017 (a)-(d) plus recency, on the plain-tree baseline.

Screen on july and january with one seed, because those are the folds whose
training months carry the rollover rows; confirm the winner on all four folds
with three seeds. Validation is untouched in every arm and every row is scored,
as always -- only the rows the model is FITTED on change.
"""
from __future__ import annotations
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exp-009-noac2"))
from threads import THREADS, gate
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_model import TailModel

EXP = D.ROOT / "experiments" / "exp-017-labels"


def arms(train):
    """(name, keep, weights) for each treatment; None means untouched."""
    a = D.label_artefacts(train)
    dep_idx = a.index
    out = [("baseline", None, None)]
    k, _ = D.clean_training_labels(train, rollover="drop", nonpositive="keep", duplicates="keep")
    out.append(("a: drop rollovers", k, None))
    k2, _ = D.clean_training_labels(train, rollover="keep", nonpositive="drop", duplicates="keep")
    out.append(("c: drop non-positive", k2, None))
    k3, _ = D.clean_training_labels(train, rollover="drop", nonpositive="drop", duplicates="drop")
    out.append(("a+c+dup: full recipe", k3, None))
    out.append(("recency hl=180d", None, D.recency_weights(train, 180.0)))
    out.append(("full recipe + recency", k3, D.recency_weights(train, 180.0)))
    return out


def run(folds, seeds, tag):
    rows = []
    for fold in folds:
        train, valid, y_valid = D.fold_frames(fold)
        for name, keep, w in arms(train):
            for sd in seeds:
                gate()
                t0 = time.time()
                m = TailModel().fit(train, n_jobs=THREADS, seed=sd, keep=keep, weights=w)
                p = m.predict(valid)
                y = y_valid.reindex(p.index).to_numpy(float)
                reg = p["regime"].to_numpy()
                e2 = (y - p["y_pred"].to_numpy(float)) ** 2
                rows.append(dict(fold=fold, arm=name, seed=sd,
                                 rmse=float(np.sqrt(e2.mean())),
                                 rmse_noac=float(np.sqrt(e2[reg == "noac"].mean())),
                                 n_dropped=0 if keep is None else int((~keep).sum()),
                                 secs=round(time.time() - t0)))
                print(f'{fold:9s} {name:24s} seed={sd} rmse={rows[-1]["rmse"]:7.2f} '
                      f'dropped={rows[-1]["n_dropped"]:4d} {rows[-1]["secs"]}s', flush=True)
                pd.DataFrame(rows).to_csv(EXP / "work" / f"metrics_{tag}.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if which == "screen":
        r = run(("july", "january"), (7,), "screen")
    else:
        r = run(("july", "july_loo", "december", "january"), (7, 23, 51), "confirm")
    print("\n" + r.groupby(["fold", "arm"]).rmse.mean().unstack().round(2).to_string())
