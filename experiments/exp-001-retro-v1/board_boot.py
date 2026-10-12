"""Day-clustered bootstrap of the BOARD estimate difference between two arms.

The blend owner measures their three-fold rotation at a 0.1025 s half-width and
their two-fold at 0.1687 s — 65 % more uncertainty — and warns that anything read
off a july_loo+january pair under about 0.17 s is not resolved. That warning
applies to this experiment's own arm table, which has been quoting board
estimates to two decimals and treating 0.2 s gaps as orderings.

This resamples airport-days within each fold, recomputes each fold's RMSE on the
resample, and recombines them at the ranking's month shares, so the interval is
on the board estimate itself rather than on either fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D  # noqa: E402

N_BOOT = 600
FOLDS = {"july_loo": "2026-07", "january": "2026-01"}


def _fold_parts(name: str, fold: str):
    o = pd.read_parquet(HERE / "oof" / f"{name}-{fold}.parquet").set_index("MVT_ID_mvt")
    d = D.load_month(D.FOLDS[fold].valid_month,
                     columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    d = d[d.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt").reindex(o.index)
    day = d.MVT_TIME_UTC_mvt.dt.floor("D").astype("int64").to_numpy()
    cl = pd.factorize(d.ADEP_mvt.astype("string") + "|" + pd.Series(day, index=d.index).astype(str))[0]
    return o, cl


def compare(a: str, b: str, n_boot: int = N_BOOT, seed: int = 7):
    rng = np.random.default_rng(seed)
    w = {}
    r = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    r = r[r.PHASE_mvt == "DEP"]
    n = r.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").value_counts()
    w = (n / n.sum()).to_dict()

    sse_a = np.zeros(n_boot); sse_b = np.zeros(n_boot); obs_a = obs_b = 0.0
    for fold, month in FOLDS.items():
        oa, cl = _fold_parts(a, fold)
        ob, _ = _fold_parts(b, fold)
        ea = ((oa.y_true - oa.y_pred) ** 2).to_numpy()
        eb = ((ob.y_true - ob.y_pred) ** 2).to_numpy()
        obs_a += w[month] * ea.mean(); obs_b += w[month] * eb.mean()
        order = np.argsort(cl, kind="stable")
        ga, gb, g = ea[order], eb[order], cl[order]
        bounds = np.searchsorted(g, np.arange(g.max() + 2))
        sizes = np.diff(bounds)
        ca = np.add.reduceat(ga, bounds[:-1]) * (sizes > 0)
        cb = np.add.reduceat(gb, bounds[:-1]) * (sizes > 0)
        G = len(sizes)
        k = rng.integers(0, G, size=(n_boot, G))
        nn = sizes[k].sum(axis=1)
        sse_a += w[month] * ca[k].sum(axis=1) / nn
        sse_b += w[month] * cb[k].sum(axis=1) / nn
    d = np.sqrt(sse_a) - np.sqrt(sse_b)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(a=a, b=b, board_a=float(np.sqrt(obs_a)), board_b=float(np.sqrt(obs_b)),
                delta=float(np.sqrt(obs_a) - np.sqrt(obs_b)), lo=float(lo), hi=float(hi),
                half_width=float((hi - lo) / 2), resolved="yes" if lo * hi > 0 else "no")


if __name__ == "__main__":
    rows = [compare(sys.argv[1], b) for b in sys.argv[2:]]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
