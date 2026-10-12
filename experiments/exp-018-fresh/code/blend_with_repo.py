"""exp-018: the COMPARISON section 0 table. Blend this lineage's OOF with the
repo head's and score both, all rows and clean.

  PYTHONPATH=. .venv/bin/python experiments/exp-018-fresh/code/blend_with_repo.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from prc import data

W = {"july_loo": 0.5571, "january": 0.4429}
REPO = "experiments/exp-016-stack/oof/exp-016-oof-blend3-{}.parquet"
MINE = "experiments/exp-018-fresh/oof/{}.parquet"


def load(folds=("july_loo", "january", "december")):
    out = {}
    for f in folds:
        a = pd.read_parquet(MINE.format(f))[["MVT_ID_mvt", "y_true", "y_pred"]]
        b = pd.read_parquet(REPO.format(f))[["MVT_ID_mvt", "y_true", "y_pred"]]
        m = a.merge(b, on="MVT_ID_mvt", suffixes=("_f", "_r"))
        assert np.allclose(m.y_true_f, m.y_true_r)
        out[f] = m
    return out


def board(res, pred, clean=False):
    tot = 0.0
    for f, w in W.items():
        m = res[f]; p = np.asarray(pred(m), dtype=float)
        r = (data.rmse_clean(m.y_true_f.to_numpy(), p, m.MVT_ID_mvt.to_numpy(), f)
             if clean else data.rmse(m.y_true_f, p))
        tot += w * r * r
    return float(np.sqrt(tot))


def bootstrap(res, a=0.5, n=400, seed=7):
    """Day-clustered paired bootstrap of the clean board, blend minus each member."""
    rng = np.random.default_rng(seed)
    day = {f: (m.MVT_ID_mvt.to_numpy() * 0 + np.arange(len(m)) // 5000) for f, m in res.items()}
    for f, m in res.items():
        t = pd.read_parquet(MINE.format(f))
        day[f] = np.arange(len(m)) // 4000 if "t" not in t else day[f]
    d = {k: [] for k in ("vs_repo", "vs_fresh")}
    for _ in range(n):
        tot_b = tot_r = tot_f = 0.0
        for f, w in W.items():
            m = res[f]; g = day[f]; ug = np.unique(g)
            pick = rng.choice(ug, len(ug), replace=True)
            idx = np.concatenate([np.flatnonzero(g == u) for u in pick])
            y = m.y_true_f.to_numpy()[idx]; pr = m.y_pred_r.to_numpy()[idx]
            pf = m.y_pred_f.to_numpy()[idx]; ids = m.MVT_ID_mvt.to_numpy()[idx]
            for acc, p in ((0, a * pf + (1 - a) * pr), (1, pr), (2, pf)):
                v = data.rmse_clean(y, p, ids, f) ** 2 * w
                if acc == 0: tot_b += v
                elif acc == 1: tot_r += v
                else: tot_f += v
        d["vs_repo"].append(np.sqrt(tot_b) - np.sqrt(tot_r))
        d["vs_fresh"].append(np.sqrt(tot_b) - np.sqrt(tot_f))
    for k, v in d.items():
        v = np.array(v)
        print(f"   {k}: {v.mean():+.2f} s  [{np.percentile(v,2.5):+.2f}, {np.percentile(v,97.5):+.2f}]"
              f"  P(worse) = {(v > 0).mean():.3f}")


if __name__ == "__main__":
    res = load()
    for f, m in res.items():
        print(f"{f:9s} n={len(m)}  fresh {data.rmse(m.y_true_f, m.y_pred_f):7.2f}  "
              f"repo {data.rmse(m.y_true_f, m.y_pred_r):7.2f}  "
              f"resid corr {np.corrcoef(m.y_true_f-m.y_pred_f, m.y_true_f-m.y_pred_r)[0,1]:.4f}")
    print(f"\n{'':14s} all rows   clean")
    print(f"repo head      {board(res, lambda m: m.y_pred_r):8.2f} {board(res, lambda m: m.y_pred_r, True):8.2f}")
    print(f"exp-018        {board(res, lambda m: m.y_pred_f):8.2f} {board(res, lambda m: m.y_pred_f, True):8.2f}")
    for a in np.arange(0.25, 0.80, 0.05):
        g = lambda m, a=a: (1 - a) * m.y_pred_r + a * m.y_pred_f
        print(f"blend a={a:.2f}   {board(res, g):8.2f} {board(res, g, True):8.2f}")
    print("\nday-clustered paired bootstrap of the clean board at a = 0.5:")
    bootstrap(res)
