"""Day-clustered paired bootstrap between two OOF sets, on the board mix.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/bootstrap.py \
      <baseline-tag> <arm-tag> [n]

Clusters are airport-days: rows on one day at one airport share weather, a
runway configuration and a queue, so treating them as independent understates
the interval. That is the repo's own correction (exp-006 review round 1,
finding 4) and it is applied here from the start.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

_F = Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"
sys.path.insert(0, str(_F))
import dataset as D            # noqa: E402
from prc import data           # noqa: E402

EXP = Path(__file__).resolve().parents[1]
W = {"july_loo": 0.5571, "january": 0.4429}


def _clusters(fold: str, ids: np.ndarray) -> np.ndarray:
    t = D.month_table(data.FOLDS[fold].valid_month)[["MVT_ID_mvt", "c_apt", "doy"]]
    m = pd.DataFrame({"MVT_ID_mvt": ids}).merge(t, on="MVT_ID_mvt", how="left")
    return (m.c_apt.fillna(-1).astype("int64") * 1000 + m.doy.fillna(-1).astype("int64")).to_numpy()


def load(tag: str, folds) -> dict:
    out = {}
    for f in folds:
        p = EXP / "oof" / f"{tag}-{f}.parquet"
        if not p.exists():
            p = Path(f"experiments/exp-018-fresh/oof/{f}.parquet")
        out[f] = pd.read_parquet(p)
    return out


def board(d: dict, clean=True) -> float:
    tot = 0.0
    for f, w in W.items():
        m = d[f]
        r = (data.rmse_clean(m.y_true.to_numpy(), m.y_pred.to_numpy(),
                             m.MVT_ID_mvt.to_numpy(), f) if clean
             else data.rmse(m.y_true, m.y_pred))
        tot += w * r * r
    return float(np.sqrt(tot))


def _cluster_index(g: np.ndarray):
    """Rows grouped by cluster, as one sorted index plus per-cluster slices.

    The obvious `np.flatnonzero(g == u)` inside the replicate loop is
    O(clusters x rows) per draw -- 38 billion comparisons over 400 replicates
    here -- and this is the instrument every adoption decision rests on, so it
    is built once.
    """
    order = np.argsort(g, kind="stable")
    gs = g[order]
    start = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    stop = np.r_[start[1:], len(gs)]
    return order, start, stop


def paired(a: dict, b: dict, n=400, seed=7, clean=True):
    """b minus a on the board mix. Negative means b is better."""
    rng = np.random.default_rng(seed)
    pre = {}
    for f in W:
        ids_a = a[f].MVT_ID_mvt.to_numpy()
        if not np.array_equal(ids_a, b[f].MVT_ID_mvt.to_numpy()):
            raise SystemExit(f"{f}: the two OOF sets are not row-aligned; a paired "
                             f"bootstrap on them would compare different rows")
        g = _clusters(f, ids_a)
        order, start, stop = _cluster_index(g)
        y = a[f].y_true.to_numpy(dtype="float64")[order]
        ids = np.asarray(ids_a, dtype="float64")[order]
        # `data.rmse_clean` refuses a row set that holds none of the fold's
        # excluded ids, and a resample legitimately misses their airport-day,
        # so the exclusion is applied here as a precomputed mask instead.
        ex = data.excluded_ids(f)
        keep = ~np.isin(ids, list(ex)) if (ex and clean) else np.ones(len(ids), bool)
        if clean and ex and keep.all():
            raise SystemExit(f"{f}: none of the excluded ids {sorted(int(i) for i in ex)} "
                             f"is in this OOF; the clean score would not be clean")
        pre[f] = dict(start=start, stop=stop, y=y,
                      ok=keep & np.isfinite(y),
                      pa=a[f].y_pred.to_numpy(dtype="float64")[order],
                      pb=b[f].y_pred.to_numpy(dtype="float64")[order])
    diffs = []
    for _ in range(n):
        ta = tb = 0.0
        for f, w in W.items():
            P = pre[f]
            k = len(P["start"])
            pick = rng.integers(0, k, k)
            idx = np.concatenate([np.arange(P["start"][u], P["stop"][u]) for u in pick])
            ok = P["ok"][idx]
            idx = idx[ok]
            y = P["y"][idx]
            ta += w * np.mean((y - P["pa"][idx]) ** 2)
            tb += w * np.mean((y - P["pb"][idx]) ** 2)
        diffs.append(np.sqrt(tb) - np.sqrt(ta))
    d = np.array(diffs)
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_worse=float((d > 0).mean()))


if __name__ == "__main__":
    base_tag, arm_tag = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    folds = list(W)
    A, B = load(base_tag, folds), load(arm_tag, folds)
    print(f"{base_tag:28s} clean board {board(A):8.3f}")
    print(f"{arm_tag:28s} clean board {board(B):8.3f}")
    r = paired(A, B, n=n)
    print(f"delta {r['mean']:+.2f} s  [{r['lo']:+.2f}, {r['hi']:+.2f}]  P(worse) = {r['p_worse']:.3f}")
