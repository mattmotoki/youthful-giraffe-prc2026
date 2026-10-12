"""Is model2 model.py plus noise, or model.py plus a small systematic change?

Magnitude alone cannot tell those apart, and "model2 equals model.py plus
noise" is the kind of sentence that gets quoted later, so it gets the test that
can falsify it. A pure re-draw is EXCHANGEABLE, which has two consequences the
data can check (the second one is the blend lane's, 2026-09-06):

  location  the signed mean of (model2 - model.py) sits at zero within its own
            day-clustered standard error, and about half the rows move up.
  ordering  across folds the two RMSEs STRADDLE. One model winning every fold
            is what a systematic change looks like; the per-fold gaps being
            small is not enough on its own, because a small systematic gain and
            unbiased noise produce gaps of the same size.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/exchangeability.py \
      <new-tag> [fold ...]        # e.g. s1m-s1 july july_loo december january
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
N_BOOT = 2000


def _clusters(fold: str, ids: np.ndarray) -> np.ndarray:
    t = D.month_table(data.FOLDS[fold].valid_month)[["MVT_ID_mvt", "c_apt", "doy"]]
    m = pd.DataFrame({"MVT_ID_mvt": ids}).merge(t, on="MVT_ID_mvt", how="left")
    return (m.c_apt.fillna(-1).astype("int64") * 1000 + m.doy.fillna(-1).astype("int64")).to_numpy()


def one_fold(tag: str, fold: str, rng) -> dict:
    new = pd.read_parquet(EXP / "oof" / f"{tag}-{fold}.parquet")
    old = pd.read_parquet(f"experiments/exp-018-fresh/oof/{fold}.parquet")
    m = new.merge(old, on="MVT_ID_mvt", suffixes=("_new", "_old"))
    if not (len(m) == len(new) == len(old)):
        raise SystemExit(f"{fold}: {len(new)} new, {len(old)} old, {len(m)} joined")
    d = (m.y_pred_new - m.y_pred_old).to_numpy()
    y = m.y_true_new.to_numpy(dtype="float64")
    g = _clusters(fold, m.MVT_ID_mvt.to_numpy())
    order = np.argsort(g, kind="stable")
    gs = g[order]
    start = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    stop = np.r_[start[1:], len(gs)]
    ds = d[order]
    k = len(start)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([np.arange(start[u], stop[u]) for u in pick])
        means[b] = ds[idx].mean()
    return dict(fold=fold, n=len(m), mean=float(d.mean()), se=float(means.std()),
                lo=float(np.percentile(means, 2.5)), hi=float(np.percentile(means, 97.5)),
                up=float((d > 0).mean()), rms=float(np.sqrt((d ** 2).mean())),
                rmse_new=data.rmse(y, m.y_pred_new), rmse_old=data.rmse(y, m.y_pred_old))


def main() -> None:
    tag = sys.argv[1]
    folds = sys.argv[2:] or ["july"]
    rng = np.random.default_rng(11)
    rows = [one_fold(tag, f, rng) for f in folds]
    print(f"{tag} (model2) against exp-018's committed OOF (model.py), both one seed\n")
    print(f"{'fold':10s} {'n':>8s} {'signed mean':>12s} {'clustered SE':>13s} "
          f"{'z':>6s} {'% up':>7s} {'rms':>8s} {'RMSE new':>9s} {'RMSE old':>9s}")
    for r in rows:
        z = r["mean"] / r["se"] if r["se"] else float("nan")
        print(f"{r['fold']:10s} {r['n']:8d} {r['mean']:+12.4f} {r['se']:13.3f} "
              f"{z:+6.2f} {100*r['up']:6.2f}% {r['rms']:8.2f} {r['rmse_new']:9.2f} "
              f"{r['rmse_old']:9.2f}")
    wins = sum(r["rmse_new"] < r["rmse_old"] for r in rows)
    print(f"\nlocation: |z| < 2 on {sum(abs(r['mean'] / r['se']) < 2 for r in rows)}/{len(rows)} folds")
    print(f"ordering: model2 has the lower RMSE on {wins}/{len(rows)} folds "
          f"(a pure re-draw straddles; a clean sweep is systematic)")


if __name__ == "__main__":
    main()
