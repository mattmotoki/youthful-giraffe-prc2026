"""The day-clustered interval on the K2 slice's gain, and its seed spread.

Both are gates the record requires and neither is optional here: the population is
airport-and-day concentrated, so a row bootstrap understates the interval badly
(`exp-007 review-2` caught exactly that), and a single-seed difference on the
january fold has a measured sd of 0.53 s. This file produces the two numbers that
set K2's PASS bar in `SPEC.md`.

Clustering is on the CALENDAR DAY, not the airport-day: it is the more
conservative of the two, and the window mechanism is shared across airports on a
disrupted day.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from window_probe import load, FOLDS, SHARES  # noqa: E402
import k2_slice as K2  # noqa: E402


def arm(seed: int, scales, rounds=400, leaves=31, threads=2):
    data = {f: load(f) for f in FOLDS}
    X = {f: K2.features(data[f], scales) for f in FOLDS}
    out = {}
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        Xtr = pd.concat([X[g] for g in src], ignore_index=True)
        ytr = np.clip(pd.concat([data[g]["err"] for g in src], ignore_index=True), -1200, 1200)
        keep = pd.concat([data[g]["regime"] != "noac" for g in src], ignore_index=True).to_numpy()
        m = lgb.train({"objective": "regression", "learning_rate": 0.03, "num_leaves": leaves,
                       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "num_threads": threads, "seed": seed},
                      lgb.Dataset(Xtr[keep], ytr[keep]), num_boost_round=rounds)
        d = data[f]
        nn = (d["regime"] != "noac").to_numpy()
        adj = np.where(nn, m.predict(X[f]), 0.0)
        day = pd.to_datetime(d["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d").to_numpy()
        out[f] = {"e0": d["err"].to_numpy(), "e1": d["err"].to_numpy() - adj,
                  "day": day, "nn": nn, "n": len(d)}
    return out


def board(res, key, nonoac=False):
    t = 0.0
    for f in SHARES:
        r = res[f]
        m = r["nn"] if nonoac else np.ones(r["n"], dtype=bool)
        t += SHARES[f] * (r[key][m] ** 2).mean()
    return t ** 0.5


def main(scales: str = "300,1800", seeds: int = 3, boots: int = 400):
    sc = tuple(float(s) for s in scales.split(","))
    runs = [arm(11 + 7 * s, sc) for s in range(seeds)]
    for tag, nonoac in (("clean board", False), ("no-noac board", True)):
        g = np.array([board(r, "e0", nonoac) - board(r, "e1", nonoac) for r in runs])
        print(f"  {tag:14s} gain over {seeds} seeds: mean {g.mean():+.3f} s  sd {g.std(ddof=1):.3f}  "
              + " ".join(f"{v:+.3f}" for v in g))
    res = runs[0]
    days = {f: np.unique(res[f]["day"]) for f in SHARES}
    rng = np.random.default_rng(5)
    draws = {"all": [], "nonoac": []}
    for _ in range(boots):
        b0 = b1 = n0 = n1 = 0.0
        for f in SHARES:
            r = res[f]
            pick = rng.choice(days[f], size=len(days[f]), replace=True)
            idx = np.concatenate([np.flatnonzero(r["day"] == d) for d in pick])
            b0 += SHARES[f] * (r["e0"][idx] ** 2).mean()
            b1 += SHARES[f] * (r["e1"][idx] ** 2).mean()
            j = idx[r["nn"][idx]]
            n0 += SHARES[f] * (r["e0"][j] ** 2).mean()
            n1 += SHARES[f] * (r["e1"][j] ** 2).mean()
        draws["all"].append(b0 ** 0.5 - b1 ** 0.5)
        draws["nonoac"].append(n0 ** 0.5 - n1 ** 0.5)
    for k, v in draws.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"  {k:8s}: calendar-day-clustered 95 % interval on the gain "
              f"[{lo:+.3f}, {hi:+.3f}] s   P(gain <= 0) = {(v <= 0).mean():.3f}   "
              f"({boots} draws, seed-0 arm)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "300,1800", int(a[1]) if len(a) > 1 else 3,
         int(a[2]) if len(a) > 2 else 400)
