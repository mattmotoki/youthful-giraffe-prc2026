"""Where does the stacker's gain come from, and does it survive the seed?

CORRECTION (review round 2, N8). This docstring said "Round 2 settled the
mechanism ... the gain is re-mixing, not a row-feature correction", drawn from a
single seed-7 comparison. Refitting both arms at five seeds (`verify_r1.py`) the
gates cost 0.019 s of board estimate -- inside the seed spread -- and are WORTH
0.586 s on december. What survives is narrower: a gates-only model is worse than
the flat average, so this is predominantly re-mixing, but the row features are
not worthless. RESULTS.md carries the corrected version.

This file asks two follow-ups the protocol requires before anything is shipped:

1. **Where.** Per regime, per airport, and by member disagreement, so the gain
   has a mechanism rather than a total.
2. **Whether it is a seed.** The stacker is one LightGBM fit; the winter member
   it sits on top of has a 3.7 s seed spread (exp-012). Five seeds, rotated, and
   the shipped form is the seed average if the spread warrants it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
import stack as S  # noqa: E402
import stack2 as S2  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402

OUT = M.EXP / "analysis"
SEEDS = (7, 107, 207, 307, 407)


def a5(fit, ap, seed):
    import lightgbm as lgb
    Xf, Xa = S.stack_features(fit, M.NAMES), S.stack_features(ap, M.NAMES)
    keep = [c for c in Xf.columns if c in S2.MEMBER_COLS]
    p = dict(S.LGB_BASE, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    ds = lgb.Dataset(Xf[keep], label=fit["y_true"].to_numpy("float64") - S.flat4(fit))
    mdl = lgb.train(p, ds, num_boost_round=S.LGB_ROUNDS)
    return np.maximum(mdl.predict(Xa[keep]) + S.flat4(ap), B.FLOOR), mdl, keep


def main() -> None:
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    rows, per_seed, imp = [], {}, None
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        cl = S.clusters(ap)
        ref = S.fit_blend2(fit, ap)
        ps = []
        for s in SEEDS:
            t0 = time.time()
            p, mdl, keep = a5(fit, ap, s)
            ps.append(p)
            g = pd.Series(mdl.feature_importance("gain"), index=keep)
            imp = g if imp is None else imp.add(g, fill_value=0.0)
            rows.append(dict(fold=f, seed=s, rmse=S.rmse(y, p),
                             vs_blend2=S.rmse(y, p) - S.rmse(y, ref),
                             seconds=round(time.time() - t0)))
        bag = np.maximum(np.mean(ps, axis=0), B.FLOOR)
        d, lo, hi = S.boot_delta(y, bag, ref, cl)
        rows.append(dict(fold=f, seed="bag5", rmse=S.rmse(y, bag), vs_blend2=d,
                         lo=lo, hi=hi, seconds=0))
        per_seed[f] = dict(bag=bag, ref=ref, flat=S.flat4(ap))
        print(f"[{f}] seeds " + " ".join(f"{S.rmse(y, p):.3f}" for p in ps) +
              f"  bag {S.rmse(y, bag):.3f}  blend2 {S.rmse(y, ref):.3f}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "seed_stability.csv", index=False)
    be_single = M.board_estimate({f: T[(T.fold == f) & (T.seed == 7)].rmse.iloc[0] for f in M.SHARES})
    be_bag = M.board_estimate({f: T[(T.fold == f) & (T.seed == "bag5")].rmse.iloc[0] for f in M.SHARES})
    print(f"\nboard: single seed 7 {be_single:.3f}   five-seed bag {be_bag:.3f}", flush=True)
    print("\n=== seed spread per fold ===")
    g = T[T.seed != "bag5"].groupby("fold")["rmse"].agg(["min", "max", "mean", "std"])
    g["spread"] = g["max"] - g["min"]
    print(g.round(3).to_string(), flush=True)
    (OUT / "a5_importance.csv").write_text((imp / (len(SEEDS) * len(M.FOLDS)))
                                           .sort_values(ascending=False).to_csv(header=["gain"]))
    print("\n=== what the stacker splits on (mean gain over 15 fits) ===")
    print((imp / (len(SEEDS) * len(M.FOLDS))).sort_values(ascending=False).round(0).to_string())

    # ---- where the gain is -------------------------------------------------
    recs = []
    for f in M.FOLDS:
        ap, d = data[f], per_seed[f]
        y = ap["y_true"].to_numpy("float64")
        P, ok = S.design(ap, M.NAMES)
        disagree = np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
        band = pd.cut(disagree, [-1, 50, 150, 400, 1e9],
                      labels=["<50", "50-150", "150-400", ">400"])
        for key, lab in ((ap["regime"].to_numpy(), "regime"),
                         (ap["airport"].astype(str).to_numpy(), "airport"),
                         (band.astype(str), "member disagreement")):
            for v in pd.unique(key):
                m = key == v
                if m.sum() < 200:
                    continue
                recs.append(dict(fold=f, cut=lab, level=str(v), n=int(m.sum()),
                                 blend2=S.rmse(y[m], d["ref"][m]),
                                 stack=S.rmse(y[m], d["bag"][m]),
                                 gain=S.rmse(y[m], d["ref"][m]) - S.rmse(y[m], d["bag"][m]),
                                 sse_share=float(((d["ref"][m] - y[m]) ** 2).sum() /
                                                 ((d["ref"] - y) ** 2).sum())))
    W = pd.DataFrame(recs)
    W.to_csv(OUT / "where_the_gain_is.csv", index=False)
    print("\n=== where the gain is (positive = the stacker is better) ===")
    for cut in ("regime", "member disagreement", "airport"):
        sub = W[W.cut == cut].pivot(index="level", columns="fold", values="gain")
        n = W[W.cut == cut].groupby("level")["n"].mean().round(0)
        sub["mean_n"] = n
        print(f"\n-- by {cut} --")
        print(sub.round(3).to_string())


if __name__ == "__main__":
    main()
