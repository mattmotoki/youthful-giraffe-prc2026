"""Null calibration for the bounded-amplitude estimator of section 2.

The estimator scores fold `f` at an amplitude fitted on rows `f` does not
contain, bounded to `[0, 1]`.  The bound is the thing to check: a direction with
no alignment gets `s = 0` and a gain of exactly zero, never a negative, so the
column cannot be symmetric around zero and the question is whether it is biased
UPWARD.

The algebra says no.  `s` is fitted on independent rows, so `s` and the scored
fold's `A_f` are independent; under a null with `E[A_f] = 0` the expected scored
gain is `E[2*s*A_f - s^2*B_f] = -E[s^2]*B_f <= 0`.  The bound therefore makes the
estimator CONSERVATIVE, not optimistic: it can only refuse to move.

This file measures that instead of asserting it.  Two placebos, each keeping the
candidate's own `d` distribution and destroying its pairing with `r`:

    global      `d` permuted uniformly inside the fold
    by-airport  `d` permuted inside each airport, so the airport composition of
                the update survives and only the row-level pairing dies

    python placebo.py [draws]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402

FOLDS = A.FOLDS
FINALISTS = ("stack-5", "k2-w300", "K2 (v9)")


def gains(r: dict, d: dict, blk3: dict, n: dict) -> tuple:
    st = {f: A.fold_stats(r[f], d[f], np.ones(len(d[f]), bool)) for f in FOLDS}
    loo = A.scale_loo({f: st[f]["A"] for f in FOLDS}, {f: st[f]["B"] for f in FOLDS}, n)
    cv = {f: A.scale_cv(r[f], d[f], blk3[f], np.ones(len(d[f]), bool)) for f in FOLDS}
    m10 = sum(A.SHARES[f] * st[f]["mse_v10"] for f in A.BOARD_FOLDS)
    g = lambda key: float(np.sqrt(m10) - np.sqrt(
        sum(A.SHARES[f] * (st[f]["mse_v10"] - key[f]["gain_mse"]) for f in A.BOARD_FOLDS)))
    return g(loo), g(cv)


def main() -> None:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cols = json.loads((EXP / "packet" / "columns.json").read_text())["d_columns"]
    inv = {v: k for k, v in cols.items()}
    P = {f: pd.read_parquet(EXP / "packet" / f"rows-{f}.parquet") for f in FOLDS}
    r = {f: P[f]["r"].to_numpy("float64") for f in FOLDS}
    blk3 = {f: ((P[f]["day"].to_numpy() - P[f]["day"].to_numpy().min()) // 3) for f in FOLDS}
    n = {f: len(P[f]) for f in FOLDS}
    apt_groups = {f: [np.flatnonzero(P[f]["airport"].to_numpy() == a)
                      for a in pd.unique(P[f]["airport"])] for f in FOLDS}

    out = []
    for name in FINALISTS:
        d0 = {f: P[f][inv[name]].to_numpy("float64") for f in FOLDS}
        real = gains(r, d0, blk3, n)
        for kind in ("global", "by-airport"):
            L, C = [], []
            for k in range(draws):
                rng = np.random.default_rng(1000 + k)
                d = {}
                for f in FOLDS:
                    if kind == "global":
                        d[f] = d0[f][rng.permutation(len(d0[f]))]
                    else:
                        v = d0[f].copy()
                        for idx in apt_groups[f]:
                            v[idx] = d0[f][idx][rng.permutation(len(idx))]
                        d[f] = v
                a, b = gains(r, d, blk3, n)
                L.append(a)
                C.append(b)
            out.append({"candidate": name, "placebo": kind, "draws": draws,
                        "real_loo": real[0], "real_cv": real[1],
                        "placebo_loo_mean": float(np.mean(L)),
                        "placebo_loo_p95": float(np.quantile(L, 0.95)),
                        "placebo_loo_max": float(np.max(L)),
                        "placebo_cv_mean": float(np.mean(C)),
                        "placebo_cv_p95": float(np.quantile(C, 0.95)),
                        "placebo_cv_max": float(np.max(C)),
                        "n_placebo_ge_real_loo": int(sum(x >= real[0] for x in L)),
                        "n_placebo_ge_real_cv": int(sum(x >= real[1] for x in C))})
            print(f"  {name:10s} {kind:11s} real LOO {real[0]:+.4f} / CV {real[1]:+.4f}   "
                  f"placebo LOO mean {np.mean(L):+.5f} p95 {np.quantile(L, 0.95):+.5f} "
                  f"max {np.max(L):+.5f}   CV mean {np.mean(C):+.5f} max {np.max(C):+.5f}   "
                  f"draws at or above the real value: {out[-1]['n_placebo_ge_real_loo']}/"
                  f"{draws} (LOO), {out[-1]['n_placebo_ge_real_cv']}/{draws} (CV)", flush=True)
    pd.DataFrame(out).to_csv(EXP / "analysis" / "q01-placebo.csv", index=False)


if __name__ == "__main__":
    main()
