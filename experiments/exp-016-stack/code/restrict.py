"""Where the stacker should be allowed to act, and the candidate's shape.

`why.py` found the gain is not uniform: `nm_bulk` and `nonm` gain on all three
folds, while `noac` LOSES 18.9 s on july_loo and `nm_late` swings +6.7 / -4.4 /
-0.6. The `noac` case has a reason that is not the score: the stacker is fitted on **152
noac rows** in the SMALLEST rotation's fit set -- 402 and 428 in the other two,
and 491 in the three-fold fit that ships (corrected, review round 2 N3; an
earlier version stated the 152 as a general property of "a rotation", which is
its best case) -- and exp-008 already measured that a weight vector fitted on
~150 heavy-tailed noac rows does not transfer to an unseen month. A non-linear function of five predictions fitted on 152 rows is a worse
version of the same problem.

So two restrictions are measured, both defined from inputs alone and both
motivated by sample size rather than by the fold scores:

  full         the stacker everywhere
  keep-noac    blend2 on `noac`, the stacker elsewhere
  bulk-nonm    blend2 on `noac` and `nm_late`, the stacker on `nm_bulk`+`nonm`

The selection this carries is disclosed in RESULTS.md rather than hidden: three
restrictions were looked at on the scored folds, and the argument for the one
adopted has to stand without them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
import stack as S  # noqa: E402
import why as W  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402

OUT = M.EXP / "analysis"
RESTRICT = {
    "full": (),
    "keep-noac": ("noac",),
    "bulk-nonm": ("noac", "nm_late"),
}


def bag(fit, ap, seeds=W.SEEDS):
    ps = [W.a5(fit, ap, s)[0] for s in seeds]
    return np.maximum(np.mean(ps, axis=0), B.FLOOR)


def main() -> None:
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    recs = []
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        cl = S.clusters(ap)
        ref = S.fit_blend2(fit, ap)
        st = bag(fit, ap)
        np.save(M.EXP / "oof" / f"a5bag-{f}.npy", np.stack([st, ref, S.flat4(ap)]))
        reg = ap["regime"].to_numpy()
        for name, off in RESTRICT.items():
            p = np.where(np.isin(reg, off), ref, st)
            d, lo, hi = S.boot_delta(y, p, ref, cl)
            recs.append(dict(fold=f, restriction=name, n=len(y), rmse=S.rmse(y, p),
                             vs_blend2=d, lo=lo, hi=hi,
                             n_offloaded=int(np.isin(reg, off).sum())))
        print(f"[{f}] " + "  ".join(
            f"{r['restriction']}={r['rmse']:.3f}" for r in recs[-len(RESTRICT):]) +
            f"   blend2={S.rmse(y, ref):.3f}", flush=True)
    T = pd.DataFrame(recs)
    be = {r: M.board_estimate({f: T[(T.fold == f) & (T.restriction == r)].rmse.iloc[0]
                               for f in M.SHARES}) for r in RESTRICT}
    T["board"] = T["restriction"].map(be)
    T.to_csv(OUT / "restrictions.csv", index=False)
    print("\n=== restriction, rotated ===")
    piv = T.pivot(index="restriction", columns="fold", values="rmse")[list(M.FOLDS)]
    piv["board"] = pd.Series(be)
    piv["vs blend2 (board)"] = piv["board"] - 318.017
    print(piv.sort_values("board").round(3).to_string(), flush=True)
    print("\n=== vs blend2, day-clustered 95 % ===")
    for r in T.sort_values(["restriction", "fold"]).itertuples():
        print(f"  {r.restriction:10s} {r.fold:9s} {r.vs_blend2:+7.3f} [{r.lo:+7.3f}, {r.hi:+7.3f}]"
              f"  ({r.n_offloaded} rows on blend2)")

    rk = M.ranking_frame()
    cov = {n: float(np.isfinite(rk[n]).mean()) for n in M.NAMES}
    print(f"\nranking frame {rk.shape}, coverage " +
          " ".join(f"{n}={c:.4f}" for n, c in cov.items()), flush=True)
    print("regimes " + str(dict(pd.Series(rk['regime']).value_counts())), flush=True)


if __name__ == "__main__":
    main()
