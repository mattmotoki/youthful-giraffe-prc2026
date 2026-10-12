"""a2 or a5, with the `noac` offload applied to both, five seeds each.

The round-1 review (B2) showed the ablation that picked `a5` was a single-seed
draw: at the seed mean the gates cost 0.019 s of board estimate, which is inside
the seed spread, while on december they are WORTH 0.586 s. The two arms are a
tie on the two folds that make the board estimate and differ on the control
month, so the arm was never actually chosen -- it was drawn.

This measures both arms in the configuration that ships (five-seed bag, `noac`
on the flat average) so the choice is made on the same object, not on the
unrestricted arms.
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
import verify_r1 as V  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402

OUT = M.EXP / "analysis"


def main() -> None:
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    recs = []
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        cl = S.clusters(ap)
        ref = S.fit_blend2(fit, ap)
        reg = ap["regime"].to_numpy()
        for lab, cols in (("a2 members+gates", V.ALLC), ("a5 members only", V.NOGATE)):
            ps = [V.arm(fit, ap, s, cols)[0] for s in V.SEEDS]
            bag = np.maximum(np.mean(ps, axis=0), B.FLOOR)
            np.save(M.EXP / "oof" / f"bag-{lab.split()[0]}-{f}.npy", bag)
            for rname, off in (("full", ()), ("keep-noac", ("noac",))):
                p = np.where(np.isin(reg, off), ref, bag)
                d, lo, hi = S.boot_delta(y, p, ref, cl)
                recs.append(dict(fold=f, arm=lab, restriction=rname,
                                 vs_blend2=d, lo=lo, hi=hi, **V.decomp(y, p, ap)))
        print(f"[{f}] " + "  ".join(
            f"{r['arm'].split()[0]}/{r['restriction']}={r['rmse']:.3f}" for r in recs[-4:]) +
            f"   blend2={S.rmse(y, ref):.3f}", flush=True)
    T = pd.DataFrame(recs)
    T.to_csv(OUT / f"arm_choice{M.TAG}.csv", index=False)
    T["key"] = T["arm"].str.split().str[0] + "/" + T["restriction"]
    be = {k: M.board_estimate({f: T[(T.fold == f) & (T.key == k)].rmse.iloc[0] for f in M.SHARES})
          for k in T.key.unique()}
    piv = T.pivot(index="key", columns="fold", values="rmse")[list(M.FOLDS)]
    piv["board"] = pd.Series(be)
    piv["vs blend2"] = piv["board"] - 318.017
    print("\n=== five-seed bags, both arms, both restrictions ===")
    print(piv.sort_values("board").round(3).to_string(), flush=True)
    print("\n=== vs blend2, day-clustered 95 % ===")
    for r in T.sort_values(["key", "fold"]).itertuples():
        print(f"  {r.key:22s} {r.fold:9s} {r.vs_blend2:+7.3f} [{r.lo:+7.3f}, {r.hi:+7.3f}]")
    print("\n=== required decomposition ===")
    print(T[["fold", "key", "rmse", "le3600", "gt3600", "no_nm", "noac"]].round(2).to_string(index=False))


if __name__ == "__main__":
    M.config_from_argv()
    main()
