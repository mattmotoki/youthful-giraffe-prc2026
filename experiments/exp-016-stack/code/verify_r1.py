"""Check the round-1 review's three load-bearing findings myself.

B2  the ablation ("the gates contribute nothing") rests on one seed. Refit BOTH
    arms at five seeds and compare means, not draws.
B3  the a1 diagnosis ("it cannot reach the level") is claimed to be wrong.
    Measure what a1 actually does: its base score, its prediction mean, a
    constant-mean baseline, and the two repairs RESULTS promised and never ran.
N3  "152 noac rows across the two fitting folds of a rotation" -- count them for
    all three rotations and for the shipped three-fold fit.

Also produces the ≤3600 / >3600 / no-NM decomposition COMMON.md requires and
this experiment's tables omitted (B4).
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


def arm(fit, ap, seed, cols, monotone=False, residual=True, rounds=S.LGB_ROUNDS, lr=None,
        init_flat=False):
    import lightgbm as lgb
    Xf, Xa = S.stack_features(fit, M.NAMES), S.stack_features(ap, M.NAMES)
    keep = [c for c in Xf.columns if cols(c)]
    p = dict(S.LGB_BASE, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    if lr:
        p["learning_rate"] = lr
    if monotone:
        p["monotone_constraints"] = [1 if c in M.NAMES else 0 for c in keep]
        p["monotone_constraints_method"] = "advanced"
    cat = [c for c in ("airport", "regime") if c in keep]
    y = fit["y_true"].to_numpy("float64")
    base_f = S.flat4(fit) if (residual or init_flat) else 0.0
    if init_flat:
        ds = lgb.Dataset(Xf[keep], label=y, categorical_feature=cat or None, init_score=base_f)
    else:
        ds = lgb.Dataset(Xf[keep], label=y - base_f, categorical_feature=cat or None)
    mdl = lgb.train(p, ds, num_boost_round=rounds)
    out = mdl.predict(Xa[keep])
    if residual or init_flat:
        out = out + S.flat4(ap)
    return np.maximum(out, B.FLOOR), mdl


ALLC = lambda c: True
NOGATE = lambda c: c in S2.MEMBER_COLS


def decomp(y, p, ap) -> dict:
    e = (np.asarray(p, "float64") - np.asarray(y, "float64")) ** 2
    bulk = y <= 3600.0
    nonm = ap["nm_missing"].to_numpy().astype(bool)
    noac = ap["noac"].to_numpy().astype(bool)
    r = lambda m: float(np.sqrt(e[m].mean())) if m.any() else float("nan")
    return dict(rmse=float(np.sqrt(e.mean())), le3600=r(bulk), gt3600=r(~bulk),
                no_nm=r(nonm), noac=r(noac))


def main() -> None:
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    # ---- N3: noac fitting rows per rotation -------------------------------
    print("=== N3: noac rows the stacker is fitted on ===")
    for f in M.FOLDS:
        n = sum(int(data[g]["noac"].sum()) for g in M.FOLDS if g != f)
        print(f"  scoring {f:9s}: {n} noac rows in the fit set")
    print(f"  shipped three-fold fit: {sum(int(data[g]['noac'].sum()) for g in M.FOLDS)} noac rows")

    # ---- B2: five seeds, both arms ----------------------------------------
    rows = []
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        for label, cols in (("a2 members+gates", ALLC), ("a5 members only", NOGATE)):
            for s in SEEDS:
                p, _ = arm(fit, ap, s, cols)
                rows.append(dict(fold=f, arm=label, seed=s, rmse=S.rmse(y, p)))
            print(f"[{f}] {label}: " +
                  " ".join(f"{r['rmse']:.3f}" for r in rows[-len(SEEDS):]), flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "r1_seed_ablation.csv", index=False)
    g = T.groupby(["arm", "fold"])["rmse"].agg(["mean", "min", "max"])
    g["spread"] = g["max"] - g["min"]
    print("\n=== B2: the ablation at five seeds ===")
    print(g.round(3).to_string())
    be = {a: M.board_estimate({f: g.loc[(a, f), "mean"] for f in M.SHARES}) for a in T.arm.unique()}
    print("board at the seed MEAN: " + "  ".join(f"{a} {v:.3f}" for a, v in be.items()))
    print(f"  gates cost {be['a2 members+gates'] - be['a5 members only']:+.3f} s at the mean; "
          f"december difference {g.loc[('a2 members+gates','december'),'mean'] - g.loc[('a5 members only','december'),'mean']:+.3f} s")

    # ---- B3: what the direct-target arm actually does ----------------------
    f = "january"
    ap, fit = data[f], pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
    y = ap["y_true"].to_numpy("float64")
    ymean = float(fit["y_true"].mean())
    print(f"\n=== B3: the direct-target arm on {f} ===")
    print(f"  fit-set label mean {ymean:.1f}; constant-mean predictor scores "
          f"{S.rmse(y, np.full(len(y), ymean)):.3f}")
    for lab, kw in (("a1 as run (lr .03, 300)", dict(residual=False, monotone=True)),
                    ("a1 repair (lr .10, 2000)", dict(residual=False, monotone=True,
                                                      rounds=2000, lr=0.10)),
                    ("a1 init_score=flat4 (lr .05, 300)", dict(residual=False, monotone=True,
                                                               init_flat=True))):
        t0 = time.time()
        p, mdl = arm(fit, ap, 7, ALLC, **kw)
        print(f"  {lab:34s} rmse {S.rmse(y, p):8.3f}   pred mean {p.mean():7.1f}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- B4: the decomposition COMMON requires -----------------------------
    print("\n=== B4: required decomposition, keep-noac against blend2 ===")
    recs = []
    for f in M.FOLDS:
        ap = data[f]
        y = ap["y_true"].to_numpy("float64")
        st, ref, flat = np.load(M.EXP / "oof" / f"a5bag-{f}.npy")
        reg = ap["regime"].to_numpy()
        keep = np.where(reg == "noac", ref, st)
        for lab, p in (("blend2", ref), ("flat4", flat), ("stack keep-noac", keep)):
            recs.append(dict(fold=f, variant=lab, **decomp(y, p, ap)))
    Dc = pd.DataFrame(recs)
    Dc.to_csv(OUT / "decomposition.csv", index=False)
    print(Dc.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
