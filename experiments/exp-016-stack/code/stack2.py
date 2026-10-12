"""exp-016 round 2: the variants round 1 mis-specified, plus the null-centred gate.

Round 1 (`logs/rotation.log`) found `a1-lgb-mono` at 605.63 on july_loo against a
296 baseline. That is not a result, it is an underfit: a direct-target GBM starts
from the label mean and 300 rounds at lr 0.03 cannot travel to a level of ~1,000 s,
so the arm was measuring its own budget. Two repairs, both keeping the desk's
monotone constraint on the member predictions:

CORRECTION (round-1 review B3). This docstring originally described two repairs
-- `fit_lgb2` at lr 0.10 / 2,000 rounds, and an `init_score = flat4` form -- and
the `VARIANTS` dict below runs NEITHER: it runs `_lgb`, the residual-target arm
at the round-1 budget, under the names `a3`/`a4`/`a5`. `fit_lgb2` is defined here
and called nowhere. Both repairs were finally measured in `code/verify_r1.py`,
and the diagnosis they were built on was wrong: the direct-target arm reaches the
label level immediately (prediction mean 1,001.6 against a fit-set mean of
1,011.5) and beats a constant-mean predictor by 155 s. It underfits the MAPPING,
not the level, and the 2,000-round repair is still 82.6 s worse than blend2 while
`init_score = flat4` works. `fit_lgb2` is kept because `verify_r1` reproduces
those numbers through the same parameterisation.

Round 1 also found `b-ridge-gated` at 299.81, worse than the 296.42 flat average
it nests. A gated ridge whose null is "all weights zero" spends its shrinkage
pulling toward a prediction of zero, which is the wrong null. Re-parameterised
(this one DID run):

  b2  ridge on the RESIDUAL from the flat average, design [p_j x g_k], so
      beta = 0 recovers the flat average EXACTLY and the penalty shrinks toward
      the blend rather than toward nothing. Three penalty strengths, all rotated.

`a2` and `c` are re-run unchanged as the bridge between the two rounds.
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
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402

OUT = M.EXP / "analysis"


def fit_lgb2(fit, ap, *, rounds, lr, monotone, init_flat, names=M.NAMES):
    import lightgbm as lgb

    Xf, Xa = S.stack_features(fit, names), S.stack_features(ap, names)
    yf = fit["y_true"].to_numpy("float64")
    p = dict(S.LGB_BASE, learning_rate=lr)
    if monotone:
        p["monotone_constraints"] = [1 if c in names else 0 for c in Xf.columns]
        p["monotone_constraints_method"] = "advanced"
    bf = S.flat4(fit) if init_flat else None
    ds = lgb.Dataset(Xf, label=yf, categorical_feature=["airport", "regime"],
                     init_score=bf if init_flat else None)
    mdl = lgb.train(p, ds, num_boost_round=rounds)
    out = mdl.predict(Xa)
    if init_flat:
        out = out + S.flat4(ap)
    return np.maximum(out, B.FLOOR), mdl, list(Xf.columns)


def fit_ridge_resid(fit, ap, names=M.NAMES, ridge: float = 1.0):
    """Variant b2: the gate is fitted on the residual from the flat average, so
    the zero solution IS the flat average and the penalty shrinks toward it."""
    def build(m):
        P, ok = S.design(m, names)
        P = np.where(ok, P, 0.0)
        G, gn = S.gate_matrix(m)
        return (np.concatenate([P[:, j:j + 1] * G for j in range(P.shape[1])], axis=1),
                [f"{n}:{g}" for n in names for g in gn])

    A, cn = build(fit)
    r = fit["y_true"].to_numpy("float64") - S.flat4(fit)
    Gm = A.T @ A
    Gm = Gm + ridge * (np.trace(Gm) / max(Gm.shape[0], 1) + 1.0) * np.eye(Gm.shape[0])
    w = np.linalg.solve(Gm, A.T @ r)
    Aa, _ = build(ap)
    return np.maximum(Aa @ w + S.flat4(ap), B.FLOOR), dict(zip(cn, map(float, w)))


#: Ablations that separate the two readings of round 1's gain. All four GBM arms
#: use the same budget and regularisation as the round-1 winner. The clean
#: single-variable contrast is a4 vs a5 (exact complements over the feature set);
#: a3 additionally adds the monotone constraint, so it is NOT part of that
#: contrast -- an earlier version of this comment said all four differed only in
#: what they see, which is false for a3 (review round 1 N7, round 2 N8).
#:   a2  member predictions + gates   (the round-1 winner, the reference)
#:   a4  gates ONLY                   -- if this keeps most of the gain, the
#:                                       "stacker" is a sixth base model
#:   a5  member predictions ONLY      -- pure re-mixing, no row features
#:   a3  a2 plus the desk's monotone constraint, started at the flat average
def _lgb(f, a, *, cols, monotone=False):
    """Round-1 settings, restricted to a column subset of the stack features."""
    import lightgbm as lgb
    Xf, Xa = S.stack_features(f, M.NAMES), S.stack_features(a, M.NAMES)
    keep = [c for c in Xf.columns if cols(c)]
    Xf, Xa = Xf[keep], Xa[keep]
    yf = f["y_true"].to_numpy("float64")
    p = dict(S.LGB_BASE)
    if monotone:
        p["monotone_constraints"] = [1 if c in M.NAMES else 0 for c in keep]
        p["monotone_constraints_method"] = "advanced"
    cat = [c for c in ("airport", "regime") if c in keep]
    ds = lgb.Dataset(Xf, label=yf - S.flat4(f), categorical_feature=cat or None)
    mdl = lgb.train(p, ds, num_boost_round=S.LGB_ROUNDS)
    return np.maximum(mdl.predict(Xa) + S.flat4(a), B.FLOOR)


MEMBER_COLS = set(M.NAMES) | {"mem_mean", "mem_std", "mem_range", "n_defined"}

VARIANTS = {
    "a3-lgb-mono":     lambda f, a: _lgb(f, a, cols=lambda c: True, monotone=True),
    "a4-gates-only":   lambda f, a: _lgb(f, a, cols=lambda c: c not in MEMBER_COLS),
    "a5-members-only": lambda f, a: _lgb(f, a, cols=lambda c: c in MEMBER_COLS),
    "b2-ridge-r0.1":   lambda f, a: fit_ridge_resid(f, a, ridge=0.1)[0],
    "b2-ridge-r1":     lambda f, a: fit_ridge_resid(f, a, ridge=1.0)[0],
    "b2-ridge-r10":    lambda f, a: fit_ridge_resid(f, a, ridge=10.0)[0],
}


def main() -> None:
    moved = M.verify()
    if moved:
        raise SystemExit(f"member bytes moved: {moved}")
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    rows = {}
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        cl = S.clusters(ap)
        got = {"flat4": S.flat4(ap), "blend2": S.fit_blend2(fit, ap),
               "a2-lgb-resid": S.fit_lgb(fit, ap, residual=True, monotone=False)[0]}
        for name, fn in VARIANTS.items():
            t0 = time.time()
            got[name] = fn(fit, ap)
            print(f"[{f}] {name}: {S.rmse(y, got[name]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        rows[f] = got
        np.save(M.EXP / "oof" / f"preds-{f}.npy",
                np.stack([got[k] for k in sorted(got)]))
        (M.EXP / "oof" / f"preds-{f}.names").write_text("\n".join(sorted(got)))
    recs = []
    for f in M.FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        cl = S.clusters(data[f])
        for v, p in rows[f].items():
            d2, lo2, hi2 = S.boot_delta(y, p, rows[f]["blend2"], cl)
            d4, lo4, hi4 = S.boot_delta(y, p, rows[f]["flat4"], cl)
            recs.append(dict(fold=f, variant=v, n=len(y), rmse=S.rmse(y, p),
                             vs_blend2=d2, b2_lo=lo2, b2_hi=hi2,
                             vs_flat4=d4, f4_lo=lo4, f4_hi=hi4))
    T = pd.DataFrame(recs)
    be = {v: M.board_estimate({f: T[(T.fold == f) & (T.variant == v)].rmse.iloc[0]
                               for f in M.SHARES}) for v in T.variant.unique()}
    T["board"] = T["variant"].map(be)
    T.to_csv(OUT / "rotation2.csv", index=False)
    piv = T.pivot(index="variant", columns="fold", values="rmse")[list(M.FOLDS)]
    piv["board"] = T.groupby("variant")["board"].first()
    print("\n=== rotated RMSE, round 2 ===")
    print(piv.sort_values("board").round(3).to_string())
    print("\n=== vs the shipped blend-2 rule (negative = better), day-clustered 95 % ===")
    for r in T.sort_values(["variant", "fold"]).itertuples():
        print(f"  {r.variant:18s} {r.fold:9s} {r.vs_blend2:+7.3f} [{r.b2_lo:+7.3f}, {r.b2_hi:+7.3f}]"
              f"   vs flat4 {r.vs_flat4:+7.3f} [{r.f4_lo:+7.3f}, {r.f4_hi:+7.3f}]")


if __name__ == "__main__":
    main()
