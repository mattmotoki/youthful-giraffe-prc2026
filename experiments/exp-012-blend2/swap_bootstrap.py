"""How small a blend delta can the rotation actually resolve?

exp-018 put a floor on the retention experiment: their s1->s3 seed gain is
0.74 s standalone, the rotation's grain looked like ~0.1 s read off printed
screen outputs, so the smallest resolvable retention fraction is ~0.1/0.74 = 14 %
and the 7 % the experiment was built to test sits inside the unresolvable band.

The premise is worth measuring rather than eyeballing. Those printed screens
compare DIFFERENT member sets on different folds; a retention measurement is a
PAIRED one-member swap, where every other member, every row and every weight is
held fixed. Its uncertainty is not the spread of a screen's output, it is the
sampling variability of a difference between two nearly identical predictors on
the same rows — which is exactly what a paired bootstrap is for and is far
tighter than either RMSE's own interval.

Weights are the rotated ones: fitted on the OTHER folds, at full size, and never
resampled. So they are not part of this fold's sampling noise, which is the
honest split — the question is whether the blend improvement is real, not whether
the weights would move.

Clusters are (airport, calendar day), the convention in exp-001's bootstrap.py:
errors are correlated within an airport-day, so resampling rows independently
would understate the interval.

Two intervals are reported, and the difference between them matters.

`fixed` holds the rotated weights constant: they come from the OTHER folds at
full size, so within this fold they are not random. That is the uncertainty of
the delta GIVEN the fit, and it is a lower bound on the total.

`refit` also resamples the fitting folds and refits the weights inside every
replicate, so weight noise enters too. It is the honest unconditional interval.
Much of that noise is common to base and variant — same fitting folds, and four
of the five members identical — so a paired delta should cancel most of it, but
"should" is why the option exists rather than an assertion in the write-up.

  python swap_bootstrap.py <base.json> <variant.json> [mode] [n_boot] [fixed|refit]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import blend as BL  # noqa: E402
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = tuple((__import__("os").environ.get("SWAP_FOLDS")
               or "july_loo,december,january").split(","))
BOARD = {"july_loo": 0.5571, "january": 0.4429}
RNG = np.random.default_rng(7)


def clusters_for(fold: str, ids: np.ndarray) -> np.ndarray:
    d = D.load_month(D.FOLDS[fold].valid_month,
                     columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    d = d[d.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
    day = d.MVT_TIME_UTC_mvt.dt.floor("D").astype("int64").reindex(ids).to_numpy()
    apt = d.ADEP_mvt.reindex(ids).astype("string").to_numpy()
    return pd.factorize(pd.Series(apt) + "|" + pd.Series(day).astype(str))[0]


def per_fold(base: dict, var: dict, mode: str):
    """Squared errors of both blends, and cluster codes, per fold."""
    db, names_b = BL.load(base, folds=FOLDS)
    dv, names_v = BL.load(var, folds=FOLDS)
    out = {}
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        fb = pd.concat([db[g] for g in src], ignore_index=True)
        fv = pd.concat([dv[g] for g in src], ignore_index=True)
        wb = B.fit_weights(fb[names_b].to_numpy("float64"), fb["y_true"].to_numpy("float64"), mode)
        wv = B.fit_weights(fv[names_v].to_numpy("float64"), fv["y_true"].to_numpy("float64"), mode)
        a, c = db[f], dv[f]
        assert np.array_equal(a["MVT_ID_mvt"].to_numpy(), c["MVT_ID_mvt"].to_numpy())
        y = a["y_true"].to_numpy("float64")
        pb = np.maximum(a[names_b].to_numpy("float64") @ wb, B.FLOOR)
        pv = np.maximum(c[names_v].to_numpy("float64") @ wv, B.FLOOR)
        cl = clusters_for(f, a["MVT_ID_mvt"].to_numpy())
        k = cl.max() + 1
        order = np.argsort(cl, kind="stable")
        out[f] = {
            "X_b": a[names_b].to_numpy("float64"), "X_v": c[names_v].to_numpy("float64"),
            "y": y, "cl_rows": np.split(order, np.cumsum(np.bincount(cl, minlength=k))[:-1]),
            "eb": np.bincount(cl, weights=(y - pb) ** 2, minlength=k),
            "ev": np.bincount(cl, weights=(y - pv) ** 2, minlength=k),
            "n": np.bincount(cl, minlength=k).astype("float64"),
            "rb": float(np.sqrt(((y - pb) ** 2).mean())),
            "rv": float(np.sqrt(((y - pv) ** 2).mean())),
        }
        print(f"  {f:>9}  base {out[f]['rb']:8.3f}  variant {out[f]['rv']:8.3f}  "
              f"delta {out[f]['rv']-out[f]['rb']:+7.3f}  ({k:,} airport-day clusters)")
    return out


def refit_boot(P, names_b, names_v, mode, n_boot):
    """Unconditional interval: resample the fitting folds and refit per replicate."""
    reps = {f: np.empty(n_boot) for f in FOLDS}
    bb, bv = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        rb, rv = {}, {}
        for f in FOLDS:
            src = [g for g in FOLDS if g != f]
            Xb, Xv, ys = [], [], []
            for g in src:
                d = P[g]
                sel = RNG.integers(0, len(d["cl_rows"]), len(d["cl_rows"]))
                idx = np.concatenate([d["cl_rows"][j] for j in sel])
                Xb.append(d["X_b"][idx]); Xv.append(d["X_v"][idx]); ys.append(d["y"][idx])
            yf = np.concatenate(ys)
            wb = B.fit_weights(np.vstack(Xb), yf, mode)
            wv = B.fit_weights(np.vstack(Xv), yf, mode)
            d = P[f]
            sel = RNG.integers(0, len(d["cl_rows"]), len(d["cl_rows"]))
            idx = np.concatenate([d["cl_rows"][j] for j in sel])
            yy = d["y"][idx]
            pb = np.maximum(d["X_b"][idx] @ wb, B.FLOOR)
            pv = np.maximum(d["X_v"][idx] @ wv, B.FLOOR)
            rb[f] = float(np.sqrt(((yy - pb) ** 2).mean()))
            rv[f] = float(np.sqrt(((yy - pv) ** 2).mean()))
            reps[f][i] = rv[f] - rb[f]
        bb[i] = np.sqrt(sum(BOARD[f] * rb[f] ** 2 for f in BOARD))
        bv[i] = np.sqrt(sum(BOARD[f] * rv[f] ** 2 for f in BOARD))
        if (i + 1) % 50 == 0:
            print(f"    refit replicate {i+1}/{n_boot}", flush=True)
    return reps, bv - bb


def main(base_cfg: str, var_cfg: str, mode="nnls", n_boot=2000, kind="fixed") -> int:
    import provenance
    print(provenance.stamp())
    n_boot = int(n_boot)
    base = json.loads((HERE / base_cfg).read_text())["members"]
    var = json.loads((HERE / var_cfg).read_text())["members"]
    ch = [k for k in base if base.get(k) != var.get(k)]
    if set(base) != set(var) or len(ch) != 1:
        print("  not a one-member swap"); return 2
    print(f"  swap {ch[0]}: {Path(base[ch[0]]).name} -> {Path(var[ch[0]]).name}")
    print(f"  mode {mode}, {n_boot} cluster-bootstrap replicates\n")
    P = per_fold(base, var, mode)
    if kind == "refit":
        nb = sorted(base); nv = sorted(var)
        reps, bd = refit_boot(P, nb, nv, mode, n_boot)
        print(f"\n  [refit] {'quantity':>10} {'2.5%':>8} {'97.5%':>8} {'P(worse)':>9}")
        for f in FOLDS:
            r = reps[f]
            print(f"  {'':8} {f:>10} {np.percentile(r,2.5):+8.3f} "
                  f"{np.percentile(r,97.5):+8.3f} {float((r>0).mean()):9.3f}")
        lo, hi = np.percentile(bd, 2.5), np.percentile(bd, 97.5)
        print(f"  {'':8} {'board':>10} {lo:+8.3f} {hi:+8.3f} {float((bd>0).mean()):9.3f}")
        half = (hi - lo) / 2
        print(f"\n  UNCONDITIONAL board delta half-width: {half:.4f} s")
        import os
        if os.environ.get("SWAP_RETENTION"):
            for gain, what in ((0.74, "exp-018 s1->s3 measured"), (1.5, "hypothetical 1->9 arm")):
                print(f"    against {gain:.2f} s standalone ({what}): "
                      f"resolvable retention floor ~{100*half/gain:.1f} %")
        print(f"  smallest blend effect this rotation separates from zero: ~{half:.3f} s")
        return 0
    reps = {f: np.empty(n_boot) for f in FOLDS}
    board_b = np.empty(n_boot); board_v = np.empty(n_boot)
    for i in range(n_boot):
        rb = {}; rv = {}
        for f in FOLDS:
            d = P[f]
            sel = RNG.integers(0, len(d["n"]), len(d["n"]))
            n = d["n"][sel].sum()
            rb[f] = np.sqrt(d["eb"][sel].sum() / n)
            rv[f] = np.sqrt(d["ev"][sel].sum() / n)
            reps[f][i] = rv[f] - rb[f]
        board_b[i] = np.sqrt(sum(BOARD[f] * rb[f] ** 2 for f in BOARD))
        board_v[i] = np.sqrt(sum(BOARD[f] * rv[f] ** 2 for f in BOARD))
    print(f"\n  {'quantity':>10} {'delta':>8} {'2.5%':>8} {'97.5%':>8} {'P(worse)':>9}")
    for f in FOLDS:
        r = reps[f]
        print(f"  {f:>10} {P[f]['rv']-P[f]['rb']:+8.3f} {np.percentile(r,2.5):+8.3f} "
              f"{np.percentile(r,97.5):+8.3f} {float((r>0).mean()):9.3f}")
    bd = board_v - board_b
    pt = (np.sqrt(sum(BOARD[f]*P[f]['rv']**2 for f in BOARD))
          - np.sqrt(sum(BOARD[f]*P[f]['rb']**2 for f in BOARD)))
    lo, hi = np.percentile(bd, 2.5), np.percentile(bd, 97.5)
    print(f"  {'board':>10} {pt:+8.3f} {lo:+8.3f} {hi:+8.3f} {float((bd>0).mean()):9.3f}")
    half = (hi - lo) / 2
    print(f"\n  board delta 95 % interval half-width: {half:.4f} s")
    print(f"  smallest blend effect this instrument separates from zero: ~{half:.3f} s")
    for gain, what in ((0.74, "exp-018 s1->s3 (measured)"), (1.5, "a hypothetical 1->9 seed arm")):
        print(f"    against a {gain:.2f} s standalone gain ({what}): "
              f"resolvable retention floor ~{100*half/gain:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
