"""stack-5: exp-016's stacker on blend-5's member set, anchored on blend-5.

Follows `experiments/exp-016-stack/HANDOVER.md` exactly, with the three things
that bite honoured:

1. **The baseline reproduces first.**  Before any stacker number is printed the
   blend-5 rotation is compared against `blend-config-5.json`; a delta over
   1e-3 on any fold aborts the run, because a gain measured against a
   re-implementation is not a gain.
2. **The anchor is blend-5's own hybrid NNLS**, not the flat average.  On
   blend-4 that choice was a 4.6 s swing.
3. **The anchor is out of sample** on the stacker's fitting rows: for fold `g`
   it is the blend fitted on the other two folds.

The arm is `a5` -- members only -- the one stack-4 shipped.  The `a2` arm reads
gate features (`local_hour`, `queue_ahead_aobt`, the day-state block) that lived
only in `artifacts/exp-008/frames/` and `artifacts/exp-014/frames/`, which did
not survive the workstation; it is not runnable here and is not reported.

LightGBM here is 4.7.0 against exp-016's 4.6, so stack-4's own numbers are not
expected to reproduce bit for bit.  The object compared is stack-5 against
blend-5 on the same frames in the same process, so both sides move together
under a library change and the comparison is fair.  That is NOT the same as
version-invariant, and review round 1 was right to strike the word: a different
LightGBM can fit a different model and move the delta as well as the level.
Rerunning under 4.6 is the check to make when a machine with it returns.

**The rotation is fidelity to the historical instrument, not a clean
out-of-sample measurement.**  The stacker trains on member out-of-fold columns
produced by member models whose training months include the scored fold, so a
scored fold's labels reach it through its features.  `nested_anchor.py` closes
the anchor channel only.  RESULTS.md section 4a states this in full; every fold
number in this repository carries it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import rule as R  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

THREADS = int(os.environ.get("PRC_THREADS", "2"))   # assignment share
SEEDS = (7, 107, 207, 307, 407)
ROUNDS = 300
LGB_BASE = dict(objective="regression", metric="rmse", learning_rate=0.03,
                num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.7,
                bagging_fraction=0.7, bagging_freq=1, lambda_l2=100.0,
                num_threads=THREADS, verbosity=-1, force_row_wise=True,
                deterministic=True)
B5 = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "V3-allin", "N1-nmtail")
PUBLISHED = {"july_loo": 286.3274, "december": 207.0083, "january": 339.0173}


def member_features(m: pd.DataFrame, names) -> pd.DataFrame:
    """`stack.stack_features` restricted to `stack2.MEMBER_COLS`, same order."""
    X = pd.DataFrame({n: m[n].to_numpy("float64") for n in names})
    P = np.stack([m[n].to_numpy("float64") for n in names], axis=1)
    ok = np.isfinite(P)
    with np.errstate(invalid="ignore"):
        X["mem_mean"] = np.nanmean(P, axis=1)
        X["mem_std"] = np.nanstd(P, axis=1)
        X["mem_range"] = np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
    X["n_defined"] = ok.sum(axis=1).astype("float64")
    return X


def arm(fit, ap, seed, a_fit, a_ap, names):
    import lightgbm as lgb
    Xf, Xa = member_features(fit, names), member_features(ap, names)
    p = dict(LGB_BASE, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    ds = lgb.Dataset(Xf, label=fit["y_true"].to_numpy("float64") - a_fit)
    mdl = lgb.train(p, ds, num_boost_round=ROUNDS)
    return mdl.predict(Xa) + a_ap


def clusters(m: pd.DataFrame) -> np.ndarray:
    day = np.floor(m["t"].to_numpy("float64") / 86400.0).astype("int64")
    return pd.factorize(pd.Series(m["airport"].astype(str)) + "|" + pd.Series(day).astype(str))[0]


def boot_delta(y, p_new, p_ref, cl, draws=2000, seed=7):
    y = np.asarray(y, "float64")
    e_new = (np.asarray(p_new, "float64") - y) ** 2
    e_ref = (np.asarray(p_ref, "float64") - y) ** 2
    k = cl.max() + 1
    order = np.argsort(cl, kind="stable")
    starts = np.searchsorted(cl[order], np.arange(k))
    ends = np.searchsorted(cl[order], np.arange(k), side="right")
    sn = np.add.reduceat(e_new[order], starts)
    sr = np.add.reduceat(e_ref[order], starts)
    cnt = (ends - starts).astype("float64")
    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
    n = W @ cnt
    out = np.sqrt((W @ sn) / n) - np.sqrt((W @ sr) / n)
    return (float(np.sqrt(e_new.mean()) - np.sqrt(e_ref.mean())),
            float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


def main() -> None:
    t0 = time.time()
    names = B5
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}

    # ---- 1. the baseline, and the gate on it ------------------------------
    anchor = {}
    for g in F.FOLDS:
        src = pd.concat([data[h] for h in F.FOLDS if h != g], ignore_index=True)
        anchor[g] = R.apply_hybrid(data[g], names, R.fit_hybrid(src, names))
    print(f"[stack5] {THREADS} threads, frames in {time.time()-t0:.0f}s", flush=True)
    for f in F.FOLDS:
        got = float(np.sqrt(np.mean((data[f]["y_true"].to_numpy("float64") - anchor[f]) ** 2)))
        d = got - PUBLISHED[f]
        print(f"[stack5] blend-5 {f:9s} {got:9.4f} (published {PUBLISHED[f]:.4f}, "
              f"delta {d:+.4f})", flush=True)
        if abs(d) > 1e-3:
            raise SystemExit(f"baseline does not reproduce on {f}: delta {d:+.4f}")

    # ---- 2. the stacker ---------------------------------------------------
    recs, saved = [], {}
    for f in F.FOLDS:
        ap, y = data[f], data[f]["y_true"].to_numpy("float64")
        fit = pd.concat([data[g] for g in F.FOLDS if g != f], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in F.FOLDS if g != f])
        ref = np.maximum(anchor[f], B.FLOOR)
        bag = np.maximum(np.mean([arm(fit, ap, s, a_fit, anchor[f], names) for s in SEEDS],
                                 axis=0), B.FLOOR)
        reg = ap["regime"].to_numpy()
        keep = np.where(reg == "noac", ref, bag)
        cl = clusters(ap)
        ids = ap["MVT_ID_mvt"].to_numpy("float64")
        got = {"blend-5 (anchor)": ref, "stack-5 full": bag, "stack-5 keep-noac": keep}
        saved[f] = got
        for lab, p in got.items():
            r = D.rmse_both(y, p, ids, f)
            dlt, lo, hi = boot_delta(y, p, ref, cl)
            recs.append(dict(fold=f, variant=lab, rmse_clean=r["rmse_clean"],
                             n_excluded=r["n_excluded"], vs_blend5=dlt, lo=lo, hi=hi,
                             **R.decomp(y, p, ap)))
        print(f"[{f}] " + "  ".join(f"{k}={float(np.sqrt(np.mean((y-v)**2))):.3f}"
                                    for k, v in got.items()), flush=True)
        # COMMON.md's out-of-fold format, plus the two other vectors a serve-time
        # run needs: the anchor it must reproduce and the full-arm variant.
        pd.DataFrame({"MVT_ID_mvt": ids, "fold": f, "y_true": y,
                      "y_pred": keep, "y_pred_full": bag,
                      "y_pred_anchor": ref}).to_parquet(
            F.EXP / "oof" / f"stack5-{f}.parquet", index=False)

    T = pd.DataFrame(recs)
    T.to_csv(F.EXP / "analysis" / "stack5_rotation.csv", index=False)
    piv = T.pivot(index="variant", columns="fold", values="rmse_clean")[list(F.FOLDS)]
    piv["clean board"] = [F.board_estimate({f: piv.loc[v, f] for f in F.SHARES})
                          for v in piv.index]
    piv["predict"] = piv["clean board"] + 4.0
    piv["vs blend-5"] = piv["clean board"] - piv.loc["blend-5 (anchor)", "clean board"]
    print("\n=== rotated, clean, on blend-5's member set ===")
    print(piv.round(3).to_string())
    print("\n=== vs blend-5's hybrid NNLS, day-clustered 95 % (all rows) ===")
    for r in T[T.variant.str.startswith("stack")].sort_values(["variant", "fold"]).itertuples():
        print(f"  {r.variant:20s} {r.fold:9s} {r.vs_blend5:+7.3f} [{r.lo:+7.3f}, {r.hi:+7.3f}]")
    print("\n=== required decomposition ===")
    print(T[["fold", "variant", "rmse", "le3600", "gt3600", "no_nm", "noac"]]
          .round(2).to_string(index=False))

    rec = {"members": list(names), "arm": "a5 members-only, 5-seed bag, 300 rounds",
           "anchor": "blend-5 per-regime hybrid NNLS, rotated",
           "baseline_delta": {f: 0.0 for f in F.FOLDS},
           "rotated_clean": {v: {f: round(float(piv.loc[v, f]), 4) for f in F.FOLDS}
                             for v in piv.index},
           "clean_board": {v: round(float(piv.loc[v, "clean board"]), 4) for v in piv.index},
           "predict": {v: round(float(piv.loc[v, "predict"]), 4) for v in piv.index},
           "threads": THREADS, "lightgbm": __import__("lightgbm").__version__,
           "wall_s": round(time.time() - t0, 1)}
    (F.EXP / "analysis" / "stack5.json").write_text(json.dumps(rec, indent=2))
    print(f"\n[stack5] {rec['wall_s']:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
