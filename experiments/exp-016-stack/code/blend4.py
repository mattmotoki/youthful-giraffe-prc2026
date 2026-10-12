"""stack-4: the stacker on blend-4's member set, against blend-4's fitted NNLS.

blend-4 changes the baseline in two ways that matter:

  * it adds a **fifth general member**, exp-018's fresh lineage, which NNLS
    weights at 0.62 in `nm_bulk` and 0.48 in `nm_late` -- it dominates;
  * its rule is **fitted NNLS everywhere**, not flat outside `nm_late`. That is
    the opposite of blend-2/3's rule, and consistent with the record rather than
    in tension with it: flat wins when members are near-equal (four members
    within 4 s), fitting wins when one is much stronger.

So the comparison is against a genuinely harder baseline than stack-3 faced, and
the stacker's own baseline changes too -- its residual is taken from the flat
average of the FIVE general members now, not four.

The rotation is unchanged: fit on two folds' OOF, score the third, never its own.
The baseline is rebuilt here rather than read from exp-012's config, and it is
checked against their published rotated numbers before any comparison is made.
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
import prc.blend as B
import prc.data as D  # noqa: E402

OUT = M.EXP / "analysis"
#: The four blend-4 general members plus the fresh lineage. `N1-nmtail` is the
#: sixth and enters only inside `nm_late`, which is why exp-012's members-4.json
#: omits it.
#: The general members: every member in the configured set except the NM-tail
#: specialist, which is defined only on the NM join and enters through the
#: nm_late split. This was a hardcoded five-tuple, which was right for blend-4
#: and silently wrong for any later set. On blend-5 it excluded V3-allin from
#: the BASELINE and from the ANCHOR while the stack arms still used it, so the
#: harness reported a 1.156 s "stacker gain" that was mostly V3-allin's own
#: contribution to a blend the baseline was not allowed to make. It now follows
#: the config, so the baseline is the rule the blend actually ships.
#: Resolved at CALL time, not definition time. `members.config_from_argv()`
#: rebinds `M.NAMES` after this module is imported, so a module-level tuple would
#: capture the DEFAULT member set - the exact trap members.py documents for its
#: own `names=NAMES` defaults. A five-member constant here is what let blend-5's
#: baseline run without V3-allin.
def general() -> tuple:
    return tuple(n for n in M.NAMES if n != "N1-nmtail")
#: exp-012's own rotated numbers for blend-4, from blend-config-4.json. The
#: baseline below must reproduce these or the comparison is against a
#: re-implementation rather than against blend-4.
PUBLISHED = {"july_loo": 289.3189, "december": 207.7722, "january": 340.8188}
#: ...but only for blend-4. On any other member set those constants make the
#: reproduction check unfailable-and-uninformative: it compares this harness's
#: blend against a DIFFERENT blend's published numbers, so a large delta is
#: expected and a small one would be luck. PUBLISHED_JSON points at the blend
#: config whose rule this baseline is supposed to reproduce.
if os.environ.get("PUBLISHED_JSON"):
    _pj = json.loads(Path(os.environ["PUBLISHED_JSON"]).read_text())
    PUBLISHED = {k: float(v) for k, v in _pj["rotated"].items()}
    print(f"[repro] baseline checked against {os.environ['PUBLISHED_JSON']}: "
          + "  ".join(f"{k} {v:.4f}" for k, v in PUBLISHED.items()), flush=True)


def flat_general(m: pd.DataFrame) -> np.ndarray:
    P = np.stack([m[n].to_numpy("float64") for n in general()], axis=1)
    return np.maximum(P.mean(axis=1), B.FLOOR)


def fit_blend4(fit: pd.DataFrame, ap: pd.DataFrame) -> np.ndarray:
    """blend-4's rule: per-regime NNLS, six members inside `nm_late`, five outside.

    `N1-nmtail` is undefined off the NM join, so it is fitted only where it is
    defined -- which is also why exp-012's shipped vector carries it at exactly
    0.000 in the three other regimes.
    """
    names6 = list(general()) + ["N1-nmtail"]
    Pf = np.stack([fit[n].to_numpy("float64") for n in names6], axis=1)
    Pa = np.stack([ap[n].to_numpy("float64") for n in names6], axis=1)
    okf = np.isfinite(Pf[:, -1])
    oka = np.isfinite(Pa[:, -1])
    selF = (fit["regime"].to_numpy() == "nm_late") & okf
    selA = (ap["regime"].to_numpy() == "nm_late") & oka
    w6 = B.fit_weights(Pf[selF], fit["y_true"].to_numpy("float64")[selF], "nnls")
    w5 = B.fit_weights(Pf[~selF][:, :-1], fit["y_true"].to_numpy("float64")[~selF], "nnls")
    out = np.empty(len(ap))
    out[selA] = Pa[selA] @ w6
    out[~selA] = Pa[~selA][:, :-1] @ w5
    return np.maximum(out, B.FLOOR), w6, w5


def arm_anchored(fit, ap, seed, cols, a_fit, a_ap):
    """The same stacker, fitting the residual from a supplied anchor."""
    import lightgbm as lgb
    Xf, Xa = S.stack_features(fit, M.NAMES), S.stack_features(ap, M.NAMES)
    keep = [c for c in Xf.columns if cols(c)]
    p = dict(S.LGB_BASE, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    cat = [c for c in ("airport", "regime") if c in keep]
    ds = lgb.Dataset(Xf[keep], label=fit["y_true"].to_numpy("float64") - a_fit,
                     categorical_feature=cat or None)
    mdl = lgb.train(p, ds, num_boost_round=S.LGB_ROUNDS)
    return mdl.predict(Xa[keep]) + a_ap


def main() -> None:
    M.config_from_argv()
    bad = M.verify()
    if bad:
        raise SystemExit(f"member set not clean: {bad}")
    data = {f: M.fold_frame(f) for f in M.FOLDS}

    # THE REPAIR. The stacker fits the residual from an anchor and adds it back.
    # That anchor has always been the FLAT average, because when this harness was
    # built flat was the best simple rule. Under blend-4 it is not: flat5 is 1.92 s
    # worse than blend-4's NNLS, so a stacker anchored on flat starts 1.92 s in the
    # hole and has to climb out before it can win anything. Anchoring on the blend
    # it is trying to beat is the obvious fix and costs one line.
    #
    # The anchor must be OUT OF SAMPLE on the stacker's own training rows, or the
    # stacker learns to correct a blend that has seen those labels. So the anchor
    # on every fold is the ROTATED blend: for fold g, NNLS fitted on the other two
    # and applied to g. That is the same object the rotation already scores.
    anchor = {}
    for g in M.FOLDS:
        src = pd.concat([data[h] for h in M.FOLDS if h != g], ignore_index=True)
        anchor[g], _, _ = fit_blend4(src, data[g])

    recs, saved = [], {}
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        cl = S.clusters(ap)
        ref, w6, w5 = fit_blend4(fit, ap)
        flat = flat_general(ap)
        reg = ap["regime"].to_numpy()
        got = {"blend-4 NNLS": ref, "flat5": flat}
        a_fit = np.concatenate([anchor[g] for g in M.FOLDS if g != f])
        a_ap = anchor[f]
        for lab, cols in (("stack a5 (members only)", V.NOGATE),
                          ("stack a2 (members+gates)", V.ALLC)):
            bag = np.maximum(np.mean([V.arm(fit, ap, s, cols)[0] for s in V.SEEDS], axis=0),
                             B.FLOOR)
            got[f"{lab}, full"] = bag
            got[f"{lab}, keep-noac"] = np.where(reg == "noac", ref, bag)
            bagB = np.maximum(np.mean([arm_anchored(fit, ap, s, cols, a_fit, a_ap)
                                       for s in V.SEEDS], axis=0), B.FLOOR)
            got[f"{lab} on blend-4 anchor, full"] = bagB
            got[f"{lab} on blend-4 anchor, keep-noac"] = np.where(reg == "noac", ref, bagB)
        ids = data[f]["MVT_ID_mvt"].to_numpy()
        saved[f] = got
        for lab, p in got.items():
            d, lo, hi = S.boot_delta(y, p, ref, cl)
            # rmse_clean alongside rmse. Since the 2026-09-06 board read the
            # desk's predictor is CLEAN board + 4 s, and january's all-rows number
            # is dominated by two unreached-rollover rows, so an arm ordering on
            # all-rows need not be the ordering on the metric we are steering by.
            recs.append(dict(fold=f, variant=lab, vs_blend4=d, lo=lo, hi=hi,
                             rmse_clean=float(D.rmse_clean(y, p, ids, f)),
                             **V.decomp(y, p, ap)))
        print(f"[{f}] blend-4 {S.rmse(y, ref):8.4f} (published {PUBLISHED[f]:.4f}, "
              f"delta {S.rmse(y, ref)-PUBLISHED[f]:+.4f})   " +
              "  ".join(f"{k.split('(')[0].strip()}={S.rmse(y,v):.3f}"
                        for k, v in got.items() if k.startswith("stack")), flush=True)

    T = pd.DataFrame(recs)
    T.to_csv(OUT / "blend4_rotation.csv", index=False)
    be = {v: M.board_estimate({f: T[(T.fold == f) & (T.variant == v)].rmse.iloc[0]
                               for f in M.SHARES}) for v in T.variant.unique()}
    piv = T.pivot(index="variant", columns="fold", values="rmse")[list(M.FOLDS)]
    piv["board"] = pd.Series(be)
    piv["vs blend-4"] = piv["board"] - be["blend-4 NNLS"]
    bec = {v: M.board_estimate({f: T[(T.fold == f) & (T.variant == v)].rmse_clean.iloc[0]
                                for f in M.SHARES}) for v in T.variant.unique()}
    piv["clean"] = pd.Series(bec)
    piv["predict"] = piv["clean"] + 4.0
    print("\n=== rotated, on blend-4's member set ===")
    print(piv.sort_values("board").round(3).to_string())
    print("\n=== vs blend-4's fitted NNLS, day-clustered 95 % ===")
    for r in T[T.variant.str.startswith("stack")].sort_values(["variant", "fold"]).itertuples():
        print(f"  {r.variant:28s} {r.fold:9s} {r.vs_blend4:+7.3f} [{r.lo:+7.3f}, {r.hi:+7.3f}]")
    print("\n=== required decomposition ===")
    print(T[["fold", "variant", "rmse", "le3600", "gt3600", "no_nm", "noac"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
