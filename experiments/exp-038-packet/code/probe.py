"""The training-size probe: RESULTS.md section 2a, next step 1.

**The question.** Reading (i) of section 2a says the fold instrument is not the
served vector: the rotation's head trains on two of three months, the served
head on twelve.  IF the shrinkage that makes `v10 + s(stack-5 - v10)` positive
is an ESTIMATION-VARIANCE effect, then the optimal amplitude must RISE as the
head's training data falls, and the served side needs LESS than the 0.27 the
rotation fits -- so 0.27 is an upper bound and the served gain is smaller than
+0.10.  If `s` does not move with training size, that reading is wrong and 0.27
is not a bound in either direction.

**What is refit and what is not.**  Only the HEAD moves: the 600 s `depx`
window stacker on v10's own seven members and its mixed anchor, refit per fold
on ONE neighbouring month instead of two, the same five seeds and the same 300
rounds.  The member frames, the hybrid NNLS anchor, the block, `v7`, `stack-5`
and both masks stay the committed bytes.  No member fit runs.  Six head fits
(three folds x two neighbours), plus the two-month reference, which must
reproduce exp-036's published triple before any probe number is read.

**Two confounds, measured and not argued.**

* One month is half the rows AND one season.  The `half-rows` arm fits on both
  neighbours at a seeded 50 % row subsample: the row count matches a one-month
  fit and the seasonal composition does not.  `half-rows` against the reference
  is the SIZE effect; `one-month` against `half-rows` is the SEASON effect.
* Off the eight kept airports v10 is `v7`, which this probe does not refit, so
  `d` is identical there in every arm.  Every quantity is therefore reported
  twice: on all rows, and on the eight kept airports alone, where every row's
  prediction really did come from a refitted head.

Reading rejected: refitting `stack-5` as well.  The amplitude is how much of an
OLD, FIXED vector a NEW head wants; the served stack-5 file is fixed bytes on
the ranking months too, so holding it fixed here is the like-for-like choice
and it keeps the probe to the six head fits the desk asked for.
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
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-034-bulk/code"))
import align as A  # noqa: E402
import candidate as C  # noqa: E402
import q02 as Q  # noqa: E402

F, R, S, SW, Sc = C.F, C.R, C.S, C.SW, C.Sc
B, D = C.B, C.D
FOLDS = C.FOLDS
THREADS = int(os.environ.get("PRC_THREADS", "4"))
SW.THREADS = THREADS
SEEDS = S.SEEDS
SUB_SEED = 3803          # the row subsample of the `half-rows` arm
DRAWS = 2000
CAND = "stack-5"


def rotate_from(data, names, anchor, blk, noac_ref, src_of, frac=1.0):
    """`q02.rotate_seeds` with the FIT months chosen per fold and an optional
    seeded row subsample of the fit frame."""
    out = {}
    for f in FOLDS:
        ap = data[f]
        src = src_of(f)
        fit = pd.concat([data[g] for g in src], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in src])
        xf = pd.concat([blk[g] for g in src], ignore_index=True)
        if frac < 1.0:
            rng = np.random.default_rng(SUB_SEED + FOLDS.index(f))
            take = rng.random(len(fit)) < frac
            fit, a_fit, xf = fit[take].reset_index(drop=True), a_fit[take], xf[take].reset_index(drop=True)
        bag = np.maximum(np.mean([SW.arm(fit, ap, s, a_fit, anchor[f], names, xf, blk[f])
                                  for s in SEEDS], axis=0), B.FLOOR)
        out[f] = np.where(ap["regime"].to_numpy() == "noac", noac_ref[f], bag)
    return out


def main() -> None:
    t0 = time.time()
    data = C.frames()
    ref = Sc.reference()
    a5 = C.anchors(data, Q.NAMES5)
    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    blk = {f: SW.block_for(f, SW.panel_of(data[f]), {"blocks": [C.BLOCK], "win": C.WIN})
           for f in FOLDS}
    v7 = C.eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)[0]
    keep = Q.MASK["keep"]
    apt = {f: ref[f]["airport"].astype(str).to_numpy() for f in FOLDS}
    y = {f: ref[f]["y_true"].to_numpy("float64") for f in FOLDS}
    ids = {f: ref[f]["MVT_ID_mvt"].to_numpy("float64") for f in FOLDS}
    cand = {f: ref[f]["y_pred_stack5"].to_numpy("float64") for f in FOLDS}
    anchor = Q.anchor_for(data, Q.NAMESK2, "mixed", a5)
    day = {f: A.day_of(ref[f]["t"].to_numpy("float64")) for f in FOLDS}
    blk3 = {f: A.blocks(day[f], 3) for f in FOLDS}
    blk1 = {f: A.blocks(day[f], 1) for f in FOLDS}
    on_keep = {f: np.isin(apt[f], keep) for f in FOLDS}

    # the fit months of each arm.  `nb(f)[k]` is fold f's k-th neighbour in the
    # rotation's own fold order, so "near" and "far" are a fixed rule and not a
    # choice made after seeing a number.
    def nb(f):
        return [g for g in FOLDS if g != f]

    ARMS = [
        ("two-month (reference)", lambda f: nb(f), 1.0, True),
        ("one-month, first neighbour", lambda f: nb(f)[:1], 1.0, False),
        ("one-month, second neighbour", lambda f: nb(f)[1:], 1.0, False),
        ("half-rows, both months", lambda f: nb(f), 0.5, False),
    ]

    rows, arms = [], {}
    for name, src_of, frac, gated in ARMS:
        t = time.time()
        p = C.align(rotate_from(data, Q.NAMESK2, anchor, blk, ref5, src_of, frac),
                    data, ref)
        pe = C.eham_mask(p, ref)[0]
        pe = {f: np.where(on_keep[f], pe[f], v7[f]) for f in FOLDS}
        clean = {f: D.rmse_clean(y[f], pe[f], ids[f], f) for f in FOLDS}
        board = Sc.P.board(clean)
        if gated:
            off = {f: round(clean[f] - Q.PUB["v10"][f], 4) for f in FOLDS}
            if max(abs(v) for v in off.values()) > Q.GATE:
                raise SystemExit(f"the reference arm does not reproduce v10: {off}")
            print(f"[probe] gate: the two-month reference reproduces v10 {off}", flush=True)
        n_fit = {f: int(sum(len(data[g]) for g in src_of(f))
                        * (frac if frac < 1.0 else 1.0)) for f in FOLDS}
        arms[name] = pe
        for pop, sel in (("all rows", {f: np.ones(len(y[f]), bool) for f in FOLDS}),
                         ("eight kept airports", on_keep)):
            st, cv = {}, {}
            for f in FOLDS:
                pr = A.per_row(y[f], pe[f], cand[f])
                st[f] = A.fold_stats(pr["r"], pr["d"], sel[f])
                cv[f] = A.scale_cv(pr["r"], pr["d"], blk3[f], sel[f])
                st[f]["_r"], st[f]["_d"] = pr["r"], pr["d"]
            loo = A.scale_loo({f: st[f]["A"] for f in FOLDS},
                              {f: st[f]["B"] for f in FOLDS},
                              {f: st[f]["n"] for f in FOLDS})
            parts = {}
            for f in A.BOARD_FOLDS:
                r_, d_, sf = st[f]["_r"][sel[f]], st[f]["_d"][sel[f]], loo[f]["s"]
                parts[f] = (blk1[f][sel[f]], r_ * r_, (r_ - sf * d_) ** 2)
            ci = A.boot_board(parts, DRAWS)
            base = sum(A.SHARES[f] * st[f]["mse_v10"] for f in A.BOARD_FOLDS)
            rows.append({
                "arm": name, "population": pop,
                "fit_rows_july": n_fit["july_loo"],
                "clean_july_loo": round(clean["july_loo"], 4),
                "clean_december": round(clean["december"], 4),
                "clean_january": round(clean["january"], 4),
                "clean_board": round(board, 4),
                **{f"scale_{f}": (round(st[f]["scale"], 4) if st[f]["scale"] is not None
                                  else None) for f in FOLDS},
                **{f"amp_loo_{f}": round(loo[f]["s"], 4) for f in FOLDS},
                "amp_loo_mean": round(float(np.mean([loo[f]["s"] for f in FOLDS])), 4),
                "gain_board_loo_amp": round(float(
                    np.sqrt(base) - np.sqrt(sum(A.SHARES[f] * (st[f]["mse_v10"]
                                                               - loo[f]["gain_mse"])
                                                for f in A.BOARD_FOLDS))), 4),
                "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                "gain_board_cv_amp": round(float(
                    np.sqrt(base) - np.sqrt(sum(A.SHARES[f] * (st[f]["mse_v10"]
                                                               - cv[f]["gain_mse"])
                                                for f in A.BOARD_FOLDS))), 4),
                **{f"A_{f}": round(st[f]["A"], 3) for f in FOLDS},
                **{f"B_{f}": round(st[f]["B"], 3) for f in FOLDS},
            })
        print(f"[{name:30s}] board {board:9.4f}  "
              + "  ".join(f"{f} {clean[f]:9.4f}" for f in FOLDS)
              + f"  {time.time() - t:.0f}s", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(EXP / "analysis" / "q01-probe.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        for pop in T.population.unique():
            s = T[T.population == pop]
            print(f"\n=== the amplitude against the head's training size, {pop} ===")
            print(s[["arm", "fit_rows_july", "clean_board", "scale_july_loo",
                     "scale_december", "scale_january", "amp_loo_july_loo",
                     "amp_loo_december", "amp_loo_january", "amp_loo_mean",
                     "gain_board_loo_amp", "ci_lo", "ci_hi",
                     "gain_board_cv_amp"]].to_string(index=False))
    (EXP / "analysis" / "q01-probe.json").write_text(json.dumps(
        {"candidate": CAND, "seeds": list(SEEDS), "subsample_seed": SUB_SEED,
         "rounds": S.ROUNDS, "block": C.BLOCK, "win": C.WIN, "threads": THREADS,
         "note": "only the head is refit; the anchor, the block, v7, stack-5 and both "
                 "masks are the committed bytes. The two-month arm gates to v10.",
         "wall_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\n[probe] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
