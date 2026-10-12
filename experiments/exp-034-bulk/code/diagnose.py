"""What this rotation can say about the bulk member's -0.0105 s, in two
directions.  Comments only were changed after the measurement (Sol round 2,
blocking 1); the arms, the numbers and `analysis/diagnose.json` are untouched.

The rotation puts the member at -0.0105 s of clean board against K2, INSIDE the
0.1025 s rotation floor, so this file measures a second thing rather than
explaining a first.  Two readings were named in advance.  They are candidate
readings, not a partition: capacity, regularisation and the four member
aggregates are live alternatives that this experiment does not separate
(RESULTS.md SS 5c).

  (a) **K2 took the ground.**  The member's information is real but the set
      already holds a better copy of it -- `K2-window` is 6.9-7.8 s ahead of it
      on its own bulk slice.  This reading PREDICTS a visible gain on the set
      that existed when L7 was written, so the lane's negative would be about
      TIMING rather than about the frame.
  (b) **The frame was already served.**  The stand-occupancy family overlaps
      the v3 frame's `stand` source (the queue's caution) and `V3-allin` is in
      every set.

One arm tests the prediction (a) makes: the member added to **stack-5's own
seven with the window block** -- v6's construction, the head as it stood before
K2 existed -- scored against v7 on the same instrument.  A gain there and none
on K2's set would support (a).  No gain in either is evidence AGAINST (a).  It
is NOT a proof of (b) and NOT an equivalence claim: no equivalence margin was
registered, and an effect of a few hundredths of a second is below what this
instrument resolves.

Second measurement, on the arm that was scored: the LightGBM **split gain** the
stacker spends on the member column, per fold, as a share of the total.  A
stacker that ignores a column and one that uses it and gains nothing are
different findings.  The share is a simplex, so the rise in other columns when
the member leaves is an ASSOCIATION, not a demonstration that particular splits
moved from one column to another.

    PRC_THREADS=6 python diagnose.py   ->  analysis/diagnose.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import candidate as C  # noqa: E402
import frames as F  # noqa: E402
import stack5 as S  # noqa: E402
import stack5w as SW  # noqa: E402
import score as Sc  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = C.FOLDS
PUB_V6_BOARD = 257.6230
#: v6's construction with the bulk member added: B5 (F1-fresh, no K2) + block.
NAMES_V6B = C._add(S.B5, C.BULK)


def importance(data: dict, names, anchor: dict, blk: dict) -> dict:
    """The share of LightGBM split gain the stacker spends on each column.

    One seed of the five (`S.SEEDS[0]`) and one rotation, which is enough for a
    share: the five seeds differ only in their bagging draws.
    """
    out = {}
    for f in FOLDS:
        fit = pd.concat([data[g] for g in FOLDS if g != f], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in FOLDS if g != f])
        xf = pd.concat([blk[g] for g in FOLDS if g != f], ignore_index=True)
        X = pd.concat([S.member_features(fit, names).reset_index(drop=True),
                       xf.reset_index(drop=True)], axis=1)
        p = dict(S.LGB_BASE, num_threads=C.THREADS, seed=S.SEEDS[0],
                 bagging_seed=S.SEEDS[0] + 1, feature_fraction_seed=S.SEEDS[0] + 2,
                 data_random_seed=S.SEEDS[0] + 3)
        mdl = lgb.train(p, lgb.Dataset(X, label=fit["y_true"].to_numpy("float64") - a_fit),
                        num_boost_round=S.ROUNDS)
        g = np.asarray(mdl.feature_importance("gain"), "float64")
        tot = g.sum()
        s = pd.Series(g / tot, index=mdl.feature_name()).sort_values(ascending=False)
        out[f] = {"total_gain": float(tot),
                  "member_share": {n: round(float(s.get(n, 0.0)), 5) for n in names},
                  "aggregate_share": {n: round(float(s.get(n, 0.0)), 5)
                                      for n in ("mem_mean", "mem_std", "mem_range", "n_defined")},
                  "window_block_share": round(float(sum(s.get(c, 0.0) for c in xf.columns)), 5),
                  "top10": {k: round(float(v), 5) for k, v in s.head(10).items()}}
        print(f"[gain] {f:9s} "
              + "  ".join(f"{n} {out[f]['member_share'][n]:.4f}" for n in names)
              + f"   block {out[f]['window_block_share']:.4f}", flush=True)
    return out


def main() -> None:
    t0 = time.time()
    data = C.frames()
    a5 = C.anchors(data, C.NAMES5)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        if abs(float(np.sqrt(np.mean((y - a5[f]) ** 2))) - S.PUBLISHED[f]) > C.GATE:
            raise SystemExit(f"gate A fails on {f}")
    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    blk = {f: SW.block_for(f, SW.panel_of(data[f]), {"blocks": [C.BLOCK], "win": C.WIN})
           for f in FOLDS}
    ref = Sc.reference()

    rows = {}
    for lab, names in (("v6", S.B5), ("v6+bulk", NAMES_V6B)):
        t = time.time()
        p = C.align(C.rotate(data, names, a5, blk, ref5), data, ref)
        pe, _ = C.eham_mask(p, ref)
        clean = {f: D.rmse_clean(ref[f]["y_true"], pe[f], ref[f]["MVT_ID_mvt"], f)
                 for f in FOLDS}
        rows[lab] = {"members": list(names), "clean": {f: round(clean[f], 4) for f in FOLDS},
                     "board": round(Sc.P.board(clean), 4), "vectors": pe}
        print(f"[{lab:9s}] " + "  ".join(f"{f} {clean[f]:9.4f}" for f in FOLDS)
              + f"  board {Sc.P.board(clean):9.4f}  {time.time()-t:.0f}s", flush=True)

    # the arms above carry v7's EHAM mask, so `v6` here is v7; gate it on v7
    v7, _ = C.eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)
    md = {f: float(np.abs(rows["v6"]["vectors"][f] - v7[f]).max()) for f in FOLDS}
    print(f"[gate] the `v6` arm with v7's mask against the committed v7: "
          f"max abs row difference {max(md.values()):.3g}", flush=True)
    if max(md.values()) > C.VEC_TOL:
        raise SystemExit(f"the diagnostic baseline is not v7: {md}")

    # the gain the member buys on the PRE-K2 set, with day-clustered intervals
    rec = {"arm": "v6+bulk-eham-out", "clean": rows["v6+bulk"]["clean"],
           "clean_board": rows["v6+bulk"]["board"],
           "baseline": "v7 (v6 with v7's EHAM mask)", "baseline_clean": rows["v6"]["clean"],
           "baseline_board": rows["v6"]["board"]}
    C.against(rec, rows["v6+bulk"]["vectors"], ref, v7, "v7", C.V7_BOARD)
    for k in ("vectors",):
        rows["v6"].pop(k), rows["v6+bulk"].pop(k)

    print("\n=== split gain of each stacker column, arm bulk-add-hold ===", flush=True)
    imp = importance(data, C.NAMESB, a5, blk)
    print("\n=== the same stacker WITHOUT the bulk member, for the contrast ===", flush=True)
    imp_k2 = importance(data, C.NAMESK2, a5, blk)

    out = {"question": "does the bulk member add to the set that existed before K2?",
           "arms": rows, "vs_v7": rec,
           "gain_split_with_bulk": imp, "gain_split_k2_only": imp_k2,
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / "diagnose.json").write_text(json.dumps(out, indent=2))
    print(f"\n[diagnose] {out['wall_s']:.0f}s -> analysis/diagnose.json", flush=True)


if __name__ == "__main__":
    main()
