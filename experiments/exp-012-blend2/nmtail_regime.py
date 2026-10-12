"""Repair attempt: give the NM-tail head weight only where its mechanism lives.

Added flat to a four-member core the head is worth -0.015 s rotated. Before
recording that, the obvious repair: the head exists because the airport block
clock sometimes sat on LOBT rather than AOBT, and `nm_late` -- NM-joined with
AOBT - LOBT above 1800 s -- is the input-defined population where that is most
likely. If the head has an edge anywhere it is there, and a member that helps on
2 % of rows and is dead weight on the rest should be given weight on the 2 %.

So: five-member weights inside `nm_late`, four-member weights everywhere else,
all fitted on the other two months. Rotation unchanged.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parents[1]), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import blend as BL  # noqa: E402
import prc.blend as B  # noqa: E402
import nmtail_rotate as NR  # noqa: E402


def enriched(fold, head_mode="phi"):
    d = NR.fold_data(fold, head_mode)
    reg = BL.regime_frame(fold)
    base = NR.load_one(NR.CORE["L2-hub"], fold, "L2-hub")[["MVT_ID_mvt"]]
    j = base.merge(reg, on="MVT_ID_mvt", how="left")
    d["regime"] = B.regimes_from_features(j)
    return d


def main(mode="nnls", head_mode="phi", target="nm_late", outer=None):
    # `outer` lets the blend OUTSIDE the target regime use a different rule
    # from the one inside it. That is the configuration the desk's rule and
    # this repair point at together: flat over the four general members,
    # where fitting three months to four weights was shown to overfit, and a
    # fitted vector inside nm_late, where a specialist member earns real and
    # stable weight (0.297/0.331/0.336 across the three rotations).
    outer = outer or mode
    data = {f: enriched(f, head_mode) for f in BL.FOLDS}
    k = len(data[BL.FOLDS[0]]["names"])
    for f in BL.FOLDS:
        d = data[f]
        sel = (d["regime"] == target) & d["ok"]
        print(f"  {f:9s} {target} & covered: {int(sel.sum()):,} of {len(sel):,} "
              f"({100*sel.mean():.2f} %)")
    per_fold, per_core = {}, {}
    for f in BL.FOLDS:
        src = [g for g in BL.FOLDS if g != f]
        selg = {g: (data[g]["regime"] == target) & data[g]["ok"] for g in src}
        # five members inside the target regime
        P5 = np.concatenate([data[g]["P"][selg[g]] for g in src])
        y5 = np.concatenate([data[g]["y"][selg[g]] for g in src])
        # four members everywhere else
        P4 = np.concatenate([data[g]["P"][~selg[g]][:, :k-1] for g in src])
        y4 = np.concatenate([data[g]["y"][~selg[g]] for g in src])
        w5 = NR.weights(P5, y5, mode, k)
        w4 = NR.weights(P4, y4, outer, k - 1)
        w4c = NR.weights(P5[:, :k-1], y5, mode, k - 1)   # control inside the regime
        d = data[f]
        sel = (d["regime"] == target) & d["ok"]
        p = np.empty(len(d["y"])); pc = np.empty(len(d["y"]))
        p[sel] = d["P"][sel] @ w5
        pc[sel] = d["P"][sel][:, :k-1] @ w4c
        p[~sel] = pc[~sel] = d["P"][~sel][:, :k-1] @ w4
        p, pc = np.maximum(p, B.FLOOR), np.maximum(pc, B.FLOOR)
        per_fold[f] = float(np.sqrt(((d["y"] - p) ** 2).mean()))
        per_core[f] = float(np.sqrt(((d["y"] - pc) ** 2).mean()))
        rr = lambda v: float(np.sqrt(((d["y"][sel] - v[sel]) ** 2).mean()))
        print(f"  {f:9s} core {per_core[f]:8.3f}  +N1 {per_fold[f]:8.3f}  "
              f"gain {per_core[f]-per_fold[f]:+6.3f}   w(N1)={w5[-1]:.3f}   "
              f"inside {target}: {rr(pc):.0f} -> {rr(p):.0f}")
    b0, b1 = BL.board_estimate(per_core), BL.board_estimate(per_fold)
    print(f"\n  board  core {b0:.3f}  ->  +N1 in {target} {b1:.3f}   gain {b0-b1:+.3f}"
          f"   [inside {mode}, outside {outer}]")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
