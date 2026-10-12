"""Why did `A` fall between section 41.5 and section 40.3's read?  Two candidates, separated.

Section 40.3 registered: "A at K = 10 differing from +56.52 on july by more than a couple of
percent means the chaos is not orthogonal to the residual and every gain figure in section
41.5 is mis-stated."  It read 40.38, so the refutation fires on its wording.  But there is a
CONFOUND that the wording did not consider, and it is mine:

  * section 41.5 measured `A = <r, d>` with `r = y - pbase - bag255_K5`   (the K = 5 head)
  * section 40.3's read measured it with `r = y - pbase - bag255_K10`     (the K = 10 head)

The K = 10 255 bag is a BETTER head -- it already took +0.0225 of board -- so its residual is
smaller and less aligned with anything.  If that is the whole story, `A` measured on the K = 5
residual with the SAME K = 10 increment comes back near 56.52 and orthogonality is intact; my
forecast was mis-specified, not the physics.  If it comes back near 40, the head change is not
the cause and the registered refutation stands as written.

This does no fits.  It loads the three bag vectors already on disk and reads `A` both ways.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/awhy.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402

OUT = EXP / "analysis/a-why.json"
REGISTERED = {"july_loo": 56.5241431942054, "december": 101.15389616752107,
              "january": 99.50464957963217}


def main() -> None:
    t0 = time.time()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    def load(name):
        return {f: v.astype("float64") for f, v in np.load(EXP / f"work/{name}").items()}
    b255_k5 = load("d-bag5-255-free.npz")
    b255_k10 = load("d-bag10-255-free.npz")
    b511_k10 = load("d-bag10-511-free.npz")
    b511_k5 = load("d-bag5-511-free.npz")

    rec = {"what": __doc__.strip().splitlines()[0], "by_fold": {}}
    for f in A.FOLDS:
        y = pan[f].y_true.to_numpy("float64")
        pb = pan[f].pbase.to_numpy("float64")
        r_k5 = y - pb - b255_k5[f]
        r_k10 = y - pb - b255_k10[f]
        d10 = b511_k10[f] - b255_k10[f]          # the K = 10 increment as priced
        d5 = b511_k5[f] - b255_k5[f]             # the K = 5 increment section 41.5 read
        row = {
            "A_registered_from_section_41_5": REGISTERED[f],
            "A_K5_increment_on_K5_residual": float(np.mean(r_k5 * d5)),
            "A_K10_increment_on_K10_residual": float(np.mean(r_k10 * d10)),
            "A_K10_increment_on_K5_residual": float(np.mean(r_k5 * d10)),
            "A_K5_increment_on_K10_residual": float(np.mean(r_k10 * d5)),
            "residual_rms_K5_head_s": float(np.sqrt(np.mean(r_k5 ** 2))),
            "residual_rms_K10_head_s": float(np.sqrt(np.mean(r_k10 ** 2))),
        }
        # how much of the fall is the head, how much is the increment?
        fall_total = row["A_K10_increment_on_K10_residual"] - row["A_K5_increment_on_K5_residual"]
        fall_head = row["A_K10_increment_on_K10_residual"] - row["A_K10_increment_on_K5_residual"]
        fall_inc = row["A_K10_increment_on_K5_residual"] - row["A_K5_increment_on_K5_residual"]
        row["fall_total"] = fall_total
        row["fall_from_the_head_change"] = fall_head
        row["fall_from_the_increment_change"] = fall_inc
        row["share_of_the_fall_that_is_the_head"] = (fall_head / fall_total) if fall_total else None
        rec["by_fold"][f] = row
        print(f"{f:10s} A: K5inc/K5res {row['A_K5_increment_on_K5_residual']:7.2f}   "
              f"K10inc/K5res {row['A_K10_increment_on_K5_residual']:7.2f}   "
              f"K10inc/K10res {row['A_K10_increment_on_K10_residual']:7.2f}", flush=True)
        print(f"           fall {fall_total:+7.2f} = head {fall_head:+7.2f} + increment "
              f"{fall_inc:+7.2f}  ({row['share_of_the_fall_that_is_the_head']:.0%} head)",
              flush=True)

    j = rec["by_fold"]["july_loo"]
    rec["verdict"] = {
        "the_head_change_explains_it": bool(j["share_of_the_fall_that_is_the_head"] > 0.8),
        "july_A_on_the_matched_head": j["A_K10_increment_on_K5_residual"],
        "july_A_registered": REGISTERED["july_loo"],
        "means": ("if the head change explains it, section 41.5's orthogonality finding is "
                  "intact and MY FORECAST was mis-specified -- it priced the K = 10 increment "
                  "against the K = 5 head. If not, section 40.3's registered refutation "
                  "stands as written and A really did move."),
    }
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("\nthe head change explains it:", rec["verdict"]["the_head_change_explains_it"])
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
