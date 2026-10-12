"""The sibling vector priced, so section 37.5 quotes a measurement and not a sum.

`d-bag5-255-rider.npz` is the bag off LFPG with the rider at LFPG -- the vector v7 would
have shipped had LFPG been held.  The desk estimated it at +0.19937 + 0.01940 = +0.21877
over v18's bytes by adding the two increments.  Board seconds are not additive (the gain
is `2A - B` and the two vectors share rows nowhere but share a residual everywhere), so
this prices it directly, on the same panel and against the same head H as section 37.1.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/siblingprice.py
"""
from __future__ import annotations

import json
import sys
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

OUT = EXP / "analysis/sibling-price.json"


def main() -> None:
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}

    rec = {"what": __doc__.strip().splitlines()[0], "prices": {}}
    for key, path, label in (
            ("bag_rider", EXP / "work/d-bag5-255-rider.npz",
             "the SIBLING: the bag off LFPG with the rider at LFPG, over v18's bytes"),
            ("bag_free", EXP / "work/d-bag5-255-free.npz",
             "what v7 ships: the bag with LFPG unheld, over v18's bytes")):
        vec = {f: v.astype("float64") for f, v in np.load(path).items()}
        pr = R.price(pan, vec, H, label)
        rec["prices"][key] = {
            "vector": str(path.relative_to(ROOT)),
            "board_all_row_gain_s": pr["board_all_row"]["gain_s"],
            "board_all_row_ci95": pr["board_all_row"]["interval"]["ci95"],
            "board_clean_gain_s": pr["board_clean"]["gain_s"],
            "every_fold_interval_clear": bool(pr["every_fold_interval_clear"]),
            "by_fold": {f: {"mse_gain_s2": pr["by_fold"][f]["mse_gain_s2"],
                            "ci95": pr["by_fold"][f]["ci95"],
                            "rows_moved": pr["by_fold"][f]["rows_moved"]}
                        for f in A.FOLDS},
        }
    a = rec["prices"]["bag_rider"]["board_all_row_gain_s"]
    b = rec["prices"]["bag_free"]["board_all_row_gain_s"]
    rec["difference_s"] = b - a
    rec["the_desk_estimate_by_addition_s"] = 0.19937 + 0.01940
    rec["addition_error_s"] = a - (0.19937 + 0.01940)
    rec["refit_floor_s"] = 0.06
    rec["difference_is_below_the_refit_floor"] = bool(abs(b - a) < 0.06)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(json.dumps({k: {"board": v["board_all_row_gain_s"],
                          "ci95": v["board_all_row_ci95"],
                          "clear": v["every_fold_interval_clear"]}
                      for k, v in rec["prices"].items()}, indent=2))
    print("difference (free - sibling):", round(b - a, 6), "s")
    print("the desk's additive estimate:", rec["the_desk_estimate_by_addition_s"],
          "| error:", round(rec["addition_error_s"], 6), "s")


if __name__ == "__main__":
    main()
