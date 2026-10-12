"""The two halves of v16's increment, priced apart and label-free of the board.

Desk order, 2026-09-09, after BOARD READING 15 (v16's increment transferred at x0.36, below
its own interval's lower bound):

    price the increment's two parts APART on the folds, label-free of the board: the guard's
    fold-side worth with intervals and its ranking footprint by month; the capacity step
    127->255 CROSS-FITTED -- leaves selected on two folds, priced on the third -- which is
    the registered rule for any climbed increment from here.

**Why cross-fitting is the right instrument.** The leaf ladder was climbed on the same three
folds that then priced the winner's increment, so the winner's margin carries the maximum of
a search. Selecting the leaf count on two folds and pricing it on the third prices the
PROCEDURE rather than its winner: on a fold where the out-of-fold selection would not have
chosen 255, the increment is zero there, and that is the honest contribution.

Nothing is refitted. Every setting's fold vector was saved by `capacity.py`
(`work/d-capacity-<setting>.npz`) and the guarded ones by `guardarm.py`, so this is a
re-pricing of vectors that already exist, on the pinned population.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/crossfit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

HEAD = "leaves_127x300_lr050"
#: the ladder that was climbed, in the order it was climbed (`capacity.CLIMB`)
LADDER = ["leaves_15x600_lr050", "predeclared_31x300_lr050", "leaves_63x300_lr050",
          "leaves_127x300_lr050", "leaves_255x300_lr050"]
HOLD = ("LFPG",)
OUT = EXP / "analysis/crossfit.json"


def held(pan: dict, v: np.ndarray, f: str) -> np.ndarray:
    """The serve's own hold, applied to a fold vector."""
    out = v.astype("float64").copy()
    out[np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD))] = 0.0
    return out


def head_panel(pan: dict, dh: dict) -> dict:
    """`pbase + d_head`, the base an increment over the head is priced against."""
    out = {}
    for f in A.FOLDS:
        q = pan[f].copy()
        q["pbase"] = q["pbase"].to_numpy("float64") + dh[f]
        q["rbase"] = q["y_true"].to_numpy("float64") - q["pbase"].to_numpy("float64")
        out[f] = q
    return out


def main() -> None:
    pan = A.panel_with_block(BA.load())
    cap = json.loads((EXP / "analysis/capacity.json").read_text())
    vec = {c: dict(np.load(EXP / f"work/d-capacity-{c}.npz")) for c in LADDER}
    dh = {f: held(pan, vec[HEAD][f], f) for f in A.FOLDS}
    pan_head = head_panel(pan, dh)

    out = {
        "what": ("v16's increment split into its two parts, each priced on the folds and "
                 "label-free of the board: the tail guard, and the capacity step 127->255 "
                 "cross-fitted so the leaf count is selected out of fold"),
        "ordered_by": "the desk, 2026-09-09, after BOARD READING 15",
        "no_refit": ("every vector was saved by capacity.py and guardarm.py on the pinned "
                     "population; this file only re-prices them"),
        "head": HEAD, "ladder": LADDER, "held": list(HOLD),
    }

    # ---------------------------------------------------------------- part 1: capacity
    # each rung's increment over the head, per fold, so the out-of-fold selection can read
    # the same quantity the gate reads
    per_fold = {}
    for c in LADDER:
        inc = {f: held(pan, vec[c][f], f) - dh[f] for f in A.FOLDS}
        rec = BA.price(pan_head, inc, np.random.default_rng(A.SEED))
        per_fold[c] = {
            f: {"mse_gain_s2_per_row": rec["by_fold"][f]["all_row"]["mse_gain_s2"],
                "ci95": rec["by_fold"][f]["all_row"]["ci95"],
                "clear": bool(rec["by_fold"][f]["all_row"]["ci95"][0] > 0),
                "tail_passes": bool(cap["settings"][c]["tail_clauses"][f]
                                    ["registered_clause_y_over_3600s"]["passes_the_clause"])}
            for f in A.FOLDS}
        per_fold[c]["all_row_gain_s"] = rec["board_all_row"]["gain_s"]
        per_fold[c]["all_row_ci95"] = rec["board_all_row"]["interval"]["ci95"]
    out["each_rung_over_the_head"] = per_fold

    # the out-of-fold selection, per scored fold
    picks, d_cross = {}, {}
    for f in A.FOLDS:
        sel = [g for g in A.FOLDS if g != f]
        cand = []
        for c in LADDER:
            if c == HEAD:
                continue
            ok = all(per_fold[c][g]["tail_passes"] and per_fold[c][g]["clear"] for g in sel)
            score = float(np.mean([per_fold[c][g]["mse_gain_s2_per_row"] for g in sel]))
            # the alternative criterion, reported so the pick can be seen not to hinge on it
            alt = float(np.mean([cap["settings"][c]["by_fold"][g]["all_row"]["mse_gain_s2"]
                                 for g in sel]))
            cand.append({"setting": c, "eligible": ok, "score_increment_s2": score,
                         "score_arm_over_v12_s2": alt})
        elig = [x for x in cand if x["eligible"]]
        best = max(elig, key=lambda x: x["score_increment_s2"]) if elig else None
        best_alt = (max(elig, key=lambda x: x["score_arm_over_v12_s2"])
                    if elig else None)
        chosen = best["setting"] if best else HEAD
        picks[f] = {
            "selected_on": sel, "chosen": chosen,
            "chosen_under_the_alternative_criterion":
                best_alt["setting"] if best_alt else HEAD,
            "the_two_criteria_agree": bool(best and best_alt
                                           and best["setting"] == best_alt["setting"]),
            "candidates": cand,
        }
        d_cross[f] = (held(pan, vec[chosen][f], f) - dh[f]) if chosen != HEAD \
            else np.zeros(len(pan[f]))
    out["out_of_fold_selection"] = picks

    rec_x = BA.price(pan_head, d_cross, np.random.default_rng(A.SEED))
    out["capacity_step_cross_fitted"] = {
        "what": ("the leaf count selected on the other two folds and priced on the scored "
                 "one; on a fold where the out-of-fold selection would not have climbed, "
                 "the increment is zero there"),
        "all_row_gain_s": rec_x["board_all_row"]["gain_s"],
        "all_row_ci95": rec_x["board_all_row"]["interval"]["ci95"],
        "clean_gain_s": rec_x["board_clean"]["gain_s"],
        "december_gain_s": rec_x["december_rmse"]["gain_s"],
        "every_fold_interval_clear": bool(rec_x["every_fold_interval_clear"]),
        "by_fold": {f: rec_x["by_fold"][f]["all_row"] for f in A.FOLDS},
    }
    out["capacity_step_as_selected_in_fold"] = {
        "what": "255 leaves over the head on every fold, the way the climb chose it",
        "all_row_gain_s": per_fold["leaves_255x300_lr050"]["all_row_gain_s"],
        "all_row_ci95": per_fold["leaves_255x300_lr050"]["all_row_ci95"],
    }
    out["selection_premium_s"] = (out["capacity_step_as_selected_in_fold"]["all_row_gain_s"]
                                  - out["capacity_step_cross_fitted"]["all_row_gain_s"])

    # ------------------------------------------------------------------- part 2: guard
    gu = dict(np.load(S.npz_path("c_lgb_shipped@leaves_255x300_lr050+guard995")))
    un = vec["leaves_255x300_lr050"]
    inc_g, base_g, hold_rows = {}, {}, {}
    for f in A.FOLDS:
        u = held(pan, un[f], f)
        g = held(pan, gu[f], f)
        inc_g[f] = g - u
        m = (g == 0) & (u != 0)
        hold_rows[f] = m
        q = pan[f].copy()
        q["pbase"] = q["pbase"].to_numpy("float64") + u
        q["rbase"] = q["y_true"].to_numpy("float64") - q["pbase"].to_numpy("float64")
        base_g[f] = q
    rec_g = BA.price(base_g, inc_g, np.random.default_rng(A.SEED))
    out["guard_on_the_folds"] = {
        "what": ("the guard's own worth: the guarded vector priced as an increment over the "
                 "UNGUARDED one, on a base of pbase + the unguarded update"),
        "all_row_gain_s": rec_g["board_all_row"]["gain_s"],
        "all_row_ci95": rec_g["board_all_row"]["interval"]["ci95"],
        "every_fold_interval_clear": bool(rec_g["every_fold_interval_clear"]),
        "by_fold": {f: {"mse_gain_s2_per_row": rec_g["by_fold"][f]["all_row"]["mse_gain_s2"],
                        "ci95": rec_g["by_fold"][f]["all_row"]["ci95"],
                        "rows_held": int(hold_rows[f].sum()),
                        "share_of_moved_rows_held": round(
                            float(hold_rows[f].sum()
                                  / max(1, int((held(pan, un[f], f) != 0).sum()))), 4)}
                    for f in A.FOLDS},
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    print("each rung over the head (all-row s):")
    for c in LADDER:
        print("  %-28s %+8.4f  per fold %s" % (
            c, per_fold[c]["all_row_gain_s"],
            " ".join("%s %+7.1f%s" % (f[:3], per_fold[c][f]["mse_gain_s2_per_row"],
                                      "" if per_fold[c][f]["clear"] else "*")
                     for f in A.FOLDS)))
    print("\nout-of-fold selection:")
    for f, r in picks.items():
        print("  scored %-9s selected on %-22s -> %s  (alt criterion %s, agree %s)" % (
            f, "+".join(x[:3] for x in r["selected_on"]), r["chosen"],
            r["chosen_under_the_alternative_criterion"], r["the_two_criteria_agree"]))
    x = out["capacity_step_cross_fitted"]
    n = out["capacity_step_as_selected_in_fold"]
    print("\ncapacity step, in-fold selected  %+8.4f [%+7.4f,%+7.4f]"
          % (n["all_row_gain_s"], n["all_row_ci95"][0], n["all_row_ci95"][1]))
    print("capacity step, CROSS-FITTED      %+8.4f [%+7.4f,%+7.4f]  clear %s"
          % (x["all_row_gain_s"], x["all_row_ci95"][0], x["all_row_ci95"][1],
             x["every_fold_interval_clear"]))
    print("selection premium                %+8.4f" % out["selection_premium_s"])
    g = out["guard_on_the_folds"]
    print("\nguard's own worth                %+8.4f [%+7.4f,%+7.4f]  clear %s"
          % (g["all_row_gain_s"], g["all_row_ci95"][0], g["all_row_ci95"][1],
             g["every_fold_interval_clear"]))
    for f, r in g["by_fold"].items():
        print("  %-9s %+8.1f s2/row [%+7.1f,%+7.1f]  held %4d (%.2f %% of moved)"
              % (f, r["mse_gain_s2_per_row"], r["ci95"][0], r["ci95"][1],
                 r["rows_held"], 100 * r["share_of_moved_rows_held"]))
    print("\nwritten to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
