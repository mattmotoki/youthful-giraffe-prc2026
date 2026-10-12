"""Price the SERVED vector on the fold side, with the held airport zeroed.

The desk holds LFPG at the base on the ranking side (2026-09-09 00:05).  The fold gate was
read on the arm INCLUDING LFPG, so the vector that ships is not exactly the vector that was
priced.  This re-prices the saved fold-side update with the same airports zeroed -- no
refit, the same `d` with a mask -- so the gate is read on exactly what the candidate
contains.  It is the arm's own contract table, not a new arm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

HELD = ("LFPG",)
ARM = "c_lgb_shipped"


def main(arm: str = ARM, held: tuple[str, ...] = HELD) -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    z = np.load(EXP / f"work/d-{arm}.npz")
    full = {f: z[f].astype("float64") for f in A.FOLDS}
    kept, rows_zeroed = {}, {}
    for f in A.FOLDS:
        d = full[f].copy()
        m = np.isin(pan[f].airport.to_numpy().astype(str), list(held))
        rows_zeroed[f] = int((d[m] != 0).sum())
        d[m] = 0.0
        kept[f] = d
    before = BA.price(pan, full, rng)
    after = BA.price(pan, kept, rng)
    out = {
        "arm": arm, "held_airports": sorted(held),
        "what": ("the same saved fold-side update with the held airports zeroed -- no "
                 "refit, so this is the arm's own contract table read on the vector the "
                 "candidate actually contains"),
        "rows_returned_to_the_base_by_fold": rows_zeroed,
        "before": {k: before[k] for k in ("board_all_row", "board_clean", "december_rmse",
                                          "every_fold_interval_clear", "board_basis",
                                          "gate_all_row_gain_and_every_interval_clear")},
        "after": {k: after[k] for k in ("board_all_row", "board_clean", "december_rmse",
                                        "every_fold_interval_clear", "board_basis",
                                        "gate_all_row_gain_and_every_interval_clear")},
        "cost_of_the_hold_s": {
            "all_row": before["board_all_row"]["gain_s"] - after["board_all_row"]["gain_s"],
            "clean": before["board_clean"]["gain_s"] - after["board_clean"]["gain_s"],
            "december": before["december_rmse"]["gain_s"] - after["december_rmse"]["gain_s"]},
        "tail_clauses_after": {f: A.tail_clauses(pan[f], kept[f]) for f in A.FOLDS},
        "projection_after": A.project_to_ranking(
            {f: A.by_group(pan[f], kept[f], "airport", rng) for f in A.FOLDS},
            after["board_all_row"]["base"]),
    }
    (EXP / "analysis/held-price.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float))
    print(f"held {sorted(held)}: all-row {before['board_all_row']['gain_s']:+.4f} -> "
          f"{after['board_all_row']['gain_s']:+.4f} s, clean "
          f"{before['board_clean']['gain_s']:+.4f} -> {after['board_clean']['gain_s']:+.4f}, "
          f"december {before['december_rmse']['gain_s']:+.4f} -> "
          f"{after['december_rmse']['gain_s']:+.4f}, every fold clear "
          f"{after['every_fold_interval_clear']}, rows zeroed {rows_zeroed}")
    for f in A.FOLDS:
        g = out["tail_clauses_after"][f]["registered_clause_y_over_3600s"]
        print(f"  tail {f}: {g['rmse_gain_s']:+.3f} s passes {g['passes_the_clause']}")
    p = out["projection_after"]
    print(f"  projected {p['projected_board_gain_s']:+.4f} s, conservative "
          f"{p['conservative']['projected_board_gain_s']:+.4f} s")


if __name__ == "__main__":
    main()
