"""WHY the 511 rung turned over: where the -0.0628 s went, on vectors that already exist.

`code/rung511.py` read the rung at **-0.06281 [-0.12236, -0.00916]** all-row over v18's
bytes, july_loo clear BELOW zero at -62.58 s2/row, and the registered tail clause IMPROVING
on every fold.  So the pre-registered risk of section 32.3 -- *"the tail clause is where it
fails, if it fails"* -- is refuted twice over: the rung fails, and not there.

A negative closes a path and never a destination (AGENTS.md), so this asks where the loss
sits before anything is proposed to repair it.  Every number is arithmetic on the saved
vectors and the pinned panel; nothing is refitted.

The decomposition, per fold, of `v = 2*r_head*inc - inc**2` summed over rows:

  * the rows the ungated pass ADDED to the class against the rest -- the ~10 % wrong
    placements are here, and a 511-leaf model can memorise a fabricated clock where a
    255-leaf one cannot;
  * the rows the guard holds, which should carry exactly zero;
  * `y > 3600 s` against the bulk, the capacity gradient's own split;
  * by airport, and by decile of |increment|, which says whether the loss is a few large
    moves or a broad drift.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/rungwhy.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import guard as G                                                     # noqa: E402
import rung511 as R                                                   # noqa: E402

OUT = EXP / "analysis/rung-why.json"
TAIL_Y_S = 3600.0


def split(v: np.ndarray, m: np.ndarray, rows: int) -> dict:
    """One split of the fold's total, in s2 summed and in s2 per row of the WHOLE fold."""
    return {"rows": int(m.sum()), "s2_sum": float(v[m].sum()),
            "s2_per_row_of_the_fold": float(v[m].sum() / rows) if rows else 0.0,
            "s2_per_row_of_the_group": float(v[m].mean()) if m.any() else 0.0}


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
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}
    Z = {f: v.astype("float64") for f, v in np.load(R.HELD["A"]).items()}

    cuts = {f: G.cut(pan, [g for g in A.FOLDS if g != f]) for f in A.FOLDS}
    out = {"what": __doc__.strip().splitlines()[0],
           "head": str(R.HEAD.relative_to(ROOT)), "rung": str(R.HELD["A"].relative_to(ROOT)),
           "no_refit": "arithmetic on the saved vectors and the pinned panel",
           "by_fold": {}}
    for f in A.FOLDS:
        p = pan[f]
        inc = Z[f] - H[f]
        r = p.rbase.to_numpy("float64") - H[f]
        v = 2 * r * inc - inc * inc
        n = len(p)
        y = p.y_true.to_numpy("float64")
        added = ((p.match_mode.fillna(-1).to_numpy("float64") == 4.0)
                 & p.usable.to_numpy(bool))
        guarded = G.mask(p.pbase.to_numpy("float64"), p.obs_taxi, cuts[f])
        moved = inc != 0
        rec = {
            "rows": n, "total_s2_per_row": float(v.mean()),
            "rows_moved_by_the_increment": int(moved.sum()),
            "rows_511_moves_but_255_does_not": int(((Z[f] != 0) & (H[f] == 0)).sum()),
            "rows_255_moves_but_511_does_not": int(((Z[f] == 0) & (H[f] != 0)).sum()),
            "mean_abs_update_255_s": float(np.abs(H[f][H[f] != 0]).mean()),
            "mean_abs_update_511_s": float(np.abs(Z[f][Z[f] != 0]).mean()),
            "splits": {
                "the_rows_the_ungated_pass_added": split(v, added, n),
                "every_other_row": split(v, ~added, n),
                "held_by_the_guard": split(v, guarded, n),
                "y_over_3600s": split(v, y > TAIL_Y_S, n),
                "the_bulk_under_3600s": split(v, y <= TAIL_Y_S, n),
            },
        }
        # by airport, worst first
        by_air = {}
        for a, idx in p.groupby("airport", observed=True).indices.items():
            by_air[str(a)] = {"rows": int(len(idx)),
                              "s2_sum": float(v[idx].sum()),
                              "s2_per_row_of_the_fold": float(v[idx].sum() / n)}
        rec["by_airport"] = dict(sorted(by_air.items(),
                                        key=lambda kv: kv[1]["s2_sum"]))
        # by decile of |increment| over the moved rows
        a_inc = np.abs(inc)
        if moved.any():
            q = pd.qcut(a_inc[moved], 10, labels=False, duplicates="drop")
            dec = {}
            for k in np.unique(q):
                sel = np.zeros(n, bool)
                sel[np.flatnonzero(moved)[q == k]] = True
                dec[f"decile_{int(k)+1}"] = {
                    "abs_increment_max_s": float(a_inc[sel].max()),
                    **split(v, sel, n)}
            rec["by_decile_of_abs_increment"] = dec
        out["by_fold"][f] = rec
        s = rec["splits"]
        print(f"\n{f}: total {rec['total_s2_per_row']:+8.2f} s2/row over {n} rows", flush=True)
        print(f"  added class   {s['the_rows_the_ungated_pass_added']['rows']:>6} rows  "
              f"{s['the_rows_the_ungated_pass_added']['s2_per_row_of_the_fold']:+8.3f} of "
              f"the fold  ({s['the_rows_the_ungated_pass_added']['s2_per_row_of_the_group']:+10.1f} "
              f"per added row)", flush=True)
        print(f"  every other   {s['every_other_row']['rows']:>6} rows  "
              f"{s['every_other_row']['s2_per_row_of_the_fold']:+8.3f}", flush=True)
        print(f"  y>3600 s      {s['y_over_3600s']['rows']:>6} rows  "
              f"{s['y_over_3600s']['s2_per_row_of_the_fold']:+8.3f}", flush=True)
        print(f"  bulk          {s['the_bulk_under_3600s']['rows']:>6} rows  "
              f"{s['the_bulk_under_3600s']['s2_per_row_of_the_fold']:+8.3f}", flush=True)
        print(f"  |d| mean 255 {rec['mean_abs_update_255_s']:.3f} s -> 511 "
              f"{rec['mean_abs_update_511_s']:.3f} s", flush=True)
        worst = list(rec["by_airport"].items())[:3]
        print("  worst airports:", ", ".join(
            f"{a} {r['s2_per_row_of_the_fold']:+.2f}" for a, r in worst), flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nwritten to", OUT.relative_to(ROOT), f"({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
