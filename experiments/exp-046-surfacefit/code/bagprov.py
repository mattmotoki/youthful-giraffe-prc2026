"""The bag's provenance: which rows each of the five perturbations moved, and how much
the five draws share.

`bagarm.py` keeps the per-member COUNTS but drops the row lists (`census` strips
`rows_moved`), and members 1-3 were loaded from `classplacebo.py`'s caches rather than
drawn again.  The desk asked for the five row sets and their overlap, so this recomputes
them -- deterministically, from `np.random.default_rng(seed)` and the same month order
`bagarm` iterates, with NO fit -- and checks the recomputation against the three draws
`analysis/class-placebo.json` recorded at the time.  If seed 1-3 reproduce, the same
construction gives 4 and 5 and the whole set is one instrument.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/bagprov.py
"""
from __future__ import annotations

import itertools
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
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
OUT = EXP / "analysis/bag-provenance.json"


def main() -> None:
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b

    draws = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cen = {}
        for m, b in blk.items():
            _, c = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
            cen[m] = {k: sorted(int(x) for x in v)
                      for k, v in c["rows_moved"].items()}
        draws[seed] = cen

    # the check: do 1-3 reproduce what class-placebo.json recorded at the time?
    rec_old = json.loads((EXP / "analysis/class-placebo.json").read_text())
    agree = {}
    for seed in (1, 2, 3):
        for m, c in draws[seed].items():
            was = rec_old["draws"][str(seed)]["census"][m]["rows_moved"]
            for side in ("forced_in", "forced_out"):
                agree[f"seed{seed} {m} {side}"] = (
                    sorted(int(x) for x in was[side]) == c[side])

    # pairwise overlap, per month and side, over all ten pairs
    over, worst = {}, 0
    for a, b in itertools.combinations(SEEDS, 2):
        for m in blk:
            for side in ("forced_in", "forced_out"):
                n = len(set(draws[a][m][side]) & set(draws[b][m][side]))
                over[f"{m} {side} draw{a}^draw{b}"] = n
                worst = max(worst, n)

    sizes = {m: {side: len(draws[1][m][side]) for side in ("forced_in", "forced_out")}
             for m in blk}
    rows_moved_per_draw = sum(v for m in sizes for v in sizes[m].values())

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "seeds": SEEDS,
        "moved_counts_per_draw": CP.MOVED,
        "rows_perturbed_per_draw": rows_moved_per_draw,
        "recomputation_reproduces_the_recorded_draws": all(agree.values()),
        "recomputation_check": agree,
        "draws_are_distinct": bool(
            len({tuple(draws[s][m][side]) for s in SEEDS
                 for m in blk for side in ("forced_in", "forced_out")})
            == len(SEEDS) * len(blk) * 2 - sum(
                1 for m in blk for side in ("forced_in", "forced_out")
                if len(draws[1][m][side]) == 0) * (len(SEEDS) - 1)),
        "largest_pairwise_overlap": worst,
        "pairwise_overlap": over,
        "rows": {str(s): draws[s] for s in SEEDS},
    }
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print("reproduces the recorded draws:", rec["recomputation_reproduces_the_recorded_draws"])
    print("rows perturbed per draw:", rows_moved_per_draw, CP.MOVED)
    print("largest pairwise overlap over all ten pairs:", worst)
    bad = [k for k, v in agree.items() if not v]
    if bad:
        print("DISAGREE:", bad)
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
