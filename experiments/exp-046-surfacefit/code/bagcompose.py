"""The vector v7 serves: the bagged 255 everywhere, the LFPG rider at LFPG.

A subtlety that would silently revert v6 if it were skipped.  The bag is an update over
the PRICE BASE, exactly like H -- `bagarm.py` prices it as `bag - H`, a REPLACEMENT of the
head, not an addition to it.  So a serve whose arm vector is `bag_held` writes
`pbase + 0 = v12` at LFPG and throws the rider away.  v7 is "v6's bytes plus the bagged
255 with LFPG held", which means the increment over v6 is `bag_held - H` (zero at LFPG,
the +0.19937 already read) and the VECTOR that produces it is

    arm = bag_held + (rider - hold_lfpg(rider))

-- the bag off LFPG, where `bag_held` is zero at LFPG; and the rider's own LFPG values at
LFPG, where the second term lives.  The two supports are disjoint and each side was
already floored and guarded on its own, so the sum needs neither again.

Three things are asserted, not described:

  (i)  `hold_lfpg(rider) == H` to the bit -- the section 34.6 provenance check re-run from
       this side, which is what makes the second term "the rider and nothing else";
  (ii) the two supports are disjoint, so nothing is added twice;
  (iii) `arm - rider == bag_held - H` to the bit -- the identity the desk asked to be
       stated in the gate record, checked instead of asserted.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/bagcompose.py
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

HEAD = EXP / "work/d-bundle-ungated1299-guard995.npz"
RIDER = EXP / "work/d-rider-lfpg-on-v19.npz"
BAG_HELD = EXP / "work/d-bag5-255-guard995.npz"
OUT = EXP / "work/d-bag5-255-rider.npz"
REC = EXP / "analysis/bag-compose.json"


def load(p: Path) -> dict:
    return {f: v.astype("float64") for f, v in np.load(p).items()}


def main() -> None:
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    H, rider, bag = load(HEAD), load(RIDER), load(BAG_HELD)
    rider_held = R.hold_lfpg(pan, rider)

    rec = {"what": __doc__.strip().splitlines()[0], "by_fold": {}}
    arm = {}
    for f in A.FOLDS:
        # (i) the rider with LFPG put back under the hold IS the head
        n_bad = int(np.count_nonzero(rider_held[f] != H[f]))
        # the LFPG-only part of the rider
        rho = rider[f] - rider_held[f]
        # (ii) disjoint supports
        both = int(np.count_nonzero((rho != 0) & (bag[f] != 0)))
        arm[f] = bag[f] + rho
        # (iii) the identity the gate record states
        lhs, rhs = arm[f] - rider[f], bag[f] - H[f]
        rec["by_fold"][f] = {
            "rows": int(len(H[f])),
            "rider_under_the_hold_differs_from_the_head_on": n_bad,
            "rows_the_rider_moves_at_lfpg": int(np.count_nonzero(rho)),
            "rows_the_bag_moves": int(np.count_nonzero(bag[f])),
            "rows_in_both_supports": both,
            "rows_the_arm_moves": int(np.count_nonzero(arm[f])),
            "identity_max_abs_difference_s": float(np.max(np.abs(lhs - rhs))),
            "identity_holds_to_the_bit": bool(np.array_equal(lhs, rhs)),
        }
        if n_bad:
            raise SystemExit(f"{f}: the rider under the hold is not the head on {n_bad} rows")
        if both:
            raise SystemExit(f"{f}: {both} rows are moved by BOTH the bag and the rider; "
                             "the supports must be disjoint or the sum double-counts")
        if not np.array_equal(lhs, rhs):
            raise SystemExit(f"{f}: arm - rider is not bag_held - head")

    np.savez_compressed(OUT, **arm)
    rec["written_to"] = str(OUT.relative_to(ROOT))
    rec["all_three_assertions_hold"] = True
    REC.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
