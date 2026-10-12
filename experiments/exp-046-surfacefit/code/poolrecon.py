"""Why the ranking block keeps a different number of chain rows than the pool did.

Section 27.3 recorded **2,493** ranking rows kept by gate (i)&(ii). That number was
measured on the pool `code/chaincheck.py` reconstructed with `block_probe.py` and the
check values `chaincheck.learn_maps` computed beside it. The v19 serve measures the same
gate on the block it actually WRITES FROM -- `block_ungated.build` per ranking month,
which computes the three check values in-build -- and keeps a different number.

Two instruments, one quantity. This joins them row by row and says where they differ, so
the served count is a reconciled number and not a second opinion.

**THE ANSWER** (measured below, not argued). Both instruments see the same 8,253 ungated
rows, and they agree on step 0 and on check (i) for every one of them: zero rows differ on
either. The whole disagreement is check (ii), and its cause is the DISTANCE METRIC used to
pick the nearest runway centroid:

  * `chaincheck.apply_checks` takes the argmin of a great-circle distance in km.
  * `block_ungated.build` takes the argmin of `(dlat**2 + dlon**2)` -- squared DEGREES, an
    unweighted lat/lon plane. At 50 deg N a degree of longitude is about 0.64 of a degree
    of latitude on the ground, so the plane metric over-weights longitude.

The candidate centroid sets are identical (both learned per month from the pass-1/2 matches
with the same 20-row minimum), so only segments near a boundary between two runways flip:
54 the pool keeps and the block refuses, 17 the other way, net -37.

**Nothing is mispriced by it.** The FOLD side (`code/chainfold.py`, section 28) was built
by the same `block_ungated`, so the class the arm was fitted and gated on and the class the
file writes are one class under one metric. What moves is the ranking-side yield: section
27.3's 2,493 was measured with the km metric, the served class is 2,456. The plane metric
is a defect of the probe and a refinement for a later reading; correcting it would change
the class and require the whole section 28 refit, which the desk's order ("exactly the 28.2
gate") rules out for this candidate.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/poolrecon.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

THR = 1299.0
GAP = 62.405
POOL = EXP / "work/chain-pairs.parquet"
#: the block instrument and where the reconciliation is written.  Both default to the
#: SHIPPED pair, so an argument-free run is exactly the run section 29.2 records; section
#: 33.4's acceptance test passes the corrected pool instead, and neither run touches the
#: other's file.
BLOCK = EXP / "work/ranking-ungated-pool-v19.parquet"
OUT = EXP / "analysis/pool-reconcile.json"
if len(sys.argv) > 1:
    BLOCK = Path(sys.argv[1]) if Path(sys.argv[1]).is_absolute() else ROOT / sys.argv[1]
if len(sys.argv) > 2:
    OUT = Path(sys.argv[2]) if Path(sys.argv[2]).is_absolute() else ROOT / sys.argv[2]


def main() -> None:
    if not BLOCK.exists():
        raise SystemExit(f"{BLOCK.name} does not exist; run the v19 serve with --ungated")
    blk = pd.read_parquet(BLOCK)
    P = pd.read_parquet(POOL)
    pool0 = P[~P.gated].copy()

    # the gate, on each instrument's own columns
    blk["keep"] = (np.isfinite(blk.chain_gap_s.to_numpy("float64"))
                   & (blk.chain_gap_s.to_numpy("float64") > GAP)
                   & (blk.chain_abs_dt_offblock.to_numpy("float64") <= THR)
                   & (blk.chain_runway_agrees.fillna(0).to_numpy("float64") == 1.0))
    pool0["keep"] = ((pool0.wheels_gap_s.to_numpy("float64") > GAP)
                     & (pool0.abs_dt_offblock.to_numpy("float64") <= THR)
                     & pool0.runway_agrees.to_numpy(bool))

    j = pool0[["MVT_ID_mvt", "month", "airport", "wheels_gap_s", "abs_dt_offblock",
               "runway_agrees", "keep"]].merge(
        blk[["MVT_ID_mvt", "chain_gap_s", "chain_abs_dt_offblock",
             "chain_runway_agrees", "match_mode", "keep"]],
        on="MVT_ID_mvt", how="outer", suffixes=("_pool", "_blk"), indicator=True)

    kp = j.keep_pool.fillna(False).to_numpy(bool)
    kb = j.keep_blk.fillna(False).to_numpy(bool)
    both, only_p, only_b = kp & kb, kp & ~kb, kb & ~kp

    def why(sub: pd.DataFrame) -> dict:
        """Which clause the block refused on, for rows the pool kept."""
        g = sub.chain_gap_s.to_numpy("float64")
        o = sub.chain_abs_dt_offblock.to_numpy("float64")
        r = sub.chain_runway_agrees.fillna(-1).to_numpy("float64")
        return {
            "not_in_the_block_pool_at_all": int((sub._merge == "left_only").sum()),
            "step_0_gap_not_over_the_cut": int(np.nansum((g <= GAP) & np.isfinite(g))),
            "step_0_gap_missing": int(np.sum(~np.isfinite(g)
                                             & (sub._merge != "left_only").to_numpy(bool))),
            "check_i_over_the_threshold": int(np.nansum((o > THR) & np.isfinite(o))),
            "check_i_missing": int(np.sum(~np.isfinite(o)
                                          & (sub._merge != "left_only").to_numpy(bool))),
            "check_ii_runway_disagrees": int(np.sum(r == 0.0)),
            "check_ii_runway_missing": int(np.sum(r < 0)),
        }

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "gate": {"step_0_gap_cut_s": GAP, "check_i_abs_dt_offblock_max_s": THR,
                 "check_ii_runway_agreement": True},
        "pool_instrument": {
            "file": str(POOL.relative_to(ROOT)),
            "built_by": "code/chaincheck.py with analysis/chain-check/block_probe.py",
            "ungated_rows": int(len(pool0)), "kept": int(kp.sum())},
        "block_instrument": {
            "file": str(BLOCK.relative_to(ROOT)),
            "built_by": ("code/serve.py --ungated, from "
                         "analysis/chain-check/block_ungated.py per ranking month"),
            "ungated_rows": int(len(blk)), "kept": int(kb.sum()),
            "kept_and_mode_4_in_the_served_block": int(
                (blk.match_mode.fillna(-1).to_numpy("float64") == 4.0).sum())},
        "agreement": {
            "kept_by_both": int(both.sum()),
            "kept_only_by_the_pool": int(only_p.sum()),
            "kept_only_by_the_block": int(only_b.sum()),
            "net": int(kb.sum() - kp.sum())},
        "why_the_block_refused_what_the_pool_kept": why(j[only_p]),
        "by_month": {},
    }
    for m in sorted(set(j.month.dropna())):
        sel = (j.month == m).to_numpy(bool)
        rec["by_month"][str(m)] = {
            "pool_kept": int((kp & sel).sum()), "block_kept": int((kb & sel).sum()),
            "only_pool": int((only_p & sel).sum()), "only_block": int((only_b & sel).sum())}
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(json.dumps(rec, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
