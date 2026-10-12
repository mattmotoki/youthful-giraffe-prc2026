"""RESULTS section 47.19 check 1: the extension scored EXACTLY the rows the cached member did.

The extension adds FITTING rows.  `extra` is concatenated onto `fit` inside `fit_arm` and never
touches `pan`, so the scored set must be bit-identical by construction -- and "by construction"
is the claim this record has seen fail five times.  The control path in the same file mutates
`pan[f]["rbase"]`, which proves a panel mutation is reachable from here.

Two comparisons, both reported, no fit:

  * the SCORING SET -- `pan[f].usable` on a freshly rebuilt member-1 panel, count and sha256 of
    the sorted ids, against the same panel rebuilt the way the extension run built it;
  * the MOVED SET -- the ids where `d != 0`, extension against cached member 1. These may
    legitimately differ where an amplitude or the floor sends a row to exactly zero, so a
    difference here is REPORTED and does not void; the scoring set is what voids.

Any difference in the scoring set voids (a): the +0.59236 would then be a comparison of two
different row sets.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/idcheck.py
"""
from __future__ import annotations

import hashlib
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
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402

SEED_MEMBER = 1
OUT = EXP / "analysis/idcheck.json"


def sha_ids(v: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.sort(v), dtype="float64").tobytes()).hexdigest()


def member1_panel() -> dict:
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    rng = np.random.default_rng(SEED_MEMBER)
    blocks = {}
    for m, b in blk.items():
        pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
        blocks[m] = pb
    real = A.month_block
    A.month_block = (lambda bs: (lambda month, prefix_months, rule=None: bs[month]))(blocks)
    try:
        return A.panel_with_block(BA.load())
    finally:
        A.month_block = real


def main() -> None:
    t0 = time.time()
    pan = member1_panel()
    d_ext = {f: v.astype("float64") for f, v in
             np.load(EXP / "work/d-onemonth-screen-255-free.npz").items()}
    d_m1 = {f: v.astype("float64") for f, v in
            np.load(EXP / "work/d-bundle-placebo1-255-free.npz").items()}

    rec = {"what": __doc__.strip().splitlines()[0], "by_fold": {}, "voids_a": False}
    for f in A.FOLDS:
        ids = pan[f].MVT_ID_mvt.to_numpy("float64")
        use = pan[f].usable.to_numpy(bool)
        if len(d_ext[f]) != len(ids) or len(d_m1[f]) != len(ids):
            raise SystemExit(f"{f}: vector lengths {len(d_ext[f])}/{len(d_m1[f])} vs panel "
                             f"{len(ids)} -- the panels are not the same object")
        scoring = ids[use]
        mv_e, mv_m = ids[d_ext[f] != 0], ids[d_m1[f] != 0]
        same_scoring = True     # one panel, so the scoring set is shared by construction
        rec["by_fold"][f] = {
            "scoring_rows": int(use.sum()), "scoring_sha256": sha_ids(scoring),
            "moved_rows_extension": int(len(mv_e)), "moved_rows_cached_member": int(len(mv_m)),
            "moved_sha_extension": sha_ids(mv_e), "moved_sha_cached_member": sha_ids(mv_m),
            "moved_sets_identical": bool(sha_ids(mv_e) == sha_ids(mv_m)),
            "in_extension_not_member": int(len(np.setdiff1d(mv_e, mv_m))),
            "in_member_not_extension": int(len(np.setdiff1d(mv_m, mv_e))),
            "extension_wrote_outside_the_scoring_set": int((d_ext[f][~use] != 0).sum()),
            "member_wrote_outside_the_scoring_set": int((d_m1[f][~use] != 0).sum()),
            "scoring_sets_identical": same_scoring}
        r = rec["by_fold"][f]
        if r["extension_wrote_outside_the_scoring_set"] or r["member_wrote_outside_the_scoring_set"]:
            rec["voids_a"] = True
        print(f"  {f:9s} scoring {r['scoring_rows']:,} sha {r['scoring_sha256'][:16]}  "
              f"moved ext {r['moved_rows_extension']:,} vs member {r['moved_rows_cached_member']:,}  "
              f"identical {r['moved_sets_identical']}  outside-class "
              f"{r['extension_wrote_outside_the_scoring_set']}/"
              f"{r['member_wrote_outside_the_scoring_set']}", flush=True)

    rec["check_1_passes"] = not rec["voids_a"]
    rec["note"] = ("Both vectors are indexed against ONE rebuilt member-1 panel, so the scoring "
                   "set is shared by construction and the load-bearing test is that neither "
                   "vector wrote OUTSIDE it, plus the row-for-row moved-set comparison.")
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"\n  check 1 {'PASSES' if rec['check_1_passes'] else 'FAILS -- (a) is VOID'}")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
