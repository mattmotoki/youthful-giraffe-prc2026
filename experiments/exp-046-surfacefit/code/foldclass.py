"""Do the FOLD members 6-10 share the ranking side's cache-key defect?  (desk, 2026-09-09)

The ranking member cache keys on seed and leaves only, so members fitted under one gate
setting were silently averaged with members fitted under another (RESULTS section 45.2).
`d-bundle-placebo{seed}-{leaves}-free.npz` keys the same way, so the question is fair.

**But the fold side fails differently, and that matters for what to test.**  `bagarm.build_bag`
applies `np.where(usable, mean, 0.0)` itself, so a member carrying rows outside the shipped
class is silently CLEANED rather than caught -- the out-of-class symptom the ranking side
showed cannot appear here.  What a wrong-panel member would do instead is carry **different
values on the same rows**, which averages in invisibly.  So this checks three things:

  1. the out-of-class count per member, seed and file named;
  2. the support each member moves;
  3. the panel identity itself, which is the check that actually binds -- `build_bag` reads
     `usable`, `pbase` and the guard cut from the panel, so if the panel this script builds
     were not the panel the members were fitted against, the K = 5 bag rebuilt from members
     1-5 could not equal the served vector BYTE FOR BYTE.  That equality is asserted here
     for both ladders, and it is what makes 1 and 2 meaningful rather than circular.

**My first two criteria were wrong and the run that used them is superseded.**  I wrote them
as "every member zero outside the shipped class" and "every member moves the same rows as
member 1", and BOTH are false of every member ever fitted on this lane, including the five
that built the served v7.  A member is fitted on a PERTURBED panel, so (a) it moves a few rows
the shipped class refuses -- 8 on july, 26 on december, 0 on january, and `build_bag`'s
`np.where(usable, mean, 0.0)` exists precisely to clean them -- and (b) it moves a slightly
different row set from its siblings, which is what a perturbation IS.  Testing for absolute
purity would condemn v7 itself.

The question the desk actually asked is whether members 6-10 were fitted on a DIFFERENT panel
from members 1-5.  That is a question about CONSISTENCY BETWEEN THE TWO BATCHES, so the
criteria below compare them instead of measuring them against an impossible standard.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/foldclass.py
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
import bagarm as BAG                                                  # noqa: E402

TEN = list(range(1, 11))
FIVE = [1, 2, 3, 4, 5]
OUT = EXP / "analysis/fold-class.json"


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
    print("mode-3 asserted on the shipped builds", flush=True)

    rec = {"what": __doc__.strip().splitlines()[0], "by_ladder": {}, "failures": []}
    for leaves, served in ((255, "d-bag5-255-free.npz"), (511, "d-bag5-511-free.npz")):
        mem = {}
        for s in TEN:
            p = EXP / f"work/d-bundle-placebo{s}-{leaves}-free.npz"
            if not p.exists():
                rec["failures"].append(f"{p.name} missing")
                continue
            mem[s] = {f: v.astype("float64") for f, v in np.load(p).items()}

        # (3) the binding check FIRST -- if this fails, 1 and 2 mean nothing
        rebuilt = BAG.build_bag(mem, pan, FIVE)[0]
        on_disk = {f: v.astype("float64") for f, v in np.load(EXP / f"work/{served}").items()}
        bitwise = {f: bool(np.array_equal(rebuilt[f], on_disk[f])) for f in A.FOLDS}
        if not all(bitwise.values()):
            rec["failures"].append(f"{leaves}: the K=5 bag does not rebuild bit for bit "
                                   f"from members 1-5 -- the panel is not the one they "
                                   f"were fitted against: {bitwise}")

        lad = {"served_vector": served, "k5_rebuilds_bit_for_bit": bitwise, "members": {}}
        for f in A.FOLDS:
            usable = pan[f].usable.to_numpy(bool)
            support0 = None
            for s in sorted(mem):
                d = mem[s][f]
                if len(d) != len(usable):
                    rec["failures"].append(
                        f"{leaves} member {s} fold {f}: {len(d)} rows, panel has "
                        f"{len(usable)}")
                    continue
                out = int(np.count_nonzero(d[~usable]))
                sup = d != 0
                if support0 is None:
                    support0 = sup
                    diff = 0
                else:
                    diff = int(np.count_nonzero(sup ^ support0))
                lad["members"].setdefault(str(s), {})[f] = {
                    "rows": int(len(d)),
                    "nonzero": int(sup.sum()),
                    "nonzero_outside_the_shipped_class": out,
                    "rows_where_support_differs_from_member_1": diff,
                }
            # the batch comparison: 1-5 built the served vectors, 6-10 are the new fits
            ms = lad["members"]
            old_b = [ms[str(s)][f]["nonzero_outside_the_shipped_class"] for s in FIVE]
            new_b = [ms[str(s)][f]["nonzero_outside_the_shipped_class"] for s in TEN[5:]
                     if str(s) in ms]
            sup_old = [ms[str(s)][f]["rows_where_support_differs_from_member_1"]
                       for s in FIVE[1:]]
            sup_new = [ms[str(s)][f]["rows_where_support_differs_from_member_1"]
                       for s in TEN[5:] if str(s) in ms]
            lad.setdefault("batch_comparison", {})[f] = {
                "out_of_class_members_1_5": old_b,
                "out_of_class_members_6_10": new_b,
                "out_of_class_identical_across_batches": bool(set(old_b) == set(new_b)
                                                              and len(set(old_b)) == 1),
                "support_spread_members_2_5": sup_old,
                "support_spread_members_6_10": sup_new,
                "support_spread_overlaps": bool(
                    not sup_new or not sup_old
                    or (min(sup_new) >= min(sup_old) - 12
                        and max(sup_new) <= max(sup_old) + 12)),
            }
            bc = lad["batch_comparison"][f]
            if not bc["out_of_class_identical_across_batches"]:
                rec["failures"].append(
                    f"{leaves} ({f}): out-of-class counts differ between the batches -- "
                    f"1-5 {old_b}, 6-10 {new_b}. Members 6-10 were fitted on a different "
                    f"panel; the {leaves} bag is a MIXED vector and must be refitted.")
            if not bc["support_spread_overlaps"]:
                rec["failures"].append(
                    f"{leaves} ({f}): members 6-10 disagree with member 1 by {sup_new}, "
                    f"outside the 2-5 range {sup_old} -- suspect a different panel.")
        rec["by_ladder"][str(leaves)] = lad
        print(f"  {leaves}: K=5 rebuilds bit for bit {all(bitwise.values())}; ten members "
              f"checked on three folds", flush=True)

    rec["all_twenty_members_pass"] = not rec["failures"]
    rec["what_the_fold_cache_keys_on"] = (
        "work/d-bundle-placebo{seed}-{leaves}-free.npz -- SEED and LEAVES only, exactly as "
        "the ranking cache did. The key carries no gate setting, no matcher pin and no "
        "panel identity, so the defect class of section 45.2 is present here BY NAME. What "
        "stops it biting is that every fold script builds its panel through the same "
        "CF.build_month + R.assert_mode3 + CF.panel_for path and then asserts the K = 5 bag "
        "rebuilds bit for bit before pricing -- an outcome check, not a key.")
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    if rec["failures"]:
        print("\nFAILURES:")
        for x in rec["failures"]:
            print("  -", x)
    else:
        print("\nall twenty fold members (255 x10, 511 x10): members 6-10 match members "
              "1-5 batch for batch on every fold, and the K=5 bag rebuilds BIT FOR BIT "
              "from members 1-5 on both ladders. No fold-side cache contamination.")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
