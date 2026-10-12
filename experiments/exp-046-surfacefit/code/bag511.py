"""The capacity read on a proper footing: the bagged 511 against the bagged 255.

Pre-registered in RESULTS section 40.1 BEFORE this file existed.  Read A (section 34.2)
compared ONE 511 draw against ONE 255 draw, so its increment carried two independent chaos
vectors -- B = 191 s2/row on july, ~14 s per row, worth about -0.06 s of board, which is
exactly what it read and exactly the placebo envelope.  It was ruled VOID.  Averaging five
draws on each side divides that energy by about five, so the resolution floor should fall
from [-0.063, -0.053] to roughly [-0.013, -0.011] and a +0.13 capacity step, if one exists,
becomes ten times the floor instead of twice it.

The registered expectation is a NEGATIVE one: inside [-0.04, +0.04] on the board and NOT
clear on every fold, most likely failing on july.  The refutation band is stated both ways
in section 40.1, so the capacity path closes on a measurement or opens on one.

**Both sides are built by `bagarm.build_bag`**, the same function that built the vector v7
ships -- averaged, restricted to the shipped class, floored, and guarded at the shipped
panel's own cut, in that order.  Before anything is priced this rebuilds the 255 bag from
its cached members and asserts it equals `work/d-bag5-255-free.npz` BIT FOR BIT: if the
head this read is taken against is not the vector on disk, the comparison is not the one
section 40.1 registered.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/bag511.py
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
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402
import bagarm as BAG                                                  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
LEAVES = 511
HEAD255 = EXP / "work/d-bag5-255-free.npz"
OUT_VEC = EXP / "work/d-bag5-511-free.npz"
OUT = EXP / "analysis/bag-511.json"

PRE = {
    "registered_in": "RESULTS.md section 40.1, committed before this file was written",
    "expectation": ("inside [-0.04, +0.04] on the board and NOT clear on every fold, most "
                    "likely failing on july"),
    "band_s": [-0.04, 0.04],
    "opens_the_path_if": ">= +0.05 with every fold clear",
    "closes_the_path_if": "<= -0.05 with every fold clear",
    "unresolved_if": ("between, in which case even a five-fold bag does not resolve it and "
                      "the path stays a MAYBE at a stated resolution"),
    "expected_resolution_floor_board_s": [-0.013, -0.011],
    "member_disagreement_expected": ("LARGER than the 255 bag's 11.48 / 14.39 / 13.09 s -- "
                                     "more capacity, more variance to average"),
}


def main() -> None:
    t0 = time.time()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    print("mode-3 asserted on the shipped builds", flush=True)
    pan_ship = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    # --- the head: the 255 bag, rebuilt by the same function and checked against the file
    m255 = {s: {f: v.astype("float64") for f, v in
                np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
            for s in SEEDS}
    bag255, spread255 = BAG.build_bag(m255, pan_ship, SEEDS)
    on_disk = {f: v.astype("float64") for f, v in np.load(HEAD255).items()}
    same = {f: bool(np.array_equal(bag255[f], on_disk[f])) for f in A.FOLDS}
    if not all(same.values()):
        raise SystemExit(f"the rebuilt 255 bag is not the vector on disk: {same}")
    print("the 255 bag rebuilds bit for bit from its members", flush=True)

    # --- the 511 members, on the SAME five perturbations, same seeds, same draw order
    members, census = {}, {}
    for seed in SEEDS:
        t1 = time.time()
        cache = EXP / f"work/d-bundle-placebo{seed}-511-free.npz"
        if cache.exists():
            members[seed] = {f: v.astype("float64") for f, v in np.load(cache).items()}
            census[seed] = {"refitted_this_run": False}
            print(f"member {seed}: from cache", flush=True)
            continue
        rng = np.random.default_rng(seed)
        blocks, cen = {}, {}
        for m, b in blk.items():
            pb, c = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
            blocks[m], cen[m] = pb, c
        real = A.month_block
        A.month_block = (lambda blocks: (
            lambda month, prefix_months, rule=None: blocks[month]))(blocks)
        try:
            pan = A.panel_with_block(BA.load())
        finally:
            A.month_block = real
        setting_was, hold_was = CF.SETTING, CF.HOLD
        CF.SETTING, CF.HOLD = {"num_leaves": LEAVES}, ()
        try:
            d = CF.fit_arm(pan)
        finally:
            CF.SETTING, CF.HOLD = setting_was, hold_was
        np.savez_compressed(cache, **d)
        members[seed] = d
        census[seed] = {"refitted_this_run": True,
                        "census": {m: {k: v for k, v in c.items() if k != "rows_moved"}
                                   for m, c in cen.items()},
                        "seconds": round(time.time() - t1, 1)}
        print(f"member {seed}: fitted ({time.time()-t1:.0f}s)", flush=True)

    bag511, spread511 = BAG.build_bag(members, pan_ship, SEEDS)
    np.savez_compressed(OUT_VEC, **bag511)

    rec = {"what": __doc__.strip().splitlines()[0], "K": len(SEEDS), "seeds": SEEDS,
           "leaves": LEAVES, "pre_registration": PRE,
           "head": {"vector": str(HEAD255.relative_to(ROOT)),
                    "rebuilds_bit_for_bit_from_its_members": True,
                    "member_disagreement_rms_s": {
                        f: spread255[f]["member_disagreement_rms_s"] for f in A.FOLDS}},
           "members": census,
           "member_spread": spread511,
           "vector": str(OUT_VEC.relative_to(ROOT))}

    pr = R.price(pan_ship, bag511, bag255, "the bagged 511 over the bagged 255")
    dec = {}
    for f in A.FOLDS:
        r = pr["by_fold"][f]
        two_a, b = 2 * r["A_mean_rd"], r["B_mean_dd"]
        dec[f] = {"two_A_alignment": two_a, "B_churn": b,
                  "ratio_2A_over_B": (two_a / b) if b else None,
                  "rms_increment_s": float(np.sqrt(b))}
    pr["decomposition"] = dec
    rec["in_fold"] = pr
    rec["cross_fitted"] = R.crossfit(pan_ship, bag511, bag255,
                                     "the bagged 511 against the bagged 255")

    g = pr["board_all_row"]["gain_s"]
    band = pr["board_all_row"]["interval"]["ci95"]
    clear = bool(pr["every_fold_interval_clear"])
    rec["verdict"] = {
        "board_all_row_s": g, "board_ci95": band, "every_fold_interval_clear": clear,
        "inside_the_registered_band": bool(PRE["band_s"][0] <= g <= PRE["band_s"][1]),
        "opens_the_capacity_path": bool(g >= 0.05 and clear),
        "closes_the_capacity_path": bool(g <= -0.05 and clear),
        "the_registered_expectation_holds": bool(
            PRE["band_s"][0] <= g <= PRE["band_s"][1] and not clear),
        "member_disagreement_larger_than_at_255": {
            f: bool(spread511[f]["member_disagreement_rms_s"]
                    > spread255[f]["member_disagreement_rms_s"]) for f in A.FOLDS},
    }
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print()
    print(R.row(pr))
    print(R.fold_rows(pr))
    for f in A.FOLDS:
        d5, d2 = spread511[f], spread255[f]
        print(f"    {f:10s} 2A/B {dec[f]['ratio_2A_over_B']:5.2f}  rms |inc| "
              f"{dec[f]['rms_increment_s']:6.2f} s  |  member disagreement "
              f"{d5['member_disagreement_rms_s']:6.2f} s at 511 vs "
              f"{d2['member_disagreement_rms_s']:6.2f} s at 255", flush=True)
    print()
    print("  registered expectation (inside [-0.04,+0.04] and NOT every-fold-clear):",
          rec["verdict"]["the_registered_expectation_holds"])
    print("  opens the path:", rec["verdict"]["opens_the_capacity_path"],
          "| closes it:", rec["verdict"]["closes_the_capacity_path"])
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
