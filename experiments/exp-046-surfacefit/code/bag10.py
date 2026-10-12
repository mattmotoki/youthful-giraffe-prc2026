"""K = 10: does doubling the bag pay what the shared-member algebra says it must?

Pre-registered in RESULTS section 40.2 BEFORE this file existed, and the prediction there
is far sharper than the K = 5 read's was, because this increment has a KNOWN SHAPE.  The
two bags share five members, so

    bag10 - bag5 = (T5 - S5) / 10,      Var = (5.Var(e) + 5.Var(e)) / 100 = Var(e)/10

while the expected gain is Var(e).(1/5 - 1/10) = Var(e)/10 as well.  With gain = 2A - B and
B = Var(e)/10 that forces **2A/B = 2.0** and an increment rms of sqrt(Var(e)/10) ~ 3.6 s on
july.  If 2A/B comes back near 1 instead, the members are NOT independent and section
37.2's whole bagging account is wrong in a way the K = 5 read could not see.

The return is saturating by construction -- 1 - 1/K -- so K = 20 would add only +0.012 more
and K = infinity only +0.025 beyond K = 10.  This run is worth its 16 minutes because it
TESTS THE MODEL, not because +0.025 is large, and section 40.2 says so before the number
arrives.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/bag10.py
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

FIVE = [1, 2, 3, 4, 5]
TEN = list(range(1, 11))
LEAVES = 255
HEAD5 = EXP / "work/d-bag5-255-free.npz"
OUT_VEC = EXP / "work/d-bag10-255-free.npz"
OUT = EXP / "analysis/bag-10.json"

PRE = {
    "registered_in": "RESULTS.md section 40.2, committed before this file was written",
    "over_v18_bytes_board_s": 0.225,
    "increment_over_the_k5_bag_board_s": 0.025,
    "increment_band_s": [0.015, 0.035],
    "every_fold_clear_expected": True,
    "ratio_2A_over_B_expected": 2.0,
    "rms_increment_expected_s": {"july_loo": 3.6},
    "falsifier": ("2A/B near 1 means the members are NOT independent and section 37.2's "
                  "bagging account is wrong in a way the K = 5 read could not see"),
    "var_e_from_member_spread_s2": {"july_loo": 131.8, "january": 171.3},
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

    members, census = {}, {}
    for seed in TEN:
        t1 = time.time()
        cache = EXP / f"work/d-bundle-placebo{seed}-255-free.npz"
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

    # the K = 5 head, rebuilt by the same function and checked against the served file
    bag5, spread5 = BAG.build_bag(members, pan_ship, FIVE)
    on_disk = {f: v.astype("float64") for f, v in np.load(HEAD5).items()}
    if not all(np.array_equal(bag5[f], on_disk[f]) for f in A.FOLDS):
        raise SystemExit("the rebuilt K=5 bag is not the vector on disk")
    print("the K=5 bag rebuilds bit for bit from its members", flush=True)

    bag10, spread10 = BAG.build_bag(members, pan_ship, TEN)
    np.savez_compressed(OUT_VEC, **bag10)

    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}
    rec = {"what": __doc__.strip().splitlines()[0], "seeds": TEN, "leaves": LEAVES,
           "pre_registration": PRE, "members": census,
           "member_spread_k10": spread10, "member_spread_k5": spread5,
           "head_rebuilds_bit_for_bit": True,
           "vector": str(OUT_VEC.relative_to(ROOT)), "prices": {}}

    for key, head, label in (
            ("over_the_k5_bag", bag5, "K=10 as an increment over the K=5 bag"),
            ("over_v18_bytes", H, "the K=10 bag over v18's bytes")):
        pr = R.price(pan_ship, bag10, head, label)
        dec = {}
        for f in A.FOLDS:
            r = pr["by_fold"][f]
            two_a, b = 2 * r["A_mean_rd"], r["B_mean_dd"]
            dec[f] = {"two_A_alignment": two_a, "B_churn": b,
                      "ratio_2A_over_B": (two_a / b) if b else None,
                      "rms_increment_s": float(np.sqrt(b))}
        pr["decomposition"] = dec
        rec["prices"][key] = pr

    inc = rec["prices"]["over_the_k5_bag"]
    g = inc["board_all_row"]["gain_s"]
    r2 = inc["decomposition"]["july_loo"]["ratio_2A_over_B"]
    rec["verdict"] = {
        "increment_board_s": g,
        "increment_ci95": inc["board_all_row"]["interval"]["ci95"],
        "inside_the_registered_band": bool(PRE["increment_band_s"][0] <= g
                                           <= PRE["increment_band_s"][1]),
        "every_fold_interval_clear": bool(inc["every_fold_interval_clear"]),
        "ratio_2A_over_B_july": r2,
        "ratio_is_near_2_not_1": bool(r2 is not None and r2 > 1.5),
        "the_model_survives": bool(
            PRE["increment_band_s"][0] <= g <= PRE["increment_band_s"][1]
            and inc["every_fold_interval_clear"] and r2 is not None and r2 > 1.5),
        "board_over_v18_bytes_s": rec["prices"]["over_v18_bytes"]["board_all_row"]["gain_s"],
    }
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    for key in ("over_the_k5_bag", "over_v18_bytes"):
        pr = rec["prices"][key]
        print()
        print(R.row(pr))
        print(R.fold_rows(pr))
        for f in A.FOLDS:
            d = pr["decomposition"][f]
            print(f"    {f:10s} 2A/B {d['ratio_2A_over_B']:5.2f}  rms |inc| "
                  f"{d['rms_increment_s']:6.2f} s", flush=True)
    v = rec["verdict"]
    print()
    print(f"  registered: +0.025 [{PRE['increment_band_s'][0]}, "
          f"{PRE['increment_band_s'][1]}], 2A/B ~ 2.0, rms ~ 3.6 s on july")
    print(f"  read:       {g:+.5f}, 2A/B july {r2:.2f}, "
          f"rms {inc['decomposition']['july_loo']['rms_increment_s']:.2f} s")
    print("  the model survives:", v["the_model_survives"])
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
