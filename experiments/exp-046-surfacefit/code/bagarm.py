"""The BAG: K perturbed fits averaged into one arm, which is what the diagnosis repairs.

Ordered by the desk, 2026-09-09, replacing the regularised-511 repair (withdrawn):

> If a fresh fit differs from H by ~20 s/row of chaos orthogonal to the residual, then the
> mean of K perturbed fits keeps the aligned part and shrinks the chaos energy by ~1/K, and
> H itself is one draw carrying its own chaos.  Pre-register, then run: K = 5 fits of the
> shipped 255 recipe on 5 distinct 33-row class perturbations, averaged into one arm, priced
> over v18's bytes per fold and against the placebo envelope.

**Pre-registered before the read**, in the desk's own band: **positive on the board by a
fraction of the chaos energy, +0.02 .. +0.06, if chaos dominates; inside the placebo envelope
if it does not.**  The envelope is `analysis/class-placebo.json`: july_loo [-81.25, -48.30],
december [-25.25, +13.56], january [-24.40, +30.09], board [-0.06344, -0.05339].

**Why this is the repair the diagnosis names.**  `gain = 2A - B` with A the alignment term
and B the churn energy.  Three unrelated changes -- a doubled model, a corrected class and an
arbitrary 33-row flip -- all read about -0.06 s because each produced an increment of ~18-21 s
per row with 2A within a quarter of B.  Averaging K draws leaves the aligned part alone and
divides the independent part of the churn energy by about K, so if the churn is what costs the
0.06 s, the bag recovers a fraction of it and the fold intervals should clear.  If the bag
lands inside the envelope, the churn is not independent across draws and the whole reading is
something else.

**Two vectors are written**, because the bag is also a candidate: `bag-held` carries the LFPG
hold, so it is comparable to H row for row and is the capacity/attribution read; `bag-free`
lifts it, which is the bag composed with the rider.

**What the bag is NOT.**  Its members are fitted on PERTURBED classes, so each is wrong on
about 33 of the 2,497 added rows.  That is a small bias accepted deliberately for the variance
it removes, and it is why the bagged vector is restricted to the SHIPPED class before it is
priced: a row the shipped gate refuses is not this lane's to move, whatever a draw thought.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/bagarm.py
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

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import guard as G                                                     # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402

K = 5
SEEDS = [1, 2, 3, 4, 5]
LEAVES = 255
HELD = EXP / "work/d-bag5-255-guard995.npz"
FREE = EXP / "work/d-bag5-255-free.npz"
OUT = EXP / "analysis/bag-arm.json"
#: carried into the record so a re-run does not lose it (the block was patched into
#: `bag-arm.json` by hand on 2026-09-09 and would otherwise vanish on the next write)
LABEL_CORRECTION = {
    "when": "2026-09-09, review round 8",
    "bag_free_label_was": "the bag (K=5) with the LFPG rider, over v18's bytes",
    "bag_held_label_was": "the bag (K=5), LFPG held, over v18's bytes",
    "why": ("'with the LFPG rider' was wrong: the five members are fitted UNHELD, so "
            "bag_free's LFPG rows carry the bag's own values. The mislabel propagated "
            "into RESULTS 36 and 37 and into a desk ruling before it was caught. The "
            "vector that really is 'the bag off LFPG plus the rider at LFPG' is a third "
            "one, work/d-bag5-255-rider.npz (code/bagcompose.py)"),
    "numbers_untouched": True,
}
PRE = {"band_s": [0.02, 0.06],
       "expectation": ("positive on the board by a fraction of the chaos energy if chaos "
                       "dominates; inside the placebo envelope if it does not"),
       "envelope": {"july_loo": [-81.2537549836145, -48.30237164543588],
                    "december": [-25.24543069800449, 13.560001365977618],
                    "january": [-24.403755104154925, 30.08664144134703],
                    "board": [-0.06344, -0.05339]},
       "registered_in": "RESULTS.md section 36, committed before this file was run"}


def build_bag(members: dict, pan_ship: dict, seeds: list) -> tuple[dict, dict]:
    """The average, and what the members disagree about.

    Extracted unchanged from `main` so that a bag at another SIZE -- `code/bag511.py`'s
    capacity read -- is built by this definition and not by a second copy of it.  A
    capacity comparison is only worth taking if both sides were averaged, restricted,
    floored and guarded identically; that is a property of having one function, not of
    two functions agreeing today.
    """
    bag_free, spread = {}, {}
    for f in A.FOLDS:
        stack = np.vstack([members[s][f] for s in seeds])
        mean = stack.mean(0)
        # **restricted to the SHIPPED class**: a row the shipped gate refuses is not this
        # lane's to move, whatever a perturbed draw thought
        usable = pan_ship[f].usable.to_numpy(bool)
        mean = np.where(usable, mean, 0.0)
        mean = A.apply_floor(pan_ship[f].pbase.to_numpy("float64"), mean)
        # the guard, at the SERVING cut of the shipped panel (the members carry their own
        # draw's cut; the vector that would ship has to carry the shipped one)
        cut = G.cut(pan_ship, [g for g in A.FOLDS if g != f])
        mean[G.mask(pan_ship[f].pbase.to_numpy("float64"), pan_ship[f].obs_taxi, cut)] = 0.0
        bag_free[f] = mean
        spread[f] = {
            "member_disagreement_rms_s": float(np.sqrt(np.mean(stack.var(0)))),
            "mean_abs_bag_s": float(np.abs(mean[mean != 0]).mean()) if (mean != 0).any() else 0.0,
            "rows_moved_by_the_bag": int((mean != 0).sum()),
            "guard_cut_s": float(cut),
        }
    return bag_free, spread


def main() -> None:
    t0 = time.time()
    if R.sha256(R.HEAD) != R.HEAD_SHA:
        raise SystemExit(f"{R.HEAD.name} is not the vector v18's bytes ship")
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    print("mode-3 asserted on the shipped builds", flush=True)
    # the SHIPPED panel: the class the bag is allowed to move, and the guard cut it serves
    pan_ship = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}

    members, census = {}, {}
    for seed in SEEDS:
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
        A.month_block = lambda month, prefix_months, rule=None: blocks[month]
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

    bag_free, spread = build_bag(members, pan_ship, SEEDS)
    bag_held = R.hold_lfpg(pan_ship, bag_free)
    np.savez_compressed(FREE, **bag_free)
    np.savez_compressed(HELD, **bag_held)

    out = {"what": __doc__.strip().splitlines()[0], "K": K, "seeds": SEEDS,
           "label_correction": LABEL_CORRECTION,
           "leaves": LEAVES, "pre_registration": PRE,
           "members": census, "member_spread": spread,
           "vectors": {"held": str(HELD.relative_to(ROOT)),
                       "free": str(FREE.relative_to(ROOT))},
           "prices": {}}
    for key, vec, label in (("bag_held", bag_held,
                             "the bag (K=5), LFPG held, over v18's bytes"),
                            ("bag_free", bag_free,
                             "the bag (K=5) with LFPG UNHELD, so LFPG carries the BAG's "
                             "own value and not the rider's (round 8: this label first "
                             "read 'with the LFPG rider', which was wrong and propagated "
                             "into RESULTS 36 and 37)")):
        rec = R.price(pan_ship, vec, H, label)
        dec = {}
        for f in A.FOLDS:
            r = rec["by_fold"][f]
            two_a, b = 2 * r["A_mean_rd"], r["B_mean_dd"]
            dec[f] = {"two_A_alignment": two_a, "B_churn": b,
                      "ratio_2A_over_B": (two_a / b) if b else None,
                      "rms_increment_s": float(np.sqrt(b))}
        rec["decomposition"] = dec
        rec["against_the_envelope"] = {
            f: {"s2_per_row": rec["by_fold"][f]["mse_gain_s2"],
                "placebo_range": PRE["envelope"][f],
                "outside_the_envelope": bool(
                    rec["by_fold"][f]["mse_gain_s2"] < PRE["envelope"][f][0]
                    or rec["by_fold"][f]["mse_gain_s2"] > PRE["envelope"][f][1])}
            for f in A.FOLDS}
        g = rec["board_all_row"]["gain_s"]
        rec["verdict"] = {
            "board_all_row_s": g,
            "inside_the_pre_registered_band": bool(PRE["band_s"][0] <= g <= PRE["band_s"][1]),
            "board_outside_the_placebo_envelope": bool(
                g < PRE["envelope"]["board"][0] or g > PRE["envelope"]["board"][1]),
            "every_fold_interval_clear": rec["every_fold_interval_clear"],
            "chaos_dominates": bool(g > 0 and rec["every_fold_interval_clear"]),
        }
        out["prices"][key] = rec
        print(f"\n[{key}] {label}", flush=True)
        print(R.row(rec), flush=True)
        print(R.fold_rows(rec), flush=True)
        for f, d in dec.items():
            rr = d["ratio_2A_over_B"]
            print(f"    {f:10s} 2A {d['two_A_alignment']:+9.2f}  B {d['B_churn']:9.2f}  "
                  f"2A/B {('%.2f' % rr) if rr else 'n/a':>5}  rms |inc| "
                  f"{d['rms_increment_s']:6.2f} s  outside envelope "
                  f"{rec['against_the_envelope'][f]['outside_the_envelope']}", flush=True)
        print(f"    band {PRE['band_s']}: inside = "
              f"{rec['verdict']['inside_the_pre_registered_band']}  | every-fold-clear "
              f"{rec['every_fold_interval_clear']}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nwritten to", OUT.relative_to(ROOT), f"({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
