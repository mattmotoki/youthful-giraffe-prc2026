"""The K = 10 capacity read: the bagged 511 against the bagged 255, both at K = 10.

Pre-registered in RESULTS section 40.3 BEFORE any 511 member 6-10 existed, and registered
BECAUSE section 41.5's split opened the path: on july the asymptote `a = 29.04` sits against
`2A = 113.05` while `b = 330.55` -- so 69 % of the K = 5 churn is the INDEPENDENT part, which
more members remove.  `B(K) = a + b/K` puts `B(10) = 62.10` and the july gain at **+50.96**
s2/row against +17.90 at K = 5, and july's interval is expected to CLEAR zero.

**Paired ladders.**  The same ten perturbations on both sides -- seeds 1-5 already cached on
each ladder, 6-10 fitted on each -- same shipped panel, same guard cut, same class
restriction, same floor, both sides through `bagarm.build_bag`.  The only difference between
the ladders is `num_leaves` 255 -> 511.

**The gate** is the standing cross-fitted `every_fold_interval_clear` rule for a fold-selected
rung (section 33.3), named here because section 41.3 recorded a registration defect for not
naming it.  The band is read on the CROSS-FITTED figure; the in-fold figure is recorded beside.

**The first refutation this run checks.**  `a + b/K` is an EXACT IDENTITY over any fixed set
of members -- section 41.5 measured its residuals at 1.1e-13 -- but that is a statement about
the five already drawn, not a promise that five NEW seeds behave the same.  So this recomputes
the identity's own moments over all ten members and compares them with the five:

    a = M11 = mean over ordered pairs i != j of <d_i, d_j>,   b = M2 - M11

If `a` over ten members has moved materially from 29.04 on july, the forecast is refuted even
though the algebra is not.  That is checked BEFORE the verdict is read.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/bag10cap.py
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
HEAD255_K5 = EXP / "work/d-bag5-255-free.npz"
HEAD255_K10 = EXP / "work/d-bag10-255-free.npz"
OUT_VEC = EXP / "work/d-bag10-511-free.npz"
OUT = EXP / "analysis/bag-10-capacity.json"

#: registered in RESULTS section 40.3, committed at e0eef8a before member 6 was fitted
PRE = {
    "registered_in": "RESULTS.md section 40.3, committed at e0eef8a before any 511 member "
                     "6-10 was fitted",
    "gate": "the standing cross-fitted every_fold_interval_clear rule for a fold-selected "
            "rung (section 33.3); the band is read on the CROSS-FITTED figure",
    "in_fold_gain_s2_per_row": {"july_loo": 50.96, "december": 106.97, "january": 122.16},
    "july_interval_expected_clear": True,
    "july_B_expected": 62.10,
    "july_A_expected": 56.52,
    "refuted_by": {
        "forecast": "B(10) on july materially above 62.10 -- say above 75 -- refutes a + b/K "
                    "as a FORECAST even though it is exact as an identity over the cached "
                    "five; five new seeds need not behave like five old ones",
        "orthogonality": "A at K = 10 differing from +56.52 on july by more than a couple of "
                         "percent means the chaos is not orthogonal to the residual and every "
                         "gain figure in section 41.5 is mis-stated",
        "signal_not_resolution": "july's interval still crossing zero at B(10) ~ 62 means the "
                                 "resolution was not the binding constraint, the signal was, "
                                 "and K cannot revive the path",
        "generalisation": "the cross-fitted reading still failing while in-fold clears means "
                          "the capacity step does not generalise across folds -- no serve",
    },
}


def moments(members: dict, other: dict, pan, seeds: list, fold: str) -> dict:
    """`M2`, `M11` and the identity's `a`, `b` for the increment over `seeds`, on one fold."""
    d = {}
    for s in seeds:
        d[s] = (BAG.build_bag(members, pan, [s])[0][fold]
                - BAG.build_bag(other, pan, [s])[0][fold])
    k = len(seeds)
    M2 = float(np.mean([np.mean(d[s] * d[s]) for s in seeds]))
    M11 = float(np.mean([np.mean(d[i] * d[j]) for i in seeds for j in seeds if i != j]))
    return {"K": k, "M2_mean_own_energy": M2, "M11_mean_pairwise": M11,
            "a_asymptote": M11, "b_over_K": M2 - M11,
            "B_at_this_K": M11 + (M2 - M11) / k}


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

    # --- the 255 ladder: all ten must already exist, written by bag10.py
    missing = [s for s in TEN
               if not (EXP / f"work/d-bundle-placebo{s}-255-free.npz").exists()]
    if missing:
        raise SystemExit(f"255 members {missing} are not cached -- run bag10.py first")
    m255 = {s: {f: v.astype("float64") for f, v in
                np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
            for s in TEN}
    bag255_k5 = BAG.build_bag(m255, pan_ship, FIVE)[0]
    bag255_k10, spread255 = BAG.build_bag(m255, pan_ship, TEN)
    for label, built, path in (("K=5", bag255_k5, HEAD255_K5),
                               ("K=10", bag255_k10, HEAD255_K10)):
        on_disk = {f: v.astype("float64") for f, v in np.load(path).items()}
        if not all(np.array_equal(built[f], on_disk[f]) for f in A.FOLDS):
            raise SystemExit(f"the rebuilt 255 {label} bag is not {path.name} on disk")
        print(f"the 255 {label} bag rebuilds bit for bit from its members", flush=True)

    # --- the 511 ladder, the SAME ten perturbations, same seeds, same draw order
    members, census = {}, {}
    for seed in TEN:
        t1 = time.time()
        cache = EXP / f"work/d-bundle-placebo{seed}-511-free.npz"
        if cache.exists():
            members[seed] = {f: v.astype("float64") for f, v in np.load(cache).items()}
            census[seed] = {"refitted_this_run": False}
            print(f"511 member {seed}: from cache", flush=True)
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
        CF.SETTING, CF.HOLD = {"num_leaves": 511}, ()
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
        print(f"511 member {seed}: fitted ({time.time()-t1:.0f}s)", flush=True)

    # the same five seeds must rebuild the K=5 511 bag section 41 priced
    bag511_k5 = BAG.build_bag(members, pan_ship, FIVE)[0]
    prev = EXP / "work/d-bag5-511-free.npz"
    if prev.exists():
        od = {f: v.astype("float64") for f, v in np.load(prev).items()}
        if not all(np.array_equal(bag511_k5[f], od[f]) for f in A.FOLDS):
            raise SystemExit("the rebuilt 511 K=5 bag is not the vector section 41 priced")
        print("the 511 K=5 bag rebuilds bit for bit -- the ladders are the same five",
              flush=True)

    bag511_k10, spread511 = BAG.build_bag(members, pan_ship, TEN)
    np.savez_compressed(OUT_VEC, **bag511_k10)

    # --- the first refutation, checked before the verdict is read
    ident = {f: {"over_the_first_five": moments(members, m255, pan_ship, FIVE, f),
                 "over_all_ten": moments(members, m255, pan_ship, TEN, f)}
             for f in A.FOLDS}
    for f in A.FOLDS:
        five, ten = ident[f]["over_the_first_five"], ident[f]["over_all_ten"]
        ident[f]["a_moved_by"] = ten["a_asymptote"] - five["a_asymptote"]
        ident[f]["a_moved_fraction"] = (ident[f]["a_moved_by"] / five["a_asymptote"]
                                        if five["a_asymptote"] else None)
        print(f"  {f:10s} a  {five['a_asymptote']:7.2f} (K=5 members) -> "
              f"{ten['a_asymptote']:7.2f} (all ten)   b {five['b_over_K']:7.2f} -> "
              f"{ten['b_over_K']:7.2f}   B(10) {ten['B_at_this_K']:7.2f}", flush=True)

    rec = {"what": __doc__.strip().splitlines()[0], "K": 10, "seeds": TEN, "leaves": 511,
           "pre_registration": PRE, "members": census,
           "member_spread_511": spread511, "member_spread_255": spread255,
           "the_identity_over_ten_members": ident,
           "head": {"vector": str(HEAD255_K10.relative_to(ROOT)),
                    "rebuilds_bit_for_bit_from_its_members": True},
           "vector": str(OUT_VEC.relative_to(ROOT))}

    pr = R.price(pan_ship, bag511_k10, bag255_k10, "the bagged 511 over the bagged 255, K=10")
    dec = {}
    for f in A.FOLDS:
        r = pr["by_fold"][f]
        two_a, b = 2 * r["A_mean_rd"], r["B_mean_dd"]
        dec[f] = {"two_A_alignment": two_a, "A_mean_rd": r["A_mean_rd"], "B_churn": b,
                  "ratio_2A_over_B": (two_a / b) if b else None,
                  "rms_increment_s": float(np.sqrt(b)),
                  "predicted_gain_s2_per_row": PRE["in_fold_gain_s2_per_row"][f],
                  "measured_gain_s2_per_row": r["mse_gain_s2"]}
    pr["decomposition"] = dec
    rec["in_fold"] = pr
    rec["cross_fitted"] = R.crossfit(pan_ship, bag511_k10, bag255_k10,
                                     "the bagged 511 against the bagged 255, K=10")

    cf = rec["cross_fitted"]["cross_fitted"]
    jd = dec["july_loo"]
    jf = pr["by_fold"]["july_loo"]
    rec["verdict_on_the_gate_reading"] = {
        "reading": "cross_fitted",
        "board_all_row_s": cf["all_row_gain_s"],
        "board_ci95": cf["all_row_ci95"],
        "every_fold_interval_clear": cf["every_fold_interval_clear"],
        "gate_passes": bool(cf["every_fold_interval_clear"]),
        "the_other_reading_recorded_beside_it": {
            "reading": "in_fold",
            "board_all_row_s": pr["board_all_row"]["gain_s"],
            "board_ci95": pr["board_all_row"]["interval"]["ci95"],
            "every_fold_interval_clear": bool(pr["every_fold_interval_clear"]),
        },
    }
    rec["scorecard"] = {
        "july_gain_predicted": PRE["in_fold_gain_s2_per_row"]["july_loo"],
        "july_gain_measured": jf["mse_gain_s2"],
        "july_B_predicted": PRE["july_B_expected"],
        "july_B_measured": jd["B_churn"],
        "july_B_above_75_forecast_refuted": bool(jd["B_churn"] > 75.0),
        "july_A_predicted": PRE["july_A_expected"],
        "july_A_measured": jd["A_mean_rd"],
        "july_A_moved_fraction": (jd["A_mean_rd"] - PRE["july_A_expected"])
                                 / PRE["july_A_expected"],
        "orthogonality_refuted": bool(abs(jd["A_mean_rd"] - PRE["july_A_expected"])
                                      > 0.02 * PRE["july_A_expected"]),
        "july_interval_clear_in_fold": bool(jf["ci95"][0] > 0),
        "signal_not_resolution_refuted": bool(jf["ci95"][0] <= 0 and jd["B_churn"] <= 75.0),
        "generalisation_refuted": bool(pr["every_fold_interval_clear"]
                                       and not cf["every_fold_interval_clear"]),
    }
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print()
    print(R.row(pr))
    print(R.fold_rows(pr))
    for f in A.FOLDS:
        d = dec[f]
        print(f"    {f:10s} 2A/B {d['ratio_2A_over_B']:5.2f}  B {d['B_churn']:7.2f}  "
              f"gain {d['measured_gain_s2_per_row']:+8.2f} (predicted "
              f"{d['predicted_gain_s2_per_row']:+8.2f})", flush=True)
    v, sc = rec["verdict_on_the_gate_reading"], rec["scorecard"]
    print()
    print(f"  cross-fitted (the gate's reading) {v['board_all_row_s']:+.5f} "
          f"[{v['board_ci95'][0]:+.5f},{v['board_ci95'][1]:+.5f}]  "
          f"every-fold-clear {v['every_fold_interval_clear']}")
    print(f"  in-fold beside it                 "
          f"{v['the_other_reading_recorded_beside_it']['board_all_row_s']:+.5f}")
    print(f"  GATE PASSES: {v['gate_passes']}")
    for k in ("july_B_above_75_forecast_refuted", "orthogonality_refuted",
              "signal_not_resolution_refuted", "generalisation_refuted"):
        print(f"    {k}: {sc[k]}")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
