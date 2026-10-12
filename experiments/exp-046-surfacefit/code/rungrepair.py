"""The ONE pre-registered repair of the 511 rung: 511 leaves, stronger regularisation.

Ordered by the desk, 2026-09-09, after read A came back at **-0.06281 [-0.12236, -0.00916]**:

> before the capacity path is written as closed, ONE pre-registered repair (AGENTS: the
> repair attempt before the FAIL): 511 leaves with stronger regularisation -- double
> min_data_in_leaf and halve the learning rate with rounds refit -- guarded, cross-fitted,
> over v18's bytes on the shipped class. State the expected sign and the refutation band
> before the read. If it is not clear on every fold, the capacity path closes provisional
> with the cause (bulk over-confidence, section 12).

Against the predeclared LightGBM (`analysis/predeclared.json`): `min_child_samples` 100 ->
**200**, `learning_rate` 0.05 -> **0.025**, and the rounds refit to hold the total shrinkage,
`n_estimators` 300 -> **600**.  Everything else is the ladder's: 511 leaves, the same seed,
the same rotation, the same five day blocks, the same per-airport amplitude, the same floor,
the frozen p99.5 guard, LFPG held, the shipped ungated class at 1299 s.

**Pre-registered before the fit ran** (RESULTS section 35): expected sign **POSITIVE**, point
**+0.03 s** cross-fitted all-row over v18's bytes, band **[0.00, +0.10]**.  Refuted if the
cross-fitted read is **<= 0** or if `every_fold_interval_clear` is **False** -- and the desk's
rule is that a repair which is not clear on every fold closes the capacity path provisional.
july_loo decides it: the whole of read A's loss was there (-62.58 s2/row, clear below zero)
while december read +12.54 and january -6.13, both spanning zero.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/rungrepair.py
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
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402

NAME = "leaves_511x600_lr025_mcs200"
SETTING = {"num_leaves": 511, "min_child_samples": 200,
           "learning_rate": 0.025, "n_estimators": 600}
FREE = EXP / f"work/d-bundle-ungated1299-{NAME}-free.npz"
HELD = EXP / f"work/d-bundle-ungated1299-{NAME}-guard995.npz"
OUT = EXP / "analysis/rung-repair.json"
#: written into the record before the fit, so the read is checked and not described
PRE = {"expected_sign": "positive",
       "point_prediction_s": 0.03,
       "band_s": [0.0, 0.10],
       "refuted_if": ("the cross-fitted all-row read is <= 0, or "
                      "every_fold_interval_clear is False"),
       "the_fold_that_decides_it": "july_loo",
       "consequence_if_refuted": ("the capacity path closes provisional with the cause "
                                  "named: bulk over-confidence, section 12"),
       "registered_in": "RESULTS.md section 35, committed before this file was run"}


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
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}

    if FREE.exists():
        d_free = {f: v.astype("float64") for f, v in np.load(FREE).items()}
        refit = False
    else:
        setting_was, hold_was = CF.SETTING, CF.HOLD
        CF.SETTING, CF.HOLD = dict(SETTING), ()
        try:
            d_free = CF.fit_arm(pan)
        finally:
            CF.SETTING, CF.HOLD = setting_was, hold_was
        np.savez_compressed(FREE, **d_free)
        refit = True
    d_held = R.hold_lfpg(pan, d_free)
    np.savez_compressed(HELD, **d_held)

    rec = R.price(pan, d_held, H, f"the repair: {NAME}, shipped class, over v18's bytes")
    xf = R.crossfit(pan, d_held, H, f"{NAME} over H: the repair, cross-fitted")
    out = {"what": __doc__.strip().splitlines()[0],
           "setting": SETTING, "name": NAME,
           "against_the_predeclared": {"min_child_samples": "100 -> 200",
                                       "learning_rate": "0.05 -> 0.025",
                                       "n_estimators": "300 -> 600 (rounds refit)"},
           "pre_registration": PRE,
           "vector": str(HELD.relative_to(ROOT)), "refitted_this_run": refit,
           "price_over_v18_bytes": rec, "cross_fit": xf,
           "read_A_for_comparison": {
               "setting": "leaves_511x300_lr050 (the unregularised rung)",
               "all_row_gain_s": -0.06281,
               "by_fold_s2_per_row": {"july_loo": -62.58, "december": +12.54,
                                      "january": -6.13}},
           "seconds": round(time.time() - t0, 1)}
    g = xf["cross_fitted"]["all_row_gain_s"]
    out["verdict"] = {
        "cross_fitted_all_row_s": g,
        "every_fold_interval_clear": xf["cross_fitted"]["every_fold_interval_clear"],
        "inside_the_pre_registered_band": bool(PRE["band_s"][0] <= g <= PRE["band_s"][1]),
        "repair_holds": bool(g > 0 and xf["cross_fitted"]["every_fold_interval_clear"]),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(f"\n[repair] {NAME}  ({out['seconds']:.0f}s, refit {refit})", flush=True)
    print(R.row(rec), flush=True)
    print(R.fold_rows(rec), flush=True)
    print("    in-fold      %+8.5f [%+8.5f,%+8.5f]"
          % (xf["in_fold"]["all_row_gain_s"], *xf["in_fold"]["all_row_ci95"]), flush=True)
    print("    CROSS-FITTED %+8.5f [%+8.5f,%+8.5f]  clear %s"
          % (g, *xf["cross_fitted"]["all_row_ci95"],
             xf["cross_fitted"]["every_fold_interval_clear"]), flush=True)
    print("    premium      %+8.5f" % xf["selection_premium_s"], flush=True)
    for f, p in xf["out_of_fold_selection"].items():
        print(f"      scored {f:10s} -> {p['chosen']}", flush=True)
    print("\nVERDICT repair_holds:", out["verdict"]["repair_holds"],
          "| inside the band:", out["verdict"]["inside_the_pre_registered_band"], flush=True)
    print("written to", OUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
