"""Is july's `2A/B` shortfall a fold property, or is it inside the error on `A`?

Section 42.4 read `2A/B` across three instruments and found july consistently short --
1.52 / 1.19 / 1.40 against december and january's 1.64 / 1.36 / 2.16 and 1.63 / 1.67 / 2.05
-- and called it "the finding".  Section 44.3 then measured that `A` carries a real sampling
error across member draws and QUALIFIED section 42.4 as *unquantified*, naming this read as
the one that would settle it.  This is that read, for the K10-over-K5 instrument, where the
shared-member algebra fixes the expected ratio at exactly **2.0** and so gives the sharpest
test available.

**The decomposition.**  With `d_i` the pipeline applied to member `i` alone,

    bag10 - bag5 = (1/10) [ sum_{6..10} d_i - sum_{1..5} d_i ]

so `A = <r, d> = (1/10) [ sum_new a_i - sum_old a_i ]` with `a_i = <r, d_i>`, and the
spread of the ten `a_i` gives `A` its standard error.  The pipeline's floor and guard are
not linear, so the decomposition is CHECKED against the directly measured `A` rather than
assumed -- section 41.5 found `build_bag` effectively linear on these rows and this is where
that matters.

**Two caveats stated before the numbers, because both weaken the answer.**

  1. `r = y - pbase - bag5` is built from members 1-5, so `a_i` for `i <= 5` has its own
     member on both sides of the inner product and is biased downward relative to `a_i` for
     `i >= 6`.  This does not affect the FOLD-TO-FOLD comparison, which is the question, but
     it does mean the ten `a_i` are not ten draws from one distribution.
  2. The three folds are different row sets but come from the SAME ten member fits, so
     their three ratios are correlated.  Three folds agreeing is therefore weaker than
     three independent results, exactly as section 44.3 said of the 1-5 vs 6-10 gap.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/julyerr.py
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

FIVE = [1, 2, 3, 4, 5]
NEW = [6, 7, 8, 9, 10]
TEN = FIVE + NEW
OUT = EXP / "analysis/july-error.json"
#: what section 42.4 quoted, so the qualification lands on the same numbers
QUOTED = {"july_loo": 1.40, "december": 2.16, "january": 2.05}
ALGEBRA = 2.0


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

    mem = {s: {f: v.astype("float64") for f, v in
               np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
           for s in TEN}
    bag5 = BAG.build_bag(mem, pan, FIVE)[0]
    bag10 = BAG.build_bag(mem, pan, TEN)[0]
    singles = {s: BAG.build_bag(mem, pan, [s])[0] for s in TEN}

    rec = {"what": __doc__.strip().splitlines()[0], "quoted_in_42_4": QUOTED,
           "algebra_says": ALGEBRA, "by_fold": {}}
    for f in A.FOLDS:
        y = pan[f].y_true.to_numpy("float64")
        pb = pan[f].pbase.to_numpy("float64")
        r = y - pb - bag5[f]
        d = bag10[f] - bag5[f]
        A_direct = float(np.mean(r * d))
        B = float(np.mean(d * d))

        a = {s: float(np.mean(r * singles[s][f])) for s in TEN}
        A_dec = (sum(a[s] for s in NEW) - sum(a[s] for s in FIVE)) / 10.0
        d_dec = (sum(singles[s][f] for s in NEW) - sum(singles[s][f] for s in FIVE)) / 10.0
        lin_err = float(np.max(np.abs(d_dec - d)))

        vals = np.array([a[s] for s in TEN])
        sd = float(vals.std(ddof=1))
        se_A = sd / np.sqrt(10.0)                    # A is a difference of two 5-means /10
        ratio = 2 * A_direct / B
        se_ratio = 2 * se_A / B
        rec["by_fold"][f] = {
            "A_measured": A_direct, "A_from_the_decomposition": A_dec,
            "decomposition_agrees": bool(abs(A_dec - A_direct) < 1e-6 * max(1.0, abs(A_direct))),
            "largest_pipeline_nonlinearity_s": lin_err,
            "B": B, "gain_s2_per_row": 2 * A_direct - B,
            "per_member_alignment": a,
            "mean_members_1_5": float(np.mean([a[s] for s in FIVE])),
            "mean_members_6_10": float(np.mean([a[s] for s in NEW])),
            "sd_across_members": sd, "se_of_A": se_A,
            "ratio_2A_over_B": ratio, "se_of_the_ratio": se_ratio,
            "ratio_ci95": [ratio - 1.96 * se_ratio, ratio + 1.96 * se_ratio],
            "sigma_from_the_algebra_2_0": (ratio - ALGEBRA) / se_ratio if se_ratio else None,
        }
        v = rec["by_fold"][f]
        print(f"{f:10s} 2A/B {ratio:5.2f} +/- {se_ratio:4.2f}  "
              f"[{v['ratio_ci95'][0]:5.2f},{v['ratio_ci95'][1]:5.2f}]  "
              f"{v['sigma_from_the_algebra_2_0']:+5.2f} sigma from 2.0  | "
              f"A {A_direct:7.2f} +/- {se_A:5.2f}  B {B:6.2f}  "
              f"decomposition ok {v['decomposition_agrees']}", flush=True)

    # the question section 42.4 actually asked: is july DIFFERENT from the other two?
    j = rec["by_fold"]["july_loo"]
    comp = {}
    for f in ("december", "january"):
        o = rec["by_fold"][f]
        diff = o["ratio_2A_over_B"] - j["ratio_2A_over_B"]
        se = float(np.hypot(o["se_of_the_ratio"], j["se_of_the_ratio"]))
        comp[f] = {"difference": diff, "se_of_the_difference": se,
                   "sigma": diff / se if se else None}
        print(f"  {f} minus july: {diff:+.2f} +/- {se:.2f}  = {diff/se:+.2f} sigma")
    rec["july_versus_the_others"] = comp
    rec["verdict"] = {
        "july_is_below_the_algebra_beyond_error": bool(
            j["sigma_from_the_algebra_2_0"] is not None
            and j["sigma_from_the_algebra_2_0"] < -1.96),
        "july_differs_from_both_others_beyond_error": bool(
            all(abs(c["sigma"]) > 1.96 for c in comp.values())),
        "caveats": [
            "r is built from members 1-5, so a_i for i<=5 sits on both sides of the inner "
            "product; this biases the LEVEL of A, not the fold-to-fold comparison",
            "the three folds come from the same ten member fits, so their ratios are "
            "correlated and three folds agreeing is weaker than three independent results",
        ],
    }
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("\njuly below 2.0 beyond error:", rec["verdict"]["july_is_below_the_algebra_beyond_error"])
    print("july differs from BOTH others beyond error:",
          rec["verdict"]["july_differs_from_both_others_beyond_error"])
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
