"""How B(K) falls, and therefore whether any K makes july clear.

The desk, 2026-09-09, correcting my diagnosis and correcting it rightly.  I fitted
B(K) = 71 + 120/K and called the 71 "shared chaos no K removes".  That lumps two different
things.  Writing the per-draw increment as `d_k = mu + e_k`, with `mu` the mean increment
and `e_k` the deviation with variance V and pairwise covariance C:

    B(K) = ||mu||^2  +  C.(K-1)/K  +  V/K        ->   B(inf) = ||mu||^2 + C

`||mu||^2` is the MEAN INCREMENT'S OWN ENERGY -- signal, already paid for by the `2A` term
and not a floor at all.  Only `C` is the floor.  My `c = 71` is `||mu||^2 + C` and my
`v = 120` is `V - C`.

**And here is what cannot be done, said before the numbers rather than after.**  With five
members that are PAIRED across the two ladders, `||mu||^2` and `C` are **not separately
identifiable**.  Two measurable quantities exist -- `B(K)` for each K, and the across-member
sample variance, whose expectation is `V - C` -- and three unknowns.  The deviations sum to
zero by construction, so the average pairwise product of `(d_j - dbar)(d_k - dbar)` is
exactly `-W/(K-1)` whatever the truth is, and carries no information.  Reporting a number
labelled "the shared covariance" would therefore be reporting an arbitrary split.

**The decision does not need them separated.**  What decides whether more members can make
july clear is the ASYMPTOTIC GAIN,

    gain(K) = 2A - B(K)      ->      gain(inf) = 2A - (||mu||^2 + C) = 2A - a

and `a` is measurable however it splits.  `A = <mu, r>` is predicted stable in K, because
the chaos is orthogonal to the residual -- and that is CHECKED here, not assumed, by reading
A at every subset size.

So this measures, per fold and with no new fits, by rebuilding the bag on all 31 non-empty
subsets of the five members through `bagarm.build_bag` (the same function that built the
served vector, so the pipeline's floor, guard and class restriction are inside every point):

  * `B(k)` and `A(k)` for every one of the 31 subsets, and averaged over the C(5,k) of
    each size (both are written, so the averages are re-derivable from the rows);
  * the least-squares fit `B(k) = a + b/k`;
  * `gain(inf) = 2A - a` and `gain(10) = 2A - (a + b/10)`.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/splitread.py
"""
from __future__ import annotations

import itertools
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

SEEDS = [1, 2, 3, 4, 5]
OUT = EXP / "analysis/split-read.json"

#: registered BEFORE the run (desk, 2026-09-09), so the read cannot be fitted to it
PRE = {
    "the_desks_prediction": {
        "a_is_mostly_the_mean_energy_not_the_floor": True,
        "july_A_from_the_K5_read": 56.5241431942054,
        "july_two_A": 113.0482863884108,
        "july_a_expected": 71.0,
        "july_gain_infinity_expected_s2_per_row": 42.0,
        "july_gain_at_K10_expected_s2_per_row": 30.0,
        "july_interval_at_K10_expected": [-3.0, 63.0],
    },
    "mine": {
        "A_is_stable_in_K": ("within a few percent across subset sizes, because the chaos "
                             "is orthogonal to the residual; CHECKED here, not assumed"),
        "what_cannot_be_measured": ("||mu||^2 and C separately, from paired members -- two "
                                    "equations, three unknowns; the decision needs only a"),
    },
    "the_decision_rule": {
        "path_open_if": ("gain at K=10 on july is comfortably positive, i.e. a is mostly "
                         "the mean increment's own energy -- then five more 511 fits "
                         "(~26 min) are worth running"),
        "path_is_a_maybe_if": ("gain(inf) on july is at or below zero, i.e. the floor C "
                               "carries the a -- then no K resolves it and that is the "
                               "revival condition"),
    },
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
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    def load(size: int) -> dict:
        return {s: {f: v.astype("float64") for f, v in
                    np.load(EXP / f"work/d-bundle-placebo{s}-{size}-free.npz").items()}
                for s in SEEDS}
    m255, m511 = load(255), load(511)

    # the residual each increment is read against: r = y - (pbase + bag255), the head
    bag255_all, _ = BAG.build_bag(m255, pan, SEEDS)

    rec = {"what": __doc__.strip().splitlines()[0], "pre_registration": PRE,
           "seeds": SEEDS, "by_fold": {}}
    for f in A.FOLDS:
        r_head = (pan[f].y_true.to_numpy("float64")
                  - pan[f].pbase.to_numpy("float64") - bag255_all[f])
        per_k, each, singles, ladders = {}, [], {}, {}
        for k in range(1, len(SEEDS) + 1):
            Bs, As = [], []
            for sub in itertools.combinations(SEEDS, k):
                b5 = BAG.build_bag(m255, pan, list(sub))[0][f]
                b1 = BAG.build_bag(m511, pan, list(sub))[0][f]
                d = b1 - b5
                Bs.append(float(np.mean(d * d)))
                As.append(float(np.mean(r_head * d)))
                # every one of the 31 subsets, so the size-averages above are re-derivable
                each.append({"members": list(sub), "K": k, "B_mean_dd": Bs[-1],
                             "A_mean_rd": As[-1],
                             "gain_s2_per_row": 2 * As[-1] - Bs[-1]})
                if k == 1:
                    singles[sub[0]] = d
                    ladders[sub[0]] = (b5, b1)
            per_k[k] = {"subsets": len(Bs), "B_mean_dd": float(np.mean(Bs)),
                        "A_mean_rd": float(np.mean(As)),
                        "gain_s2_per_row": float(2 * np.mean(As) - np.mean(Bs))}
        assert len(each) == 31, len(each)
        # least squares B(k) = a + b/k
        ks = np.array(sorted(per_k), dtype="float64")
        Bv = np.array([per_k[int(k)]["B_mean_dd"] for k in ks])
        M = np.vstack([np.ones_like(ks), 1.0 / ks]).T
        a, bq = np.linalg.lstsq(M, Bv, rcond=None)[0]
        # Queue, 2026-09-09: can the residuals say whether the 0.92 is member correlation
        # or a conversion bias?  Averaging ||mean_S d||^2 over ALL subsets of size k is
        # exactly  M11 + (M2 - M11)/k  for any five fixed vectors -- M2 the mean own
        # energy, M11 the mean inner product over ordered pairs i != j.  So the a + b/K
        # form is an ALGEBRAIC IDENTITY here, not a model being tested, and its residuals
        # are zero whatever the truth is.  They can only be non-zero if build_bag's floor,
        # guard or class restriction makes the average non-quadratic.  Both are measured.
        resid = [float(Bv[i] - (a + bq / ks[i])) for i in range(len(ks))]
        dd = [singles[s_] for s_ in SEEDS]
        M2 = float(np.mean([np.mean(x * x) for x in dd]))
        M11 = float(np.mean([np.mean(dd[i] * dd[j])
                             for i in range(len(dd)) for j in range(len(dd)) if i != j]))
        # Queue's 0.92, measured rather than left open.  b = M2 - M11 is algebraically
        # S_d / (1 - 1/5) with S_d the mean squared member deviation of d = b511 - b255,
        # which is queue's own denominator.  So the ratio queue sees is exactly
        # S_d / (s511^2 + s255^2) = 1 - 2.cov / (s511^2 + s255^2): it measures the
        # CROSS-LADDER covariance of the paired members, which the paired design creates
        # on purpose (both ladders use the same 49-row draw per seed).
        u = np.stack([ladders[s_][0] for s_ in SEEDS]); u = u - u.mean(0)
        w = np.stack([ladders[s_][1] for s_ in SEEDS]); w = w - w.mean(0)
        s255sq = float(np.mean(u * u)); s511sq = float(np.mean(w * w))
        cov = float(np.mean(u * w))
        Sd = float(np.mean((w - u) ** 2))
        Ainf = per_k[len(SEEDS)]["A_mean_rd"]
        A_spread = (max(v["A_mean_rd"] for v in per_k.values())
                    - min(v["A_mean_rd"] for v in per_k.values()))
        rec["by_fold"][f] = {
            "all_31_subsets": each,
            "by_subset_size": per_k,
            "fit_B_of_K": {"a_asymptote": float(a), "b_over_K": float(bq),
                           "means": "B(K) = a + b/K, a = ||mu||^2 + C, b = V - C"},
            "the_fit_is_an_identity": {
                "why": ("averaging ||mean_S d||^2 over all C(5,k) subsets equals "
                        "M11 + (M2 - M11)/k exactly, for any five fixed vectors, so a + b/K "
                        "is algebra and not a model under test; residuals can only measure "
                        "build_bag's own non-linearity (floor, guard, class restriction)"),
                "residual_s2_per_row_by_K": resid,
                "largest_absolute_residual": float(max(abs(x) for x in resid)),
                "mean_own_energy_M2": M2,
                "mean_pairwise_inner_product_M11": M11,
                "a_equals_M11": bool(abs(a - M11) < 1e-6 * max(1.0, abs(M11))),
                "b_equals_M2_minus_M11": bool(abs(bq - (M2 - M11))
                                              < 1e-6 * max(1.0, abs(M2 - M11))),
                "for_queue": ("the residuals ARE zero and a == M11 exactly, so they cannot "
                              "distinguish queue's two hypotheses -- and they do not need to: "
                              "a is measured directly as the mean pairwise inner product, not "
                              "extrapolated"),
            },
            "the_0_92_explained": {
                "what_queue_saw": ("b measured / (s511^2 + s255^2)/(1 - 1/5) at 0.924 / 0.918 "
                                   "/ 0.909, uniform enough to look systematic"),
                "the_answer": ("neither a member correlation nor a conversion bias: b = M2 - "
                               "M11 IS S_d/(1 - 1/5) exactly, so the ratio is "
                               "S_d/(s511^2 + s255^2) = 1 - 2.cov/(s511^2 + s255^2), and it "
                               "measures the CROSS-LADDER covariance the paired design "
                               "creates -- both ladders fit the same 49-row draw per seed"),
                "member_spread_255_s2_per_row": s255sq,
                "member_spread_511_s2_per_row": s511sq,
                "cross_ladder_covariance": cov,
                "cross_ladder_correlation": cov / float(np.sqrt(s255sq * s511sq)),
                "S_d_mean_squared_member_deviation": Sd,
                "ratio_queue_computed": Sd / (s255sq + s511sq),
                "b_equals_S_d_over_one_minus_one_fifth": bool(
                    abs(bq - Sd / (1 - 1 / len(SEEDS))) < 1e-6 * max(1.0, abs(bq))),
                "touches_a": ("no -- a is M11, measured directly; the K = 10 prediction "
                              "stands whatever this ratio is"),
            },
            "A_at_K5": Ainf,
            "A_spread_across_subset_sizes": float(A_spread),
            "A_is_stable_in_K": bool(abs(A_spread) < 0.05 * abs(Ainf) if Ainf else True),
            "gain_infinity_s2_per_row": float(2 * Ainf - a),
            "gain_at_K10_s2_per_row": float(2 * Ainf - (a + bq / 10.0)),
            "gain_at_K5_measured": per_k[5]["gain_s2_per_row"],
        }

    j = rec["by_fold"]["july_loo"]
    rec["verdict"] = {
        "july_a_asymptote": j["fit_B_of_K"]["a_asymptote"],
        "july_gain_infinity": j["gain_infinity_s2_per_row"],
        "july_gain_at_K10": j["gain_at_K10_s2_per_row"],
        "july_A_stable_in_K": j["A_is_stable_in_K"],
        "more_members_can_help_july": bool(j["gain_infinity_s2_per_row"] > 0),
        "identifiability": ("||mu||^2 and C are NOT separately identifiable from paired "
                            "members; a = ||mu||^2 + C is, and the decision needs only a"),
    }
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    for f in A.FOLDS:
        v = rec["by_fold"][f]
        print(f"\n{f}")
        for k in sorted(v["by_subset_size"]):
            u = v["by_subset_size"][k]
            print(f"  K={k} ({u['subsets']:2d} subsets)  B {u['B_mean_dd']:8.2f}  "
                  f"A {u['A_mean_rd']:8.2f}  gain {u['gain_s2_per_row']:+8.2f}")
        print(f"  fit: a {v['fit_B_of_K']['a_asymptote']:7.2f}  b "
              f"{v['fit_B_of_K']['b_over_K']:7.2f}  | A stable: {v['A_is_stable_in_K']} "
              f"(spread {v['A_spread_across_subset_sizes']:.2f})")
        print(f"  gain at K=10 {v['gain_at_K10_s2_per_row']:+7.2f}  "
              f"gain at K=inf {v['gain_infinity_s2_per_row']:+7.2f}")
    print("\nwritten to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
