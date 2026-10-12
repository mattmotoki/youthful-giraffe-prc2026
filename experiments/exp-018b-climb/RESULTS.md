# exp-018b — results

Every number here is produced by code in `code/`, on folds defined in
`prc.data.FOLDS`, and every line names the artifact it came from. The
lineage is exp-018's (`experiments/exp-018-fresh/`); this experiment climbs
it under the desk's order.

## 1. The round budget (step 1b)

**Rule, as the desk set it:** the median of the per-month optima over three
held-out months spanning both seasons. February, May and August, each as a
leave-one-month-out fold, none of them a fold this experiment reports. The
curve is the all-rows RMSE of the full three-head prediction at each
LightGBM checkpoint, taken with `booster.predict(..., num_iteration=n)` on a
single fit, step 200 up to 2800 (`code/budget.py`, `results/budget.json`,
`logs/budget.log`).

| month | n_valid | argmin | RMSE at argmin | RMSE at 1200 | regret of holding 1200 |
|---|---|---|---|---|---|
| 2025-02 | 143,731 | **1200** | 457.05 | 457.05 | 0.00 |
| 2025-05 | 185,202 | 2200 | 327.05 | 327.20 | 0.15 |
| 2025-08 | 191,181 | 1000 | 403.22 | 403.37 | 0.15 |

**median of {1200, 2200, 1000} = 1200 rounds**, which is the value exp-018
already used. The rule confirms the incumbent rather than moving it.

**The finding is the flatness, not the argmin.** The worst regret from
holding 1200 against any month's own optimum is 0.15 s, and February's whole
curve spans 3.2 s of which 3.0 s is spent by round 800. Round count is not a
lever on this lineage, so no later arm needs to re-tune it and the three
argmins disagreeing (1000/1200/2200) is noise about a flat objective rather
than a seasonal effect. Corroboration from a different lineage and a
different learner: the blend lane measured its Huber member *losing* 3.57 s
at four times the rounds.

*Caveat on the log.* `logs/budget.log` also prints an `rmse_nonghost` trace
during the fit; that series is computed on non-ghost rows only and its
minima differ from the table above (August's non-ghost trace bottoms near
2200). The table and `results/budget.json` are the all-rows scores and are
the ones the rule is applied to.

## 2. Seed bagging, and why the pair had to be rebuilt (step 1a)

*(in progress — `logs/cand-s1.log`, `logs/s1-verify.log`, `logs/s3-folds.log`)*

The blend owner fits member weights on an OOF and then serves a candidate.
That only estimates the served member's contribution if the OOF and the
candidate come from the same estimator. exp-018 shipped a one-seed OOF with
a two-seed candidate, so the blend lane's 4.70 s and weight 0.596 are floors
on the served member, not measurements of it. This step produces two matched
pairs — one seed and three seeds — that differ in nothing but seed count.

Matching is a property of the *code path*, not only of the seed count:
`exp-018-fresh/code/submit.py --seeds n` averages n whole pipelines (a fresh
exact head and a fresh ghost head per seed), while `code/model2.py` bags the
main head, CatBoost and the ghost head and fits the exact head once. Those
are different estimators, so the three-seed candidate is written by
`code/submit2.py`, the serving half of `code/run2.py`, and not by exp-018's
`submit.py`. The one-seed candidate keeps exp-018's `submit.py` because its
matched OOF is exp-018's own.
