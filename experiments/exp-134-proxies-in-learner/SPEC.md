# exp-134 — runway-state proxies as row features

## Status and disclosure

This specification is frozen before any exp-134 learner fit or scored label read. I read the
assignment from `origin/main` at `ebf2c390`. I also read the exp-130 source code at
`origin/exp128-noclock-model` commit `c5f38c50`, the exp-130 report, the exp-133 step-2 report,
and the new rows in `docs/LEDGER.md`.

**Pre-fit amendments, 2026-09-26.** The first private kernel stopped before any label read or
learner fit. Its loader asked for `PHASE_mvt`, which the label-free input pack had already
removed; the error is saved at `kaggle/run-v1/prc-exp-134-proxy-learner.log`. I removed that
unused field from both input projections. The next run used the exact kernel slug
`<your-account>/prc-exp134-proxy-learner` and input data set version 2. It built all 14 proxy months
and fit the July control arm, then stopped because two ordinary training IDs had no proxy row.
The error log is `kaggle/run-fit-v1/prc-exp134-proxy-learner.log`; no fold predictions, scores,
or ranking delta were written. For any further run, keep those base rows and give them 21
missing proxy values; record their count and the hash of their sorted IDs. Do not exclude them.
The proxy design, learner, blend, checks, and pass rule below do not change. A complete rerun
would use input data set version 3 and kernel version 2.

Ledger row **exp-133 step 2** is the positive source result for this unit: the input-only proxy
contrast gained 306 s² per ordinary no-clock row in July and 118 s² in January, although its
window correction failed frozen gain gate. Ledger row **exp-130 step 2** records an unweighted annual
learner whose fixed 50/50 blend with p12 was MAYBE. This unit keeps both results as context and
tests one variation: the same 21 proxies become row features in the exp-130 annual learner, and
each arm uses a cross-fitted scalar blend weight.

The 21 proxy definitions come from `experiments/exp-133-runway-state/SPEC-step2.md` at
`d442df6e`. I read its published results, not any exp-134 output. The exp-045 manifest lists
point files for every day in all twelve 2025 months and both 2026 ranking months. I use those
saved point outputs to build every clock-duration input. I do not use the exp-127d four-month
coverage output as a substitute for the remaining months. I reject a fold-only feature table
because each exp-130 fold needs proxies in all eleven training months.

## Inputs and proxy construction

Use the exp-130 prepared input at its frozen version 1, plus a new private exp-134 input data
set. The new data set contains only movement inputs, arrival taxi-in inputs, stand geometry, and
the exp-045 point-file inventory. For a departure row, the proxy builder does not read
`TAXITIME_SEC_mvt`. It reads that field only for ARR rows, as in the exp-133 step-2 definition.
It does not read target labels or residuals while it builds the proxies.

Build proxies for all twelve 2025 months and both 2026 ranking months. For each 2025 fold, the
training frame contains the other eleven 2025 months. Proxy construction uses only inputs from
the month being built. It does not use labels to match or aggregate rows.

The row window key is airport × movement runway × UTC date × half-hour slot. Each departure
row receives the 21 values for source slots t−1, t, and t+1. A proxy vector is shared by all
rows in its target window. The same definitions and fallbacks as exp-133 step 2 apply:

1. **Arrival taxi-in:** mean ARR taxi-in by destination airport and runway; if that cell is
   empty, use the airport mean for the slot. The ratio uses the airport × UTC-hour median from
   the eleven fit months. Fallbacks are the fit-month airport median, then the fit-month global
   median. The mean and ratio each have three slot columns.
2. **Clock durations:** for each matched departure point segment, compute `MVT_TIME_UTC −
   current 250 m onset` and `MVT_TIME_UTC − implied off-block`. The implied time uses the first
   ground point, the mapped surveyed stand, its great-circle distance, and the exp-127 rule:
   use recorded ground speed when it is at least 1 kt, otherwise use 3 kt. Match a point
   segment to one raw departure by airport, UTC date, normalized flight string, and nearest
   `MVT_TIME_UTC`; accept only a unique nearest match within 600 s, with a 120 s gap to the
   second candidate. Ambiguous or unmatched rows have missing clock values. A missing stand
   position leaves only the implied-time value missing. The target row is left out of its own
   t-slot sum and count. The two duration means each have three slot columns.
3. **Take-off spacing:** within airport × runway × UTC date, sort departures by movement time.
   The first value is the share of valid prior gaps under 90 s in the slot. The second is the
   slot departure count divided by the runway's fit-month 95th-percentile hourly count, with
   fit-month airport and then global fallbacks. These two values each have three slot columns.
4. **Filed-plan delay:** mean `MVT_TIME_UTC − EOBT_1_flt` in seconds for the slot, with three
   slot columns.

The resulting feature names are the seven bases
`arrival_mean`, `arrival_ratio`, `clock_current250`, `clock_implied`, `spacing_short_share`,
`departure_count_q95`, and `filed_delay_mean`, each crossed with lags −1, 0, and +1. This is
21 numeric columns. Do not add or drop a proxy after the fit starts.

Use the exp-045 point files by their frozen per-day byte sizes and SHA-256 values in the copied
manifest. The point sources are the exp-045 kernels for 2025-01 through 2025-12, plus the
2026-01 and 2026-07 sources. A missing, duplicate, or changed daily point file stops the run
before any label-bearing model fit. The same check applies to the 2026 point files before any
ranking prediction.

For the shifted-day arm, replace each proxy window vector with the mean from the same airport,
runway, and slot on dates seven days before and after, within the same calendar month. Average
the one or two available dates; if neither exists, the vector is missing. For the sham arm,
permute the full 21-column vector among distinct windows within airport × month. Use the same
permutation for every row in a window. Use seed `13417 + fold index`, where fold indices are
July 1, December 2, and January 3. Do not permute the base features.

## Base learner and arms

Use the exp-130 code and settings from `origin/exp128-noclock-model` commit `c5f38c50`. Do not
change that branch. Reuse the exp-130 feature builder, encodings, anchor, training population,
reachability rule, LightGBM parameters, and CPU settings. Confirm the base learner has the
same 149 features as exp-130; the proxy arms have those 149 features plus 21 numeric columns.

For each held fold, train on the other eleven months. Train only on ordinary rows, using the
unweighted control settings and row weights of one. Exclude the four frozen unreachable IDs
with `prc.data.keep_reachable`. Use the exp-130 L2 LightGBM anchor: `r_aobt`, then the first
finite value of `te_y_standrwy`, `te_y_rwy`, or `te_y_apt`; fit `y − anchor`, and floor the
prediction at 30 seconds. Use 1,500 rounds, six CPU threads, LightGBM seed 7, bagging seed 8,
and feature-fraction seed 9.

Fit four arms in every fold:

1. **Control:** the exp-130 learner with its 149 base features.
2. **Proxy:** control features plus the 21 real proxies.
3. **Sham:** control features plus the permuted 21-value window vectors.
4. **Shifted day:** control features plus the shifted-day proxy vectors.

Each arm's learner prediction is `a`. Its blended prediction is `p12 + w × (a − p12)`. For
held fold `j`, fit that arm's single scalar `w_j` on the other two folds' clean ordinary
no-clock rows. To keep every label in `j` out of `w_j`, make each source-fold prediction with
a calibration learner trained on the ten months that exclude both `j` and that source fold.
Fit one calibration learner for each unordered fold pair and arm; its two held months supply
the calibration predictions for the two opposite weights. Use the same exp-130 learner settings
and base features; the 21 proxies are present in the three proxy arms. For each source-fold
airport cell, each row has weight `ranking_rows / clean_validation_rows` for that airport.
December uses the January airport profile, as in exp-130, but has zero board weight. Solve the
weighted least-squares scalar

```text
w = clip(sum(weight × (a − p12) × (y − p12)) /
         sum(weight × (a − p12)^2), 0, 1)
```

Use `w = 0` if the denominator is zero. Fit one value per arm and held fold. For the final
ranking prediction, use one `w_rank` per arm fitted on the January and July OOF rows with the
same clean ranking weights; do not use December because its board weight is zero. This makes
the ranking blend use only out-of-fold learner predictions.

## Population, score, and frozen gain gate

The three held folds are `july_loo` (2025-07), `december` (2025-12), and `january` (2025-01).
Score only ordinary no-clock departures on the clean exp-130 view. Keep all validation rows
and the four training exclusions exactly as exp-130. The p12 prediction is `y_true − r12` on
folds. Report all four blended arms on every fold and on the January/July pool, with their
cross-fitted `w` values. Also report the unblended learner arms and the paired proxy-minus-
control, proxy-minus-sham, and proxy-minus-shifted-day differences.

Apply the common frozen gain gate to the blended proxy arm. It must have positive gain over p12 on
all three folds, have a 95% paired airport × UTC-day cluster-bootstrap lower bound above zero
on each fold and on the January/July pool, and beat the sham and shifted-day arms by point
estimate on each fold and on the pool. The pooled paired 95% intervals for proxy minus sham
and proxy minus shifted day must also be above zero. Use 2,000 paired draws, seed `20260925`,
resample UTC dates within airport, and use the same draws for every arm. Report the proxy minus
control interval as the feature-attribution check; it does not replace the frozen gain gate sham or
placebo clauses.

Use the clean exp-130 ranking counts by airport. The January fold uses the 2026-01 counts; the
July fold uses the 2026-07 counts; December uses the January profile for its check. Pool January
and July with their exact clean ordinary no-clock ranking row counts. The clean census assigns
54.4% of supported v22 squared error to ordinary no-clock rows. This class is unchanged in v23,
so its v23 baseline equals p12. Price the pooled gain on v23 with the common formula:

```text
245.0207 × (1 − sqrt(1 − share × frac))
share = clean ordinary no-clock share of supported v22 squared error
frac  = pooled clean MSE reduction / pooled clean p12 MSE
```

Do not claim the price if the formula is outside its real domain. State the fold results,
confidence intervals, weights, price, feature coverage, and limits in `REPORT.md`.

If the cross-fitted blend fails frozen gain gate, make one repair from the same saved predictions: the
fixed 50/50 proxy arm and p12 blend. This adds no learner fit and does not change the gate or
produce a ranking vector. Report its fold and pooled gains. Do not run another repair.

## Ranking output and execution

If and only if the cross-fitted proxy blend passes frozen gain gate, fit the proxy arm on all twelve 2025
months, apply `w_rank` to the 2026-01 and 2026-07 ordinary no-clock rows, and save its delta on
top of v23 in exact `data/raw/submitting.parquet` order. All other rows receive zero. Verify
unique IDs, exact coverage and order. Do not create a candidate parquet or upload a submission.

Use one private CPU Kaggle kernel for all four OOF arms and any permitted final refit. The
kernel and its private exp-134 input data set must use names beginning `prc-exp134-`. Record
all source, feature, input, output, kernel, and data-set hashes and versions in `manifest.json`.
Poll kernel status no more than once every five minutes. No exp-134 fit, score, or residual
result was read before this SPEC was frozen.
