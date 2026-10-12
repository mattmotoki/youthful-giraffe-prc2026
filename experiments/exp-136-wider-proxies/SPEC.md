# exp-136 — wider and coarser runway-state proxies

## Status and source result

This specification is frozen before any exp-136 label-bearing model fit or fold score is read.
The assignment was read from `origin/main`. The input builder, proxy code, fit code, and kernel
layout are based on exp-134 at `origin/exp129-tail-markers` commit `fc1d8008`; that branch is
read-only for this work.

Ledger row **861** records exp-134's proxy learner as frozen gain gate PASS. Its same-runway, 30-minute
window proxies improved the blended learner over p12, and its candidate became v24. The board
score was 242.8064, a 1.3172 second gain from v23, below the expected gain. This unit keeps
exp-134's learner and 21-column arm as a refit control, and tests one variation: extra runway
state at two more lags, wider time bins, and airport-wide scope.

No weather inputs are used. The ERA5 snapshot on the workstation is untested in this experiment;
no weather data will be downloaded. The stand-arrival features are not included.

## Inputs and feature definitions

Reuse exp-134's label-free departure and ARR-only arrival inputs, surveyed stand geometry, and
exp-045 daily point-file inventory. Use the exact-file discovery and hash checks in the copied
exp-134 source. Read `TAXITIME_SEC_mvt` only for ARR rows. Do not read ranking target labels or
departure taxi-time targets while building proxies. Build features for all twelve 2025 months and
2026-01 and 2026-07. Fit all normalizers on the training months for each fold.

The **control** is the exp-134 proxy arm, refit in this run: exp-130's 149 base features plus the
same 21 numeric runway-scope proxy columns from 30-minute bins at lags −1, 0, and +1. The seven
families are arrival mean, arrival ratio, current-250-m duration, implied off-block duration,
short-prior-gap share, departure count divided by its training q95, and filed-plan delay mean.
The original exp-134 definitions, row matching rules, fallbacks, and leave-self-out rule apply.

The **t2** arm adds the same seven runway-scope families for 30-minute bins at lags −2 and +2.
The **wide** arm adds those same 14 columns and these additional features:

- For arrival mean and arrival ratio, departure count divided by its training q95, and
  short-prior-gap share: runway-scope 15- and 60-minute bins, plus airport-wide 15-, 30-, and
  60-minute bins.
- Each added feature uses the target row's UTC bin and the adjacent bin at lag −1, 0, and +1.
  Bins do not cross UTC midnight. Airport-wide values combine all runways at that airport.
- Arrival mean is the ARR taxi-in mean in the destination airport and time bin. Runway-scope
  values first use the matching runway and fall back to the airport-wide value. The ratio uses
  the exp-134 fit-month airport-by-hour median, then its airport and global fallbacks.
- Departure count is the number of departures in the source bin. Its q95 denominator is fit on
  training-month bin counts for the same scope and width, with airport and global fallbacks for
  runway scope. Airport-wide values use their airport and global q95 values.
- Short-prior-gap share is the share of valid consecutive departure gaps under 90 seconds in the
  bin. Runway scope orders departures within airport, runway, and UTC date; airport-wide scope
  orders all airport departures within UTC date.

This gives 21 control proxy columns, 14 t2 columns, and 60 wider/scope columns in the wide arm.
The **sham** arm has the wide arm's columns. It leaves the 21 control columns unchanged and
permutes the 74 added values as one vector among 30-minute windows within airport × month, using
seed `13617 + held-fold index` (July 1, December 2, January 3). For 15-minute features, the
permuted value for a 30-minute window is the mean of its row values. One permutation is shared by
all rows in a 30-minute window.

## Learner, folds, and blend

Use exp-130's frozen input, 149 base features, encodings, anchor, training population,
reachability rule, LightGBM parameters, and CPU settings. Each learner has row weights of one,
1,500 rounds, six CPU threads, LightGBM seed 7, bagging seed 8, and feature-fraction seed 9. Fit
all four arms: `control` (170 columns), `t2` (184 columns), `wide` (244 columns), and `sham`
(244 columns). The new arms differ only in their listed proxy columns and sham permutation.

The held folds are `july_loo` (2025-07), `december` (2025-12), and `january` (2025-01). Score
clean ordinary no-clock departures, with the exp-130 exclusions and saved v22/p12 rows unchanged.
For each held fold and each arm, fit its blend weight on the other two folds' clean ordinary
no-clock rows. Source-fold predictions come from pairwise calibration learners trained on the
ten months that exclude both the target fold and source fold. Use exp-134's airport-density
ranking weights and clipped scalar least-squares rule. Fit the ranking weight from January and
July OOF rows only. December is a check fold with zero board weight.

## Frozen gain gate

The gate is fixed before the fit. For `t2` and `wide` separately, all conditions below must hold:

1. The blended gain over p12 is positive on each of the three folds. Its 95% paired
   airport × UTC-day bootstrap lower bound is above zero on each fold and on the January/July
   pool.
2. It beats the refit `control` blend and the `sham` blend by point estimate on every fold and on
   the January/July pool. The pooled paired 95% lower bounds for candidate minus control and
   candidate minus sham are above zero.

The refit control is the exp-134 baseline/placebo for this comparison; the within-airport-month
permutation is the sham. Use 2,000 paired draws, seed `20260925`, and resample UTC dates within
airport with the same draws for every arm. The gate verdict is `PASS` only if either candidate
meets all conditions. If both pass, select the one with the higher pooled gain over control; an
exact tie selects t2. Do not change the gate or tie rule after viewing a score.

The board-price calculation reports the estimated score reduction from v24 for any candidate:
start with v24 score 242.8064, use the candidate's incremental gain over the control blend, and
scale by the clean ordinary no-clock share of supported v22 squared error (54.4%). Adjust the
no-clock class share by the control's measured gain over p12 and state this estimate's
assumptions. Do not claim a price outside the real square-root domain.

Report the p12, control, t2, wide, and sham fold and pooled metrics, cross-fitted blend weights,
all gate checks and intervals, and board-price estimate. Report y > 3600 tail rows by fold and
airport with row counts, p12/control/t2/wide/sham RMSE, and gain over control for both candidates.
Report proxy coverage and runtime/input/output hashes.

If the gate passes, save a delta over v24 for ordinary no-clock rows only, in exact
`data/raw/submitting.parquet` order. Verify IDs, order, coverage, and zero changes outside that
class. Refit both the exp-134 control and selected candidate on the same twelve training months;
the delta is the selected blended prediction minus the control blended prediction. Do not create a
candidate parquet or upload a submission. If the gate fails, write no ranking delta and make no
second fit.

## Limits and execution

One private Kaggle CPU kernel runs all main and pairwise calibration fits, and the permitted final
fit only if the gate passes. No additional worker kernel may run at the same time. The private
kernel and input dataset use the `prc-exp136-` prefix. Record kernel and dataset versions, source,
feature, input, output, and spec hashes in `manifest.json`. Use the approved Kaggle CLI and never
print its credential.

These three held months do not cover all 2026 airport-month exposure. In particular, the ranking
exposure census reported that LIRF July is almost entirely no-clock, unlike the folds. A fold
gain may not transfer to that cell. The v24 board gain was also smaller than exp-134's estimate.
The ERA5 snapshot is untested here; no weather-based conclusion follows from this run. No
ranking labels are used, and no submission is uploaded.
