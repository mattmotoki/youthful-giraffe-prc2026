# exp-137 — runway-state proxies for with-clock rows

## Reading and result lineage

This SPEC follows `docs/assignments/2026-09-27-exp137-withclock-proxies.md` from
`origin/main` at `c688fdad04e2df1608569d80d148e7fedefd3a54`. It is frozen before any exp-137
fit or scored label read.

Source ledger row **861** records that exp-134's annual learner with its 21 same-runway window
proxies passed frozen gain gate on ordinary no-clock rows and became v24. Ledger row **865** records that
the added wider and coarser windows failed against that learner. This unit keeps exp-134's
original t−1, t, t+1 proxy features and tests one variation: use the exp-134 learner on ordinary
with-clock rows, add the row's own two clock inputs, and use v23 as the fold base.

## Base and pre-fit check

The fold base is v23 on with-clock rows. Rebuild v22 from exp-130's saved fold rows as
`p12 + d46` on rows with an observed clock and known aircraft type, and as p12 on the
complement. Add the saved exp-127 unit-2 OOF delta from
`origin/exp127-clock-precision:experiments/exp-127-clock-precision/push/fold_update_vectors-r1.npz`,
matched by movement ID. Set that delta to zero for LIRF, as v23 did. Missing OOF IDs receive
zero. The source is pinned by
`base_reproduction-r1.json` at exp-127 branch commit `f03ee873`.

Before any learner fit, each arm kernel must rebuild and record the LIRF-held full-fold v23
RMSEs from the source JSON: 270.863644 s for July LOO, 181.665918 s for December, and
327.477267 s for January, within 1e-6 s. It must also record the v23 RMSE on the exact clean
ordinary with-clock score rows used below. Stop before fitting if the full-fold values do not
match, any fold base ID is missing or repeated, or the input row metadata disagrees.

On the ranking side, v24 is the base. It equals v23 on ordinary with-clock rows. The ranking
update, if allowed by the gate, is relative to the saved v24 prediction and is zero outside the
ordinary with-clock class.

## Inputs and features

Use exp-130's frozen clean monthly frames, eval folds, 149-feature source, LIRF features, and
ranking metadata, plus exp-134's label-free departure, arrival, stand-geometry, and exp-045
point inputs. Preserve the exp-134 point matching and all 21 proxy definitions without change:
arrival taxi-in mean and ratio, current-250-m clock-duration mean, implied-off-block duration
mean, short-gap share, departure count / runway-hour q95, and filed-plan delay mean, each at
same-runway slots t−1, t, and t+1. For the t slot, leave the row out of its own aggregate.
Permute the full 21-value vector among windows within airport × month for the sham.

Also add two row-level inputs built from the matched exp-045 departure segment: `MVT_TIME_UTC`
minus the current 250-m onset, and `MVT_TIME_UTC` minus implied off-block, using exp-127's
frozen speed fallback and stand mapping. These are input features, not targets. Keep rows with
missing point matches and use missing values for these inputs. The control has 149 exp-130
features plus the two row-level clock inputs (151 features). Proxy and sham each add the same
21 columns (172 features). Do not add the wider exp-136 features.

Proxy construction must finish before the kernel reads any fit or fold labels. The departure
input must not contain `TAXITIME_SEC_mvt`; taxi time may be read from ARR rows only. The source
must verify all expected exp-045 point files against the exp-045 manifest before fitting.
Keep the exp-134 row retention rule: if a proxy source row is missing, leave the proxy values
missing; do not drop the model row.

## Learner and arms

Use exp-130's annual LightGBM learner, encodings, anchor, training population, reachability
rule, and CPU settings from `origin/exp129-tail-markers` commit `fc1d8008`'s exp-134 source.
Train on ordinary rows only, after `prc.data.keep_reachable`; use unit row weights, 1,500
rounds, six CPU threads, and LightGBM / bagging / feature-fraction seeds 7 / 8 / 9. Fit the
same three arms in each private kernel:

1. **Control:** 149 exp-130 features plus the two row-level clock inputs.
2. **Proxy:** control plus the 21 exp-134 proxies.
3. **Sham:** control plus the same 21 proxy columns, with each complete window vector
   permuted within airport × month using seed `13417 + target-fold index`.

For the main prediction in fold j, train on the other eleven 2025 months. For each unordered
pair of folds, train one calibration learner on the ten months outside the pair and predict
both pair folds. Use these calibration predictions to fit each held fold's blend weight without
using that fold's labels. Fit one final ranking weight per arm on January and July main OOF
predictions only. For every weight, use the exp-134 weighted least-squares scalar against the
fold base, clipped to [0, 1]. For a fold/airport cell, each validation row has weight
`2026 ordinary with-clock ranking rows / clean validation rows`; December uses the January
airport profile and has zero board weight.

Blend each learner prediction a with its base b as `b + w × (a − b)`. Score only clean ordinary
with-clock rows: `obs_has_clock` is true, the exp-130 ordinary-row mask is true, and the row is
not in `prc.data.excluded_ids(fold)`. Use the clean ordinary with-clock ranking counts by
airport for July and January. Use January counts for the December check. Keep all other classes
out of the score and the ranking update.

## Frozen gain gate

Gain is base squared error minus blended squared error, in s² per clean ranking row. Use 2,000
paired airport × UTC-day bootstrap draws with seed `20260925`, resampling UTC dates within
airport and using the same draws for each arm and paired difference. Pool January and July by
their exact ordinary with-clock ranking-row counts; report December separately.

The proxy arm passes only if it:

1. has positive gain over v23 on all three folds;
2. has a 95% paired interval lower bound above zero on every fold and the January/July pool;
3. beats control and sham by point estimate on each fold and the January/July pool; and
4. has pooled proxy-minus-control and proxy-minus-sham 95% paired intervals above zero.

Report proxy-minus-control intervals by fold and pooled as feature-attribution evidence; they
do not replace the gate clauses. Keep this gate fixed after the SPEC is pushed.

For a descriptive repair only, if the cross-fitted proxy blend fails, also report the fixed
50/50 proxy and v23 blend from the same saved OOF predictions. It cannot change the frozen gain gate
verdict, and it cannot produce a ranking update.

## Tail and board price

For rows with `y_true > 3600 s`, apply the tail rule tail reading: pool the three folds within each
airport and report p23 and all three blended-arm RMSEs, row counts, and changes. The secondary
tail clause is met when the pooled proxy RMSE is no more than 1 s worse than p23 at every
airport with tail support. Report LIRF separately. Include fold-level tail values as context.

The clean exp-130 census assigns 23.301073748872064% of supported v22 squared error to ordinary
with-clock rows. Estimate that class's share of v24 squared error by multiplying the census
share by the January/July ranking-weighted with-clock MSE ratio v23/v22 measured on the matched
fold rows, then by `(245.0207 / 242.8064)^2`. Estimate the board score change from v24 =
242.8064 with

```text
share_v24 = 0.23301073748872064 × (MSE_v23 / MSE_v22) × (245.0207 / 242.8064)^2
fraction = pooled proxy gain over v23 / pooled v23 MSE
score reduction = 242.8064 × (1 − sqrt(1 − share_v24 × fraction))
```

Report the inputs to this estimate and whether the square-root domain is valid. Label this as a
transfer estimate, not a measured board score.

## Ranking update and limits

Only if the proxy arm passes frozen gain gate, fit the proxy learner on all twelve 2025 months, use the
January/July OOF ranking weight, and predict 2026 ordinary with-clock rows. Write an `.npz` in
`data/raw/submitting.parquet` order with `MVT_ID_mvt` and `delta`; delta is candidate minus v24
for ordinary with-clock rows and zero elsewhere. Verify exact ID order, unique IDs, finite
values, and zero changes outside the class. Do not write a candidate parquet or upload.

Do not read 2026 target labels. Do not use or download weather data. The workstation ERA5
snapshot is untested. The three folds do not cover all ranking airport-month exposure. The
availability of t+1 source windows at live prediction time is also untested.

## Execution record

Use three private CPU Kaggle kernels from the start, one each for control, proxy, and sham, so
each kernel performs only its own main and calibration fits. Use input datasets
`<your-account>/prc-exp130-clean-input` and `<your-account>/prc-exp136-input`, plus a private exp-137 dataset
for the v23 fold base and frozen exp-137 code. Record each kernel slug, version, output hashes,
and data versions in `manifest.json`. Record and verify every dataset file and required column
against its kernel reads before pushing each kernel. Freeze and push this SPEC before any fit.

## Pre-score support clarification

This clarification was committed before any exp-137 learner score or
prediction output was read. Sham version 1 stopped after its three main fits,
before calibration or score files, because two January ranking airports had no
clean with-clock validation rows. The only downloaded table was the required
v23 base reproduction. The failure and the initial row-support audit are in
`kaggle/run-v1/sham/FAILURE.md` and
`kaggle/run-v1/support-audit-before-score.csv`. The known-aircraft support
check is in `kaggle/run-v3/support-audit-known-aircraft.csv`.

An airport-fold cell is supported only when it has at least 100 clean ordinary
with-clock, known-aircraft validation rows on at least 10 distinct UTC days.
Noac rows stay outside fold scoring and ranking updates, as in the existing
rule. This minimum keeps the airport-day bootstrap and ranking density weight
away from a single-row cell. In each fold, normalize the ordinary with-clock,
known-aircraft ranking counts over its supported airport cells and score only
those cells. Apply every frozen gain gate clause unchanged to these supported-cell fold and
pooled gains. Report all unsupported ranking rows and the supported share for
each fold and for January plus July. The gate result is evidence for supported
exposure; it does not test the unsupported cells.

The support audit found 104,823/104,823 July ranking rows and 91,433/91,433
December rows supported. January supports 68,716/91,433 rows (75.15%): LEMD
has one clean validation row on one day for 15,881 ranking rows, LFPG has no
clean validation row for 1,136 ranking rows, and LIRF has none for 5,700 rows.
The January plus July pool therefore supports 173,539/196,256 ordinary
with-clock ranking rows (88.42%).

The known-aircraft support audit uses the `known_aircraft` flag in the frozen
v23 fold base. It reproduces the counts above. A pre-push review found that
held-fold scoring must also apply this existing noac exclusion. This code and
SPEC clarification are committed before any learner score or prediction output
was read; v1 logs show the support error only, and the only downloaded table is
the required v23 base reproduction. The frozen gain gate clauses and support thresholds do
not change.

For the board-price estimate, compute v22 and v23 MSE on the supported
January/July cells. Multiply the supported-cell proxy gain fraction by the
supported share 173539/196256. This assumes zero gain on unsupported cells.
Keep the census error share and v24 adjustment from the original formula.
Label the result as a coverage-adjusted transfer estimate, not a measured
board score. If frozen gain gate passes, the final prediction still covers all ordinary
with-clock ranking rows; the unsupported cells remain unvalidated and have zero
assumed gain in this estimate. The tail table and tail rule clause stay unchanged.
