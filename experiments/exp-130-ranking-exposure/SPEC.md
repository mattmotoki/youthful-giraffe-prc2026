# exp-130 step 2 — clean ranking exposure, annual anchored learner

**Reading:** step 2 asks whether the full exp-018 anchored learner improves p12 on the
ranking-weighted ordinary no-clock population. The desk's clean-view rule applies to training
weight denominators and scoring; the ranking counts stay unchanged because the exclusions are
2025 fold IDs. `prc.lirf_features` v2 is added to every arm. The exposure-weighted fit is the
primary; unweighted, feature-sham, and weight-placebo fits are controls. I use six-month
Jan/Jul seasons because the assignment names those two ranking months. This keeps the seasonal
mapping fixed before any fit. I do not reuse exp-128's failed residual learner, and I do not
use weather.

**Status:** frozen before the step-2 fit. This pre-fit amendment names the required placebo
and defines the ranking vector against v22. No exp-130 step-2 fit output was read before this
amendment.

## Reading and source rows

The 2026-09-25 `exp-130 step 1` ledger row showed that the all-row LFPG January cell held
35.8% of supported estimated squared error. The desk identified its two 2025 January
`UNREACHED_ROLLOVER` rows as the cause. The clean census removes these rows from fold scoring:
LFPG January ordinary no-clock MSE falls from 634,813 to 63,633 s² per row. The supported
ranking-weighted v22 MSE changes from 89,041 to 60,325 s² per row. `CENSUS.md` and
`census.csv` retain both views. The 2026 ranking row counts do not change because these
exclusions are 2025 validation rows.

The 2026-09-25 `H3 planning pass (Astra)` ledger row names ranking-exposure weighting of a
full-feature annual learner as the first hypothesis. The `exp-128 unit 2` ledger row records
that the earlier residual learner failed because its unweighted control was weak. This fit
uses one direct, anchored learner trained on eleven months, with the unweighted learner as its
control. The `exp-127 clock precision` and `exp-129 unit 4` rows give no verified gain to reuse
for this no-clock population. The exp-017 A1 row reports a 1.56 s July gain from dropping only
the four unreachable rollover labels from bulk training; I keep that training rule for every
arm. The clean census correction is committed at `b3f844468ceed9680207213048878649d04723f1`.
This SPEC keeps ranking exposure as the tested variation.

## Population and data

- Training rows are all ordinary 2025 departure rows (`AIRCRAFT_TYPE_mvt` present) in the
  eleven training months for each fold. The four IDs in `prc.data.UNREACHED_ROLLOVER` are
  excluded from every training fit with `prc.data.keep_reachable`.
- The three validation folds are `july_loo` (2025-07), `december` (2025-12), and `january`
  (2025-01). Features use the ranking-masked month frame. Validation `y` stays separate from
  feature construction.
- The scored population is ordinary no-clock departures: `obs_has_clock == False` and
  aircraft type present. The noac class keeps its existing rule and receives no update.
- The 2026 ranking side supplies only its ordinary no-clock row counts by airport and month.
  Ranking target labels are never read. The 2025 fold score has both all-row and clean views;
  `prc.data.excluded_ids(fold)` defines the clean mask. The clean view is the fit-weight and
  score view. All-row results stay in the report as a sensitivity. Eligible training rows use
  `prc.data.keep_reachable`; this is exp-017 A1, a separate training-only rule that does not
  alter either validation view.
- No weather input is used. Weather remains untested, and no weather data is downloaded.

## Features and learner

For each month, use `experiments/exp-018-fresh/code/features.py::build` and
`dataset.py::add_encodings`, plus `prc.lirf_features.lirf_signals` (module version 2). The
base member supplies the MVT minus AOBT anchor, flight-plan gaps, month-local traffic and
stand features, and fold-fitted target encodings. `lirf_signals` adds the month-local LIRF
signals from supplied movement fields. Its columns get a `lirf_` prefix; the ID column is join
metadata, not a feature. `lirf_flight_prefix` and `lirf_stand_group` are categorical; fit their
category maps on training values only and map unseen validation values to -1. Departure block
and taxi labels are blank while features are built. Arrival fields and other movement
inputs follow `docs/input-rule.md`.

Fit one LightGBM L2 regressor on the ordinary rows. For each row, the anchor is `r_aobt` when
finite. If it is missing, use the exp-018 fallback: the first finite value among
`te_y_standrwy`, `te_y_rwy`, and `te_y_apt`. Fit the target `y - anchor` and predict
`max(30, anchor + residual_prediction)`. This is one main regression head; no exact-schedule
classifier, noac head, CatBoost head, or weather member is added.

Use the exp-018 main learner settings: L2 objective, learning rate 0.05, 255 leaves, minimum
60 rows per leaf, feature fraction 0.7, bagging fraction 0.8 once per round, L2 penalty 5,
maximum categorical threshold 64, categorical smoothing 20, categorical L2 penalty 10, six
threads, and 1,500 rounds. Drop the exp-018 `HIGH_CARD` columns as in its default `lean` fit.
Use the same feature list, category encoding, fold seed, and LightGBM settings for every arm
in a fold. The LightGBM seed is 7, the bagging seed is 8, the feature-fraction seed is 9, the
feature-sham seed is 13017 plus fold index, the weight-placebo seed is 13007 plus fold index,
and the bootstrap seed is 20260925. Use CPU only.

## Frozen arms and exposure weights

1. **Control:** fit the learner with every eligible ordinary training row at weight 1.
2. **Primary:** fit the same learner with ranking-exposure weights.
3. **Sham:** fit the primary weights and settings after independently permuting every
   `lirf_signals` feature within airport × month, using the fixed feature-sham seed in `fit.py`.
4. **Weight placebo:** fit the same learner after permuting ranking exposure across airport
   labels within each season group, using the fixed weight-placebo seed. Keep the training
   denominator for each airport fixed. This preserves the exposure shares but breaks their
   match to the airport.

The ranking months define two fixed six-month groups: October–March maps to January exposure;
April–September maps to July exposure. For each ordinary training row, let `p` be its
airport × group share among ordinary no-clock ranking rows and `q` be its airport × group
share among eligible ordinary training rows. The primary and sham row weight is `p / q`, then
scaled so the fit-row mean is 1. This density ratio uses only ranking row counts and the
eligible training rows. A positive-target cell with no training support stops the fit before
any model is trained.

All four fitted arms and each arm's 50/50 blend with p12 are reported. A blend is
`0.5 * arm_prediction + 0.5 * p12_prediction`.

## Scoring and frozen gain gate

The baseline is p12, reconstructed on each fold as `y_true - r12` from
`experiments/exp-047-queue/code/base.py::load(with_v12=True)`. Positive gain means lower
squared error than p12. Score both all-row and clean ordinary no-clock validation views, using
the unchanged 2026 ranking counts by airport: January fold uses 2026-01 counts, July fold uses
2026-07 counts, and the December check uses the January airport profile. December has zero
board weight. The pooled score uses the exact ordinary no-clock ranking counts in January and
July, not the overall all-class month shares.

Report the all-row and clean gain for the control, primary, sham, weight placebo, and all four
p12 blends in each fold and in the January/July pooled score. Report each supported cell's gain
and its ranking row count. The required cells are LIRF July, LTFM January and July, LFPG
January and July, and EDDF January and July. A cell gain is the clean p12 MSE minus the arm MSE
on that fold's rows; the ranking row count weights it in the pooled score.

Use 2,000 paired airport × UTC-day cluster-bootstrap draws with seed 20260925 in `fit.py`.
For the primary to pass frozen gain gate, all of these conditions must hold on clean rows:

1. Its gain over p12 is positive on July, December, and January.
2. Its 95% bootstrap interval over p12 is above zero on each fold and on the pooled January
   and July score.
3. Its paired gain over the feature sham and the weight placebo is positive, with a 95%
   bootstrap interval above zero on the pooled score. Report paired fold intervals too. The
   unweighted arm is the named control and its paired gain is reported, but it does not replace
   the frozen gain gate sham or placebo.

The primary remains the exposure-weighted arm. No arm is selected after looking at scores.
Price its clean pooled no-clock gain in board seconds with the common rule:
`245.0207 * (1 - sqrt(1 - share * frac))`, where `share` is the clean no-clock ordinary
cell's share of supported clean estimated v22 squared error, and `frac` is the weighted clean
MSE reduction divided by that cell's clean baseline MSE. Report unsupported with-clock
exposure separately; this price is a supported-cell projection, not a 2026 score.

If the primary fails frozen gain gate, the one repair attempt is its precomputed 50/50 blend with p12.
This is a fixed shrinkage step with no extra model fit; it does not change the frozen gain gate or
promote the blend as a candidate. Report the failure cause and the blend result.

## If the primary passes

Only after the frozen gate passes, refit the primary on all twelve eligible 2025 months and
predict the submitted 2026 ordinary no-clock rows. Write a compressed delta vector against
`youthful-giraffe_v22.parquet` in `submitting.parquet` ID order, with zero outside ordinary
no-clock rows. Do not submit a candidate. The desk owns any board pricing, review, or upload
decision.

## Kaggle and provenance

Use one private CPU kernel and one private input dataset named `prc-exp130-*`. Find attached
inputs by exact file name under `/kaggle/input`; require one match, as in exp-127 commit
`69a2e9d3`. Record dataset version, kernel slug and version, seed values, source hashes, and
output hashes in `manifest.json`. Poll kernel status no more often than every five minutes.
