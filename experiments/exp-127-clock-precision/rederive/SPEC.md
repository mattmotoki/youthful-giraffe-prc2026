# exp-127b implied off-block clock fit

Frozen before any label-bearing fit on 2026-09-25. The label-free coverage scan
passed on private point-output scan version 5. Its source and output hashes are
in the root `manifest.json`.

## Reading and scope

Ledger row 839 records the first exp-127 fit. Its static stand geometry model
beat its controls on the pooled point estimate, but failed frozen gain gate because July
and December confidence intervals crossed zero. This fit tests one variation:
use an implied off-block time from the first ground point and the surveyed
stand position, with the same static geometry inputs and frozen gain gate.

The point source is the saved exp-045 private Kaggle output. The fit does not
fetch or store raw points. It excludes LFPG because its stand names failed the
position check. It scores only with-clock rows with a known aircraft type, and
it leaves no-clock and noac rows at the current head. LTFM has no with-clock
rows in these folds or the ranking file, so this fit does not update its
47,581 no-clock ranking rows.

## Label-free clock definition and coverage

For each matched departure, the first ground point is the anchor point used by
the exp-045 segment rule. Its time is `t_first`; its great-circle distance to
the surveyed stand is `d_first` metres; and its recorded ground speed is
`v_first` knots. If speed is missing or below 1 kt, use the fixed pushback
speed 3 kt. Otherwise use the recorded speed. The implied off-block time is:

```text
t_implied = t_first - d_first / (v_effective * 0.514444)
```

The model input is the time from the implied off-block to the current 250 m
onset: `current_250m_s - t_implied`. It does not use the absolute Unix time.
The static geometry inputs remain `survey_to_obs_stand_m`,
`survey_to_runway_km`, and `survey_is_osm`. The coordinate sham uses their
permuted counterparts and the implied clock computed from the sham stand.

The label-free coverage denominator is every with-clock fold row with a known
aircraft type, excluding LFPG. The implied time must exist on at least half of
these rows in each fold before any fit. The checked gate passed: coverage was
91.7% in December, 88.7% in January, and 90.9% in July. The current 250 m time
exists on all rows in this denominator. The airport table in `REPORT.md` and
the committed coverage CSV show its timing difference by airport.

## Folds, baseline, and learners

The scored folds are `july_loo`, `december`, and `january`. Each fit trains on
the other two folds and predicts the held-out fold. The target is the saved
v22 residual. Before fitting, reproduce the full-fold v22 RMSE values in
`support/baseline_reproduction.json` to 1e-8 seconds. The candidate updates only
in-scope rows. All fold gains use the full fold row count in their denominator.

The primary learner is the unit-1
`HistGradientBoostingRegressor(loss="squared_error", max_iter=100,
learning_rate=0.05, max_leaf_nodes=15, max_bins=63, min_samples_leaf=250,
l2_regularization=1.0, early_stopping=False, random_state=12727)`. It uses the
four primary fields above and one-hot airport fields. It accepts missing
feature values.

The controls are:

1. **Current head:** zero residual correction.
2. **Airport-only placebo:** the training residual mean for each airport, with
   the global training mean as a fallback.
3. **Coordinate sham:** the same learner, rows, and settings, with the
   permuted geometry and sham-derived implied clock.

If the primary learner fails frozen gain gate, run one frozen Ridge repair on the same
fields, folds, controls, and gate. Use `alpha=10`, `log1p` for the two distance
fields, training-row standardization for continuous fields, training medians
for missing values, the source flag, and airport indicators. Make no further
repair after this fit.

## frozen gain gate and board price

A gain is the reduction in squared error over the full fold. The primary model
must:

1. improve on the current head on all three folds;
2. beat both the airport-only placebo and coordinate sham on every fold and on
   the July/January board-weighted point estimate; and
3. have a two-sided 95% interval above zero on every fold and on the pooled
   July/January board-weighted gain.

Use 2,000 paired airport-day cluster-bootstrap draws with seed 12728. Resample
all airport-day clusters within each fold, including clusters with no eligible
row. Divide each draw by the sum of all fold rows in its sampled clusters.
Combine pooled draws with the fixed census weights: July 0.5571321275602379,
January 0.4428678724397621, and December 0.

Price board seconds with the census formula:

```text
245.0207 * (1 - sqrt(1 - share * frac))
```

No ranking vector is written unless HGB or its one Ridge repair passes frozen gain gate.
If it passes, fit on all eligible in-scope fold rows and save a compressed
`.npz` update in the exact `MVT_ID_mvt` order of `submitting.parquet`. Do not
make a competition submission.
