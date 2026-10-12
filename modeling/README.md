# Modeling sprint 01

This folder is the complete first modeling sprint. A separate sprint docs tree
would add navigation without adding useful information, so this file is both the
plan and the runbook.

## Decision

Build two target-leakage-quarantined lineages under the current prediction-time
assumption, and make their errors easy to review before any ranking submission:

1. `airport_calendar`: a hierarchical median baseline by airport, weekday,
   hour, and quarter-hour.
2. `catboost_operational`: a categorical tree model with airport, stand,
   runway, destination, aircraft, operator, calendar, schedule-plan deltas, and
   scheduled movement counts.

The dashboard is the main review surface. It is one generated HTML file with
embedded aggregate results and no network dependencies.

## Frozen validation

| Fold | Training period | Validation period | Purpose |
|---|---|---|---|
| May | January–April 2025 | May 2025 | Early forward transfer |
| July | January–June 2025 | July 2025 | Match a ranking season |
| December | January–November 2025 | December 2025 | Late-year drift |

All group statistics and model parameters are fit on the training side of each
fold. CatBoost uses a fixed iteration count, so validation labels do not control
training. Scheduled movement counts may use the full supplied schedule because
those movements are known before the target departure. Validation keeps every
departure with a scheduled time and nonmissing target, including extreme target
values in the official scoring population.

## Quarantined inputs

The first sprint excludes actual movement time, actual block time, actual
off-block time, arrival actuals, target-derived differences, flight identifiers,
and event-hour weather. Direct ERA5 and trajectories are later ablations. They
do not block this sprint. Planned-time feature availability still needs final
confirmation against the organizer's exact prediction cut-off.

## Run

Install the exact modeling environment, run a small integration test, then run
the full three-fold comparison:

```bash
.venv/bin/pip install -r modeling/requirements.lock
.venv/bin/python modeling/run.py --smoke
.venv/bin/python modeling/run.py
```

The full run writes intermediate files under `modeling/results/` and replaces
`modeling/dashboard.html` with a self-contained review dashboard. Run metadata
records the Python platform, important package versions, and SHA-256 hashes for
all 12 training files.
After a dashboard-only code change, rebuild it from the saved results with:

```bash
.venv/bin/python modeling/dashboard.py
```

The first full run scored 566.0 seconds RMSE for `airport_calendar` and 499.0
seconds for `catboost_operational` across 541,583 validation rows. The 11.8%
gain was present in all three folds.

## Dashboard questions

- Does the tree model beat the calendar baseline on the same rows?
- Which dates form high-error clusters?
- Which airports are hard, and is the model improvement consistent?
- Does error increase with taxi duration?
- Which safe features does the tree model use most?

The dashboard includes overall scorecards, a GitHub-style daily RMSE calendar,
airport comparison, target-band errors, worst-day detail, and feature importance.
