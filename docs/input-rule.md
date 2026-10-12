# Prediction-time input rule

The organizer's recorded September answers state that the task is after operations, that the supplied fields are valid inputs, and that open ranking-month trajectories, including own-flight ground positions, can be used.

## The rule

Every column present in `ranking.parquet` is a legal model input, for every
row of the file, including the actual take-off time of the departure being
predicted, the Network Manager actual off-block time of that departure, and
the complete actual timings of all other movements (arrivals and departures)
in the ranking months. The only things a prize-eligible model must not use are:

1. leaderboard scores or any signal derived from the scoring process;
2. the blanked departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` values
   themselves (they are not in the file);
3. external data that is not open, documented, and reproducible
   (`docs/external-data-rules.md`).

## Why this is the reading

- The official data page states that the ranking set "contains the same
  columns as the training data" and that only `BLOCK_TIME_UTC_mvt` and
  `TAXITIME_SEC_mvt` for departures "have been blanked out". It defines the
  submission target as `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt -
  BLOCK_TIME_UTC_mvt`. The organizer therefore hands over the take-off time
  and asks for the block time. This is a retrospective reconstruction task,
  not a pre-departure forecasting task. The rationale page's operational
  framing describes why taxi time matters, not the input contract.
- The ranking page's only restriction is: "We will monitor submissions for
  attempts to learn from or exploit the ranking process." The object is the
  ranking *process* (scores), not the ranking *data*.
- The eligibility page restricts external data and code licensing. It says
  nothing about which supplied columns may be used.
- Read of the re-issued `ranking.parquet` (bucket date 2026-09-04, SHA-256
  `9db75e3a…325f86`, 344,841 departure rows, 10 airports, LTAI absent from
  departures): `MVT_TIME_UTC_mvt` populated on 100.0 %, `AOBT_3_flt` and all
  other `_flt` columns on 98.5 %, `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`
  on 0.0 %. Arrival rows carry full actuals including taxi-in time.
- The leaderboard on 2026-09-05 (25 scored teams) has three teams at 277 to
  279 s RMSE and eleven more between 302 and 361 s. Our first-sprint model,
  which quarantined every actual column, scores 499 s on forward 2025 folds.
  The raw difference take-off minus NM off-block alone scores 445 s on July
  2025. Fifteen teams are already inside the band that only the provided
  actuals reach.

## What the first sprint got wrong

The sprint treated "prediction cut-off not stated" as "assume pre-departure".
The data page's target definition settles it the other way. The operational
lineage (`modeling/run.py`, features restricted to schedule and plan times)
remains a valid paper story and a diversity source, but it is not the scored
task.

## Validation consequence

Emulate the ranking file exactly in every validation fold: for the validation
month, blank departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, keep
everything else, and compute every feature from that masked frame. Features
built from neighbouring departures may use their `MVT_TIME_UTC_mvt` and
`AOBT_3_flt`, never their block time or taxi time. Arrivals keep all columns.
Training months keep full labels. Fit any target encodings and group
statistics on training months only.

## Facts a model must respect (July 2025 training, 190,713 departures)

- Target: median 948 s, mean 1026 s, std 745 s; 0.02 % at or below zero,
  0.34 % above 3600 s.
- `BLOCK_TIME_UTC_mvt - AOBT_3_flt`: median 5 s, but only 20 % of rows within
  60 s and 72 % within 300 s; 1st to 99th percentile is -1616 s to +840 s.
  The airport block time and the NM off-block are different measurements. The
  offset is airport-specific (LTFM median +234 s, LIRF -179 s, EHAM +114 s)
  and heavy-tailed (LIRF RMSE 771 s, EGLL 587 s, LSZH 282 s). Modelling this
  offset is the core of the task.
- `LOBT_flt` and `IOBT_flt` are near-identical (last received EOBT); block
  minus LOBT has median 416 s, 90th percentile 1494 s.
- 1.5 % of ranking departures have no NM join (`_flt` columns null). They
  need a fallback path with no `AOBT_3_flt`.
- The ranking set is 44 % January 2026 (152,719 departures by take-off time)
  and 56 % July 2026 (192,122). The 2026-08-13 file had only 63,156 July rows;
  the organizer re-issued it on 2026-09-04 and the leaderboard restarted then.

## Data version

The 2026-08-13 ranking and submitting files are stale (215,876 departures). The
2026-09-04 files have 344,841 departures, matching the leaderboard's
"usedPairs". The stale copies are kept as `ranking_stale_2026-08-13.parquet`
and `submitting_stale_2026-08-13.parquet` next to the live files. Training
files are unchanged (same sizes and bucket dates as 2026-08-13).
