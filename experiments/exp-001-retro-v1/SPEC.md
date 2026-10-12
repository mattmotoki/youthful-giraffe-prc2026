# exp-001 retro-v1 — spec

Worker `retro-v1`, tier hard, exp-id `exp-001`, assignment `assignments/retro-v1.md`.
Metric: RMSE in seconds on the masked `july` and `december` folds of
`prc.data.fold_frames`, with the COMMON.md tail decomposition.

## My reading of the assignment, and the readings I rejected

I read the assignment as: build one feature function that cannot see a
departure's block time or taxi time, feed it a LightGBM model on raw seconds,
ship a candidate fast, then climb through the listed mechanisms. The reading I
rejected was the tempting one — start from the residual parameterisation
(`take-off − AOBT` as an offset), which I expect to be the single largest win —
because a v1 that is not the assignment's stated design makes the first ledger
row incomparable with the naive baseline already recorded. So v1 is the plain
design, and the residual parameterisation is the first logged climb step, where
its effect is measurable against v1 rather than tangled into it.

The second reading I rejected was building features across the concatenated
training months. The ranking file holds January and July 2026 as two isolated
months, so a congestion window that can reach into a neighbouring month exists
in training and not at prediction time. Every feature is built from one month
frame, in training, validation and ranking alike, and month-boundary edge
effects are therefore identical in all three.

Leakage is made structural rather than disciplinary: `build_features` blanks
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` on entry for every departure row,
exactly as the ranking file blanks them, so no branch of it can read a
departure's own answer even by mistake. Through v1 and v2 the two columns were
dropped outright, for departures and arrivals alike; the `v3` feature set
relaxes that to departures only, because arrival block time and arrival taxi-in
are present in the ranking file for all 344,693 arrivals and are the **only**
actual surface-time readout the task leaves us — every departure's taxi time is
blanked, so the surface state can be measured only on the movements coming the
other way. The guarantee that matters is unchanged and is enforced by test
rather than by reading: mutating every departure's block time by 12,345 s and
its taxi time to −99 leaves all 68 v3 feature columns bit-identical.

Two readings I rejected on the way there. First, keeping the drop-everything
rule and forgoing arrival actuals: it is safer to argue for but it throws away
the strongest congestion signal in the file (`arr_taxiin_rwy_mean_30m`
correlates 0.194 with the target against 0.13 for the best count-based feature
in v1), and the mutation test buys the same safety for the cost of one test.
Second, using the arrival's taxi-in only as an airport-hour aggregate rather
than also per stand: the aircraft that last occupied this stand is the aircraft
now leaving it, and its on-block time bounds when a real off-block could have
been recorded, which is exactly the discrimination the placeholder rows need.

## The three readings the assignment asked me to settle

**1. Departures with null AOBT are in the scored population, and I predict them
with the same model.** 1.53 % of the 344,841 ranking departures (5,289 rows)
have no Network Manager join: every `_flt` column is null, so `AOBT_3_flt`,
`LOBT_flt`, `IOBT_flt`, `EOBT_1_flt`, the operator, the market segment, the
wake category and the filed aircraft type are all missing. They are not
droppable and they are not a rounding error — on the july fold they are 2.06 %
of rows and carry 60 % of the squared error under the naive predictor.

I keep them inside one model rather than training a second one. LightGBM splits
on missing values natively, `nm_missing` and `aobt_missing` are explicit
features, and everything those rows still have (airport, stand, runway, take-off
time, scheduled time, destination, aircraft type from the movement file, and the
whole congestion block, which is computed from take-off times and needs no NM
join) is real signal. A separate head for them is climb mechanism (b) and is
measured, not assumed. Under the residual parameterisation these rows have no
offset to sit on, so they get their airport's training-median taxi time as the
offset and the model fits them from there.

**2. Training rows with a negative or extreme target stay in the scored
population and, for v1, stay in training untouched.** The scored population is
whatever the organizer scores; nothing about it is mine to choose. On the
training side the choice is real: July 2025 has 0.018 % of departures at or
below zero seconds (minimum −12 s, a clock artefact) and 0.34 % above 3600 s,
with a maximum of 88,132 s — and that 0.34 % carries 67 % of the squared error.
Clipping the training target would trade tail accuracy for bulk accuracy under a
metric that pays for the tail, so v1 clips nothing. `--train-clip-lo` and
`--train-clip-hi` exist so the trade is measured rather than argued, and the
output guard `--clip-floor` is climb mechanism (d).

**3. `local hour` is airport local wall-clock time, including daylight saving.**
Ten departure airports across six time zones (Istanbul is UTC+3 year-round;
London, Paris, Berlin, Amsterdam, Madrid, Rome and Zurich shift). The bank
structure that drives taxi-out congestion is a local-clock phenomenon, and the
two evaluation months sit on opposite sides of the DST boundary — January 2026
is standard time, July 2026 is summer time — so a UTC hour would smear the
morning bank by an hour between the two halves of the ranking file. It is one
`zoneinfo` conversion per airport and costs nothing. UTC hour is not carried
separately in v1; `d_takeoff_sched` and the congestion counts already localise
the flight in the day's traffic.

## Design

- `features.py::build_features(frame)` — one month frame in (departures and
  arrivals), one row per departure out. 49 features: nine own-flight second
  deltas plus two missingness flags, twelve categoricals (stand and runway are
  joined to the airport, since their codes are only unique within an airport),
  five calendar fields on local time, eighteen ±10/30/60-minute congestion
  counts (all take-offs, same-runway take-offs, arrival landings), a
  queue-ahead count of same-airport departures whose off-block falls between
  this flight's own off-block and its take-off, and the gap to the previous
  take-off on the same runway and at the same airport.
- `run.py` — two passes per fold. Pass one early-stops on the last 15 % of the
  training months by take-off time, never on the validation month. Pass two
  refits every training row for `best_iteration / 0.85` rounds. The submission
  fits all twelve 2025 months and predicts each ranking month from its own
  frame; v1 took its round count from the fold mean times 1.1, and from
  `v1-c6` on it runs its own early stop instead — see the note at the end of
  this file.
- Categoricals are integer codes from a vocabulary fixed on the fitting frame;
  a value unseen in training becomes −1, which LightGBM reads as missing. That
  is the right behaviour for a 2026 stand that did not exist in 2025.

## Known caveat

For `--target log` and `--target asinh` the early-stopping metric is RMSE in the
transformed space, not in seconds, so the iteration count those runs choose is
optimal for the wrong loss. That is inherent to the mechanism being tested and
is noted on the `climb.log` line rather than silently corrected.


## Readings added during the climb (2026-09-05)

**The noac rows get their own model, anchored on the schedule.** A departure
with no `AIRCRAFT_TYPE_mvt` is 0.18 % of july departures and 60 % of v1's
squared error. I rejected two readings before the one I kept. The first was to
treat them as a two-class problem — "the block time is the schedule" or "it is
real" — and predict `q·(take-off − SCHED) + (1 − q)·taxi`: measured, it scores
5,301 s against 4,008 s for a plain regression anchored on the schedule, and an
*oracle* that is told the class outright scores 9,334 s, worse than either.
That last number is what killed the reading: the non-placeholder noac rows are
themselves extreme (validation mean 2,969 s, and the month's single worst row
at 88,132 s is one of them), so no two-class treatment can work — the quantity
to predict is a continuous fraction of `take-off − SCHED`, not a coin flip. The
second rejected reading was to leave them to the main model on flags alone: the
main model reaches 8,941 s on them even with linear leaves, against 4,008 s for
a head trained on 538 rows, because 0.18 % of the data cannot compete for leaves
against two million ordinary flights.

**I steer by populations defined from inputs, not from the outcome.** COMMON.md
requires RMSE split at a 3,600 s target, and every table reports it. But that
split conditions on the answer, so a predictor that simply predicts smaller
numbers scores better on the "bulk" half without being better: the log-target
model looks like the best bulk model in the world at 256.3 s against 312.8 s for
v1, and on the input-defined core — NM join, not noac — it is the *worst* of the
four at 333.6 s against v1's 319.5 s. The split I use to choose mechanisms is
noac / no-NM-not-noac / NM-joined, all three readable before the label.

**Target encodings are fitted leave-one-month-out.** COMMON.md fixes them to
training months; within those months a row must not see its own month's
statistic either, or the encoding is a lookup of the answer. The fallback chain
runs stand → runway → airport, which is not decoration: Munich opened an apron
in 2026 whose stands have no 2025 history at all, and CDG lost two runways
(`docs/shift-2026.md`).

**The full-data fit chooses its own round count.** The assignment says to fit
the twelve 2025 months "with the iteration count from the folds". v1 did that,
taking the mean of the two folds' refit counts times 1.1. The climb then showed
the folds disagreeing by a factor of five once linear leaves were on — `c2`
early-stopped at 69 rounds on july and about 400 on december — so their mean is
a guess rather than a measurement. From `v1-c6` on, the full fit runs the same
two-pass procedure the folds run: early stop on the last 15 % of the twelve
training months by take-off time, then refit on all of them. This uses no
validation month and no ranking data, so it is the same contract, applied to
the data actually being fitted; `--fold-mean-rounds` restores the original
recipe and the fold mean is recorded next to the chosen count in every
`result-*.json`.

## What a phi fitted on the training months actually is (2026-09-05)

`prc.phi_features.PhiModel` estimates `phi` by least squares within a group,
`sum(r·gap) / sum(gap²)`, where `r = BLOCK − AOBT` is the label. Encoding a
*training* row with a model fitted on every training month therefore puts that
row's own label into its own feature. The size of that is exact and worth
writing down rather than hand-waving: row *i* contributes

    gap_i² / (sum(gap²) over the group + shrink)

to its own group's `phi`. Measured on the july fold's six training months,
`E[gap²]` is 5.57e5 s², so with the module's default `shrink` of 2e5 a group
carrying 200 flights keeps own-weight 1.00 — the default is, in this data,
**effectively no shrinkage at all**.

I first wrote that `min_rows = 100` capped the resulting self-weight near 1 %,
by dividing an *average* `gap²` by the group's total. That was wrong, and the
reviewer of exp-006 caught the same thing independently. `gap²` is heavy-tailed,
so a row's own share of its group's `sum(gap²)` is not the average share.
Measured over the 739 airport-operator groups that survive `min_rows` on the
july fold's training months, a single row's self-weight is **median 5.2 %, 90th
percentile 12.3 %, maximum 36.4 %** — the reviewer's figure exactly. Raising
`shrink` to 3e8 only brings the maximum to 28.9 %, because the rows that leak
are precisely the ones with a huge `gap²`, and those same rows dominate the
denominator that the shrinkage is added to.

So `min_rows` does not bound this and shrinkage barely does. An in-sample `phi`
is a real leak, and it is worst on exactly the rows the task is about: a large
`gap` is what makes `phi` matter at all.

The alternative I shipped first — leave-one-month-out on the training side and a
full fit at serve time — has no leak and a different defect: the two sides are
built from different amounts of data. On a six-month fold that is five months
against six, a 20 % difference, and a linear leaf will calibrate its coefficient
to whichever noise level it saw in training. This experiment has already paid
43 s for exactly that mismatch once, in the target encodings, and the `lirf`
worker measures the same shape here — leave-one-month-out costs 22 s on the
six-month fold and nothing on the eleven-month folds.

So there are three constructions and each trades one defect for the other:

| mode | training rows | serving rows | leak | construction |
|---|---|---|---|---|
| `lomo` | K−1 months | K months | none | mismatched |
| `full` | K months | K months | **up to 36 % self-weight** on the rows that matter | matched |
| `kfold` | K−1 months | mean of the K leave-one-out fits | none | matched |

`kfold` is the only one with neither defect, at the cost of K transforms per
served frame. `full` is now excluded outright rather than measured: a mechanism
whose feature carries up to 36 % of a row's own label is not a candidate,
whatever it scores. The desk has held the whole `phi` line until the `lirf`
worker ships a cross-fitted or count-weighted `PhiModel`; `kfold` is measured in
the meantime as information, because it is already the cross-fitted construction
that hold asks for, and it is not promoted until that lands.


## Which folds estimate the board (2026-09-05, prc.data VERSION 2)

The submission trains on all twelve 2025 months, so July 2026 is predicted by a
model that has July and August 2025 in training, and January 2026 by one that has
January and February. The forward `july` fold has no same-season data at all, so
it is not an estimate of the ranking month — it is an estimate of a harder task
than the one being scored. `prc.data` now carries `july_loo` (train the other
eleven months, validate July 2025), which is the matched analogue, and `january`
already was one by construction.

So the board estimate is the pair (`july_loo`, `january`), weighted by how many
ranking departures fall in each month — 55.7 % July 2026, 44.3 % January 2026 —
and `board.py` computes it. `december` is still reported and still useful as a
third reading of a model, but no ranking month is December and it is not part of
the estimate. `july` is kept as the pessimistic bound and because every earlier
line in this file is measured on it.

The distinction matters for reading this file's own history: everything before
this point was steered by the forward `july` fold, which punishes exactly the
mechanisms that need same-season data. That is a candidate explanation for
several of the fold-dependent signs recorded above, and it is testable now that
`july_loo` exists.
