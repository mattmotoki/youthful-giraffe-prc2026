# exp-007-winter — SPEC

worker `winter` | tier hard | reviewer: sol | exp-id `exp-007`
Assignment: `assignments/winter.md`.

## My reading of the assignment, and the readings I rejected

**The reading I took.** The assignment is first an *analysis* task and only then
a modelling task: question 1 decides whether the desk points more workers at
January, so it is answered on saved out-of-fold predictions before anything is
retrained. I read "where the january error lives" as a decomposition of the
**squared** error (SSE share, not row share) of the non-LIRF departures of the
`january` fold's validation month (January 2025, masked exactly like the ranking
file), with December 2025 and July 2025 as controls under the same slicing, so
that a structure that is winter-specific can be told from one that is present in
every month. Everything in questions 2–4 is then aimed at the slice question 1
finds, and the module in question 5 carries only what survived a paired,
airport-day-clustered bootstrap.

**Baseline.** `experiments/exp-001-retro-v1/oof/G1cv-s7-*.parquet` — the latest
retro-v1 line that is (a) guarded, (b) run on all three folds, and (c) the best
of those on january (346.5 s). `H0-s7` is later in `climb.log` by 47 minutes but
is a phi/lirf ablation control, not a better line (january 347.4). I report both
where the choice could matter, and re-baseline if a newer three-fold guarded
line lands. Rejected: `c6-noac`, which exp-006 used — it predates the residual
guard and the phi rebuild, and the assignment says "latest guarded line".

**Rejected reading 1: score the january fold whole.** The assignment names the
non-LIRF slice because exp-006's board arithmetic put LIRF inside the leaders'
band. Scoring the whole fold would let LIRF's 387 s move numbers that are about
the other nine. I report whole-fold RMSE too, but every ranked finding is on the
non-LIRF slice.

**Rejected reading 2: treat "the worst 5 % of airport-days" as a fixed set to
fit on.** The tail worker's 61 % figure is a *description* of where the SSE
sits, computed with labels. A feature that identifies those days must be
computable from the masked frame alone, so question 2 is scored as *separation
of a label-defined target by a label-free candidate*, and question 3 only ever
fits label-free columns. The label-defined day severity is the thing being
predicted, never an input.

**Rejected reading 3: build the de-icing proxy from the row's own span.** "The
share of same-airport departures in the surrounding hour with take-off minus
AOBT above 1800 s" includes the row itself if written naively. `span` is
label-free (both `MVT_TIME_UTC_mvt` and `AOBT_3_flt` survive the mask), so
including the row is not leakage, but it makes the feature a monotone function
of the row's own span on thin hours and destroys its meaning as *day state*. I
exclude the row itself and report the difference.

**Rejected reading 4: answer question 4 with a per-airport-day random effect
fitted on the validation month.** A day effect fitted on the validation month's
labels is the leak the whole design forbids. Question 4's "two-stage day
severity" is therefore: predict the day's median residual from *label-free* day
features, with the stage-1 model fitted on training months only.

## Path decisions I made without asking

- Slice definition: a "row" is a departure (`PHASE_mvt == 'DEP'`) with a
  non-null hidden label. "non-LIRF" is `ADEP_mvt != 'LIRF'`.
- "airport-day" is `(ADEP_mvt, date of MVT_TIME_UTC_mvt in UTC)`. The ranking
  file has no local-time column and UTC is what every other module uses.
- The tail cut stays at 3600 s (COMMON.md's convention) and `no-NM` stays the
  `FLIGHT_ID_mvt` join definition retro-v1 settled on after Sol round 1.
- Bootstrap: paired, resampling airport-days with replacement, 2000 draws,
  matched seeds between arms — the design exp-005 and exp-006 converged on.

## Re-aim from the desk, 2026-09-05, after question 1 landed

Question 1 found that 61 % of the january non-LIRF squared error is two rows of
exp-002's unreachable date-rollover class, and that removing them puts the other
nine at 234 s blended — inside the leaders' 200–240 band. The desk re-aimed the
rest of the assignment:

1. Question 2 stays but stays **short**: January 2026 is harder than any fold can
   show (`docs/shift-2026.md`), so a label-free day-state marker is insurance for
   the ranking month even though the january fold no longer shows a large gap.
2. The question-1 decomposition is then applied to **july at the other nine**
   (249.7 s against the same 200–240 band), by airport, by airport-day, by regime,
   and by whether take-off minus AOBT is itself long; with a statement of what the
   current best out-of-fold line under-predicts and what marks it in the ranking
   frame.
3. Questions 3–5 target whichever of January-insurance or July-structure has the
   larger **measured** upside. The choice and its evidence are written below when
   the measurement exists.
4. The rollover floor arithmetic goes into RESULTS.

**Which of the two I chose for questions 3–5, and why:** written in
`### The choice` below once questions 2 and 2J are measured.

### The choice for questions 3–5

**Day-state features, measured on all three folds.** The evidence, all of it in
`analysis/`:

- Question 2 (`q2_daystate.md`) found strong label-free day-level separation and
  found it in **every** fold, not only january: ranking days inside an
  (airport, fold), the day's median take-off-minus-schedule reaches ρ +0.651 with
  the day's RMSE (AUC 0.868 for the worst 5 % of days), the day's p90 of
  AOBT-minus-LOBT +0.637, and the day's dispersion of take-off-minus-AOBT +0.497.
  Weather is far weaker (ρ 0.19 at best). So the same block is january insurance
  *and* the largest unclaimed july signal. (Those figures are the corrected ones:
  the screen first ranked within the airport alone, which let a seasonal contrast
  count as a day contrast — see RESULTS finding 6.)
- Question 2J (`q2j_july.md`) found that july's remaining error has **no
  unclaimed univariate marker**: the best single label-free NUMERIC column removes 0.73 %
  of the NM-bulk squared error on top of the model (0.83 s of RMSE), and it is
  `span`, which the model already has. (Corrected twice: the screen was not
  centring its bin means, and the claim was later narrowed to the numeric family
  when a categorical airport-by-operator correction turned out to remove 0.95 %.) July's three large slices by squared error
  are no-NM (11.4 %), the NM tail above 3600 s (7.6 %) and the block-on-LOBT class
  (7.5 %) — owned by exp-002 (tail/noac) and exp-006 (phi) respectively. Taking
  one of those would duplicate another worker; the day-state block duplicates
  nobody.
- ~~The one july structure this experiment found that nobody owns is that the
  block-time convention is a **per-flight mixture**, not a per-operator average.
  An oracle that knew each row's component is worth 17.1 s on july.~~ **Refuted
  by my own follow-up the same day** — `analysis/q5_mixture.md` §"the atoms are
  not there". The two "components" are not atoms: on the oracle's own population — NM-joined, non-LIRF,
  |lob| ≥ 300 s — exact `r = 0` holds on **0.56 %** of in-scope july rows and
  exact `r = LOBT − AOBT` on **0.17 %**, and the mass inside a ±60 s window
  (19.52 % and 3.94 %) is just the continuum's local density: it grows smoothly
  with the window (6.34 / 9.70 / 19.52 / 36.99 / 71.07 % at ±5 / 30 / 60 / 120 /
  300 s). (These are the corrected figures; the first version of this paragraph
  quoted an all-airport population.) `r` is a continuous quantity, φ is its conditional
  mean, and exp-006's treatment of it is the right shape. The "oracle" measured
  before that check is an oracle that already knows `r`, not a ceiling on any
  router, and the router built from it loses 117 s to the model.

Rejected: building a july no-NM or NM-tail model (exp-002's lane, and its
measured hedge arithmetic already says the unreachable part of it is worth a
thousandth of a second); and normalising the day statistics **within** the month,
which would erase exactly the January-2026 signal — a month that is bad relative
to 2025 looks ordinary relative to itself. The reference distribution is
therefore fitted on the fold's training months.


## Second re-aim from the desk, 2026-09-05 06:30

Two instructions, both taken:

1. `prc/data.py` is VERSION 2 and adds `FOLDS["july_loo"]` — the other eleven
   months against July 2025. The submission model trains on all twelve, so July
   2026 is predicted with July and August 2025 in training; the forward `july`
   fold has no same-season data and is pessimistic, exactly as `january` is the
   analogue for January 2026. `(july_loo, january)` is the desk's board-analogue
   pair. Both arms now run it and every scored row here reports it.
2. Write the day-state result as a real answer with its revival condition, keep
   `ds_deice_hr` as the one column that beats its control, finish the Sol review,
   report done.

**One thing I did not take on trust.** The first july arm pair came back at
−10.6 s on the core other nine, thirty times what this experiment's own
cross-fitted screen predicted (finding 11), and the two arms fitted for 50 and
107 rounds because adding columns moved the early stop. retro-v1's july round
count swings between 50 and 385 across near-identical lines. So before that
number is written as a column gain, both arms run again at a **declared 300
rounds** with the round selection removed. The control is parameter-free in the
sense that matters: 300 is written on the command line before the run and no
variant of it is screened.


## Answering gpt-5.6-sol round 1

Verdict FAIL, nine blocking findings, **no leakage found in
`prc/day_state_features.py`**. What each one changed:

| # | finding | what I did |
|---|---|---|
| 1 | the `EJU` prefix separates the two rollover rows, so "unreachable" is false | accepted, and priced the rule as a detector: 0 for 9 out of sample, hedge costs +6.5 / +7.5 / 0.0 s. Status is now *unreached*. `analysis/q1_detector.md` |
| 2 | the floor coefficient assumed a uniform one-day error | accepted; it is the root mean square of the three observed residuals, 130.7 s not 147.1 |
| 3 | 234 s and the leaders' 200–240 are not like-for-like | accepted; the claim is now conditional on k = 0 and on fold transfer |
| 4 | `k ≤ 2` and "one submission measures k" do not follow | both withdrawn; a submission measures the *sum* of five things |
| 5 | the AUC label is the fold-global one, the prose said within-group | both are now reported, 0.868 and 0.854 for the top column |
| 6 | runway configuration and the cancellations proxy were not measured as specified | entropy and a modal-runway-change flag added; the cancellations candidate declared **unanswered** because a cancelled flight leaves no row |
| 7 | "no unclaimed univariate marker" is too broad | narrowed to the numeric family, and Sol's categorical probe verified and reported: `ADEP × operator` removes 0.95 % against a shuffled-key control of −0.23 % |
| 8 | the winter-weighting test is unanswered | answered, finding 15: winter-only training costs +15.2 s on the january fold, a ×3 up-weighting is worth ~1.5 s at matched rounds — weight, do not split |
| 9 | the atom test's population differs from the oracle's | rerun on the oracle's own rows (0.56 % / 0.17 %), and the refutation narrowed to point masses |

Non-blocking, all taken: the two rows' predictions span 791–1,137 s not 900–1,300;
759 is *above* july's 588, not below; and tests 5 and 6 in `test_day_state.py` are
not substantive (their mutants pass because the chosen hand value is zero and
because July's thinnest airport-day has 382 departures) — test 4 is substantive,
and the production gates they were meant to cover are present and are covered by
test 1's whole-frame equality instead.


## What changed after round 1, for round 2

Answering round 1 turned up a defect neither of us had looked for, and it moved
every arm number in the experiment. The first paired arms let each arm choose its
own round count by early stopping, and adding 25 columns moves that count (50
against 107 on july, 284 against **96** on january). Re-run with both arms
refitted at a **declared 300 rounds**:

- july's uncontrolled −6.45 s became **+0.26 s** — the whole headline was the
  round draw;
- december and january survived the control in **opposite** directions, −16.62
  and +4.34;
- and on the label-free most-disrupted 5 % of days the block is **−10.79 s on
  december and +72.09 s on january**, both intervals excluding zero.

The mechanism is measured, not guessed: the january loss is four airport-days on
which the block pushed a near-calibrated model 75 to 150 s past the truth, and on
the worst of them three of five `_z` columns sit at or beyond that airport's own
training maximum. So the block was cut to its one **bounded** column,
`ds_deice_hr`, and re-run against the same matched control: december −13.62
[−26.15, −3.01], january −3.92 [−6.81, −1.02], july null, and no slice on any fold
showing harm. That is the shippable line and the second measurement of the
mechanism.

Findings 13 to 16 are all new since round 1. Finding 10's case for shipping the
block as January insurance is withdrawn by finding 13, and finding 12 records the
uncontrolled arms explicitly as numbers not to use.
