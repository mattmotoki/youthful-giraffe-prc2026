# exp-018 fresh — COMPARISON

Written after step 4 of the assignment, i.e. after reading `docs/SUMMARY.md`,
`docs/LEDGER.md` and `experiments/exp-001-retro-v1/RESULTS.md` for the first
time. Everything above this file in `experiments/exp-018-fresh/` was produced
without them.

## 0. The headline

| | board est., all rows | board est., clean | july_loo | january | december |
|---|---|---|---|---|---|
| repo head (exp-016 stack-3, blend-3 members) | 317.22 | 267.15 | 295.88 | 342.17 (clean 225.88) | 214.37 |
| **exp-018 fresh** (one model, two heads, **1 seed**) | **316.48** | **265.78** | 293.65 | 343.06 (clean 225.90) | 207.66 |
| **flat 50-50 per-row blend of the two** | **312.58** | **261.38** | 288.82 | 340.12 (clean 222.09) | 207.87 |

Weights `0.5571 july_loo / 0.4429 january`, `prc.data.rmse_clean`, the desk's
convention. The 0.5 weight is the **optimum** of a 0.2-to-0.7 sweep (312.58 at
0.5 against 312.68 at 0.6), so nothing here is fitted to the folds. Residual
correlation between the two lineages is 0.920 on july_loo, 0.971 on january,
0.941 on december.

So the perspective rotation returns two things: a second lineage that is level
with the repo's head on its own, and **5.8 s of clean board from putting the
two together with no tuning at all**. Caveat on the comparison: the repo's OOF
is a rotated stacker (fit on two folds, scored on the third) and mine is a
plain fold model, which if anything flatters theirs; mine is one configuration
at **one** seed against their five-member five-seed bag (the *candidate* is two
seeds, but the fold line and therefore this estimate are one — review round 1,
finding 3); and the exp-018 configuration was partly selected on these folds
(RESULTS §3), worth roughly 2 to 4 s of the standalone row. **The blend row is
not affected by that**: it is an average of two fixed OOF sets with no free
parameter.

A day-clustered paired bootstrap of the clean board (400 resamples,
`code/blend_with_repo.py`) puts the blend at **−5.75 s [−8.65, −2.44]** against
the repo head and **−4.35 s [−7.83, −1.45]** against exp-018 alone, P(worse)
= 0.000 in both directions. Both members' OOFs are fixed bytes, so member seed
variance is held at zero by construction and part of the gain is variance
reduction that a second exp-018 seed would also buy.

## 1. Where the two lineages agree, having never met

These are the same conclusions reached twice from the same parquet files:

* **The null-`AIRCRAFT_TYPE_mvt` class is the largest single lever.** 388
  ranking rows, 383 at LIRF; the recorded block time is within 6 s of the
  schedule on about half of them so the label is `MVT − SCHED`. The repo calls
  it *noac* (exp-002), I called it *ghost*. **The two measurements agree to
  0.1 pp**: their placeholder share is 47.9 % and mine is 47.8 % under the same
  `|BLOCK − SCHED| <= 6 s` criterion. This file previously reported "theirs
  47.9 %, mine 53 %" as a difference between the lineages; that was comparing
  their hard criterion against the softer `ratio > 0.5` one my head uses, and
  the disagreement did not exist (review round 1, finding 1).
* **`MVT − AOBT_3` on the non-noac tail scores 3960 s.** Their exp-002 anatomy
  row and my `tail.py` produce the same number to the digit, from different
  code. That is a useful cross-check on both.
* **Stand-occupancy bounds are features, not clips.** exp-003's `q6_bounds`
  (valid on 74 % of the rows it fires on) and my Appendix A (violated on 11.5 %
  of July departures, median slack 5,045 s) are the same finding.
* **Model capacity is not a lever.** exp-011's seed/feature work and my
  127/255/511-leaf row (within 0.9 s) agree; so does the rounds curve turning
  at 500.

## 2. What I did that the repo has not

Four things, in the order I would test them.

1. **A symmetric clip on the *training* residual.** The repo's clipping is all
   prediction-side — the `residual guard` clips the *implied* `BLOCK − AOBT`
   after the fact (exp-001 c8, exp-006 guard-v2), and `prc.data`'s
   `clean_training_labels(floor=…)` is a one-sided floor that the ledger says
   has never been enabled. Clipping the *training target* of the residual model
   is a different operation and it is the largest single main-model number I
   found: **+7.1 s of july bulk** at ±3600 (281.05 → 273.92, ablation `E`) and
   another ~2.5 s of total at ±6000 (312.30 → 309.82, `X6`). It is a sweet
   spot, not a monotone knob — ±1200 is *worse* than no clip at all (288.30).
   It should compose with the guard rather than compete with it: one bounds
   what the model learns from, the other bounds what it emits.
2. **A schedule-placeholder head over the whole population, not just noac.**
   The repo's placeholder machinery is confined to the noac rows
   (`prc/tail_features.py`: "candidate observables for p(placeholder) **on the
   noac rows**"; `PLACEHOLDER_TOL` lives inside `prc.tail_model`'s noac head).
   But `BLOCK == SCHED` to the second holds on **4.67 % of all 2025
   departures** — 97,280 rows, 96,559 of them *not* noac — very unevenly by
   airport (LIRF 18.1 %, LEBL 5.6 %, LTFM 5.1 %, EDDM 5.0 %, EGLL 4.9 % against
   EDDF 0.75 %, LSZH 0.75 %, EHAM 0.79 %). On those rows `MVT − AOBT_3` has
   RMSE 845 s against 442 s elsewhere and they carry **13.5 %** of that
   baseline's squared error at 4.1 % of the rows. A posterior mixture
   `p·(MVT−SCHED) + (1−p)·model` with a LightGBM classifier (AUC 0.868 out of
   month) is worth 1.6 s on my july bulk — modest on its own, but **33 % of my
   non-noac tail rows live in this class** against 4 % of the population, so it
   is a tail instrument as much as a bulk one.
3. **Two negatives on the noac head that the ledger does not contain.**
   * *A posterior mixture is much worse than a ratio model there*: 8,308 s
     against 4,013 s on the july fold's 336 noac rows. A hard `|y − ms| ≤ 6 s`
     label puts the five largest July rows (ratios 1.05 to 1.19 — near one but
     not equal) on the "not placeholder" side and then the class mean drags
     them from 87,000 s to 3,206 s. The repo's head is a mixture with bands and
     is clearly better engineered than my first attempt, but this is the
     failure mode to check it against; the safe form is a ratio target
     `y / (MVT−SCHED)` fitted with weight `(MVT−SCHED)²`, which has no cliff.
     Softening the label to `ratio > 0.5` and using a per-row second component
     recovers it (four-fold mean 3,396 against 3,438 for the plain ratio).
   * *Widening the noac head's training population is poison.* Adding every
     departure with `MVT − SCHED > 7200 s` takes the july fold's noac training
     set from 527 to 16,162 rows and the noac RMSE from 4,013 to 8,569; the
     four-fold means are 3,438 (noac only) against 8,285. Even restricting the
     addition to LIRF costs 1,300 s. Worth knowing before anyone tries to fix
     the head's small-sample problem by relaxing its definition.
4. **Three label-free, in-month feature forms the repo's blocks do not have in
   this shape.** All three need no target statistics, so they work on stands
   and periods unseen in training — which is exactly the case exp-003 measured
   as *unimproved* by per-stand target statistics ("no gain on stands unseen in
   training, 240 vs 246").
   * `stand_taxiin_med`: the median **arrival taxi-in** at the same stand,
     computed inside the month. A distance-from-runway measurement that costs
     no labels.
   * `r_prev10/30_apt`, `r_prev10_rwy`, `r_prev5_stand`: a rolling mean of
     `MVT − AOBT_3` over the *previous k departures*. `prc.tail_features` has
     `apt_day_med_mvt_aobt` and `apt_hour_med_mvt_aobt`, which are day and hour
     aggregates; the rolling form is causal within the day and tracks a
     deteriorating surface hour by hour rather than by bucket.
   * `apt_queue` / `rwy_queue`: a count of departures that are off-block per NM
     and not yet airborne at this take-off. `prc.surface_features` has
     `taxiing_at_push`/`taxiing_at_off` and a `queue` family, so this one may
     be a duplicate — it is the least likely of the three to be new.

## 3. What the repo did that I did not

1. **`phi`.** `BLOCK = AOBT + φ·(LOBT − AOBT)` with φ per airport×operator
   (exp-006) is a continuous generalisation of my discrete anchor mixture and
   it is better founded than what I built. **This is my clearest miss, and the
   way I missed it is instructive**: I *did* test `LOBT` as an anchor, found
   that after removing the rows already matching `SCHED` it covers 2.1 % of
   rows and 1.8 % of the baseline's squared error with a *below-average*
   residual, and rejected it (Appendix A). That screen was population-wide.
   The repo's is regime-conditional — inside the late-AOBT tail, `MVT − LOBT`
   scores 1,562 s against `MVT − AOBT`'s 3,960 s. **A population-wide screen
   cannot see a regime-local anchor**, and I should have conditioned before
   killing it. My non-noac tail is 537 July rows at 2,616 s; theirs is the same
   population handled properly.
2. **The date-rollover class.** `y ≈ 86,400 + a normal taxi`, undetectable when
   `AOBT_3` carries no day shift, one such row a shared ≈147 s floor for every
   team (exp-002 rollover, exp-020 census). I never separated it: my january
   fold reads 343.06 all-rows and 225.90 clean, so **117 s of that fold is two
   rows whose class I had not identified**. I wrote in RESULTS §3 that "any
   further work should go at the January tail"; the repo's answer is that most
   of that tail is not reachable by anyone and the rest is already handled.
3. **Weather.** ERA5 as legal external data (exp-005), and the finding that it
   is worth 4-8× more once the actual-time columns are removed because those
   already carry it. I used no external data at all.
4. **Measurement hygiene I did not practise**: seed averaging and seed-spread
   reporting, guarded controls, day-clustered paired bootstraps on an
   input-defined core, provenance/rebuildability discipline, exposure audits of
   the ranking months, the clean/all-rows convention itself.
5. **The board back-out** (leaders' bulk 186-233 s), which is the instrument
   that tells you whether a bulk number is worth chasing. I estimated my board
   position by folding two fold numbers together and left it there.
6. Covariate shift, the LFPG 27R/09L retirement, retired LIRF stands, the
   near-duplicate noise floor — a whole robustness lane I have nothing to say
   about.

## 4. Every idea of mine the repo has not measured, for dispatch

Ranked by expected board value. The first is the one I would run tonight.

| # | idea | evidence I have | cost |
|---|---|---|---|
| 1 | **exp-018 as a blend member.** Flat 50-50 with stack-3 | 267.15 → **261.38 clean**, 0.5 is the sweep optimum, residual corr 0.92-0.97 | zero — both OOFs exist; a candidate is a per-row average |
| 2 | **Training-residual clip** at ±6000 s on every residual-target member | +7.1 s july bulk at ±3600, +2.5 s more at ±6000; ±1200 is worse than none | one refit per member |
| 3 | **Population-wide schedule head** on top of any member | 4.67 % of rows, 13.5 % of baseline SSE, classifier AUC 0.868, +1.6 s july bulk standalone | one classifier fit |
| 4 | **Ghost/noac head structure check**: ratio-with-`ms²`-weight vs the current banded mixture, and the "do not widen the population" result | ratio 4,013 vs mixture 8,308 (july); widening 3,438 → 8,285 | cheap, ~500-row fits |
| 5 | **`stand_taxiin_med`** and the rolling `MVT − AOBT` forms | not measured here in isolation | one feature-block ablation |
| 6 | **`phi` × my heads** — the reverse direction: take the repo's φ and put it under my SCHED head and clip | untested | one refit |

## 5. The one thing I would tell the desk

The two lineages disagree in the right places. My bulk is better on december
(207.66 against 214.37) and july_loo (293.65 against 295.88); their january
tail is better than mine in the sense that they *know what those rows are*.
The flat blend takes the better half of each and is worth more than either
lane's next increment of tuning: 5.8 s of clean board for the cost of an
average. If the desk only acts on one line of this file, it is that one.
