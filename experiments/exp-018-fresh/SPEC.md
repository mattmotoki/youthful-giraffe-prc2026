# exp-018 fresh — an independent lineage

**Reading of the assignment.** I am to build the best taxi-out model I can in
about a day from a fresh reading of the data, without looking at what the other
lanes in this repo did, and then compare. I read `assignments/COMMON.md`
(sections 1, 2, Compute, Reporting), `docs/input-rule.md`, `prc/data.py`,
`docs/board/best_per_team.csv` and the parquet files, and nothing else.
Readings I rejected: (a) that "fresh" also excludes the two facts the desk gave
me (per-month neighbour features; the score covers no-NM and null-aircraft-type
rows) — those are data properties, and the assignment states them; (b) that I
should re-derive the input rule myself — `docs/input-rule.md` is on the allowed
list and settles it; (c) that I should hold the ghost class out of the model as
"dirty labels" — the score is RMSE on all 344,841 rows, so a label that is an
artefact is still the thing being scored, and the artefact is the single
largest term in the objective.

Accidental reads: none. (`prc/data.py` docstrings quote exp-017 measurements;
that module is on the allowed list, so I have read them and they are recorded
here: 15 rollover rows carry 17.6 % of training SSE. I did not act on the
exp-017 recipe.)

## What determines taxi time here

The target is `MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` and the ranking file
hands us `MVT_TIME_UTC_mvt`. So this is not a taxi-duration forecast: it is a
**reconstruction of the departure's off-block time** from everything else in
the file. Three facts from my own read of July 2025 (190,713 departures) set
the shape of the problem:

1. **The metric lives in the tail, and the tail is an artefact.** 656 rows
   (0.34 %) have a target above 3600 s and they carry **69 %** of the sum of
   squares about the median. Over all twelve months, 1,488 rows — LIRF
   departures with **no Network Manager join and a null `AIRCRAFT_TYPE_mvt`** —
   carry **37 %** of the total sum of squares on their own, 0.07 % of the rows.
   Their median target is 3,922 s against 948 s globally, and the largest is
   131,167 s. A model that treats them as noise cannot get near 277 s: with a
   constant prediction, the ~388 such rows in the ranking file alone would cost
   about 414 s of RMSE.
2. **That artefact is half reconstructable.** For 48 % of these rows
   `BLOCK_TIME` sits within 6 s of `SCHED_TIME`, so the label is
   `MVT_TIME - SCHED_TIME` to within 6 s, which we can compute. (Corrected
   after review: *exact* equality is only 4.5 % of the class. `BLOCK - SCHED`
   is a flat plateau across -6 to +5 s with a cliff at ±7, so the class is
   real and sharply bounded but the airport is not literally copying the
   schedule field. 53 % is the share under the softer `y / (MVT - SCHED) > 0.5`
   criterion, which is what the head actually uses.) The other 47 % have a real
   off-block and a normal-ish taxi (median 1,139 s). The ratio
   `y / (MVT - SCHED)` is therefore bimodal at ~0.1 and ~1.0, and predicting it
   is a classification problem dressed as a regression.
3. **For the ordinary 98 %, the whole task is the offset between two clocks.**
   `MVT - AOBT_3` alone scores 466 s on July. The error is
   `BLOCK_TIME - AOBT_3_flt`, median 5 s but only 20 % of rows within a minute:
   airport off-block and Network Manager off-block are different measurements.
   They are not out of sync as clocks — for arrivals, `MVT_TIME - ARVT_3_flt`
   is within 60 s on 90 % of rows — so the offset is definitional (push-back
   vs. chocks-off vs. an NM update) and depends on airport, stand, operator and
   how congested the surface is. That is a learnable quantity.

**What I think the leaders at 277 s have that a naive model does not.** Split
the budget: 277 s over 344,841 rows is a sum of squares of 2.6e10. If the ~388
ghost rows were predicted with a constant they would consume 6e10 on their own.
So every team under ~400 s must already be reconstructing the ghost labels from
`MVT - SCHED`, and the teams at 277 s are doing it *selectively* — separating
the "block == sched" half from the rest. After that, 234 s on the ordinary rows
is what is left, and that is an offset model with congestion and stand
geometry, not a taxi-duration model.

## What I build

`base + residual`, where `base = MVT - AOBT_3` when the NM row joins and an
encoded group median otherwise, and LightGBM learns the residual under L2. The
ghost class is pulled out of the main fit and given a ratio model
`y = rho * (MVT - SCHED)` fitted with weight `(MVT-SCHED)^2`, so squared error
in ratio space is squared error in seconds. Features are month-local by
construction (`features.py` never reads a departure's block time or taxi time,
in training or validation, so a training row and a ranking row see the same
definition; verified by rebuilding a month masked and unmasked and diffing).
The families are: NM and schedule time deltas; runway/airport departure and
arrival counts in windows around the take-off; a **surface-queue** count of
departures already off-block per NM and not yet airborne; **live taxi-in times
of arrivals**, which are fully observed in the ranking file and are the only
in-month measurement of how slow the surface is right now; **stand occupancy**
(the previous and next arrival in-block at the same stand, which brackets the
off-block time); and leave-one-month-out smoothed target encodings of both the
taxi time and the offset by stand, runway, stand×runway, operator and type.

## Deviation from COMMON.md worth recording

The information diet forbids reading `docs/LEDGER.md` until step 4, and
COMMON.md asks for the ledger format to be read from that file. I therefore
append the ledger rows for this experiment after step 4, in one batch, rather
than as each number lands. The numbers themselves are in `climb.log` and
`RESULTS.md` as they are measured.

## Desk rules received mid-run (2026-09-05)

* No `git push` from this workstation in any circumstance; commits stay local
  on `main` and the Mac session relays the checkout to origin. Nothing in this
  experiment pushes.
* Uploads remain Matt's. The candidate enters the ready queue by way of a
  one-line row in the exp-012 candidate table in `docs/SUBMISSIONS.md`, with
  the parquet path and its SHA-256. `docs/SUBMISSIONS.md` is not on this
  assignment's forbidden list (`submissions/` the directory is), so I open it
  only to append that row and read nothing else from it before step 4.
