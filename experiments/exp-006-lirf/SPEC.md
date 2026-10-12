# exp-006-lirf — SPEC

worker `lirf` | tier hard | reviewer sol | assignment `assignments/lirf.md`

## My reading of the assignment, and the readings I rejected

**The reading I took.** The assignment names LIRF as "the airport that decides
the score" and points at four statistics as evidence (1658 s v1 RMSE, 98.7 % of
noac rows, the densest NM tail, a 737/751 s dispersion of take-off − AOBT). It
then asks, in question 1, where that dispersion lives. I read question 1 as a
genuine question rather than as a framing to confirm, and the first thing I did
was try to break each of the four statistics. Three survive; the dispersion one
does not (it is a single date-rollover row in each July, `analysis/q1_july_regime.md`).
I therefore treat the assignment's *destination* — get LIRF's RMSE down, because
it is 50.5 % of the July fold's squared error at 7.8 % of the ranking rows — as
correct and its *diagnosis* of where LIRF's difficulty comes from as something
to re-derive. Everything downstream is measured against the residual
`r = BLOCK_TIME_UTC_mvt − AOBT_3_flt`, which is what the baseline model fits and
what carries the seasonal signal, not against `MVT − AOBT`.

**Rejected reading A: "the 751 s dispersion is the regime; find the subpopulation
that produces it."** Rejected after the first trim: two rows in July 2025 and
three in July 2026 produce it, and their `BLOCK − AOBT` is ordinary. Splitting
by hour/stand/runway/operator in search of a 751 s subpopulation would have
found nothing and cost a day.

**Rejected reading B: "LIRF's problem is the tail (noac + NM tail); the core is
fine."** The error decomposition says July is three roughly equal thirds — noac
47 %, NM tail 26 %, core 27 % — and the LIRF *core* is 456 s in July against
270 s in December at the same airport. The core is a first-class target here,
not a rounding error, so the plan spends effort on it.

**Rejected reading C: "build one LIRF-only model and ship it."** Question 4 asks
for LIRF-only versus global-with-interactions. retro-v1's finding that the
`july` fold has six training months and therefore punishes data-hungry
mechanisms applies with full force to a model fitted on 15 k rows/month; a
LIRF-only model is the most data-hungry thing in the assignment. Both are built
and compared on all three folds, and a mechanism is only promoted if december
and january agree with july.

**Baseline.** retro-v1's best climb line is `c6-noac`, not the latest `c9-te`
(c9 added target encodings and lost 43 s on july; retro-v1's own conclusion is
that encodings are off in the shipped lineage). I use `c6-noac` OOF as the
per-row baseline: LIRF RMSE 864.6 (july) / 371.1 (december) / 387.1 (january).

**Metric.** RMSE on LIRF departures per fold, plus the whole-fold RMSE obtained
by substituting my LIRF predictions into the baseline's OOF for the other nine
airports. Every table also carries the COMMON.md split (overall, ≤3600, >3600,
no-NM) and the input-defined split (core / noac / NM-tail-screened) that
retro-v1 showed is the honest one.

## Plan

1. Q1 regime anatomy — done, `analysis/q1_july_regime.md`.
2. Q2 the NM tail: is `BLOCK` on LOBT or on AOBT, and is the choice predictable?
3. Q3 the noac rows: LIRF-specific signals (per-day noac count, neighbour feed
   state, stand re-occupation, the inbound airframe, the previous departure).
4. Q4 LIRF-only model vs global with LIRF interactions; the retired 5xx stands.
5. Q5 hand back `prc/lirf_features.py`.

## Compute

16 threads, no GPU, everything over two minutes detached with a log and a pid.
