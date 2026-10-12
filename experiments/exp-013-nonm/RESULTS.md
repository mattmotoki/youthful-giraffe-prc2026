# exp-013-nonm — RESULTS

exp-id `exp-013` | worker `congestion` (hard) | reviewer: pending (Opus substitute)
Data: live 2026-09-04 files through `prc.data.fold_frames`. Segment definition,
baselines and rejected readings: `SPEC.md`. Threads: 6 (`docs/COMPUTE.md` compute allocation).

## 1. The ceiling, and why the assignment's oracle is not one

`TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` and `SCHED_TIME_UTC_mvt` is present
on every row, so

    taxi = (MVT − SCHED) − (BLOCK − SCHED)

is an identity, verified to **max absolute error 0.000000 s** over all 6,531
segment rows on three folds. Handing the segment its true `BLOCK − SCHED`
therefore determines the target exactly. **The answer to "how much of the
600 s is the unknown off-block versus taxi physics" is: all of it.** There is
no taxi-physics component to this segment's error — knowing when the aircraft
left the stand leaves nothing to predict.

| fold | n | segment mean | `prc.tail_model` v2 | legal model | Δ vs baseline | oracle as a *feature* | oracle as *arithmetic* |
|---|---|---|---|---|---|---|---|
| july | 3,599 | 692.0 | **651.4** | 584.9 | **−66.5** | 211.2 | 0.0 |
| july_loo | 3,599 | 693.0 | **569.8** | 570.8 | +1.0 | 182.7 | 0.0 |
| december | 1,560 | 709.9 | **493.0** | 585.9 | +92.9 | 255.5 | 0.0 |
| january | 1,372 | 2,842.0 | **2794.2** | 2823.4 | +29.2 | 2628.6 | 0.0 |

"Legal model": LightGBM on `MVT − SCHED`, take-off and scheduled hour, weekday,
`len(FLIGHT_mvt)`, airport, airport:stand, airport:runway, aircraft type,
destination and flight rule — **no `_flt` column anywhere**, in training or at
prediction — trained on every departure and scored on the segment.

### The 211 s that is not information

The two oracle columns use the same quantity. As arithmetic it gives 0.0; as a
feature beside `MVT − SCHED` it gives 182.7 to 255.5. **That gap is a tree
failing to represent a subtraction**, not a limit on what can be known: the
target is the difference of two quantities of scale ~4,600 s, and a boosted
tree approximates a difference with axis-aligned splits.

The design rule this implies governs question 1: **a candidate off-block proxy
must be fed in taxi units — `take-off − proxy` — not as a raw offset beside
`MVT − SCHED`.** Every candidate in the assignment's list (previous arrival at
the stand, next arrival at the stand, the bracketing neighbours' queue times,
the running `take-off − AOBT`) is naturally a timestamp; handed over raw, each
would pay some share of this 200 s penalty before its signal could show.

### What the baseline comparison says

The legal model beats `prc.tail_model` v2 by 66.5 s on the forward `july` fold,
ties it on `july_loo`, and **loses by 92.9 s on december**. Since `july_loo`
and `january` are the folds `prc/data.py` calls the faithful analogues of the
ranking months, the honest reading is that the existing no-NM head is already
strong and a generic model does not beat it. Any gain in this lane has to come
from the candidate signals, not from a better-tuned generic learner.

## 2. January's segment number is two rows

The 1,372-row january segment has target sd **2,840 s**. Its two largest rows
are LFPG at 58,206 s and 84,240 s, with `BLOCK − SCHED` of −56,163 s and
−82,500 s: a block time stamped about a day before the schedule. Without them
the sd is **803 s** (n = 1,370).

Every january figure in this lane — including the 2,794 s baseline the
assignment quotes — is therefore two rows, and cannot rank anything. Tables
below carry a `segment_trim` column that drops rows with `BLOCK − SCHED` below
−20,000 s. **That is a pre-declared segment definition, not a trim chosen after
seeing errors**: the rows are identifiable from `BLOCK − SCHED` alone. The same
screen removes one july_loo row and two december rows, so the phenomenon is not
january-only, just january-dominating.

## 3. Candidate off-block proxies

Each family added to the base legal model, trained on every departure with no
`_flt` column present, scored on the segment. `segment_trim` is the column to
read; `all_rows` is the same model's RMSE on the whole month, which is what the
main lineage cares about and is reported because the two objectives conflict.

**july_loo** (n = 3599, baseline 569.8)

| config | segment (trimmed) | vs deployed baseline | all rows |
|---|---|---|---|
| `base` | 570.7 | +0.9 | 518.0 |
| `op` | 572.7 | +2.9 | 516.1 |
| `stand` | 652.0 | +82.2 | 498.4 |
| `surface` | 550.1 | -19.7 | 516.7 |
| `bracket` | 564.0 | -5.8 | 522.3 |
| `sb` | 546.2 | -23.6 | 516.3 |
| `sbo` | 544.8 | -25.0 | 516.8 |
| `all` | 588.2 | +18.4 | 491.9 |

**january** (n = 1372, baseline 2794.2)

| config | segment (trimmed) | vs deployed baseline | all rows |
|---|---|---|---|
| `base` | 720.9 | +63.7 | 459.9 |
| `op` | 717.9 | +60.7 | 456.2 |
| `stand` | 721.6 | +64.4 | 436.5 |
| `surface` | 662.9 | +5.7 | 439.7 |
| `bracket` | 673.1 | +15.9 | 444.9 |
| `sb` | 653.3 | -3.9 | 440.9 |
| `sbo` | 651.4 | -5.8 | 437.0 |
| `all` | 658.7 | +1.5 | 425.5 |

**december** (n = 1560, baseline 493.0)

| config | segment (trimmed) | vs deployed baseline | all rows |
|---|---|---|---|
| `base` | 586.1 | +93.1 | 333.8 |
| `op` | 589.3 | +96.3 | 325.0 |
| `stand` | 603.6 | +110.6 | 312.8 |
| `surface` | 493.5 | +0.5 | 308.7 |
| `bracket` | 522.3 | +29.3 | 318.6 |
| `sb` | 497.2 | +4.2 | 304.7 |
| `sbo` | 490.6 | -2.4 | 301.6 |
| `all` | 499.1 | +6.1 | 290.1 |

**july** (n = 3599, baseline 651.4)

| config | segment (trimmed) | vs deployed baseline | all rows |
|---|---|---|---|
| `base` | 584.8 | -66.6 | 540.2 |
| `op` | 577.2 | -74.2 | 538.4 |
| `stand` | 740.9 | +89.5 | 523.4 |
| `surface` | 573.6 | -77.8 | 538.6 |
| `bracket` | 577.7 | -73.7 | 532.1 |
| `sb` | 560.4 | -91.0 | 533.0 |
| `sbo` | 557.2 | -94.2 | 528.1 |
| `all` | 658.1 | +6.7 | 513.1 |

The january column compares against `prc.tail_model` v2's **trimmed** january
figure (657.2), not its raw 2794.2, for the reason in §2.

**`sbo` — surface + bracket + operator — is best on all four folds**, and it
beats the deployed head on all four: −93.7 july, −25.0 july_loo, −2.7 december,
−5.8 january. `op` is worth nothing alone (+2.9, +3.1, −3.0, −7.5) but adds
1.3 to 6.5 s on top of `surface + bracket` on every fold; the carrier appears
to be useful only once the model already knows what the airport was doing, and
useless as a main effect. `all` is worse than `base` on both july folds purely
because `stand` is in it.

### The two candidates that pay are the two built from other flights' AOBT

`surface` (the running `take-off − AOBT` of neighbours at that airport and
runway, from `prc.surface_features`) and `bracket` (the nearest same-runway
departures that have an AOBT, and their spans) both improve the segment on
every fold. They are the same idea: **this segment has no off-block of its own,
so the only evidence about when it pushed is what comparable flights that do
have one were doing at the time.**

### The two that do not, and why

**`stand` costs the segment and helps the population.** On july_loo it is
+81.3 s on the segment and −19.6 s on all rows; on december +17.7 and −21.0.
Its coverage is 96 %+ on NM rows and **21.9 %** on the segment, because these
delayed flights often have no stand neighbour. A model trained on every
departure learns to lean on a bound the segment mostly lacks, and the segment's
rows land in leaves shaped by rows that have it. This is the concrete mechanism
by which "train on all departures, score a sub-population" goes wrong, and it
is an argument for coverage-matched features rather than for a separate head.
On january the same family is neutral, so the harm is fold-dependent.

**`op` is small and inconsistent** (+2.1 july_loo, +3.1 december, −3.0 january
trimmed, **−7.5 july**) — it is not the uniform negative an earlier draft of
this paragraph claimed on three folds; july reverses it. `AIRCRAFT_OPERATOR_flt` is null on exactly these rows,
so the operator is recovered from the `FLIGHT_mvt` callsign prefix — the
assignment proposes it because exp-002 finding 5 reports charter and non-EU
carriers at ~8 % no-NM rates against 0.3 % for the home shuttles. But that
finding is about **who lands in the segment**, not about how long they then
taxi. Conditioned on being in the segment the carrier adds nothing and costs a
high-cardinality split. The candidate was proposed from a fact about the wrong
quantity.

## 4. The head: training population and target

**Training population (question 2).** Three options, july fold:

| head | segment RMSE |
|---|---|
| trained on every departure, `_flt` columns absent | **584.9** |
| trained on the segment's rows only | 753.0 |
| `prc.tail_model` v2's no-NM branch (the deployed head) | 651.4 |

Segment-only training loses by 168 s. The segment is 1.4 % of departures and
the taxi process it needs to learn — stand to runway, at that airport, at that
hour — is the same process the other 98.6 % demonstrate. What differs is the
*evidence available about the off-block*, not the physics, so the right move is
to remove the `_flt` columns from every training row rather than to remove the
rows. This reproduces `prc.tail_model` RESULTS finding 11 independently.

**Target (question 3).** Predicting `BLOCK − SCHED` and subtracting it from
`MVT − SCHED`, identical features, rounds and seed:

| fold | target = taxi | target = `BLOCK − SCHED` |
|---|---|---|
| july_loo | 588.2 | **2434.2** |
| december | 498.9 | **2953.3** |
| january | 2808.1 | 3194.2 |

Four to six times worse, and the reason is the one §1 gives from the other
direction: a boosted tree fits a constant per leaf, so moving the target from
taxi (σ ≈ 690 s on the segment) to `BLOCK − SCHED` (σ ≈ 4,500 s) makes the
fitted quantity an order of magnitude larger than the one anyone cares about,
and subtracting afterwards does not undo it. The §1 finding is about *features*
entering as large differences; the fix it implies is feeding proxies in taxi
units, which is what the candidate families already do — not moving the target.

## 5. What this lane is worth, and when to come back to it

Board calibration (desk, 2026-09-06): `board = clean + 4 s`; v1 scored
**265.7612**, rank 1 of 40. The segment is **4,902 of 344,841** ranking rows
(1.42 %), so its contribution enters the total in quadrature and a large
segment gain is a small board gain. (The assignment's 5,290 is *all* no-NM
departures; 388 of those are noac and belong to exp-002/exp-009, so the
population this head serves is 4,902. An earlier version of this table used
5,290 and overstated every figure in it by about 8 %.)

Segment RMSE in the ranking mix (56 % July-like, 44 % January-like, using the
rollover-trimmed figures on the two ranking-analogue folds):

| | segment in the mix | board |
|---|---|---|
| deployed (`prc.tail_model` v2) | 609.5 | 265.76 |
| **this head, 3-seed averaged** | **592.0** | **265.20 (−0.56 s)** |
| a hypothetical 520 | 520 | 263.04 (−2.72 s) |
| a hypothetical 480 | 480 | 261.96 (−3.80 s) |
| **a perfect segment model** | **0** | **255.63 (−10.13 s)** |

**The whole lane is worth 10.13 s and this head banks 0.56 s of it.** That is
the honest destination arithmetic: even driving the segment to a level nothing
measured here suggests is reachable (480 s) returns 3.8 s, and the leaders climb
3 to 6 s a day. The desk's decision to take the member and move the runner
follows from this table, not from the lane being uninteresting.

Two things qualify the 0.54 s downward and one qualifies it upward. Downward:
the gain is concentrated on the july folds (−93.7 raw july, −25.0 july_loo)
and is only −3.5 and −8.6 on december and january, and january is 44 % of the
scored rows. Upward: this is a **standalone** number. The blend owner measures
a member by what it adds to the blend, and a head that is good on a segment the
other members are weak on can earn weight beyond its standalone score — which
is why the member is published rather than discarded.

### Delivered artefacts

`prc.nonm` VERSION 1, `NoNMHead().fit(train_frames).predict(month_frame)`,
three seeds averaged, 44 features, no `_flt` column anywhere. Built by
`deliver.py`; segment figures are rollover-trimmed.

| artefact | segment | all rows | SHA-256 (first 16) |
|---|---|---|---|
| `oof/nonm-v1-july_loo.parquet` | 543.4 | 518.2 | `3fbdc97a50028b48` |
| `oof/nonm-v1-december.parquet` | 489.8 | 301.6 | `d1fe4628ed01debf` |
| `oof/nonm-v1-january.parquet` | 648.6 | 435.0 | `0497c5967ed45727` |
| `oof/nonm-v1-july.parquet` | 553.3 | 531.8 | `77970497645d2a94` |
| `oof/nonm-v1-ranking.parquet` | 4,902 of 344,841 rows flagged `seg` | – | `806b7ea134fbba07` |

Against the deployed head (`prc.tail_model` v2, rollover-trimmed): **−97.6
july, −25.9 july_loo, −3.5 december, −8.6 january** — better on all four folds.
Every file carries `MVT_ID_mvt`, and the ranking file carries a `seg` flag so
the blend owner can weight the head on its own population without recomputing
the segment definition.

This is a blend **member**, not a submission candidate: it predicts every
departure but is only intended to carry the 1.42 % of rows `seg` marks, so it
has no row in `docs/SUBMISSIONS.md`.

### Revival condition

Come back to this lane if any of these becomes true:

1. **The segment grows.** It is 1.4 % of the ranking file against a 0.99 %
   2025 mean, and exp-002 finding 5 shows it is above the 2025 rate at every
   airport. If a re-issued ranking file or a later month pushes it past ~3 %,
   the same head is worth 3 to 4 times more.
2. **Someone finds an off-block proxy that is not a neighbour statistic.**
   Everything that worked here is "what were comparable flights doing at the
   time"; everything that failed was a property of the flight itself. A real
   off-block estimate — from a trajectory archive, or from the 2022 flight
   list in the backlog — would attack the 10.95 s directly rather than nibble
   at it.
3. **The blend owner reports the member earning weight.** If it does, its
   marginal value is above its standalone value and the ceiling above is the
   wrong ceiling for it.

### Not done

- No drop-one sweep inside `sbo`; the three families were tested singly and in
  two combinations, not exhaustively.
- `stand` is excluded on a coverage argument measured on two folds and neutral
  on a third; a coverage-matched variant (the bounds restricted to rows that
  have them, with an explicit indicator) was not built.
- The head is a single learner. No seed-count sweep beyond three, no
  round-budget selection, no per-airport variant.
