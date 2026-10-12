# exp-047-queue — SPEC

Worker `queue`, tier hard, 2026-09-08. Assignment `assignments/queue.md` (recorded rule: the
slot exp-041 freed). Laptop, 2 threads, CPU only, no GPU, no member fit, no upload,
nothing fitted to a board reading (score-feedback rule).

## My reading of the assignment, and the readings I rejected

**The reading I took.** exp-041 measured that the error v10 shares inside an airport ×
10-minute window sits on the **runway** (68.6 / 81.8 / 70.2 % of the shared component)
and that the cell's own observables reach under a tenth of it; its close-out names the
revival as *"an observable that is a runway state and is not already in the shipped
block"*. This lane's job is to find out whether such an observable exists, price what
it reaches, and — if it reaches enough — hand it to `exp-046-surfacefit` as an input.
So the deliverable of step 1 is a **measurement of the shared component**, not a
candidate, and the block is written as an interface (step 3) rather than a model.

**Reading rejected 1 — "the block is whatever the harvest summaries give".** Before
writing a line of it I read what the shipped feature block already contains
(`experiments/exp-018-fresh/code/features.py`). It already has `apt_queue`,
`rwy_queue`, `apt_pushed_30m`, `rwy_pushed_30m`, `rwy_share_1h`, `rwy_arrivals_1h` and
the `w*_rwy_ndistinct` / `w*_rwy_mode_share` / `w*_rwy_mode_changed` configuration
columns — that is, **the queue and the runway configuration are already in the shipped
block**, built from NM's `AOBT_3_flt` and evaluated at the row's own **take-off**
instant. A block that rebuilt them from the observations and reported a gain would be
reporting the learner, not the observation. Only two things about them can be new:

* the **anchor** — the state read where the aircraft *joins* the queue (its off-block)
  rather than where it *leaves* it (its take-off);
* the **clock** — the physical pushback the network saw, instead of NM's off-block,
  and for aircraft that have no NM row at all.

So the block is built over three event sources, priced separately, and the file-clock
source is the **control the observations have to beat**: `F` (the ranking file's own
clocks), `O` (the adsb.lol observations), `H` (observed where heard, file otherwise —
the serving path). Reporting `O` without `F` beside it would credit the harvest for
something the file already gives on ten airports and fourteen months for free.

**Reading rejected 2 — the arrival side from the observations.** An arrival row keeps
every column in the ranking file: its landing time, its on-block time, its taxi-in and
its runway (`docs/input-rule.md`, and read of the frame: `RUNWAY_mvt`, `STAND_mvt`,
`TAXITIME_SEC_mvt` are 100 %, 99.99 %, 100 % populated on arrivals). There is nothing
for an observation to add there, so **only the departure side of the event set changes
between sources**, and "how many arrivals landed in the last k minutes" is a file
column in every arm.

**Reading rejected 3 — the row's own taxi span as a normaliser.** For a matched row
`MVT_TIME − t_ref` *is* `exp-046-surfacefit`'s observed taxi, which is that lane's
feature and very nearly the label. No column of this block is a function of it —
`code/audit.py` asserts it column by column — because a block that carried it would
measure exp-046's clock and not the runway state, and the fold table could not tell
the two apart.

**The reference instant `t_ref`.** The assignment leaves the unmatched row's anchor to
me:

    t_ref = obs_offblock            where the harvest heard the aircraft push back
          | AOBT_3_flt              else, where NM saw it (98.5 % of ranking rows)
          | MVT_TIME − m(apt, hour) else; `m` is the median of `MVT_TIME − AOBT_3`
                                    over the other rows of the same airport and hour
                                    of the same ranking-masked month

Rejected: (a) the take-off instant alone — that is where the shipped block already
reads, so it is kept as the control arm `Fto` and never as the block's own anchor;
(b) `LOBT_flt` / `EOBT_1_flt`, the *planned* off-block — legal, but the queue is a
realised state and the realised clock covers 98.5 % of rows; (c) dropping the rows
with no clock of their own — 37 % of the ranking file, and the population recorded rule gave
this lane. Nothing in the fallback reads a label: `AOBT_3_flt` and `MVT_TIME_UTC_mvt`
are ranking columns and the median is taken over the masked month.

**Self-exclusion.** Every count subtracts the query row's own event exactly, by index,
rather than by an approximation: `state()` takes the index of the row's own departure
event and removes it from each window it would otherwise join. The stand position used
by the geometry columns is the centroid of the *other* rows at the same `STAND_mvt`,
with this row's own contribution removed from the numerator and the denominator.

**What I did not do.** No new base model: the block is an input to a correction of the
shipped head, exactly as exp-046's is, so every mask, the noac hold and the 388 noac
rows stay as they are. Nothing on LTFM's observations: adsb.lol receives none.

## Method

* Base: fold-side **v11** = `v10 + 0.217·(stack-5 − v10)`, rebuilt by arithmetic from
  gated vectors and gated on the published clean board 256.91 (`code/base.py`).
  From the desk's message of 2026-09-08 14:50 UTC (tail rule) the head is **v12**, v11 with
  only its K2 member replaced; `code/base.v12_vectors` builds its fold side by
  `exp-035` §26a's own two lines and gates it three ways (RESULTS §6a). **Step 1 is
  reported on `r11` and step 2 on `r12`**: step 1 is a label-side diagnostic of which
  residual structure the surface observations reach, it was measured before the head
  moved, and re-basing it would not change what it diagnosed. Anything served is on
  `r12`.
* Estimator: exp-041's self-excluded cross-product, imported from
  `experiments/exp-041-cell/code/base.py`, at airport × 10-minute grain.
* The share removed is `1 − after/before` with a **day-clustered interval on the share
  itself**, both ends rebuilt from the same resampled days, because that is the
  quantity `assignments/queue.md` gates ("a third on any airport with a day-clustered
  interval clear of a tenth").
* **That statistic over-credits any family whose correction for one row is built from
  the other rows of the same cell** (FALLBACK review round 1, blocking 1): row *i*'s
  residual enters row *j*'s correction through the self-excluded mean, so `E[r_i h_j]`
  carries a term in `E[r_i r_i]`, which is idiosyncratic variance and not shared
  component. Two corrected readings therefore stand beside it on every table:
  `(2A − B) / before`, the held-out MSE the correction earns in units of the shared
  component, which forms no cross-row product; and a **split half**, each row's
  neighbourhood built from the other half's clocks with the cross-product read inside
  (cell, half), which is a lower bound because it also halves the source density. The
  `own` arm, whose correction reads the row itself and so cannot contaminate a pair,
  is the control that fixes the scale. RESULTS §2 carries the three-way table.
* **Outage days** (desk, 2026-09-08 14:45 UTC): a declared airport-day whose feeder was
  degraded is no-clock for the class boundary and out of the receipt denominator. It is
  a table and not a threshold, because a threshold also flags an airport whose low
  reception is normal; undeclared anomalies are warned about in every build record.
* Cross fit: 5 contiguous day blocks, `HistGradientBoostingRegressor(300, depth 3,
  lr 0.05)` — exp-041 T7's learner and budget, unchanged, so the two experiments'
  numbers are comparable.
* **Step 2's rotation and the desk's three rules for the fold tables** (2026-09-08
  16:50 and 17:05 UTC). The head is fitted on the fold panels the scored month is not
  in, so nothing about the scored month reaches it.
  1. *The with-clock head is fitted and priced on matched rows only* (`--rows
     matched`). Fitting it on all rows and reading it on the matched ones is a
     different head: the unmatched rows carry an imputed observation column and pull
     its coefficient toward the imputed value.
  2. *An absent observation is never a feature.* The ridge's missingness-indicator
     columns are gone and the GBM path refuses to run without them, because the 2025
     feeder network is a different instrument — EDDM 0 % over 2025-07-01..15 rising to
     99 from the 19th, LEMD 55 % against 99.4 in january, EGLL 12 % rising to 65 — and
     an indicator lets a head read "no receiver here" as a property of these flights
     and carry it to a ranking month where the receiver exists.
  3. *The fold gain is read per airport and projected to the ranking months' own
     reach*, per airport-**month**, which is exp-043 step 4's arithmetic with the
     airport and month dimensions added. The ranking file gives each airport-month's
     rows and that month's own census gives its matched count, so every ranking row
     that is not matched is in the no-clock class — including all of LTFM and LIRF's
     dark 2026-07. The projection is the board endpoint the advance clause is gated on;
     the pooled fold gain stands beside it, and a leave-one-airport-out sits beside
     both, because one airport can carry a projection.
  Rule 1 constrains the with-clock head only, so the no-clock head is also read with
  `--fit-rows all`: `Nf_*` is defined identically on both classes (it needs no
  observation at all), so fitting it on every row of the fitting month estimates the
  same coefficients from twice the data and it is then priced only where it serves.
* Every feature is computed from the **ranking-masked** month frame
  (`prc.data.mask_like_ranking`) plus the harvest summaries. `BLOCK_TIME_UTC_mvt` and
  `TAXITIME_SEC_mvt` of departures are never read on either side.
