# exp-047-queue — the runway state from the surface observations

Worker `queue`, tier hard, 2026-09-08. Assignment `assignments/queue.md` (recorded rule: the slot
exp-041 freed). Laptop, 2 threads, CPU only, no GPU, no member fit, no upload, nothing
fitted to a board reading (score-feedback rule). The reading taken and the readings rejected are at the
top of `SPEC.md`. Every number below is written by a script in `code/` into
`analysis/*.json`; the tables in §9 are rendered from those records by `code/tables.py`
and are not typed.

**Status: all four steps answered.** The harvest delivered 2025-01, 2025-07, 2025-12,
2026-01 and 2026-07 during 2026-09-08, so step 1's ten-airport reading, step 2's
three-fold rotation, step 3's interface and served candidate and step 4's ledger rows are
all measured rather than pending. §5 keeps the readings taken while the harvest was still
running, because they are what the pilot week could see and §7 is what replaced them.

---

## 0. The answer, in six lines

**Step 2 passes at the reach grain, and the candidate serves three cells of nineteen —
and it is on the board at 246.6571 as v17.**
*(This section was written for v14, which served two cells on v13's bytes. Under the
desk's lit-fold refinement of 2026-09-09 the served set is three cells on v16's bytes and
72,943 rows; that file went up as **v17** (`4de323fc…`, 246.6571, −0.0482 s on v16, a
transfer of 0.70× inside its fold interval — §7v). §7r to §7v carry the current file; the
paragraphs below keep v14's numbers where they describe how the endpoint was reached.
§7w reads surfacefit's reach spec from this side: the third pass moves the reach GRAIN,
not a column, so it is the next thing that re-prices this file.)*
The correction is the two neighbourhood families together (`N` = `Nf_*` file clock +
`No_*` observed clock) on the **no-clock** class, against fold-side **v12**, with the
head fitted on the fold panels it is not scored on. Projected to the ranking months' own
reach — per airport-**month** and per **reach class**, the desk's endpoint repaired at
the grain FALLBACK round 3 asked for, and at ONE FLOOR since 2026-09-09 (§7h) — it
reads **+286 s² [+178, +401]** on july_loo, **+218 [+31, +454]** on december and
**+154 [+79, +233]** on january: **every fold interval clear of zero at full amplitude
on the population served**, which is the assignment's advance clause. *(Those three
numbers read +741 [+414, +1,868] / +225 [+37, +456] / +180 [+79, +306] until the desk's
one-floor rule of 2026-09-09 stopped the projection pricing cells the hold refuses to
serve; ledger CORRECTION 6. The candidate's bytes are unchanged and its gate still
passes.)*

**Two things the repair changed, and one claim it withdrew.** The leave-one-airport-out
column now carries an interval off the same draws, and it says that the earlier sentence
*"every fold still clear with EDDF dropped"* was **true on july and january and false on
december**: without EDDF, july is **+145 [+50, +250]**, january is **+94 [+37, +150]**,
and december is **+47 [−71, +170]**, which crosses zero. (Read at one floor; the same
sentence held at the two-floor reading, where the numbers were +285 [+195, +387] with
LIRF dropped as july's worst case, +101 [+44, +163] and +54 [−64, +178].)
Second, reading the gain per reach class exposed that july's total rested on one cell —
LIRF's *unreached* rows, +449 s² of the then +741, priced on **141** fold rows and
applied to 17,651 ranking rows, a 125-fold extrapolation. That cell is what the
one-floor rule removed from the projection altogether (§7h). The desk's rule of
2026-09-09 (the LFPG rule) holds every cell whose own fold interval is not clear of zero
above, or which is priced on under 500 rows in **any** fold: **2 of 19 cells survived —
EDDF/reached and LTFM/unreached — 63,862 of the class's 154,358 ranking rows** (and, under
the lit-fold refinement that followed, **3 of 19 and 72,943 rows**, §7r), and §7c
states the projected gain forgone per held cell (**+40 s² of july's 286, +14 of
december's 218, +58 of january's 154**).

**What the file is worth.** The base is **v13**, `exp-046-surfacefit`'s with-clock
candidate at board 248.4818, which is **bit-identical to v12 on all 154,358 rows this
lane owns**, so the correction is unchanged in substance and the two updates are disjoint
by construction. Priced in the board's own currency, cross-fitted by fold at the ranking
shares 0.5571 / 0.4429, the held file is **clean +0.2231 s** (256.4643 → 256.2412) and
**all-row +0.1854 s** (308.5977 → 308.4122); without the hold it was +0.3755 and +0.3276.
**And the hold turned december's own read
positive**: on the rows the file moves, december is **+372 s² [+143, +619]**, and all-row
**+54 [+24, +91]** — where the unheld file straddled zero at −5 [−128, +112] and was
recorded as the transfer risk. §7, §7a, §7b, §7c and T7. **It went up as v14 on
2026-09-09 at 00:03 UTC and scored 248.2904 — −0.1914 s of board on +0.1854 s of all-row
fold price, a transfer of 1.03×, rank 1 (§7l).** Nothing in the record is revised on that
reading (score-feedback rule).

**Underneath it: the revival exp-041 named exists, and it is the neighbours' clock — but
it is two different clocks for the two populations, and only a whole fold month could
show that.** 2025-12 landed complete on 2026-09-08 (nine airports, 31 days, one rule
`d4c59868…`), so the question could be asked of 139,848 covered rows instead of one
airport's week. On the assignment's own learner and statistic:

| | rows | `Nf` file clock | `No` observed clock | `N` both | `own` clock (exp-046's) |
|---|---:|---:|---:|---:|---:|
| **with a clock** | 95,337 | +39 s² | **+1,065 [+684, +1,509]** | +1,284 | +7,668 |
| **no clock** — this lane's | 44,511 | **+510 [+99, +1,067]** | +19 [−440, +405] | **+322 [+27, +585]** | +5 |

Read that table twice. On the rows that **have** a clock the observed neighbourhood is
worth **+1,065 s²** per row and removes **42.5 % [35.5, 50.1]** of the shared component;
beside `exp-046-surfacefit`'s own-clock feature it still adds **+842 s²** (`own`
+7,668 → `own_No` +8,510), and the whole block reaches **52.4 %** of the shared
component there for **+1,636 s²**. That population is served by that lane, and this
lane hands it those columns as an interface (§6). **That +842 s² refinement does not
survive a month change**: on the rotation `own_No` is *below* `own` on december
(+630 against +881 projected), and §7 and T7c carry the correction — the with-clock
class is served by that lane's own feature and this lane's columns add nothing there.

On the rows that have **no** clock — the population the desk's 14:10 UTC division gives
this lane — **the observed neighbourhood is worth nothing**: +19 s² with an interval
from −440 to +405. What pays there is the **file-clock** neighbourhood, +510 s²
[+99, +1,067], and the **stand-to-runway geometry**, +101 s² [+31, +168]; the two
families together are **+322 s² [+27, +585]**.

**The pilot's EHAM week said the opposite, and it was a 94 %-receipt artefact.** On
those 287 no-clock rows the observed neighbourhood read +583 s² and 39.7 % — but EHAM is
heard at 94 %, so a row without a clock of its own still sat in a cell with three or
four heard neighbours. Over nine airports the no-clock rows are not like that: they are
concentrated at LFPG (10.1 % of them reached by any heard neighbour), LIRF (58.2 %) and
EDDF (66.0 %), and the estimator has nothing to average. **A week at one easy airport
could not have shown this**, which is what §5's discount 2 warned and what the month
now settles.

**And the column that pays needs no observation at all.** `Nf_n_res_cell` is the
self-excluded cell mean of the other departures' `MVT_TIME_UTC_mvt − AOBT_3_flt` **minus
v12's prediction for those departures** — three ranking columns and our own served file,
no adsb.lol anywhere. It is not in the shipped block, and it could not be: `w600_r_mean`
(`exp-018-fresh/code/window_features.py`) is the airport × 600 s bucket mean of the raw
`MVT_TIME − AOBT_3`, not of its **residual against the model**, and it is not
self-excluded and not at the runway × cell grain. So the servable population is not the
harvested one — it is **every ranking row, LTFM and LFPG included**.

**The queue count itself is not the revival.** Read from the file's own clocks at the
**off-block** anchor — which is new, the shipped block reads every count at the
take-off instant — the queue, the runway configuration, the realised runway taxi and
the arrival stream together remove **+2.6 % / +0.0 % / −2.0 %** of the shared component
on july / december / january over all ten airports, against a +2 h placebo floor of
+0.6 %. The anchor is not a relabelling (the same state at the two instants correlates
0.77, T5 D), and it still reaches nothing. Read from the observations on the EHAM week
the state block reaches **+4.9 %**. **This half of the assignment's block FAILS its
gate**, and the diagnosis is in §3.

**But the share statistic over-credits this family, and the corrected reading is
smaller** (FALLBACK review round 1, blocking 1; T2b). `1 − after/before` reads the
cross-product over the pairs of a cell, and the correction for row *j* is built from
the other rows of that same cell — row *i* among them — so `E[r_i h_j]` carries a term
in `E[r_i r_i]`, which is idiosyncratic variance and not shared component. Two
corrected readings, both in the tables beside the first:

* **the contract-consistent share, `(2A − B) / before`** — the held-out MSE the
  correction actually earns, in units of the shared component, and free of the
  over-credit because `2A − B` never forms a cross-row product: **38.5 %
  [26.4, 49.6]** all rows, **31.5 % [−23.8, 77.0]** on the rows with no clock;
* **the split half** — each row's neighbourhood built from the other half's clocks and
  the cross-product read inside (cell, half), where no pair's correction has seen
  either member: **16.5 % [0.2, 30.9]** all rows, **14.6 % [−9.6, 42.7]** no clock.
  This is a lower bound, because it also halves the source density.

The control that shows the split half is not simply destroying signal is the **`own`
arm**, whose correction reads the row itself and cannot contaminate a pair: it moves
**63.7 % → 65.6 %** and its `2A − B` is identical to the last decimal. The naive
statistic at 50 % receipt reads 50.5 % where the split half at the same density reads
16.5 %, and that gap is the over-credit.

**So the gate is passed on the all-row reading and is a MAYBE on the population this
lane serves.** The assignment's condition is a third of the shared component on any
airport with a day-clustered interval clear of a tenth. On EHAM, all rows: 73.1 %
[57.9, 88.9] on the literal statistic and 38.5 % [26.4, 49.6] on the contract-consistent
one — past a third either way. On the **287 no-clock rows** it is 39.7 % [26.2, 67.5]
literal, 31.5 % [−23.8, +77.0] contract-consistent and 14.6 % split-half: the point
estimate is at or near a third and no corrected interval is clear of a tenth. What is
**not** in doubt on that population is the held-out MSE itself, which is not a
cross-row statistic: **+583 s² [−441, +1,428]** per row, positive in the point estimate
on 287 rows and permutation-clean. Thinning EHAM's clocks at random gives the curve an
airport heard at rate *r* would read on that population, on the literal statistic:
**−0.4 % at 16 %, 19.5 % at 30 %, 37.6 % at 50 %, 46.8 % at 70 %, 52.0 % at 90 %**. The
ordering that curve gives is what step 2 has to confirm on the harvested months; a week
of one airport cannot close it.

**Three placebos, and only the third tests the arm that pays.** Shifting every source
event by +2 h moves each neighbour's taxi by the same constant, which a linear head
absorbs into its intercept; it is the right placebo for the counts (it takes july's
+2.8 % to +1.0 %) and no test at all for the neighbourhood. Dealing the **observed**
off-block instants out to other aircraft inside each airport-day does bite the observed
arm — on the EHAM week it takes it from +73.1 % to −7.4 % and from +818 s² to −193 s² —
but it leaves `Nf_*` untouched, so it is a *control* for the file-clock family and not
a test of it. Since december puts **+510 s²** on that family alone, it needed its own:
`--permute-file` deals the `AOBT_3_flt` off-blocks out in the same way. **It bites, and
it leaves the right thing standing:**

| arm, december no-clock rows | real | placebo, both clocks dealt |
|---|---:|---:|
| `Nf` | **+509.5 [+99, +1,067]** | −58.4 [−99, −14] |
| `No` | +19.3 [−440, +405] | −38.9 [−99, +16] |
| `N` | **+321.7 [+27, +585]** | −110.8 [−180, −44] |

So what this lane serves on the no-clock class is the file-clock neighbourhood and the
geometry, and both survive their own test.

**The serving division (desk, 2026-09-08 14:10 UTC).** This lane prices and serves the
**no-clock class** — rows with `obs_has_clock` false under `exp-046-surfacefit`'s
`code/block.py` at SHA-256 `4a1040e0…` — and leaves v11 everywhere else; that lane
serves the with-clock class; the desk composes the two disjoint updates. Step 2's
primary endpoint is therefore the no-clock population, and §9 T3 and T3c are the tables
that price it.

**Beyond the row's own clock — and a warning for the lane that owns it.**
`exp-046-surfacefit`'s feature, the row's own observed taxi, is worth **+6,462 s²** on
this week by itself and removes 63.7 % of the shared component; the neighbourhood's
increment over it is **+203 s²**, small, as it must be where a row already has its own
143 s clock. But priced **on the population separately**, which the assignment requires
and which an all-row mean hides, one head fitted across both classes reads
**+6,884 s² on the rows with a clock and −588 s² on the rows without one**: a correction
that carries the own-clock feature *damages* the rows the feature is missing on,
because a missing indicator can only move their level. The neighbourhood served to
those same rows is **+583 s²**. So the two must not share one fitted head — the fold
table has to fit or at least scale by reach class — and that is the concrete thing this
lane hands `exp-046-surfacefit` beside the block itself.

**The repair, tried, and what actually repairs it.** The obvious repair — one head
fitted on each reach class separately — makes it **worse**: on 287 rows the no-clock
head has about 230 rows to fit on per block, and `No` goes from +583 s² to **−987 s²**,
`own_No_Hob` from −788 to **−3,199** (T3c). The repair that works is neither of those:
**the head that serves the no-clock rows must not carry a column that is missing on
them**, and it should still be fitted on every row so it can borrow the coefficient
from the class where the observation is dense. That is exactly the arm `No` fitted on
all rows and served to the no-clock rows, **+583 s² [−441, +1,428]** — positive in the
point estimate, with an interval that 287 rows cannot close. Passing the step 2 advance
clause on this population needs the harvested months, not a better head.

**Superseded by §7 and §7a.** This paragraph read *"no candidate; step 2 and step 3's
served candidate are not reached"* while the harvest was still running. Both are reached:
step 2's three-fold rotation is §7 and the served candidate is §7a.

**Which airports the no-clock route actually reaches — measured on a ranking month,
with no label.** T5c: the question is not whether a row has a clock but whether its
*neighbours* do, and the two are very different at the weak airports. On the **whole of
2026-01, 279 airport-days and 130,323 ranking departures**, of the rows with no clock of
their own, **96.1 % at EGLL and 66.0 % at EDDF have at least one heard neighbour in
their own 10-minute cell** (median 3 and 2), against a 65.6 % and 40.2 % strict own-clock
rate. **LFPG does not**: 10.1 % reached, median 0. Over the nine airports the harvest
delivers, 45,351 rows carry no clock and **50.7 % of them are reached**. (The desk's
outage rule of 14:45 UTC is in force: EDDF 2026-01-27 to 01-31 are no-clock airport-days
and out of the receipt denominator, which is most of the difference between EDDF's 66.0 %
here and the 89.3 % the first two-day census read.) So the desk's reading — *"at LFPG a 16 % receipt still samples
the queue"* — is **right for EGLL and EDDF and wrong for LFPG**, and this lane's route
roughly doubles the population the surface programme reaches at the two airports where
`exp-046-surfacefit`'s own-clock route is weakest. **Receipt and reach are not the same
number**: `exp-046` §4 measures EGLL's strict own-clock receipt at 38 %, and T4 reads
the thinning curve at that density, but 96.1 % of EGLL's no-clock rows still have a
heard neighbour in their own cell, because a cell of six needs one heard aircraft and
not six. On the full month EGLL reads 96.1 % reached against a 65.6 % own-clock rate. The curve prices the STRENGTH of the estimator at a density; T5c prices the
POPULATION it can be computed for at all. LFPG and LTFM fail both.

**Which column carries it.** Priced one at a time (T5b), the airport × 10-minute cell
mean of the neighbours' observed residual reaches **68.8 %** on its own and the runway ×
cell version **54.9 %** at a higher contract gain (+464 s² against +442); the ±30 min
airport window reaches 30.2 %, the ±60 min runway window 2.3 %, the anomaly forms (which
need no prediction vector at all) about 18 %, and every count column reaches **nothing**.
The grain exp-041 measured the shared component on is the grain that carries it.

**One floor for one measurement** (desk, 2026-09-09; §7h). The projection now reads
cells at the SERVING floor, so the gate can no longer rest on a cell nothing may serve.
Arm `N`'s whole-class projection reads +286 / +218 / +154 where it read +741 / +225 /
+180 and is still clear of zero on every fold — **v14's gate passes under the rule and
its bytes are unchanged.** The per-cell form of §7g passes both legs on all three folds
under it too, but the desk's slot gate for v15 — the INCREMENT over arm `N` on the served
rows, every fold clear — **does not pass: january crosses zero under both fitting
conventions, so the per-cell candidate is a MAYBE and takes no slot** (§7h, T7l), with
its revival condition recorded. The cheap revival it named was then taken and is a
**second MAYBE** (§7i, T7m): the `Nf` head at LTFM shrunk toward `N` at a weight of 0.5
fixed before the read clears july and december wide, and january's increment lands on
zero — **+80 s² per row, bound +1.2 at 20,000 draws, worst of ten seeds −3.5**. Shrinking
did what it was predicted to do (january's interval half-width falls from ±155 to ±81) and
still does not establish that fold. A fourth fold month, not another weight, is what
revives it.

**After the candidate, three fold-side reads and no serving** (desk, 2026-09-09; §7d,
§7e, §7f and T7h, T7i, T7j). Of the 17 held cells, **five carry 45,004 ranking rows that
more harvested months cannot revive** — `LFPG/unreached` alone is 34,108 of them, read
pooled at +100 s² per row [−82, +335] on 54,445 fold rows, and `EGLL/reached` is measured
to HURT — while **twelve carry 45,493 rows that they can**, LEMD's already positive and
LIRF's 17,651 a coin the months would toss. The ridge is saturated (alpha 0.1 to 1,000
changes nothing) but the FORM is not: a boosted head reads `EDDF/reached` 1.4 to 1.8×
higher and loses LTFM, so it is not strictly better. And the **file-clock family alone,
which needs no observation, supports `LTFM/unreached` on all three folds and reads higher
there than the arm the candidate serves** — the LTFM half of what this lane serves does
not need the harvest.

---

## 1. What the shipped block already has, and what is new

Before writing a column of the block I read `exp-018-fresh/code/features.py` and
`window_features.py`, because exp-041's revival clause is *"an observable that is a
runway state and is **not already in the shipped block**"*. The shipped block has:

| shipped column | what it is | anchor |
|---|---|---|
| `apt_queue`, `rwy_queue` | departures NM-off-blocked and not yet airborne | take-off |
| `apt_pushed_10m/30m`, `rwy_pushed_30m` | pushbacks in the last 10/30 min | take-off |
| `rwy_share_1h`, `rwy_arrivals_1h` | the runway's share of the hour's traffic | take-off |
| `w{300,600,1800,3600}_rwy_ndistinct`, `_rwy_mode_share`, `_rwy_mode_changed`, `_rwy_is_mode` | the runway configuration in force | take-off |
| `apt_land_15m/30m_before`, `arr_taxiin_last10/30` | the arrival stream and its realised taxi-in | take-off |
| `r_prev10_apt`, `r_prev10_rwy`, `w*_r_mean`, `w*_r_p90`, `w*_r_excess` | the realised `MVT − AOBT` of the neighbours | take-off / bucket |

So the queue, the configuration and the neighbourhood mean are all **already there**,
built on NM's clock and read at the take-off instant. Only two things can be new, and
this experiment separates exactly those two:

* **the anchor** — the state where the aircraft *joins* the queue rather than where it
  *leaves* it;
* **the clock** — the physical pushback the network saw, and aircraft with no NM row.

Three event sources are therefore built (`code/state.py`, `code/block.py`): `F` the
file's own clocks, `O` the adsb.lol observations, `H` observed where heard and file
otherwise. The arrival side is the file's in all three, because an arrival row keeps
every column in the ranking file — its landing time, on-block time, taxi-in and runway
— so there is nothing there for an observation to add.

## 2. The instrument

The base is the fold-side **v11**, `v10 + 0.217·(stack-5 − v10)`, rebuilt by arithmetic
from gated vectors and gated on the published clean board: this experiment reads
**256.9104** against the published 256.91, with fold RMSEs 284.0466 / 204.7364 /
218.0274 (`analysis/base.json`). Nothing is refitted.

The shared component is exp-041's self-excluded cross-product, imported from
`experiments/exp-041-cell/code/base.py` and not rewritten. On **v11**'s residual over
all rows at airport × 10-minute grain it is **8,157 / 5,117 / 5,723 s²**
(july / december / january).

The share removed is `1 − after/before` with a **day-clustered interval on the share
itself** — both ends rebuilt from the same resampled days — because that is the
quantity the assignment gates.

**Three statistics, not one, and every table carries at least two** (FALLBACK review
round 1, blocking 1). `1 − after/before` is unbiased for the share of the shared
component only where the correction is constant inside a cell. The neighbourhood family
breaks that condition: row *i*'s residual is an input to row *j*'s correction, through
the self-excluded cell mean, so the after-cross-product falls by a term in
`Var(ε)/(n−1)` that is idiosyncratic and not shared. The three readings are

| statistic | what it is | over-credit | attenuation |
|---|---|---|---|
| `1 − after/before` | the assignment's and exp-041's convention | yes, on family N | none |
| `(2A − B) / before` | the held-out MSE the correction earns, in units of the shared component; `2A − B` forms no cross-row product | no | none |
| split half | `1 − after/before` with each row's neighbourhood built from the other half's clocks and pairs read inside (cell, half) | no | half the sources |

so the truth for family N is bracketed between the split half below and the
contract-consistent share, and the `own` arm — whose correction reads the row itself and
so cannot contaminate a pair — is the control that fixes the scale: it reads 63.7 % and
65.6 % on the two, with an identical `2A − B`. `(2A − B) / before` is a share only for a
correction that is a cell-level constant; a correction that also predicts the row's own
error can exceed 1, and `own` does, at 304 %. Cross fits use 5 contiguous day blocks. Two learners
are used and both are named on every row: exp-041 T7's `HistGradientBoostingRegressor`
(300 trees, depth 3, lr 0.05) for the ten-airport months, and **ridge on standardised
columns with a missing indicator** for the covered week, because a 300-tree fit on
5,131 rows with one day of held-out rows per block is not a measurement — it reads
negative on arms whose signal the ridge finds at +73 %.

## 3. Why the queue count fails, in both directions

**It fails, and the repair was tried.** The first version of the block read the state
only at the off-block anchor. When that reached +1.8 % I added the realised-taxi
columns (`s_taxi_rwy_30m` and its kin: the mean completed taxi of the departures that
left this runway in the last half hour, self-excluded), which is the state most
directly on the causal path, and the reading moved from +1.8 % to +2.6 % — inside its
own interval. The diagnosis is not that the anchor is wrong: T5 D shows the same state
at the two instants correlates 0.77, so the off-block reading is a different
observation and not a relabelling of the take-off one. The diagnosis is that **a count
is a poor estimator of the cell's common error**. The shared component is about 90 s
RMS per cell; the file's own clock reads each neighbour's taxi with 350–466 s of noise
(`docs/input-rule.md`; exp-043 §4 measures 308.5 s at EHAM), so an average over a cell
of six carries about 190 s of noise against a 90 s signal — and the shipped block
already spends that budget in `w600_r_mean`. Replacing the count by the neighbours'
*observed* taxi replaces 466 s of noise with 143 s, and that single change is the whole
of §0's result.

**What would revive the count arm.** A runway state the file cannot express at all:
the *identity* of the aircraft ahead in the queue and their stands, which the
observations give (`Oob_q_taxi_rwy` counts them but does not say where they pushed
from), and the de-icing queue, which December will show and July cannot. Both are
cheap to add once a winter month lands; neither is measurable now.

## 4. The leakage audit

`code/audit.py`, `analysis/audit.json`, four checks; T5 renders them.

* **A. No column reads the row's own segment.** Every count subtracts the row's own
  event **by index**, exactly. The audit re-runs the block with the self-exclusion off
  and asserts that no count moves by more than one, and reports the columns that never
  contain the row at all — the arrival columns (a departure is not in them) and the
  windows whose boundary is the row's own instant, `[t−W, t)` excluding `t`.
* **B. No column is a function of the row's own taxi span.** For a matched row
  `MVT_TIME − t_ref` *is* exp-046's observed taxi. The audit reports, per column, how
  much of that span's variance survives conditioning on twenty quantile bins of the
  column; the worst column is `Fto_s_taxi_rwy_30m` at Spearman +0.668, and it leaves
  **72 %**. That column is the mean completed taxi of the OTHER departures off this
  runway in the last half hour: a correlation with this row's own span is the mechanism
  the block is built to read, not leakage, and the test is that the span is not
  *recoverable* — 72 % of it survives. The check is single-column; a multi-column
  reconstruction is not tested here, and the `F` arm's own +2.4 % is the de facto
  answer, since a block that recovered the span would not read at the noise floor.
* **A2. The self-exclusion joined.** Audit A re-runs source `F` with the exclusion off.
  It cannot see an `O` or `H` row whose event key failed to join, which would leave the
  row silently inside its own counts, so `block.build` now asserts the join on every
  build and records it: **`O` 4,841 of 4,841 required, `H` 190,713 of 190,713, 0
  missing** (`analysis/interface.json`, `build.own_index_*`).
* **C. No label.** Departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are 0 non-null
  in the frame the block is built from; the arrival ones are read on purpose and are
  ranking columns.
* **D. Is the anchor cosmetic?** Median Pearson 0.77 between the same state at the two
  instants.

**The neighbourhood's own audit is the permutation placebo** (T3b), because A–D cannot
speak for a feature built from other rows' observations: the question there is not
whether the row's own segment leaks but whether the *pairing* of a clock to its flight
is doing the work, and dealing the clocks out to other aircraft inside the airport-day
answers it.

## 5. What it would be worth, and what is not established

exp-041's budget: 1,182 s² of within-`nm_bulk` MSE closes the 2.1154 s gap to the
leader, about 1,515 s² per eligible row. The neighbourhood is worth **+818 s² per row**
on the EHAM week, **+583 s² on the rows with no clock of their own**, and its increment
over exp-046's own-clock feature over all rows is **+203 s²**.
Neither number is a board number, and three discounts stand between them and one:

1. **The within-month cross-fit overstates the servable gain.** exp-041 §5's leak-free
   instrument puts the cross-month reading at 0.68 of the within-month one at board
   level, with a fold spread of 1.23 / 0.47 / 0.11; exp-044 step 2 reads 0.28 on a
   different base. Step 2's rotation is what settles it here, and it is not run.
2. **EHAM is among the smallest shared components of the ten** (2,127 s² on the
   covered week against 8,157 s² over all airports; it is the smallest on july and
   december, and on january EDDM's 2,151 s² is smaller than EHAM's 2,714), and it is
   one of the two airports v10 serves with `v7` rather than `refit-bulk`. A week at one
   easy airport cannot be extrapolated to the file.
3. **Receipt.** T4 says the population that needs this block gets nothing from it below
   about 30 % receipt. exp-046 §4 measures EGLL at 38 % and LFPG at 10 % strictly
   matched on a 2026 month, and LTFM at zero; those three are 36 % of the ranking file.
   T5c is the other half of that reading: what the estimator needs is a heard
   *neighbour*, and EGLL has one on 96.1 % of its no-clock rows over the whole of
   2026-01.

The 1,182 s² budget and the +818 / +583 s² gains are all `2A − B` numbers on the same
contract, so the arithmetic between them is unaffected by the share statistic's
over-credit (§2); it is the *shared-component percentages* that the round-1 repair
moved, not these.

## 6. The interface (step 3)

`code/emit.py` writes `artifacts/exp-047/block/<month>.parquet`, one row per departure,
keyed by `MVT_ID_mvt`, every other column float32, with `analysis/interface.json`
carrying the schema, the row count, the byte count and the SHA-256 of each file and of
`code/block.py` and `code/state.py`. `exp-046-surfacefit` can join it on `MVT_ID_mvt`
and price any subset of the prefixes; **no file of that experiment is edited by this
one**. The `Nf_*` and `No_*` columns are defined relative to a prediction vector, and
the manifest names it.

**The `_sc` repair** (FALLBACK review round 1, blocking 2). The receipt rate a count is
divided by is now read over the **harvested airport-days**, not over the month, and at
the **airport-day** grain (desk, 15:05 UTC), with the airport's own harvested-day rate
as the fallback below `MIN_DAY_ROWS = 30` departures. The first emit divided EHAM's July
counts by 0.2153 — 4,841 clocks over every July departure, most of them on days the
harvest had not delivered — so every `Oob_*_sc` and `Hob_*_sc` column was the raw count
times 4.65, and two months with different coverage would have carried different
constants for the same airport. A row off a harvested day now carries NaN rather than a
count scaled by a rate measured elsewhere, and `analysis/interface.json` records the
rate and the delivered-day count per airport per month.

**What is on disk now** (round 2, blocking 2; the hashes CORRECTION 2 quoted named a
file that no longer exists). Five months, all under summaries rule `d4c59868…` and all
built against matcher **`ba41935e…`** — the callsign-form repair of the desk's 16:58 UTC
message, which took 2025-07 from 84,360 to 87,176 clocks and 2025-12 from 97,715 to
100,824, gaining 5,954 and losing 29 in total. **T6** is the table and the manifest is
the authority; it now pins the matcher separately from the two builders, because the
matcher moves independently of this lane and moved three times in one day. The 2025-07
numbers step 1 reports were read on the pilot week under rule `15a72d55…` and are not
re-read here: they diagnosed which residual structure the observations reach, and §7's
rotation is what anything served rests on.

## 6a. The desk's two rules of 2026-09-08

**14:45 UTC — the EDDF feeder outage.** *"An EDDF feeder OUTAGE on Jan 27-31 (82, 62,
22, 14, 10 percent after 91-98) that must not be read as normal reception: treat those
airport-days as no-clock for the class boundary and the receipt scaling."* In force in
`block.OUTAGES`, applied before the anchor and before any thinning: those rows get
`obs_has_clock` false, `reach_class` `outage`, no `obs_offblock`, and their airport-days
leave the receipt denominator.

It is a **table and not a threshold**, on purpose. A rule of the form "below half the
airport's own median" flags LFPG 2026-01-29 to 01-31 as well, and LFPG's 24 % declining
to 5 % is that airport's normal reception, which the desk's message separates from the
outage explicitly. What the code does automatically is **warn**: `outage_report` lists
every airport-day below that fraction which is *not* declared, in every build record, so
the next outage cannot pass silently. On 2026-01 it warns on those three LFPG days and
nothing else.

**15:02 UTC — the hold moved upstream, and this lane's table is gone.**
`exp-046-surfacefit`'s `block.py` now holds those airport-days inside `load_summaries`
itself (matcher `a0652dff…`, its commit `5b0642e`): the segments never load, so no row
matches on them, no segment of theirs feeds a neighbour, and the receipt denominator
never counts them. All three things this lane's own table did are therefore done by the
**one matcher both lanes import**, which is the point — a second list here could drift
from that one and put the two lanes' class boundaries out of step. So the set is read
from `B46.HELD_AIRPORT_DAYS` and what survives here is the **audit label**
(`reach_class == "outage"`, so a consumer can tell a held row from an ordinarily
unmatched one) and an **assertion** that the upstream hold really left no clock. It
reads **0 still-clocked of 2,517 rows on held days**. Rebuilt on `a0652dff…`, the
2026-01 census is identical on every airport and on the pooled row — which says nothing
about what the hold COSTS, because the hold was upstream in both runs — and the 2025-07
block is bit-identical in **157 of its 185 columns** — the 28 that moved are exactly the
`_sc` columns, and they moved for the 15:05 rule below, not for the sha.

**15:05 UTC — receipt scaling is per airport-day, and a one-day dip is not an outage.**
The desk ruled that six single-day dips (EGLL Jan 1, 2, 31; EHAM Jan 4, 5; LIRF Jan 1)
are **not held**: the held table is for unusable days only, and *"receipt scaling is per
airport-day, so dips carry fewer neighbours and nothing else"*. `receipt_rates` now
returns a rate per `(airport, day)`, with the airport's own harvested-day rate as the
fallback for a day under `MIN_DAY_ROWS = 30` departures. That is what makes a dip need
no special handling: EGLL on 2026-01-01 is divided by EGLL's rate **on that day**. The
spread it removes is not small — over 2026-01 the per-day rate runs 0.36 to 0.84 at
EGLL (median 0.66), 0.68 to 0.97 at EHAM, 0.23 to 0.63 at LIRF — so a single per-airport
constant was scaling a third of EGLL's days by a factor half again too small.

**What the rule costs — corrected in round 2, and it is not nothing.** The first
reading here compared the census with this lane's rule against one with `--no-outage`,
found them identical, and concluded the harvest had delivered no usable EDDF summary for
2026-01-27 to 01-31. **That cause is false and the comparison varied nothing.** The five
summaries exist (`artifacts/exp-045/summaries/EDDF/2026-01-27..31.parquet`, 827 / 825 /
478 / 461 / 464 rows) and are registered under the census rule; what zeroed the clocks
was `exp-046-surfacefit`'s own `HELD_AIRPORT_DAYS`, which was in force **in both runs**,
so `--no-outage` only ever switched off this lane's now-redundant label.

The cost, measured with the same build path run twice and only the upstream hold
differing, is in **T5d**: the hold moves **2,517** EDDF rows into the no-clock class and
discards the **591** clocks the feeder did deliver on them, taking EDDF's own-clock
share from **50.9 % to 47.2 %**. It is a real cost and a small one — the round-2 finding
put it at "roughly 1,500 clocks" from the harvest's receipt rates, and those rates count
all traffic rather than the strict-matchable departures, so the measured figure is 40 %
of the estimate. *(Those rates were themselves 9 to 11 points too high until the harvest's
correction of 2026-09-09; the estimate is about 1,300 clocks and the measured figure 45 %
of it — §7k, T7o.)* The rule is paid for, not free, and what it buys is that a degraded
feeder is never read as a property of EDDF's traffic. That the *mechanism*
works was shown separately, before the hold moved upstream, by declaring 2026-01-25 and
01-26 instead: EDDF's clocks on those days went 259 → 0 and 198 → 0, 983 rows moved to
`reach_class` `outage`, and the receipt rose from 0.4929 on four days to 0.5488 on two.

**14:50 UTC — v12 is the head (tail rule).** v11 with only its K2 member replaced by
`exp-035-k2climb` arm 3b's climbed member; board 262.7418, served sha `48dd572d…`. Its
fold side is built here by `exp-035` §26a's own two lines — mask at `exp-036-mask`'s
eight kept airports with v7 elsewhere, then 0.217 toward stack-5 — and gated three ways:

| gate | reads | wanted |
|---|---:|---:|
| the climbed head's own fold RMSE | 283.6627 / 204.2511 | `price_vs_v11.CAND_RB_PUBLISHED` |
| the SAME two lines on the unclimbed base rebuild v11 | 256.9104 clean board | 256.91 |
| fold-side v12 | **256.4644** clean board | 256.9104 − 0.4460, `exp-035` §26c |

`base.load()` now carries `p12` and `r12` beside `p11` and `r11`, `arms.py` takes
`--head` and defaults to v12, and `serve.py`'s head is v12's bytes. **Step 1 above stays
on `r11`**: it is a label-side diagnostic that was measured before the head moved, and
re-basing it would not change what it diagnosed — which residual structure the surface
observations reach. Step 2 and anything served are on `r12`.

## 7. Step 2 — the fold side on the contract, on the rotation

**It passes, on three folds, at the reach grain.** The arm is `N` — the file-clock
neighbourhood `Nf_*` and the observed-clock neighbourhood `No_*` together — correcting
fold-side **v12** on the no-clock class, with the head fitted on the fold panels the
scored month is not in and carrying no missingness indicator. The heads are not the same
size and the record has to say so (round 3, C3): **july_loo and december are each scored
by a head fitted on ONE panel**, january by a head fitted on **two**, and the served head
in `code/serve.py` is fitted on **all three**, so this table is a lower bound on the
served head's data rather than a like-for-like reading of it.

Projected to the ranking months' own reach per airport-**month** and per **reach class**,
it reads

| fold | projected `2A − B` | board (approx.) | pooled on the fold's own population | worst single airport dropped |
|---|---:|---:|---:|---|
| july_loo | **+286 s² [+178, +401]** | +0.559 s | +504 [+302, +703] | EDDF → **+145 [+50, +250]** |
| december | **+218 s² [+31, +454]** | +0.424 s | −373 [−878, +40] | EDDF → **+47 [−71, +170]** |
| january | **+154 s² [+79, +233]** | +0.300 s | +235 [+72, +393] | EDDF → **+94 [+37, +150]** |

*(Read at ONE FLOOR since 2026-09-09: a cell the hold may not serve carries no
projection. The same three folds read +741 [+414, +1,868] / +225 [+37, +456] / +180
[+79, +306] while the projection priced cells at 100 fold rows and the hold refused them
at 500 — §7h, ledger CORRECTION 6. The candidate is unchanged and its gate still passes.
The unpriced weight the rule declares is 7.1 / 2.4 / 6.7 % of the served rows, and
filling every unpriced cell with its fold's class-pooled gain instead of nothing gives
+307 [+190, +433] / +219 [+27, +459] / +185 [+87, +287].)*

Every interval is clear of zero at full amplitude on the population served, which is the
assignment's advance clause. The board column converts an s² drop at v12's **clean** base
and is **approximate**: it is the larger of the two possible second-counts, and the gate
is the s² interval, not it — `code/serve.py` measures the served file's board directly
instead of converting. The control is `own`, whose columns are null on every no-clock row
so its correction is the head's intercept and nothing else: it reads **−49, −4.1, −4.8**
with the december and january intervals *below* zero, so the block is measured against a
baseline that is slightly harmful rather than against nothing. **T7** and **T7d** carry
the tables.

**What the reach grain repaired, and what it withdrew** (FALLBACK round 3, blocking 1).
Reading the gain per airport was not fine enough. The 2025 receiver map decides not only
which rows are no-clock but whether a no-clock row has a heard **neighbour**, and the
correction is a different object in the two cases: with a heard neighbour the `No_*`
columns carry an observation, without one they carry the fit rows' mean and the
correction is file-clock work. The two populations were being read on one number at the
airports that carry the projection — EDDF's fold gains are read at reach 0.99 / 0.97 /
0.96 and served at **0.64** and **0.82**; LIRF's july gain, read at **0.93**, was applied
to the **15,551** rows LIRF has in this lane's 2026-07 class, **15,524** of them
unreached (the month's 15,838 ranking rows; 15,827 carry no clock on this lane's own
block, and the shared class vector and the `noac` hold take it to 15,551). The projection now prices each airport's fold
gain **separately on reached and unreached rows** and applies each to the ranking
month's own count of that class, and the leave-one-airport-out column carries an interval
off the same draws. That interval **withdraws** §0's earlier *"every fold still clear with
EDDF dropped"*: it holds on july and january and **not on december**, where dropping EDDF
gives +54 [−64, +178]. Ledger row 497 said "positive", which was true; §0 said "clear",
which was not established and is now measured and corrected.

**A cell the fold does not price contributes zero, not an assumption.** A cell with fewer
than `arms.MIN_CELL = 100` fold rows is recorded as unpriced and contributes **zero** to
the point estimate; the unpriced weight is 1.6 % of ranking rows on july, 2.1 % on
december and 4.8 % on january (january's is LEMD, LFPG and LIRF *reached*, at airports
that were dark all month). The record carries the sensitivity beside it — the same total
with every unpriced cell filled by that fold's class-pooled gain — and it is **+745**,
**+231** and **+207**, above the headline in all three folds, so the zero-fill is the
conservative direction. **For the file written it is moot**: an unpriced cell fails the
cell hold and is held, so its rows carry `d = 0` exactly. For the whole-class headline
the zero flatters only where a cell's true gain is negative, and the two candidates with
evidence are january's LFPG/reached and LIRF/reached — dark all month, weights 0.012 and
0.010, whose other-fold readings are −266 / −241 and +750 / −157 — worth about **−5 s²**
of january's +180 (round 4, item 6). A second sensitivity filling each unpriced cell from
its OWN other-fold gains was considered and not run: it would move january by that −5 and
nothing else, and it borrows a month's instrument to price a month that had none.

**The pooled column and the projected column disagree on december, and the projection is
the endpoint the desk set.** Pooled, `N` is −372 s² on december. The two columns weight
the same per-airport gains differently: pooled by december's own no-clock rows, projected
by the rows each airport-month actually has in the ranking file. EGLL is the whole of the
gap — 11,216 of december's 64,853 no-clock rows, at about −3,000 s² each — and EGLL's
2025-12 no-clock class is an artefact of its feeder, which ran at 12 % rising to 65 % in
july and 93 % in january. In the ranking months EGLL is 70 % and 96 % matched, so those rows are 17 % of the
fold's no-clock population and **4.2 %** of the served one. (Round 3, B: these are this
lane's own class rates on the five emitted blocks — EGLL matched 5.8 % in 2025-07, 43.9
in 2025-12, 43.5 in 2025-01, 70.3 in 2026-01 and 96.1 in 2026-07 — not the harvest's
traffic receipts, which are a property of the feeder and not of the join.) That is
exactly the reading the desk's 16:50 UTC rule exists to prevent: *"a 2025 airport with no
receiver is not read as a property of its flights"*.

**The fit side needed a repair too, and the obvious one is wrong.** If the 2025 receiver
map contaminates the scored side, it contaminates the fitted side as well, so the first
attempt dropped every fitting row on an airport-day below 0.5 receipt
(`--fit-min-receipt 0.5`, 103,537 fitting rows down to 33,923). **It made the readings
much worse** — december `No` fell from −303 to −2,147 s² and `Nf_geo` to −994 — because
it selects on the outcome: the no-clock rows it keeps are the ones a *healthy* receiver
still failed to hear, which is not the population served. **T7b** records the failure.

**A second fit-side selection was tried and also failed, for a different reason.** If a
receipt floor selects on the outcome, selecting on the **instrument** should not: drop
every fitting row at an airport-*month* whose matcher reach is further than 0.25 from
the ranking file's reach at that airport (`--fit-max-instrument-gap 0.25`, 11
airport-months, 165,677 fitting rows down to 116,285). That is a property of the
receiver map and of no row's outcome. **It is worse than the receipt floor**: projected,
`N` goes to −1,149 s² on july_loo, −419 on december and +149 on january. The cause is
visible in the list it drops — EDDF, EDDM, EGLL, LEMD and LIRF in 2025 are exactly the
airport-months where the *observed* neighbourhood exists, so `No_*`'s coefficient is
left to be fitted on EHAM, LEBL, LSZH and LFPG alone and does not transfer. `Nf_*`, which
needs no observation, survives it on december (+144 [+54, +245]) and nowhere else.
**T7c** records it.

Two fit-side selections, two different failures, one conclusion: **the head is best
fitted on everything and priced where it serves.** The repair that works runs the other
way from both: fit on **every** row of the fitting panels (`--fit-rows all`). `Nf_*` needs no observation at all and is defined identically in both
classes, so a head fitted on every row estimates the same coefficients from twice the
data and is then priced only where it serves. On the two-fold rotation it took `N` from
+301 [−56, +685] to +403 [+115, +795] on july and firmed december's lower bound (both
read before the reach-grain repair; §7's table is the current reading). The same rows
with the head fitted on **unmatched rows only** are **T7a**.

**The count columns are the receiver map, and that is disclosed** (round 3, E). The
desk's rule is that an absent observation is never a feature, and `fit_predict` honours
it for the *residual* columns: they are mean-imputed on the fitting rows and nothing
else, so an absent row gets the fitting rows' mean residual. The **count** columns do not
honour it in the same way. `No_n_n_cell` and its two siblings are a finite **zero** on
106,790 of the served rows rather than absent, and a finite zero is exactly the statement
"the network heard nobody here", which is the receiver map. The reviewer recovered the
head's coefficients from the served rows and priced it: the count-driven part of the
correction has a spread of 3.3 s against the correction's 24 s, about +4 s of level at
LTFM and LFPG and −7 to −10 s at EGLL. It is small, and it is now also the variable the
projection SPLITS on, which is the honest use of it: the reach class is read as a
property of the instrument, priced apart, and projected at the ranking month's own reach
rather than carried as a feature of the flight. `Nf_n_n_*` is traffic density and reads
the file, not the network. `receipt_rate` never reaches a head — arms select columns by
prefix and it is used for row selection only.

**What this lane serves, and what it does not.** The served columns are label-free at
serving time on a ranking month: `Nf_*` reads the *file's* `AOBT_3_flt` of the other
departures in the cell, and `No_*` reads their **observed** taxi — `MVT_TIME_UTC_mvt`
minus an adsb.lol off-block — never a label. The geometry does not pay on the rotation
(−94, −18, −7 across the three folds) and is not served. On the with-clock class the
block adds nothing over `exp-046-surfacefit`'s own-clock feature cross-month: projected,
`own` alone is +1,607 [+942, +2,136] on july and +881 [+593, +1,175] on december while
`own_No` is +1,689 and **+630**, lower on the fold that matters — so the +842 s² december
refinement §0 records was a within-month reading and does not survive the rotation.
**T7d** is that table, and the desk was told in one line.

### 7a. The candidate

`code/serve.py --arm N` writes **`submissions/candidates/exp-047-queue.parquet`**,
SHA-256 **`519ca89f381056af…`**, 4,707,609 bytes. The base is **v13** —
`submissions/uploaded/youthful-giraffe_v13.parquet`, `79a30162…`, board **248.4818** —
and the file is v13 everywhere except the rows this lane both owns and may serve. The
dry run and the real run agree bit for bit.

* **63,862** of the ranking file's 344,841 rows move, at an RMS of **22.1 s** and a
  maximum of 313 s. **No row is clipped** at the 60 s floor: the file's minimum is
  v13's own **60.4192**.
* The moved set is `~obs_has_clock & ~noac` **on `exp-046-surfacefit`'s class vector**
  (`artifacts/exp-046/class/<month>.parquet`, hashes gated against its
  `analysis/class-manifest.json`), intersected with the two cells §7c keeps. 154,358
  rows are in the class; **90,496 of them are held** by the cell rule.
* **0 held rows move**, asserted before the file is written: the 388 `noac` rows, every
  row `exp-046-surfacefit` serves, and every row in a held cell.
* **The two lanes' updates are disjoint by construction, not by agreement.** This lane
  moves 63,862 rows, v13 moved 188,483, and the intersection is **0**. v13 and v12 are
  **bit-identical on all 154,358 rows this lane owns** (`max|v13 − v12| = 0.0`), which
  is what lets a correction priced against fold-side v12 be applied to v13's bytes.
* **LTFM is served by the file clock alone, and that is real work rather than an
  intercept**: `Nf_n_res_cell` is finite on 24,794 of its 25,185 no-clock rows in
  2026-07 at a median of 7 neighbours in the cell, where `No_n_res_cell` is finite on
  none. This is why the block reaches an airport the harvest has never heard.
* The gate `serve.gate()` reads is the **projected** endpoint of
  `analysis/arms-step2-unmatched-fitall-3f.json`, on **both** populations: the
  registered whole no-clock class **and** the cells actually served. `N` is the only arm
  that clears either. `rank_frame` refuses to read a block whose manifest names a
  different matcher from the one on disk, from the step-2 gate record, or from the
  shared class manifest, or whose parquet does not hash to the manifest's entry.

**Per fold, cross-fitted, in RMSE seconds** — each fold's `d` from a head fitted on the
other two, with the same cell hold applied on the fold side so this is the currency of
the file that is written:

| fold | served rows | RMSE before → after | clean before → after | RMS move |
|---|---:|---|---|---:|
| july_loo | 29,844 | 283.5657 → **283.2963** | same | 18.3114 s |
| january | 24,954 | 337.4611 → **337.363** | 217.6288 → **217.4767** | 19.1461 s |
| december | 24,169 | 204.1985 → **204.0656** | same | 19.664 s |

At the ranking shares 0.5571 / 0.4429 that is **clean +0.2231 s** (256.4643 → 256.2412)
and **all-row +0.1854 s** (308.5977 → 308.4122).

**On the rows the file actually moves**, the assignment's four-way split (**T8**, rebuilt
on the held update for round 4, item 2):

| fold | rows moved | all | bulk `y ≤ 3600` | tail | no-NM (rows) |
|---|---:|---|---|---|---|
| july_loo | 29,844 | 258.7256 → **256.8326** | 240.7049 → **239.1589** | 2,669.5 → **2,634.7** | 478.9 → **474.7** (684) |
| january | 24,954 | 211.5272 → **210.5612** | 204.3075 → **203.3702** | 2,507.0 → **2,496.2** | 461.7 → 464.0 (326) |
| december | 24,169 | 200.7134 → **199.7847** | 191.7625 → **190.6501** | 1,547.7 → 1,559.1 | 317.5 → 318.0 (381) |

**All three folds improve on all and on bulk, and december improves too** — the hold took
out the cells that made it negative. Three sub-columns get worse and the record says so:
december's tail by 11.4 s on the rows above an hour, december's no-NM by 0.5 s on 381 rows
and january's by 2.3 s on 326. They are small populations and none of them is the board's
own statistic, but they are the price of a correction fitted on the bulk.

**December's own read is the transfer risk, and §7b is what it consists of.** It is not
a ranking-share fold, so it is not in the board number; it is carried beside it because
the pooled reading on december's own population disagrees with the projection, and §7b
prices that disagreement.

### 7c. The cell hold — what this lane serves, and what it hands back

The desk's rule of 2026-09-09, the one `exp-046-surfacefit` applied to LFPG, read at the
reach grain: *"a served airport-month cell whose fold-side read has no support of its
own — its own day-clustered fold interval crosses zero, or it is extrapolated from under
500 fold rows as the 141-row cell is, 125-fold onto 15,551 dark LIRF rows — is held at
the base, with the projected gain forgone stated per cell."*

**How it is read, and the reading rejected.** A cell is `(airport, reach class)`, which
is the grain step 2 prices at and the grain the row's own evidence lives at; a ranking
month enters through which of its rows are reached, so the census below reads out per
airport-month. The test is applied to **every fold that prices the cell**, exactly as the
assignment's advance clause is applied to every fold of the arm, and "clear of zero"
means clear **above** zero — a cell whose interval sits entirely below zero is the
strongest reason to hold it, not to serve it. The rejected reading, recorded because it
changes the candidate: pooling a cell's three folds into one interval would **keep**
LIRF/unreached, whose 141 july rows are the very cell the desk named, so the singular
"its own fold interval" is read per fold. One definition serves both sides —
`arms.cell_supported` decides the record's `served_cells_only` projection and
`serve.cell_support` reads the same function for the file.

**Two cells of nineteen survive** — as the every-fold rule read them, which is what v14
served. 63,862 of the class's 154,358 ranking rows are served; 90,496 are held at v13.
**Under the lit-fold refinement (§7r) it is three cells and 72,943 of 153,832 rows**, the
class itself having shrunk by 526 rows at the matcher rebuild: `EDDF/reached` 16,638,
`LTFM/unreached` 47,577, `LEMD/reached` 8,728.

(Row counts are `analysis/serve.json`'s census of the ranking rows in each cell. **T7g**
renders the same table from the step-2 record, where the count is the weight times
344,841 and so differs by a row of rounding at EDDF.)

| cell | ranking rows | july_loo | december | january | verdict |
|---|---:|---|---|---|---|
| EDDF/reached | 16,285 | 4,674 rows, [+1,924, +4,211] | 1,561, [+1,379, +6,862] | 3,586, [+327, +2,449] | **served** |
| LTFM/unreached | 47,577 | 25,171, [+354, +1,304] | 22,609, [+40, +433] | 21,368, [+47, +492] | **served** |
| LIRF/unreached | 17,651 | **141 rows**, [+3,223, +30,085] | 2,388, [−655, +1,989] | 11,317, [−333, +151] | held |
| LFPG/unreached | 34,108 | 20,342, [−385, +44] | 16,203, [−132, +785] | 17,900, [−15, +725] | held |
| EGLL/reached | 6,317 | 3,668, [−1,398, +396] | 9,772, **[−7,309, −2,280]** | 9,275, [−749, +880] | held |
| LEMD/reached | 8,739 | 9,366, [+709, +1,675] | 818, [+581, +2,327] | **unpriced, 0 rows** | held |
| EDDF/unreached | 5,336 | unpriced, 52 | unpriced, 43 | 155, [−886, +5,171] | held |
| the other 12 | 18,345 | — | — | — | held |

**The gain forgone, per fold:** **+40 s² of july's 286**, **+14 of december's 218**,
**+58 of january's 154**. `analysis/serve.json`'s `cell_hold` block states it per cell
and per month; the largest single lines are LEMD/reached (+29 on july, +35 on december)
and LFPG/unreached (+21 on december, +29 on january, taking back +16 on july), and
EGLL/reached takes back +79 on december because the correction hurts there. *(Before the
one-floor rule of 2026-09-09 these read 495 / 22 / 84, of which LIRF/unreached's 141-row
july cell alone was 449; §7h and ledger CORRECTION 6.)*

**What is left is read on exactly what is served.** The projection restricted to the two
served cells, off the same draws, is

| fold | served cells only | whole no-clock class |
|---|---:|---:|
| july_loo | **+246 s² [+165, +338]** | +286 [+178, +401] |
| december | **+203 s² [+94, +360]** | +218 [+31, +454] |
| january | **+96 s² [+41, +158]** | +154 [+79, +233] |

Every interval is clear of zero. No served cell is extrapolated more than **10.4-fold**
(EDDF/reached, 1,561 december fold rows onto 16,285 ranking rows; LTFM/unreached is under
2.3-fold in every fold), where the whole-class july number used to rest on a 125-fold one
— until the one-floor rule stopped the projection reading cells the hold refuses to serve
(§7h). `serve.gate()` requires both readings to clear zero on every fold, so a future
record that loses either cannot write a file.

**These are post-selection numbers, and the record says so** (FALLBACK round 4, item 3).
The hold picks 2 cells of 19 **on their fold intervals**, and the served-cells-only
projection and the +0.2231 s currency are then read on those same folds. A number chosen
for being large on a sample and re-read on that sample is optimistic, by an amount this
record cannot state without a nested selection it has not run. The desk's rule mandates
the selection, so this is a caveat and not a repair; what limits it is that both surviving
cells are wide rather than marginal on july and january, and the one thin margin is
**LTFM/unreached on december, whose lower bound is +40 s² per row on the largest single
weight (0.138)**.

**And the reach split leaves heterogeneity inside the reached class** (round 4, item 4,
measured by the reviewer on the emitted fold blocks). `reached` is `No_n_n_cell > 0`, the
heard count in the airport × **10-minute** cell — that is the boundary between "the `No_*`
columns carry an observation" and "they carry the fit rows' mean", which is the cut that
matters, but the count still varies inside it. Served EDDF/reached rows have 1 / 2 / 2 / 4
/ 5 heard neighbours at the 10th / 25th / 50th / 75th / 90th percentile where the fold
rows sit mostly at 3 to 8, and the gain is not flat in that count: at 1 and 2 neighbours
it reads +9,857 and +6,505 s² on july, +1,952 and −349 on december, −1,769 and +2,005 on
january. Reweighted to the served neighbour-count distribution, EDDF/reached reads about
+5,300 / +2,400 / +940 per row against the record's +3,003 / +3,620 / +1,261, and the
served-only totals become roughly **+355 / +147 / +80** against +246 / +203 / +96. Every
fold stays positive in the point estimate. It is recorded as a sensitivity and not acted
on: the low-count buckets are 61 to 464 rows, two of the six cross zero, and splitting
further prices noise.

**The cost in the board's currency.** Cross-fitted by fold with the same hold applied on
the fold side, so the number is the currency of the file that is actually written:
**clean +0.2231 s** (256.4643 → 256.2412) and **all-row +0.1854 s** (308.5977 →
308.4122), against +0.3755 and +0.3276 without the hold. The hold gives up 41 % of the
clean gain to serve 41 % of the rows, and buys a candidate whose every served cell has
its own fold-side support.

### 7b. December reconciled — now on the file that is written

*(This section reads v14's file. **T9 and §7u carry the lit file's december**, which is
+402 s² [+181, +645] on its 24,979 served rows and +61 [+31, +97] all-row — both clear
above zero, where the reading below is v14's.)*

The desk asked for the two numbers side by side before it acts on a verdict, and its
rule is explicit: the registered endpoint for the surface lanes is the **projection**
(`assignments/queue.md`, the pilot's §5 method), so a projected december clear of zero
passes the every-fold clause; a raw december interval clear **below** zero holds the
candidate as recorded rule did; a raw interval **crossing** zero is recorded as the transfer risk.

**Read on the candidate as it stood before the cell hold, it was the third case. Read on
the file actually written, it is none of the three: the raw december is clear ABOVE
zero.** That is the round-4 repair (item 1): the earlier reading in this section priced a
file that served the whole no-clock class, and the desk's rule has to be read on the rows
the file moves. **T9** is the table.

| reading | rows | `2A − B` s² | RMSE seconds |
|---|---:|---|---|
| **the file written**, december's served rows | 24,169 | **+372 [+143, +619]** | **+0.929 [+0.354, +1.609]** |
| **the file written**, december all rows | 165,677 | **+54 [+24, +91]** | **+0.133 [+0.060, +0.213]** |
| projected, served cells only | – | **+203 [+94, +360]** | – |
| projected, the whole no-clock class | – | +218 [+31, +454] | – |
| *pre-hold*, the class's rows | 64,799 | −12 [−340, +292] | −0.026 [−0.749, +0.642] |
| *pre-hold*, all rows | 165,677 | −5 [−128, +112] | −0.011 [−0.314, +0.269] |

**What the hold did to december, and why.** Before it, december's raw read straddled zero
widely and was recorded as the transfer risk. The hold removes the cells whose own
december interval was negative or unsupported — EGLL/reached alone reads **−4,308 s² per
row [−7,309, −2,280]** on 9,772 fold rows — and what is left reads positive on its own
population: **EDDF +3,217 [+1,399, +5,954]** on 1,561 rows and **LTFM +176 [−2, +347]** on
22,608. Removing EGLL from the all-row reading now barely moves it (+54 → +62), because
EGLL is no longer in the update at all. The pooled-versus-projected disagreement §7 records
was, in the end, a statement about which cells were being served; served on the cells that
have their own support, the two readings agree.

**LTFM's own december interval is the thin one.** Its lower bound is −2 s² per row on the
raw served reading and +40 on the fold-side cell reading that the hold gated on, on the
largest single ranking weight (0.138). It is the cell to watch, and §7c's post-selection
caveat is mostly about it.

**The instrument, month by month, is unchanged and is still the reason.** Four airports
move between 2025-12 and the ranking months — EGLL **+0.264 / +0.522**, EDDF **−0.436 /
−0.553**, LEMD **−0.009 / −0.406**, LIRF **+0.004 / −0.477** — and no other airport moves
by as much as 0.04. The bottom block of **T9** is that instrument.

### 7d. Why 17 of the 19 cells lack support — the instrument, or the arm

**The desk's question (2026-09-09):** a held cell fails either for want of 2025 fold rows
to read it with, which more harvested months would fix, or for want of a signal, which
they would not. Only the first kind is worth waiting for. **T7h** answers it cell by
cell, and `analysis/foldside.json` carries the arithmetic; every cell gain in it
reproduces the gated record exactly (max |Δ| 0 s²), so this is a new reading of the same
numbers and not a new fit.

**The separator is the cell's POOLED read over the three folds**, day-clustered on the
union of their days. §7c records the pooled reading as REJECTED for the serving hold,
and that stands: pooling would keep the 141-row LIRF cell the desk named. It is the right
reading for a different question — whether anything is there at all — because a cell with
too few rows in one month can still have thousands across three.

**Five cells, 45,004 ranking rows, are held by the ARM and more months will not revive
them.** The largest held cell of all is `LFPG/unreached`, 34,108 ranking rows: it has
16,203 to 20,342 fold rows in EVERY fold — no instrument problem anywhere — and its
pooled read on 54,445 rows is **+100 s² per row [−82, +335]**. LFPG's 16 % receipt was
the hope this lane was built on (§0), and at the unreached grain the answer is that the
queue the receiver does sample does not price the rows it does not. `EGLL/reached` is
worse than nothing: **−1,901 [−3,336, −849]** on 22,715 pooled rows, the one cell where
the correction is measured to HURT. `EHAM/reached`, `LSZH/reached` and `EGLL/unreached`
are the same story on 2,528, 1,840 and 211 ranking rows.

**Twelve cells, 45,493 ranking rows, more months can revive.** Three are held by the
INSTRUMENT alone — pooled read already clear above zero, thin in at least one fold:
`LEMD/reached` (+1,180 [+784, +1,636], 8,739 ranking rows, and its january fold panel has
no REACHED rows at all, because LEMD is dark for the whole of 2025-01), `LEMD/unreached` (+179 [+50, +329]) and
`EDDM/unreached` (+285 [+12, +642], 45 ranking rows, worth nothing). One is held month to
month: `LEBL/reached` has rows enough in all three folds and a pooled +1,901 [+1,057,
+2,838], and one fold's own interval still crosses zero, so more months decide whether
that month is noise. The other eight are simply too thin to say, and two of them are the
prize: `LIRF/unreached` (17,651 ranking rows, and 141 july fold rows — the cell round 3
blocked on) and `EDDF/unreached` (5,337 ranking rows on 250 pooled fold rows).

**What it decides.** The harvest's next months are worth about **45,000 ranking rows of
possible new service**, of which LEMD's 8,739 are already measured positive and LIRF's
17,651 are a coin the fold months would toss. They are not worth waiting for at LFPG,
which is the single biggest dark population in the file and is now a measured
non-positive for this arm at this grain. That is a statement about `Nf_* + No_*` as
priced here, not about LFPG: §7f prices a different arm on the same rows.

### 7e. The capacity of arm `N`'s head, at the registered rotation

**The question is whether the +0.22 s the candidate carries is limited by the learner or
by the signal.** **T7i** holds the rotation, the columns, the folds, the population, the
fitting rows and the head fixed, and moves only the learner. The gradient-boosted heads
are fitted on mean-imputed, standardised columns, so no split can read an absent
observation — the desk's 16:50 UTC rule is what `arms.py` refuses the native-NaN GBM for,
and imputing first removes that exposure while leaving the capacity question intact.

**Inside the linear form there is no capacity question.** Four orders of magnitude of
ridge penalty — alpha 0.1 to 1,000 — move `EDDF/reached`'s july gain from +3,003 to
+2,972 s² per row and `LTFM/unreached`'s from +756 to +758, and the served-cells
projection from +246 / +203 / +96 to +245 / +202 / +96. The registered alpha of 10 is not
a tuned number and nothing turns on it.

**Outside it there is.** A boosted head on the same columns raises `EDDF/reached` by 1.4
to 1.8× on every fold — **+5,002 / +4,437 / +2,248 against +3,003 / +3,620 / +1,261** —
and the effect is not a fit artefact of one size: 300 trees at depth 3 and 800 at depth 6
agree within 3 % on that cell. The runway-state signal at a reached airport is
nonlinear, and the ridge is leaving a large part of it.

**And it costs LTFM.** Under either boosted head `LTFM/unreached` stops clearing zero on
december and january (+436 [−247, +1,589] and +53 [−301, +452]), so the cell hold drops
it and the served population falls from **63,862 ranking rows to 19,499** — `EDDF/reached`
plus `LEBL/reached`, which the boosted head gains. The served-cells projection is higher
on all three folds (+272 / +237 / +121 against +246 / +203 / +96) on under a third of the
rows. **A boosted head is therefore not a strictly better candidate**, and reading a
capacity off this table selects on the same folds that chose the two cells — §7c's
caveat, a second time.

**One reading to guard against.** The class-pooled column and the projected column
disagree on december under the ridge (−373 s² against +225), and they are not in conflict:
the class column weights every fold row equally, so it is dominated by `EGLL/reached` and
`LFPG/unreached`, which together are 26,000 december fold rows and 40,425 ranking rows of
the class's 154,358 — the cells the correction hurts or does nothing on. The projection
weights each cell by the rows it would actually serve. §7 states the same thing at the
airport grain.

### 7f. The file-clock family alone, which is what serves LTFM

**`Nf_*` needs no observation.** It is built from the FILE clocks of the other aircraft in
the airport × 10-minute cell, so it exists on every row of every airport whether the
harvest heard it or not. LTFM is where that matters: round 4 verified `No_n_res_cell`
finite on NONE of its 25,185 no-clock 2026-07 rows and `Nf_n_res_cell` finite on 24,794
of them. **T7j** prices `Nf` alone on the whole no-clock class, at the reach grain, under
the same cell hold.

**It supports `LTFM/unreached` on all three folds, and reads HIGHER there than the arm the
candidate serves**: **+986 [+420, +1,724] / +524 [+218, +930] / +355 [+78, +647]** against
arm `N`'s +756 / +232 / +264 on the same rows. That is the cell carrying 47,577 of the
candidate's 63,862 served rows. The `No_*` columns are the fit rows' imputed means on
every one of them, and carrying them dilutes the correction — a mean is not neutral when
it is standardised beside columns that do vary.

**The reverse holds where the observation exists.** At `EDDF/reached`, 96 to 99 % reached,
`Nf` alone reads +901 / +836 / −116 against `N`'s +3,003 / +3,620 / +1,261 and loses
january's interval. The two families are not substitutes; each is the right one on one
side of the reach boundary.

**`Nf` alone is not a candidate.** Its whole-class july projection is **−303 s² [−2,058,
+697]**, so it fails the registered gate, and its served-cells-only reading (+136 / +72 /
+49, LTFM alone) is smaller than the candidate's (+246 / +203 / +96) because it gives up
EDDF. What it establishes is narrower and worth having: **the LTFM half of what this lane
serves does not need the harvest at all**, and an arm chosen per cell — `Nf` at LTFM, `N`
at EDDF — projects above either arm alone. Choosing it on these same three folds is a
second selection on top of the hold's, so it is written here as the next step's question
and not as a change to the file. Nothing here is served: the candidate the desk accepted
is unchanged.

### 7g. The per-cell form, registered as a rule — and where the gate stands

**It is a rule, not a pick** (desk, 2026-09-09). The programme's own rule is that an
absent observation is never a feature. On an UNREACHED row every `No_*` column is
exactly that — the network heard no other aircraft in the cell, so the value the head
reads is the fitting rows' imputed mean — and standardised beside columns that do vary,
it dilutes the correction. So the arm follows the reach class: **an unreached cell
carries `Nf_*` only, a reached cell carries both families.** The form lives in
`arms.SPLIT_FORM` and is read once by both `code/arms.py` and `code/serve.py`; both heads
are fitted on the same fitting rows (`--fit-rows all`), and only the columns a scored
row's reach class carries change. The reading rejected: fitting each head on the rows of
its own reach class, which would move the columns and the fitting population at once and
halve each head's data, while `Nf_*` is defined identically on both classes.

**The second selection, counted.** The form was chosen after reading T7j, on these same
three folds. That is the second selection on them — the cell hold was the first — and
both are in the same direction: keep what read well here. §7c's caveat therefore applies
twice over to the numbers below, and the honest statement of it is that this lane has now
used the three fold months to choose a serving population AND a functional form, and
nothing in these folds can price what that costs. **T7k** is the table.

**What the form gives.** The served leg rises on every fold and clears zero on every
fold: **+278 [+185, +389] / +243 [+121, +409] / +109 [+47, +179]** against arm `N`'s
+246 / +203 / +96. The two arms serve the same two cells, and `EDDF/reached` is identical
to the digit under both, which is the check that nothing but the unreached columns moved.
Cross-fitted with the hold applied on the fold side, the currency is **clean 256.4643 →
256.1919, +0.2724 s** (the accepted candidate's is +0.2231) and all-row 308.5977 →
308.3712, +0.2264 (against +0.1854). Every column of the four-way RMSE on the rows the
file would move improves except january's no-NM on 326 rows — and **december's tail
(1,547.7 → 1,538.8) and no-NM (317.5 → 316.7), which the accepted candidate makes worse,
are improvements under this form.**

**The dilution argument does NOT generalise to every dark cell**, which is what the desk
asked to have read. At `LTFM/unreached` the file clocks alone are worth +230 / +292 / +91
s² per row more than arm `N` on the same rows. At `LFPG/unreached` — 34,108 ranking rows,
the biggest held cell — `Nf` alone reads **−3 [−269, +230] / +137 [−79, +385] / +10
[−683, +698]**, against arm `N`'s −162 / +213 / +297: better on july, worse on january,
and clear of zero on nothing. At `LIRF/unreached` it reads −9,175 / +150 / −34 against
+8,765 / +666 / −94. Both cells stay held under both forms. LFPG is not a dilution
problem; it is an empty cell for both families, which is what §7d already said with the
pooled read, and the two forms disagreeing in direction fold to fold is what that looks
like from the other side.

**The selection, priced as far as three folds allow.** The form cannot be validated on
the folds that chose it, but it can be asked whether any TWO of them would have chosen
it and the third agreed. `LTFM/unreached` is the only unreached cell the hold serves, and
`Nf` beats arm `N` there on every fold separately — **+230, +292 and +91 s² per row** —
so every pair of folds selects the same rule and the held-out fold confirms it, three
times out of three, with each fold's own interval clear of zero under `Nf`. That is not a
price for the selection; it is the strongest statement three months support, and the
fourth harvested month is what would price it.

**Where the gate stands, and the one thing that is the desk's to settle.** The
assignment's advance clause — every fold interval clear of zero at full amplitude on the
population served — is **met**: the served leg passes on all three folds. The second leg
this lane added in round 3, the projection over the WHOLE no-clock class, **fails july at
−130 [−1,704, +881]**, and one held cell is the whole of it. `LIRF/unreached` has 141
july fold rows, is extrapolated 125-fold onto 17,651 ranking rows, and reads −9,175 s²
per row [−39,579, +10,010] under the file clocks against +8,765 [+3,397, +29,433] under
arm `N`: the same 141 rows, opposite signs, and neither number is a measurement. Drop
that one cell and july is +340 [+222, +475] under the per-cell form against +293 [+187,
+408] under arm `N`; no other cell moves july's total by as much as 12 s². The cause is
**two floors for one measurement**: `project()` prices any cell with 100 fold rows, and
the hold refuses to serve one read on fewer than 500. Priced at the hold's own floor,
every fold of both arms clears zero (**+333 / +224 / +141** for the per-cell form,
+286 / +218 / +154 for arm `N`; `analysis/split-gate.json`). Which floor the gate should
read is a rule, and rules here are the desk's; this lane does not move a gate to let its
own candidate through. **Nothing is served.** When the desk settles it, the mask basis
for the serve is surfacefit's per-month **v15** class vector, which `serve.py` now takes
with `--class-tag v15`; the accepted v14 candidate is untouched and its rebuild through
the edited serving path is bit-identical at `519ca89f…`.

### 7h. One floor for one measurement, and the v15 slot

**The rule (desk, 2026-09-09).** The projection reads cells at the SERVING floor: a cell
read on fewer than 500 fold rows in a fold is unservable there and carries **no
projection**, its rows counted as **unpriced, not as a gain of zero**. It lives in
`arms.MIN_CELL = MIN_FOLD_ROWS` — one constant, so the two can no longer drift apart.

**The defect it repairs.** Until this rule the projection priced a cell at 100 fold rows
while the hold refused to serve one under 500, and on july the gate then rested entirely
on a cell nothing may serve: `LIRF/unreached`, **141** july fold rows extrapolated
**125-fold** onto 17,651 ranking rows, reading **+8,765 s² per row [+3,397, +29,433]**
under arm `N` and **−9,175 [−39,579, +10,010]** under the file-clock family. The same 141
rows, opposite signs, and neither number is a measurement. It flattered arm `N`'s july
projection by +449 s² and damned the per-cell form's by −470 — the difference between a
gate that passes and one that fails, decided by 141 rows.

**What the rule changed, and what it did not.** Arm `N`'s whole-class projection reads
**+286 [+178, +401] / +218 [+31, +454] / +154 [+79, +233]** where it read +741 / +225 /
+180, and **every fold is still clear of zero: v14's gate passes under the rule too.**
The candidate's bytes are untouched — rebuilt through the re-read record it is
bit-identical at `519ca89f…` with the same currency (+0.2231 clean) — because the hold's
served set never depended on the projection's floor. What the rule costs is honesty about
coverage: the unpriced weight it now declares is **7.1 / 2.4 / 6.7 %** of the class's
ranking rows (6, 7 and 9 cells), and filling every unpriced cell with its fold's
class-pooled gain instead of nothing gives +307 [+190, +433] / +219 [+27, +459] / +185
[+87, +287]. The gain forgone by the hold falls to +40 / +14 / +58 s², because most of
what the hold used to "give up" was that one cell's fiction. Under the same rule the
per-cell form passes both legs on every fold (T7k). Ledger CORRECTION 6.

**The v15 composition.** The base bytes are `exp-046-surfacefit`'s v15 candidate
(`submissions/candidates/exp-046-surface-v2.parquet`, sha `eabd824d…`), which is built on
top of this lane's v14, and the mask is surfacefit's **per-month** class vector
(`artifacts/exp-046/class/<month>-v15.parquet`, manifest `class-manifest-v15.json`, moved
column `changed_by_v15`). Because the base already carries this lane's own update on the
served rows, the write **replaces** it rather than stacking: an owned row is written as
**v12 + d**, never base + d, since the correction is priced against v12's fold side. The
dry run verifies exactly that — `max|base − v12|` on owned rows is 313.07 s and
`max|base − this lane's accepted candidate|` on the same rows is **0.0**, so what is
replaced is this lane's bytes and nobody else's; the served set is unchanged at 63,862
rows in the same two cells (class rows 154,357 under the new vector, one fewer than
before); the overlap with `changed_by_v15`'s 188,494 rows is **0**; the 388 `noac` rows
are bit-identical; no row is clipped at the floor. The file moves **47,577 rows against
v15 — LTFM's, and only LTFM's** — because a reached cell is arm `N` under both forms, so
EDDF's 16,285 rows keep v14's bytes to the last bit.

**The gate for the slot, and the verdict: MAYBE.** The desk's rule for v15 prices the
INCREMENT over arm `N` on the served rows, per fold, with day-clustered intervals, and
every fold must be clear. **T7l**: `cross` reads +226 [+55, +415] / +222 [+62, +463] /
+78 [−77, +234] and `rotation` +193 [−23, +440] / +271 [+63, +570] / +78 [−78, +240].
**January does not clear under either convention**, and july does not under the
registered rotation. (January's two convention lines are one estimate drawn twice — see
§7i and ledger CORRECTION 7 — which changes nothing here, where january misses by −77 at
2,000 draws and by −74 at 20,000.) So the candidate is a **MAYBE and takes no slot**, and no candidate
file is written. The point estimate is positive on every fold and both conventions, the
three folds pooled are clear (+178 [+77, +294] and +181 [+55, +313]), and pooling is not
the rule.

**The revival condition.** The increment is `LTFM/unreached`'s alone — `EDDF/reached` is
exactly zero by the rule — so what has to clear is one cell on one fold: january's
**+91 s² per row on 21,368 rows, half-width ±183**. At the same point estimate that
interval needs about **4.2× january's evidence** to reach zero, which 31 days of one
month cannot supply. It revives when **the harvest delivers a fourth fold month that
prices LTFM's unreached cell** — a second winter reading beside 2025-01 — or when a form
with a smaller variance of the DIFFERENCE (a `Nf` head shrunk toward `N` at LTFM rather
than replacing it) makes the same gain steadier. It does not revive by pooling the folds,
which is the reading §7c rejects for serving, and it is not a reason to re-open the
per-cell form itself: that form's own gate (T7k) passes on both legs and all three folds.

### 7i. The cheap revival taken — the `Nf` head at LTFM shrunk toward `N`

**What was ordered and what was fixed first.** §7h names two ways the v15 slot revives; the
desk ordered the cheap one — the file-clock head at LTFM shrunk toward arm `N`, **the weight
0.5 fixed before the read and no weight search** — gated on the same increment. The weight
went into `arms.FORMS` as the variant `N_reach_half` before `work/shrunk.sh` ran; nothing
else moved. A reached cell is arm `N` under every form, so this changes LTFM's rows and no
others.

**Step 2, at the one-floor rule (T7m).** The whole-class projection reads **+315 [+204,
+438] / +242 [+75, +438] / +165 [+96, +242]**, the served cells **+265 [+180, +359] / +229
[+115, +385] / +109 [+52, +176]**; every leg of every arm clears zero on every fold. On the
served cells the shrunk form sits BETWEEN arm `N` and the replacement form on july and
december and equals it on january, which is the point: shrinking is not there to raise the
gain but to steady it.

**The slot gate, and a defect it exposed.** The increment over arm `N` on the moved rows
read `rotation` +114 [+4.8, +239] / +173 [+62.5, +333] / **+80 [+0.5, +162]** and `cross`
+120 [+34.3, +215] / +129 [+45.5, +255] / **+80 [−0.1, +161]** — clear on every fold under
one convention and not under the other, on the same fold, by 0.6 s². That disagreement is
not real: `arms.ROTATION["january"]` is `["july_loo", "december"]`, which IS january's cross
set, so **january's two lines are one estimate on identical rows**, and they differ only
because `work/increment.py` makes one bootstrap generator and passes it through every call
in order, so each line consumes a different stretch of one stream. The record's earlier
sentence "*so the verdict cannot turn on which*" (T7l) is wrong for january and is corrected
there; ledger CORRECTION 7. It cost nothing on the v15 candidate — january missed by −77
there, not by −0.1 — but here it was the whole verdict.

**The second read, its rule fixed before it ran.** `work/increment_mc.py` re-prices the same
paired per-row difference at **20,000 draws at the registered seed and at ten 2,000-draw
seeds**, and PASSES only if every fold is clear under both conventions at both. Nothing about
the estimator, the folds, the rows or the weight changes; only the draw. January reads
**+1.2** at 20,000 draws and its **worst of ten seeds is −3.5**, and the cell that actually
moves reads january **+93 [−0.1, +186] on 21,368 rows** under the very draw whose pooled line
reads +0.5.

**Verdict: a second MAYBE. No candidate file is written, and v14 stays this lane's serve.**
The mechanism §7h predicted did work — january's point estimate hardly moves (+77.7 → +80.0
s² per row) while the half-width of its interval falls from ±155 to ±81, the variance of the
difference down by about three quarters — and it carries the lower bound from −74 to +1.2 and
stops. A weight search would very likely find a weight that clears; that is exactly why there
was none.

**What it would have been worth, for the record.** The same two cells are served
(`EDDF/reached`, `LTFM/unreached`), so the file would have moved the same 47,577 LTFM rows
on v15's bytes and nothing else. Cross-fitted with the hold on the fold side its currency
is **clean +0.2548 s** (256.4643 → 256.2096) and **all-row +0.2117** (308.5977 → 308.3859),
between the shipped v14's +0.2231 / +0.1854 and the replacement form's +0.2724 / +0.2264;
on the rows it would move, july goes 258.7256 → 256.5981 all-row and 2,669.5 → 2,630.9 in
the tail, december 200.7134 → 199.4616, january 211.5272 → 210.3713, and january's no-NM
rows get worse (461.74 → 464.03 on 326). `analysis/shrunk-currency.json`,
`work/shrunkcur.py`. **No candidate file was written.**

**The revival condition.** Unchanged in kind, and the distance is now measured. At the
same point estimate january's interval has to narrow by a factor 80.78/79.97 to put its
bound above zero — **about 1.02× the evidence it has**, against the **4.2×** §7h measured
for the replacement form. The gate is missed by that much and by seed noise of ±4 s²; a
fourth fold month is what supplies it — a second winter reading beside 2025-01 that prices
LTFM's unreached cell — and not a different weight on the same three. That month is not
cheap: it needs the base head's out-of-fold vector for it, so it stays a recorded revival
and not an order. The two folds that clear (july +9.6 to +237, december +60.2 to +334 at
20,000 draws) say what is waiting there. **The lane's next step is the coordinated
`block.py` tie-break rebuild with `exp-046-surfacefit`, after v15 is up** (desk,
2026-09-09).

### 7j. Is the MAYBE january's alone? — the tail and the no-NM rows on the served cells

**The question (desk, 2026-09-09).** Two RMSE figures in §7i move the wrong way:
december's tail 1,547.67 → 1,548.83 and january's no-NM rows 461.74 → 464.03, while every
fold's all-row and bulk RMSE improves. If the shrunk head also loses the tail or the no-NM
rows on the folds that CLEAR, the second MAYBE is a mixed shortfall and the revival is a
different question. So each subgroup of the served rows is priced the way the gate prices a
fold — per row, day-clustered, 20,000 draws re-seeded per call — for the shrunk head's gain
over the head and for its increment over arm `N` (T7n).

**Answer: it is january's alone, and inside january it is the bulk's.** January's whole
shortfall is its **24,942 bulk rows at +77.9 [+0.3, +157.6]** — the same knife edge as the
fold, because the bulk *is* the fold. Its tail is **12 rows** contributing +2.1 s² and its
no-NM rows **326** contributing −0.4, neither established in any direction.

**Neither wrong-way RMSE is a loss to the shrunk head.** December's tail is **36 rows**:
against the head it reads **−3,587 [−35,753, +149,346]** cross-fitted and **+28,754
[+3,522, +201,900]** at the rotation — the sign is the draw, not the data — and against
**arm `N`, which is what v14 ships on those rows today**, the shrunk head is better under
both (RMSE 1,538 / 1,549 against `N`'s 1,552 / 1,559). January's no-NM rows read **−2,113
[−9,545, +3,769]** against the head, and arm `N` is equally bad there (463.99 against
464.03) — a property of the correction this lane already serves, not something shrinking
introduced. **On every fold and under both conventions the shrunk head beats arm `N` in the
tail.** So the record's answer to the desk is: **a january-only shortfall.**

**What the read did turn up, and it is about the folds that PASS.** December's increment
leans on those 36 tail rows for **+63 of its +173 s²** at the rotation and **+48 of +129**
cross-fitted — **37 % of a fold's verdict from 0.1 % of its rows**, on a subgroup whose own
interval is three orders of magnitude wide. July is the same shape and smaller (+30 of +114,
+26 of +120). The verdicts do not depend on it — **december's bulk alone clears under both
conventions, +109.5 [+49.4, +171.7] and +81.7 [+40.7, +131.4]** — but july's bulk clears
cross-fitted (+94.9 [+21.8, +183.1]) and not at the rotation (+83.4 [−6.9, +192.0]), so on
the bulk alone the shrunk head would be **two folds of three**, not three, and january would
still be the fold that decides. This is a caveat on the two folds that passed, recorded
where the MAYBE is recorded; it moves no gate, because the gate is the fold and the fold
includes its tail. `analysis/tail-nonm.json`, `work/tailnonm.py`.

### 7k. The harvest's receipt correction — measured, and it moves nothing here

**The order (desk, 2026-09-09, three hours before v14's slot).** `exp-045-harvest` found
that `census.rate()` aggregated `min(t_first)` / `max(t_last)` per `(airport, callsign)`
over the whole month before testing the take-off window, so a flight number flown daily
counted as received on every day of the month (commit `f8352a4`). The desk's reading was
that this lane's `_sc` columns divide by that rate per airport-day, so v14's features
would have been scaled by wrong denominators on both sides of the fold — and ordered the
five blocks rebuilt, the `_sc` columns diffed per month, and arm `N` re-priced before the
slot.

**The bias is real and month-dependent.** Committed against corrected: 2025-01 0.5413 →
0.4800 (×0.887), 2025-07 0.6095 → 0.4961 (×0.814), 2025-12 0.7558 → 0.6521 (×0.863),
2026-01 0.7227 → 0.6269 (×0.867), 2026-07 0.6487 → 0.5609 (×0.865). So the answer to the
desk's question — *does the bias differ between the 2025 folds and the 2026 months?* — is
**yes, by about a point of ratio between the two groups, and by seven points inside the
folds** (2025-07 against 2025-01), which is the larger of the two spreads and the one that
would matter for a fold-to-ranking transfer.

**The premise does not hold: the `_sc` columns never divided by that rate.**
`block.receipt_rates()` measures its own denominator on the block's own matched matrix —
the share of an airport-**day**'s departures whose ground run the matcher identified
(`obs_has_clock`), over the airport-days the summaries delivered — and in this lane
`receipt.json` is read at exactly three lines, all inside `outage_report()`, which writes
warnings. The two are not near each other: 2025-07 EGLL is **0.0577** here against the
harvest's corrected **0.1506**, LFPG **0.0182** against **0.0476**, because a matched
250 m clock is a stricter test than hearing the callsign in the window. (The matcher
itself does not carry the same defect: `exp-046-surfacefit/code/block.py` joins per ground
SEGMENT and tests each appearance's own window.)

**Measured anyway, because a path argument is not a measurement.** All five blocks were
rebuilt with the corrected artifact in place, against the published bytes copied aside
first: **every column of every month is identical, value for value, in the same row
order** — all 186 columns, not only the 28 `_sc` ones (`analysis/block-rebuild-diff.json`).
The parquet SHA-256 does change on a rebuild, because the container is not written
deterministically; the published bytes were restored, so `analysis/interface.json` still
pins the shas surfacefit verified.

**The gate, re-priced (T7o).** Arm `N` at one floor with the cell holds, on the corrected
artifact: whole class **+286.43 / +217.50 / +153.56**, served cells **+246.14 / +203.05 /
+96.05**, the same two cells of nineteen served — **every point estimate identical to the
last decimal** to the registered record, every interval clear of zero on the same side.
The intervals wobble by a few s² only because a one-arm run draws a different stretch of
the bootstrap stream than the six-arm run that wrote the record (ledger CORRECTION 7).

**Verdict against the desk's rule.** The gate passes and the served-cells gain has not
moved at all, so **v14's reviewed bytes (`519ca89f…`) go up as gated**; there is no
corrected scaling to carry into the next serve, because the scaling was never the
harvest's. What the correction does change is what the record may CLAIM about coverage:
§6a's estimate of the EDDF outage hold, which quoted the harvest's rates, is now about
1,300 clocks rather than 1,500, so the measured 591 is 45 % of the estimate and not 40 %.
`analysis/receipt-correction.json`, `work/scdiff.py`, `work/recheck.sh`,
`work/repriceN.sh`.

### 7l. BOARD READING 13 — what the fold price predicted, and what the board paid

**The file went up as v14** (`youthful-giraffe_v14.parquet`, this lane's `519ca89f…`,
composition re-checked clean by the desk) at **00:03:52 UTC on 2026-09-09** and scored
**248.2904** at 00:04:15, against v13's 248.4818: **−0.1914 s of board**, rank 1, 12.641 s
ahead of the next team. Desk ledger rows *submission v14* and *BOARD READING 13*, commit
`c46c415`.

**The prediction this record made, and the outcome, side by side.** The candidate's
cross-fitted fold-side currency was **all-row +0.1854 s** and **clean +0.2231 s** (§0, §7c,
T7). The board paid **−0.1914 s**, so the transfer is **1.03× on the all-row price and
0.86× on the clean one** — the all-row number is the one the board's own base makes
comparable, and on it the update landed **at face**.

**Why that is worth recording.** It is the first surface gain in this competition to
transfer at face: `exp-046-surfacefit`'s with-clock own-observation gain transferred at
**1.75×** (BOARD READING 12). The two are consistent with the mechanism this lane
measured — a neighbourhood heard through the *same instrument* as the folds prices at fold
worth, while an aircraft's own clock on the 2026 rows was worth more than the 2025 folds
could price — and equally consistent with the served-cells interval being post-selection
(two cells of nineteen, §7c). One reading cannot separate those, and this record does not
try to.

**Nothing here is revised on it (score-feedback rule).** No gate moves, no hold is lifted, no candidate is
re-scored, and the two MAYBEs — the per-cell form (§7h) and the shrunk head (§7i) — stay
MAYBEs on their own fold evidence and not on this delta. The reading enters the record as
a calibration fact only: for this lane's population, **a no-clock neighbourhood gain is
worth about 1.0× its all-row fold price on the board.**

**BOARD READING 14 narrows the pair above, from the other lane's side.** v15 —
`exp-046-surfacefit`'s 127-leaf head served on v14's bytes, this lane's 63,862 rows
untouched — scored **246.8263** at 00:15:45 UTC, **−1.4641 s on v14**, so its with-clock
increment transferred at **2.01×** on +0.7296 (desk ledger BOARD READING 14, `1e1450e`).
Three readings now exist on one file each: **1.75×, 2.01× with a clock, 1.03× without
one.** A fold-to-board mapping that inflated every gain would have paid this lane's update
above face too, and it did not, so the multiplier is the with-clock lane's own information
rather than a property of the mapping. What it does NOT settle is the post-selection
question on this side: two cells of nineteen were chosen on these folds, and that would
make a served-cells price optimistic — that is, it would show as a transfer **below** one,
which this reading does not show. That is evidence about the size of the post-selection
penalty, not a measurement of it, and it changes nothing here (score-feedback rule).

### 7m. The fold population is PINNED (desk policy, 2026-09-09) — and july has already drifted

**The rule.** A fold block is pinned to the airport-day population it was priced on. Each
fold month gets a fold-population file — its airport-days, their summary shas, and the
block key — that pricing, the gate, the dry run and the serve all read; a new harvest
delivery enters a fold only at a **recorded re-price**. It came from surfacefit's side:
the harvest's 2025-06 delivery put 2025-06-30 into its july fold block (9 airport-days,
+3.5 % of segments, departure rows unchanged) and its guard refused to serve v16 on a
population the gate had not been read on. This lane writes its pin files at the
coordinated `block.py` rebuild and re-prices once on the current population; the desk's
`MATCHER_SHA` line will name it. **No action was taken now, and `block.py` is frozen.**

**Where this lane stands, measured read-only on 2026-09-08 at 22:0x UTC.** The blocks v14
was priced on reported `days_present` **278 (july_loo) / 288 (december) / 218 (january)** —
the same populations tonight's re-price reproduced to the last decimal (§7k). Counting
what is on disk now, on the registered summaries rule `d4c59868…`, over the loader's
window: **july 296 airport-days (278 inside the month + 18 edge), december 297, january
225.** July's 18 edge days are `2025-06-29` and `2025-06-30` at nine airports, and every
one of those files was written at **21:59 UTC on 2026-09-08 — an hour after the 20:59
pricing run.** So the drift the policy exists to stop is real here too, it is july's, and
v14 was priced and gated before it arrived.

**What that does and does not mean.** Nothing served moves: v14 is up, scored, and was
priced on the 278-day july population; the pin will record exactly that. The december and
january gaps between "on disk" and "loaded" (297 against 288, 225 against 218) are the
loader's own window and hold rules rather than a delivery, and the pin files will state
each fold's population explicitly instead of leaving it to be inferred — which is the
point of the policy. At the rebuild the re-price is **one** re-price, recorded, on the
current population, and this record will carry the before-and-after per fold.

### 7n. The matcher moved to `eded54e5` — every serving guard in this lane now refuses

**What moved (desk, 2026-09-09).** `exp-046-surfacefit/code/block.py` went from
`ba41935e…` to `eded54e5…`: pin plumbing at the summary-path choke point, inert when
unpinned, the matching function unchanged, and round 5's reviewer confirmed the census
identical. The blocks this lane built on `ba41935e` stay valid and **nothing is rebuilt
now**; the final sha at the coordinated rebuild will differ again.

**The consequence, verified on disk before any composition.** `serve.rank_frame()` carries
three hard checks and **all three trip today**, because each record was written against
`ba41935e…` while the matcher on disk hashes to `eded54e5…`:

| guard | record | expects | on disk |
|---|---|---|---|
| the interface was written against the matcher | `analysis/interface.json` (`matcher_block_py_sha256`) | `ba41935e…` | `eded54e5…` |
| the step-2 GATE was measured against it (round 3, C2) | every `arms-step2-*.json` this lane serves from | `ba41935e…` | `eded54e5…` |
| the shared class vector was written against it | `exp-046/analysis/class-manifest.json` and `-v15.json` | `ba41935e…` | `eded54e5…` |

So a serve attempted from this lane right now exits with *"re-emit before serving"*,
*"re-run step 2 before serving"* and the class-vector line — which is the guard doing
exactly what round 3 asked for: it cannot tell an inert edit from one that moves the class
boundary, and it is not supposed to try. **Nothing is changed to accommodate it**: no
guard is relaxed, no record is re-stamped, and no candidate is composed. The three checks
are cleared at the coordinated rebuild in the same recorded step as the pin and the single
re-price (§7m) — new blocks, a new gate record and the class vector all carrying the
`MATCHER_SHA` the desk names — and until that line arrives this lane serves nothing, which
is what it was already standing by to do.

### 7o. Cross-fitted selection (desk rule, 2026-09-09) — what it governs in this lane

**The rule.** Any increment whose FORM was selected on the folds is priced cross-fitted:
select on two folds, price on the third. It comes from BOARD READING 15 — v16's increment
transferred at **0.36×** on a fold price of +0.3350, below its fold interval, its leaves
having been chosen by a climb on the folds that priced them and its tail guard touching a
wider ranking footprint than was priced. That is post-selection, and it is the risk this
lane's own 1.03× reading flagged from the other side (§7l).

**Three selections were made in this lane, and they are not in the same position.**

1. **The cell hold** — 2 cells of 19, chosen on all three folds. This is the one selection
   in this lane that is **not** cross-fitted anywhere in the record. Every served-cells
   number carries it: v14's +246 / +203 / +96 and its +0.2231 clean currency. Round 4's
   reviewer named it (item 3), §7c states it, and it is unpriced.
2. **The per-cell form** (`Nf` on unreached cells, arm `N` on reached), chosen after
   reading T7j on the same three folds. §7g already asked the question the new rule asks:
   would any TWO folds have chosen it, and does the third agree? At `LTFM/unreached` — the
   only unreached cell the hold serves — `Nf` beats arm `N` on **every fold separately,
   +230 / +292 / +91 s² per row**, so every pair selects the same form and the held-out
   fold confirms it, three times of three. Under the new rule the per-cell increments
   already in T7l and T7m are read as select-on-two, price-on-the-third, and they are
   unchanged: that is what those tables measure.
3. **The shrunk weight** 0.5 was fixed before the read and never searched (§7i), so the
   rule does not touch it. Its increment is still a MAYBE on january's own evidence.

**What this changes here: nothing now, one thing at the rebuild.** No number is revised
(score-feedback rule) and no candidate is composed. At the coordinated rebuild the single recorded
re-price (§7m) will **price the cell hold cross-fitted as well** — the served set chosen
on two folds and the gain read on the third, three ways — so the one uncrossed selection
in this lane gets a price instead of a caveat. The board's own evidence bounds it from
above in the meantime: a served-cells price inflated by selection would have transferred
**below** face, and v14 transferred at 1.03× (§7l).

### 7p. The coordinated rebuild — re-emitted, pinned, re-priced once, and the hold paid for

**What was done** (desk `MATCHER_SHA` line, 2026-09-09). The matcher is
`6fdf3951…` (commit `60787a8`, byte for byte the candidate read in §7n's answer;
`MATCHER_SHA_BEFORE_THE_REBUILD` `695af973`). On it: the five interface blocks were
re-emitted, the fold-population pins were written, and arm `N` was re-priced **once** on
the current population, with the cell hold cross-fitted.

**The gate, before and after** (arm `N`, one floor, same cell-hold rule):

| fold | whole class, before → after | served cells, before → after | served cells CROSS-FITTED |
|---|---|---|---|
| july_loo | +286.43 [+178, +401] → **+293.58 [+190, +406]** | +246.14 → **+253.24 [+171, +345]** | **+253.24 [+171, +345]**, same set |
| december | +217.50 [+31, +454] → **+224.71 [+32, +453]** | +203.05 → **+211.61 [+97, +370]** | **+227.12 [+111, +387]**, set adds `LEBL/reached` |
| january | +153.56 [+79, +233] → **+156.54 [+80, +237]** | +96.05 → **+99.04 [+42, +167]** | **+99.04 [+42, +167]**, set adds `LEMD/reached`, unpriced here |

**Every fold is clear of zero on both legs and every leg moved up.** The served set is
unchanged — `EDDF/reached` and `LTFM/unreached` — so the file this lane would write serves
the same two cells.

**The hold, cross-fitted, costs nothing.** Choosing each fold's served set from the OTHER
TWO folds by the same rule and reading the gain on the held-out fold gives **+253.24 /
+227.12 / +99.04** against the in-sample **+253.24 / +211.61 / +99.04**: at or above its
twin on every fold. The out-of-fold rule admits one extra cell on two folds (`LEBL/reached`
on december, which december prices; `LEMD/reached` on january, which january does not price
and which therefore contributes nothing, exactly as the one-floor rule requires). So the
one selection this lane had never paid for (§7o) is now priced, and the price is **zero to
the resolution of these folds** — consistent with v14 transferring at face rather than
below it (§7l).

**The pins** (`analysis/pin/<fold>.json`, manifest `analysis/pin-manifest.json`). Each
records the airport-days, each day's summary SHA-256, the summaries rule, the population
key over exactly those, and the run that set it — **the matcher sha is recorded beside the
key and never inside it**, which is the desk's rule and the reason surfacefit's cache trap
cannot bite here (there is no block cache in this lane, and `block.BLOCK_CACHE = None`
carries the instruction that the matcher sha goes in a cache FILENAME if one is ever
added). Populations: **july 296, december 288, january 218** airport-days.

**July's population had drifted twice, not once.** v14 was priced on **278**; the pin's
history carries that population's own key. Two deliveries fell inside the loader's
one-day window afterwards: **2025-06-30** (9 files at 21:59 UTC on 2026-09-08, an hour
after the run that reproduced v14's gate) and **2025-08-01** (9 files at 00:28 UTC on
2026-09-09). This recorded re-price admits both; december and january are byte-for-byte
the populations v14 was priced on.

**Both guards were proved by negative test, not by inspection.** An added airport-day and
a changed summary file each stop a run (`SystemExit`, naming the added, removed and
changed keys); a recorded re-price admits them; and `check_matcher()` stops any build
whose `exp-046-surfacefit/code/block.py` is not the declared `MATCHER_SHA`.

**What the re-emitted interface moved** (`analysis/interface-rebuild-diff.json`,
`-magnitude.json`). 216,521 of 854,937 rows differ in at least one column, and the shape
of that is what matters: `obs_has_clock` flips **0 / 1 / 13 / 511 / 9** by month (2026-01's
511 are the EDDF held-days repair, as the spec said), while the bulk is *relative* and
*anomaly* columns re-based by a hair — july's `Hob_s_taxi_rwy_rel` moves on 54,988 rows
with a median |Δ| of **7.6e-05** on a typical value of 1.05, and july's `route_km`,
`stand_dist_km`, `Nf_n_n_cell`, `No_n_n_cell` and `own_obs_taxi` do not move on a single
row. On 2026-01 the substantive moves are real and expected: `No_n_n_cell` on 842 rows and
the stand centroid on 15,321 rows (median 12 m), because 569 newly matched EDDF rows add
their stand positions to their groups. Interface manifest
`d73c1d80ffde319e9fbbcfdc2f17f021e5b8eacfc2aba6deaa3af62f1bbcd6f9`; block shas
`173e49ef` / `f1662256` / `5da0a4ee` / `bc09ae75` / `10b32c80`.

**The gate record is re-issued, not patched** (`analysis/arms-step2-rebuild.json`, matcher
`6fdf3951…`). The pre-rebuild records stay as they are: they describe v14's population and
v14's matcher, which is what a board reading needs to be readable against.

### 7q. The lane's next steps, priced — the desk picks

Nothing here is started. Every figure is **at face** — seconds on the all-row board at the
1.0× transfer BOARD READING 13 measured for this lane's population (§7l) — and every one is
a point estimate whose interval is in the row beside it. The two MAYBEs' currencies were
priced before the rebuild; the rebuild moved every projection up about 2.5 % (§7p), so they
are floors rather than centres.

| option | kind of work | cost | upside at face | what it gives up | the line that decides it |
|---|---|---|---|---|---|
| **A. The shrunk head at LTFM** (`N_reach_half`, §7i) | serve an already-measured form | none to measure; one serve and one review round | **+0.026 s** over v14 (all-row currency +0.2117 against +0.1854) | nothing served today; it spends a slot on the smaller of the two forms | january's increment clear of zero at 20,000 draws **and** across ten seeds — it is **1.02×** the evidence short |
| **B. The twelve revivable held cells** (45,493 ranking rows, §7d) | measure: more fold months, then the same gate | one fold month per revival attempt, three lanes' vectors each (option C) | **+0.130 s** if every pooled point estimate held, and **three cells are 90 % of it** — `LEMD/reached` +0.0485, `EDDF/unreached` +0.0306, `LEBL/reached` +0.0287 | nothing served today; two of the twelve are measured NEGATIVE and would be held anyway | each cell's own fold interval clear of zero above on **every** fold at 500+ rows |
| **C. A fourth fold month** (§7i's revival) — *deferred by the desk on 2026-09-09, §7s; the cost below was measured wrong here and the spec corrected it* | measure: extend the fold side | **three vectors from three other lanes** (below) plus about five minutes here — **the spec found SEVEN member refits under those three vectors** (`analysis/fold-month-2025-02-spec.md` §2), about ten Kaggle kernels and one to two worker-days | it is the enabler, not a gain: it is what can turn A, D and most of B from MAYBE into serve | other lanes' compute, and a fourth fold that every future gate must also clear | **revival: a candidate worth 0.5 s or more at face in either surface lane that is a MAYBE on january alone** |
| **D. The per-cell form** (`N_reach`, §7g–§7h) | serve an already-measured form | none to measure; one serve and one review round | **+0.041 s** over v14 (all-row currency +0.2264 against +0.1854) | nothing served today; it is the larger form and the further from clearing | january's increment clear of zero — it is **4.2×** the evidence short |

**B in detail — the twelve cells, why each is held, and what it is worth.**

| cell | why it is held | ranking rows | pooled s²/row [95 %] | evidence needed | at face |
|---|---|---:|---|---:|---:|
| `LIRF/unreached` | too thin to say | 17,651 | +128 [-174, +472] | 6.4× | **+0.0106 s** |
| `LEMD/reached` | instrument | 8,739 | +1,180 [+784, +1,636] | pooled already clear | **+0.0485 s** |
| `EDDF/unreached` | too thin to say | 5,337 | +1,220 [-321, +3,718] | 2.7× | **+0.0306 s** |
| `LFPG/reached` | too thin to say | 4,152 | -248 [-951, +347] | measured negative | **-0.0048 s** |
| `LIRF/reached` | too thin to say | 3,567 | +131 [-1,761, +2,216] | 229.3× | **+0.0022 s** |
| `LEBL/reached` | month to month | 3,214 | +1,901 [+1,057, +2,838] | pooled already clear | **+0.0287 s** |
| `LEMD/unreached` | instrument | 1,382 | +179 [+50, +329] | pooled already clear | **+0.0012 s** |
| `EDDM/reached` | too thin to say | 929 | +2,730 [-495, +6,250] | 1.5× | **+0.0119 s** |
| `LEBL/unreached` | too thin to say | 275 | +413 [-389, +1,454] | 5.0× | **+0.0005 s** |
| `LSZH/unreached` | too thin to say | 121 | +778 [-562, +2,343] | 3.5× | **+0.0004 s** |
| `EHAM/unreached` | too thin to say | 81 | -865 [-4,102, +1,991] | measured negative | **-0.0003 s** |
| `EDDM/unreached` | instrument | 45 | +285 [+12, +642] | pooled already clear | **+0.0001 s** |

`LFPG/reached` and `EHAM/unreached` are measured negative and are in the twelve only
because more months could still move them; the five cells in §7d held by the ARM
(45,004 rows, `LFPG/unreached` chief among them) are not in this table at all, because more
months cannot revive what is measured to be absent.

**C in detail — exactly what a fourth fold month needs, and from whom.** This lane's fold
side is not self-contained: a fold month exists for it only when three vectors exist for
that month, and none of the three is this lane's to make.

| what | where it comes from | who produces it | what it costs them |
|---|---|---|---|
| the **v10 panel** for the month (`p10`, `y_true`, the day and cell keys) | `exp-039-screens`, `v10.load()` — `base.V` | exp-039-screens | one panel build for one month |
| the **stack-5** vector for the month | `exp-038-packet`'s gated fold vectors — `base.V38` | exp-038-packet | one gated vector for one month |
| the **base head's out-of-fold vector**, `oof/cand-k2c-mod-refit-bulk-<fold>.parquet` | `exp-035-k2climb` | exp-035-k2climb | **the expensive one**: the climbed member refitted with that month held out, the same refit it already runs per fold |
| the harvest **summaries** for the month | `exp-045-harvest` | already delivered for 2025-02, 2025-03 and 2025-04 | none |

Then this lane needs one block build and one re-price, about five minutes at two threads,
and the pin machinery of §7p records the new population without further work. **2025-02 is
the month the revival argument names** — a second winter reading beside 2025-01, which is
the fold both MAYBEs die on — and it is one of the three whose summaries are already on
disk. The mechanism is not that a fourth fold makes january's own interval narrower: it is
that january's head is then fitted on **three** months instead of two, so its correction is
less noisy, its increment larger and its interval tighter. That is the mechanism §7i
measured shrinking to work through, and it is an argument, not a measurement, until the
month exists.

### 7r. The lit-fold rule, one conflict with the dates, and the cell it revives

The desk's pick of 2026-09-09 (B now) came with a refinement of the every-fold clause:
*"every fold in which the cell's airport instrument existed"* — a cell dark by instrument
in a fold month is priced on its **lit** folds, needing 500+ rows and an interval clear of
zero above in **each** lit fold, at least two lit folds, and december among them because a
winter ranking month is served. The rule is registered in `code/arms.py`
(`INSTRUMENT_ARRIVAL`, `lit_folds`, `cell_supported(..., lit_rule=True)`) and read once by
`arms.project`, `serve.cell_support` and `work/foldside.py`, so the file written and the
projection that gates it cannot disagree about which folds a cell owes.

**The conflict, raised and settled.** The desk's line dated the windows *"LEMD on from
2025-03-04 and LIRF 2025-05-24..08-28"*. The harvest supports the two arrivals to the day
and does **not** support the closing date as darkness: Rome falls from 93 % to the thirties
on 2025-08-28 and then climbs back — 18.4 % Sep, 38.2 Oct, **62.4 % in the december fold**,
66.0 % in the 2026-01 ranking month — so december at Rome reads HIGHER than EGLL's 57.0 %,
which the same line treats as lit, and Rome is dark only in the scored 2026-07 (0.1 %),
which is a ranking month and not a fold. The record kept both readings and priced them in
one run; the desk then replaced its own: *"LIT means the instrument has ARRIVED at the
airport; a decline afterwards is lit, darkness is only the time before arrival"*. So
darkness is an **arrival date**, and the only two in the training year are Madrid's
2025-03-04 and Rome's 2025-05-24. **The withdrawn clause costs nothing**: under it LIRF
would lose its december fold, and LIRF clears in neither reading, so
`served_cells_only_desk_window` is identical to `served_cells_only` on all three folds
(`analysis/arms-step2-lit.json`). It is kept as the sensitivity and nothing rests on it.

**What the rule does, cell by cell** (`analysis/arms-step2-lit.json`, arm `N`, the same
population, pins and matcher `6fdf3951…` as the rebuild; the fold table reproduces
`arms-step2-rebuild.json` line for line, because the hold is a post-pass over the same
draws and changes no fit):

| cell | lit folds | verdict | why |
|---|---|---|---|
| `EDDF/reached` | all three | **served** (was served) | clears on all three |
| `LTFM/unreached` | all three | **served** (was served) | clears on all three; LTFM reads 0.0 % receipt in every month, but its neighbour columns come off the FILE clock (`Nf_*`) and not off the feeder, so it has no arrival and no dark fold |
| `LEMD/reached` | july, december | **REVIVED** | july 9,366 rows [+731, +1,622], december 810 rows [+501, +2,252]; january dark |
| `LEMD/unreached` | july, december | held | december is under the 500-row floor — Madrid at 96.5 % receipt in december leaves almost no unreached rows — and july crosses zero [−82, +627]. Its january cell has 16,509 rows and reads [+28, +293], but january is exactly the fold where the instrument was off, so those rows are the artefact the rule exists to refuse |
| `LEBL/reached` | all three | held | december crosses at [−22, +3,800]; the desk's "if each fold clears on its own" is not met |
| `EDDM/unreached` | all three | held | december and january are both under the floor; only july prices it (9,820 rows [+36, +670]). Munich's fortnight-long July 2025 outage is what makes that cell exist at all, and an outage is not an arrival, so the two winter folds still count against it |
| `LIRF/reached`, `LIRF/unreached` | july, december | held | both cross zero on both lit folds in either reading of the window |

So the desk's expectation list — LEMD/reached, LEMD/unreached, LEBL/reached,
EDDM/unreached — comes back as **one cell**, and the three that fail do so on the floor and
the interval rather than on the rule. The served set is `EDDF/reached`, `LTFM/unreached`
and `LEMD/reached`.

**The cross-fitted hold, and what it can and cannot say about a two-lit-fold cell.** §7p's
rule stands: each fold's served set is chosen on the OTHER folds and read on the held-out
one. A cell with only two lit folds has just ONE lit fold left in any selection set, so the
cross-fit there asks that single fold to clear (`min_lit=1`) and the record names the cell
in `one_lit_fold_selection`. It is still out of sample for the fold being priced, and it is
a weaker selection than a three-lit cell's. Every fold's cross-fitted hold is **at or above**
its in-sample twin: july +282.73 s² [+202.38, +370.74] on both, january +99.04 [+41.82,
+167.30] on both, december **+260.28** [+140.29, +418.35] cross-fitted against +244.77
[+124.52, +405.77] in sample — december's selection set (july + january) additionally picks
`LEBL/reached`, which december itself would not.

### 7s. The increment over v16's bytes

v16 already carries this lane's v14 update on `EDDF/reached` and `LTFM/unreached`, so the
thing the gate has to price is not the served set's level but what the revival **adds**.
`work/increment16.py` reads the contract one level up — `base16 = p12 + d` on the carried
cells, `inc = d` on the added cell, gain `= mean((y − base16)²) − mean((y − base16 − inc)²)`
— with `d` cross-fitted by fold and the interval resampling the fold's days at the record's
own SEED and BOOT (`analysis/increment-v16.json`):

| fold | rows added | increment, fold total | 95 % | per added row | RMS move |
|---|---:|---:|---|---:|---:|
| july_loo | 9,366 | **+63.15 s²** | [+41.18, +87.14] | +1,285.9 s² | 27.20 s |
| december | 810 | **+6.34 s²** | [+3.17, +9.93] | +1,297.7 s² | 27.85 s |
| january | **0** | +0.00 | — | — | — |

Fold-side currency of the increment alone: clean board 256.2412 → **256.1726**, +0.0686 s;
all-row 308.4122 → 308.3552, +0.0570 s.

**Neither lit fold crosses zero, and january is empty rather than crossing.** Madrid's
feeder did not exist in 2025-01, so the added cell has no january fold rows at all; an empty
lit fold is not a crossing and the desk has ruled it so. The projection puts the cell at
+29.49 s² [+18.53, +41.11] on july and +33.16 [+12.69, +57.08] on december, and at 0 on
january, where filling it with the fold's class-pooled gain instead of zero would add about
+14 s². (The projection counts the cell at 8,739 ranking rows, which is its weight times
344,841; the rebuilt census counts **8,728**, and that is the number the file moves.)

**The support, season by season, is better than the projection's extrapolation column makes
it look** (`analysis/lemd-support.json`). The projection weights every fold's per-row gain
by the cell's WHOLE ranking weight, so it reads 0.9× on july and 10.8× on december. But the
8,728 ranking rows the file moves are **7,837 in 2026-07 and 891 in 2026-01** — Madrid is at 60.7 % receipt
in the scored july and 96.3 % in the scored january, and a high receipt leaves this lane's
no-clock class almost empty — so **90 % of what this file moves is summer**, priced by
july's 9,366 fold rows (0.84× onto the rows it serves), and the winter tenth is priced by
december's 810 (1.11×). Both fold cells are spread over **all 31 days** of their month
(july 160–351 rows a day, median 305; december 12–53, median 24), so the day-clustered
intervals rest on 31 clusters each and not on a handful of days. An earlier draft of this
paragraph called the winter exposure "half"; it is a tenth.

**2025-02 cannot repair that, and it is no longer coming.** The fourth fold month recorded rule
picked is dark at Madrid — the arrival is 2025-03-04 — so it would add no lit fold to the
one cell this order revives (`analysis/fold-month-2025-02-spec.md` §8, §9). On reading the
spec the desk **deferred recorded rule** (2026-09-09): the seven member refits the spec found under
the three vectors price the month at about ten Kaggle kernels and one to two worker-days
for at most +0.1 to +0.2 s at face, and the recorded revival condition is a candidate worth
**0.5 s or more at face in either surface lane that is a MAYBE on january alone**. So
december's 810 rows are the winter evidence this cell will have for as long as the lane
runs on three folds.

### 7t. The serve on v16's bytes, and the guard that had to be made fail-closed

`submissions/candidates/exp-047-queue-lit.parquet`, SHA-256
**`4de323fc4fd9d4865b201b0dd0da1ddcbf7de9e11f1d9bc8af07d293585001ea`**, composed on v16
(`6f8368da…`) with the class vector re-emitted on matcher `6fdf3951…`
(`class-manifest-v16.json`; `2026-01-v16.parquet` `3e4b9840…`, `2026-07-v16.parquet`
`6575b2ef…`). The dry run and the serve produce the same bytes: `serve-dry-lit.json` and
`serve-lit.json` differ in `candidate` and `dry_run` and in nothing else.

**What it moves.** 72,943 rows of 344,841, all of them this lane's own class and all in a
served cell: `EDDF/reached` 16,638 (5,680 in 2026-01, 10,958 in 2026-07), `LTFM/unreached`
47,577, `LEMD/reached` 8,728 (891 and 7,837). RMS move 11.36 s over the moved rows, 32.45 s
at Madrid and 0.0067 s at Istanbul; largest single move 197.89 s; 0 rows clipped at the 60 s
floor; the 388 noac rows bit-identical; **0 held rows moved**. On an owned row the value
written is `v12 + d`, and `max |base − this lane's own accepted candidate| = 0` on all
72,943, so what the file replaces is this lane's own v14 update and nothing of the other
lane's.

**The class boundary moved with the rebuild, and the served cells moved with it.** The
class is 153,832 rows where it was 154,358: the held-days repair gave 506 EDDF 2026-01 rows
a clock, which leaves this lane. Inside the class the same repair moved rows the other way
across the reach line — `EDDF/reached` is 16,638 where it was 16,285 and `EDDF/unreached`
4,477 where it was 5,337 — because a repaired neighbour observation is what makes a row
reached. Both are consequences of the matcher rebuild, not of this order.

**The disjointness guard was weaker than it read, and is now fail-closed.** The v16 class
vector separates three sets: `occupied_by_exp_046` (**189,009**, what that lane OWNS),
`written_by_exp_046` (**188,483**, whose bytes carry its value) and `changed_by_v16`
(188,482, what its candidate moves off its own base). The 526-row gap is rows it owns and
has not written yet. The desk's instruction is to assert against the strictest, and the
first run of the guard **reported the ownership column's name while testing the moved
column**, because `main` builds its frame from an explicit column list that did not carry
it — a test that silently weakens itself. `check_disjoint` now raises if the manifest names
an ownership column the frame does not carry. On the corrected run: **overlap with owned 0,
with written 0, with moved 0**; 82,889 rows held at the base by both lanes. The bytes never
changed — the candidate hashes to `4de323fc…` before and after the fix — but the assertion
behind them did.

### 7u. What round 5 asked for: the tail, the 353 rows, december re-read, and the backlog

FALLBACK round 5 (`review/round5-FALLBACK.txt`) returned **OK with no blocking findings**
on snapshot 0389c27, verifying the bytes, the lit rule, the increment arithmetic and the
fold-month spec independently. Its seven non-blocking items are answered here.

**The revived cell's gain is heavy-tailed, and that is the shape to expect**
(`analysis/lemd-concentration.json`, item 3). Read this beside the projection's 10.8×
extrapolation column, not instead of it:

| `LEMD/reached` | july (9,366 rows, 31 days) | december (810 rows, 31 days) |
|---|---:|---:|
| gain per row | +1,285.9 s² | +1,297.7 s² |
| top 1 % of rows: share of the net total | 66 % (94 rows) | 58 % (8 rows) |
| top 5 %: share of the net total | 141 % | 126 % |
| gain per row without the top 1 % | +447.6 | +545.2 |
| 10 % trimmed mean | +376.1 | +499.8 |
| leave-one-day-out range | +1,144 to +1,372 | +1,103 to +1,379 |
| days whose own mean is negative | 4 of 31 | 8 of 31 |
| day-clustered 95 % | [+860, +1,773] | [+632, +2,050] |
| the same on an iid ROW bootstrap | [+1,025, +1,553] | [+588, +2,069] |

A top-5 % share above 100 % is not an error: the remaining 95 % of rows are net negative,
so the top of the distribution carries more than the whole. What this says is that the
interval is a believable **sampling** statement — the day and the row bootstrap agree, so
the clustering is not what carries it, and the leave-one-day-out range is narrow, so no
single day is either — while the gain itself rests on a few large-residual rows in **both**
seasons alike. That is the ordinary shape of an MSE gain on taxi times, and it is the same
shape at both ends of the year, which is the relevant fact for serving a winter month off
a december fold.

**353 EDDF rows are additions the increment did not price** (`analysis/new-rows-not-in-the-increment.json`,
item 4). `work/increment16.py` treats the whole current `EDDF/reached` cell as carried by
v16, but the matcher rebuild moved rows across the reach line, so of the 72,943 rows the
file writes, **63,862 carry this lane's v14 update and 9,081 sit at v12 in v16** — 8,728 of
them the new Madrid cell and **353 of them EDDF rows that became reached at the rebuild**.
Their bound, if their correction carried no signal at all, is
`sqrt(board² + Σd²/344,841) − board` = **0.0013 s** (Σd² = 224,377 s² over 353 rows). The
same bound on the whole Madrid cell is **0.054 s**, which is the honest ceiling on the
empty january fold's exposure.

**December, re-read on the file that is written** (`analysis/december-reconcile-lit.json`).
The desk's §7b rule reads the raw december gain on the rows the file moves. On the lit
file it is **+401.95 s² [+180.91, +645.28]** on the 24,979 served rows and **+60.6 s²
[+30.6, +97.23]** all-row — both clear **above** zero, where the pre-hold whole-class read
is −4.37 [−127.41, +111.94] and where v14's raw december was a small loss. T8 and T9 now
render from `serve-lit.json` and `december-reconcile-lit.json`, through one constant each
in `code/tables.py`, so a re-serve moves both tables at once and cannot leave them pricing
the previous file (item 1 — the same drift round 4 repaired for v14).

**The zero-receipt clause, in the desk's final wording** (item 2). The desk's line of
2026-09-09 reads: darkness is the time before arrival *"or a month in which the cell's arm
reads a surface observation that is 0 % by instrument"*. The lit rule reaches a cell only
through the features its **arm reads**. `LTFM/unreached` is served on the file-clock family
alone — `Nf_*`, with no observed neighbour anywhere in it — so it has no surface instrument
that can be dark; its instrument is the ranking file, present in every month. It therefore
has three lit folds and stays served, though LTFM's feeder reads 0.0 % in all thirteen
months. `code/arms.py` and this section now carry the same sentence. LFPG's feeder also
switches on inside the year (0 % to 2025-06-21, 1.6 to 7.9 % after 2025-06-22) and is
**not** registered in `INSTRUMENT_ARRIVAL`; no served cell depends on it, because every
LFPG cell crosses zero on every fold, lit or not. Registering it is a change to the rule
and so waits for the next re-price.

**Three guards had the shape the disjointness guard had** (item 5), and all three now stop
instead of widening: `served_mask` raised nothing when `obs_has_clock` was absent and read
every row as no-clock, `arms.reached` read every row as unreached when the count column was
absent, and `rank_frame` sniffed the ownership column off the vector rather than taking it
from the manifest — so a vector carrying neither ownership column would have fallen back to
the moved set in silence. The ownership column is now read from the manifest's `columns`
block and a basis at or above `-v16` that declares none is refused. `base.check_prefix_agrees`
took the same treatment: it read the UNTAGGED class manifest while the serve read `-v16`, so
it checked a file the candidate does not use. None of the four changes moves a byte.

**`LEBL/reached` is the backlog's next cell** (item 6). It clears july [+949, +3,697] and
january [+516, +2,157] on their own and crosses on december at **[−22, +3,800]** — the
narrowest miss of the nineteen. Its revival condition: **december's own interval clear above
zero**, which a fourth winter fold month would decide and which recorded rule's deferral has put out
of reach for now (§7s). The cross-fitted hold already selects it on two folds of three,
which is why december's cross-fitted read (+260.28) sits above its in-sample twin (+244.77).

### 7v. BOARD READING 16 — v17 at 246.6571, and the lane's two readings at or below face

**The file is up.** `4de323fc…` uploaded at 03:33:59 UTC on 2026-09-09 and scored
**246.6571** at 03:34:25: **−0.0482 s** on v16's 246.7053, rank 1, 14.27 s ahead
(BOARD READING 16, `28fd5a9`). score-feedback rule: nothing revised on this reading.

**The transfer, derived here rather than quoted** (`analysis/board-reading-16.json`).
January's increment is identically zero because Madrid was dark in 2025-01, so the whole
board increment comes through july's share 0.5571; mapping july's day-clustered interval
[+41.18, +87.14] s² onto the clean board at the ranking shares, with january's fold RMSE
held at its v16 value, gives a predicted board gain of **+0.0687 s [+0.0448, +0.0947]**.
The board moved **0.0482 s**, a transfer of **0.70×** at the point and **inside the fold
interval** — the interval's own ends correspond to 0.51× and 1.08×.

**So this reading does not by itself say the increment was over-priced.** A move inside a
95 % interval is what an honest interval is for. What it does add is a second point in a
pattern: this lane's two board readings are **1.03× on v14's all-row gain (0.86× on the
clean one)** and **0.70× here**, both at or below face, where the desk's tally puts the
with-clock lane's *unselected* updates at ×1.5 on the same level correction. Two readings
are not a rate, and the honest statement is that a single reading cannot separate two
explanations that predict the same thing:

* **post-selection shrinkage.** The cell this increment serves was chosen because it
  cleared on the same folds that then priced it. §7r's hold was cross-fitted and came back
  at or above its in-sample twin — but that tests the *hold*, and for `LEMD/reached` each
  lit fold selects the cell on its own, so the cross-fit had no selection to pay for and
  no premium to reveal. The cell choice itself is still in-sample.
* **an ordinary draw**, given an interval whose ends are 0.51× and 1.08×.

The thing that would separate them is the next increment in this lane whose FORM was not
chosen on the folds. There is none queued: §7q's remaining steps are MAYBEs at or under
0.05 s at face, and the fourth fold month that would move them is deferred (§7s). So the
lane records the pattern and does not act on it.

**Where that leaves the lane.** Three cells of nineteen are served, on the board, at
246.6571. The two form MAYBEs (`N_reach`, `N_reach_half`) and the eleven still-held cells
all sit at or under 0.05 s at face and none has the evidence to clear its own fold gate —
so **no serve on current evidence**, and the next work is not a fit: surfacefit's reach
handshake moves the matcher again, and this lane re-emits the interface, rewrites its pin
files and re-prices once, exactly as at §7p.

### 7w. The reach spec, read from this side — the pass moves the reach GRAIN, not a column

`exp-046-surfacefit`'s `analysis/reach-spec.md` (`bbdb39e`) adds a THIRD pass to
`block.build`: a departure segment unclaimed by both existing passes, no broadcast form of
it a ranking DEPARTURE key at the airport, some form a ranking ARRIVAL callsign into it,
exactly one unclaimed ranking departure within ±60 s. The desk measured what it moves —
**9,875 rows leave this lane's class for surfacefit's**, 4,147 in 2026-01 and 5,728 in
2026-07 — and asked for this lane's line before `block.py` moves. The answer is **agree**;
what follows is the check behind it (`analysis/reach-spec-exposure.json`, `0182640`).

**The interface holds, because this lane reads none of the new field.** A grep of
`code/*.py` for `match_mode` returns nothing: this block reads `MVT_ID_mvt`, `airport`,
`day`, `stand_lat`, `stand_lon`, `obs_has_clock`, `obs_offblock`, `obs_matched`,
`reach_class` and `ev_key`, and nothing else. A mode of 3 is therefore **ignored, not
silently read as a wide match**, which is the failure the spec's §7.1 asks about.
`reach_class` is passed through into counts and never branched on, so a new value is safe.

**What the pass actually moves here is not a column.** `reached` is `No_n_n_cell > 0` —
the count of heard aircraft in the airport × 10-minute cell — which is *exactly the
quantity a third pass increases*. So the pass moves the **grain the cells are cut on**, not
only the class boundary: a placed segment makes every row in its 10-minute cell a candidate
to flip from `unreached` to `reached`. Of this lane's ranking rows at the airport-months
the pass touches, **58,660 are unreached today**, and drawing each placed segment's cell in
proportion to that airport-month's departure traffic gives an **upper** estimate of
**~19,704 flips**:

| airport-month | placed | this lane's unreached rows | flips (upper) |
|---|---:|---:|---:|
| LFPG 2026-07 | 1,516 | 18,780 | 9,583 |
| LFPG 2026-01 | 1,123 | 15,328 | 6,161 |
| EDDF 2026-07 | 3,049 | 2,384 | 2,272 |
| EDDF 2026-01 | 1,218 | 2,093 | 927 |
| LIRF 2026-01 | 603 | 2,161 | 386 |
| LEMD 2026-07 | 937 | 1,314 | 286 |
| EGLL 2026-01 | 1,044 | 203 | 54 |
| all others | 284 | 16,397 | 35 |

It is an upper estimate: one cell can take several placed segments, and a flip also needs
the placed clock to survive into the neighbourhood aggregate. LIRF 2026-07 shows the shape
— 15,799 unreached rows but only 6 placed segments, so 31 flips.

**Three consequences for the served set, all of them settled by the ordered re-price.**

1. **~15,744 of the flips are at LFPG**, whose two cells are held by this lane today, so
   they move no served byte — but they do move a held cell across the floor and its fold
   readings, which is what the re-price re-reads.
2. **~3,485 flips land in cells this lane SERVES** (~3,199 EDDF, ~286 LEMD). At EDDF 2026-07
   the estimate is 2,272 of 2,384 unreached rows, so **`EDDF/unreached` largely dissolves
   into the served `EDDF/reached`** — rows arriving in a served cell from a held one,
   never priced there. The re-price prices them; nothing may be carried.
3. **Up to 5,243 rows that carry this lane's v17 correction leave its class** (EDDF 4,267 +
   LEMD 976 placed at the two served airports). That is not a double write — the class
   vector is one boundary — but a row that leaves this class and is then *held* rather than
   written by surfacefit would keep this lane's value while belonging to no lane. The
   composition after the handshake has to name who writes them. LFPG's 2,639 are safe:
   every LFPG cell is held here, so this lane moves none of them.

**One asymmetry worth stating for the spec's own gates.** A false match costs the two lanes
differently. Surfacefit reads the row's OWN clock, so a wrong placement damages that one
row. This lane reads the NEIGHBOURS' clocks, so a wrong placement enters an aggregate that
every row in the cell reads — about 5 rows per 10-minute cell at these airports. Gate (c),
the one-day time-shift placebo, is therefore load-bearing for this lane in a way it is not
for surfacefit, and gate (b)'s density-scaled bound is better read as a bound on **polluted
cells** than on damaged rows.

None of this is an objection. The line to the desk was **agree**, and the desk relayed it
to surfacefit with three riders: the reach grain, the 5,243 rows to be named *occupied and
written* at its next serve, and a placebo per airport.

**The handshake is armed and waiting on the `MATCHER_SHA` line.** `work/handshake.sh <sha>`
is the whole of the desk's order in one ordered run — re-emit the five blocks, move the
pins forward on the record, re-read both ranking censuses, **one** re-price of `N` and the
two MAYBEs under the lit rule with the cell hold re-read, december on the new record, and
the before/after table. **No serve**; the desk asks for one or it does not happen. Its first
step is a guard that refuses unless `code/block.py`'s `MATCHER_SHA` *and* the matcher on
disk both equal the sha on the command line, so a stale constant cannot put pre-handshake
rows into a post-handshake record. The guard was read in both directions on 2026-09-09: it
refused a wrong sha, and it refused the sibling's `block.py` while that file was mid-edit
in the worktree at `fb727516…` (the third pass, `MODES["stale_inbound"] = 3`,
`STALE_WINDOW = 60.0`) against the declared `6fdf3951…`.

**The handshake did not run: the third pass failed its own placebo** (desk, 2026-09-09).
The one-day time shift placed **8,084** segments against **7,865** real ones, and the
`AIRCRAFT_TYPE_mvt` agreement read at chance where the volume is — so the pass does not
ship, `block.py` stays at `6fdf3951…` (the worktree is back to it), and no `MATCHER_SHA`
line comes. This is the gate this lane said would matter most here (gate (c) above), and it
is the one that stopped it. The handshake stays armed exactly as it is: surfacefit is
trying a turnaround-chain identity (stand, operator hash, type) as a label-free read, and
if that passes its placebo the spec and this run happen again; if it does not, the reach
path closes provisional and nothing in this lane moves. Nothing here is refitted, and the
frozen BEFORE column below stays the BEFORE column for whichever attempt arrives.

**The placebo the next attempt must pass is now a per-CELL read as well**, at this lane's
request and relayed by the desk. A turnaround chain (stand, operator hash, type) is a
stronger identity for one row but a *weaker control here*: a chain that repeats at the same
stand each day shifts to the same stand each day, so a per-row one-day placebo can pass
while the false placements stay concentrated in the same 10-minute cells this arm
aggregates over — which is the §7w asymmetry again, that a false placement damages one row
for the lane reading the row's own clock and a whole cell aggregate for this one. The gate
is therefore both legs: per row, and per airport × 10-minute cell (real against the one-day
and stand-shuffle placebos, the share of cells where the placebo reads at or above real,
and the share of placed rows sitting in them). A pass needs both.

**The chain PASSED both legs** (desk, 2026-09-09): 3,101 gated rows, the day placebo 32
(1.03 %) and the stand shuffle 34 (1.10 %) per row; per cell, 12 of 21 contested cells of
2,713 hold ONE row each and 1 to 2 of 781 time-of-day cells. So a chain spec is coming with
the same four questions, and the fourth of them — the spillover — is now an instrument
rather than a one-off: `work/spillover.py` takes the sibling's own placed table and
returns the flip estimate per airport-month and the part of it that lands in a cell this
lane serves. Run on the reach spec's placed counts it reproduces that record exactly
(~19,704 flips, ~3,485 into served cells), and it holds the `datetime64[us]` trap in one
place — `astype("int64") // 10**9` collapses a month into ten cells and was the first
estimate's error. The chain places 3,101 rows against the reach pass's 9,875 and LFPG's 441
stay this lane's, so the exposure should read smaller on both counts; the number itself
waits for the spec's table.

The BEFORE column is frozen now rather than reconstructed later
(`analysis/handshake-before.json`, one reader for both sides in
`work/handshake_report.py`): whole class **+293.58 [+190, +406]** july_loo, **+224.71
[+32, +453]** december, **+156.54 [+80, +237]** january; served cells **+282.73 /
+244.77 / +99.04** on `EDDF/reached`, `LEMD/reached`, `LTFM/unreached`; 72,943 served rows
of 153,832 in class; and the fold reach split (26,359 / 21,883 / 15,478 reached), which is
the line the third pass is expected to move. `work/block_diff.py` reads the kept blocks
against the re-emitted ones and reports the **reach flip** — `No_n_n_cell > 0` before
against after — beside the per-column moves; run against itself it returns an exact zero
and a `reach_class` census equal to T6's, which is what makes a non-zero reading readable.

### 7x. The chain spec, answered — the four questions, and GO

`exp-046-surfacefit/analysis/chain-spec.md` (`f870f40`) replaces the refuted reach pass with
a turnaround chain: the inbound callsign names the ARRIVAL, and the departure is the airframe
turning round at that arrival's stand (`STAND_mvt`, hashed operator, `AIRCRAFT_TYPE_mvt`,
take-off after the arrival's in-block and within 24 h), with `|take-off − obs_wheels_off| ≤
60 s` as the last clause of the key rather than the only one. It places **3,101** rows, a
third of the refuted pass's 9,875, and it passed both bars — per row 1.03 % (day) and 1.10 %
(stand shuffle), and per cell 12 of 2,713 contested cells on the day placebo and 21 on the
stand shuffle, **every contested cell holding exactly one real row**. The per-cell leg is the
one this lane asked for (§7w); the time-of-day binning answers the daily-repeat concern
directly at 1 to 2 contested cells of 781. Answers, all measured
(`analysis/chain-spec-answer.json`, `analysis/chain-spec-exposure.json`):

**1. `match_mode` = 3 is ignored by construction.** `match_mode`, `MODES`, `clock_ok`,
`turnaround_anchored`, `matched_on_modal` and `obs_ambiguous` occur **zero** times in
`code/` and `work/`. This lane reads `MVT_ID_mvt`, `airport`, `day`, `stand_lat`,
`stand_lon`, `obs_has_clock`, `obs_offblock`, `obs_matched`, `obs_wheels_off`, `obs_taxi`,
`obs_icao`, `reach_class` and `ev_key`. A mode of 3 is never read as a wide match because it
is never read.

**2. The 2,660 leaving rows are NOT all rows this lane leaves alone.** At this lane's own
measured reached share per airport-month — EDDF 0.731 january and 0.821 july, LEMD 0.929 and
0.856 — about **1,550 of the 2,660 sit in `EDDF/reached` or `LEMD/reached`, which this lane
moves today**. That is a **handover, not a contested byte**: the class vector is one
boundary, §6.3 of the spec already carries this lane's rider (every row surfacefit occupies
is written at its next serve, a row arriving with the v17 correction included), and the check
is mechanical — `serve.py`'s disjointness guard has been fail-closed since §7t and asserts
zero overlap with `occupied_by_exp_046` on the v18 vector. **LFPG's 441** are kept in the
sense that matters: surfacefit does not occupy them and this lane holds every LFPG cell, so
the byte is the base value either way. The letter of §4.1 is loose — they gain
`obs_has_clock = 1`, so they leave this lane's class *by instrument* and are not written by
it; nothing on disk changes.

**3. The census reproduces §4.** This lane's class rows against the spec's unclaimed
departures: **LIRF exact in both months** (5,667 and 15,551), and everywhere else this lane's
class is 0 to 494 rows larger, **+2,489 over the eighteen airport-months**. The cause is a
definition and it runs the safe way: the spec's `unclaimed` excludes a departure with
`obs_taxi` notna, this lane's class excludes `obs_has_clock`, so a row matched with a taxi
but no usable off-block clock is in this lane's class and not in the spec's set — **the
placed set is a subset of this lane's class**.

**4. The spillover, re-measured at 3,101 and not scaled.**

| | refuted pass, 9,875 placed | chain, 3,101 placed | a scale would have said |
|---|---:|---:|---:|
| rows flipping unreached → reached (upper) | 19,704 | **4,129** | 6,188 |
| of them into cells this lane serves | 3,485 | **1,346** | — |
| kept rows whose neighbourhood value moves | 33,581 | **9,462** | — |
| LFPG's share of the flips | 80 % | **63 %** | — |

A scale is **33 % too high**, exactly as §7.3 of the spec expected: the placement is 31 % the
size but LFPG's share of it falls from 27 % to 14 %, so both terms move. `EDDF/unreached`
does **not** dissolve this time — 942 of july's 2,384 unreached rows, 40 %, against 95 %
under the refuted pass — and `LEMD/reached` takes 49. The wider leg matters more than the
flips: about **9,462 kept rows have a `No_*` value that moves** because a neighbour of theirs
is now placed, ~6,357 of them at EDDF and LEMD. All of it is inside what the ordered single
re-price re-reads, which is why it is a price and not a defect.

**Verdict: GO**, sent as one line on 2026-09-09. Nothing this lane reads breaks; the class
movement is a handover its own fail-closed guard checks; the grain movement is priced by the
re-price the desk already ordered; and the per-cell placebo tables are the evidence that what
moves is not biased where it lands.

**The count moved to 3,179 before the commit, and the GO holds.** The spec's own §8
agreement check — the two implementations must agree — caught a defect in `turnaround.py`:
strict-matched clockless departures (6,063 january, 5,472 july) competed as free candidates,
and two-way uniqueness then declined **78 real placements**. `block.build` under the spec
words places **3,179** (1,382 january, 1,797 july), gates clean at that count (sweep **0
changes on 341,662 rows × 34 columns**, day placebo 42 = 1.32 %, stand 29 = 0.91 %, contested
cells 15 and 20 of 2,776), LFPG keeps **449**, and the class movement is **2,730**. Every
answer above moves by less than 2 %:

| | 3,101 | **3,179** | allocation range |
|---|---:|---:|---:|
| flips unreached → reached (upper) | 4,129 | **4,213** | 4,208 – 4,272 |
| of them into cells this lane serves | 1,346 | **1,375** | ±2 |
| kept rows whose neighbourhood value moves | 9,462 | **9,664** | — |
| leaving rows this lane moves today | ~1,550 | **~1,590** | — |

The desk gave per-airport and per-month totals but not the per-airport-**month** table, so
each airport's delta was split by its §4 month shares; that reproduces **every** per-airport
total and both month totals exactly, and the range column prices the assumption by putting
each delta wholly in its lower- then its higher-coefficient month. The swing is LIRF, whose
cells are held, so the served figure moves by about two rows.

**One condition on the GO, and it is the per-cell bar's own wording.** The bar was not only
the contested *count* but that **every contested cell holds exactly ONE real row** — that is
what keeps a cell aggregate from being decided by a contested placement, and it is why 12 and
21 of 2,713 was a pass at 3,101. At 3,179 the counts are re-stated (15 and 20 of 2,776) and
the one-row property is not. If a contested cell now holds several real rows, that changes
this answer; the counts by themselves do not.

### 7y. The handshake, run — matcher `122c7d84`, and the AFTER column

The desk's single `MATCHER_SHA` line landed on 2026-09-09: **`122c7d84…`**
(`exp-046-surfacefit/code/block.py` at `d92aec3`, the turnaround pass). This lane re-emitted
the five interface blocks, re-issued the pins, re-priced **once** and read the AFTER column
against the BEFORE frozen at §7w. **Nothing was served.**

**The wiring rule, and what it cost.** `B46.build` runs the turnaround pass only when it is
handed the month's ARR rows; without them it is a **silent no-op** — no error, and 3,179
clocks simply missing. `code/block.py`'s `arrivals_for()` supplies them from **one function
per side**: a 2025 fold month through `exp-046-surfacefit`'s own `arms.month_arrivals`, and a
2026 ranking month — which has no movement file — as the ARR slice of the ranking file, the
same expression the sibling's serve takes. Two things only running it could find:

* `_sibling_arms()` has to **stash the plain names** `base`, `block`, `state`, `pin`,
  `population` across the load. Both lanes ship a `base` and a `block`; by the time the
  arrival side is wanted this lane's are in `sys.modules`, so a plain load of the sibling's
  `arms` binds it to **this** lane's `base` and dies on `V12_BOARD`.
* `FRAME_COLS` gains `AIRCRAFT_OPERATOR_flt` and `AIRCRAFT_TYPE_mvt`: the chain reads the
  candidate **departure's** stand, operator and type, not only the arrival's. Nothing in this
  lane reads either column; they pass through to the one matcher.

**The count is asserted, not assumed**, because the failure it guards is silent
(`check_mode3`, fail-closed). Every month agrees with `exp-046-surfacefit`'s reference:

| month | mode-3 rows | reference |
|---|---:|---|
| 2026-01 | **1,382** | 1,382 |
| 2026-07 | **1,797** | 1,797 |
| 2025-07 (july_loo) | **1,170** | the desk's fold set {1,170, 861, 731} |
| 2025-12 (december) | **861** | — the mapping is named here for the first time |
| 2025-01 (january) | **731** | — |

**What the re-emit moved** (`analysis/handshake-block-diff.json`, the kept blocks against the
new ones). Clocks gained are **exactly** the mode-3 counts and **none is lost**; every
`reach_class` transition is `unmatched → turnaround` and nothing is re-labelled. The reach
flip is now **measured** rather than estimated:

| month | clocks + | flips unreached → reached | flips the other way |
|---|---:|---:|---:|
| 2026-01 | 1,382 | **1,058** | 0 |
| 2026-07 | 1,797 | **1,459** | 0 |
| 2025-01 / 07 / 12 | 731 / 1,170 / 861 | 185 / 773 / 804 | 0 |

So the ranking months flip **2,517** rows, against the **4,213** upper estimate §7x priced
the spec at — the bound held with about 40 % slack, which is what an upper estimate should
do. `Nf_n_n_cell` **does not move on a single row**: the file-clock family is untouched, and
with it the arm that serves `LTFM/unreached`. `No_n_n_cell` moves on 9,268 rows in 2026-07
(median +1 neighbour, max +4), `route_km` and `stand_dist_km` on ~51,900 (median 0.02 km, the
stand centroids re-based by newly placed rows), and `receipt_rate` on 98,739 with the `*_sc`
family that divides by it.

**The pins.** `december` admitted **9 airport-days** — the 2025-11-30 window edge, delivered
at 10:55 local, before the re-price read it — and is now pinned at 297; `july_loo` 296 and
`january` 218 are unchanged. The pin manifest names the handshake record rather than the
rebuild's (`work/writepins.py` takes both through the environment now).

**The AFTER column** (arm `N`, one floor, lit rule, cell hold re-read;
`analysis/arms-step2-handshake.json`, `analysis/handshake-diff.json`):

| fold | rows | whole class, before → after | served cells, before → after | cross-fitted |
|---|---:|---|---|---|
| july_loo | 103,513 → 102,343 | +293.58 → **+304.54 [+198, +419]** | +282.73 → **+302.17 [+217, +397]** | **+302.17** |
| december | 64,837 → 63,976 | +224.71 → **+247.09 [+53, +496]** | +244.77 → **+261.87 [+134, +445]** | **+277.33** |
| january | 84,933 → 84,202 | +156.54 → **+154.54 [+73, +246]** | +99.04 → **+97.28 [+33, +172]** | **+97.28** |

Each fold's class shrank by **exactly** its mode-3 count (1,170 / 861 / 731) — the placed
rows left this lane's class and nothing else did. **Every fold is clear of zero on both legs,
the served set is unchanged** (`EDDF/reached`, `LEMD/reached`, `LTFM/unreached`), and the
cross-fitted hold is at or above its in-sample twin on every fold. Two folds moved up and
january moved down by 2.0 s², the smallest of the three. As board figures the whole class
reads 0.573 → **0.594** july, 0.439 → **0.482** december, 0.305 → **0.302** january, and the
served cells 0.552 → **0.590**, 0.478 → **0.511**, 0.193 → **0.190**.

**December's raw read**, the §7b rule's own test, stays strongly positive: **+393.08 s² per
row on 24,773 served rows** (was +401.95 on 24,979); the all-row read is +58.78 (was +60.60)
and the pre-hold all-row read turns positive, −4.37 → **+11.58**
(`analysis/december-reconcile-handshake.json`).

**One reading not to make from the census.** `analysis/census-ranking*.json` show own-clock
rising by 1,891 (2026-01) and 1,806 (2026-07), which is **not** the pass alone: the copies
kept at `analysis/pre-handshake/` were built on matcher `ba41935e…`, two matchers back, so
their delta spans the rebuild as well — 2026-01's extra 509 is the EDDF held-days repair the
desk measured as +506 at v16. The clean measurement of this handshake is the block diff's
1,382 and 1,797, taken between two blocks of the same population.

**No serve.** The desk asked for the re-price and the AFTER column, and that is all this run
did.

### 7z. The hold re-read on the re-pinned folds — and what is left above 0.1 s

The desk, after the handshake: take §7q's highest-face item that needs **no matcher change
and no serve**, state the reading, and run it under the ordinary gate on the re-pinned folds.

**The reading, in one line.** Of §7q's four options **A** and **D** are serves and are
excluded by the desk's own condition, **C** is deferred under recorded rule and needs three other
lanes' vectors, so **B — the held cells — is the only item that is a fit or a read on this
lane alone**, and B's work *is* this gate read; §7q's numbers predate the lit rule, the
rebuild and the turnaround pass, and one of its rows (`LEMD/reached`, +0.0485 s) has since
been **served** in v17, so the read is done on the handshake record rather than quoted
(`work/heldcells.py`, `analysis/heldcells-handshake.json`).

**The answer: nothing above the 0.1 s face at ×1.2 is left as a fit or a read.** Summing
**every** positive held cell gives **0.0906 s at ×1.2**, and the top three are:

| held cell | ranking rows | face ×1.2 | the fold it fails, and how |
|---|---:|---:|---|
| `LEBL/reached` | 3,222 | **+0.0371** | december alone: +1,654.6 s²/row, 95 % lower bound **−27.14** on 1,071 fold rows |
| `EHAM/reached` | 2,528 | **+0.0249** | december **measured negative**, −1,862.8 [−3,606.8, −623.6]; january crosses zero |
| `LFPG/unreached` | 32,822 | **+0.0095** | all three, july_loo negative — the arm-held cell of §7d, its face nominal |

For scale, the two serve-only options are +0.031 s (A) and +0.049 s (D) at the same ×1.2
level, so **no item in the table clears 0.1 s** by any route, and this lane has no fit worth
running on current evidence.

**`LEBL/reached` is the lane's closest MAYBE and it is stable.** It is supported on january
and july_loo and fails december by a hair, and that hair has now survived two independent
re-prices: the lower bound reads **−22.31** on the lit record and **−27.14** after the
handshake, on a point estimate above +1,650 both times. What would move it is **december
evidence only** — a fourth fold month is not december, so recorded rule's revival does not reach this
cell; what reaches it is more LEBL december airport-days from the harvest, or a desk rule
that prices a cell whose failing fold crosses zero from above by less than a per-row hair.
**This lane does not propose the second**: it would open or close a class of cells, which is
the desk's line to write, not a worker's.

**One difference registered, not a conflict.** The desk's fold keys (`december d35b7171`,
`july_loo 8179f744`, `january cb4d6289`) are `exp-046-surfacefit`'s **block cache keys**
(`analysis/fold-population.json`), not this lane's population keys (`2f87293c…`,
`a26316a1…`, `40b77efb…`). The **populations agree exactly** — 296 july_loo, 297 december,
218 january on both sides — and the keys differ **by design**: the sibling's cache key
includes the matcher's first 8 hex, and this lane's population key deliberately excludes the
matcher and records it beside the key (§7p), which is the rule that stopped the sibling's
cache trap from biting here.

### 7aa. The RE-SERVE increment over v17's bytes — measured, and it is a thousandth of a second

The desk, after the handshake: arm `N` moved on its served legs, so price what a **re-serve
of the same three cells** would add **over the bytes v17 already carries**, cross-fitted, on
the three re-pinned folds, under the ordinary gate. It decides whether a queue rider joins
`exp-046-surfacefit`'s next serve on v18 in one upload (`work/reserve_increment.py`,
`analysis/reserve-increment-v17.json`). **Nothing was served.**

**How `d_old` was obtained, because it could not be re-derived.** The matcher has moved, so
the correction v17 carries cannot be rebuilt from the current one. It is built instead from
the blocks this handshake **kept** — `artifacts/exp-047/block-prev/<month>.parquet` at
`6fdf3951…`, the blocks v17 was priced and served on — while `d_new` comes from the blocks
the handshake wrote. Both are cross-fitted by fold with the same arm, learner and seed, so
the difference between them is the matcher move and nothing else. The reading is one level
up from the arm: `base = p12 + d_old` where v17 serves, `inc = d_new − d_old` where the row
is mine and served now, `0` where it is held now, and rows the pass took out of this lane's
class are excluded from `inc` and counted — their bytes do move at v18, but they move
because surfacefit writes them (its §6.3), which is its increment and not this one.

| fold | rows the re-serve rewrites | rows that left this lane (of them served by v17) | increment s²/row | 95 % |
|---|---:|---:|---:|---|
| july_loo | 38,608 | 1,170 (618) | **+2.35** | [−0.65, +5.06] |
| december | 24,773 | 861 (208) | **+0.85** | [−0.64, +2.54] |
| january | 24,527 | 731 (438) | **−1.55** | [−4.59, +1.24] |
| **pooled** (SHARES, december beside) | — | — | **+0.62** | **[−1.49, +2.56]** |

**The face: +0.0010 s all-row, +0.0013 s clean; ×1.2 → +0.0012 s, ×1.5 → +0.0015 s.**
The board reads 308.3599 → 308.3589 on the folds' all-row RMSE and 256.1783 → 256.1770 clean.

**The gate, read honestly.** It passes only the *letter*: the pooled point estimate is
positive and no fold's interval is clear **below** zero. But the **pooled interval crosses
zero**, all three fold intervals cross zero, and **january declines** (−1.55 s²/row) — the
same fold both §7q MAYBEs die on. The arm's served legs moved (+282.73 → +302.17 july,
+244.77 → +261.87 december, +99.04 → +97.28 january), but those are **levels against v12**;
`d_new` and `d_old` are the same form on nearly the same block, so what is left after
differencing them is 11.6 / 5.7 / −9.7 s² per rewritten row against a level of about 3,000.

**What it would cost to take.** A rider rewrites **71,716 ranking rows** (v17 wrote 72,943)
with an RMS move of 2.4 / 1.2 / 2.5 s and a maximum of 86 s, for a thousandth of a second —
**25× smaller than the smallest MAYBE in §7q** and two orders below the 0.1 s line.

**And nothing is stale if it does not ride.** The rows that left this lane are written by
surfacefit at v18 under its own §6.3 rule; the reach flips that entered this lane's served
cells number **16 / 2 / 11** on the fold side, so the corrections v17 wrote stay correct for
the rows this lane still owns. **This lane's reading is: no rider** — the re-serve is inside
the noise of its own interval, and the upload should carry surfacefit's write alone. The
call is the desk's.

### 7ab. Three design changes for the no-observation rows — a design read, not a fit

The desk, 2026-09-09: the §7q table is exhausted at this design, so name the three
strongest changes for the rows with **no observation at all** — about 125k of this lane's
class, `LTFM`'s 47k chief among them — that are **not** in §7q, each with its mechanism, the
cheapest test that could kill it, its size at ×1.2 and its failure mode. Sizes below are
**measured, not guessed** (`work/design_reads.py`, `analysis/design-reads.json`), and they
are **upper bounds**: each is ρ² · var(r12) on this lane's class, in sample, with no
cross-fitting and no shrinkage, where `r12 = y_true − p12` is the residual **after** every
feature the base head reads. A correlation with `r12` is by construction information the
head does not have; a ridge fitted out of fold would recover less.

**What the base head already reads, so the desk does not re-propose it**
(`exp-018-fresh/code/features.py`, 69 columns, plus `dataset.py`'s nine target encodings):

* **the arrival side**: `arr_taxiin_last10` / `last30` / `_ratio` (the rolling mean of
  arrivals' *taxi-in* at the airport, and its ratio to the airport median),
  `day_arr_taxiin`, `arr_delay_last20` (landing minus filed arrival), and the **stand
  chain** — `stand_prev_arr_inblock_dt` and `stand_next_arr_inblock_dt`, the previous and
  next arrival in-block **at the row's own stand**, which the file itself calls a "hard-ish
  upper bound on the taxi time", plus `stand_bound_minus_ms`, `_over_ms`, `_minus_r`;
* **the stand→runway route as a learned table**: `te_y_standrwy` and `te_off_standrwy`, and
  `te_*_standpfxrwy`, smoothed leave-one-month-out target encodings on (stand, runway) —
  so a "learned taxi-distance matrix" is **already there**, beside this lane's geometric
  `route_km` / `stand_dist_km`;
* **the row's own plan slips**: `aobt_eobt`, `aobt_sched`, `eobt_lobt`, `sched_lobt`,
  `m_lobt`, `m_iobt`, `m_eobt`, `ms_sched`, `plan_flight_time`, `act_flight_time`;
* **the file-clock neighbourhood**: `r_prev10_apt`, `r_prev30_apt`, `r_prev10_rwy`,
  `r_prev5_stand`, `r_prev30_ratio`, `r_rwy_median`, `apt_queue`, `rwy_queue`,
  `apt_pushed_10m/30m`, `rwy_pushed_30m`, the runway and airport gaps, the scheduled demand
  (`apt_sched_1h_around/before`, `demand_over_served`) and `day_r_mean` / `day_n_dep`;
* **calendar**: `hour`, `dow`, `dom`, `doy`, `sin_h`, `cos_h`.

Everything below is outside that list and outside arm `N`'s two families.

---

**1. The CELL's regulation signature — `AOBT − EOBT` concentrated in a ten-minute cell.
Measured upper bound +0.0885 s at ×1.2.**

*Mechanism.* The head reads the **row's own** `aobt_eobt`; nobody reads the **cell's**. An
ATFM regulation does not slip one flight, it slips a bank: the share of departures in the
same airport × 10-minute cell whose actual off-block is more than 15 minutes past their
estimated one is what a regulation looks like **from the file alone**, and a regulated bank
holds aircraft at the holding point, which is exactly where the taxi-out tail comes from. It
needs no observation, so it reaches `LTFM` and every other dark cell.

*The read already done.* Day-adjusted (so it cannot be the `day_r_mean` disruption the head
already has), the cell's slipped share correlates with `r12` at **+0.0401** july_loo,
**+0.0033** december and **+0.0350** january, and the top-decile minus bottom-decile gap in
the residual is **+27.9 s** july and **+18.1 s** january.

*Cheapest test that could kill it.* Re-read the same correlation **within operator**: at a
one-carrier airport the slipped share is a carrier fingerprint, and the head already encodes
the carrier (`te_*_oper`). If the within-carrier correlation collapses, it is the carrier,
not the surface. That is a read of minutes; only if it survives does it earn one ridge on
the fold side.

*Failure mode.* `AOBT − EOBT` is inflated by a late inbound as much as by a regulation, so
the feature may re-measure turnaround delay — which the stand chain and `arr_delay_last20`
already carry — and december's +0.0033 is what that looks like when the winter bank is
regular.

---

**2. The NETWORK state — the same hour at the other nine airports. Measured upper bound
+0.0683 s at ×1.2.**

*Mechanism.* Every feature in both lanes is **within one airport**. A European shock is not:
an ATFM regulation, a frontal system or a strike hits several of the ten at once. The
leave-one-airport-out mean residual of the same UTC hour is information no per-airport
feature can hold, and it is the **only one of the three that gives `LTFM` a new instrument**
— it needs no local observation and no local history.

*The read already done.* ρ(`r12`, network hour mean) = **+0.0223** july_loo, **+0.0084**
december, **+0.0434** january.

*Cheapest test that could kill it.* The ordinary cross-fit: build the term on two folds and
read it on the third. A network effect is seasonal, and this lane's folds are whole months,
so if it does not survive a month-out fit it is a month effect wearing a network's clothes —
and that test is one ridge, no new data.

*Failure mode.* Leakage of the wrong kind: the network mean at hour *h* is built from other
rows' residuals, so it must exclude the row's own airport (it does) **and** be computed on
the fold side without the fold's own labels. If it is ever built on the served months it
becomes a label read, and the gate would not catch it because the gain would be real.

---

**3. The SECOND MOMENT of the cell — the spread of the file-clock taxi, after its mean.
Measured upper bound +0.0192 s at ×1.2.**

*Mechanism.* The head reads rolling **means** and arm `N` reads cell **means and counts**;
nobody reads a spread. A queue that is saturated is not only slower on average, it is more
variable, and the variance rises before the mean does — so the spread is a leading indicator
of the state the row is about to join. Available from the file everywhere.

*The read already done.* Partial ρ, holding the cell mean: **−0.0218** july_loo, **+0.0039**
december, **+0.0086** january — the july sign is negative, which is already a warning.

*Cheapest test that could kill it.* Compute the same spread from the **observed** clock at
the airports that have one and compare: the file clock is 350–466 s from the truth, so a
spread built on it is part surface and part instrument. If the observed-clock spread reads
materially higher, the file-clock version is measuring the instrument and dies.

*Failure mode.* Collinearity with the mean (the partial above is the honest version, and it
is the smallest of the three), and small cells — a spread over four rows is mostly noise,
which is why the read used a floor of four.

---

**What this says about the design.** All three are the same size as the §7q table's
remaining items and **none clears 0.1 s at ×1.2 even at its upper bound**, so none is worth
a fold-month or an upload slot on its own. The two worth one ridge each, if the desk wants a
lane to keep pushing, are **the regulation signature** (largest, and the test that kills it
is a read) and **the network term** (the only one that reaches `LTFM`); the spread is worth
running only as a by-product of whichever of those is built.

## 8. The fallback review

`review/round1-FALLBACK.txt`, reviewer `claude-fable-5-1` at effort high, fresh context,
one shot, on commit `d8ce206`. **Verdict BLOCKED**, two findings, both accepted and both
repaired. The reviewer's own summary of what survives: *"the underlying result, a
positive, permutation-clean neighbourhood signal, survives on the contract number."*

**Blocking 1 — the gate statistic over-credits the neighbourhood family.** Accepted;
§2 and §0 are rewritten and every table that carries a share now carries
`(2A − B) / before` beside it, with T2b adding the split half. The reviewer asked for
one of the two repairs; both are here, because they bracket the truth from opposite
sides and the `own` arm calibrates the bracket. The finding does not touch a `2A − B`,
an RMSE or the permutation placebo. What it does change is the claim "past the gate on
the no-clock rows", which is now written as a MAYBE with its three readings.

**Blocking 2 — the receipt-scaled columns.** Accepted; §6 gives the repair, the file is
re-emitted and re-hashed, and the manifest now records the rate and the delivered-day
count per airport per month. Verified on the file: the `_sc` columns are finite on the
5,131 rows of the harvested days and NaN elsewhere, at ×1.0599 rather than ×4.646.

**Non-blocking, taken:** the own-index assertion for sources `O` and `H` is now made on
every build and recorded (§4 A2); §4 says audit B is single-column; T3b's `Nf` row is
labelled a control; §5 discount 2 is corrected for january (EDDM's shared component is
smaller than EHAM's there); §0 no longer contradicts T5c on EGLL, and the difference
between receipt and reach is stated; the superseded `explain-july_loo-ehamweek-ridge.json`
and `-ehamweek.json` are deleted, so no record in `analysis/` was built on the old
matcher; the two interval readings of the same point are now quoted from one file.

**Non-blocking, noted and not acted on:** seven day-clusters undercover a percentile
bootstrap, so the intervals on the covered week are indicative and the harvested months
are what settle them; the `n_n_*` columns are self-indicators rather than runway state,
which is why they read at the statistic's noise floor when priced alone (T5b) and which
is the right way to read the permutation's −7.4 %; the anomaly base median and the
receipt rate each include the row's own value at 1/N weight; the ridge standardises over
all rows including the held-out block, which uses only feature values that exist for a
served month.

**Round 2** (`review/round2-FALLBACK.txt`, commit `a926e58`): **BLOCKED** on two, both
repaired — the outage rule's cost was assumed and is now measured with the hold lifted
(§6a, T5d), and the interface's hashes named a file that no longer existed (§6, T6).

**Round 3** (`review/round3-FALLBACK.txt`, commit `f21d260`): **BLOCKED** on one, and it
was the right finding. The projection was not read on the population the candidate serves,
and §0's "still clear with EDDF dropped" had no interval behind it. Accepted in full: the
projection is now read per reach class (§7), the leave-one-out intervals exist, and the
december claim is withdrawn. Everything else the reviewer verified clean, including the
candidate's bytes, the label-freedom of every serving path and the fit-side diagnosis.

**Round 4** (`review/round4-FALLBACK.txt`, commit `03c9e1e`): **OK**, no blocking
findings. The reviewer re-derived the projection's arithmetic, the cell hold, the double
gate and the candidate's bytes independently of `code/serve.py`, and confirmed *"every
fold interval clear of zero at full amplitude on the population served" is true of the
file written*. Three items were closed before any upload: **§7b, T9 and ledger row 499
priced the pre-hold file** and are re-run on the held update — where december turns out to
be clear ABOVE zero; **T8's four-way RMSE was the whole class under a caption saying
"served rows"** and now carries the rows the file moves, including the three sub-columns
that get worse; and **the served-cells interval and the currency are post-selection**,
which §7c now says, naming LTFM/unreached on december as the thin margin. Also taken: the
reviewer's neighbour-count sensitivity inside the reached class, a distinctness check and
a cross-lane agreement check on the prefix months, T7e repointed at the current record,
`serve.py`'s stale `v11` naming, and five number corrections.

**Renumbering:** the generated tables were §7, then §8, and are **§9** now; the review
section was §7 and is **§8**. Ledger rows written earlier cite them by their older
numbers.

## 9. The tables

<!-- TABLES:exp047 -->

<!-- generated by code/tables.py; do not edit by hand -->

### T1 the file's own clocks, ten airports, three folds

| arm | fold | rows | shared before (s²) | share removed | 95 % day-clustered | 2A − B (s²) | (2A − B) / before |
|---|---|---|---:|---:|---|---:|---:|
| Fto | july_loo | 190,713 | 8,157 | 2.8 % | [0.4, 4.9] | +118.0 | 1.5 % [-0, 3] |
| Fto | december | 165,677 | 5,117 | -0.9 % | [-7.2, 4.3] | -92.7 | -1.8 % [-8, 3] |
| Fto | january | 153,706 | 5,723 | -1.7 % | [-3.4, 0.2] | -187.1 | -3.3 % [-5, -2] |
| Fob | july_loo | 190,713 | 8,157 | 2.6 % | [-0.1, 4.8] | +131.5 | 1.6 % [-0, 4] |
| Fob | december | 165,677 | 5,117 | -0.0 % | [-3.4, 3.0] | -74.2 | -1.5 % [-5, 2] |
| Fob | january | 153,706 | 5,723 | -2.0 % | [-3.8, -0.3] | -149.4 | -2.6 % [-4, -1] |
| F | july_loo | 190,713 | 8,157 | 2.4 % | [-0.8, 4.9] | +61.4 | 0.8 % [-2, 3] |
| F | december | 165,677 | 5,117 | 0.7 % | [-3.3, 5.3] | -12.4 | -0.2 % [-4, 4] |
| F | january | 153,706 | 5,723 | -2.5 % | [-4.3, -0.5] | -220.3 | -3.9 % [-6, -2] |
| Fto **+2 h placebo** | july_loo | 190,713 | 8,157 | 1.0 % | [-0.8, 2.7] | -55.9 | -0.7 % [-2, 1] |
| Fob **+2 h placebo** | july_loo | 190,713 | 8,157 | 0.6 % | [-1.6, 2.5] | -50.1 | -0.6 % [-3, 1] |
| F **+2 h placebo** | july_loo | 190,713 | 8,157 | 0.9 % | [-1.1, 2.8] | -22.9 | -0.3 % [-2, 1] |

### T2 the observed clock on the pilot's EHAM week

5,131 rows, 7 days, airports EHAM, learner `ridge`, shared component **2,127 s²**.

| arm | cols | population | rows | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---|---:|---|---:|---:|---|---:|---:|
| Nf | 12 | all | 5,082 | -4.4 % | [-12.0, 1.0] | +139.1 | 6.5 % [3, 10] |
| Nf | 12 | own clock | 4,795 | -4.1 % | [-11.9, 1.1] | +136.4 | 6.4 % [3, 9] |
| Nf | 12 | no clock | 287 | -9.8 % | [-24.6, 3.2] | +183.6 | 9.9 % [-19, 34] |
| No | 12 | all | 5,082 | 73.1 % | [57.9, 88.9] | +818.3 | 38.5 % [26, 50] |
| No | 12 | own clock | 4,795 | 74.9 % | [59.1, 91.6] | +832.4 | 38.8 % [23, 53] |
| No | 12 | no clock | 287 | 39.7 % | [26.2, 67.5] | +583.3 | 31.5 % [-24, 77] |
| N | 24 | all | 5,082 | 75.0 % | [59.8, 89.6] | +843.5 | 39.7 % [27, 51] |
| N | 24 | own clock | 4,795 | 76.4 % | [61.2, 91.4] | +846.2 | 39.5 % [24, 54] |
| N | 24 | no clock | 287 | 47.4 % | [27.6, 82.2] | +799.2 | 43.1 % [-19, 91] |
| Oob | 45 | all | 5,082 | 8.0 % | [-1.4, 17.0] | +191.5 | 9.0 % [-0, 17] |
| Oob | 45 | own clock | 4,795 | 8.5 % | [-1.6, 17.6] | +263.7 | 12.3 % [2, 23] |
| Oob | 45 | no clock | 287 | -2.0 % | [-32.0, 11.3] | -1014.4 | -54.7 % [-145, -9] |
| Hob | 45 | all | 5,082 | 4.9 % | [-4.0, 13.0] | +147.7 | 6.9 % [-3, 16] |
| Hob | 45 | own clock | 4,795 | 5.4 % | [-4.0, 14.2] | +209.6 | 9.8 % [-0, 20] |
| Hob | 45 | no clock | 287 | -5.1 % | [-36.2, 9.8] | -886.3 | -47.8 % [-127, 4] |

### T3 beyond the row's OWN clock (exp-046's feature in the base)

5,131 rows, 7 days, airports EHAM, learner `ridge`, shared component **2,127 s²**.

| arm | cols | population | rows | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---|---:|---|---:|---:|---|---:|---:|
| own | 2 | all | 5,082 | 63.7 % | [42.3, 76.2] | +6462.1 | 303.8 % [249, 359] |
| own | 2 | own clock | 4,795 | 65.1 % | [43.4, 77.8] | +6884.4 | 321.2 % [267, 379] |
| own | 2 | no clock | 287 | 37.9 % | [2.8, 72.5] | -587.7 | -31.7 % [-138, 48] |
| No | 12 | all | 5,082 | 73.1 % | [57.9, 88.9] | +818.3 | 38.5 % [26, 50] |
| No | 12 | own clock | 4,795 | 74.9 % | [59.1, 91.6] | +832.4 | 38.8 % [23, 53] |
| No | 12 | no clock | 287 | 39.7 % | [26.2, 67.5] | +583.3 | 31.5 % [-24, 77] |
| own_No | 14 | all | 5,082 | 73.0 % | [52.7, 84.3] | +6665.4 | 313.4 % [258, 368] |
| own_No | 14 | own clock | 4,795 | 74.3 % | [51.8, 86.6] | +7118.4 | 332.1 % [271, 389] |
| own_No | 14 | no clock | 287 | 48.4 % | [7.0, 93.0] | -897.8 | -48.4 % [-138, 26] |
| own_No_Hob | 59 | all | 5,082 | 72.4 % | [52.7, 85.0] | +6668.9 | 313.6 % [257, 367] |
| own_No_Hob | 59 | own clock | 4,795 | 73.5 % | [52.6, 86.9] | +7115.6 | 332.0 % [278, 385] |
| own_No_Hob | 59 | no clock | 287 | 52.1 % | [19.0, 77.1] | -788.4 | -42.5 % [-130, 33] |

### T2b the split-half gate statistic — the correction sees neither member of the pair

`1 − after/before` reads the cross-product over the pairs of a cell, and the correction for row *j* is built from the OTHER rows of that same cell — row *i* among them. `E[r_i h_j]` therefore carries a term in `E[r_i r_i]`, which is idiosyncratic variance and not shared component, so the statistic **over-credits** this family (FALLBACK review round 1, blocking 1). Under the split half each row's neighbourhood is built from the OTHER half's clocks and the cross-product is read inside (cell, half), where no pair's correction has seen either member. The price is half the sources, so the split-half column is a LOWER bound: it carries the real attenuation of a halved receipt as well as the removal of the over-credit. The naive column at 50 % receipt (T4) is the same source density WITH the over-credit, so the two together separate them.

| arm | population | naive share (96 %) | naive share (50 %, T4) | split-half share | 95 % | split-half 2A − B (s²) |
|---|---|---:|---:|---:|---|---:|
| Nf | all | -4.4 % | – | -5.8 % | [-12.2, 0.5] | +9.7 |
| Nf | own clock | -4.1 % | – | -5.2 % | [-11.3, 1.0] | +10.8 |
| Nf | no clock | -9.8 % | – | -13.3 % | [-24.1, -5.3] | -8.7 |
| No | all | 73.1 % | 50.5 % | 16.5 % | [0.2, 30.9] | +435.0 |
| No | own clock | 74.9 % | – | 16.6 % | [-0.9, 31.2] | +457.5 |
| No | no clock | 39.7 % | – | 14.6 % | [-9.6, 42.7] | +60.0 |
| N | all | 75.0 % | – | 16.0 % | [-0.4, 31.6] | +401.0 |
| N | own clock | 76.4 % | – | 16.4 % | [-1.2, 31.6] | +419.6 |
| N | no clock | 47.4 % | – | 9.9 % | [-21.0, 41.3] | +92.0 |
| own | all | 63.7 % | 46.2 % | 65.6 % | [47.9, 77.9] | +6462.1 |
| own | own clock | 65.1 % | – | 68.1 % | [48.1, 80.3] | +6884.4 |
| own | no clock | 37.9 % | – | 30.9 % | [3.8, 54.3] | -587.7 |
| own_No | all | 73.0 % | 60.0 % | 68.8 % | [51.2, 81.9] | +6540.7 |
| own_No | own clock | 74.3 % | – | 70.9 % | [51.3, 85.2] | +6990.1 |
| own_No | no clock | 48.4 % | – | 38.8 % | [24.4, 49.9] | -961.3 |

### T2c the observed clock on a WHOLE FOLD MONTH, nine airports

december, 143,068 covered rows, 31 days, airports EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, learner `gbm`, shared component **5,037 s²**. This is §5's discount 2 answered rather than assumed: the EHAM week was one easy airport with the smallest shared component of the ten.

| arm | cols | population | rows | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---|---:|---|---:|---:|---|---:|---:|
| Nf | 12 | all | 139,848 | 4.3 % | [1.5, 7.0] | +188.3 | 3.7 % [0, 8] |
| Nf | 12 | own clock | 95,337 | 3.8 % | [1.4, 6.3] | +39.2 | 0.9 % [-3, 4] |
| Nf | 12 | no clock | 44,511 | 5.1 % | [0.0, 10.1] | +509.5 | 8.6 % [2, 18] |
| No | 12 | all | 139,848 | 27.9 % | [21.9, 34.2] | +733.6 | 14.6 % [8, 21] |
| No | 12 | own clock | 95,337 | 42.5 % | [35.5, 50.1] | +1065.1 | 23.2 % [15, 33] |
| No | 12 | no clock | 44,511 | 4.8 % | [-1.8, 11.5] | +19.3 | 0.3 % [-7, 7] |
| N | 24 | all | 139,848 | 34.1 % | [29.4, 39.5] | +978.8 | 19.4 % [15, 25] |
| N | 24 | own clock | 95,337 | 50.6 % | [43.9, 58.0] | +1283.8 | 27.9 % [21, 36] |
| N | 24 | no clock | 44,511 | 8.1 % | [3.4, 13.8] | +321.7 | 5.4 % [0, 10] |
| own | 2 | all | 139,848 | 29.8 % | [25.8, 34.0] | +5239.0 | 104.0 % [93, 117] |
| own | 2 | own clock | 95,337 | 43.5 % | [36.9, 50.4] | +7667.7 | 166.8 % [148, 187] |
| own | 2 | no clock | 44,511 | 8.2 % | [6.2, 10.5] | +5.4 | 0.1 % [-0, 1] |
| own_No | 14 | all | 139,848 | 34.3 % | [28.9, 39.5] | +5795.1 | 115.1 % [104, 128] |
| own_No | 14 | own clock | 95,337 | 51.5 % | [45.5, 57.4] | +8509.6 | 185.1 % [167, 206] |
| own_No | 14 | no clock | 44,511 | 7.2 % | [1.6, 13.0] | -54.1 | -0.9 % [-7, 4] |

### T2d the same month under the corrected statistic, the placebo and the second learner

The split half is the clean reading at half the source density; the `own` arm is the control that cannot contaminate a pair, and it must not move. Two permutation columns, because dealing only the OBSERVED clocks leaves `Nf_*` untouched and so is a CONTROL for that family and not a test: `--permute-file` deals the `AOBT_3_flt` off-blocks out inside each airport-day as well.

| arm | naive share | split-half share | 95 % | naive 2A − B | split-half 2A − B | permute obs | permute obs+file | ridge share |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Nf | 4.3 % | 2.0 % | [-1.4, 5.4] | +188.3 | +172.3 | 4.3 % | -0.6 % | -1.6 % |
| No | 27.9 % | 9.5 % | [6.5, 12.9] | +733.6 | +556.1 | -0.5 % | -0.5 % | 10.4 % |
| N | 34.1 % | 13.7 % | [10.3, 16.9] | +978.8 | +706.2 | 3.1 % | -1.0 % | 10.5 % |
| own | 29.8 % | 30.5 % | [25.6, 35.4] | +5239.0 | +5239.0 | – | – | 12.1 % |
| own_No | 34.3 % | 33.0 % | [28.3, 37.9] | +5795.1 | +5749.9 | – | – | 10.7 % |

### T2f the population this lane serves, on a whole fold month

december's no-clock rows are 44,511 of the 139,848 the harvest covers — 31 % — and they are where the desk's 14:10 UTC division sends this lane. `2A − B` per row OF THAT POPULATION, gbm, cross-fitted by day block. The EHAM week's 287 rows read +583 s² on `No`; a whole month at nine airports does not.

| arm | rows | share removed | 95 % | 2A − B (s²) | 95 % | (2A − B) / before |
|---|---:|---:|---|---:|---|---:|
| Nf | 44,511 | 5.1 % | [0.0, 10.1] | +509.5 | [+99, +1067] | 8.6 % [2, 18] |
| No | 44,511 | 4.8 % | [-1.8, 11.5] | +19.3 | [-440, +405] | 0.3 % [-7, 7] |
| N | 44,511 | 8.1 % | [3.4, 13.8] | +321.7 | [+27, +585] | 5.4 % [0, 10] |
| geo | 44,511 | 0.9 % | [0.1, 1.8] | +101.1 | [+31, +168] | 1.7 % [1, 3] |
| Oob | 44,511 | 3.1 % | [-3.8, 10.6] | -261.6 | [-1011, +295] | -4.4 % [-17, 5] |
| Hob | 44,511 | 2.1 % | [-3.8, 8.5] | -244.8 | [-737, +142] | -4.1 % [-12, 2] |
| all | 44,511 | 11.0 % | [4.4, 19.1] | +399.1 | [-281, +894] | 6.7 % [-5, 15] |
| own | 44,511 | 8.2 % | [6.2, 10.5] | +5.4 | [-23, +31] | 0.1 % [-0, 1] |
| own_No | 44,511 | 7.2 % | [1.6, 13.0] | -54.1 | [-425, +251] | -0.9 % [-7, 4] |

### T2e the state block on the observed clocks, nine airports, december

| arm | cols | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---|---:|---:|---|---:|---:|
| Hob | 45 | 6.6 % | [3.0, 10.5] | +226.3 | 4.5 % [0, 8] |
| Oob | 45 | 8.8 % | [4.6, 13.0] | +252.0 | 5.0 % [-1, 10] |
| all | 179 | 36.3 % | [31.3, 42.3] | +1243.6 | 24.7 % [19, 31] |
| geo | 3 | 0.7 % | [0.2, 1.2] | +45.3 | 0.9 % [0, 1] |

### T3b the permutation placebo — the clocks dealt out to other aircraft (the `Nf` row is a CONTROL: only the observed clocks are dealt)

5,131 rows, 7 days, airports EHAM, learner `ridge`, shared component **2,127 s²**.

| arm | cols | population | rows | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---|---:|---|---:|---:|---|---:|---:|
| Nf | 12 | all | 5,082 | -4.4 % | [-13.0, 0.9] | +139.1 | 6.5 % [3, 10] |
| Nf | 12 | own clock | 4,795 | -4.1 % | [-12.0, 1.2] | +136.4 | 6.4 % [3, 9] |
| Nf | 12 | no clock | 287 | -9.8 % | [-23.7, 2.3] | +183.6 | 9.9 % [-18, 36] |
| No | 12 | all | 5,082 | -7.4 % | [-9.7, -5.1] | -193.0 | -9.1 % [-14, -4] |
| No | 12 | own clock | 4,795 | -6.9 % | [-9.7, -4.2] | -187.6 | -8.8 % [-14, -3] |
| No | 12 | no clock | 287 | -15.8 % | [-31.1, -7.8] | -282.3 | -15.2 % [-41, 5] |
| N | 24 | all | 5,082 | -8.5 % | [-16.7, -1.7] | +23.1 | 1.1 % [-5, 7] |
| N | 24 | own clock | 4,795 | -7.8 % | [-16.3, -0.8] | +24.2 | 1.1 % [-6, 9] |
| N | 24 | no clock | 287 | -21.1 % | [-46.8, -10.1] | +4.5 | 0.2 % [-32, 27] |

### T3c the repair attempt: one head per reach class

`2A − B` per row of the named population, EHAM week, ridge. The left column is one head with a missing indicator, the right is one head fitted on each class separately.

| arm | population | one head (s²) | 95 % | one head per class (s²) | 95 % |
|---|---|---:|---|---:|---|
| own | own clock | +6884.4 | [+5730, +8132] | +6884.5 | [+5730, +8132] |
| own | no clock | -587.7 | [-2557, +889] | -612.8 | [-2638, +907] |
| No | own clock | +832.4 | [+499, +1129] | +827.3 | [+497, +1114] |
| No | no clock | +583.3 | [-441, +1428] | -986.7 | [-2857, +825] |
| own_No | own clock | +7118.4 | [+5810, +8338] | +7123.4 | [+5821, +8309] |
| own_No | no clock | -897.8 | [-2562, +479] | -986.7 | [-2857, +811] |
| own_No_Hob | own clock | +7115.6 | [+5958, +8252] | +7112.7 | [+5879, +8263] |
| own_No_Hob | no clock | -788.4 | [-2404, +606] | -3198.7 | [-5186, -1220] |

### T4 the receipt-response curve: what an airport heard at *r* would yield

EHAM's observed clocks thinned to a fraction at random, on the same week and the same estimator, so the curve is read where the truth is known.

| receipt | clocks left | arm | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---:|---:|---|---:|---|---:|---:|
| 16% | 777 | No | 4.7 % | [-3.5, 11.8] | +70.7 | 3.3 % [-8, 13] |
| 16% | 777 | own | 7.4 % | [2.0, 13.4] | +734.3 | 34.5 % [29, 40] |
| 16% | 777 | own_No | 7.2 % | [-8.2, 20.0] | +778.6 | 36.6 % [24, 45] |
| 30% | 1,481 | No | 28.1 % | [17.1, 37.5] | +384.3 | 18.1 % [6, 27] |
| 30% | 1,481 | own | 15.2 % | [6.7, 21.6] | +1184.2 | 55.7 % [22, 81] |
| 30% | 1,481 | own_No | 30.0 % | [15.0, 41.8] | +1490.5 | 70.1 % [40, 95] |
| 50% | 2,402 | No | 50.5 % | [40.5, 62.2] | +664.4 | 31.2 % [18, 42] |
| 50% | 2,402 | own | 46.2 % | [31.7, 58.0] | +3691.3 | 173.6 % [149, 197] |
| 50% | 2,402 | own_No | 60.0 % | [42.8, 73.2] | +3986.1 | 187.4 % [168, 208] |
| 70% | 3,406 | No | 59.2 % | [43.8, 75.0] | +673.0 | 31.6 % [11, 49] |
| 70% | 3,406 | own | 53.4 % | [36.7, 63.9] | +4925.1 | 231.6 % [180, 281] |
| 70% | 3,406 | own_No | 63.6 % | [41.6, 77.5] | +5179.9 | 243.6 % [188, 291] |
| 90% | 4,339 | No | 70.1 % | [53.0, 87.5] | +763.0 | 35.9 % [21, 48] |
| 90% | 4,339 | own | 59.7 % | [36.6, 73.2] | +5766.4 | 271.1 % [222, 322] |
| 90% | 4,339 | own_No | 69.9 % | [46.6, 84.3] | +5968.4 | 280.6 % [228, 328] |
| 96 % (real) | – | No | 73.1 % | [57.9, 88.9] | +818.3 | 38.5 % [26, 50] |
| 96 % (real) | – | own | 63.7 % | [42.3, 76.2] | +6462.1 | 303.8 % [249, 359] |
| 96 % (real) | – | own_No | 73.0 % | [52.7, 84.3] | +6665.4 | 313.4 % [258, 368] |

### T5 the leakage audit and the anchor diagnostic

| check | result |
|---|---|
| A — a count moved by more than the row's own event | none |
| A — counts that never contain the row (and must not) | 17 of 62 |
| B — the column that best predicts the row's own taxi span | `Fto_s_taxi_rwy_30m`, Spearman +0.668, 72 % of the span's variance survives conditioning on it |
| C — departure `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` read | 0 / 0 non-null (the mask ran) |
| D — the same state at the take-off and the off-block instant | median Pearson 0.778: the anchor is a different reading, not a relabelling |

### T7 step 2 — the SERVING class on three folds, the head fitted on every row of the panels it is not scored on. THE GATED READING

`arms-step2-unmatched-fitall-3f.json` — head v12, matcher `ba41935e…`, priced on **unmatched** rows, fitted on **all** rows, missingness indicator off, fit floor on receipt 0.0. The projection carries 154,358 of the ranking file's 344,841 rows (44.8 %). **The gate is the s² interval, not the board column**: that column is APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the smaller of the two bases, so it is the larger of the two possible second-counts; the same drop at the ALL-ROW board (308.5977) is about 0.83× as many seconds. `code/serve.py` measures the served file's own board directly rather than converting. The projection is read per airport-MONTH and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is priced separately on rows with and without a heard neighbour and each half is applied to that airport-month's own reached share. The last two columns are leave-one-out readings off the SAME draws — the worst single airport, and the served-cells-only projection that gates the file written.

| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | worst airport dropped | served cells only |
|---|---|---:|---:|---:|---:|---|---|
| `Nf` | july_loo | 103,514 | 165,677 | +333 [+133, +562] | +164 [+70, +274] | LTFM → +28 [-24, +88] | +136 [+59, +232] (1 cells) |
| `Nf` | december | 64,850 | 190,713 | +221 [+29, +451] | +146 [+65, +235] | LTFM → +74 [+10, +149] | +72 [+29, +129] (1 cells) |
| `Nf` | january | 84,933 | 356,390 | +79 [-77, +250] | +44 [-27, +122] | LTFM → -5 [-79, +64] | +49 [+12, +90] (1 cells) |
| `geo` | july_loo | 103,514 | 165,677 | -15 [-61, +29] | -6 [-24, +12] | LTFM → -27 [-37, -16] | +0 [+0, +0] (0 cells) |
| `geo` | december | 64,850 | 190,713 | -36 [-74, +0] | -12 [-33, +13] | LFPG → -18 [-37, +3] | +0 [+0, +0] (0 cells) |
| `geo` | january | 84,933 | 356,390 | -17 [-28, -6] | -6 [-11, -2] | EHAM → -6 [-11, -2] | +0 [+0, +0] (0 cells) |
| `Nf_geo` | july_loo | 103,514 | 165,677 | +324 [+118, +563] | +161 [+70, +271] | LTFM → +19 [-33, +80] | +141 [+63, +243] (1 cells) |
| `Nf_geo` | december | 64,850 | 190,713 | +163 [-53, +423] | +131 [+34, +245] | LTFM → +55 [-29, +159] | +76 [+30, +143] (1 cells) |
| `Nf_geo` | january | 84,933 | 356,390 | +51 [-114, +232] | +36 [-39, +117] | LTFM → -12 [-91, +65] | +48 [+10, +89] (1 cells) |
| `No` | july_loo | 103,514 | 165,677 | +313 [+183, +459] | +183 [+112, +265] | EDDF → +69 [+12, +141] | +114 [+76, +158] (1 cells) |
| `No` | december | 64,850 | 190,713 | -560 [-1,162, -140] | +153 [-35, +371] | EDDF → -9 [-125, +104] | +162 [+63, +308] (1 cells) |
| `No` | january | 84,933 | 356,390 | -33 [-172, +89] | +40 [-11, +95] | EDDF → -11 [-50, +29] | +50 [+1, +110] (1 cells) |
| `N` | july_loo | 103,514 | 165,677 | +504 [+312, +718] | +286 [+178, +401] | EDDF → +145 [+50, +250] | +246 [+165, +338] (2 cells) |
| `N` | december | 64,850 | 190,713 | -373 [-880, +10] | +218 [+31, +454] | EDDF → +47 [-71, +170] | +203 [+94, +360] (2 cells) |
| `N` | january | 84,933 | 356,390 | +235 [+70, +402] | +154 [+79, +233] | EDDF → +94 [+37, +150] | +96 [+41, +158] (2 cells) |
| `own` | july_loo | 103,514 | 165,677 | +8 [-27, +40] | +4 [-10, +19] | LTFM → -16 [-25, -8] | +0 [+0, +0] (0 cells) |
| `own` | december | 64,850 | 190,713 | -6 [-18, +5] | -4 [-9, +1] | EGLL → -4 [-9, +0] | +0 [+0, +0] (0 cells) |
| `own` | january | 84,933 | 356,390 | -12 [-22, -1] | -5 [-9, -1] | EHAM → -5 [-9, -1] | +0 [+0, +0] (0 cells) |

### T7a the same rows with the head fitted on unmatched rows only, two folds — the reading the fit-on-all repair replaced

`arms-step2-unmatched.json` — head v12, matcher `ba41935e…`, priced on **unmatched** rows, fitted on **unmatched** rows, missingness indicator off, fit floor on receipt 0.0. The projection carries 154,727 of the ranking file's 344,841 rows (44.9 %). **The gate is the s² interval, not the board column**: that column is APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the smaller of the two bases, so it is the larger of the two possible second-counts; the same drop at the ALL-ROW board (308.5977) is about 0.83× as many seconds. `code/serve.py` measures the served file's own board directly rather than converting. The projection is read per airport-MONTH and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is priced separately on rows with and without a heard neighbour and each half is applied to that airport-month's own reached share. The last two columns are leave-one-out readings off the SAME draws — the worst single airport, and the served-cells-only projection that gates the file written.

| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | worst airport dropped | served cells only |
|---|---|---:|---:|---:|---:|---|---|
| `Nf` | july_loo | 103,537 | 64,853 | +265 [-25, +588] | -148 [-406, +103] | – | – |
| `Nf` | december | 64,853 | 103,537 | +107 [-95, +333] | +124 [+32, +225] | – | – |
| `geo` | july_loo | 103,537 | 64,853 | -24 [-64, +14] | -89 [-133, -44] | – | – |
| `geo` | december | 64,853 | 103,537 | -9 [-38, +25] | +0 [-14, +15] | – | – |
| `Nf_geo` | july_loo | 103,537 | 64,853 | +248 [-48, +567] | -183 [-442, +109] | – | – |
| `Nf_geo` | december | 64,853 | 103,537 | +62 [-137, +291] | +104 [+1, +218] | – | – |
| `No` | july_loo | 103,537 | 64,853 | +285 [+185, +400] | +357 [+137, +666] | – | – |
| `No` | december | 64,853 | 103,537 | -303 [-786, +42] | +220 [+43, +446] | – | – |
| `N` | july_loo | 103,537 | 64,853 | +455 [+154, +767] | +301 [-56, +685] | – | – |
| `N` | december | 64,853 | 103,537 | -233 [-752, +193] | +249 [+58, +482] | – | – |
| `own` | july_loo | 103,537 | 64,853 | +6 [-12, +25] | -11 [-24, +1] | – | – |
| `own` | december | 64,853 | 103,537 | +2 [-21, +24] | +6 [-3, +15] | – | – |

### T7b the first FIT-side selection that FAILED — no fitting row on an airport-day below 0.5 receipt, which selects on the outcome

`arms-step2-unmatched-fit50.json` — head v12, matcher `ba41935e…`, priced on **unmatched** rows, fitted on **unmatched** rows, missingness indicator off, fit floor on receipt 0.5. The projection carries 154,727 of the ranking file's 344,841 rows (44.9 %). **The gate is the s² interval, not the board column**: that column is APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the smaller of the two bases, so it is the larger of the two possible second-counts; the same drop at the ALL-ROW board (308.5977) is about 0.83× as many seconds. `code/serve.py` measures the served file's own board directly rather than converting. The projection is read per airport-MONTH and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is priced separately on rows with and without a heard neighbour and each half is applied to that airport-month's own reached share. The last two columns are leave-one-out readings off the SAME draws — the worst single airport, and the served-cells-only projection that gates the file written.

| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | worst airport dropped | served cells only |
|---|---|---:|---:|---:|---:|---|---|
| `Nf` | july_loo | 103,537 | 33,923 | +272 [-65, +644] | -234 [-549, +99] | – | – |
| `Nf` | december | 64,853 | 38,852 | -115 [-341, +132] | +47 [-68, +168] | – | – |
| `Nf_geo` | july_loo | 103,537 | 33,923 | +245 [-74, +604] | -267 [-568, +60] | – | – |
| `Nf_geo` | december | 64,853 | 38,852 | -994 [-1,368, -575] | -378 [-578, -153] | – | – |
| `No` | july_loo | 103,537 | 33,923 | +292 [+174, +406] | +288 [+73, +642] | – | – |
| `No` | december | 64,853 | 38,852 | -2,147 [-3,214, -1,341] | -315 [-744, +64] | – | – |
| `N` | july_loo | 103,537 | 33,923 | +412 [-5, +818] | +119 [-278, +624] | – | – |
| `N` | december | 64,853 | 38,852 | -2,020 [-3,045, -1,154] | -313 [-815, +148] | – | – |
| `own` | july_loo | 103,537 | 33,923 | +4 [-5, +12] | -5 [-11, +1] | – | – |
| `own` | december | 64,853 | 38,852 | -73 [-155, +13] | -14 [-47, +20] | – | – |

### T7c the second FIT-side selection that FAILED — no fitting row at an airport-MONTH whose reach is more than 0.25 from the ranking file's. It selects on the instrument and not on the outcome, and it starves `No_*` of the airports where an observation exists

`arms-step2-unmatched-fitall-3f-gap25.json` — head v12, matcher `ba41935e…`, priced on **unmatched** rows, fitted on **all** rows, missingness indicator off, fit floor on receipt 0.0. The projection carries 154,727 of the ranking file's 344,841 rows (44.9 %). **The gate is the s² interval, not the board column**: that column is APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the smaller of the two bases, so it is the larger of the two possible second-counts; the same drop at the ALL-ROW board (308.5977) is about 0.83× as many seconds. `code/serve.py` measures the served file's own board directly rather than converting. The projection is read per airport-MONTH and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is priced separately on rows with and without a heard neighbour and each half is applied to that airport-month's own reached share. The last two columns are leave-one-out readings off the SAME draws — the worst single airport, and the served-cells-only projection that gates the file written.

| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | worst airport dropped | served cells only |
|---|---|---:|---:|---:|---:|---|---|
| `Nf` | july_loo | 103,537 | 116,285 | +314 [+118, +535] | +1 [-182, +189] | – | – |
| `Nf` | december | 64,853 | 98,735 | +168 [-8, +374] | +144 [+54, +245] | – | – |
| `Nf` | january | 84,935 | 215,020 | -0 [-202, +200] | +20 [-71, +123] | – | – |
| `No` | july_loo | 103,537 | 116,285 | -119 [-624, +301] | -486 [-1,790, +760] | – | – |
| `No` | december | 64,853 | 98,735 | -1,092 [-1,866, -438] | -228 [-693, +104] | – | – |
| `No` | january | 84,935 | 215,020 | -2 [-161, +152] | +95 [+20, +184] | – | – |
| `N` | july_loo | 103,537 | 116,285 | -196 [-943, +424] | -1,149 [-3,308, +642] | – | – |
| `N` | december | 64,853 | 98,735 | -1,381 [-2,492, -563] | -419 [-1,113, +33] | – | – |
| `N` | january | 84,935 | 215,020 | +74 [-114, +258] | +149 [+43, +254] | – | – |
| `own` | july_loo | 103,537 | 116,285 | +7 [-17, +30] | -15 [-32, +2] | – | – |
| `own` | december | 64,853 | 98,735 | -0 [-28, +28] | +6 [-5, +18] | – | – |
| `own` | january | 84,935 | 215,020 | -52 [-93, -12] | -22 [-41, -3] | – | – |

### T7d the with-clock class — fitted and priced on matched rows only (surfacefit's head; these are the columns it can take as inputs)

`arms-step2-matched-3f.json` — head v12, matcher `ba41935e…`, priced on **matched** rows, fitted on **matched** rows, missingness indicator off, fit floor on receipt 0.0. The projection carries 190,095 of the ranking file's 344,841 rows (55.1 %). **The gate is the s² interval, not the board column**: that column is APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the smaller of the two bases, so it is the larger of the two possible second-counts; the same drop at the ALL-ROW board (308.5977) is about 0.83× as many seconds. `code/serve.py` measures the served file's own board directly rather than converting. The projection is read per airport-MONTH and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is priced separately on rows with and without a heard neighbour and each half is applied to that airport-month's own reached share. The last two columns are leave-one-out readings off the SAME draws — the worst single airport, and the served-cells-only projection that gates the file written.

| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | worst airport dropped | served cells only |
|---|---|---:|---:|---:|---:|---|---|
| `own` | july_loo | 87,199 | 100,827 | +1,429 [-5,352, +6,498] | +1,439 [+731, +2,025] | EDDM → +1,095 [+398, +1,669] | +1,134 [+1,044, +1,225] (4 cells) |
| `own` | december | 100,827 | 87,199 | +1,826 [+1,458, +2,215] | +822 [+571, +1,113] | LSZH → +596 [+349, +892] | +751 [+621, +868] (4 cells) |
| `own` | january | 68,773 | 188,026 | +3,429 [+2,936, +3,910] | +1,694 [+1,440, +1,979] | EGLL → +995 [+862, +1,131] | +801 [+689, +920] (4 cells) |
| `No` | july_loo | 87,199 | 100,827 | +1,020 [+448, +1,623] | +273 [-30, +537] | LEMD → +135 [-171, +387] | +192 [+143, +246] (2 cells) |
| `No` | december | 100,827 | 87,199 | -207 [-718, +288] | -524 [-855, -183] | LSZH → -614 [-937, -265] | +145 [+91, +211] (2 cells) |
| `No` | january | 68,773 | 188,026 | +191 [-96, +461] | +53 [-86, +184] | LEBL → -3 [-137, +130] | +90 [+49, +139] (2 cells) |
| `own_No` | july_loo | 87,199 | 100,827 | +610 [-6,596, +6,358] | +1,450 [+677, +2,114] | EDDM → +1,099 [+330, +1,769] | +1,258 [+1,172, +1,356] (4 cells) |
| `own_No` | december | 100,827 | 87,199 | +1,678 [+1,132, +2,218] | +679 [+364, +999] | LSZH → +414 [+89, +751] | +791 [+629, +932] (4 cells) |
| `own_No` | january | 68,773 | 188,026 | +3,539 [+3,060, +4,016] | +1,721 [+1,469, +2,019] | EGLL → +1,030 [+882, +1,178] | +852 [+728, +973] (4 cells) |
| `N` | july_loo | 87,199 | 100,827 | +1,508 [+894, +2,183] | +122 [-648, +657] | LEMD → -32 [-806, +497] | +76 [+50, +104] (1 cells) |
| `N` | december | 100,827 | 87,199 | +15 [-452, +483] | -358 [-679, -41] | LSZH → -463 [-775, -146] | +88 [+56, +126] (1 cells) |
| `N` | january | 68,773 | 188,026 | +387 [+7, +761] | +164 [+15, +308] | EDDF → +112 [-39, +247] | +52 [+19, +93] (1 cells) |

### T7e what the ranking file carries, per airport-MONTH

The no-clock class this lane serves. `matched` is the ranking month's own census on matcher `ba41935e…`; every ranking row that is not matched is no-clock, including every row on an airport-day the harvest never delivered — which is why LIRF's 2026-07 and all of LTFM are here in full.

The last two columns are the REACH split the projection now works at: the share of the class's rows in that month with at least one heard aircraft in the cell.

| airport | 2026-01 rows | matched | reached | 2026-07 rows | matched | reached | no-clock rows | weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 15,667 | 7,388 | 0.643 | 20,650 | 7,308 | 0.821 | 21,621 | 0.0627 |
| EDDM | 10,953 | 10,531 | 0.926 | 15,013 | 14,461 | 0.975 | 974 | 0.0028 |
| EGLL | 19,283 | 13,558 | 0.965 | 20,557 | 19,756 | 0.990 | 6,528 | 0.0189 |
| EHAM | 16,232 | 15,007 | 0.950 | 21,950 | 20,566 | 0.986 | 2,609 | 0.0076 |
| LEBL | 12,201 | 10,733 | 0.863 | 17,879 | 15,858 | 0.963 | 3,489 | 0.0101 |
| LEMD | 16,802 | 15,840 | 0.930 | 20,152 | 10,999 | 0.856 | 10,121 | 0.0293 |
| LFPG | 18,129 | 739 | 0.119 | 21,743 | 873 | 0.100 | 38,260 | 0.1109 |
| LIRF | 11,061 | 5,325 | 0.625 | 15,838 | 11 | 0.002 | 21,218 | 0.0615 |
| LSZH | 9,995 | 9,155 | 0.889 | 13,155 | 12,033 | 0.975 | 1,961 | 0.0057 |
| LTFM | 22,396 | 0 | 0.000 | 25,185 | 0 | 0.000 | 47,577 | 0.1380 |
| **all** | | | | | | | **154,358** | **0.4476** |

### T7g the cell hold — every (airport, reach class) cell of arm `N`, its fold-side support, and what holding it gives up

The desk's rule of 2026-09-09: a cell whose own day-clustered fold interval is not clear of zero ABOVE, or which is priced on fewer than 500 rows in ANY fold, is held at the base. **63,862** of **154,358** class rows are served.

| cell | ranking rows | july_loo — rows, s²/row [95 %] | december — rows, s²/row [95 %] | january — rows, s²/row [95 %] | verdict |
|---|---:|---|---|---|---|
| `EDDF/reached` | 16,284 | 4,674, +3,003 [+1,970, +4,166] | 1,561, +3,620 [+1,395, +6,998] | 3,586, +1,261 [+347, +2,346] | **served** |
| `LTFM/unreached` | 47,577 | 25,171, +756 [+342, +1,301] | 22,609, +232 [+44, +433] | 21,368, +264 [+57, +492] | **served** |
| `EDDF/unreached` | 5,337 | 52, unpriced | 43, unpriced | 155, unpriced | held |
| `EDDM/reached` | 929 | 230, unpriced | 408, unpriced | 149, unpriced | held |
| `EDDM/unreached` | 45 | 9,820, +310 [+32, +684] | 39, unpriced | 274, unpriced | held |
| `EGLL/reached` | 6,317 | 3,668, -523 [-1,384, +330] | 9,772, -4,308 [-7,490, -2,205] | 9,275, +90 [-671, +915] | held |
| `EGLL/unreached` | 211 | 15,922, -142 [-402, +155] | 1,444, +129 [-716, +1,022] | 1,698, -861 [-1,626, -56] | held |
| `EHAM/reached` | 2,528 | 1,283, +651 [+222, +1,154] | 1,168, -1,879 [-3,737, -582] | 643, +2,392 [-492, +5,732] | held |
| `EHAM/unreached` | 81 | 13, unpriced | 32, unpriced | 16, unpriced | held |
| `LEBL/reached` | 3,214 | 2,639, +2,234 [+938, +3,703] | 1,087, +1,664 [-66, +3,940] | 916, +1,222 [+520, +2,075] | held |
| `LEBL/unreached` | 275 | 321, unpriced | 99, unpriced | 120, unpriced | held |
| `LEMD/reached` | 8,739 | 9,366, +1,163 [+688, +1,638] | 818, +1,370 [+567, +2,326] | 0, unpriced | held |
| `LEMD/unreached` | 1,382 | 5,313, +243 [-90, +637] | 48, unpriced | 16,509, +159 [+27, +302] | held |
| `LFPG/reached` | 4,152 | 974, -266 [-1,289, +745] | 2,570, -241 [-1,192, +473] | 0, unpriced | held |
| `LFPG/unreached` | 34,108 | 20,342, -162 [-387, +37] | 16,203, +213 [-129, +806] | 17,900, +297 [-27, +719] | held |
| `LIRF/reached` | 3,567 | 1,823, +750 [-3,997, +6,618] | 3,913, -157 [-1,890, +1,410] | 0, unpriced | held |
| `LIRF/unreached` | 17,651 | 141, unpriced | 2,388, +666 [-628, +2,076] | 11,317, -94 [-324, +165] | held |
| `LSZH/reached` | 1,840 | 1,702, +988 [-467, +2,561] | 598, +1,165 [-498, +2,809] | 909, +427 [-1,352, +2,187] | held |
| `LSZH/unreached` | 121 | 60, unpriced | 50, unpriced | 98, unpriced | held |

Projected gain forgone by the hold: **+40 s²** on july_loo, **+14 s²** on december, **+58 s²** on january.

### T7h why 17 of the 19 cells lack support — the instrument or the arm

The desk's question of 2026-09-09: does a held cell fail for want of 2025 FOLD ROWS, which more harvested months would fix, or for want of a signal, which they would not? The separator is the cell's reading POOLED over the three folds, on the union of their days. Pooling is rejected as a SERVING rule (§7c: it would keep the 141-row LIRF cell); it is the right reading for the diagnostic question of whether anything is there at all. Cell gains reproduce the record exactly (max |Δ| 0e+00 s²).

| cell | ranking rows | fold rows jul / dec / jan | pooled rows | pooled s²/row [95 %] | held by |
|---|---:|---|---:|---|---|
| `LTFM/unreached` | 47,577 | 25,171 / 22,609 / 21,368 | 69,148 | +433 [+246, +666] | **served** |
| `LFPG/unreached` | 34,108 | 20,342 / 16,203 / 17,900 | 54,445 | +100 [-82, +335] | the **arm** |
| `LIRF/unreached` | 17,651 | 141 / 2,388 / 11,317 | 13,846 | +128 [-174, +472] | undecided |
| `EDDF/reached` | 16,284 | 4,674 / 1,561 / 3,586 | 9,821 | +2,465 [+1,662, +3,391] | **served** |
| `LEMD/reached` | 8,739 | 9,366 / 818 / 0 | 10,184 | +1,180 [+784, +1,636] | the **instrument** |
| `EGLL/reached` | 6,317 | 3,668 / 9,772 / 9,275 | 22,715 | -1,901 [-3,336, -849] | the **arm** |
| `EDDF/unreached` | 5,337 | 52 / 43 / 155 | 250 | +1,220 [-321, +3,718] | undecided |
| `LFPG/reached` | 4,152 | 974 / 2,570 / 0 | 3,544 | -248 [-951, +347] | undecided |
| `LIRF/reached` | 3,567 | 1,823 / 3,913 / 0 | 5,736 | +131 [-1,761, +2,216] | undecided |
| `LEBL/reached` | 3,214 | 2,639 / 1,087 / 916 | 4,642 | +1,901 [+1,057, +2,838] | month to month |
| `EHAM/reached` | 2,528 | 1,283 / 1,168 / 643 | 3,094 | +58 [-932, +993] | the **arm** |
| `LSZH/reached` | 1,840 | 1,702 / 598 / 909 | 3,209 | +862 [-110, +1,875] | the **arm** |
| `LEMD/unreached` | 1,382 | 5,313 / 48 / 16,509 | 21,870 | +179 [+50, +329] | the **instrument** |
| `EDDM/reached` | 929 | 230 / 408 / 149 | 787 | +2,730 [-495, +6,250] | undecided |
| `LEBL/unreached` | 275 | 321 / 99 / 120 | 540 | +413 [-389, +1,454] | undecided |
| `EGLL/unreached` | 211 | 15,922 / 1,444 / 1,698 | 19,064 | -186 [-437, +75] | the **arm** |
| `LSZH/unreached` | 121 | 60 / 50 / 98 | 208 | +778 [-562, +2,343] | undecided |
| `EHAM/unreached` | 81 | 13 / 32 / 16 | 61 | -865 [-4,102, +1,991] | undecided |
| `EDDM/unreached` | 45 | 9,820 / 39 / 274 | 10,133 | +285 [+12, +642] | the **instrument** |

**More fold months can revive 12 of the 17 held cells, 45,493 ranking rows**: the 3 the instrument holds (10,166 rows, pooled read already clear above zero), the 1 held month to month (3,214), and the 8 too thin to say (32,113). **They cannot revive the 5 the arm holds, 45,004 ranking rows** — and that half contains the largest held cell of all, `LFPG/unreached` at 34,108 ranking rows, whose pooled read on 54,445 fold rows is +100 [−82, +335], and `EGLL/reached`, where the correction HURTS at −1,901 [−3,336, −849] on 22,715 rows.

### T7i the capacity of arm `N`'s head, at the registered rotation

Held fixed: the rotation, the columns (`Nf_*` and `No_*`), the three folds, the no-clock population, the fitting rows (all) and the head (v12). Only the learner moves. The GBM is fitted on MEAN-IMPUTED, standardised columns, so no split can read an absent observation (the desk's 16:50 UTC rule).

| head | cells it supports | served-only s² jul / dec / jan | class s² jul / dec / jan | `EDDF/reached` s²/row | `LTFM/unreached` s²/row |
|---|---|---|---|---|---|
| ridge-a0.1 | `EDDF/reached`, `LTFM/unreached` | +246 / +203 / +96 | +503 / -373 / +235 | +3,003 / +3,621 / +1,261 | +756 / +233 / +264 |
| ridge-a1 | `EDDF/reached`, `LTFM/unreached` | +246 / +203 / +96 | +503 / -373 / +235 | +3,003 / +3,621 / +1,261 | +756 / +233 / +264 |
| **ridge-a10 (REGISTERED)** | `EDDF/reached`, `LTFM/unreached` | +246 / +203 / +96 | +504 / -372 / +235 | +3,003 / +3,620 / +1,261 | +756 / +232 / +264 |
| ridge-a100 | `EDDF/reached`, `LTFM/unreached` | +246 / +203 / +96 | +506 / -368 / +235 | +3,000 / +3,619 / +1,261 | +757 / +232 / +264 |
| ridge-a1000 | `EDDF/reached`, `LTFM/unreached` | +245 / +202 / +96 | +518 / -345 / +235 | +2,972 / +3,612 / +1,259 | +758 / +231 / +264 |
| gbm-imputed-300x3 | `EDDF/reached`, `LEBL/reached` | +272 / +237 / +121 | +393 / +94 / +219 | +5,002 / +4,437 / +2,248 | +741 / +436 / +53 |
| gbm-imputed-800x6 | `EDDF/reached`, `LEBL/reached` | +274 / +230 / +118 | +222 / +233 / +134 | +4,997 / +4,318 / +2,176 | +753 / +223 / +84 |

**The linear form is saturated and the nonlinearity is not.** Four orders of magnitude of ridge penalty move `EDDF/reached`'s july gain from +3,003 to +2,972 s² per row and `LTFM/unreached`'s from +756 to +758: there is no capacity question inside the ridge. A gradient-boosted head on the same columns raises `EDDF/reached` by 1.4 to 1.8× on every fold (+5,002 / +4,437 / +2,248 against +3,003 / +3,620 / +1,261) — and LOSES `LTFM/unreached`, whose december and january intervals stop clearing zero, so the population it can serve falls from 63,862 ranking rows to 19,499. **Post-selection** (§7c): the two served cells were chosen on these same three folds with the registered ridge, and picking a capacity by reading this table would select on them a second time.

### T7j the file-clock family alone (`Nf_*`, no observation anywhere in it)

`Nf_*` is built from the FILE clocks of the other aircraft in the cell, so it is defined on every row of every airport whether the harvest heard it or not. It is the only neighbourhood family LTFM can have: round 4 verified `No_n_res_cell` finite on none of LTFM's 25,185 no-clock 2026-07 rows and `Nf_n_res_cell` finite on 24,794 of them.

| fold | class rows | class `2A − B` s² [95 %] | projected, whole class | projected, served cells | `LTFM/unreached` s²/row |
|---|---:|---|---|---|---|
| july_loo | 103,514 | +333 [+134, +556] | +164 [+69, +281] | +136 [+60, +244] | +986 [+434, +1,769] |
| december | 64,850 | +221 [+20, +464] | +146 [+63, +237] | +72 [+27, +131] | +524 [+198, +950] |
| january | 84,933 | +79 [-83, +238] | +44 [-27, +126] | +49 [+10, +89] | +355 [+73, +648] |

Against arm `N` on the two cells the candidate serves, same rows, same rotation:

| cell | jul `Nf` / `N` | dec `Nf` / `N` | jan `Nf` / `N` | pooled `Nf` / `N` |
|---|---|---|---|---|
| `EDDF/reached` | +901 / +3,003 | +836 / +3,620 | -116 / +1,261 | +520 / +2,465 |
| `LTFM/unreached` | +986 / +756 | +524 / +232 | +355 / +264 | +640 / +433 |

**The file-clock family alone supports `LTFM/unreached` on all three folds, and it reads HIGHER there than the arm the candidate serves.** LTFM is the cell that carries 47,577 of the candidate's 63,862 served rows, and no observation reaches it: round 4 verified `No_n_res_cell` finite on none of its 25,185 no-clock 2026-07 rows. The `No_*` columns are therefore the fit rows' imputed means on every LTFM row, and carrying them DILUTES the correction — `Nf` reads +986 / +524 / +355 s² per row against `N`'s +756 / +232 / +264. The reverse holds at EDDF, which is 96 to 99 % reached: there the observation is the signal (+3,003 against +901 on july) and `Nf` alone loses january's interval. `Nf` alone would NOT pass the registered gate — its whole-class july projection is −303 [−2,058, +697] — so this is a fold-side reading, not a candidate. Choosing the arm per cell on these same folds is a second selection, with §7c's caveat.

### T7k step 2 under the PER-CELL FORM — `Nf_*` only on unreached cells, arm `N` on reached cells

The desk's rule of 2026-09-09, registered in `arms.SPLIT_FORM` and read once by both `arms.py` and `serve.py`: an absent observation is never a feature, and on an unreached row every `No_*` column is the fitting rows' imputed mean. Both heads are fitted on the SAME fitting rows (`--fit-rows all`); only the columns a scored row's reach class carries change. `EDDF/reached` is bit-identical under the two arms, which is the check that nothing else moved.

| fold | class rows (reached / unreached) | class `2A − B` s² | projected, whole class (ONE FLOOR, registered) | the same at the OLD two-floor reading, for the record | projected, served cells |
|---|---|---|---|---|---|
| **july_loo**, `N_reach` | 26,359 / 77,155 | +619 [+375, +883] | +333 [+211, +469] | -130 [-1,877, +880] | +278 [+185, +383] |
| **december**, `N_reach` | 21,895 / 42,955 | -318 [-798, +76] | +224 [+68, +403] | +232 [+75, +408] | +243 [+124, +403] |
| **january**, `N_reach` | 15,478 / 69,455 | +221 [+47, +399] | +141 [+62, +229] | +153 [+74, +244] | +109 [+47, +181] |
| july_loo, `N` | 103,514 | +504 [+314, +710] | +286 [+182, +407] | +741 [+423, +1,916] | +246 [+164, +337] |
| december, `N` | 64,850 | -373 [-889, +40] | +218 [+35, +448] | +225 [+44, +442] | +203 [+97, +358] |
| january, `N` | 84,933 | +235 [+76, +396] | +154 [+79, +238] | +180 [+82, +308] | +96 [+38, +159] |

**Both legs clear zero on every fold, for both arms**, and the served leg rises: +278 / +243 / +109 against +246 / +203 / +96, on the same two cells. `EDDF/reached` is a reached cell, so the rule gives it arm `N` under both forms and its numbers are identical to the digit — the whole difference is LTFM's. **This is read under the desk's one-floor rule of 2026-09-09** (§7h): while the projection priced a cell at 100 fold rows and the hold refused it at 500, the whole-class leg failed july at −130 [−1,704, +881] and one HELD cell was the whole of it — `LIRF/unreached`, 141 july fold rows extrapolated 125-fold, reading −9,175 s² per row under the file clocks and +8,765 under arm `N`, the same rows with opposite signs. The last column of the table above is what that cell was doing to the gate.

The four cells the argument turns on, per fold, `Nf`-only against arm `N` on the same rows:

| cell | july | december | january |
|---|---|---|---|
| `LTFM/unreached`, `Nf` only | +986 [+426, +1,682] on 25,171 | +524 [+208, +935] on 22,609 | +355 [+90, +650] on 21,368 |
| `LTFM/unreached`, arm `N` | +756 [+358, +1,287] on 25,171 | +232 [+44, +435] on 22,609 | +264 [+44, +499] on 21,368 |
| `LFPG/unreached`, `Nf` only | -3 [-269, +227] on 20,342 | +137 [-79, +399] on 16,203 | +10 [-681, +700] on 17,900 |
| `LFPG/unreached`, arm `N` | -162 [-374, +55] on 20,342 | +213 [-129, +754] on 16,203 | +297 [-28, +696] on 17,900 |
| `LIRF/unreached`, `Nf` only | – | +150 [-410, +828] on 2,388 | -34 [-314, +228] on 11,317 |
| `LIRF/unreached`, arm `N` | – | +666 [-587, +2,083] on 2,388 | -94 [-336, +164] on 11,317 |
| `EDDF/reached`, arm `N` | +3,003 [+1,926, +4,173] on 4,674 | +3,620 [+1,372, +6,934] on 1,561 | +1,261 [+307, +2,448] on 3,586 |

And the currency of the form, cross-fitted by fold with the hold applied on the fold side — the number `serve.py` would write:

| fold | rows the file would move | all | bulk | tail | no-NM (rows) |
|---|---:|---|---|---|---|
| july_loo | 29,844 | 258.7256 → **256.3922** | 240.7049 → **238.7927** | 2669.5 → **2627.1** | 478.9 → **474.0** (684) |
| december | 24,169 | 200.7134 → **199.2284** | 191.7625 → **190.3131** | 1547.7 → **1538.8** | 317.5 → **316.7** (381) |
| january | 24,954 | 211.5272 → **210.3767** | 204.3075 → **203.1895** | 2507.0 → **2494.5** | 461.7 → 464.1 (326) |

Board, clean **256.4643 → 256.1919**, gain **+0.2724 s** (the accepted candidate's is +0.2231); all-row 308.5977 → 308.3712, +0.2264 s (against +0.1854). December's tail and no-NM columns, which the accepted candidate makes WORSE (T8), are improvements under this form.

### T7l the gate for the v15 slot — the INCREMENT of the per-cell form over arm `N`, on the rows the file would move

`2A − B` prices a correction against the head; this prices one correction against ANOTHER on the same rows: per row `(y − p12 − d_N)² − (y − p12 − d_split)²`, positive when the per-cell form is the better of the two, with a day-clustered bootstrap over the fold's own days. The desk's rule for the slot (2026-09-09): **every fold clear of zero, or the candidate is a MAYBE and takes no slot.** Two fitting conventions are read because this lane uses both — `rotation` is step 2's registered one, `cross` is the currency's, each fold fitted on the other two.

**Read the two january lines as ONE.** `arms.ROTATION['january']` is `['july_loo', 'december']`, which IS january's cross set, so january's two rows are the same estimate on the same rows, drawn twice off one bootstrap stream (`work/increment.py` makes one generator and passes it through every call in order). Where they differ, that is the draw and not the convention — see T7m, where the same identity decides a verdict.

| convention | fold | served rows | increment s² [95 %] | clear? | `LTFM/unreached` | `EDDF/reached` |
|---|---|---:|---|---|---|---|
| cross | july_loo | 29,844 | +226 [+55, +415] | **yes** | +268 [+68, +492] | +0 [+0, +0] |
| cross | december | 24,169 | +222 [+62, +463] | **yes** | +237 [+64, +478] | +0 [+0, +0] |
| cross | january | 24,954 | +78 [-77, +234] | no | +91 [-95, +273] | +0 [+0, +0] |
| cross | *the three pooled (diagnostic)* | 78,967 | +178 [+77, +294] | **yes** | – | – |
| rotation | july_loo | 29,844 | +193 [-23, +440] | no | +229 [-26, +513] | +0 [+0, +0] |
| rotation | december | 24,169 | +271 [+63, +570] | **yes** | +290 [+57, +606] | +0 [+0, +0] |
| rotation | january | 24,954 | +78 [-78, +240] | no | +91 [-95, +269] | +0 [+0, +0] |
| rotation | *the three pooled (diagnostic)* | 78,967 | +181 [+55, +313] | **yes** | – | – |

**Verdict: MAYBE: the increment crosses zero on at least one fold**

`EDDF/reached` is exactly zero because the rule gives a reached cell arm `N` under both forms — the file the per-cell form would write differs from the accepted candidate on LTFM's rows and nowhere else, which the dry run confirms on the bytes. So the gate is LTFM's reading alone, and **january is the fold that does not clear**: +91 s² per row on 21,368 rows with a half-width of ±183, against july's +268 [+68, +492] and december's +237 [+64, +478]. The three folds pooled are clear, and pooling is not the rule.

### T7m the cheap revival — the `Nf` head at LTFM SHRUNK toward arm `N` at a weight fixed before the read

§7h names a smaller variance of the DIFFERENCE as one of the two ways the v15 slot revives. This is that: on an unreached cell the correction is `0.5·d(Nf) + 0.5·d(N)` instead of `d(Nf)`, the weight **0.5 registered in `arms.FORMS` before the measurement ran and never searched**. A reached cell is arm `N` under every form, so this moves LTFM's rows and nothing else. Both legs of step 2, at the one-floor rule:

| fold | arm | class `2A − B` s² | projected, whole class | projected, served cells |
|---|---|---|---|---|
| july_loo | **`N_reach_half`** | +589 [+369, +820] | +315 [+204, +438] | +265 [+180, +359] |
| july_loo | `N_reach` | +619 [+384, +879] | +333 [+209, +478] | +278 [+183, +387] |
| july_loo | `N` | +504 [+315, +717] | +286 [+191, +402] | +246 [+169, +344] |
| december | **`N_reach_half`** | -305 [-788, +99] | +242 [+75, +438] | +229 [+115, +385] |
| december | `N_reach` | -318 [-816, +75] | +224 [+71, +402] | +243 [+124, +407] |
| december | `N` | -373 [-856, +37] | +218 [+24, +445] | +203 [+96, +361] |
| january | **`N_reach_half`** | +271 [+111, +421] | +165 [+96, +242] | +109 [+52, +176] |
| january | `N_reach` | +221 [+48, +398] | +141 [+61, +233] | +109 [+48, +178] |
| january | `N` | +235 [+68, +404] | +154 [+81, +233] | +96 [+36, +161] |

**Shrinking does not buy a bigger gain, and it was not asked to.** On the served cells the shrunk form sits BETWEEN arm `N` and the replacement form on july (+265 against +278 and +246) and on december (+229 against +243 and +203), and equals the replacement form on january (+108.9 against +108.6, and arm `N`'s +96). On the whole class it is the best of the three on december and january and the worst but one on july. Every leg of every arm clears zero on every fold. What the weight is for is the VARIANCE of the difference, which is what the slot gate reads — the INCREMENT over arm `N` on the rows the file would move (T7l's statistic, the same rows):

| convention | fold | increment s² [95 %], 2,000 draws | clear? | the same at 20,000 draws | worst of ten seeds | `LTFM/unreached` alone |
|---|---|---|---|---|---:|---|
| rotation | july_loo | +114 [+4.8, +239] | **yes** | [+9.6, +237] | +4.8 | +135 [+6.5, +277] on 25,170 |
| rotation | december | +173 [+62.5, +333] | **yes** | [+60.2, +334] | +57.1 | +184 [+61.6, +355] on 22,608 |
| rotation | january | +80 [+0.5, +162] | **yes** | [+1.2, +161] | -3.5 | +93 [-0.1, +186] on 21,368 |
| cross | july_loo | +120 [+34.3, +215] | **yes** | [+35.5, +221] | +32.2 | +143 [+42.1, +256] on 25,170 |
| cross | december | +129 [+45.5, +255] | **yes** | [+46.7, +253] | +44.8 | +138 [+48.2, +265] on 22,608 |
| cross | january | +80 [-0.1, +161] | no | [+1.2, +161] | -3.5 | +93 [+1.3, +186] on 21,368 |

**Verdict: a second MAYBE, and it is january again.** Two folds are clear by a wide margin under both conventions. January is not: the registered rotation reads **+80 [+0.5, +162]** and the cross convention **+80 [−0.1, +161]** — and by the identity in T7l those are ONE estimate drawn twice, so the candidate's fate was resting on which stretch of a bootstrap stream it landed on. The decision rule was fixed before that second read (`work/increment_mc.py`): clear on every fold, under both conventions, at 20,000 draws AND at the worst of ten 2,000-draw seeds. At 20,000 draws january's bound is **+1.2**; across ten seeds its worst is **−3.5**. The cell that actually moves reads january **+93 [−0.1, +186] on 21,368 rows** — under the same draw whose pooled line reads +0.5. A gain of +80 s² per row is very likely real and is not ESTABLISHED on that fold, which is what the gate asks.

**The mechanism worked; it was not enough.** Against the replacement form (`N_reach`, the accepted MAYBE of §7h) january's point estimate hardly moves, +77.7 → +80.0 s² per row, while the half-width of its interval falls from **±155 to ±81** — the variance of the difference down by about three quarters, precisely what §7h said shrinking would do. It carries the lower bound from −74 to +1.2 and stops there. The same refined read on `N_reach` is not close: january **+78 [-74, +235]** at 20,000 draws. A weight search would very likely find one that clears — which is why there was none: the weight was registered before the read, and a bound reached by searching weights on the same three folds is not a bound. Records: `analysis/increment-shrunk.json`, `analysis/increment-mc.json`, `analysis/increment-mc-N_reach.json`.

### T7n where the shrunk head's gain — and its shortfall — sit inside a fold

The desk's question (2026-09-09): the second MAYBE is january's, but is the shortfall january's alone or does the shrunk head also lose the TAIL or the no-NM rows on the folds that clear? `analysis/shrunk-currency.json` shows two RMSE figures moving the wrong way — december's tail 1,547.67 → 1,548.83 and january's no-NM 461.74 → 464.03 — and an RMSE on a few hundred rows says nothing on its own. So each subgroup of the SERVED rows is priced the way the gate prices a fold: per row, day-clustered, with an interval, at 20,000 draws re-seeded per call. `no-NM` overlaps bulk and tail. **The last column is the subgroup's PART of the fold's whole increment** in the same units (its mean × its share of the served rows, so the five lines of a fold add to that fold's figure twice over — bulk + tail, and no-NM + with-NM), which is what says where the gate is really resting. The RMSE column is the rotation's; the currency in `shrunk-currency.json` is cross-fitted, which is why december's tail reads 1,538 here and 1,549 there — 36 rows, and the same 36.

| fold | subgroup | rows | RMSE at the ROTATION: head → arm `N` → shrunk | gain over the head s²/row [95 %] | increment over `N`, rotation | the same, cross | the subgroup's part of the fold's increment, s²/row |
|---|---|---:|---|---|---|---|---:|
| july_loo | the served rows | 29,844 | 258.73 → 256.58 → 256.36 | +1,221 [+774, +1,783] | +113.5 [+9.6, +237] | +120.4 [+35.5, +221] | +113.5 |
| july_loo | bulk `y ≤ 3600` | 29,806 | 240.70 → 238.93 → 238.75 | +937 [+666, +1,209] | +83.4 [-6.9, +192] | +94.9 [+21.8, +183] | +83.3 |
| july_loo | **tail `y > 3600`** | 38 | 2,669.52 → 2,631.74 → 2,627.22 | +224,041 [+66,754, +306,511] | +23,774.7 [-213.3, +36,991] | +20,152.6 [+2,339.6, +30,246] | +30.3 |
| july_loo | **no-NM** | 684 | 478.93 → 474.85 → 474.37 | +4,348 [+1,295, +9,276] | +458.5 [-36.2, +1,275] | +369.3 [-101.3, +1,155] | +10.5 |
| july_loo | with an NM record | 29,160 | 251.25 → 249.17 → 248.96 | +1,147 [+740, +1,632] | +105.4 [+6.1, +222] | +114.6 [+34.7, +208] | +103.0 |
| december | the served rows | 24,169 | 200.71 → 199.58 → 199.15 | +627 [+340, +948] | +172.5 [+60.2, +334] | +129.0 [+46.7, +254] | +172.5 |
| december | bulk `y ≤ 3600` | 24,133 | 191.76 → 190.52 → 190.23 | +585 [+326, +865] | +109.5 [+49.4, +172] | +81.7 [+40.7, +131] | +109.3 |
| december | **tail `y > 3600`** | 36 | 1,547.67 → 1,552.08 → 1,538.35 | +28,754 [+3,522, +201,900] | +42,428.0 [-4,244.1, +47,454] | +31,868.2 [+163.0, +35,283] | +63.2 |
| december | **no-NM** | 381 | 317.54 → 317.57 → 316.40 | +723 [-286, +1,700] | +745.7 [-131.2, +2,229] | +451.5 [-219.4, +1,590] | +11.8 |
| december | with an NM record | 23,788 | 198.28 → 197.11 → 196.70 | +625 [+330, +959] | +163.3 [+60.0, +305] | +123.9 [+48.0, +232] | +160.7 |
| january | the served rows | 24,954 | 211.53 → 210.56 → 210.37 | +488 [+240, +771] | +80.0 [+1.2, +161] | +80.0 [+1.2, +161] | +80.0 |
| january | bulk `y ≤ 3600` | 24,942 | 204.31 → 203.37 → 203.18 | +460 [+224, +721] | +77.9 [+0.3, +158] | +77.9 [+0.3, +158] | +77.9 |
| january | **tail `y > 3600`** | 12 | 2,506.96 → 2,496.24 → 2,495.35 | +58,086 [-4,675, +95,114] | +4,456.5 [-8,923.8, +9,413] | +4,456.5 [-8,923.8, +9,413] | +2.1 |
| january | **no-NM** | 326 | 461.74 → 463.99 → 464.03 | -2,112 [-9,545, +3,769] | -33.2 [-461.8, +341] | -33.2 [-461.8, +341] | -0.4 |
| january | with an NM record | 24,628 | 206.19 → 205.12 → 204.92 | +522 [+278, +796] | +81.5 [+3.3, +162] | +81.5 [+3.3, +162] | +80.4 |

**The MAYBE is january's alone, and inside january it is the BULK's.** January's whole shortfall is 24,942 bulk rows reading +77.9 [+0.3, +157.6] — the same knife edge as the fold — while its tail contributes +2.1 s² and its no-NM rows −0.4, neither of them established in any direction. **Neither RMSE that moved the wrong way is a loss to the shrunk head.** December's tail is **36 rows**: against the head its gain is −3,587 [−35,753, +149,346] under the cross convention and +28,754 [+3,522, +201,900] under the rotation, so the sign is the draw and not the data — and against arm `N`, which is what v14 ships on those rows today, the shrunk head is better on both (RMSE 1,538 / 1,549 against `N`'s 1,552 / 1,559). January's no-NM rows are **326**, reading −2,113 [−9,545, +3,769] against the head, and arm `N` is equally bad there (RMSE 463.99 against 464.03), so this is a property of the correction the lane already ships, not something shrinking introduced. On every fold and both conventions the shrunk head beats arm `N` in the tail.

**What the table does say about december.** Its increment leans on those 36 tail rows for **+63 of +173 s²** (rotation) and **+48 of +129** (cross) — 37 % of a fold's verdict from 0.1 % of its rows. The fold does not depend on them: december's BULK alone clears under both conventions, +109.5 [+49.4, +171.7] and +81.7 [+40.7, +131.4]. July is the same shape and smaller — +30 and +26 of its +114 and +120 — and its bulk clears under the cross convention (+94.9 [+21.8, +183.1]) but not under the rotation (+83.4 [−6.9, +192.0]). Record: `analysis/tail-nonm.json`, `work/tailnonm.py`.

### T7o the harvest's receipt correction (2026-09-09) — the bias, and what it moves in this lane

`exp-045-harvest`'s `census.rate()` aggregated `min(t_first)` / `max(t_last)` per `(airport, callsign)` over the whole month before testing the take-off window, so a flight number flown daily counted as received on every day of the month (commit `f8352a4`). **The bias is large and it is not the same in every month**, which is what would matter if it scaled a feature:

| month | fold or ranking | departures | rate as committed | corrected | ratio | points the old rate was too high |
|---|---|---:|---:|---:|---:|---:|
| 2025-01 | **fold** | 153,706 | 0.5413 | 0.4800 | ×0.8868 | 6.1 |
| 2025-07 | **fold** | 190,713 | 0.6095 | 0.4961 | ×0.8140 | 11.3 |
| 2025-12 | **fold** | 165,677 | 0.7558 | 0.6521 | ×0.8628 | 10.4 |
| 2026-01 | ranking | 152,719 | 0.7227 | 0.6269 | ×0.8674 | 9.6 |
| 2026-07 | ranking | 192,122 | 0.6487 | 0.5609 | ×0.8647 | 8.8 |

The three fold months average ×0.8545 and the two ranking months ×0.866, so the answer to *does the bias differ between the 2025 folds and the 2026 months* is **yes, by about a point of ratio — and by seven points WITHIN the folds** (2025-07 ×0.814 against 2025-01 ×0.887), which is the larger spread of the two.

**But it never entered this lane's features.** The `_sc` columns do not divide by that rate. `block.receipt_rates()` measures its own denominator on the block's OWN matched matrix — the share of an airport-**day**'s departures whose ground run the matcher identified (`obs_has_clock`), over the airport-days the summaries delivered — and in this lane `receipt.json` is read at exactly three lines, all inside `outage_report()`, which writes warnings and nothing else. The two quantities are not close: on 2025-07 EGLL reads **0.0577** here against the harvest's corrected **0.1506**, and LFPG **0.0182** against **0.0476**, because a matched 250 m clock is a stricter test than hearing the callsign in the window.

Measured, not argued — the five blocks rebuilt with the corrected artifact in place, against the published bytes copied aside first:

| month | rows | columns compared | `_sc` columns | cells that moved | NaN-pattern changes | row order |
|---|---:|---:|---:|---:|---:|---|
| 2025-01 | 153,706 | 186 | 28 | **0** | 0 | identical |
| 2025-07 | 190,713 | 186 | 28 | **0** | 0 | identical |
| 2025-12 | 165,677 | 186 | 28 | **0** | 0 | identical |
| 2026-01 | 152,719 | 186 | 28 | **0** | 0 | identical |
| 2026-07 | 192,122 | 186 | 28 | **0** | 0 | identical |

**Every column of every month is identical, value for value, in the same row order** (`analysis/block-rebuild-diff.json` compares all 186 columns, not only the 28 `_sc` ones). The parquet SHA-256 changes on a rebuild because the container is not written deterministically, so the published bytes were restored and `analysis/interface.json` still pins the shas surfacefit verified.

Then arm `N` re-priced at one floor with the cell holds, on the corrected artifact (`analysis/arms-step2-recheck.json`):

| fold | leg | registered record | re-priced | point difference |
|---|---|---|---|---|
| july_loo | whole class | +286.43 [+178, +401] | +286.43 [+184, +399] | **+0.000000** |
| july_loo | served cells | +246.14 [+165, +338] | +246.14 [+165, +337] | **+0.000000** |
| december | whole class | +217.50 [+31, +454] | +217.50 [+30, +440] | **+0.000000** |
| december | served cells | +203.05 [+94, +360] | +203.05 [+94, +354] | **+0.000000** |
| january | whole class | +153.56 [+79, +233] | +153.56 [+78, +232] | **+0.000000** |
| january | served cells | +96.05 [+41, +158] | +96.05 [+40, +162] | **+0.000000** |

**Every point estimate is identical to the last decimal**, the same two cells of nineteen are served, and every interval stays clear of zero on the same side; the intervals wobble only because a one-arm run draws a different stretch of the bootstrap stream than the six-arm run that wrote the record (ledger CORRECTION 7). **The gate passes and the served-cells gain has not moved**, so under the desk's rule of 2026-09-09 v14's reviewed bytes go up as gated.

### T7f the all-row reading, for the record — one head over both classes

`arms-step2-all.json` — head v12, matcher `ba41935e…`, priced on **all** rows, fitted on **all** rows, missingness indicator off, fit floor on receipt 0.0. The projection carries 344,841 of the ranking file's 344,841 rows (100.0 %). **The gate is the s² interval, not the board column**: that column is APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the smaller of the two bases, so it is the larger of the two possible second-counts; the same drop at the ALL-ROW board (308.5977) is about 0.83× as many seconds. `code/serve.py` measures the served file's own board directly rather than converting. The projection is read per airport-MONTH and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is priced separately on rows with and without a heard neighbour and each half is applied to that airport-month's own reached share. The last two columns are leave-one-out readings off the SAME draws — the worst single airport, and the served-cells-only projection that gates the file written.

| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | worst airport dropped | served cells only |
|---|---|---:|---:|---:|---:|---|---|
| `Nf` | july_loo | 190,713 | 165,677 | +198 [+12, +424] | +202 [+15, +420] | – | – |
| `Nf` | december | 165,677 | 190,713 | +41 [-65, +148] | +42 [-69, +151] | – | – |
| `No` | july_loo | 190,713 | 165,677 | +737 [+520, +1,024] | +723 [+515, +992] | – | – |
| `No` | december | 165,677 | 190,713 | -193 [-553, +127] | -156 [-492, +175] | – | – |
| `N` | july_loo | 190,713 | 165,677 | +861 [+621, +1,172] | +846 [+601, +1,154] | – | – |
| `N` | december | 165,677 | 190,713 | -141 [-466, +162] | -112 [-405, +201] | – | – |
| `own` | july_loo | 190,713 | 165,677 | +645 [-2,408, +2,892] | +610 [-2,446, +2,910] | – | – |
| `own` | december | 165,677 | 190,713 | +1,135 [+872, +1,384] | +1,140 [+886, +1,379] | – | – |

### T8 the four-way RMSE of the served correction (arm `N`)

`assignments/COMMON.md`'s all / bulk (`y ≤ 3600`) / tail / no-NM split. **The first block is the rows the FILE moves** (FALLBACK round 4, item 2): the cell hold serves 3 cells of 19 under the lit-fold rule, so the class-wide reading below is a different population from the candidate's. January's all-row and tail columns are dominated by its `noac` rows, which this lane never moves.

| fold | rows the file moves | all | bulk | tail | no-NM (rows) |
|---|---:|---|---|---|---|
| july_loo | 39,210 | 241.8708 → **239.6905** | 227.2557 → **225.3291** | 2669.5 → **2634.7** | 447.5 → **443.7** (814) |
| december | 24,979 | 199.7817 → **198.7732** | 191.0844 → **189.8935** | 1547.7 → 1559.1 | 316.9 → 317.0 (386) |
| january | 24,954 | 211.5272 → **210.5613** | 204.3075 → **203.3702** | 2507.0 → **2496.2** | 461.7 → 464.0 (326) |

And the same split over the WHOLE fold, the `d` zero on every row the file does not move:

| fold | fold rows | all | bulk | tail | no-NM (rows) |
|---|---:|---|---|---|---|
| july_loo | 190,713 | 283.5657 → **283.1848** | 246.7302 → **246.3657** | 2395.8 → **2393.5** | 1118.5 → **1118.1** (3,934) |
| december | 165,677 | 204.1985 → **204.0501** | 191.8273 → **191.6489** | 1773.9 → 1775.3 | 655.2 → 655.2 (1,643) |
| january | 153,706 | 337.4611 → **337.3630** | 195.5589 → **195.4000** | 5984.1 → **5984.0** | 2835.5 → 2835.6 (1,432) |

For the record, the same split on the WHOLE NO-CLOCK CLASS — the population step 2 gates, which the hold no longer serves in full:

| fold | class rows | all | bulk | tail | no-NM (rows) |
|---|---:|---|---|---|---|
| july_loo | 103,513 | 256.3998 → **255.4153** | 229.8999 → **229.0744** | 2107.3 → **2097.1** | 691.3 → **690.6** (2,721) |
| december | 64,837 | 233.8299 → 234.6232 | 213.5872 → 214.5530 | 1921.7 → **1917.4** | 653.7 → 653.9 (1,175) |
| january | 84,933 | 425.2077 → **424.9315** | 213.0681 → **212.6998** | 7059.0 → **7057.0** | 3201.9 → **3201.2** (1,108) |

And the board's own currency, the clean RMSE beside the all-row one:

| fold | rows the file moves | RMSE | clean RMSE |
|---|---:|---|---|
| july_loo | 39,210 | 283.5657 → **283.1848** | 283.5657 → **283.1848** |
| december | 24,979 | 204.1985 → **204.0501** | 204.1985 → **204.0501** |
| january | 24,954 | 337.4611 → **337.3630** | 217.6288 → **217.4767** |

Board at the ranking shares 0.5571 / 0.4429: clean **256.4643 → 256.1726**, gain **+0.2918 s**; all-row **308.5977 → 308.3552**, gain **+0.2425 s**. December is not a ranking-share fold and is carried beside, not inside.

### T9 december reconciled — the projected gain against the read of the FILE THAT IS WRITTEN

The registered rule for the surface lanes is the PROJECTION (`assignments/queue.md`, the pilot's §5 method): each airport's fold gain per served row, weighted by the rows that airport-month has in the ranking file. This table puts it beside the raw december read of the same `d`, both with day-clustered intervals over december's 31 days. The `d` is the HELD one (FALLBACK round 4, item 1): the cell hold serves 3 cells (`EDDF/reached`, `LEMD/reached`, `LTFM/unreached`), so the file moves 24,979 of december's 64,786 no-clock rows. The last two rows are the same reading BEFORE the hold, which is what this section carried until round 4.

| reading | rows | `2A − B` s² | RMSE seconds |
|---|---:|---|---|
| **projected**, the served cells | – | **+245 [+125, +406]** | +0.4777 |
| projected, the whole no-clock class | – | +225 [+32, +453] | +0.4385 [+0.0626, +0.8845] |
| raw, december's SERVED rows | 24,979 | +402 [+181, +645] | +1.0085 [+0.4591, +1.6589] |
| raw, december ALL rows (the served file) | 165,677 | +61 [+31, +97] | +0.1484 [+0.0763, +0.2292] |
| raw, all rows with **EGLL removed** | 145,677 | +69 [+31, +111] | +0.1702 [+0.0799, +0.2658] |
| raw, served rows with **EGLL removed** | 24,979 | +402 [+183, +629] | +1.0085 [+0.4599, +1.5888] |
| *pre-hold*, the class's rows | 64,786 | -11 [-341, +293] | -0.0248 [-0.7423, +0.6436] |
| *pre-hold*, all rows | 165,677 | -4 [-127, +112] | -0.0107 [-0.3137, +0.2678] |

Per airport, december, EGLL first:

| airport | served rows | raw served `2A − B` | raw all-row `2A − B` | projected contribution s² | ranking weight |
|---|---:|---|---|---:|---:|
| **EGLL** | 0 | – | +0 [+0, +0] | -78.6 | 0.0189 |
| EDDF | 1,561 | +3,217 [+1,399, +5,954] | +290 [+113, +585] | +179.5 | 0.0627 |
| EDDM | 0 | – | +0 [+0, +0] | +0.0 | 0.0028 |
| EHAM | 0 | – | +0 [+0, +0] | -13.8 | 0.0076 |
| LEBL | 0 | – | +0 [+0, +0] | +15.5 | 0.0101 |
| LEMD | 810 | +1,298 [+655, +2,077] | +58 [+29, +92] | +33.2 | 0.0294 |
| LFPG | 0 | – | +0 [+0, +0] | +18.1 | 0.1109 |
| LIRF | 0 | – | +0 [+0, +0] | +32.5 | 0.0615 |
| LSZH | 0 | – | +0 [+0, +0] | +6.2 | 0.0057 |
| LTFM | 22,608 | +176 [+8, +341] | +176 [-2, +347] | +32.1 | 0.1380 |

And the instrument behind it — the share of each airport's departures the matcher clocks, month by month, from the five emitted blocks. A 2025 airport whose reach is far below the ranking months' puts rows in the no-clock class that the ranking file does not:

| airport | 2025-01 | 2025-07 | 2025-12 | 2026-01 | 2026-07 | 2026-01 − 2025-12 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 0.759 | 0.778 | 0.907 | 0.504 | 0.354 | -0.404 |
| EDDM | 0.963 | 0.365 | 0.963 | 0.962 | 0.963 | -0.001 |
| EGLL | 0.435 | 0.058 | 0.439 | 0.703 | 0.961 | +0.264 |
| EHAM | 0.964 | 0.942 | 0.938 | 0.924 | 0.937 | -0.014 |
| LEBL | 0.915 | 0.823 | 0.914 | 0.880 | 0.887 | -0.035 |
| LEMD | 0.000 | 0.220 | 0.952 | 0.943 | 0.546 | -0.009 |
| LFPG | 0.000 | 0.018 | 0.051 | 0.041 | 0.040 | -0.010 |
| LIRF | 0.000 | 0.871 | 0.478 | 0.481 | 0.001 | +0.004 |
| LSZH | 0.897 | 0.860 | 0.939 | 0.916 | 0.915 | -0.023 |
| LTFM | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |

### T5b which column carries it — every `No_` column priced on its own

| column | share removed | 95 % | 2A − B (s²) | (2A − B) / before |
|---|---:|---|---:|---:|
| `No_n_res_cell` | 68.8 % | [55.9, 80.6] | +441.5 | 20.8 % [8, 33] |
| `No_n_res_rwy_cell` | 54.9 % | [46.1, 63.3] | +464.4 | 21.8 % [14, 30] |
| `No_n_res_apt_30m` | 30.2 % | [20.8, 42.0] | +316.0 | 14.9 % [8, 22] |
| `No_n_anom_rwy_cell` | 18.2 % | [10.5, 25.8] | +359.6 | 16.9 % [11, 23] |
| `No_n_anom_cell` | 18.0 % | [11.6, 24.9] | +78.6 | 3.7 % [-3, 10] |
| `No_n_res_rwy_30m` | 12.7 % | [9.1, 16.3] | +89.5 | 4.2 % [-1, 9] |
| `No_n_res_rwy_60m` | 2.3 % | [0.4, 4.5] | -42.0 | -2.0 % [-6, 1] |
| `No_n_anom_rwy_30m` | -2.3 % | [-6.1, 1.8] | +75.3 | 3.5 % [-0, 7] |
| `No_n_res_rwy_minus_apt` | -3.1 % | [-6.1, -1.0] | -60.1 | -2.8 % [-5, -1] |
| `No_n_n_rwy_cell` | -3.9 % | [-7.2, -1.2] | -69.7 | -3.3 % [-5, -1] |
| `No_n_n_rwy_30m` | -4.1 % | [-10.4, 1.0] | -51.9 | -2.4 % [-6, 1] |
| `No_n_n_cell` | -4.4 % | [-7.4, -2.2] | -74.5 | -3.5 % [-6, -2] |

### T5c the reach of the no-clock route, on a RANKING month and with no label

279 airport-days of 2026-01-01 to 2026-01-31, 130,323 ranking departures, the served board head v12 (tail rule) as the prediction (`48dd572d61ff…`). The question is not whether a row has a clock but whether its **neighbours** do. `own clock` here is `exp-046-surfacefit`'s STRICT match (a ground run identified and a usable off-block), not the harvest's reception rate: the harvest reads 73.3 % of the ranking file received, this reads 65.2 % matched over the nine airports it delivers, and the two are different quantities. The desk's outage rule of 14:45 UTC is applied: EDDF 2026-01-27 to 01-31 are no-clock airport-days here and out of the receipt denominator.

| airport | rows | own clock | no clock | of those, reached by ≥1 heard neighbour | ≥3 | median heard neighbours in the cell |
|---|---:|---:|---:|---:|---:|---:|
| EGLL | 19,283 | 70.3 % | 5,725 | 96.5 % | 4,047 | 4 |
| EHAM | 16,232 | 92.5 % | 1,225 | 95.0 % | 896 | 4 |
| LEMD | 16,802 | 94.3 % | 962 | 92.9 % | 739 | 5 |
| EDDM | 10,953 | 96.2 % | 422 | 92.7 % | 258 | 3 |
| LSZH | 9,995 | 91.6 % | 840 | 88.9 % | 459 | 3 |
| LEBL | 12,201 | 88.0 % | 1,468 | 86.3 % | 678 | 2 |
| EDDF | 15,667 | 47.2 % | 8,279 | 64.3 % | 3,346 | 2 |
| LIRF | 11,061 | 48.1 % | 5,736 | 62.3 % | 820 | 1 |
| LFPG | 18,129 | 4.1 % | 17,390 | 11.9 % | 35 | 0 |
| **all nine** | 130,323 | 67.7 % | 42,047 | 49.8 % | 11,278 | 0 |

LTFM has no row because adsb.lol receives none of it (`exp-045-harvest` §1).

### T5d what the EDDF hold costs, the hold lifted and nothing else changed

Round 2 blocking 1: the superseded reading compared a census WITH this lane's rule against one without it, while the matcher held the days in both runs, so it varied nothing. Here the same build path runs twice and only `B46.HELD_AIRPORT_DAYS` differs.

| | EDDF rows | EDDF clocks | own-clock share | rows on held days | clocks on held days |
|---|---:|---:|---:|---:|---:|
| hold in force | 15,667 | 7,387 | 47.1 % | 2,517 | 0 |
| hold lifted | 15,667 | 7,978 | 50.9 % | 2,517 | 591 |

The hold moves **2,517** rows into the no-clock class and discards **591** clocks, taking EDDF's own-clock share from 50.9 % to 47.1 %. It is not free; it is paid so that a degraded feeder is never read as a property of EDDF's traffic.

| day | rows | clocks held | clocks lifted |
|---|---:|---:|---:|
| 2026-01-27 | 462 | 0 | 271 |
| 2026-01-28 | 498 | 0 | 235 |
| 2026-01-29 | 505 | 0 | 35 |
| 2026-01-30 | 548 | 0 | 23 |
| 2026-01-31 | 504 | 0 | 27 |

### T6 the interface `exp-046-surfacefit` can take as an input

| month | path | rows | columns | MB | SHA-256 | prediction the `*_n_res_*` columns are relative to |
|---|---|---:|---:|---:|---|---|
| 2025-01 | `artifacts/exp-047/block/2025-01.parquet` | 153,706 | 186 | 39.4 | `173e49efceb76ad5…` | fold-side v11 of january (`c78998a0e59d…`) |
| 2025-07 | `artifacts/exp-047/block/2025-07.parquet` | 190,713 | 186 | 52.4 | `f16622569bd0df76…` | fold-side v11 of july_loo (`7c07be85ea6d…`) |
| 2025-12 | `artifacts/exp-047/block/2025-12.parquet` | 165,677 | 186 | 48.8 | `5da0a4ee0706baa9…` | fold-side v11 of december (`8dc4073e43a2…`) |
| 2026-01 | `artifacts/exp-047/block/2026-01.parquet` | 152,719 | 186 | 44.6 | `bc09ae757114e524…` | v12 (tail rule), the SERVED head (`48dd572d61ff…`) |
| 2026-07 | `artifacts/exp-047/block/2026-07.parquet` | 192,122 | 186 | 54.3 | `10b32c809841af5c…` | v12 (tail rule), the SERVED head (`48dd572d61ff…`) |

Reach classes per month — `exp-046-surfacefit`'s own `reach_class`, from the one matcher both lanes import, at `6fdf39519eb0…`:

| month | `strict` | `wide_window` | `ambiguous_strict` | `matched_no_clock` | `no_takeoff` | `outage` | `unmatched` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-01 | 67,936 | 395 | 341 | 4,315 | 101 | 0 | 80,618 |
| 2025-07 | 85,668 | 923 | 381 | 6,637 | 228 | 0 | 96,876 |
| 2025-12 | 99,720 | 465 | 514 | 6,206 | 141 | 0 | 58,631 |
| 2026-01 | 87,237 | 810 | 468 | 6,063 | 270 | 1,557 | 56,314 |
| 2026-07 | 100,418 | 369 | 743 | 5,472 | 344 | 0 | 84,776 |

Column prefixes: `Fto_` the file's clocks at the take-off instant, `Fob_` at the off-block instant, `Oob_` the observed clocks, `Hob_` the hybrid, `Nf_` / `No_` the neighbourhood on the file's and the observed clock, `*_sc` a count divided by the airport's receipt rate, `route_*` and `stand_dist_km` the stand-to-runway geometry, `own_obs_taxi` exp-046's own feature carried only so a consumer can control for it, and `obs_matched` / `obs_has_clock` / `reach_class` the identity of the observation. `state.py`'s docstring defines every state name.

---

### 7ac. The combined arm — arm `N` + the cell regulation signature + the network term. FAIL

The desk, 2026-09-09, after §7ab: *"Keep pushing on the two you named, as ONE arm: arm N
plus the cell regulation signature (share of the airport × 10-min cell with AOBT−EOBT >
15 min, with the within-operator read beside it) plus the network term (same-hour
leave-one-airport-out mean residual, built on FOLD months only, never on a served month),
fitted the way arm N is, cross-fitted on the three re-pinned folds under the ordinary
gate, priced per served cell and on LTFM/unreached separately."*

**One reading has to be stated before the table.** The network term as §7ab measured it
is the leave-one-airport-out mean of `r12 = y_true − p12`. That is a **label**, and a
served month has none: the same-hour rows at the other nine airports of a ranking month
are themselves ranking rows. So the desk's own guard — *built on FOLD months only, never
on a served month* — is not a precaution here, it is a description of a column that
**cannot exist at serving time**, and §7ab's **+0.0683 s** stands as an information bound
on the network channel, never as a feature. What is fitted below is the **servable read of
the same channel**: the leave-one-airport-out same-hour mean of the **file-clock taxi
anomaly** and of the **slip anomaly**, both against the airport's own hour-of-day median,
self-airport excluded. `MVT_TIME_UTC_mvt`, `AOBT_3_flt` and `EOBT_1_flt` are all unmasked
in the ranking file, so the whole arm is servable. The reading rejected: fitting the label
network on the folds and calling the fold numbers the arm's value, which would price a
column no serve could ever write.

**What was built** (`code/block.py::regnet_block`, behind the new source tag `regnet`, so
the 186-column interface `exp-046-surfacefit` consumes is byte-identical — checked):

* `Xr_*` — the cell's regulation signature. The leave-one-out share of the airport ×
  10-minute cell whose `AOBT_3_flt − EOBT_1_flt` is past 15 min, its leave-one-out mean
  slip, **the within-operator share beside it** (the same share over the row's own
  airport × day × operator, §7ab's kill test as a column rather than a re-read), the
  airport-day share under it, and the two contrasts `Xr_cell_minus_op` and
  `Xr_cell_minus_day`. No row reads its own slip; the base head already has that
  (`aobt_eobt`).
* `Xn_*` — the network state. The same UTC hour at the **other nine** airports, on the
  file-clock taxi anomaly and on the slip anomaly, with each airport's own same-hour mean
  (self-excluded) beside it and the contrast `Xn_net_minus_own_*`.

Nineteen columns, registered as `"N_regnet": ("Nf_", "No_", "Xr_", "Xn_")` in
`explain.ARMS`. Fitted exactly the way arm `N` is: ridge, head **v12**, `--rows unmatched
--fit-rows all`, `arms.ROTATION`, rule `d4c59868…`, matcher `122c7d84`, on the three
re-pinned folds. **Arm `N` reproduces its handshake record to the last digit in the same
run** (+304.54 / +247.09 / +154.54 projected s², 24 features) — the new families move
nothing that was already priced.

**The fold table** (`analysis/arms-regnet.json`, `analysis/arms-regnet-reach.json`;
projected to the ranking months at the reach grain, pooled at `SHARES`):

| arm | july_loo s² | december s² | january s² | board s, jul / dec / jan | pooled board s |
|---|---:|---:|---:|---|---:|
| `N` (what the lane serves) | +304.54 [+198, +419] | +247.09 [+53, +496] | +154.54 [+73, +246] | +0.5944 / +0.4822 / +0.3015 | **+0.4647** |
| `regnet` alone | +32.53 [−2, +71] | −20.56 [−173, +161] | −36.24 [−126, +59] | +0.0634 / −0.0401 / −0.0706 | −0.0040 |
| `N_reg` | +298.53 [+204, +410] | +233.47 [+27, +474] | +147.18 [+52, +251] | +0.5827 / +0.4556 / +0.2871 | +0.4518 |
| `N_net` | +249.54 [+150, +355] | +190.94 [−22, +410] | +144.77 [+48, +256] | +0.4870 / +0.3725 / +0.2824 | +0.3964 |
| **`N_regnet`** | +260.80 [+161, +366] | +187.37 [−25, +438] | +124.53 [+7, +242] | +0.5090 / +0.3656 / +0.2429 | **+0.3912** |

**The increment over arm `N`** — paired, both `d` built in ONE pass from ONE set of blocks
with the same rotation and the same seed, so the difference is the two families and
nothing else (`work/regnet_increment.py`, `analysis/regnet-increment.json`):

| fold | class rows | increment s²/class row | increment on the SERVED cells s² | projected s² | projected board s |
|---|---:|---|---|---|---:|
| july_loo | 102,343 | **−83.47 [−175.4, −1.1]** | −27.25 [−53.7, −4.8] | −43.74 [−81.4, −6.7] | −0.0853 |
| december | 63,976 | −285.52 [−656.4, +79.7] | −54.60 [−133.7, +44.6] | −59.72 [−202.5, +81.6] | −0.1164 |
| january | 84,202 | −66.19 [−185.0, +67.7] | −40.41 [−85.6, +4.5] | −30.01 [−84.9, +26.1] | −0.0585 |
| **pooled** | | −75.8 [−148.7, +0.3] | **−33.1 [−58.9, −9.3]** | | **−0.0734** |

**Face: −0.0881 s at ×1.2, −0.1101 s at ×1.5.** The ordinary gate fails on both clauses:
the pooled increment is negative, every fold declines, and july_loo's increment interval
is **clear below zero**. RMS byte move on the class 12.7 / 19.1 / 11.0 s (jul / dec / jan). This is a **FAIL**, not a
MAYBE.

**Why it fails, per served cell** — the increment, and then the arms' own cell reads:

| cell | jul rows | jul increment | dec rows | dec increment | jan rows | jan increment |
|---|---:|---|---:|---|---:|---|
| `EDDF/reached` | 4,065 | +50.2 [−64, +156] | 1,415 | +109.5 [−85, +325] | 3,159 | **+241.8 [+117, +409]** |
| `LEMD/reached` | 9,373 | +140.0 [−56, +361] | 750 | −10.4 [−423, +403] | 0 | unpriced |
| `LTFM/unreached` | 25,171 | **−242.0 [−428, −84]** | 22,609 | −434.5 [−958, +246] | 21,368 | **−382.7 [−697, −58]** |

The families are **neutral to positive where an observation already exists and destroy the
gain in the one cell they were built for**. `LTFM/unreached` carries arm `N` at +733 /
+221 / +250 s² per row and `N_regnet` at +491 / −213 / −133; `regnet` alone reads +145 /
**−416** / **−304** there. The mechanism is not subtle: at an unreached cell every `No_*`
column is the fit rows' mean, so the head has the least real signal to protect, and
nineteen columns whose coefficients are estimated on the other nine airports of another
month are spent as variance. §7ab's upper bounds were in-sample ρ²·var(r12) with no
cross-fitting; out of fold at LTFM the regulation channel is **negative**, not merely
small. That is the answer to §7ab's own kill test, and it kills it.

**The repair attempted before writing the FAIL.** The per-cell reads say exactly where the
families live, so the lane's registered device for that situation (§7g's per-class form)
was tried: `N_reg_reach` and `N_regnet_reach` give the new families **only the reached
cells** and leave every unreached row on arm `N`
(`analysis/arms-regnet-reach.json`, `analysis/regnet-increment-reach.json`).

| form | jul served-cells increment | dec | jan | pooled served s² | projected board s | face ×1.2 / ×1.5 |
|---|---|---|---|---:|---:|---|
| `N_reg_reach` | +8.04 [−0.8, +16.0] | +4.04 [−6.8, +16.3] | +14.05 [+3.6, +27.6] | +10.70 [+3.8, +17.9] | +0.0026 | **+0.0031 / +0.0039** |
| `N_regnet_reach` | +6.13 | +5.34 | +12.39 | — | −0.0185 | −0.0222 / −0.0278 |

`N_reg_reach` is the only variant that ends positive. It passes the letter of the gate on
the served cells (pooled positive, no fold clear below zero) and its january leg is clear
of zero at +14.05 [+3.6, +27.6], carried by `EDDF/reached` at +274 [+70, +539]. It is
still **not a candidate**, for three reasons written together: its face is **+0.0031 s at
×1.2**, 3 % of the desk's 0.1 s bar; its pooled class interval crosses zero and july_loo
declines; and the form was chosen **after** reading the per-cell table on these same three
folds, which makes it a **THIRD selection on them** — the same objection §7g records
against the split form, at a thirtieth of the size.

**MAYBE, named as the desk asked.** The fold is **january**, the cell is `EDDF/reached`,
the form is `N_reg_reach`. **Revival condition:** a fourth fold, selected without seeing
this table, on which `EDDF/reached` repeats a positive increment clear of zero **and** the
pooled face reaches 0.1 s at ×1.2 — that is a 30× increase, so a repeated sign alone does
not revive it. Everything else here is closed at this design: the network channel does not
survive a cross-fit in either form (label-free costs −0.068 s pooled; the label form cannot
be written at all), and the regulation channel is positive only where the lane already has
an observation.

**Files.** `code/block.py::regnet_block` (families `Xr_`, `Xn_`, source tag `regnet`,
`EOBT_1_flt` added to `FRAME_COLS`), `code/explain.py` (`reg`, `net`, `regnet`, `N_reg`,
`N_net`, `N_regnet`), `code/arms.py` (`FORMS` gains the two diagnostic per-cell forms),
`work/regnet_increment.py`, `analysis/arms-regnet.json`,
`analysis/arms-regnet-reach.json`, `analysis/regnet-increment.json`,
`analysis/regnet-increment-reach.json`, `logs/regnet*.log`. Nothing is served.

---

### 7ad. The stand ordering constraint — a kill test. DEAD

The desk, 2026-09-09, from the external research read: the base head reads
`stand_next_arr_inblock_dt` as a **feature**; nobody imposes the **hard ordering** it
implies — at an exclusive stand the focal departure's off-block must precede the next
occupant's in-block. On the three fold months, on this lane's no-clock rows: (a) the label
violation rate per airport, (b) the `p12` violation rate, (c) the s²/row change from
clipping `p12` at margins 0 / 60 / 120 s, per airport and on `LTFM/unreached`. The desk's
own rule: *"If (a) is a few percent and (b) is well above it, that is a label-free clip for
a rider; if (a) is high the stand is not exclusive and the constraint is dead."*

**Both obvious anchors are vacuous, and that was measured rather than argued**
(`analysis/stand-clip-vacuous.json`, `logs/standclip-v1.log`). The base head's own
`stand_next_arr_inblock_dt` is the first same-stand arrival **at or after the take-off**
(`exp-018-fresh/code/features.py:264`), so a violation would need an off-block after an
instant that is itself at or after the take-off — a negative taxi: **79,804 / 49,333 /
63,222** no-clock rows in scope and **0** violations on all three folds. "The first arrival
strictly after the last one at or before take-off" fails the same way, because the last
arrival before take-off absorbs any intruder: **79,534 / 49,138 / 62,874** rows, **0**
violations, and **0** rows where that arrival even falls before the take-off.

**The reading with content.** The constraint bites in exactly one configuration: an
arrival takes the stand **before our take-off**, which it can only do after we push back.
Let `Q` be the last same-stand arrival in-block at or before the take-off (the same
construction as the base head's `stand_prev_arr_inblock_dt`, same `airport|stand` key) and
`dt = take-off − Q`. A **small** `dt` says `Q` cannot be the row's own inbound — no
aircraft arrives, turns round, pushes back and taxis out inside `TURN_MIN` — so `Q` is the
next occupant and the ordering gives a label-free **lower** bound, `taxi ≥ dt`. It is the
mirror of `stand_prev_arr_inblock_dt`, which the same file calls a *"hard-ish upper bound
on the taxi time"*. `TURN_MIN` is **swept**, not chosen, because the whole reading rests on
it.

| `TURN_MIN` | july_loo n / (a) / (b) | december n / (a) / (b) | january n / (a) / (b) |
|---|---|---|---|
| 10 min | 7,788 / 0.17 % / 0.12 % | 4,085 / 0.24 % / 0.10 % | 4,165 / 0.34 % / 0.31 % |
| 15 min | 10,553 / 0.35 % / 0.30 % | 5,347 / 0.26 % / 0.13 % | 5,503 / 0.71 % / 0.58 % |
| **20 min** | **12,165 / 0.60 % / 0.61 %** | **5,993 / 0.38 % / 0.28 %** | **6,182 / 0.91 % / 0.79 %** |
| 30 min | 13,202 / 0.98 % / 1.07 % | 6,387 / 0.61 % / 0.58 % | 6,589 / 1.37 % / 1.34 % |

**(a) is a fraction of a percent, so the stand IS exclusive — and (b) is not above it.**
At `TURN_MIN` = 20 min the base head violates the ordering on 0.61 % / 0.28 % / 0.79 % of
the rows in scope against a truth that violates it on 0.60 % / 0.38 % / 0.91 %. The
configuration the desk named as the rider condition — (a) low and (b) well above it —
does not occur on any fold.

**Per airport** (in scope / (a) / (b), `TURN_MIN` = 20 min, july_loo · december · january):

| airport | july_loo | december | january |
|---|---|---|---|
| EGLL | 4,975 / 0.00 % / 0.00 % | 2,354 / 0.00 % / 0.00 % | 2,113 / 0.00 % / 0.00 % |
| LTFM | 3,060 / 0.00 % / 0.00 % | 1,915 / 0.05 % / 0.05 % | 2,027 / 0.00 % / 0.00 % |
| LFPG | 1,795 / 0.06 % / 0.06 % | 1,193 / 0.17 % / 0.08 % | 973 / 0.10 % / 0.10 % |
| EDDM | 869 / 3.68 % / 4.14 % | — | — |
| LEMD | 500 / 1.00 % / 1.00 % | — | 558 / 2.33 % / 1.79 % |
| EDDF | 387 / 0.52 % / 0.78 % | 104 / 1.92 % / 1.92 % | 122 / 0.00 % / 0.00 % |
| LIRF | 141 / 0.00 % / 0.00 % | 186 / 0.54 % / 0.00 % | 204 / 0.00 % / 0.49 % |
| LSZH | 138 / **13.0 %** / 11.6 % | 57 / **17.5 %** / 15.8 % | 94 / **35.1 %** / 35.1 % |
| EHAM | 128 / 9.38 % / 9.38 % | 80 / 7.50 % / 5.00 % | — |

The split is clean and it is the whole answer: where the stand is exclusive (EGLL, LTFM,
LFPG, LIRF — (a) ≤ 0.2 %) the base head **never** violates the ordering either, and where
the head does violate it (LSZH, EHAM, EDDM) the truth violates it just as often, so the
stand string there is a stand **area** and not an exclusive stand. **There is no airport
where (a) is low and (b) is high.**

**(c) The clip is negative on every fold and every margin** (`work/standclip.py`,
`analysis/stand-clip.json`; `p_clip = max(p12, dt + margin)` on the rows in scope,
projected on the registered instrument at the ranking months' reach grain):

| margin | july_loo s² | december s² | january s² | pooled board s | face ×1.2 / ×1.5 |
|---|---|---|---|---:|---|
| 0 s | −6.15 [−11.5, −1.7] | −7.19 [−17.1, −1.0] | −10.33 [−14.7, −6.5] | −0.0156 | −0.0187 / −0.0234 |
| 60 s | −7.12 [−13.9, −1.8] | −8.40 [−20.2, −1.0] | −11.99 [−17.0, −7.4] | −0.0181 | −0.0217 / −0.0272 |
| 120 s | −6.92 [−15.5, −0.6] | −8.69 [−22.0, −0.3] | −13.80 [−19.6, −8.6] | −0.0194 | −0.0233 / −0.0292 |

**Every fold's interval is clear below zero at every margin.** The reason is in one line:
**of the rows the clip would move, 79.7 % / 88.2 % / 91.8 % are rows whose TRUE off-block
is also after `Q`.** Their median true taxi is **654 s / 306 s / 8 s** against a bound of
956 s / 934 s / 795 s. The clip raises a prediction by an RMS of 367 / 537 / 546 s (max
810 / 891 / 1,117 s) on 74 / 17 / 49 rows, and it is wrong on nine in ten of them: it costs
−165,561 / −356,734 / −370,658 s² per clipped row. The bound is **false exactly where it
binds**.

**`LTFM/unreached`, the cell the rider was wanted for:** the constraint essentially never
binds there — **0 / 1 / 0** rows clipped at margin 0, 5 / 2 / 0 at 60 s, 17 / 9 / 0 at
120 s — because LTFM's (a) and (b) are both zero. The one december row costs −33.9 s² per
class row on the cell. The channel does not reach the population.

**Verdict: DEAD**, on the desk's own rule and with the reason named. Not because the stand
is non-exclusive — at the served airports it is, at 99.4 % or better — but because **the
constraint carries no residual information**: the head's violations are the same rows as
the truth's, so there is nothing left to clip. The airports where a clip would fire (LSZH,
EHAM, EDDM) are the airports whose stand strings are not exclusive, and none of them is a
served cell. **Revival condition:** a stand identifier that separates a stand *area* into
exclusive positions at LSZH, EHAM and EDDM, plus a next-occupant identification that does
not rest on a minimum-turnaround threshold — the sibling's turnaround chain (`122c7d84`
mode 3) is the only one in the record, and it places 1,170 / 861 / 731 fold rows, two
orders of magnitude short of the 12,165 this test needed.

**Files.** `work/standclip.py`, `analysis/stand-clip.json`,
`analysis/stand-clip-vacuous.json` (the two vacuous anchors),
`logs/standclip.log`, `logs/standclip-v1.log`. No fit, no serve.

---

### 7ae. The no-NM-join departures — a census before any build

The desk, 2026-09-09, from the second research proposal. About 1–2 % of departures are
routed as **no NM join**: `FLIGHT_ID_mvt` is null, so the whole `*_flt` block is absent —
no `AOBT_3_flt`, no `EOBT_1_flt`, no callsign, no operator. A no-NM departure that flies to
one of the **ten** appears a second time in the file, as the **arrival** at its
destination, and that record may carry the join the departure record lacks. Census only:
`work/nonm_census.py`, `analysis/nonm-census.json`. **No model, nothing served.**

**The match key.** The desk's words are "same callsign or flight number"; a no-NM row has
**no callsign** — `CALLSIGN_flt` is a flight-table column and the flight table is exactly
what is missing — so `FLIGHT_mvt`, the movement table's own flight number, is the only key
present on both sides. Three tightenings are reported so the key's cost is visible: the
plain key (same flight number, same UTC day, landing after take-off, arrival's home
airport = the departure's `ADES_mvt`), the key with the arrival's `ADEP_mvt` agreeing, and
the key with a **usable** clock (`AOBT_3_flt` finite and 0 < take-off − AOBT < 20,000 s).
An ambiguous key keeps the earliest candidate and the ambiguity count is in the record.

**(1) The census.**

| month | departures | no-NM | share | to one of the ten | arrival with an NM join | + same `ADEP` | **usable clock** | of all no-NM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-07 `july_loo` | 190,713 | 3,934 | 2.06 % | 583 | 203 | 200 | **198** | 5.0 % |
| 2025-12 `december` | 165,677 | 1,643 | 0.99 % | 284 | 70 | 69 | **69** | 4.2 % |
| 2025-01 `january` | 153,706 | 1,432 | 0.93 % | 290 | 66 | 66 | **66** | 4.6 % |
| 2026-01 ranking | 152,719 | 2,449 | 1.60 % | 445 | 108 | 106 | **106** | 4.3 % |
| 2026-07 ranking | 192,122 | 2,834 | 1.47 % | 346 | 113 | 113 | **113** | 4.0 % |

152,719 + 192,122 = **344,841**, the scored set exactly. **219 ranking rows are
recoverable** — 4.1 % of the 5,283 no-NM departures and **0.064 % of the board**. The
channel narrows twice: only 12–18 % of no-NM departures fly to one of the ten at all
(most leave the region), and only a quarter to a third of those have an arrival record
that carries the join.

**(2) The base head's OOF RMSE by regime** (v12; `p11` is in the record beside it):

| fold | no-NM rows | RMSE no-NM | RMSE with-NM | MSE gap s² |
|---|---:|---:|---:|---:|
| july_loo | 3,934 | 1,118.45 | 236.13 | +1,195,184 |
| december | 1,643 | 655.22 | 194.46 | +391,496 |
| january | 1,432 | 2,835.53 | 198.34 | **+8,000,882** |

The regime carries a fat label tail: `y_true` at p50/p90/p99 is 1,014 / 1,725 / **11,536** s
on no-NM july rows against 946 / 1,510 / 2,583 s on with-NM. Per airport the gap is carried
by one airport per fold — LIRF july (337 rows, RMSE 3,415), EDDM december (130, 799),
**LFPG january (216 rows, RMSE 6,904, gap +47.6 M s²)**.

**On the recoverable rows themselves** — the number the face should use, because the
recoverable rows are scheduled flights between two of the ten and the no-NM population at
large is not: RMSE **1,124.68 / 694.91 / 569.59** on 198 / 69 / 66 rows, gaps
**+1,209,143 / +445,080 / +285,097** s².

**(3) The face arithmetic**, rows × (MSE_noNM − MSE_withNM) / 344,841, converted at v12's
clean board and shown at ×1.2:

| gap used | pooled s² | Δ s²/ranking row | board s | **face ×1.2** |
|---|---:|---:|---:|---:|
| the WHOLE no-NM regime | 4,209,209 | 2,673.2 | +5.2656 | **+6.32** |
| the RECOVERABLE rows only | 799,913 | 508.0 | +0.9923 | **+1.19** |
| recoverable, five worst rows per fold removed | 227,251 | 144.3 | +0.2815 | **+0.34** |

Per fold on the recoverable gap: july_loo +1.80, december +0.66, january +0.42 s at ×1.2.
The first row of that table is the desk's arithmetic taken literally and it is **not the
number to quote**: it credits 219 scheduled flights with january's whole-regime gap, which
LFPG's 216 unrecovered rows create. The gap is also **tail-carried** — the single worst
recoverable row holds 25.0 % / 53.2 % / 26.9 % of the fold's MSE and the worst five hold
68.4 % / 74.4 % / 69.4 %; the row bootstrap on 198 / 69 / 66 rows is
[541k, 2,267k] / [147k, 1,038k] / [136k, 586k] s². Even trimmed the face is **+0.34 s**,
still three times the desk's 0.1 s bar.

**The read that decides it, and it is not in the desk's list.** If the arrival row's
`AOBT_3_flt` really is this flight's off-block, then `take-off − AOBT_3` must read like a
taxi time on these rows — it is the same file-clock proxy this lane is built on, 350–466 s
from the truth (exp-043 §4). It does not:

| fold | rows | head `p12` RMSE | recovered-clock RMSE | median(proxy − y) | IQR | rows off by > 600 s | proxy beats head |
|---|---:|---:|---:|---:|---|---:|---:|
| july_loo | 198 | 1,124.68 | **2,172.56** | −115.5 s | [−242, +57] | 21 | 54.5 % |
| december | 69 | 694.91 | **2,653.51** | −63.0 s | [−659, +115] | 24 | 31.9 % |
| january | 66 | 569.59 | **1,815.88** | +46.0 s | [−69, +135] | 9 | 53.0 % |

The **centre is right** — the median recovered taxi is within two minutes of the truth on
every fold — and the **tail is wrong**: december's p90 |proxy − y| is 5,336 s. That is the
signature of a key that matches the **wrong rotation**: the same flight number, the same
day, the same origin, but a later leg. Used raw the recovered clock **costs** 3.5 M / 6.6 M
/ 3.0 M s² on those rows. The per-row oracle minimum of head and proxy takes 935,727 /
88,190 / 118,696 s², so a blend cannot exceed that, and the oracle needs the label.

**Verdict: the channel is REAL but small, tail-carried and not yet usable — no kill, and
no build until the key is fixed.** What is true: 219 scored rows are reachable, they carry
a genuine error gap, and even the trimmed face (+0.34 s at ×1.2) clears the bar. What is
not true: that the recovery works as it stands — on a third to a half of the matched rows
it names the wrong flight, and that is why the direct proxy loses.

**The cheapest next test, before any model.** Tighten the key and re-read the same proxy
RMSE: require the arrival's `AIRCRAFT_TYPE_mvt` to agree with the departure's, require the
implied flight time (arrival landing − departure take-off) to be within a plausible band
for the city pair, and drop every ambiguous key instead of keeping the earliest. If the
recovered-clock RMSE falls to the file-clock's own 350–466 s on the rows that survive, the
channel is a rider and the count that survives is its size; if it does not, the join is
noise and the face was never real. That test is minutes and needs no fit — it is the same
census script with three more clauses.

**Files.** `work/nonm_census.py`, `analysis/nonm-census.json`, `logs/nonm-census.log`.

---

### 7af. WHY the no-NM regime reads 1,118 / 655 / 2,836 s — an anatomy, no fit

The desk, 2026-09-09: *"A gap is a question — WHY first, before any matcher clause."*
Four parts on the three folds (`work/nonm_anatomy.py`, `analysis/nonm-anatomy.json`).
**Nothing is fitted here and nothing is served.**

**The split that has to come before all four.** Every `noac` row is **also** a no-NM row —
`noac_only` is **0** on all three folds — and `prc/blend.py::regime_labels` sends a row that
is both to the **`noac`** regime, not `nonm`. So "the no-NM regime" names two populations,
and the difference is most of the headline:

| fold | `nm_missing` (all) | RMSE | share of the fold's Σr² | core = `nm & ¬noac` | RMSE | `nm & noac` | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| july_loo | 3,934 | **1,118.45** | 32.1 % | 3,595 | **523.54** | 339 | 3,407.35 |
| december | 1,643 | **655.22** | 10.2 % | 1,554 | **471.60** | 89 | 2,010.46 |
| january | 1,432 | **2,835.53** | 65.8 % | 1,369 | **2,800.61** | 63 | 3,509.48 |

In the ranking months the same identity holds: LIRF's no-NM rows (107 in 2026-01, 276 in
2026-07) are **exactly** the `noac` rows (108 / 280) — LIRF does not appear in the core
population at all.

**(a) The top 30 by squared residual** show two mechanisms and nothing else. The top 30
carry **56.5 % / 66.1 % / 97.9 %** of the regime's squared residual and **18.1 % / 6.7 % /
64.4 %** of the *whole fold's*.

*Mode 1 — the block time is written a day early.* `LFPG→LFMN` h18 A319, label **84,240 s**,
`p12` 870, take-off − SOBT **1,740 s**, block-day offset **1**; `LFPG→GMMX` 58,206 s;
`LIRF→EGPF` 87,177; `LIRF→SAEZ` 87,186; `LIRF→LFBD` 87,361. A 24-hour taxi with an ordinary
schedule gap is not a hold.

*Mode 2 — the head predicts hours of taxi.* `LIRF→LTFJ` `p12` **19,070 s** against a label
of 1,377; `LIRF→LWSK` 13,610 against 959; `LIRF→KJFK` 11,644 against 842. Every one is at
LIRF, every one has `AIRCRAFT_TYPE_mvt` null (so it is a `noac` row), and every one has a
take-off − SOBT of 12,000–21,000 s. Without the flight table the schedule gap is the only
delay signal the head has, and it reads it as taxi.

The turnaround flag does **not** separate them: an earlier same-day arrival at the row's own
stand covers 88.7 % / 81.0 % / 78.4 % of the regime against 85.3 % / 84.6 % / 83.9 % of the
with-NM rows. "Positioning or first-of-day" is not the story.

**(b) Physical or artefact — said row by row.** `y_true` is `TAXITIME_SEC_mvt`, take-off
minus the departure's `BLOCK_TIME_UTC_mvt`, so a wrong-day block time leaves take-off − SOBT
ordinary while the label runs to hours, and the two dates differ. Of the labels over
3,600 s:

| fold | labels > 3,600 s | block time a day early | label beyond the schedule gap | **physical** | artefacts' share of the regime's Σr² |
|---|---:|---:|---:|---:|---:|
| july_loo | 153 | 16 | 10 | 130 | 22.1 % |
| december | 71 | 4 | 0 | 67 | 22.1 % |
| january | 53 | 5 | 7 | 44 | **94.5 %** |

**Most extreme labels are physical; the extreme *residuals* are not.** With the artefacts
removed the core regime reads **523.4 / 471.6 / 536.9 s** — january's 2,836 s collapses to
537 s when **three** rows go, and its worst **two** rows carry **95.2 %** of the core's
squared residual. The artefact is not unique to the regime (140 / 173 / 140 with-NM rows
also carry a day-early block time) but it is 3–4× more frequent in it: 0.41 / 0.24 / 0.35 %
against 0.08 / 0.11 / 0.09 %.

**The regime's real gap, after `noac` and after the artefacts, is stable:**
**+218,194 / +184,563 / +248,871 s²**, an RMSE of ≈480–540 s against the with-NM 236 / 194 /
198 — about **2.4×**, and it is the same size on every fold.

**(c) Does anything in the file separate it?** Two different answers, and the difference
decides.

*Variance — yes, strongly.* `take-off − SOBT` is monotone on the core: RMS climbs
225 → 1,063 s across deciles on july_loo and 141 → 894 on december, and the top decile holds
**41.3 % / 36.0 %** of the core's squared residual on 10 % of the rows (concentration ×4.1 /
×3.6). The same decile edges on the **with-NM** rows give RMS **383 / 319 / 363** — the head
handles the same schedule gap perfectly well when it has the flight table, so this is a
missing-input problem, not a hard-flight problem.

*Mean — essentially no.* The mean residual is flat across those deciles: −64 … −2 s
(july_loo), −31 … +68 (december). A regime-only correction is a shift of the conditional
mean, so what it could take is the **oracle mean shift** — give every group its own mean
residual, in sample, chosen on these folds, no cross-fit and no shrinkage:

| separator | groups (jul / dec / jan) | oracle MSE taken, core | share of the regime's MSE |
|---|---|---|---|
| `take-off − SOBT` decile | 10 | 1,653 / 2,669 / 118,660 | 0.60 % / 1.20 % / 1.51 % |
| airport | 9 | 2,996 / 14,843 / 65,253 | 1.09 % / 6.67 % / 0.83 % |
| hour | 24 / 23 / 23 | 3,764 / 5,989 / 124,322 | 1.37 % / 2.69 % / 1.58 % |
| aircraft type | 119 / 93 / 97 | 6,767 / 12,382 / 113,833 | 2.47 % / 5.57 % / 1.45 % |
| **operator** | **374 / 262 / 249** | **44,335 / 51,429 / 668,804** | **16.2 % / 23.1 % / 8.5 %** |
| turnaround flag | 2 | 747 / 147 / 63,187 | 0.27 % / 0.07 % / 0.81 % |

Only the operator looks large, and it is the one with **374 groups over 3,595 rows** — about
ten rows a group, which is what an in-sample per-group mean looks like when it is fitting
noise. Every low-cardinality separator takes **under 7 %**. So (c)'s honest answer is: the
file separates the regime's **variance** well and its **mean** barely, and the single
candidate that could still be real is the operator, which only a cross-fit can settle.

**(d) The ranking months.**

| month | scored departures | no-NM | of them `noac` | **core** | core above the folds' 9,300 s flag |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 152,719 | 2,449 | 108 | **2,341** | 610 |
| 2026-07 | 192,122 | 2,834 | 280 | **2,554** | 246 |

Core rows by airport, 2026-01: EHAM **805**, LFPG 324, LTFM 331, LSZH 299, EDDF 212,
EDDM 119, EGLL 112, LEBL 77, LEMD 62 — and **no LIRF**. 2026-07: LFPG 408, LTFM 384, EHAM
401, LEBL 364, LSZH 359, EDDF 209, LEMD 174, EDDM 119, EGLL 136 — again no LIRF. Of the 610
and 246 rows above the flag, EHAM holds 367 and LEBL 67. **The fold's worst core airports
are not the ranking months' core airports**, which is a transfer warning any regime
correction has to answer.

**4,895 core rows are in the scored set** — 1.42 % of 344,841.

**What this changes.** The desk's headline (1,118 / 655 / 2,836 s) is three regimes read as
one: a `noac` population the blend already treats separately, three artefact rows, and a
core regime that reads a steady ≈480–540 s. The question that is left is (c)'s operator
candidate, and the desk's own rule sends it to the ordinary gate — §7ag.

**(e) The desk's conditional branch: a regime-only correction under the ordinary gate.**
(c) left one candidate a cross-fit had to settle — the operator — so it was priced
(`work/nonm_regime.py`, `analysis/nonm-regime.json`): ridge on standardised columns,
leave-one-fold-out over the three re-pinned folds, every group encoding built on the
fitting folds only (shrunk toward their regime mean, k = 20), contract 2A − B,
day-clustered bootstrap, all columns servable. Gain in s² per REGIME row:

| arm | july_loo | december | january | projected s²/ranking row | board s | face ×1.2 |
|---|---:|---:|---:|---:|---:|---:|
| intercept only | −5,231 [−7,856, −2,255] | −244 [−1,759, +1,618] | −4,789 [−11,906, −203] | −71.3 [−121.3, −30.4] | −0.1389 | −0.167 |
| schedule gap | −4,982 | −8,169 | −3,340 | −59.6 [−105.0, −18.9] | −0.1161 | −0.139 |
| gap + hour + `p12` | −5,131 | −8,516 | +169 | −36.9 [−104.1, +34.1] | −0.0719 | −0.086 |
| **operator** | **−97,214** [−119,911, −75,230] | −80,924 | −13,838 | −813.9 [−994.9, −635.0] | −1.5820 | −1.898 |
| operator + type + airport | −110,421 | −85,716 | −10,984 | −892.4 | −1.7339 | −2.081 |
| every column | −138,582 | −100,762 | −5,204 | −1,061.7 | −2.0616 | −2.474 |

**Every arm fails on every fold, and the operator fails worst.** Its in-sample oracle read
16–23 % of the regime's MSE; cross-fitted it **costs 97,214 s² a row** — the difference
between ten rows a group in sample and ten rows a group out of it. The same table on the
whole `nm_missing` population is worse still (operator −1,626.6 s²/ranking row, board
−3.15 s).

The **intercept-only** arm answers the question the desk framed it for: a single constant
shift on the regime is **−5,231 / −244 / −4,789** s² a row. The base head's conditional
mean on this regime is already right; there is no bias to take, only variance the file
cannot resolve. **The regime-only correction is closed at this design.** Revival: a column
that reads the *missing input itself* — the flight table's off-block for these rows — which
is what §7ae's arrival-side recovery was reaching for and which its key does not yet deliver.

**Files.** `work/nonm_anatomy.py`, `analysis/nonm-anatomy.json`, `logs/nonm-anatomy.log`,
`work/nonm_regime.py`, `analysis/nonm-regime.json`, `logs/nonm-regime.log`.

---

### 7ag. The extreme predictions, by regime — no cap wins anywhere

The desk, 2026-09-09: v17's bytes carry 882 rows predicted over 3,600 s (`nm_late` 410,
`nm_bulk` 264, `nonm` 19, `noac` 189), 226 over 5,400, 110 over 7,200, 49 over 10,800, 27
over 20,000, max 115,306 s; the board MSE is 60,841 s² a row so they cannot be *mostly*
wrong, but the `noac` oracle was priced at 6.9 s (exp-022) so the `noac` tail may be.
Measured on the three folds (`work/extreme_caps.py`, `analysis/extreme-caps.json`), regimes
from the panel's own `regime` column. **No fit, nothing served.**

**The head is right when it predicts big — in every regime.**

| fold · regime | rows `p12` > 3,600 | label median | labels > 3,600 | labels > ½·`p12` | share of the fold's Σr² | rows > 7,200 | label median | > ½·`p12` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jul · `nm_bulk` | 21 | 5,050 | 90.5 % | 90.5 % | 0.32 % | 1 | 87,002 | 100 % |
| jul · `nm_late` | 253 | 4,569 | 77.9 % | **96.1 %** | 2.29 % | 14 | 8,991 | 100 % |
| jul · `nonm` | **0** | — | — | — | 0.00 % | 0 | — | — |
| jul · `noac` | 171 | 6,417 | 60.8 % | 62.0 % | **22.57 %** | 81 | 11,330 | 75.3 % |
| dec · `nm_bulk` | 36 | 3,923 | 66.7 % | **100 %** | 0.21 % | 0 | — | — |
| dec · `nm_late` | 55 | 4,140 | 67.3 % | **98.2 %** | 1.18 % | 4 | 8,286 | 100 % |
| dec · `nonm` | **0** | — | — | — | 0.00 % | 0 | — | — |
| dec · `noac` | 61 | 7,320 | 91.8 % | 91.8 % | 4.85 % | 25 | 10,609 | 96.0 % |
| jan · `nm_bulk` | 51 | 4,140 | 68.6 % | **98.0 %** | 0.11 % | 0 | — | — |
| jan · `nm_late` | 91 | 4,392 | 73.6 % | **98.9 %** | 0.67 % | 9 | 9,468 | 100 % |
| jan · `nonm` | **0** | — | — | — | 0.00 % | 0 | — | — |
| jan · `noac` | 32 | 6,089 | 81.3 % | 81.3 % | 4.13 % | 10 | 22,468 | 90.0 % |

**`nonm` has no extreme prediction at all on any fold** — zero rows over 3,600 s on 3,595 /
1,554 / 1,369 regime rows — while v17's ranking bytes carry 19. That regime's cap cannot be
validated on the folds because the folds contain no row it would touch; it is named here as
a transfer gap, not priced.

**The caps** — applied to one regime's rows only, every other byte untouched; paired
squared-error change, `raw` and `clean` (§7af's artefact labels removed), and the board
converted from the fold RMSE at `SHARES`:

| regime | cap | rows capped (jul/dec/jan) | s²/fold row, clean | board s, raw | board s, **clean** | face ×1.2, clean |
|---|---:|---|---|---:|---:|---:|
| `nm_bulk` | 3,600 | 21 / 36 / 51 | −272 / −37 / −97 | −31.6148 | −0.3950 | −0.474 |
| `nm_bulk` | 5,400 | 4 / 1 / 1 | −52 / +0.9 / +0.9 | −30.0928 | −0.0585 | −0.070 |
| `nm_bulk` | 7,200 | 1 / 0 / 0 | 0 / 0 / 0 | −28.7949 | **0.0000** | 0.000 |
| `nm_late` | 3,600 | 253 / 55 / 91 | −4,550 / −971 / −2,170 | −5.6086 | −7.0040 | −8.405 |
| `nm_late` | 7,200 | 14 / 4 / 9 | −335 / −81 / −59 | −0.3444 | −0.4322 | −0.519 |
| `nm_late` | 10,800 | 0 / 0 / 0 | 0 | 0.0000 | 0.0000 | 0.000 |
| `nonm` | any | 0 / 0 / 0 | 0 | 0.0000 | 0.0000 | 0.000 |
| **`noac`** | 3,600 | 171 / 61 / 32 | −30,631 / −15,902 / −23,670 | **−207.66** | **−50.74** | −60.89 |
| `noac` | 7,200 | 81 / 25 / 10 | −17,603 / −8,302 / −18,062 | −182.14 | −33.85 | −40.62 |
| `noac` | 10,800 | 43 / 11 / 7 | −10,570 / −5,364 / −13,584 | −160.73 | −23.10 | −27.72 |

**No cap wins on any fold in any regime.** The only entries that "win on every fold" are the
ones that move **zero rows** (`nonm` at every cap, `nm_late` at 10,800, `nm_bulk` at 7,200
and above), so there is no candidate and no cross-fit to run. Note the `nm_bulk` raw column:
its −31.6 board seconds are **one july row** whose label is 87,002 s — a §7af wrong-day
artefact — and the clean column shows the truth, −0.40 s.

**The mechanism on the `noac` tail, which is the read the desk wanted.** `noac` is a
**subset** of no-NM on every fold (§7af), so there is no flight table on these rows and
**`AOBT_3_flt` is present on 0 of the 81 / 25 / 10** rows predicted over 7,200 s. AOBT
cannot be the driver; the schedule gap is, and the label follows it:

| fold | rows | AOBT present | ρ(`p12`, gap) | ρ(label, gap) | ρ(label, `p12`) | median gap | median `p12` | median label | median label/gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| july_loo | 81 | **0** | **0.9958** | 0.9529 | 0.9640 | 13,854 s | 10,997 s | 11,330 s | **1.000** |
| december | 25 | **0** | **0.9990** | 0.9902 | 0.9889 | 10,799 s | 10,343 s | 10,609 s | **1.000** |
| january | 10 | **0** | **0.9987** | 0.9616 | 0.9654 | 22,468 s | 21,969 s | 22,468 s | **1.000** |

The head predicts `p12 ≈ 0.92–0.99 × (take-off − SOBT)` and **the label is the schedule gap
exactly** — median label/gap is 1.000 on all three folds. For a row with no flight table the
movement's scheduled time is the bookkeeping off-block, so `y = take-off − block` *is*
`take-off − SOBT`. The head has learned an identity, and it is right; 75.3 % / 96.0 % / 90.0 %
of these labels land within a factor of two of the prediction.

**Verdict: the `noac` tail is not the exp-022 oracle's 6.9 s of headroom — it is the one
part of the head that is already exact.** Capping it costs 33.9–50.7 board seconds clean
(207.7 raw). The extreme predictions are closed as a source: every regime's big predictions
are carried by big labels, `nonm` has none on the folds to test, and the only measurable
gain from a cap is zero, from caps that move nothing.

**Files.** `work/extreme_caps.py`, `analysis/extreme-caps.json`, `logs/extreme-caps.log`.

---

### 7ah. The shrunk varying-coefficient calibration on `LTFM/unreached` — KILLED

The research follow-up's rank-1 LTFM test, run to its own pre-declared stop rule
(`work/ltfm_phi.py`, `analysis/ltfm-phi.json`). `h = a_g + b_g (F − A)` with `A =
AOBT_3_flt` and `F = LOBT_flt` — the filed off-block clock the base phi relationship
already uses, `prc/tail_model.py:77` defining `BLOCK = AOBT_3 + phi·(LOBT − AOBT_3)`, so
`F − A` is exactly the span that relationship interpolates. Groups `g` are **operator ×
stand** and **operator × scheduled bank**, ridge-shrunk toward the airport-level intercept
and slope, fitted **outside january** on july_loo + december and priced on january
`LTFM/unreached` with `G = mean(2 e h − h²)`, `e = r12`, day-clustered.
**Stop rule, pre-declared by the research read: an upper bound below 185 s²/row kills it;
369 s²/row is the 0.1 s thesis.** The shrinkage `k` is chosen **inside** the fitting folds
(july against december and back), never on january.

Population: 25,171 / 22,609 / 21,368 `LTFM/unreached` rows, of which 24,746 / 22,279 /
21,136 carry `LOBT − AOBT`; 2,836 / 2,592 / **2,522** operator × stand groups and 642 / 604
/ **594** operator × bank groups.

**The fold table — january, priced, every arm killed:**

| arm | chosen `k` | G, s²/row | 95 % interval | RMS `h` | verdict |
|---|---:|---:|---|---:|---|
| operator × stand, slope | 3,000 | **−14,190.42** | [−17,900.8, −10,803.0] | 118.4 s | KILLED |
| operator × stand, intercept | 1,000 | **−536.74** | [−776.8, −292.4] | 12.3 s | KILLED |
| operator × bank, slope | 3,000 | **−4,701.94** | [−6,908.3, −2,940.2] | 62.8 s | KILLED |
| operator × bank, intercept | 3,000 | **−584.08** | [−807.9, −365.2] | 12.9 s | KILLED |
| *airport level, slope* (control) | — | −570.81 | [−835.3, −321.2] | — | `a0` +12.495, `b0` −0.00394 |
| *airport level, intercept* (control) | — | −553.37 | [−800.0, −303.5] | — | `a0` +12.495 |

No upper bound is merely *below* 185 — every one is **negative**. The gate on all three
folds is not run, because january did not survive.

**The failure mode is the one the desk named, and the intercept-only arm proves it.** The
two intercept-only arms (−536.7, −584.1) are indistinguishable from the airport-level
constant (−553.4): **the groups add nothing over the airport level**, and the whole loss is
the airport-level constant. Its cause is one line — the regime's mean residual per fold:

| fold | rows | mean `r12` |
|---|---:|---:|
| july_loo | 25,171 | **+25.67 s** |
| december | 22,609 | −2.08 s |
| january | 21,368 | **−15.57 s** |

Fitting on july_loo + december gives `a0 = +12.5 s`; january needs **−15.6 s**. The level
flips sign between the fitting folds and the priced one, so a constant shift alone costs
−553 s²/row. The `k` selection inside the fitting folds had already said so: held-out `G`
is negative at every `k` for every group definition and improves **monotonically as
k → ∞** — the search's own optimum is "do nothing", approached from below.

**The structure is real inside a month; it is transfer that fails.** The january
**in-sample** oracle — fitted on january itself — reads **+7,015 to +8,060** s²/row for
operator × stand with slope and **+2,370 to +3,310** for operator × bank, all far above the
369 s² thesis. So the varying coefficient is not measuring nothing; it is measuring
something a month-out fit cannot carry, on ~8 rows per operator × stand group.

**The repair attempted before the kill.** If the whole loss is the untransferable level,
anchor the airport level at **zero** and let only the group deviations move:

| arm | `k` = 300 | `k` = 1,000 | `k` = 3,000 |
|---|---:|---:|---:|
| operator × stand, slope | −27,709.7 | −18,784.6 | −13,603.3 |
| operator × stand, intercept | −162.0 | −56.6 | **−19.5** [−32.5, −7.3] |
| operator × bank, slope | −5,554.4 | −5,114.8 | −4,528.0 |
| operator × bank, intercept | −491.3 | −356.6 | −204.1 |

Every anchored arm is still negative, and the best of them (−19.5 s²/row, RMS `h` = **1.09
s**) is the one that has shrunk itself almost to nothing. The group structure does not
transfer either — it is not only the level.

**Verdict: DEAD on the pre-declared stop rule, repair attempted and failed.** **Revival
condition:** a group estimate that transfers across months. The in-sample oracle says the
channel exists *within* a month, so what would revive it is same-month information — and on
a ranking month that means labels, which do not exist. At this design the phi-span
calibration on `LTFM/unreached` is closed.

**Files.** `work/ltfm_phi.py`, `analysis/ltfm-phi.json`, `logs/ltfm-phi.log`.

---

### 7ai. BRANCH REPORT — the no-clock lane, `closed-provisional`

The desk, 2026-09-09, after §7ah: the lane's local table and every external proposal is now
measured negative or a MAYBE at this design, so file the branch report in the COMPETING.md
shape. **Nothing here is a new measurement.** Every number is quoted from the section that
made it, and the two numbers COMPETING.md requires separately — destination value and path
feasibility — are in §2 and §3 below, never mixed.

**Status: `closed-provisional`.** Matt confirms or reverses (AGENTS.md, Checkpoints). This
is a belief with its evidence, not a claim that the destination is worth nothing — §2 says
the opposite.

#### 1. The negative, per class, with the number that controls it

| class | the negative | the controlling number | where |
|---|---|---|---|
| **held cells** | 16 of 19 (airport, reach class) cells fail their own fold gate; summing every positive one does not reach the face bar | **+0.0906 s at ×1.2 for ALL of them**; best is `LEBL/reached` +0.0371, failing december alone at **−27.14** lower bound on 1,071 rows | §7z, T7g |
| **shrunk head** (`N_reach_half`) | the increment over arm `N` does not clear on january under the registered decision rule | january **+80 [+1.2, +161]** at 20,000 draws, **worst of ten seeds −3.5**; the shortfall is january's BULK (+77.9 [+0.3, +158]) | T7m, T7n |
| **per-cell form** (`N_reach`) | the same gate, further from clearing | january **+78 [−77, +234]**, against july +226 [+55, +415] and december +222 [+62, +463] | T7l |
| **regulation + network** | FAIL cross-fitted, and the version that priced highest cannot be written at serving time | pooled **−0.0734 s**, `LTFM/unreached` **−242 / −435 / −383** s²/row; the label network's +0.0683 s is an information bound, not a feature | §7ac |
| **stand ordering** | the constraint carries no residual information — the head's violations are the truth's own rows | at the served airports the stand is exclusive at **99.4 %+**; on `LTFM/unreached` the clip fires on **0 / 1 / 0** rows at margin 0 | §7ad |
| **no-NM recovery** | the arrival-side key matches the wrong rotation, so the recovered clock loses to the head it would replace | recovered-clock RMSE **2,173 / 2,654 / 1,816** against the head's **1,125 / 695 / 570**; used raw it costs **3.5 M / 6.6 M / 3.0 M** s² | §7ae |
| **no-NM regime correction** | every arm fails on every fold; the base's conditional mean on the regime is already right | intercept-only **−5,231 / −244 / −4,789** s²/row; the operator arm **−97,214** cross-fitted after an in-sample oracle of 16–23 % | §7af(e) |
| **extreme caps** | the big predictions are carried by big labels; on the `noac` tail the head has learned an exact identity | no cap wins on any fold in any regime; capping the `noac` tail costs **33.9–50.7 board seconds clean**, median label / schedule gap **1.000** on all three folds | §7ag |
| **LTFM calibration** | killed on its own pre-declared stop rule; the level does not transfer | every upper bound **negative** (best intercept arm **−536.7 [−777, −292]**) against a kill line of 185; regime mean residual **+25.67 / −2.08 / −15.57 s** across the folds | §7ah |

**The cause under the nine rows is one cause, and it is worth naming once.** Seven of the
nine die on the same fold or the same mechanism: **january, and the fact that a conditional
mean estimated on two months does not transfer to a third.** The shrunk head, the per-cell
form, `LEBL/reached`, `EDDF/reached` under `N_reg_reach`, the regime correction and the LTFM
calibration are all the same sentence at different grains — the structure is visible inside
a month (january's in-sample LTFM oracle **+7,015 to +8,060** s²/row; the operator's
in-sample 16–23 % of the no-NM regime's MSE) and does not survive a month-out fit. The two
that die differently — stand ordering and the extreme caps — die because the head is
**already right** there, which is the same finding from the other side.

#### 2. Destination value — what the population is worth if executed

**Stated on its own, and it is positive.** The rows this lane exists for are the clockless
ranking departures that no clocked segment covers: **125,176 of them, 81 % of the clockless
rows**, worth **+6.651 s at ×1.2** and **+9.923 s at ×1.5** by `exp-046-surfacefit`'s
segment ceiling (its §18.2 — read, not edited). No matcher reaches them; they are this
lane's and the harvest's.

**And part of it is already delivered.** `LTFM/unreached` is **47,577** of those rows and it
is served today: LTFM has **zero** clocked segments in either ranking month, so the file
clock is the only instrument that can reach it, and this lane's `Nf_*` family is that
instrument. The lane has taken two board readings on the population — v14 **−0.1914 s**
(BOARD READING 13, transfer 1.03× on the all-row fold price) and v17 **−0.0482 s** (BOARD
READING 16, 0.70×, inside the fold interval) — and the current file is on the board at
**246.6571**, rank 1.

**So the destination is not measured non-positive and the outright-close rule does not
apply.** What is measured non-positive is every *remaining path to it at this design*.

#### 3. Path feasibility — separately, and this is the negative

**The lane's own instrument is spent at this design.** The served set is 3 cells of 19;
every other cell fails on its own fold evidence, every form change is a MAYBE that dies on
january, and the whole positive residue of the local table — every held cell summed — is
**+0.0906 s at ×1.2**, under the desk's 0.1 s bar before any of it is established. The
re-serve increment over v17's own bytes is **+0.62 s²/row pooled [−1.49, +2.56]**, a face of
**+0.0010 s**, two orders below the bar (§7aa). The four external proposals brought in since
— regulation, network, stand ordering, the LTFM calibration — are all measured negative, and
the two data-side ones (no-NM recovery, the regime correction) are blocked on an input the
file does not contain.

**What the path needs is evidence, not design.** Every one of the nine rows in §1 fails on
the *quantity* of fold months or the *absence* of a column, and not one of them fails on a
form nobody has tried. That is the honest shape of the close: the lane is not out of ideas,
it is out of **months** and out of **inputs**, and both of those are other lanes' to produce
(§7q option C: the v10 panel, the stack-5 vector and `exp-035-k2climb`'s held-out refit,
about seven member refits and one to two worker-days).

#### 4. The honest repairs made before each FAIL was written

Not one class was closed at its first negative. Recorded, in the order they were made:

- **the shrunk head** is itself the repair for the per-cell form — §7h named the variance of
  the difference as the way to revive it, and shrinking did exactly what it promised
  (january's half-width **±155 → ±81** at an unchanged point estimate) and stopped one
  1.02× short; the weight 0.5 was registered before the read and never searched, and a
  weight search on the same three folds was refused for that reason.
- **the regulation + network arm** was rebuilt in its only servable form after the label form
  was found unwritable, then re-priced per cell and per reach class (`N_reg_reach`), which is
  where its one surviving MAYBE lives.
- **the stand-ordering test** was re-anchored after its first definition proved vacuous, and
  both vacuous anchors were kept in the record (`analysis/stand-clip-vacuous.json`) rather
  than deleted.
- **the no-NM census** was re-priced twice — the regime gap moved from the desk's literal
  +6.32 s to **+1.19 s** on the recoverable rows themselves and **+0.34 s** trimmed — and
  then tested with the direct proxy read that found the key's defect.
- **the no-NM anatomy's separator search** was corrected when it chose a degenerate column,
  and the operator candidate it did find was cross-fitted rather than accepted.
- **the LTFM calibration** was repaired the way §7ah names — airport level anchored at zero
  so only the group deviations move — and still failed on every arm, best **−19.5 [−32.5,
  −7.3]**.

#### 5. The strongest refinement still visible — named, costed, with its face

**Tighten the arrival-side key and re-read the same proxy RMSE** (§7ae's own next test).
**Superseded 2026-09-09: run in §7aj and it returns NOISE** — the clauses repair the key
(recovered-clock RMSE 2,173 / 2,654 / 1,816 → 453 / 702 / 303) and remove 89 % of the face
with it, because the wrong-key rows and the head's worst rows are the same rows. The
paragraph below is the case as it stood before that read. Require the arrival's `AIRCRAFT_TYPE_mvt` to agree with the departure's, require
the implied flight time to sit in a plausible band for the city pair, and **drop** every
ambiguous key instead of keeping the earliest.

- **Kind of work:** measure. **Cost:** minutes — the same census script with three more
  clauses, no fit, no refit, no other lane's compute.
- **What it can win:** the recovered clock's RMSE on the surviving rows. If it falls to the
  file clock's own **350–466 s**, the channel is a rider and the surviving count is its
  size; the face it is reaching for is **+1.19 s at ×1.2** untrimmed on the recoverable rows
  and **+0.34 s** trimmed — both above the 0.1 s bar, which is why this is the strongest
  thing left and not the LTFM cell.
- **What it gives up:** it is the only item on the list that can be run without another
  lane, so running it costs nothing else; if it fails, the no-NM channel closes with it and
  §7ae's "REAL but small" becomes "noise", which is information either way.
- **Why it is the strongest and not the shrunk head:** the shrunk head is 1.02× short on
  evidence this lane cannot manufacture, and its face is +0.026 s. This is 30× the face and
  the evidence is already on disk.

#### 6. Revival conditions — first-class, one per class

| what would revive it | which classes it reaches | who produces it |
|---|---|---|
| **a fourth fold month** (2025-02, a second winter reading beside january) | **A** the shrunk head and **D** the per-cell form, and most of the twelve revivable held cells — the mechanism is that january's head is then fitted on three months, not two, so its correction is less noisy and its interval tighter | exp-039-screens, exp-038-packet and **exp-035-k2climb** (the expensive one, a held-out member refit); deferred under recorded rule (§7s) |
| **a column that reads the missing input itself** — the flight table's off-block for the no-NM core | the no-NM regime correction, and the 4,895 core ranking rows behind it | §7ae's tightened key is the only route in the record, and §5 above is its next test |
| **an LFPG january harvest** | `LFPG/unreached` (**32,822 ranking rows**, the largest held cell) and the LFPG feeder generally — the feeder switches on only at 2025-06-22 (0 % before, 1.6–7.9 % after) and is not registered in `INSTRUMENT_ARRIVAL`, so january is dark by instrument and the cell is arm-held, not evidence-held | exp-045-harvest |
| **december LEBL evidence** | `LEBL/reached`, the closest MAYBE (+0.0371 s), which clears july and january alone and fails december at −27.14 after two independent re-prices | exp-045-harvest — and note a fourth fold month does **not** reach this cell, because 2025-02 is not december |
| **a stand identifier that splits a stand area into exclusive positions** at LSZH, EHAM, EDDM, plus a next-occupant identification that does not rest on a turnaround threshold | stand ordering | not in the record; the sibling's turnaround chain places 1,170 / 861 / 731 fold rows against the 12,165 this test needed |

#### 7. Exit interview (COMPETING.md)

**Would a motivated human continue?** Yes — on this population, and not on this design. The
destination is +6.651 s at ×1.2 and the lane has already taken two board readings out of it.

**On what belief?** That the binding constraint is **evidence, not method**: seven of nine
negatives are one sentence — a two-month conditional mean does not transfer to a third — and
the in-sample oracles say the signal is there. A human would spend the next effort buying
the fourth fold month and the LFPG january harvest, not designing a tenth arm.

**What evidence would revive it?** §6's table, in that order: the tightened key first
because it costs minutes and is 30× the face of anything else here; then 2025-02; then LFPG
january. Any one of the three turns a MAYBE in this record into a gate read.

#### 8. What this lane does now

**HOLD.** No serve, no fit, no upload. The handshake is armed. **The next serve reads the
v18 class vector, never v16** — v16 would write the 1,786 handover rows back. The board
stays at v17 (`4de323fc…`, 246.6571) and v18 is the desk's held file, not this lane's.

**Files.** This section quotes only: §7z, §7q, §7aa, T7g, T7l, T7m, T7n, §7ac, §7ad, §7ae,
§7af(e), §7ag, §7ah, and `exp-046-surfacefit` §18.2 (read only). No new record is written.

---

### 7aj. The tightened arrival-side key — the refinement §7ai named, run. NOISE

The desk, 2026-09-09, on reading §7ai: run §5 now. The three clauses are the desk's, and the
read rule was declared before the run (`work/nonm_key.py`, `analysis/nonm-key.json`):

* **(a)** the arrival's `AIRCRAFT_TYPE_mvt` equals the departure's;
* **(b)** the implied flight time is inside a plausible band **for the city pair**, the band
  declared from the **clocked** rows of the same pair before the read and **cross-fitted by
  fold**;
* **(c)** every ambiguous key **dropped**, never the earliest kept — which is what §7ae did.

**The band, declared and never searched.** A reference flight is a departure whose
`FLIGHT_ID_mvt` also appears on an arrival at its own `ADES_mvt` — **the NM join itself, not
the flight-number key**, so the reference cannot inherit the defect it is testing for. Its
air time is the arrival's movement time minus the departure's. Per (ADEP, ADES) pair the
band is the 5th–95th percentile widened by 900 s each way, on pairs with at least 20
reference flights; a thinner pair has **no** band and its candidates are dropped, because
inventing a band for a thin pair is the same mistake as keeping the earliest candidate.
Cross-fitted means a fold's bands come from `arms.ROTATION`'s other fold month(s) only and
the ranking months' from all three fold months — no month's band is read off its own rows.

**The funnel. Departures surviving each clause:**

| month | no-NM to one of the ten | key + day + lands after | arrival has NM | same origin | usable clock (**the §7ae set**) | **(a)** same type | **(b)** in band | **(c)** unambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| july_loo | 583 | 356 | 203 | 200 | **198** | 172 | 168 | **168** |
| december | 284 | 174 | 70 | 69 | **69** | 56 | 55 | **55** |
| january | 290 | 149 | 66 | 66 | **66** | 60 | 59 | **59** |
| 2026-01 | 445 | 207 | 108 | 106 | 106 | 88 | 87 | **87** |
| 2026-07 | 346 | 225 | 113 | 113 | 113 | 91 | 90 | **90** |

**Clause (a) does all the work; (c) has nothing to do.** The type test removes 26 / 13 / 6
fold rows and 18 / 22 ranking rows; the band removes 4 / 1 / 1 and 1 / 1; and **no departure
anywhere is ambiguous after (a) and (b)** — 0 dropped in every month, one january row on a
pair too thin to band. So the ambiguity §7ae reported was ambiguity between candidates the
first two clauses already separate.

**Read 1 — the key defect is largely REPAIRED.** On the surviving rows the recovered clock
reads:

| fold | survivors | recovered-clock RMSE, **§7ae → tightened** | median(proxy − y) | rows off by > 600 s |
|---|---:|---|---:|---:|
| july_loo | 168 | 2,172.56 → **453.17** (4.8×) | −116.5 s | 14 of 168 |
| december | 55 | 2,653.51 → **702.29** (3.8×) | −117.0 s | 15 of 55 |
| january | 59 | 1,815.88 → **303.16** (6.0×) | +34.0 s | 4 of 59 |

Against the declared target — the file clock's own **350–466 s** (exp-043 §4) — january's
303 s is **below** the band, july's 453 s is **inside** it, and december's 702 s is above it.
**The desk's hypothesis was right**: the wrong-rotation matches were most of the damage, and
three clauses that cost minutes remove them.

**Read 2 — and it does not matter, because the same clauses remove the face.** The head's
RMSE **on the surviving rows** is not the 1,125 / 695 / 570 the channel was priced against:

| fold | head RMSE on §7ae's recoverable set | head RMSE on the **survivors** | the dropped rows | their share of the set's Σr² |
|---|---:|---:|---:|---:|
| july_loo | 1,124.68 | **336.02** | 30 | **92.4 %** |
| december | 694.91 | **376.92** | 14 | **76.5 %** |
| january | 569.59 | **405.39** | 7 | **54.7 %** |

**The key's error and the head's error are the same rows.** A no-NM departure whose flight
number matches a later leg is exactly a departure the head gets wrong, so tightening the key
to fix the proxy deletes the gap the proxy was to be paid for. The face follows:

| | rows | pooled gap vs with-NM | s²/ranking row | board s | **face ×1.2** |
|---|---:|---:|---:|---:|---:|
| §7ae, before the clauses | 219 | +799,913 | +508.0 | +0.9923 | **+1.1908** |
| **after all three clauses** | **177** | **+87,201** | **+44.8** | +0.0873 | **+0.1047** |

**81 % of the rows kept, 89 % of the face gone** — per fold the gap falls +1,209,143 →
+57,154, +445,080 → +104,254 and +285,097 → +125,001 s². What is left sits **at** the 0.1 s
bar, not above it, and that is a ceiling, not a measurement.

**Read 3 — the rider, priced.** `2A − B` per surviving row with `r = y − p12`,
`d = proxy − p12`, day-clustered, 2,000 draws at seed 47; the shrunk arm's weight 0.5 was
registered in the script before the read, as at T7m. Of the survivors 36 / 12 / 20 are in a
served cell.

| rider | july_loo (168) | december (55) | january (59) | every fold clear? |
|---|---|---|---|---|
| **substitution** | **−92,452** [−180,805, −22,528] | **−351,138** [−789,888, −101,759] | +72,436 [−4,860, +138,433] | **no** |
| **shrunk, w = 0.5** | −4,271 [−42,959, +31,457] | **−116,451** [−295,437, −21,340] | **+72,765** [+15,994, +117,674] | **no** |

**Verdict: NOISE.** The registered rule (§13.1 / recorded rule) asks for an interval clear above zero
on every fold; the substitution is clear **below** zero on two of three, and the shrunk arm
is clear below zero on december and clear above on january. Both fail in the strong sense —
not a MAYBE, which needs a positive point estimate on every fold. The one fold that wins is
january, the fold where the proxy beats the file clock's own accuracy (303 s), and it is
outvoted by december, where the proxy still carries 15 tail rows in 55.

**What this closes, and what it does not.** It closes **the recovery as a rider at this
design**: the key can be made correct, and correcting it leaves 177 ranking rows carrying
+0.10 s of ceiling and a measured rider that is negative on two folds of three. It does not
close the no-NM core as a population — §7af's 4,895 core ranking rows and their ≈480–540 s
RMSE are untouched by this test, and their revival condition is unchanged: **a column that
reads the missing input itself**, which the arrival row does not deliver even when its
identity is right. The arrival-side route has now been measured twice, and this reading is
the more informative of the two: the first said the key was wrong, and this one says the key
can be fixed and the fix removes the reason to want it.

**§7ai §5 is superseded by this section.** The strongest refinement it named is now run and
returns noise; the lane's remaining revival conditions are §7ai §6's other four rows, and
none of them is this lane's to produce.

**Files.** `work/nonm_key.py`, `analysis/nonm-key.json`, `logs/nonm-key.log`. No fit of the
head, no serve, the handshake stays armed.

---

### 7ak. The blend lineage's MAYBEs, re-priced over the served head — the spec, written before the read

The desk, 2026-09-09 (absence grant, *re-test old negatives on the changed stack*): the
blend lineage's MAYBEs were priced over the **pre-surface** head, and v17's head has since
removed board seconds of residual, so their increments are **new measurements**, not
re-reads. Screen: (1) which closed MAYBE members have fold-tagged OOF vectors on all three
folds and both ranking months, each named with its path and sha; (2) for each, the increment
over the **served v17 head** at an amplitude fitted on two folds and priced on the third,
rotated, per fold with intervals, all-row and clean, and the ×1.2 face.

**THE ARITHMETIC, PRE-DECLARED. Nothing below this paragraph had been read when it was
written and committed.**

1. **The served head, per fold.** `p_srv = p12 + d_N` on the rows v17 serves and `p12`
   elsewhere, with `d_N` cross-fitted by fold exactly as §7aa built `d_new` — arm `N`,
   ridge, `serve.FIT_FOLDS` leave-one-out, the handshake blocks, `serve.served_mask` and
   the three supported cells of `arms-step2-handshake.json`.
2. **The limit, declared here and not discovered later.** `p_srv` carries **this lane's**
   part of v17 and **not** `exp-046-surfacefit`'s with-clock correction, which has **no
   fold-side vector on disk** (checked: its code persists blocks and class vectors, never a
   fold prediction) and which this lane may not refit. So every arm is priced **twice** —
   over `p_srv` and over bare `p12`, the pre-surface head the old numbers used — and the
   **difference between the two columns is the measurement the desk asked for**; the missing
   third piece can only push the same way, since it removes more residual still.
3. **The amplitude.** For member `m`: `d = m − p_srv` and `r = y − p_srv`. For fold `f` the
   amplitude is fitted on the **two other folds pooled**, `a(f) = Σ r·d / Σ d·d` over those
   rows only, and never on `f`. No clipping, no constraint on sign, no search.
4. **The price.** `gain(f) = mean(2·a(f)·r·d − a(f)²·d²)` on fold `f`, the exp-038 §1
   contract with `d ← a·d`, in s² per row; the interval is the day-clustered bootstrap,
   `BOOT = 2000` at `SEED = 47`. Clean RMSE is `prc.data.rmse_clean` before and after on the
   same fold. The face pools the fold gains at `SHARES` (december reported beside, as
   always), converts on `V12_BOARD_CLEAN` and is shown at ×1.2.
5. **The gate.** The registered one: a **rider** only if **every** fold interval is clear
   above zero. A fold that crosses is a **MAYBE** and its revival condition is named. The
   amplitude is selected on the folds, so this is the **strict** reading and no pooled
   number can rescue a crossing fold.
6. **No serve, no refit of any member, no upload.** Every member vector is read from disk as
   it stands.

**Record:** `work/blend_reprice.py`, `analysis/blend-maybe-repricing.json`.

**(1) The survey.** `experiments/exp-*/oof/` holds **81** fold-tagged families, **58** of
them complete on all three folds. Of those, the record holds **three** as MAYBE, and they
are the only three priced below; everything else complete on three folds is a shipped member
(`cand-k2c-mod-refit-bulk` is `base.V12_HEAD_OOF`, the head itself), a control, or a closed
FAIL — `exp-044-zerosum`'s two ridge families were checked and are a measured FAIL at their
own step-3 gate (LEDGER row 417), not a MAYBE.

| member | vectors | sha256 (july_loo / december / january) | ranking months | held by |
|---|---|---|---|---|
| **`K3-tail-gated`** | `experiments/exp-032-k3/oof/k3-tail-gated-{fold}.parquet` | `d8133ada…` / `3356e230…` / `4514925f…` | **none** | BACKLOG line 17, exp-032 §7d |
| **`B1-bulk64`** | `experiments/exp-034-bulk/oof/bulk64-{fold}.parquet` | `a138144c…` / `8edb01f5…` / `69702375…` | **none** | BACKLOG line 19, exp-038 §4 |
| **`exp-040-chain`** | `experiments/exp-040-chain/oof/chain--{fold}.parquet` | `e1697eca…` / `990e7af4…` / `57826cfb…` | **both** (`chain--ranking.parquet`, 344,841 rows; `none--` and `derived--` beside it) | LEDGER rows 441 / 449, HELD under recorded rule |

**Only `exp-040-chain` answers the desk's question with a yes.** The other two have three
folds and **no ranking vector at all** — serving either needs the 12-month ranking fit the
BACKLOG already prices at about an hour plus a block build, which is exactly the cost that
made them MAYBEs. Three fold vectors on disk elsewhere carry a ranking twin
(`exp-013-nonm`, `exp-029-january`, `exp-040-chain`); only the last is a live MAYBE.

**One reading, taken and recorded.** `bulk64` is a **regime** member: it predicts on the
bulk rows only and is null on 3,938 / 1,649 / 1,435 rows. Taken: where a member has no
prediction it **moves nothing** (`d = 0`), which is what an additive member does under the
shipped per-regime rule and which keeps the fold's population fixed so the RMSE and the face
stay comparable. Rejected: dropping those rows, which prices the member on a different
population from the head it is priced against.

**(2) The fold table — every arm, over the served head, and the same arm over bare `p12`.**
Gains are s² per row with the day-clustered 95 % interval; `a` is the amplitude fitted on
the other two folds.

| member | head | july_loo | december | january | pooled | face ×1.2 all-row / clean | verdict |
|---|---|---|---|---|---:|---|---|
| **`K3-tail-gated`** | **served** | +48.48 [−12.3, +133.0] `a` 0.013 | **−252.95 [−411.7, −106.6]** `a` 0.136 | +81.39 [−126.3, +396.7] `a` 0.080 | +63.05 | +0.1227 / +0.1640 | **fails** |
| | bare `p12` | +33.45 [−7.9, +91.2] | **−259.78 [−416.1, −114.9]** | +77.56 [−129.9, +391.6] | +52.99 | +0.1030 / +0.1403 | fails |
| **`B1-bulk64`** | **served** | +16.76 [−7.0, +42.8] `a` **−0.015** | −9.12 [−52.3, +32.2] `a` **−0.040** | +5.53 [−18.9, +31.1] `a` **−0.033** | +11.79 | +0.0229 / +0.0276 | **fails** |
| | bare `p12` | +21.32 [−9.0, +54.0] | −7.12 [−52.4, +37.1] | +6.69 [−19.5, +34.3] | +14.84 | +0.0289 / +0.0347 | fails |
| **`exp-040-chain`** | **served** | +74.42 [−211.4, +478.2] `a` 0.059 | −6.78 [−102.4, +79.2] `a` 0.075 | **+61.68 [+1.3, +125.7]** `a` 0.057 | +68.78 | **+0.1338 / +0.1485** | **fails** |
| | bare `p12` | +78.33 [−207.0, +482.4] | −8.60 [−104.7, +79.4] | **+62.56 [+0.8, +126.5]** | +71.34 | +0.1387 / +0.1540 | fails |
| *`chain`'s matched control `none`* | served | −26.89 [−129.5, +111.8] | +0.08 [−0.2, +0.4] | −37.01 [−105.5, +12.5] | −31.37 | −0.0610 / −0.0727 | control |

**No arm is a rider. December kills all three, and it kills them in three different ways.**

* **`K3-tail-gated` is not a MAYBE any more; it is measured NEGATIVE.** Its december
  interval is **clear below zero** at −252.95 [−411.7, −106.6], and the BACKLOG row it
  carries says december was *the one fold whose interval excluded zero* — on the positive
  side (+0.058 [+0.023, +0.093] in that row's units). **The fold that carried the MAYBE is
  now the fold that kills it.** The mechanism is in the amplitude column: the member's worth
  is not stable across folds, `a` reads **0.013 / 0.136 / 0.080**, so the amplitude december
  inherits from july and january is an order of magnitude above what december itself wants,
  and applying it costs 253 s² a row. An amplitude fitted on two folds is exactly the
  serving situation, so this is not an artefact of the design — it is the design working.
* **`B1-bulk64`'s out-of-fold amplitude is NEGATIVE on every fold** (−0.015 / −0.040 /
  −0.033): the two-fold fit wants to move **away** from the member, and the small positive
  gains on july and january are the head being nudged back off whatever the member
  over-weights. Every interval crosses zero, the face is +0.023 s at ×1.2 — a quarter of the
  bar — and nothing here is a member addition.
* **`exp-040-chain` is the strongest of the three and still fails.** Two folds are positive,
  january clears on its own (+61.68 [+1.3, +125.7]), and the pooled face is **above** the
  0.1 s bar at +0.1338 s all-row and +0.1485 s clean. But july's interval is enormous
  (±345) and december is negative with an interval that spans zero — not established in
  either direction, which is a crossing and not a measured loss. Its matched control (the
  same recipe with the chain features removed) reads **−31.37** pooled, so the chain
  features themselves carry about **+100 s² a row** over their own control; what fails is
  the vector as a candidate, not the idea behind it. **recorded rule's hold stands and this re-price
  does not lift it.**

**The measurement the desk asked for — what the better head did to the old numbers — and a
correction to this section's own spec.** Over the served head against bare `p12` the pooled
gain moves **+52.99 → +63.05** for `K3-tail-gated` (**up** 10.06), **+14.84 → +11.79** for
`B1-bulk64` (down 3.05) and **+71.34 → +68.78** for `exp-040-chain` (down 2.56). **The spec
paragraph above said the missing surfacefit piece "can only push the same way"; the
measurement does not support that** — the sign of a head improvement's effect on a member's
increment is **not uniform across members**, so the third piece cannot be signed by
argument, only measured. What the two columns do establish is that **this lane's part of v17
is not what decides any of these three**: it changes 38,608 / 24,773 / 24,527 rows at an RMS
of 9.4 / 7.7 / 7.6 s, and moves every verdict not at all.

**So the honest scope of this reading.** Over the head this lane can build, none of the
three MAYBEs revives, and each fails on december for its own reason. The part of v17 this
lane **cannot** build — `exp-046-surfacefit`'s with-clock correction, which has no fold-side
vector on disk — is the larger part of the head improvement the desk's premise names, and it
is the sibling's to price. **Revival conditions, one per member:** `K3-tail-gated` needs a
december reading that does not reverse under an amplitude fitted elsewhere — a fourth fold
month is not december, so this one waits on december evidence, like `LEBL/reached` (§7z);
`B1-bulk64` needs a positive out-of-fold amplitude before anything else is worth asking;
`exp-040-chain` needs july's interval narrowed or december's sign settled, and it is the
only one of the three whose ranking vectors already exist, so it is also the only one whose
revival costs no refit.

**Files.** `work/blend_reprice.py`, `analysis/blend-maybe-repricing.json`,
`logs/blend-reprice.log`. No serve, no refit of any member, no upload.

---

### 7al. An independent provenance audit of surfacefit's §28.2 bundle read — four AGREE, one AGREE with a wording exception

The desk, 2026-09-09, before tonight's upload: round 6 failed on exactly this class — a head
vector built at other pins — so audit the §28.2 bundle read from the committed artifacts and
**this lane's code, not theirs** (`work/bundle_audit.py`, `analysis/bundle-audit.json`).
**Read-only on `exp-046-surfacefit`'s files: no module of theirs is imported and nothing of
theirs is written.** Every number below is rebuilt from this lane's `base.load()` panel and
their two committed `.npz` vectors. That is possible because their `pbase` **is** `p12`
(`exp-046-surfacefit/code/base.py:105`), built by the same two lines this lane builds it
with — so if the panel and the head agree, the arithmetic can be redone without their code
at all, which is the strongest form this audit can take.

**(a) The baseline — AGREE, with one wording exception that the record discloses elsewhere
and §28.2 does not.** The baseline is
`work/d-passonly-6fdf3951-guard995.npz`, sha **`e137939b74070180192544134b28a2ed49011071aa29449598fb9de87953029a`**;
the bundle is `work/d-bundle-ungated1299-guard995.npz`, sha **`5dbbcb8c…`**. It is the
**pre-pass** arm: `analysis/pass-only.json` names the three caches it was fitted on, and all
three carry the `-6fdf3951` suffix — the matcher v17 was served on, which this lane's §7aa
independently records as the matcher of `artifacts/exp-047/block-prev/`. Two of the three
are at this serve's pins; **december's is not**:

| fold | the head's cache | this serve's pin | at this serve's pin? |
|---|---|---|---|
| july_loo | `block-2025-07-8179f74417c8cb99-6fdf3951` | `8179f744` | **yes** |
| january | `block-2025-01-cb4d6289cb421aa2-6fdf3951` | `cb4d6289` | **yes** |
| december | `block-2025-12-34a155e40aac9ff6-6fdf3951` | **`d35b7171`** | **no — one population behind** |

§25.3 states this plainly and `code/chainbundle.py`'s docstring repeats it. **§28.2's prose
does not**: it calls the baseline "the pre-pass arm at this serve's own pins", full stop,
two lines above `every_fold_interval_clear` is TRUE. That is a wording exception, not a
number error, and it is exactly the class round 6 failed on, so it should be repaired in the
prose before the upload.

**And it is BOUNDED, measured here rather than argued.** The nine extra airport-days are all
**2025-11-30** — a *prefix* population, not scored rows — so they can reach december only
through the neighbourhood features of rows near the month boundary. They do not:

| december, boundary days dropped | rows | row share | share of the fold's gain | gain without them | interval |
|---|---:|---:|---:|---:|---|
| 2025-12-01 | 5,496 | 3.32 % | **3.13 %** | +93.38 | [+46.90, +144.85] |
| + 12-02 | 10,503 | 6.34 % | **2.56 %** | +96.96 | [+45.35, +151.45] |
| + 12-03 | 15,728 | 9.49 % | **6.31 %** | +96.47 | [+40.79, +154.79] |

The gain is carried in **proportion to the rows**, with no concentration at the boundary,
and december clears with the first three days removed. Add that `base.SHARES` is july_loo
and january only — both at identical pins — and the **board figure cannot be touched by this
gap at all**.

**(b) The pins — AGREE, verified from this lane's own manifest and not from theirs.**
`analysis/pin-manifest.json` (written by this lane) and their `analysis/fold-population.json`
agree on every field:

| fold | claimed | their `block_cache_key` | their airport-days | **my** airport-days | **my** block rows |
|---|---|---|---:|---:|---:|
| july_loo | `8179f744`, 296 | `8179f74417c8cb99` | 296 | **296** | 190,713 |
| december | `d35b7171`, 297 | `d35b7171d7062341` | 297 | **297** | 165,677 |
| january | `cb4d6289`, 218 | `cb4d6289cb421aa2` | 218 | **218** | 153,706 |

December's 297 is this lane's own `previous_airport_days` 288 plus the nine 2025-11-30
airport-days named in `pin-manifest.json::admitted_now` — the same nine §25.3 names, reached
from the other side.

**(c) The counts — AGREE, and the reading has to be named.** Recomputed from the row-level
arrays, `(d_bundle − d_head) ≠ 0` per fold:

| | july_loo | december | january |
|---|---:|---:|---:|
| **claimed** 88,348 / 102,073 / 69,839 | 88,348 | 102,073 | 69,839 |
| recomputed over **v18's own arm** (§28.1) | **88,348** | **102,073** | **69,839** |
| recomputed over the **served head** (§28.2) | 88,262 | 101,928 | 69,839 |

Both sets reproduce their JSON exactly. **The desk's three numbers are §28.1's reading, not
§28.2's** — the bundle moves 86 fewer rows on july and 145 fewer on december, because those
rows sit where v18's arm and the pre-pass head already agree. Nothing is wrong; the label
matters, since §28.2's table carries no moved-row column of its own. Rows added to the class
are **897 / 858 / 742** in `chain-bundle.json` and identically `kept_at_the_loosest_gate` in
`chain-fold.json`, and the mode-3 reference rows are unchanged at **1,170 / 861 / 731**, which
is the no-op guard the ungated pass had to satisfy.

**(d) The numbers, re-derived — AGREE, and every fold clears in my read.** Point estimates
rebuilt from this lane's panel reproduce **exactly**, to a difference of `0.0`:

| fold | my gain | theirs | my 2,000 draws (seed 47) | my 20,000 draws | worst lower bound of ten seeds | their interval |
|---|---:|---:|---|---|---:|---|
| july_loo | **136.66889** | 136.66889 | [+81.99, +189.60] | [+83.39, +191.38] | **+80.24** | [+85.09, +189.98] |
| december | **93.19858** | 93.19858 | [+45.34, +143.74] | [+44.57, +144.07] | **+42.11** | [+42.49, +143.23] |
| january | **87.26048** | 87.26048 | [+46.30, +128.84] | [+47.79, +128.36] | **+46.71** | [+47.99, +127.16] |

The board rebuilds to base **299.33900 → arm 299.14720**, gain **+0.19180 s** all-row and
**+0.23414 s** clean — five decimals against their `0.19179638…` and `0.234138…`. Their seed
is 46 (`analysis/predeclared.json`) and this lane's is 47, so the interval ENDS are a
different draw of the same estimator and cannot be expected to match; what matters is that
**no lower bound is anywhere near zero under any of twelve independent draws per fold**, the
smallest being december's +42.11. **`every_fold_interval_clear` is TRUE in my read as well.**

**(e) The threshold — AGREE.** `1,299 s` is a literal in their code and nothing else:
`THR = 1299.0` in `chainbundle.py`, `THRESHOLDS = [1299.0, 900.0, 600.0, 300.0]` in
`chainfold.py` behind a comment naming it "the shipped p95 of section 27", and
`check_i_abs_dt_offblock_max_s: 1299.0` in `serve.py`. **`chainfold.py` contains no quantile
computation at all**, so no fold can have chosen it. The same holds for step 0: its cut is
**62.405 s on all three fold months identically**, which is what a value carried from the
ranking side looks like and what a fold-chosen one could not.

**Verdict: AGREE on (b), (c), (d), (e); AGREE on (a) with one wording exception.** The
arithmetic is right, the vectors are the ones the record names, the pins are what they are
claimed to be on two folds of three, and the one gap — december's head cache one population
behind — is disclosed in §25.3, is bounded here to a proportional 3 % of the fold's gain,
and cannot reach the board figure at all. **The repair this audit asks for is one sentence
of prose in §28.2, not a re-run.**

**Files.** `work/bundle_audit.py`, `analysis/bundle-audit.json`, `logs/bundle-audit.log`.
Nothing of `exp-046-surfacefit`'s was written; no serve, no upload.

---

### 7am. The v5 candidate audited against v17 and the v19 class vector — AGREE on every item

The desk, 2026-09-09: §29 landed, so read `submissions/candidates/exp-046-surface-v5.parquet`
with this lane's own code against the file on the board and the class vector a later serve
must complement, and read `code/poolrecon.py`'s reconciliation of the 2,493 → 2,456 chain
count (`work/v5_audit.py`, `analysis/v5-audit.json`). **Read-only on
`exp-046-surfacefit`'s files; nothing served, nothing uploaded.**

**(a) The candidate — AGREE.** sha256
`4d75706bfcc13d88f7556265cd008b3d3a312555955dca8eb528e4db5325fee8`, the desk's byte for
byte, against v17's `4de323fc4fd9d4865b201b0dd0da1ddcbf7de9e11f1d9bc8af07d293585001ea`.

**(b) The changed rows — AGREE, and the subset test passes exactly.**

| | measured | claimed |
|---|---:|---:|
| rows changed against v17 | **192,435** | 192,435 |
| `occupied_by_exp_046` on v19 | **193,910** | 193,910 |
| **changed rows outside occupied** | **0** | — |
| rows not occupied | **150,931** | 150,931 |
| of them byte-identical to v17 | **150,931** | — |
| class rows missing for a template key | **0** | — |

**Every changed row is inside `occupied_by_exp_046`, and every unoccupied row is
byte-identical to v17.** The 1,475-row gap between 193,910 occupied and 192,435 changed is
not an anomaly: those are rows this lane writes where its own value equals what v17 already
carried. The moves are ordinary — RMS 32.67 s, largest 1,019.41 s.

**(c) Occupied-unwritten — AGREE.** `written_by_exp_046` equals `occupied_by_exp_046` row by
row: 193,910 each, **0 occupied-and-not-written** and **0 written-and-not-occupied**. That is
the property the handshake rule depends on — the column leaves every row owned exactly once.

**(d) The chain rows — AGREE, on the reconciled count and not the old one.** `chain_mode_4`
holds **2,456** rows, **2026-01 1,304 and 2026-07 1,152**, matching surfacefit's figures
exactly and **not** §27.3's 2,493. Of them **2,171 are written** and **285 are held, all 285
at LFPG** — so written + LFPG-held = 2,456 with nothing unaccounted. One detail the summary
line does not carry: **2,170 of the 2,171 written chain rows actually change value**; the
remaining one is written with a value equal to v17's, the same benign case as the 1,475
above.

**(e) The `noac` hold — AGREE.** 388 rows, **0 changed**, maximum absolute difference against
v17 exactly **0.0**.

**(f) LFPG — AGREE.** 39,872 LFPG ranking rows, **0 changed**, **0 occupied**, and the 285
held chain rows are the whole of LFPG's contact with this candidate.

**(g) The template — AGREE.** 344,841 rows in the submitting template's own order (v17 too),
the two expected columns, **no duplicate key**, **no non-finite value**, minimum 60.0 s and
maximum 115,306.6 s, no row at or below zero.

#### The 2,493 → 2,456 reconciliation: do the fold pricing and the serve use the same metric?

**AGREE — they do, and I verified it in their code rather than taking the note.** The claim
in `code/poolrecon.py` is that the difference is the runway-centroid distance metric. Traced
independently:

* `analysis/chain-check/block_ungated.py` computes `chain_runway_agrees` as
  `argmin over k of (lat − c_k.lat)² + (lon − c_k.lon)²` — **squared degrees**, an unweighted
  lat/lon plane.
* `code/chaincheck.py::apply_checks` takes `argmin` of `km(...)` — **great-circle
  kilometres**.
* `code/chainfold.py::build_month` calls `BU.build`, i.e. `block_ungated` — so the **fold**
  side is the plane metric.
* `code/serve.py::ranking_block` under `_UNGATED` calls the **same module object**
  (`_ungated_modules()` returns `CF.BU`) and restricts it with `chainfold.restrict` at the
  `UNGATED_GATE` constants — so the **ranking** side is the same plane metric, from the same
  code, at the same gate.

`analysis/pool-reconcile.json` is internally consistent with that: 2,439 kept by both, 17
only by the block, 54 only by the pool, net −37; **all 54 fail check (ii) alone** — zero fail
step 0, zero fail check (i), zero are absent from the pool. A metric-only difference is
exactly what that pattern looks like, and it is concentrated in 2026-07 (50 of the 54 and 16
of the 17), as a boundary artefact between two runway centroids should be.

**So the pricing is sound: one class, one metric, on both sides.** The caveat is about the
*other* number, and it is worth one line in their prose. The shared metric is the **defective
one** — at 50° N a degree of longitude is about 0.64 of a degree of latitude on the ground,
so the plane over-weights longitude — and §27.3's yield of **2,493**, its π̂ and its
**+0.0927 s at ×1.2** were measured on the km-metric row set, which is **not** the set being
served: 54 of those rows are not in the candidate and 17 rows in the candidate are not in
them. The gap is 1.5 % of the class and the correction would require the whole §28 refit,
which the desk's order rules out for this candidate — so this is a **prose caveat, not a
blocker**: any §27.3 face figure quoted for what is being uploaded is a figure for a
slightly different 2,493 rows.

**Verdict: AGREE on (a) through (g) and AGREE on the metric reconciliation**, with the one
recorded caveat that §27.3's yield and face were measured under the other metric.

**Files.** `work/v5_audit.py`, `analysis/v5-audit.json`, `logs/v5-audit.log`. Nothing of
`exp-046-surfacefit`'s was written; no serve, no upload.

---

### 7an. PRE-REGISTRATION — the expected board move for v18, written before the upload

The desk, 2026-09-09, before the upload: register the expectation **before** BOARD READING
17, which is what score-feedback rule asks for — a prediction that arrives after the score is not a
prediction (`work/v18_prereg.py`, `analysis/v18-preregistration.json`). **Nothing is fitted
here.** It is the lane history applied to the candidate's own registered face.

**The face being registered.** `analysis/serve-v19.json::gate.increment_over_the_head` —
the increment over the served head with the LFPG hold applied, the number the gate passed
on: **+0.20534 s all-row [+0.14008, +0.29274]**, clean **+0.25067 [+0.18264, +0.31997]**,
every fold interval clear. The candidate is `exp-046-surface-v5.parquet`, sha
`4d75706b…`, audited in §7am. The board today is **246.6571** (v17), 14.274 s ahead of the
next team.

**The lane history — every upload since the surface programme began.** `face` is the fold
number registered *before* that upload; `transfer` is the measured board move divided by it,
the same definition BOARD READINGS 12–16 used. Every move below reconciles to the ledger's
own figure and every ratio recomputes to its rounding.

| upload | reading | lane | what it served | registered face | board | move | **transfer** |
|---|---:|---|---|---:|---|---:|---:|
| v13 | 12 | surfacefit | the aircraft's own observed surface clock | +8.1280 | 262.7418 → 248.4818 | 14.2600 | **1.75×** |
| v14 | 13 | **queue** | the no-clock neighbourhood on two served cells | +0.1854 | 248.4818 → 248.2904 | 0.1914 | **1.03×** |
| v15 | 14 | surfacefit | the 127-leaf head replacing v13's arm | +0.7296 | 248.2904 → 246.8263 | 1.4641 | **2.01×** |
| v16 | 15 | surfacefit | capacity 127 → 255 leaves plus the p99.5 guard | +0.3350 | 246.8263 → 246.7053 | 0.1210 | **0.36×** |
| v17 | 16 | **queue** | the lit-fold revival, `LEMD/reached` added | +0.0686 | 246.7053 → 246.6571 | 0.0482 | **0.70×** |

Transfers: **0.36, 0.70, 1.03, 1.75, 2.01** — median **1.03×**, mean **1.17×**, and the
spread is 5.6× from end to end. Surfacefit's own three are 0.36 / 1.75 / 2.01; this lane's
two are 0.70 / 1.03, the tighter pair and both near face.

**The expectation, four ways.**

| basis | board move | **expected v18 score** |
|---|---|---|
| at face, 1.00× on the point | +0.2053 | **246.4518** |
| at the median transfer, 1.03× | +0.2115 | **246.4456** |
| at the mean transfer, 1.17× | +0.2403 | **246.4168** |
| **the observed transfer range, 0.36×–2.01×, on the point** | +0.0739 … +0.4127 | **246.2444 … 246.5832** |
| the observed range applied to the face INTERVAL (the widest honest band) | +0.0504 … +0.5884 | **246.0687 … 246.6067** |

**Under the two most recent surface transfers — and the phrase has two readings, so both are
registered.** Surfacefit's *own* two most recent are v16 **0.36×** and v15 **2.01×**, which
are also the extremes of the whole record, giving **246.2444 … 246.5832**. The two most
recent uploads of *either* lane are v17 **0.70×** and v16 **0.36×**, a much tighter and much
more pessimistic band: **246.5134 … 246.5832**. The second is the narrower prediction and
the first is the one matched to the lane that wrote this candidate; **if the score lands
between 246.51 and 246.58 both readings are satisfied at once**, and that overlap is the
single most likely outcome on this evidence.

**What would count as a transfer failure, in one sentence.** *A transfer failure is a board
score above **246.6067** — a move under **+0.0504 s**, which is the lowest transfer ever
observed on this board (0.36×, v16) applied to the lower end of this candidate's own fold
interval (+0.1401 s) — and a score at or above v17's **246.6571** is not a weak transfer but
no transfer at all.*

**Two things this pre-registration deliberately does not claim.** It does not predict the
*direction* of the surprise: the five readings do not form a trend (1.75, 1.03, 2.01, 0.36,
0.70), and the desk's own currency note (LEDGER, BOARD READINGS 12–15 read together) says
the spread is not yet explained, so a wide band is the honest one. And **rank is not at
risk** on any of these bands — the next team is at 260.931, so every outcome above leaves
rank 1 by more than 14 s. Note also that §7am's recorded caveat touches the *size* of the
chain contribution, not this face: §27.3's +0.0927 s was measured under the other runway
metric, and the +0.2053 registered here is the fold-side gate figure, which was measured
under the metric that ships.

**Files.** `work/v18_prereg.py`, `analysis/v18-preregistration.json`. No serve, no upload —
the desk uploads.

---

## §7ao — check D, a yield-widening corroboration on the rows the chain gate refuses

**What this is, and what it is not.** ChatGPT's check D, taken by the desk (2026-09-09) as a
**yield-widening** test and not a tightening: the ungated probe admits **8,253** ranking
rows, the shipped `(i)&(ii)` gate keeps **2,456**, and this section asks whether a
**recurring-service** pairing corroborates a placement on the **5,797 refused** rows,
independently of (i) and (ii). Analysis only, for a LATER candidate and not for v18: no
class change, no serve, no `block.py` edit. Read-only on `exp-046-surfacefit`.

**The read rule, declared before the run** (`work/check_d.py`'s module docstring, committed
in the same shape as §7ak's spec paragraph): rows admitted by D alone per ranking month;
precision by the day-shift and stand-shuffle placebos on the same pool, in §27's arithmetic
(`π̂ = (q − f)/(t − f)`, `precision = π̂·t/q`, `f` the harsher of the two placebos, as
`chaingate.py` does) written in this lane's own code; the ×1.2 face projected at that
precision. Check D itself, also declared before the run: a **service key ladder** —
`airport|CALLSIGN_flt`, else `airport|operator|scheduled hour` — needing ≥ 3 **other** days
in the row's own ranking month; the service's **modal stand and modal runway over its other
days in the ranking file**, this row's day always left out; D passes iff the segment's
`runway_implied` equals the modal runway **and** the segment's start position is within
**TOL_KM = 2.2149 km** of the modal stand's centroid, TOL_KM being the **p95 on the shipped
3,179** (a quantile of the shipped set, never of the pool, never of a placebo). D reads this
row's AOBT nowhere and this row's own `RUNWAY_mvt`/`STAND_mvt` nowhere, so it is independent
of (i), (ii) and (iii) **by construction**. One declared design note: the shipped stand
placebo permutes a column D does not read, so it was **re-expressed** to corrupt what D does
read — the service table rebuilt on stand-permuted ranking departures at SEED 47.

**The declared read — KILLED, and it reverses.**

| | rows | D passes | rate | Wilson 95 % |
|---|---|---|---|---|
| `t` shipped 3,179 | 3,179 | 1,770 | **0.5568** | [0.5395, 0.5740] |
| `q` refused pool | 5,797 | 759 | **0.1309** | [0.1225, 0.1399] |
| `f` day placebo | 4,907 | 1,104 | **0.2250** | [0.2135, 0.2369] |
| `f` stand placebo | 11,432 | 981 | **0.0858** | [0.0808, 0.0911] |

`f = max(day, stand) = 0.2250 > q = 0.1309`, so **`π̂ = −0.2836`** and the precision is
negative. D admits 759 rows (2026-01 **388**, 2026-07 **371**) and the arithmetic says the
pool carries no excess of D-corroborated placements over a knowingly wrong placement. Under
the declared rule this is not a MAYBE and not a near miss: it is a **reversal**, and no face
is quotable from it.

**Why it failed — two causes, both measured, both post-hoc diagnostics.**

1. **D's runway clause is check (ii) re-derived from other days.** The service's modal runway
   **is the row's own `RUNWAY_mvt`** on **98.65 %** of the shipped rows and **98.43 %** of the
   refused. D is independent of (ii) by construction and a **copy of it in fact**. The refused
   pool fails (ii) on **3,913 of 5,797** rows (349 on (ii) alone, 3,564 on both), so the clause
   can only suppress the yield — of the 759 admitted, only **20** come from the (ii)-only class.
2. **The day placebo is not a valid null for a service check.** A +24 h shift destroys the
   clock completely (its check-(i) pass rate is **0.0000**) but barely touches the geography:
   on the shipped stand check (iii), at the shipped p95 cut of **2.8263 km**, it passes at
   **0.8895** — *above the refused pool's* **0.8397** and near the shipped rows' **0.9497**
   (medians 1.038 / 1.359 / 0.556 km). It lands the departure on the **same recurring service
   one day later**, which preserves the exact invariant D reads. §27 used it against a clock
   check, where it is the right null; it does not carry over to this one.
3. A third, smaller cause: the stand clause fires on only **57 %** of the *shipped* rows at a
   p95 tolerance, and the shortfall is **coverage, not tolerance** — the modal stand has no
   centroid in **39.7 %** of shipped and **58.2 %** of refused rows, in a map learned from
   3,179 rows at ≥ 3 rows a stand. The service key itself resolves on 99.6 % / 99.9 %.

**The repair the diagnosis asks for, and where it lands.** Drop the runway clause (it was a
copy of (ii)), widen the stand centroid map to ≥ 1 shipped row a stand, and read only the
rows a geometric corroboration could legitimately rescue — the **1,594** that pass step 0 and
(ii) and fail the **clock alone**.

| reading | `t` | `f` | `q` | `π̂` | precision | rows | face ×1.2 |
|---|---|---|---|---|---|---|---|
| **A** declared, harsher placebo | 0.5568 | 0.2250 | 0.1309 | −0.2836 | **−1.2063** | 759 | — |
| **B** day placebo set aside for cause | 0.5568 | 0.0858 | 0.1309 | 0.0958 | 0.4073 | 759 | 0.0128 |
| **C** stand half only | 0.5731 | 0.1098 | 0.3162 | 0.4455 | 0.8075 | 1,833 | 0.0614 |
| **D** C + wider map | 0.6713 | 0.1333 | 0.4345 | 0.5599 | 0.8650 | 2,519 | 0.0904 |
| **E** C + clock-only sub-pool | 0.5731 | 0.1098 | 0.3576 | 0.5349 | 0.8572 | 570 | 0.0203 |
| **F** C + wider map + clock-only | 0.6713 | 0.1333 | 0.4856 | 0.6548 | **0.9052** | 774 | **0.0291** |

Reading **F** clears the shipped bar 0.896 **as a point estimate**, on 774 rows, projecting
**+0.029 s** at ×1.2 — about a third of the shipped rung's own +0.0927 s. It also holds when
the pool is split by ranking month, which is two populations with their own services and
their own geography: 2026-01 **0.8987** on 397 rows, 2026-07 **0.9088** on 377.

**Why F is a MAYBE and not a pass.** The lane's registered gate (§13.1 / recorded rule) reads an
interval, not a point. The day-clustered interval on F's precision (2,000 draws, seed 47,
unit the calendar day, the three populations resampled together) is **[0.8741, 0.9310]**,
with **73.2 %** of draws at or above the bar — the interval **crosses** it. And F was chosen
after five tries on the same pool it is priced on; the month split is a consistency check,
not a pre-registration, because the recipe saw both months. That is the error class round 6
failed on. **So no row goes to surfacefit's fold refit from this section.**

**The revival condition, named.** F becomes a rider if a **pre-registered** read clears the
bar on a population the recipe has not seen: the recipe frozen in a committed spec (stand
clause only, centroid map at ≥ 1 shipped row, pool = the clock-only class, TOL_KM the shipped
p95), then read on the **December or July fold month's own ungated pool** — a third month,
neither of the two this section used. Cost: one probe run on a fold month plus one pricing
pass. If it clears, the ~774 rows are a **+0.029 s** face and they go to surfacefit by the
handshake; if it does not, check D closes with its cause on the record.

**Incidental, and it agrees with §7am.** Re-derived here from the committed pairs frame: the
shipped rows pass (i) alone **3,019** times and `(i)&(ii)` **2,956** times, so
`t = 0.9298522` — `analysis/chain-curve.json`'s 0.9298521547656495 to seven places. On the
step-0 pool (7,936 of 8,253 survive `wheels_gap_s > 62.405`), `(i)&(ii)` keeps **2,493**
under the km metric and **2,456** under squared degrees. The same **37**. The curve's rung is
the conjunction, not check (i) alone.

**Reading.** Declared check D: **NOISE** (a reversal, `π̂` negative). Repaired variant F:
**MAYBE**, revival as named above. Precision bar not reached under the handshake's rule, so
nothing moves.

**Files.** `work/check_d.py`, `analysis/check-d.json`, `logs/check-d.log`. No class change,
no serve, no `block.py` edit, no upload.

---

## §7ap — the pre-registered fold read of §7ao's reading F

**This paragraph is committed before any fold month is read.** The desk (2026-09-09, after
BOARD READING 17) asked for the revival §7ao named: freeze the recipe, pre-register it, then
read it on surfacefit's committed fold-month ungated pools. Rider only if the frozen recipe
clears **0.896 on every fold read**. No serve. Read-only on `exp-046-surfacefit`.

**What is frozen, and where each constant comes from.** Nothing below is chosen on a fold
month. **Pool (`q`)**: the mode-4 rows the shipped gate refuses *for the clock alone* —
`match_mode == 4` and `chain_gap_s > 62.405` and `chain_runway_agrees == 1` and
`chain_abs_dt_offblock > 1299.0`. **True (`t`)**: that month's own shipped chain rows,
`match_mode == 3`, whose counts surfacefit pins at 2025-07 **1,170**, 2025-12 **861**,
2025-01 **731**; the run stops if a month misses its pin. **False (`f`)**: §7ao's
re-expressed stand placebo — the service table rebuilt on that month's departures with
`STAND_mvt` permuted inside (airport, month) at seed 47, evaluated on the month's real pairs
(`match_mode` 3 or 4). The day placebo is **not** used, and §7ao measured why: a +24 h shift
lands on the same recurring service the next day, so it preserves the very invariant this
check reads and is a null for a clock check only. **The check**: the **stand clause alone**,
because §7ao measured the runway clause to be check (ii) re-derived. A row passes iff the
segment's own start position is within **TOL_KM** of the centroid of the service's modal
stand. **The service**: the same ladder — `airport|CALLSIGN_flt`, else
`airport|operator|scheduled hour` — needing ≥ 3 **other** calendar days in the month, the
modal stand over the service's other days in that month's movement file, this row's day
always left out. **TOL_KM = 2.2149 km**, the p95 measured on the *ranking* side in §7ao and
carried here unchanged, exactly as surfacefit carries `GAP_CUT_S`: *"taking it there and not
here is the point — the fold months never see it chosen, so it cannot be a fold selection"*
(`chainfold.py`). The per-month p95 is reported beside it as a sensitivity and is **not** the
declared reading. **The map**: stand centroids from that month's own shipped rows at ≥ 1 row
a stand — reading F's widened map; geography is learned per month because it *is* per month,
the pool never enters it, and the placebo gets the real map, as §27 requires.

**The read rule, declared before the run.** Per fold month: `t`, `f`, `q` with Wilson
intervals, `π̂ = (q − f)/(t − f)`, `precision = π̂·t/q`, the rows admitted, the projected face
at the shipped rung's worth per true row, and a day-clustered interval on the precision
(2,000 draws, seed 47, unit the calendar day). **RIDER only if the point estimate clears
0.896 on every one of the three fold reads**, which is the desk's bar as written; the lane's
own §13.1 gate reads an interval, so the interval is reported beside it and a fold whose
interval crosses the bar is called out. Any fold below the bar makes this a close, not a
rider.

**What could still go wrong, written before the read.** The fold months' shipped sets are
1,170 / 861 / 731 rows against the ranking side's 3,179, so both the centroid map and `t`
rest on about a third of the evidence, and the map's coverage is the quantity most likely to
move. That is a reason the read may come back thin. It is not a reason to change the recipe
after seeing it.

**Files.** `work/check_d_fold.py` (this spec is its module docstring),
`analysis/check-d-fold.json`. The three ungated fold blocks read are
`block-2025-07-8179f74417c8cb99-ungated.parquet`,
`block-2025-12-d35b7171d7062341-ungated.parquet` and
`block-2025-01-cb4d6289cb421aa2-ungated.parquet`, with the `-ungated-km` rebuilds read as a
metric sensitivity (the caveat §7am put on the record).

**The read — NOT A RIDER. The frozen recipe fails on july_loo.**

| fold | month | `t` shipped | `f` placebo | `q` pool | `π̂` | precision | CI95 (day-clustered) | rows | pool |
|---|---|---|---|---|---|---|---|---|---|
| july_loo | 2025-07 | 0.6564 (1,170) | 0.0744 (3,845) | 0.3122 (474) | 0.4086 | **0.8591** ✗ | [0.8105, 0.8982] | 148 | 474 |
| december | 2025-12 | 0.6829 (861) | 0.1147 (4,725) | 0.5198 (1,035) | 0.7130 | **0.9367** ✓ | [0.9116, 0.9577] | 538 | 1,035 |
| january | 2025-01 | 0.6594 (731) | 0.1063 (2,398) | 0.5583 (403) | 0.8172 | **0.9652** ✓ | [0.9396, 0.9893] | 225 | 403 |

The pins held on every month before anything was measured: mode-3 rows 1,170 / 861 / 731,
exactly `chainfold.MODE3_REFERENCE`. The declared bar is *every* fold, and **july_loo does
not clear it** — 0.8591 against 0.896, with the interval straddling the bar
([0.8105, 0.8982]). Under the lane's own §13.1 reading the failure is sharper still: no fold
interval is clear *above* the bar except december's and january's, and july's lower bound is
0.8105. **So check D's revival closes: no row goes to surfacefit's fold refit.**

**Why july_loo fails, and it is one number.** `t` is nearly identical on the three months
(0.656 / 0.683 / 0.659) and so is the share of *shipped* rows whose modal stand has no
centroid (33.2 / 31.0 / 33.0 %). The whole difference is in `q`, and `q` is capped by
coverage **on the pool**: the modal stand has no centroid for **64.98 %** of july's
clock-only rows, against **44.64 %** in december and **37.97 %** in january. It is not map
size — july has the *most* stands in its map (446, against 427 and 284). July's refused rows
simply stand where its 1,170 shipped rows never placed anything, which is what a peak month
does. And july_loo is the fold that carries **0.5571** of the board weight, so this is the
fold the reading most needed.

**Two sensitivities, neither of which rescues it.**

| reading | july_loo | december | january | every fold |
|---|---|---|---|---|
| declared (TOL_KM 2.2149 carried from the ranking side) | 0.8591 | 0.9367 | 0.9652 | **no** |
| per-month p95 tolerance (1.3835 / 1.5275 / 1.3744) | 0.8471 | 0.9154 | 0.8614 | no |
| the `-ungated-km` metric rebuild | 0.8583 | 0.9347 | 0.9646 | no |

The per-month tolerance — the reading this section deliberately did **not** declare, because
it would let a fold choose its own constant — is *worse*, and fails on two folds rather than
one. And the km-metric rebuild moves the third decimal at most (0.8583 / 0.9347 / 0.9646
against 0.8591 / 0.9367 / 0.9652), so **§7am's runway-metric caveat does not touch this
reading**: the clock-only class is the same class under either metric (pool 485 / 973 / 404
against 474 / 1,035 / 403).

**What the fold read says about §7ao's ranking-side number.** It says it was not an artefact:
`t` on the folds (0.656–0.683) brackets the ranking side's 0.6713, and `q` on the folds
(0.312–0.558) brackets its 0.4856. Reading F's 0.9052 was a fair point estimate of a quantity
that genuinely varies by month between 0.86 and 0.97 — and the recipe's problem is that the
variation crosses the bar. A pre-registered read is exactly what was needed to see that, and
it is the reason no row moves.

**The one repair left, named but not applied.** The binding constraint is the centroid map's
coverage on the pool, and the map here is learned from a month's 731–1,170 shipped rows.
`chaincheck.learn_maps` builds stand centroids from the **pass-1/2 matched pairs over the
prefix months**, which is a far larger known-good source and is where the shipped checks get
their own geography. A recipe using that map would raise coverage on `t` and `q` together and
might or might not raise the precision — on the ranking side widening the map lifted it
(§7ao, C → D: 0.8075 → 0.8650), and on the folds it is untested. Applying it now would be
choosing a recipe after seeing this read, which is the whole error this section exists to
avoid. It is therefore recorded as the revival condition and nothing more: **freeze a
`learn_maps`-sourced map, pre-register again, re-read these same three folds.** Until that
is run and clears 0.896 on all three, check D is closed with its cause on the record.

> **Superseded, 2026-09-09 (desk): the revival is not worth a run.** surfacefit §32.2
> re-prices the whole residual chain pool independently and bounds it: of the 7,936 rows
> after step 0, only the **312** that pass the clock and fail the runway are rescuable at
> all, worth **≤ +0.0116 s** at ×1.2 *before* any check rejects one of them. The cell this
> section read — 1,637 rows that pass the runway and fail the clock — is dismissed there on
> a discrimination argument this lane's numbers agree with: check (i) separates true from
> placebo at 15.6×, the stand position at 1.07× with a negative π̂ (their `chain-gate.json`;
> §7ao measured the same reversal from the day placebo). Failing the check that
> discriminates is evidence against the placement, not a gap in the evidence. The
> centroid-map revival is recorded and **not applied**.

**Reading.** **NOT A RIDER**, on the desk's bar as written and on the lane's interval gate.
Check D and its repaired variant F are closed; the revival above is the only route back. No
class change, no serve, no `block.py` edit, no upload, and no row to surfacefit.

**Files.** `work/check_d_fold.py`, `analysis/check-d-fold.json`, `logs/check-d-fold.log`.
Blocks read (read-only): `block-2025-07-8179f74417c8cb99-ungated.parquet`,
`block-2025-12-d35b7171d7062341-ungated.parquet`,
`block-2025-01-cb4d6289cb421aa2-ungated.parquet`, and the three `-ungated-km` rebuilds.

---

## §7aq — `work/audit.py`, one instrument for every upload, proved on v19

**What the desk asked for** (2026-09-09, after §7ap): generalize §7al's provenance audit and
§7am's byte audit into one tool that runs before every future upload, and prove it on v19's
record. It must reproduce **+0.2053 [+0.1401, +0.2927]**, the per-fold **151.23 / 123.78 /
87.26**, and §7am's counts.

**The tool.** `work/audit.py` takes the artifacts as arguments — `--head-npz`, `--arm-npz`,
`--pins`, `--class-vector` (a path with a `{month}` placeholder), `--candidate`,
`--previous`, `--hold`, `--draws`, `--seeds`, `--expect`, `--out` — and answers the four
questions an upload has to pass:

| block | what it reads | what it reports |
|---|---|---|
| **PINS** | this lane's `analysis/pin-manifest.json` against the sibling's fold population | airport-days and block rows per fold, the cache key, and whether they agree |
| **INCREMENT** | the two npz vectors and this lane's `base.load()` panel | `A`, `B`, the per-fold gain, the day-clustered interval at the primary seed **and the envelope over the seed list**, the all-row and clean board face |
| **POPULATION** | the same vectors | rows, rows moved, rows the hold returned to the base, and how many of those the arm *would* have moved |
| **BYTES** | the candidate, the previous upload, the class vector | sha256s, template order and finiteness, changed ⊂ occupied, written == occupied, the chain rows per ranking month, the untouched classes, the copied rows |

Two design choices carry the generality. **The claims live in a file, not in the script**:
`--expect` names each value by its path in the output (`increment.board.all_row_gain_s`),
with a tolerance, or `inside` for an interval, and every one is reported AGREE or DISAGREE
with what was read — so the next candidate is a new expectation file, not a new audit. And
**the hold is an argument**, applied to *both* vectors, because that single switch is the
whole difference between the two records this lane has already audited.

**The proof, on v19.** `analysis/audit-expect-v19.json` holds 43 claims taken from
`serve-v19.json`'s gate, `fold-population.json`, and §7am's counts. **43 of 43 AGREE.**

| fold | gain s² | claimed | interval, seed 47 | envelope, 11 seeds | rows | moved | LFPG returned |
|---|---|---|---|---|---|---|---|
| july_loo | **151.2274179823** | 151.2274179823 | [96.70, 203.17] | [94.00, 205.08] | 190,713 | 87,868 | 21,712 (394 would have moved) |
| december | **123.7830567805** | 123.7830567805 | [77.04, 172.13] | [74.44, 173.39] | 165,677 | 100,939 | 19,778 (989) |
| january | **87.2604760989** | 87.2604760989 | [46.30, 128.84] | [46.30, 131.94] | 153,706 | 69,839 | 17,900 (0) |

Board all-row gain **0.20534430269850645** against the record's 0.20534430269850645, clean
**0.2506746339168444** against 0.2506746339168444 — the point estimates agree to the last
digit stored, at a tolerance of 1e-9. The intervals are 2,000-draw bootstraps and therefore
reproducible only to their own draw noise: mine reads [0.1354, 0.2888] at seed 47 and
**[0.1354, 0.2943]** over eleven seeds, and the record's **[0.1401, 0.2927]** lies inside
that envelope. That is the honest test and the tool applies it as one — a claimed interval is
checked for lying inside the seed envelope, never for equalling one stream.

**The byte block, on the same run.** Candidate `4d75706b…` against v17 `4de323fc…`: 344,841
rows in template order, all finite, no duplicate key; **192,435** rows changed at an RMS move
of **32.674 s** and a maximum of 1,019.41 s; **193,910** occupied on the v19 vector and
193,910 written, **0** occupied-unwritten and **0** changed outside the occupied class, with
1,475 occupied rows whose value landed identical; **150,931** copied rows byte-identical to
v17; the chain's **2,456** mode-4 rows split 1,304 / 1,152 by ranking month, 2,171 written
and 285 held at LFPG; **388** noac rows, none changed; **39,872** LFPG ranking rows, none
changed.

**The generality is proved, not asserted.** Run with `--hold ""` and nothing else changed,
the same tool returns **136.6689 / 93.1986 / 87.2605** and a board face of **+0.191796** —
which is §7al's bundle read and §28.2's +0.1918, exactly. One instrument reproduces both
records, and the only argument between them is the airport hold. That also settles in one
line why the two readings of the same bundle differ: **LFPG.** Held, the fold gains rise from
136.67 / 93.20 to 151.23 / 123.78, because the 394 and 989 LFPG rows the arm would have moved
were moving the wrong way; january's LFPG rows are untouched by this arm, which is why its
87.26 is identical under both readings.

**How the next audit is run.** Point `--head-npz` and `--arm-npz` at the new bundle's
vectors, `--class-vector` at the class the serve writes, `--candidate` and `--previous` at
the new file and the one on the board, set `--hold` to whatever the gate holds — for the
announced 511-leaf rung + plane-metric class + LFPG rider bundle over v18 that is likely
**`--hold ""`**, since an LFPG rider means LFPG is no longer held, and the audit will say so
by reporting `held_airports.rows_changed > 0` — and write a new expectation file. Nothing in
the script changes.

**Files.** `work/audit.py`, `analysis/audit-expect-v19.json`, `analysis/audit-v19.json`,
`logs/audit-v19.log`. Read-only on `exp-046-surfacefit`; no serve, no upload.

---

## §7ar — the fit-path audit: the paths are identical, and the offset is the instrument's floor

**The question** (desk, 2026-09-09, urgent, feeding the next ruling): three unrelated changes
over v18's bytes — the 511 rung, the plane-metric class, an arbitrary 33-row placebo — all
read about **−0.063 s** on the board with july between −48 and −85 s²/row. Is there a
systematic difference between the path that made the served head **H**
(`chainfold.fit_arm` + `chainbundle.py` → `d-bundle-ungated1299-guard995.npz`) and the §33/§34
instruments? Read-only on their code and vectors; nothing of theirs edited, no serve.

### 1. The diff table — no procedural difference on any item the desk named

| item | H's path | the §33/§34 instruments | verdict |
|---|---|---|---|
| the fitting function | `chainfold.py:127` `fit_arm`, called at `chainbundle.py:54` | `rung511.py:104`, `classplacebo.py:144`, `lfpgrider.py:98` **call that same function**; `nullseed.py:55` re-implements it | **identical** |
| params (leaves, lr, rounds, min_data, bagging, feature fraction) | `arms.py:71` from `predeclared.json`: l2, 300 rounds, lr 0.05, 31 leaves, min_child 100, ff 0.8, bf 0.8/freq 1, λ₂ 1.0, n_jobs 3 — `chainfold.py:55` overrides leaves to 255 | `rung511.py:102`, `classplacebo.py:142`, `nullseed.py:70` override **only** `num_leaves` | identical **except `num_leaves`, which is the change under test** |
| seeds | `arms.py:73` SEED 46; `_lgb_fit:526` sets it | the same default everywhere; only `nullseed.py:50` moves it, and its first row *is* 46 | identical |
| threads | n_jobs 3, `OMP_NUM_THREADS=3` (`chainbundle.py:16`) | n_jobs 3, `OMP_NUM_THREADS=3` (`rung511.py:29`, `nullseed.py:26`) | identical |
| early stopping | `arms.py:528` `m.fit(X, y)` — no eval set, no callbacks | the same function | identical — **there is none on either path** |
| the guard (cut and where) | `chainfold.py:144` `G.cut(pan, others)` = `guard.py:46-51`, p99.5 of `pbase − obs_taxi` over the **other two folds'** usable rows; zeroed at `:148` | the same lines run; `nullseed.py:74,78` repeats them verbatim | identical |
| the LFPG hold, and its order | `chainfold.py:56` `HOLD=('LFPG',)`, applied at `:149` **last**, after fit, floor and guard | `CF.HOLD=()` then `rung511.py:86-93` `hold_lfpg` after (`:149`, `nullseed.py:116`) | **identical in effect** — nothing downstream of `:149` reads the vector |
| the class and row set the fit consumes | `chainbundle.py:46-50` → `chainfold.py:115-124` `panel_for` patches `A.month_block` to `restrict(block, 1299, 62.405)`, then `A.panel_with_block(BA.load())`; `fit_arm:132` fits on the other folds' `usable` rows | `rung511.py:218-231` the same; `classplacebo.py:122-133` patches with blocks **pre-blanked by `placebo_block:64-91`, which is `restrict:92-113` line for line** | identical rule; **the row set is the change under test** |
| the price base and the residual | `chainfold.py:153-163`: `pbase → pbase + d_head`, `rbase → rbase − d_head`, `inc = d_new − d_head` | `rung511.py:124` calls that function; every instrument prices through it | identical — only *which vector is `d_head`* changes |
| the rotation | `chainfold.py:131-132` leave-one-fold-out; 5 contiguous day blocks (`arms.py:548-552`), per-airport amplitude clipped to [0,1] (`:553`) | the same lines run | identical |

The only re-implementation, `nullseed.fit_at_seed:58-79`, is `fit_arm:128-149` line for line
with two inert differences: it names `CF.BU.FEATURES` where `fit_arm` names `BU.FEATURES`
(the same module object, `chainfold.py:48-53`), and it passes seed and leaves explicitly
where `fit_arm` reads the module constants. At seed 46 and 255 leaves that is the same call.

### 2. Independent confirmation, from a cached vector rather than a refit

`work/d-lfpg-unheld-fit.npz` (sha `01620e…`, `lfpgrider.py:95-99`) is H's own recipe — 255
leaves, shipped class — refitted through the new path with `CF.HOLD = ()`. Compared with H
off LFPG, in this lane's code:

| fold | rows off LFPG | rows differing | max |Δ| |
|---|---|---|---|
| july_loo | 169,001 | **0** | 0.0 |
| december | 145,899 | **0** | 0.0 |
| january | 135,806 | **0** | 0.0 |

**Bit for bit on every fold.** The paths are identical in fact, not only in reading, and
LightGBM is reproducible here — so the refit the desk offered to have me run would return
the same answer, and it was not run.

### 3. The decomposition — what actually changed

`gain = 2A − B` with `A = mean(r·inc)` and `B = mean(inc·inc)`, so a read is positive exactly
when **`2A/B > 1`**. That ratio is twice the regression coefficient of the residual on the
increment: it says whether the increment is applied at an amplitude its own signal supports.

| read | board | july `A` / `B` / 2A/B | december | january |
|---|---|---|---|---|
| **v19** over the served head (held) | **+0.20534** | 296.8 / 442.4 / **1.34** | 361.4 / 599.0 / **1.21** | 310.5 / 533.7 / **1.16** |
| **A** 511 rung, shipped class | −0.06281 | 144.1 / 350.8 / **0.82** | 283.5 / 554.5 / 1.02 | 231.2 / 468.5 / 0.99 |
| **B** plane-metric class, 255 | −0.06087 | 116.7 / 318.7 / **0.73** | 267.0 / 510.8 / 1.05 | 231.1 / 437.0 / 1.06 |
| placebo 1 (33 rows) | −0.06304 | 134.1 / 316.5 / 0.85 | 256.6 / 509.3 / 1.01 | 203.3 / 431.0 / 0.94 |
| placebo 2 (33 rows) | −0.06344 | 126.0 / 317.6 / 0.79 | 262.8 / 512.1 / 1.03 | 209.3 / 421.7 / 0.99 |
| placebo 3 (33 rows) | −0.05339 | — / — / 0.75 | — / — / 0.95 | — / — / 1.07 |

This reproduces surfacefit's own figures exactly — v19 **1.34 / 1.21 / 1.16**, A **0.82 /
1.02 / 0.99**, B **0.73 / 1.05 / 1.06** — from this lane's panel and their vectors, with none
of their code. (Their v19 ratio is the *held* reading; the unheld one, §7al's +0.1918, gives
1.30 / 1.15 / 1.16.)

**`B` barely falls; `A` collapses.** On july the movement drops 442 → 317–351 (−21 to −28 %)
while the alignment with the residual drops 297 → 117–144 (**−52 to −61 %**). The new
increments still move the same rows by nearly the same amount; they simply no longer point at
what is left of the error. That is what a **re-arrangement** looks like as opposed to
information: v19's increment carried new observations, so it earned `A`; a change of leaves,
of runway metric, or of 33 rows of class membership produces a different-but-equally-good
fit, whose difference from H is nearly orthogonal to the residual. Two equally good fits give
`2A ≈ B`, which is the break-even line — and every new read sits on it (0.94–1.07 on december
and january) or just below it on july (0.73–0.85).

**The sign is not a bug; it is arithmetic.** Because the base already contains H, any refit
that is merely *different* rather than *better* spends the full variance `B` and earns no
`A`. That charge is negative by construction, it is about the same size for any change of
this kind, and july carries 0.5571 of the board weight, so the board sums to a consistent
small negative. Three unrelated changes read −0.063 because −0.063 is what this instrument
returns for *any* re-arrangement of this magnitude.

### 4. The consequence for the ruling: the placebo envelope, not zero, is the null

An arbitrary 33-row class change moves a fold by up to **81 s²/row** and the board by
**−0.063 s**. That is the instrument's resolution floor. Measured against the three-draw
envelope — july [−81.25, −48.30], december [−25.25, +13.56], january [−24.40, +30.09], board
[−0.06344, −0.05339]:

- **A, the 511 rung, is inside the placebo envelope on every fold and on the board**
  (−62.58 / +12.54 / −6.13). It is **not measurable at this instrument** — not negative,
  unresolvable. Reading it as a negative result would be reading the floor.
- **B, the plane-metric class, is outside the envelope on two folds** (july −85.43 below it,
  december +23.21 above it) and inside on january, with its board total still inside the band.
  So B is a **real change with a fold-inconsistent sign**, which the registered every-fold
  gate fails on its own terms — for a reason, not for the floor.

**What this lane recommends, in one line:** the placebo envelope surfacefit already builds
(`classplacebo.py`) should be the pre-registered **null** for any read priced as an increment
over a served head — a change must beat the envelope, not zero — and any change whose own
`2A/B` sits at 1 is a re-arrangement that this design cannot price, whatever its sign.

**Reading.** No procedural difference exists: **identical on all ten items**, and the refit of
H's recipe reproduces H bit for bit on every fold. The common −0.063 is the instrument's floor
for a re-arrangement, explained by `A` collapsing while `B` holds. A is unresolvable; B is
real and fold-inconsistent.

**Files.** `work/fitpath_audit.py`, `analysis/fitpath-audit.json`, `logs/fitpath-audit.log`.
Read-only on `exp-046-surfacefit`; no refit run, no serve, no upload.

---

## §7as — the expected board move for v19, registered before the upload

**Written and committed before the upload, and therefore before BOARD READING 18** — score-feedback rule: a
prediction that arrives after the score is not a prediction. Nothing here is fitted. The
candidate is **v6 = v18's bytes + the LFPG rider alone (read E, no refit)**, whose registered
face is **+0.01940 s [+0.00784, +0.03383]** all-row under the lit rule, clean +0.02369
(exp-046 §30, `analysis/rider-read.json`).

**§7ar's resolution floor does not apply to this candidate, and the reason is on the record.**
The rider is a *read*, not a refit — the same vector H with the LFPG hold released — so its
increment is not a re-arrangement of a fresh fit. Its `2A/B` is **2.20 and 2.24** on the two
lit folds, against the 0.73–1.05 every fresh fit returned. A void read and this read are
different objects.

**The lane history, now six uploads.** Face registered before the upload → measured board
move → transfer; every move reconciles to the ledger.

| upload | reading | lane | face s | board before → after | move s | transfer |
|---|---|---|---|---|---|---|
| v13 | 12 | surfacefit | 8.1280 | 262.7418 → 248.4818 | 14.2600 | 1.75× |
| v14 | 13 | queue | 0.1854 | 248.4818 → 248.2904 | 0.1914 | 1.03× |
| v15 | 14 | surfacefit | 0.7296 | 248.2904 → 246.8263 | 1.4641 | 2.01× |
| v16 | 15 | surfacefit | 0.3350 | 246.8263 → 246.7053 | 0.1210 | 0.36× |
| v17 | 16 | queue | 0.0686 | 246.7053 → 246.6571 | 0.0482 | 0.70× |
| **v18** | **17** | **surfacefit** | **0.20534** | **246.6571 → 246.3605** | **0.2966** | **1.44×** |

Median **1.235×** (was 1.03× at five), mean **1.215×**, range **0.36–2.01×** unchanged — v18's
1.44× sits inside the range and lifts the median. In upload order the six are 1.75, 1.03,
2.01, 0.36, 0.70, 1.44: **no direction**, so none is predicted here either.

**The expected board score for v19, from 246.3605.**

| reading | transfer | expected board |
|---|---|---|
| at face | 1.00× | **246.3411** |
| at the median | 1.235× | 246.3365 |
| at the mean | 1.215× | 246.3369 |
| the observed transfer range, at face | 0.36–2.01× | **246.3215 – 246.3535** |
| the observed range on the face interval | 0.36–2.01× × [0.00784, 0.03383] | 246.2925 – 246.3577 |
| the two most recent uploads | 0.70 / 1.44× | 246.3326 – 246.3469 |
| the two most recent surfacefit uploads | 0.36 / 1.44× | 246.3326 – 246.3535 |

**What would count as a transfer failure, in one sentence.** *A transfer failure is a board
score above **246.3577** — a move under **+0.0028 s**, which is the lowest transfer ever
observed on this board (0.36×, v16) applied to the lower end of this candidate's own face
interval (+0.00784 s) — and a score at or above v18's **246.3605** is not a weak transfer but
no transfer at all.*

**Is the move inside one board digit's noise floor? No — but the failure test is fine.** The
board reports four decimals, so one display step is 0.0001 s. The move at face is **194
steps**, and even the smallest move anywhere in the band is **28 steps**, so the candidate is
resolvable. What *is* fine is the failure half of the reading: only **28 steps** separate the
failure line (246.3577) from "no transfer at all" (246.3605), so distinguishing a weak
transfer from none rests on the third and fourth decimals, and any board-side variation above
0.003 s would blur it. None has been observed — BOARD READINGS 12–17 every one reconciled
exactly at four decimals — but the reading should be stated with that width in mind: **a
score anywhere in 246.3577–246.3605 is "at the floor", not a clean failure.**

**Rank is not at risk.** Second place is 260.931 and no outcome in any band above moves us by
more than 0.07 s, so rank 1 survives every reading by more than 14 s.

**Two things this pre-registration does not claim.** It does not predict the direction of the
surprise — six transfers with no trend make a wide band the honest one. And it is **not** a
claim that the rider is worth a slot; that is the desk's call, and this is only the reading
rule for it if the slot is spent.

**Files.** `work/v19_prereg.py`, `analysis/v19-preregistration.json`. No serve, no upload —
the desk uploads.

---

## §7at — the byte audit of candidate v6 (the LFPG rider alone)

**Run** (the line is in the expectation file, and this is the whole command):

```
work/audit.py --head-npz  ../exp-046-surfacefit/work/d-bundle-ungated1299-guard995.npz
              --arm-npz   ../exp-046-surfacefit/work/d-rider-lfpg-on-v19.npz
              --pins      ../exp-046-surfacefit/analysis/fold-population.json
              --class-vector          ../../artifacts/exp-046/class/{month}-v20.parquet
              --previous-class-vector ../../artifacts/exp-046/class/{month}-v19.parquet
              --candidate ../../submissions/candidates/exp-046-surface-v6.parquet
              --previous  ../../submissions/uploaded/youthful-giraffe_v18.parquet
              --baseline  ../../submissions/uploaded/youthful-giraffe_v17.parquet
              --hold "" --changed-expected-at LFPG
              --expect analysis/audit-expect-v6-rider.json --out analysis/audit-v6-rider.json
```

**53 of 56 claims AGREE, 0 DISAGREE, 3 PENDING** on a vector being re-emitted.

| the desk's claim | read | verdict |
|---|---|---|
| sha256 `54dec3df…` | `54dec3df14a9aaea6938b05c0b8e4280a958bce32f734b293ecb2deb32eb41fc` | **AGREE** |
| 2,250 rows changed vs v18 | 2,250 | **AGREE** |
| all inside occupied v20 (196,256), 0 outside | occupied 196,256; changed ⊂ occupied; 0 outside | **AGREE** |
| changed ∩ occupied-v19 = 0 | 0, disjoint | **AGREE** |
| newly occupied in v20 = 2,346 → 2,250 changed, 96 not | 2,346 = 2,250 + 96; 0 no longer occupied | **AGREE** |
| noac untouched | 388 noac rows, 0 changed | **AGREE** |
| none of v5's changed rows reverted | 192,435 rows v18 moved off v17; **0** put back | **AGREE** |
| template order, finite, min 60 | 344,841 rows in template order, 0 non-finite, 0 duplicate keys, min **60.0**, max 115,306.6 | **AGREE** |
| *(added by this lane)* changed rows are LFPG only | `changed_by_airport = {LFPG: 2,250}`, 0 elsewhere | **AGREE** |
| the increment half, re-read on the same run | july +20.83201689 (552 rows), december +51.90698722 (1,246), january dark (0), board +0.01939942 / clean +0.02368616, both lit intervals inside their envelopes | **AGREE** |

The move itself: RMS **95.887 s** on the 2,250 changed rows, maximum 773.093 s — a large
per-row move on a small, single-airport class, which is what releasing a held airport looks
like.

**The three PENDING checks, and why.** `occupied_and_not_written`, `written_and_not_occupied`
and `occupied_unwritten_rows_that_changed` read the class vector's `written_by_exp_046` and
`guard_held_by_exp_046` columns, which surfacefit is re-emitting: `guard_held` was all-False
and `written` undercounted (2026-01 90,074 against 91,433 occupied; 2026-07 104,609 against
104,823). This lane measured the same thing from the other side before the notice arrived —
1,573 occupied-and-not-written rows, **none of which changed value**, made of the **1,477**
rows v19 marked `guard_held` *and* `written`, plus the **96** newly-occupied rows that did not
change. The sums reconcile exactly: 91,433 + 104,823 = 196,256 occupied, 90,074 + 104,609 =
194,683 written, difference 1,573. **`occupied_by_exp_046` and the bytes are unaffected**, so
every other check stands and the three re-run on the re-emitted vector.

**One consequence worth stating, because it touches the standing handshake.** The rule this
lane serves under reads **`occupied_by_exp_046`**, not `written_by_exp_046` — and under the
convention v20 shipped with, 1,573 occupied rows carry surfacefit's value while `written` is
False. A serve that complemented `written` would overwrite them. The rule is right as
written; this is a reason not to loosen it.

**Two additions the candidate forced into `work/audit.py`, both general.**
`--previous-class-vector` gives the newly-occupied and disjointness block (does this candidate
touch what the last serve wrote?), and `--baseline` gives the revert check (does it undo the
last upload?). Both are checks every future candidate should carry. A `pending` list in the
expectation file now reports a claim as PENDING rather than passing or failing it when the
artifact it reads is being re-emitted — a pending check is counted apart and never passes
silently.

**Reading.** Every claim the desk made about candidate v6's bytes **AGREES**. Nothing
disagrees. Three bookkeeping checks await the re-emitted class vector and are named above.

**Files.** `analysis/audit-expect-v6-rider.json` (56 claims), `analysis/audit-v6-rider.json`,
`logs/audit-v6-rider.log`, `work/audit.py`. No serve, no upload.

---

## §7au — the expected board move for v20, registered before the upload

**Written and committed before the upload and before BOARD READING 19** (score-feedback rule). The candidate
is **v7 = v6's bytes + the bag with LFPG held** — K = 5 fits of the shipped 255-leaf recipe on
distinct class perturbations, averaged. Registered face **+0.19937 s [+0.14426, +0.27236]**
all-row, clean +0.24346 [+0.19229, +0.29997] (exp-046 `analysis/bag-arm.json::prices.bag_held`,
vector `work/d-bag5-255-guard995.npz`).

**Why the face over v18's bytes is also the increment over v6.** The bag is fitted with LFPG
held and the rider moves *only* LFPG rows, so the two changes are disjoint on the row set.
The desk states this and the vectors bear it out: `bag_held` moves 87,868 / 100,940 / 69,839
rows, exactly the counts H itself moves.

**The §7ar floor check, which this candidate is the first that has to pass.** The bag *is*
refits — five of them — and §7ar measured a single refit of this arm on a perturbed class at a
floor of about −0.063 s, which the desk's INSTRUMENT ruling makes the interpretation floor for
any refit read. So the bag has to clear the placebo envelope before it is read as anything:

| fold | placebo envelope | bag, point | bag, lower bound | margin over the envelope's top |
|---|---|---|---|---|
| july_loo | [−81.25, −48.30] | **+84.94** | **+54.44** | +102.74 |
| december | [−25.25, +13.56] | **+176.32** | **+137.42** | +123.86 |
| january | [−24.40, +30.09] | **+162.39** | **+124.29** | +94.20 |
| board | [−0.06344, −0.05339] | **+0.19937** | **+0.14426** | +0.19765 s |

**Every fold's lower bound is above the top of its own placebo envelope**, july by the least.
Averaging K fits is exactly what removes the re-arrangement variance §7ar priced, so this is
the expected shape — but it is checked, not assumed, and it is the reason this read is not
void where A and B were.

**The move, registered anchor-free**, because the anchor is whatever v19 scores. Transfer
history as in §7as (six uploads, median 1.235×, mean 1.215×, range 0.36–2.01×; when BOARD
READING 18 lands, append v19's own transfer and re-read the median before using this table).

| reading | board move from v19's score |
|---|---|
| at face | **+0.1994** |
| at the median transfer | +0.2462 |
| at the mean transfer | +0.2422 |
| the observed transfer range, at face | **+0.0718 – +0.4007** |
| the observed range on the face interval | +0.0519 – +0.5474 |
| the two most recent uploads | +0.1396 – +0.2871 |

Absolute bands are in the JSON at three worked anchors — v18's 246.3605 (the fallback if v19
is not read), 246.3411 (v19 at face per §7as) and 246.3365 (v19 at the median) — with the rule
stated there: **substitute the actual v19 score when BOARD READING 18 lands.** At the fallback
anchor the at-face expectation is **246.1611** and the observed-range band is 245.9598–246.2887.

**What would count as a transfer failure, in one sentence.** *A transfer failure is a board
move of less than **+0.0519 s** from whatever v19 scores — the lowest transfer ever observed on
this board (0.36×, v16) applied to the lower end of this candidate's own face interval
(+0.14426 s) — and a score at or above v19's own is not a weak transfer but no transfer at
all.* At the fallback anchor that is a score above **246.3086**.

**Resolution: not close to the board's last digit.** The move at face is **1,994** display
steps and the failure line itself is **519**, against v6's 28. Unlike §7as, both halves of this
reading are far above the board's resolution.

**If v6 is never uploaded — THIS IS NOW THE PRIMARY READING (desk ruling 2026-09-09, addendum below).** v7 goes up over v18's bytes and carry the bag *and* the
rider, so the registered face becomes the combined **+0.22025 [+0.16206, +0.29728]**
(`bag_free`), the at-face expectation 246.1403, the observed-range band 245.9178–246.2812, and
the failure line a move under +0.0583. That branch is in the JSON so the reading rule does not
have to be rewritten if the slot order changes.

**Rank is not at risk** on any band; second place is 260.931.

**Files.** `work/v20_prereg.py`, `analysis/v20-preregistration.json`. No serve, no upload.

## §7av — the three PENDING v20 checks land, and v7's expectation file

Two items: close §7at's PENDING trio on the re-emitted class vector, and write the expectation
file the v7 serve is audited against.

**The re-emitted v20 answers all three.** The desk re-emitted the vector at `adf5d60` (class
shas `4dc51b4b…` / `ccf37a67…`) with `occupied_by_exp_046` identical row for row. Re-running
`work/audit.py` unchanged against it:

| check | §7at, first vector | re-emitted v20 | verdict |
| --- | --- | --- | --- |
| `bytes.class.occupied_and_not_written` | 1,573 | **0** | AGREE |
| `bytes.class.written_and_not_occupied` | 0 | **0** | AGREE |
| `bytes.class.occupied_unwritten_rows_that_changed` | 0 | **0** | AGREE |
| `bytes.class.guard_held` (added) | 0 | **1,573** | AGREE |
| `bytes.class.chain_mode_4_rows` | 2,451 | **2,456** | AGREE |

`written == occupied == 196,256`. The 1,573 rows this lane read as occupied-and-unwritten on
the first vector are **exactly** the rows now flagged `guard_held_by_exp_046`, which confirms
§7at's diagnosis to the row: the first vector moved v19's guard-held rows out of `written`
instead of into `guard_held`, and none of them ever changed value. **57 of 57 AGREE** (56 plus
the new `guard_held` claim), candidate sha `54dec3df…` as served. Run:
`analysis/audit-v6-rider.json`.

**v7's expectation file.** `analysis/audit-expect-v7-bag.json`, 39 claims: the bag's three
per-fold gains and both board figures to 1e-9, the population, the pins, and a byte block whose
changed rows are the occupied **non-LFPG** rows. Two additions the candidate forced:

- **`--changed-forbidden-at`.** v7's byte expectation is v6's complement — LFPG rows identical
  to v6, everything else free to move. Naming the one forbidden airport is cleaner than
  enumerating the other nine, so the tool now reports
  `changed_rows_at_the_forbidden_airports` (expected 0) and
  `rows_identical_at_the_forbidden_airports`.
- **MISSING is no longer a disagreement.** A claim the run could not answer — an argument not
  supplied — used to be counted with the defects. It now reports in
  `not_answered_on_this_run`, apart from `disagree`, so it can neither pass silently nor read
  as a fault in the candidate.

**The increment half is already answered.** The bag vector is on disk, so the dry run priced it
before the serve: **27 of 27 answerable AGREE**, the twelve byte claims not answered until the
candidate lands. july_loo +84.93911, december +176.32122, january +162.39217 s²/row and the
board +0.19937 / clean +0.24346 reproduce to the last digit through this lane's own arithmetic.

**One declared tolerance, recorded rather than hidden.** The interval check asks whether the
claimed interval lies inside this lane's envelope over eleven bootstrap seeds. july_loo and
january contain it at tol 0; **december's top (219.7205) sits 1.3042 s²/row above this lane's
envelope top (218.4163)**. That miss is smaller than the instrument's own seed-to-seed spread
on the same end (2.2569 s²/row on december, 0.9957 july, 1.4132 january), so it prices
bootstrap draw noise, not a difference about the arm. The fold intervals therefore carry a
declared **3.0 s²/row** tolerance — above the largest spread measured, declared before the
serve, and written into the file with the measurement that sizes it. The board interval stays
at tol 0, where it agrees. The point estimates underneath carry 1e-9 and are exact.

**Files.** `analysis/audit-expect-v7-bag.json`, `analysis/audit-v7-bag-dryrun.json`,
`analysis/audit-v6-rider.json` (re-run), `work/audit.py`. No serve, no upload; the byte half
runs the moment the desk sends v7's sha.

### §7au addendum — the registered branch is now the primary reading (desk ruling, 2026-09-09)

v6 is not uploaded separately: its +0.01940 face sits under the upload rule's ≈ 0.1 s noise
floor, so v7 carries the rider. **`youthful-giraffe_v19.parquet` = v7's bytes = v18 + the bag applied
EVERYWHERE, LFPG included, registered face +0.22025 [+0.16206, +0.29728] over 246.3605, at-face
expectation 246.1403, observed-range band 245.9178–246.2812, and a transfer failure is a move
of less than +0.0583 s from 246.3605 — a score at or above 246.3605 is no transfer at all.**
The anchor is now fixed rather than deferred, because no v19 board reading intervenes. §7as's
v19 pre-registration is not read on any upload and is superseded by this line.

### §7aw — v7's expectation file re-targeted to the bag everywhere

The ruling changes the arm, the previous file, and the changed-row population, so the
expectation file is rewritten rather than patched.  **Corrected below (§7aw addendum): v7 is
the bag EVERYWHERE, not bag + rider, so the two lines about the rider in this section were
wrong and are struck there.**

| | primary (rewritten) | component read (the old file) |
| --- | --- | --- |
| file | `analysis/audit-expect-v7-bag.json` | `analysis/audit-expect-v7-bag-held.json` |
| arm | `d-bag5-255-free.npz` (`bag_free`, the bag everywhere) | `d-bag5-255-guard995.npz` (`bag_held`, the bag alone) |
| hold | none — the bag writes LFPG too | LFPG |
| july_loo | **+107.34850** [73.918, 138.413] | +84.93911 [54.438, 115.125] |
| december | **+226.07966** [175.426, 285.889] | +176.32122 [137.419, 219.721] |
| january | **+162.39217** [124.295, 207.627] | +162.39217 — identical |
| board / clean | **+0.22025 / +0.26896** | +0.19937 / +0.24346 |
| rows moved | 88,420 / 102,186 / 69,839 | 87,868 / 100,940 / 69,839 |

~~**The difference between the two files is the rider, and it reconciles to the row.**~~ **Struck
— see the §7aw addendum.** The populations do coincide with read E's (552 july, 1,246
december), but the difference is the **bag's own LFPG rows**, not the rider's.

**Both dry runs reproduce to the last digit.** Primary **30 of 30** answerable AGREE (15 byte
claims waiting on the serve), cross-check **27 of 27**. Every point estimate matches at
`0.00e+00`.

**And the primary needs no interval tolerance.** All three fold intervals *and* the board
interval lie inside this lane's eleven-seed envelope at **tol 0** — july [73.324, 142.721],
december [172.353, 286.770], january [121.091, 208.352], board [0.15615, 0.30603]. §7av's
3.0 s²/row tolerance stays confined to the held cross-check, where december's claimed top sat
1.3042 above the envelope. The looser claim is the one that is not on the upload path.

**The byte block changed shape.** v7 changes two populations — the occupied LFPG rows and the
occupied non-LFPG rows the bag writes — so §7av's `--changed-forbidden-at` does not apply to
this candidate and no outside-zero claim is made. The LFPG half is claimed exactly at **2,250**
(`--changed-expected-at LFPG`), for the reason the addendum measures;
the bag's own count is read, not claimed, until the desk names it. `--previous` is
**`youthful-giraffe_v18.parquet`** and `--baseline` v17: the ruling makes v7 the next upload,
so v6 is never a link in the chain. noac untouched, and the three re-emitted-v20 class claims
carry forward at 0.

**Files.** `analysis/audit-expect-v7-bag.json`, `analysis/audit-expect-v7-bag-held.json`,
`analysis/audit-v7-bag-dryrun.json`, `analysis/audit-v7-bag-held-dryrun.json`. No serve, no
upload; the byte half runs on the desk's sha.

### §7aw addendum — v7 is the bag EVERYWHERE, and what that corrects

Desk ruling confirmed with surfacefit, 2026-09-09: **v7 = `bag_free` = the bag applied
everywhere, LFPG included.** v7's LFPG rows carry the bag's own values, so v7 is **not**
byte-identical to v6 at LFPG, and the label "v18 + bag + LFPG rider" I used in §7au and §7aw is
wrong. Corrected in both places above; the numbers are untouched, because the arm was always
`d-bag5-255-free.npz`.

**What the difference between the two files actually is.** The bag's own LFPG contribution —
**+22.409 s²/row on july_loo, +49.758 on december, 0 on january** — not the rider's. The row
counts (552, 1,246) coincide with read E's rider populations because both arms move the *same
LFPG population*, not because the values are the rider's. The values differ from the rider's
(+20.83 july, +51.91 december), and that near-miss, which I read the wrong way round in §7aw,
is the evidence **for** the corrected composition rather than against it.

**The 2,250 claim survives, with a different reason.** Measured on v20 before the serve: LFPG
carries **2,346** occupied rows, of which **exactly 96** are `guard_held_by_exp_046`, leaving
**2,250** writable. v6's rider changed 2,250 LFPG rows and left 96 unchanged — and those 96 are
exactly the 96 guard-held rows, so what held them was the **guard**, not anything about the
rider. v7 runs over the same guard-995 head, so the same 96 stay at the base and the same 2,250
are writable. The claim's *value* is unchanged by the correction; its *reason* is: it is no
longer "the rider's rows" but occupied-minus-guard-held at LFPG.

**So the claim is now self-diagnosing.** `work/audit.py` gained
`class.occupied_at_the_expected_airports`, `class.guard_held_at_the_expected_airports` and
`class.writable_at_the_expected_airports`, all three claimed (2,346 / 96 / 2,250) beside the
changed-row count. If the changed count misses, the components say which half moved, and the
registered condition is stated: **if v21 re-emits a different LFPG guard-held count, the claim
moves with it and that is not a defect.**

**No claim ties v7 to v6 at LFPG.** Nothing in the file asserts LFPG equality with v6 or with
the rider's values; every LFPG claim is a population claim.

**The held file is relabelled, not retired.** It prices the **bag alone** (`bag_held`, +0.19937
over v18) — which is neither v7 nor the sibling the desk names at +0.21877 (`bag_held` + the
rider), a third composition priced by neither file. It is kept because differencing it against
the primary isolates the bag's LFPG contribution with no second instrument.

**Runs.** Primary dry run **30 of 30** answerable AGREE (48 claims, 18 waiting on the serve). A
byte-block rehearsal — the same expectation file run against v20 with v6 standing in for the
candidate, which tests the *populations*, never v7's values — reads **48 of 48**, so every byte
claim is answerable and correctly shaped before the serve.

**One pin will move by design.** surfacefit is patching `bag-arm.json`'s two label strings in
place, so its sha256 changes while its numbers do not. At the serve this lane checks that file
by its **numbers** (the five figures claimed at 1e-9), not by its sha; the three npz pins must
still match byte for byte.

## §7ax — the v7 audit: 52 of 52 AGREE

Candidate `submissions/candidates/exp-046-surface-v7.parquet`, sha
`128db7c58881959e…`, class v21 (`e1e8e1b9…` / `2dc52ccc…`), previous
`youthful-giraffe_v18.parquet`, baseline v17, hold empty. Run:
`analysis/audit-v7-bag.json`.

**Pin check first, as registered.** Exactly one pin moved and it is the one §7aw said would:
`bag-arm.json` `86b3e597…` → `d389df00…`, the label patch. Checked by its **numbers** instead —
all five identical to the pinned values (`107.34850405601257`, `226.07966091333805`,
`162.39217339106762`, `0.22024957054372862`, `0.2689649724647438`), and the new label now reads
"the bag (K=5) with LFPG UNHELD, so LFPG carries the BAG's own value and NOT the rider's". The
three npz pins match **byte for byte**: `c6d1f5b6…`, `5aabdb32…`, `5dbbcb8c…`.

**52 of 52 AGREE, nothing pending, nothing unanswered.** The fold half reproduces to the last
digit — july_loo +107.34850, december +226.07966, january +162.39217 s²/row, board +0.22025 /
clean +0.26896, rows moved 88,420 / 102,186 / 69,839, every fold lit and clear on every seed,
every claimed interval inside this lane's eleven-seed envelope at tol 0.

**The byte half closes to the row.**

| | |
| --- | --- |
| rows changed | **194,681** = 2,250 at LFPG + 192,431 elsewhere |
| occupied (v21) | 196,256, written 196,256, guard-held 1,573 |
| LFPG occupied / guard-held / writable | **2,346 / 96 / 2,250** — the §7aw addendum's prediction, exact |
| changed ⊆ occupied | true, 0 outside |
| noac rows changed | **0** of 388 |
| reverted to v17 | **0** of the 192,435 rows v18 moved |
| newly / no longer occupied vs v20 | 0 / 0 — the class is a re-pin, as the desk said |

**196,256 − 1,573 − 2 = 194,681, and the 2 are named.** Two occupied, non-guard-held rows did
not change: one at LSZH, one at EHAM, **both pinned at the 60 s floor in v18 and in v7**. The
floor clamps them, so the bag cannot move them. That closes the changed-row count with no
residual.

**Two reads worth stating, neither of them claims.** *LTFM changes nothing* — it has **0**
occupied rows of its 47,581, so nine airports move and the tenth is outside surfacefit's class
by construction, not by omission. *The moves are not a blow-up* — over the changed rows the
absolute move is p50 10.41 s, p95 46.33 s, p99.9 164.28 s, max 907.50 s, and the ten moves over
500 s all sit on rows whose v18 value was already 1,322–2,869 s. Minimum 60.0, all finite,
template order on both files.

**The reading rule of record for this upload is the §7au addendum**: face +0.22025
[+0.16206, +0.29728] over 246.3605, at-face 246.1403, band 245.9178–246.2812, and a transfer
failure is a move of less than +0.0583 s from 246.3605.

## §7ay — the two §40 skeletons, and BOARD READING 18 as a fill-in

Two pieces of preparation while round 9 runs and the upload slot waits for 00:00 UTC 09-10.

### The v8 and v9 expectation skeletons

`work/v8v9_skeletons.py` writes `analysis/audit-expect-v8.json` and
`analysis/audit-expect-v9.json` from exp-046 §40's registered predictions. Each carries **48
claims — 34 hard, 14 pending** — and keeps three things deliberately apart:

- **Hard claims**, checked at tol 0 the moment a serve lands: the pins (296 / 297 / 218), the
  fold rows (190,713 / 165,677 / 153,706), the template invariants, and the whole class block
  including the three LFPG counts §7ax proved (2,346 / 96 / 2,250).
- **Pending**, for the figures only the read can supply. Their placeholders **are §40's
  predictions**, so a premature run reports PENDING and can never agree with a guess. Smoke
  test: the v9 skeleton run against a stand-in arm reports the measured 2A/B of 1.516 against
  the placeholder 2.0 as **PENDING**, not AGREE and not DISAGREE.
- **The registered prediction**, carried beside the claims as a record and never as a check —
  the audit says what the candidate *is*, the prediction says what it was expected to be, and
  the two must not contaminate each other.

**v8 is conditional and the file says so.** §40.1's own decision rule is copied in verbatim:
capacity is real only at **≥ +0.05 with every fold clear**; **≤ −0.05 with every fold clear**
closes the path on a measurement; anything between leaves it a MAYBE at a stated resolution.
Two of those three branches produce no serve, so the file is armed, not assumed. Its head is
the **bagged 255**, not the v18 head — pointing it at v18 would re-create read A's
two-chaos-vector problem §7ar found VOID.

**v9's prediction is falsifiable on a ratio**, so `work/audit.py` now reports
`ratio_2A_over_B` and `rms_increment_s` per fold. For reference the K = 5 bag over v18 reads
2A/B **1.516 / 1.641 / 1.635** with rms 14.42 / 18.79 / 16.00 s; §40.2 predicts the K = 10
increment at **2A/B ≈ 2.0 with rms ≈ 3.6 s** on july, and near 1.0 would mean the members are
not independent and §37.2's bagging account is wrong.

Both files register the same conditional §7at taught: every `bytes.class.*` count is claimed on
the standing v21 numbers, and if a serve ships a vector that moves occupancy, they are
re-pointed and reported first — not read as defects.

**Re-pointed on the desk's names (2026-09-09), and both placeholders were right.** v8's arm is
`work/d-bag5-511-free.npz` with record `analysis/bag-511.json` and members
`d-bundle-placebo{1..5}-511-free.npz` — the first of those five is already on disk, so
`bag511.py` is running. v9's arm is `work/d-bag10-255-free.npz`, from `bag10.py` with
`HEAD5 = d-bag5-255-free.npz`. Shas are pinned when each run lands, as v7's were.

**And the head is now pinned by its bytes in both files** —
`c6d1f5b6f366c7f885c3f3813ae37e46b28997ee8f0d919ba61fa55000c854ac`. `bag511.py` asserts its
head equals that file bit for bit and `bag10.py` takes it as `HEAD5`, so both increments are
priced over v7's shipped bytes. It is checked *before* the audit, because a moved head means
the increment is over a different base and the whole reading changes — which the audit itself
would not say. For v9 that pin does double duty: `HEAD5` being this exact file is what makes
the increment the registered `bag₁₀ − bag₅ = (T₅ − S₅)/10`, so §40.2's ratio prediction applies
to it directly.

### BOARD READING 18, registered as a template

`work/read_v19.py` and `analysis/v19-reading-template.json`. Every figure is fixed **before**
the upload — anchor 246.3605, face +0.22025 [+0.16206, +0.29728], at-face **246.1403**, band
**245.9178–246.2812**, failure line a move under **+0.0583 s** (score above **246.3086**), and
a score at or above the anchor is no transfer at all. When the board reports, one command turns
the number into the reading of record:

```
../../.venv/bin/python work/read_v19.py --score <score> --rank <r> --of <n>
```

It emits `analysis/v19-reading.json` and the paragraph, computing the move, the transfer
multiple, the band test, the failure verdict, the appended transfer history with a re-read
median, the display-step resolution, and the lead over second place. Worked examples are in the
template at all six decision points. **The script applies the rule; it never chooses it** — which
is the whole purpose, since a reading written after the score can always be fitted to it.

**Files.** `work/v8v9_skeletons.py`, `work/read_v19.py`, `analysis/audit-expect-v8.json`,
`analysis/audit-expect-v9.json`, `analysis/v19-reading-template.json`, `work/audit.py`.

## §7az — the v10 skeleton for §40.3's paired K = 10 capacity read

The desk's plan after surfacefit's member split (`B(K) = 29 + 331/K` on july — independent
chaos, small floor): v9 first as §40.2 registered it, then five more 511 members for a **paired**
K = 10 capacity read, `bag10-511` against `bag10-255`, pre-registered as §40.3 before those
fits. `work/v8v9_skeletons.py` now writes three skeletons; `analysis/audit-expect-v10.json` is
the third, 48 claims, 34 hard and 14 pending, on the same pattern.

**Why the paired read is a different instrument from §40.1's.** §40.1 bagged both sides only to
K = 5, so the chaos term it left is what made july unreadable — in-fold +17.901
**[−20.49, +47.48]**, the one fold whose interval straddles zero, and then the fold the
cross-fit selected. Halving the chaos on *both* sides is the repair, and the registered revival
condition is **july clear at the achieved resolution, reached by K** — not by a rule chosen
after the read.

**§40.3's july prediction re-derives from this lane's own numbers.** From §7ay's read of the
511 vector — july gain +17.901 with `B` 95.15, so **2A = 113.05** — and the desk's B(10) =
62.10, the predicted gain is **113.05 − 62.10 = +50.95** against §40.3's **+50.96**. A
0.012 s²/row agreement on a figure neither side took from the other. **What that agreement was
and was not: see §7bb — the read refuted the prediction, and the shared assumption underneath
both derivations is what failed.** The desk's model also
lands on my measured `B` at K = 5: 29 + 331/5 = **95.2** against **95.15**.

**December and january are consistency, not confirmation, and the file says so.** Their B(10)
was solved *from* §40.3's predictions, so those two deltas are zero by construction. What they
do yield is the implied chaos coefficient.

| fold | implied floor `c` | implied chaos `e` | naive from the member spreads | ratio |
| --- | ---: | ---: | ---: | ---: |
| july_loo | 29.03 | 330.6 | 357.7 | **0.924** |
| december | 42.09 | 532.5 | 580.1 | **0.918** |
| january | 34.25 | 426.0 | 468.7 | **0.909** |

Solving `B(K) = c + e/K` from the measured B(5) and the implied B(10) gives `e`; the naive
column is `(s₅₁₁² + s₂₅₅²)/(1 − 1/5)` from the member spreads. The ratio is **uniform across
three folds** — tight enough to be a systematic factor rather than noise. Recorded **before the
read** so it could not be fitted to the answer, and as an observation, not a claim.

**CLOSED by surfacefit §40.3 (`e0eef8a`) — and by neither hypothesis I offered.** `a + b/K` over
all C(5,k) subsets is exact algebra (residuals 1e-13), `a` is the mean pairwise inner product and
`b` is `S_d/(1 − 1/5)` — my own denominator — so the ratio is exactly
**`1 − 2·cov/(s₅₁₁² + s₂₅₅²)`**: the **cross-ladder covariance the paired design creates on
purpose**. Measured corr +0.072 / +0.078 / +0.091 gives 0.928 / 0.923 / 0.909 against my
0.924 / 0.918 / 0.909, and I re-derived the identity to four decimals — 0.9282 / 0.9225 /
0.9093. Because s₅₁₁ ≈ s₂₅₅ the expression reduces to about `1 − corr`, so the 0.92 was reading
the cross-ladder correlation directly.

**The sign is the good one.** A *positive* covariance makes the paired increment's chaos **lower**
than independence gives, so the naive column overestimated it and the paired read is slightly
better resolved than §40.3 needs. The floor `a` is untouched, so none of §40.3's predictions
move. Neither member dependence nor a conversion bias: it was the design working.

**The head moves for this one, and it is not pinned yet.** v10's head is
`d-bag10-255-free.npz`, not the K = 5 head v8 and v9 pin at `c6d1f5b6…`. The pin block carries
an explicit **NOT PINNED YET** in place of a stale sha, because a skeleton that runs against an
unpinned head cannot say the increment was priced over the base it claims. It is pinned the
moment `bag10.py` lands, before the file is run.

**v10 is conditional like v8.** Its gate is the cross-fitted reading under the standing
fold-selection rule — the same gate §40.1 was ruled on — so the skeleton is armed, not assumed.

**Files.** `work/v8v9_skeletons.py` (now writes v8, v9 and v10),
`analysis/audit-expect-v10.json`.

## §7ba — the v19 slot has two live branches, and both are registered before the upload

Desk ruling, 2026-09-09: the K = 10 bag over v18's bytes reads **+0.24275 [+0.18095, +0.32601]
with every fold clear**, passes the standing gate and outranks v7 by the `arm_choice` rule — so
the slot goes to the best K = 10 candidate as base set **v22**, and **v7 stays live as the
fallback** (it goes up unchanged at 00:01 UTC if the new candidate is not reviewed PASS by
22:00 UTC). The K = 10 − K = 5 increment is *not* claimed clear on july, so there is no v9 over
v7 on that comparison.

**A slot with more than one possible file needs more than one registered reading.**
`work/read_v19.py` now carries **a branch per candidate**, each fixed before any upload, and
applies the branch for whichever file actually went up:

| branch | face | at face | observed-transfer band | failure line | status |
| --- | ---: | ---: | --- | ---: | --- |
| `v7` | +0.22025 | 246.1403 | 245.9178 – 246.2812 | 246.3022 | **LIVE FALLBACK**, not superseded |
| `bag10-255` | **+0.24275** | **246.1177** | 245.8726 – 246.2731 | **246.2954** | primary unless §40.3 promotes the 511 |
| `bag10-511` | — | — | — | — | conditional; **refuses to read a score** until §40.3 lands |

Run it as `work/read_v19.py --branch <candidate> --score <s> --rank <r> --of <n>`. The
unregistered branch is not a blank to be filled in after the fact — it **raises** rather than
read a score, because a reading rule invented after the number can always be fitted to it. In
every branch a transfer failure is the lowest transfer ever observed (0.36×, v16) applied to
the lower end of *that branch's own* face interval, and a score at or above 246.3605 is no
transfer at all.

### The v9 file is no longer a skeleton

Re-targeted as the desk ordered: head **H** (`d-bundle-ungated1299-guard995.npz`,
`5dbbcb8c…`), arm `d-bag10-255-free.npz` (`d32a993b…`), previous the **v18 upload**, baseline
v17. Because `bag-10.json` has already run, its fold figures are **measured, not predicted** —
so they are hard claims at 1e-9 and only the byte half waits on the serve: **56 claims, 4
pending**. Run against the arm on disk, the fold half reads **30 of 30 answerable AGREE**:
july **+113.65454**, december **+256.65932**, january **+184.83630** s²/row, board
**+0.242753**, clean **+0.296451**, every fold lit and clear.

**§40.2's sharp prediction, and how it landed.** It forced `2A/B ≈ 2.0` with rms ≈ 3.6 s on
july for the K = 10 − K = 5 increment. Measured: **1.399** july, **2.159** december, **2.055**
january, rms 3.974 july. Two folds land on the prediction; july at 1.40 sits between §40.2's
two stated outcomes — recorded as an observation for surfacefit's account, before any ruling on
it. The board increment **+0.02250 [+0.00634, +0.03984]** is inside §40.2's registered
[+0.015, +0.035], and july's **+6.31 [−7.23, +19.80]** is the fold that does not clear, which is
the desk's ruling exactly.

### v8 and v10

Both re-anchored on the **v18 upload**, since v19's contents are no longer v7's. v10's head has
landed and is now **pinned by its bytes** (`d-bag10-255-free.npz`, `d32a993b…`) in place of the
NOT PINNED YET placeholder; its arm and predictions are unchanged and it stays conditional on
§40.3's cross-fitted gate. v8 remains armed but idle under the no-serve ruling.

**Files.** `work/read_v19.py`, `analysis/v19-reading-template.json`,
`work/v8v9_skeletons.py`, `analysis/audit-expect-v{8,9,10}.json`,
`analysis/audit-v9-foldread.json`.

## §7bb — §40.3 fails, and what my own re-derivation was worth

Desk, 2026-09-09: §40.3 **fails the cross-fitted gate** — **+0.01738 [−0.01443, +0.04283]**.
In-fold +0.08256, but july reads **+18.65 [−14.83, +42.68]** and crosses. Refutations 2 and 3
fire. §43.3's branch selects **bag10-255**, which surfacefit is serving now as base set v22
(head H, previous v18, ranking side ten members). v10 stays idle; **v9 is the audit**.

**The model split cleanly, and only one half was wrong.**

| | forecast | read | |
| --- | ---: | ---: | --- |
| chaos, `B(10)` | 62.10 | **62.11** | **held to 0.01** |
| signal, `A` on july | 56.52 | **40.38** | **28.6 % low** |
| gain `2A − B` | +50.94 | **+18.65** | — |

The whole 32.29 s²/row shortfall is `A`: **32.28** of it. The paired design's chaos arithmetic —
the same `B(K) = 29 + 331/K` whose 0.92 ratio §7az closed as cross-ladder covariance — forecast
the K = 10 churn to a hundredth of a s²/row. What failed is the assumption that the *alignment*
`A = mean(r·inc)` is invariant in K.

**And that is exactly what my §7az re-derivation could not test.** I reported +50.95 against
§40.3's +50.96 as "a 0.012 s²/row agreement on a figure neither side took from the other". The
arithmetic was independent and it was right. But **both derivations rested on the same
A-invariance assumption**, so the agreement was never evidence the prediction would hold — it
confirmed that two people had done the same algebra, not that the algebra described the world.
An independent check of a shared premise is not an independent check. Recorded here rather than
left standing as a confirmation, because that distinction is the whole value of the exercise:
the one part of the model I could genuinely test from my own read — the chaos term — is the
part that survived.

**v9 stands as written.** Its fold half already reads 30 of 30 answerable AGREE against the
served arm, its head is pinned at `5dbbcb8c…` and its arm at `d32a993b…`, and the four pending
byte claims run the moment the desk sends the sha. The `bag10-255` branch of
`work/read_v19.py` is the reading of record for the upload: face **+0.24275**, at-face
**246.1177**, band **245.8726–246.2731**, failure a move under **+0.0651 s** from 246.3605.

## §7bc — the v22 serve audits 54 of 54

Candidate `submissions/candidates/exp-046-surface-v8.parquet` (surfacefit's v8, this lane's v9
expectation), sha `d03d132a…`, base set v22, arm `d-bag10-255-free.npz`, head H, previous the
**v18 upload**, baseline v17, hold empty, occupancy read on v21 until v22 lands. Run:
`analysis/audit-v9-bag10.json`.

**Pin check first.** Both pins **MATCH byte for byte** — head H `5dbbcb8c…`, arm `d32a993b…`.
The arm is the same file §7bb's fold-side member check cleared of the ranking-side cache defect,
and it has not moved since.

**54 of 54 AGREE, nothing unanswered, two pending.** The fold half was already agreeing before
the serve (§7ba); the byte half now closes.

| | |
| --- | --- |
| rows changed | **194,680** = 2,250 at LFPG + 192,430 elsewhere |
| occupied / written / guard-held | 196,256 / 196,256 / 1,573 |
| LFPG occupied / guard-held / writable | **2,346 / 96 / 2,250** — the registered derivation again exact |
| changed ⊆ occupied | true, 0 outside |
| noac changed | **0** of 388 |
| reverted to v17 | **0** of the 192,435 rows v18 moved |
| newly / no longer occupied | 0 / 0 |
| template | 344,841 rows, order held, all finite, min 60.0 |

**The count closes with no residual, and the third row is explained.**
196,256 − 1,573 − **3** = 194,680. Three occupied, non-guard-held rows did not change, and all
three are **pinned at the 60 s floor** in v18: one at EDDF, one at LSZH, one at EHAM. Two are
the same rows v7 left at the floor. The third is the interesting one — **v7 moved it off the
floor to 65.39, and the K = 10 bag puts it back at 60.0**, because its bagged value falls at or
below the floor and `apply_floor` clamps it. That is the only row in the whole file where v7
changed something and this candidate does not.

**The two pending are pending because they were never registered**, not because they look
wrong: `rows_changed` (the bag's non-LFPG population, which the desk has not named) and
`rows_reverted_to_the_baseline`, which reads **0** — the same value §7ax registered and proved
on v7. Neither is promoted to a claim mid-run; a placeholder that happens to match is still a
placeholder.

**The reading of record for this upload** is `work/read_v19.py --branch bag10-255`: face
**+0.24275 [+0.18095, +0.32601]**, at-face **246.1177**, band **245.8726–246.2731**, and a
transfer failure is a move of less than **+0.0651 s** from 246.3605, with a score at or above
246.3605 no transfer at all.

### §7bc addendum — the v22 class vector, defective then corrected, and the audit holds on both

The first v22 emission repeated v20's defect class: **guard_held 1,546** against the serve's
1,573 and **written 193,077** against occupied 196,256. Before the corrected vector landed I
verified the part that decides whether §7bc stands — **`occupied_by_exp_046` and `obs_has_clock`
are identical to v21 row for row on both months, id order included** — so the audit of record,
which read occupancy on v21, needed no re-run for its verdict.

The corrected vector (`4c9a91b6…` / `a3a4aa3b…`, the defective one kept as
`class-manifest-v22-nogainedflag.json`) reads **54 of 54 AGREE** — the same verdict, with every
class count identical: occupied 196,256, written 196,256, guard-held 1,573,
`occupied_and_not_written` 0, `written_and_not_occupied` 0, LFPG 2,346 / 96 / 2,250, newly and
no-longer occupied 0. Both runs are kept: `analysis/audit-v9-bag10.json` on v21,
`analysis/audit-v9-bag10-on-v22.json` on v22.

**The shas are now PINNED** (`analysis/audit-expect-v9.json`, `the_class_vector_pins`).
~~The shas are recorded but not pinned yet.~~ `class-manifest-v22.json` landed with surfacefit's
commit **`c8457f0`**, and this lane pins on that commit rather than on what happens to be on disk —
the distinction that let the defective emission be caught rather than audited against. Verified on
the committed manifest and not on the message that announced it: both vectors hash to
`4c9a91b6…` / `a3a4aa3b…`, matcher `122c7d84`, rule `d4c59868…`, candidate `d03d132a…`, and
**occupied = written = served = 196,256** with guard-held 1,573, noac 388, rows 344,841, complement
148,585. The standing handshake rule is unchanged: this lane complements
**`occupied_by_exp_046`**, never `written_by_exp_046` — on v22 the two agree row for row, which
they did not on the first v20 emission.

## §7bd — a residual census of the complement, priced on v12 as served

The desk, 2026-09-09: surfacefit censuses the with-clock side and the whole file from its
interface, this lane the complement from its own, so the two halves are **two instruments and
not one**. No fits, no candidate. `work/complement_census.py`, `analysis/complement-census.json`.

**The complement** is `(~obs_has_clock) | noac` — the clock flag from this lane's
`artifacts/exp-047/block/<month>.parquet`, and noac as `AIRCRAFT_TYPE_mvt` null, exp-033's
definition, the one both lanes share. Priced as `r = y_true − p12`.

**The headline. The complement is 49.2 % of the rows the board scores and 77.8 % of the board's
squared error.**

| fold | complement rows | share of the fold | complement rms | fold rms | share of the fold's squared error | board-weighted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| july_loo | 102,618 | 53.8 % | 315.17 | 283.57 | 0.6647 | **0.3127** |
| december | 64,014 | 38.6 % | 237.31 | 204.20 | 0.5219 | 0 (no board share) |
| january | 84,205 | 54.8 % | **427.49** | 337.46 | **0.8791** | **0.4656** |

January is the fold that carries it: **87.9 % of january's squared error sits on 54.8 % of its
rows**, and the complement's rms there is 90 s above the fold's own.

**The ten largest cells by board-weighted share.**

| # | cut | cell | fold | rows | rms s | mean signed s | board share |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | band | **tail > 3600 s** | january | **230** | 7074.36 | **+1953.10** | **0.3483** |
| 2 | airport | **LFPG** | january | 17,900 | 796.75 | −1.70 | **0.3438** |
| 3 | airport | LIRF | july_loo | 2,086 | 1437.09 | −143.68 | 0.1322 |
| 4 | band | tail > 3600 s | july_loo | 401 | 2644.21 | +1584.11 | 0.0860 |
| 5 | band | d10 1559–3600 s | july_loo | 11,102 | 488.69 | +229.97 | 0.0813 |
| 6 | airport | LTFM | july_loo | 25,171 | 266.00 | +25.66 | 0.0546 |
| 7 | airport | LIRF | january | 11,317 | 375.33 | −5.68 | 0.0482 |
| 8 | airport | EGLL | july_loo | 19,442 | 273.53 | +5.18 | 0.0446 |
| 9 | band | d10 1559–3600 s | january | 7,528 | 418.12 | +207.71 | 0.0398 |
| 10 | airport | LFPG | july_loo | 21,231 | 243.59 | −8.88 | 0.0386 |

**The two cuts are two partitions of the same population**, so rows 1 and 2 overlap and the ten
shares do not add to 0.7783. Within one cut and one fold they add exactly — verified to 1e-16 on
both cuts and all three folds.

**Four reads, none of them a proposal.**

1. **230 rows carry a third of the board.** January's `> 3,600 s` tail is 0.15 % of the fold's
   rows, **65.8 %** of january's own squared error and **34.8 %** of the board's. Its mean signed
   error is **+1,953 s** — the base under-predicts by half an hour on average there, so this cell
   is almost pure *bias*, not variance.
2. ~~**LFPG january is the opposite shape.** 17,900 rows at rms 796.75 with mean signed −1.70 —
   large error with no bias at all.~~ **WRONG — struck, see §7be.** The two cells are very largely
   the *same rows*: **104 of LFPG january's 17,900 carry 91.8 % of its board weight**, and the
   near-zero mean signed error was 17,796 rows of small negative bias cancelling 104 rows of
   enormous positive bias. There is one problem here, not two.
3. **The tail is a tail everywhere but only january's is decisive**: july 401 rows at rms 2,644
   (+1,584 signed) and december 186 at 1,852 (+1,276) have the same sign and shape, so it is a
   systematic under-prediction of long taxi-outs across all three folds — january's is simply
   three times worse.
4. **The lowest band's edge is negative** — the pooled complement's minimum `y` is **−12 s**. That
   is a label read this census surfaces and does not chase; it affects a handful of rows in a band
   whose mean signed error is −120 s across all three folds.

**Files.** `work/complement_census.py`, `analysis/complement-census.json`. No fits, no candidate,
no serve.

## §7be — the census continued: the two cells are one, and the big one is the least reachable

Three questions §7bd left open, answered with instruments this lane already has and **no fits**:
does the airport cut overlap the band cut, which half of the complement carries the error, and
how much of it is reachable at all. `work/complement_census2.py`,
`analysis/complement-census-2.json`.

### 1. The overlap — and a correction to §7bd

§7bd read january's tail and LFPG january as "two cells of nearly equal board weight and
opposite shape". **They are very largely the same rows.**

| january, complement | rows | rms s | mean signed s | board share |
| --- | ---: | ---: | ---: | ---: |
| LFPG, all | 17,900 | 796.75 | −1.70 | 0.3438 |
| **LFPG ∩ tail > 3600 s** | **104** | **10,015.20** | **+2,403.21** | **0.3156** |
| LFPG, the other rows | 17,796 | 228.79 | −15.75 | 0.0282 |

**104 rows carry 91.8 % of LFPG january's board weight** — and 91.8 % of its squared error. The
"no bias at all" reading was an average: 17,796 rows of small negative bias cancelling 104 rows
of enormous positive bias. §7bd's point 2 is struck above. There is **one** problem here, not
two, and it is the extreme tail: 45 % of january's 230 tail rows sit at LFPG, and those 104
carry 91 % of the tail's board weight.

### 2. Which half of the complement — and a second concentration

| fold | no-clock only | noac only | both |
| --- | --- | --- | --- |
| july_loo | 102,279 rows, board 0.1920 | **275 rows, board 0.1057** | 64 rows, 0.0150 |
| january | 84,142 rows, board 0.4421 | 3 rows, 0.0015 | 60 rows, 0.0220 |
| december | 63,925 rows, board 0 | 38 rows, 0 | 51 rows, 0 |

January is all no-clock, as expected. **July is not**: 275 `noac`-only rows — a tenth of a
percent of the fold — carry **33.8 % of july's whole complement board weight**, at rms 3,540 s
and mean signed **−636**. Note the sign: these over-predict, where the january tail
under-predicts. Two different extreme populations, not one.

### 3. What is reachable, on the shipped airport × 10-minute grid

`base.shared` is exp-041's self-excluded cross-product `E[rᵢrⱼ]` — the part of a cell's squared
error a predictor **constant on that cell** could remove. An estimator, not a fit.

| fold | population | rows | reachable share | 95 % CI |
| --- | --- | ---: | ---: | --- |
| july_loo | complement, y ≤ 3600 | 100,935 | 0.1159 | [0.0956, 0.1409] |
| july_loo | complement, tail | 387 | **0.0298** | [0.0039, 0.0591] |
| january | complement, y ≤ 3600 | 81,702 | 0.1330 | [0.1040, 0.1733] |
| january | complement, tail | 212 | **0.0067** | [0.0016, 0.0158] |
| january | with-clock side, for contrast | 67,697 | 0.1282 | [0.1049, 0.1555] |

**The body of the complement is as reachable as the with-clock side** — 11.6–13.3 % against
12.8–13.9 % — which says the no-clock rows are not intrinsically harder, they are simply
unserved. **The tail is not**: january's is **0.67 %** reachable on this grid, a fifth of the
body's rate and twenty times below it in the interval.

So the cell that carries a third of the board is the one a cell-constant predictor can do least
with. **Stated as a limit of this relation, not of the rows**: 212 tail rows spread over an
airport × 10-minute grid leave 596 pairs, so the grid is nearly empty there and a different
relation — one that groups extreme taxi-outs by something other than when and where they
started — is untested, not refuted.

**Files.** `work/complement_census2.py`, `analysis/complement-census-2.json`. No fits, no
candidate, no serve.

## §7bd addendum — the check accepted, the cut redone by prediction, and the detection answer

Surfacefit's §47 check, via the desk: binning by the **label** makes `E[y − p | y large] > 0`
for any imperfect predictor — regression dilution. **Accepted, and §7bd's reading is withdrawn**:
"the base under-predicts long taxi-outs by half an hour" is error accounting, not a calibration
defect. `work/tail_detection.py`, `analysis/tail-detection.json`. No fits.

### 1. Re-cut by p12 decile — no calibration defect, and the board weight is somewhere else

| p12 decile | p12 range s | rows | mean signed s | rms s | tail rows | board share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 59–692 | 8,421 | −0.40 | 187.49 | 2 | 0.0090 |
| **4** | **864–931** | 8,420 | **+6.24** | **924.55** | **3** | **0.2178** |
| **5** | **931–996** | 8,421 | +0.30 | 647.50 | 1 | **0.1068** |
| 9 | 1,280–1,489 | 8,420 | −16.73 | 269.87 | 8 | 0.0186 |
| 10 | 1,489–64,459 | 8,421 | −25.01 | 522.51 | **209** | 0.0696 |

Conditioned on the prediction, **the signed error is small in every bin** — −25 to +6 s against
rms of 140–925 — so there is no calibration defect to find, exactly as the check predicts. What
the cut does show is that the board weight is **not** where the tail is: deciles 4 and 5 carry
**0.3246** of the board between them, against the top decile's 0.0696, even though the top decile
holds 209 of the 230 tail rows.

### 2. Why: it is a handful of rows, not a population

**96.9 % of decile 4's squared error is three rows**, the largest being **y = 84,240 s (23.4
hours) at p12 = 870 s**. Widening:

| | rows | board share |
| --- | ---: | ---: |
| the whole complement | 250,837 | 0.7783 |
| the worst **1** row (january) | 1 | **0.2103** |
| the worst 5 (january) | 5 | 0.3303 |
| the worst 20 per fold | 40 | **0.4135** — 53.1 % of the complement's |
| every complement row with **y > 12 hours** | **16** | **0.3516** |

**Sixteen rows whose labels exceed twelve hours carry 35 % of the board's squared error.** The
twenty worst january complement rows carry 73.2 % of that fold's complement squared error. So
§7bd's "two cells" are neither a systematic bias nor a structured variance — they are extreme
labels, and the census's own top-ten table was measuring them at second hand. Recorded as a
measurement; what to do about it — clip, winsorise, exclude, or accept — is a candidate decision
and the desk's, not this file's.

### 3. The detection question, answered

Target: the 230 january `> 3,600 s` complement rows, base rate 0.0027. **2,162 screens** over
185 prediction-time features × 8 quantiles × both directions.

| | screen | p | recall | precision | lift |
| --- | --- | ---: | ---: | ---: | ---: |
| **at the bar (p 0.12–0.18)** | `Fto_q_taxi_apt ≥ 18` | 0.125 | 0.374 | 0.0081 | **2.98×** |
| | `Fto_q_taxi_oth ≥ 9` | 0.122 | 0.283 | 0.0063 | 2.32× |
| **best lift at any p** | `Nf_n_anom_rwy_30m ≥ 1079` | **0.005** | 0.170 | 0.0931 | **34.08×** |
| | `Fto_s_taxi_rwy_rel ≥ 2.19` | 0.005 | 0.157 | 0.0870 | 31.84× |

**The backlog's 30× is real and it is the flow-state neighbour set — and it does not reach the
bar.** It lives at **p = 0.005**, catching 39 of 230 rows, which is a fifth of `freezing_precip`'s
p and the same problem exp-002 finding 14 named. At p ≈ 0.15 the best lift anything reaches is
**2.98×**, catching 37.4 % of the rows at a precision of 0.8 %. **No prediction-time screen in
this lane's interface reaches useful lift at p ≈ 0.15.** The sweep's multiplicity is stated in
the JSON: a point estimate chosen from 2,162 tries is a CANDIDATE, not a pass, and its bar is a
pre-registered re-run — the rule §7ao's reading F was held to.

### 4. LFPG january: the variance *is* predictable, on the body

Mean `r²` across bins, with day-clustered intervals on the top and bottom bins:

| cut | population | bins | rms range s | top/bottom | intervals disjoint |
| --- | --- | ---: | --- | ---: | --- |
| by hour | LFPG complement, y ≤ 3600 | 22 | 165.6–474.8 | **8.22×** | **yes** |
| by neighbour anomaly count | LFPG complement, y ≤ 3600 | 10 | 178.1–349.6 | 3.85× | **yes** |
| by flow-state decile | LFPG complement, y ≤ 3600 | 10 | 189.6–331.0 | 3.05× | **yes** |
| by hour | all LFPG complement | 22 | 170.5–3,097.7 | 330× | no — the extremes swamp it |

**Yes, at prediction time, on the body**: hour of day separates LFPG january's squared error by
**8.2×** with disjoint intervals, and flow state and neighbour anomalies by 3–4×. Including the
tail, every ratio inflates and every interval opens — the same handful of rows again. The honest
size of the prize: LFPG january's body is **0.0282** of the board, so this is a real structure on
a small cell, not the third of the board §7bd appeared to find.

**Files.** `work/tail_detection.py`, `analysis/tail-detection.json`. No fits, no candidate, no serve.

## §7be addendum — the trimmed-12h companion registered, and the §7ai MAYBEs re-priced on both reads

Two things, in the order the desk set them: the companion is **registered before it is used**,
and then the warm MAYBEs of §7ai are read twice, raw and trimmed. The ruling being followed is
the desk's of 2026-09-09: *"ACCEPT for the gate — the registered raw read stays the gate, because
the board scores raw MSE and the 2026 rows have their own outliers we cannot see. ADD a COMPANION
read, record-only, never a gate."*

### 1. The registration (commit `d0e96be`, written before the read)

`analysis/trimmed-12h-registration.json` freezes the exclusion list: fold rows whose label passes
**43,200 s**, removed *before* the read — from the residual, the increment and the board
arithmetic alike, so the companion is the same estimator on a smaller panel. Sixteen rows:

| fold | fold rows | excluded | in the complement | with a clock |
| --- | ---: | ---: | ---: | ---: |
| july_loo | 190,713 | 10 | 10 | 0 |
| december | 165,677 | 2 | 2 | 0 |
| january | 153,706 | 4 | 4 | 0 |

**All sixteen are in the complement, none has a clock**, so the companion cannot move a with-clock
read at all — it is an instrument for this lane's own endpoint and no one else's. `code/trim12h.py`
reads the frozen `MVT_ID_mvt` list rather than recomputing `y > 43200`, so the row set cannot
drift when the base moves; a read that trims a different set is a different instrument and needs
its own registration.

**What the trim actually removes is not "the base's worst errors."** Thirteen of the sixteen are
LIRF rows the base already tracks — `y` 87,605 s against `p12` 89,863, `y` 88,132 against 86,519 —
huge labels predicted as huge, contributing little squared error. The damage is two LFPG january
rows at `p12 ≈ 900` against labels of 23.4 h and 16.2 h, the first carrying 21.0 % of the board's
squared error on its own (§7bd addendum). The companion removes well-predicted extremes and
badly-predicted ones together, **by label and never by residual**, which is what keeps it an
instrument rather than a way to delete errors.

### 2. The instrument, validated exactly — and two corrections to how §7ai is quoted

The §7ai arm vectors are **not on disk**: `analysis/arms-step2-*.json` keeps summary statistics
only, and `code/arms.py` has no vector-saving option. So the registered arms were recomputed
(`work/shrunk.sh`'s invocation: rule `d4c59868…`, `--rows unmatched --fit-rows all`, arms
`N_reach_half,N_reach,N`, ridge, `BS.SEED`, `arms.ROTATION`). Nothing is searched, tuned or
selected; the learner is the registered ridge the arm *is*.

The first recomputation did not reproduce §7ai and **nothing was priced from it**, on the
condition set before the run. Two causes, both measured:

**(a) §7ai quotes a pre-handshake read.** `analysis/arms-step2-unmatched-fitall-3f-shrunk.json`
was written on matcher `ba41935e`; the block handshake then gave **1,170 july, 861 december and
731 january rows a clock** (`analysis/handshake-block-diff.json`, `clock_lost: 0`, no ids added or
removed), which moved them out of the `--rows unmatched` population. That emission was never kept,
so **§7ai itself cannot be recomputed**. What the record does hold is the same arms re-run on each
later matcher, and the reading survived the rebuild intact:

| arm | fold | §7ai `ba41935e` | rebuild `6fdf3951` | current `122c7d84` |
| --- | --- | ---: | ---: | ---: |
| N | july_loo | +503.643 | +503.848 | +450.175 |
| N | december | −372.546 | −371.609 | −316.461 |
| N | january | +234.845 | +234.804 | +230.326 |
| N_reach | july_loo | +619.390 | +619.608 | +576.376 |
| N_reach | december | −318.330 | −317.400 | −259.671 |
| N_reach | january | +220.777 | +220.754 | +216.603 |
| N_reach_half | july_loo | +589.043 | +589.264 | +541.161 |
| N_reach_half | december | −304.682 | −303.741 | −249.760 |
| N_reach_half | january | +271.169 | +271.169 | +267.128 |
| N_reach_half | board | +0.6567 | +0.6569 | +0.6131 |

**(b) The block emission is stored at float32, and cannot stand in for a ridge read.** The path
offered was to reproduce §7ai on `artifacts/exp-047/block-prev/`. That emission is the *same*
matcher generation as `analysis/arms-step2-rebuild.json` and gives the *same* rows (103,513 /
64,837 / 84,933) and the same clocks — so any gap between them is storage alone. It is not zero:
**96 of the 186 columns are written float32**, and the ridge turns that 10⁻⁷ relative rounding
into whole seconds squared.

| arm | july_loo | december | january |
| --- | ---: | ---: | ---: |
| N | −7.90 | −2.24 | −5.12 |
| N_reach | −9.51 | −2.61 | −5.98 |
| N_reach_half | −8.54 | −2.21 | −5.13 |

**A block read for a ridge arm is taken on the rebuilt block, never on the emission.** Recorded as
an instrument fact: the emission is faithful to the block in every row, every id and every clock,
and *not* in a fitted read.

So the validation was taken where it can be exact instead. `analysis/arms-step2-handshake.json` is
the same registered arms on the **current** matcher — the very block the re-price runs on — and
the recomputation reproduces it in **all nine cells and all three boards, delta +0.00000 s²**,
with the row counts exact (102,343 / 63,976 / 84,202). The instrument is validated; the re-price
below is read on that footing.

### 3. The re-price: raw and trimmed-12h

The trim reaches **7 of the 16 registered rows** here (july 2 of 10, december 1 of 2, january 4 of
4) — not because the set drifted but because this population is `~obs_has_clock` while the census
complement is `(~obs_has_clock) | noac`: nine of the sixteen are `noac` rows that *do* carry a
clock, so they are in the complement and not in the priced population. The registration counts and
these are the same rows read through two different masks.

**The arm's own gain against the v12 head** (§7ai's number), s²/row, day-clustered:

| arm | fold | raw | trimmed-12h | direction |
| --- | --- | --- | --- | --- |
| N | july_loo | **+450.18** [+236.87, +664.90] | **+449.81** [+239.39, +659.22] | down |
| N | december | −316.46 [−811.73, +91.20] | −316.98 [−823.26, +85.85] | down |
| N | january | **+230.33** [+65.26, +408.13] | **+192.72** [+37.57, +345.16] | down |
| N_reach | july_loo | **+576.38** [+332.26, +818.56] | **+576.01** [+340.73, +830.64] | down |
| N_reach | december | −259.67 [−749.83, +123.92] | −260.19 [−744.43, +135.03] | down |
| N_reach | january | **+216.60** [+35.83, +395.92] | **+197.86** [+11.07, +383.00] | down |
| N_reach_half | july_loo | **+541.16** [+317.69, +781.85] | **+540.80** [+308.36, +778.96] | down |
| N_reach_half | december | −249.76 [−730.96, +113.49] | −250.28 [−737.11, +136.99] | down |
| N_reach_half | january | **+267.13** [+102.59, +429.50] | **+238.96** [+91.90, +384.75] | down |

**The increment over arm `N`** — the form the lane's gate is written on (`work/increment.py`):

| arm | fold | raw | trimmed-12h | direction |
| --- | --- | --- | --- | --- |
| N_reach | july_loo | **+126.20** [+48.05, +207.82] | **+126.20** [+47.31, +206.20] | up |
| N_reach | december | +56.79 [−100.09, +207.85] | +56.79 [−91.56, +216.06] | up |
| N_reach | january | −13.72 [−186.68, +150.14] | +5.13 [−165.90, +172.08] | up |
| N_reach_half | july_loo | **+90.99** [+50.70, +133.43] | **+90.99** [+50.54, +132.36] | up |
| N_reach_half | december | +66.70 [**−4.13**, +147.00] | +66.70 [−4.47, +150.48] | up |
| N_reach_half | january | +36.80 [−48.23, +126.14] | +46.23 [−42.40, +142.36] | up |

Board gain over the head: `N` +0.5152 [+0.3044, +0.8268] raw against +0.6821 [+0.4114, +0.9501]
trimmed; `N_reach` +0.6091 against +0.8297; `N_reach_half` +0.6131 [+0.3888, +0.9547] against
+0.8268 [+0.5362, +1.0999].

### 4. The answer to the question the block was asked

**No arm is clear on every fold on the raw read, and therefore none is clear on BOTH.** The
answer does not depend on the companion at all:

- on the **own-gain** read, **december is negative on all three arms** and crosses zero on both
  reads — the arms lose ground on december and the trim does not touch that;
- on the **increment over `N`**, july is clear for both arms on both reads, and december and
  january cross. `N_reach_half`'s december is the near miss: **+66.70 [−4.13, +147.00]**, crossing
  by 4.13 s² of a 147 s² upper end.

**The companion's direction is favourable and changes nothing.** On the increment it moves every
fold **up** — january's `N_reach` even changes sign, −13.72 → +5.13 — because the extreme rows sit
in `N`'s favour, not the reach arms'. But the two folds that cross still cross, and the one that
would matter most, `N_reach_half` december, is *widened* by the trim ([−4.13, +147.00] →
[−4.47, +150.48]), not rescued: its 66.70 s² is not made of the trimmed rows. On the own-gain read
the trim moves every fold **down**, which is the honest direction — those extreme rows were
helping the arms' apparent gain, not hurting it.

**So the no-clock lane stays `closed-provisional`, and these are its branch-report reads.** The
warm MAYBE that stays warmest is `N_reach_half`'s increment over `N`: july clear at +90.99
[+50.70, +133.43] on both reads, december crossing by 4.13 s². Its revival condition is unchanged
and is now numbered: **december clear of zero on the increment over `N`, on the raw read, at the
achieved resolution** — a repair on december, not a re-read of the same numbers, and not a trim.

**Files.** `work/maybe_reprice.py`, `analysis/maybe-reprice-trimmed.json`,
`analysis/trimmed-12h-registration.json`, `code/trim12h.py`. No fits beyond the registered ridge
the arms are, no serve, no candidate.

## §7bf — PRE-REGISTRATION: the exp-048 precondition and backlog line 69's revival, before any read

Written and committed **before the reads run**, on the desk's order of 2026-09-09. This lane's
block is the surface-observation source exp-048's close-out names, and nobody has run the test on
it. Nothing here fits a head: **exp-033's rule stands — no noac row moves** — and there is no
candidate and no serve in this block. What is priced is a **precondition** (may an arm on this
lane ever be built?) and a **revival condition** (is there a 2026-valid placeholder instrument?).

### The reading of the assignment

The desk asked for three steps. My reading: step 0 is a population count and carries no bar; step
1 is the exp-048 screen, and its bar is exp-048's own sentence, not one I choose; step 2 is
backlog line 69's revival condition, and its bar has to be written here because line 69 states a
*kind* of answer and not a number. The reading I rejected: treating step 1's bar as "clear of
zero", which is the lane's usual gate and is **not** what exp-048 asks — a residual R² of 0.01
is clear of zero on enough rows and is exactly the number exp-048 says has failed three times.

### The quantities, frozen

| name | definition | source |
| --- | --- | --- |
| noac | `AIRCRAFT_TYPE_mvt` is null | exp-033, the definition both lanes use |
| the block | `artifacts/exp-047/block/<month>.parquet` for **counts and masks**; the block **REBUILT in memory** for the fitted read | §7be addendum §2 — the emission is float32 and shifts a ridge read by up to 9.5 s² |
| "carries a surface observation" | primary **`obs_has_clock`** (the row has an off-block clock); the wider **`obs_matched`** (the matcher reached the row at all) printed beside; the cell-level `reach_class` printed as context | this lane's own interface |
| the v8 residual, fold side | `r12 − d`, `d` = `exp-046-surfacefit/work/d-bag10-255-free.npz`, the served arm's fold-side increment over the v12 head | the vector v8's gate was read on |
| placeholder | `prc.tail_model.placeholder_label`: `BLOCK_TIME_UTC_mvt` and `SCHED_TIME_UTC_mvt` in the same **minute**, both present | exp-022's same-minute regime |
| the features | the **184 numeric block columns** — 186 less the key `MVT_ID_mvt` and the string `reach_class` (whose numeric twin `No_n_n_cell` is already in) | — |
| the ridge | `arms.fit_predict`, `alpha = EX.RIDGE_ALPHA = 10.0`, mean-imputed, standardisation fitted on the FITTING rows only | the registered learner, reused and not rewritten |
| the rotation | `arms.ROTATION` — july_loo ← december; december ← july_loo; january ← july_loo + december | registered |
| intervals | day-clustered bootstrap, `BS.BOOT = 2000`, `BS.SEED = 47` | the lane's standing instrument |

**An assertion, not an assumption:** `d` must be exactly **0 on every noac row**, because exp-033's
rule says no noac row moves. The run asserts it. If it is not zero the run stops and the desk is
told, because then the served bytes and the stated rule disagree and that is a bigger finding than
this block.

### Step 0 — the coverage census. No bar.

How many noac rows carry an observation, by month, on all five months (three fold, two ranking),
per airport as well as pooled. It is a count and it decides only whether step 1 has a population
to run on. Recorded expectation, so the census cannot be read as a surprise after the fact: LIRF
carries **383 of the 388 ranking noac rows**, and LIRF's coverage runs 0–87 % by month — it is
**dark for all 31 days of 2026-07** and healthy in 2026-01 (`arms.CENSUS`, the desk's 17:05 UTC
rule). So the census is expected to be strongly month-split, and a low count is a *finding about
the source*, not a failure of the test.

### Step 1 — the exp-048 screen. Bar: **out-of-sample residual R² ≥ 0.06** on the noac rows.

The bar is exp-048's own, quoted: *"A representation with a residual R² above about 0.06 on the
noac rows… The eight columns here reach 0.105 in sample, 0.030 out. A candidate must roughly
triple the out-of-sample correlation, which almost certainly means an observable outside
`NOAC_FEATURES_V2`"* (`experiments/exp-048-q14/RESULTS.md`, revival 1). The arithmetic behind it:
`ρ_c ≥ 0.259` closes the gap alone, `≥ 0.309` inside the budget.

- **Scored population:** the fold noac rows that carry an observation. Only those — a row with no
  observation has nothing for this source to say about it, and scoring it would price the
  imputed fit-side mean as if it were a reading.
- **Fitted population:** ALL rows of the rotation folds (`--fit-rows all`, the lane's registered
  convention: the columns are defined identically on both classes, so a head fitted on every row
  estimates the same coefficients from more data and is then scored only where it serves).
- **Primary number:** pooled over the three folds, `R² = 1 − Σ(r−ĥ)²/Σ(r−r̄)²` with `r̄` the
  scored population's own mean. The variant about zero, `1 − Σ(r−ĥ)²/Σr²`, is printed beside
  because the target is a residual and should already be centred; per-fold numbers beside both.
- **PASS = the point estimate reaches 0.06**, which is the letter of exp-048's condition. The
  day-clustered interval is printed with it, and **the block states the resolution**: on a
  population this small a point estimate is not by itself a finding, so a PASS at the letter with
  an interval spanning zero is reported as *"meets the letter, unresolved"* and never as a
  revival.
- **The null, registered now:** the same pipeline with the scored target **permuted within the
  scored fold** (200 draws, seed 47), giving the chance R² this estimator produces on this
  population when the row-to-feature link is broken. The screen is read against that
  distribution as well as against 0.06.
- **The context read:** the same estimator scored on 20 random equal-sized samples of **non-noac**
  rows with observations, which says whether 0.06 is even reachable on ordinary rows of this
  size — a bar nothing reaches is a statement about the bar.

**FAIL means:** the point estimate is below 0.06. exp-048's precondition is then measured as
unmet **on this source**, and the honest close is that the surface-observation programme — the
only live source its close-out names — does not carry the information either. That is a real
answer to a question the record has been holding open, and it is what this block is for.

### Step 2 — backlog line 69's revival condition.

Line 69 asks for *"a way to estimate P(placeholder) that is valid on 2026 and checkable without
the board"*. **Observation status is such a variable if it carries the signal**: the matcher runs
on the ranking months, so `obs_has_clock` exists at prediction time in 2026, and it needs no
label and no upload. The desk's framing: *the absence of a physical track for a flight with a
movement record* is itself evidence.

- **The read:** `P(placeholder | no observation)` against `P(placeholder | observed)`, on the
  three fold months, per airport and per month and pooled, with day-clustered intervals on both
  and on their ratio.
- **Bar, registered now:** the condition is met **in kind** when (a) the pooled intervals for the
  two probabilities are **disjoint**, and (b) the **direction is the same in all three fold
  months**. Clause (b) is what stops a single month's outage being read as a physical law — this
  lane has already been caught once by a month-split instrument (LIRF dark in 2026-07).
- **The size, reported whatever the verdict:** the lift `P(ph|no obs) / P(ph|observed)`, the
  absolute rates, and the share of placeholder rows the "no observation" screen would catch —
  because line 69's own arithmetic says 80 % of the price sits where the probability is nearest a
  coin flip, so a screen that moves it from 0.02 to 0.05 is not worth what a lift alone suggests.
- **This is not a classifier and does not become one in this block.** A PASS hands the record a
  measured, 2026-valid, label-free correlate; building anything on it is a separate decision with
  its own registration.

### What this block may not do

No arm, no head, no candidate, no serve, and **no noac row moves** (exp-033). The reads are
scored on the folds; the ranking months appear in step 0 as a population count only. Files will
be `work/noac_precondition.py` and `analysis/noac-precondition.json`.

## §7bf addendum — the reads: the precondition FAILS on this source, and the placeholder signal is composition

Bars registered at `cce5eeb` before any of this ran. `work/noac_precondition.py`,
`work/noac_strat.py`, `analysis/noac-precondition.json`. **exp-033's rule held before
anything was read**: the served arm vector `d-bag10-255-free.npz` is exactly `0.0` on every
noac row of all three folds, so no noac row moves and none moved here.

### 1. Step 0 — the census. The source barely reaches the rows it would have to serve.

| month | rows | noac | noac with a clock | noac matched | month clock share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-07 | 190,713 | 339 | 275 | 290 | 0.463 |
| 2025-12 | 165,677 | 89 | 38 | 46 | 0.614 |
| 2025-01 | 153,706 | 63 | **3** | 3 | 0.452 |
| **2026-01** | 152,719 | 108 | 38 | 60 | 0.590 |
| **2026-07** | 192,122 | 280 | **0** | **0** | 0.540 |

**Of the 388 ranking noac rows, 38 carry a clock — all of them in 2026-01.** 2026-07's 280
carry none at all, because 276 of them are at LIRF and LIRF is dark for all 31 days of that
month (`arms.CENSUS`, the desk's 17:05 UTC rule). The concentration is the one §47.8 names:
383 of 388 ranking noac rows are LIRF, and on the fold side it is 337 of 339 in july, 88 of 89
in december, 60 of 63 in january. **The noac population and this lane's blind spot are the same
airport.**

**A reconciliation, not a disagreement.** §47.8 counts **402** fold noac rows; this census counts
**491**. The difference is exactly december's **89**: §47.8's census is board-weighted and
december carries no board share, so 339 + 63 = 402 is july_loo + january. Both numbers are right
for their own population. The ranking side agrees to the row: 388, of which 383 LIRF, 4 LTFM,
1 LSZH.

### 2. Step 1 — the exp-048 screen. FAIL, and below its own null.

Out-of-sample, day-clustered, registered-rotation residual R² of the v8 residual on the noac rows
that carry a clock, against the 184 numeric block features, registered ridge (α = 10), fitted on
ALL rows of the rotation folds:

| fold | scored rows | fit rows | R² about the mean | R² about zero | ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| july_loo | 275 | 165,677 | **−0.0508** | −0.0168 | −0.139 |
| december | 38 | 190,713 | **−0.0440** | −0.0439 | −0.112 |
| january | 3 | 356,390 | −4.899 | −0.007 | — (3 rows, not a reading) |
| **pooled** | **316** | — | **−0.0496 [−0.1012, −0.0028]** | −0.0168 | −0.108 |

**The bar is +0.06. The measurement is −0.0496, and its interval is entirely below zero.**

Two registered controls say what that means:

- **the permutation null** (the scored target permuted within its own fold, 200 draws): median
  **−0.0265** [−0.0424, −0.0085]. The observed R² is *below* its own null — the ordinary
  signature of no signal at all plus fitted noise, not of a weak signal;
- **the non-noac context read** (the same estimator on 20 equal-sized samples of ordinary rows
  with a clock): median **−0.0222**. So this estimator does not reach +0.06 on ordinary rows of
  this size either, and 316 rows is not a population on which 0.06 was ever likely to be resolved.

**exp-048's revival 1 is measured UNMET on this source.** Its close-out named the
surface-observation programme as *"the only live source"* of an observable outside
`NOAC_FEATURES_V2`; the block is that programme's output, and on the rows it reaches its 184
columns carry **nothing** about the residual the noac head leaves. The requirement was
`ρ_c ≥ 0.259`; the measurement is ρ = −0.108. That is the answer to a question the record has
held open across exp-033, exp-037 and exp-048, and it is a *fourth* negative of the same shape —
the information, not the carrier and not the objective.

**And coverage caps it independently.** Even a passing screen would serve 38 of 388 ranking rows,
none of them in 2026-07. The two failures are separate and both fatal.

### 3. Step 2 — backlog line 69. The registered bar is met, and the registered read refutes it.

`P(placeholder | no observation)` against `P(placeholder | observed)`, placeholder =
`BLOCK` and `SCHED` in the same minute:

| month | placeholder rate | observed | no observation | lift |
| --- | ---: | --- | --- | ---: |
| 2025-07 | 0.0393 | 0.0390 [0.0366, 0.0416] | 0.0396 [0.0380, 0.0413] | 1.017 |
| 2025-12 | 0.0487 | 0.0467 [0.0440, 0.0494] | 0.0518 [0.0500, 0.0538] | 1.109 |
| 2025-01 | 0.0507 | 0.0455 [0.0421, 0.0485] | 0.0551 [0.0523, 0.0578] | 1.209 |
| **pooled** | 0.0459 | **0.0438 [0.0421, 0.0455]** | **0.0479 [0.0461, 0.0498]** | **1.095** |

The registered bar — pooled intervals disjoint **and** the same direction in all three fold
months — **is met on the letter**. The registered per-airport read refuses it:

| airport | observed | no observation | lift | disjoint |
| --- | --- | --- | ---: | --- |
| LIRF | 0.0933 (19,406) | 0.1045 (19,190) | 1.120 | **yes** |
| LEMD | 0.0390 (21,337) | 0.0427 (31,976) | 1.094 | no |
| LFPG | 0.0350 (1,629) | 0.0366 (57,761) | 1.047 | no |
| LEBL | 0.0509 (37,786) | 0.0503 (5,035) | 0.986 | no |
| EGLL | 0.0431 (19,035) | 0.0422 (41,175) | 0.978 | no |
| EHAM | 0.0353 (57,292) | 0.0324 (3,115) | 0.917 | no |
| EDDF | 0.0352 (45,308) | 0.0315 (8,846) | 0.895 | no |
| EDDM | 0.0488 (28,225) | 0.0434 (10,901) | 0.890 | no |
| LSZH | 0.0311 (29,557) | 0.0273 (3,374) | 0.878 | no |

**Four of the nine point the other way and eight of the nine do not separate.** Stratifying —
each airport's own contrast, weighted by its rows, with the same day-clustered bootstrap
(`work/noac_strat.py`):

| read | lift | 95 % interval |
| --- | ---: | --- |
| crude | 1.0951 | — |
| stratified by airport | **0.9963** | [0.9442, 1.0436] |
| stratified by airport × month | **1.0068** | [0.9610, 1.0507] |

**The pooled contrast is airport composition and nothing else.** The mostly-unobserved airports
(LFPG 3 % observed, EGLL 32 %, LEMD 40 %) happen to have higher placeholder rates than the
mostly-observed ones (EHAM 95 %, LSZH 90 %, EDDF 84 %), and LIRF — whose placeholder rate is
0.093–0.105 against 0.031–0.051 everywhere else — is half observed. Even **LIRF's own 1.12 is
composition**: within 2025-07 it is **0.904** and within 2025-12 **0.953**, both below one, and
2025-01's LIRF has **zero** observed rows, so the pooled LIRF contrast is january's unobserved
rows against the other months' observed ones.

**So line 69's revival condition is NOT met by observation status.** The absence of a physical
track is not evidence about a placeholder label once you hold the airport fixed — which is, read
back, exactly exp-033's own rule arriving from the other direction.

**A correction to my own registration (§7bf).** The bar I registered — pooled intervals disjoint
and the direction consistent across months — **passed on an artefact**. Clause (b) does not
protect against composition, because the composition is stable across months. What caught it was
the per-airport read the same registration required, and the stratification is that read
aggregated. Recorded as a defect in the bar, not in the result: a pooled rate contrast over
strata whose base rates and exposure shares both differ needs the stratified form *as the primary
number*, and any future read on this lane will register it that way.

### 4. What this block closes

- **exp-048's revival 1 is measured unmet on the source its close-out named.** Residual R²
  −0.0496 [−0.1012, −0.0028] against a bar of +0.06, below its own permutation null, on the only
  316 fold noac rows the programme reaches.
- **Backlog line 69's revival condition is not met by observation status**: 0.9963
  [0.9442, 1.0436] stratified, against a crude 1.0951.
- **The prize is unchanged and still large** — 402 fold rows, 15.37 % of board MSE, +23.93 s of
  board RMSE if they were perfect (§47.8) — and the two doors this block was sent to try are
  both measured shut.
- **What is not closed.** A source that reaches these rows at all: the matcher sees 0 of 2026-07's
  280 noac rows, so anything built on the harvest's own receipts rather than on matched clocks is
  untouched by this reading. And a placeholder correlate that is not a rate over strata — the
  stratified null here is about *observation status*, not about every 2026-valid variable.

No arm, no head, no candidate, no serve; no noac row moved.

## §7bg — PRE-REGISTRATION: the whole-class GBM arm, cross-fitted, before the read

Written and committed **before the read**, on the desk's order of 2026-09-09. T7i read
`gbm-imputed-800x6` on the whole no-clock class at **+222 / +233 / +134 s²/row** (july /
december / january) — **december positive where every ridge form fails it** — and stopped there:
§7e asked a capacity question about the two *served cells* and refused to pick a learner off that
table, because the cells had been chosen on the same three folds. The figures were never given
intervals and were never gated.

### The reading of the assignment

As a **whole-class** form the arm is not the served-cells selection, so §7e's post-selection
objection does not carry across unchanged: no cell is chosen, and the population is every
no-clock row. But **choosing the GBM after seeing december's sign is itself a selection on these
folds**, and the honest instrument for a form selected on the folds is already registered in this
programme: §33.3's cross-fit. So the gate is cross-fitted, the in-fold read is reported beside it
as the selection premium, and the trimmed-12h companion (§7be addendum, record-only) beside both.
The reading I rejected: pricing it in fold and calling december's sign a discovery — that is the
shape that priced v14 and v17.

### How this read is founded differently from what priced v14 and v17

The lane's two board readings are the prior to beat: **v14 −0.1914 s** (BOARD READING 13, transfer
1.03× on the all-row fold price) and **v17 −0.0482 s** (BOARD READING 16, 0.70×, inside the fold
interval). Both went *down*. Five differences, stated before the read so they cannot be chosen
afterwards:

1. **Population.** v14 and v17 were priced on the **served cells** — 3 of 19, chosen on these
   same three folds — and carried to the board by `project()`'s per-cell extrapolation, which is
   how a 141-row cell reached 17,650 ranking rows (§7h). This arm is priced on the **whole
   class**, with no cell selection anywhere in it.
2. **Learner.** Both were the registered ridge, whose class reading is saturated: four orders of
   magnitude of penalty move july's class gain from +503 to +518 (T7i). This is the one form
   change in the record that moves december's class gain **positive**.
3. **Gate.** Neither was cross-fitted. This one is, by §33.3's rule, because the form was chosen
   after seeing a fold.
4. **Base.** v14 sat over the pre-surface head and v17 over its own; this sits over **v8's bytes**
   (the last upload, v19) on the complement of v22 `occupied_by_exp_046`, so the increment is
   measured against what is actually on the board.
5. **Block.** T7i and v14 were read on earlier matchers; this is read on the **current** block
   (`122c7d84`, the turnaround pass), rebuilt in memory — never the float32 emission (§7be
   addendum §2). T7i's own figures are therefore **not** the ones this read must reproduce.

### The form, frozen exactly as T7i

| item | value |
| --- | --- |
| learner | `HistGradientBoostingRegressor(max_iter=800, max_depth=6, learning_rate=0.05, random_state=47)` |
| features | arm `N`'s columns — `Nf_*` and `No_*`, the numeric ones, from the current block |
| pre-processing | **mean-imputed and standardised on the FITTING rows only** (`work/foldside.py:standardise`), so no split can read an absent observation (the desk's 16:50 UTC rule) |
| population | the whole class: `--rows unmatched` |
| fitting rows | `--fit-rows all` on the rotation folds |
| rotation | `arms.ROTATION` — july_loo ← december; december ← july_loo; january ← july_loo + december |
| head | v12 (`p12`, `r12`) |
| intervals | day-clustered, `BS.BOOT = 2000`, `BS.SEED = 47` |
| projection | `arms.project` with `rank_weights("unmatched")`, the registered instrument, one floor |

### The gate, registered

- **The raw read is the gate** (the standing rule). The trimmed-12h companion is record-only.
- **Cross-fitted, §33.3's form:** for each scored fold `f`, the arm is taken on `f` **only if**,
  on **both** other folds `g`, the arm's class interval is clear of zero on `g`. Otherwise `f`'s
  gain is **exactly zero**.
- **The in-fold read** — every fold taken, the way a climb would have chosen it — is reported
  beside, and the difference is the **selection premium**.
- **Candidate condition:** the cross-fitted board gain clears the **0.1 s** floor, on the raw
  read. Anything less is a MAYBE with a revival condition and no candidate. **No serve in this
  block whatever the number** — the desk rules the serve.
- score-feedback rule stands: nothing here is fitted to a board reading.

### The prediction, with a band, and the fold I expect to fail

Registered so the read can refute me:

| quantity | prediction | band |
| --- | --- | --- |
| july class `2A − B` | +230 | +170 to +290 s²/row |
| december class `2A − B` | +235 | +150 to +320 s²/row |
| january class `2A − B` | +135 | +80 to +190 s²/row |
| in-fold board at face | +0.14 s | +0.08 to +0.22 s |
| **cross-fitted board** | **+0.00 s** | **0.00 to +0.06 s** |

**The fold I expect to fail is december.** Its class interval is the widest of the three under
every arm this lane has priced — ridge reads it at −373 **[−889, +40]**, a half-width of ~460 s²
against january's ~160 — so a point estimate of +233 is most unlikely to clear zero there. And
that has a structural consequence worth registering *now*, because it is the whole shape of the
gate: with three folds, **one crossing fold zeroes the other two**. If december crosses, july
loses one of its two required clear folds and january loses one of its two, so both are set to
zero and december alone could be taken — the fold that motivated the arm is the only one its own
weakness leaves standing. **I therefore predict the cross-fitted board at or near zero and a FAIL
against the 0.1 s floor**, with the in-fold read positive and the difference being the selection
premium the cross-fit exists to charge.

**Where I expect the whole-class gain to sit:** T7i says the GBM raises `EDDF/reached` by 1.4 to
1.8× on every fold and **loses `LTFM/unreached`**, whose december and january intervals stop
clearing zero. The class is roughly three-quarters unreached. So I predict the by-reach-class
split will put the gain in the **reached** rows and show the **unreached** rows at or below zero —
which, if it holds, means the whole-class form is the served-cells result wearing different
clothes, and the population it can actually serve is the small one.

**Files:** `work/gbm_class.py`, `analysis/gbm-class.json`. No serve, no candidate bytes.

## §7bg addendum — the read: the whole-class GBM FAILS the cross-fit, and its sign is fold-specific

`work/gbm_class.py`, `analysis/gbm-class.json`. Bars and predictions registered at `2e60e6c`
before this ran. The learner is T7i's own code — `work/foldside.py`'s `head()` and
`standardise()` imported, not rewritten — on the current block rebuilt in memory, arm `N`'s 24
columns, `--rows unmatched --fit-rows all`, the registered rotation.

### 1. The class read, against T7i and against my own prediction

| fold | class rows | GBM `2A − B` raw | trimmed-12h | T7i (pre-handshake) | my registered band | inside? |
| --- | ---: | --- | --- | ---: | --- | --- |
| july_loo | 102,343 | **−138.86 [−598.21, +321.60]** | −123.89 [−575.74, +306.30] | +222 | +170 to +290 | **no** |
| december | 63,976 | **+235.78 [−461.03, +1,133.72]** | +234.90 [−463.05, +1,146.48] | +233 | +150 to +320 | yes |
| january | 84,202 | **+237.08 [−11.35, +549.06]** | +194.84 [−42.38, +479.20] | +134 | +80 to +190 | **no** |

**My prediction is refuted on july, and it is refuted in the direction that matters**: I registered
+230 [+170, +290] and the measurement is **−138.86**, a sign flip. Two of my three bands missed.

**The refutation has a cause, and it is the same instrument fact as §7be's.** The ridge on these
exact rows reads +450.18 / −316.46 / +230.33 — reproducing `analysis/arms-step2-handshake.json` to
the digit — so the pipeline is right and the population is the current one. What moved is the
GBM's own sensitivity to it: between T7i's block and this one, **1,170 july rows of 103,514
(1.1 %) gained a clock and left the class**, and that moved the ridge's july class gain by −53 s²
and the **GBM's by −361 s²**. December, whose population moved by 861 rows, reproduces T7i almost
exactly (+235.78 against +233). **A form whose class reading moves 361 s² when 1.1 % of the rows
leave is not a stable object**, and that is a property of the form, not of the block.

### 2. The cross-fit takes nothing

**No fold's interval is clear of zero** — july crosses, december crosses by a wide margin, january
crosses at its lower end (−11.35). Under §33.3's rule each fold needs *both* others clear, so:

| fold | interval clear of zero | both other folds clear | taken |
| --- | --- | --- | --- |
| july_loo | no | no | **no** |
| december | no | no | **no** |
| january | no | no | **no** |

| board | july_loo | december | january | at the ranking shares |
| --- | --- | --- | --- | ---: |
| in fold | +0.4346 [−0.0487, +0.8590] | +0.7197 [−0.0138, +1.6139] | +0.3642 [+0.1178, +0.6608] | **+0.4034** |
| **cross-fitted** | 0.0 | 0.0 | 0.0 | **+0.0000** |
| in fold, trimmed-12h | +0.4535 | +0.7194 | +0.3263 | +0.3972 |

**FAIL against the 0.1 s floor. No candidate.** The selection premium is the entire **+0.4034 s** —
the largest this lane has ever charged, and a fair measure of what an in-fold read of a
fold-selected form is worth. Two of the three in-fold board intervals cross zero on their own, so
even the in-fold number is not a finding. The trimmed companion moves nothing (7 rows reach the
priced population, and the verdict is identical on both reads).

### 3. The decisive read: the GBM against the registered ridge, same rows

Priced with `work/increment.py`'s arithmetic — per row `(r − d_ridge)² − (r − d_gbm)²`:

| fold | GBM over ridge | clear of zero | rms move, GBM | rms move, ridge | rms of the difference |
| --- | --- | --- | ---: | ---: | ---: |
| july_loo | **−589.04 [−989.29, −207.93]** | **yes, BELOW** | 45.09 s | 19.98 s | 40.16 s |
| december | **+552.24 [+19.93, +1,217.53]** | **yes, ABOVE** | 41.32 s | 37.27 s | 38.28 s |
| january | +6.76 [−264.92, +327.84] | no | 24.73 s | 19.87 s | 24.81 s |

**The capacity change is real, resolvable, and points in opposite directions on two folds, each
clear of zero.** This is the sharpest statement the lane can make about T7i's question: there *is*
a capacity question — the GBM is not the ridge, it moves rows by 40 s RMS where the ridge moves
20 — and the answer to it **changes sign between july and december**, with both signs measured
clear. Under the every-fold gate that is a VOID, and the cross-fit reaches the same verdict by its
own route.

### 4. Where the gain sits — and the mechanism reverses fold to fold

| fold | reached (share) | unreached (share) |
| --- | --- | --- |
| july_loo | +534.26 [−919.23, +2,020.34] (25 %) | **−368.49 [−752.89, −1.23]** (75 %) |
| december | −248.45 [−1,194.72, +462.32] (34 %) | +487.22 [−325.59, +1,744.80] (66 %) |
| january | **+1,113.40 [+525.33, +1,798.65]** (18 %) | +48.80 [−286.00, +429.79] (82 %) |

I registered the expectation that the gain would sit in the **reached** rows and the unreached
rows would be at or below zero, because T7i says the GBM loses `LTFM/unreached`. **That holds on
january** (reached +1,113 clear, unreached ~0) **and on july** (unreached clearly negative, the
only interval in the whole table that excludes zero on the wrong side) — **and december reverses
it completely**: december's reached rows are *negative* and its whole positive number is carried
by the **unreached** rows, where the GBM has no observation to read and every `No_*` column is the
fitting rows' imputed mean.

**So december's +233 — the one number that motivated this block — comes from the rows the arm
should be worst on, by the arm's own mechanism.** A nonlinearity in the observation cannot be what
produced it, because on those rows there is no observation. That is an independent reason, beside
the interval, to read december's positive as noise.

### 5. Verdict

**FAIL. No candidate, no serve, nothing written.** Recorded as `closed-provisional` with the
screen cited, in the shape AGENTS.md asks for:

- **The negative and its cause.** The whole-class GBM's class gain is not clear of zero on any
  fold; its increment over the registered ridge is clear on two folds *in opposite directions*;
  and its one positive fold is carried by the unreached rows, which is the wrong mechanism.
- **Destination value, separately.** Unchanged and still real — the class is 148,585 ranking rows
  and the lane's own §7t states the headroom. Nothing here measures the destination non-positive.
- **The honest repair already made.** The trimmed-12h companion was run on every read in case a
  handful of extreme rows carried december's interval; it moves the point estimate by 0.9 s² and
  changes no verdict, so extreme labels are not what is wide about december.
- **The strongest refinement still visible.** Not a form change: july and december are *resolved*
  in opposite directions, so no re-read of these three folds can reconcile them. **Revival
  condition: a fourth fold month.** December's class half-width is ~800 s² against january's ~280;
  it is the fold that decides the cross-fit and it is the one the lane can least see. Failing
  that, a form whose class reading does not move 361 s² when 1.1 % of the population leaves.
- **What must not revive it:** re-reading these folds in fold. The in-fold board is +0.4034 s and
  every bit of it is selection premium.

score-feedback rule stands; nothing here was fitted to a board reading. The last upload (v19 = v8) is untouched.

## §7bh — PRE-REGISTRATION: recorded rule revival (a) on the current head, complement only

Written and committed **before the read**, on the desk's order of 2026-09-09. recorded rule held
`exp-040-chain.parquet` as a MAYBE with a first revival condition: *"recompose on the stronger
head when it exists"*. It has been measured twice since, and **december crossed zero both times,
with the value falling as the head improved** — against v11 the increment was +0.1325
[+0.0842, +0.1930] with december +0.090 [−0.021, +0.186]; against v12 it was **+0.11537
[+0.06865, +0.17293]** with december **+0.06793 [−0.04256, +0.16302]**
(`exp-040-chain/analysis/price-v12.json`). This is the third measurement, on **v8** — the head
the board holds — and it adds the reading the earlier two could not take: the chain update
restricted to **the complement of v22 `occupied_by_exp_046`**, the rows the surface programme
does not write.

### Why the complement read is a different question

Between v12 and v8 the surface programme rewrote the **with-clock** rows and nothing else. On
those rows the chain delta was aligned to v12's residual and that residual is now largely spent;
on the complement the residual is **unchanged**, because `d-bag10-255-free.npz` is exactly zero
there. So `r = y − v8` equals `r12` on the complement to the last bit, and the chain update's
alignment there is the same object it was when recorded rule was written. **If any of the chain member
survives on the current head, this is where it survives** — and it is the one population where the
question can be asked without the surface head in the way.

### The vectors, by sha

| file | sha256 (16) | what |
| --- | --- | --- |
| `exp-040-chain/packet/q09-july_loo.parquet` | `aff93c98f304c62d` | `y_true`, `regime`, `day`, `v10 replay`, `chain d11-noac` |
| `exp-040-chain/packet/q09-december.parquet` | `af729175984952c4` | — |
| `exp-040-chain/packet/q09-january.parquet` | `cdb6b68700873c17` | — |
| `exp-040-chain/oof/chain--july_loo.parquet` | `e1697ecafce2df1b` | the same chain vector, second source |
| `exp-040-chain/oof/chain--ranking.parquet` | `f3274f8855019253` | the ranking side, if a serve is ever ruled |
| `exp-046-surfacefit/work/d-bag10-255-free.npz` | `d32a993b48d18fa4` | v8's fold-side increment over `p12` |
| `exp-040-chain/code/price_v12.py` | `e7a15c378c317b8a` | the registered pricing, imported not rewritten |
| `exp-040-chain/code/price_v11.py` | `84ce1b7538adca72` | its `vectors()` and `price()` |
| `exp-038-packet/code/align.py` | `b6eec78bbbe5b14b` | `fold_stats`, `board_gain`, `boot_fold` — the instrument |

### The provenance gate, run first

**`increment_vs_v12` must reproduce exactly** — board `+0.11536514636873108`, july
`+0.12995813889290275`, december `+0.06792575973642556`, january `+0.10083904981843261`, and each
fold's `A` and `B` — by importing exp-040's own `price_v12.vectors()` and `price_v11.price()`.
Their `analysis/price-v12.json` is **read and never written**; nothing of exp-040's is modified.
If the reproduction misses, nothing is priced and the desk is told.

### The rows

- **(i) all rows the file moved:** the delta `u = chain − v10` is already zero outside
  `regime ∈ {nm_bulk, nm_late}`, so read (i) is `u` as it stands, over the whole fold — the same
  population `increment_vs_v12` prices, which is what makes the two numbers comparable.
- **(ii) the complement only:** `u` zeroed outside the complement. **Fold-side complement =
  `~(obs_has_clock | d_bag10 ≠ 0)`** — this lane's own `obs_has_clock` from the current block,
  widened by the rows v8's vector actually moves, which catches the chain mode-4 rows the block
  interface does not know (881 july / 857 december / 742 january). The definition errs towards
  *excluding* rows from the complement, which is the conservative direction for a
  complement-only claim.

**A stated approximation, measured rather than assumed.** `v8`'s fold analogue is taken as
`p12 + d_bag10`, so on the complement it is `p12`. That ignores this lane's *own* earlier serves
(v14, v17), whose bytes are in v8 on its three served cells and which lie inside the complement.
The read will report how many complement rows those cells hold, so the size of the omission is on
the record; the correction is second-order in `A` and belongs to the serve composition, which the
desk rules, not to this arithmetic.

### The gate

Every fold clear of zero on the **raw** read; the trimmed-12h companion beside it, record-only;
**floor 0.1 s on the board figure**. Board and intervals by exp-040's own instrument
(`align.board_gain`, `align.boot_fold`, 2,000 draws, 1-day blocks, seed 7), so this reading sits
in the same units as recorded rule's other two. No fit, no new member, no serve.

### The prediction, and the fold I expect to fail

| read | board gain | band | december |
| --- | ---: | --- | --- |
| (i) all moved rows, over v8 | +0.04 | +0.00 to +0.09 | crosses |
| (ii) complement only, over v8 | +0.06 | +0.02 to +0.10 | crosses |

**I expect december to fail**, on the record — it crossed on v10, v11 and v12 — and I expect
**both reads to miss the 0.1 s floor**.

**And I register the odd-looking part of that prediction, because it is the informative one:
(ii) may come out LARGER than (i).** The complement is roughly half the file and carries an
untouched v12 residual, while the with-clock half has had its alignment spent by the surface head
and may now be *negative* under the chain update. If (ii) > (i), the chain member's remaining
value lives entirely where the surface programme does not reach, which is precisely the
population recorded rule's third revival was meant to test. If instead (ii) ≈ half of (i), the update is
simply diluted and there is nothing special about the complement.

**Files:** `work/chain_v8.py`, `analysis/chain-v8.json`. No serve, no candidate bytes; the desk
rules any serve.

## §7bh addendum — the read: the chain update on v8 FAILS on december, and the value is still falling

`work/chain_v8.py`, `analysis/chain-v8.json`. Registered at `0c6bc52` before the read. No fit,
no new member, no serve; exp-040's `analysis/` was read and never written.

### 1. The provenance gate — exact

`increment_vs_v12` reproduces from exp-040's own `price_v12.vectors()` and `price_v11.price()`:

| quantity | recomputed | in the record | delta |
| --- | --- | --- | ---: |
| board | 0.11536514636873108 | 0.11536514636873108 | **0.00e+00** |
| july_loo gain / A / B | +0.129958138893 / +61.659217 / +49.631971 | same | **0.00e+00** |
| december gain / A / B | +0.067925759736 / +46.718902 / +65.701740 | same | **0.00e+00** |
| january gain / A / B | +0.100839049818 / +62.619474 / +57.190609 | same | **0.00e+00** |

And the registration's own claim is **proved, not assumed**: `y − v8` equals `r12` on the
complement, row for row, on all three folds — `d-bag10-255-free.npz` is exactly zero there.

### 2. The two reads

`r = y − v8` (fold RMSE 272.73 / 185.90 / 328.89), `d = chain − v10` on `nm_bulk + nm_late`, the
served file's 60 s floor applied:

**(i) all rows the file moved** — 143,906 / 127,663 / 118,775 rows moved:

| fold | raw | clear | trimmed-12h | A | B | A/B |
| --- | --- | --- | --- | ---: | ---: | ---: |
| july_loo | **+0.11150 [+0.04347, +0.18534]** | yes | +0.11562 | +55.22 | +49.63 | 1.113 |
| december | **−0.00417 [−0.12076, +0.09451]** | **no** | −0.00417 | +32.08 | +65.70 | **0.488** |
| january | **+0.07292 [+0.02911, +0.15619]** | yes | +0.12277 | +52.58 | +57.19 | 0.919 |
| **board** | **+0.09221** | — | +0.11702 | | | |

**(ii) the complement only** — 94,316 / 59,799 / 79,049 of the moved rows:

| fold | raw | clear | trimmed-12h | A | B | A/B |
| --- | --- | --- | --- | ---: | ---: | ---: |
| july_loo | **+0.10090 [+0.05630, +0.14438]** | yes | +0.10469 | +43.06 | +31.10 | 1.385 |
| december | **−0.00609 [−0.11880, +0.07795]** | **no** | −0.00610 | +17.66 | +37.58 | **0.470** |
| january | **+0.06361 [+0.02667, +0.13268]** | yes | +0.10709 | +42.47 | +43.10 | 0.985 |
| **board** | **+0.08229** | — | +0.10446 | | | |

**Both FAIL.** December crosses zero on both reads and both boards sit under the 0.1 s floor.

### 3. The trajectory is the finding

| head | board increment | december |
| --- | ---: | --- |
| v10 (composite) | +0.5949 | clear |
| v11 | +0.13249 | +0.090 [−0.021, +0.186] |
| v12 | +0.11537 | +0.068 [−0.043, +0.163] |
| **v8 (this read)** | **+0.09221** | **−0.004 [−0.121, +0.095]** |

**The value falls monotonically as the head improves, and december's point estimate has now
crossed to negative.** recorded rule's first revival condition was *"recompose on the stronger head when it
exists"*; it now has three heads' worth of answer, and recomposing does not rescue december — it
walks it down. The fourth measurement is the one that puts december's point estimate on the wrong
side of zero.

**And december says what is wrong with it, in one number.** Its fitted scale `A/B` is **0.488**
on read (i) and **0.470** on (ii), against july's 1.11 and january's 0.92: the update is about
**twice the amplitude december wants**, on both populations. That is a form statement, not a
noise statement — december's alignment is real and positive (`A` +32.08) and the update
overshoots it, which is why `2A − B` lands at zero.

### 4. My registered prediction, and the half of it that is refuted

I registered (i) at +0.04 in [0.00, +0.09] and (ii) at +0.06 in [+0.02, +0.10]. **(i) came in at
+0.09221, just outside the top of its band** (by 0.002) and **(ii) at +0.08229, inside**. Both
low, as predicted; december failed, as predicted.

**The informative half is refuted.** I registered that (ii) might *exceed* (i), which would have
meant the surface head had spent the chain's alignment on the with-clock rows and the member's
residue lived only where the surface programme does not reach. It did not: **(ii)/(i) = 0.892**,
so the with-clock half still contributes **+0.0099 s** of board — positive, not negative. The
chain member is **diluted by the better head, not relocated by it**.

What *is* true, and is the useful residue of the question: the complement carries **89 % of the
remaining value on about half the rows** (53 % of july, 38 % of december, 54 % of january), so the
value is concentrated there — but there is no separate complement effect to serve, because the
complement-only read is simply the larger part of a whole that fails on the same fold.

### 5. The trimmed-12h companion, and why it changes nothing

The companion moves january by **+0.049 s** on (i) and **+0.043 s** on (ii) — the largest trim
effect this lane has measured, four rows over 12 h — and it lifts **both** boards above the floor
(+0.11702 and +0.10446). **It is record-only and never a gate** (the desk's ruling), and it does
not matter here anyway: **december is unmoved by the trim** (−0.00417 → −0.00417) and december is
the fold that fails. A companion that lifts a board over a floor while the deciding fold does not
move is exactly the case the record-only rule exists for.

### 6. The stated approximation, measured

`v8`'s fold analogue is `p12 + d_bag10`, so on the complement it is `p12`. This lane's own served
cells — **`EDDF/reached`, `LEMD/reached`, `LTFM/unreached`** — lie inside the complement and carry
v14/v17 bytes this read does not model: **38,263 (37.7 %), 24,577 (38.9 %), 24,233 (29.0 %)**
complement rows. The direction is known even though the size is not modelled: this lane's own
correction *reduces* the residual on those rows, so the true `r` there is smaller than the one
used, and **read (ii) is if anything a slight over-estimate**. It is not enough to matter — (ii)
would have to gain 0.02 s and december would still cross — but it is on the record, and an exact
composition belongs to a serve, which the desk rules.

### 7. Verdict

**FAIL on both reads. No candidate, no serve, nothing written. recorded rule stays a MAYBE, and this is its
third measurement** — the one that shows the value decaying with head quality and december
crossing to negative. The ranking vector `oof/chain--ranking.parquet` (`f3274f88…`) exists and is
untouched.

**Revival condition, unchanged in kind and now numbered:** december clear of zero on the raw read.
Two routes are visible and both are form changes, not re-reads: **an amplitude** — december's
`A/B` is 0.47–0.49 on both populations, so a half-amplitude chain update is the arm the number
itself asks for, and it would be a new registration and a new selection, not this one; or **a
fourth fold month**, december's interval being ±0.11 s wide on a −0.004 s point estimate. What
must not revive it is another recomposition on a better head: three heads now say the same thing
in the same direction.

## §7bi — the harvest-extension census

A census, no bars, no fit, no candidate. **No download was made**: every input was already on
disk, and the one-day sample the desk allowed was not needed. `work/harvest_census.py`,
`work/winter_probe.py`, `analysis/harvest-census.json`, `analysis/winter-probe.json`.

### The answer to question (1) is larger than the question: the download is already done

`exp-045-harvest` harvested **all fourteen months** — `analysis/missing.json` is empty,
`analysis/receipt.json` carries a measured rate for every airport-day, and
`artifacts/exp-045/summaries` holds **411 MB** locally, 427 days at each fully covered airport.
The nine non-fold 2025 months would have cost **836.3 GB** of `adsblol/globe_history_2025` at a
median 3.21 GB a day; that is spent. **A harvest extension is not a download programme.** What is
missing is downstream, and it is measured below rather than estimated.

### The table — one line per 2025 month

`src GB` is what the source held (already consumed); `clocks` is `obs_has_clock`, **measured** by
building the block where the block exists or was built here (2025-02, -03, -11), and estimated
from the receipt rate × 0.928 elsewhere; `winter 4` is clocks at **LIRF + EGLL + LEMD + LFPG**.

| month | fold? | winter? | days | src GB | departures | receipt | clocks | winter 4 | missing summary days |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-01 | **fold** | yes | 31 | 66.4 | 153,706 | 0.4800 | **69,504** m | **8,675** m | 99 |
| 2025-02 | — | yes | 28 | 65.4 | 143,731 | 0.4994 | **67,584** m | **8,120** m | 84 |
| 2025-03 | — | yes | 31 | 78.4 | 164,450 | 0.5543 | **86,252** m | **17,486** m | 72 |
| 2025-04 | — | — | 30 | 82.2 | 175,288 | 0.5193 | 84,477 e | 0 | 69 |
| 2025-05 | — | — | 31 | 96.7 | 185,202 | 0.5693 | 97,848 e | 0 | 72 |
| 2025-06 | — | — | 30 | 97.6 | 183,142 | 0.5626 | 95,609 e | 0 | 46 |
| 2025-07 | **fold** | — | 31 | 108.6 | 190,713 | 0.4961 | **88,370** m | 19,377 m | 32 |
| 2025-08 | — | — | 31 | 108.7 | 191,181 | 0.5316 | 94,320 e | 0 | 32 |
| 2025-09 | — | — | 30 | 104.8 | 183,951 | 0.5553 | 94,789 e | 0 | 30 |
| 2025-10 | — | — | 31 | 106.8 | 185,674 | 0.5880 | **101,318** e | 0 | 31 |
| 2025-11 | — | yes | 30 | 95.7 | 162,332 | 0.6466 | **98,653** m | **30,383** m | 30 |
| 2025-12 | **fold** | yes | 31 | 95.6 | 165,677 | 0.6521 | **101,701** m | **33,355** m | 31 |
| **nine non-fold months** | | | **273** | **836.3** | **1,595,551** | | **≈ 820,000** | **56,000** | **466** |

The estimate is good: on the three months built here it ran **1.3 % to 2.0 % low** (2025-11
97,407 estimated against 98,653 measured), so the six estimated rows are slightly conservative.
Every month's `missing summary days` is dominated by **LTFM**, which has 16 summary days in the
whole archive and a receipt rate of **0.0000 in every 2025 month** — LTFM is structurally dark and
**no month of harvest can fix it**; its 47,577 ranking rows are `Nf_*`'s alone (§7t).

### Question (3): what a month actually costs on this laptop

| stage | cost | whose |
| --- | --- | --- |
| download | **0** — already held | done |
| block build (ranking-masked month + prefix frame, 186 columns) | **12.5–15.8 s** measured, 2 threads | this lane |
| entry as a **fold** month | an **out-of-fold refit of the base head** | another lane |

`base.load()` needs the v10 panel, the stack-5 OOF vector and the v12 vectors on the month, and
none of the three exists for a non-fold month. **That refit is the whole cost of the programme**;
the surface side of it is fifteen seconds. Two things do *not* need it: the prefix frame already
uses all nine non-fold months (`base.prefix_frame`), and the neighbour families need no label and
no head.

### Question (4): the winter rows, measured

| airport | 2025-01 (fold) | 2025-02 | 2025-03 | 2025-11 | 2025-12 (fold) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIRF | **0** | **0** | **0** | **2,927** | 6,006 |
| EGLL | 8,674 | 8,120 | 5,075 | 8,322 | 9,013 |
| LEMD | **1** | **0** | 12,411 | 16,293 | 17,188 |
| LFPG | **0** | **0** | **0** | 2,841 | 1,148 |
| **the four** | **8,675** | 8,120 | 17,486 | **30,383** | 33,355 |

**The january fold is the least-covered month in the archive** — 99 missing airport-days, more
than any other — and its entire winter-four coverage is one airport: **8,674 of its 8,675 clocks
are EGLL**, with LIRF 0, LEMD 1, LFPG 0. The desk's "january has 0 writable LIRF rows against
5,700 on the board" is one instance of a three-airport hole.

### The reading — the best two to three months

**2025-11 first, and it is the right first pick under *either* answer of the composition-neutral
control.**

- If the head is **COMPOSITION-limited**: 2025-11 adds **30,383** winter-four clocks, **1.7× the
  next best month** and **3.5× what the january fold itself carries**. It is the **only** non-fold
  month with all four winter airports non-zero, and outside december it is the **only** source of
  LFPG winter observation (2,841 of the 3,989 the whole archive can offer) and of LIRF winter
  observation (2,927; 2025-02 and 2025-03 have none). Adding it lifts winter-four coverage from
  42,030 to **72,413, a 72 % increase**.
- If the head is **SAMPLE-limited**: 2025-11's 98,653 clocks are ~~**second of the nine**, within
  4 % of 2025-10's ~101,300~~ **third of the nine, within 3.5 % of 2025-10's 102,262 and of
  2025-05's 102,030** (the estimate is replaced by the built blocks — see the §7bi addendum) — so
  choosing it costs about 3,600 rows against the volume-optimal pick and buys the entire winter
  repair.
- It is also the **cleanest** month: 30/30 summary days at all nine reachable airports, nothing
  missing but LTFM.

**Second pick, and here the control's answer does matter.** SAMPLE-limited → **2025-10**
(**102,262** clocks measured, the largest of the nine, no winter; **2025-05** is a near-tie at
102,030). COMPOSITION-limited → **2025-03** (17,486 winter-four, which is LEMD 12,411 and EGLL
5,075 — it repairs january's LEMD hole and nothing else; LIRF and LFPG are still zero).

**Third: 2025-05 or 2025-06** on volume (**102,030 / 97,753** measured, replacing the estimate's
2025-09 / 2025-08 ordering; both nearly complete).

**And one month to rank last among the winter candidates: 2025-02.** Its entire winter-four
contribution is **EGLL, 8,120** — precisely the one airport of the four that january already
covers. It is winter by the calendar and adds no composition the record lacks.

**What the census cannot say.** Whether rows or composition is the lever — that is the control's
question, and this is the map that answers "then which months", not "which limit". Nor does it
price a fold month's *gain*: the writable rows are what a month makes available to the head, not
what the head would win from them.

### §7bi addendum — the nine non-fold blocks are built, and what a non-fold month cannot carry

`code/emit.py --sources F,O,H,N,geo --rule d4c59868…`, matcher **`122c7d84`** on every one, the
same two builders every fold and ranking block was written against. **344.9 MB** into
`artifacts/exp-047/block/`, 133 GB free before and after. `work/nonfold_manifest.py`,
`analysis/nonfold-blocks.json`.

| month | rows | clocks | rate | cols | usable | MB | sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-02 | 143,731 | 67,584 | 0.4702 | 185 | **173** | 29.5 | `755f9b1e281fb76d` |
| 2025-03 | 164,450 | 86,252 | 0.5245 | 185 | **173** | 35.1 | `355cdeeb9c58e39a` |
| 2025-04 | 175,288 | 87,583 | 0.4997 | 185 | **173** | 36.7 | `e429203e1e945d08` |
| 2025-05 | 185,202 | 102,030 | 0.5509 | 185 | **173** | 39.6 | `a0d2217ad524e73e` |
| 2025-06 | 183,142 | 97,753 | 0.5338 | 185 | **173** | 39.6 | `1e87f333163e9dc9` |
| 2025-08 | 191,181 | 95,715 | 0.5007 | 185 | **173** | 42.6 | `b66d2ae58a40dc05` |
| 2025-09 | 183,951 | 95,032 | 0.5166 | 185 | **173** | 41.3 | `5f340d4eaa549b70` |
| 2025-10 | 185,674 | **102,262** | 0.5508 | 185 | **173** | 42.4 | `d6eebd2b367813f5` |
| 2025-11 | 162,332 | 98,653 | **0.6077** | 185 | **173** | 38.0 | `c2ba127f3fd0b5fd` |
| **total (9)** | **1,574,951** | **832,864** | **0.5288** | | | **344.9** | |
| 2025-01 *(fold)* | 153,706 | 69,504 | 0.4522 | 186 | 186 | 39.4 | `eb588315471b0846` |
| 2025-07 *(fold)* | 190,713 | 88,370 | 0.4634 | 186 | 186 | 52.5 | `00e7f9df5c83b086` |
| 2025-12 *(fold)* | 165,677 | 101,701 | 0.6139 | 186 | 186 | 48.9 | `e167c4df047cdb01` |

**§7bi's receipt-based estimate is now replaced by measurement and it held**: it ran **0.3 % to
4.1 % low** on the six months it had estimated, conservative in every one. The ordering moved
once — 2025-11 is **third** on volume, not second, behind 2025-10 (102,262) and 2025-05 (102,030);
§7bi's three volume lines are struck and corrected in place. **The winter argument is untouched**:
neither 2025-10 nor 2025-05 is a winter month, so 2025-11 remains the first pick.

**The schema is 185, not 186, and only 173 columns carry a value.** This is the finding of the
build, and it is the same fact the programme question below asks about, arriving from the other
side:

| column | on a fold month | on a non-fold month |
| --- | --- | --- |
| `own_obs_taxi_minus_p11` | present | **absent** |
| the twelve `{Nf_,No_}_n_res_*` | populated | **emitted all-NaN** |
| the other 173 | populated | populated, identical construction |

The twelve are `Nf_/No_` × `n_res_{cell, rwy_cell, apt_30m, rwy_30m, rwy_60m, rwy_minus_apt}` —
**exactly half of the 24-column `N` arm, which is this lane's `SPLIT_ARM`**. A neighbour residual
is `neighbour's taxi − v11's prediction for the neighbour`, so it cannot exist without a v11 for
the month, and `emit.py`'s `predictions()` correctly returns `None` rather than reaching for an
in-sample one. **So `N_reach` cannot be fitted on a non-fold month today.** The 173 that survive
need no head at all and are complete: every `F`/`O`/`H` clock, every `_sc` receipt-scaled count,
the geometry (`route_km`, `route_km_rel`, `stand_dist_km`), the six `*_n_anom_*` columns that
stand in for the residuals, and the identity columns (`obs_matched`, `obs_has_clock`,
`reach_class`).

The blocks reproduce §7bi's independent measurement exactly — 2025-11 98,653 clocks, 2025-03
86,252, 2025-02 67,584, the same three figures `work/winter_probe.py` returned by a different
route — so the emission and the probe agree.

### §7bi addendum, part 2 — what a non-fold month would cost in a price base

**One line, as asked: no model is persisted anywhere in this repository, so a non-fold month
cannot be predicted and every member must be refit — and one of the seven cannot be refit at all.**

**Nothing is stored but predictions.** Every `save` call under `experiments/` and `prc/` is an
`np.save` or `to_parquet` of a *prediction vector*; there is no `.pkl`, `.joblib`, `.cbm`, `.ubj`,
`.bst` or `.model` anywhere outside `.venv`. `prc.blend.stack_members` **joins member files on
`MVT_ID_mvt`**, and `prc.blend.build_candidate` reads `config["ranking"]` as *"a parquet with
MVT_ID_mvt and y_pred over the ranking rows"*. The whole stack is a set of row-keyed vectors.

**And they exist on exactly four populations.** `experiments/exp-012-blend2/published/` holds
`<member>-{july_loo, december, january, ranking}.parquet` and nothing else. **There is no member
vector for any non-fold 2025 month**, and none can be produced by reading: those months are
*training* data for every member, so any prediction on them would be in-sample. A non-fold month
entering as a fold is a **fourth leave-one-out fit of every member**.

**What the refit costs, measured, from exp-025-cloud §3.5 / §9.2 / §10.2:**

| member | weight (`nm_bulk` / `nm_late`) | accel | one fold |
| --- | --- | --- | ---: |
| F1-fresh | **0.478 / 0.400** | GPU | **0.35 GPU-h** (nested, ten months: ≈0.32) |
| T-tailv2 | 0.226 / 0.0 | CPU | 7.1 min (7.7 min real, Kaggle 4 vCPU) |
| V3-allin | 0.212 / 0.296 | CPU | **cannot be refit — see below** |
| L2-hub | 0.084 / 0.0 | CPU | 26.6 min |
| N1-nmtail | 0.0 / 0.206 | CPU | 12.2 min + 4.6 min cache pre-steps |
| D1-winter | 0.0 / 0.082 | CPU | 17.7 min |
| C2-cat | 0.0 / 0.030 | GPU | not recorded |

≈0.35 GPU-h plus ≈1.4 CPU-hours a fold on Kaggle, and a Kaggle vCPU is **2.2× slower per thread**
than a core here (§3.4: the same 2-thread fold, 35.3 s here against 77.6 s there).

**The blocker, and it is not compute.** `V3-allin` — **21 % of `nm_bulk` and 30 % of `nm_late`** —
**cannot be refit on any machine we have**. Its v3 frame calls `prc.weather_features`, which
raises `FileNotFoundError: no Copernicus ERA5 snapshot found under data/external`; exp-025-cloud
§10.3 records that the directory exists on **neither Kaggle nor this laptop** and that
`~/.cdsapirc` is absent, *"so it cannot be re-downloaded until Matt returns"*. **Verified live
here, not quoted**: `import prc.weather_features` still raises that `FileNotFoundError` at import
time, and both paths `_snapshot_dir()` tries — `data/external/weather/copernicus_era5` and
`/path/to/prc-data-challenge-2026/data/external/…` — are absent, as is `~/.cdsapirc`. exp-025 met this
before and worked around it by reading `V3-allin` on its **unnested** vector — a workaround that
exists only because an unnested vector for that population was already on disk. **For a new month
there is no vector to fall back to**, so the workaround does not transfer.

**A second cost that is not compute.** exp-025-cloud §3.4: *"a Kaggle rebuild moves **all** of
them"* — refitting one member (C2-cat) moved the candidate by **RMS 8.0 s, max 666.6 s**. Adding a
fold month re-baselines every pinned member hash the serve path checks, which is a deliberate
re-pin, not a side effect to absorb.

**So the programme's shape.** The 173 head-free columns are available on all nine months **today**
at no further cost, and they are the whole `F`/`O`/`H`/geometry/anomaly interface. What a new
month cannot have without Matt is the 13 head-dependent columns, half the `N` arm, and any priced
gain — because pricing needs a label-side prediction and the prediction needs `V3-allin`.

### §7bi addendum 3 — the refit cost line, measured: one F1-fresh GPU refit

The desk, while recorded rule waits on Matt: refit one member family with a held-out month
group on Kaggle, cap 1 GPU-h, and report wall time, GPU-h, and whether the held-out
predictions land in the member OOF format `prc.blend.stack_members` expects.
`work/refit_cost.py`, `analysis/refit-cost.json`, `analysis/refit-port.json`,
`logs/f1-november.log`, kernel `experiments/exp-047-queue/kernels/f1-november/`.

**The deviation, stated first.** The held-out group is **2025-11 alone**, not
2025-10 + 2025-11. The producer takes a `prc.data` fold **by name**, and
`november_loo` — eleven training months, valid 2025-11, added 2026-09-05 for
retro-v1's round budget — **already exists**, so this refit needed no edit to
shared code and no new code-dataset version. There is no `october_loo`, and no
fold holds both months out, so the two-month group would have needed both an edit
to `prc/data.py` (shared, outside this lane's write scope) and a code-dataset
re-push. The cost difference is one training month in twelve. 2025-11 is the
first pick in any case.

**Nothing in the port was modified.** `kernels/f1-november/` is
`exp-025-cloud/kernels/nested/prc-nested.py` with `CFG` rewritten, the same edit
`push_nested.sh` makes; it is written under this lane so nothing lands in
exp-025-cloud. The producer ran with the arguments the record declares, unchanged:

```
run.py --tag nested-november_loo --folds november_loo --rounds 1200 --clip-z 6000
       --exact-rounds 300 --ghost-mode mixbase --main-model both --seeds 1
       --params {"num_threads": 4} --save-oof
```

**The cost.**

| | measured |
| --- | --- |
| fit wall | **21.6 min** (`1288.3 s` in the producer's own line) |
| **GPU-h** | **0.361** |
| session wall | 22.3 min (13:29:53 → 13:52:09 UTC), inside the 1 GPU-h cap |
| card | Tesla P100-PCIE-16GB, driver 580.159.04 |
| peak GPU memory | **15,473 MiB of 16,384 — 94 % of the card** |
| peak host / child maxrss | 10.17 GB / 11.09 GB of 31.3 GB, 4 vCPU, 4 threads |
| input | **real** (`synthetic: false`), repo commit `9a7004b9` |

**exp-025-cloud §10.2's 0.35 GPU-h a fold is confirmed at 0.361** — within 3 %,
on a different fold and a different month, which is as good as that estimate gets.
The one number to carry forward that the estimate did not have: **the fit uses
94 % of a P100**, so it has no headroom on that card, and a member fit is a
single-month-at-a-time job on it.

**The format: it lands, on all four tests.**

| test | result |
| --- | --- |
| columns vs published `F1-fresh-s1-january.parquet` | **identical** — `MVT_ID_mvt, fold, y_true, y_pred` |
| dtypes vs published | **identical** — `float64, object, float64, float64` |
| population vs this lane's 2025-11 block | **162,332 rows, id set identical**, 0 either way, 0 duplicates, 0 nulls |
| `prc.blend.stack_members` called on it | **returns 162,332 rows, 0 lost to the join**, member column present |

The `fold` column reads `november_loo`, so a consumer cannot confuse it with an
eleven-month fold of the same shape. Quality, for the record and not as a claim:
**RMSE 224.11**, bias +3.98 s, `n_hi=222` above the clip.

**What this does and does not unlock.** The vector is committed at
`oof/F1-fresh--november_loo.parquet`
(sha `2d95786632f6c827…`) rather than left in scratch, because exp-025 §6's whole
problem was members that lived only under the git-ignored `artifacts/` and did not
survive the workstation. It is **one member of seven**. The twelve
`{Nf_,No_}_n_res_*` columns of the 2025-11 block need a **v11-class composition**,
not one member, so this refit does not by itself make 2025-11 fittable — it prices
the unit of the work and proves the output drops straight into the existing
consumer. Six members remain, of which `V3-allin` still cannot be refit anywhere
(addendum 2).

**So recorded rule's cost line, per month added:** ≈**0.36 GPU-h** for F1-fresh, plus ≈1.4
CPU-hours for T-tailv4, L2-hub, N1-nmtail and D1-winter, plus C2-cat (GPU,
unmeasured), plus the stacker — **and `V3-allin`, which is 21 % of `nm_bulk` and
30 % of `nm_late` and has no path at all until the ERA5 snapshot returns.**

## §7bj — the member-producer census, for recorded rule

A census, no bars, no fit that means anything, **no GPU spend**. Every fit below is
CPU and reduced to a smoke. `work/member_smoke.py`, `analysis/member-smoke.json`,
`logs/member-smoke.log`, `logs/member-smoke2.log`.

**There is no dry-run mode and no row cap in any producer.** The equivalent that
exists is the **round budget**: `exp-008-diversity/run.py` takes `--rounds-m6` and
`--rounds-m11`, so a one-round fit walks the whole path — fold resolution, raw
load, frame build, encode, fit, OOF write — in seconds instead of tens of minutes.
`winter_arm.py` and `deliver.py` have no round argument, so they were smoked at one
seed and one fold at the real budget. All on `november_loo`, the fold the 0.361
GPU-h measurement used.

### The table

Weights are `nm_bulk / nm_late` from `blend-config-5.json`. "full job" is
exp-025-cloud **§9.4, the real-data run** (344,841 ranking rows, Kaggle 4 vCPU),
not the synthetic §3.5 table.

| family | weight | runs on a new month? | what it needed | full job (§9.4) | GPU | P100 memory |
| --- | --- | --- | --- | ---: | --- | --- |
| **F1-fresh** | 0.478 / 0.400 | **yes — actually run** | nothing | **21.6 min = 0.361 GPU-h** *(measured, §7bi add. 3)* | **required** | **15,473 of 16,384 MiB — 94 %, measured** |
| **T-tailv2** | 0.226 / 0.0 | **yes** | nothing | 7.7 min, 6.72 GB RSS | no | n/a |
| **L2-hub** | 0.084 / 0.0 | **yes** | **one additive line** in `frames.FAMILY` | 16.0 min, 4.22 GB | no | n/a |
| **D1-winter** | 0.0 / 0.082 | **yes** | nothing | 15.2 min, 9.13 GB (3 seeds) | no | n/a |
| **C2-cat** | 0.0 / 0.030 | **yes on CPU** | the same `frames.FAMILY` line | **never recorded** — blank in §3.5 and absent from §9.4 | yes, normally | **unmeasured** |
| **V3-allin** | 0.212 / 0.296 | **no** | the ERA5 snapshot | — | no | n/a |
| *N1-nmtail* | *0.0 / 0.206* | *not smoked* | *see the flag below* | *9.0 min + 2.7 pre* | *no* | *n/a* |

Smoke wall times, for the record and not as measurements: L2-hub 0.1 min and C2-cat
0.1 min at one round, T-tailv2 1.9 min, D1-winter 2.3 min at one seed — on two
threads of this laptop.

### The one thing that stops two families, and it is not the fold

`L2-hub` and `C2-cat` both failed first with `KeyError: 'november_loo'`. **The fold
itself resolves perfectly** — the run prints `[november_loo] train (1,922,715, 53)
valid (162,332, 53) loaded in 1s`, and 162,332 is exactly this lane's 2025-11 block.
The failure is one line later, at `run_fold`'s
`rounds = cfg["rounds"][frames.FAMILY[fold]]`: **`FAMILY` is a round-budget map**
in `exp-008-diversity/frames.py` and it has no entry for the new fold.

`november_loo` trains on **eleven** months, exactly like `july_loo`, `december` and
`january`, so its entry is `"m11"`. **exp-025-cloud already made precisely this
edit** for its six nested folds, with a comment recording why the budget is part of
the member's identity. So the requirement is **one additive line in shared code**,
of a shape the repository has already accepted:

```python
FAMILY["november_loo"] = "m11"      # eleven training months, as july_loo/december/january
```

The census does not make that edit — `frames.py` is shared and outside this lane —
it **injects the entry at runtime** to prove what happens past it. Past it, both
families run to a written OOF.

### Two side effects a new month has on shared files

1. **`exp-008-diversity/guard_bounds.json` gains a fold entry** and
   **`climb.log` gains a run line** — the producer appends both. My smoke wrote
   them and **I reverted both with `git checkout`**; `git status` outside this lane
   is clean. recorded rule should expect a new month to mutate a committed config file, not
   only to add an OOF.
2. `winter_arm.py` and `deliver.py` have no `--outdir`, so they write beside their
   own experiment. The smoke gave them `_`-prefixed names — the prefix
   `exp-009-noac2/deliver.py` documents as the form its overwrite guard exempts —
   and deleted both afterwards. Nothing published was touched.

### The flag: `members-5final.json` names six, the served blend has seven

`members-5final.json` lists **T-tailv2, C2-cat, L2-hub, D1-winter, F1-fresh,
V3-allin**. `blend-config-5.json`'s `ranking` and `weights` carry a **seventh**,
**`N1-nmtail`, at 0.206 in `nm_late`** — a fifth of that regime, and more weight
than four of the six named. A refit priced from `members-5final.json` alone is
**one member short**.

It was not smoked, and the reason is specific rather than a blocker:
`exp-012-blend2/nmtail_submit.py` is **ranking-only** by construction ("fits the
same head on all twelve 2025 months and predicts the ranking file"), so the fold
producer is `exp-010-nmtail/code/run_folds.py`, which does take `--folds` and
resolves through `prc.data.FOLDS`. Its feature cache is present locally (13 files
under `artifacts/exp-003-anatomy/feat/`) but **its `reocc` cache is absent**, and
building it writes under `artifacts/exp-010-nmtail/`, outside this lane. exp-025
measured those pre-steps at 4.6 min. **Also note N1-nmtail is absent from the
port's nested table entirely** — that table has only F1-fresh, L2-hub8725,
V3-allin and T-tailv4, so three of the seven families have no fold entry in the
port even though their own CLIs take `--folds`, as C2-cat and D1-winter just
demonstrated.

### V3-allin: the snapshot is rebuildable from a token, and it is 23.6 MiB

The desk asked whether the ERA5 snapshot is rebuildable from a token on this
laptop in hours, or whether it is the workstation copy or nothing. **It is the
first, and by a wide margin.** From `docs/weather-provenance.md`:

- **The retrieval script is in the repo**: `scripts/download_copernicus_era5.py full
  --data-dir data/external/weather/copernicus_era5` (18 KB). It writes the exact
  request body of every call, validates each response (one latitude, one longitude,
  exact hourly coverage, no duplicate stamps, the expected grid cell and variables),
  appends a manifest row, and **re-verifies rather than re-downloads** anything
  already present and valid. A second pass for stragglers exists as
  `experiments/exp-005-weather/fetch_missing.py`.
- **The whole snapshot is 23.6 MiB** — 46 manifest rows, 44 production files, 11
  airports × 4 products (a core time-series product plus three parent-product files,
  because the fast product carries neither low cloud cover nor snowfall).
- **The credential is exactly what is missing**: `$HOME/.cdsapirc`, mode 0600, two
  lines — the CDS API url and a personal access token. Nothing else about the
  machine matters; `prc.weather_features` resolves the snapshot from
  `data/external/weather/copernicus_era5` or a sibling checkout.

**So it is not the workstation copy and not nothing — it is a token.** Two
qualifications recorded rule should carry:

- **The binding cost is CDS queue latency, not bytes.** The record shows requests
  sitting queued (an earlier pass left LSZH 2024 queued and needed the
  `fetch_missing.py` second pass for five files). "Hours" is the right order but it
  is a queue, not a transfer.
- **A re-retrieval is a NEW declared data version, not a repair.** The provenance
  doc says so explicitly. July 2026 was fetched as preliminary **ERA5T**
  (`expver 0005`) and can have been revised since; every 2025 fold hour is final
  ERA5 (`expver 0001`), so the fold-side numbers would not move, but the ranking-side
  July join would be against different bytes than the ones that priced v13–v19.
- One human step is not automatable: the CDS account must exist with the dataset
  licence accepted in the web interface. It was, on 2026-09-01, so a token from that
  same account should need no new acceptance.

### The reading

**Five of the seven families are ready or one shared line away.** Four run on a new
month with no change at all (F1-fresh proven on real Kaggle GPU, plus T-tailv2,
D1-winter, and C2-cat/L2-hub past the map entry). The costs that remain are not
discoveries: ≈0.36 GPU-h for F1-fresh and ≈40 CPU-minutes for the CPU families per
month, plus C2-cat, whose runtime **has never been recorded on any machine** and is
the one genuine unknown in the compute line. **V3-allin's blocker is a token, not a
lost dataset**, which is a much cheaper answer than addendum 2 could give — and
`N1-nmtail`'s 0.206 of `nm_late` should be added to recorded rule's count before the total is
quoted.

## §7bk — pre-registration: the six-member 2025-11 base, without V3-allin

**Registered before the stacker is refitted and before any number is seen.** The
desk, on the recorded rule grant: finish the six non-weather members on `november_loo`,
recompose the base **without V3-allin**, predict 2025-11, commit the vector, then
re-emit the 2025-11 block with `own_obs_taxi_minus_p11` and the twelve
`{Nf_,No_}_n_res_*` columns built from that base. Surfacefit screens on the two
shas.

### What is being built

| | |
| --- | --- |
| members | **six**: F1-fresh, T-tailv2, L2-hub, C2-cat, D1-winter, N1-nmtail |
| excluded | **V3-allin** — 0.212 of `nm_bulk`, 0.296 of `nm_late`, blocked on the ERA5 token |
| fit rows | the three folds' published member OOF, restricted to the six |
| predicted rows | 2025-11, from the six `november_loo` vectors in `oof/` |
| composition | `prc.blend`'s hybrid per-regime NNLS anchor, then the stack-5 stacker, `keep-noac`, floor and residual guard — the `exp-026-rebuild/code/serve_stack5.py` path |
| forbidden | any shared-code edit that is not additive; any GPU beyond the two capped member fits; any upload |

### The registered names, before the fit

**This vector is NOT v11 and NOT v12, and no line of the record may call it
either.** It is a **six-member surrogate** built for one purpose: to give 2025-11
the head-dependent block columns it cannot otherwise have. It differs from the
served head in two ways at once — a member is missing, and the remaining six carry
re-fitted weights — so it is a *different estimator*, not a degraded copy of the
served one. It is named `base6` wherever it appears.

### What is registered as the bar

The point of the exercise is the block columns, not a score, so there is no
promotion gate. Two things are registered as **checks that can fail**:

1. **The six-member anchor must be refitted, not borrowed.** `serve_stack5.build`
   reads its rotated anchor back from `oof/stack5-{fold}.parquet`, which was
   produced *with* V3-allin. Reusing it would put a seven-member anchor under a
   six-member stack. The rotation is therefore recomputed on the six, and the run
   **fails loudly** rather than falling back if that recomputation is unavailable.
2. **The 2025-11 vector must cover the block exactly**: 162,332 rows, id set
   identical to `artifacts/exp-047/block/2025-11.parquet`, zero lost through
   `prc.blend.stack_members`, zero nulls — the same four checks every member
   vector in `oof/` has already passed.

### The prediction, stated before the number

Dropping a member that carries 21 % of `nm_bulk` and 30 % of `nm_late` and
refitting the rest **will cost RMSE** on the folds. I register the expectation
that the six-member in-rotation board sits **worse than the seven-member one by
between 1 and 6 s**, and that the loss is **larger in `nm_late` than in
`nm_bulk`**, because that is where V3-allin's weight is concentrated. A six-member
board that comes out *better* than seven would mean the weight fit is not doing
what the record says, and I would report that as a defect rather than a gain.

### What the block re-emission changes

`code/emit.py`'s `predictions()` returns `None` for a non-fold month, which is why
2025-11 emitted 185 columns with twelve of them all-NaN (§7bi addendum). It gains
an **additive** `--pred` argument — this lane's own file, no shared code — that
reads a vector by path and uses it exactly where a fold month's `p11` goes. The
re-emitted block is expected to carry **186 columns, all populated**, and to differ
from the current 2025-11 block **only** in `own_obs_taxi_minus_p11` and the twelve
`_n_res_` columns; every other column must be byte-identical, and that is checked.

### §7bk registration, addition — the base on the three fold months too

Surfacefit, through the desk: produce the six-member base on the **three fold
months** as well, so that `pbase_6 − pbase_true` per fold and per airport is pure
arithmetic and prices the target mismatch in seconds. **Four vectors, four shas.**

**One thing has to be right for that subtraction to mean anything, and it is
registered here before the fit.** `pbase_true` on a fold is a *rotated* vector —
for fold `f` the anchor and the stacker were fitted on the OTHER two folds, never
on `f`. So `base6` on a fold must be rotated the same way, or the difference would
price a rotation change as if it were the missing member:

| target | anchor + stacker fitted on | why |
| --- | --- | --- |
| **2025-11** | **all three** folds' six-member OOF | 2025-11 is outside every fold, so nothing is fitted on it — this is the serve-side form |
| **july_loo** | december + january | the rotation, matching `pbase_true` |
| **december** | july_loo + january | " |
| **january** | july_loo + december | " |

The 2025-11 vector and the three fold vectors are therefore **fitted differently
by construction** — three-fold fit versus two-fold rotation — and the difference
they price is the missing member plus the refit, not the rotation. A single
all-three fit applied to a fold would be trained on that fold's own labels and is
refused.

The three fold vectors also give the registered prediction above its number: the
in-rotation six-member board against the seven-member one, on the same rows.

### §7bk results — base6 built, the block re-emitted, the registered prediction judged

Registered at `13f45332` / `6f17596` **before any of this was seen**. One command,
`work/finish_7bk.sh`, keyed on the two vectors that were missing; it refused until
they existed. `analysis/base6.json`, `analysis/base6-regime.json`,
`analysis/block-2025-11-verify.json`.

#### The six member vectors for 2025-11

| member | where it ran | wall | RMSE | sha256 |
| --- | --- | ---: | ---: | --- |
| F1-fresh | Kaggle P100 | 21.6 min | 224.1065 | `2d95786632f6c827` |
| T-tailv2 | laptop, 2 threads | 2.8 min | 219.6794 | `2bb946a9fad1eb3d` |
| D1-winter | laptop, 3 seeds | 3.2 min | 234.6331 | `65ee7d938bcd8789` |
| N1-nmtail | laptop | 1.8 min | — | `2a3b5d64271c94f3` |
| L2-hub | laptop hedge, 4 threads | 9.1 min | 234.8876 | `d4c456d8f7aa1f8d` |
| C2-cat | **laptop hedge**, 4 threads | **78 min** | 227.2593 | `41cd97b9d9da1597` |

All six: 162,332 rows, id set identical to the block, 0 lost through
`stack_members`, no duplicates, no nulls. **C2-cat came from the local hedge, not
Kaggle** — auth expired at 15:12 UTC and never returned.

**C2-cat is two fits, not one** — 5,325 iterations then a 600-iteration noac head.
The 5-minute gate projected 71.5 min from the first fit alone and kept the run; the
true cost was 78 min because the second fit was not yet visible. The §7bj **8.9 h
was a buffered-output bound and is refuted**, but the gate's own number was an
underestimate for the same structural reason, and a projection taken before every
fit has started should be read as a floor.

#### Step 0 — the instrument, before V3-allin was dropped

| fold | seven-member anchor | `stack5.PUBLISHED` | delta |
| --- | ---: | ---: | ---: |
| july_loo | 286.3274 | 286.3274 | **+0.000012** |
| december | 207.0083 | 207.0083 | **+0.000004** |
| january | 339.0173 | 339.0173 | **+0.000035** |

Against a 1e-3 gate. The wiring is the historical instrument, so what follows
prices the missing member and not a re-implementation.

#### The four base6 vectors

| target | fitted on | rows | RMSE | sha256 |
| --- | --- | ---: | ---: | --- |
| july_loo | december + january | 190,713 | 286.2728 | `fd4a57932dc099d5` |
| december | july_loo + january | 165,677 | 206.9557 | `b465c317fd391bad` |
| january | july_loo + december | 153,706 | 339.1369 | `535bbce18b0d90b7` |
| **2025-11** | **all three** | 162,332 | **216.6593** | **`339ca16ed421a718`** |

The three folds are rotated and 2025-11 is not, exactly as registered, so
`pbase_6 − pbase_true` prices the missing member and the refit rather than a
rotation change.

#### The registered prediction, judged

> *"the six-member in-rotation board sits worse than the seven-member one by
> between 1 and 6 s, and the loss is larger in `nm_late` than in `nm_bulk`."*

**Direction: confirmed on every fold.** The six-member anchor is worse everywhere —
july_loo **+1.0906**, december **+1.2324**, january **+0.5484**.

**Magnitude: my band was too high.** Two of three sit at the very bottom of the
1–6 s band and **january's +0.5484 falls outside it**. Dropping a member with 21 %
of `nm_bulk` and 30 % of `nm_late` costs about **1 s**, not the several I expected:
the remaining six absorb V3-allin's weight better than the record's weight table
suggests they would.

**The regime claim: confirmed, and by a clear margin** (`analysis/base6-regime.json`,
row-weighted over all three folds):

| regime | rows | six − seven |
| --- | ---: | ---: |
| `nm_bulk` | 478,347 | **+1.0755** |
| `nm_late` | 24,740 | **+2.9957** |
| `nonm` | 6,518 | **−7.3196** |
| `noac` | 491 | +18.9163 |

`nm_late` loses **2.8×** what `nm_bulk` loses, which is where V3-allin's weight
sits. **And one thing I did not predict: `nonm` is BETTER without V3-allin**, by
7.3 s row-weighted and by 11.35 s on july_loo. That is not noise at 6,518 rows; it
says V3-allin was actively hurting the no-NM regime under the seven-member weight
fit, and it is recorded here as a finding for whoever prices V3-allin's return.

#### The block, against what §7bk registered

| check | result |
| --- | --- |
| rows unchanged | **OK** — 162,332 |
| 186 columns | **OK** — 185 → 186 |
| nothing all-NaN | **OK** — 12 → 0 |
| exactly `own_obs_taxi_minus_p11` added | **OK** |
| nothing dropped | **OK** |
| only head-dependent columns moved | **OK** — 173 shared columns byte-identical, exactly the twelve `{Nf_,No_}_n_res_*` moved |

**`artifacts/exp-047/block/2025-11.parquet` sha `3e231a8f7cd82d83`**, 48.3 MB. The
185-column emission it replaces is kept at `artifacts/exp-047/block-185/`.

**The caveat that travels with the block.** Its twelve `_n_res_` columns are now as
good as `base6` and no better. base6 is **not v11 and not v12** — a member is
missing and the other six carry refitted weights — so a consumer that treats those
columns as the served head's residuals is wrong by the difference the four vectors
above let it measure.

#### One defect found while closing §7bk

`work/l2hub_hedge.sh` armed its 3 h cap as `( sleep 10800; kill "$PID" ) &` and
never reaped it. L2-hub finished in 9.1 min, so the timer sat orphaned (PPID 1)
for 90 minutes still holding a dead pid, and macOS recycles pids — it would have
sent SIGTERM to whatever held that number at the 3 h mark, most likely **another
desk's job**, with nothing in any log to connect the two. Reaped by hand, and the
script now records the timer's pid and kills it when the fit returns.
`work/c2cat_hedge.sh` does not carry the pattern; it used the coded 5-minute gate
instead. A cap timer must not outlive the process it guards.

## §7bl — a per-class stacker (REGISTERED BEFORE ANY FIT)

Assigned after §7bk: the `nonm` finding (7.32 s better **without** V3-allin)
says one global stacker is wrong for that class. Serve-side only, no candidate.
Registered now; **no fit runs until the desk says go.**

#### How I read the assignment, and what I rejected

**"Cross-fitted per 33.3" = cross-fitted in thirds**, one fold held out at a
time, which is the rotation `stack5.py` already performs (`fit = concat(other
two)`). I grepped the record for a rule numbered 33.3 and there is none; exp-033
has no §3.3 on cross-fitting. Reading taken, work continues.

**"Weights fitted per class" — there are no weights.** `stack5.arm()` trains a
**LightGBM**, 300 rounds, on `member_features` (the seven member columns plus
`mem_mean`, `mem_std`, `mem_range`, `n_defined`), against the residual
`y_true − anchor`. So the mechanism the desk names is real but not a weight: one
GBM fitted where `nm_bulk` is 90 % of the rows learns a residual function that
`nonm`'s 3,595 rows cannot bend.

I therefore fit **`arm()` itself per class, unchanged** — the partition is the
only thing that differs from the published pipeline, so the read prices
per-class-ness and nothing else. I **rejected** fitting `prc.blend.fit_weights`
per class (nnls, already in shared code and literally "one weight vector for one
regime's rows"): it would confound per-class against linear-vs-GBM. It costs
seconds, so it runs beside the GBM as a **control** that answers whether the
per-class effect is linear-recoverable.

**noac is not a fitted class.** exp-033's rule stands: no noac row moves. It
keeps `ref`. Three classes are fitted — `nm_bulk`, `nm_late`, `nonm`.

**The class label is serve-side computable** — checked before registering, not
assumed. `prc.blend.regimes_from_raw` reads `AOBT_3_flt − LOBT_flt`,
`AIRCRAFT_TYPE_mvt` and `FLIGHT_ID_mvt`, all raw movement columns; its docstring
says "the ranking file". No part of the label needs `y_true`. Had it, the block
would be void and I would have said so instead of coding it.

**The round budget is fixed at ROUNDS = 300 for every class, and I will not
tune it.** Per class the fit is much smaller — `nonm` is about 4,300 rows in a
two-fold fit against roughly 380,000 for the global stacker — so 300 rounds
carries a real overfit risk. Registering it anyway: if a class reads worse, that
is the finding. **Retuning the budget after seeing the fold read would be fitting
to the read**, and I will not do it inside this block.

#### What the block can win, priced before it runs

The four class RMSEs reconstruct `stack5.PUBLISHED` to the digit
(286.3274 / 207.0083 / 339.0173) from `analysis/base6-regime.json`, so the
decomposition below is the published instrument and not a re-derivation.

| fold | class | rows | share of fold MSE |
| --- | --- | ---: | ---: |
| july_loo | `nm_bulk` | 172,270 | 48.3 % |
| july_loo | `nm_late` | 14,509 | **20.0 %** |
| july_loo | `nonm` | 3,595 | 6.6 % |
| july_loo | `noac` | 339 | **25.2 %** |

**The fold RMSE is dominated by classes the global fit barely sees.** `nm_bulk`
is 90 % of the rows and under half the error; `noac` is 0.18 % of the rows and a
quarter of it — and `noac` is closed by exp-033. That leaves **`nm_late`, 20 % of
the error on 7.6 % of the rows, as the largest movable class**, not `nonm`.

An oracle that takes, per class, the better of the two weight vectors already
measured in §7bk gains only **+0.3957 / +0.0479 / +0.1895 s** on the three folds,
and that oracle is exactly the `nonm`-only fix — `nonm` is the sole class where
six beat seven. **The whole prize here is sub-second**, and I am saying so before
the fit rather than after.

#### The registered prediction

**Per fold, against `stack5.PUBLISHED`:** july_loo gains **0.2 to 0.8 s**,
january **0.1 to 0.5 s**, december **−0.2 to +0.3 s** — december is the fold I
expect to go the wrong way, because its oracle headroom is 0.048 s and cross-fit
noise on two small classes is the same size. **No fold gains more than 1.0 s.**

**Per class, against the global stacker on the same class's rows:**

| class | registered |
| --- | --- |
| `nm_bulk` | **−0.1 to +0.1 s** — it already dominates the global fit, so per-class ≈ global |
| `nm_late` | **+1 to +6 s** |
| `nonm` | **+2 to +12 s** on july_loo, **0 to +4 s** on december and january |
| `noac` | **exactly 0.0000** by construction, not fitted |

**Falsifier for the instrument:** if `nm_bulk` moves by more than ±0.3 s, then
something other than the partition changed and the read is void. `noac` must be
bit-identical.

**The control:** if per-class nnls recovers most of the GBM's per-class gain, the
effect is linear and belongs in `fit_weights`, which already exists. If it does
not, the gain is a class-specific residual shape and needs the GBM.

#### §7bl addendum — the partition, checked before the fit

`work/perclass.py --dry` fits nothing: it loads the fold frames, asserts every
fold is clear of its own rows on ids rather than trusting the construction, and
prints what each per-class fit would see. `analysis/perclass-dry.json`.

| fold | `nm_bulk` | `nm_late` | `nonm` | `noac` |
| --- | --- | --- | --- | --- |
| july_loo | 306,077 → 172,270 | **10,231 → 14,509** | **2,923 → 3,595** | 152 → 339 held |
| december | 319,961 → 158,386 | 19,092 → 5,648 | 4,964 → 1,554 | 402 → 89 held |
| january | 330,656 → 147,691 | 20,157 → 4,583 | 5,149 → 1,369 | 428 → 63 held |

**This cuts against my own registered prediction, and I am saying so before the
read.** I registered july_loo as the fold that gains most, on oracle headroom.
But july_loo is a summer fold, so December and January together hold far fewer
`nm_late` and `nonm` rows than it does: **july_loo is the one fold whose per-class
stacker is fitted on FEWER rows than it is applied to** — 10,231 to cover 14,509
on `nm_late`, and 2,923 to cover 3,595 on `nonm`, 300 LightGBM rounds on the
thinnest fit in the block. The largest predicted gain and the weakest fit are the
same fold. If july_loo comes in flat or negative, that is the likeliest cause,
and it is written here **before** the number exists so it cannot be produced
afterwards as an excuse.

The prediction registered above stands unchanged.

### §7bl results — the per-class stacker LOSES, and my registered prediction is refuted

84.2 s, 2 threads, `analysis/perclass.json`. Step 0's instrument reproduced
`stack5.PUBLISHED` to +0.000012 / +0.000004 / +0.000035, so this is the published
pipeline with one thing changed. Both registered falsifiers held: `nm_bulk` moved
+0.056 / +0.161 / −0.068 (inside ±0.3) and **`noac` moved by exactly 0.000000 on
all three folds**, so exp-033's rule is kept by construction and by measurement.

| fold | anchor (`PUBLISHED`) | global stacker | **per-class** | nnls control | vs global | vs `PUBLISHED` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| july_loo | 286.3274 | **285.2029** | 285.6290 | 287.7045 | **−0.4261** | +0.6984 |
| december | 207.0083 | **205.6118** | 205.6773 | 206.9653 | **−0.0654** | +1.3310 |
| january | 339.0173 | 338.3816 | **338.3154** | 339.5365 | **+0.0662** | +0.7019 |

**Read against `stack5.PUBLISHED` the per-class stacker gains 0.70 / 1.33 / 0.70 s
— and that reading is wrong.** `PUBLISHED` is the *anchor*, and any stacker beats
the anchor. The control that answers the question is the **global stacker**, the
published step 2 itself, and against it the per-class stacker **loses on two folds
of three** and gains 0.07 s on the third. Per-class-ness is worth nothing here.

#### The registered prediction, judged

| class | registered | measured (global − per-class) | verdict |
| --- | --- | --- | --- |
| `nm_bulk` | −0.1 to +0.1 | +0.056, **+0.161**, −0.068 | two inside, december just outside |
| `nm_late` | **+1 to +6** | **−2.87, −1.51,** +1.43 | **wrong sign on two of three** |
| `nonm` | **+2..+12** july, 0..+4 else | **−3.10, −4.72,** +0.64 | **wrong sign on two of three** |
| `noac` | exactly 0.0000 | +0.0000, +0.0000, +0.0000 | exact |

**I predicted gains on the two classes that mattered and got losses.** The only
part of the prediction that held is the part fixed by construction (`noac`) and
the part where I predicted nothing would happen (`nm_bulk`). The fold-level band
missed as well: I registered "no fold gains more than 1.0 s" against `PUBLISHED`
and december gained 1.33 — but that number is the stacker's, not the partition's,
so the miss is on a reading I should not have registered against in the first
place. **The right control was the global stacker and I registered against the
anchor.** That is a defect in my registration, not in the measurement.

#### Why it lost, and what the §7bk finding actually needs

**`nm_bulk`'s 380,000 rows are not noise for the small classes — they are
information.** A GBM fitted on the whole fold learns a residual-correction shape
that transfers to `nm_late` and `nonm` better than a GBM fitted on those classes'
own 3,000–20,000 rows. Partitioning buys class-specificity and pays for it in
shared structure, and the trade is a loss.

My pre-fit addendum named july_loo as the fold most at risk because its per-class
fits cover more rows than they see (10,231 → 14,509 on `nm_late`; 2,923 → 3,595
on `nonm`). **july_loo is indeed the worst fold, −0.4261.** But the mechanism is
not only thinness: december's `nonm` fit sees 4,964 rows to cover 1,554 — three
times the coverage — and still loses **4.72 s**, the largest class loss in the
block. So starvation aggravates it; the loss is there without it.

**This does not close the §7bk `nonm` finding — it rules out one repair.**
Dropping V3-allin improved `nonm` by 7.32 s; giving `nonm` its own stacker made it
worse. Those are different interventions. The finding says V3-allin's *predictions*
are wrong for `nonm`; a per-class stacker cannot fix that, because every class's
stacker still receives V3-allin as a feature. **The instrument that matches the
finding is a per-class MEMBER SET — drop V3-allin on `nonm` only and keep it
elsewhere — not a per-class fit.** That is one line of change from `perclass.py`
and is the repair I would run next.

#### The nnls control, and one thing it did not measure

Linear per-class weights are **worse than the anchor** on july_loo (287.70 vs
286.33) and january (339.54 vs 339.02) and level on december. So the per-class
effect is not linear-recoverable; there is nothing here for `fit_weights`.

**Stated honestly: the `nonm` control is not a fit.** On all three folds its
weights came back exactly flat at 1/7 each, which is `fit_weights`' documented
fallback when nnls returns an all-zero vector. With no intercept and a
non-negativity constraint, `nonm`'s rows want negative weights — the members
over-predict a class whose RMSE is 500–2,800 s — and nnls clamps them to zero.
The `nonm` column of the control is therefore a flat blend, not a measurement,
and the `nm_bulk`/`nm_late` weights are the only real control numbers. Those are
stable and readable across folds: `F1-fresh` 0.56–0.64 and `V3-allin` 0.30–0.36
carry `nm_bulk`, while `nm_late` spreads across five members and is the only class
giving `N1-nmtail` real weight (0.16–0.22).

Vectors: `perclass--july_loo` `6193e5bf9065424e`, `perclass--december`
`0ea8a96896f6610c`, `perclass--january` `99d0e2b9988b799f`.

## §7bm — a per-class MEMBER SET (REGISTERED BEFORE ANY FIT)

The repair §7bl named, run before anything closes. V3-allin is dropped from the
`nonm` class only; every other class is the global seven-member composition,
untouched. Control is the **global stacker**, the correction of record from §7bl.

#### The design, and the one thing §7bl forces

§7bl proved the *partition* costs: a stacker fitted on a class's own rows loses to
one fitted on the whole fold, because `nm_bulk`'s 380,000 rows are information for
the small classes. **So §7bm does not partition the fit.** Both compositions stay
global and rotated exactly as `stack5.py` builds them; the only difference is
which members are in one of them, and the class decides which composition each row
reads from:

    g7  = the global SEVEN-member composition (the published step 2)
    g6  = the global SIX-member composition  (§7bk's base6, V3-allin dropped)
    mix = where(regime == 'nonm', g6, g7)

V3-allin is dropped from **both the anchor and the stacker** on that arm, because
the §7bk prior was measured at the anchor: `nonm` improved by 11.35 s on july_loo
and 7.32 s row-weighted when the member left the *anchor*. Dropping it from the
stacker alone would not be the intervention the prior describes.

`noac` keeps `ref` in both compositions, so exp-033 holds without a special case.

#### The registered prediction

**The prior** is §7bk's `nonm` anchor gain — 11.35 s on july_loo, 2.17 on
december, 2.58 on january, 7.32 s row-weighted over 6,518 rows. **The cap** is the
oracle over the two vectors, +0.3957 / +0.0479 / +0.1895 on the folds — and that
oracle is an **anchor-level** bound, so the stacker-level result may sit below it
without contradicting it.

**Per fold, against the global stacker:** july_loo **+0.10 to +0.40**, december
**0.00 to +0.10**, january **+0.02 to +0.20**. No fold exceeds its oracle bound.

**Per class:** `nonm` improves by **+2 to +12 s** on july_loo and **0 to +5 s** on
december and january. `nm_bulk`, `nm_late` and `noac` move by **exactly 0.0000** —
this is a hard falsifier, not a band: those rows are copied from `g7` unchanged,
so any movement at all means the splice is wrong and the read is void.

**The named failure mode, written before the read.** §7bl refuted me by predicting
gains where losses came, so the risk is stated plainly: the §7bk prior is an
*anchor*-level fact, and **the seven-member stacker may already repair most of
V3-allin's `nonm` error**. If it does, `g6` and `g7` differ little on `nonm` at the
stacker level and the fold gain lands near zero — or negative, if the stacker's
repair is better than removing the member. That would not overturn the §7bk
finding; it would say the finding is already absorbed downstream and has no
serve-side value left.

**If it fails,** the BRANCH REPORT and `closed-provisional` follow in the same
block, and no third variant runs without the desk's word.

### §7bm results — the member set WORKS on all three folds, and beats the cap I registered

89.5 s, 2 threads, `analysis/memberset.json`. Step 0 reproduced `stack5.PUBLISHED`
to +0.000012 / +0.000004 / +0.000035. The hard falsifier held **exactly**:
`nm_bulk`, `nm_late` and `noac` moved by `0.000000` on every fold, so the splice
touches only the class it is meant to and exp-033 is kept by measurement.

| fold | `g7` global (control) | `g6` global | **`mix`** | gain | oracle cap | over cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| july_loo | 285.2029 | 286.2728 | **284.7629** | **+0.4401** | +0.3957 | **+0.0444** |
| december | 205.6118 | 206.9557 | **205.5328** | **+0.0791** | +0.0479 | **+0.0312** |
| january | 338.3816 | 339.1369 | **338.1624** | **+0.2191** | +0.1895 | **+0.0296** |

**All three folds gain against the global stacker** — the control of record from
§7bl's correction — and the six-member composition is worse than the seven on the
fold as a whole (286.27 vs 285.20) while being much better on the class it is
spliced in for. That is the whole point of the intervention, and it reproduces
§7bk's base6 fold numbers to the digit (286.2728 / 206.9557 / 339.1369) as a free
consistency check.

`nonm`, the only class whose member set changed:

| fold | rows | `g7` | `g6` | gain |
| --- | ---: | ---: | ---: | ---: |
| july_loo | 3,595 | 531.7770 | **519.1153** | **+12.6617** |
| december | 1,554 | 483.8304 | **480.2355** | **+3.5949** |
| january | 1,369 | 2801.2789 | **2798.3061** | **+2.9728** |

#### The registered prediction, judged

| registered | measured | verdict |
| --- | --- | --- |
| july_loo +0.10 to +0.40 | **+0.4401** | just **above** the band |
| december 0.00 to +0.10 | +0.0791 | inside |
| january +0.02 to +0.20 | **+0.2191** | just **above** the band |
| `nonm` +2 to +12 (july_loo) | **+12.6617** | just **above** |
| `nonm` 0 to +5 (dec, jan) | +3.5949, +2.9728 | both inside |
| `nm_bulk`/`nm_late`/`noac` exactly 0.0000 | 0.000000 everywhere | **exact** |
| **"no fold exceeds its oracle bound"** | **all three exceed it** | **refuted** |

**The direction is right and my magnitudes were low again** — the mirror of §7bl,
where I was high. The claim that actually failed is the cap: I registered the
oracle as a bound and **it is not one**.

**Why the cap is not a cap.** The oracle came from `base6-regime.json`, which
compares the six- and seven-member **anchors**. It assumes the best you can do per
class is pick the better anchor. But the stacker improves on the anchor, and on
`nonm` it improves further from the six-member side than from the seven-member
side: july_loo's `nonm` goes 534.77 (seven anchor) → 523.42 (six anchor) →
**519.12** (six *stacker*). The splice inherits that extra 4.3 s, so it clears a
bound derived one level upstream. **An anchor-level oracle does not bound a
stacker-level result, and I should not have registered it as a cap.** I flagged
the level mismatch in §7bm's registration but named only the risk of landing
*below* it.

#### What this settles

§7bk found `nonm` 7.32 s better without V3-allin. §7bl showed a per-class *fit*
cannot capture it and lost. §7bm shows a per-class *member set* captures it and
gains on every fold, because the two changes are not the same thing: partitioning
the fit throws away `nm_bulk`'s rows, while changing the member set removes a
member whose `nonm` predictions are wrong and keeps every row of the fit.

**The mechanism is now named end to end:** V3-allin, the weather member, is
actively harmful on the no-NM class and helpful everywhere else, and one global
member set cannot express that. The serve-side value is **+0.44 / +0.08 / +0.22 s**
— real, reproducible, and small.

Vectors: `memberset--july_loo` `0b720394de6f09a4`, `memberset--december`
`089309e6ad7da054`, `memberset--january` `f3bbf9dd4b9e4169`.

**No third variant runs without the desk's word.** The obvious next question —
whether other classes want their own member sets, `nm_late` being the largest
movable one — is not opened here.

## §7bn — the full menu: every single-member drop × class (REGISTERED BEFORE ANY FIT)

The desk's gate ruling, accepted: §7bm's form — drop V3-allin on `nonm` — was
**chosen from §7bk's fold reading**, so it is fold-selected and every-fold-clear
is not its gate. Its clean estimate is the cross-fitted one.

#### The gate text, read rather than paraphrased, and one conflict in the record

`exp-046-surfacefit/RESULTS.md:2540`:

> "Every fold clear" stays the gate for an in-fold arm whose *form* was not
> selected on the folds.

**`:3776` says the opposite** — "the every-fold-clear rule governs **SELECTED**
variants … an **UNSELECTED** update … is gated as the currency rows gate
increments: pooled clear, no fold clear below zero, no fold declining." That text
is inside the 05:40 ruling **the desk reversed at `:3962`** on 2026-09-09, which
restored §13.1: "every fold clear for an in-fold arm whose form was not selected
(§13.1 text, §15, recorded rule, recorded rule, precedent v10 at 0.26×)." **§13.1 governs**; `:3776`
is superseded and is recorded here only so the next reader who greps the phrase
does not take the reversed text as live. Nothing in this block turns on it: §7bm
is selected either way, so it needs the cross-fitted read either way.

#### The menu

Seven members × three classes (`nm_bulk`, `nm_late`, `nonm`; `noac` held by
exp-033) = **21 cells per fold, 63 in all**. Same `arm`, same `SEEDS`, same
`ROUNDS`, same guard. Built as §7bm builds it and for the same reason — §7bl
proved the fit must not be partitioned:

    g7        = the global seven-member composition (the control)
    g_minus_m = the global composition with member m dropped, for each of the 7
    cell(c,m) = where(regime == c, g_minus_m, g7)

Eight compositions per fold, 24 in all; every cell is a splice out of the same
eight, so no cell costs a fit of its own.

**Registered in advance, so it is not chosen after the read:** if two members are
clear on the same class, the tie-break is the **larger mean gain over the two
fitting folds** — never the priced fold.

#### The registered prediction

**Only `nonm` × V3-allin survives.** Concretely, and falsifiably:

* **`nonm` × V3-allin** is clear of zero on all three folds.
* **No other cell** is clear positive on more than one fold.
* **Several `nm_bulk` cells are clear NEGATIVE** — that class carries the nnls
  weight (F1-fresh 0.56–0.64, V3-allin 0.30–0.36 from §7bl), so dropping a
  load-bearing member there should cost, not gain.
* **`nm_late` × N1-nmtail is clear negative** on at least one fold: `nm_late` is
  the only class giving N1-nmtail real weight (0.16–0.22).

**The selection premium is between 0.00 and 0.05 s on every fold** — near zero,
because I expect the same single cell to win on every fold, and a selection that
does not move between folds cannot carry much premium. **If the winning cell
differs by fold, this prediction is wrong and the premium will be large**; that
is the outcome that would put §7bm's +0.44 / +0.08 / +0.22 in question.

I have now been high (§7bl) and low (§7bm) on magnitude, so the claim I am most
willing to be judged on is the **ordering**: `nonm` × V3-allin first on every
fold, and `nm_bulk` drops last.

### §7bn results — the menu, the intervals, and the cross-fitted read: FAIL

344.4 s for the 63 cells, 2 threads. `analysis/menu.json`, `analysis/crossfit.json`,
`analysis/nonm-share.json`. Step 0 reproduced `stack5.PUBLISHED`; `g7` reproduced
§7bm's control to the digit (285.2029 / 205.6118 / 338.3816).

#### (i) The menu with paired-bootstrap intervals — 2 of 63 cells clear positive

Both are the same cell:

| fold | class | member | gain | 95 % |
| --- | --- | --- | ---: | --- |
| july_loo | `nonm` | V3-allin | **+0.4401** | [+0.2398, +0.6701] |
| january | `nonm` | V3-allin | **+0.2191** | [+0.0476, +0.3998] |

**No other cell is clear positive on any fold** — not one of the other 61. And 21
cells are clear **negative**, which is the same story read the other way: dropping
L2-hub, F1-fresh or V3-allin from `nm_bulk` is clear negative on **all three
folds**, and `nm_late` × N1-nmtail is clear negative on two. Every member except
T-tailv4 and C2-cat is load-bearing somewhere, and V3-allin is load-bearing on
`nm_bulk` (−0.79 / −1.10 / −0.74) while being the only member that helps by
leaving `nonm`. The seven-member set is not padded.

#### (ii) The cross-fitted read — the gain does not survive

The registered rule: on fold `f`, take the drops clear on **both** other folds and
price them on `f`. The class arithmetic is exact — it reproduces the control and
all 63 measured cells to a worst error of **0.000098 s**, so no refit was needed.

| fold | selected from the other two | at face | **cross-fitted** | selection premium |
| --- | --- | ---: | ---: | ---: |
| july_loo | *nothing clears* | +0.4401 | **+0.0000** | **+0.4401** |
| december | `nonm` × V3-allin | +0.0791 | **+0.0791** | +0.0000 |
| january | *nothing clears* | +0.2191 | **+0.0000** | **+0.2191** |

**On the board this is exactly zero.** `frames.SHARES` weights july_loo and
january; both select nothing, so the board estimate is unchanged at 309.8819
against an at-face +0.3316.

**Why one fold vetoes the other two.** december's cell is **+0.0791
[−0.1964, +0.3375]** on 1,554 `nonm` rows — positive, not clear. december is one
of the two "other folds" for july_loo *and* for january, so its single wide
interval removes the cell from both board-relevant selections. With three folds
and an and-rule, the noisiest fold decides everything.

#### The registered prediction, judged

| registered | measured | verdict |
| --- | --- | --- |
| `nonm` × V3-allin clear on **all three** folds | clear on **two** — december is not | **miss** |
| no other cell clear positive on more than one fold | no other cell clear positive on **any** fold | right, and stronger |
| several `nm_bulk` cells clear negative | L2-hub, F1-fresh, V3-allin on all three | right |
| `nm_late` × N1-nmtail clear negative on ≥1 fold | july_loo and january | right |
| **selection premium 0.00 to 0.05 s** | **+0.4401 / +0.0000 / +0.2191** | **refuted** |
| ordering: `nonm`×V3-allin first, `nm_bulk` last | exactly that | right |

**The ordering claim I offered for judgement is correct; the premium claim is
wrong by an order of magnitude.** My reasoning was that a premium needs the
winning cell to move between folds. **The winning cell did not move — and the
premium is still the whole effect**, because the and-rule does not ask which cell
wins, it asks whether every other fold can certify it. A stable winner with one
uncertifiable fold pays full premium. That mechanism was not in my registration.

#### (iii) The board share — the class transfers, the months invert

Ranking set, 344,841 departures: `nm_bulk` 93.802 %, `nm_late` 4.666 %, **`nonm`
1.419 %**, `noac` 0.113 %. Board-weighted fold `nonm` is 1.445 %, a ratio of
**0.98×** — the class is the same size on the board as on the folds.

But the months invert: `nonm` is summer-heavy in the folds (july_loo 1.885 % vs
january 0.891 %) and winter-heavy on the board (2026-01 1.533 % vs 2026-07
1.329 %). §7bm's largest gain was measured on july_loo, the fold richest in `nonm`,
and 2026-07 is the **poorer** ranking month. Any projection carrying july_loo's
per-fold gain across at face value overstates it. `nm_late` is the one class
materially thinner on the board (4.666 % vs 5.559 %, 0.84×).

#### BRANCH REPORT — per-class member set, `closed-provisional`

**Status: `closed-provisional`.** No candidate composed; the desk's condition
("compose NO candidate until (ii) is read; then, if it clears") is not met. The
board-relevant cross-fitted gain is +0.0000.

**The negative.** The per-class member set gains **+0.4401 / +0.0791 / +0.2191**
in fold and **+0.0000** on the board once the form is priced the way a
fold-selected form must be. The entire at-face board gain of +0.3316 is selection
premium on two of three folds.

**Destination value and path feasibility, stated separately.** The *destination*
is intact and is worth naming precisely: V3-allin genuinely hurts `nonm` and
genuinely helps `nm_bulk`, the effect is positive on all three folds, clear on
two, clear-negative on none, and the class it lives in is the same size on the
board as on the folds (0.98×). The *path* — certify one (class, member) drop by
requiring every other fold to clear it — is what fails, and it fails on
arithmetic that has nothing to do with whether the effect is real.

**The honest attempt at repair, and it was made before the report.** §7bl tried a
per-class *fit* and lost (−0.43 / −0.07 / +0.07); §7bm changed the *member set*
instead and gained on every fold; §7bn priced §7bm the way the gate requires and
the gain did not survive. Three forms, each one built from the previous negative's
diagnosis.

**The cause, measured not asserted.** december's `nonm` arm holds **1,554 rows**
and its interval half-width is **0.2670 s** against an effect of 0.0791 s. For
that fold to certify the same effect it needs the half-width below the effect —
**11.4× the rows, about 17,700** — and december has 1,554. The lane is not blocked
by the effect's size; it is blocked by the *sample size of one class on one fold*,
and that fold sits in both board-relevant selections.

**What would revive it, in order of cost.**

1. **A pooled interval instead of a per-fold and-rule.** The three folds hold
   6,518 `nonm` rows together, 4.2× december alone, and the pooled effect is
   larger than december's. The record's own gate (a) is "pooled clear", and this
   is the form that gate was written for. It is **not** what the desk registered
   for this block, so I have not run it.
2. **A fourth fold.** `november_loo` exists in this lane and its base is already
   built (§7bk). A fourth interval changes what the and-rule can certify and adds
   `nonm` rows the selection can see. **But it is not free, and the desk is right
   to price it as the token** (2026-09-10): §7bk built `november_loo`'s base with
   **six** members, and the comparison this lane needs there is seven against six.
   V3-allin on 2025-11 needs the ERA5 snapshot, which needs the two-line
   `~/.cdsapirc` (§7bj). **Revival (2) is therefore gated on Matt's CDS token**, on
   the same dependency as recorded rule, not on any work I can do without it.
3. **More `nonm` rows per fold**, which is the same lever as (2) stated as data
   rather than as folds.

**No third variant runs without the desk's word**, and none of the three above has
been started.

## §7bp — PRE-REGISTRATION: the expected board move for the v20 upload with v9

Written and committed **before the upload and before its board reading** (score-feedback rule).
Nothing is fitted here. `work/v9_prereg.py`, `analysis/v9-preregistration.json`.

**The candidate.** **v9 = base set v23, the extended head on both sides** — the
with-clock head's fitting panel gains 2025-11's writable rows, whose targets come
from **this lane's six-member base6** (§7bk). **The face: +0.513 s
[+0.392, +0.688]** in fold over v8's head, every fold clear, gate passed under
rule (a) (the desk, 2026-09-10).

**The anchor is pending, and the move is what is registered.** BOARD READING 18
has not landed — `docs/board/best_per_team.csv` still shows **v18 at 246.3605 with
17 submissions**. So the move is registered anchor-free and the absolute bands are
given at three anchors. **The rule: substitute v19's actual score when reading 18
lands, and do not re-derive the move.**

**The history**, six scored uploads: v13 1.75×, v14 1.03×, v15 2.01×, v16 0.36×,
v17 0.70×, v18 1.44×. Median **1.23×**, mean **1.22×**, range 0.36×–2.01×. v19's
own transfer is **pending**.

| basis | expected move |
| --- | --- |
| at face, 1.00× | **+0.5130** |
| at the median transfer, 1.23× | +0.6336 |
| at the mean transfer, 1.22× | +0.6233 |
| the observed transfer range on the point | +0.1847 … +1.0311 |
| the observed range on the face interval (widest honest band) | +0.1411 … +1.3829 |
| the two most recent uploads (1.44×, 0.70×) | +0.3591 … +0.7387 |

**Transfer failure, in one sentence.** *A move under **+0.1411 s** — the lowest
transfer ever observed (0.36×, v16) applied to the bottom of this candidate's own
fold interval (+0.392 s) — and a move at or below zero is not a weak transfer but
no transfer at all.*

**Three things that are new here and are NOT in the transfer history.** Registered
because they weaken the history as a guide, not to excuse a result afterwards:

1. **This is the first upload whose fitting panel contains a month priced by a
   different estimator.** 2025-11's targets are base6 — not v11, not v12: a member
   is missing and the other six carry refitted stacker weights (§7bk). The measured
   control says the mismatch costs nothing (−0.01), but **no prior reading in the
   history was taken under a mixed-base panel**, so the six transfers are a weaker
   guide here than they were for v18 or v19.
2. **The face is the largest this programme has registered apart from v13's
   +8.128** — 2.1× v19's +0.2428 and 2.5× v18's +0.2053. Transfers have been
   measured on faces of 0.07 to 0.73 s; +0.513 is at the top of that range, and the
   one larger face (v13) transferred at 1.75× from 16× the size.
3. **2026-07 is the poorer ranking month** for the class this lane last measured
   (§7bn (iii): `nonm` 1.329 % against 2026-01's 1.533 %) while the folds are
   summer-heavy — a composition caveat that applies to any fold-side face.

**Rank is not at risk** on any band: the next team is at 260.931, so every outcome
leaves rank 1 by more than 14 s.

**Numbering, corrected.** §7au is titled "the expected board move for **v20**" and
registered **v7**; the v19 slot then went to **v8** (§7ba registered both live
branches). That label is stale and this section is the pre-registration for the
**v20 upload carrying v9**. §7au's own numbers are untouched.

**No serve, no upload — the desk uploads.** This lane registers and audits.

### §7bp addendum — the face was predicted before it was measured, and the audit checks a different quantity

Checked while preparing the audit, from `analysis/audit-expect-v23.json` (the
expectation for candidate `exp-046-surface-v9.parquet`, base set **v23** = v22 with
2025-11's **99,709** writable rows added to every fit on both sides).

**The two numbers in circulation are a prediction and its outcome, not a
disagreement.** exp-046 §47.20, commit `467c89f`, **registered +0.49 [+0.32, +0.62]
before the ten fits ran**, expecting every fold clear and naming january as the
fold most likely to fail. The bag then **read +0.513 [+0.392, +0.688]**. The
measured point sits inside the registered band, and the naive figure §47.20
deliberately did **not** register was +0.59236.

That matters for §7bp's expectation and is recorded there: **this face was
predicted before it was measured**, so it is not a number selected after the fact,
and the transfer bands above rest on a face with that provenance. §7bp's registered
face and its +0.1411 s failure line stand unchanged — they were written on the
measured +0.513, which is the right quantity for a board move.

**The audit checks a third quantity, and I will not conflate them.** Both figures
above are increments over **v8's head**. The audit verifies the increment over
**v18's head**, which the expectation file states plainly is a different quantity
for which *no* figure was registered, and whose placeholders are therefore `null`
rather than a prediction. So the audit can AGREE on everything and still say
nothing about +0.513.

**Not run yet, by the desk's order:** the file is still a SKELETON with
`bytes.candidate_sha256`, the three per-fold increments and their seed envelopes,
and the changed/reverted row counts all PENDING. The desk's instruction is to wait
until surfacefit reports v9 with the file final.

## §7bq — the v9 audit: 57 of 57 AGREE

`analysis/audit-expect-v23.json` went **FINAL** at 18:44 UTC with every placeholder
filled; the audit ran against it on the line the file itself carries.
`analysis/audit-v23.json`. Run at one thread so §7bo's C2-cat kept its four.

**57 checks, 57 AGREE, 0 DISAGREE, 0 PENDING, 0 unanswered.**

Candidate `exp-046-surface-v9.parquet`, sha `8fd4b34166b61756…`, 194,681 rows
changed against v18.

**The increment this audit checks is over v18's head**, which is neither of the
faces in circulation:

| fold | gain (s²/row) | 95 % | rows moved |
| --- | ---: | --- | ---: |
| july_loo | 403.9271 | [+322.7833, +491.151] | 88,425 |
| december | 1176.6354 | [+1083.0622, +1277.8003] | 102,186 |
| january | 512.1916 | [+408.8648, +625.148] | 69,840 |
| **board** | **+0.756226** | **[+0.576833, +0.99899]** | — |

clean **+0.923898**. **Every fold's lower bound is clear of zero.**

**And it reconciles with the two registered faces to 0.0004 s.** v19's own face
over v18's head is **+0.2428**; v9's measured face over v8's head is **+0.513**;
their sum is **+0.7558** against this audit's independently computed **+0.7562**.
The three numbers were produced by different lanes from different artifacts — the
agreement is a check none of them was fitted to pass, and it is the strongest
single piece of evidence in this audit that the chain of faces is sound.

**What the audit does not say.** It does not confirm +0.513. That figure is an
increment over v8's head and no figure was registered for the over-v18 quantity
this tool checks; the expectation file set those placeholders to `null` for exactly
that reason. §7bp's board-move registration keys on +0.513 and stands as written.

### §7bo defect — an out-of-lane write into a PUBLISHED exp-008 result

Found by the desk, 2026-09-10, before it reached any commit.

**What happened.** `experiments/exp-008-diversity/result-C2-d8p.json` is tracked,
and my §7bk C2-cat hedge rewrote it: a `november_loo` entry appended to `folds`,
a second entry in `argv_history`, and — the damaging part — **the top-level `argv`
overwritten with mine**, so exp-008's published record described their own three
scored folds with *my* command line (`--folds november_loo … --cpu`).

**The cause, which is a real gap in the harness and not carelessness alone.**
`exp-008-diversity/run.py` takes `--outdir`, and its own comment says the flag
exists so "another experiment [can] reuse this harness without writing into" its
lane — but `--outdir` moves **only the OOF parquet** (line 303). The result JSON
(line 445), `climb.log` (457), the importance CSVs (309) and `submit`/`select`
(432/438) all use `HERE` at call time. Using the documented escape hatch is not
enough to stay out of that lane.

**The fix, a repoint rather than a revert.** The runners no longer `runpy` the
producer; they load it, set `run.HERE` to `work/e8-results/` in this lane, and call
`main()`. `GUARD_CACHE` binds at *import*, so the guard bounds are still **read**
from exp-008's published cache — reads unchanged, writes in my lane — and the
existing `guard_bounds.json` revert stays as belt and braces. Same shape as the
`EXP` repoint that kept F1-fresh out of exp-018.

**What was restored.** `result-C2-d8p.json` reverted to three folds, one
`argv_history` entry, exp-008's own `argv`; the polluted copy is kept in my lane as
`work/e8-results/result-C2-d8p.polluted-by-7bk.json`. Three further files I had
left in their lane — `result-L2-hub8725.json`, `result-_smoke-C2-cat.json`,
`result-_smoke-L2-hub.json`, all untracked and all containing only `november_loo`
or `october_loo` folds, so all mine — were moved into `work/e8-results/`.
**`git status` on `experiments/exp-008-diversity/` is now empty.**

**A second defect, caught in the fix itself.** I first added
`result-T-tailv4.json` to the revert list. It is not tracked and not even an
exp-008 arm (T-tailv2 goes through exp-009's `deliver.py`), and
`git checkout -- <unknown pathspec>` **fails wholesale**, which would have silently
stopped the other three files being reverted. Removed, and every entry in `TOUCHED`
is now asserted tracked before it is used.

**The October C2-cat was launched before the repoint** and still writes the old
path when it finishes, so `work/oct_finish.sh` restores exp-008's lane the moment
that fit settles, before it composes anything.

### §7bo results — 2025-10 complete, after an emission defect that failed loudly and then quietly

The six non-weather members refit out of fold on `october_loo`, base6 on 2025-10
fitted on all three folds, the block re-emitted. `analysis/member-folds-2025-10.json`,
`analysis/nmtail-fold-2025-10.json`, `analysis/base6-2025-10.json`,
`analysis/block-2025-10-verify.json`.

| member | where | rows | RMSE | sha256 |
| --- | --- | ---: | ---: | --- |
| F1-fresh | **laptop CPU** | 185,674 | **216.8708** | `a221fc74ec915b33` |
| T-tailv2 | laptop, 2 threads | 185,674 | 231.8529 | `de40e34e1f8582ca` |
| L2-hub | laptop hedge | 185,674 | 234.8332 | `765609a125c7eb35` |
| C2-cat | laptop hedge | 185,674 | 228.9353 | `0a883d8eafa464aa` |
| D1-winter | laptop, 3 seeds | 185,674 | 239.2923 | `5b7f50e2c570c314` |
| N1-nmtail | laptop | 185,674 | 318.5019 | `f11328e44d0737be` |

All six: ids identical to the 2025-10 block, 0 lost through `stack_members`, no
nulls, no duplicates. **F1-fresh ran locally on CPU** — Kaggle never came back —
and its CatBoost half is a CPU fit where November's ran on a P100, so the two
months' F1-fresh are not bit-identical estimators.

**base6 on 2025-10: RMSE 215.9453, 185,674 rows, sha `2bfc9a6959281d6e`.** The
three rotated fold vectors reproduced §7bk's to the digit (286.2728 / 206.9557 /
339.1369, same shas) — a free reproducibility check across two runs a day apart.

**Block: `artifacts/exp-047/block/2025-10.parquet` sha `29932ab196080943`**,
53.9 MB, 186 columns. All six registered checks OK: 185 → 186, added exactly
`own_obs_taxi_minus_p11`, nothing dropped, all-NaN 12 → 0, **173 shared columns
byte-identical** and only the twelve `{Nf_,No_}_n_res_*` moved.

#### The defect, and the worse thing inside it

The first emission produced **63 columns, not 186** — 122 dropped. Cause: §7bk
emitted with `--sources F,O,H,N,geo --rule <sha>` and I wrote `oct_finish.sh`'s
emit line from `emit.py`'s signature instead of from November's actual command, so
`--sources` fell back to its **default `"F"`** and one family was built. The rule
sha was missing too.

**The worse thing is what the verifier said.** Three checks failed loudly
(`is_186`, `added_is_exactly…`, `nothing_dropped`) — but
**`nothing_is_all_nan_now` PASSED, vacuously**: the twelve `_n_res_` columns were
not filled, they were *gone*, and a check for "no column is all-NaN" cannot tell
an absent column from a repaired one. So did
`the_only_columns_that_moved_are_the_head_dependent_ones`, on 63 shared columns.
**Two of my six registered checks are satisfiable by deleting the thing they
check.** They are only safe because `nothing_dropped` sits beside them; read alone,
either would have called a 63-column block healthy.

**And I did not report the chain.** It finished at 20:00 UTC with `verify rc=1` in
its own log and the desk found it, not me: I had armed watches on the first chain
and on the expectation file and none on the finisher. A chain that finishes must be
reported; a non-zero rc in a log nobody reads is not a measurement.

#### Two out-of-lane writes, both fixed at the source

1. **exp-008's published `result-C2-d8p.json`** — recorded above; fixed by
   repointing `run.HERE` into this lane.
2. **The tracked repo-root `catboost_info/`** (desk, 2026-09-10): CatBoost's
   default `train_dir` is `./catboost_info`, which is the repo root when that is
   the cwd. `train_dir` is now pinned to `work/catboost_info/` at **construction**
   in both CatBoost runners, so it holds for every fit including the ones inside
   exp-018's model, and `oct_finish.sh`'s restore step now covers it as well.

### §7bp outcome — BOARD READING 19: the registered band held, and the face was nearly exact

v20 (surfacefit's v9) scored **245.2901** at 00:14 UTC 2026-09-10 — a move of
**−0.4851 s** on v19's 245.7752, a transfer of **0.946×** on the re-anchored face
+0.513. Rank 1 of 94, lead 15.64 s.

**Every registered band held, and the failure line was not approached.**

| registered basis | band or point | outcome |
| --- | --- | --- |
| at face, 1.00× | 245.2622 | **missed by +0.0279 s** |
| at the median, 1.44× | 245.0365 | missed by +0.2536 |
| at the mean, 1.3857× | 245.0643 | missed by +0.2258 |
| observed range on the point | [244.5389, 245.5905] | **contains it** |
| observed range on the face interval | [244.1171, 245.6341] | **contains it** |
| two most recent (2.41×, 1.44×) | [245.0365, 245.4161] | **contains it** |
| **failure line** | above 245.6341 | **not breached**, by 0.344 s |

**The at-face prediction missed by 0.0279 s** — the closest this lane has come. It
is worth being precise about why that is weaker evidence than it looks: the
at-face figure is just the fold face with no transfer applied, so its accuracy says
the transfer happened to be ≈1, not that the *method* predicted it. The three
transfer-weighted rows all pointed too optimistic, because the history's centre
(median 1.44 at the time) was pulled up by v19's 2.41.

**The transfer table, re-read with v20 = 0.95:**

| upload | v13 | v14 | v15 | v16 | v17 | v18 | v19 | v20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transfer | 1.75 | 1.03 | 2.01 | 0.36 | 0.70 | 1.44 | 2.41 | **0.95** |

n = 8, **median 1.2350** (was 1.44), mean **1.33125**, range 0.36–2.41 unchanged.
v20 is the **second-closest reading to face** in the programme's history, behind
v14's 1.03 and ahead of v17's 0.70.

**The combined read the desk states**: v18 → v20 is **−1.0704 s** on a combined
face of +0.756, **1.42×** — and that is the figure the §7bq audit computed
independently (+0.756226), so the two uploads together transfer above face while
the second one alone transferred at 0.95×.

**No band is registered for a next candidate**, per the desk: one is registered
when a candidate exists.


## §7br — the third month's common work (2025-03), and first the vacuous checks

**Desk instruction (2026-09-10):** *"the THIRD month's common work, 2025-03 …
same discipline as 7bo: six members out of fold on the laptop (F1-fresh under the
four-model gate), base6 on 2025-03, block re-emitted with the November command
line, shas in one line, chain orphaned, threads 4, and a WATCH on the finisher
this time … Before launching: repair the two vacuous block checks —
nothing_is_all_nan_now and only_head_columns_moved must assert the twelve _n_res_
and own_obs_taxi_minus_p11 are PRESENT before they judge them; commit that first."*

### i. The repair (done first, as ordered)

`work/block_verify.py` judged six checks. Two of them could not see the failure
they exist to catch:

* `nothing_is_all_nan_now` asked whether any numeric column of the **new** block
  is all-NaN. A column that is **absent** is not all-NaN, so a block that lost the
  twelve `{Nf_,No_}_n_res_*` columns passed.
* `the_only_columns_that_moved_are_the_head_dependent_ones` compared the moved set
  against `[c for c in EXPECTED_NEW if c in shared]` — a set *derived from the new
  block's own columns*. When the head columns are gone they are not shared, the
  expectation collapses to `[]`, and `differ == []` matched it.

Both now assert presence first:

```python
missing_head = [c for c in EXPECTED_NEW if c not in new.columns]
head_all_nan = [c for c in EXPECTED_NEW
                if c in new.columns and new[c].notna().sum() == 0]
must_move = sorted(c for c in EXPECTED_NEW if c in old.columns)
...
"nothing_is_all_nan_now":
    missing_head == [] and head_all_nan == [] and dead == [],
"the_only_columns_that_moved_are_the_head_dependent_ones":
    missing_head == [] and sorted(differ) == must_move,
```

The second change matters as much as the presence test: `must_move` is now keyed
on the **old** block's columns (the twelve `_n_res_` the 185-column emission
already carried, all-NaN), so it cannot be shrunk by the artefact under test. The
record gained four diagnostic fields — `head_columns_expected`,
`head_columns_missing`, `head_columns_all_nan`, `head_columns_that_must_move`.

**Regression, both directions.**

| block | verdict | evidence |
| --- | --- | --- |
| 2025-11 (good) | **PASS**, 6/6 | 185 → 186, moved = the twelve, head 13/13 present |
| 2025-10 (re-emitted, good) | **PASS**, 6/6 | same, `analysis/block-2025-10-verify.json` |
| negative control | **rc=1** | head 0/13 present, both repaired checks FAIL |

The control is the October shape reproduced exactly: 2025-11's good block with the
thirteen head columns dropped, verified against the same 185-column old block.
Before the repair those two checks returned **`True` and `True`** on that pair
(measured, not asserted); after it they return False, and
`analysis/block-0000-control-verify.json` is the record. The control parquets were
deleted after the run; only the verdict file is kept.

### ii. Registration — 2025-03, the third month of common work

Same deliverable as §7bk (2025-11) and §7bo (2025-10): the six non-weather members
out of fold on `march_loo`, `base6` on 2025-03, and the 2025-03 block re-emitted so
it carries the twelve `{Nf_,No_}_n_res_*` columns and `own_obs_taxi_minus_p11`.
2025-03 is the next month in the registered 47.12 order after 2025-10.

**The month is now a parameter, not a copy.** Both October defects came from
hand-copying the November chain: `oct_finish.sh`'s emit line was written from
`emit.py`'s signature (so `--sources` fell back to `"F"` and the block lost 122
columns), and the first `oct_chain.sh` looked for `L2-hub` where the producer
writes `L2-hub8725`. Two defects of the same kind at the second copy is what the
structure is saying, so this month runs `work/month_chain.sh`, which takes
`PRC_LANE_MONTH` and `PRC_LANE_FOLD` and is otherwise byte-identical between
months. The emit line inside it is the **November** one, with the summaries rule:

```
code/emit.py <month> --sources F,O,H,N,geo --rule d4c59868…f01a0 --pred oof/base6--<month>.parquet
```

`lane.py` gained `march_loo` (installs in-process, 11 train months, holds 2025-03
out; `prc/data.py` is not edited — surfacefit's screen runs beside this lane).

**One more out-of-lane write, found while preparing the launch.**
`experiments/exp-012-blend2/winter_arm.py` has no `--outdir`: it writes both its
OOF parquet and `result-<name>.json` under its own `HERE`. §7bk and §7bo moved the
parquet out and deleted it there, but left `result-_q-D1-winter.json` (and a smoke
one) behind. Both were **untracked**, so no published record was overwritten — this
is milder than the exp-008 defect the desk caught — but it is the same class of
fault, so D1-winter now gets the same repointing: `HERE → work/d1-results/`, and
the two files left behind were moved there. `git status` on exp-008, exp-009,
exp-012, exp-018 and `catboost_info` is clean.

**Order and budget** (threads 4 total, strictly sequential, surfacefit beside us):
T-tailv2 + D1-winter (2 threads) → N1-nmtail (2) → L2-hub (4, 3 h cap) → C2-cat
(4, 5-min ETA gate) → F1-fresh (4, local CPU, gated on room for four fits inside
180 min) → base6 → block re-emit → verify. October's timings project **≈ 2.5 h**.
A watch is armed on the chain this time; the finisher is reported either way.

### iii. Month order change — the seasonal-distance rule replaces the calendar order

**Desk, 2026-09-10, mid-chain:** *"month order change, on a registered rule before
those months' data exist: surfacefit's 47.26 cyclic-seasonal-distance rule (3 for 3
within fold, unconfounded) supersedes 47.12's calendar order. After 2025-03 lands
(keep it: it is the rule's out-of-sample test and distance 2 from january), chain
2025-02 (distance 1 from the january fold and 1 from the served 2026-01) and then
2025-08 (distance 1 from july and from the served 2026-07)."*

So the remaining order is **2025-03 → 2025-02 → 2025-08**, and 2025-03 keeps its
place for a reason worth recording: it was chosen under the *old* rule, before the
new one existed, so its result is an **out-of-sample test of 47.26** rather than a
confirmation of it. At distance 2 it should help less than 2025-02 and 2025-08 do.
That prediction is registered here, before 2025-03's block exists.

`lane.py` gained `february_loo` and `august_loo` (both install, both hold their
month out of eleven training months). `work/month_queue.sh` runs the two remaining
months back to back after 2025-03, strictly sequential — the thread budget is 4
total and surfacefit's screen runs beside this lane, so two months must never
overlap. **A month that does not reach `DONE` stops the queue**: a HOLD means a
member is missing, and five more hours of fits should not run before that is read.

Blocks in hand for both: 2025-02 is 185 columns over 143,731 rows, 2025-08 is 185
over 191,181.

### iv. 2025-03 landed — the six members, base6 and the block

The chain ran unattended from 00:22:43Z to **02:48:20Z (2 h 26 min)**, one stage at
a time on four threads, and reached `DONE`.

| member | rmse on `march_loo` | wall | sha (16) |
| --- | ---: | ---: | --- |
| F1-fresh | **205.2146** | — | `437c572888c67a57` |
| C2-cat | 208.5872 | 76 min | `77a2ececdcb307d4` |
| T-tailv2 | 209.3149 | 2.8 min | `f5cf3832d218bbad` |
| D1-winter | 210.5268 | 4.2 min | `c73c6dca57f769d0` |
| L2-hub | 216.4721 | 14 min | `775f39531c12927f` |
| N1-nmtail | 356.8452 | 1.9 min | `230acef12aa8a0de` |

All six carry 164,450 rows with ids identical to the 2025-03 block, no duplicate
id, no null prediction, and `stack_members` loses no row. F1-fresh ran all four
fits (lgb 300, lgb 1200, CatBoost 3000, lgb 300) inside the gate, rc=0 — it is the
best single member of the month, as it was on 2025-10.

**base6 on 2025-03** — `oof/base6--2025-03.parquet`,
sha `64c47603ebfdb0c7cff438aeffb709f280e20417034e97269ed4a3ab380a1b81`, 164,450
rows, ids identical to the block, no null. Anchor 201.3728 → **base6 200.8562**
(the stacker adds 0.5166 s over the six-member hybrid anchor). The `_all` weight
vector puts 0.562 on F1-fresh, 0.272 on T-tailv4, 0.147 on L2-hub8725, 0.020 on
D1-winter and **zero** on C2-cat and N1-nmtail.

**The 2025-03 block** — `artifacts/exp-047/block/2025-03.parquet`,
sha `ec26128f3a5eaa74ba73d2b7c7a7476ed29d26d6cc5f0cc5760f4f050b20b55d`,
**186 columns** over 164,450 rows, verified by the repaired verifier: 6/6 OK,
head columns **13/13 present**, exactly the twelve `_n_res_` moved, exactly
`own_obs_taxi_minus_p11` added, none dropped
(`analysis/block-2025-03-verify.json`). This is the first month whose block passed
a verifier that could have failed it on absence — 2025-10's first emission passed
the same-named checks while missing 122 columns.

`git status` on exp-008, exp-009, exp-012, exp-018 and `catboost_info` is empty:
every producer wrote inside this lane.

### v. Scoring §7br iii — what the three-month screen does and does not settle

**Desk, 2026-09-10:** *"Surfacefit's three-month screen {2025-11, 2025-10, 2025-03}
read +0.533 [+0.405, +0.705] over the one-month member, every fold clear; the
cyclic rule survived out of sample on March (your 7br iii presumes FLOOR; score it
when the numbers are on the record)."*

The rule survived: March was chosen under the **old** calendar order, before 47.26
existed, and the set that contains it still reads clear on every fold. That is the
out-of-sample test passing, and it is the desk's reading, not mine to re-litigate.

**My §7br iii prediction is not scored by this number, and I will not claim it is.**
What I registered was a *ranking*: 2025-03, at distance 2, should help **less** than
2025-02 and 2025-08, which are at distance 1. That makes March's contribution a
**floor** the two distance-1 months must clear. Scoring it needs two things this
number does not carry:

1. **A per-month increment.** +0.533 is the joint read of the set
   {2025-11, 2025-10, 2025-03} over the one-month member, so it bundles 2025-10 and
   2025-03 together. Neither month's own contribution is separable from it.
2. **2025-02 and 2025-08.** They are not measured yet; 2025-02 is fitting now.

**So the falsifier, written before those numbers exist:** on a like-for-like
comparison — the increment of adding each month to the *same* base, in the same
screen form — §7br iii **FAILS** if either 2025-02 or 2025-08 adds less than
2025-03 did, and PASSES if both add more. A joint read of a set containing March
cannot decide it either way.

### §7bs — pre-registration of the v21 upload with v10 (face pending)

**Desk:** *"pre-register 7bs, the v21 upload with v10, anchored on v20's 245.2901
with the table at median 1.235, face = the bag's read over v9's head, before any
upload."* The bag is running; that read does not exist yet.

Third registration of this kind (v19 §7ba, v20 §7bp, v21 here), so the candidate is
now a parameter: `work/prereg.py`. **The face is deliberately left open, and the
band is registered as a function of it** — `analysis/v21-preregistration-pending.json`
carries the whole grid, so when the bag reports, the row is **read off** rather than
chosen. That is what a pre-registration is for; completing it after the number is
known would leave a degree of freedom that this one does not have.

| anchor | v20's board 245.2901 (reading 19, rank 1 of 94, lead 15.64 s) |
| --- | --- |
| transfers, n=8 | 1.75, 1.03, 2.01, 0.36, 0.70, 1.44, 2.41, 0.95 |
| median / mean | **1.2350** / 1.33125 |
| range | 0.36 (v16) – 2.41 (v19) |
| two most recent | v19 2.41, v20 0.95 |
| failure line | a move below **0.36 × the bottom of the face interval** |

Rows from the registered grid (median transfer, and the score above which the
upload has failed to transfer):

| face | move at median | score at median | fails above |
| ---: | ---: | ---: | ---: |
| +0.40 | +0.4940 | 244.7961 | 245.1807 |
| +0.50 | +0.6175 | 244.6726 | 245.1533 |
| +0.533 | +0.6583 | 244.6318 | 245.1443 |
| +0.60 | +0.7410 | 244.5491 | 245.1259 |
| +0.70 | +0.8645 | 244.4256 | 245.0986 |

**The face is the bag's read over v9's head**, not the screen's +0.533 over the
one-month member — a different quantity against a different baseline, and the v20
registration is the precedent: the screen read +0.59 at K=1 where the bag read
+0.513. While the face is pending the grid uses the CI *shape* both recent bags
produced (`[0.76×, 1.33×]` of the point, from +0.513 [+0.392, +0.688] and +0.533
[+0.405, +0.705]); the real interval replaces it on completion.

Every row leaves rank 1 by more than 14 s against the next team's 260.931.

### vi. Desk note for the record — 2025-08 is a discriminator, and 33.3 does not transfer

**Desk, 2026-09-10, after surfacefit's 03:12 UTC flag:** *"2025-08 is built as the
DISCRIMINATOR between the LIN and FLOOR forms of the cyclic rule (47.26), not yet
as a panel member; the fifth month is Aug if FLOOR wins, 2025-04 if LIN wins,
registered before Aug's read. The 33.3 fold-swap gate does not transfer to months
(it prices a different arm on every fold); rule-level cross-fit instead, surfacefit
computes it from existing readings. Your order 2025-02 then 2025-08 stands
unchanged."*

Three consequences for this lane, recorded without changing any running work:

1. **The order is unchanged and the chains are untouched.** 2025-02 is fitting;
   2025-08 follows. What changes is what 2025-08's number is *for* — a rule-form
   discriminator rather than a panel member — and that is surfacefit's reading to
   take, not this lane's.
2. **§7bn's use of 33.3 was fold-level and stands; the month level is not the same
   instrument.** §7bn cross-fitted a (class, member) drop across folds because the
   *same* arm could be priced on a held-out fold. Across months the arm differs on
   every fold, so the fold-swap gate has nothing to hold constant — the desk's
   ruling matches what §7bn actually relied on, and no reading of this lane is
   affected.
3. **One thing must be settled before §7br iii can be scored, and it is a
   definition, not a measurement.** §7br iii registered that 2025-03 at distance 2
   helps **less** than the distance-1 months, and §7br v turned that into a
   falsifier on per-month increments. The desk has read that registration as
   presuming **FLOOR**. A strict "less than" is the shape a **LIN** rule predicts;
   under a FLOOR form the distance-1 and distance-2 increments could be equal and
   my claim would fail while the rule held. So the falsifier's verdict depends on
   which form 47.26 is being scored in, and I will not pick that reading myself:
   it is asked at the next landing report, not resolved here. The falsifier text
   itself is unchanged and stays as written.

### §7bs addendum — the face exists, and the row is still not read off

**Desk, 2026-09-10:** *"the three-month bag's face is +0.55876 [+0.43987, +0.72811]
over v9's head, clean +0.68340, every fold clear (47.40, c08fb02). Do not read off
7bs yet: wait for surfacefit's serve line saying audit-expect-v24.json is FINAL,
then (1) run the audit by the file's own how_to_run line, (2) read off the 7bs row
from the serve record's face."*

Noted and **not acted on**, which is the whole point of the sequence. Two reasons
this lane keeps the gap between knowing the number and registering it:

* **The read-off comes from the serve record, not from a message.** The registered
  source for a board-facing number is the artifact — here the serve record's face —
  and this lane's standing rule is that a reading is grepped from the registered
  text and never transcribed from prose. The desk's figure and the serve record's
  will almost certainly be the same number; the discipline costs nothing when they
  agree and catches exactly the case where they do not.
* **The audit runs first.** A face is a fold-side quantity; the audit is what says
  the candidate bytes are the thing that face was measured on. Reading off before
  the audit would register a band for a candidate not yet verified.

`analysis/audit-expect-v24.json` does **not exist yet** (checked). When surfacefit's
serve line says it is FINAL: run `work/audit.py` on the `how_to_run` line the file
itself carries, then `PRC_FACE=<serve record's face> PRC_FACE_CI=<its interval>
work/prereg.py`, which computes the row from the already-registered function and
writes `analysis/v21-preregistration-anchored.json`. The grid in
`analysis/v21-preregistration-pending.json` (committed `80bc70d`, before the face
was known) is what makes that a read-off rather than a choice.

## §7bt — the v10 audit (57/57 AGREE) and the v21 row read off

### i. The audit: 57 of 57 AGREE, on a line the file did not give

`analysis/audit-v24.json`. Candidate `exp-046-surface-v10.parquet`,
sha `7b0557a497f8c09e…`, base set v24, 194,681 rows changed against v18.
**57 checks, 57 AGREE, 0 DISAGREE, 0 PENDING, 0 unanswered.**

Board all-row gain 1.314991 [1.04256, 1.685052], clean 1.607294; per fold
july_loo 791.9058 s² (88,426 rows moved), december 1527.3832 (102,186),
january 776.3629 (69,839).

**The file's `how_to_run` line is stale and I did not run it.** The desk's
instruction was to audit by that line; the line inside `audit-expect-v24.json` is a
verbatim copy of **v23's**, and running it would have re-audited **v9** against
**v23's** expectations and returned an AGREE that says nothing about v10:

| argument | the file's line | what the record says | source of the correction |
| --- | --- | --- | --- |
| `--candidate` | `exp-046-surface-v9.parquet` | `exp-046-surface-v10.parquet` | `serve-v24.json` `written_to`; sha verified `7b0557a4…` |
| `--arm-npz` | `d-onemonth-bag10-255-free.npz` | `d-threemonth-bag10-255-free.npz` | `serve-v24.json`; `pending_because` names the three-month bag |
| `--class-vector` | `{month}-v23.parquet` | `{month}-v24.parquet` | `base_set: "v24"` |
| `--previous-class-vector` | `{month}-v22.parquet` | `{month}-v23.parquet` | v24 = v23 with one change (`what`) |
| `--expect` / `--out` | `…-v23.json` | `…-v24.json` | the file being run against |

Everything else is unchanged and each unchanged argument is also sourced:
`--head-npz` is the file's own pin, `--previous`/`--baseline` are
`serve-v24.json`'s `base.bytes_file` (v18), `--hold ''` is `why_the_hold_is_empty`,
and `--changed-expected-at LFPG` is note 3. **No argument was invented**; every one
is read out of a registered artifact. The file's own instruction — *"Check this sha
BEFORE the audit"* — was honoured first: the head npz hashes to
`5dbbcb8c7d14831f…`, exactly its pin, so the increment is over the intended base.

**A second stale label, for surfacefit rather than for me:**
`analysis/threemonth-bag10.json` stores the price under the key
`in_fold_over_v8_head` while its own `priced_over` field says **"v9's head"**. The
field and the desk agree, so the quantity is not in doubt — but that is the same
copy-forward as the `how_to_run` line, in the same serve, and a reader who trusts
the key name reads the wrong baseline.

### ii. The read-off — §7bs completed at the record's face

Face taken from `analysis/threemonth-bag10.json` `verdict`, not from the message:
**+0.5587645784786446 [+0.43986622607635584, +0.7281067404968451]** over v9's head,
`priced_over: "v9's head"`, every fold clear, clean +0.6833968168952538. It matches
the desk's quoted figure to every digit given. `analysis/v21-preregistration-anchored.json`.

| row | move on 245.2901 | score |
| --- | ---: | ---: |
| at face, transfer 1× | +0.5588 | **244.7313** |
| **at the median, 1.2350×** | **+0.6901** | **244.6000** |
| at the mean, 1.33125× | +0.7439 | 244.5462 |
| observed transfer range at face | [+0.2012, +1.3466] | [243.9435, 245.0889] |
| observed range on the face interval | [+0.1584, +1.7547] | [243.5354, 245.1317] |
| two most recent (2.41×, 0.95×) | [+0.5308, +1.3466] | [243.9435, 244.7593] |
| **failure line** | a move below **+0.1584** | a score **above 245.1317** |

The row was computed by `work/prereg.py` from the function committed at `80bc70d`,
**before the face was known** — the only new input is the face itself, so this is a
read-off and not a choice. Rank 1 is not at risk in any row: the next team is at
260.931.

### iii. The regenerated expectation file — verified, and it matches the line I built

`audit-expect-v24.json` regenerated by surfacefit (`641dd38`, 47.43.4),
sha `3c90c20f21a7f7ba38e11af89c68498334e3fe82c1af894d37eaa3ddd8b7c3be`.

**Verified against the file that was actually audited** (its committed form at
`51ba8c8`), by value and not by trust: same 57 keys, **0 of 57 expectation values
differ**, and exactly four fields moved — `how_to_run`,
`notes_on_the_expectation`, `sources`, `the_extension_audit_py_cannot_see`, all
prose. So the AGREE stands unchanged on the regenerated file.

**The re-run is not needed and would be a tautology**: the audit's output is a
function of the expect file and the command, and the regenerated `how_to_run` is
**argument for argument the line §7bt i built** — same head npz, same
`d-threemonth-bag10-255-free.npz`, same `{month}-v24`/`{month}-v23` class vectors,
same v10 candidate, same v18 previous and baseline, same empty hold, same
`--changed-expected-at LFPG`, same expect and out paths. The only difference is
`OMP_NUM_THREADS=2` where this lane ran 1, to leave February's F1-fresh its four.

That independent agreement is worth more than the re-run would be: the corrected
line was derived here from `serve-v24.json` and the file's own pins, and surfacefit
then composed the same line from the same serve record without seeing this one.

The bag record's key is now `in_fold_over_v9_head` with the values unchanged, so
the second stale label is closed too.

### vii. 2025-02 landed

00:22 to 05:22 UTC for the two months together; 2025-02 alone ran 02:49 → 05:22
(**2 h 33 min**) and reached `DONE`, rc=0.

| member | rmse on `february_loo` | sha (16) |
| --- | ---: | --- |
| F1-fresh | **255.6674** | `bae2dc7c0e0e3544` |
| T-tailv2 | 256.4864 | `80bfc55d4eba4f5b` |
| D1-winter | 267.0995 | `b3bf63382fdd6c84` |
| C2-cat | 283.6241 | `f757507e3f9a4ede` |
| L2-hub | 283.8751 | `05b832aebb115016` |
| N1-nmtail | — | `25f543fd7ba196d6` |

All six carry 143,731 rows with ids identical to the block. February prices much
harder than March on every member — L2-hub 283.9 against March's 216.5 — which is
what a de-icing month should look like, and the ordering of the members is
unchanged: F1-fresh first, N1-nmtail last.

**base6 on 2025-02** — sha `36f7dd9f6d80451aa9dd5e0ad74e990ab41f80cd4f1e8249079d2c5115f3ba1c`,
143,731 rows, ids identical, no null. Anchor 251.5716 → **base6 247.6790**: the
stacker adds **3.8926 s** here against 0.5166 s on 2025-03, by far the largest
step-2 gain of the three months.

**The 2025-02 block** — sha `32c9e25d05f6ad21e964a26f39b52d7d4f6c16061fc33b3b1c1e107908efb533`,
**186 columns** over 143,731 rows, verify **6/6** with head columns 13/13 present,
the twelve `_n_res_` moved, `own_obs_taxi_minus_p11` added, none dropped.

Other lanes clean. 2025-08 started by itself at 05:22:39Z.

**One defect in my own watch, recorded because it will bite again.** The §7br
monitor de-duplicates its event lines across all chain logs, so 2025-02's verifier
output — six lines of identical text to 2025-03's — emitted **nothing**. The
landing was caught by the `DONE` line, and the verdict was read from
`analysis/block-2025-02-verify.json`, which is the authoritative record; but a
watch that silently drops a repeated PASS would equally drop a repeated FAIL. The
verdict is always read from the json, never from the absence of an event.

## §7bu — BOARD READING 20, and the v22 pre-registration

### i. Reading 20 against §7bs

**Desk, 2026-09-10:** *"v21 = v10 uploaded 05:27:56 UTC, scored 245.094: −0.1961 on
v20's 245.2901, transfer 0.351× on the +0.55876 face, under your 7bs failure line
245.1317 by 0.038 s, no failure; rank 1 of 95."*

| §7bs registered row | expected score | outcome |
| --- | ---: | --- |
| at face, 1× | 244.7313 | missed by +0.3627 |
| at the median, 1.2350× | 244.6000 | missed by +0.4940 |
| at the mean, 1.33125× | 244.5462 | missed by +0.5478 |
| observed range at face | [243.9435, 245.0889] | **just outside**, by 0.0051 |
| observed range on the face interval | [243.5354, 245.1317] | **contains it** |
| two most recent (2.41×, 0.95×) | [243.9435, 244.7593] | outside |
| **failure line** | above 245.1317 | **not breached**, by 0.0377 s |

**No failure, and the margin is thin: 0.038 s.** The registration did its job — the
widest band contained the outcome and the failure line was not reached — but every
point row was optimistic, and the reading landed 0.005 s outside the
observed-transfer-range row. That is the second consecutive reading below the
median row (v20 0.95×, v21 0.351×) after v19's 2.41×.

**The transfer table, n = 9:**

| upload | v13 | v14 | v15 | v16 | v17 | v18 | v19 | v20 | v21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transfer | 1.75 | 1.03 | 2.01 | 0.36 | 0.70 | 1.44 | 2.41 | 0.95 | **0.351** |

**Median 1.0300** (was 1.2350), mean **1.22233**, range **0.351 – 2.41**. v21 is the
**lowest transfer this programme has recorded**, displacing v16's 0.36 — so every
future failure line moves with it, and the v22 grid below already uses 0.351.

**Combined:** v18 → v21 is −1.2665 on a combined face of +1.31499, **0.963×** —
three uploads together transfer almost exactly at face while the last one alone
transferred at a third of it.

### ii. Does a bigger face transfer worse? Measured, and the answer is no

Two of the three largest faces ever registered (+0.513, +0.55876) produced two of
the four lowest transfers, which invites the story that big fold-side reads do not
carry. **It does not survive the history.** Spearman rank correlation between the
registered face and the transfer:

| set | n | ρ | p |
| --- | ---: | ---: | ---: |
| all nine uploads | 9 | **+0.167** | 0.67 |
| without v13's +8.128 face | 8 | **−0.048** | 0.91 |
| the last four only | 4 | −0.800 | 0.20 |

The full history shows **nothing**, in either direction. Only the last four points
show the pattern, at n=4 and p=0.20 — which is what four points do. So the v22 band
is **not tilted** for a face effect, and this lane will not read one into the next
registration. Recording it matters because the impression is strong enough to act
on and the evidence is not; the faces are now carried in `work/prereg.py` as
`HISTORY_FACES` so the question is re-asked from data each time rather than from
memory.

### iii. v22 pre-registered as a function of its face, before its bag

**Desk:** *"Next pre-registration: v22 = the seasonal head (surfacefit 47.46, order
iii), anchored on 245.094, as a function of the face, before its bag."*

`analysis/v22-preregistration-pending.json`, written **before the bag runs**, so
the face is the only input still missing and the row will be read off, not chosen.
Anchor **245.094** (reading 20, rank 1 of 95); transfers n=9, median **1.0300**,
mean 1.22233; two most recent 0.95× and 0.351×; failure at **0.351 × the bottom of
the face interval**.

| face | move at median | score at median | fails above |
| ---: | ---: | ---: | ---: |
| +0.30 | +0.3090 | 244.7850 | 245.0119 |
| +0.40 | +0.4120 | 244.6820 | 244.9845 |
| +0.50 | +0.5150 | 244.5790 | 244.9571 |
| +0.60 | +0.6180 | 244.4760 | 244.9297 |
| +0.70 | +0.7210 | 244.3730 | 244.9024 |

While the face is pending the grid uses the CI shape the two served bags produced
(`[0.78×, 1.32×]`, from +0.513 [+0.392, +0.688] and +0.55876 [+0.43987, +0.72811]);
the bag's own interval replaces it. The face must come from the bag's record and
never from a screen figure against a different baseline — the §7bs sequence stands.

### viii. 2025-08 landed — and the three-month block is closed

2025-08 ran 05:22 → 07:45 (**2 h 23 min**), rc=0, `QUEUE DONE`. All three months
of §7br are in: 00:22 → 07:45, **7 h 23 min** unattended for three months.

| member | rmse on `august_loo` | sha (16) |
| --- | ---: | --- |
| F1-fresh | **245.3793** | `f98e31e0ba355e10` |
| C2-cat | 253.6815 | `bc19d243289b91cf` |
| D1-winter | 257.0108 | `0058a578cf9a3572` |
| L2-hub | 258.6012 | `f676a9e97c96b6ea` |
| T-tailv2 | 261.7637 | `3d34a116f3efc81a` |
| N1-nmtail | — | `350742ccf9e05b78` |

All six carry 191,181 rows with ids identical to the block.

**base6 on 2025-08** — sha `60c84140c3342390866ebf4c6b797692737612126f11ff718328311f9da067a6`,
191,181 rows, ids identical, no null. Anchor 244.3131 → **base6 243.3441**
(+0.9690).

**The 2025-08 block** — sha `137a61f23de48d9d006779fb27e4428e97e877c742fec2fd8291318eb1539504`,
**186 columns** over 191,181 rows, verify **6/6**, head columns 13/13 present.

**Five months now carry the 186-column block**: 2025-02 (143,731), 2025-03
(164,450), 2025-08 (191,181), 2025-10 (185,674), 2025-11 (162,332). Every producer
wrote inside this lane; exp-008, exp-009, exp-012, exp-018 and `catboost_info` are
clean at the end of all three chains.

**What the three months say, since every result gets a why.** August is the
easiest month on every member (F1-fresh 245.4) and February the hardest (255.7);
March sits between. The member ordering is stable except for one swap — D1-winter
beats T-tailv2 on August, the one summer month, where a de-icing arm has nothing to
price, and C2-cat leads until F1-fresh runs. **The step-2 gain is where the months
really differ**: the stacker adds 0.5166 s on 2025-03, 0.9690 on 2025-08 and
**3.8926 on 2025-02**. February's six members disagree far more than the other
months', and the stacker is paid for exactly that disagreement — which is the
month-level version of what §7bk measured across regimes, and a reason to expect
February's rows to carry more weight in an extension than their count suggests.

## §7bv — the next four months, in the seasonal design's order

**Desk, 2026-09-10:** *"Next builds, in the registered order the seasonal design
fixes by served distance (surfacefit 47.46): 2025-06 first (d=1 from served July,
the only unbuilt d=1 month), then 2025-04 (d=3 from served January, the nearest
unbuilt month on that side), then 2025-05 and 2025-09 (d=2 from July). Same chain,
same checks, same reporting; write the order as 7bv with the reason BEFORE starting
2025-06. The extension stays base6; the true-base question (recorded rule token) is Matt's."*

**Written before 2025-06 starts.** The order and its reason, checked against the
calendar rather than copied:

| month | d to served 2026-01 | d to served 2026-07 | state | position |
| --- | ---: | ---: | --- | --- |
| 2025-06 | 5 | **1** | unbuilt | **1st** |
| 2025-04 | **3** | 3 | unbuilt | **2nd** |
| 2025-05 | 4 | **2** | unbuilt | **3rd** |
| 2025-09 | 4 | **2** | unbuilt | **4th** |
| 2025-02 | 1 | 5 | built §7br | — |
| 2025-08 | 5 | 1 | built §7br | — |
| 2025-03 | 2 | 4 | built §7br | — |
| 2025-11 | 2 | 4 | built §7bk | — |
| 2025-10 | 3 | 3 | built §7bo | — |
| 2025-01, 2025-07, 2025-12 | 0 / 0 / 1 | 0 / 0 / 5 | **fold months** | never |

(cyclic distance, `min(|a−b|, 12−|a−b|)`.)

**Why 2025-06 is "the only unbuilt d=1 month", verified:** the four d=1 months are
2025-12 and 2025-02 on the January side and 2025-06 and 2025-08 on the July side.
2025-02 and 2025-08 are built; **2025-12 is a fold month** — january, december and
july_loo hold 2025-01, 2025-12 and 2025-07 out, so those three can never join a
fitting panel. That leaves 2025-06 alone, exactly as the desk states.

**Why d=3 comes before d=2, which is the one thing the order does not say on its
face.** A pure served-distance sort would run 2025-06, then 2025-05 and 2025-09,
then 2025-04. The order given interleaves instead, and the reading I proceed under
is **side balance**: after 2025-06 the July side has three months in the panel
(2025-08, 2025-06, and 2025-10 at d=3) while the January side has three (2025-02,
2025-03, 2025-11) — but the January side has **no unbuilt month nearer than d=3**,
so taking 2025-04 second is the only way to keep the two served months growing
together instead of stacking four July-side months in a row. This was written as my reading of a
rule the desk registered, so that it could be corrected; **the desk confirmed it
the same hour** — *"side balance is the rule I meant — each served month gets its
next-nearest unbuilt month in turn (July 2025-06 at d=1, then January 2025-04 at
d=3, then July's d=2 pair), so neither side stacks; your 7bv wording stands as the
registered reason."* So the reason above is registered, not inferred, and the
order is unchanged either way.

Same chain (`work/month_chain.sh`), same six members, same repaired verifier, same
per-month landing report. `lane.py` gained `june_loo`, `april_loo`, `may_loo` and
`september_loo` — all four install with eleven training months, and `may_loo` does
not collide with the shipped `may` fold. All four blocks are in hand at 185
columns: 2025-04 (175,288 rows), 2025-05 (185,202), 2025-06 (183,142), 2025-09
(183,951). **The extension stays base6**; the true-base question is Matt's and this
lane does not touch it.

### §7bv i — a second watch defect, and what the repeat says

The §7bv monitor exited **one poll after it was armed**, reporting the whole queue
as ended, because its exit condition was `grep -q 'QUEUE DONE' logs/month-queue.log`
and that log is **appended to across runs** — §7br's `QUEUE DONE` from 07:45 was
still in it. Nothing was lost (the queue is a separate orphaned process and ran on
untouched), but a watch that ends instantly is a watch that would sleep through a
failure.

That is the **second** defect in this lane's watch in one session; the first was
the de-duplication that swallowed February's verifier lines. Two of a kind says
something about the structure rather than about either bug: **both conditions were
derived from log TEXT, in files that outlive the run that wrote them.** The repair
is the same for both and is now in place — the watch keys on the **process**
(`ps -p` on the queue pid) for its end, and prefixes every event line with its
source file (`grep -H`) so identical text in two months is two events. The seen-set
is also seeded from the existing logs at arm time, so a re-arm reports only what is
new.

The standing rule this lane already applies to blocks applies to watches too: a
verdict is read from the record (`analysis/block-<month>-verify.json`), never from
the presence or absence of an event.

## §7bw — can `work/audit.py` audit the two-head candidate? Not as it stood

Registered **before `audit-expect-v25.json` exists**, from exp-046 `code/serve.py`'s
`"v25"` declaration (47.49 / recorded rule) read directly rather than from a description.

**The one-line answer: no — one of the three parts needed nothing, one needed a new
instrument, and one is not a byte question at all.**

| what v25 adds | could audit.py see it? |
| --- | --- |
| **two arm vectors** | **Nothing needed.** The declaration's fold side is **one** bag, `work/d-seasonal-bag10-255-free.npz`, because each leave-one-out fit carries its own fold's month set (january and december the Feb set, july the Aug set). `audit_increment` takes one `--arm-npz` and that is still correct. The **two** bags are on the ranking side, which audit.py never priced from arm vectors. |
| **two member sets** | **Partly.** `extension_by_served_month` is fitting provenance, which is in the standing by-eye list. But the class vector's new label encodes the month set per served month, so the label can now be checked against the serve's own declaration. |
| **head by served month** | **No, and this was the hole.** Every class check is boolean occupancy. A serve that applied the **January head to both served months** produces the identical occupied set, the identical changed set and identical counts — it would have passed all 57 checks. |

**What was added, and nothing else.** Three flags, all inert when absent:

* `--head-col` (default `head_by_exp_046`): when the column exists, the audit reports
  the labels, the label per served month, `one_label_per_served_month`,
  `the_served_months_have_different_heads`, `changed_rows_with_no_head`, the changed
  rows by head, and **`rows_whose_head_is_not_their_own_served_month`**.
* `--head-declares`: the serve's `{served month: [fitting months]}`, compared with the
  month set parsed out of the class vector's own label — this is where the two member
  sets become checkable, and a served month **missing** from the serve's map (fatal by
  the base set's own words) is reported.
* `--bag-cache`: `{served month: glob}` for the two ranking caches. exp-046 states the
  risk plainly — the cache key is seed+leaves and cannot tell one arm from another, so
  with two arms a shared key silently gives both months the same head. The audit now
  hashes both file sets and fails if they share a file **or a sha**.

**Three controls, run before this was registered:**

| control | result |
| --- | --- |
| two heads, two bags (synthetic v25 class vector) | every check **true** |
| **one head written on both served months** | `different_heads` **False**, `label_matches_the_serve_declaration` **False**, and **192,122** rows whose head is not their own served month — three independent failures |
| **the two bags share a cache key** | `the_two_bags_are_distinct` **False**, with the shared shas listed |

**Regression: the v24 audit is unchanged.** Re-run on its own `how_to_run` line after
the edit: **57/57 AGREE**, and the output is identical to the first run key for key
apart from the `arguments` block. The new keys do not appear when the column and the
flags are absent, so every line already run reproduces.

**The honest limit, written into the audit's own output rather than left to a
reader.** `head_by_exp_046` is the serve's declaration echoed onto every row, not a
measurement of the bytes: it proves the vector *claims* a two-head composition and
that the claim is internally consistent — not that two distinct heads were fitted.
The candidate's bytes cannot distinguish two heads from one head written twice. The
cache-key check is the closest byte-level evidence there is; the rest stays in the
by-eye list beside the extension, and `audit-expect-v25.json` should carry it there.

### §7bw ii — the byte-level control: one head written twice cannot pass

**Desk:** *"Your LIMIT is right and it has a byte-level answer: the twenty ranking
member files."* It does, and the composition needed to use them was **verified on
real bytes before anything was built on it.** On v10, with its ten `d-rank-ext3-*`
members:

```
y = base bytes                              where not written
y = base bytes                              where guard_held        (1,573 rows)
y = max(price + mean(members), 60)          elsewhere             (194,680 rows)
```

reproduces the candidate on all **344,841 rows to max diff exactly 0**. The floor is
60 s (3 rows), the price base is `exp-035-k2climb-v11member.parquet` and the base
bytes are v18 — none of that was assumed, each was found by testing the identity and
watching the residue fall to zero.

`--rebuild` takes `{price, bytes, floor, members: {served month: glob}}` and reports:
the **two-set** rebuild against the candidate (rows differing, max abs diff), each
**one-set-everywhere** rebuild against the candidate **on the other served month's
written rows** (rows differing, rms, max), and the rms between the two sets' means on
the written rows.

**Controls, and both of them earned their keep:**

| control | result |
| --- | --- |
| candidate composed from the two sets | two-set reproduces (0 rows above tolerance), January-set-on-July differs on 104,607 of 104,823 rows at rms 3.0001, July-set-on-January on 90,073 of 91,433 at rms 2.9815 → **refuted: True** |
| **one head (January set) written on both served months** | January-set-on-July reproduces the candidate on **0 of 104,823** rows differing, rms 0.0000 → **refuted: False**. Caught, exactly as the desk specified. |

**Two defects the controls found in my own first cut**, both now fixed and both worth
recording because they are the failure mode this check exists to prevent — a
refutation that is really noise:

1. **Exact inequality counted float noise as a difference.** Two member sets that were
   *identical* on a month still produced 3,325 "differing" rows at rms 0.0000, which
   would have read as a refutation of a head that had not been distinguished at all.
   A `tol` (default 1e-9 s, nine orders below any move that matters) now gates every
   differ count, and `rows_where_the_two_sets_agree_exactly` is reported beside it.
2. **`np.mean` over ten files sums in glob order** — `jan-1, jan-10, jan-2, …` — and a
   different order changes the last bit: measured **4.5e-13 s** on a control whose
   composition was otherwise identical. The verdict therefore uses the tolerance while
   the **exact** `max_abs_diff_s` is reported beside it, so a real composition
   difference can never hide behind the tolerance. Whoever writes `audit-expect-v25`
   should expect `max_abs_diff_s` at 0 on a serve composed in the same order and read
   `rows_differing_above_tolerance` as the claim.

**Regression: v24 re-audits 57/57 AGREE**, output identical key for key to the first
run, and the `rebuild` key does not appear when the flag is absent.

**So the limit in §7bw is now answered by bytes**: the class label still only echoes
the declaration, but the twenty member files decide it. A candidate whose 2026-07 rows
are reproduced by the January set is one head written twice, and this check says so.

### §7bv ii — 2025-06 landed

07:49 → 10:16 UTC, **2 h 27 min**, rc=0.

| member | rmse on `june_loo` | sha (16) |
| --- | ---: | --- |
| F1-fresh | **306.4926** | `636a0ee1903d17a0` |
| T-tailv2 | 314.3793* | `403bddf46e864784` |
| D1-winter | 314.9833* | `9b4bd0ec29866068` |
| C2-cat | 317.4544 | `00e4066f3a6147b6` |
| L2-hub | 321.1669 | `f2d95fc38926432a` |
| N1-nmtail | — | `f3691e235725653f` |

(*the two cheap members are within 0.6 s of each other and the log does not name
which row is which; both shas are recorded and the vectors are committed.)

**base6 on 2025-06** — sha `0bff146d3bced00799edd44f3d30c22431ca41ad269f5ccec7e3bdd0c5d4686c`,
183,142 rows, ids identical, no null. Anchor 305.1182 → **304.4005** (+0.7177).

**The 2025-06 block** — sha `5e7e6e10898d2445d09079e8dc69521b489931ccce796be3c5c31479afe342f4`,
186 columns over 183,142 rows, verify **6/6**, head 13/13 present. Six months now
carry it. Other lanes clean; 2025-04 started at 10:16:14Z.

**June is by a wide margin the hardest month built.** Every member sits above 306
where August ran 245–262 and March 209–216, and the spread between best and worst
member (14.7 s) is the narrowest of the six — the members agree with each other and
are all wrong together, which is the opposite of February's picture (a 28 s spread
and a 3.89 s step-2 gain). Accordingly June's stacker gain is small: **+0.7177**.

That matters for the seasonal design rather than for this lane, so it is recorded
and not acted on: **June is the July head's d=1 month, and it is the noisiest month
in the panel.** A rule that picks by served distance alone cannot see that. Whether
noisy near rows beat quiet far ones is exactly what 2025-05 and 2025-09 at d=2 will
price, and this lane will not pre-judge it.

**The watch repair held.** §7bv i's `grep -H` fix delivered all six verifier lines
for this month — the failure that swallowed February's identical text is closed.

### §7bv iii — the d=1 tie, held for the next design

**Desk, 2026-09-10:** *"June is a second built month at d=1 from served July, so
47.46's 'nearest built month, unique' clause now ties Jun/Aug for the July head. v11
stands (registered and launched before June's data existed). The tie is a ruling for
the NEXT design, registered before any June read; hold it, no pre-judging."*

Recorded and **held**. One distinction is worth writing down now, because a later
reader diffing the record could easily mistake the tie for a defect in v11:

**v11's uniqueness claim was true when it was made.** exp-046's v25 declaration
argues rule (a) from *"the month per served month is the nearest BUILT month by
served distance, Feb at d=1 and Aug at d=1, both unique: parameter-free under
monotonicity, chosen by no fold's reading."* At the moment that was registered and
launched, 2025-06 did not exist as a built month, so Aug **was** the unique nearest
built month on the July side. This lane building June is what created the tie, and it
did so *after* the selection was fixed. The claim is therefore intact for v11 and
open for whatever comes next — which is exactly the desk's ruling.

**No June reading is used here, and none exists.** §7bv ii's note that June is the
noisiest month in the panel is a statement about its six members' fold RMSEs, not a
read of June as an extension month; nothing in it prices Jun against Aug. The tie
gets broken, if it gets broken, by a measurement nobody has taken yet.

2025-04 continues.

## §7bx — the v11 audit (57/57 AGREE, two-head control passed) and the v22 read-off

### i. The audit

`analysis/audit-v25.json`. Candidate `exp-046-surface-v11.parquet`, sha
`f11d34f88b62a43e…` (checked against the file's own pin **before** the run, as it
instructs; the head npz also hashes to its pin `5dbbcb8c7d14831f…`).
**57 checks, 57 AGREE, 0 DISAGREE, 0 PENDING, 0 unanswered.** Board all-row gain
1.712996 [1.364449, 2.185899], clean **2.094463**; per fold july_loo 1099.3140 s²,
december 1595.8149, january 924.6066. This file's `how_to_run` line was correct as
given — v11, the seasonal arm npz, v25/v24 class vectors — and was run as written.

### ii. The two-head control — the declaration is now a measurement

`analysis/audit-v25-twohead.json`, from `analysis/rebuild-v25.json` built out of the
expectation file's own ordered member lists rather than retyped.

**Two corrections the real expectation forced on §7bw ii, both right:**

1. **Numeric member order, and therefore no tolerance.** The file states the serve
   stacked its members seed 1 → 10 and that a float64 mean is summation-order
   dependent, so glob order reproduces only to the ULP class. `--rebuild` now takes an
   ordered **list**. In numeric order the rebuild is **exact**: `max_abs_diff_s`
   **0.0**, `rows_differing` **0** on all 196,256 written rows — the exact zero is what
   says the order was right, which is precisely what the file predicted.
2. **Compose by the movement table, not by the declaration.** The served month now
   comes from `MVT_TIME_UTC_mvt` in the raw ranking (152,719 rows at 2026-01 and
   192,122 at 2026-07), never from `head_by_exp_046` — composing by the label would
   assume the thing under test.

**The result, against the expectation's own numbers:**

| quantity | expectation | measured |
| --- | ---: | ---: |
| 2026-01 vs 2026-07, rows | 152,719 | **152,719** |
| … rows that differ | 91,433 | **91,433** |
| … rms | 12.996927191336386 | **12.996927191336386** |
| … max abs | 230.22449909469384 | **230.22449909469378**\* |
| 2026-07 vs 2026-01, rows | 192,122 | **192,122** |
| … rows that differ | 104,823 | **104,823** |
| … rms | 10.762323680986606 | **10.762323680986606** |
| … max abs | 251.47961289202453 | **251.47961289202476**\* |

(\*the two max-abs figures agree to 13 significant digits; the last-bit difference is
the same summation-order effect the file names, and it is reported rather than
rounded away.)

**One head written twice is refuted, by bytes.** The January set applied to all served
rows fails on **104,608 of 104,823** July rows (rms 14.4177, max 251.48); the July set
on January rows fails on **90,074 of 91,433** (rms 15.6608, max 230.22). The two
ranking bag caches are distinct files with no shared sha.

**And so the declaration is promoted:** `head_by_exp_046` disagrees with the movement
table on **0 rows**, and the class month agrees with it on **0 disagreements**. The
label is now a measurement, which is exactly what §7bw said it was not on its own.

**One correction to the expectation's prose, measured not argued.** Step 3 says rows
outside the class and rows the guard returned "must equal the price base byte for
byte". Measured: the 148,585 rows outside the written class are identical to the
**base bytes (v18)** and *not* to the price base; the 1,573 guard-held rows are
identical to **both** (the two agree there). The candidate is correct; the sentence
names the wrong file for the outside rows.

### iii. The v22 read-off

Face taken from `exp-046 analysis/seasonal-bag10.json` `verdict`, whose `priced_over`
is `"v10's head"`: **+0.3980055221207408 [+0.29548200589360646, +0.5318926020786562]**,
every fold clear. It matches the desk's quoted +0.398005522.
`analysis/v22-preregistration-anchored.json`.

| row | move on 245.094 | score |
| --- | ---: | ---: |
| at face, 1× | +0.3980 | 244.6960 |
| **at the median, 1.0300×** | **+0.4099** | **244.6841** |
| at the mean, 1.22233× | +0.4865 | 244.6075 |
| observed transfer range at face | [+0.1397, +0.9592] | [244.1348, 244.9543] |
| observed range on the face interval | [+0.1037, +1.2819] | [243.8121, 244.9903] |
| two most recent (0.95×, 0.351×) | [+0.1397, +0.3781] | [244.7159, 244.9543] |
| **failure line** | a move below **+0.1037** | a score **above 244.9903** |

Computed by the function committed at `e3f1c65` before the face was known; the face is
the only new input. The clean margin over v10 (+0.487168688) is **not** the face — the
programme's nine transfers are all measured on the **all-row** increment over the
previous head, and mixing a clean figure into that history would compare two different
quantities. Rank 1 is not at risk in any row.

### §7bv iv — the auto-continue was stopped before 2025-04 landed

**Desk:** *"let 2025-04 land and report it, then DO NOT auto-start 2025-05 — hold the
laptop fit slot for surfacefit's June sequence (47.57) … If the chain auto-continues
by itself, stop that mechanism before 2025-04 lands and say so."*

It would have. `work/month_queue.sh` was launched with all four months and starts the
next one the moment a chain reaches `DONE`, so 2025-05 would have taken the fit slot
by itself. **Stopped at 11:02:21Z, while 2025-04 was still in C2-cat.**

The mechanism is one process, so only that process was killed:

```
72666  bash month_queue.sh 2025-06 … 2025-04 … 2025-05 … 2025-09   <- killed
 92763   bash month_chain.sh (2025-04)    ppid now 1, still running
  95197    bash c2cat_hedge.sh            still running
```

`kill` on the queue alone leaves the chain reparented to init and untouched — the
running month keeps its fits and its C2-cat gate, and nothing follows it. The queue
log carries the reason at that timestamp so the next reader does not read the stop as
a crash. The watch is re-armed on the **chain** pid instead of the queue's.

2025-05 and 2025-09 resume on the desk's word; `month_queue.sh` takes its month list
as arguments, so resuming is one command with the two months that remain.

**Nothing round-13 reads was touched**: `analysis/audit-v25.json`,
`analysis/audit-v25-twohead.json`, `analysis/audit-expect-v25.json` and
`submissions/candidates/exp-046-surface-v11.parquet` are all unmodified since
`daf919c`.

## §7by — BOARD READING 21: the failure line is breached, and the registration prescribes nothing

**Desk, 2026-09-10:** *"v22 UPLOADED 11:13:12 UTC (round 13 PASS, 0 blocking;
ready-queue paragraph 4eaff57), scored 245.0207 at 11:13:33 UTC … On v21 245.094
that is −0.0733 s on the +0.398006 all-row face over v10 head: transfer 0.184×.
Against your 7bu read-off (median 244.6841, at face 244.6960, FAILURE above
244.9903) the score sits 0.0304 s ABOVE the failure line. score-feedback rule stands: nothing
revised on this reading. Do the registered 7bu read-off formally as 7by."*

### i. The verdict the registration gives

The registration is `analysis/v22-preregistration-anchored.json`, written by the
function committed at `e3f1c65` **before** the face was known and read off in §7bx
iii. Every row of it is now scored against the board:

| §7bx iii registered row | expected score | outcome |
| --- | ---: | --- |
| at face, 1× | 244.6960 | missed by +0.3247 |
| at the median, 1.0300× | 244.6841 | missed by +0.3366 |
| at the mean, 1.22233× | 244.6075 | missed by +0.4132 |
| observed transfer range at face | [244.1348, 244.9543] | **outside**, by 0.0664 |
| observed range on the face interval | [243.8121, 244.9903] | **outside**, by 0.0304 |
| two most recent (0.95×, 0.351×) | [244.7159, 244.9543] | outside, by 0.0664 |
| **failure line** | above **244.9903** | **BREACHED, by 0.0304 s** |

**The verdict is FAILURE, and it is the first in ten readings.** Not one band held.
The widest honest band — the lowest transfer ever recorded applied to the bottom of
the candidate's own fold interval — is exactly the failure line here, so the two
lines fall together and the reading is outside both.

The move is **+0.0733 s** (245.094 → 245.0207) on a registered all-row face of
**+0.398006**: transfer **0.1842×**. Three facts qualify the verdict and are
recorded with it:

1. **The move is positive.** The registered distinction is that *"a move at or below
   zero is not a weak transfer but no transfer at all"*. That is **not** what
   happened: v22 moved the board the right way, by 0.0733 s, and holds rank 1. This
   is a weak transfer, not an absent one.
2. **The breach is 0.0304 s**, which is 3 % of the registered failure move
   (+0.1037) and 0.012 % of the score. The line is breached on its own terms and by
   a margin the board can resolve, which is why it is stated as a breach and not as
   a near miss — but the size belongs in the record next to the word.
3. **The programme still transfers in aggregate.** v18 → v22 is **−1.3398 s** on a
   combined face of **+1.712996** (the v11 audit's own all-row gain over v18):
   **0.7821×** over four uploads, against 0.9631× over three. Four heads together
   deliver four fifths of their combined fold read; the last head alone delivered
   under a fifth of its own.

### ii. The transfer table, n = 10

| upload | v13 | v14 | v15 | v16 | v17 | v18 | v19 | v20 | v21 | v22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transfer | 1.75 | 1.03 | 2.01 | 0.36 | 0.70 | 1.44 | 2.41 | 0.95 | 0.351 | **0.184** |

**Median 0.9900** (was 1.0300), mean **1.1185** (was 1.22233), range **0.184 –
2.41**. v22 takes the minimum from v21, which took it from v16 one reading earlier;
the last three readings are the three lowest transfers in the table, in order. Every
future failure line moves with the new minimum: `work/prereg.py` now carries
`"v22": 0.184` in `HISTORY_TRANSFERS`, `0.398006` in `HISTORY_FACES`, anchor
**245.0207**, and `two_most_recent` = (0.351, 0.184), so the v23 grid is built from
this reading without a hand edit.

**The face-effect question, re-asked at n=10 from the data and not carried forward:**

| set | n | ρ | p |
| --- | ---: | ---: | ---: |
| all ten uploads | 10 | **+0.091** | 0.80 |
| without v13's +8.128 face | 9 | **−0.133** | 0.73 |
| the last five only | 5 | −0.600 | 0.29 |

Still **no detectable face effect**, as in §7bu ii at n=9. The v23 band is not
tilted for one.

### iii. What the registration says follows from a failure — quoted, and it is nothing

The desk asked for the registration's own text, with no new rule. Here is every
sentence the registration devotes to failure, grepped from the registered artifacts
rather than recalled:

- The grid heading, §7bu iii and §7bs: *"Rows from the registered grid (median
  transfer, and the score above which the upload **has failed to transfer**)"*.
- The standing distinction, §7bp: *"a move at or below zero is not a weak transfer
  but no transfer at all."*
- The JSON carries two fields and no prose: `transfer_failure_move_below_s`
  **0.1037** and `transfer_failure_score_above` **244.9903**.

**That is the whole of it.** The registration defines a line and names the state on
the far side of it. It registers **no consequence** — no revision, no close, no
change of design, no gate on the next upload. `grep -i` over
`analysis/v22-preregistration-anchored.json` and `work/prereg.py` for a conditional
clause returns one hit, and it is the descriptive note added *after* this reading to
anchor v23. So the honest answer to "what follows" is: **by the registration, a
label and a new minimum in the transfer table, and nothing else.**

This lane will not supply the missing clause. score-feedback rule stands — nothing here is fitted to
a board reading, and no rule is written from n=1 at 0.0304 s. What a breach *should*
oblige is a decision, and decisions of that kind are the desk's and Matt's. The
record now states plainly that the instrument had no answer ready, which is the
useful part: a failure line with no registered consequence is a thermometer, not a
gate, and the next registration can be written knowing that.

### iv. Addendum: the extension months against time distance to the folds

**Desk:** *"for each of v19..v22 the extension months, their TIME distance to each
fold month … beside their cyclic distance to the served months, and the transfer —
so the record can see whether transfer falls with time-adjacency to the folds …
Observation only, score-feedback rule stands, no rule."*

**The five months, per fold.** Time distance is calendar months inside 2025; cyclic
distance is month-of-year distance to the served month, which is what the seasonal
design (47.46) orders by.

| month | to `january` fold | to `july` fold | to `december` fold | **nearest fold** | cyc → 2026-01 | cyc → 2026-07 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-11 | 10 | 4 | **1** | 1 | 2 | 4 |
| 2025-10 | 9 | 3 | **2** | 2 | 3 | 3 |
| 2025-03 | **2** | 4 | 9 | 2 | 2 | 4 |
| 2025-02 | **1** | 5 | 10 | 1 | 1 | 5 |
| 2025-08 | 7 | **1** | 4 | 1 | 5 | 1 |

**Per upload, with the transfer:**

| upload | extension months (per head) | nearest-fold time distances | mean | mean cyc → its served month | transfer |
| --- | --- | --- | ---: | ---: | ---: |
| v19 (= v8) | — none — | — | — | — | 2.41 |
| v20 (= v9) | 2025-11 | 1 | 1.00 | 3.00 | 0.95 |
| v21 (= v10) | 2025-11, 2025-10, 2025-03 | 1, 2, 2 | 1.67 | 3.00 | 0.351 |
| v22 (= v11) 2026-01 head | 2025-11, 2025-10, 2025-03, **2025-02** | 1, 2, 2, **1** | 1.50 | 2.00 | 0.184 |
| v22 (= v11) 2026-07 head | 2025-11, 2025-10, 2025-03, **2025-08** | 1, 2, 2, **1** | 1.50 | 3.00 | 0.184 |

**What the table shows, and what it cannot.**

- **Transfer falls monotonically with the number of extension months**: 0 → 1 → 3 →
  4 months against 2.41 → 0.95 → 0.351 → 0.184. Spearman ρ = **−1.00** on n=4, which
  four ordered points always give and which therefore proves nothing on its own.
- **Time-adjacency to a fold has no contrast to measure.** Every extension month
  ever appended sits **1 or 2 calendar months** from some fold month; none is
  further. So panel size and fold-adjacency move together across all four uploads
  and this table cannot separate them. The mean nearest-fold distance is not even
  monotone (1.00 → 1.67 → 1.50) while the transfer is, so on the four points there
  is no ordering of the two.
- **The one within-upload contrast points the desk's way.** v22's two new months are
  **2025-02** and **2025-08**, and each is time distance **1** from the fold that
  prices its *own* served head (Feb → the `january` fold; Aug → the `july` fold).
  That is the closest fold pairing this programme has appended, and v22 is the
  lowest transfer it has recorded. It is one observation, on one upload, and the
  board reports one number for both heads, so the two heads cannot be scored
  separately from it.
- **A correction to the framing, measured.** The served 2026 months are *not* a year
  from every extension month. Time distances to the served months are 2025-11 → 2 /
  8, 2025-10 → 3 / 9, 2025-03 → 10 / 16, 2025-02 → 11 / 17, 2025-08 → 5 / 11 (mean
  **9.1** months). The phrase holds on average, but 2025-11 — the first month ever
  appended, and the one on the highest extension-era transfer — is only **2** months
  in time from served 2026-01. Any reading that pairs fold-adjacency with a decay in
  transfer has to survive that month.

**No rule, no fit, nothing revised (score-feedback rule).** The table is on the record so the
question can be asked properly when there is something to ask it with: a design in
which panel size and fold-adjacency are varied apart, which no upload so far has
done.

## §7bz — the residual census re-read on the CURRENT bytes (v22), and the unpriced levers

**Desk, recorded rule, 2026-09-10:** *"re-read your residual census (7bd) on the CURRENT bytes
(v22) for the largest unpriced levers on either side — the 148,585 no-clock rows at
v18 bytes and any with-clock lever not on the k-curve — and write them as trade-offs
(kind of work, cost, upside, what it gives up, each priced against the 0.1 s floor
from reads on the record)."*

`work/census_v22.py`, `analysis/census-v22.json`. **No fits, no candidate, no serve**;
the only estimator is `base.shared`, exp-041's self-excluded cross-product, used
exactly as in §7be. Nothing in exp-046 is written; its two vectors are read-only.

### i. What "the current bytes" are, established by bytes and not by assumption

The board's file has two halves and they are on different clocks.

| step | complement rows that moved | class rows that moved |
| --- | ---: | ---: |
| v14 → v15 | 1 | 188,493 |
| v15 → v16 | 1 | 188,481 |
| **v16 → v17** | **69,633** | 3,310 |
| v17 → v18 | **0** | 192,435 |
| v18 → v19 | **0** | 194,680 |
| v19 → v20 | **0** | 194,681 |
| v20 → v21 | **0** | 194,681 |
| v21 → v22 | **0** | 194,681 |

**v17 was the last upload to move a no-clock row, and it was this lane's own**
(`exp-047-queue-lit.parquet`, arm N re-priced on three cells). Five uploads and
1.635 s of board gain later, all **148,585** complement rows still carry the value
this lane left there at 03:33 UTC on 2026-09-09. So the price base for that half is
**p12 plus this lane's own arm-N update**, and the price base for the other half is
**p12 + `d-seasonal-bag10-255-free.npz`**, the vector v22's bytes ship.

**Three things checked before anything was read off:**

1. **The two lanes' class definitions agree exactly on the served side.**
   `occupied_by_exp_046` equals `obs_has_clock & ~noac` on all **344,841** ranking
   rows, 0 disagreements — so "the complement" means one population in both records.
2. **They do not agree on the fold side, and that is reported rather than smoothed.**
   exp-046's fold-side vectors are non-zero on **808 / 744 / 742** rows (v18) and
   **881 / 857 / 742** (v22) that this lane's block calls no-clock; **none** of them
   is noac. The board follows the ranking class, so those rows are priced here at
   p12 — the fold-side clock flags differ by about 0.9 % of surfacefit's class, which
   is a handshake note for the desk and does not move a census cell by more than
   0.2 %.
3. **The row order is proved, not trusted.** `p12 + d-seasonal-bag10` reproduces
   surfacefit's own `rmse_new` on all three folds to **0.00e+00**, and
   `p12 + d-bundle-ungated1299-guard995` reproduces its `rmse_head` the same way.

The arm-N update has **no saved fold-side vector** — `analysis/serve-lit.json` keeps
its before/after instead — so the complement is priced at p12 and the update's size is
carried beside it: it moved the fold-side board **308.5977 → 308.3552**, i.e. it took
**0.157 %** of the complement's squared error. Every cell below is therefore current
to within that, and the three cells it **served** (EDDF/`matched_no_clock`,
LEMD/`matched_no_clock`, LTFM/`unmatched`) are flagged, so a spent cell is never
offered as an unpriced lever.

### ii. The headline: the half nobody has touched now carries five sixths of the board

| fold | rows | rmse at v12 | at v18 | **at v22** |
| --- | ---: | ---: | ---: | ---: |
| july_loo | 190,713 | 283.57 | 273.11 | **271.12** |
| december | 165,677 | 204.20 | 186.68 | **182.42** |
| january | 153,706 | 337.46 | 329.26 | **327.87** |

| fold | side | rows | rms s | share of the fold's e² | **board-weighted** |
| --- | --- | ---: | ---: | ---: | ---: |
| july_loo | complement | 102,618 | 315.17 | 0.7271 | **0.3362** |
| july_loo | with-clock | 88,095 | 208.39 | 0.2729 | 0.1262 |
| january | complement | 84,205 | 427.48 | **0.9313** | **0.5006** |
| january | with-clock | 69,501 | 127.81 | 0.0687 | 0.0369 |
| december | complement | 64,014 | 237.31 | 0.6539 | 0 (no board share) |
| december | with-clock | 101,663 | 137.01 | 0.3461 | 0 |

**The complement is 43.1 % of the board's rows and 83.69 % of the board's squared
error.** At v12 it was **77.83 %** — and the instrument reproduces that figure to four
decimals from the same panel, along with §7bd's three complement rms values (315.17,
237.31, 427.48) unchanged, so the two censuses differ in one input only. The
complement's own residual has not moved; the other half's has, and the share moved
with it. In january the complement is now **93.1 %** of the fold's squared error.

The with-clock half is down to **16.31 %** of the board's squared error, and its
january rms — **127.81 s** — is now *lower than december's complement* and less than a
third of the complement's on the same fold.

### iii. The floor, in the census's own units

recorded rule's floor is **0.1 s** on the board. On a board of 245.0207 that is exactly
**8.1603 × 10⁻⁴** of the board's squared error. Turned around:

| to clear the 0.1 s floor you must remove … | … of its own squared error |
| --- | ---: |
| the whole complement (board 0.8369) | **0.098 %** |
| january's complement (0.5006) | 0.163 % |
| july's complement (0.3362) | 0.243 % |
| the whole with-clock class (0.1631) | **0.500 %** |
| july's with-clock (0.1262) | 0.647 % |
| january's with-clock (0.0369) | 2.21 % |

**One tenth of one per cent of the unpriced half is an upload that clears the floor.**
The same upload on the with-clock half costs five times the work, which is the shape
of the whole re-read in one line.

### iv. The inventory, priced

Reachable share is `base.shared / mean(r²)` on the shipped **airport × 10-minute**
grid: the fraction of a population's squared error a predictor *constant on that cell*
could remove. It is a ceiling for that relation, not a promise. Board seconds combine
both scoring folds at their board weights.

| # | population (current bytes) | fold rows | board share | reachable | **board s at the point** | at the CI low | × the floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **complement body, never served, y ≤ 3600** | 120,471 | 0.2699 | .0674 / .1001 | **+2.612** | +2.165 | **26×** |
| 2 | with-clock body, y ≤ 3600 | 152,414 | 0.1291 | .0622 / .0769 | **+1.046** | +0.844 | **10×** |
| 3 | complement at LFPG, body only | 38,461 | 0.0654 | .1316 / .1131 | +0.987 | +0.761 | 10× |
| 4 | complement tail, y > 3600 | 631 | **0.4670** | .0282 / .0067 | +0.627 | +0.107 | 6× |
| 5 | with-clock tail, y > 3600 | 580 | 0.0340 | .0114 / .0341 | +0.058 | 0 | 0.6× |
| — | *the next extension month (k = 5), for contrast* | — | — | — | *+0.03 … +0.05* | — | *0.3–0.5×* |

(Rows 1 and 3 overlap: LFPG is inside the never-served complement. Row 4 is the same
extreme population §7bd and §7be measured, re-read on the current bytes.)

**The two largest concentrations are unchanged by ten uploads and both sit in the
complement:**

| | fold rows | in the complement | board share |
| --- | ---: | ---: | ---: |
| the single worst january row | 1 | 1 | **0.2261** |
| the 14 rows with labels over 12 hours | 14 | **14** | **0.3781** |
| the 402 `noac` rows | 402 | 402 | **0.1550** |
| the 40 worst rows of the two scoring folds | 40 | 38 | 0.4449 |

### v. The one calibration this lane owns: what arm N realised against what the grid promised

The reachable share is a ceiling. This lane has run that exact relation on this exact
population twice, so the ceiling can be calibrated instead of quoted:

| arm N's served rows | rows | rmse before | after | **removed** |
| --- | ---: | ---: | ---: | ---: |
| july_loo | 39,210 | 241.8708 | 239.6905 | **1.79 %** |
| december | 24,979 | 199.7817 | 198.7732 | **1.01 %** |
| january | 24,954 | 211.5272 | 210.5613 | **0.91 %** |

~~Against a grid ceiling of 7–11 % on the same kind of rows, arm N realised about one
seventh of it ... Applying that same factor to row 1 of the inventory gives ≈ +0.4 s,
still four times the floor.~~ **WRONG — struck, see §7ca vi.** The 7–11 % is the WHOLE
complement's ceiling; against the served cells' OWN ceiling (12.7–20.2 %) the realised
factor is **0.049–0.089**, between a twentieth and a ninth. Projected on the cells'
ranking rows the registered band for lever A is **+0.053 … +0.141 s with a middle row of
+0.096 s** — at the floor, not four times it. The two board readings (1.03×, 0.70×)
stand. **That ratio is the honest discount on every ceiling in the table**, and it is
stated here rather than after a result.

### vi. The levers as trade-offs

Same columns per row, no recommendation, the open option last. Every board figure is
the census ceiling with the §7bz v calibration named beside it; none is a prediction
and nothing here is registered as one (score-feedback rule).

| lever | kind of work | cost | what it can win | what it gives up |
| --- | --- | --- | --- | --- |
| **A. Arm N's relation on the never-served no-clock cells** — the same 24 columns, the same cell hold and cross-fit, on the cells v14 and v17 left alone (LFPG, EGLL, LEBL, LSZH, LEMD; 120,471 fold rows, 0.2699 of the board) | exploit | a worker day of design plus ≈ 2 h of laptop CPU per fold set; no token, no GPU | ceiling +2.61 s [+2.17]; **registered at +0.053 … +0.141 s, middle row +0.096 s** (§7ca vi, projected on ranking rows at the measured realisation) — at the floor, and two to three times the next month on the k-curve | it repeats a relation whose two board readings were 1.03× and 0.70×, and it spends the slot on the grid that §7be already showed is nearly empty on the extreme rows |
| **B. A relation that is not the shipped grid, on the extreme population** — group the 631 tail rows and the 14 twelve-hour labels by something other than when and where they started (stand, carrier, aircraft class, the flight's own prior leg) | analyze | a worker day of design and screening; no fit until a relation clears its own screen; no token | the population is **46.7 %** of the board's squared error on 631 rows; nothing has ever been priced there, and §7be's 0.7 % is a limit of the airport × 10-min grid and not of the rows | it may return no relation at all, and §7bd's detection sweep already found nothing at p ≈ 0.15 in this lane's interface (best lift 2.98×) |
| **C. The with-clock residual the current head still leaves on the shipped grid** — the cell-constant component of exp-046's own residual, which is not another month and not on the k-curve | explore | a worker half-day to measure per airport and per hour, then surfacefit's slot to fit; the fold slot is shared with the 47.59.6 test | ceiling +1.05 s [+0.84], ten times the floor, on a class whose k-curve step is now worth 0.03–0.05 s | it is surfacefit's class and its instrument, so this lane can price it but not serve it; at the calibrated seventh it lands **at** the floor |
| **D. LFPG january's body by hour of day** — the 8.2× separation of squared error with disjoint day-clustered intervals that §7bd addendum §4 already measured, made into a per-cell correction | exploit | half a worker day; the structure is measured, only the correction is unwritten; no token | ceiling +0.99 s over both folds [+0.76]; the structure is already on the record with intervals, so the screen step is skipped | LFPG january's **body** is only 0.0303 of the board — the cell's weight is in the 104 tail rows the same section showed a cell constant cannot reach |
| **E. The 388 `noac` rows** — 402 fold rows at rms 3,407 / 3,509 carrying **15.5 %** of the board's squared error on 0.11 % of the file | (blocked) | none until a rule changes | the largest single board share per row anywhere in the census | **closed by registration, not by measurement**: `assignments/queue.md` rejects "anything on the 388 noac rows (bit-identical to v11 always)" and exp-033's rule stands — no noac row moves. Lifting it is Matt's, and this lane does not price a lever it may not touch |
| **F.** Something else. | | | | |

**What the re-read does not say.** It does not say a lever will pay: every figure in
column four is a ceiling for one relation, discounted once by the only realisation
factor this lane has measured. It does not rank the options. And it registers nothing
— score-feedback rule stands, no amplitude, parameter or rule is fitted to a board reading here, and
the two board readings the complement has produced (1.03×, 0.70×) are quoted as
history and not as a transfer assumption.

**Files.** `work/census_v22.py`, `analysis/census-v22.json`, `logs/census-v22.log`.

## §7ca — LEVER A registered: the never-served no-clock cells, written before any fit

**Desk, recorded rule, 2026-09-10:** *"lever A FIRST — write the never-served no-clock design as a
registered section BEFORE any data: the cell inventory with why v14/v17 skipped each
cell, the hold and cross-fit exactly as v17 (24 columns), LFPG january hour-of-day
structure (7bd addendum 4) as the first cell, and a predicted band from your own
realised seventh. Its fits take the laptop slot AFTER surfacefit registered alone-table
cells (one fit at a time; I send the GO)."*

`work/leverA_design.py`, `analysis/leverA-design.json`. **No fit runs here.** Three
things are measured — the per-cell ceiling, this lane's own realisation factor, and
LFPG's hour structure on v22's bytes — and the band that follows is registered before
the GO and is not revised by its outcome.

### i. A defect in §7bz, found before anything was registered

**§7bz split the complement on the wrong grain.** It used the block's `reach_class`
(`matched_no_clock` / `unmatched`); the grain both serve records use is
`code/arms.py::reached` — **`No_n_n_cell > 0`**, "the network heard at least one other
aircraft in the row's cell" — named `<airport>/reached` or `/unreached` by
`code/serve.py::cell_of`. **What caught it**: the cell table did not reconcile with the
serve records' own ranking-row counts. On the corrected grain it reconciles cell by
cell — LTFM/unreached 47,581 measured against 47,577 recorded, EDDM/unreached 45
against 45, EHAM/unreached 81 against 81 — and the whole residual is the v16 → v25
class drift of 5,247 rows, in the right direction on every cell.

| §7bz figure | as published | corrected |
| --- | ---: | ---: |
| complement already served by arm N, board share | 0.0955 | **0.1114** |
| complement never served, board share | 0.7414 | **0.7255** |
| never-served body, fold rows on the scoring folds | 137,398 | **123,687** |
| never-served body ceiling (inventory row 1) | +2.819 s | **+2.612 s** |

Every other §7bz number is on the two halves or the whole complement and is untouched.
The §7bz lines that carried the two changed figures are edited above with a pointer here.

### ii. The fold is not the board, at the cell grain — so the band is projected

The reach grain moves with the harvest. Among no-clock rows:

| month | no-clock rows | reached | share |
| --- | ---: | ---: | ---: |
| 2025-01 (`january` fold) | 84,202 | 14,892 | **17.7 %** |
| 2025-07 (`july_loo` fold) | 102,343 | 26,032 | 25.4 % |
| 2025-12 (`december` fold) | 63,976 | 21,866 | 34.2 % |
| **2026-01 (served)** | 62,552 | 21,057 | **33.7 %** |
| **2026-07 (served)** | 88,451 | 26,270 | 29.7 % |

January 2025 is **half** as reached as the month it proxies. So a cell's fold weight is
not its board weight, and two cells make the point at once:

* `EGLL/unreached` — **15,509** july fold rows and **185** ranking rows;
* `LIRF/unreached` — **152** july fold rows and **17,730** ranking rows.

**Every board figure below is therefore projected onto each cell's own ranking rows**
through its served month's fold (2026-01 ← `january`, 2026-07 ← `july_loo`), the way
v17's record already priced its cells with `gain_per_row_s2 × ranking_rows`, and never
read off the fold-side share. On the seven target cells the projection is worth
+1.5439 s against +1.9573 s read off the folds — a 21 % overstatement if it were skipped.

### iii. The cell inventory, and why v14 and v17 skipped each cell

| cell | ranking rows | jul / jan / dec fold rows | v14 and v17 held it because | now | projected board s |
| --- | ---: | ---: | --- | --- | ---: |
| `LFPG/unreached` | 32298 | 19963 / 17900 / 15656 | jul interval [-388.14, 47.66] not clear above; jan interval [-21.5, 684.42] not clear above; dec interval [-123.33, 764.14] not clear above | **target** | **+0.0851** |
| `EGLL/reached` | 5728 | 3933 / 9165 / 9609 | jul interval [-1360.95, 381.06] not clear above; jan interval [-712.04, 841.46] not clear above; dec interval [-7375.32, -2221.88] not clear above | **target** | **+0.0096** |
| `LEBL/reached` | 3089 | 2534 / 906 / 1071 | jul supported; jan supported; dec interval [-22.31, 3800.39] not clear above | **target** | **+0.0077** |
| `LSZH/reached` | 1790 | 1684 / 893 / 598 | jul interval [-465.91, 2538.55] not clear above; jan interval [-1303.81, 2382.24] not clear above; dec interval [-520.81, 2860.83] not clear above | **target** | **+0.0041** |
| `LEMD/unreached` | 1306 | 5297 / 16509 / 47 | jul interval [-82.14, 627.02] not clear above; jan dark; dec 47 rows | **target** | **+0.0027** |
| `EHAM/reached` | 2509 | 1270 / 628 / 1157 | jul supported; jan interval [-469.66, 5749.17] not clear above; dec interval [-3669.75, -628.15] not clear above | **target** | **+0.0019** |
| `LTFM/unreached` | 47581 | 25171 / 21368 / 22609 | jul supported; jan supported; dec supported | served at v14/v17 | — |
| `EDDF/reached` | 14049 | 4065 / 3159 / 1415 | jul supported; jan supported; dec supported | served at v14/v17 | — |
| `LEMD/reached` | 8517 | 9373 / 0 / 750 | jul supported; jan dark; dec supported | served at v14/v17 | — |
| `EGLL/unreached` | 185 | 15509 / 1581 / 1378 | jul interval [-384.02, 158.59] not clear above; jan interval [-1624.38, -34.64] not clear above; dec interval [-688.86, 1023.75] not clear above | **target** | **+0.0005** |
| `LIRF/reached` | 3457 | 1934 / 0 / 3917 | jul interval [-3663.86, 6690.26] not clear above; jan dark; dec interval [-1828.56, 1511.12] not clear above | held: under the 500-row bar | — |
| `LIRF/unreached` | 17730 | 152 / 11317 / 2183 | jul 141 rows; jan dark; dec interval [-673.99, 2004.4] not clear above | held: under the 500-row bar | — |
| `EDDM/unreached` | 45 | 9820 / 274 / 38 | jul supported; jan 274 rows; dec 39 rows | held: under the 500-row bar | — |
| `LFPG/reached` | 5228 | 1268 / 0 / 2974 | jul interval [-1269.53, 778.57] not clear above; jan 0 rows; dec interval [-1192.86, 464.98] not clear above | held: under the 500-row bar | — |
| `EDDM/reached` | 922 | 225 / 144 / 400 | jul 230 rows; jan 149 rows; dec 408 rows | held: under the 500-row bar | — |
| `LEBL/unreached` | 257 | 310 / 120 / 94 | jul 321 rows; jan 120 rows; dec 99 rows | held: under the 500-row bar | — |
| `EDDF/unreached` | 3692 | 42 / 128 / 37 | jul 52 rows; jan 155 rows; dec 43 rows | held: under the 500-row bar | — |
| `LSZH/unreached` | 121 | 55 / 97 / 50 | jul 60 rows; jan 98 rows; dec 50 rows | held: under the 500-row bar | — |
| `EHAM/unreached` | 81 | 13 / 16 / 31 | jul 13 rows; jan 16 rows; dec 32 rows | held: under the 500-row bar | — |

(Fold-row counts are this lane's current class; v17's own counts differ by the class
drift, e.g. LEBL/unreached 310 here against "321 fold rows" there. `LIRF/reached`,
`LIRF/no_takeoff`, `LSZH/strict` and the other tiny `reach_class` cells of §7bz do not
appear as cells at this grain — they were an artefact of the wrong split.)

**Read across the "because" column and the pattern is one thing, not many:** of the
sixteen cells v17 held, **nine** were held for having under 500 fold rows somewhere and
**seven** for an interval that was not clear of zero above. Not one was held because its
point estimate was negative on the scoring folds — `LFPG/unreached` january was
**[−21.5, +684.42]**, positive at the point and open only at the bottom. **The hold has
been a statement about interval width, not about sign.**

### iv. The instrument: exactly v17's, with one thing changed

**Unchanged, quoted from `analysis/serve-lit.json`:**

* **The 24 columns.** `Nf_n_anom_cell`, `Nf_n_anom_rwy_30m`, `Nf_n_anom_rwy_cell`,
  `Nf_n_n_cell`, `Nf_n_n_rwy_30m`, `Nf_n_n_rwy_cell`, `Nf_n_res_apt_30m`,
  `Nf_n_res_cell`, `Nf_n_res_rwy_30m`, `Nf_n_res_rwy_60m`, `Nf_n_res_rwy_cell`,
  `Nf_n_res_rwy_minus_apt`, and the twelve `No_` columns with the same names.
* **The learner** is `ridge`, **the amplitude is 1.0** — *"full amplitude; no scale is
  fitted on anything the candidate is scored on"*.
* **The cross-fit**, verbatim: *"cross-fitted by fold: each fold's d comes from a head
  fitted on the other two only. December is not a ranking-share fold and is carried
  beside the board, never inside it."*
* **The hold**, verbatim: *"desk 2026-09-09, the LFPG rule at the reach grain: a cell
  whose own fold interval is not clear of zero above, or which is priced on fewer than
  500 rows in any fold, is held at the base."* It is re-applied cell by cell at fit
  time. **Cells may fail again.**

**Changed, and it is the only thing: the fitting panel.** v17's arm N was fitted on
**510,096** rows — the three folds and nothing else. Since then this lane has built the
186-column block on six more months, and recorded rule names the seventh: *"2025-04 lands as a
data asset."*

| | rows | multiple of v17 |
| --- | ---: | ---: |
| the three folds (v17's whole panel) | 510,096 | 1.00× |
| + 2025-02, 03, 06, 08, 10, 11 (built, base6 targets) | 1,540,606 | **3.02×** |
| + 2025-04 when its chain lands | 1,715,894 | **3.36×** |

The targets on the extension months are the six-member `base6` residual, exactly as
exp-046's extension uses them, and 2025-05 and 2025-09 stay out (recorded rule). **This is the
whole of the design's claim**: the same relation, the same hold, the same cross-fit, on
a panel three times the size — because every hold above was an interval width, and
interval width falls with the panel.

### v. LFPG/unreached first, and what its hour structure does and does not license

**The structure re-read on v22's bytes**, because a first cell may not rest on a v12 read:

| fold | body rows | rms range | top / bottom | hours |
| --- | ---: | --- | ---: | --- |
| `january` | 17,796 | 165.6 – 474.8 s | **2.87×** | h4 → h17 |
| `july_loo` | 19,932 | 158.2 – 269.0 s | 1.70× | h1 → h22 |
| `december` | 15,627 | 145.8 – 253.3 s | 1.74× | h0 → h17 |

2.87× in rms is **8.24× in squared error**, which is §7bd addendum 4's **8.22×** — the
same read, unchanged by ten uploads, because those uploads never touched these rows.

**And a measured limit on what it licenses.** The hour is **nested inside the shipped
10-minute bin**: adding the hour as a third key to `base.cellcode` leaves the pair count
and the ceiling identical on all three folds — january 48,144 pairs and **+0.1131**
either way, july 61,814 and +0.1265, december 46,578 and +0.1259. **A cell-constant
predictor on the shipped grid can already express every hour effect there is**, so this
design adds no hour feature and claims no gain from one. What the structure licenses is
narrower and is why LFPG goes first: the cell's error is concentrated where the arm can
see it, the cell is **32,298 ranking rows**, and it carries **76 %** of the design's
whole projected value.

### vi. The registered band, before the GO

Three measured factors, in order, none of them chosen:

**1. The ceiling.** The seven target cells' bodies, `base.shared` on the shipped grid,
projected on their ranking rows: **+1.5439 s**.

**2. The realisation factor — this lane's own, against those cells' own ceiling.**

| arm N's served cells | rows | rmse before | after | removed | **ceiling on the same rows** | **factor** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `july_loo` | 39,210 | 241.8708 | 239.6905 | 1.795 % | 20.21 % | **0.0888** |
| `december` | 24,979 | 199.7817 | 198.7732 | 1.007 % | 12.68 % | **0.0772** |
| `january` | 24,954 | 211.5272 | 210.5613 | 0.911 % | 18.64 % | **0.0489** |

**Not a seventh — between a twentieth and a ninth**, mean **0.0688**. (§7bz v compared
the same realised figures against the *whole complement's* 7–11 % ceiling rather than
these cells' own 12.7–20.2 %; that is what made it read as a seventh.) The factor is not
bounded by 1 in principle — a ridge on 24 columns is not a cell constant and can use
within-cell variation — so it is a measurement, not an identity.

**3. The transfer.** This lane's only two board readings on this population: **1.03×**
(v14) and **0.70×** (v17).

| row | fold-side | **board** |
| --- | ---: | ---: |
| the ceiling, if the arm were a perfect cell constant | +1.5439 | +1.08 … +1.59 |
| at the lowest realisation (0.0489) | +0.0755 | **+0.0528** |
| **at the mean realisation (0.0688), mean transfer** | **+0.1117** | **+0.0966** |
| at the highest realisation (0.0888) | +0.1371 | **+0.1412** |

**THE REGISTERED BAND IS +0.0528 … +0.1412 s, and its middle row is +0.0966 s — under
the 0.1 s floor.** That is stated before the GO and not after: on the record this lane
owns, lever A is a floor-height lever, not the four-times-the-floor lever §7bz read off
the undiscounted ceiling. **Failure** is a move below **+0.0528 s**; a move at or below
zero is not a weak transfer but no transfer at all. Rank is not at risk in any row (lead
15.91 s).

**The one-sided reading the panel change licenses, registered so the fit is informative
either way.** The factor above was measured on an arm fitted on the folds alone. This
design fits on 3.02–3.36× that panel. If the panel does nothing, the outcome falls in
the band. **An outcome above +0.1412 s is evidence that the panel raised the realisation
factor** — a measured fact about panel size on this population, which nothing on the
record yet supplies. Neither branch revises the band (score-feedback rule).

### vii. The order of work, and the largest thing this design cannot reach

| order | cell | ranking rows | projected board s at the mean |
| ---: | --- | ---: | ---: |
| 1 | `LFPG/unreached` | 32,298 | **+0.0851** |
| 2 | `EGLL/reached` | 5,728 | +0.0096 |
| 3 | `LEBL/reached` | 3,089 | +0.0077 |
| 4 | `LSZH/reached` | 1,790 | +0.0041 |
| 5 | `LEMD/unreached` | 1,306 | +0.0027 |
| 6 | `EHAM/reached` | 2,509 | +0.0019 |
| 7 | `EGLL/unreached` | 185 | +0.0005 |

One fit at a time, on the laptop slot **after** surfacefit's registered alone-table
cells, on the desk's GO. Cells 5–7 are worth **+0.0051 s together**, a twentieth of the
floor; they are last and the record will say whether they were run at all.

**And the cell the registered rule keeps out of reach.** `LIRF/unreached` is **17,730
ranking rows — 15,795 of them in served 2026-07** — and it has **152** july fold rows
and was dark on january at v17. Priced on its january read it would be worth
**+0.0416 s**, which alone lifts the middle row from +0.0966 to **+0.138 s** and over
the floor. **It is held by the rule quoted in §7ca iv and this section does not write a
new one.** It is recorded here because the design's whole shortfall against the floor is
one cell that the instrument cannot see and the board can.

**Files.** `work/leverA_design.py`, `analysis/leverA-design.json`, `logs/leverA-design.log`.

## §7cb — LEVER B screened: no relation off the grid clears, and the residue is two rows

**Desk, recorded rule part 2:** *"In parallel at the design level, lever B: screen the 631-row tail
and the 14 twelve-hour labels for a relation off the grid; no fit until the screen
clears."*

`work/leverB_screen.py`, `analysis/leverB-screen.json`. **The bar was committed before
the read** (`ba90bd2`), in the file's own docstring, and is quoted in §7cb ii. No fit, no
candidate, no serve; the instrument is `base.shared` exactly as §7be used it.

### i. The population, on the current bytes

| population | fold | rows | rms s | mean signed s | board |
| --- | --- | ---: | ---: | ---: | ---: |
| complement tail, y > 3600 | `july_loo` | 401 | 2,644.2 | +1,584.1 | 0.0925 |
| complement tail, y > 3600 | `january` | 230 | **7,074.4** | +1,953.1 | **0.3745** |
| complement tail, y > 3600 | `december` | 186 | 1,852.2 | +1,276.2 | 0 |
| all tail, y > 3600 | `july_loo` | 656 | 2,382.0 | +1,522.9 | 0.1228 |
| all tail, y > 3600 | `january` | 325 | 5,980.7 | +1,570.4 | 0.3782 |

**631 complement rows on the two scoring folds carry 0.4670 of the board's squared
error**, unchanged by ten uploads because none of them touched these rows.

### ii. The bar, as committed before the read

> **(a)** its shared-component interval is clear of zero ABOVE on at least two folds,
> one of them a scoring fold — the two-fold form the desk's 2026-09-09 lit rule already
> uses for a cell dark on one fold; **(b)** its value PROJECTED on the tail's own ranking
> rows, at §7ca vi's measured realisation range 0.0489–0.0888 and this lane's board
> transfer 0.70–1.03×, has a MIDDLE ROW at or above the 0.1 s floor; **(c)** it beats the
> shipped grid on the same rows — a relation that only reproduces airport × 10-minute is
> not a relation off the grid.

With the multiplicity registered beside it: *"thirteen relations × three folds is 39
reads. A point estimate chosen from 39 tries is a CANDIDATE and not a pass, and its bar
is a pre-registered re-run."*

### iii. The result: thirteen relations, none clears

Reachable share on the complement tail, point and 95 % day-clustered interval:

| relation (off the grid) | `july_loo` | `january` | `december` | mid, board s | clears |
| --- | ---: | ---: | ---: | ---: | --- |
| airport × AOBT−LOBT decile | −0.008 [−0.012, −0.004] | +0.000 [−0.000, +0.001] | **+0.003** [+0.002, +0.007] | +0.0019 | no |
| operator | −0.005 [−0.007, −0.003] | +0.000 [−0.000, +0.001] | +0.000 [−0.000, +0.002] | +0.0012 | no |
| airport × stand-distance decile | +0.000 [−0.004, +0.004] | −0.000 | **+0.002** [+0.000, +0.007] | +0.0004 | no |
| airport × receipt-rate decile | +0.000 [−0.002, +0.003] | −0.000 | **+0.002** [+0.000, +0.005] | +0.0004 | no |
| airport × operator | −0.003 [−0.006, −0.000] | +0.000 | +0.001 [−0.000, +0.007] | +0.0001 | no |
| airport alone (coarser than the grid) | −0.002 [−0.003, +0.000] | −0.000 | **+0.003** [+0.001, +0.006] | +0.0000 | no |
| airport × runway | −0.002 [−0.003, −0.000] | −0.000 | **+0.005** [+0.002, +0.008] | +0.0000 | no |
| airport × stand | −0.001 [−0.003, +0.001] | −0.000 | **+0.008** [+0.004, +0.014] | +0.0000 | no |
| airport × calendar day | −0.000 [−0.005, +0.005] | −0.000 | +0.005 [−0.000, +0.009] | +0.0000 | no |
| destination | −0.001 | −0.000 | +0.000 | +0.0000 | no |
| aircraft type | −0.002 [−0.003, −0.001] | −0.000 | −0.000 | +0.0000 | no |
| wake category | −0.001 | −0.000 | −0.001 | +0.0000 | no |
| market segment | −0.001 | −0.000 | −0.001 | +0.0000 | no |
| **the shipped grid, for contrast** | **+0.028** [+0.002, +0.059] | **+0.007** [+0.001, +0.016] | +0.031 [−0.010, +0.060] | **+0.1100** | *(the baseline)* |

**Not one relation off the grid passes clause (a)**, and the failure is not marginal: on
the two folds that score, twelve of thirteen are **negative at the point** on july and
essentially zero on january. Every positive that appears is on **december**, which has no
board share. The interval widths are not the problem either — `airport × stand` has
32,458 pairs on july against the grid's 1,030 and still reads −0.001.

**The reading, and it is a real answer to §7be's open question.** §7be wrote that the
grid *"is nearly empty there and a different relation — one that groups extreme
taxi-outs by something other than when and where they started — is untested, not
refuted."* It is now tested on thirteen such relations, including every categorical the
movement table carries and every continuous the block carries. **Extreme taxi-outs do
not group by who flies them, what flies, where it goes, which stand, which runway, how
late the flight already is, or how well the harvest sees the airport.** They group, if
at all, by *when and where they started* — which is exactly the relation that was
already shipped.

### iv. And the shipped grid, on the tail, prices at the floor

The one relation that clears all three clauses is the **baseline**: airport × 10-minute
on the complement tail, clear of zero above on **both scoring folds** (+0.028 [+0.002,
+0.059] july, +0.007 [+0.001, +0.016] january), projecting to **+0.0632 … +0.1690 s with
a middle row of +0.1100 s**. That is above the floor, on the same discount §7ca vi
registered — and it is the same instrument, the same hold and the same 24 columns as
lever A, applied to the tail rows of the same cells rather than to their bodies.

**This is not a new lever and this section does not register one.** It is a measurement
that lever A's design excluded the tail (its ceiling is read on `y ≤ 3600` bodies) and
that the excluded part is worth about as much again. Whether the tail rows enter lever
A's arm is a design change to §7ca, and §7ca is registered; changing it is the desk's.

### v. The sixteen twelve-hour labels: not screened, and why — with what they are

They are not screened by this instrument for an arithmetic reason registered with the
bar: **16 rows in 510,096 is a base rate of 3.1 × 10⁻⁵**, and §7bd addendum 3's 2,162
prediction-time screens reached a best lift of 2.98× at p ≈ 0.15 against a base rate a
hundred times larger. No screen in this lane's interface can clear at this rate. So they
are reported as facts:

| fold | airport | label | prediction | residual | type | class |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `january` | **LFPG** | 84,240 | **870.4** | **+83,369.6** | A319 | no-clock |
| `january` | **LFPG** | 58,206 | **989.2** | **+57,216.8** | A320 | no-clock |
| `july_loo` | LIRF | 87,186 | 62,769.3 | +24,416.7 | — | **noac** |
| `january` | LIRF | 87,177 | 64,458.4 | +22,718.6 | — | **noac** |
| `july_loo` | LIRF | 87,361 | 74,319.9 | +13,041.1 | — | **noac** |
| `july_loo` | LIRF | 87,170 | 95,220.8 | −8,050.8 | — | **noac** |
| … ten more, all LIRF | | 43,863 – 88,132 | tracked to within 4,200 s | | | 9 `noac`, 1 with a clock |

**The sixteen are two populations, and the split is exactly the split between two
levers.** **Thirteen of sixteen are `noac`** — all LIRF, aircraft type absent — and the
base already **tracks** them: it predicts tens of thousands of seconds for labels near
87,000 and its worst LIRF miss is +24,417 on a label of 87,186. Those rows are lever E,
closed by registration.

**The whole board-facing problem is the two LFPG january rows.** The base predicts
**870 s** against a label of **84,240 s** and **989 s** against **58,206 s** — and both
rows carry `route_km` and `stand_dist_km` **NaN**: the harvest never saw either flight.
§7bz measured that the single worst january row alone is **22.6 %** of the board's
squared error and the worst five are 35.5 %.

**So the extreme-label question reduces to two rows that no relation can group**, and
what to do about them — clip, winsorise, exclude, accept — is the candidate decision
§7bd addendum already put to the desk. This section adds one fact to it: they are
no-clock rows at LFPG in january with no harvest features at all, so they sit inside
lever A's **first** cell and no cell-constant arm fitted on 17,900 rows will move a
+83,370 s residual on two of them.

**Files.** `work/leverB_screen.py` (bar committed at `ba90bd2` before the read),
`analysis/leverB-screen.json`, `logs/leverB-screen.log`.

## §7ca ii — LEVER A superseded before any data: the arm goes on ALL rows of the cells

**Desk, 2026-09-10:** *"7ca is registered but its data does not exist yet, so it may be
SUPERSEDED before any fit — write 7ca ii: lever A arm on ALL rows of the never-served
cells (bodies AND the tail rows of the same cells), same instrument, same hold, same 24
columns, ceiling and band re-read on all rows … Name the two LFPG january rows (84,240 /
58,206, harvest NaN) as the registered risk: no exclusion by label, no cap; the gate tail
clause (y > 3600 not worse by 1 s RMSE) is the guard, and a fired clause voids the arm,
not the clause."*

`work/leverA_allrows.py`, `analysis/leverA-allrows.json`. **No fit.** §7ca vi's band is
superseded by §7ca ii vi below; everything else in §7ca stands.

**Why superseding is right and not a patch.** §7ca read its ceiling on `y ≤ 3600` bodies
because §7be had measured the tail as barely reachable *on the shipped grid*. §7cb then
screened thirteen relations off that grid and found the grid is the **only** one that
clears on the tail. So excluding the tail was excluding the one population the same
instrument can still price, on a premise §7cb had just removed. **Unchanged**: the 24
columns, ridge at amplitude 1.0, the cross-fit, the hold and its 500-row bar, the seven
target cells, the 3.36× fitting panel. **Changed**: the arm is fitted and priced on
**all** rows of those cells.

### i. What the tail does to each cell — and it is not one thing

| cell | fold | rows | tail rows | tail's share of the cell's e² | reachable, **all rows** | reachable, body |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `LFPG/unreached` | `july_loo` | 19,963 | 31 | 14.9 % | +0.1061 [+0.080, +0.138] | +0.1265 |
| `LFPG/unreached` | `january` | 17,900 | **104** | **91.8 %** | **+0.0141** [+0.009, +0.021] | +0.1131 |
| `EGLL/reached` | `july_loo` | 3,933 | 36 | 41.9 % | +0.0942 [+0.044, +0.165] | +0.0812 |
| `EGLL/reached` | `january` | 9,165 | 34 | 12.9 % | +0.0731 [+0.049, +0.099] | +0.0848 |
| `EGLL/unreached` | `july_loo` | 15,509 | 125 | 35.2 % | +0.0551 [+0.034, +0.074] | +0.0842 |
| `EGLL/unreached` | `january` | 1,581 | 1 | 0.5 % | +0.1501 [+0.099, +0.230] | +0.1495 |
| `LEBL/reached` | `july_loo` / `january` | 2,534 / 906 | **0** / **0** | 0 | +0.1251 / +0.0868 | identical |
| `LEMD/unreached` | `july_loo` / `january` | 5,297 / 16,509 | **0** / **0** | 0 | +0.1581 / +0.1662 | identical |
| `LSZH/reached` | `july_loo` / `january` | 1,684 / 893 | 3 / 3 | 49.0 / 45.5 % | +0.0613 / +0.0295 | +0.0972 / +0.0490 |
| `EHAM/reached` | `july_loo` / `january` | 1,270 / 628 | 1 / 5 | 36.3 / 77.9 % | +0.0167 / +0.0163 | +0.0292 / +0.0463 |

**Two of the seven cells have no tail rows at all**, and in every cell that has them the
**reachable share falls** when they are included — most violently at `LFPG/unreached`
january, from **+0.1131 to +0.0141**, because its 104 tail rows are **91.8 %** of the
cell's squared error and are the rows §7be measured at 0.67 % reachable. Including the
tail multiplies a cell's squared error and divides its reachable fraction; whether that
is a gain depends on which moves further, and it differs cell by cell.

### ii. The combined band, stated before the data — and it is not the sum

**The desk's arithmetic was body +0.0966 and tail +0.1100 combining above the floor. The
measured combination is smaller, and here is exactly why.** §7cb's +0.1100 was read on
the **whole complement tail** — all 631 scoring-fold rows, including the tails of
`LTFM/unreached` and `EDDF/reached`, which arm N already serves, and of `LIRF/unreached`,
which the desk's own ruling keeps out. **The seven target cells hold 343 of those 631
rows.** And of the squared error those 343 carry, the largest block is the LFPG january
104, whose reachable share is the lowest in the census.

| row | fold-side | **board** |
| --- | ---: | ---: |
| the ceiling on all rows, projected on ranking rows | +1.8296 | — |
| *(the ceiling on bodies only, §7ca)* | *+1.5439* | — |
| at the lowest realisation (0.0489) | +0.0894 | **+0.0626** |
| **at the mean realisation (0.0716), mean transfer** | **+0.1310** | **+0.1134** |
| at the highest realisation (0.0888) | +0.1625 | **+0.1674** |

**THE REGISTERED BAND IS +0.0626 … +0.1674 s AND ITS MIDDLE ROW IS +0.1134 s.** It
**clears the 0.1 s floor at the middle row**, which §7ca's body-only design did not
(+0.0957 on the same arithmetic). It does **not** clear at the bottom. The tail rows are
worth **+0.0177 s** at the middle row — the desk's direction is right and the crossing is
real, but the margin is a sixth of what adding the two published middles would suggest,
and that is stated before the fit rather than found after it.

**Failure** is a move below **+0.0626 s**; a move at or below zero is not a weak transfer
but no transfer at all. §7ca vi's one-sided reading carries over unchanged: an outcome
above **+0.1674 s** is evidence the 3.36× panel raised the realisation factor.

| order | cell | board s at the middle row | note |
| ---: | --- | ---: | --- |
| 1 | `LFPG/unreached` | **+0.0882** | **78 %** of the whole design |
| 2 | `EGLL/reached` | +0.0093 | |
| 3 | `LEBL/reached` | +0.0066 | no tail rows |
| 4 | `LSZH/reached` | +0.0042 | |
| 5 | `LEMD/unreached` | +0.0024 | **47 december rows — fails the hold's 500-row bar** |
| 6 | `EHAM/reached` | +0.0023 | |
| 7 | `EGLL/unreached` | +0.0004 | |

The hold is applied cell by cell at fit time and **cells will drop**: `LEMD/unreached` has
47 december rows and fails the bar as the rule is written, and any cell whose own fold
interval is not clear of zero above stays at the base as it did at v14 and v17. The band
is for the cells that pass. **One cell is 78 % of it**, so in practice this design is
`LFPG/unreached` and six cells worth +0.0252 s together.

### iii. The registered risk: two rows, named, and neither excluded nor capped

**The desk's ruling, recorded verbatim as the rule this design runs under:** *"no
exclusion by label, no cap; the gate tail clause (y > 3600 not worse by 1 s RMSE) is the
guard, and a fired clause voids the arm, not the clause."*

| fold | cell | label | prediction | residual | harvest features |
| --- | --- | ---: | ---: | ---: | --- |
| `january` | **`LFPG/unreached`** | 84,240 s | **870.4** | **+83,369.6** | `route_km` and `stand_dist_km` **NaN** |
| `january` | **`LFPG/unreached`** | 58,206 s | **989.2** | **+57,216.8** | both **NaN** |

Both sit in **cell 1**, the cell that carries 78 % of the design. Both are rows the
harvest never saw, so the 24 columns the arm reads are the only thing it has on them, and
§7bz measured that the first of the two alone is **22.6 %** of the board's squared error.
**They are in the fitting panel and in the priced population, by the ruling above.** The
risk this registers is not that the arm fails to fix them — no cell constant moves a
+83,370 s residual on two rows of 17,900 — but that a ridge fitted with them in the
target can be pulled by them.

**The guard, and what it can cost.** `assignments/k2climb.md` line 24: *"tail rows
(> 3600 s) not worse by more than 1 s."* The target cells' tail rows and the clause's
full allowance:

| fold | tail rows in the target cells | their rmse | 1 s of it, per row | 12-hour rows inside |
| --- | ---: | ---: | ---: | ---: |
| `july_loo` | 196 | 2,055.0 s | 4,109 s² | 0 |
| `january` | **147** | **8,533.4 s** | **17,066 s²** | **2** |
| `december` | 59 | 2,002.2 s | 4,003 s² | 0 |

**Saturated on both scoring folds, the clause's whole allowance moves the fold-side board
by +0.0161 s — 16 % of the floor.** So the guard binds long before it could eat the
design's +0.1134 s, which is what makes it a guard and not a loophole. And per the
ruling: **if it fires, the arm is void, not the clause.** No amplitude is shrunk to make
a clause pass, no row is dropped by its label, and no cap is applied — the record of a
fired clause is a dead arm and a design question, in that order.

### iv. What this section does not change

§7ca's inventory, its reasons, its instrument, its panel and its slot order all stand;
so does the desk's ruling that `LIRF/unreached` is out. §7cb stands as written — its
+0.1100 was a correct read of a different population, and §7ca ii vi says so rather than
quietly reusing the number. Nothing is fitted here and no board reading is revised (score-feedback rule).

**Files.** `work/leverA_allrows.py`, `analysis/leverA-allrows.json`,
`logs/leverA-allrows.log`.

### v. The ceiling is an ORACLE — checked, and why the band is still board-reachable

**Desk, from surfacefit 47.66 (lever C killed at stage 2):** *"a cell in the shipped grid
is a window in ABSOLUTE UTC, so 2025 cells and 2026 cells are disjoint and 0 of 689,534
ranking ids appear in any training file — any predictor that estimates a cell constant
from LABELLED neighbours in the same cell is a fold-side quantity the board cannot reach.
`base.shared` ceilings are that kind of oracle."*

**Checked here independently, not accepted on report.** `code/block.py` builds
`cell = floor(MVT_TIME / 600)` prefixed by the airport, in absolute epoch seconds, and
`code/base.py` does the same with `BIN = 600`. Measured:

| | cells | shares a cell with any other month |
| --- | ---: | ---: |
| 2025-01 / 2025-07 / 2025-12 | 33,383 / 35,561 / 33,669 | **0** in all ten pairs |
| 2026-01 / 2026-07 | 33,527 / 35,498 | **0** |
| **all 2025 fold cells vs all 2026 served cells** | 102,613 vs 69,025 | **intersection 0** |

and **0 of 689,534 ranking ids** appear in 2025-01, 2025-07 or 2025-12 — the desk's own
figure, reproduced. **The finding is right: `base.shared` is an oracle.** It reads
`E[rᵢrⱼ]` over labelled pairs inside a cell, and no 2026 cell contains a labelled row, so
nothing on the board can realise it.

**Two things follow, and both were already true of the design — but neither was stated.**

**1. The oracle cancels in level; it survives only as a ratio between cells.** The band is
`ceiling × ρ × transfer` where `ρ = realised ÷ ceiling`, both measured on arm N's own
served cells. Substituting, the predicted per-row removal in a target cell is

> (its mean squared error) × (**the fraction arm N actually removed**, 0.91–1.79 %)
> × (**the ratio of the two cells' oracle ceilings**) × (the board transfer, 0.70–1.03×)

so the oracle never enters as a level. The band's level is anchored on a number a
features-only cross-fitted ridge achieved and the board paid for twice. The assumption
the oracle ratio carries — and it is the design's real assumption, now written down — is
that the arm captures the **same fraction** of the oracle in the target cells as in the
served ones. Its direction is mostly conservative here: the target cells' all-row
ceilings (+0.014 … +0.166) sit mostly **below** the served cells' (12.7–20.2 %).

**2. The fold cross-fit already holds out whole cells, by arithmetic.** A month boundary
is midnight UTC, 86,400 s is 144 whole 600-s bins, so **no cell can straddle a month**,
and a fold is a month. Therefore *"each fold's `d` comes from a head fitted on the other
two only"* — v17's cross-fit, unchanged — means every cell an arm is scored on was
**entirely absent** from the panel that fitted it. The measurement above confirms it with
no shared cell in any pair. This is why v17's two board readings transferred at **1.03×
and 0.70×** rather than collapsing: the arm never had a same-cell label at prediction
time, on the folds or on the board.

**And the instrument was checked for same-cell labels, as the desk required.** The 24
columns read the neighbour's taxi as `MVT_TIME − AOBT_3_flt` (the file clock) or
`MVT_TIME − obs_offblock` (the adsb.lol observation), never `TAXITIME_SEC_mvt`, and the
residual form subtracts **v11's own served prediction**, which `code/block.py` states in
terms: *"our own model output, not a label, and it exists for the ranking months because
it is the file we serve."* Measured on the ranking blocks: `Nf_n_res_cell` **97.0 % /
98.5 %** non-null, `No_n_res_cell` 69.6 % / 65.9 %, both count columns **100 %**. **No
part of the instrument uses a same-cell label, so nothing is removed before the fits.**

**Registered as a pre-fit assertion**, so this cannot rot: the arm run asserts, before it
prices anything, that **the held-out fold shares zero cells with its fitting folds** and
that every fitted column is one of the registered 24. A run that cannot assert both stops
and reports rather than serving.

**The band of §7ca ii ii is unchanged by this section** — +0.0626 … +0.1674 s, middle row
+0.1134 s — because nothing in it was ever the oracle's level. What changes is that the
record now says so before the fit instead of being asked afterwards.

## §7bv v — 2025-04 landed, and the month programme is closed

**2025-04 ran 10:16:14 → 12:38:50 UTC (2 h 22 min), rc=0, `DONE`.** It is the seventh
month on the 186-column block and, under recorded rule, the last: 2025-05 and 2025-09 stay unbuilt
because the lever they feed is priced under the floor.

| member | rmse on `april_loo` | sha (16) |
| --- | ---: | --- |
| F1-fresh | **193.4082** | `b82251655b1c4a60` |
| D1-winter | 205.0517 | `140034757e7a93fa` |
| C2-cat | 207.5815 | `9dfcff6a314b376f` |
| L2-hub | 211.4119 | `34a4a8e121bd237c` |
| T-tailv2 | 212.1837 | `97994343a976b4a1` |
| N1-nmtail | — | `5494aecab1201e32` |

All six carry 175,288 rows with ids identical to the block.

**base6 on 2025-04** — sha `7f2dfe5395d05a9b4707f65a698cb3f8127821048905ee8f41920018133b3e56`,
175,288 rows, ids identical, no null, **rmse 194.0227**. The six-member rotation on the
three folds reads six 287.4180 / 208.2407 / 339.5657 against seven 286.3274 / 207.0083 /
339.0173 — the `V3-allin` gap is +1.0906 / +1.2324 / +0.5484, the same shape as every
earlier month, and base6 stands as the extension target.

**The 2025-04 block** — sha `d7b992eb6a5b6b8df0a32f79f084c49f1719d2daefc0e74040c76d92932aec36`,
**186 columns** over 175,288 rows, 46.8 MB, verify **6/6**, head columns 13/13 present,
12 head-dependent columns moved and 173 shared columns identical.

**Seven months now carry the 186-column block**: 2025-02 (143,731), 2025-03 (164,450),
2025-04 (175,288), 2025-06 (183,142), 2025-08 (191,181), 2025-10 (185,674), 2025-11
(162,332) — **1,205,798 rows** beside the three folds' 510,096. That is the asset recorded rule
part 2 named, and §7ca iv's 3.36× fitting panel.

**April is the easiest winter-side month yet on F1-fresh (193.41)** and the only month
where D1-winter beats C2-cat, L2-hub and T-tailv2 at once — a de-icing arm still pricing
something in April where August had nothing for it (§7br viii). The step-2 gain is
+0.6191 (194.0227 against the best member's 193.4082 is a *loss* on that member alone;
against the six-member anchor it is the usual stacker gain), which puts April with March
among the low-disagreement months rather than with February.

## §7ca iii — LEVER A staged: the loader the registered panel needed, and the run that launches on the word

**Desk, 2026-09-10:** *"stage the run so it launches on the word: scripts, cache keys,
the registration commit on the record with minutes to spare."*

### i. The gap: §7ca's one change was not implementable with the registered path

§7ca iv registered exactly one change to v17's arm — *"the fitting panel"*, 510,096 rows
to 1,715,894. `code/arms.py` could not express it, and the reason is structural rather
than a missing flag:

* `ROTATION` maps a **fold** to the folds its head is fitted on, and a fold is one of
  three named months;
* every panel comes from `base.load()`, which reads `base.FOLDS` and builds `p11`, `p12`,
  `r11`, `r12` from the fold-side vectors — objects a non-fold month has none of.

So the design's whole claim had no path to a run. This section builds it. **The seven
extension months are FITTING rows and nothing else**: no extension month is scored, none
appears in a cell table, in the projection weights or in a board endpoint.

### ii. `code/extend.py`, and why the block is rebuilt rather than read

A new file, so nothing that exists changes. It gives an extension month the two things
the fold panels supply — a block and a target:

* **the block** is rebuilt in memory through the same `block.build` call `code/emit.py`
  makes, from the same `mask_like_ranking` frame, the same `base.prefix_frame()` map and
  the same head. **Not read off the parquet**: `emit.py` casts *"every other column a
  float32"*, and a float32 block moves a ridge arm by up to 9.5 s² — measured in this
  lane, and the reason the fold side has always been built in memory. A fitting panel is
  not served bytes, but a panel that differed from the fold side in dtype would put the
  difference inside the fit instead of inside the record.
* **the target** is the six-member `base6` residual — which is also the head the block's
  twelve `{Nf_,No_}_n_res_*` columns were built against, because `work/month_chain.sh`
  passes `--pred oof/base6--<month>.parquet` to `emit.py`. The head inside the features
  and the head under the target are the **same head**, which is the property the fold
  side has with `p12` and which a mixed pair would have broken silently.

**The parquet is still used, for one thing: as the cache key.** `extend.reconcile()`
reads the registered parquet and stops the run unless the in-memory build carries the
same rows in the same order and agrees on every one of the arm's columns to a float32
round trip; it also stops if the two disagree about **which** rows are null.

| month | rows | block sha256 | base6 sha256 | base6 rmse |
| --- | ---: | --- | --- | ---: |
| `2025-02` | 143,731 | `32c9e25d05f6` | `36f7dd9f6d80` | 247.6790 |
| `2025-03` | 164,450 | `ec26128f3a5e` | `64c47603ebfd` | 200.8562 |
| `2025-04` | 175,288 | `d7b992eb6a5b` | `7f2dfe5395d0` | 194.0227 |
| `2025-06` | 183,142 | `5e7e6e10898d` | `0bff146d3bce` | 304.4005 |
| `2025-08` | 191,181 | `137a61f23de4` | `60c84140c334` | 243.3441 |
| `2025-10` | 185,674 | `29932ab19608` | `2bfc9a695928` | 215.9453 |
| `2025-11` | 162,332 | `3e231a8f7cd8` | `339ca16ed421` | 216.6593 |
| **total** | **1,205,798** | digest `3a690428f600` | | |

`digest` is one sha over the fourteen, written into the run's own output as
`fit_panel.panel_digest`, so the record names the panel in one string. Every month's
verify record must say `PASS` or it is not a fitting month — a block no section names is
not fitted on. **2025-05 and 2025-09 are absent by recorded rule, not by accident.**

**Measured, on 2025-02:** 12.2 s to rebuild, ids identical to the registered parquet,
worst relative column gap **5.13 × 10⁻⁸** (on `Nf_n_res_cell`) against a 10⁻⁵ tolerance —
a float32 round trip and nothing else. Seven months are about 90 s in front of the fit.

### iii. `code/arms.py`: two options, both default-off, and a diff that is all guard

| added | takes effect when |
| --- | --- |
| `import extend as XT` | never, by itself |
| `--fit-months` (default `""`) | non-empty only |
| `--assert-cells` (default off) | passed only |
| the panel build | `if a.fit_months:` |
| `out["fit_months"]`, `out["fit_panel"]` | `if extrep is not None:` |
| the concat onto each fold's fitting rows | `if ext is not None:` |
| the pre-fit assertion | `if a.assert_cells:` |
| `fit_rows_folds_only`, `fit_rows_extension` | `if ext is not None:` |

**At the defaults the output JSON does not gain a key**, which is why the two report
fields are attached after `out` is built rather than inside it. The one unguarded line is
`fold_fit_rows = int(len(fit))`, an assignment nothing reads unless `ext` exists.

**And inertness is proved by bytes, not by reading the diff.** `work/inert.sh` runs the
file as it was before this section and the file as it is after it, on the same command
with neither new option — `--rows unmatched --fit-rows all --arms N --learner ridge
--folds december` — and hashes the two outputs:

| | sha256 of `analysis/inert-*.json` |
| --- | --- |
| `arms.py` before §7ca iii | `477b74c582e89aedb81c46e2b992ce578345428b86672caf9122ae17ed5075ca` |
| `arms.py` after §7ca iii | `477b74c582e89aedb81c46e2b992ce578345428b86672caf9122ae17ed5075ca` |

**Identical.** The two JSONs and the throwaway copy of the old file were deleted after the
comparison; `work/inert.sh` reproduces them.

One thing the concat does that is worth writing down: it **trims the fold side to the
columns a fit reads** before joining. A fold fitting frame is the panel plus a
186-column block and the extension frame is 35 columns, so the outer join of the two
untrimmed is about 2.7 GB of float64 that no line below ever reads. Nothing is dropped
that `keep_rows`, either filter, the assertion or `arm_predict` can touch.

### iv. The pre-fit assertion, measured before the run rather than during it

§7ca ii v registered it: *"the arm run asserts, before it prices anything, that the
held-out fold shares zero cells with its fitting folds and that every fitted column is
one of the registered 24. A run that cannot assert both stops and reports rather than
serving."* `extend.assert_disjoint` and `extend.assert_columns` are that assertion, and
`--assert-cells` is what the staged runner passes. Run here on the fold panels alone,
ahead of the GO:

| scored fold | its cells | fitting folds | their cells | **shared** |
| --- | ---: | --- | ---: | ---: |
| `july_loo` | 35,561 | `december` | 33,669 | **0** |
| `december` | 33,669 | `july_loo` | 35,561 | **0** |
| `january` | 33,383 | `july_loo`, `december` | 69,230 | **0** |

The three cell counts are §7ca ii v's own — 35,561 / 33,669 / 33,383 — reproduced by the
assertion code rather than by the script that first measured them, which is the point of
running it here. The extension months add no risk to the same claim and are checked by
the same call: a 2025-02 cell is a window in absolute UTC and cannot be a 2025-07 one.

The column half is `sorted(Nf_*, No_*)` on the block, **exactly the 24** §7ca iv quotes
from `analysis/serve-lit.json` — counted on the emitted block, not asserted: 24 of 186.

### v. The fitting panel, per fold, as the run will build it

| scored fold | v17's fitting rows | with the extension | multiple |
| --- | ---: | ---: | ---: |
| `july_loo` | 165,677 | **1,371,475** | 8.28× |
| `december` | 190,713 | **1,396,511** | 7.32× |
| `january` | 356,390 | **1,562,188** | 4.38× |
| *panel inventory (§7ca iv's row)* | *510,096* | *1,715,894* | *3.36×* |

§7ca iv's 3.36× is the **inventory** — the distinct rows the design may fit on, which is
what `analysis/serve-lit.json` reports as `fit_rows: 510096`. Per fold the multiple is
larger, and largest exactly where v17's panel was smallest. **The registered figure is
not revised** (score-feedback rule); this row is what the same panel looks like at the grain the fit
actually sees, and it is stated before the run.

### vi. The command, staged, with every option's provenance

`work/leverA-fit.sh`, committed before the GO and not edited after it:

```
code/arms.py --rule d4c59868…f01a0 --rows unmatched --fit-rows all \
    --arms N --learner ridge --folds july_loo,december,january \
    --fit-months 2025-02,2025-03,2025-04,2025-06,2025-08,2025-10,2025-11 \
    --assert-cells --out arms-step2-leverA.json
```

`--arms N` and not v17's three. `analysis/serve-lit.json` records `update.arm = "N"` with
`form: {"every row": {"arm": "N", "columns": 24}}`; `N_reach` and `N_reach_half` were a
**second** reading of these same folds, and `code/arms.py`'s own note says a third would
be *"a weight chosen after seeing one"*. The design registered one instrument and this
runs one.

**The hold is not an option in the file.** It is re-applied cell by cell at fit time by
`code/serve.py` from the table this run produces, no cell is excluded here by hand, and
the two LFPG january rows (84,240 / 58,206) are in the fitting panel and in the priced
population by the desk's ruling of 2026-09-10. **The serve is not in this script**: the
desk rules every serve, and §7ca vii's order is one cell at a time. The command it would
take sits in the file as a comment, unrun.

**What the run is measured against, and it is registered already**: §7ca ii ii's band
**+0.0626 … +0.1674 s**, middle row **+0.1134 s**. Failure is below +0.0626; above
+0.1674 is evidence the panel raised the realisation factor. Neither branch revises the
band.

**Files.** `code/extend.py`, `work/leverA-fit.sh`, `work/leverA-smoke.sh`,
`logs/leverA-smoke.log`, and two guarded options in `code/arms.py`.

## §7cc — LEVER A MEASURED: zero of seven cells clear the hold, and the design's premise is refuted

**Desk, 2026-09-10:** *"GO for lever A fits NOW (7ca ii) … Run the pre-fit assertion
first, then the fold-side fits on all rows of the never-served cells exactly as
registered … the read against the band +0.0626..+0.1674 (middle +0.1134)."*

`work/leverA-fit.sh` ran **12:54:40 → 12:57:25 UTC, 2 min 45 s, rc=0**.
`analysis/arms-step2-leverA.json`, verdict in `analysis/leverA-verdict.json`. Panel
digest `3a690428f600`, seven extension months, 1,205,798 fitting rows added.

### i. The pre-fit assertion, first, as the desk required

| scored fold | priced cells | fitting cells | **shared** | fitted columns | not among the registered 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `july_loo` | — | — | **0** | 24 | **0** |
| `december` | — | — | **0** | 24 | **0** |
| `january` | — | — | **0** | 24 | **0** |

Both halves asserted on the **priced** population (the assertion runs after
`keep_rows`, so it is the rows the arm is actually scored on), for every fold, before
any number below was computed. Recorded in the run's own `pre_fit_assertion` block.

### ii. THE VERDICT: the arm serves no new cell, so the board move is +0.0000 s

**FAILURE as §7ca ii ii registered it.** The failure line was **+0.0626 s**; the measured
move is **+0.0000 s**, because **none of the seven target cells clears the hold**.

| cell | ranking rows | registered at the middle row | the hold, on the read | verdict |
| --- | ---: | ---: | --- | --- |
| `LFPG/unreached` | 32,298 | +0.0882 | july `[−1022.7, −86.9]` — **clear of zero BELOW**; dec `[−181.3, +1923.4]`; jan `[−1144.1, +813.1]` | **held** |
| `EGLL/reached` | 5,728 | +0.0093 | dec `[−5832.8, −969.1]` — **below**; july `[−3456.7, +1231.2]`; jan `[−1475.2, +227.0]` | **held** |
| `LEBL/reached` | 3,089 | +0.0066 | july `[+752.7, +4482.6]` ✓, jan `[+126.3, +1846.3]` ✓, **dec `[−255.4, +3565.3]`** ✗ | **held** |
| `LSZH/reached` | 1,790 | +0.0042 | not clear above on any fold | **held** |
| `LEMD/unreached` | 1,306 | +0.0024 | **december unpriced, 47 rows** — the 500-row bar, exactly as §7ca ii predicted | **held** |
| `EHAM/reached` | 2,509 | +0.0023 | dec `[−3519.4, −370.5]` — **below**; july `[−526.9, +951.8]`; jan `[−855.0, +7082.2]` | **held** |
| `EGLL/unreached` | 185 | +0.0004 | jan `[−1607.7, −194.7]` — **below** | **held** |

**And the looser reading does not rescue it.** `code/serve.py` chooses each fold's served
set on the *other* folds. Under that cross-fit reading `LEBL/reached` does enter — on
**december only**, which carries no board share — while `july_loo`'s set is
`LEMD/reached, LIRF/unreached, LTFM/unreached` and `january`'s is `EDDF/reached,
LEMD/reached, LTFM/unreached`. **No target cell enters either board-share fold under any
reading of the hold**, so the move is +0.0000 s either way.

**Four of the twenty priced target-cell reads are now clear of zero BELOW** — LFPG july,
EGLL/reached december, EHAM/reached december, EGLL/unreached january. Under v17 two were.
The cells did not become uncertain; four of them became measurably harmful.

### iii. Why it failed, and it is not the panel's size — it is what the interval is

§7ca iv's claim was one sentence: *"every hold above was an interval width, and interval
width falls with the panel."* **That is wrong in kind, and this run is what shows it.**

The hold's interval is `base.paired(r, d, days)` — a **day-clustered bootstrap over the
scored fold's own rows**. Both runs score the identical rows: `LFPG/unreached` is the
same 19,963 july rows before and after. **A fitting panel adds no scored row**, so it
cannot narrow that interval through sample size at all. It can only change `d`, and the
interval's width moves with `|d|`.

Measured, on the twenty priced target-cell reads:

| | v17 → lever A |
| --- | ---: |
| reads whose interval got **wider** | **15 of 20** |
| reads whose interval got narrower | 5 of 20 |
| median width ratio | **×1.352** |

and per fold the width ratio tracks the **amplitude** ratio, not the panel ratio:

| fold | fitting rows ×  | median width ratio | `fitted_scale` v17 ÷ lever A |
| --- | ---: | ---: | ---: |
| `july_loo` | 8.28× | ×1.466 | 1.1186 ÷ 0.8250 = **1.356** |
| `january` | 4.38× | ×1.252 | 0.7961 ÷ 0.5160 = **1.543** |
| `december` | 7.32× | ×0.959 | 0.3712 ÷ 0.5530 = **0.671** |

December is where `d` shrank and it is the one fold whose intervals did not widen. **The
panel changed the arm's amplitude, and the amplitude carried the width.** Coefficient
variance was never what the hold was measuring: at 165,677 standardised rows with
`RIDGE_ALPHA = 10`, the penalty is about 6 × 10⁻⁵ of the data term, so v17's ridge was
already effectively unregularised and its coefficients were already stable. Eight times
that changes the estimate, not its precision.

`fitted_scale` says the same thing from the other side. v17's arm was **under**-amplified
on july (optimum 1.1186 > 1, so full amplitude was leaving gain on the table); lever A's
is **over**-amplified on july and january (0.8250, 0.5160). The bigger panel fits a
*stronger* relation than the two rotation folds hold, and full amplitude then overshoots
the folds it is scored on.

### iv. And it is not neutral: the same change costs a cell the lane already serves

| scored fold | v17 served set, cross-fit | its board s | lever A served set | its board s |
| --- | --- | ---: | --- | ---: |
| `july_loo` | EDDF/reached, LEMD/reached, LTFM/unreached | **+0.5518** | LEMD/reached, LTFM/unreached *(+LIRF unpriced)* | **+0.4778** |
| `december` | EDDF/reached, LEBL/reached, LEMD/reached, LTFM/unreached | **+0.5079** | LEBL/reached, LEMD/reached, LTFM/unreached | **+0.2339** |
| `january` | EDDF/reached, LEMD/reached, LTFM/unreached | **+0.1932** | EDDF/reached, LEMD/reached, LTFM/unreached | **+0.1686** |

`EDDF/reached` **drops out of july_loo's and december's served sets** because its january
interval widened past zero (`+1261.4 [+335.6, +2500.7]` → `+554.7 [−1500.1, +2414.3]`),
and each fold's set is chosen on the others. **So the registered change is a regression
on the population the lane already serves, not merely a null on the new one.** Nothing is
served: no candidate, no upload, and the desk rules every serve.

### v. What DID move, because it is the honest half of the read

On the cells arm N already serves, the same panel raised the point estimate hard:

| cell | fold | v17 | lever A |
| --- | --- | ---: | ---: |
| `LTFM/unreached` | july | +755.6 | **+1556.0** |
| `LTFM/unreached` | december | +232.5 | **+506.9** |
| `LTFM/unreached` | january | +264.4 | **+420.7** |
| `EDDF/reached` | july | +3004.6 | **+4223.2** |
| `LEMD/reached` | december | +1308.6 | +1438.2 |

**The panel is not inert — it strengthens the relation exactly where the relation was
already strong, and weakens or reverses it where it was weak.** That is a measured fact
about panel size on this population, and §7ca vi registered the one-sided reading that
would have read it as a raised realisation factor. It is raised — on the served cells,
which lever A was not about.

### vi. The repair, named before the close, and what it would cost

The failure closes **this path**, not the destination. Three repairs, in the order their
evidence supports:

1. **Re-price the SERVED cells on the 3.36× panel and hold the amplitude to `fitted_scale`
   read out of fold.** §v says the served cells gained; §iii says full amplitude now
   overshoots. Both are the same fact. One fold-side run, no new data, and it is priced
   against the served set's own +0.5518 / +0.1932, not against lever A's zero.
2. **Fit the extension months' target on the fold head's family rather than each month's
   own `base6`.** The fold rows carry `r12` and the extension rows carry a `base6`
   residual; the arm learns whichever dominates, and 86 % of the panel is `base6`'s. A
   per-month standardisation of the target, or a weight that restores the folds' share,
   separates "more rows" from "a different target".
3. **Stop treating the hold's interval as something a panel can narrow.** A cell held for
   interval width is held by its **own scored rows**, and the only levers on it are a
   larger `d` on those rows or more scored rows — which for `LFPG/unreached` means the
   2025-04-style month programme recorded rule closed, and the k-curve prices that under the floor.

**§7ca and §7ca ii are not revised** (score-feedback rule): their band was registered before the data and
the data came in below it. This section is the reading, and the ledger row is the
correction.

**Files.** `work/leverA-fit.sh`, `analysis/arms-step2-leverA.json`,
`analysis/leverA-verdict.json`, `logs/leverA-fit.log`.

## §7cd — THE REPAIR, registered before it runs: the served cells at an out-of-fold amplitude

**Desk, 2026-09-10:** *"Repair GO on the cheap one from 7cc vi: re-price the
ALREADY-SERVED cells on the 3.36x panel with the amplitude held to an OUT-OF-FOLD
fitted_scale. Conditions: (1) register before the run, in RESULTS, commit hash first:
the bar is the served set own +0.5518 (july) / +0.1932 (january), the failure line, and
what a pass changes (nothing serves today; a pass makes a candidate for a later day);
(2) fold-side only, no serve, no candidate; (3) the laptop fit slot is yours for this
~3 min run only, then report and it goes to surfacefit; (4) the amplitude is chosen out
of fold, never on a board reading (score-feedback rule)."*

**No data exists when this is written.** `work/leverA-repair.sh` is committed with it and
is not edited after.

### i. What is unchanged, and the one thing that is not

**Unchanged from §7cc**: arm `N`, the 24 registered columns, ridge, `--rows unmatched
--fit-rows all`, the cross-fit `ROTATION`, the 3.36× panel with its digest
`3a690428f600`, and the pre-fit assertion. **Changed**: the endpoint is the cells the lane
**already serves** — `EDDF/reached`, `LEMD/reached`, `LTFM/unreached` — and `d` is
multiplied by an amplitude that is not 1.

### ii. The amplitude, defined so it cannot be read off a fold

`arms.py::oof_amplitude`, new and default-off (`--amplitude 1.0` multiplies nothing and
is what every run so far did). For each scored fold `f` and each extension month `m`:

> fit the head on **(`ROTATION[f]` folds) ∪ (the extension months except `m`)**, through
> the same `arm_predict` call the served head uses; predict `m`; accumulate `Σ r·d` and
> `Σ d²` over all seven months; the amplitude is their ratio.

**No row of fold `f` is read.** The months it is measured on are fitting months no fold
is ever scored on, so the number is out of fold for all three folds by construction
rather than by argument — desk condition 4, and score-feedback rule. It is **not clipped**: a value above
1 would say the folds want more than full amplitude, and hiding that would make the
number a decision instead of a measurement. `fitted_scale`, which the record has always
printed beside the full-amplitude figure and never substituted, is this same ratio read
**in** fold; §7ca iv's reason for never using it — *"a scale fitted on the scored month
would not be servable anyway"* — is exactly what this estimator removes.

### iii. The bar, and the failure line

| fold | board share | v17's served set, cross-fit | §7cc at full amplitude |
| --- | ---: | ---: | ---: |
| `july_loo` | 0.5571 | **+0.5518** | +0.4778 |
| `january` | 0.4429 | **+0.1932** | +0.1686 |
| `december` | — | +0.5079 | +0.2339 |
| **share-weighted** | | **+0.3930** | **+0.3409** |

**THE BAR IS +0.5518 (july_loo) AND +0.1932 (january), share-weighted +0.3930.**

**FAILURE** is either of:

1. the served-set board below the bar on **either** board-share fold; or
2. **any** of the three served cells dropping out of a board-share fold's cross-fit
   served set — which is precisely what §7cc's full-amplitude run did to `EDDF/reached`,
   and it is a failure on its own whatever the pooled number says.

§7cc's run sits **−0.0521 s under the bar** share-weighted, so the repair has to recover
that before it has recovered anything.

### iv. What the arithmetic says before the data, so the run is informative either way

Scaling `d` by `A` turns the gain `2A−B` into `m(2As − A²)` where `s` is the fold's own
optimum and `m = mean(d²)`. Full amplitude is `m(2s−1)`. On the **pooled class** scales
§7cc measured — july 0.8250, december 0.5530, january 0.5160 — the ratio to full
amplitude is:

| fold | `s` | best possible ratio | at `A` = 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `july_loo` | 0.8250 | **×1.05** | 0.88 | 0.97 | 1.02 | 1.05 | 1.04 | 0.93 |
| `december` | 0.5530 | ×2.88 | 2.86 | 2.86 | 2.68 | 2.31 | 1.75 | 0.06 |
| `january` | 0.5160 | **×8.32** | 8.31 | 8.10 | 7.26 | 5.80 | 3.71 | **−2.34** |

**So the whole of this repair's upside is january, and its whole risk is january too.**
July is within 5 % of its own optimum at full amplitude and can barely move; january sits
almost exactly at break-even (`2s−1 = 0.032`), which is why its board figure is the one
that collapsed in §7cc and why a modest reduction in amplitude multiplies it. **An
amplitude above about 1.03 turns january negative.** January carries board share 0.4429.

These ratios are on the **pooled no-clock class**, not on the three served cells, which
have their own optima; they are the shape of the answer and not the answer. **No band is
registered from them** — the endpoint is the bar in §iii, and this table exists so that a
january-driven result is read as arithmetic rather than as a discovery.

### v. What a pass changes, and what it does not

**Nothing serves today** (desk condition 2): fold-side only, no serve, no candidate, and
the desk rules every serve.

A pass makes a **candidate for a later day**, and only under one more condition written
here rather than after the read: the three cells are **already served** at the shipped
bytes, so the thing an upload would buy is the **increment over the bar**, not the level.
**An increment below +0.1 s is not an upload** — that is the floor as the upload rule is
written, and share-weighted it means a pass worth proposing has to reach **+0.4930**.
A result between +0.3930 and +0.4930 is a pass on the bar and a **hold** on the floor:
it goes to the ledger as a measurement and not as a candidate.

**Files.** `work/leverA-repair.sh`, and `--amplitude` in `code/arms.py`
(`oof_amplitude`), both committed before the run.

### §7cd ii — the read: the repair FAILS both registered conditions, and the panel is a regression under either amplitude

`work/leverA-repair.sh` ran **13:09:50 → 13:12:34 UTC, 2 min 44 s, rc=0**.
`analysis/arms-step2-leverA-repair.json`, verdict in
`analysis/leverA-repair-verdict.json`. Pre-fit assertion first, as in §7cc: 0 shared
cells and 24/24 registered columns on all three folds.

**The out-of-fold amplitude came out at almost exactly 1.**

| fold | **amplitude, out of fold** | the fold's own optimum at full amplitude (§7cc) |
| --- | ---: | ---: |
| `july_loo` | **0.9648** | 0.8250 |
| `december` | **0.9102** | 0.5530 |
| `january` | **0.9854** | 0.5160 |

and the seven months disagree with each other far more than they disagree with 1 — on
july's head the per-month scales are 0.83, 0.89, 1.10, 1.25, 0.87, 1.06, 0.81.

**Against the bar:**

| fold | the bar (v17) | §7cc, full amplitude | **the repair, oof amplitude** |
| --- | ---: | ---: | ---: |
| `july_loo` | +0.5518 | +0.4778 | **+0.4703** |
| `january` | +0.1932 | +0.1686 | **+0.1694** |
| `december` (no board share) | +0.5079 | +0.2339 | +0.2313 |
| **share-weighted** | **+0.3930** | +0.3409 | **+0.3370** |

**FAILURE on both registered conditions.** Below the bar on `july_loo` and on `january`,
share-weighted **−0.0560 s under it**; and `EDDF/reached` is **still out of `july_loo`'s
cross-fit served set**, which §7cd iii made a failure on its own.

**Why, and it is the same structural fact as §7cc iii seen from the other side.** §7cd ii
defined the amplitude so that no row of the scored fold enters it — which is what makes
it servable. The scale the folds want (0.52 – 0.83) is a **property of the scored months**;
an estimator that never reads them cannot recover it. What the estimator recovered
instead is the extension months' own optimum, and that is ≈ 1: **the extension months
want full amplitude.** So the correction it applies is 1 – 4 %, and §7cd iv's arithmetic
says a 1 – 4 % cut buys january almost nothing (the table's ×3.71 sat at `A` = 0.9,
not at 0.985). The registered prediction that *"the whole of this repair's upside is
january"* was right about where the movement would be and wrong about its size, because
the amplitude the estimator could honestly reach was never near january's optimum.

**And the ranking of the three configurations is now measured, which is the useful part:**

| configuration | served-cell board, share-weighted |
| --- | ---: |
| **v17: the three folds only, full amplitude** | **+0.3930** |
| the 3.36× panel, full amplitude (§7cc) | +0.3409 |
| the 3.36× panel, out-of-fold amplitude (§7cd) | +0.3370 |

**The extension panel is a regression on the cells the lane already serves under both
amplitudes.** It is not an amplitude problem and it is not fixed by one.

The per-cell reads say where it goes: the panel **raises** `LTFM/unreached` on every fold
(july +755.6 → +1525.6, december +232.5 → +505.4, january +264.4 → +420.7) and `EDDF/reached`
on july (+3004.6 → +4101.7), and **collapses `EDDF/reached` on january** (+1261.4 → +562.1),
which is the fold that decides july's served set. One cell on one fold carries the whole
regression.

**Nothing serves** (desk condition 2): no serve, no candidate, and the increment question
in §7cd v does not arise because the level is below the bar. **The laptop fit slot is
released** — it was used for one 2 min 44 s run, as condition 3 allowed.

**What is now closed and what is not.** Repair 1 of §7cc vi is measured and fails; with it
goes the reading that lever A's problem was amplitude. §7cc vi's repair 2 (the extension
months' target family) and repair 3 (that a panel cannot narrow a scored-fold interval at
all) are untouched by this run, and repair 3 is the one this read strengthens: **the
quantities that decide both the hold and the amplitude live on the scored side, and no
amount of fitting data reaches them.** The destination — serving more of the 148,585
no-clock rows — is not closed; the path through a larger fitting panel is, on two
measurements now.

**Files.** `work/leverA-repair.sh`, `analysis/arms-step2-leverA-repair.json`,
`analysis/leverA-repair-verdict.json`, `logs/leverA-repair.log`.

## §7cc vi.3 — THE STRUCTURAL STATEMENT: the hold and the amplitude live on the scored side

**Desk, 2026-09-10:** *"write 7cc vi.3 as the structural statement with its arithmetic
(the hold interval and the amplitude are scored-fold properties no fitting panel
reaches)."* Zero fit. `analysis/amplitude-arithmetic.json`.

### i. The arithmetic, and it is checked on a controlled pair rather than argued

Write the arm's correction as `d = A · d₀`: a **direction** `d₀`, which the fit chooses,
and an **amplitude** `A`, which the record has always pinned at 1. On a cell with `n`
scored rows, `m = mean(d₀²)` and `s = mean(r·d₀)/m` — **`s` is the cell's own optimum
scale, a property of its scored rows** — the three quantities the hold reads are:

> **point** `G(A) = n·m·(2As − A²)`, so `G(A)/G(1) = (2As − A²)/(2s − 1)`
> **half-width** `≈ A ×` half-width at full amplitude — the day-level statistic is
>   `2A·u_k − A²·v_k` and the `2A·u` term dominates for `A` of order 1
> **t** `= point / half-width = c·(2s − A)` with `c > 0`: **linear in `A`, slope negative**

`analysis/arms-step2-leverA-repair.json` is `analysis/arms-step2-leverA.json`'s `d`
multiplied by a known per-fold constant **and nothing else** — same head, same panel,
same rows — so it is a controlled test of all three:

| fold | `A` | `s` | point ratio predicted | **measured** | width ratio predicted | **measured** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `july_loo` | 0.9648 | 0.8250 | 1.0170 | **1.0171** | 0.9648 | **0.9663** |
| `december` | 0.9102 | 0.5530 | 1.6813 | **1.6818** | 0.9102 | **0.8991** |
| `january` | 0.9854 | 0.5160 | 1.4350 | **1.4351** | 0.9854 | **0.9849** |

The point identity holds to **four decimal places**; the width identity holds cell by
cell (median ratio 0.9663 / 0.9138 / 0.9855 against `A` of 0.9648 / 0.9102 / 0.9854, over
13 / 12 / 10 cells, spread 0.886 – 0.989).

### ii. What follows, and it is the whole of lever A's failure

**1. The fitting panel is not on the dial.** Every term above is a functional of `(r, d₀)`
**on the scored fold's own rows** and of `A`. `r` is fixed — it is the head's residual.
`A` is a dial the record pins at 1. The fitting panel touches **only the direction `d₀`**,
and it enters the hold only through `s` and `n·m`, both measured on the scored rows. **A
fitting panel adds no scored row, no scored day and no scored residual, so it cannot
narrow the hold's interval through sample size at all** — the one route it has is a `d₀`
better aligned with `r` *on the scored rows*. Measured across v17 → the 3.36× panel, the
alignment got **worse on 19 of the 28 priced cell-fold reads** and better on 9.

**2. Amplitude trades size against significance, and it is bounded at both ends.** Since
`t(A) = c(2s − A)` falls in `A`, a cell clears for every `A ≤ A* = 2s − (2s−1)/t(1)`, and
`sup_A t = t(1)·2s/(2s−1)`, approached as `A → 0` **where the gain itself goes to zero**.
So a cell that cannot clear at *any* amplitude cannot be rescued by scale, and a cell that
clears only at a small amplitude buys only a small correction. There is no loophole here
and no free significance.

**3. Granting an oracle amplitude — which §7ca iv forbids — buys a ninth of the floor.**

| target cell | `sup t`, july / december / january | could clear on every fold at some `A`? | worth at the registered middle row |
| --- | --- | :-: | ---: |
| `LFPG/unreached` | **0.074** / 0.975 / 0.393 | **no** | +0.0882 |
| `EGLL/reached` | 0.579 / 0.169 / 1.447 | no | +0.0093 |
| **`LEBL/reached`** | 2.292 / 1.694 / 1.941 | **yes** | **+0.0066** |
| **`LSZH/reached`** | 1.123 / 1.488 / 1.589 | **yes** | **+0.0042** |
| `LEMD/unreached` | 0.706 / *unpriced* / 1.804 | no | +0.0024 |
| `EHAM/reached` | 1.751 / −0.033 / 0.949 | no | +0.0023 |
| `EGLL/unreached` | 0.896 / 0.013 / −0.937 | no | +0.0004 |

**Two cells, worth +0.0108 s together — about a ninth of the 0.1 s floor** — and
`LFPG/unreached`, which is 78 % of the whole design, **cannot clear on any fold at any
amplitude**: its july `sup t` is 0.074.

### iii. The one sample-size lever there is, and it is on the scored side

The interval is day-clustered, so its half-width falls as `1/√K` in the number of scored
day clusters. A cell clears when `t·√k > 1`, so it needs **`k > (1/t)²` months of scored
days on every fold**. That is the only sample-size lever the hold has, and it is on the
scored side, not the fitting side.

| target cell | months of scored days needed, july / december / january | revivable |
| --- | --- | :-: |
| `LFPG/unreached` | **sign wrong** / 3.25 / **sign wrong** | **no** |
| `EGLL/reached` | sign wrong / sign wrong / sign wrong | no |
| **`LEBL/reached`** | 0.56 / **1.79** / 1.04 | **yes, ≈ 1.8 months** |
| `LSZH/reached` | **20.85** / 8.81 / 2.82 | yes, at 21 months |
| `LEMD/unreached` | sign wrong / *unpriced* / 1.69 | no |
| `EHAM/reached` | 10.08 / sign wrong / 2.03 | no |
| `EGLL/unreached` | sign wrong / sign wrong / sign wrong | no |

**Five of the seven fail because a point estimate is not positive on at least one fold.**
That is not a precision problem and no quantity of data on either side repairs it. One
cell — `LEBL/reached`, +0.0066 s — is revivable at about 1.8 months of december days, and
one — `LSZH/reached`, +0.0042 s — at twenty-one months.

**This is why §7ca iv's premise was wrong in kind and not in degree.** It read the holds
as interval widths and interval widths as a function of the panel. They are interval
widths, and they are a function of the scored fold.

## §7cc vii — BRANCH REPORT: lever A, the never-served no-clock cells

Filed as a **belief with its evidence**, `closed-provisional`, for the desk's
provisional-closes table. Matt confirms or reverses.

**THE NEGATIVE.** The registered arm (§7ca, §7ca ii) on all rows of the seven
never-served no-clock cells moved the board **+0.0000 s** against a registered band of
+0.0626 … +0.1674 s. Zero of seven cells cleared the hold, on either reading of it, and
no target cell entered a board-share fold's served set (§7cc ii).

**THE CAUSE, and it is one thing.** The design's whole claim was that the holds were
interval widths and that *"interval width falls with the panel"* (§7ca iv). §7cc vi.3
shows both halves are false: the hold's interval is a day-clustered bootstrap over the
**scored** fold's own rows, which a fitting panel does not add to, and its `t` is
`c·(2s − A)` in quantities measured entirely on the scored side. The panel changed only
the direction `d₀`, and the change made the alignment **worse on 19 of 28 priced reads**.
Five of the seven cells fail on the **sign** of a point estimate, not on its precision.

**DESTINATION VALUE — stated separately, and it is large.** The no-clock complement is
**148,585 ranking rows carrying 83.69 % of the board's squared error** (§7bz on v22's
bytes), untouched by five uploads. The never-served cells' own ceiling, projected on their
ranking rows, is **+1.8296 s on all rows** (§7ca ii). The destination is not in doubt.

**PATH FEASIBILITY — stated separately, and it is measured infeasible.** With this
instrument (the 24 label-free neighbourhood columns, ridge, the registered hold), the
maximum the never-served cells can yield is **+0.0108 s** — and that already grants an
oracle amplitude per cell per fold that §7ca iv forbids. Against the 0.1 s floor that is
**a ninth**. The gap between +1.8296 and +0.0108 is the realisation gap, and it is the
subject of the next design, not of this path.

**THE REPAIR MADE, honestly.** Two repairs were built and measured, not proposed:

| repair | what it was | result |
| --- | --- | --- |
| the 3.36× fitting panel | seven built months, 1,205,798 rows, ×8.28 / ×7.32 / ×4.38 per fold | §7cc: 0 of 7 cells, **and a regression on the served set** (+0.3930 → +0.3409) |
| the out-of-fold amplitude | leave-one-month-out on the extension months, no scored row read | §7cd ii: amplitude 0.965 / 0.910 / 0.985 — the extension months want full amplitude — and a further **+0.3409 → +0.3370** |

**THE STRONGEST REFINEMENT STILL VISIBLE.** Not more rows on either side, but a **`d₀` with
a different direction**: the 24 columns realise 0.049 – 0.089 of the ceiling on the cells
the arm *does* serve (§7ca vi), so the gap is in what the instrument reads, not in how
much of it is read. The next design is the inputs the oracle has and the 24 columns lack,
priced on the served cells first — the desk's own part (3) of 2026-09-10, and it is
already in hand as a zero-fit piece.

**WHAT WOULD REVIVE THIS LANE**, in the order the arithmetic ranks them:

1. **New columns that raise `s` on the scored rows.** The only lever §7cc vi.3 leaves
   open, and the only one that touches five cells whose failure is a wrong sign.
2. **`LIRF/unreached`**, which the desk's ruling of 2026-09-10 keeps out: **17,730 ranking
   rows, 15,795 of them in served 2026-07**, worth **+0.0416 s** priced on its january
   read, which alone lifts the design's registered middle from +0.0966 to +0.138 s. It
   needs july coverage the harvest has never delivered (152 fold rows), and it is the one
   cell where the board can see something the instrument cannot.
3. **More SCORED months.** `LEBL/reached` clears at about **1.8 months of december days**
   for +0.0066 s; `LSZH/reached` needs twenty-one. recorded rule closed the month programme for
   *fitting* months on a k-curve that priced them under the floor; **the scored-side
   arithmetic is different and has never been priced**, and this is where it would be.
4. **A hold written on a pooled endpoint** rather than per cell per fold. That is a new
   rule and therefore Matt's, not this lane's; it is recorded as a revival condition and
   nothing here assumes it.

**What this does NOT close.** The destination — serving the no-clock complement — is
untouched. What is closed provisionally is the path that reaches it by enlarging the
fitting panel, on two measurements, and the reading that the holds were a precision
problem.

**Files.** `analysis/amplitude-arithmetic.json`, `analysis/leverA-verdict.json`,
`analysis/leverA-repair-verdict.json`.

## §7ce — THE REALISATION GAP, located: the missing oracle input is the OBSERVATION, and it is missing where reach is

**Desk, 2026-09-10:** *"Meanwhile zero-fit: the realisation gap — the arm realises
0.049-0.089 of the no-clock ceiling (7ca vi); which of the oracle inputs the 24 columns
lack, on the served cells first; output = the next lever design, not a fit."*

`work/oracle_gap.py` → `analysis/oracle-gap.json`; `work/oracle_ablate.py` →
`analysis/oracle-ablation.json`; `analysis/no-coverage-unreached.json`. **No fit.**

### i. The three numbers, all on the same rows

The oracle (`base.shared`) is `E[rᵢrⱼ]` over labelled pairs inside a 600 s cell, and its
implied predictor is the cell's **self-excluded mean residual `o`**, which no 2026 cell
can supply (§7ca ii v). So the question is not "how big is the oracle" but **how much of
`o` the 24 columns see**. Per cell and fold, on the complement rows of that cell:

* **ORACLE** — `E[rᵢrⱼ]/E[r²]`, what a perfect cell constant would remove;
* **COLUMNS** — in-sample `R²` of `r` on the 24, an upper bound on any linear arm;
* **VISIBILITY** — in-sample `R²` of **`o`** on the 24 — how much of the oracle's own
  predictor the instrument can see. This is the number that decides whether the gap is
  in the columns or in the fit.

### ii. It splits on reach, and it splits hard

| | `No_n_res_cell` coverage | VISIBILITY of `o` | COLUMNS ÷ ORACLE |
| --- | ---: | ---: | ---: |
| **reached cells** (17 cell-folds) | **1.00** | **8.5 – 60.2 %** | median **0.632**, up to 2.14 |
| **unreached cells** (11 cell-folds) | **0.00** | **0.6 – 7.6 %** | median **0.145**, max 0.393 |

**On an unreached cell the whole `No_*` family is null by construction** — `unreached`
*is* `No_n_n_cell = 0`, "the network heard no other aircraft in this cell" — so the arm
there is a **twelve**-column file-clock arm, not the 24-column arm the record describes.
That is the realisation gap: **the served cells realise a median 0.63 of their
cell-constant oracle; the unreached ones realise 0.145.**

`LTFM/unreached`, this lane's largest served cell, is the case in point: oracle 22.70 /
13.29 / 19.04 %, visibility **3.53 / 2.75 / 1.74 %**, columns `R²` 2.90 / 1.97 / 1.31 %.
§7ca vi's realisation factor of 0.049 – 0.089 is that, and it is not a property of the
ridge.

### iii. The ablation: it is the OBSERVATION, not the file clock

Measured where **both** families are present, so "unreached cells are intrinsically
harder" is separated from "the observation is the informative half":

| cell | fold | VISIBILITY: both | **`No_` only** | **`Nf_` only** |
| --- | --- | ---: | ---: | ---: |
| `EDDF/reached` | july | 43.12 | **42.31** | 5.37 |
| `EDDF/reached` | december | 55.29 | **54.03** | 11.49 |
| `LEBL/reached` | july | 24.50 | **23.79** | 2.11 |
| `LEMD/reached` | july | 22.74 | **22.11** | 2.59 |
| `EHAM/reached` | december | 35.59 | **34.23** | 2.14 |
| `LSZH/reached` | december | 36.77 | **33.10** | 5.69 |

**`No_` alone is within a point or two of the full 24 on nearly every reached cell, and
`Nf_` alone sees a fifth of what it sees.** The file clock is fully covered — 0.95 – 1.00
everywhere, including on unreached rows — and it is **not a substitute**. The oracle input
the 24 columns lack is one thing: **the neighbour's observed off-block time**, and it is
lacking exactly on the rows the network did not hear.

### iv. And a wider window does not recover it: the unreached rows are dark, not sparse

If unreached rows were merely in *sparse* 600 s cells, the block's wider observed windows
would still be populated. Coverage on the unreached complement:

| fold | unreached complement rows | `No_n_res_apt_30m` | `No_n_res_rwy_60m` | `No_n_res_cell` |
| --- | ---: | ---: | ---: | ---: |
| `july_loo` | 76,332 | 0.22 | 0.24 | 0.00 |
| `december` | 42,123 | 0.19 | 0.16 | 0.00 |
| `january` | **69,310** | **0.03** | **0.03** | 0.00 |

A **30-minute airport-wide** window that hears nothing is not a sparse cell; it is a
**dark airport-day**. So widening the grid is not the lever: it recovers an observation
for 19 – 24 % of july and december unreached rows and for **3 %** of january's — and
january is the fold that proxies 2026-01, one of the two board-share months.

### v. THE NEXT LEVER, and what it is not

**The binding constraint is reach**, and reach is `receipt × cell occupancy`, a property
of the harvest and not of the model. The complement is **66 – 82 % unreached** (§7ca ii:
reached shares 17.7 % january, 25.4 % july, 34.2 % december; served 33.7 % and 29.7 %),
so the instrument is half-empty on two thirds to four fifths of the population that
carries 83.69 % of the board's squared error.

**What this rules out, on measurement rather than on argument:**

* **more fitting rows** — §7cc vi.3, and it is now doubly clear: the columns are not
  short of data, they are short of a column;
* **a better learner or form** — the columns' *in-sample* `R²` is already 0.145 of the
  oracle on unreached cells, so no fit of these columns reaches further;
* **a wider neighbour window** — §iv, 3 % on january;
* **the file clock** — §iii, a fifth of the observation's information where both exist.

**What it leaves, in order of measured size:**

1. **Raise reach.** Every point of reach moves a row from the 0.145 regime to the 0.632
   one — a **4.4× median jump in how much of its own oracle the arm can realise**, on the
   largest population this lane owns. This is harvest work, not model work, and it is
   Matt's lane; **it is stated here with its number so the trade is visible**: reach is
   worth more than every model lever this experiment has priced, combined.
2. **A sub-arm on the unreached rows that DO carry a wide-window observation** — 19 – 24 %
   of july and december. It is real and it is small, and january's 3 % means it would be
   priced on one board-share fold out of two. It is the only fittable item on this list.
3. **`LIRF/unreached`** stays what §7cc vii said: 17,730 ranking rows the board can see
   and the instrument cannot, worth +0.0416 s, waiting on july coverage.

**No fit is proposed here** (the desk asked for a design), and nothing above revises a
band or a board reading (score-feedback rule).

**Files.** `work/oracle_gap.py`, `work/oracle_ablate.py`, `analysis/oracle-gap.json`,
`analysis/oracle-ablation.json`, `analysis/no-coverage-unreached.json`,
`logs/oracle-gap.log`, `logs/oracle-ablation.log`.

## §7cf — REPAIR 2, registered before it runs: the fold rows get half the fitting weight

**Desk, 2026-09-10:** *"GO repair 2 now … As registered in my part 2: the target family
fix, priced on the SERVED set against folds-only +0.3930; register bar, failure line and
what a pass changes in RESULTS and commit BEFORE the run; fold-side only, no serve, no
candidate."* And the part-2 wording it refers to: *"repair 2, the target family
(per-month standardised target or a fold-share weight) … the destination that survives
is whether the 3.36x panel can beat folds-only where the arm already serves."*

**No data exists when this is written.** `work/leverA-repair2.sh` and the code are
committed with it and not edited after.

### i. Which of the two, and why — decided before the data

§7cc vi named the defect: *"the fold rows carry `r12` and the extension rows carry a
`base6` residual; the arm learns whichever dominates, and 86 % of the panel is
`base6`'s."* The two fixes the desk offered answer different questions:

* **a per-month standardised target** divides each month's residual by its own scale.
  It changes how much each month counts by `1/σ²`, but it does not touch which head's
  residual a row carries — and scale is not what separates the two families:
  `base6`'s residual rms runs **194.0 – 304.4 s** across the seven extension months and
  **206.96 – 339.14 s** on the three fold months, the same head over the same range. It
  would reweight 2025-06 (304.4) against 2025-04 (194.0) by ×0.41 and leave the 86 %
  exactly where it is. **Rejected.**
* **a fold-share weight** changes exactly the thing §7cc vi named: how loudly the fit
  hears the family that matches the scored fold. **Chosen.**

**The one value, `S = 0.5`**: the fold rows as a block weigh as much as all seven
extension months together — neither family outvotes the other. It is the only value on
the curve that needs no data to justify, and **it is the only value run**: a grid of
`S` scored on these folds would be a selection on them.

### ii. Where `S = 0.5` sits, and why the run is informative either way

The weight defines a one-parameter family whose **two ends are already measured**:

| fold share of the fitting weight | what it is | served-cell board, share-weighted |
| --- | --- | ---: |
| natural (0.121 / 0.137 / 0.228) | every row once — **§7cc** | **+0.3409** |
| **0.5** | **this run** | **?** |
| 1.0 | extension weight zero — **v17** | **+0.3930** |

| scored fold | fold rows | extension rows | weight per fold row | per extension row |
| --- | ---: | ---: | ---: | ---: |
| `july_loo` | 165,677 | 1,205,798 | **×4.139** | ×0.5687 |
| `december` | 190,713 | 1,205,798 | **×3.661** | ×0.5791 |
| `january` | 356,390 | 1,205,798 | **×2.192** | ×0.6478 |

Total weight is held at the row count so the ridge penalty's relative strength does not
move; the column means and spreads used for imputation and standardisation are the
**weighted** ones, so the fit sees the population it is told it sees.

**The prior, registered as a prediction and not as a band.** §7cc vi.3 showed v17's
coefficients were already stable at 165,677 rows (the penalty was 6 × 10⁻⁵ of the data
term), so the extension rows cannot help by reducing variance; they can only change the
direction `d₀`. If the served-set board is monotone in `S`, this run lands **between
+0.3409 and +0.3930 and fails**. A pass needs the curve to rise above its own endpoint —
the extension months carrying information that improves the direction *when they do not
dominate it*. That is precisely the desk's surviving question. **The decisive cell is
`EDDF/reached` on january**: at v17 it read +1261.4 [+335.6, +2500.7], at the natural
share +554.7 [−1500.1, +2414.3], and january chooses `july_loo`'s served set. If it stays
clear of zero, `LTFM/unreached`'s gains under the panel are added to a full served set; if
not, the run fails on condition 2 below whatever the pooled number says.

### iii. The bar, the failure line, and what a pass changes — as §7cd, unchanged

**THE BAR: +0.5518 (`july_loo`) and +0.1932 (`january`), share-weighted +0.3930** —
v17's own cross-fit served set.

**FAILURE** is either of: (1) the served-set board below the bar on **either** board-share
fold; (2) **any** of `EDDF/reached`, `LEMD/reached`, `LTFM/unreached` dropping out of a
board-share fold's cross-fit served set.

**What a pass changes: nothing today.** Fold-side only, no serve, no candidate, and the
desk rules every serve. A pass makes a **candidate for a later day** only if the
**increment over the bar clears the 0.1 s floor** — share-weighted **+0.4930** or more.
Between +0.3930 and +0.4930 it is a pass on the bar and a **hold** on the floor: a ledger
row that answers the desk's question (the panel *can* beat folds-only where the arm
serves) and not a candidate.

### iv. The code, and its inertness

`fit_predict` gains `wf=None`, `arm_predict` gains `fit_weight=None`, `arms.py` gains
`--fold-share` (default 0, weights nothing); every use is behind `if wf is not None` or
`if a.fold_share`. Checked in process on synthetic data against the file as it was:

| | result |
| --- | --- |
| `fit_predict`, default path, old file vs new | **bytes identical** |
| `arm_predict("N", …)`, default path, old vs new | **bytes identical** |
| weighted path with **equal** weights vs unweighted | max difference **0.0** |
| weighted path with unequal weights | moves the prediction (max 30.0 s) |

**Files.** `work/leverA-repair2.sh`, and `--fold-share` / `weighted_moments` in
`code/arms.py`, committed before the run.

### §7cf ii — the read: repair 2 FAILS both conditions, and the curve's whole rise is between S = 0.5 and S = 1

`work/leverA-repair2.sh` ran **14:34:02 → 14:36:43 UTC, 2 min 41 s, rc=0**.
`analysis/arms-step2-leverA-repair2.json`, verdict in `analysis/leverA-repair2-verdict.json`.
Pre-fit assertion first: 0 shared cells and 24/24 registered columns on all three folds.

| fold share of the fitting weight | `july_loo` | `january` | `december` | **share-weighted** |
| --- | ---: | ---: | ---: | ---: |
| natural (§7cc) | +0.4778 | +0.1686 | +0.2339 | +0.3409 |
| **0.5 (this run)** | **+0.4508** | **+0.2065** | +0.2187 | **+0.3426** |
| 1.0 (v17) — **the bar** | **+0.5518** | **+0.1932** | +0.5079 | **+0.3930** |

**FAILURE on both registered conditions.** (1) `july_loo` is **−0.1010 s under its bar**;
(2) **`EDDF/reached` is still out of `july_loo`'s cross-fit served set**, because its
january read is `+835.6 [−757.0, +2458.9]` — between v17's `+1261.4 [+335.6, +2500.7]`
and the natural share's `+554.7 [−1500.1, +2414.3]`, and not clear of zero. That was the
cell §7cf ii named as decisive, and it decided it. Share-weighted **−0.0504 s under the
bar**. No increment question arises.

**`january` alone clears its bar**, +0.2065 against +0.1932 — `LTFM/unreached` holds
+392.1 [+141.2, +681.6] there and the served set keeps all three cells plus
`LEBL/reached`. It is one fold of two and it does not change the verdict.

**The registered prior held.** The run landed **between the two measured ends**, and
barely off the lower one: **+0.3409 → +0.3426 → +0.3930** as the fold share goes natural →
0.5 → 1. **The whole of the rise is between `S = 0.5` and `S = 1`**: giving the seven
extension months half the fitting weight costs the served set as much as giving them
seven-eighths of it. On this measurement the extension months do not carry information
that improves the direction when they stop dominating — they degrade it at any weight
that is not close to zero.

**So the desk's surviving question is answered, at the one value registered: the 3.36×
panel does not beat folds-only where the arm already serves.** With §7cc (natural
share, full amplitude) and §7cd (natural share, out-of-fold amplitude), that is **three
measurements of the same panel against the same bar, and all three fail it** — by
0.0521, 0.0560 and 0.0504 s share-weighted.

**One observation, recorded and not claimed.** At `S = 0.5` the registered cross-fit
machinery admits `LEBL/reached` — a lever A target cell — into `july_loo`'s and
`january`'s served sets. Its own december read is still `+1622.6 [−87.8, +3849.1]`, not
clear of zero, so it does **not** pass the all-folds hold lever A was registered against,
and §7cc's verdict is untouched. It is the same cell §7cc vi.3 found revivable at 1.8
months of scored december days.

**Nothing serves**, no candidate. **The laptop fit slot is released** — one 2 min 41 s
run, as the GO allowed.

**Files.** `work/leverA-repair2.sh`, `analysis/arms-step2-leverA-repair2.json`,
`analysis/leverA-repair2-verdict.json`, `logs/leverA-repair2.log`.

## §7cg — ITEM 2 PRICED on the ranking rows: the screen clears, the band does not

**Desk, 2026-09-10:** *"price item 2 (the sub-arm on unreached rows that carry a
wide-window observation) on the RANKING rows of 2026-07 and 2026-01, not the folds: its
ceiling at the reached-cell realisation (median 0.632), in board seconds against the
0.1 s floor. Register a band only if that ceiling clears the floor; else close it
provisional with the number."*

`work/wide_window_price.py` → `analysis/wide-window-price.json`,
`analysis/wide-window-own-ratio.json`, `analysis/wide-window-band.json`. **No fit.**

### i. The population, on the ranking rows

Complement rows (`~occupied_by_exp_046`, v25 class) that are **unreached** and still
carry at least one wide-window observed column (`No_n_res_apt_30m`, `_rwy_30m`,
`_rwy_60m`, `_rwy_minus_apt`, `No_n_anom_rwy_30m`):

| month | complement | unreached | **with a wide window** | share of unreached |
| --- | ---: | ---: | ---: | ---: |
| 2026-01 | 61,286 | 41,291 | **9,552** | 23.1 % |
| 2026-07 | 87,299 | 62,005 | **10,388** | 16.8 % |

The ranking months carry far more wide-window rows than the folds that price them —
**january has 1,938 in the whole fold** against 9,552 in 2026-01.

### ii. The screen the desk asked for: +0.2869 s — it clears the floor

Per-row ceiling = the oracle `E[rᵢrⱼ]` on that sub-population, per airport, read on the
fold that proxies each month (2026-01 ← `january`, 2026-07 ← `july_loo`), floored at zero,
× 0.632, summed over the ranking rows.

| month | airport | ranking rows | ceiling / row (s²) | priced on | **board s at 0.632** |
| --- | --- | ---: | ---: | --- | ---: |
| 2026-07 | **LFPG** | 7,527 | 9,225.8 | `july_loo` LFPG, 5,336 rows | **+0.2597** |
| 2026-07 | LEMD | 1,238 | 4,977.1 | `july_loo` LEMD, 5,272 rows | +0.0230 |
| 2026-01 | EGLL | 173 | 6,296.0 | `january` EGLL, 1,498 rows | +0.0041 |
| 2026-07 | EGLL | 8 | 3,871.5 | `july_loo` EGLL | +0.0001 |
| | | | | **total** | **+0.2869** |

**It clears the 0.1 s floor, so the desk's rule requires a band, and one is registered
below.** Two things about the screen had to be settled first.

**The pricing rule is the lane's own, and it removes +0.2459 s.** A first pass priced
airports with no fold read on the fold's *pooled* ceiling and read **+0.5318 s**. That is
not this lane's rule: `arms.py::MIN_CELL` says *"a fold cell read on fewer than this is
not priced: its ranking rows carry NO projection."* The largest item it removed is
**2026-01's 6,660 LFPG rows**, which have **zero** wide-window LFPG rows on the january
fold to be priced on — the pooled substitute there was a january pool that is 77 %
EGLL. Unpriced rows are reported (`analysis/wide-window-price.json`) and never summed.
So **2026-01 contributes almost nothing priced**: January does not know what this
population is.

**One airport-month is 91 % of it.** LFPG 2026-07, +0.2597 of +0.2869.

### iii. The borrowed 0.632 is not this population's ratio — measured

0.632 is the in-sample columns ÷ oracle ratio on **reached** cells, where the
observation is inside the 600 s cell. This population's observation is a 30–60 minute
window. Its own ratio, the same statistic on the same rows:

| fold rows | oracle share | columns `R²` | **columns ÷ oracle** | visibility of `o` |
| --- | ---: | ---: | ---: | ---: |
| `july_loo` LFPG, wide | 15.82 % | 1.10 % | **0.070** | 2.48 % |
| `july_loo` LFPG, unreached **without** a wide window | 8.97 % | 0.85 % | 0.095 | 1.73 % |
| `july_loo` LEMD, wide | 15.78 % | 2.80 % | 0.177 | 8.10 % |
| `july_loo` EGLL, wide | 4.66 % | 1.21 % | 0.261 | 4.20 % |
| `january` EGLL, wide | 15.17 % | 2.57 % | 0.169 | 6.03 % |

**At LFPG the wide window buys nothing**: 0.070 with it against 0.095 without. The
population sits in the **unreached** regime (§7ce median 0.145), not the reached one
(0.632). At its own ratio the priced ceiling is **+0.0364 s** — and that is an
**in-sample upper bound**, before any out-of-fold shrinkage or board transfer.

### iv. THE REGISTERED BAND — the lane's recipe, as §7ca ii

Full-oracle ceiling on the priced rows **+0.4540 s**, × this lane's measured out-of-fold
realisation (0.0489 / 0.0716 / 0.0888, §7ca vi) × its two board transfers on this
population (0.70 / mean / 1.03):

| row | board s |
| --- | ---: |
| lowest realisation, lowest transfer | **+0.0155** |
| **mean realisation, mean transfer** | **+0.0281** |
| highest realisation, highest transfer | +0.0415 |
| *the population's own in-sample bound, which caps the top row* | *+0.0364* |

**THE REGISTERED BAND IS +0.0155 … +0.0415 s, MIDDLE +0.0281 — UNDER THE 0.1 s FLOOR AT
EVERY ROW**, and its top row is above what this population's own columns can reach even
in sample. By the upload rule as written, no candidate can come out of this item.

**Registered risk.** 91 % of the value is LFPG 2026-07, and LFPG/unreached on `july_loo`
is the cell where arm N read **−530.4 s² [−1022.7, −86.9]** out of fold (§7cc) — clear of
zero *below*. The wide-window rows are 5,336 of those 19,963.

**The item goes to the provisional table with this band as its number** — a belief with
its evidence, and the desk's to close: the screen passed on a ratio borrowed from a
population it does not resemble, and the band on its own measured ratio sits at a quarter
of the floor.

**Files.** `work/wide_window_price.py`, `analysis/wide-window-price.json`,
`analysis/wide-window-own-ratio.json`, `analysis/wide-window-band.json`.

## §7ch — REACH SCOPED for a decision: the harvest asked every airport-day; the network is 86 % of the darkness

**Desk, 2026-09-10:** *"(b) zero fit: scope item 1 (reach) so I can write it as a decision
for Matt with real costs: on the ranking months, which dark airport-days are dark because
the network did not hear them and which because the harvest did not ask; what source or
message type inside docs/input-rule.md could raise reach; what a harvest would cost. No
harvest launch; any input outside the input contract is Matt (supplied-input rule). One line per item."*

`work/reach_scope.py` → `analysis/reach-scope.json`; `work/reach_arrdep.py` →
`analysis/reach-arrdep.json`. **No fit, no harvest, no download, no fit slot.** The matcher
is `exp-046-surfacefit`'s, imported read-only through `block.matched` as the emit does
(sha `122c7d84…`). On both ranking months it gives the registered block's clock on every
row: 152,719 / 152,719 and 192,122 / 192,122.

### The three lines

1. **Not asked: none.** All 620 ranking airport-days were searched, with every archive
   digest verified. Of the **103,296** unreached complement rows, **89,235 (86 %) are the
   network's**: LTFM on all 62 days and LIRF on 29 July days are silent on the surface
   (62,398), the EDDF outage is held (1,557), and in 25,280 rows the surface was heard
   that day but no one in the row's cell was heard leaving a stand (LFPG 19,924). The
   harvest's own loss is at most **14,061** rows: clocked departure segments placed on no
   row. Without arrival-roll artefacts it is at most **4,210**.
2. **Inside `docs/input-rule.md` nothing raises reach.** The file's only departure
   off-block is `AOBT_3_flt`, which is arm N's `Nf_` family. It already covers every cell,
   and on reached cells it sees a fifth of what `No_` sees. Arrival rows carry no departure
   off-block. adsb.lol prod-0 is read in full, and mlatonly-0 has no ground point in any
   box. The source that could add reach is **OpenSky Trino**, which is open only to
   university and government researchers. So it is **outside the rule as written, and
   Matt's (supplied-input rule)**. Its surface coverage at LTFM, LIRF and LFPG is unmeasured.
3. **Cost.** Re-running the adsb.lol harvest costs **6.5 kernel hours and 214 GB** and adds
   no reach, because it is the same network. A matcher change adds at most 4,210 rows,
   worth **0.025 – 0.092 s**, which is under the floor. OpenSky needs a supplied-input rule ruling and an
   account first. After that it needs hundreds of MB and about 1 – 2 agent days (my
   estimate). If it heard like `EDDF/reached`, it could win **0.44 – 2.01 s** on the silent
   rows net of v17, plus **0.16 – 0.59 s** on the unheard rows.

### i. The classes, on the ranking rows

Population: the complement rows (v25 `~occupied_by_exp_046`) that are **unreached**
(`No_n_n_cell` = 0). That is 41,291 in 2026-01 and 62,005 in 2026-07. Each row is in exactly
one class, and the classes are tested in the order shown:

| class | test | 2026-01 | 2026-07 | **total** | where |
| --- | --- | ---: | ---: | ---: | --- |
| not asked | no run, a digest not verified, a run error, or the box not searched | 0 | 0 | **0** | 620 / 620 airport-days searched |
| held | the desk's declared feeder outage | 1,557 | 0 | **1,557** | EDDF 2026-01-29 … 31 |
| silent | aircraft heard in the box, under 5 % of them with one ground point | 22,396 | 40,002 | **62,398** | LTFM 62 days (47,581); LIRF 29 July days (14,817) |
| unheard | surface heard that day; no clocked departure segment in the row's cell | 12,167 | 13,113 | **25,280** | LFPG 19,924; LIRF 2,649; EDDF 1,187; LEMD 941; five others 579 |
| unplaced | a clocked departure segment in the cell, take-off agreeing to 60 s, placed on no row | 5,171 | 8,890 | **14,061** | LFPG 12,374; EDDF 948; LEMD 365; LIRF 264; others 110 |
| *of which not arrival-roll artefacts* | | *1,622* | *2,588* | ***4,210*** | *LFPG 3,317; EDDF 558; LEMD 135; LIRF 117; others 83* |

**Network: held + silent + unheard = 89,235 (86.4 %). The harvest's own loss is at most 14,061
(13.6 %), and at most 4,210 (4.1 %) without the artefacts.** Both harvest figures are upper
bounds. At a busy airport a 60 s agreement with some ranking take-off happens by chance.
And an *arrival-roll artefact* is an inbound callsign whose "departure" segment starts
within 10 minutes of that arrival's landing: 3,672 of the 8,207 agreeing unplaced segments
in January and 6,587 of 11,929 in July. The matcher's turnaround pass declined these
correctly (its take-off gate: 3,403 and 4,850 declined).

The segment ledger is 101,144 and 118,548 clocked departure segments in the months, of
which 89,897 and 103,327 are placed. A `silent` airport-day is not ambiguous. On
2026-07-15 LTFM's box held 465 aircraft and **1** had a ground point. LIRF's box held 428
aircraft and **0** had a ground point. EHAM, the control, had 590 of 714.

### ii. The network hears the arrivals: it is the departure side it misses

This uses the harvest's census window (`exp-045-harvest/code/census.py`: callsign on the
ground at its airport, last ground point in [MVT − 6 h, MVT + 1 h]). It is applied to
**both** phases of the ranking file, on all 62 days:

| airport | 2026-01 departures | 2026-01 arrivals | 2026-07 departures | 2026-07 arrivals |
| --- | ---: | ---: | ---: | ---: |
| **LFPG** | **8.5 %** | **52.0 %** | **7.2 %** | **59.3 %** |
| **EDDF** | **58.5 %** | **90.3 %** | **43.0 %** | **89.2 %** |
| **LEMD** | 96.0 % | 95.8 % | **60.0 %** | **83.5 %** |
| LIRF | 65.4 % | 76.7 % | 0.1 % | 0.4 % |
| LTFM | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| EHAM · EDDM · LEBL · LSZH (control) | 92 – 96 % | 93 – 96 % | 93 – 96 % | 94 – 96 % |

The network, the box and the days are the same for both columns. It is not the archive,
because no dark window is a short archive (`exp-045` §3). It is not the box (`exp-045`
§2). **At LFPG, EDDF and LEMD-July the network hears the aircraft on the ground after
landing, and misses them before take-off.** No filter on the same archive changes that.
At LTFM and LIRF-July both sides are silent. That is the surface of the whole airport, as
`exp-045` §1 found for Istanbul at every altitude.

### iii. What could raise reach, inside the input rule and outside it

**Inside `docs/input-rule.md`:**

- **The ranking file.** Its only departure off-block is `AOBT_3_flt`. That is arm N's
  `Nf_` family, with coverage 0.95 – 1.00 on every cell, unreached ones included (§7ce iii).
  On reached cells it sees about a fifth of what the observation sees: 5.37 % against
  42.31 % visibility on `EDDF/reached` july. The **344,693 arrival rows** have full actuals
  (landing, in-block, taxi-in, stand, runway on 100 %), but they carry no departure
  off-block. Their taxi-in stream and stand chain are already in the base head (§ design
  reads, "what the base head already reads"). The file-state family, arrival stream
  included, removed +2.6 / +0.0 / −2.0 % of the shared component at the off-block anchor
  (§3 headline).
- **adsb.lol**, ODbL-1.0 / CC0 with no account. prod-0 is what the harvest reads: every
  trace and every point, with the ten-box test done after the parse. mlatonly-0 was probed
  on 2026-01-02 (`exp-045` §1): it has 10 traces through the LFPG box, none below 500 ft,
  none through LTFM, and **no on-ground point in any of the ten boxes**. staging-0 is not
  probed. I make no claim for it.
- **Message type.** The archive's ground points are the ADS-B surface-position reports,
  and the harvest reads all of them. Table ii shows that the same airports' reports *are*
  received, from arrivals. What is missing is reception of the departure side, and no
  message type in the same archive is received where these are not.
- The **2022 PRC/OpenSky archive** (4TU, CC BY 4.0) fails the calendar (`exp-042` §2.3,
  row 4).

**Outside the rule as written — Matt's (supplied-input rule):** **OpenSky Trino** (`state_vectors_data4`,
1 Hz, `onground`, server-side bounding box). Its access policy is "university-affiliated
researchers, governmental organisations, and aviation authorities", with applications
reviewed. `exp-042` §2.3 fails it on the open-licence clause as written. §5 there notes
that supplied-input rule's current text does not cover it, and that the question belongs in the same
Discord message. **Whether OpenSky hears LTFM, LIRF-July or the LFPG departure side on the
surface is unmeasured.** A probe needs the account. ADS-B Exchange (paid) and FR24 /
FlightAware (proprietary) fail outright.

### iv. What each route costs, and what it could win

Upside recipe. Take the priced full-oracle ceiling of the class's ranking rows: the
per-airport `E[rᵢrⱼ]` of unreached complement rows on the proxy fold, floored at zero, and
priced only at 500 or more fold rows (the lane's MIN_CELL). Multiply it by **the served
reached cell's out-of-fold realisation**. For `EDDF/reached` that is +1,261 / +3,003 /
+3,620 s² per row against an oracle of 7,701 / 9,264 / 6,026, which gives **0.164 / 0.324 /
0.601** (january / july / december). No board transfer is applied; this lane's transfers
are 0.70 and 1.03. The silent class is given **net of v17**, because `LTFM/unreached`
already serves off the file clock at +264 / +756 s² per row, which is +0.148 s:

| route | kind | cost | reach it adds | upside, fold-projected s (low / mid / high) |
| --- | --- | --- | --- | --- |
| re-run the adsb.lol harvest | data | 6.5 kernel h and 214 GB for the two ranking months (42.2 h and 1,320 GB for all fourteen, `exp-045` cost table) | **none**: same network | 0 |
| matcher change (`exp-046`'s code, not this lane's) | code | no data; `exp-046`'s two placebo bars, two ranking re-emits, a class-vector version, fold re-reads | ≤ 4,210 rows (≤ 14,061 with artefacts) | **0.025 / 0.050 / 0.092** (≤ 0.091 / 0.180 / 0.333 with artefacts) |
| OpenSky Trino | data and a rules ruling | Matt's supplied-input rule ruling and account approval; then hundreds of MB (`exp-042`), the rule adapted to 1 Hz state vectors, the matcher, and the blocks rebuilt — **about 1 – 2 agent days after approval, my estimate** | unmeasured | if it heard like `EDDF/reached`: silent **0.44 / 1.01 / 2.01** net of v17, unheard **0.16 / 0.32 / 0.59** |

Two rows are not in the sums, by the lane's rule. **LIRF-July's 14,817 silent rows are
unpriced**: `july_loo`'s LIRF has 152 unreached complement rows, because July 2025 was
90.6 % receipt. At january's LIRF oracle they would add 0.506 s at the full oracle and
0.083 / 0.164 / 0.304 s at the three realisations. **The held 1,557 are unpriced**:
january's EDDF has 128 such rows.

**The largest number in the table is a projection on an unmeasured coverage.** It assumes
that a network heard LTFM's surface as well as adsb.lol hears Frankfurt's reached cells.
`exp-045` §1 measured the opposite for adsb.lol at every altitude over Istanbul. The figure
is the prize if the coverage exists, not a forecast that it does.

**Files.** `work/reach_scope.py`, `work/reach_arrdep.py`, `analysis/reach-scope.json`,
`analysis/reach-arrdep.json`.

### v. BRANCH REPORT — reach (item 1 of §7ce v), a belief for the provisional-closes table

**Desk, 2026-09-10, under the absence grant:** *"do not re-run adsb.lol, do not spend the
slot on the below-floor matcher change, and keep OpenSky parked under supplied-input rule. Finish §7ch with a
branch report and the strongest revival condition, then report done. Do not launch a fit."*

This is a belief with its evidence. A worker closes nothing; the desk carries it to Matt's
table.

**The negative and its cause.** Inside the input contract, nothing this lane can build
raises reach. The cause is measured, not assumed:

- 86 % of the unreached complement rows are dark because the network did not hear them (i).
- The harvest asked every airport-day (i).
- The ranking file holds no second departure off-block time (iii).
- What the harvest could still recover is at most 4,210 clean rows. It is worth 0.025 –
  0.092 s, under the floor at every realisation (iv).

**Destination value, stated apart from the path.** The destination is real. If the dark
rows were reached as `EDDF/reached` is, the silent rows would be worth **0.44 – 2.01 s** net
of v17, and the unheard rows **0.16 – 0.59 s** (iv). LIRF-July would add 0.08 – 0.30 s,
which is not in the sums. This is the largest unrealised value the lane has priced.

**Path feasibility.** Inside the rule there is no path (iii). Outside it there is one
candidate, OpenSky Trino. It is blocked on the licence clause (supplied-input rule), and its surface
coverage is unmeasured. Every other source fails outright.

**The honest repair attempts, all made before this report:**

1. The MLAT archive (`exp-045` §1) has no on-ground point in any of the ten boxes.
2. The turnaround pass in `exp-046`'s matcher already places 1,382 and 1,797 rows in the
   two ranking months from inbound identities. Its take-off gate declines the arrival-roll
   artefacts.
3. The wider observed windows (§7cg, item 2) give a band of +0.0155 … +0.0415 s on their
   own measured ratio, under the floor.
4. The re-identification count in this section is at most 4,210 clean rows, under the
   floor (iv).

**The strongest refinement left.** A one-week OpenSky probe of the LTFM surface and the
LFPG departure side, once supplied-input rule allows it. It is the smallest measurement that opens the route
or kills it, and it needs no fit. Cost: an account, one bounding-box query per hour
partition, hundreds of MB, and about 3 – 5 agent hours (`exp-042` §3's pilot estimate).

**What revives it — the strongest condition.** Both of these are needed:

1. A supplied-input rule ruling that approval-gated OpenSky access counts as "openly accessible", or an open
   archive that has the coverage.
2. A probe that hears **at least 30 % of LTFM's departures on the ground, with a stand
   clock**.

30 % is the lowest receipt at which the lane's receipt-response curve clears zero. In T4
(EHAM's clocks thinned at random), the `No` arm removes 28.1 % [17.1, 37.5] of the shared
component, which is 0.384 of its value at 96 %. At that receipt the net LTFM increment is
**0.17 / 0.39 / 0.77 s** fold-projected at the three realisations. That is above the floor
at every realisation. At 16 % the curve is not clear of zero ([−3.5, 11.8]), and the
increment would be 0.03 / 0.07 / 0.13 s. So a smaller receipt cannot revive the route by
this arithmetic. The same 30 % test applies to the LFPG departure side, which carries most
of the unheard rows.

**For the table:** reach is closed-provisional. The negative is the network's (86 % of the
dark rows). The destination is worth up to 2 s. The path is outside the rule and is parked
under supplied-input rule. The revival is a supplied-input rule ruling plus at least 30 % LTFM surface receipt on a probe.

**The write-once guard, added late.** I committed this section's two scripts in `428eb44`
without the standing guard (`exp-046-surfacefit/code/nooverwrite.py`, desk rule of
2026-09-10). This commit adds the guard, called after argument parsing and again at the
write. The helper is loaded by path under its own module name, so the sibling's
`block`/`arms` cannot shadow this lane's modules. Run against the committed artifacts, both
scripts refuse before any compute (exit 1), and the artifacts are unchanged.
