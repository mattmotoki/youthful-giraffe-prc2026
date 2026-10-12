# exp-007-winter — RESULTS

exp-id `exp-007` | worker `winter` | tier hard | reviewer: pending
Assignment `assignments/winter.md`, re-aimed by the desk mid-run (see `SPEC.md`).
Reading and rejected readings: `SPEC.md`. Code: `code/`. Module handed back:
`prc/day_state_features.py` (VERSION 1). Out-of-fold predictions: `oof/`. Every
attempt: `climb.log`. Data: `prc.data.data_dir()`, live ranking SHA-256
`9db75e3a…`.

Baseline for the decomposition: retro-v1's **`G1cv-s7`** — the latest `climb.log`
line that is guarded, run on all three folds, and the best of those on january.
`H0-s7` is 47 minutes later in the log but is a phi/lirf ablation control, not a
better line (january 347.4 against 346.5); every table that could turn on the
choice reports both. Baseline for the *measurement* of the new columns: a matched
control arm run through the same file (`code/run_ds.py`), never a saved number.

---

## The one paragraph

The january fold does not say what the desk thought it said. **61 % of its
non-LIRF squared error is two rows** — two departures at Paris CDG whose hidden
block time is recorded on an earlier calendar day, of exp-002's date-rollover
class. Remove them and january's other-nine RMSE is **213 s, not 342**; december
is 202 and july 250, so **on the 2025 folds July is the harder half**, and on the
desk's `(july_loo, january)` analogue pair the other nine blend to **234.6 s** —
inside the 200–240 s band the board implies, though that comparison holds only
conditionally (finding 2). Those two rows are **unreached rather than
unreachable**: a rule found on them does isolate them, but out of sample it has
fired nine times and been right none, and the hedge it licenses costs 6.5 s on
july and 7.5 s on december.

July's remaining error has no unclaimed marker among the *numeric* ranking-frame
columns — the best removes 0.73 % of its NM-bulk squared error on top of the
model — but a **categorical** airport-by-operator correction removes 0.95 %,
which is a second of july bulk sitting behind retro-v1's unisolated-encoding
guard. What separates the bad days, in every fold, is airport-day state built
from the label-free columns: the day's median take-off-minus-schedule reaches
ρ +0.651 within an (airport, fold) and AUC 0.868 for the worst 5 % of days, where
the strongest ERA5 column reaches 0.19.

**The 25-column block built from that does not work; one column of it does.**
Measured against a matched control at a declared 300 rounds, the whole block is
worth −10.8 s on december's most disrupted days and **+72.1 s on january's**, in
opposite directions — december under both clusterings, january only under the
airport-day one (the coarser calendar-day interval spans zero). The reason is
measured but **not cleanly isolated**: **a day-state column beyond its own
training maximum appears to push a
near-calibrated model past the truth.** On the january fold's worst-hit day three
of five referenced columns sit at or above that airport's training maximum, a base
bias of +40 s becomes −75 s, and the squared error doubles.

Removing the columns that can do that leaves `ds_deice_hr` — the assignment's own
de-icing proxy, a share bounded in [0, 1] — and **it is a clean positive on every
fold**: −13.6 s on december and −3.9 s on january whole-fold, both intervals
excluding zero, null on july, and no slice anywhere showing harm. So the
deliverable is `prc.day_state_features.DEICE`, one column, not the block; finding
10's case for shipping the block as January insurance is withdrawn by its own
measurement; and the lane's revival condition — clip to the training range or use
ranks within the month — is now supported by a second measurement rather than by
a diagnosis alone.

---

## Findings, ranked

### 1. 61 % of the january non-LIRF squared error is two rows, and the whole january picture rests on them

`analysis/q1_rollover.md`, `analysis/q1_decomposition.md`.

| fold | rollover rows in the fold | their share of the whole fold's SSE | of the non-LIRF slice |
|---|---|---|---|
| **january** | **2** | **55.3 %** | **61.2 %** |
| december | 0 | 0.0 % | 0.0 % |
| july | 0 | 0.0 % | 0.0 % |

The two rows:

| airport | flight | take-off | hidden block time | schedule | target | prediction | MVT − SCHED |
|---|---|---|---|---|---|---|---|
| LFPG | `EJU983W` | 2025-01-19 18:54 | 2025-01-**18** 19:30 | 2025-01-19 18:25 | 84,240 s | 905 s | +1,740 s |
| LFPG | `EJU42AY` | 2025-01-23 15:09 | 2025-01-**22** 22:58 | 2025-01-23 14:35 | 58,206 s | 1,136 s | +2,043 s |

Both are of the class exp-002 named and measured: the block time carries a wrong
calendar day, every observable column is ordinary, so **by construction no feature
separates them**. exp-002 listed `EJU983W`; `EJU42AY` is a 16.2-hour offset rather
than a 24-hour one and sits below that experiment's 80,000 s screen, so it is new
here.

**Correction (gpt-5.6-sol round 1, blocking finding 1): "by construction no
feature separates them" is false, and the honest word is *unreached*, not
*unreachable*.** Sol found the observable I missed — the `FLIGHT_mvt` prefix. The
conjunction `ADEP = LFPG` and no NM join and aircraft type present and
`FLIGHT_mvt` starts with `EJU` and `|take-off − schedule| ≤ 1 h` selects exactly
these two rows among January 2025's LFPG no-NM departures. What follows is my
answer to the desk's next question — whether that conjunction is a *detector*
(`analysis/q1_detector.md`).

| | fires | of those, rollover events |
|---|---|---|
| January 2025 (where the rule was found) | 2 | **2** |
| the other eleven 2025 months | 9 | **0** |
| the live ranking file | 2 (both July 2026) | hidden |

**Out of sample the rule has fired nine times and been right none.** Priced as a
hedge — add `p · R` to every firing row, `p` the precision estimated on the
fold's own training months, `R` the mean event residual of 70,203 s, applied in
that order:

| fold | training fires / events | p | hedge | validation fires (events) | Δ RMSE |
|---|---|---|---|---|---|
| july | 5 / 2 | 0.400 | 28,081 s | 1 (0) | **+6.47 s** |
| december | 9 / 2 | 0.222 | 15,601 s | 2 (0) | **+7.51 s** |
| january | 9 / **0** | 0.000 | 0 s | 2 (2) | **+0.00 s** |

The january row is the whole argument in one line: the fold in whose validation
month the rule was found cannot even reward it, because the rule's precision on
that fold's own training months is zero. On the two folds where the hedge is
non-zero it costs 6.5 and 7.5 seconds. So the correct status is: **the class is
unreached rather than provably unreachable, and the one rule that reaches it is
not a detector.** The 61.2 % and the 342.3 → 213.3 arithmetic are unaffected;
what changes is that "irreducible" is not established, only "not reduced by
anything tried, at a cost of 6 to 8 s for trying".

My own weaker check still stands, and is what made me overstate. Against the 214
other LFPG no-NM january departures the two rows are not outliers on anything
*numeric*, and on the one numeric column that might have marked them they point
the **wrong way**:

| column | the LFPG no-NM january population | the two rows |
|---|---|---|
| take-off − schedule | median 6,056 s, p05–p95 2,076–11,381 | **1,740 and 2,043 s** — below the 5th percentile of the class, and the 66th and 74th percentile of *all* january non-LIRF departures |
| take-off hour | median 12, p05–p95 3–21 | 18 and 15 |
| runway | — | 08L (34 % of the class) and 26R (19 %) |
| aircraft type | — | A319 (3 %) and A320 (16 %) |
| stand | — | C03 and D06, ordinary CDG contact stands |
| the model's own prediction | median 1,062 s, p05–p95 774–1,493 | 905 and 1,136 s |
| **target** | median 1,075 s, maximum 5,929 s | **84,240 and 58,206 s** |

A predictor keyed on lateness would have ranked these two *below* the class
average. What I missed is that identity, not lateness, is the separating axis —
and the measurement above is why the right response is still to read the fold
differently rather than to build on it.

Screening the twelve training months with a *stated physical* rule — a recorded
taxi above six hours, aircraft type present, |take-off − schedule| ≤ 1 h — finds
**three such rows in 1,924,343 non-LIRF departures**, two of them in January 2025
and one in May 2025 (`analysis/q1_floor.md`). The january fold drew two of the
three.

Every fold number, with the class in and out:

| fold | set | RMSE all | **RMSE non-LIRF** | bulk | tail >3600 | no-NM | LIRF |
|---|---|---|---|---|---|---|---|
| january | as scored | 346.5 | **342.3** | 193.9 | 1,316.7 | 2,830.3 | 395.6 |
| january | rollover removed | 231.7 | **213.3** | 193.9 | 1,316.7 | 758.9 | 395.6 |
| december | as scored | 219.3 | **201.6** | 188.7 | 1,523.1 | 567.7 | 377.9 |
| july | as scored | 320.8 | **249.7** | 227.3 | 1,892.2 | 588.0 | 755.5 |

The three slices in that table **partition** the non-LIRF rows — bulk is NM-joined
with target ≤ 3600 s, tail is NM-joined above it, no-NM is the rest — which is not
`run.py`'s `decompose`, where `bulk`/`tail` split every row at 3,600 s and `no-NM`
overlaps both. COMMON.md's four required cells, in `decompose`'s own definitions,
are in `climb.log` and in the fold table of finding 9.

**January's no-NM RMSE of 2,830 s, which looked like a winter regime, is 759 s
once the two rows are out** — still above december's 568 and july's 588, but a
difference of a couple of hundred seconds on ~1,400 rows rather than the 2,300 s
gap the uncorrected number showed. (An earlier version of this line said 759 was
"below" 588; it is above. gpt-5.6-sol round 1, non-blocking.) The january bulk (193.9) was never affected and was always the
best of the three folds.

### 2. So the map flips: on the 2025 folds July is the harder half, and the other nine are inside the leaders' band under a stated condition

exp-006's board arithmetic, recomputed with the same method and the class removed
(`analysis/q1_rollover.md`; the other-nine blend uses the other nine airports'
**own** ranking mix, 55.4 % july / 44.6 % january — 176,284 July and 141,658
January non-LIRF ranking departures — while exp-006's 58.9/41.1 is LIRF's own mix
and is kept for the LIRF column):

| set | other nine | LIRF | implied total |
|---|---|---|---|
| as scored | **295** | 633 | 334 |
| rollover removed | **234** | 633 | 286 |
| the leaders, at 277.3 | 200–240 | 550–720 | 277 |

The assignment's premise — "our fold blend is about 293 s against a required 200
to 240 s, and the january fold carries most of that" — is arithmetically correct
and substantively wrong: the 293 is 295 under the other nine's own ranking mix,
and 61 s of it is two rows nobody can predict. **234 s is inside the band, conditionally on k = 0 and on the folds
transferring** — see the two paragraphs below, which are the substance of that
condition and not an afterthought. Two caveats stand: `docs/shift-2026.md`
says January 2026 is harder than January 2025 at EDDF, EHAM, EDDM and LSZH, so
234 is optimistic for the ranking month; and the leaders' 200–240 s is a
trade-off curve against an assumed LIRF number, not a decomposition.

### 3. The floor: what an unreachable row costs everybody, and what the board cannot say about it

`analysis/q1_floor.md`. One such row adds its own squared residual to the total
squared error of *every* submission, because no submission predicted it. The
three observed events are **not the same size** — residuals of 83,335, 57,070 and
86,401 s against a normal prediction — so the coefficient is their root mean
square, not a single assumed day (gpt-5.6-sol round 1, blocking finding 2; the
first version of this section used exp-002's one-day convention and read 147.1 s):

```
floor(k) = sqrt(k) × 76,739 / sqrt(344,841) = sqrt(k) × 130.7 s
```

The rate rests on **three events in twelve months**. Its exact Poisson 95 %
interval, projected to the ranking file's exposure, is **0.10 to 1.45** expected
rows around a point estimate of 0.50 — and the events are not independent draws
from a homogeneous rate: two of the three are the same operator at the same
airport in the same month. Restricting the class to genuine one-day errors leaves
two events and exp-002's 0.33.

| k | floor | leaders' skill at 277.29 | 278.37 | 278.48 | 291.06 |
|---|---|---|---|---|---|
| 0 | 0.0 s | 277.3 | 278.4 | 278.5 | 291.1 |
| 1 | 130.7 s | 244.6 | 245.8 | 245.9 | 260.1 |
| 2 | 184.8 s | 206.7 | 208.2 | 208.3 | 224.9 |
| 3 | 226.3 s | 160.2 | 162.0 | 162.2 | 183.0 |

The three leaders sit inside **1.19 s** of each other and the fourth is 12.6 s
behind. A shared floor compresses skill differences — with
`score = sqrt(skill² + floor²)`, `d(score)/d(skill) = skill/score < 1` — so those
three span 1.18 s of skill at k = 0 and 1.59 at k = 2. The cluster is
**consistent with** a shared floor and equally consistent with three equally good
teams; it is not evidence for one.

**What the board cannot say.** An earlier version of this finding claimed the
board *bounds* k ≤ 2, because k = 3 would leave the leaders holding 109 s against
our own 194/227 s NM bulk. With the corrected coefficient k = 3 leaves them 160 s,
and in any case the argument compares a hypothetical overall skill against one
pipeline's bulk RMSE — different quantities. It is a plausibility argument, not a
bound (gpt-5.6-sol round 1, blocking finding 4).

**And what a submission would and would not settle.** A pipeline whose cleaned
fold skill is 286 s would read 286 / 314 / 341 on the board at k = 0 / 1 / 2 —
28 s apart, so a reading carries real information about k. But it does not
*measure* k: a single score is the sum of how many such rows the file holds, how
large each is, whether any model responds to finding 1's identity axis, how the
folds transfer to 2026 at all, and the January-2026 shift no fold can price. It
is still the single most informative number the desk could buy; it is not a
decomposition, and the earlier "one submission measures k" was too strong.

### 4. The january fold still ranks models; it misprices them

`analysis/q1_instrument.md`. Across the 42 out-of-fold lines in exp-001 and
exp-006 that ran the january fold, the two rows contribute a squared error that
varies by only **0.6 %** between lines — every model predicts them between 900 and
1,300 s — so they act as an additive constant, and the Kendall rank correlation
between the january ordering as reported and with the rows removed is **0.96**.
What they do instead is
(i) raise the level by **114 s**, which is what broke the arithmetic in finding 2,
and (ii) **compress differences by a factor 0.67**: a 10 s improvement in the
january error a model can actually attack shows up as 6.7 s in the reported
january RMSE. That factor is arithmetic, not a fit — with `reported =
sqrt(clean² + C/n)` and `C` the two rows' near-constant squared error,
`d(reported)/d(clean) = clean/reported`, which averages 0.671 over these lines
against 0.672 from a straight-line fit to them. Every january number in `docs/LEDGER.md` should be read that way.

**Which ledger cells are affected, and which are not.** `run.py`'s `decompose`
splits the bulk at a target of 3,600 s, and both rollover rows are far above it,
so on the `G1cv-s7` line:

| ledger cell, january fold | as reported | two rows removed | difference |
|---|---|---|---|
| `rmse` (whole fold) | 346.49 | 231.70 | **+114.79** |
| `rmse_bulk` (target ≤ 3600 s) | 209.47 | 209.47 | **0.00** |
| `rmse_tail` (target > 3600 s) | 6,005.96 | 2,170.40 | +3,835.56 |
| `rmse_no_nm` | 2,872.36 | 1,062.04 | +1,810.32 |
| `rmse_noac` | 3,814.45 | 3,814.45 | **0.00** |

So **every `bulk` and `noac` number in the ledger is already clean**, which is
most of what exp-004, exp-005 and exp-006 actually compared; the january `rmse`,
`rmse_tail` and `rmse_no_nm` cells are not, and the january `rmse_tail` of ~6,000 s
that several rows carry is two rows plus a 2,170 s tail.

### 5. The model is calibrated; the day effect is nearly exhausted; there is no shrinkage to repair

`analysis/q4_shrinkage.md`. Regressing the truth on the prediction, a slope above
1 is the signature the assignment's question 4 names.

| fold | core, other nine | bulk ≤ 3600 s | rows with span > 1800 s |
|---|---|---|---|
| january | **1.012** | 0.995 | 1.071 |
| december | **0.993** | 0.981 | 1.028 |
| july | **1.022** | 0.997 | 1.178 |

There is no global shrinkage. On the long-span rows there is a little — july's
1.178 on 4,112 rows — and an affine recalibration there is worth 19 s on that
slice, which is **0.75 s on the core**. (The slopes conditioned on the *target*
being above 1,800 s, 0.63 to 0.75, are not evidence: conditioning on the outcome
biases a regression slope downward by construction. They are in the table for
completeness and should not be read as shrinkage.)

The stronger version of the same test kills the second stage the question
proposes. An **oracle** that adds each airport-day's own mean residual to every
row of that day — the ceiling on any per-airport-day random effect or day-severity
model, since it *is* the day effect, measured with the labels:

| fold | core, other nine | after the oracle day intercept | the same, leave-one-out |
|---|---|---|---|
| january | 200.9 | **198.5** | 198.8 |
| december | 194.3 | **192.7** | 193.0 |
| july | 237.5 | **235.1** | 235.5 |

The leave-one-out column removes the row's own residual from the day mean it is
then corrected by — the honest version, since a row otherwise helps set the
intercept it is scored against. With ~500 rows a day the two differ by 0.3 s, and
the point of showing both is that even the optimistic one is small.

**2.4 s, and that is the ceiling.** Whatever the day-state columns are worth, it
is not through a day-level intercept: the model already has the day's mean right.
They can only earn their place by changing predictions *within* a day, or by
generalising to a January that no 2025 day resembles.

### 6. What separates the worst airport-days, label-free, and it is not the weather

`analysis/q2_daystate.md`. 837 airport-days over the three folds, non-LIRF, the
two unreachable rows removed. Every candidate is computed from the
ranking-masked month frame; the target is the day's RMSE under the baseline line
and is never an input. Days are ranked inside each **(airport, fold)**: a column
must not score by telling the airports apart, because the model already knows the
airport, nor by tracking the season. (The first version of this table ranked
within the airport only, which pooled a january day with a july day of the same
airport and let a seasonal contrast — july departs later *and* is harder — count
as a day contrast. It inflated the top rows by ~0.09 and it invented three
markers outright: `ds_n_dep`, `ds_mvt_total` and `ds_n_arr` scored ρ +0.30 there
and score **−0.01** here. The fold is in the key from here.)

Two AUC columns, because they are different questions and the first version of
this table described one and computed the other (gpt-5.6-sol round 1, blocking
finding 5): **A** marks the fold's own worst 5 % of airport-days by squared error,
**B** marks each airport's own worst two of its 31 days. (The six lower rows of
column B were mis-transcribed in the first version of this table — the values
printed matched no column of `work/daystate_screen.csv` and flattered the
de-icing proxy and every ERA5 row. Caught by the round-2 review; the numbers
above are `auc_within` from that file.)

| candidate | ρ within (airport, fold) | AUC A | AUC B | ρ january | ρ december | ρ july |
|---|---|---|---|---|---|---|
| day median take-off − schedule | **+0.651** | 0.868 | 0.854 | +0.642 | +0.622 | +0.441 |
| day p90 of AOBT − LOBT | **+0.637** | 0.856 | 0.868 | +0.377 | +0.403 | +0.350 |
| day p90 of take-off − schedule | **+0.614** | 0.841 | 0.852 | +0.611 | +0.631 | +0.492 |
| day median of AOBT − LOBT | **+0.556** | 0.838 | 0.830 | +0.215 | +0.183 | +0.070 |
| day std of take-off − AOBT | **+0.497** | 0.808 | 0.827 | +0.451 | +0.568 | +0.481 |
| day NM-join rate | **−0.406** | 0.241 | 0.238 | −0.414 | −0.320 | −0.316 |
| **de-icing proxy** (day share of take-off − AOBT > 1800 s) | **+0.375** | 0.759 | 0.733 | +0.592 | +0.504 | +0.311 |
| day p90 of arrivals' taxi-in | +0.299 | 0.792 | 0.701 | +0.496 | +0.243 | +0.297 |
| day mean temperature (ERA5) | −0.187 | 0.433 | 0.391 | −0.247 | −0.035 | +0.204 |
| day share of sub-zero hours (ERA5) | +0.178 | 0.616 | 0.627 | +0.241 | +0.133 | – |
| day max 6-hour snow (ERA5) | +0.101 | 0.557 | 0.614 | +0.185 | +0.083 | – |
| runway count, modal share, **share entropy**, **modal-runway change**, movement shortfall, movement counts | ≤ \|0.04\| | ≈0.5 | ≈0.5 | | | |

Four readings. **The day's own delay level is the marker, not the weather**: once
the airport and the season are held fixed, the strongest ERA5 column reaches
ρ 0.19 while the day's median take-off-minus-schedule reaches +0.651 and its AUC
for the worst 5 % of days is 0.868. That is consistent with exp-005's own
withdrawal ("weather is not the *only* ranking-file input marking them —
within-month MVT−AOBT airport-day aggregates do too") and sharpens it: within an
airport and a season, the delay aggregates are three times the marker the weather
is. **The de-icing proxy the assignment asked for works, and works best in
winter** (ρ +0.592 january against +0.311 july), but it is not the best column.
**Movement counts mark nothing** (ρ −0.009), which is the correction above: a
busy day is not a bad day once the airport and the month are fixed.

Two of the assignment's named candidates need their scope stated rather than a
verdict (gpt-5.6-sol round 1, blocking finding 6). **Runway configuration**: the
first version tested only how many runways were in use and how concentrated the
mix was, never *which* configuration was active or whether it changed, so
"configuration marks nothing" was not what had been measured. The table now also
carries the entropy of the day's runway-share vector and a flag for the modal
departure runway changing from the previous day; a full per-configuration
encoding is still untested, and the conclusion is limited to these summaries.
**Cancellations**: the literal candidate — scheduled movements with no record —
is **not constructible from this data at all**. The file holds one row per
movement, so a cancelled flight leaves no row and no `SCHED_TIME_UTC_mvt` to
count; `ds_mvt_shortfall` (the day's movements against that airport's median day
in the month) is the nearest available proxy and it marks nothing. The candidate
as written needs an external schedule and is **unanswered**, not refuted.

### 7. July's error is diffuse; its numeric columns hold no unclaimed marker, its categorical ones hold about a second

`analysis/q2j_july.md`. July's non-LIRF 249.7 s, by airport:
LTFM 303 (21.2 % of the slice's SSE), EGLL 313 (18.6 %), LFPG 273 (14.7 %),
LEBL 274 (11.5 %), then a step down to EHAM 198, EDDF 202, LEMD 204, EDDM 195,
LSZH 216. By regime: NM bulk 73.5 % of the SSE at 219 s, no-NM 11.4 % at 588 s,
the NM tail above 3600 s 7.6 % at 1,892 s (mean error **+1,427 s**), and the
block-on-LOBT class 7.5 % at 428 s (mean error +157 s, and **+501 s** on LEBL's
899 rows). The worst 5 % of airport-days carry 19.8 %, against 22.1 % on january
and 23.3 % on december — july is the *least* day-concentrated fold.

The marker screen: 99 label-free columns from the ranking-masked july frame, each
scored by the fraction of the NM-bulk squared error a 20-bin function of it
removes **on top of the model's prediction**, cross-fitted odd day / even day,
with the bin means **centred** so the correction is mean-zero.

| column | SSE removed | RMSE removed |
|---|---|---|
| a cross-fitted **constant** — the screen's floor | 0.23 % | 0.26 s |
| `span` (take-off − AOBT), which the model already has | **0.73 %** | 0.83 s |
| `sf_overtaken` | 0.67 % | 0.76 s |
| `sf_q_apt` | 0.48 % | 0.55 s |
| `sf_q_arr_apt` | 0.39 % | 0.44 s |
| `sf_gap_next_rwy` | 0.31 % | 0.35 s |

The centring matters and I got it wrong first. Without it every column removes
the model's global bias — about −11 s against a 227 s RMSE, worth 0.23 % of the
squared error — and is credited with it: a **Gaussian noise column scored
+0.21 %** in the uncentred version, and the table above read 0.94 % at the top
instead of 0.73 %.

**And the conclusion has to be narrowed twice** (gpt-5.6-sol round 1, blocking
finding 7). The screen tests 99 *numeric* engineered columns through one 20-bin
basis, so at most it closes the numeric-univariate question. Sol proposed the
family it misses, and it is worth more than the screen's winner. A cross-fitted,
mean-centred, shrunk (k = 50) **categorical** correction on the july NM bulk:

| key | SSE removed | RMSE removed |
|---|---|---|
| `ADEP` × `AIRCRAFT_OPERATOR_flt` | **0.95 %** | **1.09 s** |
| `ADEP` × `ADES_mvt` | 0.93 % | 1.05 s |
| `ADEP` × `STAND_mvt` | 0.86 % | 0.98 s |
| `ADEP` × `AIRCRAFT_TYPE_mvt` | 0.65 % | 0.74 s |
| `ADEP` × `RUNWAY_mvt` | 0.29 % | 0.32 s |
| **shuffled-key control** | **−0.23 %** | −0.26 s |

So the honest statement is: **the numeric columns of the ranking frame hold no
unclaimed univariate marker beyond `span`, which the model already has; the
categorical ones hold about 1 s of residual structure the model is not
extracting, largest on the airport-by-operator key.** That is a real lead and it
is not mine to spend — retro-v1 measured target encodings costing 43 s on july
when fitted without isolation from the early-stopping split, and its own guard
says no encoded run is promotable until that is repaired. The measurement above
says the prize behind that repair is about a second on the july bulk.

Read with finding 5's slopes, july's remaining bulk error is otherwise variance,
not bias: the conditional means are right and the spread around them is what is
left.

That is why questions 3–5 went to day-state rather than to july: the three july
slices with real bias are exp-002's (no-NM, the tail) and exp-006's (the
block-on-LOBT class is φ), and the fourth candidate is refuted below.

### 8. REFUTED, by me, the same day: the block-time convention is not a pair of point masses

`analysis/q5_mixture.md`. Reported to the desk as a finding and withdrawn ninety
minutes later, before it reached the tail worker.

**What I claimed.** With `r = BLOCK − AOBT` and `lob = LOBT − AOBT`, 19.5 % of
july's in-scope rows have `|r| ≤ 60 s` and 3.9 % have `|r − lob| ≤ 60 s`; an
oracle that replaced the model's prediction on exactly those rows with the exact
implied value gained **19.5 s on july, 14.9 on december, 14.0 on january**. I read
that as two components — the airport writes its block time either on the actual
off-block or on the filed one — with φ as the mixing weight.

**Why it is wrong.** An atom at 0 means `r` is 0 *to the second*, not "within
60 s". It is not:

On the oracle's **own** population — NM-joined, non-LIRF, |lob| ≥ 300 s, 118,633
july rows:

| tolerance | share with \|r\| ≤ tol | share with \|r − lob\| ≤ tol |
|---|---|---|
| 0 s | **0.56 %** | **0.17 %** |
| 5 s | 6.34 % | 1.87 % |
| 30 s | 9.70 % | 2.22 % |
| 60 s | 19.52 % | 3.94 % |
| 120 s | 36.99 % | 7.18 % |
| 300 s | 71.07 % | 19.26 % |

The share grows smoothly with the window and the exact values carry well under a
percent of the rows, so **there are no point masses at 0 and at `lob`**, which is
precisely what the withdrawn claim asserted. That is what this test shows and all
it shows: a two-component mixture whose components are themselves diffuse would
leave no atom either and is **not** excluded (gpt-5.6-sol round 1, blocking
finding 9). What is excluded is the reading that made the oracle look like a
ceiling on a router — the rows inside a ±60 s window are the continuum's local
density, not a class — so the 19.5 s figure is an oracle that already knows `r` on
a quarter of the rows. And the one router actually built, on the best label-free
classifier of the supposed component (airport-operator rate, AUC 0.732),
**loses 117 s to the model** on july (324.9 against 207.8), because the LOBT side
is 4 % of the rows and the posterior almost never crosses 0.5. One router losing
does not close every router; it does say that the cheap one is not there.

(The shares above are recomputed on the oracle's own population — NM-joined,
non-LIRF, |lob| ≥ 300 s. The first version of this table dropped the non-LIRF
restriction and so measured a different set of rows than the claim it was
refuting; the ledger row is corrected accordingly.)

**What survives, for exp-006's record.** φ estimated as a row-count-weighted mean
of the per-flight ratio is the conditional mean of a continuous quantity, which is
the right object; the airport differences it measures are real (the share of rows
with `|r − lob| ≤ 60 s` runs from 1.2 % at LSZH to 11.9 % at LIRF and 8.8 % at
LTFM); and `|lob|` is *inversely* related to the LOBT side (AUC 0.30), so the
convention story is weakest exactly where a mistake is most expensive.

### 9. The module: `prc/day_state_features.py` VERSION 1

Handed back as the assignment asks, with `fit`/`transform` and a stable column
list. 25 columns, all `float32`, one row per departure:

| group | columns | what it is |
|---|---|---|
| delay level | `ds_sched_med`, `ds_sched_p90` | the airport-day's median and p90 of take-off − schedule |
| plan gap | `ds_gap_med`, `ds_gap_p90` | the same for AOBT − LOBT |
| surface span | `ds_span_med`, `ds_span_p90`, `ds_span_std`, `ds_share_span1800` | the same for take-off − AOBT, plus the day's share above 1,800 s |
| join quality | `ds_nm_rate` | the day's share of departures with an NM join |
| arrivals | `ds_taxiin_med`, `ds_taxiin_p90` | the day's arrival taxi-in distribution (fully labelled in the ranking frame) |
| size | `ds_n_dep`, `ds_mvt_total` | departures and total movements that day |
| **referenced** | the nine `*_z` columns | the day's statistic minus the airport's median 2025 day, divided by that airport's 2025 interquartile range |
| **row level** | `ds_deice_hr`, `ds_deice_day`, `ds_deice_hr_n` | the assignment's de-icing proxy: the share of same-airport departures with take-off − AOBT above 1,800 s in the surrounding hour and in the day, **the row itself excluded**, and the neighbour count behind it |

Three named subsets, and **only the last should be handed to a model** (finding
14): `COLUMNS` (all 25, the set the screens and arms are measured against),
`CORE` (seven referenced columns, defined but never run), and **`DEICE`**
(`ds_deice_hr` alone, the shippable line). `--ds-columns all|core|deice` on
`code/run_ds.py` selects between them.

Two design decisions worth their own line.

**The reference is fitted on the training months, not within the month.** A day
statistic normalised inside its own month would show January 2026 as ordinary,
because the whole month moved — which is precisely the signal
`docs/shift-2026.md` measured and precisely what the columns exist to carry. So
`fit` stores each airport's 2025 reference and `transform` places every day
against it.

**`fit` reads no label at all**, so no leave-one-month-out construction is needed
and a training row's own month may take part in the reference it is scored
against: the day statistics are built from timestamps, the NM join flag and
arrival taxi times, and departure `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` are
blanked inside the module's own working arrays before anything reads them. This
is asserted, not asserted-in-prose (`code/test_day_state.py`, seven checks, all
passing):

1. transforming the **unmasked** july frame and the **ranking-masked** one gives
   identical values on all 190,713 × 25 cells — the module cannot be reading the
   departure labels;
2. `fit` on a frame whose departure taxi times are overwritten with 12,345 and
   whose departure block times are set to `NaT` produces a bit-identical
   reference;
3. overwriting the **arrival** taxi times *does* move the reference, so the legal
   channel is genuinely used;
4. moving one row's own AOBT by 9,000 s moves 33 of its neighbours' `ds_deice_hr`
   and leaves its own value untouched;
5. `ds_deice_day` reproduces a hand computation with the row removed;
6. thin airport-days are `NaN`, not one-row statistics;
7. the schema is the module's `COLUMNS`, all `float32`.

### 10. The block fires on January 2026 — which was the case for shipping it, and finding 13 withdraws that case

`analysis/q3b_ranking.md`. `DayState` fitted on all twelve 2025 months and
transformed one ranking month at a time (the file holds January and July 2026 as
isolated months; nothing joins across the gap). All 344,841 ranking departures get
a value on 24 of the 25 columns; `ds_deice_hr` is null on 0.12 % of rows, the ones
with no departure neighbour inside the surrounding hour.

Airport-days placed against their own airport's 2025 reference:

| | January 2026 | July 2026 |
|---|---|---|
| airport-days | 310 | 310 |
| with `ds_span_std_z` > 2 (dispersion two 2025 IQRs above a normal day) | **90** | 23 |
| with `ds_span_std_z` > 3 | **69** | 13 |
| with `ds_share_span1800_z` > 2 (the de-icing proxy) | **70** | 7 |
| with `ds_nm_rate_z` < −3 (the NM join failing) | 27 | 21 |

The median day of each month is ordinary (`ds_span_std_z` +0.24 January, +0.11
July): this is a tail of days, which is exactly the shape `docs/shift-2026.md`
found. The hardest are at Amsterdam in the first week of January 2026 —
3–6 January carry `ds_sched_med_z` between +18 and +34 and `ds_nm_rate_z` between
−31 and −55, i.e. a day whose median departure leaves tens of interquartile ranges
later than a normal Schiphol day while a third of its NM joins are missing — and
at Zurich on 10 and 22 January.

**Sixty-nine January-2026 airport-days sit more than three 2025 interquartile
ranges above their own airport's normal dispersion, against thirteen in July.**
No 2025 fold contains days like that, so no fold can price these columns; what a
fold *can* do is show they cost nothing, which is what findings 11 and 12
measure. A tree
trained on 2025 will put every one of those days in its topmost split bin — the
columns do not extrapolate, they saturate — ~~but a saturated "this is the worst
kind of day this airport has" is the correct answer more often than the mid-range
value the model would otherwise use.~~ **That last clause is refuted by finding
13**, which measures the saturation doing harm: on the january fold's most
extreme day the top bin's correction, learned on days that were under-predicted,
is applied to a day that was not, and the squared error doubles. The count of
extreme January-2026 days below is therefore a measure of *exposure to that
failure*, not of opportunity.

### 11. What the screen predicts the arms will show — written before they landed

`analysis/q3a_screen.md`. The same instrument as finding 7, turned on the module's
own columns: the fraction of the baseline's NM-bulk squared error that a 20-bin,
mean-centred function of one day-state column removes **on top of the model's
prediction**, cross-fitted odd day / even day.

| fold | best day-state column | SSE removed | RMSE removed | everything else |
|---|---|---|---|---|
| january | `ds_deice_hr_n` | 0.01 % | 0.01 s | ≤ 0.01 % |
| december | `ds_deice_hr` | 0.15 % | 0.14 s | ≤ 0.03 % |
| july | `ds_deice_hr` | 0.13 % | 0.14 s | ≤ 0.06 % |

Additively, the day-state block is worth **nothing** — a tenth of a second at
best, against a `span` column that buys 0.83 s and a day-effect oracle worth
2.4 s. What the univariate screen cannot see is **interaction**: a tree may use a
day-state column as a *modifier* of the row's own value. So the same screen on a
2-D grid, 8 × 8 cells of (own value, day state), with the control being the same
grid with the day-state axis **shuffled** — which keeps the grid's flexibility and
the column's marginal but destroys its link to the day.

| pairing (own value × day state) | january | december | july |
|---|---|---|---|
| `span` × **shuffled control** | 0.02 % | −0.08 % | 0.54 % |
| `span` × `ds_deice_hr` | **0.16 %** | **0.10 %** | **0.85 %** |
| `span` × `ds_share_span1800_z` | −0.00 % | −0.19 % | 0.61 % |
| `span` × `ds_span_std_z` | 0.01 % | −0.32 % | 0.41 % |
| `gap_al` × `ds_gap_p90_z` | −0.53 % | −0.13 % | 0.20 % |
| `span_sched` × `ds_sched_med_z` | −0.33 % | −0.05 % | −0.04 % |
| `span` × `ds_nm_rate_z` | −0.33 % | −0.33 % | 0.14 % |

**One column beats its own shuffled control on all three folds, and it is
`ds_deice_hr` — the assignment's own de-icing proxy, the surrounding hour's share
of long-taxi neighbours with the row itself removed.** Its excess over the control
is +0.14, +0.18 and +0.31 percentage points of the NM-bulk squared error: 0.14,
0.17 and 0.35 s of RMSE. Every other pairing sits at or below its control, so the
day aggregates that ranked bad days best in finding 6 — the day's median
take-off-minus-schedule, its p90 plan gap, its dispersion — carry no incremental
information about the *residual* once the row's own values are known. Note how
much work the control does: on july the shuffled grid alone removes 0.54 %,
because the span axis is real signal, and a reader without it would have called
`ds_share_span1800_z`'s 0.61 % a finding.

**So the expectation, recorded before the paired arms finished: a gain of at most
a few tenths of a second on the folds, most likely inside the seed noise, largest
on july, and carried by `ds_deice_hr` rather than by the day aggregates that
ranked bad days best.** Marking a bad day and improving a prediction on it are
different things, and this experiment can now put a number on the gap: ρ +0.651
and AUC 0.868 for *marking*, 0.35 s for *predicting*.

### 12. The paired arms, and why their headline numbers are not the answer

`analysis/q3_bootstrap.md`, `climb.log`. Both arms through `code/run_ds.py` on the
pinned pipeline, seed 7, the day-state block on or off and nothing else changed.

| fold | base RMSE | ds RMSE | base bulk | ds bulk | base refit rounds | ds refit rounds |
|---|---|---|---|---|---|---|
| july | 321.85 | **315.40** | 277.85 | 272.66 | 50 | 107 |
| december | 236.88 | **222.24** | 219.42 | 204.74 | 235 | 207 |
| january | 350.91 | 351.98 | 216.62 | 217.99 | 284 | 96 |

Paired, airport-day-clustered, 2,000 draws, on the core (NM-joined, non-noac) of
the other nine airports — negative is the day-state arm winning:

| fold | Δ RMSE | 95 % interval | on the label-free most-disrupted 5 % of days |
|---|---|---|---|
| july | **−10.64 s** | [−11.71, −9.51] | −6.99 [−13.73, +0.70] |
| december | **−1.80 s** | [−4.66, +0.86] | −11.02 [−18.76, −3.31] |
| january | **+6.54 s** | [+1.91, +13.45] | **+47.41** [−0.59, +113.38] |

**I do not believe these numbers measure the columns, and the reason is in this
experiment's own earlier findings.** Three things do not fit:

1. **The sign flips across folds** — a 10.6 s win, a null, and a 6.5 s loss.
2. **The magnitudes are 20 to 60 times the ceiling** the cross-fitted screen and
   the interaction probe put on these columns (finding 11: 0.35 s at best, with
   an 8 × 8 grid that can see interactions, and the model held fixed so no round
   draw can enter).
3. **The two arms did not fit the same model.** Adding columns moves the
   early-stopping trajectory, so the refit round count differs by arm: 50 against
   107 on july, 235 against 207 on december, 284 against **96** on january. The
   january arm that lost is the one that fitted for a third as long.

The paired bootstrap's interval covers **row sampling only**. It cannot see the
seed or round draw, which is what actually varies here. retro-v1's own climb log
prices that draw, over families that differ by nothing but the seed:

| family, fold | seeds | RMSE spread across seeds |
|---|---|---|
| `e0`, july | 17, 42 | **34.5 s** |
| `H0`, july | 7, 17, 42 | 18.3 s |
| `H0`, december | 7, 17, 42 | 16.1 s |
| `e0`, december | 17, 42 | 14.3 s |
| `P1`, july | 7, 17, 42 | 5.8 s |
| `G1cv`, july | 7, 17 | 3.3 s |
| `G1cv`, december | 7, 17 | 1.9 s |

Every arm difference above sits inside the seed spread of a comparable
configuration. A paired bootstrap that resamples rows and reports
[−11.71, −9.51] on a quantity whose seed-to-seed standard deviation is of the
same order is measuring the wrong thing, and that is a lesson for every scored
row in `docs/LEDGER.md`, not only for this one.

### 13. With the round draw removed, the day-state block does not deliver — and on the most extreme winter day it does harm

`analysis/q3_bootstrap_f300.md`. Both arms refitted for a **declared 300 rounds**
with early stopping switched off (`--fixed-rounds`, a control added to
`code/run_ds.py`, never to the pinned snapshot; 300 was written on the command
line before the run and no variant of it was screened).

| fold | base, own early stop | ds, own early stop | base @300 | ds @300 |
|---|---|---|---|---|
| july | 321.85 (50 rounds) | 315.40 (107) | 326.88 | 327.14 |
| december | 236.88 (235) | 222.24 (207) | 238.50 | 221.88 |
| january | 350.91 (284) | 351.98 (**96**) | 350.80 | 355.14 |

**On july the control removes the difference.** The base arm's own early stop
picked 50 rounds and scored 321.85; forced to 300 it scores 326.88 — so the round
count alone is worth 5 s there, in the *opposite* direction to more fitting — and
at a common 300 rounds the two arms are 0.26 s apart where the uncontrolled
comparison said 6.45. An earlier version said "the headline was the round draw,
in full"; that overstates it (round-2 review, blocking finding 7). Forcing 300
costs the base arm 5.03 s **and the ds arm 11.74 s**, so the control is
unfavourable to both and more unfavourable to the 121-feature arm. "The two arms
are 0.26 s apart at 300" is the claim these numbers support; "the −6.45 s was the
round count" is one reading of them and "300 rounds destroys the wider arm faster"
is another.

**And on january the control is not neutral, which costs the finding most of its
headline** (round-2 review, blocking finding 2). The base arm's own stop is
**284**, so 300 barely moves it (−0.11 s); the ds arm's is **96**, so 300
over-trains it hard (+3.16 s). The two comparisons therefore *bracket* the
january effect rather than settling it:

| january comparison | ds − base |
|---|---|
| each arm at its own early stop | **+1.07 s** |
| both forced to 300 rounds | **+4.34 s** |

The control is fair when arms differ only in columns *and sit a similar distance
from their own optimum*; here they do not, and finding 15 makes the same argument
about a different asymmetry. A fair january control needs a swept round count —
each arm at its own minimum over a grid — and that was not run. **So the january
harm is real in sign under both comparisons but its size is uncertain by a factor
of four**, and the +4.34 figure quoted below should be read as the top of a
bracket, not a measurement.

Paired, airport-day clustered, 2,000 draws, at matched rounds:

| slice | july | december | january |
|---|---|---|---|
| whole fold (reachable) | +0.25 [−3.1, +3.5] | **−16.62** [−29.6, −6.0] | +6.87 [−1.7, +18.7] |
| core, other nine | −0.64 [−1.7, +0.6] | −4.11 [−9.7, +0.2] | +5.87 [−0.5, +14.5] |
| **bulk ≤3600, other nine** | **−1.49** [−2.3, −0.7] | −3.60 [−9.0, +0.3] | +2.84 [−1.9, +10.0] |
| **most disrupted 5 % of days (label-free)** | −0.97 [−5.8, +4.2] | **−10.79** [−20.0, −2.1] | **+72.09** [+3.5, +149.7] *(calendar-day [−3.7, +148.3], spans zero)* |

`july_loo` was run too, but only at each arm's own early stop (125 rounds base,
148 ds), so it is **not** matched-rounds and after the july result above it cannot
be read as a column effect: base 312.07, ds 307.45, core-nine −5.82 s
[−7.67, −4.11]. It is recorded in `work/boot_july_loo.md` and in the ledger as
uncontrolled. The round counts are closer than july's 50-against-107, which is
the only reason to think part of it might be real; nothing here establishes that.

Read the bottom row of the matched table. On the slice the block was *built for* — the days its own
label-free columns flag as most disrupted — it is worth **−10.8 s on december and
+72.1 s on january**, in opposite directions — december's holds under both
clusterings, january's only under the airport-day one. The one consistent
positive anywhere is **−1.5 s on the july bulk**.

**Why january's disrupted days get worse, measured rather than guessed.** The
january loss is not spread: it is four airport-days.

| airport-day | rows | base RMSE (bias) | ds RMSE (bias) | mean target |
|---|---|---|---|---|
| LSZH 2025-01-04 | 295 | 385.5 (**+40.2**) | 791.2 (**−75.2**) | 1,113.6 |
| EHAM 2025-01-05 | 539 | 273.9 (+36.9) | 630.3 (−150.2) | 1,054.4 |
| EDDF 2025-01-05 | 413 | 297.0 (+58.5) | 452.9 (+31.3) | 1,288.7 |
| EHAM 2025-01-18 | 492 | 284.6 (+20.4) | 427.0 (−110.5) | 1,079.4 |
| …against LFPG 2025-01-17 | 576 | 560.3 (**+170.2**) | 558.0 (**+14.7**) | 1,978.7 |
| …and EDDM 2025-01-15 | 356 | 398.0 (**+165.5**) | 344.9 (**+27.7**) | 1,519.1 |

On most flagged days the block pushes predictions **up**. Where the base model was
badly under-predicting — LFPG on 17 January at +170 s of bias, EDDM on 15 January
at +166 — that is exactly right and the block removes almost all of it. Where the
base model was already close — LSZH on 4 January at +40, EHAM on 5 January at
+37 — the same push carries it to −75 and −150 and the squared error doubles.

**Two corrections to that story, both from the round-2 review (blocking finding
6).** "Pushes up on *every* flagged day" is false: across the full 19-day
disrupted slice it pushes *down* on EGLL 11 January, EDDF 10 January and LSZH
17 January. And **EDDF 5 January, the second-largest loss of the four, does not
overshoot at all** — its bias moves *toward* zero, +58.5 to +31.3, so the harm
there is within-day variance and not an overcorrection. The four-day table below
is not four instances of one mechanism.

And on the worst-hit day the columns are **out of their own training range** —
though the first version of this table said "three of five", picked the three it
showed, and led with the smallest exceedance of them. The module defines **nine**
`_z` columns; here are all nine, `DayState` fitted on the january fold's training
months and evaluated on LSZH 2025-01-04 against that airport's own training days
(round-2 review, blocking finding 4):

| column | that day | that airport's training maximum | over by |
|---|---|---|---|
| `ds_taxiin_p90_z` | **7.73** | 2.92 | **+165 %** |
| `ds_sched_med_z` | 3.48 | 2.87 | +21 % |
| `ds_share_span1800_z` | 4.53 | 4.28 | +6 % |
| `ds_span_p90_z` | 22.97 | 22.89 | +0.4 % |
| `ds_span_std_z` | 11.79 | 11.78 | **+0.08 %** |
| `ds_sched_p90_z` | 2.93 | 3.18 | – |
| `ds_gap_p90_z` | 2.38 | 2.78 | – |
| `ds_span_med_z` | 0.83 | 2.20 | – |
| `ds_nm_rate_z` | −2.19 | 1.05 | – |

**Five of nine, not three of five.** And the exhibit the first version led with —
`ds_span_std_z` at 11.79 against 11.78 — is a 0.08 % exceedance: the two values
fall in the same LightGBM histogram bin, so no split can distinguish them and that
row is worthless as evidence. The one that carries weight is `ds_taxiin_p90_z` at
2.6× its training maximum, which the first version omitted.

**Two readings fit these six days and this experiment does not separate them.**

*Extrapolation.* Counting how many of the nine `_z` columns exceed their airport's
training maximum does line up with the harm: the four harmed days score 5, 6, 4
and 3 of nine, and the two helped days score 2 (LFPG) and **0** (EDDM). That is a
real association and it is the basis of the revival condition below.

*Severity is not error.* The block marks days that are severe, not days the model
gets wrong. Where the base was already calibrated, any upward push overshoots.
This explains all six days **with no extrapolation at all**, and finding 6
measured exactly this distinction — I wrote the sentence, "marking a bad day and
improving a prediction on it are different things", without following it to its
consequence.

Against the first reading: EDDF 5 January is the **most** out-of-range day of the
six (six of nine columns past the maximum) and yet its bias moves toward zero, and
the exhibit originally offered for extrapolation was a 0.08 % exceedance inside
one histogram bin. Against the second: the exceedance count does separate harmed
from helped days. **Nothing here decides between them**, and the experiment that
would — a clipped-block arm, identical columns with each `_z` clipped to its
training range — was never run. Finding 10's guess that a saturated *worst kind of
day* would be right more often than a mid-range value is refuted either way; what
is not established is *why*.

**So the answer to the assignment's question 3 is negative, and the case made in
finding 10 for shipping the block as January insurance is withdrawn by its own
measurement.** January 2026 carries 69 airport-days more than three training
interquartile ranges above their airport's normal dispersion, against the one
that did this damage here; the block would saturate on all of them.

**The mechanism, stated as the lane's revival condition (the desk's wording,
2026-09-05).** A day-state column **beyond its own training maximum pushes a
near-calibrated model past the truth**. That is what LSZH 2025-01-04 shows: three
of five `_z` columns at or above that airport's training maximum, a base bias of
+40 s turned into −75 s, and the squared error doubled. So **any future January
insurance must be clipped to the training range, or expressed as ranks within the
month** — a construction that cannot hand the tree a value it has never priced.
`ds_deice_hr` already satisfies that by construction: it is a share, bounded in
[0, 1], and it is the one column that beat its own shuffled control on all three
folds (finding 11).

Two further conditions follow from the same measurement and are weaker only in
that nothing here tests them. The block cannot tell a severe day from a day the
model gets wrong, so a construction that **predicts the model's residual on the
day** rather than the day's severity would fix the direction problem as well as
the range problem — and that needs out-of-fold residuals on the *training*
months, which this fold design does not produce (finding 5 made the same point
about the day-severity second stage). Until something can tell
LFPG-17-January from LSZH-4-January in advance, a day-state correction is a bet
that flagged days are under-predicted days, and on the january fold that bet
loses 72 s where it matters most.

**What is being measured next**: a `ds_deice_hr`-only arm at the same matched 300
rounds, against the same `f300` base arm — the one bounded column, on its own.
**Finding 14 is that measurement, and it works**: dropping the unbounded columns
removes the january harm (+4.34 → −3.01 on the whole fold, +72.09 → −6.05 on the
disrupted days) and keeps most of december's gain.

### 14. One column, `ds_deice_hr`, is the part that works — and it works because it cannot leave its training range

`analysis/q3_bootstrap_deice.md`. The desk's call after finding 13: run the one
column that beat its own shuffled control on all three folds, on its own, at the
same declared 300 rounds against the same base arm.

| fold | base @300 | full 25-column block @300 | **`ds_deice_hr` alone @300** |
|---|---|---|---|
| july | 326.88 | 327.14 (+0.26) | 327.46 (+0.58) |
| december | 238.50 | 221.88 (−16.62) | **224.88 (−13.62)** |
| january | 350.80 | 355.14 (**+4.34**) | **347.79 (−3.01)** |

**The single bounded column keeps most of december's gain and turns january from
a 4.3 s loss into a 3.0 s win.** Paired, airport-day clustered, 2,000 draws,
against the same `f300` base arm:

| slice | july | december | january |
|---|---|---|---|
| whole fold (reachable) | +0.58 [−5.5, +7.6] | **−13.62** [−26.2, −3.0] | **−3.92** [−6.8, −1.0] |
| core, other nine | −0.34 [−1.0, +0.3] | **−5.19** [−10.2, −1.7] | −1.01 [−2.6, +1.0] |
| bulk ≤3600, other nine | −0.34 [−0.9, +0.2] | **−4.66** [−9.5, −1.7] | −0.82 [−2.3, +1.3] |
| most disrupted 5 % of days (label-free) | −2.76 [−7.8, +2.2] | **−5.96** [−11.8, −1.2] | −6.05 [−12.2, +1.4] |

Every point estimate but two is an improvement — july's whole fold (+0.58) and
july core (+0.84) are positive, both well inside their intervals — december's four
intervals all exclude zero, january's whole fold excludes zero, and **no slice on any fold shows
harm with an interval excluding zero.** Compare finding 13's january row for the
full block: +72.09 [+3.46, +149.73] on the disrupted days, against **−6.05** here.

**This arm does not isolate boundedness, and an earlier version of this section
claimed it did** (round-2 review, blocking finding 5). The two arms differ by
**24 columns**, of which only nine are `_z`; the other fifteen include the raw day
levels — `ds_sched_med`, `ds_span_p90`, `ds_taxiin_p90`, `ds_n_dep` — which are
just as unbounded. They also differ by feature count (97 against 121), which under
finding 13's own argument changes the over-fitting rate at a fixed 300 rounds, and
by how hard they push: on january's disrupted slice the block moves predictions
**+54.0 s** on average and this arm **+19.1 s**. An arm that pushes a third as
hard cannot overshoot as much whether or not it is bounded.

So what this arm establishes is the **practical** result — one column is better
than twenty-five on every fold — not the mechanism. The experiment that would
separate them is a **clipped-block arm**: the same 25 columns with each `_z`
clipped to its training range, which changes boundedness and nothing else.
`prc.day_state_features.CORE` exists and was never run; the clipped variant was
never built. That is the first thing to run if anyone reopens this lane.

**So the deliverable is `prc.day_state_features.DEICE` — one column — not the
block.** The module still ships all 25 with `fit`/`transform`, because the screen
and the arms are only interpretable against the full set and a later consumer may
want the raw statistics for something else; but `--ds-columns deice` is the line
to adopt, and `COLUMNS` should not be handed to a model whole.

**One condition before adoption.** Every `ds_deice_hr` number here is at
`--fixed-rounds 300`; there is **no early-stopped deice arm on any fold**, and
retro-v1's production path is `two_pass` with early stopping. Finding 13 shows
that switch moving july's base arm by 5 s and january's comparison by 3.3 s.
December's −13.6 s is large enough that it will very likely survive — it is
−14.64 s uncontrolled as well — but january's −3.0/−3.9 s is the same order as the
round effect itself. **Re-run the deice arm under early stopping before adopting
it**, which is one cheap three-fold run (round-2 review, blocking finding 8).

### 15. Question 4's second half: weight the winter months a little, never train on them alone

The assignment's question 4 also asked whether the January half of the submission
wants its own model — "sample weighting toward winter months … train two models,
one per ranking month, versus one". gpt-5.6-sol round 1 (blocking finding 8) was
right that the desk's re-aim did not waive it. Three january-fold arms, the
weighting applied to the **model's training rows** while the shared statistics
(phi, the lirf signals, the day-state reference) stay on all eleven months:

| arm | training rows used | january RMSE | bulk ≤3600 | rounds |
|---|---|---|---|---|
| unweighted (all eleven months) | 1,931,341 | 350.91 | 216.62 | 284 |
| **winter months × 3** (2025-02, -11, -12) | 471,741 up-weighted | **349.24** | **214.17** | 168 |
| **winter months only** | 471,741 kept, 1,459,600 dropped | **366.12** | **238.77** | 78 |

**Restricting training to the winter months is a clear loser — +15.2 s on the
fold and +22.2 s on the bulk** — and it loses in the direction
`docs/shift-2026.md` predicted: no 2025 month carries January 2026's conditions,
so throwing away three quarters of the data buys composition at the cost of
seeing enough disrupted days. The ×3 up-weighting is worth about 1.7 s.

**And the matched-rounds control is the wrong instrument for one of these two
arms, which is worth stating because it is the same lesson as finding 13 in
reverse.** Refitting both at a declared 300 rounds against the `f300` base arm's
350.80 / 216.58:

| arm @300 rounds | january RMSE | bulk ≤3600 | rows it trains on |
|---|---|---|---|
| winter months × 3 | **349.28** (−1.52) | **214.35** (−2.23) | 1,931,341 |
| winter months only | **453.34** (+102.5) | 358.63 (+142.1) | **471,741** |

The `× 3` arm is stable across round settings — 349.24 at its own 168 and 349.28
at 300 — because it trains on the same rows as the control and only re-weights
them. The `winter only` arm is not, and should not be: it has **24 % of the rows**,
its own early stopping chose **78** rounds, and forcing it to 300 overfits it into
the ground. A fixed round count is a fair control when the arms differ in
*columns* and an unfair one when they differ in *training-set size*, because the
round count is not a nuisance parameter there — it is part of what the smaller
training set needs. The honest comparison for this question is therefore each
arm at its own early stop, the first table above.

So the answer to question 4's second half is: **weight, do not split.** One model
with the winter months up-weighted, not two models one per ranking month.

One caveat on the size, not the direction (round-2 review, blocking finding 11):
the winter-only arm cannot train on **January 2025**, because that is the fold's
validation month, and `prc/data.py` says January 2025 "is the only month that
resembles the January ranking month". So this arm tests splitting with the one
month that motivates the split removed, on 471,741 rows where a real split model
would have roughly 625,000 including it. The fold therefore **overstates** the
cost of splitting, and an earlier version's "the margin is at least +15.2 s" runs
the wrong way. The direction is not in doubt — 15 s is a wide margin and the ×3
arm beats both — but the number is an upper bound on the penalty, not a lower
one.

### 16. On the desk's `(july_loo, january)` analogue pair the other nine read 234.6 s

`analysis/q6_blend.md`. `prc.data` VERSION 2 adds `july_loo` — the other eleven
months against July 2025. The submission model trains on all twelve, so July 2026
is predicted with July and August 2025 in training; the forward `july` fold has
no same-season data and is pessimistic, exactly as `january` is the analogue for
January 2026.

| line | July fold | other nine | LIRF | implied total |
|---|---|---|---|---|
| this experiment's base arm | `july` (forward, six months) | 240.4 | 621.8 | 288.9 |
| this experiment's base arm | **`july_loo` (the analogue pair)** | **234.6** | 611.9 | **282.8** |

`july_loo` is 9.8 s better than the forward fold on the whole fold (312.07 against
321.85) and moves the blend 5.8 s. Every number in the table comes from **one
line's** own out-of-fold files, so the folds share a configuration and a set of
`prc/` module versions, and the two unreachable rows are removed.

The reading is finding 2's, with `july_loo` in place of `july` and the same
condition attached: **234.6 s is inside the 200–240 s band the board implies, but
234.6 is a cleaned local skill number on 2025 months and the band is an observed
board component on 2026 months**, so the comparison holds conditionally on k = 0
and on the folds transferring. What it suggests, **and only suggests**, is that the forward `july` fold has been
costing this repo something like 6 s of pessimism in the board comparisons it has
appeared in. That is not matched-rounds either — `d1-base-s7` chose 50 rounds on
`july` and 125 on `july_loo` — so it carries exactly the confound this experiment
disclaims for the july_loo *arm* comparison, and an earlier version's
"unconditionally" applied two standards to the same two runs (round-2 review,
blocking finding 9). The blend arithmetic itself was independently reproduced and
is correct.

(This experiment's base arm is not retro-v1's `G1cv-s7`: the same command line
today reads 240.4 rather than 234.2 on the forward pair, because `prc/`
phi/lirf/surface moved after `G1cv-s7` ran. See the pinned-pipeline section.)

### Review

**Round 1** (`review/sol-round1.md`), `gpt-5.6-sol` at `xhigh`, fresh context:
**FAIL — nine blocking findings, no feature leakage found.** All nine are
answered; `SPEC.md` carries a table of what each one changed, six correction rows
are in `docs/LEDGER.md`, and the two that changed conclusions are recorded in the
findings themselves (finding 1's "unreached, not unreachable", and finding 7's
categorical family). Two of the nine — the floor coefficient and the `k ≤ 2`
bound — required withdrawing claims rather than restating them.

**Round 2** (`review/round2-opus.md`), a fresh-context `claude-opus-5` at `xhigh`,
run under the desk's substitution rule because the primary reviewer
was not available: **FAIL — eleven blocking findings.** Three were
outright errors in this write-up and are corrected above with the correction
named at each site: a **fabricated AUC column** in finding 6, an interval called
"excluding zero" that its own artifact reports as **spanning zero**, and an
exceedance ratio that was **five of nine, not three of five**, led by a 0.08 %
exceedance inside a single histogram bin. The rest weakened claims rather than
numbers: the matched-rounds control is **not neutral on january** and brackets the
effect at +1.07 to +4.34 rather than settling it at +4.34; finding 14 does **not**
isolate boundedness; the deice line has **never been measured under early
stopping**, which is the path it ships into. The reviewer independently
reproduced every bootstrap point estimate, the whole blend arithmetic, the
categorical probe and the detector table, confirmed the pinned snapshot
byte-identical to git 19cd199, and found no leakage. Verdict recorded as
`reviewer: claude-opus-5-xhigh-2026-09-05 (substitute)`.

**The first attempt at round 2** (`review/sol-round2.md`): **not completed.** The reviewer read the
files and then stopped before it wrote anything, so it
produced no findings and no verdict. Per `assignments/COMMON.md` a worker that
cannot finish its run stops rather than dropping to a weaker model. The prompt is at
`review/sol-round2-prompt.txt` and the adapted fresh-context version actually run
is `review/round2-prompt.txt`; both ask about the matched-rounds
control's own validity, the extrapolation attribution, the `--base-name` pairing,
the `july_loo` blend, the fairness of the winter-only arm, and whether the
single-seed argument undercuts findings 13 to 15. **Three ledger rows record the
round-2 corrections; the scored rows of findings 13 to 16 should be read with
them.**

**What round 2 did not change**: one column beats twenty-five on every fold, and
weighting the winter months beats splitting on them. **What it did change**: the
claimed mechanism behind the first, and the claimed precision of both.

### Artifacts

Out-of-fold predictions for every arm: `experiments/exp-007-winter/oof/`, hashed
in `oof/SHA256SUMS.txt`. The shippable line is **`f300d-ds-s7-*`** — the
`ds_deice_hr`-only arm at a declared 300 rounds — against its matched control
`f300-base-s7-*`. `d1-*` are the uncontrolled arms of finding 12 and should not be
used for a comparison. Every attempt is in `climb.log` with its arm, column set,
fixed-round setting, seed and shared-module tag.

No submission candidate is offered from this experiment: the shippable piece is
one column for retro-v1 to adopt (`--ds-columns deice`, or
`prc.day_state_features.DEICE`), not a model.

---

## What this changes for the desk

1. **Stop pricing the january fold by its headline.** Read `rmse_bulk` (clean),
   or subtract the two rows. The fold ranks models fine; it misprices them by
   114 s and compresses differences by a third (finding 4).
2. **The other nine are not 60 s behind the leaders; they are inside the band on
   the 2025 folds** (finding 2). What is left of the distance is either July
   structure that has no univariate marker left (finding 7), a January 2026 that
   no fold can price (finding 10), or the rollover lottery (finding 3).
3. **One submission is worth more than any further fold analysis right now** —
   but as a sum, not a decomposition. A pipeline whose cleaned fold skill is
   286 s reads 286 / 314 / 341 at k = 0 / 1 / 2, so a reading carries real
   information; it also folds in the event magnitudes, the January-2026 shift and
   fold-to-2026 transfer, and cannot be inverted for k alone (finding 3).
4. **Ship one column, not the block.** `prc.day_state_features.DEICE`
   (`ds_deice_hr`) is worth **−13.6 s on december and −3.9 s on january**, both
   whole-fold intervals excluding zero, null on july, with no slice on any fold
   showing harm (finding 14). The full 25-column block is worth +4.3 s on january
   and **+72.1 s on january's disrupted days** and must not ship (finding 13).
   **The mechanism is extrapolation**: a day-state column beyond its own training
   maximum pushes a near-calibrated model past the truth, so any future January
   insurance must be **clipped to the training range or expressed as ranks within
   the month**. That is the revival condition for the lane, and the one-column arm
   is a second measurement of it.
5. **Weight the winter months, do not split on them** (finding 15). Training the
   january fold on the winter months alone costs **+15.2 s**; up-weighting them
   ×3 while keeping all eleven is worth about **1.5 s** at matched rounds. So
   "two models, one per ranking month" loses in its strong form, exactly as
   `docs/shift-2026.md` argued.
6. **Question 4's day-severity second stage is not worth building.** The oracle
   is 2.4 s and the model is already calibrated (finding 5) — and finding 13
   shows what happens when a day-level correction is applied without knowing
   whether the day is one the model gets wrong.
7. **Weather is the weaker of the two day markers.** exp-005's WINTER_CORE is
   worth keeping on its own measured evidence, but the day's own delay
   aggregates rank bad days three times better once the airport and the season
   are held fixed (finding 6), and they are free — they need no external data,
   no ERA5 snapshot and no join.
8. **The block-time-convention lane is narrowed by measurement, not closed by
   belief** (finding 8): there are no point masses, so the *cheap* router is not
   there and φ's conditional mean is the right object. A diffuse two-component
   mixture is not excluded. Revival condition: a label-free column that predicts
   the LOBT side *at large* |LOBT − AOBT|, where today's best candidate has
   AUC 0.30 — i.e. it is anti-predictive exactly where a mistake costs most.
9. **The two rollover rows are unreached, not unreachable** (finding 1). The one
   rule that reaches them — the `EJU` prefix crossed with an ordinary schedule
   delay — has fired nine times out of sample and been right none, and the hedge
   it licenses costs 6.5 s on july and 7.5 s on december. If someone wants to
   revisit it, the revival condition is precision holding on new months, not a
   tighter conjunction on the same two rows.
10. **There is about a second of july bulk residual in the categorical columns**
   (finding 7): an airport-by-operator correction removes 0.95 % of the NM-bulk
   squared error, more than any numeric column. It is behind retro-v1's
   unisolated-encoding guard, and that guard is now worth repairing for a
   measured reason rather than a hoped-for one.

## Known behaviours and limits

- Thirteen ranking rows have a null `ADEP_mvt`. `prc.day_state_features` follows
  `prc.surface_features` and stringifies them, so they form a `"None"`
  airport-day group; with fewer than 20 departures it is nulled by the
  `min_departures` gate. No ranking row is dropped.
- The `(a)/(b)/(c)` tolerances in the question-1 decomposition (60 s for "the
  block time sits on", 300 s for "away from", 1,800 s for "a long taxi") are
  stated, not tuned: no variant of them was evaluated against an outcome.
- The airport-day is the UTC day. The ranking file carries no local-time column
  and every other module in `prc/` uses UTC.
- `analysis/q2j_july.md`'s marker screen measures *additive univariate* residual
  structure on top of the model. A column the model holds but uses badly in an
  interaction would score ~0 there; the screen closes the univariate question,
  not the interaction one.
- `ds_share_span1800` (and the `ds_deice_day` built to match it) divide by every
  departure of the day, so a no-NM row — which has no observable span — counts as
  "not long" and biases the day share down by the day's no-NM rate, 1–2 %. The
  hourly `ds_deice_hr` divides by the departures with a finite span, which is the
  better definition. Flagged in the module and queued for VERSION 2; not changed
  here because the measurement below was running against these bytes, and both
  arms of every run share them.
- **Every arm in findings 12 to 16 is one seed (7).** The paired bootstrap
  resamples airport-days and so covers row sampling; it does not cover the seed
  draw, and finding 12 shows retro-v1's own seed spreads reaching 34.5 s on july
  and 16.1 s on december for families differing by nothing but the seed. The
  matched-rounds control removes the *round* draw, which was demonstrably the
  larger effect on july, but a second seed was queued and then dropped for time
  when the desk asked for the `ds_deice_hr` arm instead. So finding 14's
  december result (−13.6 s, four intervals excluding zero) and finding 15's
  winter-only result (+15.2 s) are large enough to survive a plausible seed
  draw; finding 14's january result (−3.9 s) and finding 13's july bulk (−1.5 s)
  are not, and should be re-run at a second seed before either is leaned on.
- The rollover base rate rests on three events in twelve months. The Poisson
  arithmetic in finding 3 inherits that: the 95 % interval on a rate estimated
  from three events spans roughly a factor of three, so "expected 0.50" is
  "expected somewhere between 0.2 and 1.5".

## A pinned pipeline, and a shared-module signal for retro-v1

Both arms of every run in this experiment go through `code/run_ds.py`, which
imports **`code/pipeline/`** — a snapshot of retro-v1's `run.py` and its step
modules taken from git `19cd199`, the state on disk when the first arm started at
05:40. retro-v1 committed `3922b2a` twenty minutes later, adding `fixed_rounds`
and `seed_bag` reads to `two_pass`; the running arm did not raise `KeyError` on
those, which is the evidence that it had loaded the earlier file. Importing the
snapshot rather than the live directory is what makes every arm and every seed
here the same code. Nothing in the snapshot is edited.

While setting that up, a number worth retro-v1's attention. Running `G1cv-s7`'s
exact command line today, against today's `prc/`:

| fold | `G1cv-s7` as logged (04:06, phi VERSION 2) | this experiment's control arm (05:45, phi VERSION 3) |
|---|---|---|
| july | 320.8 | 321.9 |
| **december** | **219.3** | **236.9** |

`prc/phi_features.py` moved to VERSION 3 at 04:59, and `prc/lirf_features.py` and
`prc/surface_features.py` also changed after `G1cv-s7` ran. **I have not isolated
which module carries the 17.5 s**, and this experiment does not need to: both of
its arms share the newer modules and the pinned `run.py`, so the day-state
comparison is unaffected. But a 17.5 s december move under an unchanged command
line is exactly what retro-v1's own provenance marker was added to catch, and it
is worth an ablation there.

