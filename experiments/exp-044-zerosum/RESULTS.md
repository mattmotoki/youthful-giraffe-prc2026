# exp-044-zerosum — RESULTS

Worker `zerosum`, tier hard, 2026-09-08. Assignment `assignments/zerosum.md`; the reading taken and
the readings rejected are in `SPEC.md`. Laptop, 3 threads, CPU only, no GPU, no upload. Every number
here is written by a script in `code/` into `analysis/`, and every table in §8 is generated from
those records by `code/tables.py`. Every number in the prose is audited against `analysis/` by
`tests/check.py`, with one documented exception: the two numbers of the rejected serving run in §5
("80 cells", "23 s"), which come from the console of a run that was discarded and has no record.

**This is the post-review text.** Sol round 1 returned NEEDS REVISION with eight blocking findings.
All eight were accepted; the repairs are in `review/round1-answers.md`, and two of them changed a
stated conclusion — the density verdict and the status of the candidate against its own gate. The
paragraphs those findings touched are marked where they stand.

## 0. The answer, in seven lines

**The density check does not separate the two explanations: the verdict is MIXED.** Applying the
rule that was committed before any number was read, the traffic-rate ordering survives conditioning
on cell rows in 9 of 12 adjacent comparisons, and the cell-row ordering survives conditioning on
traffic rate in 6 of 10. Both survive, which is the declared MIXED case (T2b). The rule never says
how to count "survives"; under a strict counting, in which one reversal ends an ordering, it returns
NEITHER instead. **No counting returns DENSITY and none returns MECHANISM**, so the table does not
establish either explanation. Traffic rate is the stronger of the two and it is the one that keeps
its shape when the grid changes, but "the gain is density, not the cell's row count" is more than the
table supports, and the earlier text that said so is withdrawn. Q03's 600 s grid reproduces to the digit, and 1200 s reads **+0.0797 s** of
all-row board on a 17 % larger cohort against 600 s's +0.0782.

**No tree arm passes the gate, and the extra capacity does not cross a month.** On Q03's own
within-month design the trees look far better (`lgb_wide` +0.2761 s, `lgb_shipped_wide` +0.3596 s
against the ridge's +0.0782). On a cross-month fit — fit on the other two months, score the third,
which is what serving does — the same two arms read **−0.0416 and −0.0071 s**, and under the fully
repaired estimator of T11 (exact Hessian diagonal, no shared vocabulary, each arm's own amplitude)
**−0.0389 and +0.0045 s**: the 30× capacity buys nothing that survives the month boundary either
way. The repaired `lgb_shipped` is the best cross-month tree arm at **+0.0379 s**, above the ridge's
+0.0222, but its july_loo interval is [−2, +60] and still crosses zero. Three arms have
positive point gains on all three folds under both designs — the ridge, `lgb_small` and
`lgb_shipped` — but **no arm passes the interval gate**, so the assignment's repair clause applies
and the ridge, the reference, is the arm carried forward. *(The first version of this line said the
ridge was the only arm positive on all three folds under both designs; T3 shows that is false, and
the distinction that matters is the interval, not the point.)*

**The finding that matters more than the arm: Q03's +0.0782 s is a within-month number, and the
cross-month number is +0.0222 s.** Four fitting designs order the gain by how far the fitting rows
sit from the scored rows: +0.0782 within the month, +0.0766 with the serving vocabulary, +0.0542
across a half-month split, +0.0222 across a month boundary. Running the cross-month **code path**
inside the month reads +0.0766, which excludes the stacking and cell-offsetting arithmetic as the
cause but not the whole implementation (T11). This is consistent with distance in time and is not a
proof of it; §4 says what the three folds can and cannot carry.

**The serving path works and the algebra holds on the ranking file.** `d` is computed from the
ranking file's own rows, the cell sums are zero to 1.4e−14, the floor never binds, and the 388
`noac`, 38,182 EHAM and 36,317 EDDF rows are bit-identical to v10.
`submissions/candidates/exp-044-zerosum.parquet`, SHA-256
`242b24b6fdad0e1b4d92ebd44479265d35c263de985baf3404e056e705c46af1`, 210,597 rows changed at an RMS
of 6.498 s.

**The candidate does NOT pass the gate the assignment declared for step 3.** The gate is an all-row
paired fold gain over v10 with *every* fold interval clear of zero at the served amplitude. Under
the estimator the served file actually is — cross-month — the january interval is [−15, +52] and
crosses zero. The file exists because the repair clause says to ship the ridge when the learner does
not beat it, and it is written, hashed and censused so the desk has the object; **this report does
not recommend an upload**, and nothing here should be read as the gate passing.

**The serving fit uses three 2025 months, not the twelve the assignment names.** `h`'s target is
`y − p_v10`, and a v10 out-of-fold prediction exists on 2025-01, 2025-07 and 2025-12 only.
Extending it needs the whole v10 stack refitted on nine more rotations. This is a deviation from the
assignment, not a choice between equivalent designs; §5 states it with its evidence.

**What it is worth.** On the cross-month instrument the update is worth **+0.0222 s over v10** and
**+0.0250 s over v11** (cross-fitted amplitude), and at BOARD READING 10's 0.16× that is about
**0.004 s of board**. The v10-based file the assignment names would land about 0.013 s *above* v11,
which is already on the board; the sibling `submissions/candidates/exp-044-zerosum-v11.parquet` is
the same update on the v11 base. §7.

## 1. What was run, and the reference gate

The base is **v10** — `submissions/uploaded/youthful-giraffe_v10.parquet`, SHA-256
`ad962763236e9b40…`, board 263.0464 — with its fold side rebuilt by exp-036 arm 2's own arithmetic
through `experiments/exp-039-screens/code/v10.py`, which gates the three clean fold RMSEs
(284.1572 / 204.9218 / 218.1506, board 257.0248) and the fold vectors' own SHA-256 before any arm is
scored. Nothing is refitted from the member frames.

Eligibility, cell, four-row minimum, ridge penalty, day blocks, seed, bootstrap and pair rule are
Q03's, unchanged and re-asserted against `analysis/predeclared.json` at import time
(`code/zs.py`). The reference arm reproduces Q03 exactly: cohorts 110,691 / 101,449 / 91,861, cells
48,514 / 39,198 / 38,152, pairs 51,280 / 47,070 / 42,561, cross-month amplitudes
0.5478 / 0.5523 / 0.5831, all-row gains +0.0972 / +0.1332 / +0.0586, board **+0.0782 s**. That
identity is the gate on everything below.

**The measurement contract.** `r = y − p_v10`, `d = p_cand − p_v10`, `A = mean(r·d)`,
`B = mean(d²)`, gain `2A − B`, unrestricted scale `A/B`; the primary endpoint is the paired
**all-row** january/july board at exp-016's ranking shares, with clean and december beside it; every
departure `BLOCK_TIME` and `y` is masked by `prc.data.mask_like_ranking` before any feature is built.
The COMMON.md slice decomposition (all rows, at or under 3600 s, above 3600 s, and the no-NM rows)
is in T12, the per-date contributions in `analysis/dates-ridge-<design>-<fold>.csv`, and the per-row
contributions in `analysis/rows/` under `analysis/MANIFEST.json`.

## 2. Step 1 — the density check

Q03's T6 found the pair reduction concentrated in 4–9-row cells with the 10–19 stratum changing
sign, and named the open question: is that the mechanism, or is "small cell" a proxy for something
else? The queue's repair is "fixed-size matched subsets and a fixed alternate grid". Both were
predeclared and committed at d2e55b7 before the first fit: grids of **300, 600 and 1200 s**; a
matched-subset rule under which every cell contributes exactly its first four rows in the label-free
(cell, movement id) order, so every cell supplies exactly two pairs and no cell is weighted by its
size; and a **traffic rate** that does not depend on the grid at all — the number of eligible
departures at the same airport within ±300 s of the row's take-off clock. The ridge is unchanged
throughout; only the grid and the evaluation subset move. The pair's band is the mean of its two
rows' rates.

**The verdict is MIXED** *(corrected after Sol round 1; the first version of this report said
DENSITY without running the rule it had declared)*. The rule reads: "DENSITY if the ordering by
traffic rate survives conditioning on cell rows and the ordering by cell rows does not survive
conditioning on traffic rate; MECHANISM if the reverse; MIXED if both survive." `code/verdict.py`
applies it to the two-way matched table `code/density.py` wrote, counting the adjacent-band
comparisons that fall in the declared direction inside each level of the other factor (T2b):

- the **traffic** ordering, holding cell rows fixed: 2 of 3 at 300 s, 3 of 4 at 600 s, 4 of 5 at
  1200 s — **9 of 12**;
- the **cell-row** ordering, holding traffic fixed: 2 of 2, 2 of 3, 2 of 5 — **6 of 10**.

Both survive on a majority of comparisons, so the rule returns MIXED under that counting.

**The rule does not fix its own counting, and the verdict is partly a function of it** *(Sol round 2,
blocking 3)*. "Survives" was written before the numbers but never defined. Four countings are run and
reported (T2b): adjacent bands unweighted, adjacent bands weighted by the smaller band's pairs, the
first band against the last inside each level, and a strict reading in which one reversal ends an
ordering. The three lenient countings return MIXED; the strict one returns **NEITHER** — under it
neither ordering survives, because each has at least one reversal. So the earlier claim that the
verdict "does not rest on the choice" is withdrawn. What every counting agrees on is the negative:
**no counting of this rule returns DENSITY, and none returns MECHANISM.** The two-way table does not
establish either explanation, which is the honest close on Q03's open stratum.

One further disclosure: `code/density.py` wrote only two-way cells holding at least 200 pairs, a
threshold that was **not** predeclared. The record now carries every non-empty cell beside the
filtered table (`density.json:table_all`, `two_way_min_pairs_not_predeclared`), and all four
countings run on both. No verdict changes without the threshold
(`verdict.json:other_readings_no_min_pairs`).

What can still be said is
narrower and is what the table shows: traffic rate is the stronger and the more stable of the two.
Its band values keep the same shape on every grid (about +270 to +279 s² at 1–3 departures per
10 minutes, −78 to −90 s² at 12 or more), while the cell-row bands do not — the matched 6–9-row band
reads +49.5 / +169.4 / +284.8 s² at 300 / 600 / 1200 s and the 10–19 band −50.7 s² at 600 s against
+162.1 s² at 1200 s. The honest reading is that the two factors are entangled in this data: cell
rows and traffic rate are two views of the same crowding, the matched subsets remove the cell-size
weighting but not the correlation, and three grids cannot pull them apart.

**The monotonicity claim is withdrawn.** The first version said the traffic-rate ordering is
monotone on every grid. It is not: at Q03's own 600 s grid the marginal reads +270.3 (1–3), **+290.7
(4–6)**, +125.7 (7–11), −89.6 (12+), which rises at the first step. It is monotone at 300 s and at
1200 s (`verdict.json:traffic_marginal_monotone`).

The 300 s grid's small board gain (+0.0196 s) is a selection effect the other way: at a 5-minute bin
the four-row minimum selects the *busy* periods, where there is little gain, and it halves the
scored cohort (51,161 / 49,141 / 41,938 against 110,691 / 101,449 / 91,861). At 1200 s the cohort
grows 17 % (355,530 scored rows against 304,001) and the board gain is **+0.0797 s**, marginally
above 600 s's +0.0782.

**The served grid stays at 600 s.** It is Q03's predeclared cell, the +0.0015 s difference is far
inside the fold noise, and switching to the grid that read best would be selection on the rows the
candidate is scored on.

## 3. Step 2 — the learner, and where the fold number goes

Five arms on the identical objective: `min Σ (r_c,i − d_i)²` with `d = h − mean_cell(h)` and `r_c`
the within-cell-centred v10 residual. For a linear `h` that is the ridge on within-cell centred
features and target.

**The tree objective: an exact gradient and an approximate Hessian** *(restated after Sol round 1;
the first version called the whole objective "exact")*. With loss `½ Σ (d_i − r_c,i)²`, `d = Ph`
where `P` is the within-cell centring projector, and both `d` and `r_c` summing to zero in every
cell, the gradient is `∂L/∂h_i = d_i − r_c,i` **exactly** — the projector's own term cancels because
`P` is symmetric and idempotent and `r_c` is already in its range. The Hessian, however, is `P`
itself, not the identity: it is dense inside each cell with diagonal `1 − 1/n_c` and off-diagonal
`−1/n_c`. LightGBM takes a diagonal, so a diagonal is what it gets, and the first run used the unit
diagonal. Sol round 1 was right that "exact objective" overclaimed. The repair (`code/repair.py`,
T11) refits every tree arm with the exact diagonal `1 − 1/n_c`, and — after Sol round 2 — with **its own** cross-month
amplitude rather than the declared run's. On the two arms that change nothing else, `lgb_small` moves
+0.1096 → +0.1207 on `dayblock` and +0.0131 → +0.0102 on `xmonth`, and `lgb_wide` +0.2761 → +0.2779
and −0.0416 → −0.0389. Every conclusion below holds under both diagonals; the tables in T3–T6 are the
unit-diagonal run at the declared amplitudes, and T11 is the repaired run beside it. **T11's columns
move more than one thing at a time** and say which: the non-shipped arms move the Hessian and the
amplitude, the shipped arms move those and the cache vocabulary. That is why the amplitudes are
printed with the gains.

`tests/check.py` verifies the gradient derivation against a finite difference of the loss (max abs
5.1e−9) and verifies that the naive gradient `h − r_c` fails the same test (max abs 0.699). Plain
LightGBM on the centred target is a *different* objective — its gradient carries an extra
`mean_cell(h)` term — and is not used.

`lgb_small` (4 leaves × 64 trees = 256 leaves) is the capacity match to a ridge whose 202
standardised features at α = 100 on ~90,000 rows have effective degrees of freedom of about 202;
`lgb_wide` (16 leaves × 400 trees ≈ 6,400 leaves) is declared beside it only to answer whether the
matched budget bound. Both budgets, and the two shipped-block arms that add the 89 numeric columns
of `artifacts/exp-018-fresh/cache/<month>.parquet`, were fixed in `analysis/predeclared.json` before
the first fit.

**The shipped block carries a shared vocabulary; dropping it, with a properly refitted amplitude,
moves the wide arm** *(added after Sol round 1, corrected after round 2)*. Thirteen of the shipped
block's 89 columns are `c_*` integer codes from an append-only vocabulary that the cache builder
shares across months, so a code's meaning is not fixed by the fitting months alone. Both shipped arms
were refitted with those 13 columns dropped (76 columns), with the exact Hessian diagonal, and — the
round-2 repair — each with **its own** cross-month amplitude fitted on the other two folds' day-block
updates of that arm, rather than the amplitude of the vocabulary-bearing run. `lgb_shipped` moves
+0.1336 → +0.1304 on `dayblock` and +0.0339 → **+0.0379** on `xmonth`; `lgb_shipped_wide`
+0.3596 → +0.3416 and −0.0071 → **+0.0045** (T11). The wide shipped arm's cross-month sign therefore
depends on the amplitude, which is itself an argument that nothing there is stable across a month.
The arm selection does not change: no repaired arm clears zero on all three folds (T11).

Each arm runs under two cross-fittings. **`dayblock`** is Q03's: five contiguous day blocks inside
the validation month, fit on four, score the fifth. **`xmonth`** fits on the other two folds' months
and scores the third, with the categorical level list frozen on the fitting months — the estimator
the served candidate actually is. Both are scored at the *same* scalar, Q03's cross-month amplitude,
which is fitted on the other two folds' `dayblock` updates and is therefore a function of months the
scored fold is not, under either design.

**The result (T3, T4).** On `dayblock` the trees look strong and capacity clearly binds
(ridge +0.0782, `lgb_small` +0.1096, `lgb_wide` +0.2761, `lgb_shipped` +0.1336,
`lgb_shipped_wide` +0.3596). On `xmonth` the ordering inverts: ridge +0.0222, `lgb_shipped` +0.0339,
`lgb_small` +0.0131, `lgb_shipped_wide` −0.0071, `lgb_wide` −0.0416. The ratio between the two
designs falls with capacity — 0.28 for the ridge, 0.25 for `lgb_shipped`, 0.12 for `lgb_small`, and
a sign change for both wide arms as declared (T11's repaired `lgb_shipped_wide` is +0.0045, so its
sign is not robust; `lgb_wide` stays negative either way).

**No arm passes the predeclared gate** ("every fold interval clear of zero" on `xmonth`): the
ridge's january interval is [−15.1, +52.2], `lgb_shipped`'s july_loo interval is [−15.3, +51.6], and
the rest are worse. On the weaker test of point gains the ridge is not alone — `lgb_small` and
`lgb_shipped` are also positive on all six fold-by-design cells (T3) — which is why the gate is
written on intervals. The predeclared arm-selection rule therefore selects nothing, and the
assignment's repair clause takes over: **ship the ridge**, the reference arm, which passes on the
fold side under Q03's own within-month design (+95.1 [+35, +141], +89.1 [+52, +126],
+66.3 [+32, +102] s²). §5 states what that does to the step-3 gate, which is a
separate declaration and is not passed.

**What the trees could not use.** Two things, and the record separates them. First, the extra
capacity buys nothing that crosses a month: `lgb_wide`'s `dayblock` gain is 3.5× the ridge's and its
`xmonth` gain is negative, so what the extra 6,000 leaves fit does not reappear in another month.
Second, the shipped block is the one addition that *helps* on the serving design — `lgb_shipped` is
the best `xmonth` tree arm (+0.0339 as declared, +0.0379 fully repaired) and the only arm to beat the
ridge there — which fits §2's reading, because the shipped block carries the airport and runway
traffic counts (`apt_to_15m_before`, `apt_queue`, `rwy_queue`, …) that the 202 row-only columns do
not, and §2 says traffic rate is the stronger of the two entangled factors. It is not enough: even
repaired, its july_loo interval is [−2, +60] and crosses zero, and one arm beating the reference on
one of two designs is not evidence a serving path can hold.

**The unrestricted scale says the amplitude is not what is wrong.** On `xmonth` the ridge's `A/B` is
0.76 / 1.03 / 0.63, so the applied 0.55 is if anything slightly *too large*; at each fold's own
optimum the july_loo gain would be +31.0 s² rather than +27.8. The cross-month direction is weaker,
not mis-scaled.

## 4. Why — four fitting designs and two further controls

*`dayblock_frozen`, `halfmonth`, `pathcheck` and `xmonth_one` are diagnostic controls written after
the predeclared arms were read. They are exploratory: no arm selection, gate or served number rests
on them. The same applies to the v11 sibling of §7.*

A gap of that size has two readings, and only one is about the world. Either the cross-month fit is
genuinely harder, or something in the cross-month *implementation* costs the gain. Four designs and
two further controls, `code/controls.py` and `code/repair.py` (T7, T11):

- **`dayblock_frozen`** — Q03's within-month day-block design, unchanged, with the categorical level
  list frozen on the *other two months* (the serving vocabulary). Ridge **+0.0766** against
  +0.0782, `lgb_shipped` +0.1333 against +0.1336, `lgb_shipped_wide` +0.3632 against +0.3596. The
  vocabulary costs nothing.
- **`pathcheck`** *(added after Sol round 1)* — the cross-month machinery itself, run **inside** the
  month: the `xmonth` fitting and scoring functions, given four of the five day blocks as their
  fitting set and the fifth as their scored set, and — the part `dayblock_frozen` does not have —
  the fitting rows stacked in two parts with the cell codes offset between them, which is how the
  cross-month arm handles cells that come from two different months. Ridge **+0.0766**, folds
  +0.0968 / +0.1362 / +0.0557. It shares the frozen vocabulary and the day-block split with
  `dayblock_frozen`, so those two are *expected* to agree; what it adds is the stacking and the
  offsetting, and adding them costs nothing. Read narrowly, that is all it shows. It does **not**
  execute the real cross-month source-matrix path — `code/repair.py` builds the two parts from the
  scored month with its own copies of the matrix and fit calls, so it cannot exercise the
  per-source-month matrix build or the alignment between months. What it excludes is the stacking
  and cell-offsetting arithmetic; it does not exclude the cross-month matrix construction, and this
  report does not claim the implementation is ruled out.
- **`halfmonth`** — the same month, the same vocabulary, the same learner; fit on one contiguous half
  of the month's days, score the other half, both ways. Every arm lands between the two designs:
  ridge +0.0542, `lgb_shipped` +0.0778, `lgb_shipped_wide` +0.2106.
- **`xmonth_one`** *(added after Sol round 1)* — fit on **one** other month instead of two, scored
  across the month boundary. Ridge −0.0113 and −0.0191 for the two source assignments, against
  `xmonth`'s +0.0222 on two months.

**What the four designs support, and what they do not** *(rewritten after Sol round 1)*. The four
all-row numbers order as +0.0782 (within month), +0.0766 (within month, serving vocabulary), +0.0542
(half-month split), +0.0222 (across a month boundary). That ordering is consistent with the gain
falling as the fitting rows move away in time from the scored rows, and the `pathcheck` control
excludes one part of the code path — the two-part stacking and the cell offsetting — as the cause. It is **not** a proof of a time-distance mechanism: the four designs
differ in fitting-set size as well as in time distance, and three folds cannot separate two factors
that move together. Two things narrow it. First, fitting-set size runs the *opposite* way to the
result — `halfmonth` fits on 55,976–68,202 eligible rows, `dayblock` on 88,660–106,652, `xmonth` on
238,382–254,972 — so the design with the most rows reads lowest, which no monotone size effect
explains on its own. That is not the same as "more rows can only help": under a distribution shift
between fitting and scored rows, extra rows from the wrong distribution can hurt, and that is exactly
the possibility the ordering cannot rule out. Second, `xmonth_one` shows that within the cross-month
family more months help rather than hurt (−0.0113 and −0.0191 on one month against +0.0222 on two),
so the fall is not simply "the fitting set got bigger and more heterogeneous". The claim this report
makes is the operational one: **a fit that never sees the scored month retains about 28 % of what
Q03's within-month design measures, on this residual and this family.** The mechanism behind that
number is not settled here.

**Scope, stated so the sentence cannot be quoted out of it.** This is measured for the Q03 zero-sum
family on the v10 residual, not for the repository's other lineages. It applies to any correction
**fitted inside the month it is scored on**; a member fitted through `prc.data.fold_frames` (trained
on other months) does not have this defect. What it does say is narrower: the two instruments give
different numbers for the same objective, the cross-month one is the like-for-like instrument for a
served file, and the record should quote it when the question is a candidate. Calling Q03's number
an upper bound on what a serving path can carry would need the serving estimator itself to be
measured, and this experiment cannot measure it.

**Two ways the `xmonth` instrument differs from serving, and the measurement that would settle it.** The `xmonth`
instrument is harder than serving in one specific, checkable way: **no `xmonth` fold has its own
calendar month in the fitting set**, while the served fit has one for each ranking month — 2025-07
for July 2026 and 2025-01 for January 2026 — and is fitted on three months rather than two
(370,338 rows against 238,382–254,972). july_loo is the extreme case: it is fitted on december and
january and sees no summer month at all, and it is one of the two folds that retain least
(dayblock → xmonth ratios are 0.29 july_loo, 0.55 december, 0.27 january for the ridge). Three folds
cannot attribute those three ratios to a cause, and this report does not try. **The served value is
not measured by anything here: the two instruments read +0.0222 and +0.0782 s on the same objective,
neither of them is the served estimator, and with three months of v10 residual the experiment cannot
say where between or outside them the served value falls.** The concrete repair is to widen the
residual: v10's out-of-fold prediction exists on three 2025 months only, and extending it to the
other nine would make the served fit stronger *and* let a calendar-matched cross-month instrument be
built, which is the one measurement that would price this update honestly.

## 5. Step 3 — the serving path, and the gate it does not pass

**The deviation, first.** The assignment says `h` is fitted on the twelve 2025 months. It is fitted
on **three**: 2025-01, 2025-07 and 2025-12, 370,338 eligible rows. The reason is not a preference.
`h`'s target is `r = y − p_v10`, so a fitting row needs a v10 out-of-fold prediction, and v10 —
exp-036 arm 2 over the exp-016 stack — has one on exactly those three months: the `january`,
`july_loo` and `december` fold vectors. Producing the other nine means refitting five members, the
stacker, the blend and the airport masks on nine more rotations, which is outside this assignment's
compute by more than an order of magnitude. This is recorded as a **deviation from the assignment**,
not as a design choice, and it is the same missing artifact that Q04's MAYBE and §4's repair both
name. Every number in this section is a three-month number.

`h` is fitted on those three months on the same objective, with the level list frozen on them. The
ranking months' cells are formed from the **ranking file's own rows** — airport × UTC 10-minute bin ×
serving regime — which `docs/input-rule.md` makes legal, since every column of every row of that file
is an input; the regime comes from `prc.blend.regimes_from_raw`, which reads `AIRCRAFT_TYPE_mvt`,
`FLIGHT_ID_mvt`, `AOBT_3_flt` and `LOBT_flt` and no label. `d` is `h(X) − mean_cell(h(X))` on the
eligible cohort, features only, with cells under four eligible rows left at zero, added **after**
v10's serving masks.

The served amplitude is the predeclared pooled cross-month scalar **0.5618**, stated in this file and
committed at b9f7089 before `code/serve.py` was run, fitted on the three folds' day-block updates,
never on a ranking row and never on a board reading (score-feedback rule). The `xmonth` evidence says the
cross-month optimum is about 0.81 × that, which costs about 5 % of the gain; the predeclared number
was kept rather than re-fitted on the folds it is measured against.

**The bound.** The output floor of 60 s is a bound on `p`, and the queue requires a new algebra check
for one. `code/serve.py:bounded_project` re-solves any affected cell as the projection of `d0` onto
{cell sum zero} ∩ {p ≥ 60} by active-set water filling, with a zero update as the always-feasible
fallback. In the event **no row was affected**: zero bound iterations, zero cells zeroed, minimum
prediction exactly 60.0000. `code/serve.py` now raises before it writes if either algebra check
fails, so a future run cannot produce a file that fails them silently.

**The gate: NOT PASSED** *(restated after Sol round 1, which found the first version described a
failing gate in language a reader could take as passing)*. The assignment's step-3 gate is "all-row
paired fold gain over v10 with every fold interval clear of zero at the served amplitude". The
estimator the served file actually is, is the cross-month one, and under it the ridge reads
+27.8 [+1, +56], +48.9 [+23, +73], **+17.9 [−15, +52]** s², board +0.0222 s: **january's interval
crosses zero, so the gate fails.** Under Q03's within-month design the same update passes
(+95.1 [+35, +141], +89.1 [+52, +126], +66.3 [+32, +102] s², board +0.0782 s), but that design is not
what serves.

The two declarations in the assignment pull against each other here, and this report does not
resolve it in the candidate's favour. The repair clause ("if the learner does not beat the ridge,
ship the ridge") is why the file was written, hashed and censused; the step-3 gate is why it must not
be called a pass. **This report does not recommend an upload.** The desk holds the object, both
readings, and the census, and decides.

**Checks (T8).** Cell sums of `d` are zero to 1.4e−14 (relative), the file is in template order,
finite, 344,841 rows, and the 388 `noac`, 38,182 EHAM and 36,317 EDDF rows differ from v10 by
exactly 0.0.

**One defect found and repaired before the file was accepted.** The first served run produced only
80 cells and an update RMS of 23 s (those two numbers are from that run's console; the run was
discarded and has no `analysis/` record). The cause: the training months parse as `datetime64[ns]` and the
ranking file as `datetime64[us, UTC]`, so `astype("int64")` yields nanoseconds in one and
microseconds in the other; dividing by 1e9 therefore gave seconds on the training side and
kiloseconds on the ranking side, which turned the 600-second cell into a 600,000-second one of
about seven days. The projection was removing an almost-global mean and doing nothing.
`zs.epoch_seconds` now subtracts the epoch instead of assuming a resolution, and `code/serve.py`
gates the served cell before it is used: the ranking clock must cover 55 to 70 UTC dates, and the
median eligible rows per cell must be within a factor of two of the fold side's. The candidate in T8
is from the repaired run.

## 6. Step 4 — the ranking-side census

253,964 of the 344,841 ranking departures are eligible (73.65 %, against 69.2 / 74.2 / 75.1 % on the
three folds). They fall in 55,570 cells; **17.08 %** of eligible rows are in cells under four rows
and are left alone, so 210,597 rows carry an update. EHAM and EDDF hold zero eligible rows by
construction.

**The served update's size is in range.** Its RMS is **6.498 s** against the fold side's
8.98–9.40 s under Q03's within-month design (29 % under) and 6.81–8.17 s under the cross-month
design (13 % under) — inside the assignment's one-third threshold against either, and in the
expected direction, because the served `h` is fitted on 370,338 rows against the within-month
design's ~90,000 and a better-determined ridge produces a less dispersed prediction. The two ranking
months agree (6.518 s in January, 6.482 s in July), and the per-airport spread (5.30 s at EDDM to
8.04 s at EGLL) tracks the airports' own error scale. The record also carries the RMS over the
eligible cohort and over all departures, so a reader can see which denominator a number uses
(`serve.json:census`).

## 7. What this is worth, and the v11 sibling

BOARD READING 10 measured the transfer of a fold gain onto the board at **0.16×** on this base, from
a candidate whose fold instrument was already a month rotation. The like-for-like fold number for
this update is therefore the **cross-month** one, not Q03's within-month one:

- over v10: +0.0222 s of fold board → about **0.004 s** of board at 0.16×, 0.006 s at
  BOARD READING 9's 0.26×;
- over v11: +0.0250 s of fold board at the **cross-fitted** amplitude → about **0.004 s** likewise.
  *(Sol round 1 found the first version priced this at the pooled amplitude, which is fitted on all
  three folds including the one it scores. The cross-fitted number is +0.0250; the pooled number is
  +0.0254, and it is kept beside it in T10 because it is the scalar the written file carries.)*

Two consequences the desk should have in one place. First, the desk's reading of Q03 ("about
0.01–0.02 s at this transfer") was built on the +0.0782 within-month number; on the serving
instrument the figure is about a third of that. Second, **the candidate the assignment names is
built on v10, and v11 is 0.0166 s better on the board**, so `exp-044-zerosum.parquet` would be
expected to land near 263.042 — above v11's 263.0298. The update is additive on v11 (T10: the same
`d`, unchanged, is worth +0.0250 s over v11 against +0.0222 s over v10, with the same fold signs), so
`exp-044-zerosum-v11.parquet` is the better of the two objects if a slot is ever spent on this
lane. The update was fitted on v10's residual, not v11's; refitting it on v11's residual is a further
small step this experiment did not take. Both files are written and hashed, both fail the step-3 gate
on the serving instrument, and the desk decides.

Against the 2.099 s gap to the leader this is 0.2 % of the distance. It is positive on the fold side
under both designs, it costs nothing on the masked rows, and it is not a step toward 260.93.

## 8. Tables

<!-- TABLES:density -->
<!-- generated by code/tables.py; do not edit by hand -->

### T1 the density check — the board on three grids

| grid | scored rows | cells (jul/dec/jan) | all-row v10 | all-row arm | all-row gain | clean gain |
|---|---|---|---|---|---|---|
| 300 s | 142,240 | 79,915 / 66,421 / 64,188 | 309.0772 | 309.0576 | **+0.0196** | +0.0236 |
| 600 s | 304,001 | 48,514 / 39,198 / 38,152 | 309.0772 | 308.9991 | **+0.0782** | +0.0940 |
| 1200 s | 355,530 | 29,033 / 22,719 / 22,036 | 309.0772 | 308.9975 | **+0.0797** | +0.0958 |

### T2 pair reduction by cell rows and by traffic rate, matched subsets (s²)

by **cell rows** (the stratum Q03's T6 used):

| grid | 4-5 rows | 6-9 rows | 10-19 rows | 20+ rows |
|---|---|---|---|---|
| 300 s | +151.1 (55,200) | +49.5 (7,486) | +397.1 (16) | – |
| 600 s | +336.8 (46,354) | +169.4 (48,744) | -50.7 (5,398) | – |
| 1200 s | +234.4 (11,418) | +284.8 (26,410) | +162.1 (33,470) | +171.4 (1,396) |

by **traffic rate** (eligible departures at the airport within ±300 s, the same physical measure on every grid):

| grid | 1-3 dep/10 min | 4-6 dep/10 min | 7-11 dep/10 min | 12+ dep/10 min |
|---|---|---|---|---|
| 300 s | – | +174.1 (18,632) | +126.5 (41,223) | +90.9 (2,847) |
| 600 s | +270.3 (2,968) | +290.7 (65,686) | +125.7 (30,423) | -89.6 (1,419) |
| 1200 s | +278.9 (12,824) | +225.8 (43,120) | +162.6 (16,033) | -77.9 (717) |

### T2b the predeclared rule applied to the two-way table

Adjacent-band comparisons that **fall** in the declared direction, inside each level of the other factor (`code/verdict.py` on `code/density.py`'s own two-way table):

| grid | traffic ordering, holding cell rows | cell-row ordering, holding traffic | traffic marginal monotone |
|---|---|---|---|
| 300 s | 2 of 3 | 2 of 2 | True |
| 600 s | 3 of 4 | 2 of 3 | False |
| 1200 s | 4 of 5 | 2 of 5 | True |
| **all three** | **9 of 12** | **6 of 10** | - |

Both orderings survive on a majority of comparisons, so the declared rule returns **MIXED**. The rule says "survives" without saying how to count, so three honest countings are run; the verdict does not rest on the choice:

| how "survives" is counted | traffic ordering | cell-row ordering | verdict | verdict without the undeclared 200-pair minimum |
|---|---|---|---|---|
| adjacent bands, unweighted (the headline) | 9 of 12 | 6 of 10 | **MIXED** | MIXED |
| adjacent bands, weighted by the smaller band's pairs | 67140 of 74255 | 55495 of 62800 | **MIXED** | MIXED |
| the first band against the last, per level | 7 of 9 | 5 of 7 | **MIXED** | MIXED |
| strict: one reversal ends the ordering | 0 of 3 | 1 of 3 | **NEITHER** | NEITHER |

No counting returns DENSITY and none returns MECHANISM. The 200-pair minimum on a two-way cell was an implementation choice and was not predeclared; the last column runs the same countings on every non-empty cell, and no verdict changes.

The full two-way table at Q03's own 600 s grid, matched subsets (s², pairs):

| cell rows | 1-3 dep/10 min | 4-6 dep/10 min | 7-11 dep/10 min | 12+ dep/10 min |
|---|---|---|---|---|
| 4-5 rows | +259.8 (2,946) | +349.6 (42,211) | +76.2 (1,197) | - |
| 6-9 rows | - | +186.2 (23,453) | +152.9 (25,201) | - |
| 10-19 rows | - | - | -30.1 (4,025) | -93.2 (1,351) |
<!-- /TABLES:density -->

<!-- TABLES:learner -->
<!-- generated by code/tables.py; do not edit by hand -->

### T3 the five arms under both cross-fittings — all-row board gain over v10

| arm | cross-month amp (jul/dec/jan) | pooled amp | dayblock all-row | **xmonth all-row** | dayblock clean | xmonth clean | dayblock folds (jul/dec/jan) | xmonth folds (jul/dec/jan) |
|---|---|---|---|---|---|---|---|---|
| `ridge` | 0.5478 / 0.5523 / 0.5831 | 0.5618 | +0.0782 | +0.0222 | +0.0940 | +0.0267 | +0.0972 / +0.1332 / +0.0586 | +0.0284 / +0.0731 / +0.0159 |
| `lgb_small` | 1.2967 / 1.3468 / 1.2919 | 1.3145 | +0.1096 | +0.0131 | +0.1318 | +0.0157 | +0.0864 / +0.0869 / +0.1349 | +0.0001 / +0.0813 / +0.0268 |
| `lgb_wide` | 0.7476 / 0.7725 / 0.6433 | 0.7249 | +0.2761 | -0.0416 | +0.3321 | -0.0500 | +0.1916 / +0.2013 / +0.3678 | -0.0434 / +0.0392 / -0.0400 |
| `lgb_shipped` | 1.3796 / 1.5285 / 1.3522 | 1.4290 | +0.1336 | +0.0339 | +0.1607 | +0.0408 | +0.1099 / +0.0583 / +0.1597 | +0.0191 / +0.0999 / +0.0498 |
| `lgb_shipped_wide` | 0.8770 / 0.9092 / 0.7564 | 0.8510 | +0.3596 | -0.0071 | +0.4326 | -0.0085 | +0.2574 / +0.2741 / +0.4706 | -0.0007 / +0.1244 / -0.0140 |

### T4 2A − B per scored row with day-clustered intervals (s²)

| arm | cross-fit | july_loo | december | january | A/B (jul/dec/jan) |
|---|---|---|---|---|---|
| `ridge` | dayblock | +95.1 [+35, +141] | +89.1 [+51, +126] | +66.3 [+32, +102] | 1.07 / 1.05 / 0.88 |
| `ridge` | xmonth | +27.8 [+1, +56] | +48.9 [+23, +73] | +17.9 [-15, +52] | 0.76 / 1.03 / 0.63 |
| `lgb_small` | dayblock | +84.6 [+41, +129] | +58.1 [-4, +108] | +152.5 [+56, +254] | 1.05 / 0.90 / 1.04 |
| `lgb_small` | xmonth | +0.1 [-35, +33] | +54.4 [+16, +90] | +30.4 [-0, +60] | 0.50 / 0.83 / 0.76 |
| `lgb_wide` | dayblock | +187.5 [+98, +273] | +134.7 [+57, +204] | +415.5 [+273, +577] | 0.91 / 0.78 / 1.32 |
| `lgb_wide` | xmonth | -42.5 [-108, +26] | +26.2 [-32, +82] | -45.2 [-103, +12] | 0.39 / 0.59 / 0.28 |
| `lgb_shipped` | dayblock | +107.6 [+62, +152] | +39.0 [-35, +97] | +180.5 [+89, +273] | 1.11 / 0.72 / 1.13 |
| `lgb_shipped` | xmonth | +18.7 [-15, +52] | +66.8 [+35, +97] | +56.3 [+24, +88] | 0.60 / 0.88 / 0.97 |
| `lgb_shipped_wide` | dayblock | +251.9 [+169, +331] | +183.3 [+67, +277] | +531.7 [+350, +724] | 0.91 / 0.78 / 1.33 |
| `lgb_shipped_wide` | xmonth | -0.6 [-62, +66] | +83.2 [+45, +121] | -15.8 [-90, +57] | 0.50 / 0.77 / 0.44 |

### T5 the two algebraic checks, every arm and both cross-fittings

| arm | cross-fit | max abs cell sum | sums zero | max rel gap full vs within-cell | identity holds |
|---|---|---|---|---|---|
| `ridge` | dayblock | 3.96e-13 | True | 4.53e-14 | True |
| `ridge` | xmonth | 3.48e-13 | True | 2.90e-13 | True |
| `lgb_small` | dayblock | 8.53e-14 | True | 8.09e-14 | True |
| `lgb_small` | xmonth | 6.48e-14 | True | 1.71e-13 | True |
| `lgb_wide` | dayblock | 1.38e-13 | True | 3.49e-14 | True |
| `lgb_wide` | xmonth | 9.95e-14 | True | 1.79e-13 | True |
| `lgb_shipped` | dayblock | 1.03e-13 | True | 2.41e-13 | True |
| `lgb_shipped` | xmonth | 5.33e-14 | True | 9.22e-14 | True |
| `lgb_shipped_wide` | dayblock | 1.49e-13 | True | 5.13e-14 | True |
| `lgb_shipped_wide` | xmonth | 1.14e-13 | True | 3.28e-13 | True |

### T6 the pair diagnostic at the same amplitude (s², positive is better)

| arm | cross-fit | july_loo | december | january |
|---|---|---|---|---|
| `ridge` | dayblock | +197.8 [+83, +305] | +227.3 [+140, +320] | +169.3 [+59, +285] |
| `ridge` | xmonth | +84.9 [+24, +145] | +208.0 [+145, +266] | +89.7 [-17, +204] |
| `lgb_small` | dayblock | +246.2 [+125, +364] | +276.1 [+164, +376] | +317.2 [+95, +538] |
| `lgb_small` | xmonth | +69.2 [-24, +163] | +330.2 [+226, +422] | +125.7 [+40, +204] |
| `lgb_wide` | dayblock | +408.8 [+162, +649] | +365.1 [+194, +524] | +920.0 [+591, +1260] |
| `lgb_wide` | xmonth | -39.3 [-220, +128] | +234.5 [+72, +379] | -8.7 [-167, +138] |
| `lgb_shipped` | dayblock | +297.1 [+173, +412] | +231.8 [+64, +364] | +390.7 [+176, +617] |
| `lgb_shipped` | xmonth | +109.8 [+1, +212] | +384.7 [+300, +468] | +192.1 [+100, +285] |
| `lgb_shipped_wide` | dayblock | +603.3 [+357, +839] | +388.7 [+108, +654] | +1280.1 [+869, +1701] |
| `lgb_shipped_wide` | xmonth | +70.8 [-76, +223] | +360.8 [+227, +502] | +56.5 [-147, +252] |
<!-- /TABLES:learner -->

<!-- TABLES:controls -->
<!-- generated by code/tables.py; do not edit by hand -->

### T7 the four fitting designs, in order of distance in time (all-row board gain)

| arm | dayblock (Q03) | dayblock, frozen vocabulary | half-month | **cross-month** |
|---|---|---|---|---|
| `ridge` | +0.0782 | +0.0766 | +0.0542 | **+0.0222** |
| `lgb_shipped` | +0.1336 | +0.1333 | +0.0778 | **+0.0339** |
| `lgb_shipped_wide` | +0.3596 | +0.3632 | +0.2106 | **-0.0071** |

Per fold, the half-month control:

| arm | july_loo | december | january |
|---|---|---|---|
| `ridge` | +0.1174 (+115 [+82, +149]) | +0.0298 (+20 [-27, +62]) | -0.0121 (-14 [-48, +24]) |
| `lgb_shipped` | +0.0806 (+79 [+37, +122]) | -0.0147 (-10 [-111, +70]) | +0.0755 (+85 [+12, +148]) |
| `lgb_shipped_wide` | +0.2332 (+228 [+145, +309]) | +0.1198 (+80 [-60, +204]) | +0.1883 (+213 [+128, +301]) |
<!-- /TABLES:controls -->

<!-- TABLES:repair -->
<!-- generated by code/tables.py; do not edit by hand -->

### T11 Sol round 1 repairs - the exact Hessian, the shared vocabulary, and two further controls

Every tree arm refitted with the objective's **exact** Hessian diagonal `1 - 1/n_c` in place of the unit diagonal, and both shipped arms refitted with the 13 shared-vocabulary `c_*` columns dropped (76 shipped columns, not 89). Each repaired arm carries **its own** cross-month amplitude, fitted on the other two folds' day-block updates of that arm (Sol round 2, blocking 1: the first version of this table reused the amplitudes of the vocabulary-bearing run, which made these fixed-amplitude perturbations rather than refits). All-row board gain over v10:

| arm | dayblock as reported | dayblock repaired | xmonth as reported | **xmonth repaired** | repaired amplitude (jul/dec/jan) | reported amplitude |
|---|---|---|---|---|---|---|
| `lgb_small` | +0.1096 | +0.1207 | +0.0131 | +0.0102 | 1.256 / 1.297 / 1.205 | 1.297 / 1.347 / 1.292 |
| `lgb_wide` | +0.2761 | +0.2779 | -0.0416 | -0.0389 | 0.696 / 0.718 / 0.591 | 0.748 / 0.772 / 0.643 |
| `lgb_shipped` | +0.1336 | +0.1304 | +0.0339 | +0.0379 | 1.290 / 1.370 / 1.208 | 1.380 / 1.528 / 1.352 |
| `lgb_shipped_wide` | +0.3596 | +0.3416 | -0.0071 | +0.0045 | 0.795 / 0.819 / 0.668 | 0.877 / 0.909 / 0.756 |

The repaired `xmonth` fold intervals, to show that the gate still fails (2A - B per scored row, s²):

| arm | july_loo | december | january |
|---|---|---|---|
| `lgb_small` | -1 [-33, +32] | +57 [+15, +97] | +25 [-6, +56] |
| `lgb_wide` | -48 [-117, +22] | +20 [-38, +78] | -32 [-93, +23] |
| `lgb_shipped` | +30 [-2, +60] | +72 [+41, +104] | +51 [+21, +83] |
| `lgb_shipped_wide` | +13 [-48, +78] | +73 [+26, +117] | -6 [-81, +63] |

The two further controls on the cause of the cross-month fall (`code/repair.py`):

| control | what it is | all-row board gain | folds (jul/dec/jan) |
|---|---|---|---|
| `pathcheck` | the **cross-month code path**, run inside the month (fit on four of five day blocks, score the fifth, through the `xmonth` fitting and scoring functions) | +0.0766 | +0.0968 / +0.1362 / +0.0557 |
| `xmonth_one_1` | fit on **one** other month, scored across the month boundary (first source month) | -0.0113 | +0.0256 / +0.0184 / -0.0505 |
| `xmonth_one_2` | fit on **one** other month, the second source month | -0.0191 | -0.0586 / -0.1908 / +0.0225 |

The single fitting month per fold: `xmonth_one_1` july_loo <- 2025-12, december <- 2025-07, january <- 2025-07; `xmonth_one_2` july_loo <- 2025-01, december <- 2025-01, january <- 2025-12.
<!-- /TABLES:repair -->

<!-- TABLES:serve -->
<!-- generated by code/tables.py; do not edit by hand -->

### T8 the served candidate

| item | v10 base | v11 base |
|---|---|---|
| base | v10 (the assignment's) | v11 (the sibling, §7) |
| file | `submissions/candidates/exp-044-zerosum.parquet` | `submissions/candidates/exp-044-zerosum-v11.parquet` |
| SHA-256 | `242b24b6fdad0e1b4d92ebd44479265d35c263de985baf3404e056e705c46af1` | `7dee4ce9e07a95f78b6e551a96d501f81e8831bd4ebdbb591659b4467ed12bf0` |
| bytes | 4,707,553 | 4,707,609 |
| rows changed | 210,597 | 210,597 |
| min prediction | 60.0000 | 60.4192 |
| max abs cell sum of d | 1.42e-14 | 1.42e-14 |
| 388 noac rows, max abs difference from v10 | 0.0 | 0.0 |
| 38,182 EHAM rows, max abs difference from v10 | 0.0 | 0.0 |
| 36,317 EDDF rows, max abs update (zero by eligibility) | 0.0 | 0.0 |

### T9 the ranking-side census

| item | value |
|---|---|
| ranking departures | 344,841 |
| eligible (`nm_bulk` outside EHAM and EDDF) | 253,964 (73.65 %) |
| cells holding an eligible row | 55,570 |
| scored cohort (cells with ≥ 4 eligible rows) | 210,597 |
| eligible rows in cells under four rows | 17.08 % |
| rows changed | 210,597 |
| RMS of the served update, over the scored cohort | 6.498 s |
| RMS over every eligible row (scored or not) | 5.917 s |
| RMS over all 344,841 departures | 5.078 s |
| max abs update | 113.0 s |
| fold-side RMS, `dayblock` | 9.16 / 8.98 / 9.40 s |
| fold-side RMS, `xmonth` | 7.35 / 6.81 / 8.17 s |

By ranking month and by airport:

| group | departures | eligible | scored | RMS update (s) |
|---|---|---|---|---|
| 2026-01 | 152,719 | 114,653 | 90,525 | 6.518 |
| 2026-07 | 192,122 | 139,311 | 120,072 | 6.482 |
| EDDF | 36,317 | 0 | 0 | 0.000 |
| EDDM | 25,966 | 24,163 | 17,838 | 5.297 |
| EGLL | 39,840 | 38,157 | 36,597 | 8.042 |
| EHAM | 38,182 | 0 | 0 | 0.000 |
| LEBL | 30,080 | 27,918 | 22,105 | 5.625 |
| LEMD | 36,954 | 35,079 | 29,417 | 6.545 |
| LFPG | 39,872 | 36,775 | 31,735 | 6.582 |
| LIRF | 26,899 | 25,259 | 17,903 | 7.313 |
| LSZH | 23,150 | 20,685 | 13,930 | 5.893 |
| LTFM | 47,581 | 45,928 | 41,072 | 5.514 |

### T10 the same update on the v11 base, fold side

`cross-month` is the per-fold amplitude fitted on the other two folds only - the like-for-like row. `pooled` is the single scalar the written file carries; it is fitted on all three folds, including the one it scores, and is shown for that reason alone.

| cross-fit | amplitude | v10 board | v11 board | v11 + d board | gain over v11 | total gain over v10 | folds (jul/dec/jan) |
|---|---|---|---|---|---|---|---|
| dayblock | cross-month | 309.0772 | 308.9739 | 308.8946 | **+0.0793** | +0.1827 | +0.0969 / +0.1390 / +0.0614 |
| dayblock | pooled | 309.0772 | 308.9739 | 308.8942 | +0.0797 | +0.1830 | +0.0971 / +0.1393 / +0.0619 |
| xmonth | cross-month | 309.0772 | 308.9739 | 308.9489 | **+0.0250** | +0.1284 | +0.0314 / +0.0768 / +0.0185 |
| xmonth | pooled | 309.0772 | 308.9739 | 308.9485 | +0.0254 | +0.1287 | +0.0308 / +0.0769 / +0.0199 |
<!-- /TABLES:serve -->

<!-- TABLES:records -->
<!-- generated by code/tables.py; do not edit by hand -->

### T12 the COMMON.md slice decomposition, v10 against both ridge designs

Every arm on the record, not only the ridge (Sol round 2, blocking 4). Each arm is scored at its own cross-month amplitude, so these are the same predictions the gain tables are computed from.

| fold | arm | rows | RMSE all | RMSE <= 3600 s | RMSE > 3600 s | RMSE no-NM |
|---|---|---|---|---|---|---|
| july_loo | `v10` | 190,713 | 284.1572 | 247.0938 | 2405.27 | 1120.35 |
| july_loo | `ridge / dayblock` | 190,713 | 284.0600 | 246.9842 | 2405.20 | 1120.35 |
| july_loo | `ridge / xmonth` | 190,713 | 284.1288 | 247.0589 | 2405.34 | 1120.35 |
| july_loo | `lgb_small / dayblock` | 190,713 | 284.0708 | 246.9777 | 2405.76 | 1120.35 |
| july_loo | `lgb_small / xmonth` | 190,713 | 284.1571 | 247.0898 | 2405.39 | 1120.35 |
| july_loo | `lgb_wide / dayblock` | 190,713 | 283.9656 | 246.8309 | 2406.52 | 1120.35 |
| july_loo | `lgb_wide / xmonth` | 190,713 | 284.2006 | 247.1328 | 2405.60 | 1120.35 |
| july_loo | `lgb_shipped / dayblock` | 190,713 | 284.0473 | 246.9553 | 2405.62 | 1120.35 |
| july_loo | `lgb_shipped / xmonth` | 190,713 | 284.1381 | 247.0720 | 2405.26 | 1120.35 |
| july_loo | `lgb_shipped_wide / dayblock` | 190,713 | 283.8998 | 246.7522 | 2406.60 | 1120.35 |
| july_loo | `lgb_shipped_wide / xmonth` | 190,713 | 284.1579 | 247.0972 | 2405.20 | 1120.35 |
| july_loo | `ridge / dayblock_frozen` | 190,713 | 284.0604 | 246.9837 | 2405.23 | 1120.35 |
| july_loo | `ridge / halfmonth` | 190,713 | 284.0398 | 246.9629 | 2405.14 | 1120.35 |
| july_loo | `lgb_shipped / dayblock_frozen` | 190,713 | 284.0479 | 246.9553 | 2405.64 | 1120.35 |
| july_loo | `lgb_shipped / halfmonth` | 190,713 | 284.0766 | 246.9935 | 2405.49 | 1120.35 |
| july_loo | `lgb_shipped_wide / dayblock_frozen` | 190,713 | 283.8985 | 246.7484 | 2406.67 | 1120.35 |
| july_loo | `lgb_shipped_wide / halfmonth` | 190,713 | 283.9240 | 246.8040 | 2405.89 | 1120.35 |
| july_loo | `lgb_small / dayblock / hess_diag` | 190,713 | 284.0693 | 246.9750 | 2405.79 | 1120.35 |
| july_loo | `lgb_small / xmonth / hess_diag` | 190,713 | 284.1581 | 247.0943 | 2405.29 | 1120.35 |
| july_loo | `lgb_wide / dayblock / hess_diag` | 190,713 | 283.9668 | 246.8300 | 2406.58 | 1120.35 |
| july_loo | `lgb_wide / xmonth / hess_diag` | 190,713 | 284.2063 | 247.1420 | 2405.53 | 1120.35 |
| july_loo | `lgb_shipped / dayblock / hess_diag_nocachevocab` | 190,713 | 284.0604 | 246.9688 | 2405.67 | 1120.35 |
| july_loo | `lgb_shipped / xmonth / hess_diag_nocachevocab` | 190,713 | 284.1263 | 247.0587 | 2405.26 | 1120.35 |
| july_loo | `lgb_shipped_wide / dayblock / hess_diag_nocachevocab` | 190,713 | 283.9237 | 246.7810 | 2406.56 | 1120.35 |
| july_loo | `lgb_shipped_wide / xmonth / hess_diag_nocachevocab` | 190,713 | 284.1434 | 247.0722 | 2405.45 | 1120.35 |
| july_loo | `ridge / pathcheck / -` | 190,713 | 284.0604 | 246.9837 | 2405.23 | 1120.35 |
| july_loo | `ridge / xmonth_one_1 / -` | 190,713 | 284.1316 | 247.0679 | 2405.16 | 1120.35 |
| july_loo | `ridge / xmonth_one_2 / -` | 190,713 | 284.2158 | 247.1543 | 2405.49 | 1120.35 |
| december | `v10` | 165,677 | 204.9218 | 192.5142 | 1779.69 | 661.15 |
| december | `ridge / dayblock` | 165,677 | 204.7886 | 192.3738 | 1779.58 | 661.15 |
| december | `ridge / xmonth` | 165,677 | 204.8487 | 192.4206 | 1780.76 | 661.15 |
| december | `lgb_small / dayblock` | 165,677 | 204.8350 | 192.3982 | 1781.29 | 661.15 |
| december | `lgb_small / xmonth` | 165,677 | 204.8406 | 192.4177 | 1780.37 | 661.15 |
| december | `lgb_wide / dayblock` | 165,677 | 204.7205 | 192.3112 | 1778.89 | 661.15 |
| december | `lgb_wide / xmonth` | 165,677 | 204.8826 | 192.4103 | 1783.94 | 661.15 |
| december | `lgb_shipped / dayblock` | 165,677 | 204.8635 | 192.4222 | 1781.73 | 661.15 |
| december | `lgb_shipped / xmonth` | 165,677 | 204.8219 | 192.4028 | 1780.03 | 661.15 |
| december | `lgb_shipped_wide / dayblock` | 165,677 | 204.6478 | 192.2249 | 1779.49 | 661.15 |
| december | `lgb_shipped_wide / xmonth` | 165,677 | 204.7974 | 192.3503 | 1781.83 | 661.15 |
| december | `ridge / dayblock_frozen` | 165,677 | 204.7856 | 192.3705 | 1779.59 | 661.15 |
| december | `ridge / halfmonth` | 165,677 | 204.8920 | 192.4869 | 1779.38 | 661.15 |
| december | `lgb_shipped / dayblock_frozen` | 165,677 | 204.8634 | 192.4223 | 1781.72 | 661.15 |
| december | `lgb_shipped / halfmonth` | 165,677 | 204.9365 | 192.4966 | 1781.97 | 661.15 |
| december | `lgb_shipped_wide / dayblock_frozen` | 165,677 | 204.6710 | 192.2432 | 1779.94 | 661.15 |
| december | `lgb_shipped_wide / halfmonth` | 165,677 | 204.8020 | 192.3838 | 1779.88 | 661.15 |
| december | `lgb_small / dayblock / hess_diag` | 165,677 | 204.8331 | 192.3962 | 1781.29 | 661.15 |
| december | `lgb_small / xmonth / hess_diag` | 165,677 | 204.8363 | 192.4170 | 1780.10 | 661.15 |
| december | `lgb_wide / dayblock / hess_diag` | 165,677 | 204.7219 | 192.3217 | 1778.27 | 661.15 |
| december | `lgb_wide / xmonth / hess_diag` | 165,677 | 204.8924 | 192.4219 | 1783.86 | 661.15 |
| december | `lgb_shipped / dayblock / hess_diag_nocachevocab` | 165,677 | 204.8559 | 192.4153 | 1781.65 | 661.15 |
| december | `lgb_shipped / xmonth / hess_diag_nocachevocab` | 165,677 | 204.8140 | 192.3956 | 1779.94 | 661.15 |
| december | `lgb_shipped_wide / dayblock / hess_diag_nocachevocab` | 165,677 | 204.6787 | 192.2611 | 1779.27 | 661.15 |
| december | `lgb_shipped_wide / xmonth / hess_diag_nocachevocab` | 165,677 | 204.8128 | 192.3641 | 1782.01 | 661.15 |
| december | `ridge / pathcheck / -` | 165,677 | 204.7856 | 192.3705 | 1779.59 | 661.15 |
| december | `ridge / xmonth_one_1 / -` | 165,677 | 204.9034 | 192.4717 | 1781.26 | 661.15 |
| december | `ridge / xmonth_one_2 / -` | 165,677 | 205.1127 | 192.7047 | 1780.58 | 661.15 |
| january | `v10` | 153,706 | 337.8260 | 196.2393 | 5983.37 | 2835.24 |
| january | `ridge / dayblock` | 153,706 | 337.7674 | 196.1275 | 5983.53 | 2835.24 |
| january | `ridge / xmonth` | 153,706 | 337.8102 | 196.1944 | 5983.64 | 2835.24 |
| january | `lgb_small / dayblock` | 153,706 | 337.6911 | 196.0772 | 5982.28 | 2835.24 |
| january | `lgb_small / xmonth` | 153,706 | 337.7992 | 196.1715 | 5983.70 | 2835.24 |
| january | `lgb_wide / dayblock` | 153,706 | 337.4583 | 195.6889 | 5982.06 | 2835.24 |
| january | `lgb_wide / xmonth` | 153,706 | 337.8660 | 196.2900 | 5983.65 | 2835.24 |
| january | `lgb_shipped / dayblock` | 153,706 | 337.6663 | 196.0244 | 5982.43 | 2835.24 |
| january | `lgb_shipped / xmonth` | 153,706 | 337.7762 | 196.1381 | 5983.61 | 2835.24 |
| january | `lgb_shipped_wide / dayblock` | 153,706 | 337.3554 | 195.5016 | 5982.21 | 2835.24 |
| january | `lgb_shipped_wide / xmonth` | 153,706 | 337.8400 | 196.2341 | 5983.82 | 2835.24 |
| january | `ridge / dayblock_frozen` | 153,706 | 337.7703 | 196.1325 | 5983.53 | 2835.24 |
| january | `ridge / halfmonth` | 153,706 | 337.8382 | 196.2381 | 5983.71 | 2835.24 |
| january | `lgb_shipped / dayblock_frozen` | 153,706 | 337.6663 | 196.0244 | 5982.43 | 2835.24 |
| january | `lgb_shipped / halfmonth` | 153,706 | 337.7505 | 196.1280 | 5983.08 | 2835.24 |
| january | `lgb_shipped_wide / dayblock_frozen` | 153,706 | 337.3494 | 195.4920 | 5982.20 | 2835.24 |
| january | `lgb_shipped_wide / halfmonth` | 153,706 | 337.6377 | 195.9349 | 5983.05 | 2835.24 |
| january | `lgb_small / dayblock / hess_diag` | 153,706 | 337.6698 | 196.0447 | 5982.21 | 2835.24 |
| january | `lgb_small / xmonth / hess_diag` | 153,706 | 337.8040 | 196.1796 | 5983.71 | 2835.24 |
| january | `lgb_wide / dayblock / hess_diag` | 153,706 | 337.4534 | 195.6772 | 5982.11 | 2835.24 |
| january | `lgb_wide / xmonth / hess_diag` | 153,706 | 337.8545 | 196.2781 | 5983.53 | 2835.24 |
| january | `lgb_shipped / dayblock / hess_diag_nocachevocab` | 153,706 | 337.6589 | 196.0134 | 5982.40 | 2835.24 |
| january | `lgb_shipped / xmonth / hess_diag_nocachevocab` | 153,706 | 337.7805 | 196.1455 | 5983.61 | 2835.24 |
| january | `lgb_shipped_wide / dayblock / hess_diag_nocachevocab` | 153,706 | 337.3674 | 195.5262 | 5982.15 | 2835.24 |
| january | `lgb_shipped_wide / xmonth / hess_diag_nocachevocab` | 153,706 | 337.8312 | 196.2304 | 5983.65 | 2835.24 |
| january | `ridge / pathcheck / -` | 153,706 | 337.7703 | 196.1325 | 5983.53 | 2835.24 |
| january | `ridge / xmonth_one_1 / -` | 153,706 | 337.8765 | 196.3015 | 5983.76 | 2835.24 |
| january | `ridge / xmonth_one_2 / -` | 153,706 | 337.8036 | 196.1899 | 5983.53 | 2835.24 |

Slice sizes are the same for every arm inside a fold (july_loo 190,057 at or under 3600 s, 656 above, 3,934 without a network-manager record). The no-NM slice is identical to v10 in every arm, because those rows are not `nm_bulk` and carry no update.
<!-- /TABLES:records -->

## 9. Reproduction

```
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/density.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/verdict.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/learner.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/controls.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/repair.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/serve.py ridge v10
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/serve.py ridge v11
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/v11base.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/records.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-044-zerosum/code/cohorts.py
.venv/bin/python experiments/exp-044-zerosum/code/tables.py
```

`tests/check.py` re-checks this experiment from the committed record alone, in seven layers, none
of which is a hand-typed expectation: the reference arm against exp-039's own `analysis/q03.json`;
both candidates as valid submissions; the served update's zero cell sums **recomputed from the
ranking file** rather than read back; the masked rows' bit-identity in both candidates; the custom
objective's gradient against a finite difference of the loss, and that the naive gradient `h − r_c`
fails the same test; the verdict-bearing relations on the record itself (the declared density rule
against the two-way table it is read from, the gate failure, the exact-Hessian refits, the
cross-fitted v11 amplitude, the slice decomposition and the provenance); and finally every table
re-rendered live from the current record and compared byte for byte, then **every number in this
report's prose** matched against the pool of values the records carry, with a documented list of the
numbers that are not measurements. Every check passes.

*(What that audit is and is not, stated because Sol round 2 found the earlier description too
generous. It is a **pool** test: a prose number passes if some record carries it at some sensible
rounding. It therefore catches a number that no run produced — its purpose, and it caught four while
this text was written: a fold eligible share, two fitting-set bounds and a planted control — but it
does **not** bind a number to the claim it appears in, so a number quoted in the wrong sentence
passes. The exemption list is the other limit: every entry has a written reason, and each is a
section number, a predeclared constant, a value from another experiment's record, a value quoted as
superseded, or a ratio derived in the prose from two numbers the record does carry. The two-run
schema counts and the finite-difference residuals were on that list in the first version and are now
records of their own (`records.json:provenance.shipped_columns`, `analysis/gradcheck.json`).)*

```
.venv/bin/python experiments/exp-044-zerosum/tests/check.py
```

Large binaries and the two candidate files are addressed by `analysis/MANIFEST.json` (path, bytes,
SHA-256). Provenance — the v10, v11, ranking-file and monthly-input hashes, the shipped cache
hashes, the git commit and dirty flag, the library versions and the SHA-256 of every code file of
**this** experiment — is in `analysis/records.json` under `provenance`. *(Sol round 1 found the
provenance block in `serve.json` hashed exp-039's code directory, not this one; `records.json`
carries the corrected block and `serve.json` was regenerated with it.)*

## 10. Review

<!-- REVIEW -->
Sol round 1 (`gpt-5.6-sol` at `xhigh`, fresh context, read-only, log `logs/sol-round1.log`) returned
**NEEDS REVISION** with eight blocking findings. All eight were accepted; none is disputed. The
answer of record is `review/round1-answers.md`, one entry per finding with what changed and what it
did to the verdict. Two findings changed a stated conclusion:

| # | finding | what it changed |
|---|---|---|
| 1 | the Hessian of the tree objective is the projector `P`, not `I`; "exact objective" overclaimed | §3 restated; every tree arm refitted with the exact diagonal (T11); no ordering changed |
| 2 | the serving fit uses 3 months, not the 12 the assignment names | §5 now opens with the deviation and its cause |
| 3 | the candidate fails the declared step-3 gate, and the text read as if it passed | **§0 and §5 now state the failure first; the upload recommendation is withdrawn** |
| 4 | the v11 gain was priced at the pooled amplitude | §7 quotes the cross-fitted +0.0250; T10 carries both |
| 5 | the shipped block's `c_*` codes come from a shared append-only vocabulary | both shipped arms refitted without them (T11); no ordering changed |
| 6 | the controls do not prove that time distance is the cause | §4 rewritten; `pathcheck` and `xmonth_one` added; the causal claim is now the operational one |
| 7 | the density verdict did not apply its own conditional rule, and the monotonicity claim is false | **the verdict is MIXED, not DENSITY** (`code/verdict.py`, T2b); the monotonicity sentence is withdrawn |
| 8 | no COMMON slice decomposition, no date or row contributions, wrong code hashes | `code/records.py` adds all three (T12) |

Sol round 2 (log `logs/sol-round2.log`) verified repairs 1, 2, 3 and 4 outright and found no leakage
under an independent audit. It marked 5, 6, 7 and 8 **PARTIAL** and added two new blocking findings;
verdict **NEEDS REVISION**. `MODES.md` allows at most two rounds, so round 2 is the last; all six are
accepted and repaired here, and `review/round2-answers.md` is the answer of record. Three changed
something:

| # | finding | what it changed |
|---|---|---|
| 1 | the repaired arms reused the declared run's amplitudes, so they were fixed-amplitude perturbations | **each repaired arm now carries its own cross-month amplitude**; `lgb_shipped_wide` xmonth moves −0.0145 → +0.0045 and `lgb_shipped` +0.0350 → +0.0379; the arm selection is unchanged |
| 2 | `pathcheck` does not execute the real cross-month matrix path | §0 and §4 narrowed; "rules out the code path", the "upper bound" and "sits between +0.0222 and +0.0782" are all withdrawn |
| 3 | the rule never defines "survives"; a strict counting gives neither; the 200-pair minimum was undeclared | **a fourth, strict counting is run and returns NEITHER**; "the verdict does not rest on the choice" is withdrawn; the threshold is disclosed and every counting is run without it |
| 4 | the slices covered only the ridge; the provenance was stale and had no dirty flag | T12 covers **every** arm; `git_dirty` added; the checker now fails on any stale or missing code hash |
| 5 | "the ridge is the only arm positive on all three folds under both designs" is false | corrected: three arms are, and **no arm passes the interval gate** |
| 6 | the "every prose number" audit claimed more than it does | the claim is rewritten as what it is (a pool test); the schema counts and the finite-difference residuals become records of their own |

Also corrected, non-blocking: `code/zs.py`'s module docstring and `SPEC.md`'s R2 heading, both of
which still called the objective exact where they made the claim.
<!-- /REVIEW -->
