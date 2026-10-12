# The structural step the board asks for — where it can live, and six candidates ranked

The exp-027 analysis used **no competition data on disk**. Every number below is produced from files inside git by
`experiments/exp-027-step/code/`; the historical exp-027 results record gives
the command that makes each number, and the exp-027 specification describes
the top candidate. No fit used the ranking file. I
posted nothing anywhere.

---

## 0. The answer in one paragraph

The head is not badly fitted and it is not badly calibrated. A per-airport level is
worth **−0.09 s**, a per-airport-and-hour-of-day level **+0.10 s**, an affine
recalibration **−0.04 s**, and a level per cell of its two strongest inputs crossed
with the airport **−0.07 s** — all cross-fitted, all indistinguishable from zero.
What *is* left is a component of the error that **departures at the same airport
minutes apart share**: correlation **+0.175** at a 2–5 minute take-off gap, decaying
to +0.02 beyond an hour, and an oracle on the airport × 10-minute cell mean is worth
**14.6 s of clean board on the head we ship** (16.0 s with the noac rows set aside;
singleton-corrected). **Every one of blend-5's seven members carries that component
at the same size**, including `V3-allin`, the only member built on the surface +
stand + weather frame, and **stack-5 absorbs 0.3 of its 12.1 percentage points** on
january. It is the largest identified structural residual in the lineage and nothing
in the record has touched it. A nested probe — a 400-round LightGBM on the head's
residual, fed nothing but window aggregates of `d_takeoff_aobt` and `d_aobt_lobt` —
takes **+1.112 s of clean board and +1.249 s with the noac rows set aside, improving
on all three folds**, while the same fit with the window aggregates removed *loses*
0.98 s and a leave-one-out window holds at +1.073 s.

**But the honest size of the step is 1 to 2 s of fold gain — which BOARD READING 6
prices at 1.7 to 3.4 s of board — and not 6 to 14 s. This document says so rather
than selling the ceiling.** The transfer ratio is measured on this very mechanism:
v6 (stack-5 with the 600 s window block inside its own stacker, `noac` held) scored
**263.6922**, **−0.694 s** on v3 against a **+0.414 s** fold gain, so the window
family transfers at about **1.7× face** where the `noac` lever transfers at a
quarter and can invert. Two measurements set that bound. Delivered as
a residual *layer* on the shipping head the full family gives +0.719 s with **january
−0.176**, failing the both-folds gate; delivered as features *inside* the model the
window analysis takes **+0.402 s with january +0.317**, so the delivery was the whole
difference. And a cell-level regression of the window mean residual on 131 observable
columns — **the arrivals' unmasked taxi-in and block times, the runway split and the
configuration change included** — recovers only **3 to 4 %** of the component. The
mechanism is real, large, and survives the stacker; its proxy in this file is weak.
That gap is the strongest argument in this record for S8, the open surface data,
because the state that drives the component is not in the file we were given.

---

## 1. The currency, and what a 6-to-14 s step costs

`board_clean² = 0.5571 · MSE_clean(july_loo) + 0.4429 · MSE_clean(january)`, the
ranking months' row shares. December is a robustness read and never enters it.

> **Currency update, 2026-09-07 (BOARD READING 3).** stack-5 scored **264.3861**,
> rank 1 of 63. Its noac-discounted prediction (264.59) was right within 0.2 s and
> its face-value one (262.0) missed by 2.35 — the same pattern v2 showed. The rule
> from here is **clean-with-noac-at-the-anchor + 4.28**. This changes the ranking
> below in one direction and it is applied in place: **a candidate's board value is
> the part of its fold gain that does NOT sit on the `noac` rows.**
>
> S1 is the candidate this helps. Its gain is measured at **0 % on `noac`** at every
> configuration tried, so it transfers at face value; the ceiling to quote for it is
> the no-`noac` one, **16.04 s**, which is what §4 and §5 already use.
> S2 is the candidate this demotes: its whole population is the `noac` rows, and the
> board has now twice said fold gains there do not transfer.

The head is **blend-5**, rebuilt here from committed member files with no data:
july_loo 286.3274, december 207.0083, january 339.0173, all-rows board 310.7662 —
byte-identical to `blend-config-5.json`'s own `rotated` and `board_estimate` — and
**clean board 259.1339** against the record's 259.132.

| step on the clean board | weighted MSE it must remove | share of the budget | …with `noac` set aside (board 231.2698) |
|---|---|---|---|
| 2 s | 1,033 s² | 1.5 % | 1.7 % |
| 5 s | 2,566 s² | 3.8 % | 4.3 % |
| **10 s** | **5,083 s²** | **7.6 %** | **8.5 %** |
| 14 s | 7,060 s² | 10.5 % | 11.7 % |

**The rivals' steps are the same size.** quick-boat's 273.21 → 267.12 removed
3,291 s², **4.4 %** of the budget it started from. upstanding-firefly's
280.7 → 267.008 removed 7,499 s², **9.5 %** of its own. So the question is not
"what is worth 10 s" in the abstract — it is "which tenth of our budget is
reachable", and the tables below answer exactly that.

---

## 2. The error budget of the head

`d_perfect` is what an oracle on the cell moves on the clean board — the bound a
candidate must fit inside. `d_debias` is what a *constant* per cell moves, i.e. how
much of the cell is a level rather than a signal. The historical exp-027 error-budget tables give the full cell breakdown.

### 2a. By regime

| regime | board rows | rmse july / jan | bias july / jan | **share** | d_perfect | d_debias |
|---|---|---|---|---|---|---|
| `nm_bulk` | 319,961 | 209 / 189 | +1.8 / +9.4 | **55.41 %** | 86.09 | 0.08 |
| `noac` | **402** | 3,407 / 3,509 | +666 / +297 | **20.45 %** | 28.01 | 0.88 |
| `nm_late` | 19,092 | 464 / 439 | +1.3 / +26.2 | 17.40 % | 23.62 | 0.02 |
| `nonm` | 4,962 | 535 / 622 | −22.2 / +12.5 | 6.74 % | 8.89 | 0.01 |

With the `noac` rows removed the board is **231.2698** and the three survivors are
69.65 %, 21.88 % and 8.48 %. **The 402 noac rows are the entire 27.9 s between the
two boards**. This sets the error budget: every
ceiling below is given in both currencies.

`d_debias` is under 0.1 s on every regime except `noac`. **There is no level
correction anywhere in this model.**

### 2b. By regime × label band — where the mass is

The label is not an input, so this table diagnoses and does not prescribe. Top ten
of 23 cells by board mass:

| regime | y band | board rows | bias july / jan | share | d_perfect |
|---|---|---|---|---|---|
| `nm_bulk` | 600–1200 | 204,873 | +35 / +34 | 19.89 % | 27.19 |
| `nm_bulk` | 1800–3600 | 11,374 | −417 / −283 | 13.51 % | 18.14 |
| `nm_bulk` | 1200–1800 | 62,640 | −94 / −70 | 13.14 % | 17.63 |
| `noac` | > 14400 | **41** | −2306 / −3196 | **6.85 %** | **9.03** |
| `nm_late` | 1800–3600 | 3,075 | −383 / −242 | 6.57 % | 8.66 |
| `noac` | 600–1200 | **137** | **+2247** / +1645 | **6.12 %** | **8.06** |
| `nm_late` | 3600–14400 | 524 | −1088 / −986 | 5.37 % | 7.06 |
| `nm_bulk` | < 600 | 40,823 | +109 / +94 | 5.07 % | 6.66 |
| `nonm` | 3600–14400 | **56** | −4210 / −3123 | 4.59 % | 6.01 |
| `nm_bulk` | 3600–14400 | 250 | −2035 / −1049 | 3.78 % | 4.95 |

The bias column runs +109, +35, −94, −417, −2035 across the bands in order. **That
is a conditional mean, not a defect**: any unbiased predictor over-shoots short
labels and under-shoots long ones, and the calibration slope of `y_true` on
`y_pred` is 1.006 (july) and 0.993 (january) on the bulk. Reading this table as
"the head under-predicts long taxi times" is the trap; the two rows that are *not*
regression to the mean are the `noac` ones, where the head over-predicts a 900 s
truth by 2,247 s and under-predicts a 20,000 s truth by 3,196 s in the same
population. That is a hedge over a two-point mixture, and §4 candidate S2 owns it.

### 2c. By regime × input-readable band — where a model can act

Bands of `d_takeoff_aobt = MVT − AOBT_3`, which every row carries.

| regime | `MVT − AOBT` band | board rows | share | d_perfect | d_debias |
|---|---|---|---|---|---|
| `nm_bulk` | 600–1200 s | 212,398 | 31.19 % | 44.18 | 0.03 |
| `noac` | (no NM join) | 402 | 20.45 % | 28.01 | 0.88 |
| `nm_bulk` | 1200–1800 s | 75,840 | 16.36 % | 22.15 | 0.11 |
| `nm_late` | 600–1200 s | 12,316 | 10.67 % | 14.21 | 0.03 |
| `nonm` | (no NM join) | 4,962 | 6.74 % | 8.89 | 0.01 |
| `nm_bulk` | 1800–3600 s | 9,153 | 4.27 % | 5.59 | 0.02 |
| `nm_late` | 1200–1800 s | 3,975 | 3.72 % | 4.87 | 0.01 |
| `nm_bulk` | 0–600 s | 22,057 | 2.73 % | 3.56 | 0.00 |

`d_debias` is **at most 0.11 s** in every input-readable cell. A 10 s step cannot be
a recalibration of any cell this model can name.

### 2d. By airport

| airport | share (all) | d_perfect | share (no noac) | d_perfect |
|---|---|---|---|---|
| **LIRF** | **35.99 %** | 51.80 | 19.82 % | 24.19 |
| LTFM | 12.64 % | 16.93 | 15.88 % | 19.15 |
| EGLL | 11.36 % | 15.16 | 14.28 % | 17.14 |
| LFPG | 10.70 % | 14.26 | 13.43 % | 16.09 |
| LEBL | 6.05 % | 7.96 | 7.61 % | 8.97 |
| EHAM | 5.75 % | 7.56 | 7.23 % | 8.52 |
| EDDF | 5.24 % | 6.89 | 6.59 % | 7.75 |
| LEMD | 4.58 % | 6.00 | 5.75 % | 6.75 |
| LSZH | 4.03 % | 5.27 | 4.79 % | 5.61 |
| EDDM | 3.67 % | 4.80 | 4.61 % | 5.40 |

397 of the 402 noac rows are at LIRF and they alone are 20.22 % of the whole
budget. **Rome is not a separate problem once they are removed** — it is a large
airport whose `nm_bulk` residual `BLOCK − AOBT` has sd 498 s against 348 s at the
other nine (july). exp-006 measured a LIRF-only model and did not adopt it; nothing
here reopens that.

### 2e. Concentration, and the tail

| | july_loo | january |
|---|---|---|
| top 1,000 rows (0.5 / 0.7 %) | **48.1 %** of fold SSE | 36.5 % |
| top 5,000 (2.6 / 3.3 %) | 64.5 % | 54.9 % |
| rows with y > 3600 s | 0.34 % of rows, **24.9 %** of SSE | 0.21 %, 19.2 % |
| rows with y ≤ 3600 s | 99.66 %, 75.1 %, rmse 249 | 99.79 %, 80.8 %, rmse 198 |

Board-weighted, **76.9 % of the budget is on ordinary rows** and 23.1 % on the 979
rows above 3600 s. The first sprint's naive predictor put 67 % of its error on the
tail; the head has already taken most of that back, and **the bulk is now the
game**. Inside the bulk the mass is still concentrated — the worst 1 % of `nm_bulk`
rows with y ≤ 3600 hold 26.0 % of that population's squared error — but that is the
shape of a heavy conditional distribution, not evidence of headroom on its own.

### 2f. Oracle ceilings on the named populations

| oracle | rows | gain | no-noac gain |
|---|---|---|---|
| perfect on every y ≤ 3600 row | 343,438 | 134.64 | 134.72 |
| perfect on the **tail-crossing** rows (head and truth on opposite sides of 3600) | **702** | **33.78** | 19.90 |
| perfect on every y > 3600 row | 979 | 31.86 | 21.12 |
| perfect on the 402 `noac` rows | 402 | 28.01 | — |
| serve `MVT − AOBT` on every row whose block really is within 60 s of AOBT | 65,663 | 6.02 | 6.78 |

---

## 3. What is not there, and one floor — nine cheap kills

Every line is cross-fitted (the cell intercept is estimated on a random half of the
fold and scored on the other half, five seeds), so none of it is in-sample.

| idea | measured | verdict |
|---|---|---|
| a per-airport level | **−0.09 s** | dead |
| per airport × `d_takeoff_aobt`(10) × `d_aobt_lobt`(10) | **−0.07 s** | dead — the head is already unbiased on its two strongest inputs |
| per airport × hour-of-day | +0.10 s | dead |
| per airport × `d_takeoff_aobt` (20 bins) | +0.43 s | noise floor |
| per airport × regime | +0.47 s | noise floor |
| per airport × hour-of-day × weekday | +0.83 s | small, and probably the same object as §5 S1 |
| an affine recalibration of the head, fitted on the other fold | **−0.04 s** | dead |
| `MVT_ID_mvt` as a hidden index of the block event — if movement ids were issued at the block event, the unmasked arrivals would bracket every departure's block time | P(id order = block order) is **below** P(take-off order = block order) at **all ten** airports (EDDF .599 vs .634, EHAM .500 vs .677, LFPG .473 vs .663) | dead |
| a minute lattice on `BLOCK_TIME_UTC_mvt` — if block times were minute-rounded, the take-off's seconds would fix the label mod 60 | 6.4 % of block times land on an exact minute against 1.7 % by chance | dead |

**And one floor.** `AOBT_3_flt` lands on an exact minute on **98.4 %** (july) and
**96.6 %** (january) of rows, against 5.1 % for `MVT_TIME_UTC_mvt` and 6.4 % for
`BLOCK_TIME_UTC_mvt`. The anchor every NM member subtracts is minute-quantised,
which puts about 300 s² of independent variance into it on 93.9 % of board rows:
**about 0.5 s of clean board that no model on this file recovers.** Independent
confirmation of `docs/external-sweep-2026-09-05.md` S3 (M3 is a *model*), and the
reason exp-019's "is the taxi estimate a lookup table" test could not have passed.

---

## 4. What is there: the transient airport window

### The head's error is shared between departures minutes apart

Correlation of the head's error between two departures at the same airport, by the
gap between their take-offs (`nm_bulk`, errors winsorised to ±1200 s so 0.3 % of
rows do not set the answer):

| take-off gap | july_loo | january |
|---|---|---|
| < 2 min | +0.170 | +0.146 |
| **2–5 min** | **+0.175** | **+0.140** |
| 5–15 min | +0.134 | +0.107 |
| 15–30 min | +0.079 | +0.064 |
| 30–60 min | +0.044 | +0.042 |
| 1–3 h | +0.022 | +0.019 |
| 3–24 h | +0.017 | +0.013 |

### No member has it

Same statistic per member, july_loo `nm_bulk`, pairs under 600 s apart:

| predictor | rmse | corr | shared variance |
|---|---|---|---|
| HEAD (blend-5) | 209.3 | +0.1605 | 7,031 |
| `F1-fresh` | 209.6 | +0.1575 | 6,919 |
| **`V3-allin`** (the v3 frame: surface + stand + weather + deice + exot) | 215.2 | **+0.1624** | **7,521** |
| `C2-cat` | 223.2 | +0.1668 | 8,309 |
| `L2-hub` | 225.4 | +0.1513 | 7,687 |
| `D1-winter` | 226.5 | +0.1707 | 8,757 |
| `T-tailv2` | 223.6 | +0.1735 | 8,676 |

The member carrying every congestion, stand and weather family in the programme is
**not** better on this component than the members carrying none of them.

### The ceiling

Noise-adjusted (`Σ n·m² − s²`, unbiased, not clipped at zero):

Cells of one row are excluded: `var(ddof=1)` is undefined there, and counting it
as zero makes a lone row's whole error read as shared. Excluded rows stay in the
denominator. That direction understates a sparse population, but the figure is a **sensitivity, not an identified estimate**: the singleton cells' contribution is unidentified rather than zero, and the convex `board − sqrt(board² − mass)` conversion biases the seconds figure upward.

| cell | gain on board 259.134 | gain on the no-noac board 231.270 | rows in singleton cells |
|---|---|---|---|
| airport × 5 min | 15.00 | 16.08 | 8.4 % |
| **airport × 10 min** | **15.11** | **16.66** | 2.1 % |
| airport × 15 min | 14.49 | 15.61 | 0.9 % |
| airport × 30 min | 12.23 | 13.33 | 0.3 % |
| airport × 1 h | 8.96 | 10.42 | 0.1 % |

Winsorising the errors to ±1200 s moves the 10-minute figure from 16.66 to 16.96,
so this is the bulk's structure and not a handful of tail rows; at `minn = 4` it
reads 14.81, so the resolution that matters is stable. On **stack-5**, the head the
team serves (clean board 258.037), the same figures are **14.62 / 16.04** at ten
minutes: the stacker sees only member predictions, so it cannot see a window state
and it does not.

**These are ceilings and the gap to a real estimator is large.** §3's cross-fitted
airport × clock-hour intercept — the same cell mean, estimated on half a fold and
scored on the other half — is **+4.90 s** against the 8.96 s oracle at that
resolution.

### Part of it is already reachable — the probe

400 rounds, 31 leaves, LightGBM on the head's winsorised residual, fitted on two
folds and scored on the third, fed **only** window aggregates of `d_takeoff_aobt`
and `d_aobt_lobt` plus the regime mix and the movement count, at the departure's own
bucket and the two either side:

| arm | july_loo | december | january | clean board | **no-noac board** |
|---|---|---|---|---|---|
| window, 300 s | | | | +0.520 | **+1.097** |
| **window, 600 s** | **+0.696** | **+1.158** | **+0.641** | **+0.669** | **+1.073** |
| window, 900 s | | | | +0.557 | +1.014 |
| window, 1800 s | | | | +0.195 | +0.519 |
| window, 600 s, 900 rounds / 63 leaves | | | | +0.125 | +0.421 |
| **CONTROL: the same fit, window aggregates removed** | −2.459 | +0.535 | −0.064 | **−1.540** | **−1.279** |

Three folds of three improve on the window arm — the both-folds gate exp-022's
fitted arm and exp-017's recency arm failed — and the control loses. **The gain is
the window.**

**Every figure in the table above is superseded by the nested re-measurement**
(the exp-027 rotation analysis). The head's own rotation fits a fold's
weights on the other two, so a training fold's residual carries the scored fold's
labels. `code/k1_nested.py` rebuilds each training fold's head prediction from the
one fold that is neither the scored fold nor the fold being predicted:

| arm, 600 s, `noac` offloaded | clean board | no-noac board | folds up |
|---|---|---|---|
| as committed above (leaky) | +1.035 | +1.162 | 3 / 3 |
| **nested** | **+1.112** | **+1.249** | **3 / 3** |
| **nested, leave-one-out window** | **+1.073** | **+1.204** | **3 / 3** |
| nested row-only CONTROL | −0.981 | −1.101 | 1 / 3 |

**Window measurement pointer.** Every figure in this section is measured on
**blend-5's** residual and does not transfer to a residual stacker: the same arm on
**stack-5's** residual is **+0.002 s** of clean board and fails the three-fold gate,
while the same *fixed* blend-5 adjustment carried onto stack-5 without refitting is
still worth +0.261 s. The component itself is intact on stack-5 (oracle 14.62 s against
15.11 s at 600 s; sharing 0.168 against 0.170 under two minutes), so what fails is
estimating the map a second time on an already-corrected residual. The construction
that does work on stack-5 is the same window block **inside the stacker**, +0.407 s
clean board at 300 s passing all three folds. `experiments/exp-028-window/RESULTS.md`
§2–§4 and §7b.

The leak was making the arm look *worse*, not better. The leave-one-out row removes
the departure's own contribution from its own bucket and costs 0.04 s, so the signal
is the neighbours and not the row. One leak this does not close: a fold's labels
also reach the head through the **member models**, which were trained on month sets
that include it (the blend analysis) — a channel every blend and stack
number in this repository shares, including both uploaded candidates.

---

## 5. The candidates, ranked

Ranked by **ceiling × the evidence that any of it is reachable**, not by ceiling.
Rejected orderings are in the historical exp-027 specification.

### S1 — the transient airport window (mine; the record does not name it)

**Mechanism.** Departures pushing back at the same airport within the same ten
minutes share an error the head does not see: a queue building at a holding point,
a runway configuration turning over, a de-icing wave, an ATFM regulation biting.
The current features describe *this* departure's neighbourhood; the residual is a
property of the *cell*, and no column in `prc.surface_features` is a dispersion
statistic of a window or carries a lead and a lag.

**Cells it drains, and the ceiling.** All three non-`noac` regimes, not `nm_bulk`
alone: the measured gain lands `nm_bulk` 50 %, `nm_late` 33 %, `nonm` 17 %
(stack-5). On the shipping head, singleton-corrected, 10-minute cells, the ceiling
is **16.04 s** on the no-`noac` board, which is **14.17 s** expressed on the full
clean board (round-2 finding 2: the 14.62 s this file first quoted was the
*all-row* mass, `noac` included, which is inconsistent with a candidate that
offloads `noac`); `nm_bulk`
alone is 11.66 s of that, `nm_late` 3.32 s, `nonm` 0.78 s. At 30 minutes, 11.77 /
12.75 s. A 6-to-14 s step fits inside the ceiling, but see the next paragraph:
**the reachable fraction of it, not the ceiling, is what this candidate is worth.**

**For.** Measured four ways from committed OOF and two fits. The correlation table
(+0.175 at a 2–5 min gap, decaying to +0.022 past an hour); the per-member table
showing every member carries it at the same size, `V3-allin` on the full surface
frame included; a **nested** probe on two observable columns that takes **+1.112 s
clean / +1.249 s no-`noac` on three folds of three** while its control loses
0.981 s and a leave-one-out window holds at +1.073 / +1.204; and the component
surviving the stacker on every fold, january included (11.85 % of january's own
MSE, 12.67 s). The window model has since taken **+0.402 s of clean board with
january +0.317** by putting the columns inside stack-5's stacker.

**Against — and this is the finding that sizes the candidate.** As a residual
*layer* on the shipping head the full K2 family gives **+0.719 s with january
−0.176**, failing the both-folds gate (the historical exp-027 results record §12a). And at cell level, a
regression of the airport-window mean residual on 131 observable columns —
**including the arrivals' unmasked taxi-in and block times, the runway split and
the configuration change** — recovers only **0.4 to 0.5 % of the shared mass, 3 to
4 % at its best shrinkage**, with the arrivals and runway worth 1.3 points of that
(the historical exp-027 results record §12c). The component is large and real; its observable proxy in this
file is weak. Also: `prc.surface_features` VERSION 3 has 300–3600 s windows and a
30-minute arrival-taxi-in median already, and `V3-allin` serves them at weight
0.21–0.30, so part of the family is not new; and exp-004's −20.8 s july was
measured on a base scoring 588.5, of which very little survived to the head.

**Realistic value, stated against the ceiling.** 16.04 s is the ceiling; 3–4 % of
it is what observables reach at cell level; +0.402 s of fold is what the same
columns have actually taken on the shipping head, and that became **−0.694 s of
board** (BOARD READING 6, v6 at 263.6922), a **1.7× transfer** with `noac` held.
**Expect 1 to 2 s of fold, 1.7 to 3.4 s of board — not 6 to 14 s.** This candidate
is ranked first because it is the only one with a measured positive on the shipping
head *and* on the board, not because its ceiling is the largest.

**Why the leaders could have found it.** It is the first move a team makes when it
stops modelling the flight and starts modelling the airport, it costs no GPU, and a
residual arm on window state is a standard construction. quick-boat's 6.09 s step
and upstanding-firefly's 13.73 s are 4.4 % and 9.5 % of their budgets, which is
what this ceiling covers.

**Cheapest kill test.** K1 is run, nested and passed. K2 is the full family
delivered **inside the `F1-fresh` member frame**, never as a layer — the layer is
already measured and already fails (§12a). One fold first, july_loo, against the
committed member OOF 293.6499939.

- **FAIL** if the member's july_loo RMSE does not improve by ≥ **0.30 s**, or if
  the rebuilt member without the window columns does not reproduce 293.6499939 to
  within the environment's 0.68 s A/B band (`exp-025` acceptance rule: the fold
  RMSE, never a hash).
- **FAIL** at the head if the three-fold rotation gives **< +0.10 s** of clean
  board (the blend rotation half-width, 0.1025 s), or if **january moves the wrong
  way** — the gate the layer failed.
- **PASS** at ≥ **+0.40 s** of clean board with january positive, which is what the
  window model's in-stacker construction already reads, and a calendar-day-
  clustered 95 % interval clear of zero.
- **Kill the whole family** if the member-frame delivery also gives january a
  negative sign: that would mean the january component is not the same mechanism
  as july's, and no delivery repairs it.

**RUN, and it passes every bar above** (2026-09-07 12:20 UTC,
the historical exp-027 results record §16). The member: july_loo **+0.5256**
like for like on Kaggle against a 0.30 s bar, december +1.3012, january +1.4425
clean. The head, with the block inside the member frame and v7's serving rule:
folds **284.3195 / 205.2156 / 218.2582**, clean board **257.1653** — **+0.8719
against stack-5**, twice the +0.40 s PASS bar, and **+0.2945 against v7**, the
file the board holds, with all three folds positive and all six day-clustered
intervals clear of zero. The `y > 3600` tail improves by 2.65 / 4.81 / 1.71 s
and the 388 `noac` rows do not move at all. Predicted board **263.1678** at
face. So the kill test that would have revived S8's in-file proxy — "K2 lands
under +0.40 s" — is **not** triggered; item 2 below stays a proxy worth building
for its own reasons, not as a repair.

Full spec, code path, leakage contract and Kaggle session plan:
the historical exp-027 specification.

**Cost.** 3 session-hours plus **1.5 GPU-h for the first fold** and 4.5 GPU-h for
the remaining two plus the ranking months, of the 30 GPU-h weekly quota shared
with `cloud`. The frame build is the risk item at 1.5–2 h and is timed first.

---

### S2 — `noac` as its own population, at the 388-row scale

> **Demoted, then re-shaped, by BOARD READINGS 3, 4 and 5 (2026-09-07).**
> Reading 4 uploaded a noac instrument (blend-5 with exp-002's noac head, RMS
> 1,857 s on the 388 rows) and it scored **275.2197 against v2's 265.9631: +9.26 s
> WORSE from 388 rows**, about 2,100 s RMS of extra error per row. Reading 5
> bracketed it: a small change in the blend heads' direction (RMS 371 s) bought
> **+0.33 s** where the fold said 1.19, so **noac fold gains transfer at about a
> quarter of face**. Together: the 2026 noac rows do not behave like the 2025 rows
> every head here was fitted on, small moves in the blend's direction help a
> little, and a large move toward the fold-optimal head is catastrophic.
> **The fold cannot tell us the direction, only the size.** The 28.15 s oracle is a
> fold ceiling; at a quarter of face it is ~7 s of board, and the sign is unknown.
>
> **So this candidate's kill test cannot be a fold test, and it is rewritten below
> as a fold-free one.** score-feedback rule forbids probing the board further, so a noac head ships
> only inside a complete candidate and is read at the board once.
>
> The ceiling below is real and the
> mechanism is real, but the board has now said twice that a fold gain on these
> rows does not reach it: v2 (blend-5) missed its face-value prediction by 2.83 s
> of which 2.27 was the noac discount, and v3 (stack-5) missed by 2.35 s while its
> noac-discounted prediction was right within 0.2 s. **So 28.15 s is the ceiling on
> the fold, not on the board, and this candidate's board value is bounded by
> something much smaller that no instrument here has measured.** It stays ranked
> second on ceiling and would rank below S4 on expected board value. A candidate
> whose whole population is 388 ranking rows the board discounts is the wrong place
> to spend a session until we understand *why* the discount exists — which is
> itself the more valuable question, and is not this candidate.

**Mechanism.** 402 fold rows with no aircraft type — 397 of them at LIRF — carry
**20.45 %** of the budget. The head serves the expectation of a two-point mixture
(placeholder, where the recorded block time is the schedule, against a genuine
movement), so it over-predicts a 600–1200 s truth by **+2,247 s** (july) and under-predicts a
truth above 14,400 s by **−2,306 s** (july) and **−3,196 s** (january) in the same
population.

**Cells and ceiling.** `noac` alone: an oracle on those 402 rows moves the clean
board **28.01 s**, from 259.134 to 231.121, and removing them from the population
altogether gives 231.270 — the same 28 s by either convention. exp-023's own oracle-p on the 1–8 h
band is ≈ 8 s of board.

**For.** The largest ceiling per row anywhere in the data. The blend analysis
measures that 85 % (july) and 67 % (january) of blend-5's gain over blend-4
is on these rows.
**Against.** This is the only ceiling in the list that two experiments have already
attacked and failed to realise. exp-022's fitted arm gets **0.27 s of a 6.9 s
oracle** and fails the both-folds gate; its `detectability.log` says the ceiling is
structural — "a 3.9 % base rate against an AUC-0.86 ranking: the p-rule, not a
bug". exp-009's disambiguation "does not disambiguate": 18 of 22 rows decided, 4
hedged at the base rate. And a 402-row lever cannot be verified on the board — its
whole contribution sits inside the noise of one upload.

**Why the leaders could have found it.** Rome's placeholder rows are visible to
anyone who plots the label against `MVT − SCHED`; a team that split them out early
would have taken a large single step, which is the shape of upstanding-firefly's
74.6 → 56.3 → 13.7 s sequence.

**Cheapest kill test.** Test exp-023's two proposed repairs: fit `p` by the **scored** loss (ratio target
`(y − taxi_hat)/(ms − taxi_hat)` weighted `(ms − taxi_hat)²`) and isotonic
calibration within `ms` deciles per operator group.

**The direction clause is fold-free, because the folds have been measured wrong on
this population three times.** A 2025 fold cannot say whether a 2026 noac
prediction moves toward the truth: reading 4's head was fold-optimal and lost
9.26 s. So the test is:

- **FAIL, fold side (necessary, not sufficient).** noac LOMO RMSE must beat the
  delivered 4,234.0 s by more than **150 s**, and the fold contract must agree with
  LOMO as it did not in `exp-006 noac-head-folds`. A fold gain alone never promotes
  this candidate.
- **FAIL, direction side.** The candidate must move the 388 ranking rows by an RMS
  of **no more than 400 s** against the shipping head. Reading 5's +0.33 s came
  from a 371 s move and reading 4's −9.26 s from an 1,857 s move; nothing in the
  record supports a large move, and the fold cannot license one.
- **PASS** only as part of a **complete candidate**, read at the board **once**
  (score-feedback rule). No noac-only upload.
- **Value to quote:** the 28.15 s oracle is a fold ceiling; at the measured quarter
  transfer it is about **7 s of board**, with the sign unknown until a complete
  candidate reads it.

**Cost.** 1–2 session-hours, no GPU; the head trains on 1,507 rows.

---

### S3 — the flow-state neighbour set as a tail detector

**Mechanism.** 702 rows put the head and the truth on opposite sides of 3600 s —
341 tails it misses and 124 it invents, in july alone — and they carry **24.4 %** of
the budget. A screen that raises the detected probability of a tail high enough for
the mixture expectation to move would drain them.

**Cells and ceiling.** Perfect on the crossing rows is **33.8 s** (19.9 without
noac); perfect on every y > 3600 row is 31.9 s (21.1 no-noac).

**For.** `prc.surface_features`' neighbour flow state screens the January tail at
**30× lift** on its own against 8× for weather (`exp-002` §12), and exp-002's own
diagnosis names it as the more promising of the two and never ran it.
**Against.** The same table gives the ceiling: the best combined screen reaches
p = 0.082 on 256 rows, and exp-002's revival condition is p ≥ 0.15 on more than a
thousand rows. Under RMSE a screen at p = 0.022 with a 4,400 s gap moves the
prediction 97 s against a 4,400 s error, so almost nothing is realised.

**Why the leaders could have found it.** Tail detection is where the first 100 s of
anyone's score came from, and a team at 280 s that found a better screen would step
exactly the 13.7 s upstanding-firefly stepped.

**Cheapest kill test.** Build the neighbour flow-state screen from
`prc.surface_features` (not the airport-hour median exp-002 used), tabulate p and
recall on the january and july_loo folds. FAIL if no screen reaches p ≥ 0.15 on
more than 1,000 rows. PASS at p ≥ 0.15, at which point `prc/tail_model.py`'s mixture
machinery applies unchanged. **This is a screen, not a fit: it costs under an hour
and it kills or opens the whole candidate.**

**Cost.** 1 session-hour to screen; 3–4 more if it opens.

---

### S4 — the `v3f` frame, and the v3 families in a stronger learner

**Mechanism.** `experiments/exp-001-retro-v1/frame.py` defines
`SOURCES_V3F = SOURCES_V3 + ("fresh",)` — the served v3 frame plus exp-018's three
label-free families, 175 columns after dedup. It is coded and frozen and has **no
result file and no ledger row**. Separately, the v3 families reach the head only
through `V3-allin`'s 0.21–0.30 weight; putting them in the dominant member or in
the stacker's feature set changes what fraction of them survives.

**Cells and ceiling.** `nm_bulk`, the same 69.65 %. By the `V3-allin` precedent a
new frame lineage is worth **−1.57 s clean** as a member, so 1 to 2 s.

**For.** `V3-allin` refuted the saturation finding by exactly this route, and the
frame builder already exists. **Against.** The per-member table above says the v3
families do not resolve the transient window, so this is a different (and smaller)
axis than S1; and the `fresh` family is a *label-free* set built for a restricted
information diet, which is the least likely place for a new mechanism.

**Why the leaders could have found it.** Everyone adds features. This is the grind
lane, not the step lane, and it is here because it is nearly free.

**Cheapest kill test.** The blend criterion: rotated clean board with the member in
against the same rotation without. **FAIL** below the **0.1025 s** three-fold
half-width, or if either board fold moves the wrong way. **PASS** at ≥ **+0.30 s**
of clean board on the three-fold rotation with both board folds positive — the size
`V3-allin` itself delivered (1.57 s) discounted by the share of that gain the
blend analysis measured on `noac` rows.

**No ceiling is claimed for this candidate.** Its "1–2 s" is `V3-allin`'s *realised*
member gain, not a bound on what a frame swap can do, and this file has no
instrument that bounds it. It is ranked on cost, not on ceiling.

**Cost.** 3 session-hours, 0 GPU-h.

---

### S5 — the January-specific structure (snow days, the 2026 shift)

**Mechanism.** 18 January 2026 airport-days clear their airport's 2025 maximum
daily mean `MVT − AOBT` by more than 30 s, and **11,968 rows (7.84 % of the month)**
sit on days past that envelope against 0.73 % in the january fold. On every 2025
fold the head is biased low on such days.

**Cells and ceiling — this corrects the queue, and then is corrected twice itself.**
The team's queue on 2026-09-07 prices this at "1–2 s, ceiling 4.3 s". The arithmetic
does not support that: most of the 4.276 s offset is variance on those rows and a
level correction removes none of it. The board's **bias** term is
`(rows / 344,841) · bias²`. Three readings, in the order they were made:

| reading | rows and bias | mass | clean board |
|---|---|---|---|
| this file, round 0 (**wrong**) | 11,968 × −144 | 719.7 s² | 1.392 s |
| the exposed rows **split** | 7,564 × −143.5 + 4,404 × −2.6 | 451.8 s² | **0.877 s** |
| exp-029, the same bias on **the head we ship** | 7,564 × −41.4 + 4,404 × −2.6 | 37.6 s² | **0.073 s** |

The first was wrong because −144 s is the bias of the 7,564 rows that clear the
envelope by more than 30 s, not of all 11,968 (the historical exp-014 results record, lines 276–277; the
other 4,404 are biased −2.6 s). The second is right about the split and wrong
about the base: the January analysis re-measured that bias on the candidates we
now ship and it falls from −143.5 (exp-014's candidate 1) to −74.5 (blend-5) to
**−41.4 (stack-5)**. And exp-029 did not stop at a projection — its fitted arms
move the stack-5 board by **−0.083 s**, and its day-constant oracle on the flagged
fold rows is 0.03 / 0.23 / 0.03 s (`exp-029 analysis/arms.csv`, `ceilings.csv`).

**So S5 is worth 0.03 to 0.07 s projected and measures negative on the shipping
head.** (Round-2 finding 6: the −41.4 s in the table above is blend-5's pooled
bias. stack-5's own is **−22.3 s pooled and −41.0 s past 30 s**,
the historical exp-029 results record, line 135, giving 0.033 s pooled and 0.072 s on the split.) The head already repairs most of what this lever was going to sell. It is
ranked last, and it is here as a measured re-price rather than a candidate.

**For.** The mechanism is measured and the flag is computable from the ranking file
alone; no fit, no GPU. **Against.** exp-002 §12 measured that the whole ERA5 set is
a wash on the folds (+0.9 s where the phenomenon is, −2.3 s where it is not), and
the january fold's in-envelope worst 5 % is biased the *other* way (+15.4 s), so a
correction applied too widely trades one bias for another.

**Why the leaders could have found it.** Anyone who noticed January 2026 is not an
ordinary winter would try it. It is worth about a day of the field's improvement
rate, not a step.

**Cheapest kill test — already run, and it failed.** Leave-one-day-out over the
five 2025 analogue days applied to the january fold: FAIL unless january
`rmse_clean` improves by more than the 0.53 s single-seed sd with a
**calendar-day-clustered** interval clear of zero and july_loo is not harmed by
more than 0.1 s. The January analysis ran exactly this
(`exp-029 analysis/arms.csv`): every arm shape moves the **stack-5** board by
**−0.083 s**, july_loo by −0.144, january by +0.016 against a 0.53 s bar. **FAIL on
all three clauses.** The diagnosis is in the table above — the bias the lever
corrects fell from −143.5 s to −41.4 s when the head improved — and the repair that
would revive it is a bias that grows again on a future base, not a better arm.

**Cost.** 2 session-hours, 0 GPU-h.

---

### S6 — a different parameterisation of the mixture (predict `r` and its regime)

**Mechanism.** Instead of predicting `y`, predict `r = BLOCK − AOBT` together with
which component of the mixture the row is in: `r ≈ 0` (the airport clock and NM
agree), `r ≈ LOBT − AOBT` (the block time sits on the filed off-block), or neither.

**Cells and ceiling — measured, and it is small.** A **perfect** detector of "the
block really is within 60 s of AOBT", firing on 65,663 of 320,000 board rows, is
worth **6.02 s** (6.78 s no-noac); at a realistic half-way shrink, 4.66 s. And the
mixture does not describe the bulk: only **19.2–19.5 %** of NM rows sit within 60 s
of the AOBT anchor and **9.1–11.2 %** within 60 s of the LOBT anchor, while
**71.6–73.4 %** are near neither and carry **56.6 %** (july) and **70.3 %**
(january) of the fold's squared error at RMSE 219–254.

**For.** The three GBM members already predict `r`, so the machinery exists, and
the `nm_late` regime is a proven instance of the LOBT component (RMSE 1,562 against
3,960 for the AOBT anchor on the 503 rows above the 3600 s screen).
**Against.** The cross-fitted intercept on airport × `d_takeoff_aobt`(10) ×
`d_aobt_lobt`(10) is **−0.07 s**: the head is already unbiased on exactly the cells
this reparameterisation would name, so it cannot buy a level, only estimation
efficiency. The 6.02 s figure bounds the **narrow AOBT-anchor arm only** and is not
a cap on the reparameterisation as a whole — see "What 6.02 s does and does not
bound" below, which is the governing statement.

**Why the leaders could have found it.** It is the natural second reading of the
data page and it is what our own SUMMARY says the task is. If a leader built the
model this way from the start, their step is our S6 realised — and this file has no
instrument that bounds *that*, only the 6.02 s bound on the narrow anchor arm.

**What 6.02 s does and does not bound.** It is exact for the stated narrow object:
a perfect detector of the AOBT anchor, serving `MVT − AOBT` on the 65,663 rows where
the block really is within 60 s of it. **It does not cap the three-regime `r` model**,
and this file does not claim it does: the 71.6–73.4 % of NM rows near neither anchor
carry 56.6 % (july) and 70.3 % (january) of the fold's squared error, and nothing
here bounds what a better `r` parameterisation could take from them. That part of
S6 is ranked without a ceiling.

**Cheapest kill test.** On committed OOF alone, with no fit: train a small
classifier for `|BLOCK − AOBT| < 60` on two folds and serve
`p·(MVT − AOBT) + (1−p)·head` on the third. **FAIL** if the clean board gain is
below **0.1025 s**, or if the realised gain is under **10 %** of the 6.02 s
perfect-detector bound — the p-rule's own prediction for a detector of this quality,
and the level at which the narrow arm is not worth a second session.
**PASS** at ≥ **+0.60 s** of clean board with both board folds positive.

**Cost.** 2 session-hours, 0 GPU-h.

---

### S7 — the placeholder class as a population-wide phenomenon

**Mechanism.** `exp-002` and `exp-003` find regimes where the recorded block time is
not a block time but a copy of the schedule or the filed off-block. The proposed test
asks whether that class exists beyond `noac`.

**Cells and ceiling.** Beyond `noac` the class is thin where it would pay:
**9.1 %** (july) and **11.2 %** (january) of NM rows sit within 60 s of the LOBT
anchor, and the LOBT anchor served everywhere scores 685.9 on the july bulk against
the head's 209.3. `exp-022`'s population-wide P-oracle is worth 6.9 s of board and
its fitted arm realises **0.27 s** — 4 % — against the 17 % the p-rule predicts for
an AUC-0.87 detector.

**For.** The oracle is real and the class is real. **Against.** `detectability.log`
says the ceiling is structural: the best deterministic subset is 14.6 % at lift 3.7,
and day-level information is 1.0 % of row variance of which only 0.6 % is
between-airport and therefore available in 2026.

**Why the leaders could have found it.** Same as S2.

**Cheapest kill test.** The only repair left is the one S2 needs — fit `p` by the
scored loss rather than by log-loss. **FAIL** if the clean board gain on the
shipping head is below **0.1025 s**, if either board fold moves the wrong way, or if
`noac` RMSE moves by more than 0.01 s (this candidate must not pay itself out of
S2's population). **PASS** at ≥ **+0.30 s** of clean board with both board folds
positive, which is the point at which the fitted arm has beaten `exp-022`'s realised
0.27 s on the old base by more than the rotation half-width.

The 6.9 s figure quoted above is `exp-022`'s P-oracle **on its own base**, not a
budget cell of the head we ship; this file does not restate it as a current-head
ceiling, and the honest current-head bound for the non-`noac` part of this class has
not been measured. If it fails, this is the lane's ceiling and it should be written
up with the detectability log as the evidence rather than reopened.

**Cost.** 2–3 session-hours, 0 GPU-h.

---

### S8 — open surface-trajectory data, and the eligibility question it raises

**Mechanism.** The target is the pushback event. Open ADS-B and Mode-S surface data
covering these ten airports would let a team **observe** it rather than infer it,
and would collapse the error on every row with surface coverage.

**Cells and ceiling.** All of them. This is the only candidate in the list whose
ceiling exceeds 30 s, and by a wide margin.

**For.** The organiser is the OpenSky Network and its historical archive is the
canonical open source for this; `docs/reference-research.md` records that the 2024
winner `team_likable_jelly` used ADS-B-derived features; the team backlog already
carries a 286 GB trajectory archive as a queued item; and `docs/external-data-rules.md`
allows "historical flight or airport-movement data that anyone can access and use
under an explicit open license".
**Against, and it is not a modelling objection.** The same file's own working rule
says "**quarantine post-event actual data that can reconstruct taxi-out time**", and
this reconstructs it directly. `external-data-rules.md` lists "whether actual
January or July 2026 external observations are acceptable when they were published
only after the event" as **not settled by the published rules**, and says to ask the
organisers before this source can be used.

**Why the leaders could have found it.** A 74.6 s step followed by a 56.3 s step in
under a day (upstanding-firefly) is the shape of a team adding a whole new data
source, not of a team tuning. It is the single best explanation in the record for
that trajectory, and it is not evidence that they did it.

**Cheapest kill test.** Not a model test, and it has two numeric gates.

- **Coverage, measured first and free.** Read the OpenSky historical archive at the
  ten airports for January and July 2026 and count the ranking departures that have
  a surface state-vector track covering the whole interval from AOBT to take-off.
  **FAIL** below **70 %** coverage on the board-weighted ranking rows, below
  **70 %** in **either month taken separately** (the board is 55.7 % July and
  44.3 % January, so a combined figure can hide a failed month behind a good one),
  or below **50 %** at LIRF alone — LIRF is 36.15 % of the budget and 397 of the
  402 `noac` rows, so a source that misses it cannot pay for the eligibility
  question. **PASS** at ≥ 90 % board-weighted, ≥ 85 % in each month separately, and
  LIRF ≥ 80 % in each month.
- **Eligibility, and only if coverage passes.** One question to the organisers in
  the public `#prc-data-competition` channel.

**Ceiling.** This is the only candidate whose ceiling exceeds 30 s, and §12c is the
positive evidence for it rather than an assertion: 96 observable columns from *this*
file, arrivals and runway included, recover 3–4 % of a 16.04 s component that every
member carries. The state that drives it is not in this file.

**Cost.** 1 session-hour to size the coverage; a day and a half to build if the
answer is yes; **and it must not be built before the answer.**

---

## 6. Rejected orderings, stated

- **`noac` first (S2 → S1).** It has the largest per-row ceiling and the
  blend analysis shows it dominates the between-candidate fold variance.
  Rejected because it is the one ceiling two experiments have already attacked and
  failed to realise, and because 402 rows cannot be verified on the board.
- **Tail detection first (S3 → S1).** It has the second largest ceiling. Rejected
  because exp-002 already measured the detectable probability at p = 0.082 and
  wrote the revival condition; the screen that would revive it is an hour's work and
  is queued as such, but it is a gate, not a plan.
- **LIRF first.** 36 % of the budget. Rejected because 19 of those 36 points are the
  noac rows and the rest is not a distinct mechanism.
- **By ceiling alone.** That ordering puts "perfect prediction on ordinary rows,
  134.6 s" at the top, which is not a candidate.

## 7. What is public about the leaders, and what I did

Nothing about the 2026 leaders' methods is public. The challenge page
(`https://prc-data-challenge-2026.netlify.app/`, redirected from
`ansperformance.eu/study/data-challenge/dc2026/`) points at the OpenSky Discord
channel `#prc-data-competition` for news; reading it needs a Discord account on the
OpenSky server, **I did not join it and I posted nothing anywhere**. There is no
`github.com/prc-data-challenge-2026` organisation yet (HTTP 404 on 2026-09-07), so
the 2026 solutions are not published. The only public traces that bear on a
structural step are the prior years' organisations
(`prc-data-challenge-2024`, `-2025`), whose winning entries used trajectory-derived
features — which is the evidence behind S8 and nothing more — and the leaderboard
API's own score histories, preserved in public board snapshots.

## 8. The question this document asked, and the answer it got

The open question was whether the transient window's **observable proxy** is in this
file. The data landed while this document was being written, so it is answered
rather than left open, and the answer is the least comfortable of the three that
were possible.

`experiments/exp-027-step/code/cell_reach.py` puts the question directly: regress
the airport-window **mean** residual of the shipping head on every observable the
file has — 131 columns including **the arrivals' unmasked `TAXITIME_SEC_mvt` and
`BLOCK_TIME_UTC_mvt`, the runway split, the busiest-runway mode and its
configuration change**, with lead and lag — cross-fitted across folds.

| | 600 s | 1800 s |
|---|---|---|
| ceiling on these cells | +12.65 s | +11.11 s |
| **reached at unit scale** | **+0.06 s (0.5 %)** | **−0.01 s (−0.1 %)** |
| reached at the best shrinkage, **out of fold** | **+0.39 s (3.1 %)** | **+0.42 s (3.9 %)** |
| the same with the shrinkage chosen in sample | +0.41 s (3.3 %) | +0.44 s (4.1 %) |

Removing the arrivals, the runway and the configuration from the 1800 s family
costs about a percentage point (best scale 3.9 % → 2.9 %). **So the columns K2 was
built to test are real and small.**

> These are the repaired figures. The first run's
> 96 columns built the arrivals' block time and never aggregated it, and dropped
> the busiest-runway modes after deriving their change flag — so the feature set
> did not contain what this section claimed. Both are now in (131 columns) and the
> shrinkage is fitted out of fold. The reach moved from 3.2 % to 3.1 %: **the
> argument below survives on a feature set that now says what it does.**

**The reading, stated as a diagnosis and not as a close.** The component is real
(+0.175 correlation), large (16.04 s no-`noac`), carried identically by every
member, and it survives the stacker on every fold including january (11.85 % of
january's own MSE). What this file does not contain is its cause. Three things
follow, in the order they should be tried:

1. **The delivery repair is measured and works, twice.** Inside the stacker
   rather than on top of it, +0.402 s with january positive (the window model). Inside the `F1-fresh` member's own frame, **+0.8719 s of clean
   board against stack-5 and +0.2945 against v7**, all three folds positive
   (the historical exp-027 results record §16). The layer fails and the two
   in-model deliveries both work, so §12a's negative was the extra layer's refit
   cost and nothing about the mechanism.
2. **The in-file proxy we have not built** is stand-to-runway routing distance: no
   member carries it, and it is the one observable that would explain why two
   departures ten minutes apart share an error. It is the repair that would revive
   the in-file version of this lane if K2 lands under +0.40 s.
3. **S8 is where the cause lives.** 96 observable columns reaching 3–4 % of a 16 s
   component is the strongest evidence this record contains that the state driving
   it — queue position, runway in use, what is holding — is outside the file. That
   makes organizer clarification of eligibility a necessary step before use.
