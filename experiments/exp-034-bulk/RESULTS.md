# exp-034-bulk — RESULTS

Queue lever **L7**, `assignments/bulk.md`. Worker `bulk` (hard), Matt's laptop, 6 threads, CPU
only. Reading and rejected readings: `SPEC.md`. Every number below is a rotation this
experiment recomputed, never a quoted one.

**Reviewer: `gpt-5.6-sol-xhigh-2026-09-08`, two rounds, both BLOCKED** — round 1 five
blocking and six non-blocking, round 2 (on the answers) eight blocking and two non-blocking,
all accepted. Sections 10 and 11 list every finding and what it changed; four of the answers
are new measurements rather than corrected prose. **No measured number moved in either round,
and the verdict did not move**: every arm fails the assignment's gate against K2, and Sol
reproduced the point estimates independently.

## Verdict in six lines

1. **The frame gate passes.** The rebuilt 64-column frame refits exp-011's committed seed-7
   vector to **0.030 s** of bulk RMSE (227.7962 against 227.7665) inside a pre-registered
   0.05 s tolerance, with a row correlation of 0.99973; `features.py check` is exact.
2. **The member is what the queue said it was.** On the july_loo bulk it is **227.80**, i.e.
   **4.12 s better** than `X4-stand` 231.91, the best member in exp-011's committed inventory.
   That arithmetic reproduces exactly.
3. **The comparison it was priced on is two member generations old.** Against the seven
   members it must actually join, the same number puts it **7.85 / 7.44 / 6.93 s BEHIND
   `K2-window`** and 0.86 / 4.40 / 4.31 s behind `V3-allin` on the identical rows.
4. **Every arm FAILS the gate.** Clean board against K2's **257.1653**: the candidate
   (`bulk-add-hold`) **−0.0105**, the pre-registered repair **−0.0092**, and the two
   exploratory output-restricted forms **+0.0129** and **−0.0014**. The bar is **+0.10**; the
   best of the four reaches an eighth of it. None breaks the −0.10 fold floor. **Nothing
   ships. K2 (v9) stands.**
5. **Every point estimate is an order below the 0.1025 s rotation floor.** The honest
   statement is that no arm is distinguishable from K2 on this instrument, in either
   direction, and this experiment registered no equivalence margin, so it cannot claim the
   member is worth exactly zero.
6. **Why, as far as the instrument reaches:** the member does not add to K2's set *and this
   rotation finds no gain on the set that existed before K2 either* — added to v6's own seven
   it is **−0.0207 s** of board against v7, itself inside the floor. That is evidence against
   "K2 took the ground", which predicts a visible gain on the earlier set. It does not
   establish the alternative, and §5 lists what it leaves open.

## 1. The frame gate, and the tolerance

`code/member.py gate`, record `analysis/gate-frame.json`.

| | |
|---|---|
| instrument | exp-011 `oof/seeds_july_loo.parquet`, column `p_s7` — the same 64 columns, 600 rounds, 127 leaves, lr 0.05, seed 7 |
| committed | **227.7665** on 186,272 bulk rows (recomputed from the file, not quoted) |
| rebuilt here | **227.7962** |
| delta | **+0.0297 s**, tolerance 0.05 |
| row difference | mean −0.003 s, sd 8.09 s, 99th percentile of \|diff\| 30.3 s, correlation **0.999726** |
| fit | 1,873,266 training rows, 40 s at 6 threads |

The tolerance and its three reasons are in `SPEC.md`; in one line, exp-003 measures this
harness's own run-to-run spread at about 0.05 s, the committed vector was fitted on another
machine with another LightGBM build, and `PARAMS` sets neither `deterministic` nor
`force_row_wise`.

**What this gate is, and what it is not (Sol non-blocking 1).** It is a *performance*
reproduction, not a proof of frame identity: it compares an aggregate model RMSE built on the
cached frames against a committed vector, and no historical copy of the frame survives to
compare column by column — `artifacts/exp-003-anatomy/feat/` is git-ignored and was lost with
the workstation (recorded rule). Two things guard the gap. `features.py build_all` **skips a month that
already exists**, so a stale cache would go unnoticed; `code/verify_frame.py` therefore
rebuilds one month into a scratch directory and compares the 64 selected columns against the
cache value by value, and reports **exact agreement on 2025-07, 190,713 rows, 64 of 64
columns**. And `features.py check` returns `False` without a non-zero exit status, so a
failure there is visible only in the log; the log is quoted below and in `logs/`.

Two further checks bind and both pass:

- `experiments/exp-003-anatomy/code/features.py check` — every feature bit-identical built
  from the raw month and from the ranking-style masked month: **mask-invariant: True**.
- `code/contract.py` → `analysis/frame-contract.json`: the 64 column names in order with
  SHA-256 `6a7b3d567656…`, `prc.stand_features` VERSION 1.2.0, the frozen LightGBM
  parameters, the SHA-256 of all thirteen rebuilt month frames, and each column's null share
  and mean on each fold's validation month and on the ranking file. This is a **forward**
  contract — it makes the next drift visible — not a comparison with the old frame. This lane
  had one silent frame drift already.

One thing the rebuild surfaced: the month frames are **99 columns** where
`experiments/exp-003-anatomy/logs/features.log` records 92. The seven are exp-003's own
`CSN` callsign block, added by its wave 4 after that log line was written. The 64 columns
this member uses are a subset of both, and none of them is in `CSN`.

## 2. The member

`code/member.py folds` and `rank`; one LightGBM configuration, seed 7, 600 rounds, no tuning.
`oof/bulk64-{fold}.parquet` carry `MVT_ID_mvt, fold, y_true, y_pred, off_pred`.

`COMMON.md`'s four slices, per fold (`code/slices.py`, `analysis/member-slices.csv`). The
member is undefined off the NM join, so the no-NM slice has **no rows to score**: it is `N/A`
by construction, not omitted, and every other slice states how many of its rows the member
cannot reach.

| fold | slice | RMSE | rows scored | rows in slice | undefined |
|---|---|---|---|---|---|
| july_loo | overall | **268.3812** | 186,775 | 190,713 | 3,938 |
| | ≤3600 | **227.7737** | 186,272 | 190,057 | 3,785 |
| | >3600 | **2744.657** | 503 | 656 | 153 |
| | no-NM | **N/A** | 0 | 3,938 | 3,938 |
| | noac | **N/A** | 0 | 339 | 339 |
| december | overall | **211.0003** | 164,028 | 165,677 | 1,649 |
| | ≤3600 | **195.3868** | 163,838 | 165,416 | 1,578 |
| | >3600 | **2348.610** | 190 | 261 | 71 |
| | no-NM | **N/A** | 0 | 1,649 | 1,649 |
| | noac | **N/A** | 0 | 89 | 89 |
| january | overall | **217.5181** | 152,271 | 153,704 | 1,433 |
| | ≤3600 | **199.7475** | 151,999 | 153,381 | 1,382 |
| | >3600 | **2047.191** | 272 | 323 | 51 |
| | no-NM | **N/A** | 0 | 1,433 | 1,433 |
| | noac | **N/A** | 0 | 63 | 63 |

Those figures are the member as the harness prepares it — floor then residual guard, as
`build_candidate` prepares every member. The raw fit, before that preparation, reads 227.7962
/ 195.3933 / 199.7644 on the bulk slice; the 0.02 s difference is the floor and the guard, and
the frame gate above compares raw against raw. Fits: 40 / 46 / 48 s, 1.87 M / 1.90 M / 1.91 M
training rows.

`analysis/member-folds.json` also carries `rmse_clean_zerofill`, which scores the member on
every row of the fold with its undefined rows filled with zero. It is a shape diagnostic from
the build, it is dominated by that fill, and **no conclusion here uses it**.

Twelve-month ranking fit: 2,059,538 training rows — exactly exp-011's "2,059,538 of
2,085,047 2025 departures (98.78 %)", an independent check that the training population is
the one the queue priced — 344,841 predictions, 5,290 undefined (1.53 %, and COMMON.md's own
figure for no-NM ranking departures is 1.5 %). Validated by `code/land_rank.py`
(`analysis/bulk-ranking.json`, sha `dda5b873…`): exactly the submitting template's id set,
ids unique, and **finite on exactly the rows where the ranking file's `d_takeoff_aobt` is
finite**.

**Where the member is undefined, and why that is the contract.** It predicts
`max(nm_taxi − offset, 0)` with `nm_taxi = MVT_TIME_UTC_mvt − AOBT_3_flt`, so it has no
prediction without `AOBT_3_flt`. `code/candidate.py` checks per fold that the member is
defined on exactly the rows where the harness's own `d_takeoff_aobt` is defined — two
independent producers, exp-003's frame builder and `C2-cat`'s out-of-fold file — and that it
is never defined on a no-NM row. It holds on all three folds. That population is **not** the
NM join: 4 / 6 / 3 rows per fold carry a `FLIGHT_ID_mvt` with no `AOBT_3_flt`, so they have
an NM join and no prediction. The count is printed rather than tolerated silently. Sol
verified independently that `PARTIAL` registration changes nothing for the seven committed
members — the anchor, the residual guard, the regime labels and the nan-aware aggregates are
all equal to K2's where the added member is null (§10, non-blocking 2).

### 2a. The member against the set it has to join (`code/corr.py`)

`analysis/member-level.csv`. Every member on the **identical** bulk rows (NM join, y ≤ 3600,
not `noac`), prepared exactly as the blend prepares them.

| member | july_loo | december | january |
|---|---|---|---|
| **K2-window** | **219.928** | **187.943** | **192.815** |
| V3-allin | 226.918 | 190.984 | 195.442 |
| **B1-bulk64** | **227.774** | **195.387** | **199.748** |
| T-tailv4 | 236.081 | 200.944 | 204.845 |
| C2-cat | 236.141 | 201.303 | 204.534 |
| D1-winter | 242.528 | 199.645 | 207.919 |
| L2-hub8725 | 240.904 | 204.212 | 207.874 |
| N1-nmtail | 254.323 | 217.827 | 221.586 |
| **B1-bulk64 minus the best** | **+7.85** | **+7.44** | **+6.93** |

**This is the measurement that makes the candidate unsurprising; it is not a demonstration of its cause** (Sol round 2, blocking 1 — a member's standalone rank and its marginal value in a stacker are different quantities, and nothing here measures the second from the first). L7's headline — "4.1 s better than the
best committed single member on the july_loo bulk" — is arithmetically right and it
reproduces: `X4-stand` scores 231.914 in exp-011's committed
`analysis/oof_inventory.csv` and this member scores 227.796, a 4.118 s edge (the inventory's
bulk mask carries 4 more rows than this one, 186,276 against 186,272; the difference is far
below the gap). But that inventory is the exp-008 field. The set the member was asked to join
contains `V3-allin`, which did not exist when the by-product was measured, and `K2-window`,
which was fitted the day before this experiment. **Priced against them the member is third of
eight, not first.**

### 2b. Diversity (`analysis/member-corr.csv`)

Residual correlation on the non-`noac` rows, pairs taken where both residuals are finite.

| against | july_loo | december | january |
|---|---|---|---|
| K2-window | 0.9047 | 0.9144 | 0.9133 |
| T-tailv4 | 0.8917 | 0.9197 | 0.9057 |
| V3-allin | 0.8703 | 0.9024 | 0.9031 |
| C2-cat | 0.8660 | 0.8956 | 0.8891 |
| L2-hub8725 | 0.8524 | 0.8755 | 0.8709 |
| D1-winter | 0.8400 | 0.9095 | 0.8857 |
| N1-nmtail | 0.8428 | 0.8761 | 0.8659 |
| **mean of the seven** | **0.8668** | **0.8990** | **0.8905** |

For scale, exp-032 measured `T-tailv4`'s mean correlation with the rest of its set at 0.8985
/ 0.8955 / 0.9410 (`experiments/exp-032-k3/analysis/member-corr.csv`). The bulk member's
numbers are at or below those on every fold. **The comparison is indicative only and no
conclusion rests on it** (Sol round 2, blocking 1): the two sets of pairs are scored on
different populations — exp-032's include the no-NM rows this member is null on — and this
experiment measured no relation between a member's correlation with a set and what it adds
to that set. What the table does show is that the member is not a near-copy of any single
member it joins.

## 3. The candidate

`code/candidate.py`, `analysis/candidate-bulk.json`, log `logs/candidate.log`. K2's stacker —
the exp-027 harness, `stack5w.arm`, five seeds, 300 rounds, the 600 s `depx` window block
inside the stacker, the blend-5 anchor on the committed seven, `noac` held at that anchor,
EHAM served from stack-5 as v7 does — with `B1-bulk64` **added** before `N1-nmtail`.

### 3a. The gates, all recomputed in the same process

| gate | what | result |
|---|---|---|
| A | blend-5 on the committed seven reproduces `blend-config-5.json` | delta **+0.0000** on all three folds |
| B | the no-window stacker lands on exp-026's stack-5 | RMSE delta 0.0; **max abs row difference 0** |
| C | the 600 s window arm lands on exp-028's v6 | RMSE delta 0.0; max abs row difference 6.8e-13 |
| D | `k2-hold-eham-out` rebuilds K2 (v9) | 284.3195 / 205.2156 / 218.2582, **board 257.1653, delta −0.0000** |

Gate D matters most: the row this candidate has to beat is rebuilt here rather than quoted,
and the pre-mask `k2-hold` line (284.3938 / 205.2990 / 218.4232, board 257.2730) is
**identical to exp-032's own run of the same arm**, which is an independent check that
registering a partial eighth member changed nothing about the seven. Sol checked the saved
vectors against exp-027's own at max abs differences 2.3e-13 / 0 / 0.

### 3b. The four arms

Clean board, in the **gain** convention throughout: positive is better than K2.

| arm | pre-registered | july_loo | december | january | **clean board** | **vs K2** | gate |
|---|---|---|---|---|---|---|---|
| K2 (v9), rebuilt | — | 284.3195 | 205.2156 | 218.2582 | **257.1653** | — | — |
| `bulk-add-hold` (the candidate) | yes | 284.3591 | 205.1995 | 218.2214 | 257.1758 | **−0.0105** | **FAIL** |
| `bulk-add-bulkonly-hold` (the repair) | yes | 284.3493 | 205.2080 | 218.2338 | 257.1744 | **−0.0092** | **FAIL** |
| `bulk-add-nmbulk-out` | **no, exploratory** | 284.3170 | 205.2509 | 218.2280 | 257.1523 | **+0.0129** | **FAIL** |
| `bulkonly-nmbulk-out` | **no, exploratory** | 284.3308 | 205.2091 | 218.2434 | 257.1666 | **−0.0014** | **FAIL** |

Every arm clears the −0.10 fold floor on every fold; every arm misses the +0.10 bar. The best
of the four is +0.0129, **an eighth of the bar and an eighth of the 0.1025 s rotation floor**.

95 % intervals against K2, per fold, in **both** clustering units — the harness's
calendar-day × airport (`panel.clusters`, about 310 clusters a fold, the unit every candidate
row in `docs/LEDGER.md` from stack-5 onward is quoted on) and the literal calendar day the
assignment names (31 clusters a fold):

A bound in **bold** excludes zero. Every bound is the persisted value at six decimals, and
`analysis/candidate-bulk.json` carries an explicit `contains_zero` beside each interval, so
no reader has to infer the answer from a rounded bound (Sol round 2, blocking 2: at four
decimals the january repair bound below printed as `−0.0000` although it excludes zero).

| arm | unit | july_loo | december | january |
|---|---|---|---|---|
| `bulk-add-hold` | day × airport | −0.0396 [−0.0915, +0.0098] | +0.0162 [−0.0893, +0.1040] | **+0.0368 [+0.0045, +0.0698]** |
| | calendar day | −0.0396 [−0.0933, +0.0100] | +0.0162 [−0.0991, +0.1123] | **+0.0368 [+0.0095, +0.0641]** |
| `bulk-add-bulkonly-hold` | day × airport | **−0.0298 [−0.0616, −0.0007]** | +0.0077 [−0.0268, +0.0409] | +0.0245 [−0.0020, +0.0498] |
| | calendar day | **−0.0298 [−0.0620, −0.000035]** | +0.0077 [−0.0215, +0.0368] | **+0.0245 [+0.0051, +0.0461]** |
| `bulk-add-nmbulk-out` | day × airport | +0.0025 [−0.0223, +0.0281] | −0.0352 [−0.1156, +0.0283] | **+0.0302 [+0.0088, +0.0508]** |
| | calendar day | +0.0025 [−0.0176, +0.0206] | −0.0352 [−0.1302, +0.0390] | **+0.0302 [+0.0107, +0.0475]** |
| `bulkonly-nmbulk-out` | day × airport | −0.0113 [−0.0329, +0.0092] | +0.0066 [−0.0193, +0.0333] | +0.0148 [−0.0076, +0.0368] |
| | calendar day | −0.0113 [−0.0291, +0.0067] | +0.0066 [−0.0187, +0.0342] | +0.0148 [−0.0045, +0.0332] |

Over the two **pre-registered** arms, **four of six** intervals contain zero under the
harness's unit and **three of six** under the literal calendar day (Sol round 1, blocking 2:
the earlier text quoted the four-of-six count against the calendar-day label). The unit
changes which of the marginal intervals crosses zero and changes no point estimate and no
gate. The two exploratory arms add five more intervals containing zero and one that does not
(january, `bulk-add-nmbulk-out`, under both units).

Slices (clean rows, `score.py`'s four plus `no_noac`), for both pre-registered arms and the
best exploratory one:

| arm | fold | n | rmse | ≤3600 | >3600 | no-NM | noac | no-noac |
|---|---|---|---|---|---|---|---|---|
| `bulk-add-hold` | july_loo | 190,713 | 284.359 | 247.326 | 2405.29 | 1121.04 | 3407.35 | 245.622 |
| | december | 165,677 | 205.200 | 192.545 | 1797.81 | 663.08 | 2010.46 | 199.893 |
| | january | 153,704 | 218.221 | 196.358 | 2086.11 | 950.08 | 3509.48 | 206.373 |
| `bulk-add-bulkonly-hold` | july_loo | 190,713 | 284.349 | 247.328 | 2404.89 | 1120.97 | 3407.35 | 245.610 |
| | december | 165,677 | 205.208 | 192.593 | 1795.17 | 663.89 | 2010.46 | 199.901 |
| | january | 153,704 | 218.234 | 196.368 | 2086.26 | 949.98 | 3509.48 | 206.386 |
| `bulk-add-nmbulk-out` | july_loo | 190,713 | 284.317 | 247.290 | 2404.93 | 1120.90 | 3407.35 | 245.573 |
| | december | 165,677 | 205.251 | 192.594 | 1798.18 | 663.97 | 2010.46 | 199.945 |
| | january | 153,704 | 218.228 | 196.364 | 2086.19 | 950.06 | 3509.48 | 206.380 |
| `bulkonly-nmbulk-out` | july_loo | 190,713 | 284.331 | 247.313 | 2404.71 | 1120.90 | 3407.35 | 245.589 |
| | december | 165,677 | 205.209 | 192.600 | 1794.74 | 663.95 | 2010.46 | 199.902 |
| | january | 153,704 | 218.243 | 196.379 | 2086.26 | 950.06 | 3509.48 | 206.396 |

The `noac` column is stack-5's own bytes in every arm by construction: it is identical to K2's
and moves exactly 0.0000.

Against K2 by slice (positive is better): the candidate is `y ≤ 3600` **−0.013 / +0.072 /
+0.027** and `y > 3600` **−0.99 / −3.72 / +0.61**; its whole board loss is the december tail,
and the bulk slice it was built for is level.

All four arms beat **v7** with every fold's interval clear of zero: **+0.284 / +0.285 /
+0.3075 / +0.2932** of board. K2, already on the board as v9, beats v7 by +0.2945, so only the
`bulk-add-nmbulk-out` arm is ahead of K2 at all and it is ahead by an eighth of the bar.

## 4. The repair, and the two readings of it

`SPEC.md` registered before any rotation number existed that two additive arms would be
rotated in one process, that `bulk-add-hold` is the shipping arm, and that the
`nm_bulk`-restricted arm ships only if the shipping arm fails **and** the restricted one
passes. The registration is checkable rather than asserted: it is the "Pre-registration of the
shipping arm" section of `SPEC.md` as committed in **d59e7c8**, and the rotation's numbers
first appear in **d9662f3**.

**Sol blocking 1: the pre-registered repair is not an output restriction, and the assignment's
words also carry that reading.** "The member restricted to the `nm_bulk` regime rows in the
stacker" was implemented as *the member column nulled outside `nm_bulk`* — but the stacker is
one global LightGBM model, so its **output** still moves on **16,581 / 6,428 / 5,073**
non-`nm_bulk` rows against K2 (measured here, and the same counts Sol found independently).
The literal output form was therefore not measured. It is now, in both of its readings, and
both are marked **exploratory** because neither was pre-registered:

| arm | what | vs K2 |
|---|---|---|
| `bulk-add-bulkonly-hold` | member column null outside `nm_bulk`, one global stacker | −0.0092 |
| `bulkonly-nmbulk-out` | that arm's output inside `nm_bulk`, K2 on every other row | **−0.0014** |
| `bulk-add-nmbulk-out` | the full-column arm's output inside `nm_bulk`, K2 elsewhere | **+0.0129** |

The −0.0014 reproduces Sol's own independent computation to four decimals. **All three fail
the gate**, the best of them at an eighth of the bar. The restriction does what it is meant to
do — it takes back the december tail damage (`y > 3600` against K2 goes −3.72 → −0.65 on the
`bulkonly` line) — and it does not turn the board.

`serve.py` refuses to serve an arm whose record says it failed the gate, and does so here:

    bulk-add-hold FAILED the assignment gate against K2 (gain -0.0105, worst fold -0.0396);
    it is not servable                                                        [exit 1]

**No candidate parquet was written.** `assignments/bulk.md` step 4 serves the passing arm and
there is none. `serve.py` also now refuses an arm marked exploratory and refuses the repair
whenever the shipping arm passed, which is what `SPEC.md`'s registration says (§10,
non-blocking 6).

## 5. Why — what the diagnosis reaches, and what it does not

`code/diagnose.py`, `analysis/diagnose.json`, log `logs/diagnose.log`.

Two mechanisms were named in advance as candidate explanations of a −0.01 s addition.
**(a) K2 took the ground**: the member's information is real, the set now holds a better copy,
and the negative is about timing. **(b) The frame was already served**: the stand-occupancy
family overlaps the v3 frame's `stand` source — the queue's own caution — and `V3-allin` is in
every set. They are the two the queue's own text suggests; **they are not exhaustive**, and
§5c lists what else is live.

One arm separates (a) from a null: the member added to **stack-5's own seven with the window
block**, which is v6's construction, the head as it stood before K2 existed, scored against v7
on the same instrument. The `v6` control in that run reproduces the committed v7 at max abs
row difference 6.8e-13.

| arm | july_loo | december | january | clean board | vs v7 |
|---|---|---|---|---|---|
| v7 (the control) | 284.6251 | 205.5677 | 218.5412 | 257.4598 | — |
| **v6 + B1-bulk64** | 284.6847 | 205.5738 | 218.4987 | **257.4805** | **−0.0207** |

Per fold, day × airport 95 %: july_loo −0.060 [−0.109, −0.009], december −0.006 [−0.106,
+0.087], january +0.043 [+0.006, +0.080].

**What that supports, stated at its own strength (Sol blocking 3).** This rotation finds **no
useful pre-K2 gain**: the point estimate is negative and inside the 0.1025 s rotation floor,
and the three folds do not agree in sign (−0.060, −0.006, +0.043). That is evidence *against*
reading (a), which predicts a visible gain on the earlier set. It is **not** a proof of (b),
and it is **not** an equivalence claim: no equivalence margin was registered, so "the member
adds nothing" is a description of what this instrument measured, not a demonstration that the
true effect is zero. A gain of a few hundredths of a second would be invisible here.

### 5a. The stacker is not ignoring the column

Share of LightGBM split gain, one seed of the five, arm `bulk-add-hold`:

| fold | B1-bulk64 | K2-window | T-tailv4 | V3-allin | N1-nmtail | D1-winter | C2-cat | L2-hub8725 | window block |
|---|---|---|---|---|---|---|---|---|---|
| july_loo | **0.0351** | 0.0870 | 0.0516 | 0.0366 | 0.0418 | 0.0325 | 0.0091 | 0.0098 | 0.578 |
| december | **0.0486** | 0.0764 | 0.0324 | 0.0300 | 0.0372 | 0.0364 | 0.0139 | 0.0140 | 0.596 |
| january | **0.0097** | 0.1070 | 0.0439 | 0.0308 | 0.0263 | 0.0398 | 0.0204 | 0.0191 | 0.593 |

The member takes as much of the stacker's attention as `V3-allin` does on two folds and a
tenth of it on january. **It is used and the board does not move**, which is a different
observation from a column the model never splits on. Without the member
(`gain_split_k2_only`) `K2-window` reads 0.0939 / 0.0899 / 0.1175 and the window block 0.599 /
0.625 / 0.608, both higher on every fold. That is an **association** — the shares are a
simplex and must rise when a column leaves — and not a demonstration that those particular
splits moved from one column to the other.

### 5b. What this says about the queue, beyond this lane

L7's expected gain was set by an analogy to `V3-allin` discounted for frame overlap, on top of
a "4.1 s better than the best committed single member" headline. The headline was true and
the member is still third of eight. **The lesson is that a member's standalone edge over an
inventory ages: two of the eight members it must join were fitted after the inventory was
written, and both are ahead of it.** Any queue item whose size rests on "N s better than the
best committed member" should be re-priced against the *current* set before it is scheduled —
`docs/QUEUE-2026-09-07.md` L6 is written the same way, and its frame overlaps `V3-allin`'s
even more directly, since `v3f` **is** the v3 frame plus four columns.

### 5c. What the diagnosis does not exclude

- **A small real gain**, anywhere inside ±0.10 s. The instrument cannot see it.
- **Capacity and regularisation.** The stacker's `num_leaves` 15, `min_data_in_leaf` 2000 and
  `lambda_l2` 100 are stack-5's, tuned for seven member columns and a 50-column block; an
  eighth column may need a different budget, and none was tried (the assignment forbids tuning
  before the gate is read, and the gate was not passed).
- **The four member aggregates.** `mem_mean`, `mem_std`, `mem_range` and `n_defined` all move
  when an eighth column enters, and `n_defined` moves specifically on the rows where the new
  member is null. An additive arm is therefore never a pure superset, and this experiment did
  not separate the aggregate effect from the member's own.
- **The overlap mechanism itself.** The arm that would test it directly — the bulk member
  substituted into `V3-allin`'s slot — is forbidden by this assignment ("do not substitute").
  It is §6.1.

## 6. What would revive this lane

Stated as conditions, not as hope. The member, its three fold vectors and its validated
344,841-row ranking column are all committed, so each of these is a rotation away and none
needs a refit.

1. **The substitution arm.** The bulk member into `V3-allin`'s slot, which is the direct test
   of the overlap reading and costs 34 s. Forbidden here; it is the cheapest remaining
   question about this frame.
2. **A stacker that can hold two bulk voices without them competing.** The same revival
   condition exp-032 wrote for K3: a regime-gated weight, or the member restricted by a
   **label-free** row selector rather than by a regime. Sol round 2, blocking 7: "the rows
   where `K2-window`'s residual is largest" is not such a selector — a residual needs the
   target, which the ranking rows do not carry. A usable selector is fitted on the training
   months and reads only inputs: the stacker's own `mem_std` or `mem_range` (member
   disagreement, already columns in the frame), or a small model of |K2 residual| fitted on
   the training months and applied to the ranking rows. Either one must be gated before it
   is trusted, because it is a second model between the member and the board.
3. **An instrument with a floor below 0.02 s.** Three of the four arms sit between −0.011 and
   +0.013; a rotation that could resolve that band would turn this from "no signal" into an
   answer. The backlog already carries this as the nested-validation row.
4. **A frame that removes the overlap rather than adding to it.** exp-003's own leave-one-out
   says the airport-neighbour family (6 columns) is the part retro-v1 measured as genuinely
   new, at most 0.50 correlated with anything the v3 frame carries. A member built on that
   family plus something the v3 frame has no copy of would be a different lane, and would need
   its own sizing before it is queued.

## 7. What the intervals are, and are not

Sol non-blocking 4, accepted: the two arms were registered before their scores, so nothing
here was selected on the fold that scores it. But the **frame** was — exp-003 chose the
recommended 64-column set from a table sorted by july bulk RMSE on these same folds
(`experiments/exp-003-anatomy/RESULTS.md`, the ablation table), and `K2-window` was developed
on them too. Every interval in this file is therefore a **conditional sensitivity interval**
on a reused instrument, not an independent post-selection confidence interval. It is the same
instrument K2 and v7 are quoted on, which is what makes the comparison fair; it is not a
claim about a fresh sample.

## 8. Artifacts

| what | path | note |
|---|---|---|
| member OOF | `oof/bulk64-{july_loo,december,january}.parquet` | in git; `MVT_ID_mvt, fold, y_true, y_pred, off_pred` |
| ranking column | `candidates/bulk64-ranking.parquet` | in git; sha `dda5b873ef18879eff4cf751ca593da89ef62c8c802dea236caa7787072b53fe` |
| candidate OOF | `oof/exp-034-bulk-{arm}-{fold}.parquet` | in git; **all four scored arms**, twelve files, with K2's (and for the pre-registered arms stack-5's and v6's) vectors beside them |
| rotation record | `analysis/candidate-bulk.json` | four arms, both interval units, every gate |
| frame gate | `analysis/gate-frame.json`; frame re-verification `analysis/verify-frame.json` | |
| frame contract | `analysis/frame-contract.json` | 64 names, hash `6a7b3d567656…`, 13 month-frame hashes |
| member tables | `analysis/member-level.csv`, `member-corr.csv`, `member-slices.csv`, `member-folds.json` | |
| diagnosis | `analysis/diagnose.json` | |
| feature frames | `artifacts/exp-003-anatomy/feat/*.parquet` | git-ignored, 13 files, ~340 MB, hashes in the frame contract |
| review | `review/round1.log`, `review/round2.log` | Sol round 1 BLOCKED (5+6), round 2 BLOCKED (8+2); both answered, §10 and §11 |
| logs | `logs/{gate,folds,rank,candidate,diagnose,verify_frame}.log` | |

## 9. Reproduction

```bash
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/features.py          # 13 frames, ~30 s
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/features.py check    # mask invariance
PYTHONPATH=. .venv/bin/python experiments/exp-034-bulk/code/contract.py
PYTHONPATH=. .venv/bin/python experiments/exp-034-bulk/code/verify_frame.py         # cache vs a fresh build
PRC_THREADS=6 .venv/bin/python experiments/exp-034-bulk/code/member.py gate         # the frame gate
PRC_THREADS=6 .venv/bin/python experiments/exp-034-bulk/code/member.py folds
PRC_THREADS=6 .venv/bin/python experiments/exp-034-bulk/code/member.py rank
PYTHONPATH=. .venv/bin/python experiments/exp-034-bulk/code/land_rank.py
PYTHONPATH=. .venv/bin/python experiments/exp-034-bulk/code/corr.py
PYTHONPATH=. .venv/bin/python experiments/exp-034-bulk/code/slices.py
PRC_THREADS=6 .venv/bin/python experiments/exp-034-bulk/code/candidate.py           # 162 s, four arms
PRC_THREADS=6 .venv/bin/python experiments/exp-034-bulk/code/diagnose.py
.venv/bin/python experiments/exp-034-bulk/serve.py                                  # refuses: the arm failed
```

## 10. Sol round 1 — every finding and its disposition

`review/round1.log`, `gpt-5.6-sol-xhigh-2026-09-08`, fresh context, read-only.
**Verdict BLOCKED**, five blocking and six non-blocking. All accepted; none changed a measured
number or the verdict.

| # | finding | accepted | what it changed |
|---|---|---|---|
| B1 | the repair nulls the member COLUMN, not the arm's OUTPUT; 16,581 / 6,428 / 5,073 non-`nm_bulk` rows still move | yes | both output-restricted forms measured and marked **exploratory**: `bulk-add-nmbulk-out` **+0.0129**, `bulkonly-nmbulk-out` **−0.0014** (reproducing Sol's own figure); the moved-row count is now measured and printed by `candidate.py`. Both FAIL. §3b, §4 |
| B2 | the intervals are calendar-day × airport, not the literal calendar day the assignment names | yes | both units computed for every arm (`ci_*` and `ci_*_calday`); the zero-crossing count is now given per unit, four of six and three of six. §3b |
| B3 | §5 stated a null as equivalence and as causal exclusion; the two mechanisms were presented as exhaustive | yes | §5 restated at its own strength ("this rotation finds no useful pre-K2 gain"), the missing equivalence margin named, the split-gain claim demoted to an association, and §5c added listing capacity, regularisation, the member aggregates and a small real gain as live alternatives |
| B4 | no ledger rows | yes | four rows appended to `docs/LEDGER.md`: the member, the candidate, the diagnosis, and this review |
| B5 | `COMMON.md`'s four slices missing for the member and the repair | yes | `code/slices.py` and `analysis/member-slices.csv` added, with `N/A` and a row count wherever the member is undefined; the repair's full slice table added to §3b |
| N1 | the gate tests model performance, not frame identity; `features.py` skips existing months; the mask check does not exit non-zero | yes | `code/verify_frame.py` added: a fresh build of 2025-07 into a scratch directory against the cache, **64 of 64 columns exact on 190,713 rows**; §1 now says what the gate proves and what it does not |
| N2 | `PARTIAL` registration is correct — confirmed | — | quoted in §2 |
| N3 | K2 is like-for-like, the point arithmetic is right, only the clustering unit was wrong — confirmed | — | quoted in §3a |
| N4 | selection reuse: exp-003 chose the frame on these folds and K2 was developed on them, so the intervals are conditional | yes | §7 added |
| N5 | no prohibited input leakage — confirmed | — | quoted in §11 |
| N6 | `serve.py` could serve the repair when both arms pass, and its config check omits threads and the LightGBM version | yes | `serve.py` now refuses an exploratory arm, refuses the repair unless the shipping arm failed, and checks the thread count and LightGBM version |

## 11. Sol round 2 — every finding and its disposition

`review/round2.log`, same reviewer, fresh context, read-only, on the answers to round 1.
**Verdict BLOCKED**, eight blocking and two non-blocking. Round 2 is the last round the
protocol allows. All ten are accepted and answered below; **no measured number moved** — the
rerun of `code/candidate.py` with the fixes reproduces every board and every gain exactly
(−0.0105 / +0.0129 / −0.0092 / −0.0014).

| # | finding | what it changed |
|---|---|---|
| B1 | the causal claim survives outside §5: `code/diagnose.py`'s docstring named two mechanisms as decisive and said the member "adds nothing"; §2a said the standalone rank "explains the candidate"; §2b concluded diversity was not binding from a partly non-like-for-like comparison | the docstring is rewritten (comments only; the arms and `analysis/diagnose.json` are untouched): the two readings are candidate readings, not a partition, and "no gain in either" is evidence against (a) and not a proof of (b). §2a now says the rank measurement makes the candidate unsurprising and does not demonstrate its cause. §2b now draws no conclusion from the correlation comparison beyond "the member is not a near-copy of any member it joins" |
| B2 | the recount was right but the persisted interval contradicted it: four decimals printed the january repair bound −0.000035 as −0.0000 | `against()` persists six decimals **and** an explicit `contains_zero` beside every interval, taken from the unrounded bounds; §3b prints six decimals and names the bound at issue |
| B3 | the ledger row combined four measured arms; `COMMON.md` requires one row per measured result | three more rows appended — the repair and each exploratory arm — beside a round-2 correction row. The combined row stays as written (append-only) |
| B4 | `bulkonly-nmbulk-out` had no slice row and no out-of-fold file | its slices are in §3b's table, and `code/candidate.py` now writes out-of-fold files for **both** exploratory arms in one loop; all four scored arms have twelve files in `oof/` |
| B5 | `interval_convention` claimed every `ci_*` was gain convention, but `ci_v6` is inherited from `experiments/exp-030-window2/code/score.py` in the delta convention and carries the opposite sign to `vs_v6` | the string now names exactly which keys this file writes and in which convention, and says `ci_v6*` is inherited and delta. The shared `score.py` is not changed: other experiments' committed records are quoted against it |
| B6 | `climb.log` recorded `frac_of_bar`, not the fraction of the remaining gap `COMMON.md` asks for, and the member rows had no iteration | nine rows appended restating every attempt in the required schema, with the denominator **named in each row** — L7's own queue estimate of 1.0 s (the midpoint of 0.5–1.5 s), the gap this lever was scheduled to close. The original rows stay |
| B7 | a revival condition proposed restricting output where `K2-window`'s residual is largest; a ranking row has no target | §6.2 and the backlog row now ask for a **label-free** selector fitted on the training months (`mem_std` / `mem_range`, or a model of the absolute K2 residual), and say the selector itself must be gated |
| B8 | the L7 heading still carried "gain 0.5–1.5 s" | the heading now reads "**RUN, gain measured −0.01 s** (estimate was 0.5–1.5 s)" and keeps the pointer below it |
| N1 | the two scoring paths in `candidate.py` are duplicated and have already drifted, which is why one exploratory arm had no out-of-fold file | accepted; the consequence is fixed by B4's single loop. The duplication itself is left: this experiment is finished, and a refactor would change code that produced committed numbers without producing a measurement |
| N2 | editing queue item L6 is beyond this assignment | **reverted**: L6's table row and its section are back to what they were. The observation that L6 is sized the way L7 was stays in §5b, where it is this experiment's own finding, and it goes to the desk in the report, because the desk owns the queue |

Confirmed by Sol in round 2 and quoted here: the `np.where(regime == "nm_bulk", arm, k2)`
construction is the right one for an output restriction; both derived arms are honestly
labelled exploratory and cannot ship; the calendar-day clustering is correct;
`code/verify_frame.py` proves the limited claim §1 makes for the 64 selected columns on
2025-07; **no input leakage in any current arm**; `serve.py` refuses both failed registered
arms and its command line does not expose either exploratory arm; and the numbers in the four
ledger rows agree with the artifacts they cite.

## 12. Leakage self-check against `docs/input-rule.md`

- The member's features are exp-003's, and `features.py check` asserts every one of them is
  bit-identical built from the raw month and from the ranking-style masked month. No feature
  reads a departure's own `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. Arrival
  `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` *are* read; the ranking file keeps them on
  arrival rows and `prc.data.mask_like_ranking` blanks departures only. Sol verified this
  independently in both rounds and found no prohibited input (§10, N5; §11, round 2).
- The training filter uses the label (`y <= 3600`), which is legal on TRAINING months and is
  exp-011's own; the prediction is made on every validation row without it.
- **The rotation's own channel is inherited and is not closed here.** The stacker trains on
  member out-of-fold columns produced by models whose training months include the scored
  fold, so a scored fold's labels reach it through its features (exp-026 RESULTS §4a). Every
  fold number in this repository carries it, including K2's, and the comparison here is
  against K2 on the same instrument in the same process. It means no arm here can claim a
  fully nested out-of-sample result.
