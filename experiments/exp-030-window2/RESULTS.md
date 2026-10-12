# exp-030-window2 — RESULTS

The next four cuts on the transient airport window, priced against **v6** — stack-5 with
exp-028's 600 s window block inside its own stacker, clean 257.6230, board **263.6922**
(`docs/LEDGER.md` BOARD READING 6). Read `SPEC.md` first: §1 states which reading of "two
widths together" this experiment took and what it rejected.

Every number is computed by `code/` from committed bytes plus `data/raw`, at 4 threads. The
three stacker rotations take 79, 96 and 81 seconds each; arm 3 refits nothing. Reproduction
commands in §9. §11 carries the Sol round and the answers to it.

---

## 0. Summary

1. **The shipping arm is arm 3: EHAM out of the served mask.** +0.1632 s of clean board over
   v6, folds **+0.097 / +0.086 / +0.275**, every one of the three day-clustered intervals
   excluding zero. Gate PASS. Predicted board **263.529** at face. §4.
2. **A second window width is worth +0.073 s — real but under the gate.** Arm 1 (600 s and
   300 s aggregate sets side by side, 95 block columns) moves all three folds the right way
   and only july_loo's interval excludes zero. It FAILS the assignment's 0.10 s gain gate. §3.
3. **A third width buys board and breaks december.** Arm 2 (600 + 300 + 1800 s, 140 columns)
   has the best board of any arm, +0.1671 s over v6, and loses **0.210 s on december** with an
   interval that excludes zero. FAIL on the fold floor. Its december loss is not the noise
   exp-028 §7b diagnosed at 600 s: it is four times the rotation's 0.1025 s noise floor and
   the interval says so. §3.
4. **The arrivals keyed on runway and stand is still a negative.** Arm 4 is **−0.025 s** of
   board against v6 on its own and loses on july_loo and december. exp-028 §10's revival
   condition for its `arr` block was **the re-keying, and this experiment performs it**; the
   condition is met and **the answer is still negative**, so the block is not revived. §5.
5. **EHAM is not "any airport".** Excluding each of the ten airports in turn from v6's mask,
   three gain on the board — LFPG +0.052, LSZH +0.009 and EHAM +0.163 — but **EHAM is the only
   one that gains on all three folds and the only one that passes the gate**, and seven of the
   ten are negative, the worst LTFM at −0.196. Inside EHAM's own rows v6 is worse than stack-5
   by **1.44 / 1.00 / 2.52 s** on the three folds. This rejects "serving fewer rows is simply
   safer" **on these folds**; it is not itself independent evidence, because the same folds
   named EHAM. §4.
6. **The selection cost is stated, not bounded.** EHAM was named by a per-airport table
   computed on these same three folds (exp-028 §5), so arm 3's three-fold pass is a further
   cut on the folds that selected it and is exploratory, exactly as exp-028 §7b's caveat says
   of the 300 s width. What is not from these folds is the reason: `docs/shift-2026.md` calls
   EHAM a different regime in January 2026 on evidence that predates this experiment. §4.
7. **The candidate verifies.** `submissions/candidates/exp-030-eham-out.parquet`, SHA-256
   `67b70ee3…`, 344,841 rows in the template's order, all finite, none below the floor, the
   **388 noac rows bit-identical to stack-5** and the 38,182 EHAM rows likewise, rebuilt from
   its own record at a maximum absolute difference of **0.0** — and here that rebuild really is
   independent, because a derived candidate's record names two committed files and an airport
   and nothing else. §7.
8. **Two gates ran on every rotation and both held**: blend-5 reproduces on all three folds,
   and the no-window rotation lands on exp-026's published stack-5 (285.2029 / 205.6118 /
   219.1303) within 1e-3, or `stack5w.main` aborts. `code/score.py` re-gates both stack-5 and
   v6 against exp-028's published rows before any arm is scored. §1.

---

## 1. The base, the comparison point, and the gates

The base is unchanged from exp-028: stack-5 keep-noac, read back from
`experiments/exp-026-rebuild/oof/`. The comparison point is **v6**, read back from exp-028's
committed rotation `oof/exp-028-stack5w-w600-{fold}.parquet` (column `y_pred_depx`).

| | july_loo | december | january | clean board |
|---|---|---|---|---|
| stack-5 keep-noac | 285.2029 | 205.6118 | 219.1303 | 258.0372 |
| **v6 (600 s window in the stacker)** | **284.7219** | 205.6534 | **218.8166** | **257.6230** |

Three gates run and raise rather than warn:

* `stack5w.main` refits blend-5 on every rotation and aborts unless it reproduces on all three
  folds; it then refits the **no-window** stacker in the same process and aborts unless that
  lands on exp-026's published stack-5 within 1e-3. Both held on all three rotations.
* `code/score.py:reference()` re-reads exp-028's rotation and aborts unless stack-5 **and** v6
  reproduce the rows above and the board within 1e-3.
* `code/measure.py` aborts unless an arm's rotation is in the reference's row order **and** its
  own no-window vector equals exp-028's stack-5 to 1e-9. It is, exactly.

Board weights are exp-016's: july_loo 55.71 %, january 44.29 %. December is scored and gated
and carries no board share.

## 2. The four arms

Clean, rotated, against v6. `analysis/arm-*.json` carries each row in full.

| arm | block columns | july_loo | december | january | clean board | vs stack-5 | **vs v6** | predicted board | gate |
|---|---|---|---|---|---|---|---|---|---|
| v6 (the comparison point) | 50 | 284.7219 | 205.6534 | 218.8166 | 257.6230 | +0.414 | — | 263.6922 measured | — |
| 1 — 600 + 300 s | 95 | 284.6135 | 205.6432 | 218.7987 | 257.5495 | +0.4876 | **+0.0734** | 263.619 | **FAIL** (gain) |
| 2 — 600 + 300 + 1800 s | 140 | **284.5221** | 205.8630 | 218.6994 | **257.4559** | +0.5813 | **+0.1671** | 263.525 | **FAIL** (december) |
| **3 — EHAM out of the mask** | 50 | 284.6251 | **205.5677** | 218.5412 | 257.4598 | +0.5774 | **+0.1632** | **263.529** | **PASS** |
| 4 control — 600 s + arrivals on runway/stand, no exclusion | 90 | 284.7857 | 205.7016 | 218.7775 | 257.6476 | +0.3896 | **−0.0246** | 263.717 | **FAIL** (negative) |
| **4 — the same, on the best of arms 1–3** | 90 | 284.6827 | 205.6101 | **218.4942** | 257.4777 | +0.5595 | +0.1453 | 263.547 | pass, below arm 3 |

Per-fold gains over v6 with their day-clustered 95 % intervals — clean, and then clean with
the `noac` rows dropped, which is the population the block actually touches. Positive is an
improvement on v6.

| arm | july_loo | december | january |
|---|---|---|---|
| 1 | **+0.108 [+0.050, +0.173]** | +0.010 [−0.086, +0.092] | +0.018 [−0.048, +0.087] |
| 2 | **+0.200 [+0.053, +0.355]** | **−0.210 [−0.424, −0.013]** | +0.117 [−0.000, +0.249] |
| **3** | **+0.097 [+0.052, +0.151]** | **+0.086 [+0.023, +0.152]** | **+0.275 [+0.142, +0.452]** |
| 4 control | −0.064 [−0.138, +0.004] | −0.048 [−0.112, +0.012] | +0.039 [−0.005, +0.082] |
| **4** | +0.039 [−0.047, +0.130] | +0.043 [−0.047, +0.135] | **+0.322 [+0.183, +0.501]** |

*(No-noac intervals are in `analysis/arm-*.json` under `ci_v6_no_noac`. Every bound moves
outward by at most 0.048 s (arm 2, july_loo) and no sign changes: the `noac` rows are
identical between an arm and v6, so dropping them removes shared rows and widens the interval
slightly without moving the comparison. The four required slices per fold and per arm are
`analysis/slices.csv`, §6.)*

**The gate, stated against the assignment's words.** `assignments/window2.md`: clean gain over
v6 **≥ 0.10 s** and no fold worse than **−0.10 s**. Arm 1 clears the fold floor and misses the
gain by 0.027 s. Arm 2 clears the gain and misses the fold floor by 0.110 s. Arm 3 clears both.
Arm 4's control clears the fold floor and is negative on the board; arm 4 proper clears both
but is 0.018 s below arm 3, which is the same exclusion without the arrivals block.

**And beside the number, the caveat that does not weaken with repetition** (exp-028 §7b): every
arm here is **a further cut on the same three folds v6 was scored on**. Arm 1 and arm 2 choose
a resolution on these folds, arm 3 chooses an airport after seeing a per-airport table computed
on these folds, arm 4 chooses a key. A gate passed on the folds that selected the variant is
exploratory and not a second confirmation. The board is what settles any of it.

## 3. Arms 1 and 2: what a second and third resolution buy

The two width arms are the same construction as v6 with more aggregate sets beside the 600 s
one. Because the first 50 columns are byte-identical to v6's block
(`tests/block_identity.py`: exp-028's own builder and this one agree on every value and every
name, and arm 1's remaining 45 columns are that same builder at 300 s with the five
width-independent row-context columns dropped), arm 1 strictly contains v6's information and the delta
is "what a second resolution adds", not "what a different block does".

**A second width is a real but small positive.** All three folds move the right way — which v6
itself does not do, its december being −0.042 against stack-5 — and the board gain is +0.073 s.
Only july_loo's interval excludes zero. Read against exp-028 §7b, this is the same story from
the other side: the 300 s and 600 s blocks are nearly the same measurement (their boards
differ by 0.007 s), so giving the stacker both adds about what one more nearly-collinear view
is worth, and that is under a tenth of a second.

**A third width is where it breaks.** 1800 s adds board (+0.167 s, the best of any arm) by
adding july_loo (+0.200) and january (+0.117) — and takes 0.210 s off december, with an
interval of [−0.424, −0.013] that excludes zero. That is **four times** the rotation's 0.1025 s
noise floor and twenty times what feature order alone is worth in this construction (0.02 s,
exp-028 §4), so it is not the unresolvable december exp-028 diagnosed at 600 s against 300 s.
The shape says what the arm is: a 1800 s bucket at these airports holds about 30 movements and
is closer to an airport-hour level than to a transient, and a level feature fitted on two
months and applied to a third is exactly the construction exp-028 §7a measured collapsing
(the deviation block, +0.414 → +0.061, "it trades one fold for another"). Arm 2 trades
december for july_loo and january; on the board that is a good trade, which is why it has the
best board and still fails.

**Why arm 2 is not shipped even though its board is the best.** Its board advantage over arm 3
is 0.004 s — inside the noise floor — and it buys that by breaking the one fold the whole
family has been unable to hold. Arm 3 gains on all three.

## 4. Arm 3: EHAM out of the served mask

`code/eham.py`. Nothing is refitted: every EHAM departure keeps stack-5's own value and every
other row keeps v6's. 22,489 / 19,458 / 18,460 rows are taken back on the three folds, 11.8 %
of the rotation, and 38,182 of the 344,841 ranking rows.

**Inside EHAM's own rows, v6 is worse than stack-5 on all three folds** (`analysis/airport_inside.csv`,
clean rows, seconds of that airport's own RMSE; positive is v6 better):

| airport | july_loo | december | january | folds v6 wins |
|---|---|---|---|---|
| LTFM | +1.95 | +1.91 | +0.87 | 3 |
| LEBL | +1.84 | +2.04 | +0.54 | 3 |
| LEMD | +1.13 | +0.43 | +0.56 | 3 |
| LIRF | +0.27 | +0.36 | +1.62 | 3 |
| EDDF | +0.61 | +0.09 | +0.17 | 3 |
| EDDM | +0.68 | −3.12 | +1.76 | 2 |
| EGLL | +0.19 | −1.75 | +1.36 | 2 |
| LSZH | −0.19 | +0.07 | −0.16 | 1 |
| LFPG | −0.28 | +0.03 | −0.69 | 1 |
| **EHAM** | **−1.44** | **−1.00** | **−2.52** | **0** |

**And EHAM is the only exclusion that gains on all three folds.** Excluding each airport in turn from v6's served
mask, board gain over v6 (`analysis/airport_leave_out.csv`):

| excluded | july_loo | december | january | board vs v6 | gate |
|---|---|---|---|---|---|
| **EHAM** | **+0.097** | **+0.086** | **+0.275** | **+0.163** | **PASS** |
| LFPG | +0.028 | −0.004 | +0.092 | +0.052 | no |
| LSZH | +0.009 | −0.003 | +0.010 | +0.009 | no |
| EDDF | −0.044 | −0.007 | −0.014 | −0.033 | no |
| EDDM | −0.035 | +0.234 | −0.102 | −0.060 | no |
| LEMD | −0.072 | −0.037 | −0.043 | −0.061 | no |
| EGLL | −0.020 | +0.222 | −0.179 | −0.080 | no |
| LEBL | −0.148 | −0.145 | −0.027 | −0.101 | no |
| LIRF | −0.050 | −0.044 | −0.205 | −0.108 | no |
| LTFM | −0.244 | −0.260 | −0.122 | −0.196 | no |

**What this table does and does not settle** (sharpened after Sol round 1). It rejects one
explanation on these folds: if the gain were "serving the block on fewer rows is simply safer",
every exclusion would pay a little, and seven of ten are negative with the largest, LTFM,
costing 0.196 s. Three do gain on the board — LFPG +0.052 and LSZH +0.009 beside EHAM's +0.163
— so the honest sentence is *EHAM is the only exclusion that gains on all three folds and the
only one that passes the gate*, not "the only one that pays". And the table is **computed on
the same three folds that named EHAM**, so it cannot be independent evidence of generalisation:
it says the observed gain is concentrated at one airport rather than spread over all of them,
not that the concentration will repeat on unseen months.

**Read the intervals accordingly.** Arm 3's three intervals all exclude zero, and they are
**conditional on the airport already being chosen** — a nominal 95 % that is not adjusted for
having selected EHAM out of ten. They are quoted because they are the same instrument every
other row in this lane uses, not as a family-wise statement.

**What it costs to have chosen EHAM, said plainly.** The airport was named by exp-028 §5's
per-airport table, computed on these same three folds, so arm 3 is a selection on the folds
that score it and **this experiment puts no bound on that selection** — the same data cannot
both choose a slice and price the choice. exp-028 §6 made exactly this correction under Sol
round 1 (Spec 2) about the `nm_bulk` regime set, and it applies here word for word. What stands
outside these folds is the *reason*: `docs/shift-2026.md` calls EHAM the one airport January
2026 puts outside its own 2025 regime, with the lowest NM-join rate anywhere in the data, in
work that predates exp-028. Arm 3 is a candidate with an external rationale. Calling it a
*discovery* would need a fold this experiment did not touch, or the board.

**Where the gain sits on the board months.** The board is july_loo 55.7 % and january 44.3 %,
and arm 3 gains on both — most of it on january (+0.275), the fold whose distribution is
closest to the 2026 ranking months. The december gain (+0.086) is the first time in this family
that december has moved the right way with an interval excluding zero.

## 5. Arm 4: the arrivals keyed on runway and stand

`code/arrk.py`, 40 columns: four aggregates (count, taxi-in mean, taxi-in p90, arrival lateness
mean) over the departure's own 600 s bucket and the two either side, on two keys.

`RUNWAY_mvt` is 100 % populated on departures and arrivals in all three fold months and in the
ranking file. `STAND_mvt` is **not quite**: 3 / 1 / 7 departures and 9 / 4 / 0 arrivals are null
in july_loo / december / january, and 4 departures and 5 arrivals in the ranking file. Sol
round 1 caught the first version of this file claiming 100 % for both — a rounding of 0.99998 —
and, worse, mapping those nulls through `astype(str)` into the `N` stand block. A null stand now
takes its own `"?"` key; the arm was re-rotated after the fix and its board moved from −0.0276
to −0.0246, which is the size of a 24-row correction and changes no sign.

**The keys were chosen on coverage measured before any fit** (`analysis/arrk_keys.json`), not
on a fold number:

| key | July 2025 dep rows with a populated own bucket | ranking | median arrivals per non-empty cell |
|---|---|---|---|
| exact `RUNWAY_mvt` | 12.2 % | 8.7 % | 3 |
| **runway direction** (`27L` → `27`) | **50.7 %** | **51.2 %** | 4 |
| **stand block** (`B41` → `B`, `324` → `3`) | **58.1 %** | **55.3 %** | 1 |

An exact runway match empties the cell because six of the ten airports run segregated parallel
operations — arrivals land on one strip and departures leave from the other — so the key that
arrivals and departures share is the direction, which is the airport's configuration. Coverage
is within 3 points between the fold month and the ranking file on both chosen keys, which is
the fit/serve stability the block needs.

**The measurement.** The assignment asks for this block "added to the best of 1–3", and the
best of 1–3 is arm 3, so **arm 4 is the arrivals block with EHAM excluded** and the row above it
is its control — the same block with no exclusion, which is what isolates the block from the
mask. Added to v6's block inside the same stacker with no exclusion, the arrivals block is
**−0.0246 s of board against v6**, losing 0.064 on july_loo and 0.048 on december and gaining
0.039 on january. Against stack-5 it is +0.390 s, which is v6's own gain minus this block's
cost: it does not destroy the window, it just does not add to it. **Arm 4 proper**, with EHAM
excluded, is +0.1453 s against v6 — **0.018 s below arm 3, which is the same exclusion without
the block** — so the block's marginal contribution is negative under both masks and arm 4 is
not the shipping arm.

**What that closes and what it does not.** exp-028 §10's stated revival condition for the
arrivals block was "keying the arrivals on the runway and stand the ranking file also gives
rather than on the airport alone". That condition is now **met and the result is still
negative**, on the strongest keying the data supports. exp-028's airport-keyed `arr` block lost
0.309 s; re-keying recovers 0.28 s of that and stops 0.02 s short of neutral. The honest
reading is that the arrival's taxi-in is a measurement of the **inbound** surface and the
departure's queue is the outbound one, and sharing a runway direction or an apron block is not
enough to make them the same state. The repair this leaves is a different quantity, not a
different key: the arrivals' **stand occupancy at the departure's own stand** (a turnaround
feature, which needs a horizon longer than one window: the *stand block* cell already holds a
median of one arrival and an exact-stand cell is a refinement of it, so it can only be
sparser), or the departure queue itself — how many departures took off from this runway
direction in the last window, which is a departure-side count and belongs in `features.py`, not
here. Neither is this assignment's arm list and neither is run.

## 6. The four required slices

`analysis/slices.csv`, clean rows, per fold and per arm: `rmse`, `le3600`, `gt3600`, `no_nm`,
`no_nm_flight_id_only`, and beside them `noac` and `no_noac`. The no-NM mask is `COMMON.md`'s — every `_flt` column
null, read from the raw month via exp-028's `review1.no_nm_mask` — and the looser
`FLIGHT_ID_mvt`-only version is carried in the same file as `no_nm_flight_id_only` (added there
after Sol round 1, which found it only in the JSON records), and it is worth 0.4 to 0.8 s and
changes no sign (exp-028 Sol round 1, Standards 4).

Every slice is on the **clean** population, the same rows the point estimate and the intervals
use. January's clean row count is 153,704: the two unreached-rollover ids are excluded
(`prc.data.UNREACHED_ROLLOVER`), which is what makes its `rmse` 218.54 rather than the all-rows
338.08 — the mixed-population trap exp-028 fell into and corrected under the same review.

The shape is the same on every arm: the block moves the bulk (`le3600`) and costs a little on
the 0.2–0.3 % of rows above 3600 s, which is the tail lane's population. Arm 3's `noac` slice is
3407.3545 / 2010.4588 / 3509.4763 on the three folds — identical to stack-5's and to every
other arm's, because no arm touches a `noac` row.

## 7. Serving and verification

`serve.py --arm eham-out --name exp-030-eham-out --from submissions/candidates/exp-028-window.parquet --exclude-airports EHAM`,
1.4 s. The candidate is **v6's own bytes where the mask serves, and stack-5's everywhere
else** — nothing is refitted, so the served file is the fold-side statement applied to the
ranking rows with no second model in between.

| | exp-030-eham-out |
|---|---|
| rows | 344,841, the template's ids in the template's order |
| served from v6 | 306,271 |
| served from stack-5 | 38,570 = 38,182 EHAM + 388 noac |
| min / median / p99 / max | 60.0 / 954.52 / 2582.18 / 115,306.6 |
| values below the floor | 0 |
| non-finite | 0 |
| RMS move against stack-5 | 25.13 s, 12,655 rows over 60 s |
| SHA-256 | `67b70ee379e62a2a3bc9d91c9d771112baf3233efa3f74f4f95adbd7a3c03ef7` |

`code/verify.py exp-030-eham-out` (`analysis/verify.json`):

| check | result |
|---|---|
| id set and order equal `submitting.parquet` | yes |
| dtype | float64 |
| all finite | yes |
| values below `FLOOR` | 0 |
| SHA-256 equals the record's | yes |
| **rebuild from the record, max abs difference** | **0.0** |
| the 388 `noac` rows against stack-5, max abs difference | **0.0** |
| every row outside the mask against stack-5, max abs difference | **0.0** on 38,570 |
| verdict | **PASS** |

**What this 0 proves, and what it does not** (corrected after Sol round 1, blocker 2). exp-028
§8e had to say that its rebuild shared `block_for`, `ranking_blocks` and `arm` with `serve.py`,
so a 0 proved deterministic replay and not independence. A **derived** candidate shares no model
code: the record names two committed parquet files, their SHA-256 values, the gates table's
SHA-256, a regime set and an airport, and `verify.rebuild` is `np.where` over those bytes with
the row order taken from `submitting.parquet` rather than from `serve_stack5.ranking_frame`.
The first version of this paragraph called that "nothing shared but pandas" and said "both
source hashes are checked"; **neither was true** — the rebuild went through `ranking_frame` and
the shared regime-table cache, and only the v6 source hash was checked. Both are now fixed:
`verify.rebuild` checks the candidate, the base **and** the gates table against the record's
hashes before reading any of them, and takes the derived path without `ranking_frame`. What is
still shared is `prc.data`'s parquet reading, and what is still outside any gate is the *choice*
of EHAM, which is §4's business.

**The serving gate binds the arm to its bytes.** Sol round 1's blocker 1 found the first version
checking only the block specification and the excluded airport, so a command could have attached
arm 3's score and PASS to a different `--from` or `--base` file. The arm record now carries a
serving contract — the source and base candidates pinned by SHA-256, the served regimes and the
excluded airports — and `serve.py` refuses anything else. Five wrong commands were tried against
it (the 300 s candidate as the source, blend-5 as the base, `nm_bulk` alone as the regimes, no
exclusion, and the contract of a different arm) and all five are refused
(`logs/serve-gate-controls.log`). The correct command reproduces SHA-256 `67b70ee3…` exactly, so
none of this moved a byte of the candidate.

**The fold side and the served side are the same statement, with one inherited difference.**
On the folds the arm is `where(airport == EHAM, stack-5, v6)` over the rotation vectors; on the
ranking rows it is `where(regime is noac or airport is EHAM, stack-5, v6)` over the candidate
files. The two agree because v6's rotation already equals stack-5's on every `noac` row by
construction (`keep-noac` puts both on the anchor there), so naming `noac` in the mask adds
nothing on the fold side. The one difference is exp-028's: the served vector has passed
`prc.blend.apply_residual_guard`, which touched 780 rows when v6 was built, and the rotation
vectors have not. That difference is inside v6's own bytes and is common to the two sides of
this arm's comparison; it is stated because it means the fold numbers do not model the guard,
not because anything here changes it.

The maximum, 115,306.6 s, is one of the 388 `noac` rows and is stack-5's own value, inherited
unchanged from v3.

## 8. What the board should read

BOARD READING 6's currency rule applies directly: the 388 noac rows are held at stack-5's bytes
in v6 and in every arm here, so the noac discount and the +4.28 anchor are common to both and
cancel, and the prediction is **v6's measured board minus the clean gain over v6**.

| candidate | clean board | gain vs v6 | predicted board, at face |
|---|---|---|---|
| v6 (**measured**, 10:33 UTC) | 257.6230 | — | **263.6922** |
| **exp-030-eham-out** | 257.4598 | +0.1632 | **263.529** |

Quoted **at face**. BOARD READING 6 measured this family transferring at 1.68× face (0.414 s
clean became 0.694 s on the board), and if that coefficient holds here the reading would be
about 263.42. This file does not claim it: the transfer was measured once, on a gain spread
over every non-noac row, and arm 3's gain is a *withdrawal* of the block from one airport
rather than an extension of it — the two need not transfer alike. A board reading on this
candidate is the cheapest test of whether the 1.68× is a property of the family or of that one
submission, which is worth more than the 0.16 s itself.

## 9. Reproducing every number here

From the repo root. `data/raw` is needed by arm 4's block, the key-coverage table and the
serving path; arms 1–3 run from committed bytes plus exp-028's rotation.

```
PRC_THREADS=4 .venv/bin/python experiments/exp-030-window2/code/rotate.py --arm w600w300       --blocks depx+depx_w300            --win 600
PRC_THREADS=4 .venv/bin/python experiments/exp-030-window2/code/rotate.py --arm w600w300w1800  --blocks depx+depx_w300+depx_w1800 --win 600
PRC_THREADS=4 .venv/bin/python experiments/exp-030-window2/code/rotate.py --arm w600arrk       --blocks depx+arrk                 --win 600
.venv/bin/python experiments/exp-030-window2/code/measure.py w600w300      --blocks depx+depx_w300
.venv/bin/python experiments/exp-030-window2/code/measure.py w600w300w1800 --blocks depx+depx_w300+depx_w1800
.venv/bin/python experiments/exp-030-window2/code/measure.py w600arrk      --blocks depx+arrk    # arm 4's control
.venv/bin/python experiments/exp-030-window2/code/eham.py                       # arm 3
.venv/bin/python experiments/exp-030-window2/code/eham.py --on w600arrk         # arm 4 proper
.venv/bin/python experiments/exp-030-window2/code/airports.py                   # section 4
.venv/bin/python experiments/exp-030-window2/code/arrk.py                       # section 5's key table
.venv/bin/python experiments/exp-030-window2/serve.py --arm eham-out --name exp-030-eham-out \
    --from submissions/candidates/exp-028-window.parquet --exclude-airports EHAM
.venv/bin/python experiments/exp-030-window2/code/verify.py
```

The five commands the serving gate refuses are in `logs/serve-gate-controls.log`, and
`tests/block_identity.py` checks §3's two claims about the multi-width block.

Out-of-fold predictions in `COMMON.md`'s format are `oof/exp-028-stack5w-<arm>-{fold}.parquet`
— `MVT_ID_mvt`, `fold`, `y_true`, `y_pred` (the arm), `y_pred_<blocks>` (the same vector, named
by its block) and `y_pred_stack5` (the no-window rotation the arm is measured against). They
carry exp-028's naming because the script that writes them **is** exp-028's `stack5w.main`:
`code/rotate.py` rebinds `stack5w.block_for`, `WIN`, `TAG`, `VARIANTS` and `panel.EXP` and calls
it, so no part of the stacker, the anchor, the seeds, the bootstrap or the two gates is
reimplemented here.

## 10. What this closes, what it parks, and what it hands on

Nothing is closed. Three paths are parked with their revival conditions.

- **The width family (arms 1 and 2).** A second aggregate set is +0.073 s, a third is +0.167 s
  with december broken. Revived by a construction that gets the wide window's board without its
  december cost — the obvious candidate is the 1800 s set entered as *deviations from its own
  airport-month level* rather than as levels, which is the repair exp-028 §7a's deviation block
  botched by applying it to every width at once. That is one rotation and it is not this
  assignment's arm list.
- **The arrivals (arm 4).** exp-028 §10's revival condition is met and the answer is still
  negative (§5). The next quantity is not a re-key but a different measurement: same-stand
  turnaround over a longer horizon, or the departure-side count of take-offs on the same runway
  direction in the last window. Revived by either measuring above zero on its own.
- **The airport mask (arm 3).** Passed, shipped, and **selected on the scored folds**. Revived
  as a *discovery* rather than a candidate only by a fold this experiment did not touch or by
  the board. The board reading on this candidate is that test.
- **The `noac` rows are untouched** in every arm, as score-feedback rule requires. Nothing here says anything
  about them.

## 11. Reviewer

`reviewer: gpt-5.6-sol-xhigh-2026-09-07`, fresh context, read-only, one round, on the shipping
arm only (arm 3 and its candidate), as `assignments/window2.md` sets. Full report:
`review/sol-round1.log`.

**Verdict.** *"The shipping number and candidate bytes are correct. I found three blockers in
the serving and verification controls. None changes the reported score."* Sol recomputed the
arm independently and got 284.625076 / 205.567673 / 218.541163, board 257.459800, gain
0.1631697 over v6 and a predicted board of 263.529030 — every figure this file quotes — and
confirmed by direct inspection that the candidate has 344,841 template-ordered rows, that all
38,182 EHAM rows and all 388 `noac` rows exactly equal stack-5, that the other 306,271 exactly
equal v6, and that its SHA-256 is the recorded one.

Ten findings, three blocking. **Every one is accepted; none is disputed. None moved a number
in the shipping arm, and the candidate's SHA-256 is unchanged at `67b70ee3…` after all ten
repairs.** One nonblocking finding did move a *non-shipping* number: arm 4's stand key.

| # | finding | effect | what was done |
|---|---|---|---|
| **B1** | `serve.py`'s gate checked the block spec and the excluded airport but not `--from`, `--base` or `--regimes`, so a command could attach arm 3's score and PASS to different candidate bytes — contradicting the claim that unmeasured configurations are refused | claim | the arm record now carries a **serving contract**: the source and base candidates pinned by SHA-256, the served regimes, the excluded airports (`code/score.py:CANDIDATES`, `code/eham.py`). `serve.py` refuses anything else, and five wrong commands are shown refused in `logs/serve-gate-controls.log`. §7 |
| **B2** | the derived rebuild was called independent, but `verify.py` and `serve.py` both went through `serve_stack5.ranking_frame` and the shared gates cache, and only the v6 source hash was checked — so "nothing shared but pandas" and "both source hashes are checked" were false | claim | `verify.rebuild` now checks the candidate, the **base** and the **gates table** against the record's hashes before reading any of them, and the derived path takes its row order from `submitting.parquet` instead of `ranking_frame`. §7's paragraph is rewritten to say exactly what is still shared |
| **B3** | RESULTS pointed at a §12 that did not exist and the ledger had no shipping-arm row, so the required review-and-answer step was not met | claim | this section, and the shipping row in `docs/LEDGER.md` |
| N1 | "EHAM is the only exclusion that pays" is false — LFPG gains 0.052 s and LSZH 0.009 s | claim | corrected to "the only one that gains on all three folds and the only one that passes the gate", in §0 item 5 and §4, with the two other positives named |
| N2 | the leave-one-out table is itself computed on the folds that selected EHAM, so it rejects a uniform shrinkage story **on these folds** and proves neither generalisation nor causation; the intervals are conditional on the selection | claim | §4 rewritten: what the table settles and what it does not, and the intervals are now labelled conditional and unadjusted for having chosen one airport of ten. "not a discovery" is replaced by what would make it one |
| N3 | the arm list looks widened: RESULTS carries arm 4 and an arm 4b | neither | **respectfully re-framed rather than removed.** The assignment says arm 4 is the arrivals block "added to the best of 1–3", and the best of 1–3 **is** arm 3, so the assignment's arm 4 is the block *with* the EHAM exclusion — what the first draft called 4b. The other row is its control, the same block with no exclusion, which is the only way to say what the block itself is worth rather than what the mask is worth. §2 and §5 now label them "4" and "4 control"; no fifth arm was run |
| N4 | §0 said exp-028 §10's revival condition was "not met" while §5 correctly said it was met and the result was still negative | claim | §0 item 4 corrected to match §5 |
| N5 | `STAND_mvt` is legal but not 100 % populated — 4 ranking departures and 5 ranking arrivals are null, and `astype(str)` mapped them into the `N` stand block | **number** (arm 4 only) | the claim is corrected with the exact counts per frame (3 / 1 / 7 departures and 9 / 4 / 0 arrivals in the folds), a null stand now takes its own `"?"` key, and **arm 4 was re-rotated**: its board against v6 moved from −0.0276 to −0.0246 and arm 4 proper from +0.1429 to +0.1453. No sign changed and the shipping arm does not read either column |
| N6 | `no_nm_flight_id_only` was described as being in `slices.csv` and was only in the JSON | claim | added to `analysis/slices.csv` |
| N7 | `climb.log` logged gains, with no fold RMSE, no iteration and no gap-closed fraction | neither | reformatted to `COMMON.md`'s format by `code/score.py`, with exp-028 §3's corrected 14.62 s oracle as the gap-closed denominator; the superseded lines are kept verbatim in `logs/climb-superseded.log` and the rotation's own trace now goes to `logs/rotation-trace.log` |

**What the review did not change.** The construction of any arm, the shipping arm's fold
RMSEs, its board, its gate result, the candidate's bytes or its SHA-256. Sol's verified-pass
list also confirms independently that the point estimates and the day-clustered intervals use
the same clean rows including the two January exclusions, that the board weights are exactly
55.7132 % / 44.2868 %, that `ADEP_mvt` and arm 4's inputs are legal ranking columns with no
leakage, that the rotation's indirect channel is stated without a nested-validation overclaim,
and that the 263.529 face prediction is sound as a stated assumption with the 1.68× left
honestly unclaimed.
