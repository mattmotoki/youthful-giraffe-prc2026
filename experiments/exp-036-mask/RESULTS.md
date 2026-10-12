# exp-036-mask — RESULTS

Three cheap cuts on what is already served, priced against **K2 (v9)** —
`submissions/candidates/exp-027-k2.parquet`, SHA-256 `a3a139db…`, clean **257.1653**, board
**263.0827** (`docs/LEDGER.md` BOARD READING 8). Read `SPEC.md` first: §1 states which reading
of "the change served or not, per airport" this experiment took and what it rejected.

Every number is computed by `code/` from committed bytes; arms 1 and 2 add `analysis/` and
nothing else, arm 3 also needs `data/raw`. Arms 1 and 2 run in 90 seconds at 4 threads and
refit nothing; arm 3's four rotations take 182 seconds. Reproduction commands in §8.

---

## 0. Summary

1. **The shipping arm is arm 2: the K2 mask on refit-bulk.** +0.1404 s of clean board over K2,
   folds **+0.162 / +0.294 / +0.108**, gate **PASS**. Predicted board **262.942** at face —
   *and the mask that produces it was selected on the same three folds it is scored on, a
   selection this experiment puts no bound on.*
   `submissions/candidates/exp-036-refitbulk-mask.parquet`, SHA-256 `ad962763…`, thirteen
   verification checks, rebuild difference 0.0. §4, §6.
2. **Say the caveat in the same breath: all three of arm 2's intervals against K2 include
   zero** — july_loo [−0.122, +0.516], december [−0.108, +0.858], january [−0.157, +0.349].
   The arm is a point estimate that clears the gate, on an instrument that cannot separate it
   from zero. What is clear of zero is its gain over **v7** (all three intervals exclude zero),
   which is exp-027's already-recorded MAYBE, not this experiment's contribution. §4.
3. **Most of arm 2 is not the mask.** refit-bulk with no mask is already +0.1051 s over K2 with
   all three folds positive; **the mask adds +0.0353 s** of that, and it costs 0.017 s on
   december. That is 0.1051 to 0.0353, **2.98 to 1** — 75 % anchor repair and 25 % mask.
   (The first version of this line said "four parts to one"; Sol round 1, blocker 2.) §4.
4. **Arm 1 is a clean negative and it is the informative one.** The K2-minus-v7 change helps on
   all three folds at **eight of the ten airports**; EHAM is an exact identity (v7 and K2 both
   hold it at stack-5, so the change is zero there), so EDDF is the only airport with a nonzero
   change that the rule removes, and holding EDDF at v7 is worth **+0.0049 s** of board, with december going −0.022. Gate FAIL on
   both clauses. There is essentially nothing to cut out of K2 — which is the opposite of what
   the same table said about v6, where one airport carried +0.163 s. §3.
5. **A second window width beside K2 is +0.0688 s over K2** (july_loo +0.104, december −0.010,
   january +0.013); gate FAIL on the gain and on the december sign. On the like-for-like form —
   no EHAM mask on either side, which is exp-030 arm 1's own form — it is **+0.0547 s** beside
   K2 against exp-030's **+0.0734 s** beside F1-fresh. The K2 member is itself a window model,
   so the question was whether it had already absorbed the second resolution; **this is no
   clear evidence of absorption, and it is not evidence of additivity either.** 0.019 s apart
   inside a 0.1025 s noise floor says the instrument cannot resolve the difference, not that
   there is none — and the no-mask fold effects are +0.097 / −0.001 / −0.014, two of them the
   wrong sign. §5.
6. **The selection cost is stated, not bounded.** The mask is chosen on the same three folds it
   is scored on, and this experiment puts no bound on that. What stands outside these folds is
   the reason EDDF is the airport that fails: `docs/shift-2026.md` puts EDDF's January 2026
   daily mean at the 0.95 percentile of its own 2025 distribution with **9 of 31 days above the
   whole 2025 maximum** — tied with EHAM for the most days outside the envelope, in work that
   predates exp-027. That is an input-distribution shift and it is *consistent with* the sign
   the folds show; it is not independent evidence of the sign. §3, §4.
7. **Five gates ran on arm 3's rotations and all held**: blend-5 reproduces `blend-config-5.json`;
   the no-window stacker lands on exp-026's published stack-5 at a maximum row difference of
   **0**; the 600 s arm lands on exp-028's published v6 at 1.1e-13; **the 600 s K2 arm
   reproduces the board's own K2 vector at 2.3e-13**; and the arm's first 50 block columns are
   exp-028's 600 s block name for name and value for value. `code/bases.py` re-gates v7, K2 and
   refit-bulk against their published rotations before any arm is scored. §1.

---

## 1. The bases, and the gates on them

Nothing here is refitted for arms 1 and 2. Four fold-side vectors are read back and each is
checked against the number its own experiment published before any arm is scored
(`code/bases.py`).

| | july_loo | december | january | clean board | measured board |
|---|---|---|---|---|---|
| stack-5 keep-noac | 285.2029 | 205.6118 | 219.1303 | 258.0372 | – |
| v6 (600 s window in the stacker) | 284.7219 | 205.6534 | 218.8166 | 257.6230 | 263.6922 |
| **v7** (v6, EHAM served from stack-5) | 284.6251 | 205.5677 | 218.5412 | 257.4598 | 263.4623 |
| **K2 = v9** (K2 in F1-fresh's slot, v7's mask) | 284.3195 | 205.2156 | 218.2582 | **257.1653** | **263.0827** |
| refit-bulk (the MAYBE: anchor refitted off the NM tail) | 284.1774 | 204.9052 | 218.2115 | 257.0601 | never uploaded |

Gates that raise rather than warn:

* `exp-030/code/score.reference()` re-reads exp-028's rotation and aborts unless stack-5 **and**
  v6 reproduce their published rows and boards within 1e-3.
* `bases.load()` aborts unless v7, K2 and refit-bulk reproduce the rows above within 1e-3 and
  their boards within 1e-3, and unless K2 and refit-bulk are **identical to v7 on every EHAM
  row** to 1e-9 — which is what makes EHAM unselectable by any rule here rather than something
  the prose has to promise.
* Arm 3 adds gates A–E of §5.

Board weights are exp-016's: july_loo 55.71 %, january 44.29 %. December is scored and gated and
carries no board share.

## 2. The four arms

Clean, rotated, against K2. `analysis/arm-*.json` carries each row in full.

| arm | july_loo | december | january | clean board | vs v6 | **vs K2** | predicted board | gate |
|---|---|---|---|---|---|---|---|---|
| K2 (the comparison point) | 284.3195 | 205.2156 | 218.2582 | 257.1653 | +0.4577 | — | 263.0827 measured | — |
| 1 — the K2 mask on K2 | 284.3193 | 205.2376 | 218.2457 | 257.1604 | +0.4626 | **+0.0049** | 263.078 | **FAIL** (gain, december) |
| 3 — K2 with 300 s beside 600 s | 284.2158 | 205.2253 | 218.2452 | 257.0965 | +0.5265 | **+0.0688** | 263.014 | **FAIL** (gain, december) |
| 2 control — refit-bulk, no mask | 284.1774 | **204.9052** | 218.2115 | 257.0601 | +0.5628 | **+0.1051** | 262.978 | pass |
| **2 — the same mask on refit-bulk** | **284.1572** | 204.9218 | **218.1506** | **257.0248** | **+0.5982** | **+0.1404** | **262.942** | **PASS** |

Per-fold gains over K2 with their day-clustered 95 % intervals, clean; positive is an
improvement on K2. The no-noac intervals are in `analysis/arm-*.json` under `ci_K2_no_noac`;
every bound moves outward by at most 0.07 s and no sign changes, because the `noac` rows are
identical between every arm and K2 and dropping them removes shared rows.

| arm | july_loo | december | january |
|---|---|---|---|
| 1 | +0.000 [−0.014, +0.016] | **−0.022 [−0.036, −0.011]** | +0.013 [−0.004, +0.033] |
| 3 | **+0.104 [+0.052, +0.163]** | −0.010 [−0.087, +0.060] | +0.013 [−0.040, +0.072] |
| 2 control | +0.142 [−0.144, +0.496] | +0.310 [−0.089, +0.874] | +0.047 [−0.229, +0.296] |
| **2** | +0.162 [−0.122, +0.516] | +0.294 [−0.108, +0.858] | +0.108 [−0.157, +0.349] |

**The gate field, named for the gate it tests.** `exp-030/code/score.score` sets `gate`,
`gate_gain` and `gate_folds` against **v6**, which is exp-030's comparison point and not this
one's; on arm 1 those inherited fields read `true` while the assignment's real gate fails. The
exp-032 correction row in `docs/LEDGER.md` was written because exactly that happened, so
`code/mask.py` renames them to `gate_*_v6_inherited` and the only field named `gate_*_K2` is the
one `assignments/mask.md` sets. Arms 1 and 3 carry `gate_v6_inherited: true` and
`gate_mask: false`, and `serve.py` refuses to serve on `gate_mask`.

**The gate, against the assignment's words.** `assignments/mask.md`: a shipping arm needs a
clean gain over K2 of **≥ 0.10 s** with **all three folds positive**. Arm 2 clears both. Arms 1
and 3 miss the gain and put december the wrong way. Arm 2's control also clears both, which is
why §4 splits arm 2 into the part the mask buys and the part the anchor repair buys.

**And the caveat that does not weaken with repetition** (exp-028 §7b, exp-030 §2): every arm
here is **a further cut on the same three folds K2 was scored on**. The mask chooses eight
airports out of ten on these folds and is then scored on them; arm 3 chooses a resolution on
them. A gate passed on the folds that selected the variant is exploratory and is not a second
confirmation. The intervals are the same instrument every other row in this lane uses, quoted
conditional on the airport set already being chosen and not adjusted for having selected it.

## 3. Arm 1: where the K2-minus-v7 change helps, airport by airport

`code/mask.py`, `analysis/airport_change.csv`. Serving K2's change at one airport against v7,
clean, positive is better. `heldout_*` is the same airport held at v7 inside the full K2 mask.
The clean SSE the airport contributes is the same number in both directions, so the two RMSE
columns carry the same **sign**; their magnitudes are not equal (the square root is concave) and
the largest disagreement in this table is **1.2e-4 s**. `mask.py` gates the signs, which is the
property the mask rule uses, and raises if any pair disagrees — the mask is only meaningful when
the change is separable.

| airport | july_loo | december | january | board alone | folds up | n (july) |
|---|---|---|---|---|---|---|
| LTFM | +0.118 [+0.018, +0.261] | +0.083 [−0.043, +0.275] | +0.054 [+0.006, +0.111] | **+0.093** | 3 | 25,171 |
| EGLL | +0.048 [+0.017, +0.086] | +0.098 [+0.040, +0.170] | +0.078 [+0.038, +0.124] | +0.059 | 3 | 20,790 |
| LFPG | +0.048 [+0.026, +0.072] | +0.042 [+0.012, +0.074] | +0.034 [−0.013, +0.091] | +0.042 | 3 | 21,712 |
| EDDM | +0.008 [−0.003, +0.020] | +0.024 [−0.004, +0.055] | +0.074 [+0.035, +0.122] | +0.033 | 3 | 15,825 |
| LEMD | +0.029 [+0.014, +0.045] | +0.022 [+0.008, +0.039] | +0.023 [+0.009, +0.041] | +0.026 | 3 | 18,818 |
| LEBL | +0.034 [+0.011, +0.064] | +0.019 [+0.005, +0.035] | +0.006 [−0.004, +0.015] | +0.023 | 3 | 16,752 |
| LSZH | +0.007 [−0.001, +0.016] | +0.005 [−0.005, +0.015] | +0.019 [+0.003, +0.041] | +0.012 | 3 | 12,611 |
| LIRF | +0.015 [−0.048, +0.069] | +0.038 [+0.001, +0.072] | +0.006 [−0.025, +0.038] | +0.011 | 3 | 15,211 |
| EHAM | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 22,489 |
| **EDDF** | **−0.000 [−0.016, +0.014]** | +0.022 [+0.011, +0.036] | **−0.013 [−0.033, +0.004]** | **−0.005** | 1 | 21,334 |

**The mask keeps eight airports and holds two at v7: EDDF and EHAM.** EHAM is not a choice —
v7 and K2 both hold it at stack-5's bytes, so the change is identically zero there (gated at
1e-9 in `bases.load`) — so EHAM does not "help" either, but as an exact identity rather than as
a loss. EDDF is the only airport with a nonzero change that the rule removes.

**What that costs, and why the arm fails.** Holding EDDF at v7 is worth **+0.0049 s** of board
and takes 0.022 s off december with an interval that excludes zero, because EDDF's december
change is positive and the mask throws it away. (Nine of the ten airports have a positive
december change; what is particular about EDDF is that it is the only **held** one — the first
version of this sentence called it "the one airport", Sol round 1, non-blocking 2.) The arm is a 0.005 s move on a rotation
whose noise floor is 0.1025 s: gate FAIL on the gain and on the fold sign, and the honest
sentence is that **K2 has nothing to cut**.

**Why this is the informative negative.** The same table on v6 (exp-030 §4) found one airport
carrying +0.163 s of board, seven of ten negative, and the worst at −0.196 s. The K2 change is
the opposite shape: eight of ten positive on all three folds, the worst airport at −0.005, and
the whole spread from +0.093 to −0.005 — a change that transfers almost uniformly across the
network. That is what exp-030's exclusion was buying against and it is no longer there. The
per-airport mask family, **with K2 as the source**, is out of material; the belief, its cause,
the repair attempted and the revival condition are the branch report in §9, and the status is
`closed-provisional` for that source only — the same rule on refit-bulk is the shipping arm.

**The reason EDDF is the airport, from outside these folds.** `docs/shift-2026.md`: EDDF's
January 2026 daily mean of take-off minus AOBT sits at the **0.95 percentile** of its own 2025
distribution, with 16 of 31 days above the 2025 p95 and **9 of 31 above the whole 2025
maximum** — the joint highest with EHAM — and its dispersion rises from 330 s to 375 s. So the
two airports the mask holds at v7 are the two `shift-2026.md` names as furthest outside the
2025 envelope, which is a reason that predates exp-027 and is not from these folds. It does not
make the +0.005 s large; it makes it the sign the reason predicts.

## 4. Arm 2: the same mask on refit-bulk, and what the mask actually buys

`code/mask.py`, arms `refitbulk-mask` and its control `refitbulk-nomask`. refit-bulk is
exp-027's repair of `k2-refit` — the anchor refitted with K2 **off the NM tail only**, blend-5's
`nm_late` weights kept — and it has never been on the board.

| | july_loo | december | january | clean board | vs K2 |
|---|---|---|---|---|---|
| K2 | 284.3195 | 205.2156 | 218.2582 | 257.1653 | — |
| refit-bulk, no mask | 284.1774 | 204.9052 | 218.2115 | 257.0601 | +0.1051 |
| **refit-bulk + the K2 mask** | 284.1572 | 204.9218 | 218.1506 | 257.0248 | **+0.1404** |
| *what the mask adds* | *+0.020* | *−0.017* | *+0.061* | | ***+0.0353*** |

**Three parts anchor repair, one part mask — 0.1051 to 0.0353, a ratio of 2.98 to 1, or 75 %
and 25 %.** (The first version of this paragraph said "four parts to one"; Sol round 1, blocker
2 computed the split.) The mask's own contribution is +0.0353 s of board, with july_loo
[−0.013, +0.057], december **−0.017** [−0.039, +0.004] and january +0.061 [+0.003, +0.153]. Only
january's interval excludes zero. The arm passes the gate because refit-bulk passes it before
the mask is applied. **And the +0.1404 carries the same caveat as every number here: the eight
airports were chosen on the three folds this row is scored on.**

**Why the mask transfers to refit-bulk at all.** Its own per-airport table
(`analysis/airport_change_refitbulk.csv`, a diagnostic — arm 2 is the assignment's words, *the
same mask*, and nothing here selects on this table) makes EDDF the **worst** of the ten for
refit-bulk too, at −0.035 s of board and negative on july_loo and january. Holding EDDF at v7
is worth exactly the +0.0353 s the decomposition above shows. So the transfer is **consistent
with the EDDF diagnosis**: the airport K2's change cannot help is the airport refit-bulk's
change hurts most, and `shift-2026.md` names that airport for a reason that predates both.
It is not proof that the transfer is not a coincidence — Sol round 1, blocker 5 — because the
refit-bulk table uses the **same labels and the same three folds** as the K2 one, and
`shift-2026.md` measures an input-distribution shift, not a model-performance sign. What would
support the stronger claim is an independent labelled period, a board reading, or a mask fixed
before refit-bulk was scored. Nothing in the score, the gate or the serving code rests on it.

**What arm 2's intervals do and do not say.** All three of its gains over K2 include zero,
because refit-bulk moves the anchor on every non-`nm_late` row and the day-clustered bootstrap
sees the variance of that. Against **v7** the same arm is +0.435 s with all three intervals
excluding zero (+0.468 [+0.099, +0.956], +0.646 [+0.126, +1.394], +0.391 [+0.080, +0.680]).
The distinction matters for what is claimed: refit-bulk is clearly better than v7 and is a
point estimate ahead of K2.

**The tail** (`analysis/slices.csv`, `gt3600`, the 0.2–0.3 % of rows above 3600 s). Against v7,
exp-027's second kill clause, arm 2 is **+1.69 / +19.21 / +0.54 s** — better on all three, so
the clause holds as it did for refit-bulk itself. Against **K2** the same slice is
**−0.97 / +14.40 / −1.17 s**: refit-bulk buys a very large december tail and gives back about a
second of tail on july_loo and january. That is the trade this arm makes and it is not visible
in the board number.

## 5. Arm 3: a second window width beside K2's

`code/k2w300.py`. Two aggregate sets side by side in one stacker — the 600 s block exactly as
v6 carries it, plus the same nine aggregates and four lags at 300 s with the five
width-independent row-context columns carried once — 95 block columns against 50. The block is
exp-030's `code/blocks.py`, unchanged, and the stacker, seeds, rounds, anchor and `keep-noac`
are exp-027's `code/candidate.py`, unchanged. Five gates, each a rotation recomputed in the run:

| gate | what it checks | result |
|---|---|---|
| A | blend-5 on the seven committed members reproduces `blend-config-5.json` | pass, 3 folds |
| B | the no-window stacker lands on exp-026's published stack-5 | **max row difference 0** |
| C | the 600 s arm lands on exp-028's published v6 | 1.14e-13 |
| **D** | **the 600 s K2 arm reproduces the board's own K2 vector** | **2.27e-13** |
| E | the arm's first 50 block columns are exp-028's 600 s block, names and values | pass |

Gate D is the one that binds this driver to the file the board holds: without it the arm would
be a two-width stacker measured against a K2 it does not reproduce.

**The result: +0.0688 s of board over K2**, july_loo +0.104 [+0.052, +0.163], december −0.010
[−0.087, +0.060], january +0.013 [−0.040, +0.072]. Gate FAIL — the gain misses 0.10 by 0.031 s
and december has the wrong sign, though its interval covers zero either way.

**The question, and how far the answer goes.** The K2 member is F1-fresh's frame *with 68
window columns added*, refitted on the real data. The obvious hypothesis was that it had already
taken whatever a second in-stacker resolution was worth. **This experiment finds no clear
evidence that it did — and no evidence that it did not.**

The comparison has to be made on the same population or it is not a comparison: exp-030 arm 1
and its base v6 both carry **every** airport, while arm 3 and K2 both hold EHAM at stack-5.
`analysis/arm-k2-w300-nomask.json` therefore prices arm 3 in exp-030's own form, from the
`y_pred_nomask` column of both rotations:

| | july_loo | december | january | clean board | second width worth |
|---|---|---|---|---|---|
| K2 at 600 s, no mask | 284.3938 | 205.2990 | 218.4232 | 257.2730 | — |
| K2 at 600 + 300 s, no mask | 284.2968 | 205.2998 | 218.4367 | 257.2183 | **+0.0547** |
| *exp-030: v6 → v6 with 300 s beside it (§2 of that file)* | | | | | *+0.0734* |

So the second width is worth **+0.0547 s beside K2 against +0.0734 s beside F1-fresh**, on the
same instrument, the same population and the same base convention. With the EHAM mask on both
sides the figure is the +0.0688 s the gate is applied to.

**What that does and does not license** (sharpened after Sol round 1, blocker 4). The two
figures are 0.019 s apart on a rotation whose noise floor is 0.1025 s. That says **this
instrument cannot resolve the difference between them** — it is not a demonstration that the
second width is worth the same beside both members, and it is not a demonstration that the
second width and the K2 member are additive. The fold detail argues for caution rather than
against it: the no-mask per-fold effects are **+0.097 / −0.001 / −0.014**, so two of the three
folds are the wrong sign and almost all of the board gain is july_loo. The claim this file makes
is the weak one: *no clear evidence of absorption*. The test that would settle it is a paired,
day-clustered difference-in-differences interval on (arm 3 − K2) against (exp-030 arm 1 − v6)
with a pre-declared equivalence margin, or a fold this experiment did not touch, or the board.
None of the three is run here.

What is not in doubt is the size: under the gate on v6 and under the gate on K2. Its named
revival condition is a stack where +0.07 s clears a gate — beside another passing arm rather
than alone.

## 6. Serving and verification

`.venv/bin/python experiments/exp-036-mask/serve.py --arm refitbulk-mask`, 0.4 s. **No model
runs**: the candidate is refit-bulk's own bytes at the eight kept airports and v7's bytes
everywhere else. The serving contract is read from the arm's measured record
(`analysis/arm-refitbulk-mask.json`) and not from the command line, and `serve.py` refuses an
arm whose `gate_mask` is false, so a command cannot attach this arm's score to another mask or
another source.

| | exp-036-refitbulk-mask |
|---|---|
| rows | 344,841, the template's ids in the template's order |
| served from refit-bulk | 270,342 |
| served from v7 | 74,499 = 38,182 EHAM + 36,317 EDDF |
| min / median / p99 / max | 60.0 / 954.45 / 2579.83 / 115,306.6 |
| values below the floor | 0 |
| non-finite | 0 |
| rows moved against v7 | 269,938, at 16.78 s RMS |
| SHA-256 | `ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78` |

The arm's rotation, its record, the serve and the verification were all re-run end to end after
the gate fields were renamed (§2), and the candidate came back at the **same SHA-256** with the
same thirteen PASSes — so the hash quoted here survives a full rebuild of everything upstream of
it, not just a second read of the file.

`code/verify.py --name exp-036-refitbulk-mask` (`analysis/verify-exp-036-refitbulk-mask.json`),
thirteen checks, all PASS:

| check | result |
|---|---|
| id set and order equal `submitting.parquet` | yes, 344,841 |
| dtype | float64 |
| all finite / below `FLOOR` | 0 / 0 |
| SHA-256 equals the record's | yes |
| the 388 `noac` rows against v7, and against stack-5 | **0.0** and **0.0** |
| the 38,182 EHAM rows against v7, and against stack-5 | **0.0** and **0.0** |
| **every row equals one of its two sources** | **0.0** on all 344,841 |
| held rows equal v7 / served rows equal refit-bulk | **0.0** / **0.0** |
| **rebuild from the record, max abs difference** | **0.0** |

**One inherited difference between the fold side and the served side**, exp-030 §7's and
unchanged here: both source candidates have passed `prc.blend.apply_residual_guard` on the
serving path, and the fold-side rotations they are compared on have not. That difference is
inside v7's and refit-bulk's own bytes and no arm here changes it. "Common" here means *present
on both sides*, not *cancelling*: Sol round 1 counted the guard touching **783 refit-bulk rows
and 780 v6 rows**, which are not the same rows, so nothing is subtracted away exactly. What
follows is only that the difference is inherited rather than introduced — it is stated because it means the fold numbers do not model
the guard, not because anything moved. The masking itself is the same statement on both sides:
`airport` on the fold side is the panel's column and on the served side is
`ranking-gates.parquet`'s, and neither side conditions on regime, because both source files
already hold `noac` at stack-5's anchor.

**What the 0.0 rebuild proves here.** `verify.rebuild` reads the record's two source files by
their recorded SHA-256, reads the airport column from the recorded gates table by its recorded
SHA-256, takes the row order from `submitting.parquet`, and is one `np.where` — it does not
import `serve.py`, and for a derived candidate there is no model, frame or block between the
bytes and the answer. This is exp-030 §7's corrected position and it applies word for word:
what is still shared is `prc.data`'s parquet reading, and what is outside every gate is the
*choice* of the eight airports, which is §3's business.

## 7. What the board should read

BOARD READING 8's currency rule applies directly: the 388 `noac` rows are held at stack-5's
bytes in K2 and in every arm here, so the noac discount and the anchor offset are common to both
and cancel. The prediction is **K2's measured board minus the clean gain over K2**.

| candidate | clean board | gain vs K2 | predicted board, at face |
|---|---|---|---|
| K2 = v9 (**measured**, 00:01 UTC 2026-09-08) | 257.1653 | — | **263.0827** |
| **exp-036-refitbulk-mask** | 257.0248 | +0.1404 | **262.942** |

Quoted **at face**. This family has transferred at 1.68×, 1.41× and 1.29× face on three
same-base readings (BOARD READINGS 6, 7, 8), a sequence that is falling; at 1.29× the reading
would be about 262.90. This file does not claim it. Either way the arm is a **0.14 s** move
against a **2.15 s** gap to `enthusiastic-daisy` at 260.931, so it is a cheap increment and not
an answer to BOARD READING 8's question.

## 8. Reproducing every number here

From the repo root.

```
PRC_THREADS=4 .venv/bin/python experiments/exp-036-mask/code/mask.py       # arms 1 and 2, 90 s
PRC_THREADS=4 .venv/bin/python experiments/exp-036-mask/code/k2w300.py     # arm 3, 182 s
.venv/bin/python experiments/exp-036-mask/code/nomask.py                     # section 5's like-for-like row
.venv/bin/python experiments/exp-036-mask/serve.py --arm refitbulk-mask
.venv/bin/python experiments/exp-036-mask/code/verify.py --name exp-036-refitbulk-mask
```

## 9. Branch report: the per-airport serving mask, with K2 as the source

`AGENTS.md` asks for a belief with its evidence rather than a close, so this is written as one.

**The negative and its cause.** Arm 1 is +0.0049 s of board with december at −0.022. The cause
is measured and it is not an implementation failure: the K2-minus-v7 change helps on all three
folds at eight of the ten airports and the whole per-airport spread is +0.093 to −0.005 s. A
serving mask can only remove airports, so on a source whose change is that uniform there is
nothing above the noise floor to remove. That is the opposite of v6, where the same table had
one airport worth +0.163 s and seven negative.

**Destination value and path feasibility, stated separately.** The *destination* — a
per-airport serving restriction — is worth up to 0.230 s of board, measured once (v7 over v6,
BOARD READING 7), so the destination is not disproved. The *path* — reaching it from K2 as the
source — is what is measured non-positive here at +0.0049 s.

**The honest repair attempted before the FAIL.** Arm 2 is the repair: the same rule applied to a
different source. It works — +0.0353 s from the mask alone on refit-bulk, and EDDF, the airport
K2 cannot help at, is also the airport refit-bulk hurts most at. So the rule is fine and the
source was the problem.

**The strongest refinement still visible**, and it is not run here (§10): refit-bulk's own
three-fold rule, which would keep three airports and hold seven at v7.

**What would revive it.** A source whose per-airport table is dispersed rather than uniform —
refit-bulk's already is, and any future base that changes the network unevenly will be. Status
`closed-provisional` for **K2 as the source only**; the family is live on refit-bulk and is the
shipping arm.

## 10. What this leaves

* **A second mask, chosen on refit-bulk's own table, is not run.** `assignments/mask.md` says
  arm 2 is *the same mask*, and widening is out of scope. The table is in
  `analysis/airport_change_refitbulk.csv`: refit-bulk's own three-fold rule would keep only
  **LTFM, LEMD and LSZH** and hold seven airports at v7. It costs seconds of compute and it is
  a strictly deeper selection on the same three folds, which is exactly the cost §2 says this
  experiment cannot bound. It is a decision for the desk, not a cut this worker takes.
* **Arm 3's +0.07 s is warm, not dead.** Twice measured, twice under the gate, additive with
  the K2 member. Its revival condition is a stack where 0.07 s clears a gate.
* **Nothing here answers BOARD READING 8's question.** The three arms together are worth at
  most 0.14 s against a 2.15 s gap. The gap is still the `noac` rows or an input-contract
  question, and neither is in this assignment.

## 11. The Sol round

One fresh-context pass by GPT-5.6 Sol at `xhigh`, read-only, on the shipping arm
(`review/round1-prompt.txt`, `review/round1-raw.log`). **Verdict: BLOCKED — six blocking
findings and five non-blocking.** Sol recomputed the shipping number independently and confirmed
it (284.1571957269 / 204.9218377546 / 218.1506169042, clean board 257.0248098797, gain
+0.1404420603 over K2's 257.1652519400, predicted 262.9422579397), confirmed that the fold-side
and served-side airport columns are the same `ADEP_mvt` with zero mismatches on all three folds
and all 344,841 ranking rows, and confirmed the served file row for row and by hash. Its words:
*"The shipping number and the current parquet are correct."*

Every finding is accepted. None of them moved a number; all six blockers were a defect in a
serving guard or a claim stated more strongly than the evidence supports.

| # | finding | answer |
|---|---|---|
| B1 | **`serve.py` could overwrite the shipping filename with an invalid control build.** The control arm's record stored the *string* `"all ten"` in `keep`, and `list("all ten")` is seven one-character "airports", so `--arm refitbulk-nomask --name exp-036-refitbulk-mask` selected **0** source rows, served v7 on all 344,841, and still carried the control's clean board and gain into the record | **Fixed.** `keep` is now type-checked (a non-empty list of strings), checked against the gates table's own airport universe, and the candidate's file name is bound to its arm, so `--name` cannot be detached from the bytes. `code/mask.py` no longer writes a string there. Five wrong commands were run against the fix and all five are refused (`logs/serve-gate-controls.log`); the correct command reproduces `ad962763…` exactly. **The shipping candidate's bytes never moved** — its `keep` was always the real eight-airport list |
| B2 | **"Four parts anchor repair to one part mask" is arithmetically wrong**: 0.1051 to 0.0353 is 2.98 to 1 | **Fixed**, §0 item 3 and §4: three parts to one, 75 % and 25 %, with the ratio quoted |
| B3 | **The separability claim overstated itself.** The SSE contribution is equal in both directions; the RMSE *magnitudes* are not, the square root being concave. Sol measured the largest disagreement at 1.2e-4 s, and the gate checks signs only | **Fixed** in `SPEC.md`, `code/mask.py` and §3: the claim is now "the same SSE contribution, therefore the same sign", the magnitudes are stated as unequal with Sol's 1.2e-4 s, and the gate is described as what it is. Sol confirms the fixed exclusions do not break the additivity |
| B4 | **Arm 3's "did not absorb" and "additive, not substitutes" are unsupported.** 0.019 s apart inside a 0.1025 s noise floor means the instrument cannot resolve the difference, not that there is none; two of the three no-mask folds have the opposite sign | **Fixed**, §0 item 5 and §5: the claim is now *no clear evidence of absorption, and none of additivity*, the per-fold signs (+0.097 / −0.001 / −0.014) are stated, and the test that would settle it — a paired day-clustered difference-in-differences with a pre-declared equivalence margin, an untouched fold, or the board — is named and not run |
| B5 | **"Not transferring a coincidence" is too strong.** The refit-bulk diagnostic uses the same labels and folds, and `shift-2026.md` shows an input shift, not a performance sign | **Fixed**, §4 and §0 item 6: *consistent with the EDDF diagnosis*, with what would support the stronger claim named. Nothing in the score, the gate or the serving code rests on it |
| B6 | **The selection caveat was disclosed but not beside every number**, which is what `assignments/mask.md` asks | **Fixed**: it is now in §0 item 1 beside the headline number and in §4 beside the +0.1404, as well as in §2 and §3 where it already was |
| NB1–NB5 | EHAM described as "not helping" when it is an exact identity; EDDF called "the one airport" with a positive december change when nine are positive and it is only the held one; the gate reads four-decimal gains; arm 3's runtime was 172 s and is 182 s; `nomask.py` read a column positionally and hard-coded exp-030's number | **All fixed** except the gate's rounding, which is left as it is and stated here: `code/mask.py` compares the recorded four-decimal gain against 0.10 and the recorded worst fold against 0. Arm 2 sits at 0.1404 and 0.1076, which is 0.04 and 0.11 from the two boundaries, so no rounding at the fourth decimal can move any verdict in this file. `nomask.py` now gates the row order and re-reads exp-030's figure from its committed record |

**What Sol asked for that this file does not claim to have.** Its last answer is that the
rebuild proves *the file matches its served record*, and does not by itself prove *the served
record matches the measured arm record*. That link is now carried by the serving guard rather
than by the rebuild, and the guard is the thing the five refused commands exercise; a check that
recomputes the arm's fold vectors from the served bytes would close it properly and is not run
here.
