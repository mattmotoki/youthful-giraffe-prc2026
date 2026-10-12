# exp-040-chain — RESULTS

Worker `chain`, tier hard, 2026-09-08. Assignment `assignments/q09.md` (Pro queue Q09);
the reading taken and the readings rejected are in `SPEC.md`. Laptop, 4 threads, no GPU
locally; every member fit ran on Kaggle CPU. A candidate exists and is **HELD, not
uploaded** (§9g). Step 6's `C2-cat` GPU session is running; GPU-h is reported in §10.

## 0. The answer

**The stand chain inside `L2-hub8725` is the first full-amplitude positive this lane has
produced, and it is not a candidate yet.**

Three things, in the order they matter.

**1. The chain is a large member gain, and the member reproduces bit-for-bit without it.**
The `none` arm lands on exp-025's same-environment Kaggle baseline at an offset of
**0.000000 on all three folds**, so the frame swap is a no-op when the arm is empty and the
gate is a gate. Adding `ts_gap_arr1..3` then moves the member by **+8.06 / +3.01 / +2.30 s**
(july_loo / december / january). exp-037's five derived columns add a further
**+0.105 / +0.057 / +0.080 s** on top.

**2. Through v10's full serving path it is worth +0.1275 s of all-row board, with the
interval clear of zero — and it is not shrinkage.** exp-038 priced 37 staged vectors and 36
lost at full amplitude; the positives there were all bounded-amplitude shrinkage with
fitted scales of 0.10–0.45. Here the fitted scale is **0.74 / 1.14 / 1.22** and the gain is
at full amplitude: **+0.1275 s [+0.0705, +0.1948]** for the arm that keeps score-feedback rule's `noac`
bytes. The alignment `A` is positive on all three folds (65.6 / 51.8 / 71.8), which is what
step 6 needs to authorise `C2-cat`.

**3. It does not clear the advance clause on the base it would ship on, and the file is
HELD as a MAYBE.** The clause is "all-row paired gain over v10 with **every fold interval clear of
zero** at full amplitude". The arm as the assignment states it reads july_loo +0.1360
[+0.0623, +0.2126], january +0.1194 [+0.0389, +0.2340], **december +0.0818 [−0.0690,
+0.2108]** — december crosses. §4b said why: `nonm` loses on all three folds. Holding
`nonm` at v10 and serving the chain member on `nm_bulk` and `nm_late` gives **+0.1393 s
[+0.0900, +0.2008]** against v10 with every fold interval clear (§4c), and the
leave-one-fold-out choice equals the in-sample one on all three folds.

**Two corrections from Sol round 1 stand in front of that number and neither is small
(§9).** The hold was named after §4b's slice table was read, so under `QUEUE.md`'s
gate-isolation rule it is **discovery evidence, not a passed gate**. And the candidate the
desk asked for sits on **v11**, not v10: against v11 the same update reads **+0.1325
[+0.0842, +0.1930] with december +0.0903 [−0.0208, +0.1863], crossing zero** — so what an
upload would buy over the file already on the board does not meet the clause. The file
exists and is verified — **`submissions/candidates/exp-040-chain.parquet`, SHA-256
`c97e27d7…`**, `v11 + (H' − v10)`, 266,686 rows moved at 9.23 s RMS and 78,155 exactly v11's
(§8) — and it is a **MAYBE with a measured size**, worth about +0.021 s of board at BOARD
READING 10's 0.16× transfer. **The desk holds it and does not upload** (recorded rule, §9g); december
is scored and gated with no board share, so an interval crossing zero there is quoted at
zero. **Both revivals named in §9g are now closed by measurement** — (a) on v12 in §12, (b)
by the pre-registered test in §11 — so the candidate has no open path back.

The gain is broad, not a handful of rows: 7 of the 8 masked airports improve on all three
folds, and it lands on the `nm_bulk` regime, which is 92 % of the rows.

## 1. Table one — the chain is the shipped `gap_arr1`, on every departure and not only on the noac rows

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/equality.py

`analysis/equality.csv`. exp-037 §0 measured `g_any == gap_arr1` and `g_prev2 == gap_arr2`
at 0 s on the 1,485 chained 2025 noac rows and the 386 chained ranking rows. Q09 is scored
on the ordinary rows, so the same comparison is made here on **every departure of all
fourteen months**, twelve training and two ranking.

| statement | rows | result |
|---|---:|---|
| `gap_arr1` (`prc/tail_features.py`) against `g_any` (exp-037 `code/chain.py`) | 2,413,541 defined in both | **max abs difference 0 s, 0 rows differ** |
| `gap_arr2` against `g_prev2` | 2,382,983 defined in both | max abs difference up to 153,962 s, **21 rows differ** (0.0009 %) |
| defined in `tail_features` and not in the chain | — | 7 rows (`gap_arr1`), 4 rows (`gap_arr2`) |
| defined in the chain and not in `tail_features` | — | 0 rows |
| departures with a null `STAND_mvt` | 2,429,888 total | 32 |

**The first row is the one the assignment rests on and it holds exactly.** So an arm that
adds `ts_gap_arr1..3` to frame v2 adds exactly the instrument exp-037 built, and the two
v2-frame members are the only places in stack-5 where that is not a no-op.

**The second row is a small correction to exp-037 §0, which measured this on the noac rows
alone.** The two constructions of the *second-previous* arrival disagree on 21 rows in 2.4
million. `_stand_signals` takes `p1 = shift(1)` inside a stand's arrivals sorted by on-block
time; `stand_chain` runs a second `merge_asof` with `allow_exact_matches=False`. When two
arrivals share the same on-block second at one stand the first keeps a tied row and the
second skips every tie. It changes nothing in this design — the `derived` arm carries the
raw gaps as `tail_features` computes them and uses exp-037's `g_prev2` only inside `slack2`
— but "`g_prev2` IS `gap_arr2`" is true to 0.0009 % of rows and not exactly.

The 7 rows defined only in `tail_features` are null-stand rows: `_stand_signals` keys on the
string `"nan"` and chains them together, `stand_chain` drops them. The arm uses
`tail_features`'s reading, because that is what the shipped v4 frame serves.

### 1b. The leakage audit the size of this gain demands

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/mask_invariance.py

`analysis/mask-invariance.json`. A +8 s member gain from three columns has to be shown not
to be reading a label, so the chain columns get the check
`exp-008-diversity/frames.py::verify_mask_invariance` makes for the v2 columns.

| month | departures | `ts_gap_arr1/2/3` built from the labelled month vs the ranking-masked one | departure block times in the masked frame |
|---|---:|---|---:|
| 2025-07 | 190,713 | **0 rows differ, max abs difference 0.0** | 0 of 190,713 |
| 2025-12 | 165,677 | **0 rows differ, max abs difference 0.0** | 0 of 165,677 |

and exp-037's `stand_chain` **refuses** a frame that still carries departure labels, which
is a structural guarantee rather than a measurement, demonstrated in the same run.

Why the columns are legal at all is older than this experiment: `docs/input-rule.md` makes
every ranking-file column a legal input, an **arrival's** `BLOCK_TIME_UTC_mvt` is present
(only a **departure's** is blanked), and `gap_arr1..3` have shipped inside `_EXP002_SIGNALS`
and every `prc.tail_model` head since exp-002. The chain covers 99.2 % of ranking
departures (151,525 of 152,719 in 2026-01; 190,904 of 192,122 in 2026-07), so it is
available at serve time on nearly every row it would score.

## 2. Table two — the reproduction gate

Three Kaggle CPU sessions, one per fold, three arms each; `logs/{fold}.log`,
`results/port-{fold}.json`. The gate is against exp-025's **same-environment** Kaggle
baselines (`experiments/exp-025-cloud/results/nested-l2-hub8725-base.json`), not against the
published workstation OOF: exp-025 §10.6 measured this member's machine term at
+0.096 / −0.003 / +0.471 s by fold, with a sign that changes, which is several times any
effect this experiment could read.

| fold | exp-025 Kaggle baseline | `none` arm | offset | wall |
|---|---:|---:|---:|---:|
| july_loo | 314.503838 | **314.503838** | **+0.000000** | 47–52 min an arm |
| december | 227.164888 | **227.164888** | **+0.000000** | 47–52 min an arm |
| january | 350.863607 | **350.863607** | **+0.000000** | 47–52 min an arm |

**Bit-identical on all three folds.** The member producer is
`experiments/exp-008-diversity/run.py` unedited; `code/run_arm.py` only swaps the module
named `frames`, and with an empty arm that swap changes nothing at all. Every number below
is therefore the column list and not the harness.

## 3. Table three — the matched pair, at the member

Same seeds (7), same 8,725 / 3,225 rounds, same eleven training months, same four threads,
same session. The only thing that differs between two rows is the column list.

| fold | `none` | `+ chain` (`ts_gap_arr1..3`) | gain | `+ derived` (exp-037's five more) | gain |
|---|---:|---:|---:|---:|---:|
| july_loo | 314.5038 | **306.4436** | **+8.060** | 306.3383 | +8.166 |
| december | 227.1649 | **224.1591** | **+3.006** | 224.1023 | +3.063 |
| january | 350.8636 | **348.5648** | **+2.299** | 348.4849 | +2.379 |

**Why the effect is this large, and why nobody saw it before.** Frame v2 carries **no
stand-history column at all** — the retro frame's own `stand_prev_arr_block_gap` starts at
v3 and its import of `gap_arr1..3` starts at v4 — so `L2-hub8725` is one of the two members
in stack-5 that is blind to the stand. exp-037 §7 identified exactly this and called it "a
small thing to try rather than a lane". It is worth 8 s of member on the largest fold.

**exp-037's five derived columns are worth +0.105 / +0.057 / +0.080 s on top of the raw
chain** (july_loo / december / january), in the same direction on all three folds. (Sol round
1, non-blocking: this line first said "+0.08 to +0.11 s", which is not what the arms read.) That is the answer to the assignment's repair
question in advance: the trees can use the raw gaps, and the transforms and counts add a
little more. Neither representation is the failure.

## 4. Table four — the serving path, gated, and the contract pricing

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/q09.py --arms none,chain,derived

Two gates, both exact reproductions `code/q09.py` recomputes rather than reads, and one
self-test:

| gate | what it rebuilds | july_loo | december | january | offset |
|---|---|---:|---:|---:|---|
| A | blend-5's hybrid NNLS anchor on the seven committed members | — | — | — | reproduces `blend-config-5.json` |
| F | **the v10 replay from the committed member frames upward** — exp-027's mixed anchor, the 600 s `depx` stacker, the `noac` hold, v7's EHAM mask and exp-036's eight-airport mask | 284.1572 | 204.9218 | 218.1506 | **0.0000 / 0.0000 / 0.0000** |
| identity | the **published** member routed through the arm's own code path | 284.1572 | 204.9218 | 218.1506 | **0.0 gain against v10 on every endpoint** |

`logs/q09-identity-selftest.log`. The identity self-test is the one that matters for
believing the arms: it puts the published `L2-hub8725` through `frames_for()`, `serve()` and
`price()` and gets exactly zero, so an arm's number is the member change and nothing else.

**Two serving variants, because v10's `noac` hold is member-dependent.** The assignment
expects the `noac` rows to "stay bit-identical under the hold". They do not: v10 holds them
at **blend-5's anchor**, and blend-5's anchor is a hybrid NNLS fit on the members, so
changing `L2-hub8725` moves them — measured RMS shift **126 / 212 / 121 s** on the 339 / 89 /
63 fold `noac` rows. So every arm is served twice: as the recipe states it, and with the
`noac` rows pinned to v10's own values (`d11-noac`), which is what a candidate must ship to
keep score-feedback rule's bytes. The pinned variant moves **0** `noac` rows, checked at tolerance 0.

**The table**, all-row paired MSE at the ranking month shares, day-block bootstrap:

| arm | vs | all-row board gain | 95 % | clean | held-out (LOO) amp | july_loo | december | january |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `none` | v10 | −0.0204 | [−0.0800, +0.0405] | −0.0075 | −0.0123 | +0.0058 | +0.0131 | −0.0482 |
| **`chain` d11-noac** | **v10** | **+0.1275** | **[+0.0705, +0.1948]** | **+0.1583** | **+0.1250** | **+0.1360** | +0.0818 | **+0.1194** |
| `derived` d11-noac | v10 | +0.1155 | [+0.0369, +0.2031] | +0.1483 | +0.1145 | +0.1293 | +0.1191 | +0.1019 |
| `chain` (served) | v10 | +0.2524 | [+0.0113, +0.5106] | +0.3086 | +0.1946 | +0.4608 | +0.0923 | +0.0341 |
| `derived` (served) | v10 | +0.2784 | [−0.0009, +0.5718] | +0.3442 | +0.2259 | +0.5075 | +0.1352 | +0.0383 |
| `chain` d11-noac | `none` d11-noac | +0.1483 | [+0.0973, +0.2101] | +0.1664 | +0.1461 | +0.1321 | +0.0726 | +0.1664 |

and the per-fold intervals the advance clause turns on, for the shipping variant:

| fold | `chain` d11-noac vs v10 | 95 % | clear of zero? |
|---|---:|---|---|
| july_loo | +0.1360 | [+0.0623, +0.2126] | **yes** |
| december | +0.0818 | [−0.0690, +0.2108] | **no** |
| january | +0.1194 | [+0.0389, +0.2340] | **yes** |

**The `none` arm prices the machine term at the final predictor: −0.0204 s [−0.0800,
+0.0405].** exp-025 measured this member's machine term at up to ±0.47 s at the member; a
weight of 0.154 inside a seven-member blend under a stacker reduces it to two hundredths of
a second with an interval across zero. That is why the `chain`-against-v10 and the
`chain`-against-`none` readings agree to about 0.02 s, and it is the answer to the question
the SPEC raised about pricing an arm fitted on a different machine.

**The alignment, which is what makes this different from every row in exp-038's table:**

| fold | A | B | fitted scale |
|---|---:|---:|---:|
| july_loo | 65.64 | 53.99 | 1.216 |
| december | 51.78 | 70.05 | 0.739 |
| january | 71.75 | 62.84 | 1.142 |

`A > 0` on all three folds, and the fitted scale sits at about 1 rather than exp-038's
0.10–0.45. exp-038 §0's positives were *shrinkage* — serving less of v10's own increment.
This is not: full amplitude is where the gain is, and the held-out amplitude (+0.1250)
almost equals it (+0.1275).

### 4b. Where the gain lands

`analysis/q09-slices.csv`, the paired SSE change of `chain` d11-noac against v10 (negative
is better than v10).

| regime | rows (jul) | changed | december | january | july_loo |
|---|---:|---:|---:|---:|---:|
| `nm_bulk` | 172,270 | 131,956 | **−5.18 M** | **−14.24 M** | **−14.20 M** |
| `nm_late` | 14,509 | 11,950 | −2.09 M | +0.82 M | −2.03 M |
| `nonm` | 3,595 | 2,640 | **+1.72 M** | **+1.03 M** | **+1.48 M** |
| `noac` | 339 | **0** | 0 | 0 | 0 |

| airport | december | january | july_loo |
|---|---:|---:|---:|
| EDDF, EHAM | 0 | 0 | 0 (held at v7 by exp-036's mask) |
| EDDM | −2.50 M | −2.40 M | −1.33 M |
| EGLL | −0.02 M | −3.53 M | −1.06 M |
| LEBL | −0.02 M | −0.39 M | −2.04 M |
| LEMD | −0.33 M | −0.15 M | −0.69 M |
| LFPG | −2.67 M | −1.52 M | −1.43 M |
| LIRF | −1.60 M | −0.52 M | −1.74 M |
| LSZH | −0.81 M | −0.74 M | −0.43 M |
| LTFM | **+2.39 M** | −3.14 M | −6.02 M |

**Seven of the eight masked airports improve on all three folds.** Two structural zeros are
worth stating: EDDF and EHAM are not in exp-036's eight, so they serve v7 and a member
refit cannot touch them whatever it does; and the `noac` rows are pinned, so Q09's clause
about separating the tail member's band-table rows from its learned rows reads **0 for
both** in the shipping variant — the fixed high-band table is untouched by construction, and
that is a fact about the recipe, not a measurement of the table.

**Two things go the wrong way, and they are the same two things.** `nonm` loses on all
three folds, and LTFM loses on december alone. Those are december's missing power: the fold
where the other seven airports gain least is the fold where LTFM's loss is not covered.

### 4c. The repair §4b names, measured: the regime hold

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/regime_gate.py

`analysis/regime-gate.csv`, `analysis/regime-gate.json`. All eight subsets of the three
regimes the arm moves (`noac` is pinned in this variant and is not a choice), scored as the
paired SSE change of the gated vector against v10 — negative is better.

| served on | july_loo | december | january | total |
|---|---:|---:|---:|---:|
| **`nm_bulk`+`nm_late`** | **−16.22 M** | **−7.27 M** | **−13.43 M** | **−36.92 M** |
| `nm_bulk` | −14.20 M | −5.18 M | −14.24 M | −33.62 M |
| all three (the arm itself) | −14.74 M | −5.55 M | −12.40 M | −32.69 M |
| `nm_bulk`+`nonm` | −12.72 M | −3.46 M | −13.21 M | −29.39 M |
| `nm_late` | −2.03 M | −2.09 M | +0.82 M | −3.30 M |
| nothing (v10) | 0 | 0 | 0 | 0 |
| `nm_late`+`nonm` | −0.54 M | −0.37 M | +1.84 M | +0.93 M |
| `nonm` | +1.48 M | +1.72 M | +1.03 M | +4.23 M |

**The selection is held out from the rows it scores.** The subset that serves a fold is
chosen on the paired SSE of the **other two folds only** — `align.scale_loo`'s rotation
applied to a discrete choice instead of a scalar — because choosing on all three and then
scoring on all three is the thing the queue's contract forbids. The in-sample choice is
printed beside it as the in-sample number it is.

| arm | picks | all-row board | 95 % | july_loo | december | january | every fold clear? |
|---|---|---:|---|---|---|---|---|
| **held out** | `nm_bulk+nm_late` on all three | **+0.1393** | **[+0.0900, +0.2008]** | +0.1497 [+0.0727, +0.2291] | **+0.1071 [+0.0025, +0.1983]** | +0.1293 [+0.0731, +0.2304] | **yes** |
| in sample | `nm_bulk+nm_late` | +0.1393 | [+0.0900, +0.2008] | same | same | same | yes |
| ungated | all three regimes | +0.1275 | [+0.0705, +0.1948] | +0.1360 [+0.0623, +0.2126] | +0.0818 [−0.0690, +0.2108] | +0.1194 [+0.0389, +0.2340] | no |

**December clears at the contract's block widths and crosses at seven days.** The contract
is a day-block bootstrap with a 3-day sensitivity; `analysis/block-width.json` runs 1, 3 and
7 for both arms.

| arm | fold | 1-day | 3-day | 7-day |
|---|---|---|---|---|
| held | july_loo | [+0.0727, +0.2291] | [+0.0726, +0.2248] | [+0.0546, +0.2241] |
| held | **december** | **[+0.0025, +0.1983]** | **[+0.0199, +0.1944]** | **[−0.0172, +0.1859]** |
| held | january | [+0.0731, +0.2304] | [+0.0721, +0.2419] | [+0.0730, +0.2519] |
| ungated | december | [−0.0690, +0.2108] | [−0.0518, +0.2143] | [−0.0877, +0.2016] |

So against v10 the clause's interval condition holds at 1 day and at 3 days, the widths the
contract names, and december's lower bound goes negative at 7. (Against **v11**, the base the
candidate sits on, december crosses at every width — §9a.) That is the same fact as the +0.0025 lower
bound seen from another side: december is the fold without the power to carry this on its
own, and widening the block to a week takes the power away. The held arm improves on the
ungated one at every width — the hold is not an artefact of the block width — but it does
not make december robust.

**The held-out and in-sample choices coincide on every fold**, which is the strongest thing
that can be said for a rule named after the table: the selection is not picking a different
subset per fold to chase the fold's own noise, it is picking the same bit three times. What
it does not do is remove the fact that the regime family was proposed after §4b was read.
The `nm_bulk`-alone arm §6 predicted is the second-best subset, not the best; adding
`nm_late` is worth a further −3.30 M of SSE, and december is where it earns most.

### 4d. The ranking side, gated

    .venv/bin/python experiments/exp-040-chain/code/serve_arm.py --arm published

`analysis/head-published.json`. The candidate needs v10's recipe rebuilt on the 344,841
ranking rows with the arm's twelve-month member, which is exp-027's `serve.py` on the
`k2-refit-bulk-eham-out` arm followed by exp-036's eight-airport mask. `code/serve_arm.py`
changes exactly two strings before that runs — `frames.OOF["L2-hub8725"]` and
`frames.RANKING["L2-hub8725"]` — and nothing else.

| gate | what it rebuilds | result |
|---|---|---|
| serving path | the **shipped** member through this exact path | intermediate `refit-bulk` sha `3e6f3681…` = exp-027's own; masked head sha **`ad962763…` = v10, max abs 0.0000000000, 0 rows moved** |

So a head from this path is the member change and nothing else, and one arm costs 24 s.

## 5. Verdict

**A MAYBE with a measured size and a file that exists.** Sol round 1 corrected the earlier
verdict on this line; §9 carries both findings and this section is written after them. The
candidate `exp-040-chain.parquet` is built, verified and NOT recommended for an upload on
this evidence: on the base it sits on (v11) december's interval crosses zero, and the regime
hold is a post-hoc family, which under `QUEUE.md`'s gate-isolation rule makes the number
discovery evidence rather than a passed gate. What is not in doubt is the member: the chain
is worth +8.06 / +3.01 / +2.30 s inside `L2-hub8725`, at full amplitude, with the reproduction
gate exact.

The paragraphs below are the pre-review reading, kept because §9 quotes them:

**A candidate, on the regime-held arm, with its selection channel stated.** The arm as the
assignment states it is a MAYBE: +0.1275 s all-row over v10 at full amplitude, on a broad
population, with `noac` and EHAM bytes preserved and the chain columns' leakage audit passed
(§1b — the columns, not the inherited serving path, which carries the lineage's known
indirect label path: §9c), but december's own fold interval crosses zero. Held to `nm_bulk` and `nm_late` — the two
regimes §4b measures as gaining, with `nonm` left at v10 — it reads **+0.1393 s [+0.0900,
+0.2008]** against v10 and every fold interval is clear (§4c) — which is the clause's
arithmetic, but not a passed gate: the family is post-hoc (§9b) and the file the desk
asked for sits on v11, where december crosses zero (§9a).

Two things qualify that and neither is hidden. The hold is **post-hoc**: the regime family
was named after §4b's slice table was read, not before, so the honest statement is that one
binary choice was made on the same three folds that score it. What limits the damage is
that the choice is one bit and that its held-out form is the same bit — the subset chosen on
the two folds a fold is not is `nm_bulk+nm_late` on every fold, so the leave-one-fold-out
number (+0.1393) and the in-sample number are the same object. And december's lower bound
is **+0.0025**: it clears zero, barely.

## 6. What would settle it, in descending order of what it would be worth

**1. ~~The regime gate the slices name.~~ DONE — §4c.** It moved december's interval off
zero, and the answer was `nm_bulk`+`nm_late`, not `nm_bulk` alone.

**2. `C2-cat`, the second and last v2-frame member.** Step 6's condition (`A > 0` on all
three folds) is met. It is armed and waiting on a GPU slot. A **paired** fold costs about
0.8 GPU-h, so all three folds are 2.4 GPU-h against the assignment's 2 GPU-h cap; there is
no same-environment baseline for this member in exp-025 and its path is CatBoost-on-GPU,
which is where exp-025 §10.7 located the whole machine term, so the `none` arm cannot be
replaced by the workstation file. One fold runs first and its cost is measured before more
is asked for.

**3. A seed bag on the member.** `L2-hub8725` is a one-seed fit (seed 7) inside a stacker
that bags five. December's interval is a power problem, and averaging seeds is the standard
way to shrink member noise. Three seeds on three folds for two arms is 18 CPU fits at about
50 minutes each — expensive, and it changes the member's identity, so it is a rebuild
decision rather than a repair.

**6. ~~`C2-cat`.~~ DONE — §10 and §11.** Three folds, 2.165 GPU-h, and the pre-registered
regime test on them fails: december is negative at the head though the member gains 2.0 s.

**4. ~~The ranking member and a candidate.~~ DONE — §4d, §8, §8a.** The ranking-side path
is built and gated on v10's own bytes, the twelve-month member fits landed (76.2 / 76.4 /
80.0 min), step 1's ranking gate against the committed `dd0e8710…` reads as the machine term
at ratio 1.026 of the fold side's, and the candidate is built and HELD (§9g).

**5. What the post-hoc hold would cost to remove.** The honest version of §4c is a regime
gate declared before the folds are read, on a rule with a reason rather than a table — for
example "serve the member only where its training population is dense", which `nonm` is not.
That is a pre-registration, so it costs a second measurement on data this experiment has
already used, and only a new member or a new fold would make it clean.

## 8. The candidate

    .venv/bin/python experiments/exp-040-chain/code/rank_gate.py
    .venv/bin/python experiments/exp-040-chain/code/serve_arm.py --arm chain
    .venv/bin/python experiments/exp-040-chain/code/candidate.py --arm chain

**`submissions/candidates/exp-040-chain.parquet`, SHA-256
`c97e27d759b46d608eff072232c8c05327708ebe0eb31d8c4204f4669401455b`.** Built on the desk's
decision of 2026-09-08, on the regime-held arm, with the selection channel stated in §4c and
§5 rather than hidden. Not uploaded; the desk uploads.

**The rule.** `p = v11 + (H' − v10)`, `H'` = the chain member's served head on `nm_bulk` and
`nm_late`, v10 elsewhere. The parentheses are part of the rule: `v11 + (H' − v10)` makes the
update exactly `0.0` wherever `H' = v10`, so those rows come back as v11's own bits;
`(v11 + H') − v10` is the same real number and not the same float and would move every held
row by an ulp. `code/candidate.py` rebuilds the file from its own three hashes with that
association and gets **0.0**.

### 8a. Step 1's second gate — the ranking member against the committed `dd0e8710…`

`analysis/rank-gate.json`. The offset is not zero and it is not unexplained: the committed
file was fitted on the workstation and these on Kaggle, and the shift has to be — and is —
the same size on the ranking rows as on the fold rows between the same two machines.

| side | shift against the committed member | RMSE offset |
|---|---:|---:|
| july_loo OOF | RMS 35.09 s, max 536 s | +0.096421 |
| december OOF | RMS 29.45 s, max 347 s | −0.003025 |
| january OOF | RMS 34.28 s, max 445 s | +0.470630 |
| **ranking, `none` arm** | **RMS 33.80 s, max 359 s** | — |

Ranking-side RMS 33.80 against a fold-side mean of 32.94: **ratio 1.026**. The chain
member's own increment over `none` on the ranking rows is RMS 79.41 s at the member, which
the stacker, the blend and the mask then reduce to 9.23 s at the head. All three ranking
columns are finite on all 344,841 rows with a minimum of 60.00 s.

### 8b. Verification

Every row of this table is recomputed by `code/candidate.py`, not read.

| check | result |
|---|---|
| template order against `data/raw/submitting.parquet` | 344,841 rows, in order |
| dtype, finiteness | float64, all finite |
| minimum | **60.4192** — v11's own minimum; the 60 s floor binds on **0** rows |
| maximum | 115,306.599 — v11's own maximum, unchanged |
| rows changed against v11 | **266,686**; RMS **9.2307 s**, max 252.82 s |
| rows identical to v11 | **78,155**, at max abs difference **0.0** |
| the identical set | **exactly** (`nonm` ∪ `noac`) ∪ EHAM ∪ EDDF — 0 held rows moved, 0 unheld rows stayed |
| `noac` (388) / EHAM (38,182) / EDDF (36,317) / `nonm` (4,895) | 0.0 each |
| serving-path gate | the shipped member through the same path returns v10, `ad962763…`, max abs 0.0 |
| arithmetic self-test | the same code on the gated published head returns **v11 byte for byte**, `42ca5ee6…` |
| rebuild from the record's three hashes | **0.0** |
| served update against the fold side's | **9.2307 s** against 8.11 / 9.23 / 8.60 (mean 8.649), ratio **1.067** — inside the one-third rule |

### 8c. Where the file moves, by regime and by airport

| regime | rows | changed | RMS |
|---|---:|---:|---:|
| `nm_bulk` | 323,468 | 253,964 | 7.88 |
| `nm_late` | 16,090 | 12,722 | 12.76 |
| `nonm` | 4,895 | **0** | 0.00 |
| `noac` | 388 | **0** | 0.00 |

| airport | rows | changed | RMS |
|---|---:|---:|---:|
| EDDF | 36,317 | **0** | 0.00 |
| EDDM | 25,966 | 25,728 | 6.81 |
| EGLL | 39,840 | 39,592 | 8.85 |
| EHAM | 38,182 | **0** | 0.00 |
| LEBL | 30,080 | 29,639 | 7.08 |
| LEMD | 36,954 | 36,718 | 6.55 |
| LFPG | 39,872 | 39,140 | 9.03 |
| LIRF | 26,899 | 26,516 | 14.17 |
| LSZH | 23,150 | 22,491 | 6.61 |
| LTFM | 47,581 | 46,862 | 10.89 |

### 8d. What the board should be expected to do

**Corrected by Sol round 1 (§9a).** The right fold reading for this file is its increment
over **v11**, the base it sits on: **+0.1325 s** all-row, not the +0.1393 against v10 that
§4c measured. v11's own increment over v10 the board has already read at −0.0166 s for a
fold-side +0.1033 (BOARD READING 10, transfer **0.16×**). At that transfer this candidate is
worth about **+0.021 s** over v11. The "+0.014 to +0.032" interval that stood here is
**withdrawn**: it was a fold interval multiplied by one transfer estimate, which carries
neither the transfer's own uncertainty nor its covariance with the fold gain, so it was not a
confidence interval. It is not a 2.1 s answer and is not offered as one.

Two things could make the realised value smaller than that. The update carries the machine
term — the member is a Kaggle fit and v10's was a workstation fit — which §4 prices at
−0.0204 s [−0.0800, +0.0405] at the final predictor, so it is small and, at its point
estimate, against us. And the regime hold is post-hoc (§5).

## 9. Sol round 1, and the pricing it corrected

`review/round1-gpt-5.6-sol-xhigh-2026-09-08.txt`, verdict **BLOCKING**, five findings.
`review/round0-killed-by-disk-…txt` is an earlier attempt that died when the laptop's data
volume reached zero bytes; it is not a verdict. All five findings are accepted. Nothing in
the candidate's bytes changed: it rebuilds to `c97e27d7…` after every repair below.

### 9a. Blocking 3 — the candidate was priced against the wrong base

**This is the finding that changes the verdict, and it is mine to own.** §4c priced `H'`
against **v10** and got +0.1393 with every fold clear. The file `code/candidate.py` writes
sits on **v11**, so the residual the update is evaluated against is v11's, and the alignment
and every interval move with it. `code/price_v11.py` rebuilds v11's fold analogue from the
same two committed sources the served file was built from —
`max(v10 + 0.217·(stack-5 − v10), 60)` — and prices both readings.

| reading | board | 95 % | july_loo | december | january | every fold clear? |
|---|---:|---|---|---|---|---|
| **increment: the candidate over v11** | **+0.1325** | [+0.0842, +0.1930] | +0.1476 [+0.0712, +0.2271] | **+0.0903 [−0.0208, +0.1863]** | +0.1175 [+0.0643, +0.2128] | **no** |
| composite: the candidate over v10 | +0.2358 | [+0.1510, +0.3358] | +0.2582 [+0.1142, +0.3874] | +0.2757 [+0.0008, +0.4995] | +0.2140 [+0.1229, +0.3729] | yes |
| for reference: v11 over v10 | +0.1033 | [+0.0380, +0.1716] | +0.1105 | +0.1854 | +0.0965 | yes (1-day) |

At 3-day blocks the increment's december reads [−0.0030, +0.1806] — still crossing.

**The instrument is checked, not asserted**: the third row reproduces exp-038's published
v11-over-v10 triple (+0.1105 / +0.1854 / +0.0965, board +0.1033, `docs/LEDGER.md` submission
v11) from this experiment's own packet, so the v11 fold analogue is the right vector.

**What this means, stated precisely (Sol round 2, blocking 1).** `assignments/q09.md`
line 60 writes the clause against **v10**, and the emitted file **passes that literal test**:
+0.2358 all-row with all three folds clear at 1 and 3 days. §9a does not change the written
clause and the earlier wording here, which said the clause fails, was wrong.

What is true is narrower and is still the reason to hold the file. **+0.2358 contains v11's
own increment**, which is already on the board and which BOARD READING 10 already priced at
0.16× transfer; what an upload would buy over the file we hold is the first row, **+0.1325
with december crossing zero**. The desk applied the every-fold condition to that increment
(2026-09-08 12:25 UTC, §9g) — december is scored and gated with no board share, so a crossing
interval there is quoted at zero — and held the file. Two further reasons stand on their own
and do not depend on which base the clause is read against: the regime family is post-hoc
(§9b) and the serving instrument carries the lineage's indirect label path (§9c), each of
which makes this discovery evidence rather than a passed gate. It is a MAYBE with a measured
size. The earlier "+0.022 s expected board value" in §8d was computed from +0.1393
against the wrong base; at +0.1325 and 0.16× it is about +0.021, and the interval quoted
there was a transfer-scaled fold interval, which is not a confidence interval and is
withdrawn.

### 9b. Blocking 1 — the post-hoc family, under the gate-isolation rule

`QUEUE.md` line 16: "A frozen new packet limits further selection but does not create
untouched data… Cached OOF screens are discovery evidence while the indirect label path
remains", and line 18: "Fit update scales and gates away from their scoring rows".
`code/regime_gate.py` holds the SUBSET choice out of the fold it scores, but it cannot hold
out the choice to look at regimes at all, which was made after §4b was read on these same
three folds. **So +0.1393 (and +0.1325) is discovery evidence, not a passed gate**, and §4c
and §5 now say so. The claim "the advance clause is met" is withdrawn wherever it appeared
unqualified. What would settle it is the fixed rule tested on data these three folds have
not touched — a new fold, or the C2-cat member of step 6, which has its own rotation.

### 9c. Blocking 2 — "the leakage audit passed" was too broad

The audit in §1b covers the **chain columns**, and Sol confirms it is sufficient for them:
they read arrival block times and legal departure fields, the cache crosses no fold boundary
and holds no label, and `stand_chain` refuses a labelled frame. What it does not cover is
the inherited serving path: `code/q09.py:118` fits the scoring fold's stacker on the other
two folds' member OOF, and those members were trained on month sets containing the scored
month, so the scored month's labels reach the weight vector one step removed. That path is
documented in `prc/data.py:165` and is v10's own; this experiment does not remove it and
does not make it worse. **The claim is now: the chain columns pass the leakage audit; the
serving instrument carries the lineage's known indirect label path, which is a second reason
this result is discovery evidence.**

### 9d. Blocking 4 — the writer now enforces what it recorded

`code/candidate.py` recorded `rank-gate.json` and `update_rms_within_third` and wrote the
file regardless. Both are gates now, and two more: the ranking-member gate's `explained`
flag and the committed member's SHA must both be right, and the head's record must name the
arm's own ranking file. The regime table was also unpinned and the "independent" rebuild
reused the in-memory mask; the record now carries the table's SHA-256 and a SHA-256 of the
served mask's bits, and the rebuild reloads the table and re-derives the mask from it.

### 9e. Blocking 5 — the `COMMON.md` columns, and one provenance defect

`analysis/common-slices.csv` adds RMSE overall, ≤ 3600 s, > 3600 s and no-NM for every
vector tables two, three and four name, per fold, plus the clean endpoint.

| vector | fold | overall | ≤ 3600 | > 3600 | no-NM (strict) | clean |
|---|---|---:|---:|---:|---:|---:|
| `none` (member) | july_loo | 314.5038 | 276.6536 | 2565.4522 | 1267.6233 | — |
| `chain` (member) | july_loo | 306.4436 | 271.5197 | 2437.5910 | 1215.2131 | — |
| `none` (member) | december | 227.1649 | 210.2670 | 2176.2975 | 820.4584 | — |
| `chain` (member) | december | 224.1591 | 206.9541 | 2179.7220 | 821.2497 | — |
| `none` (member) | january | 350.8636 | 217.9568 | 5983.4726 | 2864.3434 | — |
| `chain` (member) | january | 348.5648 | 213.0685 | 6002.9888 | 2861.8034 | — |
| v10 replay (head) | july_loo | 284.1572 | 247.0938 | 2405.2728 | 1119.8114 | 284.1572 |
| **chain held** (head) | july_loo | 284.0075 | 246.8929 | 2406.1099 | 1119.8161 | **284.0075** |
| v10 replay (head) | december | 204.9218 | 192.5142 | 1779.6943 | 660.3487 | 204.9218 |
| **chain held** (head) | december | 204.8147 | 192.3539 | 1782.8417 | 660.3911 | **204.8147** |
| v10 replay (head) | january | 337.8260 | 196.2393 | 5983.3709 | 2832.4212 | 218.1506 |
| **chain held** (head) | january | 337.6967 | 196.0022 | 5983.5852 | 2832.4045 | **217.9503** |

**The no-NM column is the STRICT one, and the first version of this table was wrong (Sol
round 2, blocking 2).** `COMMON.md` line 88 defines no-NM as **all `_flt` columns null**;
the first run used `regime == "nonm"`, which is a different and smaller set — **3,938 /
1,649 / 1,435** rows strict against 3,595 / 1,554 / 1,369 by regime, the regime set being a
strict subset. `code/common_slices.py` now uses `review1.no_nm_mask`, exp-028's corrected
mask and the one `exp-030/code/score.py:129` already used, and every no-NM figure above is
the strict one. The July member figure this changes most is `none`: **1267.6233**, not the
583.5352 first recorded.

**The claim "no-NM is unchanged at the head" was also wrong and is withdrawn.** Under the
strict mask it moves: v10 → held reads **1119.8114 → 1119.8161** (july_loo, worse by
0.0047), **660.3487 → 660.3911** (december, worse by 0.0424) and **2832.4212 → 2832.4045**
(january, better by 0.0167). The reason is exactly the difference between the two masks: the
`nonm` REGIME rows are held at v10 and do not move, but the 343 / 95 / 66 rows that are
strictly no-NM without carrying that regime label sit inside `nm_bulk` and `nm_late`, are
served, and do move. The movements are hundredths of a second on 1.5 % of rows; the point is
that "unchanged" was a claim about the wrong set.

The member gain is in the bulk and it is not bought out of the tail: `≤ 3600` improves on all
three folds (+5.13 / +3.31 / +4.89 s) while `> 3600` moves +128 s on july and −3.4 / −19.5 s
on december and january. At the head the tail is flat to slightly worse (−0.8 / −3.1 / −0.2 s)
against a bulk gain of +0.20 / +0.16 / +0.24 s — the head's tail rows are the member's tail
rows after a stacker that gives them little weight.

**The provenance defect — OPEN, and deliberately not repaired now (desk, 2026-09-08
12:25 UTC).** `results/port-july_loo.json` records commit `7319fc9`, which does not contain
`code/run_arm.py`; the file was first committed 92 seconds later at `9a7004b`. December,
january and the ranking session all record `9a7004b` and are reconstructable; july_loo is
not.

*What the 92 seconds contain, as a belief with its evidence.* I believe the code dataset
july_loo ran from held the **same** `run_arm.py`, `frames_chain.py` and `chain_cols.py` bytes
that `9a7004b` committed. The evidence: all three were ADDED at `9a7004b`, and between that
commit and the three fold sessions none was modified — `frames_chain.py` did change later, at
`dfaee5b` on 2026-09-08 in answer to Sol round 1, which is **after** every fit and does not
touch them (Sol round 2, non-blocking 1, correcting an over-broad "none has been modified"
here). So the only way the dataset could differ is if I edited them inside a 92-second window
that also contained a `push_code.sh` run and a commit; the
smoke test at 14:52 (`logs/smoke.log`) and the frame build at 14:51 ran against those files
and passed; and july_loo's `none` arm reproduces exp-025's Kaggle baseline at offset
**0.000000**, which the december and january sessions — built from `9a7004b` — also do. That
is consistent evidence, not proof: no hash of the uploaded tarball was recorded, so the
record cannot close it and I do not claim it does.

*The cost and where it sits.* Settling it is one Kaggle CPU session of about **2.5 hours**
re-running july_loo's three arms from the current commit. The desk's decision is not to spend
it now — the harvest needs every CPU slot and this candidate is a MAYBE — and to make that
re-run **the first step of any revival of this lane**. It stays open here until then.

### 9g. The desk's decision: HELD

**Desk, 2026-09-08 12:25 UTC.** Finding 3 stands. **December is scored and gated with no
board share** — the ranking months are july 0.55713 and january 0.44287, and december earns
nothing on the board but is one of the three folds the every-fold clause names, so an interval
crossing zero there is quoted at zero and the clause is not met. Against v11 the increment
reads december **+0.090 [−0.021, +0.186]**.

**The desk does not upload `exp-040-chain.parquet`.** It stays in `submissions/candidates/`
as a MAYBE (`internal decision record` recorded rule; ledger row *exp-040 candidate HELD*). The bytes stay
`c97e27d7…`.

**Revival, either of two.**

1. *The increment clear on every fold against the head the board actually holds.* This file
   is `v11 + (H' − v10)`. If a later head ships — k2climb's, for instance — the same `H' − v10`
   recomposed on it is a different residual and may clear december. That is arithmetic on
   committed vectors, no fit, and `code/price_v11.py` prices it as written.
2. *The regime hold tested on data unused in choosing it.* The family was chosen on these
   three folds. **`C2-cat`, step 6, is the second v2-frame member and rotates on its own**, so
   the fixed rule — serve the chain member on `nm_bulk`+`nm_late`, hold `nonm` — can be applied
   to it without being chosen again. That is the cleanest available test of the post-hoc
   family, and it is already running.

Either revival starts with the july_loo re-run of §9e, because a lane that ships cannot carry
a fold whose fit code the record does not pin.

### 9f. Non-blocking, both accepted

`code/frames_chain.py` keeps v2's schema and numeric fingerprints after appending columns and
its per-month cache carries no code or policy fingerprint — no label reaches it, but it
permits a stale feature to go unnoticed. `SCHEMA_FINGERPRINT` and `NUMERIC_FINGERPRINT` now
carry the arm, a `CODE_FINGERPRINT` hashes the four modules that decide the appended columns
(`frames_chain.py`, `chain_cols.py`, exp-037's `chain.py`, `prc/tail_features.py`), and the
per-month cache lives under that fingerprint, so a changed chain module builds a new file
instead of reusing the old one. **This change is prospective**: it does not touch the
committed OOFs, which were produced before it, and it is not a re-verification of them.

The kernels that have already run are deliberately **not** edited — their committed source is
the record of what ran, and rewriting it after the fact would make the provenance worse, not
better. The fingerprint above is what a future run records. And §3's "+0.08 to +0.11 s" for the derived columns is wrong: the
increments are **+0.105 / +0.057 / +0.080** s (july_loo / december / january), corrected in
place.

## 10. Step 6 — `C2-cat`, one fold, and the GPU hour it cost

`kernels/c2/`, `logs/c2-july_loo.log`, `results/port-c2-july_loo.json`,
`oof/c2-{none,chain}--july_loo.parquet`. The session took the GPU slot k2climb left free at
11:56 UTC and ended at 12:42; commit `9a7004b`, Tesla P100-16GB, peak 15,473 MiB.

**GPU-h: 0.714 for one fold and two arms** (`none` 1,346.6 s, `chain` 1,224.5 s). Step 6's
condition was met before it started — §4's `A > 0` on all three folds.

| arm | july_loo RMSE | against |
|---|---:|---|
| workstation pinned `C2-cat--C2-d8p` | 315.1682 | the shipped member |
| `none` (Kaggle GPU) | **315.4379** | **+0.2697 the machine term**, not a gate |
| `chain` (Kaggle GPU) | **301.5883** | **+13.8497 s against `none`** |

**The chain is worth +13.85 s inside `C2-cat` — 1.7× what it is worth inside
`L2-hub8725` (+8.06 s on the same fold).** Both are the retro-v2 frame, which carries no
stand-history column at all, and `C2-cat` is the deeper CatBoost of the two.

**The machine term is real here and is not what this measures.** exp-025 §10.7 located the
whole machine term of that experiment in CatBoost-on-GPU, and the `none` arm reproduces it:
+0.2697 s of RMSE and 27.17 s of per-row RMS against the pinned member. The chain moves the
same member by **77.12 s of per-row RMS**, 2.8× the machine's, so the effect is well clear of
the floor the machine sets. This is why there is a `none` arm at all: no workstation baseline
can serve as the control for a GPU refit.

**The other two folds followed** (desk, 2026-09-08 12:58 UTC, cap raised to 2.2 GPU-h), one
session at a time:

| fold | `none` | `chain` | member gain | machine term | GPU-h |
|---|---:|---:|---:|---:|---:|
| july_loo | 315.4379 | 301.5883 | **+13.8497** | +0.2697 | 0.714 |
| december | 223.1859 | 221.1882 | **+1.9976** | +0.2441 | 0.726 |
| january | 348.8052 | 347.9668 | **+0.8384** | +0.5515 | 0.725 |

**Total 2.165 GPU-h** against the 2.2 cap. The chain gains on every fold at the member, and
the fold spread is steep — july is 16× december. What that does at the head is §11, and it is
not what the member numbers suggest.

**It is also the cleaner half of revival (b).** §9g names `C2-cat` as the test of the
post-hoc regime family on data that did not choose it. §11 runs that test, on a rule
pre-registered before these folds landed, and it fails.

## 11. The regime hold, tested on a member that did not choose it

**PRE-REGISTERED.** `code/regime_test.py` fixes the rule in code — *serve the chain arm on
`nm_bulk` and `nm_late`, hold `nonm` and `noac` at v10* — with no argument, no search and no
enumeration, and was committed at **`4f0cb1a`** before `C2-cat`'s december and january arms
existed. The serving path was parameterised by member in the same state and already passes
both gates with `C2-cat` in the slot (gate F, the v10 replay, exact on all three folds), so
the arms drop in without another decision.

**Why this is a test and §4c was not.** §4c chose the subset by reading `L2-hub8725`'s slice
table on these three folds. The subset choice was held out of the fold that scored it; the
choice to look at regimes at all could not be, which is why §9b calls that number discovery
evidence. `C2-cat` is the other retro-v2 member with its own arms, so the same fixed rule
applies to it without being chosen again.

**The prediction, written before the data.** If the family is real rather than a reading of
`L2-hub8725`'s noise, then on `C2-cat`: `nm_bulk` gains, `nonm` loses, the held arm gains
over v10 on all three folds, and **december's interval clears zero** — the thing december
failed to do for the ungated arm (+0.0818 [−0.0690, +0.2108]). For comparison the same rule
on `L2-hub8725` reads +0.1497 / +0.1071 / +0.1293, board +0.1393.

A `nonm` that gains here, or a december that crosses, is evidence **against** the family, and
`regime_test.py` prints that verdict itself rather than leaving it to prose.

### 11a. The result: the prediction failed

`analysis/regime-test.json`. **`VERDICT: the family does NOT reproduce on a member that did
not choose it.`** The script printed that itself, on the rule it was committed with.

| arm | board | 95 % | july_loo | december | january | every fold clear? |
|---|---:|---|---|---|---|---|
| **held** (`nm_bulk`+`nm_late`) | **+0.0700** | [+0.0366, +0.1104] | +0.0854 [+0.0431, +0.1276] | **−0.0632 [−0.1790, +0.0517]** | +0.0543 [+0.0053, +0.1337] | **no** |
| ungated (all rows) | +0.0599 | [+0.0090, +0.1205] | +0.1020 | −0.1297 [−0.2946, +0.0250] | +0.0157 [−0.0564, +0.1224] | no |
| `nm_bulk` only | +0.0463 | [+0.0214, +0.0756] | +0.0515 | −0.0257 | +0.0411 | no |
| `nm_late` only | +0.0238 | [−0.0021, +0.0528] | +0.0340 | −0.0376 | +0.0132 | no |
| **`nonm` only** | **−0.0102** | [−0.0415, +0.0271] | +0.0166 | **−0.0665 [−0.1378, −0.0023]** | −0.0386 | no |

The prediction was that december would clear. **It does not clear and it is negative**:
−0.0632 at 1-day blocks and −0.1590 to +0.0301 at 3-day.

### 11b. What reproduced and what did not — the split matters

Two halves of the family were on trial and they came apart.

**The mechanism reproduced.** `nonm` is where the chain hurts, on a member that did not choose
that reading: `nonm` alone is board-negative (**−0.0102**), and on december it is negative and
clear of zero on the losing side at both widths (**−0.0665 [−0.1378, −0.0023]**, 3-day
[−0.1345, −0.0074]). And the hold does what it was built to do — the held arm beats the
ungated one, **+0.0700 against +0.0599**, and it rescues january from crossing (+0.0543
[+0.0053, +0.1337] against +0.0157 [−0.0564, +0.1224]). §4b's reading of *why* `nonm` should be
held is not what failed here.

**The sufficiency did not.** The claim that mattered for a candidate — hold `nonm` and every
fold clears — is false on `C2-cat`. December is not merely short of clearing, it is on the
wrong side of zero, and holding `nonm` moves it from −0.1297 to −0.0632 without changing the
sign. Whatever costs december at the head is not confined to the regime the family names.

### 11c. The member gains and the head disagree, and december is where

This is the sharpest thing in the section. `C2-cat`'s member improves on **all three** folds —
+13.85 / +2.00 / +0.84 s — yet at the head the same arm **loses** on december:

| | july_loo | december | january |
|---|---:|---:|---:|
| member RMSE, `none` → `chain` | 315.4379 → 301.5883 | 223.1859 → 221.1882 | 348.8052 → 347.9668 |
| member gain | **+13.8497** | **+1.9976** | **+0.8384** |
| head clean, v10 → held arm | 284.1572 → 284.0552 | 204.9218 → **205.0515** | 218.1506 → 218.0741 |
| head gain | +0.1020 | **−0.1297** | +0.0765 |

A member that is 2.0 s better on december produces a head that is 0.13 s worse. The stacker
and the blend decide how much of a member's gain survives, and on december they turn this
one's sign. That is not a contradiction — it is the reason `COMMON.md` insists a member number
is not a head number, and it is the same lesson §10 stated before these folds existed.

It also explains §4c without appeal to noise alone: `L2-hub8725`'s december head gain
(+0.1071) and `C2-cat`'s (−0.1297) are the same fold read through two members, and they
disagree in sign. December's head behaviour is member-specific, so a rule chosen on one
member's december does not transfer to another's.

### 11d. What this closes

Revival (b) is **closed by measurement**, on the test its own record pre-registered. With
revival (a) closed in §12, **both revivals named in §9g are now closed**, and
`exp-040-chain.parquet` is a MAYBE with no open path back — it stays in
`submissions/candidates/` as a record, not as a queue item.

What is **not** closed is the finding underneath it. The stand chain is worth +8.06 / +3.01 /
+2.30 s inside `L2-hub8725` and +13.85 / +2.00 / +0.84 s inside `C2-cat`, at full amplitude,
with the reproduction gate exact and the leakage audit clean on the columns. Those are member
facts and they are large. What this experiment could not do is get them through v10's serving
path to a head that clears every fold — and §11c says why the path is where the loss is.


## 12. Revival (a), measured against v12 — and it does not revive

`code/price_v12.py`, `analysis/price-v12.json`. §9g's first revival was "the increment clear
on every fold against the head the board actually holds… recomposed on k2climb's head if that
ships". **It shipped**: v12 = `exp-035-k2climb-v11member`, sha `48dd572d…`, board **262.7418**
(BOARD READING 11, tail rule). So the same question is asked of the same update — no fit, no new
member, `H' − v10` exactly as the candidate already adds it — against the new base.

**The v12 fold analogue is gated on exp-035's own published numbers before any exp-040 number
is read.** Built as §26a builds the served file (the climbed member through exp-027's
refit-bulk arm, exp-036's eight-airport mask with v7 elsewhere, then
`max(masked + 0.217·(stack-5 − masked), 60)`), it must return §26c's reading of v12 against
v11. It does, to four decimals:

| endpoint | exp-035 §26c | rebuilt here | offset |
|---|---:|---:|---:|
| all-row board | +0.3762 | +0.3762 | **+0.0000** |
| july_loo | +0.4809 | +0.4809 | **+0.0000** |
| december | +0.5379 | +0.5379 | **+0.0000** |
| january, clean | +0.3986 | +0.3986 | **−0.0000** |

**The answer.**

| base | board | 95 % | july_loo | december | january | every fold clear? |
|---|---:|---|---|---|---|---|
| over v10 (§4c) | +0.1393 | [+0.0900, +0.2008] | +0.1497 | +0.1071 | +0.1293 | yes |
| over v11 (§9a) | +0.1325 | [+0.0842, +0.1930] | +0.1476 | **+0.0903 [−0.0208, +0.1863]** | +0.1175 | no |
| **over v12** | **+0.1154** | **[+0.0686, +0.1729]** | +0.1300 [+0.0541, +0.2076] | **+0.0679 [−0.0426, +0.1630]** | +0.1008 [+0.0514, +0.1887] | **no** |

**Revival (a) fails, and it fails in a way that says something.** December crosses zero on the
new head too, at 1 day ([−0.0426, +0.1630]) and at 3 days ([−0.0266, +0.1604]), and it is
further from clearing than it was on v11, not closer. The update's whole value falls
monotonically as the head improves — **+0.1393 → +0.1325 → +0.1154** across v10, v11, v12 — and
december's share of it falls fastest, +0.1071 → +0.0903 → +0.0679, a 37 % loss against july's
13 %.

That is the signature of an increment that partly overlaps what the better head already
captures: v12's own gain over v11 is largest on december (+0.5379, against july's +0.4809),
which is exactly the fold where the chain update then has least left to add. The two changes
are not independent, and the fold where the chain most needed help is the fold the climbed
member helped most.

**No v12-based candidate file is built.** The desk's rule is to price every new candidate
against v12; priced there, this one does not meet the condition its own revival named, so
building the file would be recording a number, not making a candidate. `exp-040-chain.parquet`
stays as it is, HELD, on v11 (§9g), and revival (a) is now **closed by measurement** rather
than left open. Revival (b) — the pre-registered regime rule on `C2-cat`, §11 — is untouched by
this and is the one still live.

## 7. Files

    experiments/exp-040-chain/
      SPEC.md                     the reading of the assignment and the rejected readings
      RESULTS.md                  this file
      code/chain_cols.py          the arms' columns, and the equality instrument
      code/frames_chain.py        the v2 frame with the arm's columns appended
      code/run_arm.py             exp-008's run.py, unedited, on the arm's frame
      code/equality.py            table one
      code/mask_invariance.py     the leakage audit
      code/q09.py                 the serving path, its two gates and the contract pricing
      code/regime_gate.py         the regime hold, chosen out of the fold it scores
      code/serve_arm.py           the ranking side of one arm, gated on v10's bytes
      code/candidate.py           v11 + (H' - v10), and everything that gates it
      code/rank_gate.py           step 1's second gate, and the machine term on both sides
      code/price_v11.py           the candidate priced on the base it ships on (Sol r1)
      code/common_slices.py       COMMON.md's four columns, on the strict no-NM mask (Sol r2)
      code/regime_test.py         §11's pre-registered rule, fixed before the data
      code/price_v12.py           revival (a): the same update on the board's new head
      code/push_kernel.sh pull_kernel.sh watch_kernels.sh
      code/gpu_peers.py take_gpu_slot.sh   the GPU slot rule
      kernels/chain/ chain-*/     the CPU kernels, as generated and pushed
      kernels/c2/                 step 6's GPU kernel, and its landed outputs
      analysis/  equality.csv mask-invariance.json q09-table.csv q09-slices.csv
                 q09-arms.json regime-gate.{csv,json} block-width.json rank-gate.json
                 head-{published,chain}.json candidate-{published,chain}.json
                 price-v11.json common-slices.csv
      candidates/  the arm heads and the published self-test, never a submission
      review/    round0 (killed by the disk), round1, round2, and the two prompts
      oof/{arm}--{fold}.parquet   nine member out-of-fold files, COMMON.md's contract
      packet/q09-{fold}.parquet   every served vector, per row, for a later re-pricing
      logs/                       every kernel log and every local run

Nothing under `prc/` changed. The one shared-code edit is two directories added to
`experiments/exp-025-cloud/code/push_code.sh`'s tarball, so the Kaggle code dataset carries
`exp-037-tail/code` and `exp-040-chain/code`; no existing file in it changed.
