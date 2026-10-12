# exp-037-tail — RESULTS

Worker `tail`, tier hard. Assignment `assignments/tail.md`, reading in `SPEC.md`.
Mac laptop, 2 threads (6 after the desk released the laptop at 08:04 UTC), no
GPU, no upload (score-feedback rule). Everything reruns with
`experiments/exp-037-tail/code/run_all.sh` from `data/raw` alone.

**This file was rewritten after each of three review rounds** — one fallback
round and two through the named reviewer (`review/NOTE.md`, §9). Every table
below is measured through the shipped serving path: the shipped categorical
encoding, the exp-009 band override above `BAND_MIN`, and the 60 s floor. Two
earlier versions used neither the encoding nor the serving path, and their
numbers are withdrawn wherever this file still quotes them. §9 lists every
finding and what each one moved.

## 0. The answer

**Verdict: a reading, not a candidate.** No arm has a fold gain that is positive
with an interval clear of zero, and the arms that move most break the movement
budget. No candidate parquet was written and the base stays v9.

Three things the experiment established, in the order they matter.

**1. The instrument is not new. The stand chain IS the shipped
`gap_arr1`/`gap_arr2`.** `g_any` equals `gap_arr1` at maximum absolute
difference **0 s** on all 1,485 chained 2025 noac rows and all 386 chained
ranking rows; `g_prev2` equals `gap_arr2` at **0 s** on all 1,460 and all 376.
`prc/tail_features.py::_stand_signals` keys on `airport|stand` and takes the
previous arrivals' on-block times — the same construction, written before this
experiment — and `gap_arr1..3` sit in `NOAC_FEATURES_V2`, in `_EXP002_SIGNALS`
and therefore in `FEATURES` and `FEATURES_NONM`. So the chain was already inside
every tail head. exp-033 §9 asked for an observable from **outside**
`NOAC_FEATURES_V2`; the chain is not one, and step 1 of the assignment should
have found that before the chain was built. What this experiment actually added
to the head is two counts (`n_dep_between`, `arr_stand_per_day`) and three
transforms of columns the head already had (`slack`, `slack2`, `log_g`).

**2. The bound is sound; the label is not physical.** `u = g − y = B_off − B_in`
is the stand's occupancy and cannot be negative on a same-aircraft chain. The
last arrival before take-off gives `u < 0` on **9.83 %** of 2,071,113 chained
2025 departures — the stand re-occupied while the aircraft taxied — and the
arrival *before* that one gives `u < 0` on **0.0110 %** of 2,044,957 chained
rows (**0.0076 %**, 156 of 2,043,497, counting ordinary rows only). Against that
bound, **33 of the 388 shipped predictions are physically impossible**. They
cannot be acted on, because on the noac rows the recorded label is not a taxi
time: **the same bound, at the same margin, is broken by the TRUE label** of a
2025 noac row on **6.78 %** of them and on **65.31 %** of the 8h+ band
(`analysis/step3_bound_safety.csv`; at zero margin, 4.66 % and 63.27 %).

**3. The sign is the opposite of the assignment's physical reading, and the
reason is a stale schedule, not a broken key.** A row whose placeholder anchor
the chain calls impossible is **more** likely to carry that anchor as its label
(+0.252 [+0.153, +0.347] in 1–2 h, +0.276 [+0.188, +0.365] in 2–4 h). The
statement that carries this uses no proxy at all: `slack < 0` is algebraically
`B_in > SCHED`, the previous arrival landing after the departure's own schedule,
and `P(ph | B_in > SCHED) = 0.678` (n = 637) against **0.334** (n = 848). A
placeholder is written when the inbound overtakes a stale schedule, so the
recorded block time predates the arrival and `u < 0` is a statement about the
*schedule*, not about two aircraft on one stand. Splitting the stand's arrivals
by airline prefix agrees — **92.0 %** of the arrivals around a 1–2 h placeholder
row's block time carry the departing flight's own prefix against a **19.2 %**
base rate at those stands — but that is an airline, not an airframe, and it does
not make the stand exclusive: another airline arrives during the taxi on
**12.2 %** of 2–4 h placeholder rows against 4.9 % of the ordinary ones, and on
**23.0 %** against 6.7 % at 4–8 h. So both things happen, with a gradient between
them: below 4 h the rotation reading dominates, and above 4 h the stand is
genuinely re-used (57.9 % of 8h+ placeholder rows). The first version of this
file claimed an exclusive key and an exact-aircraft chain; both are withdrawn
(§3b).

**The gate, both halves.** Sixteen arms — thirteen carriers over two heads and
three blends — on the 491 fold noac rows: the best pooled gain is
**+7.58 s [−27.28, +40.20]** (the classifier change alone, chain head), every
Holm-adjusted p is **1.0**, and the only two arms under the 500 s budget on the
388 (**320.0 s** and **439.0 s** served) are among the ones with no measurable
gain, while every arm that moves more breaks it (**570.4**, **849.0**, **538.6**,
**654.0**, **1,082.4 s**). **Not a candidate.**

**And the same clip on ORDINARY rows is measured harmful**, which the first two
versions of this file only argued about: scored out of fold on the shipped
vector with the margin fitted on each fold's training months, it loses on all
three folds at both margins — **−39.66 s** and **−45.68 s** on july_loo, −0.44
and −6.39 on december, −0.73 and −4.86 on january, every interval clear of zero
on the wrong side (§7). The rows it moves are the rows with genuinely long
taxis: on the July rows moved, RMSE goes from 300 s to 5,087 s.

## 1. What was built, and the population

`code/chain.py` builds the chain from a RANKING-MASKED frame and raises if it is
handed departure labels. `code/build.py` puts it on every departure of the ten
ranking airports; `code/frames.py` puts it on the noac rows with every
`tail_signals` column. The noac frames reproduce the record's population exactly
— **1,507** rows in 2025 and **388** in the ranking file, the same `MVT_ID_mvt`
sets as `experiments/exp-033-noac/work/`, asserted at the end of `frames.py`.
`code/checks.py` goes further and compares the **labels**: `y_true` and `ph` per
row against exp-033's own extraction, which read the twelve months through a
different code path — **maximum absolute difference 0.0 and 0 class
disagreements on all 1,507 rows** (`analysis/checks.csv`). That is also the
check on the row-order label join inside `frames.py`, which resets the index: a
misalignment there would move every label and this comparison would not survive
it. `frames.py` also asserts the join positionally on `MVT_ID_mvt`.

**The chain is the shipped feature, and this is the first thing the record
should say.** Measured, not argued:

| pair | scope | rows compared | max abs difference |
|---|---|---|---|
| `g_any` vs `gap_arr1` | 2025 noac | 1,485 | **0 s** |
| `g_prev2` vs `gap_arr2` | 2025 noac | 1,460 | **0 s** |
| `g_any` vs `gap_arr1` | the 388 | 386 | **0 s** |
| `g_prev2` vs `gap_arr2` | the 388 | 376 | **0 s** |

(`gap_arr1` is finite on 3 more 2025 rows than `g_any`; nothing else differs.)
`gap_arr1`, `gap_arr2` and `gap_arr3` are named in `NOAC_FEATURES_V2` and in
`_EXP002_SIGNALS`, so they are in the noac head, the NM head and the no-NM head
already. They are not everywhere, though: two of stack-5's seven members
(`C2-cat`, `L2-hub8725`) are built on the retro frame v2, which has no
stand-chain column, and `prc/nonm.py`'s `FEATURES` has none either — see §7.

The key is `(airport, stand)`: `ADEP_mvt` for a departure, `ADES_mvt` for an
arrival, and the raw `STAND_mvt`. `STAND_mvt` is null on 0.005 % of 2025-07
arrivals and 0.002 % of its departures. An arrival's time is its in-block
`BLOCK_TIME_UTC_mvt`, which is never blanked; a neighbouring departure's is its
NM off-block `AOBT_3_flt`, the only legal off-block for a row that is not the one
being predicted. Months never touch, in 2025 or in the ranking file.

**The chain is checked against a brute-force implementation**
(`code/test_chain.py`): a six-movement synthetic frame with every time written
out by hand, and one real airport-day (LSZH, 2025-07-03, 459 departures) where
`b_in`, `b_in2` and `n_dep_between` are recomputed by an O(n²) loop. **0
mismatches of 459 on each of the three columns.** The two things that were wrong
in a first version are exactly what this covers: the self-match (a departure's
own `AOBT_3_flt` precedes its own take-off, so a naive backward join returns the
row itself) and the fresh index `merge_asof` returns.

**Every noac row is also `nonm`** — 0 of the 1,507 and 0 of the 388 have an NM
off-block.

## 2. Step 1 — does the chain exist, how wide is it, and how pure is it?

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/build.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/step1.py

`analysis/step1_purity.csv`, `analysis/step1_by_airport.csv`.

| variant | scope | chained | median gap | `u < 0` | `u` at 0.1 % | `u` at 1 % |
|---|---|---|---|---|---|---|
| last arrival before take-off | all 2,085,047 | **99.33 %** | 1.68 h | **9.83 %** | −2,091 s | −1,204 s |
| the arrival before that | all | 98.08 % | 6.16 h | **0.0110 %** | **+2,281 s** | +3,127 s |
| last arrival before take-off | 1,507 noac | 98.54 % | 2.07 h | **35.56 %** | −82,259 s | −49,801 s |
| the arrival before that | 1,507 noac | 96.88 % | 6.25 h | **4.66 %** | −69,913 s | −27,938 s |

The `u < 0` column counts every chained row of its scope. Counting ordinary
(non-noac) rows only, the second-previous chain's rate is **0.0076 %** — 156 of
2,043,497, one in thirteen thousand — because 68 of the 224 violations are noac
rows, which are 0.07 % of the population.

**`u < 0` is the bound's false-positive rate, and on the ordinary rows it is a
lower bound on the chain's impurity. On the noac rows it is neither.** §3b shows
that the crowding arrival there is overwhelmingly a same-airline rotation rather
than a second aircraft on the stand, so the 35.56 % and 4.66 % in the last two
rows above are not a broken join: they are the recorded label exceeding a
physical bound, because on those rows the recorded label is not a taxi time.
(§3b does not establish that it is the same **airframe**, and it does not make
the stand exclusive; what it establishes is that the placeholder anchor is a
stale schedule the inbound overtook.) The two halves of this table are measuring different things and the word
"impurity" belongs only to the first.

**The simple chain's impurity is the taxi itself.** A departure whose taxi is
long frees its stand at push-back, another aircraft can arrive before it takes
off, and the chain then points at that aircraft. That is the whole of the 9.83 %,
and it is why the second-previous arrival — which survives one such
re-occupation — is a near-hard bound.

**The noac rows break both chains far more often than ordinary rows do**, and
monotonically in the band: under the last-arrival chain 24.1 % in 1–2 h, 46.4 %
in 2–4 h, 57.3 % in 4–8 h, **91.8 %** in 8h+; under the second-previous chain
0.68 %, 3.46 %, 15.07 %, **63.27 %**. On the median 8h+ noac row the recorded
label exceeds the gap by ten hours.

Per airport, the last-arrival chain's impurity runs from **LEMD 3.92 %** and
**LIRF 4.40 %** to **EGLL 26.32 %**; LIRF, which is 383 of the 388, is the second
purest.

## 3. Step 3 — the placeholder question, and the sign the assignment did not expect

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/step3.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/mechanism.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/rotation_check.py

`analysis/step3_conditional.csv`, `analysis/step3_bound_safety.csv`,
`analysis/mechanism.csv`, `analysis/mechanism_matched.csv`,
`analysis/rotation_reading.csv`, `analysis/rotation_legal.csv`.

`excluded` is the label-free indicator `mvt_sched > bound`: the placeholder
anchor `y = mvt_sched` is above what the chain allows. Against it, the record's
class `ph = |mvt_sched − y| < 300 s` (`prc.tail_model.PLACEHOLDER_TOL`).
Intervals are calendar-day-clustered bootstraps: 4,000 replicates for every
table in this file except §3b's exposure intervals, which `code/mechanism.py`
draws at 2,000.

**The chain separates the class, and below 4 h it separates it the wrong way
round.** Every band with both sides populated, so the reversal above 4 h is in
the table and not only in the text:

| band | n | n excluded | P(ph \| excluded) | P(ph \| rest) | difference | day-clustered |
|---|---|---|---|---|---|---|
| 30–60 m | 36 | 2 | 0.000 | 0.412 | −0.412 | [−0.583, −0.250] |
| 1–2 h | 752 | 129 | 0.581 | 0.329 | **+0.252** | [+0.153, +0.347] |
| 2–4 h | 496 | 251 | 0.725 | 0.449 | **+0.276** | [+0.188, +0.365] |
| 4–8 h | 75 | 53 | 0.774 | 0.864 | −0.090 | [−0.266, +0.082] |
| 8h+ | 49 | 45 | 0.756 | 1.000 | **−0.244** | [−0.362, −0.130] |
| all | 1,485 | 480 | 0.692 | 0.381 | **+0.311** | [+0.248, +0.371] |

(Last-arrival chain at the 1 % margin. The 8h+ and 30–60 m rows rest on 4 and 2
rows on one side; a day-clustered interval on 4 rows is an arithmetic object,
not evidence. The reversal above 4 h is nonetheless the sign §3b predicts, which
is why it is here.) The second-previous chain, which excludes far fewer rows,
gives the same sign below 4 h and weaker evidence: 1–2 h +0.063 [−0.093, +0.228]
and 2–4 h +0.090 [−0.034, +0.208] at the 1 % margin, both crossing zero; at the
0.1 % margin 2–4 h is +0.137 [+0.007, +0.259]. Pooled it is +0.161
[+0.070, +0.244] at the 1 % margin and +0.179 [+0.079, +0.272] at 0.1 %.

**The false-clip rate says the same thing.** `analysis/step3_bound_safety.csv`:
the fraction of 2025 noac rows whose TRUE label breaks the second-previous bound
at the 0.1 % level is 1.6 % in 1–2 h and 6.9 % in 2–4 h overall, but **4.0 % and
11.5 % among the placeholder rows** against 0.2 % and 0.5 % among the rest. A
clip removes placeholder rows, which is the opposite of what it was for.

### 3b. Why the sign is what it is — a same-airline inbound overtaking the schedule

The exposure test counts arrivals at the row's own stand in a window, against
`length × λ` where `λ` is that stand's own arrival rate in that month.

Two readings were tried and **both are withdrawn**. The first used the full taxi
length on each side of the block time and appeared to say the placeholder block
times are real early off-blocks (1.485 [1.388, 1.593] against 1.094 for the
rest); its before-window is as long as the taxi, so on a placeholder row it
reaches back past the aircraft's own arrival and counts it. The length-matched
form removed that confounder and gave

| w | class | after the block time | before the block time |
|---|---|---|---|
| 900 s | placeholder (722) | 2.138 [1.687, 2.637] | **1.765 [1.319, 2.284]** |
| 900 s | rest (770) | 0.720 [0.509, 0.948] | **0.167 [0.063, 0.303]** |
| 1,800 s | placeholder | 2.211 [1.873, 2.585] | **1.569 [1.273, 1.893]** |
| 1,800 s | rest | 0.959 [0.737, 1.175] | **0.146 [0.074, 0.232]** |

from which the first version of this file concluded that the `(airport, stand)`
key is not exclusive on the placeholder subclass. **That conclusion is withdrawn
too**: a count of arrivals cannot tell the departing aircraft's own inbound from
another aircraft, and "busy on both sides" is exactly what a rotation looks like
when the recorded off-block predates the arrival. `code/rotation_check.py` adds
the split the first two forms were missing — the three-letter prefix of the
arriving `FLIGHT_mvt`, against the base rate of that prefix at that stand:

Both shares are **event weighted** — arrivals with the prefix over arrivals — so
the base rate is weighted the same way. (The first version of this table compared
an event-weighted observed share against a row-weighted mean of the per-stand
base rates; round 2 of the review found it. The enrichment is unchanged in
substance and the base rates move by one to eight points.)

| band | class | rows | window arrivals | same prefix | base rate (same weighting) | P(another airline arrives during the taxi) |
|---|---|---|---|---|---|---|
| 1–2 h | placeholder | 281 | 138 | **92.0 %** | 19.2 % | 7.8 % |
| 1–2 h | rest | 473 | 99 | 20.2 % | 23.3 % | 6.3 % |
| 2–4 h | placeholder | 296 | 81 | **71.6 %** | 16.5 % | **12.2 %** |
| 2–4 h | rest | 205 | 33 | 36.4 % | 30.2 % | 4.9 % |
| 4–8 h | placeholder | 61 | **2** | 100 % | 48.2 % | **23.0 %** |
| 4–8 h | rest | 15 | **3** | 33.3 % | 27.6 % | 6.7 % |
| 8h+ | placeholder | 38 | **6** | 33.3 % | 16.9 % | **57.9 %** |
| 8h+ | rest | 11 | **2** | 50.0 % | 11.3 % | **100 %** |

**Read it, and read the limits with it.** At 1–2 h and 2–4 h the arrivals around
a placeholder row's block time carry the departing flight's own airline four to
five times as often as that stand's arrivals do in general. That is a rotation:
the aircraft that is about to operate this flight is arriving, and the recorded
block time — the stale schedule — predates it. `u = B_off − B_in < 0` then says
the *schedule* precedes the arrival, not that two aircraft shared a stand, and
that is the whole of §3's sign.

Three things this does **not** establish, all of them corrected out of an earlier
version of this paragraph after round 2 and round 3 of the review:

- **It is an airline, not an airframe.** `FLIGHT_mvt`'s first three characters
  identify the operator. A same-prefix arrival at the same stand minutes before a
  same-prefix departure is a rotation by any reasonable reading, and the flight
  numbers make it concrete on a subset (below), but the file carries no
  registration and this evidence cannot prove the same aircraft.
- **The stand is not exclusive.** Another airline arrives during the recorded
  taxi on **12.2 %** of 2–4 h placeholder rows against **4.9 %** of the ordinary
  ones, and on **23.0 %** against 6.7 % at 4–8 h. Only at 1–2 h do the two
  classes agree (7.8 % against 6.3 %). So a second aircraft does sometimes reach
  the stand, more often on placeholder rows, and "the key is exclusive" is
  withdrawn as an unqualified claim.
- **The change point is not at 8 h.** The 8h+ placeholder same-prefix share rests
  on **6** window arrivals and the 4–8 h one on **2**. What the data supports is
  a gradient: strong same-airline evidence at 1–4 h, and genuine re-use of the
  stand becoming common somewhere above 4 h. The 388's high bands are on the
  wrong side of that gradient whichever way the exact point falls.

**Three more measurements, one of which needs no proxy at all.** From round 2 of
the review, reproduced here: (a) the chain arrival itself carries the departing
flight's prefix on **79 %** of placeholder rows and on **83 %** of the
placeholder rows with `u < 0`, against **52 %** of the ordinary rows with
`u < 0`, and a third of those `u < 0` placeholder pairs are **consecutive flight
numbers** (AEZ2020 in / AEZ2021 out, ISR341 / ISR342, TWB405 / TWB406) — a
rotation written into the flight number, which is the closest this file gets to
airframe identity. And two statements that use no airline proxy: (b)
`B_in − SCHED`, how late the chain arrival is against the departure's own
schedule, has median **+0.32 h** on placeholder rows and **−0.63 h** on the rest;
(c) `P(ph | B_in > SCHED) = 0.678` (n = 637) against **0.334** (n = 848). (b) and
(c) are the load-bearing pair: they say the placeholder is written when the
previous arrival overtakes the schedule, and they hold whether or not the arrival
is the same airframe.

**And the whole of it is one existing column.** `slack = g_any − mvt_sched =
(T − B_in) − (T − SCHED) = SCHED − B_in`, so `slack < 0` **is** `B_in > SCHED`
— and it agrees with "`gap_arr_after_sched` is finite" on **1,485 of 1,485**
rows, correlation 1.0, because that shipped column is defined as the first
arrival between the schedule and take-off. The chain's entire signal about the
placeholder class is a column the head already has.

**The one indicator this uncovers, and why it is not usable.** P(ph | a
same-airline arrival within ±30 min of the **block time**) is 0.777 (n = 265)
against 0.420 (n = 1,230). It cannot be served: `docs/input-rule.md` blanks the
departure `BLOCK_TIME_UTC_mvt` on every ranking row, so conditioning on it is
conditioning on the label. The legal anchor is `SCHED_TIME_UTC_mvt`, which is
never blanked, and on a placeholder row the block time *is* the schedule — but
the legal form is saturated: **94.8 %** of 2025 noac rows and **94.8 %** of the
388 have a same-airline arrival within ±30 min of their schedule
(`analysis/rotation_legal.csv`), and what discrimination remains runs the other
way (P(ph) 0.468 with, 0.759 without). §5 fits it anyway, as the `rot` arm,
together with the continuous forms (the signed distance to the nearest
same-airline arrival, and the same-airline chain gap) that are not saturated.

**What the eight columns are anchored on.** Five are windows around
`SCHED_TIME_UTC_mvt` (`rot_same_near`, `rot_same_before`, `rot_same_after`,
`rot_same_gap`, `rot_slack`). `rot_g_same` is the gap from the departure's own
take-off to the last same-airline arrival, so it is anchored on take-off — legal,
because take-off is never blanked, but not schedule-anchored. `rot_pref_base` and
`rot_n_same_stand` are stand-month statistics of the arrival side and are
anchored on neither. All eight read only columns `docs/input-rule.md` leaves
populated, and `code/rotation.py` raises on an unmasked frame; round 3 of the
review verified the legality and the count and asked for this distinction to be
written down.

## 4. Step 2 — the 388 read through the chain

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/step2.py

`analysis/step2_margins.csv`, `analysis/step2_the_388.csv`,
`work/the_388_chained.parquet`. The shipped values are v9's
(`submissions/candidates/exp-027-k2.parquet`, sha `a3a139db…`; its 388 noac rows
are v7's, which are stack-5's). The margins are the per-airport low quantiles of
`u` on the 2025 departures, so the bound is `g − q`, and nothing here reads a
2026 label.

| band | n | median v9 | median `g_any` | median `g_prev2` | over `g_prev2` at 0.1 % | over `g_any` at 1 % | median excess |
|---|---|---|---|---|---|---|---|
| neg | 6 | 992 | 20,999 | 62,283 | 0 | 0 | — |
| 0–30 m | 24 | 1,972 | 11,727 | 71,285 | 0 | 0 | — |
| 30–60 m | 8 | 2,896 | 8,311 | 24,206 | 0 | 0 | — |
| 1–2 h | 169 | 2,392 | 7,328 | 16,979 | **2** | 17 | 598 s |
| 2–4 h | 138 | 6,259 | 8,403 | 19,259 | **12** | 46 | 1,053 s |
| 4–8 h | 21 | 15,874 | 14,098 | 27,327 | **5** | 10 | 3,601 s |
| 8h+ | 22 | 43,953 | 6,267 | 32,223 | **14** | 18 | 22,973 s |
| all | 388 | 3,465 | 7,773 | 19,918 | **33** | 91 | 3,450 s |

The chain reaches **99.48 %** of the 388 on the last-arrival form (386 rows) and
96.91 % on the second-previous form (376). Read with §3b, this table is a
description of where the stand record and the shipped prediction disagree, not a
list of errors. The 33 rows split in two ways and neither is actionable: 14 of the
33 are in the 8h+ band and 5 more in 4–8 h, where the stand is often genuinely
re-used during the recorded taxi (§3b) and whose whole movement exp-033 §8
already priced at 1.33 times the budget; the 14 below 4 h sit in the bands where
the bound holds but the label is a placeholder anchor the bound is not about
(§3b again).

## 5. Step 4 — the head, fold-measured

    OMP_NUM_THREADS=6 PRC_N_JOBS=6 .venv/bin/python experiments/exp-037-tail/code/head.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/evaluate.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/repair.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/blend.py

`analysis/lomo.csv`, `analysis/fold_gain.csv`, `analysis/repair_arms.csv`,
`analysis/repair_lambda.csv`, `analysis/blend_arms.csv`, `analysis/blend_w.csv`,
`analysis/blend_contribution.csv`, `analysis/shift_388.csv`.

The head is the shipped one refitted **and re-served**: `prc.tail_model`'s noac
v4 mixture — a LightGBM classifier for `ph`, a LightGBM regressor for
`mvt_sched − y` on the non-placeholder rows, `y = mvt_sched − (1 − p)·off_np`,
the shipped parameters, the shipped five-seed averaging, the shipped categorical
encoding (pandas Categoricals with `categorical_feature=`, so the four
categoricals get native splits), **the exp-009 two-point band table above
`BAND_MIN`, and the 60 s floor**. Two rounds of review were needed to get this
right: round 1 found the encoding coded as float ordinals, round 2 found
`predict` still returning the raw mixture with neither the band override nor the
floor. Every number in §5 and §6 is from the corrected path and differs from
both earlier versions. Three arms:

    base    NOAC_FEATURES_V2
    chain   + g_any, g_prev2, slack, slack2, log_g, n_dep_between, arr_stand_per_day
    rot     + the eight same-airline columns of §3b

§3 ruled the clip out on the evidence, so the mixture is the form the
measurement chose; the clip is measured anyway, as one of four carriers, because
the assignment named it.

**LOMO, twelve months, 1,507 rows** (`analysis/lomo.csv`, the single source for
these numbers): placeholder AUC **0.879 base → 0.884 chain → 0.888 rot**; noac
RMSE **3,969.4 base → 3,988.4 chain → 3,963.8 rot**, so the chain arm is
**−19.00 s [−52.92, +19.87]** and `rot` **+5.60 s [−69.12, +92.33]**, both
crossing zero. Under the earlier, unserved path these read +8.22 and +60.84; the
band override and the floor account for the whole difference, and the AUCs are
untouched because the classifier is unchanged. exp-033 §9 item 4 put the AUC to
beat at 0.871; the base head alone clears it under this encoding, so 0.871 is not
the bar it was taken for.

**The three shipped folds, inside the self-fitted family**
(`analysis/why_selffitted.csv`):

| arm | july_loo (339) | december (89) | january (63) | pooled (491) |
|---|---|---|---|---|
| chain | **+55.70** [+7.63, +99.50] | +14.67 | **−55.73** | +34.55 [−13.08, +76.96] |
| rot | +166.04 [−29.02, +328.27] | **+189.83** [+54.82, +339.71] | **−257.61** | +108.17 [−34.51, +238.65] |

**Neither pooled interval is clear of zero and neither arm repeats its sign on
all three folds.** The first version of this file reported +101.90 and +240.52
with intervals clear of zero and said the chain's sign repeated; both statements
were artifacts of a head that was not the shipped one, and both are withdrawn.
What survives is one fold: july_loo, the largest, where the chain arm gains
+55.70 s with an interval clear of zero.

**On the shipped vector, which is the only thing a candidate can be, no carrier
wins.** The arm is `shipped + correction`, restricted to `mvt_sched < BAND_MIN`
because above it the tail MEMBER inside stack-5 is the exp-009 band table and not
the learned mixture. The restriction was fixed before the fold numbers were read
— by file time, not by a commit, because the module and its first output entered
the repository together. It is one row too wide at the top: above `BAND_TOP`
(26 h) `_band_predict` returns NaN and the learned mixture keeps the row, which
is 1 of the 1,507, 1 of the 388 and **0 of the 491 fold rows**. It is also not
self-serving: the band it excludes is the band that most favours the chain head
in LOMO (8h+ gain **+196.5 s**), so restricting the arm gives up the chain's best
band rather than hiding its worst.

Four carriers per head, each also with a shrinkage fitted **out of fold** on the
other two folds; nothing is dropped. The shift on the 388 is the **served**
shift, after the 60 s floor `serve.py` would apply, because that is what the gate
measures:

| head | carrier | pooled gain | day-clustered | served shift on the 388 | budget |
|---|---|---|---|---|---|
| chain | `full` = the whole head difference | −20.35 | [−61.86, +20.18] | **570.4 s** | **breaks it** |
| chain | `full`, shrunk | −2.66 | [−10.54, +2.53] | — | — |
| chain | `clf` = the classifier change alone | **+7.58** | [−27.28, +40.20] | 320.0 s | under |
| chain | `clf`, shrunk | −4.31 | [−28.52, +12.74] | — | — |
| chain | `reg` = the regressor change alone | −24.03 | [−56.63, +5.85] | 439.0 s | under |
| chain | `reg`, shrunk | 0.00 | — (shrinkage 0 on all three folds) | — | — |
| rot | `full` | −36.59 | [−153.29, +72.16] | **849.0 s** | **breaks it** |
| rot | `full`, shrunk | −11.57 | [−35.33, +5.27] | — | — |
| rot | `clf` | −14.45 | [−70.75, +42.26] | **538.6 s** | **breaks it** |
| rot | `clf`, shrunk | −15.76 | [−40.58, −0.41] | — | — |
| rot | `reg` | −11.47 | [−94.60, +80.97] | **654.0 s** | **breaks it** |
| rot | `reg`, shrunk | −5.80 | [−23.92, +10.07] | — | — |
| — | `clip` = the physical clip at the bound | **−57.01** | [−198.82, +102.79] | **1,082.4 s** | **breaks it** |
| — | `clip`, shrunk | −100.75 | [−218.63, +2.66] | — | — |

Holm runs over the distinct carriers — `clip` does not depend on the head, so it
enters once, and `chain reg` shrunk moves no row and is not a test — **seven raw
and six shrunk**. (The first version of this table said six and five: the family
filter indexed `p.head`, which is the DataFrame method rather than the column, so
it silently dropped every clip row. Round 2 of the review found it.) **Every
adjusted p is 1.0.** The `chain full` arm's worst band is the largest one: 1–2 h
loses **−90.37 s [−143.69, −33.86]**, an interval clear of zero on the wrong
side. The out-of-fold shrinkages carry a caveat: `lam` for a fold is fitted on
the other two folds' rows, whose corrections come from heads trained on months
that include the held-out fold's month, so the folds overlap by construction.
"Out of fold" here means the *rows* are held out, not the training months.

**The second repair: a blend instead of a correction** (`code/blend.py`).
`arm = (1 − w)·shipped + w·y_head`, with `w` fitted out of fold, run for all
three heads so the difference from the base blend is what each arm adds under a
carrier that cannot inject the base head's weakness. The out-of-fold weights are
small — 0.000 / 0.163 / 0.101 base, 0.000 / 0.198 / 0.150 chain,
0.000 / 0.332 / 0.374 rot — and all three arms lose: **base −7.04 s
[−17.35, −0.01]**, **chain −9.61 s [−21.28, −1.20]**, **rot −22.48 s
[−48.03, −5.49]**, with each arm's own contribution negative (**chain −2.56 s
[−6.74, +0.45]**, **rot −15.44 s [−36.23, −2.23]**). Holm over the three leaves
every adjusted p at 1.0. Written after the carrier family was measured, so its
p-values are reported separately and not pooled with them.

**The gate.** "A candidate only if the fold gain is positive with the interval
clear of zero **and** the shift is under budget." Over the **sixteen** measured
arms — thirteen carriers and three blends — the best pooled gain is **+7.58 s**
with an interval crossing zero, no interval is clear of zero on the positive
side, and every arm that moves the 388 by more than 439 s breaks the 500 s
budget. **Not a candidate.** No file was written to `submissions/candidates/`,
no upload was made, and the base stays v9 (desk, 2026-09-08).

## 6. Why it fails, and what that rules out

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/why.py

`analysis/why_selffitted.csv`, `analysis/why_levels.csv`,
`analysis/why_correlation.csv`, `analysis/why_marginal.csv`.

**The carrier is wrong, and after two corrections the evidence for it is
narrower than this file first claimed.** On the **491** fold noac rows the
shipped vector scores **3,213.9 s** and my refitted base head **3,258.7 s** —
only **44.8 s** worse, where the first version of this section said 610 s and the
second 546.5 s. Both of those were reading a head that was not being served the
way the shipped one is. The two vectors are **1,619.4 s** apart at a correlation
of 0.99. On the **471** rows the `mvt_sched < BAND_MIN` restriction leaves live,
the chain correction correlates with **my** head's error at **0.171** and with
the **shipped** vector's at **0.043**, and its RMSE-optimal scale is **1.051**
against the first and **0.177** against the second — a factor of six. So the
shape of the argument survives its own correction: the correction is mostly a
repair of a weakness that is specific to my fit, and **a correction defined as
the difference of two heads fitted outside the stack cannot be assumed to
transfer to a vector inside it.** What does not survive is the size: the two
heads are much closer than this file claimed, so the *reason* the arm fails is
less about a bad base head and more about there being little to carry.

**The marginal information about the class, tested properly.** The first version
of this test asked whether the chain columns add to `logit(p_base)`, and for
`rot` it put the chain block and the rotation block together and called all
fifteen columns new. It is now nested: the chain block is tested over the base
logit alone, and the rotation block over the base logit **plus** the chain block,
so each arm is asked what it adds beyond the arm before it. The day-clustered
permutation was also wrong — it concatenated permuted day blocks and wrote them
back by row position, which splits days rather than moving them whole — and is
replaced by one that draws every row's block from a single other calendar day.
Neither arm's block is detectable. The **chain** block adds **LR = 7.63** on 7
new columns over the base logit, day-permutation **p = 0.3875** (null mean 7.20,
95th percentile 15.38) — a likelihood ratio smaller than the average of its own
null. The **rotation** block adds **LR = 8.52** on 8 new columns over the base
logit *and* the chain block, **p = 0.5120** (null mean 9.58, 95th 18.61). Both
are failures to reject at 1,507 rows, not proofs of equality, and no equivalence
margin was declared. The earlier figures (p = 0.183, then p = 0.2505 and
p = 0.3395) came from the broken permutation and the un-nested rot block, and are
withdrawn.

The first version of this section said the chain's null was "expected by
construction" because its columns are the shipped features. **That is withdrawn**:
a fitted scalar probability is not a sufficient statistic for its own inputs, so
a linear function of them can add to the logit in principle. The correct
statement is weaker and still useful — the classifier was trained on these
columns, so a linear re-entry of them is a low bar, and it is not cleared.

**What this does NOT close.** It closes two carriers (a bolt-on difference of
heads, and a blend) and one form (the physical clip) on this population. It does
not close the arrival side: §1 shows the arrival side was never outside the head
to begin with, so exp-033 §9's revival — an observable from outside
`NOAC_FEATURES_V2` — is still unanswered. The airline-prefix columns of §3b are
the first genuinely outside observable this record has tried, and they lose.

## 7. Does the chain bound ordinary rows usefully? — measured, not argued

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/bulk.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-037-tail/code/bulk_clip.py

`analysis/bulk_bound.csv`, `analysis/bulk_bound_2025.csv`, `analysis/bulk_clip.csv`.

**No. The clip is scored out of fold on the shipped vector and it loses on every
fold at both margins.** The first two versions of this section argued the case
from the violation rate and priced the movement at "+18.6 s of board if it is
uncorrelated with the shipped error". Round 2 of the review said that is the
wrong null and that the thing can simply be measured; `code/bulk_clip.py` does
it, on the non-noac rows of the three shipped folds, with the margin `q` fitted
per airport on each fold's **training** months:

| fold | margin | rows moved | shift RMS | RMSE shipped → arm | gain | day-clustered |
|---|---|---|---|---|---|---|
| july_loo | 0.1 % | 156 (0.08 %) | 149.7 s | 246.60 → 286.26 | **−39.66** | [−107.39, −0.33] |
| july_loo | 1 % | 1,827 (1.0 %) | 161.9 s | 246.60 → 292.28 | **−45.68** | [−114.29, −5.38] |
| december | 0.1 % | 136 | 13.7 s | 200.32 → 200.75 | −0.44 | [−0.77, −0.17] |
| december | 1 % | 1,371 | 53.9 s | 200.32 → 206.71 | −6.39 | [−7.65, −5.27] |
| january | 0.1 % | 147 | 23.9 s | 330.91 → 331.64 | −0.73 | [−1.54, −0.33] |
| january | 1 % | 1,231 | 61.1 s | 330.91 → 335.77 | −4.86 | [−8.38, −3.23] |

**Six arms, six losses, every interval clear of zero on the wrong side.** The
reason is in the same table: on the July rows the clip moves, RMSE goes from
**300.3 s to 5,087.2 s**. The rows the clip moves overlap heavily with the rows
whose *true label* also breaks the bound — at the 0.1 % margin, july_loo has 156
predicted violations against 178 label violations with **124 in common**,
december 136 against 127 with 99, january 147 against 131 with 111 — so the clip
is largely a filter that finds genuinely long taxis and truncates them. (An
earlier version of this paragraph said the two sets *are* the same rows. They are
not: the rates match to the third decimal, the row sets overlap by about three
quarters. Round 3 of the review found the overstatement.)

What survives of the section is the descriptive half. The bound almost never
bites: at zero margin a true ordinary 2025 label breaks it on **0.0076 %** of the
2,043,497 chained rows (156 rows, one in thirteen thousand; the 0.0110 % figure
counts noac rows too), and its median value is **6.16 h** against a median taxi
of 948 s. On the ranking file the shipped v9 sits above it on **289 of the
337,650 chained ordinary rows** (0.086 %) by a median **202 s**, a 99th
percentile of 3,244 s and a maximum of 57,352 s. Those 289 rows are now
identified, not as errors, but as the rows most likely to have a long true taxi.

**And the feature route this section used to name as the next block is already
the shipped state for the tail heads.** `gap_arr1..3` are in `_EXP002_SIGNALS`
and in `NOAC_FEATURES_V2`, so every `prc.tail_model` head already fits on the
chain. It is **not** the shipped state everywhere. stack-5 has seven members
(`experiments/exp-026-rebuild/candidates/stack-5.json`), and two of them —
`C2-cat` and `L2-hub8725` — are built on the retro **frame v2**
(`experiments/exp-008-diversity/result-C2-d8p.json`,
`result-L2-hub.json`), which carries no stand-chain column at all: the retro
frame's own `stand_prev_arr_block_gap` starts at v3 and its import of
`gap_arr1..3` from `prc/tail_features.py` starts at v4
(`experiments/exp-001-retro-v1/features.py`). `prc/nonm.py`'s `FEATURES` carries
none either, and it is a published head though not one of the seven. So the
count is **at least two current stack members, or three refit targets counting
`prc.nonm`** — not the "one place" the first version of this section claimed.
Each is a member refit, which this assignment forbade, and on this evidence it is
a small thing to try rather than a lane.

## 8. Files

    experiments/exp-037-tail/
      SPEC.md                    the reading of the assignment
      RESULTS.md                 this file
      review/                    three review rounds, their prompts and transcripts
      code/run_all.sh            every command above, in dependency order
      code/chain.py              the chain, from a ranking-masked frame
      code/rotation.py           the eight same-airline columns (§3b names the anchors)
      code/test_chain.py         its brute-force check, synthetic and on real data
      code/checks.py             population and labels against exp-033's own files
      code/build.py              every departure of the ten airports, chained
      code/frames.py             the noac frames, chained, label-free for 2026
      code/step1.py step2.py step3.py mechanism.py rotation_check.py
      code/head.py evaluate.py repair.py blend.py why.py serve.py
      code/bulk.py bulk_clip.py   the bound on ordinary rows, and the clip scored
      analysis/*.csv             every table above, one canonical source per column
      work/*.parquet             the frames and the fold evaluations
      logs/*.log                 the run output of each step

`code/serve.py` is written and unused: it builds the candidate the assignment
describes, and it is not run because the gate is not met. Nothing under `prc/`
changed; `git diff` touches no shared code.

## 9. The three review rounds, and what each moved

### Round 1 — the fallback reviewer

`review/NOTE.md` records the reviewer substitution: the named reviewer
(GPT-5.6 Sol, xhigh, Codex CLI) read the whole experiment and then stopped
before it wrote a verdict, so `MODES.md`'s documented
fallback ran the same prompt. Round 1 returned **NEEDS REVISION**: the verdict
"a reading, not a candidate" stands under every check, the chain is legal and
correctly built, and every table matched its CSV — but three claims about *what
was built and why* were refuted. All three were reproduced independently before
being accepted.

| # | finding | verified how | what moved |
|---|---|---|---|
| 1 | §3b's mechanism is wrong: the crowding arrivals are the aircraft's own inbound, not a shared stand | `code/rotation_check.py`, all twelve months, reproduces 92.0 % same-prefix in 1–2 h and 57.9 % in 8h+ | §0, §3b and §4 rewritten. **Round 1's own wording went too far and rounds 2 and 3 pulled it back**: the base rate it quoted (21.3 %) was row-weighted where the observed share is event-weighted (19.2 %), "the aircraft's own inbound" is an airline and not an airframe, and the stand is not exclusive. The surviving statement is the schedule one (§3b) |
| 2 | the chain IS the shipped `gap_arr1`/`gap_arr2` | max abs difference 0 s on 1,485 / 1,460 / 386 / 376 rows; `gap_arr1..3` in `NOAC_FEATURES_V2` and `_EXP002_SIGNALS` | §0, §1, §6 and §7 rewritten; the "outside `NOAC_FEATURES_V2`" claim withdrawn; §7's next block narrowed to `prc/nonm.py` |
| 3 | the head's categorical encoding was not the shipped one | `prc/tail_model._encode` passes Categoricals and `categorical_feature=`; `head.py` passed float ordinals | `head.py` fixed and **every number in §5 and §6 re-measured**; the "sign repeats on all three folds" claim withdrawn |
| 4 | `analysis/shift_388.csv` had no producer | none of the twelve scripts wrote it | `repair.py::shift_388` writes it; `run_all.sh` now reproduces it |
| 5 | §7 mixed the all-rows violation rate with the ordinary-row denominator | 224 / 2,044,957 all chained against 156 / 2,043,497 ordinary | §2 and §7 carry both numbers |
| 6 | "pre-registered" cannot be shown from git | the module and its first output entered in one commit | §5 says "fixed before the fold numbers were read (file times, not a commit)" |
| 7 | §3's table omitted the bands where the sign reverses | `analysis/step3_conditional.csv` | 4–8 h and 8h+ are in the table, with their small-n caveat |
| 8 | two files quoted the same statistic with different intervals (a shared module RNG) | `lomo.csv` against `why_selffitted.csv` | §5 cites `lomo.csv` as the single source for the LOMO numbers |
| 9 | "out of fold" overstates the shrinkage | folds overlap in training months by construction | §5 states what is held out |
| 10 | §6's two populations were not named | 491 fold rows against 471 live rows | §6 names both |
| 11 | §3b's first withdrawal was right for a narrower reason than stated | finding 1 | §3b withdraws both earlier forms |

Findings 12 and 13 were verifications with no defect: the chain's legality under
`docs/input-rule.md`, the `merge_asof` position handling, `test_chain.py`'s
brute-force agreement, every table against its CSV, and the gate arithmetic.

**One correction to round 1.** Its finding 1 proposes the same-airline
indicator as the revival, calling it label-free on the 388. It is not: the
indicator is anchored on the departure's `BLOCK_TIME_UTC_mvt`, which is exactly
what the ranking file blanks. The schedule-anchored form is legal, is saturated
at 94.8 %, and is measured in §5 as the `rot` arm, where it loses.

### Round 2 — GPT-5.6 Sol, xhigh, the round of record

`review/round2-gpt-5.6-sol-xhigh-2026-09-08.txt`, verdict at line 11020.
**NEEDS REVISION, eight blocking findings**, read against the committed record at
`e0bbf07`. "The 'no candidate' decision remains correct, but the round-of-record
report has blocking measurement and interpretation errors."

*(An earlier version of this section said "three blocking and eight non-blocking,
against `85d6c23`". That was wrong. The three-blocker text it summarised is
command output the transcript echoed while Sol read an intermediate draft, not
Sol's verdict; the verdict is the message at line 11020 and it has eight
blockers. Round 3 finding 1 caught the error.)*

| # | blocking finding | what moved |
|---|---|---|
| 1 | the head still did not match the shipped **prediction** path: `predict` returned the raw mixture with no band override and no 60 s floor | `head.py` fixed; §5 and §6 re-measured for the second time. LOMO chain **+8.22 → −19.00 s**, rot **+60.84 → +5.60**; self-fitted pooled chain **+101.90 → +34.55**, rot **+240.52 → +108.17**; §6 base RMSE **3,760.4 → 3,258.7** and its deficit **546.5 → 44.8 s** |
| 2 | §3b proves same **airline**, not the aircraft's own inbound, and its base rate was row-weighted against an event-weighted share | §0 and §3b rewritten: base rates **19.2 %** and **16.5 %**, the exact-aircraft and exclusivity claims withdrawn, the thin n above 4 h stated, the proxy-free `B_in > SCHED` pair made load-bearing; `mechanism.py`'s docstring marks both of its readings withdrawn |
| 3 | the LR test's rot block was chain + rot, not 8 new columns, and the day permutation split days | `why.py` fixed twice (nested null; donor-day cycling). p = 0.2505 / 0.3395 withdrawn → **chain LR 7.63, p 0.3875; rot LR 8.52 over an 8-column null, p 0.5120** |
| 4 | the Holm family used `p.head`, the DataFrame method, so every clip row was dropped | families corrected to **7 raw / 6 shrunk**, 16 tested arms; every adjusted p still **1.0** |
| 5 | the movement gate used pre-serve corrections | `repair.py::shift_388` reports the **served** shift; §5's seven numbers changed by 1–8 s and no arm changed side of the budget |
| 6 | "`prc/nonm.py` is the only shipped target without the chain" is false — C2-cat and L2-hub8725 are stack-5 members on frame v2, which has no chain column | §1 and §6 say **at least two current stack members**, three refit targets with the published non-missing head |
| 7 | round 1's measured ordinary-row clip was dropped and §7 kept the hypothetical "+18.6 s if uncorrelated" | `code/bulk_clip.py` written; §7 is now the measurement (six arms, six losses) |
| 8 | SPEC still described three measurements that did not run | `SPEC.md` amended: margins per **airport**, the third variant is **prev2** not `fresh` (`fresh` count 0), no `g ≤ 24 h` tables |

Its passing checks: both identity pairs at max abs difference 0 s, the chain
already inside every `prc.tail_model` head, `rotation.py` legal and guarded,
94.8 % schedule-anchored coverage in both years, the bootstrap paired, the
shrinkage caveat accurate, the best shipped-vector gain **+7.58 s
[−27.28, +40.20]**, no candidate written and `git diff` clean under `prc/`.

### Round 3 — GPT-5.6 Sol, xhigh

`review/round3-gpt-5.6-sol-xhigh-2026-09-08.txt`. **NEEDS REVISION, nine
blocking and two non-blocking**, read against `e0bbf07` plus the working tree.
Sol's own summary: "The shipped-feature blocker and the harmful ordinary-row
clip blocker are discharged. The mechanism blocker is only partly discharged:
the data supports a same-airline relationship, but it does not prove the same
aircraft or exclusive stand use."

Seven of its nine blockers are round 2's, restated because the fixes were in the
working tree and not yet committed when it ran (its findings 3–8 are round 2's
1, 3, 4, 5, 6 and 8). Two are new:

| # | new finding | what moved |
|---|---|---|
| 1 | §9's account of round 2 was wrong — 3 blocking against `85d6c23` where the verdict is 8 blocking against `e0bbf07` | this section, rewritten from the verdict message |
| 9 | §7 said the predicted violations **are** the label violations; they are not the same rows — july_loo **156 predicted, 178 label, 124 shared**, december 136/127/99, january 147/131/111 | §7 states the overlap and drops the identity claim; the six loss estimates and the 300 → 5,087 s moved-row RMSE are unchanged |

Its finding 2 is round 2's mechanism blocker, repeated because it was only
partly discharged: the version it read still called the chain an aircraft chain
and the stand exclusive. §3b now withdraws both.

Non-blocking: (10) the eight rotation columns are legal, but `rot_g_same` uses
legal take-off data and two are stand-month statistics rather than
schedule-anchored — the count 8 and the 94.8 % coverage are confirmed, and §3b
says which columns are anchored on what; (11) `repair.py`'s docstring carried
four stale numbers — the docstring now points at §5 and §6 instead of repeating
them.

Round 3 confirms the arithmetic that decides the gate: `slack = SCHED − B_in`
and its 1,485 / 1,485 agreement, the feature equalities, the training-only,
noac-excluded clip fit, the paired bootstrap, and the best pooled gain
**+7.58 s [−27.28, +40.20]**. "There is no exp-037 candidate and no change under
`prc/`."

### What three rounds did not move

The verdict. Round 1 (fallback), round 2 and round 3 each returned NEEDS
REVISION on the *record* and each confirmed the *answer*: **a reading, not a
candidate**. Twenty-eight findings changed how the numbers were measured and
what they are allowed to say; not one of them produced an arm that passes both
halves of the gate. The three numbers that survived every round unchanged are
the two identity checks at 0 s, the algebra `slack = SCHED − B_in`, and the best
pooled gain of +7.58 s with an interval crossing zero.

## 10. What would revive this lane

The lane is `closed-provisional` as a source of a noac candidate, on the
measurement in §5 and §6, not on an opinion. Five things could revive it, in
descending order of what they would be worth.

**1. An observable outside `NOAC_FEATURES_V2` that is not the arrival side.**
exp-033 §9 asked for one and this experiment did not deliver it: §1 shows the
chain was already inside every tail head. §3b's airline-prefix columns are the
first genuinely outside observable the record has tried and they lose (§5). The
ask is still open and it is the highest-value one, because every arm measured so
far is a re-expression of information the head already has.

**2. The two stack members that genuinely lack the chain.** `C2-cat` and
`L2-hub8725` are stack-5 members built on retro frame v2, which carries no
stand-chain column (§7). Adding `gap_arr1..3` there is not a no-op, and it is
the only place in the shipped stack where it is not. This assignment forbade
member refits, so it was not tried; it is a stack-wide change with its own gate
and its own runtime, and it belongs to a rebuild block, not to a noac head.

**3. A carrier that is not a bolt-on difference of two heads.** Inside the
self-fitted family the chain arm gains **+34.55 s pooled [−13.08, +76.96]** and
**+55.70 s [+7.63, +99.50]** on july_loo, the largest fold; on the shipped vector
the same information is worth **+7.58 s** at a correlation of 0.043 (§6). The
failure is measured to be the carrier. The carrier that would test it properly is
a tail-head refit *inside* the stack rather than a correction added to its
output, which again needs a rebuild block.

**4. Power that does not exist on this population.** The best arm is +7.58 s with
a day-clustered half-width of about ±34 s. An interval clear of zero at that
effect needs the half-width under 7.58 s, which is about **twenty times** the 491
fold rows — roughly 9,700 noac rows where 2025 has **1,507** in total. A noac-only
change of this size is not measurable on this data whatever its truth, so a
revival has to come from a larger effect or a larger population, not from a
better estimator.

**5. An airframe identifier.** The file carries no registration, and §3b's
mechanism therefore rests on an airline proxy plus two proxy-free statements. If
a registration or a rotation link ever enters the data, the exclusivity question
§3b withdraws becomes answerable, and with it whether the physical clip is safe
on the high bands. Nothing in the current files supplies it.

**What would NOT revive it:** more carriers of the kinds already measured. Seven
raw carriers, six shrunk, three blends and the physical clip have been scored on
the same 491 rows, every Holm-adjusted p is 1.0, and the two arms under the
movement budget are among the ones with no measurable gain.
