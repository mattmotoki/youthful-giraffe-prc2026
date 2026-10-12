# exp-033-noac — RESULTS

Worker `noac`, tier hard. Assignment `assignments/noac-learn.md`, reading in
`SPEC.md`. Mac laptop, 6 threads, no GPU, no upload. Everything reruns with
`experiments/exp-033-noac/code/run_all.sh` (~12 minutes), which is the order the
commands below assume.

**Third writing of this file, after two review rounds.** Sol round 1 returned
BLOCKED with eight blocking findings and round 2 returned BLOCKED with seven
(`review/round1-*.txt`, `review/round2-*.txt`); §12 and §13 list every finding
and what it changed. The protocol allows two rounds, so there is no round 3.
Four findings changed the answer rather than the prose: the two-sample test now
validates **grouped by calendar day** and permutes day labels **inside the real
comparison**; the null for a matched-subset arm is a **query permutation with
the ranking months' day-block structure** instead of a same-size random subset;
one band's apparent result did not survive that null; and one hand-computed
arithmetic block in §8 was wrong by a factor of about four. Every round made the
experiment's own claims weaker and its verdict better supported.

## 0. The answer

**Which 2025 sub-population do the 388 belong to?** The ordinary 2025 LIRF noac
population, season-matched. 383 of the 388 are LIRF, as are 1,487 of the 1,507;
the LIRF noac rate in both ranking months sits inside the 2025 range (0.967 %
and 1.743 % against 0.459–2.216 %); 272 of the 388 have their nearest 2025
neighbour in June–September; and at the row level the 388 sit at the 2025 cloud's own density
(median nearest-neighbour distance 0.1833 against 2025's own 0.1855, 198 inside
/ 163 edge / 27 outside against a 194 / 175 / 19 calibration, χ² p = 0.15). A
day-grouped two-sample classifier finds **no transferable row-level separator**:
every AUC is 0.366–0.532 and none rejects its day-relabel null (smallest
two-sided p 0.257), and the same instrument fails to separate 2025's own months
(0.346–0.528), so this is a failure to detect at these sample sizes rather than
a proof of equality. What is different is above the rows: July 2026 is a
distinguishable operating **month** in the day-level features (0.715 against
July 2025, p 0.059; 0.787 against summer 2025, p 0.020, where the largest 2025
month control is 0.744), January 2026 is not (0.487, p 0.911), the band mix is
its own, and **8h+ leaves the cloud** — 10 of its 22 rows are outside, and 4–8 h
has no row inside.

**Does a fold measure a direction on that sub-population?** One band points
positively, no band clears the assignment's gate, and the pooled sign does not
repeat across the three folds (+12.82 / −6.57 / −0.10). Conditioning the 2025
labels on the within-band nearest-neighbour match lowers P(placeholder) by
**0.064 [−0.092, −0.035]** in 1–2 h and **0.045 [−0.080, −0.008]** in 2–4 h, in
both feature spaces and at every k. The arm that follows — the shipped v7 values
plus the per-band matched-minus-all correction — moves 169 rows at **194.9 s
RMS**, well inside the 500 s budget, and gains **+8.61 s** of noac RMS out of
fold on the matched rows. But its day-clustered interval on the scope the
assignment names is **[−3.35, +23.19]**, which crosses zero, so the rule "**a
candidate only if the fold gain is positive with an interval clear of zero and
the shift is under budget**" is not met. **Verdict: a reading, not a candidate.**

**And the deeper reason, which is the experiment's central negative:** a
within-band k-nearest-neighbour subset is a systematic operation, not a random
one. Against 200 fake query sets drawn with the ranking months' own day-block
structure, the matching procedure by itself already produces −0.037 of the 1–2 h
label difference and +7.56 s of the fold gain. The real 2026 match adds −0.026
and +1.05 s beyond that, at two-sided p = 0.090 and one-sided p = 0.104 — and
**nothing in either step survives Holm over the seven bands** (smallest adjusted
p 0.418 and 0.731). That is a failure to reject at this sample size, not a
demonstration that the match carries nothing; no equivalence margin was
declared. Round 1's one apparently strong band result (8h+ label difference,
p = 0.010) does **not** survive the corrected null and is withdrawn.

Two further measured negatives stand beside it. Restricting a *learned* head's
fit set to the matched rows costs **238.2 s [−318.5, −153.9]** of
leave-one-month-out RMSE and 0.028 of placeholder AUC, so the matched
sub-population is a worse training set, not a better one. And the one band where
the shipped base and the 2025 folds openly disagree, 8h+, cannot be acted on:
moving its 22 rows to the 2025 band rate is 667.4 s RMS on the 388, **1.33 times
the movement budget, for about 1 s of board in either direction**.

## 1. The two populations, and the label the record means

`code/build.py` builds one frame per departure with `AIRCRAFT_TYPE_mvt` null,
carrying every `prc.tail_features.tail_signals` column, one month at a time so
the ranking file's January and July never touch (`signals_by_month`'s contract).
The frames are **label-free** — no target, no departure block time, no
placeholder class, asserted at the end of the build — and the 2025 labels go to
a separate `work/labels_2025.parquet` that only §5 onward joins
(`code/join_labels.py`, Sol S1).

    experiments/exp-033-noac/code/run_all.sh          # or, for this section:
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/build.py

**1,507 rows in the twelve 2025 months and 388 in the ranking file**, both
matching the record. The band counts of the 388 reproduce
`experiments/exp-026-rebuild/analysis/noac_cells.csv` exactly (6 / 24 / 8 / 169
/ 138 / 21 / 22).

Two placeholder definitions are in the repo and only one reproduces the record
(`code/join_labels.py`, `analysis/placeholder_definition.csv`; this is a §5
check because it reads labels):

| band | n (2025) | `\|mvt_sched − y\| < 300 s` | `BLOCK.floor('min') == SCHED.floor('min')` | `noac_cells.csv` |
|---|---|---|---|---|
| neg | 19 | 0.053 | 0.053 | 0.053 |
| 0-30m | 65 | 0.508 | 0.231 | 0.508 |
| 30-60m | 39 | 0.410 | 0.205 | 0.410 |
| 1-2h | 754 | 0.373 | 0.190 | 0.373 |
| 2-4h | 503 | 0.590 | 0.312 | 0.590 |
| 4-8h | 77 | 0.805 | 0.468 | 0.805 |
| 8h+ | 50 | 0.780 | 0.420 | 0.780 |

So `P(placeholder)` here is the 300 s tolerance
(`prc.tail_model.PLACEHOLDER_TOL`), the noac head's own notion. exp-022's
minute-equality class is the *non-noac* regime's label and is a different class.

## 2. Step 1 — the 388 described label-free

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/describe.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/describe2.py

`analysis/population.csv`, `analysis/nn_feature_share.csv`,
`analysis/noac_rate.csv`. `population.csv` carries description only; the
placement is `place.py`'s alone (Sol S4).

**The regime is one airport.** 383 of the 388 are LIRF (LTFM 4, LSZH 1); 1,487
of the 1,507 are LIRF. Every band above 30 minutes is LIRF-only in 2026.

**The rate is inside the 2025 range, at its summer end.** LIRF's share of
departures with no aircraft type:

| | 2025 min | 2025 max | 2026-01 | 2026-07 |
|---|---|---|---|---|
| LIRF noac share | 0.459 % (Nov) | 2.216 % (Jul) | **0.967 %** | **1.743 %** |

2026-01 sits between October 2025 (0.813 %) and June 2025 (1.259 %); 2026-07
between August 2025 (1.274 %) and July 2025 (2.216 %). Neither month is outside
what the feed did in 2025. `docs/shift-2026.md` names LIRF as the one airport
January 2026's disruption left alone (median 2026 day at the 0.52 percentile of
2025 days), which is consistent with this.

**Where the nearest neighbours live.** For each of the 388, the month of its
nearest 2025 noac row: 108 July, 77 August, 44 June, 43 September — the summer
months, in the order of their supply. Per-band descriptors (month split, hour,
`mvt_sched`, `sched_date_cross`, `noac_day_share`, `arr_day_ph_rate`, airports)
are in `analysis/population.csv`.

## 3. Step 1 — is it the same population?

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/twosample.py

A classifier separating 2026 rows from 2025 rows in `NOAC_FEATURES_V2`, scored
out of fold. Two repairs, one per round. **Validation is grouped by calendar
day** (Sol S3): the first version used row-stratified folds, so rows of one day
sat on both sides of the split, and since six of the 32 features (`dep_day_n`,
`noac_day_n`, `noac_day_share`, `noac_day_count`, `nonm_day_count`,
`arr_day_ph_rate`) are constant within a day and nearly unique to it, a
classifier could memorise a day *inside* the fold. **The null now relabels the
days of the real comparison** (Sol R2-2): the second version partitioned a
separate 2025 season pool, which gave the exact-July comparison a 31-against-59
null where the comparison itself is 31 against 31. The two samples are now
pooled, the class label is permuted over calendar days with the real class-1 day
count held, and 100 permutations are run at a **fixed recorded seed** — the
second version seeded from Python's randomised `hash()`, so the cited command
did not reproduce the committed CSV.

The p-value is **two-sided**. With day-grouped folds an AUC below 0.5 is a
learner that transfers in the wrong direction, which is its own departure from
the null, not a "no signal" reading (Sol R2-2).

Three reference levels on the same instrument (`analysis/twosample.csv`, seeds
101–105 in the file, every replicate in `analysis/twosample_reps.csv`):

**With the day block (the full separator space):**

| comparison | n | AUC | day-relabel null | two-sided p |
|---|---|---|---|---|
| 2025-07 vs **2026-07** | 339 / 280 | **0.715** | 0.475 ± 0.116 [0.247, 0.749] | 0.059 |
| summer 2025 vs **2026-07** | 726 / 280 | **0.787** | 0.496 ± 0.101 [0.237, 0.716] | **0.020** |
| 2025-01 vs **2026-01** | 63 / 108 | 0.487 | 0.460 ± 0.133 [0.201, 0.764] | 0.911 |
| winter 2025 vs **2026-01** | 210 / 108 | 0.628 | 0.461 ± 0.111 [0.249, 0.760] | 0.139 |
| all 2025 vs all 2026 | 1507 / 388 | 0.706 | 0.495 ± 0.074 [0.338, 0.652] | **0.020** |
| *2025-06 vs 2025-07* (control) | 189 / 339 | *0.744* | – | – |
| *2025-07 vs 2025-08* (control) | 339 / 198 | *0.632* | – | – |
| *2025-01 vs 2025-02* (control) | 63 / 58 | *0.604* | – | – |
| *2025-08 vs 2025-09* (control) | 198 / 170 | *0.597* | – | – |
| *2025-12 vs 2025-01* (control) | 89 / 63 | *0.449* | – | – |

**Without it (the row-level space):**

| comparison | n | AUC | day-relabel null | two-sided p |
|---|---|---|---|---|
| 2025-07 vs 2026-07 | 339 / 280 | 0.366 | 0.461 ± 0.088 [0.280, 0.691] | 0.257 |
| summer 2025 vs 2026-07 | 726 / 280 | 0.414 | 0.466 ± 0.064 [0.354, 0.657] | 0.455 |
| 2025-01 vs 2026-01 | 63 / 108 | 0.437 | 0.442 ± 0.090 [0.230, 0.697] | 0.950 |
| winter 2025 vs 2026-01 | 210 / 108 | 0.404 | 0.469 ± 0.066 [0.306, 0.680] | 0.297 |
| all 2025 vs all 2026 | 1507 / 388 | 0.532 | 0.482 ± 0.039 [0.376, 0.577] | 0.198 |
| *the five 2025 month controls* | – | *0.346 – 0.528* | – | – |

Three statements, and the third is the one a head has to live with:

1. **At the day level July 2026 is a distinguishable operating month.** 0.715
   against July 2025 and 0.787 against summer 2025, against a null whose 100
   day relabellings reach 0.749 and 0.716; the pooled year comparison is 0.706
   at p = 0.020. January 2026 separates from nothing (0.487 at p = 0.911;
   winter 0.628 at p = 0.139). This is a real effect and it lives in the six
   day-level columns.
2. **Its size, against 2025's own months, is stated in both directions.** The
   largest 2025 month control is June vs July at 0.744. The exact-month
   comparison (0.715) is **below** it; the season-pooled comparison (0.787) is
   **above** it (Sol R2-2 — the first writing of this section quoted only the
   first). The two are not like for like — the pooled comparison has 726 rows
   and 90 days on the 2025 side against 339 and 31 — so the fair summary is
   that July 2026's day-level distance from July 2025 is about the size of a
   2025 month change, and that pooling three summer months against it gives a
   larger number than any single 2025 month pair does.
3. **At the row level no transferable classifier is found.** Every AUC is
   between 0.366 and 0.532 and **no comparison rejects its own null**, the
   smallest two-sided p being 0.257. Four of five sit below 0.5, which with
   day-grouped folds means the learner transfers in the wrong direction between
   days rather than that the two sets are identical. The 2025 month controls
   behave the same way (0.346–0.528), so the instrument does not separate
   2025's own months either: **this is a failure to detect, at these sample
   sizes, not a proof of equality** — the honest reading is "no row-level
   feature combination separates the years that also transfers across days".

The earlier claim that the 0.98 of the first version was "day memorisation, not
shift" was too strong in one direction, and the claim that the row level shows
"nothing" was too strong in the other. What stands: the leakage was real, a July
day-level effect the size of a month is real, and the row-level features give a
head nothing to key on.

## 4. Step 1 — inside, edge or outside the 2025 cloud

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/place.py

Distance is Gower over `NOAC_FEATURES_V2` (IQR scale, per-feature clip at 3,
categoricals 0/1, a feature missing on either side dropped for that pair). The
reference is 2025's own leave-one-out nearest-neighbour distance **with same-day
neighbours excluded**: `describe.py`'s reference allowed a 2025 row a free
neighbour from its own day, which a 2026 row can never have, so it was biased
toward calling 2026 rows far out. Inside = at or below the reference median,
edge = up to its p95, outside = beyond, so **2025 places at 50 / 45 / 5 overall
by construction — that calibration is overall, not per band** (Sol S4).

| space | 2025 cross-day LOO nn1 median | 2026 nn1 median | inside | edge | outside |
|---|---|---|---|---|---|
| full (day block kept) | 0.2437 | 0.2668 | 167 | 194 | 27 |
| row-level (day block dropped) | 0.1855 | **0.1833** | 198 | 163 | **27** |

At the row level the 388 sit at the 2025 cloud's own density: the median
distance to a 2025 neighbour is the number 2025's rows give each other. Against
the 194 / 175 / 19 the calibration implies, χ² = 3.83 on 2 df (p = 0.15) at the
row level and 8.89 (p = 0.012) in the full space. The 27 outside are not spread
evenly:

| band | n2026 | inside | edge | outside |
|---|---|---|---|---|
| neg | 6 | 0 | 5 | 1 |
| 0-30m | 24 | 2 | 16 | 6 |
| 30-60m | 8 | 2 | 6 | 0 |
| 1-2h | 169 | 119 | 47 | 3 |
| 2-4h | 138 | 73 | 62 | 3 |
| 4-8h | 21 | 0 | 17 | 4 |
| 8h+ | 22 | 2 | 10 | **10** |

**8h+ is the band that leaves the cloud**: 10 of 22 rows outside. 4–8 h has no
row inside it. The two bands with the largest per-row stake are the two the 2025
evidence reaches least well. `analysis/placement.csv`.

## 5. Step 2 — the 2025 labels conditioned on the match

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/join_labels.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/conditional.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/conditional2.py

**The estimand is the distinct-row rate** — the unweighted mean of `ph` over the
distinct 2025 rows that are a within-band k-nearest neighbour of one of the 388
— because that is the set step 3 fits a head on (Sol S5). The
multiplicity-weighted rate (each 2026 row's five neighbour slots) is what a k-NN
predictor would average and is reported as sensitivity; the first version of
this file used the weighted rate here and the unweighted mask in step 3, which
mixed two estimands.

`conditional.py` matches against the whole 2025 set and is kept only to show why
that is not the instrument: its largest apparent difference is the negative
band's **+0.38**, and only 23 % of that band's neighbours are in the band, so
the number is band composition. Matched-band shares: neg 0.23, 0-30m 0.48,
30-60m 0.20, 1-2h 0.87, 2-4h 0.71, 4-8h 0.52, 8h+ 0.51.

`conditional2.py` is the instrument: within band, k = 5, row-level space, one
paired calendar-day-clustered bootstrap (4,000 replicates) recomputing both
means on the same resample.

| band | n2026 | n2025 band | n2025 matched | P(ph) matched | P(ph) all | diff | 95 % interval | weighted diff |
|---|---|---|---|---|---|---|---|---|
| neg | 6 | 19 | 15 | 0.067 | 0.053 | +0.014 | [0.000, +0.046] | −0.019 |
| 0-30m | 24 | 65 | 52 | 0.558 | 0.508 | +0.050 | [−0.009, +0.114] | +0.126 |
| 30-60m | 8 | 39 | 20 | 0.400 | 0.410 | −0.010 | [−0.167, +0.160] | +0.015 |
| **1-2h** | **169** | **754** | **440** | **0.309** | **0.373** | **−0.064** | **[−0.092, −0.035]** | −0.116 |
| **2-4h** | **138** | **503** | **321** | **0.545** | **0.590** | **−0.045** | **[−0.080, −0.008]** | −0.043 |
| 4-8h | 21 | 77 | 52 | 0.769 | 0.805 | −0.036 | [−0.096, +0.026] | −0.043 |
| 8h+ | 22 | 50 | 38 | 0.816 | 0.780 | +0.036 | [−0.036, +0.117] | +0.056 |

Two bands are clear of zero on this interval, 1–2 h and 2–4 h, both negative,
both in both feature spaces (full space: −0.096 [−0.130, −0.064] and −0.067
[−0.104, −0.027]) and both stable across k = 1, 3, 5, 10. The weighted estimand
would instead have made 0-30m clear (+0.126) and 2-4h not; that disagreement is
why one estimand had to be declared.

**But that interval prices the labels, not the match.** It answers "given these
matched rows, how well is their placeholder rate measured?", which is not the
question a matched-subset finding needs. §6d asks the other question — how much
of this difference does the matching procedure produce by itself — and against
that null 1–2 h is p = 0.090 and 2–4 h is p = 0.647.

## 6. Step 3 — does a fold measure a direction?

### 6a. The head, and the two arms

The head is the simplest one the matched rows support: a per-band two-class
mixture that estimates, on the fit months, `p` (shrunk toward the pooled rate,
m = 5 pseudo-rows), the mean offset `o_ph` over the band's placeholder rows and
`o_np` over its non-placeholder rows, and predicts the RMSE-optimal expectation
`mvt_sched − (p·o_ph + (1−p)·o_np)`, never an argmax
(`prc.tail_model`'s own rule). `code/direction.py`.

**Arm 1: restrict the head's FIT SET to the matched rows.** Leave-one-month-out
over the twelve 2025 months:

| head | scope | n | RMSE fit=all | RMSE fit=matched | gain | 95 % |
|---|---|---|---|---|---|---|
| per-band mixture | matched | 938 | 5055.8 | 5047.8 | +8.0 | [−18.4, +38.3] |
| per-band mixture | all noac | 1507 | 4673.5 | 4679.1 | −5.6 | [−27.1, +16.4] |
| learned two-class odds | matched | 938 | 4673.0 | 4911.3 | **−238.2** | **[−318.5, −153.9]** |
| learned two-class odds | all noac | 1507 | 4250.1 | 4468.7 | −218.7 | [−282.1, −154.9] |

The learned head (a LightGBM classifier plus a non-placeholder regressor over
`NOAC_FEATURES_V2`, five seeds, the module's own `NOAC_PARAMS`) is **worse** on
the matched fit set by 238 s, and its out-of-fold placeholder AUC falls from
0.871 to 0.843. Restricting the fit set is not a refinement; it is 938 rows
instead of 1,507. `analysis/lomo_pooled.csv`, `analysis/clf_arm.csv`.

**Arm 2: keep the shipped model and apply the matched-vs-all correction.** The
base is stack-5's own out-of-fold prediction on the noac rows of `july_loo`,
`december` and `january` (`experiments/exp-026-rebuild/oof/stack5-*.parquet`);
v7 serves stack-5's bytes on all 388 noac rows (`docs/SUBMISSIONS.md` row 7).
The arm is `shipped + (head fitted on matched − head fitted on all)`, per band,
fitted on the fold's training months only. `code/direction2.py`,
`analysis/shipped_fold_arms.csv`.

**No label reaches the match.** The matched mask is computed once over all 1,507
rows from the 2026 rows' neighbours in `NOAC_FEATURES_V2`, label-free on both
sides. The head is fitted on the fold's training months only (`fit[fit._m]`),
and the scoring scope selects validation rows by that same label-free mask.

| arm | scope | n | RMSE shipped | RMSE arm | gain | 95 % | fold shift RMS |
|---|---|---|---|---|---|---|---|
| the per-band head alone | matched | 303 | 3337.7 | 4738.3 | −1400.7 | [−2108.6, −893.1] | 2875.8 |
| every band, correction only | matched | 303 | 3337.7 | 3336.5 | +1.2 | [−49.9, +54.1] | 387.1 |
| every band, correction only | all noac | 491 | 3213.9 | 3190.6 | +23.2 | [−16.2, +59.4] | 368.8 |

The head alone is 1,400 s worse than the shipped model: the shipped model's
row-level information is worth far more than any band table, which is the direct
measurement behind BOARD READING 4 (exp-002's purpose-built head cost +9.26 s on
the board). Over every band the correction measures nothing.

### 6b. Per band, per fold

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/per_fold.py

`analysis/per_fold_bands.csv`. On the matched scope the 1–2 h correction is
**+12.82 (july_loo) / −6.57 (december) / −0.10 (january)** and 2–4 h is
**+43.31 / −4.79 / +10.52**; on all noac rows they are +16.09 / −11.51 / +16.08
and +57.40 / −22.98 / +24.28. Only 30-60m — eight 2026 rows — is positive on all
three folds. Every pooled gain below is one or two folds' gain.

### 6c. Robustness of the knobs

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/robust.py

For the 2–4 h arm, the largest: k in {1, 3, 5, 10, 20} gives +33.4 / +19.6 /
+32.6 / +24.8 / +13.3; space row-level +32.6 against full +73.2; the shrinkage
constant m in {0, 2, 5, 10, 25} moves it by 4.1 s; the placeholder tolerance in
{120, 300, 600, 1200} s by 0.03 s. The head is insensitive to the two
constants and sensitive to the two match settings, which is why the match needs
its own null.

### 6d. The null a matched-subset arm actually needs

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/placebo2.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/placebo_step2.py

The day-clustered interval prices the validation rows' label noise at a **fixed
match**. It does not price the match. Two reviewer rounds shaped the null that
does. Sol S6 removed the first design (a random 2025 subset of the same per-band
size, which preserved neither the geometry of a nearest-neighbour rule nor its
day structure, and which moved the scoring population as well as the fit set).
Sol R2-3 removed the second (row sampling across all twelve months, whose fake
query sets occupied 189–216 calendar days against the real 62).

The null used now is a **query permutation with the ranking months' day-block
structure**: 31 winter days from Dec/Jan/Feb 2025 and 31 summer days from
Jun/Jul/Aug 2025, the noac rows on them, capped per band at the 388's own
per-band counts; then the identical pipeline — the same within-band
k-nearest-neighbour rule in the same feature space, the same per-band head, the
same folds, the same base — with a fake query forbidden from taking a neighbour
from its own calendar day, which is the year-disjointness the real queries face
for free. The **scoring population is held at the real matched rows** so only
the treatment moves. 200 draws; the statistic is an add-one permutation p-value,
which cannot go below 1/201.

**The structure match is close but not exact, and the diagnostics say by how
much** (`analysis/placebo_reps_diag.csv`): the real query set is 388 rows on 62
calendar days producing 938 matched rows; the 200 fake sets are 278–379 rows
(median 334) on 59–62 days producing 782–952 matched rows (median 892). The day
count and the treatment size now match; the row count is short because 62 days
of 2025 do not hold 388 noac rows. The same-day ban applies only to the fake
side, and it moves their label effect toward zero, so it makes the real effect
slightly easier to see rather than harder.

**The fold gain** (`analysis/placebo.csv`, matched scope at the fixed scoring
population):

| band | gain, real | placebo mean ± sd | add-one p |
|---|---|---|---|
| neg | −0.76 | −2.95 ± 9.23 | 0.522 |
| 0-30m | −11.96 | −6.95 ± 6.82 | 0.816 |
| 30-60m | +1.99 | −0.27 ± 5.14 | 0.403 |
| 1-2h | +8.61 | +7.56 ± 1.33 | **0.104** |
| 2-4h | +32.58 | +23.74 ± 18.95 | 0.323 |
| 4-8h | −1.23 | −1.60 ± 3.65 | 0.602 |
| 8h+ | −27.76 | −17.45 ± 42.89 | 0.672 |
| every band | +1.16 | +1.93 ± 49.01 | 0.582 |

On all noac rows: 1–2 h +12.71 against +10.73 ± 2.19 (p = 0.129), 2–4 h +42.72
against +31.25 ± 25.00 (p = 0.313).

**The label difference** (`analysis/placebo_step2.csv`, same construction,
two-sided):

| band | diff, real | placebo mean ± sd | two-sided add-one p |
|---|---|---|---|
| neg | +0.014 | −0.036 ± 0.030 | 0.299 |
| 0-30m | +0.050 | −0.023 ± 0.062 | 0.179 |
| 30-60m | −0.010 | +0.015 ± 0.087 | 0.886 |
| 1-2h | −0.064 | −0.037 ± 0.016 | **0.090** |
| 2-4h | −0.045 | −0.039 ± 0.016 | 0.647 |
| 4-8h | −0.036 | +0.017 ± 0.031 | **0.060** |
| 8h+ | +0.036 | −0.016 ± 0.050 | 0.289 |

**This is the experiment's central negative and it is about the match, not the
folds.** Taking a within-band k-nearest-neighbour subset is a systematic
operation, not a random one: **on average over structure-matched query sets** it
lowers a band's P(placeholder) by 0.037 in 1–2 h and 0.039 in 2–4 h, and gains
+7.56 s in 1–2 h when its correction is applied to the shipped model. The real
2026 match adds a further **−0.026 on the label** (two-sided p = 0.090, against a
null sd of 0.016) and **+1.05 s of fold gain** (p = 0.104, null sd 1.33). Nothing
in either table reaches 0.05, and under Holm over the seven bands
(`analysis/band_table.csv`) the smallest adjusted p is 0.418.

**What that does and does not license.** It licenses withdrawing the claim that
the fold gain is evidence about the 2026 rows: most of it is reproduced by query
sets that have nothing to do with them, the spread is wide (the 200 fake 1–2 h
gains run **+0.36 to +8.99 s**, and the real +8.61 sits at their 90th
percentile), and 198 of the 200 fake label differences are negative as well
(**−0.094 to +0.001**). It
does **not** license "the match carries nothing": p = 0.09 is a failure to
reject at n = 200 draws and 303 scored rows, not an equivalence result, and no
equivalence margin was declared. The honest statement is that **at this sample
size the match's contribution cannot be separated from the matching procedure's
own effect**, and that its point estimate — about 40 % of the label difference
and 12 % of the fold gain — is what a follow-up would have to confirm.

### 6e. The 1–2 h arm, stated in full

    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/board_price.py
    OMP_NUM_THREADS=6 .venv/bin/python experiments/exp-033-noac/code/records.py

Every line below is in `analysis/arm_summary.csv` or `analysis/board_price.csv`
(Sol R2-7); nothing here is computed by hand.

| | |
|---|---|
| rows moved on the 388 | 169 |
| shift RMS on the 388 | **194.9 s** (mean −128.7 s) — **budget 500 s: OK** |
| fold gain, matched scope (303 rows) | +8.61 s, day-clustered [−3.35, +23.19] |
| fold gain, all noac rows (491 rows) | +12.71 s, day-clustered [+1.68, +25.83] |
| against the query-permutation null | placebo +7.56 ± 1.33; add-one p **0.104** (matched), **0.129** (all noac) |
| per fold (matched) | +12.82 / −6.57 / −0.10 |
| fold rho (`noac_cells.py` eq. 2) | −1.293 |
| board if that rho transfers | −0.129 s |
| board if the change is uncorrelated | +0.081 s |
| board ceiling at this shift size (Cauchy–Schwarz, σ = 3,542 s) | −2.884 s |

The ceiling uses the record's quoted σ, which is **blend-5's** error on the 388
(`docs/SUMMARY.md`, exp-026 §13), not stack-5's: stack-5 takes none of blend-5's
noac gain (`exp-026/analysis/noac_discount.csv`), so the two differ. The ceiling
scales as √σ, so a 10 % error in σ moves it by 5 %; nothing else depends on it.

**Verdict, against the assignment's own rule.** The rule is "a candidate only if
the fold gain is positive with an interval clear of zero **and** the shift is
under the 500 s RMS budget". The shift passes at 194.9 s. The day-clustered
interval passes on all noac rows ([+1.68, +25.83]) and **fails on the matched
sub-population** ([−3.35, +23.19]), which is the scope the assignment names, so
the rule is not met and the arm is not a candidate. Three further facts point
the same way: against the query-permutation null the gain is p = 0.104 / 0.129
unadjusted and nothing after Holm; its pooled value lives in one fold of three
(+12.82 / −6.57 / −0.10); and at face it is 0.129 s of board against a 0.081 s
price of moving, on a lane whose two board readings quote noac fold gains at
zero. **A reading.**

## 7. The table the assignment asks for

Band × {n2026, n2025 matched, P(placeholder) matched vs all, fold direction,
shift RMS}. Every column has one canonical source, named in `code/table.py`:
the label columns are `within_band_rowlevel.csv` (k = 5, distinct rows), the
placebo columns are `placebo_step2.csv` and `placebo.csv` (matched scope, fixed
scoring population, day-block query permutation, 200 draws), the shift is
computed in `table.py` itself over all 388 rows, so the column adds in
quadrature to the every-band arm's 358.8 s. Holm is applied within each step
over the seven bands. `analysis/band_table.csv`.

| band | n2026 | matched / band | P(ph) matched | P(ph) all | diff [95 %] | diff vs placebo p (Holm) | fold gain (s) | placebo mean ± sd | gain vs placebo p (Holm) | shift RMS on 388 | price of moving |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neg | 6 | 15 / 19 | 0.067 | 0.053 | +0.014 [0.000, +0.046] | 0.299 (1.000) | −0.76 | −2.95 ± 9.23 | 0.522 (1.000) | 37.7 s | 0.003 s |
| 0-30m | 24 | 52 / 65 | 0.558 | 0.508 | +0.050 [−0.009, +0.114] | 0.179 (0.896) | −11.96 | −6.95 ± 6.82 | 0.816 (1.000) | 217.3 s | 0.101 s |
| 30-60m | 8 | 20 / 39 | 0.400 | 0.410 | −0.010 [−0.167, +0.160] | 0.886 (1.000) | +1.99 | −0.27 ± 5.14 | 0.403 (1.000) | 34.4 s | 0.003 s |
| **1-2h** | **169** | **440 / 754** | **0.309** | **0.373** | **−0.064 [−0.092, −0.035]** | **0.090 (0.537)** | **+8.61** | **+7.56 ± 1.33** | **0.104 (0.731)** | **194.9 s** | **0.081 s** |
| 2-4h | 138 | 321 / 503 | 0.545 | 0.590 | −0.045 [−0.080, −0.008] | 0.647 (1.000) | +32.58 | +23.74 ± 18.95 | 0.323 (1.000) | 134.2 s | 0.038 s |
| 4-8h | 21 | 52 / 77 | 0.769 | 0.805 | −0.036 [−0.096, +0.026] | 0.060 (0.418) | −1.23 | −1.60 ± 3.65 | 0.602 (1.000) | 99.1 s | 0.021 s |
| 8h+ | 22 | 38 / 50 | 0.816 | 0.780 | +0.036 [−0.036, +0.117] | 0.289 (1.000) | −27.76 | −17.45 ± 42.89 | 0.672 (1.000) | 114.2 s | 0.028 s |
| **all** | **388** | **938 / 1507** | — | — | — | — | **+1.16** | **+1.93 ± 49.01** | 0.582 | **358.8 s** | **0.275 s** |

Read the two p-value columns as one sentence: **no band's label difference and
no band's fold gain is separated from what the matching procedure produces on a
2025 query set of the same day structure**, before any multiplicity adjustment
(smallest raw p 0.060 on the 4–8 h label, 0.104 on the 1–2 h gain) and after it
(smallest Holm 0.418 and 0.731). The 8h+ label difference that round 1 reported
at p = 0.010 is **withdrawn**: against the day-block null it is p = 0.289.

## 8. Why every fold-fitted noac head has pointed the wrong way

The experiment supplies one explanation and puts evidence against two others.

**Evidence against a row-level population shift.** §3 and §4: at the row
level the 388 are indistinguishable from 2025's noac rows, at the 2025 cloud's
own density. The assignment's framing — "the 2026 noac population is not the
2025 one the folds measure" — does not hold at the row level, and the day-level
separation that does show (§3) is the size of a 2025 month change. What
is different is the band mix and the thinness of the 2025 evidence in the top
two bands.

**Evidence against a base that behaves differently between the years.** The
per-band mixture is invertible, so any prediction names an **implied mixture
coordinate** `p = (o_np - off)/(o_np - o_ph)` (`code/why.py`,
`analysis/mixture_coordinate.csv`). It is affine within a band, so a median is a
valid summary of it, but it is **not a probability**: it leaves [0, 1] on 46 of
the 388 rows (15 in 1–2 h, 11 in 2–4 h, 11 in 8h+, 4 of the 8 in 30-60m, 3 in
4–8 h, 1 each in neg and 0-30m), because the shipped model is not a band mixture
and is under no obligation to sit between a band's two anchors (Sol S10). The
first two columns below are medians of that coordinate
(`analysis/base_odds.csv`); the last two are the §5 distinct-row rates
(`analysis/within_band_rowlevel.csv`, k = 5), the same numbers the band table
carries:

| band | stack-5 OOF on 2025 | v7 on the 388 | whole-band 2025 | matched 2025 |
|---|---|---|---|---|
| 1-2h | 0.316 | 0.297 | 0.373 | 0.309 |
| 2-4h | 0.597 | 0.641 | 0.590 | 0.545 |
| 4-8h | 0.877 | 0.917 | 0.805 | 0.769 |
| 8h+ | 0.984 | 0.997 | 0.780 | 0.816 |

The gaps between the two base columns are +0.019, −0.043, −0.040 and −0.013,
which is small beside the base-to-band gaps; these are medians of a coordinate
with a long tail and no interval is attached, so this is **evidence against a
large base shift, not proof that none exists**. In 1–2 h the shipped model sits
nearer the matched value than the whole-band value, which is what a row-level
model that already carries most of the match's information would look like — a
consistency, not a measurement, and §6d could not separate the match's
contribution in either direction.

**Supplied: a band table is a bad substitute for a row-level model.** §6a
measures it directly: replacing the shipped prediction with the per-band head
costs 1,400 s of noac RMSE out of fold, and restricting a *learned* two-class
head's fit set to the matched rows costs 238.2 s [−318.5, −153.9] LOMO, with the
placeholder AUC falling 0.871 to 0.843. That is BOARD READING 4's +9.26 s seen
on the fold instead of the board: a coarse head applied to rows a fine model
already fits does not point anywhere in particular, and its sign on the board is
whatever the fold hands it.

**What it cannot say.** 8h+ is the one place the base and the folds disagree
loudly: v7 sits at an implied median 0.997 where 2025 says 0.780 for the band
and 0.816 for the matched rows. Round 1 also read that band as the one place the
real match beat its null; against the corrected day-block null it does not
(p = 0.289), so that reading is withdrawn and only the base-versus-fold
disagreement stands. Moving those 22 rows from 0.997 to the band's 0.753 is a
shift of **2,802.9 s each**, which is **667.4 s RMS over the 388** -
`analysis/band8_arithmetic.csv`, **1.33 times the 500 s budget, so the budget
alone forbids it**. Its board price is small in both directions: about
**+0.95 s if the move is uncorrelated with the shipped error**, and about
**−0.95 s at the most favourable value the rho arithmetic allows**, against the
−28.31 s that all 388 rows would be worth if they were perfect. That band is the
exp-009 two-point mixture's own decision (`BAND_EDGES`, `docs/SUMMARY.md`: 18 of
22 decided, 4 hedged). This experiment does not touch it, and no upload may be
designed to read it (score-feedback rule).

## 9. What would revive the lane

1. **A label-free instrument for the 8h+ band.** 22 rows, 10 of them outside the
   2025 cloud, and the one band where the shipped base and the 2025 folds
   disagree by more than 0.2 of a mixture coordinate. §4 says 2025's
   evidence up there is too thin and too far away, so the observable has to come
   from outside `NOAC_FEATURES_V2` — the arrival side of the same tail number,
   or the stand's occupancy across the day boundary. It also has to be cheap in
   movement: the whole-band move above is 1.33 times the budget for about 1 s of
   board in either direction, so an instrument that revives this band must
   decide **individual rows**, not the band.
2. **A match that is not a nearest-neighbour rule.** §6d's negative is
   specific: a within-band k-NN subset is a systematic operation, and most of
   its label effect and its fold effect is reproduced by 2025 query sets that
   have nothing to do with 2026. A match that conditions on something the band
   does not already carry — a propensity model, a pair matched on the day's
   operating state, an instrument built on the arrival feed — faces a different
   null and can clear it.
3. **Evidence that did not take part in the selection.** §6c shows the
   arm is sensitive to k and to the feature space, and both were looked at
   before the primary settings were fixed; registering the 1–2 h arm now cannot
   undo that (Sol S8). The only reading that converts this into a candidate is a
   test on data that did not participate: a shipped-base out-of-fold prediction
   on the **nine 2025 months that carry no stack-5 OOF today**, with the arm
   frozen at the registered form and fitted only on the three months already
   used. The condition is the assignment's own gate applied to those nine months
   alone — **a positive fold gain whose day-clustered interval is clear of
   zero**, with the shift on the 388 still below 500 s RMS. A sign count over
   all twelve months is not a substitute, because three of the twelve chose the
   arm (Sol R2-5); to use twelve, the selection must be repeated inside every
   outer fold and the read taken from the outer loop only.
4. **A better placeholder classifier, not a better fit set.** §6a
   measures the fit-set idea as a 238 s negative and puts the ceiling where it
   has always been: the odds. AUC 0.871 out of fold over twelve months is the
   number to beat, and the 28 s of board the oracle is worth is unchanged by
   anything here.

## 10. Files

    experiments/exp-033-noac/
      SPEC.md                  the reading of the assignment
      RESULTS.md               this file
      code/run_all.sh          every command below, in dependency order
      code/*.py                20 scripts, `records.py` added in round 2
      analysis/*.csv           every table above, one canonical source per column
      work/noac_2025.parquet   the label-free 2025 frame
      work/labels_2025.parquet the labels, joined only at §5
      work/noac_2026.parquet   the 388, label-free by construction
      work/*_d.parquet, *_lab.parquet, fold_arms.parquet   derived frames (in git)
      work/*.npy               the Gower matrices (not in git; rebuilt by
                               code/describe.py then code/place.py)
      work/MANIFEST.txt        sha256 of every work file
      review/                  the two reviewer prompts and both transcripts

Nothing under `prc/` changed; `git diff` touches no shared code. No candidate
parquet was written and no upload was made. `work/MANIFEST.txt` is unchanged
after the round-2 rebuild, which is the check that moving the label extraction
out of `build.py` (R2-1) changed no frame the analysis reads.

## 11. Reviewer

`reviewer: sol`. Two fresh-context passes, `codex exec --sandbox read-only -m
gpt-5.6-sol -c model_reasoning_effort=xhigh`, prompts at `review/prompt.md` and
`review/prompt-round2.md`, transcripts at
`review/round1-gpt-5.6-sol-xhigh-2026-09-08.txt` and
`review/round2-gpt-5.6-sol-xhigh-2026-09-08.txt`. The protocol allows two
rounds; both were used and there is no round 3.

**Round 1: BLOCKED**, eight blocking findings and four non-blocking (§12). Sol
agreed with the operational verdict ("reading, not candidate") and with the
placeholder definition, the placement construction and its arithmetic, the board
arithmetic, the step 2 estimator's pairing, and scope control. The findings are
about the two nulls, one estimand mismatch, and claims stated more strongly than
the evidence carries.

**Round 2: BLOCKED**, seven blocking and one non-blocking (§13). Sol confirmed
five round-1 repairs and the verdict, and found that three of the round-1
repairs were incomplete (the label boundary, the two-sample null and its seeds,
the query null's day structure), three claims still stronger than the data, and
one class of number that lived only in prose. **All twelve of round 1 and all
eight of round 2 are accepted; none is disputed.** Two of round 2's findings
changed a number rather than a sentence: the withdrawal of the 8h+ label result
and the correction of the 8h+ movement arithmetic.

Sol's closing position, which this file follows: *"The operational verdict
should remain 'a reading, not a candidate' because the matched-scope fold
interval crosses zero and no untouched confirmation exists. The new reason — 'the
match carries almost nothing beyond the band' — is not yet supported."*

## 12. Round 1 dispositions

| # | finding | disposition |
|---|---|---|
| S1 | step 1 was not label-free as implemented: `build.py` read `y_true`, the block time and the minute class into the §1 frame | **accepted, fixed.** `build.py` writes a label-free frame and a separate `labels_2025.parquet`, and asserts the frame carries no label column; `join_labels.py` is the §5 boundary and the only place a label enters |
| S2 | the placeholder definition is right, but its check belongs after §5 (non-blocking) | **accepted.** The table moved into `join_labels.py` and `analysis/placeholder_definition.csv`; §1 says it is a §5 check |
| S3 | the two-sample test used row-stratified folds, so the day block leaked *inside* the fold and contaminated the null as well as the comparison; the cited command could not run `day_null()`; the 10 replicates were never persisted | **accepted, re-measured, and the conclusion changed** — §3. Validation is now `GroupKFold` on calendar day, every replicate is in `analysis/twosample_reps.csv`, and the null is a structure-matched day permutation with 2025 month-pair controls beside it. The corrected answer: at the row level no transferable classifier is found (AUC 0.37–0.53), and at the day level July 2026 separates at 0.715–0.787. **Round 2 (R2-2) re-cut this again**: the seeds were not fixed, the null was not a permutation of the real comparison, and 0.787 is *above* the 0.744 month control, not below it — see §3 and §13 |
| S4 | the placement construction and arithmetic are right; 50/45/5 is an overall calibration, not per band, and `population.csv` carried stale placement columns (non-blocking) | **accepted, fixed.** `describe.py` no longer emits a placement; §4 states the calibration is overall and adds the χ² against it (3.83, p = 0.15 row-level; 8.89, p = 0.012 full) |
| S5 | step 2 used the multiplicity-weighted rate while step 3 used the distinct-row mask: two estimands | **accepted, one declared.** The distinct-row rate is primary everywhere because it is the set the head is fitted on; the weighted rate is sensitivity. The table changed: 1–2 h is −0.064 [−0.092, −0.035] and **2–4 h joins it** at −0.045 [−0.080, −0.008], while 0-30m's +0.126 drops to +0.050 and stops being clear |
| S6 | the subset placebo was not the claimed null: it preserved only per-band counts, and in the matched scope it moved the scoring population; the percentile is not a p-value | **accepted, replaced, and it changed the answer** — §6d. `placebo2.py` permutes the QUERY side (2025 rows at the 388's band mix, no same-day neighbour, identical pipeline) at a scoring population fixed to the real matched rows, 200 draws, add-one p-values; `placebo_step2.py` applies the same null to the label difference. The claim "the only band that beats all 60 placebos" is withdrawn. **Round 2 (R2-3, R2-4) replaced this null in turn**, because a row sample over twelve months occupied 189–216 calendar days against the real 62; the final numbers and the final wording are §6d's, not this row's |
| S7 | multiplicity understated; the family was not declared | **accepted.** `table.py` declares the family once — seven bands at the registered settings, tested once per step, Holm within each step — and reports adjusted p-values in the table. Nothing survives: the smallest is 0.070. The k and space sweeps are labelled sensitivity, and the record says they were looked at before the settings were fixed |
| S8 | §9 claimed registration could convert the existing result into one test | **accepted, corrected.** §9 item 3 now says registration cannot undo selection and names the only evidence that can: the nine 2025 months that did not take part, with a sign that repeats on eight of twelve as the condition |
| S9 | the board arithmetic is right; the docstring wrongly bounds rho below by −1 (non-blocking) | **accepted, fixed** in `board_price.py`: rho is a ratio, not a correlation, the fold's own value is −1.293, and the real floor is Cauchy–Schwarz |
| S10 | §8 treated an affine coordinate as a probability, mixed two rates, and over-claimed "measured out" | **accepted, weakened** — §8. It is named an "implied mixture coordinate", its out-of-range share is given (46 of 388, per band), the matched column now carries the §5 distinct-row rate, one source with the band table, and the claim is "evidence against a large base shift, not proof that none exists" |
| S11 | the ledger's negative-band sign was wrong, the 10-replicate null was not in a CSV, and §7 cited the wrong source for its intervals | **accepted, all fixed.** The sign is +0.38 (`analysis/conditional_rowlevel.csv`); every replicate of every null is now persisted; `table.py` names one canonical source per column and the superseded CSVs (`placebo_bands.csv`, `twosample_daynull.csv`, `band_fold_multiplicity.csv`) are deleted rather than left to be cited. A correction row is appended to `docs/LEDGER.md` |
| S12 | scope control passed; the script count was wrong (non-blocking) | **accepted.** 19 scripts, and `code/run_all.sh` now lists them in dependency order |

## 13. Round 2 dispositions

Round 2 prompt `review/prompt-round2.md`, transcript
`review/round2-gpt-5.6-sol-xhigh-2026-09-08.txt`. **Verdict BLOCKED**, seven
blocking findings and one not. Sol confirmed S2, S4, S5, S7, S9 and the
substance of S12 as repaired, confirmed that the distinct-row table supports the
step-2 result, and confirmed the operational verdict. Every finding is accepted;
none was disputed.

| # | finding | disposition |
|---|---|---|
| R2-1 | S1 was a storage separation only: `build.py` still read the target, the block time and the minute class to build the label file, so step 1's script touched labels | **accepted, fixed.** Every label read now lives in `join_labels.py`; `build.py` writes only the two label-free frames and asserts it. The rebuild is the check: `work/MANIFEST.txt` is byte-identical afterwards, so no frame the analysis reads changed |
| R2-2 | S3 was not fully repaired: the null partitioned a separate 2025 season pool (31 vs 59 days where the real comparison is 31 vs 31), and the seed came from Python's randomised `hash(tag)`, so the cited command did not reproduce the CSV. The interpretation of AUC 0.366 as "nothing" was too strong, and 0.787 is **above** the 0.744 month control | **accepted, re-measured, seeds fixed.** The null now relabels calendar days **inside the pooled real comparison** holding the class day counts, 100 permutations, seeds 101–105 recorded in `analysis/twosample.csv`, every replicate in `twosample_reps.csv`. §3 is rewritten: the row-level reading is "no transferable classifier is found", not "nothing", and the control comparison is stated in both directions |
| R2-3 | S6's null still sampled rows across twelve months: the fake query sets occupied 189–216 calendar days against the real 62, so query clustering, season and treatment size all moved | **accepted, replaced again.** `placebo2.py` now draws **31 winter and 31 summer 2025 calendar days**, takes the noac rows on them, and caps each band at the 388's own count. Diagnostics are persisted (`analysis/placebo_reps_diag.csv`): 59–62 query days and 782–952 matched rows against the real 62 and 938. Every §6d, §7 and §6e number moved |
| R2-4 | the central negative was stronger than the data: "whatever the query rows are" is false, the placebo range was misquoted, and p = 0.060 is not equivalence | **accepted, weakened and re-stated.** §6d now says "on average over structure-matched query sets", gives the full placebo range (+0.36 to +8.99 s in 1–2 h, the real +8.61 at the 90th percentile), states the point estimate of what the match adds (about 40 % of the label difference, 12 % of the gain), and says in terms that **p ≈ 0.09 is a failure to reject at this sample size, not an equivalence result** — no equivalence margin was declared, so none is claimed. No density or neighbour-degree mechanism is asserted |
| R2-5 | S8's revival condition still reused selected evidence: a sign count over twelve months includes the three that chose the arm, and a sign count is not the assignment's interval rule | **accepted, rewritten.** §9 item 3 now names the **nine months that carry no stack-5 OOF**, with the arm frozen and fitted only on the three already used, and the gate is the assignment's own — a positive fold gain whose day-clustered interval is clear of zero, shift still under 500 s RMS. Using all twelve requires the selection repeated inside every outer fold |
| R2-6 | S10's repair was incomplete: §8 still said two explanations were "removed", and `why.py` still said "odds", "refuted" and "one band" | **accepted, fixed in both.** §8 now reads "evidence against", the coordinate is called an implied mixture coordinate everywhere, and `why.py`'s docstring says the same and names both clear bands |
| R2-7 | S11 was not repaired for the whole record: the §6e intervals, the ALL row, the 46 out-of-range count and the 8h+ hypothetical existed only in prose, and three ledger rows cited the wrong sources | **accepted, and it found a real arithmetic error.** `records.py` persists `arm_summary.csv`, `mixture_coordinate.csv` and `band8_arithmetic.csv`, and `table.py` writes the ALL row. The hand-computed 8h+ block was **wrong**: the move is 2,802.9 s per row and **667.4 s RMS on the 388**, which is **1.33 times the budget, not five times**, and it is worth about ±0.95 s of board, not ±13 s against a 28 s prize. §8 carries the machine's values and the ledger correction row cites the CSV |
| R2-8 | non-blocking: `SPEC.md`'s script inventory named two deleted scripts and omitted three new ones | **accepted, fixed.** `SPEC.md` lists all twenty scripts in the order `run_all.sh` runs them, grouped by step, with the label boundary marked |

The protocol allows two rounds, so there is no round 3. Nothing in this file is
fitted to a board reading, no candidate parquet exists, and no upload was made.
