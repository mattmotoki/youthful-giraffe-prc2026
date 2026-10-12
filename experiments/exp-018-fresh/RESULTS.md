# exp-018 fresh — results

An independent lineage built under the information diet in `SPEC.md`. Every
number below is from `prc.data.fold_frames` folds with the validation month
masked exactly like the ranking file.

## 1. What the data says before any model

Measured on the twelve 2025 training months (2,085,047 departures) unless a
line says July only. Script: `code/` plus the exploration in
`RESULTS.md#appendix`.

### 1.1 The metric is a tail metric

| population | rows | share of Σ(y-948)² |
|---|---|---|
| all departures | 2,085,047 | 100 % |
| y > 3600 s | 4,126 (0.20 %) | 49.6 % |
| LIRF | 160,704 (7.7 %) | 47.1 % |
| no Network Manager join | 22,470 (1.08 %) | 42.1 % |
| **LIRF and no NM join** | **1,488 (0.07 %)** | **37.3 %** |

The last row is the whole competition in one line: 0.07 % of the rows carry
more than a third of the squared error. In July 2025 the five largest targets
are 87,000-88,000 s at LIRF: the recorded block time is the *previous day*.

### 1.2 That class is identifiable in the ranking file

`AIRCRAFT_TYPE_mvt` is null on exactly those rows and almost nowhere else
(July 2025: 339 null rows, 337 of them LIRF and no-NM). In the live ranking
file the class is 388 rows — 383 LIRF, 4 LTFM, 1 LSZH — 0.11 % of the 344,841
scored departures. Predicting them with the global median would cost about
414 s of RMSE **on its own**, which is more than the entire score of the top
three teams. Every team below ~400 s must already be handling them.

### 1.3 Half of that class has a computable label

`BLOCK_TIME_UTC_mvt` is within 6 s of `SCHED_TIME_UTC_mvt` on **47.8 %** of
the 1,507 noac rows, so on those the label is
`MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` to within 6 s, which the ranking file
gives us. Under the softer criterion the head actually uses — the ratio
`y / (MVT - SCHED)` above 0.5 — the share is 53.0 %; *exact* equality is only
4.5 %. This paragraph said "to the second on 53 %" before review round 1;
`code/verify.py` prints all three numbers. The ratio `y / (MVT - SCHED)` over the class is bimodal
at ≈0.11 and ≈1.00 (quartiles 0.079 / 0.169 / 0.999 / 1.000 / 1.004 at the
5th, 25th, 50th, 75th and 95th percentiles). On the 1,488 noac rows with
`MVT - SCHED > 60 s`, the constant `MVT - SCHED` scores **5,374 s** and the
class mean 11,217 s. An oracle that splits on `ratio > 0.9` and predicts
`MVT - SCHED` on one side and the other side's mean on the other scores
**3,121 s** (3,106 s at `ratio > 0.5`); the population and criterion for these
two were unstated before review round 1.

An oracle on the *hard* criterion is worse than doing nothing: splitting on
`|y - (MVT - SCHED)| <= 6 s` and predicting the other side's mean gives
**7,810 s**, against 5,374 s for the constant. That is appendix B's lesson
arriving before the model: on this class the hard label is the wrong
partition, and no amount of skill at predicting it can repair that.

### 1.4 The same artefact runs through the whole file

4.666 % of *all* 2025 departures (97,280 rows, 96,559 of them not noac) have
`|BLOCK_TIME - SCHED_TIME| <= 6 s`. **Exact equality is only 0.39 %** (8,123
rows) — `BLOCK_TIME` carries a non-zero second on 93.6 % of departures, and
the difference is a flat plateau across -6 to +5 s with a cliff at ±7 s. So
this is not the schedule field being copied verbatim. The histogram of
`BLOCK - SCHED` over -8 to +8 s settles what it is:

```
-8   -7    -6    -5    -4    -3    -2    -1     0    +1    +2    +3    +4    +5    +6   +7   +8
331  321  6238  8037  7997  7984  8046  8038  8123  8106  8171  8031  8161  8074  2274  335  344
```

a flat plateau of about 8,000 rows at **every** second from -5 to +5, half
weight at ±6, and a cliff to the ~330-per-second background at ±7. The class
is `BLOCK = SCHED + U(-6, +6)`: a deliberate six-second jitter, not a copy.
For the metric the distinction does not matter — a uniform ±6 s error has RMSE
3.5 s, which is exactly the residual measured on the class — but the mechanism
sentence was wrong and is corrected here. The rate is very uneven:
LIRF 18.1 %, LEBL 5.6 %, LTFM 5.1 %, EDDM 5.0 %, EGLL 4.9 % against EDDF
0.75 %, LSZH 0.75 %, EHAM 0.79 %. On those rows the naive `MVT - AOBT_3`
predictor has RMSE 845 s against 442 s elsewhere, and they carry **13.5 %** of
that predictor's squared error while being 4.1 % of the rows — *those four
figures are July only*; over the twelve months they are 598 s, 371 s, 11.3 %
and 4.6 %.

### 1.5 For the ordinary rows the task is a clock-definition offset

`MVT - AOBT_3` scores 466 s on July 2025. Its error is
`BLOCK_TIME - AOBT_3_flt`: median 5 s, but only 20 % of rows within a minute
and 1st-99th percentile -1,616 s to +840 s, with a per-airport RMSE from 282 s
(LSZH) to 771 s (LIRF). The two systems are *not* out of sync as clocks — on
arrival rows `MVT_TIME - ARVT_3_flt` is within 60 s on 90 % of rows — so this
is a definitional difference (push-back vs. chocks-off, or an NM update), and
it is a function of airport, stand, operator and surface state.

## 2. The model

```
base   = MVT_TIME - AOBT_3_flt                    when the NM row joins (98.5 %)
       = smoothed median taxi for (stand, runway) otherwise
z      = TAXITIME_SEC - base, clipped to +/-6000 s for training only
main   = LightGBM (1200 rounds, 255 leaves, lr 0.05) blended 50-50 with
         CatBoost on GPU 0 (3000 iterations, depth 8), both on z
p      = P(BLOCK_TIME == SCHED_TIME | x), LightGBM binary, 300 rounds
pred   = p * (MVT - SCHED) + (1 - p) * (base + main)
ghost  = for the null-aircraft-type class only:
         pa = P(y / (MVT-SCHED) > 0.5 | x), weighted by (MVT-SCHED)^2
         pred = pa * (MVT - SCHED) + (1 - pa) * base
```

The main head is trained on ordinary departures only: ghosts are removed, and
so are the rows the SCHED head claims, because the mixture would otherwise
count them twice. Features are in `code/features.py` and are month-local and
ranking-legal by construction (verified: rebuilding July masked and unmasked
gives bit-identical features).

## 3. The scored line

`code/run.py --tag fold4-final`, **one seed** (`results/fold4-final.json`
records `seeds: 1`; the candidate in §4 is two seeds, so the board estimate
below is the one-seed fold line applied to a two-seed artefact). Folds are
`prc.data.FOLDS`, assembled through `code/dataset.py`'s per-month cache, which
is built from `prc.data.mask_like_ranking` — equivalent to `fold_frames`, not
a call to it. Segment columns re-derived by `code/verify.py`.

| fold | RMSE | y <= 3600 s | y > 3600 s | n | **no-NM** | n | noac | n | bulk |
|---|---|---|---|---|---|---|---|---|---|
| july (forward, 6 months train) | 305.12 | 264.18 | 2616.45 | 656 | **1253.51** | 3,938 | 3919.86 | 339 | 225.89 |
| december (forward, 11 months) | 207.66 | 194.13 | 1867.72 | 261 | **656.42** | 1,649 | 1903.65 | 89 | 188.64 |
| january (backward, winter) | 343.06 | 199.77 | 6068.43 | 325 | **2862.33** | 1,435 | 3969.92 | 63 | 194.38 |
| july_loo (ranking analogue) | 293.65 | 251.98 | 2583.42 | 656 | **1184.57** | 3,938 | 3671.25 | 339 | 220.72 |
| **november_loo (held out of all selection)** | **224.11** | 188.63 | 3278.10 | 222 | **1257.85** | 1,356 | 5877.30 | 55 | 186.37 |

`bulk` is the NM join with an aircraft type and a target at or below 3600 s.

**How much of this line is selection.** Review round 1 was right to press on
this. The configuration was chosen by ablation on the **july** fold (clip
±3600 then ±6000, the CatBoost blend) and the ghost head was chosen on a
four-fold mean over eight variants — i.e. on all four folds reported here.
Each individual step is at or below the repo's own single-seed measurability
threshold of ~3 s (`docs/LEDGER.md`, exp-005 seeds): clip ±3600 1.95 s, ±6000
2.48 s, ghost mixbase 1.70 s, CatBoost blend 2.49 s. Only one number in the
experiment clears that threshold on its own — the ±3600 training clip on the
bulk, 281.05 → 273.92 = 7.13 s. So **roughly 2 to 4 s of the 305.12 is
max-selection**, and the four-fold line should be read as an upper estimate of
this configuration's quality.

**The held-out fold.** `november_loo` is in `prc.data.FOLDS` and no part of
this experiment — no ablation, no ghost screen, no curve — ever looked at it.
The final configuration, run once on it (`code/run.py --tag holdout-november`,
one seed), scores **224.11**, with a bulk of 186.37 and a noac segment of
5877.30 on 55 rows. That sits where a calm autumn month should sit, between
december (207.66) and july_loo (293.65), so the configuration is not a july
artefact: the heads and the clip transfer to a month the selection never saw.
The one place it is visibly worse than the tuned folds is the noac segment
(5877 against 1904 on december and 3671 on july_loo) — 55 rows, and the ghost
head was the one component chosen on a four-fold mean, so this is exactly
where the selection bill should land and it does.

For scale, the only other `november_loo` number I can find in the repo is
exp-001's round-budget scan (`experiments/exp-001-retro-v1/curve-scan-november_loo.json`),
whose minimum over 1,660 rounds is 300.37. That is one lineage's plain-tree
scan and not the blended head, and retro-v1 *selected its round budget on that
fold*, so it is a loose reference rather than a matched comparison.

Out-of-fold predictions: `oof/{july,december,january,july_loo}.parquet`
(columns `MVT_ID_mvt`, `fold`, `y_true`, `y_pred`).

**Reading the spread.** The two ranking-analogue folds are `july_loo` and
`january`; the ranking file is 56 % July and 44 % January by row count, so the
crude expectation for a leaderboard score is
`sqrt(0.56*293.65^2 + 0.44*343.06^2) = 316 s` — subject to whatever 2026
differs from 2025 by. That would sit inside the pack at 302-361 and behind the
three teams at 270-278.

**Where the remaining error is.** January is the interesting fold: its bulk is
the best of the four (199.77, better than calm December's 194.13 by only 5 s)
while its tail is the worst by a factor of three (6068 over 325 rows, about
two thirds of that fold's squared error). Winter does not make ordinary taxi
harder to predict — the in-month features track a disrupted surface fine — it
makes the *artefact* population larger and wilder. Any further work on this
lineage should go at the January tail, not at the January bulk.

## 4. The candidate

`code/submit.py`, fitted on all twelve 2025 months, two seeds averaged, same
configuration as the scored line.

```
path    submissions/candidates/exp-018-fresh.parquet   (submissions/ is git-ignored)
sha256  e71b297fe620b9da9e8094196b78f7413f4bab23f7e25d0a60cd708f3bc7e5ca
rebuild PYTHONPATH=. .venv/bin/python experiments/exp-018-fresh/code/submit.py \
          --rounds 1200 --clip-z 6000 --exact-rounds 300 \
          --ghost-mode mixbase --main-model both --seeds 2
rows    344,841, IDs and order identical to submitting.parquet, float64, all finite
preds   mean 1038.2, median 960.7, min 30.0, max 111,562.3; 860 above 3600 s, 68 below 60 s
```

Board estimate with the desk's convention (`prc.data.rmse_clean`, weights
0.5571 july_loo / 0.4429 january): **316.48 all rows, 265.78 clean**. Per-fold
breakdown in §3; the comparison against the repo's own head is in
`COMPARISON.md`.

Segment RMSE on the july fold, for the ledger's columns: no-NM rows 1253.5
(n=3,938), null-aircraft-type rows 3919.9 (n=339), no-NM but not null-type
521.5 (n=3,599), and the bulk (NM join, aircraft type present, target at or
below 3600 s) **225.9** on 186,272 rows.

## 5. Review

Round 1, `claude-opus-5-xhigh-2026-09-06`, a fresh-context Opus substitute for
GPT-5.6 Sol, which was not available.
**Verdict: NEEDS REVISION**, five blocking findings, all accepted.

The reviewer executed the code rather than reading the prose and cleared the
things that would have sunk the experiment: no leakage by any path (refitting
the july fold with the validation labels blanked, and again permuted, leaves
the fold RMSE identical to thirteen digits), the masking invariance holds
bit-for-bit, folds and masking are the shared contract's, and the candidate
matches the template in ID set, order, dtype and finiteness with the SHA as
reported. The blocking findings were about the *record*: a wrong mechanism
claim for the schedule class that manufactured a disagreement with the repo
that does not exist, an undisclosed selection of the configuration on the
folds it was reported on, a seed count that was wrong in the headline, a
confounded clip measurement, and a missing required column. Every one is
corrected above; the full text and the disposition are in
`review/round1-opus.md`. Corrections to the six already-appended ledger rows
are appended as new rows, since that file is append-only.

## 6. Climb

`climb.log`, one line per measured attempt. The trajectory on the july fold:
320.33 (first working pipeline) → 314.25 (v3 features + the SCHED head)
→ 312.30 (training-residual clip) → 309.82 (clip widened to 6000)
→ 305.12 (ghost mixbase head + CatBoost blend).

## Appendix A — hypotheses tested and killed before modelling

Each of these was a cheap screen run before any fit; recording them so the
lane does not pay for them twice.

* **"`AOBT_3` with a non-zero second is the airport's own block time copied
  into the NM record, so `y = MVT - AOBT` exactly there."** `AOBT_3` is
  minute-quantised on 96.5 % of rows and `BLOCK_TIME` on 6.4 %, which made the
  hypothesis attractive. It is false: the 2.4 % of rows with a non-zero AOBT
  second are 86 % EHAM, and on them `|y - r| <= 2 s` holds for 1.0 % of rows
  against 3.3 % elsewhere — the opposite direction. Their residual RMSE is
  704 s against 389 s. Not used.
* **"`BLOCK_TIME` also defaults to `EOBT_1` or `LOBT`, so those deserve
  anchors of their own."** Partly true and not worth a head: after removing
  the rows that already match `SCHED`, `BLOCK == EOBT_1` (±6 s) is 2.1 % of
  rows but only 1.8 % of the baseline's squared error, and its residual RMSE
  (367 s) is *below* the population average, i.e. those rows are already easy.
  `LOBT` adds 0.3 %. Only the `SCHED` anchor earns a head.
* **"exp-006's parameter-free residual guard would catch this model's worst
  predictions."** Measured after review round 1 raised it (finding 15) and
  **negative**: clipping the implied `r = (MVT - AOBT) - y_hat` to the range
  of the same quantity over each fold's training departures ([-20,657,
  +9,069] s on all four) moves **zero rows on all four folds**
  (`code/verify.py`). The model does emit large upward moves on ordinary rows
  — the worst is `pred - (MVT - AOBT)` of +10,892 s on july and +11,300 s on
  january, and 14 candidate rows with an NM join and a known aircraft type are
  predicted above 7,200 s — but every one of them is inside what 2025 training
  rows actually exhibit, so the guard has nothing to bite on. The exposure is
  real and the repo's instrument does not measure it; a tighter guard would
  need a parameter, which is what exp-006 spent a review round removing.
* **"The stand's previous arrival in-block is a hard upper bound on the taxi
  time."** It is violated on 11.5 % of July departures (towing, stand changes,
  a missing arrival record), and the median slack is 5,045 s, so it is a
  feature and not a constraint. Kept as `stand_prev_arr_inblock_dt`, which is
  also the strongest single discriminator inside the ghost class.
* **"Airport and Network Manager clocks are offset."** No: on arrival rows
  `MVT_TIME - ARVT_3_flt` is within 60 s on 90 % of rows (per-airport median
  between -43 s and +17 s). The off-block disagreement is definitional, not a
  clock error, which is why it is modelled per stand and per operator rather
  than per airport.

## Appendix B — two things I got wrong

* **A posterior mixture for the ghost class is much worse than a ratio
  model** (july fold, RMSE on the 336 ghost rows: 8,308 s against 4,013 s;
  `MVT - SCHED` alone is 5,767 s). The mixture labels a row "exact" only when
  `|y - (MVT-SCHED)| <= 6 s`. The five largest ghosts in July have ratios of
  1.05 to 1.19 — near one but not exact — so they land on the "not exact" side
  and get pulled to the class mean of 3,206 s while the truth is 87,000 s. The
  ratio model has no such cliff. The same mixture *is* right for the ordinary
  population (Appendix C), where the near-one-but-not-exact rows do not exist.
  Lesson: a hard label on a continuous quantity is only safe where the
  quantity is genuinely bimodal, and the ghost ratio is bimodal in density but
  not in support.
* **Giving the ghost head more training rows makes it much worse.** Adding all
  departures with `MVT - SCHED > 7200 s` takes the ghost training set from
  527 to 16,162 rows on the july fold and the ghost RMSE from 4,013 to 8,569;
  the four-fold means are 3,438 (ghosts only) against 8,285. Even restricting
  the addition to LIRF costs 1,300 s. The class is a distinct generative
  process — a record with no aircraft type and no NM join — and rows that
  merely look like it in one coordinate carry the wrong conditional mean.

## Appendix C — the july ablation table (1200 rounds, forward july fold)

The ghost head dominates the total, so read `lo` (rows at or below 3600 s) for
the bulk and `hi` for the tail. Rows marked *mix* carry the bad ghost head and
are comparable with each other, not with `C`.

| config | rmse | lo | hi | reads |
|---|---|---|---|---|
| C ghost=ratio, SCHED head, no clip | **314.25** | 268.37 | 2800.7 | the working model |
| D ghost head off | 531.37 | 243.07 | 8060.3 | the ghost head is worth 217 s; it also *costs* 25 s on `lo`, because it over-predicts the small-target ghosts |
| A ghost=mix | 438.24 | 281.05 | 5740.2 | see Appendix B |
| B ghost=mix, no SCHED head | 439.23 | 282.67 | 5739.1 | the SCHED head is worth 1.6 s on `lo` once the main model is allowed to train on those rows itself |
| E ghost=mix, train residual clipped to ±3600 s | 436.76 | **273.92** | 5806.8 | **−7.1 s on the bulk**: the largest single main-model gain in the table |
| F ghost=mix, clipped to ±1200 s | 456.07 | 288.30 | 6032.5 | too tight; the clip is a sweet spot, not a monotone knob |
| G ghost=mix, 511 leaves | 437.53 | 280.18 | 5736.8 | capacity is flat |
| H ghost=mix, 127 leaves | 437.84 | 280.66 | 5736.9 | capacity is flat |

Two things this says. First, **the tail and the bulk want opposite things** and
have to be separated by a head, not by a loss: the ghost head buys 217 s on the
total while losing 25 s on the bulk, and clipping the training residual buys
7 s on the bulk while losing a little on the tail. Second, **none of the usual
knobs matter**: 127, 255 and 511 leaves are within 0.9 s of each other, and the
rounds curve turns at 500 (264.09 / 263.56 / 264.55 on non-ghost rows at 250 /
500 / 750). Everything in this experiment came from the structure of the label,
not from the model.

## Appendix D — the git-ignored inputs this experiment depends on

COMMON.md requires a path and hash for artifacts that live outside git. Every
number here is a function of the per-month feature cache:

```
artifacts/exp-018-fresh/cache/{2025-01..2025-12,2026-01,2026-07}.parquet
artifacts/exp-018-fresh/cache/vocab.json
```

built by `code/dataset.py::month_table(..., rebuild=True)` from the live raw
files. Two warnings for whoever touches it. First, the cache and `vocab.json`
are **one artifact**: `_vocab_encode` assigns integer codes from an append-only
vocabulary, so deleting the vocabulary while cached months survive will hand
the next build colliding codes **with no error**. Delete both or neither.
Second, the cache is rebuildable but not byte-pinned here — the SHAs below are
of the current build, and a rebuild on a different pandas would need
re-verifying rather than assuming.
e97d2996ffc848ed  artifacts/exp-018-fresh/cache/2025-01.parquet
326bc6050fc5f1c0  artifacts/exp-018-fresh/cache/2025-02.parquet
e7615f678bd051cf  artifacts/exp-018-fresh/cache/2025-03.parquet
7502e049c49c2679  artifacts/exp-018-fresh/cache/2025-04.parquet
775cf69f8c8b7008  artifacts/exp-018-fresh/cache/2025-05.parquet
afeaffef8df947f5  artifacts/exp-018-fresh/cache/2025-06.parquet
d4a8e791c2df83af  artifacts/exp-018-fresh/cache/2025-07.parquet
897bb91ac60b4f24  artifacts/exp-018-fresh/cache/2025-08.parquet
68fca853fdfd8703  artifacts/exp-018-fresh/cache/2025-09.parquet
dc53e196bf46faad  artifacts/exp-018-fresh/cache/2025-10.parquet
0113700b381e5462  artifacts/exp-018-fresh/cache/2025-11.parquet
9e5cb370739213e4  artifacts/exp-018-fresh/cache/2025-12.parquet
1b47489b483e02ff  artifacts/exp-018-fresh/cache/2026-01.parquet
f9b2636cd0599ccb  artifacts/exp-018-fresh/cache/2026-07.parquet
ef056bb16b3db001  artifacts/exp-018-fresh/cache/vocab.json
