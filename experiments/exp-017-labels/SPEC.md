# exp-017-labels — training-label hygiene

**Status: acked by the desk. Two additions folded in, marked (desk).**

## Reading of the assignment

The assignment asks whether removing or correcting artefact training labels
buys board RMSE. I read "the recipe" as *per head*, not as one mask over the
training frame, and the rest of this SPEC follows from that reading. The
readings I rejected: (1) one global `keep` mask, which is what the first screen
ran and which is unsound — see below; (2) hygiene applied to validation as well
as training, which would change the metric rather than the model and make the
folds incomparable with every other row in the ledger.

The first screen (`work/screen.log`, seed 7, one global mask) is **void as a
measurement of the recipe**. It applied `rollover="drop"` to the whole training
frame, which removes 11 of the 50 rows the v4 noac band head is fitted on and
silently collapses every band cell to p = 0. Its july win is not hygiene: 8 of
july's 12 validation band rows are placeholders, which p = 0 predicts correctly
by accident, and january, whose band skew differs, lost 3.83 s. The numbers are
kept below only as the record of what that bug produces.

## What each head is fitted on, and what the recipe does to it

Counts are the whole 2025 training year, from
`artifacts/exp-017-labels-catalogue.parquet` (2,085,047 rows with a target).

| head | its training rows | recipe: drop | recipe: correct | keep, and why |
|---|---|---|---|---|
| **bulk `nm`** (plain tree, offset from `AOBT_3`) | 2,062,577 | 137 non-positive, 22 duplicate | 1 detectable rollover, `y -= 86,400` | — |
| **bulk `nonm`** (plain tree, direct taxi) | 20,963 | 2 undetectable rollovers, 248 non-positive, 18 duplicate | none detectable | — |
| **noac offset** (classifier + regressor) | 1,507 | 3 non-positive | none | **all 12 noac rollovers.** They are ordinary structure for this head: it predicts an offset from `SCHED`, and for a rollover row that offset is small, not extreme. |
| **noac band** (two-point mixture, `mvt_sched >= 8 h`) | 50 | **nothing** | nothing | **all 11 rollovers among the 50.** They *are* the signal: `p(rollover)` is estimated from them, so dropping them sets every band cell to 0 and turns the mixture into a constant. |

Why the split is not cosmetic: the one detectable rollover sits in the `nm`
head at y = 87,002 s among neighbours near 700 s, where a single leaf mean is
badly distorted; the undetectable ones sit almost entirely in the `noac` band
head, where they are the thing being fitted. The two facts point opposite ways,
so a single mask must be wrong for one of them.

Non-positive targets are 388 rows with max |y| = 12 s (the 369 "impossible"
rows are a strict subset, so the classes are one treatment, not two). They
carry **0.062 %** of the training SSE about the mean, and the 40 duplicates
another **0.116 %**. The honest prior is that arm C is worth well under 1 s of
board RMSE: it is a cheap confirmation, not a lever. The 15 rollover rows carry
**17.6 %** of that same SSE, so arm A is the whole of this lane.

## Arms

| arm | bulk `nm` / `nonm` | noac offset | noac band | tests |
|---|---|---|---|---|
| **0 control** | untouched | untouched | untouched | the delivered model, hygiene off |
| **A** rollovers | drop 2 nonm, correct 1 nm | untouched | untouched | does an 87,002 s target hurt the bulk tree |
| **C** non-physical | drop negative + impossible + duplicate | drop 5 | untouched | the near-zero-SSE classes |
| **A+C** | both | drop 5 | untouched | the recipe as it would ship |
| **A1** unreached | drop the 3 `nonm` rows in `D.UNREACHED_ROLLOVER` | drop the 1 `noac` row (not a band row: its `MVT-SCHED` is 1,504 s) | untouched | the sharper arm A: drop only rows **no input can reach** |
| **W** recency | untouched, `w = 0.5^(age/180 d)` | same | untouched | is 2025-12 worth more than 2025-01 |
| **A+C+W** | both + weights | both | untouched | recipe and weighting together |

The band head is untouched in every arm. That is the design, not an omission:
`clean_training_labels` is applied through `fit(keep=...)`, and `keep` must not
reach `_fit_band`. That is now the documented contract of `keep` in
`prc.tail_model` 2.4.0.

**(desk) The band-cell identity check runs and prints first, before any fold
RMSE.** Arm 0 and arm A must produce byte-identical `band["cells"]`,
`band["bands"]` and `band["offset"]` on every fold. If they differ, the run
aborts: the whole measurement is void, exactly as the first screen was.

### Arm A1, added after the ack

Added after reading `prc.data` 4's `UNREACHED_ROLLOVER` (exp-020), which was
written into the shared module while this SPEC was being drafted. It changes no
arm above; it adds one, and the reason is that exp-020 measured two things this
SPEC had wrong:

1. `label_artefacts`' `|y - 86,400| < 3600` window **misses two rollover-class
   rows**, at 58,206 s and 31,268 s. Arm A therefore under-drops.
2. Of the 14 "undetectable" rollovers, **12 are reached** by `MVT - SCHED`
   above 10,000 s — the path the noac band head routes them through. This is
   independent confirmation of the per-head table above, from another lane.

A row *no input can reach* is pure noise to a learned head; a reachable
rollover is structure. A1 drops only the first kind, which is the treatment the
per-head reasoning actually argues for. A stays in as the acked arm so the two
can be compared.

### January's `rmse` cannot read this experiment

exp-020 measures the two unreached january rows at **116 s of january RMSE**.
January validates 2025-01, which holds 2 of the 4 unreached rows; july and
july_loo validate 2025-07, which holds none, so for them `rmse == rmse_clean`.
Any training change that moves the prediction on those two rows moves january's
`rmse` by far more than the effect under test — which is the most likely reading
of the void screen's 3.83 s january "loss". **On january the deciding metric is
`rmse_clean`.** Both are reported for every arm regardless.

## Folds and seeds

- Folds: `july` and `january` (the lane's two reported folds) and `july_loo`
  (the shape the submission model is in). Reported per fold, never pooled.
- Seeds: **7, 23, 51**, mean and spread reported. The first screen ran one seed
  and produced a 3.83 s january gap it cannot separate from seed noise; three
  seeds is the cheapest thing that can tell a real effect from a draw.
- Metric: `D.rmse` on the fold's own validation rows, plus `rmse_clean`.
- Gate for shipping any arm: mean improvement on **both** `january` and
  `july_loo` exceeding the across-seed spread. A one-fold win does not ship.

**(desk) Half-life is 180 days only.** 90 and 365 are added beside it *only if*
W moves anything at 180, and not before.

## Cost

18 fits (6 arms x 3 folds) x 3 seeds, ~170-320 s each at 6 threads under the
`docs/COMPUTE.md` compute allocation rule, one fit at a time: roughly 4-6 hours wall clock,
run detached. It starts only once the exp-009 v4.1 measurement releases the
single-fit slot.

## The void screen, for the record

Seed 7, one global mask, the band-starving bug above. Not evidence for or
against the recipe.

| arm | july | january |
|---|---|---|
| baseline | 318.77 | 345.71 |
| a: drop rollovers | 313.80 | 349.54 |
| c: drop non-positive | 318.38 | 344.99 |
| a+c+dup | 313.96 | 349.51 |
| recency hl=180d | 317.59 | 346.52 |
| full + recency | 313.30 | 349.70 |

The one arm whose sign is consistent across both folds is C, at -0.39 s (july)
and -0.72 s (january) — consistent with the near-zero SSE those rows carry, and
the only line in the table not contaminated by the band bug.
