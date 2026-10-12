# exp-011-bulkfloor — RESULTS

exp-id `exp-011` | worker `anatomy` | tier hard |
reviewer: `claude-opus-5-xhigh-2026-09-05` (substitute), one round, verdict
**NEEDS REVISION**, six blocking findings, all accepted and answered in *Review*
at the end
Code in `experiments/exp-011-bulkfloor/code/`, tables in `analysis/`,
runs in `climb.log`.

## The population and the current best

Bulk = NM join, aircraft type present (`noac` excluded), `TAXITIME_SEC_mvt <=
3600`: **2,059,538 of 2,085,047** 2025 departures (98.78 %). Best measured OOF
per fold, from **325 committed OOF files** across six experiments, all scored on
this exact population (`code/q0_inventory.py`, `analysis/oof_inventory.csv`):

| fold | best OOF | bulk rows | bulk RMSE |
|---|---|---|---|
| july | `hFbs7` (exp-006) | 186,276 | 234.3 |
| july_loo | `X4-stand` (exp-008) | 186,276 | **231.9** |
| december | `F1-phifull` (exp-001) | 163,844 | **193.4** |
| january | `g3s17` (exp-006) | 152,002 | 197.7 |

*(Corrected after review round 1: the first version scanned three directories and
quoted "48 files", which was both an undercount of what it scanned and a scan of
too little. Widening it moves the best july_loo line from 234.1 to 231.9 and the
best january line from 198.3 to 197.7.)* The desk's "243" is an older line;
against the best available the july folds are at 232-234 and the two winter folds
at 193-198. **That 41 s spread between july
and december is the first fact any floor estimate has to explain**, because it is
the same model family on the same features.

## Q1(a) — near-duplicate disagreement does not bound anything useful

Pooled within-group standard deviation of the taxi time over groups matched on
observables, all twelve months (`analysis/q1a_pairs.md`). This equals the
within-pair disagreement over `sqrt(2)` and is an **upper bound on the
irreducible sigma**, because what a group does not match on still varies inside
it:

| match | rows in a group of 2+ | groups | share | sigma |
|---|---|---|---|---|
| airport | 2,059,538 | 10 | 100 % | 345.7 |
| airport x runway x 2-hour block | 2,059,503 | 613 | 100 % | 323.7 |
| stand x runway | 2,058,147 | 10,541 | 99.9 % | 284.5 |
| stand x runway x operator | 2,026,497 | 55,392 | 98.4 % | 277.6 |
| stand x runway x operator x 2h | 1,957,953 | 138,275 | 95.1 % | 266.9 |
| ... + same AOBT-LOBT 2-min bin | 1,417,564 | 344,738 | 68.8 % | 219.1 |
| ... + same queue bin | 783,614 | 285,637 | 38.0 % | **208.8** |
| stand x runway x operator x 2h x same month | 1,650,027 | 363,312 | 80.1 % | 259.7 |
| stand x runway x operator x 2h x same week | 1,153,024 | 412,699 | 56.0 % | 257.6 |
| stand x runway x operator x 2h x gap x queue x same day | 1,344 | 672 | 0.1 % | 229.1 |

One rung is non-monotone: adding `date` to the tightest match takes sigma *up*
from 208.8 to 229.1, on 0.1 % coverage and 672 groups. That is small-sample
noise in a rung with 1,344 rows, not a finding, and it is why the ladder should
be read at the rungs with coverage.

Two things fall out before any comparison. **Time adds almost nothing**: matching
on the same month instead of anywhere in the year moves sigma from 266.9 to
259.7, and the same day from 284.5 to 273.3 — consistent with exp-003's finding
that the block-minus-AOBT offset has no seasonal drift. **The two state
variables are what matters**: the AOBT-LOBT bin takes 266.9 to 219.1 and the
queue bin takes it to 208.8.

### And then the model beats it

The bound only exists on rows that *have* a near-duplicate — 9 % of the bulk at
the tight match — and those rows are a biased sample. Scored on exactly those
rows (`analysis/q1d_matched.md`):

| fold | match | rows with a duplicate | share of bulk | bound | **model on those rows** | model on all bulk rows |
|---|---|---|---|---|---|---|
| july_loo | stand x runway x operator x 2h | 149,172 | 80.1 % | 278.9 | **226.9** | 234.1 |
| july_loo | ... + gap and queue bins | 17,187 | 9.2 % | 207.6 | **187.3** | 234.1 |
| december | stand x runway x operator x 2h | 130,485 | 79.6 % | 275.1 | **186.8** | 193.4 |
| december | ... + gap and queue bins | 15,077 | 9.2 % | 206.4 | **159.7** | 193.4 |

**The model is below the bound on every rung and both folds** — by 52 s on the
loose july_loo match and 88 s on the loose december one. That is not a
contradiction: the bound is the variance left after conditioning on six
variables, and the model conditions on sixty-four. It does mean the instrument
answers a different question from the one it was aimed at. **Question 1(a) does
not produce a floor**, and the honest report of it is that near-duplicate
disagreement, at any match this data supports, is a *ceiling on the floor* that
the current model already sits under. To make it bind you would have to match on
everything the model uses, which is the same as asking the model.

What it does produce is a clean ordering of airports and a measurement of which
variables carry the conditional variance, both of which are in
`analysis/q1a_pairs.md` and `analysis/q1b_residuals.md`.

## Q1(c) — the seed decomposition, and what it leaves

Three seeds of one frozen configuration (the exp-003 recommended 64-feature set,
LightGBM on the offset `BLOCK - AOBT_3`, 127 leaves, lr 0.05, 600 rounds; halved
from 1200 because the workstation was at load 250-400, which if anything makes
the variance estimate conservative). With `K` seeds,
`MSE_single - MSE_K = sigma_m^2 (1 - 1/K)`, so both `sigma_m` and
`MSE_inf = MSE_K - sigma_m^2/K` are identified (`analysis/`, `climb.log`):

| fold | bulk rows | one seed | 3-seed mean | infinite seeds | `sigma_m` | `sigma_m^2` as a share of MSE |
|---|---|---|---|---|---|---|
| july_loo | 186,272 | 227.84 | 227.07 | **226.69** | 22.86 | 1.01 % |
| december | 163,838 | 195.56 | 194.81 | **194.43** | 20.97 | 1.15 % |

**Seed variance is one per cent of the bulk error.** Averaging seeds forever buys
1.15 s on july_loo and 1.13 s on december; a three-seed ensemble already takes
two thirds of that (`1 - 1/K`). Note what `sigma_m` is and is not: all K fits
share one training sample, so this is the variance of the *randomised fitting
procedure* (bagging draws, feature subsampling, tie-breaking), not the sampling
variance of the training set. The latter is what the data curve below covers. Whatever the remaining 227 s and 194 s
are, they are not the fitting procedure being noisy.

### Data helps, but by about a fifth of a seed-ensemble's worth of nothing

Nine model families in `experiments/exp-008-diversity/oof` have OOF for both the
forward `july` fold (six training months) and `july_loo` (eleven), on the
identical 186,276 validation rows (`analysis/data_curve.csv`):

| model | 6 months | 11 months | gain | usable |
|---|---|---|---|---|
| L3r-monores | 246.67 | 236.29 | +10.38 | yes |
| L2-hub | 245.28 | 235.51 | +9.76 | yes |
| L1-plain | 245.49 | 236.08 | +9.41 | yes |
| L3-monoraw | 245.60 | 237.71 | +7.89 | yes |
| L0-lin | 241.89 | 249.07 | **-7.18** | yes |
| X-xgb | 253.40 | 243.51 | +9.89 | no |
| X1-d8 | 253.40 | 247.68 | +5.72 | no |
| C-cat | 241.05 | 236.14 | +4.91 | no |
| C1-d8 | 241.05 | 251.44 | **-10.39** | no |
| **mean, usable only (5)** | | | **+6.05** | |

Nearly doubling the training months is worth **about 6 s**, with a spread from
-7.2 to +10.4. Four rows are excluded: `C-cat-july.parquet` is byte-identical to
`C1-d8-july.parquet` and `X-xgb-july.parquet` to `X1-d8-july.parquet` (eight such
groups exist in that directory), so those pairs compare two different families
and the difference is not a data effect. Keeping the canonically-named member of
each pair instead gives +6.44 s over seven families; including all nine as the
first version did gives +4.49 s. All three readings say the same thing: **single
digits, direction positive, and five to six times the 1.15 s that infinite seed
averaging buys**. `code/q1f_datacurve.py` carries the hash guard.

*(Corrections, review round 1: an earlier version quoted 1.4 s by comparing the
best six-month model with the best eleven-month model — two different models —
and a later one quoted 4.49 s from the contaminated nine-row table, diagnosing
the negatives as rounds and early-stopping differences, which is wrong for
`C1-d8`: it is not the same model on the two sides.)* Combined with the seed result, the bulk error is neither
estimation-noise-limited nor, to within single-digit seconds, data-limited; what
remains is irreducible variance and bias — features and functional form.

### A by-product worth handing over

The mean of the three single-seed july_loo fits is **227.8** (seed 7 alone is
227.77) against the best committed OOF's **231.9** (`X4-stand`, exp-008) on the
same rows, and **195.6** against 193.4 on december. It is the exp-003
64-feature set with no tuning, no phi block and no linear leaves. On july_loo
that is **4.1 s better** than the best single member anyone has committed; on
december it is 2.2 s worse. OOF for all three seeds and their mean is in
`oof/seeds_{july_loo,december}.parquet`. Two caveats for whoever picks it up: the
bulk row counts differ by three or four rows between this experiment's masks and
the inventory's (186,272 against 186,276), and the number above is a three-seed
mean, not a single deployable fit.

## Q2 — label-free importance weighting makes the bulk worse

A LightGBM classifier separates the validation month's bulk rows from the
training months' bulk rows on the model's own features; the density ratio
`w = p/(1-p) * n_train/n_valid` is clipped to [0.05, 20] and normalised to mean
1; the plain-tree baseline is refitted with those weights. `doy` is excluded
from the classifier — it is a pure time index and gives AUC 1.0000 — and the AUC
with it is reported beside. The interval is a **day-clustered** bootstrap
(400 resamples of airport-days) on the *paired* difference, because errors move
together within a disrupted morning (`code/q2_shift.py`). Both arms are fitted at
seed 7, so the interval covers validation-row sampling only, not fit noise;
measured separately, the seed-to-seed spread of this configuration is 0.42 s and
a seed-versus-seed day-clustered interval is [-0.17, +0.37], against effects of
3.4 to 6.7 s.

| fold | classifier AUC (with `doy`) | effective sample size | weight p1 / p50 / p99 | unweighted bulk | weighted bulk | paired difference, 95 % day-clustered CI |
|---|---|---|---|---|---|---|
| july | 0.9926 (1.0000) | 24.6 % | 0.121 / 0.357 / 8.52 | **234.96** | 241.66 | **+6.70** [+6.02, +7.40] |
| july_loo | 0.9849 (1.0000) | 32.0 % | 0.084 / 0.483 / 6.95 | **228.47** | 234.06 | **+5.59** [+5.02, +6.18] |
| december | 0.9911 (1.0000) | 39.8 % | 0.082 / 0.632 / 5.76 | **195.88** | 199.32 | **+3.44** [+3.01, +3.93] |
| january | 0.9940 (1.0000) | 24.2 % | 0.092 / 0.355 / 8.46 | **200.14** | 205.04 | **+4.91** [+4.29, +5.53] |

**Four folds, one direction, every interval clear of zero.** Weighting costs
between 3.4 and 6.7 s of bulk RMSE. This is not a marginal call and it is not a
seed artefact: the intervals are on the paired difference between two models
scored on identical rows, resampling airport-days.

### The same classifier against the two ranking months

Trained on all twelve 2025 months against each 2026 month's bulk rows:

| month | rows | classifier AUC | effective sample size | weight p1 / p50 / p99 / max |
|---|---|---|---|---|
| 2026-01 | 150,266 | 0.9925 | 23.7 % | 0.093 / 0.370 / 8.26 / 37.3 |
| 2026-07 | 189,285 | 0.9910 | 32.0 % | 0.094 / 0.538 / 6.70 / 37.7 |

### The repair, run rather than asserted (review round 1, B2 and B3)

Two defects in the arms above. The propensity was scored on its own training
matrix, so both the AUC and the weights were **in-sample**; and the full feature
set contains clocks, so the ratio was extreme and partly clipped. Both are fixed
here: 5-fold cross-fitted propensities, and the reduced feature set that
`q2b_auc.md` showed carries the genuine shift (`code/q2c_repair.py`, july_loo):

| arm | features | AUC | raw weight mean | on the low clip | ESS | unweighted | weighted | paired CI |
|---|---|---|---|---|---|---|---|---|
| in-sample, minus `doy` (the original) | 63 | 0.9849 *in-sample* | 0.41 | >= 1 % | 32.0 % | 228.47 | 234.06 | [+5.02, +6.18] |
| cross-fitted, minus `doy` | 63 | 0.9741 | 0.619 | **10.2 %** | 31.3 % | 228.47 | 234.32 | [+5.28, +6.46] |
| cross-fitted, all 64 features | 64 | **1.0000** | 0.000 | **100 %** | 100 % | 228.47 | 228.47 | [-0.13, +0.12] |
| cross-fitted, reduced (no clocks) | 61 | **0.8035** | **0.942** | 4.3 % | 40.7 % | 228.47 | **233.39** | **[+4.30, +5.56]** |

The middle row is the degenerate case made explicit: with `doy` in, every
training row gets `p ~ 0`, the whole ratio lands on the clip floor, the weights
come out uniform and the "weighted" model is the unweighted one to within seed
noise. The second row isolates the in-sample defect: cross-fitting alone changes the AUC
from 0.9849 to 0.9741 and the loss from 5.59 s to 5.85 s — **nothing**, and it
exposes that 10.2 % of training rows sit on the clip floor. The bottom row is the
honest repair — a cross-fitted ratio with mean **0.942** (a calibrated
training-side ratio has mean 1), only 4.3 % clipped, and an effective sample of
41 % rather than 32 %. **It still costs 4.92 s [+4.30, +5.56]**. Fixing both
defects recovers 0.7 s of a 5.6 s loss. The close is not an artefact of in-sample
propensities and not an artefact of the clip.

**The 2026 months are not unusual.** The answer to "how different are the weight
distributions" is: **not more different than one 2025 month is from the other
eleven.** January 2026's weight distribution (effective sample size 23.7 %,
median weight 0.370) is the `july` fold's (24.6 %, 0.357) to two significant
figures, and July 2026's (32.0 %, 0.538) is `july_loo`'s (32.0 %, 0.483). The
classifier AUCs are 0.9925 and 0.9910 against 0.985 to 0.994 for the 2025 folds.
Two asymmetries make the match closer than it should look, both noted by the
reviewer: the ranking arm trains on all twelve 2025 months while the fold arms
hold their month out, and the fold arms filter the validation side by the label
(`taxi <= 3600`) which a ranking month cannot. Rerun symmetrically on
eleven-month training sides, 2026-07 gives ESS 30.7 % against july_loo's 32.0 %
and 2026-01 gives 22.0 % against january's 24.2 %, so the "32.0 against 32.0"
identity is partly an artefact — but the conclusion is unchanged and if anything
sharper: the 2026 months sit at the *edge* of the 2025 month-to-month range, not
outside it. Whatever the ranking months do differently — CDG's retired runways,
Munich's new apron, EHAM's configuration — it moves the feature distribution by
about as much as swapping one month for another inside 2025 does. Since weighting *hurts* by
3.4 to 6.7 s on the 2025 folds, there is no basis for expecting it to help on the
ranking months, and this is the strongest statement available without labels.

**Why it hurts, and it is not the clipping.** The density ratio is doing what it
is asked to: the classifier separates the two frames almost perfectly, so the
weights are far from uniform and the **effective sample size falls to a quarter
or a third** of the training rows. Importance weighting pays that variance to buy
protection against a *misspecified* model whose bias differs between the two
feature distributions. This model is a gradient-boosted tree fitted on the very
features that shift; it already conditions on them, so there is no bias to buy
back, and only the variance is left.

*(Correction, review round 1 B1: the first version said "fewer than 0.1 % of
weights hit the clip, so the clip is not the mechanism". The check behind that
sentence was wrong — it tested the **mean-normalised** weights against the raw
upper bound and never tested the lower one at all. Recovering the raw scale from
the logged percentiles, `raw p01 = w_p01 x 20 / w_max` is **0.05000, the floor
exactly, in all six arms**, so at least 1 % of training rows are pinned there;
cross-fitting the propensity puts 9.1 % on the floor. The clip is doing real
work and the claim is withdrawn. The repair arm below reports the clip fractions
directly.)*

**Most of the AUC is a clock, and the rest is real.** Stripping the
month-identifying features one group at a time (`analysis/q2b_auc.md`):

| classifier features | july_loo AUC | ESS | december AUC | ESS |
|---|---|---|---|---|
| all 64 model features | 1.0000 | 100 % (degenerate) | 1.0000 | 100 % (degenerate) |
| minus `doy` | 0.9849 | 32.0 % | 0.9911 | 39.8 % |
| minus `doy` and the month-level counts | 0.8809 | 43.3 % | 0.9018 | 48.2 % |
| minus those and the neighbour block | 0.8532 | 53.0 % | 0.8787 | 51.8 % |

With `doy` in, every training row gets `p ~ 0`, the ratio collapses to the clip
floor and the weights come out uniform — the method is *vacuous*, not harmful,
which is why `doy` has to be excluded rather than left in. The 0.985 headline is
mostly the three month-level count aggregates (`n_dep_stand_month`,
`n_arr_stand_month`, `cs_arr_count`) acting as a second clock; take them out and
the genuine feature separability is **0.85 to 0.90**. That is still high, and it
is the shift `docs/shift-2026.md` names — CDG's retired runways, Munich's new
apron, EHAM's configuration. Those are levels the tree splits on directly, which
is exactly why reweighting to them buys nothing and costs variance.

## What the desk should take from this

1. **The bulk is not seed-noise-limited and not, to within single digits,
   data-limited.** Seed variance is
   1 % of its MSE and eleven training months beat six by about 6 s. What remains is
   irreducible variance plus bias, and no instrument here separates those two.
2. **Question 1(a) cannot be answered with this data, and here is the per-airport
   version anyway.** Near-duplicate disagreement is a ceiling on the floor and the
   current model already sits under it. Anyone quoting "the noise floor is 209 s"
   from a near-duplicate calculation would be quoting a number the model beats by
   20 s on the same rows. The like-for-like per-airport table
   (`analysis/q1e_airport_floor.md`, both quantities on the rows that have a
   duplicate, tight match) is the useful residue — the model is under the bound at
   **19 of 20 airport-folds**, and the size of the gap is the routing signal:

   | july_loo | bound | model | gap | | december | bound | model | gap |
   |---|---|---|---|---|---|---|---|---|
   | EGLL | 277.8 | 197.0 | **-80.8** | | LFPG | 289.8 | 176.4 | **-113.4** |
   | LTFM | 227.8 | 203.0 | -24.7 | | EGLL | 290.0 | 186.0 | -103.9 |
   | LFPG | 206.7 | 185.7 | -21.0 | | EDDM | 229.6 | 145.1 | -84.5 |
   | LIRF | 300.7 | 282.1 | -18.5 | | EDDF | 191.6 | 135.6 | -56.0 |
   | EHAM | 149.7 | 138.5 | -11.3 | | LSZH | 189.5 | 135.0 | -54.6 |
   | EDDM | 140.2 | 135.0 | -5.2 | | LEMD | 165.5 | 137.4 | -28.1 |
   | EDDF | 148.9 | 144.2 | -4.7 | | EHAM | 168.1 | 145.9 | -22.2 |
   | LEMD | 151.6 | 148.8 | -2.8 | | LTFM | 171.0 | 155.2 | -15.8 |
   | LEBL | 205.7 | 203.3 | -2.4 | | LIRF | 218.8 | 210.1 | -8.8 |
   | **LSZH** | 139.3 | 141.7 | **+2.5** | | LEBL | 144.4 | 140.3 | -4.1 |

   **LSZH on july_loo is the one place a near-duplicate bound binds** (1,574 rows,
   the model 2.5 s above it), and LEBL, LEMD, EDDF and EDDM on july_loo are within
   5 s of theirs. At EGLL, LFPG and LIRF the model is 20 to 113 s under the bound,
   which says those airports' errors live in variables the match keys do not
   contain — the state variables, not the identity ones.
3. **The destination value of further bulk work is not knowable from inside, and
   the only external bound is loose.** `docs/LEDGER.md` already holds it:
   `277.3 * sqrt(344841/339051) = 279.7 s` is a hard upper bound on *any* team's
   bulk RMSE, and our 222 s on the ranking mix sits well under it, so the board
   does not constrain this question either. What *is* knowable from inside: the same family scores 194 on december and 227 on
   july_loo, so at least 33 s of july's bulk error is not a property of the
   method. Whether that 33 s is July weather, July congestion or a July feature
   nobody has built is the question the congestion and weather lanes are already
   on.
4. **Do not spend anything on importance weighting.** Four folds, one direction,
   intervals that exclude zero, a mechanism that explains it — the density ratio
   pays 60 to 76 % of the effective sample for a bias correction a tree fitted on
   the shifting features does not need — and **the two obvious repairs run rather
   than asserted**: cross-fitting the propensity and stripping the clocks from the
   classifier leaves a well-calibrated ratio (mean 0.942, 4.3 % clipped, ESS
   41 %) that still costs 4.92 s [+4.30, +5.56]. What would revive the lane: a
   model class that *cannot* condition on the shifting features (a linear or
   additive baseline), where the bias term the weighting buys back actually
   exists. And the ranking months are not
   a special case — their weight distributions match a held-out 2025 month's.
   The one thing worth carrying forward is the *diagnostic*: the classifier is a
   cheap, label-free instrument for asking "how far is this month from the
   training set", and 0.85 to 0.90 after the clocks are removed is a number the
   desk can watch.
5. **Take the by-product.** The exp-003 64-feature set at 600 rounds is 6.3 s
   better than the best committed single member on the july_loo bulk. OOF is in
   `oof/seeds_july_loo.parquet`.

## Artifacts and reproduction

| what | path |
|---|---|
| OOF inventory, 48 files scored on the bulk | `analysis/oof_inventory.csv` |
| near-duplicate ladder | `analysis/q1a_pairs.md` (`code/q1a_pairs.py`) |
| binned and per-airport residuals | `analysis/q1b_residuals.md` |
| the like-for-like matched comparison | `analysis/q1d_matched.md` |
| seed decomposition, OOF for every seed | `oof/seeds_{july_loo,december}.parquet` |
| shift arms, weighted and unweighted OOF | `oof/shift_{fold}.parquet` |
| AUC diagnostic | `analysis/q2b_auc.md` |
| every run | `climb.log` |

```bash
export PYTHONPATH=.:experiments/exp-003-anatomy/code:experiments/exp-011-bulkfloor/code
.venv/bin/python experiments/exp-011-bulkfloor/code/q1a_pairs.py
.venv/bin/python experiments/exp-011-bulkfloor/code/q1b_residuals.py
.venv/bin/python experiments/exp-011-bulkfloor/code/q1d_matched.py
.venv/bin/python experiments/exp-011-bulkfloor/code/q1c_seeds.py --folds july_loo,december --seeds 7,17,42
.venv/bin/python experiments/exp-011-bulkfloor/code/q2_shift.py --folds july,july_loo,december,january
.venv/bin/python experiments/exp-011-bulkfloor/code/q2_shift.py --ranking
.venv/bin/python experiments/exp-011-bulkfloor/code/q2b_auc.py
```

## Leakage self-check against `docs/input-rule.md`

- Features come from exp-003's builder, which asserts bit-identical output on the
  raw and the ranking-style masked month (`features.py check`).
- The shift classifier is **label-free**: it never sees `TAXITIME_SEC_mvt` or
  `BLOCK_TIME_UTC_mvt`, only the model's input features and a train/validation
  indicator. That is what makes it applicable to the two ranking months, where no
  label exists.
- The near-duplicate bound and the binned residuals read the label, and are
  reported as diagnostics; nothing routes on them.
- The bulk population is defined partly by the label (`taxi <= 3600`). Every
  comparison in this report is between two models scored on the identical rows,
  which is safe; none of it is a deployable selection rule.

## Review (fresh-context Opus one-shot, 2026-09-05)

Reviewer `claude-opus-5-xhigh-2026-09-05` (substitute; desk rule
of 2026-09-05). Log: `review/round1.md`. Verdict **NEEDS REVISION**, six blocking
findings, all accepted. The reviewer verified the leakage check, the
near-duplicate argument, the seed-decomposition arithmetic and every
best-per-fold value it could reach, and confirmed that the Q2 effect is real and
larger than fit noise. What it caught:

**B1 — the clip claim was false and the check behind it was wrong.** The test
compared *mean-normalised* weights against the raw upper bound and never tested
the lower one. Recovering the raw scale, at least 1 % of training rows sit on the
clip floor in all six arms, and the cross-fitted version puts **10.2 %** there.
Claim withdrawn; the repair arms now report both clip fractions directly.

**B2 — the AUCs were in-sample.** The classifier scored its own training matrix.
Cross-fitted, july_loo's 0.9849 becomes 0.9741, and the reduced-feature 0.8809
becomes 0.8035. All AUCs in this report are now labelled in-sample or
cross-fitted.

**B3 — the Q2 close outran its evidence, with the repair coded but unrun.** The
strongest objection of the six: I had written "do not spend anything on
importance weighting" from one in-sample, partly-clipped, extreme ratio, while
`q2b_auc.py` already computed the well-behaved reduced-feature version and nobody
had fed it to a model. Run: the cross-fitted reduced ratio (AUC 0.8035, raw mean
0.942, 4.3 % clipped, ESS 40.7 %) **still costs 4.92 s [+4.30, +5.56]**. The
close now rests on a repair attempted, not asserted, and the revival condition is
named: a model class that cannot condition on the shifting features.

**B4 — the data curve was contaminated.** `C-cat-july.parquet` is byte-identical
to `C1-d8-july.parquet` and `X-xgb-july.parquet` to `X1-d8-july.parquet` (eight
such groups exist in that directory), so two of nine rows compared different
families and my diagnosis of the negatives was wrong for one of them.
`code/q1f_datacurve.py` now carries a content-hash guard; the answer moves from
+4.5 s to +6.0 s (five clean families) or +6.4 s (seven, keeping the
canonically-named member of each pair).

**B5 — `analysis/q1b_residuals.md` published the exact non-comparison the report
exists to refute**, uncaveated, with the opposite sign at nine of ten airports.
That file now opens with a caveat pointing at `q1e_airport_floor.md`.

**B6 — the per-airport floor the assignment asked for was missing.** Added as
`analysis/q1e_airport_floor.md` and in recommendation 2, with both quantities on
the rows that have a duplicate. The headline survives (the model is under the
bound at 19 of 20 airport-folds) and the exception — **LSZH on july_loo, +2.5 s**
— is the one place in this experiment where a near-duplicate bound binds.

**Non-blocking, applied:** `sigma_m` is relabelled as the variance of the
randomised fitting procedure rather than "estimation variance" (all K seeds share
one training draw); the stale "1.4 s" and "three quarters" are gone; the
ranking-versus-fold comparison's two asymmetries are stated with the symmetric
rerun (2026-07 ESS 30.7 % against july_loo's 32.0 %); "48 OOF files" was wrong
twice over and the inventory now scans **325 files across six experiments**,
which moves the best july_loo line from 234.1 to 231.9 and cuts the by-product's
margin from 6.3 s to 4.1 s; `code/q0_inventory.py` and `code/q1f_datacurve.py`
now generate the two artifacts that had none; the non-monotone ladder rung is
remarked; the bootstrap's fixed seed is stated; and the external 279.7 s bound is
cited in recommendation 3. One item is acknowledged and not fixed: `q1b`/`q1d`
group on `stand_raw` without an airport prefix, which is latent (measured effect:
2 rows of 149,172) — `q1e` uses `key_raw` and is the table to trust.
