# exp-010-nmtail — RESULTS

exp-id `exp-010` | worker `anatomy` | tier hard |
reviewer: `claude-opus-5-xhigh-2026-09-05` (substitute), one round, three
blocking findings, all accepted and answered in *Review* at the end.
Module `prc/nmtail.py` (VERSION 2). Code in `experiments/exp-010-nmtail/code/`,
tables in `analysis/`, ablation log in `climb.log`.

## The segment, and what it is actually worth

NM tail = `AOBT_3_flt` present and `TAXITIME_SEC_mvt > 3600`. It is defined by
the hidden label, so nothing can route on it; every model below acts on all NM
rows and is scored on the segment to show the mechanism.

| month | departures | NM tail | share | block-on-LOBT | median phi (\|gap\|>300) |
|---|---|---|---|---|---|
| 2025-07 | 190,713 | 503 | 0.264 % | 93.6 % | 1.00 |
| 2025-12 | 165,677 | 190 | 0.115 % | 66.8 % | – |
| 2025-01 | 153,706 | 272 | 0.177 % | 64.3 % | – |
| all 2025 | 2,085,047 | 3,039 | 0.146 % | 81.7 % | – |

The block-on-LOBT share is a **July** phenomenon at 94 %: on the two folds that
stand for January 2026 it is two thirds, and a third of the segment there sits on
neither anchor cleanly.

**The sizing, which is the first thing to correct.** On `july_loo` the segment's
503 rows at the best model's 1928 s contribute `sqrt(503/190713) * 1928 = 99.0 s`
in quadrature to a fold RMSE of 310.4 — the desk's "about 100 s". But a
contribution is not a prize. Predicting the segment *perfectly* moves the fold by
16.2 s, and the part of it that is a *regime call* by 6.6 s. On the two folds that
stand for January 2026 the numbers are much smaller
(`analysis/sizing.csv`; the base model is `prc.tail_model`, which is the better of
the two available lines on every fold below):

| fold | segment rows | base fold RMSE | segment RMSE | quadrature contribution | fold gain if the segment were perfect | fold gain at the oracle anchor |
|---|---|---|---|---|---|---|
| july | 503 | 327.3 | 2010 | 103.3 | **16.7 s** | 7.6 s |
| july_loo | 503 | 310.4 | 1928 | 99.0 | **16.2 s** | 6.6 s |
| december | 190 | 223.1 | 1883 | 63.8 | **9.3 s** | 1.5 s |
| january | 272 | 350.6 | 1423 | 59.9 | **5.2 s** | 1.0 s |

Against retro-v1's `R0` the same rows read 15.1 / 8.9 / 5.2 s perfect and
5.9 / 1.2 / 1.1 s at the oracle anchor. Everything in this report is inside those
numbers.

*(Corrected after review round 1: the first version anchored on `R0` and quoted
309.7 s for "the fold with the segment removed". 309.7 is the fold with the
segment predicted at zero error, which is the right quantity for the table;
dropping the rows from the denominator instead gives 310.1. And `R0` is not the
best line — `prc.tail_model` is, on all four folds — which makes the segment
slightly *larger*, not smaller.)*

## `prc.nmtail` VERSIONs 3 and 4 — changed by another lane, 2026-09-06

Not an exp-010 change; recorded here because this is the module's owning
experiment and the desk's rule is that a default change is named for its
consumers.

`NMTailModel.num_threads` defaulted to a hardcoded **8** and never read the
environment. The diversity lane (exp-012/exp-015) found that this made a
*declared* thread count untrue: their release manifest declared 6 threads for the
NM-tail member while 8 actually executed, and their reproducibility contract
rests on that field being the count the library really used. Under
`docs/COMPUTE.md` compute allocation it was also a quiet 8-thread job inside a 6-thread
allocation.

**VERSION 3** makes the default `int(os.environ.get("PRC_THREADS", "8"))`. The
old behaviour is reachable as `NMTailModel(num_threads=8)`, and an explicit
argument still wins over the environment, so nothing in `code/` changes unless
`PRC_THREADS` is set in its shell.

**VERSION 4 adds `deterministic=True` and `force_row_wise=True`.** This model was
the *only* member of the standing blend that failed a bit-stability test at a
fixed thread count: two runs at six threads on identical data gave `10f9671d…`
and `51e98c54…`, while the four members whose builders already set those flags
were all bit-stable. The omission was relative to retro-v1's `DEFAULT_PARAMS` and
exp-008's `LgbLearner`, both of which set both — it does not look like a
deliberate choice, but it was load-bearing for the release, which has to be able
to rebuild what it ships.

**This changes the model's output bytes**, so exp-010's published figures were
produced without the flags. They should be within seed noise of a re-run, and the
flags cost a little speed, but if you need the exact published numbers back the
old behaviour is reachable by passing the parameters explicitly.

**What this means for exp-010's numbers.** Every figure in this document was
produced at 8 threads and without the determinism flags. LightGBM does not promise bit-identity across thread
counts — the diversity lane measured two members changing bytes on a 16→6 move,
one of them plain trees on CPU — so a re-run of `wave3.sh` under a shell with
`PRC_THREADS=6` will not reproduce the exact numbers here. It should land within
seed noise, but if you need the published figures back, set `PRC_THREADS=8` or
pass `num_threads=8` explicitly.


## Q1 — regime misclassification versus magnitude

For each NM-tail row the *true* anchor is whichever of `nm_taxi = MVT - AOBT_3`
and `mvt_lobt = MVT - LOBT` is closer to the label; the model's *implied* anchor
is whichever its own prediction is closer to (`analysis/q1_decompose.md`). Note
that "anchor call correct" is agreement with a *label-derived proxy* — a row
whose block time really sat on LOBT but whose taxi happens to land nearer
`nm_taxi` is scored `aobt` — so read the column as agreement, not accuracy.

| model | fold | n | model | always-AOBT | always-LOBT | oracle anchor | anchor call correct | SSE share of wrong calls |
|---|---|---|---|---|---|---|---|---|
| retro-v1 `R0` | july_loo | 503 | 1904 | 3960 | **1562** | 1475 | 72.4 % | 52.8 % |
| retro-v1 `A0` | july_loo | 503 | 1870 | 3960 | 1562 | 1475 | 73.2 % | 50.7 % |
| `prc.tail_model` | july_loo | 503 | 1928 | 3960 | 1562 | 1475 | 69.4 % | 57.4 % |
| retro-v1 `R0` | december | 190 | 1842 | 3018 | 1940 | 1717 | 65.3 % | 40.8 % |
| retro-v1 `R0` | january | 272 | 1423 | 2754 | 1589 | 1271 | 62.9 % | 52.9 % |
| `prc.tail_model` | july | 503 | 2010 | 3960 | 1562 | 1475 | 67.4 % | 60.2 % |

Three things this says.

1. **On july, a one-line predictor that always uses LOBT beats every model on
   this segment** (1562 against 1904 and 1928). It is not deployable — it would
   be catastrophic on the 99.7 % of NM rows that are not in the tail — but it
   bounds how much of the segment's error is the anchor choice.
2. **The anchor choice is the smaller term, and how much smaller depends on the
   season.** The share of the segment that is block-on-LOBT is **93.6 % on july
   but 66.8 % on december and 64.3 % on january**, so the oracle anchor beats
   always-LOBT by only 87 s on july (1475 against 1562) and by 223 s on december
   (1717 against 1940) and 318 s on january (1271 against 1589). *Within* the
   segment the regime call therefore matters two to four times more on the winter
   folds than on july. It still does not matter much *to the fold*, because the
   segment is half the size there: the oracle-anchor gain is 6.6 s on july_loo
   and 1.5 s and 1.0 s on december and january. The model's own anchor call
   agrees with the oracle on 72 % of july_loo rows and the disagreeing 28 % carry
   53 % of the segment's squared error.
   *(Corrected after review round 1: "94 % is block-on-LOBT" was a July number
   used as if it were general. January 2026 is 44 % of the ranking file and the
   `january` fold is its analogue, so the winter figures are the ones that
   travel.)*
3. **The magnitude term is still the segment.** Even on rows where LOBT *is* the closer
   anchor, `|taxi - mvt_lobt|` has median 840 s and RMSE 1614 s over all 2025
   (`analysis/q1_decompose.md`). "The block time sits on LOBT" is true as a mode
   and loose as a measurement. The best single scale is exactly `phi = 1`
   (RMSE 1614, against 1638 at 0.9 and 1661 at 1.1), and the residual is
   airport-structured: median `taxi - mvt_lobt` is **+1794 s at LIRF** and
   **+2079 s at LSZH** but **-301 s at EGLL**, **-420 s at LFPG** and
   **-506 s at EDDF**.

## Q2 — `p(block on LOBT)` is the wrong parameterisation; `E[phi]` is the right one

The RMSE-optimal prediction is `nm_taxi - E[phi | x] * gap`. The desk's
`p(block on LOBT)` is the two-point approximation `E[phi] ~ p*1 + (1-p)*0`,
which is only optimal if `phi` is concentrated on {0, 1} in the rows that matter.
It is not. On the july NM tail with `|gap| > 300` the quantiles of the true
`phi` are 0.51 / 0.83 / **1.00** / 1.34 / 3.78 at the 5th, 25th, 50th, 75th and
95th percentile, and only **34 %** of those rows have `|phi - 1| < 0.1`. The
block time sits *near* LOBT, not on it.

Both were built on the same features and the same folds (`prc.nmtail.NMTailModel`,
`mode="phi"` fits `phi` by least squares with sample weight `gap**2`, which is
exactly least squares on the taxi time; `mode="p"` fits a binary classifier and
uses its probability as `E[phi]`):

NM-tail RMSE, same fits, same features, from the twelve-month leave-one-out
(whose 2025-07, 2025-12 and 2025-01 fits are exactly the `july_loo`, `december`
and `january` folds):

| parameterisation | july_loo | december | january |
|---|---|---|---|
| `mode="phi"` (weighted regression) | **1860** | **1961** | **1518** |
| `mode="p"` (two-point classifier) | 2194 | 2138 | 1678 |

The classifier is 334 s worse on july_loo, 177 s on december and 160 s on
january. A screened second stage is 34 s worse than the single
global head on july_loo (1895 against 1861 at the same defaults). **The
two-point form is measured out.** `mode="p"` is kept in the module because a
probability is auditable in a way a regression output is not, and the desk asked
for the calibration; it is not the recommended setting.

### The calibrated `p` the desk asked for: sharp as a classifier, wrong as a predictor

Leave-one-month-out over all twelve 2025 months, 2,062,577 NM-joined departures,
each scored by a model fitted on the other eleven (`analysis/q2_calibration.md`,
`oof/loo_2025.parquet`). On the rows where the two anchors actually differ
(`|gap| > 600 s`), `p_hat` is **well calibrated and sharp**:

| `p_hat` bin | n | mean `p_hat` | observed block-on-LOBT | mean `phi_hat` | observed mean `phi` | P(taxi > 3600 s) |
|---|---|---|---|---|---|---|
| 0.00-0.05 | 297,311 | 0.020 | 0.015 | 0.003 | -0.003 | 0.05 % |
| 0.05-0.10 | 123,000 | 0.072 | 0.070 | 0.126 | 0.118 | 0.14 % |
| 0.10-0.20 | 98,480 | 0.141 | 0.146 | 0.217 | 0.218 | 0.27 % |
| 0.20-0.30 | 35,652 | 0.243 | 0.254 | 0.311 | 0.324 | 0.52 % |
| 0.30-0.50 | 33,972 | 0.390 | 0.400 | 0.418 | 0.439 | 0.92 % |
| 0.50-0.70 | 23,910 | 0.596 | 0.608 | 0.551 | 0.582 | 1.51 % |
| 0.70-0.90 | 29,162 | 0.811 | 0.820 | 0.697 | 0.723 | 1.67 % |
| 0.90-1.00 | 24,270 | 0.949 | **0.962** | 0.902 | 0.931 | **2.22 %** |

AUC **0.910** for block-on-LOBT, Brier 0.0752 against 0.1399 for the base rate;
`phi_hat` is calibrated on the same rows (its mean tracks the observed mean of
`phi` in every bin). So the classifier is a good classifier — and it is still the
worse predictor, on the same rows, over the same twelve months:

| predictor | RMSE all NM | RMSE NM tail | RMSE NM bulk |
|---|---|---|---|
| `phi` regression, weight `gap^2` | **237.3** | **1971.2** | **225.1** |
| `p` classifier, two-point mixture | 272.2 | 2206.7 | 258.9 |
| always block-on-AOBT | 384.9 | 3447.8 | 361.6 |
| always block-on-LOBT | 740.1 | 1824.6 | 737.4 |

Two things follow. First, **knowing the regime is not knowing the answer**: even
in the top `p_hat` bin, where 96 % of rows really are block-on-LOBT, only 2.2 %
are in the tail — the tail needs block-on-LOBT *and* a large gap, and the
classifier cannot supply the second. Second, the two-point mixture throws away
the magnitude: the observed mean `phi` in that bin is 0.931, not 1, and in the
0.30-0.50 bin it is 0.439 — the regression tracks that and the classifier's
`p*1 + (1-p)*0` cannot.

### The screen that is new: stand re-occupation between LOBT and AOBT

If another aircraft on-blocks at the stand between `LOBT` and `AOBT_3`, this
departure was gone before it, so its block time sat near LOBT. Unlike the
counting bound in exp-003 this needs no arrival/departure alternation
assumption — only that two aircraft are not on one stand at once — and unlike
`AOBT - LOBT` it is evidence about *this* flight rather than about the filing.
Measured over all twelve 2025 months, on the 566,220 NM rows with `AOBT_3 - LOBT > 600 s` strictly (base rates P(block-on-LOBT) 12.2 %,
P(taxi > 3600 s) 0.41 %):

| screen | rows | share | P(block-on-LOBT) | P(taxi > 3600 s) | mean phi |
|---|---|---|---|---|---|
| an arrival parked in (LOBT, AOBT) | 10,209 | 1.80 % | **49.8 %** | 5.14 % | 0.517 |
| ... of a *different* aircraft type | 6,974 | 1.23 % | **55.3 %** | 5.45 % | 0.578 |
| `AOBT - LOBT > 1 h` (the tail worker's screen) | 3,522 | 0.62 % | 37.3 % | 21.95 % | 0.442 |
| **both: `> 1 h` and a different-type re-occupation** | **542** | **0.10 %** | **74.5 %** | **43.4 %** | **0.731** |

The re-occupation evidence **doubles** P(block-on-LOBT) inside the existing
one-hour screen, from 37 % to 75 %, and doubles P(tail) from 22 % to 43 %.
Requiring a different aircraft type matters because the intervening arrival is
otherwise often this flight's own late inbound, which implies the opposite
(block near AOBT): same-type-only re-occupation scores 37.9 % against
different-type's 55.3 % over the twelve months (57.6 % and 41.0 % on july alone).

**What did not work as a screen.** A systemic NM-lag term — the fraction of other
departures at the same airport-hour whose AOBT is more than 30 minutes after
LOBT, computed leave-self-out — moves P(block-on-LOBT) from 12.2 % to 13.5 % at
a 5 % threshold and 18.1 % at a 20 % threshold. It is a real but weak signal and
is not worth a column. `LOBT` within 60 s of `SCHED` is worth nothing on its own
(12.0 % against a 12.2 % base) even though it is a strong marker *within* the
tail (98.4 % of the july tail's LOBT-at-schedule rows are block-on-LOBT): it
selects on-time filing, which is most flights.

## Q3 — a dedicated head does not beat the existing line

NM-tail RMSE, all four folds. `head` is `prc.nmtail.NMTailModel(mode="phi")`
fitted on the fold's training months; `R0` is retro-v1's guarded linear-leaf
line, `A0` the same without linear leaves, `router` is `prc.tail_model`.

| fold | n | head (this experiment) | R0 | A0 | `prc.tail_model` router |
|---|---|---|---|---|---|
| july | 503 | **1951** | – | – | 2010 |
| july_loo | 503 | **1861** | 1904 | 1870 | 1928 |
| december | 190 | 1954 | **1842** | – | 1883 |
| january | 272 | 1501 | **1423** | – | 1423 |

Every head figure is at `prc.nmtail` VERSION 2's defaults (900 rounds, lr 0.04,
63 leaves, seed 7), which is what the reproduction command below runs. The
twelve-month leave-one-out fits in `code/loo.py` (400 rounds, lr 0.05, 63 leaves)
are an independent cross-check on the three folds that coincide with a
leave-one-month-out — july_loo 1860, december 1961, january 1518 — and agree
within 7 to 17 s. *(Corrected after review round 1: the first version headlined
1840, from a hand-tuned run whose configuration was in no script
(`--rounds 1500 --lr 0.03 --leaves 127`, 1044 s). It is dropped; the module
defaults give 1861 and ship OOF.)*

**The head wins on the two july folds and loses on december and january.** July
is where LIRF's summer regime puts 54 % of the segment's rows and 60 % of its
squared error; away from it the existing line is better. A segment improvement
that reverses sign between folds is not a segment improvement.

Whole-fold effect of putting the head into the current best line on the NM rows
(`swap`), and of averaging the two (`mix 0.5`):

| fold | R0 alone | head swapped in on NM rows | 50/50 mix |
|---|---|---|---|
| july_loo | **324.8** | 325.4 | 314.9 |
| december | **222.2** | 240.5 | 224.4 |
| january | **347.6** | 359.7 | 349.4 |

Swapping is worse on every fold: the head is a single-target model with none of
retro-v1's bulk refinements, so what it gains on 503 rows it loses on 186,000.
The 50/50 mix gains 9.9 s on july_loo and *loses* 2.2 s on december and 1.8 s on
january — but **the 0.25 weight gains on all four folds and against both base
models** (recommendation 4 below). The first version of this report read the
0.5 column only and called the blend "not a result"; that was wrong, and the
blend is the one positive finding of the experiment. It is still not a *tail*
result: at weight 0.25 the NM-tail RMSE improves on july_loo (1904 to 1848) and
does not on december (1842 to 1850).

Two more forms were tested and are worse:

- **A screened second stage** (a specialist fitted only on the 4 % of NM rows with
  `AOBT - LOBT > 30 min`, which is the closest thing to the noac head's shape):
  1895 on july_loo against the single global head's 1861. The screened
  population is too small to support a second model; the global head with the
  screen as a *feature* is better than the screen as a *route*.
- **The two-point classifier**: 2177 (Q2).

## Q3b — what the re-occupation screen is worth as a feature

Leave-one-out of the seven `reoccupation_signals` columns and their two
interaction terms from the head's 78 features:

| fold | NM-tail RMSE with | without | all NM rows with | without |
|---|---|---|---|---|
| july_loo | **1861** | 1878 | **271.7** | 272.6 |
| december | 1954 | **1949** | 227.7 | 227.7 |

**+17 s on july_loo and -5 s on december: zero within noise.** The screen has a
large *conditional* lift (37 % to 75 % on P(block-on-LOBT)) and almost no
*marginal* value inside a model that already has `AOBT - LOBT`, because it fires
on 0.10 % of rows and the gap already carries most of what it says about them.
It is kept in `prc.nmtail` and offered to retro-v1 as a cheap column — its cost
is one `searchsorted` per stand — but it is not a lane and this report does not
claim a gain for it.


## What the desk should do with this

1. **Re-size the lane, but compare like with like.** Predicting the NM tail
   perfectly is worth 16.2 s of july_loo fold RMSE, 9.3 s on december and 5.2 s
   on january; a perfect *regime call* inside it is worth 6.6 / 1.5 / 1.0 s, and
   the best head measured here 1.0 s on july_loo and negative on the other two.
   The right comparison for the noac rows is their *headroom*, not the value the
   noac head already banks: exp-002's oracle table gives 275.3 / 215.1 / 338.8
   against a delivered 327.3 / 223.1 / 350.6, i.e. **52.0 / 8.0 / 11.8 s still on
   the table there** against this segment's 16.2 / 9.3 / 5.2. Noac is 3.4 times
   larger on july, the same size on december and half the size on january.
   *(Corrected after review round 1: the first version compared against exp-002's
   250 s, which is captured value, not headroom.)* **This lane still should not
   get a runner** — but on the winter folds it is not obviously smaller than the
   alternative, and the honest statement is that both are single-digit seconds
   there.
2. **The mechanism is understood and is mostly not routing.** 94 % of the july
   segment is already block-on-LOBT (67 % on december, 64 % on january); the
   residual is that "on LOBT" means ±840 s (median absolute), with an
   airport-level shift of +1794 s at LIRF and -301 to -506 s at EGLL, LFPG and
   EDDF. Further work on this segment is a *magnitude* problem at LIRF, which is
   the lirf worker's lane, not a regime problem.
3. **Do not build a `p(block on LOBT)` classifier as a predictor.** Measured out:
   2207 against 1971 on the twelve-month leave-one-out, and 272.2 against 237.3
   over all NM rows. The classifier itself is good (AUC 0.910, well calibrated);
   it is the two-point *use* of it that loses. Review round 1 tried the obvious
   repair — a leave-one-month-out binned map `p_hat -> E[phi]` instead of pinning
   the mixture at 1 — and it recovers 1.6 s of the 34.9 s gap, so the close
   stands.
4. **Take the head as a blend member, not as a tail fix.** This is the one
   positive result and the first version of this report missed it by reporting
   only the 0.5 weight. At **weight 0.25** the head improves the whole fold in
   *every* comparison available, against both base lines:

   | fold | vs `R0` | vs `prc.tail_model` |
   |---|---|---|
   | july | – | 327.31 to 325.33 (**-1.98**) |
   | july_loo | 324.76 to 317.28 (**-7.48**) | 310.39 to 308.16 (**-2.23**) |
   | december | 222.18 to 221.48 (**-0.70**) | 223.12 to 222.32 (**-0.80**) |
   | january | 347.60 to 347.39 (**-0.21**) | 350.55 to 349.88 (**-0.67**) |

   Seven comparisons, seven gains, two independent base models. The december and
   january gains are inside exp-008's seed noise on their own, but the sign holds
   on four folds. **This is not a tail result** — the NM-tail RMSE at weight 0.25
   is 1848 against `R0`'s 1904 on july_loo but 1850 against 1842 on december — it
   is an ensembling gain across all NM rows, from a model built on a different
   target parameterisation. It belongs to the diversity worker: exp-008's
   seven-member blend buys 11.2 s on july_loo and this is 7.5 s from one member.
5. **Take the re-occupation columns if they are free, and expect nothing.** They
   are in `prc.nmtail.reoccupation_signals` (VERSION 2), one call per calendar
   month on the masked frame, seven columns; inside my head they measure zero.
   They may do better inside retro-v1's frame, which is a different base — the
   same caveat retro-v1 raised about my exp-003 attribution, applied to myself.
   They transfer: they fire on 1.94 % of January 2026 rows and 1.69 % of July
   2026 rows with `AOBT_3 - LOBT > 600 s`, against 1.80 % over 2025.

## Artifacts and reproduction

| what | path |
|---|---|
| module | `prc/nmtail.py` (VERSION 2) |
| re-occupation cache | `artifacts/exp-010-nmtail/reocc_{2025-01..2025-12,ranking}.parquet`, SHA-256 in `analysis/artifact_hashes.md` |
| OOF, per fold | `experiments/exp-010-nmtail/oof/nmtail-{phi,p}_{july_loo,december,january}.parquet` (`MVT_ID_mvt, fold, y_true, y_pred`; `y_pred` NaN on rows with no NM join, which this model does not predict) |
| OOF, twelve-month leave-one-out | `experiments/exp-010-nmtail/oof/loo_2025.parquet` (adds `phi_hat`, `p_hat`, `gap`, `nm_taxi`, `mvt_lobt`) |
| ablation log | `experiments/exp-010-nmtail/climb.log` (one JSON line per run, with the configuration) |
| four COMMON.md slices, every run | `experiments/exp-010-nmtail/analysis/slices_full.md` |
| sizing arithmetic, all folds and both base models | `experiments/exp-010-nmtail/analysis/sizing.csv` |
| screens | `experiments/exp-010-nmtail/analysis/q2_screens.md` (`code/screens.py`) |

The three per-fold OOF files come from the leave-one-month-out fits, whose
training months are *identical* to the `july_loo`, `december` and `january` fold
definitions in `prc.data.FOLDS` (asserted in the emitting snippet); the forward
`july` fold has no OOF, because its six-month training set is not a
leave-one-month-out.

```bash
PYTHONPATH=. .venv/bin/python experiments/exp-010-nmtail/code/build_reocc.py
PYTHONPATH=.:experiments/exp-010-nmtail/code .venv/bin/python experiments/exp-010-nmtail/code/q1_decompose.py
PYTHONPATH=.:experiments/exp-010-nmtail/code .venv/bin/python experiments/exp-010-nmtail/code/q1b_slices.py
PYTHONPATH=.:experiments/exp-010-nmtail/code:experiments/exp-003-anatomy/code \
  .venv/bin/python experiments/exp-010-nmtail/code/run_folds.py --folds july,july_loo,december,january --modes phi
PYTHONPATH=.:experiments/exp-010-nmtail/code:experiments/exp-003-anatomy/code \
  .venv/bin/python experiments/exp-010-nmtail/code/loo.py
PYTHONPATH=.:experiments/exp-010-nmtail/code .venv/bin/python experiments/exp-010-nmtail/code/calib.py
```

## Leakage self-check against `docs/input-rule.md`

- `prc.nmtail.reoccupation_signals` reads arrival `BLOCK_TIME_UTC_mvt` (legal —
  the ranking file keeps it on arrival rows) and departure `LOBT_flt`,
  `AOBT_3_flt`, `MVT_TIME_UTC_mvt` (all present in the ranking file). It never
  reads a departure's own block or taxi time. It is called once per calendar
  month, on the masked frame, including for the ranking file, so no window
  crosses the January-to-July 2026 gap.
- `NMTailModel.fit` reads departure block times only through the label, and only
  from the fold's training months; `predict` never touches them.
- The feature frames come from exp-003's builder, which asserts bit-identical
  output on the raw and the masked month.
- The NM-tail slice is defined by the hidden label. It is used to *report*, never
  to route: every model here acts on all NM rows.

## Review (fresh-context Opus one-shot, 2026-09-05)

Sol was not available, so the reviewer is `claude-opus-5` at `xhigh` in a
fresh context (desk rule, 2026-09-05). Log: `review/round1.md`. Verdict:
*"Sound, reproducible measurement and a conclusion that survives every
re-anchoring I tried — but three things must change before these rows close."*
Three blocking findings, all accepted:

**B1 — the blend was dismissed at the one weight where it fails.** Correct, and
this was the worst error in the report: I read the `w = 0.5` column, saw it lose
on december and january, and wrote "not a result at all" while `climb.log` also
held `w = 0.25`, which **gains in all seven available comparisons against two
independent base models**. Recommendation 4 now reports it and routes the head to
the diversity worker.

**B2 — the noac comparison used the framing this report exists to correct.**
Correct. "Hundreds of seconds on the table" was exp-002's 250 s, which is value
already banked by the noac head, not headroom. The headroom from exp-002's own
oracle table is 52.0 / 8.0 / 11.8 s against this segment's 16.2 / 9.3 / 5.2 s.
The close survives; the sentence is rewritten.

**B3 — no OOF, and the headline number was not reproducible.** Correct. The 1840
came from a hand-tuned run whose flags were in no script. Fixed three ways:
`prc.nmtail` is at **VERSION 2** with its defaults set to the configuration every
number is now measured at (900 rounds, lr 0.04, 63 leaves); the headline is 1861,
which the documented command reproduces; and per-fold OOF ships for `july_loo`,
`december` and `january`, taken from the leave-one-month-out fits whose training
months are identical to those folds.

**Non-blocking, applied:** the base line is re-anchored on `prc.tail_model`
(N1, which makes the segment slightly *larger*: 16.2 s not 15.1 s); the sizing is
given on all four folds (N2); "94 % block-on-LOBT" is qualified as a July number
against december's 66.8 % and january's 64.3 %, with the consequence that the
regime call is worth 223 s and 318 s inside the segment there against 87 s on
july (N3); the module docstring carries the corrected sizing (N4); the
phi-versus-p and screened comparisons are re-stated at one configuration (N5);
the screen population is written as `> 600 s` strictly and has a committed script
(N6); the four COMMON.md slices are in `analysis/slices_full.md` (N7); "anchor
call correct" is labelled as agreement with a label-derived proxy (N8); the
309.7-versus-310.1 slip is fixed (N9); artifact hashes are in
`analysis/artifact_hashes.md` (N10).

**The reviewer also ran the repair I had not tried** — a leave-one-month-out
binned map `p_hat -> E[phi]` instead of pinning the mixture at 1 — and it
recovers 1.6 s of the 34.9 s gap. The `p` close stands with that repair attempted
rather than merely asserted.

Reviewer: `claude-opus-5-xhigh-2026-09-05` (substitute).
