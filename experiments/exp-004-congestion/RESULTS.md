# exp-004-congestion — RESULTS

exp-id `exp-004` | worker `congestion` (hard) | reviewer: gpt-5.6-sol-xhigh-2026-09-05 (round 1), claude-opus-5-xhigh-2026-09-05 (round 2, substitute reviewer)
Data: live 2026-09-04 files via `prc.data.data_dir()` (`ranking.parquet`
43,564,605 bytes, 344,841 departures / 344,693 arrivals). Folds from
`prc.data.fold_frames`; validation months are masked exactly like the ranking
file and every validation feature is computed from the masked month alone.
Module: `prc/surface_features.py`. Reading of the assignment and rejected
readings: `SPEC.md`.

## Review

| round | reviewer | verdict |
|---|---|---|
| 1 | `gpt-5.6-sol-xhigh-2026-09-05` | not ready to close; no leakage; 4 blocking findings |
| 2 | `claude-opus-5-xhigh-2026-09-05` (substitute) | not ready to close; no leakage; 7 blocking findings |

Both rounds cleared the leakage contract: no path from a hidden departure label
into any feature, through its own row, a neighbour, the fitted tables or the
cache. Every blocking finding is answered in place — the round-2 code defect
(B1) produced `prc.surface_features` VERSION 3, and each corrected claim names
what it replaces rather than quietly replacing it. The two-round cap is spent;
the items round 2 raised that need new measurement rather than correction are
listed under "What this experiment did not establish".

Reviews: `review/round1-gpt-5.6-sol-xhigh-2026-09-05.md`,
`review/round2-claude-opus-5-xhigh-2026-09-05.md`.

## 0. Base model (frozen)

CatBoost, RMSE, 3000 × 0.05, depth 8, l2 3.0, seed 7, `max_ctr_complexity=2`,
GPU 1, 16 threads. Features: own-flight deltas (`MVT − AOBT_3`, `− LOBT`,
`− EOBT_1`, `− SCHED`, `AOBT_3 − SCHED`, `AOBT_3 − LOBT`, `ARVT_3 − MVT`),
`hour`, `dow`, `has_nm`, and categoricals airport, airport:stand,
airport:runway, aircraft type, operator.

| fold | RMSE | bulk (≤3600) | tail (>3600) | no-NM | non-noac |
|---|---|---|---|---|---|
| july | 587.6 | 316.1 | 8452 (n=656) | 3407 (n=3938) | 336.7 |
| december | 316.9 | 217.9 | 5801 (n=261) | 2311 (n=1649) | 226.3 |
| january | 412.9 | 226.6 | 7509 (n=325) | 3598 (n=1435) | 347.5 |

Source `oracle_bound.json`. An earlier version quoted 586.3 / 315.1 and
318.7 / 217.7 from `oracle.log` — the **pre-fix** run round 1 required to be
redone — and omitted january.

The july total is dominated by 656 tail rows: they alone contribute
√(656 × 8436² / 190713) = 495 s of the 586 s. **The bulk column is the
meaningful metric for this lane**; the tail belongs to exp-002.

## 1. Ceiling instrument: the oracle queue count

Feature: the true number of same-runway (and same-airport) take-offs between
this departure's *true* `BLOCK_TIME_UTC_mvt` and its take-off. Illegal at
prediction time; it exists only to bound what a perfect queue feature could
give a model that already holds the own-flight deltas. Features are built one
month at a time on both sides, as the ranking file requires
(`run_oracle.py`, all three folds).

| segment | july: base → oracle | december: base → oracle | january: base → oracle |
|---|---|---|---|
| all | 587.6 → **269.3** | 316.9 → **193.2** | 412.9 → **250.5** |
| **non-noac** | 336.7 → **230.0** | 226.3 → **166.4** | 347.5 → **234.5** |
| bulk ≤3600 | 316.1 → **171.6** | 217.9 → **162.4** | 226.6 → **167.4** |
| NM bulk | 248.7 → **168.8** | 207.9 → **159.3** | 212.7 → **165.9** |
| tail >3600 | 8452.5 → **3543.8** | 5800.9 → **2642.9** | 7509.3 → **4057.4** |
| no-NM | 3407.2 → **1022.7** | 2310.5 → **1044.6** | 3598.0 → **1884.1** |
| noac | 11428.4 → **3330.0** | 9571.8 → **4238.6** | 11017.4 → **4353.9** |

**How loose is this bound?** The oracle count is partly a re-encoding of the
target: a count of take-offs during the taxi is roughly taxi time × local
runway rate. On july bulk rows `corr(oracle q_rwy, y) = 0.709`, and a
per-runway linear fit on the oracle count *alone* scores 248.7 s bulk against a
bulk standard deviation of 413.4 s and a base of 316.1 s. That fit is
in-sample, and it leaves (248.7/413.4)² = **36 %** of the bulk variance — an
earlier version of this line said 60 %, which is the ratio of standard
deviations, not of variances, and made the instrument look less like a target
re-encoding than it is. These are **upper bounds on what queue information
could ever be worth**, not reachable targets.

### What the ceiling says about the board

`non-noac` is the population exp-002's leaderboard arithmetic is stated over
(every row whose `AIRCRAFT_TYPE_mvt` is present, tail included). It is **not**
the same as the NM bulk, and the two must never be compared against the same
target band — an earlier version of this file made exactly that mistake.

| fold | base non-noac | oracle non-noac | headroom |
|---|---|---|---|
| july | 336.7 | 230.0 | 106.7 |
| december | 226.3 | 166.4 | 59.9 |
| january | 347.5 | 234.5 | 113.0 |

The leaders need the non-noac rows at roughly 230–260 s to reach 277 s overall.
**On the two folds that resemble the scored months, a perfect queue feature on
this base model reaches 230.0 and 234.5 — the bottom edge of that band and no
further.** December, the calm month, has a far lower ceiling at 166.4, which is
why a december-only reading would overstate what this lane can do. The ranking
set is 56 % July-like and 44 % January-like, so the binding numbers are the
first and third rows.

Two consequences worth stating plainly. First, this lane cannot by itself take
the model past the leaders; the remaining distance has to come from a better
base model or from a signal outside the surface picture. Second, the leaders
are therefore probably operating near the same ceiling, which makes the noac
rows the likely differentiator — and the oracle takes july noac from 11,428 to
3,330 s, inside the 3,000–4,500 s band exp-002 estimates for them.

That last figure does **not** show the noac rows are a congestion problem. The
oracle count is computed from the hidden block time and §1 says it is partly a
re-encoding of the target; on noac rows, whose recorded block time is often a
`SCHED` placeholder, a large oracle gain is what a target re-encoding predicts
whether or not congestion is the cause. An earlier version of this file drew
that conclusion; it is retracted. What the number does show is that 3,000 s on
noac is reachable by *something*, which is exp-002's and exp-009's lane.

## 2. What the surface state actually correlates with (july 2025, bulk rows)

For an NM-joined row the model knows `MVT − AOBT_3` exactly, so predicting the
taxi time *is* predicting `offset = BLOCK − AOBT_3` (σ = 418 s on bulk rows,
401 s after removing airport medians — the offset is a per-flight quantity, not
an airport constant).

| target | top surface correlates |
|---|---|
| `offset` (NM bulk) | `own_excess` +0.66, `q_apt` +0.36, `overtaken` +0.35, `q_arr_apt` +0.33, `q_apt_rate` +0.33, `overtook` −0.22, `q_rwy` +0.25, `stand_ub_offset` +0.44 (on the 11 % of rows where it bites) |
| taxi time (no-NM bulk, n=3785) | `stand_rwy_med` +0.71, `unimpeded_q` +0.71, `dep_rwy_post1800` +0.38, `nbr_span_rwy` +0.38, `n10_rwy` +0.34, `rwy_share` +0.33 |
| taxi time (NM bulk) | `stand_rwy_med` +0.66, `q_rwy_lobt` +0.45, `nbr_span_rwy` +0.45, `q_rwy` +0.40 |

## 3. The surface features transfer to 2026

The concern from `docs/shift-2026.md` is that CDG lost 27R and 09L, EHAM flips
configuration, and EDDM gained stands 103-108. Measured on the live ranking
file against all twelve 2025 months:

| key | share of ranking departures with a level absent from 2025 |
|---|---|
| `cfg_dep` (departure runway set in the surrounding hour) | 0.08 % (Jan) / 0.10 % (Jul) |
| `cfg_arr` | 0.06 % / 0.11 % |
| airport × stand | 0.46 % |
| airport × stand × runway | 0.72 % |

So the configuration key and the stand-runway geometry level both survive the
2026 changes; the hierarchical fallback (`apt×stand×rwy → apt×term×rwy →
apt×stand → apt×rwy → apt`) covers the remaining 0.7 %. `transform_by_month`
must be used on the ranking file: a whole-file transform lets a January row's
nearest-event features reach into July (0.4 % of rows, 2.7 % for
`nbr_taxiin_rwy`). Transform cost on the whole ranking file: 18 s.

## 4. Correctness and leakage checks

- `transform(masked_frame)` equals `transform(unmasked_frame)` on every
  column (105 with `sf=None`, 112 fitted): `build_features` blanks departure `BLOCK_TIME_UTC_mvt` and
  `TAXITIME_SEC_mvt` in its own arrays, so no departure label can reach a
  feature even if a caller passes an unmasked frame.
- `overtaken` / `overtook`: the VERSION 3 `dominance="exact"` count matches a
  brute-force O(n²) count of the stated definition on **every one of the 12,611
  LSZH July 2025 rows**, 0 mismatches. The VERSION 2 `"push_window"` count used
  for every table in this file does **not**: 317 `overtaken` and 26 `overtook`
  mismatches on that same airport-month, 3.0 % / 0.10 % network-wide. An
  earlier version of this line claimed exactness on the strength of a 400-row
  spot check that happened to miss the failing rows; the round-2 review caught
  it. The feature is consistent between fitting and scoring, so the ablation
  numbers measure what they say they measure — they just measure the
  push-window variant, not the exact one.
- `SurfaceFeatures.fit` reads labels only from the fold's training months;
  `_peak_throughput` likewise. Validation-month features never touch the
  validation labels.
- Runtime: 5-7 s per training month, 18 s for the whole ranking file.
- **Defect found by the Sol review and fixed**: `build_features` read the taxi
  column with `Series.to_numpy(dtype="float64")`, which returns a *view* when
  the column is already float64 — as it is after `prc.data.mask_like_ranking`
  upcasts it. The subsequent departure blanking therefore wrote into the
  caller's DataFrame. **No measured number in this file is affected**: the
  frame handed to `transform` is always the masked one, whose departure taxi
  times are already NaN, and the labels used for training and scoring are read
  from the unmasked frame, not from the one passed to `transform`. The risk was
  latent — a caller passing an unmasked float64 frame would have had its labels
  silently destroyed. Now an explicit copy, re-verified: caller frames
  unchanged, and `transform(masked)` still equals `transform(unmasked)` on all
  112 columns.

## 5. Family ablation

The ablation baseline is 588.5 / 317.4 against the oracle script's
587.6 / 316.1. Both select the same 15 base columns in the same order, so
column order — which an earlier version of this file blamed — is not the cause;
it is float32 quantisation in the feature cache plus CatBoost's
non-deterministic GPU histogram. The same base model, same seed and same
features scored 586.3 in `oracle.log` and 587.6 in `oracle_bound.json`, direct
evidence of the second. **Treat ~2 s as implementation noise** on fold totals.

The `base`, `base+queue`, `base+pressure` and `base+stand` rows below were
**not re-fitted** for VERSION 2 and are bit-identical to their generation-1
values. None of their columns depends on `bucket`, `cfg_prev`,
`rolling_anchor` or `dominance`, so their features are byte-identical between
generations; every other row was re-fitted.

### All three folds, one table

Every row is the frozen base plus one family, same feature cache, same seed,
`prc.surface_features` VERSION 2. Folds: `july` (train 2025-01..06,
n = 190,713), `december` (train 2025-01..11, n = 165,677), `january`
(backward: train Feb–Dec 2025, n = 153,706). Regenerate with
`python make_tables.py`.

#### RMSE, all rows

| config | july  | Δ | december  | Δ | january  | Δ |
|---|---|---|---|---|---|---|
| `base` | 588.5 | +0.0 | 317.5 | +0.0 | 411.7 | +0.0 |
| `base+queue` | 398.1 | -190.4 | 236.1 | -81.4 | 362.7 | -49.0 |
| `base+occupancy` | 400.9 | -187.6 | 244.8 | -72.7 | 371.5 | -40.1 |
| `base+stand` | 487.5 | -100.9 | 268.5 | -49.0 | 377.3 | -34.4 |
| `base+pressure` | 572.2 | -16.2 | 328.3 | +10.8 | 412.6 | +1.0 |
| `base+flow` | 569.3 | -19.1 | 320.6 | +3.1 | 419.5 | +7.9 |
| `base+geometry` | 582.4 | -6.0 | 328.6 | +11.1 | 420.8 | +9.2 |
| `base+config` | 593.9 | +5.4 | 312.7 | -4.8 | 398.6 | -13.1 |
| `base+all` | 426.1 | -162.3 | 219.8 | -97.7 | 354.9 | -56.8 |
| `all_repaired` | 426.1 | -162.3 | 219.7 | -97.8 | 351.8 | -59.9 |

#### RMSE, bulk (y ≤ 3600)

| config | july bulk | Δ | december bulk | Δ | january bulk | Δ |
|---|---|---|---|---|---|---|
| `base` | 317.4 | +0.0 | 219.1 | +0.0 | 228.3 | +0.0 |
| `base+queue` | 293.1 | -24.3 | 213.3 | -5.8 | 218.5 | -9.8 |
| `base+occupancy` | 298.9 | -18.5 | 213.1 | -6.0 | 221.0 | -7.2 |
| `base+stand` | 307.6 | -9.8 | 225.3 | +6.2 | 223.6 | -4.6 |
| `base+pressure` | 320.7 | +3.4 | 209.9 | -9.2 | 226.7 | -1.5 |
| `base+flow` | 315.2 | -2.1 | 211.3 | -7.8 | 222.9 | -5.4 |
| `base+geometry` | 308.5 | -8.8 | 219.1 | -0.0 | 225.9 | -2.3 |
| `base+config` | 323.9 | +6.5 | 215.5 | -3.5 | 225.5 | -2.7 |
| `base+all` | 282.4 | -35.0 | 203.0 | -16.0 | 210.1 | -18.1 |
| `all_repaired` | 281.7 | -35.6 | 201.5 | -17.6 | 209.4 | -18.8 |

#### RMSE, non-noac — the population the leaderboard arithmetic is stated over

| config | july non-noac | Δ | december non-noac | Δ | january non-noac | Δ |
|---|---|---|---|---|---|---|
| `base` | 337.3 | +0.0 | 225.9 | +0.0 | 347.6 | +0.0 |
| `base+queue` | 318.3 | -19.1 | 221.1 | -4.8 | 344.3 | -3.4 |
| `base+occupancy` | 314.5 | -22.8 | 220.8 | -5.1 | 345.5 | -2.2 |
| `base+stand` | 335.0 | -2.3 | 223.0 | -2.9 | 348.3 | +0.7 |
| `base+pressure` | 335.9 | -1.4 | 219.6 | -6.3 | 344.3 | -3.4 |
| `base+flow` | 336.3 | -1.0 | 221.7 | -4.2 | 345.8 | -1.9 |
| `base+geometry` | 337.2 | -0.2 | 221.2 | -4.7 | 344.8 | -2.9 |
| `base+config` | 340.8 | +3.5 | 225.4 | -0.5 | 347.1 | -0.6 |
| `base+all` | 316.5 | -20.8 | 210.5 | -15.4 | 339.2 | -8.4 |
| `all_repaired` | 318.3 | -19.1 | 209.6 | -16.3 | 338.9 | -8.7 |

### Model-fit noise: the fold total cannot resolve a family at all

One seed change (7 → 17), july, everything else identical:

| segment | base s7 | base s17 | Δ | `all_repaired` s7 | s17 | Δ |
|---|---|---|---|---|---|---|
| all | 588.5 | 613.5 | **+25.0** | 426.1 | 414.4 | **−11.7** |
| bulk ≤3600 | 317.4 | 357.9 | **+40.6** | 281.7 | 277.7 | −4.0 |
| **NM bulk** | 249.0 | 248.0 | **−1.0** | 233.8 | 233.2 | **−0.6** |
| **non-noac** | 337.3 | 336.6 | **−0.7** | 318.3 | 318.0 | **−0.3** |
| no-NM | 3412.0 | 3620.7 | +208.7 | 2072.9 | 1945.5 | −127.4 |
| tail >3600 | 8455.1 | 8502.8 | +47.7 | 5458.3 | 5252.7 | −205.6 |
| noac | 11441.2 | 12169.4 | +728.2 | 6728.4 | 6311.1 | −417.3 |

**A single seed moves the july fold total by 25 s and the bulk by 41 s, while
moving the NM bulk by 1.0 s and non-noac by 0.7 s.** All of the instability
lives in the 339 noac, 3,938 no-NM and 656 tail rows, whose predictions a
different tree ordering reshuffles wildly.

The consequence applies to the **bulk** table above: its largest single-family
delta is 24.3 s against 40.6 s of seed noise, so no family can be ranked on it.
The all-rows table is different — `queue`, `occupancy` and `stand` move it by
190 / 188 / 101 s on july, many times the noise — so it does separate those
three from the other four, and only that. It still cannot separate `pressure`
from `flow` from `config`.

Two caveats on the estimate, both cutting against leaning on it: it is **one
seed pair, one fold, two configs**, so it has almost no degrees of freedom; and
it was measured on july only while being used as context for december and
january, where it was never measured.

This is also the retrospective verdict on my own first reading of this
experiment. The "three families move and four do not" split, and the
tail-chasing mechanism built to explain it, were read off fold totals whose
noise is larger than every difference being interpreted.

#### RMSE, tail (y > 3600)

| config | july tail | Δ | december tail | Δ | january tail | Δ |
|---|---|---|---|---|---|---|
| `base` | 8455.1 | +0.0 | 5794.8 | +0.0 | 7453.5 | +0.0 |
| `base+queue` | 4601.8 | -3853.2 | 2562.0 | -3232.8 | 6299.1 | -1154.4 |
| `base+occupancy` | 4566.0 | -3889.1 | 3041.4 | -2753.3 | 6497.9 | -955.6 |
| `base+stand` | 6456.2 | -1998.9 | 3687.1 | -2107.7 | 6610.9 | -842.6 |
| `base+pressure` | 8086.8 | -368.3 | 6364.1 | +569.3 | 7500.4 | +46.9 |
| `base+flow` | 8089.9 | -365.2 | 6079.2 | +284.4 | 7733.0 | +279.5 |
| `base+geometry` | 8428.6 | -26.4 | 6174.8 | +380.1 | 7724.3 | +270.8 |
| `base+config` | 8493.2 | +38.2 | 5712.7 | -82.1 | 7150.0 | -303.5 |
| `base+all` | 5448.8 | -3006.3 | 2134.1 | -3660.7 | 6223.4 | -1230.2 |
| `all_repaired` | 5458.3 | -2996.8 | 2214.6 | -3580.2 | 6150.3 | -1303.3 |

#### RMSE, no-NM

| config | july no-NM | Δ | december no-NM | Δ | january no-NM | Δ |
|---|---|---|---|---|---|---|
| `base` | 3412.0 | +0.0 | 2319.9 | +0.0 | 3581.7 | +0.0 |
| `base+queue` | 1776.4 | -1635.7 | 1014.0 | -1305.9 | 2996.7 | -585.0 |
| `base+occupancy` | 1842.1 | -1569.9 | 1196.6 | -1123.3 | 3095.5 | -486.2 |
| `base+stand` | 2617.6 | -794.4 | 1637.3 | -682.5 | 3188.7 | -393.0 |
| `base+pressure` | 3290.7 | -121.3 | 2515.3 | +195.4 | 3634.0 | +52.3 |
| `base+flow` | 3253.1 | -158.9 | 2390.3 | +70.5 | 3697.5 | +115.8 |
| `base+geometry` | 3368.6 | -43.4 | 2498.6 | +178.7 | 3712.3 | +130.6 |
| `base+config` | 3454.0 | +42.0 | 2259.7 | -60.2 | 3424.9 | -156.8 |
| `base+all` | 2095.7 | -1316.4 | 816.8 | -1503.1 | 2957.9 | -623.8 |
| `all_repaired` | 2072.9 | -1339.1 | 823.2 | -1496.7 | 2915.0 | -666.7 |

COMMON requires these two columns per fold; they are reported for completeness
and, like the all-rows table, they are dominated by a few hundred rows and
should not be used to rank a family.

### The three populations disagree, and which one you use decides the answer

The fold total, the bulk, and the non-noac population rank the families
differently, so a family ablation read on the wrong one produces a confident
wrong answer. All intervals below are **airport-day cluster bootstraps**
(310 clusters per fold, median ~570 rows each), because movements share an
airport and a day and the surface features are built from shared neighbouring
events. Resampling rows independently understates the width by 1.56-2.33× on
the NM-bulk cells — but by only **1.01×** on the july non-noac cells, where the
uncertainty is dominated by a few huge-error rows rather than by within-day
correlation. Clustering matters a great deal for the NM bulk and almost not at
all for july non-noac.

#### Δ RMSE on the NM bulk

| family | july | december | january |
|---|---|---|---|
| queue | -5.2 [-6.9, -3.6] | -3.4 [-4.3, -2.6] | -4.4 [-5.7, -3.2] |
| occupancy | -2.3 [-4.1, -0.3] | -3.6 [-4.8, -2.6] | -3.5 [-4.6, -2.5] |
| stand | -5.7 [-7.1, -4.1] | -4.2 [-5.1, -3.5] | -4.2 [-5.0, -3.3] |
| pressure | -5.7 [-6.8, -4.6] | -5.6 [-6.7, -4.7] | -6.5 [-8.0, -5.1] |
| flow | -2.2 [-3.3, -1.2] | -3.4 [-4.4, -2.6] | -2.9 [-4.5, -1.3] |
| geometry | -1.7 [-2.8, -0.5] | -2.9 [-3.6, -2.1] | -2.7 [-3.9, -1.6] |
| config | -1.2 [-2.3, -0.1] | -0.6 [-1.2, -0.0] | -1.2 [-1.9, -0.4] |

**All 21 intervals exclude zero**, though december `config` clears it by 0.05 s (−0.59 [−1.18, −0.05]) and should be read as no more than that. Every family helps the NM bulk on every
fold; `pressure` — which the fold totals make look useless — is the largest and
most consistent of the seven.

#### Δ RMSE on non-noac

| family | july | december | january |
|---|---|---|---|
| queue | -19.1 [-41.8, -2.7] | -4.8 [-6.9, -2.9] | -3.4 [-6.3, -1.8] |
| occupancy | -22.8 [-56.1, +0.4] | -5.1 [-7.6, -3.0] | -2.2 [-4.0, -1.2] |
| stand | -2.3 [-24.9, +25.4] | -2.9 [-5.6, +0.6] | +0.7 [-5.1, +10.5] |
| pressure | -1.4 [-4.4, +1.5] | -6.3 [-7.8, -5.0] | -3.4 [-6.6, -1.4] |
| flow | -1.0 [-3.8, +1.7] | -4.2 [-6.5, -2.2] | -1.9 [-4.7, +0.4] |
| geometry | -0.2 [-2.3, +2.6] | -4.7 [-6.8, -2.8] | -2.9 [-5.8, -1.3] |
| config | +3.5 [+0.1, +7.5] | -0.5 [-1.6, +0.6] | -0.6 [-1.5, +0.2] |

The same families on the population that is actually scored. Only `queue` is
significant on all three folds. **`stand` is significant on none of them**,
`config` is significantly *worse* on july, and `pressure`, `flow`, `geometry`
and `occupancy` are each significant on two folds out of three.

The `stand` result is worth stating plainly because an earlier version of this
file called it "the third-strongest family" on the strength of a −101 s fold
total on july. That gain is real and it lives **entirely in the noac and tail
rows**; on the non-noac population the family is inside the noise on every
fold. The fold total was the wrong instrument, the same mistake as comparing
an NM-bulk number to a non-noac target band.

### The combination is what carries the lane

Individually most of the families are not separately defensible on non-noac.
Together they are, on every fold and every segment:

| segment | july | december | january |
|---|---|---|---|
| all | −162.3 [−246.6, −75.4] | −97.7 [−189.5, −32.5] | −56.8 [−130.9, −11.4] |
| **non-noac** | **−20.8 [−37.1, −4.7]** | **−15.4 [−19.1, −12.2]** | **−8.4 [−15.1, −4.9]** |
| bulk ≤3600 | −35.0 [−60.5, −13.6] | −16.0 [−19.1, −13.1] | −18.1 [−29.9, −10.5] |
| NM bulk | −15.4 [−17.8, −13.5] | −12.7 [−14.8, −11.0] | −12.1 [−14.0, −10.3] |
| noac | −4666 [−6768, −2479] | −6886 [−11439, −2552] | −5736 [−9433, −1340] |

`base+all` against `base`, cluster bootstrap. **The seven families together are
worth 8 to 21 s on the population the leaderboard scores, on every fold.**

### `base+all` by segment

| segment | july base → base+all | december | january |
|---|---|---|---|
| all | 588.5 → 426.1 | 317.5 → 219.8 | 411.7 → 354.9 |
| bulk ≤3600 | 317.4 → 282.4 | 219.1 → 203.0 | 228.3 → 210.1 |
| **non-noac** | 337.3 → 316.5 | 225.9 → 210.5 | 347.6 → 339.2 |
| NM bulk | 249.0 → 233.6 | 207.8 → 195.2 | 212.8 → 200.7 |
| no-NM | 3412.0 → 2095.7 | 2319.9 → 816.8 | 3581.7 → 2957.9 |
| tail >3600 | 8455.1 → 5448.8 | 5794.8 → 2134.1 | 7453.5 → 6223.4 |
| noac | 11441.2 → 6774.9 | 9630.0 → 2744.3 | 10894.8 → 5158.5 |

### Feature importance (july, `base+all`)

From `importance_base+all_july.csv`: `stand_rwy_med` 13.17, `dt_to_eobt` 9.54,
`apt_rwy` 8.70, `span` 7.96, `apt` 7.11, `q_rwy` 4.29, `span_lobt` 4.16,
`dt_to_sched` 2.96, `dt_to_aobt` 2.47, `own_excess` 2.28, `q_apt_lobt` 2.02,
`stand_arr_in_span` 2.01, `unimpeded_q` 1.82. (An earlier version of this list
came from the overwritten generation-1 run and had `q_rwy` and `span_lobt` in
the wrong order.)

The geometry family supplies the *top* feature of the combined model even
though it is among the weakest on its own. `stand_rwy_med` is the expected taxi
for this stand and runway, and it only becomes useful once the congestion
features are there to explain the deviation from it. A family that is flat
alone is not a family without value — it is a family with no independent axis.

## 6. Recommended set — the single named list for the integrator

```python
import prc.data as D
import prc.surface_features as S

sf = S.SurfaceFeatures().fit(train_frame)        # training months only
F  = sf.transform(month_frame)                   # one isolated month
F  = sf.transform_by_month(ranking_frame)        # the ranking file (two months)
```

A fitted `transform` returns **112** columns: the 96 recommended ones below
plus the 16 optional `queue2` columns, which `build_features` always computes.
Select the 96 explicitly — `[c for f in S.FAMILY_COLUMNS for c in S.FAMILY_COLUMNS[f]]`
— rather than taking every column of the frame. **Use `transform_by_month` on the ranking file** — a
whole-file transform lets a January row's nearest-event features reach into
July.

RECOMMENDED = every column of all seven families (96 columns). Two of them are
categorical and must be declared as such to the model:
`cfg_dep`, `cfg_arr` (`prc.surface_features.CAT_FEATURES`).

- **`queue`** (11): `push_src`, `span`, `span_lobt`, `q_rwy`, `q_apt`, `q_rwy_lobt`, `q_apt_lobt`, `q_rwy_rate`, `q_apt_rate`, `q_arr_rwy`, `q_arr_apt`
- **`pressure`** (47): `dep_rwy_pre300`, `dep_rwy_post300`, `dep_apt_pre300`, `dep_apt_post300`, `arr_rwy_pre300`, `arr_rwy_post300`, `arr_apt_pre300`, `arr_apt_post300`, `dep_rwy_pre600`, `dep_rwy_post600`, `dep_apt_pre600`, `dep_apt_post600`, `arr_rwy_pre600`, `arr_rwy_post600`, `arr_apt_pre600`, `arr_apt_post600`, `dep_rwy_pre900`, `dep_rwy_post900`, `dep_apt_pre900`, `dep_apt_post900`, `arr_rwy_pre900`, `arr_rwy_post900`, `arr_apt_pre900`, `arr_apt_post900`, `dep_rwy_pre1800`, `dep_rwy_post1800`, `dep_apt_pre1800`, `dep_apt_post1800`, `arr_rwy_pre1800`, `arr_rwy_post1800`, `arr_apt_pre1800`, `arr_apt_post1800`, `dep_rwy_pre3600`, `dep_rwy_post3600`, `dep_apt_pre3600`, `dep_apt_post3600`, `arr_rwy_pre3600`, `arr_rwy_post3600`, `arr_apt_pre3600`, `arr_apt_post3600`, `mixed_mode`, `gap_prev_rwy`, `gap_next_rwy`, `gap_prev_apt`, `gap_next_apt`, `rank10_rwy`, `n10_rwy`
- **`occupancy`** (6): `taxiing_at_push`, `taxiing_at_off`, `arr_taxiing_at_push`, `arr_taxiing_at_off`, `overtaken`, `overtook`
- **`config`** (6): `cfg_dep`, `cfg_arr`, `n_dep_rwy`, `n_arr_rwy`, `cfg_changed`, `rwy_share`
- **`flow`** (8): `dep_rate_hr`, `mvt_rate_hr`, `sat_dep`, `sat_mvt`, `nbr_span_apt`, `nbr_span_rwy`, `nbr_taxiin_apt`, `nbr_taxiin_rwy`
- **`geometry`** (5): `unimpeded_q`, `stand_rwy_med`, `own_excess`, `nbr_excess_apt`, `nbr_excess_rwy`
- **`stand`** (13): `stand_nxt_arr_dt`, `stand_prv_arr_dt`, `stand_nxt_dep_off_dt`, `stand_prv_dep_off_dt`, `stand_nxt_dep_push_dt`, `stand_ub_offset`, `stand_lb_taxi`, `stand_arr_in_span`, `stand_mvt_2h`, `stand_turnaround`, `stand_lb_taxi_pos`, `stand_ub_offset_clip`, `stand_bound_active`

- **`queue2`** (optional, 16): `trwy_after1`, `trwy_after2`, `trwy_after3`, `trwy_after5`, `trwy_after8`, `tapt_after1`, `tapt_after2`, `tapt_after3`, `tapt_after5`, `tapt_after8`, `trwy_before1`, `trwy_before2`, `trwy_before3`, `push_demand_apt30`, `push_demand_rwy30`, `push_rank_apt30`

`queue2` is an optional eighth block (16 columns, `EXTRA_FAMILY_COLUMNS`)
holding a finer encoding of the queue. It is **not** part of the recommended
set: it was built after the residual analysis and then dropped unmeasured when
the box went to load 439 and the correction work took priority. Section 7
records it with its revival condition.

### Why all seven families, and not a repaired subset

Two candidates were carried to the end. `base+all` is the seven families
exactly as `transform` returns them. `all_repaired` drops the two
configuration categoricals and **adds** month-relative forms of the neighbour
levels (`run_ablation.derive_relative`); it does not remove the absolute forms,
because `nbr_excess_apt`/`nbr_excess_rwy` arrive with `geometry`. Paired, cluster
bootstrap, `all_repaired` **minus** `base+all` (negative favours the repairs):

| segment | july | december | january |
|---|---|---|---|
| all | −0.0 [−6.2, +5.5] | −0.2 [−2.4, +2.9] | −3.1 [−9.2, +0.1] |
| non-noac | +1.7 [−4.9, +7.5] | **−0.9 [−1.7, −0.2]** | −0.3 [−1.1, +0.3] |
| bulk | −0.7 [−5.6, +2.9] | **−1.6 [−3.5, −0.2]** | −0.7 [−2.6, +0.7] |
| NM bulk | +0.3 [−0.2, +0.8] | −0.3 [−0.7, +0.0] | **+0.6 [+0.2, +1.0]** |

Three significant cells out of twelve, and they do not agree: december favours
the repairs on two segments, january favours the plain set on a third, july is
a wash everywhere. **The two sets are equivalent within fold-to-fold
variation**, so the recommendation is the simpler one: `base+all`, the module's
own output, with no post-processing helper for the integrator to copy and keep
in sync.

The repairs are not wasted. `config_num` and `flow_rel` remain in
`run_ablation.EXTRA_FAMILIES` and stay the right first thing to try if the
configuration categoricals or the neighbour levels misbehave on the 2026
months, where the runway configuration genuinely changes (CDG lost 27R/09L,
EHAM flips) — that is the case the repairs were designed for and the 2025 folds
cannot test.

### One-line summary for the integrator

Import `prc.surface_features`, fit on the training months, transform each month
separately, feed all 96 columns with `cfg_dep` and `cfg_arr` declared
categorical. Worth −20.8 / −15.4 / −8.4 s of non-noac RMSE on
july / december / january over a base that already holds the own-flight deltas
and the airport, stand, runway, type and operator categoricals.

### What this experiment did not establish

- **The best combined set, in the sense the assignment asks for.** Only
  `base+all` and `all_repaired` were fitted; there is no drop-one or subset
  sweep. On july non-noac `base+occupancy` alone (314.5) beats `base+all`
  (316.5), and on the july fold total `base+queue` (398.1) and
  `base+occupancy` (400.9) both beat it (426.1). "The combination carries the
  lane" holds on december and january and on every fold's NM bulk; it does not
  hold on july non-noac. A drop-one sweep is seven fits per fold.
- **The ceiling on the fold the repo calls faithful.** `prc/data.py` defines
  `july_loo` as the analogue of the ranking situation and calls the forward
  `july` fold pessimistic. Every number here uses `july`. The strategic
  reading in §1 — that this lane cannot by itself pass the leaders — rests on a
  ceiling measured on the pessimistic fold, and `july_loo` was never run. This
  is the cheapest thing that could change the lane's verdict.
- **The VERSION 3 overtaking count.** Every table here is VERSION 2
  (`dominance="push_window"`). VERSION 3's genuinely exact count changes
  `overtaken` on about 3 % of rows; its effect on any score is unmeasured.
- **The weather instrument.** `weather_slice.py` was built for the `flow`
  family and its disruption-hour tables were dropped when the january slice
  contradicted the december one. It still uses an IID row bootstrap, unlike
  everything else here, and no weather table survives in this file.
- **`queue2`**, built and never fitted (§7).

### Delivered artefacts

Out-of-fold predictions of the recommended model (`base+all`), columns
`MVT_ID_mvt, fold, y_true, y_pred` plus the segment keys `seg_noac`,
`seg_apt`, `seg_day`, `seg_has_nm` used by `compare.py`:

| path | SHA-256 (first 16) | bytes |
|---|---|---|
| `experiments/exp-004-congestion/oof/base+all_july.parquet` | 6d03b94ec6898df2… | 3175601 |
| `experiments/exp-004-congestion/oof/base+all_december.parquet` | b7e6ab59a099b830… | 2827042 |
| `experiments/exp-004-congestion/oof/base+all_january.parquet` | bbfb23353c4552af… | 2668612 |

Every other config's OOF is beside them under the same naming. Superseded
VERSION 1 predictions are in `oof/gen1/` and are never read by the tools here.
Feature caches live under `artifacts/exp-004-congestion/` (git-ignored), each
with a `.meta.json` recording the code fingerprint that produced it.

## 7. Negative results, their diagnosis, and the repairs

### The stand-turnaround bound, first attempt — wrong sign

**Idea.** A stand holds one aircraft, so the arrival that occupied this stand
before us bounds our block time from below and therefore our taxi time from
above: `taxi <= take-off − prev_arrival_on_block`.

**Result.** The bound held on only 88.6 % of july departures and the
correlation with the target was 0.01. Dead on the first measurement.

**Diagnosis.** The nearest arrival on-block at a stand is usually not our
inbound — it is the *next* occupant, arriving after we vacated. Median slack
was 5,046 s, and on the 11 % of rows where the gap was under 30 minutes the
"bound" was violated 98.5 % of the time. The inequality was inverted.

**Repair, and it worked.** Turned around: the first movement that takes our
stand at or after our push bounds our block time from *above*, so
`taxi >= take-off − that time`. That bound is positive on 11.2 % of july departures and holds on
**95.8 %** of them — 99.15 % on the 20,197 NM rows, but only about 38 % on the
1,162 no-NM rows, where `dep_push` is the `SCHED` fallback rather than a real
off-block, so it is not a bound on precisely the population the family was
built to serve. It correlates +0.435 with `BLOCK − AOBT_3` on the NM bulk
(+0.333 over all positive-bound rows) and became the `stand` family — the third-strongest family on the fold totals. The
failed version cost about twenty minutes; the repaired version is the only
family in this experiment that was not in the assignment.

### The runway-configuration categoricals — a real negative

`cfg_dep` and `cfg_arr` are 160-210 level categoricals that are near-constant
inside an airport-hour, so they largely duplicate airport + hour and pay for it
in variance. `base+config` is the weakest family on every fold. Its december
NM-bulk interval clears zero by 0.05 s (−0.59 [−1.18, −0.05]), which is as
close as any family comes to no effect there. **Repair:** `config_num` keeps
the four numeric columns and drops both categoricals. It is **not** in the
recommended set: §6 shows `all_repaired` and `base+all` are equivalent within
fold-to-fold variation, so the simpler set wins.

### The flow family's absolute levels — a repair that was aimed at the wrong cause

`nbr_span_*` and `nbr_taxiin_*` are absolute levels whose distribution moves
month to month, so a tree that splits on them may not transfer. `flow_rel`
subtracts the airport's own median inside the same month frame, turning them
into a within-month congestion index. That repair is worth keeping.

**But it was not the reason `flow` looked bad.** The Sol review found that the
rolling statistic was read at the last event before the query rather than over
the window ending at the query, so after a quiet period it returned a value
from outside the window entirely. Fixing that changed `flow` on july from
+11.1 s of bulk RMSE (a regression) to −2.2 s (an improvement), and changed
`geometry` from +2.2 to −8.9. **The elaborate tail-chasing story I built to
explain why the take-off-anchored families "must" hurt the bulk was explaining
an artefact of my own code.** The lesson is in the shape of the error: a
plausible mechanism was available for the wrong number, and having one made me
stop looking for the defect.

### The geometry family is redundant alone and essential in combination

`base+geometry` moves the fold total by 2.9 s on july. But `stand_rwy_med` is
the **top feature by importance of the combined model** (12.5 %). The
encodings duplicate what CatBoost already learns from the `apt_stand` and
`apt_rwy` categoricals when nothing else is present; once congestion features
are there to explain deviation from an expected taxi time, the expected taxi
time becomes the most useful column in the model. A family that is flat alone
is not a family without value — it is a family with no independent axis.

### Not closed, and what would revive it

- **Leave-one-month-out fitting of the unimpeded table.** SPEC decision 5
  argued the fold number is honest either way, which is true, and treated the
  in-sample mismatch as mere calibration, which the Sol review disputed with
  1,256 singleton stand-runway groups in the july fit. `exp-001-retro-v1` has
  since implemented LOMO for exactly this family. Unmeasured here.
- **The geometry family on no-NM rows.** It made the july no-NM bulk worse by
  39.6 s while correlating +0.71 with the target there. The unimpeded table is
  fitted partly on LIRF placeholder rows whose "taxi time" is not a taxi
  duration. Revival condition: the no-NM rows get their own head, so the
  estimate is fitted on and applied to one population.
- **`queue2`.** Built after the residual analysis showed the combined model had
  extracted all the linear signal from the count form, then left unmeasured
  when the correction work superseded its run, and then dropped outright when
  the box reached load 439. **Never measured.** Revival: one fit per fold on
  the existing cache, which already carries its 16 columns.

