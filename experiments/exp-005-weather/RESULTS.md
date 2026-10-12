# exp-005-weather — RESULTS

Worker `weather`, 2026-09-05. Assignment `assignments/weather.md`, reading and
rejected readings in `SPEC.md`. Provenance `docs/weather-provenance.md`.

Data version: the twelve 2025 training files (unchanged since 2026-08-13) and
the live `ranking.parquet` of 2026-09-04 (344,841 departures = 152,719 January
+ 192,122 July, reproduced by the cache builder). No number here used a
`_stale_` file.

---

## 1. The finding in one paragraph

Event-hour ERA5 is worth carrying as **seven columns, not forty-four**
(`WINTER_CORE`, section 7). Under an airport-day cluster bootstrap the one
result that holds on all three folds is the **bulk gain: −4.42 s january,
−2.05 s december, −2.72 s july** (95 % intervals all excluding zero). The much
larger conditional numbers — −54 s on december's snow and freezing hours — are
real in point estimate but imprecise, because snow concentrates in a handful of
airport-days: december's interval excludes zero, january's does not. **The gain
is small because the actual-time columns already carry most of it**: with the same seven columns on both sides, removing
`AOBT_3_flt` multiplies weather's bulk value by **3.0 to 5.5×** (january −19.18,
december −11.31, july −8.11 s against −4.42 / −2.05 / −2.72), which is also why
the tail worker measured only −0.9 s on their stronger model. The lane's
standing value is outside the fold averages: **every January 2026 airport-day
whose mean take-off-minus-AOBT exceeds its airport's entire 2025 range by more
than 30 s has snow, freezing precipitation or sub-zero temperatures — 18 of 18 —
and July 2026 has no such day at all.** Weather and the within-month
`MVT − AOBT` aggregates are the two ranking-file inputs that mark those days;
no statistic fitted on 2025 alone can.

## 2. Data: inventory, completion, verification

### 2.1 Inventory against the manifest (assignment step 1)

The snapshot held **39 of 44** airport-product files on arrival, every SHA-256
matching `manifest.json` and every ERA5 grid cell matching the expected
quarter-degree rounding. Missing: all four LSZH files, and LTFM's
January-plus-July 2026 extra. Reproduce with
`experiments/exp-005-weather/inventory.py`.

The five gaps were retrieved by `experiments/exp-005-weather/fetch_missing.py`,
which drives the shared `scripts/download_copernicus_era5.py` request builders
and validator so the new manifest rows are identical in shape to the old ones.
The order was chosen so the files the folds need arrive first.

| file | state |
|---|---|
| `LSZH_core_2024-01-01_2026-07-31.nc` | retrieved, validated, hashed |
| `LSZH_extra_2025.nc` | retrieved, validated, hashed |
| `LSZH_extra_2026_01_07.nc` | retrieved, validated, hashed |
| `LTFM_extra_2026_01_07.nc` | retrieved, validated, hashed |
| `LSZH_extra_2024.nc` | retrieved, validated, hashed (2024 feeds no fold and no ranking month) |

**Current state: retrieval complete — 44 of 44 production files present,
validated and hashed** (11 airports x 4 products; the manifest holds 46 rows
including two two-day test files). Every 2025 hour and both ranking months are
complete for all ten ranking airports. The 39-of-44 figure above is the state
on arrival, kept because it is what the assignment's step 1 asked for.

### 2.2 The hourly table

`artifacts/weather/era5_hourly.parquet` — 226,320 rows (10 airports x
22,632 hours), 15,181,053 bytes,
SHA-256 `8238378ec1f6d66a6c59712993ec54ac72af3f130cb7a8a7754a3fbec369f293`,
built at `prc.weather_features` VERSION 2. (Section 8 repeats these; an earlier
draft of this section carried the superseded VERSION 1 size and hash.)

Residual nulls are the documented boundary behaviour: the final hour of each
ranking month has no `H+1` stamp to carry the accumulation over `[H, H+1)`, and
because a rolling window now requires every hour of the window, that null
propagates into the 3- and 6-hour sums at the start of a period whose preceding
hours were never retrieved (February to June 2026 has no extra-product data by
design). These stay NaN rather than becoming zero. Concrete effect on the
ranking file is in section 7.

### 2.3 The join is verified against the raw NetCDF, not assumed

`experiments/exp-005-weather/code/verify_join.py` re-reads the source files and
checks the join by hand on real departures. All 13 checks pass. The one that
matters most is discriminating, not tautological:

```
PASS  precipitation is the accumulation over [H, H+1), not (H-1, H]
      joined 2.1896 mm; raw stamped H+1 2.1896, stamped H 1.6336
```

A one-hour error there would silently manufacture signal by giving every
departure the previous hour's rain. Also checked: unit conversions, wind speed
and the "blows from" direction convention, runway heading from the designator,
head/crosswind trigonometry against a hand computation, that the off-block join
uses AOBT and not take-off, that no-NM rows get no off-block weather rather
than silently borrowing the take-off hour, and that `transform` preserves the
frame's index and row count.

### 2.4 A defect this found, and its fix

The first version of the flags turned **unknown into false**: with LSZH's
snowfall not yet retrieved, `snowing` read a NaN as "not snowing" and
`liquid_mm` treated missing snow as zero snow. That is the shape of defect that
produces a clean-looking signal from absent data. Flags now propagate NaN when
any input is missing, so the trees see the gap as its own branch. Fixed before
any reported number was measured.

---

## 3. Which fold can answer the question at all

Winter exposure per airport-month, snowfall in mm water equivalent, from the
**current** hourly artifact (`code/winter_effect.py`, output
`logs/winter_effect_v2.txt`; an earlier draft used `winter_effect_v1.txt`, taken
before the LSZH extra-product retrieval completed, and so understated LSZH and
LTFM):

| airport | 2025-07 | 2025-12 | 2025-01 | **2026-01 (scored)** |
|---|---:|---:|---:|---:|
| EDDF | 0.0 | 0.2 | 18.1 | **27.8** |
| EDDM | 0.0 | 7.8 | 21.8 | **27.0** |
| LSZH | 0.0 | 0.8 | 13.1 | **21.0** |
| EHAM | 0.0 | 0.0 | 4.6 | **12.9** |
| LEMD | 0.0 | 0.1 | 0.0 | **9.5** |
| LFPG | 0.0 | 0.5 | 2.2 | **6.7** |
| LTFM | 0.0 | 0.1 | 0.0 | **5.5** |
| EGLL | 0.0 | 0.0 | 7.4 | **1.7** |
| LEBL | 0.0 | 0.0 | 0.0 | **0.0** |
| LIRF | 0.0 | 0.0 | 0.0 | **0.0** |

**December 2025, the winter primary fold, holds about a twelfth of the January
2026 snow across the ten airports (9.63 mm against 112.03) and has zero snow
hours at six of them.** It cannot measure the effect this lane exists to test.
January 2025 is the only month of the training year whose winter resembles the
scored month. The desk added `FOLDS["january"]` (train Feb–Dec 2025, validate
Jan 2025) on the same reasoning and this experiment uses that shared definition.
It carries 5,120 snow-or-freezing departures against december's 1,511.

## 4. Weather moves the quantity the model must actually predict

Once `AOBT_3_flt` is a feature, the learnable part of taxi-out is a clock offset,
not the physical taxi duration. If weather moved taxi but left that offset alone,
weather could not help the scored task at all.

**Sign convention.** The model fits `taxi − nm_taxi = AOBT_3 − BLOCK_TIME`. The
diagnostic below reports `Δ = BLOCK_TIME − AOBT_3`, the negative of the fitted
residual, because that is the direction `docs/input-rule.md` and COMMON.md quote
it in. Magnitudes and dispersions are identical; only the sign of the mean shift
flips. (Flagged by Sol round 3.)

Winter 2025 (Jan, Feb, Nov, Dec), NM-joined departures, current artifact:

| | n | mean taxi | mean Δ | **sd of Δ** |
|---|---:|---:|---:|---:|
| other hours | 603,484 | 986.2 s | +37.6 s | 357.1 s |
| snow or freezing hours | 16,413 | 1224.3 s | −121.4 s | **642.0 s** |

Weather moves Δ by 159 s and **nearly doubles its dispersion**. Per airport the
shift is large and airport-specific: LTFM taxi +1005 s with Δ moving 359 s and
its sd going 440 → 1088; EDDM taxi +260 s, Δ −195 s, sd 389 → 628; LSZH taxi
+141 s, Δ −43 s, sd 269 → 348. So the mean shift is learnable and the variance
increase is the ceiling on how much of it can be learned.

(An earlier draft quoted 605,666 / 14,231 rows, taxi 1280.6, Δ −126.0, sd 680.8
from the pre-completion diagnostic; the numbers above supersede those. The
conclusion is unchanged in direction and magnitude.)

## 5. The 2026 alignment check (desk request)

`docs/shift-2026.md` found January 2026 outside the 2025 envelope at the
northern airports. Daily mean take-off-minus-AOBT is computed from two columns
the ranking file carries, so this is observable with the label hidden.
`code/alignment_2026.py` ranks each 2026 day inside its airport's 2025
distribution.

**Days above the airport's 2025 maximum:** EDDF 9, EHAM 9, LEMD 3, LFPG 2,
LSZH 2, EDDM 1, LEBL 1, EGLL 0, LIRF 0, LTFM 0 — **27 in January 2026, and 0 in
July 2026 at every airport.** My counts reproduce the anatomy worker's 9 and 9
independently.

How many of the 27 are weather days depends on the margin, and the dependence is
the point (`logs/alignment_2026.csv`):

| exceedance over the 2025 maximum | days | with snow or freezing hours | with any of snow / freezing / sub-zero |
|---|---:|---:|---:|
| > 0 s | 27 | 19 | 26 |
| > 30 s | 18 | 15 | **18 (all)** |
| > 100 s | 12 | 10 | **12 (all)** |
| > 300 s | 7 | 6 | **7 (all)** |

**Every day that clears its 2025 maximum by more than 30 seconds is a winter
day.** The single day with no snow, no freezing hour and no sub-zero hour is
LEBL on 21 January, which clears its maximum by **0.7 seconds** — a boundary
artefact, not a counterexample. (An earlier draft of this file claimed all 27
were weather days; that was false and is corrected here. Sol review, finding 8.)

The 27 against the other 283:

| | snow mm | snow hours | freezing hours | subzero hours | min temp | daily mean | daily sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| within the 2025 range | 0.22 | 0.92 | 0.69 | 4.34 | +2.84 °C | 1005.6 s | 297.9 s |
| **above the 2025 max** | **1.81** | **6.52** | **6.15** | **9.81** | **−1.39 °C** | **1440.9 s** | **782.7 s** |

Every one of the nine EDDF days carries snow, freezing precipitation, or a full
day below zero; the worst (29 January, 1427 s against a 2025 maximum of 1117 s)
is the snowiest day of the month at 6.0 mm and 21 freezing hours. EHAM's run of
3–7 January reaches 2330 → 2595 → 2604 s against a 2025 maximum of 1254 s — more
than double the worst day in the training year — on 124 to 283 departures a day
where the norm is about 500. The traffic collapse is itself the disruption.

Spearman rank correlation across the 31 January 2026 days, daily weather
against daily take-off-minus-AOBT (mean / dispersion):

| airport | snow mm | freezing hours | subzero hours |
|---|---|---|---|
| EHAM | **0.82 / 0.83** | 0.58 / 0.63 | 0.60 / 0.57 |
| LFPG | **0.68 / 0.74** | 0.49 / 0.47 | 0.72 / 0.73 |
| EDDF | **0.65 / 0.64** | 0.52 / 0.56 | 0.54 / 0.55 |
| EGLL | 0.56 / 0.58 | 0.23 / 0.26 | **0.76 / 0.78** |
| LEMD | 0.39 / 0.46 | 0.38 / 0.43 | 0.58 / 0.63 |
| EDDM | 0.34 / 0.48 | 0.44 / 0.39 | 0.11 / 0.34 |
| LSZH | 0.23 / 0.21 | 0.09 / 0.10 | 0.52 / 0.61 |
| LTFM | 0.21 / 0.44 | 0.51 / 0.37 | – |
| LEBL | −0.22 / −0.24 | – | – |
| LIRF | – | – | – |

The two airports with no winter (LEBL, LIRF) show no correlation, which is the
control this table needs to be believed.

**Consequence.** The January half of the board score will be worse than any
local fold, as the desk expects. On the specific days that make it worse, two
ranking-file inputs separate them from an ordinary winter day: **weather**, and
the **within-month airport-day aggregates of `MVT − AOBT_3`**, which are equally
observable and are a direct measurement rather than an inference. An earlier
draft called weather "the only" such input; that was wrong, and it matters,
because the second input is exactly what makes weather's marginal value small on
a model that already has it (section 6.3, and exp-002 it12). What weather adds
over the aggregate is an independent physical cause that does not have to be
estimated from the month's own thin per-day samples.

---

## 6. The ablation (assignment step 3)

Frozen LightGBM, one configuration never tuned, no early stopping. Base = own-flight
deltas + airport, stand, runway, aircraft type, operator, hour, day of week, plus the
three noac columns and the `stand|runway` pair the desk routed from `exp-002` and
`docs/shift-2026.md`. Learning target is the residual `taxi − nm_taxi` anchored to 0
on no-NM rows, so L2 on the residual is L2 on taxi (verified algebraically by the
reviewer). All numbers below are from `results_v5.jsonl`, one lineage, produced
after the review fixes; OOF hashes in `oof/SHA256SUMS.txt`. Weather module
`prc.weather_features` VERSION 2.

### 6.1 Family ablation with the four mandatory slices

Every row below is current evidence on the post-review code and cache, seed 0,
from `results_v5.jsonl` (base, all, winter_core, winter_core_plus, noaobt_base,
noaobt_all) and `results_v6.jsonl` (noaobt_winter_core and the eight families). RMSE overall / bulk (≤3600
**and** NM-joined) / ≤3600 / >3600 / no-NM, per COMMON.md. Δ is against `base`,
or against `noaobt_base` for the `noaobt_` rows. **These tables are generated by
`code/make_tables.py` from the JSONL records, not transcribed** — two review
findings were transcription errors, so the class is designed out.

**january**

| config | wx cols | overall | Δ | bulk | Δ | ≤3600 | >3600 | no-NM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 383.45 | – | 211.54 | – | 228.79 | 6696 | 3253 |
| basic | 6 | 381.17 | -2.28 | 208.15 | -3.39 | 227.46 | 6656 | 3253 |
| precip | 11 | 384.73 | +1.28 | 212.06 | +0.52 | 229.89 | 6713 | 3270 |
| runway_wind | 5 | 385.07 | +1.62 | 211.83 | +0.29 | 231.17 | 6701 | 3273 |
| visibility | 4 | 383.94 | +0.49 | 210.06 | -1.48 | 228.03 | 6721 | 3271 |
| change | 5 | 381.62 | -1.83 | 211.14 | -0.40 | 226.69 | 6680 | 3234 |
| aobt | 13 | 383.27 | -0.18 | 207.33 | -4.21 | 227.63 | 6710 | 3284 |
| no_aobt | 31 | 377.20 | -6.25 | 208.10 | -3.44 | 224.47 | 6596 | 3202 |
| precip_vis | 15 | 379.05 | -4.40 | 210.65 | -0.89 | 228.81 | 6576 | 3209 |
| all | 44 | 377.80 | -5.65 | 207.72 | -3.82 | 223.78 | 6623 | 3213 |
| winter_core | 7 | 379.93 | -3.52 | 207.12 | -4.42 | 228.02 | 6613 | 3246 |
| winter_core_plus | 12 | 376.71 | -6.74 | 207.52 | -4.02 | 226.02 | 6558 | 3202 |
| noaobt_base | 0 | 411.78 | – | 255.21 | – | 268.57 | 6793 | 3260 |
| noaobt_winter_core | 7 | 397.14 | -14.64 | 236.03 | -19.18 | 253.12 | 6660 | 3240 |
| noaobt_all | 31 | 397.39 | -14.39 | 238.28 | -16.93 | 252.56 | 6677 | 3222 |

**december**

| config | wx cols | overall | Δ | bulk | Δ | ≤3600 | >3600 | no-NM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 292.08 | – | 206.83 | – | 223.39 | 4746 | 1979 |
| basic | 6 | 286.19 | -5.89 | 205.52 | -1.31 | 225.13 | 4458 | 1904 |
| precip | 11 | 285.21 | -6.87 | 205.96 | -0.87 | 218.96 | 4610 | 1886 |
| runway_wind | 5 | 289.50 | -2.58 | 207.16 | +0.34 | 223.36 | 4646 | 1937 |
| visibility | 4 | 290.15 | -1.93 | 207.02 | +0.19 | 224.06 | 4650 | 1954 |
| change | 5 | 301.70 | +9.62 | 206.72 | -0.11 | 224.60 | 5080 | 2122 |
| aobt | 13 | 289.77 | -2.31 | 205.20 | -1.63 | 221.10 | 4724 | 1961 |
| no_aobt | 31 | 291.80 | -0.28 | 205.31 | -1.51 | 225.10 | 4684 | 1989 |
| precip_vis | 15 | 284.85 | -7.23 | 205.89 | -0.94 | 219.85 | 4568 | 1881 |
| all | 44 | 292.92 | +0.84 | 205.40 | -1.43 | 224.74 | 4738 | 2004 |
| winter_core | 7 | 279.35 | -12.73 | 204.77 | -2.05 | 220.23 | 4336 | 1807 |
| winter_core_plus | 12 | 285.03 | -7.05 | 204.88 | -1.95 | 222.38 | 4498 | 1894 |
| noaobt_base | 0 | 316.22 | – | 237.42 | – | 251.77 | 4827 | 1982 |
| noaobt_winter_core | 7 | 294.73 | -21.49 | 226.10 | -11.31 | 238.95 | 4354 | 1775 |
| noaobt_all | 31 | 307.53 | -8.69 | 226.99 | -10.42 | 243.81 | 4729 | 1971 |

**july**

| config | wx cols | overall | Δ | bulk | Δ | ≤3600 | >3600 | no-NM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 513.82 | – | 255.13 | – | 353.96 | 6360 | 3036 |
| basic | 6 | 516.64 | +2.83 | 252.59 | -2.55 | 344.30 | 6577 | 3067 |
| precip | 11 | 518.84 | +5.02 | 254.40 | -0.73 | 363.73 | 6319 | 3078 |
| runway_wind | 5 | 514.18 | +0.36 | 254.47 | -0.67 | 348.24 | 6460 | 3042 |
| visibility | 4 | 510.85 | -2.97 | 254.01 | -1.12 | 352.36 | 6316 | 3014 |
| change | 5 | 511.52 | -2.30 | 255.84 | +0.71 | 354.93 | 6290 | 3010 |
| aobt | 13 | 512.07 | -1.75 | 252.52 | -2.62 | 352.07 | 6350 | 3032 |
| no_aobt | 31 | 517.99 | +4.17 | 252.14 | -2.99 | 356.31 | 6420 | 3078 |
| precip_vis | 15 | 512.55 | -1.27 | 253.12 | -2.01 | 354.84 | 6316 | 3031 |
| all | 44 | 516.65 | +2.83 | 251.50 | -3.63 | 351.52 | 6465 | 3068 |
| winter_core | 7 | 511.91 | -1.91 | 252.41 | -2.72 | 345.25 | 6454 | 3029 |
| winter_core_plus | 12 | 520.28 | +6.47 | 253.12 | -2.01 | 361.34 | 6393 | 3096 |
| noaobt_base | 0 | 549.07 | – | 271.51 | – | 363.60 | 7025 | 3007 |
| noaobt_winter_core | 7 | 548.46 | -0.61 | 263.40 | -8.11 | 355.58 | 7129 | 3028 |
| noaobt_all | 31 | 554.21 | +5.14 | 262.63 | -8.88 | 361.72 | 7169 | 3081 |

Single families are not separable at fold resolution: `precip` alone is +1.28 on
january and −6.87 on december, `change` is −1.83 and +9.62. Only the paired
cluster bootstrap below distinguishes signal from fold noise, and only the bulk
slice does so consistently.

### 6.2 Where the gain is, under an airport-day cluster bootstrap

Flights in one airport-day share a weather realisation, so a flight-level
bootstrap is far too narrow for exactly the conditional slices this experiment
cares about. Resampling unit is the (airport, UTC day) cluster.
**These intervals are post-selection**: `winter_core` was chosen by looking at
these folds, so they describe the spread of the measured difference on these
rows, not a guarantee for a set chosen in advance.

base vs `winter_core`, Δ [95 % cluster interval]; **bold excludes zero**
(`logs/cluster_bootstrap_base_vs_winter_core.log`):

| slice | january | december | july |
|---|---|---|---|
| overall | −3.52 [−11.96, +4.09] | −12.73 [−27.94, +0.60] | −1.91 [−11.85, +7.72] |
| **bulk (≤3600 & NM)** | **−4.42** [−6.75, −2.31] | **−2.05** [−3.08, −1.04] | **−2.72** [−3.94, −1.51] |
| ≤3600 s | −0.77 [−6.05, +6.08] | −3.16 [−7.80, +0.35] | −8.71 [−17.11, +0.14] |
| >3600 s | −83.0 [−300, +110] | −410.5 [−833, +144] | +93.3 [−91, +269] |
| no-NM | −7.1 [−119, +115] | −172.3 [−340, +53] | −6.7 [−89, +74] |
| snow or freezing hours | −26.0 [−79.5, +26.1] | **−54.4** [−84.8, −9.5] | – |
| bulk ∩ snow/freezing | −16.1 [−45.4, +6.3] | **−27.8** [−43.2, −2.9] | – |
| other (non-winter) hours | −2.9 [−11.8, +4.8] | −12.2 [−27.4, +1.8] | −1.9 [−12.6, +7.5] |
| temp ≤ 3 °C | −5.5 [−22.8, +8.2] | **−7.6** [−14.8, −0.7] | – |
| low visibility | **−7.8** [−25.2, −2.4] | −11.7 [−32.4, +2.8] | +3.6 [−4.9, +12.0] |
| wind ≥ 8 m/s | −1.9 [−11.9, +7.5] | +3.0 [−5.6, +9.0] | **+62.6** [+3.3, +120.8] |

**How many clusters are behind each interval.** The conditional slices rest on
few airport-days: december's snow/freezing slice has **9** clusters and july's
wind slice **18**. A nominal 95 % percentile interval from 9 clusters is
optimistic, so those two rows should be read as direction plus order of
magnitude, not as calibrated intervals (Sol round 2).

**What survives clustering, and what does not.** The bulk gain survives on all
three folds and is the claim to rely on. The dramatic conditional numbers are
large in point estimate but wide: december's snow slice still excludes zero,
january's does not, because january's 5,120 snow rows sit in few airport-days.
An earlier draft of this file reported the flight-level intervals — january snow
−25.7 [−38.0, −13.8] — which overstated the precision roughly threefold. The
tail slices (>3600 s, no-NM) never exclude zero on any fold: **the gain is not
concentrated in the tail**, it is concentrated in the bulk and, less precisely,
in winter conditions.

### 6.3 Why the gain is small: the actual-time columns already carry it

The control removes `AOBT_3_flt` and every own-flight delta derived from it, and
sets the anchor to 0, so the model must regress absolute taxi time.

Two earlier versions of this control were wrong and both are superseded. The
first kept three AOBT-derived weather columns (`wx_taxi_d_temp_c`,
`wx_taxi_d_wind_speed_ms`, `wx_taxi_precip_mm`, which carry no `wxo_` prefix) —
Sol round 1, finding 1. The second removed those but compared **7** weather
columns on the anchored side against **31** on the unanchored side, confounding
the anchor with 24 extra columns — Sol round 2, finding 1. The table below is
the matched pair: **exactly the seven `WINTER_CORE` columns on both sides**,
differing only in whether the AOBT anchor is present.

Bulk RMSE gain from those seven columns, with airport-day cluster intervals.
**Generated by `code/make_tables.py`** from `results_v5/v6.jsonl` and the two
bootstrap logs (bold = interval excludes zero):

| fold | with the AOBT anchor | without it | ratio of point gains |
|---|---|---|---:|
| january | **-4.42** [-6.75, -2.31] | **-19.18** [-26.31, -12.32] | 4.3x |
| december | **-2.05** [-3.08, -1.04] | **-11.31** [-15.26, -7.74] | 5.5x |
| july | **-2.72** [-3.94, -1.51] | **-8.11** [-9.95, -6.20] | 3.0x |

conditional slices, same seven columns:

| slice | fold | with anchor | without |
|---|---|---|---|
| snow/freezing hours | january | -26.03 [-79.47, +26.08] | **-79.85** [-141.93, -10.37] |
| snow/freezing hours | december | **-54.43** [-84.81, -9.54] | **-145.24** [-194.82, -53.12] |
| bulk & snow/freezing | december | **-27.75** [-43.19, -2.94] | **-138.12** [-192.51, -54.79] |
| temp<=3C | december | **-7.59** [-14.80, -0.72] | **-36.22** [-50.31, -22.34] |

**This is the answer to the assignment's question.** The same seven weather
columns are worth **3.0 to 5.5× more** when `AOBT_3_flt` is absent — so roughly
two thirds to four fifths of weather's value is already carried by the
actual-time columns. That 3.0–5.5× is a **range of point-estimate ratios across
the three folds, not a confidence interval for the ratio**; the intervals shown
apply to each RMSE gain separately (Sol round 3).
The mechanism: because the observed take-off-minus-off-block interval *is*
a measurement of the conditions weather would otherwise have to infer. (Earlier
drafts said "4–8×" and then "3.3–5.1×"; both came from unmatched comparisons and
are withdrawn in favour of this one.)

The same mechanism explains the tail worker's independent result
(`docs/LEDGER.md`, exp-002 it12): on their regime-router model, which adds
airport-day and airport-hour aggregates of `MVT − AOBT_3`, this module is worth
−0.9 s on january and they did not adopt it. Those aggregates measure the day's
disruption directly; weather infers the same thing from an 0.25° grid cell. When
both are available the measurement wins. Two independent bases, one consistent
picture.

### 6.4 The repair, and the set to integrate

The first pass produced one negative: the full 44-column block made july's
≤3600 s slice worse by +11.5 s — a **pre-fix, flight-level** number from
`results_v1.jsonl` on the weaker base, quoted here as the motivation for the
repair, not as current evidence (everything else in 6.1–6.4 is from
`results_v5.jsonl`/`v6`). Diagnosis: 44 weak columns on the 3,938 no-NM
rows, which have no anchor and must regress absolute taxi. Two compact sets were
built as the repair.

| config | wx cols | january | december | july | improves all three overall? |
|---|---:|---:|---:|---:|:--:|
| all | 44 | 377.80 | 292.92 (**+0.84**) | 516.65 (**+2.83**) | no |
| winter_core_plus | 12 | **376.71** | 285.03 | 520.28 (**+6.47**) | no |
| **winter_core** | **7** | 379.93 | **279.35** | **511.91** | **yes** |

**Three** tested sets lower seed-0 overall RMSE on all three folds:
`winter_core` (−3.52 / −12.73 / −1.91), `precip_vis` (−4.40 / −7.23 / −1.27) and
`aobt` (−0.18 / −2.31 / −1.75). An earlier draft called `winter_core` the only
one, twice, which the v6 table contradicts (Sol rounds 2 and 3). What is true:
of those three it is the **smallest** (7 columns against 15 and 13), it has much
the **largest december gain**, and it is the only one whose bulk gain was put
through the cluster bootstrap on all three folds and excluded zero each time. It is **not** better than the 44-column block everywhere: `all` has the better
july bulk (251.50 vs 252.41) and the better january ≤3600 (223.78 vs 228.02).
(An earlier draft also credited `all` with the larger january snow gain; it is
−25.97 against `winter_core`'s −26.03, so that one goes the other way.) An earlier draft claimed seven columns beat
forty-four "on every conditional slice", which the stored numbers contradict
(Sol review, finding 3). The case for `winter_core` is robustness and cost, not
dominance.

**Seed noise** (january, three seeds, pre-fix cache, `results_seeds.jsonl`):
base spread 2.67 s overall / 0.33 bulk; `all` 0.98 / 0.70; `winter_core`
1.60 / 0.59. Bulk gains clear the seed floor by 6–13×; single-fold *overall*
differences below about 3 s do not, which is why no conclusion here rests on one.

**One negative left standing.** On july, rows with wind ≥ 8 m/s (n = 1,785) are
worse with `winter_core`: **+62.6 s [+3.3, +120.8]**, and `all` moves the same
slice +26.7 s. It survived clustering, but that does not rule out multiplicity: one
unadjusted exclusion among 33 comparisons is entirely plausible, and the slice
rests on only 18 airport-day clusters. Treat it as a flag to test, not an
established effect. Repair not yet measured: drop `wx_wind_speed_ms` and
keep the other six columns, which costs the (insignificant) january and december
wind slices and should leave the winter effect intact. A consumer that only
wants winter behaviour should test that variant first.

### 6.5 Ceiling estimate for the lane

The assignment asked for an estimate after a first look at december's snow and
low-visibility hours. Stated now, with the measurements behind it:

- Weather's value **without the AOBT anchor** is 8–19 s of bulk RMSE
  (section 6.3, matched control; the old 9–17 s came from the unmatched one). That is an empirical gain measured on one model family, not a
  physical ceiling on what weather could ever be worth.
- With `AOBT_3_flt` present, 2–4.4 s of bulk survives; on a model that also has
  airport-day `MVT − AOBT` aggregates, ~1 s (exp-002 it12).
- January 2026 carries about 1.5× January 2025's snow at EDDF (27.8 vs 18.1 mm)
  and, summed over the ten ranking airports, **11.6× december 2025's**
  (112.03 vs 9.63 mm; an earlier draft said 5.4×, which was wrong). So the
  january-fold number is the right order for the ranking month's January half,
  and the december one is not.
- **Estimate: 1 to 3 s of overall RMSE on the full ranking file for a model that
  already exploits the actual columns well, concentrated in the January half.**
  This is a real but second-order lever against a board spread of 277 to 361 s.

Note that a fold gain is *not* a lower bound on the ranking-month gain, as an
earlier draft implied: covariate shift can make extrapolation worse as easily as
better (Sol review, finding 7). The argument for expecting more in January 2026
is the exposure difference, and it is an expectation, not a bound.

---

## 7. THE SET TO INTEGRATE (named column list)

Produced by `prc.weather_features.transform(month_frame)`. Import the list, do
not retype it: `from prc.weather_features import WINTER_CORE`.

```python
WINTER_CORE = [
    "wx_temp_c",                  # 2 m temperature, degC       - de-icing regime
    "wx_precip_mm",               # precip in the taxi hour, mm  - accumulation over [H, H+1)
    "wx_snow_6h_mm",              # snow water equiv., 6 h, mm   - what is on the surface
    "wx_freezing_precip",         # temp <= 2 degC and precip > 0.1 mm  (NaN if unknown)
    "wx_subzero",                 # temp <= 0 degC                      (NaN if unknown)
    "wx_dewpoint_depression_c",   # temp - dew point, degC       - visibility proxy
    "wx_wind_speed_ms",           # 10 m wind speed, m/s         - runway configuration
]
```

`WINTER_CORE_PLUS` (12 columns) adds `wx_cloud_low`, `wx_snow_3h_mm`,
`wx_crosswind_abs_ms`, `wx_headwind_ms`, `wx_msl_hpa`. It gives january's best
measured RMSE (376.71) but is worse on july, so it is for a January-specific
model only.

Integration contract:

- `transform(month_frame)` returns a frame with `month_frame`'s index and row
  order, 45 float32 columns, for **any** movement frame — a masked 2025 month
  or the ranking file. It reads only `ADEP_mvt`, `MVT_TIME_UTC_mvt`,
  `AOBT_3_flt` and `RUNWAY_mvt`, all present in the ranking file.
- Verified on the live ranking file: **all 344,841 departures joined,
  `wx_missing` = 0.000000 in both ranking months.** `WINTER_CORE` is not quite
  100 % covered, and an earlier draft understated the gap. Exact nulls, VERSION 2:

  | column | January 2026 (152,719 dep) | July 2026 (192,122 dep) |
  |---|---:|---:|
  | `wx_snow_6h_mm` | 15 (0.010 %) | 157 (0.082 %) |
  | `wx_precip_mm` | 0 | 11 (0.006 %) |
  | `wx_freezing_precip` | 0 | 11 (0.006 %) |
  | the other four | 0 | 0 |

  The `wx_snow_6h_mm` nulls are the full-window requirement at a period edge:
  the first hours of July 2026 need June hours that the extra product was never
  retrieved for, and the last hour of January has no `H+1` stamp. LightGBM takes
  the NaN natively; a consumer that wants zero nulls should fill only after
  deciding that "no snow retrieved" means "no snow", which it does not.
- Missing values are real NaN, never a fill — true from VERSION 2 onward. The
  single difference from VERSION 1 is the rolling 3/6-hour sums: VERSION 1 used
  `min_periods=1` and returned a partial sum when an hour of the window was
  missing. VERSION 1 does **not** turn missing condition inputs into False —
  that was a pre-commit draft, never a released version, and an earlier draft of
  this section wrongly described VERSION 1 that way (Sol round 3).
  `build_hourly(version=1)` reproduces commit `38ef8e3` with zero differing data
  columns (it does set a `weather_features_version` frame attribute, so the
  serialized parquet bytes are not byte-identical). Two flags say why a row is
  missing: `wx_missing` (no weather hour) and `wx_missing_aobt_hour` (no NM
  join, so no off-block weather — 1.6 % of january and 1.5 % of july ranking
  departures).
- Rolling-window convention: `wx_snow_3h_mm` and friends cover the `window`
  hours **ending with and including** the joined clock hour, i.e. accumulation
  over `[H−window+1, H+1)`. The assignment's "snow in the last 3 and 6 hours" is
  ambiguous between this and the hours strictly before `H`; this is the reading
  implemented.
- No fitting, no statistics, no train/valid split: weather is a function of
  (airport, clock hour), so `transform` is stateless and there is nothing to
  leak across folds.

---

## 8. Artifacts and hashes

| what | path | size | SHA-256 |
|---|---|---:|---|
| hourly airport-point table (VERSION 2) | `artifacts/weather/era5_hourly.parquet` | 15,181,053 B | `8238378ec1f6d66a6c59712993ec54ac72af3f130cb7a8a7754a3fbec369f293` |
| raw CDS snapshot manifest | `data/external/weather/copernicus_era5/manifest.json` | 46 rows (44 production + 2 test) | hashes of every NetCDF; retrieval **complete** |
| per-month feature cache | `artifacts/exp-005-weather/cache/*.parquet` | 14 months | rebuildable by `code/build_cache.py` |
| OOF predictions | `experiments/exp-005-weather/oof/` | 45 parquet (15 configs × 3 folds) + importance CSVs | `MVT_ID_mvt, fold, y_true, y_pred`; SHA-256 of each in `oof/SHA256SUMS.txt`, all tracked in git |

Raw ablation records, in lineage order:

| file | what | status |
|---|---|---|
| `results_v5.jsonl` | 6 configs × 3 folds on the post-review code and cache | current evidence |
| `results_v6.jsonl` | the matched no-AOBT control + the 8 single families, same code and cache | current evidence |
| `results_v2.jsonl` | full family ablation, pre-fix cache and leaky control | diagnosis trail only |
| `results_v3.jsonl` | compact sets, pre-fix cache | superseded by v5 |
| `results_seeds.jsonl` | 3-seed noise, january, pre-fix cache | still cited (noise floor is cache-insensitive) |
| `results_v1.jsonl` | first pass, weaker base | diagnosis trail only |
| `logs/cluster_bootstrap_*.log` | the three clustered comparisons behind section 6.2–6.3 | current, one log per comparison |

The v1–v3 OOF files were overwritten by the v5 rerun, so the flight-level
intervals logged earlier are no longer reproducible (Sol round 1, finding 4).
**The closed lineage is `results_v5.jsonl` + `results_v6.jsonl` + all 45 OOF
parquet with `oof/SHA256SUMS.txt` + the four `logs/cluster_bootstrap_*.log`**,
all tracked in git; `code/make_tables.py` regenerates the §6.1 and §6.3 tables
from them. Superseded numbers are kept as narrative, not as evidence.

## 9. What would change this conclusion

- **A model that measures the day directly.** Airport-day and airport-hour
  aggregates of `MVT − AOBT_3` already capture most of what weather infers
  (exp-002 it12: −0.9 s on january, not adopted). If those aggregates get
  stronger, weather's residual value falls further. This is the main reason to
  expect ~1 s, not ~5 s, from integrating into the current best model.
- **The January 2026 out-of-envelope days.** This is the case *for* carrying it
  anyway: every day clearing its airport's 2025 maximum by more than 30 s is a
  winter day, and the ranking month has more snow than any 2025 month (EDDF
  27.8 mm against 18.1 mm in January 2025). This makes a larger january-half
  gain *expected*, not guaranteed — covariate shift can degrade extrapolation as
  easily as improve it, so this is not a lower bound.
- **The one unresolved eligibility question is not event time, it is access.**
  `docs/input-rule.md` settles the retrospective reading, and weather cannot
  reconstruct the blanked target in any case — it is a function of (airport,
  clock hour) shared by every flight in that hour. What is *not* settled by the
  published rules is whether CDS's free-but-required registration counts as
  "openly accessible" under the eligibility page
  (`docs/weather-provenance.md` §1). The licence condition is met unambiguously
  (CC BY 4.0, commercial use with attribution). This is the desk's supplied-input rule organizer
  question, not something this experiment can close; if the answer is no, the
  lane needs an unregistered open source (METAR/SYNOP) and the feature design
  here transfers unchanged.
- **Airport-level de-icing data** would beat an 0.25° grid cell. ERA5 says the
  weather; it does not say whether the de-icing pads were saturated. If an open
  source for de-icing or runway-state (SNOWTAM/METAR runway condition codes)
  can be licensed, that is the next lane, not more ERA5 variables.
- **What did not work and is not worth retrying as-is** (current v6 numbers):
  the full 44-column block (variance on the no-NM rows), the 3-hour-change family
  alone (**+9.62 s** on december), and runway-aligned wind as a standalone family
  (**+1.62 s** on january). Wind in summer is a live negative (section 6.4).

---

## 10. Reviewer

`gpt-5.6-sol-xhigh-2026-09-05`, round 1, `logs/sol_review.log`. Four blocking
findings, all accepted; every one changed a number or a claim in this file.

| # | finding | response |
|---|---|---|
| 1 | the no-AOBT control still fed 3 AOBT-derived weather columns (`wx_taxi_d_*`, `wx_taxi_precip_mm`), so the causal claim was invalid; the table also said 2.5× for july, not "4–8×" | `ablate.py` now excludes all 13 `aobt`-family columns; control rerun. Corrected ratio **3.3–5.1×** (was "4–8×") |
| 2 | flight-level bootstrap assumes independence; airport-days share a weather realisation, and the set was chosen on these folds | `analyse.py` resamples (airport, UTC day) clusters; all intervals relabelled post-selection. **January's snow slice no longer excludes zero** |
| 3 | "7 columns beat 44 on every conditional slice" is false on ≥4 slices | claim withdrawn and replaced with the supported one (only set with lower seed-0 overall on all three folds); the slices where `all` wins are named |
| 4 | a rerun overwrote `oof/base_january.parquet`, so logged intervals no longer reproduced | everything regenerated into `results_v5.jsonl` + 18 OOF files with `oof/SHA256SUMS.txt`; superseded records marked narrative-only |
| 5 | rolling-window convention ambiguous vs the assignment's "last 3 and 6 hours" | convention stated explicitly in code, section 7 and provenance §10 |
| 6 | `min_periods=1` turned partial windows into values, contradicting "never a fill" | now requires the full window; VERSION 2, with `build_hourly(version=1)` reaching the old behaviour |
| 7 | calling the fold gain a "lower bound" on the ranking gain overstates it | withdrawn; restated as an expectation, with the covariate-shift caveat |
| 8 | "all 27 are weather days" false (19 snow/freezing, 26 incl. sub-zero, LEBL 21 Jan none); "weather is the only input" contradicted | corrected to the margin-conditioned claim (18/18 above 30 s) and LEBL named; the `MVT − AOBT` aggregate acknowledged as the second input |
| 9 | missing mandatory slice columns, no ceiling estimate, stale provenance, no verdict | four-slice tables in 6.1, ceiling in 6.5, provenance updated to 44/44 files and 46 manifest rows, this section |

Confirmed correct by the reviewer and unchanged: no path by which any feature
reads the masked `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`; the anchored
residual makes reported RMSE exactly RMSE on taxi; the ERA5 `shift(-1)` relabel
is correct and the `d3` features sit on the right side of it; the january fold
is not leakage; all 13 raw-join checks pass; the documented artifact hash
matches.

Self-checks added while answering, both passing and both runnable:

- `code/test_cluster_bootstrap.py` — the vectorised cluster bootstrap
  (`reduceat` pre-summing plus size re-weighting) reproduces a brute-force
  whole-cluster resample to 1e-9 on the same RNG stream.
- the feature list every config actually builds contains no `takeoff_hour_utc`
  (the cluster key), no `y`, and, for the `noaobt_` configs, no member of the
  `aobt` family and no `wxo_` column: `noaobt_all` builds 47 features, matching
  `results_v5.jsonl`.
- `ablate.py` now imports `WINTER_CORE` / `WINTER_CORE_PLUS` from
  `prc.weather_features` instead of re-declaring them, so the delivered list and
  the measured list cannot drift.

### Round 2 — `logs/sol_review_round2.log`, verdict **still blocked**, four new findings

| # | finding | response |
|---|---|---|
| 1 | the no-AOBT multiplier compared 7 weather columns with the anchor against 31 without, confounding the anchor with 24 extra columns; section 1 also transcribed the range as 3.3–3.8× when the table said 3.3–5.1× | added `noaobt_winter_core` (**exactly** the `WINTER_CORE` columns, anchor removed) and reran; matched ratio **3.0–5.5×**, every interval excluding zero. Both earlier ratios withdrawn (§6.3) |
| 2 | `build_hourly(version=1)` also reverted the NaN-flag fix, which predates the reviewed commit, so it did not reproduce `38ef8e3` | version now differs from v2 **only** by `min_periods`; rebuilt against `git show 38ef8e3:prc/weather_features.py` — **zero differing columns**. Docstring records that no released version read a missing input as False |
| 3 | the exported module comment still carried the withdrawn dominance claim with stale numbers | rewritten to what the evidence supports, with the slices where `all` wins named |
| 4 | 8 of 18 OOF files untracked; §2.1 still said LSZH 2024 queued and 39/40; §2.2 quoted the superseded artifact hash | all OOF committed with refreshed `SHA256SUMS.txt`; §2.1 now says 44/44 retrieval complete; §2.2 carries the VERSION 2 size and hash |
| – | december "other hours" and july "other hours" intervals transcribed from the wrong rows | corrected; and §6.1/§6.3 tables are now **generated** by `code/make_tables.py` from the JSONL, so transcription errors are designed out |
| – | §6.5 called the no-anchor gain a "physical ceiling"; the january/december snow ratio was 5.4× | restated as an empirical gain on one model family; ratio corrected to **11.6×** (112.03 vs 9.63 mm, verified against the artifact) |
| – | §7 coverage claim false: `wx_snow_6h_mm` null for 15 january and 157 july ranking departures | exact null table added; the cause is my own full-window fix at period edges |
| – | july wind: "survived clustering, so unlikely to be multiplicity" is backwards; december snow rests on 9 clusters, july wind on 18 | both recorded as stated cautions in §6.2 and §6.4; the wind result is a flag to test, not an established effect |
| – | single-family ablations were not current scored evidence; no v5 entry in `climb.log` | all eight families rerun on post-review code (`results_v6.jsonl`) and now in §6.1; `climb.log` it20–it23 |

Sol confirmed in round 2: the `reduceat` cluster-bootstrap arithmetic is correct
and reproduces the saved logs; no model-feature leakage through
`takeoff_hour_utc`, `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`; all 18 section
6.1 rows matched the JSONL; and the corrected alignment claim in §5 is supported
exactly by `alignment_2026.csv`.

### Round 3 — `logs/sol_review_round3.log`, **matched control validated**, four documentation blockers

Sol recomputed the matched control independently (4.34× / 5.50× / 2.98×) and
confirmed the §6.3 intervals reproduce from the committed OOF files, that all 45
OOF parquet are tracked with passing hashes and the right schema, that every OOF
overall RMSE matches its JSONL row, and that VERSION 1 has zero differing data
columns from `38ef8e3`. **The substantive result is verified.** The four
blockers were all accuracy of the write-up:

| # | finding | response |
|---|---|---|
| 1 | "only tested set improving all three folds" is false — `precip_vis` and `aobt` do too | corrected in §6.4 and in the module comment; the supported claim is that of the three, `winter_core` is the smallest, has much the largest december gain, and is the only one bootstrapped on all three folds |
| 2 | §§3–4 quoted `winter_effect_v1.txt`, taken before LSZH's extra product landed | rerun on the current artifact (`winter_effect_v2.txt`): 603,484 / 16,413 rows, Δ sd 357.1 → 642.0, december zero-snow airports 6 of 10 not 7; §3 table now includes LSZH (13.1 / 0.8 / 21.0 mm) and LTFM |
| 3 | `make_tables.py` generated §6.1 but not §6.3, so the "generated" claim was false | generator extended to parse the bootstrap logs and emit §6.3's six intervals and four conditional rows; §6.3 is now its output |
| 4 | stale prose: VERSION 1 described as flag-flipping, 18 OOF files, v5-only lineage | all corrected; the closed lineage is v5 + v6 + 45 OOF + four bootstrap logs |
| – | +6.46→+6.47; `all` credited with the larger january snow gain (it is −25.97 vs −26.03); 9–17 s → **8–19 s** matched range; §9 family numbers were v2, now v6 (+9.62, +1.62) | all corrected |
| – | 3.0–5.5× read as an interval | labelled a range of point-estimate ratios across folds, not a confidence interval |
| – | §4's Δ is `BLOCK − AOBT`, the negative of the fitted residual | sign convention stated explicitly in §4 |
| – | `ablate.py` assembles folds from cache rather than calling `fold_frames` | `code/test_fold_equivalence.py` proves the two are identical on all three folds — same ids, order and labels, masked columns absent (18/18 pass) |

**Review rounds: three.** COMMON.md allows two. The cap was exceeded because
each round found real defects — round 1 an invalid control, round 2 an unmatched
one, round 3 documentation that had drifted behind the reruns. Round 3's fixes
have **not** been through a further round; that is the desk's call.
