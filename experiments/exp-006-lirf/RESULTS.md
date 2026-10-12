# exp-006-lirf — RESULTS

exp-id `exp-006` | worker `lirf` | tier hard | reviewer: pending
Assignment `assignments/lirf.md`. Reading and rejected readings: `SPEC.md`.
Code: `run_lirf.py`, `work/`. Modules handed back: `prc/phi_features.py`,
`prc/lirf_features.py`. Out-of-fold predictions: `oof/`. Every attempt:
`climb.log`. Data: `prc.data.data_dir()`, live ranking SHA-256 `9db75e3a…`.

Baseline for every comparison: retro-v1's best climb line **`c6-noac`**, not its
latest `c9-te` (c9 added target encodings and lost 43 s on july; retro-v1's own
conclusion is that encodings are off in the shipped lineage). Its LIRF numbers,
computed from `experiments/exp-001-retro-v1/oof/c6-noac-*.parquet`:

| fold | whole fold | LIRF | other nine | LIRF share of fold SSE | LIRF rows |
|---|---|---|---|---|---|
| july | 343.59 | **864.62** | 251.98 | **50.5 %** | 15,211 (8.0 %) |
| december | 222.43 | 371.10 | 206.26 | 20.3 % | 12,068 (7.3 %) |
| january | 349.03 | 387.07 | 345.83 | 9.1 % | 11,317 (7.4 %) |

## Ceiling: what LIRF RMSE the leaders must be holding

LIRF is 26,899 of the 344,841 ranking departures, **7.80 %**. With the board top
at 277.3 s and the other nine airports at RMSE `s`, the leaders' LIRF RMSE is
`sqrt((277.3² − 0.922·s²)/0.078)`:

| other nine at | 200 | 220 | 240 | 260 |
|---|---|---|---|---|
| **leaders' LIRF RMSE at 277.3 s** | 716 | 643 | 552 | 432 |
| the same at 302 s (top of the chasing pack) | 835 | 773 | 699 | 608 |

Our own other-nine numbers are 252 (july) / 206 (december) / 346 (january), so
the honest reading is that the leaders sit near the **550–720 s** band at LIRF.
Blending our folds by **LIRF's own** ranking mix — 15,838 July and 11,061 January
LIRF departures, 58.88 % / 41.12 %, not the file's overall 56/44 (gpt-5.6-sol):

| | LIRF | other nine | implied total |
|---|---|---|---|
| retro-v1 `c6-noac` | `sqrt(.589·864.6² + .411·387.1²)` = **708** | `sqrt(.589·252.0² + .411·345.8²)` = **294** | 345 |
| this experiment's delivered line | **602** | **293** | **328** |
| the leaders, at 277.3 | 550–720 | 200–240 | 277 |

**LIRF is inside the band the leaders must be in; the other nine airports are
where the distance is.** Our ~293 s over the other nine against their 200–240 s is
53–93 s of gap on 92 % of the rows, while our LIRF number sits inside their
550–720 s range on 8 %. The board gives a *trade-off curve*, not a decomposition:
it cannot prove the leaders are not also better at LIRF, only that if they hold
200–240 s elsewhere then 550–720 s at LIRF is what closes their arithmetic. That is the first thing this experiment changes about
the desk's picture, and it is why the ranked findings below lead with a mechanism
that turns out to be *general* rather than with a LIRF-only model.

Two caveats on that arithmetic. The blend uses the backward `january` fold
(January 2025) for the January half, and `docs/shift-2026.md` says January 2026
is harder than January 2025 at EDDF, EHAM, EDDM and LSZH — so the other-nine
figure is optimistic and the gap is, if anything, larger. And it is a fold blend,
not a leaderboard read: the only thing it can settle is *which* airports carry
the remaining distance, not what the score will be.

---

## Findings, ranked

### 1. The airport block time sits a fixed fraction of the way from NM's actual off-block to the filed one, and the fraction is a per-operator convention

This is the finding. Write

```
gap = LOBT_flt − AOBT_3_flt              (filed minus actual off-block; observable)
r   = BLOCK_TIME_UTC_mvt − AOBT_3_flt    (the unobservable the task reduces to)
```

Then `r ≈ phi · gap`, and `phi` is a stable property of the airport. Over all
twelve 2025 months, under the two estimators `prc/phi_features.py` offers — the
shipped default averages the per-flight ratio over flights with `|gap| ≥ 300 s`
(`weight="rows"`), the alternative is the least-squares coefficient
`sum(r·g)/sum(g²)` (`weight="gap2"`):

| | EHAM | EDDF | LSZH | LEMD | EGLL | LEBL | EDDM | LFPG | **LIRF** | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| `rows` (shipped) | −0.06 | 0.03 | 0.13 | 0.14 | 0.17 | 0.22 | 0.22 | 0.24 | **0.40** | 0.41 |
| `gap2` | −0.04 | 0.06 | 0.13 | 0.15 | 0.25 | 0.17 | 0.25 | 0.20 | **0.38** | 0.40 |

The two agree on the ordering at the top and the bottom and disagree in the
middle; the shipped one is `rows` because the `gap²` weight lets a single
long-gap flight own up to 36 % of its own operator's coefficient
(gpt-5.6-sol round 1, and finding 12).

Under `gap2`, LIRF's `phi` is 0.343–0.422 in every single month of 2025. It
refines by operator, and the refinement is a **convention, not a fit**:

| estimator, and the row set it is fitted on | LIRF operators with ≥ 60 departures in each half of 2025 | `phi` range | correlation between the halves |
|---|---|---|---|
| `PhiModel(weight="rows")`, flights with `|gap| ≥ 300 s` | 74 | 0.17 – 1.00 | **0.878** |
| `PhiModel(weight="gap2")`, every flight with a finite gap | 88 | — | **0.908** |
| a hand screen on flights with `|gap| ≥ 600 s`, gap²-weighted (`work/q2f_whatisblock.py`) | 58 | 0.09 – 1.21 | **0.948** |

(The three differ in the row set, not only in the weight; the first version of
this table quoted the third row's numbers as the module's, which they are not —
gpt-5.6-sol round 2.)

Some handling agents' block time is recorded on the filed off-block, others' on
the actual one; the airport record and the Network Manager record are measuring
different events, per operator, consistently. The ordering also survives into
2026: over the twelve largest LIRF operators in July 2026, the rank correlation
of the median `AOBT − LOBT` with July 2025 is **0.954**
(`analysis/q1_july_regime.md` §7).

Why it matters more at Rome and more in summer: the model must predict `r`, and
`r = phi·gap` is large exactly when `gap` is large. `gap`'s median at LIRF is
120 s in January and **600 s in July**; 48.3 % of July rows exceed 600 s against
20.0 % in January and 10.2 % exceed 1800 s against 0.9 %. LIRF has both the
largest `phi` bar one and the largest summer swing in `gap`.

**As a predictor on its own** — `PhiModel` fitted on the fold's training months,
applied to the ranking-masked validation month, scored on the NM-joined
departures that are its domain:

| fold | LIRF: take-off − AOBT | LIRF: **phi formula** | LIRF: c6-noac | all ten: take-off − AOBT | all ten: phi formula | all ten: c6-noac |
|---|---|---|---|---|---|---|
| july | 770.6 | **553.1** | 635.9 | 465.8 | 376.3 | 287.8 |
| december | 452.8 | 374.1 | **309.9** | 374.6 | 312.9 | 208.2 |
| january | 434.4 | 353.2 | **294.7** | 377.0 | 311.5 | 212.3 |

On the july fold a formula with one coefficient per airport plus 156 LIRF
operator coefficients **beats a fitted LightGBM at LIRF by 83 s**. On december
and january the fitted model wins by 65 and 59 s. That contrast localises the
problem precisely: **the july fold's model is failing at LIRF on the phi
mechanism specifically**, not on taxi geometry and not on noise. The july fold
trains on January–June, whose `gap` distribution is the winter one, and the
pooled global fit is dominated by nine airports whose `phi` is 0.15–0.25.

(These are gap²-weighted `PhiModel` fits, the estimator this experiment started
with; the shipped default is now the row-weighted one of finding 12. The
predictor comparison was not re-run under the new default, so treat the exact
numbers as the gap² estimator's and the ordering as the finding.)

Spliced into the baseline for LIRF's NM-joined rows only:

| fold | fold RMSE before → after | LIRF before → after |
|---|---|---|
| july | 343.59 → **332.23** | 864.62 → **807.04** |
| december | 222.43 → 229.46 | 371.10 → 425.81 |
| january | 349.03 → 352.99 | 387.07 → 433.03 |

So the formula is **not** a replacement for the model. It is a feature, and
finding 5 measures it as one.

Module: `prc/phi_features.py` (`PhiModel.fit(train_frames).transform(month_frame)`
returns `phi` per level, `phi_support` counts, `block_hat = AOBT + phi·gap` and
`taxi_hat = take-off − block_hat`). Mutation-tested: perturbing departure
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` changes no column that `transform`
produces (`work/test_no_leak.py`).

### 2. Rome's 751 s July dispersion is one row, and `docs/shift-2026.md` needs the correction

`docs/shift-2026.md` records std of take-off − AOBT at LIRF = 751 s in July 2025
and 737 s in July 2026 against ~200 s in every other month, and reads it as
"Rome in July is its own regime in both years". Trimming the rows where that
difference exceeds an hour:

| set | n (NM rows) | std, all rows | rows with abs > 3600 s | std after dropping them |
|---|---|---|---|---|
| 2025-07 | 14,874 | **750.7** | 2 | **240.9** |
| 2026-07 | 15,562 | **737.1** | 3 | **248.2** |
| 2025-01 | 11,257 | 208.1 | 0 | 208.1 |
| 2026-01 | 10,954 | 218.5 | 2 | 196.1 |

Trimmed, the twelve 2025 months run 179 (March) to 241 (July): July is the top of
an ordinary seasonal range, 15–20 % above winter, not 3.6× above it. The single
row carrying the 2025 statistic is `AEZ146R` (LIRF, 2025-07-15, stand 234): every
time column shifted a day, `take-off − AOBT = 87,181 s`, and `BLOCK − AOBT =
+179 s`, i.e. an entirely ordinary block-to-off-block relation. It is one of the
fourteen LIRF members of the date-rollover class in exp-002 finding 13. July
2026's analogue is visible in the ranking file without labels: `ITY340`,
2026-07-17, stand 204, `take-off − AOBT = 86,340 s`.

**Consequence for the desk: any LIRF number quoted as a std or an RMSE of
take-off − AOBT is a one-row statistic and must be re-read trimmed.** The `july`
fold remains the right place to test LIRF machinery — for the reason in finding
1, not the one in `shift-2026.md`.

Full split by hour, stand group, runway, operator, market segment, NM status and
day — for July 2025 **and** July 2026 — is in `analysis/q1_july_regime.md`
(§7 carries the 2026 side: NM status, the twelve largest operators, and every
calendar day). Nothing localises the July elevation to a
subpopulation: every hour, every apron, all three runways and both large market
segments carry it, in proportion to their `gap`. The only day structure is two
disruption days on top of a broad month (2025-07-13, median `gap` 1,560 s and
42.5 % of rows over 1,800 s; 2025-07-28, 900 s and 23.7 %).

### 3. At LIRF, and only at LIRF, a missing NM join and a missing aircraft type are the same event

| airport | departures (July 2025) | no NM join | noac (null aircraft type) |
|---|---|---|---|
| **LIRF** | 15,211 | **337** | **337** |
| EHAM | 22,489 | 515 | 0 |
| LFPG | 21,712 | 664 | 1 |
| EDDF | 21,334 | 435 | 0 |
| EGLL | 20,790 | 338 | 0 |
| (the other five) | — | 172–430 | 0–1 |

Across all twelve months LIRF produces exactly one no-NM row that is not noac.
So the LIRF "no-NM" population is not "Network Manager failed to match a flight";
it is **Rome's own movement feed dropping the whole record**. On such a row only
`FLIGHT_mvt` (95.6 %), `FLIGHT_RULE_mvt`, `ADES_mvt`, `SCHED_TIME_UTC_mvt`,
`RUNWAY_mvt` and `STAND_mvt` survive — `FLIGHT_ID_mvt`, `CALLSIGN_flt`,
`AIRCRAFT_OPERATOR_flt` and `MARKET_SEGMENT_flt` are all null. That is why the
placeholder block time (block = schedule) appears at Rome and nowhere else, and
it is what finding 6's feature block is built for.

### 4. Where the baseline's LIRF error actually is: three roughly equal thirds on july

retro-v1 `c6-noac`, LIRF rows only, populations defined from inputs alone:

| fold | LIRF RMSE | noac (n, RMSE, share of LIRF SSE) | NM & y>3600 | NM core y≤3600 |
|---|---|---|---|---|
| july | 864.62 | 337, 3,987, **47.1 %** | 271, 3,312, 26.1 % | 14,603, **456.5**, 26.8 % |
| december | 371.10 | 88, 2,411, 30.8 % | 48, 2,418, 16.9 % | 11,932, 270.0, 52.3 % |
| january | 387.07 | 60, 3,459, 42.3 % | 41, 2,101, 10.7 % | 11,216, 266.5, 47.0 % |

The july **core** is 456 s against 270 and 267 at the same airport in the other
two folds. That gap is finding 1's mechanism, and it is the largest single block
of recoverable error at Rome.

### 5. "Block on LOBT or block on AOBT" is the wrong question: the block time is between them, and the continuous fraction is what pays

exp-002 finding 4 reads the NM tail as a binary — the block time sits *on* LOBT
while AOBT is an hour later. That is true of the extreme tail and false of the
population. Define `phi = (BLOCK − AOBT)/(LOBT − AOBT)` on the 55,465 LIRF
departures with `AOBT − LOBT ≥ 600 s`: only **10.4 %** sit within 5 % of AOBT and
**9.6 %** within 5 % of LOBT; **59.5 %** are strictly between the two and 20.5 %
are outside the interval altogether. The split
is the same in every month of 2025 (mean `phi` 0.36–0.44, `P(mid)` 57–61 %).

A LIRF-only regime classifier was built anyway, on the rows where the choice
could matter (`AOBT − LOBT ≥ 1800 s`), trained on the fold's training months:

| fold | rows | base rate | AUC | Brier |
|---|---|---|---|---|
| july | 1,618 | 0.317 | **0.762** | 0.204 |
| december | 164 | 0.177 | 0.862 | 0.091 |
| january | 117 | 0.265 | 0.836 | 0.146 |

The choice *is* partly predictable. It does not help to treat it as a choice:

| predictor on those rows | july | december | january |
|---|---|---|---|
| take-off − AOBT | 1,704 | 1,171 | 1,483 |
| take-off − LOBT | 1,940 | 2,038 | 1,972 |
| hard switch on the classifier | 1,329 | 837 | 1,102 |
| expectation `q·LOBT + (1−q)·AOBT` | 1,198 | 721 | 969 |
| **continuous `phi` regressor** | **1,012** | 642 | 882 |
| retro-v1 c6-noac | 1,088 | **628** | **813** |
| *oracle binary choice* | *682* | *431* | *531* |

The hard switch is 11 % worse than the expectation and 31 % worse than the
continuous regressor on july, which is exp-002 finding 10's rule holding again:
under RMSE, commit to a mixture's expectation, never to its mode. The new part
is that here even the expectation of a two-point mixture is the wrong object —
the truth is a continuum, and estimating the fraction directly wins.

**Who the high-`phi` rows are** (LIRF, `AOBT − LOBT ≥ 1800 s`, 6,614 rows in
2025, `P(block nearer LOBT) = 0.259`): it is an **operator** effect, modulated by
apron and hour, and not a day pattern.

| cut | range of `P(block nearer LOBT)` |
|---|---|
| operator (the twelve highest-rate operators with ≥ 25 such rows) | 0.29 → **1.00** |
| operator (the twelve largest by volume) | 0.06 → 0.65 |
| market segment | Lowcost 0.19, Mainline 0.30, Regional 0.33, **Non-Scheduled 0.74** |
| apron | 5xx 0.12, 3xx 0.20, 4xx 0.23, 2xx 0.29, 8xx 0.30, 7xx 0.35, 6xx 0.37, **9xx 0.44** |
| runway | 25 0.23, 16R 0.41, 34L 0.42 |
| hour (UTC) | 06:00 0.12 → 11:00 0.36 → 22:00 0.38 |
| day of week | 0.21 → 0.30 (no structure) |
| calendar day (≥ 5 such rows, 270 days) | 0.00 → 0.80, std 0.14 |

Per-operator rates are stable month after month (one operator 0.50–1.00 in every
month it flies, another 0.06–0.15 in every month), which is what makes finding 1
work: it is a standing convention of the handling agent, not a state of the day.

### 6. The noac head: Rome-specific signals cut it from 4,530 to 4,028 s

All 1,487 LIRF noac rows in 2025, leave-one-month-out so every row is predicted
by a model that never saw its month (4.4× the rows of the july fold alone), an
offset regression anchored on `take-off − SCHED`.

**This is a feature-set screen, not a fold-contract measurement**, and it may not
be promoted on its own: a July row here is predicted by a model that also trained
on August–December, which `prc/data.py`'s `july` fold forbids (gpt-5.6-sol round
1, blocking finding 5). The screen exists because 530 July training rows cannot
separate feature sets and 1,360 LOMO rows can. The **fold-respecting** numbers for
the same head, from the `noac` column of the global runs, are in `climb.log`:
july 3,562–3,940, december 2,500–2,553, january 3,479–3,906, against `c6-noac`'s
3,993 / 2,409 / 3,698 — i.e. the head is worth 50–430 s of noac RMSE on july and
january under the fold contract and is *behind* `c6-noac` on december.

| feature set | LOMO RMSE | july | december | january | worst month | contribution to a 344,841-row RMSE, over the 383 LIRF noac rows in the ranking file |
|---|---|---|---|---|---|---|
| constant training mean | 11,218 | 11,996 | 11,622 | 14,707 | 17,398 | 376 s |
| `take-off − SCHED`, no model | 5,498 | 6,089 | 2,786 | 5,704 | 7,278 | 184 s |
| exp-002 / retro-v1 head's feature set | 4,530 | 4,795 | 3,067 | 3,545 | 7,221 | 152 s |
| + `prc.lirf_features` signals | 4,234 | 4,140 | 2,941 | 3,554 | 6,905 | 142 s |
| everything in the frame | 4,129 | 3,386 | 2,825 | 3,289 | 6,875 | 138 s |

| **tuned, 5-seed** (40 rounds, 63 leaves, min_data 15, lr 0.05) | **4,028** | 3,514 | 2,548 | 3,541 | 6,733 | **134 s** |

For scale, exp-002's `NoacPredictor` scores 4,290 on the same LOMO design. The
tuned line is selected from a 36-point grid scored on the same predictions, so
that cell carries selection bias and is not a promotable number; the grid's whole
range is 4,017–4,265, which is *narrower than the gap to the exp-002/retro-v1
feature set* (4,530). **The feature-set difference is the result; the tuning is
not** — and re-checked under the fold contract the tuning is a *loss*. Same
harness, same features, same seed, only the head's hyper-parameters:

| fold | LOMO-tuned head (40 rounds, min_data 15, 5 seeds) | original head (60 rounds, min_data 5, 1 seed) |
|---|---|---|
| july | fold 346.55, noac **3,861** | fold **341.49**, noac **3,598** |
| december | fold **221.06**, noac **2,509** | fold 222.38, noac 2,726 |
| january | fold 345.99, noac 3,527 | fold 346.17, noac **3,278** |

The tuned head wins december and loses july and january on the noac rows it was
tuned for, which is what a hyper-parameter selected on a design that lets July
see August–December should be expected to do. **Adopt the feature set, not the
tuning**: the shipped configuration uses 60 rounds, `min_data_in_leaf` 5 and one
seed, with the `prc.lirf_features` columns in the head.

What carries it, by gain in the july LOMO fit, after the `take-off − SCHED`
anchor: `destination`, **`flight_prefix`**, **`stand_in_turnaround`**,
`arr_taxiin_mean_10m_before`, `airport_stand`, **`stand_next_dep_gap`**,
`stand_prev_arr_block_gap`, **`noac_prev_gap`**. The bolded four are new here and
three of them answer the assignment's question directly:

- **`flight_prefix`** — the operator is null on a noac row, but `FLIGHT_mvt`
  survives on 95.6 % of them, and its two-to-three-letter prefix is the airline.
  It is the third most important feature in the head.
- **`stand_in_turnaround`, `stand_in_block_gap`, `stand_next_block_gap`** — the
  arrival that last parked at this stand (it brought the airframe; its on-block
  is the earliest the departure could have been there) and the next arrival to
  park (the departure must have left before that on-block). The stand answers
  where the flight plan is silent.
- **`noac_prev_gap` / `day_noac_n`** — how isolated this stripped record is. An
  isolated one is a placeholder (`P(block = schedule)` 0.69 on days with ≤ 2 noac
  rows) and a bulk one is a feed outage with real block times (0.23 on days with
  16–32), confirming exp-002 finding 2 and giving the head the continuous version.

A noac day is also a degraded day for the rows that *did* join: LIRF's median
`AOBT − LOBT` on the other departures is 131 s on days with no noac row and 787 s
on days with 16 or more, and the median `r` moves from −55 to −294 s. That is
`day_noac_frac` and `al_day_med` earning their place in the same head.

### 7. Question 4 — a LIRF-only model loses at every seed and on every fold

Two designs, same features, same folds, same harness (`run_lirf.py`). Every
number is the whole-fold RMSE with the LIRF rows replaced (for the LIRF-only
lines) or the whole model re-fitted (for the global lines). `k` is the number of
LIRF rows removed *by the baseline's squared error*, so all models are scored on
the same population; exp-002 finding 13 says the largest LIRF errors are
date-rollover rows that are a lottery, and a mechanism has to survive their
removal.

**Whole-fold RMSE, single seed, unguarded** (`work/eval_global.csv`;
`work/eval.csv` for the LIRF-only splices). `g0` is my own global control — the
same harness and features as the global lines below. **These numbers are one seed
each and finding 8 shows the july column carries a 17.6 s run-to-run band; the
table that decides anything is the guarded, seed-averaged one at the end of
finding 8.** It is kept because the LIRF-only-versus-global comparison is
unambiguous at every seed.

| model | july k=0 | july k=2 | july k=20 | dec k=0 | dec k=20 | jan k=0 | jan k=20 |
|---|---|---|---|---|---|---|---|
| retro-v1 c6-noac (baseline) | 343.59 | 325.18 | 306.41 | 222.43 | 216.23 | 349.03 | 342.57 |
| g0 global control | 349.76 | 318.33 | 303.07 | 221.55 | 214.17 | 350.79 | 345.14 |
| **LIRF-only**, same features (b1) | 324.06 | 323.63 | 308.89 | 225.51 | 219.75 | 349.19 | 342.81 |
| **LIRF-only** + `lirf_signals` (c1) | 319.58 | 318.26 | 304.82 | 228.30 | 222.20 | 350.76 | 344.88 |
| **LIRF-only** + phi + signals (c2) | 328.68 | 328.61 | 311.73 | 223.91 | – | 355.33 | – |
| global + `lirf_signals` (g1) | 341.00 | 311.10 | **297.41** | 221.03 | 215.03 | 346.93 | 340.93 |
| global + phi (g2) | 341.46 | 324.71 | 308.28 | 220.45 | 214.65 | 346.76 | 339.82 |
| global + phi + signals (g3) | 330.70 | 319.86 | 302.97 | 218.43 | 212.58 | 345.62 | 338.69 |

And at LIRF alone:

| model | july k=0 | july k=20 | dec k=0 | dec k=20 | jan k=0 | jan k=20 |
|---|---|---|---|---|---|---|
| retro-v1 c6-noac | 864.62 | 667.12 | 371.10 | 316.96 | 387.07 | 298.49 |
| LIRF-only (b1) | 764.23 | 681.31 | 395.79 | 348.64 | 389.00 | 302.14 |
| LIRF-only + signals (c1) | 740.23 | 657.90 | 417.22 | 369.42 | 407.67 | 332.69 |
| global + signals (g1) | 862.39 | **628.67** | 373.20 | 321.71 | 387.85 | 307.14 |
| global + phi + signals (g3) | 803.95 | 653.06 | 375.53 | 326.35 | 386.83 | **291.83** |

**The answer to question 4 is that the LIRF-only model loses across the three-fold
band, and most of `b1`'s untrimmed july gain is rollover-driven.** `b1` looks like
a 20 s gain on the july fold (343.59 → 324.06); remove the two worst LIRF rows and
the gain is 1.6 s; remove twenty and it is a 2.5 s *loss*. `c1` keeps more of its
july gain than `b1` does — 6.9 s at k=2 and 1.6 s at k=20 — so the july column is
not purely a lottery for the signal-carrying variant; what closes the question is
december and january, where every LIRF-only line loses at every k. The row is `AEZ146R` from finding 2: the baseline predicts
44,397 s against a true 87,002 s and the LIRF-only model 93,863 s, and a
15,000-row-per-month model happens to extrapolate that one row less badly. Every
LIRF-only line is worse than the baseline on december (+1.5 to +5.9 at k=0) and
on january, because 90–160 k training rows cannot buy what 1.1–2.1 M can: the
LIRF-only december core is 309–332 against the global 205–312. This holds at
every seed and with or without the guard, so it is the one conclusion in this
section that needs no qualification.

The global-model side of question 4 is in finding 8, measured with the guard and
matched seeds; the single-seed unguarded lines above overstate it by an order of
magnitude and are kept only because the LIRF-only comparison does not depend on
them.

### 8. A guard in residual space, the run-to-run band it exposes, and what the features are actually worth

exp-002 and retro-v1 both measured a **ceiling on the taxi time** and both found
it always loses, because the true july maximum is 88,132 s and a model's large
predictions are earned. A clip on `r = BLOCK − AOBT` is a different object,
because it acts on the residual: the prediction can still be 87,000 s whenever the
known `take-off − AOBT` is.

**The rule is parameter-free and fold-safe:** clip the implied `r` to the range
observed over the *fold's own training departures*. An earlier version picked
[−10,800, +3,600] after evaluating five candidate ranges on the folds themselves,
with the validation labels in view — gpt-5.6-sol round 1 blocking finding 3, and
correctly so. Every number below uses the training-range rule instead, which for
all three folds is **[−20,657, +9,069] s** and involves no choice at all.

| model | july | december | january |
|---|---|---|---|
| retro-v1 `c6-noac` | 343.59 → **327.95** | 222.43 → 222.43 | 349.03 → 349.03 |
| `g0` control, 2 seeds averaged | 342.77 → **315.49** | 221.26 → 220.55 | 346.57 → 346.57 |

It costs nothing on either winter fold (no prediction leaves the range) and it is
worth 15.6 s of july on the baseline and 27.3 s on my control. **On `c6-noac` most
of that is one date-rollover row; on the linear-tree models it is not.** The seed
repeats show why:

| fold | model | 2 seeds, unguarded (mean, spread) | guarded (mean, spread) |
|---|---|---|---|
| july | `g0` v3 features | 346.53, **6.47** | 319.45, **0.94** |
| july | `hCb` phi + signals | 330.91, **8.44** | 314.65, 6.48 |
| july | `hV` exp-001's phi construction | 456.46, **206.16** | 348.19, 22.26 |
| december | `hFb` | 220.34, 1.85 | 220.34, 1.85 |
| january | `hFb` | 442.63, **192.76** | 348.20, 3.89 |

An unguarded linear-tree model on this data is a lottery whose ticket is the
seed: `hV` on july spans 206 s across two seeds and `hFb` on january 193 s,
because on a handful of rows the leaves emit an implied `r` of tens of thousands
of seconds. **The guard is the difference between an estimator and a coin, and it
should be applied before any feature comparison is believed.** It is both a
lottery hedge (on `c6-noac`) and a variance reducer (on the linear-tree lines);
which one dominates depends on how far the model extrapolates.

**What the features are worth, guarded, two matched seeds each, guard-then-average**
(`work/q10_seeds.csv`; the estimator was fixed before the comparison — the other
order changes the july sign, so it is a real degree of freedom):

| model | july | december | january |
|---|---|---|---|
| retro-v1 `c6-noac`, guarded (1 seed) | 327.95 | 222.43 | 349.03 |
| `g0` exp-001 v3 features | 315.49 | 220.55 | 346.57 |
| `h0b` + `lirf_signals` | 313.42 | 221.01 | 346.81 |
| `hFb` + phi, one full-train fit | **309.53** | **218.67** | 346.41 |
| `hCb` + phi, cross-fitted | 309.86 | 221.31 | 347.34 |
| `hV` + phi, exp-001's construction | 336.31 | 219.86 | **345.02** |

No configuration wins all three folds, and the spread between the sensible ones
is 1–4 s — of the order of the guarded seed spread. **The fold-level number
cannot separate them.** The paired bootstrap on the input-defined core can, and
it is clustered by calendar day because departures on one day share the whole
congestion state (`work/q11_guarded_control.py`, 1,000 resamples over 31 days):

| comparison | july, whole core | july, LIRF core | december, LIRF core | january, LIRF core |
|---|---|---|---|---|
| phi **cross-fitted** vs the same model without phi | **−6.35 [−7.29, −5.32]** | **−8.05 [−12.94, −2.20]** | −5.89 [−27.15, +6.93] | **−4.34 [−9.52, −0.78]** |
| phi **full-fit** vs the same | **−7.43 [−9.71, −5.01]** | +6.82 [−6.54, +21.64] | −10.24 [−33.33, +2.20] | +11.25 [−8.23, +42.59] |
| `lirf_signals` vs the v3 feature set | **+3.93 [+1.08, +6.33]** | −16.36 [−32.54, +0.70] | +2.40 [−19.82, +27.21] | +2.52 [−1.23, +7.38] |

Two conclusions, and they are narrower than the ones this file carried before the
review:

1. **The cross-fitted phi is the mechanism that survives.** It improves the LIRF
   core significantly on july (−8.1 s) and january (−4.3 s) and the whole core
   significantly on july (−6.4 s), and it never significantly hurts anything. The
   full-fit variant improves the whole core by more but its LIRF gain is not
   significant and flips sign on january — the signature of an own-row advantage
   that is not a mechanism. Ship the cross-fitted construction.
2. **The `lirf_signals` block is not a global win, and the obvious fix is worse
   than the problem.** It helps the LIRF core (−16 s on july, not significant) and
   *costs* the whole core on july (+3.9 s, significant), which is a negative
   against my own earlier recommendation, made on single-seed unguarded fold
   numbers. The natural repair — null the 39 columns everywhere but LIRF, so only
   a LIRF branch can use them — was built (`--gate-signals LIRF`) and **loses
   badly**: july 351.21 against 326.70 and december 227.79 against 223.43 for the
   same model with the columns global (seed 7, unguarded). Diagnosis: masking does
   not remove a column, it fills it with NaN on 92 % of rows, and LightGBM still
   spends splits on it while learning a worse LIRF branch than it had before. The
   repair that would work is a per-airport branch or a separate LIRF head, which
   finding 7 says is the wrong shape for this data anyway. **So: keep the block
   global, record that it costs 3.9 s of july whole core against 16 s of LIRF
   core, and treat "make it LIRF-conditional properly" as the open item.**

### 9. The retired LIRF stands 501–507

They carry **10,015 departures in 2025 (6.23 % of LIRF)** and stop after October
2025 — 49 rows in November, none in December, none in the ranking months. The
same operators fly from the 3xx apron in 2026. Two consequences, one of them
already in the folds:

- The `december` fold is the only one whose validation month is free of them, so
  it is the naturally shift-safe fold; `july` has 1,116 such rows in validation
  and `january` 775, and those rows **cannot** recur in 2026.
- The retired apron has its own convention: `phi` = 0.29 against 0.39 at the rest
  of LIRF, and it is *easier* than the rest — the baseline scores 681 s on the
  july 5xx rows against 878 s on the other LIRF rows. Restricting each fold to
  the 2026-like population (dropping 501–507 from the validation month) moves the
  july fold from 343.59 to 340.62 and january from 349.03 to 349.54.

**Does dropping them from training help?** `g4` is `g3` with the 10,015 LIRF
501–507 training rows removed, scored on the 2026-like population (validation
rows on those stands removed too), `work/q9_retired.csv`:

| fold | whole fold, g3 → g4 | LIRF only, g3 → g4 |
|---|---|---|
| july | 328.57 → 331.80 (**+3.2**) | 818.54 → 857.50 (**+39.0**) |
| december | 218.43 → 220.48 (**+2.1**) | 375.53 → 373.36 (−2.2) |
| january | 346.13 → 346.24 (+0.1) | 396.25 → 394.21 (−2.0) |

**No: keep them.** The answer is a clear no on july and a wash on the two winter
folds, where a ~2 s LIRF gain is swamped by a 2 s fold-level loss that is inside
the run-to-run band (see below). Diagnosis of why removal does not help even
though the stands cannot recur: the rows are 6 % of LIRF's training data and what
they teach — the airport's `phi`, its congestion response, its stand-to-runway
geometry from an apron adjacent to the 3xx pier the traffic moved to — transfers
to the stands that replaced them. Only the *stand identity* is dead, and the
fallback chain already handles an unseen stand. Down-weighting rather than
dropping was not run; on this evidence the expected gain is under a second and it
was not worth a fold triple.

### 10. Negative results, with the diagnosis and the repair that was tried

**A LIRF-only model on the same features (a0) lost 187 s of december.** 558.1 at
LIRF against the baseline's 371.1, whole fold 249.26 against 222.43. Diagnosis:
`month_num` and `day_of_month` are in the exp-001 feature frame, the december
fold validates on month 12 having trained on months 1–11, and **linear leaves
extrapolate a linear coefficient on an out-of-range month index**. Repair, and it
worked: drop both columns (b1), december 558.1 → 395.8. The global model does not
show this because 2 M rows put enough other splits above the month index; a
150 k-row model puts the month index near the root. Recorded because the same
trap is waiting for any per-airport model anyone builds next, and because
`doy_sin`/`doy_cos` (exp-001 v3) are the right way to spend a seasonality
feature.

**`prc.lirf_features` signals inside the LIRF-only model made december and
january worse** (c1: 417.2 and 407.7 against b1's 395.8 and 389.0) while
improving july (740.2 against 764.2). Diagnosis: 39 extra columns on 90–160 k
rows is a variance trade the july fold pays for and the other two do not.
Repair, partly successful: the same columns in the global model help the LIRF core
but *cost* the whole july core (+3.9 s, significant — finding 8), so the repair is
the airport gate rather than the global application. An earlier version of this
paragraph said the global version "improves all three folds", which was read off
single-seed unguarded fold numbers and does not survive the guard and the matched
seeds.

**The phi features inside the LIRF-only model lost everywhere** (c2: 328.7 /
223.9 / 355.3 against c1's 319.6 / 228.3 / 350.8 and the baseline's 343.6 /
222.4 / 349.0; january 458.4 at LIRF). Diagnosis: `phi_op` is an encoding fitted
on the training months, and retro-v1's c9 finding applies — a model that can put
a coefficient on a noisy estimate of the answer calibrates that coefficient to
the noise level it saw, and a 150 k-row model with linear leaves does exactly
that. In the global model the same columns are one signal among 150 and the
model leans on them less. Revival condition: build the serve-side encoding from a
subsample matched to the train-side support, which is retro-v1's own untried
repair for c9.

**The regime classifier, used as a switch, loses** (finding 5): 1,329 against
1,088 for the current model on the july rows it was built for. Diagnosis and
repair are in finding 5 — the population is a continuum, so estimate the fraction.

**A ceiling on the taxi time still loses; the guard has to be on the residual**
(finding 8). This is not a new negative, it is exp-002's and retro-v1's, and
finding 8 is the repair that passes.

**Arrivals carry no independent read of the block clock.** Checked because a live
per-hour estimate of the airport-versus-NM offset would be the ideal feature:
`ARVT_3_flt` is Network Manager's *landing* time, not its in-block time — its
median distance to the airport's landing time is −2 s and to the on-block time is
+559 s, i.e. the taxi-in. So the arrival side gives surface state (already in
exp-001 v3 as `arr_taxiin_*`) and nothing about the off-block convention. The
phi estimate therefore has to come from labelled months; there is no
contemporaneous observable substitute in these files.

### 11. The inbound airframe: the stand identifies it, the flight number does not

The assignment asks whether the arrival that brought the airframe can be found by
"same callsign family or same stand shortly before". On a noac row
`CALLSIGN_flt` is null but `FLIGHT_mvt` survives, so both were built and
compared on all 160,704 LIRF departures of 2025:

| inbound sought by | found | take-off − inbound on-block (p25 / median / p75) | "block-off ≥ inbound on-block" holds |
|---|---|---|---|
| **same stand** | 100.0 % | 4,262 / 5,399 / 9,593 s | **95.7 %** (all) / 64.6 % (noac) |
| same flight-number prefix | 100.0 % | 841 / 3,001 / 5,164 s | 70.9 % (all) / 48.9 % (noac) |

(An earlier version of this table gave 4,073 / 9,002 / 32,163 and 92.9 / 77.5 %.
It coded the arrival and departure stand columns in two separate factorisations,
so the two sides did not mean the same stand — caught by gpt-5.6-sol, fixed in
`work/q3d_inbound.py`. The ordering of the two keys is unchanged.)

The stand is the right key. The prefix match is too coarse — a base carrier lands
every few minutes, so the "inbound" it finds is usually a different aircraft, and
the physical bound it implies is violated on 29 % of ordinary departures. The
prefix is still worth carrying, but as *identity* (`flight_prefix`, the operator
substitute of finding 6), not as a turnaround link.

The stand turnaround `SCHED − inbound on-block` is a **magnitude** signal on the
noac rows rather than a class detector: `P(placeholder)` moves only from 0.44 to
0.52 across its range against a 0.48 base rate, but the mean target moves from
3,771 s to **12,908 s** in the bin where the schedule precedes the inbound
parking by more than an hour. That is exactly the shape an L2 offset regression
can use and a classifier cannot, and it is why `stand_in_turnaround` is the
fourth-ranked feature in the tuned noac head.

---

## The table COMMON.md requires

Counts are the same for every model on a fold, so they are given once: july
190,713 rows / 656 above 3,600 s / 3,938 no-NM / 339 noac; december 165,677 /
261 / 1,649 / 89; january 153,706 / 325 / 1,435 / 63. These are **single-seed,
unguarded** numbers, because the required split is defined per model fit; the
guarded, seed-averaged fold numbers that decide anything are in finding 8.

The "no-NM" column below is the harness's `nm_missing`, which is null
`FLIGHT_ID_mvt`, not null `LOBT_flt`; the two differ on a handful of rows. On the
`hCb` line the true no-NM RMSEs are 1,236.5 / 845.6 / 2,857.1 (gpt-5.6-sol round 2).

| fold | model (seed 7) | RMSE | RMSE ≤3600 | RMSE >3600 | RMSE no-NM* | RMSE noac | LIRF |
|---|---|---|---|---|---|---|---|
| july | c6-noac baseline | 343.59 | 290.6 | 3,140.4 | 1,337.5 | 3,992.5 | 864.62 |
| july | g0 v3 features | 349.76 | 278.1 | 3,626.9 | 1,276.4 | 3,764.3 | 912.63 |
| july | h0b + lirf_signals | 341.49 | 269.9 | 3577.7 | 1205.9 | 3588.7 | 864.98 |
| july | **hCb + cross-fitted phi** | 326.70 | 271.3 | 3115.6 | 1237.0 | 3609.8 | 796.59 |
| december | c6-noac baseline | 222.43 | 205.2 | 2,173.7 | 812.1 | 2,408.7 | 371.10 |
| december | g0 v3 features | 221.55 | 203.1 | 2,241.7 | 866.3 | 2,773.7 | 388.70 |
| december | h0b + lirf_signals | 222.38 | 206.0 | 2121.2 | 817.8 | 2714.3 | 384.36 |
| december | **hCb + cross-fitted phi** | 223.43 | 205.1 | 2244.6 | 846.6 | 2760.6 | 393.41 |
| january | c6-noac baseline | 349.03 | 215.3 | 5,978.9 | 2,875.5 | 3,698.3 | 387.07 |
| january | g0 v3 features | 350.79 | 216.7 | 6,002.3 | 2,869.3 | 3,796.5 | 377.63 |
| january | h0b + lirf_signals | 346.17 | 210.3 | 5983.8 | 2861.3 | 3671.7 | 382.54 |
| january | **hCb + cross-fitted phi** | 346.74 | 211.6 | 5977.5 | 2859.9 | 3650.2 | 377.38 |

The input-defined split retro-v1 argues for (`RMSE ≤3600` conditions on the
answer) is in the LIRF tables of findings 4 and 7 and in `work/eval_global.csv`.

## Reviewer

`reviewer: gpt-5.6-sol-xhigh-2026-09-05`, two rounds, both **changes required**,
both answered. `review/sol-round1.md` and `review/sol-round2.md`.

Round 1 raised five blocking findings and every one was right: a construction
comparison that changed the feature schema as well as the construction; an
unsound "not a leak" argument (the gap² weight gives one flight up to 36 % of its
own group's coefficient); a residual guard whose bounds were chosen with the
validation labels in view; a bootstrap that compared three candidate seeds with
two control seeds and treated rows as independent; and a noac screen that trains
July on August–December. The experiment was re-run: the estimator is now
row-count-weighted, the guard's parameter was removed rather than defended, the
constructions share one column schema, the bootstrap is matched and clustered by
day, and the noac screen is labelled a screen with the fold-respecting comparison
reported beside it.

Round 2 raised four more, all right as well: the `cv` versus `cvfull` headline
still mixed two noac heads (corrected to 21.6 s from 26.5); the analytic
leave-one-row-out still let a row's label reach its own feature through the global
prior (fixed, and now asserted against exact delete-one refits); the bootstrap
never paired `cv` against `full` directly, so shipping `cv` is a safety argument
and not a performance result (restated); and the deliverable embodied the feature
block the report says not to ship globally (measured — the gate loses, and the
recommendation is now to keep the block global with the trade recorded). Its
remaining record corrections — the no-NM mask, the `gap_closed` denominator, the
operator-table estimator, the ceiling blend, and a syntax error I introduced in
`work/q3b_noac_lomo.py` — are all applied above.

Both rounds verified the six deliverable artifacts by hash and RMSE.

## Deliverables and how to integrate them

**Modules, both importable and both mutation-tested for leakage**
(`work/test_no_leak.py`): perturbing every departure `BLOCK_TIME_UTC_mvt` by up
to ±50,000 s and randomising `TAXITIME_SEC_mvt` changes **no** column that
`lirf_signals` or `PhiModel.transform` produces, and each gives byte-identical
output on the ranking-masked frame and the full frame. The same test covers
exp-001's `build_features` at v3, which is why `work/cache_features.py` may build
the cache from the raw month frame: `build_features(masked)` and
`build_features(full)` agree on every column, and the mutation changes none of
them. Six `lirf_signals` columns *do* move when arrival on-block times are
blanked — `stand_in_block_gap`, `stand_in_taxiin`, `stand_in_turnaround`,
`stand_in_land_gap`, `stand_in_noac`, `stand_next_block_gap` — which is the legal
dependence, because the ranking file keeps arrival block times
(`docs/input-rule.md`).

- **`prc/phi_features.py`** — `PhiModel().fit(train_month_frames).transform(month_frame)`.
  Returns `phi` (finest level with support), `phi_airport`, `phi_airport_operator`,
  `phi_support*` counts, `block_hat = AOBT + phi·(LOBT − AOBT)` and
  `taxi_hat = take-off − block_hat`, plus `gap`. Fit on the fold's training months
  only. It reads departure block time, so it is a *training-months* object; the
  transform side never does.
- **`prc/lirf_features.py`** — `lirf_signals(month_frame)` returns 39 columns from
  one month frame: the block-clock anchors, the drift of `AOBT − LOBT` over the
  last 60/180 minutes and over the day, the pushback queue between the filed and
  actual off-block, the day's and hour's feed state, six stand-history columns,
  and the identity substitutes `flight_prefix`, `stand_group`, `stand_num`,
  `stand_retired`, `ades_is_adep`. `LIRF_CAT_FEATURES` lists the two categoricals.
  `lirf_block(X)` is the cheap subset for a caller that has exp-001's feature
  frame but not the raw month.
  Owner note: this module is new, so no owner's agreement was needed; it does not
  modify `prc/tail_features.py`, `prc/surface_features.py` or `prc/data.py`.
  Nine of its columns duplicate names in exp-001's v3 frame and the harness drops
  the duplicates on join (`run_lirf.py:load_months`).

**Integration for retro-v1, in the order the evidence supports it**

1. **The residual guard, first and on its own.** Clip the implied
   `r = take-off − AOBT − y_hat` to the range of `r` observed over the fold's
   training departures (for every fold here, [−20,657, +9,069] s). No parameter,
   no refit, six lines. Worth 15.6 s of july on `c6-noac` and 27.3 s on my
   control, nothing on either winter fold, and it takes an unguarded linear-tree
   model's seed spread from 6–206 s down to 1–7 s. **Nothing else in this list is
   measurable without it**, and it is the largest single number in the experiment.
2. **Seed averaging** — two seeds, each guarded, then averaged (predeclare that
   order; the other one changes the july sign). Worth 3.9 s of july on the
   control.
3. **`PhiModel`, cross-fitted** (`PhiCV(...).transform_train / transform_serve`
   with `mode="avg"`, the default `weight="rows"`). Significant on the
   day-clustered paired bootstrap at the LIRF core on july (−8.05 s) and january
   (−4.34 s) and at the whole core on july (−6.35 s). **Do not use the
   `cvfull` construction** — the serving side must see the mean of the K fits,
   not the all-months fit; that one difference is 26.5 s of july
   (finding 12). For the submission, fit the K models on the twelve 2025 months.
4. **The noac head's `prc.lirf_features` feature set** — but *not* the
   hyper-parameters I screened on the leave-one-month-out design, which cost
   5.1 s of july under the fold contract (finding 6). Keep 60 rounds,
   `min_data_in_leaf` 5, one seed.
5. **`lirf_signals` globally, with the trade recorded.** Applied to all ten
   airports it helps the LIRF core (−16 s on july, not significant) and costs the
   whole core on july (+3.9 s, significant); at the fold level it is a wash across
   the three folds. Masking the columns outside LIRF is **not** the fix — it costs
   24 s of july (finding 8). A per-airport branch is the open item; until someone
   builds one, ship the block global or not at all, and know which trade you are
   taking.

**Out-of-fold predictions**, `experiments/exp-006-lirf/oof/`. The deliverable is
the shipping configuration: `hCb` (v3 features + `lirf_signals` + cross-fitted
phi + the original noac head) at seeds 7 and 17, each guarded with the fold's
training range, then averaged.

| file | rows | RMSE |
|---|---|---|
| `lirf-july.parquet` | 15,211 | **715.83** |
| `lirf-december.parquet` | 12,068 | **389.99** |
| `lirf-january.parquet` | 11,317 | **382.74** |
| `hCbc-{fold}.parquet` (all ten airports) | 190,713 / 165,677 / 153,706 | 309.86 / 221.31 / 347.34 |
| every other line in `climb.log` | | `g*-`, `h*-`, `b1-`, `c1-`, `c2-{fold}.parquet` |

SHA-256 (first 16): `lirf-july` `2ea59d9ae736c9f2`, `lirf-december`
`027e0fd04ccb1772`, `lirf-january` `3ec0558789f52730`; `hCbc-july`
`f19a16cf09bfc696`, `hCbc-december` `00548aea24681847`, `hCbc-january`
`c4b568f16365586e`.

Feature caches under `artifacts/exp-006-lirf/` (git-ignored, rebuildable and
hashed by `work/cache_sha.py`, output kept in `work/cache_sha.txt`):

| cache | built by | files | MB | SHA-256 of the files concatenated in name order |
|---|---|---|---|---|
| `feat_v3` | `work/cache_features.py` | 14 | 219.9 | `b7f23c7c6ac414ac…20ef86` |
| `lirf_sig` | `work/cache_lirf_signals.py` | 14 | 122.8 | `59242dccc6971782…3fad48` |
| `phi_full` | `work/cache_phi.py --modes full` | 31 | 203.6 | `b7ad84a4ac79f627…d738f9` |
| `phi_cv` | `work/cache_phi.py --modes cv` | 31 | 203.5 | `5c4c5590c79c721d…4b5e72` |
| `phi_cvfull` | `work/cache_phi.py --modes cvfull` | 31 | 203.3 | `c9b9dc152cdd5d7c…3ccb66` |
| `phi_loo` | `work/cache_phi.py --modes loo` | 31 | 293.0 | `72c00c55105eaf5b…d6857b` |

### 12. How phi is fitted matters more than phi itself

Raised by retro-v1: on their deterministic ladder a guarded baseline scores july
321.12 and guarded phi + lirf 333.57, and a paired bootstrap on the core makes
phi + lirf **+4.10 s [+2.63, +5.41] worse on july** while significantly better on
december (−2.66) and january (−3.64). My first measurement had the opposite july
sign. Three candidate explanations were tested; the third is the answer.

**Their hypothesis — July's large `gap` interacting with a phi fitted on smaller
gaps — is measured out.** phi *does* depend on the gap (LIRF: 0.48 at
|gap| ≤ 300 s falling to 0.31 at 1,800–3,600 s, then 0.69 above an hour), but the
shape is the same in every season; summer and winter differ by about 0.03 at
every band. What moves the *marginal* phi from month to month (0.343 in September
to 0.422 in July) is the gap mix, because the least-squares weight is `gap²`.
Priced directly: predicting July's LIRF NM rows with the Jan–Jun phi (0.3788)
instead of July's own (0.4216) costs **2.0 s** — 596.3 against 594.3. Not a 12 s
effect.

**My first comparison was invalid.** It set `g3` (phi built by the superseded
`PhiEncoder`) against `g3lomo` (phi built by `PhiModel`), so it changed the
feature schema as well as the construction and the 22.4 s it reported is not
attributable to either — gpt-5.6-sol round 1, blocking finding 1. Every phi line
in this file is now built by `work/cache_phi.py`, which emits **one column schema
for all four constructions**, and `run_lirf.py` asserts on that schema.

**And the estimator had a leverage problem.** The least-squares coefficient
weights each flight by `gap²`, so on the july fold's training rows one flight
owns up to **36 %** of its own airport-operator coefficient and 163 flights
exceed 10 % — the argument that a "ratio over hundreds of rows" cannot leak was
wrong, and Sol was right to reject the december-is-easier evidence I offered for
it. `PhiModel` now defaults to `weight="rows"`: the mean per-flight ratio over
flights with `|gap| ≥ 300 s`, shrunk toward the parent by flight count, where one
flight's share is bounded by `1/(n + 150)`. The largest own-row shift in phi falls
from 0.221 to 0.024.

**The construction, measured properly.** Four constructions, same schema, same
features, same seeds, guarded and seed-averaged (`work/q10_seeds.csv`):

Guarded, two matched seeds, guard-then-average. The three lines below share the
**same** noac head (the tuned one), so only the construction differs; the `b`
variants elsewhere in this file change the head as well and must not be mixed in
here (gpt-5.6-sol round 2 caught exactly that error in the first version of this
table).

| construction | what a training row sees | what a serving row sees | july | december | january | unguarded july seed spread |
|---|---|---|---|---|---|---|
| `full` (`hF`) | one fit on all training months, its own row included | the same fit | 316.90 | **217.76** | **345.58** | 1.93 |
| `cv` (`hC`) | the fit that excluded its month | the mean of the K fits | **314.71** | 219.42 | 346.32 | 9.76 |
| `cvfull` (`hV`) | the fit that excluded its month | the all-months fit | 336.31 | 219.86 | 345.02 | **206.16** |
| `loo` | the same fit minus its own row, analytically | the same fit | *not run* | | | |

`cvfull` is **21.6 s worse than `cv` on july**, and its unguarded two-seed july
spread is **206 s** against 9.8 s. The two differ only in what the serving side
sees: one fit built from every training month against the mean of K fits each
built from K−1. A feature whose noise level differs between the training and the
serving side is what a linear leaf over-weights, and on a six-month fold the
difference is a sixth of the data. On december and january, where the difference
is a eleventh, the three constructions are within 3 s of each other.

**Which to ship, and on what grounds.** `full` has the better fold numbers on two
folds of three. A direct day-clustered paired bootstrap between `full` and `cv`
separates them on nothing: july whole core −0.64 [−2.32, +0.96], december +1.68
[−0.63, +4.51], january +1.10 [−2.43, +3.77], and the LIRF-core intervals all
straddle zero (gpt-5.6-sol round 2, who ran the pairing I had not). **So the case
for `cv` is not a performance result.** It is that `cv` is the only compared
construction with no path from a training row's own label to its own feature, and
finding 8 shows the price of getting that wrong on a six-month fold is 21.6 s in
the wrong direction. Ship `cv` as the safe construction, and record that `full`
is not measurably worse.

**`cvfull` was `experiments/exp-001-retro-v1/phi_step.py`'s construction when this
was written; retro-v1 has since moved to `cv` with mean-of-K serving**, which is
the change this finding asked for. Nothing further is needed there.

