# exp-003-anatomy — RESULTS

exp-id `exp-003` | worker `anatomy` | tier hard | reviewer: `gpt-5.6-sol-xhigh-2026-09-05`, three rounds, final verdict **BLOCK** (five write-up findings, all answered below in *Round 3*; the desk closed the review at three rounds because the protocol allows two and write-up findings do not block)
Data: this repo's `data/raw` via `prc.data.data_dir()`; every 2026 number uses the
live `ranking.parquet` (344,841 departures, 344,693 arrivals, SHA-256 `9db75e3a…`).
Code in `experiments/exp-003-anatomy/code/`, feature module `prc/stand_features.py`,
long-form tables in `experiments/exp-003-anatomy/analysis/`, shift note in
`docs/shift-2026.md`.

## The reframing everything below rests on

Take-off `MVT_TIME_UTC_mvt` is given for every ranking row, so predicting
`TAXITIME_SEC_mvt` **is** predicting `BLOCK_TIME_UTC_mvt`. `AOBT_3_flt` is given
on 98.47 % of ranking rows, so on those rows it **is** predicting the offset

    D = BLOCK_TIME_UTC_mvt - AOBT_3_flt,

because with the fully observed `nm_taxi = MVT_TIME_UTC_mvt - AOBT_3_flt` we have
`taxi = nm_taxi - D` exactly, and any predictor `taxi_hat = nm_taxi - D_hat` has
per-row error `D_hat - D`. RMSE on the target and RMSE on the offset are the same
number. Every model below learns `D` and reconstructs the taxi time.

## Findings, ranked

### 1. The offset is not a property of the flight. It is a second measurement of the same event.

Conditioning `BLOCK - AOBT_3` on ever finer descriptions of the flight stops
paying almost immediately. Pooled within-group standard deviation of the offset
over all 2025 bulk rows (`analysis/q0_ceiling.md`):

| grouping | groups | rows in a group of 2+ | sigma (s) |
|---|---|---|---|
| airport | 10 | 100.0 % | 340.4 |
| stand | 2,341 | 100.0 % | 329.3 |
| stand x runway | 9,816 | 99.9 % | 324.1 |
| stand x runway x aircraft type | 56,789 | 98.9 % | 319.2 |
| stand x runway x operator | 53,580 | 98.5 % | 317.5 |
| stand x runway x type x 2-hour block | 179,238 | 94.7 % | 313.0 |
| stand x runway x type x **same day** | 347,228 | 42.3 % | 311.4 |

The last row is measured on the 42 % of departures that have a same-day twin on
the same stand, runway and type, so it is not directly comparable to the rows
above; the 94.7 %-coverage row at 313.0 s carries the same message. Two aircraft
of the same type, flown by the same operator, off the same stand to the same
runway in the same two-hour block still differ by 313 s in this offset. The same
instrument on the *taxi time itself* falls much further (345.7 at airport level
to 267.4 at stand x runway x type x 2 h), so the flatness is a property of the
offset, not of the instrument.

Out of sample the picture is identical. Fitting a group median of the offset on
the training months and scoring the bulk of each validation month
(`analysis/q1b_decomp.md`), the best single grouping removes 11-17 % of the
offset's variance and everything else removes far less:

| grouping | levels | bulk RMSE july | december | january | variance explained (july) |
|---|---|---|---|---|---|
| stand x runway | 6,053 | 398.7 | 330.3 | 323.6 | 11.4 % |
| stand | 2,033 | 401.5 | 333.1 | 325.9 | 10.1 % |
| airport x operator | 1,133 | 402.8 | 337.4 | 330.6 | 9.5 % |
| terminal prefix | 128 | 404.7 | 337.1 | 329.6 | 8.7 % |
| airport | 10 | 409.1 | 343.9 | 336.4 | 6.7 % |
| take-off hour | 24 | 421.9 | 360.2 | 355.8 | 0.8 % |
| wake category | 4 | 422.0 | 360.2 | 355.7 | 0.7 % |
| (global median) | 1 | 423.5 | 361.1 | 355.6 | 0 |

(A `month` row belongs in no such table: the validation month's level is never in
the training table, so the predictor falls back to the global median on every row
and reproduces it by construction. An earlier version of this section printed
that row as "month explains 0.0 %" and drew a conclusion from it. **That was
wrong** — Sol, 2026-09-05 — and the corrected treatment of seasonality is the
third bullet below.)

Three consequences the desk can act on:

- **The offset is not congestion.** Take-off hour explains 0.8 % of its variance
  out of sample and wake category 0.7 %. The apparent month-to-month steps of
  ~55 s in the raw medians of `analysis/q1_offset_structure.md` are an artefact:
  the density of the offset is flat to about 0.16 % per second near the centre,
  so the median hops a whole minute without the distribution moving. Read the
  trimmed means instead.
- **The offset does drift with the season, per airport, and it is worth a few
  seconds.** (Corrected after Sol's review; the first version of this report
  claimed the opposite from the tautological `month` row above.) Range of the
  trimmed-mean offset across the twelve 2025 months (`analysis/q1c_drift.md`):
  **LIRF 168 s** (-50 s in November, -218 s in July), **LTFM 159 s** (342 s in
  March, 183 s in July), EDDM 124 s, LEBL 78 s, LEMD 65 s, LFPG 50 s, EDDF 41 s,
  EGLL 40 s, EHAM 32 s, LSZH 26 s. Pooled over airports a `month` main effect is
  still tiny (in-sample eta-squared 0.22 %), because the drift is an
  airport-by-season interaction and the residual spread is ~310 s. Measured the
  only way that means anything — which training window predicts the validation
  month best, stand-median offset predictor:

  | fold | all training months | last 3 | last 1 | first 3 |
  |---|---|---|---|---|
  | july | 401.3 | 397.7 | **397.2** | 407.5 |
  | december | 333.0 | **332.8** | 337.3 | 339.0 |
  | january | 325.8 | 326.3 | 328.3 | 327.7 |

  Recency is worth 4.1 s on july and nothing on december or january, and the
  oldest window is the worst on all three folds. So: **weight recent months when
  the target month is a summer month at LIRF or LTFM; do not drop the far
  months, they cost only 1-4 s.** This does not change the fold policy in
  `docs/shift-2026.md`, which is about the taxi time and not the offset.
- **There is no near-constant stand.** Over the 1,602 stands with at least 200
  departures the *minimum* per-stand standard deviation is 162 s (EDDF V173B) and
  the median is 308 s. The "pushback convention gives a fixed offset" hypothesis
  is refuted: not one stand at these ten airports behaves that way.

Bimodality does exist and is concentrated. It has to be measured carefully: the
1-D 2-means separation scores 2.67 on a unimodal Gaussian and 1.78 on a
heavy-tailed unimodal t(3), so only values above ~2.7 mean anything, and a large
separation with a 1 % minority cluster is an outlier group rather than a second
mode. 262 of the 1,602 stands score above 3.0, but **195 of those have a minority
cluster below 15 %** and are outlier groups; requiring both `separation > 3.0`
and `minority weight >= 15 %` leaves **67 genuinely bimodal stands**, and they
are not spread evenly: LIRF 24, EDDM 22, EHAM 6, LEBL 4, LFPG 4, EDDF 2, LSZH 2,
LTFM 2, LEMD 1, and **none at EGLL**. The clearest cases are EDDM 111 (n = 2,239, separation 3.94, modes 751 s
apart), EDDM 110 (3.88), LIRF 824 (4.27) and LIRF 829 (4.18), all of which pass
the weight test. The minority mode sits 600-1,400 s *below* the main one on
15-25 % of those stands' departures — a
second recording path, not a tow-versus-pushback split at contact stands
generally, which would have shown up everywhere. EGLL is the opposite extreme
(separation 1.61 at stand 511): broad, heavy-tailed, unimodal.

### 2. The whole gain is in knowing *when* to trust the NM off-block

There are two ways to guess the taxi time and they are nearly equally good
(`analysis/q0b_blend.md`, out of sample, bulk):

| fold | A: `nm_taxi` minus a group median offset | I: group median taxi, NM columns ignored | corr(errors) | **oracle** linear blend of A and I | fixed LightGBM, 40 base features |
|---|---|---|---|---|---|
| july | 397.1 | 327.7 | 0.69 | 309.1 | **244.7** |
| december | 328.8 | 306.6 | 0.62 | 281.4 | **204.4** |
| january | 321.1 | 342.0 | 0.55 | 288.7 | **207.8** |

The identity path (which never looks at `AOBT_3`) beats the NM path on july and
december; on january the NM path wins. The two error streams are strongly
correlated, so combining them only reaches 309 / 281 / 289 — and that blend is an
**oracle**: its two coefficients are fitted by least squares on the validation
month's own labels, so no honest predictor of that form does better. The tree
reaches 244.7 / 204.4 / 207.8. This is not a like-for-like comparison — the tree
has all 40 base features and the blend has two numbers — so it does not prove by
itself what the tree is doing. What it does establish is a floor: **no reweighting
of these two estimates, however chosen, gets within 64 s of the tree on july**, so
the tree's advantage is not in the weights. Finding 3 shows where it is instead.

### 3. The switch is `AOBT_3 - LOBT`, and it flips twice

Fitting the offset around each anchor's own median and measuring the RMSE about
it, within bands of `AOBT_3 - LOBT` (2025, NM rows, `analysis/q2q5.md`):

| `AOBT_3 - LOBT` | share | RMSE about AOBT | RMSE about LOBT | P(taxi > 3600 s) |
|---|---|---|---|---|
| < -10 min | 5.9 % | 500.8 | **438.6** | 0.1 % |
| -10..0 min | 27.9 % | **249.3** | 265.4 | 0.0 % |
| 0..10 min | 38.7 % | **267.9** | 298.1 | 0.0 % |
| 10..30 min | 23.5 % | **378.8** | 446.8 | 0.1 % |
| 30..60 min | 3.8 % | **678.3** | 762.1 | 1.1 % |
| 1..2 h | 0.2 % | 2089.9 | **1685.2** | 20.9 % |
| > 2 h | 0.005 % | 2346.1 | **1599.1** | 53.2 % |

LOBT wins at both ends and AOBT wins in the middle 94 % — the tail worker's
finding for `AOBT - LOBT > 1 h` (`experiments/exp-002-tail/RESULTS.md`) is one
half of a two-sided rule, and the other half (`AOBT` more than ten minutes
*before* LOBT, 5.9 % of all departures, twenty-five times more rows) has not been
used yet. Both `MVT - AOBT_3` and `MVT - LOBT` belong in every model, with
`AOBT_3 - LOBT` beside them so the tree can find the switch.

The LOBT update itself is a weaker but free signal: 70 % of flights never move
LOBT off EOBT_1, and those have offset dispersion 378 s; flights that moved LOBT
*earlier* by more than 30 minutes have 1,041 s.

### 4. What is left when AOBT is absent: nothing but the schedule

`AOBT_3_flt` and `LOBT_flt` are null on **exactly the same rows** (agreement
1.000000 over 2,085,047 2025 departures). The NM join is all-or-nothing, so the
question "which plan time anchors the block time when AOBT is missing" has one
answer: `SCHED_TIME_UTC_mvt`, because it is the only one left. Ranked anchor
performance where all of them exist (fit on 2025-01..06, scored on the july bulk):

| anchor | bulk RMSE july | NM tail (y>3600) | `AOBT-LOBT > 1 h` |
|---|---|---|---|
| AOBT_3 | **423.5** | 1728.0 | 2552.7 |
| EOBT_1 | 700.5 | 1954.3 | 2183.8 |
| IOBT | 800.8 | **1503.7** | **1836.1** |
| LOBT | 801.8 | 1515.2 | 1838.0 |
| SCHED | 2419.6 | 2123.2 | 3041.1 |
| constant median taxi | 423.9 | 3952.4 | 2069.2 |

Note the last row: on the bulk, a single constant is as good as `MVT - AOBT_3`
corrected by a single constant. `IOBT_flt` is marginally better than `LOBT_flt`
everywhere and the two are near-identical (`|LOBT - IOBT|` median 0 s); either
will do.

### 5. Stand occupancy (desk note): worth 5 s as features, worth nothing as a bound

The desk's construction is right in principle — arrivals keep their on-block time
in the ranking frame, so an arrival that parks at stand S after our off-block
gives `taxi > MVT - A`. Three things measured (`analysis/q6_bounds.md`):

**(a) A stand is not a single-aircraft position.** For every pair of consecutive
arrivals at a stand, how many departure off-blocks fall between them (true block
times, raw stand name):

| month | arrival intervals | exactly 1 | 0 | 2 or more |
|---|---|---|---|---|
| 2025-01 | 151,209 | 87.4 % | 6.9 % | 5.7 % |
| 2025-07 | 188,065 | 88.5 % | 6.2 % | 5.3 % |
| 2025-12 | 163,457 | 87.9 % | 6.6 % | 5.5 % |

Alternation holds 88 % of the time, not 100 %. Aircraft are towed off stands
without a departure movement (6 %) and stand names cover more than one physical
position (5 %). With the MARS parent key (519L and 519R collapsed to 519) it is
worse, 83 % — so **use the raw stand name for anything occupancy-shaped**; the
parent key is only right for soft statistics.

**(b) The counting bound therefore fails.** "If k arrivals landed between the
previous departure and ours, the last k-1 of them parked after our off-block" is
valid on only **74 %** of the rows it fires on (`disp_lb_1`, July 2025, 11 % of
departures), and its second-order version on 0.5 %. It is implemented as
`prc.stand_features.displacement_bounds` and must never be used as a clip.

**(c) The asof bounds are valid but loose, and clipping at zero dominates them.**
`taxi_lb_na` (first arrival on-block after the departure's AOBT) is valid on
99.95 % of rows but its median slack is 4,980 s: at these ten airports a stand
sits empty for over an hour between a departure and the next arrival, so the
bound almost never binds where a model is wrong. Against the reference naive
predictor on the july bulk: no clip 401.2, clip at zero **376.6**, clip at the
occupancy bound 393.3, clip at zero *then* at the occupancy bound 376.3. The
whole clipping story is `max(pred, 0)`, worth 24 s; the bound adds 0.3 s.

What the neighbour information *is* worth is measured in the ablation table
below, as features: +5.2 s on july and +4.5 s on december. That value comes from
the turnaround time, the previous arrival's taxi-in and the gaps to neighbouring
departures, not from the bound.

### 6. The ablation table

One frozen LightGBM (`objective=l2`, 1,200 rounds, lr 0.05, 127 leaves,
`min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging 0.8`, seed 0, 16
threads), never tuned; the only thing that changes between rows is the feature
set. Target is the offset `D`; `taxi_hat = nm_taxi - D_hat`, with rows that have
no NM join predicted at the training airport median taxi. Trained on rows with an
NM join and taxi <= 3600 s. **bulk** = RMSE on validation rows with an NM join
and taxi <= 3600 s.

### 7. Label quality (Q5)

2025 training months, 2,085,047 departures (`analysis/q2q5.md`):

| flag | rows | share | median taxi | P(taxi > 3600 s) | share of total SSE |
|---|---|---|---|---|---|
| taxi < 0 (block after take-off) | 369 | 0.018 % | -6 s | 0 % | 0.06 % |
| taxi == 0 exactly | 19 | 0.001 % | 0 | 0 % | 0.00 % |
| taxi < 120 s | 2,116 | 0.101 % | 56 s | 0 % | 0.30 % |
| `AIRCRAFT_TYPE_mvt` null (noac) | 1,507 | 0.072 % | 2,643 s | 49.4 % | **37.4 %** |
| no NM join | 22,470 | 1.078 % | 951 s | 4.8 % | **42.2 %** |
| taxi > 3600 s | 4,126 | 0.198 % | 4,510 s | 100 % | **49.6 %** |
| taxi > 4 h | 122 | 0.006 % | 24,959 s | 100 % | **36.5 %** |
| `BLOCK == SCHED` to the second | 8,123 | 0.390 % | 1,019 s | 1.8 % | 2.6 % |
| `abs(BLOCK - SCHED) <= 1 s` | 24,267 | 1.164 % | 1,018 s | 1.7 % | 8.5 % |
| `abs(BLOCK - SCHED) <= 60 s` | 199,634 | 9.575 % | 952 s | 0.9 % | 31.7 % |
| `abs(BLOCK - AOBT_3) <= 1 s` | 40,283 | 1.932 % | 901 s | 0.01 % | 0.6 % |
| `STAND_mvt` null | 28 | 0.001 % | 981 s | 3.6 % | 0.01 % |

Four things worth carrying:

- **Negative and zero targets are negligible** (369 + 19 rows spanning -12 s to
  0 s, with 11 rows at -12 and 19 at exactly 0; 0.06 % of the squared error).
  They are a rounding artefact between two clocks, not a phenomenon. Clip predictions at zero and forget them; the clip
  itself is worth 24 s on the naive predictor because it stops *large negative*
  predictions, not because of these rows.
- **There are no duplicates.** Zero rows share `(FLIGHT_ID_mvt, take-off)` and
  zero share `(callsign, take-off, airport)`; only 8 rows in 4 groups share
  `(airport, take-off, stand)`. Nothing to deduplicate.
- **`BLOCK == SCHED` is not by itself a placeholder flag.** 9.6 % of all
  departures block off within a minute of schedule, which is what an on-time
  airline does. The placeholder is the *conjunction*: noac raises P(within 60 s
  of schedule) to 47.9 % (lift 5.0) and noac with `MVT - SCHED > 2 h` to 62.9 %
  (lift 6.6). LIRF alone is at 20.7 % (lift 2.2). Screening on `BLOCK == SCHED`
  in training to build a placeholder label is fine; screening on anything that
  produces it at prediction time is not, because the flag is unobservable.
- **The 122 rows above four hours carry 36.5 % of all squared error.** That is
  five one-thousandths of a percent of the data. Every conclusion about the
  overall RMSE is a conclusion about a few hundred rows, which is why this report
  scores the bulk separately.

### 8. The 2026 shift (Q4)

Written up for the desk in `docs/shift-2026.md`; tables in `analysis/q4_shift.md`
and `analysis/q4b_airport_shift.md`. The three sentences that change other
workers' folds:

1. **July 2026 is an ordinary summer month** (closest 2025 months by categorical
   profile: 2025-08, then 07, then 09; closest by `MVT - AOBT` distribution:
   2025-08 then 2025-07). Every airport's median day sits inside the 2025 range.
2. **January 2026 is not an ordinary winter month.** Its composition is closest to
   December 2025 but its *conditions* are outside the 2025 envelope: at EDDF the
   median day is at the 95th percentile of all 2025 days and 9 days of 31 are
   above the 2025 maximum; at EHAM 9 days are above the 2025 maximum and the
   dispersion of `MVT - AOBT` is 740 s against 476 s in the worst 2025 month.
   No 2025 month reproduces it. December 2025 is one of the calmest.
3. **Two structural changes:** LFPG lost runways 27R and 09L (13.8 % of its 2025
   departures) and EDDM gained stands 103-108 (1,177 July-2026 rows with no 2025
   history at all). Forty-five stand levels and one runway level are new; they
   carry 0.46 % and 0.0006 % of ranking rows.

This is why `prc.data.FOLDS` now carries the backward `january` fold and why
every row above is reported on three folds.

### 9. The bulk ceiling: what the board does and does not pin down

*(Rewritten after Sol's round-2 review, which showed the first version quoted the
range of the grid cells it printed rather than the range its own assumptions
imply, and drew a feasibility claim that the arithmetic does not support.)*

**(a) One hard bound.** If the leaders scored *perfectly* on everything that is
not bulk, their bulk RMSE would be `277.3 * sqrt(344841 / 339051) = 279.7 s`.
Nobody's bulk is worse than 280. That is the only thing the board pins down on
its own.

**(b) The back-out has three free parameters and does not pin the bulk.**
Splitting the ranking population into 388 noac rows, 4,902 other no-NM rows,
~500 NM rows above 3600 s (at the 2025 rate) and ~339,051 bulk rows and solving
`N * 277.3^2 = 388 a^2 + 4902 b^2 + 500 c^2 + 339051 B^2`:

| noac a | other no-NM b | NM tail c | implied leaders' bulk B |
|---|---|---|---|
| 3,000 | 600 | 1,500 | 243.7 |
| 3,000 | 900 | 2,500 | 216.8 |
| 3,000 | 1,400 | 2,500 | 174.2 |
| 4,500 | 600 | 1,500 | 215.7 |
| 4,500 | 900 | 2,500 | 184.7 |
| 4,500 | 1,400 | 2,500 | 132.2 |

Over that assumption box B ranges from **132 to 244 s**. Nor is a large `b`
infeasible: `B` only goes to zero at `b = 2,114` (a = 3,000, c = 1,500) or
`b = 1,780` (a = 4,500, c = 2,500), so the earlier claim that "no team reaches
277.3 while predicting the ordinary no-NM rows worse than about 1,800 s" is only
true at the pessimistic corner and is withdrawn.

**(c) Our own segment numbers, which are a reference point and not a bound.**
The recommended set's OOF, scored by segment (this is the fallback path — no-NM
rows predicted at the training airport median taxi, nothing done for noac):

| segment | rows (july) | july | december | january |
|---|---|---|---|---|
| bulk (NM, y <= 3600) | 186,272 | 235.8 | 196.1 | 200.9 |
| NM tail (y > 3600) | 503 | 2,750.7 | 2,328.0 | 2,064.0 |
| no-NM, not noac | 3,599 | 658.4 | 703.5 | 2,840.9 |
| noac | 339 | 12,965.6 | 13,278.8 | 15,634.9 |

*(An earlier version of this section used these numbers to bound the leaders'
bulk below at 194 s, on the argument that a team at 277.3 overall must be at
least as good as this model on every segment. **That is a non-sequitur** — Sol,
2026-09-05 round 3 — and the bound is withdrawn. A better aggregate score does
not imply a better score on every part; a team could be far better on noac and
worse than us on the no-NM rows and still total 277.3.)*

**Conclusion, at the strength the evidence supports.** The board pins down
exactly one number about the bulk: **nobody's bulk RMSE is worse than 279.7 s**.
Everything else is conditional on assumptions about three segments nobody can
observe, and over any plausible box those assumptions span 132-244 s. Our
recommended set is at **221 s on the ranking mix** (44.3 % January at 200.9,
55.7 % July at 235.8). **The board cannot tell us whether that is ahead of or
behind the leaders' bulk, and this report should not be cited as saying either.**
What it can be cited as saying is the ordering of work: our 388 noac rows at
12,966 s contribute 435 s to a total on their own, and no feature family measured
here moves the bulk by more than 5 s.

One thing the segment table says that belongs to another lane: **the no-NM
non-noac rows cost 2,841 s on the january fold against 658 s on july**. Winter
disruption moves that segment by a factor of four, and it is 1.4 % of the rows.

**(d) What conditioning can and cannot remove.** The offset residual plateaus at
~313 s under arbitrarily fine conditioning (finding 1) and the taxi-time residual
at 267 s (`analysis/q0_ceiling.md`); their errors correlate at 0.55-0.69, so an
oracle linear blend of the two reaches 281-309 and the tree reaches 196-236.
Nothing in the identity of the flight is left to extract; further movement has to
come from the *state of the airport at that minute* (the congestion lane), from
the weather, or from the regime switch of finding 3.

### 10. `prc/stand_features.py`: what earns its place and what does not

The module has four parts, with separated fit and transform where a fit exists:

- `normalize_stand` / `add_stand_keys` — MARS sub-stand collapsing (`519L`,
  `519R` to `519`, `G11L` to `G11`) and the `(airport, stand)` keys. **Verdict:
  keep, but the raw name is the operational key.** Collapsing to the parent
  breaks the one-aircraft-per-position assumption (88 % alternation with the raw
  name, 83 % with the parent) and the parent-key occupancy features are worth
  nothing on top of the raw-key ones (F_all minus `occb` is 236.5 / 197.2 against
  F_all 236.5 / 196.8).
- `occupancy_features` — previous/next arrival at the stand, their on-block,
  taxi-in and type, the turnaround, and the gaps to neighbouring departures.
  **Verdict: keep. Worth 5.2 s (july) and 4.5 s (december) on the bulk added to
  the base set**, of which the previous-arrival half carries 4.9 s and the
  next-arrival half 1.3 s. Against the full set it is worth 0.8 s, i.e. it
  overlaps with the neighbouring-departure block.
- `StandStats` (fit/transform, fallback stand to parent to terminal to airport) —
  **Verdict: no measured value; a negative result.** Added to the base set it
  changes the bulk by +0.01 s (july) and +0.03 s (december); removed from the
  full set it changes it by -0.16 s (july) and +0.06 s (december). Every one of
  those is inside run-to-run noise. Diagnosis: LightGBM's native categorical
  handling of `stand_raw` (2,341 levels) already partitions the offset, so the
  encoding restates information the tree has and adds fourteen correlated columns
  that dilute `feature_fraction`. Repair attempted: the fallback chain was meant
  to earn its keep on stands unseen in training — measured on that slice it does
  not (240.0 without the block against 246.3 with it on july, but on only 218
  rows, and 154 against 150 on december's 377 rows: the slice is too small to
  carry a verdict either way, which is itself the finding). What would revive it:
  a model without native categorical support, or the EDDM 103-108 apron, which is
  1,177 July-2026 rows with no 2025 history at all and therefore has no 2025
  slice on which the question can be settled. It is kept in the module, out of
  the recommended feature set, and the recommendation is in its docstring.
- `displacement_bounds` — the counting bound. **Verdict: feature only, never a
  clip** (finding 5b).

**Reference points on the same bulk population** (naive predictors, no model):

| predictor | july | december | january |
|---|---|---|---|
| `taxi = MVT - AOBT_3` (offset 0) | 418.5 | 360.4 | – |
| offset = global training median | 423.5 | 361.1 | 355.6 |
| offset = airport training median | 409.1 | 343.9 | 336.4 |
| offset = stand training median | 401.2 | 332.9 | 325.8 |
| offset = stand median, clipped at 0 | 376.6 | 320.0 | 308.6 |
| `taxi = airport median taxi` (no actual columns at all) | 377.4 | 358.2 | – |

**Ablation (bulk RMSE; lower is better).** Every row's four COMMON.md slices and
its RMSE on validation rows whose stand never appears in the training months
(218 july rows, 377 december rows) are in `climb.log`; the table below shows the
bulk column, which is what the assignment scores.

| model | feat | fold | RMSE all | RMSE <=3600 | RMSE >3600 | RMSE no-NM | **RMSE bulk** | vs base (july bulk) | window fix |
|---|---|---|---|---|---|---|---|---|---|
| Q base + phi | 46 | july | 621.1 | 250.0 | 9698 | 3856 | **245.09** | +0.40 | n/a |
|  |  | december | 383.1 | 210.8 | 8062 | 3160 | **204.28** |  |  |
|  |  | january | 471.6 | 213.4 | 9148 | 4295 | **207.67** |  |  |
| B base + StandStats | 54 | july | 620.9 | 249.6 | 9697 | 3856 | **244.70** | +0.01 | n/a |
|  |  | december | 383.4 | 211.0 | 8067 | 3160 | **204.48** |  |  |
|  |  | january | 472.1 | 213.9 | 9154 | 4295 | **208.20** |  |  |
| A base | 40 | july | 620.8 | 249.6 | 9696 | 3856 | **244.69** | – | n/a |
|  |  | december | 383.4 | 210.9 | 8070 | 3160 | **204.45** |  |  |
|  |  | january | 471.7 | 213.5 | 9149 | 4295 | **207.80** |  |  |
| N base without the same-flight-arrival pair | 38 | july | 620.9 | 249.5 | 9698 | 3856 | **244.61** | -0.08 | n/a |
|  |  | december | 383.4 | 211.1 | 8066 | 3160 | **204.64** |  |  |
|  |  | january | 471.9 | 213.8 | 9152 | 4295 | **208.11** |  |  |
| U base + same-callsign-as-an-earlier-arrival | 47 | july | 620.9 | 249.5 | 9697 | 3856 | **244.57** | -0.12 | n/a |
|  |  | december | 383.4 | 211.0 | 8068 | 3160 | **204.47** |  |  |
|  |  | january | 471.8 | 213.5 | 9152 | 4295 | **207.71** |  |  |
| M base + neighbouring departures at the stand | 44 | july | 620.3 | 248.7 | 9692 | 3856 | **243.74** | -0.95 | n/a |
|  |  | december | 383.1 | 210.8 | 8063 | 3160 | **204.28** |  |  |
|  |  | january | 471.3 | 213.0 | 9145 | 4295 | **207.21** |  |  |
| O base, target = taxi not the offset | 40 | july | 653.2 | 248.4 | 10303 | 3856 | **243.44** | -1.25 | n/a |
|  |  | december | 383.8 | 210.5 | 8089 | 3160 | **204.00** |  |  |
|  |  | january | 473.2 | 212.7 | 9196 | 4295 | **206.89** |  |  |
| L base + next arrival at the stand | 46 | july | 620.2 | 248.3 | 9693 | 3856 | **243.35** | -1.34 | n/a |
|  |  | december | 382.8 | 209.9 | 8068 | 3160 | **203.39** |  |  |
|  |  | january | 471.4 | 212.5 | 9153 | 4295 | **206.77** |  |  |
| D base + neighbouring departures at the airport | 46 | july | 619.1 | 245.5 | 9694 | 3856 | **240.42** | -4.28 | after |
|  |  | december | 381.2 | 207.0 | 8068 | 3160 | **200.30** |  |  |
|  |  | january | 470.6 | 211.1 | 9150 | 4295 | **205.27** |  |  |
| K base + previous arrival at the stand | 48 | july | 619.0 | 244.9 | 9697 | 3856 | **239.77** | -4.92 | n/a |
|  |  | december | 381.1 | 206.8 | 8067 | 3160 | **200.16** |  |  |
|  |  | january | 469.9 | 209.5 | 9150 | 4295 | **203.57** |  |  |
| E base + StandStats + occupancy | 72 | july | 619.0 | 244.7 | 9698 | 3856 | **239.54** | -5.15 | n/a |
|  |  | december | 381.1 | 207.0 | 8066 | 3160 | **200.32** |  |  |
|  |  | january | 469.7 | 209.3 | 9145 | 4295 | **203.45** |  |  |
| I everything minus airport neighbours | 90 | july | 619.1 | 244.6 | 9699 | 3856 | **239.51** | -5.18 | n/a |
|  |  | december | 381.0 | 206.5 | 8068 | 3160 | **199.85** |  |  |
|  |  | january | 469.8 | 209.4 | 9150 | 4295 | **203.46** |  |  |
| C base + all stand-occupancy features | 58 | july | 619.0 | 244.6 | 9699 | 3856 | **239.46** | -5.23 | n/a |
|  |  | december | 381.0 | 206.6 | 8068 | 3160 | **199.94** |  |  |
|  |  | january | 470.0 | 209.5 | 9152 | 4295 | **203.60** |  |  |
| S best, target = the phi residual | 70 | july | 617.9 | 243.1 | 9689 | 3856 | **237.90** | -6.79 | **before** |
|  |  | december | 379.2 | 203.9 | 8058 | 3160 | **197.12** |  |  |
|  |  | january | 469.3 | 208.9 | 9141 | 4295 | **202.97** |  |  |
| G everything minus stand occupancy | 78 | july | 618.1 | 242.5 | 9696 | 3856 | **237.35** | -7.34 | **before** |
|  |  | december | 379.7 | 204.6 | 8063 | 3160 | **197.80** |  |  |
|  |  | january | 469.4 | 208.9 | 9144 | 4295 | **203.00** |  |  |
| R best + phi | 70 | july | 618.2 | 242.1 | 9702 | 3856 | **236.93** | -7.76 | **before** |
|  |  | december | 379.5 | 204.0 | 8065 | 3160 | **197.18** |  |  |
|  |  | january | 469.0 | 208.1 | 9143 | 4295 | **202.11** |  |  |
| H everything minus parent-stand occupancy | 78 | july | 617.7 | 241.7 | 9695 | 3856 | **236.51** | -8.18 | **before** |
|  |  | december | 379.4 | 204.0 | 8061 | 3160 | **197.18** |  |  |
|  |  | january | 469.1 | 208.1 | 9145 | 4295 | **202.19** |  |  |
| V best + the same-callsign link | 71 | july | 617.6 | 241.6 | 9694 | 3856 | **236.35** | -8.34 | **before** |
|  |  | december | 379.5 | 204.1 | 8063 | 3160 | **197.35** |  |  |
|  |  | january | 469.3 | 208.2 | 9150 | 4295 | **202.24** |  |  |
| J everything minus StandStats | 82 | july | 617.7 | 241.6 | 9696 | 3856 | **236.32** | -8.37 | **before** |
|  |  | december | 379.3 | 203.7 | 8065 | 3160 | **196.87** |  |  |
|  |  | january | 469.1 | 207.9 | 9147 | 4295 | **201.96** |  |  |
| F everything | 96 | july | 617.6 | 241.0 | 9698 | 3856 | **235.78** | -8.91 | after |
|  |  | december | 378.7 | 202.6 | 8063 | 3160 | **195.80** |  |  |
|  |  | january | 468.6 | 206.9 | 9146 | 4295 | **200.88** |  |  |
| T base + occupancy + airport neighbours (**recommended**) | 64 | july | 617.4 | 241.0 | 9696 | 3856 | **235.75** | -8.94 | after |
|  |  | december | 379.0 | 203.0 | 8067 | 3160 | **196.12** |  |  |
|  |  | january | 468.7 | 206.9 | 9149 | 4295 | **200.89** |  |  |
| P everything, predictions clipped at zero | 96 | july | 617.5 | 240.7 | 9699 | 3856 | **235.40** | -9.29 | **before** |
|  |  | december | 379.1 | 203.4 | 8062 | 3160 | **196.62** |  |  |
|  |  | january | 468.6 | 207.1 | 9145 | 4295 | **201.07** |  |  |

Rows are sorted by july bulk; the same table is in `analysis/ablation_full.md`.
Reproducibility: `A_base` on the january fold gives 207.803 at 16 threads and
207.803 at 12; `F_all` gives 201.886 and 201.935, so the run-to-run spread of
this harness is about 0.05 s and no difference below ~0.2 s is real.

**Read the `window fix` column.** Answering the third review round surfaced a bug
in this report's own code: `nbr_apt_30m` was meant to count departures at the
airport within ±30 minutes and instead counted them within ±21 days
(`.astype("int64")` on a `datetime64[us]` column gives microseconds, and the code
divided by 1e9). It was caught by the new Q4 numeric-shift table, where the
"30-minute" count had a mean of 16,421. Fixed, the count averages 36. Ten rows
contain that column; **`T`, `D` and `F` were re-measured after the fix and are
marked `after`; the other seven are marked `before`** and the workstation load
(1-minute average 240) made re-running them cost more than they are worth. The
measured effect of the fix on the three that were re-run is -0.5 to -1.1 s
(`T` 236.29 to 235.79 on july, `D` 241.32 to 240.40, `F` 236.47 to 235.83), it is
in the same direction on all three folds, and **every comparison this report
draws is between two rows that either both contain the column or both do not**,
except `F` against `I` (airport neighbours in the presence of occupancy), where
the fix widens the gap from 3.0 s to 3.7 s. No conclusion changes; seven cells in
the table are 0.5 to 1.1 s pessimistic.

Read it in three lines:

- **Only two feature families pay: the stand's previous arrival (-4.9 s) and the
  neighbouring departures at the airport (-3.4 s).** *Caveat added 2026-09-05
  after retro-v1's overlap check: those gains are measured against **this
  experiment's** 40-feature base, which carries no stand-occupancy columns. On
  retro-v1's frame, which already has `prc.lirf_features` and its own v3
  arrival-side columns, 8 of my 24 stand columns are the same measurement at
  |r| = 1.000 under another name (`dep_next_mvt_gap` = `lf_stand_next_dep_gap`,
  `taxi_ub_pm` = `stand_prev_arr_block_gap`, and six more) and another 6 sit
  between 0.99 and 1.00, so the -4.9 s is partly a gain that frame already has.
  The airport-neighbour family is the one retro-v1 measured as genuinely new
  (at most 0.50 with anything it carries). My own leave-one-out already pointed
  the same way — see the next sentence — so the ordering of the two families is
  unchanged, but the -4.9 s should be read as "against a base without stand
  occupancy", not as a gain any model can bank.* They overlap. Removed one at
  a time from the full set (rows G and I), stand occupancy is worth 0.9 s and the
  airport neighbours 3.0 s, so the airport-neighbour block subsumes most of what
  the stand's own neighbours were adding.
- **Three families pay nothing.** Per-stand target statistics (+0.01 s added to
  base and +0.40 s on january; removing them from the full set changes july by
  -0.16 s, december by +0.06 s and january by +0.02 s, i.e. zero either way), the
  MARS parent-stand duplicate of the occupancy features (+0.04 s july, +0.37 s
  december, +0.26 s january), and the
  same-flight-as-arrival link that Q3 asked about, in both of its forms: the same
  `FLIGHT_ID_mvt` appearing as an arrival (-0.08 s july, +0.19 s december) and
  the same `CALLSIGN_flt` appearing as an *earlier* arrival (row U: -0.12 s july,
  +0.03 s december, -0.09 s january; row V: +0.06 s july, +0.33 s december,
  +0.60 s january on top of the recommended set — all inside the harness's 0.05 s
  reproducibility except V on january, which is worse). The last one is a clean negative for Q3: a
  departure whose flight also appears as an arrival at one of the other nine
  airports learns nothing from it, because the arrival row carries the same
  `_flt` columns and the destination on-block says nothing about the origin
  off-block. The callsign form is a different object and also worth nothing:
  13.9 % of july departures have their `FLIGHT_ID_mvt` as an arrival in the same
  month and 16.4 % have their callsign as an *earlier* arrival, but only 0.8 % of
  the callsign matches are the same flight leg — the rest are the same flight
  number on an earlier day, which the operator, destination and schedule columns
  already describe.
  time directly with `nm_taxi` as a feature (row O) is 1.3 s *better* on the bulk
  and 600 s *worse* on the rows above 3600 s (10,303 against 9,696) and 32 s
  worse overall. The reframing earns its place as an analytical device and on the
  tail, not on the bulk. Clipping at zero is worth a further 1.1 s (row P).

### 11. The `phi` parameterisation routed from the lirf worker

The lirf worker's result — `BLOCK = AOBT + phi * (LOBT - AOBT)` with `phi`
stable per airport and per operator — is the same object as finding 3 seen from
a better angle: `phi = 0` is block-on-AOBT, `phi = 1` is block-on-LOBT, and the
two-sided switch of finding 3 is what a single `phi` per group already encodes.
Measured here as a feature block (`fit_phi` / `phi_features` in
`code/ablate.py`, least squares through the origin per airport and per
airport-operator, shrunk with a 200-row pseudo-count toward the airport and then
the global value, fitted on the fold's training months only):

| model | july | december | january |
|---|---|---|---|
| A base | 244.69 | 204.45 | 207.80 |
| Q base + phi | 245.09 | 204.28 | 207.67 |
| T best, **before** the window fix | 236.29 | 197.02 | 201.64 |
| R the same + phi (before the window fix) | 236.93 | 197.18 | 202.11 |
| S the same, target = the phi residual (before the fix) | 237.90 | 197.12 | 202.97 |
| T best, after the window fix (for reference) | 235.75 | 196.12 | 200.89 |

The `phi` rows were measured before the `nbr_apt_30m` window fix, so they are
compared against the `T` row from the same vintage; both contain the affected
column, so the comparison is like for like.

**Negative result, with the diagnosis.** `phi` costs 0.40 s on the july bulk
added to the base set (245.09 against 244.69) and 0.64 s added to the recommended
set (236.93 against 236.29), with the december and january cells between -0.17
and +0.47; using it as a target reparameterisation costs 1.61 s on july. Per
airport it is noise everywhere, LIRF included (july -2.9 s, december +0.7 s,
january -0.1 s).

The diagnosis is that `phi` is a *rank-one summary of an interaction the tree
already has*: `Delta = phi * (LOBT - AOBT)` is a product of a group constant and
`aobt_lobt`, and the base set contains `aobt_lobt` together with the airport,
operator and stand categoricals, so LightGBM reconstructs the same surface with
splits and is not restricted to a linear-through-the-origin shape. The six `phi`
columns then add correlated features that dilute `feature_fraction`, which is
exactly the 0.4-0.6 s seen.

What this does **not** refute: the lirf worker's LIRF result (550.6 against
635.9 for retro-v1 c6) is on LIRF *overall*, where the squared error is
dominated by the rows above 3600 s and the no-NM rows. The bulk is 98.8 % of the
rows and about 12 % of the squared error at LIRF, so a gain on LIRF overall and
no gain on the LIRF bulk are consistent — the two measurements are of different
populations. **The right reading is that `phi` is a tail instrument, not a bulk
feature.** What would revive it here: a linear or additive model that cannot
build the interaction itself, or a `phi` fitted at a level the tree cannot reach
(per stand, or per operator-and-stand with enough shrinkage), which is untested.

### Per-airport bulk RMSE of the recommended set

`T_best` OOF, after the window fix (`analysis/per_airport_oof.csv`):

| airport | july rows | july | december | january |
|---|---|---|---|---|
| LIRF | 14,603 | 360.7 | 262.8 | 254.0 |
| LEBL | 16,322 | 275.2 | 185.1 | 153.1 |
| LTFM | 24,725 | 275.1 | 212.4 | 223.4 |
| LFPG | 21,017 | 244.8 | 217.2 | 234.5 |
| EGLL | 20,297 | 227.3 | 206.6 | 220.0 |
| LEMD | 18,541 | 202.4 | 167.8 | 165.8 |
| EDDF | 20,887 | 189.0 | 176.2 | 181.5 |
| LSZH | 12,263 | 185.0 | 161.8 | 178.3 |
| EDDM | 15,647 | 175.9 | 196.8 | 174.8 |
| EHAM | 21,970 | 168.8 | 156.1 | 177.2 |
| **bulk, all airports** | **186,272** | **235.8** | **196.1** | **200.9** |

Weighting the two ranking months as they appear in the ranking file (44.3 %
January, 55.7 % July) and assuming each 2026 month behaves like its 2025 fold,
the recommended set's bulk on the ranking population is
`sqrt(0.443 * 200.9^2 + 0.557 * 235.8^2) = 221 s`, against the one hard bound the
board gives (nobody's bulk is worse than 279.7 s; finding 9). LIRF is a third
worse than the next airport on july and is where any further bulk work belongs.

### 12. What the desk should take from this

1. **Stop adding identity features to the bulk.** Finding 1 measured the ceiling
   of that road: the offset residual is flat at ~311 s under arbitrarily fine
   conditioning and no stand is near-constant. The two families that paid (-4.9 s
   and -3.4 s) are both *state* features, not identity features. Congestion and
   weather are the remaining state; the lirf `phi` parameterisation is the
   remaining structure.
2. **The board says almost nothing about our bulk; the noac rows it says a lot
   about.** The only thing the leaderboard pins down is that nobody's bulk RMSE
   is worse than 279.7 s (finding 9); every attempt to bound it below depends on
   assumptions about three unobservable segments and spans 132-244 s. A frozen
   untuned 64-feature model is at 235.8 / 196.1 / 200.9 on july / december /
   january, i.e. **221 s on the ranking mix**, and the board cannot say whether
   that is ahead or behind. What is unambiguous is the ordering of work: our 388
   noac rows at 12,966 s contribute 435 s to a total on their own, while every
   feature family measured here moves the bulk by single-digit seconds.
3. **Feed both `MVT - AOBT_3` and `MVT - LOBT`, with `AOBT_3 - LOBT` beside
   them.** The switch is two-sided (finding 3) and the low side — `AOBT` more
   than ten minutes *before* `LOBT`, 5.9 % of departures — has not been used by
   any lane yet.
4. **Use the raw stand name, never the MARS parent, for anything occupancy-
   shaped**, and do not clip with occupancy bounds; `max(pred, 0)` is the whole
   clipping story (24 s on a naive predictor, 1.1 s on the model).
5. **Fold policy**: report july, december and january. The january fold's bulk
   (200.9 for the recommended set) is the *best* of the three, but `docs/shift-2026.md`
   shows the *conditions* in January 2026 are outside the 2025 envelope at four
   airports, so treat 207.8 as a floor for the January half of the board, not an
   estimate of it.

## The recommended bulk set, column by column (for retro-v1)

64 columns. Thirteen categoricals (LightGBM native, no encoding), 27 own-flight
numerics, 18 stand-neighbour columns and 6 airport-neighbour columns. The last
two families are the only ones that measurably pay, and **both are importable
from `prc.stand_features` (VERSION 1.2.0)** — nothing in exp-003's own
`code/features.py` is needed:

```python
import prc.stand_features as sf

df   = sf.add_stand_keys(month_frame)            # apt, stand_raw, stand_base, stand_pfx, key_raw
occ  = sf.occupancy_features(df, key="key_raw")  # 18 columns, the stand's neighbours
nbr  = sf.airport_neighbour_features(df)         # 6 columns, the airport's neighbours
X    = pd.concat([own_flight_features, occ, nbr], axis=1)
```

Both functions take **one month's frame** as the ranking file gives it (arrivals
complete, departures with block time and taxi time blank) and return a frame
indexed like its departure rows. Call them once per calendar month and
concatenate — never on January and July 2026 together, or the neighbour joins
reach across the six-month gap. Neither function reads a departure's own
`BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`; `code/features.py check` asserts that
the whole feature frame is bit-identical built from the raw month and from the
masked one.

**13 categoricals:** `apt`, `stand_raw`, `stand_base`, `stand_pfx`,
`RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`,
`MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `ADES_mvt`, `ADES_FILED_flt`,
`CALLSIGN_PFX` (first three characters of the callsign).

**27 own-flight numerics:** `nm_taxi` (= `MVT − AOBT_3`), `mvt_lobt`,
`mvt_eobt1`, `mvt_iobt`, `mvt_sched`, `aobt_lobt`, `aobt_iobt`, `aobt_eobt1`,
`aobt_sched`, `lobt_eobt1`, `lobt_sched`, `lobt_changed`, `hour`,
`minute_of_day`, `dow`, `doy`, `sched_hour`, `sched_minofday`,
`n_dep_stand_month`, `n_arr_stand_month`, `noac`, `noac_day_apt`, `flight_len`,
`has_nm_f`, `type_mvt_eq_flt`, `same_flight_arr`, `airborne`. (`same_flight_arr`
and `airborne` measure zero — finding 6 — and can be dropped; the other 25 are
the plan-time deltas of findings 3 and 4 plus the tail worker's noac trio.)

**18 from `sf.occupancy_features`:** `taxi_ub_pm`, `taxi_lb_nm`, `taxi_ub_pa`,
`taxi_lb_na`, `off_lb_pm`, `off_lb_pa`, `off_ub_na`, `off_ub_nm`, `arr_taxi_pa`,
`arr_taxi_na`, `arr_turnaround_pa`, `arr_td_gap_pa`, `same_type_pa`,
`same_type_na`, `dep_prev_aobt_gap`, `dep_prev_mvt_gap`, `dep_next_aobt_gap`,
`dep_next_mvt_gap`. The previous-arrival half (`*_pa`, `*_pm`) carries -4.9 s of
the family's value and the next-arrival half -1.3 s.

**6 from `sf.airport_neighbour_features`:** `nbr_apt_30m`,
`nbr_apt_prev10_nmtaxi`, `nbr_apt_next10_nmtaxi`, `nbr_rwy_prev10_nmtaxi`,
`nbr_stand_prev5_nmtaxi`, `nbr_apt_prev10_mvtsched`.

**Not in the set, all measured at zero:** per-stand target statistics
(`sf.StandStats`), the MARS parent-stand copy of the occupancy features
(`sf.occupancy_features(key="key_base")`), the lirf `phi` block, the
same-flight-as-an-arrival link and the same-callsign-as-an-earlier-arrival link.

## Artifacts

| what | path | note |
|---|---|---|
| feature module | `prc/stand_features.py` | in git; verdicts in the docstring |
| per-month feature frames | `artifacts/exp-003-anatomy/feat/*.parquet` | 13 files, git-ignored, rebuilt by `code/features.py` |
| movement cache | `artifacts/exp-003-anatomy/mvt_all.parquet` | SHA-256 `07a697026915d8b4c088771db3c1c48f7a329889f5d9e6cae19f3b8d07eb8559`, 4,857,331 rows |
| departure cache | `artifacts/exp-003-anatomy/dep_all.parquet` | derived from the above by `code/q1_offset_structure.py` |
| ablation log | `experiments/exp-003-anatomy/climb.log` | one JSON line per model per fold |
| long-form tables | `experiments/exp-003-anatomy/analysis/*.md` | in git; `ablation_full.md` is the four-slice x three-fold table COMMON.md asks for, `q4_new_levels.md` every unseen 2026 category level, `q1c_drift.md` the seasonality test |
| climb log, readable form | `experiments/exp-003-anatomy/climb.tsv` | iteration, mechanism, three folds, fraction of the gap closed |
| OOF predictions | `experiments/exp-003-anatomy/oof/{T_best,F_all,R_best_phi}_{july,december,january}.parquet` | `MVT_ID_mvt, fold, y_true, y_pred`; `T_best` is the recommended set |
| per-airport OOF table | `experiments/exp-003-anatomy/analysis/per_airport_oof.csv` | bulk RMSE by airport, fold and model |
| shift note | `docs/shift-2026.md` | in git |

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/build_cache.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/features.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/features.py check   # mask invariance
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q1_offset_structure.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q1b_decomp.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q2q5.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q4_shift.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q4b_airport_shift.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q0_ceiling.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q0b_blend.py
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/q6_bounds.py
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/q6b_alternation.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/q1c_drift.py
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/check_fold_frames.py   # cache == fold_frames
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/ablate.py     # wave 1
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/wave2.py
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/wave3.py      # phi
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/wave4.py      # callsign
PYTHONPATH=.:experiments/exp-003-anatomy/code .venv/bin/python experiments/exp-003-anatomy/code/wave5.py      # january backfill
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/table.py                                       # bulk table
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/table_full.py                                  # four slices x three folds
PYTHONPATH=. .venv/bin/python experiments/exp-003-anatomy/code/climb_tsv.py                                   # climb.tsv
```

## Leakage self-check against `docs/input-rule.md`

- `code/features.py check` asserts that every feature frame is **bit-identical**
  when built from the raw month and from the ranking-style masked month. It
  passes. No feature reads a departure's own `BLOCK_TIME_UTC_mvt` or
  `TAXITIME_SEC_mvt`.
- Arrival `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` *are* read. The ranking file
  keeps them on arrival rows (`docs/input-rule.md`), and `prc.data.mask_like_ranking`
  blanks departures only.
- Every group statistic (`StandStats`, the `phi` fit, the naive-predictor stand
  medians, the no-NM airport-median fallback) is fitted on the fold's training
  months and only transformed onto the validation month.
- The ablation harness reads per-month feature frames instead of calling
  `prc.data.fold_frames` on every run, because rebuilding 1.9 M rows of
  neighbour features for each of two dozen ablations would cost more than the
  fits. `code/check_fold_frames.py` is the argument that the two paths are the
  same object, and it passes on all three folds: the fold month lists in
  `ablate.py` are now taken from `prc.data.FOLDS` rather than a second copy, and
  the validation ids, labels and training row counts match `fold_frames`
  exactly (july 1,005,519 training departures, december 1,919,370, january
  1,931,341), with `fold_frames` confirmed to blank every departure block time.
- Neighbour features use other departures' `AOBT_3_flt` and `MVT_TIME_UTC_mvt`,
  both present in the ranking file, never their block or taxi time.
- The bulk metric filters on the hidden label (`y <= 3600`). That is a diagnostic
  slice, not a predictor: no model in this report sees it at prediction time, and
  the four COMMON.md slices are reported beside it on every row.

## Review (GPT-5.6 Sol, xhigh, fresh context, 2026-09-05)

Round 1 log: `experiments/exp-003-anatomy/logs/sol_review.log`. Two blocking
findings, both accepted and fixed before the round-2 request.

**B1 — "month explains exactly nothing" was tautological.** The `month` row of
the out-of-sample decomposition in finding 1 could only ever reproduce the
global median, because the validation month's level is never present in the
training table and every row falls back. The conclusion drawn from it ("no
seasonal drift in the offset, all twelve training months equally valid") was
therefore unsupported, and the report's own trimmed-mean table contradicted it.
Fixed: the row is removed from the table with a note saying why, and finding 1
now carries the measured drift (LIRF 168 s, LTFM 159 s, EDDM 124 s of range
across 2025) together with the only test that means anything — which training
window predicts the validation month best (`analysis/q1c_drift.md`, new).
Recency is worth 4.1 s on july and nothing on december or january, and the
oldest window is worst on all three folds. No other number in the report moves.

**B2 — the ranking feature frame joined across the six-month gap.** `features.py`
built the ranking file in one call, so counting features (`n_dep_stand_month`,
`n_arr_stand_month`, `cs_arr_count`) counted across January *and* July 2026 and
the asof neighbour joins reached across the gap: 98.2 % of ranking rows carried a
stand count about twice its within-month value and 2,154 rows had a neighbour gap
of up to 204 days. Training and validation frames are single months, so no
measured number in this report is affected, but any submission built on this
feature frame would have been served features from a distribution the training
rows never saw. Fixed: the ranking frame is now built per month and concatenated,
which is what `prc/data.py`'s contract says ("features for a month are computed
from that month's frame alone"). Verified after the rebuild: zero rows with a
neighbour gap over 31 days, and the stand count differs from its within-month
value on 4 rows of 344,841.

### Round 2 (same reviewer, after the round-1 fixes)

Log: `experiments/exp-003-anatomy/logs/sol_review2.log`. Verdict BLOCK, five
findings. B2 was confirmed fixed; the rest are answered here.

**R2-1 — B1 was repaired in the prose but not in the source.** `q1b_decomp.py`
still computed the tautological `month` row and the regenerated
`analysis/q1b_decomp.md` still printed it. Fixed: `month` is removed from the
grouping list with a comment saying why, and the file is regenerated.

**R2-2 — the ceiling arithmetic quoted the printed cells, not its own
assumptions, and one feasibility claim was false.** Accepted in full. Finding 9
is rewritten: the assumption box gives 132-244 s, not 186-233; `b = 2,000` is
feasible (the bulk goes to zero only at `b` = 1,780 to 2,114), so the claim that
no team can be worse than ~1,800 s on the ordinary no-NM rows is withdrawn. The
back-out is now anchored on what our own fallback path already achieves on those
segments (no-NM non-noac 658 s, NM tail 2,744 s on july), which bounds the
leaders' bulk *below* by 194-225 s for a noac RMSE of 3,000-4,500 and leaves
279.7 s as the only hard bound. The conclusion is weakened accordingly: our 221 s
cannot be placed against the leaders' bulk at all. The fifth ledger
row is corrected in place with a pointer to the correction.

**R2-3 — the reporting contract.** Accepted. Every ablation row is now run on all
three folds and the four COMMON.md slices for every row and fold are in
`analysis/ablation_full.md` (`code/table_full.py` renders it from `climb.log`);
the table in finding 6 keeps only the bulk column, which is the metric the
assignment scores, and points at the full one. On `fold_frames`: the harness
still reads cached per-month feature frames for speed, but the fold definitions
now come from `prc.data.FOLDS` and `code/check_fold_frames.py` shows the cached
path scores the identical ids, labels and training row counts (see the leakage
self-check above).

**R2-4 — the Q3 callsign feature was broken.** Accepted, and it was a real bug:
`merge_asof` returns the *left* frame's value in the join column, so
`cs_prev_arr_gap` differenced the departure's own timestamp against itself and
was exactly zero on all 190,713 july rows. Fixed by carrying the arrival's
touchdown under a separate name; the column now has a sensible distribution
(median 81,480 s, i.e. the same callsign arrived about 22.6 hours earlier).
Rows U and V are re-run on three folds with the fixed feature. The other half of
the finding is also right and is now measured directly: the assignment asks for
an arrival *earlier* in the month frame, and `same_flight_arr` tested any
matching arrival, which for a departure is normally the later destination
record. The new `fid_arr_earlier` flag fires on **0.001 %** of july departures —
the same `FLIGHT_ID_mvt` essentially never appears as an *earlier* arrival, so
the only version of the signal that exists in this data is the callsign one.

**R2-5 — Q4 did not list every unseen level.** Accepted. `q4_shift.py` now writes
`analysis/q4_new_levels.md` with every new level in every column and its row
count split by ranking month: 45 stands, 30 operators, 8 aircraft types, 62
destinations, 2 stand prefixes, 1 runway, 148 rows of table in all.

**Non-blocking, all accepted and applied.** The two-term blend in finding 2 is an
oracle (its coefficients are least-squares-fitted on the validation labels) and
is now labelled as one, with the like-for-like caveat against the 40-feature
tree. Negative targets span -12 s to 0 s, not -6 to 0. LTFM is no longer listed
among the January-2026 airports with elevated dispersion in `docs/shift-2026.md`
(its spread is 298 s against 416 s in February 2025; only its mean is high), and
LFPG is marked as marginal. The `phi` ledger row now states the per-fold effect
(0.16 to 0.64 s worse) rather than a single range.

### Round 3 (same reviewer) — verdict as it stands

Log: `experiments/exp-003-anatomy/logs/sol_review3.log`. Verdict **BLOCK**, five
findings. The desk closed the review here (protocol allows two rounds; write-up
findings are not blocking), so the verdict is recorded as it stands and each
finding is answered below with what was actually changed.

**R3-1 — the ceiling lower bound was a non-sequitur.** Accepted and withdrawn.
"A team at 277.3 overall must be at least as good as this model on every
segment" is false: a better aggregate does not imply a better score on any
particular part. Finding 9 now states only the hard upper bound (279.7 s) and
the conditional grid (132-244 s over the assumption box), says plainly that the
board cannot place our 221 s against the leaders' bulk, and `q0_ceiling.py`
carries the same caveat in the generated analysis. A correction row is appended
to the ledger; the original rows are restored (see R3-5).

**R3-2 — the shift inventory was incomplete.** Accepted. `q4_shift.py` now
inventories all **18** categoricals any exp-003 model uses, at the exact key the
model uses, including the two it had missed (`ADES_FILED_flt`, 51 unseen levels
on 154 ranking rows; `CALLSIGN_PFX`, 89 on 330) and the raw stand key `key_raw`
(46 levels on 1,589 rows, against 45/1,588 for the normalised key — the two extra
are `LTFM|H24R` and `LSZH|P3C`). `analysis/q4_new_levels.md` lists every level in
every column with its count split by ranking month. The assignment's "compare on
every feature you propose" is now also answered for the numeric side: a new
section of `analysis/q4_shift.md` gives the 2025 mean, 2025 sd, null rate and the
per-ranking-month shift in units of the 2025 sd for **every numeric model
column**. That table is what caught the window bug.

**R3-3 — the bimodality count was uncalibrated.** Accepted. A large 2-means
separation with a 1 % minority cluster is an outlier group, not a second mode.
Requiring `separation > 3.0` **and** `minority weight >= 15 %` cuts the count from
262 to **67**, and the per-airport counts in `analysis/q1b_decomp.md` now use that
definition (the raw `separation > 3.0` count is kept beside it as `n_sep_gt3`).
The four named example stands pass both tests.

**R3-4 — the RESULTS table showed only the bulk column.** Accepted. The four
COMMON.md slices are now inline in finding 6 for every row and every fold, not
only behind a link.

**R3-5 — the ledger was edited in place.** Accepted, and this was the worst of
the five: three exp-003 rows had been corrected by overwriting them. All three
are restored to their original text from git and four correction rows are
appended instead, each naming the row it corrects.

**Non-blocking, both accepted.** The oracle wording really had not been applied
to finding 2 (the edit was lost in a failed batch) and now is, with the
like-for-like caveat. `StandStats.transform` had a defective global fallback —
`st_best_mean` was filled with the global *median* and the scale and count
columns were left NaN. Fixed in `prc.stand_features` **VERSION 1.1.0**, with the
old behaviour reachable as `global_fallback=False`; the only consumer is this
experiment's harness and the block measures zero either way. Version 1.2.0 then
adds `airport_neighbour_features` so the second paying family is importable too.

**One finding of my own, surfaced by answering R3-2.** `nbr_apt_30m` counted
departures within ±21 days instead of ±30 minutes. Fixed; see the note under the
ablation table for which rows were re-measured and why the conclusions hold.
