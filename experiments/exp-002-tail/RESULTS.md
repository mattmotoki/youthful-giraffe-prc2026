# exp-002-tail — RESULTS

exp-id `exp-002` | worker `tail` | tier hard | reviewer: pending
Data: `prc.data.data_dir()`, live ranking SHA-256 `9db75e3a…`. Code in
`experiments/exp-002-tail/work/`, feature module `prc/tail_features.py`.

Segment definitions used everywhere below (departures at the 10 ranking
airports):

- **noac**: `AIRCRAFT_TYPE_mvt` is null. 0.072 % of training departures.
- **no-NM**: `LOBT_flt` null (all `_flt` columns null). 1.08 % of training
  departures, 1.5 % of ranking departures. `noac` is a strict subset of no-NM.
- **tail**: `TAXITIME_SEC_mvt > 3600`.
- **placeholder**: the block time is the schedule rather than an actual
  off-block. Two thresholds appear below and both are stated where used:
  the *descriptive* rate in findings 1–3 uses `|BLOCK − SCHED| ≤ 60 s`
  (47.9 % of noac rows); the *classifier label* in the model uses
  `|BLOCK − SCHED| < 300 s` (48.4 %), which is the tolerance in
  `noac_model.py` and `prc/tail_model.py`. Because the label admits offsets up
  to 300 s, the mixture is
  `E[y] = mvt_sched − (1−p)·E[offset | not placeholder]` with the placeholder
  branch's own offset taken as 0; that approximation is worth at most 300 s on
  48 % of 388 rows and is inside the noise of the 4,290 s LOMO figure.
  Only observable in training; it is what the detector has to infer.

---

## Findings, ranked

### 1. One null column identifies almost the whole extreme tail: `AIRCRAFT_TYPE_mvt`

Over the twelve 2025 training months (2,085,047 departures) `AIRCRAFT_TYPE_mvt`
is null on **1,507 rows (0.072 %)**. Those rows are:

| statistic | noac rows | all other rows |
|---|---|---|
| n | 1,507 | 2,083,540 |
| median taxi | 2,643 s | 911 s |
| mean / std taxi | 6,451 / 11,170 s | 987 / 432 s |
| P(taxi > 3600 s) | **49.4 %** | 0.16 % |
| P(taxi > 15000 s) | **6.97 %** | 0.0003 % |

Coverage of the extreme rows, from the other direction:

| threshold | n rows | share that are noac |
|---|---|---|
| y > 3600 s | 4,126 | 18.1 % |
| y > 8000 s | 439 | 80.9 % |
| y > 15000 s | 112 | **93.8 %** |

The flag is 98.7 % concentrated at LIRF (1,487 of 1,507; LIRF's own rate is
0.93 %), with 7 at LTFM, 7 at LSZH, 5 at LFPG, 1 at EHAM and none at the other
five airports. Every noac row is also a no-NM row.

**In the ranking file the flag is directly observable: 388 noac departures
(108 in January 2026, 280 in July 2026), 383 of them at LIRF.** Ranking rate
0.113 % against 0.072 % in training — the same phenomenon, slightly denser.

### 2. The mechanism: a placeholder block time equal to the schedule

Within the noac rows, **47.9 % have `BLOCK_TIME_UTC_mvt` within 60 s of
`SCHED_TIME_UTC_mvt`** (base rate over all departures: 9.6 %). On those rows
the target is not a taxi duration at all — it is `MVT_TIME − SCHED_TIME`, an
artefact of the airport system never overwriting a scheduled off-block with an
actual one. Correlation of `y` with `MVT_TIME − SCHED_TIME` across all noac
rows is **0.896**, and 48.4 % of them satisfy `|y − (MVT−SCHED)| < 300 s`.

The placeholder probability is strongly ordered by `mvt_sched = MVT_TIME −
SCHED_TIME`, which **is observable in the ranking file**:

| `mvt_sched` band | n | P(placeholder) | mean y | RMSE of the band mean |
|---|---|---|---|---|
| ≤ 0 | 19 | 0.05 | 654 | 426 |
| 0 – 1800 s | 66 | 0.47 | 3,780 | 5,425 |
| 1800 – 3600 s | 38 | 0.40 | 3,757 | 2,740 |
| 1 – 2 h | 754 | 0.37 | 2,903 | 2,448 |
| 2 – 4 h | 503 | 0.59 | 6,345 | 5,625 |
| 4 – 8 h | 77 | **0.81** | 15,284 | 7,642 |
| > 8 h | 50 | **0.78** | 55,210 | 22,346 |

Two secondary discriminators inside the noac set: `len(FLIGHT_mvt) ∈ {4, 8}`
gives P(placeholder) 0.73–0.74 against 0.43 for lengths 6–7; and the rate is
*inversely* related to how many noac rows LIRF produced that day (0.69 on days
with ≤ 2 noac rows, 0.24 on days with > 20). Two distinct failure modes are
mixed: an isolated record that was never enriched (placeholder likely), and a
bulk feed outage that stripped the aircraft type but left real block times.

### 3. This one set of rows sets a floor on the leaderboard, and explains the board

Using the training noac taxi distribution and the 388 noac rows in the ranking
file (344,841 departures):

| treatment of the 388 noac rows | their RMSE | contribution to total ranking RMSE |
|---|---|---|
| predicted at the bulk mean (≈1000 s) | 12,430 s | **417 s** |
| predicted at the `mvt_sched`-band conditional mean | 5,881 s | **197 s** |

A model that ignores these 388 rows **cannot score better than ≈ 417 s no
matter how good the other 344,453 predictions are**. That is the shape of the
public board: a wall of teams between 302 s and 499 s, and three teams at
277–279 s. Working backwards from 277.3 s:

| noac RMSE achieved | its contribution | RMSE the other 344,453 rows must reach |
|---|---|---|
| 12,400 s (ignored) | 416 s | impossible |
| 6,000 s | 201 s | 191 s |
| 4,500 s | 151 s | 233 s |
| 3,600 s | 121 s | 250 s |
| 3,000 s | 101 s | 258 s |

So the leaders are getting noac rows to roughly 3,000–4,500 s and the bulk to
roughly 230–260 s. Neither half alone reaches 277. **The desk's arithmetic that
"no team reaches 277 while predicting the no-NM rows at their group mean" is
confirmed, and sharpened: it is not the 1.5 % no-NM rows that bind, it is the
0.11 % noac rows inside them.**

### 4. The NM-joined tail: the desk's AOBT-vs-LOBT hypothesis holds

For the 503 July-2025 tail rows that *do* have an NM join, the median relations
are: `BLOCK − LOBT = 0 s`, `AOBT_3 − LOBT = 3,300 s`, `MVT − AOBT_3 = 1,016 s`,
`BLOCK − SCHED = 177 s`. The aircraft went off-block at its last filed
off-block time; Network Manager recorded an actual off-block about an hour
later; the take-off then followed a normal 17-minute taxi *from the NM time*.
This is the leave-and-return / remote-hold signature the desk hypothesised, and
it is visible without the label:

| screen | n (July) | P(taxi > 3600) | lift over NM base rate |
|---|---|---|---|
| `AOBT_3 − LOBT > 0` | 141,024 | 0.34 % | 1.3× |
| `AOBT_3 − LOBT > 1800 s` | 14,509 | 2.65 % | 9.8× |
| `AOBT_3 − LOBT > 3600 s` | 816 | **26.7 %** | 99× |
| `AOBT_3 − LOBT > 7200 s` | 25 | **52.0 %** | 193× |

60 % of the rows passing the 3600 s screen have `LOBT` within 60 s of `SCHED`,
as the desk predicted. The consequence for prediction is direct — on those 503
rows:

| predictor | RMSE | median error |
|---|---|---|
| `MVT − AOBT_3` (the naive predictor) | 3,960 s | +3,476 s |
| `MVT − EOBT_1` | 2,119 s | +303 s |
| `MVT − SCHED` | 2,266 s | −177 s |
| **`MVT − LOBT`** | **1,562 s** | **0 s** |

`LOBT`, not `AOBT_3`, is the right block-time proxy once NM's off-block has
drifted away from the filed one. The NM tail concentrates at LIRF (1.78 % of
its NM departures) and EGLL (0.75 %); LEMD and LEBL produce none.

### 5. The no-NM rows minus noac are an ordinary population, not a tail

The other 20,963 no-NM training rows (all twelve months, no-NM and not noac):
median taxi 912 s, 99th percentile 4,668 s, P(y > 3600) = 1.6 %, 13.97 % VFR,
median `BLOCK − SCHED` 4,137 s and median `MVT − SCHED` 5,216 s. Per airport (median taxi / P(tail)): EHAM 539 s /
0.3 %, LSZH 673 s / 0.2 %, EDDM 891 s / 1.1 %, EDDF 958 s / 0.2 %, LTFM 1,010 s
/ 6.1 %, LFPG 1,023 s / 1.2 %, LEBL 1,039 s / 0.0 %, LEMD 1,096 s / 0.0 %,
EGLL 1,384 s / 3.2 %.

Why they lack the join: they are the *delayed* flights. The figures below are
**July 2025, all no-NM rows including noac** (3,938 rows; an earlier draft
introduced the all-year non-noac population above and then switched to these
without saying so — caught by the Sol review): `BLOCK − SCHED` median 4,557 s
against 658 s for NM rows, `MVT − SCHED` median 5,814 s against 1,684 s, 7.44 %
VFR against 0 % of NM rows. Excluding noac, the same July slice (3,599 rows) is
median taxi 984 s, `MVT − SCHED` median 5,745 s, 8.14 % VFR, P(y > 3600) =
0.94 %. Operator prefixes with the highest no-NM rate are
non-EU or charter carriers (FPO 8.7 %, DAH 8.2 %, AIC 7.9 %, MSR 7.7 %, CPA
7.6 %); the lowest are the home carriers' shuttle codes (SHT, LH0, DLA, BTI,
all ≤ 0.3 %). The join looks like a callsign+ADEP+EOBT-window match that fails
when the flight departs far from its filed off-block or when the operator is
outside the NM data set. Controlling for **airport and take-off hour together** (177 cells covering all
3,599 July no-NM non-noac rows), the no-NM median taxi is only **38 s** above
the NM median in the same cell and the mean **61 s** above — smaller than the
airport-only difference, so part of what looks like a no-NM effect is just when
these flights depart. Per airport, within-hour median difference: EHAM −177 s,
LSZH −86 s, LTFM +32 s, LFPG +70 s, EDDM +74 s, LEBL +106 s, EGLL +124 s, EDDF
+134 s, LEMD +143 s (LIRF has one such row and is not comparable). The whole
difficulty is therefore the missing `AOBT_3`, not a different taxi process.

**No-NM counts in the ranking months** (departures only, live file): January
2026 **2,453 of 152,719 = 1.61 %**, July 2026 **2,837 of 192,122 = 1.48 %**.
The 2025 monthly rate runs 0.62 % (March) to 2.07 % (July), mean 0.99 %, so the
ranking months sit *above* the 2025 average at every airport: EHAM 3.16 % vs
1.65 %, LSZH 2.85 % vs 1.70 %, LFPG 1.84 % vs 1.58 %, EGLL 0.62 % vs 0.59 %.
The no-NM head therefore carries more weight in the ranking months than in most
folds. (An earlier draft of this line said 1.45 %/1.35 %; that came from a
denominator that still contained arrival rows.)

Their movement-column profile does match 2025: flight-rule mix 92.1 % IFR /
7.9 % VFR against 86.9 / 13.1 in 2025, `len(FLIGHT_mvt)` distribution within
2.4 points on every level, take-off-hour total-variation distance 0.048, and 43
of 2,453+2,837 rows on a stand level unseen in 2025. The one shift is
`MVT - SCHED`: median 5,830 s in the ranking months against 5,283 s in 2025,
with a heavier upper tail (p95 15,616 s vs 11,400 s) — the ranking no-NM rows
are, if anything, *more* delayed than their 2025 counterparts.

### 6. Where the squared error actually sits (July 2025 fold, naive predictor)

Naive = `MVT_TIME − AOBT_3_flt`, airport-median taxi where `AOBT_3` is null,
**the median taken over the fold's training months (January–June 2025)**.
190,713 departures, RMSE 720.7 s. (An earlier draft of this table fitted the
airport medians on July's own labels, which violates the training-only rule in
`docs/input-rule.md`; the corrected figures are below and differ by under 0.3 %:
720.7 rather than 719.2 overall, 10,011 rather than 9,982 on the tail, 3,856
rather than 3,842 on the no-NM rows. Caught by the Sol review.)

| segment | n | % of rows | % of squared error | RMSE |
|---|---|---|---|---|
| all | 190,713 | 100 % | 100 % | 720.7 |
| no-NM | 3,938 | 2.06 % | 58.9 % | 3,856 |
| y > 3600 | 656 | 0.34 % | 66.3 % | 10,011 |
| y > 3600 **and** NM-joined | 503 | 0.26 % | 8.0 % | 3,960 |
| y > 3600 **and** no-NM | 153 | 0.08 % | **58.3 %** | 19,383 |

Across all twelve months the picture is stable: rows above 3600 s are 0.08–0.4 %
of departures and carry 24–66 % of the squared error, and the 2–18 rows per
month above 20,000 s alone carry 4–52 %. Every one of those months' extreme
rows is at LIRF.

### 7. Stand occupancy is available in the ranking months (bounds, not yet a model)

Checked because it is the natural physical constraint on the hidden block time.
The ranking file contains **344,693 arrival rows against 344,841 departures**,
and arrivals keep `BLOCK_TIME_UTC_mvt` (on-block), `TAXITIME_SEC_mvt`,
`STAND_mvt` and 99.3 % of their NM columns. So a departure's stand history is
fully reconstructible in the ranking months.

First measurement on July 2025 (last arrival on-block at the same
airport+stand at or before the departure's take-off, `gap1 = MVT − that
on-block`): coverage 99.4 %, 83.4 % within 6 h.

- `y < gap1` (the stand was *not* re-occupied between block-off and take-off,
  so `gap1` is a valid upper bound on taxi) holds for **88.0 %** of all
  departures but only **51.8 %** of tail rows.
- Equivalently: the stand was re-occupied between block-off and take-off for
  11.3 % of bulk rows and **47.6 % of tail rows** — so roughly half the tail
  carries a hard lower bound from an arrival that parked at the stand after the
  aircraft had left.

A FIFO turnaround pairing (departures in take-off order consuming arrivals in
on-block order) was tried and **failed**: the pointer desynchronises wherever
arrivals and departures are unbalanced (towing, month boundaries, remote-stand
naming), giving a valid upper bound 98.9 % of the time but with a median slack
of 14,777 s, and a lower bound valid only 54 % of the time. Diagnosis: the
pairing has no resynchronisation mechanism, so one missing counterpart
corrupts every later departure at that stand. Repair to try: local pairing with
a plausibility window on the turnaround instead of a global pointer, and a
stand-name normalisation for the 15 EGLL bases used both plain and with an
`L`/`R` suffix. Carried into the modelling section as `gap1..gap3`,
`gap_next` features rather than as hard bounds.

---

## Anatomy answers in one line each

- **Q1 (stand group and runway, by regime).** Rates are *within regime* — the
  denominator is that regime's departures in the cell, not all departures
  (an earlier draft divided by all departures and understated the no-NM rates
  by a factor of about 40; caught by the Sol review). Cells with at least 100
  rows, July 2025:

  | regime | by stand group | by runway |
  |---|---|---|
  | NM-joined (186,775 rows, 503 tail) | LIRF 9 **7.1 %** (17/241), LIRF 8 5.6 % (75/1,345), LIRF 7 3.4 % (43/1,274), LIRF 6 2.0 % (27/1,367) | LIRF 34L **8.7 %** (11/126), LIRF 16R 3.2 % (38/1,180), LIRF 25 1.6 % (222/13,512), EGLL 27R 0.9 % (81/9,002) |
  | no-NM (3,938 rows, 153 tail) | LIRF 3 **33.3 %** (41/123), EGLL 5 9.9 % (11/111), LTFM F 0.8 % (1/132) | LIRF 25 **36.4 %** (108/297), EGLL 27R 6.2 % (15/242), LTFM 35L 4.2 % (7/165) |

  The two regimes do **not** share a hot spot: the NM tail concentrates on
  LIRF's 900- and 800-series stands and runways 34L/16R, the no-NM tail on
  LIRF's 300-series stands and runway 25. (An earlier draft claimed they shared
  the same geography; that conclusion came from the wrong denominator and does
  not survive the correction.) What they do share is the airport.
- **Q1 (extreme rows with NM join):** LIRF and EGLL, mainline and low-cost
  scheduled flights, midday peak (10:00–16:00 carries 55 % of them), no
  operator or aircraft-type signature beyond airport mix. The signature is
  temporal: block-off at `LOBT`, NM off-block ~1 h later, normal taxi after
  that. `AOBT_3 − LOBT > 3600 s` screens them at 99× lift and `MVT − LOBT`
  predicts them at RMSE 1,562 s.
- **Q1 (extreme rows without NM join):** LIRF, `AIRCRAFT_TYPE_mvt` null,
  block time is the schedule. `MVT − SCHED` is the target on ~half of them.
- **Q1 (repeated identifiers): rare but real — an earlier draft of this line
  said "no" and was wrong** (caught by the Sol review; the first check only
  intersected identifier sets and never tested the block-to-take-off interval).
  Tested properly on July 2025, counting arrivals whose *landing* falls inside
  a departure's `[block, take-off]` window:

  | identifier | departures with such an arrival | of which tail rows | share of the 656 tail rows |
  |---|---|---|---|
  | `FLIGHT_mvt` | 9 | **7** | 1.1 % |
  | `CALLSIGN_flt` | 2 | 2 | 0.3 % |
  | `FLIGHT_ID_mvt` | 1 | 1 | 0.2 % |

  Examples: `AFR88WR` at LFPG (taxi 5,876 s) has an arrival carrying the same
  `FLIGHT_ID_mvt` landing four minutes before its recorded take-off; `NOS163`
  and `NOS431` at LIRF (11,151 s and 9,971 s) the same by `FLIGHT_mvt`. So the
  return-to-stand / re-attempt mechanism the assignment asked about **does
  occur**, and when it fires it is highly diagnostic (7 of 9 are tail rows) —
  but it accounts for about 1 % of the tail, so by the p-rule of finding 15 it
  cannot be worth much RMSE. Duplicate departure `FLIGHT_ID_mvt` values: exactly
  one pair in July (`DAL81` at LFPG, stands K52 and K70, take-offs 21 s apart,
  both with ordinary taxi times) — a duplicated stand record, not an aborted
  attempt. `MVT_ID_mvt` is unique. `FLIGHT_ID_mvt` is otherwise shared between a
  departure and its matching arrival *at the other network airport* (26,513 July
  pairs).
- **Q2 (who the no-NM rows are):** delayed flights and non-EU/charter
  operators, 7.5 % VFR, whose taxi times are ordinary; plus the noac subset,
  which is a different phenomenon entirely.



---

## Modelling (assignment questions 3 and 4)

### Target parameterisation

Everything below predicts an **offset from an anchor**, not the taxi time
directly:

```
anchor = AOBT_3_flt  when the NM join exists, else SCHED_TIME_UTC_mvt
base   = MVT_TIME_UTC_mvt - anchor          (known for every ranking row)
offset = BLOCK_TIME_UTC_mvt - anchor        (the training target)
y_hat  = base - offset_hat
```

Because `base` is known exactly, squared error on `offset` equals squared error
on `y`, so this is a reparameterisation, not a different loss. It matters
because `offset` is bounded and roughly symmetric where `y` is not: on NM rows
`BLOCK - AOBT_3` has std 418 s against 502 s for `y`, and on the noac rows
`BLOCK - SCHED` has no values beyond ±30,000 s where `y` reaches 131,167 s. A
gradient-boosted tree cannot extrapolate to a target value larger than anything
in its training leaves; in the offset space it does not have to, because the
large part of `y` is carried by the known `base`.

### The fixed bulk model

LightGBM (`num_leaves` 63, `lr` 0.08, 500 trees, `min_child_samples` 60), the
feature set the assignment specifies: airport, airport|stand, airport|runway,
hour, day of week, minute of day, and the own-flight deltas `MVT-SCHED`,
`MVT-LOBT`, `MVT-IOBT`, `MVT-EOBT_1`, `MVT-AOBT_3`, `LOBT-SCHED`,
`EOBT_1-SCHED`, `LOBT-IOBT`, `AOBT_3-SCHED`. Trained on the fold's training
months only, evaluated on the ranking-masked validation month. This model is
held fixed for every line after it.

### The mechanisms tested

| line | mechanism |
|---|---|
| it0 | naive `MVT - AOBT_3`, airport median where the NM join is missing |
| it1 | fixed bulk features, target `y` |
| it2 | fixed bulk features, target `offset` — **the fixed bulk model** |
| it3 | + tail signals (`noac`, `nonm`, `vfr`, `flight_len`, `aobt_lobt`, per-day noac/no-NM counts) |
| it4 | + stand-occupancy signals (arrival gaps, arrivals between SCHED and take-off, neighbouring departures) |
| it5 | two-stage: `P(y>3600)` classifier × conditional regressors, as an expectation and as a hard switch |
| it6 | Huber bulk; two-stage with a shrunk (median-quantile) tail head |
| it7 | a separate no-NM model (no `_flt` columns at all) spliced in, against the missingness flags of it4 |
| it8 | the noac sub-model spliced in, six variants plus an oracle |

---

# Modelling (assignment questions 3 and 4)

All numbers are fold RMSE in seconds on the full validation month (10 airports,
every departure), training on the fold's training months only and computing
validation features from the ranking-masked frame alone. Raw table:
`experiments/exp-002-tail/work/metrics_*.csv`. Climb log: `climb.log`.

## 8. The target parameterisation is the first decision, and it is regime-specific

Everything predicts an **offset from an anchor** rather than the taxi time:

```
base   = MVT_TIME_UTC_mvt - anchor          (known for every ranking row)
offset = BLOCK_TIME_UTC_mvt - anchor        (the training target)
y_hat  = base - offset_hat
```

Squared error in `offset` equals squared error in `y` algebraically, so this is
a reparameterisation and not a different loss. It is not, however, an equally
*learnable* target, and that turned out to be the single largest modelling
effect in the experiment.

With one global anchor (`AOBT_3` when the NM join exists, `SCHED` otherwise),
the offset target cut the NM-tail RMSE from 4,058 s to 2,293 s on July but
pushed the no-NM rows from 995 s to 2,933 s — **worse than the naive airport
median of 658 s** — and cost 174 s of total RMSE on December (307.6 → 482.1).

Diagnosis: for an NM row the offset is `BLOCK − AOBT_3`, small (std 418 s), so
the tree learns a small correction. For a no-NM row the anchor is the schedule
and the offset is `mvt_sched − y`: to predict it the tree must **reproduce a
continuous input feature**, which a piecewise-constant model can only do in
steps. The algebra says the two targets are equivalent; the function class says
they are not.

Repair: pick the anchor per regime so the offset is always small, and where no
such anchor exists predict the taxi time directly.

| regime | rows (July) | anchor | target |
|---|---|---|---|
| NM-joined | 186,775 | `AOBT_3_flt` | offset |
| noac | 339 | `SCHED_TIME_UTC_mvt` | offset (half of them sit at exactly 0) |
| other no-NM | 3,599 | — | the taxi time itself |

## 9. The climb

| line | mechanism changed | july | december | january |
|---|---|---|---|---|
| it0 | naive `MVT − AOBT_3`, airport median where no NM join | 720.7 | 488.2 | 559.5 |
| it1 | fixed bulk features, target `y` | 555.2 | 307.6 | 397.0 |
| it2 | same features, target offset, one global anchor | 540.5 | 482.1 | 432.8 |
| it3 | + tail signals (`noac`, `aobt_lobt`, day counts, `flight_len`) | 539.8 | 483.8 | 439.7 |
| it4 | + stand-occupancy signals | 506.9 | 443.3 | 413.1 |
| it5 | two-stage `P(y>3600)` × conditional regressors, as an expectation | 590.6 | 458.3 | 422.7 |
| it5 | the same, as a hard switch | 599.1 | 458.8 | 424.1 |
| it6 | Huber bulk (α = 300 s) | 565.1 | 538.8 | 464.1 |
| it6 | two-stage with a median-quantile (shrunk) tail head | 558.8 | 459.6 | 419.8 |
| it7 | a separate no-NM model spliced over it4 | 450.5 | 358.7 | 385.9 |
| it8 | noac sub-model (mixture expectation) spliced over it4 | 502.7 | 440.6 | 407.1 |
| it9 | **segment by regime**, no-NM head trained on no-NM rows | 362.2 | 234.6 | 357.5 |
| it10 | no-NM head trained on **all** departures, `_flt` columns dropped | **353.3** | **229.7** | **356.3** |
| — | it10 with the noac rows handed to the ordinary head instead (ablation) | 603.3 | 370.4 | 461.9 |
| — | it10 with a noac oracle | 284.1 | 222.4 | 345.1 |

Read the last three rows together, and read the claim precisely: the ablation is
measured on the **339 noac rows of the July 2025 validation fold**, where routing
them through the sub-model instead of the ordinary head moves the fold from
603.3 s to 353.3 s — a **250 s ablation gain on that fold** (141 s on December,
106 s on January). A perfect noac predictor would be worth another 69 s of July
on top of that. The 388 noac rows in the ranking file are the same *kind* of row
— the flag is directly observable there — but their hidden block times are not
available locally, so **no claim is made here about the realised gain on the
leaderboard**; the 47.9 % placeholder rate is a pooled 2025 training rate, not an
observed ranking rate.


### The climb with all four required metrics

Each cell is `RMSE / RMSE ≤3600 / RMSE >3600 / RMSE no-NM`, in seconds
(`assignments/COMMON.md`). Source: `work/metrics_*.csv`.

| line | july | december | january |
|---|---|---|---|
| it0 | 720.7 / 418.7 / 10011 / 3856 | 488.2 / 363.0 / 8231 / 3160 | 559.5 / 361.2 / 9299 / 4295 |
| it1 | 555.2 / 351.7 / 7334 / 3091 | 307.6 / 224.8 / 5296 / 2120 | 397.0 / 229.1 / 7055 / 3364 |
| it2 | 540.5 / 507.2 / 3228 / 3204 | 482.1 / 471.6 / 2562 / 4268 | 432.8 / 322.4 / 6290 / 3770 |
| it3 | 539.8 / 509.1 / 3104 / 3208 | 483.8 / 474.4 / 2430 / 4281 | 439.7 / 333.7 / 6236 / 3862 |
| it4 | 506.9 / 475.1 / 3050 / 2969 | 443.3 / 432.9 / 2443 / 3828 | 413.1 / 288.2 / 6442 / 3556 |
| it5a | 590.6 / 510.0 / 5104 / 3429 | 458.3 / 432.2 / 3860 / 3960 | 422.7 / 300.2 / 6478 / 3640 |
| it5b | 599.1 / 516.0 / 5215 / 3478 | 458.8 / 432.1 / 3909 / 3960 | 424.1 / 303.7 / 6443 / 3641 |
| it6a | 565.1 / 540.1 / 2881 / 3419 | 538.8 / 531.9 / 2223 / 4930 | 464.1 / 360.2 / 6373 / 4203 |
| it6b | 558.8 / 476.7 / 4994 / 3295 | 459.6 / 434.2 / 3823 / 3983 | 419.8 / 294.8 / 6507 / 3618 |
| it7 | 450.5 / 390.5 / 3852 / 2490 | 358.7 / 347.4 / 2274 / 2800 | 385.9 / 254.0 / 6323 / 3211 |
| it8 | 502.7 / 476.1 / 2790 / 2934 | 440.6 / 433.0 / 2100 / 3797 | 407.1 / 289.2 / 6239 / 3481 |
| it9 | 362.2 / 324.9 / 2749 / 1693 | 234.6 / 220.3 / 2045 / 937 | 357.5 / 222.4 / 6092 / 2921 |
| it10 | 353.3 / 314.2 / 2772 / 1600 | 229.7 / 215.2 / 2036 / 805 | 356.3 / 221.1 / 6082 / 2905 |
| it11 | 350.0 / 309.8 / 2796 / 1580 | 226.7 / 212.5 / 2004 / 758 | 353.7 / 218.3 / 6057 / 2893 |
| it12 | 352.3 / 312.0 / 2808 / 1607 | 226.7 / 212.4 / 2008 / 780 | 352.8 / 216.7 / 6059 / 2891 |

The delivered model (it16, findings 16–17) is 327.3 / 284.5 / 2,775 / 1,381 on
july, 223.1 / 208.0 / 2,042 / 763 on december and 350.6 / 214.8 / 6,029 / 2,887
on january.

## 10. Question 4 — the RMSE-optimal treatment of the rows we cannot explain

For a row with detected tail probability `p`, a tail value `m` (here
`mvt_sched`, the placeholder value) and a bulk value `b`, RMSE is minimised by
the expectation `p·m + (1−p)·b`, not by a switch on `argmax p`. Measured three
ways, the expectation wins every time:

| comparison | july | december | january |
|---|---|---|---|
| noac mixture **expectation** | **362.2** | **234.6** | 357.5 |
| noac mixture **hard switch** | 366.9 | 234.9 | 361.5 |
| two-stage tail **expectation** (it5) | **590.6** | **458.3** | **422.7** |
| two-stage tail **hard switch** (it5) | 599.1 | 458.8 | 424.1 |

On the noac rows alone (12 months, leave-one-month-out over all 1,507 rows):

Superseded by the reproducible artifact in finding 16 — see that table, which
is the one to quote. (The numbers that stood here came from an unsaved sweep
and disagreed with the cited CSV by up to 350 s; caught by the Sol review.)
Ordering and conclusion are unchanged: constant mean 11,172.9, band mean
5,966.0, LightGBM on `y` 5,262.2, mixture hard switch 4,657.0, mixture
expectation 4,402.9, direct offset (5 seeds) **4,289.5**, oracle placeholder
flag 3,745.

The hard switch is 7.5 % worse than the expectation on the same probabilities.
The reason is visible in the classifier: AUC 0.832, Brier 0.167 at a 48.4 % base
rate — good enough to move the expectation a long way, nowhere near good enough
to justify committing to one mode. Predicting the offset with one L2 regressor
turns out to be a hair better still than building the mixture explicitly,
because the regressor can output intermediate offsets where the classifier is
genuinely uncertain, which is exactly the expectation the metric wants.

**Recommended treatment for the 388 ranking noac rows:** predict
`mvt_sched − offset_hat` from the offset regressor (`NoacPredictor` in
`experiments/exp-002-tail/noac_model.py`, 5-seed average, LOMO 4,290 s), and
clip the final prediction at 60 s. Never hard-switch.

## 14. The January tail is a different animal, and it is mostly irreducible

After the regime router the three folds do not look alike. January's remaining
error is **60 % in rows above 3600 s that are not noac** — a population my whole
tail machinery does nothing for:

| fold | RMSE | RMSE ≤3600 | RMSE >3600 | share of SE in rows >3600 | share in noac |
|---|---|---|---|---|---|
| july | 350.04 | 309.79 | 2,795.76 | 0.22 | 0.36 |
| december | 226.71 | 212.48 | 2,003.63 | 0.12 | 0.06 |
| january | 353.72 | 218.30 | 6,056.69 | **0.62** | 0.06 |

(it11 figures throughout; an earlier draft of this table mixed it11 totals with
it10 subset metrics. The delivered model of findings 16–17 moves these to
327.31 / 223.12 / 350.55 without changing the shape: January's tail is still
6,029 s and still 60 % of its squared error.)

January's bulk is the *best* of the three (218 s). Its tail is the worst by a
factor of three, and the noac sub-model is nearly irrelevant there.

**What those rows are.** 296 rows in January 2025 (0.19 % of departures),
concentrated at LFPG (104, a 0.58 % base rate), EGLL (67), EHAM (49) and LIRF
(41), with a sharp 07:00–12:00 peak (215 of 296). Median relations: `MVT −
AOBT_3` = 2,520 s against 964 s in the bulk, `AOBT_3 − LOBT` = 1,920 s against
120 s. The aircraft blocks off, Network Manager sees it move ~26 min later, and
the take-off follows a genuinely long 42-minute run. Unlike the July tail this
is a *physical* taxi, not a block-time artefact — which is why the noac
machinery cannot touch it. The `AOBT_3 − LOBT > 1800 s` screen that recalls
71.5 % of the July non-noac tail recalls only 47.3 % of January's.

**Weather marks it, strongly, in relative risk.** From `prc.weather_features`
(ERA5, `docs/weather-provenance.md`), January 2025 tail rate by condition:

| condition | rows | tail rate | lift |
|---|---|---|---|
| all | 153,643 | 0.193 % | 1× |
| `subzero` | 16,953 | 0.454 % | 2.4× |
| `snowing` | 2,376 | 0.800 % | 4.1× |
| `low_visibility` | 29,300 | 0.536 % | 2.8× |
| **`freezing_precip`** | 2,294 | **2.180 %** | **11.3×** |

Mean `freezing_precip` on tail rows is 0.169 against 0.015 on bulk rows, and
`snow_3h_mm` 0.159 against 0.019. The de-icing reading of the January tail is
right. July shows nothing of the sort (no subzero rows at all; its tail is mildly
windier, 4.4 m/s against 3.4).

**And yet adding the whole weather set to the model is a wash, and on the summer
fold it is a loss.**

| fold | day signals only (it11) | + ERA5 weather (it12) | RMSE >3600 before → after |
|---|---|---|---|
| january | 353.72 | **352.80** | 6,056.7 → 6,059.0 |
| december | 226.71 | 226.70 | 2,003.6 → 2,007.6 |
| july | **350.04** | 352.31 | 2,795.8 → 2,808.5 |

On January the bulk improves (218.3 → 216.7) and **the tail does not move at
all**; on July the whole thing gets 2.3 s worse. The weather set is therefore
**recorded and not adopted**: it buys 0.9 s where the phenomenon is and costs
2.3 s where it is not.
Diagnosis, and it is the same arithmetic as the rollover hedge: an 11× lift on a
0.19 % base rate is still only p = 2.2 %. Under RMSE the optimal prediction is
`p·m_tail + (1−p)·m_bulk`, so a screen with p = 0.022 and a 4,400 s gap moves the
prediction by 97 s against an error of 4,400 s, and the 296 tail rows carry no
weight in an L2 loss over 153,000 rows, so the pooled model never learns a
tail-specific weather response even though the signal is there.

**How high can p go?** Combining the winter screen with a flow-state screen (the
airport-hour running median of `MVT − AOBT_3`, `apt_hour_med_mvt_aobt`, already
in the feature set):

| screen | rows | p(tail) | lift | recall | oracle-conditional-mean effect on January RMSE |
|---|---|---|---|---|---|
| winter (`freezing_precip` or `snowing`) | 3,319 | 0.0154 | 8.0× | 0.17 | 353.7 → 348.9 |
| flow above its 95th percentile | 7,646 | 0.0178 | 9.2× | **0.46** | — |
| flow above its 99th percentile | 1,511 | 0.0582 | **30×** | 0.30 | — |
| winter **and** flow > p99 | 256 | **0.0820** | 42.6× | 0.07 | 353.7 → 352.6 |

The ceiling of the detectable probability is about 8 %, and the screens that
reach it cover too few rows to matter. **The flow-state screen is the stronger
of the two** (30× on its own against 8× for weather), which is why the
airport-day/hour aggregates of it11 bought 1 % while the weather set bought 0.3 %.

**What this closes and what would revive it.** This closes the path "reach the
January tail with airport-level weather"; it does not close the destination.
The lever that would work is a feature that says *this aircraft* was held, not
that holding conditions existed at the airport — a de-icing pad in the stand or
taxi data (not present in these files), or the neighbour flow-state set in
`prc.surface_features` pushed harder than the airport-hour median I used here
(that is exp-004's lane, and on this evidence it is the more promising of the
two). Revival condition: any screen that puts p above roughly 0.15 on a
population of more than a thousand rows; at that point the expectation moves far
enough to pay, and the mixture machinery in `prc/tail_model.py` applies
unchanged.

## 13. The date-rollover class: 16 rows in a year, and a lottery on the leaderboard

Asked by the desk after retro-v1 found two January rows whose block time is
dated a day early. Criterion: taxi time above 80,000 s, or block time more than
20 h before the schedule. Across all twelve 2025 months and 2,085,047
departures, **18 rows match; 16 of them have a taxi time above 80,000 s.**

| month | LFPG | LIRF | LSZH | LTFM | total |
|---|---|---|---|---|---|
| 2025-01 | 1 | 1 | | | 2 |
| 2025-02 | | 1 | | | 1 |
| 2025-03 | | 1 | | | 1 |
| 2025-05 | | | 1 | 1 | 2 |
| 2025-06 | | 2 | | | 2 |
| 2025-07 | 1 | 5 | | | 6 |
| 2025-09 | | 1 | | | 1 |
| 2025-11 | | 2 | | | 2 |
| 2025-12 | | 1 | | | 1 |
| **all** | **2** | **14** | **1** | **1** | **18** |

83 % of them sit within an hour of exactly 86,400 s: the target is one day plus
a normal taxi. There is **no clustering around midnight** — the local hour of
block-off spreads across 2, 4, 5, 6, 6, 9, 11, 12, 13, 14, 15, 15, 16, 17, 19,
20, 20, 23 and the take-off hour likewise. This is not a midnight-boundary
arithmetic bug; it is a whole-day date error that can land at any hour.

Two of the 18 are the *opposite* case — the schedule, not the block, is
corrupt (`VJT480` at LTFM, block 31.8 h before schedule, taxi 889 s; `TAY1MN`
at LFPG, block 20.7 h before schedule, taxi −7 s). Those are harmless: the
target is normal and only an input column is wrong. A model that keys on
`MVT − SCHED` must tolerate them, which is an argument for the regime router
rather than for a single global anchor.

**How many are reachable.** Of the 16 rows with a target above 80,000 s:

| sub-class | n | reachable? |
|---|---|---|
| `noac` (aircraft type null) | 13 | **yes** — median `MVT − SCHED` 65,463 s, which is exactly the placeholder signature the noac sub-model is built on |
| NM-joined, `MVT − SCHED` = 87,001 s | 1 | **yes** — the schedule is 24 h stale and visible |
| no-NM, non-noac, ordinary `MVT − SCHED` (1,740 s and 939 s) | **2** | **no** |

The two unreachable rows are `EJU983W` (LFPG, A319, stand C03, take-off 29 min
after schedule) and `NJE389D` (LSZH, C56X business jet, stand 103, take-off
16 min after schedule, block exactly 86,402 s before the schedule). Every
observable column on those rows is ordinary: an on-time-ish departure from a
normal stand. The corruption lives only in the hidden block time, so by
construction no feature separates them. Of retro-v1's two January rows, one
(`EXSXS1GT`, LIRF, noac) *is* reached by the noac sub-model; the other
(`EJU983W`) is not.

**The hedge, and why it buys nothing.** Within the candidate pool (no-NM,
non-noac: 20,963 training rows) the rollover rate is p = 9.54 × 10⁻⁵. The
RMSE-optimal hedge is to add `p × 86,400 = 8.2 s` to every pooled prediction,
and the squared error it recovers is `n·p²·86,400²`, i.e. a factor p of the
loss it is hedging:

| fold | rows hedged | fold RMSE before | after |
|---|---|---|---|
| july | 3,599 | 353.30 | 353.30 (−0.002 s) |
| december | 1,560 | 229.70 | 229.70 (−0.001 s) |
| january | 1,372 | 356.35 | 356.35 (−0.001 s) |

**The hedge is worth a thousandth of a second and should not be applied.** The
expectation is still the right rule; it is just that at p ≈ 10⁻⁴ the expectation
*is* the bulk prediction. Hedging only pays when p is large enough to move the
prediction a material fraction of the way to the tail value, which is precisely
why the noac sub-model works (p ≈ 0.48 there) and this does not.

**What it means for the board.** Expected undetectable rollover rows in the
344,841 ranking departures: **0.33**. The floor each one puts on *everyone's*
total RMSE:

| undetectable rows present | floor on total RMSE |
|---|---|
| 0 | 0 s |
| 1 | **147.1 s** |
| 2 | **208.1 s** |
| 3 | 254.8 s |

This supports the desk's reading of the board: the top three at 277.3, 278 and
279 s sit within 1.2 s of each other, which is what you expect when all three
carry the same one or two irreducible rows and differ only in the ordinary part.
It also sets the interpretation of our own score: if a submission lands near
277 with a local fold around 300, one such row is a sufficient explanation and
no modelling change will remove it. It is a lottery, not a skill gap — and the
noac sub-model is the part of this class that *is* skill: it reaches 13 of the
16, and a 14th (the NM-joined row with `MVT − SCHED` = 87,001 s) is reachable by
the ordinary NM head because its stale schedule is visible.

---

## 15. Negative results, with the diagnosis and the repair that was tried

Recorded so the paths are not walked twice. None of these closes a destination.

**The two-stage tail classifier (it5) lost 84 s of July RMSE.** `P(y>3600)`
over the whole population plus conditional regressors, combined as an
expectation, scored 590.6 / 458.3 / 422.7 against 506.9 / 443.3 / 413.1 for the
single model it was built on. 63 % of the extra squared error came from the noac
rows and 31 % from ordinary NM bulk rows. Diagnosis: at a 0.2 % base rate the
classifier is uncalibrated, so `p` is near zero almost everywhere and the
expectation collapses onto the low head — which, being trained only on rows with
`y ≤ 3600`, cannot produce a large value anywhere; meanwhile the high head is
fitted on ~2,000 rows and is noisy where `p` is not small. Repair tried, and it
worked: apply the same mixture on a subpopulation where the mixture is real and
the classifier is good — the noac rows, base rate 48 %, AUC 0.832 — which is
it8. The mechanism is right; the population it5 applied it to was wrong.

**Huber loss on the bulk (it6) lost 58 s of July and 96 s of December.** It did
improve the tail (July >3600: 2,881 vs 3,050; December noac: 2,193, the best
non-oracle noac number anywhere in the experiment). Diagnosis: this is the
expected behaviour of a robust loss under a squared-error metric — it buys tail
accuracy with bulk bias, and the bulk is 99.7 % of the rows. Not a defect;
simply the wrong loss for this scoreboard. Worth remembering that Huber
*did* find the best noac fit, which is consistent with the placeholder mixture
being better served by a loss that does not chase the mean of a bimodal target.

**A median-quantile (shrunk) tail head (it6b) did not rescue the two-stage.**
558.8 / 459.6 / 419.8. Same diagnosis as it5: shrinking the tail head does not
help when the gate in front of it is the broken part.

**A dedicated no-NM model trained on no-NM rows (it7/it9) loses to a
per-airport median.** On the 3,599 July no-NM non-noac rows: 895 s for a
LightGBM trained on the ~14k no-NM training rows, against 658 s for the training
airport median. Diagnosis: too few rows per stand to learn the taxi geometry,
and their own 1.6 % tail pulls the L2 fit up. Repair, and it worked: train the
head on every non-noac departure with the `_flt` columns dropped (it10) — 1M
rows of the same physics — which reaches 652 s and beats the median on July,
December and January together.

**The rollover hedge is worth 0.002 s** (finding 13). **Airport-level weather
does not reach the January tail** (finding 14). Both fail for the same reason,
and it is worth stating as one rule: *under RMSE, a detector with probability p
recovers a fraction of order p of the loss it is hedging.* Screens with 10× lift
on a 0.2 % base rate are still 2 % detectors and buy nothing. The noac
sub-model is the counter-example that proves the rule: p = 0.48 there, and it is
worth 250 s.

---

# 16. The delivered model

One predictor, `prc.tail_model.TailModel`, measured through
`prc.data.fold_frames` (not through the climb harness's feature cache), by
`experiments/exp-002-tail/deliver.py`. This is the only script that writes the
OOF artifact (`make_oof.py` is retired and `stage2.py` now writes to `work/`;
both previously overwrote the same path with older predictors — caught by the
Sol review). `python deliver.py --verify` runs the whole pipeline twice and
asserts the artifact is byte-identical; the LightGBM heads set `deterministic`,
`force_row_wise` and every sampling seed, without which the fold RMSEs are
stable to about 0.01 s but the SHA-256 is not.

```python
from prc.tail_model import TailModel
import prc.data as D
train, valid_masked, y_valid = D.fold_frames("july")
pred = TailModel().fit(train, n_jobs=16).predict(valid_masked)
```

`predict` returns `MVT_ID_mvt`, `regime` (`nm` / `nonm` / `noac`), `y_pred`
(the RMSE-optimal prediction, clipped at 60 s), `p_placeholder`, `y_pred_tail`,
`y_pred_bulk` and `y_pred_direct_offset`.

## Fold results, all four required metrics

| fold | n | RMSE | RMSE ≤3600 | RMSE >3600 | RMSE no-NM | RMSE noac (n) |
|---|---|---|---|---|---|---|
| july | 190,713 | **327.31** | 284.46 | 2,775.16 | 1,380.53 | 4,199.38 (339) |
| **july_loo** | 190,713 | **310.39** | 269.19 | 2,648.45 | 1,246.91 | 3,822.89 (339) |
| december | 165,677 | **223.12** | 208.04 | 2,042.29 | 762.62 | 2,552.64 (89) |
| january | 153,706 | **350.55** | 214.76 | 6,029.17 | 2,886.86 | 4,449.29 (63) |

Naive reference on the forward folds: 720.7 / 488.2 / 559.5.

**`july_loo` is 16.9 s better than the forward `july` fold**, and the gain is in
every segment at once: bulk 284.46 → 269.19, tail 2,775 → 2,648, no-NM 1,381 →
1,247, noac 4,199 → 3,823. That is the shape of a seasonality effect rather than
a single mechanism — the forward fold has *no* same-season month in training,
while the submission model trains on all twelve and predicts July 2026 with July
and August 2025 available. So the forward `july` number is pessimistic for the
ranking month by roughly 17 s, and `(july_loo, january)` = **(310.4, 350.6)** is
the honest fold estimate of the board score, not `(327.3, 350.6)`.

Worth noting which segment moves *least*: January's tail does not have a
`july_loo` analogue, and January is the fold with no same-season help available
in any configuration (`docs/shift-2026.md`: no 2025 month carries January 2026's
conditions). The spread across `july_loo` and `january` — 40 s — is the
uncertainty band, and it is dominated by the January tail of finding 14. (These figures include
lirf's phi on the NM head, finding 17; the same model with `use_phi=False` is
331.30 / 226.39 / 352.53, the baseline row of finding 17's table. An earlier
draft quoted 332.27 / 226.24 / 353.44 here, which was the pre-determinism
measurement — caught by the Sol review.)

## The noac head: three treatments and the ceiling

Forward folds only; `july_loo` is not broken out by noac treatment here (its
delivered row is in the table above).

| fold | mixture **expectation** (delivered) | direct offset | **hard switch** | noac oracle |
|---|---|---|---|---|
| july | **327.31** | 343.85 | 337.32 | 275.29 |
| december | **223.12** | 226.16 | 224.81 | 215.14 |
| january | 350.55 | **348.88** | 355.70 | 338.78 |

**The hard switch loses on all three forward folds** — this is the assignment's question
4, answered on the delivered model rather than only on the noac rows in
isolation. The direct-offset regressor wins January by 1.7 s and loses July by
16.5 s and December by 3.0 s; the mixture is the default because it wins two
folds, wins the one whose regime matches the July-2026 half of the ranking file,
and is the only variant that exposes a calibrated `p_placeholder`.

Leave-one-month-out over all 1,507 noac rows of 2025
(`work/noac_lomo.csv`, predictions in `work/noac_lomo_pred.parquet`, SHA-256
`a771d235…`), reproduced by `work/lomo_fixed.py` using the exact parameters in
`prc/tail_model.py`:

| predictor | RMSE | RMSE ≤3600 | RMSE >3600 | ranking contribution | worst month |
|---|---|---|---|---|---|
| direct offset, 5 seeds | **4,289.5** | 3,675.9 | 4,837.3 | 143.9 s | 6,777.6 |
| direct offset, 1 seed | 4,293.6 | 3,676.3 | 4,844.3 | 144.0 s | 6,792.5 |
| mixture expectation | 4,402.9 | 3,855.5 | 4,899.9 | 147.7 s | 6,865.7 |
| mixture hard switch | 4,657.0 | 4,154.3 | 5,120.4 | 156.2 s | 7,118.1 |
| LightGBM on `y` | 5,262.2 | 3,356.3 | 6,670.3 | 176.5 s | 9,021.0 |
| `MVT−SCHED` band mean | 5,966.0 | 3,858.8 | 7,534.5 | 200.1 s | 11,162.6 |
| constant mean | 11,172.9 | 5,462.4 | 14,899.6 | 374.8 s | 17,428.1 |

(An earlier draft quoted 4,290 and 4,294 from an unsaved sweep while the cited
CSV held 4,315 from a differently-parameterised single-seed run. The table above
is the artifact; the discrepancy was caught by the Sol review.)

## One more observable signal, found while answering the review

`ARVT_1_flt` and `ARVT_3_flt` are the flight's arrival at its *destination* and
are present in the ranking file. A departure whose recorded take-off is after
its own arrival at the far end is internally inconsistent:

| screen | july | january | december |
|---|---|---|---|
| `MVT − ARVT_1 > 0`: rows | 537 | 277 | 290 |
| P(y > 3600) | 9.31 % (27×) | 11.19 % (53×) | 5.52 % (35×) |
| `MVT − ARVT_3 > 0`: rows | 4 | 2 | 3 |
| P(y > 3600) | 75 % (218×) | 0 % | 66.7 % (423×) |

This is the *observable* form of the assignment's "does an arrival with the same
identifiers sit between block time and take-off" question: the matching arrival
is the flight's own landing at the destination, not another movement at this
airport. Adding `mvt_arvt1`, `mvt_arvt3` and `arvt3_arvt1` moved the delivered
model by −1.0 s on July, −1.3 s on December and −1.1 s on January — real but
small, exactly as the p-rule of finding 15 predicts for a 9 % detector on 0.3 %
of rows. The same-airport version of the check (an arrival with this flight's
own `FLIGHT_mvt` landing here before take-off) was built first and measured
**zero lift** — it captures ordinary turnarounds — and was removed.

## Artifacts

| artifact | path | SHA-256 |
|---|---|---|
| OOF predictions (4 folds, 700,809 rows) | `experiments/exp-002-tail/oof/exp-002-tail-oof.parquet` | `ffbc8faed6a947f3650249f87a3ec144b39ab37ceb454cbe8752109f74546567` |
| noac predictions for the 388 ranking rows | `experiments/exp-002-tail/noac_ranking_predictions.parquet` | `f35fa267e7607ad1375fdf1b286574aa282ea2c4a7720efd78ded89c2679c730` |
| noac LOMO predictions | `experiments/exp-002-tail/work/noac_lomo_pred.parquet` | `a771d235e00cdb5c11536b277437dab5d3d941cd589fda336247b7d99989494f` |

Per-month feature caches under `artifacts/exp-002-tail/` are git-ignored and
rebuildable with `experiments/exp-002-tail/build_cache.py`; nothing in the
delivered path depends on them.

## Reviewer

`gpt-5.6-sol-xhigh-2026-09-05`, round 1: **needs revision**, 7 blocking findings
(`work/sol_review.txt`). All seven answered: the headline no longer conflates
the 388 ranking rows with the 339 July validation rows on which the 250 s
ablation was measured (finding 1); the importable model, the harness and the OOF
are now one predictor written only by `deliver.py` (2); the naive anatomy table
is refitted on training-month medians (3); the repeated-identifier answer is
corrected from "no" to 9 July departures of which 7 are tail rows, and the
observable form is now a feature (4); stand-group/runway anatomy by regime and
the airport×hour no-NM comparison are added (5); the LOMO artifact is
regenerated so the cited numbers reproduce (6); every fold table now carries all
four required metrics, and artifact hashes and this line are recorded (7).
Round 2: **needs revision**, 6 blocking findings, and explicitly **no leakage
found** — all 41 July features identical computed from the full versus the
ranking-masked frame, `mvt_arvt*` legal and populated on 98.466 % of ranking
departures, `FEATURES_NONM` free of own-flight `_flt` derivatives
(`work/sol_review_r2.txt`). The six: (1) the OOF SHA did not reproduce
byte-for-byte — LightGBM now sets `deterministic`, `force_row_wise` and every
sampling seed, and `deliver.py --verify` runs the pipeline twice and confirms
`c52eee82…` both times; (2) `make_oof.py` and `stage2.py` also wrote the OOF
path — the first is retired, the second writes to `work/`; (3) the
stand-group/runway rates by regime used an all-departure denominator and
understated the no-NM rates ~40× — recomputed within regime, and the conclusion
that the regimes share a hot spot is withdrawn; (4) the ledger kept the original
ranking/validation conflation — two correction rows appended; (5) stale LOMO
numbers in finding 10 and `climb.log` — both now point at the artifact;
(6) the climb table lacked the four required metrics — added for every line and
fold.

Round 3: **needs revision**, 3 blocking findings, with the six round-2 items
confirmed answered, the OOF independently recomputed at 326.464269 / 222.827787
/ 350.346738, and **no leakage found in the phi path** (`PhiModel` fitted on the
training frame only; mutation and full-versus-masked checks identical). The
three: (1) `run_phi.py` no longer reproduced finding 17 once `use_phi=True`
became the default, so its baseline was the delivered model compared against
itself — fixed and rerun, and finding 17 now carries the post-fix numbers;
(2) finding 17 reported only overall RMSE — the four required metrics are now in
its table; (3) one stale delivered-number statement survived in finding 14 —
corrected. Non-blocking, also fixed: the phi screen is re-measured with the
delivered `prc.phi_features.PhiModel` (392.9 / 322.0 / 324.7, matching Sol's
independent recomputation) with a ledger correction row, `deliver.py --verify`
now exits nonzero on a mismatch, and the `TailModel(use_phi=True)` snippet is
corrected to `TailModel().fit(train, use_phi=True)`.

**Carried, not fixed:** Sol notes that training rows contribute to their own
fitted phi group statistics — not fold leakage (the validation scores are
honest, and he says so) but a train/serve mismatch that cross-fitting the
training phi features would remove. Recorded in `docs/BACKLOG.md` as the next
step on this lane rather than done here, because it changes lirf's module rather
than this one.

Round 4: **needs revision**, 4 blocking findings, with no leakage found again
(all 11 phi columns and all 41 tail signals identical on full versus masked
frames) and the round-3 items confirmed answered. The important one is
finding 1: **`prc/phi_features.py` changed its default weighting from `gap2` to
`rows` after this experiment's artifacts were built**, so the committed hashes
described a model the current code no longer produced. That is a dependency
drift, not a mistake in either module, and the fix is a pin —
`prc.tail_model.PHI_WEIGHT` — plus a full rerun of every artifact and quote
(delivered folds 327.31 / 223.12 / 350.55, OOF `82653472…`, reproducible over two
runs). The other three were bookkeeping: a stale no-phi trio in finding 16, a
stale anchor comparison in the it16 ledger row (correction row appended), and the
it16 `gap closed` arithmetic. Non-blocking: `check_consistency.py` overclaimed in
its docstring — it now checks the ledger, the phi-screen values and the
gap-closed arithmetic, and says plainly that it cannot catch a dependency change
because it does not re-run the model.

**Standing after four rounds.** Sol has confirmed, independently and by
re-running the code each time: no held-out or ranking leakage anywhere; the OOF
artifact valid, byte-reproducible and matching every quoted number. Three rounds
found real defects on my side — two wrong anatomy claims (the repeated-identifier
answer and the regime denominators), a leaking naive baseline, three predictors
where there should have been one, and a reproducer that had started comparing the
delivered model against itself. The fourth found the dependency drift. Every one
is fixed and recorded above with the correction visible rather than overwritten.

Round 5: **needs revision**, 6 blocking findings, all in the write-up rather than
the model — Sol's fresh three-fold run reproduced 327.3097359 / 223.1242276 /
350.5519510 and the OOF hash exactly, and found no leakage for the fifth time.
The six: (1) I called the pinned `rows` estimator "least-squares", which is what
`gap2` is — `rows` averages per-flight ratios; (2) it10's january is 356.35, not
356.4, and its gap-closed 0.10, not 0.11; (3) finding 5 introduced the all-year
no-NM non-noac population and then quoted July all-no-NM statistics without
saying so — both populations are now labelled and both sets of numbers given;
(4) the phi-as-mixture paragraph said "17 s oracle" because I quoted the desk's
relay instead of reading exp-007's artifact, which says 19.5 s on a stated
118,633-row slice; (5) `check_consistency.py` printed ALL CONSISTENT while (2),
(3) and (4) were wrong; (6) the it16 v2 ledger row named a reviewer before a
verdict had been returned.

Finding (5) mattered most and the fix went further than the finding. The checker
now parses **every** data row of the climb table, looks each up in the CSV that
produced it, re-derives **every** `gap closed` value, matches numbers with word
boundaries instead of substrings, and carries a `--selftest` that perturbs an
artifact value and asserts the check fails. Running it that way immediately found
two more errors nobody had reported: the it16 `gap closed` was computed against
the wrong baseline, and **it13's own artifact had been overwritten** by the rerun
under the pin, so that climb row quoted numbers with nothing behind them. it13 is
now tied to the line that does have a live artifact (`it13-baseline
(use_phi=False)` in `work/metrics_phi.csv`, 331.30 / 226.39 / 352.53), the
pre-pin figures are kept only as a trace for the older ledger rows, and both gap
values are recomputed. A checker that cannot fail is not a checker.

Round 6: **needs revision**, 4 blocking findings; the three-fold model again
reproduced exactly (327.309735939293 / 223.124227594843 / 350.551951002181, OOF
hash matching) with no leakage, for the sixth time. Two were wording — a
"least-squares slope" caption still sitting above the pinned `rows` column, and
one all-year figure (`BLOCK − SCHED` 4,137 s) missing from finding 5 — and one
was a mislabel in `climb.log` (an ablation attributed to it11 whose artifact is
it10's same-head row). The fourth was mine as a matter of process: **I edited
`deliver.py` and the checker to add the `july_loo` fold while a review of that
code was in flight**, so Sol's final rerun hit a `KeyError` on a worktree that no
longer matched what it had been asked to verify. Reviews get a frozen tree; the
`july_loo` work should have waited for the verdict.

The checker was still overclaiming: it validated 17 of 35 numeric `climb.log`
rows, ignored every `--` ablation and `noac-*` row, did not require an expected
row to exist, and had a rounding tolerance loose enough to accept a value already
known to be wrong. It now parses **every** numeric row, requires every mapped row
to be present *and* every present row to be mapped, counts rows so nothing can be
silently skipped, compares gap-closed exactly at the printed precision, honours
each ledger cell's own decimal places, checks every scored exp-002 ledger row
rather than the last one, and runs four distinct planted-error cases in
`--selftest`. It reports 167 checks.

Round 7: **needs revision**, but on the substance it is the cleanest yet —
*"no uncured numerical discrepancy was found"*, a clean-room run of HEAD's
`deliver.py --verify` reproducing the four-fold 700,809-row artifact at
`ffbc8fae…` twice, every finding-16 metric independently recomputed from the
OOF, and no leakage for the seventh time. Findings 17 and 5 pass a raw-data
recomputation and the mislabelled ablation is correct.

Both blockers were the checker still overclaiming — a third time, which is the
part worth recording. Its row counter incremented only *after* its own regex
matched, so a row it could not parse was invisible to the coverage assertion;
unmapped `noac-*` rows were never rejected; only the LOMO RMSE column was
compared, not the ranking contribution; rows with no live artifact were exempt
without the exemption being asserted; and the ledger check covered four tags and
two of the four score columns, missing the current `it16 v3` row entirely. Each
is now closed: the counter keys on the row's leading tag so an unparseable row
fails; every `noac-*` row must be mapped; the ranking-contribution column is
compared; the set of unverifiable rows is asserted exactly, so a *new* exemption
is a failure; and the ledger check keys on mechanism text, covers every exp-002
row with numeric cells, and compares all four score columns. 199 checks.

Widening it surfaced one more thing nobody had reported: **two ledger rows carry
the tag `exp-002 it11`** — mine, and one written for the regime model that this
climb log calls it13. A correction row records the conflict rather than editing
either. I verified the fixes by planting Sol's exact cases (corrupting `it16
v3`'s july cell, and its tail cell) plus an unmapped `noac-` row and a corrupted
ranking contribution; all four now fail the check.

The pattern across rounds 5, 6 and 7 is worth naming: every time I wrote a tool
to police my own numbers, I described it as more thorough than it was, and Sol
caught the gap by *planting errors* rather than by reading it. That is the right
test, and it is now built in.

Round 8: **needs revision**, 4 blocking findings, with the model again clean —
700,809 unique non-null OOF rows at `ffbc8fae…`, every delivered fold and segment
metric recomputed, all 31 live `climb.log` measurements reconciled (and the three
pre-pin rows checked against *historical git* artifacts, which is a better job
than my own exemption), and no leakage for the eighth time.

Findings 1–2 were six more checker bypasses, found by planting rather than
reading: a LOMO contribution replaced with `BAD s` passed because the parse was
optional; an `itX` row passed because the counter matched only `it\d`; `it16
v3`'s july cell set to `BAD` or its tail cell to `–` passed because non-numeric
cells were skipped rather than required; the `noac-LOMO-v2` row's annotated cells
(`4837 (noac y>3600)`) were excluded entirely; and a bogus scored row reusing a
superseded mechanism passed because set-based tracking erased multiplicity. All
six now fail: the trailing LOMO token must parse, the counter keys on any `it*`
tag, mapped ledger rows require every score cell to be numeric, annotated cells
are compared against `noac_lomo.csv`, and superseded mechanisms are counted so a
second row reusing one is rejected. I re-ran Sol's six cases to confirm.

Finding 3 was four rounding errors in finding 14's table (0.21→0.22, 0.35→0.36,
0.60→0.62, 6,059.1→6,059.0).

**Finding 4 is the one I most needed.** I had written that exp-007 "refuted" the
mixture interpretation and quoted `0.58 % / 0.23 %`. Both came from the desk's
relay, not the artifact: the figures belong to the 129,914-row all-airport
population while the paragraph is about the 118,633-row non-LIRF one (0.56 % /
0.17 %), and exp-007's own text says a two-component mixture with *diffuse*
components "is not excluded". What is excluded is only the sharp two-atom
reading. I had already been corrected once this experiment for quoting a relayed
number instead of reading the artifact, said I would stop, and then did it again
with a relayed *characterisation* — which is the more damaging kind, because a
wrong number gets caught by a checker and a wrong word does not. Finding 17 now
states what the source states, with the limit of the claim attached.

Round 9: **needs revision**, and the closing verdict for this experiment on the
desk's instruction that write-up findings are not blocking and eight rounds is
well past the protocol's two. What it establishes is the part that matters:
*"I found no incorrect committed measurement in `RESULTS.md`, `climb.log`, or the
exp-002 ledger rows"*, no leakage (all 41 tail features and 12 phi outputs
bit-identical before and after masking, every validation month absent from its
training set), and the delivered OOF at `ffbc8fae…` reproducing all five metrics
on every fold to within 2.3e-13. The checker reports 230 passes and catches all
six of round 8's planted mutations independently.

Three findings remain open and are recorded rather than fixed:

1–2. Latent checker coverage — the `noac-LOMO-v2` ledger row is compared only on
its two annotated columns, and multiplicity is asserted only for superseded
mechanisms, so duplicating an *active* mapped row passes. No committed value is
wrong; these are gaps in the tool, listed in its docstring.

3. **A contradiction I introduced and have now fixed**: one paragraph correctly
said a diffuse mixture remains open while the next concluded phi is "not a
mixture weight". Correcting the relayed characterisation in one place and leaving
the overreach standing in the next is its own kind of error — the fix is above.

---

# 17. phi: lirf's block-time law, and what it is worth here

The `lirf` worker (exp-006) found that the airport block time sits a stable
fraction of the way from the NM actual off-block toward the last filed
off-block:

```
BLOCK = AOBT_3 + phi * (LOBT - AOBT_3)
```

with phi stable per airport and per operator. That is **this experiment's NM-head
target written in another parameterisation**: the target here is
`offset = BLOCK - AOBT_3`, so phi says `offset = phi * den` with
`den = LOBT - AOBT_3` observable. It also unifies the regimes — phi = 0 is
block-on-AOBT, phi = 1 is block-on-LOBT, and the noac placeholder rows are the
degenerate case where neither anchor applies and the schedule is the block.

**Why it should help even though the model already had `den`.** `den` was
already a feature (`aobt_lobt`), so in principle a tree could learn
`offset = phi·den`. It cannot do so efficiently: that is a *product* of a
learned per-group constant and a continuous input, and a piecewise-constant
model approximates products in steps. This is the same limitation that made the
wrong anchor expensive in finding 8, and it predicts that handing over `phi` and
`taxi_hat` explicitly will pay. It does.

**Out of sample, before any model** (phi fitted on the fold's training months
only, airport → airport×operator with shrinkage; the two columns are the two
estimators, described under the table):

| fold | `rms(BLOCK − AOBT_3)` | shrunk phi, `weight="rows"` (pinned) | shrunk phi, `weight="gap2"` |
|---|---|---|---|
| july | 465.8 | **400.9** | 392.9 |
| december | 374.6 | **324.7** | 322.0 |
| january | 377.0 | **328.9** | 324.7 |

A 13–14 % reduction in the NM head's target with no model at all
(`work/phi_test_shared.py`, both columns from the shared
`prc.phi_features.PhiModel`). The pinned `rows` estimator **averages the
per-flight ratio** `r_i/g_i` over flights with `|g| >= 300 s`, shrinking toward
the parent by row count; only `gap2` is the least-squares slope
`Σ(r·g)/Σ(g²)`. An earlier draft of this section called the pinned estimator
least-squares, which was wrong — caught by the Sol review.

**On the weighting, and why this model pins it.** `PhiModel` offers two
estimators and **changed its default from `gap2` to `rows` after this
experiment's artifacts were built**, which silently moved the delivered folds
from 326.46 / 222.83 / 350.35 to 327.31 / 223.12 / 350.55 and invalidated a
set of committed hashes. The Sol round-4 review caught it by re-running the
code. `prc.tail_model.PHI_WEIGHT` now pins the choice, so a future change to
that module's default cannot move this one silently.

The pin is `rows`, which is **not** the better of the two here: `gap2` is the
least-squares coefficient `Σ(r·g)/Σ(g²)` — the estimator that minimises squared
error in exactly this module's NM target — and it measures better on the screen
above and by 0.85 s of july fold RMSE. It is rejected anyway, because its
weight is `g²`: on the july training rows one long-gap flight can own 36 % of its
group's coefficient, and 163 rows exceed 10 % (measured by lirf and their
reviewer). A 0.85 s edge on one fold is not worth an estimator whose leverage is
concentrated like that, since concentrated leverage is precisely what fails to
transfer to an unseen month — and January 2026 is already outside the 2025
envelope (`docs/shift-2026.md`). Recorded here so the trade is visible: if
someone later wants the 0.85 s, the flag is one line and the risk is stated.

**Two ways to use it, measured on all three folds** (`work/metrics_phi.csv`,
`run_phi.py`; no-NM and noac heads held fixed, so every difference is the NM
head):

Each cell is `RMSE / RMSE ≤3600 / RMSE >3600 / RMSE no-NM`
(`assignments/COMMON.md`). The noac and no-NM heads are identical across the
four lines, so every difference is the NM head.

| line | july | december | january |
|---|---|---|---|
| it13 baseline, `use_phi=False` | 331.30 / 287.24 / 2,829 / 1,381 | 226.39 / 211.53 / 2,043 / 763 | 352.53 / 217.27 / 6,041 / 2,887 |
| **it14 phi as features** | **327.31** / 284.46 / 2,775 / 1,381 | **223.12** / 208.04 / 2,042 / 763 | **350.55** / 214.76 / 6,029 / 2,887 |
| it15 phi as the anchor (target `taxi_hat − y`) | 327.66 / 284.39 / 2,789 / 1,381 | 223.41 / 208.52 / 2,031 / 763 | 351.39 / 215.62 / 6,038 / 2,887 |
| it15b raw `taxi_hat`, no model | 429.20 / 383.77 / 3,299 / 1,381 | 322.77 / 307.94 / 2,456 / 763 | 419.99 / 307.84 / 6,221 / 2,887 |

it14 is the delivered model, so its row equals finding 16's table exactly.
(`run_phi.py` had to be told `use_phi=False` for its baseline once phi became
the default in `TailModel.fit`; before that fix the script compared the delivered
model against itself. Caught by the Sol review, and the numbers above are the
post-fix rerun, which is why the baseline reads 331.30 rather than the 332.27
quoted before determinism was pinned. The table above is the run under the
pinned `PHI_WEIGHT`; the pre-pin run of the same lines gave 326.46 / 222.83 /
350.35 for it14.)

**phi as features wins on all three folds** — by 4.0 s on July, 3.3 s on
December and 2.0 s on January — and is what the delivered model now uses (`prc.tail_model.PHI_COLUMNS`; `TailModel().fit(train, use_phi=True)`, the default). Why the feature
form beats the anchor form, when finding 8 said the opposite about SCHED versus
AOBT: `taxi_hat` is not a measured anchor, it is a *shrunk estimate*. Anchoring
on it commits the whole prediction to that estimate's error, while as a feature
the model can weigh it against `den`, the stand-occupancy signals and everything
else, and discount it where `phi_support` is thin. The gap is small (0.35 s on
July, 0.28 s on December, 0.84 s on January) — the two forms carry the same
information — but it is consistent in sign across all three folds.

Raw `taxi_hat` alone is much worse than the model (429.2 against 327.3 on July),
which is the expected result and worth recording: phi is a better *anchor* than
`MVT − AOBT_3`, not a substitute for the model.

**Is phi a mixture probability?** Tempting, and *partly* wrong — the precise
version matters. If the block time sat on either `AOBT_3` or `LOBT`, then
`E[y] = (MVT − AOBT) − p·den` with `p = P(block on LOBT)`, so phi *would be*
that probability and the machinery of finding 10 would apply to the bulk. The
winter worker (exp-007) tested it on the oracle's own population (2025-07,
118,633 NM-joined non-LIRF rows with `|lob| ≥ 300 s`): the ratio
`r = (BLOCK − AOBT)/(LOBT − AOBT)` sits on exactly 0 for **0.56 %** of rows and
on exactly 1 for **0.17 %**, and the share grows smoothly with the tolerance
instead of stepping. So there are **no point masses** at 0 and at `lob`.

What that does and does not establish, in their words: it excludes the reading
that made the oracle look like a ceiling on a router — the rows inside a ±60 s
window are the continuum's local density, not a class — but *"a two-component
mixture whose components are themselves diffuse would leave no atom either, and
is not excluded"*. The router built from the best available label-free
classifier of the supposed class loses **117 s** to the model on the 27,835-row
clean-component slice, and the oracle is worth **19.5 s** on the 118,633-row
in-scope slice (261.0 → 241.5), not on the whole fold
(`experiments/exp-007-winter/analysis/q5_mixture.md`).

So: phi is used here as a conditional mean, that use stands, and a *sharp*
two-atom mixture is ruled out — but "the mixture interpretation was refuted"
overstates it, and a diffuse two-component mixture remains open.

Two corrections behind this paragraph, both mine and both the same mistake.
The desk relayed the oracle as "17 s" and the result as "refuted"; I quoted the
relay twice — first the number, then, after saying I would stop, the word. The
artifact says 19.5 s on a stated slice and says explicitly that diffuse mixtures
are not excluded. I also quoted 0.58 % / 0.23 %, which are the 129,914-row
all-airport figures, against a paragraph about the 118,633-row non-LIRF
population. Numbers *and characterisations* from another worker get read from
their artifact, not from the message carrying them.
phi is therefore **used** here as a conditional mean, and that use is what the
evidence supports: it is why `phi_support` matters (a group mean with thin
support) rather than a calibration curve (which is what a probability would
need). But "phi is not a mixture weight" is a stronger statement than the atom
test can carry, and an earlier version of this paragraph made it — contradicting
the paragraph immediately above it, which correctly leaves diffuse mixtures open.
What is established: no sharp two-atom mixture, so the oracle is not a ceiling on
a router, and a router built on that reading loses 117 s. What is *not*
established: that no diffuse two-component mixture underlies phi. Nothing in this
experiment turns on which it is — the conditional mean is fitted the same way
either way — but the claim should not be made. (Sol round 9.)

Per the desk's note, LIRF July numbers are quoted trimmed of the single
date-rollover row; the offset parameterisation of finding 8 already predicts
that row through the noac head, so leaving it in would double-count a fixed
problem as dispersion.
