# exp-029 january — RESULTS

Queue lever **L4**, the label-free January correction, measured on **stack-5**.
Read `SPEC.md` first for the reading this experiment took. Every number below is
produced by `code/`, at one thread, in about four minutes of wall time. No model
was fitted: the whole experiment is arithmetic on committed out-of-fold files
plus, after 16:05 UTC, the live data.

---

## 0. Summary

1. **The envelope flag reproduces exp-014 exactly.** The same five 2025
   airport-days, the same row counts, the same exceedances to 1e-13 s, rebuilt
   from a different address; and exp-014's twelve-month day table itself now
   checks out against `data/raw` to 1e-13 s on all 3,650 airport-days (§1).
2. **The mechanism L4 is built on has shrunk by 3.5x, because the base moved.**
   Pooled over the five days the under-prediction is **−83.4 s on candidate 1**
   (exp-014's base, reproduced here to 0.00 s), **−41.4 s on blend-5** and
   **−22.3 s on stack-5**; past 30 s of exceedance, **−143.5 → −74.5 → −41.0**.
   The member that carries the winter and surface families, `V3-allin`, is at
   **+3.2 s** on those days on its own — it already prices them (§2).
3. **No shape ships, and the reason is arithmetic, not implementation.** Nine
   shapes x two honesty settings x two winsorisation settings = 36 measured
   arms, 34 of them reported (§3). The best january clean gain is **+0.020 s, 95 % day-clustered
   [−0.016, +0.076]** (a **sensitivity** interval, not a nominal 95 % one:
   §3); **no positive-gain january arm's interval excludes zero**, which is the
   queue's first kill. The best board gain over all 36 arms is **+0.010 s**
   against a ship bar of **0.2 s of clean board**, and the arm with the largest
   january gain harms july_loo by 0.108 s against a 0.1 s limit — 0.059 s, and
   so inside the limit, once the desk's BOARD READING 3 currency holds the 19
   `noac` rows at the anchor, which leaves the lever failing on the gain alone
   (§3, §3a, §4, §7a).
4. **The january fold cannot decide this question and never could.** An oracle
   additive constant per flagged day, fitted on the day it is applied to, is
   worth **0.029 s** of january clean RMSE. Even predicting all 1,129 flagged
   rows *perfectly* is worth 3.26 s. The gate is 18x the ceiling of the object
   it gates (§4).
5. **What the flagged days' error actually is: 19 rows.** On the five envelope
   days, `nm_bulk` and `nm_late` are 95 % of the rows and 42 % of the squared
   error and stack-5 is nearly unbiased on them (−19.0 and −7.8 s); **19 `noac`
   rows on one day carry 48 %** of it, at +1,464.6 s. A day-level additive
   correction moves 500 unbiased bulk rows to chase a 19-row tail. That is the
   whole negative in one line, and it says the mass belongs to the tail lane
   (L5 / `noac`), not to a day flag (§5).
6. **The extrapolation worry resolves the other way, on the live data.** exp-014
   found candidate 1 handing EHAM's January-2026 excess span to the block offset
   `r` (implied `r` 364–542 s against an ordinary 135). On the served stack-5
   vector the 27 flagged January 2026 airport-days carry a row-weighted implied
   `r` of **−65.8 s**: **one of 27** days clears its airport's 2025 maximum,
   while **16 of 27 (7,104 rows) fall below its 2025 5th percentile**. The
   flagged days *are* out of the 2025 range, downward — over-prediction of taxi
   — which a positive additive correction deepens rather than repairs (§6).
7. **Priced on the January 2026 mix, the whole lever is +0.05 to +0.10 s of
   board** — +0.049 s on slice behaviour pooled over all five analogue days,
   which is partly in sample, and **+0.101 s** on the january fold's own two
   days, which is the out-of-sample pairing (§7b); +0.065 s at its own oracle.
   Against the queue's 1–2 s and 4.3 s ceiling. The width of that bracket is
   itself the finding: neither row is a point value. For it to be worth 1 s,
   stack-5 would have to carry **±161 s** of bias on January 2026's 7,564
   past-30 s rows; the 2025 analogue measures **41 s**. The queue's estimate was
   right for candidate 1 — the same calculation on candidate 1's own behaviour
   gives **+0.77 s** — and it did not travel to stack-5 (§7). Re-quoted in the
   desk's BOARD READING 3 currency the served arm's board gain is **−0.029 s**,
   and the price on the ranking side is unchanged because **none of the 388
   `noac` ranking rows is flagged** (§7a).
8. **The durable deliverable is the flag, not the correction.**
   `oof/ranking_flag.parquet` carries `exceedance` and `flagged` for all 344,841
   ranking rows, built from committed files and then verified against
   `data/raw/ranking.parquet` to 2.3e-13 s. **July 2026 has zero flagged
   airport-days**, so any correction of this family is January-only in fact and
   cannot touch the July half of the board (§8).

**Verdict: FAIL against the queue's gate, on a base that already contains most
of the repair.** The lane is `closed-provisional` under the belief in §9, with
the three conditions that would revive it stated there. The Sol round-1 review
(§11) returned **NEEDS REVISION with the FAIL kept**; its ten findings are
answered above and none of them revives the lever.

---

## 1. The flag, and the gate on it

`code/envelope.py`. The envelope is one line:

```
span               = MVT_TIME_UTC_mvt - AOBT_3_flt          (DEP rows)
day_span_mean[a,d] = mean of span over airport-day (a, d)
envelope[a]        = max of day_span_mean over the fold's TRAINING months
exceedance[a,d]    = day_span_mean - envelope[a]
flagged            = exceedance > 0
```

Both columns are in `ranking.parquet`, so nothing here needs a label. This is
exp-014 `code/q2_days.py`'s `obs_mean` / `max25` / `over_max` under new names.
On the ranking side the envelope is the maximum over **all twelve** 2025 months,
because the served fit sees all twelve.

Three checks, all exact:

| check | result |
|---|---|
| fold-month day means recomputed from `exp-007/work/rows_{fold}.parquet` vs exp-014's `q2_days_2025.csv` | max &#124;Δ&#124; **2.27e-13 s**, row counts identical on all 930 airport-days |
| exp-014's twelve-month table vs `data/raw` (all twelve monthly files, recomputed) | 3,650 airport-days, 0 unmatched, max &#124;Δ day mean&#124; **2.27e-13 s**, max &#124;Δ rows&#124; **0** |
| the five analogue days and their exceedances vs `exp-014/analysis/q2b_fold_days.csv` | identical |

The five 2025 airport-days past their airport's training envelope:

| fold | airport | day | rows | day span mean | envelope | exceedance |
|---|---|---|---|---|---|---|
| january | LFPG | 2025-01-17 | 600 | 1,654.9 | 1,597.4 | **+57.5** |
| january | EHAM | 2025-01-18 | 529 | 1,253.6 | 1,245.4 | +8.3 |
| december | EDDM | 2025-12-24 | 273 | 1,008.9 | 967.1 | **+41.8** |
| december | LEBL | 2025-12-12 | 502 | 926.0 | 921.3 | +4.8 |
| july_loo | LIRF | 2025-07-28 | 512 | 1,209.8 | 1,136.8 | **+73.0** |

January 2026: **27 flagged airport-days, 11,968 rows, 7.84 % of the month**; 18
days and 7,564 rows clear by more than 30 s, up to **+1,350.1 s** at EHAM on
5 January. The january fold's share is **0.73 %**. Both figures match the queue.

**One weakness of the flag, recorded because it is inherited, not introduced.**
The envelope is a maximum over 365 daily means, so it is set by one day per
airport, and three of the ten are set by short days: EDDM by 2025-12-24 (273
rows), LEMD by 2025-09-04 (249), LSZH by 2025-11-21 (357), against 500–650 on an
ordinary day. A softer boundary is measured in §3 as the `stress` shape and does
not help.

---

## 2. Why the number is smaller than the queue's: the base moved

`code/base_moved.py`. exp-014 measured this mechanism on **candidate 1** — the
exp-012 `members-cand1.json` blend of `T-tailv2`, `C2-cat`, `L2-hub`, rotated
with plain global NNLS. exp-012's own `blend.py` cannot run on this machine (it
reads the lost `artifacts/exp-008/frames/`), so candidate 1 is rebuilt from
exp-026's frames, and the gate on the rebuild is exp-014's own per-day table:

| fold | airport | day | rows | bias here | bias in exp-014 | Δ |
|---|---|---|---|---|---|---|
| july_loo | LIRF | 2025-07-28 | 512 | −31.07 | −31.07 | **0.00** |
| december | EDDM | 2025-12-24 | 273 | −304.12 | −304.12 | **0.00** |
| december | LEBL | 2025-12-12 | 502 | +0.89 | +0.89 | **0.00** |
| january | EHAM | 2025-01-18 | 529 | −5.97 | −5.97 | **0.00** |
| january | LFPG | 2025-01-17 | 600 | −166.41 | −166.41 | **0.00** |

Bias is `pred − truth`, so negative is under-prediction. With the rebuild
verified, the three bases on the same rows:

| base | bias, all 5 days (2,416 rows) | bias, the 3 days past 30 s (1,385 rows) |
|---|---|---|
| **candidate 1** — what exp-014 and the queue priced | **−83.4** | **−143.5** |
| **blend-5** — stack-5's anchor | **−41.4** | **−74.5** |
| **stack-5 keep-noac** — the artifact that ships | **−22.3** | **−41.0** |

Per member of blend-5 on the same rows, which is the *why*:

| member | 5 days | past 30 s |
|---|---|---|
| `L2-hub8725` | −120.4 | −193.7 |
| `C2-cat` | −108.9 | −164.4 |
| `N1-nmtail` | −68.5 | −116.6 |
| `F1-fresh` | −66.5 | −110.6 |
| `T-tailv4` | −46.4 | −95.9 |
| `D1-winter` | −37.3 | −59.8 |
| **`V3-allin`** | **+3.2** | **−4.0** |

Candidate 1's three members are the top two of that list and the tail head.
blend-5 added `V3-allin` — the 171-column v3 frame with `WINTER_CORE`, the
surface families and the de-icing column, at 0.21 of the blend outside `nm_late`
and 0.30 inside it — and `D1-winter`, the day-state member. Those are exactly
the day-level information a day-level correction would supply, and they are
already in the served vector. **The repair L4 proposes has largely been shipped
inside the blend.** The stacker then removes half of what is left.

This is a finding in the direction the protocol asks about: it worked, and the
reason it worked is that the day is already a feature.

---

## 3. The shapes, and what each one is worth

`code/correct.py`. Every shape is `corrected = max(pred + c(exceedance), 60)`
with `c(x) = 0` for `x <= 0`, fitted on the flagged 2025 days' SSE-optimal
constants (the mean of `y_true − pred`, winsorised at (0.025, 0.975) in half the
arms), and held out two ways: `loo` leaves the scored **day** out, `lofo` leaves
the scored **fold** out. `lofo` is the honest one for a fold score; on the
**unwinsorised** step shape they coincide, because each fold contributes at most
one day to each band. They do *not* coincide once the residuals are winsorised,
because the quantiles move with the fit rows: january is +0.0202048 under `loo`
and +0.0201695 under `lofo` (Sol round 1, M8).

**36 arms are measured and 34 are reported.** Nine shapes x two hold settings x
two winsorisation settings is 36; the two `stress-all|loo` variants are dropped
from every reported table, because leave-one-**day**-out can only move the five
held days, so that arm collapses onto `stress` and is not the "served
everywhere" measurement its name claims (Sol round 1, M8/M10).
`analysis/arms.csv` carries all 36.

Gains in seconds of `rmse_clean`, positive = RMSE down, base = stack-5 keep-noac
(`analysis/arms.csv`; the full 36-row table with intervals is in
`logs/correct.log`):

| shape | winsor | july_loo | december | **january** | board |
|---|---|---|---|---|---|
| **step** (0–30 s, >30 s bands) — exp-014's first | yes | **−0.108** | +0.013 | **+0.020** | −0.059 |
| **step**, unwinsorised | no | −0.144 | −0.022 | +0.016 | −0.083 |
| **step-half** (the same, shrunk 0.5) | yes | −0.044 | +0.007 | +0.018 | −0.020 |
| **linear** in exceedance — exp-014's second | yes | −0.136 | −0.002 | −0.018 | −0.090 |
| **linear-capped** (clipped to the fitting days' range) | yes | −0.136 | −0.002 | −0.018 | −0.090 |
| **stress** (mine, §3b) | yes | +0.014 | −0.035 | +0.004 | **+0.010** |
| **stress-all** (mine, served everywhere) | yes | −0.106 | −0.123 | −0.010 | −0.069 |
| **step-winter** (repair 1, §3c) | yes | −0.108 | +0.066 | **−0.200** | −0.141 |
| **step-bulk** (repair 2, §3c) | yes | −0.011 | −0.006 | −0.004 | −0.008 |
| **step-regime** (repair 3, §3c) | yes | −0.009 | −0.151 | **−0.836** | −0.320 |

January with the day-clustered 95 % **sensitivity** interval, the four best arms.
Both cluster units are reported: `airport|calendar day` (310 clusters, the
exp-026 `stack5.boot_delta` unit) and the assignment's literal **calendar day**
(31 clusters). The unreached-rollover rows are removed before resampling, so the
interval is around the same quantity `rmse_clean` reports.

| arm | january gain | 95 %, `airport\|day` | 95 %, **calendar day** | rows moved | mean `c` | in-envelope RMSE |
|---|---|---|---|---|---|---|
| `step\|lofo\|winsor` | **+0.020** | **[−0.016, +0.076]** | **[−0.017, +0.077]** | 1,129 | +36.8 s | 337.522 → 337.522 |
| `step-half\|lofo\|winsor` | +0.018 | [−0.007, +0.060] | [−0.007, +0.060] | 1,129 | +18.4 s | 337.522 → 337.522 |
| `step\|lofo\|raw` | +0.016 | [−0.016, +0.062] | [−0.016, +0.064] | 1,129 | +23.0 s | 337.522 → 337.522 |
| `stress\|lofo\|winsor` | +0.004 | [−0.001, +0.012] | [−0.001, +0.012] | 1,129 | +2.7 s | 337.522 → 337.522 |

**Why "sensitivity" and not "95 %".** The review is right that this interval does
not have supported 95 % coverage, and it is not conservative in one direction
only, so both directions belong in the sentence. It is **too narrow** because the
fitted constants are held fixed — the uncertainty of estimating them on five
analogue days, or on three under `lofo`, is not resampled — and because the
endpoint depends on which cluster unit is chosen at all. It is **too
conservative for a positive lower bound** because only two of the january fold's
clusters carry a non-zero correction, so a measured fraction of draws contains
neither and returns exactly zero (§4a). `analysis/review_b3_clusters.csv` carries
both units for all ten reported arms; they agree to the third decimal everywhere
and no verdict in this file turns on the difference.

**No positive-gain january arm has an interval that excludes zero.** One arm's
interval does exclude it, in the other direction: `stress-all|lofo|raw` is
**−0.174 [−0.247, −0.111]** — the one arm that fires below the envelope, and so
the one arm the instrument can resolve, because it moves every row instead of
two clusters (Sol round 1, M7; this sentence read "no january interval excludes
zero" before the review, and that was false). §4a shows that the failure to
resolve the others is a property of the instrument as much as of the arms. The second gate — the in-envelope rows must not be harmed — is met
by construction for every arm except `stress-all`, which is the one arm allowed
to fire below the envelope; it moves the january in-envelope RMSE by
+0.108 s unwinsorised and +0.010 s winsorised, which is the measurement that the
restriction costs nothing and that lifting it costs a little.

The **winner among the shapes is the step**, and it wins for a reason worth
stating: the bias does not increase with exceedance *inside* the flagged
population. Past 30 s the three days run −239.8 (exceedance 41.8), −53.9 (57.5),
**+80.2** (73.0) — the sign flips at the largest exceedance. A line through
that has slope **−0.489 s per second of exceedance** on the january rotation, so
served on January 2026's largest day it would subtract **603 s** where the 2025
evidence says the model under-predicts. `linear-capped` exists only to bound
that; `code/apply.py` refuses to serve the uncapped form.

### 3a. The gate as it now reads, answered line by line

`docs/QUEUE-2026-09-07.md` L4 was revised after this assignment was written: the
0.53 s seed sd was replaced — correctly, since this correction is deterministic
arithmetic and a seed sd is not its noise level — by the interval itself, and a
ship bar of 0.2 s of **clean board** was added. Against the revised text:

| gate | measured | verdict |
|---|---|---|
| **kill 1** — january clean gain non-positive, or the day-clustered interval includes zero | best gain +0.020 s, **[−0.016, +0.076]**; no arm's interval excludes zero | **killed**, but read §4a: this kill fires on arithmetic before it fires on evidence |
| **kill 2** — the correction must not harm the in-envelope rows | in-envelope january RMSE 337.522 → 337.522 on every arm that fires only above the envelope; `stress-all` moves it +0.108 s raw and +0.010 s winsorised | **passed** |
| **ships** — interval clear of zero, gain ≥ 0.2 s of clean board, july_loo harmed by ≤ 0.1 s | board gain **+0.049 to +0.101 s** on the January-2026 population depending on which days' behaviour is used, +0.065 s at its own oracle (§7, §7b); july_loo **−0.108 s** at face value on the best january arm, **−0.059 s** in the currency the desk quotes from 2026-09-07 (§7a) | **fails on the first two**; the july_loo clause passes once the 19 `noac` rows are held at the anchor |

**The queue's own ceiling arithmetic, with the measured input instead of
candidate 1's.** L4 prices the bias term as `(11,968 / 344,841) · bias²` against
the clean board, giving 239 s² → **0.46 s** at the pooled −83 s and 719 s² →
**1.39 s** at −144 s. The same formula on the same base as the shipped artifact:

| bias input | source | s² | clean board |
|---|---|---|---|
| −83 s pooled | candidate 1, exp-014 | 239 | 0.46 s |
| −144 s past 30 s | candidate 1, exp-014 | 719 | 1.39 s |
| **−22.3 s pooled** | **stack-5, §2** | **17** | **0.033 s** |
| **−41.0 s past 30 s** | **stack-5, §2** | 37 (on the 7,564 rows that carry it) | **0.071 s** |

That agrees with §7's re-mix to within 0.01 s by a different route, so the
0.2–1.4 s band in the queue is not wrong about its own arithmetic — it is
reading it off a base two member generations old. (One detail while the formula
is in view: L4 applies the 11,968 row count to the −144 s figure as well, but
that bias belongs to the 7,564 rows past 30 s; using 11,968 over-states that row
by about a third. It does not change the conclusion in either direction.)

### 3b. The shape of my own: is there a day-stress signal at all?

The five-day fitting set is the whole problem, so the shape I added asks the same
question of every airport-day in the three fold months, on a scale-free axis:

```
stress = (day_span_mean - p95_train[apt]) / (p95_train[apt] - p50_train[apt])
```

zero at the airport's 95th training percentile, one an inter-percentile step
above it. Over **930 airport-days and 510,096 rows** the answer is that there is
no signal to fit. stack-5's bias by stress band:

| exceedance vs the p95 envelope | days | rows | bias | RMSE |
|---|---|---|---|---|
| below by more than 100 s | 286 | 189,984 | −2.4 | 247.8 |
| below by up to 100 s | 529 | 261,871 | +0.7 | 245.1 |
| over by 0–50 s | 69 | 35,945 | **+6.4** | 276.8 |
| over by 50–100 s | 23 | 10,901 | −4.1 | 838.7 |
| over by 100–200 s | 15 | 6,942 | **+3.9** | 348.4 |
| over by more than 200 s | 8 | 4,453 | **+11.3** | 570.7 |

The bias does not trend and it is mostly the *wrong sign* for the hypothesis.
Only the `exceedance > 0` bin against the hard maximum shows anything (−22.3 s
on 2,416 rows), and §5 shows that bin is one day.

### 3c. Three repairs, tried before the negative was written

- **`step-winter`** — the two days that disagree most are a summer day (LIRF,
  +80.2) and a winter day (EDDM, −239.8), and January is winter, so fit on the
  winter analogue days only. Under leave-one-fold-out that leaves january's rule
  fitted on december's two days alone: it serves **+239.8 s** to LFPG 2025-01-17,
  whose own optimum is +53.9. January **−0.200 s**. The repair fails because the
  two winter days past 30 s disagree by a factor of **4.4**.
- **`step-bulk`** — serve the band step only to `nm_bulk` rows, 77 % of the
  flagged rows and the only slice with enough of them for a mean to mean
  anything. January **−0.003 s**, board −0.008 s: there is nothing there, which
  is §5's point arriving from the other side.
- **`step-regime`** — one constant per `prc.blend` regime over the flagged days.
  January **−0.836 s**, the worst arm measured, because the `noac` constant is
  fitted on LIRF 2025-07-28's 19 rows at +1,464.6 s and january's `noac` rows do
  not agree. The failure is informative: it is the sign of the tail that does not
  transfer between days, not its size.

---

### 3d. The four required slices, every reported arm

`COMMON.md` requires RMSE overall, on rows ≤ 3600 s, on rows > 3600 s and on
no-NM rows (`nm_missing = 1`) for each fold of each reported result. The review
was right that only three arms carried it. All ten reported arms, all three
folds, are in `analysis/review_b5_decomp.csv`; the january fold, the one the
gate is read on, is here in full:

| january, arm | clean | all rows | ≤3600 s | >3600 s | no-NM (1,432 rows) |
|---|---|---|---|---|---|
| stack-5 keep-noac (base) | 219.130 | 338.382 | 197.484 | 5978.893 | 2836.157 |
| `step\|lofo\|winsor` | **219.110** | 338.369 | 197.527 | 5977.873 | 2836.172 |
| `step\|lofo\|raw` | 219.115 | 338.371 | 197.503 | 5978.319 | 2836.172 |
| `step-half\|lofo\|winsor` | 219.113 | 338.370 | 197.497 | 5978.376 | 2836.162 |
| `linear\|lofo\|winsor` | 219.148 | 338.393 | 197.545 | 5978.237 | 2836.244 |
| `linear-capped\|lofo\|winsor` | 219.148 | 338.393 | 197.545 | 5978.237 | 2836.244 |
| `stress\|lofo\|winsor` | 219.126 | 338.379 | 197.485 | 5978.813 | 2836.157 |
| `stress-all\|lofo\|winsor` | 219.141 | 338.388 | 197.506 | 5978.730 | 2836.172 |
| `step-winter\|lofo\|winsor` | 219.330 | 338.511 | 197.925 | 5975.468 | 2836.253 |
| `step-bulk\|lofo\|winsor` | 219.134 | 338.384 | 197.486 | 5978.925 | 2836.157 |
| `step-regime\|lofo\|winsor` | 219.966 | 338.923 | 198.510 | 5977.372 | 2843.010 |

**The slices say what the total says and add one fact.** Every arm that helps the
total helps it by moving the ≤ 3600 s column the *wrong* way and the > 3600 s
column the right way: the served step costs **0.043 s** on the ordinary rows and
buys **1.020 s** on the tail. That is the trade a day-level constant makes, and
it is why the winner's margin is three hundredths of a second — it pays for a
tail correction with a bulk one, on a population that is 99.3 % bulk.

## 4. The january fold cannot decide this, and the ceiling says so

`analysis/ceilings.csv`. Two oracles, both fitted **on the rows they are scored
on**, so neither is servable; they exist to bound the lever.

| fold | slice | rows | base clean | best additive constant **per flagged day** | those rows predicted **perfectly** |
|---|---|---|---|---|---|
| january | flagged | 1,129 | 219.130 | **+0.029** | +3.259 |
| january | flagged, >30 s | 600 | 219.130 | +0.026 | +2.469 |
| december | flagged | 775 | 205.612 | +0.232 | +2.235 |
| july_loo | flagged | 512 | 285.203 | +0.030 | +6.319 |

**An additive correction on the january fold has a ceiling of +0.029 s of
january clean, which is about +0.011 s of board.** The ship bar is 0.2 s of
clean board, so the fold instrument is 18x too coarse to demonstrate the thing
it is asked to demonstrate, whatever the arm. This is not an argument against
the bar — it is the reason the ship bar has to be read on the January-2026
population, as the queue's own `(11,968 / 344,841) · bias²` formula does and as
§7 does, and not on a fold whose flagged rows are 0.73 % of it. The revision
that replaced 0.53 s with the interval does not change this: an interval around
a quantity whose ceiling is 0.029 s cannot separate a shippable correction from
nothing.

### 4a. The clustered interval cannot decide this kill either, and here is why

`code/interval_power.py`. The correction moves rows inside **2 of the january
fold's 310 airport-day clusters**. A bootstrap that resamples all 310 clusters
draws neither of those two in a measured **14.8 %** of its draws, and the gain in
every such draw is **exactly zero** — a point mass at zero six times the 2.5 %
quantile. **The lower bound is therefore <= 0 for any effect size**, and the
"interval clear of zero" kill fires on the cluster count before it fires on the
evidence. On july_loo, where one cluster is touched, the mass is **37.3 %**.

| arm | fold | clusters | touched | draws with zero effect | gain | all-cluster 95 % | touched-cluster 95 % |
|---|---|---|---|---|---|---|---|
| `step\|lofo\|winsor` | january | 310 | 2 | **0.148** | +0.0202 | [−0.0164, +0.0760] | [−0.0113, +0.0512] |
| `step\|lofo\|winsor` | december | 310 | 2 | 0.123 | +0.0128 | [−0.0176, +0.0555] | [−0.0119, +0.0372] |
| `step\|lofo\|winsor` | july_loo | 310 | 1 | 0.373 | −0.1076 | [−0.3248, 0.0000] | [−0.1076, −0.1076] |
| `stress\|lofo\|winsor` | january | 310 | 2 | 0.148 | +0.0038 | [−0.0007, +0.0122] | [−0.0005, +0.0081] |

The right-hand column is the interval that *could* have power: resample only the
touched clusters and hold the untouched part of the fold fixed, which is exact
because the correction cannot reach it. It still spans zero for the step arm,
and with **one or two clusters to resample it has no power either**. Both forms
agree, and they agree for the same underlying reason as §4: **five airport-days
is not a sample.** This is the sharpest statement of revival condition 2 in §9.

**So the verdict does not rest on the interval.** It rests on the point estimate
(+0.020 s of january clean, +0.05 to +0.10 s of board on the January-2026
population)
against a ship bar of 0.2 s of clean board, and on §2's explanation of why the
point estimate is small. An interval is quoted everywhere here because the gate
asks for one; it is not doing the deciding, and saying otherwise would be
claiming evidence the instrument cannot produce.

Required decomposition (`analysis/decomp.csv`), base and the two served arms:

| fold | variant | rmse_clean | rmse | y ≤ 3600 | y > 3600 | no-NM (n) |
|---|---|---|---|---|---|---|
| july_loo | stack-5 keep-noac | 285.20 | 285.20 | 248.54 | 2,398.03 | 1,122.00 (3,934) |
| july_loo | `step\|lofo\|winsor` | 285.31 | 285.31 | 248.76 | 2,395.08 | 1,122.59 (3,934) |
| july_loo | `step-half\|lofo\|winsor` | 285.25 | 285.25 | 248.64 | 2,396.53 | 1,122.29 (3,934) |
| december | stack-5 keep-noac | 205.61 | 205.61 | 193.04 | 1,794.01 | 663.60 (1,643) |
| december | `step\|lofo\|winsor` | 205.60 | 205.60 | 193.03 | 1,793.62 | 663.16 (1,643) |
| december | `step-half\|lofo\|winsor` | 205.60 | 205.60 | 193.04 | 1,793.81 | 663.38 (1,643) |
| january | stack-5 keep-noac | **219.13** | 338.38 | 197.48 | 5,978.89 | 2,836.16 (1,432) |
| january | `step\|lofo\|winsor` | **219.110** | 338.37 | 197.53 | 5,977.87 | 2,836.17 (1,432) |
| january | `step-half\|lofo\|winsor` | **219.113** | 338.37 | 197.50 | 5,978.38 | 2,836.16 (1,432) |

**The no-NM column is `nm_missing = 1`, not `prc.blend`'s `nonm` regime.** An
earlier draft of this table used the regime, which is 3,595 rows on july_loo and
reads 531.78 — wrong, because `regime_labels` assigns `noac` before `nonm`, so
the 339 rows that are both no-NM and no-aircraft-type leave the slice with the
error they carry. The `exp-027 budget CORRECTION` ledger row is the same mistake
caught the same way, and 1,122.00 here sits beside its 1,123.3 for blend-5 on
the same rows. `code/decomp.py` now derives the slice from
`exp-026 frames.gates`'s `nm_missing` and says why in a comment.

The base reproduces `experiments/exp-026-rebuild/analysis/stack5_rotation.csv`'s
`stack-5 keep-noac` line exactly on all three folds (285.2029 / 205.6118 /
219.1303), which is the check that this experiment is scoring the artifact it
claims to.

---

## 5. What the error on those days actually is

`code/regime_split.py`. The flagged days' rows under `prc.blend`'s regime — the
taxonomy the served stack switches on — for all three bases
(`analysis/regime_split.csv`):

| regime | rows | share of rows | bias: candidate 1 | blend-5 | **stack-5** | RMSE (stack-5) | share of the flagged days' squared error (stack-5) |
|---|---|---|---|---|---|---|---|
| `nm_bulk` | 1,862 | 77.1 % | −51.3 | −29.2 | **−19.0** | 353.7 | 22.2 % |
| `nm_late` | 427 | 17.7 % | −143.2 | −54.4 | **−7.8** | 695.4 | 19.7 % |
| `nonm` | 108 | 4.5 % | −607.2 | −464.2 | −396.5 | 980.8 | 9.9 % |
| **`noac`** | **19** | **0.8 %** | +1,095.3 | +1,464.6 | **+1,464.6** | 5,160.1 | **48.2 %** |

The `noac` column is identical on blend-5 and stack-5 because the shipping
`keep-noac` variant serves the blend-5 anchor on `noac` rows by construction.

Per day, stack-5, in exp-007's three-way split:

| day | `NM bulk` | `NM tail` | `no-NM` |
|---|---|---|---|
| EDDM 2025-12-24 | n=223, **−55.5** | n=3, −1,812.2 | n=47, −1,014.0 |
| LFPG 2025-01-17 | n=543, **+0.9** | n=33, −960.6 | n=24, −46.9 |
| EHAM 2025-01-18 | n=487, **+10.7** | n=5, −259.8 | n=37, +161.2 |
| LIRF 2025-07-28 | n=458, **+105.0** | n=35, −996.1 | n=19, +1,464.6 |
| LEBL 2025-12-12 | n=502, **−13.7** | — | — |

**The "under-prediction past the training envelope" is not an envelope effect.**
Ninety-five per cent of the flagged rows are NM-joined bulk and late rows on
which stack-5 is nearly unbiased; the whole pooled −22.3 s is three tail
populations that are known to carry most of the squared error everywhere
(`COMMON.md`: no-NM rows are 2.06 % of departures and 60 % of the squared
error), and half of it is **19 `noac` rows on one July day**. An additive
correction applied to a flagged day moves ~500 unbiased bulk rows to chase them.

That is the diagnosis, and it also names where the mass really is: the tail
lane. `docs/QUEUE-2026-09-07.md` L5 (exp-023, the `noac` 1–8 h band) already owns
it, with an oracle an order larger than anything L4 can reach.

**On candidate 1 the same table reads differently** and that is consistent with
§2: its `nm_bulk` bias on the flagged days is **−51.3** against stack-5's −19.0
and its `nm_late` bias is **−143.2** against −7.8, so in 2025 there genuinely was
a bulk day-level component and the newer members have taken almost all of it.
What is left on stack-5 is the tail, and the tail is not a day-level object: the
same 19 `noac` rows are +1,464.6 s on both bases.

---

## 6. The extrapolation question, answered on the live data without a label

The 2025 analogue tops out at 73 s of exceedance and January 2026 reaches
1,350 s, so no fitted function is constrained where the mass is. `code/implied_r.py`
asks the question directly instead. With

```
span = MVT - AOBT      r = BLOCK - AOBT      y = span - r      r_hat = span - pred
```

`r` is an administrative offset between the airport's block time and the Network
Manager's off-block time. It is not a congestion quantity, so a day whose taxi
time explodes should not move it. exp-014 found candidate 1 doing exactly that on
EHAM 3–7 January 2026 — implied `r` of 364–542 s against EHAM's ordinary 135 s.
On the **served stack-5 vector**
(`submissions/candidates/exp-026-stack-5.parquet`) against each airport's 2025
daily-mean `r` range from all twelve months:

The check is **two-sided**, because a one-sided range count is exactly the error
exp-014's own round-1 review caught on its day-state exposure figure (its
finding M8):

| January 2026 | airport-days | rows | with an observed `r` | row-weighted implied `r` | ABOVE the 2025 max / p95 | BELOW the 2025 min / p05 |
|---|---|---|---|---|---|---|
| flagged | 27 | 11,968 | 11,108 | **−70.0 s** | 1 day, 391 rows / 1 day, 391 rows | **4 days, 1,252 rows / 16 days, 7,104 rows** |
| in envelope | 283 | 140,751 | 139,158 | +20.0 s | 1 day, 319 rows / 22 days, 8,663 rows | 1 day, 396 rows / 21 days, 9,857 rows |

The weights are the rows that **have** an observed `r`. Weighting by all rows
instead gives −65.8 s and +19.8 s, which mixes two denominators, since
`r_hat_mean` already skips the rows with no `AOBT_3_flt` (Sol round 1, M9). The
direction and the size are unchanged; the corrected figure is the one to quote.

EHAM 3–7 January 2026 on stack-5 reads +321, +112, −246, +147, +73 against a
2025 median of +147 and a maximum of +262; only 3 January clears the top, by
59 s. **The model is not handing January 2026's excess span to the offset, and where
the flagged days sit unusually they sit low, not high**: below the airport's 2025
5th percentile on **16 of 27 days (7,104 rows)** and below its 2025 **minimum**
on **4 days (1,252 rows)**. A negative implied `r` means the prediction exceeds
the observed take-off-minus-AOBT, i.e. **over**-prediction of taxi, which is the
opposite of what a positive additive correction repairs.

**What this does and does not establish** (Sol round 1, M9, and the reviewer is
right to press it). There is no true `r` for 2026; `r_hat = span − pred` is a
diagnostic conditional on the prediction, not a measurement, so "below the 5th
percentile" is not "outside the 2025 range" and only the four days below the
minimum are that. The honest statement is: **under the assumption that the
administrative offset is stable between 2025 and 2026 — the same assumption
exp-014 made to raise the extrapolation worry in the first place — the
diagnostic points toward over-prediction risk on the flagged days, not
under-prediction.** It does not prove that L4 would raise the ranking RMSE. It
does remove the reason to believe L4 would lower it, which is what the section
was asked for.

The same decomposition on the five 2025 days is the consistency check, since
`r_hat − r_true` is exactly minus the bias:

| fold | airport | day | true `r` | implied `r` | difference | that airport's 2025 max `r` |
|---|---|---|---|---|---|---|
| december | EDDM | 2025-12-24 | −802.5 | −723.8 | +78.8 | +62.7 |
| january | LFPG | 2025-01-17 | −323.8 | −269.6 | +54.2 | +58.7 |
| december | LEBL | 2025-12-12 | −28.2 | −14.5 | +13.7 | +47.4 |
| january | EHAM | 2025-01-18 | +174.3 | +166.3 | −7.9 | +262.1 |
| july_loo | LIRF | 2025-07-28 | −624.2 | −651.0 | −26.8 | +84.0 |

(These use the NM-joined rows only, since `r` needs `AOBT_3_flt`, which is why
the differences are smaller than §2's biases — §5 says where the rest lives.)

---

## 7. What the lever is worth on the mix that matters

`code/remix.py`. exp-014's re-mix, holding each slice's measured behaviour fixed
and changing only the mix from the january fold's to January 2026's:

| slice | january fold rows | share (fold) | share (Jan 2026) | stack-5 RMSE | stack-5 bias |
|---|---|---|---|---|---|
| in envelope | 152,575 | 99.27 % | 92.16 % | 216.67 | +8.27 |
| 0–30 s over | 529 | 0.34 % | 2.88 % | 262.24 | +2.89 |
| >30 s over | 600 | 0.39 % | **4.95 %** | 840.43 | **−40.97** |

The two exposed rows use all five analogue days pooled; using the january fold's
own two days instead gives +18.7 and −53.9. **The re-mix is unstable in exactly
the way exp-014 said** — the >30 s RMSE of 840 s is LIRF 2025-07-28's own tail
transplanted onto 7,564 January 2026 rows — so the level move it implies
(board 259.39 → 284.34, **+24.96 s**) is an order of magnitude wider than
exp-014's +5.80 and neither number should be quoted as a prediction. **The board
has already measured this premium twice and it is neither**: stack-4 landed
+4.276 s above its clean estimate and blend-5 +6.83 s (`docs/LEDGER.md`, BOARD
READING 1 and 2). So the transplant over-states the level by roughly 4x here and
exp-014's version over-stated it by 1.5x — the same failure in the same
direction, from the same cause, and the reason no level number from this family
belongs in a decision. What the
re-mix *is* good for is the **difference** a correction makes, where the slice
RMSE is common to both sides and cancels:

| correction | `c` on 0–30 s | `c` past 30 s | january gain, 2026 mix | **board gain, 2026 mix** |
|---|---|---|---|---|
| served (`step\|lofo\|winsor`), five-day behaviour — **partly in sample**, see §7b | +13.95 | +56.99 | +0.110 | **+0.049** |
| served (`step\|lofo\|winsor`), january-only behaviour — **out of sample** | +13.95 | +56.99 | +0.251 | **+0.101** |
| oracle band constants on **stack-5**, in sample | −2.89 | +40.97 | +0.147 | **+0.065** |
| oracle band constants on **blend-5**, its own base | −3.19 | +74.52 | +0.484 | +0.214 |
| oracle band constants on **candidate 1**, its own base — what exp-014 priced | +2.63 | +143.52 | +1.768 | **+0.769** |

And the inverse question, which is the one to carry forward: how large would the
bias on January 2026's 7,564 past-30 s rows have to be for the correction to be
worth a given amount of board?

| board target | required &#124;bias&#124; on those rows |
|---|---|
| 0.53 s (the seed sd) | **117 s** |
| 1 s | **161 s** |
| 2 s | 228 s |
| 4.276 s (the whole gap) | 334 s |

**Measured on the 2025 analogue: 41 s.** On candidate 1 it was 143.5 s, which is
between the 0.53 s and 1 s rows — so the queue's "1–2 s, ceiling 4.3 s" was a
fair reading of exp-014's base and does not survive the move to stack-5.

### 7b. Which of those two rows is out of sample, and why it matters

The review's first blocking finding, accepted in full. The served constants for
the january rotation are fitted on the **July and December** analogue days. The
+0.049 s row scores them against slice behaviour **pooled over all five** days,
so the July and December rows help fit the constants and then help score them.
It is not an out-of-sample number and this file called it one.

The out-of-sample pairing is the january fold's own two days, whose rows never
touched the fit. `code/review_round1.py` computes both from the same vectors:

| exposed-slice behaviour from | january gain, fold mix | board gain, fold mix | january gain, 2026 mix | **board gain, 2026 mix** |
|---|---|---|---|---|
| five days — partly in sample | +0.010 | +0.004 | +0.110 | **+0.049** |
| **january only — out of sample** | +0.020 | +0.008 | +0.251 | **+0.101** |

**The honest number is the larger one, and the lever still fails.** +0.101 s is
half the 0.2 s ship bar and a fifth of the 0.53 s seed sd. The reason it is
larger is arithmetic and not evidence: on the january fold's own two days the
past-30 s bias is −53.9 s against the pooled −41.0 s, and the served constant
+56.99 nearly cancels it — a coincidence of two days, which is the same
five-day-sample fragility §7 already reports in the level. **Neither row should
be quoted as a point value.** The bracket **+0.05 to +0.10 s of board** is the
result, and its width is the finding.

The queue's `(11,968/344,841)·bias²` route agrees with the wider bracket too:
at −53.9 s on the 7,564 past-30 s rows it gives 0.12 s, between the two rows.

---

### 7a. Re-priced under the desk's BOARD READING 3 and 5, and one gate clause flips

The desk set a new currency after this experiment's arms were measured
(`docs/LEDGER.md`, BOARD READING 3, 2026-09-07): **youthful-giraffe_v3 =
stack-5 scored 264.3861, rank 1 of 63**, and of its two predictions the
face-value one (clean 258.04 + 4.28 = 262.0) missed by +2.35 s while the one
with the noac fold gain given back (clean 260.31 + 4.28 = 264.59) missed by
−0.20 s. Same pattern on v2. The rule from here is **quote every candidate as
clean-with-noac-at-the-anchor + 4.28**: bulk fold gains transfer to the board,
`noac` fold gains do not.

`code/board_reading3.py` re-quotes the arms already measured in this currency.
It adds no arm and refits nothing; it substitutes blend-4's rotated prediction
on the `noac` rows — `experiments/exp-026-rebuild/code/noac_discount.py`'s own
substitute — on **both sides of the difference**, so the correction is credited
on the bulk rows only. The base reproduces the desk's number exactly:

| candidate | currency | july_loo | december | january | clean board | quote (+4.28) |
|---|---|---|---|---|---|---|
| stack-5 (base) | face value | 285.2029 | 205.6118 | 219.1303 | 258.0372 | 262.32 |
| stack-5 (base) | **noac on blend-4** | 287.7687 | 205.3764 | 220.9765 | **260.3115** | **264.59** |
| `step\|lofo\|winsor` | face value | 285.3106 | 205.5990 | 219.1101 | 258.0959 | 262.38 |
| `step\|lofo\|winsor` | **noac on blend-4** | 287.8279 | 205.3636 | 220.9565 | **260.3404** | **264.62** |
| `step-half\|lofo\|winsor` | noac on blend-4 | 287.7890 | 205.3693 | 220.9589 | 260.3174 | 264.60 |
| `stress\|lofo\|winsor` | noac on blend-4 | 287.7656 | 205.4118 | 220.9727 | 260.3082 | 264.59 |
| `step\|lofo\|winsor` | **quarter (BOARD READING 5)** | 285.2743 | 205.5990 | 219.1101 | **258.0735** | **262.35** |
| `step-half\|lofo\|winsor` | quarter (BOARD READING 5) | 285.2292 | 205.6048 | 219.1126 | 258.0467 | 262.33 |

260.3115 and 264.5915 are the desk's 260.31 and 264.59, so the currency is
reproduced inside this experiment and not merely quoted.

**Where the correction meets the discounted rows.** Of the 2,416 flagged fold
rows, **19 are `noac`, and all 19 are on one July 2025 day at LIRF** — the same
19 rows that carry 48.2 % of the flagged days' squared error (§5). The january
fold has **none**. So the re-quote does two things and no more:

1. **The january number does not move.** +0.0202 [−0.0164, +0.0760] at face
   value, **+0.0200 [−0.0162, +0.0755]** discounted. The kill in §3a stands
   unchanged, and it never rested on the `noac` rows.
2. **The july_loo harm halves, and the gate's third clause flips.** The step
   arm costs july_loo **0.108 s at face value and 0.059 s discounted**, because
   half of the damage it did was to rows the board does not price. The ship
   condition reads "july_loo harmed by ≤ 0.1 s"; under the currency the desk now
   quotes, that clause **passes**. It failed for the wrong reason before.

The verdict does not change. Under the same currency the arm's **board gain is
−0.029 s** — still negative, still an order of magnitude from the 0.2 s bar —
and kill 1 fires on the january gain and its interval exactly as before. The
lever now fails on the gain alone rather than on the gain and the harm, which is
a cleaner statement of the same negative: **the correction is not too damaging,
it is too small.**

**There is no single `noac` exchange rate, and this lever does not need one.**
Two instrument uploads the same morning moved only the `noac` rows: v4
(exp-002's fold-optimal head, a 1,857 s RMS move) scored **+9.26 s** against v2
and v5 (the tail-v4 head's change undone, a 371 s move) scored **+0.33 s** where
the fold said 1.19 s. That first read as "a quarter of face", and the desk's
exact decomposition then corrected it (`BOARD READING 5 CORRECTION`,
`internal decision record` score-feedback rule): for a shift `s` against a reference, `ΔSSE = Σs² +
2Σs·e`, so the board returns `ρ = Σs·e/Σs²` — **+0.145 for v4, +0.066 for v5,
about −1.07 for the tail-v4 change read from its own base.** A `noac` fold gain
transfers only when the head points the right way on the 2026 rows, and the fold
cannot tell which way that is. There is a rate per change, not a rate.

**For this lever the question does not arise on the served side at all.** The
correction never moves a `noac` ranking row — none is flagged — so no `ρ` has to
be guessed. Its only `noac` contact is 19 rows on one July 2025 day, and that
moves the july_loo *fold* number, which is not a board number. Those three
currencies bracket the july_loo harm at **0.059 to 0.108 s**, inside the gate's
0.1 s clause at the discounted end and outside it at face value; a middle
reading, the `noac` rows keeping a quarter of the correction, gives **0.071 s**
and a board gain of −0.036 s, and it is quoted here as an interpolation and not
as an exchange rate. Under every one of them the january gain is **+0.020 s** and
the board gain is **negative**, which is what the verdict rests on.

**On the served population the discount changes nothing at all.** The ranking
file holds 388 `noac` rows (108 in January 2026, 280 in July); **zero of them
are flagged**, because every one sits on an in-envelope airport-day. The
correction's population and the discounted population do not intersect, so §7's
+0.05 to +0.10 s served / +0.065 s oracle is already a BOARD READING 3 and 5
number and needs no restatement. `apply.py` now holds `noac` rows at the base regardless
(`--credit-noac` restores the old behaviour); on this ranking file the guard
moves no row, and it is there so the object stays correct under the currency if
a later month does put a `noac` row on a flagged day.

Addresses: `analysis/board_reading3.csv`, `board_reading3_folds.csv`,
`board_reading3.json`, `logs/board_reading3.log`.

---

## 8. The ranking side

`code/ranking_flag.py`, `code/apply.py`.

**The committed files are enough, and this is the answer to the assignment's
question about `ranking_daystate.parquet`.** That file carries `apt` and `day`
for all 344,841 ranking rows but no daily span mean and no `MVT_ID_mvt`;
`exp-014/analysis/q2_days_2026.csv` carries the daily span mean and the
exceedance for all 620 ranking airport-days but no row identity. The join is
**positional** — `exp-007/code/q3b_ranking_check.py` wrote the day-state file as
month-sorted blocks with the ranking frame's index preserved — and that
assumption is checked three ways before anything is written:

| check | result | control |
|---|---|---|
| `ds_n_dep` equals the row count of each airport-day group | exact on 620 days | — |
| `q2_days_2026.csv`'s `n` equals the same | exact on 620 days | — |
| `has_nm` from `N1-nmtail-v4-ranking.parquet` averaged inside each group reproduces `ds_nm_rate` | max &#124;Δ&#124; **0.0030** | row order shuffled: **0.3990** |

`oof/ranking_flag.parquet`, 344,841 rows, sha256
`71bc81d1541ea35a845cae8c2c1c4835545c08faf862c6137fef0659f2627168`. Since the
data landed, `code/apply.py --source auto` derives the same flag from
`data/raw/ranking.parquet` and cross-checks it: **0 unmatched rows, max
&#124;Δ exceedance&#124; 2.27e-13 s**. The committed path and the live path agree.

Rows the correction would touch, under the served january rule
(`step|lofo|winsor`: +13.95 s on 0–30 s, +56.99 s past 30 s;
`analysis/ranking_touched.csv` has all 27 rows):

| airport | day | rows | exceedance | `c` | total |
|---|---|---|---|---|---|
| EHAM | 2026-01-05 | 208 | 1,350.1 | +57.0 | 11,854 s |
| EHAM | 2026-01-04 | 336 | 1,341.5 | +57.0 | 19,148 s |
| EHAM | 2026-01-03 | 391 | 1,076.8 | +57.0 | 22,283 s |
| EHAM | 2026-01-07 | 194 | 857.7 | +57.0 | 11,056 s |
| EHAM | 2026-01-06 | 293 | 688.3 | +57.0 | 16,698 s |
| LSZH | 2026-01-10 | 278 | 414.2 | +57.0 | 15,843 s |
| EDDF | 2026-01-29 | 505 | 310.3 | +57.0 | 28,780 s |
| … 11 more days past 30 s, then 8 days in the 0–30 s band | | | | +14.0 | |

**11,968 of 152,719 January rows (7.84 %) across 27 airport-days.** How much
prediction that adds depends on which fit is served, and there are two, which is
worth being explicit about:

| rule | `c` on 0–30 s | `c` past 30 s | total added | mean over the month |
|---|---|---|---|---|
| the january **rotation** rule (`step\|lofo\|winsor`, fitted on december's and july's days) — the one whose fold gain §3 measures | +13.95 | +56.99 | **492,502 s** | +3.22 s/row |
| the **serve-time** fit (`code/apply.py --arm step`, all five analogue days, since at serve time no fold is held out) | −1.78 | +49.68 | 367,933 s | +2.41 s/row |
| the serve-time fit shrunk 0.5 (`--arm step-half`, `apply.py`'s default) | −0.89 | +24.84 | 183,966 s | +1.20 s/row |

The table above uses the first, so that the rows it names are the rows whose
gain §3 reports.

**July 2026 has zero flagged airport-days.** The correction is not January-only
by construction — it fires wherever the exceedance is positive — but on this
ranking file it is January-only in fact, so it cannot touch the 55.7 % of the
board that July carries.

`code/apply.py` is written, runs, and is **not adopted**. It takes any
`MVT_ID_mvt`/`y_pred` parquet, so it composes on stack-5 today and on the
exp-028 window arm later without change. It refuses the uncapped `linear` shape
for the reason in §3.

---

## 9. The instrument note, and the belief

**Every fold number in this repository shares an indirect label channel**
(`internal decision record` nested validation rule, from Sol's round 1 on exp-026): a scored fold's labels
reach a stacker and its NNLS weights through the *other* folds' member
out-of-fold files, whose models were fitted on month sets that include the
scored month. stack-5's 258.04 clean board and the 219.13 january clean number
this experiment corrects are instrument-relative, not clean out of sample, and
so is every gain measured against them here.

**A post-processing step fitted on five 2025 days adds no channel of its own,
but it does not escape the base's — and this file claimed it did.** Sol's round-1
finding B2 is accepted. Its parameters are two constants estimated from
`y_true − pred` on days in folds other than the scored one, so no january
*label* enters the fit directly under `lofo`. But `pred` there is stack-5's
out-of-fold prediction, and stack-5's July and December predictions were fitted
on rotations that include the january fold (`stack5.py` trains fold `f` on the
other two). January labels therefore reach the constants through the residuals
they are estimated from. The channel is in the difference as well as in the
level, and nested validation rule says exactly this path.

So **+0.020 s is instrument-relative, not a clean out-of-sample gain**, and so
is every fold number in this file. Closing the channel needs the member
rotations rebuilt clean, which is a larger job than this lever is worth at
+0.02 s. Under `loo` there is a second, direct channel — a january day corrected
by a rule that saw the other january day — which is why `lofo` is the headline
and why the two are reported side by side.

**The belief, stated as one.** L4 as specified — an additive, day-level,
label-free correction fitted on the 2025 envelope days — is worth **+0.05 to
+0.10 s** of board estimate on stack-5 (§7b: +0.049 s on five-day behaviour,
+0.101 s on the january fold's own, and the width of that bracket is itself the
finding), not the queue's 1–2 s, because the
under-prediction it repairs fell from −143.5 s to −41.0 s when `V3-allin` and
`D1-winter` entered the blend, and because 95 % of the rows it moves are already
unbiased. The lane is set **`closed-provisional`**; Matt confirms or reverses.

**Three conditions would revive it, and each is cheap to test:**

1. **A base with a larger envelope-day bias.** The whole pricing is
   §7's table and it is a function of one measured number. If the exp-028 window
   arm, or any later member, restores an envelope-day bias above ~117 s, the
   correction is worth the seed sd again. `code/apply.py` and
   `code/base_moved.py` re-price it in one command against any new base, and
   `oof/ranking_flag.parquet` does not have to be rebuilt.
2. **More than five analogue days.** Every negative here is stated on a fitting
   set of five airport-days whose two largest disagree by a factor of 4.4 and
   one of which flips sign. exp-014 named the fix — "a leave-one-month-out sweep
   over all twelve months" — and it is now affordable: the data is on disk, and
   what it needs is member out-of-fold predictions for the nine months that have
   none. That is a member refit, not arithmetic, so it belongs behind the nested validation rule
   clean-instrument lever, which needs the same refits and would pay for both.
3. **A January mechanism this flag does not capture.** The flag is one scalar
   per airport-day — how far the day's mean take-off-minus-AOBT clears a
   twelve-month maximum — and §3a shows that scalar carries no bias signal on
   stack-5 over 930 days at any threshold. That is a negative about *this*
   flag, not about January 2026 being different: the month genuinely is
   (`docs/shift-2026.md`), and §7's re-mix puts the level move at +25 s of board
   with a wide error either way. Two candidates the flag misses are visible in
   the evidence here and neither is a day-level additive shift. **(a) The
   `noac` tail**: 19 rows carry 48 % of the flagged days' squared error and
   their sign does not transfer between days (§5, §3c repair 3), which is L5's
   object with a bigger and better-instrumented handle. **(b) Within-day
   structure**: the flag averages a day whose exceedance runs to 1,350 s, and
   §6 shows the served implied `r` on those days is *below* the airport's 2025
   median, not above it — so on the biggest January 2026 days stack-5 leans
   toward over-prediction, which no positive additive correction repairs. An
   hour-level or queue-level version of the flag is the shape that could still
   find something, and it needs the served vector plus `ranking.parquet`, both
   now on disk.

**What I would not do.** Serve the correction to buy 0.05 s. It moves 11,968
rows on the strength of five days, it is inside the noise of the fold instrument
in both directions, and §5 says the error it aims at belongs to a lane with a
larger and better-instrumented handle on it.

---

## 10. Addresses

| artifact | path | note |
|---|---|---|
| ranking flag | `oof/ranking_flag.parquet` | 344,841 rows, sha256 `71bc81d1…2627168` |
| corrected out-of-fold | `oof/exp-029-july_loo.parquet` | 190,713 rows, sha256 `4513cff8…0d96edbe` |
| | `oof/exp-029-december.parquet` | 165,677 rows, sha256 `a021f75b…2c512b3` |
| | `oof/exp-029-january.parquet` | 153,706 rows, sha256 `1d3ee2e6…7bb080fdc` |
| | | all three: `MVT_ID_mvt, fold, y_true, y_pred` (the `step\|lofo\|winsor` arm) plus `y_pred_base`, `correction`, `exceedance`, so the file is self-checking and a later arm can be re-scored against it |
| all 36 arms (34 reported) | `analysis/arms.csv`, `analysis/arm_params.json` | |
| Sol round 1 answers | `analysis/review_b1_remix.csv`, `review_b3_clusters.csv`, `review_b5_decomp.csv`; `code/review_round1.py` | |
| BOARD READING 3 re-quote | `analysis/board_reading3.csv`, `board_reading3_folds.csv`, `board_reading3.json` | |
| ceilings | `analysis/ceilings.csv` | |
| base comparison | `analysis/base_moved.csv`, `analysis/base_moved_pooled.csv` | |
| re-mix | `analysis/remix.json`, `analysis/remix_value.csv`, `analysis/remix_bases.csv` | |
| implied `r` | `analysis/implied_r_2026.csv`, `analysis/implied_r_analogue.csv` | |
| touched ranking rows | `analysis/ranking_touched.csv` | |
| decomposition | `analysis/decomp.csv` | |
| logs | `logs/*.log` | one per script |
| intermediates | `analysis/fold_rows.parquet` (18 MB), `analysis/corrections.parquet` (7.5 MB) | git-ignored; `code/envelope.py` then `code/correct.py` rebuild them in about 40 s, and everything else in this table is built from them |

**Order to reproduce:** `envelope.py` → `base_moved.py` → `correct.py` →
`regime_split.py` → `remix.py` → `decomp.py` → `ranking_flag.py` →
`implied_r.py` (the last needs `data/raw` and the served vector);
`apply.py --arm step-half` is the dry run. One thread, about four minutes.

Base: `experiments/exp-026-rebuild/oof/stack5-{fold}.parquet`, `y_pred`
(keep-noac), whose clean numbers reproduce `stack5_rotation.csv` exactly.
Served vector for §6: `submissions/candidates/exp-026-stack-5.parquet`.

**Reviewer:** see §11.

---

## 11. The review, and what it changed

`review/round1-sol.md`, with the raw 788 KB transcript beside it.
**`gpt-5.6-sol` at `xhigh`, fresh context, read-only, 2026-09-07 16:29–17:04
local, 344,544 tokens. Verdict: NEEDS REVISION. FAIL for L4 stands.**

Ten findings, six blocking. **All ten accepted; none disputed.** Every one is a
claim or a convention rather than a defect in a measured vector: the reviewer
reproduced the oracle ceiling (0.0286046 s), the `rmse_clean` arithmetic, the
positional join and the four out-of-fold files exactly, and its own
recomputation of the january interval on calendar days lands on the same
endpoint and the same verdict.

| # | finding | what changed here |
|---|---|---|
| B1 | `+0.049 s` is partly in sample | §7b: the out-of-sample pairing is **+0.101 s**; the headline is now the bracket +0.05 to +0.10 s |
| B2 | the difference still carries the nested validation rule channel | §9: `+0.020 s` relabelled **instrument-relative**; the claim that the channel was only in the level is withdrawn |
| B3 | not a nominal 95 % interval, and not the assignment's cluster unit | §3: both units reported, relabelled a **sensitivity** interval, both directions of its bias stated |
| B4 | one ledger row per measured arm | one row carrying all 36 arms; the reading is written in `review/round1-sol.md` |
| B5 | the decomposition covered three arms | §3d: all ten reported arms × three folds |
| B6 | no §11 | this section |
| M7 | "no january interval excludes zero" is false | §3: `stress-all\|lofo\|raw` is −0.174 [−0.247, −0.111]; reworded to "no positive-gain arm" |
| M8 | `loo` and `lofo` coincide only for raw fits | §3 |
| M9 | the implied-`r` conclusion is too strong, the weighting mixes denominators | §6: **−70.0 s** on the 11,108 rows with an observed `r`; the conclusion is now conditional on a stable offset |
| M10 | 36 arms, not 34; the climb field is the gap fraction | §3 and `climb.log` |

**Two things the reviewer could not have had.** The code and this file changed
twice while it was reading — the winsorisation bounds moved inside the hold-out
at 16:37–16:38, and the no-NM slice was corrected from the `nonm` regime to
`nm_missing = 1` at 16:44 — and the desk's BOARD READING 3 landed after it
started, so §7a is not in its scope.

**The reviewer's own summary is the one this file adopts:** *"No tested
correction passes the gate; the fold estimate is instrument-relative, and the
available five-day sample does not support a precise board-value range."*
