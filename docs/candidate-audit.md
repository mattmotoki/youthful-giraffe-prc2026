# Candidate audit — what to upload first, and what to expect from the board

The exp-014 candidate audit. Tables and code
behind every number: `experiments/exp-014-robustness/`. Written because there is
no board reading. The available
protection against the 2025-to-2026 shift is to look at what the candidates
actually predict on the ranking months.

## Recommended order

**1. `exp-012-blend-2.parquet`** (`5c57799954cb…`, board estimate 318.02)
**2. `exp-012-blend-1.parquet`** (`541b1b1107fe…`, board estimate 318.81)

The order exp-012 set is right and this audit does not change it. Candidate 2
leads by 0.80 s on the reported estimate and by 0.87 s after the January exposure
is corrected for, so the exposure argument does not reverse it. The 0.07 s of
widening is **not** a finding — it is candidate 2's 9.0 s advantage on one 2025
day against its 21.2 s deficit on another. Both files pass every sanity check,
verified against `submitting.parquet`: no NaN, no negative value, nothing below
the 60 s floor, an identical id set and template row order.

**The reason to upload candidate 2 first is not the reason exp-012 gives.** Its
advantage comes from `july_loo` (+1.96 s on the whole fold); on the `january`
fold it is 0.46 s **worse**. Since January 2026 is the disrupted half, that is
worth knowing before the second upload: if the board comes back with candidate 2
behind candidate 1, the january fold predicted it.

## Expect about 5.8 s worse than the board estimate, on both

The board estimate is built from the `july_loo` and `january` folds. **January
2026 is not January 2025.** 11,968 rows — **7.84 % of the January half** — sit on
days past their airport's 2025 envelope of daily mean take-off-minus-AOBT, of
which **7,564 are on the 18 airport-days that clear it by more than 30 s** and
4,404 clear it by less. The january fold contains **0.73 %**.

Holding each candidate's own measured RMSE on each slice fixed and changing only
that mix:

| candidate | board estimate as reported | **corrected for January's exposure** |
|---|---|---|
| `exp-012-blend-2` | 318.02 | **323.74** |
| `exp-012-blend-1` | 318.81 | **324.61** |

**The base of this is thin and it cuts both ways.** The two exposed-slice RMSEs
come from a **single 2025 airport-day each** — LFPG 2025-01-17 (600 rows) and
EHAM 2025-01-18 (529 rows) — transplanted onto 7,564 and 4,404 January 2026 rows
across 18 airport-days at four airports. It is **too low** because January 2026's
days are far further out (up to 1,350 s past the maximum, against 58 s for the
worse of the two 2025 days) and the analogue's bias grows with the exceedance. It
is **unstable** because a one-day RMSE carries that day's own tail. Read 324–325
as the centre of a wide expectation, not as a prediction; a reading anywhere
between 318 and 326 is inside what this calculation can distinguish.

Against the board on 2026-09-05 — top 277.3, a pack of eleven at 302 to 361 —
that still lands inside the pack and nowhere near the top three. The gap to 277
is a question this audit does not answer.

## The direction of the error, and where

**We should expect to under-predict on January's disrupted days.** On every one
of the three folds, the blend's own out-of-fold predictions are biased *low* on
days past the airport's training envelope: −91 s (january), −107 s (december),
−31 s (july_loo), and −144 s pooled over the three days that clear by more than
30 s. On severe-but-in-range days it is calibrated to within 21 s. The failure is
not severity, it is leaving the envelope.

Concentrated at:

- **EHAM, 3–7 January 2026** — 1,422 rows, the only slice in the file that is
  genuinely extrapolating. Daily mean take-off minus AOBT reaches 2,604 s
  against a 2025 maximum of 1,254 s, and 6 to 8 of the 9 day-state `_z` columns
  are past their training maximum at once. exp-007's worst-hit 2025 day, LSZH
  2025-01-04, had 5 of 9 by its own count. At the 90th percentile the candidate
  predicts 3,453 s where the observable is 4,766 s. LSZH 10 January is a sixth
  day at 6 of 9 and is one of the two LSZH days below.
- **EDDF (7 days) and LSZH (2 days)** — over-shooting, not extrapolating: every
  day-level input is 1,370–1,560 s *inside* its training range, so the model is
  interpolating a severe day and pushing harder than the observable implies.
  These want calibration, not clipping, and they are the shape exp-007's finding
  13 warned about.
- **LFPG 7 January** — tracking; nothing to do.

The model is **already raising January by 120–150 s at EHAM, LTFM, EDDF, LSZH
and EDDM** against the 2025 winter truth, and leaving the five calm airports
within 30 s. The response exists and is aimed correctly. The analogue says it is
not large enough on the days past the envelope.

## What the audit cleared

- **No numeric extrapolation in the main frame.** Across the 41 numeric columns
  of the v2 frame that two of the three members consume, **20** ranking rows in
  the entire file sit outside the twelve-month training range, all in July, none
  in January. And a predict-time clip could not move them even if there were
  more: it is the identity map for a plain tree ensemble, measured at exactly
  0.000 s of prediction change on 2,947 and 6 affected rows for two different
  learners. Of the five members across the two candidates, only `D1-winter` has
  linear leaves and so is the only one a clip can reach.
- **Unseen categories are small and priced sensibly.** 1,589 rows on a stand
  2025 never saw (1,182 of them EDDM's new July apron), 173 on a new operator,
  150 on a new destination, 18 on a new aircraft type. On an unseen stand the
  candidate predicts 48 s *more* than a matched in-range row at the same airport
  and hour, which is the direction an airport-level fallback gives.
- **The LFPG runway retirement is not a pair-level exposure.** 48 ranking rows
  of 39,872 carry a (stand, runway) pair 2025 never contains. 27L was already
  flown from nearly every stand; only its share moved.
- **The file's extremes are earned.** 28 predictions above 20,000 s, all at LIRF,
  27 of them `noac` rows whose own take-off-minus-schedule is of the same size;
  the largest, 118,661 s, is inside 2025's observed `noac` maximum of 131,167 s.
  The 388 `noac` and 4,895 `nonm` rows both sit on their season's 2025
  distribution.

## What the audit found and did not fix

**Both candidates carry a day-level push through the same mechanism exp-007
measured as harmful.** `T-tailv2` (0.4438 of candidate 1) reads
`apt_day_p90_mvt_aobt`, which is past its training maximum on 1,422 January rows
and reaches 5,411 s against a 3,478 s maximum. `D1-winter` (0.25 of candidate 2)
reads the full 25-column day-state block including its nine unbounded `_z`
columns, and its mean prediction sits **+71 s above the other three general
members** on the 3,475 January rows with three or more of them out of range. That
is a mean-prediction difference on unlabelled rows — **not** the same object as
exp-007's +72.1 s of RMSE harm, whose own interval is [+3.5, +149.7] and spans
zero under calendar-day clustering. They are the same order of magnitude, which
is why it needs a separate fold check, and they measure different things. The
flat quarter weight delivers about **+18 s** of the push into candidate 2's
bytes; the two candidates differ by only 1 to 4 s on those rows.

The direction of the push is the one the 2025 analogue calls for, and on the
january fold's most exposed slice candidate 2 is **9.0 s better** than candidate 1
with its bias moving from −166 s to −137 s. It is worse on december's and
july_loo's exposed slices, where the exceedances are small. **This is not settled
by 600, 273 and 512 rows**, and it is the open question the board will answer.

**A predict-time clip cannot reach four of the five members.** Clipping a numeric
column that enters only as a tree split is the identity map for a plain tree
ensemble — measured at exactly 0.000 s of prediction change on 2,947 rows, 6
rows, and the 1,422 exposed ranking rows at `prc.tail_model`'s own parameters,
across LightGBM and CatBoost. Only `D1-winter` has linear leaves. On a matched
linear-leaf model the same clip moves **1,405 of those 1,422 rows**, by 29.8 s on
average and up to 493 s — but the mean signed change is only **−7.9 s**, because
a linear leaf extrapolates in whichever direction its coefficient points and the
moves largely cancel. **So clipping `D1-winter` is not a de-risking move with a
known sign**; it is a two-sided intervention that needs a fold measurement. For
the other four members the exposure shows up as **saturation**, which a clip does
not repair at all. Predict-time clipping needs a fold measurement and is in
neither candidate.

## If a board reading arrives

1. **Anywhere in 318–326 confirms nothing either way.** The re-mix's own base is
   two 2025 airport-days, so it predicts a direction and an order of magnitude
   and cannot be scored against a single reading inside that band. Treat a
   result there as the expected case and spend the next block on score, not on
   re-litigating this document.
2. **Worse than 330**: the January exposure is biting harder than the analogue
   predicts. Look first at the `nonm` upper tail — the candidate's p99 there is
   2,652 s against 6,237 s in the 2025 winter truth — then at EHAM's five
   extrapolating days, then at the January runway response at LTFM, EDDM and
   LFPG, the three airports whose runway mix moved most.
3. **Better than 316**: January 2026's disrupted days are being handled better
   than the 2025 analogue predicts, and the day-level push in both candidates is
   the reason to believe rather than the risk to hedge. In that case the clip
   routed to the members' owners should be re-priced as a risk before it ships.
4. **Either way, the second upload is the cheap experiment.** Candidate 1 and
   candidate 2 differ by 0.79 s in the estimate and differ most on `nm_late` and
   the small regimes. Uploading both turns the open question in the section above
   — whether the day-state push helps or harms January — into a measurement, for
   the price of one submission slot.
