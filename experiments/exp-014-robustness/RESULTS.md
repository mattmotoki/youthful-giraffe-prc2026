# exp-014 robustness — what the candidates predict on January and July 2026

Worker `robustness`, 2026-09-05. Audit of the shipped submission candidates
against the ranking months, with no board reading available. Every number here
is reproducible from `code/` against the live `ranking.parquet`; the tables are
in `analysis/`.

**Candidate 2 landed while this audit was running** (exp-012, 16:56). Everything
below covers both candidates; where they differ the difference is stated.

## The short version

1. **The v2 frame does not extrapolate on the ranking months at all.** Across
   its 41 numeric columns, **20** ranking rows in total sit outside the
   twelve-month training range — all in July, none in January, and the largest
   single-column exposure is 9 rows on `arr_landings_60m_after`. The
   numeric-clipping repair has nothing to act on in the artifact being audited.
2. **The extrapolation is entirely in the two members that carry day-level
   statistics.** `T-tailv2`'s `apt_day_p90_mvt_aobt` is past its training
   maximum on **1,422 January rows** (up to 5,411 s against a 3,478 s maximum),
   and `D1-winter`'s day-state `_z` block is outside its airport's training
   range on **19,895 January rows — 13.0 % of the month** — of which 15,842 are
   above the maximum and the rest below the minimum. At EHAM 32.7 % of January
   rows are above, with up to **8 of the 9 columns over at once**; exp-007's
   worst-hit 2025 day, LSZH 2025-01-04, had 5 of 9 (that count is exp-007's, not
   recomputed here).
3. **The blend under-predicts on days past the training envelope, on all three
   folds.** Pooled bias −83 s over the five such days the folds contain, −144 s
   over the three that clear it by more than 30 s. January 2026 has **18 such
   airport-days and 11,968 such rows (7.84 % of the month)** against the january
   fold's **0.73 %**.
4. **The board estimate is optimistic by about 5.8 s for that reason alone.**
   Holding each candidate's own per-slice RMSE fixed and changing only the share
   of rows on exposed days moves candidate 1 from 318.81 to **324.61** (+5.80)
   and candidate 2 from 318.02 to **323.74** (+5.72). It is a lower bound in one
   direction and rests on a thin base in the other: each of the two slice RMSEs
   comes from a **single 2025 airport-day** (see the re-mix section).
5. **The re-mix does not change the upload order.** Candidate 2 leads by 0.80 s
   as reported and by **0.87 s** re-mixed. Candidate 2 first stands.
6. **Both candidate files are clean**, checked against `submitting.parquet`:
   identical id set, template row order, float64, all finite, none below the
   60 s floor. 28 predictions above 20,000 s, of which **27** are LIRF `noac`
   rows whose own take-off-minus-schedule is of the same size and one is an
   `nm_bulk` row whose take-off minus AOBT is 86,340 s; the largest is inside
   the 2025 observed maximum. 350 rows have members disagreeing by more than
   1,800 s, 194 of them `noac`.

7. **Seven ranking rows have no NM off-block but a live NM flight id.** They are
   the difference between the assignment's 5,290 no-NM departures and the 5,283
   the regime labeller sees. They fall in `nm_bulk`, the residual guard cannot
   act on them, and no other slice of this audit names them (see question 3).

## Question 1 — out-of-range inputs

### The v2 frame (`C2-cat` 0.2227 + `L2-hub` 0.3351 of candidate 1)

Training range = the twelve 2025 months, which is what the submission fit sees.
Full table `analysis/q1_numeric_v2.csv`; every numeric column with any row past
the min-max:

| feature | month | rows past min-max | share | train range | ranking range |
|---|---|---|---|---|---|
| `arr_landings_60m_after` | 2026-07 | 9 | 0.005 ‰ | 0 – 78 | 0 – 85 |
| `arr_landings_60m_before` | 2026-07 | 6 | 0.003 ‰ | 0 – 80 | 0 – 85 |
| `arr_landings_30m_after` | 2026-07 | 2 | – | 0 – 45 | 0 – 46 |
| `d_takeoff_eobt1` | 2026-07 | 1 | – | −9,390 – 87,001 | −10,314 – 86,280 |
| `arr_landings_30m_before` | 2026-07 | 1 | – | 0 – 45 | 0 – 46 |
| `queue_ahead_aobt` | 2026-07 | 1 | – | 0 – 486 | 0 – 514 |

**20 distinct rows in the whole file, all in July, none in January.** The six
column exposures fall on six disjoint sets of rows, so the column counts above
sum to the row count exactly. Twelve months of 2025 span the numeric envelope of
both ranking months.

The 0.1–99.9 percentile band is more informative and it is where January shows
up: `d_takeoff_eobt1` 0.61 %, `d_takeoff_lobt` 0.57 %, `d_takeoff_iobt` 0.57 %,
`d_takeoff_aobt` 0.44 %, against a 0.2 % null — 10,073 distinct rows across all
columns (4,039 January, 6,034 July). That is a 2–3× elevated tail on the
plan-to-take-off gaps and it is the disruption showing through — but it is
inside the range the trees were fitted on, so it is a density shift, not
extrapolation, and a tree prices it with splits it already has.

### The tail-signal frame (`T-tailv2` 0.4438 of candidate 1)

| feature | month | rows past min-max | share of month | airport | train max | ranking max |
|---|---|---|---|---|---|---|
| **`apt_day_p90_mvt_aobt`** | 2026-01 | **1,422** | 0.93 % | **EHAM (8.76 % of its rows)** | 3,478 | **5,411** |
| `dep_day_n` | 2026-07 | 853 | 0.44 % | LTFM (3.39 %) | 852 | 853 |
| `nonm_day_count` | 2026-01 | 727 | 0.48 % | EHAM (4.48 %) | 107 | 123 |
| `apt_hour_med_mvt_aobt` | 2026-01 | 60 | 0.04 % | EHAM | 4,801 | 5,415 |
| five others | both | ≤ 3 each | – | – | – | – |

`dep_day_n` at 853 against a maximum of 852 is one busy LTFM day one movement
past the record and is not an exposure in any useful sense. The real one is
`apt_day_p90_mvt_aobt` at EHAM: **56 % past the maximum any 2025 day reached**.

### Categoricals

| column | month | unseen levels | unseen rows | share |
|---|---|---|---|---|
| `airport_stand` | 2026-07 | 31 | **1,549** | 0.81 % |
| `airport_stand` | 2026-01 | 19 | 40 | 0.03 % |
| `operator` | both (4 + 173) | 31 | **177** | 0.05 % |
| `destination` | both (24 + 150) | 63 | **174** | 0.05 % |
| `flight_prefix` (tail frame) | 2026-07 | 82 | 834 | 0.43 % |
| `actype_mvt` | both (9 + 9) | 8 | 18 | 0.005 % |
| `actype_flt` | both (1 + 8) | 6 | 9 | 0.003 % |
| `airport_runway` | 2026-07 | 1 | 2 | – |

1,182 of the 1,549 unseen-stand rows are **EDDM 103–108**, the new July apron;
299 are EDDF J5/J10/V33x. `docs/shift-2026.md` reports 177 operator rows and 174
destination rows over both months and this reproduces them; its 1,177 for EDDM
103–108 is 5 short of the 1,182 counted here, which is a difference in how the
apron is delimited and not worth chasing.

### What the candidate predicts on the exposed rows

Matched on (airport, take-off hour) — the in-range column is the in-range rows'
mean reweighted to the exposure rows' own cell counts, so the comparison is not
confounded by the exposure sitting at one airport or one hour.

| exposure | rows | pred | matched in-range | Δ pred | Δ observable | Δ implied `r` |
|---|---|---|---|---|---|---|
| unseen `airport_stand` | 1,589 | 945 | 897 | **+48** | −9 | −57 |
| `apt_day_p90_mvt_aobt` past max | 1,422 | 1,926 | 886 | **+1,040** | +1,257 | +59 |
| `nonm_day_count` past max | 727 | 1,942 | 909 | **+1,033** | +1,388 | +146 |
| `dep_day_n` past max | 853 | 1,027 | 1,113 | −86 | −53 | +31 |
| unseen `flight_prefix` | 902 | 917 | 975 | −58 | −66 | −9 |

On an unseen stand the model predicts **48 s more taxi** than a matched in-range
row while the observable is 9 s *lower* — it is pricing the unknown stand upward,
which is the direction a fallback to the airport prior would give at these
airports. On the day-level exposures the prediction rises by about 1,040 s while
the observable rises by 1,257, i.e. the model passes about 83 % of the extra
through and prices the rest — the behaviour question 2 settles.

## Question 2 — airport-day exposure

### The 18 days, and what the model does on them

18 January 2026 airport-days clear their airport's 2025 maximum daily mean of
take-off minus AOBT by more than 30 s — exactly the count the assignment
expected. **Eight are EHAM, seven EDDF, two LSZH, one LFPG**; no EDDM, LTFM,
LEMD, LIRF, LEBL or EGLL day qualifies.
Full table `analysis/q2b_exposed_days_2026.csv`; the judgement column separates
three behaviours and the airports split cleanly between them.

| airport | days | over the 2025 max | judgement | why |
|---|---|---|---|---|
| **EHAM 3–7 Jan** | 5 | +688 to +1,350 s | **extrapolating + saturating** | 6–8 of the 9 day-state `_z` columns past the training maximum; implied `r` runs 107–407 s above EHAM's own ordinary-day level |
| EHAM 9, 10, 25 Jan | 3 | +50 to +280 s | tracking / mildly saturating | inputs back inside range, implied `r` within 63 s of ordinary |
| **EDDF 7 days** | 7 | +32 to +310 s | **over-shooting** | day-level inputs 1,370–1,560 s *inside* their range; the model prices the day and pushes 88–327 s harder than tracking |
| LSZH 8, 10 Jan | 2 | +98, +414 s | over-shooting | same shape as EDDF |
| LFPG 7 Jan | 1 | +159 s | tracking | implied `r` 54 s from ordinary |

The distinction matters: **only EHAM 3–7 January is genuine extrapolation.** At
EDDF and LSZH every day-level input is comfortably inside its training range, so
the model is interpolating a severe day, not inventing one. Those days are an
exposure to *mis-calibration*, not to extrapolation, and the two need different
repairs.

### EHAM, day by day

| day | rows | obs mean | 2025 max | over | pred mean | obs p90 | pred p90 | implied `r` |
|---|---|---|---|---|---|---|---|---|
| 1 Jan | 618 | 892 | 1,254 | −362 | 723 | 1,343 | 983 | 169 |
| 2 Jan | 469 | 1,258 | 1,254 | +5 | 1,133 | 2,359 | 1,763 | 126 |
| **3 Jan** | 391 | 2,330 | 1,254 | **+1,077** | 1,847 | 4,766 | 3,453 | **483** |
| **4 Jan** | 336 | 2,595 | 1,254 | **+1,342** | 2,053 | 5,397 | 4,238 | **542** |
| **5 Jan** | 208 | 2,604 | 1,254 | **+1,350** | 2,240 | 5,411 | 4,900 | **364** |
| 6 Jan | 293 | 1,942 | 1,254 | +688 | 1,697 | 4,427 | 3,726 | 245 |
| 7 Jan | 194 | 2,111 | 1,254 | +858 | 1,869 | 4,251 | 3,660 | 242 |

EHAM's ordinary-day implied `r` in January 2026 is **135 s**. On 3–5 January it
is 364–542 s: the model is handing back 230–410 s of the day's excess
take-off-minus-AOBT to the block-versus-AOBT offset instead of to taxi time. At
the 90th percentile it predicts 3,453 s where the observable is 4,766 s.
**Whether that is right is the question the 2025 analogue answers.**

### The 2025 analogue: the blend under-predicts past the envelope, on every fold

The blend's own rotated out-of-fold predictions, sliced by how far each
(airport, day) sits past that airport's *training* maximum daily mean:

The slices are **disjoint**: an earlier version of this table put the
out-of-envelope days inside the "worst 5 %" slice as well, which made the
contrast one between a set and its own subset (round-1 review, M2). Removing
them sharpens it rather than softening it (`analysis/r1_disjoint_slices.csv`):

| fold | slice | days | rows | RMSE | bias |
|---|---|---|---|---|---|
| january | whole fold | 310 | 153,706 | 343.3 | **+6.5** |
| january | worst 5 % of days, out-of-envelope removed | 18 | 8,719 | 290.4 | **+15.4** |
| january | **past the training maximum** | 2 | 1,129 | **460.2** | **−91.2** |
| december | whole fold | 310 | 165,677 | 214.9 | +5.7 |
| december | worst 5 % of days, out-of-envelope removed | 18 | 9,542 | 315.9 | **−11.6** |
| december | **past the training maximum** | 2 | 775 | **459.8** | **−106.6** |
| july_loo | whole fold | 310 | 190,713 | 297.9 | +0.9 |
| july_loo | worst 5 % of days, out-of-envelope removed | 19 | 11,940 | 315.6 | **−9.5** |
| july_loo | **past the training maximum** | 1 | 512 | **1128.8** | −31.1 |

The five out-of-envelope days individually, with the offset decomposition
(`analysis/q2b_fold_days.csv`):

| fold | airport | day | rows | over the max | RMSE | bias | true `r` | implied `r` |
|---|---|---|---|---|---|---|---|---|
| january | LFPG | 2025-01-17 | 600 | +57.5 | 559.4 | **−166.4** | −334 | −167 |
| january | EHAM | 2025-01-18 | 529 | +8.3 | 311.5 | −6.0 | +165 | +171 |
| december | EDDM | 2025-12-24 | 273 | +41.8 | 727.9 | **−304.1** | −949 | −645 |
| december | LEBL | 2025-12-12 | 502 | +4.8 | 195.5 | +0.9 | −28 | −29 |
| july_loo | LIRF | 2025-07-28 | 512 | +73.0 | 1128.8 | −31.1 | −714 | −683 |

Pooled over the five the bias is **−83 s**, and over the three that clear by
more than 30 s it is **−144 s**. Four of the five are negative, and the two that
are not are the two that barely clear (+4.8 s and +8.3 s). Meanwhile the severe
but *in-envelope* slice is biased the **other way** on january (+15.4 s) and only
mildly negative on the other two folds.

**That is the whole mechanism in one line: severity is not error, but leaving the
training envelope is.**

**exp-007 got there first, per day, and an earlier version of this section said
otherwise** (round-1 review, M10). exp-007's finding 13 states it explicitly:
"Where the base model was badly under-predicting — LFPG on 17 January at +170 s
of bias, EDDM on 15 January at +166 — that is exactly right and the block removes
almost all of it. Where the base model was already close … the same push carries
it to −75 and −150." LFPG 2025-01-17 is the single day this experiment's january
">30 s" slice consists of, so the two experiments are looking at the same day
from opposite sides.

What this table adds is a **label-free rule for telling the two populations
apart in advance**, which is the thing exp-007 said it could not do ("until
something can tell LFPG-17-January from LSZH-4-January in advance, a day-state
correction is a bet"). The rule is: *does the day's mean take-off minus AOBT
exceed that airport's training maximum?* On the fold data it separates a
+15.4 s in-envelope bias from a −91.2 s out-of-envelope one, it is computable
from the ranking file alone, and it flags 18 January 2026 airport-days.

### Direction of the board error on those days

Applying the analogue's bias to January 2026's exposed rows: **we should expect
to under-predict on the 18 exposed days**, by of order 80–170 s on 7,564 rows,
worth about **+0.2 to +0.7 s** of board estimate through the bias term alone.
The variance term is much larger and is priced in the re-mix below.

### The re-mix: the board estimate is optimistic by about 5.8 s

The january fold carries **0.73 %** of its rows on days past the airport's
training envelope. January 2026 carries **7.84 %** — 11,968 of 152,719 rows, of
which 7,564 clear by more than 30 s and 4,404 clear by less. Holding each
candidate's own measured per-slice RMSE fixed and changing only that mix
(`analysis/q2c_remix_by_candidate.csv`):

| candidate | january fold | off exposed days | 0–30 s over | > 30 s over | january re-mixed | board reported | **board re-mixed** | move |
|---|---|---|---|---|---|---|---|---|
| candidate 1 | 343.26 | 342.25 | 311.47 | 559.41 | 355.32 | 318.81 | **324.61** | **+5.80** |
| candidate 2 | 343.73 | 342.71 | 332.64 | 550.41 | 355.58 | 318.02 | **323.74** | **+5.72** |

**Both candidates should be expected to score about 5.8 s worse than their
reported board estimate.** A coarser two-slice version of the same calculation,
using one pooled exposed-slice RMSE of 460.18 for both candidates
(`analysis/q3b_january_remix.csv`), gives +4.63 s; the three-slice per-candidate
version above is the better instrument because it lets the two candidates differ
on the slice they actually differ on, and it is the one quoted throughout. An
earlier version of this document quoted the 4.63 figure beside the 324.61 table
and the two do not belong together (round-1 review, B1).

**The base of this calculation is two 2025 airport-days, one per slice, and that
is its main weakness.** The 559.41 comes from LFPG 2025-01-17 (600 rows, 58 s
past the maximum) and the 311.47 from EHAM 2025-01-18 (529 rows, 8 s past it).
Those two single-day RMSEs are being transplanted onto 7,564 and 4,404 January
2026 rows spanning 18 airport-days at four airports with exceedances up to
1,350 s. So the number has error in both directions and they do not cancel:

- **too low**, because January 2026's days are far further out than either 2025
  day, and the analogue's own bias grows with the exceedance (−2.6 s in the
  0–30 s band, −143.5 s past 30 s);
- **unstable**, because a slice RMSE from one day carries that day's own tail.
  LIRF 2025-07-28 is the demonstration: 1,128.8 s on 512 rows, an order above
  the other four out-of-envelope days.

Read 324–325 as the centre of a wide expectation, not as a prediction. What it
is good for is the *direction and the order of magnitude*, which is
`docs/shift-2026.md`'s point 4 with a number on it, and for the comparison
between the two candidates, where the two errors are common.

**The re-mix does not change the ordering.** Candidate 2 leads by 0.80 s as
reported and 0.87 s re-mixed. That widening is itself one day on each side —
candidate 2's 9.0 s advantage on LFPG 2025-01-17 against its 21.2 s deficit on
EHAM 2025-01-18 — so the *ordering* is robust to the re-mix and the *0.07 s of
widening* is not a finding.

## The day-state block: candidate 2's added exposure, measured

`D1-winter` carries **0.25** of candidate 2 outside `nm_late` and 0.240 inside
it, and `winter_arm.py` builds it with `ds_columns="all"` — the **full
25-column block**, including the nine unbounded `_z` columns. exp-007 finding 13
measured that block at **+72.1 s of RMSE harm on january's disrupted days, with
a bootstrap interval of [+3.5, +149.7] that its own text calls "uncertain by a
factor of four"** and which spans zero under the calendar-day clustering
(`experiments/exp-007-winter/RESULTS.md`). It named the mechanism: a `_z` column
beyond its own training maximum pushes a near-calibrated model past the truth.
So the first question is how much of that mechanism January 2026 presents, and
the second is what it does to the bytes.

### January 2026 presents the mechanism at a much larger scale than 2025 did

Per airport, because `DayState.fit` standardises inside the airport. **Counted
in both tails**, which an earlier version of this section did not do (round-1
review, M8): the one-sided count dropped the `ds_nm_rate_z` collapse that
`docs/shift-2026.md` calls "EHAM January 2026's signature" and that
`prc.day_state_features.CORE` lists for exactly that reason.

| `_z` column | month | rows over the airport max | worst value ÷ that max |
|---|---|---|---|
| `ds_share_span1800_z` | 2026-01 | 9,178 | 3.5× |
| `ds_taxiin_p90_z` | 2026-01 | 5,804 | **19.5×** |
| `ds_span_p90_z` | 2026-01 | 5,852 | 8.0× |
| `ds_span_med_z` | 2026-01 | 4,414 | 11.6× |
| `ds_sched_p90_z` | 2026-01 | 3,744 | 3.5× |
| `ds_sched_med_z` | 2026-01 | 2,673 | 4.6× |
| `ds_span_std_z` | 2026-01 | 2,423 | 2.3× |
| `ds_gap_p90_z` | 2026-01 | 1,971 | 1.8× |
| every column | 2026-07 | ≤ 2,109 | ≤ 1.2× |

| rows with ≥1 `_z` outside its airport's training range | above the max | below the min | share of the month |
|---|---|---|---|
| **EHAM January** | 5,305 | 1,697 | **32.7 %** of 16,232, up to **8 of 9 at once** |
| EDDF January | 3,987 | 0 | 25.4 % of 15,667 |
| **EDDF July** | 0 | **7,998** | 38.7 % of 20,650, on `ds_span_std_z` |
| **LIRF January** | 0 | **3,148** | 28.5 % of 11,061 |
| LEMD January | 2,040 | 0 | 12.1 % |
| LFPG January | 1,721 | 0 | 9.5 % |
| LSZH January | 1,683 | 674 | 16.8 % |
| LIRF July | 2,623 | 0 | 16.6 %, never more than 1 column |
| EDDM January | 736 | 0 | 6.7 % |
| **January 2026 total** | 15,842 | – | **19,895 of 152,719 = 13.0 %** |
| **July 2026 total** | 3,407 | – | **11,405 of 192,122 = 5.9 %** |

The below-minimum side is not a curiosity: **EDDF July's 7,998 rows** and
**LIRF January's 3,148** are both larger than anything on the above-maximum side
outside EHAM and EDDF January, and a linear leaf extrapolates downward exactly
as it extrapolates upward.

exp-007 reports that its worst-hit day, LSZH 2025-01-04, had **5 of 9** columns
over its airport's training maximum (that count is exp-007's; nothing here
recomputes 2025 day-state exposure). January 2026 has **47 airport-days** with
at least one column outside the range and **six days with 6 or more above the
maximum** — EHAM 3, 4, 5, 6, 7 January and LSZH 10 January.
`ds_taxiin_p90_z`, the column exp-007 identified as carrying the weight (2.6×
its maximum on that day), reaches **19.5×** here.

### What it does to the shipped bytes: about a quarter of what it does to the member

`D1-winter` minus the mean of the other three general members, on the ranking
rows (`analysis/q1c_member_push.csv`):

| slice | rows | member push | ÷4 into the file | `T-tailv2` | `C2-cat` | `L2-hub` | `D1-winter` |
|---|---|---|---|---|---|---|---|
| 2026-01 all | 152,719 | +3.1 | +0.8 | 1,078 | 1,071 | 1,069 | 1,075 |
| 2026-01 no `_z` over | 136,877 | −0.4 | −0.1 | 1,052 | 1,050 | 1,048 | 1,049 |
| 2026-01 1–2 over | 12,367 | +22.0 | +5.5 | 1,153 | 1,132 | 1,130 | 1,160 |
| **2026-01 3+ over** | 3,475 | **+71.5** | **+17.9** | 1,834 | 1,682 | 1,673 | 1,801 |
| **2026-01 6+ over** | 1,700 | **+77.8** | **+19.4** | 2,023 | 1,799 | 1,773 | 1,943 |
| EHAM January, 6+ over | 1,422 | +44.4 | +11.1 | – | – | – | – |
| 2026-07 1–2 over | 3,407 | −20.1 | −5.0 | 1,447 | 1,453 | 1,438 | 1,426 |

**The member's mean prediction sits +71 to +78 s above the other three general
members on exactly the rows where the mechanism lives, and the flat quarter
weight delivers +18 to +19 s of it into the candidate.**

**This is not exp-007's +72.1 s and an earlier version of this section called it
"the same magnitude"** (round-1 review, B3). exp-007's number is an **RMSE
delta** against a matched base arm, on labelled 2025 rows, with an interval of
[+3.5, +149.7]. This one is a **mean-prediction difference between members**, on
unlabelled 2026 rows, with no control. Two quantities of similar size in seconds,
measuring different things. What the coincidence of size does license is a
statement about **scale**: the member's disagreement with its blend-mates on
these rows is of the same order as the harm exp-007 measured for the same block,
so it is not a rounding effect and it is worth the owner's attention.

The push is also **confounded by everything else that differs between the
members**: `D1-winter` is a linear-leaf LightGBM at 150 rounds on a 121-column
frame with lirf and phi steps, against two plain-tree GBMs on the 54-column v2
frame and a regime router. The day-state block is one of several differences, and
the next paragraph shows the confound biting.

Two things moderate the effect on the file: the flat rule is a 4× divisor, and
`D1-winter` ships at **150 rounds** under the desk's budget rule against
exp-007's 300.

The row that is easy to miss: **`T-tailv2` pushes harder than `D1-winter` does**
(1,834 against 1,801 on the 3+ slice, 2,023 against 1,943 on 6+), while the two
v2-frame GBMs sit 150–250 s lower. **The largest member of candidate 1 was
already carrying a day-level push**, through `apt_day_p90_mvt_aobt` and
`apt_day_med_mvt_aobt`, and candidate 2 adds a second one. Candidate 1 is not
free of this mechanism; it has one member with it instead of two.

### Which is why the two candidates barely differ on those rows

| slice | rows | candidate 1 | candidate 2 | Δ |
|---|---|---|---|---|
| 2026-01 all | 152,719 | 1,074.9 | 1,073.5 | −1.4 |
| 2026-01 exposed rows | 15,842 | 1,275.4 | 1,276.6 | **+1.1** |
| 2026-01 3+ `_z` over | 3,475 | 1,749.3 | 1,745.7 | **−3.6** |
| the 18 exposed airport-days | 7,564 | 1,483.4 | 1,483.6 | +0.1 |
| EHAM January | 16,232 | 943.1 | 940.1 | −3.0 |

**Candidate 2's day-state member changes the served predictions on the exposed
rows by 1 to 4 seconds.** The mechanism is loud in the member and quiet in the
file.

### And on the folds, the push is in the right direction where the exceedance is large

Both candidates rebuilt on the folds with their own shipped rule
(`analysis/q2c_candidates_on_days.csv`); positive gain = candidate 2 better:

| fold | slice | rows | cand 1 | cand 2 | gain | bias 1 | bias 2 |
|---|---|---|---|---|---|---|---|
| january | whole fold | 153,706 | 343.26 | 343.73 | −0.46 | +6.5 | +5.1 |
| january | worst 5 % of days | 8,441 | 316.44 | 318.64 | −2.20 | −5.2 | −3.0 |
| january | past the training maximum | 1,129 | 460.18 | 461.35 | −1.17 | −91.2 | −71.4 |
| january | **past it by > 30 s** | 600 | 559.41 | **550.41** | **+9.00** | −166.4 | **−137.2** |
| december | past it by > 30 s | 273 | 727.91 | 745.95 | −18.04 | −304.1 | −303.7 |
| july_loo | past the training maximum | 512 | 1128.76 | 1137.92 | −9.15 | −31.1 | +6.8 |

On january's most exposed slice candidate 2 is **9.0 s better** and moves the
bias from −166 to −137 — the day-state push doing exactly what the analogue says
is needed. It is worse on december's and july_loo's exposed slices, where the
exceedances are small (42 s and 73 s) and the base is closer to calibrated. The
counts are 600, 273 and 512 rows, so none of these is well powered; the pattern
is consistent with the mechanism and is not established by it.

**What this does not license.** Candidate 2's 0.79 s board advantage does **not**
come from the exposed days — it comes from `july_loo`'s whole fold (+1.96 s). On
january it is 0.46 s worse overall. The exposed-day gain is real on the largest
exceedance slice and does not carry the candidate.

## Question 3 — is the candidate file itself sane?

Checked against `submitting.parquet` (`code/r1_fixes.py`), which an earlier
version of this document asserted without checking (round-1 review, M11): **both
candidates have an id set identical to the template, in template row order,
float64, all finite.**

| check | candidate 1 | candidate 2 |
|---|---|---|
| sha256 | `541b1b1107fe…` | `5c57799954cb…` |
| rows / template rows | 344,841 / 344,841 | 344,841 / 344,841 |
| id set identical to template | yes | yes |
| template row order | yes | yes |
| NaN / negative / below the 60 s floor | 0 / 0 / 0 | 0 / 0 / 0 |
| minimum | 60.10 | 60.00 |
| median | 962.87 | 961.80 |
| p99 | 2,461.1 | 2,454.7 |
| maximum | 118,660.5 | 119,739.9 |
| above 20,000 s | 28 | 28 |
| mean, January / July | 1,074.9 / 1,010.3 | 1,073.5 / 1,008.9 |

### The 28 predictions above 20,000 s

All at **LIRF**; **27** are `noac` rows and one is `nm_bulk`. Every one is earned
by an observable of the same size rather than invented:

| rank | pred | regime | take-off − AOBT | take-off − SCHED | stand |
|---|---|---|---|---|---|
| 1 | 118,661 | noac | – | 111,654 | 822 |
| 2 | 88,147 | noac | – | 87,484 | 811 |
| 3 | 86,146 | **nm_bulk** | **86,340** | 94,560 | 204 |
| 4 | 69,709 | noac | – | 61,322 | 710 |
| … | … | … | … | … | … |
| 28 | 20,365 | noac | – | 15,058 | 108 |

The `nm_bulk` row predicts 86,146 s where take-off minus AOBT is 86,340 s — an
aircraft the network saw move a day before it took off, and the residual model
is answering with an offset of 194 s, which is the right shape. The `noac` rows
are placeholder-block-time rows where the target is block minus schedule and the
schedule is a day or more before take-off. The largest, 118,661 s, is **inside**
2025's observed `noac` maximum of 131,167 s. Nothing here needs a ceiling: exp-001
and exp-002 both measured a taxi-time ceiling losing, and this file is the reason.

### The two small regimes against their 2025 selves

| slice | n | mean | p50 | p90 | p99 |
|---|---|---|---|---|---|
| 2026-01 `noac` (**108 rows**) | 108 | 8,496 | 5,069 | 20,818 | 56,386 |
| 2026-07 `noac` (**280 rows**) | 280 | 6,859 | 3,553 | 10,696 | 65,053 |
| 2025 winter truth `noac` | 265 | 8,106 | 5,577 | 12,585 | 87,522 |
| 2025 summer truth `noac` | 896 | 6,134 | 1,560 | 11,795 | 55,365 |
| 2026-01 `nonm` (2,341) | 2,341 | 1,279 | 1,261 | 1,890 | 2,652 |
| 2026-07 `nonm` (2,554) | 2,554 | 976 | 976 | 1,383 | 1,796 |
| 2025 winter truth `nonm` | 5,268 | 1,206 | 948 | 1,978 | 6,237 |
| 2025 summer truth `nonm` | 10,318 | 974 | 939 | 1,498 | 2,523 |

### The seven rows that are in neither regime count

The assignment names **5,290** no-NM ranking departures. `prc.blend.regime_labels`
keys `nonm` on `FLIGHT_ID_mvt` and finds **5,283** (4,895 `nonm` + 388 `noac`).
The difference is **7 departures with no `AOBT_3_flt` but a live NM
`FLIGHT_ID_mvt`** (`analysis/r1_no_aobt_with_nm.csv`), and they were in no slice
of this audit until the round-1 review asked (supplied-input rule):

| airport | take-off | regime | candidate 1 | candidate 2 |
|---|---|---|---|---|
| LTFM | 2026-01-01 00:01 | nm_bulk | 1,177 | 1,430 |
| LTFM | 2026-01-01 00:05 | nm_bulk | 1,130 | 1,235 |
| LFPG | 2026-01-01 00:10 | nm_bulk | 1,298 | 1,511 |
| LEMD | 2026-01-01 00:11 | nm_bulk | 1,337 | 1,382 |
| LEMD | 2026-07-01 00:07 | nm_bulk | 1,164 | 1,181 |
| LFPG | 2026-07-01 00:10 | nm_bulk | 1,002 | 1,036 |
| LEBL | 2026-07-01 00:10 | nm_bulk | 825 | 962 |

**All seven are in the first twelve minutes of the first day of a ranking
month**, which is the signature of a month-boundary join: the NM record exists
but its off-block falls in the previous month, outside the file. They are routed
to `nm_bulk`, so the members' residual anchor falls back to the airport median
(`run.py:anchors`) and `apply_residual_guard` is a no-op on them because it needs
a finite AOBT. The predictions that come out — 825 to 1,511 s — are ordinary, and
the two candidates differ by up to 253 s on them, which is the widest relative
member disagreement outside `noac`/`nonm`. Seven rows cannot move an RMSE, so
this is a completeness fix, not a risk; the reason to record it is that
`docs/input-rule.md` names this population as needing a fallback path and the
regime labeller does not give it one.

### The two small regimes against their 2025 selves

The 388 `noac` rows split 108 January / 280 July and both sit on top of their
season's 2025 truth in mean and p90; the July median is 2.3× the 2025 median,
which is the head predicting an expectation rather than a mode and is the
intended behaviour under RMSE (`prc.tail_model`). The 4,895 `nonm` rows are
within 6 % of the 2025 mean in both months. **Neither small regime is being
predicted off its own distribution.** The `nonm` upper tail is compressed — p99
2,652 against 6,237 in 2025 winter — which is the residual guard and the
mean-seeking loss, not a bug, but it is where a heavy-tailed month would hurt.

### Per airport-month level against the 2025 same-season truth

| month | airport | pred mean | 2025 same-season truth | Δ |
|---|---|---|---|---|
| 2026-01 | EHAM | 943 | 795 | **+148** |
| 2026-01 | LTFM | 1,226 | 1,086 | **+141** |
| 2026-01 | EDDF | 1,020 | 880 | **+140** |
| 2026-01 | LSZH | 890 | 759 | **+131** |
| 2026-01 | EDDM | 997 | 874 | **+123** |
| 2026-01 | LEMD / LIRF / LFPG / LEBL / EGLL | – | – | +30 / +19 / +10 / −4 / −12 |
| 2026-07 | EHAM | 841 | 751 | +90 |
| 2026-07 | LIRF | 1,368 | 1,291 | +77 |
| 2026-07 | the other eight | – | – | −31 … +17 |

**The model is already raising January by 120–150 s at exactly the four airports
`docs/shift-2026.md` names as disrupted** (EDDF, EHAM, EDDM, LSZH) plus LTFM,
and leaving the five calm ones (LEMD, LIRF, LFPG, LEBL, EGLL) within 30 s. That is the single most reassuring
number in this audit: the response is there and it is aimed correctly. Question 2
is about whether it is *large enough*, and the analogue says no.

### Member disagreement above 1,800 s

350 rows in candidate 1 (356 before the floor and guard are applied to each
member), `analysis/q3_member_disagreement.csv`:

| month | regime | rows | mean spread | mean prediction |
|---|---|---|---|---|
| 2026-07 | noac | 138 | 3,683 | 8,750 |
| 2026-01 | nonm | 106 | 2,498 | 2,022 |
| 2026-01 | noac | 56 | 3,525 | 10,520 |
| 2026-01 | nm_late | 22 | 2,268 | 5,449 |
| 2026-07 | nm_late | 19 | 2,362 | 4,556 |
| 2026-07 | nonm | 4 | 4,977 | 2,933 |
| 2026-01 | nm_bulk | 4 | 2,370 | 1,729 |
| 2026-07 | nm_bulk | 1 | 1,915 | 2,347 |

**194 of the 350 are `noac` and 110 are `nonm` — 87 % of the disagreement is in
the 1.6 % of rows with no aircraft type or no NM join**, which is where the three
members have genuinely different mechanisms (a schedule-anchored head against two
residual models). Five rows of 344,841 in `nm_bulk` disagree by more than
1,800 s; their `mvt_sched` values are **−228, 3,779, 6,722, 14,766 and
26,817 s**, so four are far-from-schedule departures and one is not, and that
one is the row worth a look.

**In relative terms the disagreement is larger than an earlier version of this
section said** (round-1 review, M3). On the six largest LIRF `noac` spreads the
members differ by **11.5 % to 61.8 %** of the prediction, not 10–15 %. And the
largest relative disagreement in the file is not a `noac` row at all: **LFPG
2026-01, `nonm`, a spread of 9,640 s on a prediction of 2,895 s — 333 %**, where
`C2-cat` says 10,355 s and the other two say about 750. Two more `nonm` rows sit
above 280 %. Those are disagreements about the row's **nature**, not its size,
and `nonm` is the regime where the three members have the least in common.

## Question 4 — label-free repairs

### The LFPG runway retirement is not an exposure at the pair level

CDG lost 27R (9.0 % of its 2025 departures) and 09L (4.8 %); 27L absorbed them
(11.2 → 29.5 % in January 2026, 20.4 % in July). `docs/shift-2026.md` expected
"the stand-to-runway pairs at LFPG in 2026 are a different set from 2025". **They
are not.** Of 39,872 LFPG ranking rows, **19 in January and 29 in July** carry a
(stand, runway) pair the twelve 2025 months never contain. 27L was already flown
from essentially every stand in 2025 (26,846 rows); only its *share* moved. The
retirement changed the mix, not the support, and a per-pair repair has 48 rows to
act on. This refutes the pair-level reading of that concern; the *distributional*
reading — that a model trained on the 2025 mix is trained on routes no longer
flown — is not touched by it.

### Runway ordering: sane in July, weaker in January, and the observable agrees

Per-stand Spearman correlation between the candidate's per-runway mean
prediction and (a) the 2025 per-stand per-runway mean taxi, (b) the ranking
month's own observable take-off minus AOBT. Cells with 3+ runways and 20+ rows
each (`analysis/q4c_runway_order_all_airports.csv`):

**Every row of `analysis/q4c_runway_order_all_airports.csv`**, not a selection —
an earlier version of this table omitted LTFM January, which is the worst cell in
it (round-1 review, M1):

| airport | month | cells | pred vs 2025 | obs vs 2025 | pred vs obs |
|---|---|---|---|---|---|
| **LFPG** | 2026-01 | 41 | **0.12** | 0.38 | 0.56 |
| LFPG | 2026-07 | 56 | 0.84 | 0.72 | 0.79 |
| **EDDM** | 2026-01 | 46 | **0.24** | 0.45 | **0.40** |
| EDDM | 2026-07 | 41 | 0.81 | 0.63 | 0.53 |
| **LTFM** | 2026-01 | 26 | 0.89 | 0.54 | **0.34** |
| LTFM | 2026-07 | 1 | 0.50 | −0.50 | −1.00 |
| EDDF | 2026-01 | 40 | 0.62 | 0.64 | 0.85 |
| EDDF | 2026-07 | 56 | 0.89 | 0.82 | 0.81 |
| EHAM | 2026-01 | 18 | 0.74 | 0.74 | 0.94 |
| EHAM | 2026-07 | 98 | 0.93 | 0.89 | 0.93 |
| LSZH | 2026-01 | 11 | 0.73 | 0.59 | 0.82 |
| LSZH | 2026-07 | 14 | 0.93 | 0.89 | 0.96 |
| EGLL | 2026-01 | 115 | 0.79 | 0.77 | 0.88 |
| EGLL | 2026-07 | 126 | 0.58 | 0.57 | 0.74 |
| LEMD | 2026-01 | 20 | 0.68 | 0.70 | 0.88 |
| LEMD | 2026-07 | 74 | 0.81 | 0.87 | 0.86 |
| LEBL | 2026-07 | 1 | 1.00 | 1.00 | 1.00 |

**LFPG in January has the largest gap between the candidate and the observable's
own agreement with 2025** — 0.12 against 0.38 — and EDDM January is second at
0.24 against 0.45. But **the candidate does not track the observable as well as
the observable tracks 2025 in four cells**, not the one an earlier version
claimed: LTFM January (0.34 vs 0.54), EDDM January (0.40 vs 0.45), EDDM July
(0.53 vs 0.63) and LEMD July (0.86 vs 0.87). **LTFM January is the weakest
prediction-to-observable agreement of any populated cell**, below LFPG's.

Three things keep this a **MAYBE rather than a defect**: the bases are 11 to 126
stand cells, LTFM July and LEBL July have one cell each and are noise, and at
LFPG the candidate still tracks the January observable at 0.56, so it is a
weakened response and not a scrambled one. What survives as a pattern is that
**the weak cells are January cells at the three airports whose runway mix moved
most** (LFPG's retirement, EDDM's 26L/08L swing, LTFM's south-flow half-month),
which is the shape a mix shift would produce.

**Revival condition**: if a later session gets a board reading that shows the
January half worse than this audit's re-mix predicts, the LFPG and EDDM January
runway response is the first place to look, and the measurement to run is the
same table with a per-stand-per-runway target encoding added to the frame.

*One correction to my own working.* An earlier version of this check compared a
per-cell prediction **mean** against a per-cell observable **median** and read
LFPG January at ρ = 0.04, which would have been a scrambled response. Comparing
like with like gives 0.56. The finding above is the like-with-like one.

### Repair 1, clipping numerics to the training range: a provable no-op, measured

One fit per (fold, learner) on the unmodified training frame, then the same
model predicts the validation frame twice — raw and repaired. The fit is held
fixed, so the difference is the repair and nothing else, and this is also
exactly how a repair would be deployed.

| fold | learner | variant | touched | RMSE | Δ | ≤3600 | >3600 | no-NM | noac | mean abs change on touched |
|---|---|---|---|---|---|---|---|---|---|---|
| july_loo | `L2-hub` (LightGBM Huber, 2,100 rounds) | raw | 0 | 312.928 | – | 274.607 | 2573.12 | 1289.61 | 3971.07 | – |
| july_loo | | **clip** | 2,947 | **312.928** | **0.000** | 274.607 | 2573.12 | 1289.61 | 3971.07 | **0.000** |
| july_loo | | parent | 21 | 312.933 | +0.005 | 274.613 | 2573.12 | 1289.61 | 3971.07 | 65.3 |
| july_loo | | both | 2,968 | 312.933 | +0.005 | 274.613 | 2573.12 | 1289.61 | 3971.07 | 0.46 |
| january | `C2-cat` (CatBoost d8, 5,325 rounds, GPU) | raw | 0 | 348.560 | – | 217.135 | 5933.68 | 2861.86 | 3442.61 | – |
| january | | **clip** | 6 | **348.560** | **0.000** | 217.135 | 5933.68 | 2861.86 | 3442.61 | **0.000** |
| january | | parent | 101 | 348.552 | −0.007 | 217.123 | 5933.68 | 2861.81 | 3442.61 | 63.9 |
| january | | both | 107 | 348.552 | −0.007 | 217.123 | 5933.68 | 2861.81 | 3442.61 | 60.4 |

Neither repair moves the `>3600` or `noac` decomposition at all, which is the
expected shape: both act on ordinary rows, and the tail is carried by `noac` and
`nonm`, where nothing they touch appears.

**The clip changes the prediction by exactly zero on 2,947 rows and on 6 rows,
for two different learners.** That is not a small effect inside noise, it is a
structural identity for the columns concerned: a gradient-boosted tree splits on
thresholds drawn from the training data, so a feature value above the training
maximum takes the same path through every tree as the maximum itself.

**The claim has to be stated narrowly and an earlier version of this section did
not** (round-1 review, B2). The correct statement is:

> Clipping a numeric column **that enters the model only as a tree split
> feature** to the training range is the identity map for a plain tree ensemble.

Two exceptions, and both matter here:

1. **A column the prediction is built from arithmetically.** Every member fits a
   residual, so the served value is `anchor − tree(...)`, and
   `prc.tail_model.predict` goes further — its `nm` head returns
   `mvt_aobt − tree(...)`, its `noac` head `mvt_sched − (1−p)·off`, and
   `_band_predict` switches on `np.digitize(mvt_sched/3600, BAND_EDGES)`.
   Clipping `mvt_aobt`, `mvt_sched` or `d_takeoff_aobt` moves the prediction
   arithmetically and can move a row across a band edge. `code/q4_fixes.py`
   computes the anchor from the **raw** frame by design, so those columns are
   excluded from the intervention by construction; the 0.000 result is
   therefore a measurement about *split features only*, and the columns it
   actually touched confirm that — `noac_day_airport` and `noac_share_day` on
   2,946 of the 2,947 rows, both pure tree features.
2. **A linear-leaf tree**, where the leaf value is an affine function of the
   features and an out-of-range value is extrapolated by that function rather
   than binned.

Applied to the five members across the two candidates, for the **day-level
columns where the exposure actually is** (`apt_day_p90_mvt_aobt`,
`nonm_day_count`, `apt_hour_med_mvt_aobt`, and the day-state `_z` block) — all
of which are split features in their members and none of which appear in any
member's arithmetic:

| member | learner | can a predict-time clip of the exposed columns move it? | basis |
|---|---|---|---|
| `T-tailv2` | LightGBM `MAIN_PARAMS`, no `linear_tree` | **no** | provenance + the two measurements below |
| `C2-cat` | CatBoost oblivious trees | **no** | **measured**, 6 rows, 0.000 |
| `L2-hub` | LightGBM Huber, `variant="plain"` | **no** | **measured**, 2,947 rows, 0.000 |
| `N1-nmtail` | LightGBM, `prc.nmtail` | **no** | provenance |
| **`D1-winter`** | **LightGBM `linear_tree=True, linear_lambda=1.0`** (`winter_arm.py`) | **yes** | **provenance only — not measured here** |

**The plain-tree half is now measured a third time, on the exposure itself.**
`code/q4d_clip_noop.py` fits LightGBM at `prc.tail_model.MAIN_PARAMS` on the
twelve months of the tail frame and predicts the ranking file twice, raw and with
the six day-level columns clipped to their training maxima — the intervention
that would be deployed, on the 1,422 rows it actually touches, needing no labels
because the question is whether the clip moves the prediction:

| learner | rows the clip touches | predictions changed | max absolute change |
|---|---|---|---|
| plain trees, `prc.tail_model.MAIN_PARAMS` | 1,422 | **0 of 344,841** | **0.000000** |

Mean prediction on the touched rows is 2,247.279131979735 s before and after, to
every digit. **A clip of the exposed day-level columns is dead for the tail
model.**

**And the linear-leaf half is now measured too** (`code/q4e_linear_leaf.py`,
round-1 review M9). The full-scale `linear_tree=True` arm was still fitting after
an hour on a box at load 470 and was stopped: the question is *qualitative* — can
a clip move a linear-leaf model at all? — and a qualitative question does not
need a 2.1 M-row fit. The same two learners on a 400,000-row sample at 150
rounds, same frame, same clip, same 1,422 rows, with the caps still taken from
all twelve months:

| learner | rows the clip touches | predictions changed | mean abs change | max abs change | mean change |
|---|---|---|---|---|---|
| plain trees | 1,422 | **0** | 0.00 | 0.00 | 0.00 |
| **linear leaves** | 1,422 | **1,405** | **29.85** | **493.30** | **−7.93** |

The plain arm reproduces 0.000 at this size, so the instrument is sound and the
linear arm's answer is the answer. **A clip moves 99 % of the exposed rows on a
linear-leaf model and none of them on a plain one.**

**The direction is the part worth the owner's attention.** The mean change is
**−7.9 s** while the mean *absolute* change is **29.8 s** and the maximum is
493 s: the clip moves rows both ways, because a linear leaf's coefficient on a
given column can have either sign, and the up-moves and down-moves largely
cancel in the mean. So clipping `D1-winter` is **not** simply "remove an upward
push on disrupted rows" — it is a two-sided intervention with a small net effect
and a large per-row one, and whether it helps is a fold measurement, not a
deduction. That is exactly the measurement the desk routed to the member's owner.

*What this arm is and is not.* It is a demonstration of the **mechanism** on the
tail frame, fitting the taxi time directly at 150 rounds on a sample. It is not
`D1-winter`, which is a different frame, a residual target, 150 rounds on all
twelve months, and three seeds. The qualitative result — plain 0, linear
almost-all — is a property of the leaf model and carries; the 29.8 s does not. The point is also not
new: exp-007's round-2 review already recorded that its arm's leaves "carry
linear models that extrapolate without bound", against the same member. What is
new here is the *negative* half — that the clip is dead for the other four
members — and that half is measured.

One consequence outside this experiment. exp-007's never-run "clipped-block arm",
proposed there as the experiment that would separate boundedness from the other
differences, **only works because that arm has linear leaves**; the same
experiment on a plain-tree arm would measure nothing and would be read as
"boundedness does not matter".

*What is not shown.* This says a clip cannot move a plain tree through a split
feature; it does not say the exposure is harmless there. A tree answering an
out-of-range row from its top bin is **saturating**, and question 2 measures the
consequence: the blend under-predicts by 83–144 s on days past the envelope. The
repair for saturation is a feature the model can extrapolate along (a rank within
the ranking month, a ratio to the airport's own level, or a residual anchor), not
a clip.

### Repair 2, mapping an unseen categorical level to its parent

An unseen `airport_stand` is replaced at prediction time by a representative
training level of its parent group — same airport and same stand prefix, else
same airport — where the representative is the level whose training median
residual is closest to the group's row-weighted median. `airport_runway` has no
unseen level on either fold.

| fold | learner | rows | RMSE raw | RMSE repaired | bias raw | bias repaired | mean |Δ| | paired MSE gain [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| january | CatBoost | 101 | 333.09 | **321.19** | −65.1 | **−8.7** | 63.9 | +7,787 [−2,770, +20,792] |
| july_loo | LightGBM | 21 | 266.65 | **312.97** | +71.3 | +50.0 | 65.3 | **−26,851 [−42,397, −12,439]** |

**It helps january and hurts july_loo, and it is not adopted.** The frozen rule
in SPEC.md is that a fix must be non-negative on both folds; this one has a
january interval spanning zero and a july_loo interval excluding zero on the
wrong side. The whole-fold effect either way is ±0.007 s.

Two confounds stated rather than hidden: the two folds ran different learners,
so learner and fold are not separated here; and 21 and 101 rows are thin bases
whichever way they point. What the two folds agree on is the *size* of the
intervention — it moves the affected predictions by about 64 s either way — so
this is a repair with real leverage on 1,589 ranking rows and an unproven sign.

**MAYBE, with its revival condition**: run the same repair on both learners on
both folds (two more fits, ~1.3 GPU-h and ~35 min CPU) so learner and fold are
separated, and add the 1,182 EDDM new-apron rows as their own slice, since they
are 74 % of the ranking exposure and neither fold contains anything like them —
a whole new apron is a different object from a stand that appeared once.

### Hand-off: the exposure rows the members' owners need

The desk routed the day-level predict-time clip to the members' owners
(2026-09-05). These are the rows a clip would act on, so that the owner does not
have to rediscover them:

**`T-tailv2` / `prc.tail_features`** — clip `apt_day_p90_mvt_aobt` at the
twelve-month training maximum **3,478.4 s** (per airport limits in
`analysis/q1_numeric_by_airport_tail.csv`). It bites on **1,422 January rows,
1,422 of them EHAM**, reaching 5,411 s. Also `nonm_day_count` (727 EHAM January
rows, max 107 → 123) and `apt_hour_med_mvt_aobt` (60 rows, 4,801 → 5,415).
`dep_day_n` exceeds by one movement at LTFM and is not worth an intervention.
On the **january fold** the same columns exceed on 2 airport-days / 1,129 rows,
so the fold can measure it, and `analysis/q2b_fold_days.csv` names those days
(LFPG 2025-01-17 and EHAM 2025-01-18).

**`D1-winter` / `prc.day_state_features`** — `winter_arm.py` passes
`ds_columns="all"`. Two options with different costs, both label-free:
`--ds-columns deice` (exp-007's own deliverable, one bounded column, measured
there at −3.0 s on january against +4.3 s for the block), or clipping the nine
`_z` columns to their per-airport training range, which is exp-007's never-run
`CORE`-style clipped-block arm and the experiment its RESULTS names as "the
first thing to run if anyone reopens this lane". `analysis/q1b_daystate_*.csv`
has the per-airport limits, the 19,895 exposed January rows (15,842 above the
maximum, the rest below the minimum) and the 47 exposed January airport-days.
`analysis/r1_daystate_by_airport_symmetric.csv` is the two-tailed count and is
the one to use; the below-minimum side includes EDDF July's 7,998 rows on
`ds_span_std_z`, which a linear leaf extrapolates downward.

**What this audit can say about which way the clip should go, and cannot.** The
2025 analogue says the base under-predicts past the envelope, so a push there is
the right direction and a clip removes it. exp-007 says the push overshoots on
severe in-range days. Both are true and they act on **different day
populations**, which is why a clip is the right instrument for the first and a
regime restriction is the right instrument for the second: clipping the *inputs*
leaves the model free to price severe in-range days and stops it answering from
a bin it never fitted. That is a prediction, not a measurement; the fold
measurement is the owner's.

## What this audit does not establish

- **No board reading.** Every statement about the ranking months is an inference
  from 2025 analogues. The re-mix is arithmetic on measured slice RMSEs, not a
  measurement of 2026.
- **The analogue is five days.** The 2025 folds contain only five airport-days
  past their airport's training envelope, and one of them (LIRF 2025-07-28)
  carries an RMSE of 1,129 s that is a tail event rather than a day effect. Four
  of five are biased low and the pooled figure is −83 s, but a five-day base is
  a direction, not a coefficient.
- **The exposed-day comparison between the candidates is underpowered.** 600,
  273 and 512 rows on the three folds. The +9.0 s january gain for candidate 2
  and the −18.0 s december loss are the same kind of number.
- **The "over-shooting" judgement at EDDF and LSZH is label-free.** It says the
  model departs from the airport's own ordinary-day relation between prediction
  and observable by 88–327 s. Whether that departure is an error needs labels.
- **`nonm` tail compression is flagged, not measured.** The candidate's `nonm`
  p99 is 2,652 s against 6,237 s in the 2025 winter truth. That is what a
  mean-seeking loss and a residual guard do, and it is the right choice under
  RMSE in expectation; it is recorded because a heavy-tailed January would find
  it, not because it is shown to be wrong.

## What would change the reading

| if | then |
|---|---|
| a board reading lands near 324–325 | the exposure model here is right; the next work is EHAM's five extrapolating days |
| a board reading lands anywhere between 318 and 326 | the re-mix cannot be distinguished from its own uncertainty: its two slice RMSEs come from one 2025 day each, so it predicts a direction and an order of magnitude, not a value |
| a board reading lands near 318–319 | the 2025 analogue over-states the harm and the day-level push is why; re-price the day-state clip as a risk, not a fix |
| a board reading lands above 330 | look at the `nonm` upper tail first, then LFPG/EDDM January runway response |
| an owner's predict-time clip lands on the folds | re-run `code/q2c_candidates_on_days.py` against the clipped members; the exposed-day rows are the slice that moves |
| more 2025 days past the envelope can be found | the analogue's −83 s gets a real interval; the january fold alone will not supply them, but a leave-one-month-out sweep over all twelve months would |

## Answers, in the order asked

**1. Out-of-range inputs.** The v2 frame: 20 distinct ranking rows past the
twelve-month range in the whole file, all July, none January; the 0.1–99.9 band is 2–3× the
null on the four plan-to-take-off gaps in January, which is density, not
extrapolation. The tail frame: `apt_day_p90_mvt_aobt` on 1,422 January rows
(EHAM, to 56 % past the maximum), `nonm_day_count` on 727, `apt_hour_med_mvt_aobt`
on 60. The day-state block (candidate 2 only): 19,895 January rows, 13.0 % of the
month, and 32.7 % of EHAM's above the maximum with up to 8 of 9 columns over at
once. Categoricals:
1,589 unseen stands (74 % of them EDDM's new July apron), 177 operator rows, 174
destination rows, 18 `actype_mvt` and 9 `actype_flt` rows, 2 runway rows, all
counted over both months. On unseen stands the model
predicts 48 s more than a matched in-range row; on the day-level exposures it
predicts 1,040 s more where the observable is 1,257 s more.

**2. Airport-day exposure.** 18 January 2026 airport-days clear their 2025
maximum by more than 30 s. **EHAM 3–7 January is extrapolating and saturating**
(6–8 of 9 day-state columns out of range; implied `r` 107–407 s above EHAM's
ordinary level). **EDDF's seven days and LSZH's two are over-shooting, not
extrapolating** — their day-level inputs are 1,370–1,560 s inside range. LFPG's
one day is tracking. The 2025 analogue says the blend **under-predicts** on days
past the envelope: −91 s (january), −107 s (december), −31 s (july_loo), −144 s
pooled over the three worst. **So the board error on January's exposed days
should be an under-prediction**, and the exposure re-mix puts the January half
at 355 rather than 343, i.e. the board estimate about 5.8 s optimistic — on a
base of two 2025 airport-days, one per slice.

**3. Sanity of the candidate files.** Both clean: no NaN, no negative, none below
the floor, and an id set and row order identical to `submitting.parquet`. 28 rows
above 20,000 s, all LIRF, 27 of them `noac`, every one backed by an observable of the same size, the largest inside
2025's observed maximum. The 388 `noac` and 4,895 `nonm` rows sit on their
season's 2025 distribution. Per-airport monthly means are +120 to +150 s above
the 2025 winter truth at the five airports that need it and within 30 s at the
rest. 350 rows have members disagreeing by more than 1,800 s; 87 % of them are
`noac` or `nonm`, and five `nm_bulk` rows in 344,841.

**4. Label-free fixes.** Clipping a numeric column **that enters only as a tree
split** to the training range is **a provable no-op for every member of both
candidates except `D1-winter`** — measured at exactly 0.000 s of prediction
change on 2,947 and 6 affected rows for two different learners, and true by
provenance for the other two plain-tree members. It is live only for the one
member with linear leaves, and that row is provenance rather than a measurement
of this experiment. The exception the narrow statement protects: a column the
prediction is built from *arithmetically* — `mvt_aobt`, `mvt_sched`,
`d_takeoff_aobt` — is not covered, and none of the exposed day-level columns is
one of those. The unseen-level parent mapping
moves the affected predictions by ~64 s, helps january (CatBoost, CI spans zero)
and hurts july_loo (LightGBM, CI excludes zero), and is **not adopted** under the
frozen rule; it is recorded as a MAYBE with its revival condition. The LFPG
runway check found **48 rows** on a (stand, runway) pair 2025 never contains and
no ordering defect that survives comparison with the January observable, so no
per-airport monotone repair is proposed.

**5. `docs/candidate-audit.md`** is written, with the recommended order
(candidate 2 first, unchanged; its lead is 0.80 s reported and 0.87 s re-mixed,
and the 0.07 s of widening is one 2025 day on each side and not a finding) and
the reason.

## Why each result came out the way it did

**The v2 frame not extrapolating is not luck; it is a property of the feature
set.** Every numeric column in it is either a difference of two timestamps on the
row itself, a count inside a fixed window, or a calendar index. Twelve months of
2.1 M departures set the extremes of all three, and a disrupted month moves the
*density* inside them, not the support. The columns that do leave the range are
exactly the ones aggregated to an **airport-day** — a day mean has 310 draws
behind it in 2025, not 2.1 M, so its maximum is a much weaker bound and a
genuinely worse day walks straight past it. **The exposure lives wherever the
statistic is computed over days rather than over rows**, and that is a rule that
generalises past this candidate: any future feature aggregated to the airport-day
inherits it.

**The blend under-predicting past the envelope has the same cause seen from the
prediction side.** Every member fits a residual to take-off minus AOBT, so the
prediction tracks the observable one-for-one by construction and only the offset
is modelled. On a day past the envelope the offset's own drivers are out of
range, the tree answers from its top bin, and the answer is the offset of the
*worst day it ever saw* — which on a worse day is too small. That is why the bias
is negative and why it grows with the exceedance (−91 s at any exceedance, −144 s
past 30 s, −304 s on december's worst day).

**And that is why the clip is a no-op and the saturation is not.** The two facts
are the same fact: the tree has already collapsed everything past its top
threshold into one answer, so moving the input down to that threshold changes
nothing, and the error was never in the input's value — it was in the model
having no bin to put it in. A repair has to give the model a coordinate it can
extrapolate along, and the only ones available label-free are ranks within the
ranking month, ratios to the airport's own level, or the residual anchor the
members already use. That is the shape of the next experiment on this lane and it
is not a clip.

## Review round 1 — five blocking findings, all real, all fixed

Reviewer `claude-opus-5-xhigh-2026-09-05` (Opus substitute;
desk rule). Full transcript in `review/round1.txt`, prompt in
`review/prompt-round1.txt`. **Verdict: five blocking, eleven major, five
moderate, five minor.** Every blocking finding was a real error in this document,
and four of them were arithmetic or bookkeeping rather than method.

| # | finding | disposition |
|---|---|---|
| **B1** | the "4.6 s" headline is the abandoned two-slice re-mix while the table beside it is the three-slice one (+5.80 / +5.72) | **accepted**; 5.8 s throughout, the 4.63 figure kept and labelled as the coarser version |
| **B2** | "identity map for a plain tree ensemble" is too broad — `prc.tail_model` composes `mvt_aobt − tree(...)` and `_band_predict` switches on `mvt_sched`, so clipping those moves the prediction arithmetically | **accepted**; the claim is narrowed to columns that enter only as split features, the two exceptions are stated, and the exposed day-level columns are shown to be split features |
| **B3** | a +71.5 s mean-prediction difference and a +72.1 s RMSE delta were called "the same magnitude"; exp-007's interval [+3.5, +149.7] was suppressed | **accepted**; the two quantities are separated, the interval quoted, the confounds named |
| **B4** | "nine ranking rows" is the largest column count, not the union; the union is 20 | **accepted and already corrected** before the review returned, by the same self-check |
| **B5** | each re-mix slice RMSE rests on one 2025 airport-day, undisclosed | **accepted**; the two days are named, both error directions stated, and the 0.07 s of "widening" is retracted as a finding |

Of the eleven major findings, **M8 and M2 changed measurements** rather than
prose and both were re-run (`code/r1_fixes.py`):

- **M8**: the day-state exposure count was one-sided. Symmetric counting raises
  January from 15,842 to **19,895 rows** and July from 3,407 to **11,405**, and
  restores `ds_nm_rate_z`'s collapse — the column `docs/shift-2026.md` calls
  EHAM January's signature — to the table it was missing from.
- **M2**: the "worst 5 % of days" slice contained the out-of-envelope days, so
  the headline contrast was between a set and its own subset. Recomputing them
  disjointly **strengthens** the contrast: january's in-envelope severe days are
  biased **+15.4 s**, not −5.2, against −91.2 s out of envelope.

M11 (id order) and supplied-input rule (the 5,290 vs 4,895 gap) were unchecked claims and are now
measured; compute allocation (per-airport percentile shares) and recorded rule (July-only categorical
totals) were incomplete tables and are now complete. M1, M3, M4, M5, M6 and M7
were wrong numbers or wrong attributions in prose and are corrected in place,
each with a pointer to the finding. **M10 is the one worth reading**: this
document claimed exp-007 had not separated severe-in-range days from
out-of-envelope days, and exp-007 had — per day, in its own finding 13. What this
experiment adds is the label-free rule for telling them apart in advance, which
is the thing exp-007 said it could not do.

**M9 is closed with a measurement.** `code/q4d_clip_noop.py` delivered the
plain-tree half at full scale — 0 of 344,841 predictions changed at
`prc.tail_model.MAIN_PARAMS`, on the 1,422 rows the exposure is actually in — and
its `linear_tree=True` arm was stopped after an hour on a box at load 470 and
replaced by `code/q4e_linear_leaf.py`, the same contrast on a 400,000-row sample
at 150 rounds. **Plain 0 of 1,422 changed, linear 1,405 of 1,422**, with the
plain arm reproducing 0.000 at the smaller size so the instrument is checked.
Re-sizing the fit to the question rather than running it at full scale is the
reason this closed at all: the full arm would have taken hours for a yes/no.

The reviewer's other point on M9 stands and is worth keeping: the linear-leaf
observation is **not new** — exp-007's own round-2 review recorded it against the
same arm. What is new is the negative half (the clip is dead for the other four
members, measured three ways) and the **two-sidedness** of the linear case, which
changes what the repair is worth.

**recorded rule stands as a scope note.** The clip repair was measured on the v2 frame,
which has no exposure, because the desk routed the day-level clip to the members'
owners while this experiment was running. The hand-off section carries the rows
and the fold that can measure it.

### Two process errors of my own, recorded because both nearly shipped

**A count of column-hits was written as a count of rows.** "Nine ranking rows
past the twelve-month range" was the largest single column's count; the union is
20. Caught by a self-check against the CSVs before the review returned, but it
had already been written into `docs/candidate-audit.md`, which is the document
the desk uses to authorise an upload. Per-column tables do not sum to row counts
and a union has to be computed, not read off.

**Two statistics were compared across a type boundary.** A first version of the
LFPG runway check compared a per-cell prediction **mean** against a per-cell
observable **median** and read ρ = 0.04, which would have been a scrambled runway
response and a headline defect. Comparing like with like gives 0.56. The result
was in the report for one draft before the check that caught it.
