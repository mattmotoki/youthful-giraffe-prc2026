# The 2026 shift — what changed between training and the ranking months

The exp-003 analysis; the tables behind every
number are in `experiments/exp-003-anatomy/analysis/q4_shift.md` and
`experiments/exp-003-anatomy/analysis/q4b_airport_shift.md`. Source: the live `ranking.parquet` (344,841 departures,
SHA-256 `9db75e3a…`) split by take-off time into 152,719 January and 192,122
July rows, against the twelve 2025 training files.

## One paragraph

**July 2026 is an ordinary summer month; January 2026 is not an ordinary winter
month.** Every 2026 category level that matters is already in 2025 (new stands
carry 0.46 % of rows, new runways 2 rows), the airport mix moves by less than a
point, and take-off minus AOBT in July 2026 (mean 1004 s, std 419) sits on top
of July/August 2025. January 2026 is a disruption month at the northern
airports: take-off minus AOBT is more dispersed than in any 2025 month at
**EDDF (375 s against 330), EHAM (740 against 476), EDDM (355 against 294) and
LSZH (348 against 311)**, marginally so at LFPG (491 against 485), and on a
third of the days at those airports it is outside the whole 2025 range. LTFM and
LEMD have an elevated *mean* but not an elevated spread (LTFM 298 s against
416 s in February 2025), so they are busy months, not disrupted ones. Two structural changes also need saying: **CDG
lost runways 27R and 09L** (13.8 % of its 2025 departures, ~0.1 % in 2026) and
**Munich gained stands 103–108** (1,177 July-2026 rows that no 2025 row can
teach). The practical consequences are at the bottom.

## Which 2025 months resemble each ranking month

Mean total-variation distance over nine categorical profiles (airport, stand,
runway, operator, aircraft type, wake, segment, destination, take-off hour):

| ranking month | closest 2025 months | mean TVD |
|---|---|---|
| 2026-01 | **2025-12**, 2025-01, 2025-11 | 0.043 / 0.051 / 0.052 |
| 2026-07 | **2025-08**, 2025-07, 2025-09 | 0.051 / 0.053 / 0.057 |

By Wasserstein distance on the numeric columns the answer is the same for July
(2025-08 then 2025-07 on `MVT−AOBT`, `MVT−SCHED`, `AOBT−LOBT`). For January the
two rankings disagree: the *composition* is closest to December 2025, the
*operational conditions* (`MVT−AOBT`) are closest to January 2025, and neither
is close. Composition and conditions are different questions and January 2026
answers them with different months.

## January 2026 is outside the 2025 envelope

Daily mean of take-off minus AOBT per airport, each 2026 day placed inside that
airport's distribution of 2025 days:

| airport | mean 2025 | mean 2026-01 | median 2026-01 day's percentile in 2025 | days above 2025 p95 | days above the 2025 maximum |
|---|---|---|---|---|---|
| EDDF | 917 | 1040 | 0.95 | 16 / 31 | **9** |
| LSZH | 683 | 787 | 0.95 | 17 / 31 | 2 |
| LTFM | 1291 | 1416 | 0.86 | 7 | 0 |
| EDDM | 731 | 785 | 0.85 | 11 | 1 |
| EHAM | 939 | 1207 | 0.53 | 11 | **9** |
| LEMD | 981 | 991 | 0.64 | 7 | 3 |
| LFPG | 983 | 1027 | 0.18 | 7 | 2 |
| LIRF | 931 | 925 | 0.52 | 0 | 0 |
| EGLL | 1385 | 1378 | 0.39 | 3 | 0 |
| LEBL | 887 | 877 | 0.40 | 1 | 1 |

Dispersion moves with it. Std of take-off minus AOBT, January 2026 against
January 2025, raw and with the top and bottom 0.1 % trimmed: EHAM **740 / 713**
vs 476 / 449; EDDF 375 / 366 vs 330 / 280; LSZH 348 / 320 vs 311 / 230; EDDM
355 / 338 vs 294 / 274. The gap survives trimming and widens at three of the four
airports, so this is a change in the body of the distribution and not a handful of
outliers — unlike the LIRF July figure below. EHAM in January 2026 is a different
regime, not a heavier draw from the same one — the NM-join rate there also falls to 95.0 %, its lowest
value anywhere in the data (2025 range 97.5–98.9 %).

July 2026 shows none of this: every airport's median day sits between the 11th
and 87th percentile of 2025 and no day exceeds a 2025 maximum. ~~The one July
anomaly is LIRF, and it recurs: std of take-off minus AOBT is 737 s in July
2026 and 751 s in July 2025.~~ Refuted by the LIRF analysis (2026-09-05,
`experiments/exp-006-lirf/analysis/q1_july_regime.md`): the July 2025 figure is
one date-rollover row; the trimmed std is 241 s against 179 to 219 s in other
months. Rome's real July effect is congestion (AOBT minus LOBT median 120 s in
January, 600 s in July) driving block minus AOBT, with the same conditional
response in both seasons; at LIRF the no-NM and noac rows are the same event.

## Structural changes

**Runways.** Share of the airport's departures, 2025 / 2026-01 / 2026-07:

- **LFPG 27R 9.0 / 0.0 / 0.1 and 09L 4.8 / 0.0 / 0.1** — the northern doublet is
  gone in both ranking months; 27L absorbs it (11.2 / 29.5 / 20.4). 13.8 % of
  CDG's 2025 departures used a runway that has essentially disappeared. Stand-to-
  runway taxi distance is the physics of this task, so a CDG model trained on the
  2025 mix is trained on routes that are no longer flown.
- **EHAM** flips configuration: 18L 22.4 / 37.4 / 8.6, 36L 23.6 / 11.7 / 40.8,
  18C 0.8 / **6.2** / 0.1. January 2026 is a south-flow month, July 2026 a
  north-flow month; 2025 is a mix of both.
- **LTFM** 18 11.6 / **30.3** / 2.6 and 17R 7.7 / **19.5** / 1.9 — January 2026
  runs south flow half the time against 19 % in 2025.
- **EDDM** 26L 34.5 / 32.8 / **47.4**, 08L 19.9 / 25.2 / **9.7**.
- Only one runway level is new (`LFPG|NA`, 2 rows) and only `LTFM|34R` (308
  rows in 2025) retired.

**Stands.** 45 stand levels appear in 2026 that never appear in 2025, carrying
1,588 rows (0.46 %): **EDDM 103–108** (1,177 rows, all July 2026 — a new apron),
EDDF J5/J10 (202) and V333/335/337 (97), LSZH T50 (38). Twenty stands with 200+
2025 rows have vanished: **LIRF 501/502/503/505/507** (~10,000 rows), LSZH
A42/H11–H14 (~7,700), EDDM 120/121, LFPG E10/E13/CFW. Per-stand statistics need
the fallback chain (stand → parent stand → terminal prefix → airport), which is
what `prc.stand_features.StandStats` does; without it 0.46 % of ranking rows get
no stand statistic and the EDDM new apron gets none at all.

**Everything else is stable.** Airport shares move by at most 1.6 points (LTFM, January 2026). New
operators: 30 levels, 177 rows. New aircraft types: 8 levels, 18 rows (F35, PL2,
IL62 — military and general aviation). New destinations: 62 levels, 174 rows.
NM-join rate 98.39 % (Jan) and 98.52 % (Jul) against 97.9–99.4 % in 2025 — the
`noac` rate is 0.071 % and 0.146 % against 0.034–0.178 % in 2025, i.e. July 2026
sits at the July-2025 level and January 2026 above every winter month.

The take-off-hour profile difference between the ranking months is daylight
saving, not behaviour: 2026-01 matches 2025-01/11/12 hour for hour and 2026-07
matches 2025-04..10.

## How the team uses this analysis

1. **Do not treat `december` as the January-2026 fold.** December 2025 is the
   closest month by composition and one of the calmest by conditions; a model
   validated only there will be optimistic about January 2026 at EDDF, EHAM,
   LSZH and EDDM. The backward `january` fold (train Feb–Dec, validate Jan 2025)
   is the only fold with winter conditions in it, and even it is milder than
   January 2026 at EHAM. Report `july`, `december` **and** `january`; treat the
   spread between them as the honest uncertainty band, not the mean.
2. **Weight training months by ranking month, not uniformly.** For the July half
   of the submission, 2025-06..09 carry the composition and the conditions. For
   the January half, 2025-12/01/11 carry the composition but *no* 2025 month
   carries the conditions — so for January the right move is to keep all twelve
   months (to see enough disrupted days) rather than to weight toward December.
   Two models, one per ranking month, is defensible on this evidence; a single
   model with month features is the cheaper first cut.
3. **Give the model the runway, and do not let it lean on stand alone at CDG.**
   The 27R/09L retirement means the stand→runway pairs at LFPG in 2026 are a
   different set from 2025. A stand-only encoding transfers; a stand-only
   *taxi-distance* assumption does not.
4. **Expect the January half of the leaderboard score to be worse than any
   local fold predicts,** and size the gap by the EHAM/EDDF dispersion ratio
   (740/476 ≈ 1.55 at EHAM, 375/330 ≈ 1.14 at EDDF) rather than by the mean.
