# exp-043-pilot — the surface-observation pilot: EHAM, 2025-07-07 to 07-13

Worker `pilot`, tier hard, 2026-09-08. Assignment `assignments/pilot.md` (recorded rule, Q16 step 2).
Reading and rejected readings: `SPEC.md`. Reviewer: see §7.

## 0. The verdict in four lines

adsb.lol **does** receive aircraft on the ground at EHAM: **<!--V:receipt_week-->96.5 %<!--/V:receipt_week-->** of the week's 5,131
departures are seen standing on the apron, at a **2 s** median inter-point interval, not the
30 s trace floor the coverage check feared. On the five **held-out** days the observed
stand-departure clock reconstructs the blanked `BLOCK_TIME_UTC_mvt` at **<!--V:rmse_obs-->144.2 s<!--/V:rmse_obs--> RMSE**
against **<!--V:rmse_aobt-->308.5 s<!--/V:rmse_aobt-->** for `AOBT_3_flt` on the same rows, and adding it to the served predictor
takes v10 from **142.8 s to <!--V:rmse_after_frozen-->122.43 s<!--/V:rmse_after_frozen-->** on those rows at an offset and amplitude frozen on the
development days. That is worth **<!--V:board_gain-->+1.06 s<!--/V:board_gain--> on the board from EHAM alone**, about half the
2.12 s gap to the lead. Only **15.5 %** of the held-out no-NM rows match on the callsign; over
the **whole week** (90 such rows, because only 58 fall on the held-out days) §4(d)
identifies <!--V:nonm_identified-->56 of 72<!--/V:nonm_identified--> of the rest with a geometric key that returns the right
aircraft on **97.7 %** of the held-out rows it assigns and answers **0.9 on average** at ten
time-shifted placebos, which puts no-NM receipt near **<!--V:nonm_estimate-->80.8 %<!--/V:nonm_estimate-->**. That is an estimate under
that key, not a bound: it says the tail is *mostly* an identity problem, not that receipt
is fine there.

## 1. What was fetched, and what it cost

Seven daily releases of `adsblol/globe_history_2025`, instance `planes-readsb-prod-0`, tags
`v2025.07.07` … `v2025.07.13`, two split assets each, **<!--V:gb_transferred-->24.5 GB<!--/V:gb_transferred-->** of published release
assets — the whole transfer budget
and nothing else. No 2026 data, no other airport, no other week, no upload.

The laptop's data volume had 4.5 GB free of 460 GB when the pilot started (and 2.4 GB when it
ended), so **the archive was never written to disk on either machine**: the split assets are
read over HTTP as one resumable byte stream and fed straight into `tarfile` in stream mode,
each member decompressed, screened and dropped inside the loop. Peak disk is the output slice.
The fetch ran as two Kaggle CPU kernels with internet on
(`kernels/surface-gate`, `kernels/surface-rest`, generated from the single source
`kernels/surface/prc-surface.py`); the retained slice, 30.7 MB, came back to the laptop.

**Provenance** is in `analysis/provenance.json` (the fourteen assets, their byte counts and
the SHA-256 GitHub publishes, all recorded before a byte was fetched) and in
`analysis/run_<day>.json` (the SHA-256 computed on the bytes the filter actually consumed).
Every asset that was hashed to completion **matches the publisher's digest**: both parts on
days 3–7, the first part on days 1–2. On days 1–2 the tar's end-of-archive marker stopped the
reader before the last bytes of the second part, so its hash never completed; the member
accounting shows each archive was read whole (<!--V:days12_accounting-->07-07 65,613 files, 3.358 GB of member payload from a 3.408 GB archive; 07-08 72,459 files, 3.518 GB of member payload from a 3.573 GB archive<!--/V:days12_accounting-->,
the rest being tar headers and padding). The kernel was fixed to drain the
stream, which is why days 3–7 certify both parts. Two trace files a day, out of the
<!--V:screened_range-->52,056 to 59,375<!--/V:screened_range--> that pass the byte screen, failed to parse as
JSON and are counted in `errors`.

| Day | Archive | Members | Decompressed | Aircraft in box | with ground points | Kernel |
| --- | --- | --- | --- | --- | --- | --- |
<!--T:fetch-->
| 2025-07-07 | 3.41 GB | 65,873 | 17.1 GB | 748 | 625 | 578 s |
| 2025-07-08 | 3.57 GB | 72,719 | 18.0 GB | 711 | 585 | 625 s |
| 2025-07-09 | 3.55 GB | 74,329 | 17.9 GB | 779 | 637 | 829 s |
| 2025-07-10 | 3.65 GB | 75,901 | 18.4 GB | 770 | 636 | 992 s |
| 2025-07-11 | 3.67 GB | 76,912 | 18.4 GB | 732 | 619 | 712 s |
| 2025-07-12 | 3.33 GB | 68,798 | 16.7 GB | 669 | 540 | 779 s |
| 2025-07-13 | 3.29 GB | 66,885 | 16.4 GB | 718 | 586 | 1,484 s |
<!--/T:fetch-->

The retained slice is `artifacts/exp-043/{points,census}_<day>.parquet`, addressed by
`analysis/manifest.json` (path, byte count, SHA-256 for all fourteen files).

## 2. Step 1 — the receipt census, and the gate

`analysis/census.json`. "Seen on the ground" means the departure's broadcast callsign appears
on at least one on-ground trace point inside the EHAM box (52.25–52.40 N, 4.65–4.85 E) that
day. The callsign is `CALLSIGN_flt` where the NM join exists and the airline-prefix
reconstruction otherwise; the prefix map is learned on the **eleven 2025 months that are not
July**, so it is never fitted on the week it is scored on.

| Day | Departures | ICAO24 on ground | Ground points | Points per aircraft-day (median) | Median interval | ≤ 35 s | Receipt | on NM rows | no-NM rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!--T:census-->
| 2025-07-07 | 739 | 625 | 305,020 | 396 | 2 s | 97.6 % | 95.9 % | 97.8 % | 21 |
| 2025-07-08 | 709 | 585 | 291,486 | 394 | 2 s | 97.7 % | 96.5 % | 97.7 % | 11 |
| 2025-07-09 | 728 | 637 | 293,574 | 367 | 2 s | 97.6 % | 96.6 % | 97.4 % | 6 |
| 2025-07-10 | 752 | 636 | 296,574 | 380 | 2 s | 97.7 % | 95.5 % | 97.1 % | 17 |
| 2025-07-11 | 763 | 619 | 290,610 | 375 | 2 s | 97.7 % | 97.1 % | 98.5 % | 12 |
| 2025-07-12 | 693 | 540 | 266,385 | 412 | 2 s | 97.7 % | 95.8 % | 97.5 % | 15 |
| 2025-07-13 | 747 | 586 | 293,936 | 400 | 2 s | 97.7 % | 98.0 % | 98.9 % | 8 |
| **Week** | **5,131** | — | **2,037,585** | — | **2 s** | — | **96.5 %** | **97.9 %** | **90 → 20.0 %** |
<!--/T:census-->

**The gate passes with room to spare** (96.5 % against a 40 % threshold), so the stop after two
days was not taken and days 3–7 were fetched.

**Two things this table says that the coverage check could not.**

- *The 30 s trace floor is not the operating resolution.* readsb guarantees a point every 30 s
  and writes one on every significant change; a taxiing aircraft changes state constantly, so
  the median gap between ground points is **2 s** and 97.7 % of gaps are 35 s or less. The
  resolution of the off-block instant is limited by what the event means, not by the sampling.
- *A callsign census cannot tell a missing aircraft from a missing name.* NM-joined rows are
  seen <!--V:receipt_nm-->97.9 %<!--/V:receipt_nm--> of the time and no-NM rows <!--V:receipt_no_nm-->20.0 %<!--/V:receipt_no_nm-->, but "seen" here only means that the same
  callsign string appeared on an EHAM ground point that day, and for a no-NM row the callsign
  is a reconstruction that is wrong whenever the operator uses an unrelated ATC callsign
  (`exp-042` §2.2 predicted 22.3 % on exactly those rows). **This table therefore cannot
  support a claim about no-NM receipt, and §4(d) measures it a different way instead.**

**Definition, stated because it is a choice.** A departure counts as seen on day *d* only if
its callsign appears on a ground point of day *d*. Pooling the week's callsigns instead would
let a Sunday observation mark a Monday departure as seen and would report 98.4 %; that figure
was in the first version of this file and is wrong.

## 3. Step 2 — the match, and the clock check

`analysis/match-heldout.json`, `analysis/match-week.json`, code in `code/segments.py` and
`code/match.py`. Identity is `exp-042` REPORT §3: the modal callsign of the ground segment that
ends in a take-off, joined to the week's EHAM departures with the observed wheels-off inside
±15 min of `MVT_TIME_UTC_mvt`. Ambiguity is resolved by nearest wheels-off and counted, never
dropped: 121 departures had two candidate segments over the week and none had two candidate
departures. A `first_airborne` point whose altitude is `null` is not read as a take-off: a
null is an absent measurement, not evidence of flight. Every matched row's observation date agrees with its block-time date, so the
`(callsign, date)` form of the join would have lost nothing here.

| | Held out (days 3–7) | Whole week |
| --- | --- | --- |
<!--T:match-->
| Departures | 3,683 | 5,131 |
| Matched | 3,525 (**95.7 %**), day range 95.5 %–95.9 % | 4,901 (95.5 %) |
| Matched, NM rows | 97.0 % | 96.9 % |
| Matched, no-NM rows | **15.5 %** (9 of 58) | 20.0 % (18 of 90) |
| With an off-block instant | 3,485 | 4,827 |
<!--/T:match-->

**The clock check passes.** `obs_wheels_off − MVT_TIME_UTC_mvt` has median **−26 s** with
**97.9 %** (3,452 of 3,525) of matched rows inside 60 s (1st–99th percentile −90 s to −4 s). The last on-ground
ADS-B point sits a few seconds before the recorded take-off, which is what rotation looks like.
The identity match and the clocks are therefore sound, and the off-block number may be read.

### The one decision the data forced

The assignment defines `obs_offblock` as the first sustained motion (ground speed above 3 kt
held 30 s) **after the last stationary period at the stand**. Measured on days 1–2, the network
sees a departing aircraft for a median of **<!--V:seen_before_3kt-->161 s<!--/V:seen_before_3kt-->** before that 3 kt onset — and
**<!--V:seen_before_250m-->284 s<!--/V:seen_before_250m-->** before the 250 m one the pilot ends up freezing — and not at all before that:
transponders come on near pushback. *(These were quoted as a single 150 s until round 4;
`sweep.json` stores all three now, including the stand-stop margin's
<!--V:stand_margin_median-->50 s<!--/V:stand_margin_median-->, which is the shorter stationary RUN immediately before the onset and
is not the same quantity.)* A 600 s stand stop would have kept only
<!--V:stand600_kept-->3.5 %<!--/V:stand600_kept--> of the development days' departures (the arm is in `sweep.json`; the share is
of departures, not of matched segments, where it is <!--V:stand600_kept_matched-->3.6 %<!--/V:stand600_kept_matched-->), and anchoring on the segment's first
motion instead would have caught the previous **arrival's** taxi-in whenever an aircraft is
received continuously through its turnaround. The rule used is therefore: anchor after the last
stationary run of at least 600 s if there is one, otherwise at the start of the segment; take
the first sustained motion after the anchor; and report the observed stationary time before it
as the confidence margin, not as a filter. `code/segments.py` carries the reasoning.

**Days 1–2 are the development side and days 3–7 are held out.** The rule and its constants
were chosen on days 1–2 only, and the reach, accuracy and alignment numbers in §4(a)–(c) are
measured on days 3–7 with them frozen. The rule did **not** change on the held-out days. **One
number in §4 is not a held-out measurement and says so where it appears**: the no-NM identity
probe in §4(d) counts the whole week's 90 no-NM departures, because there are only 58 of them
on the held-out days and 9 matched; what is held out there is the probe's own accuracy, which
is measured on held-out rows and applied to that week-wide count.

## 4. Step 3 — the three numbers

### (a) Reach

**95.7 %** of held-out EHAM departures (day-clustered: 95.5–95.9 % leave-one-day-out,
95.0–96.4 % over the exact five-day resampling scheme), **97.0 %** on NM rows, **15.5 %** on
no-NM rows. The bulk is carried on the callsign; the tail is not, and §4(d) says why.

### (b) Accuracy — the decisive number

RMSE of `BLOCK_TIME_UTC_mvt − obs_offblock` after removing the EHAM median, against
`AOBT_3_flt` on **the same rows** so the comparison is paired. The interval column is the
**exact** day-clustered resampling scheme over the five held-out days (all 5⁵ = 3,125
resamples enumerated, not sampled. Five clusters do not make a textbook 95 % confidence
interval; it is a spread, and the paired table below carries the leave-one-day-out range for
the quantity that actually carries the claim.

| Held-out days 3–7, NM rows | n | Median offset | **RMSE** | day scheme 2.5–97.5 | 1st–99th pct |
| --- | --- | --- | --- | --- | --- |
<!--T:accuracy-->
| **observation, 250 m stand-departure clock** | 3,476 | -291.5 s | **144.2 s** | 141.1–147.1 | -669 … +58 |
| observation, assignment's 3 kt / 30 s rule | 3,476 | -191.6 s | **172.7 s** | 168.2–177.5 | -582 … +173 |
| observation, 3 kt with an observed stand stop | 1,612 | -210.3 s | **161.1 s** | 147.0–173.3 | -616 … +141 |
| `AOBT_3_flt` on the same rows | 3,476 | 114.0 s | **308.5 s** | 263.6–351.0 | -286 … +1434 |
<!--/T:accuracy-->

The quantity that carries the claim is the **paired difference**, recomputed inside every
resample rather than differenced across two separate intervals:

| Paired RMSE difference, held out | point | day scheme 2.5–97.5 | leave-one-day-out |
| --- | --- | --- | --- |
<!--T:paired-->
| 250 m clock − `AOBT_3_flt` | **-164.2 s** | -205.2 … -121.0 | -179.9 … -143.7 |
| 3 kt / 30 s rule − `AOBT_3_flt` | **-135.8 s** | -174.3 … -94.7 | -150.7 … -116.6 |
<!--/T:paired-->

Every held-out day favours the observation — the leave-one-day-out range never reaches zero —
and the two error series are only weakly related
(correlation **+0.24**), so they measure different things rather than the same thing twice. On
the whole week the same pair is 144.0 s against 321.0 s. `exp-042` set "near 120 s would be a
larger single mechanism than anything in the record" as the bar; the observation lands at
144 s on its own and reaches 122 s in §4(c) when combined with the model that already exists.
**What this number is:** the dispersion of a centred clock difference on the rows the source
reaches, not an all-EHAM prediction accuracy — the <!--V:no_clock_share-->5.4 %<!--/V:no_clock_share--> of held-out departures that end
with no usable clock (unmatched, or matched with no off-block instant) are outside it, and §5
prices them at their measured reach.

**Why the median offset is −291 s, and why that is not a defect.** The airport's recorded block
time is about five minutes *before* the aircraft is 250 m from its stand. Pushback runs at
2–5 kt behind a tug, so any robust motion test fires after the block event, and the offset
absorbs it. The sweep (`analysis/sweep.json`, development days only) shows the structure
plainly: a 0.5 kt gate puts the median offset at **<!--V:half_knot_median-->+0.7 s<!--/V:half_knot_median-->** — the block time *is* the first
perceptible motion — but its RMSE is **<!--V:half_knot_rmse-->219 s<!--/V:half_knot_rmse-->**, because at that threshold the trigger is GPS
jitter at a random moment. Accuracy improves as the event becomes more decisive, to a flat
plateau reached by 6–8 kt or by 200–300 m of displacement (200 m 143.8 s, 250 m 143.4 s,
300 m 143.3 s), and degrades again beyond it (9 kt 155.3 s; 500 m 148.7 s; 1,200 m 156.8 s) as
the event slides toward the take-off — `sweep.json` stores <!--V:disp_arms-->11 arms from 20 m to 1,200 m<!--/V:disp_arms-->,
so the plateau and both of its edges are in the record. **The pilot's finding is that a later, unambiguous event
is a better clock than an earlier, noisy one**, and the frozen rule takes the midpoint of the
measured plateau (250 m) rather than an endpoint. The assignment's own 3 kt / 30 s rule is
reported in every table beside it; the 250 m rule is a refinement, not a substitution.

### (c) Alignment on the shared measurement contract

`analysis/align-*-heldout.json`. `r = y − p_v10` and `d = p_obs − p_v10` with
`p_obs = MVT_TIME − obs_offblock − median_offset`, on the `july_loo` fold, 3,485 matched rows.
The fold-side v10 comes from `exp-038/code/vectors.py`, which rebuilds it from committed
rotations and gates it on exp-036's published clean triple (284.1572 / 204.9218 / 218.1506);
nothing was refitted. **EHAM is not in exp-036's kept set, so the vector this update would move
at EHAM is v7's, which is what v10 serves there.** v10 scores **142.75 s** on these rows.

The contract's `A`, `B` and `2A − B` are defined on the **unscaled** `d`, and are reported that
way first. `2A − B` is negative for every rule (<!--V:contract_2amb-->-448.1 s²<!--/V:contract_2amb--> for the frozen 250 m
clock): at full amplitude the update overshoots, which
is the expected shape when a new observation is informative but noisier than the served vector.
The gain arrives at the fitted amplitude, and the amplitude is never fitted on the rows it
scores.

| Held out, 3,485 rows | A = mean(r·d) | B = mean(d²) | **2A − B** (unscaled) | A/B |
| --- | --- | --- | --- | --- |
<!--T:contract-->
| 250 m clock | 11,358.6 | 23,165.2 | **-448.1 s²** | 0.4903 |
| assignment's 3 kt rule | 13,842.3 | 37,102.5 | **-9,418.0 s²** | 0.3731 |
| 3 kt with a stand stop (1,617 rows) | 12,079.3 | 31,070.8 | **-6,912.1 s²** | 0.3888 |
<!--/T:contract-->

| Scaled update, held out | offset and scale fitted on | MSE gain per matched row | day range (s²) | v10 → |
| --- | --- | --- | --- | --- |
<!--T:scaled-->
| **250 m clock, frozen on days 1–2** | the 1,342 development rows | **5,388.1 s²** | 4,902–5,751 | 142.75 → **122.43 s** |
| 250 m clock, leave-one-day-out cross-fit | the other four held-out days | **5,550.9 s²** | 4,228–7,283 by day | 142.75 → **121.77 s** |
| 3 kt rule, frozen on days 1–2 | the 1,342 development rows | **5,164.4 s²** | 4,819–5,445 | 142.75 → **123.34 s** |
| 3 kt rule, leave-one-day-out cross-fit | the other four held-out days | **5,157.3 s²** | 4,046–6,456 by day | 142.75 → **123.37 s** |
<!--/T:scaled-->

**The frozen row is the headline** because both its median offset (−286.9 s) and its amplitude
(0.5724) come from days 1–2 and never touch the days they score; the cross-fit row is carried
beside it because it is what a five-day pilot can also say, and the two agree to 3 %. Its
day-clustered interval, on the same exact 5⁵ scheme the accuracy table uses, is **4,404–6,408
s²** over the resampling spread and 4,902–5,751 s² leave-one-day-out, so the RMSE after the
update is 122.43 s with a leave-one-day-out range of 121.20–124.36 s. Every resample is
positive. Neither arm is a ranking replay: the ranking months' labels are blank, so nothing
here can prove 2026 transfer, and the `A`, `B` and `2A − B` above use the offset pooled over
the scored rows. Moving to a leave-one-day-out offset moves `A` by
<!--V:regime_A-->0.027 %<!--/V:regime_A--> and `B` by <!--V:regime_B-->0.016 %<!--/V:regime_B-->, but `2A − B` by <!--V:regime_2AB-->0.53 %<!--/V:regime_2AB-->, because it is a
near-cancellation of two large numbers and a small absolute move in either is a large
relative one there. *(This paragraph claimed "less than 0.02 %" for all three until round 5;
`align.py` computes the three moves now.)*

### (d) The no-NM rows — and the measurement that separates receipt from identity

These rows carry about 60 % of the naive predictor's squared error and they are the reason the
source was wanted, so this is where the pilot has to be most careful.

| Held-out no-NM rows | n | RMSE |
| --- | --- | --- |
<!--T:nonm-->
| Departures with no NM join | 58 | — |
| Matched on a reconstructed callsign | 9 (15.5 %) | — |
| With a 250 m clock | 9 | 160.4 s |
| With the 3 kt clock | 9 | 136.2 s |
<!--/T:nonm-->

Nine rows support no interval worth printing and no alignment number at all. **The measured
result on the no-NM tail is the reach, not the accuracy.**

A callsign census cannot say *why* the other 49 failed: a wrong reconstruction and a missing
observation look identical. `code/nonm_probe.py` separates them **without using the callsign at
all**. For each of the week's 72 unmatched no-NM departures it looks for an *unclaimed*
segment — one no other departure of the week matched — whose take-off is within **120 s** of
that row's `MVT_TIME_UTC_mvt` and whose trace aircraft type equals its `AIRCRAFT_TYPE_mvt`,
and it solves a maximum **one-to-one** assignment over the whole set. Both fields are supplied
by the ranking file, so the same test is legal there.

**The key is fixed on the development days and scored on the held-out ones.** On 07-07/08
alone, `|wheels-off − MVT_TIME|` is 115.1 s at the 99th percentile — rounded up to the next
10 s, that is the 120 s window — and the trace type equals `AIRCRAFT_TYPE_mvt` on 96.9 %. No
held-out row took part in either choice. The key is then run on held-out callsign-matched rows,
in draws of 72 (the size of the real question) against the same free segment pool plus those
rows' own segments, and each row is scored on whether the matcher returned **its own** segment
rather than merely some segment.

| Week, 90 no-NM departures | count |
| --- | --- |
<!--T:probe-->
| the key, on held-out rows whose answer the callsign already gave: assigned the RIGHT segment | 13,588 of 13,915 assigned (**97.7 %**), 94.4 % of all 14,400 rows |
| the same key, 500 rows against the same 988-segment pool | 96.8 % right — the number of competing rows is not what matters |
| the same key in round 3's own setup: 500 rows against every claimed segment (4,901) | 90.2 % right — the POOL's density is what matters, and round 3's reading is reproduced |
| no-NM departures | 90 |
| matched on a reconstructed callsign | 18 |
| unmatched | 72 |
| **of those, identified one-to-one by the key** | **56** (median 1 candidate, max 2) |
| the same key at 10 time-shifted placebos (±30 … ±120 min) | **0.9 on average, 0–3** |
| **expected correct at the held-out precision** | **55 of 72** |
| **no-NM receipt, ESTIMATE under this key (not a bound)** | **80.8 %** |
| *(for the record: the withdrawn ±15 min no-type test)* | *68 of 72, up to 16 candidates a row* |
<!--/T:probe-->

**So the tail is mostly an identity problem.** Fifty-six of the 72 unmatched rows are
identified with a median of one candidate each; the same key at ten placebo offsets from
±30 to ±120 minutes answers 0.9 rows on average and never more than 3, so the identifications
are not chance. On held-out rows whose answer is known, the key returns the **right** segment
for 97.7 % of the rows it assigns, so about 55 of the 56 are expected to be correct and the
no-NM receipt estimate is **<!--V:nonm_estimate-->80.8 %<!--/V:nonm_estimate-->**, four times the 20.0 % the callsign census reports.
Where the key does identify a row, the aircraft is on the apron, received, and simply not
named — `FLIGHT_mvt` is a commercial flight number and an ATC callsign such as
`LH022 → DLH3VA` is not derivable from any open table. *Expected to be*, not *is*: the
calibration records wrong assignments as well as right ones, so the claim is about the 56 as a
group, not about any one of them.

**That number is an estimate under this key, and a bound in neither direction.** It can be
low — a received aircraft whose type string or clock differs is missed, and the key leaves
part of the known held-out rows unassigned for exactly that reason (row 1 of the table). It
can be high — a small share of the rows it assigns still get the wrong aircraft. What it supports is "the no-NM tail
is *mostly* identity"; what it does not support is "receipt is not a problem there".

**Why this differs from round 3's own audit, measured rather than argued.** Round 3 ran the
same key with 500 known rows against **every claimed segment** and found 55 of them pushed onto
another aircraft's segment. Rows 1–3 of the table above reproduce that setup and separate the
two things that differ from the pilot's real question. Raising the row count alone costs almost
nothing; raising the **pool** from the free segments to every claimed one costs seven points,
because a dense pool offers a same-type decoy inside the window. The pilot's question is 72
unmatched rows against the free pool, so the first row is the calibration that applies. Round 3
is right about its own setup, all three arms are stored in `analysis/nonm_probe.json`, and the
disagreement is a measurement rather than a dispute.

**Two earlier versions of this test were withdrawn.** Version 1 asked only whether *some*
segment the callsign matcher had not used sat within ±15 min, and answered 69 of 72 — reported
here as 96.7 % receipt. Round 2 showed that is not a receipt estimate: the window admits up to
16 candidates for one row, the assignment was not one-to-one, and a half-hour placebo still
answered yes 55 times. Version 2 fixed the window and the assignment but counted rows the
matcher assigned to *any* segment as recoveries and tested only two placebo offsets, which
made "at least 82.2 %" unsupportable. Both withdrawn numbers stay visible — the last line of
the table above, and this paragraph — rather than being deleted.

**This is also the repair, demonstrated.** A geometric match — take-off instant plus aircraft
type, and `STAND_mvt` is still unused — recovers most of the tail that the callsign cannot
reach. Turning that into predictions for the ranking file is a new experiment, not a repair of
this one, and it is the single highest-value follow-up this pilot points to.

## 5. Step 4 — what it would be worth, and what it would cost

`analysis/extrapolate.json`. Arithmetic on the numbers above plus two facts from the record:
v10 scored **263.0464** on 2026-09-08 with the lead at **260.9310** (a 2.1154 s gap), and the
ranking file holds **38,182** EHAM departures of **344,841** (11.07 %), of which 1,206 have no
NM join. Projecting the measured per-group reach onto that composition gives **<!--V:eham_reach-->93.3 %<!--/V:eham_reach-->** of EHAM
ranking rows covered, or 35,641 rows — **10.3 %** of the file. The gain applied is the
**frozen-on-development** <!--V:gain_frozen-->5,388 s²<!--/V:gain_frozen--> per matched row.

| | Pooled MSE reduction | Board after | Gain | Share of the gap to the lead |
| --- | --- | --- | --- | --- |
<!--T:board-->
| **EHAM only, measured reach and gain** | 557 s² | **261.99** | **+1.06 s** | 50 % |
| Every airport at EHAM's reach and gain *(not measured)* | 5,030 s² | **253.31** | **+9.74 s** | 461 % |
<!--/T:board-->

This is a **conditional projection, not a measured board move**. It holds only if July 2025
reach and gain transfer to January and July 2026, and only if post-event 2026 observations are
allowed at all — an open rules question (`exp-042` §5), not something this pilot settles.

**Cost.** The archive is **global**: one pass over a day yields all ten airports, so the cost is
per *month*, not per airport-month — ten boxes cost exactly what one box cost here. The two
ranking months are **213.2 GB** and about **14.8 Kaggle CPU kernel-hours** at this pilot's
measured <!--V:gb_per_day-->3.50 GB<!--/V:gb_per_day--> and 857 s per day. Fitting each airport's median offset needs training months
too, at 108.6 GB and about <!--V:month_hours-->7.4<!--/V:month_hours--> kernel-hours for a 31-day month at the same rate — the
"about 2 h" here until round 5 did not follow from the measured 857 s per day. The 1.32 TB
figure in `exp-042` covers all of 2025 **plus both ranking months**, not 2025 alone; the
programme that buys the table above is 213 GB plus as many 2025 months as the offset needs.

### What this pilot cannot say

1. **Nine airports are unmeasured.** EHAM sits in the densest feeder region in Europe. LTFM,
   LEMD and LIRF are the cases where a crowd-sourced network is most likely to miss a parked
   aircraft, and they are 32 % of the ranking file. The +9.74 s row above is an *if*, and the
   only honest way to make it a number is to run one day at each of the other nine airports.
2. **2026 is unmeasured.** Calendar coverage is complete (`exp-042`: Jan 31/31, Jul 31/31), but
   receipt in 2026 is not the same measurement as receipt in July 2025.
3. **January is unmeasured.** The gain is measured on the `july_loo` fold. January 2026 is a
   disrupted winter month outside the 2025 envelope at four airports (`docs/shift-2026.md`),
   and de-icing and low-visibility pushbacks are exactly the conditions under which the
   block-time-to-motion interval would widen.
4. **One week is one week**, and five day-clusters are five. Every interval here is reported
   as both a resampling spread and a leave-one-day-out range for that reason.
5. **Nothing here is a ranking replay.** The ranking months' labels are blank, so no arm of
   this pilot can prove 2026 transfer.
6. **The rules questions are still open** (`exp-042` §5): whether post-event 2026 observations
   are acceptable at all, and whether approval-gated OpenSky access counts as openly
   accessible. Those are the organizer's to answer and Matt's to ask. Nothing here was
   submitted or uploaded, and the pilot itself touched 2025 training data only.
7. **A production join must not take its date from the block time.** `code/cohort.py` derives
   each departure's day from `BLOCK_TIME_UTC_mvt`, which the ranking file blanks. Every pilot
   row has the same block-time and take-off UTC date, so nothing here changes, but any
   generalisation must key on `MVT_TIME_UTC_mvt`.

## 6. Ledger rows

<!--V:ledger_rows-->13<!--/V:ledger_rows--> rows in `docs/LEDGER.md`, all matching `exp-043` in their experiment column: the
receipt census and the three numbers, a correction pair after each of the five BLOCKED review
rounds, and the round-6 **CLEARED** verdict row. The ledger is append-only, so a correction is
a new row; each names the round and the reviewer that judged it. The count is generated from
`docs/LEDGER.md` by `code/report_tables.py`, so it cannot fall behind the ledger again.

## 7. Divergence from the rule the harvest ships

`analysis/divergence.json`, `code/divergence.py`. **The experiment is `exp-045-harvest`** (this
section said `exp-044` until round 3; there is no `exp-044-harvest`). Its first kernels ran
`segments.py` as it stood on 2026-09-07. **Two repairs have been made to that rule since, and
the pilot and the harvest now both run both of them.**

- **The altitude repair** (Sol round 1, harvest `8906ea1`). The shipped rule reads *any* first
  non-ground point after a ground run as a take-off; readsb writes `null` for a position with
  no altitude at all, and a null is not evidence of flight, so the repair requires a real
  altitude. `segments.py` keeps both under `REQUIRE_ALT`.
- **Stable point ordering** (Sol round 4, harvest `afc77ee`). Points that share an `(icao, t)`
  exist at the UTC-midnight overlap and a few disagree about `on_ground`, so an unstable sort
  can split a ground run in two places. The harvest measured **14 and 17 segments moving
  between this laptop and the Kaggle image** on its two stored days — the same code, the same
  points, two answers — and switched to a stable sort. **The pilot ran the unstable sort until
  round 4 found it**; `segments.py` now sorts stably and keeps both under `STABLE`.

**The current state is that the harvest ships `repair-2026-09-08+stable-order` and so does this
experiment.** `code/allsegments.py` imports `REQUIRE_ALT` from this file's `segments.py` and
runs it `True`, `code/manifest.py` names the rule version of every stored month, and a
re-derive kernel backfills months already run from their stored points. The table below is not
a difference between the pilot and the harvest — there is none — it is the price of the two
repairs, so a month's summaries still carrying `shipped-2026-09-07` can be read for what they
are until they are re-derived. The earlier text here said the harvest keeps the shipped rule and
that the shipped column is what its summaries carry; both were true when written and neither is
now.

| Over the whole week | shipped 2026-09-07 | + stable order | + altitude repair | current (both) |
| --- | --- | --- | --- | --- |
<!--T:div_week-->
| Ground segments ending in a take-off | 5,395 | 5,395 | 5,389 | 5,389 |
| Matched departures | 4,906 | 4,906 | 4,901 | 4,901 |
| Rows scored (NM, paired with `AOBT_3`) | 4,815 | 4,815 | 4,810 | 4,810 |
| RMSE of the 250 m clock | 143.94 s | 143.94 s | 144.00 s | 144.00 s |
| Paired difference against `AOBT_3` | -177.01 s | -177.01 s | -176.98 s | -176.98 s |
<!--/T:div_week-->

| Held out, days 3–7 | shipped 2026-09-07 | + stable order | + altitude repair | current (both) |
| --- | --- | --- | --- | --- |
<!--T:div_held-->
| Matched departures | 3,529 | 3,529 | 3,525 | 3,525 |
| RMSE of the 250 m clock | 144.16 s | 144.16 s | 144.23 s | 144.23 s |
| Paired difference against `AOBT_3` | -164.34 s | -164.34 s | -164.24 s | -164.24 s |
<!--/T:div_held-->

**The price is nothing.** Six segments in 5,395 and a seven-hundredths-of-a-second move in the
decisive number: the repair is right in principle — a null altitude is an absent measurement,
and reading it as flight would matter at an airport where reception is thinner — and it is
below the resolution of anything this pilot or the programme decides. Every number elsewhere in
this file uses the current rule, and the four columns agree to within 0.1 s on every quantity.
The six divergent segments are listed in `analysis/divergence.json` with their ICAO24, callsign
and take-off instant, so the harvest can find them without re-deriving anything.

**The stable-order column costs nothing, and `analysis/ordering.json` says why.** By the time
`segments()` runs, `census.load_slices` has removed every duplicate `(icao, t)`, so there is no
tie left for a sort to break: under a stable sort, an unstable sort and a shuffled input order
the segment table comes out **<!--V:order_equivalent-->identical<!--/V:order_equivalent-->**. The repair is a guard
against an input that does have ties — which is the input the harvest reads, one day file at a
time without de-duplication, and where it measured the 14 and 17 moved segments.

**That the pilot and the harvest run the same rule is measured too**, on the pilot's own
de-duplicated week: <!--V:harvest_identical-->5,389 shared, 0 only in the pilot, 0 only in the harvest, 0 of 11 columns differing<!--/V:harvest_identical-->.
The scope of that check is the de-duplicated slice, which is what every number in this file
uses; on a raw day file the two can still differ, because the pilot keeps the first copy of a
tied point and the harvest keeps both. One consequence worth passing on:
`exp-045-harvest/code/check_rule.py` sets tied-timestamp aircraft aside from its assertion
precisely because the pilot sorted unstably, and that exemption can now be removed.

## 8. What the review changed

This experiment was reviewed fresh-context five times on 2026-09-08 — four rounds by
`gpt-5.6-sol` at `xhigh`, and round 5 by the **FALLBACK** reviewer `claude-fable-5-1` at effort
high, because `gpt-5.6-sol` was not available. All five returned **BLOCKED for
ledger use**: five blocking findings, then three, then four, then three, then two. All five are
in `review/`. **The direction of the result never moved through any round**: every version of
the receipt census passes the gate, and every version of the accuracy comparison favours the
observation on every held-out day. What moved, five times, is how much the record is entitled
to say — and three of those rounds blocked on a repair of mine that claimed more than it had
measured, which is the pattern worth carrying out of this experiment.

### Round 1 (`review/round1.txt`, 5 blocking)

| # | Finding | Was | Is |
| --- | --- | --- | --- |
| 1 | Pooled receipt used a week-wide callsign union, so a Sunday sighting could mark a Monday departure as seen | 98.4 % | **<!--V:receipt_week-->96.5 %<!--/V:receipt_week-->** (`census.py` pools the daily hits); the per-day column was always right |
| 2 | The scaled gain was printed as the contract's `2A − B`, which the contract defines unscaled | one number | **A, B and the unscaled `2A − B` in their own table**, the scaled update separate (§4(c)) |
| 3 | The 121.5 s figure cross-fitted the amplitude inside the held-out set | one arm | **two arms**: frozen on days 1–2 (**<!--V:rmse_after_frozen-->122.43 s<!--/V:rmse_after_frozen-->**, the headline) and the cross-fit, both in the §4(c) scaled table |
| 4 | Day-clustered intervals were missing on reach and alignment and unpaired on accuracy | RMSE only, sampled | **exact 5⁵ enumeration**, a paired-difference interval, leave-one-day-out ranges, reach intervals |
| 5 | The ledger claimed a Sol verdict before one existed | premature | correction rows appended; this section is the record |

Round 1 also fixed, as non-blocking: the §4(b) table mixed development and held-out `AOBT_3`
values; `sweep.json` did not store the arms the prose cited; day files share points across the
UTC midnight boundary and were concatenated without deduplication; a `first_airborne` point
with a `null` altitude was read as a take-off (§7 prices that one against the shipped rule);
the clock-check rate divided by all departures instead of matched rows; and `provenance.json`
lacked the licence URL and the other fields `docs/external-data-rules.md` requires.

### Round 2 (`review/round2.txt`, 3 blocking + 1 standards)

Round 2 confirmed the development freeze is genuine ("the 3,485 July 9–13 rows have no
overlapping movement IDs and only score the frozen update"), reproduced the contract quantities
and the frozen gain independently, and confirmed the pooling, deduplication, null-altitude,
clock-denominator and provenance fixes. It blocked on three things.

**1. My round-1 repair of the no-NM claim over-claimed, and it was the worst finding of either
round.** Asking whether *some* unclaimed segment sat within ±15 min answered 69 of 72, but that
window admits up to 16 candidates for a single row, the assignment was not one-to-one, and the
reviewer's own half-hour placebo still answered yes 55 times. The number was not a receipt
estimate. The test is now a 120 s window plus aircraft-type agreement, solved one-to-one,
calibrated on rows whose answer is known (486 of 500) and controlled by a placebo that
identifies **zero**. It answered **56 of 72**, read at the time as no-NM receipt of at least
82.2 % rather than 96.7 %. *(Round 3 showed both the zero placebo rate and the lower-bound
reading were themselves over-claims; see round 3 finding 1 below and §4(d) for what replaced
them.)* The withdrawn figures are kept visible in the §4(d) table and in that paragraph.

**2. `RESULTS.md` disagreed with its own regenerated JSON**, because tables had been typed by
hand while the analysis was rerun beneath them. Proof-reading is not a fix for that, so every
numeric table in this file now sits between `<!--T:name-->` markers and is rendered by
`code/report_tables.py` from `analysis/`. *(Round 3 showed this claim was too strong while the
prose still carried hand-typed numbers and two headers disagreed with their rows; see round 3
finding 2.)*

**3. The frozen headline gain had per-day values but no interval.** It now carries the same
exact 5⁵ scheme as the accuracy table: 4,404–6,408 s² resampling spread, 4,902–5,751 s²
leave-one-day-out, every resample positive.

**4. (Standards) The correction rows cited the round-1 reviewer, whose verdict was BLOCKED.**
Two further correction rows are appended for round 2, and the verdict line below names the
round that cleared them.

Round 2 non-blocking, also done: the offset regime is labelled on the `A`/`B` block, and
`seg_key` is defined once in `segments.py` instead of being rebuilt as a string in two places.
Two are recorded rather than changed: the join key is the callsign plus the ±15 min window
rather than `(callsign, date)` — every held-out match agrees on date, so the assignment's form
would have lost nothing — and the original census ledger row's "publisher SHA-256 verified" is
retracted in a correction row, because two of the fourteen assets were not hashed to completion
(§1 has always said which).

### Round 3 (`review/round3.txt`, 4 blocking)

Round 3 confirmed the parts the earlier rounds had repaired — the frozen 5⁵ interval
reproduced independently to four decimals, the 250 m rule frozen before days 3–7 were fetched,
the prefix map free of July, all twelve marked tables byte-identical to the renderer's output,
no ranking pipeline using a blanked label as a feature or a join key, and the paired accuracy
comparison genuinely row-paired with every held-out day favouring the observation. It blocked
on four things.

**1. The no-NM number over-claimed for the second time, and this was again the worst finding.**
Three faults, all real: the key was calibrated on the same rows that reported the check; the
check counted rows the matcher assigned to *any* segment rather than to the *right* one, so
486 of 500 was never an accuracy; and two placebo offsets cannot establish a zero false-match
rate. §4(d) is rebuilt: the key is fixed on the development days alone, its accuracy is scored
on held-out rows for returning **that row's own** segment, ten placebo offsets are swept, and
the number is **80.8 % as an estimate under this key** in place of "at least 82.2 %". Round 3's
own 500-row audit is reproduced rather than argued with, and the three arms show the driver is
the pool's density, not the number of competing rows.

**2. Generating the tables did not stop the prose from drifting.** Two rendered tables declared
more header cells than their rows carried; the round-1 review table sat outside every marker;
the 0.5 kt figure said +0.9 s where `sweep.json` says +0.7 s; and §6 still said four ledger
rows when there were six. `report_tables.py` now renders **inline values** as well as tables,
between `<!--V:name-->` markers — the receipt rates, the transfer size, the ledger row count,
the headline RMSEs, the frozen gain, the contract's `2A − B`, the board gain — and it **refuses
to write** a table whose header, separator and rows do not all carry the same number of cells.
That check is demonstrated rather than asserted: `report_tables.py --selftest` injects each
defect into a copy of this file in memory and requires the check to name the table. The claim that a disagreement was "no
longer possible" was false when it was made and is withdrawn in a correction row.

**3. The transfer size was the wrong quantity.** The seven days' fourteen published assets are
**24.5 GB**; 24.1 GB is the sum of tar member payloads, which leaves out 382.3 MB of tar
headers and block padding. `extrapolate.py` now reports both under their own names, and the
check that `GB_PER_MONTH['2025-07'] / 31 = 3.503 GB` against the pilot's measured 3.498 GB
confirms the per-month constants were always published-asset bytes.

**4. §7 described a state that had stopped being true.** It named `exp-044-harvest`, which does
not exist, and said the harvest keeps the shipped rule. The harvest is **`exp-045-harvest`**,
and since `8906ea1` it ships the altitude repair, names the rule version of every stored month
in its manifest, and has a re-derive kernel for the months already run. §7 now prices what the
months still carrying `shipped-2026-09-07` differ by.

Round 3's own non-blocking list was three confirmations, not a list of defects. One item that
appears in its working notes is answered with the record rather than a change: `sweep.json`
does store displacement arms past 200 m — eleven of them, from 20 m to 1,200 m — so the
plateau and both its edges were always supported; §3 now says so with a generated count. *(That
item came from round 1's list, not round 3's; round 4 corrected the attribution.)*

### Round 4 (`review/round4.txt`, 3 blocking)

Round 4 confirmed the round-3 repairs: the no-NM key genuinely free of held-out information,
its per-row scoring correct, its 80.8 % arithmetic sound and its qualification the right one;
all 12 tables and all rendered inline values matching `analysis/`; the column check firing on
header, separator and row defects with its self-test passing; the transfer arithmetic correct;
and no blanked label reaching the ranking pipeline. It blocked on three more.

**1. `segments.py` sorted points with pandas' default, which is not stable.**
`exp-045-harvest` had already found this the expensive way — the same code and the same points
gave **14 and 17 different segments** on this laptop and on the Kaggle image — and had switched
to a stable sort; the pilot had not. `segments.py`, `match.py` and `sweep.py` now sort stably,
and the whole chain was re-run from the stored slices (`code/rerun.sh`). §7 prices both repairs
in four columns instead of one, and no longer says the harvest keeps the shipped rule.

*How far the defect actually reached is a measurement, not an argument, and round 5 blocked on
the version of this paragraph that argued it.* `analysis/ordering.json`, `code/ordering.py`:
the week's day files hold <!--V:ties_before-->5,217<!--/V:ties_before--> rows in a tied `(icao, t)` group, of which
<!--V:ties_disagreeing-->72<!--/V:ties_disagreeing--> groups disagree about `on_ground` — those are the ones an unstable sort could
split differently. But `census.load_slices` has de-duplicated `(icao, t)` since the **round-1**
repair, and after it the week holds <!--V:ties_after-->0<!--/V:ties_after--> ties. With no ties left the sort key
determines the order completely, and running `segments()` under a stable sort, an unstable
sort, and a shuffled input order gives **<!--V:order_equivalent-->identical<!--/V:order_equivalent-->** output on all
three. So: only the two original ledger rows, measured before that de-duplication, were ever
exposed to the machine; every number in the record since is not, and the stable sort is a guard
against a future input that does have ties rather than a repair of a wrong number. The earlier
text here said the 144 s "could have been a property of the machine" and explained the
unchanged numbers by quicksort's determinism; both were wrong, and both are withdrawn in a
correction row.

*Every step that reads order rather than value was audited, not just the sort that was found.*
The three `sort_values` calls are stable. `census.load_slices` de-duplicates the UTC-midnight
overlap by keeping the first copy, so it now reads the days in calendar order whatever order
the caller passes. The two `mode()` calls that pick a modal callsign are order-independent
because pandas returns the modes sorted, so a tie resolves the same way on any input order.
Nothing else in `code/` selects a row by position.

**2. Three more prose numbers were stale, and one ledger claim had gone out of date.** The
share of held-out departures with no usable clock was given as 4.3 % where it is
<!--V:no_clock_share-->5.4 %<!--/V:no_clock_share-->; the stand time before motion was one number, 150 s, where there are three
different quantities and none of them is 150 s; and the ledger's "16 inline values" became 24
as the generator grew. The first two are now generated; the third is corrected in a ledger row,
and the lesson is the round-3 lesson again — a count quoted in prose has to be read from the
thing it counts.

**3. Two sentences were still stronger than the test behind them.** §4(d) said an identified
row *is* on the apron and simply not named; the calibration records wrong assignments as well
as right ones, so it now says *expected to be*, about the 56 as a group. §3 said every §4
number is measured on held-out days; §4(d)'s count is week-wide (there are only 58 no-NM rows
on the held-out days), and what is held out there is the probe's accuracy. Both say so where
they appear.

Round 4 also corrected the review history itself: round 1 had **ten** non-blocking findings,
not nine, and the sweep-arm item above came from round 1's list rather than round 3's.

### Round 5, FALLBACK (`review/round5-claude-fable-5-1-high-2026-09-08-FALLBACK.txt`, 2 blocking)

`gpt-5.6-sol` was not available for this round, so round 5 ran on the
`MODES.md` fallback reviewer — `claude-fable-5-1` at effort high, fresh context, one shot, the
same prompt, read-only — and **its verdict is recorded as a FALLBACK verdict** in the file, in
this section and in the ledger rows that cite it. It confirmed all 12 tables and all rendered
inline values against `analysis/`, the column check, every withdrawal having a retracting row,
the no-NM probe and its calibration, the stand-time quantities, the order audit, and that no
blanked label reaches the ranking pipeline. It blocked on two things, both of them claims about
repairs rather than measurements.

**1. My round-4 correction over-stated the reach of the sort defect, and cited a log that
crashed.** The correction row said "every measurement in this lineage was a property of the
machine that produced it". It was not: the round-1 de-duplication repair had already removed
every tied `(icao, t)` before `segments()` sees the points, so only the two original ledger
rows were ever exposed. The reviewer proved it by shuffling the input; `code/ordering.py` now
makes that a stored measurement of this experiment rather than a finding in a review file, and
the paragraph above is rewritten around it. The same row cited `logs/rerun-stable.log` as
evidence the chain was re-run — that log ends in a `KeyError` at the render step, because
`divergence.json`'s schema had changed under it. The chain was re-run cleanly
(`logs/rerun-clean.log`) and the row is corrected.

**2. The count of generated values was itself typed by hand, in two places that disagreed.**
The round-4 correction row claimed the count "is itself generated now"; it was not, and §8 said
24 where the generator returned 25. It is generated now — `<!--V:value_count-->36<!--/V:value_count-->` values, rendered
from `values()` by counting itself. This is the third round to block on a hand-typed number.
The rule the experiment ends with, stated as what the mechanism actually does rather than as an
ambition: **every table and every value inside a marker is rendered from `analysis/`, and the
rendered set grows each time a review finds drift.** Round 6 checked the roughly forty prose
numbers still outside a marker against `analysis/`, `machine_check.json` and the ranking file
and found them all correct — but "all correct today" is a weaker guarantee than "generated",
and the honest statement of the difference is this sentence.

Round 5's non-blocking findings are all repaired: the §4(c) offset-regime claim said "less than
0.02 %" for three quantities where `2A − B` moves by <!--V:regime_2AB-->0.53 %<!--/V:regime_2AB--> (it is a near-cancellation, so a
small absolute move is a large relative one) and `align.py` computes the three moves now; the
cost section's "about 2 h per month" did not follow from 857 s a day and is
<!--V:month_hours-->7.4<!--/V:month_hours--> kernel-hours; the 1.32 TB figure covers 2025 **plus** both ranking months, not 2025
alone; the 600 s stand-margin share is quoted with its denominator; §0 says the 72 is a
week-wide count; and this section's intro no longer says three rounds. One finding is outside
this experiment and went to the desk instead: `docs/LEDGER.md` row 420 (`exp-046 step 1`) cites
this pilot's match at 4,930 and 3,547, which the altitude repair moved to 4,901 and 3,525.

### Round 6, FALLBACK (`review/round6-claude-fable-5-1-high-2026-09-08-FALLBACK.txt`) — **CLEARED**

Round 6 was scoped to round 5's two conditions. It returned **CLEARED, 0 blocking**: both
correction-5 withdrawals complete and true with nothing in `RESULTS.md`, `SPEC.md` or `code/`
still stating either; `analysis/ordering.json` supporting every sentence §7 and §8 build on it,
including the raw-day-file scope caveat; `code/ordering.py` resolving `census` and `segments` to
the pilot's modules and restoring `STABLE` after each arm; `logs/rerun-clean.log` the whole
chain, seventeen steps, no traceback, ending in `chain complete`; all 12 tables and all 36
values matching `analysis/` at 43 marker occurrences; and each of the eight non-blocking
repairs correct. **The twelve `exp-043` ledger rows may stand.**

Its seven non-blocking findings are answered here. Three were claims that over-described their
mechanism and are rewritten: the "every prose number is rendered" rule above, and superseded
sentences in the docstrings of `report_tables.py`, `segments.py` and `divergence.py`. Two are
recorded rather than changed: "0 ties after de-duplication" is true by construction and the
measurement that carries the claim is the three-way equivalence test, and round 5's "2,796
duplicate rows" and this record's "5,217 rows in a tied group" are the same count under
`keep=False` (5,217 − 2,421 = 2,796). One is a wording repair in `SPEC.md`. **One belongs to
other workers and went to the desk, not into this record**: `exp-045-harvest/RESULTS.md` still
says the pilot sorts unstably and its `check_rule.py` still sets tied aircraft aside, and
`docs/LEDGER.md` row 464 says the pilot's `segments.py` does not sort stably — it has since
commit `09998b5`, which predates that row.

**Verdict line:** `reviewer: gpt-5.6-sol-xhigh-2026-09-08`, round 1 **BLOCKED** (5 blocking,
10 non-blocking), round 2 **BLOCKED** (3 blocking, 1 standards), round 3 **BLOCKED** (4
blocking), round 4 **BLOCKED** (3 blocking), and `reviewer: claude-fable-5-1 FALLBACK
2026-09-08` round 5 **BLOCKED** (2 blocking, 8 non-blocking), and `reviewer: claude-fable-5-1
FALLBACK 2026-09-08` round 6 **CLEARED** (0 blocking, 7 non-blocking), each answered above.
**Round 6 is the round that clears the record**, and it is a FALLBACK verdict: the primary
reviewer `gpt-5.6-sol` was not available from 12:45 UTC,
so rounds 5 and 6 ran on `claude-fable-5-1` at effort high,
fresh context, one shot, read-only. Every ledger row names the round and the reviewer that
judged it, and none borrows a verdict from a round that found its defect.
