# Taxi-out time at ten European airports: our method

Team youthful-giraffe, PRC Data Challenge 2026. Version of 2026-10-08.

This document describes the model behind our final entry, **v73, 223.3071 s RMSE** (uploaded
2026-10-08 00:01 UTC, file SHA-256 `70236754…`, section 4.10). v73 is our earlier entry v64
(225.7082 s, `9cd421dd…`, section 4.9) plus three groups of steps: five steps on rows outside the
linear stack (c5), half of a level-2 correction on clock rows (L2C), and a step toward an
airport-context transformer on ordinary rows (ORD-S). v64 is the file `e7b04ffd…` (v62,
225.9106 s), which adds two members and the time-column fix Z to our earlier best v59 (225.8463 s)
and removes one rule, plus one fixed step on rows that the member stack does not reach (T11b-L).
On 8 October we stopped the experiment work and kept v73 as the final entry; it is not refitted
before the model freeze (2026-10-08 12:00 UTC).

Board scores and file hashes are in `docs/PUBLIC-SUBMISSIONS.md`, one row per upload; we cite it by
row number. Member recipes are in `docs/LINEAGE-ZABD.md#members`, the rule stages in
`docs/LINEAGE-ZABD.md#rule-stages`, the steps after v64 in `docs/LINEAGE-ZABD.md#from-v64-to-v73`,
and the external data sources in `docs/DATA.md`. The other
measured numbers (local reads, intervals and counts) come from our internal experiment records,
which are not part of this export; we give them as plain text. The list of cited files is at the
end.

## Summary

- **Task:** reconstruct the taxi-out time (take-off minus airport off-block) of 344,841 departures
  at ten airports in January and July 2026; RMSE.
- **Final entry:** v73, 223.3071 s, 10th of 237 teams on the 06:52 UTC board of 8 October, from
  265.7612 s at our first upload (section 5).
- **What made the large steps:** observing the off-block event in open ADS-B ground tracks (v13,
  −14.26 s), and reading the rules by which the file records the label: the plan-join window, the
  LIRF schedule placeholders and date rollovers, and stand re-use bounds (sections 4.6–4.8).
- **Model:** a month-hold two-level stack of gradient-boosted members, half-blended with our
  September entry; a LightGBM clock head on rows with an ADS-B clock; members of other kinds (an
  event-sequence transformer, an MLP, runway GRU sequence models with a row control, and a twin
  trained with the clock hidden on some days) in a linear stack per class; then deterministic rules
  and one projection onto hard bounds; then a fixed quarter step toward the dark-day twin on typed
  rows without a plan (v64, section 4.9); last, the c5, L2C and ORD-S steps of v73 (section 4.10).
- **Validation:** two whole-month folds (January and July 2025), every fitted stage without labels
  of the held month, G in s² per 2026 row with airport-day intervals. Measured cumulatively, the
  board gains of the stack entries were 0.80–0.86 of the local G; every stack step but one had the
  local sign on the board. From v68 to v73 the board gain was 1.03 of the local G (section 3.3).
- **No model value comes from a board score.**

## 1. The problem and the metric

**Target.** For each departure, predict the taxi-out time in seconds:
`TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`, the take-off time minus the airport
off-block time (`docs/input-rule.md`, section "Why this is the reading").

**Scored rows.** 344,841 departures at ten airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF,
LSZH, LTFM; `prc/gscore.py`): 152,719 in January 2026 and 192,122 in July 2026 (`docs/input-rule.md`,
"Facts a model must respect"). No private split is announced; the score on these rows is the ranking score.

**Metric.** Root mean squared error over all scored rows, in seconds. Because the metric is a
square, a few rows with a very large error can dominate it: in July 2025, 0.34 % of departures have
a taxi-out time above 3,600 s (`docs/input-rule.md`), and much of our work goes to those rows.

**What the ranking file gives.** The ranking file has the same columns as the training files. Only
the departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blank. The take-off time of the
departure, the Network Manager (NM) actual off-block time `AOBT_3_flt`, the flight-plan times
(`LOBT_flt`, `EOBT_1_flt`, `IOBT_flt`; `prc/data.py`), and the complete timings of all other
arrivals and departures in the scored months are present. The task is therefore a reconstruction
after operations, not a forecast before departure (`docs/input-rule.md`, "The rule" and "Why this is
the reading"). The airport block
time and the NM off-block time are different measurements: in July 2025, `BLOCK − AOBT_3` is
within 60 s on only 20 % of rows, and its spread is airport-specific (`docs/input-rule.md`). Much of
the task is to model that offset.

**Rules we follow.** Every supplied column is a legal input. We do not use leaderboard scores or any
value derived from them to set a weight, threshold, mask, residual or rate
(`docs/input-rule.md`; `docs/VALIDATION.md` section 1). External data must be open, documented and
reproducible (`docs/external-data-rules.md`).

## 2. The data

### 2.1 Challenge data

| File | Content | Source |
|---|---|---|
| `training_2025-MM-01_….parquet`, 12 files | All movements of 2025 at the ten airports, with labels; 2,085,047 labelled departures | `prc/data.py`; `docs/VALIDATION.md` section 2 |
| `ranking.parquet` | 344,841 departures of January and July 2026 without labels, plus the arrivals of those months | `docs/input-rule.md` |
| `submitting.parquet` | The submission template (row order and IDs) | `docs/input-rule.md` |

We use the re-issued ranking files of 2026-09-04 (ranking SHA-256 `9db75e3a…325f86`); the files of
2026-08-13 are stale (`docs/input-rule.md`, "Data version").

**Three row classes.** The model treats three classes of departure differently
(`docs/VALIDATION.md` section 5):

| Class | Definition | Ranking rows |
|---|---|---:|
| clock | known aircraft type and an observed off-block clock from open ADS-B ground positions (section 2.2) | 196,256 |
| ordinary | known aircraft type, no observed clock | 148,197 |
| noac | no aircraft type in the file | 388 |

1.5 % of ranking departures have no NM join, so they have no `AOBT_3_flt` or plan times
(`docs/input-rule.md`). At LIRF the block time of such a row is often a schedule placeholder or a
time on another date; section 4.6 describes how we treat them.

### 2.2 Open external data

| Source | What it gives the model | Licence | Where | Record |
|---|---|---|---|---|
| adsb.lol `globe_history` archives (2025 and 2026) | Ground positions of each flight: an observed off-block clock (first lasting displacement from the stand), its quality, the stand position, the observed taxi path and time | ODbL-1.0 on the archives; CC0-1.0 on the contributed data | GitHub releases `adsblol/globe_history_2025` and `adsblol/globe_history_2026`, anonymous HTTPS | `docs/EXTERNAL-DATA.md` |
| OurAirports `airports.csv`, `runways.csv` | Airport boxes (runway ends and reference point, grown by 3 km) for the ADS-B match | Public domain | `ourairports.com/data/` | `docs/EXTERNAL-DATA.md` |
| X-Plane Scenery Gateway `apt.dat` | Stand and runway geometry, used to time the first ground point back to the stand (implied off-block clock) | GNU GPL version 2 (upload terms) | `gateway.x-plane.com` | `docs/GEOMETRY-DATA.md`; licence text `docs/data-licences/GPL-2.0.txt` |
| OpenStreetMap | Stand positions where the Gateway has no match | ODbL-1.0 | `openstreetmap.org` | `docs/GEOMETRY-DATA.md` |
| Copernicus ERA5 (single levels, time series and gridded) | Airport-hour weather and runway-aligned wind, in the older base chain only (section 4.2) | CC BY 4.0 | Copernicus Climate Data Store, DOIs 10.24381/1cf1ad76 and 10.24381/cds.adbb2d47 | `docs/weather-provenance.md` |
| OPDI v0.0.2 flight events | Times of the exit from the parking position, the exit from the apron and the entry to the taxiway, used as lead and distance inputs in the clock head and the sequence models | The pages and files name no licence; the OPDI About page states "open access and open use of data for non-commercial purposes" | `opdi.aero/data` | `docs/DATA.md#opdi` |

Attribution. adsb.lol: "Data © adsb.lol contributors, made available under the Open Database Licence
(ODbL-1.0) with the contributed contents under CC0-1.0." OpenStreetMap: "© OpenStreetMap
contributors". ERA5: "Contains modified Copernicus Climate Change Service information 2026. Neither
the European Commission nor ECMWF is responsible for any use of this information." The full ERA5
notice with the product DOIs and access dates, and the transformations, are in
`docs/weather-provenance.md`.

OPDI declares no file licence. The organizer allowed OPDI for this challenge when its purpose is
declared; the purpose is stated in the table above (`docs/DATA.md#opdi`).

**Not used.** We do not use OpenSky state vectors; they are not an approved open source for this
challenge (`docs/external-data-rules.md`). An hourly METAR test (Iowa
Environmental Mesonet, public domain) was measured and is not in the model (section 6). The
EUROCONTROL published monthly taxi-out means were used for a constant shift in entries v37, v40 and
v41; that shift is off in the current entry.

The organizer's answer on the task permits open observations of the scored months after
operations, including the flight's own ground positions (`docs/external-data-rules.md`).

## 3. Validation

### 3.1 Two whole-month folds

Local validation holds out whole calendar months of 2025. The scored rows are whole months of
2026, and our first protocol, which held out five week-pairs inside January and July 2025
(`docs/VALIDATION.md`), let a level-2 stacker learn from other weeks of the held month and chose a
model that lost on the board (section 6, item 3). We kept that protocol's code for the buffers, the
label-isolation checks and the scorer, and replaced its five week-pairs by two whole-month folds:

- **Fold J** holds January 2025 and **fold L** holds July 2025. Each has one buffer day before and
  after the held month; buffer labels are excluded from every fitted stage of that fold
  (`docs/VALIDATION.md` section 2, month holds).
- Every fitted stage of a fold — members, level-2 stackers, blend weights, rule statistics — fits
  on label rows outside the held month. The members fit without both held months. The later
  stages break the stronger rule of `docs/VALIDATION.md` section 3 (every fitted ancestor excludes
  the held month) through two paths. (1) The older entry v32 in the held route (section 9). (2) Reused
  fitted inputs: the blend weights that serve one held month are fitted on the other held month
  (`experiments/exp-169-stack-plan/code/dark_twin035.py` lines 18 and 29). A stage that scores held
  January then fits on July values that carry January labels: the ORD-S coefficients
  (`w2/ords_ld.py` lines 92-95), L2C (`w2/l2c_candidate_a.py` lines 61-67) and the statistics of
  C95 and C98 (`exploit_rules.py` lines 32 and 77); July is the same with the months exchanged.
  The held reads of these stages (C95 and C98 in section 4.8, L2C and ORD-S in section 4.10) and
  the held reads of later steps that run through them (section 4.11) carry this path. Its size is
  not measured. It touches the held 2025 reads only; no 2026 label exists.
- Context features (queues, neighbours, day state) are built from a frame in which every departure
  block time and taxi time is blank, as in the ranking file (`docs/VALIDATION.md` section 4).
- The local score is the two held months joined: 344,419 departures.
- A stack weight for held January is fitted on held July, and the reverse. The weights served on
  2026 are fitted on both held months pooled.

### 3.2 The local gain statistic G

We compare a candidate with a base on the held rows by G, in s² per ranking row
(`docs/VALIDATION.md` section 6; code `prc/gscore.py`):

    d_i = (y_i − base_i)² − (y_i − cand_i)²         (positive = the candidate is better)
    G   = Σ_h W_h × mean_h(d_i),   W_h = N_h / 344,841

The cells h are month × airport × class, and the weights W_h are the 2026 ranking counts of each
cell. G is thus the expected change of the 2026 mean squared error if the 2025 cell gains carry
over. A cell with fewer than two held 2025 rows contributes zero (for example January LEMD clock
rows: one 2025 row against 15,881 in 2026). We select a candidate on G > 0 over all rows of the
two held months.

Every read also reports:

- a 95 % interval from 2,000 resamples of airport-days within each airport × month, with a fixed
  seed and a frozen array of draws, so every candidate sees the same draws (`docs/VALIDATION.md`
  section 7);
- G without the five airport-days that contribute most ("wo5"), because a few days with very long
  taxi times can carry a gain;
- G by month, class and airport.

The intervals measure 2025 day-to-day variation only, not the change from 2025 to 2026
(`docs/VALIDATION.md` section 7).

### 3.3 Alignment with the board

Before each upload we froze the file hash and wrote a forecast of the board score; after the board
read we wrote the actual score. The table compares the local G with the board gain, written as
the change of mean squared error (ΔMSE, s² per row) between two board scores; the scores are in
`docs/PUBLIC-SUBMISSIONS.md`:

| Upload | Step | Local G | Board ΔMSE gain | Board / local |
|---|---|---:|---:|---:|
| v40 | nine-month level-2 route, half blend and published-mean shift | +394 | +468.5 | 1.19 |
| v45 | a known-worse level 2 (alignment check) | −400 | −529.3 | 1.32 |
| v47 | rebuilt clock head on clock rows | +807 | +1,136.7 | 1.41 |
| v48 | rows without a plan at LIRF (section 4.6) | +1,440.4 | +1,187.3 | 0.82 |
| v49 | stack2 | +2,837 | +2,261.0 | 0.80 |
| v50 | stack4 | +4,287 | +3,699.7 | 0.86 |
| v51 | stack6 | +5,028 | +4,127.6 | 0.82 |
| v52 | stack7 | +5,511 | +4,648.0 | 0.84 |
| v52 → v57 | stack8 to stack12 | +439.2 | +362.0 | 0.82 |
| v69 | L2C: half of a level-2 correction on clock rows (section 4.10) | +112.8 | +172.9 | 1.53 |
| v70 | ORD-S with the s1 and s2 transformers | +78.5 | +51.1 | 0.65 |
| v71 | ORD-S: a second s2 seed, LIRF July at half the coefficient | +50.1 | +62.7 | 1.25 |
| v72 | ORD-S: s3 in place of s1 | +71.2 | +25.8 | 0.36 |
| v73 | ORD-S: s7 at 3 seeds in place of s2 and s3 | +95.1 | +106.4 | 1.12 |
| v69 → v73 | ORD-S, the four uploads together | +294.9 | +246.0 | 0.83 |
| v68 → v73 | L2C and ORD-S | +407.7 | +418.9 | 1.03 |

Rows v48 to v52 are cumulative against v47 (`docs/PUBLIC-SUBMISSIONS.md` rows 47–52). For v48, on
the 383 rows it changes, the table uses the local value of the ladder, +1,440.4; the fold read on
the LIRF rows without a plan was +2,144.3 (section 4.6). Rows v69 to v73 are steps against the
previous upload (`docs/PUBLIC-SUBMISSIONS.md` rows 68–73); for v71 to v73 the local G is the
difference of two cumulative ORD-S reads.

**Selection in these reads.** Each local G in the table includes choices that we made on the same
held January and July rows that give the read, so each read is high by an amount that we did not
measure in full. v40 keeps the level-2 form that the read of v35 chose among 16 variants, and its
shift reads published monthly means that contain the labels of the held month (section 2.2). We
chose the pair of v45 for its large local difference. v47 has the weight 0.3, one of three weights
read on the held rows (section 4.3). v48 has the limits of the no-plan rules (section 4.6). v49 to
v57 have head settings, route variants, rule limits, member recipes and seeds that we chose on the
held months (sections 4.2 to 4.8). v69 to v73 have the forms of L2C and ORD-S (section 4.10). The
board ratios below are ratios to these reads.

What the table shows:

- The stack uploads v49–v58, v61 and v62 moved the board in the direction of the local sign. v59
  changed only the 2026 route and has no held read. v60, the time-column fix Z (section 4.4), did
  not: its held read was +251.70, and the board lost ΔMSE 103.6 against v59. Z was the better of
  two repair forms read on the same held months. Among the earlier month-hold uploads, one step on
  the 384 noac rows did not either (v41; section 6, item 4).
- Measured cumulatively against v47, the board gain of the stack entries is about **0.83** of the
  local G (range 0.80–0.86), not the 1.24 measured on the earlier fold-form pairs. Single steps
  vary more: 0.63 to 1.69 on the 10-04 ladder (v53–v57), and 0.57 to 1.37 on the steps of the 10-05
  ladder that have a held read and the local sign (v58, v61, v62). The five uploads of the 10-04
  ladder landed within 0.08 s of forecasts made with 0.83.
- The five uploads of 6 October (v63–v67) were whole entries read against e7b04ffd (section 5).
  The four with a held read moved the board in the direction of the local sign, with ratios from
  0.16 to 8.9: v64 0.94, v65 0.16, v66 0.67, v67 8.9 (`docs/PUBLIC-SUBMISSIONS.md` rows 62–67).
  Each of the four held reads includes choices made on the held months (section 5).
- The steps of v69 to v73 (section 4.10) all had the local sign on the board. The clock-row step
  L2C kept 1.53 of its local G; the four ORD-S uploads kept 0.36 to 1.25, and 0.83 together.
  From v68 to v73 the board gain was 1.03 of the local G. v68 has no ratio: most of its held read
  sits on 6 rows, and two of its five steps change only 2026 cells without held rows.
- Two later uploads lost against v73. v75 read +86.4 locally, with a form chosen among four read on
  the same held data, and lost ΔMSE 91.6 on the board (section 4.10). v74 read +1,044.7 against its
  own parent (an earlier form of v73), mostly on rows without a type, with region bounds chosen on
  the pooled held table; it lost ΔMSE 419.1 against v73 (`docs/PUBLIC-SUBMISSIONS.md` rows 73–75).
- We do not know the cause of the 0.83 yet. A test of transfer-aware stack weights did not help
  (section 6, item 15).

The board was also used to check the step from fold models to a fit on all of 2025. The no-clock
learner held out of January and July 2025 scored 2.55 s worse on the board than the same learner
fitted on all of 2025 (v38 245.4387 against v39 242.8854; `docs/PUBLIC-SUBMISSIONS.md` rows 38
and 39). Same-month labels of 2025 carry strongly to 2026; this is why most parts of the entry
use fits on all of 2025 for 2026 (section 7).

**No value of the model comes from a board score.** Six early uploads (v26–v29, v31, v33) were score
probes; they are history only (`docs/PUBLIC-SUBMISSIONS.md`). Since then no released weight,
residual, mask or rate comes from a board reading (`docs/PUBLIC-SUBMISSIONS.md`, introduction;
`docs/input-rule.md`). For v68 to v75 the board reads served only for forecasts and for the
choice between whole entries.

## 4. The model

### 4.1 Overview

The entry is one deterministic pipeline over saved model outputs. The file e7b04ffd of section 4.9
was built by `experiments/exp-169-stack-plan/code/build035.py`; the T11b-L step of section 4.9 is
applied to it, and then the steps of section 4.10 (`release/v73_stage.py`); the general harness
`experiments/exp-169-stack-plan/code/assemble_final_032.py` runs the same steps for the earlier
five-column form:

```text
supplied columns ─┬─ eight level-1 members ──► level-2 stacker R4mks ─┐
ADS-B ground clock│                                                    ├─► route = ½ R4mks + ½ v32
stand geometry    ├─ older entry v32 ──────────────────────────────────┘
OPDI events       │
                  ├─ clock head HB3u ──────────────► half blend on clock rows
                  │
                  ├─ members of other kinds: event-sequence transformer, MLP,
                  │  runway sequence models, row control, dark-day twin
                  │                     └──────────► linear stack per class
                  │
                  └─ rules read from how the label is recorded:
                     LIRF rows without a plan, LIRF placeholder practice,
                     stand and runway rules, then one projection onto hard bounds
                                        └──────────► 60 s floor, file checks
                                                     └──► T11b-L: ¼ step toward the dark-day
                                                          twin on typed rows without a plan
                                                          └──► c5: five steps outside the stack
                                                               └──► L2C on clock rows
                                                                    └──► ORD-S on ordinary rows
```

Each part except the older entry v32 (section 9) fits on label rows outside the held month in
the local folds; the later stages also read values that were fitted on the held month (section
3.1). The whole chain is
scored on held January and July 2025 by the same code that builds the 2026 file
(`experiments/exp-169-stack-plan/code/`).

### 4.2 The route: a month-hold stack, half-blended with an older entry

**Older entry v32.** v32 (242.7480) is the chain of our September entries: a blend and residual
stack of gradient-boosted members with surface heads and a queue correction (v22), an update from
the implied off-block clock (v23), a learner for no-clock rows with 21 runway-state window proxies
(v24), and the same proxies in a learner for clock rows (v32). ERA5 weather enters through this
chain; the level-2 inputs of the route and the clock head read no weather (`docs/WRITEUP.md`
section 2; `docs/LINEAGE-ZABD.md#members`).

**Level 1.** Eight members, each fitted under the fold rules
(`experiments/exp-148-short-route/SPEC.md`; `docs/LINEAGE-ZABD.md#members`):

| Member | Model |
|---|---|
| T-tailv4 | regime router: NM rows on an `AOBT_3` offset, rows without NM on a taxi-time head, noac rows on a placeholder mixture (exp-002) |
| C2-cat | CatBoost, depth 8 |
| L2-hub8725 | LightGBM with Huber loss |
| winter member | gradient-boosted member with airport day-state and de-icing features (exp-007) |
| F1 | LightGBM with month-out target encodings (exp-018) |
| N1-nmtail | model of the NM tail by the offset fraction between `AOBT_3` and `LOBT` (exp-010) |
| L | the no-clock learner with runway-state proxies (exp-134) |
| Lp | the clock-row learner with the same proxies (exp-137) |

**Level 2.** LightGBM and CatBoost on the target y − a, where a is the mean of the first five
members. The 163 features are the member deviations from a, context, the clock-head observation
columns, label-free queue columns, the runway-state proxies and the row clocks
(code `experiments/exp-148-short-route/code/`). The level 2 learns from member predictions made out
of month for nine 2025 months, and it never reads a label of the month it scores (route R4). A
first version, selected on held weeks inside the scored month, learned patterns of the held month
and lost on the board (v34; section 6, item 3).

**Two repairs to the level 2.**

- *R4mk:* rows that the route never serves (noac rows, and ordinary rows without `AOBT_3`) get zero
  weight in the level-2 loss.
- *R4mks:* the early stop counts only the served rows. Before the fix, the stop counted rows the
  route passes through unchanged; with the fix LightGBM stops at 4,002 and 3,740 rounds instead of
  915 and 522. The fix changes both the training and the validation rows of the stop, and we did not
  separate the two effects. Gain +425.4 [+358.6, +495.3] at the route against R4mk, and
  +89.76 [+43.22, +143.00] in the full stack. R4mks was one of several route variants read on the
  same held months; the review put the gain from that choice at about 10 or less in the full
  stack. On the board this step gave ΔMSE −100.1, ratio 1.11 (v55 against v53, which had the same
  stack without the fix; `docs/PUBLIC-SUBMISSIONS.md` rows 53 and 55).

**Half blend (B50).** The route is 0.5 × R4mks + 0.5 × v32, with the weight fixed before any score.
On rows that R4 does not serve, the route is v32. When it was first used, the half blend gained
1.73 s on the board over the month-hold stack alone (v36 against v35; `docs/PUBLIC-SUBMISSIONS.md`
rows 35 and 36). The final file has no v32 half: its route is R4mks alone (section 4.11).

**For 2026** the level 2 is refitted on 1,540,605 rows and scored on the 2026 predictions of the
level-1 members fitted on all of 2025. This final route changes only 2026 rows, so it has no held
read. On the board it gained ΔMSE 214.7 (v59 against v58; `docs/PUBLIC-SUBMISSIONS.md` rows 58
and 59).

### 4.3 The clock head

On clock rows we know when the aircraft left the stand from its own ADS-B ground positions. The
clock head predicts the remaining error.

- **Target:** z = y − the observed taxi time (take-off minus the off-block clock seen in the ground
  track) (`docs/LINEAGE-ZABD.md#members`; code `experiments/exp-159-clock-head/code/`).
- **Model:** LightGBM, squared loss; a bag of ten fits with two tree shapes (five seeds each), and a
  code for the 31 most frequent aircraft types (head HB3u).
- **Inputs:** movement inputs only: the surface observation columns of the ground track, the row
  clocks, the implied off-block clock rebuilt from the track and the stand geometry, OPDI
  parking-exit and taxiway-entry leads, and ten pushback-phase inputs. Six
  inputs that describe ADS-B reception were removed, because reception shifts from 2025 to 2026.
  Weather was tested and lost: −138 [−213, −60].
- **Training:** one fit on 934,128 rows of the ten months without January and July 2025; the same
  fit predicts the held months and 2026.
- **Use:** on clock rows, value = max(60, 0.5 × route + 0.5 × max(60, head)). This covers 173,539
  ranking rows plus about 17,000 January clock rows at LEMD and LFPG that have no 2025 support,
  where the same formula is applied without a local read.
- **Gain:** the first rebuilt head at weight 0.3 gave +807.4 locally and ΔMSE −1,136.7 on the board
  against the fold anchor (v47; `docs/PUBLIC-SUBMISSIONS.md` row 47). The swap to HB3u added
  +70.34 [−23.36, +175.06]. Both reads include choices on the same held rows: the weight 0.3 was
  one of three weights read there, and the settings and inputs of HB3u came from about twenty
  settings, six mixes and three input arms read there.

**The clock coverage changes between years.** The share of departures with an ADS-B clock moved from
86.3 % to 0.08 % at LIRF in July, from 6.5 % to 96.4 % at EGLL in July, and from 1 row to 15,877 rows
at LEMD in January. A model fitted on 2025 coverage therefore meets a
different mix of clock and no-clock rows in 2026. Section 4.4 (dark-day twin) and the class weights
of G (section 3.2) deal with this.

### 4.4 Members of different kinds

Members that use the same inputs as the route add little. For a blend of the route and a member
whose mean squared error is r times the route's (r ≥ 1), the best member weight is zero once their
error correlation reaches sqrt(1/r); most of our earlier member vectors sat on that limit (section
6, item 11). The members below see the data in a different form.

**Event-sequence transformer** (role `g`; `docs/LINEAGE-ZABD.md#members`). One token per arrival or departure on the same runway and UTC day: up to 8
movements before and 8 after the departure, within 30 minutes. Inputs are differences of the
schedule, plan and NM times, categories, time cycles, gaps and counts; no ADS-B field and no tree
prediction. Transformer of width 64, two layers, four heads. The target is a correction to the
anchor max(60, take-off − `AOBT_3`), with Huber loss. The candidate uses a bag of 12 fits (six of
width 64, six of width 128 with three layers). Gain of the transformer together with sequence
pack 021 at stack5: +175.03; the 12-fit bag (AB): +48.37 [+7.50, +90.90]. We chose the members
and their gates on the same held months. AB went in after held reads on three bases; its seeds and
its equal mean were fixed before its read.

**MLP** (role `mlp`; `docs/LINEAGE-ZABD.md#members`; code `experiments/exp-164-double-down/code/`).
Embedding MLP 512-512-256 with
LayerNorm, SiLU and dropout 0.10, squared loss, 210 numeric and 18 categorical inputs (the clock-row
learner's columns plus the movement-only clock, OPDI and pushback inputs). The target is y minus a
row time-difference anchor, in units of 1,000 s. Ordinary rows use two seeds; clock rows use a
bag of 12 fits of two recipes (80 epochs; 40 epochs without dropout). Gains: +411.01
[+269.24, +553.37] for the MLP with the six-seed sequence bag at stack4; +56.09 for the clock-row MLP
at stack7; +16.71 for the 12-fit clock bag at stack8. The MLP form, the six-seed bag, the clock-row
recipe and the clock-only use were chosen after reads on the same held months; without one sequence
seed (17006) the stack4 step is +244.8.

**Runway sequence models** (`docs/LINEAGE-ZABD.md#members`; code `experiments/exp-170-diverse/code/`).
A bidirectional GRU runs
over the departures of one airport, day and runway in take-off order: a core of 4 rows with 4
context rows on each side. Inputs are the NM time differences, schedule and movement times, the
movement-only clock differences, runway direction, time of day and the spacing on the runway;
categories are airport, runway, stand, aircraft type, operator and month. No neighbour label enters.
The loss is Huber, with the NM taxi time (take-off − `AOBT_3`) as an additive anchor. The held model
trains on the ten other months and predicts January and July; the 2026 model trains on all
2,085,047 labelled departures of 2025.

The stack uses three sequence columns:

| Column | Recipe | Use |
|---|---|---|
| primary (pack 039) | 30 fits: 12 of width 256 with 24 rounds, 12 of width 128 with 48 rounds, 6 of width 256 with 48 rounds; three layers | all served rows |
| small (pack 021) | 6 seeds, width 64, two layers | all served rows |
| fallback pair | 1 seed of the sequence model and 1 of the row control | the January LEMD and LFPG clock rows without 2025 support |

**Row control.** The same network and inputs without the neighbour context. It enters the stack on
ordinary rows with a negative weight, so the stack keeps what the neighbours add and removes what
both networks share. The control adds +382.09 over the sequence model alone; we chose this pooled
form on the same held months.

Gains of the sequence models: the first runway pair +1,230.17 over the stack of that time (route,
head, no-plan form, practice rule and joined-row bracket); the 12-seed pack 030 +477.42 (stack7);
pack 037 +168.95 (stack8); pack 039 in place of 037 +29.33 [−13.43, +75.02] (stack12). These reads
include choices on the same held months. The first pair is one seed, a high draw, with a gate and
a form chosen there, so its read is optimistic. Pack 030 was one of four bags read there, and the
sources and recipes of packs 037 and 039 were chosen there.

**A time column that did not transfer (Z).** One input of the sequence models was a time-of-day
value computed from the full microsecond timestamp modulo 86,400. Every 2026 value lay above the
2025 range, so the column acted as a ramp in time. We set the two affected columns to zero and
refitted the five sequence columns. Gain over the previous candidate: +251.70 [+125.66, +378.42],
+182.37 without the five best days. Z was the better of two repair forms in a pilot read on the same
held months (+224.96; the true day seconds +161.49), by a rule written before that read. On the
board Z lost: ΔMSE +103.6 against v59 (v60; `docs/PUBLIC-SUBMISSIONS.md` rows 59 and 60). A check
found no defect in the build. Most of the held gain (+189 of +252) is a level repair in held
January: in that fold, January lies below the training date range of the old column, and the old
members were too high there. 2026 lies above the range, where Z moves the level up. We keep Z in the
entry.

**Dark-day twin (dark18).** 18 seeds of one recipe of the primary pack (width 128, three layers,
48 rounds), trained with the 38 clock-derived inputs and their flags set to zero on whole training
airport-days. The days are drawn at each airport's 2026 coverage loss (for example LIRF July 0.999,
EDDF July 0.441; the January rate for October–March, the July rate for April–September), so the
member learns rows that lose their clock in 2026 (`docs/LINEAGE-ZABD.md#members`). It enters as a
sixth column. Gain: +33.35 [−5.71, +70.61], +5.28 without the five best days. The bag grew from 6
to 18 seeds after the read of the first six on the same held months.

### 4.5 The stack

After the route and the head, the members are combined by one linear stack per class
(code `experiments/exp-169-stack-plan/code/build035.py`):

    value = max(60, p + Σ_k w_k (m_k − p))

where p is the value after the route and head, and m_k are the member predictions.

- **Scope.** Rows with a known aircraft type, an `AOBT_3`, a value of the frozen fold route below
  2,400 s, and a supported cell. Other rows keep p.
- **Columns.** In order: event-sequence transformer, MLP, small sequence, primary sequence,
  dark-day twin, row control. Ordinary rows use all six; clock rows use the first five.
- **Fit.** Weighted least squares on y − p with the frozen 2026 cell weights, no intercept, no
  penalty, no clipping; weights may be negative. On rows that the LIRF practice rule (section 4.7)
  moves later, the member columns are scaled by one minus the practice weight.
- **No label of the scored month.** Held January uses the weights fitted on held July, and the
  reverse. The 2026 weights pool both held months.

The 2026 weights of the candidate (the replay of section 7 refits them and writes them to its
record):

| Class | transformer | MLP | small seq. | primary seq. | dark twin | row control |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 0.145413 | 0.066182 | 0.017129 | 0.288076 | 0.225951 | −0.238054 |
| clock | −0.040154 | 0.26964 | −0.308048 | 0.807753 | −0.116542 | – |

A fallback pair (one sequence seed and one row-control seed) is applied first; the primary stack
overwrites it on its own scope, so only the January extension rows keep the fallback values. Its
clock weight comes from a fixed grid in steps of 0.02.

### 4.6 LIRF departures without a plan

The largest single rule comes from how the file is joined. The flight-plan columns are joined to a
departure only when the airport block time is within 3,606 s of `LOBT`: there are no exceptions in
2,062,577 joined 2025 departures. A departure without a plan therefore has a block time more than
an hour away from its plan. At LIRF such a block time is often a schedule placeholder (block =
scheduled time) or a time on another date.

The LIRF rows without a plan and without an aircraft type get a separate form (no-plan v3-cal;
`docs/LINEAGE-ZABD.md#rule-stages`; 383 ranking rows):

| Rule | What it reads | Gain on the 2025 folds |
|---|---|---:|
| C30 join rule | When the same flight lands at one of the ten airports, its arrival row holds `LOBT` and `AOBT_3`. The row is a placeholder if the schedule is more than 3,606 s from `LOBT` and `AOBT_3` is inside the window (value take-off − schedule); it is real if `AOBT_3` is outside the window (value take-off − `AOBT_3`); an open state takes 0.89 × real + 0.11 × placeholder. | +722 [+197, +1,470] |
| C20 rollover | When take-off is on a later UTC date than the schedule, the label is either take-off − schedule (placeholder) or 86,400 s + the taxi time (date rollover); the value is projected onto that range. | +117 |
| C32 local offset | Some LIRF block times are early by the local UTC offset (1 h in winter, 2 h in summer). When 0 < take-off − schedule < 3,900 s, value = q × (offset + 900) + (1 − q) × (take-off − schedule), with q the airline's 2025 share of shifted rows. | +246 [+51, +472] |
| C39(a), C41, C45 | Placeholder points for late rows by the airline's arrival default share; date of the block time nearest the schedule; very late wrapped times | +143.4 (C39(a)) |
| class model with C43, C44 | A two-point class model (placeholder or real) for the other same-date rows, with turnaround and local-delay columns, and one calibrated cell | +104.5 [+4.6, +219.4] (C44 with C45); +386.3 [+20.6, +886.0] (C43 cell; the served form +374.1) |

We chose the limits of this form after a look at January and July 2025: the weight 0.25, the C30
state thresholds, the C32 band of 3,900 s, the C39(a) limits, the C44 inputs, the calibration and
the C45 limit. The limits of the C43 cell were chosen on all of 2025. The fold reads in the table
are therefore not out of sample. The C30 split 0.89/0.11 comes from all of 2025 and acts on 2 fold
rows; the C20 tables are fitted on the ten other months.
The fold read of the whole form on the LIRF rows without a plan was +2,144.3 [+1,249, +3,091]. The
ladder used the local value +1,440.4, and the board gave ΔMSE −1,187.3 (v48, 2.51 s;
`docs/PUBLIC-SUBMISSIONS.md` rows 47 and 48), a ratio of 0.82 to that value.

### 4.7 The LIRF placeholder practice

For some airlines at LIRF almost every late block time is the scheduled time (10 airlines:
1,689 placeholders in 1,699 rows of 2025). On joined LIRF rows after
the schedule, the practice rule moves the value to the placeholder point take-off − schedule when
the airline's share of placeholder arrivals in the ranking file is at least 0.95 over at least 10
arrivals and the airline is confirmed in 2025; airlines new in 2026 move 0.766667 of the way, and two
excluded prefixes stay at the base (`docs/LINEAGE-ZABD.md#rule-stages`, practice). Gain +43.19
[+11.57, +91.26]. The same practice
holds for single routes of airlines that otherwise record real blocks (rule C73, airline ×
destination, for example 101 of 101 late rows of 2025 on one route): +233.43 [+72.17, +476.73].
We wrote the limits of the practice rule after tables of all twelve 2025 months and designed C73 on
all of 2025; the held months are in both, so both reads include that choice.

### 4.8 Stand and runway rules, and the hard bounds

These rules read supplied columns: the arrival in-block times of other aircraft, the plan times,
the file's join window and the movement times. C95 also uses 2025 residual statistics, and C98 the
stack values of nearby clock rows.

| Rule | What it does | Ranking rows | Local G [95 %] |
|---|---|---:|---:|
| C55 stand re-use floor | At EDDF, EGLL, LFPG and LTFM: take the last arrival at the same stand before take-off; if it has another known aircraft type, g = take-off − its in-block time (0 < g ≤ 7,200 s, inside a plan-time window; on rows without a plan g > 300 s). The departing aircraft left before the next one came in, so label ≥ g. | 27 moved | +843.78 [+51.99, +2,083.36]; +242.26 without one EGLL row |
| C71 hard stand floor | The same bound outside C55's window, when no other departure leaves the stand in between and either g ≤ NM taxi time + 300 s, or g ≥ take-off − `EOBT_1` + 900 s and g ≤ 3,600 s | 41 | +28.49 [+6.13, +59.95] |
| C95 soft stand floor | On plan rows to non-local destinations at the same four airports, when the stand bound lies 0–450 s below the value, add the mean of the 2025 residuals that respect the bound minus the mean of all residuals (by airport and class) | 3,108 | +60.21 [+44.72, +78.65] |
| C98 runway view | Ordinary plan rows at EDDF, EGLL, EHAM, LEBL and LSZH: add a per-airport slope × the mean (stack − frozen fold route) of at least two clock rows on the same runway within ±900 s | 23,158 | +45.51 [+24.03, +66.98] |
| C66 copied arrival rows | LIRF departure rows that copy an arrival record: take-off − schedule if above 30 s, else 600 s | 17 | +10.15 [+1.70, +22.96] |
| C58 double stand exit | LFPG: the first record of a flight that left its stand twice moves half way to a least-squares point on plan delays | 21 | +136.16 [+42.92, +255.33]; wo5 +31.30 |

We chose these rules on the held months that score them: the airports, limits and windows of C55,
the forms of C71, C95 and C98, and the half form of C58 (one of three forms read there). The hold
audits of C40, C63, C64 and C66 used all of 2025. The reads in this section are therefore not out
of sample.

C66 and C58 are read on stack12. C95 and C98 statistics are refitted from the held stack on
every build: each held month uses the other month's residuals, and 2026 uses both. C71, C95 and C98
together gained +134.46
[+96.33, +181.46] locally and ΔMSE −84.8 on the board (v56, ratio 0.63; `docs/PUBLIC-SUBMISSIONS.md`
rows 55 and 56). Rule definitions: `docs/LINEAGE-ZABD.md#rule-stages`.

**A rule we removed (C65).** The tail as first built had one more rule: at the five airports
without a minute grid, a backward time step between neighbouring rows of the supplied file marked a
±6 s recording offset. It read the row order of the file, not a recorded time of the flight, and
its read on stack12 was slightly negative: −1.03 [−2.51, +0.37]. We took it out of the entry.
Removing it from the candidate changed 35,054 ranking rows by at most 12 s, with G +0.99
[−0.42, +2.48] (wo5 −0.04). A check of the assembly found no other step that reads the order of
the supplied file.

**One projection onto hard bounds** comes last
(`docs/LINEAGE-ZABD.md#rule-stages`, bounds). The lower bound is the largest of 60 s, the joined-row
bracket, C55, C71, C64 and C68; the upper bound is the smallest of the joined-row bracket, C40 and
C63.

- *Joined-row bracket (C33):* on rows with a plan, label ∈ [take-off − `LOBT` − 3,606,
  take-off − `LOBT` + 3,606], from the join window. +107.80 on the first stack.
- *C40 and C64:* an arrival row whose NM flight ID has no departure row belongs to a departure that
  lost its plan; its `LOBT` then bounds the label by the join window.
- *C63:* LEBL and LEMD have no departure above 1 h in the files (0 of 391,947 rows of 2025 above
  3,612 s), so the value is capped at 3,600 s.
- *C68:* one LIRF row whose recorded take-off date we read as a day late (the flight landed at its
  destination the day before); on that reading the join window puts its label at 86,400 s or more.

The bounds alone gained +101.38 [+19.65, +201.51]. The tail as first built (C65, C66, C58 and the
projection) gained +246.68 [+111.70, +397.83], +94.00 without the five best days, on stack12; like
the reads above, this read is not out of sample. This
tail, with C65, went up as v58: 226.3211 s, ΔMSE −273.4 against v57, a ratio of 1.11
(`docs/PUBLIC-SUBMISSIONS.md` rows 57 and 58).

### 4.9 The entry v64: e7b04ffd plus T11b-L

The file `e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636` is stack12 with the
final 2026 route, the bound tail of section 4.8 (without C65), the time-column fix Z, the 12-fit
transformer bag AB and the dark-day twin. It equals the earlier candidate
`0eec38e86fadd6a658dabd1e877bd7603301429b6523d375421466efd8cb7779` (the same chain with C65) on all
rows except the 35,054 rows that C65 moved. Its checks: 344,841 unique IDs in the template order,
float64 columns, finite values, minimum 60 s, and all bounds hold. With the 0.83 ratio its frozen
forecast was 225.9056 s; as v62 it scored 225.9106 s, 0.064 s behind v59, which has the same route
and tail (with C65) but not Z, AB and the dark-day twin (`docs/PUBLIC-SUBMISSIONS.md` rows 59 and
62). We keep the candidate with Z (section 4.4).

**T11b-L.** The entry v64,
`9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b`, is e7b04ffd with one fixed step
on the 4,902 typed rows of 2026 without an NM join. These rows have no `AOBT_3`, so they are
outside the scope of the linear stack (section 4.5): no member of other kinds enters their value.
On each such row

    new = max(60, clip(base + 0.25 (p' − base), lower, upper)),   p' = max(60, clip(m, lower, upper))

where base is the e7b04ffd value, m the dark-day twin (section 4.4; for 2026 its fit on all twelve
months of 2025, for the held months its fit without January and July), and lower and upper the
bounds of section 4.8. The quarter is fixed: there is no fit and no weight choice.

- **Long-taxi airport-days keep the base.** A long-taxi airport-day is an airport and UTC day with
  8 or more departures whose NM taxi time (take-off minus `AOBT_3`) is 3,600 s or more, and these
  are at least 2 % of the departures with an `AOBT_3`. 2026 has 15 such days (EGLL 1; EHAM 8,
  2–7, 9 and 29 January; LFPG 3; LTFM 3), with 802 of the 4,902 rows. On the 3 such days of held
  January, the step without this rule moves 76 rows and reads −14.85 [−55.10, +10.05]. In 2026,
  these days held 57 % of the squared change of the step without this rule (90.85 against 38.66 s²
  per ranking row).
- **2026 file.** 4,078 rows change; the squared change is 38.66 s² per ranking row, and 24 rows move
  by more than 250 s. Every other row equals e7b04ffd. The checks of e7b04ffd hold (template order,
  unique IDs, float64, finite, minimum 60 s, bounds).
- **Held read** over e7b04ffd, on the held rows of the same 2026 rule: +97.31 [+18.38, +174.96],
  P(G ≤ 0) 0.013, +49.89 without the five best days (January +41.43, July +55.88).
- **Choice after one look.** The row set and the member came from one look at a wider test. Two
  checks written before their reads moved the same rows a quarter toward two members that the look
  did not choose: +117.3 [+53.7, +186.5] and +64.9 [+19.5, +106.8]. The gain comes from rows that
  the stack does not reach, not from the choice of member. The long-taxi rule came after the read
  of the step without it; the form with the rule was written before its own read.
- **Board.** The frozen forecast was 225.7286 s at 0.83. As v64 it scored 225.7082 s, 0.2024 s
  better than v62 (ΔMSE 91.4) and 0.1381 s better than v59 (`docs/PUBLIC-SUBMISSIONS.md` rows 59,
  62 and 64).
- **Code:** `release/t11b_stage.py`, the form of the step that ships in the public export (section 7).

### 4.10 The final entry: v64 plus c5, L2C and ORD-S

After 6 October, two agents built these steps from the v64 release export, the official files and
the files of the agent before them (section 8). The steps read the v64 file, the saved member
values and the challenge files. Reads are G against the named parent (sections 3.1 and 3.2). A step
that fits a coefficient is read cross-month: fitted on one held month and scored on the other; for
2026 it is fitted on both held months. ROLL fits on ten months for its held read, twelve for 2026.

**c5 (v68).** Five steps on rows outside the linear stack. Each step changes the route value, the
head value or a value after the rules; the rule chain of section 4.8 then runs unchanged.

| Step | Rows of 2026 | What it does | Held read [95 %], wo5 |
|---|---|---|---|
| EXT | 17,017 January clock rows at LEMD and LFPG, without 2025 rows of their kind | a pooled clock stack of five members on the head value, in place of the fallback pair (section 4.5) | no held rows; on held clock cells the stack form gains +571 [+490, +661] over the pair form |
| LONG | 4,222 long-route rows (route value 2,400 s or more, with `AOBT_3`, outside the stack scope) | half of a step toward a member stack on the head value; weights from the other held month | +40.01 [+3.01, +83.93], wo5 +5.31 |
| LIRF | 5,700 LIRF January clock rows, served by the route alone in v64 | half of a step toward the pooled ordinary stack of six members on the route value | no held rows; on held LIRF July clock rows, +1,972 against the route alone |
| NOAC | 232 LIRF rows without an aircraft type and with a mixed value | a quarter of a shrink toward 900 s; the slope from the other held month | +108.69 [+15.90, +226.81], wo5 +3.30 |
| ROLL | 3 LIRF rows without an aircraft type that take off 56,520 to 61,322 s after the schedule, on the next local date | the expected value of the two label values of rule C20 (section 4.6): the placeholder value, and 86,400 s plus 1,172 s. The share of a rollover comes from a logistic in take-off − schedule, fitted on the 2025 rows of this kind (for the held read, on the ten other months). v64 gave these rows shares of 0.10 to 0.21; the logistic gives 0.33, 0.54 and 0.58 | +1,841.41 [−1,109.73, +5,412.70], wo5 −651.02, on 6 held rows (4 rollovers) |
| c5, all five | 52,468 (the refits of the rules move more rows) | | +1,990.11 [−945.00, +5,561.74], wo5 −565.11; without ROLL +148.70 [+44.58, +279.98], wo5 +35.65 |

- **Few rows carry the read.** With the rows without a type kept at v64, c5 reads +40.0
  [+3.0, +83.9]. EXT and LIRF change only 2026 cells without held rows, so their reads are proxies
  on other cells.
- **Why ROLL.** In 2025, the LIRF rows without a type that take off 50,000 to 86,400 s after the
  schedule, on the next local date, were rollovers in 10 of 17 cases (4 of 6 from 55,000 to
  62,000 s). At v64's share of about 0.2, 4 of 6 or more has a probability of 0.017. In a
  leave-one-month-out test, the logistic beat a fixed share of 0.15 by 2,103 s² per ranking row.
  The 17 rows include the 6 held rows that score ROLL, and the 50,000 s band and the choice to use
  the step were made with these rows in view, so its held read is not independent.
- **File.** The uploaded file `63850254…` was built on macOS. The same steps on x86-64 Linux give
  `58a2ac91…`; the values differ by at most 1.5e-11 s per row, and the held G is the same. L2C and
  ORD-S build on the x86 file.
- **Board.** The forecast before the upload was 223.7 s at the expected value of ROLL, with a
  median of 225.0 s and a 90 % range of 218.7 to 228.3 s. As v68 it scored 224.243 s, 1.4652 s
  better than v64 (ΔMSE 659.3; `docs/PUBLIC-SUBMISSIONS.md` rows 64 and 68).

**L2C (v69).** On the clock rows of 2026 with an NM join (171,914 rows change):

    value = max(60, clip(c5 + 0.5 f, lower, upper))

where f is a LightGBM model of clip(y − c5, ±2,000 s), and lower and upper are the bounds of
section 4.8. The January extension rows and the LIRF January clock rows keep c5, because they have
no held rows of their kind. The inputs of f are the deviations of the saved members from c5 (the
stack members, the route, the clock head and the other saved members), the row features, the
airport and the class. Huber loss with a threshold of 300 s, 31 leaves, 800 rounds, the mean of
3 seeds.

- **Held read** over c5: +112.8 [+47.4, +178.9], wo5 +48.2 (January +32, July +81; EGLL +79).
  EGLL July holds 58 % of the gain: its 1,479 held rows stand for a 2026 cell with 96 % clock
  coverage (6.5 % in July 2025).
- **Choices on the held read.** We chose the factor 0.5 on a flat held curve (Mac build: 0.4
  +108, 0.5 +115.9, 0.6 +117, 0.7 +110, 1.0 +45). Other tree settings gave +93 to +122 with a
  lower wo5. The clock-row scope came after a read on all typed joined rows (clock +112, ordinary
  −87), and other input sets read no better. These choices used the held read that scores the
  step.
- **Board.** The forecast was 224.03 s (0.83 × 112.8). As v69 it scored 223.8572 s, 0.3858 s
  better than v68 (ΔMSE 172.9, 1.53 of the local G; `docs/PUBLIC-SUBMISSIONS.md` row 69).

**ORD-S (v70 to v73).** On the ordinary rows inside the stack scope (section 4.5):

    value = c5 + β_a (s − c5)

where s is a sequence column and β_a is one coefficient per airport, fitted by weighted least
squares on the held ordinary rows with |y − c5| < 3,000 s and floored at 0. Clock rows (with
L2C), rows without a type, and the rows of long-taxi airport-days (section 4.9) keep their value.
From v71 on, the LIRF July rows take half of the LIRF coefficient, which is fitted on LIRF January.
In 2026, LIRF July has no clock rows, as LIRF January 2025 has none. The held LIRF July rows alone
give a coefficient of 0.284 [0.038, 0.494]; LIRF January gives 0.365 [0.264, 0.494].

The sequence members are airport-context transformers. The tokens of a departure are the movements
at its airport before and after it, inside a time window: arrivals with their taxi-in times, and
departures with their NM times. The members never read a departure block time: it is blank in 2026,
and the preparation code blanks it in 2025. Each member is fitted in two modes: without January
and July 2025 for the held values, and on all of 2025 for 2026.

| Upload | Sequence column s | ORD-S held read over v69 [95 %] | Step |
|---|---|---|---:|
| v70 | mean of s1 (12 movements on each side within 40 min, 4 seeds) and s2 (24 within 60 min, 1 seed); LIRF July keeps its value | +78.5 [+22.1, +138.4] | +78.5 |
| v71 | s1 and s2, with s2 at 2 seeds; LIRF July at half of the LIRF coefficient | +128.6 [+32.5, +227.6] | +50.1 |
| v72 | mean of s2 and s3 (36 within 90 min, 1 seed) | +199.8 [+81.6, +314.6] | +71.2 |
| v73 | s7 alone (64 within 180 min; model width 192, 4 layers, 16 epochs, half precision; 3 seeds per mode) | +294.9 [+180.0, +414.4] | +95.1 |

- **v73.** 121,592 rows of 2026 differ from v69. On held rows EDDF reads −22.2, and its 2026
  coefficient is 0; without EDDF the read is +317.1. Over c5, L2C and ORD-S together read +407.7.
- **Selection.** We chose the form of ORD-S on the held rows that score it: the rows it skips, the
  floor at 0, one coefficient per airport, LIRF July at half of the coefficient (after its held read
  of +40.2 [−34.3, +113.3]), the members, the member window and the seed count. The reviews sized
  parts of it: the choices of v70 can explain up to about 52 of its +78.5; dropping s1 at v72 added
  +20.9 of its +71.2; s7 alone added +12.05 over a fixed rule of s3, s4 and s7 on the one-seed form
  of v73. The intervals above do not include this selection. The board is a check on data that did
  not choose the form: the four uploads together kept 0.83 of the held gain (+246.0 against +294.9).
- **Board.** v70 223.743 s, v71 223.6029 s, v72 223.5452 s and v73 223.3071 s
  (`docs/PUBLIC-SUBMISSIONS.md` rows 70–73); per step, 0.65, 1.25, 0.36 and 1.12 of the local
  step (section 3.3). The forecast of v73 before its upload was 223.396 s.
- **Code:** `iso/` (c5), `w2/` (L2C and ORD-S) and `release/v73_stage.py`, which runs both on the
  v64 chain in the public export (section 7). The s7 values are in `release/saved-vectors/s7c/`.

**Two uploads after v73.** Neither scored better than v73.

- **v75** (`caf22c21…`) gave ORD-S two columns: value = c5 + b1 (s0 − c5) + b2 (s7 − s0), where s0 is
  the same transformer with the departure token alone (5 seeds per mode), and each coefficient is
  floored at 0. We chose this form among four read on the same held data. Its held read over v73 was
  +86.4 [+25.5, +154.9], wo5 +28.3; the part for 2026 is about +64. It scored 223.5121 s, ΔMSE 91.6
  worse than v73. A review, a bug check and a rebuild found no error. Our belief about the cause: the
  part s0 − c5 rests on day-level traffic inputs that move by 1 to 2.8 held standard deviations from
  2025 to 2026. On 2026 January rows, the full-year fits of s0 and s7 differ from their held fits by
  46 to 139 s rms at five airports, and the coefficients come from the held fits. The held split
  cannot show this, because its members train on the same year. The test that would check it, member
  fits on February to June read on July, stopped with the work.
- **v74** (`feb3a157…`) came from a third agent, on the one-seed form of v73: a bag of three sequence
  fits in the ORD-S column, a calibration and two rules for LIRF rows without a type, a scaled stage
  C95, and a shift of the rows near a hard bound. Its held read over its parent was +1,044.7 [+496.2,
  +1,743.9], +878.9 of it on rows without a type; the bounds of the calibration regions were chosen
  after a look at the pooled held table. It scored 224.2436 s, 0.9365 s worse than v73.

### 4.11 The final file

After the ranking phase, the organizers released two more files, `final_ranking.parquet` and
`final_submitting.parquet`, with 670,790 departures of January, February, June and July 2026. The final file
`youthful-giraffe_final.parquet` holds one prediction for each departure, in the order of
`final_submitting.parquet`, with the columns `MVT_ID_mvt` and `TAXITIME_SEC_mvt` (SHA-256
`eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02`). The organizers rank it on January and July,
on February and June, and on all four months. The January and July departures are those of the ranking file,
with take-off times drawn again.

**Design.** The final file applies the model of v73 (sections 4.1 to 4.10) to the final files with four changes
(more fits of the s7 member, the s7 member as an input of L2C, the 12-fit MLP values on every member-scope row
of the assembly, and ORD-S on the ordinary rows of long-taxi airport-days, below), and then one more step on
14,587 EDDF rows (the DH step, below). The model of v73 gets no new member or rule. Each part
that reads 2026 rows was built again on the four final months: the route members and level 2, the clock head, the
members, the LIRF rule frames, the assembly, c5, L2C and ORD-S. The held January and July 2025 rows keep their
labels and their saved member values as released. Their exposure weights follow the four-month mix of the final
file: a held January row stands for its airport and class in January and February 2026, and a held July row for
June and July. With these weights the assembly fits its blend weights again on the same held rows, so the January
and July values differ from those of v73 also through these weights. February takes the scope of January, and
June the scope of July. The s7 member was fitted again in both modes, and ORD-S reads those fits.
The open sources of section 2.2 were extracted again for the four months by the same code.

| Path | What it adds | File SHA-256 |
|---|---|---|
| R | the route, the clock head and the rules | `5a1278f3…` |
| Z | R plus the members, the zabd assembly and the stage T11b-L (section 4.9) | `e47b8afb…` |
| V | Z plus c5, L2C and ORD-S (section 4.10) | `121e33b9…` |
| V9 | V with 9 fits of the s7 member (V: 3) | `280db205…` |
| V9A | V9 with the s7 input of L2C, 9 LightGBM seeds | `6de02372…` |
| V9A-R9 | V9A with the 12-fit MLP values on every member-scope row of the assembly (R9) | `1c1ed67b…` |
| V9A-R9-LD | V9A-R9 with ORD-S also on the ordinary rows of long-taxi airport-days (LD) | `260b427b…` |
| V15A | V9A with 15 fits of the s7 member | `a9146e9d…` |
| V15A-R9 | V15A with R9 | `18a7bfb7…` |
| V15A-R9-LD | V15A-R9 with LD | `31b09989…` |
| final | V15A-R9-LD plus the DH step | `eea300c0…`, the final file |

R, Z, V, V9, V9A, V9A-R9, V9A-R9-LD, V15A and V15A-R9 were built on the way. V15A-R9, V15A, V9A-R9-LD, V9A-R9,
V9A, V9 and V with the DH step are `5ce37d69…`, `833533ad…`, `07d4a4e6…`, `4ac66707…`, `c4511650…`, `bf47551e…`
and `2fe6c56b…`.

**The route and F without the v32 half.** In v73 the route is the half blend B50 (section 4.2), and the start
value F of the composition is the fold anchor v43 (R4's month-hold level 2, then 0.5 × that + 0.5 × v32). In the
final entries the route, F and the no-plan base are R4mks `pred_Sb` alone on all 670,790 rows
(`refit-src/experiments/exp-189-final-route/code/make_parts.py` lines 145-157; the official-columns GBDT serves
a row without a finite R4mks value, 0 rows). A rebuild of the v32 half needs the v32 chain again, which we did
not reproduce. The held January and July 2025 rows keep the released F (B50 with v32), so the rules fitted on
held F serve on the final F. Held cost, read at the route and at F only (R4mks month fits, 344,419 held rows):
R4mks against B50, G +150.5 [−1,624, +2,634], P(G ≤ 0) 0.48, −1,573.6 without the five top days; RMSE 293.35 s
against 293.03 s. R4mks against the released F: G +705 [−1,085, +3,187]. The full chain has no held read of this
change, since its held rows keep the released route and F.

**A second design change: the base of the LIRF rows without a plan.** The no-plan form (section 4.6) reads a base
value in two rules: r1, value = max(60, a × base + (1 − a) × mix), and C20, which clips the base to the rollover
range. In v73 the base is a saved route value F of an earlier entry, which exists for the January and July rows only.
The final file uses the R4mks route (section 4.2) as the base on all four months, with a = 0.444 in place of
0.75. Held reads on the 397 held LIRF no-plan rows of January and July 2025:

- R4mks base at a = 0.75 against the released base: G +1,158 [−297, +3,544]. **Two held rows carry this gain:**
  two C20 rows near the date rollover (25 January and 30 July 2025, labels about 87,180 s) give −1,100.5 s² per
  held row of the plain held change of −970.6; the other 395 rows give +129.9 (worse). Without the top days the
  read is −383.
- a = 0.444 against a = 0.75 on the R4mks base, fitted on one held month and read on the other: G +194.1
  [−178, +584]; −183.2 without the top days. The weight comes from 229 r1 rows, 207 of them in July.

We kept the change on its average gain (section 3.2); the gain comes from fewer than 50 rows. In 2026 the no-plan
value reads this base on 368 rows (r1 and C20), 230 of them in January and July.

**Two changes after V: nine s7 fits and the s7 input of L2C.** Both came after the board check below.

- V9: the s7 member is the mean of 9 fits in each mode (V: 3). Another agent fitted seeds 4 to 9 with the released
  fit code and flags; only the seed, the data folder and the output path change. Held RMSE of the member: 214.28 s
  (3 fits: 215.83 s). ORD-S reads this member, and its 2026 β rise at every airport except EDDF, which stays 0.
- V9A: L2C also reads the deviation of the s7 member from c5 as an input, with 9 LightGBM seeds (V: 3); the other
  L2C settings are those of section 4.10 (`w2/l2c_candidate_a.py`).

Held reads against V (G with the exposure weights of the final file, interval over day blocks):

| file | 2026 rows moved | held G against V | without the top 5 days |
|---|---|---|---|
| V9 | 236,289 (rms 7.4 s) | +32.39 [+7.08, +58.07] | +14.01 |
| V9A | 569,604 (rms 5.3 s) | +38.09 [+13.01, +65.44] | +19.71 |
| the L2C change alone, on the 3-fit member | 333,314 (rms 2.5 s) | +3.06 [−2.63, +8.76] | +0.03 |

The gain of V9 comes mostly from LTFM (+19.44), LIRF (+7.13), LFPG (+4.03) and EGLL (+2.40); EDDF gives −2.23.
In V9A the L2C change adds +5.70 [−0.15, +11.63] against the L2C of V9. None of the 14,587 rows of the DH step
moves.

Selection facts. The number of s7 fits (9) was fixed before any read on the final files. The L2C input was the best
of three input variants on held January and July 2025 with LightGBM seeds 1 to 3 (s7 +4.30, a feature variant
−1.05, both +2.15). The seed sets 4 to 6 and 7 to 9 then read +8.11 and +11.80, and the 9 seeds are their union. So
the held read of the L2C change comes from the rows that chose its input. On February and June 2025, which did not
choose it, the L2C change read +6.96 and +44.77 with one s7 fit and wide intervals, and February +3.03
[−7.84, +12.69] with a second s7 fit. On February 2025, ORD-S on two s7 fits against one (capped) read +624.59
[+164.22, +1,176.63] (LTFM +599, without the top 5 days +102). Neither change had a board read.

**One change after V9A: R9, the 12-fit MLP values on every member-scope row.** In the assembly, the MLP column of
the primary frame held the 12-fit bag on the member-scope clock rows and the two-seed fit on the other
member-scope rows (section 4.4). R9 puts the 12-fit values on every member-scope row, in 2026 and on the held rows.
The held rows read out-of-fold 12-fit values (ten-month fits without the scored month); R9 adds no fit. Only
ordinary rows of the frame change. Every blend weight refits (the clock weights stay equal), and c5, L2C and ORD-S
run again with the s7 member and the L2C of V9A. In 2026, 292,360 rows move against V9A (rms 2.3 s, mean −0.09 s):
282,499 ordinary rows, and 9,861 LIRF clock rows of January and February that the c5 LIRF step serves with the
pooled ordinary weights at half step (the refit moves them by rms 0.4 s). 14,160 of the 14,587 rows of the DH
step move (rms 3.1 s in V9A-R9).

Held reads against V9A, with the exposure weights of the final file (January read with the weights and β fitted
on July, July with those fitted on January):

| stage of the R9 chain | held G against V9A | without the top 5 days | January | July |
|---|---|---|---|---|
| zabd, after T11b-L | +31.57 [+11.15, +53.25] | +14.27 | +9.89 | +21.68 |
| V9A-R9, after ORD-S | +23.11 [+4.77, +41.83] | +7.29 | +6.69 | +16.42 |

ORD-S takes 8.50 of the step: the s7 member corrects some of the same rows. EDDF (+10.46) and LIRF (+7.93) give
most of the step in V9A-R9; every class but the ordinary rows gives 0.00.

Selection facts. R9 was found on the held January and July 2025 rows (a read of 5 October against zabd of the
ranking phase, +26.2 [+9.4, +43.4]), and the reads above measure it again on the same months. Five July
airport-days give +15.82 of the +23.11 (LIRF 10, 31 and 4 July; EDDF 20 and 27 July); the three LIRF days give
+10.76 from 128 moved rows. On February and June 2025, which did not choose R9, a reduced chain read −6.0
[−88.8, +72.5]. That read is too wide to judge: four of the five blend columns, the s7 member and route level 2
have no held values on those months. A fresh review passed. R9 had no board read.

**One change after V9A-R9: LD, ORD-S on the ordinary rows of long-taxi airport-days.** In v73, the rows of
long-taxi airport-days (section 4.9) keep their value at ORD-S. LD applies ORD-S also to the ordinary
member-scope rows of these days. The fit of the β still leaves out these days, so every β is unchanged (2026: pooled
0.3036, LTFM 0.3716). The ORD-S code is `w2/ords_ld.py`, the `w2/ords.py` of V9A-R9 with the flag `--long-days`.
In 2026, 6,753 rows move against V9A-R9 (rms 51.4 s, mean +1.4 s, largest 833 s): LTFM 5,366 (June and July),
LFPG 1,064, EGLL 199 and EHAM 124. No row of the DH step moves.

Held read against V9A-R9, with the exposure weights of the final file: +3.16 [−0.20, +8.37], P(G ≤ 0) 0.111,
−0.10 without the top 5 days. It comes from 354 rows on 3 January airport-days (EGLL 11 January: 325 rows,
+2.55); July 2025 has no long-taxi day.

Selection facts. Another agent, which worked without our notes, found LD on held 2025 months: with the β of v73
it read January +11.4, March +54.2, November +44.0 and December +33.7 (1 to 3 days each). Our held read uses the
same January rows, so it measures the step again. LTFM holds 79.5 % of the moved rows, and no held month of our
chain has an LTFM long-taxi day. The LTFM long-taxi days of 2026 are mild (long taxis are 2.0 to 2.7 % of the
departures with an `AOBT_3`), so LD applies the ordinary ORD-S to 97 to 99 % of their rows. On the ordinary LTFM
rows of the held months, which have no long-taxi day, ORD-S reads +75.78 [+40.16, +121.49] in January and +90.95
[+49.19, +134.78] in July. On 10 September 2025, the one LTFM long-taxi day of 2025, a read on another base gave a
positive sign (+27.5). No February or June 2025 day is a long-taxi day, so the 3,014 February and June rows of LD
have no held read. A fresh review passed. LD had no board read.

**Six more s7 fits (15 in all) on the same chain.** The chain V9A, V9A-R9 and V9A-R9-LD ran again with the s7
member as the mean of 15 fits in each mode: V15A, V15A-R9 and V15A-R9-LD. Fits 10 to 15 ran the command of fits 1
to 3; only the seed and the output path change. Held RMSE of the member: 213.94 s (9 fits: 214.28 s). R9 and LD
are the steps above. On the 15-fit member, R9 reads +23.16 [+4.94, +41.93] against V15A (+7.10 without the top 5
days), and LD reads +3.08 [−0.19, +7.95] against V15A-R9 on the same 354 rows. In 2026 the final file differs from
the 9-fit file (V9A-R9-LD plus the DH step) on 576,351 rows (rms 2.5 s, mean +0.01 s, largest 115 s). No row of
the DH step moves.

Held reads of the 15 fits, with the exposure weights of the final file:

| read | held G | without the top 5 days | January | July |
|---|---|---|---|---|
| 15 fits against 9, with the L2C of v73 | +14.12 [+2.39, +26.22] | +5.08 | +4.16 | +9.96 |
| V15A against V9A | +16.43 [+4.55, +28.31] | +7.39 | +4.47 | +11.96 |

The step from 9 to 15 fits comes mostly from LTFM (+11.34) and LIRF (+6.70); EDDF gives −4.13. The reads share
their fold rows and weights, so the steps add: V15A-R9-LD against V9A-R9-LD is +16.40.

Selection facts. The number of fits (15) and the seeds 10 to 15 were fixed before the fits started, with an
expected step of +6.5 (+1 to +13). No fit was dropped or repeated. All reads use the held January and July 2025
rows, on which the step from 3 to 9 fits was read before. The read of the step from 9 to 15 fits is above the
expected range. The member RMSE fell by 0.34 s, near the 0.31 s that a fit of the seed count predicts, and the
cause of the larger read at ORD-S is not known. A fresh review passed. The step had no board read.

**One step after V: the DH step on EDDF rows without surface inputs.** Some 2026 EDDF departures have an
aircraft type and no clock-head value: their own adsb.lol inputs are absent, and their airport-day has an observed
taxi time for under 20 % of its departures. A flag marks these rows from the inputs alone, before any error read:
14,587 rows (January 1,557, February 12,374, June 656, July 0). On these rows the final file serves
max(60, V + 0.164 × (DH − V)), with V the value of V15A-R9-LD. Every other row is V15A-R9-LD. V, V9, V9A and
V15A are equal on these rows, so the step gives the same values on each of them; V9A-R9, V9A-R9-LD, V15A-R9 and
V15A-R9-LD are equal on these rows too, and on the 14,160 rows where they differ from V, the step keeps 0.836 of
that difference. DH is a reduced clock head: a LightGBM bag (seeds 601 to 603,
the tree settings of the clock head) on the 2025 clock rows of the head's frame, with every adsb.lol input removed
and six OPDI surface columns added. Its 934,128 training rows hold no January or July row. The weight 0.164 is
fitted on held January and July 2025 together, with V out of sample on each month.

Held reads, with the rows' surface inputs set to their dark values and every fitted stage (V's weights, the route
level 2 and the weight) set on the other month:

- EDDF clock rows: +206.0 s² per row [+105.0, +303.6]; January +144.9, July +267.1.
- EDDF rows of the ordinary class: −604.0 [−1,180.8, −127.0]; July −1,312.0. By the 2025 clock share they are
  about 11 % of the February rows of the step.
- Size in 2026 per final row: EDDF +2.9 [+0.8, +5.1], with +2.59 from February. On held rows that are dark in
  fact, with no inputs set, the size is +0.39.

Selection facts: the flag and the dark values of the inputs were fixed before any error read, and LIRF stays out.
A first fresh review found labels of the scored month in the held reads. We moved every fitted stage out of the
scored month. The weight stayed 0.164 (equal at full precision), and the reads above come from this repair; before
it, the EDDF size was +4.8. A second fresh review passed. The step had no board read.

**Take-off times.** The take-off times of the January and July rows differ from those of `ranking.parquet`: the
mean squared change is 24.2 s², and 8.2 % of the rows keep their time. Against v73, the January and July
predictions move by 1.13 s per second of take-off change on clock rows and by 0.83 s on the other rows (a fit
without labels).

**Checks.** Each part has a record with its SHA-256, and a clean rebuild from git snapshots on x86-64 Linux ran
the steps again and compared each part with its record. 38 parts gave the same bytes, among them the V step and
the five parts of the DH step; the 21 input files of the s7 member did too. The plan's pack differs in 43 path
strings only and is byte-equal when the rebuild folder is mapped back. Three member fits run on a GPU and do not
repeat bit for bit, so we compared them by distance: C2-cat moved by rms 24.8 s, F1-fresh by 12.6 s, and seed 1
of the s7 member by 48.9 s, which is 0.72 of the distance between two of its seeds. With both GPU refits, route
level 2 moved the route by rms 8.3 s. The s7 fits 4 to 9 of V9 ran in another Python environment than fits 1 to 3;
each is 47 to 72 s rms from the mean of fits 1 to 3, in line with the spread of fits 1 to 3 (fit 1 against fits 2
and 3: 52 to 76 s). The effect of the environment itself is not measured. Fits 10 to 15 ran in the environment
of the repository (Python 3.10.12, torch 2.10.0); each is 47 to 77 s rms from the mean of fits 1 to 3 (fits 4 to
9: 47 to 72 s; the largest values of both sets are on full January 2026). The release replays read the saved
member values. On x86-64 Linux the replay of V9A gives the same bytes as the final build, also in its L2C step,
which ran the nine LightGBM seeds of the new L2C a second time. The replay of V9A-R9 gives the bytes of its
build too, in c5, L2C and the V file; it was the second run of that chain. The replays of V9A-R9-LD, V15A,
V15A-R9 and V15A-R9-LD give the bytes of their builds too, in c5, L2C and the V file. Before the upload: the schema, the IDs in template order, finite
values, the floor of 60 s (minimum 60 s, maximum 111,666 s), the stage counts by month, the LIRF tail (23 January
and July rows above 8 h, all at LIRF; v73 has 22), the day 2026-06-11 with an ADS-B gap (4,898 rows, day means in
line with the days next to it), and January and July against v73: rms 34.5 s, bias +0.73 s. By class: clock rows
17.0 s, other rows 41.8 s, and the 388 rows without an aircraft type 494 s. The largest group is 383 LIRF rows
without an aircraft type (rms 490 s): the design change above moves their calibrated no-plan value. The final
file passes the same checks (minimum 60 s, maximum 111,666 s, 28 rows above 8 h, all at LIRF). It differs from
V15A-R9-LD on the 14,587 rows of the DH step only (rms 22.1 s on those rows, mean −0.57 s; the floor binds on no row).
On January and July against v73 it gives rms 34.8 s and bias +0.59 s (23 rows above 8 h; v73 has 22). The clock head predicts with the saved boosters of a refit at 4 threads,
because the released boosters were not kept; a refit moves the head by rms 5.1 s. A refit at 8 threads in the
released order gives the released bits, and its head moves ours by rms 4.9 s. We kept ours, which the board check
read.

**Board check.** We uploaded the January and July rows of V once, as v76 (344,841 rows). The
forecast written before the upload was 220.401 s [215.746, 224.959], and 223.447 s without the design change.
As v76 it scored 223.8069 s (`docs/PUBLIC-SUBMISSIONS.md` row 76): inside the forecast range, and 0.36 s above
the forecast without the design change.
The read is inside the range, so we made no bug check. The read chooses nothing on the no-plan rows. The four
changes after V and the DH step came after this read; in the final file the DH step moves 1,557 of these January
rows (rms 29.7 s, mean −8.6 s).

**Code.** `scripts/build_entry.py` writes the final file (default target final-v-dh-r9-ld): it applies the DH
step to the saved values of V15A-R9-LD and the saved DH vectors, and fits nothing; it has no BLAS or tree step, so
its bytes are the same on every platform. `--target final-v-r9-ld` runs the chain zabd-final-r9, v64-final-r9 and
V15A-R9-LD from the saved member values (section 7), with `w2/ords_ld.py` as ORD-S; `--target final-v-r9`,
`--target final-v`, `--target final-iii-r9-ld`, `--target final-iii-r9`, `--target final-iii`, `--target final-ii`
and `--target final` give V15A-R9, V15A, V9A-R9-LD, V9A-R9, V9A, V9 and V, and `--target final-v-dh-r9`,
`--target final-v-dh`, `--target final-iii-dh-r9-ld`, `--target final-iii-dh-r9`, `--target final-iii-dh`,
`--target final-ii-dh` and `--target final-dh` the DH step on them. The bytes of V, V9, V9A, V9A-R9, V9A-R9-LD,
V15A, V15A-R9 and V15A-R9-LD are exact on x86-64 Linux. No entry of these chains can use the published-mean-shift recipe
of v37 and v40.

## 5. The main steps

Each row is a new team best on the board. The board step is the change in seconds from the previous
best; local G is the held read where one exists (s² per row). For v48–v52 it is cumulative against
v47, as in section 3.3; from v53 on it is the step against the named parent. The Row column gives
the row of `docs/PUBLIC-SUBMISSIONS.md` with the score and the file hash. In the section 5 tables, a
dagger (†) marks a local G that is high for a known reason; the note under each table names it.

| Entry | Date (UTC) | Board RMSE | Board step | What changed | Local G | Row |
|---|---|---:|---:|---|---:|---|
| v1 | 09-06 | 265.7612 | first | Residual stacker over six members on supplied columns | – | 1 |
| v3–v12 | 09-07 to 09-08 | 264.3861 → 262.7418 | −3.02 in total | Stacker on seven members; a 600 s airport window block (nearby departures share their error); per-airport masks; a member with added offset, surface and de-icing modules | – | 3–12 |
| **v13** | 09-08 | **248.4818** | **−14.26** | The observed off-block clock from adsb.lol ground positions, as a LightGBM residual head on the clock rows | fold RMSE −8.128 s † | 13 |
| v14–v22 | 09-09 to 09-10 | 248.2904 → 245.0207 | −3.46 in total | Runway-queue features on no-clock cells; larger clock head with a tail guard; more clock rows from a turnaround pass; 10-fit bag; more training months; a seasonal head | – | 14–22 |
| v23 | 09-25 | 244.1236 | −0.90 | Implied off-block clock: the first ground point timed back to the stand position from X-Plane and OpenStreetMap geometry | – | 23 |
| v24 | 09-26 | 242.8064 | −1.32 | A learner for the no-clock rows with 21 runway-state window proxies, weight 0.4967 on 148,197 rows | – | 24 |
| v32 | 09-28 | 242.7480 | −0.06 | The same proxies in a learner for clock rows, weight 0.128 on 173,539 rows | – | 32; `docs/WRITEUP.md` |
| v35 | 09-30 | 241.7342 | −1.01 | Level-2 stacker over eight members, fitted without labels of the scored month (month hold) | +178.2 † | 35 |
| v36 | 09-30 | 240.0047 | −1.73 | B50: half the month-hold stack and half v32, weight fixed before any score | +855.7 † | 36 |
| v37 | 09-30 | 239.9380 | −0.07 | Constant shift per airport-month toward the published 2026 means (off in the current entry) | +114.6 † | 37 |
| v40 | 10-01 | 238.9597 | −0.98 | The level 2 trained on nine months (route R4) | +394.1 † | 40 |
| v47 | 10-02 | 237.7071 | −1.25 | Clock head rebuilt on movement inputs (implied clock, OPDI leads, pushback-phase inputs), weight 0.3 | +807.4 vs the fold anchor † | 47 |
| v48 | 10-03 | 235.1964 | −2.51 | LIRF rows without a plan: the join rule and the placeholder rules (section 4.6) | +1,440.4 (383 rows) † | 48 |
| v49 | 10-03 | 232.9026 | −2.29 | stack2: half-weight clock head HB1, route with masked loss R4mk, joined-row bracket, practice rule | +2,837.3 † | 49 |
| v50 | 10-03 | 229.7933 | −3.11 | stack4: runway-pair sequence member, stand re-use floor, MLP member, six-seed bag, clock head on two January cells | +4,286.8 † | 50 |
| v51 | 10-03 | 228.8603 | −0.93 | stack6: event-sequence transformer, sequence pack 021, LIRF placeholder practice by airline route, clock head HB3u | +5,027.9 † | 51 |
| v52 | 10-03 | 227.7205 | −1.14 | stack7: runway sequence bag (12 seeds, pack 030), MLP on clock rows | +5,511.0 † | 52 |
| v53 | 10-04 | 227.4400 | −0.28 | stack8: sequence pack 037, mixed MLP on clock rows | +185.66 vs v52 † | 53 |
| v54 | 10-04 | 227.2506 | −0.19 | stack9: three stand and runway rules | +135.82 vs v53 † | 54 |
| v55 | 10-04 | 227.2199 | −0.03 | stack10: stack8 with the route's early stop on served rows (R4mks) | +89.76 vs stack8 † | 55 |
| v56 | 10-04 | 227.0332 | −0.19 | stack11: stack10 with the three rules | +134.46 vs v55 † | 56 |
| v57 | 10-04 | 226.9242 | −0.11 | stack12: sequence pack 039 in place of 037 | +29.33 vs v56 † | 57 |
| v58 | 10-05 | 226.3211 | −0.60 | stack12 with the tail: three fixed rules (C65, C66, C58) and one bound projection | +246.68 vs v57 † | 58 |
| v59 | 10-05 | 225.8463 | −0.47 | v58 with the final 2026 route (section 4.2); 2026 rows only | – | 59 |
| v64 | 10-06 | 225.7082 | −0.14 | e7b04ffd (v62: v59 with Z, AB and the dark-day twin, without C65) plus the T11b-L step (section 4.9) | +97.31 vs e7b04ffd † | 64 |
| v68 | 10-07 | 224.2430 | −1.47 | c5: v64 with the steps EXT, LONG, LIRF, NOAC and ROLL (section 4.10) | +1,990.11 vs v64 † | 68 |
| v69 | 10-07 | 223.8572 | −0.39 | L2C: half of a level-2 correction on clock rows | +112.8 vs v68 † | 69 |
| v70 | 10-07 | 223.7430 | −0.11 | ORD-S with the s1 and s2 transformers | +78.5 vs v69 † | 70 |
| v71 | 10-07 | 223.6029 | −0.14 | ORD-S: a second s2 seed, LIRF July at half of the coefficient | +50.1 vs v70 † | 71 |
| v72 | 10-07 | 223.5452 | −0.06 | ORD-S: s3 in place of s1 | +71.2 vs v71 † | 72 |
| **v73** | 10-08 | **223.3071** | −0.24 | ORD-S: s7 at 3 seeds in place of s2 and s3 | +95.1 vs v72 † | 73 |

† v13 is the arm with the largest clean gain among arms read on three folds that include January and
July 2025. v35 is the best of 16 level-2 variants read on the held months. v36 blends in v32, whose
held values come from the week-pair protocol (section 9), so its read is high; the keep rows of v35
add +206 to its read and are high for the same reason. v37 and v40 shift toward published monthly
means that contain the labels of the held month, and v40 keeps the form of v35. The choices of v47
to v58 are in sections 3.3 and 4.2 to 4.8, v64 in 4.9, and v68 (ROLL) to v73 in 4.10.

From v1 to v73 the board score fell by 42.45 s. The largest single step is the open ADS-B ground clock (v13, −14.26 s),
and the later work on the same data (v14–v23, v47) added about 5.6 s more. The rules on how the
label is recorded (v48, parts of v49 and v50, v54, v56, v58), the members of other kinds (v50–v53,
v57) and the month-hold stack with its half blend and final route (v35–v40, v55, v59) made most of
the rest. v64 added 0.14 s, and the steps of v68 to v73 (section 4.10) added the last 2.40 s.

Steps after v57, with their board reads (board gain as ΔMSE, s² per row, positive is better):

| Step | Change over its parent | Local G [95 % interval] | wo5 | Upload | Board gain |
|---|---|---:|---:|---|---:|
| exploit tail on stack12 | three fixed rules (C65, C66, C58) and one bound projection (section 4.8) | +246.68 [+111.70, +397.83] † | +94.00 | v58 | +273.4 |
| final route | the level 2 refitted for 2026; 2026 rows only (section 4.2) | – | – | v59 | +214.7 |
| Z | sequence members refitted with the time-of-day columns set to zero | +251.70 [+125.66, +378.42] † | +182.37 | v60 | −103.6 |
| AB | a 12-fit bag of the event-sequence member | +48.37 [+7.50, +90.90] † | +16.72 | v61 | +27.4 |
| dark18 | a sixth member trained with clock inputs hidden on some airport-days | +33.35 [−5.71, +70.61] † | +5.28 | v62, with the next row | +47.1 |
| C65 removed (`e7b04ffd…`) | the file-order rule C65 taken out of the tail | +0.99 [−0.42, +2.48] † | −0.04 | v62 | (above) |
| T11b-L (`9cd421dd…`) | the quarter step toward the dark-day twin on typed rows without an NM join, none on long-taxi airport-days (section 4.9) | +97.31 [+18.38, +174.96] † | +49.89 | v64 | +91.4 |

† Choices on the held months: the rules of the tail (section 4.8); Z, the better of two repair
forms, and AB, admitted after held reads on three bases (section 4.4); dark18, grown from 6 to 18
seeds after the first read (section 4.4); the removal of C65, after its read of −1.03 (section
4.8); T11b-L (section 4.9).

On 6 October we made five information uploads, each a whole entry read against e7b04ffd. Only v64
set a new best. Board gain is against v62 (ΔMSE, s² per row, positive is better):

| Upload | Entry, over e7b04ffd | Local G | Board RMSE | Board gain |
|---|---|---:|---:|---:|
| v63 | half fold fit and half annual fit for seven member vectors (a hedge) | no held gain claim | 225.9127 | −0.9 |
| v64 | T11b-L | +97.31 † | 225.7082 | +91.4 |
| v65 | T11b-L and three smaller steps (C102, K, R12) | +108.50 † | 225.8717 | +17.6 |
| v66 | six fits of a sequence member as a seventh stack column (one fit read alone first, five new seeds) | +90.59 † | 225.7757 | +60.9 |
| v67 | a January-only season head in the clock-head role | +9.83 † | 225.7162 | +87.8 |

† v64: section 4.9. v65: T11b-L gives +84.43 of the +108.50; the learner blend C102 for rows without
a plan entered because an earlier held read of it was positive, the clock-class copy K gains +4.95
on 20 held rows, and the 12-seed bags R12 repair a six-seed read of −2.56. v66: one of the six fits
(seed 17001) was read alone first, and we built the column because that read was high; about 60 of
the +90.59 depends on that fit, and the five new seeds alone give +31.51 [−41.89, +99.01]. v67: we
chose the January-only scope after the July read.

On 7 and 8 October we made eight uploads on top of v64 (section 4.10). Board gain is against the
previous best (ΔMSE, s² per row, positive is better):

| Upload | Entry | Local G over its parent | Board RMSE | Board gain |
|---|---|---:|---:|---:|
| v68 | c5: v64 with EXT, LONG, LIRF, NOAC and ROLL | +1,990.11 (+148.70 without ROLL) † | 224.2430 | +659.3 |
| v69 | v68 with L2C | +112.8 † | 223.8572 | +172.9 |
| v70 | v69 with ORD-S (s1 and s2) | +78.5 † | 223.7430 | +51.1 |
| v71 | v70 with a second s2 seed and LIRF July at half | +50.1 † | 223.6029 | +62.7 |
| v72 | v71 with s3 in place of s1 | +71.2 † | 223.5452 | +25.8 |
| v73 | v72 with s7 at 3 seeds in place of s2 and s3 | +95.1 † | 223.3071 | +106.4 |
| v74 | the one-seed form of v73 with five more steps, most of the gain on rows without a type | +1,044.7 † | 224.2436 | −419.1 |
| v75 | v73 with a second ORD-S column | +86.4 † | 223.5121 | −91.6 |

† v68: the band of ROLL and the choice to use it were made with the held rows in view (section
4.10). v69 to v73: the forms of L2C and ORD-S (section 4.10). v74: the bounds of the calibration
regions were chosen after a look at the pooled held table. v75: its form was one of four forms read
on the same held data (section 4.10).

## 6. What did not work

Each item gives the measured result and the cause we measured. A negative result closes the path
we tried, not the idea. G values are s² per row on the two held months unless the item says
otherwise.

1. **Weather in the learner.** ERA5 columns on the early learner moved the fold RMSE by −0.9 s in
   January and +2.3 s in July; the tail above 3,600 s did not move. A matched control showed that the
   same columns are worth 3.0–5.5 times more without `AOBT_3`: most of what weather explains is
   already in the actual times. Weather also lost in the clock head: −138 [−213, −60].
2. **A separate model for LIRF.** Its July gain came from one date-rollover row; without the two
   worst rows the gain fell to 1.6 s, and January and December lost. 90–160 thousand LIRF training
   rows cannot replace the 1.1–2.1 million rows of all airports
   (`experiments/exp-006-lirf/RESULTS.md`).
3. **A level 2 selected on held weeks inside the month (v34).** The week-pair read said
   +3,130 [+1,036, +5,644]; the board lost ΔMSE 381.7 against v32. The level 2 had learned
   airport-month patterns from other weeks of the held month, and 2026 has no labels of its month.
   The whole-month check had the right sign (−3,873). We moved to whole-month holds (v35: −1.01 s;
   `docs/PUBLIC-SUBMISSIONS.md` rows 32, 34 and 35).
4. **Specialist models for the noac rows.** A saved specialist on 21 LIRF noac rows screened at
   +2,188.8 [+5.5, +4,864.7] and lost 1.25 s on the board: the band was chosen on the same 20 fold
   rows. A level 2 on the 384 noac rows lost ΔMSE 303.7 (v41; `docs/PUBLIC-SUBMISSIONS.md` rows 40
   and 41): two rows with labels near 87,180 s carried all of its local gain, and this read chose its
   form among six variants. The fix for these rows
   was the rule set
   of section 4.6, not a model.
5. **Wider and coarser queue windows.** +14 lag features: −44.6 [−307.2, +221.4]; +60 features of
   15- and 60-minute airport windows: −233.8 [−578.8, +47.4]. The same-runway windows just before
   and after the departure hold the signal; wider windows add noise.
6. **Richer categories.** Stand, operator, destination and flight-prefix categories with target
   encodings got blend weight 0.00 in every fold. This was a screen on
   250,000 rows, not a test of a full-data model.
7. **A compact joint clock head.** Gradient boosting on y minus the observed taxi time on 100,000
   rows, blended 50:50: −1,193 [−1,808, −679].
8. **One offset per class for the 2025 → 2026 shift.** Every candidate offset changed sign between
   months, because clock coverage moved strongly between the years (section 4.3).
9. **Forecasting the LIRF placeholder per flight.** A perfect placeholder flag would give
   +2,598 [+928, +4,206]; a learned mixture used in place of the route gave −2,034 [−3,471, −666],
   and a 50:50 blend −402 [−1,020, +212]. The placeholder is a property of the flight (the
   airline's practice), not a state that neighbouring flights share. The airline and route practice
   rules of section 4.7 use this instead.
10. **Pseudo-labels for rows without a clock.** A clock-row teacher labelled lit rows, and a student
    was scored on unlit rows: −1,252.22 [−2,466.45, −11.06]. The student lowered the recorded-label
    regimes of LIRF July (placeholders and long rows).
11. **More learners on the same inputs.** 414 earlier member vectors sat on the limit where the
    best blend weight is zero (median error correlation minus sqrt(1/r): +0.003). A retrieval
    member scored −52.57 [−106.85, +3.20], with weights that changed sign between months. A new
    member needs new information or a new form of the data.
12. **A sequence model on the tree residual.** Six seeds gave +0.5 [−8.6, +8.7]. The seeds share
    most of their error: at an error ratio of 1.193 to the stack for one seed, the six-seed mean
    reaches 1.152 against a floor of 1.144 for an endless bag. On the final base it read −25.94
    [−46.12, −7.38].
13. **Hourly METAR.** As a linear carrier of the runway-hour state at LTFM and LFPG: −53.1 across
    months, with negative R² in all four directions. The runway-hour state is not in the hourly
    weather.
14. **Season weights in the no-clock learner.** Weighting training months by distance to the
    target month, over three seed pairs: +28.1 [−73.8, +121.7] on all held rows, −77.3
    [−169.2, +4.6] on the no-clock rows that the learner serves, +105.4 [+61.4, +149.9] on clock
    rows it does not serve. On the clock head the same idea reads +67.8 [+23.9, +120.9] in
    January; it is open and not built.
15. **Transfer-aware stack weights.** Reweighting the held rows toward the 2026 neighbour
    structure lost −342.0 [−511.2, −198.2]: the effective sample collapsed (EGLL July from 1,629
    rows to 28). A pooled-slope repair was stable but still lost −194.1 [−296.2, −117.6], and even
    perfect knowledge bounds the gain near +28 at the 0.83 ratio.
16. **More clocks at EDDF from inbound chains.** Even a perfect clock on the 337 recoverable rows
    is bounded at +15.3 [+9.1, +22.7]; about 81 % of the candidate rows already had a clock.
17. **Seed bags under the level 2.** A four-seed mean of one level-1 member in place of its single
    seed: local +3.0 [−130.0, +137.0], board +0.038 s worse (v46; `docs/PUBLIC-SUBMISSIONS.md`
    row 46). We did
    not measure the cause; our reading is that the level-2 stacker already averages the member's
    seed noise. Seed bags did help the members that enter the linear stack directly (section 4.4).

## 7. How to reproduce

The public export of this repository holds the source, this document, and a replay of the entry
from saved member values. Its `README.md` and `docs/REPRODUCE.md` are the instructions; this
section only names the paths.

**Replay (no member fit).** The replay reads the 14 official challenge files from
`<data-root>/data/raw/` and nothing else: `submitting.parquet`, `ranking.parquet` and the twelve
training files of 2025.

```bash
python scripts/build_entry.py --target v73 --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/v73-replay.json
```

The target `v73` replays our last entry of the ranking phase; it fits no member. It checks the 14 official files
by SHA-256. The default target of `scripts/build_entry.py` is `final-v-dh-r9-ld`, the final file (section 4.11;
`release/ENTRIES.json` `default_target`). It reads the two final files `final_ranking.parquet` and
`final_submitting.parquet` from `<data-root>/data/raw/`, checks both by SHA-256 and fits nothing. Without
`--target`, the command runs that default.
- **The v64 reference.** The saved zabd values must give the bytes of e7b04ffd (section 4.9), and
  the T11b-L step of `release/t11b_stage.py` on them the bytes of the v64 file `9cd421dd…`. These
  two checks hold on every platform.
- **The v64 chain.** `iso/dump_chain.py` reruns the assembly code of section 4
  (`experiments/exp-169-stack-plan/code/`, in the order of `build035.py`) on saved member
  predictions for held 2025 and for 2026, which ship without organizer IDs or labels. It refits the
  stack weights and the statistics of rules C95 and C98 (all other rules are fixed) from the
  official January and July 2025 labels, and compares each file with the reference: bytes equal on
  macOS arm64, values within 1e-6 s elsewhere. `--target zabd` and `--target v64` are the separate
  replays of these two files, with byte checks.
- **The steps of v73.** `release/v73_stage.py` runs `iso/` (c5) and `w2/` (L2C and ORD-S) of
  section 4.10 on that chain. It refits the LONG weights, the NOAC slope, the ROLL logistic, L2C
  (3 seeds) and the ORD-S coefficients, and reads the s7 values from `release/saved-vectors/s7c/`.

The s7 fits ran on a GPU and do not repeat bit for bit; the header of `w2/rebuild.sh` lists their
commands. The record gives the file hashes, the platform and the fitted values.

**Platform.** The bytes `70236754…` of v73 are checked on x86-64 Linux with Python 3.12, numpy
2.4.6 with OpenBLAS, pandas 2.3.3, pyarrow 24.0.0, lightgbm 4.7.0 (8 threads), scikit-learn 1.9.0
and scipy 1.17.1. The thread count changes the L2C trees (4 threads give `70715d41…`); on macOS
arm64 they differ too (`780d7ac8…`, 1.26 s rms from v73). v64 and the earlier targets replay byte
for byte on macOS arm64; on x86-64 Linux the v64 chain differs by at most 1.5e-11 s per row.

**Assembly from member vectors (development repository only).** The candidate was built by
`experiments/exp-169-stack-plan/code/build035.py` from pinned member vectors. These vectors are not
in the public export, so this step and the harness below run only in our development repository;
in the export, use the replay above. The general harness runs the same steps for the earlier
five-column form:

```bash
python experiments/exp-169-stack-plan/code/assemble_final_032.py run \
  --manifest <member manifest> --out <new directory>
```

It refits the stack weights when a member changes, rebuilds the residual-rule statistics and the
bound tail, and writes the candidate with a receipt.

**Full refit.** Each member is fitted by its own experiment code; `docs/LINEAGE-ZABD.md#members`
gives, for each member, the code, the recipe, the training months, the compute, the external data
and the label use. The level-1 members and the level 2 are in
`experiments/exp-148-short-route/code/`, the clock head in `experiments/exp-159-clock-head/code/`,
the MLP in `experiments/exp-164-double-down/code/` and the runway sequence models in
`experiments/exp-170-diverse/code/`. The inputs are the challenge files and the open sources of
section 2.2; the ADS-B ground clock is built by `experiments/exp-046-surfacefit/code/` and the
implied clock and stand geometry by `experiments/exp-127-clock-precision/` (`docs/REPRODUCE.md`).
Measured fit times, as sums of the fit receipts: the two parts of the clock-head bag took 3,224 s,
the six-seed sequence and row-control bags 1,651 s and 418 s, and the final level-2 route fit
R4mks 15,973 s. The sequence members of ORD-S are fitted by the code in `w2/`; the header of
`w2/rebuild.sh` lists the commands.

**Fits of the entry.** For 2026, v64 uses fits on all twelve months of 2025 for every part except
two, because same-month labels of 2025 transfer strongly to 2026 (section 3.3): the route's level 2
uses nine months of 2025 (route R4, section 4.2), and the clock head uses its ten-month fit. A
twelve-month refit of the clock head was planned for the freeze; we stopped the work before it. The
held reads of section 5 stay the evidence for that model; a full-year model has no held rows of its
own. The steps of v73 use the same split: the s7 member for 2026 is fitted on all twelve months of
2025, its held values come from fits without January and July, and the step coefficients for 2026
are fitted on both held months; the ROLL logistic uses all twelve (section 4.10).

## 8. How we worked

AI coding agents, built on Claude and GPT models, wrote most of the code and ran most of the
analysis for this entry. The human team member set the direction and made the decisions on which
kinds of work to run and when to stop a line of work. Each agent worked from a written task and
reported its results in files in the team's private repository. Four practices kept the results
checkable:

- **Fresh-context review.** Before an upload, a separate model instance with fresh context, which
  had no part in building the candidate, reviewed it. Every upload from v6 on had such a review,
  v73 included. Of the first five uploads, v2 and v3 were reviewed; v1, v4 and v5
  were not. Before v12, some uploads went up after a review that still had blocking findings; from
  v12 on, the last review before each upload passed.
- **Written expectation first.** Our rule was to write the form of each step and its expected
  result before its read (for uploads, `docs/VALIDATION.md` section 1). Every upload from v34
  (30 September) on, 42 in all, had a board forecast written before the upload; the forecasts are in
  our internal submission log, and section 3.3 compares the board with the local reads.
- **Records.** Results go to a ledger whose rule is append-only: a correction is a new row, and no
  row has been edited since 13 September. Uploads go to a submission log with their file hash, and
  the board score is added after the read; its public extract is `docs/PUBLIC-SUBMISSIONS.md`.
  Questions for the human go to a decision log, one section per question, with the answer. The
  ledger and the two logs are internal and are not part of this export.
- **No model value from a board score.** No weight, threshold, mask, residual or rate of the entry
  comes from a board score (`docs/input-rule.md`; `docs/PUBLIC-SUBMISSIONS.md`, introduction). We
  used the board to check our entries and our local validation (section 3.3). Board scores also
  guided some choices between whole entries. Uploads v26–v29, v31 and v33 were score probes
  under an earlier rule; no value of the entry comes from them (`docs/PUBLIC-SUBMISSIONS.md`).

Two agents built the steps of v73 from the v64 release export, the official files and the files of
the agent before them (section 4.10), without the development repository; a third agent built v74
in the same way. Each of their uploads had a forecast written before it and a fresh-context review.

## 9. Limits

- **Local and board scale.** The board gain of stack entries is about 0.83 of the local G, and we
  do not know why (section 3.3). All stack steps but Z had the local sign on the board; Z lost on
  the board against a held read of +251.70, the better of two repair forms read on the held months
  (section 4.4).
- **Small steps.** The last stack steps are +29 to +252 per step locally; some intervals contain
  zero, for example sequence pack 039 and the dark-day twin (sections 4.4 and 5). We report them with their intervals and their wo5 values.
  Most of them include choices made on the same held months (section 3.3).
- **The v73 bytes are an x86-64 Linux fact.** On macOS arm64 the L2C trees come out different, and
  the replay gives other bytes (section 7).
- **ORD-S and L2C forms were chosen on the held read.** The rows that ORD-S skips, the floor at 0,
  the per-airport coefficients, LIRF July at half of the LIRF coefficient, the members, the member
  window and the seed count, and the L2C factor of 0.5 and its clock-row scope, were chosen on the
  held rows that score them. Their
  intervals do not include this selection. On the board the ORD-S steps together kept 0.83 of
  their held G, and L2C 1.53 (section 3.3).
- **Few rows under c5.** ROLL rests on 6 held rows and moves 3 rows of 2026. EXT and LIRF change
  only 2026 cells without held rows; their reads are proxies on other cells (section 4.10).
- **Members differ between the held and the 2026 fits.** L2C and ORD-S fit their coefficients on
  members fitted without the held months, and apply them to members fitted on all of 2025. For the
  two-column form of v75 this is our belief about its board loss (section 4.10).
- **T11b-L rests on a small held read.** T11b-L reads +97.31 with an interval from +18.38 to
  +174.96. Its rows and its long-taxi rule came after reads on the held months. The rule rests on 3
  held days; 2026 has 15 such days, which no held read tests (section 4.9).
- **Rows without local support.** About 17,000 January clock rows at LEMD and LFPG have no 2025
  support; the clock head serves them by the same formula without a local read (section 4.3). In
  v73 the EXT step replaces their fallback pair with a pooled stack, with only a proxy read
  (section 4.10).
- **Rules for single rows.** Some bounds touch very few rows (C68 one row, C66 17 rows). They come
  from the join window and other supplied columns, not from labels of the scored months, but their
  local reads rest on few 2025 rows.
- **An older entry in the held route.** The held route contains v32 (section 4.2), whose held 2025
  values come from our earlier week-pair protocol, which trained on other weeks of the same month
  (section 3.1). The held stack weights are fitted on these values, so held reads on the final can
  be high by about +12 G [−36, +62] (wo5 −24). The submitted 2026 file is not affected: no 2026
  label enters it, and this path can only make the held reads high.
- **A known small defect in the clock match.** One check of the ADS-B clock match, the runway that
  the wheels-off position implies, measures distance in degrees without scaling longitude by
  cos(latitude) (`experiments/exp-046-surfacefit/analysis/chain-check/block_ungated.py`; the
  corrected copy `block_ungated_km.py` in the same folder is not used). With the corrected check,
  71 of the 344,841 ranking rows (and 20 held rows) would get the other clock class. On an older base
  this was priced at about 0.001 s RMSE, without a refit of the parts that depend on the class. The
  entry keeps the defect.
- **A single-candidate track match.** The track match of the clock head skips its wheels-off test
  when only one candidate segment matches (`experiments/exp-159-clock-head/code/track.py`, lines
  238–241). No held row is affected. In January 2026, 419 rows (159 + 260; 0.12 % of the ranking
  rows) carry track inputs from a later point set of the same aircraft segment, with the same onset
  and stand stop. The effect is not measured; the entry keeps it.
- **Gap inputs of the fallback sequence model.** The sequence model of the fallback pair (section
  4.5) reads the gaps to the previous and next departure in units of 1,000 s instead of seconds, and
  its input scale clips every gap above about 3,500 s to one value
  (`experiments/exp-170-diverse/code/prepare.py`). In v64 it serves 16,748 extension rows of 2026,
  99 of them at the clip, and no held row, so no local read tests it. In v73 the EXT step replaces
  the fallback pair on these rows (section 4.10).
- **A design choice in the runway-state proxies.** Before it finds ground segments, the proxy
  builder keeps only the points whose callsign is a departure flight of that day
  (`experiments/exp-134-proxies-in-learner/code/proxy_features.py`, lines 366–375). On two test days
  of 2026, keeping all points of the aircraft instead moves the segment anchor and changes the
  movement-onset clock on about a quarter of the rows that have it (median change +30 to +34 s). The
  same builder makes the 2025 training rows and the 2026 rows, so this is not a difference between
  training and serving. The gain of the other choice is not measured.
- **File order.** The entry reads no row order of the supplied file; the one rule that did (C65)
  was removed (section 4.8).
- **OPDI licence.** OPDI files name no licence (section 2.2).
- **Refit.** The replay rebuilds the entry from saved member values (section 7). No fresh fit
  from the raw files has reproduced the members end to end. The s7 values of v73 are saved for the
  same reason: its GPU fits do not repeat bit for bit.

## Records cited

Every path below is part of the public export.

- Scores and file hashes: `docs/PUBLIC-SUBMISSIONS.md` (cited by row).
- Members and rule stages: `docs/LINEAGE-ZABD.md` (sections "Members" and "Rule stages").
- Data and licences: `docs/DATA.md`, `docs/EXTERNAL-DATA.md`, `docs/GEOMETRY-DATA.md`,
  `docs/weather-provenance.md`, `docs/data-licences/GPL-2.0.txt`, `docs/external-data-rules.md`.
- Task reading and validation: `docs/input-rule.md`, `docs/VALIDATION.md`, `prc/data.py`,
  `prc/weekpair.py`, `prc/gscore.py`.
- Reproduction: `README.md`, `docs/REPRODUCE.md`, `docs/CLAIM-MAP.md`.
- Earlier write-up (v32): `docs/WRITEUP.md`.
- Route specification: `experiments/exp-148-short-route/SPEC.md`.
- The T11b-L step: `release/t11b_stage.py`.
- The steps of v73: `release/v73_stage.py`, `iso/` (c5), `w2/` (L2C and ORD-S; `w2/rebuild.sh`),
  the saved s7 member values `release/saved-vectors/s7c/`, and
  `docs/LINEAGE-ZABD.md#from-v64-to-v73`.
- Code: `experiments/exp-148-short-route/code/`, `experiments/exp-159-clock-head/code/`,
  `experiments/exp-164-double-down/code/`, `experiments/exp-170-diverse/code/`,
  `experiments/exp-169-stack-plan/code/`, `experiments/exp-046-surfacefit/code/`,
  `experiments/exp-127-clock-precision/`.
- Negative result with its record: `experiments/exp-006-lirf/RESULTS.md`.

## Glossary

- **G**: local gain in s² per 2026 ranking row on the held January and July 2025 departures
  (section 3.2). Positive is better.
- **wo5**: G without the five airport-days that contribute most.
- **Board ΔMSE**: the difference of the squares of two board RMSE values, s² per row.
- **Board / local ratio**: board ΔMSE gain divided by local G.
- **Classes**: clock (an ADS-B off-block clock exists), ordinary (known aircraft type, no clock),
  noac (no aircraft type).
- **Route**: the value before the clock head and the member stack (section 4.2).
- **Placeholder**: a block time recorded as the scheduled time.
- **Rule labels** (C20, C33, C55, …): the names of the rules in the code and in
  `docs/LINEAGE-ZABD.md#rule-stages`.
- **T11b-L**: the last step of v64, a quarter step toward the dark-day twin on typed rows
  without an NM join, except on long-taxi airport-days (section 4.9).
- **c5**: the five steps of v68 on rows outside the linear stack: EXT (a pooled clock stack on the
  January extension rows), LONG (half of a member-stack step on long-route rows), LIRF (half of an
  ordinary-stack step on LIRF January clock rows), NOAC (a quarter of a shrink toward 900 s on LIRF
  rows without a type) and ROLL (an expected value on three rollover rows) (section 4.10).
- **L2C**: half of a level-2 LightGBM correction of the residual on clock rows with an NM join
  (section 4.10).
- **ORD-S**: a step on ordinary rows toward an airport-context transformer, with one coefficient
  per airport, floored at 0 (section 4.10).
- **s1, s2, s3, s7**: airport-context transformers with 12, 24, 36 and 64 movements on each side
  of the departure; s7 is the member of v73 (section 4.10).
- **Held fit, full-year fit**: a member fitted without January and July 2025 gives the held
  values; a member fitted on all of 2025 gives the 2026 values.
