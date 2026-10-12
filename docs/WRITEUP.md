# Taxi-out time: the accepted entry v40

Team youthful-giraffe, PRC Data Challenge 2026. The accepted best entry is
**v40, 238.9597 s RMSE**, on **344,841 departures**. Its file is **4,707,529 bytes**,
with SHA-256 `c29c9dc434e2b2e0b910a382d8fd83a9dd6de3e05cf865203ba1a53383623e45`.
The public [model record](PUBLIC-MODEL.json), [results](PUBLIC-SUBMISSIONS.md)
and [entry registry](../release/ENTRIES.json) fix this target.
[CLAIM-MAP.md](CLAIM-MAP.md) maps the claims to source and hashed records.

## 1. Task and inputs

The target is take-off time minus airport off-block time, in seconds. The ranking
file retains actual take-off and Network Manager off-block fields; the two
movement target fields are blank. This is reconstruction after operations.
The [input rule](input-rule.md) and [external-data rules](external-data-rules.md)
state the allowed inputs. Open adsb.lol tracks supply surface clocks and state;
Gateway and OpenStreetMap supply stand geometry. ERA5 and OurAirports inputs
remain in the older base members. These inputs have their own data licences.

## 2. Short route and serving rule

V40 uses **R4-kno → 0.5 × R4-kno + 0.5 × v32 → frozen mean correction**.
R4 extends the R2 route below from six to nine months: March, October and
November add **24 fits** (eight members times three held months). These fits
exclude January, July and their own scored month with the same boundary
buffers. The level-2 owners q01 and q07 exclude their scored month's labels;
qfinal trains on the nine months: January, February, March, June, July,
August, October, November and December 2025. The selected form averages raw
LightGBM and CatBoost residuals before the floor. Both learners ran on the CPU.
The unchanged feature rule removes calendar month and v23 columns. The 12
queue residual columns enter; the 12 queue count columns do not enter R4.

R4-kno keeps v32 on the same **26,310 selected fallback rows** as v35.
The weight of 0.5 on R4-kno and 0.5 on v32 was fixed before this entry's score.
The frozen published-mean correction below is then applied **once** to that
unshifted half blend. It changes **322,108 rows**, holds **22,721** fixed, and
has RMS change **10.916 s**, maximum absolute change **27.578 s** against
unshifted R4-B50. Twelve admitted values remain at the floor.
[The choice record](../release/V40-CHOICE.json) fixes the route and hashes.

The earlier component routes remain available:

V36 is B50: **0.5 × v35 + 0.5 × v32** in official template order. The weight
was fixed before the blend's score; it was not fitted. Both component vectors
use a 60-second floor, so their mean needs no new floor. V35 uses the R2 route
below; v32 uses the older [fallback route](V32-FALLBACK.md). The component
records are [PUBLIC-MODEL-v35.json](PUBLIC-MODEL-v35.json) and
[PUBLIC-MODEL-v32.json](PUBLIC-MODEL-v32.json).

Exp-148 fits eight members under one split-aware path: T-tailv4, C2-cat,
L2-hub8725, D1-winter, F1-fresh, N1-nmtail, the annual no-clock proxy learner L,
and the annual with-clock proxy learner Lp. The anchor is the row mean of the
first five members. The stacker learns the residual target y minus that anchor.
LightGBM and CatBoost use the same out-of-fold member rows; the chosen form
adds the mean of their raw residual predictions to the anchor, then applies
its 60-second floor. It is not the mean of two separately floored predictions.

The R2 training rows cover December, February, June, August, January and July
2025. The four added jjx member fits exclude all January and July labels, their
own scored month's labels, and boundary buffer days: eight members times four
months, **32 added fits**. The January/July member fits exclude both months.
The local January stacker trains on the other five months; July does the same.
The final stacker trains on all six months, **1,028,150 rows**, then scores the
ranking rows from the eight members' final all-2025 fits. It uses **461 LightGBM
rounds and 2,208 CatBoost rounds**, chosen by leave-one-training-month-out CV.
The R2 feature list removes the four v23 columns and the calendar-month column.

The frozen variant choice is R2b-w100-kno: weight **1**, with v32's served values
kept on **388 no-aircraft-type rows**, **3,205 ordinary rows without AOBT**, and
**22,721 rows in unsupported cells**. Their union has **26,310 selected fallback
rows**. One other prediction is equal by chance, so **26,311 rows are unchanged**
and **318,530 change** against v32. The exact selection rules are in
[the public specification](../experiments/exp-148-short-route/SPEC.md),
[the choice](../release/V35-CHOICE.json), and the serving source.

V37 adds the frozen published-mean correction to v36. For each of 20
airport-month cells, the target is the published mean minus an OLS estimate of
the all-label coverage offset. The model shrinks the cell's mean toward that
target with lambda = tau² / (tau² + predictive variance). The zero-centred
bias prior uses the mean squared historical v24 cell error. The parameters
were frozen before shift scores and contain no board inputs.

Only rows in month-airport-class cells with at least two January/July 2025
training rows move: **322,120 admitted**, **22,721 held**. The cell's original
mean includes held rows. The same shrunk mean change is applied to admitted
rows; the model does not compensate for the held share. An 80-step bisection
finds the additive shift with a 60-second floor. **322,108 values change**;
12 admitted values stay equal at the floor. RMS change is **11.305 s**, maximum
absolute change **29.728 s**. The [frozen parameters](../experiments/exp-151-published-means/frozen-parameters.json)
and [applicator](../scripts/published_shift.py) define the exact calculation.

## 3. Validation and limits

V40 scored **238.9597**. Development gain of shifted R4-B50 against the
local shifted B50 stand-in for v37 was **394.1 s² per ranking row**, airport-day
95% interval **[294.3, 495.4]** (January **117.1**, July **277.0**). The route
alone against local B50 gave **395.7 [296.7, 496.1]**; the new shift against
local unshifted R4-B50 gave **113.0 [10.3, 213.0]**. All are development reads.
Full-route valid gain is unmeasured. Published held-month means contain held
labels; saved v32 references and historical bias priors retain earlier paths.
No coefficient, shrinkage weight, mask or rate is set from a board reading.


V37 scored **239.9380**. Its development gain against the local B50 stand-in for v36 was **114.6 s² per
ranking row**, airport-day 95% interval **[11.9, 215.6]**; January contributed **3.9**, July
**110.8**. The common-day and week-block intervals include zero.
This is an optimistic diagnostic: the published held-month aggregate
contains held labels. Leaving the held month out of coverage coefficients does
not remove that path. The bias prior and base retain earlier label paths.

B50's local gain against v35 was **855.7 s² per ranking row**, 95% interval
**[578.1, 1,156.6]**. The local v32 component has inherited label paths, so this
check is optimistic. V36 scored **240.0047**. The partner choice compared
existing entries, including v34's board loss; it did not fit a blend weight.
The fixed-weight blend source and public model record state this distinction.

The v35 component was chosen from a frozen grid of four weights and four keep sets
using 2025 scores under the month-hold level-2 condition. The chosen local gain
was **178.2 s² per ranking row**, 95% interval **[-365.8, 688.3]**, with probability
**0.245** of a non-positive gain. January contributed **-40.6**, July **218.8**.
R2b alone gave **-27.9**; the keep sets supply about **206**. Their local v32
reference predictions inherit held-label paths, so this part is optimistic.
The earlier week-pair stacker v34 scored **243.5329**; v35 scored **241.7342**.
The v35 board result checks the uploaded file, not the validity of that local
estimate or transfer to another year. No weight was fitted to a board score.

V32 remains the fallback, with its 16 inherited faults in
[PUBLIC-MODEL-v32.json](PUBLIC-MODEL-v32.json) and
[V32-FALLBACK.md](V32-FALLBACK.md). Its exp-137 stage failed the frozen gain gate
and was fitted under the positive-point rule. Probe uploads v26-v29, v31 and v33
are history. No released parameter, residual, mask or rate was derived from
those board readings. The R2 model and its variant were fixed before v35's upload.

## 4. External data and licences

The published monthly means are attributed to EUROCONTROL under its separate
[source terms and notice](EUROCONTROL-DATA.md). The workbook is not exported.
Source code is GNU GPL version 3; see [LICENSE](../LICENSE).
adsb.lol archives have ODbL-1.0 terms and CC0-1.0 contributed contents.
OurAirports is public domain. Gateway and OpenStreetMap stand geometry retain
GPLv2 and ODbL terms, separate from the source code; the primary Gateway upload
terms and the scenery README's version-2-or-later reading are both stated in
[GEOMETRY-DATA.md](GEOMETRY-DATA.md). The exact ERA5 bundle is separate under
CC BY 4.0. Contains modified Copernicus Climate Change Service information 2026.
Neither the European Commission nor ECMWF is responsible for any use of this
information. [weather-provenance.md](weather-provenance.md) records DOIs and
transformations; [EXTERNAL-DATA.md](EXTERNAL-DATA.md) records archives and matches.
Private derived Kaggle packages are rebuilt from challenge data and these open
sources. They are not independent open external sources.

## 5. Replay and a new fit

The export contains saved R4-kno prediction values and the v32 replay chain,
without movement IDs or labels. V40 replays both, applies their fixed half
blend, then the frozen mean correction once. A saved binary admission vector
supplies the support decision without IDs or class rows. Airport and UTC
month come from the official ranking input. The replay verifies all six
output hashes along the v40 route. The earlier v37 route remains an explicit
target and verifies its own six hashes. The official template supplies IDs.
The target parameter uses the entry registry.

**No fresh fit has reproduced v40 or its components.** A new fit needs the twelve official 2025
training files, exact raw and open-source versions, rebuilt monthly features,
class rows, proxy packages and member artifacts, then the R4 stacker fits and
v32 fallback. The mean correction also needs the published workbook and
2025 coverage summaries, historical v24 held predictions and raw/open-clock
class rows to rebuild the support decision. The two boundary-row exceptions are derived from caller-owned raw
files; their IDs are not distributed. Library versions, vocabularies, CV rounds
and thread/device choices can change the fitted values.
[REPRODUCE.md](REPRODUCE.md) gives the sources, ordered commands and open gaps.
