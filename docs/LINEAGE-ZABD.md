# Lineage of the release targets zabd, v64 and v73

`zabd` is the candidate file with SHA-256
`e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636`
(344,841 rows, 4,707,119 bytes). It scored 225.9106 s RMSE as upload v62. It is the assembly
with SHA-256 `0eec38e86fadd6a658dabd1e877bd7603301429b6523d375421466efd8cb7779` without the C65 row-order step: 35,054 of the
2026 rows move, by at most 12 s. The parent of both, the same assembly without
the dark18 member, has SHA-256
`9436b85b0c5b2c83dd4b6b1be16076166337e7c274a3d4e6a59481843fd9cf9c`.
The name lists its parts: the **Z** sequence members (time-of-day input set to
zero), the **AB** bag of the g member, and **dark18**. Scored uploads are in
[PUBLIC-SUBMISSIONS.md](PUBLIC-SUBMISSIONS.md); data sources and licences are
in [DATA.md](DATA.md); the replay command is in [REPRODUCE.md](REPRODUCE.md).
Release target v64 is zabd with one more fixed stage
([From zabd to v64](#from-zabd-to-v64-the-t11b-l-stage)). The final entry v73
is v64 with five fixed-scope steps, a level-2 correction on clock rows and a
sequence-member column on ordinary rows ([From v64 to v73](#from-v64-to-v73)).

Two prediction sets exist for each part. The **held** set predicts the
344,419 departures of January and July 2025 from fits that did not train on
those two months (one exception, the v32 half of the route, is in
[Label use](#label-use)); the level-2 weights and rule statistics are fit on
it. The **2026** set predicts the 344,841 ranking departures. Most members
were refit on all twelve months of 2025 for it; the clock head HB3u uses its
ten-month fit (2025 without January and July), and the route's final level 2
uses nine months. The [members](#members) table gives each part's training
data.

## Assembly order

`experiments/exp-169-stack-plan/code/assemble_final_032.py` runs the steps
below for both sets, and `experiments/exp-169-stack-plan/code/build035.py`
adds the dark18 column to step 4.

1. Route: R4mks B50, the mean of a level-2 route stacker and the v32 route.
2. Clock head HB3u at weight 0.5 on the rows of its scope (`assembly.py`, `head_blend`).
3. Fallback pair: sequence011 and row011 on the pair scope, row011 on its
   ordinary rows only (`pair_blend.py`, `route_swap.py`).
4. Primary blend: weighted least squares of six members on the residual after
   the head: g, mlp, sequence021, sequence039, dark18 and row6; ordinary rows
   use six columns, clock rows the first five (`dark_twin035.py`,
   `lane_c_swap.py`, `stack7.py`).
5. Rule stages C39b practice, no-plan v3cal, C73, C33 and C55
   (`stack7.py`, `stand_floor.py`).
6. Rule stages C71, C95 and C98; C95 and C98 statistics are refit (`exploit_rules.py`).
7. Rule stages C66 and C58 (`exploit029_rules.py`). The tail runs with the C65
   row-order offset set to zero, so C65 changes no row of zabd.
8. One projection onto the lower and upper bounds.

## Members

All member code paths below are in this export. Member predictions are saved
values in `release/saved-vectors/zabd/`; the replay does not refit members.

| member | model | code | training data | compute | external data |
|---|---|---|---|---|---|
| route R4mks B50 | mean of a LightGBM and a CatBoost residual stacker over eight level-1 members (158 features), averaged 0.5/0.5 with the v32 route; v32 alone on 26,310 rows without aircraft type, without AOBT_3 or without member support | `experiments/exp-148-short-route/code/`, `experiments/exp-147-foldserve/code/` (see [gaps](#gaps)) | final level 2: 1,540,605 out-of-fold rows of nine 2025 months; held: fits without January and July | CPU, 16 threads, 4.4 h for the final level 2; level-1 members on CPU and GPU | adsb.lol, OurAirports, Gateway and OpenStreetMap stands; ERA5 in the v32 half |
| head HB3u | LightGBM bag, 10 seeds by 2 shapes, 255 leaves, 1,800 rounds; target y minus the observed ADS-B clock | `experiments/exp-159-clock-head/code/` (`frame.py`, `track.py`, `fit.py`, `merge.py`, `pack.py`) | 934,128 clock rows of 2025 without January and July; the 2026 set uses the same fit | CPU, 8 threads | adsb.lol, OPDI, Gateway and OpenStreetMap stands |
| g (AB bag) | equal mean of 12 runway-sequence transformers: recipe A (width 64, 2 layers) and recipe B (width 128, 3 layers), 6 seeds each, 17-movement windows | `experiments/exp-181-modela-bag-read/serve-packet/code/` | held: 1,723,505 rows of ten months; 2026: 2,085,047 rows of twelve months | held: CPU; 2026: GPU, 12 fits | none; official columns only |
| mlp | embedding MLP 512-512-256 on the residual to an anchor prediction (route level-1 member Lp); ordinary rows 2 seeds, clock rows 12 seeds | `experiments/exp-164-double-down/code/mlp007/`, `experiments/exp-164-double-down/code/mlp008/` | held: ten months; 2026: twelve months | GPU | adsb.lol, OPDI, Gateway and OpenStreetMap stands |
| sequence021 | bidirectional GRU over airport-day-runway windows, width 64, 6 seeds | `experiments/exp-170-diverse/code/fit_runway.py`, `average_seeds_n.py` | held: ten months; 2026: twelve months | GPU | adsb.lol, OPDI, Gateway and OpenStreetMap stands |
| sequence039 | same GRU family, 30 fits of three widths and lengths, last-epoch weights | `experiments/exp-170-diverse/code/fit_runway_fast_029.py`, `average_mixed_033.py` | as sequence021 | GPU | as sequence021 |
| dark18 | GRU width 128, 3 layers, 48 epochs, 18 seeds; 38 clock, observation, track and OPDI inputs set to zero on selected training airport-days at the 2026 coverage rates | `experiments/exp-170-diverse/code/fit_dark_annual_064.py`, `prepare_coherent_dark_053.py`, `average_size_027.py` | held: ten months; 2026: twelve months | GPU, two processes of 9 seeds | as sequence021 |
| row6 | per-row embedding MLP without sequence context, 6 seeds; ordinary rows only | `experiments/exp-170-diverse/code/fit.py`, `average_rows_021.py` | as sequence021 | GPU | as sequence021 |
| sequence011 (fallback) | GRU, one seed, runway-ordered view | `experiments/exp-170-diverse/code/fit_runway.py`, `prepare_runway_time_059.py` | as sequence021 | GPU | as sequence021 |
| row011 (fallback) | one seed of the row6 bag | `experiments/exp-170-diverse/code/fit.py` | as sequence021 | GPU | as sequence021 |

The sequence and row members read the input arrays built by
`experiments/exp-170-diverse/code/prepare.py` and its variants; the Z inputs
come from `prepare_time_day_052.py`. The clock head and the sequence members
share the frame of `experiments/exp-159-clock-head/code/frame12.py`.

## Rule stages

Each stage changes only the rows of its scope. "Refit" means the replay fits
the statistic again from the caller's 2025 labels.

| stage | what it does | code | in the replay |
|---|---|---|---|
| primary and pair blends | weighted least squares of members on the post-head residual, with weights for the 2026 cell mix; each held month uses the other month's coefficients, the 2026 set pools both | `experiments/exp-169-stack-plan/code/dark_twin035.py`, `stack7.py`, `route_swap.py` | refit |
| no-plan v3cal | LIRF departures without a planned off-block time (397 held, 383 in 2026): rule chain, then a class mixture | `experiments/exp-167-target/code/noplan_form_v3.py`, `noplan_form_v3_cal.py` | saved |
| C39b practice | LIRF departures of airlines with a high default share: move toward the scheduled-time gap | `experiments/exp-167-target/code/practice_arm_r1.py`, `experiments/exp-169-stack-plan/code/prepare_unit2.py` | saved inputs |
| C73 | LIRF route practice on 7 airline and destination routes | `experiments/exp-169-stack-plan/code/c73.py`, `experiments/exp-165-levers/checks/c73_route_practice.py` | saved inputs |
| C33 | clip to the planned gap plus or minus 3,606 s, floor 60 s | `experiments/exp-169-stack-plan/code/read_stacks.py`, `assembly.py` | constant |
| C55 | stand re-use floor at EDDF, EGLL, LFPG and LTFM | `experiments/exp-169-stack-plan/code/stand_floor.py`, `read_c55.py` | saved inputs |
| C71 | wider stand floor at the same four airports | `experiments/exp-168-fresh-look/code/u13_inbound.py` | saved inputs |
| C95 | soft stand floor by airport and class | `experiments/exp-168-fresh-look/code/u27_floor3.py`, `experiments/exp-169-stack-plan/code/exploit_rules.py` | refit |
| C98 | ordinary rows at 5 airports: slope times the mean residual of nearby same-runway clock rows | `experiments/exp-169-stack-plan/code/exploit_rules.py` | refit |
| C66 | LIRF mirror rows: max(60, movement time minus scheduled time, or 600) | `experiments/exp-169-stack-plan/code/exploit029_rules.py`, `experiments/exp-168-fresh-look/code/u12_mirror.py` | saved inputs |
| C58 | LFPG twin rows: halfway to a least-squares point | `experiments/exp-169-stack-plan/code/exploit029_rules.py`, `experiments/exp-168-fresh-look/code/u11_twins.py` | saved inputs |
| bounds | lower: max of C33, C55, C71, C64, C68; upper: min of C33, C40, C63 (3,600 s at LEBL and LEMD) | `experiments/exp-169-stack-plan/code/exploit029_rules.py`, `experiments/exp-160-feature-worker/code/exploit_links_014.py` | saved inputs |

"Saved inputs" means the rule's frame columns are saved values built from
official input fields; "saved" means the stage's predictions are saved values.
No rule stage reads external data.

C65, a plus or minus 6 s offset read from the row order of the official files,
is not in zabd. Its frame column `order_delta` stays in the saved values,
because the replay checks the assembly with C65 before it removes the step.

## Label use

- No fit used 2026 labels; the ranking file has none.
- Every member's held predictions come from fits without January and July
  2025. One exception: the held half of the route's v32 component comes from
  saved month-fold references that carry known label paths. It enters the held
  weight fits and held reads, not the 2026 predictions.
- The fitted blend weights and the C95 and C98 statistics use the other held
  month for each held month and pool both months for 2026.
- Rule constants and their checks (C55, C71, C63, C64, C40, C66) were set or
  checked on all 2025 labels, the held months included. The choice of the Z
  inputs and the admission of dark18 were read on the held months. Held
  readings of these choices are therefore optimistic.
- The v73 steps fit on the held January and July 2025 labels (LONG weights,
  NOAC slope, L2C trees, ORD-S coefficients); each held read fits on one held
  month and scores the other, and the 2026 fit pools both. The ROLL logistic
  fits on the 2025 rows of its kind. The s7 member reads the taxi-in times of
  arrivals (the ranking file holds those of 2026) and the NM times of
  departures; it reads no departure block time.

## What the replay reproduces

`scripts/build_entry.py --target zabd` rebuilds every input frame from the
four official files and the saved values, checks each frame against its
recorded digest, refits the level-2 weights and the C95 and C98 statistics,
and runs the original assembly code. It writes the parent, the assembly with
C65 and the candidate, and compares all three with the recorded SHA-256 values. On the build machine
(Apple M-series, numpy 2.4.6 with Accelerate, pandas 2.3.3, pyarrow 24.0.0,
two threads) the bytes are exact; it takes about 7 s and 1.9 GB of memory.
Another BLAS can change the last bits of the weights; the record then gives
the largest difference to the saved entry in seconds.

The replay does not refit any member. No fresh member fit has reproduced the
saved member values.

## From zabd to v64: the T11b-L stage

Release target v64 is the file with SHA-256
`9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b`
(344,841 rows, 4,707,183 bytes). It scored 225.7082 s RMSE as upload v64. It is
zabd with one fixed stage, T11b-L (`release/t11b_stage.py`). The stage fits
nothing and reads no label.

1. P2 rows: the 4,902 rows of 2026 with an aircraft type, a row class other
   than noac, and no NM flight join. None of them is in the scope of the
   primary blend (step 4 of the [assembly order](#assembly-order)), so zabd
   gives them no dark18 term.
2. On these rows the value moves a quarter of the way toward the dark18 2026
   value, inside the bounds of step 8, with the 60 s floor.
3. Every row of a long-taxi airport-day keeps the zabd value. Departures are
   grouped by airport and UTC day. A day is long when 8 or more departures
   have an NM taxi (take-off minus AOBT_3) of one hour or more, and these are
   2 % or more of the departures with AOBT_3. 2026 has 15 such days (EGLL 1,
   EHAM 8, LFPG 3, LTFM 3); 802 of the P2 rows are on them.

4,078 rows change, by −6.0 s on the mean (RMS 57.2 s). On the held months,
with the held dark18 values and the same rule, the stage reduced the mean
squared error of zabd by 97.31 s² per row (95 % interval 18.38 to 174.96). The
rows, the member and the long-taxi guard were chosen with reads on the held
months, so this read is optimistic. We wrote the long-taxi guard after the
read of the step without it, from a check that read no labels; on the three
long-taxi days of the held months the move then increased the error by 14.85
s² per row (76 rows). No value of the stage comes from a board score.

`scripts/build_entry.py --target v64` replays zabd,
applies the stage and compares the bytes with the recorded SHA-256 value. Its
inputs are those of zabd: four saved frames of `release/ZABD-PACK-ranking.json` (the
dark18 2026 values, the bounds, the row class with the plan brackets, and the
primary-blend scope) and the official ranking file. The stage adds about 2 s.
The held read is not part of the replay: it needs the 2025 labels and the
private scorer.

## From v64 to v73

The final entry v73 is the file with SHA-256
`70236754db85e5f23fac281ebbc9c14a3a4fe33c4e6f34e7c95262eebe1659d8`
(344,841 rows, 4,707,451 bytes). It scored 223.3071 s RMSE as upload v73, our
best scored upload. Two agents built it after upload v64 from the v64 release
export, the official files and the files of the agent before them. The code is
in `iso/` and `w2/`; it comes from
the commit `48468cad9103967451faa67f4b3c3c58bcee2c00` of their work folder.
`iso/dump_chain.py` replays the v64 chain with the release code and keeps
every stage of it for both splits. The steps change the stage values, and the
v64 rule chain then runs again (`iso/chain.py`).

Held reads are gains in mean squared error (s² per 2026 ranking row, + is
better) on the held January and July 2025 departures, weighted to the 2026
mix: each step fits on one held month and is scored on the other. The
interval is a 95 % interval over airport-day resamples; wo5 is the read
without the five airport-days that add most.

1. **c5, five steps on rows outside the linear stack (upload v68,
   `iso/candidate.py`).**
   - EXT: the 17,017 January 2026 clock rows at LEMD and LFPG, which have no
     2025 rows of their kind, get a pooled clock stack of five members on the
     head base in place of the one-seed fallback pair.
   - LONG: the 4,222 long-route rows (route 2,400 s or more, with AOBT_3,
     outside the stack scope) move half a step toward a member stack.
   - LIRF: the 5,700 LIRF January 2026 clock rows move half a step toward the
     pooled ordinary stack of six members on the route base.
   - NOAC: the 232 LIRF rows without an aircraft type and with a mixed value
     move a quarter step toward the 900 s mode of real taxi times.
   - ROLL: three LIRF rows without an aircraft type took off 56,520 to
     61,322 s after their schedule, one local date later. Their value is the
     expectation over the two possible labels: the schedule placeholder,
     and a label one day too long (86,400 s plus 1,172 s).
     A logistic in take-off minus schedule, fit on the 2025 rows of the same
     kind, gives the share of each.

   52,468 rows of 2026 change. Held read against v64: +1,990.11
   [−945.00, +5,561.74], wo5 −565.11. ROLL on six held rows carries most of
   it; without ROLL the read is +148.70 [+44.58, +279.98]. Board: 225.7082 to
   224.243 s.
2. **L2C, a level-2 correction on clock rows (upload v69, `w2/l2c.py`).** On
   the clock rows with an NM join, a LightGBM model learns the residual of c5
   (clipped to ±2,000 s) from the deviations of the saved members from c5,
   the row inputs of `iso/features.py`, the airport and the row class (Huber
   loss 300, 31 leaves, 800 rounds, 3 seeds, 8 threads). Half of its
   correction is added, inside the bounds of c5, with the 60 s floor. The EXT
   rows and the LIRF January clock rows keep c5. 171,914 rows of 2026 change.
   Held read against c5: +112.8 [+47.4, +178.9], wo5 +48.2. Board: 224.243 to
   223.8572 s.
3. **ORD-S, a sequence-member column on ordinary rows (uploads v70 to v73,
   `w2/ords.py`).** The member s7 is a transformer over the traffic of the
   departure's airport: the 64 movements before and after it within 3 hours,
   arrivals with their taxi-in times and departures with their NM times
   (`w2/seq/`; width 192, 4 layers, 16 epochs, 3 seeds in each of the held
   and the full mode). On the ordinary rows of the stack scope, the value is
   c5 plus β times (s7 minus c5), with one β per airport from weighted least
   squares on the held rows, floored at 0. LIRF July rows take half the LIRF
   β. Rows of the long-taxi days, clock rows and rows without an aircraft type
   keep the L2C value. 2026 β: EDDF 0, EGLL 0.194, LEBL 0.229, LSZH 0.269,
   LFPG 0.316, LTFM 0.34, LIRF 0.375, LEMD 0.436, EHAM 0.59, EDDM 0.617.
   121,592 rows of 2026 change. Held read against L2C: +294.9 [+180.0,
   +414.4]. Uploads v70 to v72 used earlier members (shorter windows, fewer
   seeds). Board: 223.8572 to 223.3071 s over the four uploads, 0.83 of the
   held read.

From c5 to v73 the held read is +407.7 and the board gain +418.9 s² per row
(1.03). The forms of L2C and ORD-S were chosen on the same held months that
score them: the step size, tree settings and clock-row scope of L2C; the rows
that keep their value, the floor at 0, one β per airport, the half β of LIRF
July, the members, the window width and the number of seeds of ORD-S. The 17
rows of 2025 that showed the ROLL practice include the 6 held rows that score
it, and the band of ROLL and the choice to use it were made with these rows in
view. The held reads are therefore optimistic. Board scores were read for
calibration only; no value of v73 comes from a board score. Uploads v74 (from
a third agent, 224.2436 s) and v75 (a two-column form of ORD-S, 223.5121 s)
did not beat v73.

`scripts/build_entry.py --target v73` rebuilds v73
(`release/v73_stage.py`). It checks the 14 official files by SHA-256 (the
submitting file, the ranking file and the twelve 2025 training months),
builds the v64 chain reference from the saved zabd values and the T11b-L
stage, and writes the s7 member values of `release/saved-vectors/s7c/` back as
float32 arrays. It then runs the steps of `w2/rebuild.sh` in a new run folder
with the thread settings of that script. The chain replay refits the zabd
level-2 weights and the C95 and C98 statistics, and the steps refit the LONG
weights, the NOAC slope, the ROLL logistic, the L2C trees and the ORD-S
coefficients, all from the 2025 labels; no member is refit. On x86-64 Linux the bytes are exact. On
macOS arm64 the chain and c5 are exact (c5 `63850254…`), and the L2C trees
differ, so the file differs from v73 by 1.26 s rms; the record gives the
difference to the saved v73 values. The L2C thread count also changes the
trees: with 4 threads the file was `70715d41…`.

## From v73 to final

V (SHA-256 `121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a`,
670,790 rows, 8,641,506 bytes) applies the model of v73 to the organizers'
final files. Four later V files change V:

- final-ii (SHA-256 `280db2058f10a5cc3d44acdcf20aed80aae143cfadf2c9c299ddf7212c4f7afc`):
  the s7 member is the mean of 9 fits in each mode (V: 3). 236,289 rows of 2026
  change against V (ORD-S rows). Held read against V: +32.39 [+7.08, +58.07],
  wo5 +14.01.
- final-iii (SHA-256 `6de02372de5d1aea062300a88af7af51606ac2d6aee645c384b09398c9c57378`):
  final-ii, and L2C also reads the s7 deviation from c5, with 9 LightGBM seeds
  (`w2/l2c_candidate_a.py`). 569,604 rows of 2026 change against V. Held read
  against V: +38.09 [+13.01, +65.44], wo5 +19.71; against final-ii, the L2C
  change gives +5.70 [−0.15, +11.63] on the held months that chose its input.
- final-iii-r9 (SHA-256 `1c1ed67bdd7e54182bafb9a3eb08b48e6e708e981af6bb607c0016ca1f010c4b`):
  final-iii on the R9 assembly. The MLP column of the primary frame holds the
  12-fit MLP values on every member-scope row (final-iii: on clock rows only,
  2 seeds on the other rows; the held rows read out-of-fold 12-fit values,
  saved in `release/saved-vectors/final-r9/fold/primary_frame/mlp.json`).
  Every blend weight refits, and the steps after the assembly run again.
  292,360 rows of 2026 change against final-iii: 282,499 ordinary rows and
  9,861 LIRF clock rows of January and February that the c5 LIRF step serves
  with the refitted ordinary weights. Held read against final-iii: +23.11
  [+4.77, +41.83], wo5 +7.29, on the held months that found R9.
- final-iii-r9-ld (SHA-256 `260b427b83f99921b26ce372bad78fbcb3718b402e7a8bd0725a914ae3da114d`):
  final-iii-r9 with LD, another ORD-S step (`w2/ords_ld.py` with the flag
  `--long-days`). Its apply masks also take the ordinary member-scope rows of
  long-taxi airport-days (v73: these rows keep their value); its fit still
  leaves out these days, so the 2026 β are equal. 6,753 rows of 2026 change against
  final-iii-r9 (LTFM 5,366, LFPG 1,064, EGLL 199, EHAM 124), none of them a
  row of the DH step. Held read against final-iii-r9: +3.16 [−0.20, +8.37],
  wo5 −0.10, from 354 rows on 3 January airport-days of the held months that
  found LD.
- final-v, final-v-r9 and final-v-r9-ld (SHA-256
  `a9146e9d3c057e5fbad3c7acf7f8e448efbbfdfdbe02f9e073948fc6ab34e903`,
  `18a7bfb77bfdc5a68387cc9d96acbf3f276d441748fa7c2fd807a340e753ab82` and
  `31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759`): final-iii,
  final-iii-r9 and final-iii-r9-ld with the s7 member as the mean of 15 fits in
  each mode (fits 10 to 15 with the released fit code and flags; only the seed
  and the output path change). Held reads: final-v against final-iii +16.43
  [+4.55, +28.31], wo5 +7.39; final-v-r9 against final-v +23.16 [+4.94, +41.93],
  wo5 +7.10; final-v-r9-ld against final-v-r9 +3.08 [−0.19, +7.95], wo5 −0.10.
  The fit count and the seeds were fixed before the fits; the reads use the
  held months of the read from 3 to 9 fits. 576,351 rows of 2026 change
  between final-v-r9-ld and final-iii-r9-ld (rms 2.5 s), none of them a row of
  the DH step.

The final file is final-v-r9-ld with the DH step (next section).
[The write-up, section 4.11](writeup/METHOD.md) gives the design, the design
changes, the DH step, their held reads and selection facts, and the checks. The
release builds each V file as the chain zabd-final, v64-final and final (or
final-ii, final-iii, final-v; for final-iii-r9, final-iii-r9-ld, final-v-r9 and final-v-r9-ld the chain
zabd-final-r9, v64-final-r9 and one of these four) in `release/ENTRIES.json`, each entry with `"mode": "final"`:

- zabd-final: the zabd assembly on the final pack `release/ZABD-PACK-final.json`
  with the member and frame values of the final build, saved in
  `release/saved-vectors/final/`. The ranking bundle frame comes from the LIRF
  rule frames of the final build, which use the R4mks route as the base of the
  LIRF rows without a plan, with r1 weight a = 0.444; its calibrated no-plan
  value is the saved column `np_v3cal`.
- v64-final: the T11b-L stage on zabd-final (7,497 rows change).
- final: the v73 steps of the previous section on v64-final, with the final
  files (`iso/finalmode.py`), the row order of the chain (`chain_ids.py`), and
  the s7 member values of the final build
  (`release/saved-vectors/s7c/final-fold.json` and
  `release/saved-vectors/s7c/final-ranking.json`).
- final-ii and final-iii: the same steps with the 9-seed s7 member values
  (`release/saved-vectors/s7c9/final-fold.json` and
  `release/saved-vectors/s7c9/final-ranking.json`). For final-iii the build
  writes the float32 member values as the npz that L2C reads, and runs
  `w2/l2c_candidate_a.py` as the L2C step.
- zabd-final-r9, v64-final-r9 and final-iii-r9: the same three steps on the R9
  pack `release/ZABD-PACK-final-r9.json`, the final pack with the R9 MLP column
  of the fold and 2026 primary frames (`release/saved-vectors/final-r9/`).
- final-iii-r9-ld: the steps of final-iii-r9, with `w2/ords_ld.py` as the
  run's `w2/ords.py` and the flag `--long-days` at ORD-S.
- final-v, final-v-r9 and final-v-r9-ld: the steps of final-iii, final-iii-r9
  and final-iii-r9-ld with the 15-seed s7 member values
  (`release/saved-vectors/s7c15/final-fold.json` and
  `release/saved-vectors/s7c15/final-ranking.json`).

February takes the January scope of each fitted step and June the July scope.
In the zabd assembly, `experiments/exp-169-stack-plan/code/final_months169.py`
gives the final months and the row count. No member is refit.

## Release targets final-v-dh-r9-ld, final-v-dh-r9, final-v-dh, final-iii-dh-r9-ld, final-iii-dh-r9, final-iii-dh, final-ii-dh and final-dh

`final-v-dh-r9-ld` is the final file `youthful-giraffe_final.parquet` (SHA-256
`eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02`, 670,790
rows, 8,641,508 bytes): final-v-r9-ld with one more step on 14,587 EDDF rows
(`release/dh_stage.py`; [REPRODUCE.md](REPRODUCE.md#release-target-final-v-dh-r9-ld)).
It is the default target of `release/ENTRIES.json`. `final-v-dh-r9` (SHA-256
`5ce37d698943fed7a1f512aacf9500140ea7ee0809592670fe3a73ec2039fb7e`),
`final-v-dh` (SHA-256
`833533adfc04e76dbd3f300b5c69945ce305a0111a858b5a08a67ad85f880cfa`),
`final-iii-dh-r9-ld` (SHA-256
`07d4a4e6344f5c681761a00a851d23dd9dd6bc8675861179b0de80b4fe669865`),
`final-iii-dh-r9` (SHA-256
`4ac66707333eab1b32c43db0f412907e05d35e268d6a869065dba3fdc99b8139`),
`final-iii-dh` (SHA-256
`c45116502cad96dbb270467b9f2ae26fb6e06d9a50ca34ff214253609f44612d`),
`final-ii-dh` (SHA-256
`bf47551e420f781527901f9178d4d5c6e3f1089943c3c5c38baa4ced81c49c19`) and
`final-dh` (SHA-256
`2fe6c56b2705b0731be2b093f30d2763bccfb0b6ae9600855767cb0d12fef700`) are the
same step on final-v-r9, final-v, final-iii-r9-ld, final-iii-r9, final-iii, final-ii and V. The V files of
final-v, final-iii, final-ii and V are equal on the 14,587 rows; final-v-r9, final-v-r9-ld, final-iii-r9 and
final-iii-r9-ld are equal on them and differ from the other four on 14,160 of the rows.
The value is max(60, V + 0.164 (DH − V)) on the scoped
rows and V elsewhere, where V is the file of the base entry. DH is a reduced clock head: a LightGBM bag of three seeds
on the 2025 clock rows without January and July, with no adsb.lol input. Its
values, the scope flags and the weights are saved in
`release/saved-vectors/final-entries/dh/`. The scope is the typed EDDF rows
without a clock-head value whose adsb.lol inputs are absent on an airport-day
where less than 20 % of the departures have an observed taxi time. The weight
0.164 comes from held January and July 2025 rows made dark.

## Gaps

1. Route code version. The route member was fit with a later version of the
   level-2 stacker than the one in `experiments/exp-148-short-route/code/`
   (the v40 route). The held and the 2026 route fits used two different
   versions. Nine exp-147 and exp-148 files differ.
   The versions that ran ship as refit sources: `refit-src/` holds those of
   the 2026 fits, and the units `held-r4f` and `held-r4mks` of `refit-pins/`
   hold those of the held fits
   ([REPRODUCE.md](REPRODUCE.md#refit-the-final-file-from-source)).
2. The build of the v32 route component is not in the export. The other
   producers of this gap (the OPDI extractor, the input builders of the g
   member, the harness version that built the MLP inputs) ship as refit
   sources (`refit-src/`, `refit-pins/`;
   [REPRODUCE.md](REPRODUCE.md#refit-the-final-file-from-source)).
3. Member inputs are not in the export: the feature tables and input arrays
   of the members were built on the training host from challenge data and the
   sources in [DATA.md](DATA.md). The code is here; the arrays are not.
4. One pinned source was not found in version history: one version of
   `experiments/exp-159-clock-head/code/common.py`. The MLP held-input builder
   (a version of `experiments/exp-164-double-down/code/mlp007/prepare.py`) was
   recovered from a later version with its pinned SHA-256; it ships in
   `refit-pins/mlp-held-source/` with its derivation. The clock-head fit records
   hold no hash of `frame.py` and `track.py`.
5. Compute was not recorded for the no-plan and practice fits.
6. Development checks, reads and transfer scripts of exp-169 are not in the
   export. The files they copied are pinned by SHA-256 in
   `release/ZABD-PACK-ranking.json` (`source_sha256` of each frame).
7. The Gateway per-pack licence grants and the mixed-table obligations are
   not reviewed ([GEOMETRY-DATA.md](GEOMETRY-DATA.md)); OPDI names no licence
   ([DATA.md](DATA.md#opdi)).
8. The s7 member fits ran on a GPU and do not repeat bit for bit. The export
   has their values (`release/saved-vectors/s7c/`) and the fit commands
   (header of `w2/rebuild.sh`). `w2/seq/export.py` reads
   `work/w2/chain_ids_{split}.parquet`, the `MVT_ID_mvt` column of
   `work/chain/{split}.parquet`. `chain_ids.py` writes that file and asserts
   its order against the official files; in final mode `release/v73_stage.py`
   runs it after the chain dump (`chain_ids.py . --official fold,ranking`).
9. The exact v73 bytes need x86-64 Linux with the versions of
   `w2/requirements-remote.txt`; other platforms give other L2C trees.

## Imported source text

The exported exp-159 to exp-181 code, and the `iso/` and `w2/` code of v73, is
pinned to its source revision in `release/IMPORTED-SOURCES.json`. Changed lines are listed with their new text
in `release/REWRITE-LINES.json`: personal paths became `/path/to/...`
placeholders, movement IDs and one callsign in assertions became SHA-256
digests, record fields named after an account role became constants,
internal labels left the comments, one builder no longer pins a private note,
and two review file names became file patterns. The replay gives the same
bytes with the changed text.
