# exp-148 SPEC revision 2: a level 2 without labels of the scored month (repair 2)

Amends SPEC.md (sha256 f0e19c6a…) and SPEC-r1.md (0a8024c2…); both stay in force for the
candidate Sb. Written 2026-09-30 about 14:00 UTC, after REPORT.md (commit f0f69af4) and before
any R2 member fit or R2 stacker fit. Assignment: the desk's message of 2026-09-30 13:1x UTC
(repair 2 of REPORT.md): month-hold member fits for December, February, June and August (8
members x 4), then a level-2 fit for each scored month trained only on other months, never on
its own month. Lanes: cores 32-55/96-119 and GPU 1 (56-63/120-127 stay lent to exp-150).

## Reading

The board's level 2 is fitted on 2025 and serves January and July 2026, months without any
label. In the week-pair owners the level 2 was fitted on other weeks of the scored month; in
the whole-month check (SPEC-r1 item 3) on one month of the other season. R2 fits the level 2
of each scored month on other months of both seasons and never on the scored month, and the
served level 2 on all six months. It is judged in the same place as the month check: January
and July 2025, members that saw no day of January or July.

Readings rejected:
(a) Fits that hold out only month X (the literal "month-hold"). They read January and July
    labels, so `weekpair.month_hold("2025-01").assert_admits` refuses them as inputs of the
    January level 2 (LabelLeak; checked on the code before this file). They could feed only the
    served level 2, so the local check would test a different procedure.
(b) Pairwise fits that exclude {January, X} and {July, X} separately: members one month
    stronger for the same check, 64 fits instead of 32.
(c) Keep the four `v23_*` columns with NaN on February, June and August (exp-127's fold frames
    exist for January, July and December only): the stacker would learn a missingness pattern
    that neither the scored months nor 2026 have. They carried 2.5 % of the final LightGBM gain.
(d) Keep `ctx_month`: the served level 2 would read 2025 labels of the served calendar month
    through it, which no local owner can read, so the check would not test the served model.

## 1. Member fits `jjx-MM` (`pilot.jjx`, `pilot.FITS_R2`)

For MM in 12, 02, 06, 08: excluded days = every day of 2025-01, 2025-07 and 2025-MM and the
month-hold buffer day on each side of the three months; held (scored) days = 2025-MM. One fit per
member and month (8 x 4 = 32), with the same drivers, settings and write-once artifacts as every
other fit (`code/lane.sh`; L and Lp through `code/learner_lane.sh` with the kernel
`kaggle/learners-r2/kernel.py`, whose code blob is in `kaggle/learner-code-r2.json`). The only
code changes: `pilot.py` adds the split, its owner and the consumers r01, r07, rfinal;
`learner_wp.py` scores the months of the split's held days (January and July for every earlier
fit, as before) and exports the proxies of the new months (`--fit features-r2`). The `janjul`
fits (SPEC-r1 item 3) give January and July.

## 2. Level-2 rows (owners)

| owner | consumer | training rows (member fits) | scored rows (member fits) |
|---|---|---|---|
| r01 | month_hold(2025-01) | every 2025 departure of December, February, June, August (their jjx fits) and July (janjul) | January 2025 (janjul) |
| r07 | month_hold(2025-07) | the same four months (jjx) and January (janjul) | July 2025 (janjul) |
| rfinal | final | all six months (jjx for the four, janjul for January and July) | the 344,841 ranking departures (the members' `final` fits) |

Every member artifact passes `pilot.load` (split, owner, hash) and `consumer.assert_admits`;
the training frame passes `assert_training_frame` with the target. Stage name `R2` (artifacts
`R2-month-2025-01`, `R2-month-2025-07`, `R2-final`).

## 3. Model

SPEC section 3 and SPEC-r1 items 1-2 unchanged (anchor = mean of the five base members, target
y - a, the LightGBM and CatBoost parameters, the forms S, Scat and Sb, floor 60 s), except:

1. Features: SPEC's full list less the four `v23_*` columns and less `ctx_month` (reasons (c),
   (d) above). Feature list hash in every record.
2. CV groups for the rounds: the calendar month of the training rows (leave one training month
   out; five groups for r01 and r07, six for rfinal).
3. Labels: the raw 2025 month files (`TAXITIME_SEC_mvt` of the departures), read for the
   training rows only, after the isolation check.
4. Comparator stacker R2-22 for r01 and r07: the same rows and model on v22's inputs (SPEC
   section 4 (2)), less `ctx_month`.

**Form** (fixed now): SPEC-r1 item 1's rule on r01 and r07: the form with the lowest `cv_pooled`
MSE averaged over the two owners.

## 4. Features of the new months (`code/features_r2.py`, label-free)

For every departure of 2025-12, 2025-02, 2025-06 and 2025-08: the class by exp-146's rule
(`build_frozen.rows_2025`: exp-046's ungated block, the mode-4 gate, the exp-047 cross-check;
December must equal the frozen class of `frozen/rows.parquet` on every row), `ctx_*`, `obs_*`
(the month's ungated block through `chainfold.restrict` with the served gate), `shp_*` and
`q_*`, exactly as `features.py` builds them for January and July. Proxies and row clocks:
`learner_wp.py --fit features-r2`, which also re-exports January; the January rows must equal
the committed `proxy-features.parquet` (max difference 0). The R2 stacker reads the committed
tables and the two new ones, concatenated.

## 5. Scores (January and July joined, 344,419 rows; frozen G, interval and views)

The chosen R2 form against: the janjul anchor (clean); R2-22 (clean); v32's and v22's saved
routes (reference); Sb of the whole-month check and Sb of the serve read (the same janjul
members; the serve-read vector is rebuilt by `code/quick_serve.py`'s path). Month, class,
class x AOBT, airport and band views. No clock-loss view (the masks exist for the week-pair
owners only).

## 6. Ranking vector and decision use

`R2-final` of the chosen form, served as `rankvec.py` serves Sb (unsupported cells keep v32),
344,841 rows, sha256. R2 is a submission candidate if it beats v32's saved route in section 5
(G > 0, all rows); the desk decides any upload. The frozen forecast is written in the R2 report
before any upload, from these scores and the board read of Sb when it exists.
