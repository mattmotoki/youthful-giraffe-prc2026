# R4 split contract for extra L members (for grind; valid, 2026-10-01 01:58 UTC)

SPEC-r4 item M (branch `exp148-short-route`, `SPEC-r4.md` sha256 02e7d6b5…). The same contract as
R2's four jjx fits (1b676fc0 SPEC-r2), for three more months. An L member enters R4 with these
three fits in addition to its six R2 artifacts (`jjx-12`, `jjx-02`, `jjx-06`, `jjx-08`, `janjul`,
`final`), which stay as they are.

## Splits

Defined by `pilot.jjx` in `experiments/exp-147-foldserve/code/pilot.py` (commit 6f3f6985 or later
on `exp148-short-route`): `pilot.split("jjx-03")` returns the exact `WP.Split`. Each excludes every
day of 2025-01, 2025-07 and its own month with their month-hold buffer days, and holds its month.

| split | held days | excluded days | token |
|---|---|---|---|
| `jjx-03` | 2025-03-01 … 2025-03-31 (31) | 99 | `74be48796d29ae83` |
| `jjx-10` | 2025-10-01 … 2025-10-31 (31) | 99 | `b0d549b7b3d4a662` |
| `jjx-11` | 2025-11-01 … 2025-11-30 (30) | 98 | `10e173cdad96e9d8` |

The owner is the split name (`jjx-03` …), as in R2. `pilot.consumer("q01")` and `consumer("q07")` admit
all three (checked); `qfinal` reads them as training splits.

## Artifacts

As the R2 pack: `<Stage>-<split>.parquet` with columns `MVT_ID_mvt`, `month`, `utc_day`,
`airport`, `pred_<Stage>`, and its `.json` record (`owner`, `split`, `output_sha256`, …) that
`pilot.load(path, owner)` accepts. Exact scored IDs: equal to my L fits of the same split, which
are on the workstation (read only):

| split | my L reference | rows | output sha256 |
|---|---|---|---|
| `jjx-03` | `prc-exp148-work/kaggle/r4-la/out/L-jjx-03.parquet` | 164,389 | `6ceb81bb…0606` |
| `jjx-10` | `prc-exp148-work/kaggle/r4-la/out/L-jjx-10.parquet` | 185,556 | `294219f7…a8b7` |
| `jjx-11` | `prc-exp148-work/kaggle/r4-lb/out/L-jjx-11.parquet` | 162,277 | `39535747…51d6` |

(Paths are relative to the competition root.) Training rows of my L: 1,548,232 (`jjx-03`),
1,526,552 (`jjx-10`), 1,549,884 (`jjx-11`). The 2025-02-28 23:59:57 departure that sits
in the March raw file is not in my `L-jjx-03`; leave it out too (the R4 stacker drops it in any
case).

## Learner inputs

The L learner reads the proxy features of the scored month: `learner_wp.py --fit features-r4`
exports months 2025-01, 03, 10, 11 (`FEATURE_MONTHS_R4`); my export is
`prc-exp148-work/kaggle/r4-la/out/proxy-features-r4.parquet` (sha256 `712dc1f0…5b3d`), from the
Kaggle kernel group `r4` (`kaggle/build_learner_kernels.py --r4`, `kaggle/learner-code-r4.json`).
Use the same 12 raw month files as R2 (their hashes are in each L record's `inputs`).

## Delivery

A pack directory with the three splits of each member (Lcontrol, Lbag4, Lweatherbag, then
Lgeometry if done), a `PACK.json`, and a committed hash list. I copy it read-only into my work
root, check every hash, and add the members to R4 as extra members (as `r2Lc`/`r2Lv` on R2).
