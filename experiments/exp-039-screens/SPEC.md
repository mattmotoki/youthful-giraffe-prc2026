# exp-039-screens — the four cheap diagnostic screens Q03–Q06

Assignment: `assignments/screens.md`. Worker tier hard, reviewer Sol, CPU only, laptop,
no member fits, no uploads.

## The reading this experiment took, and the readings it rejected

`assignments/screens.md` names the measurement contract as "residual alignment against the
**v10** vector, `submissions/uploaded/youthful-giraffe_v10.parquet`". That file is the RANKING
side and carries no labels, so `r = y − p_v10` cannot be formed on it. **This experiment reads
"the v10 vector" as the candidate, not the file**: the fold side of the same candidate is
exp-036 arm 2 — refit-bulk served at the eight mask airports, v7 held at EDDF and EHAM — and the
served ranking file that arm produced is byte-identical to the uploaded v10 (SHA-256
`ad962763…`, checked in `code/v10.py` on every run). So every screen scores against
`p10 = np.where(airport in KEEP, refit-bulk, v7)` on `july_loo`, `december` and `january`, and
`code/v10.py` refuses to run unless that vector reproduces exp-036 arm 2's own published rotation
(284.1572 / 204.9218 / 218.1506, clean board 257.0248). Two readings were rejected: scoring
against the ranking file with a pseudo-label (there is none), and scoring against K2/v9, which is
the *previous* board file and not what the assignment names.

Second reading: **"all departure B/y masked before any context is built"** is
`prc.data.mask_like_ranking`, which blanks departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`
and nothing else (`docs/input-rule.md`). So B is `BLOCK_TIME_UTC_mvt`, **not** `AOBT_3_flt`:
`AOBT_3_flt` is NM's off-block estimate, survives the ranking mask and is a legal input. Every
feature in `code/rowfeat.py` and every donor field in `code/donor.py` is built on the masked
frame. `B` and `y` appear only in the labelled audit tables of Q05 stages 2, 3 and 5, which are
tabulations and never predictors. An arrival's `BLOCK_TIME_UTC_mvt` is **not** masked (the ranking
file carries it) and is used as an input in Q04 and Q05; Q06 never reads it as a departure B.

Third reading: the queue gives each screen a "First test" paragraph and a longer conditional
programme. **Only the First test is run here**, plus the repair each screen's own "Failure and
repair" paragraph names when its first test is negative. No member is refitted and no candidate
is written.

## What each screen runs

| screen | population | learner | endpoint |
|---|---|---|---|
| Q03 | `nm_bulk` outside EHAM/EDDF | fixed ridge, within-cell centred in the fitting split | zero-sum update, paired 2A−B and clean board |
| Q04 | the same, and cells with ≥6 departures and ≥6 arrivals | 64 depth-two trees, lr 0.05, seed 39 | h1 − h0 increment at matched capacity |
| Q05 | joined rows outside EHAM/EDDF; noac for the witness | shrunk group means, blocked splits | incremental residual mean beyond existing controls |
| Q06 | 388 noac and 4,895 nonm ranking rows; nonm fold rows | none (census + one anchor) | donor counts, ID agreement, squared-error price |

Predeclared before any number was read: the 10-minute cell, the four-row minimum cell, the ridge
penalty, five contiguous day blocks, the ±60 s duplicate tolerance and the [0, 12 h] arrival
window, the ≥6/≥6 cell support and the 200-cell / 20-date rule, the 20,000-row joined sample and
seed 39. They are stored in each `analysis/q0*.json` under `predeclared`.

## Reproduction

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/v10.py   # the gate
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q03.py   # ~47 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q04.py   # ~90 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q05.py   # ~5 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q06.py   # ~5 s

Needs `data/raw` and the committed rotations of exp-026/exp-027/exp-028/exp-036. Logs in
`logs/`, records in `analysis/`, one report per screen at `Q03.md` … `Q06.md`.

## Defect found and fixed during the run

`code/donor.py` first converted the parquet timestamps with `astype("int64") / 1e9`. pandas 2.x
keeps a `timestamp[us]` column at MICROSECOND resolution, so every clock was 1,000 times too
small and both donor windows were effectively unbounded. The first Q06 run reported 20.55 %
accepted-ID agreement because of it. `donor.epoch()` now reads the unit off the dtype; the
repaired run reports 99.87 %. Q03's cell binning and day blocks were never affected — they read
`t` from exp-016's out-of-fold frame, which is already epoch seconds, and the 93 fold-days the
bootstrap reports are the check on that.

## Round 1 review and the repairs it produced

`review/prompt.md` and `review/round1-gpt-5.6-sol-xhigh-2026-09-08.txt`. Sol returned
**NEEDS REVISION** with 12 blocking findings. All 12 were accepted as legitimate and repaired; the
reports in this directory are the post-repair ones and the pre-repair numbers survive only in the
original ledger rows, which are corrected by a second set of rows.

| # | finding | repair | effect on the verdict |
|---|---|---|---|
| 1 | the v10 fold gate compared RMSE only, so a moved vector could pass | `code/v10.py` pins a SHA-256 over ordered (`MVT_ID_mvt`, `p10`) float64 bytes per fold | none |
| 2 | the measurement packet was incomplete and the CLEAN board was headlined | `code/packet.py`: all-row january/july board is now PRIMARY, plus per-date CSVs, file and vector hashes, versions, commit, 1/2/3-day moving blocks | headline numbers restated; signs unchanged |
| 3 | Q03's pair evidence read at the out-of-block amplitude, which has an indirect path from the scored block | the headline pair row is the CROSS-MONTH amplitude | pooled pair +179.47 → **+199.07 s²** |
| 4 | Q03's uncentered control ran on 370,338 rows against the arm's 304,001 | the control is restricted to the arm's rows; a `projection_off` ablation was added | control pooled −5.80 → **−7.93 s²**; new −2,607 s² ablation |
| 5 | Q04's `h0` was too thin, its departure counts spanned all regimes, its census included EHAM/EDDF, and the nuisance size was mis-stated | `h0` is 19 features; counts are `nm_bulk`; census excludes the two airports; nuisance is 1.89–1.93 M | h1−h0 −21.9/−43.5/−31.4 → **−30.9/+28.0/−10.4** |
| 6 | Q04's `h1c` kept the real `zprev`, and the wide window was not fixed-width | the whole innovation block is 7-day shifted on matched-cell support; the window rolls a fixed consecutive-bin grid | h1c becomes a clean placebo and BEATS h1 |
| 7 | Q04.md carried intervals from a superseded run; `check_reports.py` checked only signs | every table is generated by `code/tables.py` and injected between markers; `check_reports.py` asserts the injected block byte-for-byte and every displayed bound exactly (326 checks) | none; the record is now self-checking |
| 8 | Q05 declared 5/15/60 s phases but scored one 60 s head, with no capacity-matched control | `phase_5s`, `phase_15s`, `phase_60s` and `sham_same_cardinality` all run | all four phase heads negative |
| 9 | "the anchor absorbs the phase" was algebraically wrong | the record now carries `delta_y`, `delta_BminusAOBT`, `delta_anchor_error` and `delta_r` side by side; the claim is **v10 absorbs the phase, the raw anchor does not** (Δ raw anchor +434.7 s against Δ v10 residual +4.59 s) | the sentence went from wrong to right; the verdict is unchanged |
| 10 | Q06 conflated counterpart existence with candidate recall | four denominators kept apart | recall **7.74 % → 51.04 %** (of the 3,033 that exist) |
| 11 | reciprocal uniqueness was computed per donor ROW | it is `nunique` over the accepted `FLIGHT_ID` | acceptance unchanged; the claim is now the one made |
| 12 | Q06 labelled an all-row SSE share as clean | both shares are reported | january **0.079 % → 0.189 %** clean |

Sol's round 1 also confirmed, explicitly, that Q03 has no departure-label path at inference and its
algebra checks are real and non-degenerate; that the board weights are right; that Q04's nuisance
excludes the scoring month and its departure cross-product is correctly named an oracle; that Q05's
artifact dependency is accurate; that Q06 removes every gold-ID donor, prices errors on labelled
rows only, and that the repaired timestamp conversion matches direct timestamp subtraction with max
error 0; and that nothing widened beyond the assignment.

## Reproduction after the repairs

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q03.py   # 49.9 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q04.py   # 132.9 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q05.py   # 6.5 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q06.py   # 6.5 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/tables.py
    .venv/bin/python experiments/exp-039-screens/tests/check_reports.py          # 326 checks

## Round 2 review and the repairs it produced

`review/prompt-round2.md` and `review/round2-gpt-5.6-sol-xhigh-2026-09-08.txt`. Sol returned
**NEEDS REVISION** with 11 blocking findings and verified 5 of the 12 round-1 repairs outright,
6 as PARTIAL and 1 as verified-in-the-effective-record. All 11 new findings were accepted and
repaired; the answers are in `review/round2-answers.md`. `MODES.md` allows at most two review
rounds, so this is the last one and the answer document stands in place of a third.

| # | finding | repair | effect on the verdict |
|---|---|---|---|
| a1 | `packet.primary()` rounded each fold RMSE before the weighted board aggregation | the board is aggregated on raw float64 RMSE, rounded only for display; `gain_exact` beside it | Q03 primary **+0.0781 → +0.0782 s** |
| a2 | the pooled moving-block bootstrap drew contiguous runs from one 93-label sequence, so blocks could straddle month boundaries | `packet.boot_blocks_pooled` draws inside each fold and pools numerators and counts | Q03 pooled 3-day [+117.2, +274.0] → **[+133.1, +278.4]** |
| a3 | "only the zero-sum constraint differs" overstated the uncentered control | the uncentered arm is described as matched-support, matched-budget; `projection_off` is the one-term ablation, with its scope stated | none |
| b1 | Q04's `h0` was matched learner capacity but not the queue's complete current block | `h0` carries the shipped 89-column block from `exp-018-fresh` (108 features against 19) | h1−h0 −30.94/+27.96/−10.39 → **−45.65/+5.94/+0.05** |
| b2 | `h1c` was not a matched-support placebo, and the shift direction was mis-stated | both shifted columns gated on the shifted cell's support; all arms scored again on the intersection; the shift is **+7 days later** | the round-1 "the placebo beats the real thing" is **withdrawn**; h1 and h1c are not separable |
| b3 | the MAYBE's revival test was not operational | it names its two dependencies: a v10 replay on the new month, and a nuisance refit excluding it | none |
| c1 | "+0.0014 s" was called a phase-only increment | it is a **full-fingerprint contrast**; the negative is limited to the tested global heads | none |
| c2 | the report closed more of the family than it tested | the negative is stated for the global shrunk heads run here, with a revival condition | none |
| e1 | provenance incomplete: truncated hashes, no scikit-learn, no code state, no row contributions, no Q06 multi-day | full 64-character hashes, `scikit-learn` and `pyarrow` versions, a git-dirty flag and a SHA-256 per code file; `packet.rows()` writes per-row contribution parquets addressed by `analysis/rows/MANIFEST.json`; Q06 gains 1/2/3-day intervals | none |
| e2 | the stale-report guarantee was not real: JSON, saved tables and report could go stale together, and only chosen prose numbers were asserted | `tests/check_reports.py` re-renders every table **from the current JSON** and requires the saved file and the injected block to match it, and audits **every** number in the prose against the record with a documented list of non-measurements | none |
| e3 | the round-1 rewrite of Q05 dropped the queue-required feature inventory and the unmet-artifact dependency | both restored as §1 and §6, and asserted by the checker | the witness arm is **blocked**, not closed |

One more defect was found in this round's own work and is recorded here rather than hidden: q06's
predeclared 20,000-row sample drew from the shared generator, so adding the multi-day bootstrap
moved it. It now draws from its own seeded generator. The re-draw showed that the gold-removed
**price** on 1 to 5 rows is unstable across draws while the **rate** (0.01 %) is not, so Q06 now
quotes the rate.

## Reproduction after the round-2 repairs

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q03.py   # 63.4 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q04.py   # 848 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q05.py   # 6.1 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/q06.py   # 7.5 s
    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-039-screens/code/tables.py
    .venv/bin/python experiments/exp-039-screens/tests/check_reports.py          # 102 checks

Q04 now reads `artifacts/exp-018-fresh/cache/<month>.parquet`, a build cache that is not committed;
its SHA-256 per fold is in `analysis/q04.json` under `folds.<fold>.W_cache_sha256`, and the
committed code that builds it is `experiments/exp-018-fresh/code/features.py`.
