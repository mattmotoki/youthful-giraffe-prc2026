# `D1-winter-deice-v1` — published member interface

For exp-016 and any other consumer. **Use these files; do not refit this member.**
The reason is in "Reproducibility" below and it is not a preference.

## What it is

retro-v1's `G1cv-s7` line — residual target, linear leaves, phi (cv-fitted),
lirf, noac head, residual guard, v2 features — plus exp-007's day-state step
restricted to **one column**, `prc.day_state_features.DEICE` = `["ds_deice_hr"]`.

It replaces a member that carried the full 25-column day-state block. That block
reached its fold score partly by extrapolating: on the january fold it put
**4,640 validation cells outside its training range**, this member puts **0**,
and `ds_deice_hr` is bounded in [0, 1] by construction so it cannot leave the
range at all. On january's most disrupted 5 % of airport-days the block scores
400.85 and this member **349.59**.

## Files

| what | path | sha256 |
|---|---|---|
| OOF july_loo | `published/D1-winter-deice-v1-july_loo.parquet` | `bbe71ea2eec6d47e057d0a90f5a78e4f30785f82d8f2d82a9d2bd0471aed607c` |
| OOF december | `published/D1-winter-deice-v1-december.parquet` | `bfc3497a47c51a4792d841b0cda5239ccea0a07c21cabdaf448e24ae7e9341cf` |
| OOF january | `published/D1-winter-deice-v1-january.parquet` | `cdb15e328154a10cfa269375765accd767ae835700963e470a713f74c7908dd6` |
| ranking | `artifacts/exp-012/pred/D1-winter-deice-v1-ranking.parquet` | `8865d3d866012d1421a08eee4c24ce9cfe53dbc0e255d8a3ff1894b68c368fd2` |

OOF columns: `MVT_ID_mvt`, `fold`, `y_true`, `y_pred`, `nm_missing`, `airport`,
`d_takeoff_aobt`, `y_pred_s0`. `y_pred` is floored at 60 s and residual-guarded;
`y_pred_s0` is the raw per-seed prediction before both.

## Scores

| fold | RMSE | rounds | features |
|---|---|---|---|
| july_loo | 310.28 | 142 | 97 |
| december | 223.98 | 176 | 97 |
| january | 349.20 | 221 | 97 |

Board estimate as a single member: **328.09**.

## How it was built

`experiments/exp-012-blend2/winter_arm.py --ds-columns deice`, one seed (7) for
the OOF, three (7/107/207) for the ranking predictions, **`PRC_THREADS=6`** under
`docs/COMPUTE.md` compute allocation, against exp-007's pinned pipeline snapshot
(`experiments/exp-007-winter/code/pipeline`, git 19cd199). Module versions are
recorded in `result-D1w-deice.json` and in each run's `[provenance]` line.

The ranking budget is drawn under the desk's multi-month rule, and this member
is the first to hit its **wide-spread branch**: per-month optima
**2025-07 107, 2025-01 148, 2025-11 223**, median 148, relative spread
**0.784** — above the 0.5 threshold, so the rule takes the MINIMUM rather
than the median and the member ships at **125 rounds**. The block arm's scan on
the same three months gave [109, 129, 142] at spread 0.256, so dropping 24
columns more than tripled the month-to-month disagreement about the budget.

Note the consequence: the served member uses 125 rounds while the OOF above drew
142 / 176 / 221 per fold. The weights a consumer fits on that OOF are therefore
applied to a slightly different estimator. The same mismatch on the previous
winter member measured 0.014 s on the board estimate; it is disclosed here rather
than assumed to transfer.

## Reproducibility — refit only at six threads, or use these files

**Corrected 2026-09-06 after measurement.** An earlier version of this document
said this member "cannot reproduce itself" and told consumers never to refit it.
That was measured across *different* thread counts and was too strong. Run twice
at six threads on identical data, this member is **bit-stable**
(`034fa693…` both runs, exp-015 separation).

What is true is narrower and still worth carrying: this arm sets
`linear_tree=True`, and LightGBM's `deterministic=True` is documented as
reproducible *"with the same data and the same number of threads"* — the linear
leaves do not change that, but the thread count does. The same nominal
configuration has been observed at three levels when the count moved:

| run | threads | rounds (july_loo) | july_loo RMSE |
|---|---|---|---|
| exp-007's original block arm | 24 | 148 | 307.45 |
| exp-012 rebuild, block | 24 | 186 | 311.30 |
| exp-012, block | 6 | 85 | 307.61 |

The 24-thread pair differing is the one observation that does *not* fit a pure
thread-count story, and it is left standing rather than explained away: those two
runs also straddled a change in another lane's module, which the manifest of the
day records but which I did not isolate.

**So: refit at `PRC_THREADS=6` and you should get these bytes. Refit at any other
count and you get a different member, and any weight fitted on this OOF is then
being applied to something else.** Using the published files avoids the question
entirely, which is why they are published.

## Caveats a consumer should carry

- **It is a trade, not a free win.** Against the block it is 4.91 s better on the
  january fold and **2.67 s worse on july_loo** (+13.83 s on july_loo's disrupted
  days, where the 24 dropped columns were carrying congestion signal). It was
  adopted for the January-2026 extrapolation exposure, which no 2025 fold prices,
  not for the 0.93 s it gains on the member board estimate.
- **The block-vs-deice comparison is unpaired** — separate fits, separately drawn
  round counts. The 51 s on january's disrupted days is far outside the re-draw
  noise; the 2.67 s on july_loo is not comfortably so.
- **Boundedness is not why it works.** Clipping the full block to its training
  range at predict time, which changes boundedness and nothing else, recovers
  0.72 s on the january rows it touches and nothing elsewhere. The gain comes
  from dropping 24 columns, not from bounding them. This answers the open
  question exp-007 finding 14 left.
