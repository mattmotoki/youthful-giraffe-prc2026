# exp-130 step 1 — ranking-exposure census

**Status:** step 1 census, with the desk-requested clean view. No model was fitted for this census.

## Question and source rows

The exp-128b ledger row records a negative full-data A/B: arm A was far weaker than p12, and stand turn and geometry worsened the update. The H3 planning row in `docs/LEDGER.md` points to changed airport and month exposure as the next question. This census uses the ranking row counts for each cell instead of the validation fold mix.

The row counts come from the 2026 ranking rows whose IDs appear in `submitting.parquet`. I read only ranking input columns and submission IDs; I did not read the ranking target. The shared v25 class vectors define the current served class: `with-clock = occupied_by_exp_046`, `no-clock_ordinary = not occupied and not noac`, and `noac = missing aircraft type`. The served class matched `obs_has_clock & ~noac` for every row, and the `noac` mask matched missing `AIRCRAFT_TYPE_mvt` for every row.

January 2026 maps to the January 2025 fold. July 2026 maps to the July leave-one-out fold. On each 2025 fold, the code rebuilds v22 residuals as the exp-047 census does: `r12 - d-seasonal` on ordinary with-clock rows, and `r12` on the complement. Both rebuilt full-fold RMSE values match `census-v22.json` within 1e-6 seconds. No model was trained or refit.

The full cell table is in [`census.csv`](census.csv). It has one row for every airport × ranking month × class cell and reports both all-row and clean-row fold errors. The clean view drops only the IDs in `prc.data.excluded_ids(fold)` from fold scoring; it does not change the 2026 ranking row counts because the exclusions are 2025 validation rows. The clean view is the step-2 scoring baseline. For no-clock ordinary cells, the CSV reports p12 error on **all ordinary rows** in the matching fold and on the clean subset, with `y ≤ 3600 s` and `y > 3600 s` split apart; separate p12 counts and errors for fold no-clock and fold with-clock ordinary rows are included too.

| Matching fold | All-row v22 RMSE (s) | exp-047 census RMSE (s) | Clean v22 RMSE (s) | Excluded rows |
|---|---:|---:|---:|---:|
| january | 327.870224 | 327.870224 | 202.437984 | 2 |
| july_loo | 271.124433 | 271.124433 | 271.124433 | 0 |

## Coverage and weighted result

The ranking has **344,841 rows**: January 152,719 and July 192,122. Of these, 196,256 are with-clock, 148,197 are ordinary no-clock, and 388 are noac. Matching fold cells support 338,005 rows in the all-row view and 338,005 in the clean view; 6,836 rows have no same-airport, same-class reference and are left unestimated. The unsupported cells are: 2026-01 LFPG with-clock (1,136 rows), 2026-01 LIRF with-clock (5,700 rows). The all-row ranking-weighted MSE is **89,041.1 s²/row** (RMSE 298.4 s); the clean-view MSE is **60,325.3 s²/row** (RMSE 245.6 s). These are supported-cell estimates, not observed 2026 scores or full-board estimates. The 2026 ranking exposures remain the same in both views; step-2 fit weights and scores use these ranking counts with clean training and validation rows.

| Class | Ranking rows | All-row MSE | Clean MSE | All-row share | Clean share |
|---|---:|---:|---:|---:|---:|
| with-clock | 196,256 | 25,082.6 | 25,082.6 | 15.8% | 23.3% |
| no-clock_ordinary | 148,197 | 140,407.4 | 74,913.0 | 69.1% | 54.4% |
| noac | 388 | 11,693,907.4 | 11,693,907.4 | 15.1% | 22.3% |

### Weighted MSE by airport and class

Each MSE below weights the matching fold cells by their 2026 ranking row counts. The MSE columns report the all-row and clean views; the shares use each view’s supported estimated squared error.

| Airport | Class | Ranking rows | All-row MSE | Clean MSE | All-row share | Clean share |
|---|---|---:|---:|---:|---:|---:|
| EDDF | with-clock | 18,576 | 13,502.6 | 13,502.6 | 0.8% | 1.2% |
| EDDM | with-clock | 24,999 | 17,353.9 | 17,353.9 | 1.4% | 2.1% |
| EGLL | with-clock | 33,927 | 48,804.8 | 48,804.8 | 5.5% | 8.1% |
| EHAM | with-clock | 35,592 | 14,172.2 | 14,172.2 | 1.7% | 2.5% |
| LEBL | with-clock | 26,734 | 36,461.8 | 36,461.8 | 3.2% | 4.8% |
| LEMD | with-clock | 27,131 | 21,280.3 | 21,280.3 | 1.9% | 2.8% |
| LFPG | with-clock | 2,346 | 50,853.0 | 50,853.0 | 0.2% | 0.3% |
| LIRF | with-clock | 5,712 | 161,572.6 | 161,572.6 | 0.0% | 0.0% |
| LSZH | with-clock | 21,239 | 13,686.1 | 13,686.1 | 1.0% | 1.4% |
| LTFM | with-clock | 0 | — | — | — | — |
| EDDF | no-clock_ordinary | 17,741 | 44,267.1 | 44,267.1 | 2.6% | 3.9% |
| EDDM | no-clock_ordinary | 967 | 27,583.1 | 27,583.1 | 0.1% | 0.1% |
| EGLL | no-clock_ordinary | 5,913 | 53,787.3 | 53,787.3 | 1.1% | 1.6% |
| EHAM | no-clock_ordinary | 2,590 | 156,396.2 | 156,396.2 | 1.3% | 2.0% |
| LEBL | no-clock_ordinary | 3,346 | 44,807.1 | 44,807.1 | 0.5% | 0.7% |
| LEMD | no-clock_ordinary | 9,823 | 30,818.8 | 30,818.8 | 1.0% | 1.5% |
| LFPG | no-clock_ordinary | 37,526 | 319,845.8 | 61,196.6 | 39.9% | 11.3% |
| LIRF | no-clock_ordinary | 20,804 | 180,635.5 | 180,635.5 | 12.5% | 18.4% |
| LSZH | no-clock_ordinary | 1,910 | 116,643.7 | 116,643.7 | 0.7% | 1.1% |
| LTFM | no-clock_ordinary | 47,577 | 59,627.6 | 59,627.6 | 9.4% | 13.9% |
| EDDF | noac | 0 | — | — | — | — |
| EDDM | noac | 0 | — | — | — | — |
| EGLL | noac | 0 | — | — | — | — |
| EHAM | noac | 0 | — | — | — | — |
| LEBL | noac | 0 | — | — | — | — |
| LEMD | noac | 0 | — | — | — | — |
| LFPG | noac | 0 | — | — | — | — |
| LIRF | noac | 383 | 11,787,938.6 | 11,787,938.6 | 15.0% | 22.1% |
| LSZH | noac | 1 | 16,518,517.3 | 16,518,517.3 | 0.1% | 0.1% |
| LTFM | noac | 4 | 1,484,267.7 | 1,484,267.7 | 0.0% | 0.0% |

### Ten cells with the largest all-row estimated error share

| Ranking month | Airport | Class | Ranking rows | Matching-fold MSE (s²/row) | Share of all-row estimated SSE |
|---|---|---|---:|---:|---:|
| 2026-01 | LFPG | no-clock_ordinary | 16,993 | 634,813.4 | 35.8% |
| 2026-07 | LIRF | no-clock_ordinary | 15,550 | 215,618.9 | 11.1% |
| 2026-07 | LIRF | noac | 276 | 11,664,508.4 | 10.7% |
| 2026-07 | LTFM | no-clock_ordinary | 25,181 | 70,700.3 | 5.9% |
| 2026-01 | LIRF | noac | 107 | 12,106,319.4 | 4.3% |
| 2026-07 | LFPG | no-clock_ordinary | 20,533 | 59,180.3 | 4.0% |
| 2026-07 | EGLL | with-clock | 19,810 | 60,029.5 | 4.0% |
| 2026-01 | LTFM | no-clock_ordinary | 22,396 | 47,178.0 | 3.5% |
| 2026-07 | LEBL | with-clock | 15,928 | 54,483.8 | 2.9% |
| 2026-07 | EDDF | no-clock_ordinary | 11,147 | 51,485.5 | 1.9% |

### Ten cells with the largest clean-view estimated error share

| Ranking month | Airport | Class | Ranking rows | Clean matching-fold MSE (s²/row) | Share of clean estimated SSE |
|---|---|---|---:|---:|---:|
| 2026-07 | LIRF | no-clock_ordinary | 15,550 | 215,618.9 | 16.4% |
| 2026-07 | LIRF | noac | 276 | 11,664,508.4 | 15.8% |
| 2026-07 | LTFM | no-clock_ordinary | 25,181 | 70,700.3 | 8.7% |
| 2026-01 | LIRF | noac | 107 | 12,106,319.4 | 6.4% |
| 2026-07 | LFPG | no-clock_ordinary | 20,533 | 59,180.3 | 6.0% |
| 2026-07 | EGLL | with-clock | 19,810 | 60,029.5 | 5.8% |
| 2026-01 | LFPG | no-clock_ordinary | 16,993 | 63,632.9 | 5.3% |
| 2026-01 | LTFM | no-clock_ordinary | 22,396 | 47,178.0 | 5.2% |
| 2026-07 | LEBL | with-clock | 15,928 | 54,483.8 | 4.3% |
| 2026-07 | EDDF | no-clock_ordinary | 11,147 | 51,485.5 | 2.8% |

## p12 check on ordinary no-clock cells

The LIRF July no-clock ranking cell has 15,550 ranking rows. On the matching 2025 fold, p12 was measured on 14,874 ordinary rows: RMSE 426.9 s overall, 351.1 s on 14,603 body rows, and 1833.8 s on 271 tail rows. On AOBT-present body rows it has RMSE 351.1 s across 14,603 fold rows. Its clean values are identical because the July fold has no excluded rows. LFPG January ordinary no-clock has 2 excluded rows: all-row v22 MSE is 634,813.4 s² and clean v22 MSE is 63,632.9 s². The CSV gives p12 metrics for every no-clock airport-month cell in both views.

## Limits

This is a label-free ranking exposure census and a 2025 fold error projection. It does not measure 2026 labels or prove that the exposure-weighted MSE will occur on the hidden ranking target. The all-row view retains every fold row; the clean view drops exactly the IDs returned by `prc.data.excluded_ids(fold)`. For January this removes the two exp-020 LFPG `UNREACHED_ROLLOVER` rows; July has no excluded rows. The saved exp-047 fold census has no row-level vector for its served arm-N adjustment in three no-clock cells; both views follow its published fold construction and use p12 on the complement. The separate arm-N fold effect is not included in the cell MSEs. 6,836 rows have no same-airport, same-class reference fold and are not imputed in either view. Weather inputs are untested; this census did not use them.

## Input hashes

The census script is `code/census.py`. The direct input hashes below pin the ranking files, class vectors, fold blocks, fold labels, v22 vector, and fold-census reference.

- `artifacts/exp-046/class/2026-01-v25.parquet` — `e69f78b4f7d17fa3c0c95151fc171f2c7cd9022efd79489add1bbce119494697`
- `artifacts/exp-046/class/2026-07-v25.parquet` — `4eee477d219d32fa2ebb31346dbb105af0736e37ceaf786ec0eb0d0906be13fe`
- `artifacts/exp-047/block/2025-01.parquet` — `eb588315471b084656fba8b20e1e411e03560203beb5d63debd5a4959ca62a3f`
- `artifacts/exp-047/block/2025-07.parquet` — `00e7f9df5c83b0862a11c60defff02077c5ef68dffc8d17d50a9c17ebbc38a77`
- `data/raw/ranking.parquet` — `9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86`
- `data/raw/submitting.parquet` — `0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b`
- `data/raw/training_2025-01-01_2025-02-01.parquet` — `a4dc8f152c082c650cd7eb745fe9c641089f34f74df127e6944a188990a0ed94`
- `data/raw/training_2025-07-01_2025-08-01.parquet` — `34d1c5bba7c8596a1c20c09edd3cb2f63277b5a684fa801911660fe0eaf63c9d`
- `experiments/exp-046-surfacefit/work/d-seasonal-bag10-255-free.npz` — `3edfa35fee55e1d03ec2df72192eeb596732e62f8164b162e9ef42597ecea6f5`
- `experiments/exp-047-queue/analysis/census-v22.json` — `2951bc7fabaf815db72e125389f6d0d68e7c3346b01d7e64ab9149e50e1761d3`
- `experiments/exp-047-queue/code/base.py` — `4bf01384d27d603719e1165286ffdda0fab9079cc071859af205a2c0ea29209f`
- `prc/_private_clean_score_rows.py` — `f99e2c722d305013e553ed5fd1263525dc8f73b99c7ab1b69acc9226569bd7b4`
- `prc/data.py` — `471218c5ac3cec804efd955e6cfaf19bdcce14a61cff6bea9622c30e7f9bd1af`
- `experiments/exp-130-ranking-exposure/code/census.py` — `8174a19064d3ece22cfb81db0129d6ac10f1b3b36f1eec190d62b11d90297e40`
