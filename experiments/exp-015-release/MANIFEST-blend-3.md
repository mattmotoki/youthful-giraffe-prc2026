# MANIFEST — exp-012-blend-3

Produced by `scripts/build_candidate.py experiments/exp-015-release/config/blend-3.json` at 2026-09-05T22:58:01.

Regenerate with the same command; `--mode verify` re-hashes every declared
artifact without fitting anything.

## Environment

| item | value |
|---|---|
| python | 3.10.12 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| xgboost | 3.2.0 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| pyarrow | 24.0.0 |
| scipy | 1.15.3 |
| sklearn | 1.7.2 |
| prc.blend | 5 |
| prc.data | 5 |
| prc.nmtail | 2 |
| prc.tail_model | 2.4.0 |
| prc.day_state_features | 1 |
| prc.phi_features | 3 |
| prc.lirf_features | 2 |
| prc.surface_features | 3 |
| prc.stand_features | 1.2.0 |
| prc.weather_features | 2 |
| prc.tail_features | 1.3.0 |
| git commit | bc70f5c890c8fd2244753a0044646f7269c6f9a1 |
| uncommitted paths | 44 |

## Raw inputs

| file | bytes | sha256 |
|---|---|---|
| data/raw/ranking.parquet | 43564605 | 9db75e3a56fa09b87aab72e191a53da3… |
| data/raw/submitting.parquet | 1677097 | 0d383408ed5ac27bfcfd12d6b961cee7… |
| data/raw/training_2025-01-01_2025-02-01.parquet | 21218481 | a4dc8f152c082c650cd7eb745fe9c641… |
| data/raw/training_2025-02-01_2025-03-01.parquet | 19824510 | b6f9ff2cdb1cfe07c5442f4e1f11bbcd… |
| data/raw/training_2025-03-01_2025-04-01.parquet | 22606072 | 58fde7e5f8b539546af63b41d7f6ec0f… |
| data/raw/training_2025-04-01_2025-05-01.parquet | 23840340 | 9ffe0cf5741c4863799c41be81475dfa… |
| data/raw/training_2025-05-01_2025-06-01.parquet | 25187756 | 5481c0642a8fbb4dc395f93bc713c57f… |
| data/raw/training_2025-06-01_2025-07-01.parquet | 24991697 | c3ef5080f2cc636c3416baf1555f7d0e… |
| data/raw/training_2025-07-01_2025-08-01.parquet | 26014977 | 34d1c5bba7c8596a1c20c09edd3cb2f6… |
| data/raw/training_2025-08-01_2025-09-01.parquet | 25969246 | 6c8d2aa7a1b6c9dff9a0305a37eb02de… |
| data/raw/training_2025-09-01_2025-10-01.parquet | 25102673 | 955308a4000947e32a00586d3ee13928… |
| data/raw/training_2025-10-01_2025-11-01.parquet | 25392098 | 4a31a35601c1bdcd74b8cd6379231ba5… |
| data/raw/training_2025-11-01_2025-12-01.parquet | 22173840 | ac36b17db7a537e3baeb682fa4763dac… |
| data/raw/training_2025-12-01_2026-01-01.parquet | 22653598 | 2d357f2f403a86778a69e93f72d8754e… |

## Reproducibility gate

| stage | byte-identical | detail |
|---|---|---|
| blend (two independent derivations) | True | f91db30a7f3bc4da… both runs |
| member T-tailv2 | declared reproducible | noac head pinned to v2 by NOAC_PIN in the producer. The OOF is exp-009's versioned v2 file, byte-identical to what these weights were fitted on. |
| member C2-cat | declared reproducible | The only GPU member. GPU 1 per docs/COMPUTE.md compute allocation, which counts 2 feeding CPU threads against this lane's cap. |
| member L2-hub | declared reproducible |  |
| member D1-winter | NO — declared non-deterministic | linear_tree=True, which LightGBM's deterministic=True does not cover across thread counts. Measured: the same seed at the same thread count drew 148 rounds on one run and 186 on an |
| member N1-nmtail | declared reproducible | Predicts NM-joined departures only; y_pred is null on 5,290 non-NM ranking rows. prc.blend VERSION 5 permits that ONLY because this member's weight outside nm_late is exactly 0.000 |

The gate passes for every stage this pipeline controls. It cannot pass for a member whose estimator is not thread-count reproducible, and that member is named above with its measured spread rather than left to be discovered as a mismatched hash.
