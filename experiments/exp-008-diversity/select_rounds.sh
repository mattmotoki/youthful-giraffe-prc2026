#!/usr/bin/env bash
# One-time round choice per learner per fold family. Early stop is the LAST
# TRAINING MONTH of the fold (2025-06 for the six-month family, 2025-11 for the
# eleven-month family); no validation month is touched. The chosen count is
# then frozen and used for every fold and every seed of that arm.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16
stream=$1
if [ "$stream" = cpu ]; then
  PRC_GPU=0 $P run.py --select --kind lgb --name S-lgb-lin   --variant lin   --folds july,december --max-rounds 1500 --seed 7
  PRC_GPU=0 $P run.py --select --kind lgb --name S-lgb-mono  --variant plain --monotone --target raw --folds july,december --max-rounds 3000 --seed 7
  PRC_GPU=0 $P run.py --select --kind lgb --name S-lgb-hub   --variant plain --huber-delta 1200 --folds july,december --max-rounds 3000 --seed 7
else
  PRC_GPU=1 $P run.py --select --kind cat --name S-cat-d8  --depth 8  --folds july,december --max-rounds 4000 --seed 7
  PRC_GPU=1 $P run.py --select --kind cat --name S-cat-d10 --depth 10 --folds july,december --max-rounds 4000 --seed 7
  PRC_GPU=1 $P run.py --select --kind xgb --name S-xgb-d8  --depth 8  --folds july,december --max-rounds 4000 --seed 7
fi
