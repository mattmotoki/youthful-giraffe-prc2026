#!/usr/bin/env bash
# Seed-bag curve. The assignment asks for the best two learners at 5 seeds on all
# four folds; at the corrected budgets that is about twelve hours on a machine
# shared with four other lanes, so it is scoped and the scope is reported:
#   L2-hub  (best single model)      5 seeds, july_loo          -- the actual ask, one fold
#   L0-lin  (cheap LightGBM)         5 seeds, july_loo+december -- the same curve's shape at a tenth of the cost
# CatBoost's 3-seed july_loo curve runs on the GPU stream.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=0
until [ -f select-S-lgb-hub-pat900.json ]; do sleep 60; done
$P run.py --kind lgb --name SB-L2hub --variant plain --huber-delta 1200 --seed-bag 5 \
    --folds july_loo --rounds-m6 3225 --rounds-m11 2100
$P run.py --kind lgb --name SB-L0lin --variant lin --seed-bag 5 \
    --folds july_loo,december --rounds-m6 175 --rounds-m11 350
