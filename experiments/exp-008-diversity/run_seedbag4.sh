#!/usr/bin/env bash
# Seed-bag curve, third attempt. The first two were killed under machine-wide
# memory pressure partway through, losing everything because run.py writes the
# OOF only after all seeds finish. This runs the cheap arm first so the curve
# exists, then the expensive one, and each --folds/--seed-bag combination is its
# own process so a kill costs at most one arm.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=8 PRC_GPU=0
$P run.py --kind lgb --name SB-L0lin --variant lin --seed-bag 5 \
    --folds july_loo --rounds-m6 175 --rounds-m11 350
$P run.py --kind lgb --name SB-L0lin-dec --variant lin --seed-bag 5 \
    --folds december --rounds-m6 175 --rounds-m11 350
$P run.py --kind lgb --name SB-L2hub --variant plain --huber-delta 1200 --seed-bag 3 \
    --folds july_loo --rounds-m6 3225 --rounds-m11 2100
