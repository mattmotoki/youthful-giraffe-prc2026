#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=12 PRC_GPU=0
$P run.py --kind lgb --name SB-L2hub --variant plain --huber-delta 1200 --seed-bag 5 \
    --folds july_loo --rounds-m6 3225 --rounds-m11 2100
$P run.py --kind lgb --name SB-L0lin --variant lin --seed-bag 5 \
    --folds july_loo,december --rounds-m6 175 --rounds-m11 350
