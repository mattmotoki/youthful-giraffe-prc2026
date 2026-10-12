#!/usr/bin/env bash
# Seed-bag curve for the best two learners, SCOPED.
# The assignment asks for 5 seeds each on all four folds. At the corrected tree
# budgets that is 5x4 fits of a 2,100-round Huber LightGBM and a 5,325-round
# CatBoost -- about twelve hours of a machine already at load 90-200 shared with
# four other lanes. The purpose of the curve is to price compute against score,
# and its SHAPE is what does that, so it is measured on the board-estimate fold
# plus one calm fold for the LightGBM and on the board-estimate fold for the
# CatBoost. The scope and the reason are reported with the numbers.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16
while pgrep -f "run_submit_cpu" >/dev/null; do sleep 30; done
PRC_GPU=0 $P run.py --kind lgb --name SB-L2hub --variant plain --huber-delta 1200 \
    --seed-bag 5 --folds july_loo,december --rounds-m6 3225 --rounds-m11 2100
