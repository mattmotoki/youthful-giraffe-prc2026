#!/usr/bin/env bash
# Kill test before committing 9 fits: does the Huber arm's corrected budget help?
# exp-008 shipped L2-hub at 2,100 rounds from a patience-100 early stop;
# select-S-lgb-hub-pat900.json reads 7,990 -> 8,725 on the same fold and month.
# One fold, one seed, both budgets already measured at 2,100 (july_loo 310.84).
set -u
cd "$(dirname "$0")/../exp-008-diversity"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=0
$P run.py --kind lgb --name E12-hub8725 --variant plain --huber-delta 1200 \
    --folds july_loo --rounds-m6 3225 --rounds-m11 8725 \
    --outdir ../exp-012-blend2/oof
