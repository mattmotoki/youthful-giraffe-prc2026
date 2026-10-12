#!/usr/bin/env bash
# Corrected tree budgets. Finding 13: on the SAME fold and the SAME early-stop
# month, CatBoost's best iteration is 405 at patience 100 and 4,879 at patience
# 900 -- the frozen counts every arm has been using are truncated up to twelvefold.
# Every selection from here uses a patience that is a real fraction of the
# learner's own convergence length, and the eleven-month folds are refitted.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
# CatBoost eleven-month count is measured: 5325 (select-S-cat-d8-pat900.json).
$P run.py --kind cat --name C2-d8p --depth 8 --folds july_loo,december,january \
    --rounds-m6 3625 --rounds-m11 5325
# XGBoost: re-select at the same patience, then refit the same three folds.
$P run.py --select --kind xgb --name S-xgb-d8-pat900 --depth 8 --folds december \
    --es-rounds 900 --max-rounds 6000 --seed 7
XS=$($P rounds_of.py S-xgb-d8-pat900 | cut -d' ' -f2); echo "xgb m11 at patience 900 = $XS"
$P run.py --kind xgb --name X2-d8p --depth 8 --folds july_loo,december,january \
    --rounds-m6 75 --rounds-m11 "$XS"
