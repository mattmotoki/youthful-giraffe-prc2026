#!/usr/bin/env bash
# The july fold at a corrected CatBoost budget, so the C-cat member stops mixing
# a patience-100 six-month count with patience-900 eleven-month ones. Queued last
# because july is the pessimistic fold and is not part of the board estimate.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
while pgrep -f "run_gpu5" >/dev/null; do sleep 30; done
$P run.py --select --kind cat --name S-cat-d8-m6pat900 --depth 8 --folds july \
    --es-rounds 900 --max-rounds 9000 --seed 7
CS=$($P rounds_of.py S-cat-d8-m6pat900 | cut -d' ' -f1); echo "cat m6 at patience 900 = $CS"
$P run.py --kind cat --name C2-d8p --depth 8 --folds july --rounds-m6 "$CS" --rounds-m11 5325
