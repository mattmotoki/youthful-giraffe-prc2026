#!/usr/bin/env bash
# Season-matched round selection for july_loo, then the repaired july_loo folds.
# The prediction under test, stated before running: if the eleven-month count is
# small because the early-stop month is calm winter rather than because there are
# eleven months, then early-stopping july_loo on 2025-08 gives a count near the
# six-month fold's, not near 425.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
# Wait for the whole earlier GPU script, not for its individual arms: between
# two of them no python is running and a check on arm names would race into the
# gap and put two CatBoost fits on one 24 GB card.
while pgrep -f "run_gpu2" >/dev/null; do sleep 30; done
$P run.py --select --kind cat --name S-cat-d8-jloo --depth 8 --folds july_loo --es-month 2025-08 --max-rounds 6000 --seed 7
$P run.py --select --kind xgb --name S-xgb-d8-jloo --depth 8 --folds july_loo --es-month 2025-08 --max-rounds 6000 --seed 7
CS=$($P rounds_of.py S-cat-d8-jloo | cut -d' ' -f2); echo "cat m11s=$CS"
XS=$($P rounds_of.py S-xgb-d8-jloo | cut -d' ' -f2); echo "xgb m11s=$XS"
set -- $($P rounds_of.py S-cat-d8 S-cat-d8-jan)
$P run.py --kind cat --name C1-d8 --depth 8 --folds july_loo --rounds-m6 "$1" --rounds-m11 "$2" --rounds-m11s "$CS"
set -- $($P rounds_of.py S-xgb-d8 S-xgb-d8-jan)
$P run.py --kind xgb --name X1-d8 --depth 8 --folds july_loo --rounds-m6 "$1" --rounds-m11 "$2" --rounds-m11s "$XS"
