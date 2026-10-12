#!/usr/bin/env bash
# The GPU family: CatBoost (native categoricals, ordered target statistics) and
# XGBoost hist (native categorical splits), both on the residual target.
# The eleven-month round count gets a second early-stop month (2025-12, from the
# january fold) because the first reading disagreed with the six-month family by
# a factor of seven and one reading cannot tell a real family difference from a
# property of November.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1

$P run.py --select --kind cat --name S-cat-d8-jan --depth 8 --folds january --max-rounds 4000 --seed 7
$P run.py --select --kind xgb --name S-xgb-d8-jan --depth 8 --folds january --max-rounds 4000 --seed 7
set -- $($P rounds_of.py S-cat-d8 S-cat-d8-jan); echo "cat rounds m6=$1 m11=$2"
$P run.py --kind cat --name C1-d8 --depth 8 --rounds-m6 "$1" --rounds-m11 "$2"
set -- $($P rounds_of.py S-xgb-d8 S-xgb-d8-jan); echo "xgb rounds m6=$1 m11=$2"
$P run.py --kind xgb --name X1-d8 --depth 8 --rounds-m6 "$1" --rounds-m11 "$2"
