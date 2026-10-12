#!/usr/bin/env bash
# GPU arms, with the round counts already frozen in select-*.json.
# C1-d8h60 is the head-budget sensitivity: identical to C1-d8 except that the
# noac head keeps LightGBM's 60 rounds, so the per-kind head budget is a
# measured choice rather than an argument.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
set -- $($P rounds_of.py S-cat-d8 S-cat-d8-jan); echo "cat rounds m6=$1 m11=$2"
$P run.py --kind cat --name C1-d8 --depth 8 --rounds-m6 "$1" --rounds-m11 "$2"
set -- $($P rounds_of.py S-xgb-d8 S-xgb-d8-jan); echo "xgb rounds m6=$1 m11=$2"
$P run.py --kind xgb --name X1-d8 --depth 8 --rounds-m6 "$1" --rounds-m11 "$2"
set -- $($P rounds_of.py S-cat-d8 S-cat-d8-jan)
$P run.py --kind cat --name C1-d8h60 --depth 8 --folds july --noac-rounds 60 --rounds-m6 "$1" --rounds-m11 "$2"
