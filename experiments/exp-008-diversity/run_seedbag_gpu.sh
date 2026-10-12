#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
while pgrep -f "run_submit_gpu" >/dev/null; do sleep 30; done
$P run.py --kind cat --name SB-C2 --depth 8 --seed-bag 3 --folds july_loo \
    --rounds-m6 3625 --rounds-m11 5325
