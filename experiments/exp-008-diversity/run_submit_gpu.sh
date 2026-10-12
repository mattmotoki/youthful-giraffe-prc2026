#!/usr/bin/env bash
# CatBoost ranking predictions at the corrected eleven-month budget.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
while pgrep -f "run_gpu6" >/dev/null; do sleep 30; done
$P run.py --submit --kind cat --name C2-d8p --depth 8 --rounds-m6 3625 --rounds-m11 5325
