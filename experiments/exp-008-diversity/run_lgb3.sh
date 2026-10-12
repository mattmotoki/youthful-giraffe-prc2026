#!/usr/bin/env bash
# Is the LightGBM side truncated too? L2-hub is the best single model and the
# highest-weighted LightGBM member, and its patience-100 best iteration was 1,926
# -- a 100-round patience is 5 % of that trajectory, the same ratio that cost
# CatBoost twelvefold. This changes only the patience, on the same fold and month.
# It waits on an OUTPUT FILE rather than on a process pattern: a pgrep wait can
# match the waiter's own command line, which has deadlocked two queues already.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=0
until [ -f ../../artifacts/exp-008/pred/T-tail-ranking.parquet ]; do sleep 60; done
$P run.py --select --kind lgb --name S-lgb-hub-pat900 --variant plain --huber-delta 1200 \
    --folds december --es-rounds 900 --max-rounds 8000 --seed 7
