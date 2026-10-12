#!/usr/bin/env bash
# Ranking predictions for the members that carry non-zero blend weight.
# NNLS on 510,096 OOF rows gives L2-hub 0.333, T-tail 0.404, C-cat 0.235,
# L0-lin 0.030 and exactly zero to L1-plain, L3-monoraw and X-xgb, so only four
# members need a twelve-month fit. The submission round count is the eleven-month
# frozen count scaled by 12/11; no early stop runs here.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=0
$P run.py --submit --kind lgb --name L0-lin --variant lin --rounds-m6 175 --rounds-m11 350
$P run.py --submit --kind lgb --name L2-hub --variant plain --huber-delta 1200 --rounds-m6 3225 --rounds-m11 2100
$P tail_ranking.py
