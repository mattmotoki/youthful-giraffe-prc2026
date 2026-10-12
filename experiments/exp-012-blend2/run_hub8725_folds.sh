#!/usr/bin/env bash
# Last item in the desk's priority order: the Huber corrected budget on the two
# folds I do not hold it for.
#
# Why this is worth 0.2 s rather than the -3.57 s I first recorded. Measured by
# MEMBER RMSE the 8,725-round arm is 3.57 s worse than the shipped 2,100. Measured
# by what it adds to the BLEND - the criterion the desk has since made the rule -
# it is 0.18 s BETTER on july_loo, because its errors move into the tail where the
# blend is weak and its correlation with the other members falls 0.9309 -> 0.9132.
# december and january decide whether that survives a three-fold rotation.
set -euo pipefail
cd "$(dirname "$0")/../exp-008-diversity"
export PRC_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
../../.venv/bin/python run.py --kind lgb --name E12-hub8725 --variant plain \
    --huber-delta 1200 --folds december,january \
    --rounds-m6 3225 --rounds-m11 8725 --outdir ../exp-012-blend2/oof
