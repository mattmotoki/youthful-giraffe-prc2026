#!/usr/bin/env bash
# exp-012 MLP member, july_loo, three seeds. Usage: ./run_mlp.sh [epochs] [folds]
#
# Budget: the 30-epoch curve (logs/mlpselect.log) does NOT identify one. Train
# RMSE falls monotonically 462 -> 232 while the early-stop month bounces 281,
# 359, 285, 316, 566, ... with no trend after epoch 9, so run.py's
# best_iteration=9 is a lucky draw and not a minimum. A budget also cannot be
# read off that curve even in principle: the LR schedule is OneCycle over the
# declared budget, so "epoch 9 of a 30-epoch schedule" is a different estimator
# from "a 9-epoch schedule", which anneals to zero. 12 is a round number in the
# flat region, taken deliberately rather than selected, and the MLP is treated
# as a SCREEN of whether a non-tree member earns weight at all -- if it does,
# tuning it is the follow-up, and if it does not, no tuning would have saved it.
#
# Three seeds because the epoch-to-epoch spread above says a single MLP draw
# cannot be trusted; the per-seed spread is itself part of the answer. This is
# not the seed-bagging case priced at 7 % retention -- see RESULTS.
#
# --lr 0.001 is an AdamW rate; run.py's --lr default of 0.05 is a tree rate.
set -euo pipefail
EPOCHS="${1:-12}"; FOLDS="${2:-july_loo}"
cd "$(dirname "$0")/../exp-008-diversity"
P=../../.venv/bin/python
export PRC_GPU=0
$P run.py --kind mlp --name E12-mlp --lr 0.001 --seed-bag 3 --folds "$FOLDS" \
    --rounds-m6 "$EPOCHS" --rounds-m11 "$EPOCHS" --outdir ../exp-012-blend2/oof
