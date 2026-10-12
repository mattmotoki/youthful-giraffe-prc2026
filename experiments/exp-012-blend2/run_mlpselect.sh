#!/usr/bin/env bash
# exp-012 member: the MLP the exp-008 assignment deferred. The desk asked for it
# because cross-library residual correlations of 0.84-0.91 say a non-tree member
# may add more than another tree.
#
# This is the epoch-count instrument, run once on the m11 family (all three
# exp-012 folds are m11). It runs a FULL 30-epoch OneCycle with early stopping
# disabled (--es-rounds 99) so the whole es-month curve is printed: an MLP's
# budget cannot be chosen the way a tree's is, because the LR schedule is a
# function of the budget, so "best epoch under a 30-epoch schedule" is not the
# same estimator as "a 30-epoch schedule stopped at that epoch". Reading the
# whole curve and refitting at a chosen budget with its own annealed schedule is
# the honest version. --lr 0.001 because MlpLearner takes an AdamW rate and
# run.py's --lr default (0.05) is a tree learning rate.
set -euo pipefail
cd "$(dirname "$0")/../exp-008-diversity"
P=../../.venv/bin/python
export PRC_GPU=0          # GPU 1 is held by exp-004-congestion
$P run.py --kind mlp --name S-mlp-e12 --select --folds july_loo \
    --lr 0.001 --max-rounds 30 --es-rounds 99 --seed 7
