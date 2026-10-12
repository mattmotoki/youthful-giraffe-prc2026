#!/usr/bin/env bash
# The serve side of the Huber corrected budget, for blend-5.
#
# The desk's instruction was to ship the 0.12 s Huber budget into the next blend
# rather than as its own candidate. The three OOF folds exist and are published
# as L2-hub8725-{fold}.parquet, but a blend cannot SERVE a member it only holds
# out of fold: blend-5 needs the twelve-month ranking fit at the same budget.
#
# run_submission scales the eleven-month frozen count by 12/11, so 8,725 becomes
# 9,518 rounds; no early stop runs here, so no ranking row and no held-out month
# can influence it. Output is L2-hub8725-ranking.parquet, a NEW name: the shipped
# 2,100-round L2-hub-ranking.parquet stays on disk untouched as the fallback if
# the rotation turns against the corrected budget.
set -euo pipefail
cd "$(dirname "$0")/../exp-008-diversity"
export PRC_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
       OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
../../.venv/bin/python run.py --kind lgb --name L2-hub8725 --variant plain \
    --huber-delta 1200 --rounds-m6 3225 --rounds-m11 8725 --submit
