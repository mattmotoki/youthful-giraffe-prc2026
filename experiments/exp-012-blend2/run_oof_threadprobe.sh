#!/usr/bin/env bash
# Does the OOF's thread count matter to the blend?
#
# The pinned OOF that fits the blend weights was built at 16 threads; the ranking
# predictions the candidate serves will be built at 6. That is an estimator
# mismatch of the same class this lane has documented twice, and the honest
# question is its SIZE, not whether it exists. Rebuilding one fold of L2-hub's
# OOF at 6 threads and comparing to the pinned 16-thread version answers it for
# about fifteen minutes of compute, against several hours to rebuild every
# member's OOF on the assumption that it matters.
set -euo pipefail
cd "$(dirname "$0")/../exp-008-diversity"
export PRC_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
../../.venv/bin/python run.py --kind lgb --name OOFT6-hub --variant plain \
    --huber-delta 1200 --folds july_loo --rounds-m6 3225 --rounds-m11 2100 \
    --outdir ../exp-012-blend2/oof
