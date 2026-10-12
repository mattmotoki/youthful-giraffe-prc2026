#!/usr/bin/env bash
# Which of two things made CatBoost's eleven-month count 425 when its six-month
# count is 3,625? Two hypotheses, separated by two training-only selections:
#   (a) SEASON  -- the early-stop month is calm winter. Tested by run_gpu3.sh,
#       which early-stops july_loo on 2025-08 at the same patience.
#   (b) PATIENCE -- 100 rounds is 57 % of LightGBM's trajectory (137 rounds) but
#       3 % of CatBoost's (2,969), so a fixed patience is not scale-free and can
#       stop CatBoost on a flat stretch it would have come off. Tested here, by
#       re-selecting the SAME december fold and the SAME 2025-11 early-stop month
#       at patience 900 (about 30 % of CatBoost's six-month trajectory).
# Neither selection touches a validation month.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=1
while pgrep -f "run_gpu3" >/dev/null; do sleep 30; done
$P run.py --select --kind cat --name S-cat-d8-pat900 --depth 8 --folds december \
    --es-rounds 900 --max-rounds 5000 --seed 7
