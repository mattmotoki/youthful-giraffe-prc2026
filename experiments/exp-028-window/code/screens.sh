#!/bin/bash
# The screen the assignment asks for, run one job at a time so the 2-thread share
# is never exceeded.  Each step appends to climb.log and writes into analysis/.
set -u
cd "$(dirname "$0")/../../.."
export PRC_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
E=experiments/exp-028-window
# 1. the second-layer arm's own screen: width, rounds, leaves, fit population,
#    and whether the arm may read the head's prediction
.venv/bin/python $E/code/arm.py screen              > $E/logs/screen-arm.log 2>&1
# 2. the candidate's window width, inside the stacker (the stacker's rounds and
#    leaves are exp-026's and moving them moves stack-5 too, so they are not
#    screened here -- see RESULTS section 6)
PRC_WIN=300 PRC_VARIANTS=depx .venv/bin/python $E/code/stack5w.py > $E/logs/stack5w-w300.log 2>&1
PRC_WIN=900 PRC_VARIANTS=depx .venv/bin/python $E/code/stack5w.py > $E/logs/stack5w-w900.log 2>&1
# 3. did stack-5 already make the move the second-layer arm makes on blend-5?
.venv/bin/python $E/code/overlap.py                 > $E/logs/overlap.log 2>&1
echo "screens done"
