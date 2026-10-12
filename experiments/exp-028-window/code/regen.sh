#!/bin/bash
set -u
cd "$(dirname "$0")/../../.."
export PRC_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
E=experiments/exp-028-window
PRC_WIN=600 PRC_VARIANTS=depx .venv/bin/python $E/code/stack5w.py > $E/logs/stack5w-w600.log 2>&1
PRC_WIN=300 PRC_VARIANTS=depx .venv/bin/python $E/code/stack5w.py > $E/logs/stack5w-w300.log 2>&1
PRC_WIN=900 PRC_VARIANTS=depx .venv/bin/python $E/code/stack5w.py > $E/logs/stack5w-w900.log 2>&1
for t in w600 w300 w900; do
  .venv/bin/python $E/code/where.py depx $t    > $E/logs/where-$t.log 2>&1
  .venv/bin/python $E/code/restrict.py depx $t > $E/logs/restrict-$t.log 2>&1
done
echo "regen done"
