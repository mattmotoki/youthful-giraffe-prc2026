#!/usr/bin/env bash
# exp-035-k2climb: the repair attempt for arm 2's negative CPU screen.
#
# The screen ran at 400 rounds, which is a third of the gate's 1200. A 225-column
# frame at `feature_fraction` 0.7 sees each column less often per round than a
# 157-column one, so "the wide arm is worse" and "the wide arm is not trained yet"
# have the same signature at 400 rounds. `--curve 200 --rounds 1200` prints the
# held-out RMSE every 200 rounds inside ONE fit, so both readings come out of the
# same compute the screen would have cost anyway.
#
# The printed numbers are `rmse_nonghost` and `main_only`, not the fold RMSE
# (the ghost head is applied after the loop), so they are compared arm to arm at
# equal rounds and never against a gate.
#
#   bash experiments/exp-035-k2climb/code/local_curve.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-035-k2climb"
PY="$ROOT/.venv/bin/python"
mkdir -p "$EXP/logs"
cd "$ROOT"
run () {
  local tag="$1" scales="$2"
  echo "=== $tag  PRC_WINDOW_SCALES=${scales:-<unset: shipped 600,3600>}"
  env PYTHONPATH="$ROOT" PRC_THREADS=3 OMP_NUM_THREADS=3 PRC_WINDOW=1 \
  ${scales:+PRC_WINDOW_SCALES="$scales"} \
  "$PY" experiments/exp-018-fresh/code/run.py --tag "$tag" --folds july \
      --rounds 1200 --curve 200 --exact-rounds 300 --ghost-mode mixbase \
      --main-model lgb --params '{"num_threads": 3}' 2>&1 \
    | grep -Ev "PerformanceWarning|^  X\[|^  newframe"
}
{
  date -u +"start %Y-%m-%dT%H:%M:%SZ"
  run k2c-curve-shipped ""
  run k2c-curve-wide "300,600,1800,3600"
  date -u +"end %Y-%m-%dT%H:%M:%SZ"
} 2>&1 | tee -a "$EXP/logs/local-curve.log"
