#!/usr/bin/env bash
# exp-035-k2climb: the CPU screen for arm 2, before a GPU session is asked for.
#
# exp-027 s14b's own local A/B recipe, unchanged, so the two are comparable:
# the `july` fold, LightGBM only, 400 rounds, one seed, 3 threads. It is a
# weaker configuration than the Kaggle gate (no CatBoost, a third of the rounds,
# 5 training months instead of 11) and is a screen, not the gate -- exp-027's
# same screen read +0.80 s where the gate then read +0.53 s.
#
#   bash experiments/exp-035-k2climb/code/local_ab.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-035-k2climb"
PY="$ROOT/.venv/bin/python"
mkdir -p "$EXP/logs" "$EXP/analysis"
cd "$ROOT"
run () {   # <tag> <scales>
  local tag="$1" scales="$2"
  echo "=== $tag  PRC_WINDOW_SCALES=${scales:-<unset: shipped 600,3600>}"
  # `env` and not an assignment prefix: bash recognises assignments at parse
  # time, so a `${scales:+VAR=value}` prefix expands into a command word and the
  # shell reports "command not found" after the arm has already been priced.
  env PYTHONPATH="$ROOT" PRC_THREADS=3 OMP_NUM_THREADS=3 PRC_WINDOW=1 \
  ${scales:+PRC_WINDOW_SCALES="$scales"} \
  "$PY" experiments/exp-018-fresh/code/run.py --tag "$tag" --folds july \
      --rounds 400 --exact-rounds 300 --ghost-mode mixbase --main-model lgb \
      --params '{"num_threads": 3}' 2>&1 | grep -v PerformanceWarning | grep -v "^  X\[" 
}
{
  date -u +"start %Y-%m-%dT%H:%M:%SZ"
  [ -n "${SKIP_SHIPPED:-}" ] || run k2c-local-shipped ""
  run k2c-local-wide "300,600,1800,3600"
  date -u +"end %Y-%m-%dT%H:%M:%SZ"
} 2>&1 | tee -a "$EXP/logs/local-ab.log"
