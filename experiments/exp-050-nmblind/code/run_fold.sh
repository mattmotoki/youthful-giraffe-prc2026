#!/bin/bash
# One fold's expert fit behind the assignment's launch check: wait while the 1-minute load
# average is above 10, then fit (2 threads; nb_experts.py gates each fit launch again).
# Usage: run_fold.sh <fold> [--overwrite]
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FOLD="$1"; shift
LIMIT=10
while :; do
  L1=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$L1" -v m="$LIMIT" 'BEGIN{exit !(l <= m)}'; then break; fi
  echo "$(date -u +%FT%TZ) load $L1 > $LIMIT: holding the launch"
  sleep 60
done
echo "$(date -u +%FT%TZ) load $L1 <= $LIMIT: launching $FOLD"
cd "$ROOT" && OMP_NUM_THREADS=2 exec .venv/bin/python experiments/exp-050-nmblind/code/nb_experts.py "$FOLD" "$@"
