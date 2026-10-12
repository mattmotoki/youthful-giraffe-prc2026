#!/usr/bin/env bash
# exp-045-harvest: retry one push until the shared Kaggle CPU pool has a slot.
#   bash code/push_when_free.sh <name> '<json days>' <limit> <budget> [tries]
set -uo pipefail
NAME="${1:?}"; DAYS="${2:?}"; LIMIT="${3:-0}"; BUDGET="${4:-3600}"; TRIES="${5:-90}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
for i in $(seq 1 "$TRIES"); do
  OUT="$(bash "$ROOT/experiments/exp-045-harvest/code/push_kernel.sh" "$NAME" "$DAYS" "$LIMIT" "$BUDGET" 2>&1)"
  if grep -qi "error" <<<"$OUT"; then
    echo "$(date -u +%H:%M:%SZ) try $i refused: $(grep -i error <<<"$OUT" | head -1)"
    sleep 120
  else
    echo "$(date -u +%H:%M:%SZ) try $i pushed $NAME"
    echo "$OUT" | tail -3
    exit 0
  fi
done
echo "$(date -u +%H:%M:%SZ) gave up after $TRIES tries"
exit 1
