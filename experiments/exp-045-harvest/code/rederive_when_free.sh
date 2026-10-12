#!/usr/bin/env bash
# exp-045-harvest: push one re-derive as soon as the shared pool has a slot, then
# hand the slot back to the fetch dispatcher.
#   bash code/rederive_when_free.sh <name> <source slug> [tries]
set -uo pipefail
NAME="${1:?}"; SRC="${2:?}"; TRIES="${3:-180}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-045-harvest"
for i in $(seq 1 "$TRIES"); do
  OUT="$(bash "$EXP/code/push_rederive.sh" "$NAME" "$SRC" 2>&1)"
  if grep -qi "error" <<<"$OUT"; then
    echo "$(date -u +%H:%M:%SZ) try $i refused"; sleep 120; continue
  fi
  echo "$(date -u +%H:%M:%SZ) pushed prc-rederive-$NAME"
  until "$HOME/.local/bin/kaggle" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/prc-rederive-$NAME" 2>&1 \
        | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
  echo "$(date -u +%H:%M:%SZ) finished: $("$HOME/.local/bin/kaggle" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/prc-rederive-$NAME" 2>&1 | sed 's/.*status //')"
  bash "$EXP/code/pull_rederive.sh" "$NAME" 2>&1 | tail -2
  echo 4 > "$EXP/analysis/max_slots"
  echo "$(date -u +%H:%M:%SZ) slots back to 4"
  exit 0
done
echo "$(date -u +%H:%M:%SZ) gave up"; echo 4 > "$EXP/analysis/max_slots"; exit 1
