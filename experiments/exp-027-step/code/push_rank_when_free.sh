#!/usr/bin/env bash
# exp-027-step: push the ranking-side K2 member kernel as soon as a slot frees.
# Kaggle caps concurrent GPU batch sessions at 2 and `cloud` holds both, so the
# push is retried rather than queued. Exits on the first successful push.
set -u
K="$HOME/.local/bin/kaggle"
DIR="/path/to/competitions/prc-data-challenge-2026/experiments/exp-027-step/kernels/f1-window-rank"
for i in $(seq 1 240); do
  OUT="$("$K" kernels push -p "$DIR" 2>&1)"
  if printf '%s' "$OUT" | grep -qi "successfully pushed\|Kernel version"; then
    echo "$(date -u +%H:%M:%S) PUSHED: $OUT"; exit 0
  fi
  echo "$(date -u +%H:%M:%S) waiting: $(printf '%s' "$OUT" | tail -1)"
  sleep 60
done
echo "gave up after 4 hours"; exit 1
