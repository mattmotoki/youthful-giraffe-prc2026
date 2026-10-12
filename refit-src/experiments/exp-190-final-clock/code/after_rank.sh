#!/usr/bin/env bash
# exp-190: the steps after the ranking-mode run, in one guarded job, from the repo root of a snapshot (remote).
#   bash experiments/exp-190-final-clock/code/after_rank.sh <ranking run pid> <ranking run log>
# 1. waits until the ranking run (run_chain.sh ranking, through step pack) has ended and its log says done;
# 2. read_chain.sh ranking: refit read, pack against head, head parity (refit noise) against v73-parts;
# 3. run_final.sh predict pmerge head: the final head from the ranking-mode boosters (predict-only path; it stops
#    unless the final frame's training matrix equals the ranking-mode one);
# 4. read_chain.sh final: tables, head parity against v73-parts on January/July;
# 5. day_count.py: the 2026-06-11 rows per airport with and without a head value.
set -euo pipefail
PID=$1
RANKLOG=$2
ROOT=$(pwd)
CODE=$ROOT/experiments/exp-190-final-clock/code
FINAL_ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
PY=${PY:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python}
stamp() { date -u +%FT%TZ; }
echo "$(stamp) after_rank: waiting for pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
grep -q 'run_chain done' "$RANKLOG" || { echo "$(stamp) the ranking run ended without 'run_chain done'" >&2; exit 1; }
echo "$(stamp) after_rank: ranking run done; commit $(cat "$ROOT/SNAPSHOT_COMMIT")"
bash "$CODE/read_chain.sh" ranking
bash "$CODE/run_final.sh" predict pmerge head
bash "$CODE/read_chain.sh" final
OUT=$FINAL_ROOT/finalclock/results
mkdir -p "$OUT"
[ -e "$OUT/day-2026-06-11.json" ] || $PY "$CODE/day_count.py" --head "$FINAL_ROOT/finalclock/parts/head.parquet" \
  --ranking "$FINAL_ROOT/finalcore/raw/final/final_ranking.parquet" --day 2026-06-11 --out "$OUT/day-2026-06-11.json"
echo "$(stamp) after_rank done"
