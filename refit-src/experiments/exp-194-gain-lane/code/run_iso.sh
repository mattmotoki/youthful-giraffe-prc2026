#!/usr/bin/env bash
# exp-194 review repair on remote (SPEC addendum 14): r4mk_iso.py for x01, then x07, from a git-archive snapshot
# (prc, experiments/exp-047-queue/code, exp-146-validation/frozen, exp-147-foldserve/code, exp-148-short-route/code,
# exp-194-gain-lane/code). Inputs of run o2: dark_values.json and the L/Lp dark refits.
#   setsid nohup env THREADS=2 bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <snap>/experiments/exp-194-gain-lane/code/run_iso.sh <tag> [<smoke rows>] >/dev/null 2>&1 < /dev/null &
# Outputs (write-once): ${PRC_FINAL_ROOT:-/path/to/final}/gainlane/iso-<tag>/R4mksX-x0M.{model.txt,cat.cbm,parquet,json}.
set -Eeuo pipefail
TAG=$1
SMOKE=${2:-}
SNAP=$(cd "$(dirname "$0")/../../.." && pwd)
C=$SNAP/experiments/exp-194-gain-lane/code
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane
O=$G/iso-$TAG
D=$G/dark-o2
T=${THREADS:-2}
PYR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/.venv/bin/python   # lightgbm/catboost of exp-148
W148=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
export PYTHONDONTWRITEBYTECODE=1 PRC_SOURCE_ROOT=$SNAP
export PRC_DATA_DIR=${PRC_DATA_DIR:?the folder of the 2025 raw month files (labels of the jjx training rows)}
unset PRC_RANKING_MODE
mkdir -p "$O"
echo "run_iso: tag $TAG threads $T smoke ${SMOKE:-none} snapshot $SNAP pid $$ $(date -u +%FT%TZ)"
for own in x01 x07; do
  echo "run_iso: $own start $(date -u +%FT%TZ)"
  if [ -n "$SMOKE" ]; then set -- --smoke "$SMOKE"; else set --; fi
  $PYR "$C/r4mk_iso.py" --owner $own --work $W148 --dark-values $D/dark_values.json \
    --member-dark L=$D/learners/L-dark.parquet --member-dark Lp=$D/learners/Lp-dark.parquet \
    --out "$O" --threads "$T" "$@"
  echo "run_iso: $own done $(date -u +%FT%TZ)"
done
sha256sum "$O"/R4mksX-*
echo "run_iso: all done $(date -u +%FT%TZ)"
