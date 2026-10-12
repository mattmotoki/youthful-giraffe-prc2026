#!/usr/bin/env bash
# exp-190: the reads of one chain run, from the repo root of a snapshot (development reads; remote, under the guard).
#   bash experiments/exp-190-final-clock/code/read_chain.sh <ranking|final>
# refit_read.py (tables, seed vectors, merged values, boosters, pack against head) and finalcore's parity.py of
# the head part against v73-parts.parquet (kinds refit and deterministic). Outputs next to the work folder's
# parts: refit-read.json, head-parity-refit.json, head-parity-deterministic.json (write-once).
set -euo pipefail
MODE=$1
ROOT=$(pwd)
FINAL_ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
WORK=${EXP190_WORK:-$FINAL_ROOT/finalclock/work-$MODE}
case "$MODE" in ranking) PARTS=$WORK/parts ;; final) PARTS=$FINAL_ROOT/finalclock/parts ;; *) exit 2 ;; esac
PY=${PY:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python}
REFS=$FINAL_ROOT/finalcore/refs/v73-parts.parquet
export PRC_DATA_DIR=${EXP159_RAW:-$FINAL_ROOT/finalcore/raw} OMP_NUM_THREADS=4
echo "$(date -u +%FT%TZ) read_chain $MODE work $WORK parts $PARTS commit $(cat "$ROOT/SNAPSHOT_COMMIT" 2>/dev/null)"
[ -e "$WORK/refit-read.json" ] || $PY "$ROOT/experiments/exp-190-final-clock/code/refit_read.py" "$WORK" "$WORK/refit-read.json"
for kind in refit deterministic; do
  out=$PARTS/head-parity-$kind.json
  [ -e "$out" ] && continue
  # the deterministic kind exits 1 when a value differs; that is a result here, not a failure of the read
  $PY "$ROOT/experiments/exp-186-final-core/code/parity.py" --refs "$REFS" --new "$PARTS/head.parquet" \
    --part head --col pred_HB3u --kind "$kind" --out "$out" || echo "parity $kind exit $?"
done
sha256sum "$PARTS/head.parquet" "$WORK"/out/*/HB3u-*.parquet "$WORK"/inputs/*.parquet
echo "$(date -u +%FT%TZ) read_chain done"
