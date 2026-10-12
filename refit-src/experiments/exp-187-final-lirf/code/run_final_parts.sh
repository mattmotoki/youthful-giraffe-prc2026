#!/usr/bin/env bash
# The final parts of the three frames (path V), from a snapshot of GITPATHS.txt on remote, under the hold guard:
#   1. clock input from finaldata's v25 class files of all four months (clock_input.py; a missing month stops the run),
#   2. frames_v.py --mode final --parts --tag <tag> with finalroute's np_base part (no fill), write-once under
#      $PRC_FINAL_ROOT/finallirf/parts/frames_<frame>-<tag>.parquet and frames-record-<tag>.json,
#   3. the January/July parity read of each part (parts_parity.sh, finalcore parity.py, kind refit).
#   bash experiments/exp-187-final-lirf/code/run_final_parts.sh <tag> <np_base.parquet> <class file> [<class file> ...]
set -euo pipefail
TAG=$1; NPB=$2; shift 2
FR=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}/finallirf
PY=${PYTHON:-python3}
CK=$FR/inputs/clock-$TAG.parquet
if [ -e "$CK" ]; then echo "clock input $CK exists (write-once); use a new tag" >&2; exit 1; fi
mkdir -p "$FR/inputs"
args=()
for f in "$@"; do args+=(--class-file "$f"); done
"$PY" -W ignore::FutureWarning experiments/exp-187-final-lirf/code/clock_input.py "${args[@]}" --out "$CK"
"$PY" -W ignore::FutureWarning experiments/exp-187-final-lirf/code/frames_v.py --mode final --clock "$CK" \
  --np-base "$NPB" --fill none --parts --tag "$TAG"
bash experiments/exp-187-final-lirf/code/parts_parity.sh "$FR/parts" "$TAG"
echo "FINAL-PARTS-DONE $TAG"
