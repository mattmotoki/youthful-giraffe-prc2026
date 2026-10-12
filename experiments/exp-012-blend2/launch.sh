#!/usr/bin/env bash
# Launch gate for this lane under the box-wide compute rule (docs/COMPUTE.md, compute allocation).
#
#   ./launch.sh <logfile> <command...>
#
# This lane's allocation is 6 CPU threads plus GPU 1 for CatBoost. Rather than
# remembering to set six different environment variables on every launch, every
# job in this experiment goes through here: it exports the allocation, refuses to
# start while the one-minute load is above the gate, and records what it did.
# A GPU fit passes CAP=2 explicitly, since GPU work counts only its feeding
# threads against the cap.
set -euo pipefail
CAP="${CAP:-6}"
GATE="${GATE:-140}"
LOG="${1:?usage: launch.sh <logfile> <command...>}"; shift
LOAD=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($LOAD > $GATE)}"; then
  echo "HOLDING: 1-minute load $LOAD is above the $GATE gate (docs/COMPUTE.md D3)."
  echo "  not launching: $*"
  exit 3
fi
export PRC_THREADS="$CAP" OMP_NUM_THREADS="$CAP" MKL_NUM_THREADS="$CAP" \
       OPENBLAS_NUM_THREADS="$CAP" NUMEXPR_NUM_THREADS="$CAP"
echo "launching at load $LOAD with $CAP threads: $*"
nohup "$@" > "$LOG" 2>&1 &
echo "  pid $! -> $LOG"
