#!/bin/bash
# exp-043-pilot: rebuild every measurement in analysis/ from the stored slices.
#
# Round 4 found the segment builder using an unstable sort, which needed the whole
# chain re-run.  Doing that from memory is how a step gets missed, so the order is
# written down here and the report is rendered last, from what this produced.
#
#   bash experiments/exp-043-pilot/code/rerun.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
C=experiments/exp-043-pilot/code
WEEK="2025-07-07 2025-07-08 2025-07-09 2025-07-10 2025-07-11 2025-07-12 2025-07-13"
HELD="2025-07-09 2025-07-10 2025-07-11 2025-07-12 2025-07-13"
DEV="2025-07-07 2025-07-08"

step() { echo "=== $* ==="; "$@"; }

step $PY $C/cohort.py
step $PY $C/census.py $WEEK
step $PY $C/match.py --tag=week $WEEK
step $PY $C/match.py --tag=heldout $HELD
for k in obs_offblock_disp obs_offblock_loose obs_offblock_strict; do
  step $PY $C/align.py $k heldout
done
for k in obs_offblock_disp obs_offblock_loose; do
  step $PY $C/align.py $k
done
step $PY $C/sweep.py $DEV
step $PY $C/nonm_probe.py
step $PY $C/divergence.py
step $PY $C/ordering.py
step $PY $C/extrapolate.py
step $PY $C/report_tables.py
step $PY $C/report_tables.py --selftest
echo "=== chain complete ==="
