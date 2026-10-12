#!/bin/bash
# exp-054-arrivalrepr: the review-repair re-run (SPEC amendment A1; Sol round 1, finding 1).
# The registered pipeline, one job at a time, every job in its own process, in the clean order:
#   1. the three representations, one fold per process, ARR-only reader (represent_r.py);
#   2. the iteration-1 departure fits, one fold per process (fit.py, registered, unchanged);
#   3. the iteration-1 pricing (price.py), then the gate: the re-run's iteration-1 files must be the
#      recorded bytes, because fit2.py and price2.py read the experiment's own iteration-1 files;
#   4. the iteration-2 departure fits, one fold per process (fit2.py), and pricing (price2.py);
#   5. the full check (rerun_check.py --stage all).
# Every output goes to rerun/ (write-once); the check records go to analysis/rerun-check*.json.
# Every fit launch holds while the 1-minute load is above 10 (e54.hold_load).
#
#   cd <repo> && set -o noclobber && nohup bash experiments/exp-054-arrivalrepr/code/rerun.sh \
#       > experiments/exp-054-arrivalrepr/logs/rerun.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
X=experiments/exp-054-arrivalrepr
OUT=$X/rerun
PY=.venv/bin/python
export OMP_NUM_THREADS=3
if [ -e "$OUT" ]; then echo "refusing: $OUT exists (write-once)"; exit 1; fi
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ); $(uptime)"
for f in july_loo december january; do
  echo "== represent_r.py $f $(date -u +%H:%M:%SZ)"
  $PY $X/code/represent_r.py $f --out-dir $OUT
done
for f in july_loo december january; do
  echo "== fit.py $f $(date -u +%H:%M:%SZ)"
  $PY $X/code/fit.py $f --out-dir $OUT
done
echo "== price.py $(date -u +%H:%M:%SZ)"
$PY $X/code/price.py --out-dir $OUT
echo "== rerun_check.py --stage iter1 $(date -u +%H:%M:%SZ)"
$PY $X/code/rerun_check.py --stage iter1
for f in july_loo december january; do
  echo "== fit2.py $f $(date -u +%H:%M:%SZ)"
  $PY $X/code/fit2.py $f --out-dir $OUT
done
echo "== price2.py $(date -u +%H:%M:%SZ)"
$PY $X/code/price2.py --out-dir $OUT
echo "== rerun_check.py --stage all $(date -u +%H:%M:%SZ)"
$PY $X/code/rerun_check.py --stage all
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
