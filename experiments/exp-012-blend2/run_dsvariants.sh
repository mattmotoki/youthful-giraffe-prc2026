#!/usr/bin/env bash
# Robustness audit (desk, 2026-09-05): the shipped D1-winter member carries the
# full unbounded day-state block, measured by exp-007 at +72.1 s on january's
# most disrupted days. Two repairs, measured on the two folds the desk named:
#   arm A  the full block, and the SAME model's clipped-at-predict-time twin --
#          exp-007's "clipped-block" experiment, which they named as the one that
#          separates boundedness from column count and never ran
#   arm B  ds_deice_hr alone, exp-007 finding 14's arm, bounded by construction
set -euo pipefail
cd "$(dirname "$0")"
export PRC_THREADS=${PRC_THREADS:-6}   # docs/COMPUTE.md D3: this lane gets 6
P=../../.venv/bin/python
case "${1:?arm}" in
  A) $P winter_arm.py --name D1w-all   --folds july_loo,january --seeds 7,107,207 --clip ;;
  B) $P winter_arm.py --name D1w-deice --folds july_loo,january --seeds 7,107,207 --ds-columns deice ;;
esac
