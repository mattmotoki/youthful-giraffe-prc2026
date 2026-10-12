#!/usr/bin/env bash
# The reproduction gate the desk set: does exp-012's copy of the d1-ds arm land
# within seed noise of exp-007's 307.45 on july_loo? Seed 7 is run first and its
# own RMSE is recorded separately from the three-seed bag, so the comparison is
# like for like against exp-007's single-seed number.
# PRC_THREADS=24 because this arm has linear leaves and exp-007 ran at the
# pipeline default of 24; deterministic=True does not cover linear trees.
set -euo pipefail
cd "$(dirname "$0")"
export PRC_THREADS=${PRC_THREADS:-6}   # docs/COMPUTE.md D3: this lane gets 6
../../.venv/bin/python winter_arm.py --name D1-winter --folds "${1:-july_loo}" --seeds 7,107,207
