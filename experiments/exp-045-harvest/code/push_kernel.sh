#!/usr/bin/env bash
# exp-045-harvest: generate and push one ten-box surface kernel.
#
#   bash experiments/exp-045-harvest/code/push_kernel.sh smoke '["2025-07-07"]' 40000
#   bash experiments/exp-045-harvest/code/push_kernel.sh 2026-01 "$(python3 code/days.py 2026-01)"
#
# The kernel is assembled from three sources, and the assembled file is committed,
# so the record holds exactly what ran:
#   kernels/surface/prc-harvest.py            the stream, the ten boxes, the day loop
#   exp-043-pilot/code/segments.py            the FROZEN off-block rule, verbatim, at HEAD
#   code/allsegments.py                       every ground segment, not only departures
# The only edit made to the two inlined files is the removal of their `from __future__`
# line and of the import that binds them together; `code/check_rule.py` proves the
# assembled rule still reproduces the pilot's numbers.
#
# The assembly is code/assemble_kernel.py: a here-document needs a spool file, and this
# laptop's volume runs at 100 %.
set -euo pipefail
NAME="${1:?kernel name, e.g. 2026-01}"
DAYS="${2:?json list of days}"
LIMIT="${3:-0}"
BUDGET="${4:-39600}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# exp-064: the account and the kaggle executable were hard-coded, so the
# published route could not be run by anybody else.  Unset, both keep the
# values this repository always used, so nothing about our runs changes.
KAGGLE_USER="${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}"
KAGGLE_BIN="${PRC_KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
EXP="$ROOT/experiments/exp-045-harvest"
DIR="$EXP/kernels/surface-$NAME"
SLUG="prc-harvest-$NAME"
PRC_KAGGLE_USER="$KAGGLE_USER" \
  python3 "$EXP/code/assemble_kernel.py" "$ROOT" "$NAME" "$DAYS" "$LIMIT" "$BUDGET"
echo "pushing $NAME as $KAGGLE_USER/$SLUG  limit=$LIMIT  budget=$BUDGET"
if [ "${PUSH:-1}" = "0" ]; then echo "PUSH=0: assembled only"; exit 0; fi
cd "$DIR" && "$KAGGLE_BIN" kernels push -p .
