#!/usr/bin/env bash
# exp-045-harvest: re-derive one month's summaries from its stored points on Kaggle.
#   bash experiments/exp-045-harvest/code/push_rederive.sh 2025-07
# Mounts ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/prc-harvest-<month> as the input and writes the summaries again under
# the current rule.  Nothing is re-fetched.
#
# The assembly moved to code/assemble_rederive.py: a here-document needs a spool file,
# and this laptop's volume runs at 100 %.
set -euo pipefail
MONTH="${1:?month, e.g. 2025-07}"
SRC="${2:-prc-harvest-$MONTH}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-045-harvest"
DIR="$EXP/kernels/rederive-$MONTH"
SLUG="prc-rederive-$MONTH"
python3 "$EXP/code/assemble_rederive.py" "$ROOT" "$MONTH" "$SRC"
echo "pushing $SLUG from ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SRC"
if [ "${PUSH:-1}" = "0" ]; then echo "PUSH=0: assembled only"; exit 0; fi
cd "$DIR" && "$HOME/.local/bin/kaggle" kernels push -p .
