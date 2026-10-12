#!/usr/bin/env bash
# exp-025-cloud: pull one kernel's outputs and file them in the record.
#
#   bash experiments/exp-025-cloud/code/pull_kernel.sh prc-member-l2-hub [<log name>]
#
# Writes logs/<name>.log (the plain-text kernel log) and copies every *.json
# output into results/. Parquet outputs stay in the scratch directory: a member
# ranking prediction built from synthetic input must not enter the record.
set -euo pipefail
SLUG="${1:?kernel slug without the owner}"
NAME="${2:-$SLUG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-025-cloud"
TMP="$(mktemp -d)"
"$HOME/.local/bin/kaggle" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" >/dev/null
mkdir -p "$EXP/logs" "$EXP/results"
python3 - "$TMP/$SLUG.log" "$EXP/logs/$NAME.log" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as fh:
    for rec in raw:
        fh.write(rec.get("data", ""))
PY
for j in "$TMP"/*.json; do [ -e "$j" ] && cp "$j" "$EXP/results/$(basename "$j")"; done
echo "logs/$NAME.log  $(wc -l < "$EXP/logs/$NAME.log") lines"
ls -1 "$TMP"
