#!/usr/bin/env bash
# exp-043-pilot: pull one surface kernel's log and slices.
#   bash experiments/exp-043-pilot/code/pull_kernel.sh probe
# Log -> logs/<name>.log ; run_*.json -> analysis/ ; parquets -> artifacts/exp-043/
set -euo pipefail
NAME="${1:?kernel name}"
SLUG="prc-surface-$NAME"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-043-pilot"
mkdir -p "$EXP/work"
# stay off /private/tmp: the data volume runs out there first
TMP="$(mktemp -d "$EXP/work/pull-XXXXXX")"
"$HOME/.local/bin/kaggle" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" >/dev/null
mkdir -p "$EXP/logs" "$EXP/analysis" "$ROOT/artifacts/exp-043"
python3 - "$TMP/$SLUG.log" "$EXP/logs/$NAME.log" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as fh:
    for rec in raw:
        fh.write(rec.get("data", ""))
PY
for p in "$TMP"/run_*.json; do [ -e "$p" ] && cp "$p" "$EXP/analysis/$(basename "$p")"; done
for p in "$TMP"/*.parquet; do [ -e "$p" ] && cp "$p" "$ROOT/artifacts/exp-043/$(basename "$p")"; done
echo "logs/$NAME.log  $(wc -l < "$EXP/logs/$NAME.log") lines"
ls -1 "$TMP"
