#!/usr/bin/env bash
# exp-040-chain: pull one kernel's outputs and file them in the record.
#
#   bash experiments/exp-040-chain/code/pull_kernel.sh july_loo
#
# Writes logs/<what>.log, results/port-<what>.json and the arm OOF parquets into
# oof/ under the names `code/q09.py` reads (`<arm>--<fold>.parquet`).
set -euo pipefail
WHAT="${1:?july_loo | december | january | ranking}"
SLUG="prc-chain-$(echo "$WHAT" | tr '_' '-')"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-040-chain"
TMP="$(mktemp -d)"
"$HOME/.local/bin/kaggle" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" >/dev/null
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/oof"
python3 - "$TMP/$SLUG.log" "$EXP/logs/$WHAT.log" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as fh:
    for rec in raw:
        fh.write(rec.get("data", ""))
PY
[ -e "$TMP/port.json" ] && cp "$TMP/port.json" "$EXP/results/port-$WHAT.json"
for p in "$TMP"/*.parquet; do [ -e "$p" ] && cp "$p" "$EXP/oof/$(basename "$p")"; done
echo "logs/$WHAT.log  $(wc -l < "$EXP/logs/$WHAT.log") lines"
ls -1 "$TMP"
