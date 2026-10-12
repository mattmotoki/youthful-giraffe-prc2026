#!/usr/bin/env bash
# exp-027-step: pull the K2 kernel's outputs into this experiment's record.
#   bash experiments/exp-027-step/code/pull_k2.sh
# Writes logs/k2-window.log, results/*.json and oof/k2-window-july_loo.parquet.
set -euo pipefail
SLUG="prc-k2-window"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-027-step"
TMP="$(mktemp -d)"
"$HOME/.local/bin/kaggle" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" >/dev/null
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/oof"
python3 - "$TMP/$SLUG.log" "$EXP/logs/k2-window.log" <<'PY'
import json, sys
try:
    raw = json.load(open(sys.argv[1]))
    txt = "".join(r.get("data", "") for r in raw)
except Exception:
    txt = open(sys.argv[1], errors="replace").read()
open(sys.argv[2], "w").write(txt)
PY
for f in "$TMP"/*.json; do [ -e "$f" ] && cp "$f" "$EXP/results/"; done
for f in "$TMP"/*.parquet; do [ -e "$f" ] && cp "$f" "$EXP/oof/k2-window-$(basename "$f")"; done
echo "pulled into $EXP"; ls -la "$EXP/logs/k2-window.log"
grep -E "^july_loo|rmse=|TAG " "$EXP/logs/k2-window.log" | tail -8 || true
