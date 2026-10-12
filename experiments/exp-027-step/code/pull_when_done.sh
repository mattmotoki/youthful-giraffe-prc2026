#!/usr/bin/env bash
# exp-027-step: wait for a Kaggle kernel to leave RUNNING, then pull its output.
#   bash pull_when_done.sh <slug> <dest-prefix>
# Writes logs/<slug>.log, results/*.json and oof/<prefix>-<file>.parquet.
set -u
K="$HOME/.local/bin/kaggle"
SLUG="$1"; PREFIX="$2"
ROOT="/path/to/competitions/prc-data-challenge-2026"
EXP="$ROOT/experiments/exp-027-step"
for i in $(seq 1 480); do
  ST="$("$K" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" 2>&1 | tail -1)"
  case "$ST" in
    *RUNNING*|*QUEUED*) echo "$(date -u +%H:%M:%S) $ST"; sleep 60; continue;;
  esac
  echo "$(date -u +%H:%M:%S) $ST"; break
done
TMP="$(mktemp -d)"
"$K" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" >/dev/null 2>&1
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/oof"
python3 - "$TMP/$SLUG.log" "$EXP/logs/$SLUG.log" <<'PY'
import json, sys
try:
    raw = json.load(open(sys.argv[1]))
    txt = "".join(r.get("data", "") for r in raw)
except Exception:
    txt = open(sys.argv[1], errors="replace").read()
open(sys.argv[2], "w").write(txt)
PY
for f in "$TMP"/*.json; do [ -e "$f" ] && cp "$f" "$EXP/results/"; done
for f in "$TMP"/*.parquet; do [ -e "$f" ] && cp "$f" "$EXP/oof/$PREFIX-$(basename "$f")"; done
echo "pulled into $EXP; files:"; ls -la "$TMP" | tail -8
