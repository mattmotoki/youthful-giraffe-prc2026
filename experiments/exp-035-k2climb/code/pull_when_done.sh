#!/usr/bin/env bash
# exp-035-k2climb: wait for a Kaggle kernel to leave RUNNING, then pull its output.
#
#   bash pull_when_done.sh <slug>
#
# Writes logs/<slug>.log, results/*.json and oof/<name>.parquet under the names
# the kernel already gave them (`<tag>.parquet`), so `code/land.py` finds them
# with no renaming step. exp-027's version renamed on the way in and needed a
# second script to undo it.
set -u
K="$HOME/.local/bin/kaggle"
SLUG="$1"
ROOT="/path/to/competitions/prc-data-challenge-2026"
EXP="$ROOT/experiments/exp-035-k2climb"
ART="$ROOT/artifacts/exp-035-k2climb/extras"
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/oof" "$EXP/candidates"
for i in $(seq 1 720); do
  ST="$("$K" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" 2>&1 | tail -1)"
  case "$ST" in
    *RUNNING*|*QUEUED*) echo "$(date -u +%H:%M:%S) $ST"; sleep 60; continue;;
  esac
  echo "$(date -u +%H:%M:%S) $ST"; break
done
TMP="$(mktemp -d)"
# The download can come back empty on a kernel that has only just left RUNNING,
# and it says nothing when it does: `prc-k2c-phi-dj` filed zero files on
# 2026-09-08 and its two folds had to be pulled by hand. Retry until files
# appear, and shout if they never do -- a silent empty pull looks exactly like
# a kernel that produced nothing.
for try in 1 2 3 4 5 6 7 8 9 10; do
  "$K" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" >/dev/null 2>&1
  [ -n "$(ls -A "$TMP" 2>/dev/null)" ] && break
  echo "$(date -u +%H:%M:%S) output empty, retry $try"; sleep 20
done
if [ -z "$(ls -A "$TMP" 2>/dev/null)" ]; then
  echo "FAILED: $SLUG left RUNNING but downloaded no files after 10 tries"; exit 1
fi
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
for f in "$TMP"/*.parquet; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in
    # arm 3's module cache is 900 MB of feature columns, not a measurement:
    # it goes to the git-ignored artifacts tree, never into the record.
    *-train.parquet|*-valid.parquet) mkdir -p "$ART"; cp "$f" "$ART/$(basename "$f")";;
    *rank*) cp "$f" "$EXP/candidates/$(basename "$f")";;
    *)      cp "$f" "$EXP/oof/$(basename "$f")";;
  esac
done
# The assignment asks for every kernel's log and metadata under `kernels/`, so
# each kernel's own directory gets its run record beside the source that ran.
ARM="${SLUG#prc-k2c-}"
if [ -d "$EXP/kernels/$ARM" ]; then
  cp "$EXP/logs/$SLUG.log" "$EXP/kernels/$ARM/run.log"
  for f in "$TMP"/port-*.json "$TMP"/extras.json; do
    [ -e "$f" ] && cp "$f" "$EXP/kernels/$ARM/$(basename "$f")"
  done
fi
echo "pulled $SLUG; files:"; ls -la "$TMP" | tail -8
