#!/usr/bin/env bash
# exp-045-harvest: pull one re-derive session and replace that month's summaries.
#   bash experiments/exp-045-harvest/code/pull_rederive.sh 2026-01-gate
# Writes analysis/rederived/<day>.json so manifest.json can say which rule wrote each
# day's summaries, and overwrites artifacts/exp-045/summaries/<airport>/<day>.parquet.
# The landing itself is code/land_rederive.py, shared with the laptop re-derive.
set -euo pipefail
NAME="${1:?month or kernel name}"
SLUG="prc-rederive-$NAME"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-045-harvest"
ART="$ROOT/artifacts/exp-045"
mkdir -p "$EXP/work" "$EXP/analysis/rederived" "$EXP/logs" "$ART/summaries" "$ART/callsigns"
TMP="$(mktemp -d "$EXP/work/rd-XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
"$HOME/.local/bin/kaggle" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" -q --page-size 200 \
  --file-pattern '^(segments|callsigns|rederive)' >/dev/null
"$ROOT/.venv/bin/python" "$EXP/code/land_rederive.py" "$TMP" "$ART" "$EXP" \
  "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" "$TMP/$SLUG.log" "$EXP/logs/rederive-$NAME.log"
