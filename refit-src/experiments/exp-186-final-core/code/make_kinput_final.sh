#!/usr/bin/env bash
# exp-186: a final EXP148_KINPUT root for the mlp chain (exp-164 mlp008/prepare.py --fit final), in the board
# layout of prc-exp148-work/kinput and built from links only.
#   bash experiments/exp-186-final-core/code/make_kinput_final.sh <root>
# KINPUT_MANIFEST  exp-185 MANIFEST.tsv (required; the snapshot holds no exp-185); its SHA-256 must equal
#                  KINPUT_MANIFEST_SHA (default: the MANIFEST of 5cf1388c8)
# KINPUT_FINAL     finaldata root (default ${PRC_FINAL_ROOT:-/path/to/final}/finaldata)
# KINPUT_BOARD     board kinput root (default ${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work/kinput); only its
#                  point files of the days the final points_expected.json shares with the board are linked
# make_kinput_final.py names every lookup of the final prepare, checks each final file against its MANIFEST row,
# links, repeats every lookup on the root and writes <root>/KINPUT-ROOT.json. Stops when <root> exists or on any
# difference or missing file. Python stdlib only.
set -euo pipefail
ROOT=${1:?usage: make_kinput_final.sh <root>}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
MANIFEST=${KINPUT_MANIFEST:?set KINPUT_MANIFEST to the exp-185 MANIFEST.tsv}
MANIFEST_SHA=${KINPUT_MANIFEST_SHA:-d3152ef749850fa4d53cb33df052cac78c4494c8299ba2bf6707f43bd38b4e16}
FINAL=${KINPUT_FINAL:-${PRC_FINAL_ROOT:-/path/to/final}/finaldata}
BOARD=${KINPUT_BOARD:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work/kinput}
digest() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
if [ -e "$ROOT" ] || [ -L "$ROOT" ]; then echo "$ROOT exists; stop" >&2; exit 1; fi
[ -f "$MANIFEST" ] || { echo "no MANIFEST $MANIFEST; stop" >&2; exit 1; }
[ "$(digest "$MANIFEST")" = "$MANIFEST_SHA" ] || { echo "$MANIFEST SHA-256 is not $MANIFEST_SHA; stop" >&2; exit 1; }
[ -d "$FINAL" ] && [ -d "$BOARD" ] || { echo "no $FINAL or no $BOARD; stop" >&2; exit 1; }
case "$ROOT" in /*) ;; *) ROOT=$(pwd)/$ROOT;; esac
PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/make_kinput_final.py" "$ROOT" --manifest "$MANIFEST" --final "$FINAL" --board "$BOARD"
