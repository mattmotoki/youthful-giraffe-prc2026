#!/usr/bin/env bash
# exp-189: finish the remote source tree of the final route fits.
#   make_src.sh <src tree>   (the tree holds main's code from `git archive`, see SPEC.md)
# Copies the 2025 member caches of the frozen tree of the v73 member fits (FROZEN, fad0b6d5) into
# <src>/artifacts (copies, never links: a builder that appends to a cache file must not touch FROZEN),
# links <src>/data/raw to the final-mode data folder, and writes <src>/artifacts/CACHES.sha256.
set -euo pipefail
SRC=$1
FROZEN=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src
RAW=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw
cd "$FROZEN"
files=()
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  files+=("artifacts/exp-003-anatomy/feat/2025-$m.parquet" "artifacts/exp-008/frames/v2-2025-$m.parquet"
          "artifacts/exp-018-fresh/cache/2025-$m.parquet")
done
files+=("artifacts/exp-018-fresh/cache/vocab.json" "artifacts/exp-047/block/2025-01.parquet" "artifacts/exp-047/block/2025-07.parquet")
for f in "${files[@]}"; do
  [ -e "$SRC/$f" ] && { echo "exists: $SRC/$f"; exit 1; }
  mkdir -p "$SRC/$(dirname "$f")"
  cp -p "$f" "$SRC/$f"
  chmod a-w "$SRC/$f"
done
mkdir -p "$SRC/data"
[ -e "$SRC/data/raw" ] || ln -s "$RAW" "$SRC/data/raw"
(cd "$SRC" && sha256sum "${files[@]}") > "$SRC/artifacts/CACHES.sha256"
(cd "$FROZEN" && sha256sum -c --quiet "$SRC/artifacts/CACHES.sha256")
echo "copied ${#files[@]} files; sha256 equal to $FROZEN"
