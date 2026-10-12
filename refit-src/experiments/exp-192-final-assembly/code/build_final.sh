#!/usr/bin/env bash
# exp-192 final build (finalplan): parts zabd and r, the Z and R files, the final pack in an export copy, and the
# checks (pack replay, check_final_candidate --stages, January/July parity). The MANIFEST `command` of these parts.
#
# Run on remote x86 through the hold guard, from a snapshot of the repo at <SRC> (git archive of
# experiments/exp-146-validation/frozen, experiments/exp-186-final-core/code, experiments/exp-192-final-assembly and
# prc, with data/raw -> $PRC_DATA_DIR):
#   bash <SRC>/experiments/exp-192-final-assembly/code/build_final.sh <SRC> <sources json> <name> [r-only]
# r-only: the R part and the R file before the member parts land (assemble192.py --r-only: member placeholders; no
# zabd part, Z file or pack). The R values do not read the members (REPORT section 10).
# Rebuild tolerance: deterministic SHA-256 on the same machine and threads (x86, OMP 2 / BLAS 1; REPORT section 4).
set -euo pipefail
SRC=$1; SOURCES=$2; NAME=$3; RONLY=${4:-}
test -z "$RONLY" || test "$RONLY" = r-only
ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
D=$ROOT/finalplan
PY=${PRC_PY:-$ROOT/finalv/venv312/bin/python}
export PRC_DATA_DIR=${PRC_DATA_DIR:-$ROOT/finalcore/raw} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REFS=${PRC_V73_REFS:-$ROOT/finalcore/refs/v73-parts.parquet}
E=experiments/exp-192-final-assembly
P=${PRC_PARTS_DIR:-$D/parts}/$NAME
cd "$SRC"
test ! -e "$P" && test ! -e "$D/export-$NAME" && test ! -e "$D/runs/$NAME.json"
stamp() { date -u +%FT%TZ; }
if [ -n "$RONLY" ]; then
  echo "assembly (R only) $(stamp)"
  $PY -I $E/code/assemble192.py --mode final --r-only --repo "$SRC" --export $D/export-d07bf4a0a621 \
    --sources "$SOURCES" --parts-dir "$P" --out $D/runs/$NAME.json
  FILES="r-file"; PARTS="r"
else
  echo "export copy $(stamp)"
  $PY -I $E/code/export_copy.py $D/export-d07bf4a0a621 $D/export-$NAME "$SRC" > $D/runs/$NAME-copy.json
  echo "assembly $(stamp)"
  $PY -I $E/code/assemble192.py --mode final --repo "$SRC" --export $D/export-d07bf4a0a621 --sources "$SOURCES" \
    --parts-dir "$P" --pack-out $D/export-$NAME --out $D/runs/$NAME.json
  echo "pack check $(stamp)"
  PRC_RANKING_MODE=final $PY -I $E/code/pack_check.py $D/export-$NAME $D/input-root $D/runs/$NAME-pack-check.json
  FILES="z-file r-file"; PARTS="zabd r"
fi
for f in $FILES; do
  echo "check $f $(stamp)"
  $PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$P/$f.parquet" "$SRC" \
    --stages "$P/$f-stages-record.json" --out $D/runs/$NAME-check-$f.json || echo "check $f: blocking failure (see json)"
done
echo "parity $(stamp)"
for part in $PARTS; do
  $PY -I experiments/exp-186-final-core/code/parity.py --refs "$REFS" --new "$P/$part.parquet" \
    --pair $part=saved:zabd-final --kind refit --out "$P/$part-parity.json"
done
echo "hashes $(stamp)"
sha256sum "$P"/*
test -n "$RONLY" || sha256sum $D/export-$NAME/release/ZABD-PACK.json
echo "done $(stamp)"
