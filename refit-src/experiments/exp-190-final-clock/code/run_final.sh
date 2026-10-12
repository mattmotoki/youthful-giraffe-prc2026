#!/usr/bin/env bash
# exp-190: the final-mode chain with its inputs read by MANIFEST path and full SHA-256, run from the repo root of a
# snapshot (remote only).
#   bash experiments/exp-190-final-clock/code/run_final.sh [step ...]    (steps of run_chain.sh; default its final list)
# Inputs: finalroute's features-r4c and proxy-features-r4c (exp-189 MANIFEST, 02c12cc51); finaldata's opdi-first of
# the final file and its February/June harvest folders (exp-185 MANIFEST); the saved exp-155 opdi-first, whose 2025
# rows the released frame used. opdi_final.py joins the two OPDI files (the final file has no 2025 row).
set -euo pipefail
ROOT=$(pwd)
CODE=$ROOT/experiments/exp-190-final-clock/code
FINAL_ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
PY=${PY:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python}
R4C=$FINAL_ROOT/finalroute/inputs/r4c-final
FD=$FINAL_ROOT/finaldata
SAVED_OPDI=${PRC_OPDI_ROOT:-/path/to/prc-exp155-opdi}/r2/inputs/complete/opdi-first.parquet
OPDI_ALL=$FINAL_ROOT/finalclock/inputs/opdi-first-all.parquet
check() { [ "$(sha256sum < "$1" | cut -c1-64)" = "$2" ] || { echo "$1 is not $2" >&2; exit 1; }; echo "sha256 ok $2 $1"; }
check "$R4C/features-r4c.parquet" 39466f229682c802f6f39204341dae58fae26164d82e9cb8b58d4b8629054302
check "$R4C/proxy-features-r4c.parquet" e445405e36b2220ace34c8ec0956738225b6dfe5d745d964207c0c79a9755ccf
check "$FD/opdi/first-final/opdi-first.parquet" 753c3ee9bdd6eb2965e632f1469be91a03534b45d55f7eb8ce154e0b96062aae
check "$SAVED_OPDI" 84b5c5ea3957ba1e1635c3db248f468ca4ec89d4cbfd501d82bda896a288bef0
check "$FD/kinput/prc-harvest-2026-02.sha256" f1b804a72d8600ca76c059d82c11fa965c2b909d46f726a9fb95b12032b4d319
check "$FD/kinput/prc-harvest-2026-06.sha256" 61a9aaee57366298bbd38f7dc2c287db81b2647f48e3836c55031bb0e7b506ef
for m in 02 06; do
  (cd "$FD/kinput/prc-harvest-2026-$m" && sha256sum -c --quiet "../prc-harvest-2026-$m.sha256")
  echo "sha256 ok: the files of prc-harvest-2026-$m"
done
[ -e "$OPDI_ALL" ] || $PY "$CODE/opdi_final.py" --saved "$SAVED_OPDI" --final "$FD/opdi/first-final/opdi-first.parquet" \
  --out "$OPDI_ALL"
export EXP159_FEATURES=$R4C/features-r4c.parquet EXP159_PROXY=$R4C/proxy-features-r4c.parquet EXP159_OPDI=$OPDI_ALL
export EXP159_HARVEST_DIRS=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work/kinput:$FD/kinput
exec bash "$CODE/run_chain.sh" final "$@"
