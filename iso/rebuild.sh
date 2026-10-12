#!/bin/sh
# Rebuild the isolated candidate from the v64 release export and the 14 official files.
# Usage: sh iso/rebuild.sh [out.parquet]                     c5 (default outputs/iso-c5.parquet)
#        STEPS=ext,long,lirf,noac sh iso/rebuild.sh outputs/iso-c3.parquet     c3
set -eu
cd "$(dirname "$0")/.."
OUT="${1:-outputs/iso-c5.parquet}"
STEPS="${STEPS:-ext,long,lirf,noac,roll}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
PY=.venv/bin/python
$PY iso/dump_chain.py --data-root "$PWD/input-root" --out work/chain
$PY iso/features.py
$PY iso/features.py months
$PY iso/candidate.py --steps "$STEPS" --out "$OUT"
shasum -a 256 "$OUT"
