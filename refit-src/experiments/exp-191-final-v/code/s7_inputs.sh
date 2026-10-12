#!/bin/sh
# The s7 member inputs (w2/rebuild.sh header) in a run tree, label-free except the 2025 training target.
# Usage (from the run tree): sh s7_inputs.sh <python>      mode from PRC_RANKING_MODE (ranking or final)
# Outputs: work/feat/months/*.parquet, work/feat/ranking.parquet (template order), work/w2/feat2_{2025,2026}.parquet,
#          work/seq7/*.npy + meta.json, work/w2/chain_ids_{fold,ranking}.parquet (official order; the chain
#          dump later asserts the same order).
set -eu
PY=$1
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "mode ${PRC_RANKING_MODE:-ranking} start $(date -u +%FT%TZ)"
$PY chain_ids.py . --official fold,ranking
$PY iso/features.py months
$PY iso/features.py ranking
$PY w2/feat2.py
$PY w2/seq/prepare.py --nb 64 --win 10800 --out work/seq7
sha256sum work/seq7/*.npy work/seq7/meta.json work/feat/ranking.parquet work/w2/feat2_2025.parquet work/w2/feat2_2026.parquet \
  work/w2/chain_ids_fold.parquet work/w2/chain_ids_ranking.parquet
echo "end $(date -u +%FT%TZ)"
