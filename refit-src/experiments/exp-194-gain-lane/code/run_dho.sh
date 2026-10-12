#!/usr/bin/env bash
# exp-194 lever (b), after hold h27: 2026 OPDI coverage (label-free), a DHo smoke, then the three DHo fits.
#   cd ${PRC_FINAL_ROOT:-/path/to/final}/gainlane; setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh \
#     ${PRC_LOG_ROOT:-/path/to/logs}/gainlane-dho-<tag>.log bash code-dh/run_dho.sh <tag> >/dev/null 2>&1 < /dev/null &
set -eu
TAG=$1
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane
PY=${PRC_FINAL_ROOT:-/path/to/final}/finalv/venv312/bin/python
OPDI=${PRC_FINAL_ROOT:-/path/to/final}/finaldata/opdi/first-final/opdi-first.parquet
OPDI_SHA=753c3ee9bdd6eb2965e632f1469be91a03534b45d55f7eb8ce154e0b96062aae
HEAD=${PRC_FINAL_ROOT:-/path/to/final}/finalclock/parts/head.parquet
HEAD_SHA=a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6
RAW=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw/final
[ -f $RAW/final_ranking.parquet ] || RAW=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw
COV=$G/opdi_cov_2026-$TAG.json
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
echo "run_dho: coverage start $(date -u +%FT%TZ)"
$PY -I $G/code-dh/dh_opdi_cov.py --opdi $OPDI --opdi-sha $OPDI_SHA --head $HEAD --head-sha $HEAD_SHA --raw $RAW --out $COV
echo "run_dho: smoke start $(date -u +%FT%TZ)"
EXP159_SET=12 $PY -I $G/code-dh/dh_fit.py --c159 ${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work/code --split jjx-02 \
  --seeds 601 --threads 2 --smoke 20000 --opdi-cov $COV --out $G/dh-smoke-$TAG
echo "run_dho: fits start $(date -u +%FT%TZ)"
bash $G/code-dh/run_dh.sh $TAG $COV
echo "run_dho: all done $(date -u +%FT%TZ)"
