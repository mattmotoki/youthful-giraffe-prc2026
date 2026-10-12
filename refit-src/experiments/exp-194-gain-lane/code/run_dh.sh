#!/usr/bin/env bash
# exp-194 lever (b): the DH fits in sequence on remote. Launch the runner under the hold guard (the guard stops and
# continues the whole process group):
#   cd ${PRC_FINAL_ROOT:-/path/to/final}/gainlane; setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh \
#     ${PRC_LOG_ROOT:-/path/to/logs}/gainlane-dh-<tag>.log bash code-dh/run_dh.sh <tag> [<opdi_cov_2026.json>] \
#     >/dev/null 2>&1 < /dev/null &
# With the coverage file: stage DHo (six OPDI columns thinned to the 2026 coverage); without: stage DH.
# Order: jjx-02 (EDDF February read), jjrank (January/July read and the 2026 part; boosters saved), jjx-06.
set -eu
TAG=$1
COV=${2:-}
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane
PY=${PRC_FINAL_ROOT:-/path/to/final}/finalv/venv312/bin/python
C159=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work/code
OUT=$G/dh-$TAG
export EXP159_SET=12 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
for split in jjx-02 jjrank jjx-06; do
  set -- --c159 $C159 --split $split --seeds 601-603 --threads 2 --out $OUT
  if [ "$split" = jjrank ]; then set -- "$@" --save-models; fi
  if [ -n "$COV" ]; then set -- "$@" --opdi-cov "$COV"; fi
  echo "run_dh: $split start $(date -u +%FT%TZ)"
  $PY -I $G/code-dh/dh_fit.py "$@"
  echo "run_dh: $split done $(date -u +%FT%TZ)"
done
echo "run_dh: all done"
