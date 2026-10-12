#!/usr/bin/env bash
# exp-194 R9 (SPEC-R9.md): the --stages record (r9_dh_stages.py) and check_final_candidate.py on the dh files of
# chain f1 (base = final-iii-dh c4511650, as the reference; r9). One short job on remote x86 under the hold guard:
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <code dir>/r9_dh_check.sh <code dir> <snapshot> <chain dir> >/dev/null 2>&1 < /dev/null &
# <code dir> holds this file and r9_dh_stages.py (git archive of experiments/exp-194-gain-lane/code); <snapshot> is the
# chain's snapshot (check_final_candidate.py); <chain dir> is ${PRC_FINAL_ROOT:-/path/to/final}/gainlane/r9-<tag>.
set -euo pipefail
CODE=$1; SNAP=$2; G=$3
PY=${PRC_FINAL_ROOT:-/path/to/final}/finalv/venv312/bin/python
export PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1
echo "r9_dh_check start $(date -u +%FT%TZ) pid $$"
for v in base r9; do
  test ! -e "$G/dh-$v/dh-file-stages-record.json" && test ! -e "$G/dh-$v/dh-check.json"
  $PY -I "$CODE/r9_dh_stages.py" "$G/root-$v/finalv/parts/v-file-stages-record.json" \
    "$G/root-$v/finalv/youthful-giraffe_final.parquet" "$G/dh-$v/youthful-giraffe_final.parquet" \
    "$PRC_DATA_DIR/final" "$G/dh-$v/dh-file-stages-record.json"
  (cd "$SNAP" && $PY -I experiments/exp-186-final-core/code/check_final_candidate.py \
    "$G/dh-$v/youthful-giraffe_final.parquet" "$SNAP" --stages "$G/dh-$v/dh-file-stages-record.json" \
    --out "$G/dh-$v/dh-check.json") || echo "check $v: blocking failure (see json)"
done
sha256sum "$G"/dh-*/*.json "$G"/dh-*/youthful-giraffe_final.parquet
echo "r9_dh_check done $(date -u +%FT%TZ)"
