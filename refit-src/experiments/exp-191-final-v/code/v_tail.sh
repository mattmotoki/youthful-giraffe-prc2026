#!/bin/sh
# Final-mode V tail in a run tree after final_run.sh: the s7c member files from the s7 run tree (s7_part.py), ORD-S
# (--names s7, K = 0.5 on the LIRF July-scope rows), the stages record, part v and the V file.
# Usage (from the run tree): sh v_tail.sh <python> <src> <s7 run tree> <parts dir> <V file>
set -eu
PY=$1; SRC=$2; S7=$3; PARTS=$4; VFILE=$5
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PRC_RANKING_MODE=final
echo "v tail start $(date -u +%FT%TZ)"
# the member files are aligned to the s7 tree's chain_ids; both trees write the official order
for s in fold ranking; do
  cmp "$S7/work/w2/chain_ids_$s.parquet" "work/w2/chain_ids_$s.parquet"
  test ! -e "work/w2/s7_$s.parquet"
  cp "$S7/work/w2/s7_$s.parquet" "work/w2/s7_$s.parquet"
done
$PY w2/ords.py --on work/w2/l2c --names s7 --lirf-jul 0.5 --save-npz work/w2/s7c-final.npz --out outputs/w2-l2c-s7c.parquet
$PY "$SRC/stages_record.py" . outputs/stages-v.json
$PY "$SRC/v_part.py" . "$PARTS" "$VFILE"
cp outputs/stages-v.json "$PARTS/v-stages-record.json"
sha256sum work/w2/s7_fold.parquet work/w2/s7_ranking.parquet work/w2/s7c-final.npz outputs/w2-l2c-s7c.parquet \
  "$PARTS/v.parquet" "$PARTS/v-stages-record.json" "$VFILE"
echo "end $(date -u +%FT%TZ)"
