#!/bin/sh
# After the six s7 fits (code/s7_fit_queue.sh): part s7c, its held fold vector and the reads (s7_part.py), then the
# January/July parity read of s7c against the saved s7c (exp-186 parity.py, refit kind).
# Usage, from a repo snapshot that holds prc/, experiments/exp-186-final-core/code and this code, with PRC_DATA_DIR:
#   sh experiments/exp-191-final-v/code/s7_finish.sh <python> <s7 run tree> <parts dir> <v73-parts.parquet>
set -eu
PY=$1; RUN=$2; PARTS=$3; REFS=$4
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PRC_RANKING_MODE=final
: "$PRC_DATA_DIR"
echo "s7 finish start $(date -u +%FT%TZ)"
for m in full held; do
  test "$(ls "$RUN"/work/seq7/s7-$m-*.npy | wc -l)" -eq 3
  for s in 1 2 3; do test -s "$RUN/work/seq7/s7-$m-$s.npy"; done
done
test ! -e "$PARTS/s7c.parquet"
test ! -e "$RUN/work/w2/s7_fold.parquet"
$PY experiments/exp-191-final-v/code/s7_part.py "$RUN" "$PARTS"
$PY experiments/exp-186-final-core/code/parity.py --refs "$REFS" --new "$PARTS/s7c.parquet" --pair s7c=saved:s7c \
  --kind refit --out "$PARTS/s7c-parity.json"
sha256sum "$RUN"/work/seq7/s7-*.npy "$RUN/work/w2/s7_fold.parquet" "$RUN/work/w2/s7_ranking.parquet" \
  "$PARTS/s7c.parquet" "$PARTS/s7c-held-fold.parquet" "$PARTS/s7c-reads.json" "$PARTS/s7c-parity.json"
echo "end $(date -u +%FT%TZ)"
