#!/bin/sh
# PORT-S7X9 step b: s7_finish.sh for N seeds. Part s7c of an N-seed s7 tree (seeds 1..N of each mode), its held fold
# vector and the reads (s7_part.py), then the January/July parity read against the saved s7c (exp-186 parity.py,
# refit kind). The parts directory is new, so part s7c of 10-09 and its record stay as they are.
# Usage, from a repo snapshot that holds prc/, experiments/exp-186-final-core/code and this code, with PRC_DATA_DIR:
#   sh experiments/exp-191-final-v/code/s7n_finish.sh <python> <s7 run tree> <parts dir> <v73-parts.parquet> <N>
set -eu
PY=$1; RUN=$2; PARTS=$3; REFS=$4; N=$5
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PRC_RANKING_MODE=final
: "$PRC_DATA_DIR"
echo "s7 finish N=$N start $(date -u +%FT%TZ)"
for m in full held; do
  test "$(ls "$RUN"/work/seq7/s7-$m-*.npy | wc -l)" -eq "$N"
  for s in $(seq 1 "$N"); do test -s "$RUN/work/seq7/s7-$m-$s.npy"; done
done
test ! -e "$PARTS"
mkdir -p "$PARTS"
test ! -e "$RUN/work/w2/s7_fold.parquet" && test ! -e "$RUN/work/w2/s7_ranking.parquet"
$PY experiments/exp-191-final-v/code/s7_part.py "$RUN" "$PARTS"
$PY experiments/exp-186-final-core/code/parity.py --refs "$REFS" --new "$PARTS/s7c.parquet" --pair s7c=saved:s7c \
  --kind refit --out "$PARTS/s7c-parity.json"
sha256sum "$RUN"/work/seq7/s7-*.npy "$RUN/work/w2/s7_fold.parquet" "$RUN/work/w2/s7_ranking.parquet" \
  "$PARTS/s7c.parquet" "$PARTS/s7c-held-fold.parquet" "$PARTS/s7c-reads.json" "$PARTS/s7c-parity.json"
echo "end $(date -u +%FT%TZ)"
