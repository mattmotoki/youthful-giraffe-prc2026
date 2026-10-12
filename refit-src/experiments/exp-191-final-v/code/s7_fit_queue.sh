#!/bin/bash
# The six s7 fits of the released recipe (w2/rebuild.sh: d 192, 4 layers, 16 epochs, amp), one seed per job,
# each job through arc_holdguard.sh with PRC_GPU=1 (GPU 1 only, the GPU lock frees between jobs).
# Waits for the inputs (work/seq7/meta.json and the 'end' line of the input log) and for a GPU 1 grant.
# Usage (detached, from anywhere): bash s7_fit_queue.sh <run> <torch python> <input log> [jobs...]
#   jobs default: full:1 held:1 full:2 held:2 full:3 held:3
# Outputs: <run>/work/seq7/s7-<mode>-<seed>.npy, logs ${PRC_LOG_ROOT:-/path/to/logs}/finalv-s7-<tag>-<mode>-<seed>.log
set -u
RUN=$1; PY=$2; INLOG=$3; shift 3
JOBS=${*:-full:1 held:1 full:2 held:2 full:3 held:3}
GUARD=${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh
GRANT=${PRC_TOOLS:-/path/to/tools}/grant.env
TAG=$(basename "$RUN")
stamp() { date -u +%FT%TZ; }
echo "$(stamp) queue start: $JOBS"
until [ -s "$RUN/work/seq7/meta.json" ] && grep -q '^end ' "$INLOG" 2>/dev/null; do
  if grep -q -E 'Traceback|holdguard: exit [1-9]' "$INLOG" 2>/dev/null; then echo "$(stamp) input build failed; stop"; exit 1; fi
  sleep 60
done
echo "$(stamp) inputs ready"
for job in $JOBS; do
  mode=${job%%:*}; seed=${job##*:}
  out=$RUN/work/seq7/s7-$mode-$seed.npy
  log=${PRC_LOG_ROOT:-/path/to/logs}/finalv-s7-$TAG-$mode-$seed.log
  while [ ! -s "$out" ]; do
    until grep -q '^PRC_GRANT_GPU=1$' "$GRANT"; do sleep 300; done
    echo "$(stamp) $mode $seed start"
    (cd "$RUN" && PRC_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
       bash "$GUARD" "$log" "$PY" w2/seq/fit.py --data work/seq7 --d 192 --layers 4 --epochs 16 --mode "$mode" \
       --seed "$seed" --amp --out "work/seq7/s7-$mode-$seed.npy")
    rc=$?
    echo "$(stamp) $mode $seed exit $rc, out $(ls -l "$out" 2>/dev/null | awk '{print $5}') bytes"
    if [ "$rc" = 3 ]; then echo "$(stamp) grant has no GPU now; retry"; sleep 300; continue; fi
    [ "$rc" = 0 ] && [ -s "$out" ] || { echo "$(stamp) fit failed; stop"; exit 1; }
  done
done
echo "$(stamp) queue end"
