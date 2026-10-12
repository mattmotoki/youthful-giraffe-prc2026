#!/usr/bin/env bash
set -euo pipefail
cd ${PRC_OPDI_ROOT:-/path/to/prc-exp155-opdi}
extract_pid=$(cat logs/extract-full.pid)
while kill -0 "$extract_pid" 2>/dev/null; do
    if ps -o stat= -p "$extract_pid" | grep -q '^Z'; then
        break
    fi
    sleep 60
done
if ! grep -q '^done$' logs/extract-full.log; then
    printf 'Extraction stopped before completion.\n' >&2
    exit 1
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
printf '%s\n' "$$" > logs/match-full.pid
exec taskset -c 14,15,78,79 nice -n 19 python3 -u code/match_full.py \
    --work . --raw ${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/data/raw \
    --class-root ${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/artifacts/exp-046/class \
    --out results/extract
