#!/usr/bin/env bash
set -euo pipefail
mlp_work=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp164-work
mlp_job=${1:?prepare, full, physical or screen-full/screen-physical}
mlp_attempt=${2:-1}
mlp_name=$mlp_job
if [[ "$mlp_attempt" != 1 ]]; then
    printf -v mlp_name '%s-%02d' "$mlp_job" "$mlp_attempt"
fi
case "$mlp_job" in prepare|full|physical|screen-full|screen-physical) ;; *) exit 2 ;; esac
if [[ -e "$mlp_work/results/mlp-007/$mlp_name-status.json" ]]; then
    echo 'Phase has a launch record; inspect it before a retry.' >&2
    exit 1
fi
if [[ ! -f "$mlp_work/results/mlp-007/WAKE-CONTRACT.json" ]]; then
    echo 'Verified wake contract is required before a detached launch.' >&2
    exit 1
fi
mlp_python=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python
if [[ "$mlp_job" == full || "$mlp_job" == physical ]]; then
    mlp_python=/usr/bin/python3
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv
fi
ulimit -v 50331648
nohup env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PRC_DATA_DIR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/data/raw \
    PRC_VALIDATION_DIR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work/source/experiments/exp-146-validation/frozen \
    nice -n 19 taskset -c 24-27,88-91 "$mlp_python" "$mlp_work/code/mlp007/supervise.py" --job "$mlp_job" --attempt "$mlp_attempt" \
    > "$mlp_work/logs/mlp007-$mlp_name-supervisor.log" 2>&1 < /dev/null &
mlp_pid=$!
printf '%s\n' "$mlp_pid" > "$mlp_work/logs/mlp007-$mlp_name.pid"
printf 'Phase %s PID %s\n' "$mlp_job" "$mlp_pid"
