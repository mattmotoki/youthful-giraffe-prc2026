#!/usr/bin/env bash
# Unit 008: start one detached step on remote with the lane limits; write its pid and a status file.
# usage: launch.sh NAME CPUS STEP [ARGS...]   (STEP: prepare, check, fit, apply, lane_queue, check_fit, fit_bag, check_bag, bag_vectors, fit_sweep, sweep_vectors, vectors6_010, check_sweep, vectors011, pack_level2_011, mix012_vectors, pack_level2_012)
# CPUS is a subset of the grant 24-27,88-91 (two physical cores and their siblings per job).
set -euo pipefail
w=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp164-work
name=${1:?name}; cpus=${2:?cpus}; step=${3:?step}; shift 3
case "$cpus" in 24,25,88,89|26,27,90,91|24-27,88-91) ;; *) echo "cpus outside the grant" >&2; exit 2 ;; esac
case "$step" in prepare|check|fit|apply|lane_queue|check_fit|fit_bag|check_bag|bag_vectors|fit_sweep|sweep_vectors|vectors6_010|check_sweep|vectors011|pack_level2_011|mix012_vectors|pack_level2_012) ;; *) echo "unknown step" >&2; exit 2 ;; esac
log=$w/logs/mlp008-$name.log
if [[ -e "$log" ]]; then echo "log exists: $log (logs are write-once)" >&2; exit 1; fi
py=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python
if [[ "$step" == fit || "$step" == apply || "$step" == check_fit || "$step" == fit_bag || "$step" == check_bag || "$step" == bag_vectors || "$step" == fit_sweep || "$step" == sweep_vectors || "$step" == vectors6_010 || "$step" == check_sweep || "$step" == vectors011 || "$step" == mix012_vectors ]]; then py=/usr/bin/python3; fi
if [[ "$step" == lane_queue ]]; then py=/usr/bin/python3; fi
ulimit -v 50331648
setsid nohup env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PRC_DATA_DIR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/data/raw \
    PRC_VALIDATION_DIR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work/source/experiments/exp-146-validation/frozen \
    nice -n 19 taskset -c "$cpus" "$py" "$w/code/mlp008/$step.py" "$@" > "$log" 2>&1 < /dev/null &
echo $! > "$w/logs/mlp008-$name.pid"
echo "started $name pid $(cat "$w/logs/mlp008-$name.pid")"
