#!/usr/bin/env bash
# exp-189: one route level-1 member fit `--fit final` in the final mode (or the ranking mode with
# MODE=ranking), from the remote source tree, one after another for each member named.
#   run_member.sh <src tree> <out dir> <threads> <member> [<member> ...]
# member: T-tailv4 | D1-winter | N1-nmtail | F1-fresh | L2-hub8725 | C2-cat. F1-fresh and C2-cat run
# on the GPU (the caller sets PRC_GPU=1 for arc_holdguard); the others on the CPU.
# Writes <out>/<member>-final.{parquet,json} (pilot.write, write-once) and one status line per member
# to <out>/status.tsv.
set -uo pipefail
SRC=$1; OUT=$2; THR=$3; shift 3
PY=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/.venv/bin/python
export PRC_SOURCE_ROOT=$SRC PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1
# the BLAS and OpenMP pools take the job's thread count (without it the T-tailv4 noise run held 16 threads
# on the 8 granted CPUs); LightGBM and CatBoost keep their own --threads argument
export OMP_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR MKL_NUM_THREADS=$THR NUMEXPR_NUM_THREADS=$THR
if [ "${MODE:-final}" = final ]; then export PRC_RANKING_MODE=final; else unset PRC_RANKING_MODE; fi
M=$SRC/experiments/exp-147-foldserve/code/members
cd "$SRC" || exit 1
mkdir -p "$OUT"
rc_all=0
for member in "$@"; do
  case $member in
    T-tailv4|D1-winter|N1-nmtail) cmd=("$PY" "$M/${member}_wp.py" --fit final --out "$OUT" --threads "$THR") ;;
    F1-fresh) cmd=("$PY" "$M/F1-fresh_wp.py" --fit final --out "$OUT" --threads "$THR" --device gpu) ;;
    L2-hub8725) cmd=("$PY" "$M/exp008_wp.py" --member L2-hub8725 --fit final --out "$OUT" --threads "$THR" --device cpu) ;;
    C2-cat) cmd=("$PY" "$M/exp008_wp.py" --member C2-cat --fit final --out "$OUT" --threads "$THR" --device gpu) ;;
    # N1's ranking frame: exp-003 build_all() writes artifacts/exp-003-anatomy/feat/<stem of the mode's
    # ranking file>.parquet (the 2025 months are present and skipped)
    N1-cache) cmd=("$PY" -c "import sys; sys.path[:0]=['$SRC', '$SRC/experiments/exp-003-anatomy/code']; import features; features.build_all()") ;;
    *) echo "unknown member $member"; exit 2 ;;
  esac
  t0=$(date -u +%s)
  echo "$(date -u +%FT%TZ) start $member mode=${PRC_RANKING_MODE:-ranking} ${cmd[*]}"
  "${cmd[@]}"
  rc=$?
  [ $rc -ne 0 ] && rc_all=1
  printf '%s\t%s\t%s\trc=%s\tseconds=%s\n' "$(date -u +%FT%TZ)" "$member" "${PRC_RANKING_MODE:-ranking}" "$rc" "$(( $(date -u +%s) - t0 ))" >> "$OUT/status.tsv"
  echo "$(date -u +%FT%TZ) end $member rc=$rc"
done
exit $rc_all
