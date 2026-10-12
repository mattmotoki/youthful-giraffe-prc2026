#!/usr/bin/env bash
set -eu
root=${PRC_OPDI_ROOT:-/path/to/prc-exp155-opdi}
r2="$root/r2"
py=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONPATH="$r2/source"
export PRC_VALIDATION_DIR="$root/reference/frozen"
export PRC_DATA_DIR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/data/raw
trap 'rc=$?; echo "$rc" > "$r2/logs/screen-lane.exit"' EXIT
exec 9> "$r2/logs/compute.lock"
flock -n 9
if [ ! -f "$r2/inputs/complete/opdi-first.json" ]; then
  taskset -c 14,15,78,79 nice -n19 "$py" -u "$r2/code/prepare_inputs.py" --root "$root" --out "$r2/inputs/complete" --episode first --member-input-root ${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work/kinput > "$r2/logs/prepare-first.log" 2>&1
fi
fit() {
  learner="$1"; arm="$2"; split="$3"
  if [ "$learner" = L ]; then control=r2-L-pack-008-repair; else control=r2-Lp-pack-008; fi
  if [ -f "$r2/out/$learner/initial/${learner}155-$arm-$split.json" ]; then return; fi
  taskset -c 14,15,78,79 nice -n19 "$py" -u "$r2/code/fit_member.py" \
    --source-root "$r2/source" --input-root ${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work/kinput \
    --extra-inputs "$r2/inputs/complete" --control-pack "${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp153-work/out/$control" \
    --reference "$r2/inputs/r2_reference.json" --out "$r2/out/$learner/initial" --cache "$r2/cache" \
    --learner "$learner" --arm "$arm" --fit "$split" > "$r2/logs/$learner-$arm-$split.log" 2>&1
}
fit L control janjul
fit L opdi janjul
fit Lp opdi janjul
printf 'first arms complete\n'
