#!/usr/bin/env bash
# CatBoost reproducibility separation (desk, 2026-09-05).
#
# The question the release needs answered is NOT whether two thread counts agree
# -- the original 16-thread conditions are unrecoverable -- but whether a fixed
# DECLARED configuration is bit-stable when run twice. So each configuration runs
# twice at one seed and the two hashes are compared.
#
#   gpu6   GPU 1, PRC_THREADS=6      the D3-legal GPU configuration
#   gpu2   GPU 1, PRC_THREADS=2      the feeding-thread reading of compute allocation
#   cpu6   CPU,   PRC_THREADS=6      deterministic if anything is, and slow
#
# BUDGET: 300 rounds, not the shipped 5,325. Bit-reproducibility is a property of
# the code path, and if any single boosting round is non-deterministic the two
# runs diverge in the first few rounds and compound from there. A short fit is
# therefore a strictly MORE sensitive test per unit of compute than a long one,
# and it turns a 2-7 hour CPU arm into minutes. The winning configuration is
# re-confirmed at the full budget afterwards.
#
# Each run writes to its own --name, so no published artifact is touched.
set -euo pipefail
cd "$(dirname "$0")/../exp-008-diversity"
P=../../.venv/bin/python
R=300
run() {  # run <tag> <threads> <gpu|cpu> <rep>
  local tag=$1 th=$2 dev=$3 rep=$4
  local extra=""; [ "$dev" = cpu ] && extra="--cpu"
  export PRC_THREADS=$th OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th
  [ "$dev" = gpu ] && export PRC_GPU=1 || unset PRC_GPU
  echo "--- $tag rep$rep: threads=$th dev=$dev rounds=$R"
  $P run.py --submit --kind cat --name "SEP-$tag-$rep" --depth 8 \
      --rounds-m6 $R --rounds-m11 $R $extra
}
for tag_spec in "gpu6 6 gpu" "gpu2 2 gpu" "cpu6 6 cpu"; do
  set -- $tag_spec
  for rep in a b; do run "$1" "$2" "$3" "$rep"; done
done
echo "=== bit-stability at a fixed declared configuration ==="
cd ../../artifacts/exp-008/pred
for tag in gpu6 gpu2 cpu6; do
  A=$(sha256sum "SEP-$tag-a-ranking.parquet" 2>/dev/null | cut -d' ' -f1)
  B=$(sha256sum "SEP-$tag-b-ranking.parquet" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$A" ] || [ -z "$B" ]; then echo "  $tag: MISSING a run"; continue; fi
  [ "$A" = "$B" ] && echo "  $tag: BIT-STABLE   ${A:0:16}" || echo "  $tag: NOT STABLE   ${A:0:16} vs ${B:0:16}"
done
