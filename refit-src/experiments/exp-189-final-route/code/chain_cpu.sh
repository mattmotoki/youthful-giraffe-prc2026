#!/usr/bin/env bash
# exp-189: the CPU members of the final route, in the final mode, after the jobs they wait for end
# (one model fit of mine at a time on the grant's CPUs). Run it under arc_holdguard.sh.
#   chain_cpu.sh <src tree> <out dir> <threads> <wait log> [<wait log> ...] -- <member> [<member> ...]
# Each wait log is an arc_holdguard log; the chain starts when every one holds "holdguard: exit".
# Then the 2025 caches of the tree must equal the frozen tree's files (make_src's record).
set -uo pipefail
SRC=$1; OUT=$2; THR=$3; shift 3
logs=()
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do logs+=("$1"); shift; done
[ "$#" -gt 1 ] || { echo "usage: chain_cpu.sh <src> <out> <threads> <log>... -- <member>..."; exit 2; }
shift
for l in "${logs[@]}"; do
  until grep -q 'holdguard: exit' "$l" 2>/dev/null; do sleep 60; done
  echo "$(date -u +%FT%TZ) waited: $l: $(grep 'holdguard: exit' "$l" | tail -1)"
done
[ -s "$SRC/artifacts/CACHES.sha256" ] || { echo "no $SRC/artifacts/CACHES.sha256"; exit 1; }
(cd ${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src && sha256sum -c --quiet "$SRC/artifacts/CACHES.sha256") \
  || { echo "caches differ from the frozen tree"; exit 1; }
(cd "$SRC" && sha256sum -c --quiet artifacts/CACHES.sha256) || { echo "tree caches differ from the record"; exit 1; }
exec bash "$SRC/experiments/exp-189-final-route/code/run_member.sh" "$SRC" "$OUT" "$THR" "$@"
