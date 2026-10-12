#!/usr/bin/env bash
# Bit-stability separation for ALL FIVE members (desk, 2026-09-05).
#
# The question is not whether two thread counts agree - they demonstrably do not,
# and the original 16-thread conditions are unrecoverable - but whether a member
# is bit-stable when run TWICE at the DECLARED release thread count. 6 threads is
# the release setting because 6 is what compute allocation lets us verify twice.
#
# Short budgets where the producer allows one. Determinism is a property of the
# code path: if any step is non-deterministic the two runs diverge immediately and
# compound, so a short fit is a strictly more sensitive test per unit compute.
# Every run writes to its own name; no published artifact is touched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
P="$ROOT/.venv/bin/python"
export PRC_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6

# Replicates are resumable: pass "b" to run only the second, so a killed run
# does not throw away the first. (A monitor-launched job does not survive
# TaskStop even under nohup - the stop kills the process group.)
for rep in ${REPS:-a b}; do
  echo "########## replicate $rep"
  # tail_ranking.py has a HARDCODED output path, so it would clobber the live
  # member. Save and restore around it: safe by construction, not by my having
  # remembered to take a backup. This is the same failure that destroyed
  # 48acb7e7, so it gets a guard rather than a note.
  echo "--- T-tailv2"
  LIVE="$ROOT/artifacts/exp-012/pred/T-tailv2-ranking.parquet"
  cp -p "$LIVE" "$LIVE.sepsave"
  ( cd "$ROOT/experiments/exp-008-diversity" && PRC_GPU=1 "$P" tail_ranking.py >/dev/null )
  cp -p "$LIVE" "$ROOT/artifacts/exp-012/pred/SEPX-tail-$rep.parquet"
  mv "$LIVE.sepsave" "$LIVE"
  echo "    live T-tailv2 restored: $(sha256sum "$LIVE" | cut -c1-16)"
  echo "--- C2-cat (GPU 1)"
  ( cd "$ROOT/experiments/exp-008-diversity" && PRC_GPU=1 \
    "$P" run.py --submit --kind cat --name "SEPX-cat-$rep" --depth 8 \
        --rounds-m6 300 --rounds-m11 300 >/dev/null )
  echo "--- L2-hub (CPU)"
  ( cd "$ROOT/experiments/exp-008-diversity" && \
    "$P" run.py --submit --kind lgb --name "SEPX-hub-$rep" --variant plain \
        --huber-delta 1200 --rounds-m6 300 --rounds-m11 300 >/dev/null )
  echo "--- N1-nmtail"
  ( cd "$HERE" && PRC_SEP_SEEDS=7 PRC_SEP_ROUNDS=150 PRC_SEP_NAME="SEPX-nmtail-$rep" \
    "$P" nmtail_submit.py >/dev/null )
  echo "--- D1-winter (linear leaves; already declared non-deterministic across counts)"
  ( cd "$HERE" && "$P" winter_arm.py --name "SEPX-winter-$rep" --submit \
        --seeds 7 --ds-columns deice >/dev/null )
done

echo
echo "=== bit-stability at the declared release configuration (6 threads, one seed) ==="
cmp2() {  # cmp2 <label> <fileA> <fileB>
  local A B
  A=$(sha256sum "$2" 2>/dev/null | cut -d' ' -f1); B=$(sha256sum "$3" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$A" ] || [ -z "$B" ]; then echo "  $1: MISSING a replicate"; return; fi
  if [ "$A" = "$B" ]; then echo "  $1: BIT-STABLE     ${A:0:16}"
  else echo "  $1: NOT STABLE     ${A:0:16} vs ${B:0:16}"; fi
}
cd "$ROOT/artifacts"
cmp2 "T-tailv2 " exp-012/pred/SEPX-tail-a.parquet          exp-012/pred/SEPX-tail-b.parquet
cmp2 "C2-cat   " exp-008/pred/SEPX-cat-a-ranking.parquet   exp-008/pred/SEPX-cat-b-ranking.parquet
cmp2 "L2-hub   " exp-008/pred/SEPX-hub-a-ranking.parquet   exp-008/pred/SEPX-hub-b-ranking.parquet
cmp2 "N1-nmtail" exp-012/pred/SEPX-nmtail-a-ranking.parquet exp-012/pred/SEPX-nmtail-b-ranking.parquet
cmp2 "D1-winter" exp-012/pred/SEPX-winter-a-ranking.parquet exp-012/pred/SEPX-winter-b-ranking.parquet
