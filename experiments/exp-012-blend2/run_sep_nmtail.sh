#!/usr/bin/env bash
# Re-test ONLY N1-nmtail after prc.nmtail VERSION 4 added the determinism flags.
# Same protocol as the five-member separation: two runs, 6 threads, one seed,
# short budget, each to its own name.
set -euo pipefail
cd "$(dirname "$0")"
export PRC_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
P=../../.venv/bin/python
for rep in a b; do
  echo "--- N1-nmtail V4 rep$rep"
  PRC_SEP_SEEDS=7 PRC_SEP_ROUNDS=150 PRC_SEP_NAME="SEPV4-nmtail-$rep" $P nmtail_submit.py >/dev/null
done
A=$(sha256sum ../../artifacts/exp-012/pred/SEPV4-nmtail-a-ranking.parquet | cut -c1-16)
B=$(sha256sum ../../artifacts/exp-012/pred/SEPV4-nmtail-b-ranking.parquet | cut -c1-16)
[ "$A" = "$B" ] && echo "  N1-nmtail V4: BIT-STABLE  $A" || echo "  N1-nmtail V4: NOT STABLE  $A vs $B"
