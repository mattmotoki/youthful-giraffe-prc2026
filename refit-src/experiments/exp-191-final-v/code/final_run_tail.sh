#!/bin/sh
# The steps of final_run.sh from L2C on, in the same run tree (restart after a stopped L2C).
# Usage (from the run tree): sh final_run_tail.sh <python> <src>
set -eu
PY=$1; SRC=$2
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NT=8 PRC_RANKING_MODE=final
unset CHAIN_REF
echo "tail start $(date -u +%FT%TZ) OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-unset}"
sha256sum release/ZABD-PACK.json outputs/iso-c5.parquet work/w2/base_fold.parquet work/w2/base_ranking.parquet
$PY w2/l2c.py --out outputs/w2-l2c.parquet
$PY "$SRC/stages_record.py" . outputs/stages-c5-l2c.json
sha256sum outputs/iso-c5.parquet outputs/w2-l2c.parquet outputs/stages-c5-l2c.json work/chain/fold.parquet \
  work/chain/ranking.parquet work/w2/chain_ids_fold.parquet work/w2/chain_ids_ranking.parquet
echo "end $(date -u +%FT%TZ)"
