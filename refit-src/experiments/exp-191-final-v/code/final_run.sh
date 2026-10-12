#!/bin/sh
# Final-mode V steps (assignment item 3) in a run tree whose release/ is a final export copy of finalplan
# (export_copy.py + assemble192.py --pack-out: release/ZABD-PACK.json of the final files, 670,790 ranking rows).
# Order and threads of release/v73_stage.py: chain dump (byte check against the pack's candidate and v64 hashes,
# same machine), chain_ids, features, c5 (EXT, LONG, LIRF, NOAC, ROLL), w2 base, L2C (NT=8), stages record.
# ORD-S runs later with the s7c part (ords_run.sh): it needs the s7 fits.
# Usage (from the run tree): sh final_run.sh <python> <src>
set -eu
PY=$1; SRC=$2
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NT=8 PRC_RANKING_MODE=final
unset CHAIN_REF
echo "mode final start $(date -u +%FT%TZ)"
sha256sum release/ZABD-PACK.json
$PY iso/dump_chain.py --data-root "$PWD/input-root" --out work/chain
$PY chain_ids.py . --official fold,ranking
$PY iso/features.py
$PY iso/features.py months
$PY iso/candidate.py --steps ext,long,lirf,noac,roll --out outputs/iso-c5.parquet
$PY w2/base.py
$PY w2/l2c.py ${L2C_S7:+--s7 "$L2C_S7"} --out outputs/w2-l2c.parquet   # L2C_S7: candidate A only (v_run.sh)
$PY "$SRC/stages_record.py" . outputs/stages-c5-l2c.json
sha256sum outputs/iso-c5.parquet outputs/w2-l2c.parquet outputs/stages-c5-l2c.json work/chain/fold.parquet \
  work/chain/ranking.parquet work/w2/chain_ids_fold.parquet work/w2/chain_ids_ranking.parquet
echo "end $(date -u +%FT%TZ)"
