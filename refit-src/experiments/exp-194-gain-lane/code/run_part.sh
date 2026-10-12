#!/usr/bin/env bash
# exp-194 lever (b): the DH part (one entry command). Run on remote under the hold guard:
#   cd ${PRC_FINAL_ROOT:-/path/to/final}/gainlane; setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh \
#     ${PRC_LOG_ROOT:-/path/to/logs}/gainlane-part-<tag>.log bash <snapshot>/experiments/exp-194-gain-lane/code/run_part.sh \
#     <tag> <fit tag> <scope> <w> [<snapshot>] >/dev/null 2>&1 < /dev/null &
# (the python entries are the ones next to this script: a snapshot's code folder or code-dh)
# <scope>: comma list of airports (EDDF, LIRF, LEMD), each with an optional own weight (EDDF:0.2), or "none"; the part
# gets w_pair = <w> (or the airport's own weight) on the typed no-head rows
# of the 2026 made-dark months of those airports (EDDF 1/2/6/7, LIRF 6/7, LEMD 6; dh_read made_dark_cells_2026) and 0
# elsewhere. pred_DH is the same for every scope. Optional row scope: FLAG=dark_a|dark_b with DARK_ROWS=<dh_dark_rows.py
# parquet> sets w only on the rows of those cells that carry the flag (FLAG=cell, the default: every row).
# Inputs: the DHo jjrank boosters of run_dho.sh <fit tag> (under DH_FIT_ROOT, default the gainlane folder), finalroute's
# final features-r4c, finalclock's head part, finaldata's final OPDI table and the raw final files, all under
# PRC_FINAL_ROOT (default ${PRC_FINAL_ROOT:-/path/to/final}); the part is written to $PRC_FINAL_ROOT/gainlane/parts/.
# Checks: the diff against V (part_diff.py; V_PART, V_SHA: finalv's part v) and, with a snapshot, the parity read.
set -eu
TAG=$1
FIT=$2
SCOPE=$3
W=$4
FLAG=${FLAG:-cell}
ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
G=$ROOT/gainlane
FITROOT=${DH_FIT_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}/gainlane}
C=$(cd "$(dirname "$0")" && pwd)
PY=${PRC_FINAL_ROOT:-/path/to/final}/finalv/venv312/bin/python
FEAT=$ROOT/finalroute/inputs/r4c-final/features-r4c.parquet
FEAT_SHA=39466f229682c802f6f39204341dae58fae26164d82e9cb8b58d4b8629054302
HEAD=$ROOT/finalclock/parts/head.parquet
HEAD_SHA=a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6
OPDI=$ROOT/finaldata/opdi/first-final/opdi-first.parquet
OPDI_SHA=753c3ee9bdd6eb2965e632f1469be91a03534b45d55f7eb8ce154e0b96062aae
RAW=$ROOT/finalcore/raw/final
[ -f $RAW/final_ranking.parquet ] || RAW=$ROOT/finalcore/raw
V_PART=${V_PART:-$ROOT/finalv/parts/v.parquet}
V_SHA=${V_SHA:-871b7e041810375aacb9ec024fc073dd11cd371c812d5526fd0962120167887a}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
WEIGHTS=$G/parts/dh-$TAG-weights.json
mkdir -p $G/parts
[ ! -e $WEIGHTS ] || { echo "run_part: $WEIGHTS exists; write-once"; exit 1; }
$PY -I -c "
import json, sys
months = {'EDDF': [1, 2, 6, 7], 'LIRF': [6, 7], 'LEMD': [6]}
scope, w, flag = sys.argv[1], float(sys.argv[2]), sys.argv[4]
items = [] if scope == 'none' else [x.split(':') for x in scope.split(',')]
aw = {x[0]: (float(x[1]) if len(x) == 2 else w) for x in items}
assert all(a in months and 0 <= v <= 1 for a, v in aw.items()) and flag in ('cell', 'dark_a', 'dark_b'), (scope, w, flag)
cells = [{'month': m, 'airport': a, 'w': v} for a, v in aw.items() for m in months[a]]
json.dump({'scope': scope, 'w': w, 'flag': flag, 'cells': cells}, open(sys.argv[3], 'w'), indent=1)
" "$SCOPE" "$W" $WEIGHTS "$FLAG"
ROWS=()
[ "$FLAG" = cell ] || ROWS=(--dark-rows "${DARK_ROWS:?FLAG $FLAG needs DARK_ROWS}")
echo "run_part: start $(date -u +%FT%TZ) fit=$FITROOT/dh-$FIT scope=$SCOPE w=$W flag=$FLAG"
[ -f $G/input_cov-$TAG.json ] || $PY -I $C/dh_input_cov.py --features $FEAT --features-sha $FEAT_SHA \
  --head $HEAD --head-sha $HEAD_SHA --raw $RAW --out $G/input_cov-$TAG.json
$PY -I $C/dh_apply.py --fit-dir $FITROOT/dh-$FIT --stage DHo --features $FEAT --features-sha $FEAT_SHA \
  --raw $RAW --head $HEAD --head-sha $HEAD_SHA --opdi $OPDI --opdi-sha $OPDI_SHA --weights "$WEIGHTS" \
  --threads 2 "${ROWS[@]}" --out $G/parts/dh-$TAG.parquet
sha256sum $G/parts/dh-$TAG.parquet
echo "run_part: done $(date -u +%FT%TZ)"
# the part applied to V minus V, per airport and month (desk 12:50 UTC 10-09)
$PY -I $C/part_diff.py --part $G/parts/dh-$TAG.parquet --v "$V_PART" --v-sha "$V_SHA" --raw $RAW \
  --out $G/parts/dh-$TAG-diff-v.json > /dev/null
echo "run_part: diff against V written $G/parts/dh-$TAG-diff-v.json $(date -u +%FT%TZ)"
# January/July parity read of pred_DH against the released head (finalcore parity.py, kind refit), when a repo
# snapshot (prc/ and experiments/exp-186-final-core/code/ of git archive) is given as the 5th argument
SNAP=${5:-}
if [ -n "$SNAP" ]; then
  cd "$SNAP"
  PRC_DATA_DIR=$ROOT/finalcore/raw PRC_RANKING_MODE=final $PY -I \
    experiments/exp-186-final-core/code/parity.py --refs $ROOT/finalcore/refs/v73-parts.parquet \
    --new $G/parts/dh-$TAG.parquet --pair pred_DH=vector/head:pred_HB3u --kind refit \
    --out $G/parts/dh-$TAG-parity.json > /dev/null
  echo "run_part: parity written $G/parts/dh-$TAG-parity.json $(date -u +%FT%TZ)"
fi
