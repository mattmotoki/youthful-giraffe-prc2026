#!/usr/bin/env bash
# exp-186: part mlp (MLP008 seeds 701/702 and mix12, the 14 saved exp-164 models) on remote, CPU, from a
# git-archive snapshot, in a work root of make_e164_work.sh (same snapshot). Run it through the ARC guard (the
# guard sets the CPU list, nice and ionice):
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#       bash <snapshot>/experiments/exp-186-final-core/code/run_mlp_186.sh <mode> <root> <threads> \
#       >/dev/null 2>&1 < /dev/null &
# Both modes first build the parity reference once (mlp_refs_186.py: the saved mlp008 and mix12 vectors in the
# v73-parts layout, checked bit for bit against the released primary_frame:mlp; reused when it exists).
# ranking: apply_final186.py and apply_mix12_186.py replay the 14 models on the board inputs (each seed, bag and
#   mean within 1e-3 s of the saved GPU vectors), part_mlp_186.py writes $ROOT/parts-ranking/mlp.parquet, and
#   parity.py reads it against the reference (refit kind: CPU replay of GPU fits).
# final: prepare.py --fit final builds the final inputs (needs EXP148_KINPUT, EXP159_INPUTS, EXP164_LP_PART and
#   EXP164_LP_PART_SHA; no default; the 2025 proxy months take the ranking stand table EXP164_STANDS_2025, default the
#   snapshot's exp-127 stand_geometry.csv 2ce7ad6b), the two apply steps predict them, part_mlp_186.py writes part mlp
#   ($PARTS/mlp.parquet) and parity.py reads it.
# Entry command of part mlp (argv from the repo root): bash experiments/exp-186-final-core/code/run_mlp_186.sh
#   final <root> <threads>; parts go to ${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}/finalcore/parts/.
set -euo pipefail
MODE=${1:?usage: run_mlp_186.sh <ranking|final> <root> <threads>}
ROOT=${2:?usage: run_mlp_186.sh <ranking|final> <root> <threads>}
T=${3:?usage: run_mlp_186.sh <ranking|final> <root> <threads>}
case "$MODE" in ranking|final) ;; *) echo "mode $MODE: ranking or final"; exit 1;; esac
case "$T" in 1|2|3|4) ;; *) echo "threads $T: 1 to 4"; exit 1;; esac
SNAP=$(cd "$(dirname "$0")/../../.." && pwd)
E164=$SNAP/experiments/exp-164-double-down/code/mlp008
C186=$SNAP/experiments/exp-186-final-core/code
COMMIT=${MLP_COMMIT:-$(basename "$SNAP")}
BOARD=${E164_BOARD_WORK:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp164-work}
V73=${MLP_V73_REFS:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet}
REFS=${MLP_REFS_DIR:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/mlp}
PARTS=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}/finalcore/parts  # COMMON part contract
PY=${PY:-python3}
# Replay stop of the ranking mode: CPU replays of the saved float32 nets differ from the saved predictions by
# float32 rounding (MLP008 seed 701: 0.0062 s; exp-170: up to 32 float32 spacings, 0.0078 s; exp-186 REPORT).
# A wrong model or input moves every row by seconds. APPLY.json records the largest difference per seed.
TOL=${MLP_TOL:-0.05}
[ -f "$ROOT/ROOT.txt" ] || { echo "no work root $ROOT: run make_e164_work.sh $ROOT $SNAP first"; exit 1; }
[ "$(cd "$ROOT/code" && pwd -P)" = "$(cd "$SNAP/experiments/exp-164-double-down/code" && pwd -P)" ] \
  || { echo "$ROOT/code is not the exp-164 code of $SNAP"; exit 1; }
export PRC_RANKING_MODE=$MODE OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
export PRC_DATA_DIR=${PRC_DATA_DIR:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw}
export PYTHONDONTWRITEBYTECODE=1 EXP164_WORK=$ROOT
if [ "$MODE" = final ]; then
  export EXP148_KINPUT=${EXP148_KINPUT:?final mode: EXP148_KINPUT = finaldata final kinput root (its MANIFEST path)}
  export EXP159_INPUTS=${EXP159_INPUTS:?final mode: EXP159_INPUTS = finalclock folder with frame12/track12 .parquet and .json}
  export EXP164_LP_PART=${EXP164_LP_PART:?final mode: EXP164_LP_PART = finalroute part lp (its MANIFEST path)}
  export EXP164_LP_PART_SHA=${EXP164_LP_PART_SHA:?final mode: EXP164_LP_PART_SHA (finalroute MANIFEST)}
  export EXP164_HARNESS=${EXP164_HARNESS:-$ROOT/results/HARNESS-final.json}
  # 2025 proxy months with the ranking stand table of the saved fit, 2026 months with the final one (prepare.py)
  export EXP164_STANDS_2025=${EXP164_STANDS_2025:-$SNAP/experiments/exp-127-clock-precision/stand_geometry.csv}
  export EXP164_STANDS_2025_SHA=${EXP164_STANDS_2025_SHA:-2ce7ad6bf2b2c338272f1f04042027ca6d25d8c03d03708595c27f4882088008}
  [ -f "$EXP164_STANDS_2025" ] || { echo "no 2025 stand table $EXP164_STANDS_2025 (snapshot path experiments/exp-127-clock-precision/stand_geometry.csv)"; exit 1; }
  export EXP153_CODE=$SNAP/experiments/exp-153-grind/code EXP164_CPUS=${EXP164_CPUS:-8-11,72-75}
  [ -f "$EXP164_HARNESS" ] || { echo "no final harness record $EXP164_HARNESS (harness_final186.py --commit $COMMIT)"; exit 1; }
  [ -e "$PARTS/mlp.parquet" ] && { echo "part exists: $PARTS/mlp.parquet"; exit 1; }
fi
digest() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
stamp() { date -u +%FT%TZ; }
echo "$(stamp) run_mlp_186 $MODE threads $T snapshot $SNAP root $ROOT commit $COMMIT"
if [ -e "$REFS/mlp-refs.parquet" ]; then
  echo "$(stamp) refs exist: $REFS/mlp-refs.parquet"
else
  echo "$(stamp) refs"
  $PY "$C186/mlp_refs_186.py" --v73-parts "$V73" --work "$BOARD" --out "$REFS"
fi
if [ "$MODE" = final ]; then
  echo "$(stamp) prepare"
  $PY "$E164/prepare.py" --fit final
  OUT=$PARTS
else
  OUT=$ROOT/parts-ranking
fi
echo "$(stamp) apply mlp008"
$PY "$E164/apply_final186.py" --threads "$T" --tol "$TOL"
echo "$(stamp) apply mix12"
$PY "$E164/apply_mix12_186.py" --threads "$T" --tol "$TOL"
echo "$(stamp) part"
$PY "$C186/part_mlp_186.py" --out "$OUT" --commit "$COMMIT" \
    --rebuild-tolerance "${MLP_REBUILD_TOLERANCE:-not measured (one run)}"
echo "$(stamp) parity"
$PY "$C186/parity.py" --refs "$REFS/mlp-refs.parquet" --new "$OUT/mlp.parquet" \
    --pair mlp008=saved:mlp008 --pair mix12=saved:mix12 --kind refit --out "$OUT/mlp-parity.json"
digest "$OUT/mlp.parquet" "$OUT/mlp.json" "$OUT/mlp-parity.json"
echo "$(stamp) done"
