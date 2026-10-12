#!/usr/bin/env bash
# exp-186: part g (role g, the 12 saved AB serve fits) on remote, CPU, from a git-archive snapshot.
# Run it through the ARC guard (the guard sets the CPU list, nice and ionice):
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#       bash <snapshot>/experiments/exp-181-modela-bag-read/serve-packet/code/run_g_186.sh <mode> <out> <threads> \
#       >/dev/null 2>&1 < /dev/null &
# ranking: g inputs of the board plan through prepare_annual_186.py, compared file by file with the saved inputs
#   (MANIFEST dc598af3); check_annual_186.py on them; the 12 fits predict the saved inputs on CPU, each against
#   its own ranking.parquet (--tol), the mean against v73-parts g (parity.py, refit kind: CPU replay of GPU fits).
# final: final g inputs, checks and the final plan; the 12 fits predict the 670,790 final rows; the mean is
#   part g ($PARTS/g.parquet) with its parity read against v73-parts g.
# Entry command of part g (argv from the repo root): bash experiments/exp-181-modela-bag-read/serve-packet/code/
#   run_g_186.sh final <out> <threads>; parts go to ${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}/finalcore/parts/.
set -euo pipefail
MODE=$1; OUT=$2; T=$3
SNAP=$(cd "$(dirname "$0")/../../../.." && pwd)
G=$SNAP/experiments/exp-150-model-a/sequence-g
SP=$SNAP/experiments/exp-181-modela-bag-read/serve-packet
PARITY=$SNAP/experiments/exp-186-final-core/code/parity.py
SAVED=${G_SAVED_INPUTS:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp150-work/out/sequence-g-annual-inputs-001}
FITS=${G_FITS:-${PRC_MODEL_ROOT:-/path/to/models}/prc-exp181-work/out/gpu0}
REFS=${G_REFS:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet}
PARTS=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}/finalcore/parts  # COMMON part contract
export PRC_RANKING_MODE=$MODE OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
export PRC_DATA_DIR=${PRC_DATA_DIR:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw}
PY=${PY:-python3}
[ -e "$OUT" ] && { echo "output folder exists: $OUT"; exit 1; }
mkdir -p "$OUT"
FIT_DIRS=()
for r in mixed64 mixed128; do for s in 150081 150082 150083 150084 150085 150086; do
  FIT_DIRS+=("$FITS/serve-ab-$r-$s"); done; done
stamp() { date -u +%FT%TZ; }
echo "$(stamp) run_g_186 $MODE threads $T snapshot $SNAP"
cd "$G/code"
echo "$(stamp) prepare"
$PY prepare_annual_186.py --board-plan "$G/ANNUAL-PLAN.json" --plan-out "$OUT/ANNUAL-PLAN-$MODE.json" \
    --out "$OUT/inputs" --compare "$SAVED"
echo "$(stamp) check"
$PY check_annual_186.py --inputs "$OUT/inputs" --out "$OUT/ANNUAL-CHECKS-$MODE.json"
cd "$SP/code"
echo "$(stamp) predict"
if [ "$MODE" = ranking ]; then
  $PY predict_saved_186.py --inputs "$SAVED" --plan "$SP/SERVE-PLAN.json" --checks "$SP/ANNUAL-CHECKS.json" \
      --fits "${FIT_DIRS[@]}" --out "$OUT/predict" --threads "$T"
  NEW=$OUT/predict/mean.parquet; READ=$OUT/g-ranking-parity.json
else
  mkdir -p "$PARTS"
  $PY predict_saved_186.py --inputs "$OUT/inputs" --plan "$SP/SERVE-PLAN.json" --checks "$OUT/ANNUAL-CHECKS-$MODE.json" \
      --saved-inputs "$SAVED" --board-checks "$SP/ANNUAL-CHECKS.json" \
      --fits "${FIT_DIRS[@]}" --out "$OUT/predict" --part "$PARTS/g.parquet" --threads "$T"
  NEW=$PARTS/g.parquet; READ=$PARTS/g-parity.json
fi
echo "$(stamp) parity"
$PY "$PARITY" --refs "$REFS" --new "$NEW" --part g --col pred_Gmixed --kind refit --out "$READ"
sha256sum "$NEW" "$READ"
echo "$(stamp) done"
