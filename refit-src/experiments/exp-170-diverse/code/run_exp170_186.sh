#!/usr/bin/env bash
# exp-186: the six exp-170 members (sequence021, sequence039, dark18, row6, sequence011, row011) on remote, CPU,
# from a git-archive snapshot. Run it through the ARC guard:
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#       bash <snapshot>/experiments/exp-170-diverse/code/run_exp170_186.sh <mode> <out> <threads> [member...] \
#       >/dev/null 2>&1 < /dev/null &
# Each view's 2026 rows come from rank_inputs_186.py with the clock and track tables E170_CLOCK/E170_TRACK.
# Ranking mode: frame12/track12 of exp-159, the tables the views record (default). Final mode: the four-month
# frame12/track12 of finalclock, named by path and SHA-256 from its MANIFEST (E170_CLOCK, E170_CLOCK_SHA256,
# E170_TRACK, E170_TRACK_SHA256; no default); the run stops when a hash differs.
# predict_rank_186.py runs each member's saved twelve-month fits (the parents of its saved mean) and the equal
# mean. Ranking mode: each fit against its ranking2026.parquet and the mean against the saved member vector
# (--tol), then parity.py (refit kind) against v73-parts. Final mode: the mean is part <member> in template
# order ($PARTS/<member>.parquet) with its parity read.
set -euo pipefail
MODE=$1; OUT=$2; T=$3; shift 3
MEMBERS=("$@")
[ ${#MEMBERS[@]} -eq 0 ] && MEMBERS=(row6 row011 sequence011 sequence021 dark18 sequence039)
SNAP=$(cd "$(dirname "$0")/../../.." && pwd)
C=$SNAP/experiments/exp-170-diverse/code
PARITY=$SNAP/experiments/exp-186-final-core/code/parity.py
REPO=${E170_REPO:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/experiments/exp-170-diverse}
VIEWS=${E170_VIEWS:-$REPO/inputs}
FITS=${E170_FITS:-${PRC_MODEL_ROOT:-/path/to/models}/prc-data-challenge-2026/experiments/exp-170-diverse/out}
if [ "$MODE" = ranking ]; then
  CLOCK=${E170_CLOCK:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work/inputs/frame12.parquet}
  TRACK=${E170_TRACK:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work/inputs/track12.parquet}
else
  CLOCK=${E170_CLOCK:?final mode: E170_CLOCK = finalclock frame12 (its MANIFEST path)}
  TRACK=${E170_TRACK:?final mode: E170_TRACK = finalclock track12 (its MANIFEST path)}
  echo "${E170_CLOCK_SHA256:?final mode: E170_CLOCK_SHA256}  $CLOCK" | sha256sum -c -
  echo "${E170_TRACK_SHA256:?final mode: E170_TRACK_SHA256}  $TRACK" | sha256sum -c -
fi
REFS=${E170_REFS:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet}
PARTS=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}/finalcore/parts  # COMMON part contract
# Replay stop of the ranking mode: the saved fits are float32 GPU nets replayed on CPU; row6 fits 17001/17002 differ
# from their saved vectors by up to 32 float32 spacings of the value (largest 0.0078 s at 28,700-40,600 s values,
# rms 6e-5 s, 76 % of rows bit-equal; exp-186 REPORT). A wrong model or input moves every row by seconds.
TOL=${E170_TOL:-0.05}
export PRC_RANKING_MODE=$MODE OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
export PRC_DATA_DIR=${PRC_DATA_DIR:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw}
PY=${PY:-python3}
stamp() { date -u +%FT%TZ; }
mkdir -p "$OUT/inputs" "$OUT/predict"
view_of() { case $1 in
  sequence021|sequence039) echo time-day-Z-052 ;; dark18) echo coherent-dark-Z-annual-064 ;;
  row6|row011) echo row-time-day-Z-054 ;; sequence011) echo runway-time-day-Z-059 ;; esac; }
mean_of() { case $1 in
  sequence021) echo fixed-sequence021-bag6-annual-060 ;; sequence039) echo fixed-close30-annual-061 ;;
  dark18) echo dark18-Z-annual-064 ;; row6) echo fixed-control6-bag6-annual-060 ;;
  sequence011) echo fixed-fallback_sequence011-17001-annual-060 ;; row011) echo fixed-control6-17001-annual-060 ;; esac; }
echo "$(stamp) run_exp170_186 $MODE threads $T members ${MEMBERS[*]} snapshot $SNAP"
for m in "${MEMBERS[@]}"; do
  v=$(view_of "$m"); ref=$(mean_of "$m")
  [ -n "$v" ] && [ -n "$ref" ] || { echo "unknown member $m"; exit 1; }
  if [ ! -e "$OUT/inputs/$v" ]; then
    echo "$(stamp) inputs $v"
    $PY "$C/rank_inputs_186.py" --view "$VIEWS/$v" --clock "$CLOCK" --track "$TRACK" --out "$OUT/inputs/$v"
  fi
  # the member's fits: the parents of its saved mean, or the one fit itself
  mapfile -t fits < <($PY -c "
import json, sys
from pathlib import Path
rec = json.loads(Path(sys.argv[1], 'FIT.json').read_text())
parents = rec.get('parents') or [sys.argv[1]]
# average_fits_026.py writes each parent as {'path', 'fit_sha256', 'model_sha256', 'weight'}; the mean is equal
weights = {p['weight'] for p in parents if isinstance(p, dict) and 'weight' in p}
assert len(weights) <= 1 and all(abs(w * len(parents) - 1) < 1e-9 for w in weights), weights
for p in parents:
    p = p['path'] if isinstance(p, dict) else p
    print(Path(sys.argv[2]) / Path(p).relative_to(Path(p).parents[1] if Path(p).name == 'last' else Path(p).parent))
" "$REPO/out/$ref" "$FITS")
  [ ${#fits[@]} -gt 0 ] || { echo "$m: no fits read from $REPO/out/$ref/FIT.json"; exit 1; }
  echo "$(stamp) predict $m: ${#fits[@]} fits, view $v"
  extra=()
  [ "$MODE" != ranking ] && { mkdir -p "$PARTS"; extra=(--part "$PARTS/$m.parquet"); }
  $PY "$C/predict_rank_186.py" --inputs "$OUT/inputs/$v" --fits "${fits[@]}" --out "$OUT/predict/$m" \
      --reference "$REPO/out/$ref/ranking2026.parquet" --tol "$TOL" --threads "$T" "${extra[@]}"
  if [ "$MODE" = ranking ]; then NEW=$OUT/predict/$m/ranking2026.parquet; READ=$OUT/$m-ranking-parity.json
  else NEW=$PARTS/$m.parquet; READ=$PARTS/$m-parity.json; fi
  $PY "$PARITY" --refs "$REFS" --new "$NEW" --part "$m" --col pred_seq170 --kind refit --out "$READ"
  sha256sum "$NEW" "$READ"
done
echo "$(stamp) done"
