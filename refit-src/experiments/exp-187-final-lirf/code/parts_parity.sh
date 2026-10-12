#!/usr/bin/env bash
# January/July parity read of the three frame parts against v73-parts.parquet (finalcore parity.py, kind refit),
# written next to each part as <part>-parity.json. Run from the repo root (or a snapshot of GITPATHS.txt).
#   bash experiments/exp-187-final-lirf/code/parts_parity.sh <dir with frames_<frame>[-tag].parquet> [tag] [refs]
set -euo pipefail
DIR=$1; TAG=${2:-}; REFS=${3:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet}
SFX=${TAG:+-$TAG}
PY=${PYTHON:-python3}
run() {  # frame, then NEWCOL=REFCOL pairs
  local f=$1; shift
  local args=()
  for p in "$@"; do args+=(--pair "$p"); done
  "$PY" experiments/exp-186-final-core/code/parity.py --refs "$REFS" --new "$DIR/frames_${f}${SFX}.parquet" \
    "${args[@]}" --kind refit --out "$DIR/frames_${f}${SFX}-parity.json" > /dev/null
  echo "parity written: $DIR/frames_${f}${SFX}-parity.json"
}
run bundle lo=bundle:lo hi=bundle:hi rule=bundle:rule extension=bundle:extension noplan=bundle:noplan \
  np_v3cal=bundle:np_v3cal practice=bundle:practice practice_weight=bundle:practice_weight \
  practice_point=bundle:practice_point stand_floor=bundle:stand_floor cls=bundle:cls c73=bundle:c73 \
  c73_weight=bundle:c73_weight c73_point=bundle:c73_point
run rule_frame cls=rule_frame:cls t=rule_frame:t scope=rule_frame:scope g=rule_frame:g g71=rule_frame:g71 c71=rule_frame:c71
run exploit_frame C40=exploit_frame:C40 C58=exploit_frame:C58 C63=exploit_frame:C63 C64=exploit_frame:C64 \
  C66=exploit_frame:C66 C68=exploit_frame:C68 C71=exploit_frame:C71 order_delta=exploit_frame:order_delta \
  point58=exploit_frame:point58 point66=exploit_frame:point66 lower=exploit_frame:lower upper=exploit_frame:upper
