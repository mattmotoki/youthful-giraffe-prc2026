#!/usr/bin/env bash
# exp-185: final-mode builds of one month in a snapshot (all four months from final_ranking.parquet), each
# output copied to remote with tools/gput.sh (writer under arc_holdguard.sh) and checked by SHA-256 (tools/gread.sh):
#   class  exp-046 class (code/class_final.py)          -> F/class-final/<m>-v25.parquet (Mac copy kept, 1 MB)
#   block  exp-047 block (emit.py, pinned final mode)   -> F/exp-047/block-final/<m>.parquet (Mac copy deleted)
#   e130   exp-130 month files (code/exp130_months.py)  -> F/exp-130/months/{month,rank_meta}_<m>.parquet,
#                                                          exp130-months-<m>.json (Mac copies deleted)
#   SN=<snapshot> W=<Mac work dir> STEPS="class block e130" [NOPUT=1] bash experiments/exp-185-final-data/code/run_month.sh <month>
# NOPUT=1 builds on the Mac only (an ARC hold); code/push_month.sh copies, checks and frees the Mac later.
# W holds e018-final/ (exp-018 cache of the final mode, all four months, one vocab.json).
set -euo pipefail
M=$1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-185-final-data"
PY="$ROOT/.venv/bin/python"
F=${PRC_FINAL_ROOT:-/path/to/final}/finaldata
RULE=d4c59868864523ce6513151c1627ec2ba997fd25d6394cfe2ff03a14d89f01a0
STEPS=${STEPS:-class block e130}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 PRC_RANKING_MODE=final PRC_DATA_DIR="$ROOT/data/raw"
SNC=$(cat "$SN/COMMIT")
. "$EXP/code/remote_lib.sh"
put() {  # $1 Mac dir (all files), $2 remote dir
  [ -n "${NOPUT:-}" ] && return 0
  hold_wait
  (cd "$1" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf - .) | ssh "${PRC_HOST:-remote}" "bash $F/tools/gput.sh '$2'" >/dev/null
}
check() {  # $1 Mac dir, $2 remote dir: SHA-256 of every file equal, else stop
  local mac remote
  [ -n "${NOPUT:-}" ] && return 0
  hold_wait
  mac=$(cd "$1" && shasum -a 256 * | awk '{print $1"  "$2}' | sort -k2)
  remote=$(ssh -n "${PRC_HOST:-remote}" "bash $F/tools/gread.sh bash -c 'cd $2 && sha256sum $(cd "$1" && ls | tr '\n' ' ')'; true" | sort -k2)
  [ "$mac" = "$remote" ] || { echo "$M: remote copy of $1 differs"; echo "$mac"; echo "$remote"; exit 1; }
}
for step in $STEPS; do
  case $step in
    class)
      d="$W/class/$M"; mkdir -p "$d"
      "$PY" "$EXP/code/class_final.py" "$SN" "$d" "$M" > "$EXP/logs/class-final-$M.log" 2>&1
      cp "$d/class-final.json" "$EXP/checks/class-final-$M-$SNC.json"
      put "$d" "$F/class-final/$M"; check "$d" "$F/class-final/$M"
      echo "$M class: $(shasum -a 256 "$d/$M-v25.parquet" | cut -c1-16)";;
    block)
      (cd "$SN" && PRC_ROOT="$SN" "$PY" experiments/exp-047-queue/code/emit.py "$M" --sources F,O,H,N,geo --rule "$RULE") \
        > "$EXP/logs/exp047-emit-final-$M.log" 2>&1
      d="$W/block/$M"; mkdir -p "$d"; mv "$SN/artifacts/exp-047/block-final/$M.parquet" "$d/$M.parquet"
      cp "$SN/experiments/exp-047-queue/analysis/interface-final.json" "$EXP/checks/exp047-interface-final-$SNC.json"
      put "$d" "$F/exp-047/block-final"; check "$d" "$F/exp-047/block-final"
      echo "$M block: $(shasum -a 256 "$d/$M.parquet" | cut -c1-16)"
      [ -n "${NOPUT:-}" ] || bash "$EXP/code/mac_delete_verified.sh" --dir "$d" "$F/exp-047/block-final" >/dev/null;;
    e130)
      d="$W/e130/$M"; mkdir -p "$d"
      "$PY" "$EXP/code/exp130_months.py" "$SN" "$d" "$W/e018-final" "$W/class/$M" "$M" > "$EXP/logs/exp130-final-$M.log" 2>&1
      mv "$d/exp130-months.json" "$d/exp130-months-$M.json"; cp "$d/exp130-months-$M.json" "$EXP/checks/"
      put "$d" "$F/exp-130/months"; check "$d" "$F/exp-130/months"
      echo "$M e130: month $(shasum -a 256 "$d/month_$M.parquet" | cut -c1-16) rank_meta $(shasum -a 256 "$d/rank_meta_$M.parquet" | cut -c1-16)"
      [ -n "${NOPUT:-}" ] || bash "$EXP/code/mac_delete_verified.sh" --dir "$d" "$F/exp-130/months" >/dev/null;;
    *) echo "unknown step $step"; exit 2;;
  esac
done
