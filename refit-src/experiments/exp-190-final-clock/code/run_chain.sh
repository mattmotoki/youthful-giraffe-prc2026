#!/usr/bin/env bash
# exp-190: the HB3u chain of exp-159 in one ranking mode, run from the repo root of a checkout or a git snapshot.
#   bash experiments/exp-190-final-clock/code/run_chain.sh <ranking|final> [step ...]
# Work folder: $EXP190_WORK, default $PRC_FINAL_ROOT/finalclock/work-<mode>. Parts: the final mode writes
# $PRC_FINAL_ROOT/finalclock/parts/ (the part contract); the ranking mode writes <work>/parts/ (a check, no part).
# Steps, in this order. Default, ranking mode: source frame frame12 track trackr5 track12 hu18 hu30 merge predict
# pmerge head pack. Default, final mode: source frame frame12 track trackr5 track12 predict pmerge head (the saved
# boosters of the ranking-mode refit predict the final frame; predict_head.py stops unless the final frame's training
# matrix equals the ranking-mode one, and then the refit steps hu18 hu30 merge are needed; SPEC design row 8).
# `head` takes HB3u-jjrank from out/fits (ranking) or out/pred (final); EXP190_HEAD_FROM=fits|pred overrides.
# Step `pack` (not in the default list): exp-159 pack.py of HB3u on jjrank and janjul; ranking mode with the
# released L references (the check that the head part equals the pack's values), final mode with $EXP190_LREF.
# A step whose output exists is skipped, so a stopped chain resumes; the scripts themselves are write-once.
# The final mode needs EXP159_FEATURES, EXP159_PROXY, EXP159_OPDI and EXP159_HARVEST_DIRS (finalroute and
# finaldata outputs; SPEC.md) in the environment. Remote only (the released source tree and inputs are there).
# Launch under the grant: setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> bash <this> ... &
set -euo pipefail
MODE=$1
shift
if [ "$MODE" = final ]; then
  DEFAULT="source frame frame12 track trackr5 track12 predict pmerge head"
else
  DEFAULT="source frame frame12 track trackr5 track12 hu18 hu30 merge predict pmerge head pack"
fi
STEPS="${*:-$DEFAULT}"
ROOT=$(pwd)
CODE=$ROOT/experiments/exp-190-final-clock/code
[ -f "$CODE/common.py" ] || { echo "run from the repo root" >&2; exit 2; }
FINAL_ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
RAW=${EXP159_RAW:-$FINAL_ROOT/finalcore/raw}
WORK=${EXP190_WORK:-$FINAL_ROOT/finalclock/work-$MODE}
OLD=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp159-work           # the released exp-159 work folder (read only)
PY=${PY:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src/.venv/bin/python}
THREADS=${THREADS:-4}
case "$MODE" in
  ranking) PARTS=$WORK/parts; TRAINREF=""; HEAD_FROM=${EXP190_HEAD_FROM:-fits} ;;
  final) : "${EXP159_FEATURES:?}" "${EXP159_PROXY:?}" "${EXP159_OPDI:?}" "${EXP159_HARVEST_DIRS:?}"
         PARTS=$FINAL_ROOT/finalclock/parts; HEAD_FROM=${EXP190_HEAD_FROM:-pred}
         TRAINREF="--train-ref ${EXP190_TRAIN_REF:-$FINAL_ROOT/finalclock/work-ranking/out/pred/HU18-jjrank.json}" ;;
  *) echo "mode must be ranking or final" >&2; exit 2 ;;
esac
export EXP159_WORK=$WORK EXP159_RAW=$RAW PRC_DATA_DIR=$RAW PRC_RANKING_MODE=$MODE
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
FITS=$WORK/out/fits
PRED=$WORK/out/pred
MODELS=${EXP190_MODELS:-$FINAL_ROOT/finalclock/work-ranking/out/fits}
FIT="$PY $CODE/fit.py --arm A4u --split jjrank --also-janjul --rounds 1800 --threads $THREADS --save-models --out $FITS"
stamp() { date -u +%FT%TZ; }
echo "$(stamp) run_chain $MODE work $WORK parts $PARTS [$STEPS] root $ROOT commit $(cat "$ROOT/SNAPSHOT_COMMIT" 2>/dev/null || git -C "$ROOT" rev-parse HEAD) threads $THREADS"
env | grep -E '^(EXP159|PRC_)' | sort

for step in $STEPS; do
  echo "$(stamp) step $step"
  case $step in
    source)
      # the released SOURCE tree (prc, pilot, the frozen rows of ec40e740) with prc/data.py of this commit, which has
      # the ranking modes; nothing on the ranking-mode path imports prc.data (SPEC.md)
      if [ ! -e "$WORK/source/SOURCE_COMMIT" ]; then
        mkdir -p "$WORK/source"
        (cd "$OLD/source" && tar --exclude=__pycache__ -cf - .) | (cd "$WORK/source" && tar -xf -)
        cp "$ROOT/prc/data.py" "$WORK/source/prc/data.py"
        echo "$(cat "$OLD/source/SOURCE_COMMIT")+prc/data.py:$(sha256sum "$ROOT/prc/data.py" | cut -c1-64)" > "$WORK/source/SOURCE_COMMIT"
        (cd "$WORK/source" && find . -type f | sort | xargs sha256sum) > "$WORK/source.sha256"
      fi
      mkdir -p "$WORK/inputs"
      [ -e "$WORK/inputs/features-r5.parquet" ] || ln -s "$OLD/inputs/features-r5.parquet" "$WORK/inputs/features-r5.parquet"
      [ "$(sha256sum < "$WORK/inputs/features-r5.parquet" | cut -c1-64)" = 78d22fe292e1896f0889a09fc1445f1b754a7cb28d99c88b0004ab8999e577ef ] \
        || { echo "features-r5 is not the released table" >&2; exit 1; }
      ;;
    frame)   [ -e "$WORK/inputs/frame.parquet" ] || $PY "$CODE/frame.py" ;;
    frame12) [ -e "$WORK/inputs/frame12.parquet" ] || $PY "$CODE/frame12.py" frame ;;
    track)   [ -e "$WORK/inputs/track.parquet" ] || EXP159_SET=9 $PY "$CODE/track.py" --procs "$THREADS" ;;
    trackr5) [ -e "$WORK/inputs/track-r5.parquet" ] || EXP159_SET=12 $PY "$CODE/track.py" --months 2025-04 2025-05 2025-09 \
               --out "$WORK/inputs/track-r5.parquet" --procs 3 ;;
    track12) [ -e "$WORK/inputs/track12.parquet" ] || $PY "$CODE/frame12.py" track ;;
    hu18)    [ -e "$FITS/HU18-jjrank.json" ] || EXP159_SET=12 $FIT --seeds 601-605 --stage HU18 ;;
    hu30)    [ -e "$FITS/HU30s2-jjrank.json" ] || EXP159_SET=12 $FIT --seeds 606-610 --param min_child_samples=30 --stage HU30s2 ;;
    merge)   [ -e "$FITS/HB3u-jjrank.json" ] || EXP159_SET=12 $PY "$CODE/merge.py" --dir "$FITS" --parts HU18 HU30s2 \
               --stage HB3u --splits jjrank jjrefit janjul --mixed-settings ;;
    pack)    if [ "$MODE" = final ]; then LREF="--lref ${EXP190_LREF:?}"; else LREF=""; fi
             [ -e "$WORK/pack/PACK.json" ] || EXP159_SET=12 $PY "$CODE/pack.py" --stage HB3u --src "$FITS" \
               --out "$WORK/pack" --splits jjrank janjul $LREF ;;
    predict) [ -e "$PRED/HU30s2-jjrank.json" ] || EXP159_SET=12 $PY "$CODE/predict_head.py" --models "$MODELS" \
               --out "$PRED" --threads "$THREADS" $TRAINREF ;;
    pmerge)  [ -e "$PRED/HB3u-jjrank.json" ] || EXP159_SET=12 $PY "$CODE/merge.py" --dir "$PRED" --parts HU18 HU30s2 \
               --stage HB3u --splits jjrank --mixed-settings ;;
    head)    [ -e "$PARTS/head.parquet" ] || EXP159_SET=12 $PY "$CODE/head_part.py" --fits "$WORK/out/$HEAD_FROM" \
               --out "$PARTS/head.parquet" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
echo "$(stamp) run_chain done"
