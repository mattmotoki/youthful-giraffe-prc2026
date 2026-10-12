#!/usr/bin/env bash
# §7br: ONE month's common work, end to end, for ANY month.
#
# This is the third month (2025-11 §7bk, 2025-10 §7bo, 2025-03 §7br).  The first
# two ran from month-specific copies, and the copies are where the two defects
# came from: `oct_finish.sh` was written from emit.py's signature instead of the
# November command line and lost 122 columns, and the first `oct_chain.sh` looked
# for L2-hub under the wrong filename.  What the repeat says about the structure
# is that the month is a PARAMETER, not a variant -- so this script takes it from
# the environment and nothing else changes between months.
#
#   PRC_LANE_MONTH=2025-03 PRC_LANE_FOLD=march_loo bash work/month_chain.sh
#
# Thread budget 4 TOTAL (surfacefit's screen runs beside us), so every stage is
# strictly sequential: the cheap members take 2, the three big fits take 4.
# Order: T-tailv2+D1-winter -> N1-nmtail -> L2-hub -> C2-cat -> F1-fresh (gated)
#        -> base6 -> block re-emit -> verify -> shas.
set -u
# exp-064: this script is now part of the public release (it builds the
# extension panel of the shipped head), so the repository root is derived
# rather than hard-coded to one laptop.  PRC_ROOT overrides it.
ROOT="${PRC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP="$ROOT/experiments/exp-047-queue"
PY="$ROOT/.venv/bin/python"
MONTH="${PRC_LANE_MONTH:?set PRC_LANE_MONTH}"
FOLD="${PRC_LANE_FOLD:?set PRC_LANE_FOLD}"
export PRC_LANE_MONTH PRC_LANE_FOLD
# the summaries rule §7bk emitted with; --sources must be named too or emit.py
# falls back to "F" and writes 63 columns instead of 186 (the §7bo defect).
RULE=d4c59868864523ce6513151c1627ec2ba997fd25d6394cfe2ff03a14d89f01a0
LOG="$EXP/logs/chain-$MONTH.log"
cd "$ROOT"

say() { echo "[chain $MONTH $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
have() { [ -f "$EXP/oof/$1--$FOLD.parquet" ]; }

say "start: month=$MONTH fold=$FOLD"

if ! have T-tailv2 || ! have D1-winter; then
  say "cheap members (T-tailv2, D1-winter), 2 threads"
  $PY -u "$EXP/work/member_folds.py" T-tailv2 D1-winter >> "$LOG" 2>&1 \
    || say "member_folds rc=$?"
fi

if ! have N1-nmtail; then
  say "N1-nmtail, 2 threads"
  OMP_NUM_THREADS=2 PRC_THREADS=2 PYTHONPATH="$ROOT" \
    $PY -u "$EXP/work/nmtail_fold.py" >> "$LOG" 2>&1 || say "N1-nmtail rc=$?"
fi

if ! have L2-hub; then
  say "L2-hub (hedge, 4 threads, 3 h cap)"
  bash "$EXP/work/l2hub_hedge.sh" >> "$LOG" 2>&1 || say "L2-hub rc=$?"
  # the producer's --name is L2-hub8725, NOT L2-hub: looking for the member name
  # here is what made the first October chain think a finished fit was lost.
  if [ -f "$EXP/work/hedge/L2-hub8725-$FOLD.parquet" ]; then
    $PY "$EXP/work/promote_hedge.py" L2-hub \
        "$EXP/work/hedge/L2-hub8725-$FOLD.parquet" >> "$LOG" 2>&1 || say "promote L2-hub rc=$?"
  fi
fi

if ! have C2-cat; then
  say "C2-cat (hedge, 4 threads, 5-min ETA gate)"
  bash "$EXP/work/c2cat_hedge.sh" >> "$LOG" 2>&1 || say "C2-cat rc=$?"
  if [ -f "$EXP/work/hedge/C2-d8p-$FOLD.parquet" ]; then
    $PY "$EXP/work/promote_hedge.py" C2-cat \
        "$EXP/work/hedge/C2-d8p-$FOLD.parquet" >> "$LOG" 2>&1 || say "promote C2-cat rc=$?"
  fi
fi

if ! have F1-fresh; then
  say "F1-fresh (local CPU, 4 fits assumed, 180 min budget)"
  bash "$EXP/work/f1_local.sh" >> "$LOG" 2>&1 || say "F1-fresh rc=$?"
  if [ -f "$EXP/oof/$FOLD.parquet" ]; then
    $PY "$EXP/work/promote_hedge.py" F1-fresh "$EXP/oof/$FOLD.parquet" >> "$LOG" 2>&1 \
      && rm -f "$EXP/oof/$FOLD.parquet"
  fi
fi

# Other lanes are restored the moment the producers that write to them are done.
git checkout -- experiments/exp-008-diversity/guard_bounds.json \
                experiments/exp-008-diversity/climb.log \
                experiments/exp-008-diversity/result-C2-d8p.json 2>/dev/null || true
git checkout -- catboost_info 2>/dev/null || true
# an UNTRACKED result left in another lane damages no record, but it is still our
# litter in someone else's directory; a TRACKED one was restored just above.
mkdir -p "$EXP/work/e8-results" "$EXP/work/d1-results"
for p in experiments/exp-008-diversity/result-*.json \
         experiments/exp-012-blend2/result-_*.json; do
  [ -f "$p" ] || continue
  case "$(basename "$p")" in
    result-_q-*|result-_smoke-*|result-L2-hub8725.json) ;;
    *) continue ;;
  esac
  git ls-files --error-unmatch "$p" >/dev/null 2>&1 && continue
  case "$p" in
    *exp-012*) mv "$p" "$EXP/work/d1-results/" ;;
    *)         mv "$p" "$EXP/work/e8-results/" ;;
  esac
  say "moved out of another lane: $p"
done
say "other lanes restored: $(git status --porcelain experiments/exp-008-diversity catboost_info experiments/exp-012-blend2 | wc -l | tr -d ' ') dirty paths"

miss=""
for m in T-tailv2 D1-winter N1-nmtail L2-hub C2-cat F1-fresh; do
  have "$m" || miss="$miss $m"
done
if [ -n "$miss" ]; then
  say "HOLD: missing members:$miss -- base6 and the block are NOT composed"
  exit 3
fi

say "base6 on $MONTH"
OMP_NUM_THREADS=2 PRC_THREADS=2 PYTHONPATH="$ROOT" \
  $PY -u "$EXP/work/base6.py" >> "$LOG" 2>&1 || { say "base6 rc=$?"; exit 4; }

say "re-emitting the $MONTH block (November command line)"
mkdir -p "$ROOT/artifacts/exp-047/block-185"
cp -n "$ROOT/artifacts/exp-047/block/$MONTH.parquet" \
      "$ROOT/artifacts/exp-047/block-185/$MONTH.parquet" 2>/dev/null || true
$PY -u "$EXP/code/emit.py" "$MONTH" \
   --sources F,O,H,N,geo --rule "$RULE" \
   --pred "$EXP/oof/base6--$MONTH.parquet" >> "$LOG" 2>&1 \
  || { say "emit rc=$?"; exit 5; }
$PY -u "$EXP/work/block_verify.py" >> "$LOG" 2>&1 || say "VERIFY rc=$?"

say "DONE"
{
  echo "base6 $MONTH sha $(shasum -a 256 "$EXP/oof/base6--$MONTH.parquet" | cut -d' ' -f1)"
  echo "block $MONTH sha $(shasum -a 256 "$ROOT/artifacts/exp-047/block/$MONTH.parquet" | cut -d' ' -f1)"
} | tee -a "$LOG"
