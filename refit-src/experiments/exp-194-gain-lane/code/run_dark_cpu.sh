#!/usr/bin/env bash
# exp-194 PLAN-DARK CPU jobs on remote, one step after another, from a git-archive snapshot
# (prc, experiments/exp-047-queue/code, exp-147-foldserve/code, exp-148-short-route/code, exp-164-double-down/code,
# exp-170-diverse/{code,results}, exp-146-validation/frozen, exp-194-gain-lane/code):
#   setsid nohup env THREADS=<n> PROXY_FINAL=<p> L_CACHE=<d> bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <snap>/experiments/exp-194-gain-lane/code/run_dark_cpu.sh <tag> <step>... >/dev/null 2>&1 < /dev/null &
# THREADS (default 2) is the thread count of every step. Two chains may run side by side with THREADS=1 each (desk
# 09:02 UTC 10-09: 2 threads in all); they meet through markers $O/steps/<step>.done (or .failed).
# Steps (outputs under $G/dark-<tag>/, write-once):
#   values      dark_values.py: the value of a no-reception row per surface input (finalroute final tables)
#   flags       dh_dark_rows.py: label-free dark-row flags, 2025 and 2026
#   mlp         mlp_dark.py: mix12 lit and dark, January and July
#   x170cpu     x170_dark.py on CPU for sequence021 and row6
#   x170bags    x170_dark.py on CPU for dark18 and sequence039 (the January-mask job; GPU dropped, desk 09:02 UTC)
#   r4lit       r4mk_dark.py with L and Lp lit
#   learners    l_dark.py: L and Lp refit with the dark view (learnerL, learnerLp: one learner each)
#   r4dark      r4mk_dark.py with L and Lp dark (after the learners)
#   pack        pack_fetch.py: the EDDF/LIRF rows of every output present (the flag file: see pack_fetch.py), to
#               $O/fetch-<time>-<pid>/ for the Mac
#   packboth    pack, once, when x170bags and x170cpu are both done (the second chain to get there packs)
#   r4darkboth  r4dark and pack, once, when learnerL and learnerLp are both done
#   wait:<s>    wait (sleep) until step <s> of the other chain is done; stop if it failed
set -Eeuo pipefail
TAG=$1; shift
SNAP=$(cd "$(dirname "$0")/../../.." && pwd)
C=$SNAP/experiments/exp-194-gain-lane/code
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane
O=$G/dark-$TAG
T=${THREADS:-2}
PYF=${PRC_FINAL_ROOT:-/path/to/final}/finalv/venv312/bin/python          # pandas/pyarrow (my DH jobs)
PYR=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/.venv/bin/python   # lightgbm/catboost of exp-148/exp-147
PYT=python3                                                        # torch (exp-186 run_exp170_186.sh, run_mlp_186.sh)
W148=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work
W164=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp164-work
FEAT=${PRC_FINAL_ROOT:-/path/to/final}/finalroute/inputs/r4c-final/features-r4c.parquet
FEAT_SHA=39466f229682c802f6f39204341dae58fae26164d82e9cb8b58d4b8629054302
PROXY_FINAL=${PROXY_FINAL:?the final proxy table of finalroute (path)}
HEAD=${PRC_FINAL_ROOT:-/path/to/final}/finalclock/parts/head.parquet
HEAD_SHA=a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6
RAW=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw/final
[ -f $RAW/final_ranking.parquet ] || RAW=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw
L_CACHE=${L_CACHE:?the proxy cache folder for learner_wp (path)}
export THREADS=$T OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
export PYTHONDONTWRITEBYTECODE=1 PRC_SOURCE_ROOT=$SNAP PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw
unset PRC_RANKING_MODE
stamp() { date -u +%FT%TZ; }
mkdir -p "$O/steps"
cd "$O"
learner() { $PYR "$C/l_dark.py" --input-root $W148/kinput --cache "$L_CACHE" --saved-art $W148/out-jj \
              --saved-art $W148/kaggle/fj-L/out --dark-values "$O/dark_values.json" --out "$O/learners" \
              --learners "$1" --threads $T; }
r4dark() { $PYR "$C/r4mk_dark.py" --work $W148 --saved $W148/fold/jjrefit --dark-values "$O/dark_values.json" \
              --member-dark L="$O/learners/L-dark.parquet" --member-dark Lp="$O/learners/Lp-dark.parquet" \
              --out "$O/r4mk-darkL" --threads $T; }
pack() { local items=() f b
         for f in r4mk-litL r4mk-darkL; do if [ -f "$O/$f.parquet" ]; then items+=("$f=$O/$f.parquet"); fi; done
         if [ -f "$O/mlp/mix12-dark.parquet" ]; then items+=("mix12-dark=$O/mlp/mix12-dark.parquet"); fi
         if [ -f "$O/dark_rows.parquet" ]; then items+=("dark_rows=$O/dark_rows.parquet"); fi
         for b in sequence021 sequence039 dark18 row6; do
           if [ -f "$O/x170/$b/mean.parquet" ]; then items+=("x170-$b=$O/x170/$b/mean.parquet"); fi; done
         if [ ${#items[@]} -eq 0 ]; then echo "$(stamp) pack: no output yet"; return 0; fi
         $PYF -I "$C/pack_fetch.py" --out-dir "$O/fetch-$(date -u +%H%M)-$$" "${items[@]}"; }
done_() { [ -f "$O/steps/$1.done" ]; }
step=start
trap 'rc=$?; if [ $rc -ne 0 ]; then touch "$O/steps/$step.failed"; echo "$(stamp) step $step FAILED (exit $rc)"; fi' EXIT
echo "$(stamp) run_dark_cpu: tag $TAG threads $T steps $* snapshot $SNAP pid $$"
for step in "$@"; do
  echo "$(stamp) step $step start"
  case $step in
    values) $PYF -I "$C/dark_values.py" --features $FEAT --proxy "$PROXY_FINAL" --head $HEAD --raw $RAW \
              --out "$O/dark_values.json" ;;
    flags) $PYF -I "$C/dh_dark_rows.py" --features $FEAT --features-sha $FEAT_SHA --head $HEAD --head-sha $HEAD_SHA \
              --raw $RAW --out "$O/dark_rows" ;;
    mlp) EXP164_WORK=$W164 $PYT "$C/mlp_dark.py" --code164 "$SNAP/experiments/exp-164-double-down/code/mlp008" \
              --out "$O/mlp" --threads $T ;;
    x170cpu) bash "$C/run_x170_dark.sh" cpu "$O/x170" sequence021 row6 ;;
    x170bags) bash "$C/run_x170_dark.sh" cpu "$O/x170" dark18 sequence039 ;;
    r4lit) $PYR "$C/r4mk_dark.py" --work $W148 --saved $W148/fold/jjrefit --dark-values "$O/dark_values.json" \
              --out "$O/r4mk-litL" --threads $T ;;
    learners) learner L,Lp ;;
    learnerL) learner L ;;
    learnerLp) learner Lp ;;
    r4dark) r4dark ;;
    pack) pack ;;
    packboth) if done_ x170bags && done_ x170cpu && mkdir "$O/steps/packboth.lock" 2>/dev/null; then pack
              else echo "$(stamp) packboth: the other chain packs (or has packed)"; fi ;;
    r4darkboth) if done_ learnerL && done_ learnerLp && mkdir "$O/steps/r4darkboth.lock" 2>/dev/null; then
                  r4dark; pack
                else echo "$(stamp) r4darkboth: the other chain runs it (or has run it)"; fi ;;
    wait:*) w=${step#wait:}
            until done_ "$w"; do
              if [ -f "$O/steps/$w.failed" ]; then echo "$(stamp) step $w failed in the other chain"; false; fi
              sleep 60
            done ;;
    *) echo "unknown step $step"; false ;;
  esac
  touch "$O/steps/$step.done"
  echo "$(stamp) step $step done"
done
echo "$(stamp) run_dark_cpu: done"
