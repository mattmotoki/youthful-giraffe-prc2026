#!/usr/bin/env bash
# §7bk hedge 2: C2-cat on november_loo, LOCAL CPU, 4 threads, with real iteration
# marks and a PROJECTION at 5 minutes.
#
# The 8.9 h figure in §7bj was a BUFFERED-OUTPUT BOUND, not a rate: CatBoost's
# verbose=200 goes from C++ to stdout, which is block buffered into a redirected
# file, so twenty minutes of silence proved nothing.  L2-hub's real rate then came
# in just under the cap, so the CatBoost bound may be as loose.
#
# This measures instead of bounding: `python -u`, and CatBoostRegressor.fit is
# wrapped to inject an after_iteration callback that prints to stdout every 100
# rounds.  At 5 minutes the wrapper reads the measured rate and DECIDES: keep if
# the ETA is 3 h or under, kill and say so if over.
set -u
FOLD="${PRC_LANE_FOLD:-november_loo}"
ROOT=/path/to/competitions/prc-data-challenge-2026
EXP="$ROOT/experiments/exp-047-queue"
OUT="$EXP/work/hedge"
LOG="$EXP/logs/hedge-C2-cat-$FOLD.log"
DECIDE_AFTER=300                 # 5 minutes
ETA_LIMIT_MIN=180                # 3 hours
mkdir -p "$OUT"

OMP_NUM_THREADS=4 PRC_THREADS=4 PYTHONPATH="$ROOT" PRC_GPU=0 \
"$ROOT/.venv/bin/python" -u -c "
import sys, runpy, time
sys.path.insert(0, 'experiments/exp-008-diversity')
sys.path.insert(0, '$ROOT/experiments/exp-047-queue/work')
import frames
import lane; lane.install(); lane.family()
import catboost
_t0 = time.time()
class Mark:
    def __init__(self, total): self.total = total
    def after_iteration(self, info):
        n = info.iteration + 1
        if n % 100 == 0 or n == self.total:
            el = time.time() - _t0
            rate = n / el if el > 0 else 0.0
            left = (self.total - n) / rate if rate > 0 else float('nan')
            print(f'[iter] {n}/{self.total} {el:.1f}s {rate:.2f} it/s '
                  f'eta {left/60:.1f} min', flush=True)
        return True
# CatBoost writes catboost_training.json, learn_error.tsv and tfevents into
# ./catboost_info -- the TRACKED repo root when cwd is the root (desk, 2026-09-10).
# Pin train_dir into this lane at construction, for every fit.
_cinit = catboost.CatBoostRegressor.__init__
def _init(self, *a, **k):
    k.setdefault('train_dir', '/path/to/competitions/prc-data-challenge-2026/experiments/exp-047-queue/work/catboost_info')
    return _cinit(self, *a, **k)
catboost.CatBoostRegressor.__init__ = _init
_orig = catboost.CatBoostRegressor.fit
def _fit(self, *a, **k):
    total = int(self.get_params().get('iterations') or 0)
    cbs = list(k.get('callbacks') or [])
    cbs.append(Mark(total))
    k['callbacks'] = cbs
    print(f'[fit] catboost iterations={total} thread_count='
          f'{self.get_params().get(\"thread_count\")}', flush=True)
    return _orig(self, *a, **k)
catboost.CatBoostRegressor.fit = _fit
sys.argv = ['experiments/exp-008-diversity/run.py',
            '--kind', 'cat', '--name', 'C2-d8p', '--depth', '8',
            '--folds', '$FOLD',
            '--rounds-m6', '3625', '--rounds-m11', '5325',
            '--cpu', '--outdir', '$OUT']
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('e8run', 'experiments/exp-008-diversity/run.py')
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
_mod.HERE = __import__('pathlib').Path('$ROOT/experiments/exp-047-queue/work/e8-results')
_mod.HERE.mkdir(parents=True, exist_ok=True)
print('[lane] exp-008 HERE repointed to', _mod.HERE, flush=True)
_mod.main()
" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$EXP/logs/hedge-C2-cat-$FOLD.pid"

sleep "$DECIDE_AFTER"
if ! ps -p "$PID" > /dev/null 2>&1; then
  echo "[decide] process already exited before the 5 min mark" >> "$LOG"
else
  eta=$(grep -o 'eta [0-9.]* min' "$LOG" | tail -1 | awk '{print $2}')
  if [ -z "${eta:-}" ]; then
    echo "[decide] NO iteration mark after ${DECIDE_AFTER}s -- still no measured rate; killing" >> "$LOG"
    kill "$PID" 2>/dev/null
  elif awk "BEGIN{exit !($eta > $ETA_LIMIT_MIN)}"; then
    echo "[decide] measured ETA ${eta} min is OVER the ${ETA_LIMIT_MIN} min limit; killing" >> "$LOG"
    kill "$PID" 2>/dev/null
  else
    echo "[decide] measured ETA ${eta} min is within the ${ETA_LIMIT_MIN} min limit; keeping" >> "$LOG"
  fi
fi
wait "$PID" 2>/dev/null
echo "[hedge] rc=$? " >> "$LOG"
cd "$ROOT" && git checkout -- experiments/exp-008-diversity/guard_bounds.json \
                             experiments/exp-008-diversity/climb.log 2>/dev/null
