#!/usr/bin/env bash
# §7bo: F1-fresh on october_loo, LOCAL CPU.  Kaggle is dead, so the member that
# cost 0.361 GPU-h in §7bi runs here with a measured gate instead of an estimate.
#
# WHY THE GATE IS NOT §7bk's.  §7bk's C2-cat gate projected 71.5 min and the run
# took 78, because CatBoost fits TWICE there and the 5-minute mark saw only the
# first.  F1-fresh is worse: `--main-model both` fits FOUR models --
# lgb(exact_rounds=300), lgb(rounds=1200), CatBoost, and a ghost lgb -- so a
# single-fit projection is a floor, not an estimate.  This gate therefore
#   * wraps BOTH lgb.train and CatBoostRegressor.fit and marks every one of them,
#   * keeps only if the CURRENT fit's ETA leaves room for FITS_ASSUMED fits,
#   * and arms a hard wall cap that is REAPED when the run ends (a cap timer that
#     outlives its target fires kill at a recycled pid -- see the c80b1e9 note).
#
# EXP is repointed into THIS lane so the producer's three writes (oof/, results/,
# climb.log) land here and not in exp-018-fresh.  EXP.parent is `experiments/`
# either way, so its one sys.path read is unchanged.
set -u
ROOT=/path/to/competitions/prc-data-challenge-2026
EXP="$ROOT/experiments/exp-047-queue"
FOLD="${PRC_LANE_FOLD:-october_loo}"
LOG="$EXP/logs/f1-$FOLD.log"
DECIDE_AFTER=300                 # 5 minutes
BUDGET_MIN=180                   # 3 hours, the desk's cap
FITS_ASSUMED=4                   # lgb(300), lgb(1200), catboost, ghost lgb
CAP=$((BUDGET_MIN * 60))

OMP_NUM_THREADS=4 PRC_THREADS=4 PYTHONPATH="$ROOT" PRC_LANE_MONTH="${PRC_LANE_MONTH:-2025-10}" \
"$ROOT/.venv/bin/python" -u -c "
import sys, time, json, importlib.util
from pathlib import Path
ROOT = Path('$ROOT')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/exp-047-queue/work'))
sys.path.insert(0, str(ROOT / 'experiments/exp-018-fresh/code'))
import lane; lane.install()

_t0 = time.time()
_fits = {'n': 0}
def _mark(tag, n, total):
    el = time.time() - _t0
    rate = n / el if el > 0 else 0.0
    print(f'[iter] {tag} fit{_fits[\"n\"]} {n}/{total} {el:.1f}s '
          f'{rate:.2f} it/s eta {(total - n) / rate / 60 if rate else float(\"nan\"):.1f} min',
          flush=True)

import lightgbm as lgb
_lgb_train = lgb.train
def _train(params, ds, num_boost_round=100, **k):
    _fits['n'] += 1
    total = int(num_boost_round)
    print(f'[fit] lgb fit{_fits[\"n\"]} rounds={total}', flush=True)
    t = time.time()
    cbs = list(k.get('callbacks') or [])
    def _cb(env):
        n = env.iteration + 1
        if n % 100 == 0 or n == total:
            el = time.time() - t
            r = n / el if el > 0 else 0.0
            print(f'[iter] lgb fit{_fits[\"n\"]} {n}/{total} {el:.1f}s {r:.2f} it/s '
                  f'eta {(total - n) / r / 60 if r else float(\"nan\"):.1f} min', flush=True)
    cbs.append(_cb)
    k['callbacks'] = cbs
    return _lgb_train(params, ds, num_boost_round=num_boost_round, **k)
lgb.train = _train

try:
    import catboost
    # exp-018-fresh/code/model.py pins task_type='GPU', devices='0'.  There is no
    # CUDA card here, and CatBoost REFUSES user callbacks on GPU -- that is what
    # killed the first attempt.  Force CPU at construction, before model.py builds
    # one.  Recorded: October's F1-fresh is a CPU CatBoost where November's ran on
    # a Kaggle P100, so the two months' F1-fresh are not bit-identical estimators.
    _cinit = catboost.CatBoostRegressor.__init__
    def _init(self, *a, **k):
        k.setdefault('train_dir', '/path/to/competitions/prc-data-challenge-2026/experiments/exp-047-queue/work/catboost_info')   # never ./catboost_info at the repo root
        if k.get('task_type') == 'GPU':
            k['task_type'] = 'CPU'
            k.pop('devices', None)
            k.setdefault('thread_count', 4)
            print('[lane] CatBoost forced GPU -> CPU (no card here)', flush=True)
        return _cinit(self, *a, **k)
    catboost.CatBoostRegressor.__init__ = _init
    _cfit = catboost.CatBoostRegressor.fit
    class _Mark:
        def __init__(self, total, t): self.total, self.t = total, t
        def after_iteration(self, info):
            n = info.iteration + 1
            if n % 100 == 0 or n == self.total:
                el = time.time() - self.t
                r = n / el if el > 0 else 0.0
                print(f'[iter] cat fit{_fits[\"n\"]} {n}/{self.total} {el:.1f}s {r:.2f} it/s '
                      f'eta {(self.total - n) / r / 60 if r else float(\"nan\"):.1f} min', flush=True)
            return True
    def _fit(self, *a, **k):
        _fits['n'] += 1
        total = int(self.get_params().get('iterations') or 0)
        print(f'[fit] cat fit{_fits[\"n\"]} iterations={total}', flush=True)
        cbs = list(k.get('callbacks') or []); cbs.append(_Mark(total, time.time()))
        k['callbacks'] = cbs
        return _cfit(self, *a, **k)
    catboost.CatBoostRegressor.fit = _fit
except ImportError:
    print('[fit] catboost not importable; only lgb is marked', flush=True)

# load run.py WITHOUT executing main(), repoint EXP into this lane, then call it
spec = importlib.util.spec_from_file_location(
    'f1run', str(ROOT / 'experiments/exp-018-fresh/code/run.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.EXP = ROOT / 'experiments/exp-047-queue'
print(f'[lane] EXP repointed to {mod.EXP}', flush=True)
sys.argv = ['run.py', '--tag', 'f1-$FOLD', '--folds', '$FOLD',
            '--rounds', '1200', '--clip-z', '6000', '--exact-rounds', '300',
            '--ghost-mode', 'mixbase', '--main-model', 'both', '--seeds', '1',
            '--params', json.dumps({'num_threads': 4}), '--save-oof']
mod.main()
" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$EXP/logs/f1-$FOLD.pid"

( sleep "$CAP"; kill "$PID" 2>/dev/null && echo "[cap] ${BUDGET_MIN} min wall cap reached, killed" >> "$LOG" ) &
CAPPID=$!

sleep "$DECIDE_AFTER"
if ! ps -p "$PID" > /dev/null 2>&1; then
  echo "[decide] finished or died before the 5 min mark" >> "$LOG"
else
  eta=$(grep -o 'eta [0-9.]* min' "$LOG" | tail -1 | awk '{print $2}')
  seen=$(grep -c '^\[fit\]' "$LOG")
  if [ -z "${eta:-}" ]; then
    echo "[decide] NO iteration mark after ${DECIDE_AFTER}s -- no measured rate; killing" >> "$LOG"
    kill "$PID" 2>/dev/null
  else
    # room for FITS_ASSUMED fits, not just the one being measured -- the §7bk repair
    per_fit=$(awk "BEGIN{print $BUDGET_MIN / $FITS_ASSUMED}")
    if awk "BEGIN{exit !($eta > $per_fit)}"; then
      echo "[decide] fit $seen ETA ${eta} min exceeds ${per_fit} min/fit "\
"(${BUDGET_MIN} min budget over ${FITS_ASSUMED} assumed fits); killing and HOLDING" >> "$LOG"
      kill "$PID" 2>/dev/null
    else
      echo "[decide] fit $seen ETA ${eta} min is within ${per_fit} min/fit; keeping" >> "$LOG"
    fi
  fi
fi
wait "$PID" 2>/dev/null; rc=$?
kill "$CAPPID" 2>/dev/null          # reap the cap timer; never outlive the target
echo "[f1] rc=$rc" >> "$LOG"
