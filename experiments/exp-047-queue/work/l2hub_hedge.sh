#!/usr/bin/env bash
# §7bk hedge: L2-hub on november_loo, LOCAL, uncapped rounds, 4 threads, 3 h wall cap.
#
# The §7bj/§7bk laptop run tripped the 20-minute cap with NO iteration marker,
# because run.py leaves LightGBM's verbosity off and the output was block
# buffered.  This fixes both: `python -u`, and `lgb.train` is wrapped to append a
# callback that prints `[iter] n/total elapsed` every 100 rounds.  So even if this
# is killed -- by the 3 h cap or because the Kaggle fetch lands first -- a
# projection exists.
#
# import lane; lane.install(); lane.family() is injected at runtime (§7bj); frames.py
# is shared code and this lane does not edit it.
set -u
FOLD="${PRC_LANE_FOLD:-november_loo}"
ROOT=/path/to/competitions/prc-data-challenge-2026
OUT="$ROOT/experiments/exp-047-queue/work/hedge"
LOG="$ROOT/experiments/exp-047-queue/logs/hedge-L2-hub-$FOLD.log"
CAP=10800                       # 3 hours
mkdir -p "$OUT"

OMP_NUM_THREADS=4 PRC_THREADS=4 PYTHONPATH="$ROOT" \
"$ROOT/.venv/bin/python" -u -c "
import sys, runpy, time
sys.path.insert(0, 'experiments/exp-008-diversity')
sys.path.insert(0, '$ROOT/experiments/exp-047-queue/work')
import frames
import lane; lane.install(); lane.family()
import lightgbm as lgb
_t0 = time.time()
def _mark(env):
    n = env.iteration + 1
    if n % 100 == 0 or n == env.end_iteration:
        el = time.time() - _t0
        rate = n / el if el > 0 else 0.0
        left = (env.end_iteration - n) / rate if rate > 0 else float('nan')
        print(f'[iter] {n}/{env.end_iteration} {el:.1f}s {rate:.2f} it/s '
              f'eta {left/60:.1f} min', flush=True)
_orig = lgb.train
def _train(*a, **k):
    cbs = list(k.get('callbacks') or [])
    cbs.append(_mark)
    k['callbacks'] = cbs
    return _orig(*a, **k)
lgb.train = _train
sys.argv = ['experiments/exp-008-diversity/run.py',
            '--kind', 'lgb', '--name', 'L2-hub8725', '--variant', 'plain',
            '--huber-delta', '1200', '--folds', '$FOLD',
            '--rounds-m6', '3225', '--rounds-m11', '8725', '--outdir', '$OUT']
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('e8run', 'experiments/exp-008-diversity/run.py')
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
_mod.HERE = __import__('pathlib').Path('$ROOT/experiments/exp-047-queue/work/e8-results')
_mod.HERE.mkdir(parents=True, exist_ok=True)
print('[lane] exp-008 HERE repointed to', _mod.HERE, flush=True)
_mod.main()
" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$ROOT/experiments/exp-047-queue/logs/hedge-L2-hub.pid"
( sleep "$CAP"; kill "$PID" 2>/dev/null && echo "[hedge] 3 h cap reached, killed" >> "$LOG" ) &
CAPPID=$!
wait "$PID"
rc=$?
# Reap the cap timer.  It outlives the fit it guards, and macOS recycles pids,
# so a timer left running fires a bare kill at whatever holds $PID three hours
# later -- another desk's job.  Measured 2026-09-09: one such timer survived
# 90 min past its target.
kill "$CAPPID" 2>/dev/null
echo "[hedge] rc=$rc" >> "$LOG"
# the exp-008 producer appends to two shared files; leave them as they were.
cd "$ROOT" && git checkout -- experiments/exp-008-diversity/guard_bounds.json \
                             experiments/exp-008-diversity/climb.log 2>/dev/null
