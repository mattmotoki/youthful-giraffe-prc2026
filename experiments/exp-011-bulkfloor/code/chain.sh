cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH=.:experiments/exp-003-anatomy/code:experiments/exp-011-bulkfloor/code
export OMP_NUM_THREADS=8
L=experiments/exp-011-bulkfloor/logs
while [ "$(grep -c '^\[shift\]' $L/q2.log 2>/dev/null)" -lt 4 ]; do sleep 60; done
.venv/bin/python experiments/exp-011-bulkfloor/code/q2_shift.py --ranking > $L/q2_ranking.log 2>&1
.venv/bin/python experiments/exp-011-bulkfloor/code/q2b_auc.py > $L/q2b.log 2>&1
touch $L/chain.done
