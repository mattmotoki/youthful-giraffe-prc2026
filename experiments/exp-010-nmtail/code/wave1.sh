set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH=.:experiments/exp-010-nmtail/code:experiments/exp-003-anatomy/code
export OMP_NUM_THREADS=8
P=.venv/bin/python
$P experiments/exp-010-nmtail/code/run_folds.py --folds july_loo --modes phi_screened,p --name S2 --rounds 900 --lr 0.04
$P experiments/exp-010-nmtail/code/run_folds.py --folds july,december,january --modes phi --name S2 --rounds 900 --lr 0.04
