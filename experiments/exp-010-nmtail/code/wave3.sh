set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH=.:experiments/exp-010-nmtail/code:experiments/exp-003-anatomy/code
export OMP_NUM_THREADS=8
P=.venv/bin/python
# one canonical configuration for every arm: prc.nmtail VERSION 2 defaults
$P experiments/exp-010-nmtail/code/run_folds.py --folds july,july_loo,december,january --modes phi --name C
$P experiments/exp-010-nmtail/code/run_folds.py --folds july,july_loo,december,january --modes p --name C
$P experiments/exp-010-nmtail/code/run_folds.py --folds july_loo,december --modes phi_screened --name C
