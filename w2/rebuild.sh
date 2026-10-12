#!/bin/sh
# Rebuild the current best solution s7c on the workstation (x86-64 Linux), from the repository root.
# Environment: .venv with Python 3.12 and w2/requirements-remote.txt.
# The v64 chain replay on x86 differs from the Mac replay by at most 1.5e-11 s per row, so the byte check
# of iso/dump_chain.py compares values with w2/inputs/chain_ref_ranking.parquet (the Mac replay).
# Outputs: outputs/w2-l2c.parquet     (c5 + the L2C clock-row level 2)
#          outputs/w2-l2c-s7c.parquet (w2-l2c + the ORD-S sequence column on ordinary rows: the s7 recipe at 3 seeds)
# The member in w2/inputs/s7c.npz was made on a GPU (not bitwise repeatable):
#   .venv/bin/python w2/feat2.py      # work/w2/feat2_{2025,2026}.parquet, the stand and apron inputs that prepare.py reads
#   python3 w2/seq/prepare.py --nb 64 --win 10800 --out work/seq7
#   python3 w2/seq/fit.py --data work/seq7 --d 192 --layers 4 --epochs 16 --mode {held,full} --seed {1,2,3} --amp \
#     --out work/seq7/s7-{held,full}-<seed>.npy
#   python3 w2/seq/export.py s7 work/seq7          # the mean over the seed files of each mode
#   .venv/bin/python w2/ords.py --on work/w2/l2c --names s7 --lirf-jul 0.5 --save-npz w2/inputs/s7c.npz
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 NT=8
export CHAIN_REF=w2/inputs/chain_ref_ranking.parquet
PY=.venv/bin/python
sh iso/rebuild.sh                      # work/chain, work/feat, outputs/iso-c5.parquet (x86: 58a2ac91...)
$PY w2/base.py                         # work/w2/base_{fold,ranking}.parquet (c5 values and chain columns)
$PY w2/l2c.py --out outputs/w2-l2c.parquet
$PY w2/ords.py --on work/w2/l2c --npz w2/inputs/s7c.npz --lirf-jul 0.5 --out outputs/w2-l2c-s7c.parquet
shasum -a 256 outputs/iso-c5.parquet outputs/w2-l2c.parquet outputs/w2-l2c-s7c.parquet
