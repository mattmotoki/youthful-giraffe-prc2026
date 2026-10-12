#!/usr/bin/env bash
# exp-037-tail, every step in dependency order, from data/raw alone.
# 6 threads throughout (the desk released the laptop).  No GPU, no upload.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export PRC_N_JOBS=${PRC_N_JOBS:-6}
P=.venv/bin/python
C=experiments/exp-037-tail/code
L=experiments/exp-037-tail/logs
mkdir -p "$L" experiments/exp-037-tail/{work,analysis}

$P $C/test_chain.py 2>&1 | tee $L/test_chain.log   # the chain against brute force
$P $C/build.py      2>&1 | tee $L/build.log        # every departure, chained
$P $C/frames.py     2>&1 | tee $L/frames.log       # the noac frames, chained
$P $C/checks.py     2>&1 | tee $L/checks.log      # population and labels vs exp-033
$P $C/step1.py      2>&1 | tee $L/step1.log        # existence, gap, purity
$P $C/step2.py      2>&1 | tee $L/step2.log        # the 388 through the chain
$P $C/step3.py      2>&1 | tee $L/step3.log        # the placeholder question
$P $C/mechanism.py  2>&1 | tee $L/mechanism.log    # the exposure test (both readings withdrawn)
$P $C/rotation_check.py 2>&1 | tee $L/rotation_check.log  # the airline split that decides it
$P $C/head.py       2>&1 | tee $L/head.log         # the two heads, out of fold
$P $C/evaluate.py   2>&1 | tee $L/evaluate.log     # LOMO and the fold arm
$P $C/repair.py     2>&1 | tee $L/repair.log       # four carriers, shrunk out of fold
$P $C/blend.py      2>&1 | tee $L/blend.log        # the second repair, a blend
$P $C/why.py        2>&1 | tee $L/why.log          # the diagnosis
$P $C/bulk.py       2>&1 | tee $L/bulk.log         # §7: the bound on ordinary rows
$P $C/bulk_clip.py  2>&1 | tee $L/bulk_clip.log    # §7: the clip on them, scored out of fold
# code/serve.py is NOT run: the gate in RESULTS.md §5 is not met.
