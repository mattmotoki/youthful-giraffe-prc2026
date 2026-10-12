#!/usr/bin/env bash
# The LightGBM family: the linear-leaf reference (re-measured under the fixed
# round protocol, because exp-001's arms were withdrawn by its protocol-halt),
# the plain-tree sibling, the Huber sibling, and the monotone sibling on the raw
# target where the physics claim it encodes is actually true.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=0
R() { read -r A B < <($P rounds_of.py "$1"); echo "$A $B"; }

set -- $(R S-lgb-lin);   $P run.py --kind lgb --name L0-lin  --variant lin   --rounds-m6 "$1" --rounds-m11 "$2"
$P run.py --select --kind lgb --name S-lgb-plain --variant plain --folds july,december --max-rounds 3000 --seed 7
set -- $(R S-lgb-plain); $P run.py --kind lgb --name L1-plain --variant plain --rounds-m6 "$1" --rounds-m11 "$2"
set -- $(R S-lgb-hub);   $P run.py --kind lgb --name L2-hub   --variant plain --huber-delta 1200 --rounds-m6 "$1" --rounds-m11 "$2"
set -- $(R S-lgb-mono);  $P run.py --kind lgb --name L3-monoraw --variant plain --monotone --target raw --rounds-m6 "$1" --rounds-m11 "$2"
set -- $(R S-lgb-plain); $P run.py --kind lgb --name L3r-monores --variant plain --monotone --rounds-m6 "$1" --rounds-m11 "$2"
