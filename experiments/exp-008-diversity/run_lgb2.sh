#!/usr/bin/env bash
# Season-matched july_loo round counts for the LightGBM family, then the repaired
# july_loo folds. Same reasoning as run_gpu3.sh: july_loo's "last training month"
# is 2025-12, so every arm's eleven-month budget for a July validation month was
# chosen on calm winter. july_loo is 55.7 % of the board estimate.
set -u
cd "$(dirname "$0")"
P=../../.venv/bin/python
export PRC_THREADS=16 PRC_GPU=0
while pgrep -f "run_lgb[.]sh" >/dev/null; do sleep 30; done
R() { $P rounds_of.py "$@"; }
for a in "lin:S-lgb-lin:L0-lin::" "plain:S-lgb-plain:L1-plain::" "mono:S-lgb-mono:L3-monoraw:--monotone:--target raw"; do
  IFS=: read -r v sel name f1 f2 <<< "$a"
  $P run.py --select --kind lgb --name "S-lgb-$v-jloo" --variant "$( [ "$v" = lin ] && echo lin || echo plain )" \
      $f1 $f2 --folds july_loo --es-month 2025-08 --max-rounds 3000 --seed 7
  S=$(R "S-lgb-$v-jloo" | cut -d' ' -f2); echo "$name m11s=$S"
  set -- $(R "$sel")
  $P run.py --kind lgb --name "$name" --variant "$( [ "$v" = lin ] && echo lin || echo plain )" $f1 $f2 \
      --folds july_loo --rounds-m6 "$1" --rounds-m11 "$2" --rounds-m11s "$S"
done
$P run.py --select --kind lgb --name S-lgb-hub-jloo --variant plain --huber-delta 1200 --folds july_loo --es-month 2025-08 --max-rounds 4000 --seed 7
S=$(R S-lgb-hub-jloo | cut -d' ' -f2); echo "L2-hub m11s=$S"
set -- $(R S-lgb-hub)
$P run.py --kind lgb --name L2-hub --variant plain --huber-delta 1200 --folds july_loo --rounds-m6 "$1" --rounds-m11 "$2" --rounds-m11s "$S"
