#!/usr/bin/env bash
# exp-040-chain step 6: push one GPU kernel into a slot k2climb leaves free.
#
#   bash experiments/exp-040-chain/code/take_gpu_slot.sh            # kernels/c2
#   bash experiments/exp-040-chain/code/take_gpu_slot.sh c2-december
#
# Kaggle's GPU cap is 2 and k2climb owns them when it is running. The desk's rule
# (2026-09-08): ONE GPU session at a time, only in a slot k2climb leaves free, and
# never stop theirs. So this script only ever PUSHES ITS OWN kernel, and only when
# it has counted a free slot AND confirmed that no exp-040 GPU session of its own
# is still running. It pushes once per kernel: the marker is written before the
# push, so a re-run after a crash cannot start a second session.
#
# Exit 0 = pushed (or already pushed), 9 = no free slot yet or one of ours is
# still running, 1 = the peer list could not be read and nothing was done.
set -u
DIR="${1:-c2}"
K="$HOME/.local/bin/kaggle"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-040-chain"
KDIR="$EXP/kernels/$DIR"
MARK="$EXP/logs/.pushed-$DIR"
CAP=2

[ -d "$KDIR" ] || { echo "no kernel dir $KDIR"; exit 1; }
ID=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" \
     "$KDIR/kernel-metadata.json") || exit 1

if [ -e "$MARK" ]; then
  echo "already pushed: $(cat "$MARK")"
  exit 0
fi

# --- ours: one GPU session at a time -----------------------------------------
# Every exp-040 GPU kernel that has been pushed is checked, not just the previous
# one, because a marker says a push happened and says nothing about whether that
# session has ended.
for m in "$EXP"/logs/.pushed-*; do
  [ -e "$m" ] || continue
  d=$(basename "$m"); d=${d#.pushed-}
  [ "$d" = "$DIR" ] && continue
  [ -f "$EXP/kernels/$d/kernel-metadata.json" ] || continue
  oid=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" \
        "$EXP/kernels/$d/kernel-metadata.json") || continue
  os=$("$K" kernels status "$oid" 2>&1 | sed 's/.*status //')
  case "$os" in
    *RUNNING*|*QUEUED*) echo "ours still running: $oid $os"; exit 9 ;;
    *KernelWorkerStatus*) : ;;
    *) echo "OURS UNREADABLE $oid $os -- treating as running"; exit 9 ;;
  esac
done

# --- theirs: only a slot k2climb leaves free ---------------------------------
THEIRS=$(python3 "$EXP/code/gpu_peers.py") || {
  echo "cannot read the peer GPU kernel list; refusing to guess at a free slot"
  exit 1
}
busy=0
unknown=0
for id in $THEIRS; do
  s=$("$K" kernels status "$id" 2>&1 | sed 's/.*status //')
  case "$s" in
    *RUNNING*|*QUEUED*) busy=$((busy + 1)); echo "busy    $id $s" ;;
    *KernelWorkerStatus*) : ;;
    *) unknown=$((unknown + 1)); echo "UNKNOWN $id $s" ;;
  esac
done
echo "$(date +%H:%M) peers hold $busy of $CAP GPU slots ($unknown unreadable)"

# An unreadable status counts as BUSY. Taking a slot that is not free would queue
# this session behind k2climb's work, which is what the desk's rule exists to
# prevent, so silence is treated as occupied and never as free.
if [ $((busy + unknown)) -ge "$CAP" ]; then
  echo "no free GPU slot"
  exit 9
fi

date -u +"%Y-%m-%dT%H:%M:%SZ pushed $ID into a free GPU slot" > "$MARK"
cd "$KDIR" && "$K" kernels push -p .
