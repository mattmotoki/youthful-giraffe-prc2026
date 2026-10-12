#!/usr/bin/env bash
# exp-040-chain: report each Kaggle session's end to the desk the moment it lands.
#
#   bash experiments/exp-040-chain/code/watch_kernels.sh
#
# The desk's slot cap is 5 batch CPU sessions and the harvest is waiting behind
# these four, so a session's end is reported as its own line rather than in one
# batch at the end.  A status read that does not come back as a
# `KernelWorkerStatus` is treated as UNKNOWN and never as "finished": a
# transient API error once produced a false "all done" here.
set -u
K="$HOME/.local/bin/kaggle"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DESK="/path/to/competitions/tools/desk.sh"
SEEN="$ROOT/experiments/exp-040-chain/logs/.reported"
touch "$SEEN"
for k in july-loo december january ranking; do
  grep -qx "$k" "$SEEN" && continue
  s=$("$K" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/prc-chain-$k" 2>&1 | sed 's/.*status //')
  case "$s" in
    *KernelWorkerStatus*)
      case "$s" in
        *RUNNING*|*QUEUED*) echo "$(date +%H:%M) $k still $s" ;;
        *)
          echo "$k" >> "$SEEN"
          echo "$(date +%H:%M) $k ENDED $s"
          bash "$ROOT/experiments/exp-040-chain/code/pull_kernel.sh" \
               "$(echo "$k" | tr '-' '_')" >/dev/null 2>&1 || true
          bash "$DESK" report "$ROOT" chain working \
            "Kaggle CPU session prc-chain-$k has ended ($s) and its slot is free; outputs pulled to experiments/exp-040-chain/." ;;
      esac ;;
    *) echo "$(date +%H:%M) $k UNKNOWN (status read failed), treated as still running" ;;
  esac
done
n=$(grep -c . "$SEEN" 2>/dev/null); n=${n:-0}
echo "reported $n of 4"
[ "$n" -ge 4 ] && exit 0 || exit 9
