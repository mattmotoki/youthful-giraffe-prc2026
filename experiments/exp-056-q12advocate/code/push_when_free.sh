#!/usr/bin/env bash
# exp-056-q12advocate: push the December-rotation kernels once Kaggle frees a slot.
# Kaggle caps a user at 5 CPU and 2 GPU batch sessions; the first push of these two was refused.
# L2 waits for a July/January CPU kernel to finish, F1 for a July/January GPU kernel.
#
#   nohup bash experiments/exp-056-q12advocate/code/push_when_free.sh > logs/push-when-free.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
K="$HOME/.local/bin/kaggle"
PUSH="$ROOT/experiments/exp-056-q12advocate/code/push_iso.sh"
done_any() {   # true when any named kernel is no longer running or queued
    for k in "$@"; do
        s="$("$K" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$k" 2>&1 | tail -1)"
        case "$s" in *RUNNING*|*QUEUED*) ;; *) echo "$(date -u +%FT%TZ) $k: $s"; return 0 ;; esac
    done
    return 1
}
cpu_pushed=0; gpu_pushed=0
while [ $cpu_pushed -eq 0 ] || [ $gpu_pushed -eq 0 ]; do
    if [ $cpu_pushed -eq 0 ] && done_any prc-iso-t-tailv4-jul prc-iso-t-tailv4-jan prc-iso-t-tailv4-dec prc-iso-l2-hub8725-jul prc-iso-l2-hub8725-jan; then
        bash "$PUSH" L2-hub8725 dec iso_dec_v iso_dec_s iso_dec_i iso_dec_x && cpu_pushed=1
    fi
    if [ $gpu_pushed -eq 0 ] && done_any prc-iso-f1-fresh-jul prc-iso-f1-fresh-jan; then
        bash "$PUSH" F1-fresh dec iso_dec_v iso_dec_s iso_dec_i iso_dec_x && gpu_pushed=1
    fi
    [ $cpu_pushed -eq 1 ] && [ $gpu_pushed -eq 1 ] && break
    sleep 300
done
echo "$(date -u +%FT%TZ) both December kernels pushed"
