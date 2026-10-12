#!/usr/bin/env bash
# exp-056-q12advocate: push the three matched-size exposure kernels (push_thin.sh) when Kaggle
# frees a slot (caps: 2 GPU and 5 CPU batch sessions per user).
#   F1-fresh thin waits until prc-iso-f1-fresh-dec exists (push_when_free.sh pushes it) and fewer
#   than 2 of these GPU kernels run; L2-hub8725 thin, then T-tailv4 thin, wait until fewer than 5
#   of these CPU kernels run. A refused push (the kernel does not exist afterwards) removes its
#   unpushed directory and tries again at the next poll.
#
#   nohup bash experiments/exp-056-q12advocate/code/push_thin_when_free.sh > logs/push-thin.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
K="$HOME/.local/bin/kaggle"
PUSH="$ROOT/experiments/exp-056-q12advocate/code/push_thin.sh"
KDIR="$ROOT/experiments/exp-056-q12advocate/kernels"
GPU_SLUGS="prc-iso-f1-fresh-jul prc-iso-f1-fresh-jan prc-iso-f1-fresh-dec prc-iso-f1-fresh-thin"
CPU_SLUGS="prc-iso-t-tailv4-jul prc-iso-t-tailv4-jan prc-iso-t-tailv4-dec prc-iso-t-tailv4-thin
prc-iso-l2-hub8725-jul prc-iso-l2-hub8725-jan prc-iso-l2-hub8725-dec prc-iso-l2-hub8725-thin"
status() { "$K" kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$1" 2>&1 | tail -1; }
active() {
    local n=0 k
    for k in $1; do case "$(status "$k")" in *RUNNING*|*QUEUED*) n=$((n + 1)) ;; esac; done
    echo "$n"
}
exists() { case "$(status "$1")" in *RUNNING*|*QUEUED*|*COMPLETE*|*ERROR*|*CANCEL*) return 0 ;; *) return 1 ;; esac; }
try_push() {
    local m="$1" low
    low="$(echo "$m" | tr 'A-Z' 'a-z')"
    bash "$PUSH" "$m" iso_jul_xm iso_jan_sm iso_dec_sm
    sleep 30
    if exists "prc-iso-$low-thin"; then
        echo "$(date -u +%FT%TZ) $m thin pushed: $(status "prc-iso-$low-thin")"
        return 0
    fi
    echo "$(date -u +%FT%TZ) $m thin push refused; removing the unpushed directory"
    rm -rf "$KDIR/iso-$m-thin"
    return 1
}
gpu=1; l2=1; tt=1
while [ $((gpu + l2 + tt)) -gt 0 ]; do
    if [ $gpu -eq 1 ] && exists prc-iso-f1-fresh-dec && [ "$(active "$GPU_SLUGS")" -lt 2 ]; then
        try_push F1-fresh && gpu=0
    fi
    if [ $l2 -eq 1 ] && [ "$(active "$CPU_SLUGS")" -lt 5 ]; then try_push L2-hub8725 && l2=0; fi
    if [ $tt -eq 1 ] && [ "$(active "$CPU_SLUGS")" -lt 5 ]; then try_push T-tailv4 && tt=0; fi
    [ $((gpu + l2 + tt)) -eq 0 ] && break
    sleep 300
done
echo "$(date -u +%FT%TZ) all three thin kernels pushed"
