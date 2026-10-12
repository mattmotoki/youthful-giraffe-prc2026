#!/usr/bin/env bash
# Unit 012: after the two drop0 pack lanes (d0pA, d0pB) end complete, check every new drop0 fit
# (check_sweep), then the drop0 six-seed vectors (vectors011 --arm drop0), the mixed twelve-model
# vectors (mix012_vectors) and the level-2 pack (pack_level2_012). Each step runs through launch.sh (own
# write-once log, nice 19, CPUs 24,25,88,89) and must write its record; the first failure stops it.
# usage: setsid nohup bash chain012.sh > logs/mlp008-chain012.log 2>&1 < /dev/null &
set -euo pipefail
w=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp164-work
c=$w/code/mlp008
r=$w/results/mlp-011
step() {   # step NAME RECORD ARGS...: start one launch.sh step, wait for its pid, need its record
    local name=$1 record=$2; shift 2
    if [[ -e "$record" ]]; then echo "$(date -u +%FT%TZ) skip $name (record exists)"; return; fi
    bash "$c/launch.sh" "$name" 24,25,88,89 "$@"
    local pid; pid=$(cat "$w/logs/mlp008-$name.pid")
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
    if [[ ! -e "$record" ]]; then echo "$(date -u +%FT%TZ) FAILED $name: no $record"; tail -5 "$w/logs/mlp008-$name.log"; exit 1; fi
    echo "$(date -u +%FT%TZ) done $name"
}
for q in d0pA d0pB; do
    pid=$(cat "$w/logs/mlp008-$q.pid")
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    python3 -c "import json,sys; sys.exit(0 if json.load(open('$w/results/mlp-008/QUEUE-$q.json'))['state'] == 'complete' else 1)" \
        || { echo "$(date -u +%FT%TZ) FAILED lane $q"; exit 1; }
    echo "$(date -u +%FT%TZ) lane $q complete"
done
for f in final jjx-02 jjx-03 jjx-06 jjx-08 jjx-10 jjx-11 jjx-12; do
    for s in "$f" "$f-seeds-703-704-705-706"; do
        step "chk-drop0-$s" "$r/CHECK-drop0-$s.json" check_sweep --arm drop0 --sub "$s"
    done
done
step drop0-vectors011 "$w/out/mlp-011/vectors/BAG-drop0.json" vectors011 --arm drop0 --jjrank-subs jjrank jjrank-seeds-703-704-705-706
step mix-vectors012 "$w/out/mlp-012/vectors/BAG-mix.json" mix012_vectors
step mix-pack012 "$w/out/mlp-012/pack-level2/PACK.json" pack_level2_012
echo "$(date -u +%FT%TZ) chain complete"
