#!/bin/sh
# PORT-S7X9 after seeds 8-9 (desk 2026-10-10 04:37 UTC), one chain run through the hold guard: copy seeds 8-9 into
# the 9-seed s7 tree with the SHA256SUMS check, checks 3-4 on 8-9, the 9-seed member (s7n_finish.sh), run (ii) (the
# 9-seed s7 alone, release L2C) and its read, then run (iii) (candidate A's L2C on the 9-seed npz of run (ii)) and its
# read. Each V run starts at MemAvailable 18 GB or more and runs under /usr/bin/time -v.
#   sh port_89_chain.sh <snapshot> <port_read.py>
set -eu
S=$1; PR=$2
F=${PRC_FINAL_ROOT:-/path/to/final}/finalv; T=$F/run-s7x9; W=${PRC_ISO5_ROOT:-/path/to/prc-isolated-5}/outputs/w5-s7seeds
PY=$F/venv312/bin/python; C=$S/experiments/exp-191-final-v/code
REFS=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet
DH=${PRC_FINAL_ROOT:-/path/to/final}/release73/rebuild/2140d645167a/release73/files/final-dh/youthful-giraffe_final.parquet
export PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
stamp() { date -u +%FT%TZ; }
memok() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 18 ]; do echo "MemAvailable below 18 GB $(stamp)"; sleep 60; done
  echo "MemAvailable GB $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) $(stamp)"
}
echo "chain start $(stamp)"
for s in 8 9; do for m in held full; do test ! -e $T/work/seq7/s7-$m-$s.npy; cp $W/s7-$m-$s.npy $T/work/seq7/; done; done
test ! -e $T/w5-SHA256SUMS-seeds4to9
cp $W/SHA256SUMS $T/w5-SHA256SUMS-seeds4to9
(cd $T/work/seq7 && sha256sum -c $T/w5-SHA256SUMS-seeds4to9)
$PY $C/s7_seed_check.py $T $F/port/s7-seed-check-8to9.json 8 9
echo "member $(stamp)"
(cd $S && sh experiments/exp-191-final-v/code/s7n_finish.sh $PY $T $F/port/parts-s7x9 $REFS 9)
V="${PRC_FINAL_ROOT:-/path/to/final}/finalplan/export-final-2"
TAIL="$F/refs-finallirf-final-a444 $F/run-ranking-cf74f4261 ${PRC_FINAL_ROOT:-/path/to/final}/finalplan/parts/final-2/z-file-stages-record.json ${PRC_FINAL_ROOT:-/path/to/final}/finalclock/parts/head.parquet a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6"
echo "run (ii) $(stamp)"; memok
mkdir -p $F/port/root-9
PRC_FINAL_ROOT=$F/port/root-9 PRC_V73_REFS=$REFS /usr/bin/time -v sh $C/v_run.sh $S $PY $V $F/run-port9-a0c484bab $T $TAIL
PRC_RANKING_MODE=final $PY $PR $F/run-final2-66b11c579 $F/run-port9-a0c484bab $F/youthful-giraffe_final.parquet $DH $F/port/read-9.json
echo "run (iii) $(stamp)"; memok
mkdir -p $F/port/root-A9
NPZ=$F/run-port9-a0c484bab/work/w2/s7c-final.npz
PRC_FINAL_ROOT=$F/port/root-A9 PRC_V73_REFS=$REFS PRC_V_L2C_DIFF=$S/experiments/exp-191-final-v/isolated5-A/l2c.diff PRC_V_L2C_S7=$NPZ \
  /usr/bin/time -v sh $C/v_run.sh $S $PY $V $F/run-portA9-a0c484bab $T $TAIL
PRC_RANKING_MODE=final $PY $PR $F/run-final2-66b11c579 $F/run-portA9-a0c484bab $F/youthful-giraffe_final.parquet $DH $F/port/read-A9.json $NPZ
sha256sum $F/port/s7-seed-check-8to9.json $F/port/parts-s7x9/* $F/port/read-9.json $F/port/read-A9.json \
  $F/port/root-9/finalv/youthful-giraffe_final.parquet $F/port/root-A9/finalv/youthful-giraffe_final.parquet
echo "chain done $(stamp)"
