#!/usr/bin/env bash
# exp-194 R9 on the final base (SPEC-R9.md): one job on remote x86 under the hold guard.
#  1. build `base` (the released mlp rule through r9_build.py) and `r9` (mix12 on every member-scope row): finalplan's
#     build_final.sh steps (export copy, assembly, pack check, checker on the Z and R files, zabd parity), outputs here;
#  2. run (iii)'s V chain (finalv's v_run.sh with candidate A's l2c.diff and the 9-seed s7 npz of run (ii), ORD-S on
#     the 9-seed member; port_89_chain.sh run (iii)) on each pack;
#  3. the dh step (r9_dh.py, dh-final1) on each V file; base must give final-iii-dh c4511650;
#  4. r9_read.py (base against r9).
# Code check: base gives final-2's zabd e6503136, Z file e47b8afb and pack 29c0ace1, then V 6de02372 and dh c4511650.
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <snap>/experiments/exp-194-gain-lane/code/r9_chain.sh <tag> >/dev/null 2>&1 < /dev/null &
# Snapshot (git archive): prc, experiments/exp-146-validation/frozen, experiments/exp-164-double-down/results/stack-pack-012,
# experiments/exp-186-final-core/code, experiments/exp-191-final-v/code, experiments/exp-191-final-v/isolated5-A,
# experiments/exp-192-final-assembly, experiments/exp-194-gain-lane/code.
set -euo pipefail
TAG=$1
SNAP=$(cd "$(dirname "$0")/../../.." && pwd)
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/r9-$TAG
FP=${PRC_FINAL_ROOT:-/path/to/final}/finalplan
F=${PRC_FINAL_ROOT:-/path/to/final}/finalv
PY=$F/venv312/bin/python
SRCX=$FP/export-d07bf4a0a621
SOURCES=$SNAP/experiments/exp-192-final-assembly/results/final/final-sources-2.json
REFS=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet
T=$F/run-s7x9
NPZ=$F/run-port9-a0c484bab/work/w2/s7c-final.npz
LIRF=$F/refs-finallirf-final-a444
RRUN=$F/run-ranking-cf74f4261
HEAD=${PRC_FINAL_ROOT:-/path/to/final}/finalclock/parts/head.parquet
HEADSHA=a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6
DHPART=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/parts/dh-final1.parquet
E=experiments/exp-192-final-assembly
C=experiments/exp-194-gain-lane/code
export PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1
stamp() { date -u +%FT%TZ; }
memok() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 20 ]; do echo "MemAvailable below 20 GB $(stamp)"; sleep 60; done
  echo "MemAvailable GB $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) $(stamp)"
}
test ! -e "$G"
mkdir -p "$G/runs" "$G/parts"
echo "r9_chain $TAG start $(stamp) snapshot $SNAP pid $$"
sha256sum "$SOURCES" "$NPZ" "$DHPART" "$SNAP/experiments/exp-191-final-v/isolated5-A/l2c.diff" \
  "$SNAP/experiments/exp-164-double-down/results/stack-pack-012/mlp-fold.parquet"
cd "$SNAP"
for v in base r9; do
  P=$G/parts/$v
  test ! -e "$P" && test ! -e "$G/export-$v"
  echo "build $v: export copy $(stamp)"
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I $E/code/export_copy.py $SRCX $G/export-$v "$SNAP" > $G/runs/$v-copy.json
  echo "build $v: assembly $(stamp)"; memok
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -v $PY -I $C/r9_build.py --variant $v --repo "$SNAP" \
    --export $SRCX --sources "$SOURCES" --parts-dir "$P" --pack-out $G/export-$v --out $G/runs/$v.json \
    --fold-out "$P/fold-stages.npz"
  echo "build $v: pack check $(stamp)"
  PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I $E/code/pack_check.py \
    $G/export-$v $FP/input-root $G/runs/$v-pack-check.json
  for f in z-file r-file; do
    $PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$P/$f.parquet" "$SNAP" \
      --stages "$P/$f-stages-record.json" --out $G/runs/$v-check-$f.json || echo "check $v $f: blocking failure (see json)"
  done
  $PY -I experiments/exp-186-final-core/code/parity.py --refs "$REFS" --new "$P/zabd.parquet" --pair zabd=saved:zabd-final \
    --kind refit --out "$P/zabd-parity.json"
  sha256sum "$P"/* $G/export-$v/release/ZABD-PACK.json
done
for v in base r9; do
  echo "V chain $v $(stamp)"; memok
  mkdir -p $G/root-$v
  PRC_FINAL_ROOT=$G/root-$v PRC_V73_REFS=$REFS PRC_V_L2C_DIFF=$SNAP/experiments/exp-191-final-v/isolated5-A/l2c.diff \
    PRC_V_L2C_S7=$NPZ /usr/bin/time -v sh $SNAP/experiments/exp-191-final-v/code/v_run.sh "$SNAP" $PY $G/export-$v \
    $G/run-$v $T $LIRF $RRUN $G/parts/$v/z-file-stages-record.json $HEAD $HEADSHA
  cd "$SNAP"
done
echo "dh step $(stamp)"
$PY -I $C/r9_dh.py --v-file $G/root-base/finalv/youthful-giraffe_final.parquet --part $DHPART --raw $PRC_DATA_DIR/final \
  --out-dir $G/dh-base --expect-sha256 c45116502cad96dbb270467b9f2ae26fb6e06d9a50ca34ff214253609f44612d \
  || echo "dh base: not final-iii-dh (see the log)"
$PY -I $C/r9_dh.py --v-file $G/root-r9/finalv/youthful-giraffe_final.parquet --part $DHPART --raw $PRC_DATA_DIR/final \
  --out-dir $G/dh-r9
echo "read $(stamp)"
PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY $SNAP/$C/r9_read.py $G/run-base $G/run-r9 \
  $G/parts/base $G/parts/r9 $G/dh-base/youthful-giraffe_final.parquet $G/dh-r9/youthful-giraffe_final.parquet $G/r9-read.json
sha256sum $G/root-*/finalv/youthful-giraffe_final.parquet $G/dh-*/youthful-giraffe_final.parquet $G/r9-read.json
echo "r9_chain $TAG done $(stamp)"
