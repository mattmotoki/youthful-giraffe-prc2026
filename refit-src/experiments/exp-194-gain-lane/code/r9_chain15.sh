#!/usr/bin/env bash
# exp-194 R9 on finalv's 15-seed s7 base (SPEC-R9-15.md): one job on remote x86 under the hold guard, CPU only.
#  1. checks: finalv's run (v) is finished; the chain code of my f1 snapshot equals finalv's; 15 + 15 s7 seed files;
#  2. finalv's v_run.sh on the R9 pack of chain f1 (export-r9) with candidate A's l2c.diff, the 15-seed npz of finalv's
#     run (iv) and the s7 tree run-s7x15; c5 must be c5-r9 0f828024 (c5 reads no s7 value);
#  3. r9_dh.py on finalv's run (v) V file (base) and on the R9-15 V file; stages record and checker on both dh files;
#  4. reads: r9_read.py (base tree = finalv's run (v)), r9_wo5_days.py, r9_lirf_rows.py, r9_dh_rows.py; SHA-256 lines;
#  5. LD variant (SPEC-LD.md): ORD-S with --long-days on hard-link copies of the R9-15 tree (flag off = R9-15 bytes),
#     the LD-15 V file, dh step, checker on both, ld_read.py (LD-15 against R9-15).
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <code dir>/r9_chain15.sh <tag> <finalv snapshot repo-TAG> <code dir> >/dev/null 2>&1 < /dev/null &
# <code dir>: git archive of experiments/exp-194-gain-lane/code at this commit (my scripts).
set -euo pipefail
TAG=$1; FSNAP=$(readlink -f "$2"); CODE=$(readlink -f "$3")
FTAG=$(basename "$FSNAP"); FTAG=${FTAG#repo-}
SNAP=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/snap-046b08bea
G1=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/r9-f1
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/r9-15-$TAG
F=${PRC_FINAL_ROOT:-/path/to/final}/finalv
P=$F/port15
PY=$F/venv312/bin/python
T=$F/run-s7x15
R4=$F/run-port15-$FTAG
R5=$F/run-portA15-$FTAG
NPZ=$R4/work/w2/s7c-final.npz
BV=$P/root-A15/finalv
REFS=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet
LIRF=$F/refs-finallirf-final-a444
RRUN=$F/run-ranking-cf74f4261
HEAD=${PRC_FINAL_ROOT:-/path/to/final}/finalclock/parts/head.parquet
HEADSHA=a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6
DHPART=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/parts/dh-final1.parquet
C5R9=0f8280246f02162d4232577af04f9a23bdf78dd3bc29028536fdbbce18fcf56a
export PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1
RAW=$PRC_DATA_DIR/final
stamp() { date -u +%FT%TZ; }
die() { echo "STOP: $* $(stamp)"; exit 1; }
memok() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 20 ]; do echo "MemAvailable below 20 GB $(stamp)"; sleep 60; done
  echo "MemAvailable GB $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) $(stamp)"
}
codelist() {  # $1: snapshot root; the chain code of finalv's V runs
  (cd "$1/experiments/exp-191-final-v/code" && { ls v_run.sh final_run.sh v_tail.sh setup_run.sh;
    find iso w2 -type f \( -name '*.py' -o -name '*.sh' \); } | LC_ALL=C sort | xargs sha256sum)
  (cd "$1/experiments/exp-191-final-v" && sha256sum isolated5-A/l2c.diff)
}
test ! -e "$G"
echo "r9_chain15 $TAG start $(stamp) pid $$ finalv snapshot $FSNAP (TAG $FTAG) code $CODE"

# 1. checks
for f in "$BV/youthful-giraffe_final.parquet" "$BV/parts/v.parquet" "$BV/parts/v-file-stages-record.json" \
         "$P/read-A15-vs-iii.json" "$R5/outputs/w2-l2c-s7c.parquet" "$R5/outputs/iso-c5.parquet" "$NPZ"; do
  test -s "$f" || die "missing $f"
done
diff <(codelist "$SNAP") <(codelist "$FSNAP") > /dev/null || { diff <(codelist "$SNAP") <(codelist "$FSNAP") || true; die "chain code differs"; }
echo "chain code equal: $(codelist "$SNAP" | wc -l) files"
nh=$(ls "$T"/work/seq7/s7-held-*.npy | wc -l); nf=$(ls "$T"/work/seq7/s7-full-*.npy | wc -l)
[ "$nh" = 15 ] && [ "$nf" = 15 ] || die "s7 seed files held $nh full $nf"
echo "s7 seed files: held $nh full $nf"
cmp -s "$BV/youthful-giraffe_final.parquet" "$R5/outputs/w2-l2c-s7c.parquet" || die "run (v) V file differs from its run output"
sha256sum "$NPZ" "$R5/work/w2/s7c-final.npz" "$BV/youthful-giraffe_final.parquet" "$BV/parts/v.parquet" \
  "$R5/outputs/iso-c5.parquet" "$R5/outputs/w2-l2c.parquet" "$G1/export-r9/release/ZABD-PACK.json" \
  "$G1/parts/r9/z-file-stages-record.json" "$SNAP/experiments/exp-191-final-v/isolated5-A/l2c.diff" "$DHPART"
mkdir -p "$G"

# 2. the R9 V chain on the 15-seed s7
echo "V chain r9-15 $(stamp)"; memok
mkdir -p "$G/root-r9"
cd "$SNAP"
PRC_FINAL_ROOT=$G/root-r9 PRC_V73_REFS=$REFS PRC_V_L2C_DIFF=$SNAP/experiments/exp-191-final-v/isolated5-A/l2c.diff \
  PRC_V_L2C_S7=$NPZ /usr/bin/time -v sh "$SNAP/experiments/exp-191-final-v/code/v_run.sh" "$SNAP" "$PY" "$G1/export-r9" \
  "$G/run-r9" "$T" "$LIRF" "$RRUN" "$G1/parts/r9/z-file-stages-record.json" "$HEAD" "$HEADSHA"
cd "$SNAP"
got=$(sha256sum "$G/run-r9/outputs/iso-c5.parquet" | cut -d' ' -f1)
[ "$got" = "$C5R9" ] || die "c5 $got is not c5-r9 $C5R9"
echo "c5 equal c5-r9 $C5R9"
# v_run.sh runs the checker with "|| echo"; the job stops here unless its verdict is PASS
$PY -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["verdict"] == "PASS" else 1)' \
  "$G/root-r9/finalv/parts/v-check.json" || die "check V r9-15: verdict is not PASS (see json)"

# 3. dh
echo "dh step $(stamp)"
$PY -I "$CODE/r9_dh.py" --v-file "$BV/youthful-giraffe_final.parquet" --part "$DHPART" --raw "$RAW" --out-dir "$G/dh-base"
$PY -I "$CODE/r9_dh.py" --v-file "$G/root-r9/finalv/youthful-giraffe_final.parquet" --part "$DHPART" --raw "$RAW" \
  --out-dir "$G/dh-r9"
for v in base r9; do
  if [ $v = base ]; then VD=$BV; else VD=$G/root-r9/finalv; fi
  $PY -I "$CODE/r9_dh_stages.py" "$VD/parts/v-file-stages-record.json" "$VD/youthful-giraffe_final.parquet" \
    "$G/dh-$v/youthful-giraffe_final.parquet" "$RAW" "$G/dh-$v/dh-file-stages-record.json"
  $PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$G/dh-$v/youthful-giraffe_final.parquet" "$SNAP" \
    --stages "$G/dh-$v/dh-file-stages-record.json" --out "$G/dh-$v/dh-check.json" || die "check dh $v: blocking failure (see json)"
done

# 4. reads
echo "read $(stamp)"
PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY "$CODE/r9_read.py" "$R5" "$G/run-r9" \
  "$G1/parts/base" "$G1/parts/r9" "$G/dh-base/youthful-giraffe_final.parquet" "$G/dh-r9/youthful-giraffe_final.parquet" \
  "$G/r9-read.json"
PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY "$CODE/r9_wo5_days.py" "$R5" "$G/run-r9"
OMP_NUM_THREADS=2 $PY -I "$CODE/r9_lirf_rows.py" "$G1/parts/r9/zabd-stages.parquet" "$R5/outputs/iso-c5.parquet" \
  "$G/run-r9/outputs/iso-c5.parquet" "$RAW" "$BV/parts/v.parquet" "$G/root-r9/finalv/parts/v.parquet"
OMP_NUM_THREADS=2 $PY -I "$CODE/r9_dh_rows.py" "$G1/parts/r9/zabd-stages.parquet" "$BV/youthful-giraffe_final.parquet" \
  "$G/root-r9/finalv/youthful-giraffe_final.parquet" "$G/dh-base/youthful-giraffe_final.parquet"
# the R9-15 dh file against f1's R9 dh file on the 14,587 dh rows (SPEC-R9-15: byte-equal there)
$PY -I -c 'import sys, numpy as np, pandas as pd
Y = "TAXITIME_SEC_mvt"
v, d15, d9 = (pd.read_parquet(p)[Y].to_numpy(float) for p in sys.argv[1:4])
m = v != d15
print("R9DH15VS9", {"dh_rows": int(m.sum()), "equal_on_dh_rows": bool(np.array_equal(d15[m].view(np.uint64), d9[m].view(np.uint64))),
      "max_abs_on_dh_rows": float(np.abs(d15[m] - d9[m]).max())})' "$G/root-r9/finalv/youthful-giraffe_final.parquet" \
  "$G/dh-r9/youthful-giraffe_final.parquet" "$G1/dh-r9/youthful-giraffe_final.parquet"
sha256sum "$G"/root-r9/finalv/youthful-giraffe_final.parquet "$G"/root-r9/finalv/parts/* "$G"/run-r9/outputs/* \
  "$G"/run-r9/work/w2/s7c-final.npz "$G"/dh-*/* "$G"/r9-read.json

# 5. LD variant (SPEC-LD.md; SPEC-R9-15.md "LD variant"): ORD-S with --long-days (ords_ld.py) on hard-link copies of
#    the R9-15 tree, as ld_chain.sh on f1: run-off15 (no flag) must give the R9-15 ORD-S bytes; run-ld15 gives LD-15.
echo "LD variant $(stamp)"; memok
R1=$G/run-r9
for p in work/w2/ords_fold.npy work/w2/ords_ranking.npy work/w2/s7c-final.npz outputs/w2-l2c-s7c.parquet outputs/stages-v.json; do
  test -s "$R1/$p" || die "R9-15 tree has no $p"
done
for v in off15 ld15; do
  rsync -a --link-dest="$R1/" --exclude='/work/w2/ords_fold.npy' --exclude='/work/w2/ords_ranking.npy' \
    --exclude='/work/w2/s7c-final.npz' --exclude='/outputs/w2-l2c-s7c.parquet' --exclude='/outputs/stages-v.json' \
    --exclude='/outputs/noac-roll-check.json' --exclude='/outputs/fold-shift-read.json' \
    --exclude='/outputs/l2c-clock-check.json' "$R1/" "$G/run-$v/"
  cp "$CODE/ords_ld.py" "$G/run-$v/w2/ords_ld.py"
done
for v in off15 ld15; do
  if [ $v = ld15 ]; then FLAG=--long-days; else FLAG=; fi
  ROOT=$G/root-$v; PARTS=$ROOT/finalv/parts; VFILE=$ROOT/finalv/youthful-giraffe_final.parquet
  test ! -e "$ROOT"
  echo "tail run-$v flags [$FLAG] $(stamp)"
  ( cd "$G/run-$v"
    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PRC_RANKING_MODE=final
    $PY w2/ords_ld.py --on work/w2/l2c --names s7 --lirf-jul 0.5 $FLAG --save-npz work/w2/s7c-final.npz \
      --out outputs/w2-l2c-s7c.parquet
    $PY "$SNAP/experiments/exp-191-final-v/code/stages_record.py" . outputs/stages-v.json > /dev/null
    $PY "$SNAP/experiments/exp-191-final-v/code/v_part.py" . "$PARTS" "$VFILE"
    cp outputs/stages-v.json "$PARTS/v-stages-record.json" )
done
for p in work/w2/ords_fold.npy work/w2/ords_ranking.npy work/w2/s7c-final.npz outputs/w2-l2c-s7c.parquet; do
  cmp "$G/run-off15/$p" "$R1/$p" || die "flag off: $p differs from R9-15"
done
$PY -c 'import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["rows"] == b["rows"] and a["stages"] == b["stages"], "stages differ"' "$G/run-off15/outputs/stages-v.json" \
  "$R1/outputs/stages-v.json" || die "flag off: stages-v.json stages differ from R9-15"
cmp "$G/root-off15/finalv/parts/v.parquet" "$G/root-r9/finalv/parts/v.parquet" || die "flag off: part v differs from R9-15"
echo "FLAGOFF15 equal to R9-15 $(stamp)"
P=$G/root-ld15/finalv/parts
VLD=$G/root-ld15/finalv/youthful-giraffe_final.parquet
$PY -c 'import json, sys
z, v, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])
names = [s["name"] for s in z["stages"]]
k = names.index("T11b-L")
st = [dict(s, name="Z/" + s["name"]) for s in z["stages"][:k + 1]] + v["stages"]
assert len({s["name"] for s in st}) == len(st)
json.dump({"z_record": sys.argv[1], "v_record": sys.argv[2], "z_stages_after_v64_left_out": names[k + 1:],
           "stages": st}, open(out, "x"), indent=1)' \
  "$G1/parts/r9/z-file-stages-record.json" "$P/v-stages-record.json" "$P/v-file-stages-record.json"
$PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$VLD" "$SNAP" \
  --stages "$P/v-file-stages-record.json" --out "$P/v-check.json" || die "check V ld15: blocking failure (see json)"
$PY -I "$CODE/r9_dh.py" --v-file "$VLD" --part "$DHPART" --raw "$RAW" --out-dir "$G/dh-ld15"
$PY -I "$CODE/r9_dh_stages.py" "$P/v-file-stages-record.json" "$VLD" "$G/dh-ld15/youthful-giraffe_final.parquet" "$RAW" \
  "$G/dh-ld15/dh-file-stages-record.json"
$PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$G/dh-ld15/youthful-giraffe_final.parquet" "$SNAP" \
  --stages "$G/dh-ld15/dh-file-stages-record.json" --out "$G/dh-ld15/dh-check.json" || die "check dh ld15: blocking failure (see json)"
echo "read LD-15 $(stamp)"
PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY "$CODE/ld_read.py" "$R1" "$G/run-ld15" \
  "$G/dh-r9/youthful-giraffe_final.parquet" "$G/dh-ld15/youthful-giraffe_final.parquet" "$G/ld15-read.json"
sha256sum "$VLD" "$P"/* "$G"/dh-ld15/* "$G"/run-ld15/outputs/w2-l2c-s7c.parquet "$G"/run-ld15/work/w2/ords_*.npy \
  "$G"/ld15-read.json
echo "r9_chain15 $TAG done $(stamp)"
