#!/usr/bin/env bash
# exp-194 LD port (SPEC-LD.md; docs/assignments/2026-10-10-ld-port.md): isolated-5's LD step on the R9 default
# final-iii-dh-r9 (4ac66707): ORD-S also applies on the ordinary rows of long-taxi days; the beta fit is unchanged.
# One job on remote x86 under the hold guard, CPU only, outputs in ${PRC_FINAL_ROOT:-/path/to/final}/gainlane/ld-<tag>/:
#  1. trees run-off and run-ld: hard-link copies (rsync --link-dest) of chain f1's R9 run tree without the files that
#     the ORD-S tail writes; code/ords_ld.py goes in as w2/ords_ld.py (every stage before ORD-S is reused);
#  2. in each tree v_tail.sh's commands with w2/ords_ld.py (run-off: no flag; run-ld: --long-days), stages_record.py,
#     v_part.py;
#  3. run-off must give f1's bytes (ords npy files, s7c npz, stages-v.json, part v 614ce19f, V file 1c1ed67b);
#  4. LD: the V-file stages record (v_run.sh form), the checker, r9_dh.py, the dh stages record, the checker;
#  5. reads: r9_read.py (base = f1's R9 tree and dh file, candidate = run-ld and the LD dh file), ld_read.py.
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <code dir>/ld_chain.sh <tag> <code dir> >/dev/null 2>&1 < /dev/null &
# <code dir>: git archive of experiments/exp-194-gain-lane/code at this commit (my scripts).
set -euo pipefail
TAG=$1; CODE=$(readlink -f "$2")
SNAP=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/snap-046b08bea
G1=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/r9-f1
G=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/ld-$TAG
F=${PRC_FINAL_ROOT:-/path/to/final}/finalv
PY=$F/venv312/bin/python
T=$F/run-s7x9
DHPART=${PRC_FINAL_ROOT:-/path/to/final}/gainlane/parts/dh-final1.parquet
C=$SNAP/experiments/exp-191-final-v/code
V_F1=1c1ed67bdd7e54182bafb9a3eb08b48e6e708e981af6bb607c0016ca1f010c4b
PARTV_F1=614ce19f2c8b8feda61c4a7de7e33496e4977f7cd483c45f88943af82326f06e
DH_F1=4ac66707333eab1b32c43db0f412907e05d35e268d6a869065dba3fdc99b8139
ORDS_SHA=d2b5c7cd1a6e52d9794e8833bfcf909cd2b6d852b02cda58cf8038534dc20acf
export PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1
RAW=$PRC_DATA_DIR/final
stamp() { date -u +%FT%TZ; }
die() { echo "STOP: $* $(stamp)"; exit 1; }
memok() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 20 ]; do echo "MemAvailable below 20 GB $(stamp)"; sleep 60; done
  echo "MemAvailable GB $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) $(stamp)"
}
sha() { sha256sum "$1" | cut -d' ' -f1; }
test ! -e "$G"
echo "ld_chain $TAG start $(stamp) pid $$ code $CODE"
R1=$G1/run-r9
[ "$(sha "$G1/root-r9/finalv/youthful-giraffe_final.parquet")" = "$V_F1" ] || die "f1 V file"
[ "$(sha "$G1/root-r9/finalv/parts/v.parquet")" = "$PARTV_F1" ] || die "f1 part v"
[ "$(sha "$G1/dh-r9/youthful-giraffe_final.parquet")" = "$DH_F1" ] || die "f1 dh file"
cmp "$G1/root-r9/finalv/youthful-giraffe_final.parquet" "$R1/outputs/w2-l2c-s7c.parquet" || die "f1 V file is not its run output"
[ "$(sha "$R1/w2/ords.py")" = "$ORDS_SHA" ] || die "f1 tree ords.py"
for s in fold ranking; do
  cmp "$T/work/w2/chain_ids_$s.parquet" "$R1/work/w2/chain_ids_$s.parquet"
  cmp "$T/work/w2/s7_$s.parquet" "$R1/work/w2/s7_$s.parquet"
done
sha256sum "$CODE/ords_ld.py" "$R1/w2/ords.py" "$DHPART"
diff "$R1/w2/ords.py" "$CODE/ords_ld.py" || true
mkdir -p "$G"

# 1. trees (hard links; the tail writes only the excluded paths, so f1's files keep their bytes)
for v in off ld; do
  rsync -a --link-dest="$R1/" --exclude='/work/w2/ords_fold.npy' --exclude='/work/w2/ords_ranking.npy' \
    --exclude='/work/w2/s7c-final.npz' --exclude='/outputs/w2-l2c-s7c.parquet' --exclude='/outputs/stages-v.json' \
    "$R1/" "$G/run-$v/"
  cp "$CODE/ords_ld.py" "$G/run-$v/w2/ords_ld.py"
  for p in work/w2/ords_fold.npy work/w2/ords_ranking.npy work/w2/s7c-final.npz outputs/w2-l2c-s7c.parquet outputs/stages-v.json; do
    test ! -e "$G/run-$v/$p" || die "$p in run-$v"
  done
done
echo "trees $(stamp): $(find "$G/run-ld" -type f | wc -l) files in run-ld"

# 2. the ORD-S tail (v_tail.sh's commands; the s7 member files are f1's hard links, equal to $T's)
tail_run() {  # $1 tree, $2 root, then the ords_ld.py flags
  local R=$1 ROOT=$2; shift 2
  local PARTS=$ROOT/finalv/parts VFILE=$ROOT/finalv/youthful-giraffe_final.parquet
  test ! -e "$ROOT"
  echo "tail $(basename "$R") flags [$*] $(stamp)"; memok
  ( cd "$R"
    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PRC_RANKING_MODE=final
    $PY w2/ords_ld.py --on work/w2/l2c --names s7 --lirf-jul 0.5 "$@" --save-npz work/w2/s7c-final.npz \
      --out outputs/w2-l2c-s7c.parquet
    $PY "$C/stages_record.py" . outputs/stages-v.json > /dev/null
    $PY "$C/v_part.py" . "$PARTS" "$VFILE"
    cp outputs/stages-v.json "$PARTS/v-stages-record.json"
    sha256sum work/w2/ords_fold.npy work/w2/ords_ranking.npy work/w2/s7c-final.npz outputs/w2-l2c-s7c.parquet \
      outputs/stages-v.json "$PARTS/v.parquet" "$VFILE" )
}
tail_run "$G/run-off" "$G/root-off"
tail_run "$G/run-ld" "$G/root-ld" --long-days

# 3. flag off = f1's bytes
for p in work/w2/ords_fold.npy work/w2/ords_ranking.npy work/w2/s7c-final.npz outputs/w2-l2c-s7c.parquet; do
  cmp "$G/run-off/$p" "$R1/$p" || die "flag off: $p differs from f1"
done
stages_equal() {  # stages-v.json holds its run path; compare the rows and the stage list
  $PY -c 'import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["rows"] == b["rows"] and a["stages"] == b["stages"], "stages differ"' "$1" "$2"
}
stages_equal "$G/run-off/outputs/stages-v.json" "$R1/outputs/stages-v.json" || die "flag off: stages-v.json stages differ from f1"
[ "$(sha "$G/root-off/finalv/parts/v.parquet")" = "$PARTV_F1" ] || die "flag off: part v"
[ "$(sha "$G/root-off/finalv/youthful-giraffe_final.parquet")" = "$V_F1" ] || die "flag off: V file"
[ "$(sha "$R1/outputs/w2-l2c-s7c.parquet")" = "$V_F1" ] || die "f1 tree changed"
cmp "$G/run-off/work/w2/s7c-final.npz" "$G/run-ld/work/w2/s7c-final.npz" || die "s7c npz differs between off and ld"
echo "FLAGOFF equal to f1: ords_fold ords_ranking s7c npz V file $V_F1 part v $PARTV_F1 stages-v $(stamp)"

# 4. LD: V-file stages record and checker; dh step, dh stages record and checker
P=$G/root-ld/finalv/parts
VLD=$G/root-ld/finalv/youthful-giraffe_final.parquet
$PY -c 'import json, sys
z, v, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])
names = [s["name"] for s in z["stages"]]
k = names.index("T11b-L")
st = [dict(s, name="Z/" + s["name"]) for s in z["stages"][:k + 1]] + v["stages"]
assert len({s["name"] for s in st}) == len(st)
json.dump({"z_record": sys.argv[1], "v_record": sys.argv[2], "z_stages_after_v64_left_out": names[k + 1:],
           "stages": st}, open(out, "x"), indent=1)' \
  "$G1/parts/r9/z-file-stages-record.json" "$P/v-stages-record.json" "$P/v-file-stages-record.json"
$PY -c 'import json, sys
M = ("2026-01", "2026-02", "2026-06", "2026-07")
print("stage | changed Jan Feb Jun Jul | eligible Jan Feb Jun Jul | board_only or note")
for s in json.load(open(sys.argv[1]))["stages"]:
    c, e = s["changed_by_month"], s.get("eligible_by_month")
    print(s["name"][:44], "|", *[c[m] for m in M], "|", *([e[m] for m in M] if e else ["-"]), "|",
          s.get("board_only") or s.get("note") or "")' "$P/v-file-stages-record.json"
cd "$SNAP"
$PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$VLD" "$SNAP" \
  --stages "$P/v-file-stages-record.json" --out "$P/v-check.json" || echo "check V ld: blocking failure (see json)"
echo "dh step $(stamp)"
$PY -I "$CODE/r9_dh.py" --v-file "$VLD" --part "$DHPART" --raw "$RAW" --out-dir "$G/dh-ld"
$PY -I "$CODE/r9_dh_stages.py" "$P/v-file-stages-record.json" "$VLD" "$G/dh-ld/youthful-giraffe_final.parquet" "$RAW" \
  "$G/dh-ld/dh-file-stages-record.json"
$PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$G/dh-ld/youthful-giraffe_final.parquet" "$SNAP" \
  --stages "$G/dh-ld/dh-file-stages-record.json" --out "$G/dh-ld/dh-check.json" || echo "check dh ld: blocking failure (see json)"

# 5. reads
echo "read $(stamp)"; memok
PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY "$CODE/r9_read.py" "$R1" "$G/run-ld" \
  "$G1/parts/r9" "$G1/parts/r9" "$G1/dh-r9/youthful-giraffe_final.parquet" "$G/dh-ld/youthful-giraffe_final.parquet" \
  "$G/ld-read-r9form.json"
PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY "$CODE/ld_read.py" "$R1" "$G/run-ld" \
  "$G1/dh-r9/youthful-giraffe_final.parquet" "$G/dh-ld/youthful-giraffe_final.parquet" "$G/ld-read.json"
[ "$(sha "$R1/outputs/w2-l2c-s7c.parquet")" = "$V_F1" ] && [ "$(sha "$G1/dh-r9/youthful-giraffe_final.parquet")" = "$DH_F1" ] \
  || die "f1 files changed"
sha256sum "$VLD" "$P"/* "$G"/dh-ld/* "$G"/run-ld/outputs/w2-l2c-s7c.parquet "$G"/run-ld/outputs/stages-v.json \
  "$G"/run-ld/work/w2/ords_*.npy "$G"/ld-read*.json
echo "ld_chain $TAG done $(stamp)"
