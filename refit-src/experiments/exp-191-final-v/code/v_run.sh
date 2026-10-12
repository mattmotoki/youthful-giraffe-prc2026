#!/bin/sh
# exp-191 final V build on finalplan's final export copy (assignment item 5): the run tree (setup_run.sh), c5, w2
# base and L2C (final_run.sh), ORD-S with the s7c member, part v and the V file (v_tail.sh), then the checks: NOAC
# and ROLL against finallirf's exp-187 output (noac_roll_check.py), the fold G split against the ranking tree
# (fold_shift_read.py), check_final_candidate.py --stages (finalplan's Z stages through T11b-L, prefixed "Z/", then
# the V stages, printed per month with the eligible rows), L2C's eligible rows against finalclock's head part
# (l2c_clock_check.py), and the January/July parity of v against the saved v73 values. The MANIFEST `command` of part v.
#
# Run on remote x86 through the hold guard, from a repo snapshot <SRC> (git archive of prc,
# experiments/exp-186-final-core/code and experiments/exp-191-final-v/code), with PRC_DATA_DIR:
#   sh <SRC>/experiments/exp-191-final-v/code/v_run.sh <SRC> <python> <export> <run> <s7 run> <finallirf out> \
#     <ranking run> <Z stages record> <finalclock head part> <its SHA-256>
# Rebuild tolerance of part v: deterministic SHA-256 on the same machine and threads (x86, OMP 2 / BLAS 2, NT=8).
# Port option (PORT-S7X9.md, desk 10-10): PRC_V_L2C_DIFF=<isolated5-A/l2c.diff> and PRC_V_L2C_S7=<s7 npz, keys fold and
# ranking> run candidate A's L2C (the diff on the release w2/l2c.py, with --s7 <npz>); both variables or neither.
set -eu
SRC=$(readlink -f "$1"); PY=$2; EXPORT=$3; RUN=$4; S7=$5; LIRF=$6; RRUN=$7; ZSTAGES=$8; HEAD=$9; HEADSHA=${10}
ROOT=${PRC_FINAL_ROOT:-${PRC_FINAL_ROOT:-/path/to/final}}
PARTS=$ROOT/finalv/parts
VFILE=$ROOT/finalv/youthful-giraffe_final.parquet
REFS=${PRC_V73_REFS:-$ROOT/finalcore/refs/v73-parts.parquet}
C=$SRC/experiments/exp-191-final-v/code
: "$PRC_DATA_DIR"
stamp() { date -u +%FT%TZ; }
test ! -e "$RUN" && test ! -e "$PARTS/v.parquet" && test ! -e "$VFILE" && test -s "$ZSTAGES" && test -s "$HEAD"
test -s "$LIRF/record.json" && test -s "$LIRF/lirf-notype.parquet"
echo "setup $(stamp)"
sh "$C/setup_run.sh" "$C" "$EXPORT" "$RUN"
cd "$RUN"
unset L2C_S7
if [ -n "${PRC_V_L2C_DIFF:-}${PRC_V_L2C_S7:-}" ]; then
  test -s "$PRC_V_L2C_DIFF" && test -s "$PRC_V_L2C_S7"
  echo "candidate A L2C $(stamp)"
  echo "c9cd2869063d5534e81b37ed57675994ee71a2f57eed3a22ca7b78dc5f2358f9  w2/l2c.py" | sha256sum -c
  patch -p1 --no-backup-if-mismatch < "$PRC_V_L2C_DIFF"
  echo "c9931e0a6e2192b2391df6d4b4a74b67ab2592934ce5eb420dc177f59e82c9ae  w2/l2c.py" | sha256sum -c
  $PY -c 'import sys, numpy as np
z = np.load(sys.argv[1])
assert sorted(z.files) == ["fold", "ranking"], z.files
assert z["fold"].dtype == np.float32 and z["fold"].shape == (344419,), z["fold"].shape
assert z["ranking"].dtype == np.float32 and z["ranking"].shape == (670790,), z["ranking"].shape
print("s7 npz", sys.argv[1], "fold", z["fold"].shape, "ranking", z["ranking"].shape)' "$PRC_V_L2C_S7"
  sha256sum "$PRC_V_L2C_DIFF" "$PRC_V_L2C_S7"
  L2C_S7=$PRC_V_L2C_S7; export L2C_S7
fi
echo "final run $(stamp)"
sh "$C/final_run.sh" "$PY" "$C"
echo "v tail $(stamp)"
sh "$C/v_tail.sh" "$PY" "$C" "$S7" "$PARTS" "$VFILE"
echo "checks $(stamp)"
PRC_RANKING_MODE=final $PY "$C/noac_roll_check.py" . "$LIRF" outputs/noac-roll-check.json > /dev/null
PRC_RANKING_MODE=final $PY "$C/fold_shift_read.py" "$RRUN" . outputs/fold-shift-read.json > /dev/null
PRC_RANKING_MODE=final $PY "$C/l2c_clock_check.py" . "$HEAD" "$HEADSHA" outputs/l2c-clock-check.json
# V starts from the pack's v64 (zabd + T11b-L, byte-checked by dump_chain): Z's stages through T11b-L, then the V
# stages; Z's later stages (its NOAC and ROLL) are the c5 post steps, which the V stages hold once
$PY -c 'import json, sys
z, v, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])
names = [s["name"] for s in z["stages"]]
k = names.index("T11b-L")
st = [dict(s, name="Z/" + s["name"]) for s in z["stages"][:k + 1]] + v["stages"]
assert len({s["name"] for s in st}) == len(st)
json.dump({"z_record": sys.argv[1], "v_record": sys.argv[2], "z_stages_after_v64_left_out": names[k + 1:],
           "stages": st}, open(out, "x"), indent=1)' \
  "$ZSTAGES" "$PARTS/v-stages-record.json" "$PARTS/v-file-stages-record.json"
$PY -c 'import json, sys
M = ("2026-01", "2026-02", "2026-06", "2026-07")
print("stage | changed Jan Feb Jun Jul | eligible Jan Feb Jun Jul | board_only or note")
for s in json.load(open(sys.argv[1]))["stages"]:
    c, e = s["changed_by_month"], s.get("eligible_by_month")
    print(s["name"][:44], "|", *[c[m] for m in M], "|", *([e[m] for m in M] if e else ["-"]), "|",
          s.get("board_only") or s.get("note") or "")' "$PARTS/v-file-stages-record.json"
cd "$SRC"
$PY -I experiments/exp-186-final-core/code/check_final_candidate.py "$VFILE" "$SRC" \
  --stages "$PARTS/v-file-stages-record.json" --out "$PARTS/v-check.json" || echo "check: blocking failure (see json)"
$PY -I experiments/exp-186-final-core/code/parity.py --refs "$REFS" --new "$PARTS/v.parquet" --pair v=saved:v73-final \
  --kind refit --out "$PARTS/v-parity.json"
echo "hashes $(stamp)"
sha256sum "$EXPORT/release/ZABD-PACK.json" "$ZSTAGES" "$PARTS"/v.parquet "$PARTS"/v-*.json "$VFILE" \
  "$RUN/outputs/noac-roll-check.json" "$RUN/outputs/fold-shift-read.json" "$RUN/outputs/l2c-clock-check.json"
echo "done $(stamp)"
