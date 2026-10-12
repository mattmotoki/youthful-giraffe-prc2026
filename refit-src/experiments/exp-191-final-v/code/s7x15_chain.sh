#!/bin/bash
# s7 seeds 10-15 with the recipe of seeds 1-3, then the 15-seed port. ONE job under
# the hold guard with PRC_GPU=1: the guard holds the GPU 1 lock for the whole job and stops the job in each ARC hold.
# Launch (detached, its own statement):
#   PRC_GPU=1 setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> \
#     bash <snapshot>/experiments/exp-191-final-v/code/s7x15_chain.sh <snapshot> >/dev/null 2>&1 </dev/null &
# The job can start again after a stop: each output is written once, and a finished step is kept and skipped.
# 1. Tree run-s7x15: hard links of run-s7x9 (inputs and seeds 1-9; its 9-seed member files are left out), checked
#    against s7x15-tree.sha256 (16 inputs, 18 seed files) and fit.py d6c9458f.
# 2. Per seed 10..15, held then full: MemAvailable >= fit peak + 10 GB (peak 12 GB, the estimate of release73's s7
#    refit, or the largest peak measured here if higher), then the command of seeds 1-3 (s7_fit_queue.sh; w2/rebuild.sh
#    line 11 flags) with only --seed and --out changed, under /usr/bin/time -v. The fit writes a tmp- file, which is
#    renamed after exit 0. After each pair: s7_seed_check.py (dtype, shape, NaN pattern, minimum, distances).
# 3. After seed 15: the 15-seed member (s7n_finish.sh, N=15); run (iv), the 15-seed s7 with the release L2C; run (v),
#    the 15-seed s7 with candidate A (l2c.diff, --s7 the npz of run iv); each V run at MemAvailable >= 18 GB. Reads
#    (port_read2.py): (iv) against V, run (ii) and run (iii); (v) against V and run (iii).
set -u
S=$(readlink -f "$1"); TAG=$(basename "$S"); TAG=${TAG#repo-}
F=${PRC_FINAL_ROOT:-/path/to/final}/finalv; O=$F/run-s7x9; T=$F/run-s7x15; P=$F/port15
PYT=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/.venv/bin/python   # the torch python of seeds 1-3
PY=$F/venv312/bin/python
C=$S/experiments/exp-191-final-v/code
LOGS=${PRC_LOG_ROOT:-/path/to/logs}
FIT_SHA=d6c9458f3c1febbd78ddcbe85c81a02a0884622915ca4f7582281b4ac9e3a40f
export PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw PYTHONDONTWRITEBYTECODE=1
stamp() { date -u +%FT%TZ; }
say() { echo "$(stamp) finalv: $*"; }
die() { say "STOP: $*"; exit 1; }
memgate() {  # $1: gate in GB
  local n=0 m
  while m=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo); [ "$m" -lt "$1" ]; do
    [ $n = 1 ] || say "MemAvailable $m GB, under the gate of $1 GB; waiting"; n=1; sleep 60
  done
  say "MemAvailable $m GB (gate $1 GB)"
}
say "chain start, pid $$, group $(ps -o pgid= $$ | tr -d ' '), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, snapshot $S"
[ "${CUDA_VISIBLE_DEVICES:-}" = 1 ] || die "CUDA_VISIBLE_DEVICES is not 1"
mkdir -p "$P"

# 1. the tree
if [ ! -e "$T" ]; then
  say "tree: hard links of $O"
  rsync -a --link-dest="$O" --exclude 'work/w2/s7_*.parquet' "$O/" "$T/" || die "rsync"
fi
[ "$(stat -c %i "$O/work/seq7/nbr.npy")" = "$(stat -c %i "$T/work/seq7/nbr.npy")" ] || die "nbr.npy is no hard link"
(cd "$T/work/seq7" && sha256sum --quiet -c "$C/s7x15-tree.sha256") || die "tree hashes"
(cd "$T" && echo "$FIT_SHA  w2/seq/fit.py" | sha256sum -c) || die "fit.py hash"
"$PYT" -c 'import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())'
say "tree ok: 16 inputs and 18 seed files equal s7x15-tree.sha256; $(ls "$T/work/seq7" | grep -c '^s7-') seed files"

# 2. the fits
PEAK=12
for seed in 10 11 12 13 14 15; do
  for mode in held full; do
    out=work/seq7/s7-$mode-$seed.npy; tmp=work/seq7/tmp-s7-$mode-$seed.npy
    if [ -s "$T/$out" ]; then say "fit $mode $seed: kept ($out exists)"; continue; fi
    if [ -e "$T/$tmp" ]; then   # the output of a stopped fit: kept aside, never read
      mv "$T/$tmp" "$T/work/seq7/stopped-$(date -u +%Y%m%dT%H%M%SZ)-s7-$mode-$seed.npy" || die "move $tmp"
    fi
    memgate $((PEAK + 10))
    say "fit $mode $seed start"
    (cd "$T" && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      /usr/bin/time -v -o "$P/time-fit-$mode-$seed.txt" "$PYT" w2/seq/fit.py --data work/seq7 --d 192 --layers 4 \
      --epochs 16 --mode "$mode" --seed "$seed" --amp --out "$tmp")
    rc=$?
    kb=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$P/time-fit-$mode-$seed.txt" 2>/dev/null)
    say "fit $mode $seed exit $rc, peak RSS ${kb:-?} kB, $(grep -E 'Elapsed' "$P/time-fit-$mode-$seed.txt" 2>/dev/null)"
    [ "$rc" = 0 ] && [ -s "$T/$tmp" ] || die "fit $mode $seed"
    mv "$T/$tmp" "$T/$out" || die "rename $tmp"
    sha256sum "$T/$out"
    if [ -n "$kb" ]; then gb=$(( (kb + 1048575) / 1048576 )); [ "$gb" -gt "$PEAK" ] && PEAK=$gb; fi
  done
  chk=$P/s7-seed-check-$seed.json
  if [ ! -e "$chk" ]; then
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" "$C/s7_seed_check.py" "$T" "$chk" "$seed" || die "seed check $seed"
  fi
  say "seed $seed pair done ($(sha256sum "$chk" | cut -c1-16))"
done
say "fits done: $(ls "$T/work/seq7" | grep -c '^s7-') seed files"

# 3. the 15-seed port
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
REFS=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet
DH=${PRC_FINAL_ROOT:-/path/to/final}/release73/rebuild/2140d645167a/release73/files/final-dh/youthful-giraffe_final.parquet
VF=$F/youthful-giraffe_final.parquet
V="${PRC_FINAL_ROOT:-/path/to/final}/finalplan/export-final-2"
TAIL="$F/refs-finallirf-final-a444 $F/run-ranking-cf74f4261 ${PRC_FINAL_ROOT:-/path/to/final}/finalplan/parts/final-2/z-file-stages-record.json ${PRC_FINAL_ROOT:-/path/to/final}/finalclock/parts/head.parquet a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6"
RV=$F/run-final2-66b11c579; RII=$F/run-port9-a0c484bab; RIII=$F/run-portA9-a0c484bab
SV=121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a
SII=280db2058f10a5cc3d44acdcf20aed80aae143cfadf2c9c299ddf7212c4f7afc
SIII=6de02372de5d1aea062300a88af7af51606ac2d6aee645c384b09398c9c57378
R4=$F/run-port15-$TAG; R5=$F/run-portA15-$TAG
if [ -s "$P/parts-s7x15/s7c-parity.json" ]; then say "member: kept"
elif [ -e "$P/parts-s7x15" ]; then die "member stopped before its parity read; $P/parts-s7x15 needs a look"
else
  say "member"
  (cd "$S" && sh experiments/exp-191-final-v/code/s7n_finish.sh "$PY" "$T" "$P/parts-s7x15" "$REFS" 15) || die "member"
fi
if [ -s "$P/root-15/finalv/youthful-giraffe_final.parquet" ]; then say "run (iv): kept"
elif [ -e "$R4" ]; then die "run (iv) stopped before its V file; $R4 needs a look"
else
  say "run (iv): the 15-seed s7 with the release L2C"; memgate 18; mkdir -p "$P/root-15"
  PRC_FINAL_ROOT=$P/root-15 PRC_V73_REFS=$REFS /usr/bin/time -v sh "$C/v_run.sh" "$S" "$PY" "$V" "$R4" "$T" $TAIL || die "run (iv)"
fi
rd() {  # $1 out name, $2 base tree, $3 base SHA, $4 port tree, [$5 npz]
  [ -e "$P/$1" ] && { say "read $1: kept"; return 0; }
  say "read $1"
  PRC_RANKING_MODE=final "$PY" "$C/port_read2.py" "$2" "$3" "$4" "$VF" "$DH" "$P/$1" ${5:+"$5"} || die "read $1"
}
rd read-15-vs-V.json "$RV" "$SV" "$R4"
rd read-15-vs-ii.json "$RII" "$SII" "$R4"
rd read-15-vs-iii.json "$RIII" "$SIII" "$R4"
NPZ=$R4/work/w2/s7c-final.npz
if [ -s "$P/root-A15/finalv/youthful-giraffe_final.parquet" ]; then say "run (v): kept"
elif [ -e "$R5" ]; then die "run (v) stopped before its V file; $R5 needs a look"
else
  say "run (v): the 15-seed s7 with candidate A"; memgate 18; mkdir -p "$P/root-A15"
  PRC_FINAL_ROOT=$P/root-A15 PRC_V73_REFS=$REFS PRC_V_L2C_DIFF=$S/experiments/exp-191-final-v/isolated5-A/l2c.diff \
    PRC_V_L2C_S7=$NPZ /usr/bin/time -v sh "$C/v_run.sh" "$S" "$PY" "$V" "$R5" "$T" $TAIL || die "run (v)"
fi
rd read-A15-vs-V.json "$RV" "$SV" "$R5" "$NPZ"
rd read-A15-vs-iii.json "$RIII" "$SIII" "$R5" "$NPZ"
sha256sum "$T"/work/seq7/s7-*-1[0-5].npy "$P"/s7-seed-check-*.json "$P"/parts-s7x15/* "$P"/read-*.json \
  "$P"/root-15/finalv/youthful-giraffe_final.parquet "$P"/root-15/finalv/parts/v.parquet \
  "$P"/root-A15/finalv/youthful-giraffe_final.parquet "$P"/root-A15/finalv/parts/v.parquet
say "chain done"
