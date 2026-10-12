#!/bin/sh
# Make a run tree for the v73 steps (layout of release/v73_stage.py run_folder), on remote.
# Usage: sh setup_run.sh <src> <export> <run>
#   <src>    snapshot of experiments/exp-191-final-v/code (git archive of one commit)
#   <export> copy of the v73 release export (release/, experiments/exp-169-stack-plan/code, saved vectors)
#   <run>    new folder: iso/ w2/ (copies), chain_ids.py, release -> <export>/release,
#            input-root/data/raw/ (links to the 14 official files and the two final files, checked by SHA-256)
# Official files: $PRC_DATA_DIR (default ${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw, finalcore's data folder).
set -eu
SRC=$1; EXPORT=$2; RUN=$3
DATA=${PRC_DATA_DIR:-${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw}
[ ! -e "$RUN" ] || { echo "refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/input-root/data/raw" "$RUN/work" "$RUN/outputs" "$RUN/logs"
cp -a "$SRC/iso" "$SRC/w2" "$SRC/chain_ids.py" "$RUN/"
ln -s "$EXPORT/release" "$RUN/release"
for f in $(cut -c67- "$SRC/w2/inputs/raw_sha256.txt" | sed 's|input-root/data/raw/||'); do
  ln -s "$(readlink -f "$DATA/$f")" "$RUN/input-root/data/raw/$f"
done
for f in final_ranking.parquet final_submitting.parquet; do
  ln -s "$(readlink -f "$DATA/final/$f")" "$RUN/input-root/data/raw/$f"
done
(cd "$RUN/input-root" && sed 's|input-root/||' "$RUN/w2/inputs/raw_sha256.txt" | sha256sum -c --quiet -)
(cd "$RUN/input-root/data/raw" && printf '%s  %s\n%s  %s\n' \
  4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f final_ranking.parquet \
  b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924 final_submitting.parquet | sha256sum -c --quiet -)
echo "run tree $RUN: 16 official files checked"
