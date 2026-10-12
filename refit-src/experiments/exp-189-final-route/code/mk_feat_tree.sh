#!/usr/bin/env bash
# exp-189: a run-only tree of main at a commit for `features_2026.py` on the Mac (the host of the exp-045
# summaries, where finaldata lands the February/June harvest).
#   bash experiments/exp-189-final-route/code/mk_feat_tree.sh <commit> <dir> <inputs dir>
# Code: `git archive <commit>` of every tracked file under 2 MB in prc/, experiments/exp-0*/,
# experiments/exp-146-validation/frozen/ and two docs; tracked files of 2 MB and more are links to this
# checkout, and the run stops when the checkout's copy is not the commit's blob. Inputs, read only:
#   artifacts/exp-045            link to this checkout (summaries of every month)
#   artifacts/exp-046/summaries  link to this checkout
#   artifacts/exp-046/class      <inputs dir>/class          (v25 class export, ranking mode)
#   artifacts/exp-018-fresh/cache <inputs dir>/e018          (2026 shipped tables)
#   artifacts/exp-047/block       <inputs dir>/e047          (2026 queue blocks, ranking mode)
#   artifacts/exp-047/block-final <inputs dir>/e047-final    (final mode)
#   data, submissions            links to this checkout
set -euo pipefail
C=$1; DIR=$2; IN=$3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -e "$DIR" ] && { echo "$DIR exists"; exit 1; }
mkdir -p "$DIR"
git -C "$ROOT" ls-tree -r -l "$C" prc docs/EXTERNAL-DATA.md docs/input-rule.md experiments/ \
  | awk '$4 != "-" && ($5 ~ /^prc\// || $5 ~ /^docs\// || $5 ~ /^experiments\/exp-0/ || $5 ~ /^experiments\/exp-146-validation\/frozen\//) {print $4 "\t" $5}' \
  > "$DIR/.files"
awk -F'\t' '$1 < 2000000 {print $2}' "$DIR/.files" > "$DIR/.small-files"
awk -F'\t' '$1 >= 2000000 {print $2}' "$DIR/.files" > "$DIR/.big-files"
xargs -n 400 bash -c 'git -C "$0" archive "$1" "${@:3}" | tar -x -C "$2"' "$ROOT" "$C" "$DIR" < "$DIR/.small-files"
# a big file is linked only when the checkout holds the commit's blob
while read -r f; do
  want=$(git -C "$ROOT" rev-parse "$C:$f")
  have=$(git -C "$ROOT" hash-object "$ROOT/$f")
  [ "$want" = "$have" ] || { echo "checkout copy of $f is not the blob of $C"; exit 1; }
  mkdir -p "$DIR/$(dirname "$f")"; ln -s "$ROOT/$f" "$DIR/$f"
done < "$DIR/.big-files"
mkdir -p "$DIR/artifacts/exp-046" "$DIR/artifacts/exp-047" "$DIR/artifacts/exp-018-fresh"
ln -s "$ROOT/artifacts/exp-045" "$DIR/artifacts/exp-045"
ln -s "$ROOT/artifacts/exp-046/summaries" "$DIR/artifacts/exp-046/summaries"
ln -s "$IN/class" "$DIR/artifacts/exp-046/class"
ln -s "$IN/e018" "$DIR/artifacts/exp-018-fresh/cache"
ln -s "$IN/e047" "$DIR/artifacts/exp-047/block"
ln -s "$IN/e047-final" "$DIR/artifacts/exp-047/block-final"
ln -s "$ROOT/data" "$DIR/data"
ln -s "$ROOT/submissions" "$DIR/submissions"
git -C "$ROOT" rev-parse "$C" > "$DIR/COMMIT"
echo "tree $(cat "$DIR/COMMIT") in $DIR: $(wc -l < "$DIR/.small-files") files, $(wc -l < "$DIR/.big-files") links"
