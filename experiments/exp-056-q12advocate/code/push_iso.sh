#!/usr/bin/env bash
# exp-056-q12advocate: push one private kernel of label-isolated base refits.
#
#   bash experiments/exp-056-q12advocate/code/push_iso.sh F1-fresh jul iso_jul_v iso_jul_s iso_jul_i iso_jul_x
#
# Arguments: member, a group tag (the slug suffix), then the prc.data iso folds to
# run in that one session, in order. The kernel writes predictions only.
set -euo pipefail
MEMBER="${1:?member: F1-fresh L2-hub8725 T-tailv4}"
GROUP="${2:?a group tag, e.g. jul}"
shift 2
FOLDS=("$@")
[ ${#FOLDS[@]} -gt 0 ] || { echo "name at least one iso fold" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-056-q12advocate"
SRC="$EXP/kernels/iso/prc-iso.py"
LOW="$(echo "$MEMBER" | tr 'A-Z' 'a-z')"
SLUG="prc-iso-$LOW-$GROUP"
DIR="$EXP/kernels/iso-$MEMBER-$GROUP"
case "$MEMBER" in
  F1-fresh) GPU=true ;;
  L2-hub8725|T-tailv4) GPU=false ;;
  *) echo "unknown member $MEMBER" >&2; exit 1 ;;
esac
for f in "${FOLDS[@]}"; do [[ "$f" == iso_* ]] || { echo "not an iso fold: $f" >&2; exit 2; }; done
JFOLDS=$(printf '"%s", ' "${FOLDS[@]}"); JFOLDS="[${JFOLDS%, }]"
[ ! -e "$DIR" ] || { echo "$DIR exists; a pushed kernel directory is write-once" >&2; exit 3; }
mkdir -p "$DIR"
python3 - "$SRC" "$DIR/prc-iso.py" "$MEMBER" "$JFOLDS" <<'PY'
import sys, re
src, dst, member, folds = sys.argv[1:5]
s = open(src).read()
s = re.sub(r'"member": "[^"]*",', f'"member": "{member}",', s, count=1)
s = re.sub(r'"folds": \[[^\]]*\],', f'"folds": {folds},', s, count=1)
assert f'"member": "{member}"' in s and f'"folds": {folds}' in s, "CFG rewrite failed"
open(dst, "w").write(s)
PY
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG",
  "title": "prc iso $LOW $GROUP",
  "code_file": "prc-iso.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": $GPU,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": [],
  "dataset_sources": ["${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/experiment-code", "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/${PRC_KAGGLE_DATA_SLUG:?set PRC_KAGGLE_DATA_SLUG}"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
JSON
python3 -c "import ast; ast.parse(open('$DIR/prc-iso.py').read())"
echo "pushing $MEMBER ${FOLDS[*]} as ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG (gpu=$GPU)"
cd "$DIR" && "$HOME/.local/bin/kaggle" kernels push -p .
