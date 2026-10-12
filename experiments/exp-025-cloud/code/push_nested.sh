#!/usr/bin/env bash
# exp-025-cloud: push one nested-OOF kernel for the clean instrument.
#
#   bash experiments/exp-025-cloud/code/push_nested.sh F1-fresh a july_loo_nx_dec december_nx_jul
#
# Arguments: member, a short group tag (the slug suffix), then the nested folds
# to run in that one session. Put the two folds that share a training set in the
# same call when you can: they cost two fits either way, but one container.
set -euo pipefail
MEMBER="${1:?member: F1-fresh L2-hub8725 V3-allin T-tailv4}"
GROUP="${2:?a short group tag, e.g. a}"
shift 2
FOLDS=("$@")
[ ${#FOLDS[@]} -gt 0 ] || { echo "name at least one nested fold" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/experiments/exp-025-cloud/kernels/nested/prc-nested.py"
LOW="$(echo "$MEMBER" | tr 'A-Z' 'a-z')"
SLUG="prc-nested-$LOW-$GROUP"
DIR="$ROOT/experiments/exp-025-cloud/kernels/nested-$MEMBER-$GROUP"

case "$MEMBER" in
  F1-fresh) GPU=true ;;
  L2-hub8725|V3-allin|T-tailv4) GPU=false ;;
  *) echo "unknown member $MEMBER" >&2; exit 1 ;;
esac
JFOLDS=$(printf '"%s", ' "${FOLDS[@]}"); JFOLDS="[${JFOLDS%, }]"

mkdir -p "$DIR"
python3 - "$SRC" "$DIR/prc-nested.py" "$MEMBER" "$JFOLDS" <<'PY'
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
  "title": "prc nested $LOW $GROUP",
  "code_file": "prc-nested.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": $GPU,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": [],
  "dataset_sources": ["${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/source-code", "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/${PRC_KAGGLE_DATA_SLUG:?set PRC_KAGGLE_DATA_SLUG}"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
JSON
python3 -c "import ast; ast.parse(open('$DIR/prc-nested.py').read())"
echo "pushing $MEMBER ${FOLDS[*]} as ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG (gpu=$GPU)"
cd "$DIR" && "$HOME/.local/bin/kaggle" kernels push -p .
