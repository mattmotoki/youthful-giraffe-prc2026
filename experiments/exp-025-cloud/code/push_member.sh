#!/usr/bin/env bash
# exp-025-cloud: push one stack-4 member kernel.
#
#   bash experiments/exp-025-cloud/code/push_member.sh L2-hub  [<data dataset slug>]
#
# Writes kernels/member-<slug>/ from kernels/member/prc-member.py with CFG's
# member swapped and the GPU flag set from the MEMBERS table, then pushes. The
# generated directory is committed, so the record holds exactly what ran.
set -euo pipefail
MEMBER="${1:?member name, one of L2-hub T-tailv2 D1-winter C2-cat N1-nmtail}"
DATASET="${2:-${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/synthetic-data}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/experiments/exp-025-cloud/kernels/member/prc-member.py"
SLUG="prc-member-$(echo "$MEMBER" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-' | sed 's/-\{1,\}/-/g;s/-$//')"
DIR="$ROOT/experiments/exp-025-cloud/kernels/member-$MEMBER"

case "$MEMBER" in
  C2-cat) GPU=true ;;
  L2-hub|T-tailv2|D1-winter|N1-nmtail) GPU=false ;;
  *) echo "unknown member $MEMBER" >&2; exit 1 ;;
esac

mkdir -p "$DIR"
sed "s/^CFG = {\"member\": \"[^\"]*\"/CFG = {\"member\": \"$MEMBER\"/" "$SRC" > "$DIR/prc-member.py"
grep -q "\"member\": \"$MEMBER\"" "$DIR/prc-member.py" || { echo "CFG rewrite failed" >&2; exit 1; }
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG",
  "title": "prc member $(echo "$MEMBER" | tr 'A-Z' 'a-z')",
  "code_file": "prc-member.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": $GPU,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": [],
  "dataset_sources": ["${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/source-code", "$DATASET"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
JSON
python3 -c "import ast,sys; ast.parse(open('$DIR/prc-member.py').read())"
echo "pushing $MEMBER as ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG (gpu=$GPU, data=$DATASET)"
cd "$DIR" && "$HOME/.local/bin/kaggle" kernels push -p .
