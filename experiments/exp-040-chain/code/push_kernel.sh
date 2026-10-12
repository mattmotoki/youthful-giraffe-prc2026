#!/usr/bin/env bash
# exp-040-chain: generate and push one arm/fold kernel from the single source.
#
#   bash experiments/exp-040-chain/code/push_kernel.sh july_loo
#   bash experiments/exp-040-chain/code/push_kernel.sh ranking
#
# The generated directory is committed, so the record holds exactly what ran.
# `kernels/chain/prc-chain.py` is the only place the producer command lives; this
# script rewrites CFG's fold list and the submit flag and nothing else.
set -euo pipefail
WHAT="${1:?july_loo | december | january | ranking}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/experiments/exp-040-chain/kernels/chain/prc-chain.py"
DIR="$ROOT/experiments/exp-040-chain/kernels/chain-$WHAT"
SLUG="prc-chain-$(echo "$WHAT" | tr '_' '-')"

if [ "$WHAT" = "ranking" ]; then
  FOLDS='[]'; SUBMIT=True
else
  FOLDS="[\"$WHAT\"]"; SUBMIT=False
fi

mkdir -p "$DIR"
python3 - "$SRC" "$DIR/prc-chain.py" "$FOLDS" "$SUBMIT" "$WHAT" <<'PY'
import re, sys
src, dst, folds, submit, tag = sys.argv[1:6]
s = open(src).read()
s = re.sub(r'"folds": \[[^\]]*\],', f'"folds": {folds},', s, count=1)
s = re.sub(r'"submit": \w+,', f'"submit": {submit},', s, count=1)
s = re.sub(r'"tag": "[^"]*",', f'"tag": "chain-{tag}",', s, count=1)
open(dst, "w").write(s)
PY
grep -q "\"tag\": \"chain-$WHAT\"" "$DIR/prc-chain.py" || { echo "CFG rewrite failed" >&2; exit 1; }
python3 -c "import ast; ast.parse(open('$DIR/prc-chain.py').read())"
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG",
  "title": "prc chain $WHAT",
  "code_file": "prc-chain.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": [],
  "dataset_sources": ["${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/source-code", "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/${PRC_KAGGLE_DATA_SLUG:?set PRC_KAGGLE_DATA_SLUG}"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
JSON
echo "pushing $WHAT as ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG"
cd "$DIR" && "$HOME/.local/bin/kaggle" kernels push -p .
