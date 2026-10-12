#!/usr/bin/env bash
# exp-043-pilot: generate and push one surface-filter kernel from the single source.
#
#   bash experiments/exp-043-pilot/code/push_kernel.sh probe '["2025-07-07"]' 200000000
#   bash experiments/exp-043-pilot/code/push_kernel.sh gate  '["2025-07-07","2025-07-08"]' 0
#
# The generated directory is committed, so the record holds exactly what ran.
# `kernels/surface/prc-surface.py` is the only place the filter lives; this script
# rewrites CFG's day list and probe size and nothing else.
set -euo pipefail
NAME="${1:?kernel name, e.g. probe}"
DAYS="${2:?json list of days, e.g. '[\"2025-07-07\"]'}"
PROBE="${3:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/experiments/exp-043-pilot/kernels/surface/prc-surface.py"
DIR="$ROOT/experiments/exp-043-pilot/kernels/surface-$NAME"
SLUG="prc-surface-$NAME"

mkdir -p "$DIR"
python3 - "$SRC" "$DIR/prc-surface.py" "$DAYS" "$PROBE" <<'PY'
import json, re, sys
src, dst, days, probe = sys.argv[1:5]
json.loads(days); int(probe)
s = open(src).read()
s, n1 = re.subn(r'"days": \[[^\]]*\],', f'"days": {days},', s, count=1)
s, n2 = re.subn(r'"probe_bytes": \d+,', f'"probe_bytes": {probe},', s, count=1)
assert n1 == 1 and n2 == 1, (n1, n2)
open(dst, "w").write(s)
PY
python3 -c "import ast; ast.parse(open('$DIR/prc-surface.py').read())"
grep -qF "\"days\": $DAYS," "$DIR/prc-surface.py" || { echo "day rewrite failed" >&2; exit 1; }
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG",
  "title": "prc surface $NAME",
  "code_file": "prc-surface.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
JSON
echo "pushing $NAME as ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG  days=$DAYS probe=$PROBE"
cd "$DIR" && "$HOME/.local/bin/kaggle" kernels push -p .
