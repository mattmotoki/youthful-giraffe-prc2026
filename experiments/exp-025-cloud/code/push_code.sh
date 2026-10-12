#!/usr/bin/env bash
# exp-025-cloud: re-version the private Kaggle dataset that carries the repo code.
#
#   bash experiments/exp-025-cloud/code/push_code.sh "why this version exists"
#
# Kaggle extracts the tarball server side, so the dataset serves `repo/...`.
# COPYFILE_DISABLE=1 keeps macOS from writing AppleDouble `._` files into it.
set -euo pipefail
MSG="${1:?a one-line reason for this version}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
K="$HOME/.local/bin/kaggle"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/repo"
cd "$ROOT"
# Only source and small config. The member directories also hold published
# parquet (exp-012-blend2 alone is 150 MB) and a kernel needs none of it.
COPYFILE_DISABLE=1 tar cf - \
    --exclude='*.parquet' --exclude='*.csv' --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    prc \
    experiments/exp-018-fresh/code \
    experiments/exp-018b-climb/code \
    experiments/exp-008-diversity \
    experiments/exp-012-blend2 \
    experiments/exp-007-winter/code \
    experiments/exp-003-anatomy/code \
    experiments/exp-010-nmtail/code \
    experiments/exp-009-noac2 \
    experiments/exp-001-retro-v1 \
    experiments/exp-016-stack \
    experiments/exp-037-tail/code \
    experiments/exp-040-chain/code \
    scripts/build_candidate.py docs/REPRODUCE.md docs/input-rule.md requirements.txt \
    | (cd "$STAGE/repo" && tar xf -)
git rev-parse HEAD > "$STAGE/repo/GIT_COMMIT.txt"
git status --porcelain >> "$STAGE/repo/GIT_COMMIT.txt"
cd "$STAGE" && COPYFILE_DISABLE=1 tar czf prc-repo.tar.gz repo && rm -rf repo
cat > dataset-metadata.json <<JSON
{"title": "prc 2026 repo code", "id": "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/source-code", "licenses": [{"name": "other"}]}
JSON
"$K" datasets version -p . --dir-mode skip -m "$MSG"
echo "staged from $(git -C "$ROOT" rev-parse --short HEAD)"
