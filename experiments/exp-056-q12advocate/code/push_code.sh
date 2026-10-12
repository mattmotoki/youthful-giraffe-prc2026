#!/usr/bin/env bash
# exp-056-q12advocate: create (first call) or re-version the private Kaggle dataset
# that carries the COMMITTED repo code for the label-isolated base refits.
#
#   bash experiments/exp-056-q12advocate/code/push_code.sh "why this version exists"
#
# exp-025's push_code.sh tars the working tree into the shared dataset
# ${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/source-code. This one archives HEAD with `git archive`, so the
# dataset holds exactly one commit and no uncommitted edit of another worker, and
# it uses its own slug, so no other worker's next kernel picks it up.
set -euo pipefail
MSG="${1:?a one-line reason for this version}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
K="$HOME/.local/bin/kaggle"
SLUG="${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/experiment-code"
STAGE="$(mktemp -d)"
cd "$ROOT"
# The same source list as exp-025's push_code.sh (it ran these three producers).
git archive --format=tar --prefix=repo/ HEAD -- \
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
    requirements.txt \
    | (cd "$STAGE" && tar xf -)
find "$STAGE/repo" \( -name '*.parquet' -o -name '*.csv' -o -name '__pycache__' \) -prune -exec rm -rf {} +
git rev-parse HEAD > "$STAGE/repo/GIT_COMMIT.txt"
echo "git archive of HEAD; committed tree only" >> "$STAGE/repo/GIT_COMMIT.txt"
cd "$STAGE" && COPYFILE_DISABLE=1 tar czf prc-repo.tar.gz repo && rm -rf repo
echo "tarball sha256 $(shasum -a 256 prc-repo.tar.gz | cut -d' ' -f1)  bytes $(stat -f %z prc-repo.tar.gz)"
cat > dataset-metadata.json <<JSON
{"title": "prc 2026 exp056 code", "id": "$SLUG", "licenses": [{"name": "other"}]}
JSON
if "$K" datasets status "$SLUG" >/dev/null 2>&1; then
    "$K" datasets version -p . --dir-mode skip -m "$MSG"
else
    "$K" datasets create -p . --dir-mode skip
fi
echo "staged $SLUG from $(git -C "$ROOT" rev-parse --short HEAD)"
