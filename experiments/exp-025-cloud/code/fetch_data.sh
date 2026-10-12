#!/usr/bin/env bash
# exp-025-cloud: pull the competition data and our uploaded submissions once the
# S3 key lands, verify what the record can verify, and stop on anything unexpected.
#
#   bash experiments/exp-025-cloud/code/fetch_data.sh
#
# Needs caller-supplied AWS_PROFILE, PRC_DATA_BUCKET, and PRC_SUBMISSION_BUCKET.
# Nothing here writes
# outside data/raw and submissions/uploaded, and nothing here uploads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AWS="${HOME}/.local/bin/aws"
EP="--endpoint-url https://s3.opensky-network.org"
: "${AWS_PROFILE:?Set AWS_PROFILE to the credential profile for the data service}"
: "${PRC_DATA_BUCKET:?Set PRC_DATA_BUCKET to the private data bucket name}"
: "${PRC_SUBMISSION_BUCKET:?Set PRC_SUBMISSION_BUCKET to the private submission bucket name}"
PROFILE=(--profile "$AWS_PROFILE")

# docs/input-rule.md and prc.data.LIVE_RANKING_BYTES
RANKING_SHA_PREFIX="9db75e3a"
RANKING_BYTES=43564605
SUBMITTING_BYTES=1677097

if ! "$AWS" configure list-profiles 2>/dev/null | grep -Fqx "$AWS_PROFILE"; then
  echo "the requested AWS_PROFILE is not in ~/.aws/credentials; the key has not landed" >&2
  exit 2
fi

mkdir -p "$ROOT/data/raw" "$ROOT/submissions/uploaded"

echo "== datasets =="
"$AWS" "${PROFILE[@]}" $EP s3 sync "s3://${PRC_DATA_BUCKET}/" "$ROOT/data/raw/"

echo
echo "== our uploaded submissions (desk, 2026-09-07) =="
# v1 = exp-016 stack-4, sha 48472e44..., board 265.7612, logged in docs/SUBMISSIONS.md.
# v2 = the unlogged upload of 2026-09-06 20:31 UTC, board 265.9631; its hash is
# what says whether it is blend-5 (suspected 5d6a9961...) or something else.
for v in v1 v2; do
  "$AWS" "${PROFILE[@]}" $EP s3 cp \
    "s3://${PRC_SUBMISSION_BUCKET:?set PRC_SUBMISSION_BUCKET}/youthful-giraffe_${v}.parquet" \
    "$ROOT/submissions/uploaded/youthful-giraffe_${v}.parquet" || \
    echo "  (youthful-giraffe_${v}.parquet not fetched)" >&2
done

echo
echo "== hashes =="
for f in "$ROOT/data/raw/ranking.parquet" "$ROOT/data/raw/submitting.parquet" \
         "$ROOT/submissions/uploaded/"*.parquet; do
  [ -f "$f" ] || continue
  printf '%-70s %12d  %s\n' "${f#$ROOT/}" "$(wc -c < "$f")" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
done

echo
echo "== checks =="
rk="$ROOT/data/raw/ranking.parquet"
sb="$ROOT/data/raw/submitting.parquet"
fail=0
if [ -f "$rk" ]; then
  b=$(wc -c < "$rk"); s=$(shasum -a 256 "$rk" | cut -c1-8)
  [ "$b" = "$RANKING_BYTES" ] || { echo "FAIL ranking.parquet is $b bytes, expected $RANKING_BYTES"; fail=1; }
  [ "$s" = "$RANKING_SHA_PREFIX" ] || { echo "FAIL ranking.parquet sha starts $s, expected $RANKING_SHA_PREFIX"; fail=1; }
  [ "$b" = "$RANKING_BYTES" ] && [ "$s" = "$RANKING_SHA_PREFIX" ] && echo "OK   ranking.parquet is the live 2026-09-04 file"
else
  echo "FAIL no ranking.parquet"; fail=1
fi
if [ -f "$sb" ]; then
  b=$(wc -c < "$sb")
  [ "$b" = "$SUBMITTING_BYTES" ] && echo "OK   submitting.parquet is the 2026-09-04 template" \
    || { echo "FAIL submitting.parquet is $b bytes, expected $SUBMITTING_BYTES"; fail=1; }
fi
v1="$ROOT/submissions/uploaded/youthful-giraffe_v1.parquet"
if [ -f "$v1" ]; then
  s=$(shasum -a 256 "$v1" | cut -c1-8)
  [ "$s" = "48472e44" ] && echo "OK   v1 is exp-016 stack-4 (candidate-4.json)" \
    || echo "NOTE v1 sha starts $s, not the 48472e44 of exp-016 stack-4 -- record this"
fi
v2="$ROOT/submissions/uploaded/youthful-giraffe_v2.parquet"
if [ -f "$v2" ]; then
  s=$(shasum -a 256 "$v2" | cut -c1-8)
  [ "$s" = "5d6a9961" ] && echo "OK   v2 is the suspected blend-5 object" \
    || echo "NOTE v2 sha starts $s, so it is NOT the suspected blend-5 5d6a9961 -- record this"
fi
echo
[ "$fail" = 0 ] && echo "ALL CHECKS PASSED" || { echo "CHECKS FAILED"; exit 1; }
