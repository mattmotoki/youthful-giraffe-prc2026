#!/usr/bin/env bash
# Upload a reviewed candidate as youthful-giraffe_v<N>.parquet, verify by listing, print the log row stub.
# Usage: scripts/upload_candidate.sh <candidate.parquet> <N>
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
: "${AWS_PROFILE:?Set AWS_PROFILE to the credential profile for the submission service}"
F="$1"; N="$2"; E="${PRC_S3_ENDPOINT:-https://s3.opensky-network.org}"; B=s3://${PRC_SUBMISSION_BUCKET:?set PRC_SUBMISSION_BUCKET}
[ -f "$F" ] || { echo "no such file: $F"; exit 1; }
SHA=$(shasum -a 256 "$F" | cut -d' ' -f1); SIZE=$(stat -f%z "$F")
KEY="youthful-giraffe_v${N}.parquet"
if aws --profile "$AWS_PROFILE" --endpoint-url "$E" s3 ls "$B/$KEY" | grep -q "$KEY"; then echo "REFUSED: $KEY already exists in the bucket"; exit 2; fi
T=$(date -u +%Y-%m-%dT%H:%M:%SZ)
aws --profile "$AWS_PROFILE" --endpoint-url "$E" s3 cp "$F" "$B/$KEY" --only-show-errors
aws --profile "$AWS_PROFILE" --endpoint-url "$E" s3 ls "$B/$KEY"
echo "UPLOADED $T $KEY sha256=$SHA bytes=$SIZE from $F"
echo "| $N | $T | $F | $SHA | | | (pending) | object key $KEY, $SIZE bytes, verified by listing |"
