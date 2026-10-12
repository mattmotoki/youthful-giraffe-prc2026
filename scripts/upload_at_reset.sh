#!/usr/bin/env bash
# Waits until 00:01 UTC (the scorer's daily limit is 5 submissions per UTC day), then uploads the
# file named in submissions/next_upload.txt ("<path> <N>") as youthful-giraffe_v<N>.parquet and polls
# the board for its score. Edit the pointer file any time before 00:01 UTC to change what goes up.
# The pointer may carry a third field, the expected sha256; the upload is refused if the bytes differ.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
: "${AWS_PROFILE:?Set AWS_PROFILE to the credential profile for the submission service}"
LOG=artifacts/upload_at_reset.log; E="${PRC_S3_ENDPOINT:-https://s3.opensky-network.org}"; B=s3://${PRC_SUBMISSION_BUCKET:?set PRC_SUBMISSION_BUCKET}
now=$(date -u +%s); target=$(date -u -j -f %Y-%m-%dT%H:%M:%S "$(date -u -v+1d +%Y-%m-%d)T00:01:00" +%s)
echo "$(date -u +%FT%TZ) waiting $((target-now)) s until 00:01 UTC" >> $LOG
sleep $((target-now))
read -r F N S < submissions/next_upload.txt
if [ -n "${S:-}" ]; then ACT=$(shasum -a 256 "$F" | cut -d' ' -f1)
  if [ "$ACT" != "$S" ]; then echo "$(date -u +%FT%TZ) REFUSED: $F sha $ACT != pinned $S; not uploading" >> $LOG; exit 3; fi; fi
bash scripts/upload_candidate.sh "$F" "$N" >> $LOG 2>&1 && cp "$F" "submissions/uploaded/youthful-giraffe_v${N}.parquet"
for i in $(seq 1 20); do sleep 60; .venv/bin/python scripts/board_snapshot.py >> artifacts/board_snapshot.log 2>&1
  R=$(aws --profile "$AWS_PROFILE" --endpoint-url "$E" s3 cp "$B/youthful-giraffe_v${N}.parquet_result.json" - 2>/dev/null)
  if [ -n "$R" ]; then echo "$(date -u +%FT%TZ) result: $R" >> $LOG; break; fi; done
tail -1 artifacts/board_snapshot.log >> $LOG
