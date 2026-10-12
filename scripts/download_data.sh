#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${AWS_PROFILE:?Set AWS_PROFILE to the credential profile for the data service}"
: "${PRC_DATA_BUCKET:?Set PRC_DATA_BUCKET to the private data bucket name}"
profile="$AWS_PROFILE"
endpoint="${PRC_S3_ENDPOINT:-https://s3.opensky-network.org}"
bucket="$PRC_DATA_BUCKET"
data_dir="${PRC_DATA_DIR:-${repo_dir}/data/raw}"
artifacts_dir="${PRC_ARTIFACT_DIR:-${repo_dir}/artifacts/eda}"
python_bin="${PYTHON_BIN:-${repo_dir}/.venv/bin/python}"
jupyter_bin="${JUPYTER_BIN:-${repo_dir}/.venv/bin/jupyter}"

if [[ ! -x "$python_bin" || ! -x "$jupyter_bin" ]]; then
  echo "Missing project environment. Run: python3 -m virtualenv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

mkdir -p "$data_dir"

echo "Downloading s3://${bucket} to ${data_dir} with profile ${profile}"
aws \
  --profile "$profile" \
  --endpoint-url "$endpoint" \
  s3 sync "s3://${bucket}" "$data_dir" \
  --exclude "*" \
  --include "*.parquet" \
  --only-show-errors

PRC_DATA_DIR="$data_dir" \
PRC_ARTIFACT_DIR="$artifacts_dir" \
PRC_REPO_DIR="$repo_dir" \
  "$jupyter_bin" nbconvert \
    --to notebook \
    --execute \
    --output 00_data_quality_eda.executed.ipynb \
    --output-dir "$artifacts_dir" \
    --ExecutePreprocessor.timeout=900 \
    "${repo_dir}/notebooks/00_data_quality_eda.ipynb"

"$python_bin" -m json.tool "$artifacts_dir/summary.json"
