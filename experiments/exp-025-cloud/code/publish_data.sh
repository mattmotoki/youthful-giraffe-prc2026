#!/usr/bin/env bash
# exp-025-cloud: publish the real competition data as a PRIVATE Kaggle dataset.
#
#   bash experiments/exp-025-cloud/code/publish_data.sh [create|version] ["message"]
#
# Run this only after code/fetch_data.sh has pulled data/raw/ and verified it.
#
# The competition data may not be redistributed. Private is the only acceptable
# visibility, so this script refuses to run unless it can prove the result is
# private: it never passes --public, and after the upload it downloads the
# dataset's own metadata and stops with a loud message if `isPrivate` is not
# true. If that check ever fails, delete the dataset on Kaggle before doing
# anything else.
set -euo pipefail
MODE="${1:-create}"
MSG="${2:-competition data}"
SLUG="${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/${PRC_KAGGLE_DATA_SLUG:?set PRC_KAGGLE_DATA_SLUG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW="$ROOT/data/raw"
K="$HOME/.local/bin/kaggle"

# ---------------------------------------------------------------- preflight
[ -d "$RAW" ] || { echo "no $RAW; run code/fetch_data.sh first" >&2; exit 2; }
rank="$RAW/ranking.parquet"
[ -f "$rank" ] || { echo "no $rank" >&2; exit 2; }
bytes=$(wc -c < "$rank" | tr -d ' ')
[ "$bytes" = "43564605" ] || { echo "ranking.parquet is $bytes bytes, not the live 43564605" >&2; exit 2; }
sha=$(shasum -a 256 "$rank" | cut -c1-8)
[ "$sha" = "9db75e3a" ] || { echo "ranking.parquet sha256 starts $sha, not 9db75e3a" >&2; exit 2; }
n=$(ls "$RAW"/training_2025-*.parquet 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "12" ] || { echo "found $n training months, expected 12" >&2; exit 2; }
echo "preflight ok: 12 months, ranking 43564605 bytes sha 9db75e3a…"

# ------------------------------------------------------------------- upload
STAGE="$(mktemp -d)"
cp "$RAW"/*.parquet "$STAGE/"
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "prc 2026 data",
  "id": "$SLUG",
  "licenses": [{"name": "other"}]
}
JSON
cd "$STAGE"
case "$MODE" in
  create)  "$K" datasets create  -p . --dir-mode skip ;;          # private by default; --public is never passed
  version) "$K" datasets version -p . --dir-mode skip -m "$MSG" ;;
  *) echo "mode must be create or version" >&2; exit 2 ;;
esac

# ------------------------------------------------- prove it came out private
# A brand-new private dataset answers 403 to every read while Kaggle processes
# it, and the CLI exits non-zero on that, so the loop must not be under `set -e`.
ready=0
for i in $(seq 1 90); do
  s=$("$K" datasets status "$SLUG" 2>&1 | head -1 | tr -d '\n') || true
  echo "status $s"
  if [ "$s" = "ready" ]; then ready=1; break; fi
  sleep 20
done
[ "$ready" = 1 ] || { echo "*** $SLUG never reached ready; check it by hand before using it" >&2; exit 1; }
CHK="$(mktemp -d)"
"$K" datasets metadata "$SLUG" -p "$CHK" >/dev/null
priv=$(python3 -c "import json; d=json.load(open('$CHK/dataset-metadata.json')); d=d.get('info', d); print(d.get('isPrivate'))")
if [ "$priv" != "True" ]; then
  echo "*** STOP: $SLUG reports isPrivate=$priv. The competition data may not be" >&2
  echo "*** redistributed. Delete the dataset on Kaggle now, then investigate." >&2
  exit 1
fi
echo "$SLUG is ready and private (isPrivate=$priv)"
echo "point kernel-metadata.json dataset_sources at $SLUG and unset PRC_SYNTHETIC"
