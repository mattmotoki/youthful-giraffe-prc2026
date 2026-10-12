#!/usr/bin/env bash
# exp-045-harvest: pull one harvest session's log, run records and summaries.
#
#   bash experiments/exp-045-harvest/code/pull_kernel.sh 2026-01
#
# The points parquets are NOT pulled: they are the month's product and they stay on
# Kaggle (SPEC.md, decision 2).  `--file-pattern` keeps the transfer to a few hundred
# KB a day, which is the only size this laptop has room for.
#   log            -> logs/<name>.log
#   run_<day>.json -> analysis/runs/
#   session.json   -> analysis/sessions/<name>.json
#   segments       -> artifacts/exp-045/summaries/<airport>/<day>.parquet
#   callsigns      -> artifacts/exp-045/callsigns/<day>.parquet
#   census         -> artifacts/exp-045/census/<day>.parquet
set -euo pipefail
NAME="${1:?kernel name, e.g. 2026-01}"
SLUG="prc-harvest-$NAME"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-045-harvest"
ART="$ROOT/artifacts/exp-045"
# stay off /private/tmp, as exp-043 does: the data volume runs out there first
mkdir -p "$EXP/work"
TMP="$(mktemp -d "$EXP/work/pull-XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

"${PRC_KAGGLE_BIN:-$HOME/.local/bin/kaggle}" kernels output "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$SLUG" -p "$TMP" -q --page-size 200 \
  --file-pattern '^(segments|callsigns|census|run|session)' >/dev/null
mkdir -p "$EXP/logs" "$EXP/analysis/runs" "$EXP/analysis/sessions" \
         "$ART/summaries" "$ART/callsigns" "$ART/census"
if [ -e "$TMP/$SLUG.log" ]; then
  python3 - "$TMP/$SLUG.log" "$EXP/logs/$NAME.log" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as fh:
    for rec in raw:
        fh.write(rec.get("data", ""))
PY
fi
[ -e "$TMP/session.json" ] && cp "$TMP/session.json" "$EXP/analysis/sessions/$NAME.json" || true
for p in "$TMP"/run_*.json; do [ -e "$p" ] && cp "$p" "$EXP/analysis/runs/$(basename "$p")" || true; done
for p in "$TMP"/callsigns_*.parquet; do [ -e "$p" ] && cp "$p" "$ART/callsigns/$(basename "$p")" || true; done
for p in "$TMP"/census_*.parquet; do [ -e "$p" ] && cp "$p" "$ART/census/$(basename "$p")" || true; done
"$ROOT/.venv/bin/python" - "$TMP" "$ART/summaries" <<'PY'
import sys
from pathlib import Path
import pandas as pd
tmp, out = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for p in sorted(tmp.glob("segments_*.parquet")):
    day = p.name[9:19]
    df = pd.read_parquet(p)
    for airport, g in df.groupby("airport", observed=True):
        d = out / str(airport)
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["airport"]).to_parquet(d / f"{day}.parquet", index=False,
                                               compression="zstd")
        n += 1
print(f"summaries: {n} airport-days written under {out}")
PY
echo "$NAME: $(ls -1 "$EXP/analysis/runs" | wc -l | tr -d ' ') run records on disk"
