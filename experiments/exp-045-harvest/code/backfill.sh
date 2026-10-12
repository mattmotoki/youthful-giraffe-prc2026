#!/usr/bin/env bash
# exp-045-harvest: bring every month fetched under the shipped rule up to the current one.
#
# The pilot's reviewer repaired the rule on 2026-09-08, after the gate and four months had
# been fetched.  The points are unaffected, so each of those is re-derived from its stored
# Kaggle output rather than re-fetched.  The desk's condition: a month is not reported
# complete until its summaries carry the current rule.
#
# The re-derive runs HERE, not in a Kaggle session.  It reads points and writes summaries;
# it needs no internet and no archive, a day of points is 6 MB, and a month re-derives in
# under two minutes.  A session is the scarce thing -- five in the account, shared with the
# other workers -- so spending one on arithmetic would have slowed the fetch for nothing.
# The kernel file is executed as it stands with its directories redirected, so the bytes
# are the same either way.  See code/rederive_local.py.
#
#   bash experiments/exp-045-harvest/code/backfill.sh 2025-07 2026-07 2026-01-b 2025-12
#
# No here-strings and no here-documents below this line: bash spools both to a file under
# TMPDIR, and the volume this laptop shares filled at 11:38 UTC.  The spool failed, the
# grep that reads a command's output never ran, and its non-zero exit was read as success.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-045-harvest"
K="$HOME/.local/bin/kaggle"
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
state() { $K kernels status "${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/$1" 2>&1 | tail -1; }
# the terminal states, matched on Kaggle's own enum: a 404 also says "Error"
final() { state "$1" | grep -qE 'KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)'; }

# Whichever month is ready, not whichever is first.  The four fetches were pushed
# together and finish in any order; 2026-01-b completed at 13:29 UTC while this loop
# was still waiting on 2025-07, and until it is re-derived manifest.json reports two
# rules and surfacefit reads a mixture.  So each pass takes every queued month whose
# fetch has finished and leaves the rest queued.
PENDING=("$@")
while [ ${#PENDING[@]} -gt 0 ]; do
  READY=0
  REST=()
  for NAME in "${PENDING[@]}"; do
    SRC="prc-harvest-$NAME"
    if ! final "$SRC"; then REST+=("$NAME"); continue; fi
    ST="$(state "$SRC")"
    case "$ST" in
      *COMPLETE*) ;;
      *) say "$NAME: source ended ${ST##*status }, not re-deriving"; READY=1; continue ;;
    esac
    READY=1
    OK=0
    for i in $(seq 1 60); do
      if "$ROOT/.venv/bin/python" "$EXP/code/rederive_local.py" "$NAME" 2>&1 | tail -3; then
        OK=1; break
      fi
      say "$NAME: re-derive did not finish (try $i, $(df -m "$EXP" | awk 'NR==2{print $4}') MB free), waiting"
      sleep 300
    done
    if [ "$OK" = 0 ]; then say "$NAME: NOT BACKFILLED, gave up after 60 tries"; continue; fi
    "$ROOT/.venv/bin/python" "$EXP/code/manifest.py" 2>&1 | head -1
    "$ROOT/.venv/bin/python" "$EXP/code/tables.py" --write 2>&1 | head -1
    say "$NAME: BACKFILLED"
  done
  PENDING=("${REST[@]}")
  if [ ${#PENDING[@]} -eq 0 ]; then break; fi
  [ "$READY" = 0 ] && say "waiting on ${PENDING[*]}"
  sleep 180
done
say "backfill queue empty"
