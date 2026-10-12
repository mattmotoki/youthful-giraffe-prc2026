#!/usr/bin/env bash
# exp-056-q12advocate: pull one finished kernel's predictions into artifacts/exp-056/kaggle/.
#
#   bash experiments/exp-056-q12advocate/code/pull.sh <member> <group>    # group: jul jan dec thin
#
# Write-once. Copies only `<member>--iso_*.parquet`, `port.json` (as port-<member>-<group>.json)
# and the kernel log; checks every copied parquet against the SHA-256 the kernel recorded; lists
# any other downloaded file by name only; appends one receipt line to climb.log.
set -euo pipefail
MEMBER="${1:?member: F1-fresh L2-hub8725 T-tailv4}"
GROUP="${2:?group: jul jan dec thin}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/exp-056-q12advocate"
K="$HOME/.local/bin/kaggle"
LOW="$(echo "$MEMBER" | tr 'A-Z' 'a-z')"
SLUG="${PRC_KAGGLE_USER:?set PRC_KAGGLE_USER}/prc-iso-$LOW-$GROUP"
DST="$ROOT/artifacts/exp-056/kaggle"
PORT="$DST/port-$MEMBER-$GROUP.json"
mkdir -p "$DST"
[ ! -e "$PORT" ] || { echo "$PORT exists; pulls are write-once" >&2; exit 3; }
S="$("$K" kernels status "$SLUG" 2>&1 | tail -1)"
case "$S" in *COMPLETE*) ;; *) echo "$SLUG is not complete: $S" >&2; exit 4 ;; esac
TMP="$(mktemp -d)"
"$K" kernels output "$SLUG" -p "$TMP" > "$TMP/.pull.txt" 2>&1 || { cat "$TMP/.pull.txt" >&2; exit 5; }
LOG="$(ls "$TMP"/*.log 2>/dev/null | head -1 || true)"
[ -n "$LOG" ] && grep -q "ISO OK\|ISO THIN OK" "$LOG" || { echo "$SLUG: no OK line in its log" >&2; exit 6; }
[ -e "$TMP/port.json" ] || { echo "$SLUG: no port.json" >&2; exit 6; }
for f in "$TMP/$MEMBER"--iso_*.parquet; do
    [ ! -e "$DST/$(basename "$f")" ] || { echo "$DST/$(basename "$f") exists" >&2; exit 3; }
done
cp "$TMP/$MEMBER"--iso_*.parquet "$DST/"
cp "$LOG" "$DST/kernel-$MEMBER-$GROUP.log"
cp "$TMP/port.json" "$PORT"
RECEIPT="$(python3 - "$PORT" "$DST" "$TMP" "$MEMBER" <<'PY'
import hashlib, json, os, sys
port, dst, tmp, member = sys.argv[1:5]
p = json.load(open(port))
out, bad = [], []
for fold, w in sorted(p["written"].items()):
    f = os.path.join(dst, f"{member}--{fold}.parquet")
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    (out if h == w["file_sha256"] else bad).append(f"{fold} {w['rows']} rows {h[:16]}")
extra = sorted(x for x in os.listdir(tmp)
               if not (x.startswith(member + "--iso_") or x in ("port.json", ".pull.txt") or x.endswith(".log")))
if bad:
    sys.exit("sha mismatch: " + "; ".join(bad))
wall = sum(s["wall_s"] for s in p["steps"]) / 3600
print(f"git {p['git'][0][:12]}; {'; '.join(out)}; wall {wall:.2f} h; peak host {p['peak_host_used_gb']} GB; "
      f"peak GPU {p['peak_gpu_mib']} MiB; {p['gpu']}; other files: {extra or 'none'}")
PY
)"
echo "$(date -u +%FT%TZ) pulled $SLUG: $RECEIPT" | tee -a "$EXP/climb.log"
