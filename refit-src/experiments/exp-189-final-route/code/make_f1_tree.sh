#!/usr/bin/env bash
# exp-189: the source tree of F1-fresh in the final mode: the code of <src> (its artifacts left out),
# the twelve 2025 exp-018 month caches of the frozen tree of the v73 member fits, and finaldata's four
# final exp-018 tables with their vocab.json (exp-185 MANIFEST, 996c97548: one vocab for the four
# months; the frozen vocab ef056bb1 is a prefix of it in all 13 columns, so the 2025 codes hold).
#   make_f1_tree.sh <src tree> <new tree>
# Writes <new>/artifacts/CACHES.sha256 (the frozen copies, checked against the frozen tree) and
# <new>/artifacts/FINALDATA.sha256 (finaldata's files, checked against the MANIFEST SHA-256).
set -euo pipefail
SRC=$1; NEW=$2
FROZEN=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-src
FD=${PRC_FINAL_ROOT:-/path/to/final}/finaldata/mac/e018-final
RAW=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw
C=artifacts/exp-018-fresh/cache
[ -e "$NEW" ] && { echo "exists: $NEW"; exit 1; }
mkdir -p "$NEW"
(cd "$SRC" && tar --exclude=./artifacts --exclude=./data -cf - .) | tar -xf - -C "$NEW"
mkdir -p "$NEW/$C" "$NEW/data"
ln -s "$RAW" "$NEW/data/raw"
files=()
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do files+=("$C/2025-$m.parquet"); done
for f in "${files[@]}"; do cp -p "$FROZEN/$f" "$NEW/$f"; chmod a-w "$NEW/$f"; done
(cd "$NEW" && sha256sum "${files[@]}") > "$NEW/artifacts/CACHES.sha256"
(cd "$FROZEN" && sha256sum -c --quiet "$NEW/artifacts/CACHES.sha256")
cat > "$NEW/artifacts/FINALDATA.sha256" <<'EOF'
45c7ddadd4dfc897b809528d7a2adfb8798f0656a15a8431d265378967f92a00  artifacts/exp-018-fresh/cache/2026-01-final.parquet
467e7bfa47b5d40ded5a6d283d4505bffe0722960c37def7a0fd0e2f828950cf  artifacts/exp-018-fresh/cache/2026-02-final.parquet
08a8b475c8f2db73802280d427de9fe44d0c57be514c27e98ef0c680df8c979a  artifacts/exp-018-fresh/cache/2026-06-final.parquet
1b33836683ae22d482abbdd1d9a905187cecde6a2b53481b15e105f3ceaf0b27  artifacts/exp-018-fresh/cache/2026-07-final.parquet
3319756d588505a73693342d1a0eb3f0436cad32a931dfc29b8460a3a09edc23  artifacts/exp-018-fresh/cache/vocab.json
EOF
for m in 2026-01 2026-02 2026-06 2026-07; do cp -p "$FD/$m-final.parquet" "$NEW/$C/$m-final.parquet"; done
cp -p "$FD/vocab.json" "$NEW/$C/vocab.json"
chmod a-w "$NEW/$C"/2026-*-final.parquet "$NEW/$C/vocab.json"
(cd "$NEW" && sha256sum -c --quiet artifacts/FINALDATA.sha256)
echo "F1 tree $NEW: ${#files[@]} frozen caches and 5 finaldata files, SHA-256 checked"
