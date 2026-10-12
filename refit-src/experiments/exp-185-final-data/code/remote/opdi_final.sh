#!/usr/bin/env bash
# exp-185: OPDI final match and opdi-first for the four months of final_ranking.parquet (remote, under
# arc_holdguard.sh, in F). Inputs: class-final/<m>/<m>-v25.parquet (ad7cce060, all four months), exp-130/months
# (member rows), opdi/full (exp-155 windows linked, February/June windows extracted by code/opdi/extract_final.py).
#   setsid nohup bash ${PRC_TOOLS:-/path/to/tools}/arc_holdguard.sh <log> bash F/tools/opdi_final.sh >/dev/null 2>&1 < /dev/null &
set -euo pipefail
F=${PRC_FINAL_ROOT:-/path/to/final}/finaldata
cd "$F"
export PYTHONPATH=${PRC_OPDI_ROOT:-/path/to/prc-exp155-opdi}/code OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
M="2026-01 2026-02 2026-06 2026-07"
mkdir -p class-final/v25
for m in $M; do ln -sfn "$F/class-final/$m/$m-v25.parquet" "class-final/v25/$m-v25.parquet"; done
[ ! -e opdi/match-final ] && [ ! -e opdi/first-final ] || { echo "opdi/match-final or opdi/first-final exists"; exit 1; }
# shellcheck disable=SC2086
python3 -u code/opdi/match_final.py --work ${PRC_OPDI_ROOT:-/path/to/prc-exp155-opdi} \
  --raw ${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/data/raw --class-root class-final/v25 \
  --out opdi/match-final --months $M --flights opdi/full/flights.parquet --events-dir opdi/full \
  --ranking raw/final/final_ranking.parquet
# shellcheck disable=SC2086
python3 -u code/opdi/prepare_first_final.py --root "$F" --out opdi/first-final --episode first \
  --member-input-root exp-130/months --manifest opdi/match-final/extract-manifest.json --months $M
sha256sum class-final/v25/* opdi/match-final/* opdi/first-final/*
echo OPDI-FINAL-OK
