#!/usr/bin/env bash
# exp-189: the final route chain after the member fits, on remote in the final mode. Run it under
# arc_holdguard.sh (COMMON: every remote process); steps 10b, 12 and 13 of SPEC.md.
#   route_parts.sh <learner out dir> <stage> [<member> ...]
#   stage: combine | level2 | parts | parity | all | members
# members: the part and the parity read of each member named (default all eight) from its member file, before
# level 2 (desk 23:45 UTC: lp, l and the CPU members while the GPU members wait). A member part that exists
# stays when its recorded input has the SHA-256 of the member file the stage would read; parts and parity skip it.
# <learner out dir>: the output of the private kernel prc-exp189-learners-final (L-final, Lp-final with their
# records, proxy-features.parquet). Members: $FR/members (CPU), $FR/members-t6 (D1-winter and T-tailv4 at 6
# threads) and $FR/members-gpu (C2-cat, F1-fresh).
# Writes (each write-once): $FR/inputs/r4c-final (combine_final.py), $FR/art-final (links to the eight member
# files), $FR/stack (fold_serve.py --split final), $FR/parts/<part>.{parquet,json} (make_parts.py) and
# $FR/parts/<part>-parity.json (exp-186 parity.py --kind refit: members and r4mks against member-refs,
# route/F/np_base against v73-parts). combine allows the four clock_implied proxy columns to differ on 2025 rows
# (the final stand table names 39 more stands; results/proxy-2025-diff.txt); the saved 2025 rows stay.
set -euo pipefail
LDIR=$1; STAGE=$2; shift 2
ONLY="$*"
FR=${PRC_FINAL_ROOT:-/path/to/final}/finalroute
SRC=$FR/src-b165ab4ee
PY=${PRC_COMP_ROOT:-/path/to/competitions}/prc-data-challenge-2026/.venv/bin/python
CODE=$SRC/experiments/exp-189-final-route/code
PAR=$SRC/experiments/exp-186-final-core/code/parity.py
W=${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp148-work
R4C=$FR/inputs/r4c-final
FEAT26=$FR/inputs/features-final-c3f7e62d/final/features-2026.parquet
FEAT26_SHA=c3f7e62de68b8dae213675072fb6dc04dd15d46eaadca634e66077438cbba6e4
REFS=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/refs/v73-parts.parquet
MREFS=$FR/refs/member-refs.parquet
MREFS_SHA=88d422cb4e1d8d3da2a3f0523048d4fe4caf34df450f8e20918799202f9c64a5
MEMBERS="T-tailv4 D1-winter N1-nmtail L2-hub8725 C2-cat F1-fresh L Lp"
export PRC_RANKING_MODE=final PRC_SOURCE_ROOT=$SRC PRC_DATA_DIR=${PRC_FINAL_ROOT:-/path/to/final}/finalcore/raw
export PRC_FINAL_ROOT=${PRC_FINAL_ROOT:-/path/to/final} PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P=$FR/parts
cd "$SRC"
on() { [ "$STAGE" = "$1" ] || [ "$STAGE" = all ]; }
stamp() { echo "$(date -u +%FT%TZ) route_parts: $*"; }
# D1-winter and T-tailv4: the 6-thread fits (released count, desk 18:10 and 18:36 UTC); $FR/members holds
# their 4-thread noise reads
srcdir() { case $1 in C2-cat|F1-fresh) echo "$FR/members-gpu" ;; L|Lp) echo "$LDIR" ;;
                      D1-winter|T-tailv4) echo "$FR/members-t6" ;; *) echo "$FR/members" ;; esac; }
pname() { case $1 in L) echo l ;; Lp) echo lp ;; *) echo "$1" ;; esac; }
pcol() { case $1 in L) echo pred_L ;; Lp) echo pred_Lp ;; *) echo y_pred ;; esac; }
# member part: write it, or keep an existing one whose recorded input is this file (same SHA-256)
mpart() { local m=$1 f; f=$(readlink -f "$2"); local js=$P/$(pname "$m").json
  if [ -e "$js" ]; then
    $PY -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["inputs"].get(sys.argv[2])==sys.argv[3] else 1)' \
      "$js" "$f" "$(sha256sum < "$f" | cut -d' ' -f1)" || { echo "$js: other input than $f"; return 1; }
    echo "part $(pname "$m") exists, input equal: kept"; return 0; fi
  $PY "$CODE/make_parts.py" member --name "$m" --src "$f" > /dev/null; echo "part $(pname "$m") written"; }
par() { local out=$1; shift; if [ -e "$out" ]; then echo "parity: $out exists, kept"; return 0; fi
        $PY "$PAR" --kind refit --out "$out" "$@" > /dev/null; echo "parity: $out"; }
mpar() { local m=$1 pn; pn=$(pname "$m")
  par "$P/$pn-parity.json" --refs "$MREFS" --refs-sha256 "$MREFS_SHA" --new "$P/$pn.parquet" --pair "$(pcol "$m")=saved:$m"; }

if on combine; then
  stamp combine
  [ "$(sha256sum < "$FEAT26" | cut -d' ' -f1)" = "$FEAT26_SHA" ] || { echo "features-2026 SHA-256 differs"; exit 3; }
  $PY "$CODE/combine_final.py" --features-saved "$W/inputs/features-r4c.parquet" --features-2026 "$FEAT26" \
    --proxy-saved "$W/inputs/proxy-features-r4c.parquet" --proxy-new "$LDIR/proxy-features.parquet" --out "$R4C" \
    --allow-2025-diff clock_implied_row,clock_implied_l-1,clock_implied_l+0,clock_implied_l+1
fi

if on level2; then
  stamp level2
  A=$FR/art-final
  [ -e "$FR/stack" ] && { echo "$FR/stack exists"; exit 1; }
  if [ -e "$A" ]; then
    # a level2 run that stopped after its links (10-09 06:57, the mode check below): keep them when each one
    # points to the member file this stage links
    [ "$(ls -A "$A" | wc -l)" -eq 16 ] || { echo "$A: other files"; exit 1; }
    for m in $MEMBERS; do d=$(srcdir "$m"); for x in parquet json; do
      [ "$(readlink "$A/$m-final.$x")" = "$(readlink -f "$d/$m-final.$x")" ] || { echo "$A/$m-final.$x: other target"; exit 1; }
    done; done
    echo "$A exists, 16 links equal: kept"
  else
    mkdir -p "$A"
    for m in $MEMBERS; do
      d=$(srcdir "$m")
      for x in parquet json; do
        [ -s "$d/$m-final.$x" ] || { echo "missing $d/$m-final.$x"; exit 3; }
        ln -s "$(readlink -f "$d/$m-final.$x")" "$A/$m-final.$x"
      done
    done
  fi
  # every member file is a final-mode output: the members' own record says so (exp008_wp: fit final and
  # 670,790 rows); L/Lp (learner_wp writes no mode key) and all others hold rows of all four final months
  $PY - "$A" <<'PY'
import json, os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.environ["PRC_SOURCE_ROOT"])
from prc import data as D
A = Path(sys.argv[1])
rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
rk = rk[rk.PHASE_mvt == "DEP"]
month = pd.Series(rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").to_numpy(), index=rk.MVT_ID_mvt.astype("int64").to_numpy())
want = sorted(month.unique())
assert want == ["2026-01", "2026-02", "2026-06", "2026-07"], want
for p in sorted(A.glob("*-final.json")):
    m = p.name[: -len("-final.json")]
    ids = pd.read_parquet(p.with_suffix(".parquet"), columns=["MVT_ID_mvt"]).MVT_ID_mvt.astype("int64")
    got = month.reindex(ids.to_numpy())
    assert got.notna().all(), f"{m}: ids outside the final departures"
    by = got.value_counts().sort_index().to_dict()
    if m not in ("L", "Lp"):
        r = json.loads(p.read_text())
        rm = r.get("ranking_mode")
        if rm is not None:
            assert rm.get("mode") == "final" and rm.get("n_ranking") == 670_790, (m, rm)
        else:  # exp008_wp records (C2-cat, L2-hub8725) carry no mode key: fit name and row count (ranking: 344,841)
            assert r.get("fit") == "final" and r.get("rows") == 670_790, (m, r.get("fit"), r.get("rows"))
    assert sorted(by) == want, (m, by)
    print(m, len(ids), by, flush=True)
PY
  $PY experiments/exp-148-short-route/code/fold_serve.py --stage R4mks --inputs r4mks --split final --member-art "$A" \
    --stack ${PRC_MODEL_ROOT:-/path/to/models}/prc-exp148-work/stack --features "$R4C/features-r4c.parquet" \
    --proxy "$R4C/proxy-features-r4c.parquet" --out "$FR/stack"
fi

if on parts; then
  stamp parts
  for m in $MEMBERS; do mpart "$m" "$FR/art-final/$m-final.parquet"; done
  $PY "$CODE/make_parts.py" level2 --src "$FR/stack/R4mksP-final.parquet" > /dev/null
  $PY "$CODE/make_parts.py" route --r4mks "$P/r4mks.parquet" > /dev/null
  ls -l "$P"
fi

if on parity; then
  stamp parity
  for m in $MEMBERS; do mpar "$m"; done
  par "$P/r4mks-parity.json" --refs "$MREFS" --refs-sha256 "$MREFS_SHA" --new "$P/r4mks.parquet" \
    --pair pred_Sb=saved:R4mks-Sb --pair anchor_served=saved:R4mks-anchor
  par "$P/route-parity.json" --refs "$REFS" --new "$P/route.parquet" \
    --pair TAXITIME_SEC_mvt=route --pair TAXITIME_SEC_mvt=saved:r4-kno
  par "$P/F-parity.json" --refs "$REFS" --new "$P/F.parquet" --pair F=rule_frame:F
  par "$P/np_base-parity.json" --refs "$REFS" --new "$P/np_base.parquet" --pair np_base=rule_frame:F
fi
if [ "$STAGE" = members ]; then
  stamp "members ${ONLY:-$MEMBERS}"
  for m in ${ONLY:-$MEMBERS}; do
    case " $MEMBERS " in *" $m "*) ;; *) echo "unknown member $m"; exit 2 ;; esac
    f=$(srcdir "$m")/$m-final.parquet
    [ -s "$f" ] || { echo "missing $f"; exit 3; }
    mpart "$m" "$f"
    mpar "$m"
  done
fi
stamp "stage $STAGE done"
