#!/usr/bin/env bash
# exp-186: a final EXP164_WORK root for the mlp chain (exp-164 mlp008/prepare.py --fit final,
# apply_final186.py, apply_mix12_186.py; then part_mlp_186.py reads its APPLY outputs).
#   bash experiments/exp-186-final-core/code/make_e164_work.sh <root> <snapshot>
# <snapshot> is the git archive of main (${PRC_FINAL_ROOT:-/path/to/final}/finalcore/src/<commit>); code/ links to its
# exp-164 code. The saved inputs, models and records that the chain reads are symlinks into the board work tree
# (E164_BOARD_WORK, default prc-exp164-work); the receipts inside them name board paths, which the chain only
# reads. Every folder the chain writes to is a real folder of <root>, so no write lands in the board tree:
#   cache/proxy-raw-months-<key>.pkl        prepare.py:260 -> exp-153 learner_wp.py:138-142 (final key)
#   inputs/mlp-008-final/final/             prepare.py:213-214 (folder), :154-178 (arrays, preprocessing.json),
#                                           :379-381 (held-metadata, train-target), :410 (INPUTS.json)
#   results/mlp-008/CHECK-INPUTS-final-final.json   prepare.py:411
#   out/mlp-008-final186/final/             apply_final186.py:60 (folder), :87, :112 (seed files), :119 (mean),
#                                           :120 (APPLY.json)
#   out/mlp-012-final186/final/             apply_mix12_186.py:48 (folder), :72, :112 (seed files), :117 (bags),
#                                           :123 (mix), :124 (APPLY.json)
# The ranking-mode folders (inputs/mlp-008/<fit> except final, out/mlp-*-replay186) are also real. The root
# holds no results/HARNESS.json: prepare.py then needs EXP164_HARNESS (the final record) and cannot fall back
# on the board record. Stops when <root> exists. Writes ROOT.txt (links, targets, SHA-256 of the pinned records).
set -euo pipefail
ROOT=${1:?usage: make_e164_work.sh <root> <snapshot>}
SNAP=${2:?usage: make_e164_work.sh <root> <snapshot>}
BOARD=${E164_BOARD_WORK:-${PRC_COMP_ROOT:-/path/to/competitions}/prc-exp164-work}
digest() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
CODE=$SNAP/experiments/exp-164-double-down/code
if [ -e "$ROOT" ] || [ -L "$ROOT" ]; then echo "$ROOT exists; stop" >&2; exit 1; fi
for f in mlp008/prepare.py mlp008/apply_final186.py mlp008/apply_mix12_186.py mlp008/io_utils.py seasonal/common.py; do
  [ -f "$CODE/$f" ] || { echo "snapshot has no $CODE/$f; stop" >&2; exit 1; }
done
SUBS="out/mlp-010/e80/final out/mlp-010/e80/final-seeds-703-704-705-706 out/mlp-010/drop0/final out/mlp-010/drop0/final-seeds-703-704-705-706"
LINKS="inputs/mlp-007 inputs/mlp-008/final out/mlp-008/final $SUBS out/mlp-012/vectors results/mlp-008/SPEC-008.md"
PINNED="inputs/mlp-007/INPUTS.json inputs/mlp-008/final/INPUTS.json out/mlp-008/final/FIT.json out/mlp-012/vectors/BAG-mix.json results/mlp-008/SPEC-008.md"
for s in $SUBS; do PINNED="$PINNED $s/FIT.json"; done
for p in $PINNED; do [ -f "$BOARD/$p" ] || { echo "board tree has no $BOARD/$p; stop" >&2; exit 1; }; done

WRITES="cache inputs/mlp-008-final results/mlp-008 out/mlp-008-final186 out/mlp-012-final186"
mkdir -p "$ROOT"
for d in $WRITES inputs/mlp-008 out/mlp-008 out/mlp-010/e80 out/mlp-010/drop0 out/mlp-012; do mkdir -p "$ROOT/$d"; done
ln -s "$CODE" "$ROOT/code"
for l in $LINKS; do ln -s "$BOARD/$l" "$ROOT/$l"; done

# every write folder (and each parent up to the root) is a real folder that resolves inside the root
REAL=$(cd "$ROOT" && pwd -P)
for d in $WRITES inputs out results out/mlp-010 inputs/mlp-008 out/mlp-008 out/mlp-012; do
  [ -d "$ROOT/$d" ] && [ ! -L "$ROOT/$d" ] || { echo "$ROOT/$d is not a real folder; stop" >&2; exit 1; }
  case "$(cd "$ROOT/$d" && pwd -P)" in "$REAL"/*) ;; *) echo "$ROOT/$d resolves outside the root; stop" >&2; exit 1;; esac
done
{
  echo "# exp-186 final EXP164_WORK root; made $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "board	$BOARD"
  echo "snapshot	$SNAP"
  echo "link	code	$CODE"
  for l in $LINKS; do echo "link	$l	$BOARD/$l"; done
  for d in $WRITES; do echo "write	$d"; done
  for p in $PINNED; do echo "sha256	$p	$(digest "$ROOT/$p")"; done
} > "$ROOT/ROOT.txt"
cat "$ROOT/ROOT.txt"
