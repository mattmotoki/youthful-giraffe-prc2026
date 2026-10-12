#!/usr/bin/env bash
# exp-033-noac, end to end, in dependency order. ~12 minutes on 6 threads, no
# GPU. Needs `data/raw` (the twelve 2025 months and ranking.parquet) and
# `experiments/exp-026-rebuild/oof/stack5-*.parquet`.
#
# The order matters in two places: `place.py` must run after `describe2.py`
# (both write `work/noac_2026_d.parquet`), and `join_labels.py` is the §5
# boundary -- nothing before it may read a 2025 label.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=".venv/bin/python"
export OMP_NUM_THREADS=6
C=experiments/exp-033-noac/code

# --- step 1, label-free ------------------------------------------------------
$P $C/build.py            # the two population frames + labels_2025.parquet
$P $C/describe.py         # per-band description and the raw distances
$P $C/describe2.py        # NN provenance, per-feature shares, the noac rate
$P $C/place.py            # inside / edge / outside, cross-day reference
$P $C/twosample.py        # day-grouped two-sample test, controls and null
# --- step 2, the labels enter here -------------------------------------------
$P $C/join_labels.py      # noac_2025_lab.parquet + the placeholder-definition table
$P $C/conditional.py      # the whole-set match (why it is not the instrument)
$P $C/conditional2.py     # the within-band match, distinct-row estimand
$P $C/placebo_step2.py    # the query-permutation null for the label difference
# --- step 3 ------------------------------------------------------------------
$P $C/direction.py        # LOMO, fit set all vs matched
$P $C/direction2.py       # the shipped base and its arms
$P $C/why.py              # implied mixture coordinate, band and shrinkage repairs
$P $C/transfer.py         # base coordinate 2025 vs 2026, oof band choice, shift
$P $C/clf_arm.py          # the learned two-class head
$P $C/robust.py           # k / space / m / tolerance sensitivity
$P $C/placebo2.py         # the query-permutation null for the fold gain
$P $C/per_fold.py         # every band, one fold at a time
# --- step 4 ------------------------------------------------------------------
$P $C/table.py            # the band table and the Holm family
$P $C/board_price.py      # rho and the board arithmetic
$P $C/records.py          # arm summary, mixture coordinate, the 8h+ arithmetic
