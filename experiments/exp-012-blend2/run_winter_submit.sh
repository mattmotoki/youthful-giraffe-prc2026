#!/usr/bin/env bash
# The winter member's ranking predictions: twelve months, three seeds, and the
# budget drawn under the desk's rule (2026-09-05) - median of per-month optima
# over three held-out months spanning both seasons, spread reported, the smaller
# count when the spread is wide. This replaces retro-v1's single contiguous
# last-15%-by-take-off-time slice, which is one late-autumn month and is exactly
# the instrument that cost the Huber arm 3.57 s.
# PRC_THREADS=24 matches exp-007 and is recorded in the run record; this arm has
# linear leaves and is NOT reproducible run to run even at a fixed seed and
# thread count (see RESULTS), so the pin is a statement of what was run, not a
# guarantee of what a re-run would give.
set -euo pipefail
cd "$(dirname "$0")"
export PRC_THREADS=${PRC_THREADS:-6}   # docs/COMPUTE.md D3: this lane gets 6
../../.venv/bin/python winter_arm.py --name D1-winter --submit --seeds 7,107,207
