# exp-033-noac — the 388 ranking noac rows as their own population

Assignment: `assignments/noac-learn.md`. Worker `noac`, tier hard, reviewer `sol`.
Started 2026-09-08 00:30 UTC on the Mac laptop (12 cores, 6 threads used, no GPU).

## My reading of the assignment, and the readings I rejected

The assignment asks one question in two halves: **which 2025 sub-population do
the 388 belong to**, and **on that sub-population, does a fold measure a
direction**. I read the deliverable as an *instrument plus a verdict*, not as a
candidate: step 3 states the condition under which an arm becomes a candidate
(positive fold gain, interval clear of zero, shift under 500 s RMS) and says
"otherwise it is a reading", so a reading is a complete answer and I do not have
to manufacture an arm that passes.

Readings I rejected:

1. **"Matched sub-population" = the 2025 rows in the same BAND as the 388.**
   Rejected: the bands are already the record's cells and a band is not a match;
   the assignment says "the 2025 rows that are the 2026 rows' nearest
   neighbours", which is a statement about the feature space, not the band.
2. **Match in the whole `SIGNALS` space.** Rejected: the assignment names "the
   features the placeholder separator uses", which is
   `prc.tail_model.NOAC_FEATURES_V2` (the v4 noac head's own list). Using a
   wider space would make the match answer a different question from the one
   the head can act on.
3. **Match globally, ignoring bands.** Measured and reported (§3), then rejected
   as the primary instrument: a band whose neighbours mostly sit in another band
   (neg: 23 % in band) reports a difference that is band composition, not a
   property of the match. The primary instrument matches **within band**.
4. **Score the arm on all noac rows.** The assignment's scope is "the matched
   sub-population only", so the matched scope is primary; the all-noac-rows
   scope is reported beside it because that is the population a served arm
   actually moves, and the two disagree at the gate.

## Decisions I made without asking

- **The matched sub-population is the DISTINCT rows** that are a within-band
  k-nearest neighbour of one of the 388, counted once each, not the neighbour
  slots weighted by multiplicity. Reason: that is the set a head is fitted on,
  so steps 2 and 3 estimate one object. The weighted rate is reported as
  sensitivity. (Registered after Sol round 1, S5; before it the two steps used
  different estimands.)
- **The placeholder label** is `|mvt_sched − y| < 300 s`
  (`prc.tail_model.PLACEHOLDER_TOL`), not exp-022's `BLOCK.floor('min') ==
  SCHED.floor('min')`. Reason: only the first reproduces the record's own
  `P_ph_2025` column (`exp-026/analysis/noac_cells.csv`) on every band; the
  second gives 0.19–0.47 where the record says 0.37–0.81. Verified in §1.
- **The bands** are the record's seven (`noac_cells.py` `BAND`/`LAB`), a
  superset of the five the assignment names.
- **The distance** is Gower with an IQR scale and a per-feature clip at 3;
  categoricals contribute 0 or 1; a feature missing on either side is dropped
  for that pair. Chosen because the feature list is mixed-type with heavy tails
  (`mvt_sched` reaches 130,000 s) and NaNs, and Gower handles all three without
  imputation.
- **k = 5 and the row-level space** for the matched set, fixed after §4 measured
  the label difference at k = 1, 3, 5, 10 in both spaces and found it stable.
  Sensitivity to both is reported for every downstream number (§6c).
- **Two feature spaces**, because the six day-level columns (`dep_day_n`,
  `noac_day_n`, `noac_day_share`, `noac_day_count`, `nonm_day_count`,
  `arr_day_ph_rate`) are constant within a calendar day and nearly unique to
  it. `full` keeps them, `rowlevel` drops them.

## What I built

Twenty scripts under `code/`, in the dependency order `code/run_all.sh` runs
them (Sol R2-8; the round-1 inventory named two scripts that no longer exist and
omitted three that replaced them):

- **step 1, label-free**: `build.py` (the two population frames, no label
  column), `describe.py`, `describe2.py`, `twosample.py` (the shift test, its
  day-permutation null and the 2025 month controls), `place.py` (the placement
  and the Gower matrices).
- **the label boundary**: `join_labels.py` — the only script that reads a
  target, and everything after it may.
- **step 2**: `conditional.py` (whole-set match, kept to show why it is not the
  instrument), `conditional2.py` (within band, the distinct-row estimand).
- **step 3**: `direction.py`, `direction2.py`, `clf_arm.py`, `transfer.py`,
  `robust.py`, `per_fold.py`, `why.py`, `board_price.py`, and the two nulls
  `placebo2.py` (fold gain) and `placebo_step2.py` (label difference), both
  permuting the query side with the ranking months' day-block structure.
- **the record**: `records.py` (the arm summary, the mixture coordinate and the
  8h+ arithmetic, so no number in RESULTS.md is computed by hand) and
  `table.py` (the band table, the declared Holm family, the ALL row).

The round-1 `placebo.py` and `placebo_bands.py` were deleted rather than left to
be cited, with the superseded CSVs they wrote. Nothing in `prc/` was changed; no
candidate parquet was written; no upload was made (score-feedback rule).
