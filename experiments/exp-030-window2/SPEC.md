# exp-030-window2 — the next cuts on the transient airport window

Assignment `assignments/window2.md`. Worker `window2`, tier medium, 4 threads, reviewer `sol`
(one round, on the shipping arm only). Base and comparison point: the **v6 candidate**,
`submissions/candidates/exp-028-window.parquet` (SHA-256 `d55b1de9…`), clean 257.6230, board
**263.6922** (`docs/LEDGER.md` BOARD READING 6).

---

## 1. The reading this experiment took

**"Two widths together" means two separate aggregate sets side by side in one stacker: the
600 s block exactly as v6 carries it, plus the same nine aggregates and four lags recomputed
at 300 s, appended as their own columns.** Concretely, arm 1's design matrix is the eleven
member columns, then v6's 50-column `depx` block unchanged, then 45 more columns
(`w_ta_mean_w300` … `w_n_l2_w300`) — 95 block columns in total. The five row-context columns
(`hour`, `dow`, `apt`, `own_ta`, `own_al`) do not depend on the width and are carried **once**;
repeating them per width would hand the stacker five identical columns per extra width and
change what `feature_fraction=0.7` samples, which is a different construction from "the same
block at a second width". Arm 2 is the same statement with 1800 s added, 140 block columns.

The width-tagged columns are built by exp-028's own `stack5w.block_for` at the second width
and then renamed; nothing about how a window aggregate is computed changes. Because the first
50 columns are byte-identical to v6's block, arm 1 strictly contains arm v6's information and
the comparison is "what does a second resolution add", not "what does a different block do".

### What was rejected

1. **A nested window** — the 300 s aggregate expressed as a deviation from the 600 s parent,
   or a 300 s bucket indexed inside its 600 s parent. Rejected: that is a third construction,
   not a second cut on this family, and its nearest relative is already measured — exp-028
   §7a's deviation block, which re-expressed every aggregate against an airport level and
   collapsed the board from +0.414 to +0.061 by trading one fold for another. The assignment's
   own words ("two widths together", "three widths") name a union of aggregate sets.
2. **Two stackers, one per width, averaged or blended.** Rejected: exp-028's whole result is
   that the window belongs *inside* one layer — a second layer on a residual stacker pays a
   refit cost the window columns only buy back (+0.002 s, RESULTS §2–3). An average of two
   stackers is a second layer wearing different clothes, and it also doubles the serving path.
3. **Re-bucketing with overlapping windows** (a 600 s window stepped every 300 s). Rejected:
   it changes what a bucket is on both sides of the fit/serve line, so it is not exp-028's
   code any more, and it answers a resolution question the union of two widths already answers
   more cheaply.
4. **Repeating the row-context columns per width.** Rejected for the reason in §1: they are
   width-independent, so the repeat is a sampling change dressed as a feature.

### The other three arms, as read

- **Arm 3, EHAM out of the served mask**, is a *serving* restriction on the v6 form and not a
  refit: every EHAM departure keeps stack-5's own value and every other row keeps v6's. On the
  fold side that is arithmetic on exp-028's committed w600 rotation; on the serving side it is
  arithmetic on two already-verified candidate files. No stacker runs for this arm.
- **Arm 4, the arrivals keyed on runway and stand**, is exp-028 §10's named revival condition
  for its `arr` block, added to whichever of arms 1–3 is best. The key is the runway's
  **direction** (`27L` → `27`) and the stand's **leading character** (`B41` → `B`, `324` → `3`),
  chosen on coverage measured before any fit: an exact runway match populates only 12.2 % of
  July departure rows because six of the ten airports run segregated parallel operations,
  against 50.7 % for the direction and 58.1 % for the stand block (`code/arrk.py`,
  `analysis/arrk_keys.json`).

## 2. What is not touched

The 388 `noac` rows. They stay stack-5's bytes in every arm and in every served file, exactly
as v6 holds them (score-feedback rule), which is what makes BOARD READING 6's cancellation argument apply here:
the noac discount and the +4.28 anchor are common to v6 and to every arm, so the board
prediction is **263.6922 minus the arm's clean gain over v6**, at face.

## 3. The instrument, and what it can and cannot say

Inherited from exp-028 §3 and restated rather than re-argued: the lane's three-fold rotation
has an indirect channel — a scored fold's labels reach the stacker through the other folds'
member out-of-fold columns — and no number here is described as nested. Every arm is compared
against v6 **on the same instrument**, which makes the two sides like-for-like and makes the
difference instrument-relative, not channel-free.

**Every arm here is a further cut on the same three folds v6 was scored on**, which is
exp-028 §7b's caveat and it does not weaken with repetition: arm 1 and arm 2 choose a
resolution on these folds, arm 3 chooses an airport after seeing a per-airport table computed
on these folds, and arm 4 chooses a key. A gate passed on the folds that selected the variant
is exploratory. What is not exploratory is the board: v6's own reading is the measured
transfer this family gets, and the next board reading is what settles any arm here.

## 4. Deliverables

`code/blocks.py` (multi-width block names, on both sides of the fit/serve line),
`code/arrk.py` (arm 4's block and its key-coverage measurement), `code/rotate.py` (exp-028's
`stack5w.main` with three names rebound), `code/score.py` (one scoring function, gated on
exp-028's published stack-5 and v6 rows), `code/eham.py` (arm 3), `code/airports.py`, `serve.py`, `code/verify.py`,
`tests/block_identity.py`,
`oof/`, `analysis/arm-*.json`, `climb.log`, `RESULTS.md`, one row per arm in `docs/LEDGER.md`,
one Sol review under `review/`.
