# exp-036-mask — cheap cuts on what is already served

Assignment `assignments/mask.md`. Worker `mask`, tier medium, 4 threads, CPU only, reviewer `sol`
(one round, on the shipping arm). The comparison point for every arm is **K2 (v9)**,
`submissions/candidates/exp-027-k2.parquet`, SHA-256 `a3a139db…`, clean **257.1653**, board
**263.0827** (`docs/LEDGER.md` BOARD READING 8). The base under the mask is **v7**,
`submissions/candidates/exp-030-eham-out.parquet`, SHA-256 `67b70ee3…`, clean 257.4598, board
263.4623.

---

## 1. The reading this experiment took

**"The K2-minus-v7 change served or not, per airport" is a serving restriction on two files
that already exist, and nothing is refitted.** v7 gained 0.230 s of board for 0.163 s clean by
taking one airport back out of a served change (exp-030 arm 3); K2 is a second served change
over the same rows, 306,253 of them; the arm asks the same question of it. Concretely, an arm
here is `np.where(airport in keep, K2, v7)` on the fold side over two committed rotations, and
the same statement on the serving side over two verified candidate files. No stacker runs for
arms 1 and 2, and the served candidate is the fold-side statement applied to the ranking rows
with no second model in between.

**The per-airport statistic is separable, and the two directions agree in sign.** For each
airport the table reports *serving K2's change at that airport alone against v7* and *holding
that airport at v7 inside the full K2 mask*. **The clean sum of squared errors is additive over
rows, so the airport's SSE contribution is the same number in both directions; clean RMSE is
monotone in that sum, so both RMSE movements carry the same sign.** Their RMSE *magnitudes* are
not algebraically equal — the square root is concave — and Sol round 1, blocker 3 measured the
largest disagreement in this table at **1.2e-4 s**. `code/mask.py` gates the signs, which is the
property the mask rule uses; it does not gate the magnitudes, because they are not equal. The
fixed unreached-rollover exclusions do not break the additivity: the same clean mask is applied
to every vector.

### What was rejected

1. **A per-airport refit — one stacker, one anchor or one weight set per airport.** Rejected:
   the assignment says "the change served or not", which is a serving restriction, and it says
   "no refits" in as many words. exp-030 arm 3, the result this arm is built on, is itself a
   pure serving restriction; making this one a refit would answer a different question and
   would not be comparable with the 0.230 s the board already paid for that one.
2. **Choosing the mask by board gain per airport.** Rejected: the assignment names the rule —
   *keep the airports where the change helps on all three folds* — and it is exp-030 arm 3's
   own rule. A board-gain rule would keep an airport that trades december for july, which is
   the trade exp-030 arm 2 failed on.
3. **A second mask for refit-bulk, derived from refit-bulk's own per-airport table.** Rejected:
   arm 2 is "**the same** mask on refit-bulk". refit-bulk's own table is computed and printed
   as a diagnostic — it is what says whether arm 2's number is the mask or the anchor repair —
   but no arm selects on it, and the count of airports it would keep is reported rather than
   used.
4. **Treating EHAM as a selectable member of the ten.** v7 and K2 both hold every EHAM
   departure at stack-5's own bytes, so the K2-minus-v7 change is identically zero there and no
   rule can keep it. `code/bases.py` gates that difference at 1e-9 rather than assuming it.
5. **For arm 3, "the 300 s width beside the 600 s block" as a replacement width, or as two
   stackers averaged.** Rejected: exp-030 §1 settled this reading and measured it — two
   aggregate sets side by side in one stacker, the five width-independent row-context columns
   carried once, 95 block columns. This experiment calls exp-030's own `code/blocks.py`, so the
   construction is that one and the only thing that changes is the member in F1-fresh's slot.

## 2. What is not touched

The 388 `noac` rows and the 38,182 EHAM ranking rows. Both are stack-5's bytes in v7, in K2, in
refit-bulk and in every arm here, which is score-feedback rule and exp-030 arm 3 held unchanged. That is what
makes BOARD READING 8's currency rule apply: the noac discount and the anchor offset are common
to K2 and to every arm here and cancel, so the board prediction is **263.0827 minus the arm's
clean gain over K2**, at face.

## 3. The instrument, and what it can and cannot say

Inherited from exp-028 §3 and exp-030 §3, restated rather than re-argued: the lane's three-fold
rotation has an indirect channel — a scored fold's labels reach the stacker through the other
folds' member out-of-fold columns — and no number here is described as nested. Every arm is
compared against K2 **on the same instrument**, which makes the two sides like-for-like and the
difference instrument-relative.

**The mask is selected on the three folds it is scored on.** The per-airport table is computed
on july_loo, december and january, and the arms are then scored on those same three folds. A
gate passed this way is exploratory and is not a second confirmation; the intervals quoted with
it are conditional on the airport set already being chosen and are not adjusted for having
selected 8 of 10. `docs/shift-2026.md` is cited where an airport has a reason that predates
these folds. The board is what settles any of it.

## 4. Deliverables

`code/bases.py` (the four gated fold-side vectors and the gain-over-a-named-base function),
`code/mask.py` (arms 1 and 2, the per-airport tables, the mask), `code/k2w300.py` (arm 3),
`serve.py`, `code/verify.py`, `analysis/`, `oof/`, `logs/`, `climb.log`, `RESULTS.md`, one row
per arm in `docs/LEDGER.md`, one Sol review under `review/`.
