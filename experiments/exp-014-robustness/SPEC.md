# exp-014 robustness — SPEC

## My reading of the assignment, and the readings I rejected

**The reading I took.** The deliverable is an *exposure audit of the shipped
artifact*, not a new model. "The candidate's frame" is the union of the frames
the shipped blend actually consumes: the frozen **v2 frame**
(`experiments/exp-008-diversity/frames.py`, 54 columns) which `C2-cat` (0.2227)
and `L2-hub` (0.3351) both consume, plus the **tail-signal frame**
(`prc.tail_features.SIGNALS` + exp-009's noac head columns) which `T-tailv2`
(0.4438) consumes. "Training range" is the range over the **twelve 2025 months**,
because that is what the submission fit sees — not a fold's training months.
Exposure is measured on the rows of the live `ranking.parquet`, and every
"what does the model predict there" comparison is made against the **shipped
candidate file** `submissions/candidates/exp-012-blend-1.parquet`, so the audit
describes the bytes that would be uploaded.

**Readings I rejected.**

1. *"Audit only the v2 frame."* Cheaper, and it covers 0.558 of the blend's
   weight, but the largest member is the tail model and its inputs are a
   different set. An audit that silently omitted the largest member's frame
   would be the same failure mode the assignment is written against.
2. *"Re-fit the members with clipping and compare board estimates."* That is the
   honest way to price question 4, and it costs ~3 GPU-hours plus ~2 CPU-hours
   per variant on a box at load 235. I take the cheaper and equally decisive
   form: fit each member **once** per fold, then predict **twice** — raw frame
   and repaired frame — so the fix is measured against its own control with the
   fit held fixed. A clip applied at prediction time is the shipped deployment of
   the fix, so this is the measurement and not an approximation of it.

   **Corrected after round 1 (B2).** That last sentence is true only for a
   feature the model consumes as a tree split. The members fit a *residual*, so
   the served value is `anchor − tree(...)`, and `prc.tail_model` composes
   further (`mvt_aobt − tree`, `mvt_sched − (1−p)·off`, a band switch on
   `mvt_sched`). For those columns a prediction-time clip changes the answer
   arithmetically and re-fitting is a genuinely different experiment.
   `q4_fixes.py` computes the anchor from the raw frame by design, so the
   measurement it reports is about split features only, and RESULTS says so.
3. *"Treat the 2026 out-of-range share as the harm."* Rejected: a share is
   exposure, not error. Every exposure row in this report carries a **2025
   analogue** — the same feature out of range on a held-out 2025 month — whose
   RMSE and bias are measurable, and the judgement column is written from the
   analogue, never from the 2026 share alone.
4. *"Use `december` as the January-2026 fold."* `docs/shift-2026.md` rules it
   out; the january fold (train Feb–Dec 2025, validate Jan 2025) is the winter
   instrument and december is reported only as the calm control.

## Frozen before any number was computed

- Exposure thresholds: a numeric exposure is **material** at ≥ 200 ranking rows
  outside the twelve-month min–max, or ≥ 0.5 % of an airport-month.
- A fix is **adopted** only if it is non-negative on `july_loo` *and* `january`
  within their measured spread and reduces a demonstrated exposure. A fix that
  helps one fold and hurts the other is recorded as a MAYBE with its revival
  condition, not adopted.
- `nm_late`, `noac`, `nonm`, `nm_bulk` keep `prc.blend`'s definitions.

## Threads and compute

12 threads (`PRC_THREADS=12`, `OMP_NUM_THREADS=12`), GPU 1 for CatBoost when a
refit is needed. Box was at load 235 at start.
