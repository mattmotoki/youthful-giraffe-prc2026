# exp-041-cell — SPEC

Worker `cell`, tier hard, 2026-09-08. Assignment `assignments/cell.md`
(exp-041-cell, Pro queue Q10 + Q11's cheap arm). Laptop, 3 threads, CPU only,
no upload, no fit against a board reading.

## The reading I took

The assignment asks one question — **what carries v10's shared within-cell
departure error at scoring time** — and answers it in three measured steps on
the fold-side v10 of `exp-039-screens/code/v10.py`. I read it as (a point marked
"corrected" carries the review finding that changed it; two Sol rounds ran, and
the repairs made after round 2 were passed by the MODES.md **fallback** reviewer,
not by Sol, because Sol was not available --
`review/round3-fallback-claude-fable-5-1-2026-09-08.txt`, VERDICT OK, no blocking
finding, ten non-blocking of which seven are answered in RESULTS.md section 1):

1. **Step 1 is a decomposition of a covariance, not a regression on a cell
   mean — and the assignment's named statistics are reported beside it.** The
   quantity Q04 measured is `mean(r_i · rbar_i)` with `rbar_i` the
   self-excluded cell mean of the other departures' residuals. Under
   `r_i = μ_c + ε_i` that is an unbiased estimate of `Var(μ_c)` and needs no
   cell-size correction. Every **headline** anatomy number is built from
   **self-excluded pairwise cross-products of the same kind**, restricted to a
   relation (same runway, same operator, lag L bins), because a plug-in cell
   mean's variance is inflated by `Var(ε)/n_c`.
   **Corrected after Sol round 1 (blocking finding 1): naming a better estimator
   is not the same as running the one the assignment asked for.** The three
   statements the assignment makes in words — the cell-mean residual's
   **variance**, the **autocorrelation** of the cell mean at lags 1–6, and the
   cross-fitted **R²** of the cell mean — are now all computed and reported
   (T2b, T6's second table, T7's last two columns), each next to the unbiased
   reading and with the bias term that separates them. The estimator swap is
   therefore a claim the record can check, not one the reader has to accept.
   The runway/operator split is the four-relation crossover the queue's Q10
   names, read as a variance decomposition:

       cov(diff rwy, diff op)                              = Var(airport-window)
       cov(same rwy, diff op) − cov(diff rwy, diff op)     = Var(runway | window)
       cov(diff rwy, same op) − cov(diff rwy, diff op)     = Var(operator | window)

   with the fourth cell `cov(same, same)` reported as the additivity check, not
   assumed. **Corrected after Sol round 1 (blocking finding 2): the four
   relations must be read on ONE population.** Over all eligible rows they are
   not — a single-runway cell contributes only same-runway pairs — so the
   components are now read on the **matched rows**, the rows for which all four
   relations are non-empty, with the unmatched reading kept beside them.
   **Corrected again after Sol round 2 (blocking finding 2): the matched rule is
   not the queue's rule.** `QUEUE.md` Q10 predeclares two operators across two
   runways with at least three rows in each intersection, and asks for the
   unsupported cells to be reported.  `anatomy.q10_subgrid` applies exactly that,
   before any covariance, and the answer at the 10-minute bin is that **no cell
   in any fold satisfies it**; the rule is therefore scanned over wider windows,
   and the verdict is stated as an ordering all three populations agree on.
   **Corrected after Sol round 2 (blocking finding 1) as well:** the components
   are reported over both denominators -- the sum of the components, which is
   what a decomposition adds to, and the shared component read on the same rows,
   which is what the assignment's sentence asks for.
   **Corrected after the round-3 fallback pass (note 4):** neither denominator
   answers the assignment's sentence on its own -- one does not divide by the
   shared component, the other does not add to 100 %. T3c reports the split that
   does both, by weighting each component by the share of pairs that share the
   attribute, which is how the component enters the pooled cross-product. The
   four contributions then add to the shared component exactly, and the runway
   share of it is 68.6 / 81.8 / 70.2 %. The ordering is unchanged.

2. **"How much do the shipped window features already explain" is the same
   contract number, not an R².** A model `ĥ` fitted cross-fitted on cell-level
   observables and constant inside a cell is a servable cell correction; what it
   is worth is `2·mean(r·ĥ) − mean(ĥ²)`, the contract's `2A − B`, on held-out
   day blocks. Reporting it that way makes the diagnostic and the repair
   clause's "oracle-free upper bound" the same measurement, and it prices the
   route instead of describing it.

3. **The matched base is v10's own information, which includes the K2
   transient window block.** Corrected after Sol round 1 (blocking finding 3):
   the first pass built every base on the flag-off 89-column feature cache,
   while v10 serves `refit-bulk` — a K2-lineage member whose frame is built with
   `PRC_WINDOW=1`. `assignments/cell.md` step 2 says so in words ("the shipped
   window block already carries unordered counts **and the K2 transient
   window**"). Every fit here now reads `<month>-w.parquet`: 89 numeric columns
   plus the 68-column transient block plus the 13 shared categorical codes.

4. **An advance clause about a PAIR of arms is decided by a paired interval on
   the primary endpoint.**  Corrected after Sol round 2 (blocking finding 3): the
   first version priced each arm against v10 on the board and compared the arms
   with each other only on a per-fold, eligible-row statistic.  `code/contrast.py`
   now rebuilds both arms inside every draw of the ranking-weighted all-row
   january/july board, with 1/2/3-day moving blocks, so "the ordered
   representation beats the unordered one" and "two factors beat one" are settled
   on the endpoint the contract names.

5. **Portability must be measured on an instrument with no v10 vector in it.**
   Corrected after Sol round 2 (blocking findings 4 and 5): every cross-month arm
   built on `r = y - p10` trains on a target whose `p10` came from members that
   saw the scored month, and the round-1 claim that this leak could only inflate
   the cross arm is **withdrawn** -- a contaminated training target has no proven
   sign.  The v10-based arms stay in the record as the contaminated reading, and
   the verdict rests on the `clean_family`: a base fitted inside this experiment
   on one other month with `p10` dropped from the columns, where every base a
   scored fold uses excludes that fold.  The amplitude is calibrated with both
   the scored fold and the calibration fold excluded.

6. **Step 2's ordered-vs-unordered contrast is a permutation of one feature
   block, not two different feature sets.** The ordered arm carries the last
   four same-runway take-off gaps and the four interleaved arrival counts in
   **time order**; the unordered arm carries the **same eight numbers sorted**.
   Same learner, same capacity, same rows, same seed: the only difference is
   whether the order survives. That is the tightest reading of "at equal
   capacity" the queue's sentence supports.

7. **Step 3's innovation is `AOBT − LOBT` residual to a nuisance fitted on the
   other eleven months**, in the shape of Q04's `Nuisance` (back-off group
   means, scoring month excluded), because the queue says "fit expectations
   without inputs that determine the innovation target" and the off-month
   back-off model is the instrument this repository has already gated.

8. **A placebo is neighbours at another time, not the same neighbours
   relabelled.** Corrected after Sol round 1 (blocking finding 6): step 3's
   first placebo binned every query at its own shifted clock and then grouped,
   which moves the whole month together and reproduces the real grouping. The
   aggregates are now built on the real bins and each query looks its own key up
   at the shifted bin.

## Readings I rejected

- **Reporting only the estimator I preferred.** Rejected after Sol round 1: the
  assignment names three statistics and the record must carry them, whatever the
  headline rests on. They are in T2b, T6 and T7 and they agree with the
  cross-product once the noise term is removed — which is the evidence that the
  swap changed the variance of the estimate and nothing else.
- **Estimating the cell mean and regressing on it.** Rejected: with a median
  cell of 4–6 eligible departures the plug-in cell mean carries `Var(ε)/n_c` of
  noise, which is the same order as the shared component itself, so every
  stratum table would read the cell-size distribution rather than the mechanism.
  The self-excluded cross-product has no such term.
- **Running steps 2 and 3 in the queue's written order.** The assignment says
  the anatomy table decides which runs first; I follow the table.
- **Taking the assignment's `+6,323 / +4,526 / +4,876 s²` as the record.**
  `experiments/exp-039-screens/analysis/q04.json` carries
  `+5,849.9 / +4,214.1 / +4,797.5` on its supported rows (≥6 eligible
  departures **and** ≥6 arrivals in the cell). The conflict is recorded here and
  in RESULTS §1; this experiment recomputes the quantity on its own cohort
  (the arrival condition is irrelevant to a departure-side question) and reports
  both supports, so no number is carried across on trust.
- **Q04's supported cohort as this experiment's cohort.** Rejected: its ≥6
  arrivals condition selects busy cells for a reason that belongs to the arrival
  screen. The cohort here is Q03's — `nm_bulk` outside EHAM and EDDF — with the
  cell-size cut stated per table.
- **Serving anything the anatomy alone suggests.** Step 4 is conditional on an
  arm advancing on the contract; a positive anatomy row is a description of the
  label side and cannot be served.

## Contract

`assignments/COMMON.md` and `QUEUE.md`'s shared measurement contract, as
exp-038 and exp-039 read them: paired **all-row** January/July MSE at exp-016's
ranking month shares is the primary endpoint (`code/packet.py`), clean and
December secondary; day-clustered block bootstrap with a 1/2/3-day moving-block
sensitivity; gates and amplitudes fitted away from the rows they score; all
departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` masked by
`prc.data.mask_like_ranking` before any context is built.
