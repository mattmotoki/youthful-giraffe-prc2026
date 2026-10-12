# exp-013-nonm — departures without a Network Manager join

worker: congestion (hard) | reviewer: Opus substitute | started 2026-09-05

## My reading of the assignment, and the readings I rejected

The segment is departures with `LOBT_flt` null (which makes every `_flt` column
null) and `AIRCRAFT_TYPE_mvt` present: 3,599 july rows, 1,560 december, 1,372
january, and 5,290 of the 344,841 ranking rows. The noac rows are excluded;
they are a different mechanism and another lane's.

1. **The assignment's oracle is degenerate, and I report that rather than
   report its number as a bound.** `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME`
   and `SCHED_TIME` is on every row, so

       taxi = (MVT − SCHED) − (BLOCK − SCHED)

   holds as an identity. I verified it on all three folds: max absolute error
   **0.000000 s** over 6,531 segment rows. Handing the model the true
   `BLOCK − SCHED` therefore determines the target exactly, and any residual it
   shows is the gradient-boosted tree failing to represent a subtraction, not
   information about the problem. The answer to "how much of the 600 s is the
   unknown off-block versus taxi physics" is **all of it**: for this segment
   there is no taxi-physics component of the error at all, because knowing the
   off-block leaves nothing to predict.
   I still fit it, because the size of the tree's approximation error on an
   exactly-determined target is worth knowing before trusting any other
   feature that enters as a difference of two large quantities — which is what
   every candidate in question 1 is.
2. **The replacement ceiling is NM parity.** The useful question is not "what
   if we knew the block time" but "what if we could estimate the off-block as
   well as the NM feed does on the rows it covers". `BLOCK − AOBT_3` has an
   airport-specific distribution measured in exp-004 (σ ≈ 418 s on july bulk),
   and the NM rows' achieved RMSE is the operational answer. Both are reported
   as the ladder the segment sits on.
3. **Features are restricted to columns that survive on a no-NM row.** No
   `_flt` column appears anywhere in this experiment, including in the
   training rows, so the model cannot learn a relationship it will not have at
   prediction time. Rejected: training on all departures with the `_flt`
   columns present and masking them only at prediction — that is what makes a
   model calibrate on a world it will not see.
4. **Two training populations are measured, not assumed**: all departures with
   NM columns absent, and the segment's own rows only. `prc.tail_model`'s
   RESULTS finding 11 reports that segment-only training loses to a per-airport
   median, so the comparison is worth redoing here rather than inheriting.
5. **`prc.stand_features` (VERSION 1.2.0, another worker) already implements
   the stand-occupancy bound** that I built independently as the `stand` family
   of `prc.surface_features`. I will use theirs where the assignment points at
   it and reconcile the two rather than add a third copy; the duplication is
   reported to the desk.

## Baselines on the segment (RMSE, measured 2026-09-05)

| fold | n | `prc.tail_model` v2 | best retro-v1 line |
|---|---|---|---|
| july | 3,599 | **651.4** | 764.4 |
| july_loo | 3,599 | **569.8** | 633.3 (X4-stand) |
| december | 1,560 | **493.0** | 604.5 |
| january | 1,372 | **2794.2** | 2822.4 |

`prc.tail_model` v2 is the best baseline on every fold, so it is the number to
beat. January is dominated by two LFPG date-rollover rows.
