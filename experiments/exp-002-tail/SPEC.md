# exp-002-tail — SPEC

tier: hard | reviewer: sol | threads: 16 | gpu: none | assignment: `assignments/tail.md`

## Reading of the assignment (and the readings rejected)

I read the assignment as: **find the mechanism that generates the extreme
squared error, then build the RMSE-optimal predictor for it on top of a fixed
bulk model**, reporting the gain separately on each subset. The metric is RMSE
on the whole fold, so the deliverable is judged by fold RMSE, not by tail
accuracy in isolation.

Readings I rejected:

1. *"Model the tail as a heavy-tailed regression problem"* — rejected after the
   anatomy. The extreme rows are not a heavy tail of a physical taxi-time
   distribution; they are rows whose recorded `BLOCK_TIME_UTC_mvt` is a
   **placeholder equal to `SCHED_TIME_UTC_mvt`**. The target on those rows is a
   data convention, not a taxi duration. That makes the problem a *detection*
   problem (does this row carry a placeholder block time?) with a closed-form
   value when detected (`y = MVT_TIME - SCHED_TIME`), not a regression on
   physical covariates.
2. *"The no-NM rows are the tail"* — rejected. The two sets overlap only
   partly. 22,470 training rows have no NM join; 1,507 of them (the ones with
   `AIRCRAFT_TYPE_mvt` also null) carry essentially the whole extreme tail. The
   remaining 20,963 no-NM rows have median taxi 912 s and P(y>3600) = 1.6 %:
   they are an ordinary population that merely lacks `AOBT_3_flt`. Treating
   "no NM" as one segment mixes a 0.07 %-of-rows catastrophe with a 1 %-of-rows
   nuisance and gets the treatment wrong for both.
3. *"Hard-switch on the tail classifier"* — rejected on the metric. Under RMSE
   the optimal prediction is the conditional expectation
   `p·m_tail + (1-p)·m_bulk`, never `argmax`. Question 4 asks for exactly this
   and the measurement in RESULTS.md confirms it.
4. *"Wait for retro-v1's bulk features"* — rejected per the assignment; I build
   a minimal fixed bulk model here so tail gains are measured against a stable
   reference, and export the tail signals as `prc/tail_features.py` so retro-v1
   imports them rather than the reverse.

## Frozen conventions

- Airports: the 10 with departures in the ranking file (`LTFM EHAM LFPG EDDF
  EGLL LEMD LEBL EDDM LIRF LSZH`). LTAI has no departures.
- Folds: `july` and `december` from `prc.data.fold_frames`; validation features
  computed from the masked frame alone.
- Every table reports RMSE overall, on `y<=3600`, on `y>3600`, and on no-NM
  rows, per fold, as `assignments/COMMON.md` requires.
- Data: `prc.data.data_dir()` (live 2026-09-04 ranking, SHA-256 `9db75e3a…`).
