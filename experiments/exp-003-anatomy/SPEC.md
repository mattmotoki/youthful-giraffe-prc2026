# exp-003-anatomy — SPEC

## My reading of the assignment (and the readings I rejected)

**Reading taken.** Because `MVT_TIME_UTC_mvt` (take-off) is given for every
ranking row, predicting `TAXITIME_SEC_mvt` is *identically* the problem of
predicting `BLOCK_TIME_UTC_mvt`. And because `AOBT_3_flt` is given on 98.5 % of
ranking rows, on those rows the problem is *identically* the problem of
predicting the offset `Δ = BLOCK_TIME_UTC_mvt − AOBT_3_flt`: writing
`nm_taxi = MVT_TIME_UTC_mvt − AOBT_3_flt` (fully observed), we have
`taxi = nm_taxi − Δ`, so for any predictor `taxi_hat = nm_taxi − Δ_hat` the
per-row error is exactly `Δ_hat − Δ` and RMSE on taxi equals RMSE on Δ. I
therefore model Δ as the learning target throughout and reconstruct taxi. I
rejected (a) modelling taxi directly with `nm_taxi` as a feature — algebraically
equivalent but it forces the trees to reproduce an identity across a 3-decade
range; I measure this once as an ablation rather than assume it, and (b)
modelling `log` or a winsorised Δ, which changes the loss the metric rewards.

**Bulk definition.** The assignment scores "rows with taxi time at or below
3600 s and an NM join". I read that literally: the ablation metric is RMSE over
validation departures with `y_true <= 3600` and `AOBT_3_flt` non-null. This
filter uses the hidden label, so it is a *diagnostic* metric, not a deployable
one; every table also reports the four COMMON.md slices (overall, ≤3600, >3600,
no-NM) so the numbers compose with the other workers'. Rejected reading: bulk =
"predicted ≤ 3600", which would make ablations non-comparable across models.

**Fixed ablation model.** One LightGBM configuration, frozen at the first
iteration and never tuned, so that every reported gain is attributable to the
feature set and not to hyperparameters. Train on the fold's training months
(full labels), validate on the masked month. All group statistics are fit on
training months only and transformed onto the masked month, which is what
`prc/stand_features.py` separates.

**Scope I did not take.** I do not build a submission candidate; my OOF
predictions are for the bulk lineage only, and the tail/no-NM path belongs to
the `tail` worker. I do not tune, ensemble, or add weather.

## Deliverables
- `experiments/exp-003-anatomy/RESULTS.md` — ranked findings with measured gains
- `prc/stand_features.py` — importable, fit/transform separated
- `docs/shift-2026.md` — five-minute read for the desk (Q4)
- `experiments/exp-003-anatomy/oof/*.parquet` — best bulk ablation
- ledger rows, Sol review

## Q6 (desk note, 2026-09-05): stand occupancy as hard bounds

The desk added: arrivals keep `BLOCK_TIME_UTC_mvt` (on-block) in the ranking
frame, so stand occupancy brackets a departure's hidden off-block. My reading:
at a stand S the occupancy alternates arrival-on-block → departure-off-block, so
for departure `d` at S with off-block `B`, the arrival that brought the aircraft
gives `B > A_prev` and the next aircraft to park at S gives `B < A_next`; in
taxi terms `MVT − A_next < taxi < MVT − A_prev`. The construction problem is
that "previous/next arrival" has to be defined without knowing `B`: anchoring on
`AOBT_3` breaks exactly on the leave-early tail the bound is meant to catch
(block 13:51, take-off 16:08, another aircraft parks at 14:30 — that arrival is
*after* B, so "last arrival before take-off" is not a lower bound). I therefore
build several candidate anchors, **validate each empirically on 2025 where B is
known** (what share of rows the bound actually bounds, and how tight), and only
hard-clip with bounds whose measured violation rate is negligible; the rest
enter as features. Stand keys are normalised (naming variants such as `519` vs
`519L`) and keyed by (airport, stand) since stand names repeat across airports.
Rejected reading: clip predictions with the bounds unvalidated — a 1 % wrong
hard bound on a heavy-tailed target can cost more than the bound wins.
