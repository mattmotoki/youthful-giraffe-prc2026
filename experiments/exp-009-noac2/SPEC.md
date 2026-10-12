# exp-009-noac2 — SPEC

tier: hard | reviewer: sol (2 rounds max) | assignment: desk, 2026-09-05

## Reading of the assignment

Sharpen `p(placeholder)` on the 388 ranking noac rows. The desk's arithmetic:
the error on those rows is `p(1−p)·(mvt_sched − normal_taxi)²`, so with the two
component values essentially fixed, **the whole lever is the sharpness of p** —
pushing it toward 0 or 1, not improving either conditional mean. exp-002 reached
a leave-one-month-out RMSE of 4,289.5 s (ranking contribution 143.9 s) with an
oracle on the placeholder flag at 3,745 s, so the headroom this experiment is
chasing is that ~545 s gap, worth about 18 s of ranking contribution.

Readings rejected:

1. *"Improve the conditional means."* Rejected by the desk's own decomposition
   and by exp-002's oracle: with a perfect flag the RMSE is 3,745 s, so
   everything above that is `p`-sharpness and nothing else is worth touching.
2. *"Bigger model."* Rejected: 1,507 rows. exp-002's sweep found the optimum at
   8 leaves and 300 trees, and 31 leaves was 390 s worse. The lever is
   information, not capacity.
3. *"Treat the arrival-side signal as given."* Tested first because it was the
   assignment's sharpest idea and the cheapest thing that could kill the lane.
   It is refuted — see `work/arr_evidence.py` and RESULTS finding 1. Recording
   this as a reading I rejected *after* measuring, not before.

## Frozen conventions

- LOMO over all 1,507 training noac rows, seed-averaged, one month held out at
  a time; the reported number is pooled RMSE over all 1,507 predictions plus the
  per-month breakdown the desk asked for.
- Ranking contribution = `sqrt(388 · MSE / 344,841)`, the same formula as
  exp-002 so the two are comparable.
- Delivery is a VERSION bump of `prc.tail_model` with the exp-002 head reachable
  by flag, so a regression is one argument away from being undone.
