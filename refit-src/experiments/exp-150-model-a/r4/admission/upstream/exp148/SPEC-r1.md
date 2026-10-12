# exp-148 SPEC revision 1 (amends SPEC.md sha256 f0e19c6a…; made before any stacker fit)

Reason: the desk's scout priors of 2026-09-30 (`docs/scout/2026-09-30.md` items 2-3): (1) two
teams near the top report that a second tree family on the same inputs, averaged, helped more
than new features (one other team saw a loss); (2) January ↔ July whole-month checks tracked
the board in direction, not in size. No stacker had been fitted and no G read when this
revision was written. Everything in SPEC.md stays; this file adds three items.

1. **Second family.** Each stacker run also fits CatBoost (`CAT_PARAMS` in `code/stacker.py`:
   RMSE, depth 8, learning rate 0.05, l2 3, 254 borders, seed 601, CPU) on the same rows,
   target and feature list, with its rounds chosen by the same leave-one-pair-out early
   stopping. Outputs per row: `pred_S` (LightGBM), `pred_Scat` (CatBoost), `pred_Sb` (their
   mean), each served as max(a + f, 60 s). **Choice of the candidate form** (fixed now, read
   from inner predictions only): the form among LightGBM, CatBoost and the mean with the
   lowest `cv_pooled` MSE averaged over the five outer owners. The other forms are reported.
2. **CV groups** are the week pairs of the training rows (`weekpair.pair_of`); for an outer
   owner this equals the grouping by inner split in SPEC.md.
3. **Whole-month check** (a check, never a selection input). Each member gets a 17th fit,
   `janjul` (`pilot.janjul()`: excludes every day of 2025-01 and 2025-07 and the month-hold
   buffers, scores both months). Stacker owner `m01` trains on the July rows of that fit and
   scores January; `m07` the reverse (consumers `weekpair.month_hold`, which admit `janjul`).
   Reported: G of the joined January + July vector against the anchor of the same fits, and
   against the saved v22 and v32 route vectors (month-hold vectors, reference only), next to
   the week-pair G of the same comparison.
