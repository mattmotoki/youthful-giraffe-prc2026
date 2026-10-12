# exp-016 stack — SPEC

## My reading of the assignment, and the readings I rejected

**The reading I took.** The question is whether the mixing weights should be a
**function of the row** rather than a constant over a population. The blend owner
has measured constant-global (fitted NNLS) and constant-per-population
(per-regime) and found flat weights beat the first and the second adds little;
the untested thing is a weight that moves with the row's own features. So the
deliverable is a *rotated* measurement of three row-dependent parameterisations
against two constant-weight baselines, with a day-clustered interval on every
difference, and a candidate only if the gain survives the rotation.

**The four-rotation instruction cannot be honoured, and this is the one place I
depart from the assignment.** The desk asked for "all four rotations". `prc/data.py`
defines six folds (`may`, `july`, `december`, `january`, `july_loo`,
`november_loo`), four of which the member set could plausibly use — but
`july` and `july_loo` **validate the same month**: both score 2025-07, the same
190,713 departures with the same labels. They differ only in how many months
their members were trained on (six against eleven). A rotation that fits a
stacker on a set containing one and scores it on the other therefore fits and
scores on **the same rows**, which is not a weaker measurement but a leak: the
stacker would see July 2025's labels in training and be scored on July 2025.

Two further facts settle it independently of that argument: `C2-cat`, `L2-hub`
and `N1-nmtail` have **no `july` OOF** at their pinned addresses, so a fourth
rotation could not be built from the named member set anyway; and exp-012 dropped
the forward `july` fold for its own reasons. **This experiment therefore runs
three rotations — `july_loo`, `december`, `january` — which is also what the
blend owner's board estimate is built from, so the numbers are comparable to
318.02 without conversion.** Recorded here rather than asked, per the protocol:
it is a validation-design call, not a stopping or kind-of-work call.

**Readings I rejected.**

1. *"Fit the stacker on the members' ranking predictions and validate on the
   folds."* Rejected: the stacker would then be fitted on rows with no labels.
   Everything here is fitted on OOF only, and the rotation is the rule.
2. *"Give the LightGBM stacker the whole v2 frame as gating features."* Rejected:
   two folds of OOF is the entire fitting set for each rotation, and a stacker
   with the full frame is a second base model, not a gate. The gate set is the
   desk's list plus a handful of congestion columns already in the cached frame,
   and it is frozen before any number is read.
3. *"Report the in-sample gain as well."* Rejected as a headline. exp-012
   established that in-sample NNLS is the most weight a member can earn and that
   the honest direction is the negative one; an in-sample stacker gain is
   guaranteed and says nothing. In-sample appears only as an upper bound where a
   rotated number is ambiguous.
4. *"Use `prc.blend.build_candidate` for the submission."* Rejected: it applies
   one weight vector per regime and cannot express a row-dependent weight. The
   candidate path here is its own, and it reuses `build_candidate`'s member
   preparation and `prc.data.write_submission`'s checks so the artifact is
   verified the same way.

## Frozen before any number was computed

- **Gate features** (`members.GATE_NUM`, `GATE_CAT` plus `ds_deice_hr`, `noac`,
  `nm_missing`, `regime`): the desk's list — regime flags, airport, hour,
  AOBT−LOBT band, noac and no-NM flags, take-off minus AOBT, `ds_deice_hr` — plus
  `d_takeoff_lobt`, `d_takeoff_sched`, `local_minute_of_day`, `queue_ahead_aobt`,
  `dep_takeoffs_30m_before`, `arr_landings_30m_before`, `flight_len`, all of
  which are already in the cached v2 frame and cost nothing.
- **Adoption rule**: a variant is a candidate only if its rotated RMSE beats the
  shipped `blend2` rule on **every one of the three folds** and its board
  estimate improves. A variant that wins on average but loses a fold is recorded
  as a MAYBE with its revival condition, because three folds cannot support
  choosing a winner on a mean.
- **Interval**: day-clustered bootstrap (airport × calendar day), 2,000 draws, on
  the RMSE difference. A row bootstrap would understate it.
- `phi` is **not** in the first gate set. It needs a per-fold `PhiModel` fit,
  which is the only feature on the desk's list that is not already cached; it
  goes in only if the first cut shows a stacker worth extending, which is the
  cheapest order.
  **Status at the end of the experiment: the condition was met and phi was still
  not run.** It is carried as an open item in RESULTS.md, not a close. Recorded
  here so the deferral cannot read as a decision that was taken.

## Leakage discipline

- Members are pinned by sha256 in `pins.json` and re-verified on every run; the
  run aborts if any member's bytes moved.
- A fold's stacker never sees that fold's OOF, its labels, or its rows.
- Gates come from exp-008's frozen per-month v2 frames (built from the masked
  month alone, mask invariance verified there) and from `ds_deice_hr`, which
  `prc.day_state_features` computes from the frame itself with no fitted
  reference, so it does not depend on which months `DayState` saw.
- Members are floored and residual-guarded on the **fold's own** training-month
  bounds before stacking, so the stacker is fitted on the estimator that is
  served.

## The mismatch this design carries, stated up front

The stacker is fitted on OOF predictions from members trained on eleven months
and applied at serve time to ranking predictions from members trained on twelve.
The served members are *better* than the ones the stacker learned to correct. A
stacker that has learned "member X is unreliable here" may therefore over-correct
a member that is no longer unreliable. This is the standard stacking mismatch,
it is not removable without a twelve-month OOF that does not exist, and it is the
first thing to suspect if a rotated gain does not appear on the board.

## Threads and compute

12 threads, no GPU. Box was at load 400 at start; the whole rotation is minutes.
