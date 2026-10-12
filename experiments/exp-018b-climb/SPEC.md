# exp-018b — climbing the fresh lineage

**Reading of the assignment.** exp-018's lineage is now the repo's best single
model and the diet is lifted, so this experiment keeps climbing it in the
desk's order: (1) seed-bag to three and re-select the round budget by the
multi-month rule; (2) adopt what the repo measured and exp-018 lacks — phi,
the surface families, the deice column, the four-row training drop — each
measured on the four folds with a day-clustered paired bootstrap, adopting
what pays; (3) exp-018's own remaining ideas in COMPARISON order. Every
improved OOF and candidate is published under a versioned name with a hash and
handed to the blend owner directly.

Readings I rejected: (a) that "adopt what the repo measured" means adopt it
unconditionally — the desk said *measure each, adopt what pays*, and exp-003
already found phi negative on its own bulk, so a null here is a real outcome
and gets recorded as one; (b) that the surface families should be added whole
— exp-018 already has label-free equivalents of about a dozen of those columns,
and adding both copies would measure nothing, so `code/extras.py::SKIP` names
the overlap and the arm adds only the rest; (c) that the budget curve needs
CatBoost at every checkpoint — the budget being chosen is the LightGBM round
count, and the extra four hours would not move a number the 200-round
checkpoint grid can resolve.

## What review round 1 of exp-018 forces on this experiment

The first lineage's blocking finding was that its configuration was selected on
the folds it reported. This one is built so that cannot recur:

* **The budget is chosen on months nothing here reports.** February, May and
  August, each as a leave-one-month-out fold, none of them a scored fold.
* **Every arm is scored on all four folds and judged by a day-clustered paired
  bootstrap**, not by a fold-level delta. The repo's own threshold is that
  single-fold deltas under about 3 s are not measurable at one seed; the
  bootstrap is what replaces that eyeball.
* **Three seeds throughout**, so a 1.5 s adoption decision is not a seed draw.
* **`rmse` and `rmse_clean` are reported together**, always, per the exp-020
  convention.

## The four modules, and what each is expected to do here

| module | what it is | why exp-018 lacks it | expectation |
|---|---|---|---|
| `phi` | `BLOCK = AOBT + phi*(LOBT - AOBT)`, per airport×operator, cross-fitted (`PhiCV(PhiModel(weight="rows"))`) | exp-018 screened `LOBT` as a *population-wide* anchor, found it worth ~nothing, and dropped it. The repo's is regime-conditional, where it is worth 1,562 s against 3,960 s. This is exp-018's clearest miss | the largest of the four |
| `surface` | `prc.surface_features` VERSION 2, seven families, minus the ~15 columns exp-018 already covers | exp-018 built its own label-free congestion set and never saw theirs | small; the overlap is real |
| `deice` | `prc.day_state_features.DEICE` (`ds_deice_hr`) | exp-018 used no external or day-state signal | january only |
| `drop` | `prc.data.keep_reachable`: four training rows no input can reach | exp-018 trained on them | −1.6 s on july_loo per exp-017, from four rows |

## Compute

Six threads, one fit at a time, launch gate at a 1-minute load of 140, GPU 0
for CatBoost. Module columns are built once per fold and cached under
`artifacts/exp-018b/extras/`, so an ablation arm costs one fit and not one
feature build.
