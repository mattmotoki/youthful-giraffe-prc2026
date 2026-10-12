# exp-010-nmtail — SPEC

## My reading of the assignment (and the readings I rejected)

**The segment.** The NM-joined tail is `AOBT_3_flt` present and
`TAXITIME_SEC_mvt > 3600`. It is defined by the hidden label, so it is a
*diagnostic* population, not a routable one: no model can select it at
prediction time. Everything I build therefore acts on **all NM rows** and is
scored both on the NM tail (to see the mechanism) and on the whole fold (to see
what it costs the rows it was not aimed at). Rejected reading: build a router
that fires on the tail, which cannot be deployed.

**The mechanism, stated as one equation.** With `nm_taxi = MVT - AOBT_3` and
`mvt_lobt = MVT - LOBT` both observable, and `phi = (BLOCK - AOBT) / (LOBT -
AOBT)` the lirf worker's coefficient,

    taxi = nm_taxi - phi * (LOBT - AOBT).

`phi = 0` is block-on-AOBT, `phi = 1` is block-on-LOBT. Under squared error the
optimal prediction is `nm_taxi - E[phi | observables] * (LOBT - AOBT)`, so the
quantity to estimate is a *conditional mean of phi*, and the desk's
`p(block on LOBT)` is the two-point approximation `E[phi] ~ p*1 + (1-p)*0`. I
build both — a calibrated classifier for `p` and a direct regression for
`E[phi]` — and let the fold RMSE choose, because the two-point form is only
optimal if phi really is concentrated on {0, 1} in the rows that matter. That
is a measurement, not an assumption.

**Regime versus magnitude.** "How much is misclassification versus magnitude" is
answered with an oracle: for each NM-tail row the *true* anchor is whichever of
`nm_taxi` and `mvt_lobt` is closer to the label. An oracle-anchor predictor's
RMSE is the magnitude error that survives a perfect regime call; the gap between
it and the model is what the regime call costs. Rejected reading: define the
regime by a fixed threshold on `AOBT - LOBT` and call disagreement with it
misclassification — that measures the threshold, not the model.

**Validation.** Leave-one-month-out over all twelve months for the `p`/`phi`
estimator (the desk's instruction), and the four folds `july`, `july_loo`,
`december`, `january` for every scored row, with the four COMMON.md slices plus
the NM-tail slice. Group statistics are fitted inside the training months of
each fold; the LOO estimator never sees the month it scores.

**Scope.** A dedicated NM-tail head is tested against the residual-guarded
linear-leaf line and against `prc.tail_model`'s router (question 3). I do not
build a submission candidate.

## Deliverables
- `prc/nmtail_features.py` (or an extension of an existing module) with VERSION,
  fit/transform separated
- `experiments/exp-010-nmtail/RESULTS.md`, `climb.log`, OOF, ledger rows
- at most two Sol rounds
