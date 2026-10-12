# exp-148 SPEC revision 3: served variants under the R2 condition (after the board read of Sb-r1)

Amends SPEC-r2.md (sha256 c1f68e7d…) sections 5 and 6; SPEC.md, SPEC-r1.md and the rest of
SPEC-r2.md stay in force. Written 2026-09-30 about 13:45 UTC, before any R2 stacker fit and
before any R2 score (no `R2-*` artifact exists). Assignment: the desk's message after the upload
of v34 (= `ranking-pred_Sb-r1`, 243.5329 against v32's 242.7480, scenario C): repair 2 is the
main line and its month-hold level-2 condition is the selection condition for the next stacked
candidate; under that condition, and chosen before any upload, score Sb with v32 kept on the
noac rows, Sb with v32 on the ordinary rows without AOBT, and fixed blends of Sb with the anchor
route. No board-derived weight.

## Reading

"Sb" is the R2 form that SPEC-r2 section 3 chooses from the training-month CV (called `R2f`
below; it is R2b if the mean of the two stackers wins). "The anchor route" is the anchor of the
same member fits (the mean of the five base members), served as max(a, 60 s): `anchor_served`
of the R2 artifacts. "Chosen before any upload": the variant set, the weights and the selection
rule are fixed here; the R2 scores of January and July 2025 choose among them; no board read
chooses anything.

Readings rejected:
(a) Apply the variants to S-final (the served Sb): its local condition is the week-pair view,
    which the board has just shown does not transfer. The whole-month-check Sb (owners m01, m07:
    a level 2 without labels of the scored month) gets the same variants as a reference row: it
    is the local analog of v34 and had the board's sign. It is not selectable.
(b) Fit the blend weight on the local rows (a continuous optimum): a fit on the rows that score
    it. A fixed grid with a check on the month that did not choose shows the same trade-off with
    less selection.
(c) Blends with v32's vector (REPORT hedge 3): not asked.

## 1. Variants (16)

For the R2f vector and the anchor route of the same member fits (local rows: the `janjul`
members; ranking rows: the `final` members):

    blend(w) = w * R2f + (1 - w) * anchor_served,   w in {1, 0.75, 0.5, 0.25}

and a keep set K in {none, noac, ordinary without AOBT, both}: on the rows of K, v32's value
replaces blend(w). Names `R2f-w100-k0`, `-kn` (noac), `-ko` (ordinary without AOBT), `-kno`
(both); w100, w75, w50, w25.

Row sets. noac: class noac. Ordinary without AOBT: class ordinary and `AOBT_3_flt` missing in
the raw row. Local rows: the frozen rows (`gscore.load_rows`: `cls`, `aobt`; 402 noac rows,
3,709 ordinary rows without AOBT). Ranking rows: the class of `features.parquet` (the v25 class
export) and `AOBT_3_flt` of `ranking.parquet` (388 noac rows; REPORT counts 3,205 ordinary rows
without AOBT; `serve_r2.py` records its own count).

v32's value. Local rows: the saved v32 month-fold route (`refs.parquet`). VALIDATION section 3
does not admit it as an input (label paths T1, L1-L7); the local G of a keep variant is
therefore a reference, optimistic by whatever those label paths give on the kept rows. Ranking
rows: v32's served vector. The unsupported cells keep v32 in every variant (as `rankvec.py`).

## 2. Scores and selection

Every variant: G against v32's saved route, January and July 2025 joined (344,419 rows; all
rows and clean; frozen interval; the month, class and class x AOBT views), and G against
`R2f-w100-k0`.

**Selection (fixed now).** The candidate is the variant with the highest joined G against v32's
saved route (all rows). A difference under 1 s² per row goes to the simpler variant (larger w
first, then the smaller keep set). Check on data that did not choose it: the variant with the
highest January contribution (the `month` view) and its July contribution, and the reverse;
both reported next to the candidate. No gate: SPEC-r2 section 6 stands (R2 is a submission
candidate if the chosen variant's joined G > 0; the desk decides any upload).

**Reference row, not selectable.** The same 16 variants on the whole-month-check Sb (`Sb_month`)
with the `janjul` anchor. Its w100-k0 row is the local analog of v34 (local −3,873 s² against
the board's −382 s² per row).

## 3. Ranking vector and forecast

`code/serve_r2.py`: R2-final (owner rfinal, `pilot.load`) in the chosen form, blended with its
`anchor_served` at the chosen w, v32's served vector on the chosen K and on the unsupported
cells; w and K read from `score-r2.json`; template order, finite, at least 60 s; write-once
`ranking-<variant>.parquet` and `.json`. The R2 report writes the frozen forecast before any
upload, calibrated with the board read of v34 against its local month-check row.
