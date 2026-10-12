# exp-007 question 4 — is the under-prediction of long taxis shrinkage?

Regressing the truth on the prediction: a slope above 1 says the predictions are
compressed toward the conditional mean, which is the signature the question names.
Measured on the baseline out-of-fold line with the two unreachable rows removed.

| fold | slice | rows | RMSE | calibration slope | intercept | RMSE after an affine recalibration (odd/even day) |
|---|---|---|---|---|---|---|
| january | core, other nine | 141,017 | 200.9 | **1.012** | -16 s | 201.0 |
| january |   of which target ≤ 3600 s | 140,786 | 193.9 | **0.995** | -1 s | 193.9 |
| january |   of which target > 1800 s | 6,364 | 537.2 | **0.754** | +810 s | 410.1 |
| january |   of which span > 1800 s | 5,824 | 415.8 | **1.071** | -96 s | 413.5 |
| january |   of which on the worst 5 % of days | 8,663 | 350.1 | **1.093** | -105 s | 346.6 |
| december | core, other nine | 152,054 | 194.3 | **0.993** | +1 s | 194.2 |
| december |   of which target ≤ 3600 s | 151,912 | 188.7 | **0.981** | +11 s | 188.5 |
| december |   of which target > 1800 s | 5,748 | 539.2 | **0.632** | +1005 s | 409.8 |
| december |   of which span > 1800 s | 5,049 | 374.8 | **1.028** | -35 s | 375.6 |
| december |   of which on the worst 5 % of days | 7,938 | 376.6 | **0.968** | +57 s | 395.6 |
| july | core, other nine | 171,905 | 237.5 | **1.022** | -31 s | 237.2 |
| july |   of which target ≤ 3600 s | 171,673 | 227.3 | **0.997** | -8 s | 227.0 |
| july |   of which target > 1800 s | 6,225 | 752.8 | **0.691** | +1060 s | 513.1 |
| july |   of which span > 1800 s | 4,112 | 400.8 | **1.178** | -405 s | 381.8 |
| july |   of which on the worst 5 % of days | 10,026 | 392.4 | **1.059** | -31 s | 393.2 |

## What the slopes say

A slope of 1.000 is a calibrated predictor. The table above is the test; the reading is written in RESULTS.md finding 5.

## The day effect: how much would a per-airport-day intercept be worth?

An oracle that added each airport-day's own mean residual to every row of that day. This is the ceiling on any day-severity second stage, including a random effect; it uses the validation month's labels and is not a model. The leave-one-out column removes the row's own residual from the day mean it is then corrected by, which is the honest version -- with ~500 rows a day the two barely differ, and the point of showing both is that the optimistic one is already small.

| fold | slice | rows | RMSE | after an oracle day intercept | the same, leave-one-out | days |
|---|---|---|---|---|---|---|
| january | core, other nine | 141,017 | 200.9 | **198.5** | 198.8 | 279 |
| january | bulk ≤3600 | 140,786 | 193.9 | **191.6** | 192.0 | 279 |
| december | core, other nine | 152,054 | 194.3 | **192.7** | 193.0 | 279 |
| december | bulk ≤3600 | 151,912 | 188.7 | **187.1** | 187.5 | 279 |
| july | core, other nine | 171,905 | 237.5 | **235.1** | 235.5 | 279 |
| july | bulk ≤3600 | 171,673 | 227.3 | **224.7** | 225.0 | 279 |

And the honest version of the same thing: the day intercept fitted on the fold's *training* months cannot exist (there are no out-of-fold residuals for a training month), so the only implementable second stage predicts the day effect from label-free day state. `analysis/q2_daystate.md` measured how well those columns rank days; the ceiling above says what a perfect ranking would be worth.
