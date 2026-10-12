# exp-007 — the EJU rule priced as a detector

gpt-5.6-sol round 1, blocking finding 1: my claim that no observable column
separates the two unreachable rows is false. The conjunction

    ADEP = LFPG  and  no NM join  and  aircraft type present
    and  FLIGHT_mvt starts with 'EJU'  and  |take-off - schedule| <= 1 h

selects exactly `EJU983W` and `EJU42AY` among January 2025's LFPG no-NM rows.
That disproves "by construction unreachable". The desk's question is the next
one: a conjunction that isolates two rows post hoc is not yet a detector, so
measure its precision out of sample and price the hedge it would license.

The hedge: on a firing row, add `p * R` seconds, where `p` is the rule's
precision estimated on the fold's TRAINING months and `R` is the mean residual of
an event. That is the RMSE-optimal additive correction under the rule's own
model. A firing row that is not an event costs `(p*R)^2`; one that is recovers
`2*p*R*residual - (p*R)^2`.

## Where the rule fires

| month | rows it fires on | of those, rollover events |
|---|---|---|
| 2025-01 | 2 | 2 |
| 2025-03 | 1 | 0 |
| 2025-05 | 1 | 0 |
| 2025-06 | 1 | 0 |
| 2025-07 | 1 | 0 |
| 2025-08 | 1 | 0 |
| 2025-09 | 2 | 0 |
| 2025-12 | 2 | 0 |
| **2025 total** | **11** | **2** |

In the live ranking file the rule fires on **2** rows, 2 in 2026-07. Their labels are hidden, so the rule cannot be checked there.

## Precision, in sample and out

The rule was found *on* January 2025's two events, so its all-2025 precision of 2/11 = **18.2 %** is in-sample. Removing the month it was found on leaves 0/9 = **0.0 %**: **out of sample the rule has fired 9 times and been right 0 times.**

## What the hedge it licenses does to each fold

`p` is estimated on the fold's own training months, `R` is the mean residual of an event under the baseline line (83,335 and 57,070 s, mean 70,203). Both are then applied to the validation month, which is the only honest order.

| fold | training fires / events | p | hedge p·R | validation fires | of those events | RMSE before | after | Δ |
|---|---|---|---|---|---|---|---|---|
| july | 5 / 2 | 0.400 | 28,081 s | 1 | 0 | 320.78 | 327.25 | **+6.47 s** |
| december | 9 / 2 | 0.222 | 15,601 s | 2 | 0 | 219.26 | 226.76 | **+7.51 s** |
| january | 9 / 0 | 0.000 | 0 s | 2 | 2 | 346.49 | 346.49 | **+0.00 s** |

And the same with the two unreachable rows removed from the validation month, i.e. the hedge's cost when it is right about nothing:

| fold | validation fires (clean) | RMSE before | after | Δ |
|---|---|---|---|---|
| july | 1 | 320.78 | 327.25 | **+6.47 s** |
| december | 2 | 219.26 | 226.76 | **+7.51 s** |
| january | 0 | 231.70 | 231.70 | **+0.00 s** |
