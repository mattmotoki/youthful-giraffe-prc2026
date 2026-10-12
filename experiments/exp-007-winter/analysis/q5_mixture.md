# exp-007 question 5 — a per-flight block-time convention: RAISED, THEN REFUTED

**Status: withdrawn.** This file first reported that the airport's recorded block
time sits either ON the NM actual off-block or ON the filed one, per flight, with
φ as the mixing weight rather than a coefficient, and offered a 19.5 s july
oracle. It was reported to the desk for the tail worker and **retracted ninety
minutes later, before it was routed**. The section "The check that refuted the
mixture" is the refutation; RESULTS.md finding 8 is the current view. Everything
between here and there is kept as the record of what was claimed.

## Definitions

```
r    = BLOCK_TIME_UTC_mvt - AOBT_3_flt        (label-derived, the unobservable)
lob  = LOBT_flt - AOBT_3_flt                  (label-free)
component = "AOBT"  if |r|       <= 60 s
            "LOBT"  if |r - lob| <= 60 s
            "other" otherwise
```
on NM-joined non-LIRF departures with |lob| >= 300 s.

## The oracle

| fold | rows in scope | AOBT | LOBT | other | oracle on the two clean components |
|---|---|---|---|---|---|
| january | 86,842 | 18.7 % | 4.8 % | 76.5 % | 223.2 → **209.2** s (+14.0) |
| december | 95,323 | 19.2 % | 4.8 % | 76.0 % | 215.4 → **200.5** s (+14.9) |
| july | 118,633 | 19.5 % | 3.9 % | 76.5 % | 261.0 → **241.5** s (+19.5) |

The oracle is exact on the two clean components by construction and touches nothing else. ~~So the gain is the **ceiling** on a perfect per-flight router.~~ **Withdrawn**: the last section shows the two "components" are not point masses, so this is an oracle that already knows `r` on a quarter of the rows, not a ceiling on any router. It is measured on the in-scope slice above, not on the whole fold.

## What marks the component, label-free

Every candidate is fitted on the fold's training months and scored on the validation month, on the in-scope rows whose component is clean (`AOBT` or `LOBT`). AUC is for `component == LOBT`.

| fold | candidate | AUC | n scored |
|---|---|---|---|
| january | `op` rate, shrunk (training months) | **0.742** | 20,372 |
| january | `apt` rate, shrunk (training months) | **0.711** | 20,372 |
| january | `seg` rate, shrunk (training months) | **0.728** | 20,372 |
| january | `wake` rate, shrunk (training months) | **0.721** | 20,372 |
| january | `ftype` rate, shrunk (training months) | **0.719** | 20,372 |
| january | `stand` rate, shrunk (training months) | **0.743** | 20,372 |
| january | `abs_lob` | 0.337 | 20,372 |
| january | `lobt_moved` | 0.500 | 20,372 |
| january | `eobt_moved` | 0.508 | 20,372 |
| january | `hour` | 0.452 | 20,372 |
| january | **router on the airport-operator rate, hard 0.5** | RMSE 253.1 vs model 165.4 | 20,372 |
| january | **router on the airport-operator rate, soft (expected r)** | RMSE 232.1 vs model 165.4 | 20,372 |
| december | `op` rate, shrunk (training months) | **0.755** | 22,882 |
| december | `apt` rate, shrunk (training months) | **0.721** | 22,882 |
| december | `seg` rate, shrunk (training months) | **0.739** | 22,882 |
| december | `wake` rate, shrunk (training months) | **0.735** | 22,882 |
| december | `ftype` rate, shrunk (training months) | **0.730** | 22,882 |
| december | `stand` rate, shrunk (training months) | **0.752** | 22,882 |
| december | `abs_lob` | 0.312 | 22,882 |
| december | `lobt_moved` | 0.500 | 22,882 |
| december | `eobt_moved` | 0.547 | 22,882 |
| december | `hour` | 0.469 | 22,882 |
| december | **router on the airport-operator rate, hard 0.5** | RMSE 273.3 vs model 165.3 | 22,882 |
| december | **router on the airport-operator rate, soft (expected r)** | RMSE 243.9 vs model 165.3 | 22,882 |
| july | `op` rate, shrunk (training months) | **0.732** | 27,835 |
| july | `apt` rate, shrunk (training months) | **0.714** | 27,835 |
| july | `seg` rate, shrunk (training months) | **0.722** | 27,835 |
| july | `wake` rate, shrunk (training months) | **0.720** | 27,835 |
| july | `ftype` rate, shrunk (training months) | **0.709** | 27,835 |
| july | `stand` rate, shrunk (training months) | **0.725** | 27,835 |
| july | `abs_lob` | 0.297 | 27,835 |
| july | `lobt_moved` | 0.502 | 27,835 |
| july | `eobt_moved` | 0.594 | 27,835 |
| july | `hour` | 0.461 | 27,835 |
| july | **router on the airport-operator rate, hard 0.5** | RMSE 324.9 vs model 207.8 | 27,835 |
| july | **router on the airport-operator rate, soft (expected r)** | RMSE 287.9 vs model 207.8 | 27,835 |

## The check that refuted the mixture

### The atoms are not there (2025-07, 118,633 in-scope rows — the oracle's own population: NM-joined, non-LIRF, |lob| >= 300 s)

| tolerance | share with |r| ≤ tol | share with |r − lob| ≤ tol |
|---|---|---|
| 0 s | 0.56 % | 0.17 % |
| 5 s | 6.34 % | 1.87 % |
| 30 s | 9.70 % | 2.22 % |
| 60 s | 19.52 % | 3.94 % |
| 120 s | 36.99 % | 7.18 % |
| 300 s | 71.07 % | 19.26 % |

An atom would put a step at 0 s. Instead the share grows smoothly with the tolerance, and the exact values carry 0.56 % and 0.17 % of the rows. So there are no POINT MASSES at 0 and at `lob`, which is what the withdrawn claim asserted. That is what this test can show and all it can show: a two-component mixture whose components are themselves diffuse would leave no atom either, and is not excluded. What is excluded is the reading that made the oracle look like a ceiling on a router — the rows inside a ±60 s window are the continuum's local density, not a class — and the router built from the best available label-free classifier of that supposed class loses 117 s to the model. φ as a conditional mean stands.

| airport | in-scope rows | exact r = 0 | exact r = lob | \|r\| ≤ 60 s | \|r − lob\| ≤ 60 s |
|---|---|---|---|---|---|
| EDDF | 14,555 | 0.12 % | 0.01 % | 21.83 % | 1.44 % |
| EDDM | 10,505 | 1.01 % | 0.14 % | 21.29 % | 3.20 % |
| EGLL | 13,167 | 0.53 % | 0.15 % | 16.93 % | 4.04 % |
| EHAM | 14,027 | 0.24 % | 0.03 % | 22.59 % | 1.63 % |
| LEBL | 11,750 | 0.97 % | 0.60 % | 20.91 % | 8.17 % |
| LEMD | 12,960 | 1.00 % | 0.12 % | 24.80 % | 2.11 % |
| LFPG | 15,336 | 0.78 % | 0.14 % | 20.27 % | 3.63 % |
| LSZH | 9,611 | 0.18 % | 0.03 % | 21.84 % | 1.23 % |
| LTFM | 16,722 | 0.32 % | 0.28 % | 8.77 % | 8.76 % |
