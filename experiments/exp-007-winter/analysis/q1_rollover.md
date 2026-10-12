# exp-007 question 1c — the two unreachable rows and what they do to the january fold

Screen (stated, not fitted): `TAXITIME_SEC_mvt > 21600 (physically impossible), and the observable columns ordinary: aircraft type present and |MVT - SCHED| <= 3600 s`, i.e. exp-002's date-rollover class
restricted to the rows whose observable columns are ordinary.

| fold | taxi > 6 h (S1) | of those, unreachable (S2) | S2 SSE share of the whole fold | of the non-LIRF slice |
|---|---|---|---|---|
| january | 7 (2 non-LIRF) | 2 (2 non-LIRF) | **55.3 %** | **61.2 %** |
| december | 4 (0 non-LIRF) | 0 (0 non-LIRF) | **0.0 %** | **0.0 %** |
| july | 17 (0 non-LIRF) | 0 (0 non-LIRF) | **0.0 %** | **0.0 %** |

## The rows themselves

| fold | airport | flight | take-off | hidden block | schedule | y (s) | prediction | no-NM | noac | MVT−SCHED (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| january | LFPG | EJU983W | 2025-01-19 18:54 | 2025-01-18 19:30 | 2025-01-19 18:25 | 84,240 | 905 | True | False | 1,740 |
| january | LFPG | EJU42AY | 2025-01-23 15:09 | 2025-01-22 22:58 | 2025-01-23 14:35 | 58,206 | 1,136 | True | False | 2,043 |

## Every fold number, with the class in and out

| fold | set | rows | RMSE all | RMSE non-LIRF | non-LIRF bulk | non-LIRF tail | non-LIRF no-NM | RMSE LIRF |
|---|---|---|---|---|---|---|---|---|
| january | as scored | 153,706 | 346.5 | **342.3** | 193.9 | 1316.7 | 2830.3 | 395.6 |
| january | rollover removed | 153,704 | 231.7 | **213.3** | 193.9 | 1316.7 | 758.9 | 395.6 |
| december | as scored | 165,677 | 219.3 | **201.6** | 188.7 | 1523.1 | 567.7 | 377.9 |
| december | rollover removed | 165,677 | 219.3 | **201.6** | 188.7 | 1523.1 | 567.7 | 377.9 |
| july | as scored | 190,713 | 320.8 | **249.7** | 227.3 | 1892.2 | 588.0 | 755.5 |
| july | rollover removed | 190,713 | 320.8 | **249.7** | 227.3 | 1892.2 | 588.0 | 755.5 |

## The board blend of exp-006, recomputed

| set | other nine (jul/jan blend) | LIRF (jul/jan blend) | implied total |
|---|---|---|---|
| as scored | **295** | 633 | 334 |
| rollover removed | **234** | 633 | 286 |

(The other-nine blend uses the other nine airports' OWN ranking mix, 55.4 % july / 44.6 % january — 176,284 July and 141,658 January non-LIRF ranking departures. exp-006's 58.9/41.1 is LIRF's own mix and is kept for the LIRF column. The implied total uses exp-006's 7.8 % LIRF share, 26,899 of 344,841.)
