# Q1c — does the seasonal drift of the offset matter?

## Range of the trimmed-mean offset across the twelve 2025 months, per airport

| airport | min | max | range | argmin month | argmax month |
|---|---|---|---|---|---|
| LIRF | -218 | -50 | 168 | 2025-07 | 2025-11 |
| LTFM | 183 | 342 | 159 | 2025-07 | 2025-03 |
| EDDM | -110 | 14 | 124 | 2025-12 | 2025-03 |
| LEBL | -76 | 2 | 78 | 2025-07 | 2025-01 |
| LEMD | -35 | 30 | 65 | 2025-08 | 2025-01 |
| LFPG | -52 | -2 | 50 | 2025-07 | 2025-03 |
| EDDF | 59 | 100 | 41 | 2025-09 | 2025-03 |
| EGLL | 26 | 66 | 40 | 2025-06 | 2025-01 |
| EHAM | 102 | 134 | 32 | 2025-12 | 2025-07 |
| LSZH | -59 | -33 | 26 | 2025-12 | 2025-10 |

## Which training window predicts the validation month best?

Predictor: `taxi = (MVT - AOBT_3) - offset`, offset = the median of `BLOCK - AOBT_3` over the named window, per airport and per stand (stand falling back to airport, minimum 30 rows). Bulk RMSE.

| fold | window | months | airport offset | stand offset |
|---|---|---|---|---|
| july | all training months | 6 | 409.12 | 401.31 |
| july | last 3 training months | 3 | 406.95 | 397.71 |
| july | last 1 training month | 1 | 403.64 | 397.22 |
| july | first 3 training months | 3 | 414.77 | 407.47 |
| december | all training months | 11 | 343.93 | 333.04 |
| december | last 3 training months | 3 | 343.97 | 332.84 |
| december | last 1 training month | 1 | 345.73 | 337.31 |
| december | first 3 training months | 3 | 347.18 | 338.98 |
| january | all training months | 11 | 336.38 | 325.82 |
| january | last 3 training months | 3 | 336.26 | 326.29 |
| january | last 1 training month | 1 | 335.43 | 328.25 |
| january | first 3 training months | 3 | 336.67 | 327.70 |
