# Leave-one-month-out over all twelve 2025 months

2,062,577 NM-joined departures, each scored by a model fitted on the other eleven months (`code/loo.py`, 400 rounds, lr 0.05, 63 leaves).

## What each parameterisation is worth

| predictor | RMSE all NM | RMSE NM tail | RMSE NM bulk |
|---|---|---|---|
| phi regression (weight gap^2) | 237.3 | 1971.2 | 225.1 |
| p classifier (two-point) | 272.2 | 2206.7 | 258.9 |
| always block-on-AOBT | 384.9 | 3447.8 | 361.6 |
| always block-on-LOBT | 740.1 | 1824.6 | 737.4 |

## Calibration of `p`, on the rows where the two anchors differ (|gap| > 600 s)

| p_hat bin | n | mean p_hat | observed block-on-LOBT | mean phi_hat | observed mean phi | P(taxi>3600) % |
|---|---|---|---|---|---|---|
| (-0.001, 0.05] | 297311 | 0.020 | 0.015 | 0.003 | -0.003 | 0.049 |
| (0.05, 0.1] | 123000 | 0.072 | 0.070 | 0.126 | 0.118 | 0.137 |
| (0.1, 0.2] | 98480 | 0.141 | 0.146 | 0.217 | 0.218 | 0.274 |
| (0.2, 0.3] | 35652 | 0.243 | 0.254 | 0.311 | 0.324 | 0.516 |
| (0.3, 0.5] | 33972 | 0.390 | 0.400 | 0.418 | 0.439 | 0.915 |
| (0.5, 0.7] | 23910 | 0.596 | 0.608 | 0.551 | 0.582 | 1.514 |
| (0.7, 0.9] | 29162 | 0.811 | 0.820 | 0.697 | 0.723 | 1.667 |
| (0.9, 1.0] | 24270 | 0.949 | 0.962 | 0.902 | 0.931 | 2.217 |

AUC of `p_hat` for block-on-LOBT on those rows: **0.910**; Brier score 0.0752 against 0.1399 for the base rate.

AUC of `p_hat` for `taxi > 3600 s` on the same rows: **0.837**; of `phi_hat`: **0.816**.

## By month

| month | NM rows | NM tail | RMSE phi, tail | RMSE p, tail | RMSE phi, all NM |
|---|---|---|---|---|---|
| 2025-01 | 152271 | 272 | 1518.0 | 1677.8 | 231.3 |
| 2025-02 | 142622 | 522 | 1934.3 | 1951.7 | 246.4 |
| 2025-03 | 163367 | 96 | 2207.1 | 2561.1 | 217.9 |
| 2025-04 | 174203 | 124 | 2216.0 | 2467.7 | 217.5 |
| 2025-05 | 183597 | 160 | 1931.9 | 2276.5 | 227.6 |
| 2025-06 | 180640 | 292 | 2213.1 | 2493.9 | 244.0 |
| 2025-07 | 186775 | 503 | 1860.0 | 2193.6 | 272.1 |
| 2025-08 | 188852 | 240 | 2014.8 | 2404.0 | 246.4 |
| 2025-09 | 181485 | 224 | 2084.0 | 2327.8 | 242.3 |
| 2025-10 | 183761 | 247 | 2180.7 | 2439.5 | 239.8 |
| 2025-11 | 160976 | 169 | 1774.6 | 1969.0 | 223.3 |
| 2025-12 | 164028 | 190 | 1961.2 | 2137.7 | 228.5 |