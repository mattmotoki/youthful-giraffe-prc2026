# Two independent estimates of the same number

`taxi = nm_taxi - offset`. There are two ways to guess it:

- **A (the NM path)**: take the observed `nm_taxi = MVT - AOBT_3` and subtract a group median of the offset. Its error is the unpredictable part of `BLOCK - AOBT_3`.
- **I (the identity path)**: ignore the NM columns and predict the group median taxi time for this stand, runway and aircraft type. Its error is the unpredictable part of the taxi time itself.

If the two errors were independent, the inverse-variance blend would score `1/sqrt(1/sA^2 + 1/sI^2)`. Measured, out of sample, on the bulk of each fold:

| fold | n | RMSE A (NM path) | RMSE I (identity path) | corr(eA,eI) | w on A | inverse-variance blend | independent-blend prediction | least-squares blend |
|---|---|---|---|---|---|---|---|---|
| july | 186272 | 397.061 | 327.692 | 0.691 | 0.391 | 327.731 | 244.378 | 309.121 |
| december | 163838 | 328.783 | 306.588 | 0.622 | 0.458 | 285.572 | 221.825 | 281.400 |
| january | 151999 | 321.148 | 342.039 | 0.548 | 0.526 | 290.880 | 232.817 | 288.670 |
