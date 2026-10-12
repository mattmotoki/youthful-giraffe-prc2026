# Q2(b) — how much of the separability is real shift

A classifier that separates the validation month from the training months at AUC 0.98 is only measuring covariate shift if the features it uses are features, not clocks. The model's set contains a day-of-year and three month-level counts; dropping them one group at a time:

| fold | classifier features | n | AUC | w p01 | w p50 | w p99 | effective sample size % |
|---|---|---|---|---|---|---|---|
| july_loo | all model features | 64 | 1.000 | 1.000 | 1.000 | 1.000 | 100.000 |
| july_loo | minus `doy` | 63 | 0.985 | 0.084 | 0.483 | 6.946 | 32.049 |
| july_loo | minus `doy` and the month-level counts | 61 | 0.881 | 0.056 | 0.625 | 5.322 | 43.323 |
| july_loo | minus those and the neighbour block | 55 | 0.853 | 0.056 | 0.758 | 4.428 | 53.040 |
| december | all model features | 64 | 1.000 | 1.000 | 1.000 | 1.000 | 100.000 |
| december | minus `doy` | 63 | 0.991 | 0.082 | 0.632 | 5.757 | 39.765 |
| december | minus `doy` and the month-level counts | 61 | 0.902 | 0.056 | 0.743 | 4.806 | 48.174 |
| december | minus those and the neighbour block | 55 | 0.879 | 0.055 | 0.780 | 4.450 | 51.792 |
