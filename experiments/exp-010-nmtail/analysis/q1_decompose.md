# Q1 — where the NM-joined tail error lives

NM tail = `AOBT_3_flt` present and `TAXITIME_SEC_mvt > 3600`. `taxi = nm_taxi - phi * gap` with `gap = LOBT - AOBT` and `phi = 0` block-on-AOBT, `phi = 1` block-on-LOBT.

## The segment

| month | departures | NM rows | NM tail | NM tail % | share block-on-LOBT | median phi (|gap|>300) | frac |phi-1|<0.1 |
|---|---|---|---|---|---|---|---|
| 2025-01 | 153706 | 152271 | 272 | 0.18 | 64.34 | 0.73 | 17.36 |
| 2025-02 | 143731 | 142622 | 522 | 0.36 | 64.94 | 0.84 | 17.30 |
| 2025-03 | 164450 | 163367 | 96 | 0.06 | 87.50 | 1.10 | 37.66 |
| 2025-04 | 175288 | 174203 | 124 | 0.07 | 82.26 | 1.11 | 22.12 |
| 2025-05 | 185202 | 183597 | 160 | 0.09 | 93.12 | 1.00 | 34.48 |
| 2025-06 | 183142 | 180640 | 292 | 0.16 | 92.12 | 1.03 | 29.21 |
| 2025-07 | 190713 | 186775 | 503 | 0.26 | 93.64 | 1.00 | 34.10 |
| 2025-08 | 191181 | 188852 | 240 | 0.13 | 93.75 | 1.08 | 34.36 |
| 2025-09 | 183951 | 181485 | 224 | 0.12 | 90.62 | 1.02 | 33.00 |
| 2025-10 | 185674 | 183761 | 247 | 0.13 | 88.66 | 1.10 | 27.06 |
| 2025-11 | 162332 | 160976 | 169 | 0.10 | 70.41 | 0.83 | 19.72 |
| 2025-12 | 165677 | 164028 | 190 | 0.11 | 66.84 | 0.79 | 13.66 |

## Model error on the NM tail, against the two anchors and the oracle

`oracle anchor` picks, per row, whichever of `nm_taxi` and `mvt_lobt` is closer to the label: its RMSE is the magnitude error that survives a perfect regime call. `model implied anchor` is whichever of the two the model's own prediction is closer to.

| fold | model | n | RMSE model | RMSE always-AOBT | RMSE always-LOBT | RMSE oracle anchor | anchor call correct % | RMSE where correct | RMSE where wrong | SSE share of wrong % |
|---|---|---|---|---|---|---|---|---|---|---|
| december | prc.tail_model router | 190 | 1882.8 | 3018.4 | 1940.4 | 1717.0 | 67.4 | 1715.9 | 2187.6 | 44.0 |
| january | prc.tail_model router | 272 | 1423.3 | 2753.5 | 1588.5 | 1270.6 | 59.9 | 1256.9 | 1641.1 | 53.3 |
| july | prc.tail_model router | 503 | 2010.4 | 3960.1 | 1562.4 | 1474.8 | 67.4 | 1545.4 | 2731.1 | 60.2 |
| july_loo | prc.tail_model router | 503 | 1927.7 | 3960.1 | 1562.4 | 1474.8 | 69.4 | 1510.6 | 2639.5 | 57.4 |
| july_loo | retro-v1 A0 (guarded, plain leaf) | 503 | 1869.6 | 3960.1 | 1562.4 | 1474.8 | 73.2 | 1534.9 | 2569.4 | 50.7 |
| december | retro-v1 R0 (guarded, linear leaf) | 190 | 1841.7 | 3018.4 | 1940.4 | 1717.0 | 65.3 | 1753.8 | 1996.2 | 40.8 |
| january | retro-v1 R0 (guarded, linear leaf) | 272 | 1423.3 | 2753.5 | 1588.5 | 1270.6 | 62.9 | 1231.3 | 1699.6 | 52.9 |
| july_loo | retro-v1 R0 (guarded, linear leaf) | 503 | 1904.5 | 3960.1 | 1562.4 | 1474.8 | 72.4 | 1537.9 | 2632.9 | 52.8 |
