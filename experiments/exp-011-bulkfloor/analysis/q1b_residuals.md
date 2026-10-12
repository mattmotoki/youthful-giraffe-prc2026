> **Caveat added after review round 1 (B5).** The per-airport tables below place
> the model's RMSE on **100 %** of an airport's bulk rows beside a bound computed
> on the **5 to 14 %** of them that have a near-duplicate. That is not a
> comparison, and it gives the opposite sign to the like-for-like one at nine of
> ten airports. Use `q1e_airport_floor.md`, where both quantities are on the same
> rows; the tables here are kept only for the residual-by-prediction-decile
> panels and for the coverage figures.

# Q1(b) — where the bulk residual sits, and the bound inside the same month

## fold `july_loo` (validation month 2025-07), best OOF `SB-L2hub-july_loo`

186,272 bulk rows, RMSE **234.1**.

### by predicted value

| predicted taxi, decile | n | mean pred | mean actual | bias (pred-actual) | residual sd | RMSE | share of bulk SSE % |
|---|---|---|---|---|---|---|---|
| (59.999, 618.196] | 18628 | 507.0 | 515.3 | -8.3 | 140.9 | 141.1 | 3.6 |
| (618.196, 721.754] | 18627 | 673.7 | 680.0 | -6.4 | 151.3 | 151.4 | 4.2 |
| (721.754, 803.231] | 18627 | 763.4 | 769.2 | -5.8 | 164.7 | 164.8 | 5.0 |
| (803.231, 878.747] | 18627 | 840.9 | 846.4 | -5.5 | 180.7 | 180.8 | 6.0 |
| (878.747, 956.133] | 18627 | 917.0 | 920.3 | -3.4 | 200.6 | 200.6 | 7.3 |
| (956.133, 1041.691] | 18627 | 998.1 | 999.3 | -1.2 | 212.9 | 212.9 | 8.3 |
| (1041.691, 1140.931] | 18627 | 1089.3 | 1084.7 | 4.7 | 229.7 | 229.7 | 9.6 |
| (1140.931, 1265.566] | 18627 | 1200.3 | 1195.4 | 4.9 | 254.2 | 254.3 | 11.8 |
| (1265.566, 1458.432] | 18627 | 1351.7 | 1343.1 | 8.6 | 291.3 | 291.4 | 15.5 |
| (1458.432, 6689.092] | 18628 | 1744.6 | 1700.8 | 43.8 | 394.1 | 396.5 | 28.7 |

### by airport, against the same-month near-duplicate bound

| airport | n | model RMSE | bias | bound, stand x rwy x op x 2h (same month) | bound, + gap and queue (same month) | coverage of the tight bound % | share of bulk SSE % |
|---|---|---|---|---|---|---|---|
| LIRF | 14603 | 372.5 | 16.1 | 420.5 | 300.7 | 12.0 | 19.9 |
| LTFM | 24725 | 270.5 | -21.3 | 339.5 | 227.8 | 10.0 | 17.7 |
| LFPG | 21017 | 240.0 | 10.7 | 262.7 | 206.7 | 5.8 | 11.9 |
| LEBL | 16322 | 269.4 | -16.3 | 288.8 | 205.7 | 13.7 | 11.6 |
| EGLL | 20297 | 239.6 | 24.5 | 356.3 | 277.8 | 9.2 | 11.4 |
| EDDF | 20887 | 185.4 | 12.1 | 203.9 | 148.9 | 10.5 | 7.0 |
| LEMD | 18541 | 193.2 | 9.6 | 204.8 | 151.6 | 7.6 | 6.8 |
| EHAM | 21970 | 158.4 | 5.9 | 164.8 | 149.7 | 5.7 | 5.4 |
| EDDM | 15647 | 170.0 | -10.4 | 205.8 | 140.2 | 7.6 | 4.4 |
| LSZH | 12263 | 179.3 | 1.9 | 191.6 | 139.3 | 12.8 | 3.9 |

All airports, same month: model 234.1; bound at stand x runway x operator x 2h 278.9 (coverage 80 %); bound with the gap and queue bins 207.6 (coverage 9 %).

## fold `december` (validation month 2025-12), best OOF `F1-phifull-december`

163,838 bulk rows, RMSE **193.4**.

### by predicted value

| predicted taxi, decile | n | mean pred | mean actual | bias (pred-actual) | residual sd | RMSE | share of bulk SSE % |
|---|---|---|---|---|---|---|---|
| (59.999, 629.817] | 16384 | 524.9 | 523.1 | 1.8 | 120.1 | 120.1 | 3.9 |
| (629.817, 721.17] | 16384 | 678.8 | 673.9 | 4.9 | 123.1 | 123.2 | 4.1 |
| (721.17, 794.095] | 16384 | 758.0 | 753.2 | 4.9 | 138.7 | 138.7 | 5.1 |
| (794.095, 863.686] | 16383 | 828.9 | 822.8 | 6.1 | 150.1 | 150.2 | 6.0 |
| (863.686, 937.582] | 16384 | 900.5 | 892.8 | 7.7 | 158.8 | 159.0 | 6.8 |
| (937.582, 1017.474] | 16384 | 976.8 | 968.3 | 8.6 | 175.3 | 175.5 | 8.2 |
| (1017.474, 1113.457] | 16383 | 1064.1 | 1057.4 | 6.8 | 191.5 | 191.6 | 9.8 |
| (1113.457, 1235.805] | 16384 | 1171.2 | 1160.0 | 11.1 | 207.1 | 207.4 | 11.5 |
| (1235.805, 1430.02] | 16384 | 1322.3 | 1315.7 | 6.6 | 241.1 | 241.2 | 15.6 |
| (1430.02, 5748.812] | 16384 | 1713.8 | 1701.1 | 12.7 | 329.3 | 329.5 | 29.0 |

### by airport, against the same-month near-duplicate bound

| airport | n | model RMSE | bias | bound, stand x rwy x op x 2h (same month) | bound, + gap and queue (same month) | coverage of the tight bound % | share of bulk SSE % |
|---|---|---|---|---|---|---|---|
| LTFM | 22250 | 206.2 | 6.8 | 277.4 | 171.0 | 10.5 | 15.4 |
| LFPG | 19431 | 213.7 | 14.7 | 326.0 | 289.8 | 5.4 | 14.5 |
| EGLL | 19809 | 208.1 | 11.1 | 331.2 | 290.0 | 9.0 | 14.0 |
| LIRF | 11932 | 267.0 | -3.1 | 307.0 | 218.8 | 14.2 | 13.9 |
| LEMD | 17925 | 166.3 | 7.7 | 207.6 | 165.5 | 9.5 | 8.1 |
| EDDF | 17222 | 166.7 | 8.2 | 244.9 | 191.6 | 9.0 | 7.8 |
| EHAM | 19141 | 155.9 | 4.2 | 190.9 | 168.1 | 6.5 | 7.6 |
| EDDM | 11899 | 195.4 | 10.7 | 372.4 | 229.6 | 4.9 | 7.4 |
| LEBL | 13746 | 176.6 | 4.8 | 219.0 | 144.4 | 12.0 | 7.0 |
| LSZH | 10483 | 158.6 | -0.7 | 233.5 | 189.5 | 14.0 | 4.3 |

All airports, same month: model 193.4; bound at stand x runway x operator x 2h 275.1 (coverage 80 %); bound with the gap and queue bins 206.4 (coverage 9 %).
