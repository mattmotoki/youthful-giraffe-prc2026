# Q1(e) — the floor per airport, both quantities on the same rows

The bound and the model RMSE are both computed on the rows that have a near-duplicate, which is the only comparison that means anything (review round 1, B5/B6). Group key is `key_raw` = airport|stand, because stand names repeat across airports.

## july_loo, tight match (... + AOBT-LOBT and queue bins)

| airport | rows with a duplicate | share of the airport's bulk % | bound | model on those rows | model minus bound |
|---|---|---|---|---|---|
| EGLL | 1877 | 9.2 | 277.8 | 197.0 | -80.8 |
| LTFM | 2467 | 10.0 | 227.8 | 203.0 | -24.7 |
| LFPG | 1221 | 5.8 | 206.7 | 185.7 | -21.0 |
| ALL | 17187 | 9.2 | 207.6 | 187.3 | -20.3 |
| LIRF | 1756 | 12.0 | 300.7 | 282.1 | -18.5 |
| EHAM | 1257 | 5.7 | 149.7 | 138.5 | -11.3 |
| EDDM | 1191 | 7.6 | 140.2 | 135.0 | -5.2 |
| EDDF | 2200 | 10.5 | 148.9 | 144.2 | -4.7 |
| LEMD | 1410 | 7.6 | 151.6 | 148.8 | -2.8 |
| LEBL | 2234 | 13.7 | 205.7 | 203.3 | -2.4 |
| LSZH | 1574 | 12.8 | 139.3 | 141.7 | 2.5 |

## july_loo, loose match (stand x runway x operator x 2h)

| airport | rows with a duplicate | share of the airport's bulk % | bound | model on those rows | model minus bound |
|---|---|---|---|---|---|
| EGLL | 17343 | 85.4 | 356.3 | 241.1 | -115.2 |
| LTFM | 19294 | 78.0 | 339.5 | 259.7 | -79.8 |
| LIRF | 12239 | 83.8 | 420.5 | 352.0 | -68.5 |
| ALL | 149170 | 80.1 | 278.9 | 226.9 | -52.1 |
| EDDM | 12102 | 77.3 | 205.8 | 168.3 | -37.5 |
| LEBL | 13342 | 81.7 | 288.8 | 257.4 | -31.4 |
| LFPG | 16449 | 78.3 | 262.7 | 234.8 | -28.0 |
| LSZH | 9500 | 77.5 | 191.6 | 168.7 | -22.9 |
| EDDF | 17309 | 82.9 | 203.9 | 181.4 | -22.5 |
| LEMD | 13842 | 74.7 | 204.8 | 183.6 | -21.2 |
| EHAM | 17750 | 80.8 | 164.8 | 152.2 | -12.6 |

## december, tight match (... + AOBT-LOBT and queue bins)

| airport | rows with a duplicate | share of the airport's bulk % | bound | model on those rows | model minus bound |
|---|---|---|---|---|---|
| LFPG | 1058 | 5.4 | 289.8 | 176.4 | -113.4 |
| EGLL | 1790 | 9.0 | 290.0 | 186.0 | -103.9 |
| EDDM | 586 | 4.9 | 229.6 | 145.1 | -84.5 |
| EDDF | 1551 | 9.0 | 191.6 | 135.6 | -56.0 |
| LSZH | 1464 | 14.0 | 189.5 | 135.0 | -54.6 |
| ALL | 15077 | 9.2 | 206.4 | 159.7 | -46.7 |
| LEMD | 1711 | 9.5 | 165.5 | 137.4 | -28.1 |
| EHAM | 1237 | 6.5 | 168.1 | 145.9 | -22.2 |
| LTFM | 2332 | 10.5 | 171.0 | 155.2 | -15.8 |
| LIRF | 1695 | 14.2 | 218.8 | 210.1 | -8.8 |
| LEBL | 1653 | 12.0 | 144.4 | 140.3 | -4.1 |

## december, loose match (stand x runway x operator x 2h)

| airport | rows with a duplicate | share of the airport's bulk % | bound | model on those rows | model minus bound |
|---|---|---|---|---|---|
| EDDM | 8436 | 70.9 | 372.4 | 188.8 | -183.5 |
| EGLL | 16936 | 85.5 | 331.2 | 207.4 | -123.7 |
| LFPG | 15460 | 79.6 | 326.0 | 204.1 | -121.9 |
| ALL | 130485 | 79.6 | 275.1 | 186.8 | -88.2 |
| LTFM | 17597 | 79.1 | 277.4 | 193.1 | -84.3 |
| LSZH | 8048 | 76.8 | 233.5 | 152.4 | -81.1 |
| EDDF | 13973 | 81.1 | 244.9 | 165.2 | -79.7 |
| LIRF | 9893 | 82.9 | 307.0 | 253.7 | -53.2 |
| LEBL | 11400 | 82.9 | 219.0 | 168.7 | -50.3 |
| LEMD | 13748 | 76.7 | 207.6 | 159.8 | -47.8 |
| EHAM | 14994 | 78.3 | 190.9 | 151.6 | -39.3 |
