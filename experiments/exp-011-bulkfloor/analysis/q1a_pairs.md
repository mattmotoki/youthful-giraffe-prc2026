# Q1(a) — near-duplicate disagreement, all twelve 2025 months

Bulk = NM join, aircraft type present, taxi <= 3600 s: 2,059,538 of 2,085,047 departures (98.78 %). `sigma` is the pooled within-group standard deviation of the taxi time, which equals the within-pair disagreement over sqrt(2). **It is an upper bound on the irreducible sigma**, not a floor: what a group does not match on still varies inside it. Rungs without a date or month key pool rows across the whole year, so they also carry the seasonal variation the model can condition on: the `same month`, `same week` and `same day` rungs are the honest ones, at the cost of coverage.

| match | rows | groups | share | sigma |
|---|---|---|---|---|
| airport | 2059538 | 10 | 100.0 | 345.7 |
| airport x 2-hour block | 2059537 | 118 | 100.0 | 339.2 |
| airport x runway x 2-hour block | 2059503 | 613 | 100.0 | 323.7 |
| stand x runway | 2058147 | 10541 | 99.9 | 284.5 |
| stand x runway x operator | 2026497 | 55392 | 98.4 | 277.6 |
| stand x runway x operator x 2h | 1957953 | 138275 | 95.1 | 266.9 |
| ... + same AOBT-LOBT 2-min bin | 1417564 | 344738 | 68.8 | 219.1 |
| ... + same queue bin | 783614 | 285637 | 38.0 | 208.8 |
| ... + same day (the tightest, and the thinnest) | 1344 | 672 | 0.1 | 229.1 |
| stand x runway x same day | 1652017 | 486153 | 80.2 | 273.3 |
| stand x runway x operator x same day | 1072475 | 380360 | 52.1 | 267.2 |
| stand x runway x operator x 2h x same month | 1650027 | 363312 | 80.1 | 259.7 |
| stand x runway x operator x 2h x same week | 1153024 | 412699 | 56.0 | 257.6 |

## The two tightest rungs, per airport

| airport | bulk rows | sigma, stand x runway x operator x 2h | sigma, + gap and queue bins | rows in a group of 2+ | share % |
|---|---|---|---|---|---|
| EDDF | 228115 | 213.5 | 167.7 | 91400 | 40.1 |
| EDDM | 166276 | 246.2 | 163.8 | 49068 | 29.5 |
| EGLL | 237449 | 323.7 | 276.4 | 113521 | 47.8 |
| EHAM | 243796 | 206.8 | 183.1 | 72319 | 29.7 |
| LEBL | 177842 | 233.7 | 164.5 | 85331 | 48.0 |
| LEMD | 211295 | 211.3 | 160.2 | 75681 | 35.8 |
| LFPG | 235429 | 291.1 | 234.6 | 67220 | 28.6 |
| LIRF | 157994 | 365.6 | 272.1 | 72170 | 45.7 |
| LSZH | 132595 | 217.6 | 174.9 | 59011 | 44.5 |
| LTFM | 268747 | 298.0 | 202.2 | 97893 | 36.4 |
