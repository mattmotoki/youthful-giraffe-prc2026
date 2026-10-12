# Bulk ceiling: three estimates

The bulk is departures with an NM join and taxi time at or below 3600 s: 2,059,538 of 2,085,047 2025 departures (98.78 %).

## 1. Matched-group residual: how much of the offset survives perfect conditioning

Pooled within-group standard deviation of `BLOCK - AOBT_3` over 2025 bulk rows, for groups that share everything named. A model that conditions on exactly these columns and nothing else cannot beat the `sigma` column; a model that also uses take-off minus AOBT, neighbours and congestion can go below it. The row where sigma stops falling as the grouping gets finer is where the offset stops being a function of the flight's identity.

| grouping | rows | groups | share_pct | sigma |
|---|---|---|---|---|
| airport | 2059538 | 10 | 100.0 | 340.4 |
| airport x hour | 2059535 | 232 | 100.0 | 337.7 |
| airport x runway x hour | 2059472 | 1144 | 100.0 | 333.1 |
| stand | 2059385 | 2341 | 100.0 | 329.3 |
| stand x runway | 2058332 | 9816 | 99.9 | 324.1 |
| stand x runway x 2h | 2046670 | 59084 | 99.4 | 319.3 |
| stand x runway x aircraft type | 2037894 | 56789 | 98.9 | 319.2 |
| stand x runway x operator | 2028192 | 53580 | 98.5 | 317.5 |
| stand x runway x type x 2h | 1951245 | 179238 | 94.7 | 313.0 |
| airport x same 30-min slot (same day) | 2053443 | 138184 | 99.7 | 311.4 |
| airport x runway x same 30-min slot | 2035688 | 210345 | 98.8 | 301.7 |
| stand x same day | 1911000 | 404939 | 92.8 | 322.1 |
| stand x runway x same day | 1674372 | 474488 | 81.3 | 317.9 |
| stand x runway x type x same day | 872031 | 347228 | 42.3 | 311.4 |

Same instrument on the *taxi time itself* (what a model without the NM columns faces):

| grouping | rows | groups | share_pct | sigma |
|---|---|---|---|---|
| airport | 2059538 | 10 | 100.0 | 345.7 |
| airport x hour | 2059535 | 232 | 100.0 | 337.3 |
| airport x runway x hour | 2059472 | 1144 | 100.0 | 321.5 |
| stand | 2059385 | 2341 | 100.0 | 309.7 |
| stand x runway | 2058332 | 9816 | 99.9 | 284.9 |
| stand x runway x 2h | 2046670 | 59084 | 99.4 | 274.4 |
| stand x runway x aircraft type | 2037894 | 56789 | 98.9 | 278.1 |
| stand x runway x operator | 2028192 | 53580 | 98.5 | 277.9 |
| stand x runway x type x 2h | 1951245 | 179238 | 94.7 | 267.4 |

## 2. Back-out from the leaderboard

Hard upper bound on anyone's bulk RMSE: 277.3 * sqrt(N / n_bulk) = 279.7 s (achieved only by a team that scores zero error on every non-bulk row).

Ranking population: 344,841 departures = 388 noac + 4,902 other no-NM + 500 NM rows above 3600 s (at the 2025 rate of 0.147 %) + 339,051 bulk rows.

Solving `N * 277.3^2 = n_noac*a^2 + n_other_noNM*b^2 + n_NMtail*c^2 + n_bulk*B^2` for the bulk RMSE `B`, over a grid of what the leaders might reach on the other three. **This is a conditional grid, not a bound.** The only unconditional statement the board supports is the hard upper bound above: a better total does not imply a better score on any particular segment, so our own segment numbers cannot be used to bound the leaders' segment numbers either (gpt-5.6-sol round 3).

| noac RMSE | other no-NM RMSE | NM tail RMSE | implied bulk RMSE |
|---|---|---|---|
| 3000 | 600 | 1200 | 246.1 |
| 3000 | 600 | 1800 | 240.7 |
| 3000 | 600 | 2500 | 231.3 |
| 3000 | 900 | 1200 | 232.5 |
| 3000 | 900 | 1800 | 226.8 |
| 3000 | 900 | 2500 | 216.7 |
| 3000 | 1400 | 1200 | 193.5 |
| 3000 | 1400 | 1800 | 186.5 |
| 3000 | 1400 | 2500 | 174.2 |
| 3000 | 2000 | 1200 | 89.2 |
| 3000 | 2000 | 1800 | 72.8 |
| 3000 | 2000 | 2500 | 29.2 |
| 4000 | 600 | 1200 | 229.3 |
| 4000 | 600 | 1800 | 223.4 |
| 4000 | 600 | 2500 | 213.2 |
| 4000 | 900 | 1200 | 214.6 |
| 4000 | 900 | 1800 | 208.3 |
| 4000 | 900 | 2500 | 197.4 |
| 4000 | 1400 | 1200 | 171.6 |
| 4000 | 1400 | 1800 | 163.6 |
| 4000 | 1400 | 2500 | 149.5 |
| 4000 | 2000 | 1200 |  |
| 4000 | 2000 | 1800 |  |
| 4000 | 2000 | 2500 |  |
| 4500 | 600 | 1200 | 218.4 |
| 4500 | 600 | 1800 | 212.2 |
| 4500 | 600 | 2500 | 201.5 |
| 4500 | 900 | 1200 | 203.0 |
| 4500 | 900 | 1800 | 196.3 |
| 4500 | 900 | 2500 | 184.7 |
| 4500 | 1400 | 1200 | 156.8 |
| 4500 | 1400 | 1800 | 148.0 |
| 4500 | 1400 | 2500 | 132.2 |
| 4500 | 2000 | 1200 |  |
| 4500 | 2000 | 1800 |  |
| 4500 | 2000 | 2500 |  |
| 5000 | 600 | 1200 | 205.6 |
| 5000 | 600 | 1800 | 199.0 |
| 5000 | 600 | 2500 | 187.5 |
| 5000 | 900 | 1200 | 189.1 |
| 5000 | 900 | 1800 | 182.0 |
| 5000 | 900 | 2500 | 169.3 |
| 5000 | 1400 | 1200 | 138.3 |
| 5000 | 1400 | 1800 | 128.4 |
| 5000 | 1400 | 2500 | 109.7 |
| 5000 | 2000 | 1200 |  |
| 5000 | 2000 | 1800 |  |
| 5000 | 2000 | 2500 |  |
