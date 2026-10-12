# Q1(d) — the model scored on the rows the bound is computed on

The near-duplicate bound only exists on rows that have a near-duplicate: 9 % of the bulk at the tight match, 80 % at the loose one, and those rows are a biased sample (busy stands, repeated identical operations). Comparing a bound computed on 9 % of rows with a model RMSE computed on 100 % is not a comparison. Here both are on the same rows.

| fold | match | rows with a duplicate | share of bulk % | bound (pooled within-group sd) | model RMSE on those rows | model RMSE on all bulk rows |
|---|---|---|---|---|---|---|
| july_loo | stand x runway x operator x 2h | 149172 | 80.1 | 278.9 | 226.9 | 234.1 |
| july_loo | ... + AOBT-LOBT and queue bins | 17187 | 9.2 | 207.6 | 187.3 | 234.1 |
| december | stand x runway x operator x 2h | 130485 | 79.6 | 275.1 | 186.8 | 193.4 |
| december | ... + AOBT-LOBT and queue bins | 15077 | 9.2 | 206.4 | 159.7 | 193.4 |
