# exp-007 question 1e — the january fold as an instrument

Every retro-v1 out-of-fold line that ran the `january` fold, ranked by january RMSE
as reported and with the two unreachable date-rollover rows removed.

| line | january RMSE as reported | rank | january RMSE, 2 rows removed | rank | non-LIRF as reported | non-LIRF clean | SSE share of the 2 rows |
|---|---|---|---|---|---|---|---|
| `exp-006/g3s17` | 344.34 | 1 | **228.27** | 1 | 340.94 | **210.93** | 56.1 % |
| `exp-006/hFs7` | 344.97 | 3 | **229.11** | 2 | 341.38 | **211.52** | 55.9 % |
| `exp-006/g3s29` | 344.88 | 2 | **229.38** | 3 | 341.45 | **212.11** | 55.8 % |
| `exp-006/g0s17` | 345.02 | 4 | **229.58** | 4 | 341.60 | **212.33** | 55.7 % |
| `exp-006/g3lomo17` | 345.17 | 5 | **229.58** | 5 | 341.74 | **212.30** | 55.8 % |
| `exp-006/g3` | 345.62 | 7 | **230.07** | 6 | 342.13 | **212.71** | 55.7 % |
| `exp-006/g4` | 345.82 | 9 | **230.39** | 7 | 342.42 | **213.19** | 55.6 % |
| `exp-001/L1-philirf-guard` | 345.57 | 6 | **230.62** | 8 | 340.89 | **211.44** | 55.5 % |
| `exp-006/g5s7` | 345.91 | 10 | **230.62** | 9 | 341.96 | **212.57** | 55.5 % |
| `exp-006/h0s7` | 345.99 | 11 | **230.75** | 10 | 341.88 | **212.44** | 55.5 % |
| `exp-006/hVs7` | 346.01 | 12 | **230.80** | 11 | 342.26 | **213.08** | 55.5 % |
| `exp-001/L1-philirf` | 345.71 | 8 | **230.82** | 12 | 341.04 | **211.68** | 55.4 % |
| `exp-006/h0bs7` | 346.17 | 14 | **231.02** | 13 | 343.12 | **214.43** | 55.5 % |
| `exp-006/hFbs7` | 346.25 | 15 | **231.05** | 14 | 343.02 | **214.17** | 55.5 % |
| `exp-001/F2-phikfold` | 346.04 | 13 | **231.33** | 15 | 341.49 | **212.41** | 55.3 % |
| `exp-006/hVs17` | 346.48 | 17 | **231.53** | 16 | 342.98 | **214.27** | 55.3 % |
| `exp-001/G1cv-s7` | 346.49 | 18 | **231.70** | 17 | 342.29 | **213.34** | 55.3 % |
| `exp-001/G2fu-s7` | 346.52 | 19 | **231.75** | 18 | 342.28 | **213.33** | 55.3 % |
| `exp-001/F1-phifull` | 346.43 | 16 | **231.78** | 19 | 342.14 | **213.30** | 55.2 % |
| `exp-006/g2` | 346.76 | 21 | **231.82** | 20 | 343.21 | **214.49** | 55.3 % |
| `exp-006/hCbs7` | 346.74 | 20 | **231.83** | 21 | 344.19 | **216.10** | 55.3 % |
| `exp-006/hCs7` | 346.86 | 22 | **232.03** | 22 | 343.32 | **214.75** | 55.3 % |
| `exp-006/g1` | 346.93 | 23 | **232.13** | 23 | 343.47 | **214.97** | 55.2 % |
| `exp-006/g3lomo7` | 347.39 | 25 | **232.77** | 24 | 343.86 | **215.53** | 55.1 % |
| `exp-006/hCbc` | 347.34 | 24 | **232.77** | 25 | 344.37 | **216.44** | 55.1 % |
| `exp-001/H0-s7` | 347.45 | 26 | **233.60** | 26 | 344.22 | **216.98** | 54.8 % |
| `exp-001/e0-base` | 347.45 | 26 | **233.60** | 26 | 344.22 | **216.98** | 54.8 % |
| `exp-001/e0-base-guard` | 347.53 | 28 | **233.72** | 28 | 344.31 | **217.12** | 54.8 % |
| `exp-006/hCs17` | 348.62 | 29 | **234.73** | 29 | 345.34 | **218.04** | 54.7 % |
| `exp-001/c9-te` | 348.68 | 30 | **234.73** | 30 | 344.15 | **216.04** | 54.7 % |
| `exp-001/c6-noac` | 349.03 | 31 | **235.18** | 31 | 345.83 | **218.63** | 54.6 % |
| `exp-001/c5-v2` | 349.07 | 33 | **235.27** | 32 | 344.04 | **215.84** | 54.6 % |
| `exp-006/g5s17` | 349.06 | 32 | **235.54** | 33 | 346.22 | **219.62** | 54.5 % |
| `exp-006/hGs7` | 349.71 | 34 | **236.50** | 34 | 347.75 | **222.02** | 54.3 % |
| `exp-006/hCbs17` | 350.91 | 36 | **238.10** | 35 | 347.21 | **220.98** | 54.0 % |
| `exp-006/g0` | 350.79 | 35 | **238.14** | 36 | 348.57 | **223.34** | 53.9 % |
| `exp-006/h0s17` | 351.04 | 37 | **238.36** | 37 | 347.84 | **222.03** | 53.9 % |
| `exp-006/h0bs17` | 351.42 | 38 | **238.91** | 38 | 349.03 | **223.90** | 53.8 % |
| `exp-001/H1-s7` | 356.71 | 39 | **246.70** | 39 | 346.23 | **219.58** | 52.2 % |
| `exp-006/hFs17` | 375.70 | 40 | **273.34** | 40 | 374.77 | **262.19** | 47.1 % |
| `exp-006/phi-formula` | 518.03 | 41 | **449.34** | 41 | 414.33 | **316.13** | 24.8 % |
| `exp-006/hFbs17` | 539.01 | 42 | **473.34** | 42 | 343.23 | **214.56** | 22.9 % |

Spread across the 42 lines: **194.7 s** as reported, **245.1 s** with the two rows removed.

Kendall rank correlation between the two orderings: **0.96**.

The squared error the two rows add is nearly the same for every line: 10.167e9 to 10.224e9 s² across the 39 lines below 250 s clean, a spread of 0.6 %. Every model predicts them between 900 and 1,300 s, so they behave as an additive constant, not as a tie-breaker. That is why the ranking survives.

(The empirical slope is a check on the arithmetic, not a fit: with `reported = sqrt(clean² + C/n)` and C constant, `d(reported)/d(clean) = clean/reported`, which averages **0.671** over these lines against the fitted 0.672.)

What they do instead is (i) raise the LEVEL by 114 s, which is what breaks any arithmetic against the board, and (ii) COMPRESS differences by a factor **0.67**: a 10 s improvement in the january error a model can actually attack shows up as 6.7 s in the reported january RMSE.
