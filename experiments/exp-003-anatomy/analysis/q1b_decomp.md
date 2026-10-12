# Q1 quantitative: what explains `BLOCK - AOBT_3`

## Out-of-sample: RMSE of `offset = group median fitted on the training months`

Scored on the bulk (NM join, taxi <= 3600 s) of each validation month. `taxi_hat = (MVT - AOBT_3) - offset_hat`, so this is RMSE on the target. `explained %` is the share of the *variance* the grouping removes relative to the global-median predictor, out of sample.

| grouping | rmse december | rmse january | rmse july | explained_pct december | explained_pct january | explained_pct july | levels |
|---|---|---|---|---|---|---|---|
| apt_stand_rwy | 330.32 | 323.59 | 398.66 | 16.33 | 17.18 | 11.40 | 6053 |
| apt_stand | 333.11 | 325.90 | 401.47 | 14.91 | 15.99 | 10.14 | 2033 |
| apt_stand_hour | 332.67 | 325.69 | 402.01 | 15.13 | 16.10 | 9.90 | 5990 |
| apt_op | 337.40 | 330.57 | 402.83 | 12.71 | 13.57 | 9.53 | 1133 |
| apt_type | 336.66 | 329.75 | 403.77 | 13.09 | 13.99 | 9.11 | 671 |
| apt_pfx | 337.14 | 329.57 | 404.65 | 12.84 | 14.09 | 8.71 | 128 |
| apt_rwy | 340.20 | 332.32 | 404.91 | 11.25 | 12.65 | 8.59 | 58 |
| apt_hour | 341.60 | 335.36 | 407.41 | 10.52 | 11.05 | 7.46 | 222 |
| apt | 343.93 | 336.38 | 409.12 | 9.29 | 10.50 | 6.68 | 10 |
| stand_pfx | 350.13 | 346.49 | 413.77 | 5.99 | 5.04 | 4.55 | 43 |
| AIRCRAFT_OPERATOR_flt | 349.41 | 342.92 | 414.45 | 6.38 | 6.99 | 4.24 | 424 |
| ADES_mvt | 352.04 | 346.36 | 417.54 | 4.96 | 5.11 | 2.81 | 739 |
| AIRCRAFT_TYPE_mvt | 356.56 | 352.80 | 418.33 | 2.51 | 1.55 | 2.44 | 139 |
| MARKET_SEGMENT_flt | 359.59 | 354.61 | 421.47 | 0.85 | 0.54 | 0.97 | 8 |
| hour | 360.18 | 355.78 | 421.91 | 0.52 | -0.12 | 0.76 | 24 |
| WK_TBL_CAT_flt | 360.17 | 355.70 | 422.04 | 0.52 | -0.07 | 0.70 | 4 |
| (global median) | 361.12 | 355.57 | 423.52 | 0.00 | 0.00 | 0.00 | 1 |

## In-sample eta-squared on 2025 (share of the offset's variance between group means)

| grouping | levels | eta2_pct |
|---|---|---|
| apt_stand_rwy | 11022 | 19.84 |
| apt_stand_hour | 8801 | 17.96 |
| apt_stand | 2494 | 16.91 |
| apt_op | 3114 | 15.40 |
| apt_type | 1255 | 14.85 |
| apt_pfx | 167 | 14.70 |
| apt_rwy | 68 | 12.97 |
| apt_hour | 235 | 12.51 |
| apt | 10 | 11.07 |
| AIRCRAFT_OPERATOR_flt | 676 | 7.75 |
| stand_pfx | 65 | 6.44 |
| ADES_mvt | 1481 | 5.12 |
| AIRCRAFT_TYPE_mvt | 239 | 2.06 |
| MARKET_SEGMENT_flt | 8 | 0.89 |
| hour | 24 | 0.57 |
| WK_TBL_CAT_flt | 4 | 0.26 |

## Stand-level shape of the offset (2025, NM join, taxi <= 3600 s, stands with n >= 200)

1602 stands with at least 200 departures cover 97.6 % of the bulk rows.

### distribution of the per-stand dispersion

|  | count | mean | std | min | 5% | 25% | 50% | 75% | 95% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| std | 1602.0 | 320.4 | 95.7 | 162.2 | 196.1 | 248.1 | 307.6 | 370.3 | 495.2 | 860.3 |
| iqr | 1602.0 | 296.9 | 87.3 | 130.5 | 188.0 | 236.0 | 291.1 | 357.4 | 476.0 | 1214.8 |
| sep_sd | 1594.0 | 2.5 | 0.9 | 0.8 | 1.6 | 2.0 | 2.3 | 2.7 | 4.2 | 15.2 |
| gap_s | 1594.0 | 540.1 | 226.5 | 258.9 | 315.8 | 404.8 | 479.7 | 616.7 | 926.9 | 3330.6 |

### per airport: how tight are its stands?

| airport | stands | rows | med_std | min_std | n_std_lt_120 | med_sep | n_sep_gt3 | n_bimodal |
|---|---|---|---|---|---|---|---|---|
| EDDF | 175 | 225178 | 226.9 | 162.2 | 0 | 2.5 | 4 | 2.0 |
| EDDM | 133 | 162403 | 305.7 | 196.9 | 0 | 3.9 | 101 | 22.0 |
| EGLL | 148 | 236287 | 349.2 | 272.9 | 0 | 1.8 | 0 | 0.0 |
| EHAM | 170 | 240169 | 273.6 | 171.2 | 0 | 2.7 | 66 | 6.0 |
| LEBL | 143 | 172323 | 289.7 | 184.8 | 0 | 2.5 | 38 | 4.0 |
| LEMD | 209 | 199667 | 254.4 | 183.4 | 0 | 2.2 | 14 | 1.0 |
| LFPG | 235 | 223708 | 329.4 | 193.7 | 0 | 2.0 | 5 | 4.0 |
| LIRF | 110 | 155848 | 417.4 | 331.6 | 0 | 2.7 | 28 | 24.0 |
| LSZH | 95 | 128161 | 214.2 | 162.5 | 0 | 2.6 | 2 | 2.0 |
| LTFM | 184 | 266400 | 446.6 | 313.2 | 0 | 1.9 | 4 | 2.0 |

### the 20 tightest stands (near-constant pushback convention)

| apt|stand | n | median | std | iqr | sep_sd |
|---|---|---|---|---|---|
| EDDF|V173B | 881 | 205.0 | 162.2 | 176.0 | 2.7 |
| LSZH|REG | 710 | 255.0 | 162.5 | 162.0 | 2.3 |
| EDDF|V162 | 1533 | 258.0 | 162.7 | 193.0 | 2.9 |
| EDDF|V172 | 1309 | 210.0 | 163.0 | 184.0 | 2.8 |
| EDDF|V173 | 251 | 195.0 | 165.4 | 184.5 | 2.4 |
| LSZH|F72 | 1122 | 166.0 | 166.6 | 202.8 | 2.7 |
| EDDF|V176 | 1028 | 178.0 | 167.8 | 184.2 | 2.4 |
| EDDF|V173A | 948 | 201.0 | 169.9 | 163.0 | 2.4 |
| EHAM|A65 | 1238 | 103.0 | 171.2 | 194.8 | 2.4 |
| LSZH|F71 | 1166 | 173.0 | 171.2 | 194.0 | 2.6 |
| EDDF|V160 | 1615 | 245.0 | 171.3 | 188.0 | 2.7 |
| LSZH|H12 | 1599 | 163.0 | 171.9 | 208.0 | 2.7 |
| EDDF|A62A | 721 | 87.0 | 172.0 | 206.0 | 2.7 |
| EHAM|A55 | 1331 | 118.0 | 172.5 | 187.5 | 2.3 |
| LSZH|F70 | 1244 | 196.0 | 173.1 | 191.0 | 2.8 |
| EDDF|V171B | 1244 | 208.0 | 175.6 | 180.2 | 2.3 |
| EDDF|V158 | 1616 | 243.0 | 176.1 | 183.2 | 2.3 |
| EDDF|V136 | 1879 | 78.0 | 176.5 | 190.5 | 2.8 |
| LSZH|D11 | 867 | 138.0 | 176.6 | 229.0 | 2.8 |
| LSZH|D01 | 1118 | 188.5 | 177.0 | 218.8 | 2.7 |

### the 20 most clearly bimodal stands (2-means separation in pooled sd, minority share >= 15 %)

A unimodal Gaussian scores 2.67 on this statistic and a t(3) scores 1.78, so `n_bimodal` above counts stands over 3.0.

| apt|stand | n | median | std | sep_sd | minor_share | gap_s |
|---|---|---|---|---|---|---|
| EDDM|906 | 218 | -57.00 | 376.40 | 4.29 | 0.17 | 850.84 |
| LIRF|824 | 452 | -125.00 | 570.42 | 4.27 | 0.16 | 1317.89 |
| EDDM|907 | 258 | -186.00 | 501.92 | 4.24 | 0.26 | 1003.55 |
| LIRF|829 | 438 | -150.00 | 633.58 | 4.18 | 0.17 | 1430.02 |
| EDDM|111 | 2239 | -123.00 | 345.81 | 3.94 | 0.18 | 750.95 |
| EDDM|110 | 1819 | -126.00 | 401.80 | 3.88 | 0.18 | 870.07 |
| EDDM|250 | 1211 | -119.00 | 433.84 | 3.71 | 0.15 | 967.51 |
| EDDM|216 | 1631 | -116.00 | 375.28 | 3.69 | 0.17 | 813.60 |
| LIRF|827 | 473 | -182.00 | 604.03 | 3.65 | 0.18 | 1284.40 |
| LIRF|815 | 276 | -115.00 | 527.11 | 3.62 | 0.16 | 1156.24 |
| EDDM|318 | 1928 | -62.00 | 414.54 | 3.61 | 0.17 | 884.03 |
| EDDM|207 | 2014 | -57.00 | 305.50 | 3.54 | 0.16 | 661.60 |
| EDDM|204 | 1585 | -120.00 | 347.40 | 3.49 | 0.17 | 736.55 |
| LIRF|108 | 320 | -57.50 | 543.54 | 3.49 | 0.21 | 1095.75 |
| EDDM|349 | 393 | -115.00 | 260.60 | 3.47 | 0.24 | 507.27 |
| LIRF|828 | 446 | -179.00 | 537.26 | 3.38 | 0.20 | 1077.49 |
| EDDM|803 | 236 | 119.00 | 274.40 | 3.34 | 0.21 | 544.01 |
| LIRF|710 | 1234 | -174.50 | 530.83 | 3.33 | 0.22 | 1037.72 |
| LIRF|101 | 204 | -57.00 | 622.77 | 3.33 | 0.19 | 1257.57 |
| EDDM|201 | 1490 | -121.00 | 336.88 | 3.33 | 0.19 | 678.84 |

## Drift across 2025: trimmed mean of the offset (10 % each tail) by airport and month

| airport | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | 2025-07 | 2025-08 | 2025-09 | 2025-10 | 2025-11 | 2025-12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 91 | 97 | 100 | 88 | 70 | 68 | 72 | 80 | 59 | 65 | 65 | 73 |
| EDDM | -21 | 4 | 14 | -2 | -17 | -12 | -46 | -38 | -53 | -44 | -56 | -110 |
| EGLL | 66 | 38 | 51 | 29 | 31 | 26 | 30 | 28 | 29 | 34 | 46 | 47 |
| EHAM | 110 | 107 | 122 | 121 | 127 | 125 | 134 | 132 | 126 | 126 | 106 | 102 |
| LEBL | 2 | -13 | -35 | -41 | -58 | -61 | -76 | -57 | -55 | -54 | -17 | -22 |
| LEMD | 30 | 27 | -29 | 11 | 14 | -16 | -25 | -35 | -30 | -15 | 12 | 13 |
| LFPG | -18 | -8 | -2 | -14 | -22 | -23 | -52 | -38 | -38 | -20 | -22 | -27 |
| LIRF | -77 | -68 | -112 | -165 | -170 | -189 | -218 | -174 | -147 | -157 | -50 | -85 |
| LSZH | -41 | -38 | -37 | -44 | -43 | -43 | -42 | -43 | -39 | -33 | -47 | -59 |
| LTFM | 316 | 307 | 342 | 271 | 251 | 241 | 183 | 219 | 239 | 253 | 300 | 265 |

(The *median* of this offset is a bad drift statistic: the density is flat to about 0.16 % per second near the centre, so the median hops by a whole minute between adjacent months without the distribution moving. The trimmed mean above is what the drift tables in `q1_offset_structure.md` should be read against.)
