# exp-007 question 2J — where july's other-nine 249.7 s lives

Baseline `G1cv-s7`. 175,502 non-LIRF july departures, RMSE **249.7 s**, against the 200-240 s band the board implies.
July carries no unreachable rollover row (`analysis/q1_floor.md`), so every number here
is error a model can attack.

## By airport

**july, non-LIRF, by airport**

| apt | rows | RMSE | SSE % | mean error | mean y | median span |
|---|---|---|---|---|---|---|
| LTFM | 25,171 | 303.3 | **21.2** | -8 | 1,094 | 1,198 |
| EGLL | 20,790 | 312.8 | **18.6** | -5 | 1,431 | 1,376 |
| LFPG | 21,712 | 272.5 | **14.7** | -8 | 1,028 | 898 |
| LEBL | 16,752 | 273.9 | **11.5** | +13 | 1,008 | 767 |
| EHAM | 22,489 | 197.7 | **8.0** | -33 | 787 | 900 |
| EDDF | 21,334 | 202.4 | **8.0** | -17 | 868 | 896 |
| LEMD | 18,818 | 203.5 | **7.1** | -7 | 1,032 | 985 |
| EDDM | 15,825 | 194.7 | **5.5** | +10 | 805 | 721 |
| LSZH | 12,611 | 216.2 | **5.4** | -22 | 742 | 688 |

The three airports the desk named are the three largest contributions, and they are large for different reasons: see the regime split below.

## By regime, with the block-on-LOBT class broken out

**july, non-LIRF, by regime**

| reg2 | rows | RMSE | SSE % | mean error | mean y | median span |
|---|---|---|---|---|---|---|
| NM bulk | 167,188 | 219.4 | **73.5** | -16 | 981 | 958 |
| no-NM | 3,597 | 588.0 | **11.4** | +1 | 1,052 | nan |
| NM tail > 3600 s | 232 | 1,892.2 | **7.6** | +1,427 | 4,773 | 605 |
| block on LOBT/SCHED | 4,485 | 427.7 | **7.5** | +157 | 1,178 | 1,162 |

**the same, per airport (top 20 cells)**

| apt/reg2 | rows | RMSE | SSE % | mean error | mean y | median span |
|---|---|---|---|---|---|---|
| ('LTFM', 'NM bulk') | 22,999 | 281.2 | **16.6** | -13 | 1,094 | 1,198 |
| ('LFPG', 'NM bulk') | 20,515 | 247.0 | **11.4** | -13 | 1,016 | 898 |
| ('EGLL', 'NM bulk') | 19,870 | 242.1 | **10.6** | -17 | 1,397 | 1,376 |
| ('LEBL', 'NM bulk') | 15,425 | 221.7 | **6.9** | -16 | 979 | 771 |
| ('EHAM', 'NM bulk') | 21,786 | 184.5 | **6.8** | -34 | 790 | 899 |
| ('EDDF', 'NM bulk') | 20,739 | 186.5 | **6.6** | -21 | 863 | 897 |
| ('LEMD', 'NM bulk') | 18,314 | 197.5 | **6.5** | -10 | 1,026 | 985 |
| ('EGLL', 'no-NM') | 338 | 1,244.9 | **4.8** | +131 | 1,806 | nan |
| ('EDDM', 'NM bulk') | 15,376 | 172.7 | **4.2** | +6 | 799 | 721 |
| ('LEBL', 'block on LOBT/SCHED') | 899 | 705.0 | **4.1** | +501 | 1,476 | 708 |
| ('LSZH', 'NM bulk') | 12,164 | 185.6 | **3.8** | -23 | 741 | 688 |
| ('EGLL', 'NM tail > 3600 s') | 155 | 1,382.2 | **2.7** | +1,019 | 4,673 | 602 |
| ('LFPG', 'no-NM') | 663 | 519.0 | **1.6** | -63 | 1,084 | nan |
| ('LTFM', 'NM tail > 3600 s') | 21 | 2,882.5 | **1.6** | +2,699 | 4,651 | 1,138 |
| ('LTFM', 'no-NM') | 424 | 640.5 | **1.6** | +19 | 1,189 | nan |
| ('LTFM', 'block on LOBT/SCHED') | 1,727 | 294.5 | **1.4** | +14 | 1,028 | 1,199 |
| ('LFPG', 'NM tail > 3600 s') | 31 | 1,983.3 | **1.1** | +1,633 | 5,259 | 965 |
| ('LSZH', 'no-NM') | 345 | 591.1 | **1.1** | -110 | 659 | nan |
| ('EHAM', 'NM tail > 3600 s') | 4 | 4,825.0 | **0.9** | +3,893 | 6,516 | 2,778 |
| ('EDDM', 'no-NM') | 172 | 648.7 | **0.7** | +42 | 941 | nan |

## Is a long take-off-minus-AOBT the marker?

**july, NM-joined non-LIRF, by take-off − AOBT**

| span_bin | rows | RMSE | SSE % | mean error | mean y | median span |
|---|---|---|---|---|---|---|
| <300 | 1,204 | 538.0 | **3.2** | +145 | 1,260 | 124 |
| 300-600 | 12,125 | 231.5 | **5.9** | +9 | 671 | 546 |
| 600-900 | 62,882 | 206.6 | **24.5** | +2 | 805 | 733 |
| 900-1200 | 53,391 | 223.7 | **24.4** | -13 | 1,039 | 1,082 |
| 1200-1800 | 38,187 | 265.0 | **24.5** | -24 | 1,261 | 1,319 |
| 1800-2700 | 3,521 | 329.7 | **3.5** | -83 | 1,589 | 1,984 |
| 2700-3600 | 346 | 582.3 | **1.1** | -238 | 1,751 | 2,944 |
| >3600 | 245 | 809.3 | **1.5** | -205 | 1,377 | 4,420 |
| nan | 4 | 262.4 | **0.0** | -189 | 786 | nan |

`span` is label-free — both `MVT_TIME_UTC_mvt` and `AOBT_3_flt` survive the ranking mask — and it is already a feature (`features.py` `d_takeoff_aobt`, and the model's residual target is built on it). The table therefore reads as *conditional* bias: what the model still gets wrong once it knows the span.

## By airport-day

279 airport-days.

| worst | days | cumulative SSE share |
|---|---|---|
| 1 % | 3 | **8.3 %** |
| 2 % | 6 | **12.3 %** |
| 5 % | 14 | **19.8 %** |
| 10 % | 28 | **29.5 %** |
| 20 % | 56 | **44.7 %** |
| 50 % | 140 | **75.4 %** |

**the fifteen worst july airport-days**

| airport | day | rows | SSE % | RMSE | mean error | mean y | median span | p90 span |
|---|---|---|---|---|---|---|---|---|
| EGLL | 2025-07-30 | 652 | 3.33 | 747 | +85 | 1,598 | 1,383 | 1,862 |
| EGLL | 2025-07-06 | 646 | 2.54 | 656 | +0 | 1,704 | 1,445 | 2,409 |
| LTFM | 2025-07-26 | 825 | 2.43 | 568 | +68 | 1,172 | 1,197 | 1,204 |
| LTFM | 2025-07-21 | 838 | 1.41 | 429 | +63 | 1,376 | 1,438 | 1,505 |
| EGLL | 2025-07-31 | 682 | 1.32 | 461 | +49 | 1,630 | 1,498 | 2,166 |
| LTFM | 2025-07-13 | 822 | 1.23 | 405 | +100 | 1,279 | 1,201 | 1,745 |
| EHAM | 2025-07-20 | 733 | 1.22 | 426 | +12 | 686 | 736 | 922 |
| LTFM | 2025-07-25 | 840 | 1.00 | 361 | +38 | 1,156 | 1,197 | 1,443 |
| LTFM | 2025-07-23 | 816 | 0.99 | 364 | +55 | 1,214 | 1,202 | 1,501 |
| EGLL | 2025-07-20 | 664 | 0.98 | 403 | -3 | 1,579 | 1,380 | 2,160 |
| LFPG | 2025-07-16 | 722 | 0.89 | 366 | +39 | 1,086 | 904 | 1,497 |
| LFPG | 2025-07-24 | 719 | 0.88 | 367 | +3 | 1,187 | 1,020 | 1,503 |
| LSZH | 2025-07-01 | 400 | 0.85 | 481 | +9 | 756 | 690 | 819 |
| LTFM | 2025-07-09 | 792 | 0.76 | 325 | +45 | 1,327 | 1,494 | 1,503 |
| LFPG | 2025-07-25 | 728 | 0.76 | 338 | +16 | 1,074 | 956 | 1,440 |

## What the model under-predicts

Under-predicted rows: 76,564 (43.6 %), 62.3 % of the SSE, mean error +175 s. Over-predicted: 98,938 (56.4 %), 37.7 %, mean -152 s. Whole-slice bias -9.3 s.

**the under-predicted half, by regime**

| reg2 | rows | RMSE | SSE % | mean error | mean y | median span |
|---|---|---|---|---|---|---|
| NM bulk | 72,512 | 242.6 | **39.0** | +158 | 1,164 | 956 |
| no-NM | 1,503 | 810.7 | **9.0** | +354 | 1,409 | nan |
| NM tail > 3600 s | 215 | 1,947.7 | **7.5** | +1,595 | 4,771 | 605 |
| block on LOBT/SCHED | 2,334 | 564.4 | **6.8** | +448 | 1,541 | 904 |

## What in the ranking frame marks it

Every column below is computable from the ranking-masked july frame alone. The score is
the fraction of the NM-bulk squared error a 20-bin function of that one column removes,
cross-fitted odd day / even day, *on top of the model's own prediction*, with the bin
means CENTRED so the correction is mean-zero. A column the model already uses fully
scores ~0, and so does pure noise -- see the floor below.

NM-bulk rows: 171,673, RMSE 227.3 s.

For reference, a cross-fitted CONSTANT correction -- the model's global bias, which every column would remove if the bin means were not centred -- is worth **0.23 % / 0.26 s** on this slice. That is the amount by which an uncentred version of this screen flatters every column, and it is why a Gaussian noise column scored +0.21 % before the fix.

| column | SSE removed | RMSE removed | Spearman with the residual |
|---|---|---|---|
| `sf_span` | 0.73 % | 0.83 s | -0.079 |
| `span` | 0.73 % | 0.83 s | -0.079 |
| `sf_overtaken` | 0.67 % | 0.76 s | -0.052 |
| `sf_q_apt` | 0.48 % | 0.55 s | -0.049 |
| `sf_q_arr_apt` | 0.39 % | 0.44 s | -0.037 |
| `sf_gap_next_rwy` | 0.31 % | 0.35 s | -0.067 |
| `sf_dep_rwy_post300` | 0.30 % | 0.34 s | +0.066 |
| `sf_q_rwy` | 0.29 % | 0.33 s | -0.022 |
| `sf_stand_ub_offset_clip` | 0.23 % | 0.26 s | -0.030 |
| `sf_stand_ub_offset` | 0.22 % | 0.25 s | -0.030 |
| `sf_stand_nxt_arr_dt` | 0.22 % | 0.25 s | -0.026 |
| `sf_arr_rwy_pre600` | 0.21 % | 0.24 s | +0.044 |
| `gap_al` | 0.21 % | 0.24 s | -0.001 |
| `sf_n10_rwy` | 0.20 % | 0.23 s | +0.057 |
| `sf_overtook` | 0.19 % | 0.21 s | +0.024 |
| `sf_arr_rwy_pre900` | 0.18 % | 0.21 s | +0.040 |
| `sf_stand_turnaround` | 0.16 % | 0.18 s | +0.020 |
| `sf_stand_prv_arr_dt` | 0.16 % | 0.18 s | +0.020 |
| `sf_stand_arr_in_span` | 0.15 % | 0.17 s | +0.017 |
| `sf_dep_rwy_post600` | 0.14 % | 0.16 s | +0.044 |
| `sf_q_arr_rwy` | 0.14 % | 0.16 s | +0.035 |
| `sf_stand_nxt_dep_off_dt` | 0.13 % | 0.15 s | -0.032 |
| `sf_trwy_before2` | 0.13 % | 0.15 s | -0.046 |
| `sf_stand_nxt_dep_push_dt` | 0.13 % | 0.14 s | -0.032 |
| `sf_arr_rwy_pre1800` | 0.13 % | 0.14 s | +0.035 |

(Full ranking of 99 columns: `work/july_markers.csv`.)
