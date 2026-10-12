# Q6 — stand occupancy as a bound on the hidden block time

A stand holds one aircraft at a time and arrivals keep their on-block time in the ranking frame, so an arrival at stand S bounds the departure's hidden off-block `B`. In taxi terms `taxi = MVT - B`, so an arrival that parks *after* B gives a **lower bound on taxi** and an arrival that parked *before* B gives an **upper bound**. The anchors below differ in how 'the next arrival' is defined without knowing `B`.

## Validity and tightness, measured on labelled months

`coverage` = share of departures the bound is defined on; `valid` = share of those it does not violate; `slack` = bound minus truth in seconds (>= 0 when the bound holds).

| month | stand key | bound | coverage | valid | slack_p10 | slack_p50 | slack_p90 | n |
|---|---|---|---|---|---|---|---|---|
| 2025-07 | key_raw | taxi_ub_pm | 0.9937 | 0.8853 | -303.0000 | 5045.0000 | 30297.0000 | 189504 |
| 2025-07 | key_raw | taxi_ub_pa | 0.9730 | 0.9913 | 3057.0000 | 5621.0000 | 31239.0000 | 185570 |
| 2025-07 | key_raw | taxi_lb_nm | 0.9911 | 1.0000 | 1681.0000 | 6831.5000 | 44759.7000 | 189014 |
| 2025-07 | key_raw | taxi_lb_na | 0.9714 | 0.9995 | 960.0000 | 4980.0000 | 41695.0000 | 185261 |
| 2025-07 | key_base | taxi_ub_pm | 0.9938 | 0.8778 | -361.0000 | 4883.0000 | 29937.0000 | 189529 |
| 2025-07 | key_base | taxi_ub_pa | 0.9732 | 0.9906 | 2942.0000 | 5460.0000 | 30959.0000 | 185594 |
| 2025-07 | key_base | taxi_lb_nm | 0.9917 | 1.0000 | 1647.0000 | 6461.0000 | 42041.0000 | 189129 |
| 2025-07 | key_base | taxi_lb_na | 0.9720 | 0.9972 | 902.0000 | 4729.0000 | 39234.0000 | 185369 |
| 2025-12 | key_raw | taxi_ub_pm | 0.9937 | 0.9074 | 2165.0000 | 5285.0000 | 34863.5000 | 164626 |
| 2025-12 | key_raw | taxi_ub_pa | 0.9836 | 0.9951 | 3058.2000 | 5875.0000 | 35817.0000 | 162953 |
| 2025-12 | key_raw | taxi_lb_nm | 0.9906 | 1.0000 | 1798.0000 | 7165.0000 | 47092.0000 | 164126 |
| 2025-12 | key_raw | taxi_lb_na | 0.9814 | 0.9997 | 1134.9000 | 5636.5000 | 44339.0000 | 162590 |
| 2025-12 | key_base | taxi_ub_pm | 0.9939 | 0.9012 | 117.9000 | 5098.0000 | 34496.2000 | 164660 |
| 2025-12 | key_base | taxi_ub_pa | 0.9837 | 0.9948 | 2946.0000 | 5690.0000 | 35468.0000 | 162983 |
| 2025-12 | key_base | taxi_lb_nm | 0.9914 | 1.0000 | 1743.0000 | 6782.0000 | 44224.3000 | 164258 |
| 2025-12 | key_base | taxi_lb_na | 0.9821 | 0.9973 | 1069.0000 | 5318.0000 | 41889.2000 | 162709 |
| 2025-01 | key_raw | taxi_ub_pm | 0.9918 | 0.9172 | 2396.0000 | 5455.0000 | 36119.0000 | 152453 |
| 2025-01 | key_raw | taxi_ub_pa | 0.9825 | 0.9961 | 3004.0000 | 6007.0000 | 37028.0000 | 151021 |
| 2025-01 | key_raw | taxi_lb_nm | 0.9891 | 1.0000 | 1871.0000 | 7628.0000 | 51296.5000 | 152030 |
| 2025-01 | key_raw | taxi_lb_na | 0.9802 | 0.9997 | 1204.0000 | 6185.0000 | 48661.0000 | 150666 |
| 2025-01 | key_base | taxi_ub_pm | 0.9920 | 0.9110 | 1561.0000 | 5222.0000 | 35643.6000 | 152475 |
| 2025-01 | key_base | taxi_ub_pa | 0.9827 | 0.9959 | 2937.0000 | 5820.0000 | 36656.1000 | 151040 |
| 2025-01 | key_base | taxi_lb_nm | 0.9899 | 1.0000 | 1843.0000 | 7252.0000 | 48244.6000 | 152158 |
| 2025-01 | key_base | taxi_lb_na | 0.9810 | 0.9969 | 1140.0000 | 5878.0000 | 45482.0000 | 150789 |

Reading: `taxi_lb_nm` (first arrival on-block after the departure's *take-off*) is a bound by construction and never violated. `taxi_lb_na` (first arrival after the departure's *AOBT*) is tighter and violated on 0.05 % of rows with the raw stand name and 0.3 % with the MARS parent name — the parent key is wrong here because two aircraft legitimately occupy 519L and 519R at once. `taxi_ub_pm` (last arrival before take-off) is violated on 12 % of rows: at a busy stand the next aircraft frequently parks between our off-block and our take-off, which is the phenomenon the bound is meant to catch, not a bound at all. **Use `key_raw`, use the next-arrival bounds as hard bounds, and use the previous-arrival ones as features only.**

## What the bound is worth: clipping a predictor at `taxi >= MVT - A_next`

Predictor is the reference naive one (`taxi = MVT - AOBT_3 - stand median offset`, stand medians fitted on the fold's training months). This isolates the bound's value from the model's.

| fold | bound | segment | n | binds % | RMSE before | RMSE after clip | gain |
|---|---|---|---|---|---|---|---|
| july | taxi_lb_na (after AOBT) | all | 190713 | 0.23 | 710.41 | 710.25 | 0.16 |
| july | taxi_lb_na (after AOBT) | bulk (NM, y<=3600) | 186272 | 0.24 | 401.23 | 400.95 | 0.28 |
| july | taxi_lb_na (after AOBT) | y>3600 | 656 | 0.00 | 9998.05 | 9998.05 | 0.00 |
| july | taxi_lb_na (after AOBT) | no NM | 3938 | 0.00 | 3855.85 | 3855.85 | 0.00 |
| july | taxi_lb_nm (after take-off) | all | 190713 | 0.02 | 710.41 | 706.06 | 4.35 |
| july | taxi_lb_nm (after take-off) | bulk (NM, y<=3600) | 186272 | 0.03 | 401.23 | 393.30 | 7.93 |
| july | taxi_lb_nm (after take-off) | y>3600 | 656 | 0.00 | 9998.05 | 9998.05 | 0.00 |
| july | taxi_lb_nm (after take-off) | no NM | 3938 | 0.00 | 3855.85 | 3855.85 | 0.00 |
| december | taxi_lb_na (after AOBT) | all | 165677 | 0.11 | 468.29 | 468.21 | 0.08 |
| december | taxi_lb_na (after AOBT) | bulk (NM, y<=3600) | 163838 | 0.12 | 332.94 | 332.83 | 0.10 |
| december | taxi_lb_na (after AOBT) | y>3600 | 261 | 0.38 | 8225.70 | 8225.52 | 0.18 |
| december | taxi_lb_na (after AOBT) | no NM | 1649 | 0.00 | 3159.91 | 3159.91 | 0.00 |
| december | taxi_lb_nm (after take-off) | all | 165677 | 0.02 | 468.29 | 465.16 | 3.13 |
| december | taxi_lb_nm (after take-off) | bulk (NM, y<=3600) | 163838 | 0.02 | 332.94 | 328.47 | 4.46 |
| december | taxi_lb_nm (after take-off) | y>3600 | 261 | 0.00 | 8225.70 | 8225.70 | 0.00 |
| december | taxi_lb_nm (after take-off) | no NM | 1649 | 0.00 | 3159.91 | 3159.91 | 0.00 |
| january | taxi_lb_na (after AOBT) | all | 153706 | 0.09 | 538.79 | 538.72 | 0.06 |
| january | taxi_lb_na (after AOBT) | bulk (NM, y<=3600) | 151999 | 0.09 | 325.76 | 325.65 | 0.11 |
| january | taxi_lb_na (after AOBT) | y>3600 | 325 | 0.00 | 9291.71 | 9291.71 | 0.00 |
| january | taxi_lb_na (after AOBT) | no NM | 1435 | 0.00 | 4295.13 | 4295.13 | 0.00 |
| january | taxi_lb_nm (after take-off) | all | 153706 | 0.01 | 538.79 | 535.41 | 3.38 |
| january | taxi_lb_nm (after take-off) | bulk (NM, y<=3600) | 151999 | 0.01 | 325.76 | 320.07 | 5.69 |
| january | taxi_lb_nm (after take-off) | y>3600 | 325 | 0.00 | 9291.71 | 9291.71 | 0.00 |
| january | taxi_lb_nm (after take-off) | no NM | 1435 | 0.00 | 4295.13 | 4295.13 | 0.00 |

## Does the stand really hold one aircraft at a time?

The bound above assumes occupancy alternates arrival, departure, arrival. Tested directly with the true block times: for every pair of consecutive arrivals at a stand, how many departure off-blocks fall between them?

| month | stand key | arrival intervals | exactly 1 departure % | 0 departures % | 2+ departures % |
|---|---|---|---|---|---|
| 2025-01 | key_raw | 151209 | 87.36 | 6.90 | 5.74 |
| 2025-01 | key_base | 151361 | 82.02 | 9.60 | 8.38 |
| 2025-07 | key_raw | 188065 | 88.53 | 6.18 | 5.28 |
| 2025-07 | key_base | 188213 | 83.56 | 8.69 | 7.74 |
| 2025-12 | key_raw | 163457 | 87.86 | 6.60 | 5.54 |
| 2025-12 | key_base | 163610 | 82.55 | 9.28 | 8.17 |

**It alternates about 88 % of the time, not 100 %.** In 6 % of the gaps between two arrivals no aircraft departs (the aircraft is towed away, or stays overnight) and in 5 % two or more depart (the stand name covers more than one physical position, or an aircraft is towed *on*). An 11 % failure rate rules out the counting construction as a hard clip: a bound built by counting arrivals between two departures is valid on only 74 % of the rows it fires on (measured, `disp_lb_1`, July 2025), and its second-order version on 0.5 %. It is kept in `prc.stand_features.displacement_bounds` as a *feature*, never as a clip.

## Clipping: the occupancy bound against clipping at zero

Reference naive predictor, bulk RMSE (NM join, taxi <= 3600 s):

| fold | negative raw predictions | no clip | clip at 0 | clip at the next-arrival-after-take-off bound | clip at 0, then at the next-arrival-after-AOBT bound |
|---|---|---|---|---|---|
| july | 413 | 401.23 | 376.64 | 393.30 | 376.34 |
| december | 192 | 332.94 | 319.98 | 328.47 | 319.90 |
| january | 127 | 325.76 | 308.64 | 320.07 | 308.53 |

Clipping at zero recovers 24 s of bulk RMSE on the july fold; the occupancy bound on top of that recovers a further 0.3 s. As a *clip* the bound is worth nothing that `max(pred, 0)` does not already give. Its value is as a **feature**, which the ablation table measures.
