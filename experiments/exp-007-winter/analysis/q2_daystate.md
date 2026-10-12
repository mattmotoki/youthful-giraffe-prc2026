# exp-007 question 2 — what marks the worst airport-days, label-free

837 airport-days over the three folds (non-LIRF, the two unreachable rows
removed). Every candidate comes from the ranking-masked month frame; the target
(the day's RMSE under the baseline out-of-fold line) is label-defined and is never
an input. `rho within airport-fold` ranks days inside each (airport, fold), so a
column can score neither by telling the airports apart -- the model already knows
the airport -- nor by tracking the season, which pooling within the airport alone
would have let it do. `AUC` is for the worst 5 % of days, on the same
within-(airport, fold) ranks, under TWO labels: the fold's own worst 5 % of
airport-days by squared error, and each airport's own worst two of its 31 days.
The first version of this table computed the first and described the second.
The per-fold rho columns are plain correlations across that fold's days and DO
carry the between-airport contrast.

| column | rho within airport-fold | AUC, fold's worst 5 % | AUC, each airport's worst 2 of 31 | rho january | rho december | rho july | missing days |
|---|---|---|---|---|---|---|---|
| `ds_sched_med` | **+0.651** | 0.868 | 0.854 | +0.642 | +0.622 | +0.441 | 0 |
| `ds_gap_p90` | **+0.637** | 0.856 | 0.868 | +0.377 | +0.403 | +0.350 | 0 |
| `ds_sched_p90` | **+0.614** | 0.841 | 0.852 | +0.611 | +0.631 | +0.492 | 0 |
| `ds_gap_med` | **+0.556** | 0.838 | 0.830 | +0.215 | +0.183 | +0.070 | 0 |
| `ds_span_std` | **+0.497** | 0.808 | 0.827 | +0.451 | +0.568 | +0.481 | 0 |
| `ds_span_p99` | **+0.415** | 0.791 | 0.739 | +0.593 | +0.473 | +0.377 | 0 |
| `ds_span_p90` | **+0.413** | 0.781 | 0.694 | +0.613 | +0.576 | +0.437 | 0 |
| `ds_nm_rate` | **-0.406** | 0.241 | 0.238 | -0.414 | -0.320 | -0.316 | 0 |
| `ds_span_iqr` | **+0.395** | 0.759 | 0.705 | +0.152 | +0.272 | +0.252 | 0 |
| `ds_share_span1800` | **+0.375** | 0.759 | 0.733 | +0.592 | +0.504 | +0.311 | 0 |
| `ds_span_mean` | **+0.360** | 0.786 | 0.715 | +0.531 | +0.452 | +0.405 | 0 |
| `ds_span_med` | **+0.327** | 0.770 | 0.704 | +0.441 | +0.388 | +0.368 | 0 |
| `ds_taxiin_p90` | **+0.299** | 0.792 | 0.701 | +0.496 | +0.243 | +0.297 | 0 |
| `ds_share_span2700` | **+0.269** | 0.728 | 0.693 | +0.565 | +0.362 | +0.310 | 0 |
| `ds_taxiin_med` | **+0.218** | 0.725 | 0.676 | +0.386 | +0.177 | +0.197 | 0 |
| `ds_wx_temp_mean` | **-0.187** | 0.433 | 0.391 | -0.247 | -0.035 | +0.204 | 0 |
| `ds_wx_subzero_share` | **+0.178** | 0.616 | 0.627 | +0.241 | +0.133 | +nan | 0 |
| `ds_span_med_prev` | **+0.175** | 0.505 | 0.572 | +0.406 | +0.360 | +0.356 | 27 |
| `ds_wx_precip_sum` | **+0.131** | 0.588 | 0.584 | -0.021 | +0.108 | +0.051 | 0 |
| `ds_span_med_delta` | **+0.117** | 0.754 | 0.621 | +0.054 | +0.145 | +0.065 | 27 |
| `ds_wx_snow6_max` | **+0.101** | 0.557 | 0.614 | +0.185 | +0.083 | +nan | 0 |
| `ds_wx_dewdep_min` | **-0.100** | 0.450 | 0.485 | -0.154 | +0.052 | -0.089 | 0 |
| `ds_wx_temp_min` | **-0.081** | 0.474 | 0.474 | -0.192 | -0.144 | +0.193 | 0 |
| `ds_wx_freezing_share` | **+0.074** | 0.551 | 0.590 | +0.165 | +0.123 | +nan | 0 |
| `ds_wx_wind_max` | **+0.060** | 0.576 | 0.550 | -0.042 | +0.098 | +0.072 | 0 |
| `ds_noac_rate` | **+0.045** | 0.458 | 0.532 | +0.154 | +0.067 | +0.107 | 0 |
| `ds_rwy_modal_share_prev` | **+0.037** | 0.498 | 0.506 | -0.063 | -0.086 | -0.008 | 27 |
| `ds_rwy_n` | **+0.036** | 0.488 | 0.536 | -0.064 | -0.180 | -0.121 | 0 |
| `ds_rwy_modal_share` | **+0.032** | 0.508 | 0.506 | -0.133 | -0.043 | +0.040 | 0 |
| `ds_rwy_modal_changed` | **+0.024** | 0.547 | 0.511 | +0.018 | +0.038 | +0.012 | 27 |
| `ds_n_arr` | **-0.015** | 0.522 | 0.466 | +0.435 | +0.298 | +0.247 | 0 |
| `ds_mvt_total` | **-0.009** | 0.533 | 0.469 | +0.436 | +0.301 | +0.249 | 0 |
| `ds_mvt_shortfall` | **-0.009** | 0.533 | 0.469 | +0.033 | -0.016 | -0.113 | 0 |
| `ds_n_dep` | **-0.006** | 0.543 | 0.466 | +0.433 | +0.299 | +0.254 | 0 |
| `ds_rwy_entropy` | **-0.002** | 0.487 | 0.531 | +0.060 | -0.048 | -0.139 | 0 |
