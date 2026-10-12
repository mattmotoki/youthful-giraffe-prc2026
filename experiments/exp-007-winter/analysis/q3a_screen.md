# exp-007 question 3a — what the day-state columns add on top of the model

The paired arms in `run_ds.py` answer the question the assignment asks. This
screen answers a cheaper, sharper version of it first, and it is the same
instrument `q2j_july.md` used on the surface columns: for each day-state column,
the fraction of the baseline's NM-bulk squared error that a 20-bin function of
that column removes ON TOP of the model's prediction, cross-fitted odd day /
even day so no row's own residual sets its own correction.

A column the model already reconstructs from the row's own values scores ~0 here
even if `q2_daystate.md` showed it ranks bad days well -- which is exactly the
distinction finding 5 draws between marking a day and improving a prediction.

| fold | column | SSE removed | RMSE removed | Spearman with the residual |
|---|---|---|---|---|
| january | `ds_deice_hr_n` | 0.01 % | 0.01 s | -0.012 |
| january | `ds_span_p90_z` | -0.02 % | -0.02 s | +0.015 |
| january | `ds_deice_hr` | -0.03 % | -0.03 s | +0.029 |
| january | `ds_sched_p90_z` | -0.09 % | -0.08 s | -0.036 |
| january | `ds_share_span1800_z` | -0.11 % | -0.11 s | +0.023 |
| january | `ds_span_med_z` | -0.15 % | -0.14 s | -0.012 |
| january | *(best of the other 19)* | -0.18 % | -0.18 s | |
| december | `ds_deice_hr` | 0.15 % | 0.14 s | +0.024 |
| december | `ds_share_span1800_z` | 0.03 % | 0.03 s | -0.012 |
| december | `ds_taxiin_p90_z` | 0.00 % | 0.00 s | +0.004 |
| december | `ds_deice_hr_n` | -0.00 % | -0.00 s | -0.008 |
| december | `ds_taxiin_p90` | -0.02 % | -0.02 s | -0.009 |
| december | `ds_span_p90_z` | -0.03 % | -0.03 s | -0.001 |
| december | *(best of the other 19)* | -0.05 % | -0.05 s | |
| july | `ds_deice_hr` | 0.13 % | 0.14 s | -0.000 |
| july | `ds_mvt_total` | 0.06 % | 0.07 s | -0.027 |
| july | `ds_deice_day` | 0.06 % | 0.07 s | -0.008 |
| july | `ds_share_span1800` | 0.04 % | 0.05 s | -0.009 |
| july | `ds_n_dep` | 0.03 % | 0.03 s | -0.027 |
| july | `ds_share_span1800_z` | -0.03 % | -0.03 s | -0.001 |
| july | *(best of the other 19)* | -0.03 % | -0.04 s | |

The bin means are centred, so a column can only score on VARIATION. For scale, a cross-fitted CONSTANT correction -- the model's global bias -- is worth 0.09 % on january, 0.17 % on december, 0.23 % on july: that is what an uncentred screen would have added to every column, noise included.

And for scale against the rest of the ranking frame, the same centred screen over the 99 surface and own-flight columns put the best at 0.73 % of the july NM-bulk squared error (`analysis/q2j_july.md`), and that column was `span`, which the model already holds.

## Interaction probe

The screen above is additive and univariate. This one fits a mean residual per
(own-value bin, day-state bin) cell on the odd days and applies it to the even days,
and the reverse: it can see a day-state column used as a *modifier* of the row's own
value, which is the only way the paired arms could beat the univariate screen.

| fold | own value | day state | SSE removed | RMSE removed |
|---|---|---|---|---|
| january | `span` | **control: `ds_span_std_z` shuffled** | 0.02 % | 0.02 s |
| january | `span` | `ds_span_std_z` | 0.01 % | 0.01 s |
| january | `span` | `ds_share_span1800_z` | -0.00 % | -0.00 s |
| january | `span` | `ds_deice_hr` | 0.16 % | 0.16 s |
| january | `gap_al` | `ds_gap_p90_z` | -0.53 % | -0.51 s |
| january | `span_sched` | `ds_sched_med_z` | -0.33 % | -0.32 s |
| january | `span` | `ds_nm_rate_z` | -0.33 % | -0.32 s |
| december | `span` | **control: `ds_span_std_z` shuffled** | -0.08 % | -0.07 s |
| december | `span` | `ds_span_std_z` | -0.32 % | -0.30 s |
| december | `span` | `ds_share_span1800_z` | -0.19 % | -0.18 s |
| december | `span` | `ds_deice_hr` | 0.10 % | 0.09 s |
| december | `gap_al` | `ds_gap_p90_z` | -0.13 % | -0.12 s |
| december | `span_sched` | `ds_sched_med_z` | -0.05 % | -0.04 s |
| december | `span` | `ds_nm_rate_z` | -0.33 % | -0.31 s |
| july | `span` | **control: `ds_span_std_z` shuffled** | 0.54 % | 0.62 s |
| july | `span` | `ds_span_std_z` | 0.41 % | 0.47 s |
| july | `span` | `ds_share_span1800_z` | 0.61 % | 0.69 s |
| july | `span` | `ds_deice_hr` | 0.85 % | 0.97 s |
| july | `gap_al` | `ds_gap_p90_z` | 0.20 % | 0.22 s |
| july | `span_sched` | `ds_sched_med_z` | -0.04 % | -0.05 s |
| july | `span` | `ds_nm_rate_z` | 0.14 % | 0.16 s |
