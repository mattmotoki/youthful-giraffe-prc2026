# exp-007 question 3 — day-state columns, paired and clustered

Arms: `f300-ds-s<seed>` against `f300-base-s<seed>`, seeds [7], both
produced by `code/run_ds.py` on retro-v1's `G1cv-s7` configuration. A negative
number is the day-state arm winning.

## Fold numbers

| fold | seed | arm | RMSE | bulk ≤3600 | tail >3600 | no-NM | noac | rounds | features |
|---|---|---|---|---|---|---|---|---|---|
| july | 7 | base | 326.88 | 284.41 | 2761.97 | 1295.53 | 3711.23 | 300 | 96 |
| july | 7 | ds  | 327.14 | 285.64 | 2733.94 | 1283.37 | 3684.26 | 300 | 121 |
| december | 7 | base | 238.50 | 221.28 | 2253.09 | 860.70 | 2707.54 | 300 | 96 |
| december | 7 | ds  | 221.88 | 204.62 | 2171.21 | 804.06 | 2688.08 | 300 | 121 |
| january | 7 | base | 350.80 | 216.58 | 6005.31 | 2873.54 | 3660.93 | 300 | 96 |
| january | 7 | ds  | 355.14 | 221.95 | 6033.11 | 2855.73 | 3790.23 | 300 | 121 |

## Paired airport-day-clustered bootstrap, 2,000 draws

| fold | slice | rows | clusters | Δ RMSE (ds − base) | 95 % interval | calendar-day interval |
|---|---|---|---|---|---|---|
| july | whole fold (reachable) | 190,713 | 310 | **+0.25 s** | [-3.07, +3.51] | [-2.71, +3.06] |
| july | core | 186,779 | 310 | **+1.52 s** | [-3.29, +6.13] | [-3.06, +6.10] |
| july | core, other nine | 171,905 | 279 | **-0.64 s** | [-1.66, +0.62] | [-1.69, +0.51] |
| july | bulk ≤3600, other nine | 171,673 | 279 | **-1.49 s** | [-2.31, -0.73] | [-2.17, -0.81] |
| july | worst 5 % of airport-days (label-defined) | 9,771 | 14 | **-1.44 s** | [-10.34, +8.97] | [-10.83, +8.98] |
| july | most disrupted 5 % of days (label-free) | 6,808 | 12 | **-0.97 s** | [-5.78, +4.21] | [-7.03, +4.97] |
| december | whole fold (reachable) | 165,677 | 310 | **-16.62 s** | [-29.61, -6.02] | [-29.36, -6.06] |
| december | core | 164,034 | 310 | **-15.72 s** | [-28.84, -4.61] | [-28.73, -5.46] |
| december | core, other nine | 152,054 | 279 | **-4.11 s** | [-9.66, +0.19] | [-9.30, -0.16] |
| december | bulk ≤3600, other nine | 151,912 | 279 | **-3.60 s** | [-9.03, +0.27] | [-8.88, -0.10] |
| december | worst 5 % of airport-days (label-defined) | 8,176 | 14 | **-30.68 s** | [-77.98, +6.60] | [-83.81, +5.26] |
| december | most disrupted 5 % of days (label-free) | 7,583 | 18 | **-10.79 s** | [-20.02, -2.10] | [-22.03, -0.74] |
| january | whole fold (reachable) | 153,704 | 310 | **+6.87 s** | [-1.70, +18.68] | [-2.81, +18.88] |
| january | core | 152,274 | 310 | **+9.18 s** | [-1.33, +23.71] | [-1.59, +23.05] |
| january | core, other nine | 141,017 | 279 | **+5.87 s** | [-0.54, +14.49] | [-1.65, +15.19] |
| january | bulk ≤3600, other nine | 140,786 | 279 | **+2.84 s** | [-1.87, +10.02] | [-2.11, +11.18] |
| january | worst 5 % of airport-days (label-defined) | 8,349 | 14 | **+2.67 s** | [-5.82, +9.94] | [-7.10, +10.33] |
| january | most disrupted 5 % of days (label-free) | 7,768 | 19 | **+72.09 s** | [+3.46, +149.73] | [-3.74, +148.27] |
