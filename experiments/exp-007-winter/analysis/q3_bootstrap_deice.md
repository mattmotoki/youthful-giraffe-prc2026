# exp-007 question 3 — day-state columns, paired and clustered

Arms: `f300d-ds-s<seed>` against `f300d-base-s<seed>`, seeds [7], both
produced by `code/run_ds.py` on retro-v1's `G1cv-s7` configuration. A negative
number is the day-state arm winning.

## Fold numbers

| fold | seed | arm | RMSE | bulk ≤3600 | tail >3600 | no-NM | noac | rounds | features |
|---|---|---|---|---|---|---|---|---|---|
| july | 7 | ds  | 327.46 | 285.93 | 2736.17 | 1294.23 | 3722.53 | 300 | 97 |
| december | 7 | ds  | 224.88 | 207.93 | 2167.92 | 829.49 | 2721.60 | 300 | 97 |
| january | 7 | ds  | 347.79 | 213.15 | 5980.39 | 2847.89 | 3601.23 | 300 | 97 |

## Paired airport-day-clustered bootstrap, 2,000 draws

| fold | slice | rows | clusters | Δ RMSE (ds − base) | 95 % interval | calendar-day interval |
|---|---|---|---|---|---|---|
| july | whole fold (reachable) | 190,713 | 310 | **+0.58 s** | [-5.45, +7.57] | [-5.64, +8.08] |
| july | core | 186,779 | 310 | **+0.84 s** | [-6.34, +9.31] | [-5.99, +9.44] |
| july | core, other nine | 171,905 | 279 | **-0.34 s** | [-1.00, +0.34] | [-1.03, +0.34] |
| july | bulk ≤3600, other nine | 171,673 | 279 | **-0.34 s** | [-0.85, +0.17] | [-0.86, +0.18] |
| july | worst 5 % of airport-days (label-defined) | 9,771 | 14 | **-0.75 s** | [-5.38, +4.18] | [-5.41, +2.99] |
| july | most disrupted 5 % of days (label-free) | 6,808 | 12 | **-2.76 s** | [-7.78, +2.18] | [-8.80, +2.16] |
| december | whole fold (reachable) | 165,677 | 310 | **-13.62 s** | [-26.15, -3.01] | [-25.41, -2.62] |
| december | core | 164,034 | 310 | **-13.48 s** | [-27.18, -1.59] | [-27.02, -1.54] |
| december | core, other nine | 152,054 | 279 | **-5.19 s** | [-10.24, -1.71] | [-10.20, -1.77] |
| december | bulk ≤3600, other nine | 151,912 | 279 | **-4.66 s** | [-9.47, -1.67] | [-9.43, -1.76] |
| december | worst 5 % of airport-days (label-defined) | 8,176 | 14 | **-33.53 s** | [-76.85, -1.74] | [-81.38, -2.35] |
| december | most disrupted 5 % of days (label-free) | 7,583 | 18 | **-5.96 s** | [-11.80, -1.16] | [-11.38, -2.00] |
| january | whole fold (reachable) | 153,704 | 310 | **-3.92 s** | [-6.81, -1.02] | [-6.69, -1.31] |
| january | core | 152,274 | 310 | **-1.73 s** | [-3.99, +0.60] | [-3.82, +0.50] |
| january | core, other nine | 141,017 | 279 | **-1.01 s** | [-2.56, +0.97] | [-2.49, +0.94] |
| january | bulk ≤3600, other nine | 140,786 | 279 | **-0.82 s** | [-2.32, +1.29] | [-2.30, +1.09] |
| january | worst 5 % of airport-days (label-defined) | 8,349 | 14 | **-0.84 s** | [-7.43, +5.21] | [-7.02, +5.49] |
| january | most disrupted 5 % of days (label-free) | 7,768 | 19 | **-6.05 s** | [-12.23, +1.39] | [-13.58, +1.78] |
