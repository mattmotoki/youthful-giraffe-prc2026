# exp-007 question 3 — day-state columns, paired and clustered

Arms: `d1-ds-s<seed>` against `d1-base-s<seed>`, seeds [7], both
produced by `code/run_ds.py` on retro-v1's `G1cv-s7` configuration. A negative
number is the day-state arm winning.

## Fold numbers

| fold | seed | arm | RMSE | bulk ≤3600 | tail >3600 | no-NM | noac | rounds | features |
|---|---|---|---|---|---|---|---|---|---|
| july | 7 | base | 321.85 | 277.85 | 2783.73 | 1230.48 | 3711.23 | 50 | 96 |
| december | 7 | base | 236.88 | 219.42 | 2259.29 | 862.23 | 2707.54 | 235 | 96 |
| january | 7 | base | 350.91 | 216.62 | 6007.62 | 2874.51 | 3660.93 | 284 | 96 |
| july | 7 | ds  | 315.40 | 272.66 | 2716.63 | 1225.06 | 3684.26 | 107 | 121 |
| december | 7 | ds  | 222.24 | 204.74 | 2187.45 | 806.97 | 2688.08 | 207 | 121 |
| january | 7 | ds  | 351.98 | 217.99 | 6013.88 | 2860.96 | 3790.23 | 96 | 121 |

## Paired airport-day-clustered bootstrap, 2,000 draws

| fold | slice | rows | clusters | Δ RMSE (ds − base) | 95 % interval | calendar-day interval |
|---|---|---|---|---|---|---|
| july | whole fold (reachable) | 190,713 | 310 | **-6.45 s** | [-9.72, -3.24] | [-9.41, -3.61] |
| july | core | 186,779 | 310 | **-7.31 s** | [-9.75, -4.44] | [-9.65, -4.50] |
| july | core, other nine | 171,905 | 279 | **-10.64 s** | [-11.71, -9.51] | [-11.42, -9.71] |
| july | bulk ≤3600, other nine | 171,673 | 279 | **-10.43 s** | [-11.32, -9.57] | [-11.03, -9.83] |
| july | worst 5 % of airport-days (label-defined) | 10,026 | 14 | **-11.92 s** | [-17.25, -4.67] | [-17.37, -5.78] |
| july | most disrupted 5 % of days (label-free) | 6,808 | 12 | **-6.99 s** | [-13.73, +0.70] | [-14.36, +0.97] |
| december | whole fold (reachable) | 165,677 | 310 | **-14.64 s** | [-26.73, -4.62] | [-28.06, -4.41] |
| december | core | 164,034 | 310 | **-13.63 s** | [-26.61, -2.89] | [-27.74, -3.08] |
| december | core, other nine | 152,054 | 279 | **-1.80 s** | [-4.66, +0.86] | [-4.39, +0.49] |
| december | bulk ≤3600, other nine | 151,912 | 279 | **-1.22 s** | [-3.11, +1.05] | [-2.80, +0.56] |
| december | worst 5 % of airport-days (label-defined) | 7,921 | 14 | **-12.80 s** | [-32.45, +7.49] | [-31.63, +6.29] |
| december | most disrupted 5 % of days (label-free) | 7,583 | 18 | **-11.02 s** | [-18.76, -3.31] | [-21.05, -2.48] |
| january | whole fold (reachable) | 153,704 | 310 | **+2.02 s** | [-2.39, +7.62] | [-2.81, +8.23] |
| january | core | 152,274 | 310 | **+3.43 s** | [-1.74, +10.56] | [-1.90, +11.21] |
| january | core, other nine | 141,017 | 279 | **+6.54 s** | [+1.91, +13.45] | [+1.59, +14.56] |
| january | bulk ≤3600, other nine | 140,786 | 279 | **+5.75 s** | [+1.13, +12.98] | [+1.09, +13.16] |
| january | worst 5 % of airport-days (label-defined) | 8,349 | 14 | **+3.41 s** | [-4.27, +10.60] | [-4.07, +11.34] |
| january | most disrupted 5 % of days (label-free) | 7,768 | 19 | **+47.41 s** | [-0.59, +113.38] | [-5.95, +109.77] |
