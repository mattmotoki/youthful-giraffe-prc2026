# Q1(f) — six training months against eleven, same family, same rows

9 families have OOF for both folds; **4 are excluded** because their six-month file is byte-identical to another family's, so the pair would compare two different models (review round 1, B4).

| model | n | 6 months | 11 months | gain | usable | note |
|---|---|---|---|---|---|---|
| C-cat | 186276 | 241.05 | 236.14 | 4.91 | False | six-month file is byte-identical to C1-d8 |
| C1-d8 | 186276 | 241.05 | 251.44 | -10.39 | False | six-month file is byte-identical to C-cat |
| L0-lin | 186276 | 241.89 | 249.07 | -7.18 | True |  |
| L1-plain | 186276 | 245.49 | 236.08 | 9.41 | True |  |
| L2-hub | 186276 | 245.28 | 235.51 | 9.76 | True |  |
| L3-monoraw | 186276 | 245.60 | 237.71 | 7.89 | True |  |
| L3r-monores | 186276 | 246.67 | 236.29 | 10.38 | True |  |
| X-xgb | 186276 | 253.40 | 243.51 | 9.89 | False | six-month file is byte-identical to X1-d8 |
| X1-d8 | 186276 | 253.40 | 247.68 | 5.72 | False | six-month file is byte-identical to X-xgb |

Mean gain over the 5 usable families: **+6.05 s** (median +9.41, range -7.18 to +10.38). Including the two contaminated rows would give +4.49 s.
