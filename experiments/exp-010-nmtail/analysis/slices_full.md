# The four COMMON.md slices, plus the NM tail, for every head run

`S2` is at `prc.nmtail` VERSION 2's defaults (900 rounds, lr 0.04, 63 leaves), which is what the
reproduction command runs. `S0` is 600/0.05/63 and `S1` 1500/0.03/127, both superseded.
The head predicts NM rows; no-NM rows take the training airport median taxi, so the
`RMSE no-NM` and `RMSE all` columns describe that fallback, not a model.

| run | mode | fold | RMSE all | RMSE <=3600 | RMSE >3600 | RMSE no-NM | RMSE bulk | **RMSE NM tail** | n |
|---|---|---|---|---|---|---|---|---|---|
| S0-phi | phi | july_loo | 616.0 | 259.2 | 9531 | 3855 | 254.9 | **1858** | 503 |
| S1-phi | phi | july_loo | 614.9 | 257.0 | 9529 | 3855 | 252.6 | **1840** | 503 |
| S2-p | p | july_loo | 636.0 | 298.3 | 9583 | 3855 | 295.3 | **2177** | 503 |
| S2-phi | phi | december | 388.2 | 223.8 | 7994 | 3160 | 217.8 | **1954** | 190 |
| S2-phi | phi | january | 474.1 | 226.9 | 9056 | 4295 | 221.6 | **1501** | 272 |
| S2-phi | phi | july | 619.9 | 266.6 | 9546 | 3856 | 262.4 | **1951** | 503 |
| S2-phi_screened | phi_screened | july_loo | 615.9 | 258.4 | 9537 | 3855 | 254.0 | **1895** | 503 |
