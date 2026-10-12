# Draft ledger rows for exp-004 (append to docs/LEDGER.md when final)

Filled from `ablation_<fold>.json` / `ablation_<fold>_v2.json`. Every number
uses the live 2026-09-04 data through `prc.data.fold_frames`.

| date | exp | worker | mechanism | july | dec | july tail>3600 | july no-NM | artifact | reviewer | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-05 | exp-004 base | congestion | frozen base: own-flight deltas + airport/stand/runway/type/operator/hour, CatBoost 3000x0.05 d8 | 588.5 | 317.5 | 8455 | 3412 | experiments/exp-004-congestion/RESULTS.md | pending | january: PENDING; NM bulk 249.0 |
| 2026-09-05 | exp-004 oracle | congestion | ceiling instrument: + true same-runway/airport take-off counts between true block time and take-off (illegal) | 271.6 | 193.9 | 3590 | 985 | experiments/exp-004-congestion/oracle_bound.json | pending | loose bound: the count is a quantised function of the target |
| 2026-09-05 | exp-004 surface | congestion | + surface-state families (PENDING recommended set) | – | – | – | – | – | pending | – |
