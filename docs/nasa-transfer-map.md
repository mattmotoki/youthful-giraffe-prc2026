# NASA-to-PRC transfer map

This note converts the prior NASA airport-throughput solution into testable PRC
2026 hypotheses. It is a map for new implementation, not permission to reuse a
feature without checking its timestamp and license.

## Task boundary

The NASA solution predicts aggregate arrival throughput in 15-minute intervals.
PRC 2026 predicts flight-level departure taxi-out seconds. The PRC ranking data
blank departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`; other movement and
flight columns remain present. The implementation must derive every feature from
columns supplied in the ranking data or from documented open data.

Official PRC sources:

- [Data and columns](https://ansperformance.eu/study/data-challenge/dc2026/data.html)
- [Ranking and submission format](https://ansperformance.eu/study/data-challenge/dc2026/ranking.html)

## Transfer candidates

| Prior mechanism | Prior source | PRC 2026 adaptation | First test |
|---|---|---|---|
| Arrival and departure count windows | `Feature Engineering 1 - Arrival Count Lags and Leads v2.1.ipynb`; departure counterpart | Per-airport counts in fixed windows around each departure movement time, split by phase and optionally runway | Compare airport/time baseline with and without count windows on held-out months |
| Outstanding scheduled arrivals | `Feature Engineering 1 - Number of Outstanding Scheduled Arrivals v1.0.ipynb` | Scheduled arrivals due by the target departure time but not yet recorded as landed | Add one count and age summary to a fixed baseline |
| Schedule and estimate errors | `Feature Engineering 1 - Scheduling and Estimation Errors v1.0.ipynb` | Recent airport-level deviations between scheduled and actual movement times; flight-level differences among IOBT, EOBT, LOBT, and available actual movement times | Test recent-window medians and quantiles with strict row-time construction |
| Time-of-week residuals | `Feature Engineering 1 - Time of Week Residuals v1.1.ipynb` | Hierarchical taxi-out priors by airport, local time, weekday, runway, stand, operator, aircraft, route, and market segment | Compare smoothed group priors with plain airport medians |
| METAR and forecast weather | METAR, TAF, and LAMP notebooks | Join open observations or forecasts by airport and timestamp; document the external source and license | Start with visibility, wind, precipitation, ceiling, and adverse-weather flags |
| Horizon-aware modeling | `Feature Engineering 1 - Horizon v1.0 .ipynb` | Replace NASA forecast horizon with a precise PRC availability or data-quality state, if one exists after data audit | Use only if the field has a clear ranking-time analogue |
| Diverse tabular models | LightGBM, XGBoost, and MLP notebooks | Train strong tree baselines first; add a neural model only when it supplies measured residual diversity | Time-split out-of-fold RMSE and residual correlation |
| Weighted blend | `Modeling 1 - Blend.ipynb` | Blend only out-of-fold predictions from structurally different lineages | Optimize weights on frozen folds, then confirm on a later holdout |

## Taxi-out-specific structure from NASA's reference implementation

NASA's separate
[ML-airport-taxi-out](https://github.com/nasa/ML-airport-taxi-out) project
models unimpeded and impeded taxi time separately. Its configuration uses stand,
runway, carrier, stand-to-airline time, and arrival/departure surface counts, with
XGBoost regressors. This suggests two PRC lineages:

1. **Physical baseline:** estimate unimpeded travel from airport, stand, runway,
   aircraft, operator, route, and local-time history.
2. **Congestion residual:** predict the excess over that baseline from movement
   counts, schedule displacement, runway demand, weather, and disruption state.

Do not copy the NASA implementation. Its relevant code uses the NASA Open Source
Agreement 1.3, while the PRC prize rules require a GNU GPLv3 public solution.
Reimplement measured ideas from first principles unless a later license review
clears specific files.

## Validation requirements

The public score uses January and July 2026, so a random row split would reward
memorization and understate calendar shift. Initial validation should include:

- blocked full-month folds within 2025;
- January-like and July-like seasonal holdouts;
- airport-level error and tail-error reports in addition to global RMSE;
- target and feature availability checks run against the ranking schema; and
- a simple airport/time/group-prior baseline that every complex model must beat.

The first expensive build should wait until the supplied files confirm data
types, null patterns, identifier repetition, and whether movement rows expose
safe same-day congestion features exactly as the data page describes.
