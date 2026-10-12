# Reference research: PRC Data Challenge 2026

Last verified: 2026-09-01

## Decision this pack supports

This pack identifies the first methods to test for the 2026 taxi-out prediction task. It also records the rules, source licenses, and validation risks that can change that decision. It uses official competition pages, source repositories, papers, and first-party technical reports. It does not use third-party summaries as evidence.

## 1. Official 2026 task contract

The [official challenge page](https://ansperformance.eu/study/data-challenge/dc2026/) defines a taxi-out prediction task for 11 major European airports. The training set contains 2025 movements. The ranking set contains January and July 2026 movements. The score is root mean squared error (RMSE). The challenge runs from 1 September to 11 October 2026 at 23:59:59 CET, and the prize is EUR 5,000. The page also requires public code and documentation and encourages a paper submission to the Journal of Open Aviation Science.

The [official data page](https://ansperformance.eu/study/data-challenge/dc2026/data.html) is the source of truth for the files and fields:

- The training data has 4,167,797 arrival and departure records from EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTAI, LTFM, and LSZH.
- The 12 training files are monthly Parquet files for 2025.
- `ranking.parquet` combines January and July 2026. It has the same columns as training data, but departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blank.
- A valid submission has exactly `MVT_ID_mvt` and `TAXITIME_SEC_mvt`. Its movement IDs must match the departure rows (`PHASE_mvt == "DEP"`) in the ranking data.
- The movement data includes airport, phase, schedule, block time, movement time, aircraft type, runway, stand, and taxi time. The joined Network Manager data includes LOBT, IOBT, EOBT, AOBT, flight rules, flight type, aircraft, wake class, market segment, and an anonymized operator.
- The organizers warn that these are operational data. Missing values, inconsistent values, and join errors are part of the task.

The [official ranking rules](https://ansperformance.eu/study/data-challenge/dc2026/ranking.html) keep the best RMSE for each team. A file name must use `<team-name>_v<integer>.parquet`. The movement IDs must exactly match the departure rows in the ranking data. The organizers also warn teams not to learn from or exploit the ranking data.

The [official eligibility rules](https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html) allow worldwide participation, subject to sanctions and other stated restrictions. To qualify for a prize, a team must meet these main conditions:

- External data must be open and documented.
- All source code produced for the entry must be on GitHub under GPLv3.
- Added data must use an open license.
- The documentation must let the organizers reproduce the result.
- The solution must be original. A reused implementation needs the required rights and a significant change. A simple input or output rewrite is not enough.

The [official rationale](https://ansperformance.eu/study/data-challenge/dc2026/rationale.html) says that taxi-out time varies with airline constraints, airport procedures, airport load, and air traffic flow management. It also links the output to constrained-period detection and excess fuel and CO2 analysis. This supports explicit congestion and operational-state features, not only static route features.

### Immediate leakage risk

This is an inference from the published schema, not an organizer ruling. The data page says that departure block time and taxi time are blank in the ranking set. It does not say that movement time or Network Manager `AOBT_3_flt` is blank. Since actual taxi-out time is normally the difference between actual take-off time and actual off-block time, a difference such as `MVT_TIME_UTC_mvt - AOBT_3_flt` could reconstruct or closely proxy the target.

Until the organizer gives a clear answer, quarantine these inputs:

- `AOBT_3_flt` and any later actual off-block value;
- any difference between movement time and an actual off-block value;
- any field created after the prediction cut-off;
- any aggregate or encoding that includes the validation row, later rows, or ranking outcomes.

This check is the first task before model work. A good public score from a target proxy would not meet the spirit of the [ranking warning](https://ansperformance.eu/study/data-challenge/dc2026/ranking.html), and it could fail reproducibility review.

## 2. Prior PRC challenges and code

### Official 2024 material

The [2024 challenge page](https://ansperformance.eu/study/data-challenge/dc2024/) describes an aircraft take-off mass prediction task. It used 369,013 training flights, 105,959 intermediate-ranking flights, and 52,190 final-ranking flights. The [2024 ranking](https://ansperformance.eu/study/data-challenge/dc2024/rankings.html) lists `team_likable_jelly` first with an RMSE of 1,561.63. The official public code is under the [PRC Data Challenge 2024 GitHub organization](https://github.com/prc-data-challenge-2024).

High-value implementation: [`team_likable_jelly`, pinned at `fd8f0c4`](https://github.com/prc-data-challenge-2024/team_likable_jelly/tree/fd8f0c45e737848b9da53cb4739fd463c9775504). It uses LightGBM, seed bagging, categorical flight and airport features, weather data, ADS-B-derived features, and a Make-based reproducible pipeline. Its GPLv3 license is compatible with study and reuse under the stated terms. For 2026, the useful parts are the reproducible pipeline, native categorical boosting, weather joins, and seed ensemble. The take-off mass target and ADS-B feature details do not transfer directly.

### Official 2025 material

The [2025 challenge page](https://ansperformance.eu/study/data-challenge/dc2025/) describes a fuel-burn prediction task with two ranking phases. The [official outcome report](https://ansperformance.eu/study/data-challenge/dc2025/outcome.html) lists 179 registered teams, 53 teams with a submission, and 2,127 submissions. It names `resourceful-quiver`, `jubilant-vase`, and `outspoken-tornado` as the first three teams. The report says that the winning work combined domain features, a 2024 mass estimator, custom traffic cleaning, and complete documentation. The second team used a take-off mass estimator and a three-model gradient-boosting ensemble. The third team used trajectory summaries and a structured pipeline. Public code is under the [PRC Data Challenge 2025 GitHub organization](https://github.com/prc-data-challenge-2025).

Representative repositories:

1. [`resourceful-quiver`, pinned at `2484125`](https://github.com/prc-data-challenge-2025/resourceful-quiver/tree/2484125852ebb6cfb1fe55c253fb3f7539c924ff) is the first code study. It has data cleaning, interpolation, domain feature groups, LightGBM, and a reproducible report. It is GPLv3. For 2026, port the data audit and experiment structure, then replace the fuel-burn feature logic.
2. [`jubilant-vase`, pinned at `57b9430`](https://github.com/prc-data-challenge-2025/jubilant-vase/tree/57b94308bfefe6b110ade31cda24bec2f08b799f) shows XGBoost, LightGBM, and CatBoost out-of-fold blending, with grouped folds. For 2026, the useful mechanism is diverse tree-model blending. The pinned repository has no root license file. Study it, but do not copy its code unless the rights are clear.
3. [`outspoken-tornado`, pinned at `9d3154b`](https://github.com/prc-data-challenge-2025/outspoken-tornado/tree/9d3154b351306504555979373f2a9a4011cfc92f) has a modular CatBoost pipeline, Optuna search, saved run configurations, evaluations, and submission history. It is GPLv3. For 2026, reuse the experiment-record and configuration pattern.

These prior tasks have different targets. Their strongest transfer is engineering discipline, data cleaning, categorical tree models, and ensemble design. They are not evidence that a target-specific feature will work for taxi-out time.

## 3. NASA code references: high priority, study first

### ML Airport Taxi-Out

NASA's [`ML-airport-taxi-out`, pinned at `add029a`](https://github.com/nasa/ML-airport-taxi-out/tree/add029a55966d846498d42ac54d698d3ed724775) is the closest public reference implementation. It defines airport-specific unimpeded and impeded taxi models for airport movement area, ramp, and full taxi-out time. It uses Kedro, MLflow, and XGBoost. Its configuration includes stand, runway, carrier, surface arrival and departure counts, and a stand-airline-time feature. It also has stand clustering and grouped runway-terminal baselines.

Reusable mechanism for 2026:

- separate a stable unimpeded component from a congestion component;
- train per-airport models or airport-specific residual heads;
- use stand, runway, operator, and surface counts;
- use hierarchical stand or terminal clusters when a stand has little history;
- compare against grouped operational baselines before using a complex model.

Do not port its random split as-is. The 2026 hidden period is later than the training period, so validation must also move forward in time.

### ML Airport Data Services

NASA's [`ML-airport-data-services`, pinned at `3389776`](https://github.com/nasa/ML-airport-data-services/tree/3389776d67e8d2564b627434296c9b62b82ec380) supplies feature and data-service code used by airport ML models. The high-value mechanisms are surface traffic counts, stand-runway taxi-time aggregates, stand clusters, time-series sampling, consistent row filters, stable feature order, and shared error metrics. For 2026, use this as a design reference for a small feature service with one definition for training, validation, and submission.

### NASA license boundary

Both repositories use the NASA Open Source Agreement 1.3: [taxi-out license](https://github.com/nasa/ML-airport-taxi-out/blob/add029a55966d846498d42ac54d698d3ed724775/license.txt) and [data-services license](https://github.com/nasa/ML-airport-data-services/blob/3389776d67e8d2564b627434296c9b62b82ec380/license.txt). That agreement requires distribution of subject software under the same NASA agreement. This can conflict with the PRC rule that all produced source code must be under GPLv3.

This pack does not make a legal conclusion. Treat NASA code as study material. Reimplement the mechanisms independently. Do not copy, import, or modify NASA source in the competition repository until the organizer or a qualified license review confirms a compliant path.

## 4. Taxi-out and surface-operation evidence

### A. Strong baselines and target decomposition

1. **EUROCONTROL additional taxi-out indicator.** The [official indicator specification](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf) defines actual taxi-out as actual take-off time minus actual off-block time. It separates actual time into a reference time and additional taxi-out time. Its reference is a local, rolling previous-12-month 10th percentile for the same stand and departure-runway combination, with support and data-quality rules. For 2026, this is the first domain baseline. It also suggests a better target form: predict the additional component and add it to the reference time.

2. **EUROCONTROL taxi-time evidence.** [EUROCONTROL Data Snapshot 40](https://www.eurocontrol.int/publication/eurocontrol-data-snapshot-40-taxi-times) links taxi time to airport layout, runway use, traffic demand, and weather, and separates normal movement time from added congestion. For 2026, it supports airport-specific models plus a congestion residual.

3. **Paris Charles de Gaulle tree model.** Herrema and co-authors report a CDG departure taxi-time model in [Journal of Aerospace Information Systems](https://doi.org/10.2514/1.I010502), with an open metadata record at [TU Delft](https://repository.tudelft.nl/record/uuid%3Acd1bb9b5-772b-4ca4-8417-f0ab2afa836a). A regression tree was the best method in that study, with an average daily RMSE near 1.6 minutes. LFPG is in the 2026 data. This supports a strong tree baseline before a graph model.

### B. Congestion, queues, and operational state

4. **Surface surveillance decomposition.** The MITRE paper [“Improving Departure Taxi Time Predictions Using ASDE-X Surveillance Data”](https://www.mitre.org/news-insights/publication/improving-departure-taxi-time-predictions-using-asde-x-surveillance-data) uses queue position, taxi distance, arrival and departure rates, and weather. It separates movement to the queue from queue wait. For 2026, approximate these states with concurrent departure and arrival counts, runway demand, stand/runway group, and schedule deltas.

5. **Airport congestion and throughput.** Simaiakis and Balakrishnan connect surface congestion, runway departure throughput, and excess taxi time in [Transportation Research Record](https://doi.org/10.3141/2184-03). For 2026, this is the basis for rolling surface counts, departure-queue proxies, and a nonlinear saturation term rather than a simple linear count.

6. **Off-block uncertainty.** Badrinath and co-authors study departure metering when off-block times are uncertain in [Transportation Science](https://doi.org/10.1287/trsc.2019.0957). For 2026, differences among IOBT, EOBT, LOBT, and schedule time can describe plan instability. Use only values available at the chosen prediction cut-off. Do not use an actual AOBT target proxy.

7. **Probabilistic adaptive prediction.** Balakrishna, Ganesan, and Sherry present an adaptive dynamic-programming approach for taxi-out prediction under changing airport state in [Transportation Research Record](https://doi.org/10.3141/2052-07). For 2026, its useful idea is state-dependent updating. Test this only after static gradient-boosting and rolling-count baselines.

### C. Route, weather, and schedule features

8. **Graph and flow model.** The SESAR/NTU paper [“Variable Taxi-Out Time Prediction Using Graph Neural Networks”](https://www.sesarju.eu/sites/default/files/documents/sid/2021/papers/SIDs_2021_paper_78.pdf) uses intended surface route, route distance, unimpeded time, queue position, runway activity, wake class, carrier, and time-varying edge occupancy. It uses the last 10% of flights as a chronological test. Its gradient-boosting RMSE was 78.4 seconds and its graph model RMSE was 76 seconds. For 2026, first port the queue, route-proxy, and flow features to a tree model. A graph model is a later test because the PRC files do not give explicit taxi routes.

9. **NASA Charlotte study.** NASA's [CLT taxi-out prediction report](https://ntrs.nasa.gov/citations/20180002162), also published under [DOI 10.2514/6.2016-3910](https://doi.org/10.2514/6.2016-3910), uses terminal concourse, spot, runway, departure fix, weight class, traffic, and weather. Linear regression and random forest performed best among the compared methods. For 2026, map concourse and spot to stand clusters, and map departure fix to available destination or runway-flow proxies.

10. **NASA unimpeded service.** NASA's [real-time unimpeded taxi-out service report](https://ntrs.nasa.gov/citations/20210017591) treats unimpeded and impeded travel as different parts of the problem and uses airport-specific historical models. NASA also provides the [ITOM/UTOM software record](https://ntrs.nasa.gov/citations/20210021577). For 2026, test a reference-time model plus an added-delay model and keep airport-specific parameters.

11. **NASA airport-surface ML architecture.** NASA's [Machine Learning Airport Surface Model report](https://ntrs.nasa.gov/citations/20230014305) describes airport-specific learned surface models in a service-oriented operational system. For 2026, its main value is the interface: one stable feature contract, separate airport configuration, model versioning, and repeatable inference.

12. **Open weather data.** The official [ERA5 single-level data set](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) provides global hourly weather on a 0.25-degree grid from 1940 and is updated daily. Its persistent identifier is [DOI 10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47). For 2026, test wind components, precipitation, temperature, dew point, pressure, and cloud variables at each airport. Keep the data-retrieval script, variable list, grid rule, time rule, and license notice in the public repository.

### D. Leakage-safe validation and uncertainty

13. **Rolling-origin evaluation.** Tashman describes out-of-sample and rolling-origin evaluation for forecasts in the [International Journal of Forecasting](https://doi.org/10.1016/S0169-2070(00)00065-0). The origin moves forward and the model is evaluated on later observations. For 2026, use several origins in 2025 rather than random rows.

14. **Time-ordered split implementation.** The official [scikit-learn `TimeSeriesSplit` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) explains that ordinary cross-validation can train on future observations and test on past observations. For 2026, a custom month-blocked splitter is likely clearer, but this is a good reference test for ordering and gap behavior.

15. **Rolling conformal uncertainty.** The paper [“Rolling Conformal Prediction for Aircraft Taxi-Time Forecasting”](https://doi.org/10.19139/soic-2310-5070-3782) studies time-adaptive prediction intervals for taxi time. RMSE is the competition objective, so intervals are not a first-score feature. After the mean model is stable, rolling residual scale can identify regimes, tail errors, and airport-month drift. Any residual calibration must use only earlier observations.

## 5. Recommended validation contract

The official test has two seasons, January and July 2026. Random validation will overstate performance because the same days, demand banks, and operational regimes can appear in both train and validation data.

Use forward-only folds such as:

1. Train January-April 2025; validate May 2025.
2. Train January-June 2025; validate July 2025.
3. Train January-October 2025; validate November or December 2025.

Keep the exact folds in configuration. Add a gap only if the feature-generation process can see near-future operations. For every fold:

- fit stand, runway, operator, and route aggregates on the training side only;
- calculate rolling counts from data that would be known at the prediction cut-off;
- do not use validation targets in lag, mean, percentile, or cluster features;
- report total RMSE and RMSE by airport, month, support level, and target tail;
- compare the same rows for every candidate;
- save out-of-fold predictions so ensemble weights do not use the ranking board.

Use July 2025 as the main season-matched fold. Keep another late-year fold so one month does not control the design. January 2025 has no earlier 2025 data, so a pure 2025 rolling reference has limited history. Use hierarchical fallbacks or external open history if the rules and license allow it.

## 6. Prioritized reading and porting shortlist

### P0: read and implement first

1. Read the 2026 [data](https://ansperformance.eu/study/data-challenge/dc2026/data.html), [ranking](https://ansperformance.eu/study/data-challenge/dc2026/ranking.html), and [eligibility](https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html) pages. Freeze the prediction cut-off and leakage exclusions.
2. Implement the [EUROCONTROL additional taxi-out reference](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf): stand-runway rolling low percentile, with minimum support and hierarchical fallbacks.
3. Study the pinned NASA [taxi-out](https://github.com/nasa/ML-airport-taxi-out/tree/add029a55966d846498d42ac54d698d3ed724775) and [data-services](https://github.com/nasa/ML-airport-data-services/tree/3389776d67e8d2564b627434296c9b62b82ec380) repositories. Reimplement only the mechanisms after the license boundary is recorded.
4. Train simple airport-aware LightGBM and CatBoost models with stand, runway, operator, aircraft type, wake class, local calendar, schedule deltas, and leakage-safe surface counts.
5. Use the month-blocked validation contract before any ranking submission.

### P1: test after the baseline is stable

1. Predict `actual taxi-out - reference taxi-out` instead of the full target.
2. Add nonlinear departure and arrival counts, runway demand, demand-bank features, and plan-change differences among schedule, IOBT, EOBT, and LOBT.
3. Add stand clusters and shrink rare stand-runway estimates toward airport, runway, terminal, and global estimates.
4. Compare one global multi-airport model, separate airport models, and a global model with airport-specific residual correction.
5. Blend diverse tree models only with out-of-fold predictions.

### P2: useful ablations

1. Join ERA5 weather with a documented, reproducible time and grid rule.
2. Add tail or residual-scale features and test whether they improve mean RMSE.
3. Re-test useful 2024 and 2025 pipeline patterns: seed averaging, run manifests, source hashes, and submission provenance.

### P3: expensive research

Build a taxiway graph or GNN only if open airport geometry and a reliable route inference method are available, licensed, and measurably better in a cheap pilot. The SESAR result shows only a small gain over gradient boosting in its setting, while the PRC data does not provide explicit taxi routes.

## 7. Gaps and risks

- **Prediction-time contract is not explicit.** The public pages do not state exactly when a prediction must be made or which actual fields are allowed. Ask the organizer whether `AOBT_3_flt`, actual movement time, and same-flight actual events are valid inputs.
- **Ranking months are known.** Month matching can help, but tuning against the ranking board can become an exploit. Use 2025 pseudo-tests and keep the ranking board as a final check.
- **January history is weak.** A 12-month rolling baseline needs 2024 data for full January 2025 emulation. A fallback tree or hierarchical reference is necessary unless open historical data is added.
- **External data creates release duties.** Record URL, retrieval time, version, license, transforms, and redistribution terms for every added data set.
- **NASA code has a license conflict risk.** Study and clean-room reimplementation are safer than copying. Get a license decision before any source transfer.
- **One prior finalist repository has no visible root license.** Treat `jubilant-vase` as read-only study material.
- **Surface counts can leak.** A count made from final actual event times can include information that was not known at prediction time. Define each window against the prediction cut-off.
- **Airport operations change.** Runway use, stands, airline schedules, and procedures can drift between 2025 and 2026. Track error by airport and month and prefer stable operational groupings.
- **Public source list is not a result guarantee.** Each mechanism still needs a controlled ablation on the fixed temporal folds.

## 8. Source-use rule

For each source that affects code, add a short provenance record with the source URL, pinned commit or DOI, license, mechanism learned, files that implement the independent version, and the validation result. This makes the final GPLv3 release and reproducibility review easier.
