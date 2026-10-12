# Validation protocol `week-pair-1` (exp-146)

Status: historical. This is our first protocol. Its five week-pairs were replaced by two whole-month folds,
fold J (January 2025) and fold L (July 2025), which the current entries use (`docs/writeup/METHOD.md` section
3.1). The month folds keep this protocol's buffer days (section 2, month holds), its training-frame checks
(section 3) and its scorer. The rule of section 3 for every fitted ancestor has two exceptions in the month
folds: the v32 half of the held route and reused fitted inputs (METHOD section 3.1). Code: `prc/weekpair.py`
(splitter, with `weekpair.month_hold`), `prc/gscore.py` (scorer).
The public rules are below; private review and process records are withheld.

## 1. Rule of use

- Select a candidate bundle on **G > 0** (section 6), all-row view, full five-pair score of the
  complete bundle. Always report the interval and the views of section 8. There is no per-cell,
  per-airport, per-month or December win rule (positive-point rule). An interval that contains zero is
  reported; it does not stop the candidate.
- No board value sets a weight, threshold, mask, class residual or rate.
  Before an alignment upload, freeze the bytes and write the local forecast (G, interval,
  changed classes). A miss starts a local audit of support, weights and the served path.
- Never change a split, weight or mask to agree with an old board gain.

## 2. Days and folds

- A row's day is the **UTC calendar date of `MVT_TIME_UTC_mvt`** (the field that splits the
  ranking file into its two months; the raw month files put a few edge rows in the next file).
- Five pairs, held at all ten airports at the same time (inclusive days of 2025):

  | pair | January | July | held departures |
  |---|---|---|---:|
  | 1 | 1–5 | 1–6 | 62,149 |
  | 2 | 6–12 | 7–13 | 78,159 |
  | 3 | 13–19 | 14–20 | 77,239 |
  | 4 | 20–26 | 21–27 | 77,642 |
  | 5 | 27–31 | 28–31 | 49,230 |

- **Buffers**: one day before and one day after each block. Their labels are excluded from
  every fitted stage of that fold; their rows are not scored. (Pair 1's 2024-12-31 has no data.)
- **Outer fold k** (`weekpair.outer(k)`): scores pair k; every fitted stage trains on all other
  2025 departures (1.98–2.01 M labelled rows of 2,085,047; `dryrun/splits.csv`).
- **Inner rotation** (`weekpair.inner(k, j)`, j ≠ k): excludes pairs k and j and both buffers,
  and predicts pair j **without fold k's buffer days**: `inner(k, j).held` = the days of pair j
  minus `buffer_days(k)`. So every calibration row of fold k is a label row of `outer(k)`, and no
  stage of fold k reads a label of fold k's buffer days. The rule removes
  two days (one in January, one in July) from the 8 rotations with adjacent pairs, for example
  01-05 and 07-06 from `inner(2, 1)`. For outer fold k, the four inner predictions are the only
  labels that a calibration may read: blend weights, mixture parameters, thresholds, early
  stopping, round counts. `inner(k, j)` and `inner(j, k)` exclude the same days, so one fit
  serves both; they score different rows.
- **Join**: the five outer predictions join by `MVT_ID_mvt` (`weekpair.join_outer`). Each
  January and July departure is predicted once, by its own fold: 344,419 rows, 620 airport-days.
- **Checks, not selection**: whole-month holds of 2025-01 and 2025-07 (season dependence) and the
  2025-12 control (`weekpair.month_hold`, one buffer day each side). December has zero ranking
  weight; it is scored as a plain row mean.
- **Final fit**: all 2025 labels (`weekpair.FINAL`). Its coefficients come from the outer
  predictions.

## 3. Label isolation for every fitted ancestor

A stage is "fitted" if it reads a departure label, directly or through another artifact: base
models, residual targets, target encodings and their priors, residual or neighbour features
made from labels or from fitted predictions, blend weights, calibration parameters, learned
guards, and label-cleaning lists used in training.

- Each fitted artifact writes `Split.manifest()` into its run manifest.
- A stage under split B may read an artifact fitted under split A only if A excluded every day
  that B excludes: `B.admits(A)`, enforced by `B.assert_admits(A)` at load time.
- Each training frame passes `split.assert_training_frame(frame, target=...)`. `target` is the
  column name or the array that the stage fits (y, a residual z, a class label); a departure on
  an excluded day with a non-missing target raises. Without `target`, every departure row on an
  excluded day raises, whatever its columns hold.
- The saved month-fold vectors (v22 fold panel, v23 fold vector, exp-134 fold predictions) have
  no week-pair manifest and known label paths (restart report T1). They are not admissible in a
  week-pair score. They are rebuilt, not reused.

## 4. Context masking

Every context builder (neighbours, queues, day state, routes, proxies, weather joins) receives
`weekpair.context_frame(frame)`: departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blank
on every departure row of every date, as in the ranking file. Arrival times, arrival taxi-in
and movement times stay. Each builder calls `weekpair.assert_context_masked` on its input. A
departure label appears only as the target of a fitted stage.

## 5. Classes, cells and weights

- **Class** (the served rule of v19–v32): `noac` = no `AIRCRAFT_TYPE_mvt`; `clock` = known
  type with an observed surface clock; `ordinary` = known type, no clock. 2025 clocks: exp-046's
  ungated block with the gated mode-4 matches kept (gap > 62.405 s, |P − AOBT| ≤ 1,299 s, runway
  agrees). 2026 clocks: `obs_has_clock` of the v25 class export. The 2025 class equals the v22
  fold panel's flag on every January and July row.
- **Cells** h = month × airport × class. **Weights** W_h = N_h / 344,841, N_h counted on the
  ranking departures (`frozen/cells.csv`: clock 196,256, ordinary 148,197, noac 388).
- **Support**: a cell is supported when it has at least 2 held 2025 rows over the five pairs.
  Unsupported ranking cells (22,721 rows): January LEMD clock 15,881 (1 row in 2025), January
  LIRF clock 5,700 (0), January LFPG clock 1,136 (0), July LTFM noac 4 (1). They contribute
  zero; the candidate keeps the comparator's route on their ranking rows unless another valid
  test supports a change. Their weight is not given to other cells.
- **Unchanged rows** have d = 0 and stay in their cell mean: row gains are pooled inside the
  cell before weighting. The 3,205 ordinary ranking rows without AOBT stay in the ordinary
  cells; the AOBT split is a view (section 8).

## 6. The G statistic

For each held departure i: d_i = (y_i − base_i)² − (y_i − cand_i)² (positive = better).

    G = Σ over supported h of W_h × mean_h(d_i)          [s² per ranking row]
    gain in seconds = R − sqrt(R² − G),  R = sqrt(Σ_supported W_h mean_h((y − base)²) / Σ_supported W_h)

- R is a **local** reference: the comparator's ranking-weighted RMSE on the supported cells of
  the same score. It only turns G into seconds; G > 0 does not depend on R. No board number is
  in the protocol.

- **Primary**: all rows. **Sensitivity**: clean view without prc.data's two unreached-rollover
  January rows.
- Score the **served form** of both candidate and comparator: holds, guards and the 60 s floor.
  The scorer refuses a prediction below 60 s.
- The comparator is the accepted-best route (now v32 = v24 + exp-137 delta) rebuilt under this
  protocol. Until that rebuild exists, a gain on it is reported as unmeasured.

## 7. Resampling

- **Primary interval**: 2,000 draws, seed 601, airport-days resampled with replacement within
  each airport × month; all classes of an airport-day move together. The draws are one frozen
  array of uniforms (`gscore.draw_uniforms`, hash in `MANIFEST.json`), so every candidate sees
  the same draws. 95 % percentile interval, standard error, and P(G ≤ 0).
- **Checks**: calendar days common to all airports; circular seven-day blocks within the month;
  the analytic standard error from day-level residual sums (T2's formula, summed per
  airport-day).
- In a draw, a supported cell with fewer than 2 rows contributes zero.
- Limits: these intervals measure 2025 day-to-day variation. They do not measure the change
  from 2025 to 2026, fit variation or selection on the same pairs.

## 8. Views to report

G and interval for all rows and clean; contributions (each partition sums to G) by class,
airport, month, class × AOBT, and label band (y ≤ 0, 0–3,600, 3,600–14,400, > 14,400 s); the
unsupported cells and any rows the candidate changed in them; the five largest airport-day
contributions and G without those days; G per pair; December and the month holds as checks.

## 9. Clock-loss masks

- **Frozen days** (`frozen/coverage.csv`, `frozen/clock_loss_days.csv`): for each
  airport-month, the removal rate is max(0, 1 − cov2026 / cov2025), with cov = clock rows /
  known-type rows. It never adds a clock where 2025 has none. `weekpair.dark_days` takes the
  floor(rate × days + 0.5) days with the lowest fixed hash. Scheme `coverage`: EDDF January
  9 days (rate 0.283), EDDF July 14 (0.441), EHAM January 1, LEBL January 1, LIRF July 31
  (0.999). Scheme `lirf_july`: all 31 LIRF July days only.
- **Evaluation mask**: remove every source observation (ADS-B points and the clocks and
  features made from them) on the dark airport-days before every derived feature and route.
  Rebuild the predictions of both comparator and candidate from the masked inputs; a made-dark
  row routes as a no-clock row. Score with `gscore.score(..., clock_mask=scheme)`: made-dark
  rows count in the ordinary cell, and the `darkness` view splits made-dark from naturally dark
  rows. Saved unmasked predictions are not a valid comparator for a masked day.
- **Training mask**: none by default (served models train on 2025 coverage). A candidate that
  adds a masked training view uses `dark_days` with the airport's frozen rate (the January rate
  for October–March, the July rate for April–September, `weekpair.training_rate_month`); the
  original and the masked view each get weight 0.5, so the total training weight is unchanged.
- Run `coverage` and `lirf_july` early, for the comparator and for each candidate.

## 10. Seeds, budgets, feature lists

- Every new fit uses seed 601 (LightGBM `seed=601`, `deterministic=True`, a recorded
  `num_threads`). Candidate and comparator use the same seed, rounds, leaves, learning rate and
  training rows in every fold. A refitted comparator keeps its recorded settings (for example
  exp-130's 1,500 rounds).
- Early stopping, round counts and any tuned setting are chosen on inner predictions only,
  never on the outer pair.
- Each model has an explicit feature list; its hash goes into the run manifest before the
  first scored run. No automatic column selection. A changed list is a new candidate. The rule
  is frozen, not the lists by value.
- Forbidden inputs: departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`; any label-derived
  column without an admissible manifest; the schedule-source flag `abs(BLOCK − SCHED) ≤ 6 s`;
  any board-derived value.

## 11. Run record

Each scored run records the split tokens, `weekpair.fingerprint()`, the sha256 of
`frozen/MANIFEST.json`, the feature-list hashes, seeds, run times and the `gscore.score` JSON,
and adds one entry to the team's run record.

## 12. Power (planning only)

A broad ordinary change needs G of about 390–530 s² for strict 80 % power on the full sample
(T2, historical, stated there as 0.8–1.1 s). The replay of v24's month-hold vectors gives SE
160 s² (0.26 s at its local R of 303.1 s; `dryrun-r1/dryrun.json`), the same as T2's historical
SE of about 164 s². Bundle small gains; recompute the interval for each candidate.
