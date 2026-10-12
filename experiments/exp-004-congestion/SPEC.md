# exp-004-congestion — surface-state features from the month frame

worker: congestion (hard) | reviewer: sol | started 2026-09-05

## My reading of the assignment, and the readings I rejected

The ranking frame gives every movement's take-off/landing time, every
departure's NM off-block, and every arrival's on-block **and taxi-in** time.
Only the departure's own `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` are blanked.
I read the task as: turn that surface picture into per-departure features,
measure each family against a frozen base model on the `july` and `december`
folds, and instrument the ceiling first.

Decisions I made and the alternatives I turned down:

1. **Features are computed per month, from that month's frame alone — for
   training months too.** Rejected: building features on the concatenated
   training frame. The ranking file holds January and July 2026 as two isolated
   months, so a training row must not see neighbours across a file boundary
   either; a model trained on richer context than it gets at prediction time
   mis-calibrates the features.
2. **`build_features` blanks departure `BLOCK_TIME_UTC_mvt` and
   `TAXITIME_SEC_mvt` in its own working arrays**, instead of trusting the
   caller to pass a masked frame. Verified: `transform(masked_frame)` is
   identical to `transform(unmasked_frame)` on every one of the 112 columns.
   Rejected: an assertion-only guard, which would have made the training path
   (labels present) either illegal or unchecked.
3. **Push proxy = `AOBT_3_flt`, else `LOBT_flt`, else `SCHED_TIME_UTC_mvt`.**
   The assignment asks for "AOBT replaced by LOBT for rows without NM", but
   `LOBT_flt` is null on exactly the rows where `AOBT_3_flt` is null (100 % of
   the 2.06 % no-NM July departures), so LOBT cannot serve as the no-NM
   fallback. `SCHED_TIME_UTC_mvt` is populated on 100 % of departures and is
   the only anchor those rows have. The LOBT-anchored variants are still built
   (`q_rwy_lobt`, `q_apt_lobt`, `span_lobt`) as a *different anchor for all
   rows*, which is what makes them worth measuring.
4. **Unimpeded taxi estimate = the 20th percentile of training taxi time**,
   hierarchically shrunk (airport → airport×runway → airport×terminal×runway →
   airport×stand×runway, with airport×stand as a side level), pseudo-count 20,
   fitted on training months only. Rejected: the median as the unimpeded
   estimate (it contains the queueing we are trying to isolate) and raw
   per-pair medians without shrinkage (rare stand-runway pairs are noise).
5. **The unimpeded table is fitted on all training months and applied to
   training rows as well** (each row contributes to its own group's quantile).
   Rejected leave-one-month-out: the validation month contributes nothing to
   the table, so the measured fold gain is honest either way; LOMO would only
   improve training-side calibration, and it costs a re-fit per month. If the
   geometry family turns out to matter, LOMO is the first refinement.
6. **Neighbour "what taxi-out is running now" statistics are 30-minute rolling
   medians over the window `(q - 1800, q)` anchored on the query take-off.**
   Rejected: clipped rolling means (cheaper, vectorised by cumsum, but the tail
   is the game here and the mean is not robust to it). The first
   implementation read the rolling median *at the last event before the query*,
   which after a quiet period returns a statistic from outside the window
   entirely; the Sol review caught it and it is fixed. That defect moved
   `nbr_span_apt` on 48.9 % of july rows by a mean of 21.9 s.
7. **Overtaking counts by a forward scan in push order.** The scan stops when
   no `i` satisfies `push_{i+s} < off_i`. That rule needs `off_j >= push_j`,
   which fails on the 0.14 % of departures whose take-off is stamped before
   their NM off-block; VERSION 2 also used the same test as a *mask*, which
   dropped real pairs and left `overtaken` wrong on 3.0 % of july rows. Round 2
   caught it. VERSION 3 runs the scan on the non-negative-span rows, where the
   rule is sound, and counts every pair involving a negative-span row directly;
   that matches brute force exactly on a full airport-month. **Every number in
   RESULTS is VERSION 2** (`dominance="push_window"`). Rejected: a Fenwick tree (exact but a per-row Python loop) and an
   n² dominance matrix (too big at 25 k departures per airport-month). The
   first version stopped after 25 consecutive empty offsets, which is *not* a
   proof; the Sol review produced a counterexample and the real data contains
   it — the heuristic under-counted `overtaken` on 2.88 % of july rows.
8. **Runway-configuration key is the sorted set of runway *names* in use in the
   surrounding hour**, decoded from a per-airport bitmask. Rejected: the raw
   bitmask as the categorical value — bit assignment depends on which runways
   appear in the frame, so the same configuration would get different keys in
   different months and the categorical would not transfer from train to valid.

## Two families beyond the six in the assignment

9. **`stand` — stand occupancy.** A stand holds one aircraft, so the first
   movement that takes this flight's stand at or after its push bounds its
   block time from above, and therefore its taxi time from below. The
   assignment's family 3 asks for arrivals whose `[landing, on-block]` interval
   overlaps; this is the same idea narrowed to the flight's own stand, where it
   becomes a constraint rather than a count. Measured before building it: the
   bound is positive on 11.1 % of july departures, holds on 99.5 % of those,
   and correlates +0.44 with `BLOCK − AOBT_3`, the quantity the whole task
   turns on. The first version of the idea (the previous arrival at the stand
   as an *upper* bound on taxi time) failed — 11 % violations, r = 0.01 — and
   the diagnosis is that the nearest arrival at a stand is usually the *next*
   occupant, not the flight's own inbound; that inverted the bound.
10. **`queue2` — a finer encoding of the queue.** Instead of counting take-offs
   between push and take-off, hand the model the runway's actual service times
   relative to the push (the 1st, 2nd, 3rd, 5th and 8th same-runway take-off
   after it), the mirror times before its own take-off, and pushback demand in
   the 30 minutes around its push. Added after the residual analysis showed the
   combined model had extracted all the *linear* signal from the count form
   (max |r| between residual and any of the 96 columns on the july NM bulk =
   0.049). Kept out of `FAMILY_COLUMNS` so that "all families" keeps meaning
   the assignment's six plus `stand`.

## Base model (frozen for every row of the ablation)

CatBoost, RMSE loss, 3000 iterations, lr 0.05, depth 8, l2 3.0, seed 7,
`max_ctr_complexity=2`, GPU 1, 16 threads. Features: `dt_to_aobt`,
`dt_to_lobt`, `dt_to_eobt`, `dt_to_sched`, `dt_aobt_sched`, `dt_aobt_lobt`,
`flight_time_est` (`ARVT_3_flt` − take-off), `hour`, `dow`, `has_nm`, and the
categoricals airport, airport:stand, airport:runway, aircraft type, operator.

## Files

- `prc/surface_features.py` — `SurfaceFeatures.fit(train)` / `.transform(month)` /
  `.transform_by_month(frame)`, 96 columns in seven families (`FAMILY_COLUMNS`)
  plus an optional 16-column `queue2` block (`EXTRA_FAMILY_COLUMNS`).
- `run_oracle.py` — ceiling instrument.
- `run_ablation.py` — cached per-month features, family ablation, OOF dumps.
- `climb.log`, `RESULTS.md`.
