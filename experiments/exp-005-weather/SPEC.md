# exp-005-weather — SPEC

Assignment: `assignments/weather.md` (tier hard, reviewer sol, 12 threads, no GPU).

## My reading of the assignment (and the readings I rejected)

**The question.** "What does open weather data add *once the actual-time columns
are in the model*." So the deliverable is a *difference*, not a model: one
frozen base model, and the RMSE it loses when weather features are added. Every
design choice below is subordinate to making that difference attributable to the
weather columns and nothing else.

**Base model.** Own-flight deltas + airport, stand, runway, aircraft type,
operator, hour, as the assignment names them. I added one column the assignment
does not name — day of week — because without it weather can proxy weekday
traffic (weekend snow would score as a snow effect). Adding it makes the
measured weather gain strictly more conservative, which is the direction an
honest ablation should err in. I rejected adding day-of-month or day-of-year:
inside a single validation month those index the specific weather days and would
let the base memorise what the weather features are supposed to supply.

**Learning target.** The residual `taxi − nm_taxi`, where
`nm_taxi = MVT_TIME_UTC_mvt − AOBT_3_flt`, anchored to 0 on the rows with no NM
join. Since the anchor is a constant per row, L2 on the residual *is* L2 on
taxi, so the ablation metric is the scored metric with no reweighting; and one
model covers both regimes because `has_nm` is a feature the trees split on at
the root. I rejected (a) predicting taxi directly, which forces the trees to
reproduce the identity `taxi ≈ nm_taxi` across a three-decade range and spends
capacity that should be measuring weather, and (b) two separate models for the
NM and no-NM regimes, which doubles the ablation surface for a 2 % slice.

**Frozen configuration.** One LightGBM configuration, fixed before the first
result and never tuned, no early stopping (early stopping on the validation
month would tune to it). Every reported gain is therefore a feature-set
difference. Seed noise is measured separately on the best config so that a gain
can be compared against it rather than assumed to exceed it.

**Time rule for the join.** ERA5 instantaneous variables are valid *at* their
timestamp; total precipitation and snowfall are the accumulation over the hour
*ending* at their timestamp. I relabel the accumulated variables back one hour
so a single clock-hour key `floor(event)` serves both kinds, and a departure at
13:42 gets the rain that actually fell while it was taxiing. I rejected joining
everything on the raw stamp, which shifts precipitation an hour early and can
manufacture signal, and I rejected interpolating the instantaneous variables to
the exact event minute: at a 0.25° grid cell the sub-hour precision is not the
binding error, and it would make `transform` stateful. Recorded as a refinement
if the lane shows gain.

**Legality.** `docs/input-rule.md` (2026-09-05) settles the task as
retrospective: the ranking file hands over the actual take-off time. Event-hour
ERA5 for January and July 2026 is therefore a legal input, and it cannot
reconstruct the blanked target in any case — weather is a function of (airport,
clock hour) alone and is shared by every flight in that hour. This supersedes
the conservative "quarantine post-event actual data" paragraph in
`docs/external-data-rules.md` (2026-09-02), which was written under the
pre-departure reading; the open-license, documented, reproducible conditions in
that file still bind and are met (CC BY 4.0, `docs/weather-provenance.md`).
The unresolved eligibility question is *account-gated access*, not event time:
CDS needs a free registration. That question is supplied-input rule's, not this experiment's.

**Scope I did not take.** No tuning, no ensembling, no submission candidate, no
tail or no-NM modelling — the tail worker owns that path. My tables report the
four COMMON.md slices so the numbers compose with the other workers'.

## Deliverables

- `prc/weather_features.py` with `build_hourly()` and `transform(month_frame)`
- `artifacts/weather/era5_hourly.parquet` (hash in RESULTS.md)
- `experiments/exp-005-weather/RESULTS.md`, `climb.log`, `oof/*.parquet`
- `docs/weather-provenance.md` and the attribution line
- ledger rows, Sol review

## Notes for the desk

- The ERA5 snapshot was missing every LSZH file and LTFM's 2026 extra. Those
  five retrievals are queued at CDS (`logs/fetch_missing.log`). The first
  ablation pass runs with LSZH low-cloud and snowfall null (10 % of
  airport-hours) and is re-run when they land.
- `xarray`/`netCDF4` are not in this repo's `.venv`; they are in the sibling
  checkout's. `build_hourly()` imports xarray lazily inside the NetCDF reader,
  so `transform()` and everything downstream run in this repo's venv against the
  parquet. Rebuilding the parquet from NetCDF needs the sibling venv (or
  `pip install xarray netCDF4`).
