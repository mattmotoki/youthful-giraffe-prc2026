# Weather data provenance — Copernicus ERA5 at the airport point

Reproducibility record required by `docs/external-data-rules.md` for the
external data set used by `prc/weather_features.py` and `exp-005-weather`.

Status of the frozen source record: **retrieval completed on 2026-09-05**.
Current v32 release status: the exact derived table was recovered and
packaged with its frozen hash. The raw 44-file snapshot is not on this Mac.
A later ERA5T fetch is a new input version, not a replacement for these bytes.

## Attribution line

> Contains modified Copernicus Climate Change Service information 2026.
> Generated using Copernicus Climate Change Service information 2026:
> Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to
> present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS),
> DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47), and
> ERA5 hourly time-series data on single levels, DOI
> [10.24381/1cf1ad76](https://doi.org/10.24381/1cf1ad76). Accessed
> 2026-09-01 to 2026-09-05. Neither the European Commission nor ECMWF is
> responsible for any use of this information.

The data were modified: unit conversion, wind speed and direction derivation,
one-hour relabelling of the accumulated variables, rolling windows, and 3-hour
differences. All transformations are in `prc/weather_features.py`.

## 1. Provider, product, source, license

| | |
|---|---|
| Provider | Copernicus Climate Change Service (C3S), through the Climate Data Store (CDS) |
| Core product | `reanalysis-era5-single-levels-timeseries` — [catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries), DOI `10.24381/1cf1ad76` |
| Extra product | `reanalysis-era5-single-levels` — [catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), DOI `10.24381/cds.adbb2d47` |
| License | CC BY 4.0 — <https://creativecommons.org/licenses/by/4.0/> |
| Access | Free CDS account, dataset licence accepted in the web interface, personal access token |
| Assessment | `docs/copernicus-era5-assessment.md` (why direct CDS and not Open-Meteo) |

Two products are used because the fast time-series product does not carry low
cloud cover or snowfall; those two come from the stable parent product at the
same grid cell.

**Access and redistribution are different.** CDS requires registration, licence
acceptance and a token to use its download service. The Copernicus product
licence is free, worldwide, royalty-free and perpetual, and permits reproduction,
public distribution, adaptation, modification and combination with attribution.
Both catalogue entries name CC BY. An anonymously downloadable exact bundle
therefore removes the service-access issue for a reproducer.

## 2. Version, query, time range, retrieval date

- Time range retrieved: 2024-01-01 00:00 UTC through 2026-07-31 23:00 UTC for
  the core product (22,632 hourly steps per airport); calendar years 2024 and
  2025 plus January and July 2026 for the extra product.
- Only 2025 (the training year) and January and July 2026 (the ranking months)
  are used by the model. 2024 is EDA headroom.
- Retrieval dates: 2026-09-01 to 2026-09-05 UTC.
- **ERA5 release version.** The extra files carry an `expver` coordinate:
  `0001` (final ERA5) for every 2025 hour and for January 2026, `0005`
  (preliminary ERA5T) for the 744 hours of July 2026. ERA5T values can be
  revised when the final release appears two to three months later, so the
  July 2026 join is against a frozen preliminary snapshot. The 2025 folds that
  produce every measured number in `exp-005-weather` are entirely final ERA5.
  Re-running the retrieval later would be a **new declared data version**, not
  a repair of this one.

## 3. Download command

```bash
# credentials: $HOME/.cdsapirc, mode 0600, two lines, no secret in this repo
#   url: https://cds.climate.copernicus.eu/api
#   key: <PERSONAL-ACCESS-TOKEN>
python scripts/download_copernicus_era5.py full \
  --data-dir data/external/weather/copernicus_era5
# the five files missing from the first pass (all LSZH, plus LTFM's 2026 extra):
python experiments/exp-005-weather/fetch_missing.py
```

`scripts/download_copernicus_era5.py` writes the exact request body of every
call to `copernicus_era5/requests/<file>.request.json`, validates each response
(one latitude, one longitude, exact hourly coverage, no duplicate stamps, the
expected grid cell, the expected variables), and appends a manifest row. It
re-verifies rather than re-downloads a file that is already present and valid.

## 4. Raw files, sizes, hashes

The generated raw-download manifest has 46 rows
(44 production files for 11 airports plus 2 two-day test files),
23.6 MiB total, each row carrying `file`, `bytes`, `sha256`, `request_file`,
`latitude`, `longitude`, `hours`, `first_hour_utc`, `last_hour_utc`,
`data_variables`, `doi`, `license`, and the retrieval timestamp.

`experiments/exp-005-weather/inventory.py` re-hashes every raw file and prints
one row per (airport, product) with the manifest check. Its output for the
verified snapshot is in `experiments/exp-005-weather/RESULTS.md`.

The generated derived table has 226,320 rows = 10 ranking airports × 22,632
hours. Its hash is in [the released ERA5 manifest](../release/ERA5-DATA-MANIFEST.json).
The original output layout was:

```text
artifacts/weather/era5_hourly.parquet
```

## 5. Airport mapping, join key, tolerance, missing rule

ERA5 is a 0.25° grid; each airport is mapped to its nearest grid cell, asserted
against the cell the first CDS response returned and reused verbatim in the
parent-product `area` box so both products describe the same cell.

| Airport | Requested lat, lon | ERA5 cell |
|---|---|---|
| EDDF | 50.026, 8.543 | 50.00, 8.50 |
| EDDM | 48.354, 11.786 | 48.25, 11.75 |
| EGLL | 51.478, -0.461 | 51.50, -0.50 |
| EHAM | 52.309, 4.764 | 52.25, 4.75 |
| LEBL | 41.297, 2.078 | 41.25, 2.00 |
| LEMD | 40.494, -3.567 | 40.50, -3.50 |
| LFPG | 49.013, 2.550 | 49.00, 2.50 |
| LIRF | 41.804, 12.251 | 41.75, 12.25 |
| LSZH | 47.465, 8.549 | 47.50, 8.50 |
| LTFM | 41.275, 28.752 | 41.25, 28.75 |

(LTAI is retrieved too but has no departures in the ranking file.)

- **Join key**: `(ADEP_mvt, floor(event time to the hour))`, UTC throughout. No
  local time, no daylight-saving handling, no tolerance window: an exact
  clock-hour key, left join.
- **Two joins per departure**: at the take-off hour (`MVT_TIME_UTC_mvt`, the
  `wx_` prefix) and at the actual off-block hour (`AOBT_3_flt`, the `wxo_`
  prefix). Both source columns are present in the ranking file.
- **Missing rule**: no fill and no nearest-future substitution. A row that
  finds no weather hour keeps NaN and raises `wx_missing`; a row with no NM
  join has no off-block hour and raises `wx_missing_aobt_hour`. LightGBM
  handles the NaN directly.

## 6. Time rule for accumulated variables

ERA5 instantaneous variables (2 m temperature, 2 m dew point, mean sea-level
pressure, 10 m wind components, total and low cloud cover) are valid **at** the
timestamp. Total precipitation and snowfall are the accumulation over the hour
**ending** at the timestamp
([ECMWF accumulation guide](https://confluence.ecmwf.int/spaces/CKB/pages/197702790/Conversion%2Btable%2Bfor%2Baccumulated%2Bvariables%2Btotal%2Bprecipitation%2Bfluxes)).

The hourly table is keyed by clock-hour start `H`. Instantaneous columns carry
the analysis at `H`; accumulated columns carry the value stamped `H+1`, i.e.
the accumulation over `[H, H+1)`. A departure at 13:42 therefore joins to
`H = 13:00` and receives the precipitation that fell while it was taxiing.
Consequence: the last hour of the snapshot (2026-07-31 23:00) and the last hour
of January 2026 for snowfall have no `H+1` stamp and stay NaN rather than zero.

## 7. Transformations and generated features

All in `prc/weather_features.py`.

Unit conversions: temperature and dew point K → °C; mean sea-level pressure
Pa → hPa; total precipitation and snowfall m → mm (snowfall as water
equivalent, its native unit). Cloud covers stay fractions in 0–1.

Derived: wind speed `sqrt(u10² + v10²)`; meteorological wind direction
`(270 − atan2(v10, u10)·180/π) mod 360`, the direction the wind blows *from*
([ECMWF wind guide](https://confluence.ecmwf.int/spaces/CKB/pages/133262398/ERA5%2BHow%2Bto%2Bcalculate%2Bwind%2Bspeed%2Band%2Bwind%2Bdirection%2Bfrom%2Bu%2Band%2Bv%2Bcomponents%2Bof%2Bthe%2Bwind));
dew-point depression; liquid precipitation `max(tp − sf, 0)`; freezing
precipitation flag `temp ≤ 2 °C and precip > 0.1 mm`; snowing flag; sub-zero
flag; low-visibility flag `dew-point depression < 2 °C and low cloud > 0.8`;
cloud-base proxy `125 × dew-point depression` (m); rolling 3 h and 6 h sums of
precipitation and snowfall ending with the joined hour; 3-hour changes in
temperature, pressure, wind speed, low cloud and dew-point depression.

Runway-aligned wind: the runway heading comes from the designator
(`RUNWAY_mvt` `'26L'` → 260°; `'NA'` and the LSZH helipad `'H'` → NaN), and
with `θ = wind_direction − heading`, headwind `= speed·cos θ`, crosswind
`= speed·sin θ`, plus `|crosswind|` and tailwind `= max(−headwind, 0)`.

Across-taxi change: take-off-hour minus off-block-hour temperature and wind
speed, and the larger of the two hours' precipitation.

No clipping, no imputation, no smoothing of the source values.

## 8. Exact v32 bundle and raw rebuild

The v32 release must use the separate data bundle defined by
`release/ERA5-DATA-MANIFEST.json`. Import and verify frozen custody with:

```bash
.venv/bin/python release/era5_data.py pack artifacts/weather/era5_hourly.parquet \
  --out copernicus-era5-v32
.venv/bin/python release/era5_data.py verify copernicus-era5-v32
```

The packager proves the byte count, SHA-256, 226,320 rows, airport-hour key,
ten-airport set, and absence of movement ids and label columns. Its `NOTICE.txt`
keeps the data under CC BY 4.0 and does not relabel it as GPLv3.

To rebuild a new declared version from raw CDS data:

1. Create a free CDS account, accept the ERA5 licence for both catalogue
   entries, put the token in `$HOME/.cdsapirc` (mode 0600).
2. `pip install cdsapi xarray netCDF4` (this repo's `.venv` carries the
   modelling stack only; the NetCDF read runs in the sibling checkout's venv).
3. Run the two commands in section 3. The scripts generate request JSON beside
   a new download; those generated request files are not committed.
4. Rebuild the derived table: `python -c "import prc.weather_features as w;
   w.build_hourly()"`; compare its SHA-256 against the released ERA5 manifest linked in section 4.
5. Re-run `experiments/exp-005-weather/inventory.py` to confirm every raw hash.

ERA5T (July 2026) is the one part that a later retrieval can legitimately
change; see section 2.

## 9. Completeness

`experiments/exp-005-weather/inventory.py` on 2026-09-05 found 39 of the 44
airport-product files present with every SHA-256 matching the manifest and
every grid cell as expected, and five absent: all four LSZH files and LTFM's
January-plus-July 2026 extra. All five were retrieved by
`experiments/exp-005-weather/fetch_missing.py` in the same session
(`logs/fetch_missing.log`, final line `fetch_missing complete: files=5`).

**Custody state:** all 44 production files were present, validated and hashed on
the machine that built exp-005. They are absent on this Mac. The exact derived table was recovered for the v32 release and packaged
with its frozen hash. Raw-file completeness is not re-run here. `inventory.py --data-dir PATH` fails cleanly without a supplied snapshot.

## Exact v32 bundle and event timing

The v32 release packages the exact inherited hourly table as a separate
CC BY 4.0 bundle. `release/ERA5-DATA-MANIFEST.json` records the bytes, hash,
ordered schema, airport-hour key, product URLs and notices.
`release/era5_data.py pack` and `verify` check the real file. The frozen file
has no version attr; its column order begins `hour_utc, airport`. Those are
properties of the original snapshot, not a reason to rewrite its bytes.
A later ERA5T fetch is a new input version. Public anonymous hosting remains
a publication step, separate from this private package check.

Accumulated precipitation and snowfall keyed to hour H cover [H,H+1), so a
departure inside the hour receives a post-take-off portion. The organizer's
recorded post-operations answer removes the earlier assumed cut-off. Open
terms and reproduction still apply. See
[external-data-rules.md](external-data-rules.md). No eligibility ruling for
this specific entry is claimed.

## 10. Data corrections

Two corrections changed the derived data:

- **Rolling windows no longer fill partial windows.** `snow_3h_mm`,
  `snow_6h_mm`, `precip_3h_mm` and `precip_6h_mm` were computed with
  `min_periods=1`, so a window containing a missing hour returned the sum of
  the hours that were present. The last hour of each ranking month has a null
  accumulation by construction (no `H+1` stamp), and it was reporting a
  non-null rolling total. They now require the full window and return NaN
  otherwise.
- **Window convention stated precisely.** These are the `window` hours **ending
  with and including** the joined clock hour — accumulation over
  `[H-window+1, H+1)` — not the hours strictly before `H`. The assignment's
  phrase "snow in the last 3 and 6 hours" is ambiguous between the two; this
  note fixes the reading actually implemented.
