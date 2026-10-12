# Direct Copernicus ERA5 assessment

Last verified: 2026-09-02

## Decision

Use the direct Copernicus Climate Data Store (CDS) as the replacement for the
Open-Meteo ERA5 route. Direct CDS removes the Open-Meteo free-service
non-commercial restriction. The ERA5 products use the CC BY 4.0 license, which
permits commercial use with attribution. This license is suitable for a prize
entry. The PRC rules also require an external data set to be open, usable, and
documented. Direct CDS meets the license part of that rule. We must still publish
the exact requests, transformations, file hashes, and attribution.

Do not yet call event-hour ERA5 leakage-safe. ERA5 is reanalysis. It uses
observations and becomes available about five days after the event. The PRC
ranking months are January and July 2026, and the challenge starts on 1 September
2026, so those data exist before submission. However, the published PRC pages do
not define an operational prediction cut-off. Event-hour ERA5 is valid for a
retrospective contest interpretation, but it is invalid for a real-time taxi-out
prediction made before departure. Get written organizer confirmation before it
enters the final prize model.

Recommended retrieval:

1. Use `reanalysis-era5-single-levels-timeseries` for seven core variables at
   each airport point.
2. Use the stable parent product, `reanalysis-era5-single-levels`, for
   `low_cloud_cover` and `snowfall`, which the time-series product does not
   provide.
3. Retrieve NetCDF, one airport at a time, for 2024, 2025, and January plus July
   2026. Freeze every raw file with its request JSON, retrieval time, byte size,
   and SHA-256 hash.

This hybrid plan transfers only about 10--25 MiB. It needs 66 requests. The
fallback is to use the stable parent product for all nine variables, split by
airport and month. That fallback needs 286 small requests.

## Why direct CDS is suitable

The official PRC [eligibility page](https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html)
says that prize-eligible external data must be openly accessible and usable,
documented, and available under an open license. The direct ERA5
[single-levels catalogue entry](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
and the [time-series catalogue entry](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview)
both identify their license as CC BY 4.0. The official license record links to
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). That license permits
copying, adaptation, and use for any purpose, including commercial use, with
attribution.

The direct CDS route has no separate non-commercial API condition in the ERA5
data license. This is the main improvement over the free Open-Meteo endpoint.
CDS still requires account registration, acceptance of the dataset license, and
an API token. The PRC rules do not state whether free account-gated access counts
as "openly accessible." This is a small eligibility uncertainty that the team
should include in the organizer question.

Use this concise attribution in the repository and final paper, with the actual
access date: Copernicus Climate Change Service (C3S), ERA5 hourly data on single
levels, DOI `10.24381/cds.adbb2d47`, accessed YYYY-MM-DD.

The official [ERA5 citation example](https://confluence.ecmwf.int/spaces/CKB/pages/222493847/Use%2BCase%2B2%2BERA5%2Bhourly%2Bdata%2Bon%2Bsingle%2Blevels%2Bfrom%2B1940%2Bto%2Bpresent)
also asks for visible Copernicus attribution and a statement when the data were
modified.

## Product comparison

| Property | Hourly single-levels | Hourly single-levels time-series |
|---|---|---|
| CDS identifier | `reanalysis-era5-single-levels` | `reanalysis-era5-single-levels-timeseries` |
| Main form | Gridded area selection | One location or a small area |
| Grid | 0.25 degrees for atmosphere | 0.25 degrees for atmosphere; nearest grid point for a location |
| Time | Hourly, 1940 to present | Hourly, 1940 to present |
| Output | GRIB; NetCDF4 is marked experimental | NetCDF or CSV |
| Requested PRC variables | All nine are available | Seven are available; low cloud and snowfall are absent |
| Current request-cost limit | 121,000 units from the live official costing API | 760 units from the live official costing API |
| Service status | Supported parent catalogue product | Fast ARCO derivative; non-operational and can be changed, disabled, or deprecated |
| Best use here | Missing variables and fallback | Long point time series with low transfer volume |

The [parent dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
lists the requested temperature, dew point, pressure, wind, precipitation,
cloud, and snow variables. It reports a regular 0.25-degree atmospheric grid.
The [time-series overview](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview)
states that a location is mapped to the nearest grid point and that the product
returns hourly NetCDF or CSV.

The [ARCO product guide](https://confluence.ecmwf.int/spaces/CKB/pages/505390919/ERA5+hourly+Analysis+Ready+Cloud+Optimised+ARCO+data+on+single+levels+from+1940+to+present+Product+User+Guide+PUG)
shows that the time-series product is built from the parent ERA5 GRIB data. It
also gives two important warnings: the service is non-operational, and it can be
disabled or deprecated; its file formats and structures can change. Therefore,
the stable parent product is the required fallback.

### Current request limits

The general CDS documentation says that limits can change and that excess work
is queued. Its last published table gives a 120,000-field limit for hourly ERA5
single levels and recommends month-sized ERA5 requests. The live official
costing API returned these values on 2026-09-02:

| Request | Reported cost | Reported limit | Result |
|---|---:|---:|---|
| Time-series, seven variables, one point, 2024-01-01 through 2025-12-31 | 7 | 760 | Fits |
| Parent, low cloud plus snowfall, one airport cell, full 2024 | 105,408 | 121,000 | Fits |
| Parent, low cloud plus snowfall, one airport cell, full 2025 | 105,120 | 121,000 | Fits |
| Parent, low cloud plus snowfall, one airport cell, Jan and Jul 2026 | 17,856 | 121,000 | Fits |
| Parent, all nine variables, all 2024--2026 month choices in one request | 1,420,416 | 121,000 | Fails |

The official [CDS limits and efficiency guide](https://confluence.ecmwf.int/spaces/CKB/pages/601307425/Climate+Data+Store+CDS+documentation)
says that limits also apply to API calls, that values can change, and that
smaller ERA5 requests get better queue treatment. Before the full download,
repeat the cost checks through the official endpoints:

- `https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-single-levels-timeseries/costing`
- `https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-single-levels/costing`

## Variables and transformations

Use these nine source variables:

| Needed feature | ERA5 variable | Product | Unit and handling |
|---|---|---|---|
| Air temperature | `2m_temperature` | Time-series | K; optionally subtract 273.15 for degrees C |
| Humidity state | `2m_dewpoint_temperature` | Time-series | K; retain dew point instead of deriving relative humidity initially |
| Total precipitation | `total_precipitation` | Time-series | m of water equivalent over the hour ending at the timestamp; multiply by 1000 for mm |
| Snowfall | `snowfall` | Parent | m of water equivalent over the hour ending at the timestamp; multiply by 1000 for mm water equivalent |
| Liquid precipitation proxy | derived | Both | `max(total_precipitation - snowfall, 0)` after unit alignment |
| Mean sea-level pressure | `mean_sea_level_pressure` | Time-series | Pa; divide by 100 for hPa if wanted |
| Total cloud | `total_cloud_cover` | Time-series | Fraction from 0 to 1 |
| Low cloud | `low_cloud_cover` | Parent | Fraction from 0 to 1 |
| Wind components | `10m_u_component_of_wind`, `10m_v_component_of_wind` | Time-series | m/s |

The official [ERA5 accumulation guide](https://confluence.ecmwf.int/spaces/CKB/pages/197702790/Conversion%2Btable%2Bfor%2Baccumulated%2Bvariables%2Btotal%2Bprecipitation%2Bfluxes)
states that hourly total precipitation is the accumulation over the hour ending
at the validity timestamp. The [ERA5 parameter table](https://confluence.ecmwf.int/pages/viewpage.action?pageId=185086723)
gives snowfall in metres of water equivalent. This timestamp meaning must be
preserved in the join.

Derive wind speed as `sqrt(u10**2 + v10**2)`. Derive meteorological direction
with the tested language-specific `atan2` convention. The official
[ECMWF wind calculation guide](https://confluence.ecmwf.int/spaces/CKB/pages/133262398/ERA5%2BHow%2Bto%2Bcalculate%2Bwind%2Bspeed%2Band%2Bwind%2Bdirection%2Bfrom%2Bu%2Band%2Bv%2Bcomponents%2Bof%2Bthe%2Bwind)
defines direction as the direction from which the wind blows and gives the
component formulas.

## Expected size

The exact requested calendar contains 19,032 hours for each airport:

- 2024: 8,784 hours.
- 2025: 8,760 hours.
- January 2026: 744 hours.
- July 2026: 744 hours.

For 11 airports, this is 209,352 airport-hours. Nine float32 variables contain
1,884,168 values and have a raw numeric payload of 7,536,672 bytes, or 7.19 MiB.
Repeated time coordinates add about 1.60 MiB. File headers, coordinate arrays,
variable grouping, and compression determine the final amount.

Planning estimate:

- NetCDF: 10--25 MiB total.
- CSV: 25--60 MiB total.
- A Europe-wide 0.25-degree grid download would be orders of magnitude larger
  and is not needed.

These are engineering estimates, not CDS guarantees. Run the two-day tests
below and scale their measured bytes before the complete retrieval.

## Reproducible point extraction

Use the airport coordinates already recorded in this repository. The expected
nearest ERA5 grid cells below come from deterministic quarter-degree rounding.
The first CDS response is authoritative: assert its returned latitude and
longitude, record them in the manifest, and use the same exact cell in the
parent request.

| Airport | Requested latitude, longitude | Expected ERA5 cell |
|---|---|---|
| EDDF | 50.026, 8.543 | 50.00, 8.50 |
| EDDM | 48.354, 11.786 | 48.25, 11.75 |
| EGLL | 51.478, -0.461 | 51.50, -0.50 |
| EHAM | 52.309, 4.764 | 52.25, 4.75 |
| LEBL | 41.297, 2.078 | 41.25, 2.00 |
| LEMD | 40.494, -3.567 | 40.50, -3.50 |
| LFPG | 49.013, 2.550 | 49.00, 2.50 |
| LIRF | 41.804, 12.251 | 41.75, 12.25 |
| LTAI | 36.899, 30.800 | 37.00, 30.75 |
| LTFM | 41.275, 28.752 | 41.25, 28.75 |
| LSZH | 47.465, 8.549 | 47.50, 8.50 |

For each output file:

1. Require exactly one latitude and one longitude.
2. Require hourly UTC timestamps with no duplicates.
3. Require the expected row count for the selected period.
4. Confirm units from the NetCDF metadata before conversion.
5. Check temperature, pressure, cloud fraction, precipitation, and wind against
   broad physical bounds. Do not silently clip values.
6. Join on airport and UTC hour. Keep a missing-weather flag. Do not use nearest
   future time.
7. Save raw request JSON, response coordinates, bytes, SHA-256, retrieval time,
   dataset DOI, and license URL.

## Credentials

The official [CDS API setup page](https://cds.climate.copernicus.eu/en/how-to-api)
requires an ECMWF/CDS account and a personal access token. The user must also
accept the ERA5 dataset license through the web page before the API can retrieve
data. Install `cdsapi>=0.7.7` and place only these fields in
`$HOME/.cdsapirc` on the workstation:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

Set file mode `0600`. Do not commit, print, or copy the token into request JSON.
No CDS credential was present on this computer when this note was written.

## Practical test requests

First test the fast point product for EDDF and two complete days:

```python
import cdsapi

dataset = "reanalysis-era5-single-levels-timeseries"
request = {
    "variable": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
        "total_cloud_cover",
    ],
    "location": {"latitude": 50.026, "longitude": 8.543},
    "date": ["2024-01-01/2024-01-02"],
    "data_format": "netcdf",
}

cdsapi.Client().retrieve(dataset, request, "eddf-core-test.nc")
```

Then test the stable parent product for its two extra variables at the exact
cell returned by the first request:

```python
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["low_cloud_cover", "snowfall"],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01", "02"],
    "time": [f"{hour:02d}:00" for hour in range(24)],
    "area": [50.0, 8.5, 50.0, 8.5],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

cdsapi.Client().retrieve(dataset, request, "eddf-extra-test.nc")
```

Accept the license in the CDS web interface before either test. The API setup
page recommends generating request code from the current web form because field
names can change.

Test gates before the full retrieval:

- Both calls succeed twice without a manual repair.
- The returned cell is the same in both files.
- Each variable has 48 hourly values for the two-day test.
- Precipitation and snowfall have the documented previous-hour meaning.
- The measured full-download estimate stays below 100 MiB.
- A forward validation ablation shows a stable RMSE improvement.
- The organizers confirm that event-hour reanalysis and free account-gated CDS
  data are valid for prize eligibility.

## Minimal full retrieval plan

Use three periods: full 2024, full 2025, and January plus July 2026.

For each of 11 airports:

1. Submit three time-series requests for the seven core variables: one
   continuous request for 2024--2025, one for January 2026, and one for July
   2026. This is 33 requests.
2. Submit three parent requests for low cloud and snowfall: full 2024, full
   2025, and January plus July 2026. The live cost checks show that each request
   fits. This is 33 requests.
3. Limit concurrency to the current CDS user limit and let CDS queue requests.
   Do not retry a queued request as a new request.
4. Validate and hash each file before conversion.
5. Convert to one tidy Parquet file with one row per airport and UTC hour.
6. Keep the immutable NetCDF files and request manifests for reproducibility.

The exact-period plan therefore has 66 requests. A simpler 44-request option
uses one continuous 2024-01-01 through 2026-07-31 time-series request per
airport, then discards February through June 2026. That option transfers only
about one additional MiB of float payload and is the practical minimum.

## Update latency and leakage

The ERA5 catalogue says that ERA5 is updated daily with about five days of
latency. The first release is ERA5T. It can be revised when the final release is
made two to three months later. The ARCO time-series guide says that new data
normally enters the time-series store a few hours after it reaches ERA5, but it
also warns that delivery is non-operational and can be delayed.

This creates three time concepts:

1. **Event time:** the flight and weather hour in January or July 2026.
2. **ERA5 availability time:** about five days after the event.
3. **Competition submission time:** 1 September through 11 October 2026.

The official [PRC overview](https://ansperformance.eu/study/data-challenge/dc2026/)
sets the competition dates, and the official [data page](https://ansperformance.eu/study/data-challenge/dc2026/data.html)
states that ranking uses January and July 2026. Thus, the ranking weather was
available before the first submission date. That supports a retrospective
contest-clock interpretation.

It does not support an operational interpretation. ERA5 assimilates
observations and is not available at the flight's scheduled time, off-block
time, or take-off time. If the intended model must work before departure, join
only a forecast that was issued before the declared cut-off, or use historical
climatology. A time split in model validation does not repair this availability
leak.

July 2026 can also still be ERA5T during part of the competition. Freeze a
retrieval snapshot and record whether the files contain the early or final
release. Re-run only as a declared new data version.

## Adversarial review

### Strongest failure modes

1. **The organizer defines an operational prediction cut-off.** Event-hour ERA5
   then uses information from after the cut-off. This is the most serious risk.
2. **Account-gated CDS access is judged not openly accessible.** CC BY 4.0 solves
   the use license, but registration is still required for retrieval.
3. **The time-series product changes or closes.** Its own guide says this can
   happen and that it has lower operational support than the parent product.
4. **The 0.25-degree cell is not representative of runway weather.** Airport
   weather, coastal boundaries, terrain, and convection can differ materially
   inside a grid cell.
5. **Accumulation timestamps are joined incorrectly.** Total precipitation and
   snowfall describe the hour ending at the validity time. A one-hour shift can
   create false signal.
6. **ERA5T changes after retrieval.** July values can change during the
   competition, so results can become hard to reproduce without immutable raw
   files and hashes.
7. **Weather has little incremental value.** Airport, runway, queue, stand, and
   schedule features can absorb most of the signal. The existing EDA already
   suggests small and airport-specific weather effects.

### Evidence that would reverse this recommendation

Stop final-model use of event-hour ERA5 if the organizers state that every
feature must exist at an operational pre-departure cut-off. Replace it with an
open forecast archived by issuance time or with leakage-safe climatology.

Stop direct CDS use for the prize entry if the organizers state that free
registration is not "openly accessible," despite the open CC BY license.

Prefer airport METAR or another open observation source if a registered,
forward validation test shows a materially larger and stable RMSE improvement,
with an equally clear license and the same cut-off compliance.

Drop the weather lane from the production model, but keep the data and results,
if repeated forward, airport-held-out ablations show no stable gain. A negative
first ablation closes only that feature design. It should trigger checks of lag
choice, nonlinear wet/snow indicators, airport interactions, and runway-aligned
wind before the lane is judged weak.

## Organizer question

Send one precise question before final model use:

> The ranking flights are from January and July 2026, while submissions start in
> September 2026. For prize eligibility, can we use event-hour ERA5 reanalysis
> downloaded directly from the Copernicus Climate Data Store under CC BY 4.0?
> CDS needs free account registration and ERA5 becomes available about five days
> after each event. Is the prediction cut-off the submission date, or must every
> feature have been available before each flight's departure?

## Primary sources

- [PRC 2026 overview](https://ansperformance.eu/study/data-challenge/dc2026/)
- [PRC 2026 eligibility rules](https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html)
- [PRC 2026 data description](https://ansperformance.eu/study/data-challenge/dc2026/data.html)
- [ERA5 hourly single-levels catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
- [ERA5 hourly single-levels time-series catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview)
- [ERA5 ARCO product guide](https://confluence.ecmwf.int/spaces/CKB/pages/505390919/ERA5+hourly+Analysis+Ready+Cloud+Optimised+ARCO+data+on+single+levels+from+1940+to+present+Product+User+Guide+PUG)
- [CDS API setup](https://cds.climate.copernicus.eu/en/how-to-api)
- [CDS limits and efficiency guidance](https://confluence.ecmwf.int/spaces/CKB/pages/601307425/Climate+Data+Store+CDS+documentation)
- [ERA5 data documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=185086723)
- [ERA5 accumulation conversion guide](https://confluence.ecmwf.int/spaces/CKB/pages/197702790/Conversion%2Btable%2Bfor%2Baccumulated%2Bvariables%2Btotal%2Bprecipitation%2Bfluxes)
- [ERA5 citation and attribution example](https://confluence.ecmwf.int/spaces/CKB/pages/222493847/Use%2BCase%2B2%2BERA5%2Bhourly%2Bdata%2Bon%2Bsingle%2Blevels%2Bfrom%2B1940%2Bto%2Bpresent)
