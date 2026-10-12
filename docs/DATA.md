# Data used by the release target zabd

This page lists every data source that the release target `zabd` (candidate
SHA-256 `e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636`)
depends on: the challenge files and six external sources. For each source it
gives what was used, by which part of the model, the licence or terms, and
where the data is hosted. [LINEAGE-ZABD.md](LINEAGE-ZABD.md) maps the sources
to the model parts. Earlier entries are covered by
[EXTERNAL-DATA.md](EXTERNAL-DATA.md), [GEOMETRY-DATA.md](GEOMETRY-DATA.md) and
[weather-provenance.md](weather-provenance.md).

The export redistributes no raw challenge rows, no labels and no raw external
data. It holds source code, saved model values without movement IDs or labels
(`release/saved-vectors/zabd/`), and two small derived tables that earlier
entries already shipped: the airport boxes and the stand geometry table.

| source | used by | licence or terms | hosting | in the export |
|---|---|---|---|---|
| Challenge files | every part | challenge terms; not redistributed | challenge organizer | no |
| adsb.lol `globe_history` | route, head, MLP, all six sequence and row members, clock class | ODbL 1.0 (archives), CC0 1.0 (contributed data) | GitHub releases of `adsblol/globe_history_2025` and `_2026` | no (code only) |
| OurAirports | airport boxes of the adsb.lol harvest; older route base members | public domain | ourairports.com | boxes (`experiments/exp-045-harvest/analysis/boxes.json`) |
| X-Plane Scenery Gateway | stand geometry for route, head, MLP and sequence members | GPL-2.0 per upload terms | gateway.x-plane.com | stand table (`experiments/exp-127-clock-precision/stand_geometry.csv`) |
| OpenStreetMap | stand-name fallback in the same table | ODbL 1.0 | openstreetmap.org (Overpass API) | in the stand table |
| OPDI v0.0.2 | head, MLP, all six sequence and row members | no licence named; see [OPDI](#opdi) | eurocontrol.int and opdi.aero | no |
| Copernicus ERA5 | older route base members only | CC BY 4.0 | Copernicus Climate Data Store; the derived table: release `era5-v1` of this repository ([era5_hourly.parquet](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/era5_hourly.parquet)) | separate bundle (`release/ERA5-DATA-MANIFEST.json`) |

No part of `zabd` reads METAR, NOTAM, OpenSky state vectors or the
EUROCONTROL Taxi-Out Additional Time workbook. The workbook fed the v40
correction only; see [EUROCONTROL-DATA.md](EUROCONTROL-DATA.md).

## Challenge files

The replay reads four official files and checks each by SHA-256 before use
(`release/ZABD-PACK-ranking.json`, field `official_sha256`): the submission template,
the 2026 ranking file and the January and July 2025 training files. The 2025
labels (`TAXITIME_SEC_mvt` of departures) refit the level-2 blend weights and
two rule statistics (C95 and C98; C71 is fixed). The 2026 ranking file has no
labels. The member models were fit on the other ten months of 2025 for the
held predictions (one exception: the v32 half of the route). For the 2026
predictions most members were refit on all twelve months; the clock head uses
its ten-month fit and the route's final level 2 uses nine months
([LINEAGE-ZABD.md](LINEAGE-ZABD.md#members)).

## adsb.lol

ADS-B surface positions from the adsb.lol `globe_history` archives give the
observed departure clocks (`obs_*`), track features (`trk_*`) and the
proxy features of the route members. The harvest and feature code is in
`experiments/exp-045-harvest/`, `experiments/exp-046-surfacefit/code/`,
`experiments/exp-134-proxies-in-learner/code/` and
`experiments/exp-159-clock-head/code/`. Licence: ODbL 1.0 for the archives,
CC0 1.0 for contributed data
(<https://www.adsb.lol/privacy-license/>). Hosting:
<https://github.com/adsblol/globe_history_2025> and
<https://github.com/adsblol/globe_history_2026>. Details and attribution are
in [EXTERNAL-DATA.md](EXTERNAL-DATA.md) and
[external-data-archives.md](external-data-archives.md).

## OurAirports

`airports.csv` and `runways.csv` define the ten airport boxes of the adsb.lol
harvest. Terms: "All data is released to the Public Domain"
(<https://ourairports.com/data/>).

## X-Plane Scenery Gateway and OpenStreetMap

The stand geometry table `experiments/exp-127-clock-precision/stand_geometry.csv`
holds stand positions from Gateway `apt.dat` packs (GPL-2.0 under the
Gateway upload terms; each row names its scenery pack) and OpenStreetMap
parking positions (ODbL 1.0) as a fallback. The track features `trk_survey_*`
read it. Builders: `experiments/exp-127-clock-precision/build_stand_geometry.py`.
Terms and the open licence questions of the mixed table are in
[GEOMETRY-DATA.md](GEOMETRY-DATA.md); the GPL-2.0 text is in
`docs/data-licences/GPL-2.0.txt`.

## OPDI

The Open Performance Data Initiative (OPDI) publishes flight-list and
flight-event tables, release v0.0.2, built with OpenSky Network data.
Use in `zabd`: six `opdi_*` features in the frame of the clock head, in the
MLP and in all six sequence and row members. For three OPDI events of a
departure (exit parking position, exit apron, entry taxiway) they give a lead
time in seconds and a distance in metres; a departure with a duplicate OPDI
identity gets no value. The dark18 member sets 38 clock, observation, track
and OPDI values and flags to zero on selected training airport-days. The
per-departure OPDI table is read by
`experiments/exp-159-clock-head/code/frame.py`; the extractor that builds it
ships in `refit-src/` ([REPRODUCE.md](REPRODUCE.md#s2-opdi-first-event-tables), S2).

- Hosting: <https://www.opdi.aero/data> (OPDI v0.0.2; the data portal pages
  <https://www.opdi.aero/flight-list-data.html> and
  <https://www.opdi.aero/flight-event-data.html> list every file). Each file has
  its own URL:
  `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_<YYYYMM>.parquet`
  and
  `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_<YYYYMMDD>_<YYYYMMDD>.parquet`
  (ten-day windows). The folder `…/OPDI/v002` itself has no listing (HTTP
  403). On 2026-10-11 both files that we pinned served with no login and kept
  their SHA-256 (`flight_list_202607.parquet` `5b38d9bc…`,
  `flight_events_20260709_20260719.parquet` `bd06eb4e…`).
- Licence: the OPDI data pages and the Parquet file metadata name no licence.
  The OPDI About page states: "The focus of the OPDI is on open data, in
  particular open access and open use of data for non-commercial purposes"
  (<https://www.opdi.aero/about>). The general OpenSky terms require written
  permission for commercial use
  (<https://opensky-network.org/about/terms-of-use>).
- Challenge guidance: in the challenge's public Discord channel the organizer
  answered on 2026-09-06 that OPDI can be used when a purpose for it is found,
  and on 2026-09-08 that open-source data are allowed when declared in the
  solution's documentation or repository (paraphrase). The purpose here is a
  second departure clock where the ADS-B clock is missing.
- The export ships no OPDI rows and no table derived from them. A reader
  downloads v0.0.2 from the hosting pages above.

Because no licence is named, OPDI is the one source whose open-licence status
is not settled by its own terms; this page declares its use.

## Copernicus ERA5

Two ERA5 single-level products feed weather features of older base members
inside the route's v32 half. Licence: CC BY 4.0. The exact derived table is a
separate bundle described by `release/ERA5-DATA-MANIFEST.json`; provenance is
in [weather-provenance.md](weather-provenance.md).
The three bundle files are the assets of the release `era5-v1` of this
repository: [era5_hourly.parquet](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/era5_hourly.parquet),
[ERA5-DATA-MANIFEST.json](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/ERA5-DATA-MANIFEST.json) and
[NOTICE.txt](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/NOTICE.txt). [REPRODUCE.md](REPRODUCE.md) gives the
download and verify commands.
