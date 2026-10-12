# External data used by accepted entry v40

The adsb.lol method below supplies observations to v40 through its earlier
base and later clock/proxy stages. V35 also uses stand geometry from Gateway
and OpenStreetMap, documented in [GEOMETRY-DATA.md](GEOMETRY-DATA.md). The code
licence does not replace the data licences. Exact ERA5 data are packaged
separately under CC BY 4.0; see [weather-provenance.md](weather-provenance.md).
No original private Kaggle package is an open external source; rebuild it from
challenge data and these named open sources as [REPRODUCE.md](REPRODUCE.md) states.

The eligibility rules bind every external dataset a prize-eligible entry uses,
as recorded in the historical exp-042 surface assessment:

1. **openly accessible and usable**;
2. **openly available under an open source licence**;
3. **documented well enough that another person can reproduce the result**.

This file is condition 3 for the **adsb.lol surface observation**, the external
dataset that carries the entry. It is not the only one: `V3-allin` reads
Copernicus ERA5 (`docs/weather-provenance.md`) and the airport boxes come from
OurAirports, both listed in `docs/WRITEUP.md` §4. This note names the
source, its licence, every archive consumed with the digest its publisher signed, the
code that consumed it, the rules that turn bytes into a feature, and the path another
person walks to get the same numbers. It carries no leaderboard reading and no number
derived from the scoring process, under the team's score-feedback rule.

The exp-046 digest tables are
regenerated from the harvest's own run records by
`experiments/exp-046-surfacefit/code/provenance.py`.

---

## 1. The source

**adsb.lol `globe_history`** — daily archives of the aircraft position traces that the
adsb.lol volunteer receiver network collected, published as GitHub releases.

| | |
| --- | --- |
| Publisher | adsb.lol (community ADS-B network) |
| Repositories | `adsblol/globe_history_2025`, `adsblol/globe_history_2026` |
| Release form | one release per UTC day per instance; tag `v<YYYY>.<MM>.<DD>-planes-readsb-prod-0`; the day's `.tar` split into `~2 GB` assets `.tar.aa`, `.tar.ab`, … |
| Access | anonymous HTTPS from `github.com/adsblol/globe_history_<year>/releases/download/…`. **No account, no API key, no approval, no rate agreement.** |
| Licence | **ODbL-1.0** on the archive repositories and **CC0-1.0** on the contributed data: adsb.lol's licence page records that feeders "waive all copyright … under the CC0 license". <https://www.adsb.lol/privacy-license/> |
| Format | a tar of per-aircraft gzipped JSON trace files, one file per ICAO24 hex per day |
| Content | `trace` is a list of `[dt, lat, lon, alt, gs, track, …, aircraft]`; `alt` is the string `"ground"` when the aircraft reports a surface position; `aircraft.flight` is the broadcast callsign |
| Calendar | 2025: 364 of 365 days in the 2025 repository (2025-12-31 is published in the 2026 repository). 2026: January 31/31 and July 31/31 |

**Why this source and not another.** `exp-042-surface/REPORT.md` §2.3 prices eight
candidates against the same three conditions. OpenSky's historical database is the
better instrument technically (1 Hz against ~2 s) but its access is approval-gated to
university, governmental and aviation-authority applicants, which does not satisfy
condition 1 as written; ADS-B Exchange and the commercial feeds fail condition 2; the
2022 PRC/OpenSky archive and the EUROCONTROL research repository do not cover 2025 or
2026. adsb.lol is the only candidate that clears all three.

**Post-operations input rule.** The organizer's recorded answers permit open
trajectory observations of January and July 2026, including a flight's own
ground positions. This removes the earlier assumed cut-off. The open licence
and reproduction requirements still apply. See
[external-data-rules.md](external-data-rules.md).

## 2. What was consumed

One pass over a day's archive yields all ten airports, so the cost is per day, not per
airport-day. Each day is streamed over HTTPS into `tarfile` in stream mode, every member
decompressed, parsed and screened inside the loop, and dropped: **no archive is ever
written to disk**, on Kaggle or on the laptop. Only the filtered slice is kept.

The complete list of archives, their publisher digests and the digests computed on the
bytes actually read is in **`docs/external-data-archives.md`**, regenerated from
the per-day harvest and pilot run records. Every asset that was
read to completion matches the digest the publisher signed; the two exceptions on the
pilot's first two days are stated in that file with their cause.

## 3. The filter: from a day's archive to ground points

`experiments/exp-045-harvest/kernels/surface/prc-harvest.py`, run as one Kaggle CPU
kernel per calendar month (`kernels/surface-<month>/`, committed as it ran).

1. **Ten airport boxes.** `experiments/exp-045-harvest/code/boxes.py` builds each box as
   the bounding rectangle of every open runway endpoint and the airport reference point
   from OurAirports `runways.csv` and `airports.csv` (public domain), grown by 3 km on
   all four sides and rounded outward. The 40 runway rows used are committed as
   `analysis/runways-10.csv` with the source SHA-256; the boxes are in `analysis/boxes.json`
   and repeated in every `run_<day>.json`. No box reaches a neighbouring aerodrome.
2. **Every trace is parsed.** Latitude and longitude are extracted for all ~65,000 traces
   of the day and the trace's own bounding box is tested against the ten boxes; the ~99 %
   that come nowhere near an airport are dropped before any per-point work. There is no
   text prefix screen, so the census of aircraft inside a box is complete by construction.
3. **The callsign is carried.** readsb writes the aircraft object only when it changes, so
   `aircraft.flight` is filled forward and then backward over the whole trace before any
   point is dropped.
4. **What is kept.** For each box: every point inside the box whose altitude field reads
   `"ground"`, plus the first airborne point after a ground run that was inside that box
   (the take-off). Columns kept per point: `airport, icao, t, lat, lon, gs_kt, track,
   alt_ft, on_ground, first_airborne, in_box, callsign`.

## 4. The off-block rule

`experiments/exp-043-pilot/code/segments.py`, widened to every ground segment by
`experiments/exp-045-harvest/code/allsegments.py`. The rule and its five constants were
**frozen on the pilot's development days (2025-07-07 and 07-08) and did not move on the
held-out days**; `exp-045/code/check_rule.py` and
`experiments/exp-046-surfacefit/code/test_block.py` §A0 prove that `all_segments()`
reproduces it row by row.

A ground segment is a run of on-ground points of one aircraft inside one box with no
reception hole longer than `GAP = 1800 s`, split at an airborne point. Within it:

| Constant | Value | Meaning |
| --- | --- | --- |
| `GAP` | 1800 s | a larger reception hole splits the segment |
| `MOVE_KT` | 3 kt | ground speed above this is motion (displacement between neighbouring points where the broadcast speed is missing) |
| `HOLD_S` | 30 s | motion must be held this long to count as sustained |
| `TURNAROUND` | 600 s | a stationary run at least this long is a stand turnaround, and the segment is anchored after the last one |
| `STAND_STOP` | 60 s | observed stationary time that makes a motion onset "strict" |
| `DISP_M` | **250 m** | the served clock: the first time the aircraft is this far from the stand position and never returns inside that radius |

`obs_offblock` is the **250 m displacement onset**, and `obs_wheels_off` is the last
on-ground point before the take-off. The pilot chose 250 m by a sweep on the development
days alone (`exp-043/analysis/sweep.json`): accuracy improves monotonically as the event
becomes more decisive, reaching a flat 143–148 s plateau by 6–8 kt or 200–300 m and
degrading beyond it, and the frozen rule takes the midpoint of the measured plateau. A
0.5 kt gate puts the median offset at +0.9 s but its RMSE at 218 s, because at that
threshold the trigger is GPS jitter at a random moment.

The stand position is the median latitude and longitude of the stationary points the
rule anchored on. `stand_stop_s` (the stationary time actually observed before the
onset) is reported as a confidence margin, never used as a filter.

### 4a. The rule has versions, and every number says which one it rests on

The rule above was shipped on 2026-09-07 and has been changed twice since. Both changes
are recorded here because a summary file is only meaningful beside the version that
wrote it.

| version | `summaries_rule_sha256` | what changed |
| --- | --- | --- |
| `shipped-2026-09-07` | `a8cf1b7441fc51d4a7d118eafc8d42e7770542b5ac30b8f77286f1381337ba09` | any first non-ground point after a ground run is the take-off |
| `repair-2026-09-08` | — | **that point must also carry a real altitude.** `readsb` writes a null altitude on some first-airborne points; such a point is not a take-off, and a ground run that ends in one is not a departure (`exp-043-pilot/analysis/divergence.json`: 6 of 5,395 departure segments, 0.07 s of held-out RMSE) |
| `repair-2026-09-08+stable-order` (laptop re-derivation) | `15a72d550241faa28bc3af1acaa5e7f34e448065f8d80cb3b518d3e885796fa9` | the same repaired rule, run on this laptop over the pilot's stored points by `exp-046-surfacefit/code/summaries.py`. Its identity is a hash of the two SOURCE FILES, while the harvest's is lifted from the kernel block that actually ran, so the two identities cannot agree even when the rule does. The resolver therefore rejects these seven EHAM days from any fold table or serve, and says so in `analysis/arms.json` |
| `repair-2026-09-08+stable-order` | `6ae78fec2d9047ff3b399b7b56c8257c34d02844142fcd7323139e277ffb91f6` | **the trace is sorted stably.** 92 of one day's 553,178 points share an `(airport, icao, t)` with another and 22 of those pairs disagree about `on_ground`, so an unstable sort split the ground run at a different point on a different machine — 14 of 8,301 segments between a laptop and a Kaggle session (`exp-045-harvest`, commit `afc77ee`) |

Three mechanisms keep a measurement on one version:

* **The manifest states it per day.** The generated harvest manifest carries
  `summaries_rule_sha256` for every day it delivered, and the harvest re-derives an
  already-run month from its stored points rather than refetching it.
* **The loader refuses a mixture.** `experiments/exp-046-surfacefit/code/block.py`
  resolves one version across every fold month and every ranking month at once
  (`one_rule`), reads only the days that carry it, and reports the rest in
  `days_wrong_rule`. Without a version it reads nothing at all: a diagnostic that wants
  whatever is on disk must say `allow_any_rule=True`.
* **Where the file carries the evidence, the repair is normalised on read.** A summary
  written under `shipped-2026-09-07` that carries the `airborne_alt_ok` column is
  corrected in memory, so it gives the same rows as one written after the repair.

The pilot's `segments.py` is frozen at the shipped sort order, so it and the harvest's
`all_segments()` differ on exactly the aircraft whose points carry an equal-instant tie.
The identity test sets those aircraft aside and asserts the two implementations agree
row by row on the rest (`exp-046-surfacefit/code/test_block.py` §A0,
`exp-045-harvest/code/check_rule.py`).

## 5. The match rule

`experiments/exp-046-surfacefit/code/block.py`. A departure of the supplied file is
matched to a ground segment when

* **any callsign the segment broadcast** equals the departure's broadcast callsign, and
* the segment's observed `obs_wheels_off` is within **±15 min** of the departure's
  `MVT_TIME_UTC_mvt`, and
* both belong to the same airport.

A ground run is one continuous surface movement of one aircraft, and 23 % of them carry
more than one callsign form, because an aircraft that lands under one flight number
departs under another and it is still one run. The summary therefore records both the
modal form (`callsign_modal`) and every form seen (`callsigns_all`, `|`-delimited), and
the join reads all of them, with leading zeros of the numeric run normalised on both
sides (`ABC0123` and `ABC123` are one name). Ties are resolved to the modal form and the
form that matched is carried as the feature `matched_on_modal`. Reading the modal form
alone — which this rule did until 2026-09-08 — loses 9.3 points of EDDF, 8.0 of LIRF and
5.4 of EGLL on 2026-01.

The departure's broadcast callsign is `CALLSIGN_flt` where the Network Manager join
exists (98.5 % of ranking departures) and, where it does not, the flight number's letter
prefix mapped through an **airline prefix map learned on the nine 2025 months that are
not fold months** (2025-02 to 2025-06 and 2025-08 to 2025-11), **departure rows only**.
Nine and not twelve: the map is then out of fold for every fold at once, and it is one
and the same function on the fitting side and the served side. The months are fixed
rather than "the months other than this one" because a second lane building the same
block from a different set of months produces a different reconstructed callsign, and
the match flips on exactly the rows where `CALLSIGN_flt` is null — 30 ranking rows on
2026-01 and 2026-07 when the two were allowed to differ. Ambiguity is resolved by
nearest wheels-off in both
directions and counted, never dropped silently. Summaries for the take-off day and the
day either side are loaded, and the window discriminates; the day of the observation is
never taken from `BLOCK_TIME_UTC_mvt`, which the ranking file blanks.

**The wide rule.** A departure the rule above leaves unmatched is matched a second time,
on the same `(airport, callsign)`, when a segment of that airport carries an off-block
instant that precedes the take-off by a plausible taxi time (60 s to 7,200 s) and whose
ground run ends within ±60 min of the take-off. A take-off inside ±60 min is preferred
where one exists. The reason is physical: at EGLL and EDDF the network loses the aircraft
before the runway, so its ground run has no airborne successor at all and the strict rule
cannot see it — but the pushback it saw at the stand is unaffected, and `obs_taxi` is
`MVT_TIME − obs_offblock` and never reads the take-off. Three guards keep it honest: an
ARRIVAL segment is excluded (its ground run begins at the runway, so its off-block is a
landing roll-out); the ground run must end near the take-off, so a callsign flown twice in
a day cannot lend the morning pushback to the evening flight; and the assignment is
one-to-one in both directions. The mode of every match is carried as a feature
(`match_mode`, `clock_ok`, `has_wheels_off`) and every table is reported by reach class,
so a wide match is never counted as a strict one.

**No label enters the feature block.** `block.py` reads a fixed list of ranking-legal
columns and `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are not among them, so the
identical code computes the block for a masked validation month and for the ranking
file. The features it produces are the observed taxi estimate
`obs_taxi = MVT_TIME_UTC_mvt − obs_offblock`, the observation's quality (points, median
interval, largest hole, observed stand stop, time from first motion to 250 m, time seen
before moving), the stand position and its distance and bearing from the airport centre,
the arrival-side segment of the same aircraft where present, and the reach flags.

## 6. Reproduction path

Everything below runs from a clone of this repository plus the supplied competition
files; the external archives are fetched from their public URLs and never stored.

```
# 1. the ten boxes, from OurAirports (public domain).  The two CSVs are the published
#    runways.csv and airports.csv; the third argument is the UTC day they were retrieved
#    and is written into the boxes record.
.venv/bin/python experiments/exp-045-harvest/code/boxes.py \
    <path/to/runways.csv> <path/to/airports.csv> 2026-09-08

# 2. the frozen rule is unchanged
.venv/bin/python experiments/exp-045-harvest/code/check_rule.py

# 3. one month of observations: push the kernel, wait, pull the summaries
bash experiments/exp-045-harvest/code/push_kernel.sh <YYYY-MM>
bash experiments/exp-045-harvest/code/pull_kernel.sh  <YYYY-MM>

# 4. the receipt census by airport
.venv/bin/python experiments/exp-045-harvest/code/census.py <YYYY-MM>

# 5. the feature block, and its match test against the pilot's published counts
.venv/bin/python experiments/exp-046-surfacefit/code/test_block.py

# 6. the pilot's EHAM numbers through this pipeline
.venv/bin/python experiments/exp-046-surfacefit/code/reproduce_pilot.py

# 7. which of receipt, box or clock fails at each airport, with no label
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/reachwhy.py <YYYY-MM>

# 8. the fold table and the served candidate.  serve.py writes to the candidate path
#    only when the fold gate of analysis/arms.json and the per-airport RMS gate both
#    pass; otherwise it writes work/dry-run.parquet and analysis/serve-dryrun.json.
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/arms.py
OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/serve.py <arm>

# 9. this digest appendix, regenerated from the per-day run records
.venv/bin/python experiments/exp-046-surfacefit/code/provenance.py
```

A day of one archive is about 3.5 GB of transfer and 10–25 minutes of one Kaggle CPU
kernel. Nothing in the path needs a GPU, an account with the data publisher, or a paid
service.

## 7. Attribution

Data © adsb.lol contributors, made available under the Open Database Licence (ODbL-1.0)
with the contributed contents under CC0-1.0. Airport and runway geometry from
OurAirports (public domain). Neither publisher endorses this work.

The v37 and v40 monthly-mean correction uses EUROCONTROL taxi-out statistics.
[EUROCONTROL-DATA.md](EUROCONTROL-DATA.md) gives the source, digest, attribution
and separate data terms. The source workbook is not included.
