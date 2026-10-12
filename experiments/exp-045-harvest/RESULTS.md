# exp-045-harvest — results

Worker `harvest`, tier hard. Data stage of the surface-observation programme (recorded rule): ten
airports, fourteen months of adsb.lol ground observations. `SPEC.md` holds the reading,
the boxes and the order; `manifest.json` holds every address and hash.

**Status: complete.** All fourteen months are in: **427 of 427 days, none missing**,
3,570 airport-days and 3,484,267 summary rows, from 1,320 GB streamed through 42.2
Kaggle kernel hours. Every day carries the same rule, `repair-2026-09-08+stable-order`
(`d4c59868…`). Tables are rendered from `analysis/` by `code/tables.py`, never typed.

---

## 1. Coverage is a property of the month, not of the airport

Receipt rate = the share of an airport's departures whose broadcast callsign is seen on
the ground inside the airport box, in a window around the actual take-off
(`code/census.py` states it exactly). It is an **upper bound** on the rows a feature can
use, not a match rate. Seven months are in, each over all of its days:

<!-- table:receipt -->
| Month | days | departures | EHAM | LFPG | EGLL | LTFM | LEMD | EDDF | LEBL | LIRF | EDDM | LSZH | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-01 | 31 | 153,706 | 98.0 | 0.0 | 57.1 | 0.0 | 0.0 | 81.2 | 97.7 | 0.0 | 97.1 | 93.3 | **48.0** |
| 2025-02 | 28 | 143,732 | 98.2 | 0.0 | 59.4 | 0.0 | 0.0 | 90.5 | 98.5 | 0.0 | 90.3 | 94.4 | **49.9** |
| 2025-03 | 31 | 164,449 | 98.0 | 0.0 | 37.0 | 0.0 | 74.4 | 91.4 | 98.4 | 0.0 | 90.4 | 93.7 | **55.4** |
| 2025-04 | 30 | 175,288 | 82.4 | 0.0 | 8.7 | 0.0 | 95.7 | 90.6 | 98.4 | 0.0 | 84.0 | 94.2 | **51.9** |
| 2025-05 | 31 | 185,202 | 98.4 | 0.0 | 8.7 | 0.0 | 97.9 | 88.6 | 98.7 | 20.5 | 93.3 | 93.7 | **56.9** |
| 2025-06 | 30 | 183,142 | 94.2 | 1.5 | 12.0 | 0.0 | 94.0 | 82.4 | 94.3 | 69.2 | 64.7 | 87.9 | **56.3** |
| 2025-07 | 31 | 190,713 | 96.2 | 4.8 | 15.1 | 0.0 | 28.8 | 82.9 | 89.2 | 90.6 | 36.9 | 88.5 | **49.6** |
| 2025-08 | 31 | 191,182 | 96.4 | 4.0 | 20.9 | 0.0 | 27.1 | 79.9 | 94.4 | 82.7 | 82.1 | 88.5 | **53.2** |
| 2025-09 | 30 | 183,950 | 95.7 | 5.2 | 22.1 | 0.0 | 87.8 | 89.3 | 90.2 | 18.4 | 86.1 | 89.0 | **55.5** |
| 2025-10 | 31 | 185,674 | 95.6 | 8.6 | 27.8 | 0.0 | 93.8 | 80.7 | 91.5 | 38.2 | 93.8 | 90.4 | **58.8** |
| 2025-11 | 30 | 162,332 | 96.8 | 21.0 | 52.5 | 0.0 | 95.0 | 93.5 | 94.7 | 40.7 | 97.4 | 92.0 | **64.7** |
| 2025-12 | 31 | 165,677 | 96.5 | 9.2 | 57.0 | 0.0 | 96.5 | 92.6 | 95.9 | 62.4 | 97.1 | 95.9 | **65.2** |
| 2026-01 | 31 | 152,719 | 94.3 | 8.8 | 76.6 | 0.0 | 96.3 | 58.9 | 92.6 | 66.0 | 97.4 | 93.9 | **62.7** |
| 2026-07 | 31 | 192,122 | 96.3 | 7.3 | 97.2 | 0.0 | 60.7 | 43.3 | 95.0 | 0.1 | 96.9 | 93.7 | **56.1** |
<!-- /table:receipt -->

<!-- table:weighted -->
Weighted by the ranking file, which is January and July 2026:

| Airport | ranking departures | share | receipt 2026-01 | receipt 2026-07 | reachable |
| --- | ---: | ---: | ---: | ---: | ---: |
| EDDM | 25,966 | 7.5 % | 97.4 % | 96.9 % | 25,207 |
| EHAM | 38,182 | 11.1 % | 94.3 % | 96.3 % | 36,441 |
| LEBL | 30,080 | 8.7 % | 92.6 % | 95.0 % | 28,293 |
| LSZH | 23,150 | 6.7 % | 93.9 % | 93.7 % | 21,718 |
| EGLL | 39,840 | 11.6 % | 76.6 % | 97.2 % | 34,766 |
| LEMD | 36,954 | 10.7 % | 96.3 % | 60.7 % | 28,410 |
| EDDF | 36,317 | 10.5 % | 58.9 % | 43.3 % | 18,161 |
| LIRF | 26,899 | 7.8 % | 66.0 % | 0.1 % | 7,320 |
| LFPG | 39,872 | 11.6 % | 8.8 % | 7.3 % | 3,182 |
| LTFM | 47,581 | 13.8 % | 0.0 % | 0.0 % | 5 |
| **total** | **344,841** | | **62.7 %** | **56.1 %** | **203,503 = 59.01 %** |

The **59.01 %** headline is a count of scored rows: every departure in the ranking file weighted equally, which is each airport's departures in each scored month at that month's own rate (203,503 of 344,841). The month cells are the months' own rates over all ten airports. Weighting each airport's two months equally instead gives **59.69 %**, because July carries 39,403 more departures than January and is the worse month; the two differ by 0.68 points. A third weighting, the unweighted mean of the 20 airport-month cells, reads 63.77 %, and is the one to avoid: it gives LTFM's 47,581 unreachable rows the same weight as LSZH's 23,150 good ones.
<!-- /table:weighted -->

**The scored set is 59.01 % reachable** — 203,503 of the ranking file's 344,841 rows,
every row weighted equally — but the two halves are not the same problem:
62.7 % of the January departures and 56.1 % of the July ones, and the airports that go
missing differ. LTFM is dark in both. **LIRF is dark in July only** — 66.0 % in January
and 0.1 % across all 31 days of July 2026, with zero ground points in the box on
2026-07-15 against 10,023 on 2026-01-15, while EHAM and EDDM are healthy that same day.
It is Rome's feeders, not the archive and not the box. LFPG is weak in both.

Four airports hold above 92 % in both scored months — EDDM, EHAM, LEBL and LSZH — which
is 117,378 ranking departures, 34 % of the file. Only EDDM holds above 95 % in both.

### The receipt number was wrong until 2026-09-08, and by 9 points

Every figure above is 6 to 12 points lower than the one this file carried before
2026-09-08, because `census.rate()` did not implement the definition at the head of
this section. It aggregated each callsign's ground appearances to `min(t_first)` and
`max(t_last)` **over the whole month** and then applied the take-off window, which asks
"does this callsign's month-long span overlap the window" — true for every day of the
month once a flight number is seen once. Heathrow's flight numbers repeat daily and its
reception does not, so EGLL priced at 83.2 % in March 2025 where the stated definition
gives 37.0 %.

The identity that catches it is now `code/check_receipt.py`, run for every month: a
departure counted as received must have been observed as a departure segment in its
airport's box, so `received <= departure segments written`. The old answer broke that in
**18 of 80 airport-months** — it priced EGLL in March 2025 at 16,023 received against
13,752 departure segments actually written. The repaired answer breaks it in none of 68.
At flight level, of 200 sampled EGLL misses on 2025-03-10, 199 have no ground sighting
at Heathrow that day at all and 195 have one on another day of the month — which is
exactly the channel the aggregate opened.

One casualty is worth naming, because it is an hour old: commit `c4ef54d` reads March
2025 at Heathrow as a flat 85 to 91 % that starts declining on March 27. Under the
repaired definition Heathrow is erratic from the start of that month (62, 50, 51, 59 …
12, 3, 22 …) and the decline has no edge to point at. The commit message stands as
written; this file is the current view.

Two consequences run through the rest of this file. **The old number hid outages**: a
callsign seen before and after a dark fortnight counted through it, so EHAM's April 2025
dip to 54 % (Apr 5 to 21) read as a flat 97.8 % month. And **the gap between the two
numbers is itself a measurement** — the share of departures whose callsign the network
hears on another day but not on this one — so it is kept in `analysis/receipt.json` as
`receipt_rate_anyday` rather than deleted.

### The two-day gate was wrong, and the month says why

The gate reported 63.0 % over two days and called EGLL, LIRF and EDDF weak alongside
LFPG. **Its total was right and three of its airports were not.** The corrected month
is 62.7 % against the gate's two-day 62.3 %, so the headline this section used to carry
— that the month refuted the gate — was an artefact of the inflated number repaired
above, and it is withdrawn. What survives is per airport, where two days still cannot
stand for a month: the gate is 26.6 points low at EGLL, 10.7 high at EDDF, 7.4 high at
LFPG, and within 2 points at the other seven. Receipt by airport and day is in
`analysis/receipt.json` under `by_airport_day`; the shape is not noise:

| Airport | Jan 1 | Jan 5 | Jan 12 | Jan 19 | Jan 26 | Jan 29 | Jan 31 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGLL | 51 | 76 | 71 | 86 | 92 | 91 | 70 |
| LIRF | 70 | 57 | 66 | 66 | 54 | 64 | 82 |
| EDDF | 73 | 52 | 59 | 67 | 61 | **13** | **10** |
| EHAM | 98 | 70 | 98 | 95 | 98 | 90 | 97 |
| LFPG | 16 | 19 | 6 | 1 | 7 | 4 | 5 |
| LTFM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Three separate things are in that table, and only the month-by-day view separates them:

- **EGLL climbs out of its first week; LIRF does not climb at all.** Heathrow opens at
  51 and 49 %, reaches 84 % by Jan 11 and holds 67 to 93 % after, so its 39,840 ranking
  departures move from the gate's "half reachable" to 76.6 % — the largest correction
  the month makes, and much smaller than this file used to claim. Rome, by contrast,
  sits at 47 to 84 % all month for a mean of 66.0 %: the gate's 67.6 % was not a bad
  first week, it was Rome's ordinary January. (LIRF then goes dark entirely: 0.1 %
  across all of July 2026, the other scored month — see the coverage series below.)
- **EDDF lost its receivers, from a lower base than this file used to say.** Frankfurt
  runs 52 to 73 % through January — not 91 to 98 % — and then falls to 13, 10 and 10 %
  on Jan 29 to 31. The outage is real and costs EDDF about 5 points of month rate; the
  healthy base it fell from was never healthy. A model fitted on January must not read
  those three days as EDDF's normal reception, and must not read the other 28 as full
  coverage either.
- **LFPG is genuinely weak, and getting weaker.** It never exceeds 20 % and ends the
  month at 5 %. LTFM is 0.0 % on all 31 days.

The pilot's leave-one-day-out gain of 8,368 s² per matched row is measured at EHAM only;
if a comparable gain holds wherever the network reaches, that is **59.01 % of the ranking
file** (203,503 of 344,841 rows) against a 1,108 s² gap to the leader. What the programme
cannot do is close the gap uniformly: LFPG and LTFM are 25 % of the file and stay
unreachable, and EDDF and LIRF lose a further 35,771 rows between them — 10 % of the
file — to reception that is present but partial.

<!-- table:nm -->
| Month | all | with an NM join | with no NM join | no-NM departures |
| --- | ---: | ---: | ---: | ---: |
| 2025-07 | 49.6 | 49.9 | 34.4 | 3,938 |
| 2026-07 | 56.1 | 56.4 | 35.7 | 2,837 |
| 2026-01 | 62.7 | 63.2 | 30.5 | 2,453 |
| 2025-12 | 65.2 | 65.5 | 31.7 | 1,649 |
| 2025-01 | 48.0 | 48.2 | 23.3 | 1,435 |
| 2025-02 | 49.9 | 50.2 | 16.2 | 1,110 |
| 2025-03 | 55.4 | 55.6 | 21.1 | 1,082 |
| 2025-04 | 51.9 | 52.1 | 19.4 | 1,085 |
| 2025-05 | 56.9 | 57.2 | 27.4 | 1,605 |
| 2025-06 | 56.3 | 56.5 | 34.8 | 2,502 |
| 2025-08 | 53.2 | 53.4 | 32.8 | 2,330 |
| 2025-09 | 55.5 | 55.9 | 29.6 | 2,465 |
| 2025-10 | 58.8 | 59.1 | 31.1 | 1,913 |
| 2025-11 | 64.7 | 65.0 | 24.3 | 1,356 |
<!-- /table:nm -->

**By NM join.** The no-NM rows — the ones with no `AOBT_3_flt` at all, which carry about
60 % of the naive predictor's squared error — are received at **30.5 %** (2,453 rows),
against 63.2 % for rows with an NM join. The surface clock is weakest exactly where the
incumbent is absent. This is the pilot's finding repeated at scale: its EHAM week read
99.8 % on NM rows and 24 % on no-NM rows. The gate's figure for this was 26.9 %, on 2 of
31 days.

### Why LTFM and LFPG are weak, and the repair that was tried

The cause is **not** the boxes. `code/edges.py` (§2) shows no airport has a single
departure stand within 500 m of a box edge, LFPG's nearest stand is 1.6 km inside its
west edge, and the boxes are computed from open runway geometry rather than drawn by
hand. The two failures have different causes:

- **LTFM is dark at every altitude, not only on the ground.** Only 79 and 156 aircraft
  were seen anywhere inside the Istanbul box on the two days, against 500 to 700 at every
  other airport, and **none of them had a single on-ground point**. adsb.lol barely
  receives Istanbul at all; this is feeder density in Turkey, not a surface-reception
  problem, and no filter change reaches it.
- **LFPG is received on the ground only in snatches.** 338 and 388 aircraft had at least
  one ground point — a normal count — but the box holds only 2,863 and 3,260 ground
  points in total, about 7 per aircraft at a 10 to 12 s median interval, against EHAM's
  374 per aircraft at 2.3 s. The network hears CDG surface transmissions briefly and
  loses them, which is a line-of-sight problem at the airport.

**Repair attempted: the MLAT instance.** adsb.lol publishes three archives a day
(`planes-readsb-prod-0`, `-staging-0`, `-mlatonly-0`). The MLAT one positions aircraft
that do not broadcast ADS-B by timing across feeders and is only 73 MB a day against
2.39 GB for the main archive, so adding it would have cost about 3 % more. Probed on
2026-01-02 (`code/` scratch probe, 27,324 traces): it carries **10 traces through the
LFPG box with 13 in-box points, none below 500 ft, and zero traces through the LTFM box
at all**. Not one MLAT point anywhere in the ten boxes is flagged on-ground. MLAT needs
four feeders to hear an aircraft, so it fails hardest exactly where ADS-B reception is
already thin. **The repair does not work and the instance is not worth its 3 %.** What
would revive these two airports is a denser network (OpenSky's historical database, which
`exp-042` §2.3 records as the best technical fit and blocked on the licence clause, not
on data), not a different filter on this one.

### Each airport's coverage is its own time series, and it goes both ways

Read the receipt table down a column rather than across a row. Eight months in, **no
airport holds one level, EHAM included**:

- **LEMD**: 0.0, 0.0 in January and February 2025 — never seen at all — then 74.4 in
  March. **Madrid switches on on 2025-03-04**: 0 % on March 1, 2 and 3, then 34, 41, 42,
  36, 24, 30, 47 % and **98 % from March 11**, holding 92 to 99 % for the rest of the
  month. That is an arrival with a date, not a trend. It does not hold either: 95.7,
  97.9 and 94.0 through the spring, then **28.8 and 27.1 across July and August 2025** —
  two whole months at a quarter — then 87.8, 93.8, 95.0, 96.5 and 96.3 to January 2026,
  and 60.7 in the scored July. Madrid is high, then low for a summer, then high, then
  middling.
- **LIRF has two dated days and they are eleven weeks apart in the wrong direction.**
  0.0 through April 2025, then 20.5, 69.2, 90.6, 82.7, 18.4, 38.2, 40.7, 62.4, 66.0,
  and **0.1** in July 2026. **Rome switches on on 2025-05-24** — 0 % on every day from
  January to May 23, then 16, 83, 97, 97 — holds 84 to 97 % through June, July and
  August to the 27th, and **falls on 2025-08-28**, from 93 % to 31, 25, 50, 28 % in four
  days. It never recovers that level: 5 to 36 % through September, a slow climb back to
  the fifties by December, and dark in the scored July. One airport, an arrival and a
  collapse, both inside training months, and neither visible in a month-level figure. **Rome switches on on 2025-05-24**, the same shape as Madrid eleven weeks
  earlier: 0 % on every day from January to May 23, then 16, 83, 97, 97, 76, 96, 97 and
  73 %. It never returns to that level after 2025, and the leaving falls inside a scored
  month. Two of the ten airports therefore begin inside the training year on a dated
  day, which is why a month-level coverage figure for either is wrong on both sides of
  it.
- **EDDM**: 97.1, 90.3, 90.4, 84.0, 93.3, then 64.7 in June and **36.9** in July 2025,
  then 82.1, 86.1, 93.8 and 97 % onward. The July figure is 0 % on every one of Jul 1 to
  18 (three days reach 9 to 10 %) and 94 to 99 % from Jul 19 — an outage that starts
  inside June and recovers inside July, not an arrival. (An earlier draft read it
  as a receiver coming online at Munich; January and February 2025 refute that, because
  Munich was already at 90 to 97 % five months before.)
- **EGLL is a year-long climb out of a hole, and it never once goes backwards.** 57.1,
  59.4, 37.0, **8.7**, 8.7, 12.0, 15.1, 20.9, 22.1, 27.8, —, 57.0, 76.6, **97.2**.
  Heathrow falls through March 2025 (62, 50, 51 … 12, 3, 22 …), bottoms at 8.7 % in
  April and May, and then rises in **every single month measured** to 97.2 % in the
  scored July. It is not one outage with an edge; it is a year of bad reception mending
  itself, and no month in it stands for its neighbour. This file predicted, before the
  month that tests it had landed, that 2025-11 would fall between October's 27.8 % and
  December's 57.0 % if the climb was real. **It reads 52.5 %**, and the series rises in
  all eleven months from the April floor with no exception.
- **LFPG** barely works and never before June 2025: 0.0 for five months, then 1.5, 4.8,
  4.0, 5.2, 8.6, **21.0** in November, 9.2, 8.8, 7.3. Its best month reaches a fifth of
  its departures. **LTFM is 0.0 in all fourteen months** — 47,581 ranking departures, and
  not one of the 427 days puts a single Istanbul ground callsign inside the box.
- **EHAM** is the steadiest and still not steady: 98 % in the first three months, then a
  **16-day dip to 54 % (Apr 5 to 21, month 82.4)**, 96 % in July 2025, and a single day
  at 70 % on 2026-01-05. It is also the only airport the pilot measured, which is why
  that week looked so clean.

So there is no single "the network grew" story to condition on, and no per-airport
constant either. What the modelling has to carry is a **per-airport, per-day coverage
series** — `analysis/receipt.json`, `by_airport_day` — because the surface features are
missing for different flights in the training months than in the scored ones, and the
difference is structured by airport and date rather than random. An absent observation
in July 2025 is a fact about adsb.lol that month, not about the flight.

### The coverage a model trains on is not the coverage it is scored on

The fourteen months make one comparison that no partial harvest could: what each
airport's coverage does between the training year and the two months that are scored.

<!-- table:shift -->
| Airport | 2025 (mean of months) | scored 2026 | shift |
| --- | ---: | ---: | ---: |
| EGLL | 31.5 % | 87.3 % | **+55.7** |
| EDDF | 87.0 % | 50.0 % | **-37.0** |
| EDDM | 84.4 % | 97.1 % | **+12.6** |
| LEMD | 65.9 % | 76.9 % | **+11.0** |
| LIRF | 35.2 % | 27.2 % | **-8.0** |
| LFPG | 4.5 % | 8.0 % | **+3.4** |
| LSZH | 91.8 % | 93.8 % | **+2.0** |
| LEBL | 95.2 % | 94.1 % | **-1.1** |
| EHAM | 95.6 % | 95.4 % | **-0.1** |
| LTFM | 0.0 % | 0.0 % | **+0.0** |
| **all ten** | **55.5 %** | **59.0 %** | **+3.5** |
<!-- /table:shift -->

Two airports move more than a third of their departures, in opposite directions.
**EGLL is 31.5 % across 2025 and 87.3 % in the scored months** — Heathrow is close to
unusable where a model learns and close to complete where it is judged. **EDDF is 87.0 %
across 2025 and 50.0 % in the scored months**, the same size of move the other way. EDDM
(+12.6) and LEMD (+11.0) rise, LIRF (−8.0) falls, and the ten together move only +3.5,
which is how a real problem hides inside an aggregate: the total is nearly stable because
the two largest movements cancel.

So any quantity learned from 2025 about *how often a surface observation exists* — a
per-airport coverage rate, a missingness indicator's base rate, a weight on the surface
arm — is wrong at Heathrow and Frankfurt in 2026, and wrong by more than the whole
programme's average coverage. `analysis/receipt.json` carries the per-airport-per-day
series so the correction can be made per row rather than per airport.

### What one month cannot say

Receipt is a property of the feeder network, and January 2026 showed it is **not**
stable even within a month: EDDF moved 52 points in two days. Eight months are now in,
and they settle the question this paragraph used to leave open. An earlier draft read
the 2025 months as a network that "grew over that year"; that is wrong in both
directions — LEMD arrives on one dated day in March 2025 and is at 28.8 % four months
later, LIRF arrives and then leaves again, EDDM has a fortnight's outage in July 2025
and is fine either side of it, and EGLL spends a year low and recovers only in 2026.

The one number that holds is EHAM's, and it holds against the pilot rather than across
time: the pilot read 98.2 % for July 2025 under its same-day rule and 96.2 % under this
file's windowed one, and the repaired month rate is **96.2 %** — the same number to the
tenth, computed on 31 days rather than 7, on a different machine and by different code.
That agreement is what makes the repair above believable; the old, inflated figure for
the same month was 98.0 %.

The lesson the gate taught is the one to carry: **a receipt rate quoted over fewer days
than a month is not a property of the airport.** Two days put EGLL at 50.0 % when its
month is 76.6 %. Every receipt number in this file is now stated over a whole month, and
`analysis/receipt.json` keeps the per-airport-per-day series so an outage can be seen
rather than averaged away — which is exactly what the repaired definition restores, since
the aggregate one averaged outages away by construction.

## 1b. The take-off clock, and why it is not a box problem

The modelling worker found `obs_wheels_off` broken at LIRF and EDDF and 253 s early at
EGLL. It is not the boxes. Each box **is** the runway hull plus 3 km, so every runway end
at all ten airports sits at least **3,002 m inside** its own box; widening would cost a
re-fetch and fix nothing. The gap between the last on-ground point and the first airborne
point tracks reception density instead, over the two gate days:

| Airport | departure segments | median gap to take-off | 90th pct | over 60 s | ground points per segment | median interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EHAM | 1,363 | 1.5 s | 8.8 s | 0 % | 85 | 2.7 s |
| LSZH | 684 | 3.3 s | 118.5 s | 18 % | 92 | 3.2 s |
| LEMD | 1,185 | 4.8 s | 15.9 s | 1 % | 96 | 3.1 s |
| EDDM | 764 | 6.0 s | 36.4 s | 5 % | 100 | 3.8 s |
| LEBL | 955 | 23.4 s | 49.3 s | 8 % | 80 | 3.1 s |
| LIRF | 757 | 34.2 s | 3,298 s | 34 % | 15 | 3.4 s |
| EDDF | 907 | 84.0 s | 1,393 s | 60 % | 17 | 4.6 s |
| EGLL | 1,088 | 522.8 s | 7,129 s | 95 % | 12 | 8.3 s |
| LFPG | 835 | 1,231.4 s | 12,295 s | 84 % | 3 | 7.6 s |

The ordering is the ordering of points per segment, not of geometry: where the network
holds an aircraft for 80 to 100 points it holds it to rotation, and where it catches 3 to
17 points it loses it in the departure queue. LIRF and EDDF have dense intervals (3.4 s
and 4.6 s) but short spans, so their reception is bursty rather than sparse.

**This costs the programme nothing**, because the ranking file already supplies the actual
take-off time `MVT_TIME_UTC_mvt` on every row. The quantity this stage exists to observe
is `obs_offblock`, the stand-departure clock, and that survives: it is present on 88 % to
98 % of departure segments at EDDM, LSZH, LEBL, LEMD and EDDF, 79 % at EHAM, 73 % at EGLL,
71 % at LIRF, and 44 % at LFPG.

So that a reader can tell the two cases apart per row rather than per airport, the
summaries now carry `wheels_lat`, `wheels_lon`, `wheels_gs_kt` (where the aircraft was
when the network last saw it on the ground) and `air_lat`, `air_lon`, `air_alt_ft`.

## 1c. One rule across fourteen months

The pilot's reviewer repaired the rule on 2026-09-08, after the gate and four months had
been fetched: a first-airborne point carrying readsb's **null** altitude is not evidence of
a take-off. The pilot priced it at 6 of 5,395 departure segments and 0.07 s of held-out
RMSE (`exp-043-pilot/analysis/divergence.json`). The harvest ships the repair, and the
condition the desk set is that one rule must hold everywhere:

- `manifest.json` names it. `summaries_rule` per day, `summaries_rules` per month, a
  top-level `summaries_rule` and `summaries_rule_uniform`, and a `rules` block describing
  each version in words.
- The points are untouched by the change (the repair reads `alt_ft`, which every points
  file carries), so the months already fetched are **re-derived from their stored Kaggle
  output, not re-fetched** — `kernels/rederive/`, `code/rederive_local.py`,
  `code/backfill.sh`. No month is reported complete until its summaries carry the current
  rule.
- Both assemblers hash the rule from the same source bytes, so the fetch kernel and the
  re-derive kernel provably inline the same one.
- `summaries` also gained `airborne_alt_ok`, so either rule can be read off a row without
  re-deriving anything again.

**"One rule" needed a second repair, and this one was mine.** Checking that the laptop and
a Kaggle session compute the same summaries (`code/check_machine.py`, run against the rule
block taken out of the gate kernel that actually ran) found that they do not, quite: 14 of
8,301 segments on 2026-01-01 and 17 of 9,270 on 2026-01-02 landed differently. The cause is
in the points, not the machines. **92 of 2026-01-01's 553,178 points share an
`(airport, icao, t)` with another point, and 22 of those pairs disagree about
`on_ground`.** `sort_values("t")` uses pandas' default sort, which is not stable, so the
two are ordered either way round and the ground run splits at a different point. Every
single differing segment — 31 of 31 across the two days — is on an aircraft with such a
tie, so the two machines agree exactly everywhere the input orders itself.

`all_segments` now sorts with `kind="stable"`, which keeps the order readsb wrote the
trace in: that is the truth about which of two same-second points came first, and it is
the same answer on every machine and every pandas version. The rule version says so —
**`repair-2026-09-08+stable-order`** — because `surfacefit` reads that string and the
numbers do move: the gate's two days go from 8,538 departures under the shipped rule to
8,534.

The pilot's `segments()` sorted unstably when this was found, so between 12:00 and 14:00
UTC `code/check_rule.py` set the tied aircraft aside and asserted the identity on the
rest. exp-043's `09998b5` adopted the same ordering and the exemption is gone: **all 866
of the pilot's departure rows on 2025-07-07 and all 760 on 2025-07-08 are identical on
all 14 columns**, tied aircraft included. `rule_check.json` still counts the ties (15 and
7 aircraft), because that count is the size of what the two sorts used to disagree about.
`version()` now reads `STABLE` out of the pilot's file as well as the sort call out of
this one, so `+stable-order` cannot be claimed while either half is unstable. The rule
hash moved to `d4c59868…` with the pilot's edit; the gate was re-derived under it and its
numbers did not change, which is the check that the pilot's sort and this one agree.

## 2. The boxes, checked

No box clips its aprons. Distances from each departure stand to the nearest edge of its
box, over the two gate days:

<!-- table:edges -->
| Airport | departure stands | 1st pct m to the nearest edge | median m | stands within 500 m |
| --- | ---: | ---: | ---: | ---: |
| EHAM | 1,086 | 2850 | 4976 | 0.00 % |
| LFPG | 397 | 3072 | 3768 | 0.00 % |
| EGLL | 851 | 2849 | 3304 | 0.00 % |
| LEMD | 1,162 | 1900 | 3507 | 0.00 % |
| EDDF | 797 | 2534 | 3385 | 0.00 % |
| LEBL | 928 | 2471 | 3658 | 0.00 % |
| LIRF | 546 | 3176 | 3882 | 0.00 % |
| EDDM | 683 | 2904 | 4157 | 0.00 % |
| LSZH | 672 | 2799 | 3979 | 0.00 % |
<!-- /table:edges -->

LTFM has no row because it produced no departure stand at all.

## 3. Cost

Ten boxes cost **less** than the pilot's one: 276 s and 351 s a day here, against 578 s
and 625 s for EHAM alone, because the parse moved to `orjson`, the per-point Python loop
moved to numpy behind a per-trace bounding-box test, and the archive's `heatmap` members
(0.6 to 0.7 GB a day) are now named and skipped before they are read. At that rate the
whole programme is about 40 kernel hours.

<!-- table:cost -->
| Month | days | kernel hours | GB consumed | archive digests verified | missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-07 | 31/31 | 3.5 | 109 | 62/62 | 0 |
| 2026-07 | 31/31 | 3.6 | 122 | 76/76 | 0 |
| 2026-01 | 31/31 | 2.9 | 92 | 62/62 | 0 |
| 2025-12 | 31/31 | 3.1 | 96 | 61/61 | 0 |
| 2025-01 | 31/31 | 1.8 | 66 | 0/57 | 0 |
| 2025-02 | 28/28 | 2.4 | 65 | 0/56 | 0 |
| 2025-03 | 31/31 | 2.1 | 78 | 0/62 | 0 |
| 2025-04 | 30/30 | 2.2 | 82 | 0/60 | 0 |
| 2025-05 | 31/31 | 2.7 | 97 | 4/62 | 0 |
| 2025-06 | 30/30 | 3.6 | 98 | 59/59 | 0 |
| 2025-08 | 31/31 | 3.8 | 109 | 62/62 | 0 |
| 2025-09 | 30/30 | 3.5 | 105 | 60/60 | 0 |
| 2025-10 | 31/31 | 2.9 | 107 | 62/62 | 0 |
| 2025-11 | 30/30 | 3.8 | 96 | 60/60 | 0 |
| **total** | **427/427** | **42.2** | **1320** | **568/861** | **0** |
<!-- /table:cost -->

### What the volume outage cost

The Mac's volume ran at 100 % for some hours on 2026-09-08 from work outside this
experiment. Nothing written was truncated — `code/verify_disk.py` re-read all 558
summary files, 541,925 rows, 135 analysis records and 35 sources against the bytes,
hashes and row counts the manifest had recorded, and found none wrong. The cost was
paid in Kaggle sessions instead. At 14:31 UTC the Kaggle CLI lost its credentials for
ten minutes; `status()` answered `UNKNOWN` for four live sessions, `UNKNOWN` is not a
live state, so their days left the busy set and the dispatcher pushed each month a
second time. Kaggle has no way to cancel a running session, so all four ran to the end.

<!-- table:wasted -->
| Duplicate session | days | pushed | kernel hours | fetched days already held by |
| --- | ---: | ---: | ---: | --- |
| 2025-07-b | 31 | 14:41 UTC | 3.5 | 2025-07 |
| 2026-07-b | 31 | 15:01 UTC | 3.6 | 2026-07 |
| 2025-01-b | 31 | 15:37 UTC | 1.8 | 2025-01 |
| 2025-02-b | 28 | 16:02 UTC | 2.4 | 2025-02 |
| 2025-05-b | 4 | 21:08 UTC | 0.4 (est.) | 2025-05 |
| 2025-05-c | 4 | 21:59 UTC | 0.5 | 2025-05 |
| 2025-06-b | 14 | 22:30 UTC | 1.5 | 2025-06 |
| **total** | **143** | | **13.8** | |

Each duplicate's hours are its own recorded run seconds: it landed over its original, so the day records the manifest carries are the duplicate's.  1 row(s) are estimated at the measured 355 s a day over ten boxes, because the manifest no longer names that session. Nothing is lost: each duplicate delivers the same days its original delivered, and the remaining months queue behind them.
<!-- /table:wasted -->

All four ran to the end and all four landed over their originals without changing
anything: 211, 278, 196 and 269 airport-days, compared file by file against the
manifest committed before each landing, **none differing on bytes, SHA-256 or row
count**. Two of those months (2025-07 and 2025-01) had been written by the local
re-derive and were rewritten from Kaggle, so the accident also produced the
cross-machine check at month scale that `code/check_machine.py` could only make on two
days. The hours were still wasted; the days were not.

### A fortnight the publisher tagged differently

`2025-05` delivered 27 of its 31 days on the first session. The four that failed are not
a broken stream and not a missing day: for **2025-05-28 to 2025-06-10** adsb.lol tagged
the release `v<date>-planes-readsb-prod-0tmp` and never made the plain
`...-prod-0` tag, so the name this stage builds 404s on all fourteen. The `tmp` release
holds the same archive — 2025-05-28 is two parts and 3.17 GB, the shape of 2025-05-27
beside it — so the days are published under another name rather than absent.

`assets_for()` now tries the plain tag and then the `tmp` one, and treats a 404 as an
answer instead of spending eight retries and the API rate limit on it. That second half
matters as much as the first: **`2025-06` lost fourteen days where only ten are tmp-tagged.**
Its log holds 60 genuine 404s and **174 rate-limit 403s** — the ten tmp days spent the
hourly budget GitHub allows an unauthenticated address, and 2025-06-11 to 06-14, which
have ordinary tags, then failed on the rate limit rather than on anything about those
days. A 403 is now waited out for 90 s a try rather than retried in three, because a
rate limit is a clock and not a transient. The tag actually used is recorded per
day in the manifest, so a reader can see which of the two names each day came from. The
retry budget for those fourteen days was reset when the fix went in, because the
assignment's "re-queued once, then listed missing" is a budget for a stream that breaks,
not for a defect in this stage that has since been repaired; the reset is logged in
`logs/dispatch.log`.

### What the publisher shipped, and what it does not explain

`code/published_bytes.py` records the bytes adsb.lol published for each of the 427 days,
so that a dark airport can be told from a short archive.

<!-- table:published -->
| Month | days published | median GB | short days | receipt |
| --- | ---: | ---: | ---: | ---: |
| 2025-01 | 31 | 2.12 | – | 48.0 % |
| 2025-02 | 28 | 2.39 | – | 49.9 % |
| 2025-03 | 31 | 2.54 | – | 55.4 % |
| 2025-04 | 30 | 2.76 | – | 51.9 % |
| 2025-05 | 31 | 3.14 | – | 56.9 % |
| 2025-06 | 30 | 3.34 | 2025-06-11 | 56.3 % |
| 2025-07 | 31 | 3.55 | – | 49.6 % |
| 2025-08 | 31 | 3.53 | – | 53.2 % |
| 2025-09 | 30 | 3.50 | – | 55.5 % |
| 2025-10 | 31 | 3.44 | – | 58.8 % |
| 2025-11 | 30 | 3.23 | – | 64.7 % |
| 2025-12 | 31 | 3.20 | – | 65.2 % |
| 2026-01 | 31 | 3.03 | – | 62.7 % |
| 2026-07 | 31 | 3.99 | – | 56.1 % |

All 427 days of the programme are published: none is absent from the archive, and 14 carry only the `...-prod-0tmp` tag (2025-05-28 to 2025-06-10). So a day this stage cannot deliver is a fetch failure, never a day adsb.lol did not ship.
<!-- /table:published -->

The archive grows by two thirds across 2025 — 2.12 GB a day in January, 3.55 in July —
and the receipt rate at these ten airports does not follow it: 48.0 % in January 2025
against 49.6 % in July. Whatever the network gained that year, it did not gain it over
Amsterdam, Heathrow, Frankfurt or Madrid, which is the same conclusion §1's per-airport
series reaches from the other side.

None of the dark windows this file reports is a short archive. LIRF's blank July 2026
sits on the **largest** files in the programme (3.99 GB median, minimum 3.25); EDDM's
dark Jul 1 to 18 2025 is 3.54; EDDF's collapse on Jan 29 to 31 2026 is 3.29. The one
genuinely short day is **2025-06-11 at 0.73 GB** against 3.34 for its month, and it is
short in the archive rather than in the boxes.

That was a prediction when it was written, and the month has since landed on it:
**2025-06-11 reads 5 % receipt where every other day of June reads 53 to 63 %**, on 781
segment rows against 8,350 the day before. It is the largest single-day fall anywhere in
the programme, and it is not about the feeders. Without the published size beside it the
day looks exactly like a network-wide outage, so the manifest carries `published_bytes`
and `published_short` on every day record: a consumer can tell the two apart without
leaving the manifest, and 2025-06-11 is the only day in 427 that carries the flag.

### The days the publisher never signed

adsb.lol's releases carry a SHA-256 on each asset from about mid-2025. Before that they
carry none: the GitHub API returns `digest: null` for every January and February 2025
asset, which is why §3 reports **0 of 57** digests verified for 2025-01. That is an
absent check, not a failed one, and the assignment's "a day whose digest fails" does not
describe it — the streams completed, the tars parsed, and the row counts are ordinary,
so those days are kept rather than listed missing.

What the publisher does offer for every asset is a byte length.
`code/check_lengths.py` compares it against the bytes the kernel consumed, for exactly
the months that carry no signature: **113 unsigned assets over 59 days, none
disagreeing**. It distinguishes "GitHub could not be asked" from "the lengths differ",
so an outage cannot look like a corruption.

Two repairs came out of it, both in `code/dispatch.py`: `status()` now returns the
previous state when Kaggle does not answer and never invents a third value, and a
session whose own rule already matches the current one drops the re-derive records its
pull replaced, so a record cannot outlive the bytes it describes.

Every byte consumed so far matched the SHA-256 the publisher signed (4 of 4 assets).

## 4. Missing days

<!-- table:missing -->
**None. All 427 days of the programme delivered a run record**, so no day is
listed missing with a reason and the missing table is the empty one this stage hoped for.
<!-- /table:missing -->

## 5. What the stage does not measure

Accuracy. The pilot's held-out 144.0 s against NM's 309 s is an EHAM number from one
week. Nothing here pairs any other airport's summaries with a block time; that is the
modelling worker's first job and its Sol round, not this one's.
