# exp-042-surface — SPEC

Assignment: `assignments/surface.md` (Q16, research only, tier medium, reviewer none).

## My reading of the assignment

Q16 asks a **gate question**, not a modelling question: does any source of contemporary
surface observations exist that (a) the competition rules permit, (b) covers the twelve
2025 months and January/July 2026, (c) contains **on-ground** positions at the ten
departure airports, (d) can be joined to `ranking.parquet` rows without a registration
column, and (e) is stable and affordable. The deliverable is a source table, the rules
quotes, and a matched-sample plan. No data is fetched and nothing is ingested.

I treat "coverage" as three separate things that a source can pass or fail independently:
**calendar coverage** (are the days present at all), **surface coverage** (are on-ground
positions in the product, and at what sampling), and **identity coverage** (can a row of
`ranking.parquet` be tied to one aircraft track). A source passes only if all three pass.
I refuse to infer surface coverage from airborne coverage, per the assignment.

Readings I rejected:

- *"Answer the rules question by quoting `docs/external-data-rules.md` alone."* Rejected:
  the assignment says to quote the clause with its URL. I re-fetched the eligibility and
  data pages (they now redirect to `prc-data-challenge-2026.netlify.app`) and quote the
  live wording. The repo's reading survived the re-fetch unchanged.
- *"Measure the surface receipt rate now to settle the coverage check."* Rejected: that is
  data acquisition, which the queue does not authorise. The receipt rate is named as the
  one open item and priced as a pilot instead.
- *"Treat the 286 GB 2022 archive as a Q16 candidate."* Rejected: the queue separates it
  explicitly. I recheck its location and coverage (step 2 asks for it) and place it in the
  table as a **prior** source, not a contemporary observation.
- *"Score sources by how good the mechanism could be."* Rejected: the check is a
  gate. A source that fails the licence clause or the calendar does not get a mechanism
  discussion; it gets the precise gap recorded, per the assignment's failure-and-repair line.

## What I measured locally (no external fetch)

All from `data/raw/`, one thread, seconds of runtime. Numbers in `REPORT.md` and
`analysis/identity.json`.

## Deliverable

`experiments/exp-042-surface/REPORT.md`.
