# exp-043-pilot — the surface-observation pilot: EHAM, one training week

Worker `pilot`, tier hard, 2026-09-08. Assignment: `assignments/pilot.md` (recorded rule, Q16 step 2).
Reviewer: `gpt-5.6-sol` at `xhigh`, one round on the three numbers before they enter the ledger.
It took six: four Sol rounds, then rounds 5 and 6 on the `MODES.md` FALLBACK reviewer
(`claude-fable-5-1`, effort high) when Sol was not available. Round 6
CLEARED the record; `RESULTS.md` §8 is what each round changed.

## My reading of the assignment

The assignment asks one question and stops at the first number that answers it: **does the
adsb.lol network actually receive aircraft on the ground at EHAM, and if it does, is the
observed start of motion a better reconstruction of the blanked `BLOCK_TIME_UTC_mvt` than
NM's `AOBT_3_flt` plus an airport median (355 s at EHAM, `exp-042/analysis/identity.json`)?**
Everything else in the file — the box, the ±15 min identity window, the 3 kt / 30 s motion
rule, the paired `A`, `B`, `2A − B` on the fold-side v10 residual — is the machinery that
makes that question answerable, and the 40 % receipt gate after two days is permission to
stop early. I read the deliverable as three ledger-grade numbers plus an honest statement of
what one airport-week cannot say, not as a model. No model is fitted here.

**Readings rejected.**

- *That the pilot should fetch all seven days before measuring anything.* The assignment puts
  the census first and names a stop after two days; fetching seven days first would spend
  24 GB to learn something two days can teach. Days 3–7 are queued only if days 1–2 pass.
- *That `obs_wheels_off` needs its own accuracy claim.* It is a **check**, not a product: if
  the last on-ground point is not within about a minute of `MVT_TIME_UTC_mvt`, the identity
  match or the clocks are wrong and the off-block number must not be read. I treat a failed
  clock check as a reason to repair the matcher, never as a result.
- *That the no-NM rows can be dropped when the callsign cannot be reconstructed.* They are
  about 60 % of the naive predictor's squared error (`COMMON.md`) and the assignment asks for
  the three numbers on that tail alone. Unmatched no-NM rows are reported as unmatched, with
  their count, not silently removed from a denominator.

## The forced change of route, and why

The assignment budgets "never hold more than 12 GB of transient files" on the laptop. When I
started, `df -h /System/Volumes/Data` showed **4.5 GB free of 460 GB (99 % used)** — not from
this repo (which holds 9 GB) but from other work on the machine. One daily archive is 3.4 GB
compressed and expands to roughly five times that. The desk confirmed the ceiling and moved
the fetch to **Kaggle CPU kernels** (internet on, about 20 GB of scratch each), pulling only
the filtered slices back to the laptop. I took that route with one change of my own:

**The archive is never written to disk at all, on either machine.** The two split release
assets (`*.tar.aa`, `*.tar.ab`) are read over HTTP as one continuous byte stream and fed
straight into `tarfile` in stream mode (`r|`); each member is decompressed, screened and
dropped inside the loop. Peak disk is the output slice (a few MB); peak memory is one trace
file. The stream is resumable: a broken connection is re-opened with a `Range` request from
the last byte delivered, so a failure costs the remainder of one asset, not the day.

**Consequence for provenance, stated because it is a change.** The assignment says to record
each asset's byte count and SHA-256 "before extraction". With no disk to hold a copy, the
hash is computed **on the same bytes that are extracted**, in stream order, and compared to
the SHA-256 GitHub publishes in the release metadata (recorded in `analysis/provenance.json`
before any byte was fetched). This certifies exactly the bytes the filter consumed, which is
a stronger statement than hashing a separate on-disk copy, and it is the only one available
under the 4.5 GB ceiling. Both digests appear in every `analysis/run_<day>.json`.

## The screen, and why it has no false negatives

A day is about 3.4 GB compressed and roughly 15–20 GB of trace JSON; parsing all of it is the
only real cost. Before `json.loads`, each decompressed trace is screened for the one-decimal
text prefixes of the box (`52.2`, `52.3`, `52.4` and `4.6`, `4.7`, `4.8`, generated from the
box constants, not typed). Every decimal rendering of a coordinate inside the box begins with
one of them, so an aircraft with a point in the box is never skipped; a false positive costs
one JSON parse. The census of aircraft seen inside the box is therefore complete, not a
sample.

Two further decisions inside the filter, both recorded because they are mine:

- **The callsign is carried forward and then backward over the whole trace before any point
  is dropped.** readsb writes the aircraft object only when it changes, so a ground segment
  can carry no callsign of its own while the same trace names it minutes later. Filtering
  first and filling second would lose exactly the stand-side points the pilot is about.
- **The first airborne point after a ground segment is kept even if it falls outside the
  box.** It is one point per segment, it is the wheels-off cross-check, and a climbing
  aircraft can leave a 17 × 13 km box between two 30 s trace points.

## What the pilot cannot say, written before it runs

One airport, one week, one network. It cannot say what the receipt rate is at LTFM, LEMD or
LIRF, where feeder density is lower and the case for a crowd-sourced network seeing a parked
aircraft is weakest; it cannot say that 2026 coverage matches 2025; and the 30 s default trace
interval sets a floor on the resolution of the off-block instant that no amount of matching
removes. The extrapolation in `RESULTS.md` §5 states the EHAM share of the ranking file
(38,182 of 344,841 rows) and prices the rest as unmeasured, never as assumed.

## Files

| Path | What it is |
| --- | --- |
| `kernels/surface/prc-surface.py` | the single source: stream, screen, filter, census |
| `kernels/surface-<name>/` | the generated kernel that actually ran (committed) |
| `code/push_kernel.sh`, `code/pull_kernel.sh` | generate/push one kernel; pull its log and slices |
| `analysis/provenance.json` | the seven releases, their assets, sizes and published SHA-256 |
| `analysis/run_<day>.json` | per-day provenance, digest check, member census, timings |
| `artifacts/exp-043/` (git-ignored) | `points_<day>.parquet`, `census_<day>.parquet` + manifest |
