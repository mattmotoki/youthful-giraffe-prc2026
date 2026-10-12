# exp-037-tail — SPEC

Worker `tail`, tier hard. Assignment `assignments/tail.md`. Mac laptop, 2 threads
until the desk says `mask` has closed, then 6. No GPU. No upload (score-feedback rule).

## What round 1 of the review refuted in this reading

This SPEC is the reading written before the work. Two of its statements did not
survive; `RESULTS.md` §9 carries the evidence.

- **The reading is not new.** The chain defined below is `gap_arr1`/`gap_arr2` in
  `prc/tail_features.py::_stand_signals`, at maximum absolute difference 0 s, and
  those columns are already in `NOAC_FEATURES_V2` and `_EXP002_SIGNALS`. The
  first thing this SPEC should have specified is a check of the chain against the
  shipped feature module. It did not, and step 1 built an instrument the head
  already had (`RESULTS.md` §1).
- **Three measurements this SPEC promised did not run, and the substitutions
  were never written down until round 2 of the review found them.** (a) The
  margins are fitted **per airport** (`code/step2.py`), not per gap decile: the
  purity spread across airports is 3.92 % to 26.32 % (`RESULTS.md` §2) and no
  comparable spread across gap deciles was found, so the airport is the cut that
  carries the variation. (b) The third variant built is `prev2`, the arrival
  BEFORE the last one, not `fresh` (`strict` inside `g ≤ 24 h`). `prev2` is the
  near-hard bound the whole experiment turns on and `fresh` is not measured at
  all; the count of `fresh` arms in the record is **0**. (c) No table is reported
  twice, once over all chained rows and once inside `g ≤ 24 h`. These are changes
  of plan made while the work ran; they are recorded here rather than defended.

- **Rejected reading 1 rejected too much.** `FLIGHT_mvt` identifies the flight and
  not the airframe, which is why it cannot be the chain — that part stands. But
  its three-letter prefix is a usable *airline* proxy, and splitting the stand's
  arrivals by it is what decides the mechanism: the arrivals around a placeholder
  row's block time carry the departing flight's own airline at four to five times
  the stand's base rate (`RESULTS.md` §3b). Rejecting the column as a chain key
  was right; not trying it as a feature was not. It is still not an airframe id,
  so it supports a same-AIRLINE reading and not a proof of the same aircraft.

## My reading of "the previous arrival at the same stand", and what I rejected

**The key is `(airport, stand)`**, where the airport of a departure is `ADEP_mvt`
and of an arrival is `ADES_mvt`, and the stand is the raw `STAND_mvt` string with
no normalisation. **The time of an arrival is its in-block time
`BLOCK_TIME_UTC_mvt`** — the moment the aircraft starts to occupy the stand — and
arrivals are never blanked, in the ranking file or in a masked validation month.
**The chain for a departure with take-off `T` is the arrival at the same
`(airport, stand)` with the largest in-block time `B_in ≤ T`.** The quantity I
carry per departure is the **gap `g = T − B_in`**, and, where a label exists,
the **turn `u = g − y = B_off − B_in`**, which is the stand's occupancy time for
that aircraft if the chain is the same aircraft. `u` is what the bound is fitted
on and what its purity is read from: a pure chain cannot produce `u < 0`, because
two aircraft cannot occupy one stand at once.

**No time window is imposed at the join, and a window is applied afterwards as a
band, not as a filter.** An old previous arrival gives a large `g` and therefore a
weak bound, which is harmless; dropping it would only lose rows. Every table
below reports the result both over all chained rows and inside `g ≤ 24 h`, and
the head fits its margin per gap decile, so a weak chain cannot borrow a tight
margin from a strong one. **[Neither happened. No table is reported twice, and
`code/step2.py` fits the margin per AIRPORT. See the amendment above.]**

**Three variants of the chain are built and the purity of all three is measured,
because the failure mode is known and is not the same in each.** (a) `any`: the
last arrival before `T`, as above. (b) `strict`: the same, and additionally the
last **movement** at the stand before `T` — an arrival at its in-block time, a
departure at its NM off-block `AOBT_3_flt` — must be that arrival, so a departure
that follows another departure with no arrival between them is unchained. (c)
`fresh`: `strict` with `g ≤ 24 h`. **[The third variant built is `prev2`, the
arrival before the last one, not `fresh`. The count of measured `fresh` variants
is 0. See the amendment above.]**

**What I rejected, and why.**

1. **Registration or callsign as the chain.** There is no registration column, and
   `CALLSIGN_flt`/`FLIGHT_mvt` identify the *flight*, not the airframe; on the
   noac rows the NM join is exactly what is thin. The stand is populated on 100 %
   of arrivals and departures (exp-028 §10), which is why it is the key.
2. **Ordering the stand's timeline by the departure's take-off time.** A departure
   stops occupying the stand at its off-block, not at its take-off, and on the
   rows this experiment is about those two times are hours apart. Using take-off
   would put our own row's occupancy end hours late and would systematically
   break the chain for exactly the population under study. The ranking file blanks
   `BLOCK_TIME_UTC_mvt` for departures, so the only legal off-block proxy for a
   *neighbouring* departure is `AOBT_3_flt`, and variant (b) is the only place it
   is used.
3. **Treating the bound as exact.** It is not. If the true taxi is genuinely long,
   the stand is free from `B_off` onward and another aircraft can arrive before
   `T`; the chain then picks that arrival and the bound is violated by a *true*
   row. This is the one way a physical row can look impossible, it is not
   removable by any join, and it is why step 1 measures the violation rate on
   labelled rows per band and per gap decile instead of assuming it is zero. The
   violation rate is the instrument's false-positive rate and it enters the head
   as such.
4. **Towing.** A towed aircraft moves between stands with no movement record, so
   the departure stand can differ from the arrival stand and the chain then points
   at a different aircraft. Nothing in the file observes a tow. Its effect is
   inside the measured violation rate, and it is one reason the margin is fitted
   from the empirical `u` distribution rather than set from an operational
   turnaround figure.

## The gates I am held to

A candidate only if the out-of-fold gain on 2025 noac rows is positive with a
**day-clustered** interval clear of zero **and** the shift on the 388 against v9
is under **500 s RMS** (`docs/QUEUE-2026-09-07.md` §4). Otherwise a reading.
v9 = `submissions/candidates/exp-027-k2.parquet` (sha `a3a139db…`); its 388 noac
values are v7's, which are stack-5's.
