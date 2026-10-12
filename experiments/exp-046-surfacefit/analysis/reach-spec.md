# Change spec: the time-placement pass for stale-inbound-identity segments

**Lane** `exp-046-surfacefit`. **Shared file** `code/block.py`, matcher
`6fdf39519eb0f500c06230e2972d3c8300aa6222ed8d9b003998ddf6e286093b`. **Counterparty**
`exp-047-queue`. **Handshake** as §14: this spec, queue's one line of agreement, **one**
commit of `block.py`, **one** `MATCHER_SHA` line from the desk to both lanes, both lanes
re-price **once**. Ordered by the desk 2026-09-09 after §19.

Nothing in this spec is served. It changes the matcher and therefore both lanes' inputs; the
serve that follows is a separate decision.

---

## 1. What is being repaired, in one paragraph

`analysis/identity-read.json` (§19) found one cause behind 92–97 % of the clocked ground
segments no ranking row claims at LFPG and EDDF: **99.4–99.9 % of them carry the callsign of
a ranking ARRIVAL into the same airport**. They are departure runs — they end airborne and
carry a take-off — whose transponder never switched off the inbound callsign. The parity
proof, on LFPG july's purely-numeric AFR suffixes: ranking departures 183 even against 6 odd,
matched segment forms 49 even against 2 odd, **unmatched segment forms 0 even against 142
odd**. No callsign join reaches them at any window, tolerance or tie-break — which is why the
`contended` bucket, the one a window or dedup change would attack, is **0.1–3.5 %**.

The repair places such a segment on a departure by **take-off time**, and takes uniqueness
inside the window as its control.

## 2. The change

A **third pass**, after the strict pass and the wide pass, in `block.build`. It runs only on
what both earlier passes left, and it is fail-closed at every step.

```
mode:  MODES["stale_inbound"] = 3          # a new mode, so every placed row is separable
gate:  the segment is a departure segment, unclaimed by passes 1 and 2, with
       obs_offblock notna and obs_wheels_off notna
   AND no broadcast form of it is a ranking DEPARTURE join key at this airport
   AND some broadcast form of it IS a ranking ARRIVAL callsign into this airport
   AND exactly ONE unclaimed ranking departure of this airport has
       |MVT_TIME - obs_wheels_off| <= 60 s
place: that departure, one-to-one; a segment with two candidates or none places nothing
```

**Why the identity condition is in the gate and not a diagnostic.** Restricting to the
arrival-callsign bucket means the pass can never touch a segment the callsign join could
claim, so **every strict and wide row stays bit-identical**, exactly as the wide pass is
already constrained. A segment with no identity at all — the `neither` bucket, 2 to 32 rows
per airport-month — is **not** placed: an anonymous segment placed by time alone is the
false-match case, and this pass does not take it.

**Why ±60 s and not wider.** It is `block.py`'s own `clock_ok` cut, so it is not a new
number. Widening destroys the control rather than helping: the exactly-one share collapses
from 36–48 % at ±60 s to 5–7 % at ±300 s and 1.5–4.3 % at ±900 s, because a busy airport puts
several departures into any wide window.

## 3. The gates, all four written before the build

**(a) Uniqueness.** Exactly one unclaimed ranking departure within ±60 s. Two candidates
places nothing; zero candidates places nothing. Reported per airport-month as
`placed / eligible / declined_ambiguous / declined_none`.

**(b) §12.3's per-airport false-match bound.** The pass is priced under the same
density-scaled bound the month already carries, `month_false_match_bound(density)` in
`code/test_block.py`, and the bound is re-asserted per airport with the pass on. A pass that
puts any airport over its own bound does not ship. Densities as recorded: LFPG 0.78, LIRF
1.62, EDDF 1.70, EGLL 1.85, LEBL 2.06, LEMD 2.24, LSZH 2.38, EHAM 3.24, EDDM 3.37.

**(c) The day-shift placebo.** The same placement with **the segments' times shifted by one
day** must place approximately nothing. Reported per airport, against
`PLACEBO_MAX_*_SHARE`. This is the gate that decides whether ±60 s uniqueness is identity or
coincidence: a real placement cannot survive a 24 h shift, and a coincidental one will.

**(d) The aircraft-consistency secondary read.** The ranking file carries no registration and
its operator is hashed, but it carries **`AIRCRAFT_TYPE_mvt`** on 99.9 % of rows. A stale
inbound identity means one airframe: the segment's arrival callsign names a ranking ARR row,
and the placed departure is the same airframe turning round, so their aircraft types should
agree. Reported per airport as `agree / disagree / unknown`. It is a **read, not a gate** —
a type mismatch may be a code-share or an equipment swap — but a disagreement rate far above
the airport's base rate would refute the mechanism, and that outcome stops the pass.

## 4. Expected class diff

Measured, not projected: `code/identityread.py --all` → `analysis/identity-read-all.json`,
which counts the placement the pass would make on both ranking months at all ten airports,
under the exact gate of §2. **9,875 rows enter this lane's class**, 4,147 in 2026-01 and
5,728 in 2026-07.

| month | airport | arrival-callsign segments | **placed** (exactly one within ±60 s) | share | declined (2+ candidates) |
|---|---|---|---|---|---|
| 2026-07 | EDDF | 6,875 | **3,049** | 44.4 % | 2,061 |
| 2026-07 | LFPG | 4,187 | **1,516** | 36.2 % | 1,336 |
| 2026-01 | EDDF | 3,027 | **1,218** | 40.2 % | 395 |
| 2026-01 | LFPG | 2,801 | **1,123** | 40.1 % | 698 |
| 2026-01 | EGLL | 2,162 | **1,044** | 48.3 % | 165 |
| 2026-07 | LEMD | 2,721 | **937** | 34.4 % | 328 |
| 2026-01 | LIRF | 1,301 | **603** | 46.4 % | 102 |
| 2026-01 | LEBL | 324 | **94** | 29.0 % | 6 |
| 2026-07 | LEBL | 268 | **93** | 34.7 % | 2 |
| 2026-07 | EGLL | 174 | **65** | 37.4 % | 3 |
| 2026-01 | LEMD | 118 | **39** | 33.1 % | 3 |
| 2026-07 | LSZH | 125 | **27** | 21.6 % | 0 |
| 2026-07 | EHAM | 148 | **25** | 16.9 % | 1 |
| 2026-01 | EHAM | 174 | **18** | 10.3 % | 1 |
| 2026-07 | EDDM | 124 | **10** | 8.1 % | 0 |
| 2026-07 | LIRF | 13 | **6** | 46.2 % | 2 |
| 2026-01 | LSZH | 66 | **5** | 7.6 % | 0 |
| 2026-01 | EDDM | 51 | **3** | 5.9 % | 0 |
| | **total** | **24,659** | **9,875** | | **5,103** |
LTFM places nothing in either month (§18.1: 32 segments across 16 days, none with a clock).
Every airport-month's `neither departure nor arrival` bucket — 2 to 32 segments — is refused
by the gate and is **not** in the placed column.

**Face, at §16.3's currency**: 9,875 rows, **+0.518 s at the level ×1.2 (the lower bound)**,
**+0.773 s at ×1.5**. This is the class-entry face and not a priced gain: the rows enter the
class and this lane's existing head then values them, which the fold table prices after the
re-price, not before it.

**LFPG is not un-held by this.** LFPG gains 2,639 rows across the two months, and every one
of them stays inside the airport this lane holds at the price base (§12.1). The hold is
re-read afterwards as §17 row 4 on the new evidence; **this spec does not lift it**, and
LFPG's contribution to the face above is therefore **not** available until that separate
decision is taken. Excluding LFPG the face is 7,236 rows, +0.380 / +0.567 s.

## 5. Interface columns that move

The handover is `artifacts/exp-047/block/<month>.parquet` keyed by `MVT_ID_mvt` (34 emitted
columns, `analysis/blockpy-rebuild/interface-columns.json`). For a **placed** row every
observation column goes from null to a value — `obs_icao`, `obs_taxi`, `obs_wheels_off`,
`obs_seg_taxi`, `obs_taxi_motion`, `dt_takeoff`, `time_to_250m`, `seen_before_move_s`, the
`seg_*`, `stand_*` and `arr_*` families, `turnaround_anchored`, `callsign_n`,
`obs_n_candidates`, `obs_matched`, `obs_has_clock`, `reach_class` — and:

- **`match_mode` gains the value 3** (`MODES["stale_inbound"]`), so a placed row is separable
  from a strict or wide row in the block itself and in every arm that reads it;
- **`matched_on_modal`** is meaningless for a placed row (no callsign matched) and is written
  **0.0**, which is what the wide pass already writes for a form that is not the modal;
- **`clock_ok`** is 1.0 by construction for every placed row, since the gate is `|dt| ≤ 60`;
- **`obs_ambiguous`** is 0.0 by construction, since the gate is uniqueness.

**No existing value changes.** Passes 1 and 2 are untouched and the pass runs only on
segments and departures they both left, so every currently-matched row is bit-identical. This
must be verified the way §14 verified the rebuild — a full 34-column sweep, not a five-column
one, which is the error §14.1 records.

## 6. Pins, manifests and the re-price

1. `MATCHER_SHA` changes. `code/arms.py` carries it and the block cache key already includes
   the matcher's first 8 hex (§14.4), so no stale block can be served into the re-price.
2. **Fold pins re-issued** — `code/population.py`, with its `superseded` block naming the
   previous keys, counts, `days_added` and `days_gone`. The pass changes which rows are in the
   fold blocks, so every fold pin is re-priced, not carried.
3. **Class vector re-emitted** on the new sha: `artifacts/exp-046/class/<month>-v18.parquet`
   plus `class-manifest-v18.json`, same columns, `occupied_by_exp_046` still the mask and
   `written_by_exp_046` still the bytes. **9,875 rows leave `exp-047-queue`'s class and enter
   this lane's**, which is the largest class movement since v13 and is why the re-emit is part
   of this spec rather than the serve that follows.
4. **Both lanes re-price once**, on the desk's single `MATCHER_SHA` line.

## 7. What queue is asked to check

The same three questions §14 asked, and one new one:

1. that no column this pass writes is read by queue's block in a way the new mode breaks —
   in particular that a `match_mode` of 3 is either used or ignored deliberately, not
   silently treated as a wide match;
2. that the 9,875 rows leaving queue's class are rows queue does **not** currently move, so
   no served byte is contested;
3. that queue's own census reproduces the per-airport-month table of §4;
4. **new**: whether queue's neighbourhood features change for rows it keeps, because a
   neighbour of a kept row may now be a placed row — this is the spillover §14.1 found the
   hard way, and it is asked for explicitly this time rather than discovered afterwards.

## 8. What would stop this shipping

Any one of: an airport over its §12.3 bound with the pass on; a day-shift placebo that places
materially above `PLACEBO_MAX_*_SHARE` at any airport; an aircraft-type disagreement rate far
above the airport's base rate, which would refute the stale-identity mechanism itself; or any
existing row changing value in the 34-column sweep.

**And one honest limit.** The uniqueness control makes a placement unambiguous *in time*; it
does not make it *correct*. The day-shift placebo is what separates those two, and it is the
gate this spec would fail on if the mechanism is wrong.
