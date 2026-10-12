# Change spec: the turnaround-chain pass for stale-inbound-identity segments

exp-046-surfacefit, 2026-09-09. Desk GO under the standing order (a per-row placebo at a few
percent with thousands placed → spec and handshake). Supersedes `analysis/reach-spec.md`,
whose pass was built, gated and **refuted** (RESULTS §21.2): a placebo that moved the
departures one day placed 8,084 rows against the real pass's 7,865. `code/block.py` was never
committed with it and its sha stays
`6fdf39519eb0f500c06230e2972d3c8300aa6222ed8d9b003998ddf6e286093b`.

The repair was measured label-free first (`code/turnaround.py` → `analysis/turnaround.json`,
RESULTS §21.3) and specified only after it passed both bars — the desk's per-row placebo bar
and queue's per-cell bar. It reaches **3,101 rows**, not 9,875, and it is a third the size
because it is the part that is real.

**`code/block.py` is not committed until the desk relays queue's GO.** `§` references are to
`experiments/exp-046-surfacefit/RESULTS.md` unless marked otherwise.

## 1. What is being repaired, in one paragraph

24,959 clocked ground segments across the ten airport-months carry a callsign that is no
ranking DEPARTURE key, and **99.4–99.9 % of them are a ranking ARRIVAL into the same
airport** (§19): a departure still broadcasting the inbound identity of the flight that
brought the airframe in. The refuted pass tried to close that gap with **time** — the one
unclaimed departure within ±60 s — and time cannot identify, because at EDDF july the
unclaimed departures run 0.27 a minute and `P(exactly one in ±60 s) ≈ 0.31`, which is exactly
the "signal" §19.3 reported. The repair replaces time with the **airframe**: the inbound
callsign already names the arrival, and the departure is the aircraft turning round at that
arrival's stand.

## 2. The change: the exact key

One new pass in `code/block.py`, after passes 1 and 2, on the segments they both left:

```
gate:  departure segment, unclaimed by passes 1 and 2, obs_offblock and obs_wheels_off notna
   AND no broadcast form is a ranking DEPARTURE key at this airport
   AND some form IS a ranking ARRIVAL callsign into this airport      -> that ARR row
   AND an unclaimed ranking DEPARTURE exists with
         STAND_mvt             == the ARRIVAL's STAND_mvt
         AIRCRAFT_OPERATOR_flt == the ARRIVAL's operator (hashed)
         AIRCRAFT_TYPE_mvt     == the ARRIVAL's aircraft type
         MVT_TIME (take-off)   >  the ARRIVAL's BLOCK_TIME_UTC_mvt (its in-block)
         MVT_TIME - in_block   <= MAX_TURNAROUND_S = 86400
   AND that departure is UNIQUE inside the chained set, and one-to-one the other way
   AND |MVT_TIME - obs_wheels_off| <= 60          (the take-off agreement, part of the key)
```

`MODES` gains `"turnaround": 3`. The difference from the refuted pass is the whole point:
**the key is the airframe's stand chain, and time is the last clause rather than the only
one.**

### 2.1 The leak statement

**No departure `BLOCK_TIME_UTC_mvt` is read, at any point, by any clause of this key.** A
departure's in-block is blanked by `mask_like_ranking` and is the quantity this competition
predicts. Every field the key reads is an untouched input:

| field | read from | presence | note |
|---|---|---|---|
| `STAND_mvt` | the ARRIVAL row, and the candidate DEPARTURE row | 100 % of both phases | 1,697 distinct values |
| `AIRCRAFT_OPERATOR_flt` | both | 100 % | hashed, 587 values |
| `AIRCRAFT_TYPE_mvt` | both | 99.9 % | 199 values |
| `BLOCK_TIME_UTC_mvt` | **the ARRIVAL row only** | 100 % of ARR rows | never masked; `mask_like_ranking` blanks it on **DEP** rows only |
| `MVT_TIME` | the candidate DEPARTURE row | 100 % | the take-off, an input |
| `obs_wheels_off`, `obs_offblock` | the ground segment | — | the observation, this lane's own instrument |

`code/turnaround.py` asserts this and records `no_departure_block_time_is_read: true` in its
JSON. The ARR side of the ranking file is 344,693 rows and `mask_like_ranking` touches
neither its in-block nor any callsign.

### 2.2 Why ±60 s is inside the key and not beside it

Ungated, the chain places 11,579 and only 26.8 % agree on take-off; the placebos are 0.76 %
and 2.04 %. So the chain finds real turnarounds, but three quarters of what it places is not
corroborated. Moving the check into the key costs **8,478 rows** and buys the placebo rates
of §3. That trade is this spec's central choice and it is deliberate.

**The 8,478 ungated rows are a MAYBE, not part of this spec.** Revival condition: **a second
independent check** — something that corroborates the chain without being the take-off clock,
so that the check kept in the key stays an independent test. Until one exists they do not
ship, and this spec makes no claim on them.

## 3. The placebo tables

**(a) The 34-column sweep, on rows the pass does not place.** Each month built twice, all 34
emitted columns compared on the complement of the placed set. **Zero values may change.**
`code/stalepass.py` already runs this sweep and the refuted pass passed it (§21.1); it is
repeated because the pass is new. It is a *34-column* sweep, not the five-column one §14.1
records as an error.

**(b) §12.3's per-airport false-match bound**, re-asserted per airport with the pass on,
under `month_false_match_bound(density)` in `code/test_block.py`. Densities as recorded:
LFPG 0.78, LIRF 1.62, EDDF 1.70, EGLL 1.85, LEBL 2.06, LEMD 2.24, LSZH 2.38, EHAM 3.24,
EDDM 3.37. Any airport over its own bound stops the pass.

**(c) The per-row placebos.** Two, each breaking the key in a different place:

| | placed | gated (chain ∧ ±60 s) | share of real |
|---|---|---|---|
| the chain | 11,579 | **3,101** | — |
| **day placebo** — departure take-offs +24 h | 4,218 | **32** | **1.03 %** |
| **stand placebo** — `STAND_mvt` permuted among the airport's departures | 1,664 | **34** | **1.10 %** |

The stand placebo keeps the chain's shape and makes only the stand wrong, so it tests the
stand chain itself rather than the calendar. Both return about 1 %.

**(d) Gate (c) per airport**, as queue asked, because queue's arm reads neighbours' clocks:

| month | airport | **gated** | day placebo | stand placebo |
|---|---|---|---|---|
| 2026-07 | EDDF | 1,264 | 13 | 19 |
| 2026-01 | EDDF | 466 | 5 | 5 |
| 2026-01 | EGLL | 306 | 1 | 3 |
| 2026-01 | LIRF | 260 | 0 | 1 |
| 2026-01 | LFPG | 231 | 5 | 2 |
| 2026-07 | LFPG | 210 | 5 | 2 |
| 2026-07 | LEMD | 162 | 3 | 1 |
| 2026-07 | EGLL | 45 | 0 | 1 |
| 2026-07 | LEBL | 42 | 0 | 0 |
| 2026-01 | LEBL | 39 | 0 | 0 |
| 2026-01 | LEMD | 34 | 0 | 0 |
| 2026-07 | LSZH | 16 | 0 | 0 |
| 2026-07 | EHAM | 12 | 0 | 0 |
| 2026-01 | EHAM | 7 | 0 | 0 |
| 2026-07 | EDDM | 4 | 0 | 0 |
| 2026-01 | EDDM | 2 | 0 | 0 |
| 2026-07 | LIRF | 1 | 0 | 0 |
| 2026-01 | LSZH | 0 | 0 | 0 |
| | **total** | **3,101** | **32** | **34** |

Seven of the eighteen airport-months place zero under both placebos; no airport-month has a
placebo above 4.3 % of its own real count.

**(e) Queue's per-cell bar.** Placed rows binned by `(airport, 10-minute cell)`. The bar
queue set: *no cell class where the placebo matches the real count.* On the gated set:

| binning | cells with a real placement | contested (placebo ≥ real) | real rows in them | placebo-only cells |
|---|---|---|---|---|
| absolute, day placebo | 2,713 | **12** (0.44 %) | 12 of 3,101 | 15 |
| absolute, stand placebo | 2,713 | **21** (0.77 %) | 21 of 3,101 | 8 |
| time-of-day, day placebo | 781 | **2** | 2 | 2 |
| time-of-day, stand placebo | 781 | **1** | 1 | 1 |

Every contested cell holds exactly one real row, so no cell's aggregate is decided by a
contested placement. The time-of-day rows answer queue's own concern directly — a stand chain
that repeats daily would concentrate its false placements in the same time-of-day cells, and
1 to 2 contested cells of 781 is not that.

**(f) The aircraft-type read** is dropped as a gate: type agreement is now a *key clause*, so
it is 100 % by construction and carries no information. (c), (d) and (e) replace it, and they
are the tests the refuted pass failed.

## 4. Measured class diff, per airport-month

Measured by `code/turnaround.py` on both ranking months at all ten airports under the exact
key of §2. **3,101 rows gain a clock**, 1,345 in 2026-01 and 1,756 in 2026-07.

| month | airport | eligible segments | unclaimed departures | chained | **placed (gated)** |
|---|---|---|---|---|---|
| 2026-07 | EDDF | 6,677 | 13,190 | 3,487 | **1,264** |
| 2026-01 | EDDF | 2,934 | 7,616 | 1,655 | **466** |
| 2026-01 | EGLL | 2,064 | 5,660 | 1,071 | **306** |
| 2026-01 | LIRF | 1,248 | 5,667 | 712 | **260** |
| 2026-01 | LFPG | 2,671 | 17,084 | 1,237 | **231** |
| 2026-07 | LFPG | 3,979 | 20,478 | 1,909 | **210** |
| 2026-07 | LEMD | 2,603 | 9,093 | 1,010 | **162** |
| 2026-07 | EGLL | 170 | 791 | 53 | **45** |
| 2026-07 | LEBL | 256 | 1,993 | 123 | **42** |
| 2026-01 | LEBL | 313 | 1,463 | 142 | **39** |
| 2026-01 | LEMD | 111 | 951 | 48 | **34** |
| 2026-07 | LSZH | 119 | 860 | 57 | **16** |
| 2026-07 | EHAM | 139 | 1,077 | 16 | **12** |
| 2026-01 | EHAM | 161 | 731 | 10 | **7** |
| 2026-07 | EDDM | 115 | 546 | 12 | **4** |
| 2026-01 | EDDM | 46 | 411 | 5 | **2** |
| 2026-07 | LIRF | 13 | 15,551 | 9 | **1** |
| 2026-01 | LSZH | 62 | 604 | 23 | **0** |
| | **total** | **23,681** | | **11,579** | **3,101** |

LTFM places nothing in either month (§18.1: the feed, not the join).

### 4.1 The LFPG ruling — desk, 2026-09-09

LFPG's **441** chain rows (231 january, 210 july) **get `obs_has_clock = 1` in the block and
stay OUTSIDE `occupied_by_exp_046`.** v16 already carries 777 + 873 clocked-unoccupied rows —
the held LFPG observation — and these join them. **`exp-047-queue` keeps writing every one of
them, and the hold on LFPG stands unchanged.** Nothing in this spec lifts it and nothing in
it re-opens §17 row 4.

So the class movement is **2,660 rows**, not 3,101: EDDF 1,730, EGLL 351, LIRF 261, LEMD 196,
LEBL 81, EHAM 19, LSZH 16, EDDM 6.

**Face, at §16.3's currency: 2,660 rows, +0.139 s at the level ×1.2 (the lower bound),
+0.208 s at ×1.5.** That is the figure to quote. It is a class-entry face, not a priced gain —
the rows enter the class and this lane's head then values them, which the fold table prices
after the re-price.

## 5. Interface columns that move

`artifacts/exp-047/block/<month>.parquet`, 34 emitted columns keyed by `MVT_ID_mvt`
(`analysis/blockpy-rebuild/interface-columns.json`). For a placed row every observation column
goes from null to a value, and:

- **`match_mode` gains the value 3** (`MODES["turnaround"]`), so a placed row is separable
  from a strict or a wide row in the block and in every arm that reads it;
- **`matched_on_modal`** is written `0.0` — no callsign matched — as the wide pass already
  writes for a non-modal form;
- **`clock_ok`** is `1.0` by construction (the key's last clause is `|dt| ≤ 60`);
- **`obs_ambiguous`** is `0.0` by construction (uniqueness both ways);
- **`turnaround_anchored`** is `1.0`, and for these rows it is the literal truth rather than
  an inference.

**No existing value changes** — gate (a) is the proof of that, not the claim.

## 6. Pins, manifests and the re-price

1. **`MATCHER_SHA` changes.** `code/arms.py` carries it and the block cache key already
   includes the matcher's first 8 hex (§14.4), so no stale block can be served into the
   re-price.
2. **Fold pins re-issued** — `code/population.py`, with its `superseded` block naming the
   previous keys, counts, `days_added` and `days_gone`. The pass changes which rows are in the
   fold blocks, so every fold pin is re-priced, not carried.
3. **Class vector re-emitted as v18**: `artifacts/exp-046/class/<month>-v18.parquet` plus
   `class-manifest-v18.json`, same columns, `occupied_by_exp_046` still the mask and
   `written_by_exp_046` still the bytes. **2,660 rows leave `exp-047-queue`'s class and enter
   this lane's**; LFPG's 441 do not (§4.1). Carrying the desk's standing rule into the emit:
   **every row this lane occupies is written at the next serve, whatever value it carries** —
   the 526 occupied-unwritten rows and any EDDF/LEMD row that arrives holding queue's v17
   correction included.
4. **Both lanes re-price once, on the desk's single `MATCHER_SHA` line.**
5. **No serve until the desk gives the word**, and the serve is on v17's bytes
   (`submissions/uploaded/youthful-giraffe_v17.parquet`).

## 7. The four questions to queue

1. That **no column this pass writes is read by queue's block in a way the new mode breaks** —
   in particular that a `match_mode` of 3 is either used or ignored deliberately, and never
   silently treated as a wide match.
2. That the **2,660 rows leaving queue's class are rows queue does not currently move**, so no
   served byte is contested; and that queue **keeps** LFPG's 441 under §4.1.
3. That **queue's own census reproduces the per-airport-month table of §4**, and that its
   re-measured reach-grain read is produced at 3,101 rather than scaled from the 19,704 it
   measured at 9,875 — the placement is 31 % the size and LFPG's share of it is 14 % rather
   than 27 %, so both terms move.
4. **The neighbourhood spillover**: whether queue's neighbourhood features change for rows it
   **keeps**, because a neighbour of a kept row may now be a placed row. This is what §14.1
   found the hard way; it is asked for explicitly rather than discovered afterwards, and
   §3(e)'s per-cell tables are the evidence offered against it being biased where it happens.

## 8. What would stop this shipping

- Gate (a) finding **any** changed value on a row the pass did not place.
- Any airport over its §12.3 bound with the pass on.
- Either placebo rising above a few percent when re-run **inside `block.py`** rather than in
  `turnaround.py`: the two implementations must agree, and a disagreement is a defect in one
  of them, not a result.
- Queue's answer to question 4 showing a neighbourhood movement it cannot absorb, or to
  question 2 showing a contested byte.

**And one honest limit.** With the ±60 s check inside the key, that check is no longer an
independent test of the rows that ship. The placebos are the test. They break the key in two
different places — the calendar, and the stand itself — and both return about 1 %. That is the
strongest statement available without a label, and it is weaker than a label.

---

## 9. The count is 3,179, and the gates at that count

Desk ruling, 2026-09-09: **3,179 stands.** §2 of this spec says *an unclaimed ranking
DEPARTURE*, meaning unclaimed by passes 1 and 2; `block.build` implements exactly that, and
a claimed-but-clockless departure already owns its segment. Queue's GO holds at 3,179
(690315b). This section replaces the 3,101 of §3, §4 and §4.1 — those sections are left as
written and are superseded here rather than edited, so the reconciliation stays legible.

### 9.1 Why the two instruments disagreed, reconciled to the row

`code/chainpass.py` ran §3's gates on the branch copy **before** the commit, and §8's
agreement check failed: **3,179** inside `block.build` against `code/turnaround.py`'s
**3,101**.

The cause is a single definition. `turnaround.py` approximated *"a departure neither pass
claimed"* by *"a departure with no clock"*. Those are different sets: a **strict** match onto
a segment that carries a take-off but no off-block is claimed **and** clockless — 6,063 such
rows in 2026-01 and 5,472 in 2026-07, every one `match_mode` 0. `turnaround.py` let them
compete as free candidates, and the two-way uniqueness rule then **declined 78 real
placements** they contended for.

Running the same in-block pass with that approximation restored reproduces `turnaround.py`
exactly — 1,345 and 1,756 against 1,382 and 1,797 under the spec's own words
(`analysis/chain-reconcile.json`). The defect was in the measurement, not in the pass.

**The repair.** `code/turnaround.py` now reads the block's own `obs_matched`, which is the
claimed set `block.build` uses; the approximation is kept behind `FREE_IS_CLOCKLESS`,
defaulting **off**, so the reconciliation stays reproducible. Re-run, the two instruments
agree at **3,179 and airport-month by airport-month**, with placebos 29 and 32.

### 9.2 The per-airport-month table at 3,179

| month | airport | eligible | chained | **placed** | day placebo | stand placebo | in-block | agree |
|---|---|---|---|---|---|---|---|---|
| 2026-07 | EDDF | 6,677 | 3,493 | **1,292** | 12 | 9 | 1,292 | yes |
| 2026-01 | EDDF | 2,934 | 1,628 | **474** | 4 | 5 | 474 | yes |
| 2026-01 | EGLL | 2,064 | 1,068 | **315** | 1 | 4 | 315 | yes |
| 2026-01 | LIRF | 1,248 | 637 | **272** | 0 | 8 | 272 | yes |
| 2026-01 | LFPG | 2,671 | 1,243 | **237** | 5 | 1 | 237 | yes |
| 2026-07 | LFPG | 3,979 | 1,908 | **212** | 4 | 1 | 212 | yes |
| 2026-07 | LEMD | 2,603 | 1,001 | **168** | 3 | 2 | 168 | yes |
| 2026-07 | LEBL | 256 | 104 | **47** | 0 | 1 | 47 | yes |
| 2026-07 | EGLL | 170 | 52 | **45** | 0 | 0 | 45 | yes |
| 2026-01 | LEBL | 313 | 129 | **41** | 0 | 1 | 41 | yes |
| 2026-01 | LEMD | 111 | 44 | **34** | 0 | 0 | 34 | yes |
| 2026-07 | LSZH | 119 | 56 | **16** | 0 | 0 | 16 | yes |
| 2026-07 | EHAM | 139 | 13 | **12** | 0 | 0 | 12 | yes |
| 2026-01 | EHAM | 161 | 9 | **7** | 0 | 0 | 7 | yes |
| 2026-07 | EDDM | 115 | 11 | **4** | 0 | 0 | 4 | yes |
| 2026-01 | EDDM | 46 | 4 | **2** | 0 | 0 | 2 | yes |
| 2026-07 | LIRF | 13 | 9 | **1** | 0 | 0 | 1 | yes |
| 2026-01 | LSZH | 62 | 23 | **0** | 0 | 0 | 0 | yes |
| | **total** | **23,681** | **11,432** | **3,179** | **29** | **32** | **3,179** | |

1,382 in 2026-01 and 1,797 in 2026-07. LTFM places nothing (§18.1).

### 9.3 The class movement and the face

**§4.1's LFPG ruling is unchanged and applies to the larger count.** LFPG's **449** rows
(237 january, 212 july) gain `obs_has_clock = 1` and stay **outside**
`occupied_by_exp_046`, joining v16's 777 + 873 clocked-unoccupied rows. Queue keeps writing
them, in effect if not by instrument, and the hold on LFPG stands unchanged.

**Class movement: 2,730 rows.**

| airport | rows leaving queue's class |
|---|---|
| EDDF | 1,766 |
| EGLL | 360 |
| LIRF | 273 |
| LEMD | 202 |
| LEBL | 88 |
| EHAM | 19 |
| LSZH | 16 |
| EDDM | 6 |
| **total** | **2,730** |

**Face at §16.3's currency: 2,730 rows, +0.143 s at the level ×1.2 (the lower bound),
+0.214 s at ×1.5.** Against the 2,660 / +0.139 / +0.208 of §4.1 this is +70 occupied rows and
+0.004 s. Queue's riders were re-measured at 3,179 (690315b): about 4,213 reach-grain flips,
about 1,375 into cells it serves, and about 1,590 of the 2,730 are rows it moves today — a
**handover, not a contest**, and the standing rule carries: **the next serve writes every
occupied row, whatever value it carries**, on v17's bytes.

### 9.4 The gates, all four, at 3,179

**(a) The 34-column sweep — PASSES.** Both months built twice, `chain=False` and
`chain=True`, all 34 emitted columns compared on the rows the pass did **not** place:
**0 values changed**, over 34 columns and 341,662 rows (151,337 in 2026-01, 190,325 in
2026-07). The pass disturbs nothing it does not place.

**(b) §12.3's per-airport false-match bound — PASSES**, `code/chainboundb.py` →
`analysis/chain-bound-b.json`, `finding7.py`'s method run against the branch copy so
`code/block.py` need not move to take a reading. 2026-01, shifts +1.0 h, +1.5 h, +2.0 h,
with mode 3 counted in the false share beside strict and wide:

| airport | density | worst false share | its bound | headroom | mode-3 rows in a shift | within |
|---|---|---|---|---|---|---|
| EDDF | 1.700 | 0.000191 | 0.000850 | ×4.45 | **0** | yes |
| EDDM | 3.371 | 0.001187 | 0.001686 | ×1.42 | **0** | yes |
| EGLL | 1.850 | 0.000259 | 0.000925 | ×3.57 | **0** | yes |
| EHAM | 3.235 | 0.000986 | 0.001618 | ×1.64 | **0** | yes |
| LEBL | 2.061 | 0.000246 | 0.001030 | ×4.19 | **0** | yes |
| LEMD | 2.236 | 0.000119 | 0.001118 | ×9.39 | **0** | yes |
| LFPG | 0.782 | 0.000110 | 0.000391 | ×3.56 | **0** | yes |
| LIRF | 1.615 | 0.000181 | 0.000808 | ×4.46 | **0** | yes |
| LSZH | 2.385 | 0.000700 | 0.001192 | ×1.70 | **0** | yes |

Every airport is inside its own bound, the tightest at ×1.42, and the false shares are the
ones `finding7.json` already records — the pass moves them by at most a row, because **it
places nothing at all under any of these shifts**. That is honest rather than impressive:
§12.3's shifts are 1 to 2 hours and the key's last clause is ±60 s, so mode 3 is zero by
construction here. The shift that can test this key is the 24-hour one, where daily
periodicity puts a real departure near the same clock time — and that is gate (c).

**(c) The day placebo — PASSES.** 29 placements against 3,179, **0.91 %**, per airport in
§9.2 (worst EDDF july, 12 against 1,292).

**(d) The stand placebo — PASSES.** 32 against 3,179, **1.01 %**, worst LIRF january, 8
against 272.

### 9.5 Queue's per-cell bar at 3,179, with the row column

Queue's condition on the relayed GO: **every contested cell holds exactly one real row.**
Measured on both instruments, pooled over the two months:

| instrument | binning | placebo | cells with a real placement | contested (placebo ≥ real) | **real rows in them** | placebo-only |
|---|---|---|---|---|---|---|
| `turnaround.py` | absolute | day | 2,776 | 11 | **11** | 14 |
| `turnaround.py` | absolute | stand | 2,776 | 26 | **26** | 2 |
| `turnaround.py` | time-of-day | day | 789 | 2 | **2** | 1 |
| `turnaround.py` | time-of-day | stand | 789 | 2 | **2** | 0 |
| `block.build` | absolute | day | 2,776 | 15 | **15** | 29 |
| `block.build` | absolute | stand | 2,776 | 20 | **20** | 10 |
| `block.build` | time-of-day | day | 789 | 2 | **2** | 3 |
| `block.build` | time-of-day | stand | 789 | 3 | **3** | 1 |

**Every contested cell holds exactly one real row, on both instruments and both binnings.**
The equality is a proof and not a coincidence: a cell is contested only when it holds at
least one real row, so N contested cells holding N real rows between them is one apiece. No
cell's aggregate is decided by a contested placement, and the time-of-day rows — 2 to 3
contested of 789 — answer the daily-repeat concern that prompted the bar.

Per month the in-block figures are 2026-01 day 5 and stand 12 of 1,283 cells, 2026-07 day 10
and stand 8 of 1,493, each again one real row per contested cell.

**Queue's condition is met, so the relayed GO is live and the pass proceeds to the single
`code/block.py` commit.**
