# The coordinated `block.py` rebuild — change spec

**Written for the handshake** the desk set on 2026-09-09 (order 3): this lane writes the
spec, the desk hands it to `exp-047-queue`, queue answers the desk in one line — agree, or
the objection — the desk relays *go*, and only then does `code/block.py` change, **once,
with every item in it**. Nothing is served before the rebuild's re-price.

`code/block.py` is **untouched** while this is read. Every number below was measured on a
branch copy, `analysis/blockpy-rebuild/block_v2.py`, against the shipped file, on all five
months both lanes build.

| | |
|---|---|
| shipped `code/block.py` | `695af9737e5b082f…` (the sha both lanes' manifests carry today is `ba41935e…`; see §7) |
| the proposed file, ready to read | `analysis/blockpy-rebuild/block_candidate.py`, `6fdf39519eb0f500c06230e2972d3c8300aa6222ed8d9b003998ddf6e286093b` |
| what it is | the branch copy with its own path anchor removed, i.e. **exactly** what `code/block.py` becomes if queue's answer changes nothing |
| measured by | `analysis/blockpy-rebuild/`: `measure.py` (both variants, every emitted column), `diffcols.py`, `diffrows.py`, `density.py` |
| evidence | `analysis/blockpy-rebuild/boundary-diff.json`, `row-diff.json`, `interface-columns.json`, `reach-diff.json`, `summary-{shipped,v2}.json`, and `analysis/eddm-density.json` |

**The one number queue needs first.** Across **854,937** departure rows in five months,
**605 rows differ in any way at all** — swept over every one of the 34 columns the block
emits, not a chosen subset. Of those, **602 change class** (601 gaining, one losing) and
**3 keep their class** and move only in features read off a NEIGHBOUR: the arrival at the
same stand. Everything else in all five months is bit-identical. §8(3) has the split.

---

## 1. The tie-break (round 3 finding 2)

**The defect.** Each de-duplication re-sorted on `absdt` **alone** with pandas' default
sort (quicksort, unstable), so a tie between two candidates resolved by the frame's own row
order. The same matcher, on the same summaries, could give a row a different clock
depending on whether it was handed the whole file or one month — which is not a matcher, it
is a coin.

**The change.** The order becomes **total** and the sort **stable** (`kind="mergesort"`).
Strict pass:

```python
j["absdt"] = j.dt_takeoff.abs()
n_cand = j.groupby("MVT_ID_mvt").size().rename("obs_n_candidates")
j = j.merge(n_cand, on="MVT_ID_mvt", how="left")
n_multi_seg = int((n_cand > 1).sum())
j = j.sort_values(["absdt", "matched_on_modal", "seg_key"],
                  ascending=[True, False, True], kind="mergesort")
j = j.drop_duplicates("MVT_ID_mvt")
n_multi_dep = int((j.groupby("seg_key").size() > 1).sum())
j = j.sort_values(["absdt", "matched_on_modal", "MVT_ID_mvt"],
                  ascending=[True, False, True], kind="mergesort")
j = j.drop_duplicates("seg_key")
```

Wide pass, the same treatment on its own keys:

```python
w = w.sort_values(["in_wide_window", "absdt", "implied_taxi", "seg_key"],
                  ascending=[False, True, True, True], kind="mergesort")
w = w.drop_duplicates("MVT_ID_mvt")
w = w.sort_values(["in_wide_window", "absdt", "implied_taxi", "MVT_ID_mvt"],
                  ascending=[False, True, True, True], kind="mergesort")
w = w.drop_duplicates("seg_key")
```

**Reading the keys.** Each dedup sorts on the tie-breaker the *other* key supplies, so the
last key is unique within the group being reduced and no frame ordering can permute it:
reducing to one row per departure breaks a tie on `seg_key`; reducing to one row per
segment breaks it on `MVT_ID_mvt`. `matched_on_modal` stays second, before either id, so a
tie between two **forms** of one segment still resolves to the modal — what the modal-only
join would have chosen, which is the property the shipped comment already claims.

One shipped line disappears: the pre-sort `j.sort_values(["absdt", "matched_on_modal"],
ascending=[True, False])` that ran before `n_cand`. `n_cand` is a `groupby().size()` and is
order-free, and both dedups now sort for themselves, so the line was dead.

**What it moves — the whole of it, per airport, both directions.**

| month | airport | matched + | matched − | clock + | clock − | `reach_class` changed |
|---|---|---|---|---|---|---|
| 2025-01 (january fold) | — | 0 | 0 | 0 | 0 | **0** |
| 2025-07 (july_loo fold) | LEMD | 0 | 0 | **2** | 0 | 2 |
| 2025-12 (december fold) | EGLL | 0 | 0 | **3** | 0 | 3 |
| 2025-12 | EHAM | 0 | 0 | **1** | 0 | 1 |
| 2025-12 | LEMD | 0 | 0 | **6** | 0 | 6 |
| 2026-01 (ranking) | LEMD | 0 | 0 | **12** | 0 | 12 |
| 2026-01 (ranking) | EGLL | 0 | 0 | 0 | **1** | 1 |
| 2026-07 (ranking) | EGLL | 0 | 0 | **7** | 0 | 7 |
| 2026-07 (ranking) | LEMD | 0 | 0 | **1** | 0 | 1 |

Every other airport in every month is 0/0/0/0. **33 rows in 854,937**, one of them a loss.
On the fold side that is **12 rows of 510,096** (0.0024 %); the fold prices cannot move
measurably and the re-price will say so rather than assume it.

**What kind of row moves.** All 33 were already matched, all **strict** (`match_mode = 0`),
all with a take-off, and all **without a stand-departure onset** — so `matched +` is 0 and
`clock +` is 33. The different survivor of the segment dedup carries an onset where the old
one did not, so the row gains a clock while `match_mode` and `obs_wheels_off` stay exactly
as they were, to the digit. The rows sit at three airports only: LEMD 9, EGLL 10, EHAM 1
across the four months, plus 12 LEMD and 1 EGLL in 2026-01.

---

## 2. The stale `HELD_AIRPORT_DAYS` values

**The defect.** The table holds five EDDF airport-days and carries a receipt rate per day
as the value. Those rates were read from a superseded `receipt.json`; at the committed
`49c953b7` the same days read **0.1347 / 0.0985 / 0.1012** on 01-29/30/31, and **01-27 and
01-28 are not outage days at all** — the harvest's corrected receipt puts them with the
airport's ordinary reception, and the five-day span was the first reading of a three-day
outage.

**The change.**

```python
HELD_AIRPORT_DAYS = {
    ("EDDF", "2026-01-29"): 0.1347,
    ("EDDF", "2026-01-30"): 0.0985,
    ("EDDF", "2026-01-31"): 0.1012,
}
HELD_REASON = ("EDDF feeder outage 2026-01-29..31 (desk 2026-09-08 14:45, corrected "
               "from the harvest's receipt.json at 49c953b7 on 2026-09-09)")
```

The values stay what they are — recorded to identify the days, never read as a threshold —
and `HELD_REASON` states the correction so a reader of either lane sees why the span
shortened. `exp-047-queue` reads this set through `B46.HELD_AIRPORT_DAYS` and keeps no
table of its own, so its `outage_set()`, its `reach_class == "outage"` audit label and its
`outage_report` all follow with no edit.

**What it moves.** 2026-01 at EDDF only: **+569 matched, +506 clocks**, 0 lost. Split by
day, 2026-01-27 gives 271 and 2026-01-28 gives 235. No other airport, no other month, and
no fold month at all — the whole table lives in a ranking month.

---

## 3. EDDM — the item as posed is refuted, and the honest replacement

The desk routed *"a look at EDDM's callsign pool"* into this rebuild, from round 3 finding
7: EDDM's worst-shift false-match share is **0.00120** against a lane-wide 0.00029, the
worst of the ten and the only airport over the test's asserted 0.001. The obvious
hypothesis is that EDDM broadcasts an unusually large pool of callsign forms, so the join
offers more keys and more chances for a wrong departure to land.

**Measured, that hypothesis is wrong** (`analysis/blockpy-rebuild/density.py` →
`analysis/eddm-density.json`, on 2026-01, the month finding 7 was read on):

| airport | departures | dep. segments | all-kind segments / departure | forms / dep. segment | keys reused that day | false share |
|---|---|---|---|---|---|---|
| **EDDM** | 10,953 | 11,339 | **3.371** | 1.609 | 2.35 % | **0.001200** |
| EHAM | 16,232 | 18,458 | **3.235** | 1.610 | 4.88 % | 0.000972 |
| LSZH | 9,995 | 10,094 | **2.385** | 1.596 | 1.26 % | 0.000516 |
| EGLL | 19,283 | 18,653 | 1.850 | 1.573 | 1.06 % | 0.000261 |
| LEBL | 12,201 | 12,330 | 2.061 | 1.636 | 0.80 % | 0.000247 |
| LIRF | 11,061 | 10,067 | 1.615 | 1.522 | 0.20 % | 0.000183 |
| EDDF | 15,667 | 12,853 | 1.594 | 1.457 | 0.55 % | 0.000129 |
| LEMD | 16,802 | 18,151 | 2.236 | **1.643** | 2.52 % | 0.000119 |
| LFPG | 18,129 | 10,621 | 0.782 | 1.053 | 0.14 % | 0.000112 |

LTFM is the tenth and is left out of the table and the correlations: it delivers no
summaries at all, so it has no density to read and its false share is 0 by construction.

*Re-measured on the rebuilt matcher, 2026-09-09 after the commit (`analysis/finding7.json`,
`code/finding7.py` now measures the density and asserts the bound). Every airport is within
its own bound, tightest EDDM at ×1.40. Only EDDF's row moves, because its two held days came
back: density 1.594 → **1.700**, false share 0.000129 → **0.000194**, headroom ×4.38. The
table above is the reading the handshake was agreed on and is left as it stood.*

- **The pool does not explain it.** EDDM's 1.609 forms per departure segment is **fourth of
  nine** — LEMD (1.643), LEBL (1.636) and EHAM (1.610) all broadcast more. Spearman against
  the false share is **+0.35**, the weakest of the four quantities measured. **LEMD is the
  counterexample that settles it**: the largest pool of all nine, and the second *smallest*
  false share, a factor of ten under EDDM's.
- **Segment density does.** All-kind ground segments per departure ranks EDDM, EHAM, LSZH
  first, second and third — **the same three airports, in the same order, as the false
  share**. Pearson **+0.884**, Spearman **+0.817**.
- Key reuse within a day (Spearman +0.60) and departure segments per departure (+0.567) sit
  between the two, which is what one expects if density is the cause and they are its
  correlates.

**So the change is not to callsign handling.** Nothing in the join is wrong: a larger pool
is what buys the non-modal matches §12.3 measured at three quarters the worth of a modal
one on all three folds, every interval clear of zero. What is wrong is the **assertion**,
which generalises one pilot-week EHAM number to ten airports of very different density.

**Proposed — and note which assertion moves.** `test_block.py` has two constants,
`PLACEBO_MAX_WIDE_SHARE` and `PLACEBO_MAX_STRICT_SHARE`, both 0.001, and it asserts them on
the **pilot week** (EHAM, 5,131 rows), where the worst shift gives 1 row. **Those stay
exactly as they are**: they are calibrated on the cohort they are asserted on, they pass
there, and loosening them would be the opposite of the repair. What is missing is any
assertion at all on the other nine airports — which is why EDDM can read 0.00120 while the
suite is green.

So the addition is a **second, per-airport bound on the whole-month cohort finding 7 reads**
— a different quantity on a different cohort, not a replacement:

```python
#: the whole-month per-airport bound (finding 7's cohort), scaled by the airport's own
#: measured density.  Distinct from PLACEBO_MAX_*_SHARE, which stay on the pilot week.
def month_false_match_bound(density: float) -> float:      # density = all-kind ground
    return max(0.0003, 0.0005 * density)                   # segments per departure
```

asserted by `code/finding7.py` when it re-runs, and re-read cheaply by `test_block.py` from
`analysis/finding7.json` so the recorded number cannot drift from the bound in silence.

Every airport passes, tightest EDDM at **1.40×** and loosest LEMD at **9.4×** (EHAM 1.66,
LSZH 2.31, LFPG 3.49, EGLL 3.54, LEBL 4.17, LIRF 4.41, EDDF 6.18). No airport rides the
floor today — it is there for LTFM, which delivers no summaries at all, and for any airport
whose reception falls below 0.6 segments per departure. **It moves no row and no byte**: the
matcher is unchanged by this item, which is why EDDM's column in §8 is zero in every month.

A byte-moving remedy — a narrower identity window at dense airports — is a different claim
and cannot ride this handshake: it changes the class boundary and must be priced as an
**arm under the ordinary gate** (every fold clear, else MAYBE). It goes to
`docs/BACKLOG.md` with that as its revival condition, not into `block.py`.

---

## 4. The `published_short` read

The harvest's `manifest.py` (91b64a5) now writes `published_bytes` and `published_short` on
every day record. A short archive is a **partial** one: its summaries are built from part
of the day, so the day reads as sparse rather than absent, and no receipt rate can tell the
two apart.

**The change** — a hold with its own criterion, recorded apart from the hand-diagnosed one:

```python
def short_days() -> set[str]:
    """Days whose ARCHIVE is short of the size the source published for it."""
    global _SHORT
    if _SHORT is None:
        out: set[str] = set()
        if MANIFEST.exists():
            man = json.loads(MANIFEST.read_text())
            for rec in man.get("months", {}).values():
                for day, d in rec.get("days", {}).items():
                    if d.get("published_short"):
                        out.add(day)
        _SHORT = out
    return _SHORT
```

read at the same choke point as the outage table, one branch later, and reported as
`days_held_short` in the build attrs. It holds **every airport** on such a day, because the
archive is the day and not one airport's feeder.

**What it moves: nothing, today.** Exactly one of the 397 days that carry a day record is
flagged —
**2025-06-11** (0.73 GB against 3.30 published, 5 % receipt on 781 rows). It is in no fold
month and no ranking month, so the measured diff for this item is **0 rows in all five
months**. It is in the rebuild so that the next short day is held the moment it is
delivered, rather than after someone notices a dip.

The population key is deliberately **not** changed: a short day's file is present and its
bytes are what they are, so it stays in `population_triples`, and the hold happens at read.
Holds have never been in the key (`HELD_AIRPORT_DAYS` is not either), and putting one there
would silently re-key both lanes' pins.

---

## 5. Round 5 finding 6 — the revert assertion that would stop the next serve

`serve.py:866` requires every row the head class vector calls occupied to differ from the
price base in the head file:

```python
bad = int((head_occupied & (headv == pbasev)).sum())
```

`classvec.py:114` defines occupancy as `served_by_exp_046 & ~held`. On **v16** that includes
the **1,481 rows the tail guard returned to v12** — occupied by this lane, and equal to the
price base by construction. A v17 composed on v16 therefore fails the assertion and stops,
on a file that is correct.

**The change**, in this lane's files and not in `block.py`: `classvec.py` publishes a third
column, `guard_held_by_exp_046`, on the v16 class vector, and `serve.py` accepts

```python
bad = int((head_occupied & (headv == pbasev) & ~guard_held).sum())
```

so the exemption is **declared in a sha-pinned file and checked against it**, exactly as the
occupancy repair on v15 was, rather than inferred from the values. Queue reads neither
column; it is listed here because it lands in the same commit block and because the
`served / occupied / guard-held` vocabulary is shared.

## 6. Round 5 finding 10 — the serving cut is re-derived, not trusted

`serve.py:789` reads the guard's serving cut out of `analysis/guard-arms.json`, an unhashed
file, and checks only that the quantile matches. The pinned panel is already loaded at that
point, so the cut can simply be recomputed:

```python
assert abs(G.cut(pan, list(A.FOLDS)) - guard_cut) < 0.01
```

This closes the gap between the pricing and the serve at no cost. (`guardarm.py:55` rounds
the serving cut to 0.01 s where the fold side used full precision; the assertion's tolerance
is that rounding, and the difference is immaterial — 1,864.41 either way.)

---

## 7. What queue re-issues, and what it must not patch

`exp-047-queue`'s class manifests and gate records are pinned to matcher sha `ba41935e…`.
They are **re-issued at the rebuild, not patched**: the rebuild changes the class boundary
on **590 ranking rows** (and 12 fold rows), so a record that names the old sha describes a population that no
longer exists, and editing the sha in place would leave the rest of the record describing
the old rows. The sequence is the desk's:

1. queue answers this spec in one line to the desk.
2. the desk relays *go*; this lane commits `block.py` **once**, with §1, §2, §3 (test only),
   §4, §5 and §6 in it, and reports the sha.
3. the desk sends **one** `MATCHER_SHA` line to both lanes. Neither lane sends one per edit.
4. both lanes rebuild every month, re-write their pins on the current population, and
   re-price. `exp-046`'s climbed capacity step is re-priced **cross-fitted** (§13.1), and
   the tail guard is re-priced **as an arm on its own** under the ordinary gate.
5. nothing is served until step 4 is on the record.

`MATCHER_SHA` in `code/arms.py` and `MATCHER_SHA_BEFORE_THE_POPULATION_PIN` are updated in
the same commit, so the handover report and the two manifests name the file that built them.

---

## 8. The four questions the desk asked to be answerable as a check

**(1) The per-month class-boundary diff at EDDF and EDDM by name.**

| | 2025-01 | 2025-07 | 2025-12 | 2026-01 | 2026-07 |
|---|---|---|---|---|---|
| **EDDF** | 0 | 0 | 0 | **+569 matched, +506 clocks** (01-27: 271, 01-28: 235) | 0 |
| **EDDM** | **0** | **0** | **0** | **0** | **0** |
| EGLL | 0 | 0 | +3 clocks | −1 clock | +7 clocks |
| EHAM | 0 | 0 | +1 clock | 0 | 0 |
| LEMD | 0 | +2 clocks | +6 clocks | +12 clocks | +1 clock |
| LEBL, LFPG, LIRF, LSZH, LTFM | 0 | 0 | 0 | 0 | 0 |

**EDDM does not move at all**, in any month — §3 is a change to a test assertion, not to the
matcher. EDDF moves only in 2026-01, and only from §2.

**(2) The effect on EDDF's days returned to service, in queue's own grain.**
Read with queue's definitions — `own` = a heard clock (`0 < taxi < 20000`), a cell is
`airport | floor(MVT_TIME / 600 s)`, `No_n_n_cell` = OTHER heard rows in the cell, *reached*
= `~own & No_n_n_cell ≥ 1` — on EDDF's 15,667 ranking rows in 2026-01:

| | before | after | change |
|---|---|---|---|
| own clock (this lane's class) | 7,374 | 7,879 | **+505** |
| **reached** (queue's served cell) | 5,339 | **5,693** | **+354** |
| unreached | 2,954 | **2,095** | **−859** |
| unreached → reached | | | **354** |
| reached → unreached | | | **0** |
| no clock → own clock | | | **505** |
| own clock → no clock | | | **0** |

So **EDDF/reached grows by 354 rows** and EDDF/unreached shrinks by 859. Every one of the
960 rows on the two returned days was previously unreached — the whole airport-day had no
clock, so no row had a heard neighbour; 505 gain their own clock and leave queue's
population for this lane's, 354 gain a neighbour and enter EDDF/reached, and 101 stay
unreached. Nothing moves out of reached. Month-wide, 2026-01 goes 21,481 → 21,825 reached
and 43,581 → 42,721 unreached; 2026-07 goes 26,865 → 26,857 reached (the eight EGLL and
LEMD tie-break clocks) with unreached unchanged. (Queue's own count will differ from the
class diff of §1–§2 by one or two rows in each direction, because queue additionally drops
a taxi outside `(0, 20000)` s; the tables above already apply that filter.)

**(3) Which interface columns' inputs move, and which do not.**
`analysis/blockpy-rebuild/diffcols.py` → `interface-columns.json` (per column) and
`diffrows.py` → `row-diff.json` (per row), swept over **all 34 emitted columns**.

**Every column moves on at least one row, so the honest answer is the ROW set, not a list
of safe columns:**

| month | rows | **rows differing at all** | of those, class changed | **spillover: class held** |
|---|---|---|---|---|
| 2025-01 (january fold) | 153,706 | **0** | 0 | 0 |
| 2025-07 (july_loo fold) | 190,713 | **3** | 2 | **1** (EHAM) |
| 2025-12 (december fold) | 165,677 | **11** | 10 | **1** (LEMD) |
| 2026-01 (ranking) | 152,719 | **583** | 582 | **1** (EGLL) |
| 2026-07 (ranking) | 192,122 | **8** | 8 | 0 |
| **total** | **854,937** | **605** | 602 | **3** |

**The three spillover rows are the part that is not obvious and the part to check.** Their
own match is unchanged, but a feature computed from a neighbour moves because a NEARBY row's
match changed:

- 2025-07 EHAM and 2025-12 LEMD, one row each: `arr_stand_occupancy_s` and
  `arr_gap_reception_s` only. Both are read off the arrival that used the same stand, so a
  neighbour gaining a clock re-dates the stand's occupancy.
- 2026-01 EGLL, one row: the same two, plus `seg_points`, `n_ground_points`,
  `seg_dt_median_s`, `seg_gap_max_s`, `seg_span_s`, `callsign_n` — a different segment
  survived the dedup while the row's class held either way, so the shape columns follow the
  new segment and the clock is the same to the digit.

Grouped by what moves them, on the 602 class-changing rows:

| group | columns | what moves them |
|---|---|---|
| match identity | `obs_icao`, `match_mode`, `obs_matched`, `obs_has_clock`, `reach_class`, `obs_ambiguous`, `obs_n_candidates`, `matched_on_modal`, `clock_ok`, `has_wheels_off` | a row entering or leaving the matched set (569 in / 0 out in 2026-01; the class flips of §1 elsewhere) |
| clocks | `obs_taxi`, `obs_seg_taxi`, `obs_taxi_motion`, `obs_wheels_off`, `dt_takeoff`, `time_to_250m`, `seen_before_move_s`, `stand_stop_s` | the same rows; 538 gains and 1 loss across the five months |
| geometry → `route_km`, `stand_dist_km` | `stand_lat`, `stand_lon`, `stand_dist_km`, `stand_sin`, `stand_cos` | the same rows; 555 gains, 1 loss |
| segment shape | `seg_points`, `n_ground_points`, `seg_dt_median_s`, `seg_gap_max_s`, `seg_span_s`, `callsign_n` | the same rows, plus the one EGLL spillover row |
| turnaround | `arr_present`, `arr_taxi_in_s`, `arr_stand_occupancy_s`, `arr_gap_reception_s`, `turnaround_anchored` | the same rows, plus all three spillover rows |

Nothing outside those 605 rows differs by a single bit in any month, and the summaries, the
callsign join and `prefix_map` are untouched by every item in this spec.

**`obs_offblock` is not one of the 34, and is covered transitively** (queue, at the
handshake: its `Oob_` and `Hob_` anchors and `own_obs_taxi` read it). This lane emits the
taxi, not the onset: `obs_taxi = mvt_s - obs_offblock`, and `mvt_s` is a ranking-file column
no item here touches, so `obs_offblock = mvt_s - obs_taxi` moves exactly where `obs_taxi`
moves and nowhere else. Where queue instead re-derives the onset from the summaries through
its own join on `(airport, obs_icao, obs_offblock)`, the summaries are untouched and
`obs_icao` **is** one of the 34 — 569 gained in 2026-01, nothing changed on a row present in
both — so that path lands on the same rows. Either way the answer is the 605-row set above.

**(4) Queue's pinned records.** See §7: re-issued at the rebuild, never patched.

---

## 9. What this spec does not carry

- **No new hold, threshold or window.** §4 adds a hold whose criterion is the harvest's own
  flag and which fires on no day either lane builds today.
- **No change to the wide rule, the identity window, `REQUIRE_ALT`, the callsign
  normalisation or `prefix_map`.**
- **No re-price.** The re-price is step 4 of §7, after the commit, and is where the fold
  numbers and both pins are re-read. Nothing here is a price.
- **No serve.** The desk's standing order holds: nothing is served before the rebuild's
  re-price.
