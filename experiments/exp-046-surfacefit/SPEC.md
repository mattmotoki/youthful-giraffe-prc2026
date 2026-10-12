# exp-046-surfacefit — the surface-observation programme, modelling stage

Worker `surfacefit`, tier hard, 2026-09-08. Assignment `assignments/surfacefit.md` (recorded rule, stage 2).
Reviewer: `gpt-5.6-sol` at `xhigh`, one round on the fold table and one on the served candidate.

## My reading of the assignment

The assignment asks for **a correction of the shipped head, driven by an observation, priced on the
contract** — not for a model. So three things are fixed before any code is written and stay fixed:

1. **The base is v11** (`docs/SUBMISSIONS.md` row 11: `v10 + 0.217·(stack-5 − v10)`, board 263.0298).
   Fold-side v11 is built by that same arithmetic from `exp-038-packet/code/vectors.py`'s gated
   `v10 (exp-036 refitbulk-mask)` and `stack-5`, so no member and no rotation is refitted anywhere in
   this experiment. Every number is `r = y − p_v11`, `d = p_arm − p_v11`, `2A − B`.
2. **The observation is an input, never a label and never a base.** The feature block is a function of
   `ranking.parquet`-legal columns and of the adsb.lol ground summaries alone. `BLOCK_TIME_UTC_mvt` and
   `TAXITIME_SEC_mvt` enter only as `y` on the training side of a fit, never as a feature, and the
   validation month's block time is never read at all (the block is computed from the masked frame).
3. **The 388 noac ranking rows stay bit-identical to v11.** Their error is the label's placeholder
   semantics (exp-033), not the physical off-block event, so an observation of the event cannot be
   evidence about them. `d = 0` on them by construction, checked, not assumed.

**The reach question is the airport question, and it is already open.** The harvest's first two days
(`exp-045/analysis/receipt.json`, 2026-01-01/02) put receipt at 96–99 % at EDDM, LEBL, LEMD, 95 % at
LSZH, 98 and 81 % at EHAM, and at **73 and 66 % (EDDF), 70 and 65 % (LIRF), 51 and 49 % (EGLL),
16 % (LFPG) and 0.0 % (LTFM)**. *Re-read 2026-09-09 from the harvest's repaired `receipt_rate` (a
flight number flown daily used to be counted as received on every day). A single day's rate barely
moves under the repair — only EHAM 2026-01-02 (81 %, not 91–95) changes this line's reading — so the
airport ordering this design was built on stands. A MONTH's rate moves a great deal; see §4 and §8.1
of `RESULTS.md`.* The
assignment's repair clause covers exactly this, and the design answers it structurally rather than by
hand: every arm carries **one weight per airport fitted off the scored rows**, so an airport whose
observation does not carry gets a fitted weight at or near zero and moves nothing, and an airport with
no observation at all has no matched row and is untouched by construction. No airport is ever excluded
by my judgement; the exclusion is a number the rotation produces.

**The base is v12, not v11, from 2026-09-08 14:50** (desk, tail rule). v12 is v11 with only its K2
member replaced by `exp-035-k2climb` arm 3b's climbed member (board 262.7418, file sha `48dd572d`).
`code/base.py` rebuilds its fold side by exp-035 §26a's own composition over vectors that lane
already gated, and gates the rebuild on §26c's published gain over v11 through this experiment's own
`price()` (+0.376248 against +0.3762 all-row, +0.446034 against +0.4460 clean). **EDDF and EHAM are
exp-036's two hold airports and sit at v7 in both compositions, so v12 IS v11 there row for row** —
which is where this lane's observation reaches, so every EHAM number measured against v11 carries
over unchanged and a held row stays bit-identical to v12. `p11` stays on the panel because
`code/reproduce_pilot.py` reproduces a v11-measured table. No amplitude is revised on a board
reading (score-feedback rule).

**Five EDDF airport-days are held out as an outage, not read as reception** (desk, 2026-09-08
14:45). EDDF's feeder failed on 2026-01-27..31 (64.7, 54.8, 13.5, 9.9, 10.1 % receipt against a
64.2 % median over the 27 other days), so `block.HELD_AIRPORT_DAYS` drops those summaries entirely:
no row matches on them, no segment of theirs feeds a neighbour in `exp-047-queue`'s block, and both
lanes see the same no-clock class. EDDF's receipt is then read on the 27 remaining days (64.6 %,
against 58.9 % for the month). *Corrected 2026-09-09 on the harvest's repaired rate; this paragraph
read "82.3 → 10.1 % receipt after 91–98 %" and "91.7 %, against 83.0 % for the month". Two
consequences for the coordinated `block.py` rebuild with `exp-047-queue`, neither of them acted on
here: on the repaired rate only **01-29, 01-30 and 01-31** are dips, so 01-27 and 01-28 are held
without being dips; and `HELD_AIRPORT_DAYS` carries the superseded rates as its dictionary values —
they are recorded there, never read, so the hold itself is unaffected.* The
hold is a declared list and not a threshold, because no threshold separates an outage from LFPG's
steady 24 %; `reachwhy.receipt_dips()` names every other candidate for the desk to rule on. This
edits `code/block.py`, so the shared matcher sha moves from `4a1040e0` to `a0652dff` and both the
desk and `exp-047-queue` were told in one line; the held days are all in 2026-01, so no 2025-07
class moves and the handover census is unchanged under the new sha.

**The head serves one reach class, and the assignment's scope narrowed to it** (desk,
2026-09-08 13:40 and 14:10). `exp-047-queue`'s step 1 priced a head carrying the row's own observed
taxi on the two reach classes apart: **+6,884 s² on the rows with a clock of their own and −588 s² on
the rows without one**, because where the feature is missing the correction can only move the level.
So this lane's arms fit and serve `arms.SERVED_CLASS` — `obs_has_clock & obs_taxi.notna() & ~noac` —
and leave every other row bit-identical to v11; `exp-047-queue` prices and serves the no-clock class
from its neighbourhood block (`artifacts/exp-047/block/<month>.parquet`, keyed by `MVT_ID_mvt`) and
leaves mine at v11; each candidate gets its own review round; the desk composes the two disjoint
updates on v11 and verifies the composed file; and either may go up alone. The two lanes use **one**
predicate — that lane imports this lane's `code/block.py` as the matcher and keys on the same
`obs_has_clock` — pinned here as `arms.MATCHER_SHA` and re-checked in every report, so a
change to the matcher cannot move the boundary under two lanes in silence. The pin has
moved three times and the current value is **`eded54e5…`**: `4a1040e0` was the first
matcher, `a0652dff` added the EDDF outage hold, `ba41935e` is the callsign-form repair
(§4.2 of `RESULTS.md`), and `eded54e5` is the population pin of 2026-09-09 (§12.7). *Updated
2026-09-09 (round 5 FALLBACK, finding 2); this sentence read "has moved twice and the
current value is `ba41935e…`" after the pin had already moved it.* The last move is the
only one that changes **no** row's class: the pin adds `_PIN`, `pinned()`,
`pin_for_month()` and one guard clause in `_summary_path`, all inert when no population is
pinned, and the ranking census is identical across it, so `exp-047-queue` is told but needs
no interface rebuild for it. Round 3 (FALLBACK) records that the pin alone does **not** fix the
boundary: `block.py`'s tie-break between two segments at equal |dt| is an unstable sort,
so the same matcher on the same summaries can give a different clock depending on the
frame it is passed (21 rows whole-file against per-month). That is finding 2 of round 3
and it is the desk's to call, because the fix changes the sha both lanes pin. The reading I rejected:
that "plan your no-clock rows on it" meant a second head inside this candidate fitted on that lane's
columns; the desk chose the split, and building it here would have duplicated that lane's step 2.

### The readings I took where the assignment leaves a choice

- **The block reads one schema, the harvest's.** The pilot's stored EHAM week is re-expressed as
  `exp-045` per-airport-per-day summaries by `code/summaries.py`, which calls the harvest's own
  `all_segments()` on the pilot's stored points. The block therefore never has a pilot branch and a
  harvest branch that could drift apart, and step 1's unit test is a test of the code that will serve.
- **The match key is the take-off day, not the off-block day.** The pilot keyed the cohort on
  `BLOCK_TIME_UTC_mvt.floor("D")`, which is blanked in the ranking file and so cannot be a serving key.
  The block keys on `MVT_TIME_UTC_mvt` and loads the summaries of that day and its two neighbours, then
  lets the ±15 min take-off window discriminate. On the pilot's week the two keys agree on every matched
  row (`match.json`: `date_agrees` on all of them), so the change is free where it is checkable.
- **The arrival-side segment joins on `icao`, not on the callsign.** An aircraft's inbound leg carries a
  different callsign from its outbound leg, so a callsign-day join would find almost nothing. The
  aircraft's ICAO24 is in the observation itself; the arrival is the last segment of the same
  `(airport, icao)` that ends before the departure segment starts, and it yields the observed taxi-in
  and the observed stand occupancy before pushback. Neither reads a label.
- **The amplitude is fitted per airport by rotation, not pooled and not per airport-month.** Pooled
  throws away the fact that LTFM has no observation and LEMD has a good one; per airport-month has 10–40
  matched rows in the thin cells and would fit noise. One weight per airport per fold, fitted on the
  2025 months that are not the scored fold, is the coarsest thing that can answer the reach question.

- **The strict join reads every callsign form the ground run broadcast, not the modal alone**
  (`block.FORMS`, added 2026-09-08 after the desk's 15:05 identity question). A summary carries
  `callsign_modal` and `callsigns_all`, and 23 % of ground runs carry more than one form, because an
  aircraft that lands as one flight number departs as another and it is one run. Joining on the modal
  alone therefore loses real departures — 9.3 points of EDDF, 8.0 of LIRF, 5.4 of EGLL on 2026-01. Ties
  are still resolved to the modal, the form is recorded as the feature `matched_on_modal`, and leading
  zeros are normalised on both sides. The price is measured, not assumed: it costs one strict false
  match in 5,131 on the +1 h placebo where the modal-only join costs none, so the placebo test now
  **bounds** the strict false-match rate at 0.1 % instead of forbidding it (`RESULTS.md` §1.2, §4.2).
- **The projection of a fold gain to the board is per airport-MONTH** (desk, 2026-09-08 16:50 and
  17:05). The 2025 feeder network is a different instrument from the 2026 one, and the 2026 one is not
  one number per airport either: LIRF is reached in January 2026 and dark for all 31 days of July 2026.
  So `arms.project_to_ranking()` multiplies each airport's per-moved-row fold gain by the rows and the
  reach of each ranking airport-month separately, and reports a seasonal sensitivity beside it that
  projects 2026-07 from the July fold and 2026-01 from the two winter folds. Every fold-side number
  stays on the contract and is stated apart from the projection.
- **The `c` arm is priced against a control that carries no observation** (`c_lgb_shipped_noobs`, added
  2026-09-08). The assignment names "the same with the shipped block" as an arm; but a residual model on
  the shipped block alone could read the base's own underfit on the with-clock rows and the observation
  would be credited with it. The control is the identical fit with the observation features removed, and
  the observation is worth the difference between the two.

### Readings rejected

- *A new base model.* The observation enters as an input to a correction of the shipped head. Every mask
  v11 holds (the EHAM mask inside v10, the noac hold) survives unchanged, which is what makes the
  candidate priceable against the record instead of against a new lineage.
- *Using the observation on the noac rows.* See point 3 above.
- *Serving the observation as a direct predictor* (`p = MVT_TIME − obs_offblock − offset`). The pilot
  measured its full amplitude negative (−475 s² in sample) and its fitted scale at about one half: the
  observation is informative and noisier than the served vector, so it enters at a fitted amplitude or
  not at all.
- *Waiting for the whole harvest before writing anything.* The harvest is at two gate days of 2026-01
  and the fold months come after the ranking months. Step 1 and the whole of step 2's machinery are
  built and unit-tested on the pilot's stored EHAM week first, so the fold table is one command when the
  months land.

## The rule conflict, recorded and resolved (2026-09-08)

The pilot worker's Sol round 1 changed the frozen rule while this stage was building on it:
a `first_airborne` point whose altitude is **null** is no longer read as a take-off
(`exp-043-pilot/code/segments.py`, commit `0beec1f`), and the pilot's published reach fell
from 4,930 to 4,901 matched on the week. The harvest's 2026-01 kernels had already run with
the rule as it was, and `exp-045-harvest/SPEC.md` forbids re-running a month's kernel once it
succeeds, because the re-run replaces its output.

**The reading taken at the time, and how it was superseded the same day.** The reading was:
the harvest's `all_segments()` as the kernels ran is the definition of the observation for
this stage, because the summaries on disk are the artifact and re-defining the rule after the
fact would make the record and the artifact disagree. **That reading no longer stands**, and
the paragraphs below are kept as the record of a change, not as the current rule. The desk
corrected it the same day — the harvest adopted the repair and re-derives an already-run month
from its stored points — so the current reading is the one in the next section: the rule is a
VERSION, every day says which version wrote it, and a fold table or a candidate reads exactly
one version (`code/block.py: one_rule`, `load_summaries(require_rule=…)`). Sol round 1 was
right that the two readings sat side by side in this file with nothing saying which won.

The divergence below is therefore measured rather than asserted away
(`analysis/block-test.json` §A0), and it is small and one-directional: **6 departure segments
of 5,441 over the pilot week (0.11 %)**, every shared segment bit-identical on the off-block
instant, and **5 matched departures of 4,930 (0.10 %)** whose chosen segment the repair would
remove. On those five the clock check reads −16 to −56 s, inside the −88…−4 s band the pilot
published for real take-offs, so the repair is not obviously an improvement on the matched
population; that judgement is the pilot's and the desk's, not this stage's.

A second, larger difference is **not** a rule change at all and is recorded so it is not
mistaken for one: the pilot applies its rule to seven days of points **concatenated**, and the
harvest kernel reads **one daily archive**. The same pilot rule gives 5,433 departure events
per day and 5,389 on the concatenation (46 only per-day, 2 only on the concatenation),
because a ground run that crosses midnight is one segment in one view and two in the other.
The per-day assembly is not a choice this stage made; it is the shape of the source. A third, found later, is the **sort order**: 92 of one
day's 553,178 points share an `(airport, icao, t)` and 22 of those pairs disagree about
`on_ground`, so an unstable sort splits the ground run at a different point on a different
machine. The harvest now sorts stably (exp-045, commit `afc77ee`) and the pilot's
`segments.py` is frozen and does not, so the identity test sets the tied aircraft aside and
asserts the two implementations agree row by row on the rest. Together the three account for
every row on which the block and the pilot's matcher differ — the rows only one of them
matches, the rows that choose a different segment, and the rows that carry a different
`obs_taxi` — with **zero unattributed** in all three, which `analysis/block-test.json`
asserts.

## The wide match rule, and the rule-version gate (desk, 2026-09-08)

Two instructions arrived after the block was built, and both are readings this stage had
to make concrete.

**The wide rule.** The desk: *"for LIRF, EDDF and EGLL the off-block observation is at the
stand, so a broken take-off clock need not discard the match: validate identity by
callsign-date-airport plus the first airborne point or the take-off inside a wider window,
keep `obs_offblock`, and carry the clock-check status as a quality feature so the fold
decides what those rows are worth."* My reading: the strict rule discards a match for a
reason that has nothing to do with the quantity being served — `obs_taxi` is
`MVT_TIME − obs_offblock` and never reads `obs_wheels_off` — so the clock belongs in the
evidence, not in the filter. Implemented as a **second pass** that touches only departures
the strict pass left unmatched and segments the strict pass did not claim: the same
`(airport, callsign)` join, an implied taxi inside [60 s, 7200 s], and a preference for a
take-off inside ±60 min where one exists. `match_mode` (0 strict / 1 wide window / 2 no
take-off), `clock_ok` and `has_wheels_off` are features, and `reach_class` names the mode,
so nothing downstream can treat a wide match as a strict one by accident.

One choice inside it is load-bearing and was not obvious: **candidates are drawn from every
segment carrying an `obs_offblock`, not only from segments that end in a take-off**. At
EGLL and EDDF the aircraft's ground run frequently has no airborne successor at all, so the
segment is `kind == "other"` and the strict matcher cannot see it. Drawn only from
departures, the wide rule gained 3–4 rows per airport-day; drawn from every segment it
recovers 3.4 points at EDDF and 4.0 at EGLL (`analysis/reachwhy.json`).

The measured answer is that **this is not the repair those two airports need**: what limits
them is receipt, not the matcher (§4 of `RESULTS.md`). The rule is kept because it is
cheap, because it is priced by reach class rather than assumed, and because the fold — not
this SPEC — decides what a wide row is worth. The EHAM evidence already says a wide row is
worse than a strict one (`analysis/reproduce-pilot.json`: 4 extra rows, +0.60 s of pooled
RMSE), which is exactly the kind of thing a per-reach-class price is for.

**The rule-version gate.** The desk: *"build the fold table and the serve only on summaries
carrying one and the same version."* My reading: the version is a property of a day, not of
the experiment, so the gate belongs in the loader and the answer belongs in the record.
`block.one_rule()` returns the `summaries_rule_sha256` with the widest cover over EVERY
span a run will read -- all three fold months and both ranking months resolved together,
not one month at a time (`dominant_rule` is the single-span diagnostic wrapper); `load_summaries(require_rule=…)` admits only those days and reports
the rest in `days_wrong_rule`; `arms.py` keys its block cache on the sha AND on every summary file present for the
month, so neither a re-derived month nor a half-delivered one can be served from a stale
cache, and `serve.py` refuses a fold table priced under a different sha or different
caches; and `serve.py` writes the cover into
`analysis/serve.json`. Where a summary carries `airborne_alt_ok` the repair is additionally
normalised in the loader, so a file written before the repair and one written after give
identical rows — the gate refuses a mixture, the normalisation makes most mixtures
unnecessary. Rejected reading: pinning a single sha as a constant in the code, which would
have to be edited every time the harvest re-derives a month and would silently exclude
everything on the first day of a migration.

## Status

Written as the stage runs. `RESULTS.md` carries the fold table, the served census and the review
rounds. As of 2026-09-08 18:00 UTC: step 1 delivered, tested and repaired once (the callsign
forms); step 2 measured on all three folds against v12; step 3 served under the desk's order of
18:05, uploaded as v13 and scored 248.4818 (rank 1 of 74, transfer 1.75x, unexplained -- §8);
step 4 in `docs/EXTERNAL-DATA.md`. Since then, on the desk's schedule and NOT part of the four
steps: a 28-entry fold-side push (§9), and a second candidate **v15**
(`exp-046-surface-v2.parquet`, sha `eabd824d...`) served on v14's bytes and gated on the
**increment over v13**, FALLBACK round 4 PASS with ten non-blocking findings, all repaired
(§10, §11). Two arms outside the assignment carry the attribution —
a control with no observation and a both-clocks placebo — and the arm that ships is chosen by
`serve.choose_arm()` under the desk's rule, not by hand.
