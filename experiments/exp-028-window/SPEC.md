# exp-028-window — the transient-window arm on stack-5

Assignment `assignments/window.md` (decision recorded rule). Worker `window`, tier hard, 2 threads,
laptop rules (`assignments/LAPTOP.md`: no `data/`, no `artifacts/`, committed bytes only).

---

## 1. The reading this experiment took

**The arm is a residual correction on top of stack-5 keep-noac, measured on the lane's own
three-fold rotation, and it is a candidate only if every one of the three folds moves the
right way.** Concretely:

- **Base.** `experiments/exp-026-rebuild/oof/stack5-{fold}.parquet`, column `y_pred`
  (= `stack-5 keep-noac`, the variant exp-026 §4 says is the one to serve). The base is
  *read back*, not refitted: `stack5.py` needs no data and would reproduce it, but reading the
  committed rotation makes it impossible for this experiment and exp-026 to disagree about what
  stack-5 is. `code/panel.py` gates on it — the three clean fold RMSEs must equal
  `experiments/exp-026-rebuild/analysis/stack5.json` to 1e-3 or the run aborts.
- **The arm.** One LightGBM per scored fold, fitted on the other two folds' stack-5 residual
  (`y_pred − y_true`, winsorised to ±1200 s), predicting an adjustment subtracted from
  stack-5. Features: aggregates of `d_takeoff_aobt` and `d_aobt_lobt`, the regime mix and the
  movement count over the departure's own airport-window and the two windows either side, plus
  the row's own two deltas, hour, weekday, airport and the base prediction.
- **The gate.** `both-folds`, extended to three: an arm that loses on any of `july_loo`,
  `december`, `january` is a FAIL of that arm, whatever the board estimate says.
- **The candidate is pre-registered, the screen is sensitivity.** The configuration carried
  forward is the one the K1 probe already chose on blend-5 (600 s window, 400 rounds, 31
  leaves), fixed before any stack-5 number was computed. The width / rounds / leaves /
  fit-population table in RESULTS §4 is reported as sensitivity around that point, not as a
  selection over three folds — selecting a configuration on the same three folds that carry
  the scored row would put the selection's optimism inside the number.

## 2. The readings rejected

1. **Refit stack-5 rather than read it back.** Rejected: `stack5.py` is 15 LightGBM trainings
   and reproduces, but any difference between a refit and the committed rotation (LightGBM
   4.7.0 vs the bytes on disk, thread count, an accidental argument) would silently move the
   base under the arm's delta. Reading the rotation back and gating on the published numbers
   is strictly stronger for what this experiment claims.
2. **Fit the arm on blend-5's residual and add it to stack-5.** Rejected: the arm's target
   must be the error of the vector it will be added to. blend-5 and stack-5 differ by 1.1 s of
   clean board and the difference is concentrated exactly where the arm looks (`nm_bulk`), so
   an arm fitted on blend-5's residual would re-correct what stack-5 already corrected. The
   K1 probe's numbers are therefore *not* the prediction for this experiment.
3. **Claim the rotation is nested.** Rejected on the desk's instruction of 2026-09-07 (§3).
4. **Screen widely and report the best cell.** Rejected — see §1, last bullet.
5. **Serve stack-5 from this experiment.** Rejected: `experiments/exp-026-rebuild/code/serve_stack5.py`
   is the blend lane's own serving path and is the only place stack-5's served vector should
   be built. `serve.py` here consumes that vector and adds the arm; it does not rebuild it.

## 3. The rotation's channel, stated rather than claimed away

The desk relayed Sol's finding on 2026-09-07: **the lane's three-fold rotation has an indirect
channel.** A fold's labels reach a stacker through the other folds' *member* out-of-fold files,
whose underlying models were trained on month sets that include the scored month. This is the
shared instrument of every fold number in this repo, including the board-calibrated stack-4,
and it is not closable from committed bytes — refitting the stacker leave-two-out would close
one narrow path (exp-026 §4 measures that one at 0.011 s) and leave the member channel open.

**Consequences adopted here, in place of a nesting claim:**

- No number in this experiment is described as nested. The arm's delta is reported **against
  stack-5 on the same instrument**: base and arm are scored on identical rows with identical
  member OOF behind them, which makes the comparison like-for-like.
  **Corrected after Sol round 1** (Spec 3): the first version of this line said the channel
  "cancels out" of that comparison, and that is more than the construction supports. Adding
  features changes how the stacker uses the member columns, so it can change how much of the
  channel the fit exploits; a common instrument makes the two sides comparable, it does not
  make the channel disappear from the difference. What is claimed is an **instrument-relative
  delta**, and the two controls below are what bound the instrument's free gain.
- **Two controls run beside every arm**, so the instrument's free gain is measured rather than
  assumed:
  - `row` — the identical fit with every window aggregate deleted (own deltas, hour, weekday,
    airport, base prediction only). A gain here is re-fitting the head on its own row columns.
  - `shuffle` — the identical fit with the window-aggregate block permuted **within
    airport-day**, so the marginal distribution of every window column and the airport-day
    level are preserved and only the assignment of a window state to a departure is destroyed.
    A gain here is an airport-day level effect or a free gain of the instrument, not the
    transient window.
  The existing evidence that this instrument does not hand out gains for free is K1's `row`
  control on blend-5: −1.540 s clean, −1.279 s no-noac (`experiments/exp-027-step/` §8d).
- The stacker-level leave-two-out variant is **not** run. It would cost about two minutes and
  bound one path of a channel whose other path stays open, which is a number that invites the
  nesting claim §2.3 rejects.

## 4. Leakage contract

Every column the arm reads must be computable, for a ranking row, from the ranking file
(`docs/input-rule.md`: every column present in `ranking.parquet` is legal; the blanked
departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are not).

| feature | built from | ranking-file column | legal |
|---|---|---|---|
| `t` (bucket, hour, weekday) | `MVT_TIME_UTC_mvt` | present, 100 % populated | yes |
| `own_ta`, `w_ta_*` | `MVT_TIME_UTC_mvt − AOBT_3_flt` | both present | yes |
| `own_al`, `w_al_*` | `AOBT_3_flt − LOBT_flt` | both present | yes |
| `w_late` | regime from `AOBT_3_flt − LOBT_flt` | present | yes |
| `w_nonm` | `FLIGHT_ID_mvt` null | present | yes |
| `w_n` | movement count at airport × bucket | row existence | yes |
| `apt` | airport | present | yes |
| base prediction | stack-5's served vector | model output | yes |

**Not read, deliberately:** the departure's own `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`
(blanked), and **any other departure's** block time or taxi time — the window aggregates are
built from take-off, NM off-block and last-received-EOBT only, never from a departure's block
or taxi time, because those are blanked for every departure in the ranking file, not just the
row being predicted. Arrivals' `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` **are** legal
(`docs/input-rule.md`: "Arrival rows carry full actuals including taxi-in time") and are the
first thing K2 should add; they are **not** in this experiment because no committed
out-of-fold file carries arrival rows at all (RESULTS §6). No label of any row enters a
feature; the target is stack-5's own residual and it exists only inside the fit.

## 5. Deliverables and where they are

`code/panel.py` (base + canonical panel + the reproduction gate), `code/features.py` (the one
window-feature function used by both the fold path and the serving path, plus the two panel
adapters), `code/arm.py` (rotation, controls, screen, intervals), `serve.py`, `tests/shape_check.py`
(the train/serve equivalence test, run without data), `oof/exp-028-window-{fold}.parquet`,
`climb.log`, `RESULTS.md`, one row per measured arm in `docs/LEDGER.md`, one Sol review under
`review/`.

---

## 6. Addendum, 2026-09-07 09:20 UTC — the construction changed, and why

Written after the first rotated numbers, before the second-layer arm was abandoned.
The readings above stand; this replaces §1's second bullet.

**The measured fact.** The assignment's construction — a LightGBM on stack-5's residual
fed window aggregates — is worth **+0.002 s** of clean board on stack-5 and fails the
three-fold gate, against **+0.591 s** for the identical code on blend-5's residual
(RESULTS §2–3). Its controls are large and negative on both bases (`row` −2.25 s on
stack-5), so a second residual layer on a residual stacker pays a refit cost that the
window columns only buy back.

**What did not cause it.** stack-5 has not consumed the component: the airport × 600 s
oracle is 14.62 s on stack-5 against 15.11 s on blend-5 (corrected after Sol round 1;
the first pass read 15.25 and 15.67), and the error correlation
between departures under two minutes apart is 0.168 against 0.170 (RESULTS §4). The
transient is intact; what changed is that a second layer cannot reach it.

**The construction adopted instead**, on the desk's path hint of 2026-09-07 09:10 UTC:
the window aggregates go **inside stack-5's own stacker**, beside the seven member
columns, in one layer — same rotation, same rotated blend-5 anchor, same five seeds,
same 300 rounds, same LightGBM parameters, the no-window variant reproducing exp-026's
published rotation in the same process. `code/stack5w.py`. This is the candidate.

**Two things this addendum adds to §4's contract.** `data/raw` landed at 09:05 UTC, so
(1) the arrivals' unmasked `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` window
aggregates, which no committed out-of-fold file carries, are now buildable and are
measured (`code/arrivals.py`); they are legal because arrival rows are not blanked. And
(2) the airport-level baselines the deviation block subtracts are computed **inside the
month being predicted**, from the same four ranking-file quantities, so they cross no
month boundary and read no label.

> **Two corrections after Sol round 1.** On (1): the two columns are one measurement,
> not two. On an arrival row `BLOCK_TIME_UTC_mvt − MVT_TIME_UTC_mvt` **is**
> `TAXITIME_SEC_mvt`, exactly, on every row of all three fold months and of the ranking
> file (`analysis/arrival_block_is_taxi.json`, max difference 0.000 s), so the taxi-in
> aggregate that was measured answers both halves and a separate block-time aggregate
> would be a duplicate column. On (2): the deviation baselines are per month **on the
> fold side only**. The serving side would have had to compute them over January and
> July 2026 together, which is a different feature; `stack5w.ranking_blocks` now refuses
> to serve the deviation block instead of silently doing that. The deviation form
> measured +0.061 s of board and was not carried, so nothing shipped depends on it.

**What is not this experiment's lane.** K2 proper — the window aggregates inside a
member's own feature frame, refit (`F1-fresh`, about 19 minutes per fold on Kaggle) — is
the `step` worker's. Recorded here so the two are not confused: this experiment moves
the stacker, not a member.


---

## 7. Addendum, 2026-09-07 09:40 UTC — the shipping width, and what selecting it costs

The 600 s window is the **pre-registered** configuration of §1: it is K1's, fixed before
any stack-5 number existed. The width screen the assignment asked for (300 / 600 / 900 s)
was then run inside the stacker, and 300 s is the variant that passes the three-fold gate
(RESULTS §7b). On the desk's instruction of 09:40 UTC the 300 s form ships, with the cost
of that choice stated rather than absorbed:

**The width was chosen on the same three folds the variant is scored on.** Taking the
best of three widths buys roughly the spread between them — about 0.05 s of clean board
and 0.07 s on december — so the 300 s figure of +0.407 s is optimistic by about that
much, and the honest pair to quote is:

| form | july_loo | december | january | clean board | three folds |
|---|---|---|---|---|---|
| 300 s, selected on these folds (shipped) | +0.484 | +0.027 | +0.289 | +0.407 | yes |
| **600 s, pre-registered** | +0.481 | −0.042 | +0.314 | **+0.414** | MAYBE |

Both are staged. Neither resolves december: its sign flips between the two widths for a
swing smaller than the rotation's noise floor.
