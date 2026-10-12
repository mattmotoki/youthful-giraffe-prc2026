# exp-031-nested — the rotation recompute on the nested instrument

Reading of the assignment and the readings rejected: `SPEC.md`. This file is
the version after Sol round 1 (`docs/reviews/nested-round1.md`), whose one
blocking finding is answered by measurement, not by wording: §7.

Artifacts: `analysis/rotation.json` and `analysis/rotation-nestedanchor.json`
(every number), `analysis/tables.md` and `analysis/tables-nestedanchor.md` (the
full candidate x fold x column tables), `analysis/anchor.json`,
`analysis/slices.json`, `oof/exp-031*.parquet`, `logs/`.
Producer: `PRC_THREADS=4 python code/rotate.py [--anchor nested]`, 529 s and
525 s wall, LightGBM 4.7.0.

## 0. The answer, in four lines

**No gain we ship on reverses its sign.** The five candidates keep their order
on all eight instruments measured here — four member columns x two anchor
constructions — and every board gain that was clear of zero on the published
instrument is still clear of zero on the instrument with the anchor channel
closed and the member channel closed **for three of the seven members**. That
partiality is the standing limit on everything below and is not repeated at
every number; §1 and §11 state it.

**The K2-versus-refit-bulk ordering holds and was never significant.**
`refit-bulk` is ahead of `K2` on all eight instruments, by 0.036 to 0.122 s of
board, and on all eight the interval crosses zero — the published one included.
The change of instrument does not change what was already true; it halves the
gap, from 0.105 s to 0.045 s (both channels closed, three members: 0.036 s).

**The member-level swings do not reach the board.** Members moved +0.39, +1.11
and +1.25 s on average under nesting, with a 3.86 s move on one member-fold
(`exp-025-cloud` §10.8, same-environment basis). On the same basis each
candidate's board moves **+0.21 to +0.32 s** and the *differences between
candidates* move by at most **0.1119 s**. That answers §10.8's first caution:
member shifts of several seconds do largely cancel in the weighted blend.

**Neither channel is worth much at candidate level.** Closing the anchor channel
as well moves a board level by less than 0.043 s and a gain by less than 0.028 s
(exactly: 0.0424 and 0.0274), which is consistent with exp-026's own 0.011 s
reading of it on stack-5.

## 1. What this instrument is, and what it is not

`exp-026-rebuild/RESULTS.md` §4a names two label channels in the fold rotation.

- **Channel B, the members'.** The stacker's feature columns are the members'
  own out-of-fold predictions, from models fitted on month sets that include the
  scored fold. Closed here for **three of seven members** by substituting
  `exp-025-cloud/oof-nested/` (`instrument.py`); `V3-allin` cannot be refit at
  all (§10.3, no ERA5 snapshot) and `C2-cat`, `D1-winter`, `N1-nmtail` and
  `K2-window` were never in exp-025's scope.
- **Channel A, the anchor's.** Scoring fold `f`, the stacker's *training* anchor
  on fold `g` is the blend fitted on the other two folds, and "the other two"
  includes `f`. Closed by `rotate.py --anchor nested`, with
  `exp-026/code/nested_anchor.py`'s construction: fold `g`'s training anchor is
  fitted on the single remaining fold.

**These numbers are not the clean score, and they are not a bound on it.** The
first draft of this file called the nested rotation an upper bound; Sol round 1
finding 11 refuted that and is right. The design confounds closing a channel
with removing one month of training data, and the measured member shifts include
negative values (`L2-hub8725`, `july_loo_nx_dec`, −2.55 s), so nothing here is
monotone in the direction of the clean score. What the numbers are: **five
candidates measured against each other on one instrument at a time**, which is
what every decision they feed needs.

**Do not add +4.28 to any column but `published`.** §10.5 states it before these
numbers existed: the offset was calibrated on the unnested instrument, so a
nested fold number carries it wrong by exactly the amount the instrument moved.

## 2. The gates

Run before any arm, and all abort. Under `--anchor rotated`:

| gate | what it checks | result |
|---|---|---|
| pre | `tests/direction.py`, called by `rotate.py` itself | pass, §8 |
| A | blend-5's all-rows rotation on `published` reproduces `blend-config-5.json` (286.3274 / 207.0083 / 339.0173) | 0.0000 on all three folds |
| B–F | the five candidates on `published` reproduce their published triples | 0.0000 on all fifteen |
| vector | `stack-5` and `v6` reproduce exp-028's committed vectors row by row | max abs 1.14e-13 |

| arm | july_loo | december | january | source of the published triple |
|---|---:|---:|---:|---|
| stack-5 | 285.2029 | 205.6118 | 219.1303 | exp-026 `analysis/stack5.json` |
| v6 | 284.7219 | 205.6534 | 218.8166 | exp-028 §4 |
| v7 | 284.6251 | 205.5677 | 218.5412 | exp-027 `analysis/refit-bulk.json` `v7_clean` |
| K2 | 284.3195 | 205.2156 | 218.2582 | exp-027 `analysis/candidate-k2.json` `k2-hold-eham-out` |
| refit-bulk | 284.1774 | 204.9052 | 218.2115 | exp-027 `analysis/refit-bulk.json` |

Under `--anchor nested` the published triples are deliberately **not** gated and
`rotate.py` says so in the log: closing channel A changes the rotation the
shipped numbers were measured on, so landing on them would be a defect.

## 3. Four member columns, because two would confound the machine

`exp-025-cloud` §10.6 and the `oof-nested/` README: a shift taken against the
published workstation files is **mostly the machine** — ±0.3 s for `F1-fresh`
and `L2-hub8725`, sign changing by fold, three times L11's 0.1025 s rotation
floor. At the size of the differences this assignment weighs (+0.29 and +0.40 s
of board) that confound is the same size as the answer.

| column | the three refittable members read from |
|---|---|
| `published` | the committed workstation OOF (the record) |
| `kaggle` | `oof-nested/<member>--<fold>` — the ordinary folds, re-run on Kaggle |
| `nested` | the scored fold from its Kaggle baseline, the two **fitting** folds from `<g>_nx_<mon>` |
| `nested-f1held` | `nested` with `F1-fresh` held at its Kaggle baseline |

`published → kaggle` is the machine term; `kaggle → nested` is the nesting term.

`nested-f1held` exists because `K2-window` has no nested refit. On the `nested`
column `v7` carries three nested members as stacker features and `K2` only two,
so `K2 − v7` there would carry one member's change of instrument. On
`nested-f1held` both carry the same two and the difference is theirs.
**A precision Sol round 1 finding 9 is right to ask for:** "two members nested"
counts *direct stacker features*. On the `nested` column `K2` still sees a
nested `F1-fresh` through the blend-5 anchor it is anchored on, because that
anchor is fitted over `NAMES5`. `nested-f1held` removes that too, which is why
it, and not `nested`, is the symmetric read of `K2 − v7`.

`K2` and `refit-bulk` share a member set, so the ordering question is unaffected
by any of this and is read on every column.

## 4. Candidate x fold, clean RMSE

Fold order july_loo / december / january; the arrow is the clean board
(`frames.board_estimate`). Full per-fold tables with intervals:
`analysis/tables.md` and `analysis/tables-nestedanchor.md`.

**Channel B closed for three of seven members (`--anchor rotated`, the shipped
anchor, so channel A open):**

| candidate | published | kaggle | nested (3) | nested (2) |
|---|---|---|---|---|
| stack-5 | 285.2029 / 205.6118 / 219.1303 → **258.0372** | 285.2966 / 205.7453 / 218.8464 → 257.9882 | 285.8168 / 205.7628 / 218.8548 → **258.3119** | 285.8095 / 205.7327 / 218.9498 → 258.3431 |
| v6 | 284.7219 / 205.6534 / 218.8166 → **257.6230** | 284.8145 / 205.7823 / 218.5840 → 257.5926 | 285.1602 / 205.6201 / 218.5809 → **257.8044** | 285.2847 / 205.6316 / 218.6202 → 257.8959 |
| v7 | 284.6251 / 205.5677 / 218.5412 → **257.4598** | 284.7202 / 205.6994 / 218.3004 → 257.4279 | 285.0890 / 205.5407 / 218.3297 → **257.6662** | 285.1905 / 205.5498 / 218.3649 → 257.7420 |
| K2 | 284.3195 / 205.2156 / 218.2582 → **257.1653** | 284.3970 / 205.2905 / 218.0581 → 257.1378 | 284.7455 / 205.1466 / 218.1092 → **257.3717** | 284.8786 / 205.1999 / 218.1689 → 257.4762 |
| refit-bulk | 284.1774 / 204.9052 / 218.2115 → **257.0601** | 284.2301 / 204.9073 / 218.1473 → 257.0685 | 284.6339 / 204.7908 / 218.1738 → **257.3273** | 284.7399 / 204.7673 / 218.2383 → 257.4167 |

**Channel A closed as well (`--anchor nested`); channel B is still three of
seven members:**

| candidate | published | kaggle | nested (3) | nested (2) |
|---|---|---|---|---|
| stack-5 | 285.2834 / 205.4460 / 219.0260 → 258.0476 | 285.3732 / 205.5918 / 218.7985 → 258.0173 | 285.8739 / 205.6663 / 218.7960 → **258.3251** | 285.9061 / 205.6537 / 218.8509 → 258.3655 |
| v6 | 284.7696 / 205.3597 / 218.8247 → 257.6554 | 284.8271 / 205.5136 / 218.6201 → 257.6139 | 285.1529 / 205.4025 / 218.6010 → **257.8075** | 285.2996 / 205.4490 / 218.5915 → 257.8943 |
| v7 | 284.6747 / 205.2630 / 218.5204 → 257.4826 | 284.7331 / 205.4162 / 218.3230 → 257.4444 | 285.0883 / 205.3225 / 218.3366 → **257.6684** | 285.2113 / 205.3595 / 218.3236 → 257.7393 |
| K2 | 284.3373 / 204.8968 / 218.3178 → 257.1986 | 284.4183 / 204.9791 / 218.1361 → 257.1802 | 284.7576 / 204.8966 / 218.1185 → **257.3827** | 284.9093 / 204.9307 / 218.1699 → 257.4955 |
| refit-bulk | 284.1889 / 204.6033 / 218.2359 → 257.0764 | 284.2540 / 204.6280 / 218.1999 → 257.1029 | 284.6720 / 204.5618 / 218.1640 → **257.3470** | 284.7465 / 204.5259 / 218.2056 → 257.4086 |

## 5. The chain, on eight instruments

Positive means the newer candidate is better. Board 95 % is a day-clustered
bootstrap over both ranking months, each resampled with its own day x airport
clusters and combined in quadrature (`code/rotate.boot_board`, §9); the per-fold
intervals in `analysis/tables*.md` are `stack5.boot_delta`, exp-028's own.

| gain | pub A-open | kag | nest(3) | nest(2) | pub A-closed | kag | nest(3) | nest(2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v6 − stack-5 | +0.4142 | +0.3956 | +0.5076 | +0.4472 | +0.3921 | +0.4034 | +0.5176 | +0.4712 |
| v7 − v6 | +0.1632 | +0.1647 | +0.1382 | +0.1539 | +0.1728 | +0.1696 | +0.1391 | +0.1550 |
| K2 − v7 | +0.2945 | +0.2901 | +0.2945 | +0.2658 | +0.2840 | +0.2642 | +0.2857 | +0.2438 |
| refit-bulk − v7 | +0.3997 | +0.3594 | +0.3390 | +0.3253 | +0.4062 | +0.3414 | +0.3213 | +0.3307 |
| K2 − refit-bulk | −0.1051 | −0.0693 | −0.0445 | −0.0595 | −0.1222 | −0.0773 | −0.0357 | −0.0869 |

Every one of the first four is positive in all eight columns and its board
interval excludes zero in all eight. The board 95 % on `nested (3)`, both
channels closed: `v6 − stack-5` [+0.232, +0.824], `v7 − v6` [+0.086, +0.205],
`K2 − v7` [+0.184, +0.392], `refit-bulk − v7` [+0.079, +0.599].

Channel B split into its two terms (`--anchor rotated`):

| gain | machine (kaggle − published) | nesting (nested − kaggle) |
|---|---:|---:|
| v6 − stack-5 | −0.0186 | +0.1119 |
| v7 − v6 | +0.0015 | −0.0265 |
| K2 − v7 | −0.0044 | +0.0043 |
| refit-bulk − v7 | −0.0402 | −0.0205 |
| K2 − refit-bulk | +0.0358 | +0.0248 |

**Three of the five barely move.** `K2 − v7` is +0.2945 on both the published
and the `nested` column, its two terms −0.0044 and +0.0043 — both an order below
the 0.1025 s floor. `v7 − v6` loses 0.027 s. `v6 − stack-5` *grows* by 0.1119 s:
the window block buys more, not less, once the members cannot see the scored
month. That is the direction one would expect if the block were carrying real
information rather than a leak — but it is not proof of it, because nesting and
the loss of one training month are confounded here (Sol round 1, finding 11).

**The `kaggle` column corroborates exp-027's environment control on all three
folds.** `K2-window` is a Kaggle-fitted member sitting in a slot whose incumbent
`F1-fresh` is a workstation file, so part of `K2 − v7` could be the machine.
exp-027's `analysis/envctl.txt` priced that on `july_loo` alone, by refitting
`F1-fresh`'s own slot on Kaggle: **+0.0068 s of apparent gain**. The `kaggle`
column does the same on all three folds at once and gets **+0.0044 s of board**.
Two designs, the same sign and the same order of magnitude: the machine adds a
twentieth of the K2 gain and hides none of it.

**`refit-bulk` is the one that pays.** Its gain over v7 falls from +0.3997 to
+0.3390 (nested, three members), and **two thirds of the fall is the machine,
not the nesting** (−0.0402 against −0.0205). Where it pays is january: +0.3297
published, +0.1530 kaggle, +0.1558 nested, and the january interval stops being
clear of zero (published [+0.004, +0.639], nested [−0.180, +0.477]). So
`refit-bulk`'s published +0.3997 — clear of zero on all three folds — reads on
the nested instrument as **+0.3390 with january no longer clear of zero**. The
board claim is unaffected; the three-fold claim weakens. (The assignment's
"clear of zero on no fold" is about `K2 − refit-bulk`, §6, not about this gain.)

## 6. The ordering question

`K2 − refit-bulk`, the quantity that decides which of tomorrow's two candidates
goes first. Negative means `refit-bulk` is ahead.

| column | anchor | july_loo | december | january | board | board 95 % |
|---|---|---:|---:|---:|---:|---|
| published | A open | −0.1422 | −0.3105 | −0.0467 | **−0.1051** | [−0.336, +0.095] |
| kaggle | A open | −0.1669 | −0.3832 | +0.0892 | −0.0693 | [−0.286, +0.122] |
| nested (3) | A open | −0.1115 | −0.3558 | +0.0646 | **−0.0445** | [−0.264, +0.146] |
| nested (2) | A open | −0.1387 | −0.4327 | +0.0694 | −0.0595 | [−0.282, +0.139] |
| published | A closed | −0.1484 | −0.2935 | −0.0819 | −0.1222 | [−0.342, +0.072] |
| kaggle | A closed | −0.1643 | −0.3511 | +0.0638 | −0.0773 | [−0.285, +0.111] |
| nested (3) | A closed | −0.0856 | −0.3348 | +0.0456 | **−0.0357** | [−0.243, +0.151] |
| nested (2) | A closed | −0.1627 | −0.4048 | +0.0357 | −0.0869 | [−0.308, +0.108] |

**`refit-bulk` is ahead on all eight, and the interval crosses zero on all
eight.** The assignment's own phrasing ("clear of zero on no fold") was already
the honest reading of the published number; the change of instrument does not
change it. What it does is halve the gap: 0.105 s published to 0.045 s nested
with channel A open, 0.036 s with it closed.

Two of the three folds are stable under the change of instrument; january flips
sign (−0.047 published to +0.065 nested, channel A open), and the flip is the
machine term (+0.136) against the nesting's (−0.025), on a total move of +0.111.

## 7. Channel A, and what closing it costs — Sol round 1's blocking finding

Sol round 1 finding 1: member substitution closes part of channel B and leaves
channel A open, so the first draft's "clean instrument" was an overclaim. That
is correct, and the answer is a second run rather than a caveat:
`rotate.py --anchor nested` closes it and the whole table above is measured
twice.

| quantity, on `nested` (3 members) | A open | A closed | closing A costs |
|---|---:|---:|---:|
| stack-5 board | 258.3119 | 258.3251 | +0.0131 |
| v6 board | 257.8044 | 257.8075 | +0.0031 |
| v7 board | 257.6662 | 257.6684 | +0.0022 |
| K2 board | 257.3717 | 257.3827 | +0.0110 |
| refit-bulk board | 257.3273 | 257.3470 | +0.0198 |
| gain K2 − v7 | +0.2945 | +0.2857 | −0.0088 |
| gain refit-bulk − v7 | +0.3390 | +0.3213 | −0.0176 |
| gain K2 − refit-bulk | −0.0445 | −0.0357 | +0.0088 |

**Closing channel A moves a board level by less than 0.043 s and a gain by less
than 0.028 s** across all four columns — exactly 0.0424 s (K2 on `kaggle`) and
0.0274 s (`K2 − refit-bulk` on `nested-f1held`) — and on `nested` by at most
0.0198 s. That
agrees with exp-026's own reading of channel A on stack-5, 0.011 s of board.
**No sign changes, on any gain, on any column.**

`nested_anchor.py`'s caution carries: the nested anchor is also a **weaker**
anchor — one fold of fitting rows instead of two — so a move under it is two
effects at once, and the small numbers above are the sum of both.

## 8. Two tests no RMSE gate can make, and `rotate.py` runs one of them

The gate in §2 runs on the `published` column, where nothing is substituted, so
it proves the harness and not the substitution: it would pass after a wrong
nested tag or a reversed direction (Sol round 1, finding 4).

**`tests/direction.py` — called by `rotate.py` before any number.** If the
substitution were the other way round — the scored fold nested and the two
fitting folds not — the channel would stay open and every number here would be
wrong while every gate passed. The test checks the direction against
`prc.data.NESTED_FOLDS` (all six nested folds used, each in the slot and the
rotation it declares) and then on the frames: **the scored fold holds the
member's ordinary Kaggle baseline, both fitting folds are substituted**, on all
three members and all three rotations, and the four unnested members are
byte-identical to the published column.

**`tests/swap_equivalence.py` — run separately, 12 cases.** `instrument._swap`
replaces a prepared member column rather than rebuilding the frame from the
substituted address. The test builds the second route explicitly — each
`oof-nested/` file rewritten to a temporary parquet whose `fold` column carries
the ordinary name, `frames.OOF` pointed at it, `frames.fold_frame` run unchanged
— and compares every column of every frame. **Identical, bit for bit, on all
twelve (column, scored fold) cases**, with three preconditions checked first: no
duplicated `MVT_ID_mvt` in any of the 27 files, and every file's id set *and row
order* equal to the fold frame's.

## 9. Why the member swings do not reach the board

`exp-025-cloud` §10.8 left this open: "member shifts of several seconds can
cancel in a weighted blend, and nothing here says whether they do." They do.
All three rows below are on the **same** basis, `kaggle → nested`, the
same-environment nesting-only move §10.8 itself uses (Sol round 1, finding 6):

| level | largest single move | mean move |
|---|---:|---:|
| member, per fold (§10.8) | 3.862 s (`L2-hub8725`, `july_loo_nx_jan`) | +0.39 / +1.11 / +1.25 s per member |
| candidate board | +0.3237 s (stack-5) | +0.2533 s |
| gain between candidates | 0.1119 s (v6 − stack-5) | 0.0376 s in absolute value |

Three reasons, in order of size. The blend is a rotated NNLS over seven members
of which three move, so a member's shift enters weighted and partly offset by
the others; the stacker fits the residual of that anchor and is fitted on the
same moved columns, which absorbs more; and the two members that move most
(`T-tailv4`, `L2-hub8725`) carry less anchor weight than `F1-fresh`, which moves
least. The `noac` rows are held at blend-5's own apply-time anchor in every arm
**within a column**, so they are identical across the five candidates there and
cancel out of every difference this experiment reports. They are **not** common
across columns — the anchor is refitted on the substituted members, and against
`published` all 339 / 89 / 63 `noac` rows move, by up to 594 / 432 / 244 s
(Sol round 2, finding 7). That is why every gain here is read down a column and
never across one.

**The board interval, and what it is conditional on.** `rotate.boot_board`
resamples day x airport clusters inside each ranking month independently — the
two calendars are disjoint — pairs the new and reference errors within a
cluster, and combines the per-draw month RMSEs by the fixed row-count shares. It
is conditional on the fitted vectors: it carries neither model-fit nor
cross-month airport uncertainty. Sol round 1 finding 7 confirmed the
construction and that its point estimate matches the table.

**One arithmetic defect, found by Sol and fixed.** The first run computed each
board from three fold RMSEs already rounded to four decimals, which moved four
of the twenty board gains in the last digit (`K2 − v7` published read +0.2946
where the exact value is +0.29455). `rotate.py` now keeps the fold RMSE at full
precision, computes every board and every gain from it, and rounds once on the
way into the JSON, which also carries `clean_exact` and `board_exact`.

## 10. The stacker's edge over its own anchor, which the backlog asks about

`docs/BACKLOG.md`'s nested-instrument row names one thing the instrument would
"tell us": whether "the stacker's −1.10 s edge over the blend is real or
inflated". The anchor is fitted and gated inside every rotation above and then
thrown away; `code/anchor.py` keeps it. It is not a new arm — it is the arm
every arm above is anchored on — and it is NNLS only, so it costs seconds.

blend-5's rotation is the shipped one on every column: it has no stacker, so
exp-026 §4a's channel A does not arise inside it. The stacker side of the edge
does have one, so the edge is quoted against both stack-5 constructions
(Sol round 2, finding 6).

| column | blend-5 (clean board) | edge over stack-5, A open | edge over stack-5, A closed |
|---|---:|---:|---:|
| published | 259.1339 | **+1.0967** | +1.0864 |
| kaggle | 259.1398 | +1.1516 | +1.1225 |
| nested (3 members) | 259.4181 | **+1.1062** | +1.0931 |
| nested (2 members) | 259.4240 | +1.0809 | +1.0585 |

Gate: blend-5's all-rows rotation on `published` reproduces
`blend-config-5.json` on all three folds to 0.0000 (286.3274 / 207.0083 /
339.0173, `stack5.PUBLISHED`).

**Nesting these three members does not reduce the stacker's edge.** +1.0967 s on
the instrument every number in the ledger is quoted on, +1.1062 s with the
member channel closed for three of seven — a move of +0.010 s, two orders below
the edge and one below the 0.1025 s rotation floor. With channel A closed as
well the same comparison is +1.0864 → +1.0931. blend-5's board rises +0.284 s
under nesting and stack-5's +0.275 s, so the two move together.

**That is weaker than "the edge is not inflated", and the difference matters.**
Four of seven members are still unnested, `V3-allin` among them; what is
measured is that closing the channel for the three that could be refit leaves
the edge where it was. Artifact: `analysis/anchor.json`.

## 11. What this does not settle

- **Four of seven members are unnested on every column**, `V3-allin` among them,
  and it is the member the other three bracket least well (§10.3). Its
  contribution to the residual channel is unmeasured, and §9 says the board sees
  a fraction of a member move — but that fraction was measured on three members
  and not on this one.
- **The two terms are confounded by design.** Every nested column drops a month
  of training data at the same time as it closes the channel, and the design
  cannot separate them (§10.4). A gain that grows under nesting is consistent
  with a real feature and does not prove one.
- **Seed-set variation is not measured.** Every column uses the same five seeds
  and 300 rounds, so the columns are paired and their difference is the member
  columns and the anchor construction alone; a different seed set would move
  each level by an amount this experiment does not quote.
- **The board offset.** No board estimate is derived from any nested number
  (§10.5). Tomorrow's predictions stay on the published column.

## 12. What it means for tomorrow's queue

Nothing in `docs/SUBMISSIONS.md` changes. `refit-bulk` stays ahead of `K2` on
all eight instruments; both stay clear of zero against v7 on the board on all
eight; and `K2 − refit-bulk` never clears zero on any of them, which is a reason
to hold the queue as it is rather than a new reason to reorder it. The one line
that weakens is `refit-bulk`'s three-fold claim: on the nested instrument its
january gain is +0.156 s with an interval that includes zero.

## 13. The review record

Two Sol rounds, both at xhigh through Codex CLI, fresh context each
(`docs/reviews/nested-round1.md`, `nested-round2.md`, raw transcripts beside
them).

**Round 1: BLOCKED, one blocking finding, ten non-blocking.** The blocking one
was right and is answered by measurement: the rotation has a second label
channel, the stacker's training anchor, and a table measured with it open cannot
be called a clean-instrument result. `--anchor nested` closes it and the whole
four-column table is measured twice (§7). Of the ten non-blocking findings, four
changed a number or a claim and all four were taken: the board was being
computed from rounded fold RMSEs (§9), §9's three rows mixed two comparison
bases, "two members nested" needed to say *direct stacker features* (§3), and
"an upper bound on the clean score" and "a genuine feature" were overclaims and
are gone (§1, §5). Round 1 independently confirmed the substitution direction
against `prc.data.NESTED_FOLDS`, the column swap on all 27 files, the five
published triples against their sources, and the board bootstrap's construction.

**Round 2: UNBLOCKED.** It traced `_src` over all nine (scored, applied) pairs
and confirmed no anchor touches the scored fold's labels under `--anchor
nested`; replayed all five arms row by row against exp-027's and exp-028's
committed vectors (max 2.27e-13); and confirmed the ordering, the eight-way sign
check and the interval statements. Seven further corrections, all taken:

| round 2 finding | what changed |
|---|---|
| 3 | `anchor.py` computed its board from rounded folds; fixed, and §6's and §7's tables now carry the exact per-fold and cost values (−0.1422, −0.3105, −0.1115; +0.0131, +0.0198, −0.0176) |
| 4 | "at most 0.042 / 0.027" rounded the maxima downward; now "less than 0.043 / 0.028", with the exact 0.0424 and 0.0274 quoted |
| 5 | "both label channels closed" overstated a member channel that is closed for three of seven; qualified in §0, §4, the ledger and `oof/README.md`. SPEC still carried the withdrawn "upper bound" line; removed |
| 5 | ledger arithmetic: january 0.3296/0.1558, july −0.1115, `anchor.py`'s values |
| 6 | `anchor.py` measured only the shipped anchor, so §10's claim is narrowed to "nesting these three members does not reduce the edge", and the edge is now also quoted against the channel-A-closed stack-5 (+1.0864 → +1.0931) |
| 7 | "clear of zero on no fold" was attached to `refit-bulk − v7`; it belongs to `K2 − refit-bulk` |
| 7 | "the `noac` rows are common to the whole table" was false across columns — all 339 / 89 / 63 move, by up to 594 / 432 / 244 s. They are common **within** a column, which is what makes the differences cancel (§9) |
| 7 | `slices.py` would have silently overwritten its own record from a trimmed `oof/`; it now merges and refuses to drop a recorded family. `rotate.py`'s comment claimed both substitution tests ran there; only `direction.py` does |
