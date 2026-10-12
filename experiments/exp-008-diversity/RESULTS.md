# exp-008 diversity — results

Worker `diversity` (hard), assignment `assignments/diversity.md`, spec `SPEC.md`.
All numbers are RMSE in seconds on `prc.data.fold_frames`, four folds
(`july`, `july_loo`, `december`, `january`), with the COMMON.md decomposition:
overall / target ≤ 3600 s / target > 3600 s / no-NM rows, plus noac.

Frame: retro-v1's frozen v2, verified before any model was fitted —
`frame_contract.py v2` printed schema `cf956e1b26b3c11f` and numeric
`605fe71f984628d5` on 2025-07 at 08:30 on 2026-09-05, matching the contract
retro-v1 published. `frames.verify_mask_invariance` shows the ranking-masked
validation month and the unmasked training month give bit-identical columns
(54/54, no differing column, labels equal), which is why the month cache under
`artifacts/exp-008/frames/` stores one build per month.

## Finding 1 — the two existing lineages disagree less than a blend needs

The lane's premise is that structurally different learners disagree. Measured on
the residuals of the two lineages that already exist — exp-001's guarded
baseline `H0-s7` and `prc.tail_model` — the disagreement is real but modest, and
it shrinks exactly where the score is worst:

| fold | retro-v1 H0-s7 | tail model | flat 0.5/0.5 | gain over better | residual corr | corr on bulk | corr on no-NM |
|---|---|---|---|---|---|---|---|
| july | 322.80 | 327.31 | 314.03 | **8.77** | 0.867 | 0.892 | 0.772 |
| december | 222.59 | 223.12 | 216.93 | **5.65** | 0.895 | 0.904 | 0.688 |
| january | 347.45 | 350.55 | 344.48 | **2.96** | 0.949 | 0.886 | 0.970 |

The assignment quotes about 20 s for c7 against c6. That was measured against
`c6` (july 343.6); against the day's better single model (`H0-s7`, july 322.8)
the same flat blend is worth 8.8 s. The lane's baseline is therefore restated:
**8.8 / 5.7 / 3.0 s is what one already has, and a new member has to beat that,
not the 20 s figure.**

**Ceiling estimate (mine, stated as the assignment asks).** For k members of
equal strength with common residual correlation ρ, the flat blend's RMSE is
σ·√((1 + (k−1)ρ)/k). The measured pair sits at ρ ≈ 0.87 on july and the
two-member arithmetic (√((1+ρ)/2) = 0.966, so 3.4 %) predicts 11.0 s where 8.8 s
was measured — the shortfall is the members not being of equal strength. Tree
learners on an *identical* frame typically correlate higher than two lineages
with different features do, so I expect the new members to land at ρ ≈ 0.92 to
0.96 against `L0`, not at 0.87. At k = 6 and ρ = 0.93 the arithmetic gives
3.0 %; at ρ = 0.96, 1.7 %. **My estimate before measuring: 6 to 13 s on july,
4 to 8 s on december, 2 to 5 s on january, against the best single model** —
that is, the same order as the existing two-lineage blend, not a multiple of it.
The 2 to 8 % the assignment quotes for tabular blends is not reachable here
because the members share a frame; the part of that range this task can reach is
the low end. The one place the estimate could be beaten is per-regime weighting:
the two lineages already disagree far more on no-NM rows (ρ 0.69 to 0.77 on two
of three folds) than overall, and that population carries 8 to 59 % of the
squared error depending on the fold.

## Baselines, restated under this experiment's protocol

`prc.tail_model` OOF (`experiments/exp-002-tail/oof/exp-002-tail-oof.parquet`),
recomputed here with the 60 s floor and the regime split this experiment uses:

| fold | n | rmse | ≤3600 | >3600 | no-NM | noac | nm_bulk | nm_late | nonm |
|---|---|---|---|---|---|---|---|---|---|
| july | 190,713 | 327.3 | 284.5 | 2775.2 | 1381.1 | 4199.4 | 227.7 | 525.0 | 651.5 |
| july_loo | 190,713 | 310.4 | 269.2 | 2648.5 | 1247.5 | 3822.9 | 223.6 | 500.5 | 570.0 |
| december | 165,677 | 223.1 | 208.0 | 2042.3 | 763.7 | 2552.6 | 198.4 | 432.3 | 493.4 |
| january | 153,706 | 350.6 | 214.8 | 6029.2 | 2889.7 | 4449.3 | 201.2 | 455.0 | 2797.1 |

Regime sizes (july / december / january): nm_bulk 172,270 / 158,386 / 147,691;
nm_late 14,509 / 5,648 / 4,583; nonm 3,595 / 1,554 / 1,369; noac 339 / 89 / 63.
`noac` is the smallest regime and carries a fifth of the squared error, which is
why `prc.blend.fit_per_regime` sets `min_rows` low enough (50) to fit it from
two months rather than falling back to pooled weights.

## Protocol: round counts are frozen per fold family

Chosen once per arm by `--select`, early-stopping on the **last training month**
(2025-06 for the six-month `july` fold, 2025-11 for the eleven-month folds via
the `december` fold and 2025-12 via the `january` fold), rescaled by 1/(1−held).
No validation month is touched. Recorded in `select-*.json`.

| arm | m6 | m11 | note |
|---|---|---|---|
| `S-lgb-lin` | 175 | 350 | linear leaves, residual target |
| `S-lgb-mono` | 225 | 250 | plain trees, monotone, raw target |

CatBoost's two families disagreed by a factor of seven on the first reading
(m6 3,625 rounds, m11 450), which is far outside what one extra half of the
training set explains, so the eleven-month count for the GPU arms is the mean of
two readings taken on two different early-stop months (2025-11 and 2025-12)
rather than one. `rounds_of.py` does the averaging and the numbers land in the
table above.

Depth-10 CatBoost round selection was started and **deferred**: it was still
running after 25 minutes and was blocking the depth-8 arm, which is the number
the desk asked for first. It is a queued follow-up, not a negative result.

## Finding 2 — the blend's money is in the regimes, and a flat weight loses some of it

The same pair (`H0-s7` against `prc.tail_model`), split by the four
input-defined regimes. `gain_in_regime` is what a flat 0.5/0.5 saves inside the
regime; `fold_rmse_saved` is what that regime's gain alone is worth on the whole
fold, in quadrature.

| fold | regime | n | retro | tail | flat 0.5 | corr | SSE share | gain in regime | fold RMSE saved |
|---|---|---|---|---|---|---|---|---|---|
| july | nm_bulk | 172,270 | 229.0 | 227.7 | 224.5 | 0.933 | 0.455 | 4.5 | 2.88 |
| july | nm_late | 14,509 | 520.5 | 525.0 | 507.7 | 0.886 | 0.198 | 12.8 | 1.56 |
| july | nonm | 3,595 | 646.2 | 651.5 | 626.8 | 0.867 | 0.076 | 19.4 | 0.72 |
| july | noac | 339 | 3992.5 | 4199.4 | 3829.7 | **0.737** | 0.272 | 162.8 | 3.53 |
| december | nm_bulk | 158,386 | 195.8 | 198.4 | 193.5 | 0.928 | 0.740 | 2.3 | 1.94 |
| december | nm_late | 5,648 | 432.2 | 432.3 | 422.0 | 0.907 | 0.129 | 10.2 | 0.67 |
| december | nonm | 1,554 | 601.5 | **493.4** | 515.9 | 0.776 | 0.068 | 85.6 | 2.02 |
| december | noac | 89 | 2408.7 | 2552.6 | 2235.4 | **0.620** | 0.063 | 173.4 | 0.97 |
| january | nm_bulk | 147,691 | 198.0 | 201.2 | 195.3 | 0.915 | 0.312 | 2.6 | 1.44 |
| january | nm_late | 4,583 | 461.6 | 455.0 | 445.0 | 0.886 | 0.053 | 16.6 | 0.65 |
| january | nonm | 1,369 | 2825.2 | 2797.1 | 2801.0 | **0.986** | 0.589 | 24.2 | 1.75 |
| january | noac | 63 | 3698.3 | 4449.3 | 3894.6 | 0.824 | 0.046 | **−196.3** | **−0.88** |

Three things follow, and they set the design of the blend rather than merely
describing it.

**The bulk is where the members agree most and there is least to win.**
`nm_bulk` is 90 % of rows and correlates 0.915 to 0.933 — the high end of my
predicted range. A flat blend buys 2.3 to 4.5 s inside it, 1.4 to 2.9 s on the
fold. Any large blend gain has to come from somewhere else.

**The `noac` regime is where the members genuinely disagree** (ρ 0.62 to 0.82,
the lowest anywhere) **and it is 4.6 to 27 % of the squared error on 63 to 339
rows.** That is the single most valuable place for a fitted weight.

**And it is exactly where a flat weight fails.** On january the tail model's
noac predictions are much worse than retro-v1's (4449 against 3698), so
averaging them *costs* 196 s inside the regime and 0.9 s on the fold; on
december the opposite holds for `nonm` (tail 493 against retro 601) and the flat
average again lands between the two rather than near the better one. A flat
average is only optimal when the members are of equal strength, and by regime
they are not. This is the measured case for per-regime least squares, and it is
why `prc.blend` fits weights per regime with `min_rows` low enough that `noac`
gets its own vector. Whether the fitted weights *transfer* to an unseen month is
a separate question, and the rotation is what answers it.

### The frozen round counts

| arm | m6 (july) | m11 (july_loo, december, january) | m11 readings |
|---|---|---|---|
| `L0-lin` LightGBM linear leaves, residual | 175 | 350 | 350 (ES 2025-11) |
| `L3-monoraw` LightGBM monotone, raw target | 225 | 250 | 250 (ES 2025-11) |
| `C1-d8` CatBoost depth 8, residual | 3,625 | 425 | 450 (ES 2025-11), 400 (ES 2025-12) |
| `X1-d8` XGBoost hist depth 8, residual | 75 | 500 | 350 (ES 2025-11), 650 (ES 2025-12) |

**CatBoost's 8.5× gap between the two fold families is real, not an artefact of
which month was held out.** That was the reason for taking a second reading, and
the two eleven-month readings agree closely (450 and 400) while the six-month
reading sits at 3,625. So it is a property of the training-set size, not of
November: with 1.0 M rows CatBoost's symmetric trees and ordered target
statistics need thousands of weak trees to fit what 1.9 M rows let it fit in
400. The practical consequence is that a single round count shared across fold
families would have been wrong for CatBoost by a factor of eight, which is worth
knowing before anyone reads a CatBoost-vs-LightGBM row.

XGBoost shows the same sign more mildly (75 against 500) but its two m11
readings disagree by a factor of two (350 and 650), so its frozen 500 is the
least well determined count in the table; the spread is recorded here rather
than hidden in an average.

One detail that avoids a known trap: the Huber arm changes `objective` only, and
leaves `metric="rmse"`, so its round count is still chosen against the loss the
competition scores. exp-001's caveat about `--target log` choosing rounds in a
transformed space does not apply here.

### One correction made before any arm was scored: the noac head's round budget

The first CatBoost run was stopped mid-fold and restarted. The per-learner noac
head had inherited LightGBM's 60-round budget for every learner, which is not a
portable number: the same frozen-round measurement that gives the main models
3,625 CatBoost rounds against 175 LightGBM rounds on the six-month family says
CatBoost's oblivious trees take much smaller steps at the same learning rate, so
a 60-round CatBoost head would have been underfitted on precisely the 63 to 339
rows that carry a fifth of the squared error — and the arm table would have read
that as CatBoost being a worse learner. `NOAC_ROUNDS` is now per kind
(LightGBM and XGBoost 60, CatBoost 600), and `C1-d8h60` re-runs the july fold at
60 so the choice is backed by a measurement rather than by the argument above.
No scored arm was produced under the old budget.

## Finding 3 — the re-baselined LightGBM reference reproduces exp-001

`L0-lin` is exp-001's guarded configuration (v2 frame, residual target, linear
leaves, schedule-anchored noac head, 60 s floor, parameter-free residual guard)
refitted here under this experiment's protocol: 175 fixed rounds on the
six-month family instead of exp-001's independently early-stopped 125, and a
noac head that fits `y − (take-off − SCHED)` directly rather than passing the
anchor as an `init_score`.

| july | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|
| exp-001 `H0-s7` (early-stopped, 125 rounds) | 322.8 | 282.4 | 2682.1 | 1324.8 | 3992.5 |
| exp-008 `L0-lin` (fixed, 175 rounds) | **324.1** | 282.5 | 2724.1 | 1351.0 | 4035.9 |

1.3 s apart overall and 0.1 s apart on the bulk. That is inside the run-to-run
spread exp-001 measured for `linear_tree` (about 9 s on the december core, and
125/186/231 rounds across three seeds on this very configuration), so the
harness reproduces the lineage it is meant to extend and the re-baselining is
safe to build the blend table on. Every comparison below is against `L0-lin`,
measured here, not against the quoted exp-001 numbers.

## Finding 4 — a different learner on an identical frame disagrees more than predicted

This is the number the lane turns on. `C1-d8` (CatBoost, native categoricals
with ordered target statistics, depth 8, 3,625 frozen rounds) against `L0-lin`
(LightGBM, linear leaves, 175 frozen rounds) on the **same 54 columns**, july
fold, 190,713 rows:

|  | L0-lin | C1-d8 | T (tail model) |
|---|---|---|---|
| L0-lin | 1.000 | **0.905** | 0.859 |
| C1-d8 | 0.905 | 1.000 | 0.863 |
| T | 0.859 | 0.863 | 1.000 |

| single | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|
| `L0-lin` | 324.1 | 282.5 | 2724.1 | 1351.0 | 4035.9 |
| `C1-d8` | **320.6** | 278.4 | 2725.5 | 1309.1 | 3982.8 |
| `T` tail model | 327.3 | 284.5 | 2775.2 | 1381.1 | 4199.4 |

| flat average | july rmse |
|---|---|
| L0 + C1 | 314.6 |
| L0 + T (the existing two-lineage blend) | 314.0 |
| C1 + T | 312.6 |
| **L0 + C1 + T** | **310.3** |

**I was wrong about the ceiling in the conservative direction, and the reason
matters.** Finding 1 predicted that two learners sharing a frame would correlate
0.92 to 0.96, higher than the 0.87 between the two lineages, because feature
diversity is the usual source of blend gain. Measured, CatBoost and LightGBM
correlate **0.905** — barely above the cross-lineage figure, and *below* my
predicted floor. Changing the learner on a fixed frame turns out to buy almost
as much disagreement here as changing the whole feature set did. The mechanism
is visible in the regime split: LightGBM's linear leaves fit a local linear
function of the congestion columns, CatBoost's oblivious trees with ordered
target statistics fit the high-cardinality categoricals instead, and the two
make different errors on the rows where those two things compete.

| regime | n | corr(L0, C1) | L0 | C1 |
|---|---|---|---|---|
| nm_bulk | 172,270 | 0.932 | 230.1 | 227.5 |
| nm_late | 14,509 | 0.902 | 507.0 | **521.2** |
| nonm | 3,595 | **0.758** | 679.2 | **616.0** |
| noac | 339 | 0.904 | 4035.9 | 3982.8 |

CatBoost is the better learner overall but not everywhere — it is 14 s worse on
`nm_late` and 63 s better on `nonm`. That is a fitted-weight opportunity, not a
tie-break, and it is the second independent piece of evidence for per-regime
blending after Finding 2.

**Revised estimate:** the three-way flat blend is already 10.3 s better than the
best single model on july, against the 6 to 13 s I predicted for the *whole*
lane. With XGBoost still to land, per-regime fitted weights, and seed bagging,
the low end of that estimate is already beaten and the lane is worth pushing.

## Finding 5 — more training months help the rare regimes and hurt the bulk

`july` and `july_loo` validate the *same month* (2025-07, the same 190,713 rows)
from six and eleven training months respectively, so the pair isolates what the
extra five months buy. For `L0-lin`:

| regime | july (6 months, 175 rounds) | july_loo (11 months, 350 rounds) | delta |
|---|---|---|---|
| overall | 324.1 | 323.7 | **−0.4** |
| nm_bulk (172,270) | 230.1 | 236.1 | **+6.0** |
| nm_late (14,509) | 507.0 | 512.9 | +5.9 |
| nonm (3,595) | 679.2 | 636.1 | −43.1 |
| noac (339) | 4035.9 | 3880.1 | **−155.9** |

Five extra months nearly double the training population of the rare regimes
(noac is 0.07 % of departures, so six months give roughly 300 rows and eleven
give roughly 550) and they improve exactly there. The bulk gets *worse* by 6 s,
and the two effects almost cancel — which is why the overall number moves 0.4 s
and looks, wrongly, like the extra data did nothing.

Two candidate causes for the bulk regression, both testable and neither settled
here:

1. **The round count.** The eleven-month count (350) was chosen by early
   stopping on 2025-11, a calm month; the six-month count (175) on 2025-06, a
   month that resembles the validation month. If the right number of trees for
   a summer validation month is season-dependent, `july_loo` is being fitted at
   the wrong budget. `run.py --es-month` exists for exactly this test (select on
   `july_loo` holding out 2025-08 instead of 2025-12) and it is a queued
   follow-up, not run here.
2. **`month_num` is out of range on every fold.** The frame carries a numeric
   month and each fold's validation month is absent from its training months —
   7 for `july_loo`, 12 for `december`, 1 for `january`. The submission fit has
   all twelve, so both ranking months are *inside* the training range there.
   Every fold in this repo is therefore pessimistic in this one specific way,
   and `july_loo` is not the exception. This is an observation about the frozen
   v2 frame and belongs to retro-v1, not to this experiment; it is passed on
   rather than acted on, since the frame is a published interface.

The consequence for this lane is narrow and worth stating: **the board estimate
(july_loo, january) is the one that matters and it is the one where the rare
regimes are best trained**, so a blend fitted per regime has more to work with
there than the forward `july` fold suggests.

## Finding 6 — a fixed early-stopping patience is not scale-free, and it broke the CatBoost arm

`C1-d8`'s four folds, against the re-baselined LightGBM reference:

| arm | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|
| `L0-lin` | 324.1 | 323.7 | 224.5 | 347.6 | 334.5 |
| `C1-d8` | **320.6** | 334.1 | 243.9 | 361.1 | 346.3 |

CatBoost wins the one fold where it was given 3,625 rounds and loses all three
where it was given 425, by 10 to 19 s, and the loss is in the bulk (december
226.6 against LightGBM's 206.8, where on july it is 278.4 against 282.5 — a 24 s
swing in the same direction as the round count). The `july`/`july_loo` pair makes
this unambiguous, because those two folds score the *same 190,713 rows*: with
five more months of training data and 425 instead of 3,625 rounds, CatBoost gets
13.5 s worse, while the tail model gets 16.9 s better over the same pair.
**So the arm as first measured is an artefact of my protocol, not a property of
CatBoost, and the C1 rows above are withdrawn pending the repair.**

There are two candidate causes and they are separable by two selections that
touch no validation month:

- **(a) Season.** The protocol early-stops on the last training month, which is
  2025-12 for `july_loo` and 2025-11 for `december` — calm winter — while the
  six-month fold early-stops on 2025-06. If the number of trees a summer month
  wants is larger, the eleven-month count is being chosen on the wrong kind of
  month. Test: early-stop `july_loo` on 2025-08 at the same patience.
- **(b) Patience.** `--es-rounds 100` is 57 % of LightGBM's trajectory (its best
  iteration is 137) and 3 % of CatBoost's (2,969). A fixed patience is not
  scale-free, and the logged curves show exactly the shape that exposes it: on
  the June early-stop month CatBoost improved only 5.9 s between round 600 and
  round 2,968 — a very flat descent that survived a 100-round patience only
  because a tiny improvement kept arriving — while on the November month it went
  100 rounds without one at round 404 and stopped. Test: re-select the *same*
  fold on the *same* month at patience 900, about 30 % of CatBoost's six-month
  trajectory.

Both tests are running (`run_gpu3.sh`, `run_gpu4.sh`). The general lesson is
already safe to state, and it applies to any future arm in this repo: **an
early-stopping patience has to be set relative to the learner's own convergence
scale, and a patience chosen for LightGBM will silently truncate CatBoost.**
exp-001's `protocol-halt` established that independently early-stopped round
counts make arm comparisons into round-count comparisons; this is the same
failure one level down, in how the frozen count itself is chosen.

### The same failure on XGBoost, in the mirror fold — and what it says the real cause is

| arm | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|
| `L0-lin` (175 / 350) | 324.1 | 323.7 | 224.5 | 347.6 | 334.5 |
| `X1-d8` (75 / 500) | **357.5** | 336.2 | **227.5** | 356.1 | 345.1 |
| `C1-d8` (3625 / 425) | **320.6** | 334.1 | **243.9** | 361.1 | 346.3 |

XGBoost fails on the fold where it got 75 rounds (33 s behind the reference) and
is within 3 s of it on the fold where it got 500 — the same pattern as CatBoost
with the folds swapped. Two learners, opposite folds, same sign: whenever the
frozen count came out small, the arm is bad.

The logged curve says why, and it is not really about patience or season:

    XGBoost, july fold, early-stopping on 2025-06
    [0]    train-rmse 474.12   es-rmse 535.12
    [165]  train-rmse 273.34   es-rmse 399.86      best was iteration 66

The held-out **month's** RMSE genuinely gets worse after round 66 — so early
stopping is behaving correctly with respect to the signal it is given. The
signal is the problem. RMSE on one month is dominated by that month's handful of
extreme rows: 0.34 % of departures carry 67 % of the squared error (COMMON.md),
so a single 88,000 s row moving a little swamps two million ordinary flights
improving. June's extremes turned upward at round 66 while the bulk was still
descending, and the stop fired. exp-001 saw the same instrument fail from the
other side — the same configuration choosing 125, 186 and 231 rounds at three
seeds — and its `protocol-halt` withdrew a ladder of feature arms over it.

So the finding is stronger and more general than "CatBoost needed a bigger
patience": **on this task, early stopping on held-out-month RMSE is an unreliable
instrument for choosing a tree budget, for any learner, because the metric is
dominated by a few rows.** Anything downstream of it — a frozen count, a feature
comparison, a learner comparison — inherits that noise. The two selections in
flight measure how much of it a large patience recovers; the honest fallback, if
patience is not enough, is to choose the budget on the input-defined `nm_bulk`
regime, which is 90 % of the rows and is not hostage to a handful of them, and
to say plainly that the count is then chosen for the bulk.

## Finding 7 — the first blend: fitting beats flat, and per-regime does not beat global

Four members with complete four-fold OOF (`L0-lin`, `C1-d8`, `X1-d8`, and
`prc.tail_model`), weights rotated so that a fold's weights are fitted only on
the *other* validation months — and never on `july` when scoring `july_loo` or
vice versa, since those two folds are the same 190,713 rows.

Residual correlation, july:

|  | L0-lin | C1-d8 | X1-d8 | T-tail |
|---|---|---|---|---|
| L0-lin | 1.000 | 0.905 | 0.866 | 0.859 |
| C1-d8 | 0.905 | 1.000 | 0.877 | 0.863 |
| X1-d8 | 0.866 | 0.877 | 1.000 | 0.842 |
| T-tail | 0.859 | 0.863 | 0.842 | 1.000 |

Every cross-learner correlation is between 0.84 and 0.91 — no pair is
redundant, and XGBoost is the *most* different member despite being the weakest.

| blend | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|
| best single member | 320.6 | 310.4 | 223.1 | 347.6 | 328.8 |
| flat average | 315.5 | 308.4 | 220.7 | 347.5 | 326.7 |
| NNLS, one global weight vector | 314.1 | 304.5 | 217.7 | 345.4 | **323.3** |
| NNLS per regime, no shrinkage | 316.1 | 310.0 | 217.0 | 344.3 | 325.6 |
| NNLS per regime, shrinkage chosen by inner rotation | 313.5 | 304.7 | 217.1 | 345.0 | 323.2 |
| OLS, one global weight vector | 314.8 | 304.7 | 217.7 | 345.4 | 323.4 |

Gains over the best single model on each fold: **6.5 / 5.9 / 6.0 / 2.6 s**, and
5.5 s on the board estimate — inside the 6-to-13 / 4-to-8 / 2-to-5 s range
Finding 1 predicted, and achieved with two of the four members crippled by the
round-count defect of Finding 6.

**Fitting the weights beats a flat average** (1.4 to 3.9 s), which is worth
saying because exp-001's `c7` found the opposite with two members — with four
members of visibly unequal strength there is now something for least squares to
do. NNLS and OLS land within 0.7 s of each other; the non-negativity constraint
costs nothing and is kept for conditioning.

**Per-regime weights, fitted honestly, do not beat one global vector.** Raw
per-regime weighting is 2.0 s *worse* on july and 5.5 s worse on july_loo, and
its loss is concentrated in the noac column (3946.8 against the global blend's
3837.6). That is the opposite of what Finding 2 predicted from the two-lineage
measurement, and the reason is Finding 8.

## Finding 8 — why per-regime weights fail: the ranking is a property of the month, not the regime

Best member in each regime, on each fold:

| regime | july | july_loo | december | january |
|---|---|---|---|---|
| nm_bulk | C1-d8 | T-tail | L0-lin | L0-lin |
| nm_late | L0-lin | T-tail | T-tail | T-tail |
| nonm | C1-d8 | T-tail | T-tail | T-tail |
| noac | C1-d8 | T-tail | **X1-d8** | C1-d8 |

`noac` is the clearest case and the most damaging one. `X1-d8` is the *worst*
member on those rows on three folds (4848 on july, 4492 on july_loo, 4449 on
january) and the *best* on december (2131 against the next best 2251) — on 89
rows. A weight vector fitted on december's noac rows therefore hands weight to
the member that is worst everywhere else. Fitting 4 weights on 152 heavy-tailed
rows does not estimate a regime, it estimates two months.

The repair — shrinking each regime's weights toward the pooled vector by sample
size, `n/(n+n0)`, with `n0` chosen by an inner rotation *inside the fitting
months* so the scored fold never enters the choice — recovers the loss and
brings per-regime level with global (board estimate 323.2 against 323.3). The
inner rotations chose n0 = 250, 250, 500 and 2000, i.e. heavy shrinkage, which
is the same conclusion stated as a number.

So the per-regime machinery is kept (it is in `prc.blend` and costs nothing) but
**the honest reading today is that it buys nothing over a global vector**, and
the shipped blend uses global weights unless the members change that. The
revival condition is concrete: more validation months, or members whose regime
ordering is stable across months — the three large regimes already show a stable
ordering between `december` and `january`, and it is `noac`, at 63 to 339 rows,
that carries the instability. The two July folds cannot yet be read on this
question because `C1-d8` and `X1-d8` hold different tree budgets there.

## Finding 9 — the noac head's round budget is worth 10 s of total RMSE

`C1-d8h60` is `C1-d8` with one thing changed: the noac head keeps LightGBM's
60-round budget instead of CatBoost's 600. July fold, same seed, same frozen
main-model count:

| arm | july rmse | ≤3600 | noac (339 rows) | noac share of SSE |
|---|---|---|---|---|
| `C1-d8h60` (head 60) | 330.7 | 292.9 | 4429.0 | 0.319 |
| `C1-d8` (head 600) | **320.6** | 278.4 | **3982.8** | 0.274 |

446 s on the 339 noac rows, **10.1 s on the whole fold**. So the correction made
before any arm was scored was not housekeeping: had the per-learner head kept
LightGBM's budget, CatBoost would have read 330.7 against LightGBM's 324.1 on
the one fold where its main model was correctly sized, and the lane's headline
finding — that a different learner on an identical frame is worth blending —
would have been inverted by a hyperparameter that belongs to neither learner's
main model.

For reference, retro-v1's LightGBM head scores 3992.5 on these same rows, so a
correctly budgeted CatBoost head (3982.8) is its equal rather than its superior;
the 10 s is the cost of the wrong budget, not a gain from CatBoost. It also puts
a number on how much of this task's score lives in 0.18 % of the rows: moving
339 predictions moves the fold by 10 s.

## Finding 10 — season is refuted; the tree budget tracks the training-set size

The season hypothesis (a) of Finding 6 predicted that early-stopping `july_loo`
on 2025-08 — a summer month, eleven months of training data — would return a
count near the six-month fold's 3,625 rather than near the winter-early-stopped
425. It did not:

| learner | fold | training months | early-stop month | patience | best iter | frozen |
|---|---|---|---|---|---|---|
| CatBoost | july | 6 | 2025-06 (summer) | 100 | 2,969 | 3,625 |
| CatBoost | december | 11 | 2025-11 (winter) | 100 | 405 | 450 |
| CatBoost | january | 11 | 2025-12 (winter) | 100 | 366 | 400 |
| CatBoost | july_loo | 11 | **2025-08 (summer)** | 100 | **547** | **600** |
| XGBoost | july | 6 | 2025-06 (summer) | 100 | 66 | 75 |
| XGBoost | december | 11 | 2025-11 (winter) | 100 | 326 | 350 |
| XGBoost | january | 11 | 2025-12 (winter) | 100 | 599 | 650 |
| XGBoost | july_loo | 11 | **2025-08 (summer)** | 100 | **106** | **125** |

CatBoost's three eleven-month readings are 450, 400 and 600 across two winter
early-stop months and one summer one — they agree within a factor of 1.5, and
all of them are five to eight times smaller than the six-month reading. **So the
eleven-month budget is a property of the training-set size, not of the season**,
and hypothesis (a) is closed by measurement. The mechanism is plausible and
specific to CatBoost: its ordered target statistics on the high-cardinality
categoricals (`airport_stand` and friends) are far more informative with 1.9 M
rows than with 1.0 M, so with more data the model reaches its fit in a few
hundred oblivious trees and with less data it needs thousands of weak ones.

XGBoost does not share that pattern — its counts move with the *month* (75 and
125 on summer early-stop months, 350 and 650 on winter ones) rather than with
the training size — which is a second reason not to read one learner's round
behaviour off another's.

This leaves hypothesis (b), patience, as the live one, and it is the one that
matters: if 425 is where a *noisy* instrument stopped rather than where CatBoost
converged, all three eleven-month readings are truncated together and agreeing
with each other proves nothing. The patience-900 re-selection of the same fold
and the same early-stop month is the discriminator and is running.

The pre-repair `july_loo` predictions for both GPU arms are kept as
`oof/*-m11winterES-july_loo.parquet` so the two budgets can be compared on the
same rows rather than one silently replacing the other.

## Finding 11 — the plain-tree LightGBM sibling is the better *board* model

`L1-plain` is `L0-lin` with linear leaves turned off and nothing else changed.

| arm | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|
| `L0-lin` (linear leaves) | **324.1** | 323.7 | **224.5** | **347.6** | 334.5 |
| `L1-plain` (plain trees) | 330.7 | **314.5** | 225.2 | 349.0 | **330.2** |

The linear-leaf model wins three folds and loses the board estimate by 4.3 s,
because the one fold it loses is `july_loo` — 55.7 % of the ranking mix — and it
loses it by 9.2 s. On the forward `july` fold, which has no same-season training
data, linear leaves win by 6.6 s. This is the pattern exp-001's SPEC predicted
for itself and could not test: *"everything before this point was steered by the
forward `july` fold, which punishes exactly the mechanisms that need same-season
data"*. Here the sign flips the other way on the fold that actually estimates the
board, and the flip is worth more than the whole blend gain on january.

Two consequences. For this lane: both siblings stay as members, because a pair
that disagrees about which fold it is good at is exactly what a blend wants. For
retro-v1: **the linear-leaf decision, which the v2 ladder was built on, was
settled on `july` and reverses on `july_loo`** — that is a message through the
desk, not something I act on, since the frame and the lineage are theirs.

### A detail that limits the noac regime's diversity

Every LightGBM arm shares one noac head: the head is built from the arm's
*kind*, not its variant, so `L0-lin`, `L1-plain`, `L2-hub` and `L3-monoraw` have
bit-identical noac predictions (4035.9 / 3880.1 / 2586.9 / 3625.4 on the four
folds). That is deliberate — it isolates main-model diversity, which is what the
learner comparison is about — but it means the `noac` regime's diversity comes
only from CatBoost, XGBoost and the tail model, three members rather than six.
Since `noac` is where the members disagree most (Finding 2) and where per-regime
weights failed for lack of rows (Finding 8), a per-variant noac head is a cheap
and specific follow-up: 339 rows, seconds to fit.

## Finding 12 — the round-count instrument is not measuring convergence

Both GPU arms re-ran `july_loo` under the season-matched count. The two repairs
moved in opposite directions, and that is the point:

| arm | july_loo, winter-ES count | july_loo, summer-ES count | better |
|---|---|---|---|
| `C1-d8` CatBoost | 425 rounds → 334.1 | **600 rounds → 330.6** | the larger count |
| `X1-d8` XGBoost | **500 rounds → 336.2** | 125 rounds → 340.5 | the larger count |

Season-matching helped CatBoost (whose summer reading was the larger one) and
hurt XGBoost (whose summer reading was the smaller one). Put together with the
folds already measured, there are four independent comparisons on this frame, and
**in every one the larger tree budget gave the better model**:

| comparison | small count | large count |
|---|---|---|
| CatBoost july_loo | 425 → 334.1 | 600 → 330.6 |
| XGBoost july_loo | 125 → 340.5 | 500 → 336.2 |
| CatBoost july vs eleven-month folds | 425 → 243.9 (dec) | 3,625 → 320.6 (jul, best CatBoost fold) |
| XGBoost july vs december | 75 → 357.5 | 350 → 227.5 |

So the `m11s` season family, which Finding 10 already refuted as a *cause*, is
not the fix either — it is orthogonal to the real problem. **Early stopping on a
single held-out month's RMSE is not measuring where these models converge.** Its
error is one-directional: it stops early, because the metric it watches is
dominated by the handful of rows that carry two thirds of the squared error, and
a few of those rows drifting upward looks exactly like convergence. Whichever
early-stop month happens to return a larger number returns a better model, which
is the signature of a truncating instrument rather than of a fold property.

This is the same instrument exp-001 withdrew a ladder of feature arms over
(`protocol-halt`, 125/186/231 rounds at three seeds). The finding here adds the
direction of the bias and the mechanism, and it means the fix cannot be "pick the
early-stop month more carefully". The two candidate fixes that remain are a
patience large enough not to fire on tail noise (under test) and averaging the
early-stopping curve over several held-out months instead of trusting one.

## Finding 13 — CONFIRMED: the early-stopping patience truncated every tree budget, twelvefold

Same fold, same training months, same early-stop month, same seed. One
parameter changed:

| CatBoost, december fold, early-stop on 2025-11 | patience | best iteration | frozen count |
|---|---|---|---|
| as every arm in this experiment was selected | 100 | 405 | 450 |
| re-selected | **900** | **4,879** | **5,325** |

**A twelvefold truncation from the early-stopping patience alone**, and 4,879 is
close enough to the 5,000-round cap that the true optimum may be higher still.
This closes the question opened in Finding 6 and settles it against hypothesis
(a): season explains nothing (Finding 10), patience explains everything.

The mechanism is the one Finding 12 identified. RMSE on a held-out month is
dominated by the 0.34 % of rows carrying 67 % of the squared error, so the
curve is not smooth — it wanders. A 100-round patience fires on the first
wander. The descent is real but slow and noisy: CatBoost improved only 5.9 s
between rounds 600 and 2,968 on the july fold's early-stop month, which is
exactly the shape a short patience cannot survive.

**This is not a fact about exp-008.** Every frozen tree budget in this repo that
was chosen by a patience-100 early stop on a held-out month is suspect by the
same argument, and that includes exp-001's ladder, whose `protocol-halt` already
withdrew its feature arms for the related reason of seed instability. The
finding adds what the halt could not say: the counts were not merely unstable,
they were systematically **too small**, and every one of the four independent
comparisons in Finding 12 pointed the same way. A worker choosing a round count
on this data should use a patience that is a real fraction of the learner's own
convergence length — for CatBoost here that is hundreds of rounds, not 100 —
or average the early-stopping curve over several held-out months.

**Actions taken.** The `m11s` season-matched fold family, added on the
hypothesis Finding 10 refuted, is retired and `july_loo` is back in the plain
`m11` family; `--rounds-m11s` and `ES_MONTH` survive as the instruments that
produced the measurement. `C2-d8p` (CatBoost) and `X2-d8p` (XGBoost) refit the
three eleven-month folds at corrected budgets. The `july` fold keeps its
patience-100 budget for now and is flagged: it is the pessimistic fold and is
not part of the board estimate, so it is the cheapest thing to leave until last.

## Finding 14 — the truncation was worth 18.9 s on a board-estimate fold

`july_loo`, the same 190,713 rows, the same learner, the same seed, the same
features. Only the frozen tree budget differs, and each budget came from the
same early-stopping procedure run with a different patience or early-stop month:

| CatBoost budget | how it was chosen | july_loo rmse | ≤3600 | no-NM |
|---|---|---|---|---|
| 425 rounds | patience 100, ES 2025-12 | 334.1 | 290.5 | 1309.8 |
| 600 rounds | patience 100, ES 2025-08 | 330.6 | 287.6 | 1309.5 |
| **5,325 rounds** | **patience 900, ES 2025-11** | **315.2** | **274.3** | **1305.8** |

**18.9 s between the first and the last, on the fold that is 55.7 % of the board
estimate**, from a parameter of the *measurement procedure* rather than of the
model. For scale, the entire four-member blend gains 5.9 s on this fold, and the
best mechanism exp-001 found all day — the noac head plus the residual
parameterisation — is worth about 20 s. The gain is almost all in the bulk
(290.5 → 274.3); the no-NM rows barely move, which is what one expects if the
model was simply not finished fitting the ordinary flights.

The corrected CatBoost now stands where the lane expected it: 315.2 against
`L0-lin`'s 323.7 and `L1-plain`'s 314.5 on the same rows, with the tail model at
310.4. Every conclusion in Findings 4 and 7 that compared learners on the
eleven-month folds was reading a truncation, and the withdrawal of those rows
was correct.

**The wider claim, stated plainly for the desk.** This repo's experiments freeze
tree budgets chosen by a patience-100 early stop on one held-out month. That
instrument is biased low, here by up to twelvefold, and the bias is worth more
score than the mechanisms the experiments were built to compare. Any arm in any
lane whose count came from it — including exp-001's v2 ladder — is measuring its
mechanism through a variable roughly four times larger than the effects being
tested. Re-running one arm at a corrected budget costs about 25 GPU-minutes and
is the cheapest high-value check available to any lane right now.

## Per-learner four-fold table, with the COMMON.md decomposition

Generated by `report.py` from `result-*.json`, not typed. Added after review,
which noted that the summary pivot below carries only overall RMSE while
COMMON.md requires overall / ≤3600 / >3600 / no-NM on every fold, and that
`X2-d8p` (the corrected XGBoost) had no row anywhere.

| arm | fold | rounds | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|---|---|
| `L0-lin` | july | 175 | 324.1 | 282.5 | 2724.1 | 1351.0 | 4035.9 |
| `L0-lin` | july_loo | 350 | 323.7 | 285.9 | 2601.7 | 1291.2 | 3880.1 |
| `L0-lin` | december | 350 | 224.5 | 206.8 | 2216.2 | 840.7 | 2586.9 |
| `L0-lin` | january | 350 | 347.6 | 213.4 | 5972.4 | 2867.9 | 3625.4 |
| `L1-plain` | july | 350 | 330.7 | 289.7 | 2736.2 | 1395.3 | 4035.9 |
| `L1-plain` | july_loo | 300 | 314.5 | 276.0 | 2586.7 | 1301.3 | 3880.1 |
| `L1-plain` | december | 300 | 225.2 | 207.3 | 2224.6 | 841.4 | 2586.9 |
| `L1-plain` | january | 300 | 349.0 | 214.7 | 5987.3 | 2873.9 | 3625.4 |
| `L2-hub` | july | 3225 | 323.0 | 281.2 | 2723.0 | 1314.5 | 4035.9 |
| `L2-hub` | july_loo | 2100 | 310.8 | 271.7 | 2588.1 | 1265.5 | 3880.1 |
| `L2-hub` | december | 2100 | 223.6 | 205.9 | 2207.8 | 818.0 | 2586.9 |
| `L2-hub` | january | 2100 | 348.1 | 213.7 | 5980.1 | 2862.2 | 3625.4 |
| `L3-monoraw` | july | 225 | 331.1 | 285.8 | 2866.1 | 1357.6 | 4035.9 |
| `L3-monoraw` | july_loo | 250 | 318.3 | 276.6 | 2699.7 | 1296.1 | 3880.1 |
| `L3-monoraw` | december | 250 | 227.3 | 208.8 | 2274.3 | 834.4 | 2586.9 |
| `L3-monoraw` | january | 250 | 350.7 | 215.9 | 6013.8 | 2872.8 | 3625.4 |
| `L3r-monores` | july | 350 | 331.6 | 290.8 | 2729.9 | 1397.9 | 4035.9 |
| `L3r-monores` | july_loo | 300 | 315.3 | 276.6 | 2594.5 | 1308.4 | 3880.1 |
| `L3r-monores` | december | 300 | 226.2 | 208.3 | 2234.7 | 844.7 | 2586.9 |
| `L3r-monores` | january | 300 | 349.8 | 215.6 | 5995.3 | 2876.6 | 3625.4 |
| `C2-d8p` | july_loo | 5325 | 315.2 | 274.3 | 2659.6 | 1305.8 | 4020.0 |
| `C2-d8p` | december | 5325 | 222.9 | 207.5 | 2062.3 | 765.4 | 2251.4 |
| `C2-d8p` | january | 5325 | 348.3 | 216.8 | 5931.0 | 2858.8 | 3442.6 |
| `X2-d8p` | july_loo | 350 | 337.1 | 299.7 | 2646.0 | 1458.5 | 4492.1 |
| `X2-d8p` | december | 350 | 228.9 | 211.8 | 2201.4 | 773.8 | 2131.2 |
| `X2-d8p` | january | 350 | 356.7 | 219.2 | 6123.1 | 2922.9 | 4448.7 |
| `C1-d8` | july | 3625 | 320.6 | 278.4 | 2725.5 | 1309.1 | 3982.8 |
| `C1-d8` | july_loo | 600 | 330.6 | 287.6 | 2795.8 | 1309.5 | 4020.0 |
| `C1-d8` | december | 425 | 243.9 | 226.6 | 2281.5 | 796.0 | 2251.4 |
| `C1-d8` | january | 425 | 361.1 | 234.6 | 5975.8 | 2852.6 | 3442.6 |
| `X1-d8` | july | 75 | 357.5 | 314.8 | 2906.0 | 1544.2 | 4848.3 |
| `X1-d8` | july_loo | 125 | 340.5 | 300.9 | 2731.8 | 1438.6 | 4492.1 |
| `X1-d8` | december | 500 | 227.5 | 210.6 | 2179.0 | 773.7 | 2131.2 |
| `X1-d8` | january | 500 | 356.1 | 218.9 | 6111.7 | 2920.7 | 4448.7 |


### overall RMSE, arm by fold

| arm | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|
| `L0-lin` | 324.1 | 323.7 | 224.5 | 347.6 | 334.5 |
| `L1-plain` | 330.7 | 314.5 | 225.2 | 349.0 | 330.2 |
| `L2-hub` | 323.0 | 310.8 | 223.6 | 348.1 | 327.9 |
| `L3-monoraw` | 331.1 | 318.3 | 227.3 | 350.7 | 333.0 |
| `L3r-monores` | 331.6 | 315.3 | 226.2 | 349.8 | 331.0 |
| `C2-d8p` | – | 315.2 | 222.9 | 348.3 | 330.2 |
| `X2-d8p` | – | 337.1 | 228.9 | 356.7 | 345.9 |
| `C1-d8` | 320.6 | 330.6 | 243.9 | 361.1 | 344.5 |
| `X1-d8` | 357.5 | 340.5 | 227.5 | 356.1 | 347.5 |

`C1-d8` and `X1-d8` are the withdrawn arms at truncated budgets (Findings 6 and
13); `C2-d8p` and `X2-d8p` supersede them on the eleven-month folds. `C2-d8p`'s
july fold is queued. Note `C2-d8p` december: 2062 on the tail and 765 on no-NM
against `L0-lin`'s 2216 and 841, so the corrected CatBoost is the best member
on the hardest rows of that fold, which is what earns it 0.235 of the blend.

## Per-learner summary (superseded by the table above; kept for the prose that cites it)

Every arm on the frozen v2 frame, residual target unless stated, 60 s floor and
the parameter-free residual guard, seed 7, round counts frozen per fold family.

| arm | learner and what differs | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|---|
| `L0-lin` | LightGBM, linear leaves (exp-001's configuration) | 324.1 | 323.7 | 224.5 | 347.6 | 334.5 |
| `L1-plain` | LightGBM, plain trees | 330.7 | 314.5 | 225.2 | 349.0 | 330.2 |
| `L2-hub` | LightGBM, plain trees, **Huber δ=1200** | 323.0 | **310.8** | 223.6 | 348.1 | **327.9** |
| `L3-monoraw` | LightGBM, plain, monotone, **raw target** | 331.1 | 318.3 | 227.3 | 350.7 | 333.0 |
| `L3r-monores` | LightGBM, plain, monotone, residual target | 331.6 | 315.3 | 226.2 | 349.8 | 331.0 |
| `C1-d8` | CatBoost depth 8, **truncated budgets** (withdrawn) | 320.6 | 330.6 | 243.9 | 361.1 | 344.5 |
| `C2-d8p` | CatBoost depth 8, corrected budgets | – | 315.2 | **222.9** | 348.3 | 330.2 |
| `X1-d8` | XGBoost hist depth 8 | 357.5 | 340.5 | 227.5 | 356.1 | 347.5 |

**The Huber sibling is the best single model of the experiment** — 327.9 on the
board estimate against `L0-lin`'s 334.5, and it is the best LightGBM on
`july_loo` at 310.8, just behind the tail model's 310.4 there. (Corrected after
review: an earlier version of this sentence had the comparison the wrong way
round. Lower RMSE wins; Finding 15's best-single-member row lists 310.4 for
T-tail. L2-hub is still the best single model on the board estimate.) A large-delta Huber is L2
in the bulk and L1 in the tail; on a task where 0.34 % of rows carry 67 % of the
squared error, that trades a little tail accuracy for a lot of stability in the
two million ordinary flights, and the trade pays. It also needed by far the
largest LightGBM budget (3,225 / 2,100 rounds against 175 / 350), because a
clipped gradient takes smaller steps — which is the same reason CatBoost needed
thousands, and a reminder that budget and mechanism are not separable here.

### What the monotone arms actually say

The assignment asked for monotone constraints on the congestion columns, and
SPEC.md argued they can only be asserted where the model predicts taxi time,
not the recording residual. Both variants were run, and the two clean
one-factor comparisons are:

| comparison | effect on board estimate |
|---|---|
| `L1-plain` → `L3r-monores` (add monotone, residual target both) | **+0.8 s (worse)** |
| `L3r-monores` → `L3-monoraw` (residual → raw target, monotone both) | **+2.0 s (worse)** |

So the constraint is mildly negative wherever it is applied, and the raw target
costs 2 s on top. My SPEC reading was right about the direction — asserting
congestion physics on a recording residual is not free — but wrong about the
size: it costs 0.8 s, not enough to justify the separate raw-target arm the
reading motivated. The honest summary is that the monotone idea is a small
negative on this frame, and both arms stay only as blend members, where a
constrained model is a differently-wrong model and that is worth something.

## Finding 15 — the final blend: 8.4 s on the board estimate, 11.8 s on july

Seven members, weights rotated so a fold's weights are fitted only on the other
validation months, and `july`/`july_loo` never used for each other.

Residual correlation on july:

|  | L0-lin | L1-plain | L2-hub | L3-monoraw | C-cat | X-xgb | T-tail |
|---|---|---|---|---|---|---|---|
| L0-lin | 1.000 | 0.960 | 0.937 | 0.953 | 0.905 | 0.866 | 0.859 |
| L1-plain | 0.960 | 1.000 | 0.942 | 0.965 | 0.888 | 0.858 | 0.857 |
| L2-hub | 0.937 | 0.942 | 1.000 | 0.932 | 0.897 | 0.844 | 0.838 |
| L3-monoraw | 0.953 | 0.965 | 0.932 | 1.000 | 0.894 | 0.868 | 0.855 |
| C-cat | 0.905 | 0.888 | 0.897 | 0.894 | 1.000 | 0.877 | 0.863 |
| X-xgb | 0.866 | 0.858 | 0.844 | 0.868 | 0.877 | 1.000 | 0.842 |
| T-tail | 0.859 | 0.857 | 0.838 | 0.855 | 0.863 | 0.842 | 1.000 |

**The block structure is the lane's answer in one picture**, and it holds on
every fold — within-LightGBM correlations always exceed cross-library ones:

| fold | within LightGBM | across libraries | blend gain (shipped) |
|---|---|---|---|
| july | 0.931 – 0.965 | **0.838 – 0.905** | 11.8 s |
| july_loo | 0.930 – 0.971 | **0.841 – 0.912** | 11.2 s |
| december | 0.946 – 0.976 | 0.873 – 0.945 | 7.9 s |
| january | 0.973 – 0.990 | 0.944 – 0.974 | 4.2 s |

Changing the loss, the leaf model or the constraints inside one library buys
little; changing the library buys, on the two July folds, about as much
disagreement as changing the entire feature set did between the two lineages
(0.867). **But the levels are strongly fold-dependent and an earlier version of
this section quoted july's alone.** On january every pair is above 0.94 — the
members are nearly redundant there — and that is exactly the fold where the
blend gains least (4.2 s against 11.8 s on july). The ordering of the blocks is
the robust finding; the numbers are a property of the month as much as of the
learners.

| blend (rotated) | july | july_loo | december | january | board est. |
|---|---|---|---|---|---|
| best single member per fold | 320.6 | 310.4 | 222.9 | 347.6 | 327.9 (`L2-hub`) |
| flat average of all seven | 314.6 | 304.8 | 218.0 | 345.2 | 323.3 |
| NNLS, one global vector | 308.8 | **299.2** | 215.0 | 343.4 | **319.5** |
| NNLS per regime, shrinkage by inner rotation | 310.1 | 300.1 | 214.7 | 343.1 | 319.5 |
| OLS, one global vector | 307.8 | 299.6 | 214.9 | 343.0 | 319.5 |
| OLS per regime, shrinkage by inner rotation | **307.7** | 299.8 | **214.6** | **342.7** | 319.5 |

Gains over the best single model on each fold, **for the shipped estimator**
(NNLS, one global vector): **11.8 / 11.2 / 7.9 / 4.2 s**, and **8.4 s on the
board estimate** (327.9 → 319.5). That is inside or above the 6-to-13 / 4-to-8 /
2-to-5 s range Finding 1 predicted, on three of the four folds.

(Corrected after review. An earlier version of this line read 12.9 / 11.2 / 8.3
/ 4.9 s, which took each fold's best cell across blend variants rather than
reading one estimator down the column — 12.9, 8.3 and 4.9 are OLS-per-regime's
cells. Taking a per-fold maximum over variants is exactly the selection this
experiment's rotation exists to avoid, and the shipped-estimator row is the
honest one.)

The four fitted variants land at 319.52 to 319.85 on the board estimate — they
differ by at most 1.1 s on any fold, which is inside this harness's run-to-run
spread. The
shipped blend is therefore **NNLS with one global weight vector**, chosen for
conditioning and not for its score: non-negative weights on members correlating
above 0.9 are the stable choice, and the per-regime and OLS variants buy nothing
that survives the noise. Per-regime weighting is again level with global rather
than ahead of it (Findings 7 and 8 unchanged by the corrected members).

### The seven-member weights, and what they say

Fitted by NNLS on all 510,096 OOF rows of `july_loo` + `december` + `january`:

| member | weight |
|---|---|
| `T-tail` (`prc.tail_model`, the second lineage) | **0.404** |
| `L2-hub` (LightGBM, Huber) | **0.333** |
| `C-cat` (CatBoost) | **0.235** |
| `L0-lin` (LightGBM, linear leaves) | 0.030 |
| `L1-plain`, `L3-monoraw`, `X-xgb` | **0.000** |

These are the **seven-member** weights, recorded here because they are what the
correlation structure looks like as a decision; they are *not* the weights in
the shipped candidate, which has three members at T-tail 0.475 / L2-hub 0.431 /
L0-lin 0.095 (`blend-config-exp-008-blend3.json`). Corrected after review, which
noted the earlier heading called these "the shipped weights" and that they had
no durable address; they are now written to `blend-config-exp-008-seven.json`.

The weights sum to 1.001 without being constrained to, which is a small
reassurance that the members are unbiased. Three members are dropped outright,
and the survivors are one from each *library* plus the other lineage — the same
block structure the correlation matrix shows, now expressed as a decision. The
casualty worth noting is `X-xgb`: it holds the lowest correlations in the whole
matrix and still gets zero weight, because it is 30 s behind on july and 26 s
behind on july_loo. **Diversity does not buy a place in a blend on its own; a
member has to be both different and good.** That is why the corrected CatBoost
matters so much here — before the budget fix it was different but not good, and
it would have been dropped too.

## Finding 16 — the MLP is not built, and the gate is the reason

The assignment makes the MLP conditional: run it *"only if the tree trio leaves
residual correlation above 0.9; report why either way."* The trio's pairwise
residual correlations on july are:

| pair | correlation |
|---|---|
| LightGBM (`L0-lin`) vs CatBoost (`C-cat`) | 0.905 |
| CatBoost vs XGBoost (`X-xgb`) | 0.877 |
| LightGBM vs XGBoost | **0.866** |

Two of the three pairs are below 0.9 and the mean is 0.883 — **but that is july
alone, and the gate is fold-dependent.** Corrected after review, the trio's
minimum pairwise correlation by fold is:

| fold | trio min | trio mean | gate at 0.9 |
|---|---|---|---|
| july | 0.866 | 0.883 | closed |
| july_loo | 0.877 | 0.881 | closed |
| december | **0.930** | 0.932 | **open** |
| january | **0.968** | 0.970 | **open** |

So the gate opens on two of four folds, including january, which is 44 % of the
board mix. **The first argument for skipping the MLP does not survive**, and the
earlier claim that "the gate does not open" was a one-fold reading. Two of
july's three pairs also involve `X-xgb` = `X1-d8-july` at 75 rounds, an arm this
file has already withdrawn as unfair to XGBoost, so the fold where the gate looks
closed is also the fold with the weakest evidence.

What does survive is the second reason, and it comes from the blend itself
rather than from the gate. NNLS gives `X-xgb` **zero weight**
despite its holding the lowest correlations in the whole matrix, because it is
30 s behind on july and 26 s behind on july_loo. An MLP on this frame would
almost certainly be more different still and also weaker still — my smoke test
reached july 346.7 after two epochs against the reference's 324.1 — and a member
that is different but not good is exactly what this blend has just demonstrated
it will discard. Diversity is necessary but not sufficient.

**So the honest verdict is: the assignment's gate is met on two folds and the
MLP was still not run, on a judgement about member strength rather than about
diversity.** That is a decision the desk can reverse cheaply, and the code is
ready for it.

The MLP is **built and working**, not skipped for lack of code:
`learners.MlpLearner` (embeddings for the twelve categoricals, numerics
standardised with explicit missing indicators, residual target in seconds,
OneCycle schedule, early stopping on the last training month with epochs as the
frozen unit) runs end to end on GPU and is smoke-tested. It is a queued
follow-up with a concrete revival condition: **if a future member set has its
weakest weighted member within about 10 s of the best, an MLP at 8 to 12 epochs
becomes worth its 20 minutes**, because at that point being different starts to
pay again.

## Submission candidate

`prc/blend.py` (VERSION 3) turns member prediction files plus a weight vector
into a candidate, so a later candidate is a config change and not new code:
`make_candidate.py <members.json> --ranking <ranking.json>` fits the shipped
weights on every available OOF month (each month represented once by its
ranking-analogue fold, so 2025-07 is not counted twice) and writes the parquet
through `prc.data.write_submission`.

**Candidate `exp-008-blend3`** — three members, written and verified:

| field | value |
|---|---|
| path | `submissions/candidates/exp-008-blend3.parquet` |
| SHA-256 | **`26c40e69e0a8f3b018d2eb9b3c8d2c64e8ce7df5167d7aa194a038b997f9a231`** |
| rows | 344,841, IDs identical to the template and in template order |
| members and weights | `T-tail` 0.475, `L2-hub` 0.431, `L0-lin` 0.095 (NNLS on 510,096 OOF rows) |
| predictions | all finite, min 60.081, median 962.1, p99 2,466.8, max 118,994.2, none below the floor |

(This table was rebuilt after review round 1 and **corrected in place after round
2**, which found it still carrying the superseded build's hash and statistics.
The first build was `df4570d2…` with min 60.0 / p99 2,466.1 / max 118,964.1; it
is superseded by the row above and by a correction row in `docs/LEDGER.md`.)

Its rotated four-fold numbers, the same protocol as every blend above:

| fold | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|
| july | 310.9 | 271.9 | 2586.8 | 1259.8 | 3829.4 |
| july_loo | 300.0 | 262.6 | 2488.4 | 1188.6 | 3615.3 |
| december | 216.1 | 200.6 | 2031.9 | 728.0 | 2308.5 |
| january | 344.3 | 207.0 | 5986.3 | 2856.5 | 3881.4 |

Board estimate **320.4**, against 327.9 for the best single model and 334.5 for
the exp-001 lineage refitted here. The four-member candidate including CatBoost
scores 319.5 on the same rotation and is blocked only on CatBoost's twelve-month
fit; this three-member candidate exists so that a valid, verified artifact is on
disk regardless of how that queue finishes.

### A defect worth recording, because it cost an hour and looked like something else

`make_candidate.py` put exp-001's directory ahead of its own on `sys.path` and
then did `import blend_eval`. exp-001 has a `blend_eval.py` too, so the import
resolved to *that* file and ran its module-level code, which prints
`no paired predictions yet` and exits 1. Every invocation therefore failed with a
message that appears nowhere in this experiment's source, which I first read as
an external guard on candidate creation and went looking for a hook. The fix is
one line of ordering, and the lesson is specific: **in this repo an experiment
directory on `sys.path` is a namespace collision waiting to happen**, because
sibling experiments use the same obvious file names (`run.py`, `blend_eval.py`,
`report.py`, `guard.py`). Any script importing across experiments should put its
own directory first, and `frames.py`, `learners.py` and `blend_eval.py` here
already do.

## What this lane established, and what it costs to keep

**The lane's premise held, and more strongly than predicted.** Structurally
different learners on a byte-identical feature frame correlate 0.84 to 0.91 —
about what changing the entire feature set bought between the two existing
lineages (0.867). Within one library, changing the loss, the leaf model or the
constraints moves the correlation only from 0.93 to 0.97. **The library is the
diversity axis; the hyperparameter is not.**

**The blend is worth 8.4 s on the board estimate** (327.9 → 319.5 with all
members, 320.4 for the shipped three-member candidate), and the members that
earn weight are one per library plus the second lineage. Fitted weights beat a
flat average by 1.4 to 3.9 s — the opposite of exp-001's two-member `c7` result,
because with members of visibly unequal strength there is something to fit.

**Per-regime weighting does not pay,** honestly evaluated, and the reason is
specific and re-testable: the best member in a regime is a property of the month
rather than of the regime, so weights fitted on other months do not transfer.
The machinery is in `prc.blend` and costs nothing to keep; its revival condition
is more validation months, or a member set whose regime ordering is stable
across the ones we have.

**Diversity alone does not earn a place in a blend.** `X-xgb` holds the lowest
correlations in the matrix and gets exactly zero weight, because it is 26 to
30 s behind on the folds that matter. This is the finding that kills the MLP arm
(Finding 16) and it is the one to remember when the next worker proposes a
member: *different and good*, not *different*.

### The thing that reaches past this lane

Two of the fourteen findings are about the blend. The largest single number in
the experiment is not: **18.9 s on `july_loo` from a frozen tree budget**,
because the instrument that chooses those budgets — early stopping on one
held-out month's RMSE with a patience of 100 — is biased low, by up to
twelvefold, on a metric where 0.34 % of rows carry 67 % of the squared error.
Every lane in this repo that froze a count that way is comparing its mechanisms
through a confound roughly four times larger than the effects it is testing.
Re-running one arm at a corrected patience costs about 25 GPU-minutes.

### Known limitations of what is shipped

1. **The candidate's own members may still be under-budgeted.** `L2-hub` and
   `L0-lin` carry patience-100 counts (2,100 and 350 on the eleven-month folds).
   `L2-hub`'s best iteration was 1,926, so its patience was 5 % of its
   trajectory — the same ratio that cost CatBoost twelvefold. A patience-900
   selection for it is running; if it comes back much larger, the shipped
   candidate is leaving score on the table and the fix is a refit, not a
   redesign.
2. **`C-cat`'s july fold mixes budgets** (a patience-100 six-month count with
   patience-900 eleven-month ones). `july` is the pessimistic fold and is not in
   the board estimate, and `run_gpu6.sh` is replacing it, but until it lands the
   july column of any table containing `C-cat` should be read with that in mind.
3. **The seed-bag curve is scoped** below what the assignment asks — see the
   seed-bagging section for the numbers actually measured and the scope.
4. **`X1-d8`'s july fold used 75 rounds**, a count from the same broken
   instrument. It is the worst arm in the table and gets zero blend weight, so
   nothing downstream depends on it, but its row is not a fair reading of
   XGBoost on this frame. The corrected `X2-d8p` reads july_loo 337.1,
   december 228.9, january 356.7 (board estimate 345.9) and is the row to quote
   for XGBoost.
5. **A selection effect the SPEC's blanket claim overstates away.** No label
   reached any fit, and every round count was chosen on training months only —
   that part is verified. But the *decision* to change the patience from 100 to
   900, and to replace `C1`/`X1` with `C2`/`X2`, was prompted by scored-fold
   RMSE (Findings 6, 12, 14). The counts are honest; the choice of which
   protocol to ship was informed by the folds, so the four-fold numbers carry a
   small optimistic bias that "no validation month is touched at any point"
   does not capture. The board is the instrument that settles it.

### The truncation is on both fold families, not just the eleven-month one

| CatBoost selection | patience 100 | patience 900 | ratio |
|---|---|---|---|
| six-month family (`july`, early-stop 2025-06) | best 2,969 → frozen 3,625 | best **8,703** → frozen **10,650** | **2.9×** |
| eleven-month family (`december`, early-stop 2025-11) | best 405 → frozen 450 | best **4,879** → frozen **5,325** | **12×** |

Both readings sit close to their search caps (8,703 against a 9,000 cap; 4,879
against 5,000), so **both are still lower bounds**. The eight-and-a-half-fold gap
between the families that Finding 10 attributed to training-set size shrinks to
about twofold once both are measured with a patience that does not fire on tail
noise — so even that conclusion was partly an artefact of the instrument, and the
honest statement is now: *the six-month fold wants somewhat more trees than the
eleven-month fold, and the earlier factor of eight was mostly truncation.*

This changes nothing about the shipped candidate, whose CatBoost member uses the
corrected eleven-month count, but it does mean the `C-cat` july column and every
`C1-d8`/`X1-d8` row remain what they were labelled: measurements through a broken
instrument, kept for the record and not for comparison.

## Review round 1 and what it changed

Reviewer: fresh-context `claude-opus-5` at `xhigh` (the desk substituted an Opus
one-shot for Sol while Sol was not available). Full text in `review/round1.txt`. It raised **three blocking findings,
all real, all fixed**, and nine non-blocking ones, all acted on. It also verified
clean, by recomputation rather than by reading: leakage (v2 reads no block or
taxi time; mask-invariance re-run on two months), the training-only anchors and
guard bounds, that `--select` never touches a validation month, the rotation
(neither July fold ever fits the other), that 2025-07 enters the candidate fit
once (510,096 = 190,713 + 165,677 + 153,706), the candidate parquet against the
template, the `C-cat`/`X-xgb` assembly by SHA, and that every blend number in
Findings 7 and 15 reproduces from the stored JSON.

**B1 — the shipped candidate was not the estimator that was evaluated.**
*(Round 2 found the `run_submission` half of this fix had not yet executed — see
the round-2 section below. The half that protects the shipped candidate,
member preparation inside `prc.blend.build_candidate`, did run and is verified.)*
`run_fold` floors at 60 s and applies the residual guard before writing OOF;
`run_submission` wrote ranking predictions raw. So the weights were fitted on
floored, guarded members and applied to unfloored ones — and the error was
*asymmetric*, because `T-tail` floors inside `prc.tail_model` and the two
LightGBM members do not. Confirmed on the artifacts: `L0-lin` had **78** ranking
rows below the floor (minimum **−4,962 s**), `L2-hub` **77** (minimum −2,383 s),
`T-tail` none. Fixed in two places, both idempotent: `run_submission` now applies
the same floor and guard as `run_fold` (guard bounds from all twelve 2025 months,
no ranking row enters them), and `prc.blend.build_candidate` prepares every
member the same way before blending, so a candidate built from any member file is
correct regardless of which writer produced it.

**B2 — the largest member served a different model from the one that earned its
weight.** `prc.tail_model.NOAC_VERSION` moved from `v1` to `v2` at 09:24 today,
after this experiment's weights were fitted. `tail_ranking.py` constructed
`TailModel()` with nothing pinned, so it served **v2** while the OOF that fits
its 0.475 weight is **v1** — and the two differ precisely on the noac rows that
carry a fifth of the squared error. Fixed by pinning `noac_head="v1"` to match
the OOF, with a comment saying that moving to v2 means refitting the weights on a
v2 OOF, not just unpinning. The member was rewritten and its hash changed from
`72a38085…` to `d1255faf…`, which is itself the proof that the two versions were
being mixed.

**B3 — seed bagging was claimed but not delivered.** The section is below, with
the numbers and the scope.

The candidate was rebuilt with both fixes and re-verified. **Its SHA-256 is now
`26c40e69e0a8f3b018d2eb9b3c8d2c64e8ce7df5167d7aa194a038b997f9a231`**, superseding
`df4570d2…`; 344,841 rows, IDs identical to the template and in template order,
all finite, min 60.1, median 962.1, max 118,994.2, none below the floor.

The nine non-blocking findings are corrected in place, each with a pointer at the
line that carried the refuted claim: the headline gains now read down the shipped
estimator's column instead of taking each fold's best cell across variants
(11.8 / 11.2 / 7.9 / 4.2, not 12.9 / 11.2 / 8.3 / 4.9); the correlation claim is
given per fold rather than from july alone; the MLP gate is shown to *open* on
december and january; a reversed L2-hub/T-tail comparison is fixed; the
seven-member weights are labelled as such and given a durable address; the
COMMON.md decomposition table is added, generated by `report.py`; `X2-d8p` gets
its row; `prc/blend.py`'s docstring carries its version history; and the
selection effect in choosing *which protocol to ship* is stated as a limitation.

## Finding 17 — seed bagging (REWRITTEN after review round 2)

**The first version of this finding was wrong in three ways and is withdrawn in
full.** It reported the wrong arm, its cited artifact was overwritten by a
concurrent job while the review was running, and its key comparison was
inverted. What follows is measured on artifacts that are now stable (every
seed-bag job stopped before recomputation) and states the like-for-like
comparison correctly. `review/round2.txt` has the full account.

### The arm the assignment asks for

`assignments/diversity.md` asks for the **best two learners**. By board estimate
the ranking is `L2-hub` 327.9, `C2-d8p` 330.2, `L1-plain` 330.2, `L3r` 331.0,
`L3-monoraw` 333.0, `L0-lin` 334.5 — so the best LightGBM is `L2-hub`, which is
also the candidate's 0.431-weight member. `SB-L2hub`: five seeds, `july_loo`,
2,100 frozen rounds, RMSE averaged over all subsets of each size.

| seeds | mean RMSE | best subset | worst subset | marginal gain |
|---|---|---|---|---|
| 1 | 311.18 | 310.10 | 312.18 | – |
| 2 | 309.94 | 309.09 | 310.95 | **+1.24** |
| 3 | 309.52 | 308.85 | 310.07 | **+0.42** |
| 4 | 309.31 | 309.05 | 309.57 | +0.21 |
| 5 | 309.19 | 309.19 | 309.19 | +0.12 |

**The desk's pricing answer, for the member that actually ships: the second seed
buys 1.24 s, the third 0.42 s, the fourth 0.21 s, the fifth 0.12 s — 1.99 s in
total from five seeds.** Two seeds capture 62 % of it and three capture 83 %.
Each seed of this arm costs about 5½ minutes, so **two seeds is the sensible
default and three is the most that is defensible**; five buys 0.33 s over three
for twice the compute.

The per-seed spread of this arm is **2.08 s** (310.10 to 312.18), and its
post-processed seed-pair residual correlations are **0.983 to 0.985**. Seeds of
`L2-hub` are nearly interchangeable.

### The correlation comparison, stated correctly this time

The withdrawn version claimed seed residuals correlate *less* than different
learners do. That compared **raw** seed residuals against **floored and guarded**
cross-learner residuals, on two different folds. Like for like — post-processed,
both on `july_loo`:

| pair | correlation |
|---|---|
| seeds of `L2-hub` with each other | **0.983 – 0.985** |
| seeds of `L0-lin` with each other | 0.902 – 0.959 (median 0.929) |
| `L0-lin` vs `C-cat` (across libraries) | **0.883** |
| `L2-hub` vs `T-tail` (across lineages) | **0.862** |

**Seeds of one model correlate more than two different learners do, at every
pair.** That is the expected result and it *strengthens* the lane's conclusion
rather than undermining it: re-seeding one learner is not a substitute for
changing the learner. The withdrawn sentence said the opposite and was the
load-bearing sentence of the section.

Consequently the earlier claim that "roughly half the blend's gain is available
from re-seeding a single learner" is also withdrawn: 1.99 s against the blend's
11.2 s on `july_loo` is **about a sixth**, not a half.

### A separate and more serious finding: the linear-leaf arm is not reproducible

`SB-L0lin` was run twice with identical seeds, identical rounds, identical data
— differing only in `PRC_THREADS` (8 and 12). Per-seed `july_loo` RMSE:

| seed | 8 threads | 12 threads | delta |
|---|---|---|---|
| 7 | 323.76 | 321.93 | −1.83 |
| 107 | 334.44 | 335.68 | +1.24 |
| 207 | 332.31 | 329.11 | −3.20 |
| 307 | 351.55 | 345.64 | −5.91 |
| 407 | 336.23 | 351.63 | **+15.40** |

`learners.py` sets `deterministic=True` and `force_row_wise=True`, but the `lin`
variant also sets `linear_tree=True`, and LightGBM's determinism guarantee does
not cover the per-leaf linear solve across thread counts. **So `L0-lin` is not
reproducible across a change in thread count, by up to 15.4 s on one seed.**

This means the withdrawn "27.8 s seed spread" was never purely seed noise: it
mixed seed variation with run-to-run non-reproducibility of the linear-leaf
arm. The honest decomposition, from the two arms measured here:

- `L2-hub` (plain trees): seed spread **2.08 s**, correlations 0.983+ — well
  behaved.
- `L0-lin` (linear leaves): apparent spread 27.8 s at 8 threads and 29.7 s at
  12, **but not reproducible across thread counts at all**, so its spread cannot
  be attributed to the seed and its single-seed level cannot be trusted to
  better than several seconds.

The control is `prc.tail_model`, which sets full determinism and uses no linear
trees: the reviewer reproduced its ranking predictions to 7.8e-06 s across
`n_jobs` 8 against 16.

**What this costs the rest of the file.** Every `L0-lin` number here is a single
run at one thread count, and the linear-leaf arm's non-reproducibility is a real
caveat on those rows — including the Finding 3 claim that this harness
reproduces exp-001's `H0-s7` to 1.3 s, which is now known to be within the arm's
own run-to-run spread rather than a tight agreement. The plain-tree arms
(`L1-plain`, `L2-hub`, `L3-*`), CatBoost and XGBoost carry no such defect. The
practical recommendation for this repo is narrower than the withdrawn version's:
**do not compare linear-leaf arms across runs at all without pinning threads,
and prefer plain trees where an arm's level has to be trusted.**

## Review round 2 and what it changed

Same reviewer and protocol (fresh-context `claude-opus-5` at `xhigh`, the desk's
substitute while Sol was not available). Full text in
`review/round2.txt`. It raised **six blocking findings**, all real. This is the
second and final round permitted by COMMON.md, so the findings are answered here
rather than sent back.

**B1 — the cited seed-bag artifact was overwritten mid-review by a concurrent
job of mine, and the new data reversed the finding.** Two seed-bag scripts were
alive at once (`run_seedbag3.sh` and `run_seedbag4.sh`, at 12 and 8 threads),
both writing `oof/SB-L0lin-july_loo.parquet`, so `result-SB-L0lin.json` recorded
329.71 while the parquet it names yielded 334.03. Every seed-bag job is now
stopped, the numbers above were recomputed against the stable files, and
Finding 17 is rewritten. **The underlying process failure is mine and worth
naming: I let two jobs share an output path.** `run.py` writes an arm's OOF
under `oof/<name>-<fold>.parquet` with no interlock, so two runs of the same
`--name` race silently; the fix for any future worker is to make the arm name
unique per launch or to refuse to overwrite.

**B2 — the "27.8 s seed spread" was not seed noise.** The same five seeds moved
up to 15.4 s between an 8-thread and a 12-thread run of the identical
configuration, because the `lin` variant sets `linear_tree=True` and LightGBM's
`deterministic=True` does not cover the per-leaf linear solve across thread
counts. Rewritten in Finding 17 as a *reproducibility* finding, which is a
worse problem than the one I originally claimed.

**B3 — the seed-bag measured the wrong arm.** The assignment asks for the best
two learners; `L0-lin` is sixth of nine. `SB-L2hub` — the best learner and the
candidate's 0.431-weight member — was already on disk, five seeds, and went
unreported. It is now the primary result, and it changes the desk's answer
substantially: 1.99 s from five seeds, not 5.95 s.

**B4 — the correlation comparison was inverted.** Raw seed residuals on one fold
were compared against floored-and-guarded cross-learner residuals on another.
Like for like, seeds correlate *more* than different learners (0.983–0.985 for
`L2-hub`'s seeds against 0.862–0.883 across libraries). Corrected in Finding 17;
the correction strengthens the lane's central claim rather than weakening it.

**B5 — the candidate table and ledger row carried the superseded SHA-256.** The
table is corrected above and a superseding ledger row is appended (the ledger is
append-only, so the stale row stands with a pointer rather than being edited).

**B6 — the `run_submission` half of the B1 fix has never executed.** Confirmed:
`L0-lin`, `L2-hub` and `C2-d8p` ranking parquets all predate the fix and hold
78, 77 and 73 sub-floor rows (minima −4,962.3, −2,382.8, −2,775.0), and
`guard_bounds.json` has no `_full12` key, which proves the new branch has not
run. **The record previously asserted a state no artifact had.** The shipped
three-member candidate is nevertheless correct, because the other half of the
fix — member preparation inside `prc.blend.build_candidate` — did run and the
reviewer verified it reproduces the shipped SHA exactly, with 98 rows and up to
742.6 s differing if it is disabled. The correct statement is therefore: **the
defect is fixed at the blend boundary, which is what protects the candidate, and
is fixed in `run.py` only for runs launched after 11:48 — of which there have
been none.** Any future ranking fit will floor and guard at source; the existing
member files remain unfloored on disk and must keep going through
`build_candidate`.

### Non-blocking findings from round 2 worth carrying forward

- **The v1 pin costs 0.64 s, and the v2 OOF already exists.** Substituting
  `experiments/exp-009-noac2/oof/exp-009-noac2-oof.parquet` as the `T-tail`
  member and re-rotating gives board estimate **319.73 against the shipped
  320.37**, with weights moving to T-tail 0.500 / L2-hub 0.416 / L0-lin 0.085.
  Pinning v1 was the right call for consistency with the fitted weights, but the
  cheaper and better repair was available: refit the weights on exp-009's v2 OOF
  and re-run `tail_ranking.py` without the pin, about four minutes. **This is
  the single highest-value follow-up in the lane.**
- **`L2-hub`'s tree budget is truncated 4.2×.** `select-S-lgb-hub-pat900.json`
  landed at 11:47 with `best_iteration` 7,990 → 8,725 rounds against the shipped
  2,100. Finding 14 measured 18.9 s on `july_loo` from exactly this instrument on
  CatBoost, so the candidate's 0.431-weight member is very likely leaving
  several seconds on the table. Limitation 1 above said this selection "is
  running"; it had already landed and I did not check.
- Two cells of the blend-variant board column read 319.5 where the artifact says
  319.8 and 319.6; the prose beneath them is right.
- "Fitted weights beat a flat average by 1.4 to 3.9 s" is the four-member
  number. On the **shipped three-member** candidate the margin is 0.34 / 1.86 /
  0.65 / **−0.22** — the flat average is *better* on january, 44 % of the board
  mix.
- `prc/blend.py`'s VERSION was not bumped for the `prepare_members` default, and
  `blend-config-*.json` does not record it, so the stored config does not
  identify which estimator built the file.
- `SPEC.md` still says `prc/blend.py` "starts at VERSION = 1" and "no validation
  month is touched at any point"; both are corrected in RESULTS but not in SPEC,
  so my round-1 claim that all nine non-blocking findings were "corrected in
  place" was itself inaccurate for two of them.

**Verdict: `reviewer: claude-opus-5-xhigh-2026-09-05 (substitute), two
rounds, round 2 NEEDS REVISION on six blocking findings — all six answered
above; Finding 17 withdrawn and rewritten, the candidate hash corrected, and the
`run_submission` claim restated to what the artifacts support.** The two fixes
that touch the shipped artifact were verified correct by recomputation in both
rounds; the candidate itself reproduces its SHA and matches the template.

### Seed bagging on a second fold, and what the two folds together say

`SB-L0lin-dec` (linear leaves, five seeds, `december`, 350 rounds) completed
after the round-2 recomputation and is included because it is the only seed-bag
curve on a calm month:

| seeds | `SB-L2hub` july_loo | `SB-L0lin` july_loo | `SB-L0lin-dec` december |
|---|---|---|---|
| 1 | 311.2 | 336.8 | 231.2 |
| 2 | 309.9 | 334.2 | 226.6 |
| 3 | 309.5 | 333.5 | 225.0 |
| 4 | 309.3 | 333.5 | 224.2 |
| 5 | 309.2 | 334.0 | 223.7 |
| total gain | **1.99** | 2.77 | **7.53** |

Three readings, and they do not agree, which is itself the result:

- On the arm that ships (`L2-hub`, plain trees), five seeds buy **1.99 s** and
  the curve is smoothly diminishing. Two seeds is the sensible default.
- On the linear-leaf arm the curve is **not monotone** — the fifth seed makes it
  worse — which is the reproducibility defect of the section above showing
  through, not a property of bagging.
- On `december` the same linear-leaf arm gains **7.53 s**, nearly four times its
  july_loo figure, and monotonically. A calm month has less tail noise for the
  average to fight, so bagging looks better there.

So the honest answer to "price compute against score" is **fold- and
arm-dependent, between 2 and 7.5 s for five seeds**, with two to three seeds
capturing most of it in every case. The single number the withdrawn version gave
was not robust across either axis.
