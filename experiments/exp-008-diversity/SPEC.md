# exp-008 diversity — spec

Worker `diversity`, tier hard, exp-id `exp-008`, assignment `assignments/diversity.md`.
Metric: RMSE in seconds on the four folds `july`, `july_loo`, `december`,
`january` of `prc.data.fold_frames`, with the COMMON.md tail decomposition
(overall / target ≤ 3600 s / target > 3600 s / no-NM rows), plus the noac split.

## My reading of the assignment, and the readings I rejected

I read the assignment as: hold the feature frame fixed and vary only the
learner, so that the disagreement between members is attributable to the
learner; produce four-fold out-of-fold predictions for each member on one
shared row alignment; then fit per-regime blend weights on the folds a member
is *not* being scored on, and compare against the best single model. The
deliverable is the blend and the machinery to re-blend, not a new feature idea.

**The reading I rejected was to let each learner bring its own preprocessing.**
It is tempting — CatBoost is better with more categorical interactions,
XGBoost likes one-hot for low-cardinality columns, an MLP wants log-scaled
counts — and every one of those would raise that member's own score. But then
a blend gain could not be attributed: two members would differ in learner *and*
in frame, and the residual-correlation matrix, which is the whole instrument
for deciding whether this lane is worth more compute, would measure the wrong
thing. So the frame is retro-v1's frozen v2 (54 columns, 12 object-dtype
categoricals, schema fingerprint `cf956e1b26b3c11f`, numeric fingerprint
`605fe71f984628d5` on 2025-07, verified with `frame_contract.py` at 08:30 on
2026-09-05 before any model was fitted). What each learner *is* allowed to own
is its categorical convention, because that is not preprocessing, it is the
algorithm: LightGBM takes integer codes and splits on them, CatBoost takes
strings and builds ordered target statistics, XGBoost takes a `category` dtype
with native categorical splits, the MLP takes an embedding index. Encoding a
categorical the same way for all four would mean giving three of them
LightGBM's algorithm.

**The second reading I rejected was to reuse the existing OOF files as
members.** exp-001 has 40 named OOF sets and exp-002 has one. Reusing them
costs nothing and would produce a blend table today. It is wrong for two
reasons that only became visible on reading `climb.log`: none of exp-001's OOF
covers `july_loo`, which is half of the board estimate
(`RANKING_ANALOGUE_FOLDS`), and the 06:34 `protocol-halt` line withdraws every
exp-001 arm whose round count was chosen by an independent early stop, which is
all of them. A blend built on members with per-fold-chosen round counts inherits
that instability into the fitted weights. So every member here is refitted under
one protocol — fixed rounds per fold family — including the LightGBM reference,
which is re-measured rather than quoted. That re-baselining is stated in
RESULTS.md rather than assumed.

**The third reading I rejected was one shared noac head for all members.** It
makes the learner comparison cleaner: the 0.18 % of rows that carry 20-25 % of
the squared error would be identical across members, so the table would read as
a pure main-model comparison. I rejected it because the blend is specified as
*per regime* and `noac` is one of the four regimes: a shared head would make
every member's noac column identical, the noac regime's weights unidentifiable,
and the blend's largest single opportunity unreachable. Each arm therefore
builds its own head with its own algorithm, anchored on take-off minus schedule
exactly as exp-001's head is. The cost is that a learner-vs-learner row mixes
main model and head; the four-way decomposition in every table separates them.

## What the monotone constraint can and cannot be asserted on

The assignment asks for "monotone constraints where the physics is monotone
(queue ahead, occupancy)". The physics claim is about **taxi-out time**: more
departures queued ahead of you, and more movements in your window, is never
faster. Under this lineage's residual parameterisation the model does not
predict taxi-out time. The target is

    r = y − (take-off − AOBT) = AOBT − BLOCK,

the signed gap between the Network Manager off-block and the recorded block
time — a *recording* quantity. Congestion has no asserted sign on it, and
constraining it would impose physics the object does not have. So the monotone
sibling is the one arm that fits the **raw** target in seconds, where the
assertion is true; that also makes it the most structurally different tree
member, which is what the lane is for. The residual-target monotone variant is
run as a screen and reported, so the reading is measured rather than argued.

Signs asserted (`learners.MONOTONE`): +1 on `dep_takeoffs_*`, `rwy_takeoffs_*`,
`arr_landings_*` in all three windows and both sides and on `queue_ahead_aobt`;
−1 on `prev_rwy_gap` and `prev_apt_gap` (a longer gap to the movement before you
is never slower). Nothing else is constrained.

## Round counts are fixed per fold family, and chosen once

exp-001's `protocol-halt` (2026-09-05 06:34) established that an independently
early-stopped round count is unstable enough — 125 / 186 / 231 july rounds at
seeds 7 / 17 / 42 on one configuration — that a comparison between two arms so
chosen is a round-count comparison. Every arm here uses a count frozen per fold
family: `m6` for the six-month forward `july` fold, `m11` for `july_loo`,
`december` and `january`. The count is chosen once per arm by `--select`, which
early-stops on the **last training month** of the fold (2025-06 for `m6`,
2025-11 for `m11`) and rescales by `1/(1−held)`; no validation month is touched
at any point, and the same frozen number is then used for all four folds and all
seeds. Chosen counts are recorded in `select-*.json` and in RESULTS.md.

**Corrected after review round 2.** That paragraph is true of every count — no
label and no validation month enters any selection, and the reviewer verified it
by recomputation — but it overstates as a blanket claim about the experiment.
The *decisions* to change the early-stopping patience from 100 to 900, and to
replace the truncated `C1`/`X1` arms with `C2`/`X2`, were prompted by scored-fold
RMSE (RESULTS.md findings 6, 12, 14). The counts are honest; the choice of which
protocol to ship was informed by the folds, so the four-fold numbers carry a
small optimistic selection bias. The board is the instrument that settles it.

## Alignment and the shared-module rule

Members are joined on `(fold, MVT_ID_mvt)`. `prc.tail_model`'s OOF
(`experiments/exp-002-tail/oof/exp-002-tail-oof.parquet`) already carries all
four folds and exactly the departure rows of each validation month, so the join
is total; any member that does not cover a row disqualifies that row from the
blend rather than being filled in.

`prc/blend.py` is new, so it started at `VERSION = 1` and changed nothing
existing. It is now at **VERSION 4** (2: sample-size shrinkage toward pooled
weights, default off; 3: scale-relative ridge and a singular-safe OLS solve;
4: `build_candidate` prepares every member with the 60 s floor and the residual
guard before blending, so a served member is the same estimator as the
out-of-fold member its weight was fitted on — `prepare_members=False` restores
VERSION 3). Corrected after review, which found this sentence still claiming
VERSION 1. Nothing else in `prc/` has a default changed by this experiment; the
one default that moved *underneath* it is `prc.tail_model.NOAC_VERSION`, which
went v1 → v2 after these weights were fitted and is therefore pinned explicitly
in `tail_ranking.py`. The month
feature cache under `artifacts/exp-008/frames/` is a cache of `build_features`,
not a new definition: `frames.verify_mask_invariance` checks that the
ranking-masked validation month and the unmasked training month give
bit-identical feature columns (54/54, zero differing columns on 2025-07), which
is what makes one cached build per month legitimate.

## Two readings settled during the build

**A round count is not portable between learners, and neither is a head's.**
The frozen counts differ by an order of magnitude between learners (LightGBM
175 rounds on the six-month family, CatBoost 3,625), which is what oblivious
trees at a fixed learning rate look like. The per-learner noac head inherited
LightGBM's 60 rounds from exp-001 for every arm; that was wrong for the same
reason, and it would have shown up as CatBoost being a worse learner on the
rows that carry a fifth of the squared error. `NOAC_ROUNDS` is now per kind and
`C1-d8h60` measures the difference rather than asserting it. The rejected
alternative was to select the head's rounds by early stopping inside the head:
principled, but the early-stop slice would have been the ~90 noac rows of one
training month with a target whose 99th percentile is 11,524 s, which is a
noisier instrument than a declared constant.

**For the MLP, the frozen per-family count is a number of epochs.** The unit of
`--select` elsewhere is a tree and the refit rescales by 1/(1−held) because the
refit sees more rows; an epoch already sees every row, so the MLP's count is
carried across unscaled. Its early-stop set is the same last training month
every other arm uses.

## Reviewer (desk rule, 2026-09-05)

Sol is not available, so this
experiment's fresh-context review is a one-shot Opus run instead of Sol:

    claude --model claude-opus-5 --effort xhigh --dangerously-skip-permissions \
      -p "<review prompt>"

run from the repo root with no `CLAUDE_CONFIG_DIR`, output saved under
`review/`, verdict recorded as `reviewer: claude-opus-5-xhigh-2026-09-05
(substitute)`. The COMMON.md cap is unchanged: at most two rounds, and
write-up findings are not blocking.
