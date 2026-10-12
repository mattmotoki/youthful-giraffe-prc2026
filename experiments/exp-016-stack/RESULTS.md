# exp-016 stack — does the mixing weight want to depend on the row?

Worker `robustness`, 2026-09-05. A learned stacker over the five member OOF sets,
fitted on out-of-fold predictions only, rotated over three folds, against the
shipped `blend-2` rule and the flat four-member average.

**This document was substantially rewritten after review round 1**, which found
three blocking errors in the first version — a false uniqueness claim, an
ablation conclusion drawn from a single seed, and a wrong diagnosis of a failed
arm. All three are corrected below with my own re-measurements, and the review
section at the end says what changed.

## The short version

1. **A stacker is worth 0.49 to 0.74 s of board estimate**, and the four
   configurations measured span only 0.25 s. That range is the finding; the
   choice within it is not supported by three folds.
2. **The gain is mostly re-mixing.** A stacker on the member predictions alone
   captures it on the two folds that make the board estimate (but not on
   december); the row features the desk proposed as gates are a
   **wash on the two folds that make the board estimate** and worth **0.59 s on
   december**, the control month.
3. **The stacker learns to distrust the average where the members disagree.**
   `mem_std` — the dispersion of the five member predictions — is the third
   most-used feature by gain, ahead of three of the five members themselves. A
   constant-weight blend cannot express that.
4. **`noac` must be offloaded**, and for a reason that is not the score: the
   stacker sees 152 noac rows in the smallest rotation's fit set, and exp-008
   already measured that even a *linear* weight vector fitted on ~150
   heavy-tailed noac rows does not transfer.
5. **Two candidates are handed over**, both `keep-noac`, differing only in
   whether the stacker sees the row features: `exp-016-stack-1` at a board
   estimate of **317.280** and `exp-016-stack-2` at **317.529**, against
   candidate 2's 318.017.
6. **The blend owner's repaired winter member absorbs 40 % of the gain.** Swapped
   in, the stacker's advantage over `blend2` falls from 0.737 s to **0.446 s** —
   and the arm choice stops being a coin-flip, because `stack-2` then loses
   july_loo while `stack-1` still wins all three folds. The repaired member has
   no ranking predictions yet, so the candidates here are built on the
   pre-repair one.
7. **The risk this carries is January.** The gain concentrates in rows where the
   members disagree, and *there* its sign flips between folds. `docs/candidate-audit.md`
   measured January 2026 at 7.84 % of rows on envelope-exceeding days against the
   january fold's 0.73 % — the slice that grows ten-fold on the board is the
   slice where this candidate is least stable.

## Round 1

`analysis/rotation.csv`, `logs/rotation.log`. Fit on two folds' OOF, score the
third, never its own. Board estimate is the quadrature of `july_loo` and
`january` at the ranking-month shares, directly comparable to exp-012's 318.017.

| variant | july_loo | december | january | **board** |
|---|---|---|---|---|
| `a2-lgb-resid` | 296.035 | 213.619 | 342.677 | **317.538** |
| `blend2` (shipped) | 295.992 | 214.152 | 343.726 | 318.017 |
| `flat4` | 296.416 | 214.462 | 343.666 | 318.208 |
| `b-ridge-gated` | 299.812 | 214.397 | 343.627 | 319.957 |
| `c-regime-nnls` | 302.014 | 215.219 | 343.641 | 321.116 |
| `a1-lgb-mono` | 605.628 | 346.073 | 450.496 | 542.426 |

**The harness reproduces the shipped rule exactly** — 295.992 / 214.152 / 343.726
against exp-012's 295.9916 / 214.1522 / 343.7263 — so the comparison is against
the real baseline, not a re-implementation of it.

`c-regime-nnls` **loses july_loo and december and ties january** (343.641 against
blend2's 343.726 and flat4's 343.666, i.e. it wins that fold narrowly against
both). The july_loo loss is decisive — +5.6 [+0.5, +12.0] against flat4 — and it
reproduces exp-012's finding from a different direction: four weights per regime
fitted on two months of OOF do not transfer, and `noac` (63–339 rows) is why.

**The round-1 intervals were computed with a loop bootstrap that was later
replaced by a vectorised equivalent** (same estimator, different draw sequence).
Round-1 interval numbers therefore do not reproduce to the last decimal on a
re-run; the RMSEs do. Every interval quoted from here on is from the vectorised
version.

## The direct-target arm: the first version's diagnosis was wrong

`a1-lgb-mono` scored 605.6 on july_loo. The first version of this document said
"a direct-target GBM starts from the label mean and travels by
`learning_rate × rounds`; at 0.03 and 300 rounds it cannot reach a level of
~1,000 s". **That is false, and the review caught it.** Measured on the january
rotation (`logs/verify_r1.log`):

| | value |
|---|---|
| fit-set label mean | 1,011.5 |
| a1's prediction mean | **1,001.6** |
| a constant fit-mean predictor | **605.05** |
| a1 as run (lr 0.03, 300 rounds) | 450.50 |
| a1 "repaired" (lr 0.10, 2,000 rounds) | **426.29** |
| a1 with `init_score = flat4` (lr 0.03, 300 rounds) | **342.81** |

**The arm reaches the level immediately and travels far past a constant** — LightGBM
boosts from the label mean for this objective. It underfits the **mapping**, not
the level, and the budget repair the first version promised does not fix it: at
2,000 rounds it is still 82.6 s worse than `blend2`. What fixes it is starting
from the blend (`init_score = flat4`, or equivalently a residual target), which
lands at 342.81 — in line with every other residual arm.

**So the real finding is about parameterisation, not budget.** A stacker asked to
predict the *level* has to reconstruct ~1,000 s of structure that its inputs
already encode, and at this regularisation (15 leaves, `min_data_in_leaf` 2,000,
`lambda_l2` 100) it cannot. A stacker asked for the *correction to a blend* has
only the correction to learn. That is why every arm in this experiment that works
is a residual arm, and it is a stronger result than the one it replaces.

A process note, because it nearly shipped a fiction: the first version described
two repairs — `fit_lgb2` at lr 0.10/2,000 rounds and an `init_score` form — in the
indicative, and **neither was ever called**. `code/stack2.py` defines `fit_lgb2`
and its `VARIANTS` dict runs `_lgb` instead; the docstring and RESULTS both
described the code that was written rather than the code that ran. Both are now
measured above and `stack2.py`'s docstring is corrected.

## The ablation: the gates are not worthless, and one seed said they were

The first version concluded "the row features contribute nothing" from a
comparison of one seed-7 fit of each arm. Refitting both at five seeds
(`analysis/r1_seed_ablation.csv`) — my own re-measurement, matching the
reviewer's to the decimal:

| arm | july_loo | december | january | **board (seed mean)** |
|---|---|---|---|---|
| `a2` members + gates | 295.956 | **213.614** | 342.630 | 317.474 |
| `a5` members only | **295.937** | 214.199 | **342.612** | **317.456** |
| seed spread, `a2` | 0.151 | 0.104 | 0.100 | – |
| seed spread, `a5` | 0.099 | 0.050 | 0.085 | – |

**Averaged over seeds the gates cost 0.019 s of board estimate — not the 0.091 s
the first version bolded — and that is inside the seed spread.** Seed 7 happened
to be `a2`'s worst of five draws and `a5`'s typical one.

**And on december the gates are worth 0.586 s**, six times `a5`'s own seed spread
there. December carries no weight in the board estimate, so the headline metric
structurally cannot see the one fold where the row features demonstrably carry
information. The defensible statement is therefore:

> On the two ranking-analogue folds the gates are a wash. On the calm control
> month they are worth 0.6 s. The claim that the information is entirely in the
> member disagreement is **not** supported.

What *is* supported is the weaker and still useful version: **a stacker on the
member predictions alone captures the gain on the two folds that make the board
estimate**, so this is predominantly re-mixing rather than a sixth base model. It
is **not** true on december, where at the seed mean `a5` *loses* to `blend2` by
0.047 s and captures none of the gain while `a2` takes 0.538 s of it (round-2
review, N10). The gates-only arm
(`a4`, 318.363) is worse than the flat average it corrects and worse than
`blend2` on july_loo by +0.870 [+0.402, +1.383] — a row-feature model with no
member information is not what is working here.

One honesty note on the labels: `a5` is called "members only" but `n_defined` is
effectively the NM-join indicator, so it carries one row feature (importance
9.9e7, the smallest of the nine). And `a3` differs from `a2` by the monotone
constraint as well as by nothing else, so the round-2 arms are not a single-variable
contrast across the whole set; the two ablation arms `a4`/`a5` are.

## What the stacker uses

Mean gain over fifteen fits (`analysis/a5_importance.csv`): `N1-nmtail` 9.5e8,
`T-tailv2` 9.2e8, **`mem_std` 9.1e8**, `L2-hub` 7.8e8, `mem_range` 5.9e8,
`D1-winter` 4.7e8, `C2-cat` 4.4e8, `mem_mean` 2.4e8, `n_defined` 1.0e8.

**The dispersion of the members is the third most-used feature**, ahead of three
of the five members. That is the mechanism: where the members agree, the flat
average is already right and the stacker leaves it alone; where they disagree it
has something to decide. The desk's monotone constraint costs **0.072 s** against
the arm with the same columns and no constraint (`a3` 317.610 against `a2`
317.538) and is not adopted — it is not harmful, it constrains a model that was
not violating it. A first version quoted 0.16 s, which is `a3` minus the
*members-only* arm and so mixes the constraint with the gate columns — the exact
confound the honesty note above flags (round-2 review, N3).

The gated ridge, re-parameterised so `β = 0` recovers the flat average exactly,
lands at 318.19–318.25 across three penalties and its best value is the most
heavily penalised one: **it converges to flat rather than beating it.** A linear
gate on the weights is the wrong functional form for what the GBM finds.

## The seed is not the result

Five seeds, rotated (`analysis/seed_stability.csv`): spread 0.099 (july_loo),
0.050 (december), 0.085 (january); board 317.447 at seed 7 against **317.446**
for the five-seed bag. The bag ships because it costs five ten-second fits and
removes the question.

The first version explained this as the stacker "inheriting very little of the
members' variance". **That is not measurable here** and the claim is withdrawn:
the member OOF files are fixed bytes pinned by sha256, so member seed variance is
held at exactly zero by construction. What is measured is the stacker's own
fitting seed, and it is small.

## Where the gain is, and the one place the stacker must not go

`analysis/where_the_gain_is.csv`, five-seed bag against `blend2`, positive =
stacker better. Row counts are july_loo's; december and january have 89 and 63
`noac` rows and fall below the table's 200-row floor, so the `noac` line is **one
fold's measurement**, not a mean over three.

| regime | july_loo rows | december | january | july_loo |
|---|---|---|---|---|
| `nm_bulk` | 172,270 | +0.29 | +0.84 | +0.59 |
| `nonm` | 3,595 | **+11.91** | +4.37 | +3.92 |
| `nm_late` | 14,509 | −4.41 | +6.73 | −0.56 |
| **`noac`** | **339** | not measured | not measured | **−18.91** |

| member disagreement | mean rows | december | january | july_loo |
|---|---|---|---|---|
| < 50 s | 22,601 | +0.08 | +0.09 | +0.03 |
| 50–150 s | 100,768 | +0.17 | +0.56 | +0.34 |
| 150–400 s | 40,817 | +0.30 | +0.92 | +0.46 |
| **> 400 s** | **5,846** | **−2.35** | **+5.51** | **−1.28** |

**The gain rises with member disagreement up to 400 s and becomes unstable past
it.** That is the mechanism behaving as it should and then running out of data,
and it is the single most important caveat on this experiment — see the risk
section.

**`noac` is offloaded on a sample-size argument, not a fold score.** The noac
rows in each rotation's fit set are **152 (scoring july_loo), 402 (december) and
428 (january)**, and the shipped three-fold fit contains **491** — the first
version quoted "152 across a rotation" as if it were general and "191 across
three" for the shipped fit, both wrong. The argument survives in its honest form:
the smallest rotation fits a non-linear function of five predictions on 152
heavy-tailed rows, and exp-008 finding 7 measured that even a linear weight
vector fitted on ~150 such rows does not transfer to an unseen month.

## The arm was never chosen — it was drawn

All four (arm × restriction) configurations, five-seed bags, on the same object
(`analysis/arm_choice.csv`):

| configuration | july_loo | december | january | **board** | vs blend2 |
|---|---|---|---|---|---|
| `a5 / keep-noac` | **295.527** | 213.934 | **342.689** | **317.280** | −0.737 |
| `a5 / full` | 295.928 | 214.185 | 342.602 | 317.446 | −0.571 |
| `a2 / full` | 295.938 | 213.590 | 342.614 | 317.457 | −0.560 |
| `a2 / keep-noac` | 295.950 | **213.166** | 342.752 | 317.529 | −0.488 |

Day-clustered, 2,000 draws, against `blend2`:

| configuration | july_loo | december | january |
|---|---|---|---|
| `a5 / keep-noac` | −0.464 **[−0.964, −0.018]** | −0.218 [−0.995, +0.439] | −1.038 **[−1.739, −0.705]** |
| `a2 / keep-noac` | −0.042 [−0.508, +0.415] | −0.986 **[−1.957, −0.137]** | −0.974 **[−1.831, −0.512]** |
| `a2 / full` | −0.054 [−0.528, +0.400] | −0.563 [−1.652, +0.443] | −1.112 **[−2.086, −0.574]** |
| `a5 / full` | −0.064 [−0.611, +0.453] | +0.033 [−0.833, +0.762] | −1.124 **[−1.858, −0.765]** |

**Three of the four beat `blend2` on all three folds. `a5 / full` does not** — it
loses december by **+0.033 s**, the only positive point estimate in the interval
table above. The first version of this section said "all four", two lines under a
table printing that +0.033, which is the same re-reading failure round 1 caught
in the same document (round-2 review, B1). **The frozen rule does discriminate:
it excludes `a5 / full`.** The three that pass span 0.25 s, about twice the seed
spread, and the choice among *those three* is what three folds cannot make.

**The `a5 / keep-noac` july_loo advantage is an artefact of the offload, and it
is fragile.** The offload is worth 0.40 s to `a5` on july_loo and *costs* `a2`
0.012 s, because `a2` sees the `noac` and `regime` flags and already handles
those rows while `a5` has no way to know a row is `noac`. Those 339 rows carry
**25.0 %** of july_loo's squared error, and they are the fold on which the
offload's value was observed. On top of that the interval's upper end is −0.018
at 2,000 draws and the reviewer found it crossing zero on some bootstrap seeds.
**Read that row as "one interval excludes zero (january), one sits on it".**

## Adoption, against the rule frozen before the numbers

The rule in `SPEC.md`: *a variant is a candidate only if its rotated RMSE beats
`blend2` on every one of the three folds and its board estimate improves.*

**Three of the four bagged configurations satisfy it**: `a5 / keep-noac`,
`a2 / keep-noac` and `a2 / full`. **`a5 / full` fails it**, losing december by
0.033 s. This document has now been wrong about this claim twice — first as
"`keep-noac` is the only configuration that does" (round 1 B1; `bulk-nonm` also
passed) and then as "all four" (round 2 B1; `a5 / full` does not). Both times the
contradicting table was on the same screen. The rule does have discriminating
work left: it removes one configuration, and among the remaining three the choice
falls to the board estimate and to judgement.

### The selection I am carrying

Ten non-baseline variants and three restrictions were scored on the same three
folds; with the two-arm bagged comparison that is **fifteen configurations on
three months**. Named honestly:

- **Selected on the folds**: `a5` over `a2` (0.019 s at the seed mean, inside the
  spread — the first version wrongly presented this as a pre-registered mechanism
  test; the ablation was proposed *after* round 1's numbers were read);
  `keep-noac` over `bulk-nonm` (0.09 s); `keep-noac` over `full` (0.17 s, though
  the frozen rule independently excludes `a5 / full`); and, on the repaired
  member, `a5` over `a2` again on july_loo (+0.410) — a fourth fold-based choice,
  listed here after round 2 flagged that the swap section introduced one
  silently.
- **Not selected on the folds**: the `noac` offload's *motivation*, which is
  exp-008's prior; the five-seed bag, which is not a choice; and the residual
  parameterisation, which the direct-target measurement settles on mechanism.

Given that three configurations pass and they span 0.25 s, I am **not** claiming
the 0.737 s; the claim is **0.49–0.74 s**, and which end depends on a choice
three folds cannot make. On the repaired winter member the whole range moves down
to about **0.45 s**, which is the number to plan with.

## The candidates

| field | `exp-016-stack-1` | `exp-016-stack-2` |
|---|---|---|
| SHA-256 | `099d46702b4ac393e4f604b50979333f913b00daaad7033da2e0f46740b7f99d` | `cafb2d49ece4f8c783ea4f69049d087dde06d5f876471b29b96a80c12ebf9127` |
| stacker | five-seed bag, **member predictions only** | five-seed bag, **members + row features** |
| `noac` rows | flat four-member average | flat four-member average |
| fitted on | all three folds' OOF, 510,096 rows (491 noac) | same |
| rotated | 295.527 / 213.934 / **342.689** | 295.950 / **213.166** / 342.752 |
| **board estimate** | **317.280** | 317.529 |
| verified | 344,841 rows, ids identical to the template and in template order, float64, all finite, min 60.000, median 956.98, p99 2,501.4, max 119,739.9, none below the floor, guard touched 0 | same checks, median 956.71, p99 2,525.5 |
| vs candidate 2 | 344,532 rows differ, RMS 27.80 s | 344,537 differ, RMS 33.19 s |

**Recommended first: `exp-016-stack-1`**, on the declared metric — it has the best
board estimate, it passes the frozen adoption rule on all three folds, and the
difference between it and the two other passing configurations is inside what
three folds resolve, so the metric should break the tie rather than my judgement
about which interval looks sturdier.

**The case for `exp-016-stack-2`, which the desk can act on cheaply**: its gain
does not depend on 339 rows of one fold, its two decisive intervals are on the
two *independent* months rather than on the contaminated one, and it is **0.99 s**
better than `blend2` on december where stack-1 is 0.22 s better. (A first version
said 0.77 s, which is stack-2 minus *stack-1* — a different comparison from the
one this sentence makes. Round-2 review, N2.) If the board
comes back with stack-1 behind `blend2`, stack-2 is the next upload and not a
re-derivation.

### The shipped stack-2 was not the model I scored, and now is

**Review round 2, B2, and it is the worst defect in this experiment.** Every
rotation that measured the members+gates arm declared `airport` and `regime` as
LightGBM **categorical** features. `code/submit.py` built its dataset without
`categorical_feature`, and because `stack_features` passes those columns as
`int32` codes, LightGBM's `auto` does not pick them up — so the served model
split airport as an *ordered numeric axis* while the scored model split it as a
category.

I verified it before acting, rebuilding the ranking prediction both ways against
the pinned members and comparing to the file that was on disk:

| stack-2 built as | max abs difference from the shipped file |
|---|---|
| **no categorical** (what `submit.py` did) | **0.000000** — bit-exact |
| categorical declared (what the rotation did) | **149.68 s** |

So the file was the no-categorical model and **every rotated number recorded for
stack-2 — 295.950 / 213.166 / 342.752, board 317.529, in RESULTS, `climb.log`,
`candidate-2.json`, `MANIFEST.md` and the ledger — belonged to a model that was
never written to disk.** `submit.py` now declares the categoricals, and
`exp-016-stack-2.parquet` has been rebuilt: its SHA-256 moves from
`643a18db…857c44` to **`cafb2d49…f9127`**, and it is now the estimator the
rotation scored.

**Candidate 1 is untouched and rebuilds to `099d4670…` byte for byte**, because
the members-only arm's column set contains neither categorical, so the omission
was a no-op for it.

Two things made this survivable and one made it possible. It was survivable
because `submit.py` hardcodes the rotated numbers rather than recomputing them,
so nothing could contradict itself — and that is also what made it invisible: **a
build that copies literals from a dict can never fail a consistency check.** It
is exp-008's B2 failure — a served member that is not the estimator whose weight
was fitted — reappearing one level up, in the experiment that cites that failure
in its own module docstring.

### Required decomposition, and both baselines

`analysis/final_tables.csv`, regenerated with `code/final_tables.py` from the
committed OOF. COMMON.md requires the ≤3600 / >3600 / no-NM split on every table
and the assignment asks for intervals against **both** baselines; the first
version gave the split on one table and never compared a shipped configuration
to the flat average at all (round-2 review, N5, N6).

| fold | configuration | RMSE | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|---|
| july_loo | blend2 | 295.99 | 258.73 | 2464.88 | 1160.02 | 3511.06 |
| july_loo | flat4 | 296.42 | 258.69 | 2481.08 | 1160.02 | 3511.06 |
| july_loo | **stack-1** | **295.53** | 259.04 | **2439.36** | 1158.31 | 3511.06 |
| july_loo | stack-2 | 295.95 | 259.00 | 2455.36 | **1158.18** | 3511.06 |
| december | blend2 | 214.15 | 199.26 | 1986.87 | 711.28 | 2199.44 |
| december | flat4 | 214.46 | 199.33 | 2003.31 | 711.28 | 2199.44 |
| december | stack-1 | 213.93 | 199.23 | 1973.78 | 703.29 | 2199.44 |
| december | **stack-2** | **213.17** | **198.62** | **1960.11** | **703.24** | 2199.44 |
| january | blend2 | 343.73 | 207.42 | 5964.33 | 2846.81 | 3642.04 |
| january | flat4 | 343.67 | 207.16 | 5966.92 | 2846.81 | 3642.04 |
| january | **stack-1** | **342.69** | 206.45 | **5951.98** | **2842.70** | 3642.04 |
| january | stack-2 | 342.75 | **206.43** | 5954.00 | 2843.79 | 3642.04 |

**The gain is in the tail and the no-NM rows, not the bulk.** On january the
≤3600 s bulk moves 207.42 → 206.45 while the >3600 s tail moves 5964 → 5952 and
no-NM 2846.8 → 2842.7. `noac` is identical across all of them by construction —
the offload serves the flat average there, which is what `blend2` serves too.

Day-clustered 95 % intervals against **both** baselines:

| fold | configuration | vs `blend2` | vs `flat4` |
|---|---|---|---|
| july_loo | **stack-1** | −0.464 [−0.964, −0.018] | **−0.888 [−1.381, −0.451]** |
| july_loo | stack-2 | −0.042 [−0.508, +0.415] | **−0.466 [−0.910, −0.052]** |
| december | stack-1 | −0.218 [−0.995, +0.439] | −0.527 [−1.305, +0.089] |
| december | **stack-2** | **−0.986 [−1.957, −0.137]** | **−1.295 [−2.269, −0.430]** |
| january | **stack-1** | **−1.038 [−1.739, −0.705]** | **−0.978 [−1.608, −0.657]** |
| january | **stack-2** | **−0.974 [−1.831, −0.512]** | **−0.914 [−1.724, −0.447]** |

**Against the flat average the case is stronger than against `blend2`** — stack-1
excludes zero on both board folds and stack-2 on all three. That matters for the
mechanism claim: the thing being beaten is the constant-weight average of the
same members, so what the stacker adds is the row-dependence, not better members.

## The repaired winter member landed mid-experiment, and it absorbs 40 % of the gain

exp-012 adopted `ds_columns="deice"` for the winter member (commit `ec54ccf`,
2026-09-05, after this experiment's candidates were built): the one bounded
day-state column instead of the 25-column block, worth 51 s on january's most
disrupted 5 % of days. The desk's assignment named "the repaired D1-winter when
it lands". It has landed **as out-of-fold predictions only** — there is no
`D1w-deice-ranking.parquet` — so it can be measured here and **not served**.

Same rotation, same arms, same five-seed bag, one member address changed
(`analysis/swap_deice.csv`):

| configuration | july_loo | december | january | **board** |
|---|---|---|---|---|
| blend2, block member | 295.992 | 214.152 | 343.726 | 318.017 |
| **blend2, deice member** | 296.040 | 214.820 | **342.929** | **317.661** |
| stack-1, block member | 295.527 | 213.934 | 342.689 | 317.280 |
| **stack-1, deice member** | 295.882 | 214.374 | **342.168** | **317.215** |
| stack-2, block member | 295.950 | 213.166 | 342.752 | 317.529 |
| stack-2, deice member | 296.450 | 213.724 | 342.283 | 317.565 |

**The stacker's advantage over `blend2` falls from 0.737 s to 0.446 s once the
winter member is repaired.** About 40 % of what the stacker was buying, the
member repair buys too — both of them are fixing the same thing, which is the
blend being wrong on january's disrupted days. That is exp-012's absorption
finding arriving from the other side, and it is the honest number to plan with:
**a stacker on top of a repaired member set is worth about 0.45 s, not 0.74 s.**

Two things sharpen rather than blur under the swap:

- **The arm choice stops being a coin-flip.** With the repaired member `stack-2`
  no longer beats `blend2` on july_loo (**+0.410** [−0.006, +0.865]) while
  `stack-1` still beats it on all three folds (−0.158, −0.447, **−0.762
  [−1.289, −0.471]**). **On the repaired member the arm choice is made by a fold
  rather than by 0.019 s of board estimate** — which is itself a *new*
  fold-based selection and is listed as one in the disclosure below. It does not
  overturn "the arm was never chosen — it was drawn" three headings above: both
  are true of different member sets, undetermined on the pre-repair one and
  determined on the repaired one (round-2 review, N13).
- **The january gain survives with its interval intact** (−0.762 [−1.289,
  −0.471]), which is the fold that carries 44 % of the board mix and the one
  this whole lane is exposed on.

**The desk's call on this (2026-09-05), recorded here because it is a path
question and not a `internal decision record` one**: the blend owner is already building
**blend-3** on the deice member, including its ranking predictions (three seeds,
both months per frame, pinned). When blend-3's members are published, this lane
rebuilds the stacker on them and hands over **`stack-3`**. Until then `stack-1`
and `stack-2` **stand as measured on the pre-repair block member**, and their
board estimates (317.280 / 317.529) are against a `blend2` that uses the same
pre-repair member — so the comparison is internally consistent even though the
level will move.

**The rebuild is a config change, not an edit.** `code/members.py` reads member
addresses from `members-*.json` when one is passed, so stack-3 is:

```
python code/members.py --config members-blend3.json     # pins the new bytes
python code/stack.py    --config members-blend3.json     # the rotation
python code/submit.py 3 --config members-blend3.json     # the candidate
```

`members-blend3.json` is committed with the expected addresses and will fail
loudly on the sha256 check until the files exist, which is the intended
behaviour: it cannot silently build on the wrong member.

## stack-3: the candidate on blend-3's published member set

exp-012 published the repaired winter member as an interface
(`experiments/exp-012-blend2/published/MEMBER-D1-winter-deice-v1.md`) with hashes
and an explicit instruction — **use these files, do not refit**, because the arm
is `linear_tree=True` and LightGBM's determinism does not cover linear leaves
across thread counts, so a refit is a different member. Nothing here refits it;
the stacker consumes its predictions. `published/verify_member.py` reports
`INTERFACE OK` against the four files this experiment reads.

Same rotation, same arms, same five-seed bag, on that base
(`analysis/arm_choice-blend3.csv`):

| configuration | july_loo | december | january | **board** | vs blend-3's base |
|---|---|---|---|---|---|
| **`a5 / keep-noac` = stack-3** | **295.882** | **214.374** | **342.168** | **317.215** | **−0.446** |
| `a5 / full` | 296.304 | 214.670 | 342.069 | 317.387 | −0.274 |
| `a2 / full` | 296.439 | 214.220 | 342.129 | 317.486 | −0.175 |
| `a2 / keep-noac` | 296.450 | 213.724 | 342.283 | 317.565 | −0.096 |
| `blend2` on this member set | 296.040 | 214.820 | 342.929 | 317.661 | – |

**On the repaired base the frozen rule selects a unique configuration.** Only
`a5 / keep-noac` beats the baseline on all three folds; the other three lose
july_loo (+0.264, +0.399, +0.410). The arm question that three folds could not
settle on the pre-repair member is settled here by a fold, and it picks the same
form that ships as stack-1.

Day-clustered 95 %, against **both** baselines:

| fold | vs `blend2` | vs `flat4` |
|---|---|---|
| july_loo | −0.158 [−0.535, +0.232] | **−0.850 [−1.342, −0.418]** |
| december | −0.447 [−1.223, +0.200] | **−0.739 [−1.495, −0.114]** |
| january | **−0.762 [−1.289, −0.471]** | **−0.853 [−1.430, −0.524]** |

**Against the flat average all three intervals exclude zero.** Against the fitted
`blend2` rule only january does, which is the honest shape: most of what a
constant-weight blend can do, `blend2` already does, and what the stacker adds on
top of it is concentrated in the disrupted winter fold.

| field | value |
|---|---|
| path | `submissions/candidates/exp-016-stack-3.parquet` |
| SHA-256 | `c39bb173e4233ac5cb5a30f290282571d0f61fccb53a20550ee1c4420c4a4fbe` |
| stacker | five-seed bag, member predictions only, `noac` on the flat four-member average |
| member set | `members-blend3.json`, pinned in `pins-members-blend3.json` |
| fitted on | all three folds' OOF, 510,096 rows (491 noac) |
| rotated | 295.882 / 214.374 / **342.168** |
| **board estimate** | **317.215** against blend-3's 317.661 |
| verified | 344,841 rows, ids identical to the template and in template order, float64, all finite, min 60.000, median 957.57, p99 2,489.2, max 120,110.6, none below the floor, guard touched 0 |

### Board estimates across the two bases are not comparable, and the table says so

stack-1 (317.280) is **not** better than stack-3 (317.215) by any meaningful
comparison: they sit on different member sets and are measured against different
baselines — 318.017 and 317.661. On its own base stack-1 is worth 0.737 s and
stack-3 is worth 0.446 s, and **the repaired member is the better base for a
reason the fold estimate cannot see**: it puts 0 january validation cells outside
its training range where the block member put 4,640, and January 2026 has 18
airport-days past their airport's envelope that no 2025 fold can price
(`docs/candidate-audit.md`). The desk's ordering — prefer the repaired member
over 0.4 s of fold estimate — follows from that, not from these numbers.

`experiments/exp-012-blend2/candidate_table.py` now carries all three, with the
base blend-2 in the note column so the two families cannot be read against each
other by accident. **Shared-file note**: that edit is three additive `KNOWN`
entries and a comment in the blend owner's script; no existing row or behaviour
changed.

## Why the stacker is insulated from member quality, and what that costs

Substituting exp-008's 600-round CatBoost OOF for the shipped 5,325-round one — a
**14.24 s** member degradation, far larger than any same-configuration numerical
difference — and re-running the same rotation (`analysis/sensitivity_cat.csv`):

| rule | board estimate | delta | retention |
|---|---|---|---|
| `C2-cat` member alone | 330.230 → 344.465 | +14.24 | – |
| rotated NNLS, four members | – | −0.05 | **0 %** |
| **stack-3 form** | 317.215 → 317.929 | **+0.71** | **5 %** |
| blend-3 hybrid | 317.661 → 319.495 | +1.83 | 12.9 % |
| flat, four members | – | +2.49 | 17 % |

**The series is monotone in how much the rule can reallocate weight away from a
degraded member, not in how much fitting it does.** The stacker fits 300 trees
over nine inputs against NNLS's four scalars, so an ordering by fitting would put
it below 0 % and it is at 5 %. exp-012's lane confirmed the mechanism directly:
refitting the `nm_late` vector on the degraded member drives **C2-cat from 0.2177
to exactly 0.0000** while the other four absorb its share, and the same candidate
holds it pinned at 0.25 outside `nm_late` where the rule is flat.

**The stacker's 5 % is a property of its parameterisation, not its capacity.** It
fits the residual from the flat average and adds it back, so flat is its
baseline: it can learn to distrust a member's disagreement but it cannot zero
one, which places it strictly between flat and NNLS.

Two consequences, and they point opposite ways, which is why the measurement must
not be quoted in only one direction:

- **stack-3's board estimate is robust to exp-012's reproducibility outcomes.**
  Even the worst branch — CatBoost forced to CPU, both member sides moving —
  costs it well under 0.71 s. The number would still have to be **re-derived**,
  because fitting weights on one estimator and serving another is wrong at any
  magnitude, but it will not move materially.
- **Tuning a member cannot reach the stacker either.** Insulation from a worse
  member and unresponsiveness to a better one are the same property. Anyone
  reading the 5 % as "CatBoost quality barely matters, so cheapen it" is reading
  it backwards.

*Caveat carried with the 0.71.* It is a single five-seed-bag measurement, not an
interval. The stacker's board estimate has a five-seed spread of **0.045 s**, so
the delta is 16× its own noise and safe — but exp-012's 1.83 has no seed
component and is the firmer of the two numbers.

## The risk, stated as the reviewer put it

**The slice whose weight grows most on the board is the slice where this
candidate is least stable.** The stacker's gain is concentrated in rows where the
members disagree, and above 400 s of disagreement its sign flips between folds
(+5.51 january, −2.35 december, −1.28 july_loo); `nm_late` does the same
(+6.73 / −4.41 / −0.56). Disrupted days are precisely where members disagree, and
`docs/candidate-audit.md` measured January 2026 at **7.84 %** of rows on
envelope-exceeding days against the january fold's **0.73 %**.

So the exposure correction in that document — about +5.8 s for both exp-012
candidates — **cannot simply be carried across to these**. It holds each
candidate's own per-slice RMSE fixed and changes only the mix; these candidates
move 344,532 rows by 27.8 s RMS, so their slice RMSEs are unknown, and the slice
that grows is the unstable one. A first version of this section asserted "about
323.0, and the 0.74 s of separation is unaffected". **That is withdrawn**: the
direction of the unmeasured term is against this candidate, not neutral.

## Two mismatches, and one gate never run

1. **The stacker learned to correct eleven-month members and will serve
   twelve-month ones.** Its training signal is where the OOF members go wrong;
   the served members are strictly better fits. Not removable without a
   twelve-month OOF that does not exist.
2. **The shipped stacker is fitted on three folds; the rotation's stackers saw
   two.** More data should help, so the board estimates are mildly conservative
   in that one respect, which does not offset (1) or the risk above.
3. **The floor is applied at a different point at serve than in the rotation.**
   The rotation floors each seed's prediction and then averages
   (`why.a5`, `verify_r1.arm`); `submit.py` averages the five raw outputs and
   floors once. Round 1 measured the effect at 4 ranking rows and a maximum of
   0.36 s, so it is small — but it is the same class of defect as the categorical
   omission above and belongs here rather than under "verified correct"
   (round-2 review, N11).
4. **`SPEC.md` was edited after the numbers were read.** Its whole evidential
   value is that it predates them, and round 1 verified exactly that from
   mtimes. I then edited it twice — correcting the fold count and adding the
   `phi` status paragraph — before the experiment was in git, so there is no
   diff to show what moved (round-2 review, N9). Neither edit changed the frozen
   gate set, the adoption rule or the interval definition, and the `phi`
   paragraph only records that a deferral stayed a deferral; but the freeze is
   now asserted rather than demonstrable, and that is my fault.
5. **`phi` was on the desk's gate list and was never run.** `SPEC.md` deferred it
   conditionally on the first cut showing a stacker worth extending, and the
   first cut did. It is the only gate on that list needing a per-fold `PhiModel`
   fit rather than a cached column, and I did not spend it. **This is an open
   item, not a close.** Its expected value is now bounded by the ablation: the
   gate class as a whole is a wash on the two board folds and worth 0.59 s on
   december, so phi's own contribution is likely small on the board and possibly
   real on december — which is exactly the ambiguity `a2` already carries.

## Fold design: three rotations, not four

The assignment asked for four. `prc/data.py` defines six folds, but **`july` and
`july_loo` validate the same month** — both score 2025-07, the same 190,713
departures with the same labels — so a rotation containing both fits and scores a
stacker on the same rows. Independently, `C2-cat`, `L2-hub` and `N1-nmtail` have
no `july` OOF at their pinned addresses. Three rotations, matching the folds the
board estimate is built from. Full reasoning in `SPEC.md`.

## Review rounds 1 and 2

Reviewer `claude-opus-5-xhigh-2026-09-05` (Opus substitute;
desk rule; `assignments/robustness.md` sets this substitution).
Transcripts `review/round1.txt`, `review/round2.txt`. **Two rounds, the cap the
assignment set.**

**Round 1: four blocking, ten important. Every blocking finding was real.**

| # | finding | disposition |
|---|---|---|
| B1 | "the only configuration that satisfies the rule" is false — `bulk-nonm` passes too | accepted |
| B2 | the ablation conclusion is a single-seed artefact; at the seed mean gates cost 0.019 s and on december are worth 0.586 s | accepted, re-measured myself, matched to the decimal |
| B3 | the `a1` diagnosis is wrong, the promised repairs were never run, and the budget repair does not work | accepted, re-measured myself |
| B4 | missing decomposition, OOF format, `climb.log`, ledger, verdict, `phi` | accepted |

**Round 2: two blocking, thirteen important. Both blocking findings were real,
and one of them is the worst defect in this experiment.**

| # | finding | disposition |
|---|---|---|
| **B1** | "all four configurations pass the adoption rule" is false — `a5 / full` loses december by 0.033 s | **accepted**; three of four pass, and the rule does discriminate. This is the second time this document made this class of error, both times with the contradicting table on the same screen |
| **B2** | `submit.py` omitted `categorical_feature`, so the shipped stack-2 was not the model whose RMSEs were recorded — up to 149.7 s per row | **accepted, verified myself** (no-categorical build is bit-exact to the old file, categorical build differs by 149.68); `submit.py` fixed, stack-2 rebuilt, SHA `643a18db…` → `cafb2d49…`, every record corrected |

Non-blocking, all accepted: **N1** the deice swap ran on an unpinned member from
a lane that had written it minutes earlier — the exact failure this experiment's
own pinning docstring cites, now fixed with a per-member-set pin file; **N2**
0.99 s not 0.77 s; **N3** the monotone constraint costs 0.072 s not 0.16 s;
**N4** the `init_score` arm ran at lr 0.03 not 0.05; **N5/N6** intervals against
the flat average and the required decomposition, both now delivered for every
configuration; **N7** the OOF deliverable had no producing code, now
`final_tables.py --rebuild-oof`; **N8** four code docstrings still asserting what
the report had withdrawn; **N9** `SPEC.md` was edited after the numbers;
**N10** "captures nearly all the gain" is false on december; **N11** the floor is
applied at a different point at serve than in the rotation; **N12** a seed spread
in the ledger that is not in the CSV; **N13** the swap section introduced an
undisclosed fold-based selection.

The reviewer also confirmed, independently and against the live files: no
leakage anywhere in the rotation or the gates; the fourth-rotation refusal;
that the vectorised bootstrap is the same estimator as the loop it replaced; that
the gates ablation is well-formed and the five-seed re-measurement reproduces;
and the round-2 mechanism for the direct-target arm — with an arm I had not run,
relaxing capacity at the same budget recovers 84.5 s while adding budget at high
capacity makes it worse, which is the "capacity, not budget" claim tested
directly.

**What I would have caught alone and did not.** Both round-2 blocking findings
are re-reading failures rather than measurement failures. B1 is a sentence
contradicted by the table above it — the same mistake round 1 had already caught
in this document, which means I corrected the instance and not the habit. B2 is
the more instructive one: `submit.py` hardcodes the rotated numbers into the
candidate record rather than recomputing them, so the build could never
contradict itself. **A build path that copies literals cannot fail a consistency
check**, and this experiment consumed an exp-008 finding of exactly that shape
while reproducing it one level up. The fix that would have caught it is cheap and
is the thing to carry forward: have the candidate builder re-score its own
configuration on one fold and refuse if it does not match the recorded number.

## Compute

Round 1 ran at 12 threads before the desk's cap (`docs/COMPUTE.md`, compute allocation). From the
cap onward this experiment runs at **4 threads**, one job at a time; the bootstrap
is vectorised over draws; every launch held until `/proc/loadavg` 1-minute was
under 140 (round 2 waited for 194 → 104). Every stacker fit is 8–14 s, so the
whole experiment after round 1 cost about 25 minutes of 4-thread CPU.
