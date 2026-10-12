# exp-012 blend2 — the standing blend

Owner of the live blend from 2026-09-05. Three folds (`july_loo`, `december`,
`january`), each a distinct validation month, each used once: a fold's weights
are fitted on the other two and applied unseen. Board estimate is the quadrature
combination of `july_loo` and `january` weighted by ranking-month share
(0.5571 / 0.4429). See `SPEC.md` for why the forward `july` fold was dropped.

## Every candidate in the repo, best first

Regenerate with `python candidate_table.py`.

| candidate | what it is | board estimate | folds | sha256 |
|---|---|---|---|---|
| `exp-012-blend-3.parquet` | exp-012 standing blend: T-tailv2, C2-cat, L2-hub, D1-winter, N1-nmtail | **317.66** | july_loo 296.0398 / december 214.8202 / january 342.9295 | `f91db30a7f3b…` |
| `exp-012-blend-2.parquet` | exp-012 standing blend: T-tailv2, C2-cat, L2-hub, D1-winter, N1-nmtail | **318.02** | july_loo 295.9916 / december 214.1522 / january 343.7263 | `5c57799954cb…` |
| `exp-012-blend-1.parquet` | exp-012 standing blend: T-tailv2, C2-cat, L2-hub | **318.81** | july_loo 297.9491 / december 214.9318 / january 343.2647 | `541b1b1107fe…` |
| `exp-008-blend3.parquet` | exp-008, NNLS of tail v1 + Huber LGB + linear-leaf LGB | **320.37** | july_loo 300.0 / dec 216.1 / jan 344.3 | `26c40e69e0a8…` |
| `exp-001-retro-v1-c6.parquet` | exp-001 c6, residual + linear leaves + noac head | – | july 343.6 / dec 222.4 / jan 349.0 | `504d489d239e…` |
| `exp-001-retro-v1-c7.parquet` | exp-001 c6 blended 0.5/0.5 with prc.tail_model | – | july 323.5 / dec 218.1 / jan 347.1 | `e8261bb4fb58…` |
| `exp-001-retro-v1-c8.parquet` | exp-001 c8 | – | see exp-001 RESULTS | `bb720e1a81aa…` |
| `exp-001-retro-v1-c9.parquet` | exp-001 c9 | – | see exp-001 RESULTS | `ee8fa94dfec2…` |
| `exp-001-retro-v1.parquet` | exp-001 v1, LightGBM raw seconds | – | july 531.4 / dec 287.0; no july_loo | `fa6195ffb216…` |
| `exp-016-stack-1.parquet` | (unrecorded) | – |  | `099d46702b4a…` |
| `exp-016-stack-2.parquet` | (unrecorded) | – |  | `643a18db897e…` |

**Upload order: `exp-012-blend-3` first, then `exp-012-blend-2`.** Candidate 3 is
0.36 s better than candidate 2 on the board estimate, and the reason to prefer it
is larger than that: its winter member no longer extrapolates on disrupted days,
which January 2026 has 18 airport-days and 7,564 rows of and no 2025 fold prices. The exp-001 candidates have no `july_loo` measurement and cannot be
ordered against these without one; they are listed for completeness, not as
alternatives.

## The blend is saturated for general members, and only specialists still pay

This is the lane's central finding and it now rests on four independent
candidates, each screened the same way: NNLS fitted **in sample** against the
four-member core, which is the most weight a member can possibly earn.

| candidate member | lineage | alone (july_loo) | in-sample gain vs core4 | weight |
|---|---|---|---|---|
| `A0-plain` | retro-v1, plain trees | 315.23 | **0.000** | 0.000 |
| `X1-deice` | retro-v1, de-icing | 314.29 | 0.061 | 0.069 |
| `N1-nmtail` | exp-010, phi target | — | 0.115 | 0.064 |
| `E12-mlp` | **embedding MLP, no trees at all** | 315.35 | 0.113 | 0.080 |

**Four members from four lineages, one of them not a tree at all, and every one
of them is worth about a tenth of a second.** The MLP is the sharpest version of
the test: exp-008's whole diversity lane was built on the premise that a
non-tree learner would decorrelate, and it does not — its residual correlates
0.856 with the tail model, 0.891 with the winter arm, 0.915 with CatBoost and
0.927 with the Huber LightGBM, which is inside the range the tree members
already occupy with each other (0.86–0.92). A neural network on this feature
frame correlates with a GBM about as much as two GBMs correlate.

Both the MLP and the NM-tail head show the same absorption curve as the core
grows — the MLP's is 7.51 → 0.70 → 0.34 → 0.113 across cores of one to four
members, the head's is 10.85 → 1.85 → 0.43 → 0.20 → 0.115. The decay is a
property of the blend, not of the candidate.

### What still pays: a member restricted to an input-defined regime

The one thing that has worked is giving a member weight only where its mechanism
lives. Inside `nm_late` (NM-joined, AOBT − LOBT above 1800 s; 7.6 % of july_loo
rows and 20 % of its squared error), fitted in sample:

| member set inside `nm_late` | RMSE | gain | weights |
|---|---|---|---|
| core4 | 476.51 | – | – |
| **+ `N1-nmtail`** | **471.69** | **4.82** | N1 0.340 |
| + `E12-mlp` | 474.77 | 1.75 | MLP 0.218 |
| + both | 470.90 | 5.62 | N1 0.309, MLP 0.149 |

The two specialists are *partly* complementary — the MLP adds 0.80 s on top of
N1 rather than its standalone 1.75 — and their residuals correlate 0.870 inside
the regime against 0.911 and 0.923 for each against the core.

So the shape of the remaining opportunity is not "find a better general model".
It is: **find a population defined by inputs where some model has a mechanism
the core lacks, and give it weight only there.** `nm_late` is one such
population and `N1-nmtail` is worth 0.433 s rotated inside it. `noac` is the
obvious next one — 339 rows carrying 24 % of july_loo's squared error, where the
core sits at 3,405 s and the MLP earns weight exactly 0.000.

**Provisional on one thing:** every screen above uses `D1-winter` as exp-007's
single-seed linear-tree OOF. My own three-seed rebuild is in flight; if it lands
materially different, the core changes and these margins move. The direction —
general members absorbed, regime specialists not — is too large and too
consistent across four candidates to turn on that, but the individual tenths are
not final until it lands.

## The saturation screens re-run by the blend criterion: the finding survives

Ordered by the desk after I found that standalone RMSE is the wrong criterion.
Each candidate is now judged by the **rotated blend RMSE with it in**, against the
same core on the same folds (`screen_by_blend.py`, OOF-only, no fits).

| candidate | folds it has | core | + candidate | gain |
|---|---|---|---|---|
| `A0-plain` | july_loo | 295.798 | 295.798 | **+0.000** \* |
| `X1-deice` | july_loo + january | 320.028 | 320.413 | **−0.385** |
| `E12-mlp` | july_loo + december | 256.072 | 256.217 | **−0.145** |
| `F1-fresh` | all three | 284.979 | 279.576 | **+5.402** |

\* one fold only, so its weights are fitted in sample and the number is an upper
bound. That three of the four were never run on all three folds is exactly *why*
they were screened in sample originally; it is a limit on this re-screen rather
than something it can fix.

**The saturation finding survives the criterion change, slightly strengthened.**
Under the old in-sample screen the three were worth 0.000, +0.061 and +0.113 —
small positives. Under an honest rotation they are 0.000, **−0.385** and
**−0.145**: at or below zero. The in-sample numbers were upper bounds, as
labelled, and the honest numbers sit under them.

So the answer to the gap I flagged is that the ranking did survive. It survived
for the reason predicted — all three were *more* correlated with the core than
the members they were joining — and it has now been checked rather than assumed.

**A flaw in my own screen, caught and removed.** The first version of
`screen_by_blend.py` ranked all four candidates against each other by mean
standalone RMSE and reported that the two orderings "disagree". That comparison
is invalid: the candidates sit on different fold sets, and the folds have very
different levels (december ~210, july_loo ~290, january ~343), so a mean over
july_loo + december is not comparable with one over all three. `E12-mlp` looked
like the best standalone member purely because december is in its set and january
is not. The script now prints no cross-candidate ordering and says why.

## The Huber corrected budget, judged by the blend: confirmed on three folds

Last item in the desk's queue, and the case that produced the screening-rule
correction. Measured by **member RMSE** the 8,725-round arm is worse on every
fold; measured by **what it adds to the blend** it is better.

| fold | member @2,100 | @8,725 | delta | member tail | @8,725 tail |
|---|---|---|---|---|---|
| july_loo | **310.84** | 314.41 | +3.57 | 2588 | **2561** |
| december | **223.60** | 227.17 | +3.57 | 2208 | **2180** |
| january | **348.12** | 350.39 | +2.27 | 5980 | **5974** |
| **member board estimate** | **327.87** | 330.83 | **+2.96** | | |

| blend (blend-4's member set) | 2,100 | 8,725 | delta |
|---|---|---|---|
| five general members, NNLS | 313.45 | **313.31** | −0.14 |
| **full hybrid, with the `nm_late` specialist** | 313.173 | **313.050** | **−0.12** |

**The member is 2.96 s worse and the blend is 0.12 s better, on three folds
rather than the one that first showed it.** The tail improves on all three, which
is the mechanism: the longer budget trades bulk accuracy for tail accuracy, the
member's own RMSE is dominated by the bulk, and the blend's marginal value is
not.

This is the finding that produced the desk's screening rule, now verified on the
full rotation rather than a single fold. It is also the cleanest example the lane
has of the rule mattering: judged by the criterion I used all evening the answer
is "3 s worse, do not ship"; judged by the criterion the desk installed, "0.12 s
better".

**Not shipped, and why.** A candidate needs the member's *ranking* predictions,
and those exist only at 2,100 rounds — the 8,725-round arm has fold OOF but no
twelve-month submission fit. That is roughly 1.5 h at six threads for 0.12 s. The
measurement is banked either way, and the desk placed this item last precisely
because it is worth about 0.2 s.

## Frame beats learner, and it decided a day of GPU (desk, 2026-09-06)

The desk's v3 assignment was to rebuild CatBoost and Huber on retro-v1's new v3
frame — two budget scans, eighteen fold fits and six ranking fits, most of a day
of GPU — on the reasoning that a different *learner* on the new frame would add
what the blend lacks. Before starting it, the cheapest test available: retro-v1
already has five v3 arms with OOF, so screening them by the blend criterion costs
nothing.

Against blend-4's five-member core, on july_loo + january:

| retro-v1 v3 arm | gain to the blend |
|---|---|
| **`V3-allin`** | **+1.130** |
| `V3-rollover` | +0.996 |
| `V3-full` | +0.423 |
| `V3-fresh` | +0.273 |
| `V3-wins3600` | +0.273 |

`F1-fresh` screens at +0.000 on the same run, which is the sanity check: it is
already in the core, so it must add nothing.

**All five earn weight, where the four candidate members screened before them all
came in at or below zero.** The v3 frame is the second-largest source of blend
gain anyone in this repo has found, after exp-018's lineage at +5.4.

**Corrected 2026-09-06: the RANKING of those five is confounded and I drew a
recommendation from it anyway.** retro-v1 found that their v3 source list grew
from 171 to 175 columns mid-programme, with arms already queued: `V3-allin` sits
on 171, the other four on 175. So each arm's gain remains a valid measurement of
that artifact — real OOF, honest rotation — but the five are **not comparable
with each other**, because they differ in the arm *and* in the feature set.

I nonetheless told retro-v1 that `V3-allin` adds more to the blend than
`V3-rollover` despite being worse standalone, and advised them to develop
`V3-allin` as the member on that basis. That advice is withdrawn. `screen_by_blend.py`
prints a warning that its rows are not comparable with each other, added after the
fold-set confound earlier the same day; I then compared them in prose. **The
guard was in the tool and the error was in the sentence.**

What the decision rests on is unaffected: `V3-allin` +1.130 against blend-4's
core is a single-artifact measurement, and the desk's deferral of the CatBoost
rebuild follows from that number alone, not from any ordering.

**The decision this produced.** `V3-allin` already earns 1.13 s as it stands, so
the question was never "is v3 worth having" but "is a *different learner* on v3
worth a day of GPU on top of the frame". Two things said no. The comparison is
asymmetric — I know what retro-v1's arm is worth and would have been guessing at
mine, since a v3 CatBoost cannot be screened without first building it. And this
lane's own saturation evidence says a fifth member of an existing lineage buys
nearly nothing while a genuinely better member buys a lot, which points at the
frame rather than the learner.

The desk took the cheap path: request `V3-allin`'s missing december fold and a
twelve-month ranking fit from retro-v1 — roughly two hours in their lane — and
defer the CatBoost rebuild until blend-5 is measured. **If `V3-allin` captures
the frame's signal, the learner rebuild is not worth the GPU; if it earns less
than 1.13 s once it and the fresh member interact, revisit.**

The general form, which is the part worth keeping: **when a new feature frame and
a new learner are both on the table, the frame can be screened with someone
else's existing arms and the learner cannot.** That asymmetry is a reason to test
the frame first every time, independent of which one turns out to matter.

## Why the blend criterion inverts standalone RMSE: members agree in the bulk and disagree in the tail

Three lanes independently hit the same bulk/tail structure tonight, and this is
the mechanism underneath all three.

Residual correlation between blend-4's five members on july_loo, split:

| slice | rows | mean pairwise correlation | range |
|---|---|---|---|
| bulk ≤ 3600 s | 190,057 (99.66 %) | **0.889** | 0.873–0.912 |
| tail > 3600 s | 656 (**0.34 %**) | **0.764** | 0.698–0.846 |

And the tail carries **23–27 %** of every member's squared error despite being a
third of one percent of the rows.

**So a blend has almost nothing to gain in the bulk and almost everything to gain
in the tail.** Where members correlate 0.889 their errors survive averaging;
where they correlate 0.764 averaging removes far more of them. A member that
improves the bulk is improving the part the blend already handles by consensus; a
member that improves the tail is improving the part only it can contribute to.

That is why **standalone RMSE is the wrong criterion**, in one sentence: a
member's own RMSE is 99.66 % a bulk statistic, and the blend's marginal value is
about a quarter tail. The two measure different things and can order candidates
oppositely.

### The same structure in three lanes

| lane | change | bulk | tail | own RMSE says | blend says |
|---|---|---|---|---|---|
| **exp-012** (this) | Huber 2,100 → 8,725 rounds | worse | **better** | 2.96 s worse | **0.12 s better** |
| **retro-v1** | winsorise at 3600 on v3 | flat | worse | 3.74 s worse | not measured |
| **exp-018** | clip at 3600 | **7.13 s better** | 67 s worse | 1.48 s better | not measured |

exp-018's decomposition, which they produced from an existing ablation rather
than a new fit, is the one that closes it: the sign pattern is the same in every
lineage, and what differed was only which half each of us quoted. They quoted the
bulk gain; retro-v1 quoted the fold total.

**A prediction this makes, for exp-018's clip.** It buys bulk and sells tail. On
the evidence above that is the wrong direction for a blend member — it improves
what the blend already gets by consensus and damages what only that member can
supply. So the clip should look better standalone than it does in the rotation,
and I would expect it to screen at or below zero even though it is +1.48 s on
their own fold. That is testable the moment their clipped and unclipped arms are
both available with OOF, and it is a falsifiable claim rather than a caution.

exp-018 also framed the trade as an exchange rate — widening 3600 → 6000 buys
back 96 s of tail for 0.71 s of bulk — which is the right shape. This section
adds the rate the *blend* would pay, which is not the same rate a member pays.

### The prediction, revised down before the measurement

exp-018 objected that the mechanism gives the sign but not a large magnitude:
clipping pulls their tail toward the centre, and a member that is worse *and
differently wrong* can still earn its weight. Their prediction is −0.5 to 0.0 s,
falsified past −1.5.

That turns on one step — **is a worse tail a more central tail?** If it were, the
blend would lose twice, once for accuracy and once for lost disagreement. Tested
on the Huber pair, which isolates it (identical but for the round budget, 8,725
having the better tail), correlation with the other four **on tail rows only**:

| arm | tail RMSE | mean tail correlation |
|---|---|---|
| `HUB2100` | 2588.1 | 0.777 |
| `HUB8725` | **2561.5** | 0.773 |

**A 26 s difference in tail accuracy moves centrality by 0.004.** So worse tail
does not imply more central, and their objection survives. The limitation cuts
against my own test: a round-budget change is not a clip, and a clip compresses
the learned response far more directly, so this shows the two are not
*automatically* coupled rather than that they are uncoupled under their
intervention.

**Revised prediction: sign kept, large effect withdrawn.** Uniqueness-weighted
arithmetic on their figures — bulk gain ≈ 3,944 MSE units discounted by the 0.11
disagreement available at correlation 0.889, tail loss ≈ 2,630 discounted by 0.24
at 0.764 — turns a member-level improvement into a *small* blend-level loss. That
is inside their −0.5 to 0.0.

Which makes the test weaker than it was: we now agree on sign and roughly on
magnitude. The informative outcomes left are **positive**, killing the mechanism,
or **past −1.5 s**, killing "small" and saying the rotation prices centrality
over accuracy. Recorded that way rather than left looking sharper than it is.

## Candidate 4 — the fresh lineage joins, and the blend rule flips to fitted

| field | value |
|---|---|
| path | `submissions/candidates/exp-012-blend-4.parquet` |
| SHA-256 | `3b7e558a0b542690673f5ad01ade7a0c210a998905928e5e95434c752549337d` |
| rule | **NNLS over five general members**, plus the NM-tail specialist inside `nm_late` |
| weights, elsewhere | `F1-fresh` 0.6222, `L2-hub` 0.1538, `T-tailv2` 0.1213, `D1-winter` 0.0683, `C2-cat` 0.0366 |
| weights, `nm_late` | `F1-fresh` 0.4751, `N1-nmtail` 0.2528, `D1-winter` 0.1498, `C2-cat` 0.0709, `T-tailv2` 0.0676, `L2-hub` 0.0000 |
| rotated | july_loo **289.32**, december **207.77**, january **340.82** |
| board estimate | **313.173** (blend-3b 317.612, blend-3 317.661) |
| verified | 344,841 rows, template order, all finite, min 60.138, none below the floor |

**4.44 s better than blend-3b**, the largest single step this lane has taken.

**The fit-serve mismatch is closed.** exp-018 published a matched one-seed pair —
`exp-018b-s1` served against the one-seed OOF that fits the weights, through one
code path. The earlier 4.70 s was measured with a one-seed OOF on both sides and
was therefore a clean measurement of the member, but the *candidate* on offer then
was two-seed, so it would have served an estimator the weights never saw. They
also bounded that perturbation for us: one-seed against two-seed is RMS 16.20 s
with 1.21 % of rows moving more than 60 s.

**Two rules changed hands, both by their own criteria.**

*Flat gives way to fitted.* The desk's rule is flat unless a fitted vector beats
flat in **every** rotation. With four comparable members it never did. With
`F1-fresh` in the set it does, on all three — 289.81/207.92/340.90 against
291.92/211.33/341.50 — because the members are no longer comparable and equal
weighting wastes the best one. `F1-fresh` takes 0.622 outside `nm_late`.

*The specialist survives, smaller.* `N1-nmtail` was worth 0.52 s inside `nm_late`
against the four-member core and is worth **0.29 s** against this one. The fresh
member absorbed roughly half of it — which is saturation behaving exactly as
described, on the one member that had escaped it.

**What the blend is still worth.** `F1-fresh` alone is 316.47; the blend is
313.17. The five older members are worth **3.30 s** on top of the best single
model, so this is still a blend rather than a pass-through.

## Candidate 3 — the repaired standing blend

| field | value |
|---|---|
| path | `submissions/candidates/exp-012-blend-3.parquet` |
| SHA-256 | `f91db30a7f3bc4da41a1f1599627cdbe48741ba80c8017105b8a1ba79f9631e0` |
| blend rule | **hybrid**: flat over the four general members, NNLS over five inside `nm_late` |
| weights, `nm_late` | `N1-nmtail` 0.3522, `C2-cat` 0.2177, `T-tailv2` 0.2132, `D1-winter` 0.1454, `L2-hub` 0.0799 |
| weights, elsewhere | 0.25 each of the four; `N1-nmtail` exactly 0.000 |
| rotated | july_loo **296.04**, december **214.82**, january **342.93** |
| board estimate | **317.661** (candidate 2 318.017, candidate 1 318.81) |
| verified | 344,841 rows, template order, float64, all finite, min 60.000, median 961.8, p99 2,450.8, none below the floor |
| against candidate 2 | 344,803 rows differ, max delta 3,339 s, RMS 14.19 s |

**What changed and why.** Only the winter member: the full 25-column day-state
block is replaced by `ds_deice_hr` alone, published as
`D1-winter-deice-v1`. It is worth 0.36 s on the board estimate, and that is the
smaller half of the reason — the block reached its score partly by extrapolating
on disrupted days, and January 2026 has 18 such airport-days and 7,564 rows that
no 2025 fold can price.

**The blend absorbs some of the member's cost, as it has absorbed everything
else.** The repaired member is 2.67 s *worse* than the block on july_loo as a
standalone, but candidate 3 loses only 0.05 s there (295.99 → 296.04) while
gaining 0.80 s on january. The same insensitivity that made member tuning worth
0.5 s makes a robustness repair nearly free.

**Weights moved more than the score did.** `D1-winter`'s share inside `nm_late`
falls from 0.2401 to 0.1454 and `N1-nmtail`'s rises from 0.3299 to 0.3522: the
bounded member brings less to the late-push regime, and the NM-tail specialist
takes up the slack. That is the fitted blend doing what it is for.

## Candidate 2 — the standing blend, five members

| field | value |
|---|---|
| path | `submissions/candidates/exp-012-blend-2.parquet` |
| SHA-256 | `5c57799954cbf0b5df0aff249b758a8d836ce71a91d8836b005e24bb869f8579` |
| blend rule used | **hybrid**: flat over the four general members, NNLS over five inside `nm_late` |
| weights, `nm_late` (24,740 fit rows) | `N1-nmtail` 0.3299, `D1-winter` 0.2401, `T-tailv2` 0.1906, `C2-cat` 0.1894, `L2-hub` 0.0578 |
| weights, elsewhere | 0.25 each of the four; `N1-nmtail` exactly 0.000 |
| rotated | july_loo **295.99**, december **214.15**, january 343.73 |
| board estimate | **318.017** (core-4 flat 318.208, candidate 1 318.81) |
| verified | 344,841 rows, ids identical to the template and in template order, float64, all finite, min 60.000, median 961.8, p99 2,454.7, max 119,739.9, none below the floor |
| against candidate 1 | every row differs, max delta 3,511.7 s, RMS 31.02 s |

**Why this rule.** The desk's four-member rule is flat unless a fitted vector
beats flat in every rotation, and it does not — NNLS wins only december, by
0.001. The NM-tail head then earns 0.297–0.336 inside `nm_late` across the three
rotations and exactly 0.000 in `nm_bulk`, so it is carried as a regime
specialist rather than as a fifth general member, which is worth −0.492 s.

**Member provenance**, every one at a durable address (`pin_members.py`,
0 problems):

| member | address | budget | drawn how |
|---|---|---|---|
| `T-tailv2` | exp-009 `…-oof-v2.parquet` (git blob) | `n_estimators` 500/300 | declared |
| `C2-cat` | `pinned/C2-cat--C2-d8p-*` | 5,325 → ×12/11 | one es-month |
| `L2-hub` | `pinned/L2-hub--L2-hub-*` | 2,100 → ×12/11 | one es-month, **validated by kill test** |
| `D1-winter` | exp-007 `d1-ds-s7-*` (git blob) | **150** | **the desk's rule** |
| `N1-nmtail` | exp-010 `nmtail-phi_*` (git blob) | `n_estimators` 900 | declared |

`D1-winter`'s ranking predictions are the first artifact drawn under the budget
rule: per-month optima **July 109, January 129, November 142**, median 129,
relative spread 0.256, rescaled to **150 rounds**. The seasonal ordering is the
rule's own justification appearing in the data — the summer month wants the
fewest rounds and November the most, which is exactly how a November-drawn count
came to overfit July for the Huber arm. The old single-slice instrument drew 165
for this member.

**Two mismatches disclosed rather than assumed away.** `D1-winter` serves a
three-seed twelve-month bag while its weights were fitted on exp-007's single
seed; measured at **0.014 s**. And `N1-nmtail` is undefined on 5,290 non-NM
ranking rows, permitted by `prc.blend` VERSION 5 only because its weight there
is exactly 0.000, which the builder verifies rather than trusts — zero `nm_late`
rows are null.

## Candidate 1

| field | value |
|---|---|
| path | `submissions/candidates/exp-012-blend-1.parquet` |
| SHA-256 | `541b1b1107fe85a13777409ecf3d66e93ab9c7fbc7a0dfedf054fc50daf77618` |
| members and weights | `T-tailv2` 0.4438, `C2-cat` 0.2227, `L2-hub` 0.3351 (NNLS on 510,096 OOF rows) |
| rotated | july_loo **297.95**, december **214.93**, january **343.26** |
| board estimate | **318.81** |
| verified | 344,841 rows, IDs identical to the template and in template order, all finite, min 60.099, median 962.9, p99 2,461.1, max 118,660.5, none below the floor |

Single members on the same three folds:

| member | july_loo | december | january | board |
|---|---|---|---|---|
| `T-tailv2` (exp-009 noac head v2) | **306.77** | 222.86 | 349.36 | **326.32** |
| `L2-hub` (Huber LightGBM) | 310.84 | 223.60 | **348.12** | 327.87 |
| `C2-cat` (CatBoost, corrected budget) | 315.17 | **222.86** | 348.25 | 330.23 |
| `R0-retro` (retro-v1) | 324.76 | 222.18 | 347.60 | 335.07 |
| `X2-xgb` (XGBoost, corrected budget) | 337.05 | 228.95 | 356.71 | 345.90 |

**The blend beats the best single member by 7.5 s on the board estimate**
(326.32 → 318.81) and by 8.8 s on `july_loo`.

Two members were dropped and it is worth saying why, because both are the same
lesson exp-008 learned: `X2-xgb` gets **zero** NNLS weight despite holding the
lowest correlations in the matrix, and `R0-retro` gets 0.052, worth **0.01 s** —
dropping it costs nothing measurable and it has no ranking predictions. A member
has to be different *and* good.

## The weight-stability check the desk asked for

Rotated NNLS produces one weight vector per held-out fold. If those disagree,
a fitted blend is not safe to ship, because the shipped vector is fitted on all
three months and would be a compromise between incompatible regimes.

| fitted on | R0-retro | T-tailv2 | C2-cat | X2-xgb | L2-hub |
|---|---|---|---|---|---|
| december + january (scores july_loo) | 0.089 | 0.376 | 0.278 | 0.000 | 0.257 |
| july_loo + january (scores december) | 0.043 | 0.450 | 0.194 | 0.000 | 0.316 |
| july_loo + december (scores january) | 0.049 | 0.460 | 0.185 | 0.000 | 0.307 |
| **all three (shipped)** | 0.052 | 0.437 | 0.210 | 0.000 | 0.302 |
| **spread across the three** | 0.046 | 0.084 | 0.093 | **0.000** | 0.058 |

**The weights are stable.** The largest movement is 0.093 on a weight of about
0.21, every member keeps the same sign and the same rank order in all three
vectors, and the zero on `X2-xgb` is unanimous. The shipped vector sits inside
the range of the three rotated ones on every member. So the rotation is
measuring one blend, not three different ones, and the board estimate is an
honest read of what the shipped weights will do.

The one asymmetry worth naming: the vector fitted on `december + january` — the
two winter months — puts noticeably more weight on CatBoost (0.278 against 0.185
and 0.194) and less on the Huber LightGBM. That is the only sign of month
dependence in the table, it is the fold whose fit set excludes the summer month,
and it is small enough not to change the ordering.

## Rotated blend variants

| variant | july_loo | december | january | board |
|---|---|---|---|---|
| best single member | 306.77 | 222.86 | 347.60 | 326.32 |
| flat average of five | 302.9 | 216.1 | 344.7 | 322.10 |
| **NNLS (shipped)** | **297.9** | **214.9** | 343.2 | **318.80** |
| OLS | 297.9 | 214.9 | **342.7** | 318.51 |

NNLS and OLS are 0.3 s apart and NNLS is shipped for the same reason as in
exp-008: non-negative weights on members correlating above 0.84 are the stable
choice, and OLS's edge is inside the run-to-run spread. Fitting beats a flat
average by 3.3 s here, more than the 1.4-to-3.9 s exp-008 measured, because the
members are more unequal.

## Residual correlation, july_loo

|  | R0-retro | T-tailv2 | C2-cat | X2-xgb | L2-hub |
|---|---|---|---|---|---|
| R0-retro | 1.000 | 0.840 | 0.879 | 0.875 | 0.918 |
| T-tailv2 | 0.840 | 1.000 | 0.873 | 0.849 | 0.863 |
| C2-cat | 0.879 | 0.873 | 1.000 | 0.883 | 0.909 |
| X2-xgb | 0.875 | 0.849 | 0.883 | 1.000 | 0.889 |
| L2-hub | 0.918 | 0.863 | 0.909 | 0.889 | 1.000 |

The two LightGBM members correlate 0.918 with each other, every cross-library
pair is 0.84–0.91, and the lowest number in the matrix is the second lineage
against retro-v1 at 0.840. Same block structure exp-008 found, on a different
member set and a different fold.

## The budget audit: only one member is still truncated

The desk asked for "CatBoost and XGBoost at their corrected budgets". Both are
already there, and saying so is worth more than rebuilding them, because it
saves about four GPU-hours. The patience-100 early stop that exp-008 shipped was
re-run at patience 900 for all three learners on the same fold and the same
early-stop month (`december`, es-month 2025-11):

| learner | patience 100 | patience 900 | truncation | already shipped at | worth |
|---|---|---|---|---|---|
| LightGBM Huber (`L2-hub`) | 1,926 → 2,100 | 7,990 → **8,725** | **4.2×** | 2,100 | **−3.57 s: the correction is harmful, see below** |
| CatBoost d8 (`C2-cat`) | 405 → 450 | 4,879 → **5,325** | **11.8×** | **5,325** | 15.5 s on july_loo |
| XGBoost d8 (`X2-xgb`) | 326 → 350 | 326 → **350** | **1.0×** | **350** | nothing to win |

Sources: `select-S-lgb-hub-pat900.json`, `select-S-cat-d8-pat900.json`,
`select-S-xgb-d8-pat900.json` against `select-S-lgb-hub.json`,
`select-S-cat-d8.json`, `select-S-xgb-d8.json`, all in
`experiments/exp-008-diversity/`. The CatBoost figure is measured, not inferred:
`C1-d8` at 600 rounds scores 330.62 on july_loo and `C2-d8p` at 5,325 scores
315.17.

**XGBoost returns the identical best iteration under a nine-fold longer
patience**, so it is genuinely converged and no correction is due. That is the
useful shape of the finding: the patience-100 instrument is not uniformly
biased, it is biased in proportion to how weak a learner each round is.
XGBoost at depth 8 and lr 0.05 is done in ~330 rounds; CatBoost's oblivious
trees at the same rate need an order of magnitude more, and were cut off at 8 %
of their budget. A truncation audit therefore has to be run per learner — a
single global "raise the patience" correction would have been three separate
answers.

So the rebuild list item (1) reduces to: the Huber arm's budget (running as a
one-fold kill test), seed bags, and the MLP.

## A blend keeps 7 % of a member's seed-bagging gain

The desk asked for seed bags of three. Before spending six hours of fits on
them, the cheapest test that could kill the idea: exp-008 already left a 5-seed
Huber bag on `july_loo` (`oof/SB-L2hub-july_loo.parquet`, per-seed columns
stored raw), so the question can be answered with no refit at all. Reproduce
with `python bagblend.py` (output in `logs/bagblend.txt`).

| seeds | Huber member | blend | member gain | blend gain | kept |
|---|---|---|---|---|---|
| 1 | 311.18 | 298.04 | – | – | – |
| 2 | 309.94 | 297.95 | 1.24 | 0.086 | **0.07** |
| 3 | 309.52 | 297.93 | 1.66 | 0.114 | **0.07** |
| 4 | 309.31 | 297.91 | 1.87 | 0.128 | **0.07** |
| 5 | 309.19 | 297.90 | 1.99 | 0.137 | **0.07** |

Every point averages all subsets of that size rather than the first k seeds, so
k = 1 is the average seed and not a lucky one. The weights are the `july_loo`
rotated vector, held fixed — they were fitted on december + january, which have
no bagged members, so they cannot honestly be refitted. Fixed weights understate
the gain, so the bound matters: refitting NNLS on `july_loo` **itself**, which is
in-sample and therefore the most a refit could possibly extract, gives 0.265 s at
k = 3 and 0.116 s at k = 5. **The gain is non-monotone in k under the refit**,
which is the tell that we are inside the noise.

**So seed bagging a member is worth 0.11 to 0.27 s to the blend, against 1.66 s
to the member — a retention of about 7 %.** The mechanism is not subtle once
measured: bagging averages away the independent part of a member's error, and a
five-member blend of predictors correlating 0.84–0.92 has already averaged away
most of the same thing. The two techniques are buying the same variance
reduction twice, and the blend gets there first.

The cost side: the Huber arm at its corrected budget is about 2,450 s per fold,
so three folds at one seed is ~2 h and at three seeds is ~6 h. **Four extra hours
for ~0.2 s.** The same argument applies to CatBoost, whose weight (0.19–0.28) is
close to the Huber's (0.26); it is untested there, and the mechanism is a
property of the blend rather than of the learner, so I expect the same retention
rather than assert it.

I have therefore ordered the rebuild as: corrected budget first at one seed,
then the MLP, and seed bags last, as the cheapest remaining tenth of a second
rather than as a member rebuild. This is a re-ordering inside the assignment,
not a close: the bags stay on the list with their price now measured.

## The retro slot is empty, and the de-icing arm will not fill it

Candidate 1's retro member `R0-retro` is a **linear-tree, single-seed,
patience-100** arm (`experiments/exp-001-retro-v1/climb.log`, `--name R0-jloo
--linear-tree --es-rounds 100`) scoring 324.76 on july_loo. That is three things
this experiment's SPEC rules out, and it is also nine seconds worse than
retro-v1's plain-tree arms. It carries 0.052 weight in the five-member fit and
is **not in the shipped candidate**, so candidate 1 is unaffected — but the slot
was clearly filled with the wrong arm, and the desk asked for retro-v1's
plain-tree member specifically.

The better arms have a hard blocker: **no plain-tree retro-v1 arm has a
`december` fold.** `A0-plain-jloo` (315.2), `Q2-plain-jloo` (315.9), `X1-deice`
(314.3 / 346.5) and `X2-wint3` all cover july_loo and january only; every arm
retro-v1 ran on december is linear-tree. So the whole plain-tree lineage, the
de-icing arm included, is locked out of a three-fold rotation by one missing
measurement.

Before asking retro-v1 for that fold, the screen (`screen_retro.py`, output in
`logs/screen_retro.txt`): fit NNLS **in sample on the scored fold** with each
candidate added to the three-member core. In-sample is the most weight a member
can possibly earn, so a member that earns little here earns less out of sample
and the negative is the trustworthy direction.

| retro candidate | alone (jloo) | weight | in-sample gain | corr with blend residual |
|---|---|---|---|---|
| `R0-retro` (linear, shipped fit) | 324.76 | 0.037 | 0.034 | 0.910 |
| `A0-plain` | **315.23** | **0.000** | **0.000** | 0.947 |
| `Q2-plain` | 315.93 | 0.000 | 0.000 | 0.945 |
| `X1-deice` (de-icing) | **314.29** | 0.069 | 0.061 | 0.940 |
| `X2-wint3` | 319.66 | 0.000 | 0.000 | 0.937 |

On january the same screen gives `X1-deice` 0.227 weight for 0.238 s and every
other candidate exactly zero.

**`A0-plain` is nine seconds better than `R0-retro` alone and earns strictly
less weight — zero.** That is the sharpest form of the lesson exp-008 learned
from XGBoost, and it runs the other way here: there, a member was different but
not good; here, a member is good but not different. `R0-retro`'s only claim to
weight was a residual correlation of 0.910 against the blend, and that came from
its linear leaves — the one property this experiment excludes for
reproducibility. Improving the retro member destroys the only thing it was
contributing.

**So the december fold is not worth requesting, and the de-icing arm will not
move this blend.** Best case across both folds is 0.06–0.24 s in sample, less
out of it, for a dependency on another lane. The retro slot stays empty and the
desk can stop holding a place for `X1-deice` in the member set.

One honest limit: this is measured against the *current* three-member core. If
the core changes — the MLP is the live candidate — a retro member's marginal
value is worth re-screening, and `screen_retro.py` re-runs in seconds.

## A fourth member exists, and it inverts the case for fitting

A repo-wide sweep for OOF sets covering all three board folds (every
`experiments/*/oof/`) turned up one I had not considered: **`d1-ds-s7` from
exp-007-winter**, a different lineage again (day-state / de-icing columns),
scoring **307.45** on july_loo — better than every member of candidate 1 except
the tail model, and better than the Huber arm. Screen it with
`python screen_member.py "d1-ds=experiments/exp-007-winter/oof/d1-ds-s7-{fold}.parquet"`.

Added as `D1-winter` to candidate 1's three, honestly rotated (`blend.py
members-2.json --stability`, saved as `blend-2.json`):

| variant | 3 members (candidate 1) | 4 members (+`D1-winter`) |
|---|---|---|
| rotated NNLS | **318.81** | 318.61 |
| rotated OLS | 318.81 | 318.60 |
| flat average | 318.98 | **318.21** |

**Adding the member is worth 0.20 s under NNLS and 0.77 s under a flat average,
and it flips which blending rule wins.** With three members the fit is worth
0.17 s over equal weights; with four it *costs* 0.40 s. The stability table says
why — `D1-winter`'s fitted weight swings 0.088 → 0.211 → 0.268 across the three
rotated vectors, a spread of 0.180, by far the largest this experiment has
measured (candidate 1's worst was 0.093). It is the only member that is worse
than the core on a fold it is asked to help (january: 351.98 alone, weight
0.000). Once four members are of comparable standalone strength — 326.3, 327.9,
327.9, 330.2 on the board estimate — three months is not enough data to fit four
weights, and equal weighting is both better and has nothing to be unstable.

### The variant choice is itself a selection, and I could not make it honestly

Reading "flat beats NNLS" off the three rotated folds is selection on the scored
folds — the same class of finding exp-008's review raised as N3. So I tried to
make the choice *inside* the rotation (`shrink_test.py`): for each held-out
fold, pick the mix `w = L·flat + (1−L)·NNLS` using only the two fit folds, one
fitting and one scoring, both directions averaged.

| rule | board estimate |
|---|---|
| rotated NNLS | 318.605 |
| **flat** | **318.208** |
| mix, `L` chosen in-rotation on a 0.1 grid | 318.273 |
| variant chosen in-rotation, `L ∈ {0,1}` | 318.338 |

The in-rotation choice picks `L* = 0.6 / 1.0 / 0.4` on the grid and lands at
318.27; restricted to a binary choice it picks NNLS on january and lands at
318.34. **Every honest selection mechanism I could build is worse than simply
taking the flat average.** So flat's 318.21 is the best number, and about 0.1 s
of it is variant selection I cannot defend from the folds alone.

What I can defend without the folds is the argument from the weights: four
members of near-equal strength, correlating 0.86–0.91, with a fitted weight
whose spread across held-out months is 0.180, is the textbook case for equal
weighting. That argument is independent of the scored RMSE, and it is the reason
I would ship flat rather than the 0.1 s.

### The member cannot be served yet

`d1-ds-s7` has out-of-fold predictions on all three folds and **no ranking
predictions**: there is no `d1-ds` file under `artifacts/`, and
`experiments/exp-007-winter/code/run_ds.py` has no `--submit` or ranking mode to
produce one (its arguments are `--name --arms --folds --seed --month-weight
--month-weight-alpha --ds-columns --fixed-rounds`). So candidate 2 cannot be
built from this member today, and the blocker is a submission path in exp-007's
lane, not a measurement in mine. This is the one dependency worth asking for:
unlike the de-icing arm screened above, this member earns real weight on two of
three folds.

## The NM-tail head: worth 7.5 s against one base model and nothing against four

The desk passed exp-010's NM-tail head over as a member, citing seven gains in
seven comparisons at weight 0.25 — `july_loo` −7.48 against `R0`, −2.23 against
`prc.tail_model`. Both numbers are right. Added to exp-012's four-member core
the honestly rotated value is **−0.015 s**. This section is why, and what to do
about it, because the "why" generalises to every member offered to this blend.

### It gets absorbed, and the absorption is measurable

`nmtail_absorb.py`, july_loo, NM-covered rows, weights fitted in sample at every
core size so the points are comparable (an upper bound at each):

| core the head is added to | core RMSE | +head | gain | weight | corr with core residual |
|---|---|---|---|---|---|
| `R0` alone (exp-010's base) | 268.73 | 257.87 | **10.85** | 0.466 | 0.823 |
| `T-tailv2` alone | 256.11 | 254.26 | 1.85 | 0.242 | 0.886 |
| + `L2-hub` | 249.37 | 248.93 | 0.43 | 0.120 | 0.891 |
| + `C2-cat` | 248.04 | 247.84 | 0.20 | 0.083 | 0.895 |
| + `D1-winter` (the core) | 247.55 | 247.44 | **0.115** | 0.064 | 0.897 |

**The head's marginal value falls by a factor of ninety-four between a
one-member base and a four-member one**, and its correlation with the blend's
residual *rises* from 0.823 to 0.897 as the core grows. The head's contribution
is an independent second opinion; exp-010 measured the second opinion, and this
blend is already on its fifth. Nothing is wrong with either measurement.

**The general rule this sets for members offered to the standing blend: a
member's value has to be quoted against the standing blend, not against a single
base line.** Measured against `R0`, this head, `A0-plain` and `X2-xgb` all look
like members; against the core, all three are worth under a tenth of a second.

### Added globally it is negative, and the split is most of the cost

`nmtail_rotate.py` — the head covers NM rows only (97.9 % of july_loo, 99.0 %
and 99.1 % of the winter folds), so the blend carries two vectors, one fitted
where every member is defined and one where only the core is. Filling the gap
instead would invent a prediction the head never made.

| | board estimate |
|---|---|
| core4, one vector, NNLS | 318.605 |
| core4, coverage-split, NNLS | 318.719 |
| core4 + head, coverage-split, NNLS | 318.734 |

Charged honestly against a **matched** control, the two-vector machinery costs
0.114 s and the member itself costs 0.015 s. Under flat weights the member costs
0.492 s, because flat forces 0.2 on a member NNLS values at 0.02–0.04. The `p`
variant of the head earns weight exactly 0.000 on all three folds; `phi` is the
better of the two and is used throughout.

### The repair: give it weight only where its mechanism lives

A negative closes a path, not a destination, so before recording it: the head
exists because the airport block clock sometimes sat on LOBT rather than AOBT,
and `nm_late` — NM-joined with AOBT − LOBT above 1800 s — is the **input-defined**
population where that is most likely. The prediction was made before the
measurement. Five-member weights inside `nm_late`, four-member weights outside,
all rotated (`nmtail_regime.py`):

| held-out fold | core (matched control) | + head in `nm_late` | gain | head's weight | `nm_late` RMSE |
|---|---|---|---|---|---|
| july_loo | 296.664 | **295.992** | **+0.673** | 0.297 | 481 → 475 |
| december | 214.462 | **214.152** | **+0.310** | 0.331 | 416 → 411 |
| january | 343.902 | **343.726** | **+0.176** | 0.336 | 458 → 454 |

**Positive on all three folds, and the head's weight is 0.297 / 0.331 / 0.336 —
a spread of 0.039, the most stable weight any member has held in this
experiment.** In `nm_bulk` the same fit gives it weight exactly 0.000 on all
three folds. A member that earns a third of the weight in one input-defined
regime and none in the other is behaving like its mechanism says it should,
which is the part that makes this a result rather than a fitted artifact.

### Where that leaves the blend

| configuration | board estimate |
|---|---|
| candidate 1 (three members, NNLS) | 318.81 |
| core4, NNLS | 318.605 |
| core4, flat | 318.208 |
| **core4 flat outside `nm_late`, five members fitted inside** | **318.017** |

The hybrid is the best measured configuration, 0.79 s under candidate 1.

**The selection I am carrying, stated plainly.** Choosing flat over NNLS was
already selection on the scored folds (0.1 s, disclosed above). This adds a
structural choice — regime split, which regime, which rule inside and outside —
and those degrees of freedom are not free. What is *not* selected on the folds:
`nm_late` was named from the head's mechanism before the number was computed,
the gain is positive on all three folds separately rather than only in
aggregate, and the head's weight is stable across the three rotations. Those
three facts are the reason I would ship it; the 0.19 s it beats plain flat by is
not on its own.

## What a member improvement is worth to the blend: 17 % flat, 0 % fitted

This prices any tuning of the members, so it is measured before any is done.
Seed bagging was measured at 7 % retention, but that is variance reduction and
the blend already performs it; a tuning gain is a genuinely better model and
should carry further. The largest member improvement this lane has is CatBoost's
round budget — `C1-d8` at 425–600 rounds against `C2-d8p` at 5,325 — so both are
swapped into the same rotated blend with everything else held
(`tuning_retention.py`).

| blend rule | member board | blend board | member gain | blend gain | retention |
|---|---|---|---|---|---|
| rotated NNLS | 344.46 → 330.23 | 318.556 → 318.605 | 14.23 | **−0.049** | **0 %** |
| flat | 344.46 → 330.23 | 320.699 → 318.208 | 14.23 | **2.490** | **17 %** |

**A fitted blend keeps none of it and a flat blend keeps a sixth.** The mechanism
is plain once seen: NNLS responds to a worse member by giving it less weight and
leaning on the others, so the fitted blend self-insures against member quality —
improving a member mostly just takes back weight the fit had already moved
elsewhere. Flat weights cannot reweight, so member quality passes through.

**Extended by exp-016 (2026-09-05).** They ran the same substitution through two
more rules and I verified their blend figure through my own rotation:

| rule | member delta | blend delta | retention |
|---|---|---|---|
| flat, four members | 14.23 | 2.49 | **17 %** |
| **hybrid** (flat + fitted in `nm_late`) | 14.24 | 1.83 | **12.9 %** |
| exp-016 stacker | 14.24 | 0.71 | **5 %** |
| rotated NNLS, four members | 14.23 | −0.05 | **0 %** |

**I first wrote that this was monotone in how much fitting each rule does. That
is wrong, and exp-016 caught it.** Their stacker fits 300 trees over nine inputs
against NNLS's four non-negative scalars, so by any parameter count it is the
most fitted rule in the set — and it retains *more* than NNLS, not less. The
ordering is not fitting.

**The axis is how far a rule can REALLOCATE WEIGHT AWAY from a degraded member**,
and on that it is exact: flat cannot reallocate at all (17 %); the hybrid
reallocates inside `nm_late` only (12.9 %); the stacker fits the residual from
the flat average, so flat is baked in as its baseline and it can learn to
distrust a member but never zero one (5 %); NNLS can drive a weight to zero
(0 %).

Tested directly on this experiment's own machinery rather than argued.
Substituting the degraded CatBoost and refitting the `nm_late` vector:

| member | weight, good C2-cat | weight, degraded | change |
|---|---|---|---|
| `T-tailv2` | 0.2132 | 0.2727 | +0.0596 |
| **`C2-cat`** | **0.2177** | **0.0000** | **−0.2177** |
| `L2-hub` | 0.0799 | 0.1614 | +0.0815 |
| `D1-winter` | 0.1454 | 0.1966 | +0.0512 |
| `N1-nmtail` | 0.3522 | 0.3755 | +0.0233 |

**Inside `nm_late` the degraded member is driven to exactly zero and the other
four absorb its share; outside `nm_late` the flat rule holds it at 0.25 whatever
it does.** One rule, both behaviours, in one candidate — which is as direct a
demonstration of the reallocation axis as the data can give.

Worth recording how the wrong version happened: the paragraph above this one
already said NNLS "responds to a worse member by giving it less weight and
leaning on the others", which *is* the reallocation mechanism. I named it
correctly for one rule and then, on adding three more, reached for parameter
count instead of generalising what I had already written.

**The symmetry, which is exp-016's framing and is better than the one this
section shipped with.** I had recorded this as a fact about tuning being
unrewarding. It is really one property with two faces: **insulation from a worse
member and unresponsiveness to a better one are the same thing.** So this
measurement cannot be quoted in one direction only — the number that says "do not
bother tuning CatBoost" is the same number that says "do not cheapen it either",
and anyone reaching for it to justify a cheaper member is reading it backwards.

Two consequences, and they point the same way:

1. **The desk's flat rule and member tuning are complements, not alternatives.**
   Under the fitted blend that candidate 1 ships, tuning a member is worth
   nothing; under the flat blend the four-member rule selects, it is worth about
   a sixth of what it buys the member. Tuning is only worth doing *because* we
   are shipping flat.
2. **It bounds what a tuning screen can return.** A coordinate screen that finds
   3 s on a member — a good result for hyperparameter tuning on top of a
   corrected round budget — moves the board estimate about 0.5 s.

One limit on the number: it is measured across a large degradation (a member
14 s worse), where flat weights are punished hardest. Retention between two
*good* members may be lower, so 17 % is an optimistic reading, not a
conservative one.

## The blend rule, and which vector each candidate uses

Set by the desk (2026-09-05) from the finding above: **with four members, ship
the flat average unless a fitted vector beats flat in every rotation.** The rule
is decided per candidate and the answer goes in the candidate record.

Applied to the four-member set, from `shrink_test.py`:

| held-out fold | rotated NNLS | flat | winner |
|---|---|---|---|
| july_loo | 296.931 | **296.416** | flat, by 0.515 |
| december | **214.461** | 214.462 | NNLS, by 0.001 |
| january | 343.937 | **343.666** | flat, by 0.271 |

NNLS wins one rotation of three, and by a margin four hundred times smaller than
the margin by which it loses july_loo. So the rule selects **flat**, which is
also the variant with the better board estimate (318.21 against 318.61) — the
rule and the number agree here, which is the case where the rule costs nothing.

Recorded for candidate 1 as well, retrospectively: it is a **three**-member
blend, the rule does not apply to it, and it ships rotated NNLS at 318.81
against flat-of-three's 318.98.

## A torch job took 29 cores against a 16-thread cap

The desk caught this while the MLP folds were running: `E12-mlp` was using 29
cores. `PRC_GPU=0` put the *model* on GPU 0, but PyTorch's intra-op pool defaults
to every core on the box, and this learner does real CPU work per batch —
`Ct[idx]` and `Ft[idx]` index 1.9 M-row tensors on the host before the copy to
device. So a job that looked like a GPU job was competing with the Huber and
winter fits for CPU on a machine at load 240–320.

Stopped on the desk's instruction; the MLP was measured at 0.11 s against the
core (saturation section above), so the remaining `january` fold was not worth
the cores. `oof/E12-mlp-july_loo.parquet` and `oof/E12-mlp-december.parquet`
survive if anyone wants the member later — but the finding it produced is already
recorded and does not need the third fold.

**The rule for any future torch run here: set `torch.set_num_threads()` to the
declared cap, or keep the work on GPU.** `learners.MlpLearner` does neither
today, and the tree learners' `THREADS` constant does not reach it. Noted rather
than fixed, because this experiment is not running torch again.

Related, and mine: `pkill -f "name E12-mlp"` matched its own command line and
killed both of my monitors and the shell running it along with the target. The
same self-match cost time earlier in this session with `pgrep -f`. A pattern
that appears in the killing command kills the killer.

## Boundedness is not the mechanism — exp-007's open question, answered

exp-007's finding 14 shipped a caveat: the `ds_deice_hr`-only arm beats the full
25-column day-state block on every fold, but it "does not isolate boundedness",
because the two arms differ by 24 columns and by how hard they push, not only by
whether the columns can leave their training range. They named the experiment
that *would* separate them — "the same 25 columns with each `_z` clipped to its
training range, which changes boundedness and nothing else" — and recorded that
it was never run.

It is nearly free to run from here, because clipping at predict time reuses the
**same fitted model**: one fit produces the block's prediction and its clipped
twin, so the comparison is paired and carries no seed, thread or round-count
noise. `winter_arm.py --clip`, one seed, six threads.

| fold | slice | rows | block | clipped | delta |
|---|---|---|---|---|---|
| july_loo | whole fold | 190,713 | 307.61 | 307.62 | +0.00 |
| july_loo | most disrupted 5 % of days | 9,909 | 586.86 | 586.90 | +0.05 |
| july_loo | rows the clip moved | 1,199 | 567.38 | 567.78 | +0.39 |
| **january** | whole fold | 153,706 | 354.11 | 354.10 | −0.01 |
| **january** | most disrupted 5 % of days | 8,037 | 400.85 | 400.71 | −0.15 |
| **january** | rows the clip moved | 4,117 | 291.48 | 291.08 | −0.40 |
| **january** | disrupted **and** moved | 1,550 | 430.96 | 430.25 | **−0.72** |

The disrupted-day slice uses exp-007's own label-free definition, imported from
`bootstrap_ds.disrupted_days` rather than re-implemented.

**Clipping recovers 0.72 s on the 1,550 january rows where it does anything, and
nothing anywhere else.** On january it moves 4,117 rows of 153,706 by a mean of
1.7 s; on july_loo it moves 1,199 rows and slightly hurts. Against exp-007's
+72.1 s for the block on those days, that is not the mechanism.

**So the answer to their open question is: boundedness is not what the
`ds_deice_hr` arm was buying.**

*Refined 2026-09-06, prompted by exp-018 asking how the column is fitted.*
`ds_deice_hr` is not only bounded, it is **reference-free**:
`DayState.transform` uses the fitted `reference_` for the nine `_z` columns
only, and joins `_deice(frame)` unconditionally, computed from the frame it is
handed. So the column carries no training reference at all — fitting DayState on
the training months, on all twelve, or not at all gives the same values. The 24
columns dropped include nine `_z` columns that *are* reference-dependent and can
leave their training range, plus raw day levels that are equally unbounded.
`ds_deice_hr` survived because it is the only one that is **both bounded and
reference-free**, which is a stronger safety property than the boundedness this
section originally credited. What could still have differed between consumers is
the FRAME, not the fit: `_deice` reads what it is handed. **exp-018 then measured
that too**, rebuilding 2025-03 masked and unmasked on a month where the mask
blanks 164,450 BLOCK cells: all 92 columns bit-identical, deice 1/1, surface
80/80, phi 11/11 (`experiments/exp-018b-climb/code/verify_extras.py`, non-zero
exit on any difference). So the frame does not matter either — the module's own
leakage guard blanks exactly the cells the mask blanks. The two lanes' deice
columns are the same object whatever either passes, which is now measured in both
directions: my code read says the fit cannot matter, their diff says the frame
cannot either. Bounding the same 25 columns, changing nothing
else, leaves the block essentially where it was. The deice arm's gain has to come
from the other thing that differs — dropping 24 columns, and pushing a third as
hard on the disrupted slice, which finding 14 already measured (+54.0 s of
movement against +19.1 s). Their caveat was right to refuse the boundedness
claim, and the refusal is now measured rather than cautious.

**Consequence for the repair.** The clipping option the desk offered does not fix
the exposure, so the repair is the `ds_deice_hr`-only member. That arm is running
now under the same six-thread conditions, so it can be compared to this block on
equal terms rather than against exp-007's differently-configured `f300` base.

## The repair: `ds_deice_hr` alone, and it is worth 51 s where it matters

Both repairs the desk offered were run at six threads on the two folds it named.
Clipping does nothing (previous section). Dropping to the one bounded column
does a great deal, and almost none of it is visible on the whole fold.

| fold | slice | rows | block (25 cols) | deice (1 col) | delta |
|---|---|---|---|---|---|
| july_loo | whole fold | 190,713 | **307.61** | 310.28 | +2.67 |
| july_loo | most disrupted 5 % of days | 9,909 | **586.86** | 600.69 | +13.83 |
| july_loo | rest of the fold | 180,804 | **284.49** | 285.96 | +1.47 |
| **january** | whole fold | 153,706 | 354.11 | **349.20** | −4.91 |
| **january** | **most disrupted 5 % of days** | 8,037 | 400.85 | **349.59** | **−51.26** |
| **january** | rest of the fold | 145,669 | 351.35 | **349.17** | −2.17 |

**On january's most disrupted days the bounded member is 51 s better**, which is
the same order as the +72.1 s exp-007 measured the block costing on that slice.
The whole-fold view understates it by a factor of ten, because 8,037 rows of
153,706 moving 51 s is 4.9 s on the fold — and that is exactly why the desk's
audit had to be a robustness argument rather than a leaderboard one.

The member's own board estimate barely moves: 329.02 for the block against
**328.09** for deice, a 0.93 s improvement. **The case for the repair is not the
0.93 s.** It is that the block reaches its fold score by extrapolating on
disrupted days — 4,640 january validation cells outside the training range
against the bounded column's **0** — and January 2026 has 18 such airport-days
and 7,564 rows, which no 2025 fold can price.

The july_loo cost is real and should not be glossed: +2.67 s on the fold and
+13.83 s on its disrupted days. July's disrupted days are congestion, not
de-icing, and the 24 dropped columns were carrying something there. That is the
trade the repair makes, and it is a trade rather than a free win.

**One caution on how to read this table.** Block-against-deice is *unpaired* —
two fits with separately drawn round counts, and a re-draw alone on this arm has
been measured at several seconds. 51 s is far outside that; +2.67 s is not
comfortably so. The paired clipping comparison in the previous section is the one
that carries no such noise.

**Adopted**: `ds_columns="deice"` becomes the winter member for blend-3.

## I destroyed a member artifact, and what it cost

The end-to-end rebuild's CatBoost member came back **MISMATCHED**: declared
deterministic in the config, it rebuilt from `48acb7e7…` to `56f95373…`. The
loud report was the intended behaviour. What followed was not.

**My protection logic only restored the original for members declared
NON-deterministic.** For a deterministic member that mismatched it kept the
rebuild, deleted the snapshot, and left the original gone — and `artifacts/` is
gitignored, so there was nothing to recover from. `48acb7e7` is the CatBoost
ranking prediction that **candidates 2 and 3 were both built from**, and it no
longer exists anywhere.

**What is and is not damaged.** The candidate files themselves are intact and
their hashes are unchanged, so the upload queue is unaffected: `f91db30a`
(blend-3) and `5c577999` (blend-2) are the same bytes they always were. What is
lost is the ability to rebuild them: one of their five inputs is gone. The
board estimate is also unaffected, because the rotation is fitted on OOF, which
did not move — rebuilding blend-3 with the replacement member gives the same
**317.661** and a different file (`43fd63d9…`, 344,396 rows differing, RMS
7.84 s). That gap between "same quoted number, different file" is precisely what
this pipeline exists to catch, and it caught it about itself.

**What I did on finding it.** Before any analysis, I copied every remaining
member ranking artifact to `artifacts/_member_backup/`, including L2-hub's,
which existed only as a live `.prerebuild` snapshot that the still-running job
would have deleted within the hour. Four originals secured; one lost.

**The fix.** The deterministic-mismatch branch now preserves both artifacts and
restores the original, exactly as the non-deterministic branch does. A mismatch
is a *finding*, and a finding is worthless without both objects to compare — the
old branch destroyed the evidence it had just produced. The message also warns
against the tempting repair, which is to update the expected hash and move on.

**The finding underneath, still open.** CatBoost on GPU 1 did not reproduce. The
config declared it deterministic on no evidence beyond it being a tree model.
The original ran at `PRC_THREADS=16`; the rebuild ran at 2 under compute allocation, so this may
be the same thread-count sensitivity the winter member has, or GPU histogram
non-determinism, or both. It cannot be separated without two runs at fixed
settings, which is ~46 GPU-minutes and is queued rather than assumed.

## The end-to-end rebuild, and what makes a member reproducible

exp-015's whole point was to find out whether the members rebuild. All five were
refit from raw data on 2026-09-05/06:

| member | built by | `deterministic=True`? | threads then → now | result |
|---|---|---|---|---|
| `T-tailv2` | retro-v1 pipeline | yes | 16 → 6 | **MATCH** |
| `D1-winter` | retro-v1 pipeline | yes | 6 → 6 | **MATCH** |
| `L2-hub` | exp-008 `LgbLearner` | yes | 16 → 6 | MISMATCH |
| `N1-nmtail` | `prc.nmtail` | **no** | 8 → 8 | MISMATCH |
| `C2-cat` | CatBoost, GPU | n/a | 16 → 2 | MISMATCH |

**The rule that fits every row is LightGBM's own documented one**: `deterministic
= True` gives reproducibility *"with the same data and the same number of
threads"*. Members carrying the flag reproduce at a fixed thread count and fail
across counts. The one member without it fails **at a fixed count**. CatBoost is
a separate library and also changed count.

Two intermediate claims of mine died on the way to this, and both are worth
keeping visible because each was a reasonable reading of the evidence available
at the time:

1. *"CatBoost on GPU is the problem."* Killed by `L2-hub`, which is plain-tree
   LightGBM on CPU and also failed. **CPU is not a safe harbour.**
2. *"Thread count is the entire explanation."* Killed by `N1-nmtail` failing at
   8 → 8. Before accepting that I checked whether an input had drifted instead —
   exp-003's ranking frame, the reocc cache, `prc.stand_features` and exp-003's
   feature code were all untouched, and `prc/nmtail.py` changed only *after* the
   subprocess had imported it. So it is genuine same-configuration
   non-reproducibility, not drift.

**The fix is a flag, not a workaround.** Adding `deterministic=True` and
`force_row_wise=True` to `prc.nmtail` should convert a member the release would
have had to document as unreproducible into one that passes gate 1. That
prediction is being tested before it is applied: the five-member separation now
running should report `N1-nmtail` NOT STABLE and the other four stable. If
`N1-nmtail` comes back stable twice at six threads, the mechanism is wrong and
this section gets rewritten rather than the story fitted to it.

### Why the pipeline caught this and a score check would not have

Every one of these three mismatching members produces a candidate with the
**same board estimate** — the rotation is fitted on out-of-fold predictions,
which did not move. Only the served bytes differ. A team checking their rebuild
by score alone would have seen 317.661 three times and concluded everything
reproduced. That is the argument for gate 1 existing at all, and it is also why
gate 2 is the honest gate for a *stranger*: the two answer different questions.

## The separation: four of five bit-stable, and the fifth had a missing flag

Each member run **twice at six threads**, one seed, short budget, each to its own
name (`run_sep_all.sh`). The question is not whether two thread counts agree —
they do not, and the original 16-thread conditions are unrecoverable — but
whether a member is reproducible at the **declared** count.

**These hashes identify SHORT-BUDGET SCRATCH runs, not the members.** The
separation fits at 300 rounds (150 for the NM-tail head, one seed) to its own
names, so that it cannot touch a real artifact — the right design, and then I
quoted the hashes to another lane as if they were member identities, which cost
them a verification pass against files that could never have matched. A hash is
meaningless without its artifact class. The numbers below are evidence that *a
configuration is stable*; they are not identities for *a member*.

| member | two runs at 6 threads (scratch, short budget) | notes |
|---|---|---|
| `T-tailv2` | **BIT-STABLE** | `e9d97994…` |
| `C2-cat` | **BIT-STABLE** | `da38d05c…` — **GPU CatBoost is stable at a fixed thread count** |
| `L2-hub` | **BIT-STABLE** | `5c700f3f…` |
| `D1-winter` | **BIT-STABLE** | `034fa693…` — **linear leaves are stable at a fixed count** |
| `N1-nmtail` | **NOT STABLE** | `10f9671d…` vs `51e98c54…` |

**Three of the evening's conclusions are overturned by this table, all in the
same direction — the problem was smaller than it looked.**

1. **CatBoost does not need to move to CPU.** The desk's contingency was that if
   GPU CatBoost were not bit-reproducible the release would fit it on CPU.
   It is reproducible, so the contingency is not needed and the GPU member stays.
2. **The winter member is not the fragile one.** I published it with a "do not
   refit this member" warning on the grounds that it could not reproduce itself.
   That was measured across thread counts; at a fixed count it is bit-stable, and
   the warning is being narrowed to what the evidence supports.
3. **The one genuinely non-reproducible member had a one-line cause**:
   `prc.nmtail` set neither `deterministic=True` nor `force_row_wise=True`, while
   retro-v1's `DEFAULT_PARAMS` and exp-008's `LgbLearner` set both. Every row in
   the table fits LightGBM's own documented guarantee — reproducible "with the
   same data and the same number of threads" — and the only member that failed is
   the only one that never asked for it.

`prc.nmtail` **VERSION 4** adds both flags. It changes the model's output bytes,
so it is a version bump with the old behaviour reachable and a note filed in
exp-010's RESULTS for the owning experiment. The re-test of that member alone is
running; if it comes back bit-stable, **all five members are reproducible at the
declared six-thread configuration** and gate 1 passes for the whole release
rather than four fifths of it.

## A thread change scrambles every row and moves the score by 0.002 s

The pinned OOF that fits the blend weights was built at 16 threads; the ranking
predictions the candidate serves are built at 6. That is an estimator mismatch of
the class this lane has flagged twice, and the question was whether it forces
rebuilding every member's out-of-fold predictions — several hours — or can be
disclosed. Rather than assume either, one fold of `L2-hub`'s OOF was rebuilt at 6
threads and compared (`run_oof_threadprobe.sh`):

| | july_loo RMSE |
|---|---|
| 16 threads (pinned) | 310.8363 |
| 6 threads (rebuilt) | 310.8342 |
| **delta** | **−0.0021 s** |

| | |
|---|---|
| rows differing | **190,322 of 190,713** (99.8 %) |
| max per-row difference | **319.8 s** |
| RMS per-row difference | **22.3 s** |

**Essentially every prediction moves, by an RMS of 22 seconds, and the fold score
moves by two thousandths of a second.** The thread count reorders floating-point
reductions; it does not make the model better or worse. The scrambling is noise
and it cancels.

This is the clearest statement of the two gates the release documents. **Gate 1
fails spectacularly here — 99.8 % of bytes differ. Gate 2 passes to three decimal
places.** A team reproducing this work on a machine with a different core count
will see exactly that pattern, and `docs/REPRODUCE.md` §4 now has a measurement
behind its claim that a different thread count gives "a different file with the
same score" rather than an assurance.

**Decision: the OOF side is not rebuilt.** A 0.002 s member delta cannot move a
blend that retains at most 17 % of a member change, and exp-016 independently
bounded the score consequence for their stacker at far under 0.71 s by a
different route. The mismatch is recorded in the manifest instead — OOF at 16
threads, ranking at 6, with this measurement as the reason it is acceptable
rather than an assertion that it is small.

## exp-018 changes the blend's shape, and the flat rule's escape clause fires

The desk offered exp-018-fresh as a fifth general member on the hypothesis that
it "should decorrelate". **It does not decorrelate — and it does not need to.**

| | |
|---|---|
| `F1-fresh` correlations with the four members | **0.870 – 0.900** |
| the four members with each other | 0.863 – 0.909 |

Squarely inside the range they already occupy. What it has instead is quality:
**293.63 on july_loo against the best other member's 306.77**, and a board
estimate as a *single model* of **316.47** — better than blend-3's five-member
317.66. That is the first time anything in this repo has beaten the standing
blend on its own.

### Preview rotation (pre-3b artifacts; not a candidate)

| member set | rotated NNLS | rotated flat |
|---|---|---|
| four members (blend-3 core) | 318.15 | 318.06 |
| **+ `F1-fresh`** | **313.45** | 315.09 |

**Adding it is worth 4.70 s** — an order above anything else measured in this
lane — and it *reverses which blending rule wins*. Applying the desk's rule per
rotation:

| held-out fold | NNLS | flat | winner |
|---|---|---|---|
| july_loo | **289.741** | 292.438 | NNLS |
| december | **208.050** | 211.704 | NNLS |
| january | **340.938** | 341.464 | NNLS |

**A fitted vector beats flat in every rotation, so the rule selects fitted.**
This is the escape clause firing for the first time, and it fires for the reason
it exists: flat was never the right answer in general, it was the right answer
*while the members were of comparable strength*. They no longer are — `F1-fresh`
is 316.47 on the board estimate against 326–330 for the rest — and equal
weighting now wastes the best member. The fitted vector gives it **0.596** and
splits 0.404 across the other four.

The saturation finding is not overturned; it is bounded. Four candidate members
were each worth about a tenth of a second because each was *about as good as the
core*. This one is thirteen seconds better than the best member on the fold that
carries 56 % of the board. **Saturation was never a statement about how many
members a blend can hold — it was a statement about members that bring nothing
the blend does not already have.**

And the blend still earns its place: 313.45 against `F1-fresh` alone at 316.47,
so the four older members are worth 3.02 s on top of it.

### A caveat on the 4.70 s, from exp-018's owner

They flagged that **their published OOF is one seed while their candidate is
two** (their review round 1, finding 3). So the preview above fits weights on a
one-seed estimator and would serve a two-seed one — the served-is-not-scored
mismatch this repo has now hit four times.

It biases in a knowable direction. The served two-seed member is *better* than
the one-seed OOF the fit sees, so the fit **under**-weights it: 4.70 s and the
0.596 weight are both understatements of what a matched pair would give. That is
the comfortable direction to be wrong in, and it is still wrong.

Asked them to publish an OOF and a candidate at the **same** seed count,
whichever count they pick, in preference to spending on three seeds — matching is
the part that changes correctness rather than the fourth decimal. On the seed
count itself I gave them this lane's measurement (7 % retention) with the caveat
that it was taken on a member carrying 0.26 weight and theirs carries 0.596, so
it does not transfer cleanly and I have not re-measured at that weight.

**Nothing is shipped on the preview number until a matched pair exists.**

## Pre-registered: what the s1-vs-s3 pair should show

Written **before** exp-018 publishes the matched pair, at their suggestion, so
the prediction cannot be fitted to the result afterwards.

The experiment: same lineage, same features, same fold code, both OOF and
candidate matched on each side, differing only in seed count (1 against 3), with
the member carrying ~0.596 weight. It measures seed-bagging retention through
this blend at a weight that matters, replacing a 7 % figure measured once on a
0.26-weight member — a figure currently load-bearing, since it is why the desk
cancelled a 31-to-126-hour tuning screen.

**exp-018's prediction (theirs, recorded as theirs).** If lineage diversity
drives retention rather than weight, retention should track residual correlation
with the other members. Theirs is 0.870–0.900, at the *top* of the members' own
0.863–0.909 rather than outside it, so on their story retention should be only
**modestly above 7 %**. If it comes back near 30 %, neither story is right.

**My prediction, and it exposes an asymmetry in my own number.** The 0 % I
measured for rotated NNLS was measured on a member **degradation** — CatBoost
swapped from 5,325 rounds to 600. NNLS answered by reweighting *away* from the
worsened member, which is why nothing reached the blend. An **improvement** is
not the mirror of that: NNLS reweights *toward* a better member, so the fit works
in the improvement's favour instead of cancelling it.

So I expect **retention for an improvement under a fitted rule to exceed
retention for a degradation**, and my 0 % / 7 % / 12.9 % / 17 % series to be a
lower bound when read in the improving direction. If the pair shows retention at
or below 7 %, that asymmetry does not exist and the series is symmetric after
all — which would be the more surprising result and the more useful one.

Both predictions are falsifiable by the same measurement, and they differ: theirs
says "modestly above 7 %", mine says "above the degradation figure, possibly well
above". Recording both.

## Standalone RMSE is the wrong measure of a member, and I have been using it all session

Trying to fill a cell exp-018 identified — a quality change with the seed count
held fixed — I re-used the Huber budget test, which is exactly that: same seeds,
2,100 rounds against 8,725. The result overturns how this lane has been screening
members.

**The member gets 3.57 s worse. The blend gets 0.18 s better.** Under *flat*
weights, where nothing can reweight, so this is not the self-insurance mechanism.

| slice | rows | member @2,100 | member @8,725 | delta |
|---|---|---|---|---|
| whole fold | 190,713 | **310.84** | 314.41 | +3.57 |
| bulk ≤ 3600 s | 190,057 | **271.72** | 276.67 | +4.95 |
| tail > 3600 s | 656 | 2588.08 | **2561.45** | **−26.62** |

| slice | blend @2,100 | blend @8,725 | delta |
|---|---|---|---|
| whole fold | 296.732 | **296.550** | **−0.183** |
| bulk ≤ 3600 s | **259.723** | 259.774 | +0.051 |
| tail > 3600 s | 2460.611 | **2452.634** | **−7.977** |

And its residual correlation with the other three members averaged **falls from
0.9309 to 0.9132**.

**The longer-budget member is worse at what its own RMSE measures and better at
what the blend needs**: less correlated with its colleagues, and stronger on the
656 tail rows that carry a quarter of the fold's squared error. Its own RMSE is
dominated by the 190,057 bulk rows; the blend's marginal value is not.

### What this revises

Earlier tonight I recorded the corrected Huber budget as a **negative** — "the
corrected budget is worse for the Huber arm, −3.57 s" — and concluded that
`L2-hub` ships at 2,100 rounds and needs no rebuild. **That conclusion was
reached on member RMSE, and member RMSE is the wrong criterion for a member.**
From the blend's side the 8,725-round arm is the better member.

The correction is small in points and large in method. It is 0.18 s on one fold,
and I only have the 8,725-round arm on `july_loo`, so acting on it means two more
folds at six threads — behind blend-4, which is worth 4.7 s. But every member
screen in this document ranked candidates by standalone fold RMSE, including the
four that produced the saturation finding. Those four were also *more correlated*
than the members they were joining, so the ranking probably survives; "probably"
is doing real work in that sentence and it is now a known gap rather than an
assumption.

### For the retention experiment

exp-018 flagged that seed bagging moves quality and variance together and a
single retention number cannot separate them. This says something sharper:
**"quality" measured as standalone RMSE is not the axis either.** A member change
can be negative on its own RMSE and positive in the blend, in the same fold, with
no reweighting available to explain it. Whatever the s1-vs-s3 pair returns, it
measures the retention of *a bundle* — quality, variance, and a change in
complementarity — and the third of those is not small.

## The screening rule, corrected (desk, 2026-09-06)

**A member is screened by what it adds to the blend, not by its own score.**

Every candidate screen in this document — `A0-plain`, `X1-deice`, `N1-nmtail`,
`E12-mlp`, and the four-member saturation table built from them — ranked
candidates by standalone fold RMSE and then checked the blend gain second. That
ordering is backwards, and the Huber budget case shows why it can invert the
answer: a member 3.57 s *worse* on its own RMSE made the flat blend 0.18 s
*better*, because its errors moved to where the blend was weak and its
correlation with the other members fell.

The criterion is the **rotated blend RMSE with the candidate in**, against the
same rotation without it. Standalone RMSE is a diagnostic, not a screen: it tells
you whether a member is a competent model, which is necessary and not sufficient.

Two things follow that are worth stating separately, because they pull in
opposite directions:

- **The saturation conclusion probably survives.** The four candidates it rests
  on were each *more* correlated with the core than the members they were
  joining, so a complementarity criterion should rank them no higher. "Probably"
  is doing real work in that sentence.
- **It has not been checked.** The desk has ordered the four screens re-run
  under the blend criterion, after blend-4. Until then this is a known gap in a
  headline finding, flagged rather than quietly assumed away.

## Compute rule compute allocation adopted, and what it cost

`docs/COMPUTE.md` (2026-09-05 19:50) gives this lane **6 CPU threads plus GPU 1**,
with a launch gate at one-minute load 140. Arm A of the day-state audit was
running at `PRC_THREADS=24`, four times the allocation, so under the rule's
"restart now if the current one would take more than an hour under the old
setting" it was stopped rather than reduced at the next fit — its *current* fit
had already run 2 h 50 m without finishing seed 7 on july_loo.

Compliance is made durable rather than remembered: `launch.sh` exports
`PRC_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`NUMEXPR_NUM_THREADS` to the allocation, refuses to start above the gate, and
every run script in this experiment now defaults to 6 instead of 24. A GPU fit
passes `CAP=2`, since GPU work counts only its feeding threads.

**What the restart costs, and how the plan changes to absorb it.** Three seeds of
the winter arm on two folds is roughly four hours at six threads. It does not
need three seeds: the question arm A answers is *clipped against unclipped*, and
that comparison is **paired on the same fitted model**, so it carries no seed
noise at all. One seed answers it in about a third of the time. Seeds are only
needed if the repaired member is then shipped, which is a later question. The
relaunch is therefore one seed, two folds, ~1.3 h at six threads.

**What was salvaged before the stop**: on july_loo the full 25-column block puts
**3,229 validation cells outside its training range** and `ds_deice_hr` puts
**0**. That is the boundedness mechanism measured directly rather than argued,
and it survives the restart because it is a property of the features, not of the
fit.

## Two process errors of mine, recorded because both nearly shipped something wrong

Neither changed a number, but both could have, and the second one produced an
artifact that a candidate would have been built from.

**`pkill -f` matched its own command line.** `pkill -f "name E12-mlp"` killed the
MLP, both of my monitors, and the shell running the command, because that shell's
own argv contains the pattern. The same self-match cost time earlier in the
session with `pgrep -f`. A pattern that appears in the killing command kills the
killer; match on the pid instead.

**`kill <pid>` on a `nohup`-ed shell does not kill its python child.** I stopped
the winter submission's wrapper when the desk's budget rule arrived, reported it
stopped, and it was not: the python process had already loaded the module, so it
ran the *old* single-slice budget to completion twenty-five minutes later and
wrote `artifacts/exp-012/pred/D1-winter-ranking.parquet` (165 rounds,
sha `9b19cae2…`) — a **non-compliant** artifact sitting at the exact path the
compliant run writes to. Had I built candidate 2 in that window it would have
served a budget the desk had just replaced, with nothing in the record to say so.
Kill the process group (`kill -- -$pid` after `setsid`, or kill the child pid),
and verify the child is gone rather than the wrapper.

The recovery is that the compliant run overwrites the same path, so the fix is to
wait for it rather than to repair anything. The near-miss is the point.

### A rehearsal of the candidate build caught a stale member path

Before building candidate 2 I ran the whole builder against a throwaway number,
because a bug found there costs nothing. It found one: **the analysis scripts
each carry their own hardcoded member paths**, independent of `members-*.json`,
and five of them still pointed at exp-009's *unsuffixed* file — which reverted
to v1's bytes at 14:42. The rehearsal therefore blended v1 and reported a board
estimate of 318.168 where the same configuration measured 318.017 earlier in the
day.

Repointing them at the versioned v2 path and at the pinned exp-008 copies brings
the builder to **318.017 with a core-4-flat reference of 318.208** — identical
to what `nmtail_regime.py` measured independently, which is now a cross-check of
the builder rather than a coincidence.

The lesson is narrower than the pinning rule and worth stating separately:
**pinning the member configs is not enough while analysis code holds its own
copies of the paths.** `pin_members.py` verifies what `members-*.json` points at;
it cannot see a path hardcoded in a screening script.

## The desk's submission budget rule, and where each member stands

Set by the desk after the Huber correction (2026-09-05): **a member's submission
budget is the median of its per-month optima over at least three held-out months
spanning both seasons, the spread is reported, and a wide spread takes a smaller
count.** It exists because a count drawn on one November month cost the Huber arm
3.57 s on July.

Implemented in `winter_arm.budget_from_scan` over 2025-01, 2025-07 and 2025-11.
"Smaller count on a wide spread" is fixed in code rather than judged at the time
of use: with three points the available order statistics are min, median and
max, so a relative spread above half the median takes the **minimum**
(`WIDE = 0.5`). One limitation is stated rather than hidden — the held-out
month's rows leave the tree fit, but the lirf, phi and day-state statistics are
still the twelve-month ones. That matches how the count being replaced was
drawn, so the scan is comparable to what it replaces; a fully clean version costs
three more frame builds at ~490 s each.

**Compliance across the member set, honestly:**

| member | how its submission budget is drawn | compliant | cost to comply |
|---|---|---|---|
| `D1-winter` | **median of three held-out months, both seasons** | **yes** | done |
| `C2-cat` | `m11 × 12/11` from one es-month, 2025-11 | no | ~1.3 GPU-h (3 CatBoost early stops) |
| `L2-hub` | `m11 × 12/11` from one es-month, 2025-11 | no | ~4–6 CPU-h at current load |
| `T-tailv2` | `n_estimators` 500/300 declared in `prc.tail_model` | n/a | exp-009's lane |
| `N1-nmtail` | `n_estimators = 900` declared in `prc.nmtail`, no early stopping | n/a | ~0.5 h to justify 900 |

Two of the five draw a count by early stopping on a single month and are
therefore exposed to exactly the failure the rule exists to prevent; two others
use a declared constant with no early stop, so there is no per-month optimum to
take a median of and the rule does not bite — what those need is a scan to
justify the constant, which is a different question.

**`L2-hub` is the interesting case: it is non-compliant by method and validated
by measurement.** Its 2,100-round budget came from one November es-month, which
is the disfavoured instrument — and the kill test above shows the alternative
count is 3.57 s worse, so the number it produced is right even though the way it
was produced is not. Re-drawing it under the rule would cost 4–6 hours to
confirm a count already measured better than its only competitor. I have not
done it, and that is a judgement the desk can reverse cheaply.

## Correction: the corrected budget is worse for the Huber arm, and the general claim was over-stated

The kill test finished. **The corrected budget loses 3.57 s**, and this walks
back part of the finding exp-008 reported as its largest.

| `L2-hub` on july_loo | rmse | bulk | tail | no_nm | noac |
|---|---|---|---|---|---|
| 2,100 rounds (patience 100, shipped) | **310.84** | **271.72** | 2588 | 1265 | 3880 |
| 8,725 rounds (patience 900, "corrected") | 314.41 | 276.67 | **2561** | 1268 | 3880 |

**The whole loss is in the bulk** — 271.72 → 276.67 across 99.7 % of rows —
while the tail improves slightly and the noac head, unchanged at 60 rounds, is
identical to the last decimal. The extra 6,600 trees buy a little on the rows
that carry the tail and overfit the rows that carry the fold.

The diagnosis is not "patience 900 was the wrong instrument". **Both counts were
selected on the same early-stop month**, 2025-11 (`select-S-lgb-hub.json` and
`select-S-lgb-hub-pat900.json`): patience 100 stopped at 1,926 and patience 900
kept going to 7,990, so on November data the extra rounds really did keep
helping. They then hurt on July. That is a **month generalisation gap**, not an
early-stopping artefact, and it is invisible to any instrument that selects the
count on one month and applies it to another.

### What this does and does not change

exp-008 reported, and the desk logged as the day's largest finding, that "the
round-count instrument is biased low and a corrected budget is worth 18.9 s on
july_loo". The measurement behind it was CatBoost's, and it is intact. The
**generalisation** of it — that a patience-100 stop truncates and that raising
the patience recovers score — is what this refutes. Three learners, three
answers:

| learner | patience 100 → 900 | corrected budget is worth |
|---|---|---|
| CatBoost d8 | 405 → 4,879 (11.8×) | **+15.5 s** (measured, `C1-d8` 330.62 → `C2-d8p` 315.17) |
| XGBoost d8 | 326 → 326 (1.0×) | nothing; already converged |
| **LightGBM Huber** | 1,926 → 7,990 (4.2×) | **−3.57 s** |

**A truncation audit has to be run per learner and per fold family, and its
answer can have either sign.** The budget-audit section above already said the
first half of that; this supplies the second, which I did not anticipate.

**Practical consequence: `L2-hub` ships at 2,100 rounds and needs no rebuild.**
That closes the last open budget question in this lane, and it is the reason the
cancelled tuning screen would have been more expensive than I quoted — its Huber
axis was priced at the 8,725-round budget, which is both slower and worse.

## The winter member is not reproducible, and the blend does not care

The desk's gate for taking exp-007's `d1-ds` arm over was: rebuild it and confirm
it lands within seed noise of 307.45 on july_loo. It does — and getting there
found something worth more than the gate.

`winter_arm.py`, same pinned pipeline snapshot, same `day_state[VERSION=1]`, same
121 features, **same seed 7**, same `PRC_THREADS=24`:

| | rounds | july_loo |
|---|---|---|
| exp-007, seed 7 | 148 | **307.45** |
| exp-012, seed 7 | 186 | **311.30** |
| exp-012, seed 107 | 186 | 311.44 |
| exp-012, seed 207 | 186 | 307.72 |
| **exp-012, three-seed bag** | 186 | **308.22** |

**Same seed and same code drew a different round count — best iteration 158
against exp-007's ~126 — and a 3.85 s different score.** This arm is
`linear_tree=True`, and exp-008 already measured that LightGBM's
`deterministic=True` does not cover linear leaves across thread counts; this is
the same defect reached from a different direction, on an arm whose round count
is then chosen *by* the non-deterministic fit. My own three seeds span 3.7 s, so
307.45 is inside the distribution — it is simply the favourable end of it, and
**the member that flipped this blend from fitted to flat was justified on a
single lucky draw.**

That is the kind of sentence that usually invalidates a finding, so it was
measured rather than assumed. Substituting the three-seed rebuild for exp-007's
single seed on july_loo, everything else held:

| D1-winter version | flat board | NNLS board |
|---|---|---|
| exp-007 single seed (used throughout above) | 318.208 | 318.605 |
| exp-012 three-seed bag | **318.222** | **318.599** |

**A 0.77 s change in the member moves the blend 0.014 s.** The finding stands:
the member's contribution was never about its exact level, and the retention
measurement predicted this — a small member change retains even less than the
17 % measured across a large one. So the rebuild is not re-run on december and
january, which would have cost two hours to move the fourth decimal.

**What this does change is what gets served.** The out-of-fold predictions that
fit the weights are exp-007's single seed; the ranking predictions are a
three-seed twelve-month bag, so the served member is not literally the estimator
whose weight was fitted. That is exp-008's B2 failure by construction — except
that here the size of the mismatch is measured at **0.014 s** rather than
assumed away, and the alternative is serving one draw of an arm with a 3.7 s seed
spread. Serving the bag is the better estimator and the mismatch is quantified;
both facts are in the candidate record.

## The tail member resolved to a versioned address (desk, 2026-09-05)

exp-009 published the seed-averaged noac head at a stable versioned path,
`experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet`, tracked and clean,
and reverted the unsuffixed file to v1. **Its sha256 `a14d0403…` is byte-identical
to the copy pinned here**, so candidate 1 and every measurement in this document
already used v2 and nothing numeric changes — candidate 1 rebuilds to
`541b1b11…` at 318.81 from the new path.

That is also the pinning rule earning its keep in one step. The unsuffixed file
`exp-009-noac2-oof.parquet` changed to v1's bytes (`7dacc589…`) at 14:42 while
this experiment was measuring against it; without the pin the member would have
silently become the worse head mid-experiment. The duplicate pin is now dropped
and the members point at the versioned path, which `pin_members.py` records as a
git blob and re-verifies on every run.

**Two numbers in the hand-off do not match the artifacts**, flagged for the
record rather than as an objection — the instruction to use v2 is unaffected:

| fold | v1 (`7dacc589…`) | v2 (`a14d0403…`) |
|---|---|---|
| july_loo | 308.01 | **306.77** |
| december | **222.59** | 222.86 |
| january | 349.52 | **349.36** |
| july | 327.77 | **326.64** |

v2 wins three folds of four, not all four: **december goes the other way** by
0.27 s. And the quoted v1 july_loo of 310.18 is not this v1, which measures
308.01; 310.18 is close to `prc.tail_model`'s 310.39, so the comparison behind
that figure may be against the exp-002 tail model rather than the v1 noac head.
v2 is still the better member on the two folds that make the board estimate.

## Review round 1: one blocking finding, and it was real

Reviewer `claude-opus-5-xhigh-2026-09-05` (substitute, desk rule), one
round, output in `review/round1.txt`. Verdict: **one blocking finding, eleven
non-blocking, candidate verified clean** including a bit-for-bit rebuild and an
artifact-level check that the served member is the estimator whose weight was
fitted — the two failures exp-008 shipped are both closed here.

**B1 — the largest member's OOF had no verified address.** `T-tailv2` carries
0.4438 of candidate 1, and its source
(`experiments/exp-009-noac2/oof/exp-009-noac2-oof.parquet`) is **uncommitted and
still being edited by exp-009**. Confirmed by recomputation: working tree
`a14d0403…` against HEAD `7dacc589…`, differing on **816 rows, every one of them
`noac`**, because exp-009 changed the noac head after the file was committed.

| fold | member, working tree | member, committed |
|---|---|---|
| july_loo | **306.77** | 308.01 |
| december | 222.86 | **222.59** |
| january | 349.36 | **349.52** |
| july | 326.64 | 327.77 |

The reviewer's number for the blend on committed bytes is 319.06 against the
318.81 I reported. **Accepted and fixed**: the exact bytes the weights were
fitted on are now copied to `oof/T-tailv2-pinned.parquet`, hashed in
`MANIFEST.md`, and every `members-*.json` points at that copy. Rebuilding
candidate 1 against it reproduces SHA `541b1b11…` and board estimate 318.81
unchanged, so this is a provenance fix, not a number change.

I kept the working-tree version rather than reverting to the committed one
because it is the better member on `july_loo`, which is 55.7 % of the board mix,
and because it is what every measurement in this document was made against.
**The general problem is worth the desk's attention: exp-009's OOF is
uncommitted and at least two lanes are consuming it.** A member being edited
underneath its consumers is the same failure exp-008 shipped as B2, arriving
from the other direction.

*Limitation of this review.* It was run with `claude -p`, which returns only the
final turn, so `review/round1.txt` holds the reviewer's closing summary and not
the eleven non-blocking items themselves. The blocking finding is recorded and
fixed; the non-blocking list is lost. Round 2 will capture the full transcript.

## Member tuning screen: not run (desk decision, 2026-09-05)

**The desk cancelled it on the estimate below, and the retention measurement is
the reason.** A flat blend keeps 17 % of a member improvement and a fitted blend
keeps 0 %, so 31–126 hours of a box already at load 220–300 buys about half a
second of board estimate. The axes are recorded here so a later session with a
quiet box can pick them up without re-deriving them; nothing about them was
found wanting, they are simply priced above what they can return.

## Expected compute for the member tuning screen (as estimated, for the record)

Asked for by the desk before the screen runs, so the box's load can be weighed
against it. All per-fit costs below are **measured on this box under its current
load** (average 240–320 on 128 cores), not extrapolated from an idle machine —
the difference is large: the Huber arm's 8,725-round fold was predicted at 41
minutes from its 2,100-round time and has been running **1 h 47 m and counting**,
a 2.6× degradation.

Measured per-fit costs: CatBoost 5,325 rounds on GPU **1,500 s** (1,476/1,569/1,517
across the three folds); Huber LightGBM **589 s** at 2,100 rounds and **>6,400 s**
at 8,725; the winter arm **~1,000 s** per seed plus a 520 s frame build that is
cacheable across configs.

A coordinate screen varies one axis at a time from a baseline that is already
measured, so the desk's axes come to **22 new configurations**: CatBoost 7
(depth 6/10, lr 0.03/0.06, l2_leaf_reg 10/30, border_count), Huber 11 (delta ×2,
num_leaves 63/127, min_data 300/1000, feature_fraction 0.7/1.0, lambda_l2
0/10/100), winter 4 (lr ×2, leaves ×2).

**As specified** — 22 configs × 2 folds × 3 seeds = **132 fits**:

| member | fits | per fit | total |
|---|---|---|---|
| CatBoost (GPU) | 42 | 1,500 s | 17.5 h |
| Huber (CPU, at the corrected 8,725 rounds) | 66 | 6,500 s | **119 h** |
| winter (CPU) | 24 | 1,000 s | 6.7 h |

GPU and CPU run in parallel, so **wall ≈ 126 h — five and a half days**, and the
Huber arm is 94 % of it purely because its corrected budget is 4.2× the shipped
one.

**What I would run instead**, same axes, staged so the expensive budget is only
paid by configurations that have already earned it:

- *Stage 1*, screen on `july_loo` only (55.7 % of the board mix), 3 seeds, Huber
  at its shipped 2,100 rounds: 21 CatBoost fits (8.75 GPU-h), 33 Huber fits
  (5.5 h), 12 winter fits (3.3 h) → **wall ≈ 9 h**.
- *Stage 2*, confirm the best configuration per member on `january`, and the two
  best Huber configurations at the corrected 8,725 rounds → **wall ≈ 22 h**.

**Total ≈ 31 h against 126 h, for the same answer on the axes that move.** The
risk it takes is real and worth naming: screening the Huber at 2,100 rounds
biases toward configurations that converge fast, because `num_leaves`,
`min_data_in_leaf` and `lambda_l2` all interact with the round count. Stage 2
re-runs the survivors at the full budget, which catches a wrong winner but not a
good configuration wrongly eliminated in stage 1.

**What it can return.** Retention above says a flat blend keeps ~17 % of a member
improvement, and that 17 % is itself optimistic. A coordinate screen finding 3 s
per member on top of an already-corrected round budget — a good outcome — is
about **1.5 s of member gain and 0.5 s of board estimate** across the three
members. That is the honest expected value to weigh against the hours.

I am not starting it until the four jobs now in flight clear, because adding 130
fits to a box already at load 240–320 would slow the winter and Huber runs that
the current member set depends on.

## In flight

- Huber LightGBM at its corrected budget (8,725 rounds against the shipped
  2,100 — a 4.2× truncation by the patience-100 instrument). Run as a one-fold
  kill test before committing nine fits.
- Three-seed bags for CatBoost and Huber, now priced at ~0.2 s for ~4 h and
  ordered last accordingly (see the retention section above).
- The MLP, running on july_loo at 12 epochs, three seeds, as a screen rather
  than a tuned member: its 30-epoch curve does not identify a budget (train RMSE
  falls 462 → 232 while the early-stop month bounces 281/359/285/316/566 with no
  trend after epoch 9, so `best_iteration = 9` is a lucky draw, and a budget
  cannot be read off that curve in principle because the LR schedule is OneCycle
  over the declared budget).
- The winter member's submission path, now owned here: `winter_arm.py`, with the
  july_loo reproduction gate against exp-007's single-seed 307.45 running.
- ~~The de-icing arm~~ — screened and dropped, see the retro-slot section
  above: it earns 0.06 s in sample on july_loo. Not worth its december fold.

## The instrument for the clip test: tail centrality (2026-09-06)

exp-018 proposed a sharper test than the rotation for whether their +/-6000 clip
costs the blend. The rotation says what a change cost; this says why. The
quantity is the change in their member's **mean residual correlation with my
other members, on tail rows only**, between `s1m` (clipped) and `s1m-noclip`.
A member can be worse in the tail and keep its weight if it is *differently*
wrong; it loses twice only if its tail also moves toward the consensus. That is
the step my mechanism assumed and had not measured under a clip.

`tail_centrality.py` computes it. It prepares every member the way the blend
does (60 s floor, then the residual guard at the fold's bounds) so the residuals
are the ones a weight would actually be fitted on.

**A contamination the self-test caught.** Calibrating on the Huber pair, the arm
under test (`L2-hub` at 2,100 rounds) is *also* one of the four comparison
members, so it contributed a self-correlation of 1.0 to arm A and its
merely-high correlation with arm A to arm B. That inflated the level from 0.792
to 0.844 and put a spurious difference into the delta — the whole quantity being
measured. The tool now refuses to run when an arm's filename is in the
comparison set and `--exclude` was not passed. It does not bite on exp-018's
arms, whose member is not in my set; it bit on exactly the pair that sets the
scale.

**The scale, recomputed clean.** On `july_loo`, tail = 656 rows of 190,713
(0.34 %):

    HUB2100 (worse tail)   tail rmse 2588.1   mean tail corr 0.792
    HUB8725 (better tail)  tail rmse 2561.5   mean tail corr 0.789

so **0.0033 of correlation per 26 s of tail RMSE** under a round-budget change.
The earlier 0.777 / 0.773 in this file is the same delta (0.004) at a different
level, because the level depends on which members are in the set.

**The delta is set-independent, the level is not.** Across four comparison sets
the tail RMSE change is identical and the centrality change moves only in the
fourth decimal:

    T-tailv2, C2-cat, D1-winter   -0.0033
    T-tailv2, C2-cat              -0.0034
    C2-cat,   D1-winter           -0.0031
    T-tailv2, D1-winter           -0.0033

This matters for the test: exp-018's arm will be compared against whatever set is
current when it lands, and the yardstick survives that.

**Pre-registered, before their arms land.** A clip should move the tail by far
more than 26 s, so the null is not "no change" but "change at the budget scale".
If the clip moves tail RMSE by dT seconds, the no-coupling prediction is a
centrality change of 0.0033 * dT/26. Coupling is indicated only if the observed
change exceeds **twice** that. At or below it, the clip makes the member worse in
the tail without making it more central, my "sells tail into the middle"
mechanism is not the operative one, and exp-018's read (sign right, magnitude
small) is confirmed by a second and sharper instrument than the rotation. I will
report this delta alongside the rotation number rather than in place of it: the
rotation is what decides whether the member ships, and this only explains it.

### The pre-registration, corrected to the pair that will actually be measured

exp-018 corrected the input before their arms landed. The 67 s of tail damage I
had in mind is their +/-3600 pair; the pair I will measure is +/-6000 against no
clip, whose closest read is the ab3 X-series at **dT ≈ 39 s** (tail 2839.3
clipped against 2800.7 unclipped). Put through the rule above:

    null (no coupling)    0.0033 * 39/26  =  0.005
    coupling indicated    2x that         =  0.010

against a set-to-set spread of +/-0.0002, so the quantity is resolvable with
about a factor of fifty in hand. Two honest caveats, both theirs and both
accepted: the rate is calibrated at 26 s and applied at 39, so the null is a
1.5x linear extrapolation rather than an exact figure; and **a delta landing
between 0.005 and 0.010 is "not resolved", not "no coupling"**. That band is the
most likely outcome and it is agreed before the measurement rather than argued
after it.

### The clip calibration I tried to supply, and why it failed

The standing limitation was that my rate comes from a round-budget change and a
clip is a different intervention. I hold a clip pair of my own — `D1w-all` against
`D1w-all-clip`, the 25-column day-state block with each `_z` clipped to its
training range — and it is a *better* instrument than the Huber pair in one
respect: clipping happens at predict time on the same fitted model, so the pair
carries no seed, thread or round-count noise at all.

It supplies nothing. On july_loo the clip moves tail RMSE by **+0.1 s** and
centrality by 0.0000; on january, −0.3 s and −0.0000. The rows the clip touches
are not tail rows, so a feature-range clip cannot calibrate a rate here at any
precision.

Two consequences, one of them a defect in my own tool. First, the 26 s Huber
budget remains the **only** calibration point, so exp-018's linearity caveat
stands unrelieved and their offer of a +/-3600 arm — which roughly doubles dT to
~67 s and would price the exchange rate at two widths instead of asserting one —
is the only way to remove it. Second, `tail_centrality.py` computed and printed
"+0.0050 of correlation per 26 s" from that july_loo pair: 0.0000 divided by
0.1 s, noise over noise, in the same confident format as a real rate. It now
refuses to quote a rate below a 5 s tail move and reports a failed calibration
instead. A near-zero denominator was the one input I had not considered when I
wrote a tool whose whole output is a ratio.

## The Huber corrected budget now has a serve side (2026-09-06)

The desk's instruction was to ship the 0.12 s Huber budget into the next blend
rather than as its own candidate. Three OOF folds existed; a blend cannot serve a
member it only holds out of fold, so blend-5 needed the twelve-month ranking fit
at the same budget. `run_hub8725_ranking.sh`, 2,628 s wall, one seed, six threads.

    rounds            9,518  (the 8,725 eleven-month count scaled by 12/11)
    threads           declared 6, observed 6, library lightgbm
    residual guard    [-20657, 9069], touched 0 rows
    rows              344,841, ids identical to the template and in its order
    values            float64, all finite, min 60.0, median 957.4, p99 2517.4
    below the floor   0
    sha256            dd0e8710000ab29b930ec72d8b058b00e6ed86ba3240cad01402c13813527882

Against the shipped 2,100-round serve file it correlates 0.9982, shifts the mean
by +0.66 s, moves 9.38 % of rows by more than 60 s, and has an RMS difference of
38.5 s — a four-fold change in round count, visible but not a different model.

**Durable address.** The producer writes into `artifacts/`, which `.gitignore`
line 20 excludes, so the file was born with no address in git. That is exactly how
this lane permanently lost member `48acb7e7`. It is published as
`published/L2-hub8725-ranking.parquet` and backed up to
`artifacts/_member_backup/`, all three copies verified identical by hash. The
shipped 2,100-round `L2-hub-ranking.parquet` is untouched (sha256 29adfc27…, still
dated 2026-09-05 23:42) and remains the fallback if the rotation turns against the
corrected budget.

## The Huber corrected budget survives a three-fold rotation (2026-09-06)

Earlier this file recorded the corrected budget as worth +0.12 s to the blend and
noted the limit honestly: "I only have the 8,725-round arm on `july_loo`, so
acting on it means two more". Those two folds finished overnight, so the reading
that was one fold is now three. `screen_swap.py members-4.json
members-4hub8725.json` — a one-member swap inside blend-4's five, weights fitted
only on the folds not being scored.

| mode | july_loo | december | january | board |
|---|---|---|---|---|
| nnls, 2,100 | 289.741 | 208.050 | 340.938 | 313.450 |
| nnls, 8,725 | **289.576** | **207.879** | **340.821** | **313.308** |
| delta | −0.165 | −0.172 | −0.117 | **−0.141** |
| flat, 2,100 | 292.438 | 211.704 | 341.464 | 315.094 |
| flat, 8,725 | **292.248** | **211.456** | **341.337** | **314.935** |
| delta | −0.190 | −0.248 | −0.127 | **−0.159** |

**Six of six.** Every fold in both modes moves the same way, so the desk's
instruction to ship this budget into the next blend is confirmed on three folds
rather than one, and blend-5 takes `L2-hub8725` as its Huber member.

**The weights are the sharpest evidence this lane has produced for its own
screening rule.** The 8,725-round member is **3.57 s worse** on its own score,
and NNLS gives it **more** weight on every fold:

    july_loo  0.1575 -> 0.1603
    december  0.1363 -> 0.1440
    january   0.1176 -> 0.1298

A fitted blend, free to allocate weight anywhere, chooses more of the member that
scores worse. That is not a near-tie the rotation resolved by luck; it is the
blend disagreeing with the standalone ranking in the direction the bulk/tail
mechanism predicts, with the allocation itself as the readout rather than a
downstream RMSE. Third demonstration of the inversion (after the Huber july_loo
reading and retro-v1's v3 arms), and the first where the evidence is the weight.

Note for the four-member flat rule: it does not bind here. At five members NNLS
beats flat by 1.64 s on the board (313.450 against 315.094), so blend-5's mode is
a fitted vector unless the rotation says otherwise at candidate time.

### Prediction B's mechanism is dead, and I killed it with my own measurement

exp-018 caught me doing the thing this file has criticised twice. I offered the
Huber swap as the sign asymmetry "showing up in a different experiment at a
resolvable size". It is not. Prediction B (line 1308) names its mechanism:

> NNLS reweights *toward* a better member, so the fit works in the improvement's
> favour instead of cancelling it.

That ties weight to member **quality**, and it is the entire reason an
improvement should retain more than a degradation. The Huber swap has NNLS
reweighting toward a member that got 3.57 s **worse** — weight following
tail-uniqueness *against* quality. It is evidence for the bulk/tail criterion
divergence (which has 0.889 and 0.764 behind it) and evidence **against** the
reweighting story Prediction B was built on. Both cannot be the explanation.

So: **the mechanism is withdrawn.** If weight tracks tail-uniqueness rather than
quality, there is no reason retention should be larger in the improving
direction — retention would track how much a change moves the member's
tail-uniqueness, in either direction, and the sign asymmetry loses its basis. The
bare outcome claim could still turn out true, but it is now unmotivated rather
than supported, and it will be scored that way.

**Scoring rule, adopted from exp-018 and applying to both lanes: a prediction is
scored against its stated mechanism, not against whichever outcome it happens to
survive.** Theirs is in the same position — Prediction A argued from lineage
diversity, and my correlation measurement says their member does not decorrelate
and wins on quality, so their mechanism is contradicted whatever retention
measures. Neither prediction gets to be rescued by an outcome its reasoning did
not produce.

The reason this is worth a section rather than a line: "the effect showed up
elsewhere" is exactly how a superseded mechanism stays alive, and I reached for
it within an hour of writing the criterion-divergence finding that supersedes it.
The bulk/tail result does not need Prediction B, and Prediction B cannot borrow
the bulk/tail result's evidence.

What survives all of this, and is worth more than either prediction: standalone
RMSE is 99.66 % a bulk statistic, blend value is about a quarter tail, and the two
can order candidates oppositely. That conclusion depends on no prediction and is
the durable result of the exchange.

### What the retention experiment still settles, after both mechanisms died

exp-018 marked Prediction A **MECHANISM REFUTED** in their PREDICTIONS.md before
the retention number exists, on the same rule I accepted for Prediction B: their
text argues from lineage diversity and predicts modestly above 7 % *because*
their correlations sit at the top of the existing range, and my measurement says
their member does not decorrelate and earns 0.622 on quality. The premise is
contradicted whatever retention measures.

They also made the point I should have made myself, so it is recorded as theirs:
**with both mechanisms dead, this measurement is no longer adjudicating between
two accounts of how blending works.** It was designed as a decision between
"weight follows quality" and "retention follows lineage diversity". Neither
survives. What is left is a bare quantity with a resolvable floor.

That is still worth having, and the write-up must claim exactly this much and no
more: a bound of the form "retention below X %, not further resolved" is a real
constraint on how much member-level tuning can reach a blend, and it is the
defensible version of what this desk already decided on the strength of the 7 %
figure when it cancelled a 31-to-126-hour tuning screen. It is not the experiment
either lane set up, and saying so is part of reporting it.

The record should also show that the experiment weakened **twice** before it ran:
once when both lanes converged on the coupling prediction, leaving the rotation a
weak test of something we already agreed on; and once when both mechanisms were
refuted. Neither was visible from inside one lane — each was found by the other
lane reading our reasoning rather than our numbers.

**What is untouched by all of it**, and what both lanes would keep if the
retention experiment vanished: standalone RMSE is 99.66 % a bulk statistic, blend
value is about a quarter tail, members correlate 0.889 in the bulk and 0.764 in
the tail, and the two criteria can order candidates oppositely. It explains
exp-018's clip result, retro-v1's opposite winsorise result and this lane's Huber
result as one phenomenon. Three lanes measured it independently and none
recognised it alone.

### The unconditional interval: exp-018's floor stands and my correction was wrong

The conditional bootstrap gave a board half-width of 0.056 s and I told exp-018
their 14 % floor was about twice too pessimistic. I flagged one assumption as
unmeasured: that weight noise "should" largely cancel in a paired delta, since
both configurations share fitting folds and four of five members. **It does not.**

Resampling the fitting folds and refitting NNLS inside all 500 replicates:

| interval | july_loo | december | january | board | half-width |
|---|---|---|---|---|---|
| fixed weights | [−0.248, −0.083] | [−0.240, −0.106] | [−0.196, −0.059] | [−0.202, −0.090] | 0.056 |
| refit weights | [−0.360, −0.060] | [−0.278, −0.092] | [−0.299, −0.032] | [−0.276, −0.071] | **0.1025** |

Weight refit noise nearly **doubles** the interval rather than cancelling. The
resolvable retention floor against a 0.74 s standalone gain is **13.8 %**, which
is exp-018's 14 % to within the width of their own rounding.

**So their floor stands and my correction is withdrawn.** Their reasoning was
wrong — they estimated instrument noise from the spread of screen outputs across
different inputs, and a paired one-member swap really is the tightest case — but
the number they reached is right, and it is right for a reason neither of us
named: the dominant term is not row noise on the scored fold, it is the
instability of the fitted weights themselves. In their words before the result
existed, right for the wrong reason. Mine was the reverse: correct instrument,
wrong conclusion, because I asserted a cancellation I had explicitly listed as
unmeasured and then wrote the headline as if I had measured it.

**Two things this does not change.** The Huber swap stays significant
unconditionally (board P(worse) 0.000, interval [−0.276, −0.071] excluding zero),
so blend-5 keeps `L2-hub8725`. And the 7 % figure remains inside the unresolvable
band at the current step size, so a null retention result must be written as
"below ~14 %, not further resolved", never as confirmation of 7 %.

**The methodological point is worth more than the arithmetic**: in a paired
comparison, holding the fitted weights fixed understates the uncertainty about
twofold. Every rotated delta this lane has quoted — including the −0.141 Huber
board delta — carries a conditional interval unless it says otherwise. The
practical rule is that a blend delta under about 0.1 s needs the refit interval
before it is called resolved.

## blend-5: V3-allin measured, and a fold-name trap in the member series (2026-09-06)

retro-v1's `V3-allin` landed with the december fold, so the blend-5 rotation ran.
It is the strongest member addition this lane has measured:

| mode | july_loo | december | january | board | weight on V3-allin |
|---|---|---|---|---|---|
| blend-4 (nnls) | 289.741 | 208.050 | 340.938 | 313.450 | — |
| +V3-allin (nnls) | **288.088** | **206.701** | **340.112** | **312.201** | 0.292 / 0.271 / 0.263 |
| delta | −1.652 | −1.349 | −0.826 | **−1.248** | |
| blend-4 (flat) | 292.438 | 211.704 | 341.464 | 315.094 | — |
| +V3-allin (flat) | **291.009** | **210.430** | **340.774** | **314.025** | 0.167 |
| delta | −1.429 | −1.274 | −0.690 | **−1.070** | |

−1.248 s is **twelve times** the 0.1025 s unconditional half-width measured above,
so this is resolved with room to spare and needs no refit interval to be believed.
A sixth member earning 0.26–0.29 also says the saturation finding was about
members drawn from *this* lineage: a genuinely different frame still buys 1.25 s.

**The fold-name trap, recorded because it would have been silent.** exp-001's
`oof/` now holds two overlapping sets. The december and january files of the
`V3-allin-member-*` series are **byte-identical** to the originals (sha256
52b9f7ed… and 003cf339…), but `V3-allin-member-july.parquet` carries fold
**`july`**, not `july_loo`:

    V3-allin-july_loo.parquet        fold july_loo   RMSE 301.23
    V3-allin-member-july.parquet     fold july       RMSE 316.46

Same 190,713 validation rows, identical ids, identical `y_true` — and different
models, 121 s RMS apart, because `july` trains on six months and `july_loo` on
eleven. Anything that templates `V3-allin-member-{fold}` over a rotation
containing `july_loo` gets a missing file (safe), but anything that maps "the
july one" onto the july_loo slot silently blends a member trained differently
from every other member on that fold, at a standalone cost of 15 s.

The members pinned here are therefore the config-consistent three:
`V3-allin-july_loo` and `V3-allin-january` from `result-V3-allin.json`, and
`V3-allin-dec-december` from `result-V3-allin-dec.json` — same argv but for
`--folds`, both at 660 fixed rounds — copied to
`published/V3-allin-v1-{fold}.parquet`. Confirmation requested from retro-v1.

**blend-5 cannot be issued yet.** There is no twelve-month ranking fit for
V3-allin anywhere in the repo, so the member can be rotated but not served — the
same gap the Huber budget had this morning, and the reason that one was built.

### The rollover drop, and what a two-fold rotation costs in resolution

retro-v1 confirmed the pinned three as V3-allin's published member set and
quarantined the `-member-` series; the fold-name trap above is closed at source
rather than worked around. They also diagnosed why no ranking fit existed, and it
is the same failure one layer down: `build_submission` was building a **50-column**
frame while the folds ran on 171, and it survived only because it crashed on
`KeyError: 'noac'` before writing. With the noac head off it would have written a
member with the right row count, the right ids and no complaint from any join.

`V3-roll171` is the clean common-frame pair: it differs from `V3-allin` in the
rollover drop and nothing else, both on the frozen 171-column set. Two folds only.

| | july_loo | january | board |
|---|---|---|---|
| V3-allin standalone | 301.23 | 343.95 | |
| V3-roll171 standalone | 299.49 | 343.52 | |
| blend with V3-allin | 288.548 | 340.100 | 312.431 |
| blend with V3-roll171 | **288.299** | **340.026** | **312.268** |

**The two criteria agree here**, which is worth recording as clearly as the cases
where they diverge: roll171 is better standalone *and* better in the blend. The
divergence this lane has been chasing is a property of particular changes — ones
that trade bulk for tail — not a general law that member score and blend value
disagree.

Unconditional interval, by the rule set earlier today:

    july_loo  [-0.557, +0.023]  P(worse) 0.030
    january   [-0.249, +0.044]  P(worse) 0.068
    board     [-0.362, -0.025]  P(worse) 0.014

So the rollover drop is **resolved at the board level and on neither fold alone**:
a real improvement of about 0.16 s, with the interval excluding zero only just.
Reported at that strength and no more.

**A two-fold rotation costs 65 % more uncertainty than a three-fold one.** The
half-width here is 0.1687 s against 0.1025 s for the same instrument on three
folds. `screen_by_blend.py` marks one-fold rotations with a star and says they are
upper bounds; it says nothing about two-fold ones, which are honest rotations but
measurably blunter. Every two-fold number in this file should be read against
0.17 s, not 0.10 s.

Related tool fix: `swap_bootstrap.py` printed a "resolvable retention floor" for
every swap, because those lines were written for the s1/s3 question. On the
rollover pair that reads as a claim about the rollover drop, which it is not. Now
behind `SWAP_RETENTION`; the general line — the smallest effect the rotation
separates from zero — always prints.

## Requoted in the desk's currency: clean + 4 s (2026-09-06, after the board read)

exp-016's stack-4 scored **265.7612**, rank 1 of 40, and the desk's calibration
makes the board predictor **clean board + 4 s**, not the all-rows quadrature this
file has been quoting. Every blend number above is in the wrong currency. Requoted:

| config | mode | july_loo | december | january | board (all) | board (clean) | **predict** |
|---|---|---|---|---|---|---|---|
| blend-4 | nnls | 289.74 | 208.05 | 223.17 | 313.45 | 262.35 | 266.35 |
| blend-4 | flat | 292.44 | 211.70 | 224.42 | 315.09 | 264.48 | 268.48 |
| blend-5 | nnls | **288.09** | **206.70** | **221.71** | **312.20** | **260.78** | **264.78** |
| blend-5 | flat | 291.01 | 210.43 | 223.23 | 314.02 | 263.15 | 267.15 |

Fold columns are clean RMSE. january moves 340.11 → 221.71 between all-rows and
clean, which is the two unreached-rollover rows and is why the currency matters.

**blend-5 predicts 264.78 against the board's current best of 265.7612** — about
1.0 s ahead, and ten times the 0.1025 s rotation noise floor the desk has adopted
as its upload gate. It cannot be served yet: retro-v1's twelve-month V3-allin
ranking fit is still running.

Caveat on the calibration: +4 s was derived from one candidate's outcome
(stack-4), so applying it to a differently-composed candidate assumes the January
shift costs the same. Both numbers are reported so the assumption stays visible.

**The bigger lever is not this blend.** What is on the board at rank 1 is exp-016's
stacker over this lane's member set, and V3-allin is worth −1.248 s as a member.
The stacker should be rebuilt on the six-member set before this blend is treated
as the candidate to beat.

### What V3-allin's arrival did to the other members

| member | july_loo | december | january | | july_loo | december | january |
|---|---|---|---|---|---|---|---|
| | *blend-4* | | | | *blend-5* | | |
| T-tailv2 | 0.1040 | 0.1372 | 0.1331 | → | 0.0475 | 0.0926 | 0.0930 |
| C2-cat | 0.0929 | 0.0533 | 0.0096 | → | 0.0643 | 0.0498 | 0.0058 |
| L2-hub | 0.1575 | 0.1363 | 0.1176 | → | 0.0569 | 0.0552 | 0.0441 |
| D1-winter | 0.0099 | 0.1325 | 0.1190 | → | **0.0000** | 0.0658 | 0.0481 |
| F1-fresh | 0.6377 | 0.5457 | 0.6254 | → | 0.5402 | 0.4687 | 0.5488 |
| V3-allin | — | — | — | | 0.2916 | 0.2713 | 0.2631 |

V3-allin takes roughly half of every v2-lineage member's weight and only about a
sixth of F1-fresh's — the fresh lineage is the member it displaces least, which is
the weight-side version of the lineage finding. D1-winter falls to exactly zero on
july_loo but holds 0.066 and 0.048 on december and january: it is a **winter**
member and july_loo is summer, so this is the member behaving as named rather
than dying. It is not dropped.

### Fit and serve now prepare members identically

`build_candidate` floors each member at 60 s and applies the residual guard
*before* blending, but the rotation fitted weights on raw member columns. Same
class as the exp-008 review's B1 finding. Measured before fixing: board **0.003 s**,
weights moving by at most 0.0005, 625 of 1,144,278 member-values touched on
july_loo — 30 times below the resolution floor, so it changes no conclusion in this
file. Fixed anyway, because the principle is that the served member is the
estimator its weight was fitted on, and the cost of holding to it here is nothing.

### retro-v1's measurement, which sharpens the inversion further

They built the same bootstrap for their standalone table and found half-widths of
**0.84 to 1.99 s** — five to twelve times blunter than my 0.1687 s, because a blend
difference is paired between highly correlated members and a standalone RMSE has
nothing to cancel against. Every ordering in their table under ~1.5 s is
unresolved, including a V3-allin/V3-rollover gap they had already withdrawn for a
different reason. On the one ordering that survives:

    V3-roll171 - V3-allin, common frozen frame
    standalone   -1.112  [-2.039, -0.363]
    in blend     -0.163  [-0.362, -0.025]

**The rollover drop is worth about eight times more to a member's own score than
to the blend** — same direction as the Huber arm, much larger ratio, and measured
cleanly on both sides with one difference between the arms. The sharpest example
either lane has of why a member must be chosen by what it adds.

## blend-5's member set, and the three swaps compose (2026-09-06)

Desk asked for two more swaps by the blend criterion. Both measured, and the
member set that results is blend-5. Clean board, the desk's currency.

| config | mode | july_loo | december | january | clean board | predict | vs blend-4 |
|---|---|---|---|---|---|---|---|
| blend-4 | nnls | 289.74 | 208.05 | 223.17 | 262.35 | 266.35 | — |
| +V3-allin | nnls | 288.09 | 206.70 | 221.71 | 260.78 | 264.78 | −1.566 |
| +V3-allin +tail v4 | nnls | 286.88 | 207.24 | 220.53 | 259.60 | 263.60 | −2.753 |
| **+V3-allin +tail v4 +hub8725** | **nnls** | **286.79** | **207.13** | **220.39** | **259.48** | **263.48** | **−2.867** |
| same, flat | flat | 289.95 | 210.26 | 221.92 | 262.01 | 266.01 | −2.471 |

**The three gains compose almost exactly**: measured individually they are
−1.57, −1.19 and −0.14, summing to −2.90 against a combined −2.867. Only 0.03 s
of interference between three changes to three different members — worth
recording because the opposite was the live risk, and because it means the swaps
did not have to be measured jointly to be trusted.

**blend-5 predicts 263.48 against the board's current best of 265.7612**, about
2.3 s ahead and 22 times the 0.1025 s upload gate.

### The tail member at the v4 noac head: −1.187 s

`exp-009-noac2-oof-v3.parquet` (sha `d7e028fa…`, the v4 noac mixture) in place of
the pinned v2. Clean board 260.78 → 259.60 in nnls, 263.15 → 262.17 flat. The tail
member's own weight rises **0.078 → 0.201**: at the v2 head the blend had been
backing away from this member, and at v4 it wants two and a half times as much.
december moves the wrong way (+0.54) while july_loo and january both improve;
december is not in the board mix, but the split is recorded rather than averaged
away.

Serving it needed a new ranking fit, and exp-008's `tail_ranking.py` carries an
explicit instruction against unpinning the head: *"To move to v2, refit the
weights on a v2 OOF and re-report; do not just unpin this."* That precondition is
now met — the weights are refit on the v4 OOF and re-reported above — so
`tail_ranking_v4.py` is a copy with the head repinned, leaving the v2 builder
intact so the shipped member stays rebuildable.

**Module drift checked rather than assumed**, because this is exactly the B2 shape.
OOF v3 was fitted under `tail_model` 2.2.0 and the new ranking fit runs under
2.5.0. 2.4.0 ends the band mixture's domain at 26 h: zero rows on every delivered
fold, one ranking row. 2.5.0 adds the schedule-placeholder regime, off by default
and left off. So the fold predictions the weights were fitted on are unchanged and
the pairing is matched.

### Still blocked on two fits, both running

blend-5 cannot be issued until both ranking files exist: retro-v1's twelve-month
V3-allin fit, and `tail_ranking_v4.py` here. The desk has since closed the stacker
lane and assigned it here, so when blend-5 is issued the stacker runs on the same
six members as **stack-5** and whichever wins goes to the head of the queue.

## blend-5 issued (2026-09-06)

`submissions/candidates/exp-012-blend-5.parquet`, sha256
`5d6a99610907bc432e2e46fb9ea11fc8103feb8aa9efdd3f7c86f1188866a4cb`,
344,841 rows in template order, all finite, min 60.0, median 962.0, p99 2507.3,
none below the floor. Config `blend-config-5.json`.

Rule: the same hybrid blend-4 shipped — NNLS inside `nm_late` over all seven
members, NNLS outside over the six general ones. Members: the six of
`members-5final.json` plus `N1-nmtail`, which enters only through the regime split.

| | july_loo | december | january | board (all) | board (clean) | **predict** |
|---|---|---|---|---|---|---|
| blend-4 hybrid | 289.32 | 207.77 | 340.82 | 313.175 | 262.010 | 266.01 |
| **blend-5 hybrid** | **286.33** | **207.01** | **339.02** | **310.768** | **259.132** | **263.13** |

**−2.88 s against blend-4 and 2.63 s under the board's current 265.7612.**

Shipped weights (fitted on all three months; the rotation scores the rule, it does
not choose the weights):

    nm_late    F1-fresh 0.4005  V3-allin 0.2962  N1-nmtail 0.2061  D1-winter 0.0819  C2-cat 0.0298
    elsewhere  F1-fresh 0.4784  T-tailv2 0.2265  V3-allin 0.2115  L2-hub 0.0839

`T-tailv2` takes 0.227 outside the regime and **zero** inside it, while `L2-hub`
is the reverse shape; `C2-cat` and `D1-winter` are live only inside `nm_late`.
The regime split is doing real work rather than reproducing one vector twice.

**A rename caught by a raised error rather than a wrong number.** V3-allin's
twelve-month fit is written in the *submission* format, so its value column is
`TAXITIME_SEC_mvt`, not `y_pred`. `prc.blend.stack_members` raised `KeyError`
instead of silently taking a wrong column — the same defensive shape that saved
retro-v1 from shipping a 50-column member. The published copy is renamed, values
verified identical, and the source left untouched.

**Member sets are now configs, not edits.** `NMTAIL_CORE` and `HYBRID_RANKING`
override `nmtail_rotate.CORE` and the hybrid builder's `RANKING`. Editing those
defaults in place would have quietly rebuilt blend-4 as something else the next
time anyone reran it; the blend-4 path is byte-reproducible as it stands.
