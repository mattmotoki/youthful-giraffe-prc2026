# Predictions registered before the s1-vs-s3 retention measurement

Written 2026-09-06, before `exp-018b-s3` exists. The blend lane
(`prc-data-challenge-2026-diversity`, exp-012/exp-015) holds the same two
predictions in its own RESULTS. The point of writing them down is that they
disagree, so the measurement can settle something.

## What is being measured

Seed-bagging **retention** through a blend: how much of the standalone gain a
member gets from 1 → 3 seeds survives into the blended rotation. The repo's
current figure is **about 7 %**, measured once, on the Huber member, at weight
0.26. It is load-bearing: the desk cancelled a 31-to-126-hour member-tuning
screen on the strength of it and two other lanes cite it.

The pair `exp-018b-s1` / `exp-018b-s3` is the same lineage, same features, same
fold code, same round budget, both arms matched OOF-to-candidate, differing
only in seed count, at weight ~0.596.

## Prediction A (mine)

Retention is driven by **lineage diversity**, not by weight: a member's gain
survives blending to the extent its errors are not already cancelled by the
other members. My residual correlation against the four is 0.870–0.900, which
sits at the **top** of the 0.863–0.909 they already share with each other — not
outside it. So I predict retention **modestly above 7 %**, not dramatically
above. A result near 30 % falsifies this and the weight story together.

## Prediction B (the blend lane's, recorded as theirs)

The 0 % they measured for rotated NNLS came from a member **degradation**
(CatBoost 5,325 → 600 rounds), and NNLS answered by reweighting *away* from the
worsened member, which is why nothing reached the blend. An improvement is not
the mirror image: NNLS reweights *toward* a better member, so the fit works in
the improvement's favour. They therefore predict retention for an improvement
**exceeds** retention for a degradation, and that the 0 / 7 / 12.9 / 17 % series
is a lower bound when read in the improving direction.

## The confound I want on the record before the number arrives

Seed bagging does two things at once and this design does not separate them: it
**improves** the member and it **reduces the member's variance**. A blend is
itself an averaging operation, so the variance half is partly redundant with
blending by construction, while the quality half is not. Prediction B's
mechanism (NNLS reweights toward a better member) speaks to the quality half;
my mechanism (errors already cancelled by correlated members) speaks to the
variance half. A single retention number will not tell the two apart, and
whichever way it lands, neither of us should claim our mechanism was confirmed
without a second design that moves quality without moving variance.

## Falsification

* Retention ≈ 7 % → prediction B's asymmetry does not exist; the series is
  symmetric after all. The blend lane has said it would rather learn this than
  keep quoting the number in a direction it was never measured in.
* Retention ≫ 7 % (say 30 %) → both mechanisms above are wrong, or the
  variance half dominates and the diversity argument is beside the point.
* Retention ≈ 0 % → the improvement does not reach the blend at all, and the
  three-seed work is worth doing only for the fit-serve matching.

---

# Prediction registered before the clip-in-blend rotation

Written 2026-09-06 06:05, before `exp-018b-s1m-noclip` exists and before the
blend lane runs the rotation. Their prediction is theirs and is quoted, not
paraphrased into agreement.

## What is being measured

Whether the training-residual winsorise at ±6000, which is +1.48 s on this
lineage's own july total, is worth anything to a rotation that already holds
this member. Two arms differing only in `clip_z`, both one seed, both through
`model2`, four folds, matched OOF: `s1m` (±6000) against `s1m-noclip` (none).

## Their prediction (the blend lane's, recorded as theirs)

Members agree in the bulk and disagree in the tail — on blend-4's five members,
july_loo, mean pairwise residual correlation is **0.889** over the 190,057
bulk rows and **0.764** over the 656 tail rows, and the tail carries 23–27 %
of every member's squared error. So a blend has almost nothing to gain where
members correlate 0.889 and almost everything to gain where they correlate
0.764, while a member's own RMSE is 99.66 % a bulk statistic. Their conclusion:
**the clip buys bulk and sells tail, which is the wrong direction for a blend
member**, and they expect it to screen at or below zero despite being positive
standalone.

## My prediction

I agree with the direction and disagree about the size. I expect the clip to
screen **negative but small — roughly −0.5 to 0.0 s — not decisively below
zero**, and I would be surprised by anything past −1.5 s.

The reasoning, so it can fail for a stated reason. Their mechanism is about
*where* the blend gains, and it is right; but the clip's whole effect on my own
fold total is 1.48 s, of which the tail half is +67 s of RMSE on 0.34 % of rows
— 2,661 of 190,759 MSE units. For the blend to lose more than that from my
member, clipping would have to make my tail predictions not merely worse but
*more like everyone else's*, and that is the step I am not confident in.
Clipping the training residual pulls my tail predictions toward the centre,
which is a move away from the other members as often as toward them; a member
that is worse **and differently wrong** can still earn its weight. Their
mechanism predicts the sign; it does not by itself predict a large magnitude.

## Falsification

* Clip screens at or below **−1.5 s** → my "small" is wrong, their mechanism is
  stronger than the standalone arithmetic suggests, and the right reading is
  that centrality, not accuracy, is what the rotation is pricing.
* Clip screens **positive** → their mechanism is wrong here, or the bulk gain
  reaches the blend after all, and my ±6000 width needs no revision.
* Clip screens between **−1.5 and 0** → both of us are right about the sign and
  I am right about the size; the actionable consequence is theirs either way:
  the width tuned for my own score and the width tuned for the blend are
  different numbers, and mine is tuned for the first.

## What I will do with it either way

Nothing automatically. A width tuned for the blend is the desk's call, not
mine: this lineage is also a standalone candidate in `docs/SUBMISSIONS.md`, and
those two consumers want different numbers. I will report both and say so.

### Convergence before the measurement (2026-09-06 06:10)

Recorded here rather than after the number, because it makes the test weaker
and that is easier to say now than later.

The blend lane tested the step my objection turned on — does a worse tail mean
a more central tail? — on a pair that isolates it: their Huber arm at 2,100 and
8,725 rounds, identical but for the budget. On july_loo, tail rows only, mean
correlation with the other four members: HUB2100 (tail RMSE 2588.1) 0.777,
HUB8725 (tail RMSE 2561.5) 0.773. **A 26 s difference in tail accuracy moves
centrality by 0.004.** Their own limitation, which they state: a round-budget
change is not a clip, and a clip explicitly compresses the learned response, so
the pair shows the two are not automatically coupled and not that they are
uncoupled under this intervention.

They have withdrawn the expectation of a large effect and now predict a small
blend-level loss, inside my −0.5 to 0.0 rather than outside it. So we agree on
sign and roughly on magnitude, and the rotation number is now a **weak test**:
the only informative outcomes left are positive (kills their mechanism) or past
−1.5 s (kills my "small").

**The sharper test the arm also supports.** The rotation measures the
consequence; the coupling itself is measurable directly, and my arm is the pair
that measures it under a clip rather than under a round budget. The quantity is
the change in this member's mean tail-row residual correlation with the other
members between `s1m` (±6000) and `s1m-noclip`. Their Huber pair puts a scale
on it: 0.004 per 26 s of tail RMSE. My clip moves the tail by considerably more
than 26 s, so if centrality and accuracy are coupled under a clip the delta
should be visible against that scale, and if the delta is of order 0.004 or
less the coupling is absent under this intervention too. That result does not
depend on how the rotation lands, which is why it is worth naming before the
rotation runs.

### The blend lane's pre-registered threshold, and the resolution my arm will give it (2026-09-06 06:12)

Theirs, recorded as theirs. Instrument `experiments/exp-012-blend2/tail_centrality.py`
at commit 4216d22, calibrated before my arms exist. Their correction to the
yardstick they had quoted me: the arm under test was itself one of the four
comparison members, contributing a self-correlation of 1.0, so the clean scale
on july_loo is **0.0033 of correlation per 26 s of tail RMSE** at a level of
0.792/0.789, not the 0.777/0.773 first quoted. The delta survives; the level
does not. Across four comparison sets the delta moves only in the fourth
decimal (−0.0033, −0.0034, −0.0031, −0.0033), so the yardstick is
set-independent even though the level is not.

Their null and threshold: if the clip moves tail RMSE by `dT` seconds,
no-coupling predicts a centrality change of `0.0033 * dT / 26`; they call
coupling only above **twice** that.

**What resolution that leaves, from my side, before the arms run.** The pair
being measured is ±6000 against no clip, and the closest read I have on its
`dT` is the ab3 X-series: tail 2839.3 at ±6000 against 2800.7 unclipped, so
`dT ≈ 39 s`, not the ≈67 s of the ±3600 A/E pair. That puts the null at about
**+0.005** and the coupling threshold at about **+0.010** — a quantity only
three times their measured set-to-set spread of ±0.0002, so the instrument can
resolve it, but not with much room. Two consequences worth stating first: the
test is tighter than the 26 s calibration suggests, and their rate is
calibrated at 26 s while my `dT` is about 39 s, so the null is a linear
extrapolation of about 1.5×, which is modest but not nothing. If the observed
delta lands between the null and the threshold, the honest answer is "not
resolved", not "no coupling".

---

# Both predictions scored against their mechanisms, before the retention number exists (2026-09-06 07:25)

The rule, agreed by both lanes: **a prediction is scored against its stated
mechanism, not against whichever outcome it survives.** Applied here to mine
first, because I proposed it.

## Prediction A (mine): MECHANISM REFUTED

The registered text argues from lineage diversity — "retention is driven by
lineage diversity, not by weight: a member's gain survives blending to the
extent its errors are not already cancelled by the other members", and predicts
only modestly above 7 % *because* my correlations sit at the top of the
existing range rather than outside it.

The blend lane's independent measurement says my member does not decorrelate at
all: 0.870–0.900 against the 0.863–0.909 the other members already share, and
it still earns weight 0.622. The weight is buying quality, not independence. So
the premise that made my prediction a diversity argument is contradicted, and
it is contradicted **whatever retention measures**. If the number comes back
near 7 % I do not get to claim the prediction; the reasoning that produced it
was wrong about why.

What survives from it: nothing mechanistic. The bare number may land in range
and that is luck, not support.

## Prediction B (the blend lane's): MECHANISM WITHDRAWN BY THEM

Their registered text names it — "NNLS reweights toward a better member, so the
fit works in the improvement's favour instead of cancelling it" — which ties
weight to member quality and is the whole basis for expecting an improvement to
retain more than a degradation. Their own Huber swap has NNLS reweighting
toward a member that got **3.57 s worse** standalone, on all three folds: weight
following tail-uniqueness *against* quality. They withdrew the mechanism
(commit a8d9d9f) and drew the further consequence themselves — if weight tracks
tail-uniqueness rather than quality, retention should track how much a change
moves tail-uniqueness in *either* direction, and the sign asymmetry has no
basis. The bare outcome claim could still come true, unmotivated.

## What the retention number can now settle

Less than either of us intended, and it is better to say so before it arrives.
Both mechanisms are dead, so the measurement is no longer adjudicating between
two stories — it is a bare quantity with a resolvable floor near 7.6 %
(pending the unconditional interval) on a 0.74 s standalone gain. It is still
worth having as a bound. It is no longer an experiment that distinguishes two
accounts of how blending works.

## What survives the whole thread, depending on no prediction

Standalone RMSE is **99.66 % a bulk statistic**; a blend's marginal value is
about **a quarter tail**; members correlate **0.889 in the bulk and 0.764 in
the tail**; and the two criteria can therefore order candidates oppositely.
That is measured, it explains this lineage's clip result, retro-v1's opposite
winsorise result and the blend lane's Huber result as one phenomenon, and it
would be the result to keep if the retention experiment went away entirely.
