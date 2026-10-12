# exp-032-k3 — the transient window inside a second member's frame

## My reading of the assignment, and the readings I rejected

**The member frame I read `T-tailv4` as.** `T-tailv4` is not one model on one
frame. It is `prc.tail_model.TailModel` with `noac_head="v4"`: a **regime
router** over three sub-models, each with its own target and its own feature
list, built on `prc.tail_features.tail_signals` (VERSION 1.3.0, 53 signals, one
row per departure at the ten ranking airports) plus `prc.phi_features.PhiModel`'s
nine columns on the NM head.

| head | rows | target | feature list |
|---|---|---|---|
| `nm` | NM join present | `mvt_aobt − y`, added back | `FEATURES` (41) + `PHI_COLUMNS` (9) |
| `nonm` | every non-`noac` departure | the taxi time itself | `FEATURES_NONM` (27: `FEATURES` less the fourteen own-flight columns) |
| `noac` | `AIRCRAFT_TYPE_mvt` null | `mvt_sched − y`, five seeds, plus the fitted two-point band mixture above 8 h | `NOAC_FEATURES_V2` (32) |

**Where the window block enters, and where it does not.** Into the `nm` and
`nonm` heads' feature lists, and into neither the `noac` head nor the band
mixture. That is K2's third design rule — *offload `noac` from the fit*
(`exp-027-step/SPEC.md`, addendum 16:05; the shape family put 361 % of its gain
on the `noac` rows and took the clean board the wrong way) — and in this member
it costs nothing to obey, because the router already gives `noac` a separate
head. In `F1-fresh` the offload had to be arranged; here it is the architecture.
The band mixture is a counting estimator over ~50 rows and has no feature list
at all, so it is untouched by construction.

**Three readings I rejected.**

1. **That the block belongs in the `noac` head too**, on the grounds that
   `T-tailv4`'s whole reason for existing is the `noac` regime and 20 % of the
   board's mass sits there. Rejected on a measurement that already exists, not
   on taste: exp-027 §11 measured the identical family with and without the
   `noac` rows in the fit and the offload is worth **+0.366 s of clean board for
   nothing**. And score-feedback rule holds the 388 served `noac` rows at stack-5's bytes in
   every candidate on this base, so a `noac` move could not reach the board even
   if it were good.
2. **That "T-tailv4's own feature frame" means `prc.tail_features.SIGNALS`, so
   the block should be added there.** Rejected: `SIGNALS` is consumed by
   `_EXP002_SIGNALS`, which the module freezes precisely so that a new observable
   cannot silently move the delivered heads, and it is also read by
   `exp-002`/`exp-009`/`exp-022` harnesses that are not this arm's to move. The
   block enters as a **caller-supplied frame** joined onto the signals at fit and
   predict time, so `tail_features.py` is untouched and the flag-off path is the
   published path byte for byte.
3. **That the block should be rebuilt for the tail frame** (window statistics of
   `BLOCK − AOBT_3` rather than of `MVT − AOBT_3`, to match the NM head's own
   target). Rejected without measurement: the assignment says *the same block K2
   used*, and rebuilding it would confound "a second member frame" with "a second
   block". `BLOCK` of another departure is also not a legal input
   (`docs/input-rule.md`), which is why K2's block is built on `MVT − AOBT_3` in
   the first place.

**What "refit on the three rotation folds" means here.** `july_loo`, `december`,
`january` — `frames.FOLDS` — each a `TailModel` fit on its eleven training months
and predicted on the masked held-out month, exactly as
`exp-009-noac2/deliver.py` produced the committed OOF, with `noac_head="v4"` and
`n_jobs=6`.

**One thing I did not widen.** The assignment's arm list is two arms and the
second is gated on the first. Nothing else is measured here.

## Why this arm can run on the laptop, and the baseline that makes it like-for-like

`T-tailv4` is **CPU-only LightGBM** (`docs/CLOUD-COMPUTE.md` §6: the tail
producer is `prc.tail_model` + `prc.tail_features`, no GPU) and it is
**bit-identical across machines** — `exp-025-cloud/RESULTS.md` §10.7 ran its
unnested folds on Kaggle and got the committed workstation OOF exactly, 510,096
rows, maximum absolute difference 0. So, unlike K2:

- no GPU slot is needed and no GPU hour is spent;
- the **committed OOF is the like-for-like baseline**. K2 had to buy a Kaggle
  re-baseline for 1.5 GPU-h because CatBoost-on-GPU moved `F1-fresh`'s july_loo
  by +0.275 s (§9.3 there). That machine term is measured to live *only* in
  F1-fresh's path. Here the treatment is the only difference.

Published `T-tailv4` folds (`exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet`,
sha `d7e028fa…`): **july_loo 301.6430, december 222.7363, january 345.7125**.

The claim is checked rather than assumed: the flag-off path is re-run on
`july_loo` and must land on 301.6430 with the OOF vector bit-identical to the
committed one. That is this experiment's environment control and it costs about
five minutes of CPU.

## The change to `prc/tail_model.py`

One optional argument on `fit` and `predict`, defaulting to `None`, following the
file's own precedent for an off-by-default arm (`placeholder=False`, VERSION
2.5.0, exp-022):

    fit(train_frames, ..., window: pd.DataFrame | None = None)
    predict(month_frame, ..., window: pd.DataFrame | None = None)

`window` is a frame indexed like the input frame's departure rows. It is joined
onto the signals and its columns are appended to the `nm` and `nonm` feature
lists only. The fitted model records the column names, and `predict` raises if
the frame it is given does not carry them, so a model fitted with the block can
never be served without it. With `window=None` no branch is taken and the module
is the published module.

`prc/tail_features.py` and `experiments/exp-018-fresh/code/window_features.py`
are **not** edited. The block is K2's, imported.

## Gates, written before the numbers

**Environment control.** Flag-off `july_loo` reproduces 301.6430 and the
committed OOF vector at a maximum absolute difference of 0. An environment
fault, not a result; stop and report if it fails.

**Member gate (the assignment's).** `july_loo` improves by **≥ +0.30 s**
like-for-like with only the window block changed, and `rmse` on `y > 3600` is
not worse by more than **1 s**. Fail → diagnose in one paragraph, one repair,
report; a negative is a result.

**Candidate gate (the assignment's).** K2's stacker (the exp-027 harness) with
the K3 member in `T-tailv4`'s slot and the K2 member in `F1-fresh`'s slot,
window block and EHAM mask served as v7. Clean gain **≥ +0.10 s over K2
(257.1653)** with **no fold worse than −0.10 s**, day-clustered intervals stated.

**Identity gates, before any K3 number** (exp-027 §16a's discipline, extended to
two substituted slots): with `PRC_K3` pointed back at `T-tailv4`'s own committed
OOF and `PRC_K2` at `F1-fresh`'s, the harness must reproduce blend-5, stack-5,
v6 and v7; with only `PRC_K3` identity-pointed it must reproduce the K2
candidate (257.1653) exactly.

**Serving.** `submissions/candidates/exp-032-k3.parquet`: template order, finite,
none below the floor, the 388 `noac` and 38,182 EHAM rows bit-identical to v7,
rebuilt from its own record at a maximum absolute difference of 0.0.

## What this is expected to be worth, stated before it runs

`T-tailv4` carries **0.2265** of blend-5's anchor outside `nm_late` and **0.0000**
inside it, against `F1-fresh`'s 0.4784 — a little under half F1-fresh's weight —
and it is also one of seven member columns the stacker sees. K2 delivered
**+0.526 s** of member fold gain into **+0.157 s** of head gain with no window
block in the stacker and **+0.077 s** with v6's block already there (§16b), a
delivery ratio of about 0.15–0.30.

So if the tail member takes a similar member-level gain, the head-level gain on
top of K2 is of the order of **+0.05 to +0.15 s of clean board**, and the +0.10 s
candidate gate is at the top of that range rather than the middle. Two things
could push it up: the mechanism enters a *third* time here (member, stacker
block, and now a second member), and the tail member is the one whose rows the
window has never described. Two could push it down: the block's information is
already in the stacker's own 50-column block and in K2, so the third entry buys
the residual of a residual; and the `nm` head's target is `BLOCK − AOBT_3`, which
is the quantity the window block's `MVT − AOBT_3` is *derived from* on the same
row, so the block may be closer to redundant here than it was in F1-fresh.

Writing the low end down first is the point: a +0.05 s result is the expected
value of this arm, not a surprise, and the candidate gate at +0.10 s is a real
bar that this arm can fail while the member gate passes.

## The decision it feeds

What goes up as v11. A PASS says the window mechanism is still not exhausted at
its third entry and the next block is a fourth. A FAIL at the candidate with a
PASS at the member says the mechanism is real, the member carries it, and the
**head is saturated** — which routes the next block away from the window family
and is a stronger statement than a member-level negative would have been.

---

## Addendum, 2026-09-07 22:45 UTC — the member gate splits, and the repair, registered before the rotation

`july_loo` is measured (`RESULTS.md` §3): the member arm improves the fold by
**+2.048 s** against the assignment's +0.30 bar, and moves the 656 rows above
3600 s **7.33 s the wrong way** against its 1 s tolerance. So the member gate as
written is **split: the improvement clause passes at 6.8× the bar and the tail
clause fails.** The assignment's failure branch applies — diagnose, one repair,
report — and this addendum is the repair, written before any candidate rotation
has run.

**The diagnosis** (`analysis/member-decompose.csv`, `RESULTS.md` §4). The whole
gain is the bulk (+2.250 s of the +2.048 comes from `y ≤ 3600`) and the damage is
503 tail rows inside the `nm` head: `nm_bulk` −25.2 s on 119 rows and `nm_late`
−14.3 s on 384. The `noac` head is untouched at **exactly 0.000**, which is the
offload holding by construction. That is the same shape `exp-027` §13 measured
on the K1 slices — "the arm takes 1.0–1.8 s off `y ≤ 3600` on every fold while
moving the rows above 3600 s the wrong way" — so this is the mechanism's known
signature in a new frame, not a new defect.

**The repair, and why this one.** `exp-027/SPEC.md` registered it before this
experiment existed: *"Gate the arm to `y ≤ 3600`. The tail lane owns that
population."* `y` is a label, so the servable form gates on the **base member's
own prediction**, which is a function of inputs alone and is committed for every
ranking row:

    K3g  =  the window arm where T-tailv4's own prediction ≤ 3600 s,
            T-tailv4 itself above it.

The threshold is the record's. `analysis/gate-sweep.csv` reports 1800 and 7200
beside it; **neither is taken**, because a threshold chosen after seeing three
fold numbers is a second cut on the same data in exactly the sense `exp-030` §4
and `exp-027` §16e record of EHAM and of `refit-bulk`.

Two repairs I rejected, both measured on july_loo before choosing
(`RESULTS.md` §4): serving the base member on the whole `nonm` regime (fold
+2.258 but tail **−10.71**, so it repairs the wrong clause — the `nonm` rows are
where the window arm's tail is *better*), and serving it on `nm_late` (fold
+2.038, tail −1.20, still outside the 1 s tolerance, and it gives up the gain on
14,125 bulk rows to buy 384 tail ones).

**What ships, decided here rather than after the rotation.** The candidate is
**`k3g-hold-eham-out`** — the gated member in `T-tailv4`'s slot, K2 in
F1-fresh's, anchor held at blend-5, EHAM served as v7. The ungated
`k3-hold-eham-out` is rotated beside it and recorded, and it is **not** servable
whatever it reads, because its member fails a clause the assignment wrote. The
candidate gate is unchanged: ≥ +0.10 s of clean board over K2's 257.1653 with no
fold worse than −0.10 s.

## Addendum, 2026-09-07 23:20 UTC: the additive arm, added by the desk

Written **after** the candidate rotation of §7b and before the additive one, so
the record says plainly which arm was pre-registered and which was not.

The assignment says *do not widen the arm list*, and the arm list above was not
widened by me. After I reported the candidate FAIL, the desk asked for one
repair and named the two readings it wanted separated: whether the block's
information already reaches the stacker through K2 (redundancy), or whether the
loss is the **substitution** taking `T-tailv4`'s slot away from the tail rows.
The desk's instruction, verbatim in substance: *keep T-tailv4 in its slot and
add the K3 member as an eighth member (additive, not substituted), same
harness, same gates; if that also fails, write both readings in the ledger row
and report done; do not start K3-hub.*

So arm 2 of the assignment (**K3-hub**, the block inside L2-hub8725's frame) is
**not run**, on the desk's instruction, even though arm 1 passed its member
gate and had therefore earned it. That is the desk's call to make and it is
recorded here as the desk's, not as a finding of mine.

The additive arms are `k3-add-hold` and `k3g-add-hold`: K2's seven members plus
the K3 member as an eighth, inserted **before** `N1-nmtail`, which must stay
last. Everything else is the arm above — same anchor, same five seeds, same 300
rounds, same `depx` block, same EHAM mask, same gates A to D. **Gate E is
skipped for these two arms and says so in the log**: under `PRC_IDENTITY` an
additive arm carries `T-tailv4` twice, which legitimately moves `mem_mean`,
`mem_std`, `mem_range` and `n_defined`, so v7 is not the right target for it.
Gate D still rebuilds the K2 baseline in the same process.

One diagnostic was added with them, `code/corr.py`, which fits nothing: it reads
the OOF already on disk and reports each member's residual correlation with K2
and with the field. It is the screen §7e proposes for the next member.
