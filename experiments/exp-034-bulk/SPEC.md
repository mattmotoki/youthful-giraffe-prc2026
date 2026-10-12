# exp-034-bulk — a member on exp-003's 64-column bulk frame, added to K2's stacker

Queue lever L7 (`docs/QUEUE-2026-09-07.md`). Feeds the v11 decision. Worker `bulk`, tier hard,
laptop, 6 threads, CPU only. Reviewer: Sol, one round on the shipping candidate.

## My reading of the assignment, and the readings I rejected

**Which exp-003 arm is "the 64-column set at 600 rounds".** exp-003 `RESULTS.md`'s ablation
table has one 64-column row: `T`, "base + occupancy + airport neighbours (**recommended**)",
and `RESULTS.md`'s own section "The recommended bulk set, column by column (for retro-v1)"
enumerates it as 13 categoricals + 27 own-flight numerics + 18 `sf.occupancy_features` +
6 `sf.airport_neighbour_features` = 64. exp-011 `code/q1c_seeds.py` builds exactly that tuple
(`FEATURES = CAT + BASE_NUM + OCC + NBR`) and runs it at **600 rounds** with 127 leaves and
lr 0.05, "halved from the 1200 of exp-003 `T_best`" because the workstation was loaded; the
27.8 s figure the queue quotes (`july_loo` 227.84, `december` 195.56) is that run's
single-seed number. So "the 64-column set at 600 rounds" is **exp-011's `q1c_seeds`
configuration**, and this experiment imports it from that file rather than restating it, so
there is exactly one address for the object the queue priced.

**The `stand` overlap (the queue's caution): I keep every column and measure the
consequence instead of trimming the frame.** The caution is that 8 of exp-003's 24
stand-occupancy columns are the same measurement at |r| = 1.000 under another name in the v3
frame (`dep_next_mvt_gap` = `lf_stand_next_dep_gap` and so on) and 6 more sit between 0.99
and 1.00, so the family's −4.9 s is partly a gain `V3-allin` already has. That is a statement
about **expected gain**, priced into L7's 0.5–1.5 s band; it is not a statement about which
columns to build. Dropping the overlapping eight would make the member a different, never-measured
object with no reproduction gate behind it, and the thing the stacker actually pays for is
residual diversity, which is measurable directly. So the frame is exp-003's, unchanged, and
`code/corr.py` measures the member's residual correlation with each of K2's seven — the same
instrument exp-032 used to explain K3's loss. If the candidate fails, that table is the
diagnosis rather than a guess.

**"Ninth slot" is a count I cannot reproduce, and I take the binding half of the
instruction.** K2's member set is seven columns (`T-tailv4`, `C2-cat`, `L2-hub8725`,
`D1-winter`, `K2-window`, `V3-allin`, `N1-nmtail`); adding one makes **eight**, and
exp-032's additive arm called the same position "an EIGHTH member". The instruction that
decides the work is "**added**, do not substitute", and that is what is built: K2's seven
plus `B1-bulk64`, inserted before `N1-nmtail` as exp-032's `_add` does. The count is recorded
here so the record and the assignment can be read against each other.

**The member is partial, and I do not build a second head to fill it in.** The model's
prediction is `taxi = max(nm_taxi − offset, 0)` with `nm_taxi = MVT_TIME_UTC_mvt − AOBT_3_flt`,
so on a row with no NM join there is no `nm_taxi` and no prediction — 2.06 % of `july_loo`,
0.94 % of `december`, 0.89 % of `january`. I serve it as a partial member (null there), which
is `N1-nmtail`'s existing contract in `frames.PARTIAL`, rather than fitting a second no-NM
head. Reason: a no-NM head is a new member, not this one, and the assignment says do not
widen. LightGBM reads the null natively and `stack5.member_features`'s four aggregates are
already nan-aware because `N1-nmtail` made them so.

**Readings I rejected.**
1. *Substitute the bulk member into `V3-allin`'s slot*, on the grounds that the two frames
   overlap. The assignment forbids substitution, and exp-032 measured what substitution costs
   when the displaced member is the diverse one: giving `T-tailv4` its slot back was worth
   +0.10 s of board and +3.27 s of july tail.
2. *Bag three seeds for the member*, which exp-011 measured at 0.77 s of member RMSE on
   `july_loo` for 3× the fit. It is not tuning, so it would be admissible, but it is not the
   object the queue priced and exp-032 measured the transmission from member RMSE to board at
   under 1 % on a member 2.0–3.5 s better. One seed (7), exp-011's own, is the member; the
   three-seed bag is a backlog refinement if the candidate lands borderline.
3. *Tune before the gate is read.* Explicitly excluded by the assignment, and by exp-003's
   own frozen-configuration rule.

## The gate on the frame, and the tolerance I accept

The frame builder is exp-003's `code/features.py` and `prc/stand_features.py` 1.2.0,
unchanged and in git, so the strongest available gate is not a logged fold RMSE but a
**committed per-row vector**: exp-011's `oof/seeds_july_loo.parquet` carries `p_s7`, `p_s17`,
`p_s42` — three seeds of exactly this configuration. `code/member.py gate` refits seed 7 on
`july_loo` from the rebuilt frame and reads it against the committed `p_s7`, whose own bulk
RMSE is recomputed from the file rather than quoted.

**Tolerance: 0.05 s of bulk RMSE, not row equality.** Why: (a) exp-003 `RESULTS.md` measures
this harness's own run-to-run spread at about 0.05 s (`A_base` 207.803 at 16 threads against
207.803 at 12; `F_all` 201.886 against 201.935) and states that "no difference below ~0.2 s
is real"; (b) the committed vector was fitted on the Linux workstation with another LightGBM
build and another thread count, and exp-026 already records that 4.7.0 against 4.6 is not
expected to reproduce bit for bit; (c) exp-011's `PARAMS` set neither `deterministic` nor
`force_row_wise`, so thread count alone moves the fit. A row-exact gate would fail for
reasons that have nothing to do with the frame, which is what the gate is asking about.

Two exact checks run beside it, and they are the ones that bind on the frame itself:
`features.py check` (every feature bit-identical built from the raw month and from the
ranking-style masked month — **passes**), and a column contract in `analysis/frame-contract.json`
(the 64 names, in order, with their SHA-256, and per-column null shares) so a later drift is
visible rather than silent — this lane has had one silent frame drift already.

## Pre-registration of the shipping arm (written before any rotation number existed)

Two additive arms are rotated in one process, because a second process would cost a second
frame build and a second set of gates for one 40-second fit:

- **`bulk-add-hold`** — the member as built, defined wherever the NM join is. **This is the
  shipping arm.**
- **`bulk-add-bulkonly-hold`** — the assignment's named repair: the same member column, null
  outside the `nm_bulk` regime rows.

The repair arm ships **only if** `bulk-add-hold` fails the assignment gate (+0.10 s of clean
board over K2's 257.1653, no fold worse than −0.10 s) **and** the repair arm passes it. If
both pass, the shipping arm is still `bulk-add-hold`; if both fail, nothing ships and the
lane is reported as a negative with its diagnosis. This is written down now so that "which
arm ships" is not a choice made after reading the fold that scores it — the failure exp-032's
Sol round 2 recorded against the K3 member gate.

## What is built

| | |
|---|---|
| member | `oof/bulk64-{july_loo,december,january}.parquet`, `MVT_ID_mvt, fold, y_true, y_pred, off_pred` |
| ranking column | `candidates/bulk64-ranking.parquet`, twelve-month fit, 344,841 rows |
| candidate | `code/candidate.py`: K2's stacker with `B1-bulk64` added as an eighth member, window block `depx` at 600 s, EHAM served from stack-5 as v7 does, anchor blend-5 on the committed seven, `noac` held at blend-5's anchor |
| gates | A blend-5 reproduces `blend-config-5.json`; B stack-5 reproduces exp-026; C v6 reproduces exp-028; D `k2-hold-eham-out` rebuilds K2 (257.1653); all at 1e-3 on the RMSE and 1e-9 on the vector |
| assignment gate | clean board gain ≥ **+0.10 s** over K2, no fold worse than **−0.10 s**, day-clustered 95 % intervals stated |
| repair if it fails | one diagnosis (`code/corr.py`, the residual-correlation table) and one repair: the member restricted to the `nm_bulk` regime rows in the stacker |
| serve | `serve.py`, exp-027's serving path with the eighth member; 388 noac and 38,182 EHAM rows bit-identical to v7 at max abs difference 0.0 |

## Compute and rules

6 threads, CPU only, launch held while the 1-minute load is above 25. No uploads. Commit
locally after each measured step; the desk pushes. Ledger rows for the member and the
candidate in `docs/LEDGER.md`.
