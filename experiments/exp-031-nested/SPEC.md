# exp-031-nested — SPEC

## My reading of the assignment, and the readings I rejected

I read `assignments/nested.md` as **"rebuild the five shipped rotations on an
instrument whose member out-of-fold columns cannot see the scored month's
labels, and say whether the differences we ship on survive"** — a measurement of
*differences between candidates*, not of levels. That reading fixes three
things. (1) Nothing is refitted: the nested instrument is a column swap on
`exp-027-step`'s own harness, so every arm is the arm that produced the shipped
number. (2) The chain is priced on each instrument against itself, never across
instruments, because §10.5 of `exp-025-cloud/RESULTS.md` shows the +4.28 board
anchor does not carry onto a nested fold number. (3) The gate runs first: with
the published files in place the harness must land on the five published fold
triples, or nothing below is reported.

**Two readings I rejected, and one control the contract forces.**

*Rejected: price the nested rotation against the published one directly.* The
`oof-nested/` README and §10.6 both say a shift taken against the published
workstation files is mostly the machine — ±0.3 s for `F1-fresh` and
`L2-hub8725`, with the sign changing by fold, three times L11's 0.1025 s
rotation floor. So a `published → nested` difference confounds the machine with
the nesting, and at the size of the differences this assignment weighs
(+0.2945 and +0.3997 s of board) the confound is the same size as the answer.
The instrument therefore has **three columns, not two**: `published` (the
record), `kaggle` (the same three members at their `oof-nested/` *baselines*,
same environment, no nesting) and `nested`. `published → kaggle` is the machine;
`kaggle → nested` is the nesting. This is not a new arm; it is the control the
`oof-nested/` contract names in its own third paragraph.

*Rejected: report `K2` against `v7` on the `nested` column alone.* `K2` replaces
`F1-fresh` with `K2-window`, which has no nested refit, so on the `nested`
column `v7` has three members nested and `K2` has two — and the difference
between them would carry one member's change of instrument. A fourth column,
`nested-f1held`, holds `F1-fresh` at its Kaggle *baseline* everywhere and nests
only `T-tailv4` and `L2-hub8725`, so `v7` and `K2` are nested identically and
their difference is theirs. `K2` and `refit-bulk` share a member set, so the
ordering question the assignment asks — `K2` minus `refit-bulk` — is unaffected
by this and is read on both columns.

**What the answer can and cannot be.** Three of seven members are nested at
most, and `V3-allin` cannot be refit at all (`exp-025-cloud` §10.3). The first
version of this paragraph called the nested rotation an **upper bound on the
clean score**; Sol round 1 finding 11 refuted that and is right — the design
confounds closing the channel with removing one month of training data, and the
measured member shifts include negatives (`L2-hub8725`, `july_loo_nx_dec`,
−2.55 s), so nothing here is monotone in the direction of the clean score. What
the numbers are: five candidates measured against each other on one instrument
at a time. Levels are quoted only to carry the differences.

## What Sol round 1 changed

The first version of this SPEC read the assignment's "clean instrument" as the
member channel alone, because that is what `oof-nested/` supplies. Sol round 1
(`docs/reviews/nested-round1.md`, finding 1, blocking) is right that the
rotation has a **second** label channel — the stacker's training anchor on a
fitting fold is the blend fitted on the other two folds, one of which is the
scored fold (`exp-026-rebuild/RESULTS.md` §4a, channel A) — and that a table
measured with it open cannot be called a clean-instrument result. The reading
taken now: **both channels are switches**, the whole table is measured with each
setting, and every claim is stated for both. `--anchor nested` closes channel A
with `exp-026/code/nested_anchor.py`'s construction, unchanged.

Three smaller readings Sol corrected and this SPEC now holds: the nested numbers
are **not** a bound on the clean score (the design confounds the channel with a
lost training month, and measured member shifts include negatives); "two members
nested" counts direct stacker features, and only `nested-f1held` removes the
third from the anchor as well; and the board must be computed from unrounded
fold RMSEs.

## What runs

`code/instrument.py` builds the four columns as a member-column swap on
`exp-026-rebuild/code/frames.py`'s own fold frame; `code/rotate.py` runs
`exp-028`'s stacker (`stack5w.arm`, five seeds, 300 rounds), `exp-026`'s rotated
hybrid-NNLS anchor, `exp-027`'s mixed anchor and v7's EHAM mask over the four
columns; `code/report.py` writes the tables. Gates, all on the `published`
column and all abort:

  A  blend-5 reproduces `blend-config-5.json` on all three folds (1e-3)
  B  stack-5 lands on 285.2029 / 205.6118 / 219.1303 (1e-3) and on exp-028's
     committed vectors (1e-9)
  C  v6 lands on 284.7219 / 205.6534 / 218.8166
  D  v7 lands on 284.6251 / 205.5677 / 218.5412
  E  K2 lands on 284.3195 / 205.2156 / 218.2582
  F  refit-bulk lands on 284.1774 / 204.9052 / 218.2115

Under `--anchor nested` the published triples are deliberately NOT gated, and
`rotate.py` logs why: closing channel A changes the rotation the shipped numbers
were measured on, so landing on them would be a defect.

Threads 4, CPU only. No refits, no uploads, no new members.
