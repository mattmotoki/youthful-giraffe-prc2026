# exp-012 blend2 — spec

Worker `diversity`, tier hard, exp-id `exp-012`, desk assignment 2026-09-05:
own the standing blend. Supersedes exp-008's blend as the live one.

## My reading of the assignment, and the readings I rejected

The desk asked for the member set rebuilt at corrected budgets with three-seed
bags, rotated NNLS with a month-to-month weight-stability check, a numbered
candidate per improvement with its board estimate, and a table of every
candidate in the repo so uploads can be ordered. I read "own the standing blend"
as: the blend is now a standing artifact that improves as members land, not a
one-shot experiment, so the machinery has to make adding a member cheap and the
record has to say which candidate is currently best.

**I moved from four folds to three.** exp-008 scored `july`, `july_loo`,
`december`, `january`. This experiment drops the forward `july` fold. Three
reasons, in order of weight: retro-v1 has already moved its ladder to
`july_loo`/`december`/`january` and its recent arms have no `july` OOF at all,
so keeping it would exclude the member the desk named first; `july` is not part
of the board estimate (`RANKING_ANALOGUE_FOLDS` is `july_loo` + `january`); and
dropping it removes the same-month trap exp-008 had to special-case, because
each remaining fold is a distinct validation month, so the rotation is simply
"fit on the other two". The cost is losing the pessimistic no-same-season read,
which I state rather than hide: **no number in this experiment is a forward-fold
number, and a mechanism that only helps when same-season data exists will look
better here than in exp-008.**

**I rejected reusing exp-008's `T-tail` member.** exp-008 pinned
`prc.tail_model` to noac head v1 because its weights had been fitted on a v1
OOF. Here the weights are fitted fresh, so the pin is unnecessary and costly:
exp-009's v2 OOF scores july_loo 306.8 against v1's 310.4 and is the best single
member available. exp-012 uses **v2** and serves v2, with no pin.

**I rejected carrying exp-008's `L2-hub` budget.** Its 2,100 rounds came from a
patience-100 early stop, and the patience-900 re-selection that landed at 11:47
on 2026-09-05 reads 7,990 → **8,725** rounds on the same fold and month, a 4.2x
truncation. Every LightGBM member here is rebuilt at a patience-900 budget.

## Process rules this experiment adopts, from exp-008's failures

1. **One arm name, one launch.** exp-008 lost a finding because two of my own
   scripts wrote `oof/SB-L0lin-july_loo.parquet` concurrently and the cited
   artifact changed under the write-up. Every run here carries a launch-unique
   name, and `blend.py --check` refuses to score an OOF file whose mtime is
   newer than the `result-*.json` that names it.
2. **No linear trees in any member.** exp-008 measured the `lin` variant moving
   up to 15.4 s on one seed between an 8-thread and a 12-thread run of the same
   configuration, because LightGBM's `deterministic=True` does not cover the
   per-leaf linear solve. Plain trees only, and `PRC_THREADS` is pinned and
   recorded in every result file.
3. **Members are prepared at the blend boundary**, floor then residual guard,
   by `prc.blend.build_candidate` (VERSION 4). Ranking members are also floored
   at source now, but the boundary is what the candidate depends on.
4. **Commit as I go.** exp-008 had 6 of ~120 files tracked when it was reviewed.

## Fold rotation and the weight-stability check

Three folds, three validation months, each used once. A fold's weights are
fitted on the other two months' OOF and applied unseen. The desk additionally
asked whether those weights are stable month to month; the rotation already
produces three weight vectors, and RESULTS reports them side by side with the
all-month vector that ships. Instability there is the same failure exp-008 found
per regime — a member's value being a property of the month — and it is the
thing that decides whether a fitted blend is safe to ship at all.

## Screening rule (desk, 2026-09-06)

A candidate member is judged by the **rotated blend RMSE with it in**, compared
against the same rotation without it — never by its own standalone fold RMSE.
The two can disagree in sign: exp-012 measured a member 3.57 s worse on its own
score making the flat blend 0.18 s better, because its errors moved into the tail
where the blend was weak and its correlation with the other members fell from
0.9309 to 0.9132.

Standalone RMSE stays as a diagnostic — it says whether a candidate is a
competent model at all, which is necessary and not sufficient.
