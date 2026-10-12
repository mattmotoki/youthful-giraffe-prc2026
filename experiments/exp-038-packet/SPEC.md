# exp-038-packet — the alignment packet on v10 (Q01), Q07's cached audit, Q02

Assignment `assignments/packet.md`. Worker `packet`, tier hard, laptop, 4 threads, CPU only,
no GPU, no base fits, no uploads. Reviewer `sol`, one round on the final table.

## 1. The reading this experiment took, and the readings rejected

**Q01 is a rebasing, not a re-ranking.** The assignment and
`docs/pro-reasoning-2026-09-08/QUEUE.md` define one algebra — `r = y − p_v10`,
`d = p_candidate − p_v10`, `A = mean(r·d)`, `B = mean(d²)`, full-amplitude gain `2A − B`,
fitted scale `A/B` — and ask what every staged vector since stack-5 is worth **at v10's
insertion point**. So no vector is refitted, no arm is selected, and the table's rows are
not competing for a slot: each is one measurement of one direction against the submitted
predictor.

**"Every served or staged vector since stack-5" is read as the sentence, not as the list
after the colon.** The colon names nine families; `exp-030-window2`'s three staged window
arms are also vectors staged since stack-5 and their rotations are committed, so they are
priced and marked `staged-extra`. *Rejected*: treating the list as exhaustive, which would
have made the row "every staged vector" false while three of them sat in git.

**A member is not a candidate.** `bulk64`, `k2s1`, `k2s2`, `k2w`, `k3-tail-*` are member
out-of-fold columns. `d` is defined on the served vector, so a member enters this table only
through a head that carries it. *Rejected*: pricing member columns directly, which would
have measured a quantity no submission can be built from.

**exp-035's seed arm is skipped, not rebuilt.** Its head-level fold vector does not exist:
`oof/` holds the member columns `k2s1` (three folds), `k2s2` and `k2w` (july_loo only), the
seed-33 december/january member fits were still on Kaggle (`NEXT.md`), and `rotate.py --arm
seeds` has not run. Rebuilding it is a member fit, which the assignment forbids.
`assignments/packet.md`: "Where a candidate's vector cannot be reproduced to its hash, say
so and skip it rather than rebuild it."

**Q02's weather member is `V3-allin`.** `docs/BACKLOG.md` line 15 names its frame's weather
family as the ERA5 event-hour snapshot. *Rejected*: `D1-winter`, whose shipped `deice` arm
carries no external weather (`docs/CLOUD-COMPUTE.md` line 291) — substituting it would answer
a different question.

**Q02's "fully refitted blend/head" is refit at both stages on the arm's own members.** The
shipped blend rule cannot carry a partial general member (`exp-026/code/rule._split` treats
only the `N1-nmtail` specialist's nulls) and `B1-bulk64` is partial by construction, so the
replacement arm's anchor is six members where v10's is seven. That confound is measured
rather than hidden: `replacement-anchorheld` holds v10's anchor and changes only the head,
and `drop-weather` changes only the anchor. *Rejected*: reusing exp-034's committed
`bulk-add-hold` as the addition arm — it sits on the blend-5 anchor, not v10's mixed one, so
it is a gate here and the addition is run again on v10's recipe.

**Q07 stage 1 recomputes the change; it transports no historical number.** Q07's own words
forbid carrying the 0.1119 s of exp-031 to a new candidate, so the endpoint of this packet is
recomputed on exp-031's two nested instruments, with v10 rebuilt on each by the same
`np.where` mask arithmetic.

## 2. What is not touched

No member is refitted. No mask is selected — exp-036's eight-airport list is read from
`analysis/mask.json` and applied unchanged everywhere. The 388 ranking `noac` rows and the
EHAM hold are untouched by every candidate but exp-037's, which claim only `noac`. Nothing is
uploaded and no candidate parquet is written.

## 3. Deliverables

`code/vectors.py` (the gated registry), `code/align.py` (the algebra and the block
bootstrap), `code/q01.py`, `code/amplitude.py`, `code/q07.py`, `code/q02.py`,
`code/checks.py`; `packet/` (per-row `d` per candidate per fold, hashed); `analysis/`;
`RESULTS.md`; ledger rows; one Sol round under `review/`.
