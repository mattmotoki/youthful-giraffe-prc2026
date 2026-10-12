# exp-026 rebuild — SPEC

## My reading of the assignment, and the readings I rejected

**The reading I took.** The assignment's premise is that blend-5 and stack-5 can
be rebuilt on this laptop with no model fit and no competition data, and that
the only missing object is the submission template. **The first half is right
and the second half is wrong**, and finding that out was the first task. Every
*fold* number in the assignment — the ladder's clean rotations, stack-5's
rotation, the swap decomposition — is reachable from committed bytes alone, and
this experiment computes all of them. Every *ranking* vector needs two per-row
quantities that exist in no committed file: the regime label and
`d_takeoff_aobt`. So the deliverable splits in two: the OOF-side work is done
now and reviewed now; the ranking-side work is reduced to one cached side table
(`code/regime_table.py`) and two commands, so that the minute `data/raw` exists
the candidates are minutes away rather than a rebuild away. The desk confirmed
this reading on 2026-09-07 after checking every committed parquet itself.

**Readings I rejected.**

1. *"Report `blocked`."* Rejected. The assignment's own escape clause asks for
   `blocked` only if the blend step cannot be separated from the data step. It
   can: `prc.blend.build_candidate` touches the raw data in exactly one call,
   `prc.data.load_ranking`, for exactly two columns' worth of derived
   quantities. `code/regime_table.py` is that call and nothing else;
   `code/rebuild_blend.py` is the rest of `build_candidate` with the write step
   replaced. Three quarters of the assignment does not depend on it at all.

2. *"Stage a best-effort blend-5 vector now and label it provisional."*
   Rejected. Without the regime mask the `nm_late` rows (16,090, 4.67 % of the
   file) would carry the wrong weight vector, and without `d_takeoff_aobt` the
   residual guard cannot be evaluated. A staged vector that is wrong on 4.67 %
   of rows is worse than no vector, because it looks like the object it is not
   and its SHA would enter the record. The fingerprint check the assignment
   asks for is exactly the thing that would have caught it, and it cannot run.

3. *"Rebuild blend-4's candidate as the ladder's foot."* Rejected on evidence:
   two of blend-4's six ranking inputs, `T-tailv2` (`e9d97994…`) and `L2-hub`
   (`29adfc27…`), lived only in `artifacts/` and are gone. The other four are
   byte-identical to files in `published/` (hashes in RESULTS §2). blend-4 is
   therefore fold-computable and not servable, and so is every ladder rung that
   holds either member. The servable ladder is the leave-one-out family over
   blend-5's seven published members, which is what this experiment writes
   configs for.

4. *"Reuse `experiments/exp-016-stack/code/members.py` for the fold frames."*
   Rejected: its `gates()` reads `artifacts/exp-008/frames/` and
   `artifacts/exp-014/frames/`, both gone. `code/frames.py` builds the same
   frame from the five gate columns the C2-cat and L2-hub OOF files carry
   beside their predictions, and proves the substitution rather than asserting
   it — the derived regime is checked row for row against the `regime` column
   exp-016 froze, and the rule fitted on these frames reproduces both
   `blend-config-4.json` and `blend-config-5.json` to ±0.0000 s on all six
   folds. That gate runs before anything downstream is believed.

5. *"Take exp-012's swap table as the ladder's numbers."* Rejected after
   measuring: that table (`exp-012 RESULTS.md`, "the three swaps compose") sits
   about 0.45 s above the rotated numbers in the same lane's own configs, so the
   two are different instruments. This experiment's ladder agrees with the
   configs exactly, which is the instrument the shipped candidates were scored
   on, and RESULTS records the disagreement rather than averaging it away.

## Two addresses in the assignment that do not exist

Recorded so a later reader does not go looking. `experiments/exp-012-blend2/code/`
has no `code/` subdirectory — that lane's modules sit flat in
`experiments/exp-012-blend2/` (`blend.py`, `make_candidate_hybrid.py`,
`screen_by_blend.py`, …). And `published/V3-allin-v1-ranking.meta.json` names
`experiments/exp-001-retro-v1/result-v3-allin-rank.json` as its `source_record`;
that file is not in git and went with the workstation (RESULTS §5e item 3).

**How the readings turned out** (2026-09-07 16:30 UTC, after `data/raw` landed
at 16:04). Reading 1 held: the split was real and three quarters of the
assignment ran without data. Reading 2 held and was worth holding — the gates
table showed 16,090 `nm_late` rows, so a best-effort vector would have carried
the wrong weight vector on 4.67 % of the file, and the fingerprint check that
would have caught it did run, and passed. Reading 3 held: blend-4 is still
unrebuildable and the leave-one-out family is still the servable ladder.
Reading 4 held: the substituted gates reproduce the ranking regime census to the
row (`nm_late` 16,090 and `noac` 388 against exp-014's). Reading 5 held. The
strongest confirmation is one this SPEC did not predict: the blend-5 vector
these readings produce matches the object that was actually uploaded and scored
to **1.455e-10 s** on all 344,841 rows (RESULTS §6a).

## Frozen before any number was computed

- The rule is the one both configs declare: NNLS over all members inside
  `nm_late` where the specialist is defined, NNLS over the general members
  elsewhere, the specialist at exactly 0.000 there.
- Weights for a served rung are fitted on all three folds; the rotation scores
  the rule and never chooses the weights (exp-012's convention).
- `clean` is exp-020's fixed exclusion list through `prc.data.rmse_both`; the
  board predictor is `clean + 4` (desk, 2026-09-06); the board estimate is the
  two ranking months in quadrature at their row counts.
- The stacker arm is `a5` (members only), the one stack-4 shipped. The `a2`
  arm reads gate features that did not survive and is not reported.

## Compute

2 threads (assignment share), no GPU, no fits beyond the 15 LightGBM trainings
of the stacker rotation (40 s wall in total).
