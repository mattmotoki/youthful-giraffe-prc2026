# exp-021 method — SPEC

## My reading of the assignment, and the readings I rejected

**The reading I took.** `docs/METHOD.md` is the future public README, so it is
written for a stranger who has never seen this repository and who may be a
competitor, a reviewer, or the organiser checking reproducibility. That means
findings before architecture, every number traceable, and the negatives given the
same weight as the positives — a reader's time is worth more than our appearance.

**Readings I rejected.**

1. *"Lead with the model."* Rejected. The model is the least interesting part of
   this task and the table in §5 says so: the feature work is worth tens of
   seconds, the blend a few, the stacker under one. Leading with the architecture
   would misrepresent where the work was.
2. *"Quote the headline numbers."* Rejected in favour of the corrected ones. The
   ledger carries corrections on exp-002, exp-004, exp-006 (twice), exp-007,
   exp-009, exp-016 (twice) and exp-019, and several of the headline figures a
   casual read would pick up are the superseded ones.
3. *"Omit the four unreached rows and the clean estimate."* Rejected: a reader
   who sees our january fold number without them will draw the wrong conclusion
   about where the difficulty is, and the honest statement is that with them
   removed **July is the harder half**.
4. *"Present the release as finished."* Rejected. Three release deliverables do
   not exist yet and are marked `[TODO — release lane]` rather than described in
   the future tense as though they did.

## Constraints held

- Under 3,000 words (2,7xx). Plain language; no experiment shorthand without
  expansion.
- Every number cites a ledger row or a RESULTS file, and each citation was
  checked to exist mechanically rather than from memory.
- Nothing published; the document says the public repository does not yet exist.

## Compute

None. No fits, no data reads beyond the checks that verified the citations.
