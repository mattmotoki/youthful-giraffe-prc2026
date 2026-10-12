# exp-015 release — spec

Worker `diversity`, exp-id `exp-015`. Assignment: the reproducible release
pipeline the prize requires (public GPLv3 repository, organisers must reproduce
the result) and that we need anyway to rebuild candidates from scratch.

## My reading of the assignment, and the readings I rejected

I read this as: **one config, one entry point, and the guarantee that the number
we quote and the file we upload come from the same derivation.** The deliverable
is not a script that happens to rebuild the members; it is the property that a
stranger with the raw data gets our submission, and that we cannot quote a board
estimate the artifact does not support.

**The reading I rejected was to reimplement each member inside the release
script.** It reads better — one file, no subprocesses, no dependency on five
experiment directories — and it is wrong, because the members that produced the
shipped candidate live in those directories and a reimplementation would be a
*different* estimator that happens to be named the same. The entry point
therefore *invokes* each member's producer with its declared arguments,
environment and seeds, and records what came out. The one thing it does not
delegate is the blend: weights are fitted here, from the members' out-of-fold
predictions, and applied here to the members' ranking predictions, because that
is exactly the seam where a quoted number and a shipped file drift apart.

**The second reading I rejected was to take regimes from exp-008's feature
cache**, which is what the experiment code does. `artifacts/exp-008/frames/` is
a cache of a build, and a release path that depends on another experiment's
cache is not a release path. Regimes come from the raw movement files through
`prc.blend.regimes_from_raw`, so this pipeline depends on `prc/` and
`data/raw/` alone.

**The third reading I rejected was that the reproducibility gate should pass.**
It cannot, and saying so is the deliverable. The winter member has
`linear_tree=True`, which LightGBM's `deterministic=True` does not cover across
thread counts; exp-012 measured the same seed at the same thread count drawing
148 rounds on one run and 186 on another, 3.85 s apart. A gate that reported
"byte-identical" would be a gate that had not been run. The manifest names the
component, marks it `deterministic: false`, and carries the observed spread —
which is what the assignment asks for in the alternative branch.

## What "no NASA code" turned out to mean

The audit found the answer is narrower and more specific than the phrase.
`references/nasa-airport-throughput-prediction-challenge/` is a tracked snapshot
of **a team member's own repository**, MIT-licensed, and MIT combines with GPLv3
without difficulty — so there is no NASA-owned *code* problem at all.

What cannot ship is that **19 of its notebook cells carry rendered dataframes of
FUSER rows**: NASA challenge data, distributed to participants under the
challenge's terms, and not ours to republish under any license. Stripping the
outputs would make the directory publishable. It is excluded from the release
instead, because nothing in the pipeline imports it and the release loses
nothing. `scripts/check_licenses.py` enforces this as a check on embedded data
rather than as a rule about a directory name, so the same failure is caught
wherever it appears next.

## Deliverables

| | |
|---|---|
| `scripts/build_candidate.py` | the entry point; modes `all`, `members`, `folds`, `blend`, `verify` |
| `experiments/exp-015-release/config/*.json` | one config per candidate |
| `scripts/check_licenses.py` | what may ship, enforced |
| `docs/REPRODUCE.md` | for a stranger |
| `LICENSE` | verbatim GPLv3, 674 lines |
