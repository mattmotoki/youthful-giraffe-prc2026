# refit-pins unit held-bundle

The first exp-169 fold bundle (inputs/bundle-fold.parquet): prepare_noplan.py, then read_stacks.py.

Overlay `files/` on the refit tree (release/make_refit_tree.py --unit held-bundle) to run this step with the bytes that ran. SHA-256 "ran" is the blob that ran; "shipped" is this file after the path rules.

| path | source | SHA-256 ran | SHA-256 shipped | record |
|---|---|---|---|---|
| `experiments/exp-169-stack-plan/code/assembly.py` | `a690f9ec3` | `0e4536c46d50aa97…` | `0e4536c46d50aa97…` | exp-169 results/stack-reads.json code_sha256 (git a690f9ec3) |
| `prc/data.py` | `a690f9ec3` | `e613518e6762c6b0…` | `69732391b9599ab3…` | import of the step at its commit (trace against the refit tree); no run record pins it |
