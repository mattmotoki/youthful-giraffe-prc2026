# refit-pins unit mlp008-inputs

The code that wrote the mlp-008 final inputs (receipts/final-INPUTS.json, SHA-256 230f9eb8): prepare.py 75ea88c9 and io_utils.py e420c53f of cff672a79 (the export holds this version with the release rewrite). It reads learner_wp.py and exp-147 pilot.py from the exp-153 source snapshot and seasonal common.py 29fe8669 (refit-src).

Overlay `files/` on the refit tree (release/make_refit_tree.py --unit mlp008-inputs) to run this step with the bytes that ran. SHA-256 "ran" is the blob that ran; "shipped" is this file after the path rules.

| path | source | SHA-256 ran | SHA-256 shipped | record |
|---|---|---|---|---|
| `experiments/exp-164-double-down/code/mlp008/prepare.py` | `cff672a79` | `75ea88c95f38fb97…` | `79c4622f373b245e…` | experiments/exp-164-double-down/results/mlp-008/receipts/final-INPUTS.json (code_sha256) |
| `experiments/exp-164-double-down/code/mlp008/io_utils.py` | `cff672a79` | `e420c53ff4729d53…` | `ed6973d0a069d962…` | experiments/exp-164-double-down/results/mlp-008/receipts/final-INPUTS.json (code_sha256) |
