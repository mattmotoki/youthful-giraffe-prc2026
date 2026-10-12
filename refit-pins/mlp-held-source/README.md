# refit-pins unit mlp-held-source

The code that wrote the held mlp-007 inputs (prepare-03): prepare.py and the base sources it read from the exp-153 source snapshot (exp-018 dataset.py, exp-134 fit_exp134.py, prc/data.py). Run prepare.py of this unit, then repair_physical.py (refit-src), which writes the 178-name physical list.

Overlay `files/` on the refit tree (release/make_refit_tree.py --unit mlp-held-source) to run this step with the bytes that ran. SHA-256 "ran" is the blob that ran; "shipped" is this file after the path rules.

| path | source | SHA-256 ran | SHA-256 shipped | record |
|---|---|---|---|---|
| `experiments/exp-164-double-down/code/mlp007/prepare.py` | `5087b9d91:experiments/exp-194-gain-lane/b1-recovery/mlp007-held/prepare.py` | `e46066759e983f99…` | `76db18690bb36305…` | results/mlp-007/input-records/INPUTS.json line 109 code_sha256 |
| `experiments/exp-018-fresh/code/dataset.py` | `48468cad9` | `be5d9207560c92e3…` | `be5d9207560c92e3…` | INPUTS.json base_source_code dataset.py |
| `experiments/exp-134-proxies-in-learner/code/fit_exp134.py` | `b289887a8` | `ef861ceeea06cb13…` | `416bcef28c6cd4a8…` | INPUTS.json base_source_code fit_exp134.py |
| `prc/data.py` | `a690f9ec3` | `e613518e6762c6b0…` | `69732391b9599ab3…` | INPUTS.json base_source_code data.py (e613518e) |
