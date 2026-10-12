# refit-pins unit held-r4mks

The R4mks fold refits of column R4mk of the held-route bundle (R4mksFjjrefit-month-2025-01 and -07; stage R4mks, 2026-10-02).

Overlay `files/` on the refit tree (release/make_refit_tree.py --unit held-r4mks) to run this step with the bytes that ran. SHA-256 "ran" is the blob that ran; "shipped" is this file after the path rules.

| path | source | SHA-256 ran | SHA-256 shipped | record |
|---|---|---|---|---|
| `experiments/exp-148-short-route/code/fold_serve.py` | `ec40e740c` | `6b37360e467f4dc1…` | `f3cb3a9e5d05ac6a…` | R4mksFjjrefit records code_sha256 (exp-169 inputs/stack10-025/R4mksFjjrefit-month-2025-{01,07}.json) |
| `experiments/exp-148-short-route/code/stacker.py` | `b7fee8fba` | `2a37769e0e95a7c4…` | `598f3f7ab533b847…` | R4mksFjjrefit records code_sha256 (exp-169 inputs/stack10-025/R4mksFjjrefit-month-2025-{01,07}.json) |
| `prc/data.py` | `b7fee8fba` | `471218c5ac3cec80…` | `69732391b9599ab3…` | import of the step at its commit (trace against the refit tree); no run record pins it |
