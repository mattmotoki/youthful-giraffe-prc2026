# refit-pins unit held-r4f

The R4 fold refits of column F of the held-route bundle (R4Fjjrefit-month-2025-01 and -07; stage R4, 2026-10-01).

Overlay `files/` on the refit tree (release/make_refit_tree.py --unit held-r4f) to run this step with the bytes that ran. SHA-256 "ran" is the blob that ran; "shipped" is this file after the path rules.

| path | source | SHA-256 ran | SHA-256 shipped | record |
|---|---|---|---|---|
| `experiments/exp-148-short-route/code/fold_serve.py` | `3ba3625a3` | `4a8dd063b91d0b3b…` | `23faaaf901621ae0…` | R4Fjjrefit records code_sha256 (exp148-short-route results/fold/R4Fjjrefit-month-2025-{01,07}.json) |
| `experiments/exp-148-short-route/code/stacker.py` | `3ba3625a3` | `71a1381ac687d1e5…` | `9d1ed005e25e00d4…` | R4Fjjrefit records code_sha256 (exp148-short-route results/fold/R4Fjjrefit-month-2025-{01,07}.json) |
| `prc/data.py` | `3ba3625a3` | `471218c5ac3cec80…` | `69732391b9599ab3…` | import of the step at its commit (trace against the refit tree); no run record pins it |
