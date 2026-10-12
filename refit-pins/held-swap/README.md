# refit-pins unit held-swap

The stack10-025 fold bundle (inputs/stack10-025/bundle-fold.parquet): stack10.py prepare() with route_swap.py; it changes only R4mk.

Overlay `files/` on the refit tree (release/make_refit_tree.py --unit held-swap) to run this step with the bytes that ran. SHA-256 "ran" is the blob that ran; "shipped" is this file after the path rules.

| path | source | SHA-256 ran | SHA-256 shipped | record |
|---|---|---|---|---|
| `experiments/exp-169-stack-plan/code/route_swap.py` | `31b83a31a` | `a65b66b886899fa9…` | `a65b66b886899fa9…` | git blob at the unit-025 commit (gainlane TRACE.md) |
| `experiments/exp-169-stack-plan/code/lane_c_swap.py` | `31b83a31a` | `fc90a4cf3d12e13d…` | `87fb8385f622805c…` | import of the step at its commit (trace against the refit tree); no run record pins it |
| `experiments/exp-169-stack-plan/code/stack6.py` | `31b83a31a` | `61a0d690d32d51f8…` | `bf0cda7cbd5d2efc…` | import of the step at its commit (trace against the refit tree); no run record pins it |
| `prc/data.py` | `31b83a31a` | `e613518e6762c6b0…` | `69732391b9599ab3…` | import of the step at its commit (trace against the refit tree); no run record pins it |
