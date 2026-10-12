# `analysis/blockpy-rebuild/` — the branch copy the rebuild spec was measured on

`code/block.py` is untouched by everything in this directory. Read
`analysis/blockpy-rebuild-spec.md` first; this is its evidence.

| file | what it is |
|---|---|
| `block_v2.py` | the branch copy, with its own path anchor so it can run from here |
| `block_candidate.py` | the same file with the shipped anchor restored — **exactly** what `code/block.py` becomes at the rebuild, `6fdf3951…` |
| `measure.py` | builds all five months under each variant, every emitted column, fold months under their pin and ranking months unpinned |
| `diffcols.py` | per column: gained / lost / changed where present in both |
| `diffrows.py` | per row: the set that differs at all, and the spillover onto rows whose own class held |
| `density.py` | the EDDM question: pool size against segment density, against finding 7's per-airport false shares |
| `boundary-diff.json`, `row-diff.json`, `interface-columns.json`, `reach-diff.json`, `summary-*.json` | the measurements the spec quotes |

The `class-<month>-<variant>.parquet` intermediates (21 MB) are written to
`work/blockpy-rebuild/`, which is ignored, and are rebuilt in about ten minutes by

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/analysis/blockpy-rebuild/measure.py shipped
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/analysis/blockpy-rebuild/measure.py v2
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/analysis/blockpy-rebuild/diffcols.py
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/analysis/blockpy-rebuild/diffrows.py

Nothing in `work/block-*.parquet` — the caches the serve reads — is touched by any of it.
