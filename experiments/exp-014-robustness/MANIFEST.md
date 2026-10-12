# exp-014 artifacts

Frames are rebuildable from `code/frames14.py` and `code/q1b_daystate.py`; they
live under `artifacts/exp-014/frames/` (git-ignored) and are listed here with
their sha256 prefix so a later session can tell whether it has the same bytes.

| artifact | sha256 (first 16) | built by |
|---|---|---|
| `ds-ranking.parquet` | `7bf40f3e59b8d9a4` | `code/q1b_daystate.py` (`DayState` fitted on the twelve 2025 months) |
| `ds-train.parquet` | `3513347ae62e3101` | same |
| `tail-ranking.parquet` | `5e2fb1e57075d5c7` | `code/frames14.py` (`prc.tail_features.tail_signals`, per ranking month) |
| `tail-2025-01.parquet` | `20f60edf2e252865` | `code/frames14.py` |
| `tail-2025-02.parquet` | `7cf0935d3ae641dd` | |
| `tail-2025-03.parquet` | `7e44603cbb7855c0` | |
| `tail-2025-04.parquet` | `543049b3be30c2e9` | |
| `tail-2025-05.parquet` | `e4a18a773d8bcdea` | |
| `tail-2025-06.parquet` | `911f1d46c870f499` | |
| `tail-2025-07.parquet` | `be2da4025de62786` | |
| `tail-2025-08.parquet` | `97c382ccb885ffc8` | |
| `tail-2025-09.parquet` | `88a8469eb3e72038` | |
| `tail-2025-10.parquet` | `1618f564e8180dbc` | |
| `tail-2025-11.parquet` | `824d1ff87a04cd25` | |
| `tail-2025-12.parquet` | `f94c451c084f450c` | |

The v2 frames this audit reads are exp-008's (`artifacts/exp-008/frames/`), not
rebuilt here.

**Candidates audited**, both read from `submissions/candidates/`:

| candidate | sha256 |
|---|---|
| `exp-012-blend-1.parquet` | `541b1b1107fe85a13777409ecf3d66e93ab9c7fbc7a0dfedf054fc50daf77618` |
| `exp-012-blend-2.parquet` | `5c57799954cbf0b5df0aff249b758a8d836ce71a91d8836b005e24bb869f8579` |
