# `V3-allin-v1` — published member interface

For exp-016, this lane's blend-5, and any other consumer.
**Use these files; do not refit this member.**

## What it is

retro-v1's `V3-allin` on the **v3 frame** (171-column frozen source set) —
residual target, noac head, learning rate 0.03, 660 fixed rounds, one seed,
12 threads. It is the first member in this blend from outside the v2 lineage,
and that is the point of it: see "Why it matters" below.

## Files

| what | path | fold | sha256 |
|---|---|---|---|
| OOF | `published/V3-allin-v1-july_loo.parquet` | `july_loo` | `149114c433bd02e4e0ca56ff8919c11ad19b9f843c7ddb207a79e2f78a87e47d` |
| OOF | `published/V3-allin-v1-december.parquet` | `december` | `52b9f7ed7265ac936d2b5abf14cf692d12dab41a6f574e17b8f4f2a9c404d83b` |
| OOF | `published/V3-allin-v1-january.parquet` | `january` | `003cf3399686f97c847d615c9dbc6b337a85f6fa8735bc78ddbe1ffce5e04bf2` |
| ranking | pending | `ranking` | retro-v1's twelve-month fit is running |

Each artifact has a `.meta.json` beside it declaring the fold, the frame column
count, and the thread count. `verify_member.py` asserts the fold against the
parquet and reports the frame contract.

## The two failure modes this interface exists to stop

Both are real and both happened in the source lane on 2026-09-06. Neither is
visible in a filename, a row count, or an id join.

1. **Wrong fold, right everything else.** `V3-allin-member-july.parquet` carried
   fold `july` (six-month training, RMSE 316.46) where a consumer expected
   `july_loo` (eleven-month, 301.23) — identical ids, identical `y_true`, same
   190,713 rows, 121 s RMS apart. Now quarantined. **A four-column Files table
   declares the fold and `verify_member.py` asserts it.**
2. **Wrong frame, right rows.** `build_submission` was building a 50-column frame
   while the folds ran on 171. It survived only by crashing on `KeyError: 'noac'`;
   with the noac head off it would have written a member with the right rows, the
   right ids and a silently wrong frame. **No consumer can detect this from the
   artifact**, so the producer must state it: `n_frame_columns` and a
   `column_sha` in the sidecar. The sha is not yet emitted — exp-018 emits one
   (`35b5b6687e73` over 111 features) and that is the shape to copy.

## Why it matters to this blend

Board **−1.248 s** as a sixth member (nnls), weight 0.26–0.29 across folds —
the largest single-member addition this lane has measured, and twelve times the
rotation's 0.1025 s unconditional half-width.

It also refutes a finding of this lane's: member saturation after the fifth
member was measured entirely inside the v2 lineage, and a sixth member from a
different frame earning 0.29 says that ceiling was a property of the lineage,
not of blending.

## Scores

| fold | RMSE | tail RMSE | no-NM RMSE | rows |
|---|---|---|---|---|
| july_loo | 301.23 | 2739.44 | 1228.14 | 190,713 |
| december | 217.55 | 2031.01 | 877.59 | 165,677 |
| january | 343.95 | 6001.84 | 2868.49 | 153,706 |

`july_loo` and `january` from `result-V3-allin.json`; `december` from
`result-V3-allin-dec.json` — same argv but for `--folds`, both at 660 fixed
rounds, which is what makes these three one config rather than three.
