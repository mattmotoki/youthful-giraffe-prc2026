# exp-016 artifacts

`submissions/` is git-ignored repo-wide (`.gitignore:22`), so no candidate
parquet is tracked here — the same convention exp-012's candidates follow. The
durable address is the path plus the sha256 below, and `candidate-1.json` /
`candidate-2.json` are the committed records that carry them.

## Candidates

| path | sha256 | board estimate |
|---|---|---|
| `submissions/candidates/exp-016-stack-1.parquet` | `099d46702b4ac393e4f604b50979333f913b00daaad7033da2e0f46740b7f99d` | 317.280 |
| `submissions/candidates/exp-016-stack-2.parquet` | `cafb2d49ece4f8c783ea4f69049d087dde06d5f876471b29b96a80c12ebf9127` | 317.529 |
| `submissions/candidates/exp-016-stack-3.parquet` | `c39bb173e4233ac5cb5a30f290282571d0f61fccb53a20550ee1c4420c4a4fbe` | 317.215 |

stack-1 and stack-2 are built on the pre-repair block winter member (baseline
blend-2 318.017); **stack-3 is built on the published deice member** (baseline
317.661, `members-blend3.json`, pins in `pins-members-blend3.json`). Board
estimates are not comparable across the two bases.

**None of the three is rebuildable as of 2026-09-05.** All three consume
`artifacts/exp-008/pred/C2-d8p-ranking.parquet`, whose bytes moved from
`48acb7e7…` to `56f95373…` when exp-012's end-to-end reproducibility run refit
the CatBoost member; `artifacts/` is git-ignored and the original was not
preserved. `code/members.py`'s pin check catches it — `verify(require_ranking=True)`
returns `['rank:C2-cat']` against both `pins.json` and `pins-members-blend3.json`,
which is what the pinning exists for.

**The fold numbers are unaffected.** The stacker is fitted on OOF and scored on
OOF, and no OOF address moved; only the bytes the fitted stacker is *applied* to
changed. So every board estimate in this experiment stands and what is lost is
the ability to regenerate the served files. Rebuilding stack-3 on the C2-cat present at
2026-09-05T23:25 (`56f95373`) moved it by **RMS 8.0 s, max 666.6 s, 344,392 of
344,841 rows** — for scale, stack-3 and blend-3 differ by RMS 29.3 s, so the
rebuild is the same object in character.

**That 8.0 s is provisional and describes a member set that will not ship.**
exp-012's end-to-end rebuild is still rewriting these files: as of
2026-09-06T00:03 three of the five ranking artifacts had moved again
(`rank:C2-cat`, `rank:L2-hub`, `rank:N1-nmtail` all flagged by
`members.verify`), and `prc.nmtail` VERSION 4 deliberately changes N1-nmtail's
bytes by adding the `deterministic` / `force_row_wise` flags it was missing. The
figure is re-measured against the final set before stack-3b ships.

`exp-016-stack-3b` supersedes stack-3 once exp-012 publishes a final member set;
`members-blend3b.json` is written and waiting. The three files here stay on disk
with this note rather than being deleted.

Before that, both rebuilt bit-for-bit from `code/submit.py 1` and
`code/submit.py 2` against the then-pinned members. Candidate 1 reproduced `099d4670…` exactly after both review
rounds. **Candidate 2's hash changed in round 2**: `submit.py` was not declaring
`airport`/`regime` as LightGBM categoricals while every rotation was, so the
served model was not the scored one (up to 149.7 s per row). The superseded
build was `643a18db897eb6bcb14c6ba0506e0df1c41070471c67b2647cbf181fe7857c44` —
recorded here so an upload log naming it can be identified as the wrong object.

## Members, pinned

`pins.json` holds the sha256 of every member OOF and ranking file this
experiment reads, and `members.verify()` re-checks them at the start of every
run — the run aborts if any moved. Five members across four lanes:

| member | OOF | ranking |
|---|---|---|
| `T-tailv2` | `experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet` | `artifacts/exp-012/pred/T-tailv2-ranking.parquet` |
| `C2-cat` | `experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet` | `artifacts/exp-008/pred/C2-d8p-ranking.parquet` |
| `L2-hub` | `experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet` | `artifacts/exp-008/pred/L2-hub-ranking.parquet` |
| `D1-winter` | `experiments/exp-007-winter/oof/d1-ds-s7-{fold}.parquet` | `artifacts/exp-012/pred/D1-winter-ranking.parquet` |
| `N1-nmtail` | `experiments/exp-010-nmtail/oof/nmtail-phi_{fold}.parquet` | `artifacts/exp-012/pred/N1-nmtail-ranking.parquet` |

## Out-of-fold predictions

`oof/exp-016-oof-{fold}.parquet`, written by `code/final_tables.py --rebuild-oof`
(round 2 found them committed with no producing code), tracked, in the
`COMMON.md` contract
(`MVT_ID_mvt`, `fold`, `y_true`, `y_pred`) plus the four other configurations and
the two baselines as extra columns, and the regime/airport columns needed to
re-slice them. `y_pred` is the shipped candidate 1 configuration and reproduces
its rotated 295.527 / 213.934 / 342.689.

## Frames read, not written

exp-008's `artifacts/exp-008/frames/v2-*.parquet` and exp-014's
`artifacts/exp-014/frames/ds-*.parquet`. Nothing here rebuilds them.
