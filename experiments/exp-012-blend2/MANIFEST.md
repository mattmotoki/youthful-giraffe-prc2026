# exp-012 artifacts outside git, addressed by hash

`submissions/` and `artifacts/` are git-ignored repo-wide, so candidates and
member predictions are addressed here instead, per COMMON.md's rule that a
measurement cites a path inside the repo and a verified hash.

Regenerate a candidate with
`python make_candidate.py <members>.json <ranking>.json <n>`; the weights and
every input are recorded in `blend-config-<n>.json`, which IS committed.

## Submission candidates

| file | bytes | sha256 | note |
|---|---|---|---|
| `submissions/candidates/exp-001-retro-v1-c6.parquet` | 4707513 | `504d489d239e704e790f4b876450f132349d5eaf90b9288a89ea0d9c7d919c39` | retro-v1 c6: + residual offset, noac head, floor 60; july 343.6 / dec 222.4 / jan 349.0. Superseded by c9. |
| `submissions/candidates/exp-001-retro-v1-c7.parquet` | 4707585 | `e8261bb4fb589951ccfc012d8c18ccb083b3e93af42fc391b67b56509b5c7c24` | retro-v1 c7: c6 blended 50/50 per row with prc.tail_model, no fitted weight; rotated july 323.5 / dec 218.1 / jan 347.1. |
| `submissions/candidates/exp-001-retro-v1-c8.parquet` | 4707497 | `bb720e1a81aa8ef36a7917344cb9b1ee88c6ebcf324dcb616bd127a41b622a1f` | retro-v1 c8: c6 under pinned seeds + residual guard; july 321.1 / dec 222.8 / jan 347.5. Guard later shown to be a linear-leaf instrument. |
| `submissions/candidates/exp-001-retro-v1-c9.parquet` | 4707497 | `ee8fa94dfec2c06f6f2cc545c1cff1595981e418582c7c00d807baeb568c71d3` | retro-v1 c9: plain trees (linear leaves reversed on the ranking analogue), 5-seed bag, no guard; july_loo 315.9 / jan 348.4, board est 330.7. |
| `submissions/candidates/exp-001-retro-v1.parquet` | 4707625 | `fa6195ffb21682e10f784d02af94e43e6236c9520e387335f2ae3dcbcda68201` | retro-v1 v1: LightGBM on the actual columns, linear leaves, guarded; july 531.4 / dec 287.0. Superseded; kept as a far mapping point. |
| `submissions/candidates/exp-008-blend3.parquet` | 4707529 | `26c40e69e0a8f3b018d2eb9b3c8d2c64e8ce7df5167d7aa194a038b997f9a231` | exp-008: rotated NNLS of tail-v1 0.475 + Huber-LGB 0.431 + linear-leaf LGB 0.095; july_loo 300.0 / dec 216.1 / jan 344.3, board est 320.4. Superseded by exp-012-blend-1. |
| `submissions/candidates/exp-012-blend-4.parquet` | 4707609 | `3b7e558a0b542690673f5ad01ade7a0c210a998905928e5e95434c752549337d` | **Head of the queue.** Six members: five general fitted by NNLS, plus the NM-tail specialist inside `nm_late`. Board est **313.17**, 4.44 s better than blend-3b, on exp-018's fresh lineage at weight 0.62. Uses the MATCHED one-seed pair (`exp-018b-s1` + its one-seed OOF), so the served member is the estimator the weights were fitted on. |
| `submissions/candidates/exp-012-blend-3b.parquet` | 4707593 | `b6b864a21d73ba4ebc6d2293282eb1b1c898e501866726a7ee4c3b09cd261001` | **Head of the exp-012 queue, and REBUILDABLE.** Re-issue of blend-3 on members that all exist and can be rebuilt at the declared 6-thread configuration; board est 317.61. The NM-tail member is prc.nmtail VERSION 4 on both sides. Replaces f91db30a, whose CatBoost input was destroyed. |
| `submissions/candidates/exp-012-blend-3.parquet` | 4707611 | `f91db30a7f3bc4da41a1f1599627cdbe48741ba80c8017105b8a1ba79f9631e0` | **NOT REBUILDABLE — superseded by `exp-012-blend-3b`.** Its CatBoost input `48acb7e7` was destroyed by my own reproducibility rebuild (RESULTS, "I destroyed a member artifact"); the file and its numbers are intact and honest, but one of its five inputs no longer exists, and the desk's rule is that the queue carries only candidates we can rebuild. Kept on disk as the pre-loss reference. Flat 0.25 over T-tailv2, C2-cat, L2-hub and the REPAIRED winter member, plus N1-nmtail 0.352 inside nm_late only; board est 317.66. Winter member is ds_deice_hr alone, which cannot leave its training range - the block it replaces put 4,640 january cells outside it. |
| `submissions/candidates/exp-012-blend-2.parquet` | 4707577 | `5c57799954cbf0b5df0aff249b758a8d836ce71a91d8836b005e24bb869f8579` | Same rule and members but the winter member carries the full 25-column day-state block; board est 318.02. Kept as the pre-repair comparison, not for upload ahead of blend-3: its January 2026 exposure is 18 airport-days and 7,564 rows the folds cannot price. |
| `submissions/candidates/exp-012-blend-1.parquet` | 4707569 | `541b1b1107fe85a13777409ecf3d66e93ab9c7fbc7a0dfedf054fc50daf77618` | Three members, rotated NNLS (T-tailv2 0.444, C2-cat 0.223, L2-hub 0.335); board est 318.81. The last candidate before the four-member flat rule and the nm_late specialist. Superseded twice. |

## Member ranking predictions

| file | bytes | sha256 |
|---|---|---|
| `artifacts/exp-012/pred/T-tailv2-ranking.parquet` | 4707333 | `e9d97994642c9c3a95dbb5249a7e6f687709755411b4d633f3c9829ec783e3c0` |
| `artifacts/exp-008/pred/C2-d8p-ranking.parquet` | 7739019 | `48acb7e7cba9f2135239246466a56040a0b756ae439b9983031997f0bdf3ae5c` |
| `artifacts/exp-008/pred/L0-lin-ranking.parquet` | 7739019 | `5ba75dd24ee3340e3d895d697d7d61474433fdfab605f0597dbe5569d0125782` |
| `artifacts/exp-008/pred/L2-hub-ranking.parquet` | 7739019 | `8da562fc1b6590cc9f58e5b8fffa8a940f3b3928beaac7becf7ee497f9432d83` |
| `artifacts/exp-008/pred/T-tail-ranking.parquet` | 4707341 | `d1255fafb42738d951ce46840aff603c475be861e1a30173aed2fcc0c044b4fd` |

## exp-012 out-of-fold members

| file | bytes | sha256 |
|---|---|---|
| `experiments/exp-012-blend2/oof/R0-retro-december.parquet` | 2810859 | `8291b08f92555b345b7eb52672bea8c264e460794931ceb9b124463e512258bd` |
| `experiments/exp-012-blend2/oof/R0-retro-january.parquet` | 2651453 | `ccd938011f77beade2b41fe8f4ecd4e702f497913e3ed593f63021eb12822810` |
| `experiments/exp-012-blend2/oof/R0-retro-july_loo.parquet` | 3157396 | `44c2f6cb1b1919f8986eb9ba7767c7ffa4850e3148a6fc59446d7bb1c8807bd6` |

## Pinned member (review round 1, blocking finding B1)

- `oof/T-tailv2-pinned.parquet`  `a14d040316ba282c35071c6f30ef854b8ae077f1c7c0999a1fe795554f06f79e`

A byte copy of `experiments/exp-009-noac2/oof/exp-009-noac2-oof.parquet` as it
stood when candidate 1's weights were fitted. exp-009's file is **uncommitted and
still moving**: the working tree is `a14d0403` and HEAD is `7dacc589`, differing
on 816 rows, all of them `noac`, because exp-009 changed the noac head. The
0.4438-weight member of a submission candidate cannot sit on an address that is
not in git and is being edited by another lane, so the bytes live here now.
Rebuilding candidate 1 against this copy reproduces
`541b1b1107fe85a13777409ecf3d66e93ab9c7fbc7a0dfedf054fc50daf77618` exactly and
leaves the board estimate at 318.81, so this is a provenance fix and not a
number change.

## Member pins (durable addresses for every blend input)

Regenerate and verify with `python pin_members.py [--write]`. A member whose
source is tracked and clean keeps its git blob as the address; anything
untracked or modified is copied into `pinned/` and committed, because a
submission candidate cannot rest on bytes another lane is still editing.

| member | address | sha256 | how the address is held |
|---|---|---|---|
| `C2-cat` | `pinned/C2-cat--C2-d8p-july_loo.parquet` | `f84fd89b4ff7cf8e…` | pin, committed |
| `C2-cat` | `pinned/C2-cat--C2-d8p-december.parquet` | `fdef30450a62e535…` | pin, committed |
| `C2-cat` | `pinned/C2-cat--C2-d8p-january.parquet` | `8450c5ac967dd8e1…` | pin, committed |
| `D1-winter` | `experiments/exp-007-winter/oof/d1-ds-s7-july_loo.parquet` | `9701134d50d447db…` | git blob, d347b0d |
| `D1-winter` | `experiments/exp-007-winter/oof/d1-ds-s7-december.parquet` | `7ab24bc3c6694fcb…` | git blob, 8f9a5e1 |
| `D1-winter` | `experiments/exp-007-winter/oof/d1-ds-s7-january.parquet` | `2a64a703de0f1639…` | git blob, 8f9a5e1 |
| `D1-winter` | `experiments/exp-012-blend2/published/D1-winter-deice-v1-july_loo.parquet` | `bbe71ea2eec6d47e…` | git blob, 45048d1 |
| `D1-winter` | `experiments/exp-012-blend2/published/D1-winter-deice-v1-december.parquet` | `bfc3497a47c51a47…` | git blob, 45048d1 |
| `D1-winter` | `experiments/exp-012-blend2/published/D1-winter-deice-v1-january.parquet` | `cdb15e328154a10c…` | git blob, 45048d1 |
| `L2-hub` | `pinned/L2-hub--L2-hub-july_loo.parquet` | `5d3a93e5cd518cda…` | pin, committed |
| `L2-hub` | `pinned/L2-hub--L2-hub-december.parquet` | `96e6f17db95c791d…` | pin, committed |
| `L2-hub` | `pinned/L2-hub--L2-hub-january.parquet` | `e60baec681d15145…` | pin, committed |
| `T-tailv2` | `experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet` | `a14d040316ba282c…` | git blob, 4a4696e |
