# exp-008 large artifacts (outside git, addressed by hash)

`oof/*.parquet` and `artifacts/exp-008/**` are git-ignored. This file gives each
one a durable address, per COMMON.md's rule that a measurement cites a path
inside the repo and a verified hash.

Regenerate any OOF file with the `argv` recorded in the matching
`result-<arm>.json`; regenerate a ranking member with `run.py --submit` (or
`tail_ranking.py` for `T-tail`) and the same arm name.

## Out-of-fold predictions

| file | bytes | sha256 |
|---|---|---|
| `experiments/exp-008-diversity/oof/C-cat-december.parquet` | 4890338 | `fdef30450a62e5350e67d2b447dfdbbd5b3ceaac1c0e9f71acbc6d006f75144f` |
| `experiments/exp-008-diversity/oof/C-cat-january.parquet` | 4617507 | `8450c5ac967dd8e1a67f8c368fb70ddecd13f209e5a4dfa57f3f007825b84409` |
| `experiments/exp-008-diversity/oof/C-cat-july.parquet` | 5510070 | `f996f375893cf8d5e3aeba03cdfbea32bac4809d199dee1654b564f1b27eb90c` |
| `experiments/exp-008-diversity/oof/C-cat-july_loo.parquet` | 5510187 | `f84fd89b4ff7cf8e7fc8d4620a0a8854e7d86075e8390437fd186ea6e31514c1` |
| `experiments/exp-008-diversity/oof/C1-d8-december.parquet` | 4890264 | `e6ecbc9fe66626aa9843045e131af270474a0f775a9ce4f13c7bc5941e31a810` |
| `experiments/exp-008-diversity/oof/C1-d8-january.parquet` | 4617685 | `0cecc4da4cea360ef505e0d16ec5cf1d5516428179a46540d0b803ddc7da31c5` |
| `experiments/exp-008-diversity/oof/C1-d8-july.parquet` | 5510070 | `f996f375893cf8d5e3aeba03cdfbea32bac4809d199dee1654b564f1b27eb90c` |
| `experiments/exp-008-diversity/oof/C1-d8-july_loo.parquet` | 5510197 | `c8679b39d7dc8635e1f16f3c96d3a1b67f7a5a3c3a6e31300f0b92d4c8461294` |
| `experiments/exp-008-diversity/oof/C1-d8-m11winterES-july_loo.parquet` | 5509349 | `436747ab9cb682a62c004b4094d7342a61e86b0bbeaba5932d860f7d9bc87133` |
| `experiments/exp-008-diversity/oof/C1-d8h60-july.parquet` | 5510086 | `25df1316486397265062c2b721e19b9b994bee2b8e583848e06037b9604a1856` |
| `experiments/exp-008-diversity/oof/C2-d8p-december.parquet` | 4890338 | `fdef30450a62e5350e67d2b447dfdbbd5b3ceaac1c0e9f71acbc6d006f75144f` |
| `experiments/exp-008-diversity/oof/C2-d8p-january.parquet` | 4617507 | `8450c5ac967dd8e1a67f8c368fb70ddecd13f209e5a4dfa57f3f007825b84409` |
| `experiments/exp-008-diversity/oof/C2-d8p-july_loo.parquet` | 5510187 | `f84fd89b4ff7cf8e7fc8d4620a0a8854e7d86075e8390437fd186ea6e31514c1` |
| `experiments/exp-008-diversity/oof/L0-lin-december.parquet` | 4890290 | `626f9d4c1136c524063e8ef689d64aca3d6e25207987bffd2bf4e24232b62182` |
| `experiments/exp-008-diversity/oof/L0-lin-january.parquet` | 4617451 | `3757df1efe764020b00a385b1e588fd7e41520a8df9c519ee78d727395ef6e18` |
| `experiments/exp-008-diversity/oof/L0-lin-july.parquet` | 5510238 | `a728efda4d343a50ad9a62586e9d0971897bcb888d32621803511b3744b1e9b7` |
| `experiments/exp-008-diversity/oof/L0-lin-july_loo.parquet` | 5510315 | `a4e9c97be00bc7d651543b3a074ed1217d29e2f10aa01c5c00376c12b457ffce` |
| `experiments/exp-008-diversity/oof/L1-plain-december.parquet` | 4890378 | `55d9bbbdfc8b1bf0c9d99bd7605475aba1b9d30f5962453d076d3c32f885c828` |
| `experiments/exp-008-diversity/oof/L1-plain-january.parquet` | 4617467 | `b26bdaab334b8e3d35eb788a3889ad2eea075eaa5cb84a6756511f30132178a4` |
| `experiments/exp-008-diversity/oof/L1-plain-july.parquet` | 5510126 | `94df765182d8c3ef8164b01c4b6724697f80788d135338e3d89b627821b9e05c` |
| `experiments/exp-008-diversity/oof/L1-plain-july_loo.parquet` | 5510275 | `d04ad6d8e1166caf15f10f075ccaf8487d676b3ff2f2658fb6e7dd4c47b41327` |
| `experiments/exp-008-diversity/oof/L2-hub-december.parquet` | 4890378 | `96e6f17db95c791d2a122227d5ffceb1e84e5e3b80249f32b767cb79e8de1081` |
| `experiments/exp-008-diversity/oof/L2-hub-january.parquet` | 4617507 | `e60baec681d15145b6346fe86d66e4d6a9b467d836cd3291a16d4a853176b7a6` |
| `experiments/exp-008-diversity/oof/L2-hub-july.parquet` | 5510118 | `52c41b6e179ce159827874ccfdc79f44f95a56390737469288760b44c12fee33` |
| `experiments/exp-008-diversity/oof/L2-hub-july_loo.parquet` | 5510259 | `5d3a93e5cd518cda7b20e32802b945311f200aa949fa98f9c5579f2309d32d04` |
| `experiments/exp-008-diversity/oof/L3-monoraw-december.parquet` | 4890362 | `deeb207aee72d2a042409bcd3b72eea52d625e825e0b9d96802a6d35cee8c03e` |
| `experiments/exp-008-diversity/oof/L3-monoraw-january.parquet` | 4617547 | `c0bc8257d4a220792a01b2fa724dfb628f11e1022f403a72da73078bff86c724` |
| `experiments/exp-008-diversity/oof/L3-monoraw-july.parquet` | 5510238 | `54fc423f601b75f799315335cd3bc46fbb08bbbbfd313dc12fd801d1c237bde7` |
| `experiments/exp-008-diversity/oof/L3-monoraw-july_loo.parquet` | 5510323 | `37b326f5b6e5357b266466ff9f1e25f804e0b7581ab160d2617eabe2cd7456e4` |
| `experiments/exp-008-diversity/oof/L3r-monores-december.parquet` | 4890370 | `50459d78ce810e9a8b0b8d1dd60c1e94a249ca10bd34f8d988a3bf4456362b5e` |
| `experiments/exp-008-diversity/oof/L3r-monores-january.parquet` | 4617483 | `ecc7b5ead0b1239e448d24f45e188b95f07419c320f410641f2ac923a11487ec` |
| `experiments/exp-008-diversity/oof/L3r-monores-july.parquet` | 5510126 | `0bcf9827bf47f2597f7fecd72d1a4f9591e7b71fcff34259f0f2ded3850b2238` |
| `experiments/exp-008-diversity/oof/L3r-monores-july_loo.parquet` | 5510275 | `b00988e4ecaeed3fc7ac328154475105cdb310d8328373a6ad19220104e2f107` |
| `experiments/exp-008-diversity/oof/SB-L0lin-dec-december.parquet` | 11279811 | `ea2391c99de4a2394c0c866749b6e24c2bf1a0374fcf319267a09f6695c27d4c` |
| `experiments/exp-008-diversity/oof/SB-L0lin-july_loo.parquet` | 12701236 | `e59b96bc725e1cba242fc2f02c2f7d6461cde08ffccd24aa707a5e3cabeff4de` |
| `experiments/exp-008-diversity/oof/SB-L2hub-july_loo.parquet` | 12701212 | `e1f648de1eeadfae74b2db4796f1058b99793f2f8c339ac779cfece72ea04f54` |
| `experiments/exp-008-diversity/oof/X-xgb-december.parquet` | 4254693 | `2f1e9adfd1b08085a098e53943b76273aba06ed4c307a28aa9e333819d109597` |
| `experiments/exp-008-diversity/oof/X-xgb-january.parquet` | 4027308 | `475a7c9c7a6eaac65fe3cc130cfbce474c0829295a8f2f4f236cc558908c7ad3` |
| `experiments/exp-008-diversity/oof/X-xgb-july.parquet` | 4753358 | `1e519daac84ead468cab6975476eb0b2437d10c33dc481b01484ef3dd1337b5d` |
| `experiments/exp-008-diversity/oof/X-xgb-july_loo.parquet` | 4765835 | `be6c5f98f25ac21c3fcf97b2ddb59378891d20f88b0228eab0787c66c6058001` |
| `experiments/exp-008-diversity/oof/X1-d8-december.parquet` | 4254629 | `62c3d4555db1e05a372b78d168bf6060f9f3b793bd8c4f3a0961ff7fd4d41a26` |
| `experiments/exp-008-diversity/oof/X1-d8-january.parquet` | 4026881 | `ec7947292f5ccbd8dc55975d9665e20a52d9808db6180d2a4e2e155f99e2e794` |
| `experiments/exp-008-diversity/oof/X1-d8-july.parquet` | 4753358 | `1e519daac84ead468cab6975476eb0b2437d10c33dc481b01484ef3dd1337b5d` |
| `experiments/exp-008-diversity/oof/X1-d8-july_loo.parquet` | 4767567 | `e2120451cf39b915c40071acdc41a8e02b495e6720d712a9c33dcc8e69ee4ceb` |
| `experiments/exp-008-diversity/oof/X1-d8-m11winterES-july_loo.parquet` | 4767818 | `2c581f95dfddbf88a425c5c3f31c7d1243bda2f3e6925aa30576474652c56c06` |
| `experiments/exp-008-diversity/oof/X2-d8p-december.parquet` | 4254693 | `2f1e9adfd1b08085a098e53943b76273aba06ed4c307a28aa9e333819d109597` |
| `experiments/exp-008-diversity/oof/X2-d8p-january.parquet` | 4027308 | `475a7c9c7a6eaac65fe3cc130cfbce474c0829295a8f2f4f236cc558908c7ad3` |
| `experiments/exp-008-diversity/oof/X2-d8p-july_loo.parquet` | 4765835 | `be6c5f98f25ac21c3fcf97b2ddb59378891d20f88b0228eab0787c66c6058001` |

## Ranking member predictions

| file | bytes | sha256 |
|---|---|---|
| `artifacts/exp-008/pred/C2-d8p-ranking.parquet` | 7739019 | `48acb7e7cba9f2135239246466a56040a0b756ae439b9983031997f0bdf3ae5c` |
| `artifacts/exp-008/pred/L0-lin-ranking.parquet` | 7739019 | `5ba75dd24ee3340e3d895d697d7d61474433fdfab605f0597dbe5569d0125782` |
| `artifacts/exp-008/pred/L2-hub-ranking.parquet` | 7739019 | `8da562fc1b6590cc9f58e5b8fffa8a940f3b3928beaac7becf7ee497f9432d83` |
| `artifacts/exp-008/pred/T-tail-ranking.parquet` | 4707341 | `d1255fafb42738d951ce46840aff603c475be861e1a30173aed2fcc0c044b4fd` |

## Submission candidate

| file | bytes | sha256 |
|---|---|---|
| `submissions/candidates/exp-008-blend3.parquet` | 4707529 | `26c40e69e0a8f3b018d2eb9b3c8d2c64e8ce7df5167d7aa194a038b997f9a231` |
