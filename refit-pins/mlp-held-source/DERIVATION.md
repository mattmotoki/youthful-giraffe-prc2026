# Held mlp007 prepare.py: exact recovery by reversal (B1 item 1)

Written 2026-10-11 about 04:00 UTC by worker gainlane. Desk answer of 10-11 (after 03:52 UTC): "reconstruct the held
mlp007 prepare.py by reversing the physical-list fix (git 4a6aefb6d; repair_physical.py 055dfbb0 records it), and
compare the SHA-256 with e46066759e98.... If it is equal, it is an exact recovery: ship it with its derivation (base
commit, reversed hunks, hash)."

## Result

| item | value |
|---|---|
| recovered file | `b1-recovery/mlp007-held/prepare.py`, 17,600 bytes, 293 lines |
| SHA-256 | `e46066759e983f99e4fedb8bc22e9292f25b3e24c7c440dffca94ffd2d5a1924` |
| pin | `experiments/exp-164-double-down/results/mlp-007/input-records/INPUTS-prepare03.json` and `INPUTS.json` `code_sha256["prepare.py"]`; `results/mlp-007/REVIEW-PREFIT.md` source snapshot table |
| match | yes (byte-exact) |
| base | git `4a6aefb6d:experiments/exp-164-double-down/code/mlp007/prepare.py`, SHA-256 `d91d9802312b7b487b0fc456f6e6affdf533ec43782a8a0d04ce537a0bed85dc`, 17,754 bytes, 294 lines |

## Reversed hunk

`reversed-hunks.diff` (`diff recovered base`). Base lines 252-253 become one line 252:

- removed: `physical_admin = {'lirf_flight_len','lirf_ades_is_adep','lirf_stand_num','lirf_stand_retired'}`
- base: `physical = [c for c in nums if (c.startswith(prefixes) or c in PX.Z_COLS or c in physical_names or c.startswith('lirf_')) and c not in physical_admin]`
- recovered: `physical = [c for c in nums if c.startswith(prefixes) or c in PX.Z_COLS or c in physical_names]`

All other bytes are the base bytes. The `physical_names` set and its comment (base lines 246-251) stay: the first
saved list (INPUTS-prepare03.json) holds `hour`, `dow`, `dom`, `doy`, `sin_h`, `cos_h`, `plan_flight_time`,
`act_flight_time`, `demand_over_served` and `no_nm`, which no prefix and no proxy column selects. The review that
asked for them (REVIEW-PREFIT finding 3) came before prepare-03.

## How it was found

`code/b1_prepare_variants.py` builds 24 variants of the two base lines: the three remaining filter terms in all 6
orders, with and without outer parentheses, and with and without the `physical_admin` line. One variant equals the
pin. The SHA-256 match is the evidence; the variant count changes nothing.

Before compaction, my stated guess also removed `physical_names` (lines 246-251). The saved first list shows that
guess was wrong; the variants keep those lines.

## Checks (Mac, `.venv/bin/python -I`, no data read)

- The recovered filter, applied to the 210 full numeric names of INPUTS-prepare03.json with `PX.Z_COLS` from
  `experiments/exp-134-proxies-in-learner/code/proxy_features.py` (SHA-256 `106450ef…`, equal to INPUTS
  `base_source_code["proxy_features.py"]`), gives 146 names, equal in order to INPUTS-prepare03.json
  `features.physical.numeric`.
- The base filter gives 178 names, equal in order to INPUTS.json `features.physical.numeric` (the list
  `repair_physical.py` wrote).
- The other mlp007 code files in git at 4a6aefb6d equal the REVIEW-PREFIT snapshot: check_inputs.py, fit.py,
  io_utils.py, supervise.py, monitor.py, launch_phase.sh. Only prepare.py changed after the snapshot.
- Line evidence: the prepare-02 traceback (`results/mlp-007/logs/mlp007-prepare-02.log`) names `main()` at line 293;
  the base has it at line 294, the recovered file at line 293.

## Use

The file imports `io_utils` from its own folder and reads remote roots (`prc-exp153-work`, `prc-exp148-work/kinput`,
`prc-exp164-work`). It is a record of the code that wrote the held mlp-007 inputs (prepare-03). The physical arm
was fitted on the repaired 178-name list, which `repair_physical.py` (055dfbb0, in git at 4a6aefb6d) wrote over this
file's 146-name list; the arrays are the same for both lists (`arrays_changed: false`).
