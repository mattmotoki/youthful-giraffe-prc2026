# Reproduce the release target and the earlier entries

The release target is `final-v-dh-r9-ld`: the final file of the challenge, one
prediction for each of the 670,790 departures of `final_submitting.parquet`
(SHA-256 `eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02`).
It is the target `final-v-r9-ld` (SHA-256
`31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759`, the model
of `v73` on the organizers' final files with the 15-seed s7 member, an L2C step
that also reads that member, the 12-fit MLP values on every member-scope
row of the zabd assembly, and an ORD-S step that also changes the ordinary rows
of long-taxi airport-days) with one more step on 14,587 EDDF departures (next
sections). The targets `final-v-dh-r9` and `final-v-dh` apply the same step to
the targets `final-v-r9` (SHA-256
`18a7bfb77bfdc5a68387cc9d96acbf3f276d441748fa7c2fd807a340e753ab82`, the first
three changes) and `final-v` (SHA-256
`a9146e9d3c057e5fbad3c7acf7f8e448efbbfdfdbe02f9e073948fc6ab34e903`, the first
two changes). The targets `final-iii-dh-r9-ld`, `final-iii-dh-r9`,
`final-iii-dh`, `final-ii-dh` and `final-dh` apply it to the targets
`final-iii-r9-ld` (SHA-256
`260b427b83f99921b26ce372bad78fbcb3718b402e7a8bd0725a914ae3da114d`),
`final-iii-r9` (SHA-256
`1c1ed67bdd7e54182bafb9a3eb08b48e6e708e981af6bb607c0016ca1f010c4b`) and
`final-iii` (SHA-256
`6de02372de5d1aea062300a88af7af51606ac2d6aee645c384b09398c9c57378`), the same
changes with the 9-seed s7 member, `final-ii` (the 9-seed s7 member only) and `final` (V, SHA-256
`121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a`, the model
of `v73` on the final files).

`v73` is our final entry of the ranking phase, candidate file SHA-256
`70236754db85e5f23fac281ebbc9c14a3a4fe33c4e6f34e7c95262eebe1659d8`. It scored
223.3071 s RMSE as upload v73. It is the target `v64` (SHA-256
`9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b`, 225.7082 s
as upload v64) with five fixed-scope steps, a level-2 correction on clock rows
and a sequence-member column on ordinary rows. `v64` is the target `zabd`
(SHA-256 `e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636`,
225.9106 s as upload v62) with one fixed stage, T11b-L.
The target is a parameter; [the entry registry](../release/ENTRIES.json)
sets `final-v-dh-r9-ld` as the default and retains final-v-r9-ld, final-v-dh-r9, final-v-r9, final-v-dh, final-v,
final-iii-dh-r9-ld, final-iii-r9-ld, final-iii-dh-r9, final-iii-r9, v64-final-r9, zabd-final-r9, final-iii-dh,
final-iii, final-ii-dh, final-ii, final-dh, final, v64-final, zabd-final, v73, v64, zabd, v40, v37, v36, v35 and
v32 as explicit targets.
[LINEAGE-ZABD.md](LINEAGE-ZABD.md) gives the code path of each member and stage, and
[DATA.md](DATA.md) lists every data source with its licence and hosting.
The part "Refit the final file from source" tells how each saved value of the final file was made, from the
official files, the open data and the source, and how to make it again.

## Release target final-v-dh-r9-ld

### What the build does

`final-v-dh-r9-ld` is the final file `youthful-giraffe_final.parquet`: the target
`final-v-r9-ld` (V) with one more step. On 14,587 EDDF departures the value is max(60, V + 0.164 (DH − V)); every other
row keeps V. V is the value of the target `final-v-r9-ld`, and DH is the value of a
reduced clock head. `final-v-dh-r9`, `final-v-dh`, `final-iii-dh-r9-ld`, `final-iii-dh-r9`, `final-iii-dh`,
`final-ii-dh` and `final-dh` are the same step with V from the targets `final-v-r9`, `final-v`, `final-iii-r9-ld`,
`final-iii-r9`, `final-iii`, `final-ii` and `final`.

- The rows: typed EDDF departures of January (1,557), February (12,374) and
  June (656) 2026 that have no clock-head value and whose own adsb.lol inputs
  are absent on an airport-day where less than 20 % of the departures have an
  observed taxi time (a flag without labels).
- DH: a LightGBM bag of three seeds on the 2025 clock rows without January and
  July, with no adsb.lol input. Its values are saved in
  `release/saved-vectors/final-entries/dh/` (`pred_DH.json`, `has_head.json`,
  `w_pair.json`); the build fits nothing.
- The weight 0.164 comes from held January and July 2025 rows made dark.
  [METHOD.md](writeup/METHOD.md) section 4.11 gives the held reads.

`release/dh_stage.py` checks the two final files and its own code by SHA-256.
It reads V from the saved values of its base entry
(`release/saved-vectors/final-entries/s15/v-r9-ld.json`, `s15/v-r9.json`, `s15/v-v.json`,
`r9-ld/v-iii.json`, `r9/v-iii.json`, `v-iii.json`, `v-ii.json` or `v.json`), which must give the bytes of that
entry. The three saved DH vectors must give the bytes of the DH part of the final build; all eight targets
use the same part. The V files of `final-v`, `final-iii`, `final-ii` and `final` are equal
on these 14,587 rows, so those four targets give the same values there; the V
files of `final-v-r9`, `final-v-r9-ld`, `final-iii-r9` and `final-iii-r9-ld` are equal on these rows and differ
from them on 14,160 of the rows. The step has no
BLAS or tree step, so the bytes are the same on every platform with IEEE
float64. SHA-256:

| target | base | SHA-256 |
|---|---|---|
| `final-v-dh-r9-ld` | `final-v-r9-ld` | `eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02` |
| `final-v-dh-r9` | `final-v-r9` | `5ce37d698943fed7a1f512aacf9500140ea7ee0809592670fe3a73ec2039fb7e` |
| `final-v-dh` | `final-v` | `833533adfc04e76dbd3f300b5c69945ce305a0111a858b5a08a67ad85f880cfa` |
| `final-iii-dh-r9-ld` | `final-iii-r9-ld` | `07d4a4e6344f5c681761a00a851d23dd9dd6bc8675861179b0de80b4fe669865` |
| `final-iii-dh-r9` | `final-iii-r9` | `4ac66707333eab1b32c43db0f412907e05d35e268d6a869065dba3fdc99b8139` |
| `final-iii-dh` | `final-iii` | `c45116502cad96dbb270467b9f2ae26fb6e06d9a50ca34ff214253609f44612d` |
| `final-ii-dh` | `final-ii` | `bf47551e420f781527901f9178d4d5c6e3f1089943c3c5c38baa4ced81c49c19` |
| `final-dh` | `final` | `2fe6c56b2705b0731be2b093f30d2763bccfb0b6ae9600855767cb0d12fef700` |

### Inputs and command

The inputs are the two final files, `final_submitting.parquet` and
`final_ranking.parquet`, in `input-root/data/raw/`.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/build_entry.py --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/final-v-dh-r9-ld-build.json
```

With no `--target` the script builds the default target, `final-v-dh-r9-ld`;
`--target final-v-dh-r9-ld` gives the same file, and `--target final-v-dh-r9`,
`--target final-v-dh`, `--target final-iii-dh-r9-ld`, `--target final-iii-dh-r9`,
`--target final-iii-dh`, `--target final-ii-dh` or `--target final-dh` gives the
other seven files.

The build takes 2 seconds and 0.3 GB of memory (measured on macOS arm64). The
record holds the checks, the number of rows that change against V and their
rms change.

## Release targets final-v-r9-ld, final-v-r9, final-v, final-iii-r9-ld, final-iii-r9, final-iii, final-ii and final

### What the build does

`final-v-r9-ld` makes V, the base of the final file: one prediction for each of
the 670,790 departures of `final_submitting.parquet` (January, February, June
and July 2026), SHA-256
`31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759`.
`final-v-r9` (SHA-256
`18a7bfb77bfdc5a68387cc9d96acbf3f276d441748fa7c2fd807a340e753ab82`), `final-v`
(SHA-256 `a9146e9d3c057e5fbad3c7acf7f8e448efbbfdfdbe02f9e073948fc6ab34e903`),
`final-iii-r9-ld` (SHA-256
`260b427b83f99921b26ce372bad78fbcb3718b402e7a8bd0725a914ae3da114d`),
`final-iii-r9` (SHA-256
`1c1ed67bdd7e54182bafb9a3eb08b48e6e708e981af6bb607c0016ca1f010c4b`), `final-iii`
(SHA-256 `6de02372de5d1aea062300a88af7af51606ac2d6aee645c384b09398c9c57378`),
`final-ii` (SHA-256
`280db2058f10a5cc3d44acdcf20aed80aae143cfadf2c9c299ddf7212c4f7afc`) and `final`
(SHA-256 `121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a`) are
the other seven V files. Each is the model of v73 on the organizers' final
files:

| target | s7 member | L2C step |
|---|---|---|
| `final-v-r9-ld` | mean of 15 fits (`release/saved-vectors/s7c15/`) | `w2/l2c_candidate_a.py`: also the s7 deviation from c5 as an input, 9 LightGBM seeds |
| `final-v-r9` | mean of 15 fits (`release/saved-vectors/s7c15/`) | `w2/l2c_candidate_a.py`: also the s7 deviation from c5 as an input, 9 LightGBM seeds |
| `final-v` | mean of 15 fits (`release/saved-vectors/s7c15/`) | `w2/l2c_candidate_a.py`: also the s7 deviation from c5 as an input, 9 LightGBM seeds |
| `final-iii-r9-ld` | mean of 9 fits (`release/saved-vectors/s7c9/`) | `w2/l2c_candidate_a.py`: also the s7 deviation from c5 as an input, 9 LightGBM seeds |
| `final-iii-r9` | mean of 9 fits (`release/saved-vectors/s7c9/`) | `w2/l2c_candidate_a.py`: also the s7 deviation from c5 as an input, 9 LightGBM seeds |
| `final-iii` | mean of 9 fits (`release/saved-vectors/s7c9/`) | `w2/l2c_candidate_a.py`: also the s7 deviation from c5 as an input, 9 LightGBM seeds |
| `final-ii` | mean of 9 fits (`release/saved-vectors/s7c9/`) | `w2/l2c.py` of v73, 3 LightGBM seeds |
| `final` | mean of 3 fits (`release/saved-vectors/s7c/`) | `w2/l2c.py` of v73, 3 LightGBM seeds |

The chain of each has three entries, each with `"mode": "final"`:

- `zabd-final` (SHA-256 `30d26797a964e6d7119adaa713cd865f5b6f25f3877c79cbd3c5096f0b5655f8`):
  the zabd assembly on the final pack `release/ZABD-PACK-final.json`, with the
  member values of the final build under `release/saved-vectors/final/`;
- `v64-final` (SHA-256 `8cfb8efb552eb0ee545a3d7ccf476c711157c44757287496b59099b5e70f4af1`):
  the T11b-L stage on zabd-final (7,497 rows change);
- `final-v`, `final-iii`, `final-ii` or `final`: the v73 steps on v64-final.

The chain of `final-v-r9` and `final-iii-r9` (R9) and of `final-v-r9-ld` and `final-iii-r9-ld` (LD) has its own two
first entries:

- `zabd-final-r9` (SHA-256 `d26dadd03005d6847daf9ed2e3beb42908b350811def494dcb07d56a63e8f203`):
  the zabd assembly on the R9 pack `release/ZABD-PACK-final-r9.json`. It is the
  final pack with one other column: the MLP column of the fold and 2026
  primary frames holds the 12-fit MLP values on every member-scope row
  (`release/saved-vectors/final-r9/`; the final pack has them on clock rows
  only). The assembly refits every blend weight;
- `v64-final-r9` (SHA-256 `3884f9058606bb6f0d1b5ec5813cc56014e30293ab4ffb31b703e993b3c70112`):
  the T11b-L stage on zabd-final-r9 (7,497 rows change);
- `final-iii-r9`: the steps of `final-iii` on v64-final-r9;
- `final-iii-r9-ld`: the steps of `final-iii-r9` with the ORD-S step of LD
  (difference 5 below). Its apply masks also take the ordinary member-scope rows
  of long-taxi airport-days, and its fit still leaves out these days. 6,753 rows change
  against `final-iii-r9`;
- `final-v-r9` and `final-v-r9-ld`: the steps of `final-iii-r9` and `final-iii-r9-ld` with the 15-seed s7 member.
  6,752 rows of `final-v-r9-ld` change against `final-v-r9`.

`scripts/build_entry.py` sets `PRC_RANKING_MODE=final` from the entry before it
imports a release module. The steps are those of the target v73 (next section)
with three differences (four for `final-v`, `final-v-r9`, `final-iii` and `final-iii-r9`, five for
`final-v-r9-ld` and `final-iii-r9-ld`):

1. It checks 16 official files: the 14 of v73 and the two final files.
2. After `iso/dump_chain.py` it runs `chain_ids.py`, which writes the row order
   of the chain, checked against the official files.
3. ORD-S reads the s7 member values from the member files
   `work/w2/s7_{fold,ranking}.parquet`, which the build writes from the saved
   member values (`release/saved-vectors/s7c9/final-fold.json` and
   `release/saved-vectors/s7c9/final-ranking.json`; for `final-v`, `final-v-r9` and `final-v-r9-ld` the two
   files of the same names in `release/saved-vectors/s7c15/`, for `final` those in `release/saved-vectors/s7c/`)
   and checks by SHA-256 against
   the member files of the final build.
4. `final-v`, `final-v-r9`, `final-v-r9-ld`, `final-iii`, `final-iii-r9` and `final-iii-r9-ld`: the run folder
   gets `w2/l2c_candidate_a.py` as
   `w2/l2c.py` (its SHA-256 is registered in the entry). After `chain_ids.py`
   the build writes the float32 member values as
   `work/w2/s7c-l2c.npz` and checks each array by the SHA-256 of its float32
   bytes; L2C runs with `--s7 work/w2/s7c-l2c.npz`.
5. `final-v-r9-ld` and `final-iii-r9-ld` only: the run folder gets `w2/ords_ld.py` as
   `w2/ords.py` (its SHA-256 is registered in the entry), and ORD-S runs with
   the flag `--long-days` (`member.ords_flags`).

February takes the January scope of each fitted step and June takes the July
scope. The fitted steps refit from the 2025 labels as in v73; no member is
refit. No entry of the final chain can use the published-mean-shift recipe of
v37 and v40; the build stops if one does.

The final bytes come from x86-64 Linux with the platform of target v73 and the
same thread counts (2 BLAS threads, 8 LightGBM threads in L2C). On another
platform the level-2 least squares and the L2C trees can differ. The build then
stops with a FAIL verdict, and its record gives the difference to the saved
V values (`release/saved-vectors/final-entries/s15/v-r9-ld.json`, `s15/v-r9.json`, `s15/v-v.json`,
`r9-ld/v-iii.json`, `r9/v-iii.json`, `v-iii.json`, `v-ii.json` or `v.json`). On macOS arm64 with 2 LightGBM threads, `final-iii` gave a file at
rms 0.69 s (largest 16.5 s) from the saved `final-iii` values and at rms 1.99 s
from the `final-ii` values.

### Inputs and command

The inputs are the 16 official files in this layout:

```text
input-root/data/raw/final_submitting.parquet
input-root/data/raw/final_ranking.parquet
input-root/data/raw/submitting.parquet
input-root/data/raw/ranking.parquet
input-root/data/raw/training_2025-01-01_2025-02-01.parquet
...
input-root/data/raw/training_2025-12-01_2026-01-01.parquet
```

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target final-iii --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/final-iii-build.json
```

`--target final-v-r9-ld`, `--target final-v-r9`, `--target final-v`, `--target final-iii-r9-ld`,
`--target final-iii-r9`, `--target final-ii` or `--target final` builds the other seven V files. On x86-64 Linux, on 16 logical
CPUs that other jobs also used, `final-v-r9-ld` took 8 min 59 s, `final-v-r9` 9 min 7 s, `final-v` 8 min 29 s,
`final-iii-r9-ld` 8 min 33 s,
`final-iii-r9` 8 min 37 s, `final-iii` 8 min 2 s, `final-ii` 5 min
13 s and `final` 5 min 10 s, with a peak resident memory of 2.76, 2.76, 2.76, 2.76, 2.59, 2.68,
2.67 and 2.87 GB. The output folder
gets the file `<target>-replay.parquet` and the run folder `v73-run`. The record holds the platform, the step commands, the
chain facts, the member checks, the intermediate hashes, the checks and the
difference to the saved V values.

## Release target v73

### What the build does

`v73` runs the code of `iso/` and `w2/` (`release/v73_stage.py`), in the order
of `w2/rebuild.sh`:

1. It checks the 14 official files by SHA-256: the submitting file, the ranking
   file and the twelve training files of 2025.
2. It builds the v64 chain reference: the saved zabd values must give the zabd
   bytes, and the T11b-L stage on them the v64 bytes. It writes the reference to
   `v73-run/work/chain_ref_ranking.parquet`.
3. It makes the folder `v73-run` in your output folder, with checked copies of
   the Python step code, links to `release/` and to your official files, and the
   s7 member values of `release/saved-vectors/s7c/` written back as the float32
   arrays of `v73-run/w2/inputs/s7c.npz`.
4. It runs `iso/dump_chain.py` (the v64 chain with the release code: it refits
   the zabd level-2 weights and the C95 and C98 statistics, and its result is
   compared with the reference), `iso/features.py` twice, `iso/candidate.py` (the c5
   steps), `w2/base.py`, `w2/l2c.py` (L2C) and `w2/ords.py` (ORD-S), with
   2 BLAS threads and 8 LightGBM threads. Each step has a log in
   `v73-run/logs/`.

The chain replay refits the zabd level-2 weights and the C95 and C98
statistics, and the steps refit the LONG weights, the NOAC slope, the ROLL
logistic, the L2C trees and the ORD-S coefficients, all from the 2025 labels.
The record lists the step fits under `fitted`. No member is refit; the s7
member values are saved, because its GPU fits do not repeat bit for bit.
[LINEAGE-ZABD.md](LINEAGE-ZABD.md#from-v64-to-v73) describes each step and its
held read.

The v73 bytes come from x86-64 Linux with Python 3.12, numpy 2.4.6, pandas 2.3.3,
pyarrow 24.0.0, LightGBM 4.7.0, scikit-learn 1.9.0 and scipy 1.17.1
(`w2/requirements-remote.txt`). On another platform the L2C trees can differ.
The build then stops with a FAIL verdict, and its record gives the hashes of
c5, L2C and the final file and their difference to the saved v73 values
(`release/saved-vectors/v73-final.json`). On macOS arm64 c5 is exact
(`63850254…`) and the final file differs by 1.26 s rms. The LightGBM thread
count is part of the recipe: with 4 threads the final file is `70715d41…`.

### Inputs and command

The inputs are the 14 official files in this layout:

```text
input-root/data/raw/submitting.parquet
input-root/data/raw/ranking.parquet
input-root/data/raw/training_2025-01-01_2025-02-01.parquet
...
input-root/data/raw/training_2025-12-01_2026-01-01.parquet
```

`w2/inputs/raw_sha256.txt` lists the 14 files with their SHA-256 values.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target v73 --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/v73-build.json
```

The build takes about 2 minutes and 2.5 GB of memory on an Apple M-series
laptop (measured: 128 s and 2.45 GB peak resident memory), and 4 to 5 minutes
and 2 GB on x86-64 Linux with 8 logical CPUs (measured: 4 min 48 s and 1.95 GB;
a second run 4 min 18 s and 1.98 GB). The output folder
gets the file v73-replay.parquet and the run folder `v73-run` with the
intermediate files c5 (`outputs/iso-c5.parquet`) and L2C
(`outputs/w2-l2c.parquet`). The record holds the platform, the step commands,
the chain facts, the intermediate hashes, the checks and the difference to the
saved v73 values. In `intermediate`, each file is compared with the hash that
is registered for the platform of the run (on macOS arm64, the macOS hashes, so
`equal` can be true for each file); the verdict compares the final file with
the v73 bytes.

`w2/rebuild.sh` and `iso/rebuild.sh` are the scripts of the original build;
the header of `w2/rebuild.sh` gives the GPU commands that made the s7 member.
They read `w2/inputs/s7c.npz` and `w2/inputs/chain_ref_ranking.parquet`, which
the export does not hold. The build above runs the steps of the two scripts
itself and leaves the scripts out of the run folder. It writes the member
arrays to `v73-run/w2/inputs/s7c.npz` and gives the reference
`v73-run/work/chain_ref_ranking.parquet` to `iso/dump_chain.py` as `CHAIN_REF`.

## Release target v64

### What the replay does

`v64` first replays `zabd` exactly as in the next section and checks its bytes.
Then it applies the T11b-L stage of `release/t11b_stage.py`, which fits nothing
and reads no label. On the 4,902 rows of 2026 with an aircraft type, a row class
other than noac and no NM flight join, the value moves a quarter of the way
toward the dark18 2026 value, inside the bounds and with the 60 s floor. Rows of
a long-taxi airport-day keep the zabd value; the stage finds these days in your
ranking file and checks their count by airport (15 days). The stage reads four
frames of `release/ZABD-PACK-ranking.json` and checks each by its content digest. It
checks that 4,078 rows change, then compares the file with the recorded SHA-256
value. [LINEAGE-ZABD.md](LINEAGE-ZABD.md#from-zabd-to-v64-the-t11b-l-stage)
gives the rule in full and its held read.

### Inputs and command

The inputs are the four official files of zabd (next section).

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target v64 --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/v64-replay.json
```

The run takes about 9 seconds and 2 GB of memory on an Apple M-series laptop
(measured: 9.3 s and 1.93 GB peak resident memory with two threads). The output
folder gets the file v64-replay.parquet, and zabd in the folder v64-components.
The record holds the record of zabd, the long-taxi days, the row counts and the
stage checks.

## Release target zabd

### What the replay does

`zabd` is the final assembly of experiment 169. The replay reruns that assembly
with its original code (`experiments/exp-169-stack-plan/code/`), in the order of
its build script `build035.py`: the level-2 fits of the parent (the fallback pair
and the five-column primary blend), the parent stages and rules, then the
six-column primary blend with the dark-day member, the C71, C95 and C98 residual
rules (C95 and C98 statistics refit, C71 fixed), the fixed tail and one bound
projection. Then it runs the tail once more with the C65 row-order offset set
to zero; that gives the candidate. The level-2 fits use the 2025
labels of January and July from your official training files. The replay checks
three files byte for byte: the parent (SHA-256
`9436b85b0c5b2c83dd4b6b1be16076166337e7c274a3d4e6a59481843fd9cf9c`, the same
assembly without the dark-day member), the assembly with C65 (SHA-256
`0eec38e86fadd6a658dabd1e877bd7603301429b6523d375421466efd8cb7779`) and the candidate.

The replay does not fit any member. The member predictions, the earlier stage
values and the rule records are saved in `release/saved-vectors/zabd/` as
compressed arrays, without organizer movement IDs and without labels. Their row
order comes from your official files: the 2025 rows are the departures (`PHASE_mvt`
equal to `DEP`) of the January 2025 training file, then of the July 2025 file, in
file order; the 2026 rows follow the submission template. Columns that are
copies of official columns are read from your files. The index
`release/ZABD-PACK-ranking.json` gives the kind of each column, the hash of each saved
file and a content digest of each input frame. The replay rebuilds each frame and
stops if one differs from the original input of the build.

### Inputs

Get the four official files from the challenge data page and place them so:

```text
input-root/data/raw/submitting.parquet
input-root/data/raw/ranking.parquet
input-root/data/raw/training_2025-01-01_2025-02-01.parquet
input-root/data/raw/training_2025-07-01_2025-08-01.parquet
```

The replay checks the SHA-256 of each file against `release/ZABD-PACK-ranking.json`.

### Command

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target zabd --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/zabd-replay.json
```

The run takes about 7 seconds and 2 GB of memory on an Apple M-series laptop
(measured: 7.4 s and 1.94 GB peak resident memory with two threads).
The record gives both hashes, the fitted level-2 weights and the rule statistics.
The original build ran on macOS arm64 with Python 3.11, numpy 2.4.6 on Apple
Accelerate, pandas 2.3.3 and pyarrow 24.0.0; one and two BLAS threads give the
same bytes there. Another BLAS library can change the last bits of the
least-squares fits. Then the bytes differ, and the record field
`max_abs_difference_to_saved_entry_s` gives the largest difference to the saved
candidate values.

### A fresh fit

A fresh fit starts from the members. [LINEAGE-ZABD.md](LINEAGE-ZABD.md) names,
for each member and stage, the fitting code, the training months, the seeds and
the data sources. No fresh fit has reproduced the members of `zabd`.

# Refit the final file from source

**What this part does.** The release replays the final file from saved values (`scripts/build_entry.py`; section
"Release target final-v-dh-r9-ld" above). This part tells how each saved value was made, so that a reader can make it
again from the official files, the open data and the source. S0 gives the setup: the root variables, the refit source
tree, the packages and the hardware. Sections S1 to S15 follow the chain of the final mode (the four 2026 months). S16
covers the held side (January and July 2025), which the final build reads as saved values.

**Two kinds of result.**

- *Exact.* A step that is deterministic on the same platform gives the same bytes. The section says so when a record
  shows a byte check.
- *Recipe refit.* A GPU fit, a fit at another thread count or a fit on another BLAS gives other bits. Such a step
  gives a new vector of the same recipe. Its hash differs; compare its quality on the held rows. Do not compare its
  hash with the saved one.

**The target.** The default entry is `final-v-dh-r9-ld`: `youthful-giraffe_final.parquet`, SHA-256
`eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02`, 670,790 rows (`release/ENTRIES.json`, key
`default_target`; section "Release target final-v-dh-r9-ld" above). It is the V file of `final-v-r9-ld`
(`31b09989…`, the LD chain with the 15-seed s7 member) with the DH step. The 9-seed entries stay in the release:
`final-iii-r9-ld` (`260b427b…`) and `final-iii-dh-r9-ld` (`07d4a4e6…`). S13 gives both s7 seed sets.

Chain of saved entries: `zabd-final-r9` → `v64-final-r9` → `final-v-r9-ld` (V, `31b09989…`) →
`final-v-dh-r9-ld` (`eea300c0…`) (the `base` fields of `release/ENTRIES.json`).

**Citations.** Each fact names its source.

- A path in code format that starts at the repository root, for example `release/ENTRIES.json`, is in this release.
  A path that starts with `refit-src/` or `refit-pins/<unit>/files/` is a refit source (S0.2); in the refit tree it
  has the same path without that prefix.
- An experiment name with a path in code format, for example exp-189 `results/parts/route.json`, names a file in the
  folder of that experiment (for exp-189: experiments/exp-189-final-route/). When the release ships it (in the export or in
  `refit-src/`), the refit tree holds it. Otherwise it is a record of our runs that the release does not ship, and the
  text gives the value that the step needs. A file name with `:<n>` (for example `route_parts.sh:66-117`) is a line
  or a line range of that file.
- A path in plain text names a record of our work repository that the release does not ship.
- A hex word of seven or nine characters, for example ad7cce060, is a commit of our work repository.
  `release/REFIT-SOURCES.json` gives the commits that ran each refit source.
- "Our run" is the run that made the saved values of the release. A command is the command of that run, with its
  private path prefixes replaced by a root variable (S0.1).

## S0 Setup

### S0.1 Roots

Each private path prefix of our runs is one root variable in the refit sources. A shipped file reads the variable,
with a `/path/to/...` default where our run had our folder. Set each root to your folder before a step; with our
folders, a step reads the same bytes as our run. `release/REFIT-SOURCES.json` key `roots` lists the roots, and
Appendix A tells what each one held in our runs.

| root | default | holds |
|---|---|---|
| `${PRC_FINAL_ROOT}` | `/path/to/final` | the final-mode run folders: `finaldata` (`F`), `finalroute` (`FR`), `finalclock`, `finalcore`, `finallirf` (`FL`), `finalv`, `finalplan`, `gainlane` and their `parts/` |
| `${PRC_TOOLS}` | `/path/to/tools` | the site guard tools (Appendix C); no step makes bytes with them |
| `${PRC_LOG_ROOT}` | `/path/to/logs` | run logs |
| `${PRC_MODEL_ROOT}` | `/path/to/models` | saved model files: g fits, sequence fits, route level 2 |
| `${PRC_OPDI_ROOT}` | `/path/to/prc-exp155-opdi` | the exp-155 OPDI work folder: code, event windows, `r2/`, `work/airports.csv` |
| `${PRC_OPDI_RAW}` | `/path/to/opdi-raw` | the OPDI raw downloads (cached flight events and lists) |
| `${PRC_COMP_ROOT}` | `/path/to/competitions` | the parent of the checkout and of the 2025-side work trees: `prc-exp148-work`, `prc-exp148-src`, `prc-exp150-work`, `prc-exp153-work`, `prc-exp159-work`, `prc-exp164-work`, `prc-exp181-work` |
| `${PRC_ISO5_ROOT}` | `/path/to/prc-isolated-5` | the run tree of the isolated worker (s7 seed fits 4-9, S13) |
| `${PRC_SOURCE_EXPORT}` | `/path/to/source-export` | the release export that the assembly and R9 steps read (S11, S12) |
| `${PRC_HOST}` | `remote` | the ssh host name in record commands |
| `${PRC_KAGGLE_KERNEL}` | `<kaggle-user>/prc-exp189-learners-final` | the route L1 learner kernel (S5) |

Two more variables that the scripts read already (they are not root variables):

| variable | holds |
|---|---|
| `${PRC_SOURCE_ROOT}` | the refit source tree (S0.2): the export source plus the `refit-src/` overlay, with a `refit-pins/` unit for the steps that need one. A script that reads `PRC_SOURCE_ROOT` gets this tree. |
| `${PRC_DATA_DIR}` | the 16 official files: the twelve 2025 training months, `ranking.parquet`, `submitting.parquet`, `final_ranking.parquet`, `final_submitting.parquet` |

Environment inputs:

- `PRC_R2_NO_FEATURES`: the boundary-row JSON that `scripts/exp148_missing_features.py` writes. The steps of exp-148
  `fold_serve.py` and exp-194 `r4mk_dark.py` and `r4mk_iso.py` stop without it: they call `r2_no_features()` of
  exp-148 `stacker.py`. It replaces the row-ID constant `R2_NO_FEATURES` of our run (rule `r2-ids` in the file
  records of `release/REFIT-SOURCES.json`).
- `PRC_KAGGLE_USER`: the reader's Kaggle account for the harvest kernels and the private input datasets (S1, S5).
  Our runs set our account; the shipped files carry `<kaggle-user>`.

External inputs (`release/REFIT-SOURCES.json` key `external_inputs`): files that the release does not ship.

- OurAirports `airports.csv`, at `${PRC_OPDI_ROOT}/work/airports.csv`: download it from
  `https://davidmegginson.github.io/ourairports-data/airports.csv` (Public Domain). Our run read SHA-256
  `5cb29a038a92dbdb94d4fcdd90ce116b2b47394f456d9c43e373d683ef0c4eff` (12,733,703 bytes). The site writes a new file each
  day, so a download of another day gives other bytes; the steps read only the rows of the ten airports (ICAO ident,
  reference point, elevation). Readers: exp-155 `gate_check.py`, `match_full.py`, exp-185 `code/opdi/match_final.py`
  (S2).
- The 12 g saved fits (S7), the exp-170 views, means and fits (S8), the three held frames of `np_r1_weight_held.py`
  (S11), and the withheld module prc/\_private\_clean\_score\_rows.py (S1): see each section.
- `<external>` in the commands of S1 and S3 is a folder that the reader chooses for the other downloads (OurAirports
  `runways.csv` and `airports.csv` of the harvest boxes, the Gateway cache, ERA5, the exp-045 day summaries). No script
  reads it as a variable.

`PRC_RANKING_MODE=final` selects the final mode (four 2026 months, `final_ranking.parquet`). Without it, a producer
runs the old ranking mode (January and July 2026, `ranking.parquet`). Keep the old files: the held-input and encoder
contracts read them.

### S0.2 The refit source tree

Our final runs did not run from one tree. Each producer ran from a copy of the release export with newer files laid
over it (exp-191, exp-192, exp-194), or from a snapshot of our main branch with overlays (exp-185 to exp-190). The
refit tree puts these in one place:

1. A copy of the released source: each file of `release/MANIFEST.tsv` outside `refit-src/` and `refit-pins/`.
2. `refit-src/` laid over it at the same paths. It holds each producer file that the export lacks (add) or holds in
   other bytes (replace: a three-way merge of the export file, its base and the version that ran, so the release's own
   edits stay).
3. For a step that the table below names, one unit `refit-pins/<unit>/files/` laid over that. A unit holds the
   version that one step ran at a path where the final chain ran other bytes (S16.4).

Build it with `python release/make_refit_tree.py <out dir> [--unit <unit>]`. The script checks each file against
`release/MANIFEST.tsv` and each overlay file against `release/REFIT-SOURCES.json`, and writes `REFIT-TREE.sha256` and
`REFIT-TREE.json` in the out folder. Set `${PRC_SOURCE_ROOT}` to that folder. `release/REFIT-SOURCES.json` gives, for
each shipped file, the commits that ran it, its source, the SHA-256 that ran, the SHA-256 shipped and the rules that
changed it; its key `out` lists the files that stay out, each with its reason.

**Run tree of each step.** Our runs used snapshots: `${PRC_FINAL_ROOT}/<run>/src-<commit>`, `repo-<commit>`,
`snap-<commit>` and `code-<commit>` on the x86 host, the route trees `FR/src-b165ab4ee` and `FR/src-b165ab4ee-f1b`,
and the 2025-side trees `${PRC_COMP_ROOT}/prc-exp153-work/source`, `prc-exp159-work/source` and `prc-exp148-src`.
Each one is a copy of git files at one commit. In the refit, each step runs in the refit tree, with the unit below
laid over it where the table names one:

| step | tree |
|---|---|
| S1 to S8, S10 to S13, S15; S9 except the next row; S14 except the next row | the refit tree |
| S9: the mlp-008 final inputs (`receipts/final-INPUTS.json` `230f9eb8`) | refit tree + unit `mlp008-inputs` |
| S14: the dh fallback file `2fe6c56b` (not the default) | refit tree + unit `dh-2fe6c56b` |
| S16: the held mlp-007 inputs (prepare-03) | refit tree + unit `mlp-held-source` (it replaces our `${PRC_COMP_ROOT}/prc-exp153-work/source`) |
| S16.2: the R4 fold refits of F; the R4mks fold refits of R4mk; the first fold bundle; the R4mk swap | refit tree + unit `held-r4f`; `held-r4mks`; `held-bundle`; `held-swap` |

The interpreter of `prc-exp148-src` (`.venv`, Python 3.10.12) is the Python 3.10 environment of S0.3.

The file record of each shipped file names the rules that changed it (field `rules`) and their cause (field
`reason`): root variables for private path prefixes, the R2 boundary-row file, the removal of challenge row IDs,
decision labels and account names from records and comments, the release rewrite of the same file, a three-way
merge with the export version (`merge`), an export copy that holds the release rewrite of the same bytes
(`export-copy`) and SHA-256 citations of changed files set to their shipped bytes (`repin`). With the roots set to
our folders, a step reads the bytes that our run read.

### S0.3 Python and packages

| use | versions | source |
|---|---|---|
| replay of the final file (x86 build record) | Python 3.12.13, NumPy 2.4.6, pandas 2.3.3, pyarrow 24.0.0 | exp-193 build record `9970755a2f70-final-iii-dh-r9-ld-20261010T154328Z/record.json` lines 27-32; all 17 targets replayed on the same host at 91bccceef, the release with these refit sources (exp-193 `results/release-x86-91bccceefe4d/`) |
| assembly, R9, c5, L2C, ORD-S, the V chain and the DH fit (S12-S14; exact bytes need x86-64 Linux) | numpy 2.4.6, pandas 2.3.3, pyarrow 24.0.0, lightgbm 4.7.0, scipy 1.17.1, scikit-learn 1.9.0, polars 1.44.1; Python 3.12 on the x86 host | `w2/requirements-remote.txt`; `docs/LINEAGE-ZABD.md` Gaps item 9 |
| route members and level 2 (S5, S6), clock head (S10), 2025 OPDI first table (S2) | Python 3.10.12, LightGBM 4.6.0, CatBoost 1.2.10, numpy 2.2.6, pandas 2.3.3 (x86 host: `prc-exp148-src` venv and the system `python3`) | exp-189 `SPEC.md:88-89`; M190:2; exp-155 `r2/REPORT.md:46` |
| member predicts (S7-S9) and their GPU fits | `/usr/bin/python3` (3.10), torch 2.10.0+cu128, CUDA 12.8 | the `*-PREDICT.json` records (S7-S9); exp-191 `REPORT.md:22-28` |
| route kernel L, Lp (S5) | Python 3.12.13, lightgbm 4.6.0, numpy 2.0.2, pandas 2.3.3; image `gcr.io/kaggle-images/python@sha256:dafd4ce5…` | exp-189 `kaggle/learners-final-v2/out-v2/L-final.json` |
| all other steps | the ranges of `requirements.txt` (catboost, lightgbm, xgboost, scikit-learn, numpy, pandas, polars, pyarrow, scipy, duckdb; cdsapi, netCDF4, xarray for ERA5); torch for the neural members (versions per step in S7-S9, S13) | `requirements.txt` |
| Kaggle client (route L1 kernel, adsb.lol harvest kernels) | installed outside the virtual environment | `requirements.txt` (end note) |

### S0.4 Hardware

| host | facts | used by |
|---|---|---|
| x86 compute host | AMD Threadripper 3990X, 64 cores / 128 threads, 251 GB RAM, 2× NVIDIA TITAN RTX (24 GB), Ubuntu 22.04 | all final-mode producers (S1-S15), the held-side fits of S16, the x86 replay |
| Mac (Apple M-series) | numpy with Accelerate; the release replay of zabd is exact there at two threads | ranking-mode replays and checks (`docs/LINEAGE-ZABD.md` "What the replay reproduces") |
| Kaggle | CPU kernels | the route L1 learner kernel (S5) and the adsb.lol harvest (S1) |

Source of the x86 host facts: a read of the host on 2026-10-11 (GPUs, CPU, 263,769,584 kB memory, Ubuntu 22.04.5,
kernel 6.8.0-138-generic).

Each section gives the threads and the wall time of our run. A different thread count can change the bits of a
LightGBM or BLAS step (S5, S10 give measured cases).

### S0.5 The chain

| step | makes | final part or saved value (pack provenance) |
|---|---|---|
| S1 data | class tables, queue blocks, caches, packages, harvest | inputs only |
| S2 OPDI | 2025 and final OPDI first-event tables | inputs only |
| S3 stands | old and final stand tables | inputs only |
| S4 route features | r4c features and proxies | inputs only |
| S5 route L1 | eight level-1 route members (T-tailv4, C2-cat, L2-hub8725, D1-winter, F1-fresh, N1-nmtail, L, Lp) | member parts |
| S6 route level 2 | R4mks `pred_Sb` = `route` = `F` = `np_base` | `finalroute/parts/route.parquet` (232d2670…), `F.parquet` (66829b3a…) |
| S7 g | AB bag of 12 runway-sequence transformers | `finalcore/parts/g.parquet` (8758a971…) |
| S8 seq | sequence021, sequence039, row6, sequence011, row011, dark18 | `finalcore/parts/<member>.parquet` (c21db6dc…, 636ab22a…, 56c60492…, 33652831…, 399f6987…, 86c049d0…) |
| S9 mlp | mlp008 and mix12 | `finalcore/parts/mlp.parquet` (e48a88c6…) |
| S10 head | HB3u head | `finalclock/parts/head.parquet` (a0f28193…) |
| S11 rule frames | bundle, rule frame, exploit frame of the final mode | `finallirf/parts/frames_bundle-npee84b9f3-a444.parquet` (47d737e1…), `frames_rule_frame-npee84b9f3.parquet` (f225b96c…), `frames_exploit_frame-npee84b9f3.parquet` (886db355…) |
| S12 assembly and R9 | zabd-final-r9 (pack 29c0ace1 with R9) | entry `zabd-final-r9` |
| S13 V | T11b-L, c5, s7, L2C, ORD-S with LD | entries `v64-final-r9`, `final-v-r9-ld` (V 31b09989…, 15 seeds); `final-iii-r9-ld` (260b427b…, 9 seeds) |
| S14 DH | DH values, scope flags, weights, dh stage | entry `final-v-dh-r9-ld` (eea300c0…, the default); `final-iii-dh-r9-ld` (07d4a4e6…) |
| S15 entry registration | entry recipes and saved-vector files | `release/ENTRIES.json`, `release/saved-vectors/` |
| S16 held side | the fold frames of the pack (January and July 2025) | 46 "pack saved" columns, 4 rule columns, 1 R9 column |

The part paths above are relative to `${PRC_FINAL_ROOT}`; the hash prefixes and the pack provenance are
`release/ZABD-PACK-final-r9.json` `final_build.provenance`.

## Short names used in S1 to S16

- `F`: `${PRC_FINAL_ROOT}/finaldata`. `FR`: `${PRC_FINAL_ROOT}/finalroute`. `FL`: `${PRC_FINAL_ROOT}/finallirf`.
- `E185` before a path: the experiment exp-185 (S7 to S10 and S12 to S15 define more names of this form).
- `MAN185:<n>`: line n of `refit-src/experiments/exp-185-final-data/MANIFEST.tsv`, the run manifest of exp-185 (one
  row per file: host, path, SHA-256, bytes, role, group, command, check).
- `RULES:<n>`: line n of exp-185 `code/manifest_rules.tsv`, the command rules of that manifest (a record; the command
  column of the MAN185 rows of each group holds the command).
- `M186`, `M187`, `M189`, `M190`, `M191`, `M192`, `M194`: the run manifest `MANIFEST.tsv` of that experiment, in the
  same form as MAN185 (records of our runs). `M189:14` is its line 14.
- `T193:<n>`: line n of `refit-src/experiments/exp-193-final-build/targets.json`, the rebuild targets of exp-193 (one
  step per part, with its command and its expected SHA-256). `R193:<n>`: line n of exp-193 REPORT.md (a record).
- `export` (column of the source tables): the state of the file in the release export against the version that ran:
  `y` equal bytes, `v` other bytes, `n` absent. For a `v` or `n` file, `refit-src/` or a `refit-pins/` unit holds the
  version that ran, or `release/REFIT-SOURCES.json` key `out` gives the reason that it stays out; Appendix B gives
  the files that ran in two or more versions.
- A hash with `…`, or with 8 or 16 hex digits, is a prefix; the cited source holds the full value.
- `python`: the interpreter of S0.3. Our runs used the checkout `.venv` on the Mac, `/usr/bin/python3` or
  `python3` from `PATH` on the x86 host, and `${PRC_COMP_ROOT}/prc-exp148-src/.venv/bin/python` for some clock and
  OPDI steps. Each section names the one that ran.

## S1 Data: official files, harvest, class, queue blocks, caches, packages

- **Purpose.** Get the 16 official files. Build the data inputs of the 2025 months and the four final months: the
  adsb.lol harvest and its day summaries, the airport boxes, the class tables, the exp-047 queue blocks, the exp-018
  caches, the exp-130 package, the exp-127 frame, keys and coverage, the exp-134 proxy inputs and the exp-146 frozen
  rows.

- **Source files.** Commit that ran; SHA-256 prefix; export state.

| file | commit | SHA-256 | export |
|---|---|---|---|
| `refit-src/experiments/exp-185-final-data/code/run_month.sh` | ad7cce060 | `0aeab6fc9f0ae853` | n |
| `refit-src/experiments/exp-185-final-data/code/class_final.py` | ad7cce060 | `0ed9d1c39d27ba18` | n |
| `refit-src/experiments/exp-185-final-data/code/exp018_cache.py` | ad7cce060 | `a9eb68444986d5a9` | n |
| `refit-src/experiments/exp-185-final-data/code/exp130_months.py` | ad7cce060 | `93bed195715074c8` | n |
| `refit-src/experiments/exp-185-final-data/code/exp130_2025.py` | 9e3b45039 (the only version) | `c252642012fc92f8` | n |
| `refit-src/experiments/exp-185-final-data/code/exp130_final_package.py` | 3e042edc2; b5aab4dde | `3a65690f832f9072` | n |
| `refit-src/experiments/exp-185-final-data/code/assemble_harvest.py` | 18e5f2400 | `49b3da9a581f7bf3` | n |
| exp-185 `code/land_harvest.sh`, `census_final.py` | ad7cce060; 84605a82b | `817ae136f8f13f27`, `631674cb9c5cf49d` | n |
| exp-185 `kernels/surface-2026-{01-check,02,06}/prc-harvest.py` | 18e5f2400 | `bedd1ec08d2b339e`, `1a52fcdd3c7981fc`, `2338bf1e3e179576` | n |
| `refit-src/experiments/exp-185-final-data/kernels/coverage-final/rederive_clock.py` | f95716ca3 | `ae6a1a74efc7ec25` | n |
| exp-045 `code/push_kernel.sh`, `assemble_kernel.py`, `rule_source.py` | ad7cce060 | `0d78bf375d772b3b`, `e285fc60e759990c`, `e572873b9b9094b8` | y |
| exp-045 `code/days.py` | 774c1628d; 84605a82b | `5607e96a9680da9a` | v |
| exp-045 `code/boxes.py` | 25e773127 (the only commit) | `f35cda5667a2ed50` (computed from git) | y |
| exp-045 `manifest.json` (index of 16 months) | ad7cce060 | `1a7f991a7c926fb2` | n (the export holds the older index `dee46269`; the newer one lists our Kaggle kernels and stays out) |
| exp-046 `code/serve.py`, `arms.py`, `block.py`, `chainfold.py` | ad7cce060 | `e6963242dad519ec`, `2d75bb9ba53f47df`, `122c7d84571b53e0`, `97099cac55f73f6c` | v, v, y, y |
| exp-047 `code/emit.py` | ad7cce060 (same bytes as 0d36f766b) | `5ae9d9a2358191e0` | v |
| exp-018 `code/dataset.py`, `features.py`, `model.py`, `window_features.py` | ad7cce060 | `4a1c3d6328d27824`, `ae591d7b7d84c7ad`, `5ed75a5151413bbd`, `43fca5d5d8815163` | v, y, y, y |
| exp-130 `code/prepare.py`, `fit.py`; `census.csv` | ad7cce060 | `9bb28bd7acd2623c`, `df9ae2c5be4f0a4b`; `25c0d43da482ac70` | y |
| exp-127 `prepare_ranking_frame.py`, `rederive/prepare_clock_keys.py` | ad7cce060 | `9fba1126ea14ab25`, `83616e77080fb295` | v |
| exp-127 `prepare_fold_frames.py` | snapshot bf1b952b9 (log) | `a7a91cd2ed914d01` | y |
| exp-134 `code/prepare_inputs.py`, `proxy_features.py`, `fit_exp134.py` | ad7cce060 | `40068deeb7c75991`, `106450ef73b59e23`, `7001010d1ed37d31` | v, y, v |
| exp-146 `code/build_frozen.py` | 3abf271bb | `fb6e0042479f0c3c` | v |
| `prc/data.py`, `prc/lirf_features.py`, prc/\_private\_clean\_score\_rows.py | ad7cce060 | `8bc762676e3c6376`, `95444c846ef405e5`, `f99e2c722d305013` | v, y, n (left out of the export on purpose; the exp-130 package holds a copy in `source/prc/`, MAN185:1141) |

- **Inputs.**
  - The official files in `${PRC_DATA_DIR}`, the final pair in `${PRC_DATA_DIR}/final/`. Full SHA-256 values:
    `release/ENTRIES.json` lines 797-814; the final and ranking pairs also in `prc/data.py` lines 101-110.

    | file | SHA-256 | bytes | rows |
    |---|---|---|---|
    | `ranking.parquet` | `9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86` | 43,564,605 | 689,534 |
    | `submitting.parquet` | `0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b` | 1,677,097 | 344,841 |
    | `final/final_ranking.parquet` | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` | 85,853,399 | 1,341,197 |
    | `final/final_submitting.parquet` | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` | 3,003,722 | 670,790 |
    | `training_2025-MM-01_…parquet` | 01 `a4dc8f15`, 02 `b6f9ff2c`, 03 `58fde7e5`, 04 `9ffe0cf5`, 05 `5481c064`, 06 `c3ef5080`, 07 `34d1c5bb`, 08 `6c8d2aa7`, 09 `955308a4`, 10 `4a31a356`, 11 `ac36b17d`, 12 `2d357f2f` | | |

    We downloaded the final pair on 2026-10-08 at 11:17 UTC from the organizers' data store (`docs/DATA.md`).
    `prc/data.py` lines 170-187 accepts a final file only when its SHA-256 is correct.
  - adsb.lol: the GitHub releases `adsblol/globe_history_2025` and `adsblol/globe_history_2026`, tag
    `v<date>-planes-readsb-prod-0`, parts `.tar.aa` and `.tar.ab`. The published and the read SHA-256 of each part
    and day are in exp-045 `manifest.json`, key `months.<m>.days.<d>.archives`. February 2026: 56 of 56 digests
    verified, 90,815,275,008 bytes read; June 2026: 107,820,895,744 bytes.
  - OurAirports `runways.csv` `47afb109bb0b7fa4` (3,963,747 bytes) and `airports.csv` `ca72a3404144b947` (12,720,205
    bytes), retrieved 2026-09-08 (exp-045 `analysis/boxes.json`, key `source`). Put them in `<external>/ourairports/`.
  - The frozen exp-018 vocabulary `ef056bb16b3db0013d84c885558122e88a535d3e2dac1b82077872f1a3ca72f3` (MAN185:22),
    from `${PRC_COMP_ROOT}/prc-exp148-src`.
  - The saved 2025 exp-130 package `${PRC_COMP_ROOT}/prc-exp148-work/kinput/prc-exp130-clean-input`
    (exp-185 `code/remote/pkg130_assemble.sh` line 9). The final package copies five of its files as they are:
    `eval_december` `d7dbe9a3`, `eval_january` `54142a27`, `eval_july_loo` `e2968dd0`, `preflight.json` `0b7a2b9d`,
    `census.csv` `25c0d43d` (MAN185:1106-1110, 1129).
  - The 2025 exp-047 blocks artifacts/exp-047/block/2025-MM.parquet (ignored by git). SHA-256 and rows per month:
    `experiments/exp-047-queue/analysis/interface.json`, key `months.<m>.sha256` (2025-01 `eb588315`, 2025-07
    `00e7f9df`, 2025-12 `e167c4df`). `emit.py` wrote them.
  - The 2025 exp-046 block caches exp-046 `work/block-2025-MM-<key>[-122c7d84|-ungated].parquet`
    (ignored by git). SHA-256: exp-185 `logs/exp127-keys-final.log` lines 20-31 and
    `refit-src/experiments/exp-146-validation/frozen/MANIFEST.json` (`inputs`). The names follow `chainfold.py:76`; the suffix
    `122c7d84` is the `block.py` prefix. The exp-046 code writes a cache file when it is absent (our reading
    of the code).
  - The exp-127 fold frames exp-127 `work/fold_frames.parquet` `fb811f693697ca35`. A
    rebuild in snapshot bf1b952b9 gave the same SHA-256 (exp-185 `logs/exp127-fold-frames.log`, key
    `prepared_parquet_sha256`; commit 84605a82b). The rebuild needs the v22 vector
    exp-046 `work/d-seasonal-bag10-255-free.npz` `3edfa35fee55e1d0` (on the x86 host at this path, and on the Mac in the exp-185 scratch `remote-inputs/`
    with the same SHA-256; exp-185 `code/mk_snapshot.sh` lines 26-27) and the old stand table (S3).
  - The stand tables of S3. ERA5 `era5_hourly.parquet` `8238378ec1f6d66a` (15,181,053 bytes; older route members
    only), at `<external>/era5/era5_hourly.parquet`.

- **Refit commands.** As our run gave them. `<W>` is a scratch folder for one run; our run copied its results to
  `F/mac/` and `F/`.
  1. Put the official files in `${PRC_DATA_DIR}`, the final pair in `${PRC_DATA_DIR}/final/` (RULES:17).
  2. Boxes, in `experiments/exp-045-harvest`:
     `python code/boxes.py <external>/ourairports/runways.csv <external>/ourairports/airports.csv 2026-09-08`
     (`boxes.py:14,100`).
  3. Harvest, one private Kaggle kernel per month:
     - `PRC_KAGGLE_USER=${PRC_KAGGLE_USER} bash experiments/exp-045-harvest/code/push_kernel.sh <name> "$(python code/days.py <month>)"`
       (`push_kernel.sh:5`; RULES:18).
     - February and June 2026: `PRC_KAGGLE_USER=${PRC_KAGGLE_USER} python experiments/exp-185-final-data/code/assemble_harvest.py <month> <days-json>`
       writes exp-185 `kernels/surface-<month>/` (`assemble_harvest.py:8`). Our run pushed the two kernels on
       2026-10-08 at 11:59:27 and 11:59:29 UTC, version 1 (`git show 48f7ed1db:experiments/exp-185-final-data/REPORT.md`,
       line 12). The push command is not in the record.
     - Land: `bash experiments/exp-185-final-data/code/land_harvest.sh 2026-02 2026-06` (it runs `pull_kernel.sh`, the dispatch row and
       `PRC_RANKING_MODE=final manifest.py`); receipt: `python experiments/exp-185-final-data/code/census_final.py 2026-02 2026-06`.
     - Kernels that ran under the old fetch rule (2025-12, 2026-01-b, 2026-01-gate):
       `python experiments/exp-045-harvest/code/rederive_local.py <suffix>` (`rederive_local.py:13`).
  4. exp-018 caches: `PRC_RANKING_MODE=final PRC_SRC=${PRC_SOURCE_ROOT} python experiments/exp-185-final-data/code/exp018_cache.py <W>/e018-final <vocabulary ef056bb1> 2026-01 2026-02 2026-06 2026-07`
     (RULES:24). For 2025: `PRC_SRC=${PRC_SOURCE_ROOT} python experiments/exp-185-final-data/code/exp018_cache.py <W>/e018-2025 <vocabulary> 2025-01 … 2025-12`
     (RULES:22).
  5. When the day summaries of all four months are on disk, for each month:
     `SN=${PRC_SOURCE_ROOT} W=<W> STEPS="class block e130" bash experiments/exp-185-final-data/code/run_month.sh <month>` (`run_month.sh:8`;
     RULES:20, 21, 23). The script sets `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2
     PRC_RANKING_MODE=final PRC_DATA_DIR=${PRC_DATA_DIR}` (line 19) and runs:
     - `python experiments/exp-185-final-data/code/class_final.py ${PRC_SOURCE_ROOT} <d> <M>` (line 39);
     - `cd ${PRC_SOURCE_ROOT} && PRC_ROOT=${PRC_SOURCE_ROOT} python experiments/exp-047-queue/code/emit.py <M> --sources F,O,H,N,geo --rule d4c59868864523ce6513151c1627ec2ba997fd25d6394cfe2ff03a14d89f01a0`
       (line 44);
     - `python experiments/exp-185-final-data/code/exp130_months.py ${PRC_SOURCE_ROOT} <d> <W>/e018-final <W>/class/<M> <M>` (line 53).
  6. exp-130 package:
     `python experiments/exp-185-final-data/code/exp130_2025.py ${PRC_SOURCE_ROOT} <W>/m2025 <W>/e018-2025 <saved input-manifest.json>`, then
     `python experiments/exp-185-final-data/code/exp130_final_package.py <saved and 2025 folder> <the four exp-130 month folders> ${PRC_DATA_DIR}/final/final_submitting.parquet <out> ad7cce060 ${PRC_SOURCE_ROOT}`
     (RULES:22); the assembly on the x86 host is exp-185 `code/remote/pkg130_assemble.sh` lines 9-13. Our run put the package
     in the private Kaggle dataset `${PRC_KAGGLE_USER}/prc-exp130-clean-input-final`, version 1.
  7. exp-127 frame and keys:
     `cd ${PRC_SOURCE_ROOT} && PRC_RANKING_MODE=final PRC_ROOT=${PRC_SOURCE_ROOT} PRC_DATA_DIR=${PRC_DATA_DIR} python experiments/exp-127-clock-precision/prepare_ranking_frame.py`;
     then the same with `rederive/prepare_clock_keys.py` and `PRC_KAGGLE_USER=${PRC_KAGGLE_USER}` (RULES:26). Put the keys in
     the private dataset `${PRC_KAGGLE_USER}/prc-exp127b-clock-rederive-keys-final`, then
     `kaggle kernels push -p experiments/exp-185-final-data/kernels/coverage-final` (RULES:25; the kernel reads 8 harvest kernels,
     `kernel-metadata.json`).
  8. exp-134:
     `cd ${PRC_SOURCE_ROOT} && PRC_RANKING_MODE=final PRC_ROOT=${PRC_SOURCE_ROOT} PRC_DATA_DIR=${PRC_DATA_DIR} PRC_EXP134_OUT=<out> PRC_KAGGLE_USER=${PRC_KAGGLE_USER} python experiments/exp-134-proxies-in-learner/code/prepare_inputs.py`
     (RULES:27).
  9. exp-146 frozen rows (ranking mode; no final-mode run is in the record):
     `PRC_SOURCE_ROOT=${PRC_SOURCE_ROOT} python experiments/exp-146-validation/code/build_frozen.py` (`build_frozen.py:3-4`).
  10. ERA5 check: `python experiments/exp-185-final-data/code/era5_cover.py <external>/era5/era5_hourly.parquet` (`era5_cover.py:5`).

- **Outputs.** Months January, February, June, July 2026, in this order.

| output | SHA-256 | rows | source |
|---|---|---|---|
| exp-045 day summaries `<external>/exp-045/summaries/<airport>/<day>.parquet` (our run: `<checkout>/artifacts/exp-045`; 2026 also in `F/exp-045/`) | per file in exp-045 `manifest.json` | 4,073 files, 4,035,289 rows over 16 months (sum over `manifest.json`) | MAN185 group `exp045-summaries`: 1,051 rows for the four 2026 months |
| harvest points `points_<day>.parquet` (kernel output) | per day, `outputs.points.sha256` | February 18,310,891; June 22,370,035 | exp-045 `manifest.json` key `kernels`; copies in `${PRC_COMP_ROOT}/prc-exp148-work/kinput/prc-harvest-<kernel>/` and `F/kinput/prc-harvest-2026-0{2,6}/` (exp-186 `code/make_kinput_final.py:20-23`) |
| exp-045 `analysis/boxes.json` | `195b3ee410ddc9fa` | 10 boxes | git; MAN185:1074 |
| `F/class-final/<m>/<m>-v25.parquet` | `e1744f69`, `6232183e`, `df03331b`, `d6693ea0` | 152,719; 144,158; 181,791; 192,122 | MAN185:7-14; E185 `REPORT.md:17` |
| `F/exp-047/block-final/<m>.parquet` | `21766bb4`, `5c232805`, `c31b84d9`, `bbb1ae40` | as the class; 185 columns | MAN185:1078-1081 |
| `F/mac/e018-final/<m>-final.parquet`; `vocab.json` | `45c7ddad`, `467e7bfa`, `08a8b475`, `1b338366`; `3319756d` | 92 columns | MAN185:1490-1494 |
| exp-018 2025 caches (deleted after use) | 2025-01 `ea215884` … 2025-12 `1dc95e2b` | per month | exp-185 `checks/exp130-2025-vs-saved.json` |
| `F/exp-130/months/month_<m>`, `rank_meta_<m>` | `6b6c0a04`, `34002779`, `39158443`, `9fe2b1a8`; `dd3dc93b`, `aad99931`, `f3a52cb0`, `2f0c692a` | 131 columns | MAN185:1098-1105 |
| `F/exp-130/package-final/` (37 files; `input-manifest.json` `3ac6381a`, `submitting_ids` `62cf73a3`, `rank_counts.csv` `f4410a83`) | per file | 414,719,781 bytes | MAN185:1106-1142; exp-185 `checks/kaggle-datasets.tsv` |
| `F/exp-127/final/ranking_frame-final.parquet` | `96bef303bad470d4` | 670,790 | MAN185:1087 |
| `F/exp-127/keys-final/clock_keys.parquet` | `18ff0314e77dc0ce` | 930,049 (259,259 fold and 670,790 final) | MAN185:1089 |
| `F/exp-127/coverage-final/clock_features.parquet` | `5f711ca447f47470` | 930,049; 58 columns | MAN185:1082; E185 `REPORT.md:24` |
| `F/exp-134/input-final/`: `proxy_departures`, `proxy_arrivals`, `points_expected.json`, `stand_geometry.csv` | `756971b0`, `afb102fe`, `13830e7c`, `83f30cbb` | departures per month as the class | MAN185:1143-1151; E185 `REPORT.md:25` |
| `refit-src/experiments/exp-146-validation/frozen/`: `rows.parquet`, `cells.csv`, `coverage.csv`, `clock_loss_days.csv`, `MANIFEST.json` | `991252d4`, `90f7c9a0`, `5512d320`, `4f9a697d`, `7574314e` | not in the record | git; `MANIFEST.json` key `files` |

- **Hardware, threads, time.**
  - Harvest: Kaggle CPU kernels with internet (exp-185 `kernels/surface-2026-02/kernel-metadata.json`). Sum of the day
    times: February 11,412.7 s, June 11,228.9 s (`manifest.json` key `months.<m>.seconds`). The first 14 months
    took 42.2 kernel hours and streamed 1,320 GB (exp-045 `RESULTS.md:7-9`).
  - Class, block, exp-130, exp-018, exp-134, exp-127 frame and keys: Mac arm64, checkout `.venv`, 2 threads
    (`run_month.sh:19`; RULES:20, 21, 23, 27). Wall time: not in the record. Coverage kernel: Kaggle CPU; time not
    in the record.
  - exp-146: numpy 2.4.6, pandas 2.3.3 (`frozen/MANIFEST.json`); host and time not in the record.

- **Check command.** `sha256sum` of each output against its MAN185 row (E185 `REPORT.md:5-7, 86-93`). Checks in our
  run:
  - Ranking-mode January rebuilds with the same code (at 2d30237b5): class 0 differences; block `e4ccfa09`; exp-018
    `ec45aef1`; exp-130 `6cc0c002` and `c043d31b`; exp-127 frame `3481827b` and keys `bf0616ac` (E185
    `REPORT.md:17-26`; RULES:20-27).
  - The 12 rebuilt 2025 exp-130 month files equal the saved package (exp-185 `checks/exp130-2025-vs-saved.json`).
  - A second exp-134 run gave five equal files (E185 `REPORT.md:25`).
  - The harvest check kernel `prc-harvest-2026-01-check` gave an equal frame for 2026-01-15
    (exp-185 `checks/harvest-2026-01-15.json`).
  - `build_frozen.py` writes each file once; a second run on existing files is a reproduction check (lines 15-18).

- **Differences from our run.**
  - Our January and July class read every landed summary day (`rule_cover`, 1,072 airport-days). We built it at
    ad7cce060, after February and June landed. It differs from the 2d30237b5 build on 16 January rows and 1 July row
    (E185 `REPORT.md:40-45`). Build the class after all four months landed.
  - For the same reason the ranking-mode exp-127 frame at ad7cce060 is `5fb40302`; the saved one is `3481827b`
    (E185 `REPORT.md:46-49`).
  - The harvest check bytes differ by the pyarrow version (Kaggle 23.0.1, our run 24.0.0); the frames are equal
    (48f7ed1db, E185 REPORT.md line 31).
  - The final package holds no `v22_submission.parquet` (E185 `REPORT.md:69-76`).
  - A new harvest reads the adsb.lol releases of today. Our `manifest.json` records the digests that we read.

## S2 OPDI first-event tables

- **Purpose.** Make the label-free OPDI first-event table of each departure (lead and distance of exit
  parking_position, exit apron and entry taxiway). exp-155 made it for 2025 and for January and July 2026; exp-185
  made it for the four final months; exp-190 `opdi_final.py` joins the two into `opdi-first-all`, the OPDI input of
  the clock frame (S10).

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `refit-src/experiments/exp-155-opdi-second-clock/code/extract_full.py` | git: be38b00ea (`d5c1d89c6ca78c16`). The copy that ran on the x86 host is `222a612f`, the bytes of 7012a9b1c; lines 3-4 (docstring) differ (exp-194 `b1-recovery/gate-check/RESULT.md` line 33) | see the commit column | n |
| `refit-src/experiments/exp-155-opdi-second-clock/code/after_extract.sh` | c08b9e56e | `830b65aee6875711` | n |
| `refit-src/experiments/exp-155-opdi-second-clock/code/match_full.py` | 3e1152a0c (the x86 copy is equal; RESULT.md:33) | `97c136b558a2bc1f` | n |
| `refit-src/experiments/exp-155-opdi-second-clock/code/gate_check.py` | bca080949 (RESULT.md:9-12; 3e1152a0c and 6c7b152ed hold the same bytes) | `c93ff0fe02818dd7` | n |
| exp-155 `source/exp125_analyze.py` (gate_check `matcher()` runs it; `bca080949:…/gate_check.py:10,21-36`) | 7012a9b1c (the only version) | `d2b7379f2d92fe0e` | n |
| `refit-src/experiments/exp-155-opdi-second-clock/source/exp125_opdi_sources.json` | 3e1152a0c; 6c7b152ed | `ed4a3bbc12f55376` | n |
| `refit-src/experiments/exp-155-opdi-second-clock/r2/code/prepare_inputs.py` | 2c19fec81 (exp-155 `r2/opdi-first-complete.json` line 204) | `c2165683713dfd19` | n |
| `refit-src/experiments/exp-155-opdi-second-clock/r2/code/screen_lane.sh` (the entry, line 14) | 2c19fec81 (exp-155 `r2/results/ARTIFACTS.json` lines 1140-1142) | `548d941605ac6966` | n |
| `refit-src/experiments/exp-185-final-data/code/opdi/extract_final.py` | 18e5f2400 | `c66d9fe3f2ca6872` | n |
| `refit-src/experiments/exp-185-final-data/code/opdi/match_final.py`, `prepare_first_final.py` | dbddbc427 (the x86 copies equal git) | `96827855a7888df6`, `8082922dd23457b5` | n |
| `refit-src/experiments/exp-185-final-data/code/remote/opdi_final.sh` | 711230c26 (MAN185:1563) | `e11ce591a49fcb91` | n |
| exp-190 `code/opdi_final.py`, `run_final.sh` | b12a4100b (same bytes at 259409f3f) | `5d2972d73b836bed`, `5ecc313b92a5bca3` | n |

  The wrappers keep the originals: exp-185 `code/opdi/match_full.orig.py` (`97c136b5`), `r2_prepare_inputs.orig.py`
  (`c2165683`).

- **Inputs.**
  - OPDI v002 from `https://www.eurocontrol.int/performance/data/download/OPDI/v002` (`extract_final.py:27,64,84`).
    2025: 18 flight lists and 45 event windows; bytes and SHA-256 in exp-155 `results/raw-source-manifest.csv`
    (`17ea927e`). The window 20251002_20251012 was absent at the source (line 48). `flight_list_202608` was 7,546
    bytes, `0c8856cd` (line 19).
  - The six new final windows (MAN185:1503-1508): 20260209 `cb8a2c83`, 20260219 `27eb39d1`, 20260301 `d588c4a5`,
    20260530 `4906af57`, 20260609 `28467511`, 20260619 `6c6e355c`. February 1-8 and June 29-30 come from the
    exp-155 windows 20260130 and 20260629 (48f7ed1db, E185 REPORT.md lines 51-52).
  - A check on 2026-10-11 found `flight_events_20260709_20260719` `bd06eb4e` and `flight_list_202607` `5b38d9bc`
    equal to their pins (exp-194 FULL-REVIEW-TEXT.md lines 314-315).
  - `${PRC_OPDI_ROOT}/work/airports.csv` (`match_final.py:35`, `match_full.py:27`; column `elevation_ft`, line 91).
    An external input (S0.1): the OurAirports file, our copy SHA-256 `5cb29a03…`.
  - Official rows: the 2025 months (`raw_sha256` in exp-155 `results/extract/extract-manifest.json`); the 2025 run
    reads `ranking.parquet` (`match_full.py:52`), the final run `final_ranking.parquet` `4b0692a7`
    (exp-185 `checks/remote-opdi-match-final-extract-manifest.json`).
  - Class: the 2025 run read `<checkout>/artifacts/exp-046/class` (`after_extract.sh:19`; hash not in the record);
    the final run read `F/class-final/v25`, `e1744f69`, `6232183e`, `df03331b`, `d6693ea0` (S1).
  - Member month files: the 2025 run read `${PRC_COMP_ROOT}/prc-exp148-work/kinput/prc-exp130-clean-input/month_*.parquet`
    (14 hashes, exp-155 `r2/opdi-first-complete.json` lines 76-201); the final run read `F/exp-130/months` (S1).

- **Refit commands.** `${PRC_OPDI_ROOT}/code` holds the 13 files of the exp-155 `code/` folder of branch
  exp155-opdi-second-clock (all in git; `b1-recovery/gate-check/RESULT.md`).
  1. 2025 extract (the docstring and the log give it):
     `cd ${PRC_OPDI_ROOT} && taskset -c 14,15,78,79 nice -n 19 python code/extract_full.py` (writes `full/`;
     exp-155 `logs/extract-full.log`).
  2. 2025 match (`c08b9e56e:…/code/after_extract.sh:15-20`):
     `cd ${PRC_OPDI_ROOT} && OMP_NUM_THREADS=4 taskset -c 14,15,78,79 nice -n 19 python -u code/match_full.py --work . --raw ${PRC_DATA_DIR} --class-root <checkout>/artifacts/exp-046/class --out results/extract`.
  3. 2025 first table (`2c19fec81:…/r2/code/screen_lane.sh:3-14`; lines 6-9 set 4 threads,
     `PYTHONPATH=${PRC_OPDI_ROOT}/r2/source`, `PRC_VALIDATION_DIR=<frozen rows>` and `PRC_DATA_DIR`):
     `python -u ${PRC_OPDI_ROOT}/r2/code/prepare_inputs.py --root ${PRC_OPDI_ROOT} --out ${PRC_OPDI_ROOT}/r2/inputs/complete --episode first --member-input-root ${PRC_COMP_ROOT}/prc-exp148-work/kinput`
     (interpreter: `${PRC_COMP_ROOT}/prc-exp148-src/.venv/bin/python`).
  4. Final extract, in `F/opdi` (RULES:30):
     `python ${PRC_SOURCE_ROOT}/experiments/exp-185-final-data/code/opdi/extract_final.py full 20260209 20260219 20260301 20260530 20260609 20260619`.
     Link the 44 exp-155 windows into `F/opdi/full` before the match.
  5. Final match and first table (`opdi_final.sh:7-22`; RULES:28-29), in `F`, with
     `PYTHONPATH=${PRC_OPDI_ROOT}/code OMP_NUM_THREADS=4`:
     - `python -u ${PRC_SOURCE_ROOT}/experiments/exp-185-final-data/code/opdi/match_final.py --work ${PRC_OPDI_ROOT} --raw ${PRC_DATA_DIR} --class-root class-final/v25 --out opdi/match-final --months 2026-01 2026-02 2026-06 2026-07 --flights opdi/full/flights.parquet --events-dir opdi/full --ranking ${PRC_DATA_DIR}/final/final_ranking.parquet`;
     - `python -u ${PRC_SOURCE_ROOT}/experiments/exp-185-final-data/code/opdi/prepare_first_final.py --root ${PRC_FINAL_ROOT}/finaldata --out opdi/first-final --episode first --member-input-root exp-130/months --manifest opdi/match-final/extract-manifest.json --months 2026-01 2026-02 2026-06 2026-07`.
  6. Join (exp-190 `run_final.sh:15,28-29`):
     `python experiments/exp-190-final-clock/code/opdi_final.py --saved ${PRC_OPDI_ROOT}/r2/inputs/complete/opdi-first.parquet --final ${PRC_FINAL_ROOT}/finaldata/opdi/first-final/opdi-first.parquet --out ${PRC_FINAL_ROOT}/finalclock/inputs/opdi-first-all.parquet`.

- **Outputs.**

| output | SHA-256 | rows | source |
|---|---|---|---|
| `${PRC_OPDI_ROOT}/full/flights.parquet` (also `F/opdi/full/`) | `b49a007866648de6651db26b6dd45b69dc49e168ac09a0018d07d0f2c5c2f3ed` | 2,713,598 | exp-155 `results/ARTIFACTS.json` lines 118-121; MAN185:1515 |
| `${PRC_OPDI_ROOT}/results/extract/events-YYYY-MM.parquet` (14 files) | per file (2026-01 `d99638c5`) | 2,429,886 | exp-155 `results/extract/extract-manifest.json` |
| `${PRC_OPDI_ROOT}/r2/inputs/complete/opdi-first.parquet` | `84b5c5ea3957ba1e1635c3db248f468ca4ec89d4cbfd501d82bda896a288bef0` | 2,429,888 | exp-155 `r2/results/ARTIFACTS.json` lines 11-14 |
| `F/opdi/full/events_2026*.parquet` (6 new) | MAN185:1509-1514 | not in the record | MAN185 |
| `F/opdi/match-final/events-2026-0{1,2,6,7}.parquet` | `57bc0ded`, `d71ce3bd`, `738881ef`, `417a55c8` | 152,719; 144,158; 181,791; 192,122 | MAN185:1520-1524 |
| `F/opdi/first-final/opdi-first.parquet` | `753c3ee9bdd6eb2965e632f1469be91a03534b45d55f7eb8ce154e0b96062aae` | 670,790 | MAN185:1500; E185 `REPORT.md:28` |
| `${PRC_FINAL_ROOT}/finalclock/inputs/opdi-first-all.parquet` | `4e6f2ee03660302eff1346111918616153d01bb3b376fd2d8dd2b28cd59542fc` | 2,755,837 (2,085,047 of 2025 and 670,790) | M190:2 |

- **Hardware, threads, time.**
  - exp-155: x86 host, cores 14, 15, 78, 79, nice 19, 4 threads (`after_extract.sh:15-17`; `screen_lane.sh:6,14`).
    Extract and match with the system `python3` (cpython-310 pyc, RESULT.md:18); the first table with the
    `prc-exp148-src` venv (Python 3.10.12, NumPy 2.2.6; exp-155 `r2/REPORT.md` line 46). Wall times: not in the record.
  - exp-185: x86 host, 4 threads (`opdi_final.sh:5,9`); started 2026-10-08 17:19 UTC (E185 `REPORT.md:102`); match
    outputs written 17:23-17:33 UTC (RESULT.md:20). Extract time: not in the record.
  - Join: 1 thread, 5.0 s (M190:2).

- **Check command.** exp-190 `run_final.sh:17-21` checks `753c3ee9` and `84b5c5ea` before the join; `opdi_final.sh:22`
  prints the `sha256sum` of each output. Checks in our run:
  - January in ranking mode: the extract of window 2026-01-10 again gave equal bytes (`1ddeea49`, MAN185:1495);
    `match_final.py` on `ranking.parquet` gives `events-2026-01` `d99638c5`, equal to exp-155 (RULES:28); the first
    table frame is equal (RULES:29).
  - On the 344,841 rows that the final and the saved tables share: lead rms 4.9 s, largest 12 s (E185
    `REPORT.md:58-60`).

- **Differences from our run.**
  - `extract_full.py` ran as the 7012a9b1c bytes and took cached raw files from `${PRC_OPDI_RAW}/`
    (`extract_full.py@be38b00ea:47-49`). A reader downloads them from the OPDI site.
  - The 2025 match used `match_full.py` at 3e1152a0c (our reading: only that version writes
    `all_type_offsets_sha256`). The recorded command is the c08b9e56e one.
  - A new download can differ from ours: one window was absent at the source, and `flight_list_202608` was 7,546
    bytes on our download day.

## S3 Stand tables

- **Purpose.** Map (airport, `STAND_mvt`) to the stand position from the X-Plane Gateway or OpenStreetMap and to the
  distances to the runway ends. The old table covers the stands of `ranking.parquet`; the final table covers the
  stands of `final_ranking.parquet`.

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| exp-127 `build_stand_geometry.py` | dd33cb5dc (the final run and the ranking rebuild) | `5fb13ef8d8f3b24b` | v |
| exp-127 `fetch_osm_parking_positions.py`, `merge_osm_geometry.py` | 9f95a42d6 (the only version) | `540e67086d2d5d23`, `bd123ea5fbb61ac4` (computed from git) | y (the export list holds both) |
| exp-127 `manifest.json` (Gateway packs), `osm_manifest.json`, `osm_parking_positions.csv` (data inputs) | 4e75933ef (release git tree) | `e196d74637b41e27`, `ccd73587852a5d51`, `0392699f6b59f7ab` (46,905, 769 and 3,624 bytes) | n: `refit-src/` ships them at their exp-127 paths |
| `refit-src/experiments/exp-185-final-data/code/exp127/prefetch_gateway.py` | 18e5f2400 | `2e4ac7d505a896ad` | n |

  `merge_osm_geometry.py:13` imports `build_stand_geometry` by name: keep its folder on `sys.path` (do not use
  `python -I` for this step).

- **Inputs.**
  - The three exp-127 data inputs are in the refit tree at their exp-127 paths (from `refit-src/`):
    `refit-src/experiments/exp-127-clock-precision/manifest.json`, `osm_manifest.json` and `osm_parking_positions.csv`.
  - Gateway `apt.dat`, 10 packs pinned in `refit-src/experiments/exp-127-clock-precision/manifest.json`, key `source_files`
    (file `e196d746`): EDDF 103368 `065c3fb5de426519`; EDDM 111124 `d7fa09fb2c0cba1d`; EGLL 111341
    `63bde47e67dc3771`; EHAM 111943 `a10e69d92923067a`; LEBL 102255 `700163e6f017c5fe`; LEMD 99851
    `4c5eb8cb712a3fd8`; LFPG 111022 `b1787f8cbd8378f6`; LIRF 110482 `34ec83782108ec31`; LSZH 106102
    `d530c122ac93bda5`; LTFM 105504 `cb849f454addd9a6`. URL `https://gateway.x-plane.com/apiv1/scenery/<id>`
    (`build_stand_geometry.py:69-71,93`). A check on 2026-10-11 (03:00-03:33 UTC) got all 10 with HTTP 200 and
    equal hashes (exp-194 FULL-REVIEW-TEXT.md lines 280-283, 293, 298-307).
  - OSM snapshot in git: `osm_parking_positions.csv` (30 rows, `0392699f6b59f7ab`) and `osm_manifest.json`
    (response `978a0b8306bddf58b5b2fe0a6831679e1f73500c4310e77d220a2316e2eeef59`, retrieved 2026-09-25T09:58:50Z;
    `osm_manifest.json:3,6-11`). The final run used the same snapshot (same values in `manifest-final.json`).
  - Stand names: ranking mode `${PRC_DATA_DIR}/ranking.parquet` (`build_stand_geometry.py:26-27`); final mode
    `prc.data.ranking_file("final")` (lines 32-36).

- **Refit commands.** The record gives
  `cd ${PRC_SOURCE_ROOT} && PRC_RANKING_MODE=final PRC_ROOT=${PRC_SOURCE_ROOT} python experiments/exp-127-clock-precision/build_stand_geometry.py`
  (RULES:33). The full arguments are not in the record; these come from argparse
  (`build_stand_geometry.py:200-203`; `merge_osm_geometry.py:33-36`) and exp-127 `README.md:60-63`:
  1. `python experiments/exp-185-final-data/code/exp127/prefetch_gateway.py experiments/exp-127-clock-precision/manifest.json <external>/gateway-cache`
  2. `cd ${PRC_SOURCE_ROOT} && [PRC_RANKING_MODE=final] PRC_DATA_DIR=${PRC_DATA_DIR} PRC_ROOT=${PRC_SOURCE_ROOT} python experiments/exp-127-clock-precision/build_stand_geometry.py --cache-dir <external>/gateway-cache --output <out>/stand_geometry[-final].csv --manifest <out>/manifest[-final].json`
  3. `python experiments/exp-127-clock-precision/merge_osm_geometry.py --geometry <out>/stand_geometry[-final].csv --osm-nodes experiments/exp-127-clock-precision/osm_parking_positions.csv --osm-manifest experiments/exp-127-clock-precision/osm_manifest.json --manifest <out>/manifest[-final].json`

  Step 3 writes the CSV and the manifest again in place (`merge_osm_geometry.py:121-123,142-143`). Do not run
  `fetch_osm_parking_positions.py`: it sends a live Overpass query (lines 16-20, 41).

- **Outputs.**

| output | SHA-256 | rows | source |
|---|---|---|---|
| old `experiments/exp-127-clock-precision/stand_geometry.csv` (4,224,601 bytes) | `2ce7ad6bf2b2c338272f1f04042027ca6d25d8c03d03708595c27f4882088008` | 13,086 (12,964 Gateway, 122 OSM) | exp-127 `REPORT.md:11` |
| final `refit-src/experiments/exp-127-clock-precision/stand_geometry-final.csv` (4,315,535 bytes) | `83f30cbbd2f4cc54bdebf1eaa56ae7b680c9a0fa4d70998ca10daf0797ccc02e` | 13,368 (13,236 Gateway, 132 OSM) | MAN185:1613; E185 `REPORT.md:26`; git fe30be466 |
| exp-185 `outputs/exp-127/manifest-final.json` (ignored by git) | `394f2bd630277306` | - | MAN185:1612 |

  The final table holds the 13,086 old rows with equal bytes, and 282 rows for 39 new stands
  (exp-190 SPEC.md lines 68-70). The exp-134 input folder holds a copy,
  `F/exp-134/input-final/stand_geometry.csv` `83f30cbb` (MAN185:1151).

  Which table each step reads: the head reads the old table on all rows (exp-193 `PACKET.md:399`). L, Lp and the
  MLP read the old table for 2025 and the final table for 2026 (`PACKET.md:447-448`; exp-186 `REPORT.md:222-224`).
  The exp-127 final frame and keys and the exp-134 final inputs read the final table
  (`prepare_ranking_frame.py:22,59`; `prepare_clock_keys.py:43`; `prepare_inputs.py:38-39`). The exp-127 fold frames
  read the old table (exp-185 `logs/exp127-fold-frames.log`, key `input_hashes`).

- **Hardware, threads, time.** Mac (the old script has private Mac paths in it,
  `9f95a42d6:…/build_stand_geometry.py:24-25`; MAN185:1613 gives the output location "repo"). Threads and wall time:
  not in the record.

- **Check command.** `shasum -a 256` of each CSV (also `output_sha256` in each manifest). In our run, a ranking-mode
  rebuild with dd33cb5dc and the pinned sceneries gave the old table with equal bytes
  (exp-185 `checks/exp127-stand-geometry-ranking-rebuild-manifest.json`); the prefetch gave "ok" for 10 packs
  (exp-185 `logs/prefetch-gateway.log` lines 1-10); the old rows are a subset of the final rows.

- **Differences from our run.** Path roots only, when the prefetch runs first. Without the cache,
  `download_airport` takes the scenery that the Gateway recommends today (`build_stand_geometry.py:59-71`). The old
  table came from version `acb2b6b3` (with fixed paths); the `5fb13ef8` rebuild gives equal bytes.

## S4 Route features (r4c tables and proxies)

- **Purpose.** Build `features-r4c` (132 columns) and `proxy-features-r4c` (25 columns) for the 2025 training months
  and the four final months. Level 2 (S6) and the clock frame (S10) read them (M189:2-3).

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| exp-189 `code/features_2026.py` | 76a65e731 | `1fc379cb49eb139a` | n |
| exp-189 `code/combine_final.py` | 76a65e731; b8426be56 | `b794a5da5bcd0586` | n |
| exp-189 `code/route_parts.sh` (stage `combine`) | 76a65e731 | `426144a037184bbb` | n |
| exp-189 `code/mk_feat_tree.sh` | c59e8d7a4 | `285e85f35cc01dd1` | n |
| exp-148 `code/features.py` (`features_2026.py` is its copy; exp-189 `code/PROVENANCE.md:5`) | 5e936513f | `4e7df021caaf3d2a` | v |
| imports: exp-044 `zs.py`, exp-046 code (`serve.py` and others), exp-047 `state.py`, exp-018 code | `release/REFIT-SOURCES.json` | `release/REFIT-SOURCES.json` | y or v |
| exp-134 `code/proxy_features.py` (in the Kaggle kernel) | ad7cce060; b165ab4ee; b5aab4dde | `106450ef73b59e23` | y |
| exp-134 `fit_exp134.py`, `prepare_inputs.py` | `release/REFIT-SOURCES.json` | `release/REFIT-SOURCES.json` | v |
| exp-148 `features.py`, `features_r2.py`, `combine_r4.py` (the saved 2025 rows) | as in the export | as in the export | y |

- **Inputs.** Hash sources: exp-189 `results/combine-final.json` (first four) and `results/features-2026-final.json`
  (the others).
  - `${PRC_COMP_ROOT}/prc-exp148-work/inputs/features-r4c.parquet` (the saved 2025 rows) `9101b06ae74eb0e0…`.
  - `${PRC_COMP_ROOT}/prc-exp148-work/inputs/proxy-features-r4c.parquet` (the saved 2025 proxies) `3597ac712b28643c…`.
  - features-2026 `c3f7e62de68b8dae213675072fb6dc04dd15d46eaadca634e66077438cbba6e4` (also `FEAT26_SHA` in
    `route_parts.sh`).
  - `FR/kout/learners-final-v1/out/proxy-features.parquet` (kernel version 1) `a91c043f89447fad…`.
  - Inputs of `features_2026.py` (S1): `clock_features` `5f711ca447f47470…`; exp-127 final ranking frame
    `96bef303bad470d4…`; exp-018 caches 2026-{01,02,06,07}-final `45c7ddadd4dfc897…`, `467e7bfa47b5d40d…`,
    `08a8b475c8f2db73…`, `1b33836683ae22d4…`; exp-047 block-final `21766bb4e30525db…`, `5c232805d9b20b4c…`,
    `c31b84d90d72ac24…`, `bbb1ae4023a58e4c…`. Our record gives Mac scratch paths for them; read them from `F/`.
  - Stand tables for the proxies: the old table for the 2025 rows, the final table for the 2026 rows, 39 names
    dropped (`kaggle/learners-final-v2/build.json`, key `mixed_stands`).

- **Refit commands.**
  1. Run tree: `bash experiments/exp-189-final-route/code/mk_feat_tree.sh <commit> <dir> <inputs dir>` (our Mac tree
     `feat-tree-c2415b9ae`, exp-189 `REPORT.md:17`). In the refit tree, `${PRC_SOURCE_ROOT}` takes its place.
  2. `PRC_SOURCE_ROOT=${PRC_SOURCE_ROOT} PRC_RANKING_MODE=final python experiments/exp-189-final-route/code/features_2026.py --out <dir> --implied F/exp-127/coverage-final/clock_features.parquet --v23-rank F/exp-127/final/ranking_frame-final.parquet`
     (`features_2026.py:3-4`).
  3. The proxies come from the Kaggle kernel of S5 (`learner_wp.py`, fit `features`).
  4. Combine (`route_parts.sh:58-64`; our call: `bash experiments/exp-189-final-route/code/route_parts.sh FR/kout/learners-final-v1/out combine`, M189:2):
     `python experiments/exp-189-final-route/code/combine_final.py --features-saved ${PRC_COMP_ROOT}/prc-exp148-work/inputs/features-r4c.parquet --features-2026 <features-2026> --proxy-saved ${PRC_COMP_ROOT}/prc-exp148-work/inputs/proxy-features-r4c.parquet --proxy-new FR/kout/learners-final-v1/out/proxy-features.parquet --out FR/inputs/r4c-final --allow-2025-diff clock_implied_row,clock_implied_l-1,clock_implied_l+0,clock_implied_l+1`.

- **Outputs.**

| output | SHA-256 | rows | source |
|---|---|---|---|
| `FR/inputs/r4c-final/features-r4c.parquet` | `39466f229682c802f6f39204341dae58fae26164d82e9cb8b58d4b8629054302` | 2,211,395 | M189:2 |
| `FR/inputs/r4c-final/proxy-features-r4c.parquet` | `e445405e36b2220ace34c8ec0956738225b6dfe5d745d964207c0c79a9755ccf` | 2,211,395 | M189:3 |
| features-2026 (written on the Mac, copied to the x86 host) | `c3f7e62d…` | 670,790 | `results/features-2026-final.json` |

  `combine-final.json`: rows_saved 1,885,446, rows_saved_2026 344,841, rows_out 2,211,395.

- **Hardware, threads, time.** `features_2026.py`: Mac, 60.3 s (`features-2026-final.json`; exp-189 `REPORT.md:17`).
  Combine: x86 host, 4 threads, 47.0 s, 2026-10-08T18:31:41Z (`combine-final.json`; M189:2).

- **Check command.** Ranking-mode parity: 344,841 rebuilt rows equal the saved `features.parquet`
  `b7d42132af8193ba…` in 132 columns (`results/features-2026-ranking-parity.json`; exp-189 `REPORT.md:16`). The
  exp-193 step `route-l2` (T193:260-274) checks both tables by SHA-256: PASS (R193:97).

- **Differences from our run.** The combine read the version 1 kernel proxies (`a91c043f`); the version 2 proxies of
  2026 equal them (exp-189 `REPORT.md:19`). The 2025 proxy rows are the saved rows of the old stand table; the combine
  accepts differences in four `clock_implied` columns of 2025 rows (`combine-final.json`).

## S5 Route level-1 members, L and Lp

- **Purpose.** Fit the eight level-1 route members on the final data and write one part per member: T-tailv4,
  C2-cat, L2-hub8725, D1-winter, F1-fresh, N1-nmtail (x86 host), L and Lp (private Kaggle kernel).

- **Source files.** `$M` = `experiments/exp-147-foldserve/code/members`.

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| exp-189 `code/run_member.sh` | 7d177a58a | `bf2f05903edee6eb` | n |
| the same, older (`chain_cpu.sh` calls it) | f8a5fe89e | `86b5e3084cbae54f` | n |
| exp-189 `code/chain_cpu.sh` | f8a5fe89e | `288a714ffc96c1ac` | n |
| exp-189 `code/make_src.sh` | 09234fca5 | `b211af78ed373211` | n |
| exp-189 `code/make_f1_tree.sh` | 5bbf4729d | `267c569ce2221de8` | n |
| exp-189 `code/build_final_learner_kernel.py` | b8426be56 | `d965d4cd0ef2544c` | n |
| exp-189 `code/make_parts.py` (stage `member`) | 76a65e731; b8426be56 | `47f3549dc8224a07` | n |
| `$M/T-tailv4_wp.py`, `D1-winter_wp.py`, `F1-fresh_wp.py`, `N1-nmtail_wp.py` | b165ab4ee | `b9991197b8710e21`, `ad077d236edcf539`, `8252e6b72f28448b`, `a5fda0f03067c156` | v |
| `$M/exp008_wp.py` (C2-cat, L2-hub8725), `$M/_wp_common.py` | b165ab4ee | `34cc5ac6494bfed3`, `0cc40ce52a9ff769` | v |
| exp-147 `code/learner_wp.py` (L, Lp, proxies) | b165ab4ee; b8426be56; e2bb55a2d | `91c90b979998cd72` | v |
| exp-147 `code/pilot.py` | `release/REFIT-SOURCES.json` | `f578b1782098eefe` | v |
| `prc/tail_model.py`, `phi_features.py`, `nmtail.py`, `weekpair.py`, `data.py` | `release/REFIT-SOURCES.json` | `release/REFIT-SOURCES.json` | y (`data.py` v) |

  The member scripts load more code by path: `N1-nmtail_wp.py:61-63` (exp-003-anatomy `common.py`, `features.py`;
  exp-010 `run_folds.py`); `D1-winter_wp.py:124-126` (exp-012 `winter_arm.py`; the exp-007 pipeline);
  `exp008_wp.py:143` (exp-008-diversity `run.py`); `F1-fresh_wp.py:128,136` (exp-018 code, `learner_wp.py`);
  `T-tailv4_wp.py:52-53` (`prc.data`, `prc.tail_model`).

- **Inputs.**
  - Tree `FR/src-b165ab4ee`: the git archive of b165ab4ee, the overlay 6002ff6ee, and the 2025 caches that
    `make_src.sh` copied from the frozen tree `${PRC_COMP_ROOT}/prc-exp148-src` (`fad0b6d5`); check file
    artifacts/CACHES.sha256 (exp-189 `REPORT.md:10`).
  - Tree `FR/src-b165ab4ee-f1b` (F1-fresh) adds the final exp-018 tables from `F/mac/e018-final` (the four hashes of
    S4; `vocab.json` `3319756d588505a7…`; artifacts/FINALDATA.sha256; exp-189 `REPORT.md:11`).
  - The official files `${PRC_DATA_DIR}` (`run_member.sh:12`).
  - Kernel: the private Kaggle datasets `${PRC_KAGGLE_USER}/prc-exp130-clean-input-final` version 1 (36 of 36 files checked) and
    `${PRC_KAGGLE_USER}/prc-exp134-input-final` version 1 (8 of 8), checked against exp-185 `checks/kaggle-datasets.tsv`; 19
    harvest kernels (exp-189 `REPORT.md:18`; `build.json`). Input hashes per month:
    `kaggle/learners-final-v2/out-v2/L-final.json`, key `inputs`.
  - The stand tables as in S4.

- **Refit commands** (`run_member.sh:10-29`; `<SRC>` is the tree above in `${PRC_SOURCE_ROOT}`).
  - `bash experiments/exp-189-final-route/code/run_member.sh <SRC> <OUT> <threads> <member>...` sets
    `PRC_SOURCE_ROOT`, `PRC_DATA_DIR` and the OMP, OPENBLAS, MKL and NUMEXPR threads, then runs:
    - T-tailv4, D1-winter, N1-nmtail: `python $M/<member>_wp.py --fit final --out <OUT> --threads 6`;
    - L2-hub8725: `python $M/exp008_wp.py --member L2-hub8725 --fit final --out <OUT> --threads 6 --device cpu`;
    - the N1 cache: member name `N1-cache` runs exp-003 `features.build_all()` (line 29).
  - F1-fresh: `python $M/F1-fresh_wp.py --fit final --out FR/members-gpu --threads 6 --device gpu` (tree
    `src-b165ab4ee-f1b`, M189:7).
  - C2-cat, as our run gave it (M189:4): `PRC_RANKING_MODE=final PRC_GPU=0 python $M/exp008_wp.py --member C2-cat --fit final --device gpu --out FR/members-gpu --threads 8`.
  - L and Lp: `python experiments/exp-189-final-route/code/build_final_learner_kernel.py --mode final --fits features final --mixed-stands --docker-image gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`;
    the kernel runs `learner_wp.py --fit final --learners L,Lp` (M189:11; `build.json` at b5aab4dde).
  - Parts: `bash experiments/exp-189-final-route/code/route_parts.sh <learner out dir> members <member>` (M189:4-12);
    it runs `make_parts.py member`.

- **Outputs.** All `FR/parts/<part>.parquet`, 670,790 rows (M189:4-12).

| part | SHA-256 | member file read (MANIFEST input) | M189 |
|---|---|---|---|
| T-tailv4 | `8c11e14a0a490a904c963f692754392889d98abd039ec468c6b7f39901e653b0` | `FR/members-t6`, `9522146d49de9de7…` | 10 |
| C2-cat | `b938268eeb54a72d964d220318ae05dfd089e6eb1e076b916b9ee33afbfa5c49` | `FR/members-gpu`, `4333d7839046951a…` | 4 |
| L2-hub8725 | `b6e8ff6cc3f96de3333c2ce458d6e8981d6583ccb54ec610aff150dbafef2ed4` | `FR/members`, `2021ea62b43c0bb6…` | 8 |
| D1-winter | `73acfe84bec9997dc349573ecf81c412cc90b39e3aab4f630f0aae0fca62d85d` | `FR/members-t6`, `bc12aa31472270b1…` | 5 |
| F1-fresh | `12ce9be093942a8093ace8e13c3f6b0b224caad58a29f4ab16cc85edc4d75bc6` | `FR/members-gpu`, `e1b09d8f888f05e5…` | 7 |
| N1-nmtail | `b85bbcc1cbcfd41a2ec1fac99eb6e4803d296665ad8090ce4a4898267f6d73cd` | `FR/members`, `0503ca2c0d47e4ba…` | 9 |
| l | `284d7354665a7f18a96bf24b3cec4222a7ab849b0fc86d60510d1e4e55b39150` | `FR/kout/learners-final-v2/out/L-final.parquet` `877e53c5435ee607…` | 11 |
| lp | `57bc0810b3da7655857f2566a083c6d05665da40a3229c2517880425131da74b` | the same folder, `Lp-final.parquet` `f76b7f83c9af8a67…` | 12 |

  The l and lp parts hold 624 NaN values (R193:40-41). The kernel proxy output is `2f13cdb393b058ff…`, 1,015,209 rows
  (`out-v2/proxy-features.json`). The lp part also feeds the MLP (S9; exp-186 `code/run_mlp_186.sh:47-48`); the l part
  feeds the `pack.py --lref` check of the clock (exp-193 `PACKET.md:398`).

- **Hardware, threads, time.**
  - x86 host, CPU members at 6 threads (M189:5, 8, 9, 10). Fit times: T-tailv4 606 s, D1-winter 993 s, N1-nmtail
    1,127 s, L2-hub8725 9,412 s (T193:148, 158, 168, 178); exp-189 `REPORT.md:20` gives 519 s for T-tailv4 at 6
    threads. exp-189 `REPORT.md:14` describes the shared CPU run.
  - GPU 1 of the x86 host: C2-cat 8 threads, 1,236 s; F1-fresh 6 threads, 10,162 s (T193:242, 252; exp-189
    `REPORT.md:15`).
  - Kaggle CPU, no GPU, no internet (`build.json`): L fit 1,368.8 s (total 1,457.4 s); Lp fit 1,350.6 s (total
    2,877.5 s); lightgbm 4.6.0, numpy 2.0.2, pandas 2.3.3, Python 3.12.13 (`out-v2/L-final.json`, `Lp-final.json`).
    Kernel version 2 COMPLETE at 20:47 UTC on 2026-10-08 (`results/kernel-learners-final-v2.txt`).
  - x86 stack: Python 3.10.12, LightGBM 4.6.0, CatBoost 1.2.10 (exp-189 `SPEC.md:88-89`).

- **Check command.** The exp-193 rebuild compares by SHA-256: T-tailv4, D1-winter, N1-nmtail, L2-hub8725
  (T193:146-184) PASS (R193:87-90); l and lp from the saved kernel outputs with `ROUTE_KERNEL_SHA256`
  (T193:186-202) PASS (R193:40-41). C2-cat and F1-fresh have read checks (T193:240-258): rms 24.8 s and 12.58 s
  (R193:101-102). Part parity: `route_parts.sh <dir> parity` against member-refs `88d422cb4e1d8d3d…` and v73-parts
  (`route_parts.sh:131-140`).

- **Differences from our run.**
  - The CPU members give equal bytes at 6 threads. D1-winter at 4 threads moved by rms 79.6 s (M189:5).
  - C2-cat ran with `PRC_GPU=0` after a device-index error at `PRC_GPU=1` (M189:4; exp-189 `REPORT.md:322-325`).
    C2-cat and F1-fresh are GPU fits and give a recipe refit: C2-cat refit noise rms 31.17 s (M189:4); exp-189 did not
    measure F1-fresh (M189:14).
  - The Kaggle host adds noise: one L refit moved by rms 16.67 s (M189:11), Lp by 3.6e-4 s (M189:12). The pinned image
    does not remove it.
  - M189:4-12 give the code commit b8426be56 for every part; T193:188 gives 0cf58d001 for l. exp-189
    `REPORT.md:327-328`: six early parts were written at 0cf58d001 with the same `make_parts.py` `47f3549d`.

## S6 Route level 2, F and np_base

- **Purpose.** Predict the saved R4mks level 2 on the eight member parts and write the route vector. `route` = `F`
  = `np_base` = R4mks `pred_Sb` on all 670,790 rows (`make_parts.py:139-157`).

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| exp-189 `code/route_parts.sh` (stages `level2`, `parts`) | b8426be56 | `e811412aee87c1b8` | n |
| exp-189 `code/make_parts.py` | 76a65e731; b8426be56 | `47f3549dc8224a07` | n |
| exp-148 `code/fold_serve.py` | 76a65e731; b8426be56; c6faf428d; cb18f1903 | `6ec7b92732868e1a` | n |
| exp-148 `code/stacker.py` | `release/REFIT-SOURCES.json` | `b99dfbb21f79823c` | v |
| exp-186 `code/parity.py` (stage `parity`) | `release/REFIT-SOURCES.json` | `0ac593fdcf58812e` | n |

- **Inputs.**
  - The eight member files of S5, linked into `FR/art-final` (16 links, `route_parts.sh:66-117`).
  - `features-r4c` `39466f22…` and `proxy-features-r4c` `e445405e…` (S4).
  - The saved level-2 fit in `${PRC_MODEL_ROOT}/prc-exp148-work/stack`: `model.txt` `1336c30386ca1379…`, `cat.cbm`
    `069357f24e248ffb…`, record `R4mks-final.json`, feature list `e2376cbc89d81d00…`
    (exp-189 `results/level2-ranking-parity.json`).

- **Refit commands** (`route_parts.sh:118-127`; level 2 only predicts). Environment (`route_parts.sh:33-35`):
  `PRC_RANKING_MODE=final PRC_SOURCE_ROOT=FR/src-b165ab4ee PRC_DATA_DIR=${PRC_DATA_DIR} PRC_FINAL_ROOT=${PRC_FINAL_ROOT}`,
  OMP, OPENBLAS and MKL at 4 threads. In the refit, `PRC_SOURCE_ROOT` is the refit tree, and `fold_serve.py`
  needs `PRC_R2_NO_FEATURES` (S0.1).
  1. `python experiments/exp-148-short-route/code/fold_serve.py --stage R4mks --inputs r4mks --split final --member-art FR/art-final --stack ${PRC_MODEL_ROOT}/prc-exp148-work/stack --features FR/inputs/r4c-final/features-r4c.parquet --proxy FR/inputs/r4c-final/proxy-features-r4c.parquet --out FR/stack`
  2. `python experiments/exp-189-final-route/code/make_parts.py level2 --src FR/stack/R4mksP-final.parquet`
  3. `python experiments/exp-189-final-route/code/make_parts.py route --r4mks FR/parts/r4mks.parquet`

  Our calls: `bash experiments/exp-189-final-route/code/route_parts.sh <learner out dir> level2`, then the same with
  `parts` (M189:14).

- **Outputs.**

| output | SHA-256 | rows | source |
|---|---|---|---|
| `FR/stack/R4mksP-final.parquet` | `f28f13f8745cfc44…` | not in the record | M189:14 (input) |
| `FR/parts/r4mks.parquet` (`pred_Sb`, `pred_S`, `pred_Scat`, `anchor`, `anchor_served`) | `f1587a173ffc9ba5bf355aa5c8a457b6cc0e44f866bc64d626bf712203e1e490` | 670,790 | M189:14 |
| `FR/parts/route.parquet` (`TAXITIME_SEC_mvt`, `route_source`) | `232d267058af79c6cc0fd628c431f7f748d9719a711830272479b4f06a28dac0` | 670,790 | M189:15 |
| `FR/parts/F.parquet` | `66829b3a086cc1983782e5aba45be4fd509fc3a1296436cb55ddffe5701fb853` | 670,790 | M189:6 |
| `FR/parts/np_base.parquet` | `ee84b9f32393f86a994e59e061632937866db8b599f14a1232c81c7dc1d3ceee` | 670,790 | M189:13 |

  The GBDT fallback took 0 rows (`results/parts/route.json`, `rows_by_source`: R4mks 670,790).

- **Hardware, threads, time.** x86 host, 4 threads (M189:14). Level 2 and parts 2.5 min (T193:262). Our chain ran
  from 07:03:22 to 07:06:52 UTC on 2026-10-09 (exp-189 `REPORT.md:22, 273-279`).

- **Check command.**
  - Release replay: pack `release/ZABD-PACK-final-r9.json` line 1507 (`ranking/rule_frame:F`, part
    `F.parquet` `66829b3a086c`) and line 1552 (`ranking/vector/route`, part `route.parquet` `232d267058af`). Both saved
    columns have SHA-256 `f2f3f0194f9ab7d1056f1943b45fe51aa18b28329150973efcf2c4421acc929b`
    (`release/saved-vectors/final/ranking/vector/route/TAXITIME_SEC_mvt.json` and
    `…/ranking/rule_frame/F.json`). Script: `release/assembly_replay.py` lines 59, 110-157 (file and column SHA-256).
  - Rebuild: `bash experiments/exp-193-final-build/code/route_level2.sh <root>` (T193:260-274): PASS 6 of 6 (R193:97).
  - Ranking-mode level-2 parity: equal to the saved `R4mks-final.parquet` `f6da084a2e4646de…`, max_abs 0
    (`results/level2-ranking-parity.json`; exp-189 `REPORT.md:13`).

- **Differences from our run.** Level 2 is deterministic. The noise of the member refits reaches `pred_Sb`: the
  C2-cat draw rms 7.88 s, the L draw 4.15 s, about 8.9 s together (M189:14). exp-193 `route-l2-gpuread` with both GPU
  refits: route, F and np_base move by rms 8.31 s; F1-fresh adds about 2.6 s (R193:104, 122-127).

## S7 to S10: shared facts

Each member part comes from saved fits of the released member. A predict step applies them to inputs built on
`final_ranking.parquet`. To refit, run the fit commands of the section first; a GPU fit gives a recipe refit
(section "Two kinds of result").

- `E186`, `E170`, `E190`, `E193`, `E164`, `E181` before a path: the experiments exp-186, exp-170, exp-190,
  exp-193 and exp-164, and the folder `serve-packet` of exp-181.
- The three core scripts use `PY=${PY:-python3}` (`run_g_186.sh@25cbf691e:26`, `refit-src/experiments/exp-170-diverse/code/run_exp170_186.sh` line 42,
  `refit-src/experiments/exp-186-final-core/code/run_mlp_186.sh` line 33); on the x86 host that was `/usr/bin/python3` (E186 `REPORT.md:141`).
- `${PRC_DATA_DIR}` on the x86 host was `${PRC_FINAL_ROOT}/finalcore/raw`: links to the 2025 files, `ranking.parquet`
  and `submitting.parquet`, and copies of the final pair in `final/` (E186 `REPORT.md:147-152`).
- Check tool for all four sections: `python experiments/exp-193-final-build/code/build_final.py run-step --step <s> --root <root>`
  and `build_final.py compare --step <s> --rebuild <root>` (E193 `code/build_final.py:6-8`).
- The parity reference of all four: `${PRC_FINAL_ROOT}/finalcore/refs/v73-parts.parquet`
  `8af18086019e3867defdc1b6c997e7f3500360573fd2c4c96ec28ac140c9f8e5` (path: `run_g_186.sh@25cbf691e:22`; SHA-256: key
  `refs_sha256` in exp-186 `results/e170-final-2c3b8ac99/*-parity.json` line 3).

## S7 g

- **Purpose.** Part `g.parquet` (`pred_Gmixed`): the mean of 12 saved sequence-g fits (recipes mixed64 and mixed128;
  seeds 150081-150086).

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `refit-src/experiments/exp-181-modela-bag-read/serve-packet/code/run_g_186.sh` | 25cbf691e | `d0102715d8fc979e` | n |
| `refit-src/experiments/exp-181-modela-bag-read/serve-packet/code/predict_saved_186.py` | 25cbf691e | `9bf9413940d8fe42` | n |
| `experiments/exp-181-modela-bag-read/serve-packet/code/serve_bag.py` | 25cbf691e | `3e499d2cca0cd641` | y |
| `experiments/exp-181-modela-bag-read/serve-packet/code/fit_bag_033.py` | as in the export | `616b92239e849403` | y |
| exp-150 sequence-g `fit.py`, `model.py` | as in the export | `e7b3a6cad301b9c4`, `117214b1e2006f92` | y |
| exp-150 sequence-g `prepare_annual_186.py` (it runs `prepare_annual.py` `5e6b21c7ce371df5`, which uses `prepare.py` `c1f7e45080cc0a8f`) | 25cbf691e | `d557b195dbfebbbe` | n |
| exp-150 sequence-g `check_annual_186.py`; `ANNUAL-PLAN.json` | 25cbf691e; `release/REFIT-SOURCES.json` | `c904ec96c8e2b016`; `492f97f3b5a05cb6` | n |
| exp-150 `r4/admission/upstream/` at bb23613f7: `MANIFEST.json`, `r4_admission_runtime.py`, copies | bb23613f7 | `5d55fadae10a1d8f`, `73b112c2b39a0e5c` | n |
| `refit-src/experiments/exp-186-final-core/code/parity.py` | 25cbf691e | `493bd478d4d1d5b5` | n |

- **Inputs.**
  - The saved 2025 inputs `${PRC_COMP_ROOT}/prc-exp150-work/out/sequence-g-annual-inputs-001`, MANIFEST
    `dc598af343a9f60dca39cf87b4d158924677595f9ceb22298e0b432be9414798` (`run_g_186.sh@25cbf691e:20`; M186:2).
  - The fits `${PRC_MODEL_ROOT}/prc-exp181-work/out/gpu0/serve-ab-<recipe>-<seed>/` (`model.pt`, `FIT.json`;
    `run_g_186.sh@25cbf691e:21,30`); the SHA-256 of each `FIT.json` is in
    exp-186 `results/g-final-25cbf691e/predict_PREDICT.json`. `SERVE-PLAN.json` `8fd0d621`.
  - `${PRC_DATA_DIR}` (`run_g_186.sh@25cbf691e:25`), with `final_ranking.parquet` `4b0692a7` (E186 `REPORT.md:20-34`).
  - The generated `ANNUAL-PLAN-final.json` is `53b70693`; its encoder and train labels equal the saved inputs; the
    4,167,797 events of 2025 are equal (exp-186 `results/g-final-25cbf691e/inputs_CALLER-186.json`).

- **Refit commands.**
  - Our call (M186:2): `bash experiments/exp-181-modela-bag-read/serve-packet/code/run_g_186.sh final <out> 2`. Its
    steps (`run_g_186.sh@25cbf691e:36-54`; `$G` the sequence-g folder, `$SP` the serve-packet folder):
    1. `python $G/prepare_annual_186.py --board-plan "$G/ANNUAL-PLAN.json" --plan-out "$OUT/ANNUAL-PLAN-final.json" --out "$OUT/inputs" --compare <saved inputs>`
    2. `python $G/check_annual_186.py --inputs "$OUT/inputs" --out "$OUT/ANNUAL-CHECKS-final.json"`
    3. `python $SP/code/predict_saved_186.py --inputs "$OUT/inputs" --plan "$SP/SERVE-PLAN.json" --checks "$OUT/ANNUAL-CHECKS-final.json" --saved-inputs <saved inputs> --board-checks "$SP/ANNUAL-CHECKS.json" --fits <the 12 fit folders> --out "$OUT/predict" --part "${PRC_FINAL_ROOT}/finalcore/parts/g.parquet" --threads 2`
    4. `python experiments/exp-186-final-core/code/parity.py --refs <v73-parts> --new <g.parquet> --part g --col pred_Gmixed --kind refit --out <g-parity.json>`
  - The fits (refit): `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -u <root>/code/serve_bag.py --inputs <root>/inputs --plan <root>/SERVE-PLAN.json --checks <root>/ANNUAL-CHECKS.json --recipe <recipe> --seed <seed> --out <out>/serve-ab-<recipe>-<seed>`
    (E181 HANDOFF-GRIND.md; E181 `SPEC-SERVE.md:87`). `SERVE-PLAN.json`: mixed64 width 64, 2 layers, ff 128, 4 heads,
    dropout 0.1; mixed128 width 128, 3 layers, ff 256; 12 epochs; SmoothL1 beta 2; AdamW lr 0.001, weight decay
    0.0001; batch 256; clip 1; floor 60; 2,085,047 labels (the 12 months of 2025).

- **Outputs.** `${PRC_FINAL_ROOT}/finalcore/parts/g.parquet`
  `8758a9716eba08b33fef5f2ce9753594cb68d55755a29e8a5b4cfb2c345ab6ef`, 670,790 rows, columns `MVT_ID_mvt`,
  `pred_Gmixed`, written 2026-10-08T17:26:31Z; a second run at 2c3b8ac99 gave the same SHA-256 (M186:2). Move against
  the saved vector: rms 9.1805 s, max 1,211.6 s (`predict_PREDICT.json`).

- **Hardware, threads, time.** Predict: x86 host CPU, 2 threads, torch 2.10.0+cu128 (`predict_PREDICT.json`); 74 min
  (E193 `REPORT.md:50-55`); the exp-193 rebuild took 35.2 min, PASS (E193 `REPORT.md:31`). Fits: GPU0 NVIDIA TITAN
  RTX, 3 streams, nice 19, 2 threads; mixed128 2,473-2,495 s per fit, mixed64 1,630 s; 2026-10-03 17:05-19:23 UTC
  (E181 `REPORT-SERVE.md:6-22`; `FIT.json` in E181 `results/serve-fits/`).

- **Check command.** `parity.py` (step 4 above); `build_final.py` step `g`; the pack row
  `ranking/vector/g/pred_Gmixed`, "Saved file: yes; SHA-256 PASS".

- **Differences from our run.** Path roots only for the predict. A fit refit is a GPU recipe refit. The held (fold)
  g bag ran on CPU, six threads, nice 19, on 1,723,505 labels outside January and July 2025 (`BAG-PLAN-033.json`
  `fd485bc7ef9ccd2b`; E181 REPORT-BAG-033.md); S16 covers it.

## S8 Sequence members (sequence021, sequence039, row6, sequence011, row011, dark18)

- **Purpose.** Six parts with column `pred_seq170`. Each is the equal mean of the saved twelve-month exp-170 fits of
  the member.

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `refit-src/experiments/exp-170-diverse/code/run_exp170_186.sh` | 2c3b8ac99 | `39605995f4230602` | n |
| `refit-src/experiments/exp-170-diverse/code/rank_inputs_186.py` | 2c3b8ac99 | `67e47c7455eb93b0` | n |
| `experiments/exp-170-diverse/code/prepare.py` | 2c3b8ac99 | `2b96a09215cd267d` | y |
| `refit-src/experiments/exp-170-diverse/code/predict_rank_186.py` | 2c3b8ac99 | `01d7d091b942d62a` | n |
| `experiments/exp-170-diverse/code/fit.py` (row6, row011) | as in the export | `79ec20fea289f93c` | y |
| `experiments/exp-170-diverse/code/fit_runway.py` (sequence011, sequence021) | as in the export | `7207612fc686b192` | y |
| `experiments/exp-170-diverse/code/fit_runway_fast_029.py` (dark18, sequence039) | as in the export | `4e86dc93cf5a698b` | y |
| `refit-src/experiments/exp-170-diverse/code/average_fits_026.py` | `release/REFIT-SOURCES.json` | `a429f9915e5ae83d` | n |
| `experiments/exp-170-diverse/code/rebuild_legacy_final_060.py` | 751f9457b | `923971aa8f9aaa6f` | y |
| `refit-src/experiments/exp-170-diverse/code/rebuild_fixed039_annual_061-pre-split-1511.py` | 751f9457b | `cc582b0be851df9d` | n |
| `refit-src/experiments/exp-170-diverse/code/prepare_gap.py` | 41213d0b7 | `eb2c5b24bcdfc1f6` | n |
| `refit-src/experiments/exp-186-final-core/code/parity.py` | 2c3b8ac99 | `0ac593fdcf58812e` | n |

  In the export with the same bytes (SHA-256 of the git file): `rebuild_fixed039_annual_061.py`
  `01919a873b574b42`, `fit_dark_annual_064.py` `29315baa20b10ec7`, `prepare_time_day_052.py` `198b12f19f6dc3c4`,
  `prepare_row_time_054.py` `c838aeb1764a4730`, `prepare_runway_time_059.py` `7587648101592b07`,
  `prepare_coherent_dark_053.py` `fad2b2cbe313d96c`, `prepare_runway.py` `08a6211c61c6e915`.

- **Inputs.**
  - Views `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/inputs/<view>` (`run_exp170_186.sh:22-23`;
    the main checkout of the x86 host). Member to view (`run_exp170_186.sh:46-47`): sequence021 and sequence039
    `time-day-Z-052`; dark18 `coherent-dark-Z-annual-064`; row6 and row011 `row-time-day-Z-054`; sequence011
    `runway-time-day-Z-059`. View MANIFEST prefixes: `cd2c84e0` (row6), `8b312d3e` (sequence011), `d052ad02`
    (sequence021, sequence039), `8bd9349b` (dark18) (M186:3-8).
  - Saved means `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/<mean>/FIT.json` (they name
    the parent fits; `run_exp170_186.sh:49-51`): sequence021 `fixed-sequence021-bag6-annual-060`; sequence039
    `fixed-close30-annual-061`; dark18 `dark18-Z-annual-064`; row6 `fixed-control6-bag6-annual-060`; sequence011
    `fixed-fallback_sequence011-17001-annual-060`; row011 `fixed-control6-17001-annual-060`.
  - Fits `${PRC_MODEL_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out` (`run_exp170_186.sh:24`).
  - The clock frame12 `47b5b6bf58143f971c9ec81f8cc90f74baef5acb2d70ca94dad7e91af6741f3d` and track12
    `d748df61fd2d504df14387e5555cc8d8ae77ae0253e3f40b821257e829c56f68` (M186:3-8; S10 builds them).

- **Refit commands.**
  - Our call, one per member (M186:3-8):
    `env E170_CLOCK=<frame12> E170_CLOCK_SHA256=47b5b6bf… E170_TRACK=<track12> E170_TRACK_SHA256=d748df61… bash experiments/exp-170-diverse/code/run_exp170_186.sh final <out> 2 <member>`.
    Its steps (`run_exp170_186.sh:58,77-82`):
    1. `python experiments/exp-170-diverse/code/rank_inputs_186.py --view <views>/<view> --clock <frame12> --track <track12> --out "$OUT/inputs/<view>"`
    2. `python experiments/exp-170-diverse/code/predict_rank_186.py --inputs "$OUT/inputs/<view>" --fits <parents> --out "$OUT/predict/<member>" --reference <exp-170>/out/<mean>/ranking2026.parquet --tol 0.05 --threads 2 --part "${PRC_FINAL_ROOT}/finalcore/parts/<member>.parquet"`
    3. `python experiments/exp-186-final-core/code/parity.py --refs <v73-parts> --new <part> --part <member> --col pred_seq170 --kind refit --out <member>-parity.json`
  - The fits (x86 GPUs, 2026-10-03, nice 19, 2 threads):
    - 060 (row6, row011, sequence011, sequence021): `rebuild_legacy_final_060.py` on GPU0, 11:23:35-12:15:37 UTC
      (exp-170 `results/FINAL-LEGACY-REBUILD-060.json`). Sequence fits: `fit_runway.py`, 12 rounds, batch 512, width 64,
      2 layers. Row fits: `fit.py`, 12 rounds, batch 128, core 32, halo 0 (E170 `SPEC-FINAL-LEGACY-060.md:5`).
    - 061 (sequence039): `rebuild_fixed039_annual_061.py`; GPU1 12 fits wide24 (width 256, 24 rounds); GPU0 12 fits
      r48 (width 128, 48 rounds) and 6 fits wide48 (width 256, 48 rounds); 3 layers, batch 512, Huber, stage LAST;
      ended 15:53:17 UTC; mean FIT `7cab87ba`, vector `63077384` (exp-170 `results/FIXED039-ANNUAL-061.json`; E170
      `SPEC-FIXED039-ANNUAL-061.md:5-7`).
    - 064 (dark18): `fit_dark_annual_064.py`, fast029 r48, width 128, 3 layers, 48 rounds, batch 512, Huber, `--final`;
      seeds 17001-17009 on GPU0, 17010-17018 on GPU1; 20:16:18-21:46:55 UTC; mean FIT `a22e074e`, vector `d6ac7cc5`
      (exp-170 `results/DARK-ANNUAL-064.json`; E170 `SPEC-064.md:7-9`).
  - The views (in the exp-170 folder; interpreter `${PRC_COMP_ROOT}/prc-exp148-src/.venv/bin/python`):
    - 059: `nice -n 19 taskset -c 8-11 python -u code/prepare_runway_time_059.py --row-inputs inputs/runway-006 --fixed-inputs inputs/time-day-Z-052 --selection results/TIME-DAY-SELECTION-052.json --selection-review REVIEW-TIME-DAY-SELECTION-ACTUAL-052.md --selection-review-sha b8c9eaa96bacf853bbc4fbe912d0ad73bc1828c4bae5538e8c14491483230062 --out inputs/runway-time-day-Z-059`
      (exp-170 `results/LAUNCH-RUNWAY-Z-059.json`);
    - dark view: `nice -n 19 taskset -c 8-11,72-75 python -u code/prepare_coherent_dark_053.py --inputs inputs/time-day-Z-052 --selection results/TIME-DAY-SELECTION-052.json --coverage ${PRC_COMP_ROOT}/prc-exp159-work/source/experiments/exp-146-validation/frozen/coverage.csv --out inputs/coherent-dark-Z-annual-064 --annual`
      (exp-170 `results/LAUNCH-ANNUAL-DARK-INPUT-064.json`);
    - the chain before them: `prepare.py` (root), `prepare_runway.py` (`runway-006`), `prepare_gap.py`
      (`runway-gap-014`), 052 (`time-day-Z-052`) (`experiments/exp-186-final-core/pins/D-neural.md:7-21, 42-59`), on the exp-159 frame12 and
      track12 (`D-neural.md:84`).

- **Outputs.** `${PRC_FINAL_ROOT}/finalcore/parts/<member>.parquet`, 670,790 rows each, columns `MVT_ID_mvt`,
  `pred_seq170`, code 2c3b8ac99; a second run gave the same SHA-256 (M186:3-8).

| member | SHA-256 | fits |
|---|---|---|
| row6 | `56c60492e8adf67e73b7dd2dfbb11960359ec5239a1ebf67abd9542d667228c5` | 6 (seeds 17001-17006) |
| row011 | `399f6987883ba242a394560ed5fd73d0ec8cf420b38f0749bc3f9fe785f76680` | 1 |
| sequence011 | `336528311f023d90311b119135b5fa0b5fb411725023a1246c5cc45acad4d3a8` | 1 |
| sequence021 | `c21db6dca6f8ecabd37abe973c27078368e8c3947977b4cc23b1ae7a46d448fd` | 6 |
| dark18 | `86c049d0795bdaf741fca6f2ee546b427287b070fd4fb7b8f1c6bf360435467f` | 18 |
| sequence039 | `636ab22a0b44d191fd6e0e35dcb7e8fdb52cdc9485490d0661c74c17f7f4203e` | 30 |

- **Hardware, threads, time.** Predict: x86 host CPU, 2 threads, torch 2.10.0+cu128; seconds row6 24.9, row011 5.6,
  sequence011 36.4, sequence021 216.8, dark18 5,651.5, sequence039 10,326.9
  (exp-186 `results/e170-final-2c3b8ac99/*-PREDICT.json`). exp-193 rebuilds: row011 1.1, row6 1.4, sequence011 1.2,
  sequence021 3.2, dark18 36.4, sequence039 192.7 min (E193 `REPORT.md:33-38, 68-70`). Fits: GPU0 TITAN RTX (E181
  `FIT.json`); the GPU1 model is not in the record (S0.4 gives two TITAN RTX on the host).

- **Check command.** `parity.py` (step 3); the `--reference --tol` replay of `predict_rank_186.py` against the saved
  mean; `build_final.py` step `<member>`; the pack rows, "Saved file: yes; SHA-256 PASS".

- **Differences from our run.** Path roots only for the predict. A fit refit is a GPU recipe refit. Two versions of
  the sequence039 launcher ran: `rebuild_fixed039_annual_061-pre-split-1511.py` (`cc582b0b`) ran first; the
  split version `rebuild_fixed039_annual_061.py` (`01919a87`) made the saved fits
  (exp-170 `results/annual061-pre-split-1511/`; E170 `SPEC-FIXED039-ANNUAL-061.md:5-7`).

## S9 MLP (mlp008 and mix12)

- **Purpose.** Part `mlp.parquet` with columns `mlp008` and `mix12`: the saved mlp-008 final fits and the mix12 bag
  (arms e80 and drop0; seeds 701-706) on the final inputs; 670,166 rows with a type and 624 rows without
  (exp-186 `results/mlp-final-e2bb55a2d/mlp.json`).

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `refit-src/experiments/exp-186-final-core/code/run_mlp_186.sh` | e2bb55a2d | `63c06e44b795541f` | n |
| `refit-src/experiments/exp-186-final-core/code/make_e164_work.sh`, `mlp_refs_186.py`, `part_mlp_186.py` | `release/REFIT-SOURCES.json` | `1cbeb4972c53f275`, `1c2c4eaf5f37a718`, `7ae2d713a6c394ad` | n |
| `experiments/exp-164-double-down/code/mlp008/prepare.py` | e2bb55a2d | `ccce0d660e0b24a5` | v |
| `experiments/exp-164-double-down/code/mlp008/io_utils.py` | e2bb55a2d | `00ca0ab9c38cdc6f` | v |
| `refit-src/experiments/exp-164-double-down/code/mlp008/apply_final186.py`, `apply_mix12_186.py`, `harness_final186.py` | `release/REFIT-SOURCES.json` | `e66b119376ff701c`, `270f4d843a8d5260`, `e18be58d59817060` | n |
| `refit-src/experiments/exp-164-double-down/code/seasonal/common.py`, `seasonal/fit.py` | `release/REFIT-SOURCES.json` | `release/REFIT-SOURCES.json` | n |
| fit code `mlp008/fit.py`, `fit_sweep.py`, `pack_level2_012.py` | `release/REFIT-SOURCES.json` | `1f528e927b490901`, `95bd3e4a15472bac`, `9cea733a082bf333` | y, y, n |
| learner path: exp-134 `fit_exp134.py`; exp-018 `dataset.py`; exp-153 `learner_wp.py`; exp-153 `pilot.py` | `release/REFIT-SOURCES.json` | `7001010d1ed37d31`, `4a1c3d6328d27824`, `9162b70e04507749`, `49621dd1e7fe6074` | v, v, n, n |
| stand tables: exp-127 `stand_geometry.csv`, `stand_geometry-final.csv` | `release/REFIT-SOURCES.json` | `2ce7ad6bf2b2c338`, `83f30cbbd2f4cc54` | y, n |

  Also in the export: `mlp008/mix012_vectors.py` `f5948b007e8434a3`.

- **Inputs** (M186:9).
  - `EXP148_KINPUT` = `${PRC_FINAL_ROOT}/finalcore/d/kinput-final` (months 2026-01 `6b6c0a04`, 02 `34002779`, 06
    `39158443`, 07 `9fe2b1a8`; `KINPUT-ROOT.json` `4820088d`).
  - `EXP159_INPUTS` = `${PRC_FINAL_ROOT}/finalclock/work-final/inputs` (frame12, track12; S10).
  - The route lp part `57bc0810b3da7655857f2566a083c6d05665da40a3229c2517880425131da74b` (S5).
  - Stand tables: the old one for the 2025 proxy months (`run_mlp_186.sh:51-52`), the final one for 2026 (S3).
  - The work tree from `make_e164_work.sh` (links into `${PRC_COMP_ROOT}/prc-exp164-work`). Pinned prefixes in
    exp-186 `results/mlp-final-e2bb55a2d/ROOT.txt`: mlp-007 `INPUTS.json` `7899899a`; mlp-008 final `INPUTS.json`
    `230f9eb8bbfc3a858231d1096c51d9cc46591946cf80f0923828e3ecea1a9b22`; mlp-008 `FIT.json` `22a2e877`;
    `BAG-mix.json` `fd2428f7`; e80 final `bd74a375`, seeds 703-706 `607d36a1`; drop0 final `4036d9c5`, seeds 703-706
    `1bfad8cb`.
  - The harness record `<root>/results/HARNESS-final.json`: our run read `1c297de150a0`; git holds
    exp-193 `inputs/HARNESS-final.json` `bebef61cd6eb4cf3` (see the differences).

- **Refit commands.**
  - Our call (M186:9):
    `env EXP148_KINPUT=<kinput root> EXP159_INPUTS=<finalclock work-final/inputs> EXP164_LP_PART=<lp> EXP164_LP_PART_SHA=57bc0810… EXP164_CPUS=<CPU list> bash experiments/exp-186-final-core/code/run_mlp_186.sh final <root of make_e164_work.sh> 2`.
    Its steps (`run_mlp_186.sh:68-86`): `mlp_refs_186.py --v73-parts <v73> --work <board> --out <refs>`;
    `mlp008/prepare.py --fit final`; `apply_final186.py --threads 2 --tol 0.05`; `apply_mix12_186.py --threads 2 --tol 0.05`;
    `part_mlp_186.py --out "$OUT" --commit "$COMMIT" --rebuild-tolerance …`;
    `parity.py --refs <refs>/mlp-refs.parquet --new "$OUT/mlp.parquet" --pair mlp008=saved:mlp008 --pair mix12=saved:mix12 --kind refit`.
  - The exp-193 form (T193, step `mlp`): `bash experiments/exp-186-final-core/code/make_e164_work.sh <work>/root "$PWD"`;
    `cp experiments/exp-193-final-build/inputs/HARNESS-final.json <work>/root/results/`;
    `bash experiments/exp-186-final-core/code/run_mlp_186.sh final <work>/root 2` with `EXP164_CPUS=8-15,72-79`.
  - The fits: mlp-008 final by `mlp008/fit.py --fit`, seeds 701 and 702, inputs `230f9eb8`; 40 epochs, batch 8192,
    hidden 512/512/256, SiLU, LayerNorm, AdamW lr 0.001, weight decay 0.0001, MSE, dropout 0.1, 2,083,537 rows,
    1,554 s (exp-164 `results/mlp-008/receipts/final-FIT.json` and the seed-701 record). mix12 by `fit_sweep.py`: e80
    (80 epochs) and drop0 (dropout 0, 40 epochs), weights 0.5 and 0.5 (`APPLY-mix12.json`; E164 REPORT-012.md).
    The launch commands are not in the record.

- **Outputs.** `${PRC_FINAL_ROOT}/finalcore/parts/mlp.parquet`
  `e48a88c6bc1c0a3868b4472aae8ab5de032e4ee435374e6c58a7b038891145e2`, 670,790 rows, columns `MVT_ID_mvt`, `mlp008`,
  `mix12`, code e2bb55a2d, written 2026-10-09T01:00:11Z; a second run (01:03-03:34 UTC) gave the same SHA-256 (M186:9).

- **Hardware, threads, time.** x86 host CPU, 2 threads (`APPLY-*.json`); prepare 3,153 s (T193); part step 8.3 s
  (`mlp.json`). The exp-193 rebuild: 162.04 min wall, 74 min running, `/usr/bin/python3`, part `e48a88c6`
  (exp-193 `results/run-mlp-20261009T200056.json`; E193 `REPORT.md:39`). Fits: CUDA 12.8, cores 24, 25, 88, 89, nice 19;
  the GPU model is not in the record.

- **Check command.** The `parity.py` pairs (`run_mlp_186.sh:85-86`); the apply steps stop beyond `--tol`;
  `build_final.py` step `mlp`; the pack row `ranking/primary_frame/mlp`, "Saved file: yes; SHA-256 PASS", part
  `e48a88c6bc1c`.

- **Differences from our run.**
  - The harness record. The part run read `1c297de150a0` (the x86 work tree). exp-193 read the git copy
    exp-193 `inputs/HARNESS-final.json` (`bebef61cd6eb4cf3`, commit 75d632e2c) and gave the same part `e48a88c6`. Use the
    git copy.
  - `EXP164_CPUS` must be the CPU list that the run may use (E186 `REPORT.md:225-232`).
  - A fit refit is a GPU recipe refit.

## S10 Clock head (HB3u)

- **Purpose.** Part `head.parquet` (`pred_HB3u`): the exp-159 clock head HB3u, merged from the HU18 boosters (seeds
  601-605) and the HU30s2 boosters (606-610), on the final frame12 rows (E193 `PACKET.md:380-450`).
  The same run builds the OPDI join (S2), frame12 and track12.

- **Source files.** `C` = `refit-src/experiments/exp-190-final-clock/code`; export state `n` unless named.

| file | commit that ran | SHA-256 |
|---|---|---|
| `C/run_final.sh` | 259409f3f; b12a4100b | `5ecc313b92a5bca3` |
| `C/run_chain.sh` | 259409f3f; b12a4100b; fe9e8a459 | `b8856e255f7ec558` |
| `C/after_rank.sh`, `read_chain.sh`, `day_count.py`, `common.py` | `release/REFIT-SOURCES.json` | `6aed08e2a0bd3b2b`, `31ec380a094d90b4`, `6f8cb4d25c903245`, `f96f36e3cc32b51a` |
| `C/fit.py`, `frame.py`, `frame12.py` (export y), `head_part.py` | `release/REFIT-SOURCES.json` | `43a7c6e8c7fb64c8`, `a86fc1ec9b27d83a`, `6c84cdcf9593dabc`, `3877bfea1e02513f` |
| `C/merge.py`, `opdi_final.py`, `pack.py`, `predict_head.py`, `track.py` | `release/REFIT-SOURCES.json` | `1d5c5af36b8ad5f2`, `5d2972d73b836bed`, `d2525352697344d9`, `6b81e124af39cf49`, `4e4f91641a75ebdb` |
| `C/actype_table.json`, `actype_table.py` | `release/REFIT-SOURCES.json` | `9b03b1762c731293`, `74e60f5f3f171ebc` |
| r5 builders: exp-159 `block12.py`, `features12.py`; exp-148 `features.py` (y), `features_r2.py` (y) | `release/REFIT-SOURCES.json` | `21e60813ee76eb90`, `80fa4c8acd997542`, `8e6bdb7f`, `283bcb77` |
| the source tree `prc-exp159-work/source` = ec40e740c, with `prc/data.py` (v) | ec40e740c | `8bc762676e3c6376` (data.py) |

- **Inputs.**
  - r4c (S4): `FR/inputs/r4c-final/features-r4c.parquet` `39466f229682c802f6f39204341dae58fae26164d82e9cb8b58d4b8629054302`
    and `proxy-features-r4c.parquet` `e445405e36b2220ace34c8ec0956738225b6dfe5d745d964207c0c79a9755ccf`.
  - OPDI (S2): 2026 `F/opdi/first-final/opdi-first.parquet` `753c3ee9…`; 2025
    `${PRC_OPDI_ROOT}/r2/inputs/complete/opdi-first.parquet` `84b5c5ea…`.
  - Harvest: `F/kinput` (2026-02 `.sha256` `f1b804a7`, 2026-06 `61a9aaee`) and `${PRC_COMP_ROOT}/prc-exp148-work/kinput`
    (`run_final.sh@259409f3f:13-23,31`).
  - features-r5: a link from `${PRC_COMP_ROOT}/prc-exp159-work/inputs/features-r5.parquet`
    `78d22fe292e1896f0889a09fc1445f1b754a7cb28d99c88b0004ab8999e577ef` (`run_chain.sh@259409f3f:66-68`).
  - `${PRC_FINAL_ROOT}/finalcore/raw` = `${PRC_DATA_DIR}` (`run_chain.sh@259409f3f:30`).
  - Boosters `${PRC_FINAL_ROOT}/finalclock/work-ranking/out/fits/{HU18,HU30s2}-jjrank.models` and the train reference
    `work-ranking/out/pred/HU18-jjrank.json` (`run_chain.sh@259409f3f:39,46`). Seed prefixes: 601 `19203a52`, 602
    `927bb351`, 603 `64f1c1b1`, 604 `e2245242`, 605 `f3e22684`, 606 `1ae77c60`, 607 `6352cd7e`, 608 `4edb66af`, 609
    `abcd8dbc`, 610 `f9ea04d1`; 1,800 trees (E190 `REPORT.md:211-230`).
  - The old stand table on all rows; the type table frozen, 31 of 201 types (E193 `PACKET.md:380-450`).

- **Refit commands** (M190:2-5). In our run the source step copied `prc-exp159-work/source` (ec40e740c)
  (`run_chain.sh@259409f3f:55-64`) and the interpreter was `${PRC_COMP_ROOT}/prc-exp148-src/.venv/bin/python`
  (`run_chain.sh@259409f3f:33`).
  1. Tables (our snapshot b12a4100b): `bash experiments/exp-190-final-clock/code/run_final.sh`. It runs the S2 join,
     then `exec bash run_chain.sh final` (`run_final.sh@259409f3f:28-32`), which runs `frame.py`; `frame12.py frame`;
     `EXP159_SET=9 track.py --procs 4`; `EXP159_SET=12 track.py --months 2025-04 2025-05 2025-09 --out .../track-r5.parquet --procs 3`;
     `frame12.py track` (`run_chain.sh@259409f3f:70-75`).
  2. Boosters (our snapshot fe9e8a459): `run_chain.sh ranking`, with
     `FIT = python fit.py --arm A4u --split jjrank --also-janjul --rounds 1800 --threads 4 --save-models --out $FITS`:
     `EXP159_SET=12 $FIT --seeds 601-605 --stage HU18`;
     `EXP159_SET=12 $FIT --seeds 606-610 --param min_child_samples=30 --stage HU30s2`; then merge and pack
     (`run_chain.sh@259409f3f:47,76-82`).
  3. Head (our snapshot 259409f3f): `after_rank.sh` runs `run_final.sh predict pmerge head`
     (`after_rank.sh@259409f3f:23`):
     `predict_head.py --models <fits> --out <pred> --threads 4 --train-ref <HU18-jjrank.json>`;
     `merge.py --dir <pred> --parts HU18 HU30s2 --stage HB3u --splits jjrank --mixed-settings`;
     `head_part.py --fits <work-final>/out/pred --out ${PRC_FINAL_ROOT}/finalclock/parts/head.parquet`
     (`run_chain.sh@259409f3f:83-88`).
  4. The exp-193 form (T193, step `head`): `bash experiments/exp-190-final-clock/code/run_final.sh source predict pmerge head`
     with `PRC_FINAL_ROOT`, `EXP190_WORK` and `EXP190_PARTS` set.
  5. Type table: `python experiments/exp-190-final-clock/code/actype_table.py <released frame12.parquet> <out.json>`
     (it needs the released frame12 `250fb20b`; usage text of `actype_table.py`).

- **Outputs** (M190:2-5).

| output | SHA-256 | rows | notes |
|---|---|---|---|
| `${PRC_FINAL_ROOT}/finalclock/inputs/opdi-first-all.parquet` | `4e6f2ee0…` | 2,755,837 | S2 |
| `${PRC_FINAL_ROOT}/finalclock/work-final/inputs/frame12.parquet` | `47b5b6bf58143f971c9ec81f8cc90f74baef5acb2d70ca94dad7e91af6741f3d` | 1,480,587; 130 columns | the 2025 rows equal the released `250fb20b` |
| `${PRC_FINAL_ROOT}/finalclock/work-final/inputs/track12.parquet` | `d748df61fd2d504df14387e5555cc8d8ae77ae0253e3f40b821257e829c56f68` | 19 columns | |
| `${PRC_FINAL_ROOT}/finalclock/parts/head.parquet` | `a0f28193f359d607cb23fc0852c7e3d06d644967a804c1bc220252f92edc5be6` | 670,790 | code 259409f3f; HB3u-jjrank `ae9b1c23`, `head.json` `55f8e622`; written 2026-10-08T21:07:09Z; 378,728 clock rows (January 91,415, February 79,545, June 102,949, July 104,819) |

- **Hardware, threads, time.** x86 host, Python 3.10.12, lightgbm 4.6.0, numpy 2.2.6, pandas 2.3.3, 4 threads,
  `OMP_WAIT_POLICY=PASSIVE`; the OPDI join at 1 thread (M190:2). The booster refit ran 2026-10-08 15:13-20:53 UTC
  (E190 `REPORT.md:92-127`). Seconds per seed: HU18 6,358.3, 1,701.2, 1,665.4, 1,706.4, 1,051.0; HU30s2 784.0,
  485.2, 506.4, 693.5, 1,136.5; 934,128 training rows, the 10 months of 2025 without January and July
  (exp-190 `results/ranking/fits-HU18-jjrank.json`). `head.json`: 5.4 s. The exp-193 rebuild: 6.6 min, PASS (E193
  `REPORT.md:32, 57-62`).

- **Check command.** `read_chain.sh final` (head parity against v73-parts, January and July) and `day_count.py`
  (`after_rank.sh@259409f3f:24-27`); `predict_head.py` stops when the train matrix differs (`19a52837a5a7`);
  `build_final.py` step `head`; the pack row, "Saved file: yes; SHA-256 PASS".

- **Differences from our run.**
  - The boosters. The released HB3u boosters are absent (E190 `REPORT.md:32-37`). The saved boosters are the
    4-thread ranking refit; the released fits used 8 threads. Seeds 601 and 606 are equal in bits to the released
    fits; the others differ by rms 13-25 s; the merged HB3u differs from the released jjrank by rms 5.14 s (E190
    `REPORT.md:92-127`). A booster refit at 4 threads on another CPU is a recipe refit.
  - A t8 head `9fb65fb4` (rms 4.88 s; M190:6) has no consumer (E193 `PACKET.md:380-450`; Appendix B).
  - `run_chain.sh` ran at three commits (fe9e8a459 boosters, b12a4100b tables, 259409f3f head) with one SHA-256
    `b8856e25`. The released fits ran exp-159 `common.py` `fa9ea5c8` and `fit.py` `5e10825c` (git 07dfbfe04;
    E190 PROVENANCE.md).

## S11 Rule frames (LIRF bundle, rule frame, exploit frame)

S11 reads only the route `np_base` (S6) and the S1 class files; it can run before S7.

- **Purpose.** Build the three LIRF frames on the final data from the route `np_base` and the clock input: the
  bundle (columns `rule`, `extension`, `np_v3cal` and others), the rule frame (`MVT_ID_mvt`, `month`, `cls`, `t`,
  `scope`, `g`, `g71`, `c71`) and the exploit frame (M187:3-5). February takes the January cells and June takes the
  July cells (`frames_v.py:44-71`, `SEASON`).

- **Source files.** `C` = `refit-src/experiments/exp-187-final-lirf/code`.

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `C/run_final_parts.sh` | c3e3c593f | `68334bcb308af2c1` | n |
| `C/clock_input.py` | c3e3c593f | `3a24fb41145b1e7f` | n |
| `C/frames_v.py` (the `npee84b9f3` parts) | c3e3c593f | `6d6afaf128bb4241` | n |
| `C/frames_v.py` (the `a444` bundle) | f094d1aa0 | `d1f84c02f2ddb216` | n |
| `C/noplan_v3cal.py` | 34df39e94; c3e3c593f; d6160347f; f094d1aa0 | `4219adbf230115b5` | n |
| `C/np_design_held.py`, `np_r1_weight_held.py` | 8b300baef, f094d1aa0; f094d1aa0 | `deb1ce9596c60370`, `914a277fbffc8a1f` | n |
| `C/pack.py`, `parts_parity.sh` | `release/REFIT-SOURCES.json` | `d44668c053eda1ba`, `29970119919cade5` | n |
| exp-167 helpers (`common`, `noplan_form*`, `practice_arm_r1`, `regime_signal` and others) | as in the export | as in the export | y |
| exp-169 `assembly`, `c73`, `exploit029_rules`, `exploit_rules`, `read_stacks`, `stack9`, `stand_floor` | as in the export | as in the export | y |
| read by `ast` (`frames_v.py:103-109`): exp-160 `exploit_links_014.py`, exp-169 `prepare_unit2.py`, `stack9-024/u13_inbound.py`, `u27_floor3.py` | as in the export | as in the export | y |

- **Inputs.**
  - `np_base` `FR/parts/np_base.parquet` `ee84b9f3…` (S6; M187:3).
  - The class files `F/class-final/2026-MM/2026-MM-v25.parquet` (S1): January `e1744f698ca77f7f…`, February
    `6232183edd13f330…`, June `df03331bc1a04f3a…`, July `d6693ea00090ddf7…` (M187:3).
  - `${PRC_DATA_DIR}/final/final_ranking.parquet` `4b0692a792846079…`; key `submitting` `b3445d80e29bf599…`
    (exp-187 `out/v/final-npee84b9f3/frames-record.json`).
  - Data files in git (exp-187 `code/GITPATHS.txt`; prefixes from git at c3e3c593f): exp-167
    `results/noplan-form-v3-002.json` `1969d692edd89a2c`, `results/practice-arm-r1-001.json` `6dbd7b96f5329c9f`;
    exp-169 `inputs/exploit-029/C58-statistics.json` `d85cfade5f560818`; exp-187
    `out/v/ranking-pack-digests.json` `0d061a6439db17c5`; submissions/candidates/exp159-fold-F-HB1-w50-x.parquet
    `a5572156bf548dd1` (`noplan_v3cal.py:38`; read only on the ranking parity path, `frames_v.py:261-267, 335`).
  - The weight a = 0.444 from `np_r1_weight_held.py` (Mac) on 229 held rows (January 22 rows, a 0.439; July 207 rows,
    a 0.444) (exp-187 `out/v/np-r1-weight-held.json`; exp-187 `REPORT-V.md:260-288`). Its inputs: exp-167
    `noplan-form-v3-fold-001.parquet` `5b6855d93e35b5ba…` (in git); exp-169 `inputs/stack9-024/frame.parquet`
    `473a6aac051cafda…` and the stack10-025 R4mksFjjrefit-month files 2025-01 `d103398cfee7a669…` and 2025-07
    `142baf8b6b07a0e5…` (not in git; S16.2).

- **Refit commands.**
  1. `bash experiments/exp-187-final-lirf/code/run_final_parts.sh npee84b9f3 FR/parts/np_base.parquet <January class> <February class> <June class> <July class>`
     (M187:3). With `PY=${PYTHON:-python3}` it runs (`run_final_parts.sh:11-20`):
     - `python C/clock_input.py --class-file <each of the four> --out FL/inputs/clock-npee84b9f3.parquet`;
     - `python C/frames_v.py --mode final --clock <clock> --np-base <np_base> --fill none --parts --tag npee84b9f3`;
     - `bash C/parts_parity.sh FL/parts npee84b9f3`.
  2. Weight fit (Mac): `OMP_NUM_THREADS=2 python experiments/exp-187-final-lirf/code/np_r1_weight_held.py`.
  3. The a444 bundle (M187:6): `PRC_FINAL_ROOT=${PRC_FINAL_ROOT} OMP_NUM_THREADS=2 python experiments/exp-187-final-lirf/code/frames_v.py --mode final --clock FL/inputs/clock-npee84b9f3.parquet --np-base FR/parts/np_base.parquet --fill none --np-r1-a 0.444 --parts --tag npee84b9f3-a444`.

- **Outputs.** In `FL/parts/` unless named.

| output | SHA-256 | rows | source |
|---|---|---|---|
| `FL/inputs/clock-npee84b9f3.parquet` | `06a62c1674f6d4333e07b68db928d609c7abcd28e61f29fb0b36f34f72913438` | clock rows per month 91,415; 79,545; 102,949; 104,819 | M187:3; `out/v/final-npee84b9f3/clock-record.json` |
| `frames_bundle-npee84b9f3` (replaced by the a444 bundle) | `a890789ea25a21b7601d6b255bb2d9bff06fc828732683fc58697cddb2d57446` | 670,790 | M187:3 |
| `frames_rule_frame-npee84b9f3` | `f225b96cea8a33860f47b99f5ebc3c5ef8464d4d4cd832cf9b4e85dedb822f14` | 670,790 | M187:4 |
| `frames_exploit_frame-npee84b9f3` | `886db355f2f91a4c7bcc8478b795ce931901764411eb78e5d1c145464841a2ea` | 670,790 | M187:5 |
| `frames_bundle-npee84b9f3-a444` (the one the pack uses) | `47d737e114efa5bc9c7c94a55c9569ae6ecd0e18b08d77fc10e590bd98973c68` | 670,790 | M187:6 |

  The pack uses the a444 bundle, the rule frame and the exploit frame
  (`release/ZABD-PACK-final-r9.json` lines 1479, 1506, 1525).

- **Hardware, threads, time.** The npee84b9f3 parts: x86 host, 2 threads, 07:10 to 07:13 UTC on 2026-10-09 (exp-187
  `REPORT-V.md:15-20, 152-172`; M187:3); 2.5 min at 2 threads (T193:291-302); `python3` from `PATH` with pandas 2.3.3
  and pyarrow 24.0.0 (T193:297). The a444 bundle: Mac arm64, 2 threads, 2026-10-09T07:36Z (M187:6).

- **Check command.** exp-193 steps `frames` (T193:291-303) and `frames-a444` (T193:304-315; it first runs
  `sha256sum -c` on the clock file `06a62c16…`): PASS 3 of 3 and PASS (R193:98-99). `parts_parity.sh` runs exp-186
  `parity.py --kind refit` against `${PRC_FINAL_ROOT}/finalcore/refs/v73-parts.parquet`. Saved column digests of the
  release: `ranking/rule_frame` `cdc7af6fc35760b3…`, `ranking/bundle` `617dadb6c53147d3…` (`rule`
  `429f1d528c818073…`, `extension` `89af2c58150be8f7…`, `np_v3cal` `f871cc240117044a…`), `ranking/exploit_frame`
  `0758a078b2c44a94…` (release/saved-vectors/final/…; pack lines 1479-1525). Replay script:
  `release/assembly_replay.py` (as S6).

- **Differences from our run.** `frames_v.py` ran in two versions for the final parts: c3e3c593f (`6d6afaf1`)
  wrote the rule frame, the exploit frame and the replaced bundle; f094d1aa0 (`d1f84c02`, equal to HEAD) wrote
  the a444 bundle. exp-193 rebuilt all four with `d1f84c02` and got the same SHA-256 (R193:98-99), so the refit tree
  needs only `d1f84c02`. The a444 bundle is a Mac build; the same Mac build gave the x86 SHA-256 of the rule frame and
  the exploit frame (exp-187 `REPORT-V.md:260-288`). The C58 least-squares fit differs between the Mac and the x86 host
  by 4.3e-12; since a921ee1ef the code uses saved coefficients (`REPORT-V.md:128-140`). The fold frames stay saved
  (S16).

## S12 to S15: shared facts

- `M192`, `M194`, `M191`: the `MANIFEST.tsv` of exp-192, exp-194, exp-191.
- Python for S12 to S14 on the x86 host: Python 3.12.13 with the packages of S0.3, row 2 (our environment
  `${PRC_FINAL_ROOT}/finalv/venv312`; `final-2.sh:8`). The s7 GPU fits used the checkout `.venv` (torch) and
  `/usr/bin/python3` (S13).
- `${PRC_DATA_DIR}` on the x86 host: `${PRC_FINAL_ROOT}/finalcore/raw` (`final-2.sh:9`; `run_part.sh:33-34`;
  `setup_run.sh:19-21`).
- The source export of the assembly: `${PRC_FINAL_ROOT}/finalplan/export-d07bf4a0a621`, the export 513030db at
  d07bf4a0a. Each build copies it first (exp-192 `code/export_copy.py`).
  In the refit tree, `${PRC_SOURCE_ROOT}` holds the same files.

## S12 Assembly and R9

- **Purpose.** Build zabd (the Z chain), v64 = zabd + T11b-L, the Z and R files and the ZABD pack from the member
  parts (exp-192 build final-2). R9 then puts mix12 on all primary member-scope rows (fold and 2026) and builds zabd,
  the Z file and the pack again (exp-194 `r9_build.py` docstring; exp-194 `REPORT-R9.md:85-106`). The result is
  the entry `zabd-final-r9`.

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| exp-192 `code/build_final.sh` | 70d3707d7 | `aa935777ffef47cd` | n |
| exp-192 `code/assemble192.py` | 046b08bea; 70d3707d7 | `a10a587c57d34318` | n |
| exp-192 `code/frames192.py`, `post192.py`, `export_copy.py` | `release/REFIT-SOURCES.json` | `47d02c60a0d25324`, `4a8560f2bd820820`, `2ce7fb90c60ffcde` | n |
| exp-192 `code/final_sources.py` | 696d473de `ee6abb185fe3f5b6`; our reading is that this version wrote `final-sources-2.json` (see the differences) | see the commit column | n |
| exp-192 `code/preflight_parts.py`, `standin192.py`, `pack_check.py` | 70d3707d7; `release/REFIT-SOURCES.json` | `61c84521ebfc78e6`, `6b1844ecfd93bc8c`, `b1c5cb7efd48ef34` | n |
| exp-169 assembly unit: the patched copies exp-192 `export/experiments/exp-169-stack-plan/code/*` (`assemble_final_032` `9452f5df`, `dark_twin035` `faacd912`, `final_months169` `6f1ea4b5` and others) | as in the export | as in the export | y |
| the patch `refit-src/experiments/exp-186-final-core/pins/E-final-months.patch` | see exp-192 PROVENANCE.md | `935d0c9b` | n |
| exp-169 main code at d07bf4a0a | d07bf4a0a | as in the export | y |
| `export/release/assembly_replay.py` | as in the export | `779def01ad37e773` | y |
| `release/t11b_stage.py`; `refit-src/release/ZABD-PACK.json` | d07bf4a0a; as in the export | `8eb2204009038f27`; `3b52d708d4338d21` | y |
| exp-194 `code/r9_build.py`, `assemble_r9.py`, `r9_chain.sh` | 046b08bea; `release/REFIT-SOURCES.json` | `1f39032d71a4b013`, `e4b2c28f1b220158`, `adee2e5a04c16a0a` | n |
| checks: exp-186 `check_final_candidate.py`, `parity.py` | `release/REFIT-SOURCES.json` | `beda972bbdcc44da`, `0ac593fd` | n |

- **Inputs.**
  - exp-192 `results/final/final-sources-2.json` `4ec85c78cf1c085f` (git); with
    `final-sources-2-from.json`: ref 18be8b8cf, 43 needed columns, 43 with a part (exp-192 `REPORT.md:383-386`).
  - The member parts (M192:2, inputs; paths in `final-sources-2.json`): head `a0f28193f359d607`, dark18 `86c049d0`,
    g `8758a971`, mlp `e48a88c6bc1c0a38`, row011 `399f6987`, row6 `56c60492`, sequence011 `33652831`, sequence021
    `c21db6dc`, sequence039 `636ab22a`, frames_bundle-npee84b9f3-a444 `47d737e1`, exploit frame `886db355`, rule frame
    `f225b96c`, F `66829b3a`, route `232d2670` (S6 to S11).
  - The R9 fold mix12 exp-164 `results/stack-pack-012/mlp-fold.parquet`
    `fe4558cc807f4128552ac2d54e0aa27e769be3a6b84cadcee6124c6bb188494a` (`assemble_r9.py` `VEC`; M194:8; S16.3).
  - `${PRC_DATA_DIR}`; the parity reference `${PRC_FINAL_ROOT}/finalcore/refs/v73-parts.parquet` (`8af18086…`, S7).
  - The saved fold frames of the pack (S16).

- **Refit commands.**
  1. final-2 (exp-192 `results/final/final-2.sh` lines 9-17):
     ```
     export PRC_DATA_DIR=${PRC_DATA_DIR} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
     cd ${PRC_SOURCE_ROOT}
     python -I experiments/exp-192-final-assembly/code/preflight_parts.py ${PRC_SOURCE_ROOT} <runs>/final-sources-2.json <runs>/preflight-2.json
     grep '"all_ok": true' <runs>/preflight-2.json
     bash experiments/exp-192-final-assembly/code/build_final.sh ${PRC_SOURCE_ROOT} <runs>/final-sources-2.json final-2
     ```
     `build_final.sh:30-53` runs `export_copy.py`;
     `assemble192.py --mode final --repo ${PRC_SOURCE_ROOT} --export ${PRC_SOURCE_EXPORT} --sources <sources> --parts-dir <parts> --pack-out ${PRC_FINAL_ROOT}/finalplan/export-final-2 --out <runs>/final-2.json`;
     `pack_check.py` (with `PRC_RANKING_MODE=final`); `check_final_candidate.py --stages` on the Z file and the R file;
     `parity.py --pair part=saved:zabd-final --kind refit`; `sha256sum`.
  2. R9 (M194:8-11; `r9_chain.sh:46-63`; our call `bash experiments/exp-194-gain-lane/code/r9_chain.sh f1`), with
     `G=${PRC_FINAL_ROOT}/gainlane/r9-f1` and `SRCX=${PRC_FINAL_ROOT}/finalplan/export-d07bf4a0a621`:
     ```
     export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
     python -I experiments/exp-192-final-assembly/code/export_copy.py ${PRC_SOURCE_EXPORT} $G/export-r9 ${PRC_SOURCE_ROOT}
     python -I experiments/exp-194-gain-lane/code/r9_build.py --variant r9 --repo ${PRC_SOURCE_ROOT} --export ${PRC_SOURCE_EXPORT} \
       --sources <final-sources-2.json> --parts-dir $G/parts/r9 --pack-out $G/export-r9 --out $G/runs/r9.json \
       --fold-out $G/parts/r9/fold-stages.npz
     ```
     Then `pack_check.py`, the two checkers and `parity.py` as in final-2.

- **Outputs.** 670,790 rows each.

| output | SHA-256 | source |
|---|---|---|
| final-2 `${PRC_FINAL_ROOT}/finalplan/parts/final-2/zabd.parquet` | `e6503136921882b66dfb822775273e2134fd153801f83082e58d71906fa845d4` | M192:2 |
| final-2 `r`, `zabd-stages`, `r-stages`, Z file, R file | `b138d580`, `9617371e`, `d1677cf5`, `e47b8afb`, `5a1278f3` | M192:3-8 |
| final-2 pack `${PRC_FINAL_ROOT}/finalplan/export-final-2/release/ZABD-PACK.json` | `29c0ace1a5c30d981b962d042db77fbf828a52b5de8d8c194a72a8b996ea3317` | M192 |
| R9 `$G/parts/r9/zabd.parquet` (columns `MVT_ID_mvt`, `zabd`, `v64`) | `1124b91ca9b6db67bab00dd3dd23b3092e158d4ec3ac07b2be7c0a0c2eba60b2` | M194:8 |
| R9 `zabd-stages`, Z file | `cde8a7d1aa634493`, `0b00c4b752a5bd93` | M194:9-10 |
| R9 pack `$G/export-r9/release/ZABD-PACK.json` | `50091dbc5c5c5e104ba42222f0c325d59869ecad39335a53451c907427733f14` | M194:11 |

  R9 changes 179,173 ordinary fold rows and 0 clock rows of 2026 (`REPORT-R9.md:85-106`). T11b on build 2: 8,453 P2
  rows, 7,497 changed, 21 long taxi days (exp-192 REPORT.md; the same counts at `release/ENTRIES.json` lines 640-659).

- **Hardware, threads, time.** x86 host, OMP 2, BLAS 1. final-2 ran 07:45-07:48 UTC on 2026-10-09 (exp-192
  `REPORT.md:383-386`), about 3 min per build (line 387). R9: base 11:35:51-11:38:37, r9 11:38:37-11:41:23 UTC on
  2026-10-10, 2.4 GB, no GPU (`REPORT-R9.md:49-58`).

- **Check command.** Replay:
  `PRC_RANKING_MODE=final python scripts/build_entry.py --target zabd-final-r9 --data-root <data root> --out-dir <dir> --record <json>`
  On the x86 host: 27.37 s (exp-193 `results/release-x86-9970755a2f70/SOURCE.json`).
  The entry SHA-256 is at `release/ENTRIES.json` line 637 and line 659. Build checks: `build_final.sh:37-50`.

- **Differences from our run.**
  - The assembly gives equal bytes at OMP 2 and BLAS 1 on x86-64 (exp-192 `REPORT.md:388`). Across
    machines it moves by 1e-9 s (exp-193 `PACKET.md:420`, design row 7). zabd-final-r9 on the Mac differs in the last bits,
    6.0e-12 s (exp-193 `REPORT.md:626-634`).
  - The R9 base variant gave the final-2 bytes `e6503136`, `9617371e`, `e47b8afb` and `29c0ace1`; R9 ran once
    (M194:8).
  - `r9_chain.sh` has fixed private paths (lines 18-31); the refit sets them from the roots.
  - `final_sources.py`: the build snapshot holds `8afd7914` (70d3707d7). The build does not
    run `final_sources.py`; it reads `final-sources-2.json`. Commit 696d473de (2026-10-09 07:44:16 UTC) adds
    `final-sources-2.json` with the version `ee6abb18`, and M192 records the argument `--lirf-tag bundle=npee84b9f3-a444,…`,
    which only `ee6abb18` takes. Our reading: `ee6abb18` wrote the sources file; `refit-src/` ships that version (696d473de).
  - The design rows 1 to 7: exp-192 `SPEC.md:25-36`; exp-193 `PACKET.md:414-420`; the mlp row choice
    `PACKET.md:421`. Row 3 (`PACKET.md:416`): w = (N_final(1,a,z) + N_final(2,a,z)) / N_final / count_held(1,a,z); July
    with June the same.

## S13 V (c5, s7, L2C, ORD-S, LD)

- **Purpose.** Make V from v64: c5 (EXT, LONG, LIRF, NOAC, ROLL), the w2 base, L2C candidate A (the s7 deviation as
  an input), ORD-S (`--names s7 --lirf-jul 0.5`) and LD (`--long-days`). The default V is `final-v-r9-ld`
  `31b09989…` with the 15-seed s7 member. The 9-seed V `final-iii-r9-ld` `260b427b…` stays in the release; its
  steps are the same, with the 9-seed member.

- **Source files.** `C` = `refit-src/experiments/exp-191-final-v/code`.

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `C/v_run.sh` | 046b08bea; 4d8a6532a; a0c484bab | `d3355c828d9c227a` | n |
| `C/final_run.sh`, `v_tail.sh`, `setup_run.sh`, `v_part.py`, `stages_record.py` | `release/REFIT-SOURCES.json` | `8995596fb25b985a`, `a89e697e40f766be`, `8d9590d015627002`, `a1249d1d9f70098a`, `2565485dea04c1df` | n |
| `C/chain_ids.py` | as in the export | `21278f57df7c026f` | y |
| `C/iso/`: `candidate`, `chain`, `dump_chain`, `features`, `finalmode`, `roll_step`, `score` | as in the export | `65ba56e30aafa013`, `c3bca4935c7616d5`, `cc411d597d600c66`, `ca96fc42e65f84af`, `3cd4d8c2525e116f`, `9ed97f61ed1d7024`, `4b4cfaa5d0816e7f` | y |
| `C/w2/`: `base`, `feat2`, `l2`, `l2c`, `ords`, `out` | as in the export | `ae1de2a55c792b65`, `7140010c5f18859e`, `6e6ccbc458854bae`, `c9cd2869063d5534`, `d2b5c7cd1a6e52d9`, `45dbb8e9ee35211d` | y |
| s7: `C/s7_inputs.sh`, `s7_fit_queue.sh`, `s7_finish.sh`, `s7n_finish.sh`, `s7_part.py`, `port_89_chain.sh` | c2657cca1; d62cb5858; 1168757cc; `release/REFIT-SOURCES.json`; `release/REFIT-SOURCES.json`; 4d8a6532a | `f2d90855e81eaba1`, `841b0fb405f4f6c2`, `1e23d9470e68a21f`, `ad0551ec9d1e1fcc`, `443f024149ca43e1`, `5e1d498a38ee2345` | n |
| `C/w2/seq/prepare.py`, `fit.py`, `export.py` | as in the export; d62cb5858; as in the export | `01f593bf70f82e24`, `d6c9458f3c1febbd`, `421273f5d255a3d8` | y |
| s7 seed fits 4-9 (isolated worker tree `${PRC_ISO5_ROOT}`; `w5/seqfit.py` runs the released `w2/seq/fit.py`) | none | `w2/seq/fit.py` `d6c9458f` | - |
| LD: exp-194 `code/ords_ld.py`, `ld_chain.sh` | fa170fcd0 | `1da4f3a09fd4b26b`, `4163ba0b15b27773` | y, n |
| candidate A: `refit-src/experiments/exp-191-final-v/isolated5-A/l2c.diff` (patches `l2c.py` `c9cd2869` to `c9931e0a`, `v_run.sh:34-36`) | see M194:12-15 | `9e72607ad896dc09` | the export ships the result as `w2/l2c_candidate_a.py` `c9931e0a6e2192b2` |
| 15-seed: exp-191 `code/s7x15_chain.sh` (seeds 10-15, the member, runs iv and v); exp-194 `code/r9_chain15.sh` (V chain on R9, LD) | 46a2f4ab7; a6365d75d | `0cb5d47e8ea36a1f`, `5bec5ba4b11b5d4f` | n |

- **Inputs** (M194:12-27).
  - s7 (15 seeds, the default): the tree `${PRC_FINAL_ROOT}/finalv/run-s7x15` (15 held and 15 full seed files; seeds
    1-9 are hard links of `run-s7x9`), npz `${PRC_FINAL_ROOT}/finalv/run-port15-46a2f4ab7/work/w2/s7c-final.npz`
    `92aa25c8…`; member files `s7_fold` `40bdf34b…`, `s7_ranking` `f2995bc3…`; part s7c `b44afa46…` (exp-191
    REPORT.md section 10, lines 421-470).
  - The R9 pack `50091dbc`; the Z stages record `${PRC_FINAL_ROOT}/gainlane/r9-f1/parts/r9/z-file-stages-record.json`
    `1672ef0853572cb0`; the head `a0f28193`.
  - s7 (9 seeds): `${PRC_FINAL_ROOT}/finalv/run-port9-a0c484bab/work/w2/s7c-final.npz`
    `505ceafd1059287e674777654dbe62269b1880770b3512b1abdb8a61cb59fc0b`; the s7 run `${PRC_FINAL_ROOT}/finalv/run-s7x9`
    (`s7_fold` `7060e1604900decd`, `s7_ranking` `3614ed1e94c5331c`). chain_ids: fold `47a0128f337c04ff`, ranking
    `fd895855236c8931`.
  - Check only (`v_run.sh:51-53`): `${PRC_FINAL_ROOT}/finalv/refs-finallirf-final-a444`,
    `${PRC_FINAL_ROOT}/finalv/run-ranking-cf74f4261`, the head.
  - The s7 member: 16 seq7 input files (for example `meta.json` `a18eae08`, `ids.npy` `a425d4f9`; exp-191
    `results/port/seq7-and-seeds-sha256-20261010.txt`); the seed files of seeds 1-3 (M191:2) and 4-9 (exp-191
    `isolated5-A/REPORT-4.md:25-67`).

- **Refit commands.** `$R` is the run tree that `setup_run.sh` makes.
  1. s7 inputs (`s7_inputs.sh:11-15`, 2 threads): `python chain_ids.py . --official fold,ranking`;
     `python iso/features.py months`; `python iso/features.py ranking`; `python w2/feat2.py`;
     `python w2/seq/prepare.py --nb 64 --win 10800 --out work/seq7`.
  2. s7 fits, modes `full` and `held`, seeds 1-15 (`s7_fit_queue.sh:28-30`; seeds 10-15 with the same command,
     only `--seed` and `--out` changed, `s7x15_chain.sh:62-64`):
     `PRC_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python w2/seq/fit.py --data work/seq7 --d 192 --layers 4 --epochs 16 --mode $mode --seed $seed --amp --out work/seq7/s7-$mode-$seed.npy`.
  3. The s7 member, N = 15 (`s7x15_chain.sh`, step 3): `sh experiments/exp-191-final-v/code/s7n_finish.sh <python> <s7 tree> <parts> <refs> 15`
     (N = 9 for the 9-seed member, `port_89_chain.sh:26`); it runs `s7_part.py` (`s7n_finish.sh:19`), which calls
     `w2/seq/export.py s7 work/seq7`. The export reads the seed files in sorted name order and takes the equal-weight
     mean of each mode (exp-191 REPORT.md section 10).
  4. The V chain on R9 (`r9_chain15.sh:66-72` for the 15-seed V; `r9_chain.sh:66-73` for the 9-seed V):
     ```
     PRC_FINAL_ROOT=$G/root-r9 PRC_V73_REFS=${PRC_FINAL_ROOT}/finalcore/refs/v73-parts.parquet \
     PRC_V_L2C_DIFF=${PRC_SOURCE_ROOT}/experiments/exp-191-final-v/isolated5-A/l2c.diff PRC_V_L2C_S7=<s7 npz> \
     sh ${PRC_SOURCE_ROOT}/experiments/exp-191-final-v/code/v_run.sh ${PRC_SOURCE_ROOT} <python> $G/export-r9 $G/run-r9 \
       <s7 tree: run-s7x15 or run-s7x9> <LIRF refs> <ranking run> <R9 z-file-stages-record.json> <head> <head SHA-256>
     ```
     Its steps (exp-193 `results/release-x86-9970755a2f70/final-iii-r9-ld.json`): `iso/dump_chain.py`, `chain_ids.py`,
     `iso/features.py` (two times), `iso/candidate.py --steps ext,long,lirf,noac,roll --out outputs/iso-c5.parquet`,
     `w2/base.py`, `w2/l2c.py --s7 work/w2/s7c-l2c.npz --out outputs/w2-l2c.parquet`, ORD-S.
  5. The LD tail, in a hard-link copy of the R9 run tree (`r9_chain15.sh:115-141`, run `run-ld15`, for the 15-seed V;
     `ld_chain.sh:67-83`, our job `bash ld_chain.sh b <code dir>`, for the 9-seed V; M194:17-18, 25-26):
     ```
     export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PRC_RANKING_MODE=final
     python w2/ords_ld.py --on work/w2/l2c --names s7 --lirf-jul 0.5 --long-days --save-npz work/w2/s7c-final.npz --out outputs/w2-l2c-s7c.parquet
     python <C>/stages_record.py . outputs/stages-v.json
     python <C>/v_part.py . $ROOT/finalv/parts $ROOT/finalv/youthful-giraffe_final.parquet
     ```

- **Outputs.** 670,790 rows each unless named.

| output | SHA-256 | source |
|---|---|---|
| s7c9 member | `22279068ff78a2e1` | M191:6 |
| s7c9 held fold (344,419 rows) | `7060e160` | M191:7 |
| c5-r9 `$G/run-r9/outputs/iso-c5.parquet` | `0f8280246f02162d` | M194:12 |
| l2c-r9 | `d71244b4f41911bb` | M194:13 |
| v-r9 part; v-r9 file | `614ce19f2c8b8fed`; `1c1ed67bdd7e5418` | M194:14-15 |
| v-ld part | `4b46443f6a2bcf61` | M194:17 |
| v-ld file `${PRC_FINAL_ROOT}/gainlane/ld-b/root-ld/finalv/youthful-giraffe_final.parquet` (entry `final-iii-r9-ld`, 9 seeds) | `260b427b83f99921b26ce372bad78fbcb3718b402e7a8bd0725a914ae3da114d` | M194:18 |
| s7c15 member: part s7c; fold; ranking | `b44afa46…`; `40bdf34b…`; `f2995bc3…` | exp-191 REPORT.md section 10 |
| l2c-r9-15 | `b09ec11091a8b020` | M194:21 |
| v-r9-15 part; v-r9-15 file (entry `final-v-r9`) | `ac132019504f250c`; `18a7bfb77bfdc5a6` | M194:22-23 |
| v-ld15 part | `54abc4ae276cea1c` | M194:25 |
| **v-ld15 file** `${PRC_FINAL_ROOT}/gainlane/r9-15-a/root-ld15/finalv/youthful-giraffe_final.parquet` (entry `final-v-r9-ld`, the default V) | `31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759` | M194:26 |

  LD moves 6,753 rows on the 9-seed V (exp-193 `REPORT.md:696-708`); the held LD step is +3.16 [−0.20, +8.37]
  (M194:18), and +3.08 [−0.19, +7.95] on the 15-seed V (M194:26).

- **The two s7 seed sets.**
  - 15 seeds (s7x15; LD V `31b09989`; dh `eea300c0`): the release default.
    `release/ENTRIES.json` line 2 `default_target "final-v-dh-r9-ld"`; `final-v-r9-ld` at line 1182, `seeds_per_mode 15`
    line 1229, `l2c_seeds 9` line 1251, SHA-256 line 1271. Registered in release commit 781388a80 (s7x15 delta review
    PASS); the default moved in 0f5b3265e.
  - 9 seeds (s7c9; LD V `260b427b`; dh `07d4a4e6`): the default from release commit 51e85f25b (manifest 32b8cfac1;
    exp-193 `REPORT.md:719-722`; exp-193 `code/ld_entries.py:42-48`) to 0f5b3265e; `final-iii-r9-ld` at 0f5b3265e
    `release/ENTRIES.json` line 791 (`seeds_per_mode 9` line 838, `l2c_seeds 9` line 860).
  - The 15-seed facts: npz `${PRC_FINAL_ROOT}/finalv/run-port15-46a2f4ab7/work/w2/s7c-final.npz` `92aa25c815248bd2`;
    `s7_fold` `40bdf34b6ebfcd57`, `s7_ranking` `f2995bc391d8ba3d`. Chain at 757fbb271 (`r9_chain15.sh a`); c5 = c5-r9;
    l2c-r9-15 `b09ec11091a8b020`; v-r9-15 file `18a7bfb77bfdc5a6`; v-ld15 part `54abc4ae276cea1c`; v-ld15 file
    `31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759`; ld15-dh
    `eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02` (M194:21-27). Held LD-15 +3.08 [−0.19, +7.95]
    (M194:26). Fits 07:49:47 UTC on 2026-10-10 to 02:38:52 UTC on 2026-10-11; held 4,267-4,327 s, full 5,133-5,262 s
    per seed; peak 10.30-10.38 GB (exp-191 `REPORT.md:421-470`). Chain job 02:50:41-03:03:37 UTC on 2026-10-11,
    3.19 GB (exp-194 `REPORT-R9.md:313-373`).

- **Hardware, threads, time.** V: x86 host, OMP 2, BLAS 2, L2C at NT=8 (`final_run.sh:10`). The R9 V chain ran
  11:51:47-13:03:32 UTC on 2026-10-10 with a pause of the host from 12:00:21 to 13:01:51; 3.2 GB (`REPORT-R9.md:49-58`).
  The LD job ld-b ran 14:36:29-14:39:29 UTC on 2026-10-10 (exp-194 `REPORT-LD.md:50`). s7 fits: GPU 1, TITAN RTX 24 GB
  (docs/CLOUD-COMPUTE.md line 19; exp-191 `REPORT.md:63-102`); per seed full 5,304-5,806 s, held 4,490-4,557 s
  (exp-191 `PORT-S7X9.md:130`); seeds 1-3 and 10-15 the checkout `.venv`, Python 3.10.12, torch 2.10.0+cu128
  (exp-191 `REPORT.md:22-28` and section 10); seeds 4-9 `/usr/bin/python3`, Python 3.10.12, torch 2.10.0+cu128
  from the user site packages (a check of the host on 2026-10-11 and the isolated worker's report; the logs print no versions). s7 inputs: 13:54:39-14:06:38 UTC
  (exp-191 `REPORT.md:33-35`; date not given); the exp-193 rebuild 06:40:31-06:45:31 UTC on 2026-10-09, 2 threads
  (exp-193 `REPORT.md:44-45`).

- **Check command.**
  `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/build_entry.py --target final-v-r9-ld --data-root <data root> --out-dir <work>/v-check --record <work>/v-check.json`;
  expected `31b09989`. On the x86 host: 8 min 59 s, 2.76 GB (section "Release targets final-v-r9-ld, final-v-r9, …" above, "Inputs and command").
  The 9-seed V: `--target final-iii-r9-ld`
  expected `260b427b` (exp-193 `code/ld_entries.py:43`). On the x86 host: 8 min 33 s,
  2.76 GB (exp-193 `REPORT.md:709-717`; `SOURCE.json` 8:32.97, max_rss 2,898,296 kB). The replay fits LONG, NOAC, ROLL,
  L2C (9 LightGBM seeds) and ORD-S again from the saved s7 values (`final-iii-r9-ld.json`). Checker:
  `ld_chain.sh:121-122`.

- **Differences from our run.**
  - The s7 GPU fits use AMP and do not give the same bits again: a seed-1 refit is rms 48.9 s from the full seed and
    51.0 s from the held seed (exp-193 `PACKET.md:85-89`; `REPORT.md:46-47`). A refit gives new hashes from L2C on;
    compare the held quality.
  - Seeds 4-9 ran in another tree (`${PRC_ISO5_ROOT}`) with `/usr/bin/python3`; the fit code is the same file.
  - L2C is `l2c.py` with the patch applied at run time; the release ships the patched file.
  - The R9 base variant gave the c5 `2f0130a8`, L2C `72c3e909`, part `dd89caf0` and V `6de02372` of run (iii)
    (M194:12-15; M191:12-13). Jobs ld-a and ld-b gave equal hashes; the tail without the flag gave the f1 bytes
    (M194:17-18).

## S14 DH and the dh stage

- **Purpose.** On 14,587 EDDF typed no-head dark_b rows (January 1,557, February 12,374, June 656): value =
  max(60, V + w (pred_DH − V)), w = 0.164 (`release/ENTRIES.json` lines 895-898; exp-193 `ld_entries.py:140`; exp-193
  `PACKET.md:435-439`). On the 15-seed V `31b09989` the result is the default entry `final-v-dh-r9-ld` `eea300c0`;
  on the 9-seed V `260b427b` it is `final-iii-dh-r9-ld` `07d4a4e6`. The DH part is the same for both.

- **Source files.** `C` = `refit-src/experiments/exp-194-gain-lane/code`.

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| `C/dh_dark_rows.py`, `dark_values.py`, `run_dark_cpu.sh` | c6faf428d; d01e1757a; `release/REFIT-SOURCES.json` | `064eb5322fc7d9f4`, `db8c18033b758af2`, `ec9a816707dcc750` | n |
| `C/dh_fit.py` (the x86 copy `gainlane/code-dh`) | 2f11c76bc; 73e2fce46 | `6e78f8c37e8b0921` | n |
| `C/run_dh.sh`, `run_dho.sh`, `dh_opdi_cov.py` | 73e2fce46; dc58be8b1; `release/REFIT-SOURCES.json` | `e72e361ec060365f`, `313a192c4908334d`, `0d188f5f03a752e8` | n |
| `C/exp159-at-fit/common.py`, `frame.py` (the bytes that the fit imported; main exp-159 `common.py` `b95da6cd`, `frame.py` `a86fc1ec`) | dc023184e | `dc92b51c2266960e`, `5ba004effb041a30` | y |
| `C/dh_apply.py`, `run_part.sh`, `dh_input_cov.py`, `part_diff.py` | 2f11c76bc; `release/REFIT-SOURCES.json` | `343c9e9cf9c08252`, `c265e0a482ab3441`, `156c7c49ac06451e`, `3596e9f6fab02415` | n |
| `C/dh_read_nostand.py`, `dh_read_iso.py`, `dh_read.py` (the reads that chose w) | `release/REFIT-SOURCES.json`; cb18f1903; `release/REFIT-SOURCES.json` | `a67dfa24dc17231e`, `c3c55136462e9d89`, `73346f7da6c96371` | n |
| dark helpers `l_dark`, `mlp_dark`, `r4mk_dark`, `r4mk_iso`, `x170_dark` (`r4mk_dark` and `r4mk_iso` need `PRC_R2_NO_FEATURES`, S0.1) | `release/REFIT-SOURCES.json` | `release/REFIT-SOURCES.json` | n |
| `C/r9_dh.py`, `r9_dh_stages.py` | 046b08bea; fa170fcd0 | `9ec11199d47d2294`, `b2dae30753567d20` | n |
| `release/dh_stage.py` (the replay) | release | `689bb6f41ddf4627` | y |

- **Inputs.**
  - Part (M194:3; `run_part.sh:27-36`): `FR/inputs/r4c-final/features-r4c.parquet` `39466f229682c802`; the head
    `${PRC_FINAL_ROOT}/finalclock/parts/head.parquet` `a0f28193`; `F/opdi/first-final/opdi-first.parquet`
    `753c3ee9bdd6eb29`; `${PRC_DATA_DIR}/final` (final_ranking `4b0692a7`, final_submitting `b3445d80`); dark rows
    `78f9dbbb4daf4591`; the weights exp-194 `out/part-final1/dh-final1-weights.json`
    `d36d7aa8b9ddc431fb1190b40cce5d169f2d599d6bff4f9beee603c593ec1ec5` (scope "EDDF:0.164", cells EDDF months 1, 2,
    6, 7); the V part check `871b7e041810375a` (`run_part.sh:36`).
  - Fit (M194:5-7): split jjrank, token `4abce89100ca3bad`, 934,128 clock rows of 2025-02 to 2025-06 and 2025-08 to
    2025-12; features_r4c `${PRC_COMP_ROOT}/prc-exp148-work/inputs/features-r4c.parquet` `9101b06ae74eb0e0`;
    features_r5 `${PRC_COMP_ROOT}/prc-exp159-work/inputs/features-r5.parquet` `78d22fe292e1896f`;
    `${PRC_OPDI_ROOT}/r2/inputs/complete/opdi-first.parquet` `84b5c5ea3957ba1e`;
    `refit-src/experiments/exp-194-gain-lane/out/opdi_cov_2026-o1.json` `6c853f2c0f9a6a43` (git).
  - The step: the V file `31b09989` (default; `260b427b` for the 9-seed entry) (S13); the part dh-final1 `a89366d1`.

- **Refit commands.**
  1. Dark rows (M194:4; `run_dark_cpu.sh:72-73`, step `flags`):
     `python -I experiments/exp-194-gain-lane/code/dh_dark_rows.py --features FR/inputs/r4c-final/features-r4c.parquet --features-sha 39466f22… --head ${PRC_FINAL_ROOT}/finalclock/parts/head.parquet --head-sha a0f28193… --raw ${PRC_DATA_DIR}/final --out ${PRC_FINAL_ROOT}/gainlane/dark-o2/dark_rows`.
  2. Fit (M194:5-7; `run_dho.sh o1` calls `run_dh.sh o1`; only jjrank saves models). `run_dho.sh` first runs
     `dh_opdi_cov.py` and a `--smoke 20000` jjx-02 run.
     ```
     cd ${PRC_FINAL_ROOT}/gainlane
     EXP159_SET=12 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -I code-dh/dh_fit.py --c159 <exp-159 code folder> \
       --split jjrank --seeds 601-603 --threads 2 --out dh-o1 --save-models --opdi-cov <opdi_cov_2026-o1.json>
     ```
     Our run gave `--c159 ${PRC_COMP_ROOT}/prc-exp159-work/code`. The exp-193 rebuild gave
     `--c159 ${PRC_SOURCE_ROOT}/experiments/exp-194-gain-lane/code/exp159-at-fit` (same bytes; exp-193
     `results/rebuild-dh-f73cbd8557aa/`, run `run-dh-fit`). Use the second form.
  3. Part (M194:3): `FLAG=dark_b DARK_ROWS=<dark_rows.parquet> bash ${PRC_SOURCE_ROOT}/experiments/exp-194-gain-lane/code/run_part.sh final1 o1 EDDF:0.164 0.164 ${PRC_SOURCE_ROOT}`.
     It writes the weights file, runs `dh_input_cov.py`,
     `dh_apply.py --fit-dir <fit root>/dh-o1 --stage DHo … --threads 2 --dark-rows … --out $G/parts/dh-final1.parquet`
     (`run_part.sh:56`), `part_diff.py` (line 62) and `parity.py`.
  4. The step (M194:27 with `r9_chain15.sh`, run `dh-ld15`, for the default; M194:19 with `ld_chain.sh:124-126` for
     the 9-seed entry):
     `python -I experiments/exp-194-gain-lane/code/r9_dh.py --v-file <v-ld file> --part <dh-final1> --raw ${PRC_DATA_DIR}/final --out-dir $G/dh-ld`,
     then `r9_dh_stages.py` and `check_final_candidate.py` (line 127).
  5. The weight w = 0.164: fitted on January and July 2025 with V out of sample; parity value 0.1639176053329452
    (exp-194 `REPORT.md:30-32`; `out/dh_read_nostand-o2-darkL.json`, `out/dh_read_iso-i1-darkL.json`). It goes into
    `run_part.sh` as an argument.

- **Outputs.**

| output | SHA-256 | rows | source |
|---|---|---|---|
| dark rows `${PRC_FINAL_ROOT}/gainlane/dark-o2/dark_rows.parquet` | `78f9dbbb` | 1,342,082 | M194:4 |
| boosters `${PRC_FINAL_ROOT}/gainlane/dh-o1/DHo-jjrank.models/seed-60{1,2,3}.txt` (1,800 trees, 95 inputs) | `011407c146a340c3`, `90ac6615730d4ab9`, `9fe96e455b312229` | 40,118,802; 40,119,929; 40,124,119 bytes | M194:5-7; exp-193 rebuild `run-dh-fit` |
| fit record | `4ec4b2e193d82429` | - | M194:3 |
| dh-final1 `${PRC_FINAL_ROOT}/gainlane/parts/dh-final1.parquet` (`MVT_ID_mvt`, `pred_DH`, `has_head`, `w_pair`) | `a89366d17a80ee751fb23b7331d9c6ff907896c8bf12a59f6bc7702d11d1ff4f` | 670,790 | M194:3 |
| **ld15-dh file** `${PRC_FINAL_ROOT}/gainlane/r9-15-a/dh-ld15/youthful-giraffe_final.parquet` (entry `final-v-dh-r9-ld`, the default) | `eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02` | 670,790 | M194:27; checker PASS (exp-194 `out/r9-15-a/dh-ld15/dh-check.json`) |
| ld-dh file `${PRC_FINAL_ROOT}/gainlane/ld-b/dh-ld/youthful-giraffe_final.parquet` (entry `final-iii-dh-r9-ld`) | `07d4a4e6344f5c681761a00a851d23dd9dd6bc8675861179b0de80b4fe669865` | 670,790 | M194:19; value_sha `b5c91ae7` (`release/ENTRIES.json` line 922); records exp-194 `out/ld-b/dh-ld/{dh-record,dh-check,dh-file-stages-record}.json` |

- **Hardware, threads, time.** x86 host, 2 threads, LightGBM 4.7.0, Python 3.12.13, numpy 2.4.6, `deterministic`,
  `force_row_wise`, the seeds in order in one process (M194:5-7). The exp-193 rebuild from code at f73cbd8557aa ran
  16:40:34-17:50:36 UTC on 2026-10-09 (exp-193 `REPORT.md:315-326`); its run records give dark rows 0.13 min, fit
  49.15 min (16:43:25-17:32:34), part 12.93 min, final dh 0.03 min. The LD dh step: 0.98 s on the x86 host
  (`SOURCE.json`).

- **Check command.**
  `python scripts/build_entry.py --data-root <data root> --out-dir <work>/final-check --record <work>/final-check.json`
  (no `--target`: the default `final-v-dh-r9-ld`); expected `eea300c0` (section "Release target final-v-dh-r9-ld" above).
  The 9-seed entry: `--target final-iii-dh-r9-ld`
  expected `07d4a4e6` (exp-193 `ld_entries.py:44`). It runs `release/dh_stage.py`: the
  official file and code hashes; the saved V (`dh_stage.py:73`); the decoded `pred_DH` `d2ed372b`, `has_head`
  `ed730bcb`, `w_pair` `7e115f54` (lines 77-79); the part bytes (82-83); the scope (87-90); the formula (91-93). Mac:
  2 s, 0.3 GB (section "Release target final-v-dh-r9-ld" above). The exp-193 rebuild compared all three boosters by SHA-256: PASS.

- **Differences from our run.**
  - Our fit ran from the x86 copy `gainlane/code-dh/` and read the exp-159 code in `prc-exp159-work/code`. The rebuild
    used the repo copy `exp159-at-fit` (same SHA-256; `exp159-at-fit/README.md`) and gave the same boosters.
  - exp-159 `common.py` reads the x86 work trees by fixed paths (`COMP=`, a private path); the refit sets
    `${PRC_COMP_ROOT}` there.
  - `r9_dh.py` made the dh files `eea300c0` and `07d4a4e6`; the release replays them with `dh_stage.py`. Neither
    version of exp-193 `final_dh.py` made them: `cc2b7a1f` (f73cbd855) made `2fe6c56b`, `f739c19c` (a4d1b5290) made
    `c4511650` (Appendix B).

## S15 Entry registration

- **Purpose.** Check the files of the final chain (run folder `gainlane`) by SHA-256 on the Mac and write the saved vectors, the R9 pack and the
  entries `zabd-final-r9`, `v64-final-r9`, `final-iii-r9`, `final-iii-dh-r9`, `final-iii-r9-ld` and
  `final-iii-dh-r9-ld` into the release tree (exp-193 `r9_entries.py:3-19`; `ld_entries.py:4-11`). Then the
  15-seed set: the saved 15-seed member values, the saved V values and the entries `final-v`, `final-v-dh`,
  `final-v-r9`, `final-v-dh-r9`, `final-v-r9-ld` and `final-v-dh-r9-ld`, with `default_target`
  `final-v-dh-r9-ld` (exp-193 `code/s15_entries.py`, docstring lines 1-25; release commit 781388a80).

- **Source files.**

| file | commit that ran | SHA-256 | export |
|---|---|---|---|
| exp-193 `code/r9_entries.py` | 20cba3274 `b6a9c7bd7bf94bc3`, the version of the write run (see the differences) | see the commit column | n |
| exp-193 `code/r9_pack_text.py`, `ld_entries.py`, `targets.json` | 0f982e499; 8eaa7bbae; `release/REFIT-SOURCES.json` | `26069e33f7acbd7d`, `792f2734fc346632`, `0b0f3168` | n |
| exp-193 `code/s15_entries.py` (the 15-seed entries) | cbd92d78f, after the run (the write run used `e76bec1b`, rebuilt) | `c3b01e06876d36b5` | n |
| exp-193 `code/final_dh.py` (the `final-v-dh` file `833533ad`) | cbd92d78f, after the run (the run recorded `61efbb25`); this version gives `833533ad` too | `1bd302c14bcc9bba` | n |
| `scripts/build_entry.py`, `release/ENTRIES.json` (925 lines), `release/dh_stage.py`, `release/v73_stage.py`, `release/ZABD-PACK-final-r9.json` | release | `cd4c2c2d47b8a972`, `434a52207bf614cc`, `689bb6f41ddf4627`, `5724277f395916a2`, `10547e756f77b849` | y |

- **Inputs.** `r9_entries.py:43-50`: the pack `50091dbc`, the fold mlp column `5e83a292`, the ranking mlp column
  `20a7dc72`, zabd `1124b91c`, V `1c1ed67b`, dh `4ac66707`, the x86 c5 and l2c, `FINAL_PACK` `29c0ace1`.
  `ld_entries.py:42-48`: V `260b427b`, dh `07d4a4e6`, `ORDS_LD` `1da4f3a0`, `ORDS` `d2b5c7cd`, closure commit
  fa170fcd0; c5 `0f828024`, l2c `d71244b4` (lines 10-11). Our run copied them from the x86 host with `gget.sh`
  (`r9_entries.py:6-19`).

- **Refit commands** (Mac, as our run gave them).
  - `PRC_RANKING_MODE=final OMP_NUM_THREADS=2 python experiments/exp-193-final-build/code/r9_entries.py --release <release tree> --inputs <folder of the r9-f1 files> --data-root <folder with data/raw> --closure <code commit> [--write]`
    (`r9_entries.py:3-4`).
  - `PRC_RANKING_MODE=final OMP_NUM_THREADS=2 python experiments/exp-193-final-build/code/ld_entries.py --release <release tree> --inputs <folder of the ld-b files> --ords-ld experiments/exp-194-gain-lane/code/ords_ld.py --data-root <folder with data/raw> [--write]`
    (`ld_entries.py:4-6`).

- **Outputs, 15-seed set** (`release/ENTRIES.json`): line 2 `default_target
  "final-v-dh-r9-ld"`; `final-v` line 924, `final-v-dh` 1012, `final-v-r9` 1053, `final-v-dh-r9` 1141, `final-v-r9-ld`
  1182 (SHA-256 `31b09989` line 1271), `final-v-dh-r9-ld` 1274 (SHA-256 `eea300c0` line 1312). The saved member values
  release/saved-vectors/s7c15/final-\{fold,ranking\}.json and the saved V values
  release/saved-vectors/final-entries/s15/v-\{v,r9,r9-ld\}.json (781388a80). Release bf4cccf4b (after 0f5b3265e)
  changes the 43 provenance strings of the two final packs and the `pack_sha256` of four earlier entries; the target
  file stays `eea300c0`.

- **Outputs, 9-seed set** (`release/ENTRIES.json`; in export 32b8cfac1, line 2 was `default_target "final-iii-dh-r9-ld"`).
  `zabd-final-r9` line 623 (pack `136b3f87`, `10547e75` before release commit bf4cccf4b; SHA-256 line 637, `d26dadd0`); `v64-final-r9` line 640 (SHA-256 line 659, `3884f905`);
  `final-iii-r9` line 662 (closure 046b08bea, line 686); `final-iii-dh-r9` line 750; `final-iii-r9-ld` line 791
  (closure fa170fcd0, line 815; SHA-256 line 880; value_sha `f61d4755`); `final-iii-dh-r9-ld` line 883 (part_sha line
  896, w line 897, scope_rows 14,587 line 898, SHA-256 `07d4a4e6` line 921). The s7 member: fold
  `s7c9/final-fold.json` `f1befc3f` (file `7060e160`, 339,046 finite of 344,419 rows); ranking
  `s7c9/final-ranking.json` `25360d39` (file `3614ed1e`, 661,713 finite).

- **Hardware, threads, time.** Mac arm64, 2 threads. `r9_entries.py --write` at 13:30 UTC on 2026-10-10;
  `r9_pack_text.py` at 14:03 UTC (pack `18d0a13c` to `10547e75`) (exp-193 `results/r9-entries-mac/README.md:8-9`);
  `ld_entries.py --write` at 15:22:47 UTC (exp-193 `results/ld-entries-mac/README.md`). The x86 replays at 9970755a2,
  15:43:31-16:24:51 UTC on 2026-10-10, all PASS (exp-193 `REPORT.md:709-717`).

- **Check command.** The x86 replays at 9d94e0386 (05:11:01-05:38:01 UTC on 2026-10-11) of the six 15-seed targets,
  each PASS; `final-v-r9-ld` 8 min 59 s, peak 2.76 GB (exp-193 `REPORT.md:766-772`, cbd92d78f; records
  `results/release-x86-9d94e0386b74/`). The Mac checks of the 15-seed set: `results/s15-entries-mac/` (REPORT
  lines 755-764). The 9-seed default before: the Mac replay PASS `07d4a4e6`
  (exp-193 `results/ld-entries-mac/mac-replay-final-iii-dh-r9-ld.json`); the H8 checker
  (`final-iii-dh-r9-ld-check.json`). x86 records: exp-193 `results/release-x86-9970755a2f70/`.

- **Differences from our run.**
  - `ld_entries.py:100` (committed) asserts that the default stays `final-iii-dh-r9`; release commit 51e85f25b moved
    it to `final-iii-dh-r9-ld` (exp-193 `REPORT.md:719-722`). cbd92d78f changes the asserts of `ld_entries.py`,
    `r9_entries.py` and `s15_entries.py` to "default_target names an entry" (exp-193 `REPORT.md:786-787`;
    `results/b1-items-mac/b1_entries_assert.py`). No run used these versions.
  - `s15_entries.py` and a check run of `final_dh.py` ran (04:52:06 and 04:49:58 UTC on 2026-10-11) before the
    bytes that cbd92d78f holds; cbd92d78f `final_dh.py` gives the same `833533ad`.
  - The release reached the target in steps: d91cff63c (R9), d1c6061d6 and 9970755a2 (LD) (exp-193
    `REPORT.md:626-634, 696-717`).
  - `r9_entries.py`: the write run at 13:30 UTC used the version committed in 20cba3274 (`b6a9c7bd`; exp-193
    `REPORT.md:626`). 0f982e499 (`cda727c0`) adds the pack mode-text change that `r9_pack_text.py` made at
    14:03; its docstring names pack `10547e75`. No record shows a run of `cda727c0`.

## S16 Held side: the saved fold inputs and `refit-pins/`

- **Purpose.** The held rows (January and July 2025, 344,419 rows, the `fold` split of the pack) fit the blend
  weights and the rule statistics of S12-S14. The final build reads them as saved values; this section tells where
  they came from and which old file versions rebuild them.

- **What the final build reads.** `release/ZABD-PACK-final-r9.json` holds 14 fold frames. Its
  `final_build.provenance` marks 46 fold columns "pack saved", 4 "rule" (`fold/primary_frame:member_scope`,
  `fold/pair_frame:member_scope`, `fold/pair_frame:control_scope`, `fold/exposure:weight`) and 1 R9 column
  (`fold/primary_frame:mlp`: "R9: held mix12 on non-clock member-scope rows (exp-194 r9_build.py)"). The design
  keeps the released values on the fold side (exp-193 PACKET.md, design rows "Fold side"). The "pack saved"
  values are the zabd fold values of export d07bf4a0a621.

### S16.1 Fold frames and their source files

The pack field `source_sha256` is the SHA-256 of the file that the frame was taken from. The exp-169 records name
that file. The saved values ship in `release/saved-vectors/zabd/fold/` (pack column field `file`).

| frame | columns | `source_sha256` | source file (record) | made by |
|---|---|---|---|---|
| `fold/bundle` | MVT_ID_mvt, F, R4mk, rule, extension, lo, hi, airport, month, noplan, np_base, cls, np_v3cal, practice, practice_weight, practice_point, stand_floor, c73, c73_weight, c73_point | `3a647e38283bd6c39b243d4acae197e3e44af95939efb6b16c5a2811436c0bf7` | exp-169 `inputs/stack10-025/bundle-fold.parquet` | first bundle `read_stacks.py` (a690f9ec3), R4mk swap `stack10.py` + `route_swap.py` (31b83a31a); full trace in exp-194 `b1-recovery/held-route/TRACE.md` |
| `fold/primary_frame` | MVT_ID_mvt, month, cls, member_scope, mlp | `84ed780b1d716f26e091387a3a8b7ca24d53129be8d0d9e8ccb404f3616e7a72` | exp-169 `inputs/stack12-028/matrix-fold.parquet` (`results/final-route-before-030.json`) | exp-169 unit 028; `mlp` replaced by R9 (S16.3) |
| `fold/pair_frame` | MVT_ID_mvt, member_scope, control_scope | `a2bb3d7f908d908e435fb9700e861b7c55aad48ee5189265f14d5997ab3549e4` | exp-169 `inputs/stack3-007/members-fold.parquet` (`ARTIFACTS.json`) | exp-169 unit 007; both columns by rule in the final build |
| `fold/rule_frame` | MVT_ID_mvt, cls, y, F, t, ap, mon, rw, scope, fit, g, g71, c71 | `473a6aac051cafdac6c0bfa9660f528a5a17cc642bd80f27ffc20ce34173d3dd` | exp-169 `inputs/stack9-024/frame.parquet` (`results/final-route-before-030.json`) | exp-169 unit 024 |
| `fold/exploit_frame` | MVT_ID_mvt, C40, C58, C63, C64, C66, C68, C71, order_delta, point58, point66, lower, upper | `e9970001804f8a2b71d6908c358b9388ddcebc95f8f1852ec03773623eddac7e` | exp-169 `inputs/exploit-029/rules-fold.parquet` (`results/final-route-before-030.json`) | exp-169 unit 029 |
| `fold/exposure` | MVT_ID_mvt, weight | `449fe8dc8e458235eeae543f932f350ab7345f65d6f5e8514208ad85539330fd` | exp-169 `inputs/final-harness-032/fold-exposure.parquet` (`results/dark-twin-035-build/RECEIPT.json`) | the final build writes the weights by rule: the four-month mix of exp-192 design row 3 (`release/saved-vectors/final/fold/exposure`) |
| `fold/vector/g` | AB | `01bbbbb5a46bd5ee11b736eb15e31646409185116802757db98f683d99465942` | `${PRC_COMP_ROOT}/prc-exp181-work/out/sequence-g-bag-read-033/BAGS.parquet`, column AB (exp-169 `inputs/read-032b/MANIFEST.json`) | g held fits (S7 recipe, held mode) |
| `fold/vector/sequence021` | pred_seq170 | `24ae475f76d236ce595ed697c7d364d3a53daf0f884b0ea04d3b369ba7fcc033` | `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/fixed-sequence021-bag6-held-060/jjrefit.parquet` (read-032b MANIFEST) | S8 recipe, held mode |
| `fold/vector/sequence039` | pred_seq170 | `3473b0d760e4679dd8f1d31bde945dab190a07c93cc4a13cac7c37e4775da6a1` | `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/fixed-close30-057/jjrefit.parquet` (read-032b MANIFEST) | S8 recipe, held mode |
| `fold/vector/row6` | pred_seq170 | `6d04e4465201c7e1aece2211a016dcb6e62b65c0b8905309308cea0cbb53c209` | `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/fixed-control6-bag6-held-060/jjrefit.parquet` (read-032b MANIFEST) | S8 recipe, held mode |
| `fold/vector/sequence011` | pred_seq170 | `cbe1ac3b9de1a3e04dd742f48ba9d7bf00d69b2c8227d57af48265d240258e73` | `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/fixed-fallback_sequence011-17001-held-060/jjrefit.parquet` (read-032b MANIFEST) | S8 recipe, held mode |
| `fold/vector/row011` | pred_seq170 | `0fa9fca6863d2cd0581457dd9697d14a2c5fa349c58b018de256751f722ed4d4` | `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/fixed-control6-17001-held-060/jjrefit.parquet` (read-032b MANIFEST) | S8 recipe, held mode |
| `fold/vector/head` | pred_HB3u | `a13ec08170987c2652b4de82d7ead96767f04effe18b4e92dd3c05e0e3ba76a1` | `${PRC_COMP_ROOT}/prc-exp159-work/out/pack-HB3u/full/HB3u-jjrefit.parquet` (exp-169 `ARTIFACTS.json`, copy `inputs/stack6-021/HB3u-jjrefit.parquet`) | HB3u held fits (exp-159 `fit.py`, `merge.py`, `pack.py`; split jjrefit) |
| `fold/vector/dark18` | pred_seq170 | `573a641e796104361e6f1aff1d450712c96de444fd4b6399085ddf1b2f1be017` | `${PRC_COMP_ROOT}/prc-data-challenge-2026/experiments/exp-170-diverse/out/dark18-Z-063/jjrefit.parquet` (copy under exp-169 `inputs/dark-twin-035/remote/`; `results/dark-twin-035-build/RECEIPT.json`) | S8 dark18 recipe, held mode |

The exp-170 `out/` folders are in the main checkout of the x86 host (`${PRC_COMP_ROOT}/prc-data-challenge-2026`); git
holds none of these files.

The member models, data and compute of each held fit are in `docs/LINEAGE-ZABD.md` "Members" (code
paths, training rows, CPU or GPU). The held member fits train without January and July ("held: fits without January
and July"; route level 2: "1,540,605 out-of-fold rows of nine 2025 months", CPU 16 threads, 4.4 h).

### S16.2 Rebuild of the held F (the route of the fold bundle)

`F`, `R4mk` and `rule` of `fold/bundle` were traced to code in git with their recorded hashes (all 19 rows of
exp-194 `b1-recovery/MANIFEST.tsv` match):

1. `refs.parquet` (v22, v23, v24, v32 reference vectors, `a07552f7…`, 344,419 rows): `experiments/exp-148-short-route/code/refs.py`
   at 72799c21a (`92ecf9f4…`), record `refs.json` (`7dc8cb0a…`, copy in `b1-recovery/held-route/prc-exp148-work/inputs/`).
   Command (docstring of refs.py): `PRC_SOURCE_ROOT=$S $S/.venv/bin/python experiments/exp-148-short-route/code/refs.py --out DIR`.
2. R4 month refits `R4Fjjrefit-month-2025-{01,07}.parquet` (`0095135f…`, `42dee7eb…`): exp-148 `fold_serve.py`
   `4a8dd063…` with `stacker.py` `71a1381a…` (both at 3ba3625a3); records on branch exp148-short-route,
   exp-148 `results/fold/R4Fjjrefit-month-2025-{01,07}.json` (stage R4, written 2026-10-01).
   Unit `held-r4f`; set `PRC_R2_NO_FEATURES`.
3. First bundle `inputs/bundle-fold.parquet` (`3a697a17…`): `prepare_noplan.py` then `read_stacks.py`
   (exp-169 `ARTIFACTS.json`), with `F = .5*where(keep, v32, R4 pred_Sb) + .5*v32`,
   `keep = noac | (ordinary & ~aobt)` (`read_stacks.py:65-73`) and the label-free `rule` (`read_stacks.py:76-77`).
4. R4mks refits `R4mksFjjrefit-month-2025-{01,07}.parquet` (`d103398c…`, `142baf8b…`): exp-148 `fold_serve.py`
   `6b37360e…` (ec40e740c) with `stacker.py` `2a37769e…` (b7fee8fba); records in
   `refit-src/experiments/exp-169-stack-plan/inputs/stack10-025/`. Unit `held-r4mks`; set `PRC_R2_NO_FEATURES`.
5. R4mk swap (unit 025): `stack10.py` `prepare()` with `route_swap.py` `make_route` (31b83a31a); only R4mk changes
   (340,306 rows); the bundle becomes `3a647e38…`, the pack's source.

The bundle files between the first and the last (`unit2/`, `unit2-ready/`, `c55-006/` = `stack3-007/`,
`stack5-016/`) carry F, rule, np_base, extension and noplan unchanged.

### S16.3 The R9 fold column

`fold/primary_frame:mlp` is the held mix12 vector on the non-clock member-scope rows:
`refit-src/experiments/exp-194-gain-lane/code/assemble_r9.py` reads
exp-164 `results/stack-pack-012/mlp-fold.parquet`, SHA-256
`fe4558cc807f4128552ac2d54e0aa27e769be3a6b84cadcee6124c6bb188494a` (`assemble_r9.py:35-36`), written by exp-164
`code/mlp008/pack_level2_012.py`. The mix12 held fits read
the held MLP inputs of S16.4.

### S16.4 `refit-pins/` units

Each unit holds the version of a file that one step ran, where the final chain ran another version at the same path.
Lay a unit over the refit tree for that step only: `python release/make_refit_tree.py <out dir> --unit <unit>`. The
file `refit-pins/<unit>/README.md` of each unit names its step and its files.

| unit | step | files (SHA-256 that ran) | record |
|---|---|---|---|
| `mlp-held-source` | the held mlp-007 inputs (prepare-03); then `repair_physical.py` (refit-src) writes the 178-name physical list | exp-164 `code/mlp007/prepare.py` `e46066759e983f99…` (rebuilt byte-exact), exp-018 `dataset.py` `be5d9207…`, exp-134 `fit_exp134.py` `ef861cee…`, `prc/data.py` `e613518e…` | exp-164 `results/mlp-007/input-records/INPUTS.json` line 109 |
| `held-r4f` | the R4 month refits of F (S16.2 step 2) | exp-148 `fold_serve.py` `4a8dd063…`, `stacker.py` `71a1381a…` (3ba3625a3); `prc/data.py` `471218c5…` | exp-148 `results/fold/R4Fjjrefit-month-2025-{01,07}.json` (branch exp148-short-route) |
| `held-r4mks` | the R4mks month refits of R4mk (S16.2 step 4) | exp-148 `fold_serve.py` `6b37360e…` (ec40e740c), `stacker.py` `2a37769e…` (b7fee8fba); `prc/data.py` `471218c5…` | `refit-src/experiments/exp-169-stack-plan/inputs/stack10-025/` (`R4mksFjjrefit-month-2025-{01,07}.json`) |
| `held-bundle` | the first fold bundle (S16.2 step 3) | exp-169 `assembly.py` `0e4536c4…`; `prc/data.py` `e613518e…` (a690f9ec3) | exp-169 `results/stack-reads.json` |
| `held-swap` | the R4mk swap (S16.2 step 5) | exp-169 `route_swap.py` `a65b66b8…`, `lane_c_swap.py` `fc90a4cf…`, `stack6.py` `61a0d690…`; `prc/data.py` `e613518e…` (31b83a31a) | exp-194 `b1-recovery/held-route/TRACE.md` |
| `dh-2fe6c56b` | the dh file `2fe6c56b` of `final-dh` (S14; final side) | exp-193 `final_dh.py` `cc2b7a1f…` (f73cbd855) | S14 |
| `mlp008-inputs` | the mlp-008 final inputs `receipts/final-INPUTS.json` `230f9eb8` (S9; final side) | exp-164 `mlp008/prepare.py` `75ea88c9…`, `io_utils.py` `e420c53f…` (cff672a79); shipped as `79c4622f…` and `ed6973d0…` after the release rewrite and the path rules (the inputs read `PRC_COMP_ROOT`) | S9 |

Other held-side files are in `refit-src/` at their own paths, or the export holds them with only their release
rewrite lines changed:

- exp-148 `refs.py` `92ecf9f4…`, exp-169 `read_stacks.py` `2615e3bb…` and `stack10.py` `7cc6bc37…`: the export
  copies differ from them only in their release rewrite lines (`release/REWRITE-LINES.json`); `prepare_noplan.py`
  `00b5322e…` is in `refit-src/`.
- The held MLP harness: `refit-src/experiments/exp-164-double-down/results/member-002/source/` (`learner_wp.py`
  `d15b9431…`, `data-own.py` `e613518e…`, shipped as `69732391…` after the path rules, `scored_pilot.py`
  `f578b178…`). The final-mode inputs need the final harness (`9162b70e…`, exp-193
  `inputs/HARNESS-final.json:19`). Keep the two apart.
- The held MLP input packages carry their own code copies (`INPUTS.json:100-105`): exp-130 `fit.py` `df9ae2c5…`,
  `source/exp018/dataset.py` `be5d9207…`, `features.py` `ae591d7b…`, `model.py` `5ed75a51…`; exp-134
  `fit_exp134.py` `ef861ceeea06cb13…` (b289887a8), `proxy_features.py` `106450ef…`. These are inputs (the saved
  Kaggle packages of S1); a rebuild of the packages needs these versions. Unit `mlp-held-source` holds `ef861cee` and
  `be5d9207`.
- `mlp007/io_utils.py` `e420c53f…` (export `y`); seasonal `common.py` `29fe8669…` (in `refit-src/`).
- The g upstream bundle: `refit-src/experiments/exp-150-model-a/r4/admission/upstream/` (`MANIFEST.json` and 14
  copies of the exp-148 and exp-147 files of bb23613f7, with the release path and label rules applied; `MANIFEST.json`
  holds the shipped SHA-256 values). `r4_admission_runtime.py` `load()` (lines 27-32) checks the SHA-256 of each copy
  at that path. The exp-193 g rebuild (S7) passed from a main snapshot, so the final g path does not need them; the
  held g inputs (exp-150) do.
- exp-155 `gate_check.py` `c93ff0fe…` and the 12 other files of its code folder: in `refit-src/` (S2).

- **Inputs.** The official 2025 training months (`${PRC_DATA_DIR}`), the S1-S4 inputs in their old (ranking) form,
  the exp-146 frozen rows (`build_frozen.py`; `PRC_VALIDATION_DIR`), and the held labels of January and July 2025.
- **Refit command.** No record holds one command from the raw files to the saved fold values. The candidate chain:
  `python scripts/build_v22.py --plan` for the earlier base, then sections 3 to 9 of "Earlier entries" below for the
  later packages and the exp-148 R4 steps. S16.2 gives the commands that the records name for F.
- **Outputs.** The 14 fold frames above; 344,419 rows each (`pack.rows.fold`).
- **Hardware, threads, time.** Per member in `docs/LINEAGE-ZABD.md` "Members": route level 2 CPU, 16
  threads, 4.4 h; head CPU, 8 threads; g held CPU; mlp and the sequence members GPU. The R4 and R4mks month refits:
  our Mac (`R4Fjjrefit-month-*.json`, `R4mksFjjrefit-month-*.json` field `host`). Other wall times: not in the
  record.
- **Check command.** The replay rebuilds each frame from its saved values and checks the frame `digest` of the pack
  (`scripts/build_entry.py`; `docs/LINEAGE-ZABD.md` "What the replay reproduces"). A rebuilt fold file
  checks against `source_sha256` (table above).
- **Differences from our run.** Path roots only for S16.2-S16.4. The fold vectors of the GPU members (g, mlp,
  sequence members, dark18) are recipe refits (section "Two kinds of result"). LINEAGE-ZABD Gaps 1-4 still apply
  to the fold side: the route members were fit with a later level-2 stacker than the exp-148 code; the held and the
  2026 route fits used two versions (Appendix B lists the versions).

## Appendix A. Roots in our runs

What each root held in our runs. Our private folder names stay out of this release; the content of each root is
what a reader needs to rebuild it.

| root | content in our runs |
|---|---|
| `${PRC_FINAL_ROOT}` | the final-mode run folders on the x86 host: `finaldata`, `finalroute`, `finalclock`, `finalcore`, `finallirf`, `finalv`, `finalplan`, `gainlane`; `finalv/venv312` is the Python 3.12 environment |
| `${PRC_TOOLS}` | the guard tools of the x86 host (Appendix C) |
| `${PRC_LOG_ROOT}` | the run logs of the x86 host |
| `${PRC_MODEL_ROOT}` | saved model files: `prc-exp181-work/out/gpu0` (g fits), `prc-data-challenge-2026/experiments/exp-170-diverse/out` (sequence fits), `prc-exp148-work/stack` (route level 2); other saved models stay in their run folders |
| `${PRC_OPDI_ROOT}` | the exp-155 work folder: its code, `work/airports.csv`, the OPDI windows and `r2/` |
| `${PRC_OPDI_RAW}` | the OPDI raw downloads |
| `${PRC_COMP_ROOT}` | the parent folder of the checkout on the x86 host and on the Mac: `prc-exp148-work` (saved 2025 features and proxies, `kinput`, the R4mks month fits), `prc-exp148-src` (the frozen tree of the v73 member fits) and the other 2025-side work trees of S0.1 |
| `${PRC_ISO5_ROOT}` | the run tree of the isolated worker (s7 seed fits 4-9) |
| `${PRC_SOURCE_EXPORT}` | the source export `export-d07bf4a0a621` (exp-187 `pack.py`, S11) and its copy on the x86 host (S12) |
| `${PRC_HOST}` | the ssh host name of the x86 host in the record commands |
| `${PRC_KAGGLE_KERNEL}` | the route L1 kernel `prc-exp189-learners-final` of our Kaggle account |
| `${PRC_SOURCE_ROOT}` | the run trees of S0.2 ("Run tree of each step") and the main checkout |
| `${PRC_DATA_DIR}` | Mac: `<checkout>/data/raw`, the final pair in `final/`. x86 host: `${PRC_FINAL_ROOT}/finalcore/raw` (links to the 2025 files, `ranking.parquet` and `submitting.parquet`; copies of the final pair in `final/`) |
| `<external>` | a folder for the other downloads: `ourairports/` (`runways.csv`, `airports.csv` of 2026-09-08), `gateway-cache/`, `era5/era5_hourly.parquet` and the exp-045 day summaries |

## Appendix B. Files that ran in two or more versions

Each path that ran in two or more versions, with the version that the release ships. Sources: the run records of each
step and `release/REFIT-SOURCES.json`. Hashes are 16-hex or 8-hex prefixes. Nodes name the steps that ran a version
(`plan-*` nodes are held-side and earlier-plan steps). "Shipped" gives the commit of the shipped version, the action
(add or replace: in `refit-src/`; keep: the export file; inplace: the export file with one line changed) and the
SHA-256 of the shipped file. A `refit-pins/` unit is named where it holds the held version (S16.4).

| path | versions that ran (commit → SHA-256; nodes) | shipped (commit, action, SHA-256 shipped) |
|---|---|---|
| `experiments/exp-045-harvest/code/days.py` | 774c1628d, 84605a82b → 5607e96a (plan-data, plan-features); 8605e8608 → d4b1bba0 (plan-features) | 84605a82b, replace, 5607e96a |
| exp-045 `manifest.json` | six commits → dee46269 (plan-features, plan-head); 76a65e731, ad7cce060 → 1a7f991a (data, route) | out (Kaggle account index; Appendix C) |
| `experiments/exp-046-surfacefit/code/arms.py` | 51ab611f8 → b11349ac; 5e936513f, 76a65e731, ad7cce060 → 2d75bb9b (data, route); 9b55df664, c86ae6557 → fbf6f2ce; ff4f61dd2 → 0945eee9 (plan-head) | 76a65e731, replace, 2d75bb9b |
| `experiments/exp-046-surfacefit/code/base.py` | 51ab611f8, 9b55df664, c86ae6557 → e331d0bf; four commits → f06c96c8 (data, route, plan-head) | 51ab611f8, keep, f06c96c8 |
| `experiments/exp-046-surfacefit/code/chainfold.py` | 51ab611f8 → ba731d32; four commits → 97099cac (data, route, plan-head); 9b55df664, c86ae6557 → fad0149e | 51ab611f8, keep, 97099cac |
| `experiments/exp-046-surfacefit/code/rung511.py` | 51ab611f8, 9b55df664, c86ae6557 → 12aaf687; four commits → 0f7dce8d (data, route, plan-head) | 51ab611f8, keep, 0f7dce8d |
| `experiments/exp-046-surfacefit/code/serve.py` | 51ab611f8, c86ae6557 → 32e9045b; 5e936513f, 76a65e731, ad7cce060 → e6963242 (data, route); 9b55df664 → 805762e0; ff4f61dd2 → 1f12aa75 (plan-head) | 76a65e731, replace, e6963242 |
| `experiments/exp-047-queue/code/block.py` | 5e936513f → 1f6e14fe (plan-features); ad7cce060 → a82fa9c0 (named only) | 5e936513f, keep, 1f6e14fe |
| `refit-src/experiments/exp-185-final-data/code/emit_final.py` and exp-047 `code/emit.py` | the January and July class, block and rank_meta ran at 2d30237b5 (replaced) and at ad7cce060 (the final run); the block wrapper `emit_final.py` (0d36f766b) was replaced by the pinned `emit.py` (S1) | `emit_final.py`: 0d36f766b, add, 4b0215fd (the replaced wrapper) |
| `experiments/exp-127-clock-precision/build_stand_geometry.py` | acb2b6b3 (made the old table); 67133941 (f48487d40, the export file); 5fb13ef8 (dd33cb5dc, both exp-185 builds) | dd33cb5dc, replace, 5fb13ef8 |
| `experiments/exp-134-proxies-in-learner/code/prepare_inputs.py` | 40068dee (ad7cce060, ran); d8f62a3f (1daecf197, the Kaggle title only) | ad7cce060, replace, 40068dee |
| `experiments/exp-134-proxies-in-learner/code/fit_exp134.py` | three commits → 8998699c (named only); six commits → 7001010d (data) | e2bb55a2d, replace, 7001010d. The held MLP package ran `ef861cee` (S16.4; unit `mlp-held-source`) |
| `experiments/exp-147-foldserve/code/pilot.py` | 260aa186f → 5fc8f735 (plan-route); seven commits → f578b178 (clock, route) | b8426be56, replace, 49f2b57c |
| `experiments/exp-148-short-route/code/features.py` | 5e936513f → 4e7df021; e6d9a138f → 8e6bdb7f | 5e936513f, replace, ae55929d |
| `experiments/exp-148-short-route/code/stacker.py` | 260aa186f → 21baf9cd; 73ff84fb7 → fc17ef35; five commits → b99dfbb2 (route); held: 3ba3625a3 → 71a1381a, b7fee8fba → 2a37769e | cb18f1903, replace, 3f0f1494; units `held-r4f`, `held-r4mks` |
| `refit-src/experiments/exp-148-short-route/code/fold_serve.py` | four commits → 6ec7b927 (route); held: 3ba3625a3 → 4a8dd063, ec40e740c → 6b37360e | cb18f1903, add, b06fd3f5; units `held-r4f`, `held-r4mks` |
| `refit-src/experiments/exp-153-grind/code/learner_wp.py` | 132a93d2d, 8b8018743 → d15b9431 (plan-mlp); 9edc8beb4 → 9162b70e (plan-mlp) | 9edc8beb4, add, abbc137e. The held MLP inputs ran d15b9431 (copy `results/member-002/source/learner_wp.py`, added) |
| `refit-src/experiments/exp-155-opdi-second-clock/code/extract_full.py` | git: be38b00ea → d5c1d89c. The x86 copy that ran: 222a612f (the 7012a9b1c bytes; docstring lines 3-4 differ; `b1-recovery/gate-check/RESULT.md:33`) | 7012a9b1c, add, 2310f685 |
| `refit-src/experiments/exp-155-opdi-second-clock/code/gate_check.py` | 3e1152a0c, 6c7b152ed → c93ff0fe (data, opdi2025); c08b9e56e → 7e37565f (opdi2025) | 6c7b152ed, add, c93ff0fe. The final `match_final.py` imported c93ff0fe (`b1-recovery/gate-check/RESULT.md`) |
| `refit-src/experiments/exp-155-opdi-second-clock/code/match_full.py` | 3e1152a0c → 97c136b5; c08b9e56e → 4f933101; four versions in git, 97c136b5 ran | 3e1152a0c, add, 97c136b5 |
| `refit-src/experiments/exp-155-opdi-second-clock/r2/code/prepare_inputs.py` | three versions; c2165683 (2c19fec81) made `84b5c5ea`; the older output `b12a6b2a` is unused | 2c19fec81, add, c2165683 |
| `experiments/exp-164-double-down/code/mlp007/prepare.py` | 4a6aefb6d → d91d9802 (plan-mlp, export y). The held input build (prepare-03) ran `e4606675` (rebuilt byte-exact) | unit `mlp-held-source` (shipped `76db1869` after the path rules) |
| `experiments/exp-164-double-down/code/mlp008/io_utils.py` | seven commits → e420c53f (plan-mlp); e2bb55a2d → 00ca0ab9 (core) | e2bb55a2d, replace, 98795a0f |
| `experiments/exp-169-stack-plan/code/assembly.py` | nine commits → 3bda6f12 (assembly, gain, lirf, v); 21d4c0f59, 591844413 → e652a185 (plan-rule); a690f9ec3 → 0e4536c4 (first fold bundle) | d6160347f, keep, 3bda6f12; unit `held-bundle` |
| `experiments/exp-169-stack-plan/code/diverse_blend.py` | nine commits → e2c84382; 21d4c0f59, 591844413 → bca3d82f (plan-rule) | d6160347f, keep, e2c84382 |
| `experiments/exp-169-stack-plan/code/lane_c_swap.py` | eight commits → fc90a4cf (lirf, plan-rule); d07bf4a0a → 834570d4 (assembly, gain, lirf, v) | d07bf4a0a, keep, 6b04e807; unit `held-swap` (fc90a4cf) |
| `experiments/exp-169-stack-plan/code/read_stacks.py` | ten commits → 2615e3bb (plan-rule; the first fold bundle); d07bf4a0a → d2817b04 (assembly, gain, lirf, v) | d07bf4a0a, keep, d2817b04 (the export copy differs from 2615e3bb only in its release rewrite lines) |
| `experiments/exp-169-stack-plan/code/route_swap.py` | four commits → 0f7d67a8; 31b83a31a → a65b66b8 (the R4mk swap) | d07bf4a0a, keep, 908f56a9; unit `held-swap` (a65b66b8) |
| `experiments/exp-169-stack-plan/code/stack6.py`, `stack7.py`, `stack8.py`, `stack8_transition.py` | eight commits → 61a0d690, eb430068, f8702c24, 51523a88 (lirf, plan-rule); d07bf4a0a → bc9fdf0e, 34d67b34, 09fbb031, 76409b9d (assembly, gain, lirf, v) | d07bf4a0a, keep (each); `stack6.py` 61a0d690 in unit `held-swap` |
| exp-170 `code/rebuild_fixed039_annual_061*.py` | the pre-split launcher `…-pre-split-1511.py` (751f9457b → cc582b0b) ran first; the split `rebuild_fixed039_annual_061.py` (`01919a87`; in the export) made the saved sequence039 fits (S8) | pre-split: 751f9457b, add, cc582b0b. The split version: the export file |
| `refit-src/experiments/exp-186-final-core/code/parity.py` | sixteen commits → 0ac593fd; 25cbf691e → 493bd478 (check, g) | fa170fcd0, add, 0ac593fd |
| `refit-src/experiments/exp-187-final-lirf/code/frames_v.py` | 34df39e94 → 90dcddb8 (the side file); c3e3c593f → 6d6afaf1 (rule frame, exploit frame); f094d1aa0 → d1f84c02 (the a444 bundle). exp-193 rebuilt all four parts with d1f84c02: same SHA-256 | f094d1aa0, add, f1c0519f |
| `refit-src/experiments/exp-189-final-route/code/route_parts.sh` | 76a65e731 → 426144a0 (combine); 0cf58d001 (six early member parts; exp-189 `REPORT.md:254-258`); b8426be56 → e811412a (level 2, GPU parts, route parts) | b8426be56, add, 68397f48 |
| `refit-src/experiments/exp-189-final-route/code/run_member.sh` | 7d177a58a → bf2f0590; f8a5fe89e → 86b5e308. The version per final CPU fit is not in the record; exp-193 rebuilt with bf2f0590 and got equal bytes | 7d177a58a, add, 8653b49c |
| `refit-src/experiments/exp-191-final-v/code/final_run.sh` | 046b08bea, 4d8a6532a, a0c484bab → 8995596f (gain, v); 66b11c579 → e73f0677 (v) | 046b08bea, add, 8995596f |
| `refit-src/experiments/exp-191-final-v/code/stages_record.py` | five commits → 2565485d (gain, v); ae66aae39 → 08a89402 (plan-v) | fa170fcd0, add, 2565485d |
| `refit-src/experiments/exp-191-final-v/code/v_run.sh` | 046b08bea, 4d8a6532a, a0c484bab → d3355c82 (ran for R9 and LD); 66b11c579 → ff478428 | 046b08bea, add, d5b58a72 |
| `refit-src/experiments/exp-191-final-v/code/w2/l2c.py` | c9cd2869; the run applied `isolated5-A/l2c.diff` (`9e72607a`) at run time and ran `c9931e0a`; the release ships `w2/l2c_candidate_a.py` `c9931e0a` | `w2/l2c.py`: fa170fcd0, add, c9cd2869 (before the patch). `isolated5-A/l2c.diff` (`9e72607a`) is in `refit-src/` |
| `refit-src/experiments/exp-192-final-assembly/code/final_sources.py` | 70d3707d7 → 8afd7914; our reading: `ee6abb18` (696d473de) wrote `final-sources-2.json` (S12) | 696d473de, add, 878f6afa |
| `release/assembly_replay.py` | 779def01 (the export overlay, ran); 68b42ad1 (root, v) | d07bf4a0a, keep, 779def01 |
| `refit-src/experiments/exp-193-final-build/code/final_dh.py` | a4d1b5290 → f739c19c (made `c4511650`); f73cbd855 → cc2b7a1f (made `2fe6c56b`). Neither made the dh entries; exp-194 `r9_dh.py` made `07d4a4e6` and the default `eea300c0` | cbd92d78f, add, 44c5d83e |
| `refit-src/experiments/exp-193-final-build/code/r9_entries.py` | the `--write` run used `b6a9c7bd` (20cba3274); 0f982e499 → cda727c0 is a later version that nothing ran (S15) | 20cba3274, add, b35d96c8 |
| `experiments/exp-159-clock-head/code/common.py`, `frame.py` | b95da6cd, a86fc1ec (gain, plan-head). The DH fit imported the `prc-exp159-work/code` copies dc92b51c and 5ba004ef (the bytes of 94d89b362; exp-194 `code/exp159-at-fit/README.md`) | 94d89b362: `common.py` inplace (97c8cc08), `frame.py` keep (5ba004ef) |
| `prc/data.py` | five versions: 8bc76267, e613518e (held MLP, first bundle, R4mk swap), 471218c5 (g, route, R4 and R4mks refits), d2fd3fda, 69732391 (export copy d07bf4a0a) | 046b08bea, keep, aec4b437; units `held-*` hold 471218c5 or e613518e (each shipped as 69732391 after the path rules) |
| `prc/gscore.py` | thirty commits → bdba893c; five commits → 2f92c7c8 (plan-g, plan-lib, plan-route) | 73ff84fb7, keep, bdba893c |
| `experiments/exp-148-short-route/code/refs.py` | one version ran for the held F: 72799c21a → 92ecf9f4 (export y except path lines) | in the export |

Parts that exist in two versions (not files): `head` a0f28193… (4-thread refit; the default) and `head-t8` 9fb65fb4…
(8-thread refit). The default reads `head` (exp-193 PACKET.md, finalclock row "Part head a0f28193 stays").

## Appendix C. Site steps left out of the guide

These steps moved files between our machines, waited for a host, or made code snapshots. They make no bytes of a
part. A reader runs each step on one machine with the refit tree, so these steps have no place in the recipe.

| step | files | reason (`release/REFIT-SOURCES.json` key `out`, or the record) |
|---|---|---|
| copies between the Mac and the compute host, with SHA-256 checks | exp-185 `push_month.sh`, `after_hold.sh`, `mac_delete_verified.sh`, `transit_harvest.sh`; exp-189 copies of `features-2026` and the kernel output; exp-187 copies of the a444 bundle and the side file; exp-193 and exp-194 `gget.sh` copies | site transfer |
| Kaggle account steps | exp-185 `land_harvest.sh` (kernel status, output download); exp-045 `manifest.json` (our kernel index); the private dataset creates and the 404 check | a harvest run of the reader writes its own index with exp-045 `manifest.py` (S1) |
| hold waits | exp-185 `remote_lib.sh` (waits while the timing hold of another user is on the compute host); exp-189 `chain_cpu.sh` waits on guard logs; the host pauses during holds h24-h27, h31, h34 | site helper |
| guarded runs and copies | `arc_holdguard.sh`, `gput.sh`, `gread.sh` (finaldata tools); exp-194 `code/remote/gget.sh`; the guard `grant.env` that sets `EXP164_CPUS` | make no bytes; a released script does not call them |
| memory gates | exp-194 `r9_chain.sh` `memok` (MemAvailable 20 GB) and the other MemAvailable checks | site helper |
| code snapshots | exp-185 `mk_snapshot.sh` (git archive of a commit); exp-189 `make_src.sh`, `make_f1_tree.sh`, `mk_feat_tree.sh`; the `src-<commit>`, `repo-<commit>`, `snap-<commit>` and `code-<commit>` snapshots on the compute host; exp-192 `export_copy.py` copies of the release export; exp-191 `rsync --link-dest` run trees | the refit tree builder (S0.2) takes their place |
| checks that write no part | exp-185 `era5_cover.py` and its output `era5/era5_cover.json` | check output |
| inputs absent from the tree that ran the final build | exp-169 `inputs/final-harness-032/reference.json`, `stack3-ext-009/pair.json`, `stack5-016/joint.json`, `stack6-021/PACK.json`, `head-swap.json`, its approval record, `stack7-022/030/lane-c-swap.json` | exp-192 `export_copy.py` copies the release export, which does not hold them |
| stage not run | exp-194 `run_dark_cpu.sh` dark-arm scripts (seven files) | the stage did not make the final parts |

# Earlier entries

V40 scored 238.9597 s RMSE, file SHA-256
`c29c9dc434e2b2e0b910a382d8fd83a9dd6de3e05cf865203ba1a53383623e45`.
[WRITEUP.md](WRITEUP.md) explains its short route and fallback.
[CLAIM-MAP.md](CLAIM-MAP.md) links claims to source and hashed records.

## 1. Two different checks

**Saved-vector replay is verified.** The current route includes saved R4-kno
values, the fixed half blend with v32 and the frozen mean correction.
The export holds complete v35 prediction
values without IDs or labels. It also holds the v22 prediction
values and the v23, v24 and v32 deltas in `release/saved-vectors/`. These are
compressed float arrays encoded as JSON text. They have no organizer movement
IDs or training labels. The official submission template supplies the row IDs.
`release/V32-REPLAY.json` records each array digest, original source vector
path and digest, template and ranking digests, output digests and build counts.
The original NPZ containers and private build receipts do not ship.

**A new fit is not verified.** No fresh fit has reproduced v40, v37, v36, v35 or the older vectors. Section 6 gives the R2 route. The
fitting code is included. The seven original Kaggle input packages are private,
derived packages of challenge data, cached features and open-source observations.
A reproducer must rebuild them. Sections 3-5 state each input and its producer.
Private clean-score exclusions are not included; new clean scores and blend
calibration can differ. The accepted arrays preserve the recorded model.

## 2. Replay from a public export

Use Python 3.11 or later. Install `requirements.txt`. For exact Parquet bytes,
use **pandas 2.3.3 and pyarrow 24.0.0**, the versions that wrote the upload.

Obtain the official ranking file and submission template from the
[challenge data page](https://ansperformance.eu/study/data-challenge/dc2026/data.html).
Place them in a directory with this layout:

```text
input-root/data/raw/ranking.parquet
input-root/data/raw/submitting.parquet
```

The live ranking file is the September re-issue, not the first August file.
The replay checks both official file hashes before it uses any saved values.
Run from the export; `input-root` contains data only. It does not need any
private source, saved private v22 file, Git history or Kaggle account.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/build_entry.py --target v40 --data-root /path/to/input-root \
  --out-dir /path/to/new-replay-output --record /path/to/replay-record.json
```

The output and record must be new. The replay checks official inputs, saved
values, population, value hash and Parquet hash. It fits nothing. V40 replays
R4-kno and the v32 chain, applies their fixed half blend, then the frozen
correction once. It verifies the six hashes for v23, v24, v32, R4-kno, R4-B50
and v40. No complete R4-B50 or v40 prediction array is saved in the export.
The historical v37 target remains available: v35 plus v32, the fixed half
blend, then the same correction. Its six hashes are checked separately.
Run with `--target v32` to rebuild that fallback. Expected outputs by target:

| File | SHA-256 |
|---|---|
| v40-replay.parquet (target v40) | `c29c9dc434e2b2e0b910a382d8fd83a9dd6de3e05cf865203ba1a53383623e45` |
| r4-b50-replay.parquet (target r4-b50) | `4c66c9d6ff10fad3a1999df69cdcf7e46cf95a1cd9b79b4be2ea3d53a6a9c585` |
| r4-kno-replay.parquet (target r4-kno) | `f960ec40010c8343cc90c0ffe52f160ac7c279fcd0242ac15e48b778139cdc86` |
| v37-replay.parquet (target v37) | `2d847eb965ef7b0e58e6655569c2e9546a366b58cad33e7dbfe9c1010097699b` |
| v36-replay.parquet (target v36) | `3ae49f67d62a832c6da7a25ac7538dc62273a9c5f2f7815778f451a24ab5c32f` |
| v35-replay.parquet (target v35) | `4ece6a4b3616a57cc1a54d2fa63657b590226ff052dafc2f791821566b470e35` |
| v23-replay.parquet (target v32) | `fd420d855fe8217a3915cce42fa7dc7794d2c19d7cb26dbf90ab456650f28220` |
| v24-replay.parquet (target v32) | `984884e1f8d2b5403f7e791659fd9902eacac7206484fb14ed3116004df9dc36` |
| v32-replay.parquet (target v32) | `f09f609429c81fd888e19e75e16f51b57978d55cc2639d6161c894da0baaa0de` |

This command fits nothing. It does not prove that a new model can produce
these arrays or that the method transfers to new data.

## 3. Rebuild the older v22 inputs

Put all twelve official 2025 training months, `ranking.parquet` and
`submitting.parquet` in the raw-data directory in your complete source tree:

```text
complete-source-tree/data/raw/
```

Set
`PRC_DATA_DIR` to that directory for the older producers. Source, training,
cache and artifact roots must describe the same challenge version.

```bash
python scripts/build_v22.py --plan
```

This prints the ordered base pipeline, with source commands, output paths,
recorded hashes and records. It does not run the commands. The path consists
of the base members, residual stack, window member, anchor repair, surface
heads, no-clock queue correction and seasonal bag. The required seasonal
producers under exp-047 `work/`, including `census_v22.py`, are released.
Intermediate OOF files, targets, raw challenge files and prediction Parquets
must be made by these producers; they do not ship. Refit members at the
recorded device and thread counts. Some member fits do not reproduce exactly
across device or thread changes.

For the open observations, follow [EXTERNAL-DATA.md](EXTERNAL-DATA.md) and
[external-data-archives.md](external-data-archives.md). The exp-045 harvest
reads adsb.lol archives and writes ground points and segment summaries for
the training and ranking months. Use your own Kaggle account, metadata and
point-output kernels. `PRC_KAGGLE_USER` is required by the new stage producers.
No script supplies the original account name. Do not publish challenge data
when you create your own private derived datasets.

The exp-018 `code/dataset.py` `month_table(month)` cache builder makes monthly
Parquet files in this generated layout:

```text
complete-source-tree/artifacts/exp-018-fresh/cache/<month>.parquet
```

The exp-046 and exp-047
producers make the saved blocks, class vectors, v12 fold residuals, v22
seasonal delta and baseline census. They are inputs to the stage preparers.
The full rebuild needs more work than the measured later-stage fit time.
There is no verified command that runs the full fit chain from raw data.

The base weather member reads the exact derived ERA5 table. It is a separate
CC BY 4.0 bundle of three files, the assets of the release `era5-v1` of this
repository: [era5_hourly.parquet](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/era5_hourly.parquet),
[ERA5-DATA-MANIFEST.json](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/ERA5-DATA-MANIFEST.json) and
[NOTICE.txt](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/NOTICE.txt). Download them into one folder, verify them
and set `PRC_ERA5_HOURLY`:

```bash
mkdir copernicus-era5-v32
for f in era5_hourly.parquet ERA5-DATA-MANIFEST.json NOTICE.txt; do
  curl -fsSL -o copernicus-era5-v32/$f \
    https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/$f
done
python release/era5_data.py verify copernicus-era5-v32
export PRC_ERA5_HOURLY=$PWD/copernicus-era5-v32/era5_hourly.parquet
```

To pack a recovered copy, run `release/era5_data.py pack SOURCE --out NEW_DIR`.
The required file is 15,181,053 bytes with SHA-256
`8238378ec1f6d66a6c59712993ec54ac72af3f130cb7a8a7754a3fbec369f293`.
Its column order begins `hour_utc, airport`; it has no version attr.
Do not rewrite it or replace it with a new ERA5T download.
See [weather-provenance.md](weather-provenance.md) for raw products, DOIs and
transformations. The source export has the bundle manifest and notice script;
the data files are the `era5-v1` release assets above.

## 4. Rebuild each later-stage package

`PRC_ROOT` defaults to the complete source tree containing the preparer. Set
it to your own complete tree if data and generated artifacts are elsewhere.
`PRC_EXP130_OUT`, `PRC_EXP134_OUT`, `PRC_EXP136_OUT` and `PRC_EXP137_OUT` select
output directories; their defaults are under the generated rebuild directory:

```text
complete-source-tree/artifacts/rebuild/
```
Exp-130 and exp-137 also accept the output directory as the first argument.
Use new output directories: the historical exp-136 producer deletes an
existing output directory before rebuilding it.

The table names original logical package labels and versions. They are not
public downloads. Use your own account to upload the rebuilt package privately.
Keep the directory basenames that the kernels locate, or change their mount
configuration as a group.

| Original package/version | Inputs from challenge data and open sources | Producer and contents |
|---|---|---|
| exp127-clock-fold-frames v2 | v22 fold residuals, saved blocks/classes, raw 2025 months and ranking inputs, stand geometry | exp-127 `prepare_fold_frames.py`, then `prepare_ranking_frame.py`: fold/ranking frames, baseline record, cluster table and template IDs |
| exp127b-clock-rederive-spec v1 | Frozen method specification | `experiments/exp-127-clock-precision/rederive/SPEC.md`, copied into the package as its rederive specification |
| exp127b-clock-rederive-keys v2 | Rebuilt fold frames, raw movements, geometry and exp-045 point manifest | exp-127 `rederive/prepare_clock_keys.py`: `clock_keys.parquet`, `ranking_ids.csv`, `points_expected.json` |
| exp130-clean-input v1 | exp-018 monthly caches, raw months, v22 predictions, exp-046 classes, exp-047 blocks, seasonal vector and census | exp-130 `code/prepare.py`: monthly feature frames, fold evaluation frames, ranking classes/order, v22 values and runtime sources |
| exp134-input v4 | Raw departures/arrivals, exp-045 point manifest and stand geometry | exp-134 `code/prepare_inputs.py`: input-only proxy movement tables, geometry, point inventory and runtime source |
| exp136-input v4 | Same raw and open point sources, plus the wider-proxy runtime sources | exp-136 `code/prepare_inputs.py`: proxy input tables, point inventory, geometry and runtime source; no fitted exp-136 arm is used by v32 |
| exp137-input v3 | Rebuilt exp130 package, complete exp127 fold deltas, v22 fold vector, v24 ranking predictions and official template | exp-137 `code/prepare_inputs.py`: `fold_base_v23.parquet`, `ranking_base_v24.parquet`, repaired fold deltas and runtime source |

The released stand lookup is in exp-127 `stand_geometry.csv`. It records source
URLs, IDs and hashes per row. To rebuild it, run `build_stand_geometry.py`,
`fetch_osm_parking_positions.py`, then `merge_osm_geometry.py`, with the same
open-source snapshots. A fetch of current recommended scenery or current OSM
is a new input version. [GEOMETRY-DATA.md](GEOMETRY-DATA.md) gives the notices.

Run preparers in order after their inputs exist:

```bash
export PRC_ROOT=/path/to/complete-source-tree
export PRC_KAGGLE_USER=YOUR_ACCOUNT
python experiments/exp-127-clock-precision/prepare_fold_frames.py
python experiments/exp-127-clock-precision/prepare_ranking_frame.py
python experiments/exp-127-clock-precision/rederive/prepare_clock_keys.py
python experiments/exp-130-ranking-exposure/code/prepare.py
python experiments/exp-134-proxies-in-learner/code/prepare_inputs.py
python experiments/exp-136-wider-proxies/code/prepare_inputs.py
# Run the v23 fit and v24 fit before preparing exp137 inputs.
python experiments/exp-137-withclock-proxies/code/prepare_inputs.py
```

The exp-137 preparer reads locally rebuilt exp-127 files under `push/`, not a
moving branch. Recreate `fold_update_vectors-r1.npz` from v23's full
`primary_fold_predictions.parquet` with `push/build_oof_fold_updates.py`.
The original fitter exports only its last fold; that file is insufficient.
The public `release/V23-FOLD-BASE.json` retains the original aggregate
RMSE checks with LIRF held. The preparer compares new complete fold values
with those checks; a mismatch is a reproduction failure, not a new score.
The private sidecar is optional in `prc.data` and is neither required nor
packaged by the public exp-130 preparer. Clean scores from a public rebuild
can differ from the original recorded private calibration.

## 5. Fits and checks that remain

The v23 coverage kernel is `rederive/kaggle/coverage-kernel/rederive_clock.py`.
It attaches your point-output kernels and the rebuilt clock keys and writes
`clock_features.parquet`. Attach that output and the fold/spec packages to
`rederive/kaggle/fit-kernel/fit_rederived_clock.py`. The recorded fit used
Python 3.12.13 and scikit-learn 1.6.1. Save the complete fold predictions.

The v24 launcher is exp-134 `kaggle/kernel.py`; it loads the exp-130 package,
the exp-134 package and point kernels. It calls `code/fit_exp134.py`.
The recorded fit used Python 3.12.13 and LightGBM 4.6.0.

For v32, run exp-137's control, proxy and sham arm kernels on exp-130,
exp-136, exp-137 and the point outputs. Combine with
`code/combine_exp137_arms.py`. The recorded final all-year proxy fit is
`kaggle/run-v4/final/kernel.py`. Its model rules and the recorded full-precision weight
are preserved. Its old metadata and run name differed; create one consistent
caller-owned name. Set `PRC_KAGGLE_USER` inside the Kaggle runtime as well.
No account-bound kernel metadata is shipped.

The original seven package versions matched 67 of 67 recorded file hashes;
three downloaded files had no recorded hash. This verifies existing inputs,
not raw-data regeneration. Some inputs remain only in private derived
packages. Released sources and sanitized specifications have new hashes.
The exp-137 spec guard now checks the sanitized public spec, whose method
rules are unchanged. Other old fixed input checks can reject a rebuilt package. Resolve each
mismatch against the published schema and original model rules, record a new
input manifest and compare fresh outputs; do not bypass checks or call the
old vectors freshly reproduced. A complete automatic raw-input rebuild is
still a gap. The fault list is in PUBLIC-MODEL-v32.json.

The measured later-stage runs imply about 21 Kaggle CPU hours after inputs
exist. CPU images and libraries can change results. Compare new ranking
arrays with `release/saved-vectors/`, then compare the composed value and
file hashes. A score alone does not verify a vector. No fresh fit, new model,
score upload or public release was performed for this source export.

## 6. Refit the exp-148 short route for v35

The fitting sources are `experiments/exp-147-foldserve/code/` and
`experiments/exp-148-short-route/code/`. The public specification is
`experiments/exp-148-short-route/SPEC.md`; the recorded serving choice is
`release/V35-CHOICE.json`. These commands fit models; they are not part of
saved-value replay, and this release has not run them as a fresh rebuild.

First rebuild the v22 member caches, surface/queue blocks and class inputs in
section 3, the clock frames and proxy packages in sections 4-5, and exp-146's
frozen row/class and cell files from the same official and open-source versions.
The twelve official training months are required. The original private derived
Kaggle input packages and fitted member/model files are not public inputs.
The same rebuilt exp130/exp134 packages supply the L and Lp drivers locally;
Kaggle is optional. Keep their exact runtime sources and complete 2025/2026
feature frames together in one caller-owned input directory.

Choose thread and device settings that match the recorded fits, and new output
paths. Set the source root to the complete released tree; its data and caches
must have the layouts described above. Set PRC_EXP127_ROOT to that same complete
tree if the clock producer used another source root. The driver source records
model settings and thread/device choices. Original vocabularies, CV rounds and
CPU/GPU versions can change fresh-fit values even when the formulas match.

```bash
export PRC_SOURCE_ROOT=/path/to/complete-source-tree
export PRC_DATA_DIR=/path/to/official-raw-data
export PRC_EXP127_ROOT="$PRC_SOURCE_ROOT"
python scripts/exp148_missing_features.py --data-dir "$PRC_DATA_DIR" \
  --out /path/to/new-r2-boundary-exceptions.json
export PRC_R2_NO_FEATURES=/path/to/new-r2-boundary-exceptions.json
```

The boundary producer derives the two departures stored in March and September
files at 23:59:57 UTC on the last day of the previous month. Its output has IDs;
keep it with your challenge inputs. The public source contains no such IDs.
The R2 fitter rejects any missing-feature row outside those exceptions.

For each fit in final, janjul, jjx-12, jjx-02, jjx-06 and jjx-08, run the six
base-member drivers below, one fit per command, with a new output directory.
The exp008 driver supplies two members, selected by its member argument.
The six split names form the minimum accepted R2 route; the four jjx splits
add 32 fits across all eight members. Use all original outer/inner splits only
if also rebuilding the earlier week-pair comparison.

```bash
M=experiments/exp-147-foldserve/code/members
python "$M/T-tailv4_wp.py" --fit final --out /path/to/member-output --threads 6
python "$M/N1-nmtail_wp.py" --fit final --out /path/to/member-output --threads 6
python "$M/D1-winter_wp.py" --fit final --out /path/to/member-output --threads 6 --device cpu
python "$M/exp008_wp.py" --member L2-hub8725 --fit final --out /path/to/member-output --threads 6 --device cpu
python "$M/exp008_wp.py" --member C2-cat --fit final --out /path/to/member-output --threads 8 --device gpu
python "$M/F1-fresh_wp.py" --fit final --out /path/to/member-output --threads 6 --device gpu
python experiments/exp-147-foldserve/code/learner_wp.py --fit final \
  --input-root /path/to/rebuilt-input-packages --out /path/to/learner-output \
  --cache /path/to/learner-cache --learners L,Lp
```

Repeat with each remaining fit name. Members record split admission, ownership
and hashes. No step can reuse an artifact whose training labels reach the
scored month. The stacker uses member output from final, janjul and the four
jjx fits. It does not need a saved original fitted model.

Build the row features with `experiments/exp-148-short-route/code/features.py`
and `features_r2.py`; the latter takes `--reference` for the first feature table.
Use `learner_wp.py --fit features` and `--fit features-r2` to export proxy and
row-clock tables. They take the same input-root, output and cache arguments.
`combine_r2.py` checks January proxy parity, then joins the old and new months.
It reads the following caller-owned work layout:

```text
work/inputs/features.parquet
work/inputs/features-r2.parquet
work/inputs/proxy-features.parquet
work/kaggle/r2-la/out/proxy-features-r2.parquet
```

The last path is an output layout name; local learner output can be copied there.
Set WORK to this work directory and run the combining script. It writes the
combined feature and proxy tables. Fit owner rfinal with inputs r2:

```bash
python experiments/exp-148-short-route/code/stacker.py --owner rfinal --inputs r2 \
  --art /path/to/member-output --art /path/to/learner-output \
  --features /path/to/work/inputs/features-r2c.parquet \
  --proxy /path/to/work/inputs/proxy-features-r2c.parquet --out /path/to/new-stack --threads 6
python experiments/exp-148-short-route/code/serve_r2.py \
  --final /path/to/new-stack/R2-final.parquet --score release/V35-CHOICE.json \
  --features /path/to/work/inputs/features.parquet \
  --v32 /path/to/v32-replay.parquet --out /path/to/new-v35
```

The published choice keeps the v35 form, weight and row sets fixed. For local
validation, also fit owners r01 and r07 with inputs r2. Repeating original
variant selection needs the old v32 reference month-fold vectors used by
`experiments/exp-148-short-route/code/refs.py`; its original generator reads
private Git artifacts that do not ship. Rebuilt references can produce another
choice. The original local keep-set gain is optimistic and is not a fresh
isolation claim. New candidates use the month-hold level-2 condition.

Compare a fresh output with the public entry hash and values. A mismatch is a
reproduction failure, not permission to replace the recorded target. No fresh
fit has produced the accepted v35 bytes in this release check.

## 7. Compose the fixed v36 blend

After rebuilding v35 and v32, use the released serving source. The accepted
weight is fixed at 0.5; it is not a fitting parameter.

```bash
python experiments/exp-148-short-route/code/serve_blend.py \
  --a /path/to/rebuilt-v35.parquet --b /path/to/rebuilt-v32.parquet \
  --w 0.5 --name B50 --out /path/to/new-blend-output
```

Use the source and raw-data environment from section 6. The driver aligns each
input to the official template, rejects duplicate or missing IDs, checks
finite values and the floor, and writes the recorded Parquet schema. A new
fit of either component can change the blend. Saved replay proves the accepted
bytes; it does not prove that both fresh-fit routes return those bytes.

## 8. Frozen published-mean correction

The v37 target first replays v36. The default v40 target first replays R4-B50
as described in section 9. Both apply the frozen correction exactly once. It checks the hash of
`experiments/exp-151-published-means/frozen-parameters.json`, the unchanged
parameter record frozen at revision a105637f202e8b082e3b4fc0e6b6951d1c01c979.
The 20 ranking-inspection cells give corrected means and shrinkage weights;
the 20 development cells remain in the record with their validation warning.
It also checks `release/saved-vectors/v37-support.json`: binary admission
values in template order, without IDs or class rows. The official ranking
supplies airport and UTC month. `scripts/published_shift.py` applies the same
80-step floored shift calculation as the pinned fit source. The support rule
holds 22,721 unsupported rows fixed. No source workbook is shipped.

A new fit needs the component inputs in sections 3-7, the exact monthly
EUROCONTROL workbook, 2025 challenge coverage summaries by airport/month,
historical v24 held predictions and labels, and raw/open-clock class rows.
The [source notice](EUROCONTROL-DATA.md) gives the workbook URL, digest and terms.
The caller supplies the workbook to the extractor's source directory. Run the
extractor to read monthly means, rebuild the historical all-label offsets and
coverage table, then use `step_d.py`'s freeze calculation in a new results
folder. A new support vector requires January/July training counts for each
month-airport-class cell and ranking class rows from the same open-clock
pipeline. This export includes the formulas, frozen table and admission values;
it does not include those raw inputs, labels, row maps or fit receipts. No new
fit or fresh source download was used for the verified replay.

## 9. R4 nine-month fits and the v40 route

Start with the eight R2 members and their split-aware input rebuild in section 6.
Run each member on jjx-03, jjx-10 and jjx-11: 24 additional fits. Each excludes
January, July and its own scored month, including the month-hold buffer days.
Rebuild the March, October and November raw/open-clock features, with the same
class rules and source versions. The boundary-ID configuration from section 6
now includes missing_feature_ids_by_file_month for the March exception. These
IDs stay only in the caller's output.

```bash
python experiments/exp-148-short-route/code/features_r2.py \
  --out /path/to/work/inputs-r4 --reference /path/to/work/inputs/features.parquet \
  --months 2025-03,2025-10,2025-11 --name features-r4 --drop-outside-days
WORK=/path/to/work python experiments/exp-148-short-route/code/combine_r4.py
python experiments/exp-148-short-route/code/stacker.py --owner qfinal --inputs r4 \
  --art /path/to/member-pack --features /path/to/work/inputs/features-r4c.parquet \
  --proxy /path/to/work/inputs/proxy-features-r4c.parquet --threads 16 --out /path/to/new-stack
```

The proxy feature export and member pack must include all seven jjx splits,
janjul and final. q01 and q07 score January and July from the nine-month panel
with that month's labels excluded. Run them too, then score_r4.py selects the
form from their training CV. The accepted choice is R4b, weight 1, keep set kno.
serve_r2.py supports --owner qfinal; it keeps v32 on noac, ordinary rows without
AOBT and unsupported cells. serve_blend.py applies weight 0.5 with v32. Apply
section 8's unchanged frozen correction once to that unshifted R4-B50 vector.
The public source is pinned to b80d39f4afa21f2019e21d2ad810c1de14a4d7fd;
the accepted shift wrapper was pinned to 5f3260a56e249e34973e6f41dcdb81dbdf86915d.
The parameter table stays byte-equal to its a105637f freeze. Its original
results location is excluded; the copied model table has EUROCONTROL attribution.

The default data-only replay uses saved r4-kno values instead of these new fits.
It reconstructs v23, v24 and v32, r4-kno, r4-b50 and v40, checking all six hashes.
No fresh fit has reproduced R4 or v40. New fits need exact source versions,
raw challenge files, open observations, proxy packages, member artifacts and
CPU library/thread settings. The development reference is a local stand-in;
it does not validate a complete fresh served route.
