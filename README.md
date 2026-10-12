# PRC Data Challenge 2026 — youthful-giraffe

The release target is **final-v-dh-r9-ld**: the final file
`youthful-giraffe_final.parquet`, a prediction for each of the 670,790
departures of the organizers' final submitting file (January, February, June
and July 2026). File SHA-256:
`eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02`. It is the
entry **final-v-r9-ld** (SHA-256
`31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759`) with one
more step on 14,587 EDDF departures that have no clock-head value and no
adsb.lol track input. final-v-r9-ld is the model of v73 on the final files
with four changes: the s7 member is the mean of 15 fits (v73: 3), the L2C step
also reads the s7 member, with 9 LightGBM seeds (v73: 3), the zabd assembly
reads the 12-fit bag of the MLP member on every member-scope row (R9; v73: on
clock rows only, 2 seeds on the other rows), and the ORD-S step also changes
the ordinary rows of long-taxi airport-days (LD; v73: these rows keep their
value). In all final entries the route has no v32 half: the route, its start
value F and the no-plan base are R4mks alone (v73: half R4mks, half v32). The
entries **final-v-dh-r9** (on the entry **final-v-r9**, the first
three changes, SHA-256
`5ce37d698943fed7a1f512aacf9500140ea7ee0809592670fe3a73ec2039fb7e`) and
**final-v-dh** (on the entry **final-v**, the first two changes, SHA-256
`833533adfc04e76dbd3f300b5c69945ce305a0111a858b5a08a67ad85f880cfa`) apply the
same step to the other two 15-fit V files. The entries **final-iii-dh-r9-ld**,
**final-iii-dh-r9** and **final-iii-dh** (on **final-iii-r9-ld**,
**final-iii-r9** and **final-iii**: the same changes with a 9-fit s7 member,
SHA-256 `07d4a4e6344f5c681761a00a851d23dd9dd6bc8675861179b0de80b4fe669865`,
`4ac66707333eab1b32c43db0f412907e05d35e268d6a869065dba3fdc99b8139` and
`c45116502cad96dbb270467b9f2ae26fb6e06d9a50ca34ff214253609f44612d`),
**final-ii-dh** (the 9-seed s7 member only) and **final-dh** (on the entry
**final**, V, the model of v73 on the final files, SHA-256
`121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a`) apply the
same step to the other final V files.

**v73** is our final entry of the ranking phase: a prediction for the 344,841
departures of the ranking file. File SHA-256:
`70236754db85e5f23fac281ebbc9c14a3a4fe33c4e6f34e7c95262eebe1659d8`.
It scored 223.3071 s RMSE as upload v73, our best scored upload. It is **v64**
(SHA-256 `9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b`,
225.7082 s as upload v64) with three groups of steps. v64 is the assembly
**zabd** (SHA-256
`e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636`, 225.9106 s
as upload v62) with one fixed stage, T11b-L.
[Public results](docs/PUBLIC-SUBMISSIONS.md) keep the upload history. Probe
uploads are history only; no released parameter is derived from the board.

Read the [method write-up](docs/writeup/METHOD.md), the
[lineage of zabd, v64 and v73](docs/LINEAGE-ZABD.md) (members, rule stages,
label use and gaps), the [data sources and licences](docs/DATA.md) and the
[reproduction instructions](docs/REPRODUCE.md).
[WRITEUP.md](docs/WRITEUP.md) describes the earlier accepted entry v40
(238.9597 s RMSE), and [the claim map](docs/CLAIM-MAP.md) links each public
claim to its code and record.

## The model in brief

zabd starts from a route prediction: a LightGBM and CatBoost stack over eight
level-1 members (R4mks), averaged with the older v32 route. The final entries
use R4mks alone as the route, with no v32 half ([METHOD](docs/writeup/METHOD.md)
section 4.11). A LightGBM clock head
corrects it where an ADS-B departure clock exists. A fallback pair (a GRU
sequence model and a row MLP) covers rows outside the main member scope. Six
members then enter one weighted least-squares blend on the remaining
residual: a bag of runway-sequence transformers (g), an embedding MLP, three
GRU sequence models (one of them, dark18, trained with clock inputs hidden on
some days) and a row MLP. Rule stages for LIRF practice, stand re-use, mirror
rows and twin flights follow, and one projection onto rule bounds ends the
chain. The members were fit on 2025 without January and July for the held
predictions (except the v32 half of the route). For 2026 most members were
refit on all of 2025; the clock head keeps its ten-month fit and the route's
final level 2 uses nine months. The blend weights and two rule statistics are
fit on the held January and July 2025 departures.

v64 adds one fixed stage to zabd. On the 4,902 rows of 2026 with an aircraft
type but no NM flight join, which the blend does not serve, the value moves a
quarter of the way toward the dark18 member. Rows of the 15 long-taxi
airport-days of 2026 keep the zabd value.

v73 adds three groups of steps to v64 (code in `iso/` and `w2/`). Five steps
serve rows outside the linear stack: January 2026 clock rows at LEMD, LFPG and
LIRF that have no 2025 rows of their kind, long-route rows, LIRF rows without
an aircraft type, and three of those rows that can carry a date rollover
(upload v68, 224.243 s). A LightGBM level-2 model adds half of its correction
on clock rows with an NM join (upload v69, 223.8572 s). On ordinary rows, a
transformer over the 64 movements before and after the departure at its
airport (s7) enters with one coefficient per airport, floored at 0 (uploads
v70 to v73).

## Reproduce the final file

The caller supplies the two final files, `final_submitting.parquet` and
`final_ranking.parquet`, in `input-root/data/raw/`.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/build_entry.py --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/final-v-dh-r9-ld-build.json
```

With no `--target` the script builds the default target final-v-dh-r9-ld. On
14,587 EDDF departures of January, February and June 2026 that have no
clock-head value and no adsb.lol track input, the value is
max(60, V + 0.164 (DH − V)), where V is the file of final-v-r9-ld; every
other row is V. It reads the saved values of V and of the reduced clock head DH
and fits nothing, so its bytes are the same on every platform; it takes about 2
seconds. `--target final-v-dh-r9`, `--target final-v-dh`,
`--target final-iii-dh-r9-ld`, `--target final-iii-dh-r9`, `--target final-iii-dh`,
`--target final-ii-dh` and `--target final-dh` apply the same step to the files
of final-v-r9, final-v, final-iii-r9-ld, final-iii-r9, final-iii, final-ii and final.
[REPRODUCE.md](docs/REPRODUCE.md#release-target-final-v-dh-r9-ld) gives the
details.

The V file itself is the entry final-v-r9-ld. It needs 16 official files in
`input-root/data/raw/`: the 14 files of v73 (next section) and the two final
files.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target final-v-r9-ld --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/final-v-r9-ld-build.json
```

The build runs the chain zabd-final-r9 (the zabd assembly on the R9 pack, which
is the final pack with the 12-fit MLP values on every member-scope row),
v64-final-r9 (the T11b-L stage) and the v73 steps on the four final months, in
final mode; its ORD-S step is `w2/ords_ld.py` with the flag `--long-days`.
February takes the January scope of each fitted step and June the July scope.
It reads the saved values of the 15-seed s7 member
(`release/saved-vectors/s7c15/`) and refits no member; the L2C step of
final-v-r9-ld reads the float32 values of that member. The bytes are exact on
x86-64 Linux with the versions of `w2/requirements-remote.txt`; on another
platform the record gives the difference to the saved V values. It takes about
9 minutes on x86-64 Linux. `--target final-v-r9` (the same chain with the
ORD-S step of v73) and `--target final-v` (the chain zabd-final and v64-final on
the final pack) write the other two 15-fit V files. `--target final-iii-r9-ld`,
`--target final-iii-r9` and `--target final-iii` are the same three with the
9-seed member (`release/saved-vectors/s7c9/`), `--target final-ii` (9 seeds,
also the L2C step of v73) and `--target final` (also the 3-seed s7 member)
write the other final V files. No entry of the final chain can use the published-mean-shift recipe of
v37 and v40. [REPRODUCE.md](docs/REPRODUCE.md#release-targets-final-v-r9-ld-final-v-r9-final-v-final-iii-r9-ld-final-iii-r9-final-iii-final-ii-and-final) gives the
details.

## Reproduce v73

The caller supplies the 14 official files (the submitting file, the ranking
file and the twelve training files of 2025) in `input-root/data/raw/`;
`w2/inputs/raw_sha256.txt` lists their names and SHA-256 values.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target v73 --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/v73-build.json
```

The build checks the 14 files by SHA-256, builds the v64 chain reference from
the saved zabd values, replays the v64 chain with the release code and runs the
steps of `w2/rebuild.sh`. It refits the step models from the 2025 labels and
reads the saved values of the s7 member (`release/saved-vectors/s7c/`), whose
GPU fits do not repeat bit for bit. The bytes are exact on x86-64 Linux with
the versions of `w2/requirements-remote.txt`; on macOS arm64 the L2C trees
differ, and the record gives the difference to the saved v73 values. It takes
about 2 minutes on an Apple M-series laptop and 4 to 5 minutes on x86-64 Linux
with 8 logical CPUs. [REPRODUCE.md](docs/REPRODUCE.md#release-target-v73) gives
the details.

## Reproduce v64

The caller supplies four official files in this input layout:

```text
input-root/data/raw/submitting.parquet
input-root/data/raw/ranking.parquet
input-root/data/raw/training_2025-01-01_2025-02-01.parquet
input-root/data/raw/training_2025-07-01_2025-08-01.parquet
```

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/build_entry.py --target v64 --data-root /path/to/input-root \
  --out-dir /path/to/new-output --record /path/to/v64-replay.json
```

The replay checks the four files by SHA-256, rebuilds every input frame from
them and the saved member values in `release/saved-vectors/zabd/`, refits the
blend weights and the two rule statistics on the 2025 labels, and runs the
original assembly code in `experiments/exp-169-stack-plan/code/`. It writes
zabd and checks its bytes, then applies the T11b-L stage
(`release/t11b_stage.py`, no fit) and checks the bytes of v64. It refits no
member, and no fresh member fit has reproduced the saved member values. Exact
bytes need numpy 2.4.6, pandas 2.3.3 and pyarrow 24.0.0; another BLAS can
change the last bits, and the record then gives the largest difference in
seconds. The saved values contain no movement IDs or labels; the official
files supply both.

`release/ENTRIES.json` registers the targets; final-v-dh-r9-ld is the default,
`--target final-v-r9-ld` writes its V file, `--target final-v-dh-r9`, `final-v-r9`,
`final-v-dh`, `final-v`, `final-iii-dh-r9-ld`, `final-iii-r9-ld`, `final-iii-dh-r9`,
`final-iii-r9`, `final-iii-dh`, `final-iii`, `final-ii-dh`, `final-ii`, `final-dh` and `final` write the other
final files, `--target v73` writes the ranking-phase entry, and `--target zabd` writes zabd alone. Targets v32, v35, v36, v37, r4-kno, r4-b50
and v40 remain available and need only the submitting and ranking files.

Set `PYTHONDONTWRITEBYTECODE=1` before a checker run with `--root`, too.
Otherwise Python writes cache files into the export and gate D rejects them:

```bash
PYTHONDONTWRITEBYTECODE=1 python release/check_release.py --root /path/to/export \
  --contract /path/to/trusted-MANIFEST.tsv
```

## Licence and data

Source code is **GNU GPL version 3**; see [LICENSE](LICENSE).
External data keeps its own terms; [DATA.md](docs/DATA.md) lists each source
with its licence and hosting. OPDI names no licence; DATA.md declares its use
under the challenge guidance. The exact ERA5 table is a separate CC BY 4.0
bundle: the three assets of the release `era5-v1` of this repository
([table](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/era5_hourly.parquet),
[manifest](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/ERA5-DATA-MANIFEST.json),
[notice](https://github.com/mattmotoki/youthful-giraffe-prc2026/releases/download/era5-v1/NOTICE.txt)). Raw challenge rows and labels are not
redistributed.

`release/MANIFEST.tsv` lists the exported paths and hashes;
`release/check_release.py` checks the tree. Private repository history and
internal coordination records are not included.

The experiment folders keep their working records of method development.
In those records and in `prc/` and `scripts/` comments, "the desk" is the
team lead agent and Matt is the team's human lead. Citations of
`assignments/...` name internal task texts that are not part of this release.

For this release, three kinds of text in the older records were edited:
home-directory paths became `/path/to/...` placeholders, notes on the
availability of internal review tools were shortened, and one personal name
became a role. Nothing else in those records changed.

Some code strings use internal decision labels of the form `Dnn`. Their
records are not part of this release. The glossary below states the rules
and distinguishes diagnostic and test labels. [WRITEUP.md](docs/WRITEUP.md)
describes the accepted model, its gate failure and positive-point rule, and
the current ban on fitting to board readings. Gate K checks narrative
citations, Python comments and docstrings, and whole-line shell comments.
Inline shell comments are outside its narrative scope; it permits these
public glossary definitions and opaque labels in executable code strings.
It also checks links to excluded paths and inline repository path citations
in the README and top-level documents against the export manifest. Fenced
command and input/output layouts are instructions, not path citations.

- `D1` — Diagnostic section: reconcile the no-NM row count; this is not a model rule.
- `D2` — In exp-169 assembly records, the mean squared difference between two prediction vectors, in s² per row; this is not a rule.
- `D3` — In exp-014, report airport quantile shares; in exp-012, follow the team compute split with lane thread caps and GPU allocation; lowercase d3_ weather fields are three-hour differences.
- `D6` — Early release selection: serve the recorded blend-config-5 as entry v2.
- `D9` — Nested validation rule: exclude the scored month from member fits used to estimate blend weights and stackers.
- `D10` — Nested validation work: rebuild member predictions with the scored month excluded before comparing model changes.
- `D11` — No board fitting: submit model candidates, never probes designed to recover hidden labels.
- `D18` — In fitting code, alias of the exp-018 dataset module; also a test example of an opaque source identifier. It sets no fitted model rule.
- `D20` — Tail verdict: keep the tail loss visible as a secondary guard, with the historical pooled reading stated.
- `D23` — Discounted gain gate: cap credited fold gain at its registered prediction and require each credited interval to clear zero.
- `D24` — Seasonal head rule: choose a separate fitting panel for each served month by cyclic month distance.
- `D28` — Redundancy test: compare added-month fits with matched draws before making a new serving claim.
- `D32` — Same-season transport rule: the tested carrier path is closed provisionally; the destination remains open to another mechanism.
- `D36` — Diagnostic advance rule: no negative fold point and clear intervals on two folds, with matched raw-key controls.
- `D43` — Origin diagnostic: use a frozen block-value ladder and matching band; do not treat it as a causal season test.
- `D46` — Near-origin diagnostic: choose the closest qualifying frozen row, stop if none matches, and create no candidate.
- `D47` — Sparse diagnostic contract: keep the registered split, controls, order and stopping rules fixed; no serving claim follows.
- `D107` — Frozen gain gate: require positive fold gains, sham and placebo controls, and clear pooled and per-fold intervals before the registered advance.
- `D119` — Test example of a forbidden narrative citation; it sets no fitted model rule.
- `D123` — Candidate review rule: a fresh-context correctness review of each exact candidate file before any upload.
- `D132` — Positive-point rule: a positive supported-cell gain estimate allowed the final proxy fit even when the frozen gain gate failed.
- `D134` — Current score-feedback rule: no fitted weight, residual, mask or rate comes from a board reading.

No public publication or entry eligibility ruling is claimed by local checks.

Gate L checks every imported file against its pinned full-text line hashes,
including comments and docstrings. Every changed line must be listed in
`release/REWRITE-LINES.json` and must match its expected new text. This check
runs from the release checker and the tests.
