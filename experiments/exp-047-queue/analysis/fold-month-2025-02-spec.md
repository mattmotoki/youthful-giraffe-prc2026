# What a fourth fold month needs: the 2025-02 contract

**Who this is for.** `assignments/foldmonth.md` (recorded rule) sends a worker to produce 2025-02
as a fourth fold month for the surface programme. This file is that assignment's
contract: the exact vectors, paths, columns, keys, formats, producing commits and
expected sizes, so the month can be built **without this lane in the loop**. Written by
`exp-047-queue` on 2026-09-09 at the desk's order (desk pick 2/2).

Where this file and a lane's code disagree, the code wins and the conflict belongs in
the worker's RESULTS. Nothing here asks for a new model, a tuned parameter or a served
file: a fold month is a **reproduction of three committed rotations on one more held-out
month**, and every number below is copied from a rotation that already exists.

---

## 0. The one-line summary, and the thing that is not one line

The surface lanes read **three** vectors on a fold month. Those three sit on top of a
**member layer of seven out-of-fold vectors that is fold-tagged all the way down**, and
that layer does not exist for 2025-02. `exp-026-rebuild/code/frames.py:fold_frame` reads
one file per member per fold, `_read` filters on a `fold` column where there is one, and
`frames.FOLDS` is the literal `("july_loo", "december", "january")`. So the deliverable
is four layers, not three files:

| layer | what | files for `february` | who produces it |
|---|---|---|---|
| 0 | the seven blend-5 members, out of fold, plus exp-016's frozen frame | 8 (§2) | exp-009 / exp-012 / exp-016 — **does not exist, and is the cost** |
| 1 | the window stacker's pair `y_pred_stack5`, `y_pred_depx` (= v6) | `experiments/exp-028-window/oof/exp-028-stack5w-w600-february.parquet` | exp-028 (§3) |
| 2 | the refit-bulk head | `experiments/exp-027-step/oof/exp-027-k2-refit-bulk-february.parquet` | exp-027 (§4) |
| 3 | the climbed member through that same arm | `experiments/exp-035-k2climb/oof/cand-k2c-mod-refit-bulk-february.parquet` | exp-035 (§5) |

Every file is keyed on `MVT_ID_mvt` (float64) and carries **143,731 rows** in the
panel's own order. Layers 1 to 3 are the three vectors the assignment names; **layer 0
is the finding this spec exists to deliver** and the desk should price recorded rule against it
before a kernel is pushed. §2 says exactly what is missing; §9 says which month the
desk should spend it on if it re-opens the choice.

## 1. The fold itself

A fold is a `prc.data.Fold`: a name, a training month tuple, and one validation month.
2025-02 is a **leave-one-out** fold with the same shape as `january` (eleven training
months, one held out), which is what makes it an analogue of a ranking month:

```python
# prc/data.py, beside the november_loo line — additive, no existing fold changes
for _m, _name in (("2025-11", "november_loo"), ("2025-02", "february")):
    FOLDS.setdefault(_name, Fold(_name, tuple(m for m in MONTHS if m != _m), _m))
```

Name it **`february`** (not `february_loo`): `january` is already the leave-one-out
winter fold under a bare month name, and the surface lanes key their records on the fold
NAME, so the two winter folds should read alike.

**`excluded_ids("february")` is empty.** `prc.data.UNREACHED_ROLLOVER` carries entries
for `2025-01` and `2025-05` only, so `rmse_clean` on this fold equals the all-row RMSE
and `rmse_clean` will not raise (the empty-set branch is explicit in the function). Two
consequences the worker must write down rather than discover:

* the month has had **no unreached-rollover audit**; the clean/all-row identity is an
  absence of evidence, not a finding. Say so in RESULTS and, if any row's target exceeds
  the month's 99.99th percentile by an order of magnitude, list it for the desk.
* every "clean" gate quoted for this fold is therefore an all-row gate.

**Size.** `data/raw/training_2025-02-01_2025-03-01.parquet`, 19.8 MB, 287,411 rows, of
which **143,731 are `PHASE_mvt == "DEP"` and all 143,731 carry `TAXITIME_SEC_mvt`.**
That is the row count of every vector in this file. It is the smallest fold month
(28 days; january 153,706, december 165,677, july 190,713).

---

## 2. Layer 0 — the member vectors, which do not exist for 2025-02

**Read this section before starting.** `exp-028`'s stacker does not read the raw month:
it reads `frames.fold_frame(fold, stack5.B5)`, which merges **seven member out-of-fold
vectors** on `MVT_ID_mvt` and then the gate frame. Every one of them is addressed as
`…-{fold}.parquet`, or filtered on a `fold` column, and none of them has a `february`.

| member | address (`{fold}` = `february`) | today |
|---|---|---|
| `T-tailv4` | `experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet`, rows where `fold == "february"` | file exists, folds `july` 190,713 / `july_loo` 190,713 / `december` 165,677 / `january` 153,706 — **no february rows** |
| `C2-cat` | `experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-february.parquet` | **missing** (three folds present) |
| `L2-hub8725` | `experiments/exp-012-blend2/published/L2-hub8725-february.parquet` | **missing** |
| `D1-winter` | `experiments/exp-012-blend2/published/D1-winter-deice-v1-february.parquet` | **missing** |
| `F1-fresh` | `experiments/exp-012-blend2/published/F1-fresh-s1-february.parquet` | **missing** |
| `V3-allin` | `experiments/exp-012-blend2/published/V3-allin-v1-february.parquet` | **missing** |
| `N1-nmtail` | `experiments/exp-012-blend2/published/N1-nmtail-v4-february.parquet` | **missing** (`PARTIAL`: joined `how="left"`, so it may legitimately not cover every row) |
| the gate frame | `frames.gates("february")` reads `C2-cat…-february` for the five regime/guard columns and cross-checks them against `L2-hub8725-february`; `t` and the regime check come from `experiments/exp-016-stack/oof/exp-016-oof-february.parquet` | **missing** |

`gates()` raises if the two producers disagree on any of `noac`, `nm_missing`,
`d_aobt_lobt`, `d_takeoff_aobt`, `airport`, and raises again if the derived regime
differs from exp-016's. Those are the checks that make a new month believable; they
cannot be run on files that do not exist and they must not be switched off.

**Each of the seven is a model refitted with 2025-02 held out.** A member trained on a
set containing 2025-02 leaks the fold, and the leak is invisible downstream: the stacker
would reproduce its own gates and the fold would simply read too good. So layer 0 is
seven member refits (their lineages are `exp-009-noac2` and `exp-012-blend2`; the two
`MEMBER-*.md` files in `exp-012-blend2/published/` carry the recipes), plus one
`exp-016-stack` rotation for the frozen frame. **Price that before pushing anything.**

**The rotation trap, which will look like a bug and is not.** `stack5w.py`,
`refit_bulk.py` and every other rotation in this chain fit fold `g` on *the other folds'
rows*: `src = pd.concat([data[h] for h in FOLDS if h != g])`. Add `february` to
`frames.FOLDS` and july_loo, december and january are each fitted on **three** folds
instead of two — their vectors move, and `stack5w.py`'s own gate (`blend-5 does not
reproduce on {f}`) fails on all three. That failure is the design working. The rule:

* **the three existing folds' files are frozen.** Never overwrite them. If a run
  rewrites them, restore from git before doing anything else.
* february's vector is fitted on the other three folds' rows, which is out of fold for
  february and is the correct construction; it is **not** like-for-like with the existing
  three, which were fitted on two. Write that asymmetry into RESULTS — this lane already
  carries the same sentence for january, which is fitted on two panels where july_loo and
  december are fitted on one (`exp-047-queue/code/arms.py` docstring).

## 3. Layer 1 — the window stacker (`exp-028-window`)

**What it is.** stack-5 with exp-028's 600 s window aggregates inside the stacker, in one
process with the no-window control, so the pair `(y_pred_stack5, y_pred_depx)` comes out
of one fit and the difference between them is the window columns and nothing else.

* **producer** `experiments/exp-028-window/code/stack5w.py` — commit **6c32c23**
  (2026-09-07), unchanged since.
* **command** `PRC_THREADS=2 python stack5w.py` from that `code/` directory, after
  `february` is added to **`experiments/exp-026-rebuild/code/frames.py:FOLDS`** — that
  tuple, not `panel.FOLDS`, is what the rotation iterates (`data = {f: F.fold_frame(f,
  names) for f in F.FOLDS}`). Do not retype a fold list inside the script. Read §2's
  rotation trap first.
* **panel** `experiments/exp-028-window/code/panel.py` (commit **a3ef5c6**, 2026-09-07)
  is what turns the raw month into the modelling frame. `panel.load_fold("february")`
  must work before anything else is run — it is the gate on the month's inputs.
* **output** `oof/exp-028-stack5w-w600-february.parquet`
* **columns, in this order** `MVT_ID_mvt` (float64, the key), `fold` (string, the fold
  NAME, so `"february"`), `y_true` (float64, seconds), `y_pred` (float64),
  `y_pred_depx` (float64, **this is v6**), `y_pred_stack5` (float64).
* **expected size** ≈ 5.3 MB (january 153,706 rows is 5.65 MB, so ~36 bytes/row).
* **row order** the panel's own order for the month; `exp-030`'s `reference()` joins the
  panel on `MVT_ID_mvt` and **fails closed if the join reorders**, so the file must be
  written in panel order and not sorted afterwards.

**Cost.** A laptop stacker rotation: five seeds, 300 rounds, no GPU. Budget it as the
same wall-clock the january fold takes on the same machine, plus 7 % for the extra
training month. Follow `assignments/LAPTOP.md`.

---

## 4. Layer 2 — the refit-bulk head (`exp-027-step`)

**What it is.** The K2 step's head with the blend anchor refitted off the NM tail, EHAM
held at stack-5's own bytes. It is the vector `v10` serves at the eight mask airports.

* **producer** `experiments/exp-027-step/code/refit_bulk.py` — commit **6f2aa5d**
  (2026-09-08). It honours `PRC_TAG`; unset, it writes exp-027's own names. Its fold
  list is `C.FOLDS` from `exp-027-step/code/candidate.py`, so that is the third place a
  fold name has to be added (`prc/data.py`, `frames.py`, `candidate.py`).
* **output** `oof/exp-027-k2-refit-bulk-february.parquet`
* **columns** `MVT_ID_mvt`, `fold`, `y_true`, `y_pred` (the EHAM-masked served vector),
  `y_pred_nomask` — all float64 but `fold`.
* **expected size** ≈ 3.9 MB (january 4.15 MB on 153,706 rows).
* **the EHAM identity is a gate, not a courtesy**: `exp-036/code/bases.py` asserts
  `max |refit-bulk − v7| = 0` on every EHAM row of every fold and stops if it is not
  exactly zero. If the new month fails that assertion the mask was not applied, not that
  the tolerance is too tight.

**If the K2 vector is wanted as well** (`exp-027-k2-k2-hold-february.parquet`, columns
`MVT_ID_mvt fold y_true y_pred y_pred_nomask y_pred_stack5 y_pred_v6`, ≈ 6.7 MB) it is
the same arm with a different anchor and it is **not needed** for `p12`. Produce it only
if the desk asks; `exp-038`'s inventory reads it, `exp-047` does not.

---

## 5. Layer 3 — the climbed member (`exp-035-k2climb`)

**What it is.** exp-035 arm 3b's climbed K2 member pushed through **the same refit-bulk
arm as vector 2**, which is what makes v12 "v11 with only its K2 member replaced".

* **producer** `experiments/exp-035-k2climb/code/rotate_refit_bulk.py` — commit
  **6f2aa5d** (2026-09-08) — which does not reimplement anything: it points `PRC_K2` at
  the climbed member's OOF, tags the run so it cannot collide with exp-027's committed
  vectors, runs `exp-027/code/refit_bulk.py`, moves the artifacts to
  `experiments/exp-035-k2climb/oof/` and records each member file with its SHA-256.
* **command** `PRC_THREADS=3 python rotate_refit_bulk.py --arm mod`
* **output** `oof/cand-k2c-mod-refit-bulk-february.parquet`
* **columns** `MVT_ID_mvt`, `fold`, `y_true`, `y_pred`, `y_pred_nomask`.
* **expected size** ≈ 3.9 MB.
* **the GPU leg.** The climbed MEMBER itself (the input `PRC_K2` points at) is a Kaggle
  kernel: `code/make_kernel.py` + `code/kernel_body.py` (commit **3d5d263**) push it and
  `code/pull_when_done.sh` + `code/land.py` (commit **7319fc9**) bring it back. The
  member must be refitted with 2025-02 held out — a member trained on a set containing
  2025-02 leaks the fold. `rotate_refit_bulk.py` itself is **0 GPU-h**; only the member
  is. Kaggle quota is shared with the harvest: check `kaggle kernels list` before
  pushing, push ONE kernel, and verify within minutes that it started.

---

## 6. What the surface lanes then compute, and why nothing else is needed

None of the following is a file the worker produces; it is what the three files above
make possible, and it is here so the worker can check its output is usable.

```
v6          = oof exp-028 …-february.parquet   y_pred_depx
stack-5     = the same file                    y_pred_stack5
v7          = where(airport == "EHAM", stack-5, v6)            exp-036 bases.eham_mask
refit-bulk  = oof exp-027 …-february.parquet   y_pred
p10 (v10)   = where(airport in KEEP, refit-bulk, v7)           exp-039 v10.load
climbed     = oof exp-035 …-february.parquet   y_pred
p11         = p10 + 0.217 * (stack-5 - p10)                    exp-038 §2c
p12         = masked(climbed) + 0.217 * (stack-5 - masked)     exp-035 §26a
```

`KEEP` is the eight airports of `experiments/exp-036-mask/analysis/mask.json`, read from
the file and never retyped. `0.217` is `base.S5_AMP`.

**The panel columns the lanes need beside the vectors**, all from
`exp-028-window/code/panel.load_fold("february")`: `MVT_ID_mvt`, `airport`, `t` (take-off
clock, epoch seconds), `regime`, `nm_missing`, `noac`, `y_true`, `d_takeoff_aobt`,
`d_aobt_lobt`. `exp-039/code/v10.py` joins the last two on `MVT_ID_mvt` with
`validate="one_to_one"` and stops if the join loses or reorders a row.

---

## 7. The gates a new fold month cannot inherit, and what replaces them

Every loader in this chain gates a vector against **the number its own experiment
published**, per fold — `exp-030 score.PUBLISHED`, `exp-036 bases.PUBLISHED`,
`exp-039 v10.PUBLISHED` and `v10.VECTOR_SHA`, `exp-038 vectors.DIRECT`. **2025-02 has no
published number**, so those dicts have no entry for it and a naive run either raises a
`KeyError` or, worse, gates on three folds and lets the fourth through unchecked.

The rule that replaces them, and the worker's real deliverable:

1. **Prove the reproduction first.** Rebuild an EXISTING fold month (january, the
   smallest of the three) end to end from the commits named above and show it matches the
   committed artifact — `y_pred` value-for-value, and the file's SHA-256 if the writer is
   deterministic. Only a pipeline that reproduces january is allowed to speak for
   february. This is step 1 of the assignment and it is the whole proof.
2. **Publish the new month's numbers as the gate.** For each of the three vectors write
   the clean (= all-row, §1) RMSE on 2025-02 into
   `experiments/exp-049-foldmonth/manifest.json`, and extend each lane's `PUBLISHED`
   dict with the february entry **in the same commit** that lands the vector, so the next
   reader gates on it exactly as it gates on the other three.
3. **Pin the bytes, not just the number.** `exp-039/code/v10.py:vector_sha` hashes
   `MVT_ID_mvt` then `p10`, both float64, in row order; an RMSE gate cannot tell two
   different vectors apart (Sol round 1, blocking 1). Record the february `vector_sha`
   in the manifest.
4. **The board weights do not change.** `SHARES = {july_loo 0.5571, january 0.4429}` are
   the two RANKING analogues and 2025-02 is not one; a february fold enters the record
   the way december does — **beside** the board, never inside it. Do not add a share for
   it and do not re-normalise the two.

**Manifest** — `experiments/exp-049-foldmonth/manifest.json`, one entry per file:
`path`, `sha256`, `rows`, `columns`, `producing_commit`, `command`, `rmse_all`,
`vector_sha` where one is defined, and `built_utc` from `date -u`.

---

## 8. The surface side: what 2025-02 already has, and the one thing it cannot buy

The harvest has **already delivered 2025-02 in full** — nothing to fetch:

* `artifacts/exp-045/summaries/<airport>/2025-02-<dd>.parquet`, **28 days** at EDDF,
  EDDM, EGLL, EHAM, LEBL, LFPG and LSZH; **no files at all** at LEMD, LIRF and LTFM.
* `experiments/exp-045-harvest/analysis/receipt.json` `["2025-02"]`, all 28 days
  measured. Per-airport receipt: EHAM 98.2, LEBL 98.5, LSZH 94.4, EDDF 90.5, EDDM 90.3,
  EGLL 59.4, and **0.0 at LEMD, LFPG, LIRF and LTFM**.
* the callsign frames `artifacts/exp-045/callsigns/callsigns_2025-02-<dd>.parquet`.

So `exp-047`'s block and `exp-046`'s matcher can build 2025-02 the moment the vectors
land, with no new harvest work — and 2025-02 is **already one of the nine prefix-map
months** (`docs/EXTERNAL-DATA.md`), which is unaffected by its becoming a fold month
because the prefix map excludes fold months for a reason that applies to the map, not to
the fold; **the register's line must be re-read when february lands**, since a fold month
inside the prefix map would learn the map partly on the scored month's rows. That is a
decision for the desk, not a step for the worker: name it in RESULTS.

**The one thing this month cannot buy.** 2025-02 is a second WINTER fold for EDDF, EDDM,
EGLL, EHAM, LEBL and LSZH. It is **not** one for Madrid or Rome: LEMD's instrument
arrives on 2025-03-04 and LIRF's on 2025-05-24, so both are dark in 2025-02 and a cell at
either airport gains no lit fold from it (`exp-047` RESULTS §7r). This lane's one revived
cell, `LEMD/reached`, rests on december's 810 rows for its winter evidence and 2025-02
cannot strengthen it. The month buys winter evidence for the six lit airports and for
the class-wide reads, and nothing for the two dated arrivals.

---

## 9. Which month, if the desk re-opens the choice

This section is not a step. It is the one thing the spec turned up that the pick could
not have known, put where the desk can act on it: **the member layer of §2 costs the same
for any month, and 2025-02 is the worst of the eight candidates for the cells that are
now in play.**

Per-airport receipt by month, `exp-045-harvest/analysis/receipt.json` `by_airport.rate`
(%; the three fold months and the two ranking months are marked):

| month | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-01 *(fold)* | 81.2 | 97.1 | 57.1 | 98.0 | 97.7 | **0.0** | 0.0 | **0.0** | 93.3 | 0.0 |
| 2025-02 | 90.5 | 90.3 | 59.4 | 98.2 | 98.5 | **0.0** | 0.0 | **0.0** | 94.4 | 0.0 |
| 2025-03 | 91.4 | 90.4 | 37.0 | 98.0 | 98.4 | 74.4 | 0.0 | **0.0** | 93.7 | 0.0 |
| 2025-04 | 90.6 | 84.0 | 8.7 | 82.4 | 98.4 | 95.7 | 0.0 | **0.0** | 94.2 | 0.0 |
| 2025-05 | 88.6 | 93.3 | 8.7 | 98.4 | 98.7 | 97.9 | 0.0 | 20.5 | 93.7 | 0.0 |
| 2025-06 | 82.4 | 64.7 | 12.0 | 94.2 | 94.3 | 94.0 | 1.5 | 69.2 | 87.9 | 0.0 |
| 2025-07 *(fold)* | 82.9 | 36.9 | 15.1 | 96.2 | 89.2 | 28.8 | 4.8 | 90.6 | 88.5 | 0.0 |
| 2025-08 | 79.9 | 82.1 | 20.9 | 96.4 | 94.4 | 27.1 | 4.0 | 82.7 | 88.5 | 0.0 |
| 2025-09 | 89.3 | 86.1 | 22.1 | 95.7 | 90.2 | 87.8 | 5.2 | 18.4 | 89.0 | 0.0 |
| 2025-10 | 80.7 | 93.8 | 27.8 | 95.6 | 91.5 | 93.8 | 8.6 | 38.2 | 90.4 | 0.0 |
| 2025-12 *(fold)* | 92.6 | 97.1 | 57.0 | 96.5 | 95.9 | 96.5 | 9.2 | 62.4 | 95.9 | 0.0 |
| 2026-01 *(ranking)* | 58.9 | 97.4 | 76.6 | 94.3 | 92.6 | 96.3 | 8.8 | 66.0 | 93.9 | 0.0 |
| 2026-07 *(ranking)* | 43.3 | 96.9 | 97.2 | 96.3 | 95.0 | 60.7 | 7.3 | 0.1 | 93.7 | 0.0 |

2025-11 is not in the table: the harvest has not delivered it, so it cannot be a surface
fold month whatever `prc/data.py` says about `november_loo`.

The trade-off the table exposes, stated without a recommendation:

* **2025-02 is the right SEASON and the wrong INSTRUMENT.** It is a winter month, which
  is what the 2026-01 ranking month needs, and it is the only candidate that shares
  january's weather. But Madrid's feeder arrives on 2025-03-04 and Rome's on 2025-05-24,
  so both are dark in it, and this lane's newly served `LEMD/reached` — whose winter
  evidence is december's 810 rows — gains **nothing** from it. Nor can any LIRF cell.
* **2025-10 is the right INSTRUMENT and the wrong season.** It is the best non-fold month
  for the two dated arrivals (LEMD 93.8, LIRF 38.2) and strong at EDDM and LEBL, but it
  is autumn, so it does not answer the winter question.
* **2025-03 splits them**: Madrid lit from the 4th (74.4 % over the month), still the
  cold half of the year, Rome still dark.

Whichever is chosen, §2's member layer is the cost and §7's gate rules are unchanged.
