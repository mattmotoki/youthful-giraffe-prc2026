# exp-026 rebuild — RESULTS

Rebuilding blend-5, the ablation ladder and stack-5 from the committed members
on Matt's laptop, with the workstation and every git-ignored artifact gone.
Read `SPEC.md` first for the reading this experiment took.

Everything below is computed by `code/`, at 2 threads, in about 90 s of wall
time in total. No model was fitted except the 15 LightGBM trainings of the
stacker rotation. All member bytes are addressed by SHA-256 in §2.

---

## 0. Summary

1. **The rebuild reproduces a shipped artifact to 1.5e-10 s.** blend-5 rebuilt
   here from the committed members equals the object that was actually uploaded
   as `youthful-giraffe_v2` on **all 344,841 rows**: max abs difference
   **1.455e-10 s**, 8.6e-15 relative, which is float64 summation order and
   nothing else. The fold side reproduces `blend-config-4.json`'s and
   `blend-config-5.json`'s published rotations to **±0.0000 s on all six
   folds**. Everything else here rests on those two gates. §1, §6a.
2. **`youthful-giraffe_v2` is blend-5** — settled by SHA-256
   (`5d6a9961…`) and independently by every value. It scored **265.9631**
   against a clean + 4 prediction of 263.13, so blend-5's three fold-measured
   swaps did not carry, and blend-5 landed **0.20 s worse than stack-4**, which
   it beat by 2.36 s on the folds. §6.
3. **Four candidates are built, verified and staged** in
   `submissions/candidates/`: **stack-5** (`94f18e47…`, the `keep-noac`
   variant, the desk's v3), blend-5 (`9167a4ae…`, already on the board as v2),
   and the two ablations that separate the swaps, `b5-drop-V3-allin`
   (`4d4f9f9e…`) and `b5-drop-T-tailv4` (`c47deeb1…`). The three config-built
   vectors cross-check against `prc.blend.build_candidate` at max abs diff
   **exactly 0**. blend-4 itself is **not rebuildable** — two of its six ranking
   inputs were `artifacts/`-only and are gone — so the servable ladder is the
   leave-one-out family over blend-5's seven published members. §7.
4. **stack-5 shipped as `youthful-giraffe_v3` and scored 264.3861 — rank 1 of
   63.** It beat blend-5 by 1.10 s clean on the shared instrument (258.04
   against 259.13, all three folds' day-clustered 95 % intervals excluding
   zero) and it beat both earlier uploads on the board (v1 265.7612, v2
   265.9631). **And the warning this section carried was correct.** §4a said a
   stacker can exploit the leakage channel harder than a 7-weight blend, which
   would make the face-value 262.0 optimistic, and that only the board could
   say. It was optimistic by **2.35 s**. The `noac`-discounted prediction,
   264.59, was right within **0.20 s**. §4d.
5. **Where the lost gain sat.** 85 % of blend-5's
   july gain and 67 % of its january gain over blend-4 sit on the `noac`
   regime — **339 and 63 rows**. The ranking file holds **388** noac rows
   (0.113 %). On december the same change **costs** 30 %. The tail-v4 swap is
   *only* a noac-head change (verified: v2 and v4 differ on noac rows and
   nowhere else, max 1.7e-9 s elsewhere). A gain carried by 0.1 % of rows whose
   sign flips between folds is exactly the shape that does not transfer.
   **stack-5's gain is the opposite shape**: 0.0 % from noac by construction,
   49–87 % from `nm_bulk`, same sign on all three folds. On the served side
   every member that moves the vector at all moves those 388 rows **17 to 28
   times harder** than the other 339,551 (§5d, exact served shifts), so this is
   not one suspect member — it is a 402-row measurement of a 388-row bet.
6. **The noac discount predicted the miss, before the identity was known.**
   Substituting blend-4 on the noac rows costs blend-5 2.27 s (259.13 → 261.40
   clean, predict **265.40**). The board then said **265.9631**: the discount
   accounts for **2.27 of the 2.83 s** miss, residue 0.56 s. The same discount
   leaves stack-5 at **264.31**, still 1.45 s under the board's standing
   265.7612, so the upload decision holds under either currency. §5f, §6b.
7. **The leak has two channels; one is measured, one cannot be closed here.**
   Re-fitting the anchor so no scored fold touches it moves the gain from
   −1.097 to −1.086 s, so the *anchor-weight* channel is worth 0.011 s. The
   *member-feature* channel — the stacker's input columns are member
   predictions from models fitted on month sets that include the scored fold —
   stays open, and closing it needs member refits from training data this
   laptop does not have. §4a states it; §4b prices the repair.
8. **L3, the free re-fits, returns zero — and refutes its own premise
   backwards.** All three free items are killed against the 0.1025 s
   half-width: the clean objective *costs* 0.103 s, and down-weighting december
   because it has no board counterpart **costs 1.173 s of no-noac board** at
   weight 0. December is carrying the rule, not diluting it. Dropping
   `D1-winter` stays 0.044 s, already measured twice, a decision not a run. The
   fourth item, `V3-roll171`, clears its two-fold threshold at **−0.186 s** and
   stays **alive**: it cannot be fitted on this laptop at any price, because the
   v3 frame's weather family needs the ERA5 event-hour snapshot that lives only
   on the workstation. Recorded with its revival condition. §11.
9. **The weather-coverage hypothesis is not supported.** The 2026 ranking
   months joined completely — 44 of 44 ERA5 files, `wx_missing` 0.000000 on
   all 344,841 departures — and `docs/BACKLOG.md`'s "files partly downloaded"
   is stale and now carries a pointer. Two related things stay open and need
   data, not record: July 2026 is served on preliminary ERA5T (`expver 0005`)
   while every fitted hour is final ERA5, 55.7 % of the board; and whether
   V3-allin's twelve-month ranking run kept the same 171 columns its
   eleven-month fold runs did, which `build_v3` guarantees within a run but not
   across runs. §5e.

---

## 1. The gate: does the rule reproduce the published rotations?

`code/reproduce.py`, output in `analysis/reproduce.json`. The rule is refitted
from committed OOF and rotated (fit on two folds, score the third).

| config | fold | mine | published | delta | clean | excluded |
|---|---|---|---|---|---|---|
| blend-4 | july_loo | 289.3189 | 289.3189 | **+0.0000** | 289.3189 | 0 |
| blend-4 | december | 207.7722 | 207.7722 | **+0.0000** | 207.7722 | 0 |
| blend-4 | january | 340.8188 | 340.8188 | **−0.0000** | 222.9606 | 2 |
| blend-5 | july_loo | 286.3274 | 286.3274 | **+0.0000** | 286.3274 | 0 |
| blend-5 | december | 207.0083 | 207.0083 | **+0.0000** | 207.0083 | 0 |
| blend-5 | january | 339.0173 | 339.0173 | **+0.0000** | 220.2052 | 2 |

Clean board 262.013 (blend-4) and **259.134** (blend-5) against exp-012's
issued 262.010 and 259.132 — 0.003 s apart, which is their rounding.

**The serving weights reproduce too, not just the scores.** `configs/config-blend-5.json`
is fitted here on all three folds, the way blend-5's shipped weights were, and
against the 35 numbers in `blend-config-5.json` (five regimes by seven members)
the largest disagreement is **3.6e-15** — floating-point, at `nm_bulk`
F1-fresh 0.4784452030217556 against 0.47844520302175203. So the config this
experiment would serve blend-5 from is numerically the config blend-5 shipped,
and the ladder's other rungs are that same fit with a member removed.

*(`blend-config-5.json` names its members after their SLOTS — `T-tailv2` and
`L2-hub` — while the files behind them are the v4 tail and the 8,725-round
Huber. `code/frames.py` keeps a `SLOT` map for that, because reading the config
at face value builds blend-4's members under blend-5's weights.)*

The frames these numbers come from are built by `code/frames.py` from committed
bytes only. Two independent checks run inside it on every call and raise rather
than warn: the five gate columns are read from **two** producers' OOF files
(C2-cat's pin and L2-hub8725's publication) and must agree row for row, and the
derived regime must equal the `regime` column exp-016 froze into
`oof/exp-016-oof-{fold}.parquet` on every row it covers. Both pass on all three
folds.

**Where the numbers come from, since `artifacts/` is gone.** exp-016's
`members.gates()` reads `artifacts/exp-008/frames/v2-{month}.parquet` and
`artifacts/exp-014/frames/ds-train.parquet`; neither survived. The five columns
the blend rule actually needs — `noac`, `nm_missing`, `d_aobt_lobt`,
`d_takeoff_aobt`, `airport` — are carried inside the C2-cat, L2-hub and
D1-winter OOF parquets themselves, which are committed. That substitution is
the only difference between `code/frames.py` and `members.fold_frame`, and the
±0.0000 above is its proof.

**Disagreement recorded, not averaged.** exp-012's swap table
(`RESULTS.md`, "blend-5's member set, and the three swaps compose") gives
blend-4 289.74 / 208.05 / 223.17 and blend-5 286.79 / 207.13 / 220.39 —
about 0.45 s above the rotated numbers in that lane's own
`blend-config-4.json` and `blend-config-5.json`, on the same configs. Two
instruments, one lane. This experiment agrees with the configs, because the
configs are what the shipped candidates were scored on. The ladder in §3 is
therefore on the config instrument throughout, and its deltas differ slightly
from exp-012's for that reason and not because a member moved.

---

## 2. What survived, and what did not

`published/` and `pinned/` hold every out-of-fold member of blend-4 and
blend-5. On the ranking side, four of blend-4's six inputs are **byte-identical**
to files in `published/` (comparing `pins-members-blend4.json`, the pins
exp-016 took when stack-4 was built, against a fresh `shasum -a 256`):

| blend-4 member | pinned ranking sha256 | surviving file | verdict |
|---|---|---|---|
| C2-cat | `52a453481a8bcc96…` | `published/C2-cat-ranking.parquet` | **identical** |
| D1-winter | `8865d3d866012d14…` | `published/D1-winter-deice-v1-ranking.parquet` | **identical** |
| F1-fresh | `e216ab9e0dc7243a…` | `published/F1-fresh-s1-ranking.parquet` | **identical** |
| N1-nmtail | `5feb03d96490d9e7…` | `published/N1-nmtail-v4-ranking.parquet` | **identical** |
| T-tailv2 | `e9d97994642c9c3a…` | — (`published/T-tailv4-ranking.parquet` is `89706c6c…`, a different model) | **lost** |
| L2-hub | `29adfc27e96dcbcf…` | — (`published/L2-hub8725-ranking.parquet` is `dd0e8710…`, a different model) | **lost** |

The two lost files are `artifacts/exp-012/pred/T-tailv2-ranking.parquet` and
`artifacts/exp-008/pred/L2-hub-ranking.parquet`, and they are exactly the two
members blend-5 swapped. Consequences:

- **`exp-012-blend-4.parquet` is not rebuildable.** It joins `exp-012-blend-2`,
  `exp-012-blend-3`, `stack-1`, `stack-2` and `stack-3` in the class
  `docs/SUBMISSIONS.md` already keeps: numbers intact and honest, rebuildability
  gone. Under the desk's rebuildability rule it cannot sit in the ready queue as
  a candidate. This is a ledger fact, not a candidate.
- **Every intermediate rung of the blend-4 → blend-5 swap ladder is
  fold-only**, because each holds `T-tailv2` or `L2-hub` or both. The one
  exception is the last rung, which is blend-5 itself.
- **`stack-4`, the uploaded `youthful-giraffe_v1`, is not rebuildable either**,
  for the same two files. `candidate-4.json` records what it was.

Members of blend-5, all seven servable, with the SHA-256 of the ranking file
this experiment would blend (`configs/*.json` carry the same list per rung):

| member | ranking file | sha256 (16) |
|---|---|---|
| T-tailv4 | `published/T-tailv4-ranking.parquet` | `89706c6c3a268726` |
| C2-cat | `published/C2-cat-ranking.parquet` | `52a453481a8bcc96` |
| L2-hub8725 | `published/L2-hub8725-ranking.parquet` | `dd0e8710000ab29b` |
| D1-winter | `published/D1-winter-deice-v1-ranking.parquet` | `8865d3d866012d14` |
| F1-fresh | `published/F1-fresh-s1-ranking.parquet` | `e216ab9e0dc7243a` |
| V3-allin | `published/V3-allin-v1-ranking.parquet` | `199fd5146966e98c` |
| N1-nmtail | `published/N1-nmtail-v4-ranking.parquet` | `5feb03d96490d9e7` |

### Why the ranking vectors wait for `data/raw`

`prc.blend.build_candidate` calls the raw data exactly once,
`prc.data.load_ranking`, and uses it for two per-row quantities and nothing
else: the **regime** (`regimes_from_raw`, needing `AIRCRAFT_TYPE_mvt`,
`FLIGHT_ID_mvt`, `AOBT_3_flt`, `LOBT_flt`) and **`d_takeoff_aobt`**
(`MVT_TIME_UTC_mvt − AOBT_3_flt`, the residual guard's argument).

Neither is in a committed file and neither is derivable from the published
members. What is available falls short: `N1-nmtail-v4-ranking.parquet` carries
`has_nm` (339,551 true), `exp-013`'s `nonm-v1-ranking.parquet` carries a `seg`
flag (4,902 true) and a month, and `exp-002`'s `noac_ranking_predictions.parquet`
names the 388 noac rows. Those three counts do not even reconcile with each
other (`has_nm` false is 5,290; exp-014's census gives `nonm` 4,895 and `noac`
388), so they are not three views of one mask. The `nm_late` split — 16,090
rows, 4.67 % — and `d_takeoff_aobt` have no committed source at all. Since
blend-5 uses one weight vector inside `nm_late` and the *same* vector in the
other three regimes, the missing mask changes 4.67 % of rows, which is why no
provisional vector is staged (SPEC, rejected reading 2).

The template is a **second and separate** dependency, needed only by
`prc.data.write_submission` inside `finalise.py`.

**The guard is not a no-op, and that is why `d_takeoff_aobt` is a hard
requirement rather than a refinement.** The two recorded ranking-side counts
are both zero — exp-012's hub8725 build reports `residual guard [-20657, 9069],
touched 0 rows` over 344,841 rows, and `exp-016/candidate-4.json` records
`guard_touched: 0` for stack-4's served vector, whose values run to 115,010 s —
but both are counts at the **output** stage, or for a member that floors
internally. Measured here on the out-of-fold side, member by member:

| fold | rows the member guard moves | which member |
|---|---|---|
| july_loo | 553 (0.290 %) | **F1-fresh only**; the other six members: 0 |
| december | 288 (0.174 %) | **F1-fresh only** |
| january | 199 (0.129 %) | **F1-fresh only** |

`F1-fresh` is the one member whose own writer does not floor — its ranking file
has `min 30.0` against 60.0 for every other member — so it is the only one the
preparation step actually moves, by the floor (37–72 rows) and by the guard
(199–553). Nothing records how many ranking rows it moves. A rebuild that
guessed the guard away would therefore be wrong on a few hundred rows of the
member carrying weight 0.478, which is the largest weight in blend-5.
`code/rebuild_blend.py` counts both member-level and output-level touches into
the staged record, so this becomes a measurement the moment the data lands.

---

## 3. The ablation ladder

`code/ladder.py`; tables in `analysis/ladder_*.csv`, configs in `configs/`.
Every number is the clean rotation (exp-020's exclusion list through
`prc.data.rmse_both`); `predict` is `clean + 4` (desk, 2026-09-06); the clean
board is the two ranking months in quadrature at their row counts.

### 3a. blend-4 to blend-5, one swap at a time (fold-only, not servable)

| config | july_loo | december | january | clean board | predict | vs blend-4 |
|---|---|---|---|---|---|---|
| blend-4 | 289.32 | 207.77 | 222.96 | 262.013 | 266.01 | — |
| +V3-allin | 287.81 | 206.44 | 221.51 | 260.537 | 264.54 | −1.476 |
| +V3-allin +tail v4 | 286.43 | 207.10 | 220.32 | 259.243 | 263.24 | −2.770 |
| **+V3-allin +tail v4 +hub8725 (= blend-5)** | **286.33** | **207.01** | **220.21** | **259.134** | **263.13** | **−2.879** |

### 3b. Each swap alone, off blend-4 — the composition claim re-measured

| config | july_loo | december | january | clean board | predict | vs blend-4 |
|---|---|---|---|---|---|---|
| blend-4 +V3-allin only | 287.81 | 206.44 | 221.51 | 260.537 | 264.54 | −1.476 |
| blend-4 +tail v4 only | 287.57 | **208.41** | 221.37 | 260.339 | 264.34 | −1.674 |
| blend-4 +hub8725 only | 289.19 | 207.61 | 222.81 | 261.874 | 265.87 | −0.139 |

**The three swaps do not compose as cleanly as recorded, and one of exp-012's
three numbers is conditional rather than individual.** Measured alone off
blend-4 the swaps are **−1.476, −1.674 and −0.139**, summing to **−3.289**
against a combined **−2.879**: **0.41 s of interference**, not 0.03 s.

Where exp-012's three came from, checked one at a time
(`exp-012/RESULTS.md` "the three swaps compose", and `docs/LEDGER.md`
2026-09-06 "exp-012 members"):

| swap | exp-012's number | what it measures | independent? |
|---|---|---|---|
| V3-allin | −1.57 | blend-4 → +V3-allin, the first ladder increment | **yes** (mine −1.476, the same object on this instrument) |
| tail v4 | −1.19 | +V3-allin → +V3-allin +tail v4 | **no** — it is the tail swap's value *after* V3-allin is already in. Alone it is **−1.674** |
| hub8725 | −0.14 | a separate unconditional measurement | **yes** (mine −0.139, agreeing to 0.001) |

So the arithmetic in exp-012 is internally correct and two of its three numbers
are what they say. The claim that rests on the third — "the three gains compose
almost exactly … summing to −2.90 against a combined −2.867. Only 0.03 s of
interference" — does not hold, for two reasons. The −2.867 is the sum of the
same ladder's increments, and ladder increments sum to the total by
construction, so comparing them with the total cannot detect interference at
all. And the middle term is conditional, which is exactly where the
interference hides: the tail swap is worth −1.674 alone and only −1.294
after V3-allin, so V3-allin has already taken 0.38 s of what the tail swap
would otherwise have earned. §5a says why — both swaps change the `noac` head
and are largely buying the same few hundred rows twice.

*(Deltas as well as levels differ slightly between the two instruments: −1.476
here against −1.566 there for the same V3-allin swap. The instrument
disagreement is §1's; it does not change the sign or the argument.)*

### 3c. Leave one out of blend-5 — every rung servable, configs written

| config | july_loo | december | january | clean board | predict | vs blend-5 | config file |
|---|---|---|---|---|---|---|---|
| blend-5 −F1-fresh | 290.83 | 211.57 | 222.30 | 262.695 | 266.69 | **+3.561** | `config-b5-drop-F1-fresh.json` |
| blend-5 −T-tailv4 | 287.69 | 206.34 | 221.13 | 260.319 | 264.32 | **+1.185** | `config-b5-drop-T-tailv4.json` |
| blend-5 −V3-allin | 287.42 | 208.24 | 221.20 | 260.178 | 264.18 | **+1.044** | `config-b5-drop-V3-allin.json` |
| blend-5 −N1-nmtail | 286.66 | 207.09 | 220.38 | 259.405 | 263.40 | +0.271 | `config-b5-drop-N1-nmtail.json` |
| blend-5 −L2-hub8725 | 286.52 | 207.15 | 220.43 | 259.341 | 263.34 | +0.207 | `config-b5-drop-L2-hub8725.json` |
| blend-5 −D1-winter | 286.39 | 207.02 | 220.23 | 259.178 | 263.18 | +0.044 | `config-b5-drop-D1-winter.json` |
| blend-5 −C2-cat | 286.27 | 206.97 | 220.21 | 259.099 | 263.10 | **−0.035** | `config-b5-drop-C2-cat.json` |
| **blend-5** | 286.33 | 207.01 | 220.21 | 259.134 | 263.13 | — | `config-blend-5.json` |

`C2-cat` is worth −0.035 s to blend-5, i.e. nothing: dropping it improves the
fold estimate. It survives on the strength of its 0.0298 weight inside
`nm_late` alone. The two uploads that would locate a non-transferring swap are
**−T-tailv4** and **−V3-allin**; both are 1.0–1.2 s worse on the folds, so if
either scores *better* than blend-5 on the board, that member is the one whose
fold gain is not real. Note that −T-tailv4 drops the tail member outright
rather than reverting it to `T-tailv2`, which is not possible any more (§2).

Weights in every config are fitted on all three folds, as blend-5's shipped
weights were.

### 3d. The decomposition COMMON.md requires

**All 14 rungs, not two.** Round 1 listed the intermediate and leave-one-out
rows as missing; every rung was already computed, so this costs only the
writing. `analysis/ladder_decomp.json` and `analysis/ladder_decomp.csv`,
all rows, RMSE within each slice.

**The composing ladder, blend-4 to blend-5:**

| config | fold | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|---|
| blend-4 | july_loo | 289.32 | 251.69 | 2445.72 | 1150.24 | 3526.63 |
| blend-4 | december | 207.77 | 194.35 | 1860.85 | 665.49 | 1965.15 |
| blend-4 | january | 340.82 | 199.48 | 6012.95 | 2850.46 | 3781.37 |
| +V3-allin | july_loo | 287.81 | 248.57 | 2486.22 | 1144.44 | 3483.36 |
| +V3-allin | december | 206.44 | 193.31 | 1835.26 | 669.12 | 1964.89 |
| +V3-allin | january | 339.98 | 198.02 | 6013.35 | 2850.39 | 3729.81 |
| +V3-allin +tail v4 | july_loo | 286.43 | 248.53 | 2440.70 | 1123.63 | 3408.06 |
| +V3-allin +tail v4 | december | 207.10 | 194.09 | 1830.63 | 666.12 | 2008.23 |
| +V3-allin +tail v4 | january | 339.11 | 198.18 | 5987.42 | 2836.41 | 3512.55 |
| +V3-allin +tail v4 +hub8725 (= blend-5) | july_loo | 286.33 | 248.58 | 2435.62 | 1123.30 | 3407.35 |
| +V3-allin +tail v4 +hub8725 (= blend-5) | december | 207.01 | 193.99 | 1830.74 | 666.18 | 2010.46 |
| +V3-allin +tail v4 +hub8725 (= blend-5) | january | 339.02 | 198.11 | 5986.17 | 2836.05 | 3509.48 |

**Each swap alone, off blend-4:**

| config | fold | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|---|
| blend-4 +V3-allin only | july_loo | 287.81 | 248.57 | 2486.22 | 1144.44 | 3483.36 |
| blend-4 +V3-allin only | december | 206.44 | 193.31 | 1835.26 | 669.12 | 1964.89 |
| blend-4 +V3-allin only | january | 339.98 | 198.02 | 6013.35 | 2850.39 | 3729.81 |
| blend-4 +tail v4 only | july_loo | 287.57 | 251.35 | 2395.52 | 1125.02 | 3433.15 |
| blend-4 +tail v4 only | december | 208.41 | 195.14 | 1854.47 | 666.25 | 2017.95 |
| blend-4 +tail v4 only | january | 339.69 | 199.48 | 5982.83 | 2834.32 | 3520.05 |
| blend-4 +hub8725 only | july_loo | 289.19 | 251.65 | 2442.32 | 1149.78 | 3524.62 |
| blend-4 +hub8725 only | december | 207.61 | 194.26 | 1855.70 | 665.11 | 1966.38 |
| blend-4 +hub8725 only | january | 340.70 | 199.36 | 6011.82 | 2850.22 | 3780.29 |

**Leave one out of blend-5 — every rung servable:**

| config | fold | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|---|
| blend-5 -T-tailv4 | july_loo | 287.69 | 248.65 | 2479.54 | 1145.39 | 3486.87 |
| blend-5 -T-tailv4 | december | 206.34 | 193.15 | 1839.10 | 674.35 | 1962.08 |
| blend-5 -T-tailv4 | january | 339.83 | 197.86 | 6011.88 | 2849.83 | 3688.04 |
| blend-5 -C2-cat | july_loo | 286.27 | 248.48 | 2436.57 | 1123.21 | 3407.27 |
| blend-5 -C2-cat | december | 206.97 | 193.94 | 1831.24 | 666.18 | 2010.46 |
| blend-5 -C2-cat | january | 339.02 | 198.11 | 5986.21 | 2836.05 | 3509.48 |
| blend-5 -L2-hub8725 | july_loo | 286.52 | 248.50 | 2444.69 | 1124.11 | 3409.04 |
| blend-5 -L2-hub8725 | december | 207.15 | 194.26 | 1822.07 | 665.10 | 1999.97 |
| blend-5 -L2-hub8725 | january | 339.19 | 198.23 | 5988.89 | 2836.97 | 3518.53 |
| blend-5 -D1-winter | july_loo | 286.39 | 248.53 | 2439.15 | 1123.30 | 3407.35 |
| blend-5 -D1-winter | december | 207.02 | 194.01 | 1830.51 | 666.18 | 2010.46 |
| blend-5 -D1-winter | january | 339.03 | 198.06 | 5987.21 | 2836.05 | 3509.48 |
| blend-5 -F1-fresh | july_loo | 290.83 | 253.48 | 2444.25 | 1135.97 | 3393.06 |
| blend-5 -F1-fresh | december | 211.57 | 197.51 | 1920.99 | 715.36 | 2228.10 |
| blend-5 -F1-fresh | january | 340.05 | 202.13 | 5950.31 | 2829.56 | 3350.62 |
| blend-5 -V3-allin | july_loo | 287.42 | 251.29 | 2391.92 | 1124.58 | 3430.87 |
| blend-5 -V3-allin | december | 208.24 | 195.01 | 1850.73 | 666.19 | 2019.67 |
| blend-5 -V3-allin | january | 339.57 | 199.34 | 5981.65 | 2834.04 | 3517.37 |
| blend-5 -N1-nmtail | july_loo | 286.66 | 248.62 | 2445.92 | 1123.30 | 3407.35 |
| blend-5 -N1-nmtail | december | 207.09 | 193.98 | 1836.63 | 666.18 | 2010.46 |
| blend-5 -N1-nmtail | january | 339.13 | 198.11 | 5989.14 | 2836.05 | 3509.48 |

Read the `noac` column against §5a: it is where the ladder's gain lives and
where it changes sign. blend-4 to blend-5 moves july's noac RMSE 3526.63 →
3407.35 and january's 3781.37 → 3509.48, but december's the wrong way,
1965.15 → 2010.46. Dropping `F1-fresh` is the only change that costs more than
a second everywhere at once, which is the weight it carries.

---

## 4. stack-5

`code/stack5.py`, following `exp-016/HANDOVER.md`. The arm is `a5`
(members only, 5-seed bag, 300 rounds), the one stack-4 shipped; the anchor is
**blend-5's own hybrid NNLS**, rotated so it is out of sample on the stacker's
fitting rows; `keep-noac` serves the 63–339 noac rows from the anchor.

**The baseline reproduces first.** `stack5.py` aborts unless the blend-5
rotation it builds matches `blend-config-5.json` to 1e-3; it printed
`delta +0.0000` on all three folds before any stacker number was computed.

Clean rotation:

| variant | july_loo | december | january | clean board | predict | vs blend-5 |
|---|---|---|---|---|---|---|
| blend-5 (anchor) | 286.327 | 207.008 | 220.205 | 259.134 | 263.13 | — |
| stack-5 full | 285.808 | 205.780 | 219.077 | 258.390 | 262.39 | −0.744 |
| **stack-5 keep-noac** | **285.203** | **205.612** | **219.130** | **258.037** | **262.04** | **−1.097** |

Day-clustered 95 % intervals against blend-5, all rows, `analysis/stack5_rotation.csv`:

| variant | fold | delta | 95 % |
|---|---|---|---|
| stack-5 keep-noac | july_loo | −1.124 | [−1.598, −0.703] |
| stack-5 keep-noac | december | −1.396 | [−1.916, −0.904] |
| stack-5 keep-noac | january | −0.636 | [−1.309, −0.263] |
| stack-5 full | july_loo | −0.519 | [−1.108, −0.019] |
| stack-5 full | december | −1.228 | [−1.767, −0.713] |
| stack-5 full | january | −0.670 | [−1.366, −0.267] |

All six exclude zero. `keep-noac` beats `full` on july and december and is level
on january, the same ordering stack-4 found; it is the variant to serve.

All-rows decomposition (`analysis/stack5_rotation.csv`, columns
`le3600`/`gt3600`/`no_nm`/`noac`):

| fold | variant | rmse | ≤3600 | >3600 | no-NM | noac |
|---|---|---|---|---|---|---|
| july_loo | blend-5 | 286.33 | 248.58 | 2435.62 | 1123.30 | 3407.35 |
| july_loo | stack-5 keep-noac | 285.20 | 248.54 | 2398.03 | 1122.00 | 3407.35 |
| december | blend-5 | 207.01 | 193.99 | 1830.74 | 666.18 | 2010.46 |
| december | stack-5 keep-noac | 205.61 | 193.04 | 1794.01 | 663.60 | 2010.46 |
| january | blend-5 | 339.02 | 198.11 | 5986.17 | 2836.05 | 3509.48 |
| january | stack-5 keep-noac | 338.38 | 197.48 | 5978.89 | 2836.16 | 3509.48 |

### 4a. What 258.037 is, and what it is not

Sol's review of 2026-09-07 (round 1, finding 1) is correct and is recorded here
before the number is used. **The rotation is not clean out-of-sample. It is
fidelity to the historical instrument.** There are two channels, and they are
different in kind.

**Channel A, the anchor weights — measured and small.** The anchor is out of
sample on the stacker's training *rows*: for fold `g` it is the blend fitted on
the other two folds, HANDOVER item 3. It is not nested across folds. Scoring
january, the training anchor includes `anchor[july_loo]`, whose NNLS weights
were fitted on {december, january}, so january's labels reach the stacker's
training residual through 7 weights fitted on roughly 320,000 rows.
`code/nested_anchor.py` closes exactly this channel: scoring `f`, the anchor on
training fold `g` is fitted on the single remaining fold, which is neither `f`
nor `g`.

| variant | july_loo | december | january | clean board | vs blend-5 |
|---|---|---|---|---|---|
| blend-5 (anchor) | 286.327 | 207.008 | 220.205 | 259.134 | — |
| stack-5, shipped anchor | 285.203 | 205.612 | 219.130 | 258.037 | **−1.097** |
| stack-5, anchor channel closed | 285.283 | 205.446 | 219.026 | 258.048 | **−1.086** |

Channel A is worth **0.011 s**, one hundredth of the gain. An earlier draft of
this section called that variant "truly nested" and said it "closes the channel
completely". **That was wrong**, and the sentence is struck: it closes channel A
only.

**Channel B, the member features — open, and not closable on this laptop.** The
stacker's input columns are the members' own out-of-fold predictions. Scoring
july_loo, the stacker still trains on rows from december and january, and the
member predictions on *those* rows come from member models whose training month
sets **include July**. So July's labels reach the stacker through its features,
not only through its anchor. `nested_anchor.py` does nothing about this, and
nothing at this level can: closing channel B needs the level-one members refit
per rotation with the scored month held out of every training set, which needs
the training data and the fitting budget, neither of which exists here (§4b).

**What that means for the numbers.** Three statements, in decreasing strength:

- **Reproduction is proved.** The ±0.0000 gate of §1 shows this code is the
  same instrument the lane has always used. That is a fidelity claim and it
  holds.
- **The comparison is like-for-like.** blend-5 at 259.134 and stack-5 at 258.037
  are measured on the same frames, in the same process, through the same
  channel. The **−1.097 s** delta is the honest content of this section.
- **The level is instrument-relative.** 258.037 is not a clean out-of-sample
  RMSE and is not entered as one. Neither is blend-5's 259.134, stack-4's
  261.49, or any other fold number in this repository — channel B is a property
  of the lane's whole OOF construction, not of stack-5.

**The one asymmetry that the delta does not cover.** A gradient-boosted stacker
has more capacity to exploit channel B than a 7-weight NNLS blend does, so
−1.097 s may be flattered relative to blend-5 by an amount this instrument
cannot see. The board can: `youthful-giraffe_v1` (stack-4, the same arm and the
same channel) read **265.7612** against a clean estimate of 261.49, so the
clean + 4 calibration was itself measured through the channel and carries it.
Predicting **262.0** for stack-5 is therefore consistent, not clean. Desk call,
2026-09-07: stack-5 stays at the head of the ready queue as v3, because the
board is the only clean instrument available and one upload buys the reading.

### 4b. The repair, priced but not run

The fully nested instrument is: for each rotation fold, refit **every level-one
member** with the scored month excluded from its training set, rebuild the
member OOF columns from those refits, then run this same stacker rotation on
them. That closes channel B by construction, and it is the only thing that
does. Cost, from the members' recorded fit times, is about **19 min per fold**
on a Kaggle T4 session, so roughly an hour of GPU per member family plus the
data this laptop does not hold. It is written up as a backlog line, not done
here: it needs `data/raw` and a fitting machine, and it re-measures the whole
lane rather than exp-026 alone. When it runs, the quantity it settles is
whether the stacker's edge over the blend is larger or smaller than −1.097 s —
the level, for every candidate, will still come from the board.

**Not a byte-for-byte reproduction of stack-4's arm.** LightGBM here is 4.7.0
against exp-016's 4.6, so stack-4's own fold numbers are not expected to
reappear and are not claimed. What is compared is stack-5 against blend-5 on
the same frames in the same process, so both sides move together under a
library change and the comparison is fair. That is **not** the same as
version-invariant: a different LightGBM can fit a different model and move the
delta as well as the level. Rerunning under 4.6 is the check to make when a
machine with it returns.

The candidate itself waits on `data/raw/ranking.parquet`. `code/stack5.py`
saves the three folds' rotated vectors under `oof/stack5-{fold}.parquet` in
COMMON.md's out-of-fold format (`MVT_ID_mvt`, `fold`, `y_true`, `y_pred` =
keep-noac) with `y_pred_full` and `y_pred_anchor` beside them, which is what a
serve-time run needs to reproduce its anchor.

### 4c. The serving implementation, and its six checks

Round 1 finding 2 was **"there is no stack-5 serving implementation"**, and it
was right: `stack5.py` measured the rotation and nothing built the object that
would be uploaded. `code/serve_stack5.py` is that implementation and
`tests/serve_check.py` is the evidence for it. The serve path, step for step
after `exp-016/code/submit4.py`:

| step | what serving does |
|---|---|
| members | the seven ranking parquets, floored at 60 then residual-guarded per member, exactly as `build_candidate` prepares them |
| anchor | blend-5's per-regime hybrid NNLS fitted on **all three folds**, applied to those ranking members |
| stacker | **five** LightGBM models (seeds 7, 107, 207, 307, 407; 300 rounds; members-only `a5` columns) fitted on **all three folds' 510,096 rows**, target `y_true − rotated anchor`, bagged by mean |
| serve | `bag + anchor`, then `keep-noac`: the 388 `noac` rows take the anchor |
| finish | floor 60, then the residual guard on the served vector |

The fitting anchor is read back from `oof/stack5-{fold}.parquet`'s
`y_pred_anchor` rather than recomputed, so the serving script and the rotation
cannot disagree about what stack-5 is anchored on. **The read-back is checked
row for row**, not by length: `MVT_ID_mvt` must match the fold frames element
for element and the labels must match exactly, because a per-row misalignment
between the saved rotation and the fold frames would pass a length check in
silence and mistrain all five stackers. That was a self-caught hole of the same
shape as round 1 finding 3's month-only join, found by auditing this file
against that finding rather than by the review.

**The whole path is deterministic.** Rerunning `serve_stack5.py` reproduces
`candidates/stack-5.parquet` **bit for bit** (sha
`484fe8b65e165b31da40df8b75caafa7dedad3a810cc336a08ec7a66dce277e2` on both
runs, max abs difference 0.0 over 344,841 rows), and `finalise.py` reproduces
`submissions/candidates/exp-026-stack-5.parquet` at sha `94f18e47…` again. So
the 15 LightGBM trainings are stable at 2 threads and the SHA quoted to the
desk is the SHA a rebuild produces.

`tests/serve_check.py`, all six passing:

| # | check | result |
|---|---|---|
| 1 | shape | 344,841 rows in the template's id order, float64, all finite, min 61.6152, none under the floor |
| 2 | the anchor **is** blend-5 | max abs diff **0.0** against the staged blend-5 vector, and **1.455e-10 s** against the shipped `youthful-giraffe_v2` object |
| 3 | keep-noac | 388 `noac` rows, **0** of them differ from blend-5; the other 344,453 all move |
| 4 | the stacker | 510,096 fitting rows = 190,713 + 165,677 + 153,706, five seeds, and the fitting anchor differs from the all-fold fit by up to 759.1 s, which is what makes it a rotation |
| 5 | the guard | **it never binds on this candidate**: 0 rows outside the bounds, and the residual range [−12031.2, +4912.4] sits well inside [−20657.0, +9069.0]. The unguarded production control **cannot** make this check fail — the guard is inert, so the two vectors agree either way; what can fail is the **planted** control, 1,000 rows put 5,000 s past the bound, and it is that control alone which proves the call happens |
| 6 | the fit/serve convention | priced below, and gated at 0.0005 s |

**Check 2 is the one that matters.** It closes a chain: stack-5's serve-time
anchor equals the staged blend-5 vector exactly, and that vector equals the
object that was actually uploaded and scored to a ten-billionth of a second. So
the anchor 344,453 of stack-5's rows are built on is not this experiment's
arithmetic — it is a shipped, board-scored artifact.

**Check 5 turned up a finding, by having a control.** Round 2 noted that
asserting the invariant does not prove the guard ran, so check 5 now builds the
same vector one step earlier and compares. The result:

| | rows |
|---|---|
| output rows where the guard **binds** (residual outside the bounds) | **0** |
| output values that change at all | 739 |
| output values that change by more than 1e-6 s | **0** |
| largest change | **8.95e-13 s** |
| member-values where the guard or floor binds | **1** of 1337 that change |

**The residual guard is inert on this candidate.** Its bounds are
[−20657.0, +9069.0] and stack-5's served residuals span [−12031.2, +4912.4],
comfortably inside. The 739 "touched" values are float round-trip noise:
`apply_residual_guard` computes `a − clip(a − y)`, which is not bit-equal to
`y` even when the clip does nothing. An earlier draft of this section reported
those counts as "rows the guard touched", which read as an effect that is not
there; `serve_stack5.py` now reports `guard_bound_*` beside `guard_changed_*`
so the two cannot be confused again.

This corroborates the `exp-018 guard` ledger row from a different lane and a
different lineage: zero rows moved there on all four folds, because the bounds
are the range of the same quantity over training departures and the models do
not leave it. The guard is insurance, and it has never yet been claimed on.

**Check 6, the one convention difference, measured and gated.** The rotation
floors its output and does not guard it; `build_candidate` guards it too.
Applying the guard to the three folds' stack-5 vectors:

| fold | clean | guarded | change | values changed |
|---|---|---|---|---|
| july_loo | 285.2029 | 285.2029 | **+0.0000** | 562 |
| december | 205.6118 | 205.6118 | **+0.0000** | 347 |
| january | 219.1303 | 219.1303 | **+0.0000** | 230 |

Those changes are the same float noise, so the asymmetry between the rotation
and the serve path is worth **0.0000 s** and the check fails above 0.0005 s. It
is the lane's own convention, inherited from `exp-016`, not introduced here.

---

### 4d. The board answered, and it answered the question §4a asked

`youthful-giraffe_v3` = `submissions/candidates/exp-026-stack-5.parquet`, sha
`94f18e47…`, uploaded 2026-09-07 09:56:48 UTC, **scored 264.3861, rank 1 of
63** (desk; `docs/SUBMISSIONS.md` is the desk's file, not this experiment's).

| candidate | clean | face predict | discounted predict | **board** | face error | disc. error |
|---|---|---|---|---|---|---|
| v1 stack-4 | 261.49 | 265.49 | — | 265.7612 | +0.27 | — |
| v2 blend-5 | 259.13 | 263.13 | 265.40 | 265.9631 | **+2.83** | **+0.56** |
| **v3 stack-5** | **258.04** | 262.04 | 264.59 | **264.3861** | **+2.35** | **+0.20** |

**Twice now, the discounted currency has been right and the face value has
missed by more than 2 s.** That is the strongest thing this experiment learned,
and it was not what it set out to measure. §4a warned that a gradient-boosted
stacker has more capacity to exploit the member-feature channel than an NNLS
blend and that 262.0 would be optimistic if so. It was, by 2.35 s — almost
exactly blend-5's 2.83 s miss, on a different head. The channel is not a
stacker-specific effect; it is the whole lane's fold instrument, and its size on
the board is now measured twice at about 2.4–2.8 s.

**The mechanism is the one §5a and §5d isolated.** Both misses are the
`noac`-concentrated part of the fold gain failing to arrive: 388 ranking rows,
0.113 % of the file, that every member disagrees on 17–28× harder than the rest.
Substituting blend-4 on exactly those rows is what the discount does, and it
predicted both boards inside 0.6 s.

**The rule from here** (desk, `BOARD READING 3`): quote every candidate as
**clean-with-noac-at-the-anchor + 4.28**. This experiment's own numbers restated
under it: stack-5 264.59 (board 264.39), blend-5 265.68, the two ablations from
§7's table when their discounts are computed. The face-value column stays in the
tables above because it is what the fold instrument says, but it is no longer
the prediction.

**What still cannot be concluded.** stack-5 beat blend-5 on the board by 1.577 s
(264.3861 against 265.9631) where the fold instrument said 1.097 s, so the
stacker's edge over the blend **transferred and then some** — but v2 and v3 are
one upload each, and the difference between 1.577 and 1.097 is well inside what
two single readings can resolve. The honest statement is that the stacker's
advantage is confirmed in sign and order of magnitude, not that it is larger
than measured.

---

---

## 5. Where the swaps' gain sits, and what that says about transfer

`code/concentration.py` and `code/transfer.py`. RMSE differences do not
decompose; sums of squared error do, so each row group's contribution to
MSE(new) − MSE(old) is exact and the shares below add to 100 %. The two
excluded rollover rows are dropped first, so this is the **clean** gain — the
one the board predictor uses.

### 5a. blend-5's gain over blend-4 is a `noac` gain

| fold | RMSE | nm_bulk (90–96 % of rows) | nm_late | nonm | **noac** | noac rows |
|---|---|---|---|---|---|---|
| july_loo | 289.32 → 286.33 | +16.6 % | +10.1 % | −12.0 % | **+85.4 %** | 339 (0.178 %) |
| december | 207.77 → 207.01 | +64.9 % | +38.0 % | +27.7 % | **−30.5 %** | 89 (0.054 %) |
| january | 222.96 → 220.21 | +30.6 % | +7.1 % | −4.3 % | **+66.5 %** | 63 (0.041 %) |

The ranking file holds **388 noac rows, 0.113 %** (exp-014's census). So about
three quarters of the 2.88 s that separates blend-5 from blend-4 on the two
board months is bought on a class of 402 fitting rows and has to reappear on
388 serving rows — and on the third fold the same change moves the wrong way.

Per swap (`analysis/concentration.csv`, share of that swap's own gain):

| swap | july_loo noac | december noac | january noac |
|---|---|---|---|
| +V3-allin | +62.0 % | +0.1 % | +24.5 % |
| +tail v4 | **+114.6 %** | +42.3 % | **+111.0 %** |
| +hub8725 | +32.9 % | −3.8 % | +5.0 % |

**The tail-v4 swap is nothing but a noac-head change.** `exp-009-noac2-oof-v2`
and `-v3` differ on **333 of 339** july noac rows, **89 of 89** december, **63
of 63** january, and on **zero** other rows: the largest absolute difference
outside the noac class is **1.7e-9 s**, floating-point noise. Over 100 % of its
july and january gain sits there (it loses a little elsewhere), and on december
it is a net loss of 0.64 s. That also explains §3b's 0.41 s of interference:
V3-allin carries its own noac head, so two of the three swaps are improving the
same few hundred rows and cannot both be paid for them.

`hub8725`, the swap that is almost entirely `nm_bulk`, is also the swap worth
almost nothing (−0.139 s).

### 5b. The january gain is not bought on the rollover rows

The concern was that January-dominated calibration hides the gain. Measured
(`analysis/transfer_folds.csv`), the january gain is **larger** clean than raw:

| swap | january raw | january clean | Δ raw | Δ clean | difference |
|---|---|---|---|---|---|
| +V3-allin only | 339.98 | 221.51 | −0.837 | −1.455 | +0.618 |
| +tail v4 only | 339.69 | 221.37 | −1.124 | −1.586 | +0.461 |
| +hub8725 only | 340.71 | 222.81 | −0.114 | −0.151 | +0.037 |
| blend-5 | 339.02 | 220.21 | −1.801 | −2.755 | +0.954 |

Every swap makes the two unreached-rollover rows **worse**, eating about a
third of the raw january gain. So the clean convention is not flattering these
swaps; it is the raw number that understates them. This particular transfer
worry is closed: the gain is not a rollover artefact.

### 5c. V3-allin's fit-serve pairing

`published/V3-allin-v1-ranking.meta.json` and `MEMBER-V3-allin-v1.md`. The OOF
is retro-v1's per-fold fit and the ranking file its twelve-month fit, both
declared at 660 fixed rounds, 1 seed, 12 threads, 171-column v3 frame — the
same shape every member in this blend has, so the fold/twelve-month split is
not by itself a mismatch.

**One thing is unverified, and the lane named it as a live failure mode.** Both
sidecars carry `"column_sha": "declared-by-producer-not-yet-emitted"`.
`MEMBER-V3-allin-v1.md` §"The two failure modes this interface exists to stop"
records failure mode 2, *"wrong frame, right rows"*, as having actually happened
in retro-v1 on 2026-09-06 — `build_submission` building a 50-column frame while
the folds ran on 171, caught only by an unrelated `KeyError` — and states that
**no consumer can detect it from the artifact**. `n_frame_columns: 171` is a
producer declaration, not a measurement. exp-018 emits a real `column_sha`
(`35b5b6687e73`); V3-allin does not. So V3-allin's served frame is the one input
in blend-5 that is asserted rather than checked, and V3-allin is the largest
single swap.

That is a hypothesis with a cheap test and it is **not** a finding that the
member is wrong: retro-v1 can emit the column sha of both fits and compare.

### 5d. V3-allin's shift on the 344,841 ranking rows

`code/transfer.py`, rerun 2026-09-07 16:10 UTC against
`candidates/ranking-gates.parquet`, so **these are the exact served shifts**:
the hybrid rule picks each row's weight vector from the real regime, the floor
is applied, and so is the residual guard. blend-5 minus blend-5−V3-allin, both
built the way `build_candidate` builds them.

| | mean | rms | min | max | \|d\|>60 s |
|---|---|---|---|---|---|
| **served (344,841 rows)** | **−0.816** | **21.75** | −990.7 | +2640.0 | **1.37 %** |

*Round 1 finding 3 was right and is fixed.* Before the data landed this section
carried two unguarded pure-vector populations (bulk 20.55, `nm_late` 36.52) and
called them a bracket. They are not a bracket: each row's served value is one of
the two, so the per-row set is exact, but an aggregate over the mixture takes
subsets of both and need not lie between the two whole-population aggregates.
The served rms, 21.75, does land between them — that is an outcome, not a proof,
and the number quoted now is the measured one.

**By regime** (`analysis/transfer_shift_by_regime.csv`). `noac` and `nonm` are
disjoint from `nm_late` by definition, so those two rows were already right;
"everything else" moves from 15.40 to 16.99 once the guard and the true
`nm_late` mask are applied.

| group | n | mean | rms | max \|d\| | \|d\|>60 s |
|---|---|---|---|---|---|
| **noac** | 388 | +3.85 | **281.25** | 1756.0 | **70.6 %** |
| nonm (exp-013 `seg`, minus noac) | 4,902 | +4.18 | 83.82 | 1454.3 | 17.4 % |
| everything else | 339,551 | −0.89 | 16.99 | 2640.0 | 1.1 % |

**This is the transfer question in one row.** V3-allin moves the served
prediction on the 388 `noac` rows by an RMS of **281 s**, with 70.6 % of them
moving over a minute, against 17.0 s and 1.1 % everywhere else — a **17-fold**
larger disturbance on exactly the 0.113 % of rows §5a says carries three
quarters of the fold gain. Whether that is the gain arriving or a coin-flip is
what the board decides, and `blend-5 −V3-allin` is the upload that asks it.

**And V3-allin is not special in this — every member is.** The same served
shift for each of blend-5's seven members in turn
(`analysis/transfer_shift_all_members.csv`, RMS seconds):

| dropped | noac (388) | nonm (4,902) | everything else (339,551) | ratio noac / else |
|---|---|---|---|---|
| F1-fresh | 883.37 | 174.51 | 31.89 | 28× |
| T-tailv4 | 370.67 | 74.09 | 16.56 | 22× |
| V3-allin | 281.25 | 83.82 | 16.98 | 17× |
| L2-hub8725 | 149.96 | 32.33 | 7.01 | 21× |
| N1-nmtail | 0.00 | 0.00 | 9.20 | — |
| D1-winter | 0.00 | 0.00 | 3.35 | — |
| C2-cat | 0.00 | 0.00 | 1.18 | — |

**The three zeros stay zero, and now for a measured reason.** `C2-cat`,
`D1-winter` and `N1-nmtail` carry weight **0.000** outside `nm_late`, so an NNLS
refit without them returns the same solution on the 328,751 rows that are not
`nm_late` and those rows do not move by one bit — including all 388 `noac` rows,
which is why their noac column is exactly 0.00. Their "everything else" column
is no longer 0.00 the way the unguarded draft had it: it is 1.18 to 9.20 s,
which is their real effect inside `nm_late`, previously hidden because the old
fallback applied one pure vector to every row. They earn their place inside
`nm_late` alone, and now the record shows how much.

The four general movers all move in the same shape: **17 to 28 times harder on
the 388 `noac` rows than on the other 339,551.** So the noac population is
simply where blend-5's members disagree, and the ladder's fold gains are
measured on 402 of them. That is the structural reason a 2.88 s fold gain can
arrive at the board as nothing: it is not one suspect member, it is a 402-row
measurement of a 388-row bet. `blend-5 −V3-allin` and `blend-5 −T-tailv4` are
both built and staged (§7), and §5f prices the whole of it at 2.27 s.

**The census disagreement is settled**, by the gates table, exactly as this
section predicted it would be. exp-013's `seg` flags 4,902 ranking rows;
**4,895 of them are `nonm` and 7 are `nm_bulk`**, which is the whole of the gap
to exp-014's census. exp-002's 388 `noac` ids are the gates' 388 `noac` rows,
**388 of 388, no row either way**. The regime counts are `nm_bulk` 323,468,
`nm_late` 16,090, `nonm` 4,895, `noac` 388, and `nm_late` reproduces exp-014's
16,090 to the row. The three near-views were one mask; the by-regime table above
keeps exp-013's labelling so its rows stay comparable with the earlier draft.

**By month and airport** — and this table is now built from the ranking file's
own `ADEP_mvt`, joined on `MVT_ID_mvt`. Round 1 finding 3 objected that the
earlier version validated a *positional* join on `month` alone, which any
permutation inside a month passes; that check is withdrawn and the join it was
guarding is gone. Six largest cells by rms
(`analysis/transfer_shift_by_airport.csv` has all twenty):

| month | airport | n | mean | rms | p99 |
|---|---|---|---|---|---|
| 2026-07 | LIRF | 15,838 | -1.75 | 52.33 | 110.65 |
| 2026-01 | LIRF | 11,061 | -0.60 | 32.70 | 54.51 |
| 2026-01 | EHAM | 16,232 | +2.58 | 30.90 | 107.91 |
| 2026-01 | LTFM | 22,396 | -0.27 | 23.21 | 64.43 |
| 2026-01 | LFPG | 18,129 | -0.52 | 22.72 | 59.82 |
| 2026-07 | LFPG | 21,743 | -5.25 | 21.41 | 30.94 |

**The claim survives its own correction.** **LIRF july rms 52.3 s** (p99 110.7)
and **EHAM january rms 30.9** (p99 107.9) are the two largest cells, against
11–23 s at most other airport-months, and LFPG july still carries the largest
mean shift at −5.25 s. The positional join had in fact been right — the
verified numbers are within 2 s of the withheld ones — but it was right by luck,
not by evidence, and these are the numbers to quote. LIRF is the airport
COMMON.md already flags as anomalous (naive-predictor RMSE 1658 against 182–325
elsewhere). If V3-allin is the member that did not transfer, LIRF july is where
to look first.

### 5e. Was the 2026 ranking frame fully covered? (desk question, 2026-09-07)

The v3 frame V3-allin sits on is
`("base", "phi", "lirf", "surface", "stand", "weather", "deice", "exot")`
(`exp-001-retro-v1/frame.py`, `SOURCES_V3`, marked FROZEN). A family that was
NaN or default-filled on the ranking months and complete on the folds would
produce exactly a fold gain that does not transfer. What the record settles:

**Weather coverage on the ranking months: complete. Settled, and positively.**

- `docs/weather-provenance.md` §9, dated 2026-09-05: **44 of 44 production
  files present, validated and hashed; every hour of 2025 and both ranking
  months complete for all ten ranking airports.** The five files that were
  missing on the first pass (all four LSZH, LTFM's 2026 extra) were retrieved
  in the same session.
- `exp-005-weather/RESULTS.md` §7, on the live ranking file: **all 344,841
  departures joined, `wx_missing` = 0.000000 in both ranking months.** The only
  nulls anywhere in `WINTER_CORE` are period-edge artefacts of the rolling
  windows — `wx_snow_6h_mm` 15 rows in January (0.010 %) and 157 in July
  (0.082 %), `wx_precip_mm` and `wx_freezing_precip` 11 rows each in July, the
  other four columns zero.
- The missing rule is **no fill** (`weather-provenance` §5): a row with no
  weather hour keeps NaN and raises `wx_missing`, and LightGBM takes the NaN
  natively. So there is no default-filled family here to mistake for data.
- V3-allin's ranking fit ran on 2026-09-06, a day after the retrieval was
  recorded complete.

**`docs/BACKLOG.md` line 5 is stale.** It queues the ERA5 work with the note
"files partly downloaded"; that was true before 2026-09-05 and is refuted by
the two records above. A pointer is added to that line.

**What the record cannot settle — three items, in order of how much they could
be worth.**

1. **July 2026 is served on a different ERA5 release from every hour any member
   was fitted on.** `weather-provenance` §2: the extra product's `expver`
   coordinate is `0001` (final ERA5) for every 2025 hour **and for January
   2026**, and `0005` (preliminary ERA5T) for the 744 hours of **July 2026**,
   whose values "can be revised when the final release appears". July 2026 is
   192,122 ranking rows, **55.7 % of the board**, and it is the one month whose
   weather inputs were never observed in the same release as the training data.
   This is a real fit-serve difference in a family the v3 frame uses, it is
   documented, and it is unquantified. Quantifying it needs a second retrieval
   of July 2026 after the final release — data, not record. Its likely size is
   small (ERA5T-to-ERA5 revisions usually are), and "likely small" is the
   honest state of it.
2. **Whether V3-allin's ranking run used the same columns as its fold runs.**
   Inside a run this is closed by code, not by declaration: `run.py`'s
   `build_v3` freezes the kept-column list on the last training month and
   asserts `list(o.columns) == list(Xtr.columns)` on every target frame,
   including both ranking-month frames, so the "wrong frame, right rows" mode
   `MEMBER-V3-allin-v1.md` warns about would have raised rather than shipped.
   **Across runs it is open.** The dedup list is learned on `months[-1]`, and
   that is a different month for each fold run (11 training months) than for
   the ranking run (12 months, last 2025-12), and `fb.learn_dedup` is
   correlation-driven and therefore data-dependent. `n_frame_columns: 171` in
   both sidecars is a producer declaration and `column_sha` is
   `"declared-by-producer-not-yet-emitted"` in all four of them. Re-running
   `learn_dedup` on the two month sets and diffing the column lists costs
   minutes and needs the training months.
3. **The ranking run's own record is gone.** The sidecar names
   `experiments/exp-001-retro-v1/result-v3-allin-rank.json` as its
   `source_record`; that file is **not in git** and the workstation is. So the
   round count, wall time and any coverage line the ranking run printed cannot
   be read back. The sidecar's own `rounds: 660`, `seeds: 1`, `threads: 12` are
   all it left.

**Two families checked and cleared, so they are not carried forward as
suspects.**

- **deice.** exp-012 restricted the D1-winter member to `ds_deice_hr` because
  the wider 25-column day-state block put 4,640 January 2026 cells outside its
  training range (`exp-012/MANIFEST.md`, blend-3 note). retro-v1's `deice`
  source is **already the bounded one**: `daystate_step.COLUMNS = tuple(DEICE)`,
  and its docstring states "Their wider block harmed january by extrapolating,
  so only this column is taken." V3-allin therefore does not carry the
  extrapolation exposure exp-012 removed elsewhere.
- **surface.** `prc.surface_features` changed a default on 2026-09-05
  (VERSION 3, `dominance="exact"`), which is the drift shape that breaks a
  fit-serve pair. retro-v1 pins **every** behaviour constant by name in
  `surface_step.SURFACE_SETTINGS` rather than defaulting, and V3-allin's fold
  and ranking runs are both dated 2026-09-06, after the change. One
  configuration on both sides.

**So the desk's hypothesis is not supported for weather coverage and stays open
for the ERA5 release and for the cross-run column list.** Neither of the two
open items can be closed from the record; both are minutes of work once
`data/raw` exists, and item 2 is the cheaper and the more likely of the two.

### 5f. The noac gain given back, measured rather than adjusted

`code/noac_discount.py`. The desk carries stack-5 in two currencies
(`docs/SUBMISSIONS.md`, 2026-09-07 08:50). The discount need not be estimated:
substituting **blend-4's** rotated prediction on the noac rows and re-scoring
gives exactly what each candidate is worth if the noac swaps turn out to be
worth nothing. Clean rotation throughout:

| variant | july_loo | december | january | clean board | predict | vs blend-5 |
|---|---|---|---|---|---|---|
| blend-4 | 289.319 | 207.772 | 222.961 | 262.013 | 266.01 | +2.879 |
| blend-5 | 286.327 | 207.008 | 220.205 | 259.134 | 263.13 | — |
| blend-5, noac on blend-4 | 288.883 | 206.774 | 222.042 | **261.399** | **265.40** | +2.265 |
| **stack-5 keep-noac** | 285.203 | 205.612 | 219.130 | **258.037** | **262.04** | −1.097 |
| **stack-5, noac on blend-4** | 287.769 | 205.376 | 220.976 | **260.312** | **264.31** | +1.178 |

Two things follow.

**It accounts for most of the v2 gap, if v2 is blend-5.** blend-5's face
prediction is 263.13 and the board's `youthful-giraffe_v2` is **265.9631** — a
miss of +2.83 s (`docs/LEDGER.md`, 2026-09-07). Discounting the noac gain to
zero moves the prediction to **265.40**, which accounts for **2.27 of those
2.83 s** and leaves 0.56 s. That is not proof: v2's identity is not known, the
discount is a bound rather than a forecast, and 0.56 s is still unexplained.
But it is a mechanism that predicts the right sign and most of the right size,
which the alternatives on the table do not.

**stack-5 is ahead of the board's best under either currency.** 262.04 at face
value and **264.31** fully discounted, against the standing 265.7612. The
discount costs 2.27 s and the lead survives it with 1.45 s to spare, so the
upload decision does not depend on which currency is right.

*(The desk's table quotes 264.6 and 265.7 where this gives 264.31 and 265.40;
the difference is the board constant, `clean + 4` here per the 2026-09-06
convention against `clean + 4.3` from the first calibration reading. Same
measurement.)*

### 5g. Why stack-5 is the safer upload

The same decomposition on stack-5 keep-noac against blend-5:

| fold | RMSE | nm_bulk | nm_late | nonm | noac | y>3600 |
|---|---|---|---|---|---|---|
| july_loo | 286.33 → 285.20 | +49.1 % | +41.6 % | +9.4 % | **0.0 %** | +97.3 % |
| december | 207.01 → 205.61 | +86.9 % | +7.2 % | +5.9 % | **0.0 %** | +36.4 % |
| january | 220.21 → 219.13 | +78.3 % | +14.2 % | +7.6 % | **0.0 %** | +47.8 % |

Zero from `noac` by construction (those rows are served by the anchor), the
gain sitting on the 93.8 % of ranking rows that are `nm_bulk` plus the 4.7 %
that are `nm_late`, and the **same sign on all three folds** — the property
blend-5's swaps do not have. On the transfer argument alone, stack-5's 1.10 s
is a better bet than blend-5's 2.88 s, independently of which is larger.

---

## 6. v2 is blend-5. Settled by hash, and by every value.

**The unlogged upload `youthful-giraffe_v2` is `exp-012-blend-5`, byte for
byte.** `submissions/uploaded/youthful-giraffe_v2.parquet` hashes to
`5d6a99610907bc432e2e46fb9ea11fc8103feb8aa9efdd3f7c86f1188866a4cb`, which is
the SHA recorded for blend-5 when it was issued at 14:42 UTC on 2026-09-06.
Desk, 2026-09-07 16:05 UTC, and confirmed here by
`code/identify_upload.py`. It scored **265.9631**.

`code/identify_upload.py` with no argument compares every parquet under
`submissions/uploaded/` against the recorded hashes, the recorded fingerprints
and every candidate on disk. Both uploads name themselves:

| object | sha256 | hash match | median | board |
|---|---|---|---|---|
| youthful-giraffe_v1 | `48472e44…` | exp-016 stack-4 | 956.4153 | 265.7612 |
| youthful-giraffe_v2 | `5d6a9961…` | **exp-012 blend-5** | 961.9852 | **265.9631** |

### 6a. The rebuild is validated against a shipped artifact

This is the strongest evidence in the experiment, and it is a by-product. The
blend-5 vector rebuilt here from the committed members
(`submissions/candidates/exp-026-blend-5.parquet`) is compared against the
object that was actually uploaded:

| | value |
|---|---|
| rows compared | **344,841** |
| id order | identical |
| **max abs difference** | **1.455e-10 s** |
| rms difference | 1.359e-12 s |
| max relative difference | 8.6e-15 (39 float64 ulps) |

The two files' `MVT_ID_mvt` columns are bitwise identical; the value columns
differ only at the 15th significant figure, which is floating-point summation
order — a different BLAS and thread count on this laptop against the
workstation's. **Nothing of substance differs.** §1's ±0.0000 fold gate proved
this code reproduces the lane's published *rotations*; this proves it
reproduces a candidate that was actually shipped and scored, to a ten-billionth
of a second. Everything the ladder, §5d's served shifts and stack-5's anchor are
built on rests on that.

The container SHAs differ (`9167a4ae…` against `5d6a9961…`) purely because the
value bytes differ in their last bits; both files are single-row-group SNAPPY
parquet written by `parquet-cpp-arrow 24.0.0` with the same two columns. A
rebuilt candidate is therefore not expected to hash-match its shipped original,
and `docs/SUBMISSIONS.md`'s rebuildability note — RMS 7.8 s for a blend pair —
describes rebuilds from *refitted members*, not this case, where the members
themselves are the committed bytes.

### 6b. What the settled identity does to the transfer question

It turns §5's hypothesis into a measurement. blend-5 predicted **263.13 to
263.48** on the clean + 4 rule and scored **265.9631**: the three fold-measured
swaps (V3-allin −1.57, tail v4 −1.19, hub8725 −0.14 clean) **did not carry to
the board**, and blend-5 landed **0.20 s worse than stack-4**, which it beat by
2.36 s on the folds.

§5f predicted exactly this, before the identity was known and without using it.
Giving back the `noac`-concentrated part of blend-5's fold gain — the 388 rows
§5a and §5d isolate — moves blend-5's prediction from 263.13 to **265.40**. The
board said **265.9631**. The discount accounts for **2.27 of the 2.83 s** miss,
and the residue is 0.56 s. That is the closest thing this experiment has to a
forward prediction, and it held.

What it does **not** settle is which swap failed, because all three were shipped
together in one object. `submissions/candidates/exp-026-b5-drop-V3-allin.parquet`
and `exp-026-b5-drop-T-tailv4.parquet` are built and verified (§7) and are the
two uploads that separate them. Ranking them stays the desk's call.

---

## 7. What is built, and how it was verified

`data/raw` landed at 16:04 UTC on 2026-09-07 and everything below is built,
verified and committed. The commands, in the order they ran:

```
.venv/bin/python experiments/exp-026-rebuild/code/regime_table.py
.venv/bin/python experiments/exp-026-rebuild/code/rebuild_blend.py \
    --config .../config-blend-5.json --name blend-5 --expect <fingerprint>
.venv/bin/python experiments/exp-026-rebuild/code/rebuild_blend.py \
    --config .../config-b5-drop-V3-allin.json --name b5-drop-V3-allin
.venv/bin/python experiments/exp-026-rebuild/code/rebuild_blend.py \
    --config .../config-b5-drop-T-tailv4.json --name b5-drop-T-tailv4
.venv/bin/python experiments/exp-026-rebuild/code/serve_stack5.py
.venv/bin/python experiments/exp-026-rebuild/finalise.py
.venv/bin/python experiments/exp-026-rebuild/tests/serve_check.py
.venv/bin/python experiments/exp-026-rebuild/code/identify_upload.py
```

**The gates table.** `candidates/ranking-gates.parquet`, sha
`7f14aeea2219eef62b918637b015268cafe15c905a2d4072b878f791e9e2dbfd`, 344,841
rows: `nm_bulk` 323,468, `nm_late` 16,090, `nonm` 4,895, `noac` 388. `nm_late`
and `noac` reproduce exp-014's census to the row. It also carries `airport`
(`ADEP_mvt`) and `day`, which is what §5d's per-airport table now joins on;
round 1 found that an earlier version probed for a column named `airport`,
which the ranking file does not have, so that field had always been blank.

**The four candidates.**

| candidate | sha256 | min | median | p99 | cross-check |
|---|---|---|---|---|---|
| **exp-026-stack-5** | `94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1` | 61.6152 | 954.3871 | 2569.6764 | §4c, six checks |
| exp-026-blend-5 | `9167a4ae3e7f78aae17b109892b4e555b341b5e6df0289efcbb887a0c1ad91d6` | 60.0219 | 961.9852 | 2507.3347 | **0.0** |
| exp-026-b5-drop-V3-allin | `4d4f9f9e751b2a1d5f935ad39450130b0a47b2e128cd01a7d543fdc8c2e0c039` | 60.0734 | 963.4551 | 2499.3042 | **0.0** |
| exp-026-b5-drop-T-tailv4 | `c47deeb1ea88ce7f44555ab7c17adb7e2422b7c8d966f3d93b83aa8b9f689f00` | 60.0072 | 961.3071 | 2516.8551 | **0.0** |

"Cross-check" is `max |work file − rebuild|` over all 344,841 rows, computed by
`code/verify.py` on the **work file**, before anything is moved into
`submissions/candidates/`. **Exactly zero** on all four. Three of those four
rebuilds are *independent* — a second implementation of the same rule. **stack-5's
is not**: it goes through the same `serve_stack5.build` that produced the file,
so it is a reproduction check. What is independent for stack-5 is its anchor;
see the table below.

**The gate, after round 2 found the round-1 fix incomplete.** Round 1 said
`finalise.py` wrote before it checked; the round-1 fix moved the write into
`work/` but gated only on a config cross-check, and moved the file
unconditionally when no config matched — which is exactly stack-5, the one
candidate queued for upload. It reached `submissions/candidates/` with
`cross_checked: false`, and §7's claim that nothing is written before the check
was false for the candidate it most needed to be true for. `--no-cross-check`
was a second bypass.

`code/verify.py` now holds the rule with no hole in it: **a candidate it cannot
verify is refused, not published.** There is no bypass flag; `--dry-run` stages
and verifies into `work/` and simply does not move. Two verifiers:

| candidate | verifier | independent? |
|---|---|---|
| the three blends | `prc.blend.build_candidate` from the config that declares `_stage_name` | **yes** — two implementations of one rule |
| stack-5 | `serve_stack5.build`, plus structural pins | **no, and it is labelled no** — same builder, so this is reproduction, not independence |

For stack-5 the reproduction catches a stale, corrupt, truncated, mis-ordered or
wrong-variant file. What it cannot catch is a bug inside `build` itself, and
`verify.py` says so in its own docstring rather than claiming more. The
*independent* part is the anchor: stack-5's 388 `noac` rows must equal blend-5
exactly, blend-5 must be present, and **blend-5 is re-verified through
`build_candidate` on the spot** — round 3 found that check optional and its bytes
trusted, so an absent blend-5 skipped it and a tampered one passed it. Both now
fail closed.

**The gate was tested by trying to defeat it**, with
`submissions/candidates/exp-026-stack-5.parquet` at sha `94f18e47…` before each
attempt:

| attack | result |
|---|---|
| corrupt one row of the staged stack-5 vector by +5 s | refused: "differs from a fresh `serve_stack5.build` by 5 s"; destination sha unchanged |
| finalise a candidate with no verifier (`mystery-arm`) | refused: "no verifier … a candidate that cannot be verified is not published"; nothing written |
| serve the `full` variant and publish it as stack-5 | refused: "differs … by 248.058 s"; destination sha unchanged |
| truncate blend-5 to 300,000 rows | refused by `write_submission` before the gate is even reached |

All four left `submissions/candidates/` byte-identical.

**blend-5's fingerprint check, which the assignment asked for, passed**:
`n` 344,841, `min` 60.0, `median` 962.0, `p99` 2507.3 against
`blend-config-5.json`. §6a then went further and matched it against the shipped
object itself, to 1.455e-10 s.

**`finalise.py` publishes only what `verify.py` passed**, re-hashing after the
move to prove the file did not change between the two.

blend-5 needs no upload — it **is** `youthful-giraffe_v2` (§6), and the file is
issued only so the ladder's four vectors sit together with one provenance.

---

## 8. Ledger rows this produces

Nine, appended after the review clears, each citing a path in this repository:

| row | what it records |
|---|---|
| `exp-026 gate` | the rebuild reproduces the shipped `youthful-giraffe_v2` object to 1.455e-10 s |
| `exp-026 stack-5` | the candidate, its −1.097 s, and that the level is instrument-relative |
| `exp-026 serving` | the serving path and its six checks |
| `exp-026 ladder` | the servable ablation ladder's clean rotations |
| `exp-026 transfer` | where the lost gain sat, on the exact served vectors |
| `exp-026 blend-4` | provenance: blend-4 is not rebuildable as a candidate |
| `exp-026 census` | the ranking regime mask reconciled against the ranking file |
| `exp-026 L3` | the free re-fits: two refuted, one decided against |
| `exp-026 V3-roll171` | the one live L3 item and its revival condition |

## 9. Reproducing every number here

2 threads throughout. Run from the repo root; each script prints the tables it
writes into `analysis/`. The first block needs no data:

```
for m in reproduce ladder stack5 concentration noac_discount nested_anchor refit; do
    .venv/bin/python experiments/exp-026-rebuild/code/$m.py
done
```

The second needs `data/raw` (`ranking.parquet` for the gates, plus
`submitting.parquet` for `finalise.py`):

```
.venv/bin/python experiments/exp-026-rebuild/code/regime_table.py
.venv/bin/python experiments/exp-026-rebuild/code/serve_stack5.py
.venv/bin/python experiments/exp-026-rebuild/finalise.py          # the gate
.venv/bin/python experiments/exp-026-rebuild/code/transfer.py     # exact served shifts
.venv/bin/python experiments/exp-026-rebuild/tests/serve_check.py
.venv/bin/python experiments/exp-026-rebuild/code/identify_upload.py
```

`transfer.py` runs after `finalise.py` on purpose: it asserts each served shift
equals the difference of the two verified candidates, so it needs them to exist.
L3 (§11) is `code/refit.py` and needs no data at all.

Verified end to end from a clean `__pycache__` on 2026-09-07: `reproduce.py`
prints `clean+4 266.0126` for blend-4 and `263.1339` for blend-5; `stack5.py`
prints the baseline at `delta +0.0000` on all three folds and `stack-5
keep-noac` at 285.203 / 205.612 / 338.382; `serve_check.py` prints PASS on all
six checks. `stack5.py`, `nested_anchor.py` and `serve_stack5.py` take about
40 s each, the rest are seconds. `tests/shape_check.py` exercises the two
serving scripts against a synthetic gates table and is what proved them before
the data landed.

## 10. Review

**Round 1** — `gpt-5.6-sol` at `xhigh`, fresh context, on commit `23bd33d`,
2026-09-07. Verdict: **blocked**. Log in `logs/sol-review.log`.

Sol confirmed as correct: the `frames.py` gate substitution, the clean scoring
convention, the board formula, `concentration.py`'s decomposition, §2's
provenance, the 0.41 s interference refutation, and that nothing violates
`docs/input-rule.md`. Four blocking findings, all four addressed:

| # | finding | answer |
|---|---|---|
| 1 | the rotation leaks every scored fold's labels through the member features; "truly nested" and "closes the channel completely" are false | **Accepted in full.** The claim is struck. §4a now states both channels, prices the one that can be closed at 0.011 s, and says plainly that the member-feature channel cannot be closed without refits from data this laptop does not hold. 258.037 is labelled instrument-relative everywhere it appears. §4b prices the repair. Desk adjudication nested validation rule, 2026-09-07: the channel is the instrument every fold number in this repository is measured on, stack-4 included, and stack-4 is the one candidate with a board reading, so the numbers stay comparable and the board is the clean instrument. |
| 2 | there is no stack-5 serving implementation | **Fixed.** `code/serve_stack5.py` (written before the review returned, committed after) plus `tests/serve_check.py`, six checks, all passing. §4c. |
| 3 | the transfer tables are not exact served shifts: no residual guard, `_all`/`nm_late` do not bracket, the per-airport join was validated on month alone | **Fixed, and the numbers changed.** `transfer.py` now applies the floor and the guard and uses the real regime mask, so §5d's shifts are exact. The "bracket" claim is withdrawn with an explanation of why it was wrong. The positional join is gone: the per-airport table joins the ranking file's own `ADEP_mvt` on `MVT_ID_mvt`. The LIRF and EHAM claims were re-measured and survive — LIRF july 52.3, EHAM january 30.9. |
| 4 | `finalise.py` can leave a failed candidate in place | **Fixed.** It builds into `work/`, cross-checks there, and moves into `submissions/candidates/` only on success. §7. |

The non-blocking list, all six addressed: `identify_upload.py`'s 8-char prefix
moved out of the full-hash dict into `KNOWN_PREFIX` (it could never have
matched); the fingerprint "settles" claim narrowed to the hash alone;
`rebuild_blend.py` now exits non-zero on a fingerprint mismatch instead of
printing MISMATCH and returning 0; `regime_table.py` reads `ADEP_mvt`, so the
airport field is populated for the first time; the "version-invariant" sentence
in `stack5.py` struck and replaced with what is actually true; `climb.log`
written; `ladder_decomp.json` extended from 2 rungs to **all 14**, with
`analysis/ladder_decomp.csv` beside it.

**Known and accepted, not fixed:**

- The member-feature leakage channel itself (finding 1). Not closable here;
  §4b prices the repair at ~19 min per fold per member family on a fitting
  machine, and it is on the backlog.
- stack-4's own fold numbers are not reproduced, because this laptop has
  LightGBM 4.7.0 against exp-016's 4.6. Only stack-5 against blend-5 in the
  same process is claimed.
- blend-4 is not rebuildable as a candidate and never will be; two of its six
  ranking inputs died with the workstation. It stays a ledger fact.
- exp-013's `seg` over-counts `nonm` by 7 rows (§5d). Nothing here depends on
  it; the exp-013 lane owns the fix.

**Round 2** — `gpt-5.6-sol` at `xhigh`, fresh context, on commit `f5fd0b9`,
scoped by the desk to findings 2 and 4. Verdict: **blocked**, on one finding,
and it was right. Log in `logs/sol-review-r2.log`.

Sol passed finding 2 by code and by direct reproduction — it rebuilt the serving
path in memory and got **max difference 0.0** against the staged vector — and
confirmed finding 1's correction is honest and that the regime mask, the
`ADEP_mvt` join and the bracket withdrawal are correct. What it blocked on:

| # | finding | answer |
|---|---|---|
| 1 | **finding 4 is still open.** `finalise.py` gated only on the config cross-check, so with no config it set `cross_checked: false` and published anyway — stack-5's exact case. `--no-cross-check` was a second bypass. `analysis/finalised.json` recorded stack-5 in `submissions/candidates/` with `cross_checked: false`, so §7's claim was false | **Fixed properly this time.** `code/verify.py` refuses to publish anything it cannot verify, runs on the work file, and has no bypass. Tested by four attacks that each tried to get a bad file into `submissions/candidates/`; all four refused, destination byte-identical. §7 |

Three non-blocking findings, all fixed:

- **`transfer.py` still omitted the per-member floor and guard**, so "exact
  served shifts" was false by up to 14.3 s on 359 rows. The preparation is now
  applied, and the claim is **self-verifying**: the run asserts each shift
  equals the difference of the two verified candidates, and prints it
  (`max 0 s` for both V3-allin and T-tailv4).
- **Three of the six serving checks were weak** — check 4 counted rows instead
  of rebuilding, check 5 could not fail, check 6 never called `check()`. All
  three are now substantive and gating, and check 4 calls `verify.py` itself so
  the test and the gate cannot drift apart.
- **The "no library version can move" sentence** contradicted `stack5.py`. Struck.

**Fixing check 5 turned up a real finding**: giving it a control showed that the
residual guard **never binds** on this candidate, and that the counts this file
had been reporting as "rows the guard touched" were float round-trip noise of
order 1e-13. §4c carries the measurement and `serve_stack5.py` now reports
`guard_bound_*` beside `guard_changed_*`.

**Round 3** — `gpt-5.6-sol` at `xhigh`, fresh context, on commit `9c5ec91`,
scoped by the desk to one question: does the publication gate hold? Verdict:
**blocked**, on three findings, all real and all mine. Log in
`logs/sol-review-r3.log`.

| # | finding | answer |
|---|---|---|
| 1 | **a time-of-check/time-of-use hole in my own round-2 fix.** One predictable work path, and the move happened *before* the re-hash — so a file swapped after verification was already published when the hash check raised | **Fixed.** The work file now carries a per-run unique name (`pid` + random), and the hash is re-checked **before** the move, not after. Tested by swapping the work file inside that exact window: refused, destination byte-identical, no work file left |
| 2 | **the blend-5 anchor check was optional and trusted bytes.** `if blend5.exists()` — so an absent blend-5 skipped stack-5's only independent anchoring, and a present one was believed without being verified | **Fixed.** blend-5 must exist and is re-verified through `build_candidate` inside stack-5's own verification. Tested twice: hiding blend-5 is refused, tampering with it by 7 s is refused |
| 3 | **a config could capture the reserved name.** The config lookup ran before the `stack-5` branch, so a config declaring `_stage_name: stack-5` routed the file through `verify_blend` and skipped every stack-5 pin | **Fixed.** Reserved names dispatch first, and a config that claims one is itself an error. Tested by planting such a config: refused by name |

Four accuracy findings, all fixed:

- **The float64 gate never fired**, because `_values` cast to float64 before the
  dtype was checked. It now reads the **stored** parquet schema; a float32
  column is refused (verified).
- **The cleanup claim was false** for exceptions other than `Failed`. The work
  file is now removed on any exception.
- **Check 5 could not fail**, and my claim that it could was wrong. Because the
  guard is inert on this candidate, deleting the guard call would leave every
  check passing — no test in this file can prove the call was made. The check
  now says exactly that, and a **positive control** proves the guard works when
  it is needed: 1,000 rows planted 5,000 s past the bound, all 1,000 repaired.
- **"All four rebuilds are independent" was wrong.** stack-5's is a reproduction
  check. §7 says so.

Sol also confirmed the candidate's measured claims are sound — the SHA, the
stored dtype, the serving test's exit 0, the guard measurements, and all 388
`noac` rows matching blend-5 — and that the shared-builder circularity is
disclosed and is not itself a blocker.

**Round 4** — `gpt-5.6-sol` at `xhigh`, fresh context, on commit `e082c63`,
scoped to two questions: does the gate hold after round 3, and is the noac
verifier sound? Verdict: **blocked**, six findings. Log in
`logs/sol-review-r4.log`. Sol separately re-derived the uploaded noac candidate
and **confirmed the science**: 388 rows differ from the local verified blend-5
and zero of the other 344,453; those 388 ids exactly equal exp-002's source id
set and the values match bit for bit; the 1.455e-10 s base noise cannot reach
the +9.26 s reading, so §13's attribution stands.

| # | finding | answer |
|---|---|---|
| 1 | **the TOCTOU window is narrowed, not closed**, and worse, a destination that failed the post-move hash was **left published** | **Stated as a limitation, and the second half fixed.** A destination that does not hash to the verified value is now removed and the previous candidate restored, so a failed run can only return the prior state. The window itself is not closable by another check of the same kind — see §10a |
| 2 | **the candidate name was not validated** while the destination is built from it, so `x/../exp-026-stack-5` dispatched as a generic blend and resolved to the reserved stack-5 destination | **Fixed.** Names must match `^[a-z0-9][a-z0-9._-]*$`, carry no `..`, and resolve inside `submissions/candidates/`. Five attacks refused |
| 3 | **the repository root sat ahead of `code/` on `sys.path`**, so a root-level `verify.py` would shadow the gate — in `finalise.py` and `serve_check.py` both | **Fixed twice over.** The root is appended rather than prepended, and `finalise.py` loads the gate from its own path with `importlib`. Tested by planting an approve-everything `verify.py` at the root: the real gate still resolved |
| 4, 5 | **NaN failed open in three comparisons.** They read `if diff > TOL: raise`, and `diff` is NaN whenever either side holds a NaN or a reindex missed a row — and `NaN > TOL` is **False**, so the check passed on exactly the inputs it existed to catch | **Fixed at the root.** Every comparison goes through `close()`, which requires both sides finite and equal in shape and then asks `d <= tol` rather than `d > tol`. Five controls: NaN left, NaN right, shape mismatch, a real 5 s difference, and identical — only the last passes |
| 5 | **the noac row set was a label count**, so a gate table with one label swapped would keep the count and free a row | **Fixed.** The row set now comes from exp-002's 388-row file **pinned by hash**, for every variant, and the mutable gate cache must agree with it |
| 6 | **the noac base was only the local rebuild**, never the uploaded bytes, so a coherently changed base and member set would pass | **Fixed.** `youthful-giraffe_v2` is pinned by SHA-256 and the non-`noac` rows are checked against it at 1e-9, beside the 0.0 check against the local base. Each source file is pinned too |

The desk adjudicated round 4 (2026-09-07) and scoped the repair to findings 2,
3 and the NaN class, with the TOCTOU window recorded as a stated limitation and
**no round 5**: 16 % of the cc2 week was left and the gate is a helper, not a
result. Findings 5 and 6 were already written when that call arrived and are
kept — they cost nothing further, and they are **not reviewed**.

### 10a. What four repairs of one gate say about its shape

The publication gate has now been repaired in four consecutive rounds, and each
round found a fresh bypass in the previous round's fix. That pattern is the
finding. It is not that the fixes were careless; it is that **the gate was
being written as a security boundary, and a path-based check inside a mutable
working tree cannot be one.** Every check has the form *read a path, decide,
then act on that path*, and each one opens a new window of exactly the kind the
last fix closed. Adding a fifth check adds a fifth window.

Two things follow, and both are now in the code:

- **Verify against what cannot change under you.** Round 4's findings 5 and 6
  were both this: the row set came from a mutable cache and the base came from
  a rebuild of mutable inputs. Both now come from artifacts pinned by
  SHA-256 — exp-002's 388-row file, and the uploaded `youthful-giraffe_v2`
  bytes. A pinned reference has no window.
- **Fail closed by construction, not by remembering to.** The NaN class was
  three separate instances of one habit, `if diff > tol: raise`. Writing the
  comparison once, in a helper that demands finiteness and asks `d <= tol`,
  removes the habit rather than the three instances.

And one thing follows that is **not** in the code, by decision: the
check-to-rename window stays open. Closing it needs file-descriptor-based
verification (`O_TMPFILE` + `linkat`) that the parquet readers here do not
offer. **The risk is closed elsewhere and better**: the desk re-verifies every
upload by SHA-256 and a tolerance diff against the base immediately before it
leaves the machine, after any such window has passed. That is the binding
check, and it always was. This gate's real job is to catch operator error — a
stale staged file, the wrong variant, a config typo, a NaN — and for that it is
sound. `finalise.py`'s header states this rather than implying atomicity it
does not have.

**The upload went ahead before this round closed**, on the desk's call
(2026-09-07): Sol's own in-memory rebuild matched the staged vector at 0.0, the
desk checked the file itself, and the cc2 window was about to close. The three
blockers were gate *robustness*, not candidate soundness, and the board reading
(§4d) was worth more than the wait. The gate was finished afterwards for the
record, which is what this section is.

---

## 11. L3, the free re-fits (desk nested validation rule)

`code/refit.py`, `analysis/refit_l3.csv`, `refit_v3roll.csv`,
`refit_interval.csv`, `refit_l3.json`. Weights only, no model, on committed OOF.
The queue's kill rule: a change that does not improve the rotated clean board by
more than the blend lane's measured rotation half-width — **0.1025 s** on three
folds, **0.1687 s** on two — is not adopted. Both currencies are reported,
because §6b showed the discounted one is what predicted the board.

**The instrument passes its own gate first.** `refit.py` fits weighted NNLS,
which is ordinary NNLS on `sqrt(w)·P` and `sqrt(w)·y`; at `w = 1` it must
reproduce the shipped rule, and it does, to **0.000e+00 s** on all three folds.
The baseline no-noac board comes out at **231.2698**, which is
`exp-027 budget`'s ledger row to four decimals — a cross-check of this
instrument against a different worker's code that neither of us arranged.

### 11a. Every free item is killed, and one of them is refuted backwards

| arm | july_loo | december | january | clean board | vs control | discounted | no-noac | face | disc. |
|---|---|---|---|---|---|---|---|---|---|
| blend-5 as shipped (control) | 286.327 | 207.008 | 220.205 | **259.1339** | — | 261.3987 | 231.2698 | — | — |
| 1. clean-objective weights | 286.494 | 206.874 | 220.205 | 259.2366 | **+0.103** | 261.3254 | 231.1867 | killed | killed |
| 2. month-share, december 0.00 | 286.662 | 207.008 | 220.309 | 259.3789 | +0.245 | 262.4352 | 232.4423 | killed | killed |
| 2. month-share, december 0.25 | 286.280 | 207.008 | 220.173 | 259.0929 | −0.041 | 261.6953 | 231.6054 | killed | killed |
| 2. month-share, december 0.50 | 286.338 | 207.008 | 220.209 | 259.1418 | +0.008 | 261.3791 | 231.2475 | killed | killed |
| 2. month-share, december 0.75 | 286.479 | 207.008 | 220.273 | 259.2525 | +0.119 | **261.2066** | **231.0522** | killed | *see below* |
| 1+2. clean + december 0.00 | 286.676 | 206.875 | 220.309 | 259.3877 | +0.254 | 262.2538 | 232.2373 | killed | killed |
| 1+2. clean + december 0.50 | 286.506 | 206.875 | 220.209 | 259.2452 | +0.111 | 261.3080 | 231.1671 | killed | killed |

**Item 1, the clean objective, is refuted.** The hypothesis was that the two
unreached-rollover rows in the january fold pull the fitted vector toward a
population the ranking months do not contain. Removing them from the objective
**costs 0.103 s** of clean board. The mechanism is visible in the row: january's
clean score does not move at all (220.205 either way — those rows are excluded
from *scoring* regardless), december improves by 0.134, and july_loo gets
**0.167 s worse**. So the two rows were not distorting the weights; they were
two of 510,096 rows, and dropping them is a small perturbation that happened to
land the wrong way. exp-017's A1 finding does not generalise one layer up.

**Item 2, month-share weighting, is refuted backwards — and this is the useful
result.** The queue's reasoning was that december contributes 32 % of the fit
rows while having no counterpart in the board mix. The sweep says december is
carrying real information:

| december's weight | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 (control) |
|---|---|---|---|---|---|
| no-noac board | 232.442 | 231.605 | 231.248 | **231.052** | 231.270 |
| clean board | 259.379 | 259.093 | 259.142 | 259.253 | **259.134** |

**Removing december entirely costs 1.173 s of no-noac board and 0.245 s of clean
board.** The curve falls monotonically from 0.00 to 0.75 and then turns, so the
dominant, well-resolved effect is the opposite of the one the queue proposed: a
month with no board counterpart is still worth its rows, because what the fit
learns from it is the *rule*, not the month. That is a fact about this blend
worth keeping, and it is the answer to the item rather than a null.

**The one arm that beats the control does not survive its own sweep.** December
0.75 improves the no-noac board by 0.218 s and the discounted board by 0.192 s,
which clears the 0.1025 s half-width — and it loses 0.119 s on face value. The
day-clustered 95 % intervals against the control on the non-noac rows:

| fold | delta | 95 % | excludes zero |
|---|---|---|---|
| july_loo | **−0.290** | [−0.350, −0.229] | **yes** |
| december | +0.000 | [−0.000, +0.001] | no |
| january | −0.035 | [−0.137, +0.040] | no |

**It does not ship, for three reasons stated against my own result.** It is one
point of five, chosen by the metric then quoted — selection. It sits 0.22 s
below a control that is itself only 0.22 s away, against a sweep whose dominant
slope runs the other way by 1.17 s. And its gain is one fold's: july_loo's
interval excludes zero, january's does not, and january is 44 % of the board.
The trade it makes is real — bulk quality bought with noac-fold quality, which
is the transfer-safe direction the board endorsed once — but a 0.2 s effect
selected from a sweep is not evidence for it. **If the desk wants that trade, the
lever to pull is a member or an objective that targets it, not december's
weight.**

**Item 3, drop `D1-winter`: not run, because it is already measured twice, and
now decided.** `analysis/ladder_loo.csv` and
`docs/queue-subsets-2026-09-07.csv` agree: the six-member rotation is clean
**259.178** against blend-5's **259.134**, a cost of **0.044 s**, under half the
half-width. §5d adds the served-side fact: `D1-winter` carries weight exactly
0.000 outside `nm_late`, so dropping it does not move one bit of the 328,751
non-`nm_late` ranking rows — the 0.044 s is bought entirely inside `nm_late`.

**Desk decision, 2026-09-07: no.** 0.044 s is below the floor, so the member
stays and the rebuild keeps its seventh input. Recorded as decided rather than
open, so the queue does not re-ask it.

### 11b. Item 4: `V3-roll171` clears its threshold, and is not free to serve

| arm | july_loo | january | clean board (2 folds) | vs blend-5 |
|---|---|---|---|---|
| blend-5, same two folds | 286.662 | 220.309 | 259.379 | — |
| `V3-allin` → `V3-roll171` | 286.412 | 220.223 | **259.193** | **−0.186** |

Read against the **0.1687 s** two-fold half-width, −0.186 s **clears**, and it
agrees with the record: exp-012 measured this swap at −0.163 s [−0.362, −0.025]
in the blend. Note the instrument, though — with only two folds the rotation
degenerates to a single fitting fold each time, which is why the blend-5 row
here reads 259.379 rather than 259.134, and it is exactly the "december 0.00"
arm above. Both arms sit on that same weaker instrument, so the *comparison* is
like-for-like even though the *level* is not the shipped one.

**It cannot be served, and not for the reason the queue gave.** The queue
priced this at ≈0.5 session-h — a december fold plus a twelve-month ranking fit
— and treated the cost as the obstacle. The real obstacle is an input, not a
budget. `V3-roll171` sits on the **v3 frame**, whose weather family is built
from the ERA5 event-hour snapshot, and that snapshot lives **only on the
workstation**: this laptop has no Copernicus token, and the token is on Matt's
list rather than obtainable by a worker (desk, 2026-09-07). So the fit cannot be
run here at any price.

**Status: ALIVE, not killed** — this is the one L3 item whose measurement is
positive, and it is blocked on an input rather than refuted.

> **Revival condition.** The workstation returns with its ERA5 event-hour
> snapshot, *or* a Copernicus token reaches this machine and the snapshot is
> re-retrieved (`docs/weather-provenance.md` §9 records the 44-of-44 retrieval
> that produced it). Then: fit `V3-roll171`'s december fold and its twelve-month
> ranking run, and re-measure the swap on the full three-fold rotation against
> the 0.1025 s half-width rather than the two-fold 0.1687 s used here. The
> two-fold reading is **−0.186 s** and exp-012's independent blend measurement is
> **−0.163 s [−0.362, −0.025]**, so the expected three-fold result is a real but
> small gain; the interval's upper end does not exclude a result under the
> half-width.

§5e already flagged the related open question this shares: whether V3-allin's
twelve-month ranking run kept the same 171 columns its eleven-month fold runs
did. Both wait on the same machine.

### 11c. What L3 returns to the queue

L3's band was **0–0.5 s**. Measured: **0 s realised**. Three of the four items
are now measured and none ships; the fourth measures positive and cannot be run
on this machine.

| item | result | state |
|---|---|---|
| 1. clean-objective weights | **+0.103 s** (a cost) | killed, refuted |
| 2. month-share weighting | **+0.245 s** face, **+1.173 s** no-noac at weight 0 | killed, refuted backwards |
| 3. drop `D1-winter` | 0.044 s, measured twice | **desk decision: no**, below the floor |
| 4. `V3-allin` → `V3-roll171` | **−0.186 s** on two folds | **alive**, blocked on the ERA5 snapshot |

**L3 is closed with its negatives written.** The lever returns nothing today.
What it leaves behind is worth more than the zero: december's rows are load
bearing (§11a), the rollover rows never were, and the one live swap is now
recorded with a revival condition instead of a cost estimate that was not the
real obstacle.

Member subsets were not chased, per nested validation rule:
`docs/queue-subsets-2026-09-07.csv` already shows the blend saturated at four
members, the other three adding 0.199 s together.

---

## 12. The noac candidate (desk, 2026-09-07)

The desk asked for **blend-5 as uploaded, carrying blend-4's predictions on the
388 `noac` ranking rows**, so a board reading against v2's 265.9631 prices those
rows directly and decides whether noac work earns a slot. The instrument is
right. The source does not exist.

**blend-4's noac prediction cannot be rebuilt, and this is the same loss §2
recorded.** Rows outside `nm_late` take blend-4's `_all` vector:

| member | weight outside `nm_late` | ranking file |
|---|---|---|
| F1-fresh | 0.6222 | survives |
| **L2-hub** | **0.1538** | **`artifacts/`-only, gone** |
| **T-tailv2** | **0.1213** | **`artifacts/`-only, gone** |
| D1-winter | 0.0683 | survives |
| C2-cat | 0.0366 | survives |
| N1-nmtail | 0.0000 | (zero outside `nm_late`) |

**27.5 % of the weight sits on two members whose ranking vectors died with the
workstation.** blend-4's *fold* numbers are exact — its OOF survives, which is
why §3's ladder and `code/noac_discount.py` are unaffected — but its served
values are gone for good and no arithmetic recovers them.

### 12a. Three substitutes, each measuring something slightly different

`code/noac_swap.py`. Each is blend-5 everywhere and one substitute on the 388
rows. All three are built, verified and staged, because which one earns the
slot is the desk's call and building all three costs nothing.

| candidate | sha256 | substitute | noac shift, rms | mean |
|---|---|---|---|---|
| `exp-026-noac-exp002` | `08d4c6c019763fbb…` | exp-002's purpose-built noac head | **1857.5 s** | +38.9 |
| `exp-026-noac-drop-tailv4` | `af179065c7d7052d…` | the `blend-5 −T-tailv4` rung | 370.7 s | −37.1 |
| `exp-026-noac-drop-v3` | `6706ff97d96936e8…` | the `blend-5 −V3-allin` rung | 281.2 s | −3.8 |

**The defining property is verified at 0.0 against the local base and at 1e-9
against the uploaded bytes** — the two are not the same claim, and an earlier
draft said "exactly" for both. `verify.py`'s `verify_noac_swap` re-verifies
blend-5 through `prc.blend.build_candidate` on the spot, then requires every one
of the
**344,453** non-`noac` rows to equal it — `max abs diff 0.0`, not a tolerance —
and the 388 `noac` rows to equal the committed source. So a board delta against
v2 can only have come from those 388 rows. This verifier is **independent** of
its builder, unlike stack-5's (§7).

**What each one answers.**

- **`exp002` answers the desk's question as asked.** "Is noac work worth a
  slot" is a question about what dedicated noac modelling would produce, and
  exp-002's head *is* that thing, with its own lineage and its own ledger rows.
  It also has the leverage to show up: `exp-027 budget` prices the noac oracle
  at **28.01 s** of clean board, and a 1,858 s rms shift on rows with that much
  headroom can move the board by seconds in either direction.
- **`drop-tailv4` answers the discount's question.** §5a verified the tail v4
  swap is *only* a noac-head change, so this prices exactly what that swap's
  noac head bought on the board — the cleanest possible reading of "was the
  noac fold gain real".
- **`drop-v3`** does the same for the V3-allin swap.

**One caution against the smaller two.** A 371 s or 281 s rms shift on 0.113 %
of rows may not clear the board's own resolution: v1 and v2 differ by 0.20 s in
total, and these move a few hundred rows that are already among the worst
predicted in the file. `exp002` is the one with a reading large enough to be
unambiguous. That is the recommendation; the choice is the desk's.

**None of these is blend-4, and none is labelled blend-4** anywhere — in the
file names, the JSON records, the verifier, or the ledger. The row this
produces says what was substituted and why the thing the desk asked for was not
available.

---

## 13. What the board said about the noac rows (desk, 2026-09-07)

Two candidates that differ from the uploaded blend-5 (v2, 265.9631) on exactly
the 388 `noac` rows and nowhere else now have board readings. §12's verifier is
what makes them readable: every non-`noac` row equals v2 at exactly 0.0, so the
whole delta comes from those rows.

| | candidate | shift RMS on the 388 | board | vs v2 |
|---|---|---|---|---|
| **v4** | exp-002's purpose-built noac head | 1857.5 s | 275.2197 | **+9.2566** |
| **v5** | the tail v4 head's noac change undone | 370.7 s | 266.2918 | **+0.3287** |

### 13a. The price of moving, and the only term that carries information

Squared error is per-row, so with `s` the shift on a row and `e = blend5 − y`
the error blend-5 makes there, the change in the board's total squared error is
**exact**:

```
dSSE = sum s²  +  2 · sum s·e                                            (1)
```

The first term is the **price of moving**. It is paid whatever the direction
and it is known exactly from the two vectors — no labels needed. The second is
the only part that says anything about the truth, and since the board gives
`dSSE`, equation (1) **solves for it**. Writing

```
rho = sum s·e / sum s²                                                   (2)
```

a change pays only when `rho < −0.5`; `rho = 0` is a change uncorrelated with
blend-5's error, which costs exactly its own size; `rho > 0` moves away from the
truth; the oracle is `rho = −1`.

| | shift RMS | price of moving | if rho = 0 the board reads | observed | **rho** |
|---|---|---|---|---|---|
| v4 | 1857.5 s | +7.2005 s | 273.1636 | +9.2566 | **+0.1452** |
| v5 | 370.7 s | +0.2905 s | 266.2536 | +0.3287 | **+0.0658** |

**Most of v4's +9.26 s is not a lost gain — it is the size of the change.** A
shift of RMS 1,858 s on 388 rows that knew nothing at all about the truth would
have cost +7.20 s. exp-002's head cost +9.26 s, so it is worse than an
equally-large random perturbation: it is positively aligned with the error
blend-5 already makes.

### 13b. rho belongs to a change *and the base it was made from*

Reversing a shift sends `rho` to `−(rho + 1)` exactly, because the reversed
shift is measured against an error that has itself moved by `s`. So a change
must be read from the base it was really made from, and read that way the two
readings say opposite things (desk, BOARD READING 5 CORRECTION):

| change, read from the base it was made from | rho | reading |
|---|---|---|
| **adding tail v4's noac head** to the head before it (v5 → v2) | **−1.066** | direction-correct, near the oracle's −1 |
| **adding exp-002's head** to blend-5's (v2 → v4) | **+0.145** | away from the truth |

**The tail v4 noac change is the one direction-correct noac change on record.**
It aimed essentially straight at the truth. It gained only 0.33 s because it is
small — 371 s RMS on 0.113 % of rows — not because it failed to transfer. The
fold credited it with 1.19 s and the board paid 0.33 s, a ratio of 0.276; with
the direction near-perfect, **that shortfall is a magnitude and population
effect, not a direction effect**, and the earlier "noac gains transfer at about
a quarter" reads better as: the fold overstates the size of a noac gain, and it
cannot see direction at all.

exp-002's head, by contrast, is fold-optimal on 2025 and points the wrong way on
2026. Both statements come from the same two board points.

### 13c. The cells, as a reading — not a plan

`analysis/noac_cells.csv` (exp-002's own `mvt_sched` bands) and
`analysis/noac_regime_cells.csv` (its three-way regime). `P(placeholder)` is
measured on the 1,507 noac rows of 2025; `rollover` is the label-free test in
`prc.tail_features._sched_date_cross` — `mvt_sched ≥ 8 h` **and** a local
calendar-day crossing — which on 2025 caught all 11 rollovers above 8 h and no
non-rollover.

| regime | n | blend-5 mean | exp-002 mean | shift RMS | share of v4's price |
|---|---|---|---|---|---|
| placeholder-band (`3600 ≤ mvt_sched < 8 h`, or ≥ 8 h without a date cross) | 338 | 6,086 s | 5,920 s | 1,783 s | **80.3 %** |
| rollover (`≥ 8 h` **and** date-crossing) | 12 | 56,882 s | 60,042 s | 3,662 s | 12.0 % |
| genuine-band (`< 3600 s`) | 38 | 2,048 s | 2,926 s | 1,649 s | 7.7 % |

By band, the mass sits in `2-4h` (38.6 %) and `8h+` (29.1 %), where
`P(placeholder)` on 2025 is 0.59 and 0.78 — the cells where the two-point
mixture is most nearly a coin flip and a head has the most room to be confidently
wrong.

**This table stops here.** Because `dSSE` is additive over disjoint rows, one
upload carrying a head on one cell only would measure that cell's `rho` exactly,
and the desk asked for two such probes. **score-feedback rule (desk, 2026-09-07) forbids them
and is right**: choosing noac predictions by what the board pays for them is
learning from the ranking process, which `docs/input-rule.md` prohibits. The
cell shares above are a decomposition of a price already paid, which is a
reading of two uploads that were made for another reason. The per-cell `rho`
stays unmeasured, and this is the record of why.

### 13d. What this leaves

- **The lever is real and large.** blend-5's own error on these 388 rows is
  **3,542 s RMS** — the value implied by exp-027's 28.01 s noac oracle through
  equation (1) at `rho = −1`: removing that error removes 4.867e9 s² of board
  squared error, which over 388 rows is 3,542 s RMS. That is the largest single-cell headroom
  in the record.
- **Nothing in the record knows how to spend it.** Of the two noac changes with
  board readings, one is direction-correct and too small to matter and the other
  points the wrong way. Every other noac head was selected on the fold
  instrument, which equation (1) shows cannot see direction.
- **The revival condition is a 2026-valid noac instrument**, not another head:
  something that gets `rho < −0.5` without asking the board. The placeholder
  probability is the quantity to attack — 80 % of the price sits in the band
  where it is nearest a coin flip. `BACKLOG.md` carries this.
