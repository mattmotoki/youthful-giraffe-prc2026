# exp-028 window — RESULTS

The transient airport window on stack-5. Read `SPEC.md` first, including its §6
addendum: the construction changed once, from the second-layer arm the assignment
specified to the one-layer stacker the desk's path hint of 09:10 UTC named, because
the first was measured at +0.002 s and the second at ten times that.

Everything here is computed by `code/` from committed bytes plus `data/raw`, at 2
threads, in about 20 minutes of wall time. Reproduction commands in §9.

---

## 0. Summary

1. **The base reproduces exactly.** stack-5 keep-noac, read back from
   `experiments/exp-026-rebuild/oof/`, is 285.2029 / 205.6118 / 219.1303 clean and
   258.0372 on the board, matching exp-026's published rotation to ±0.0000 on all
   three folds. `code/panel.py` aborts if it does not. Every number below rests on it.
2. **The assignment's construction does not survive the better base.** A LightGBM on
   stack-5's residual fed 600 s window aggregates takes **+0.002 s** of clean board
   and moves january the wrong way, against **+0.591 s** for the same code on
   blend-5's residual. Its `row` control loses 2.25 s and its `shuffle` control 1.66 s
   on stack-5, so a second residual layer on a residual stacker pays a refit cost the
   window columns only buy back. §2.
3. **The component is not consumed, so the path was wrong and not the destination.**
   The airport × 600 s oracle is **14.62 s** of clean board on stack-5's residual
   against 15.11 s on blend-5's, and the error correlation between departures under
   two minutes apart is **0.168 against 0.170**. stack-5 removed 1.1 s of error
   without touching the transient. §3.
4. **The one-layer construction is the candidate.** In its **pre-registered** 600 s
   form it is **+0.414 s of clean board**, 258.0372 → 257.6230, predicting 261.62;
   folds **+0.481 july_loo [−0.866, −0.115]**, −0.042 december [−0.636, +0.763],
   +0.314 january [−0.542, +0.140]. Only july_loo's interval excludes zero. December
   is worse by less than the rotation's 0.1025 s noise floor with an interval covering
   zero, so it is a **MAYBE** and not a FAIL under the desk's gate of 09:28 UTC. The
   variant that ships is the 300 s one in item 8. §4.
5. **December's loss is `nm_late` and a handful of rows, not the bulk.** `nm_bulk`
   gains on all three folds (+0.383 / +0.237 / +0.066 s of fold RMSE); december's
   aggregate is set by `nm_late` at −0.227 and by its most-moved 1 % of rows, which
   carry **366 %** of that fold's whole squared-error change. Five of ten airports
   gain on all three folds; EHAM loses on all three. §5.
6. **A conservative variant passes the gate outright**: the block served on `nm_bulk`
   only is +0.383 / +0.237 / +0.102, board **+0.274**. It is 0.14 s of board cheaper
   and its regime set was chosen after seeing the regime table. `serve.py --regimes
   nm_bulk` is one command. §6.
7. **The arrivals' unmasked taxi-in is a measured negative.** No committed
   out-of-fold file carries arrival rows, so this could not be answered until
   `data/raw` landed at 09:05 UTC. Built (`code/arrivals.py`) and measured: the
   arrival block **alone** loses 0.309 s and loses on all three folds; added to the
   departure block it is worth +0.013 s of board and makes december worse. §7.
8. **A 300 s window has all three folds positive**: +0.484 / **+0.027** / +0.289, board
   **+0.407**, with no regime chosen after the fact. It is the shipping candidate. But
   the width was screened on the same three folds it is scored on **and was chosen for
   crossing the sign gate on them**, so that pass is exploratory and not a second
   confirmation, and no number here bounds what the choice cost. The pre-registered
   600 s form is quoted beside it throughout. December's sign flips between the two
   widths for a swing smaller than the noise floor, which is the clearest evidence in
   this file that december's deficit is noise. §7b.
9. **Three candidates are staged and all three verify.** SHA-256
   `dfa4d780…` (300 s, shipping), `d55b1de9…` (600 s, pre-registered) and
   `52c9f4ac…` (600 s, `nm_bulk` only). Each is 344,841 rows in the template's order,
   all finite, none below the floor; each was rebuilt from its own record without
   calling `serve.py` at a **maximum absolute difference of 0.0** — which proves
   deterministic replay and file assembly, not a second implementation, because the
   two paths share the block builders and the arm (§8e); and every row outside the
   served regimes is
   `exp-026-stack-5.parquet`'s own bytes, bit for bit. Staging twice in separate
   processes reproduced all three SHA-256 values. §8d–8e.
10. **The fitted and served features are the same function, checked on real months.**
   `tests/panel_vs_raw.py` reads each fold's real month from `data/raw`, masks it the
   way the ranking file is masked, runs the serving adapter on it and compares to the
   fitted panel: **zero differing values on every column and zero of 26.0 M design
   cells**, with the out-of-fold rows covering 100 % of each month's departures.
   `tests/shape_check.py` does the algebraic round trip beside it — Sol round 1
   (Standards 6) is right that the round trip alone could not have caught a wrong
   source column, and `panel_vs_raw.py` is what closes that. §8b.

**What the Sol round changed.** No finding moved a staged candidate's number and the
leakage contract passed for the shipping block. Four measurements were recomputed on the
right population or with a corrected estimator — the oracle (§3), the intervals (§2 and
§6), the no-NM slice (§5) — and six claims were weakened or withdrawn: the gate reading
(§2), the width-selection cost (§7b), the regime-selection bound (§6), the independence
of the rebuild (§8e), the arrivals' block-time item (§7a, closed by an identity), and the
"cancels out" reading of the rotation (`SPEC.md` §3). Every correction is marked in place
and the superseded values are kept beside the new ones. `review/sol-round1.log`.

---

## 1. The base, and the gate on it

`code/panel.py`. The base is **stack-5 keep-noac** — exp-026 §4's shipped variant —
read back from `experiments/exp-026-rebuild/oof/stack5-{fold}.parquet`, joined to
`exp-027-step`'s head frame for the regime and the two deltas and to `exp-016`'s
frozen frame for the take-off time `t`. Three checks run on every load and raise
rather than warn: the joins must cover every stack-5 row without reordering it, no row
may lack a take-off time, and `exp-027-step`'s `y_pred` must equal stack-5's
`y_pred_anchor` to 1e-9 (it does, exactly — the two OOF families describe the same
rows).

| | july_loo | december | january | clean board |
|---|---|---|---|---|
| blend-5 (anchor) | 286.3274 | 207.0083 | 220.2052 | 259.134 |
| **stack-5 keep-noac (the base)** | **285.2029** | **205.6118** | **219.1303** | **258.0372** |
| exp-026 published | 285.2029 | 205.6118 | 219.1303 | 258.0372 |

Rows: 190,713 / 165,677 / 153,706. January excludes 2 unreached-rollover ids from the
clean score (`prc.data.UNREACHED_ROLLOVER`); its all-rows RMSE is 338.3816.

---

## 2. The assignment's construction, and its two controls

`code/arm.py candidate`, 600 s window, 400 rounds, 31 leaves, residual winsorised to
±1200 s, fitted on two folds and scored on the third. The configuration was fixed
before any stack-5 number was computed: it is the one K1 chose on blend-5.

Clean board gain against the base, and the per-fold gains:

| arm | july_loo | december | january | clean board | no-noac board | three folds |
|---|---|---|---|---|---|---|
| **window** | +0.187 | +0.486 | −0.300 | **+0.002** | +0.059 | no |
| `row` control | −2.987 | −0.153 | −1.085 | −2.248 | −1.603 | no |
| `shuffle` control | −2.239 | +0.043 | −0.753 | −1.663 | −1.455 | no |

`row` is the same fit with every window aggregate deleted; `shuffle` is the same fit
with the window block permuted **within airport-day**, so each window column keeps its
marginal distribution and each airport-day its own set of states, and only the
assignment of a state to a departure is destroyed (98.8 % of rows move). Both controls
lose heavily, which is the evidence the desk asked for that this rotation does not hand
out gains for free — and it is also the diagnosis: the window columns are worth about
2.25 s **against a refit that costs 2.25 s**, so the net is zero.

**The same code on blend-5's residual, for comparability** (`code/diagnose.py`):

| base | arm | july_loo | december | january | clean board | no-noac |
|---|---|---|---|---|---|---|
| blend-5 | window | +0.550 | +1.237 | +0.671 | **+0.591** | +0.796 |
| blend-5 | `win_only` (window columns only) | +0.778 | +0.185 | +0.435 | +0.643 | +0.603 |
| blend-5 | `row` | −2.307 | +0.656 | −0.072 | −1.449 | −0.897 |
| blend-5 | `shuffle` | −1.675 | +0.809 | +0.272 | −0.930 | −0.720 |
| stack-5 | window | +0.187 | +0.486 | −0.300 | +0.002 | +0.059 |
| stack-5 | `win_only` | +0.174 | −0.381 | −0.100 | +0.069 | +0.077 |
| stack-5 | `row` | −2.987 | −0.153 | −1.085 | −2.248 | −1.603 |
| stack-5 | `shuffle` | −2.239 | +0.043 | −0.753 | −1.663 | −1.455 |

The blend-5 row reproduces K1's finding in sign, in the three-fold pass and within
0.08 s of its board figure (K1: +0.669 clean, +1.073 no-noac; here +0.591 and +0.796).
The remaining difference is the scoring convention: K1 dropped the two unreached-rollover
ids from the frame and scored plain RMSE on the rest, while this file keeps them in the
population, drops them from the fitting rows, and scores `prc.data.rmse_clean`.

**Deleting the refit channel does not recover it.** `win_only` — the window aggregates
alone, no `base`, no `own_ta`/`own_al`, no hour, weekday or airport — is +0.643 s on
blend-5 and **+0.069 s** on stack-5. So the failure is not the row columns.

---

## 3. Why: the component is intact, the second layer cannot reach it

`code/diagnose.py`. Two statistics from `docs/STRUCTURAL-STEP.md`, recomputed on the
residual each arm actually faces.

**The oracle that knows every airport × window cell's mean error**, as a clean-board
gain. `raw` is `Σ n·m²`; `unbiased` subtracts one within-cell variance per cell, which
is what `E[n·m̄²] = n·μ² + σ²` costs. A cell of one row has no mean to estimate and
contributes nothing to `unbiased`.

| window | blend-5 raw | blend-5 unbiased | stack-5 raw | stack-5 unbiased | singleton cells |
|---|---|---|---|---|---|
| 300 s | 69.086 | 14.997 | 68.518 | 14.568 | 12.5–15.7 k per fold |
| **600 s** | 44.196 | **15.114** | 43.663 | **14.619** | 2.7–4.1 k |
| 1800 s | 23.886 | 12.229 | 23.425 | 11.771 | 0.5–0.6 k |

> **Corrected after Sol round 1** (Standards 1), `analysis/diagnose_oracle.json`;
> the superseded values are kept beside it as `diagnose_oracle-superseded.json`.
> Two defects, both in the same direction: the two excluded january rows were kept
> on one side of the difference only, and a cell of one row reached `unbiased` with
> its whole squared error, because `var(ddof=1)` is NaN on a singleton and the NaN
> became a zero cost. The old table read 22.699 / 22.269 at 300 s, where 15.7 k of
> january's cells hold one row each — **the old numbers made the narrowest window
> look like the richest one, and it is not**. The corrected curve peaks at 600 s.

**The sharing of the residual between same-airport departures** (`nm_bulk`, winsorised
±1200 s), correlation by take-off gap:

| gap | blend-5 january | blend-5 july_loo | stack-5 january | stack-5 july_loo |
|---|---|---|---|---|
| < 2 min | 0.1461 | 0.1703 | 0.1454 | **0.1681** |
| 2–5 min | 0.1407 | 0.1727 | 0.1398 | 0.1694 |
| 5–15 min | 0.1049 | 0.1247 | 0.1034 | 0.1216 |
| 15–30 min | 0.0608 | 0.0750 | 0.0606 | 0.0724 |
| 30–60 min | 0.0388 | 0.0411 | 0.0400 | 0.0411 |

**stack-5 took 1.1 s of clean board out of blend-5 and left the transient where it
was.** The component is 14.62 s of headroom on the base the candidate corrects, 96.7 %
of what it was on blend-5 (14.568 / 14.997 = 97.1 % at 300 s, 11.771 / 12.229 = 96.3 %
at 1800 s — the same answer at every width). So the +0.002 s is a statement about the second layer, not
about the window: what a residual-on-residual model can reach is small, and the refit
cost it pays to get there is not.

**And it is not that stack-5 already made the arm's move** (`code/overlap.py`). The
`win_only` arm's blend-5 adjustment correlates with stack-5's own move
(`blend-5 − stack-5`) at only **0.12 to 0.21**, regression slope 0.16 to 0.37 — stack-5
made at most a third of it. Carry that **fixed** blend-5 adjustment onto stack-5 without
refitting anything and it is still worth **+0.261 s** of clean board (258.0372 →
257.7765), against **+0.069 s** for refitting the same arm on stack-5's own residual.

| | july_loo | december | january | clean board |
|---|---|---|---|---|
| blend-5 | 286.3274 | 207.0083 | 220.2052 | 259.1339 |
| blend-5 + its own arm | 285.5489 | 206.8232 | 219.7701 | 258.4910 (**+0.643**) |
| stack-5 | 285.2029 | 205.6118 | 219.1303 | 258.0372 |
| stack-5 + **blend-5's** arm, unrefitted | 284.8209 | 206.2131 | 219.0626 | 257.7765 (**+0.261**) |

So the second layer's failure is specifically **refitting on stack-5's residual**. The
signal is still reachable from the window columns and the fixed map still pays; what
does not survive is estimating that map again on a residual whose predictable structure
has already been partly removed, where the same 400 rounds and 31 leaves now fit noise.
That is the mechanism, and it is why the one-layer construction in §4 works: it
estimates one map, once, on the label.

---

## 4. The candidate: the window block inside stack-5's own stacker

`code/stack5w.py`, the desk's construction (a). The identical stacker — exp-026's
`stack5.LGB_BASE`, five seeds, 300 rounds, the rotated blend-5 hybrid NNLS anchor,
`keep-noac` on the 63–339 noac rows — with the window block appended to the seven
member columns and the four member summary columns. Nothing else differs, and the
no-window variant is refitted in the same process and must land on exp-026's published
rotation before either variant is reported. It does, to ±0.0000 on all three folds.

Rotated, clean:

| variant | july_loo | december | january | clean board | predict | vs stack-5 |
|---|---|---|---|---|---|---|
| blend-5 (anchor) | 286.327 | 207.008 | 220.205 | 259.134 | 263.13 | +1.097 |
| stack-5 keep-noac | 285.203 | 205.612 | 219.130 | 258.037 | 262.04 | — |
| stack-5 full | 285.808 | 205.780 | 219.077 | 258.390 | 262.39 | +0.353 |
| **stack-5 + window, keep-noac** | **284.722** | 205.653 | **218.817** | **257.623** | 261.62 | **−0.414** |
| stack-5 + window, full | 285.282 | 205.869 | 218.856 | 257.983 | 261.98 | −0.055 |

Day-clustered 95 % intervals against stack-5 keep-noac (`analysis/review1.json`,
`analysis/restrict_depx_w600.json`; negative is an improvement). **The population is
the one the point estimate uses**: clean, and then clean with the `noac` rows dropped,
which is what the assignment asks for and what the block actually touches.

| fold | clean delta | clean 95 % | clean, no noac | its 95 % |
|---|---|---|---|---|
| july_loo | **−0.481** | **[−0.866, −0.115]** | **−0.558** | **[−0.995, −0.132]** |
| december | +0.042 | [−0.636, +0.763] | +0.043 | [−0.654, +0.776] |
| january | **−0.314** | [−0.711, +0.092] | −0.332 | [−0.752, +0.101] |

> **Corrected after Sol round 1** (Standards 2). The first pass quoted a clean point
> estimate beside a bootstrap over **all** rows, and said so in a footnote. That is two
> populations in one row, and on january it is not a footnote: the two unreached-rollover
> rows are worth **0.16 s of the delta on their own** (all-rows −0.128 against clean
> −0.314). The point gains are unchanged; the january interval moves and the sign of the
> difference between the delta and the gain disappears.

**The three-fold gate.** The assignment (`assignments/window.md`) says a fold that moves
the wrong way is a failure, and on that reading **the 600 s form fails**: december is
−0.042. The desk's clarification of 09:28 UTC replaces that rule with one that calls a
fold worse by less than the rotation's 0.1025 s noise floor, with an interval covering
zero, a MAYBE. December qualifies under it. Stated plainly, so the two are not confused:
**an interval that covers zero does not turn a negative point estimate into a pass** —
what the desk's gate says is that this instrument cannot resolve a fold this small, not
that the fold moved the right way. Under the assignment's own words the 600 s form is a
FAIL on december; under the desk's gate it is a MAYBE. §6 and §7b have the 300 s variant,
whose three point estimates are all positive.

**How much run-to-run spread this construction has.** The `depx` block was measured
twice, once with the fifty columns in `window_features`'s own order and once in
`features.select`'s, which changes nothing but the order LightGBM's `feature_fraction`
samples them in: **+0.460 against +0.481** on july_loo, **−0.050 against −0.042** on
december, +0.317 against +0.314 on january, board +0.402 against +0.414. So 0.02 s on a
fold is what column order alone is worth here, and december's whole deficit is twice
that.

---

## 5. Where the gain sits, and what december's sign is made of

`code/where.py`, read off the saved rotation vectors with nothing refitted.

**The required decomposition**, clean rows, 600 s (`analysis/slices.csv`; the shipping
300 s width and every regime variant are in the same file, and the fold RMSE column is
the rotation's own clean number):

| fold | variant | rmse | ≤3600 | >3600 | no-NM | no-NM, old definition |
|---|---|---|---|---|---|---|
| july_loo | stack-5 | 285.2029 | 248.5415 | 2398.0325 | 1121.4552 | 1121.9978 |
| july_loo | **+window** | **284.7219** | 247.6790 | 2407.2481 | **1121.4234** | 1121.9493 |
| december | stack-5 | 205.6118 | 193.0407 | 1794.0059 | 662.7748 | 663.5964 |
| december | **+window** | 205.6534 | 193.0104 | 1799.0909 | 664.3735 | 665.2053 |
| january | stack-5 | 219.1303 | 197.4837 | 2080.9765 | 952.7929 | 953.3204 |
| january | **+window** | **218.8166** | 196.9245 | 2090.4095 | 952.0867 | 952.6555 |

> **Corrected after Sol round 1** (Standards 4 and Spec 7). Two things. **(1)** The no-NM
> slice is `COMMON.md`'s: every `_flt` column null, read from the raw month. This file
> used `FLIGHT_ID_mvt` null alone, which is an `_mvt` column and misses 4 / 6 / 3 rows
> per fold and 7 ranking rows; the right definition moves the slice by 0.5 to 0.8 s and
> changes no sign. **(2)** The table is now clean and per-fold consistent. The first
> pass mixed populations: january's `rmse` column was the all-row 338.38 while the same
> row's board figure was the clean 219.13, because the two unreached-rollover rows sit
> in one and not the other. `analysis/slices.csv` carries all four required slices for
> every variant §6 and §7b name, which is where the tables in those sections send you
> rather than repeating them in four shapes.

The block improves the bulk (`≤3600`) on all three folds and costs a little on the
0.2–0.3 % of rows above 3600 s, which is the tail lane's population and not the
window's.

**By regime**, as seconds of that fold's RMSE (`analysis/where_depx_by_regime.csv`):

| regime | july_loo | december | january |
|---|---|---|---|
| **nm_bulk** | **+0.383** | **+0.237** | **+0.066** |
| nm_late | +0.094 | **−0.227** | +0.120 |
| nonm | +0.004 | −0.052 | −0.058 |
| noac | 0.000 | 0.000 | 0.000 |

**`nm_bulk` gains on all three folds. December's aggregate loss is `nm_late`.** That is
the regime the NM-tail specialist owns and where blend-5's own weights are a separate
NNLS fit; the block is competing with a specialist there.

**By airport** (`analysis/where_depx_by_airport.csv`), same units:

| airport | july_loo | december | january | folds positive |
|---|---|---|---|---|
| LTFM | +0.244 | +0.261 | +0.079 | 3 |
| LEBL | +0.148 | +0.145 | +0.018 | 3 |
| LIRF | +0.050 | +0.044 | +0.133 | 3 |
| LEMD | +0.072 | +0.037 | +0.028 | 3 |
| EDDF | +0.044 | +0.007 | +0.009 | 3 |
| EDDM | +0.035 | −0.234 | +0.066 | 2 |
| EGLL | +0.020 | −0.222 | +0.115 | 2 |
| LSZH | −0.009 | +0.003 | −0.006 | 1 |
| LFPG | −0.028 | +0.004 | −0.135 | 1 |
| EHAM | −0.097 | −0.086 | −0.178 | 0 |

Five airports gain on all three folds and the two largest gainers are the two the
lineage has always found hardest (LTFM 1291 s mean take-off-minus-off-block, LIRF the
phi outlier). **EHAM loses on all three** — the one airport `docs/shift-2026.md` calls a
different regime in January 2026, with its NM-join rate at its lowest anywhere in the
data. That is a named refinement, not a defect: an EHAM exclusion is a one-line change
to the served regime mask and is left for the desk because it is a fourth cut on the
same three folds.

**Concentration of the change** (`analysis/where_depx_concentration.csv`):

| fold | total SSE drop | share carried by the most-moved 1 % | rows moved > 60 s | RMS move | gain |
|---|---|---|---|---|---|
| july_loo | +5.23e7 | 0.347 | 11,466 | 29.45 | +0.481 |
| december | −2.83e6 | **3.658** | 5,255 | 24.73 | −0.042 |
| january | +1.33e7 | 0.637 | 4,155 | 22.53 | +0.128 |

December's most-moved 1 % of rows carries **366 %** of that fold's whole squared-error
change — the rest of the fold gains and 1,657 rows take it back. A fold sign set by 1 %
of rows is exactly the shape exp-026 §5 warned about on blend-5's noac gain; here it
sits on the fold that is **not** a board month, and on the two that are, the same
statistic is 0.35 and 0.64.

---

## 6. The regime restriction: the variant that passes the gate outright

`code/restrict.py`. Arithmetic on the saved vectors — the same fitted stacker, with the
served regimes varied. Every row outside the set keeps stack-5's own value.

| served on | july_loo | december | january | clean board | gain | three folds |
|---|---|---|---|---|---|---|
| **all but noac (shipped)** | **284.722** | 205.653 | **218.817** | **257.623** | **+0.414** | MAYBE |
| nm_bulk + nm_late | 284.726 | 205.602 | 218.844 | 257.636 | +0.402 | **yes** |
| nm_bulk + nonm | 284.817 | 205.426 | 219.002 | 257.751 | +0.286 | **yes** |
| **nm_bulk only** | 284.820 | **205.374** | 219.029 | 257.763 | +0.274 | **yes** |
| stack-5 | 285.203 | 205.612 | 219.130 | 258.037 | — | — |

Per-fold gains and their day-clustered intervals:

| served on | july_loo | december | january |
|---|---|---|---|
| all but noac | +0.481 [−0.866, −0.115] | −0.042 [−0.636, +0.763] | +0.314 [−0.711, +0.092] |
| nm_bulk + nm_late | +0.477 [−0.822, −0.145] | +0.010 [−0.646, +0.671] | +0.287 [−0.663, +0.101] |
| nm_bulk + nonm | +0.387 [−0.688, −0.102] | +0.186 [−0.598, +0.197] | +0.129 [−0.377, +0.147] |
| nm_bulk only | +0.383 [−0.657, −0.126] | +0.237 [−0.623, +0.103] | +0.102 [−0.358, +0.164] |

*(Clean rows, the same population as the point estimate, after Sol round 1; the same
table with the `noac` rows dropped as well is in `analysis/restrict_depx_w600.json`
under `ci_no_noac_*` and moves every bound by 0.01 to 0.09 s. Only january's bounds
changed from the first pass, for the reason given in §2.)*

**Which is shipped, and why.** `all but noac` is staged. It is the construction as
designed with no regime chosen after the fact, it has the best board, and — the reason
that decides it — the board is july_loo at 55.7 % and january at 44.3 %, and it beats
every restricted variant **on both board months**. December is not a board month and its
deficit is inside the noise floor with an interval covering zero.

**At 300 s the restriction is not needed at all** (`analysis/restrict_depx_w300.csv`).
Every regime set passes the three-fold gate there and `all but noac` is again the best
of them: +0.4067 board against +0.3962 for `bulk+late`, +0.3101 for `bulk+nonm` and
+0.2996 for `bulk` only. So the regime cut is a repair for the 600 s width and not a
property of the construction, which is another reason the 300 s form is the one that
ships (§7b).

**`nm_bulk only` is the conservative candidate** and is one command away
(`serve.py --regimes nm_bulk --name exp-028-window-bulk`). It passes the three-fold gate
with margin on every fold at 0.14 s less board. Its regime set was chosen after seeing
§5's table, which is a selection on the same three folds that carry the scored row; what
this does NOT bound is the selection. **Corrected after Sol round 1** (Spec 2): the
first pass said the choice was bounded because the chosen slice passes the gate on each
fold independently. It is not — the slice was chosen on those same folds, and the same
data cannot both select a slice and bound the selection. What stands on its own is
external: `nm_bulk` is where `docs/STRUCTURAL-STEP.md` §8a–8c measured the transient, in
work that predates this experiment. It is a candidate, not a discovery, and the number
beside it is descriptive.

---

## 7. The screen

### 7a. The block's width and shape, inside the stacker

Every row is the same stacker, the same anchor, the same seeds, the same 300 rounds;
only the block changes. `analysis/stack5w.json` per run, `logs/stack5w*.log`.

| block | columns | july_loo | december | january | clean board |
|---|---|---|---|---|---|
| **`depx`** — 9 aggregates × (own + 4 lags) + hour, weekday, airport, own two deltas | 50 | **+0.481** | −0.042 | **+0.314** | **+0.414** |
| `depx` + `arr` | 85 | +0.517 | −0.109 | +0.289 | +0.427 |
| `depc` — 6 aggregates × (own + ±1) | 18 | +0.427 | −0.188 | +0.065 | +0.287 |
| `dep1` — 9 aggregates × (own + ±1) | 27 | +0.416 | −0.329 | +0.034 | +0.269 |
| `dep0` — 9 aggregates, own bucket only | 9 | +0.239 | −0.180 | +0.166 | +0.210 |
| `depxd` — `depx` plus every aggregate as a deviation from its airport level | 104 | +0.169 | +0.029 | −0.115 | +0.061 |
| `arr` alone | 35 | −0.311 | −0.350 | −0.311 | −0.309 |

*(Four required slices for every row of this table: `analysis/slices.csv` for the
regime variants and `analysis/stack5w_rotation-w*.csv` for the block variants, which
carry `rmse/le3600/gt3600/no_nm/noac/no_noac` per fold as `where.decomp` computes them.)*

**The `arr` block answers both halves of the assignment's item 2, and the second half
is an identity, not a second fit** (Sol round 1, Spec 5). The assignment asks for the
arrivals' unmasked `TAXITIME_SEC_mvt` **and** their `BLOCK_TIME_UTC_mvt`. On an arrival
row those are the same quantity: on-block minus landing **is** taxi-in, and it is exactly
that — `analysis/arrival_block_is_taxi.json`, max |difference| **0.000 s** on 190,448
July, 165,871 December, 153,551 January and all 344,693 ranking arrivals, no row over
half a second, none non-finite. `code/arrivals.py` derives `a_block` and drops it before
aggregating for that reason; a block-time aggregate would be a duplicate column, and the
taxi-in row above is the measurement of both. The `arr` block's other aggregate is
lateness against `SCHED_TIME_UTC_mvt`, which the assignment did not ask for and which is
reported here as part of the same negative.

Two hypotheses were tested and both are refuted by their own row.

**Dilution — "fifty new columns crowd out eleven member columns under
`feature_fraction=0.7`" — is wrong.** Every narrower block is worse on the board *and*
worse on december, which is the fold the hypothesis was invented to explain. The row
context (`own_ta`, `own_al`, hour, weekday, airport) that `depx` carries and `dep`
does not is doing real work inside the stacker, where it did damage as a second layer.

**A cross-month level shift — "`w_ta_mean` at LTFM in July is not the same number as at
LSZH in December, so a level feature fitted on two months does not transfer" — is
wrong too.** The deviation block moves december from −0.042 to +0.029 and january from
+0.314 to −0.115, and the board collapses from +0.414 to +0.061. It trades one fold for
another instead of fixing anything, which is what a variance effect looks like and not
a bias one.

### 7b. Window width — and the variant that passes the gate outright

The `depx` block inside the stacker at three widths, everything else identical
(`logs/stack5w-w{300,600,900}.log`, `analysis/stack5w-w*.json`). Width was on the
assignment's own screen list, so this is a three-point screen and not a search.

| window | july_loo | december | january | clean board | gain | three folds |
|---|---|---|---|---|---|---|
| **300 s** | **+0.484** | **+0.027** | +0.289 | **257.630** | **+0.407** | **yes** |
| 600 s | +0.481 | −0.042 | +0.314 | 257.623 | +0.414 | MAYBE |
| 900 s | +0.387 | −0.135 | +0.326 | 257.676 | +0.361 | no |

**300 s and 600 s are the same number.** The board differs by 0.007 s and every fold by
less than 0.07 s, which is three times the 0.02 s that feature order alone is worth
here (§4). What separates them is the sign of december, which flips from −0.042 to
+0.027 — a swing of 0.069 s, inside the noise floor, on the fold whose own interval is
[−0.64, +0.76]. **That is the strongest evidence in this file that december's deficit at
600 s is noise and not a shape**: the same block on the same rows with a slightly
narrower window puts it the other way for nothing.

It also gives a candidate that **passes the three-fold gate outright with no regime
chosen after the fact**, at 0.13 s more board than §6's `nm_bulk`-only variant. It is
staged as `exp-028-window-w300.parquet` and is the shipping variant.

**The width was chosen on the same three folds the variant is scored on**, and it was
chosen for crossing the sign gate on those folds — not for its board, which is 0.007 s
*worse* than 600 s. So **the 300 s three-fold pass is exploratory, not a second
confirmation of anything**: a variant selected for passing a gate on a set of folds
cannot then be said to have passed that gate on those folds. The honest reading is that
the construction is worth about 0.4 s of clean board at any width in the screen, and
that this instrument does not resolve december's sign at any of them.

> **Corrected after Sol round 1** (Spec 2). The first pass put a number on the width
> selection — "about 0.05 s of board and 0.07 s on december, the spread between the
> widths". That spread is the dispersion of three measurements, not an estimate of
> selection bias, and quoting it as one understates the cost of the choice by an unknown
> amount. There is no bound here; there is a caveat.

The **pre-registered** form, fixed before any stack-5 number was computed, is the 600 s
one: **+0.414 s of board, +0.481 / −0.042 / +0.314**, december a MAYBE under the desk's
gate and a FAIL under the assignment's (§2). It is staged beside the 300 s file so the
board can read the one that was not selected.

> **Corrected after Sol round 1.** This paragraph used to add that 300 s has the larger
> oracle (22.27 s against 15.25 s) and so is where the remaining headroom is. That was
> an artefact of the singleton defect in §3: at 300 s a third of january's cells hold
> one row, and each was counted as free signal. On the corrected estimator the oracle
> is **14.57 s at 300 s against 14.62 s at 600 s** — the two widths have the same
> headroom, and nothing here says a narrower window is the richer direction.

**The leave-one-out window is not run here, and does not need to be.**
`docs/STRUCTURAL-STEP.md` §8d measures it on blend-5's arm at −0.04 s, removing the
departure's own contribution from its own bucket to show the signal is the neighbours.
Including the row in its own aggregate is not a leak — the row's own `d_takeoff_aobt`
and `d_aobt_lobt` are ranking-file columns and are legal — and the `depx` block already
carries `own_ta` and `own_al` as separate features, so the stacker can separate the row
from its neighbours without the aggregate being recomputed. The step worker's 0.04 s is
the size to expect if it were.

900 s is worse on both the board and december, so the trend is monotone in width and
300 s is at the edge of the screen. A 150 s cell is the obvious next point and is not
run here: at 300 s the median window already holds only about 3 movements, and below
that the aggregates become one flight's own number.

### 7c. The second-layer arm's own screen

`code/arm.py screen`, `logs/screen-arm.log`, `analysis/screen_board.csv`. Ten
configurations around the pre-registered point, clean-board gain against stack-5:

| change | july_loo | december | january | clean board | no-noac |
|---|---|---|---|---|---|
| *(the pre-registered point, §2)* | +0.187 | +0.486 | −0.300 | +0.002 | +0.059 |
| window 300 s | −0.372 | +0.316 | −0.069 | −0.255 | +0.131 |
| window 900 s | +0.171 | +0.087 | −0.449 | −0.064 | +0.091 |
| 200 rounds | +0.360 | +0.437 | −0.081 | +0.191 | +0.218 |
| 800 rounds | −0.173 | +0.463 | −0.449 | −0.275 | −0.204 |
| 15 leaves | +0.307 | +0.401 | −0.195 | +0.116 | +0.201 |
| 63 leaves | −0.017 | +0.493 | −0.202 | −0.086 | +0.026 |
| fitted outside noac | +0.908 | +0.250 | −0.246 | **+0.467** | +0.084 |
| fitted and applied outside noac | +0.417 | +0.277 | −0.214 | +0.176 | +0.084 |
| applied outside noac | +0.444 | +0.443 | −0.349 | +0.142 | +0.059 |
| without the head's prediction | +0.252 | +0.111 | −0.142 | +0.102 | +0.033 |

**All ten fail, and january is negative in ten out of ten.** Fewer rounds and fewer
leaves help, which is the overfitting §3 diagnoses; the best cell, the arm fitted
outside the noac rows, reaches +0.467 s of board and still loses 0.246 s on january.
A construction whose winter fold is negative at every setting is not a candidate at
any of them, so no cell of this table is carried forward.

This answers the assignment's screen item on the construction it was written for.
**The stacker's own rounds and leaves are not screened**: they are exp-026's
(`stack5.LGB_BASE`: 300 rounds, 15 leaves, `min_data_in_leaf` 2000, `lambda_l2` 100),
and moving them moves stack-5 as well as stack-5+window, so the comparison would stop
being "the same stacker plus the block". Tuning them is a change to stack-5 and belongs
to the blend lane.

---

## 8. Serving

### 8a. The leakage contract, checked column by column

`SPEC.md` §4 is the contract. Every input is a `ranking.parquet` column: take-off time
`MVT_TIME_UTC_mvt`, NM off-block `AOBT_3_flt`, last-received EOBT `LOBT_flt`, the
departure airport `ADEP_mvt`, and the two null flags `AIRCRAFT_TYPE_mvt` and
`FLIGHT_ID_mvt`. **No departure's `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt` is read,
its own or any other's** — they are blanked for every departure in the file, so a
window aggregate over them would be unservable, not merely leaky. No label enters a
feature; the stacker's target is the label minus the anchor and exists only inside the
fit.

Two facts about the serving side that would each have been a silent break:

- **The raw file has no column called `airport`.** The departure airport is `ADEP_mvt`;
  `airport` exists only inside the derived feature frames the members were fitted on.
  `features.panel_from_movements` reads `ADEP_mvt` and asserts every value is a
  four-letter ICAO code. (exp-026's round-1 review caught the same thing in
  `regime_table.py`, where probing for `airport` left every per-airport split empty.)
- **The airport code must not be built per frame.** `pd.Categorical(sorted(unique))`
  renumbers every airport the moment one is absent. `features.AIRPORTS` is the fixed
  eleven-airport list, with `LTAI` last because the training data has it and no fold or
  ranking month has a departure there.

### 8b. Fitted features against served features

`tests/shape_check.py`, run without any data at all. A fold's canonical panel carries
`t`, `d_takeoff_aobt`, `d_aobt_lobt`, `noac`, `nm_missing` and `airport`, which invert
exactly into the raw movement columns that produced them. So the test rebuilds a real
month as a raw frame, pushes it through the serving adapter, and compares:

```
  july_loo  panel (190713, 12) design (190713, 51)  fit==serve on all 9,726,363 cells
  december  panel (165677, 12) design (165677, 51)  fit==serve on all 8,449,527 cells
  january   panel (153706, 12) design (153706, 51)  fit==serve on all 7,839,006 cells
```

It also checks that 20,000 injected arrival rows change nothing (the window population
is departures on both sides), that dropping LEBL does not renumber the other airports,
and that the `shuffle` control preserves every window column's multiset while moving
98.8 % of rows.

**What that test cannot prove, and the test that does.** A round trip through the panel
is algebra: it shows the two feature paths are one function, but not that the frozen
out-of-fold columns this experiment fits on are the quantities the raw month holds. If
`exp-016`'s `t` were the off-block time rather than the take-off time, or if
`exp-027`'s `d_aobt_lobt` had the opposite sign, every aggregate would be built on one
definition at fit time and another at serve time and `shape_check.py` would still pass.
`tests/panel_vs_raw.py` closes it with the data: each fold's month is loaded from
`data/raw`, masked exactly as the ranking file is (`prc.data.mask_like_ranking`), pushed
through the **serving** adapter, and compared against the fitted panel row by row.

```
july_loo  2025-07  oof 190713 of 190713 month departures  t=0 d_takeoff_aobt=0 d_aobt_lobt=0
                   airport=0 regime=0 nm_missing=0 noac=0   design 0/9,726,363   PASS
december  2025-12  oof 165677 of 165677 month departures  … design 0/8,449,527   PASS
january   2025-01  oof 153706 of 153706 month departures  … design 0/7,839,006   PASS
```

Zero differing values on every column and every one of 26.0 M design cells, and the
out-of-fold files cover **100 %** of each month's departures. The frozen panel is the
raw month.

### 8c. The ranking file, read

`data/raw/ranking.parquet`, 689,534 rows: 344,841 departures and 344,693 arrivals.
The departure panel is 344,841 rows at the expected ten airports, split 192,122 July
2026 and 152,719 January 2026, with the regime census **nm_bulk 323,468, nm_late
16,090, nonm 4,895, noac 388** — the same 16,090 and 388 exp-026 §5 predicted from
exp-014's census. Feature distributions against the three fold months:

| | fold median | ranking median | fold p90 | ranking p90 | fold NaN | ranking NaN |
|---|---|---|---|---|---|---|
| `w_n` | 6.000 | 6.000 | 10.000 | 10.000 | 0.000 | 0.000 |
| `w_ta_mean` | 962.75 | 984.00 | 1440.00 | 1471.75 | 0.000 | 0.001 |
| `w_ta_sd` | 185.96 | 191.63 | 395.24 | 412.51 | 0.022 | 0.023 |
| `w_ta_p90` | 1188.40 | 1200.80 | 1671.80 | 1736.00 | 0.000 | 0.001 |
| `w_al_mean` | 339.82 | 330.00 | 996.00 | 960.00 | 0.000 | 0.001 |
| `w_al_p90` | 840.00 | 828.00 | 1920.00 | 1872.00 | 0.000 | 0.001 |
| `w_late` | 0.000 | 0.000 | 0.182 | 0.167 | 0.000 | 0.000 |
| `own_ta` | 961.00 | 978.00 | 1471.00 | 1498.00 | 0.014 | 0.015 |

The served months are a little busier than the fitted ones and otherwise the same
distribution. No feature has a missing-rate the folds did not have.

### 8d. The three staged candidates

`serve.py`, about 40 s each. Records in `analysis/served-<name>.json`.

| | **exp-028-window-w300** (shipping) | exp-028-window | exp-028-window-bulk |
|---|---|---|---|
| window | **300 s** | 600 s (pre-registered) | 600 s |
| served on | every non-noac row | every non-noac row | `nm_bulk` only |
| clean board | **257.6305** | 257.6230 | 257.7634 |
| gain vs stack-5 | **+0.407** | +0.414 | +0.274 |
| folds | **+0.484 / +0.027 / +0.289** | +0.481 / −0.042 / +0.314 | +0.383 / +0.237 / +0.102 |
| three-fold gate | **pass**, but the width was selected on these folds (§7b) | MAYBE on december under the desk's gate, FAIL under the assignment's | pass, on a regime set chosen on these folds (§6) |
| SHA-256 | `dfa4d78017f7b5b4…` | `d55b1de9ce8fe757…` | `52c9f4ac6468d06d…` |
| rows served by the block | 344,453 | 344,453 | 323,468 |
| rows taken from stack-5 | 388 | 388 | 21,373 |
| RMS move vs stack-5 | 24.46 s | 26.47 s | 21.48 s |

Full SHA-256:

```
dfa4d78017f7b5b41107823a6bb05204644b3f8ef0ae3d82d277ac416d95cfe2  exp-028-window-w300.parquet
d55b1de9ce8fe7577c8c566c6efc57301d18484242aaa06ce808a111d38dabb0  exp-028-window.parquet
52c9f4ac6468d06d0c748546cbed8b72a4c8fb7dcd09dd88f5135f8e6e31a7ff  exp-028-window-bulk.parquet
```

Shared by all three: 344,841 rows, min 60.0, 0 below the floor, 0 non-finite, the base
is `submissions/candidates/exp-026-stack-5.parquet` (SHA-256 `94f18e47…`). For
`exp-028-window`, median 954.14, p99 2574.10, max 115,306.6, guard touched 780 rows on
the block side; the move against stack-5 runs −76.1 s to +91.2 s between the 1st and
99th percentile with a median of −1.0 s.

**The maximum, 115,306.6 s, is one of the 388 noac rows and is stack-5's own value,
unchanged.** The tail extreme these files carry is inherited, not new.

### 8e. Verification, the way exp-026 verifies stack-5

`code/verify.py`. `finalise.py` refuses to issue a blend candidate whose values differ
from `prc.blend.build_candidate`'s by more than 0; this candidate has no second
implementation in the library, so the served vector is **rebuilt from the record the
candidate itself carries**, by straight-line code that does not call `serve.py`, and
compared.

**What that rebuild does and does not prove** (corrected after Sol round 1, Standards 3;
the first pass said the two share "only the feature builders", which is wrong).
`verify.py` calls the same `stack5w.block_for`, `stack5w.ranking_blocks` and
`stack5w.arm` as `serve.py`, and the same exp-026 frame, anchor and member functions.
A `max abs difference` of 0 therefore proves **deterministic replay and correct file
assembly** — the record describes the file, the ordering and the substitution are right,
nothing drifted between staging and checking — and it **cannot** detect a defect that
lives inside a shared function: a wrong block, a wrong alignment, a wrong anchor, a
wrong parameter. Those are covered by other evidence and not by this row: the panel is
checked against the real month files by `tests/panel_vs_raw.py` (§8b, zero differences
over 26.0 M design cells), the alignment by the id-order check `serve.py` now makes on
the saved rotation, and the anchor by the reproduction gate that aborts unless the
no-window rotation lands on exp-026's published stack-5.

| check | exp-028-window-w300 | exp-028-window | exp-028-window-bulk |
|---|---|---|---|
| id set and order equal `submitting.parquet` | yes | yes | yes |
| dtype | float64 | float64 | float64 |
| all finite | yes | yes | yes |
| values below `FLOOR` | 0 | 0 | 0 |
| SHA-256 equals the record's | yes | yes | yes |
| **rebuild from the record, max abs difference** | **0.0** | **0.0** | **0.0** |
| rows outside the served regimes, max abs difference against stack-5 | **0.0** on 388 | **0.0** on 388 | **0.0** on 21,373 |
| verdict | **PASS** | **PASS** | **PASS** |

**The whole path is deterministic end to end.** All three files were staged twice, in
separate processes, and the second staging reproduced all three SHA-256 values exactly;
the rotation was also rerun at all three widths after the artifacts were renamed and
reproduced every fold number to the printed digit (+0.481 / −0.042 / +0.314 at 600 s,
+0.484 / +0.027 / +0.289 at 300 s, +0.387 / −0.135 / +0.326 at 900 s).

### 8f. What the board says these are worth

**stack-5 was uploaded as v3 at 09:57 UTC and scored 264.3861, rank 1 of 63**
(v1 265.76, v2 265.96; rivals at 267.0). Its noac-discounted prediction of 264.59 was
right within 0.2 s and its face-value one of 262.04 missed by 2.35 — the same pattern
v2 showed, and the reason `docs/LEDGER.md` BOARD READING 3 now says to quote a
candidate as clean-with-noac-at-the-anchor plus 4.28.

**No noac discount applies to this experiment's gain.** All three candidates serve the
388 noac rows from `exp-026-stack-5.parquet` bit for bit (§8e), so their gain over
stack-5 is zero on noac by construction and sits in `nm_bulk` (§5). The discount and the
4.28 are therefore common to stack-5 and to these files, and cancel: **the board
prediction is stack-5's own board reading minus the measured clean gain.**

| candidate | clean board | gain vs stack-5 | predicted board |
|---|---|---|---|
| stack-5 (v3, **measured**) | 258.0372 | — | **264.3861** |
| **exp-028-window-w300** | 257.6305 | +0.4067 | **263.98** |
| exp-028-window (600 s) | 257.6230 | +0.4142 | 263.97 |
| exp-028-window-bulk | 257.7634 | +0.2737 | 264.11 |

*(The `predict` column in §4's table is exp-026's older clean + 4.0 convention and is
superseded by this one. The 0.28 s between the two conventions is why it is stated.)*

**This is below a 0.5 s upload floor on its own**, and the desk's reason for taking it
anyway is that it is the cheapest test of whether the in-stacker window transfers to the
board, which is what decides K2's value. A board reading on this candidate turns §7b's
+0.407 s into a measured transfer coefficient for the whole window family.

**Uploaded: the 600 s form, as v6 at 10:32:42 UTC** (`youthful-giraffe_v6.parquet` =
`exp-028-window.parquet`, sha `d55b1de9…`; `docs/SUBMISSIONS.md` row 6, ledger row
*submission v6*). The desk took the **pre-registered** width over the shipping 300 s one
for the reason §7b gives against itself: the two differ by 0.007 s of board, and the
600 s estimate carries no width-selection cost while the 300 s one was chosen for
crossing the sign gate on the folds it is scored on. The prediction to read it against is
**263.97**, and what it measures is the transfer of an in-stacker window gain that is
entirely `nm_bulk` — which is what prices K2. The 300 s and `nm_bulk`-only files stay
staged and verified.

*(This line supersedes the slot the desk named at 10:07 UTC — v4 the `noac` instrument,
v5 the 300 s form. v4 went up as planned; the 300 s form was not uploaded.)*

### 8g. The commands

Everything the candidates need already exists on this machine.

```
.venv/bin/python experiments/exp-028-window/serve.py --win 300 --name exp-028-window-w300
.venv/bin/python experiments/exp-028-window/serve.py                       # the 600 s form
.venv/bin/python experiments/exp-028-window/serve.py --regimes nm_bulk --name exp-028-window-bulk
.venv/bin/python experiments/exp-028-window/code/verify.py                 # all three
```

`serve.py` reads `analysis/choice.json` for the block, the width and the served regimes,
so it cannot serve a configuration no rotation measured, and it raises rather than
writes if any row outside the served regimes differs from the base by one bit, if any
prediction is non-finite, or if the file changes between the SHA-256 check and the move
into `submissions/`.

---

## 9. Reproducing every number here

From the repo root, 2 threads. `data/raw` is needed only by §7a's `arr` rows, §8c and
§8d; everything else runs from committed bytes.

```
.venv/bin/python experiments/exp-028-window/code/panel.py            # section 1, the gate
.venv/bin/python experiments/exp-028-window/tests/shape_check.py     # section 8b
PRC_THREADS=2 .venv/bin/python experiments/exp-028-window/code/arm.py candidate      # section 2
PRC_THREADS=2 .venv/bin/python experiments/exp-028-window/code/diagnose.py           # sections 2-3
PRC_THREADS=2 PRC_VARIANTS=depx .venv/bin/python experiments/exp-028-window/code/stack5w.py   # section 4
.venv/bin/python experiments/exp-028-window/code/where.py depx       # section 5
.venv/bin/python experiments/exp-028-window/code/restrict.py depx    # section 6
bash experiments/exp-028-window/code/screens.sh                      # section 7
.venv/bin/python experiments/exp-028-window/serve.py                 # section 8d
```

Out-of-fold predictions in COMMON.md's format are
`oof/exp-028-stack5w-{w300,w600}-{fold}.parquet` — `MVT_ID_mvt`, `fold`, `y_true`,
`y_pred` (the candidate), `y_pred_depx` (the same vector, named by its block) and
`y_pred_stack5` (the no-window rotation this is measured against) — and, for §2's
second-layer construction, `oof/exp-028-window-{fold}.parquet`. Nine files, 55 MB.

**Corrected after Sol round 1** (Spec 8): the description above used to name the files
without their width tag and to list a `y_pred_anchor` column. That column is
deliberately not written — blend-5's rotated vector is already committed in exp-026's
and exp-027's out-of-fold files and this experiment reads it from there, which
`code/panel.py` checks to 1e-9 on load. The w900 vectors and the block-screen variants
were deleted after their numbers were recorded, to keep this directory under 60 MB;
`code/regen.sh` rebuilds any of them, and the rotation reproduces exactly (§8e).

---

## 10. What would revive the paths this closes, and what it hands on

Nothing here is closed; two paths are parked with their revival conditions.

- **The second-layer arm (§2).** Revived by a base that is not itself a fitted residual
  model — it worked on blend-5, which is a linear blend — or by a way to fit the second
  layer that does not pay the refit cost, for instance freezing everything but a
  window-only monotone correction. The measurement that would revive it: `row` control
  above −0.5 s on stack-5.
- **The arrivals block (§7a).** Alone it loses 0.309 s and loses on all three folds,
  which says arrival taxi-in over the same 600 s window is not a measurement of the
  departure transient. The obvious repair is not this experiment's to make: an arrival's
  taxi-in describes the inbound surface, and the departure's queue is the outbound one.
  Revived by keying the arrivals on the **runway and stand** the ranking file also
  gives (`RUNWAY_mvt`, `STAND_mvt`, both 100 % populated on the 344,693 ranking
  arrivals) rather than on the airport alone.
- **EHAM (§5)** loses on all three folds and is the one airport January 2026 puts
  outside its own 2025 regime. A served regime mask that excludes it is one line; it is
  a fourth cut on the same three folds, so it is the desk's call and not taken here.
- **K2 proper is the `step` worker's lane**: the window aggregates inside a member's
  own feature frame, refit (`F1-fresh`, about 19 minutes per fold on Kaggle). This
  experiment moves the stacker, not a member. The number it should be compared against
  is +0.414 s of clean board, not K1's +0.669 on blend-5.

## 11. Reviewer

`reviewer: gpt-5.6-sol-xhigh-2026-09-07`, fresh context, read-only, one round.
Full report: `review/sol-round1.log` (the findings are the last two sections of it).

**Verdict.** *"Leakage result: pass for the shipping block. Departure features use only
ranking-visible take-off, AOBT, LOBT, airport, aircraft/flight null flags, and row
counts. Airport levels use the same legal fields. Arrival code filters to `ARR` before
it reads block or taxi-in. No departure block time, taxi time, or label enters a
feature."* Seventeen findings, **none of which moves a staged candidate's number**:
nine on standards, of which four change a measurement, and eight on spec, of which none
does. Every one is accepted; none is disputed.

| # | finding | effect | what was done |
|---|---|---|---|
| S1 | the "clean, unbiased" oracle kept the excluded january rows and counted a singleton cell's whole squared error as signal | numbers | recomputed and `code/diagnose.py` fixed; §3 table replaced (15.25 → **14.62** at 600 s), superseded values kept as `analysis/diagnose_oracle-superseded.json`. The 300 s figure falls 22.27 → 14.57, which **withdrew** a claim in §7b |
| S2 | clean point estimates quoted beside all-row intervals; no no-noac intervals | numbers | recomputed on both populations; `code/stack5w.py` and `code/restrict.py` fixed and the restriction rerun; §2 and §6 tables replaced. On january the two excluded rows were worth 0.16 s of the delta |
| S3 | `verify.py` shares `block_for`, `ranking_blocks` and `arm` with `serve.py`, so the rebuild is not independent | claim | §8e rewritten to say what the 0.0 proves (replay and assembly) and what it cannot (a defect inside a shared function), and what covers that instead |
| S4 | no-NM used `FLIGHT_ID_mvt` alone, not `COMMON.md`'s "every `_flt` column null" | numbers | `code/slices.py` written; §5's table replaced with the clean, correct-definition one, both definitions shown |
| S5 | the reproduction gate was recorded and never enforced; `serve.py` checked anchor length, not alignment | claim | `stack5w.py` now aborts above 1e-3; `serve.py` now requires the saved rotation's ids to equal the fold frames' ids in order. Current vectors already matched |
| S6 | `shape_check.py` is an algebraic round trip and cannot catch a wrong source column | neither | agreed; `tests/panel_vs_raw.py` (written mid-review) is what closes it, and §0 item 10 now leads with that test |
| S7 | 257.7376 quoted for the 600 s bulk candidate; that is the 300 s bulk number | number | corrected to **257.7634** everywhere |
| S8 | `serve.py` could write a configuration nothing had measured, with empty metadata | claim | it now refuses unless `--unmeasured` is passed, which stamps the record; the header's "`nm_bulk` only" description corrected |
| S9 | the deviation block has no serving path and would use one baseline across two months | neither | `ranking_blocks` now refuses it with that reason; `SPEC.md` §6 corrected. The form was measured at +0.061 and not carried |
| P1 | the 600 s form fails the assignment's literal gate; an interval covering zero does not make a negative point estimate a pass | claim | §2 now states both readings and says which is the desk's |
| P2 | both post-selection caveats too weak: the width spread is not a bound, and the selected slice cannot bound its own selection | claim | §7b and §6 rewritten; the 300 s three-fold pass is called exploratory |
| P3 | "the channel cancels out" oversteps what a common instrument gives | claim | withdrawn in `SPEC.md` §3 and `code/arm.py`; the delta is now called instrument-relative |
| P4 | no ledger row and no verdict recorded | neither | this section, and six rows in `docs/LEDGER.md` |
| P5 | the arrivals' `BLOCK_TIME_UTC_mvt` aggregate was asked for and not measured | claim | closed by measurement: on arrival rows it **is** taxi-in, exactly, on 510 k fold and 345 k ranking rows (`analysis/arrival_block_is_taxi.json`). §7a |
| P6 | the required clean and no-noac intervals were missing | neither | same as S2 |
| P7 | the four required slices are missing from three tables | neither | `analysis/slices.csv` gives them for every variant those tables name, with pointers from each |
| P8 | the out-of-fold description is stale (no width tag, names a column that is not written) | claim | §9 corrected |

**What the review did not change.** The construction, the three staged files and their
SHA-256 values, every fold RMSE and every board figure in this file. The shipping
candidate was re-served after the code fixes and reproduces bit for bit: same
`n_served_by_block`, same guard count, same median, same maximum to the last digit
(`logs/serve-w300-dryrun.log`).
