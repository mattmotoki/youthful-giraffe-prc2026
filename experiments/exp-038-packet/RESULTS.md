# exp-038-packet — RESULTS

Worker `packet`, tier hard, 2026-09-08. Assignment `assignments/packet.md`; the reading taken
and the readings rejected are in `SPEC.md`. Laptop, 4 threads, CPU only, no GPU, no member
fit, no upload. Every number here is written by a script in `code/`. `code/checks.py` re-runs
the packet's eighteen structural identities (hashes, the join, the masks, the cancellation, the
algebra and the two record endpoints) from the committed packet alone — it is a structural
audit, not a re-run of every number in this file. Sol round 1 is answered in §5.

## 0. The answer, in five lines

**Nothing in the record beats v10 at full amplitude.** Of the **37 registered candidates**
since stack-5 (38 vectors including v10 itself), **36 lose** on the contract's primary endpoint
and one (`exp-037 chain_shr_reg`) is bit-identical to v10 because its out-of-fold shrinkage came
out at zero on every fold. Four exp-032 heads (`k3-only`, `k3g-only`, `k3-add`, `k3g-add`) are
staged in that experiment's tables but their predictor vectors are not persisted anywhere, so
they are named as absent rather than counted (Sol round 1, blocking 12). The endpoint is the
contract's primary one: paired **all-row** January/July MSE at the ranking month shares. The
closest losers are `exp-037 chain_raw_clf` (−0.0165 s), `refit-bulk` (−0.0294 s) and
`k2-w300` (−0.0616 s); the rest lose by 0.06 to 1.52 s.

**But the residual alignment is positive nearly everywhere, and a bounded amplitude fitted
away from the rows it scores turns the window lineage positive.** `A = mean(r·d) > 0` on both
board months for every vector of that lineage, with fitted scales of 0.10–0.45. Scoring at a
LEAVE-ONE-FOLD-OUT amplitude, `stack-5` reads **+0.1017 s [+0.010, +0.192]** of all-row board
and `k2-w300` **+0.0788 s [+0.036, +0.130]**; a second cross-fit on the same data (cross-day
inside each month, not an independent sample) gives +0.0922 and +0.0783. Once the amplitude fit
is inside the resample, `stack-5`'s lower bound falls onto zero ([+0.0015, +0.1862]) and
`k2-w300`'s does not ([+0.0120, +0.1421]) — §2. The direction is not new information — it is *shrinkage*:
`v10 + s(p_old − v10)` serves only `1 − s` of v10's own increment over its predecessor.

**The shrinkage preserves every mask v10 holds, at zero direct cost on those rows** (which
says nothing about opportunity cost elsewhere). On the fold side `d = 0` on
all 60,407 EHAM rows and all 491 `noac` rows for every candidate in the lineage (`checks.py`,
tolerance 0). On the served side v10 and `submissions/candidates/exp-026-stack-5.parquet`
(SHA-256 `94f18e47…`) are **bit-identical on the 388 ranking `noac` rows and the 38,182 EHAM
rows** and differ on 306,271 of 344,841, max |d| 803.48 s — so a blend of the two moves only
the rows exp-036 already moves and keeps score-feedback rule's bytes exactly.

**Q07 stage 1: the finalist's number moves little on the two partial instruments exp-031 can
build, and those instruments do not identify a cause.** Recomputed on the nested instrument the
`stack-5` held-out gain reads +0.1023 (member channel closed) and +0.0771 (anchor channel
closed as well) against +0.1017 published; `K2`'s gain *grows* under nesting (+0.0344 →
+0.0618). But `published → nested` carries the machine change, the nesting and the loss of one
training month together, and exp-031's `kaggle` column — the one that separates the machine
term — is not in the committed OOF, so **this stage cannot say the channels are not producing
the gain; it can only say the gain changes little on two confounded instruments**. `k2-w300`, a
Q01 finalist, has no cached nested vector and is **not audited at all**. The channel nothing
here closes is the mask selection, which an upload has already priced.

**Q02: the substitution loses at every depth, and `V3-allin`'s value turns out to be an
interaction between the two stages.** The contracted *fully refitted* substitution needed a
blend rule that can carry a partial general member; one was written (fitted twice per regime
block, each row served by the fit that matches its own availability) and it reproduces v10 at
**0.0000** when there is no partial member. On it the replacement costs **−0.4985 s** of clean
board — worse than the −0.2272 of the head-only version this file first reported. The 2 × 2
factorial says why nothing simpler explains it: taking `V3-allin` out of the head alone costs
−0.0068, out of the anchor alone **+0.0105**, and out of **both** −0.2084, an interaction of
−0.2121. Either stage alone carries what the member contributes. The addition remains the one
positive: across five **matched** seed draws it is **+0.0167 s on the primary all-row endpoint,
5 of 5 draws, exact sign-flip p = 0.031** (clean +0.0067, 4 of 5, p = 0.125) — stable, one sixth
of the candidate gate, and a **MAYBE**. Putting bulk64 in the anchor as an eighth member is
**not** the way to grow it: that arm reads −0.0575 clean. See §4.

## 1. The instrument and the contract

The baseline is **v10**, `submissions/uploaded/youthful-giraffe_v10.parquet`, SHA-256
`ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78`, board 263.0464
(BOARD READING 9). Its fold-side form is exp-036 arm 2 and is rebuilt here by that arm's own
arithmetic — `np.where(airport in keep, refit-bulk, v7)` over two committed rotations, with
`keep` read from `experiments/exp-036-mask/analysis/mask.json` (SHA-256 `c89caa9f1be45d9d…`,
eight airports: EDDM, EGLL, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) — and gated on exp-036's
published clean triple **284.1572 / 204.9218 / 218.1506, board 257.0248**. `code/q02.py`
additionally rebuilds it from the member frames upward and lands on the same three numbers to
**0.0000** on every fold, so the whole v10 recipe is a function of committed bytes.

Every candidate is joined one-to-one on `MVT_ID_mvt` in the reference panel's own order and
`code/vectors.py` refuses any vector that is not in that order. Each is gated against the
number its own experiment published before it enters the table; four of the five
`no-EHAM-mask` variants have no published clean triple of their own (K2's has one) and are
gated by the identity that applying v7's EHAM mask to them reproduces their published masked
sibling **row by row** at max abs difference 0. exp-037's sixteen
arms are gated on `analysis/repair_arms.csv`'s own `rmse_arm` over the fold's `noac` rows.
**One family is skipped rather than rebuilt** (`SPEC.md` §1): exp-035's seed arm, which has no
head-level fold vector.

Endpoints, all on the same vectors:

| endpoint | definition |
|---|---|
| **primary** | paired **all-row** MSE on july_loo and january at `SHARES = {july 0.55713, january 0.44287}`, reported as board seconds `sqrt(Σ share·MSE)` |
| clean | the same with each fold's unreached-rollover rows removed — the convention every LEDGER row since stack-5 is quoted on |
| december | reported alone; it is scored and gated but carries no board share |

The all-row and clean endpoints differ on **two rows**, both `nonm` January rows with
`y = 84,240 s` and `y = 58,206 s` (`prc.data.UNREACHED_ROLLOVER`). They are the reason january
reads 337.83 all-row against 218.15 clean, and they are why the clean gain of a candidate that
moves them is not the all-row gain: for `K2` the two endpoints read −0.1362 and −0.1404.
Rows a candidate does not move contribute **exactly zero** to the paired SSE — checked, not
assumed (`checks.py`, max |paired SSE| on an unmoved row = 0).

Uncertainty is a block bootstrap over **calendar days shared across airports**, resampled
inside each fold and recombined through the same board formula on every draw, so the interval
is on the endpoint. The multi-day sensitivity check repeats it on fixed 3-day and 7-day
blocks; the 3-day column is in the table and moves no sign.

**Composition risk, vectors fixed.** Reweighting each board fold's rows to its ranking month's
airport × regime mix (`experiments/exp-026-rebuild/candidates/ranking-gates.parquet`,
344,841 rows) moves the endpoint by up to **0.3620 s** — the four exp-037 `clip` arms, which
move twelve rows and whose full-amplitude number is −1.5225 either way — and by up to
**0.0241 s** among everything else (`k2-w300 no-EHAM-mask`). **It does change the ordering**:
`exp-037 chain_shr_full` and `refit-bulk no-EHAM-mask` both move up. An earlier version of this
line said "at most 0.02 s and changes no ordering"; both halves were wrong and Sol round 1
(blocking 4) is right (`analysis/q01-composition.csv`, column `gain_board_rankmix`). The
reweighting does **not** change which candidates are the finalists, and the two finalists' own
moves at full amplitude are 0.0016 (`stack-5`) and 0.0169 (`k2-w300`). **Unsupported target mass is 0.0000 on both months**: all 31
airport × regime cells of 2026-07 and all 31 of 2026-01 have fold rows behind them. One
july_loo row sits in a cell 2026-07 never visits.

`gain_board_rankmix` prices the **full-amplitude** candidate. The bounded blend is priced under
the same reweighting in `analysis/q01-composition-blend.csv` and in §2a.

## 2. Q01 — the table

Sorted by the held-out amplitude gain. `full amplitude` is `2A − B` expressed in board
seconds; `held-out (LOO)` scores each fold at an amplitude fitted on the two months it is not,
bounded to `[0, 1]`; `(CV)` scores it at an amplitude fitted on the alternating 3-day blocks
the row is not in, inside its own month. Positive is better than v10 throughout. The last two
columns are the held-out blend on the clean population and on december alone, so a row whose
december amplitude came out at zero reads +0.0000 there. **They are not the same blend**: the
clean column refits its own amplitude on the clean rows, which Sol round 1 (blocking 12,
non-blocking) is right to call out as different vectors under one heading. Holding the all-row
amplitude fixed and only changing the population scored, the clean gains are **+0.1100**
(`stack-5`), **+0.0956** (`k2-w300`) and **+0.0468** (`K2`) instead of +0.1122, +0.0959 and
+0.0481 — a difference of at most 0.0022 s, which changes nothing but is stated because the
heading did not say it.

| candidate | rows moved (jul) | full amplitude | 95 % | 3-day 95 % | fitted amp jul/jan | **held-out (LOO)** | 95 % | (CV) | clean, at the LOO amp | december, at the LOO amp |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| stack-5 | 167,885 | -0.8047 | [-1.276, -0.466] | [-1.311, -0.413] | 0.304 / 0.270 | **+0.1017** | [+0.010, +0.192] | +0.0922 | +0.1122 | +0.1985 |
| k2-w300 | 167,880 | -0.0616 | [-0.214, +0.073] | [-0.213, +0.080] | 0.303 / 0.347 | **+0.0788** | [+0.036, +0.130] | +0.0783 | +0.0959 | -0.0210 |
| k2-w300 no-EHAM-mask | 190,369 | -0.1630 | [-0.329, -0.021] | [-0.324, -0.022] | 0.227 / 0.304 | **+0.0586** | [+0.024, +0.099] | +0.0567 | +0.0716 | -0.0112 |
| w600w300w1800 (staged-extra) | 190,369 | -0.3433 | [-0.591, -0.156] | [-0.605, -0.133] | 0.137 / 0.194 | **+0.0577** | [+0.029, +0.088] | +0.0558 | +0.0630 | -0.0918 |
| bulk-add-nmbulk-out | 167,880 | -0.1254 | [-0.286, +0.024] | [-0.285, +0.029] | 0.236 / 0.232 | **+0.0392** | [+0.005, +0.079] | +0.0321 | +0.0538 | -0.0242 |
| bulk-add-hold | 167,880 | -0.1320 | [-0.298, +0.010] | [-0.299, +0.019] | 0.296 / 0.232 | **+0.0391** | [-0.001, +0.084] | +0.0267 | +0.0486 | -0.0034 |
| bulkonly-nmbulk-out | 167,880 | -0.1373 | [-0.296, +0.009] | [-0.295, +0.014] | 0.245 / 0.234 | **+0.0364** | [+0.002, +0.077] | +0.0281 | +0.0499 | -0.0121 |
| bulk-add-bulkonly-hold | 167,880 | -0.1388 | [-0.300, +0.007] | [-0.302, +0.012] | 0.262 / 0.223 | **+0.0361** | [+0.000, +0.078] | +0.0270 | +0.0473 | -0.0109 |
| K2 (v9) | 167,880 | -0.1362 | [-0.295, +0.012] | [-0.292, +0.016] | 0.225 / 0.231 | **+0.0344** | [+0.002, +0.073] | +0.0278 | +0.0481 | -0.0172 |
| k2-mask | 146,546 | -0.1321 | [-0.285, +0.016] | [-0.281, +0.019] | 0.215 / 0.218 | **+0.0337** | [+0.003, +0.070] | +0.0277 | +0.0470 | -0.0259 |
| k3-hold | 167,880 | -0.1894 | [-0.367, -0.049] | [-0.377, -0.026] | 0.173 / 0.132 | **+0.0255** | [+0.001, +0.052] | +0.0178 | +0.0309 | -0.0454 |
| k3g-hold | 167,880 | -0.1852 | [-0.358, -0.048] | [-0.375, -0.024] | 0.169 / 0.127 | **+0.0255** | [+0.002, +0.051] | +0.0191 | +0.0302 | -0.0511 |
| K2 no-EHAM-mask | 190,369 | -0.2258 | [-0.394, -0.079] | [-0.389, -0.072] | 0.156 / 0.197 | **+0.0250** | [+0.000, +0.055] | +0.0114 | +0.0364 | -0.0138 |
| w600w300 (staged-extra) | 190,369 | -0.4273 | [-0.691, -0.246] | [-0.663, -0.234] | 0.102 / 0.134 | **+0.0234** | [+0.003, +0.044] | +0.0024 | +0.0257 | -0.0319 |
| k3-hold no-EHAM-mask | 190,369 | -0.2833 | [-0.472, -0.135] | [-0.483, -0.116] | 0.123 / 0.121 | **+0.0207** | [+0.002, +0.041] | +0.0043 | +0.0252 | -0.0360 |
| k3g-hold no-EHAM-mask | 190,369 | -0.2796 | [-0.466, -0.133] | [-0.480, -0.114] | 0.118 / 0.116 | **+0.0206** | [+0.003, +0.040] | +0.0056 | +0.0243 | -0.0411 |
| refit-bulk no-EHAM-mask | 43,823 | -0.0673 | [-0.129, -0.020] | [-0.126, -0.024] | 0.413 / 0.185 | **+0.0115** | [-0.007, +0.031] | +0.0253 | +0.0138 | +0.0079 |
| v7 | 146,546 | -0.3605 | [-0.588, -0.192] | [-0.584, -0.170] | 0.104 / 0.033 | **+0.0075** | [-0.009, +0.024] | +0.0024 | +0.0090 | -0.0355 |
| w600arrk (staged-extra) | 190,369 | -0.5075 | [-0.777, -0.314] | [-0.758, -0.306] | 0.063 / 0.029 | **+0.0055** | [-0.005, +0.016] | -0.0071 | +0.0059 | -0.0240 |
| v6 | 169,035 | -0.4962 | [-0.756, -0.311] | [-0.734, -0.295] | 0.035 / 0.023 | **+0.0033** | [-0.003, +0.009] | -0.0125 | +0.0039 | -0.0194 |
| exp-037 chain_shr_clip | 12 | -1.5225 | [-3.526, -0.019] | [-4.226, +0.233] | 0.000 / 0.000 | **+0.0000** | [+0.000, +0.000] | -0.0957 | +0.0000 | +0.0000 |
| v10 (exp-036 refitbulk-mask) | 0 | +0.0000 | [+0.000, +0.000] | [+0.000, +0.000] | 0.000 / 0.000 | **+0.0000** | [+0.000, +0.000] | +0.0000 | +0.0000 | +0.0000 |
| exp-037 chain_shr_reg | 0 | +0.0000 | [+0.000, +0.000] | [+0.000, +0.000] | 0.000 / 0.000 | **+0.0000** | [+0.000, +0.000] | +0.0000 | +0.0000 | +0.0000 |
| exp-037 chain_raw_reg | 312 | -0.2103 | [-0.708, +0.219] | [-0.599, +0.152] | 0.000 / 0.000 | **+0.0000** | [+0.000, +0.000] | -0.1485 | +0.0000 | +0.0000 |
| exp-037 rot_shr_clip | 12 | -1.5225 | [-3.526, -0.019] | [-4.226, +0.233] | 0.000 / 0.000 | **+0.0000** | [+0.000, +0.000] | -0.0957 | +0.0000 | +0.0000 |
| refit-bulk | 21,334 | -0.0294 | [-0.065, -0.003] | [-0.069, -0.001] | 0.245 / 0.389 | **-0.0022** | [-0.010, +0.004] | -0.0007 | -0.0027 | +0.0032 |
| exp-037 chain_raw_full | 327 | -0.3447 | [-0.980, +0.228] | [-0.720, +0.117] | 0.000 / 0.291 | **-0.0371** | [-0.126, +0.022] | -0.0329 | -0.0446 | +0.0142 |
| exp-037 chain_shr_full | 0 | -0.0371 | [-0.126, +0.022] | [-0.101, +0.027] | 0.000 / 1.000 | **-0.0371** | [-0.126, +0.022] | +0.0000 | -0.0446 | +0.0000 |
| exp-037 chain_raw_clf | 327 | -0.0165 | [-0.459, +0.439] | [-0.423, +0.476] | 0.188 / 1.000 | **-0.1305** | [-0.441, +0.100] | +0.1707 | -0.1569 | +0.2146 |
| exp-037 rot_raw_reg | 314 | -0.2354 | [-1.390, +1.062] | [-1.104, +0.887] | 0.000 / 0.545 | **-0.1416** | [-0.343, -0.014] | +0.2951 | -0.1703 | +0.1780 |
| exp-037 rot_shr_reg | 0 | -0.1416 | [-0.343, -0.014] | [-0.275, -0.006] | 0.000 / 1.000 | **-0.1416** | [-0.343, -0.014] | +0.0000 | -0.1703 | +0.0000 |
| exp-037 chain_shr_clf | 327 | -0.1305 | [-0.441, +0.100] | [-0.360, +0.106] | 0.000 / 1.000 | **-0.1887** | [-0.498, +0.027] | +0.0582 | -0.2269 | +0.0000 |
| exp-037 rot_shr_full | 0 | -0.2423 | [-0.536, -0.050] | [-0.435, -0.047] | 0.000 / 1.000 | **-0.2423** | [-0.536, -0.050] | +0.0000 | -0.2913 | +0.0000 |
| exp-037 rot_raw_full | 327 | -0.8616 | [-2.594, +0.808] | [-1.719, +0.029] | 0.000 / 0.483 | **-0.2423** | [-0.536, -0.050] | +0.3792 | -0.2913 | +0.2746 |
| exp-037 rot_shr_clf | 0 | -0.2656 | [-0.567, -0.045] | [-0.467, -0.055] | 0.000 / 1.000 | **-0.2656** | [-0.567, -0.045] | +0.0000 | -0.3193 | +0.0000 |
| exp-037 rot_raw_clf | 327 | -0.3544 | [-1.167, +0.478] | [-0.924, +0.320] | 0.000 / 0.710 | **-0.2656** | [-0.567, -0.045] | -0.0015 | -0.3193 | +0.1051 |
| exp-037 rot_raw_clip | 12 | -1.5225 | [-3.526, -0.019] | [-4.226, +0.233] | 1.000 / 0.258 | **-1.5225** | [-3.526, -0.019] | -0.0957 | -1.8288 | +0.0000 |
| exp-037 chain_raw_clip | 12 | -1.5225 | [-3.526, -0.019] | [-4.226, +0.233] | 1.000 / 0.258 | **-1.5225** | [-3.526, -0.019] | -0.0957 | -1.8288 | +0.0000 |

**The 95 % column beside the held-out gain is CONDITIONAL on the fitted amplitudes** and is
the wrong interval for a cross-fitted procedure: it fixes each `s_f` and resamples only the
scored rows (Sol round 1, blocking 2). The interval for the procedure resamples day blocks in
**all three** folds and refits `scale_loo` on every draw (`code/round1.py`,
`analysis/q01-procedure-ci.csv`, 2,000 draws):

| candidate | 1-day | 3-day | 7-day | conditional 1-day, for comparison |
|---|---|---|---|---|
| stack-5 | **[+0.0015, +0.1862]** | [+0.0024, +0.1930] | [+0.0047, +0.2082] | [+0.010, +0.192] |
| k2-w300 | **[+0.0120, +0.1421]** | [+0.0194, +0.1291] | [+0.0419, +0.1281] | [+0.036, +0.130] |
| K2 (v9) | [−0.0117, +0.0890] | [−0.0080, +0.0778] | [+0.0023, +0.0757] | [+0.002, +0.073] |
| refit-bulk | [−0.0174, +0.0040] | [−0.0181, +0.0043] | [−0.0260, +0.0073] | [−0.010, +0.004] |

Refitting the amplitude in every draw **moves `stack-5`'s lower bound from +0.010 to +0.0015**
and turns `K2`'s interval negative at 1-day and 3-day. Sol's own independent 2,000-draw
implementation of the same estimator returns [+0.0063, +0.1892] at 1 day and
[−0.00002, +0.1879] at 3 days for `stack-5`; the two agree that the lower bound sits on zero
and disagree by less than the Monte-Carlo noise of 2,000 draws. **The honest statement is that
`stack-5`'s held-out gain is positive with a lower bound that touches zero once the amplitude
fit is inside the resample**, and `k2-w300`'s is the one that stays clear of it on every block
width. §2a's earlier sentence claiming both cleared zero "on the 1-day and the 3-day blocks
alike" was written from the conditional interval and is withdrawn.

`analysis/q01-table.csv` carries every column, including `A`, `B`, `2A−B`, the fitted scale
and the row counts per fold, and `gain_board_rankmix` (the composition-reweighted endpoint).
`analysis/q01-dates.csv` is the per-date paired SSE, 3,724 rows; `packet/rows-{fold}.parquet`
is the per-row `d` for all 38 vectors on all 510,096 fold rows, hashed in
`analysis/q01-provenance.json`.

### 2a. What the table says

**No rebasing succeeds at full amplitude, and the failure is not marginal.** The best row is
`exp-037 chain_raw_clf` at −0.0165 s with an interval [−0.459, +0.439] on 327 moved rows — a
number that says nothing. The best row that moves the ordinary population is `refit-bulk` at
−0.0294 s [−0.065, −0.003], which is exp-036 arm 2's own mask gain read backwards and is
already in the record. **Q01's first question is therefore answered no**: no member gain in
the record helps the full predictor when inserted at v10's point at its own amplitude.

**Q01's second question is answered yes, and it is the finding.** The failure mode is
uniform: `A > 0` and `B > 2A`, i.e. the direction helps and the size does not. The fitted
scale `A/B` sits at 0.10–0.45 for the whole window lineage and is stable across the two board
months (`stack-5` 0.304 / 0.270; `k2-w300` 0.303 / 0.347; `K2` 0.225 / 0.231). Scoring at an
amplitude that never sees the rows it scores:

| finalist | LOO amplitude jul / dec / jan | all-row gain | 95 % | CV gain | clean | december | folds positive |
|---|---|---:|---|---:|---:|---:|---|
| **stack-5** | 0.304 / 0.254 / 0.270 | **+0.1017** | [+0.010, +0.192] | +0.0922 | +0.1122 | **+0.1985** | 3 of 3 |
| **k2-w300** | 0.303 / 0.432 / 0.347 | **+0.0788** | [+0.036, +0.130] | +0.0783 | +0.0959 | −0.0210 | 2 of 3 |
| k2-w300 no-EHAM-mask | 0.227 / 0.349 / 0.304 | +0.0586 | [+0.024, +0.099] | +0.0567 | +0.0716 | −0.0112 | 2 of 3 |
| w600w300w1800 | 0.137 / 0.298 / 0.194 | +0.0577 | [+0.029, +0.088] | +0.0558 | +0.0630 | −0.0918 | 2 of 3 |
| bulk-add-nmbulk-out | 0.236 / 0.336 / 0.232 | +0.0392 | [+0.005, +0.079] | +0.0321 | +0.0538 | −0.0242 | 2 of 3 |
| K2 (v9) | 0.225 / 0.081 / 0.231 | +0.0344 | [+0.002, +0.073] | +0.0278 | +0.0481 | −0.0172 | 2 of 3 |

`stack-5` is the only finalist positive on **all three folds** and the only one whose december
moves the right way, by +0.1985 s. It is also the **worst** row in the table at full amplitude
(−0.8047 s). That contrast is the mechanism: the quantity being shrunk is v10's whole
increment over stack-5 — the 600 s window block, the K2 member, the bulk anchor refit and the
two masks together — and the measurement says the rotation serves about **27 % too much of
it**.

**Where the shrinkage lands** (`analysis/q01-amplitude-*.csv`, the blend at the LOO
amplitude, paired SSE in millions of s², negative is better):

| | nm_bulk | nm_late | nonm | noac |
|---|---:|---:|---:|---:|
| stack-5 blend, july_loo | −9.77 | −1.25 | −0.08 | 0.00 |
| stack-5 blend, december | −5.46 | −7.74 | −0.26 | 0.00 |
| stack-5 blend, january | −12.02 | +1.52 | −0.06 | 0.00 |
| k2-w300 blend, july_loo | −7.11 | −3.72 | +0.53 | 0.00 |
| k2-w300 blend, december | −2.39 | +0.33 | +3.48 | 0.00 |
| k2-w300 blend, january | −8.49 | −0.43 | +2.46 | 0.00 |

The `stack-5` blend improves eight of eleven 3-day blocks on july_loo, ten of eleven on
december and eight of eleven on january, and it is not a tail effect: `nonm` improves on all
three folds where `k2-w300`'s blend makes it worse on two. By airport it gains most at LFPG
(−13.97 M s² pooled), EGLL (−11.03), EDDM (−10.27) and LIRF (−5.93), loses at LTFM (+9.95) and
LEBL (+3.85), and is exactly zero at EHAM.

**The estimator was calibrated against a null before the number was believed**
(`code/placebo.py`, `analysis/q01-placebo.csv`). The bound `s ∈ [0, 1]` means a direction with
no alignment returns exactly zero rather than a negative, so the column cannot be symmetric
about zero and the question is whether it is biased upward. It is not, and the algebra says
why: `s` is fitted on rows independent of the ones it scores, so `s ⊥ A_f`, and under a null
with `E[A_f] = 0` the expected scored gain is `−E[s²]·B_f ≤ 0` — the bound can only make the
estimator refuse to move — **but only where the fitted `s` is genuinely independent of the
rows it scores, and on the leave-one-fold-out fit it is not** (Sol round 1, blocking 1). Fold
`f` is out of the rows that fit `s_f`, but each `d_g` for `g ≠ f` is the difference of two heads
whose own training months include `f`, so `f`'s labels reach `s_f` through the vectors. The
algebraic proof therefore covers the cross-day fit — where the amplitude for a scored day uses
other days, and the fold's own prediction was trained without that whole month — and not the
LOO fit. What covers the LOO fit is the empirical null below, which destroys the pairing of `r`
and `d` while leaving that label path exactly where it is. Measured, on **40 draws** of each of
two placebos that keep each
candidate's own `d` distribution and destroy its pairing with `r` — `d` permuted uniformly
inside the fold, and `d` permuted inside each airport so the airport composition of the move is
preserved:

| candidate | real LOO / CV | placebo permuted in fold, LOO mean / max | same, CV mean / max | permuted **within airport**, LOO mean / max | same, CV mean / max | draws ≥ real |
|---|---|---|---|---|---|---|
| stack-5 | +0.1017 / +0.0922 | −0.00010 / +0.00154 | −0.00113 / +0.00121 | −0.00366 / +0.02044 | +0.01009 / +0.02476 | 0 of 40, all four |
| k2-w300 | +0.0788 / +0.0783 | −0.00014 / +0.00189 | −0.00064 / +0.00154 | −0.00043 / +0.00083 | −0.00240 / +0.00068 | 0 of 40, all four |
| K2 (v9) | +0.0344 / +0.0278 | −0.00019 / +0.00145 | −0.00070 / +0.00106 | −0.00034 / +0.00293 | −0.00207 / +0.00125 | 0 of 40, all four |

Null means run from −0.0037 to +0.0101 and the largest value any null draw produced anywhere in
the table is **+0.0248** (the within-airport placebo's cross-day fit on `stack-5`), against a
real +0.0922 on the same fit.
The estimator is also visibly able to return large negatives rather than resting on the bound:
exp-037's four `clip` arms take a leave-one-fold-out amplitude of 1.000 on july_loo and score
**−1.5225 s**.

**Support and ambiguity.** This is the part of §2 that decides what the number is worth.

*(i) The instrument is not the served vector, and this is the largest open question.* The
amplitude is fitted on the rotation, whose stacker is trained on two of three months; the
served vector's stacker is trained on twelve. If the fold-side and served-side increments
differ **only in estimation variance**, then the optimal shrinkage falls as training data
grows and `s ≈ 0.27` is an upper bound on what a served vector needs — the served gain at
0.27 would then be smaller than +0.10 and could be negative. That reading is an assumption,
not a measurement: the two increments could also differ in bias, in which case the bound does
not hold in either direction. **Nothing in this packet measures the served side**, and §2a's
first next step exists to settle it.

*(ii) Six finalists are read off one table of 37 rows and the intervals are not adjusted for
that.* On the procedure bootstrap `k2-w300` stays clear of zero at 1, 3 and 7 days
([+0.0120, +0.1421] / [+0.0194, +0.1291] / [+0.0419, +0.1281]) and **`stack-5`'s lower bound
sits on zero** ([+0.0015, +0.1862] at 1 day). Neither lower bound clears the rotation's own
0.1025 s refit half-width, so on the desk's own yardstick both are MAYBEs and not gate passes,
and they are **discovery intervals**: the two rows were chosen from the same 37-row table and
nothing here adjusts for that.

*(iii) The two January rollover rows carry a large share of the all-row January MSE.* The
clean column is quoted beside every row, and it is *larger* than the all-row column for both
finalists (+0.1122 and +0.0959), so those two rows work against the result rather than
creating it.

*(iiib) The fold side does not model the serving residual guard.* Both served files have
passed `prc.blend.apply_residual_guard`; the fold rotations they are compared on have not
(exp-036 §7, exp-030 §7). The guard touched **783 refit-bulk rows and 780 v6 rows**, and those
are not the same rows, so it does not cancel in the served difference. The fold-side amplitude
is therefore fitted on a predictor that differs from the served one on about 1,500 rows in a
way this packet does not price. It is inherited and not introduced here, and it is the second
named difference between the instrument and the served vector after training size.

*(iv) The eight-airport mask inside v10 was selected on these same three folds* (exp-036 §3)
and no instrument in §3 closes that channel. Every candidate here inherits it, so it cancels in
the paired difference — but it does not cancel in the level, and it is why the board reading
below is quoted at a discount.

**Composition, for the blend and not only the full-amplitude candidate** (Sol round 1,
blocking 4; `analysis/q01-composition-blend.csv`). Reweighting the fold rows to the ranking
months' airport × regime mix, with the LOO amplitudes held fixed, the held-out gain reads
**+0.0825** for `stack-5` (against +0.1017) and **+0.0666** for `k2-w300` (against +0.0788).
Both stay positive; both lose about a fifth of their size. The blend had never been priced this
way — `gain_board_rankmix` in the §2 table prices the full-amplitude candidate — and the
composition-adjusted number is the one the current-predictor line below uses.

**Current-predictor benefit.** At face and on the fold instrument, a
`v10 + s·(stack-5 − v10)` blend is worth **+0.10 s of all-row board** at the per-fold LOO
amplitudes, **+0.083 s** once the fold rows are reweighted to the ranking months' composition,
with all three folds positive and every mask held bit-identical. BOARD READING 9 measured the
last ordinary-row gain on this base transferring at 0.26× face, so the honest read of the board
value is **0.02 to 0.14 s**. That is real but small against the 2.115 s gap to the lead.

**The next repair or fit.** Two, in the order they should run; both are the desk's to
authorise, and this experiment performs neither.

1. *The training-size probe, and it is cheap: one rotation, about 35 s of laptop CPU per arm.*
   Refit the rotation's head on **one** month instead of two (each fold scored from each of
   its two neighbours separately, six head fits, no member fit) and re-measure the amplitude.
   If `s` rises as training data falls, reading (i) above is confirmed and the served-side
   optimum is **below** 0.27; if `s` does not move, the shrinkage is not an
   estimation-variance effect and reading (i) is wrong. This is the measurement that decides
   whether the upload below is worth a slot, and it should run first.
2. *The upload, once (1) has pointed.* The served blend is a two-file weighted sum,
   `p = v10 + s·(stack-5 − v10)` over `submissions/candidates/exp-026-stack-5.parquet`
   (SHA-256 `94f18e47…`) and `youthful-giraffe_v10.parquet` — no model runs, the 388 `noac`
   and 38,182 EHAM rows are bit-identical in both files so score-feedback rule's rows do not move, and one
   board reading settles whether the fold-side amplitude transfers. It is the only measurement
   that can separate reading (i) from a real gain, and it costs one of five daily slots.

**Not** a per-airport amplitude. The per-airport table above would be selected on the same
three folds it is scored on, which is exactly the channel exp-036 already spent and the one §3
cannot close.

### 2b. The training-size probe — reading (i) is confirmed

`code/probe.py`, `analysis/q01-probe.csv`, `logs/probe.log`. §2a's next step 1, run.

**What moves and what does not.** Only the head: the 600 s `depx` window stacker on v10's
own seven members, refit per fold on **one** neighbouring month instead of two, the same five
seeds and 300 rounds. The member frames, the hybrid NNLS anchor, the block, `v7`, `stack-5`
and both masks are the committed bytes; no member fit runs. Six head fits (three folds × two
neighbours), plus the two-month reference, which **reproduces v10's published triple at 0.0000
on every fold** before any probe number is read.

Two confounds are measured rather than argued. One month is half the rows **and** one season,
so a `half-rows` arm fits on both neighbours at a seeded 50 % row subsample: the row count
matches a one-month fit and the seasonal composition does not. And off the eight kept airports
v10 **is** `v7`, which this probe does not refit, so every quantity is reported again on the
eight kept airports alone, where every row's prediction did come from a refitted head.

| head trained on | fit rows (july) | its own clean board | LOO amplitude jul / dec / jan | mean | gain at that amplitude | 95 % | CV gain |
|---|---:|---:|---|---:|---:|---|---:|
| two months (the rotation) | 319,383 | 257.0248 | 0.304 / 0.254 / 0.270 | **0.276** | +0.1017 | [+0.010, +0.192] | +0.0922 |
| both months, half the rows | 159,691 | 257.2211 | 0.280 / 0.297 / 0.319 | **0.299** | +0.1405 | [+0.058, +0.230] | +0.1380 |
| one month, first neighbour | 165,677 | 257.3458 | 0.403 / 0.369 / 0.381 | **0.384** | +0.2782 | [+0.143, +0.434] | +0.2747 |
| one month, second neighbour | 153,706 | 257.5603 | 0.340 / 0.377 / 0.365 | **0.360** | +0.2250 | [+0.121, +0.347] | +0.2220 |

On the eight kept airports alone the same four rows read 0.271 / 0.295 / 0.385 / 0.360 — the
same ordering and the same sizes, so the unrefitted `v7` rows are not carrying the effect.

**Measured evidence.** The amplitude **rises monotonically as the head's training data falls**,
and so does the head's own board: 257.0248 → 257.2211 → 257.3458 → 257.5603. A worse head wants
more of the old vector. That is the estimation-variance signature reading (i) named, and it is
the first direct measurement of it in this record.

**The two effects separate.** Halving the rows with the seasonal composition held at two months
moves the mean amplitude by **+0.023** (0.276 → 0.299). Narrowing to a single month moves it a
further **+0.074** (0.299 → 0.372 averaged over the two neighbours). So most of the one-month
move is the seasonal narrowing and only a quarter of it is the row count — which is exactly why
the `half-rows` control is in the table, and why the extrapolation below uses the +0.023 and
not the +0.096.

**What it says about reading (i).** Reading (i) is **confirmed in direction**: `s = 0.276` from
the two-month rotation is an **upper bound** for a head trained on more, and the served head
trains on twelve months. The extrapolation, written down so it can be disagreed with:
twelve months against two is `log2(6) = 2.585` doublings of training data, and the measured
size effect is −0.0227 of amplitude per doubling, so

> `s_served ≈ 0.276 − 2.585 × 0.0227 = 0.217`.

**Its limits, plainly.** This is a two-point extrapolation from **one** measured halving,
linear in the logarithm of the row count, carried 2.6 doublings past the last measurement; the
true curve may flatten. It varies only v10's head — `stack-5` is held at its committed
two-month fold vector, while the served `stack-5` file is a twelve-month fit, and a better old
vector would push `s` back up, so the two asymmetries work against each other by an amount this
packet does not measure. And a one-month head is not a small perturbation: its board is 0.32 to
0.54 s worse, well outside the range any served head lives in.

**Why the choice is not delicate** (`analysis/q01-amplitude-profile.csv`). The fold-side gain
is quadratic in `s` and flat near its top: serving at 0.27 when the true optimum is 0.217 keeps
94 % of the available gain, and serving at 0.217 when the optimum is 0.276 keeps 95 %. The
amplitude question decides a few hundredths of a second, not the sign.

### 2c. The served candidate

`submissions/candidates/exp-038-v10-stack5-blend.parquet`, SHA-256
**`42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c`**. Built by
`code/serve.py`, checked by `code/verify.py`, recorded in
`analysis/served-exp-038-v10-stack5-blend.json` and `analysis/verify-…json`. **Not uploaded**:
the desk uploads.

**The amplitude was stated before the file was written**, and it is in the record's `s_reason`
field: `s = 0.217`, from §2b's extrapolation (0.2761 at the rotation's two months, −0.0227 of
amplitude per doubling of the head's training rows, 2.585 doublings to twelve months). No board
reading was consulted, and the value is not the fold-optimal one — the fold-side curve peaks
near 0.25 (`analysis/q01-amplitude-profile.csv`).

`p = v10 + 0.217·(stack-5 − v10)` over `submissions/uploaded/youthful-giraffe_v10.parquet`
(SHA-256 `ad962763…`) and `submissions/candidates/exp-026-stack-5.parquet` (SHA-256
`94f18e47…`). No model, no frame, no block: two hashed columns and one scalar.

| check | result |
|---|---|
| template order against `data/raw/submitting.parquet` | **PASS**, 344,841 rows in order |
| dtype, all finite | **PASS**, float64, 0 non-finite |
| minimum ≥ 60 | **PASS**, min **60.4192** |
| the 388 `noac` rows bit-identical to v10 | **PASS**, max \|difference\| **0.0** |
| the 38,182 EHAM rows bit-identical to v10 | **PASS**, max \|difference\| **0.0** |
| the two sources agree on those rows in the first place | **PASS**, **0.0** and **0.0** |
| every other row equals `v10 + s·(stack-5 − v10)` | **PASS**, max \|difference\| **0.0**, tolerance 1e-9 |
| independent rebuild from the record's two hashes | **PASS**, max \|difference\| **0.0** |
| every row lies between its two sources | **PASS** |
| SHA-256 equals the record | **PASS** |

**306,271 of 344,841 rows moved** against v10, at 6.7018 s RMS and a maximum move of 174.36 s.
The **38,570 unmoved rows are exactly the 388 `noac` plus the 38,182 EHAM rows** — the two
populations score-feedback rule and v7's mask hold — so the arithmetic of "which rows this candidate touches"
closes with no remainder.

**What it is predicted to be worth, on the fold instrument, at this exact scalar.** Serving one
amplitude on every fold rather than three fitted ones: **+0.1033 s all-row** and +0.1144 clean,
july_loo +0.1105, january +0.0965, december +0.1854. At `s = 0.27` the same curve reads
+0.1051 all-row, so the amplitude choice is worth 0.0018 s and the extrapolation is not what
decides this candidate. Reweighted to the ranking months' composition the held-out gain is
about a fifth smaller (§2a). Against BOARD READING 9's 0.26× transfer on this base, the board
value is **0.02 to 0.14 s**, and the interval whose lower bound touches zero (§2) is the reason
that range starts where it does.

## 3. Q07 stage 1 — the cached partial-sensitivity audit

Q07's first stage, in its own words, is to "use existing exp-031 baseline/candidate vectors to
compute the CHANGE in paired loss difference under partial nesting, by date/group/month, and
map every supervised ancestor", and explicitly **not** to transport a historical sensitivity
number. So the endpoint above is recomputed on exp-031's two nested instruments, with v10
rebuilt on each by exp-036's own mask arithmetic on that instrument's `refit-bulk` and `v7`.
No fit runs. The five candidates are the five exp-031 carries.

| candidate | full amplitude pub / nested / +anchor | **held-out (LOO)** pub / nested / +anchor | (CV) pub / nested / +anchor | amplitude (jul) |
|---|---|---|---|---|
| stack-5 | −0.8047 / −0.7879 / −0.7836 | **+0.1017 / +0.1023 / +0.0771** | +0.0922 / +0.1084 / +0.0883 | 0.304 / 0.307 / 0.298 |
| K2 (v9) | −0.1362 / −0.0684 / −0.0664 | **+0.0344 / +0.0587 / +0.0618** | +0.0278 / +0.0589 / +0.0578 | 0.225 / 0.297 / 0.292 |
| refit-bulk | −0.0294 / −0.0380 / −0.0341 | **−0.0022 / −0.0089 / −0.0073** | −0.0007 / −0.0004 / −0.0004 | 0.245 / 0.086 / 0.171 |
| v7 | −0.3605 / −0.3004 / −0.2935 | **+0.0075 / +0.0118 / +0.0139** | +0.0024 / +0.0314 / +0.0139 | 0.104 / 0.160 / 0.146 |
| v6 | −0.4962 / −0.4154 / −0.4093 | **+0.0033 / +0.0097 / +0.0090** | −0.0125 / +0.0094 / −0.0080 | 0.035 / 0.093 / 0.079 |

The clean endpoint on the same three instruments, which Q07 owes the same as Q01 does
(`analysis/q07-instruments.csv`, columns `gain_board_loo_amp_clean` and `..._cv_amp_clean`):

| candidate | clean LOO pub / nested / +anchor | clean CV pub / nested / +anchor |
|---|---|---|
| stack-5 | +0.1122 / +0.1093 / +0.0820 | +0.1059 / +0.1097 / +0.0880 |
| K2 (v9) | +0.0481 / +0.0680 / +0.0755 | +0.0452 / +0.0717 / +0.0775 |

**Measured evidence.** The finalist's held-out gain moves by **+0.0006** on the `nested`
instrument and by **−0.0246** on `nestedanchor`; on the cross-day fit the same two moves are
+0.0162 and −0.0038, and on the clean endpoint −0.0029 and −0.0302. The fitted amplitude itself
barely moves (0.304 → 0.307 → 0.298 on july_loo). `K2`'s held-out gain **grows** under both
closures, from +0.0344 to +0.0587 and +0.0618, and `refit-bulk` stays at zero on all three.
The level of every candidate moves together (+0.2239 and +0.2272 of board on the shared v10),
which is exp-031's own "+0.21 to +0.32 s" reading of the instrument shift, and it cancels in
every paired difference.

**What the change contains, and therefore what it cannot say** (Sol round 1, blocking 7).
exp-031 splits the move into `published → kaggle`, the machine term, and `kaggle → nested`, the
nesting term; the `kaggle` column is not in the committed OOF, so this stage measures the two
together, and it also carries the loss of one training month. Offsetting machine, nesting and
training-size terms can produce a small net change. **The supported claim is therefore the
narrow one: the gain changes little on two partial, confounded instruments, and these
instruments do not identify a cause.** The earlier wording here — that the channels "do not
produce" the gain and that two explanations are removed — did not follow and is withdrawn. The
0.1025 s historical refit half-width is quoted as a scale for reading the size of the move, not
as a standard error for it; the shared contract says it is neither a universal standard error
nor a hard gate.

**One finalist is not audited.** exp-031's OOF carries `stack_5`, `v6`, `v7`, `K2` and
`refit_bulk`. It has no `k2-w300`, so Q07 stage 1 **cannot audit the second Q01 finalist at
all**, and `K2` is not a substitute for it (Sol round 1, blocking 8).

**Ancestor map** (`analysis/q07-ancestors.json`). Eight member slots reach these six vectors:
T-tailv4, C2-cat, L2-hub8725, D1-winter, V3-allin, N1-nmtail and then either F1-fresh (blend-5's
seven, so stack-5, v6, v7) or K2-window in its slot (K2, refit-bulk, v10).
`experiments/exp-025-cloud/oof-nested/` holds exactly three members — **T-tailv4, L2-hub8725 and
F1-fresh** — so as **direct stacker feature columns** the closure is three of seven for
`stack-5`, `v6` and `v7` and two of seven for `K2`, `refit-bulk` and `v10`.

**That count is not the ancestor map, and an earlier version of this section read it as one**
(Sol round 1, blocking 6). F1-fresh still *reaches* K2, refit-bulk and v10: both are anchored on
the blend-5 hybrid, which is fitted over `NAMES5` and therefore over F1-fresh, and v10
additionally serves `v7` — an F1-fresh descendant — on every airport outside the eight-airport
keep list. exp-031 RESULTS §3 says exactly this, and it is why that experiment built
`nested-f1held` rather than reading `nested` as symmetric. So the *ancestor* closure of the v10
lineage is **not** smaller than stack-5's, and the sentence claiming it was is withdrawn.

**V3-allin cannot be refit at all** on this machine (no ERA5 snapshot; exp-031 §10.3, exp-025
`oof-nested/README.md`), and C2-cat, D1-winter, N1-nmtail and **K2-window itself** were never
in exp-025's scope — so the member that carries the whole window lineage is open on every
instrument here. The anchor channel is closed only on `nestedanchor`. The head is refitted per
fold on every instrument. **The mask channel is closed by nothing**: exp-036's eight-airport
list was selected on the same three folds it is scored on, and it is the one channel an upload
has already priced (BOARD READING 9, transfer 0.26×).

**Support and ambiguity.** exp-031 RESULTS §1 is the standing limit and it is not weakened
here: the design confounds closing a channel with removing one month of training data, the
measured member shifts include negative values, and no column is monotone in the direction of
the clean score. **Survival is not proof of cleanliness and movement is not proof of leakage.**
What this stage supports is the narrow statement above. Evidence for the stronger one would need
same-machine baseline vectors (exp-031's `kaggle` column), a matched training-size control, or
full outer isolation — none of which is cached, and all of which are refits this stage is
forbidden to run.

**Current-predictor benefit.** None directly. The audit's value is negative and useful: it does
**not** license the claim that the label channels are ruled out, and it names the two
instruments that would.

**The next repair or fit.** Q07's paid stage is not bought by this evidence, because this
evidence does not identify anything; what is bought is cheaper. (1) Commit exp-031's `kaggle`
vectors so `published → kaggle → nested` can be split, which is a write and not a fit.
(2) Produce a `k2-w300` nested vector, or carry in the ledger that the second finalist is
unaudited. (3) §2b's training-size probe has now run, and training size is the one difference
between the fold instrument and the served vector that is measured rather than assumed.

## 4. Q02 — the weather member replaced by the bulk member

Q02's first test is "weather-member to bulk64 substitution, which is unmeasured. Compare
unchanged refit, addition and replacement on identical data with the same seed, search budget
and fully refitted blend/head; retain airport masks. Evaluate all final vectors against the
v10 replay and sham control."

The weather member is **`V3-allin`** (`docs/BACKLOG.md` line 15: its frame's weather family is
the ERA5 event-hour snapshot). `D1-winter` was checked and rejected as the referent — its
shipped `deice` arm carries no external weather (`docs/CLOUD-COMPUTE.md` line 291). The
insertion point is **v10's**, not K2's: exp-027's mixed anchor (blend-K off the NM tail,
blend-5 on it) + the 600 s `depx` stacker + `noac` held at blend-5's anchor + v7's EHAM mask +
exp-036's eight-airport mask, applied unchanged. Eighteen rotations, `PRC_THREADS=4`, 695 s
total, LightGBM 300 rounds, five seeds each. `analysis/q02-arms.json`, `analysis/q02-table.csv`,
`packet/q02-{fold}.parquet`, `logs/q02.log`.

**Three gates, each an exact reproduction, run before any Q02 number:**

| gate | arm | july_loo | december | january | offset from the record |
|---|---|---:|---:|---:|---|
| A | blend-5 anchor | — | — | — | reproduces `blend-config-5.json` on all three folds |
| D | `gate-k2-hold` | 284.3195 | 205.2156 | 218.2582 | **0.0000 / 0.0000 / 0.0000** against K2 (v9) |
| E | `gate-bulk-add-hold` | 284.3591 | 205.1995 | 218.2214 | **0.0000 / 0.0000 / 0.0000** against exp-034 |
| F | `unchanged (v10 replay)` | 284.1572 | 204.9218 | 218.1506 | **0.0000 / 0.0000 / 0.0000** against exp-036 arm 2 |

Gate F is the one that matters for this packet: v10 rebuilds from the member frames upward,
through the anchor, the stacker, both masks and the seed bag, and lands on exp-036's published
triple to four decimals on every fold. Every Q02 arm below is that run with one thing changed.

**The arms**, each priced against the v10 replay. **The primary endpoint is the all-row one**
and it is quoted first with its own day-block interval; the clean endpoint now carries its own
interval too (`analysis/q02-clean-ci.csv`), so a clean point is never read against an all-row
interval — Sol round 1, blocking 5, which caught exactly that. Positive is better.

| arm | members | anchor | clean board | **vs v10, all-row** | 95 % | vs v10, clean | 95 % | december |
|---|---|---|---:|---:|---|---:|---|---:|
| unchanged (v10 replay) | 7 | mixed, 7 | 257.0248 | 0.0000 | — | 0.0000 | — | 0.0000 |
| unchanged (avail rule) | 7 | avail, 7 | 257.0248 | **0.0000** | — | **0.0000** | — | 0.0000 |
| sham refit (seed draw) | 7 | mixed, 7 | 257.0352 | −0.0101 | [−0.029, +0.007] | −0.0103 | [−0.032, +0.010] | −0.0055 |
| **addition** | 8 head, 7 anchor | mixed, 7 | **257.0163** | **+0.0145** | [−0.017, +0.040] | +0.0085 | [−0.024, +0.039] | +0.0221 |
| addition (avail anchor) | 8 head, **8 anchor** | avail, 8 | 257.0823 | +0.0834 | [−0.179, +0.360] | **−0.0575** | [−0.211, +0.083] | +0.3576 |
| replacement | 7 head, 6 anchor | mixed, 6 | 257.2520 | −0.1388 | [−0.340, +0.031] | −0.2272 | [−0.378, **−0.080**] | −0.5073 |
| **replacement (avail anchor)** | 7 head, **7 anchor** | avail, 7 | 257.5233 | **−0.2727** | [−0.661, +0.097] | **−0.4985** | [−0.731, **−0.285**] | −0.0954 |
| replacement (sham seed) | 7 | mixed, 6 | 257.2474 | −0.1328 | [−0.325, +0.032] | −0.2226 | [−0.365, −0.082] | −0.5085 |
| replacement-anchorheld | 7 | mixed, 7 (v10's) | 257.0224 | −0.0049 | [−0.057, +0.047] | +0.0024 | [−0.058, +0.061] | −0.0082 |
| drop-weather (head **and** anchor) | 6 | mixed, 6 | 257.2332 | −0.1264 | [−0.307, +0.030] | −0.2084 | [−0.343, **−0.078**] | −0.5556 |
| drop-weather, head only | 6 head | mixed, 7 (v10's) | 257.0316 | −0.0169 | [−0.053, +0.022] | −0.0068 | [−0.050, +0.035] | −0.0429 |
| drop-weather, anchor only | 7 head | mixed, 6 | 257.0144 | +0.0585 | [−0.069, +0.178] | +0.0105 | [−0.089, +0.113] | −0.1665 |
| addition-holdanchor | 8 | blend-5 | 257.1728 | −0.1294 | [−0.289, +0.013] | −0.1479 | [−0.324, +0.021] | −0.3021 |
| replacement-holdanchor | 7 | blend-5 | 257.1787 | −0.1491 | [−0.315, +0.007] | −0.1539 | [−0.343, +0.023] | −0.3114 |

and the two heads the seed question is about, at five **matched** five-seed draws each — draw
`k` differs between the two only by `B1-bulk64`, so the five differences are the paired quantity
(`analysis/q02-seed-spread.json`):

| seeds | seven-member clean board | eight-member clean board | matched all-row gain | matched clean gain |
|---|---:|---:|---:|---:|
| 7, 107, 207, 307, 407 | 257.0248 | 257.0163 | +0.0145 | +0.0085 |
| 13, 113, 213, 313, 413 | 257.0352 | 257.0222 | +0.0175 | +0.0129 |
| 23, 123, 223, 323, 423 | 257.0198 | 257.0320 | +0.0039 | −0.0123 |
| 31, 131, 231, 331, 431 | 257.0405 | 257.0282 | +0.0249 | +0.0124 |
| 41, 141, 241, 341, 441 | 257.0292 | 257.0171 | +0.0230 | +0.0122 |
| **mean** | 257.0299 | 257.0232 | **+0.0167** | **+0.0067** |
| paired se, t, wins, exact sign-flip p | | | **0.0037, t 4.51, 5 of 5, p = 0.031** | 0.0048, t 1.40, 4 of 5, p = 0.125 |

**Measured evidence.**

1. **The contracted substitution — bulk64 in the head *and* in the anchor — loses, and it loses
   more than the partial one.** Q02 asks for a *fully refitted* blend and head. The shipped
   blend rule cannot carry a partial general member, so an availability-aware rule was written
   for this: the hybrid NNLS is fitted twice per regime block, once on the rows where the
   partial members exist with them in the member list and once on the rows where they do not
   without them, and each row is served by the fit that matches its own availability. With no
   partial member the two fits coincide and the rule reduces to the shipped one — which is why
   `unchanged (avail rule)` reproduces v10 at **0.0000 on every fold** and is the gate for the
   whole block. On that rule the **replacement reads −0.2727 all-row and −0.4985 clean**, worse
   than the −0.1388 / −0.2272 of the version that could only change the head. The earlier
   reading — that −0.2272 was the contracted number — was wrong, and Sol round 1 (blocking 9)
   is right that it measured "replace V3 with B1 in the head and remove V3 from the off-tail
   anchor" instead.
2. **Putting bulk64 in the anchor as an EIGHTH member is not free either, and its one positive
   is two rows wide.** `addition (avail anchor)` reads **+0.0834** on the primary all-row
   endpoint but **−0.0575** on the clean one, with december +0.3576 and january +0.1430 all-row.
   The two endpoints disagree because the all-row January is dominated by the two
   `UNREACHED_ROLLOVER` rows; the clean endpoint, which drops them, says the arm loses. Its
   all-row interval is [−0.179, +0.360] and its clean interval [−0.211, +0.083], so neither
   endpoint resolves it. **The eighth-anchor-member question raised as a revival condition
   earlier in this file is therefore answered here, and answered no.**
3. **V3-allin's value is an INTERACTION between the two stages, not a level in either.** The
   factorial (Sol round 1, blocking 10) is now complete:

   | | V3 in the anchor | V3 out of the anchor |
   |---|---:|---:|
   | **V3 in the head** | 0.0000 (v10) | +0.0105 (`anchor only`) |
   | **V3 out of the head** | −0.0068 (`head only`) | **−0.2084** (`drop-weather`) |

   Clean board against the v10 replay. Removing V3 from one stage costs nothing measurable
   (−0.0068 and +0.0105, both intervals spanning zero); removing it from **both** costs
   −0.2084. The interaction is **−0.2121** clean and −0.1680 all-row. So `V3-allin` is
   *redundant with itself across the head and the anchor* — either stage alone carries what it
   contributes — and the earlier claim that "92 % of the replacement's loss is the anchor
   losing a full member" is **withdrawn**: `drop-weather` moved the head and the anchor
   together and could not allocate anything. What survives from that paragraph is the ratio
   itself, correctly stated: removing V3 from the head and the off-tail anchor reproduces
   **91.75 %** of the replacement's clean loss (0.2084 / 0.2272).
4. **In the head alone the substitution is a null.** `replacement-anchorheld` holds v10's
   seven-member anchor and swaps the two members only in the stacker's feature set: +0.0024 s
   clean [−0.058, +0.061], all-row −0.0049 [−0.057, +0.047].
5. **The addition is the one positive, and on the PRIMARY endpoint it is the stronger of its
   two readings.** On the blend-5 anchor it reproduces exp-034 exactly (`gate-bulk-add-hold`
   −0.1510 against v10, K2 −0.1404, difference **−0.0106** against exp-034's published
   −0.0105). On v10's mixed anchor, across five matched seed draws, it is **+0.0167 s of
   all-row board, 5 of 5 draws positive, paired t 4.51, exact one-sided sign-flip p = 0.031**;
   on the clean endpoint the same design gives +0.0067, 4 of 5, p = 0.125. That is seed
   uncertainty only and it is **not** added to the day-block interval, which spans zero on
   every single draw. The two uncertainties answer different questions: the seed test says the
   eighth member helps this head more often than not, the day-block interval says a month of
   days cannot see it.

**Support and ambiguity.** Q02's advance clause is "a stable paired gain over the sham refit".
On the primary endpoint the addition now meets *stability* — five matched draws, all positive,
p = 0.031 — and fails *size*: +0.0167 s is one sixth of the desk's 0.10 s candidate gate and
inside the rotation's 0.1025 s refit half-width, and the day-block interval spans zero on every
draw. On the clean endpoint it does not even meet stability (4 of 5, p = 0.125). It stays a
**MAYBE**. The replacement arms carry a confound that is now measured rather than argued: the
shipped blend rule cannot carry a partial general member and bulk64 is partial (defined on
97.9 / 99.0 / 99.1 % of rows, exactly where `d_takeoff_aobt` is finite), which is why the
availability-aware rule exists in `q02.py` and why both versions of the replacement are in the
table. `V3-allin` cannot be refit on this machine (no ERA5 snapshot), so no arm here can ask
what a *retrained* weather member would be worth.

**Current-predictor benefit.** The replacement: none, negative on every variant, and the fully
refitted version is the worst of them; it should not be built. bulk64 in the anchor: negative on
the clean endpoint and undecided on the primary one; not built either. The addition in the head
alone: **+0.0167 s of all-row board** across five matched seed draws, below the desk's 0.10 s
candidate gate and inside the rotation's 0.1025 s refit half-width, so it is a **MAYBE**, not a
candidate.

**The next repair or fit.** (1) *The addition's revival condition is now narrow and specific.*
The seed measurement is done and it did not settle the size; more draws buy a decimal and no
decision. The eighth-anchor-member route, which was the revival condition this file proposed
earlier, has now been run and is negative on the clean endpoint. What is left is the size
problem itself: at +0.0167 s the addition needs an instrument whose floor is below that, or a
second additive member that makes an eighth slot pay for more than this one thing — the same
condition exp-032's K3 arm carries in `docs/BACKLOG.md`. Do not revive it by re-running the
rotation and reading a friendlier seed. (2) *For the replacement*: do not repeat it in any
form. Three variants have now been measured — head only, head plus off-tail anchor, and head
plus availability-aware anchor — and they lose 0.005, 0.227 and 0.499 s of clean board
respectively. (3) *For `V3-allin`*: the factorial in evidence 3 is the finding to carry
forward, and it changes the `V3-roll171` item in `docs/BACKLOG.md`: because either stage alone
carries the member's contribution, a refit that reaches only one of them will measure close to
nothing, and the test has to move the member in both. Q02's own failure clause ("if neither
insertion helps, retain the feature idea with a new residual-alignment requirement") is met by
§2: bulk64's addition has `A > 0` and a fitted scale of 0.43 on july_loo, so the direction is
right and the amplitude is the open question, exactly as for the rest of the lineage.

## 5. Sol round 1, answered

`review/round1-gpt-5.6-sol-xhigh-2026-09-08.txt` (gpt-5.6-sol at `xhigh`, read-only, fresh
context). **Verdict: BLOCKED, thirteen blocking findings.** Every one is accepted; nine of them
moved a number or withdrew a claim, and four required new measurement. Nothing in this table is
argued away.

| # | the finding | what changed |
|---|---|---|
| 1 | `scale_loo` has an indirect scored-label path, so `s ⊥ A_f` is false and the null-bias proof does not cover the LOO fit | §2a: the algebraic claim is now made only for the cross-day fit, where it holds. The LOO fit rests on the 40-draw placebo, which leaves that label path in place |
| 2 | the LOO interval is conditional on the fitted amplitudes | §2: a **procedure bootstrap** added (`code/round1.py`), resampling all three folds and refitting `scale_loo` on every draw. `stack-5`'s 1-day lower bound moves from +0.010 to **+0.0015** |
| 3 | the multi-day LOO sensitivity was never computed | §2: 3-day and 7-day procedure intervals computed. `stack-5` [+0.0024, +0.1930] and [+0.0047, +0.2082]; the claim that both finalists clear zero on 1-day and 3-day is withdrawn for `stack-5` |
| 4 | the composition statement is numerically false and the blend was never composition-scored | §2: the true maxima are **0.3620** (exp-037 `clip` arms) and **0.0241** otherwise, **and the ordering does change**. §2a: the blend priced under the ranking mix at **+0.0825** / +0.0666 |
| 5 | Q02 quotes a clean point against an all-row interval | §4: the primary all-row endpoint leads every row and the clean endpoint has its own interval (`analysis/q02-clean-ci.csv`) |
| 6 | the Q07 ancestor map is wrong — F1-fresh reaches K2, refit-bulk and v10 through the blend-5 anchor | §3: withdrawn and corrected. The two-of-seven count is direct stacker features only; the **ancestor** closure of the v10 lineage is not smaller than stack-5's |
| 7 | Q07 does not identify channel causation; `published → nested` mixes machine, nesting and training size | §3: the "does not produce the gain" conclusion is withdrawn. The supported claim is that the gain changes little on two partial, confounded instruments |
| 8 | Q07 omits `k2-w300` and writes no date-level or clean output | §3: `k2-w300` is recorded as **unauditable** (no cached nested vector). The clean endpoint is added to `q07-instruments.csv` and `analysis/q07-dates.csv` is written, 1,674 rows |
| 9 | Q02 never puts `B1-bulk64` in the anchor, so it is not the contracted fully refitted substitution | §4: an **availability-aware anchor rule** written and run. It reduces to the shipped rule exactly (`unchanged (avail rule)` = v10 at 0.0000). The contracted replacement reads **−0.4985** clean |
| 10 | the 92 % anchor attribution is not identified — `drop-weather` moves head and anchor together | §4: the **2 × 2 factorial** run. Neither single removal matters (−0.0068, +0.0105); the joint one costs −0.2084. The attribution is withdrawn and replaced by an **interaction of −0.2121** |
| 11 | the seed analysis ignores its matched design and omits the primary endpoint | §4: rewritten as five **matched** differences with a paired SE and an exact sign-flip test, on both endpoints. The primary all-row endpoint reads **+0.0167, 5 of 5, p = 0.031** |
| 12 | registry and packet accounting: the candidate count, the dates row count, missing per-row paired SSE, missing source hashes, a mask check that checks nothing, rounded Q02 gates | headline now says **37 registered candidates** and names the four unpersisted exp-032 heads; `q01-dates.csv` is **3,038** rows and the note says why; `s<NN>` per-row paired SSE columns added to the packet; `analysis/q01-source-sha256.json` hashes 53 source files; `checks.py` now hashes the **live** mask file and compares the three Q02 gate arms **row by row** (max difference 2.27e-13). 18 checks, 0 failed |
| 13 | the fold side does not model the serving residual guard (783 vs 780 rows, not the same rows) | §2a, ambiguity (iiib): disclosed as the second named difference between the instrument and the served vector |

**Two numerical disagreements, both resolved in Sol's favour or immaterial.** Sol's own
2,000-draw procedure bootstrap returns [+0.0063, +0.1892] at 1 day and [−0.00002, +0.1879] at
3 days for `stack-5`, against this packet's [+0.0015, +0.1862] and [+0.0024, +0.1930]. The two
implementations agree that the lower bound sits on zero and differ by less than the Monte-Carlo
noise of 2,000 draws; the committed numbers are quoted because they have a script in the repo,
and Sol's are quoted beside them. Sol's matched seed statistics (+0.006738 clean, paired SE
0.004819, `t` 1.398, sign-flip `p` 0.125; +0.016745 all-row, SE 0.003715, `t` 4.51, `p`
0.03125) reproduce exactly.

**Sol's non-blocking findings adopted.** The clean column of §2's table refits its own amplitude
and is not "the same blend" (§2, corrected with the fixed-amplitude values). `scale_cv` is a
second cross-fit on the same data, not an independent fit — the word "independent" is removed
where it appeared. "At zero cost" means zero direct cost on held rows and nothing about
opportunity cost. And "variance only" does not prove a monotone decrease of the optimal scale
with training size: §2b measures it instead of assuming it, and states the extrapolation's
limits.

**What the round did not change.** The alignment algebra, the primary endpoint and the share
weights are confirmed correct by an independent pass (`max` algebra gap 2.27e-13). The positive
direction is not created by the `[0, 1]` clip. The served-byte claim — 388 `noac` and 38,182
EHAM rows bit-identical between v10 and stack-5, 306,271 moved, `max |d|` 803.48134095 — was
independently reproduced.
