# exp-017 — training-label hygiene

**Recipe: `prc.data.keep_reachable` (VERSION 6), default off.** Drop the four
rows no input can reach from bulk training. Worth **−1.56 s on july_loo** and
**−3.52 s on july**, from four rows. Recency weighting is closed by the gate.

## 1. The first screen was void, and the reason is the whole experiment

A single `keep` mask over the training frame removes **11 of the 50 rows the v4
noac band head is fitted on**, collapsing every band cell to p = 0. Its july
"win" was not hygiene: 8 of july's 12 validation band rows are placeholders,
which p = 0 predicts right by accident, and january lost 3.83 s.

The fix is two changes, both shipped before any arm ran:

- `prc.tail_model` 2.4.0: a `keep` mask **reaches the learned heads only**. The
  band head is a counting estimator over ~50 rows and is fitted on all labelled
  rows regardless.
- `prc.data` 5: `clean_training_labels(protect=…)`, rows a recipe must never
  drop. The noac rollovers are passed as `protect`, so arm A drops **2** rows
  rather than 6.

Both were verified before the fits, at the desk's instruction: arm 0 and every
hygiene arm produce byte-identical `band["cells"]`, `band["bands"]` and
`band["offset"]` on all three folds — and identically **even if `keep` did reach
the band**, because `protect` already spares those rows.

## 2. The classes do not share a regime

2,085,047 labelled 2025 departures (`catalogue.py`):

| class | n | nm | nonm | noac | max \|y\| | share of training SSE |
|---|---|---|---|---|---|---|
| rollover detectable | 1 | **1** | 0 | 0 | 87,002 | 1.2 % |
| rollover undetectable | 14 | 0 | 2 | **12** | 88,392 | 16.4 % |
| non-positive | 388 | 137 | 248 | 3 | 12 | 0.062 % |
| duplicate | 40 | 22 | 18 | 0 | 21,308 | 0.116 % |

`impossible` (369) is a **strict subset** of non-positive, so they are one
treatment. The one *detectable* rollover is in the `nm` bulk head at y = 87,002 s
among neighbours near 700 s; 12 of the 14 *undetectable* ones are `noac`, and 11
of those are the band head's own training rows. **The two facts point opposite
ways, so a single mask must be wrong for one of them** — which is exactly what
the void screen did.

## 3. The arms

7 arms × 3 folds × seeds 7/23/51 = 63 fits (`work/arms.csv`). Deltas against
arm 0:

| arm | july | july_loo | january (raw) | january (clean) | rows dropped |
|---|---|---|---|---|---|
| **A1 unreached** | **−3.52** | **−1.56** | +0.49 | −0.27 | **4** |
| A rollovers | −2.92 | −1.29 | +0.49 | −0.30 | 2 |
| A+C | −3.26 | −0.93 | +0.45 | −0.39 | 395 |
| A+C+W | −3.63 | −0.66 | +0.59 | −0.15 | 395 |
| C non-physical | −0.57 | +0.13 | −0.19 | −0.15 | 393 |
| W recency | −1.49 | +0.13 | +0.40 | +0.12 | 0 |

Control seed sd: july_loo **0.09**, july 0.61, january 1.13 raw / 0.54 clean.

**Three readings.**

1. **January's sign flips with the metric**, on every rollover arm: +0.49 raw
   against −0.27 clean. January validates 2025-01, which holds two of the four
   unreached rows, and exp-020 measures those two at 116 s of that fold's RMSE.
   The raw move is also inside the control's own 1.13 s seed spread, so it is
   not even a reliable loss. **The void screen's 3.83 s january "reversal" was
   this artefact, not hygiene.**
2. **The rollover drop is the entire effect, and C and W dilute it.** Adding C
   to A costs july_loo (−1.29 → −0.93) and adding W costs more (−0.66). C alone
   is a wash (−0.57 july, +0.13 july_loo) — which is what its 0.062 % of
   training SSE predicted before any fit. 757 non-physical rows are rounding.
3. **Four rows beat 395.** A1 drops only `UNREACHED_ROLLOVER` and beats every
   arm on july_loo. A row *no input can reach* is pure noise to an L2 fit; a
   *reachable* rollover is structure, and dropping it costs more than it saves.
   That distinction, not the artefact count, is the finding.

## 4. Recency weighting: closed by the gate

`w = 0.5^(age/180 d)` gives −1.49 s on july but **+0.13 on july_loo and +0.40 on
january**. Mixed sign across the ranking-analogue folds, so it fails the
both-folds gate. Per the desk's rule the 90-day and 365-day half-lives were
**not run**: they were conditional on 180 showing something, and it did not.

**Revival condition**: a fold design where train and validation are separated by
a *fixed* horizon. All three folds here hold one month out of twelve, so the
newest training month is adjacent to validation in two of them and eleven months
away in the third; a half-life cannot mean the same thing across that.

## 5. What this does not claim

The recipe is **training-side only**. The four rows stay in the scored
population — `rmse_clean` is the separate scoring-side convention (exp-020) and
the two questions must not be merged. Dropping a row from training does not make
it reachable.

## 6. The day-clustered interval on A1

The table in §3 reports seed spreads, which cannot settle january: rows within a
calendar day are not independent, so an unclustered spread overstates
significance. `interval.py` reruns arm A1 and the control saving **per-row**
predictions (18 fits), averages the three seeds per row, and resamples calendar
days of take-off with replacement, 2,000 draws.

| fold | metric | A1 − control | 95 % CI | |
|---|---|---|---|---|
| july | `rmse` | −3.192 | [−5.694, −0.720] | clear of zero |
| july | `rmse_clean` | −3.192 | [−5.882, −0.758] | clear of zero |
| january | `rmse` | **+0.560** | **[−0.538, +1.733]** | **spans zero** |
| january | `rmse_clean` | −0.171 | [−0.594, +0.401] | spans zero |
| july_loo | `rmse` | −1.258 | [−2.633, −0.080] | clear of zero |
| july_loo | `rmse_clean` | −1.258 | [−2.723, −0.053] | clear of zero |

**The january loss is not real.** The +0.49 s that voided the first screen, and
that §3 still shows at +0.49, **spans zero** under day clustering. A1 does not
harm january; it is neutral there and wins the two july folds. The clean-metric
gain on january is equally not significant, so the honest statement is
*neutral*, not *positive*.

Two qualifications on this table, both against my own result:

1. **`july` and `july_loo` validate the same month.** 2025-07 is the validation
   month for both, so this is *one* month's evidence twice, not two independent
   confirmations, and the two rows are strongly correlated by construction. The
   real reading is: significant on 2025-07, neutral on 2025-01.
2. **`july_loo`'s interval only just clears** ([−2.633, **−0.080**]). With 31
   day-clusters the intervals are wide. The recipe is adopted on a point
   estimate that is consistent across every arm containing the rollover drop and
   on a mechanism that is understood, not on the strength of this bound.

The deltas here (−3.192 / −1.258 / +0.560) differ slightly from §3's
(−3.52 / −1.56 / +0.49) because this run averages *predictions* across seeds
before scoring, while §3 averages the seeds' RMSEs. Prediction-averaging is the
better estimator and the one a shipped model would use; the two agree in sign
and magnitude everywhere.
