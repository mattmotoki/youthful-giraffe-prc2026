# exp-007 — the rollover floor on everybody's score

## Base rate of the unreachable class in the twelve training months

Screen: recorded taxi above 6 h (physically impossible), aircraft type present
(so the `noac` sub-model cannot reach it) and |take-off − schedule| ≤ 1 h (so the
corruption is invisible in every observable column). This is exp-002's
*unreachable* sub-class; its reachable siblings are handled by the noac head.

| slice | 2025 departures | unreachable rows | rate | expected in the ranking file |
|---|---|---|---|---|
| all | 2,085,047 | 3 | 1.439e-06 | **0.50** |
| nine | 1,924,343 | 3 | 1.559e-06 | **0.50** |
| lirf | 160,704 | 0 | 0.000e+00 | **0.00** |

The rate rests on **three events in twelve months**. The exact Poisson 95 % interval on it, projected to the ranking file's exposure, is **0.10 to 1.45** expected rows — and the events are not independent draws from a homogeneous rate: two of the three are the same operator at the same airport in the same month (gpt-5.6-sol round 1, blocking finding 4). Restricting the class to genuine one-day errors leaves two events and exp-002's mean of 0.33. Treat 0.50 as a point estimate with that interval, not as a calibrated probability.

Poisson with mean 0.50: P(0 rows) = 0.609, P(1 rows) = 0.302, P(2 rows) = 0.075, P(3 rows) = 0.012.

## What each one costs every team

An unpredicted row of this class adds its own squared residual to the total squared error of *every* submission. The three observed events are not the same size — 83,335, 57,070, 86,401 s against a normal prediction — so the coefficient is their root mean square, **76,739 s**, not a single assumed day:

```
floor(k) = sqrt(k) * 76,739 / sqrt(344,841) = sqrt(k) * 130.7 s
```

(exp-002's rows quote `sqrt(k) x 147.1 s`, which assumes every event is a clean one-day error. Individually the two January events are worth 141.9 s and 97.2 s. gpt-5.6-sol round 1, blocking finding 2.)

| k | floor | a team whose score is 277.29 has skill | 278.37 | 278.48 | 291.06 |
|---|---|---|---|---|---|
| 0 | 0.0 s | 277.3 | 278.4 | 278.5 | 291.1 |
| 1 | 130.7 s | 244.6 | 245.8 | 245.9 | 260.1 |
| 2 | 184.8 s | 206.7 | 208.2 | 208.3 | 224.9 |
| 3 | 226.3 s | 160.2 | 162.0 | 162.2 | 183.0 |

The three leaders sit at 277.29 / 278.37 / 278.48 — a spread of **1.19 s** — and the fourth is 12.6 s behind. A shared floor compresses skill differences: with `score = sqrt(skill² + floor²)`, `d(score)/d(skill) = skill/score < 1`, so the same three teams span 1.19 s of score but
- k = 0: 1.18 s of skill (floor 0 s)
- k = 1: 1.34 s of skill (floor 131 s)
- k = 2: 1.59 s of skill (floor 185 s)
- k = 3: 2.04 s of skill (floor 226 s)

so the cluster at the top is what a shared floor looks like, and it is also what three equally good teams look like. The cluster is **consistent with** a floor; it is not evidence for one. What is not in doubt is the expected count above and the cost per row: the January half of the ranking file is a lottery ticket worth 131 s of RMSE per undetectable row, drawn identically for every team, and our own local folds draw from the same urn — the january fold drew two.

## What the board can and cannot say about k

Read the table the other way. At k = 3 the leaders' skill would be 160 s; at k = 2, 207 s. Our own NM-bulk RMSE — the easiest slice there is, 98 % of rows, an NM join present and a target under an hour — is 194 s on january and 227 s on july, and the bulk is where the irreducible noise of a taxi-out time lives. So a large k asks the leaders to beat us on the easy slice by a lot. **That is a plausibility argument and not a bound**: it compares a hypothetical overall skill against one pipeline's bulk RMSE, the two are different quantities, and nothing here forbids a team from holding 160 s (gpt-5.6-sol round 1, blocking finding 4). With the corrected coefficient the earlier version of this paragraph — which claimed the board *bounds* k <= 2 because k = 3 would need a 109 s bulk — no longer even has its arithmetic: 109 was computed with the one-day coefficient.

### What a submission would and would not settle

A board reading of a pipeline whose cleaned fold skill is 286 s would come back at `sqrt(skill² + floor(k)²)`:

| k | predicted board score at a fold skill of 286 s |
|---|---|
| 0 | 286 s |
| 1 | 314 s |
| 2 | 341 s |

Those are 28 and 27 seconds apart, so a reading does carry information about k. But it does **not** measure k, and the earlier claim that it does was too strong. A single score conflates at least five things: how many such rows the ranking file holds, how large each one is, whether any model responds to the identity axis of finding 1, how the folds transfer to 2026 at all, and the January-2026 shift that `docs/shift-2026.md` measured and no fold can price. What a reading gives is the SUM of those, which is still the single most informative number the desk could buy — it is just not a decomposition.

## Per-month history of the draw

| month | departures | unreachable | of those, non-LIRF |
|---|---|---|---|
| 2025-01 | 153,706 | 2 | 2 |
| 2025-02 | 143,731 | 0 | 0 |
| 2025-03 | 164,450 | 0 | 0 |
| 2025-04 | 175,288 | 0 | 0 |
| 2025-05 | 185,202 | 1 | 1 |
| 2025-06 | 183,142 | 0 | 0 |
| 2025-07 | 190,713 | 0 | 0 |
| 2025-08 | 191,181 | 0 | 0 |
| 2025-09 | 183,951 | 0 | 0 |
| 2025-10 | 185,674 | 0 | 0 |
| 2025-11 | 162,332 | 0 | 0 |
| 2025-12 | 165,677 | 0 | 0 |