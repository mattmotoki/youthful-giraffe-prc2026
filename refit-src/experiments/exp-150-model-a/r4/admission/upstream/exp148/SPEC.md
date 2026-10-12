# exp-148 SPEC: the shorter route, one stacked learner fitted per split and served

Worker `valid` (claude-opus-5-5, tier hard), 2026-09-30. Branch `exp148-short-route` (from
`exp147-foldserve`). Assignment `docs/assignments/2026-09-30-exp148-short-route.md` (Matt's decision:
train). Protocol `docs/VALIDATION.md` week-pair-1 (`prc/weekpair.py`, `prc/gscore.py`).
This SPEC is frozen before the first stacker fit; its sha256 goes into every stacker record
and the scoring record. A change after the freeze is a new SPEC revision (`SPEC-r1.md`).

## Reading of the assignment

Matt's words: "one learner over the member predictions, head, queue, v23 and proxy features,
fitted per split and served". I read it as a **two-level** route. Level 1 is a set of members,
each fitted on departure labels only (no fitted input). Level 2 is ONE LightGBM that reads the
members' out-of-fold predictions and the label-free INPUTS of the head, the queue arm, v23 and
the proxies. Every level then meets VALIDATION §3 strictly: an outer(k) stacker trains on the
inner(k, j) predictions, whose fits exclude pairs k and j, and scores pair k with the outer(k)
fits; the served stacker trains on the five outer predictions and scores the ranking rows
with the final fits. Rejected readings: (a) keep the head, the queue arm and v23 as fitted
corrections and stack their outputs: each is fitted on a residual of a fitted base, so each
adds a level, and three levels do not fit five pairs with the strict rule (exp-147 §2);
(b) a weighted blend of v32's saved stages: the saved vectors are not admissible (§3);
(c) the six-member stack of v22 (exp-026 anchor + stack5) as level 2: it cannot read the
clock inputs, which carry most of v32's gain over v12.

## 1. Members (level 1)

Each member is fitted 16 times (`pilot.FITS`): outer-1..5, the ten inner pairs {a, b} (one
fit writes inner-a-b and inner-b-a, owners a and b), and `final` (all 2025 labels, scores the
344,841 ranking departures). Every fit: training rows = `Split.label_mask`, checked by
`assert_training_frame(target=...)`; features label-free (masked context); artifacts through
`pilot.write` (split manifest, owner, input and code hashes); read back through `pilot.load`
(`admits`, owner, hash). Recorded settings of each member are kept (VALIDATION §10: a refitted
member keeps its recorded settings); the round count is the member's fold count on every fit.

| member | driver | settings | where |
|---|---|---|---|
| T-tailv4 | `members/T-tailv4_wp.py` | TailModel, 500 rounds, seed 7, 6 threads | workstation CPU |
| C2-cat | `members/exp008_wp.py --member C2-cat` | CatBoost depth 8, 5,325 rounds, seed 7, GPU | workstation GPU 1 |
| L2-hub8725 | `members/exp008_wp.py --member L2-hub8725` | LightGBM Huber 1,200, 8,725 rounds, seed 7, 6 threads | workstation CPU |
| D1-winter | `members/D1-winter_wp.py` | exp-012 winter arm, 1 seed (7), 6 threads | workstation CPU |
| F1-fresh | `members/F1-fresh_wp.py` | exp-018, 1,200 rounds, seed 1, GPU; encodings month-out for scored rows (`learner_wp.add_encodings_month_out` rule) | workstation GPU |
| N1-nmtail | `members/N1-nmtail_wp.py` | exp-010, mode phi, 900 rounds, seed 7; NaN off NM | workstation CPU |
| L (exp-134) | `learner_wp.py` | exp-130 learner, 149 + 21 proxies, 1,500 rounds, final_refit logic, encodings month-out for scored rows | Kaggle CPU |
| Lp (exp-137) | `learner_wp.py` | as L plus the 2 row clocks (172 features), all non-ghost rows scored | Kaggle CPU |

## 2. Label-free row features

`code/features.py` (Mac; checks: 2026 class 196,256 / 148,197 / 388, 0 class mismatches
against the frozen 2025 class and the served 2026 class) and the learner kernel's
`proxy-features.parquet` (21 proxies with 12-month denominators, 2 row clocks):
context (airport, month, hour, weekday, class), the head's 27 surface-block columns
(`obs_*`) and 76 shipped columns (`shp_*`), the queue's 12 label-free neighbour columns
(`q_*n_anom*`, `q_*n_n_*`) and the raw file and observed taxi, exp-127's four HGB inputs
(`v23_*`), exp-134's proxies and exp-137's row clocks.

## 3. The stacker (level 2): `code/stacker.py`

- **Anchor** a = row mean of the available base members {T-tailv4, C2-cat, L2-hub8725,
  D1-winter, F1-fresh} (the fitted members read no fitted input).
- **Target** y − a; LightGBM, L2 loss; `PARAMS` in the code (learning rate 0.03, 63 leaves,
  min 100 rows per leaf, feature fraction 0.7, bagging 0.8, λ2 = 10, seed 601,
  deterministic); categorical airport and class.
- **Features** (`feature_list()`, hash in each record): the 8 member predictions minus a, a;
  the context; `obs_*`, `shp_*`, `v23_*`; the 12 label-free queue columns; the 12 queue
  residual columns recomputed as self-excluded neighbour means of (taxi − a) with the
  exp-047 `state.neighbourhood` code, neighbours restricted to the same UTC day, so each
  row's neighbours come from the same fit on fold and serve alike; the raw file and observed
  taxi; the proxies and row clocks; five differences (observed taxi, row clocks, file and
  observed queue taxi minus a).
- **Rounds**: leave one training pair out over the training pairs (four for outer(k), five
  for final), early stopping 200 on the held pair, rounds = mean best iteration; then one
  refit on all training rows. Only inner predictions choose the rounds (VALIDATION §10).
- **Rows and classes**: all classes are fitted and served. Served value max(a + f(x), 60 s).
  On the ranking rows of the unsupported cells (VALIDATION §5, 22,721 rows) the candidate
  keeps v32's served value.
- **Cross-fitting**: owner k trains on the held rows of inner(k, j), j ≠ k (pairs other than
  k, minus k's buffer days), with the members' inner(k, j) predictions; it scores pair k with
  the outer(k) predictions. Owner `final` trains on the five outer predictions (all January and
  July rows) and scores the ranking rows with the final fits. `pilot.load` checks every read;
  the training frame passes `assert_training_frame(target=y)` against the consumer split.

## 4. Scores (`code/score.py`)

- **Candidate**: the five outer stacker predictions joined by `weekpair.join_outer`.
- **Comparators**: (1) **anchor** (the served mean of the five base members): clean, from
  the same fits; (2) **v22-input stacker**: the same stacker restricted to the inputs that
  v22's route reads (the six base members, `obs_*`, `shp_*`, the 24 queue columns, context):
  clean, "v22's route as one learner"; (3) **v22 and v32 saved routes** (`code/refs.py`,
  saved month-fold vectors, served weights): reference only, not admissible (they carry the
  label paths T1 L1-L7, which favour them).
- **G** by `prc.gscore.score` (all rows primary, clean view), with the §7 airport-day interval,
  the class, airport, month, class × AOBT and label-band views, G per pair, the top five
  airport-days, and the unsupported cells.
- **Clock-loss view** (§9, schemes `coverage` and `lirf_july`): the candidate re-predicted
  with the ADS-B-derived stacker inputs blanked on the dark airport-days (`obs_*`, `v23_*`,
  row clocks, observed taxi, the `No_` queue columns) and the class set to ordinary there,
  against the anchor (whose members read no ADS-B input). Limitation: the members L and Lp
  are not refitted on masked points.
- **Decision rule for the upload candidate**: G of the candidate against v32's saved route
  is positive (all rows) and the candidate beats the anchor and the v22-input stacker. If
  not, report `needs-decision` with the repairs.

## 5. Final fit and ranking vector

`stacker.py --owner final` on the final member fits; ranking vector in template order
(`submitting.parquet`), 344,841 rows, finite, ≥ 60 s, v32's values on the unsupported cells;
sha256 in the REPORT. Forecast: local G against v32's saved route and the board range.
