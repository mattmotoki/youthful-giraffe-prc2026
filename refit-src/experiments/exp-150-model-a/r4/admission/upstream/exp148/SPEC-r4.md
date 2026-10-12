# exp-148 SPEC revision 4: push R2 (order of work, variants, rules)

Amends SPEC-r2.md (sha256 c1f68e7d…) and SPEC-r3.md (e0905db8…); SPEC.md, SPEC-r1.md and the
rest of SPEC-r2/r3 stay in force. Written 2026-09-30 about 19:00 UTC, after the board read of v35
(`R2b-w100-kno`, 241.7342 against v32's 242.7480) and the why read (REPORT section "Why v35 beat
v34", `results/dev-why-v35.md`), and before any fit, feature build or score of anything below.
Assignment: the desk's messages after v35 (2 of 2): push R2 until it stops moving; the
order is the worker's choice, written here before any fit. Lanes: cores 32-55/96-119 and GPU 1
(free since 18:52 UTC); GPU 0 from 10-01 00:00 UTC. No board-derived weight, mask or rate.

## Order

1. **V, a fixed blend of v35 with v32 (no fit; tonight).** For any two vectors p and q on the
   same rows, MSE(w·p + (1−w)·q) = w·MSE(p) + (1−w)·MSE(q) − w(1−w)·mean((p − q)²): an identity;
   its last term needs no label. v35 and v32 are both strong (v35 is a level-2 stack, v32 is
   exp-137's route) and differ by RMS 65.7 s on the ranking rows (`ranking-R2b-w100-kno.json`),
   so the equal blend gains 0.25 × 65.7² ≈ 1,079 s² per row over the mean of the two. Largest
   expected gain per hour of all items; the only one ready before 00:00 UTC.
2. **M, month-hold member fits for March, October and November, then a level 2 on nine months
   (R4).** The why read: the level 2 must learn corrections that hold from month to month
   (R2-final's CV removes 8 % to 26 % of the anchor MSE by held month, January the least). More
   months give more of that signal. Long pole: about 1.5 to 2 h of member fits on four lanes,
   then the stackers.
3. **L, a level-2 variant: a season input.** `ctx_summer` = 1 for months 4 to 9, else 0, added
   to the R2 input list (the rest unchanged), on the best month set when its fits exist (R4 if
   it beat R2, else R2's six months).
4. **N, new members under the R2 contracts** (modela's Am; the grind's improved L and Lp): each
   needs the fits `jjx-12`, `jjx-02`, `jjx-06`, `jjx-08`, `janjul`, `final` (and `jjx-03`,
   `jjx-10`, `jjx-11` if R4 is the base). It enters the level 2 as a ninth member, or replaces
   the member it improves, when its fits exist.
5. **K, keep-set variants (last).** The why read shows that a keep set judged by local G against
   v32's saved route is biased toward v32 (its local values carry label paths). One variant is
   scored with V for information only: `knoc` = kno plus the clock rows without AOBT (local 1,262
   rows, R2b there −160 against v32; 1,697 ranking rows). Not a candidate by itself.

## Readings rejected

(a) All ten other months now: April, May and September have no exp-046 ungated block (the class
    and the `obs_*` inputs of `features_r2.py` need it); each block took hours to build
    (2025-09-09/10). Revival: R4 beats R2 locally; then build those three blocks.
(b) A blend weight fitted on the local rows, or computed from the board scores. The weight
    of V is fixed here at 0.5: the symmetric choice when neither vector is known to be better
    locally (R2b against v32: −28, a tie).
(c) Blends of v35 with v34 or with the anchor: the anchor blends lost at every weight (SPEC-r3);
    v34 is worse than v32 on the board and differs from v35 by RMS 63.6 s, with no local
    evidence that it adds anything.
(d) CPU CatBoost for the R4 stackers: R2-final took 3 h on six months; nine months with nine CV
    folds is about 2.2 times the work. R4 uses CatBoost on the GPU (item M.3), with a control.

## V. The blend (no fit)

- Served vector `B50` = 0.5 × v35 + 0.5 × v32 on every ranking row (v35 =
  `results/ranking-R2b-w100-kno.parquet`; v32 = the served v32 vector, `v32-exp137-proxy.parquet`).
  On the 26,310 rows where v35 already holds v32's value the blend is v32. Template order,
  finite, at least 60 s (both inputs are at least 60 s). Write-once `ranking-B50.parquet`/`.json`.
- Local check under R2's condition (`code/score_r4v.py`): the local v35 vector (R2b-w100-kno,
  rebuilt; must give the saved G +178.2) blended with v32's saved route, w on v35 in {0.25, 0.5,
  0.75}, and `knoc`; G against v32's saved route and against the local v35 (all rows, clean,
  frozen interval, views). V is a candidate if the local G of the w 0.5 blend against the
  local v35 is above 0. The local G is optimistic for every blend with v32 (label paths), so it
  is a sign check, never a weight choice; the w 0.25 and 0.75 rows show the pattern only.
- Frozen forecast (`code/forecast_r4v.py`): the identity with the board RMSE of v32 and v35 and
  the ranking-row mean((v35 − v32)²). The board scores enter the forecast only, never the weight
  or the vector. The forecast assumes that the board is the plain RMSE over the ranking rows or a
  random part of them; a large miss says the metric or the scored rows differ from that.
- If B50 beats v35 on the board, later candidates X of items M, L and N are served the same way:
  0.5 × X-kno + 0.5 × v32, and judged locally by the same blend against the local B50.

## M. Nine months (R4)

1. **Splits.** `pilot.R4_MONTHS` = 2025-03, 2025-10, 2025-11; `jjx(month)` accepts them (the same
   rule: excluded = January, July, the month, and their month-hold buffer days; held = the
   month); `FITS_R4` = `jjx-03`, `jjx-10`, `jjx-11`. R2's constants and owners do not change.
2. **Member fits.** 8 members × 3 months = 24 fits, the same drivers and settings as SPEC-r2 §1
   (`code/lane.sh`; L and Lp through `code/learner_lane.sh` with a new kernel whose group adds
   `features-r4`, the proxy export of the three months). Write-once artifacts, one log per fit.
3. **Row features** of the three months: `code/features_r2.py` with a month argument (the same
   code and checks; December's frozen-class check applies only to December), then a combined
   table of all nine months. Label-free.
4. **Level 2 (stage `R4`).** Owners `q01` (training: every month but January), `q07` (every
   month but July), `qfinal` (all nine; scores the ranking rows). SPEC-r2 §3 unchanged (inputs
   less `v23_*` and `ctx_month`, leave one training month out, buffer rows dropped, the two
   recorded rows without features dropped), except: CatBoost with `task_type` GPU (GPU 1; GPU 0
   from 00:00 UTC), the other CatBoost settings as before. **Control `R2g`**: owners r01 and r07
   on R2's six months with the same GPU CatBoost, so R4 against R2g reads the months alone.
5. **Form** by SPEC-r2 §3's rule on q01 and q07. **Candidate rule:** the chosen R4 form with the
   kno keeps (SPEC-r3) is a candidate if its joined local G against the local v35
   (R2b-w100-kno) is above 0; served as SPEC-r3 §3 (or as the blend of section V if B50 won).

## L. Season input

Stage `R4s` (or `R2s` on R2's six months): as M.4 with `ctx_summer` added. Candidate rule: its
joined local G against the base it extends (R4 or R2b, with the kno keeps) is above 0.

## Every candidate

Frozen forecast before any upload; the desk's review; the desk uploads. Reports to the desk
with `needs-decision` when a candidate is ready. Scores: `prc.gscore.score`, January and July 2025
joined (344,419 rows), all rows and clean, frozen interval, views. No board number enters a
vector, a weight, a keep set or a choice.
