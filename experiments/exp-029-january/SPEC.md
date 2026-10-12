# exp-029 january — SPEC

Queue lever **L4**, `docs/QUEUE-2026-09-07.md`; assignment `assignments/january.md`.
One thread, no GPU, ~4 minutes of compute in total.

## My reading of the assignment, and the readings I rejected

**The reading I took.** The deliverable is a **measurement of L4 on the artifact
that ships**, not the delivery of a correction. The assignment names the base:
stack-5's `keep-noac` rotated out-of-fold vector
(`experiments/exp-026-rebuild/oof/stack5-{fold}.parquet`). exp-014 measured the
mechanism on **candidate 1** — the exp-012 `members-cand1.json` blend of
`T-tailv2`, `C2-cat` and `L2-hub` — two member generations earlier. So the first
question any honest execution has to answer is whether the mechanism is still
there on the new base, and the answer sets the size of everything after it. I
therefore put candidate 1, blend-5 and stack-5 on the same five airport-days
before fitting anything, and I report the correction's value on all three.

**Readings I rejected.**

1. *"Fit the correction and report the january fold gain against the 0.53 s
   gate."* That is the literal instruction and it is not decidable: the january
   fold carries 1,129 flagged rows of 153,706, and the best additive per-day
   constant on them — an oracle fitted on the day it is applied to — is worth
   **0.029 s** of whole-fold clean RMSE. The gate is 18x the ceiling of the
   thing it gates. I report the gate result as asked *and* the ceiling beside
   it, because a FAIL that does not say the instrument was too coarse is a
   claim about the idea when it is a claim about the ruler.
2. *"Take exp-014's -83 s / -144 s as the size of the effect."* Rejected after
   measuring: those are candidate 1's numbers. On blend-5 the same five days
   give -41.4 / -74.5 and on stack-5 -22.3 / -41.0. Quoting the old number
   would have priced the lever at 10x its value.
3. *"Fit on the january fold's own two flagged days."* That is the fold whose
   score the gate reads, so a rule fitted there is not a measurement. I fit
   leave-one-**fold**-out (`lofo`) as the headline and report the assignment's
   leave-one-**day**-out (`loo`) beside it. On the step shape the two coincide,
   because each fold contributes at most one day to each exceedance band.
4. *"Build the correction into the submission."* Not mine to decide and not
   supported by the numbers. `code/apply.py` is written, runs against the live
   `ranking.parquet`, and is **not adopted**; `RESULTS.md` §7 says what would
   revive it.
5. *"Use `docs/shift-2026.md`'s `december` fold as the January analogue."*
   Ruled out there; december is the calm control and is reported as one.

## Frozen before any number was computed

- **The envelope, in one line.** `span = MVT_TIME_UTC_mvt - AOBT_3_flt` on `DEP`
  rows; `day_span_mean[apt, day]` is its mean over an airport-day;
  `envelope[apt]` is the maximum of that over the fold's **training** months
  (over all twelve 2025 months on the ranking side); `exceedance =
  day_span_mean - envelope`; `flagged = exceedance > 0`. Both columns survive
  into `ranking.parquet`, so the flag is label-free. This is exp-014
  `code/q2_days.py`'s `obs_mean` / `max25` / `over_max` renamed, and the gate on
  the rebuild is that it reproduces exp-014's five analogue days and their
  exceedances exactly.
- **The correction form.** `corrected = max(pred + c(exceedance), FLOOR)` with
  `c(x) = 0` for `x <= 0`, so in-envelope rows cannot be harmed by construction
  and the object is a function of a served vector — it composes on stack-5 today
  and on the exp-028 window arm later.
- **Adoption.** A shape is adopted only if january clean improves by more than
  the 0.53 s seed sd with a calendar-day-clustered 95 % interval clear of zero,
  july_loo is not harmed by more than 0.1 s, and the in-envelope rows are not
  harmed. Declared before fitting; nothing was moved afterwards. **The queue
  revised this gate mid-run** — the 0.53 s seed sd became the interval itself
  plus a 0.2 s clean-board ship bar (`docs/QUEUE-2026-09-07.md` L4, 2026-09-07).
  `RESULTS.md` §3a answers the revised wording line by line and the verdict is
  the same under both.
- **Winsorisation** is `(0.025, 0.975)` on the row residual before the day's
  target constant is taken, and both settings are reported for every shape.

## Threads and compute

1 thread (`PRC_THREADS=1`, `OMP_NUM_THREADS=1`), no GPU, box under the laptop
load ceiling at every run. The competition data landed on this machine at
about 16:05 UTC on 2026-09-07, part way through; every number computed before
that from committed tables was re-verified against `data/raw` afterwards
(`code/envelope.py` `verify_against_raw`, `code/apply.py` cross-check).
