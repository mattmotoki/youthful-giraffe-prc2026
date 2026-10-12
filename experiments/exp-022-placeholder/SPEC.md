# exp-022 — the schedule-placeholder regime over the whole population

**Status: acked by the desk. Runs after exp-017 releases the single-fit slot.**

## Reading of the assignment

Extend the placeholder machinery from the `noac` rows to the population. I read
the deliverable as **a new regime inside `TailModel`, fitted on its own
population**, not as a widening of the noac head — see the negatives section;
exp-018 measured that widening directly and it is poison. The readings I
rejected: (1) a post-hoc correction applied to the existing NM head's output,
which cannot be scored against a control cleanly; (2) putting the class in as a
feature and letting the bulk tree find it, which the p-rule says wastes the
structure — the payoff here is an *anchor swap*, and a tree cannot reproduce a
continuous input feature.

## Correction to the assignment's premise, measured before designing

The brief says `BLOCK` equals `SCHED` **to the second** on 4.67 % of all 2025
departures. That wording is the **fresh lineage's** (`exp-018-fresh`
COMPARISON.md §2 item 2), relayed intact by the desk, and **it does not
reproduce.** Exact equality holds on **0.32 %** of
2025-07 departures (618 rows) and 0.39 % of 2025-01. What reproduces the 4.67 %
is equality **at minute resolution**: 3.93 % in july, 5.07 % in january, and the
airport ordering the brief gives (LIRF 8.27 %, LEBL 5.16 %, LTFM 4.91 %, EGLL
4.05 %, EDDM 3.85 % in july) is the same-minute one. So the class is
`BLOCK.floor('min') == SCHED.floor('min')`, and this SPEC uses that — desk
confirmed 2026-09-06. Everything else in their item 2 stands; only the
resolution word was wrong, and their 4.67 %, their airport ordering and their
over-representation of the class in the tail all reproduce at minute resolution.

The distinction changes what the regime can promise:

| | exact equality | same minute |
|---|---|---|
| share of 2025-07 departures | 0.32 % | 3.93 % |
| `y = MVT − SCHED`? | **exactly, on 100 % of rows** | to within δ = `BLOCK − SCHED` |
| δ | 0 | mean **23.3 s**, sd 24.7, bimodal (25 % ≤ 2 s, 75 % ≥ 55 s) |
| RMS of predicting `MVT − SCHED` | 0 | 33.97 s, or **24.72 s** after the fitted mean |

So the placeholder branch is **not an identity**; it is an anchor with a ~25 s
residual. In practice that barely matters — see the bound — but the head must
fit the offset rather than assume zero, and no write-up may call it exact.

## The bound, measured before building

Against the delivered model's own july OOF (`exp-009-noac2-oof-v3.parquet`):

| | |
|---|---|
| july RMSE, delivered model | 318.767 |
| same-minute non-noac rows | 7,439 (3.90 %) |
| model RMSE on them | **394.6 s** (315.3 s elsewhere) |
| their share of july SSE | 5.98 % |
| **july with an ORACLE detector** | **309.130 (+9.637 s)** |

**A perfect detector is worth 9.64 s of july.** The offset is worth 0.035 s of
that, which is why the exact/same-minute correction changes the description and
not the value. exp-018 realised **1.6 s** with an AUC-0.868 classifier — about
17 % of the bound, which is what the p-rule predicts for a detector of that
quality. The lane is worth building; it is not worth promising 9.6 s.

## The regime and the heads

| head | its training rows | what changes | why |
|---|---|---|---|
| **placeholder classifier** (new) | all non-`noac` departures with a label | fits `p(BLOCK.floor('min') == SCHED.floor('min'))` | the detector; label is training-side only |
| **placeholder anchor** (new) | non-`noac` rows in the class | fits the scalar offset `E[(MVT − SCHED) − y]` (≈ 23.3 s july) | the branch is an anchor with a residual, not an identity |
| **bulk `nm`** | unchanged | unchanged | the mixture composes at predict time; the tree is not refitted on a subset |
| **bulk `nonm`** | unchanged | unchanged | 4 of july's same-minute rows are no-NM non-noac; too few to justify a separate branch, and they ride the same mixture |
| **noac offset / noac band** | **unchanged, and asserted so** | nothing | exp-018's negative 2; the control is that noac RMSE moves by < 0.01 s |

Prediction, for a non-noac row: `p·(MVT − SCHED − offset) + (1 − p)·head`, the
RMSE-optimal expectation over the two anchors. **Never the argmax** — that is
exp-018's negative 1, and the p-rule says the expectation is what pays.

## Observables for the classifier

No `BLOCK`, no `TAXITIME`. Measured separators from 2025-07 in brackets.

- `LOBT == SCHED` [0.867 in the class against 0.671 outside], `MVT − LOBT`
- `AOBT_3 − LOBT`, `MVT − AOBT_3`, `MVT − SCHED`, `AOBT_3` present at all
- `phi`, `phi_support` — `BLOCK = AOBT + φ(LOBT − AOBT)`, so a row with
  `BLOCK == SCHED == LOBT` is a **φ = 1 atom**. exp-007 refuted a *sharp*
  two-atom reading of φ and left the diffuse mixture open; this class is a
  candidate for what the φ = 1 side of it actually is, and the lane should say
  which way that goes.
- airport, operator prefix, stand, stand group, runway, hour, dow,
  minute-of-day, and the airport-day rate of the class
- seconds of `SCHED` are always `:00`, so they carry nothing; **not** included

## Arms

| arm | | tests |
|---|---|---|
| **0 control** | `TailModel` 2.4.0 unchanged | the delivered model |
| **P** | the mixture at the fitted `p` | the lane |
| **P-oracle** | the mixture at the true label | the bound, per fold |
| **P-hard** | argmax instead of the expectation | reproduces exp-018's negative 1 on our head, or refutes it |

`P-oracle` is not a candidate; it is the instrument that says what fraction of
the bound the detector realised, so a weak result is diagnosed rather than
merely recorded.

## Folds, seeds, gate

- Folds `july`, `january`, `july_loo`; seeds **7, 23, 51**; `rmse` and
  `rmse_clean` both, per fold, never pooled.
- **Gate** (desk): mean improvement on `january` **and** `july_loo`, each with a
  **day-clustered paired bootstrap 95 % interval clear of zero**, across the
  three seeds, and noac RMSE unmoved (< 0.01 s) on every fold. Clustering is by
  calendar day of take-off: placeholder rows are not independent within a day —
  the class is airport-and-day concentrated — so an unclustered interval would
  overstate significance, which is the error `exp-007 review-2` caught on the
  day-state result.
- Report realised / oracle per fold, and the classifier's calibration, not only
  its AUC — the p-rule pays on calibrated posteriors, not on ranking.

## The two exp-018 negatives, checked against our head

1. **The hard-label cliff.** Their `|y − ms| ≤ 6 s` label put rows with ratio
   1.05–1.19 on the "not placeholder" side and a class mean then dragged them
   from 87,000 s to 3,206 s. **Our noac head cannot fail this way in the band**:
   above 8 h the two-point mixture classifies by the *outcome* (`|y − 86,400| <
   3600`), not by a tolerance around `ms`, and their example rows (y ≈ 87,000)
   are rollovers under that definition and are predicted near `rollv`. Below the
   band our head is still anchored at `ms` and blends `ms − (1−p)·offset`, so no
   row can be dragged to a global class mean. **To be verified, not asserted**:
   count noac rows with `ratio ∈ [1.02, 1.3]` outside the band and report their
   error. Recorded as a check this lane owes.
2. **Widening the noac population is poison** (527 → 16,162 rows, 4,013 → 8,569
   noac RMSE). **We do not widen**: the noac head trains on `AIRCRAFT_TYPE_mvt`
   null only. This negative confirms the existing design and, more usefully,
   forbids the obvious shortcut here — exp-022 must not be built by feeding
   non-noac placeholder rows into the noac head. Hence a separate classifier and
   a separate anchor above.

## Cost

12 fits (4 arms × 3 folds) × 3 seeds = 36, but P-oracle and P-hard reuse arm P's
fitted trees and differ only at predict time, so the real cost is **18 fits**,
~170–320 s each at 6 threads under `docs/COMPUTE.md` compute allocation, one at a time:
roughly 1.5–2 hours detached.
