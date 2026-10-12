# exp-020 cleanboard — SPEC

## My reading of the assignment, and the readings I rejected

**The reading I took.** The deliverable is a **scoring convention plus the
arithmetic that justifies it**, not a cleaning exercise. The january fold's
number is dominated by two rows of a class that most likely does not occur in the
ranking file at all, so every comparison drawn from it inherits an accident of
2025. The job is to identify that class by its mechanism, measure what it is
worth to every candidate, put the exclusion list somewhere every lane will use it,
and say what the corrected number does and does not license.

**Readings I rejected.**

1. *"Drop the rows and report the clean number."* Rejected. The clean number is
   not what the board will do to us — it is what our modelling is worth. Both are
   needed, which is why `rmse_both` returns them together and `rmse` is untouched.
2. *"Reuse `label_artefacts()['rollover_undetectable']`."* Rejected after
   measuring it: 12 of its 14 rows are reached by `MVT − SCHED`, the path
   `prc.tail_model`'s noac head v4 actually uses, and its window misses two
   rollover-class rows at 58 ks and 31 ks. It is a training-side tool and correct
   for that; the name does not transfer to scoring.
3. *"Define the class by a label threshold."* Rejected: the label has no gap at
   40,000 s. **And rejected in the other direction too** — my own first version
   used the date lag alone and caught 1,696 rows, 90 % of them ordinary
   midnight-crossing departures. Both conditions are needed and the joint table is
   printed rather than asserted.
4. *"Compare our clean estimate to the leaders' board scores."* Rejected as the
   headline, because it is not like-for-like — the same error `exp-007`'s review
   round 1 caught in this repo. The board number already contains the ranking
   file's own rollovers; the correction goes on OUR side.

## Frozen before any number was computed

- A row is **unreached** only if the block date is at least a day early, the
  label is abnormal, and **every** timing input (`SCHED`, `AOBT_3`, `LOBT`,
  `IOBT`, `EOBT_1`) is ordinary. Sensitivity to the abnormal cut is reported.
- The exclusion list is a **fixed list of MVT_IDs with reasons**, not a rule
  re-evaluated at scoring time: a rule would silently change what "clean" means
  when a model changes.
- The clean number never replaces the reported one; both are always shown.

## Compute

4 threads (`docs/COMPUTE.md` compute allocation), no fits, no GPU. Three passes over the twelve
labelled months and a read of committed OOF.
