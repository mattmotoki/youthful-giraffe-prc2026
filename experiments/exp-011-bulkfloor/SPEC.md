# exp-011-bulkfloor — SPEC

## My reading (and the readings I rejected)

**The population.** The bulk is departures with an NM join, an aircraft type, and
`TAXITIME_SEC_mvt <= 3600`. Best measured OOF on it: `SB-L2hub` 234.1 on
july_loo, `F1-phifull` 193.4 on december and 235.5 on july, `L1-philirf-guard`
198.3 on january (`analysis/oof_inventory.csv`, 48 OOF files scored). The desk's
"243" is close to the older lines; I report against the best available per fold
and say which.

**What a "floor" is, and what each instrument actually bounds.** Three
instruments, and they do not measure the same thing:

- **(a) near-duplicate pairs.** Pooled within-group variance of the taxi time
  over groups matched on observables gives `sigma^2` *given those observables*.
  Because the unmatched observables still vary inside a group, this is an
  **upper bound on the irreducible variance** — a model that conditions on more
  can go below it. It is not a floor; it is a ceiling on the floor. Reported as
  a ladder of ever tighter matches so the reader can see where it stops falling.
- **(b) binned residual variance of the best OOF.** Descriptive: it shows where
  the error is and whether it is heteroscedastic. It separates nothing on its
  own and is not quoted as a floor.
- **(c) seed ensemble.** With `K` seeds, `MSE_single - MSE_K = sigma_m^2 (1 -
  1/K)`, so `sigma_m^2` (pure estimation variance) and `MSE_inf = MSE_K -
  sigma_m^2 / K` are both identified. `MSE_inf` is the best this model family
  reaches with infinite seed averaging: it is `irreducible + bias^2`. **The
  difference between (a) and (c) is the bias term — what better features or a
  better learner could still take.** That difference is the destination value
  the desk asked for.

Rejected reading: quoting the near-duplicate sigma as "the floor". It is an
upper bound on the floor and a lower bound on nothing, and treating it as a
floor would understate what is left.

**Covariate shift.** A classifier separating validation-month rows from training
rows on the model's own features, density ratio `w = p/(1-p) * n_tr/n_va`
clipped, then the plain-tree baseline retrained with those weights and scored
with the day-clustered bootstrap of `experiments/exp-001-retro-v1/bootstrap.py`
(clusters are airport-days, because one disrupted morning moves hundreds of rows
together). The same classifier is then fitted for each ranking month, and the
weight distributions compared. Label-free throughout: the classifier never sees
the target.

**Scope.** No submission candidate. One Opus review round.
