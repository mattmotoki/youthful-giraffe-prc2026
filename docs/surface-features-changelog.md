# `prc/surface_features.py` — behaviour changes

Team rule for shared modules: never change a default silently; bump
`VERSION`; keep the old behaviour reachable by argument; report the consumers.

**Consumers as of this entry**
- `experiments/exp-001-retro-v1/surface_step.py` — imports `FAMILY_COLUMNS`,
  `SurfaceFeatures`, `build_features`; uses all seven families and fits
  leave-one-month-out for training rows. Column set is unaffected (`queue2`
  lives in `EXTRA_FAMILY_COLUMNS`, not `FAMILY_COLUMNS`), **values are not**.
- `experiments/exp-004-congestion/` — the owning experiment.

## How to pin

```python
from prc.surface_features import SurfaceFeatures, VERSION

sf = SurfaceFeatures()                 # VERSION 3 defaults
sf = SurfaceFeatures.v2()              # what every exp-004 number used
sf = SurfaceFeatures.v1()              # the pre-2026-09-05 behaviour
sf = SurfaceFeatures(bucket=1800.0)    # or pin one constant
```

Every constant a produced value depends on is a constructor argument:
`bucket`, `cfg_now`, `cfg_prev`, `windows`, `nbr_window`, `unimpeded_q`,
`shrink_k`, `rolling_anchor`, `dominance`. `sf.version` reports which
generation an instance is configured as. A `SurfaceFeatures` pickled before
these fields existed unpickles as VERSION 1, because that is what it was
fitted and used under.

Verified when the arguments were introduced: with the defaults, the module
reproduces the VERSION 2 feature cache on july 2025-03, july 2025-07,
december 2025-12 and january 2025-01 with **0 of 112 columns differing**, and
`SurfaceFeatures.v1()` reproduces the VERSION 1 cache with **0 of 96
columns differing**. The refactor is parameter threading only.

## VERSION 3 (2026-09-05)

One default changed: `dominance` now defaults to `"exact"`, a genuinely exact
overtaking count. **What VERSION 2 called `"exact"` was not** — it used
`push_{i+s} < off_i` both as the scan's stopping rule and as a mask on the
hits, and the premise behind that (`off_j >= push_j`) fails on the 0.14 % of
departures whose take-off is stamped before their NM off-block. Measured
against brute force over all ten airports, July 2025:

| mode | `overtaken` wrong | `overtook` wrong | pairs missed |
|---|---|---|---|
| v2 `"push_window"` (was called `"exact"`) | 5,765 rows (3.02 %) | 197 (0.10 %) | 6,484 |
| v1 `"heuristic"` | 284 rows (0.15 %) | 1,435 (0.75 %) | 1,716 |
| **v3 `"exact"`** | **0** | **0** | **0** |

The v2 behaviour is reachable as `dominance="push_window"` or
`SurfaceFeatures.v2()`; it is a coherent definition ("overtaking pairs whose
push falls inside my taxi window"), it was simply not the definition the
docstring stated. Affects `overtaken` and `overtook` only.

**Consumer action.** `experiments/exp-001-retro-v1/surface_step.py` pins
`dominance='exact'`, and that string now means something different. To
reproduce numbers measured before this entry, pin `dominance='push_window'`;
to take the correction, keep the default and re-fit. Every number in
`experiments/exp-004-congestion/RESULTS.md` is VERSION 2.

## VERSION 2 (2026-09-05)

Three defaults changed. All three were defects, so v2 is the correct default,
but v1 stays reachable so a consumer can reproduce an older number.

| # | change | columns affected | measured impact (july 2025) |
|---|---|---|---|
| 1 | rolling neighbour statistics are evaluated over `(q − 1800, q)` anchored on the query take-off. v1 computed the rolling median on the event series and read it **at the last event before the query**, i.e. `(e − 1800, e]`, which after a quiet period returns a statistic from outside the window entirely. | `nbr_span_apt`, `nbr_span_rwy`, `nbr_taxiin_apt`, `nbr_taxiin_rwy`, `nbr_excess_apt`, `nbr_excess_rwy` | `nbr_span_apt` changed on **48.9 %** of rows, mean abs delta 21.9 s. `nbr_taxiin_rwy` populated **64.9 % → 17.6 %**: arrivals seldom land on a departure runway within 30 minutes, so 47 % of that column was a stale value dressed as a current one. |
| 2 | the overtaking scan stops on `push_{i+s} < off_i` instead of after 25 consecutive empty offsets (**superseded by VERSION 3 — this variant is not exact**) | `overtaken`, `overtook` | changed on **2.88 %** / 0.85 % of rows, max delta 394 |
| 3 | the runway-configuration window is a real hour (three 20-minute buckets); v1 used three 30-minute buckets = 90 minutes, and `cfg_changed` compared a 60-minute union with that 90-minute union | `cfg_dep`, `cfg_arr`, `n_dep_rwy`, `n_arr_rwy`, `cfg_changed`, `rwy_share` | `rwy_share` changed on **79.8 %** of rows, `n_dep_rwy` on 8.1 % |

A fourth change is a pure bug fix with no numerical effect on any correct
usage: `build_features` copied the taxi column before blanking departures.
`Series.to_numpy(dtype="float64")` returns a **view** when the column is
already float64 (as it is after `prc.data.mask_like_ranking`), so v1 wrote NaN
into the caller's frame. Callers that passed an unmasked frame — which is what
`surface_step.py` does — were never affected, because an unmasked training
month carries `TAXITIME_SEC_mvt` as int32 and the conversion copies.

**What a consumer must do:** regenerate any cached surface columns. Features
built before this entry are v1. `experiments/exp-004-congestion` keys its
feature cache on a hash of the module so a stale generation cannot be reused
silently; a consumer without such a guard should delete its cache by hand.

**Which value the exp-004 RESULTS tables used:** every table in
`experiments/exp-004-congestion/RESULTS.md` dated 2026-09-05 or later is
VERSION 2 — `BUCKET = 1200.0`, `cfg_prev = (-4, -3, -2)`,
`rolling_anchor = "query"`, `dominance = "exact"` — except the family ablation
tables still labelled generation 1, whose superseded copies were kept as
generated per-fold ablation records and generation-1 OOF files. The oracle
ceiling tables are VERSION 2 (the oracle feature itself
does not touch any of the changed constants, but the base model's features do).

**Effect on the congestion lane's own conclusions:** fix 1 reverses the sign of
the `flow` and `geometry` families on the july bulk (`flow` +11.1 → −2.2,
`geometry` +2.2 → −8.9 against a base of 317.4). Any statement resting on the
v1 numbers for those two families should be re-read.
