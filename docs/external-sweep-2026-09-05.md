# External research sweep — the AOBT mechanism, prior art, and what to run next

The exp-019 research used web search and direct source retrieval.
The team's external-research routine applies: **every proposal below
is mapped to the record that already measured it, or marked unmeasured**, and
every unmeasured one is reduced to a falsifiable local experiment with a cost and
a gate. No fits were run for this document.

## One paragraph

The sweep set out to identify **what `AOBT_3_flt` actually is**, because the
offset our models spend their capacity on — `BLOCK − AOBT`, which `exp-006`
named **phi** — is the whole task. EUROCONTROL's DPI Implementation Guide
documents a path that would make it an artefact of the airport's own taxi-time
estimate (EXOT), and the organizer confirms `AOBT_3_flt` is the off-block of a
Network Manager *model*, not an airport measurement. **The discriminating test
was run and came back negative**: `MVT − AOBT_3` does not behave like the
stand-and-runway lookup table that reading requires, at any of the ten airports,
and a seconds-based argument I initially offered for it turned out to be
circular. So the mechanism is documented but **not identified in this data**, the
experiments that depended on it are withdrawn, and what remains is one small
measured fact that was not in the ledger — `AOBT_3_flt` is minute-quantised while
the target's two clocks are not — plus two clean negatives on prior art.

## Sources

| # | source | what it establishes | changes anything? |
|---|---|---|---|
| S1 | [EUROCONTROL NM DPI Implementation Guide, Ed. 2.700, June 2025](https://www.eurocontrol.int/sites/default/files/2025-06/eurocontrol-dpi-impl-guide-2-700.pdf.pdf) | §4.7.3.3 "The aobt- and aobd-fields are **currently not used**"; §4.7.26.1 "the ttot-field shall supply ETFMS with an estimate of the actual take-off-time, **i.e., AOBT+EXOT**"; §4.7.26.4 "**Derived OBT in DPI msg is TOT – last-received-taxi-time**"; §4.7.21.1 taxitime "shall at least depend on **gate/stand/parking position and departure runway**", "**shall also include any time it takes to de-ice**"; §4.7.21.2 update only when it "changes by **5min or more**", widened to ±8 min when start-up delay > 90 min; §4.7.21.3 "value shall be between **1min and 90min**" | **Documents a mechanism; E1 could not confirm it applies to this data.** The linked guide is the source. |
| S2 | [PRC Data Challenge 2026 data page](https://prc-data-challenge-2026.netlify.app/data.html) | `AOBT_3_flt` = "Actual Off-Block Time (AOBT) for **flown (M3) trajectory**"; `_flt` = "Network Manager flight list", `_mvt` = "the reporting airport"; "The movements table rows have been (left) joined with the flight table ones"; "**Real world data is messy. You will find some inconsistencies between movement and flight information**" from "limitations of the underlying processes trying to match different sources" | **Yes.** The two clocks are two different systems by construction, and the organizer says so |
| S3 | [EUROCONTROL DDR2](https://www.eurocontrol.int/ddr) and the SO6/M1–M3 literature (e.g. [arXiv:1606.04241](https://arxiv.org/pdf/1606.04241), [arXiv:1603.02859](https://arxiv.org/pdf/1603.02859)) | M1 is the last-filed-flight-plan trajectory; **M3 is M1 updated with radar**, treated by NM as the flown trajectory | **Yes.** Consistent with AOBT_3 being minute-resolution and model-derived, which is measured; the specific derivation is not |
| S4 | Okwir et al., *J. Air Transport Management* 122 (2025) 102684, on CDM off-block delay prediction | Treats **`AXOT − EXOT`** (actual minus estimated taxi-out) as a first-class modelled feature alongside `AOBT−SOBT`, `AOBT−ASAT`; 162,955 flights; SHAP magnitudes −15 to +25 min | **Yes, weakly.** The literature models the airport's taxi-estimate error explicitly. Whether phi *is* that error is what E1 failed to establish |
| S5 | [PRC Data Challenge 2026 team pages](https://prc-data-challenge-2026.netlify.app/teams/outgoing-wizard.html) and the GitHub orgs | `PRC-Data-Challenge-2026` **does not exist** (GitHub API: `Not Found`); team pages carry registration only — no repo, method, or score. The 2025 org has 20 team repos but 2025 was **fuel burn** and 2024 was **take-off weight**; neither used the movement↔NM-flight join | **Yes, as a negative.** Nothing to copy and nothing to defend against yet |
| S6 | Taxi-out prediction literature: [NASA Charlotte](https://ntrs.nasa.gov/api/citations/20180002162/downloads/20180002162.pdf), [GNN spatiotemporal flow](https://doi.org/10.2514/1.D0369), [additional taxi-out with flow control](https://doi.org/10.3390/app14219968), [stochastic departure metering](https://doi.org/10.3390/aerospace13080684) | Take-off queue size is the dominant surface feature across the literature | **No.** Already measured here — `exp-004 final` and `exp-004 oracle` below |

## The mechanism — tested, and the strong reading is refuted

**E1 was run (2026-09-05, `experiments/exp-019-research/code/e1_exot_concentration.py`,
no fit, twelve labelled 2025 months, 2,062,577 NM-joined departures). Its frozen
gate FAILS, and so does the alternative fallback description. The honest position is that the derivation of `AOBT_3_flt` is not
identified by this data.**

### What was being tested

The tail analysis refuted the first derivation on seconds and was right:
`AOBT_3_flt` is on `:00` **97.7 %** of the time, so it cannot be the airport's
second-resolution take-off minus a whole-minute EXOT — that would inherit the
take-off's seconds. The reconciliation to test was that ETFMS uses its own
minute-resolution take-off:

```
AOBT_3 = ATOT_nm(minute) − EXOT(minute)              -> lands on :00
MVT − AOBT_3 = EXOT + (MVT − ATOT_nm)
```

Under that reading `round(MVT − AOBT_3)` to the minute **is the airport's EXOT
lookup value**, which DPI guide §4.7.21.1(3) says depends on stand and runway and
§4.7.21.2(2) says only moves in ≥5-minute steps — so it should take **few
distinct values per (stand-prefix, runway) cell**. The competing reading —
`AOBT_3` is a minute-rounded *reported* off-block — makes `MVT − AOBT_3` the true
taxi time plus rounding, no more concentrated than the taxi time itself.

### The gate fails

426 (airport, stand-prefix, runway) cells with ≥ 50 low-delay rows
(`|AOBT − LOBT| ≤ 300 s`, 881,845 rows, 42.8 % of the joined departures).
Gate, frozen before the numbers: *at least one airport shows ≥ 30 % of low-delay
rows on cells with ≤ 10 distinct whole-minute values.*

| airport | cells | rows | share of rows on cells with ≤10 distinct minutes | same for the **true taxi** control | median distinct, gap / taxi |
|---|---|---|---|---|---|
| LSZH | 41 | 44,463 | **0.137** | 0.004 | 7 / 17 |
| LEMD | 38 | 93,158 | 0.039 | 0.000 | 14 / 25 |
| LIRF | 23 | 62,807 | 0.036 | 0.000 | 9 / 30 |
| LTFM | 62 | 106,854 | 0.029 | 0.000 | 15 / 26 |
| LEBL | 23 | 74,560 | 0.025 | 0.000 | 13 / 25 |
| EDDF | 28 | 98,388 | 0.019 | 0.000 | 17 / 27 |
| EDDM | 29 | 75,655 | 0.019 | 0.000 | 18 / 23 |
| LFPG | 104 | 90,888 | 0.002 | 0.000 | 22 / 26 |
| EGLL | 15 | 115,033 | 0.000 | 0.000 | 58 / 42 |
| EHAM | 63 | 115,969 | 0.000 | 0.000 | 34 / 28 |

**Best airport LSZH at 0.137 against a bar of 0.30. The gate fails, and it is not
close.** `MVT − AOBT_3` does not behave like a small stand-and-runway lookup
table at any of the ten airports.

### What survives, and it is weak

On the same cells and the same rows, `MVT − AOBT_3` **is** more minute-concentrated
than the true taxi time:

| statistic, row-weighted over 426 cells | `MVT − AOBT_3` | true taxi (control) |
|---|---|---|
| distinct whole-minute values per cell | 34.1 | 38.6 |
| share on the modal minute | **0.320** | 0.143 |
| share on the top three minutes | **0.642** | 0.398 |

A 2.2× concentration on the modal minute is a real difference in the direction
the mechanism predicts — but it is nowhere near a lookup table, and it has a
mundane competing explanation: `MVT − AOBT_3` and the taxi differ by
`r = BLOCK − AOBT_3`, so the gap being *less* dispersed than the taxi simply
means `AOBT_3` tracks within-cell taxi variation better than `BLOCK` does. That
is what any off-block estimate would do. **It does not identify the derivation.**

### The seconds argument I made earlier is withdrawn

A first version of this document argued that `MVT − AOBT_3` clusters at whole
minutes (68 % within ±10 s) against a uniform prediction, and read that as
evidence for an NM-side derivation. **The control refutes it.** The airport's own
clocks are themselves whole-minute-preferring:

| quantity | share of seconds in 0–9 / 10–19 / 20–29 / 30–39 / 40–49 / 50–59 |
|---|---|
| seconds of `MVT_TIME_UTC_mvt` itself | **0.352** 0.081 0.080 0.080 0.079 **0.329** |
| seconds of `BLOCK_TIME_UTC_mvt` itself | **0.418** 0.048 0.049 0.048 0.048 **0.389** |
| `MVT − AOBT_3` | **0.352** 0.087 0.085 0.075 0.075 **0.327** |
| `BLOCK − AOBT_3` (the residual `r`) | **0.417** 0.049 0.049 0.049 0.049 **0.388** |
| `MVT − BLOCK` (the target) | 0.296 0.135 0.080 0.080 0.113 0.296 |

`MVT` is within ±10 s of a whole minute on **68.02 %** of rows and `MVT − AOBT_3`
on **67.82 %** — the same distribution. Because `AOBT_3` sits on `:00`, the gap
inherits `MVT`'s seconds exactly, and the clustering says nothing about `AOBT_3`
beyond its minute resolution, which was already known. The argument was
circular and is retracted.

### The correction, stated as what is and is not established

The fallback description was *"`AOBT_3` is a minute-rounded reported
off-block"*. **That is also unsupported**, because the evidence against
it was the seconds argument that just failed. What the data supports:

- **Established**: `AOBT_3_flt` is minute-resolution (97.7 %); the airport's own
  `MVT` and `BLOCK` clocks are second-resolution but strongly whole-minute-
  preferring (68 % and 81 % within ±10 s); the target inherits that (59 %).
- **Established**: `MVT − AOBT_3` is ~2.2× more concentrated on its modal minute
  per (stand, runway) cell than the true taxi is, and does **not** resemble a
  lookup table.
- **Not established, either way**: how `AOBT_3` is produced. The DPI guide (S1)
  documents a derivation path that would produce a minute-resolution,
  EXOT-dependent off-block, and that remains the best available account of the
  *institutional* mechanism; this data cannot confirm that ETFMS applied it to
  these flights, and the discriminating test I designed came back negative.
- **Withdrawn**: the claim that phi *is* the airport's EXOT error. It remains a
  plausible reading of `exp-006 phi` with a documented mechanism behind it, and
  it is no longer supported by anything measured here.

## What the ledger already measured

Mapped before proposing anything, per the protocol.

| mechanism class | ledger row | controlling measurement | status |
|---|---|---|---|
| the phi offset itself | `exp-006 phi` | `BLOCK−AOBT = phi·(LOBT−AOBT)`; airport phi 2025: EHAM −0.03, EDDF 0.06, LSZH 0.13, LEMD 0.15, LEBL 0.17, LFPG 0.20, EDDM 0.23 | **measured, delivered** |
| phi as a standalone predictor | `exp-006 phi-formula` | 376.3 (july, all ten) / 312.9 (december) fitted on training months only | measured |
| shrunk airport×operator phi | `exp-002 phi-screen` + `exp-002 correction` | rms(BLOCK−AOBT) 465.8 → 392.9 (july), 374.6 → 322.0 (dec), 377.0 → 324.7 (jan) | measured |
| phi construction (cross-fit vs full-fit, leave-one-row-out) | `exp-006 phi-v2`, `phi-construction-v2`, `review-2`, `loo-fix` | cross-fitted serving 309.86 vs full-fit 336.31; the direct pairing separates them on nothing | measured, and the naive reading was corrected twice |
| surface congestion / queue | `exp-004 final`, `exp-004 oracle` | 7 families worth −20.8 (july non-noac); a **perfect** oracle queue reaches only 269.3 / 193.2 | measured, with a ceiling |
| weather | `exp-005 final`, `exp-002 it12` | WINTER_CORE bulk −2.72 (july) / −2.05 (dec, jan −4.42); ERA5 on the tail model not adopted | measured, small |
| de-icing / day state | `exp-007` findings 13–14, `exp-012` deice adoption | the 25-column block **harms** january's disrupted days; the one bounded column `ds_deice_hr` is worth 51 s there | measured, and the repair shipped |
| `noac` placeholder block times | `exp-002 anatomy`, `exp-009 it2` | block time = SCHED placeholder on ~48 % of noac rows; 388 ranking rows | measured, delivered |
| the 2026 shift and January exposure | `exp-014 exposure`, `analogue`, `remix` | 19,895 January rows outside the day-state training range; blend biased **−83 s** on days past the envelope; board estimate ~5.8 s optimistic | measured |
| stand→runway taxi distance | `exp-003 final`, `exp-014 lfpg` | 18 stand-neighbour columns in the 64-column bulk set; LFPG's retirement creates only 48 unseen (stand, runway) ranking rows | measured |

**Nothing in the ledger names the EXOT mechanism**, and nothing measures the
minute quantisation of `AOBT_3_flt`. Every other question in the research plan
has a receipt above.

## Ranked experiments

**After E1 the list is short.** Five of the eight depended on the EXOT derivation
E1 failed to establish and are withdrawn or demoted below, each with the revival
condition that would bring it back. What remains standing is **E3′** (0.5 h, no
fit, answers a question `exp-007` and `exp-012` left open), **E6** (2 h, external,
low expectation, high payoff), and **E8** (zero compute, an organizer question).
Nothing here proposes a fit.

Costs are 4-thread hours under the team's compute split.

### E1 — Is `MVT − AOBT` the airport's EXOT? — **RUN, GATE FAILED**
**Result**: see the mechanism section. 426 cells, 881,845 low-delay rows, best
airport LSZH at 0.137 against a frozen bar of 0.30. `MVT − AOBT_3` is 2.2× more
concentrated on its modal minute per cell than the true taxi is, which is real
but far short of a lookup table and has a mundane alternative explanation.
**Cost paid**: 0.4 h, no fit. **Consequence**: E2–E5 are withdrawn below, per the
team rule fixed before the test was run.

### E2 — An explicit EXOT-error target — **WITHDRAWN**
Depended on E1 identifying a recoverable EXOT table. It did not. The feature it
proposed — per-(airport, stand-prefix, runway) medians of `MVT − AOBT` — is in any
case a re-encoding of signal `exp-003 final`'s 18 stand-neighbour columns and
`exp-006`'s phi already carry, which is why its gate was set at 1.5 s. **Revival
condition**: a ground-truth EXOT table from E6, or an organizer answer (E8)
confirming the derivation.

### E3 — De-icing enters through EXOT, not the weather — **WITHDRAWN as posed**
The mechanism it tested requires the EXOT derivation E1 failed to establish.
**But the underlying question is still open and cheap**, and it does not need the
mechanism: does `r = BLOCK − AOBT` move with `ds_deice_hr` *within* airport-day?
That is a fact about the data either way, and `exp-007` finding 14 plus
`exp-012`'s deice adoption make it decision-relevant — they measured the bounded
de-icing column at 51 s on january's disrupted days without establishing whether
it works through the anchor or through the physics. **Kept as E3′, 0.5 h, no fit**,
with the gate unchanged (consistent sign across three of EDDF, EHAM, EDDM, LSZH)
and the interpretation weakened: a confirmed sign no longer implies EXOT.

### E4 — The 5-minute EXOT update granularity as a staleness feature — **WITHDRAWN**
Its premise was that EXOT is stale by construction and that `MVT − AOBT` exposes
it. E1 says `MVT − AOBT` does not expose EXOT. **Revival condition**: same as E2.

### E5 — Split the anchor by whether NM saw a DPI — **WITHDRAWN**
Its premise was that the derived-OBT path exists only where DPIs arrive. Without
E1 the proxy (`AOBT == LOBT` exactly) is no longer tied to a mechanism, and it
would be a bare re-cut of `prc.blend`'s existing `nm_late` split, which
`exp-012` already measured a specialist inside at 0.43 s. **Disfavoured rather
than banned**: if E6 or E8 identifies which airports send A-DPI, the split
becomes mechanism-backed and is worth 1.5 h.

### E6 — Per-airport EXOT conventions from the A-CDM implementation docs *(unmeasured, external)*
**Do**: the sweep found the *general* rule but no per-airport EXOT convention for
EDDF, EGLL, EHAM, LFPG, LIRF or LTFM. Each A-CDM airport publishes a local
implementation manual (Frankfurt, Heathrow, Schiphol and CDG all have public
A-CDM pages). Fetch them and extract the stand-group → EXOT tables if published.
**Cost**: 2 h, no compute. **Gate**: at least two airports publish a table
granular enough to join on stand prefix. **Honest expectation**: low — these are
usually operational annexes, not public — but the payoff is a *ground-truth*
EXOT table, which would make E1 and E2 unnecessary by giving the answer directly.
**Note**: an external table is reference data, so `docs/external-data-rules.md`
governs whether it may enter a prize-eligible model; it may be used to *validate*
E1 regardless.

### E7 — Quantisation-aware anchoring — **withdrawn as posed, and one line is worth keeping**
The `+30 s` half is dead on arrival: a constant shift is absorbed by any model
with an intercept. The "sub-minute part of the take-off as a feature" half is
also weaker than it looked, because E1's control shows the airport's own clocks
are whole-minute-preferring — `MVT` is within ±10 s of a whole minute on 68 % of
rows and `BLOCK` on 81 % — so the sub-minute part is mostly a spike at zero
rather than a uniform nuisance. **The line worth keeping**: the *target* inherits
that preference (59 % of `MVT − BLOCK` within ±10 s of a whole minute against
33 % under uniformity), and **this is not exploitable under RMSE** — snapping
predictions to a minute grid raises squared error unless the target sits exactly
on the grid, and it does not. Recorded so nobody spends an afternoon on it.

### E8 — Ask the organizer how `AOBT_3_flt` is produced *(unmeasured; not mine to send)*
**Do**: one question — is `AOBT_3_flt` the DPI-derived OBT (`TOT − EXOT`), an
airport-reported AOBT passed through, or something else? S1 §4.7.3.3 says the
transmitted field is unused by ETFMS, which makes the second option unlikely, but
the M3 model's provenance is not public at this granularity.
**Cost**: zero compute. **Gate**: none — an answer either confirms the mechanism
or replaces it. **Team rule**: ask the organizer together with the question
about use of supplied inputs.

## What I did not find

- **No 2026 team has published anything.** The `PRC-Data-Challenge-2026` GitHub
  organization returns `Not Found`; team pages carry a name, a country and a
  one-line self-description. There is no early-published solution to learn from
  or defend against.
- **The 2024 and 2025 repos do not help.** 2024 predicted take-off weight and
  2025 predicted fuel burn; neither edition joined an airport movements table to
  the NM flight list, so neither has NM-timestamp-quirk handling for this shape
  of data.
- **No per-airport EXOT convention** for any of the ten airports, in any public
  source I reached. E6 is the follow-up and its expectation is low.
- **No published taxi-out study on the airport-block versus NM-AOBT gap.** S4 is
  the closest — it models `AXOT − EXOT` as a feature — and it is about off-block
  *delay*, not taxi-out, on a single airport's CDM feed.

## Reading this document

The protocol says to treat every response as a proposal and to verify locally
where a practical check exists. That is what happened, and it went against the
proposal twice.

- **S1 is a specification of what implementers are told to do**, not evidence
  about what any airport's system did in 2025. Its strongest local prediction was
  tested (E1) and **failed at every one of the ten airports**.
- **The argument I built for it was circular.** I read a whole-minute clustering
  in `MVT − AOBT_3` as evidence of an NM-side derivation; the control shows
  `MVT`'s own seconds carry that clustering, so the statistic was uninformative
  about `AOBT_3`. The proposed fallback description — "a minute-rounded
  reported off-block" — is **also** unsupported, because the only evidence
  against it was that same retracted argument.
- **What is left is worth having anyway**: one measured fact absent from the
  ledger (`AOBT_3_flt` is minute-quantised, the target's clocks are not, and the
  airport clocks are whole-minute-preferring), a documented institutional
  mechanism that remains the best available account even though this data cannot
  confirm it, two prior-art negatives with receipts, and three experiments that
  do not depend on any of it.

The proposed modelling interpretation — *phi is the airport's
EXOT error* — **should not be routed as established.** It is a plausible reading
with a published mechanism behind it and no local confirmation, and the one test
designed to confirm it came back negative.
