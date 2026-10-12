# Q1 — What is Rome doing in July?

Scripts: `work/q1_regime.py`, `work/q1d_split.py`, `work/q1c_error.py`.
Data: LIRF rows cached from all twelve 2025 months and the live ranking file
(`work/cache_lirf.py` → `lirf_2025.parquet` 321,208 rows, `lirf_2026.parquet`
53,738 rows).

## 1. The 751 s dispersion is one row, not a regime

`docs/shift-2026.md` reports std of take-off − AOBT at LIRF = 751 s in July
2025 and 737 s in July 2026 against ~200 s in every other month, and reads it
as "Rome in July is its own regime in both years". That reading does not
survive a trim.

| set | n (NM rows) | std, all rows | rows with abs(mvt−aobt) > 3600 | std after dropping them |
|---|---|---|---|---|
| 2025-07 | 14,874 | **750.7** | 2 | **240.9** |
| 2026-07 | 15,562 | **737.1** | 3 | **248.2** |
| 2025-01 | 11,257 | 208.1 | 0 | 208.1 |
| 2026-01 | 10,954 | 218.5 | 2 | 196.1 |

Trimmed monthly std across 2025 runs 179 (March) to 241 (July); July is the
top of that range, 15–20 % above the winter months, not 3.6× above them.

The single row that carries the 2025 statistic is `AEZ146R`, LIRF, 2025-07-15,
stand 234, runway 25: `MVT − AOBT = 87,181 s`, `MVT − SCHED = 87,001 s`,
`AOBT − LOBT = −180 s`, target 87,002 s, and **`BLOCK − AOBT = +179 s`** — an
entirely ordinary block-to-off-block relation with every time column shifted a
day. It is one of the 14 LIRF members of the date-rollover class in
`exp-002-tail` finding 13. July 2026's analogue is visible in the ranking file
without labels: `ITY340`, 2026-07-17, stand 204, runway 16L, `MVT − AOBT =
86,340 s`. Because the rollover moves AOBT with the take-off, the residual
parameterisation (`init_score = MVT − AOBT`) predicts these rows correctly
already; they are a statistic problem, not a modelling problem.

**Consequence for the desk: any LIRF number quoted as a std or an RMSE of
take-off − AOBT is a one-row statistic and must be re-read trimmed.** The
`july` fold is still the right place to test LIRF machinery, but for the reason
in §2, not the reason in `shift-2026.md`.

## 2. The real July regime is in `r = BLOCK − AOBT`, and it is congestion

`r` is what the model actually fits (retro-v1 hands `MVT − AOBT` over as an
init_score and fits `r`). At LIRF it has a clean seasonal cycle:

| month | n | r median | r std | % rows r < −600 s | AOBT−LOBT median | % AOBT−LOBT > 600 | % > 1800 |
|---|---|---|---|---|---|---|---|
| 2025-01 | 11,257 | −59 | 418 | 8.4 | 120 | 20.0 | 0.9 |
| 2025-04 | 13,958 | −122 | 472 | 13.8 | 300 | 29.9 | 2.7 |
| 2025-06 | 14,511 | −176 | 559 | 15.8 | 480 | 41.2 | 6.1 |
| **2025-07** | 14,874 | **−179** | **688** | **19.9** | **600** | **48.3** | **10.2** |
| 2025-08 | 15,262 | −123 | 576 | 15.6 | 480 | 40.5 | 6.4 |
| 2025-12 | 11,980 | −59 | 431 | 9.5 | 180 | 22.5 | 1.2 |
| 2026-01 | 10,954 | — | — | — | 120 | 19.4 | 1.1 |
| 2026-07 | 15,562 | — | — | — | 480 | 41.7 | 7.3 |

(`r` is unobservable in the 2026 rows; `AOBT − LOBT` is observable in both.)

The driver is `AOBT_3 − LOBT`: Network Manager's actual off-block drifts later
and later behind the last filed off-block as the summer peak builds. The
airport's own block time does *not* follow it, so `r` goes negative.

**July 2026 is a slightly milder July 2025** (median 480 vs 600 s, 41.7 % vs
48.3 % over 600 s, 7.3 % vs 10.2 % over 1800 s — 2026-07 sits between June and
August 2025). **January 2026 is January 2025** (120 s median, 19.4 % vs 20.0 %).

## 3. It is the same mechanism at higher exposure, not a different mechanism

Conditioning on the observable `AOBT − LOBT`, July and January agree:

| AOBT−LOBT band | Jul n | Jul r median | Jan n | Jan r median | Jul P(block≈LOBT) | Jan P(block≈LOBT) |
|---|---|---|---|---|---|---|
| (−60, 60] | 854 | −56 | 1,072 | −56 | 41.7 % | 39.8 % |
| (300, 600] | 2,603 | −180 | 1,793 | −183 | 18.3 % | 22.9 % |
| (600, 1200] | 3,902 | −296 | 1,767 | −294 | 10.5 % | 10.8 % |
| (1200, 1800] | 1,776 | −478 | 384 | −417 | 5.9 % | 4.7 % |
| (1800, 3600] | 1,356 | −484 | 96 | −365 | 6.9 % | 4.2 % |
| > 3600 | 157 | −2,156 | 7 | −896 | 19.7 % | 14.3 % |

The conditional response transfers between seasons; only the *mix* over bands
moves. So the winter folds are informative about July at LIRF once the model is
conditioned on `AOBT − LOBT`, and a LIRF model does not need summer training
months as much as it needs that conditioning.

Take-off − AOBT is flat at ~920–1,015 s across every band: from NM's point of
view every one of these flights taxis normally. The whole phenomenon is a
disagreement between two clocks.

## 4. Not a stand group, a runway, an operator, or a set of days

- **Hour**: every hour of the day shows the July elevation (r std 315–912 s in
  July against 310–748 s in January, with July higher in 19 of 20 hours).
  The peak is 09:00–14:00 UTC, which is also the traffic peak.
- **Stand group**: r std rises monotonically with the apron number (2xx 565,
  3xx 669, 6xx 735, 7xx 778, 8xx 966, 9xx 1,165 in July) and does the same in
  January at a lower level (392, 357, 502, 517, 706, 719). The remote/high
  aprons are worse in both months; nothing switches on in July.
- **Runway**: 25 carries 91 % of departures; 16R (8 %) and 34L (1 %) both show
  the same July elevation. No configuration effect.
- **Market segment**: mainline and lowcost move together.
- **Operator**: the r std spread across the fifteen largest operators is 380 to
  1,050 s and tracks their apron and their median `AOBT − LOBT`, not an
  operator-specific convention.
- **Days**: broad. Daily median `AOBT − LOBT` in July 2025 is 420–780 s on 29
  of 31 days, with two disruption days on top — **2025-07-13** (median 1,560 s,
  42.5 % of rows over 1,800 s, r std 1,127) and **2025-07-28** (900 s, 23.7 %,
  r std 1,007). Removing those two days does not change the monthly picture.

## 5. A LIRF-only structural fact the other nine airports do not share

At LIRF, **no-NM join and null aircraft type are the same event**: 337 of 337
July-2025 no-NM rows are `noac`, and across all twelve months only one LIRF row
(April) is no-NM without being noac. At every other airport the two are
disjoint: EHAM 515 no-NM / 0 noac, LFPG 664 / 1, EDDF 435 / 0.

So the `noac` population — the one that carries 47 % of LIRF's July squared
error — is not "NM failed to join"; it is **Rome's own movement feed dropping
the whole record**, aircraft type and all. That is why the placeholder block
time (block = schedule) appears there and nowhere else, and it is the reason a
LIRF-specific treatment of these rows is worth building.

## 6. Where the baseline's LIRF error actually is

retro-v1 `c6-noac` out-of-fold, LIRF rows only:

| fold | LIRF RMSE | noac (n, RMSE, % of LIRF SSE) | NM & y>3600 | NM core y≤3600 |
|---|---|---|---|---|
| july | **864.6** | 337, 3,987, **47.1 %** | 271, 3,312, 26.1 % | 14,603, **456.5**, 26.8 % |
| december | **371.1** | 88, 2,411, 30.8 % | 48, 2,418, 16.9 % | 11,932, 270.0, 52.3 % |
| january | **387.1** | 60, 3,459, 42.3 % | 41, 2,101, 10.7 % | 11,216, 266.5, 47.0 % |

Three roughly equal thirds on July. Note the core: 456 s in July against 270 s
in December and 267 s in January at the same airport — the July core is the
largest single block of recoverable error at LIRF, and §2–3 say what it is.

## 7. The July 2026 side of the split (operator, NM status, day)

`work/q1e_july2026.py`. `AOBT − LOBT` and `take-off − AOBT` are observable in the
ranking file; `r` and the target are not, so this is the whole label-free split.

**NM status.** The LIRF identity of finding 3 holds in the ranking months too:
every LIRF ranking departure without an NM join also has a null aircraft type.

| month | departures | % noac | % no-NM | no-NM but not noac |
|---|---|---|---|---|
| 2025-07 | 15,211 | 2.216 | 2.216 | 0 |
| **2026-07** | 15,838 | **1.743** | 1.743 | **0** |
| 2025-01 | 11,317 | 0.530 | 0.530 | 0 |
| **2026-01** | 11,061 | **0.967** | 0.967 | **0** |

July 2026 is a *milder* noac month than July 2025 (1.74 % against 2.22 %) and
January 2026 a *heavier* one than January 2025 (0.97 % against 0.53 %).

**Operator.** For the twelve operators with the most July 2026 LIRF departures,
the median `AOBT − LOBT` is uniformly lower in 2026 than in 2025 (e.g. 660 → 480,
540 → 360, 600 → 420 s) but the *ordering across operators is preserved*: the
Spearman correlation of the operator medians between the two Julys is **0.954**.
`take-off − AOBT` medians agree to within a few seconds per operator (905 → 902,
896 → 895, 839 → 840). So July 2026 is July 2025 at a lower congestion level with
the same per-operator structure — which is what makes a per-operator `phi` fitted
on 2025 usable in 2026.

**Days.** July 2026's daily median `AOBT − LOBT` runs 300 to 720 s (median 480),
and **no day exceeds 1,500 s**. July 2025 ran 420 to 1,560 s with one day
(2025-07-13) above 1,500. The two disruption days of July 2025 have no
counterpart in July 2026; its busiest day, 2026-07-24, has 17.1 % of rows above
1,800 s against 42.5 % on 2025-07-13.
