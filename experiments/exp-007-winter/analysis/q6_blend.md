# exp-007 — the board blend on the (july_loo, january) analogue pair

`prc.data` VERSION 2 adds `july_loo`: the other eleven months against July 2025.
The submission model trains on all twelve, so July 2026 is predicted with July
and August 2025 in training; the forward `july` fold has no same-season data and
is pessimistic, exactly as `january` is the analogue for January 2026. So
`(july_loo, january)` is the fold pair that matches the ranking situation, and it
is the pair this file blends.

Every number comes from ONE line's out-of-fold files, so the folds share a model
configuration and a set of `prc/` module versions. The two unreachable
date-rollover rows (`analysis/q1_rollover.md`) are removed, because they are a
lottery every team draws from rather than a difference in skill; the floor they
put on a board score is finding 3's separate arithmetic.

Ranking composition: the other nine carry 176,284 July and 141,658 January
departures (55.4 % / 44.6 %); LIRF carries 15,838
and 11,061 (58.9 % / 41.1 %) and is
7.80 % of the file.

| line | July fold used | other nine | LIRF | implied total |
|---|---|---|---|---|
| `d1-base-s7` | `july` | **240.4** | 621.8 | 288.9 |
| `d1-base-s7` | `july_loo` **(the analogue pair)** | **234.6** | 611.9 | 282.8 |
| `f300-base-s7` | `july` | **238.0** | 651.8 | 292.1 |

The leaders' band, from exp-006's arithmetic at a board top of 277.3 s, is 200 to 240 s on the other nine against 550 to 720 s at LIRF.
