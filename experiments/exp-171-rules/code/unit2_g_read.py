"""Unit 2, step 2: the two rules of the pass under the shared G (prc/gscore.py), base B. No fit.

G weights each fold cell (airport x month x class) by its 2026 row count, so it differs from
the plain fold sum of the pass (LIRF July 2025 rows with a clock have a weight near 0).
Rules: S1 (route rate >= 0.98 from the ten months outside the fold, n >= 5, on rows late
>= 20 min or without a plan -> MVT - SCHED) and S3 (open-zone rows without a plan -> two-point
value with q = 0.33; practice route or same local date -> MVT - SCHED).

Run: .venv/bin/python experiments/exp-171-rules/code/unit2_g_read.py
"""
import os
import sys

import numpy as np
import pandas as pd

from unit2_lirf_cache import ROOT, Log, turn, vectors

sys.path.insert(0, ROOT)
from prc import gscore  # noqa: E402

say = Log(__file__)
B, _ = vectors()
m = turn()
T = m[m.year == 2025].copy()
T["fold"] = T.B.notna()
H = T[~T.fold & (T.late >= 900)]
kr = H.groupby(["al", "ADES_mvt"]).ph.agg(["size", "mean"])
kr.columns = ["rn", "rr"]
kf = H.groupby("FLIGHT_mvt").ph.agg(["size", "mean"])
kf.columns = ["fn", "fr"]
F = T[T.fold].join(kr, on=["al", "ADES_mvt"]).join(kf, on="FLIGHT_mvt")
late = ((F.late >= 1200) & F.present) | (~F.present & (F.ms >= 3600))
W, RP = 37080, 87400.0
off = np.where((F.sched >= 1743296400) & (F.sched < 1761440400), 7200, 3600)
same_day = ((F.mvt + off) // 86400) == ((F.sched + off) // 86400)
openz = (F.ms > W) & ((F.ms - 86400).abs() < W) & ~F.present
practice = (F.rr >= 0.9) & (F.rn >= 5)


def read(name, ids, values):
    cand = B.copy()
    cand.loc[ids] = np.maximum(values, 60.0)
    r = gscore.score(B.index.to_numpy(), B.to_numpy(), cand.to_numpy())
    keys = [k for k in r if not isinstance(r[k], (dict, list, pd.DataFrame, np.ndarray))]
    say(f"{name}: rows changed {len(ids)}; " + "; ".join(f"{k} {r[k]:.1f}" if isinstance(r[k], float) else f"{k} {r[k]}" for k in keys))
    for k in r:
        if isinstance(r[k], dict) and len(str(r[k])) < 1500:
            say(f"   {k}: {r[k]}")
    return r


s1 = F[(F.rr >= 0.98) & (F.rn >= 5) & late]
read("S1 route rule", s1.id.to_numpy(), s1.ms.to_numpy())
s1p = s1[s1.present]
read("S1 route rule, rows with a plan only", s1p.id.to_numpy(), s1p.ms.to_numpy())
c = F[(F.fr >= 0.98) & (F.fn >= 5) & late]
read("control: flight-string rule", c.id.to_numpy(), c.ms.to_numpy())
o = F[openz]
w = np.where(practice[openz] | same_day[openz.to_numpy()], 1.0, 0.33)
read("S3 open-zone two-point value, q = 0.33", o.id.to_numpy(), RP + w * (o.ms.to_numpy() - RP))
say.close()
