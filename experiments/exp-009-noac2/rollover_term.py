"""it3: keep exp-002's head, add the rollover term only where it is separable.

The full three-class model (`three_mode.py`) LOSES: 4,371.7 s against 4,234.0 s.
Diagnosis: with 14 rollover rows in 1,507, leave-one-month-out leaves 12-13 in
training and 1-2 held out, so p3 cannot calibrate, and imposing a 3-class head
degrades p1 on the two modes that carry the rows. The oracle gain is real
(2,455 s) but it is not reachable by learning p3 as a class.

Repair: do not replace the head. Add the rollover term as a closed-form
correction *only inside the band where the class is separable* --
`MVT - SCHED >= T`, which on the training months contains placeholder and
rollover rows and no ordinary ones -- with p taken as the training-month
rollover share in that band, not learned per row:

    pred = (1 - p_band) * base + p_band * (86400 + m_taxi)

This is the same expectation, applied where the probability is estimable.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent))
from threads import THREADS   # caps BLAS at import time; docs/COMPUTE.md compute allocation
import sys
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-009-noac2"))
from lomo import BASE, NEW, CATS_NEW, P, SEEDS, PH_TOL, enc
from three_mode import modes, DAY, ROLL_MIN, FEATS, CATS

ART = D.ROOT / "artifacts" / "exp-009-noac2"
EXP = D.ROOT / "experiments" / "exp-009-noac2"
NR, NN = 344841, 388


def run(df, thresholds):
    reg = lambda s: lgb.LGBMRegressor(**dict(P, seed=s, bagging_seed=s,
                                             feature_fraction_seed=s, data_random_seed=s))
    ys, base_all, mons, ms_all, taxi_all, pband = [], [], [], [], [], {t: [] for t in thresholds}
    for m in D.MONTHS:
        te, tr = df[df.month == m], df[df.month != m]
        if not len(te):
            continue
        lev = {c: pd.Index(sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))) for c in CATS}
        Xtr, Xte = enc(tr, FEATS, CATS, lev), enc(te, FEATS, CATS, lev)
        otr = tr.mvt_sched.to_numpy(float) - tr.y.to_numpy(float)
        ms = te.mvt_sched.to_numpy(float)
        base = ms - np.mean([reg(s).fit(Xtr, otr, categorical_feature=CATS).predict(Xte) for s in SEEDS], axis=0)
        k = modes(tr)
        ty = np.where(k == 2, tr.y.to_numpy(float) - DAY, tr.y.to_numpy(float))
        taxi = float(np.median(ty[k != 0]))          # E[taxi], median: robust at n=14
        for t in thresholds:                          # p from TRAINING months only
            inb = tr.mvt_sched.to_numpy(float) >= t
            pband[t].append(float((k[inb] == 2).mean()) if inb.sum() >= 10 else 0.0)
        ys.append(te.y.to_numpy(float)); base_all.append(base); ms_all.append(ms)
        taxi_all.append(np.full(len(te), taxi)); mons.append(np.repeat(m, len(te)))
    y, base, ms, taxi, mon = (np.concatenate(x) for x in (ys, base_all, ms_all, taxi_all, mons))
    month_idx = {m: i for i, m in enumerate(sorted(set(mon)))}
    rows = []
    se0 = ((y - base) ** 2).mean()
    rows.append(dict(model="exp-002 head (base)", threshold=np.nan, p_band=np.nan,
                     rmse=np.sqrt(se0), ranking_contrib_s=np.sqrt(NN * se0 / NR)))
    for t in thresholds:
        p = np.array([pband[t][month_idx[m]] for m in mon])
        inb = ms >= t
        pred = base.copy()
        pred[inb] = (1 - p[inb]) * base[inb] + p[inb] * (DAY + taxi[inb])
        se = ((y - pred) ** 2).mean()
        rows.append(dict(model=f"+ rollover term, MVT-SCHED >= {int(t)}", threshold=t,
                         p_band=float(np.mean([pband[t][i] for i in range(len(pband[t]))])),
                         n_in_band=int(inb.sum()), rmse=np.sqrt(se),
                         ranking_contrib_s=np.sqrt(NN * se / NR)))
    return pd.DataFrame(rows), pd.DataFrame({"month": mon, "y": y, "base": base, "mvt_sched": ms})


if __name__ == "__main__":
    df = pd.read_parquet(ART / "noac_train.parquet")
    res, raw = run(df, [14400.0, 21600.0, 28800.0, 43200.0, 60000.0])
    res.to_csv(EXP / "work" / "rollover_term.csv", index=False)
    print(res.round(2).to_string(index=False))
    raw.to_parquet(EXP / "work" / "rollover_raw.parquet", index=False)
