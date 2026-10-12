"""The noac population is a THREE-component mixture, not two.

exp-002 modelled it as placeholder (block = schedule) versus everything else.
The decomposition in `work/decompose.py` shows the residual after an oracle p is
3,618.7 s, and `work/` shows **96.2 % of that sits in 14 rows** whose block time
is dated one calendar day early -- the date-rollover class of exp-002 finding 13,
which lives *inside* the noac population and is not a placeholder (block does not
equal schedule) and not ordinary (y is a day plus a taxi).

    mode 1  placeholder   y = MVT - SCHED                  729 rows, exact
    mode 2  ordinary      y = a real taxi                   764 rows, median 1,030 s
    mode 3  rollover      y = 86,400 + a real taxi           14 rows, median 87,304 s

so the RMSE-optimal prediction is the three-way expectation

    E[y] = p1*mvt_sched + p2*m_taxi + p3*(86400 + m_taxi)

Mode 3 is separable: 11 of the 14 have `MVT - SCHED >= 28,800 s`, a band that
contains 39 placeholder rows and **zero** ordinary rows, so within it the choice
is a two-point one between `mvt_sched` and `86,400 + taxi` at p ~ 0.22 -- large
enough for the expectation to pay by the p-rule (exp-002 finding 15).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent))
from threads import THREADS   # caps BLAS at import time; docs/COMPUTE.md compute allocation
import sys
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-009-noac2"))
from lomo import BASE, NEW, CATS_NEW, CATS_BASE, P, SEEDS, PH_TOL, enc

ART = D.ROOT / "artifacts" / "exp-009-noac2"
EXP = D.ROOT / "experiments" / "exp-009-noac2"
NR, NN = 344841, 388
DAY = 86400.0
ROLL_MIN = 60000.0        # a day-early block; 15,000 s also swept in two long
                          # taxis (25,018 and 31,268 s) whose targets then entered
                          # the taxi regressor at -61,382 and -55,132 s (Opus r1, B3)
FEATS, CATS = BASE + NEW, CATS_NEW


def modes(df):
    """0 = placeholder, 1 = ordinary, 2 = rollover."""
    o = df.mvt_sched.to_numpy(float) - df.y.to_numpy(float)
    ph = np.abs(o) < PH_TOL
    roll = (~ph) & (df.y.to_numpy(float) > ROLL_MIN)
    return np.where(ph, 0, np.where(roll, 2, 1))


def _avg(maker, X, y_, Xte, cats):
    return np.mean([maker(s).fit(X, y_, categorical_feature=cats).predict(Xte) for s in SEEDS], axis=0)


def _avg_proba(maker, X, y_, Xte, cats, ncls):
    out = np.zeros((len(Xte), ncls))
    for s in SEEDS:
        m = maker(s).fit(X, y_, categorical_feature=cats)
        pr = m.predict_proba(Xte)
        for j, c in enumerate(m.classes_):
            out[:, int(c)] += pr[:, j]
    return out / len(SEEDS)


def run_lomo(df, use_three=True, name="three-mode"):
    reg = lambda s: lgb.LGBMRegressor(**dict(P, seed=s, bagging_seed=s,
                                             feature_fraction_seed=s, data_random_seed=s))
    ys, ps, mons, pr3 = [], [], [], []
    for m in D.MONTHS:
        te, tr = df[df.month == m], df[df.month != m]
        if not len(te):
            continue
        lev = {c: pd.Index(sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))) for c in CATS}
        Xtr, Xte = enc(tr, FEATS, CATS, lev), enc(te, FEATS, CATS, lev)
        k = modes(tr)
        ms = te.mvt_sched.to_numpy(float)
        otr = tr.mvt_sched.to_numpy(float) - tr.y.to_numpy(float)
        if not use_three:                                   # exp-002's two-mode head
            pred = ms - _avg(reg, Xtr, otr, Xte, CATS)
            p2 = np.full(len(te), np.nan)
        else:
            cls = lambda s: lgb.LGBMClassifier(**dict(P, objective="multiclass", num_class=3,
                                                      seed=s, bagging_seed=s,
                                                      feature_fraction_seed=s, data_random_seed=s))
            p = _avg_proba(cls, Xtr, k, Xte, CATS, 3)
            p = p / p.sum(axis=1, keepdims=True)
            # E[taxi] fitted on the rows whose y IS a taxi: modes 1 and 2 with the
            # day removed, so the 14 rollover rows inform the taxi mean instead of
            # poisoning the non-placeholder regressor as they did in exp-002
            taxi_rows = k != 0
            t_y = np.where(k[taxi_rows] == 2, tr.y.to_numpy(float)[taxi_rows] - DAY,
                           tr.y.to_numpy(float)[taxi_rows])
            m_taxi = _avg(reg, Xtr[taxi_rows], t_y, Xte, CATS)
            pred = p[:, 0] * ms + p[:, 1] * m_taxi + p[:, 2] * (DAY + m_taxi)
            p2 = p[:, 2]
        ys.append(te.y.to_numpy(float)); ps.append(pred); pr3.append(p2)
        mons.append(np.repeat(m, len(te)))
    y, pr, mon, p3 = (np.concatenate(x) for x in (ys, ps, mons, pr3))
    se = (y - pr) ** 2
    per = {m: float(np.sqrt(se[mon == m].mean())) for m in np.unique(mon)}
    return dict(model=name, n=len(y), rmse=float(np.sqrt(se.mean())),
                ranking_contrib_s=float(np.sqrt(NN * se.mean() / NR)),
                worst_month=max(per.values())), per, pd.DataFrame(
                    {"month": mon, "y": y, "pred": pr, "p_rollover": p3})


if __name__ == "__main__":
    df = pd.read_parquet(ART / "noac_train.parquet")
    k = modes(df)
    print(f"{len(df)} noac rows: placeholder {(k==0).sum()}, ordinary {(k==1).sum()}, rollover {(k==2).sum()}\n")
    rows, preds = [], {}
    for three, nm in ((False, "exp-002 two-mode head (offset regressor)"), (True, "exp-009 three-mode expectation")):
        r, per, p = run_lomo(df, three, nm)
        rows.append(r); preds[nm] = (per, p)
        print(f'{nm:42s} rmse={r["rmse"]:7.1f}  contrib={r["ranking_contrib_s"]:5.1f} s  worst month={r["worst_month"]:.0f}', flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(EXP / "work" / "three_mode_lomo.csv", index=False)
    best = res.sort_values("rmse").iloc[0].model
    per, p = preds[best]
    p.to_parquet(EXP / "work" / "three_mode_pred.parquet", index=False)
    print(f"\nbest: {best}\nper-month RMSE:")
    print(pd.Series(per).round(0).to_string())
