"""exp-018: re-derivations forced by review round 1. Durable address for every
number the review corrected, so the next reader can rerun rather than trust.

  PYTHONPATH=. .venv/bin/python experiments/exp-018-fresh/code/verify.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D                                    # noqa: E402
from prc import data                                   # noqa: E402

FOLDS = ["july", "july_loo", "december", "january"]


def sched_class():
    """Finding 1: exact equality versus the +/-6 s window."""
    ex = tot = 0
    win = 0
    noac_ex = noac_win = noac_soft = noac_n = 0
    for m in data.MONTHS:
        d = data.load_month(m); dep = d[d.PHASE_mvt == "DEP"]
        bs = (dep.BLOCK_TIME_UTC_mvt - dep.SCHED_TIME_UTC_mvt).dt.total_seconds()
        ex += int((bs == 0).sum()); win += int((bs.abs() <= 6).sum()); tot += len(dep)
        g = dep[dep.AIRCRAFT_TYPE_mvt.isna()]
        gbs = (g.BLOCK_TIME_UTC_mvt - g.SCHED_TIME_UTC_mvt).dt.total_seconds()
        ms = (g.MVT_TIME_UTC_mvt - g.SCHED_TIME_UTC_mvt).dt.total_seconds()
        y = g.TAXITIME_SEC_mvt.astype(float)
        noac_ex += int((gbs == 0).sum()); noac_win += int((gbs.abs() <= 6).sum())
        noac_soft += int(((y / ms > 0.5) & (ms > 0)).sum()); noac_n += len(g)
    print(f"[1] all departures n={tot}: BLOCK==SCHED exactly {ex} ({ex/tot*100:.2f} %), "
          f"|d|<=6 s {win} ({win/tot*100:.3f} %)")
    print(f"[1] noac n={noac_n}: exact {noac_ex/noac_n*100:.1f} %, |d|<=6 s "
          f"{noac_win/noac_n*100:.1f} %, ratio>0.5 {noac_soft/noac_n*100:.1f} %")


def segments():
    """Finding 5: the four required columns plus noac and bulk, every fold."""
    print("\n[5] fold | all | <=3600 | >3600 | no-NM | n | noac | n | bulk")
    for f in FOLDS:
        o = pd.read_parquet(f"experiments/exp-018-fresh/oof/{f}.parquet")
        t = D.month_table(data.FOLDS[f].valid_month)[["MVT_ID_mvt", "no_nm", "ghost"]]
        o = o.merge(t, on="MVT_ID_mvt")
        y, p = o.y_true.to_numpy(), o.y_pred.to_numpy()
        lo = y <= 3600; nm = o.no_nm.to_numpy() > .5; gh = o.ghost.to_numpy() > .5
        bulk = (~nm) & (~gh) & lo
        print(f"    {f:9s} {data.rmse(y,p):7.2f} {data.rmse(y[lo],p[lo]):7.2f} "
              f"{data.rmse(y[~lo],p[~lo]):8.2f} {data.rmse(y[nm],p[nm]):8.2f} {nm.sum():5d} "
              f"{data.rmse(y[gh],p[gh]):8.2f} {gh.sum():4d} {data.rmse(y[bulk],p[bulk]):7.2f}")


def guard():
    """Finding 15: does exp-006's parameter-free residual guard help here?

    Clip the implied r = (take-off - AOBT) - y_hat to the range of the same
    quantity over the fold's TRAINING departures, then re-derive y_hat.
    Only defined on rows with an NM join.
    """
    print("\n[15] fold | as shipped | guarded | rows moved | worst inflation on ordinary rows")
    for f in FOLDS:
        fold = data.FOLDS[f]
        tr = D.load(fold.train_months)
        ytr = tr.y.to_numpy(float); rtr = tr.r_aobt.to_numpy(float)
        ok = np.isfinite(ytr) & np.isfinite(rtr)
        lo_b, hi_b = np.nanmin((rtr - ytr)[ok]), np.nanmax((rtr - ytr)[ok])
        o = pd.read_parquet(f"experiments/exp-018-fresh/oof/{f}.parquet")
        v = D.month_table(fold.valid_month)[["MVT_ID_mvt", "r_aobt", "no_nm", "ghost"]]
        o = o.merge(v, on="MVT_ID_mvt")
        y, p, r = o.y_true.to_numpy(), o.y_pred.to_numpy(), o.r_aobt.to_numpy()
        has = np.isfinite(r)
        implied = np.where(has, r - p, np.nan)
        pg = p.copy()
        clipped = np.clip(implied, lo_b, hi_b)
        moved = has & (clipped != implied)
        pg[moved] = np.clip(r[moved] - clipped[moved], 30.0, None)
        ordinary = has & (~(o.ghost.to_numpy() > .5))
        worst = np.nanmax((p - r)[ordinary])
        print(f"    {f:9s} {data.rmse(y,p):7.2f}  {data.rmse(y,pg):7.2f}  {int(moved.sum()):5d}"
              f"   +{worst:8.0f} s   (train range [{lo_b:.0f}, {hi_b:.0f}])")
        del tr


if __name__ == "__main__":
    sched_class(); segments(); guard()
