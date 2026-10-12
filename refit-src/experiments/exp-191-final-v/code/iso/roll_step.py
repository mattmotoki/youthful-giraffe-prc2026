"""ROLL: the day-rollover share of LIRF no-type rows with a very long schedule gap.

On these rows the label is either the placeholder point (y = MVT - SCHED = ms, block = SCHED) or the
rollover point (y = 86,400 s + a normal taxi: the real off-block time written on the scheduled date).
LIRF writes a block time on the scheduled or the take-off local date only, so a rollover needs a take-off
one Europe/Rome local date after the scheduled one ("cross" rows); on other rows ROLL changes nothing.
2025 cross rows: 0 rollovers below 50,000 s; 10 in 18 from 50,000 to 86,400 s.
p(rollover | ms) is a logistic fit on the no-type LIRF cross rows with 30,000 <= ms < 86,400 s of the
training months (held read: the ten non-held months; 2026: all twelve); the value is the expectation
p (86,400 + offset) + (1 - p) ms on the cross rows with ROLL_LO <= ms < 86,400 s.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import finalmode as FM

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "input-root/data/raw"
HELD_FILES = ("training_2025-01-01_2025-02-01", "training_2025-07-01_2025-08-01")
DAY, FIT_LO, ROLL_LO = 86400.0, 30000.0, 50000.0
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]


def _frame(paths, with_y: bool) -> pd.DataFrame:
    parts = []
    for p in paths:
        f = pd.read_parquet(p, columns=COLS + (["TAXITIME_SEC_mvt"] if with_y else []))
        f = f[(f.PHASE_mvt == "DEP") & (f.ADEP_mvt == "LIRF") & f.AIRCRAFT_TYPE_mvt.isna()].copy()
        f["file"] = Path(p).stem
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    f["ms"] = (f.MVT_TIME_UTC_mvt - f.SCHED_TIME_UTC_mvt).dt.total_seconds()
    loc = lambda s: s.dt.tz_convert("Europe/Rome").dt.normalize().dt.tz_localize(None)
    f["cross"] = (loc(f.MVT_TIME_UTC_mvt) - loc(f.SCHED_TIME_UTC_mvt)).dt.days == 1
    f["mvt_month"] = f.MVT_TIME_UTC_mvt.dt.month
    return f


def train_rows(exclude_held: bool) -> pd.DataFrame:
    t = _frame(sorted(glob.glob(str(RAW / "training_*.parquet"))), with_y=True)
    t = t[t.TAXITIME_SEC_mvt.notna() & t.cross & (t.ms >= FIT_LO) & (t.ms < DAY)].copy()
    if exclude_held:
        t = t[~t.file.isin(HELD_FILES) & ~t.mvt_month.isin([1, 7])]
    y = t.TAXITIME_SEC_mvt
    ph = (y - t.ms).abs() <= 60
    roll = ((y - DAY).abs() < 3600) & ~ph
    assert (roll | ph).all(), "a row in the fit band is neither placeholder nor rollover"
    return t.assign(y=y, roll=roll.astype(int))


def cross_rows(split: str, ids: np.ndarray) -> np.ndarray:
    """Per row of the split (in ids order): True for a LIRF no-type cross row."""
    paths = [FM.ranking_path(RAW)] if split == "ranking" else [RAW / f"{h}.parquet" for h in HELD_FILES]
    f = _frame(paths, with_y=False)
    s = pd.Series(f.cross.to_numpy(), index=f.MVT_ID_mvt.astype("int64").to_numpy())
    s = s[~s.index.duplicated()]
    return s.reindex(np.asarray(ids).astype("int64")).astype("boolean").fillna(False).to_numpy(bool)


def model(exclude_held: bool):
    t = train_rows(exclude_held)
    lr = LogisticRegression(C=1e4).fit((t[["ms"]] / 3600.0).to_numpy(), t.roll.to_numpy())
    off = float((t.y[t.roll == 1] - DAY).mean())
    return lr, off, t


def value(lr, off, ms):
    p = lr.predict_proba((np.asarray(ms, float) / 3600.0)[:, None])[:, 1]
    return p, p * (DAY + off) + (1 - p) * ms


if __name__ == "__main__":
    for ex in (True, False):
        lr, off, t = model(ex)
        grid = np.array([50000, 53500, 55000, 56500, 58000, 60000, 61500, 65000, 70000, 80000.0])
        p, _ = value(lr, off, grid)
        print("exclude_held" if ex else "all months", "rows", len(t), "rollovers", int(t.roll.sum()),
              "offset %.0f" % off, dict(zip(grid.astype(int).tolist(), p.round(3).tolist())))
