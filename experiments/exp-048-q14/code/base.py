"""The frozen base: `prc.tail_model`'s v4 noac HEAD, leave-one-month-out over 2025.

`prc.tail_model`'s noac branch is self-contained -- it reads only the noac rows'
own `NOAC_FEATURES_V2` and never the phi, window or bulk heads -- so a noac-only
harness runs the same estimator on these rows: the seed-averaged placeholder
classifier and non-placeholder offset regressor, then the v4 high-delay band
override, then the 60 s floor.  SPEC.md registers this as the base; nothing here
reads a 2026 label (none exists) or a board number.

**This is the head, NOT the served vector.**  The 388 ranking noac rows are
served with stack-5's bytes (`v7`), and `code/served.py` measures the distance:
1,328 s RMS, median |difference| 578 s, 1 row of 388 inside 1 s.  So this base
is the shipped noac ESTIMATOR fitted out of fold, not a reconstruction of the
served vector, and RESULTS section 2 carries that as the reading's transfer
caveat (FALLBACK review round 1, blocking 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from prc.tail_model import (NOAC_FEATURES_V2, NOAC_CATS_V2, NOAC_PARAMS, NOAC_SEEDS,
                            PLACEHOLDER_TOL, MIN_TAXI, BAND_MIN, BAND_EDGES,
                            BAND_SHRINK, BAND_TOP, DAY, TailModel)  # noqa: E402

SRC = ROOT / "experiments" / "exp-033-noac" / "work"
N_JOBS = 2
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def encode(d: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = d[list(NOAC_FEATURES_V2)].copy()
    for c in NOAC_CATS_V2:
        X[c] = pd.Categorical(d[c].astype(str), categories=cats[c])
    return X


def fit_path(fit: pd.DataFrame, cats: dict):
    """The shipped noac head on one training set: clf, non-placeholder reg, band."""
    X = encode(fit, cats)
    o = (fit.mvt_sched - fit.y_true).to_numpy(float)
    ph = (np.abs(o) < PLACEHOLDER_TOL).astype(int)
    seeded = lambda s: dict(NOAC_PARAMS, n_jobs=N_JOBS, seed=s, bagging_seed=s,
                            feature_fraction_seed=s, data_random_seed=s)
    clf = [lgb.LGBMClassifier(**dict(seeded(s), objective="binary")).fit(
        X, ph, categorical_feature=list(NOAC_CATS_V2)) for s in NOAC_SEEDS]
    reg = [lgb.LGBMRegressor(**seeded(s)).fit(
        X[ph == 0], o[ph == 0], categorical_feature=list(NOAC_CATS_V2))
        for s in NOAC_SEEDS] if (ph == 0).sum() >= 20 else []
    band = TailModel._fit_band(fit[["mvt_sched", "sched_date_cross"]],
                               fit.y_true.to_numpy(float))
    return clf, reg, band


def band_predict(band: dict, d: pd.DataFrame) -> np.ndarray:
    """`TailModel._band_predict`, verbatim in behaviour."""
    if not band:
        return np.full(len(d), np.nan)
    ms = d.mvt_sched.to_numpy(float)
    cross = d.sched_date_cross.to_numpy(float)
    b = np.digitize(ms / 3600.0, BAND_EDGES) - 1
    p = np.full(len(ms), np.nan)
    for i in range(len(ms)):
        if not np.isfinite(ms[i]) or ms[i] < BAND_MIN or ms[i] >= BAND_TOP:
            continue
        key = (int(b[i]), int(cross[i])) if np.isfinite(cross[i]) else None
        p[i] = band["cells"].get(key, band["bands"].get(int(b[i]), band["global"]))
    rollv = DAY + band["offset"]
    return np.where(np.isnan(p), np.nan, p * rollv + (1.0 - p) * ms)


def apply_path(path, d: pd.DataFrame, cats: dict) -> np.ndarray:
    clf, reg, band = path
    X = encode(d, cats)
    ms = d.mvt_sched.to_numpy(float)
    p = np.mean([c.predict_proba(X)[:, 1] for c in clf], axis=0)
    off = np.mean([r.predict(X) for r in reg], axis=0)
    pred = ms - (1.0 - p) * off
    bp = band_predict(band, d)
    ok = ~np.isnan(bp)
    pred[ok] = bp[ok]
    return np.clip(pred, MIN_TAXI, None)


def main() -> None:
    a = pd.read_parquet(SRC / "noac_2025_lab.parquet")
    b = pd.read_parquet(SRC / "noac_2026_d.parquet")
    cats = {c: pd.Index(sorted(set(a[c].astype(str)) | set(b[c].astype(str))))
            for c in NOAC_CATS_V2}

    base = pd.Series(np.nan, index=a.index, dtype=float)
    for m in MONTHS:
        va = a.month == m
        base[va] = apply_path(fit_path(a[~va], cats), a[va], cats)
        print(f"  {m}: {int(va.sum())} rows", flush=True)
    assert base.notna().all()
    out = a[["MVT_ID_mvt", "month", "day", "band", "y_true", "mvt_sched",
             "mvt_sched_log", "sched_date_cross", "minute_of_day", "dow",
             "stand_busy_frac_6h"]].copy()
    out["base"] = base.to_numpy()
    out.to_parquet(EXP / "work" / "base_lomo.parquet", index=False)

    r = out.y_true.to_numpy(float) - out.base.to_numpy(float)
    print(f"\nLOMO shipped v4 noac path, 1,507 rows: RMSE {np.sqrt((r ** 2).mean()):.1f} s")
    per = out.assign(r=r).groupby("month").apply(
        lambda g: pd.Series({"n": len(g), "rmse": float(np.sqrt((g.r ** 2).mean()))}),
        include_groups=False)
    print(per.round(1).to_string())
    per.to_csv(EXP / "analysis" / "base_lomo_per_month.csv")


if __name__ == "__main__":
    main()
