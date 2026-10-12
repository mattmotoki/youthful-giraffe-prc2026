"""Shared loading for exp-007-winter analysis.

Every function here may read labels: this file serves the *decomposition*
(question 1), not feature building. Anything that feeds a model lives in
`prc/day_state_features.py` and reads the masked frame only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import prc.data as D

FOLDS = ("january", "december", "july")
BASELINE = "G1cv-s7"
OOF_DIR = D.ROOT / "experiments" / "exp-001-retro-v1" / "oof"
TAIL_CUT = 3600.0


def _sec(a, b):
    return (a - b).dt.total_seconds()


def fold_rows(fold: str, line: str = BASELINE) -> pd.DataFrame:
    """One fold's scored departures joined to the baseline out-of-fold predictions.

    Columns: identity, the label-free time deltas, the label-derived ones, the
    prediction and its error. Rows are exactly the ones the model scored.
    """
    f = D.FOLDS[fold]
    full = D.load_month(f.valid_month)
    dep = full[full["PHASE_mvt"] == "DEP"].copy()
    dep = dep[dep[D.TARGET].notna()]

    oof = pd.read_parquet(OOF_DIR / f"{line}-{fold}.parquet")
    m = dep.merge(oof[[D.ID, "y_pred", "nm_missing"]], on=D.ID, how="inner", validate="1:1")

    m["apt"] = m["ADEP_mvt"].astype(str)
    m["day"] = m["MVT_TIME_UTC_mvt"].dt.floor("D")
    m["hour"] = m["MVT_TIME_UTC_mvt"].dt.hour
    m["y"] = m[D.TARGET].astype("float64")
    m["pred"] = m["y_pred"].astype("float64")
    m["err"] = m["y"] - m["pred"]
    m["se"] = m["err"] ** 2

    # label-free (survive the ranking mask)
    m["span"] = _sec(m["MVT_TIME_UTC_mvt"], m["AOBT_3_flt"])          # take-off - AOBT
    m["gap_al"] = _sec(m["AOBT_3_flt"], m["LOBT_flt"])                # AOBT - LOBT
    m["span_sched"] = _sec(m["MVT_TIME_UTC_mvt"], m["SCHED_TIME_UTC_mvt"])
    # label-derived
    m["r"] = _sec(m["BLOCK_TIME_UTC_mvt"], m["AOBT_3_flt"])           # block - AOBT
    m["b_lobt"] = _sec(m["BLOCK_TIME_UTC_mvt"], m["LOBT_flt"])
    m["b_sched"] = _sec(m["BLOCK_TIME_UTC_mvt"], m["SCHED_TIME_UTC_mvt"])

    m["no_nm"] = m["nm_missing"].astype(bool)
    m["noac"] = m["AIRCRAFT_TYPE_mvt"].isna()
    m["tail"] = m["y"] > TAIL_CUT
    m["regime"] = np.where(m["no_nm"], "no-NM", np.where(m["tail"], "NM tail", "NM bulk"))
    m["fold"] = fold
    return m


def worst_days(df: pd.DataFrame, frac: float = 0.05) -> pd.Index:
    """Airport-days holding the largest SSE, taking the worst `frac` of days."""
    g = df.groupby(["apt", "day"])["se"].sum().sort_values(ascending=False)
    k = max(1, int(round(frac * len(g))))
    return g.index[:k]


def rmse(x: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")
