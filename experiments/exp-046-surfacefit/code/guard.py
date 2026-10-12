"""The tail guard, frozen once and read from here by everything that uses it.

The desk, 2026-09-08 19:40 UTC: *"Freeze the guard now: quantile p99.5 of pbase - obs_taxi
on the fitting months, registered in code before the serve, not moved again."*

**What it is.**  `analysis/lirf-july.json` found that a tail row is an aircraft that
WAITED, not one that taxied slowly: in rows with `y > 3600 s` the correlation between the
observed ground run and the label collapses to 0.018 (LIRF july) and 0.013 (every other
airport), against 0.599 and 0.723 in the bulk, and the gap between the label and the
observation is 4,005 s where the bulk's is 714 s.  A ground run cannot see a stand hold or
a slot delay, so the arm is applying a bulk-fitted correction on rows whose input carries
no signal -- and a bigger model does it more confidently, which is the capacity gradient.

The tail is a LABEL and can never be a serving rule.  The shape that identifies those rows
is not: **the base predicts a long taxi while the observation shows a short run.**  That is
`pbase - obs_taxi`, and it is on every ranking row.

**Why this file exists.**  The threshold is a quantile of the FITTING months, so it is
label-free, but it is also a number that must not drift between the price and the serve.
Here it is frozen: `Q = 99.5`, chosen by climbing 95 / 99 / 99.5 / 99.8 / 99.9 / 99.95 to
the turnover and stopping (`analysis/tail-guard.json`), the two arms peaking at the same
quantile independently.  Nothing may move it afterwards; a different quantile is a
different arm with its own name.

**The two cuts are different on purpose.**  Scoring a fold, the quantile is taken over the
OTHER two folds -- the rows the arm was fitted on -- so no scored row informs its own
guard.  Serving, there is no scored fold, so it is taken over all three.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: frozen; a different value is a different arm and takes a different name
Q = 99.5
TAG = "guard995"
NAME = f"+{TAG}"


def gap(pbase: np.ndarray, obs_taxi) -> np.ndarray:
    """`pbase - obs_taxi`: how much longer the base thinks the taxi is than the run seen."""
    return (np.asarray(pbase, dtype="float64")
            - pd.to_numeric(pd.Series(obs_taxi), errors="coerce").to_numpy("float64"))


def cut(pan: dict, folds) -> float:
    """The quantile of the gap over the with-clock rows of the given fold months."""
    v = np.concatenate([gap(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi)
                        [pan[f].usable.to_numpy(bool)] for f in folds])
    return float(np.nanpercentile(v, Q))


def mask(pbase: np.ndarray, obs_taxi, c: float) -> np.ndarray:
    """True where the guard holds the row at the base."""
    g = gap(pbase, obs_taxi)
    return np.isfinite(g) & (g > c)
