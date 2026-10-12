"""Block-clock signals from prc.lirf_features.

The lirf worker measured these on this experiment's own code path: the global
model plus `prc.phi_features` plus `prc.lirf_features` scores july 330.7 and
december 218.4 against c6's 343.6 and 222.4, and the gain survives dropping the
twenty worst LIRF rows (`experiments/exp-006-lirf/work/eval_global.csv`). A
LIRF-only model loses, so this is a global feature import, not a per-airport
branch.

`lirf_signals` uses no labels — it blanks departure block time and taxi time in
its own working copy before computing anything — so unlike the surface model it
needs no leave-one-month-out treatment, and unlike phi it needs no fit at all.
Columns are prefixed `lf_` because two of them (`flight_len`, and the anchors
that `lirf_block` also produces) collide with names this experiment already
builds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.lirf_features import LIRF_CAT_FEATURES, lirf_signals

CAT_COLUMNS = tuple(f"lf_{c}" for c in LIRF_CAT_FEATURES)


def _one(raw: pd.DataFrame) -> pd.DataFrame:
    s = lirf_signals(raw)
    return s.rename(columns={c: f"lf_{c}" for c in s.columns if c != "MVT_ID_mvt"})


GATE_AIRPORT = "LIRF"


def _attach(X: pd.DataFrame, out: pd.DataFrame, gate: bool = True) -> pd.DataFrame:
    """Join the lirf columns, nulled outside LIRF when gated.

    Applied to all ten airports these columns cost 3.9 s of the july core while
    helping LIRF (exp-006's re-measurement under the fixed construction), which
    is almost exactly the +4.10 s july core regression measured here. They
    describe a Rome-specific recording convention, so outside Rome they are
    variance with no signal behind it. Gating nulls them elsewhere rather than
    dropping them, so the model keeps one feature set across airports and
    LightGBM reads the nulls as missing.
    """
    j = out.set_index("MVT_ID_mvt").reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    if gate:
        off = (X["airport"].to_numpy() != GATE_AIRPORT)
        for c in j.columns:
            j.loc[off, c] = pd.NA if c in CAT_COLUMNS else np.nan
    return pd.concat([X, j], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]], gate: bool = True):
    tr = pd.concat([_one(raw_by_month[m]) for m in train_months], ignore_index=True)
    return _attach(Xtr, tr, gate), [_attach(X, _one(raw), gate) for X, raw in others]
