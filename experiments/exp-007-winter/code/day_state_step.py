"""prc.day_state_features as a retro-v1 pipeline step.

Same shape as `experiments/exp-001-retro-v1/surface_step.py`: `fit_transform`
takes the training months' raw frames and the frames to serve, and returns the
feature matrices with the day-state columns appended.

Unlike `surface_step`, no leave-one-month-out construction is needed. `DayState.fit`
reads no label -- the reference distribution is built from timestamps, the NM
join flag and arrival taxi times -- so a training row's own month may take part
in the reference it is scored against without any path from its own target to
its own feature.
"""
from __future__ import annotations

import pandas as pd

from prc.day_state_features import COLUMNS, CORE, DEICE, DayState

PREFIX = ""          # the module's own names already carry ds_

SETS = {"all": list(COLUMNS), "core": list(CORE), "deice": list(DEICE)}


def _one(raw: pd.DataFrame, ds: DayState, cols) -> pd.DataFrame:
    f = ds.transform(raw)
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    out = f.reindex(dep.index)[list(cols)].copy()
    out.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].to_numpy())
    return out.reset_index(drop=True)


def _attach(X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    j = out.set_index("MVT_ID_mvt").reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    return pd.concat([X, j], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]], columns: str = "all"):
    cols = SETS[columns]
    months = list(train_months)
    ds = DayState().fit(pd.concat([raw_by_month[m] for m in months], ignore_index=True))
    tr = pd.concat([_one(raw_by_month[m], ds, cols) for m in months], ignore_index=True)
    return _attach(Xtr, tr), [_attach(X, _one(raw, ds, cols)) for X, raw in others]
