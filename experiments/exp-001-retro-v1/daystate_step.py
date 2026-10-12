"""One column from prc.day_state_features: the de-icing share.

`ds_deice_hr` is the share of same-airport departures taking off within 1800 s of
this one whose `take-off − AOBT` exceeds 1800 s, with the row itself excluded.
exp-007 measures it at −13.6 s on december and −3.0 s on january at matched
rounds, and null on july — a winter signal, which is what the January half of the
ranking set needs. Their wider block harmed january by extrapolating, so only
this column is taken.

`DayState.fit` learns a reference distribution from the training months, so it is
fitted on those alone; the de-icing column itself is a same-month count of an
observable (`take-off − AOBT` is available for every ranking row), which is why
one bounded column can be added without the leave-one-month-out machinery the
label-derived families need. The module's own
`assert_no_departure_labels` is called on every frame as a check rather than a
promise.
"""

from __future__ import annotations

import pandas as pd

from prc.day_state_features import DEICE, DayState, assert_no_departure_labels

COLUMNS = tuple(DEICE)


def _one(raw: pd.DataFrame, ds: DayState) -> pd.DataFrame:
    # `transform` must see the frame as the ranking file gives it. Training months
    # arrive here with labels, so they are masked first; the module's own check
    # then confirms it rather than trusting the caller, and it fired the first
    # time this ran, which is what it is for.
    m = raw.copy()
    dep = m["PHASE_mvt"].eq("DEP").to_numpy()
    m.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    m["TAXITIME_SEC_mvt"] = m["TAXITIME_SEC_mvt"].astype("float64")
    m.loc[dep, "TAXITIME_SEC_mvt"] = float("nan")
    assert_no_departure_labels(m)
    out = ds.transform(m)
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    deprows = raw[raw["PHASE_mvt"].eq("DEP")]
    out = out.reindex(deprows.index)[list(COLUMNS)].copy()
    out.insert(0, "MVT_ID_mvt", deprows["MVT_ID_mvt"].to_numpy())
    return out.reset_index(drop=True)


def _attach(X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    j = out.set_index("MVT_ID_mvt").reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    return pd.concat([X, j], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]]):
    ds = DayState().fit(pd.concat([raw_by_month[m] for m in train_months], ignore_index=True))
    tr = pd.concat([_one(raw_by_month[m], ds) for m in train_months], ignore_index=True)
    return _attach(Xtr, tr), [_attach(X, _one(raw, ds)) for X, raw in others]
