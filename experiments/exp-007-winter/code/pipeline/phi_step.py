"""phi features from prc.phi_features, fitted on training months only.

The lirf worker's PhiModel estimates where the airport's block time sits between
the NM actual off-block and the last filed off-block. Its `fit` takes labelled
month frames, so it must never see the validation month, and a training row must
not see its own month either — the encodings that failed earlier in this
experiment failed on exactly that kind of construction mismatch, so phi is
fitted leave-one-month-out for training rows and on every training month for the
validation and ranking rows.

The shared module is used rather than a local re-derivation so that phi means
the same thing in both experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc.phi_features import PhiCV, PhiModel

#: every PhiModel setting this experiment depends on, named rather than defaulted
PHI_SETTINGS = {"weight": "rows", "shrink_rows": 150.0, "min_rows": 100}

COLUMNS = ("phi", "phi_support", "taxi_hat", "gap")


def _attach(X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Join phi columns onto a feature frame by movement id."""
    o = out.set_index("MVT_ID_mvt")[list(COLUMNS)]
    j = o.reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    return pd.concat([X, j], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]],
                  mode: str = "cv", shrink: float | None = None):
    """Return (Xtr + phi columns, [X + phi columns for each other frame]).

    Both constructions come from `prc.phi_features.PhiCV` now, so the two arms
    differ in the construction and in nothing else:

    ``cv``    a training row in month *m* is transformed by the fit that excluded
              *m*, and a serving row by the average of the K fits, so both sides
              carry a coefficient estimated from K-1 months. No leak, matched.
    ``full``  every row transformed by the all-months fit. Matched, and on the
              training side an in-sample aggregate of the label; the module's
              row-count weighting caps a single row's share of its group near
              2 %, against the 36 % the gap-squared weighting allowed.

    The local leave-one-month-out-train / full-serve hybrid this file used before
    is gone: it was the one construction with a mismatch, and exp-006 measures
    that mismatch at 9.5 s on the six-month fold.
    """
    months = list(train_months)
    # Pinned, not defaulted. prc/phi_features.py's `weight` default changed under
    # the tail worker and moved their folds by about a second with no signal that
    # anything had changed, so every consumer names it. `shrink_rows` is the
    # parameter the row-count weighting actually reads; `shrink` is the gap-squared
    # one and is inert here, which is why --phi-shrink maps to shrink_rows.
    kw = dict(PHI_SETTINGS)
    if shrink is not None:
        kw["shrink_rows"] = float(shrink)
    base = PhiModel(**kw)
    cv = PhiCV(base).fit({m: raw_by_month[m] for m in months})
    tr = pd.concat([cv.transform_train(m, raw_by_month[m]) for m in months], ignore_index=True)
    serve = "avg" if mode == "cv" else "full"
    return _attach(Xtr, tr), [_attach(X, cv.transform_serve(raw, mode=serve)) for X, raw in others]
