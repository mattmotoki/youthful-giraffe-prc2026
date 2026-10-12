"""Surface-state features from prc.surface_features, all seven families.

The congestion worker's first ablation split the families on whole-fold RMSE and
found four of them flipping sign between the july and december folds. Scored
instead on the NM bulk with a paired bootstrap, all seven improve both folds
(`experiments/exp-004-congestion/RESULTS.md`): the apparent flips were the
no-NM slice's 300-500 s confidence intervals moving the whole-fold number
around. On december's snow and freezing hours `flow` and `geometry` are worth
about 22 s each. So all seven are imported.

That changes the fitting discipline. `geometry` reads `SurfaceFeatures.fit`'s
unimpeded taxi quantiles per stand and runway, which are derived from the
training labels. (`flow`'s saturation columns also come from `fit`, but from
per-airport peak throughput, which is counts and timestamps rather than
departure targets.) Applied to a training row, a statistic that saw that row's own month is
the construction mismatch that cost this experiment 43 s when it appeared in the
target encodings. So the model is fitted leave-one-month-out for training rows
and on every training month for validation and ranking rows. The count-only
families do not need it and are unaffected by it.
"""

from __future__ import annotations

import pandas as pd

from prc.surface_features import FAMILY_COLUMNS, SurfaceFeatures, build_features

FAMILIES = tuple(FAMILY_COLUMNS)
COLUMNS = tuple(c for fam in FAMILIES for c in FAMILY_COLUMNS[fam])
#: the family whose columns come from SurfaceFeatures.fit's label-derived tables
LABEL_FITTED = ("geometry",)


def _one(raw: pd.DataFrame, sf: SurfaceFeatures) -> pd.DataFrame:
    """Surface columns for one month frame, keyed by movement id."""
    f = build_features(raw, sf)
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    out = f.reindex(dep.index)[list(COLUMNS)].copy()
    out.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].to_numpy())
    return out.reset_index(drop=True)


def _attach(X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    j = out.set_index("MVT_ID_mvt").reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    return pd.concat([X, j.add_prefix("sf_")], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]]):
    months = list(train_months)
    parts = []
    for m in months:
        rest = [raw_by_month[o] for o in months if o != m]
        sf = SurfaceFeatures().fit(pd.concat(rest if rest else [raw_by_month[m]], ignore_index=True))
        parts.append(_one(raw_by_month[m], sf))
    tr = pd.concat(parts, ignore_index=True)
    full = SurfaceFeatures().fit(pd.concat([raw_by_month[m] for m in months], ignore_index=True))
    return _attach(Xtr, tr), [_attach(X, _one(raw, full)) for X, raw in others]
