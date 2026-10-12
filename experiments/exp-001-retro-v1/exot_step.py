"""A stand-runway deviation feature. The mechanism that motivated it is withdrawn.

exp-019 proposed that `AOBT_3` is derived as take-off minus the last received
EXOT (the airport A-CDM taxi estimate), making `MVT - AOBT_3` the airport's own
estimate and `r = BLOCK - AOBT_3` its error. **That derivation is not identified
by this data and is withdrawn.** The discriminating test — whether low-delay
rows' `MVT - AOBT_3` concentrates on a few whole-minute values per stand and
runway, as a table lookup would — fails at every airport (best LSZH 0.137 against
a 0.30 gate; EGLL and EHAM 0.000).

My own check of it was wrong, and in an instructive direction. I reported
"`AOBT_3` sits on a whole minute for 98.45 % of rows, confirmed". It is true and
it is **evidence against** the mechanism, not for it: if `AOBT_3` were
`take-off - EXOT` with a minute-resolution EXOT, it would inherit the take-off's
seconds, and the take-off is at second resolution (5.14 % on :00). A field that
is minute-aligned while its supposed minuend is not cannot be that difference. I
read a consistency check as a confirmation and reported it as one.

What survives is arithmetic, not mechanism: `r = BLOCK - AOBT_3` is whatever it
is, and these columns are a stand-runway deviation of `MVT - AOBT_3` from its
monthly median. That deviation correlated 0.427 with `r` against 0.377 for
`MVT - AOBT_3` itself on 2025-07, which is why the arm is still worth running —
but it now has to earn its gate on its own measured fold result, with no story
behind it.


"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAIR = 20


def _one(raw: pd.DataFrame) -> pd.DataFrame:
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    exot = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    key = (dep["ADEP_mvt"].astype("string") + "|" + dep["STAND_mvt"].astype("string")
           + "|" + dep["RUNWAY_mvt"].astype("string"))
    g = exot.groupby(key)
    med, n = g.transform("median"), g.transform("size")
    out = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()}, index=dep.index)
    out["ex_dev_sr"] = np.where(n >= MIN_PAIR, exot - med, np.nan)
    out["ex_sr_med"] = np.where(n >= MIN_PAIR, med, np.nan)
    out["ex_sr_n"] = n.to_numpy(dtype="float64")
    out["ex_mod300"] = exot.to_numpy() % 300.0
    out["ex_aobt_sec"] = dep["AOBT_3_flt"].dt.second.to_numpy(dtype="float64")
    out["ex_lobt_eq_sched"] = (dep["LOBT_flt"] == dep["SCHED_TIME_UTC_mvt"]).to_numpy().astype("float64")
    out["ex_lobt_iobt"] = (dep["LOBT_flt"] - dep["IOBT_flt"]).dt.total_seconds().to_numpy()
    return out.reset_index(drop=True)


def _attach(X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    j = out.set_index("MVT_ID_mvt").reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    return pd.concat([X, j], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]]):
    """No fit: every column is a within-month function of observable columns."""
    tr = pd.concat([_one(raw_by_month[m]) for m in train_months], ignore_index=True)
    return _attach(Xtr, tr), [_attach(X, _one(raw)) for X, raw in others]
