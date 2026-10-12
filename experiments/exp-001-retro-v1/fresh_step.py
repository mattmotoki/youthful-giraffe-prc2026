"""Three label-free in-month features from the fresh lineage (exp-018).

All three are functions of one month frame and of columns the ranking file
carries for every row, so none needs a fit and none needs leave-one-month-out
treatment:

``fr_stand_arr_taxiin_med``
    Median arrival taxi-in at the same stand within the month. A distance proxy
    that needs no labels at all — the taxi-in of an *arrival* is given, unlike a
    departure's taxi-out — so it estimates how far a stand is from the runway
    without touching the quantity being predicted.
``fr_roll_*``
    Causal rolling means of ``take-off − AOBT`` over the previous N departures
    at the airport (10, 30), the previous 10 on the runway and the previous 5 at
    the stand. Causal: each row sees only departures that took off before it.
``fr_queue_pushed_airborne``
    Departures that have gone off-block per NM and are not yet airborne at this
    take-off.

The queue count duplicates `prc.surface_features`' ``taxiing_at_off`` at
|r| = 0.898, measured on 2025-07 — below the frame's 0.98 dedup threshold but
close enough that it is reported rather than quietly added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BIG = 1 << 34
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _sec(s: pd.Series) -> np.ndarray:
    return (s - _EPOCH).dt.total_seconds().to_numpy(dtype="float64")


def _causal_roll(group: np.ndarray, t: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    """Mean of `v` over the previous `n` events of the same group, by time."""
    order = np.lexsort((t, group))
    g, vv = group[order], v[order]
    out = np.full(len(t), np.nan)
    acc: dict[int, list] = {}
    res = np.empty(len(t))
    for i in range(len(g)):
        buf = acc.setdefault(g[i], [])
        res[i] = np.mean(buf) if buf else np.nan
        if np.isfinite(vv[i]):
            buf.append(vv[i])
            if len(buf) > n:
                buf.pop(0)
    out[order] = res
    return out


def _one(raw: pd.DataFrame) -> pd.DataFrame:
    is_dep = raw["PHASE_mvt"].eq("DEP").to_numpy()
    dep = raw[is_dep]
    arr = raw[~is_dep]
    out = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()}, index=dep.index)

    # 1. stand distance proxy from ARRIVAL taxi-in, which is never blanked
    akey = (arr["ADES_mvt"].astype("string") + "|" + arr["STAND_mvt"].astype("string"))
    med = arr["TAXITIME_SEC_mvt"].astype("float64").groupby(akey).median()
    dkey = (dep["ADEP_mvt"].astype("string") + "|" + dep["STAND_mvt"].astype("string"))
    out["fr_stand_arr_taxiin_med"] = dkey.map(med).to_numpy(dtype="float64")

    t = _sec(dep["MVT_TIME_UTC_mvt"])
    span = t - _sec(dep["AOBT_3_flt"])
    apt = pd.factorize(dep["ADEP_mvt"].astype("string"))[0]
    rwy = pd.factorize(dkey.str.cat(dep["RUNWAY_mvt"].astype("string"), na_rep="NA"))[0]
    std = pd.factorize(dkey)[0]
    for name, grp, n in (("apt10", apt, 10), ("apt30", apt, 30), ("rwy10", rwy, 10), ("stand5", std, 5)):
        out[f"fr_roll_{name}"] = _causal_roll(grp, t, span, n)

    # 3. pushed per NM and not yet airborne at this take-off
    aobt = _sec(dep["AOBT_3_flt"])
    base = apt.astype("int64") * _BIG
    ok = np.isfinite(aobt)
    pushed_ref = np.sort((base[ok] + aobt[ok].astype("int64")))
    off_ref = np.sort((base[ok] + t[ok].astype("int64")))
    q = (np.searchsorted(pushed_ref, base + t.astype("int64"), "right")
         - np.searchsorted(off_ref, base + t.astype("int64"), "right"))
    out["fr_queue_pushed_airborne"] = q.astype("float64")
    return out.reset_index(drop=True)
