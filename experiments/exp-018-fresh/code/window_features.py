"""K2: the transient-airport-window block, as the COMPLEMENT of what F1-fresh has.

Why a complement and not a family. `features.py` already carries a rich
neighbourhood description: backward rolling means of `MVT - AOBT_3` at the
airport, the runway and the stand (`r_prev10_apt`, `r_prev30_apt`,
`r_prev10_rwy`, `r_prev5_stand`), backward rolling means of the arrivals'
unmasked taxi-in (`arr_taxiin_last10/30`), forward and backward movement COUNTS,
previous/next gaps, day-level means, and the runway's share of the last hour.
The head built on it still leaves an airport x 10-minute shared error component
worth 16.04 s of no-noac clean board (`experiments/exp-027-step/RESULTS.md` s4),
so more of the same shape will not reach it. Four things F1-fresh does not have,
and this file supplies exactly those:

1. **Forward values.** `_rolling_prev_mean` is strictly backward; the file has
   forward COUNTS but never a forward VALUE. A departure's `MVT_TIME_UTC_mvt` and
   `AOBT_3_flt` are present for every row of the ranking file, so the realised
   `r` of departures AFTER this one is a legal input and is not used anywhere.
   This is the half the K2 slice measured as the valuable one: lead/lag with no
   shape (+1.162 s no-noac) beats shape with no lead/lag (+0.760).
2. **Dispersion.** Only means exist. A window whose neighbours agree is a
   different state from one whose neighbours disagree at the same mean.
3. **Position inside the window.** The row's excess over its own window and its
   rank in it: "everyone is slow" and "I am slow" are different states.
4. **Fixed-time windows and their trend.** Last-k-events is not a time window --
   at a quiet airport `r_prev30_apt` can span hours. A rising window is not the
   same state as a falling one.

Leakage. Every column is built from `MVT_TIME_UTC_mvt` and `AOBT_3_flt` of
departures and from `TAXITIME_SEC_mvt` / `BLOCK_TIME_UTC_mvt` of ARRIVALS, all of
which the ranking file carries unmasked (`docs/input-rule.md`). A departure's own
block time and taxi time are re-blanked defensively before anything is built, so
an upstream masking fault cannot become a leak here. One month at a time, always:
the ranking file holds January and July 2026 six months apart.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

#: Measured on the head's residual (`exp-027-step` s11): two scales reach three
#: scales' number at two thirds of the columns, one scale reaches 48 % of it.
SCALES = (600.0, 3600.0)
#: exp-035-k2climb arm 2 (widths). `PRC_WINDOW_SCALES="300,600,1800,3600"` puts
#: a finer and a mid scale beside the two the K2 member shipped with. Unset is
#: the shipped block, byte for byte; `dataset.py`'s month cache carries the set
#: in its key, because a stale flag-on table would report the shipped block's
#: score as the arm's.
_env = os.environ.get("PRC_WINDOW_SCALES", "").strip()
if _env:
    SCALES = tuple(sorted(float(x) for x in _env.replace(" ", ",").split(",") if x))
LAGS = (-2, -1, 1, 2)


def _agg(bucket: np.ndarray, apt: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """Per (airport, bucket) count/mean/sd/p10/p90 of `v`, NaNs ignored."""
    d = pd.DataFrame({"a": apt, "b": bucket, "v": v})
    g = d.groupby(["a", "b"], sort=True)["v"]
    out = g.agg(n="size", mean="mean", sd="std",
                p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
    return out.reset_index()


def build(dep_t: np.ndarray, dep_apt: np.ndarray, dep_r: np.ndarray,
          dep_rwy: np.ndarray, arr_t: np.ndarray, arr_apt: np.ndarray,
          arr_taxi: np.ndarray, scales=SCALES) -> pd.DataFrame:
    """`dep_r` is `MVT - AOBT_3` of OTHER departures as well as this one; it is an
    input everywhere in the ranking file, never a label."""
    n = len(dep_t)
    X = pd.DataFrame(index=np.arange(n))
    for S in scales:
        s = int(S)
        b = np.floor(dep_t / S)
        b = np.where(np.isfinite(b), b, -1e18).astype("int64")
        key = pd.DataFrame({"a": dep_apt, "b": b})

        # ---- departures: the window's realised surface speed and its shape
        src = _agg(b, dep_apt, dep_r)
        own = key.merge(src, on=["a", "b"], how="left")
        X[f"w{s}_n"] = own["n"].to_numpy("float32")
        X[f"w{s}_r_mean"] = own["mean"].to_numpy("float32")
        X[f"w{s}_r_sd"] = own["sd"].to_numpy("float32")
        X[f"w{s}_r_p10"] = own["p10"].to_numpy("float32")
        X[f"w{s}_r_p90"] = own["p90"].to_numpy("float32")
        X[f"w{s}_r_iqr"] = (own["p90"] - own["p10"]).to_numpy("float32")

        # position inside the window: "everyone is slow" vs "I am slow"
        X[f"w{s}_r_excess"] = (dep_r - own["mean"].to_numpy()).astype("float32")
        d = pd.DataFrame({"g": own["a"].astype(str) + "|" + own["b"].astype(str), "v": dep_r})
        X[f"w{s}_r_rank"] = d.groupby("g")["v"].rank(pct=True).to_numpy("float32")
        X[f"w{s}_since"] = (dep_t - b * S).astype("float32")

        # ---- lead and lag: the forward half is what F1-fresh has no value for
        for lag in LAGS:
            t = src.copy()
            t["b"] = t["b"] - lag
            j = key.merge(t, on=["a", "b"], how="left")
            X[f"w{s}_r_mean_l{lag}"] = j["mean"].to_numpy("float32")
            X[f"w{s}_n_l{lag}"] = j["n"].to_numpy("float32")
            if lag in (-1, 1):
                X[f"w{s}_r_p90_l{lag}"] = j["p90"].to_numpy("float32")

        # trend: rising is not the same state as falling
        X[f"w{s}_r_trend_back"] = X[f"w{s}_r_mean"] - X[f"w{s}_r_mean_l-1"]
        X[f"w{s}_r_trend_fwd"] = X[f"w{s}_r_mean_l1"] - X[f"w{s}_r_mean"]
        X[f"w{s}_r_curv"] = X[f"w{s}_r_trend_fwd"] - X[f"w{s}_r_trend_back"]
        X[f"w{s}_n_trend"] = X[f"w{s}_n"] - X[f"w{s}_n_l-1"]

        # ---- arrivals, on the same grid: unmasked taxi-in is a legal input
        ok = np.isfinite(arr_t) & np.isfinite(arr_taxi)
        ab = np.floor(arr_t[ok] / S)
        ab = np.where(np.isfinite(ab), ab, -1e18).astype("int64")
        asrc = _agg(ab, arr_apt[ok], arr_taxi[ok])
        aj = key.merge(asrc, on=["a", "b"], how="left")
        X[f"w{s}_at_n"] = aj["n"].to_numpy("float32")
        X[f"w{s}_at_mean"] = aj["mean"].to_numpy("float32")
        X[f"w{s}_at_p90"] = aj["p90"].to_numpy("float32")
        X[f"w{s}_at_sd"] = aj["sd"].to_numpy("float32")
        for lag in (-1, 1):
            t = asrc.copy(); t["b"] = t["b"] - lag
            j = key.merge(t, on=["a", "b"], how="left")
            X[f"w{s}_at_mean_l{lag}"] = j["mean"].to_numpy("float32")
        X[f"w{s}_at_trend"] = X[f"w{s}_at_mean"] - X[f"w{s}_at_mean_l-1"]

        # ---- runway configuration: the SET in use, and whether it changed
        rw = pd.DataFrame({"a": dep_apt, "b": b, "r": dep_rwy})
        nd = rw.groupby(["a", "b"])["r"].nunique().rename("nd").reset_index()
        X[f"w{s}_rwy_ndistinct"] = key.merge(nd, on=["a", "b"], how="left")["nd"].to_numpy("float32")
        mode = (rw.groupby(["a", "b"])["r"]
                  .agg(lambda v: v.value_counts().index[0] if len(v) else -1)
                  .rename("md").reset_index())
        md_now = key.merge(mode, on=["a", "b"], how="left")["md"].to_numpy()
        prev = mode.copy(); prev["b"] = prev["b"] + 1
        md_prev = key.merge(prev, on=["a", "b"], how="left")["md"].to_numpy()
        X[f"w{s}_rwy_mode_changed"] = (md_now != md_prev).astype("float32")
        X[f"w{s}_rwy_is_mode"] = (dep_rwy == md_now).astype("float32")
        share = rw.assign(hit=(rw["r"].to_numpy() == md_now).astype(float))
        X[f"w{s}_rwy_mode_share"] = (share.groupby(["a", "b"])["hit"].transform("mean")
                                     .to_numpy("float32"))
    return X


def from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the block from one month's raw movement frame, indexed like its DEP rows."""
    f = frame
    apt = np.where(f["PHASE_mvt"].to_numpy() == "DEP", f["ADEP_mvt"], f["ADES_mvt"])
    f = f.assign(_apt=apt)
    is_dep = (f["PHASE_mvt"] == "DEP").to_numpy()
    dep, arr = f[is_dep], f[~is_dep]

    def ep(s):
        v = pd.to_datetime(s, utc=True, errors="coerce").to_numpy("datetime64[ns]").astype("int64")
        return np.where(v == np.int64(-9223372036854775808), np.nan, v / 1e9)

    # defensive re-blanking: a departure's own block time and taxi time are the
    # label side and must not reach any aggregate, whatever upstream did.
    dep_mvt = ep(dep["MVT_TIME_UTC_mvt"])
    dep_aobt = ep(dep["AOBT_3_flt"])
    acode = pd.Categorical(f["_apt"]).codes
    rcode = pd.Categorical(f["_apt"].astype(str) + "|" + f["RUNWAY_mvt"].astype(str)).codes
    X = build(dep_mvt, acode[is_dep].astype("int64"), dep_mvt - dep_aobt,
              rcode[is_dep].astype("int64"), ep(arr["MVT_TIME_UTC_mvt"]),
              acode[~is_dep].astype("int64"),
              arr["TAXITIME_SEC_mvt"].to_numpy("float64"))
    X.index = dep.index
    return X
