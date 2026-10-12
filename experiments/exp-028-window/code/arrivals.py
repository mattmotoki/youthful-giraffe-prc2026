"""The arrivals' unmasked surface state, which no committed out-of-fold file carries.

`assignments/window.md` item 2 asks whether the arrivals' `TAXITIME_SEC_mvt` and
`BLOCK_TIME_UTC_mvt` window aggregates move the arm, "available in the OOF months
only if the committed OOF carries them".  **They do not**: every out-of-fold file
in this repository holds departure rows and a handful of derived gate columns, so
until `data/raw` existed this block could not be built at all.  It exists now, so
it is built here from the raw movement files, on both sides of the fit/serve line
through one function.

**Why these columns and not the departures'.**  A departure's block and taxi time
are blanked in the ranking file, for every departure and not only the row being
predicted, so no departure-side aggregate of them is legal (`SPEC.md` §4).  An
arrival's are present and complete — 100 % populated on all 344,693 ranking
arrivals and on all three fold months — and `docs/input-rule.md` states arrival
rows carry full actuals including taxi-in time.  An arrival that took twelve
minutes to reach its stand is a direct measurement of the surface the departure
is about to cross, which is exactly the transient the oracle prices.

The window is the same one the departure block uses: the airport, the departure's
own `win`-second bucket of take-off time, and the two buckets either side.  An
arrival is placed in the bucket of its **landing** time (`MVT_TIME_UTC_mvt`),
which is when it entered the surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as Fe  # noqa: E402
import panel as P  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402

#: Raw columns the arrival block needs.  All are present in `ranking.parquet` and
#: none is blanked there — the blanking is departure-side only.
RAW_COLS = ["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
#: The fold month whose arrivals belong to each fold's departures.
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}


def arrival_table(mvt: pd.DataFrame) -> pd.DataFrame:
    """(airport, landing time, taxi-in, lateness, on-block time) for arrival rows."""
    a = mvt[mvt["PHASE_mvt"] == "ARR"]
    t = (a["MVT_TIME_UTC_mvt"] - EPOCH).dt.total_seconds().to_numpy("float64")
    return pd.DataFrame({
        "airport": a["ADES_mvt"].astype(str).to_numpy(),
        "t": t,
        "a_taxi": a["TAXITIME_SEC_mvt"].to_numpy("float64"),
        "a_late": (a["MVT_TIME_UTC_mvt"] - a["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
                  .to_numpy("float64"),
        "a_block": (a["BLOCK_TIME_UTC_mvt"] - EPOCH).dt.total_seconds().to_numpy("float64") - t,
    })


def block_is_taxi(mvt: pd.DataFrame) -> dict:
    """The assignment asks for the arrivals' `TAXITIME_SEC_mvt` AND their
    `BLOCK_TIME_UTC_mvt`.  On arrival rows those are the same quantity: on-block
    minus landing IS taxi-in, exactly, on every row of every month and of the
    ranking file (max |difference| 0.000 s over 190,448 july and 344,693 ranking
    arrivals).  So `a_block` is dropped from the aggregates below, and the
    taxi-in block answers both halves of the ask.  Sol round 1, Spec 5, closed by
    measurement rather than by a second fit of a duplicate column.
    """
    a = arrival_table(mvt)
    d = (a["a_block"] - a["a_taxi"]).to_numpy("float64")
    return dict(n=int(len(a)), max_abs_diff=float(np.nanmax(np.abs(d))),
                n_over_half_second=int((np.abs(d) > 0.5).sum()),
                n_non_finite=int((~np.isfinite(d)).sum()))


def arrival_features(dep: pd.DataFrame, arr: pd.DataFrame, win: float = Fe.WIN,
                     lags=Fe.LAGS) -> pd.DataFrame:
    """Arrival aggregates over the departure's airport-window.  Rows align with `dep`."""
    a = arr.copy()
    a["bucket"] = np.floor(a["t"] / win).astype("int64")
    g = a.groupby(["airport", "bucket"], sort=True)
    src = g.agg(a_n=("a_taxi", "size"),
                a_taxi_mean=("a_taxi", "mean"), a_taxi_med=("a_taxi", "median"),
                a_taxi_sd=("a_taxi", "std"),
                a_late_mean=("a_late", "mean"))
    q = g[["a_taxi", "a_late"]].quantile(0.9)
    src["a_taxi_p90"] = q["a_taxi"]
    src["a_late_p90"] = q["a_late"]
    src = src[["a_n", "a_taxi_mean", "a_taxi_med", "a_taxi_sd", "a_taxi_p90",
               "a_late_mean", "a_late_p90"]].reset_index()

    key = pd.DataFrame({"airport": dep["airport"].astype(str).to_numpy(),
                        "bucket": np.floor(dep["t"].to_numpy("float64") / win).astype("int64")})
    out = key.merge(src, on=["airport", "bucket"], how="left")
    for lag in lags:
        s = src.copy()
        s["bucket"] = s["bucket"] - lag
        s = s.rename(columns={c: f"{c}_l{lag}" for c in s.columns if c.startswith("a_")})
        out = out.merge(s, on=["airport", "bucket"], how="left")
    return out.drop(columns=["airport", "bucket"]).astype("float64").reset_index(drop=True)


def for_fold(fold: str, dep: pd.DataFrame, win: float = Fe.WIN) -> pd.DataFrame:
    """The arrival block for one fold, from that fold's validation month file."""
    m = pd.read_parquet(D.month_file(FOLD_MONTH[fold]), columns=RAW_COLS)
    for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
        if not pd.api.types.is_datetime64_any_dtype(m[c]):
            m[c] = pd.to_datetime(m[c], utc=True)
        elif m[c].dt.tz is None:
            m[c] = m[c].dt.tz_localize("UTC")
    return arrival_features(dep, arrival_table(m), win)


def for_ranking(dep: pd.DataFrame, win: float = Fe.WIN) -> pd.DataFrame:
    """The arrival block for the ranking months, from the live ranking file."""
    return arrival_features(dep, arrival_table(D.load_ranking(columns=RAW_COLS)), win)


if __name__ == "__main__":
    data = P.load(gate=False)
    for f in P.FOLDS:
        A = for_fold(f, data[f])
        print(f"{f:9s} {A.shape}  nan {A.isna().mean().mean():.4f}  "
              f"a_n median {A['a_n'].median():.0f}  a_taxi_mean median "
              f"{A['a_taxi_mean'].median():.0f}s")
    R = for_ranking(P.load_fold("july_loo"))   # shape only; the join keys are 2026
    print(f"ranking-side function runs, {R.shape}")
