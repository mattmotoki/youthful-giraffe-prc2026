"""The arrivals block keyed on runway and stand — exp-028 §10's revival condition.

exp-028 measured the arrivals' unmasked taxi-in aggregated over the **airport's**
window and it lost 0.309 s on all three folds (RESULTS §7a).  Its own diagnosis:
"an arrival's taxi-in describes the inbound surface, and the departure's queue is
the outbound one", and the repair it names is to key the arrivals on the runway
and the stand the ranking file also gives rather than on the airport alone.  This
is that repair.

**Which key, and why not the exact runway.**  A departure's `RUNWAY_mvt` is the
strip it takes off from and an arrival's is the strip it lands on, and six of the
ten airports run segregated parallel operations, so an *exact* match empties the
cell: on July 2025 only **12.2 %** of departure rows have any arrival in their own
(airport, exact runway, 600 s) cell, against **50.7 %** when the parallel
designator is dropped and the key is the runway's direction (`27L`, `27R` -> `27`),
which is the airport's configuration and is what arrivals and departures share.
The stand key is the stand code's leading character — the pier or apron block —
which is a letter at the letter-coded airports (`B41` -> `B`) and the hundreds
block at the number-coded ones (`324` -> `3`), 8 to 23 values per airport and
**58.1 %** of departure rows with a populated own-bucket cell.  Measured by
`python arrk.py`, which prints the table above.

**Legality and population.**  `RUNWAY_mvt` is 100 % populated on departures and on
arrivals in the ranking file and in all three fold months.  `STAND_mvt` is very
nearly so and **not exactly** — 3 / 1 / 7 departures and 9 / 4 / 0 arrivals are
null in july_loo / december / january, and 4 departures and 5 arrivals in the
ranking file (Sol round 1; the first version of this file claimed 100 % for both,
which was a rounding of 0.99998).  A null stand takes the key `"?"`, its own
value, rather than being folded into the `N` block by `astype(str)`.  Neither
column is a blanked column (`docs/input-rule.md`: only the departure `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` are blanked).  The taxi-in and lateness read here are the
**arrivals'**, which carry full actuals.  No label and no departure block or taxi
time enters a feature.

One function, `features`, builds the block on both sides of the fit/serve line;
`for_fold` and `for_ranking` differ only in which raw file they open.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
W28 = ROOT / "experiments/exp-028-window/code"
sys.path.insert(0, str(W28))
import features as Fe  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
#: Raw columns this block needs.  Every one is present in `ranking.parquet`.
RAW_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            "STAND_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt"]
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
#: The two keys, and the prefix their aggregates carry.
KEYS = (("rwy", "rwy"), ("std", "std"))


def runway_dir(s: pd.Series) -> np.ndarray:
    """`27L` -> `27`.  The airport's configuration, shared by arrivals and departures."""
    out = s.astype("string").str.replace(r"[A-Za-z]+$", "", regex=True)
    return out.fillna("?").to_numpy().astype(object)


def stand_area(s: pd.Series) -> np.ndarray:
    """`B41` -> `B`, `324` -> `3`.  The pier or apron block; a null stand is `"?"`.

    The null is given its own key rather than `astype(str)`'s `"None"` -> `"N"`,
    which would silently pool a handful of unknown stands with every stand whose
    code starts with N (Sol round 1).  It is 0 to 9 rows per frame.
    """
    out = s.astype("string").str.slice(0, 1).str.upper()
    return out.fillna("?").to_numpy().astype(object)


def _times(m: pd.DataFrame) -> pd.DataFrame:
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
        if not pd.api.types.is_datetime64_any_dtype(m[c]):
            m[c] = pd.to_datetime(m[c], utc=True)
        elif m[c].dt.tz is None:
            m[c] = m[c].dt.tz_localize("UTC")
    return m


def dep_keys(mvt: pd.DataFrame) -> pd.DataFrame:
    """(id, airport, take-off time, runway direction, stand area) per departure."""
    d = mvt[mvt["PHASE_mvt"] == "DEP"]
    return pd.DataFrame({
        "MVT_ID_mvt": d["MVT_ID_mvt"].astype("float64").to_numpy(),
        "airport": d["ADEP_mvt"].astype(str).to_numpy(),
        "t": (d["MVT_TIME_UTC_mvt"] - EPOCH).dt.total_seconds().to_numpy("float64"),
        "rwy": runway_dir(d["RUNWAY_mvt"]),
        "std": stand_area(d["STAND_mvt"]),
    })


def arr_table(mvt: pd.DataFrame) -> pd.DataFrame:
    """(airport, landing time, taxi-in, lateness, runway direction, stand area)."""
    a = mvt[mvt["PHASE_mvt"] == "ARR"]
    t = (a["MVT_TIME_UTC_mvt"] - EPOCH).dt.total_seconds().to_numpy("float64")
    return pd.DataFrame({
        "airport": a["ADES_mvt"].astype(str).to_numpy(),
        "t": t,
        "a_taxi": a["TAXITIME_SEC_mvt"].to_numpy("float64"),
        "a_late": (a["MVT_TIME_UTC_mvt"] - a["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
                  .to_numpy("float64"),
        "rwy": runway_dir(a["RUNWAY_mvt"]),
        "std": stand_area(a["STAND_mvt"]),
    })


def _one_key(dep: pd.DataFrame, arr: pd.DataFrame, key: str, pre: str,
             win: float, lags) -> pd.DataFrame:
    a = arr.copy()
    a["bucket"] = np.floor(a["t"] / win).astype("int64")
    g = a.groupby(["airport", key, "bucket"], sort=True)
    src = g.agg(**{f"{pre}_n": ("a_taxi", "size"),
                   f"{pre}_taxi_mean": ("a_taxi", "mean"),
                   f"{pre}_late_mean": ("a_late", "mean")})
    src[f"{pre}_taxi_p90"] = g["a_taxi"].quantile(0.9)
    src = src[[f"{pre}_n", f"{pre}_taxi_mean", f"{pre}_taxi_p90",
               f"{pre}_late_mean"]].reset_index()
    key_f = pd.DataFrame({"airport": dep["airport"].astype(str).to_numpy(),
                          key: dep[key].to_numpy(),
                          "bucket": np.floor(dep["t"].to_numpy("float64") / win).astype("int64")})
    out = key_f.merge(src, on=["airport", key, "bucket"], how="left")
    for lag in lags:
        s = src.copy()
        s["bucket"] = s["bucket"] - lag
        # the key column is called `rwy`/`std` and the aggregates are prefixed the
        # same way, so the three join keys are excluded by name rather than by prefix
        s = s.rename(columns={c: f"{c}_l{lag}" for c in s.columns
                              if c not in ("airport", key, "bucket")})
        out = out.merge(s, on=["airport", key, "bucket"], how="left")
    return out.drop(columns=["airport", key, "bucket"]).astype("float64").reset_index(drop=True)


def features(dep: pd.DataFrame, arr: pd.DataFrame, win: float = Fe.WIN,
             lags=Fe.LAGS) -> pd.DataFrame:
    """The 40-column block: 4 aggregates x (own + 4 lags) x 2 keys.  Rows align with `dep`."""
    return pd.concat([_one_key(dep, arr, k, pre, win, lags) for k, pre in KEYS],
                     axis=1).reset_index(drop=True)


def _align(pan: pd.DataFrame, mvt: pd.DataFrame, win: float, lags) -> pd.DataFrame:
    """Build the block on the raw file's departures, then put it in `pan`'s order."""
    dk = dep_keys(mvt)
    order = pd.Index(dk["MVT_ID_mvt"].to_numpy("float64")).get_indexer(
        pan["MVT_ID_mvt"].to_numpy("float64"))
    if (order < 0).any():
        raise SystemExit(f"arrk: {int((order < 0).sum())} panel rows are not departures "
                         f"in the raw file")
    X = features(dk, arr_table(mvt), win, lags)
    return X.iloc[order].reset_index(drop=True)


def for_fold(fold: str, pan: pd.DataFrame, win: float = Fe.WIN, lags=Fe.LAGS) -> pd.DataFrame:
    m = _times(pd.read_parquet(D.month_file(FOLD_MONTH[fold]), columns=RAW_COLS))
    return _align(pan, m, win, lags)


def for_ranking(rk: pd.DataFrame, win: float = Fe.WIN, lags=Fe.LAGS) -> pd.DataFrame:
    return _align(rk, D.load_ranking(columns=RAW_COLS), win, lags)


if __name__ == "__main__":
    import json
    rows = []
    for label, m in [("july_loo", _times(pd.read_parquet(D.month_file("2025-07"),
                                                         columns=RAW_COLS))),
                     ("ranking", D.load_ranking(columns=RAW_COLS))]:
        d, a = dep_keys(m), arr_table(m)
        for win in (300.0, 600.0):
            r = {"frame": label, "win": win, "n_dep": int(len(d)), "n_arr": int(len(a))}
            for key, pre in [("exact_rwy", None), ("rwy", "rwy"), ("std", "std")]:
                if key == "exact_rwy":
                    dk = m[m.PHASE_mvt == "DEP"]["RUNWAY_mvt"].astype(str).to_numpy()
                    ak = m[m.PHASE_mvt == "ARR"]["RUNWAY_mvt"].astype(str).to_numpy()
                    dd, aa = d.assign(k=dk), a.assign(k=ak)
                else:
                    dd, aa = d.assign(k=d[key]), a.assign(k=a[key])
                cells = aa.assign(b=np.floor(aa.t / win).astype("int64")) \
                          .groupby(["airport", "k", "b"]).size()
                idx = pd.MultiIndex.from_arrays(
                    [dd.airport, dd.k, np.floor(dd.t / win).astype("int64")])
                hit = idx.isin(cells.index)
                r[f"{key}_cover"] = round(float(hit.mean()), 4)
                r[f"{key}_median_per_cell"] = float(cells.median())
                r[f"{key}_cells"] = int(len(cells))
            rows.append(r)
            print(json.dumps(r))
    (EXP / "analysis" / "arrk_keys.json").write_text(json.dumps(rows, indent=2))
