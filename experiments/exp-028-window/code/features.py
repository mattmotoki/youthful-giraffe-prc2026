"""The window features, and the two adapters that reach them.

**One function builds the features on both sides of the fit/serve line.**
`window_features` takes a canonical panel (`panel.CANON`) and returns the design
matrix; `panel.load_fold` produces that panel from committed out-of-fold files
and `panel_from_movements` produces it from a raw movement frame (the ranking
file, or a training month).  `tests/shape_check.py` inverts a real fold's panel
into raw movement columns, runs it back through `panel_from_movements`, and
requires the two design matrices to be equal — so "the same function" is a
checked property, not a comment.

**What a window is.** The departure's airport and its own `win`-second bucket of
take-off time, plus the two buckets either side (`lags`).  Every aggregate is
over the *departures* in that bucket, because the out-of-fold files that the arm
is fitted on carry departure rows only; a serving path that aggregated arrivals
as well would be a different function from the fitted one.  Arrivals are legal
input (`SPEC.md` §4) and are K2's lane, not this one.

**What a window is not.** No aggregate reads a label, a taxi time or a block
time of any row: the inputs are take-off time, NM off-block, last-received EOBT
and two null flags.  See `SPEC.md` §4 for the column-by-column contract.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel as P  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.blend as B  # noqa: E402

WIN = 600.0
LAGS = (-2, -1, 1, 2)
#: The ten airports with departures, fixed so that `apt` means the same integer in
#: every fold and on the ranking file.  Taking the codes from whatever airports
#: happen to be in the frame is the classic fit/serve break: `pd.Categorical` on a
#: sorted per-frame list renumbers every airport the moment one is absent.
#: `LTAI` is listed last because the training data has eleven airports and it has
#: no departures in any fold or in the ranking file (`docs/weather-provenance.md`);
#: naming it keeps the other ten codes fixed if it ever appears.
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
            "LSZH", "LTFM", "LTAI")

#: Columns that describe the row itself and not its window.  The `row` control is
#: the same fit restricted to these.
ROW_ONLY = ["hour", "dow", "apt", "own_ta", "own_al", "base"]


def window_features(panel: pd.DataFrame, win: float = WIN, lags=LAGS) -> pd.DataFrame:
    """Design matrix for one month's departures.  Rows align with `panel`."""
    missing = [c for c in P.CANON if c not in panel.columns and c != "y_true"]
    if missing:
        raise ValueError(f"panel is missing {missing}")
    d = pd.DataFrame({
        "airport": panel["airport"].astype(str).to_numpy(),
        "t": panel["t"].to_numpy("float64"),
        "d_takeoff_aobt": panel["d_takeoff_aobt"].to_numpy("float64"),
        "d_aobt_lobt": panel["d_aobt_lobt"].to_numpy("float64"),
        "regime": panel["regime"].astype(str).to_numpy(),
        "nm_missing": panel["nm_missing"].to_numpy("float64"),
    })
    d["bucket"] = np.floor(d["t"] / win).astype("int64")
    d["late"] = (d["regime"] == "nm_late").astype("float64")
    g = d.groupby(["airport", "bucket"], sort=True)
    src = g.agg(
        w_n=("t", "size"),
        w_ta_mean=("d_takeoff_aobt", "mean"), w_ta_med=("d_takeoff_aobt", "median"),
        w_ta_sd=("d_takeoff_aobt", "std"),
        w_al_mean=("d_aobt_lobt", "mean"),
        w_late=("late", "mean"),
        w_nonm=("nm_missing", "mean"),
    )
    q = g[["d_takeoff_aobt", "d_aobt_lobt"]].quantile(0.9)
    src["w_ta_p90"] = q["d_takeoff_aobt"]
    src["w_al_p90"] = q["d_aobt_lobt"]
    src = src[["w_n", "w_ta_mean", "w_ta_med", "w_ta_sd", "w_ta_p90",
               "w_al_mean", "w_al_p90", "w_late", "w_nonm"]].reset_index()
    key = d[["airport", "bucket"]]
    out = key.merge(src, on=["airport", "bucket"], how="left")
    # the same aggregates one and two buckets back and forward: a state that is
    # rising is not the same state as one that is falling
    for lag in lags:
        s = src.copy()
        s["bucket"] = s["bucket"] - lag
        s = s.rename(columns={c: f"{c}_l{lag}" for c in s.columns if c.startswith("w_")})
        out = out.merge(s, on=["airport", "bucket"], how="left")
    out = out.drop(columns=["airport", "bucket"])
    ts = pd.to_datetime(d["t"], unit="s", utc=True)
    out["hour"] = ts.dt.hour.to_numpy().astype("float64")
    out["dow"] = ts.dt.dayofweek.to_numpy().astype("float64")
    out["apt"] = pd.Categorical(d["airport"], categories=AIRPORTS).codes.astype("float64")
    out["own_ta"] = d["d_takeoff_aobt"].to_numpy()
    out["own_al"] = d["d_aobt_lobt"].to_numpy()
    out["base"] = panel["base"].to_numpy("float64")
    if int((out["apt"] < 0).sum()):
        raise ValueError(f"{int((out['apt'] < 0).sum())} rows at an airport outside {AIRPORTS}")
    return out.astype("float64").reset_index(drop=True)


def window_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith("w_")]


def shuffle_within_day(X: pd.DataFrame, panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The `shuffle` control: permute the window block within airport x day.

    Every window column keeps its marginal distribution and every airport-day
    keeps its own set of window states; only *which departure gets which state*
    is destroyed.  A gain that survives this is an airport-day level effect or a
    free gain of the rotation, not the transient window (`SPEC.md` §3).
    """
    X = X.copy()
    cols = window_cols(X)
    cl = P.clusters(panel)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    perm = idx.copy()
    order = np.argsort(cl, kind="stable")
    starts = np.searchsorted(cl[order], np.arange(cl.max() + 1))
    ends = np.searchsorted(cl[order], np.arange(cl.max() + 1), side="right")
    for a, b in zip(starts, ends):
        g = order[a:b]
        perm[g] = rng.permutation(g)
    X[cols] = X[cols].to_numpy()[perm]
    return X


# ------------------------------------------------------------------ serving --
#: The raw movement columns `panel_from_movements` needs.  All are present in
#: `data/raw/ranking.parquet` (`docs/input-rule.md`); none is a blanked column.
RAW_COLS = ["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
            "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "ADEP_mvt"]
#: The departure airport.  The raw file has no column called `airport` — that name
#: exists only inside the derived feature frames the members were fitted on.
#: `exp-026`'s round-1 review caught the same mistake in `regime_table.py`, where
#: probing for `airport` left every per-airport split silently empty.
AIRPORT_COL = "ADEP_mvt"


def _secs(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a - b).dt.total_seconds().to_numpy("float64")


def panel_from_movements(mvt: pd.DataFrame, base: pd.Series | None = None,
                         airport_col: str = AIRPORT_COL) -> pd.DataFrame:
    """Canonical panel from a raw movement frame's DEPARTURE rows.

    `base` is the served prediction indexed by `MVT_ID_mvt`; it is optional so
    that the equivalence test can build the panel without one.  `y_true` is NaN
    on the ranking file by construction — the arm never reads it at serve time.
    """
    d = mvt[mvt["PHASE_mvt"] == "DEP"].copy()
    ids = d["MVT_ID_mvt"].astype("float64").to_numpy()
    apt = d[airport_col].astype(str)
    if not (apt.str.len() == 4).all():
        raise ValueError(f"{airport_col!r} is not a 4-letter ICAO code on every departure")
    out = pd.DataFrame({
        "MVT_ID_mvt": ids,
        "airport": apt.to_numpy(),
        "t": _secs(d["MVT_TIME_UTC_mvt"], pd.Timestamp("1970-01-01", tz="UTC")),
        "d_takeoff_aobt": _secs(d["MVT_TIME_UTC_mvt"], d["AOBT_3_flt"]),
        "d_aobt_lobt": _secs(d["AOBT_3_flt"], d["LOBT_flt"]),
        "regime": B.regimes_from_raw(d, B.LATE_CUT).astype(str),
        "nm_missing": d["FLIGHT_ID_mvt"].isna().to_numpy().astype("float64"),
        "noac": d["AIRCRAFT_TYPE_mvt"].isna().to_numpy().astype("float64"),
    })
    out["y_true"] = np.nan
    out["base"] = (np.nan if base is None
                   else pd.Series(base).astype("float64")
                   .reindex(pd.Index(ids)).to_numpy())
    if base is not None and not np.isfinite(out["base"]).all():
        raise ValueError(f"{int((~np.isfinite(out['base'])).sum())} departures have no base "
                         f"prediction; the base vector must cover every ranking departure")
    return out[P.CANON].reset_index(drop=True)


# ------------------------------------------------------------------- blocks --
#: The nine aggregates in `window_features`, most informative first as
#: `docs/STRUCTURAL-STEP.md` §8 ranks them: the take-off-minus-off-block level and
#: spread are the transient; the lateness mix and the movement count are context.
AGGS = ("w_ta_mean", "w_ta_med", "w_ta_sd", "w_ta_p90", "w_al_mean", "w_al_p90",
        "w_late", "w_nonm", "w_n")
#: Named subsets of the departure block.  A stacker whose own job needs eleven
#: member columns is diluted by fifty new ones under `feature_fraction=0.7`, so
#: the width of the block is a design choice and gets measured like any other.
#: `extras` are the row-level context columns; a block that carries them lets the
#: stacker re-read the departure's own inputs as well as its window.
BLOCKS = {
    "depx": dict(lags=LAGS, aggs=AGGS, extras=("hour", "dow", "apt",
                                               "own_ta", "own_al")),   # 50 columns
    "dep":  dict(lags=LAGS, aggs=AGGS),                                # 45
    "dep1": dict(lags=(-1, 1), aggs=AGGS),                             # 27
    "depc": dict(lags=(-1, 1), aggs=("w_ta_mean", "w_ta_sd", "w_ta_p90",
                                     "w_al_mean", "w_late", "w_n")),   # 18
    "dep0": dict(lags=(), aggs=AGGS),                                  # 9
}


def select(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """One named subset of a full `window_features` frame, row columns dropped.

    Built by subsetting rather than by rebuilding, so every block is a subset of
    the identical numbers and no two blocks can disagree about a shared column.
    """
    spec = BLOCKS[name]
    keep = [a for a in spec["aggs"]]
    for lag in spec["lags"]:
        keep += [f"{a}_l{lag}" for a in spec["aggs"]]
    keep += list(spec.get("extras", ()))
    missing = [c for c in keep if c not in X.columns]
    if missing:
        raise ValueError(f"block {name!r} wants columns that are not built: {missing}")
    return X[keep]


# -------------------------------------------------------------- deviations --
def airport_level(panel: pd.DataFrame, win: float = WIN) -> pd.DataFrame:
    """The same nine aggregates over the whole month at each airport.

    Legal at serve time: it is computed inside the month being predicted, from
    the same four ranking-file quantities the window aggregates use, exactly as
    the window itself is.  Nothing crosses a month boundary and nothing reads a
    label.
    """
    d = pd.DataFrame({
        "airport": panel["airport"].astype(str).to_numpy(),
        "bucket": np.floor(panel["t"].to_numpy("float64") / win).astype("int64"),
        "d_takeoff_aobt": panel["d_takeoff_aobt"].to_numpy("float64"),
        "d_aobt_lobt": panel["d_aobt_lobt"].to_numpy("float64"),
        "late": (panel["regime"].astype(str).to_numpy() == "nm_late").astype("float64"),
        "nm_missing": panel["nm_missing"].to_numpy("float64"),
    })
    g = d.groupby("airport", sort=True)
    L = pd.DataFrame({
        "w_ta_mean": g["d_takeoff_aobt"].mean(), "w_ta_med": g["d_takeoff_aobt"].median(),
        "w_ta_sd": g["d_takeoff_aobt"].std(), "w_ta_p90": g["d_takeoff_aobt"].quantile(0.9),
        "w_al_mean": g["d_aobt_lobt"].mean(), "w_al_p90": g["d_aobt_lobt"].quantile(0.9),
        "w_late": g["late"].mean(), "w_nonm": g["nm_missing"].mean(),
        # the count baseline is per bucket, so that `w_n` and its baseline are
        # the same kind of number
        "w_n": d.groupby(["airport", "bucket"]).size().groupby("airport").mean(),
    })
    return L


def deviate(X: pd.DataFrame, panel: pd.DataFrame, win: float = WIN,
            keep_level: bool = True) -> pd.DataFrame:
    """Every window aggregate re-expressed as a deviation from its airport level.

    A level feature fitted on two months and applied to a third is a shifted
    input: `w_ta_mean` at LTFM in July is not the same number as at LSZH in
    December, and the stacker has to learn the offset from the fitting months.
    The deviation is the same state description with the offset removed, so it
    transfers.  `keep_level` also keeps the ten airport levels themselves, which
    is the information the deviation drops.
    """
    L = airport_level(panel, win)
    apt = pd.Index(panel["airport"].astype(str).to_numpy())
    out = {}
    for c in window_cols(X):
        # strip only a trailing lag suffix; `c.split("_l")` would eat the `_late`
        # family, which is where the transient's regime mix lives
        agg = re.sub(r"_l-?\d+$", "", c)
        if agg not in L.columns:
            continue
        out[f"{c}_d"] = X[c].to_numpy("float64") - L[agg].reindex(apt).to_numpy()
    D_ = pd.DataFrame(out, index=X.index)
    if keep_level:
        for agg in L.columns:
            D_[f"lvl_{agg}"] = L[agg].reindex(apt).to_numpy()
    return D_
