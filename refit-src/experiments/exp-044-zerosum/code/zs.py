"""Shared machinery for exp-044: Q03's zero-sum objective, in a serving path.

Everything the four scripts of this experiment share.  The base, the fold-side
v10 vector, the packet writer and the 202 row-only features are exp-039's,
imported rather than copied, so the ridge reference reproduces Q03's numbers.

What is new here:

* `design()` builds the same 202 row-only columns as `rowfeat.features` but with
  the categorical level list supplied by the caller.  A fit that crosses months
  cannot use `rowfeat`'s month-local levels: the column set would not match
  between the fitting months and the served month.  `levels_of()` freezes the
  list on the fitting frames alone, by frequency, using no label.
* `lgb_h()` fits LightGBM on Q03's objective with an EXACT GRADIENT and an
  APPROXIMATE Hessian.  With loss `1/2 sum (d_i - rc_i)^2`, `d = h - mean_cell(h)`
  and both `d` and `rc` summing to zero in every cell, `dL/dh_i = d_i - rc_i`
  exactly, so `grad = project(h) - rc` is the true gradient.  The Hessian,
  however, is the projector P itself: diagonal `1 - 1/n_c`, off-diagonal
  `-1/n_c`.  LightGBM takes a diagonal, so `hess_mode="unit"` passes 1 and
  `hess_mode="diag"` passes the exact diagonal `1 - 1/n_c`; both are run and both
  are on the record (Sol round 1, blocking finding 1: an earlier version of this
  docstring called the whole objective exact, which overclaimed the second order).
  Plain LightGBM on `rc` is a DIFFERENT objective (its gradient carries an extra
  `mean_cell(h)`), and is not used.
* `arms()` runs one arm under both cross-fittings: `dayblock` (Q03's, five
  contiguous day blocks inside the month) and `xmonth` (fit on the other two
  folds' months, score the third -- the estimator the served candidate is).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
SCREENS = ROOT / "experiments/exp-039-screens/code"
sys.path.insert(0, str(SCREENS))
import v10 as V           # noqa: E402
import rowfeat as R       # noqa: E402
import packet as PK       # noqa: E402

D = V.D
FOLDS = list(V.FOLDS)
PRE = json.loads((EXP / "analysis" / "predeclared.json").read_text())

# ---- predeclared, and checked against analysis/predeclared.json --------------
BIN = 600
MIN_CELL = 4
ALPHA = 100.0
BLOCKS = 5
BOOT = 2000
SEED = 39
TOL = 1e-8
HOLD_AIRPORTS = ("EHAM", "EDDF")
ELIGIBLE_REGIME = "nm_bulk"
GRIDS = tuple(PRE["step1_density_check"]["grids_seconds"])
RATE_HALF_WINDOW = 300.0
THREADS = 3

_INH = PRE["inherited_from_q03_unchanged"]
assert (_INH["min_cell_rows"], _INH["ridge_alpha"], _INH["day_blocks"],
        _INH["seed"], _INH["bootstrap_draws"], _INH["algebraic_tolerance"],
        _INH["eligible_regime"], tuple(_INH["hold_airports"])) == (
    MIN_CELL, ALPHA, BLOCKS, SEED, BOOT, TOL, ELIGIBLE_REGIME, HOLD_AIRPORTS), \
    "the code does not match analysis/predeclared.json"

LGB_PARAMS = {
    "lgb_small": PRE["step2_learner"]["arms"]["lgb_small"],
    "lgb_wide": PRE["step2_learner"]["arms"]["lgb_wide"],
}
W_CACHE = ROOT / "artifacts" / "exp-018-fresh" / "cache"
W_DROP = ("y", "month", "MVT_ID_mvt")


# ------------------------------------------------------------------ frames ---
_DEP_CACHE: dict[str, pd.DataFrame] = {}
_KEEP = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt",
         "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
         "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
         "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "FLIGHT_ID_mvt",
         "ADEP_mvt"]


def dep(month: str) -> pd.DataFrame:
    """A month's masked departure rows, trimmed to the columns the design uses."""
    if month not in _DEP_CACHE:
        d = R.departures(month)
        _DEP_CACHE[month] = d[[c for c in _KEEP if c in d.columns]].reset_index(drop=True)
    return _DEP_CACHE[month]


def epoch_seconds(s: pd.Series) -> np.ndarray:
    """UTC seconds from a timestamp column, whatever its resolution or tz.

    The training months parse as `datetime64[ns]` and the ranking file as
    `datetime64[us, UTC]`; `astype("int64")` therefore means nanoseconds in one
    and microseconds in the other, and dividing by 1e9 silently turns a
    10-minute cell into a 7-day one.  This subtracts the epoch instead.
    """
    v = pd.to_datetime(s, utc=True)
    return (v - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy("float64")


def ranking_dep() -> pd.DataFrame:
    """The ranking file's departure rows, already blanked by the organizer."""
    rk = D.load_ranking()
    d = rk[rk["PHASE_mvt"] == "DEP"].copy()
    if d["BLOCK_TIME_UTC_mvt"].notna().any() or d[D.TARGET].notna().any():
        raise SystemExit("the ranking file carries a departure block time or target")
    d["month"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    return d.reset_index(drop=True)


# ----------------------------------------------------------------- design ----
def levels_of(frames: list[pd.DataFrame]) -> dict[str, list[str]]:
    """The categorical level list, frozen on the FITTING frames by frequency.

    The same rule `rowfeat.features` applies to one month, applied to whatever
    frames the caller says it is allowed to see.  Frequency uses no label.
    """
    cats = {c: pd.concat([_cat_series(f, c) for f in frames], ignore_index=True)
            for c in R.TOPN}
    return {c: list(s.value_counts().index[:R.TOPN[c]]) for c, s in cats.items()}


def _cat_series(d: pd.DataFrame, col: str) -> pd.Series:
    if col == "pier":
        s = d["STAND_mvt"].astype("string").fillna("").str.slice(0, 2)
    else:
        s = d[col].astype("string")
    return s.fillna("<na>")


def design(d: pd.DataFrame, levels: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """`rowfeat.features`'s columns with a supplied level list."""
    out = {}
    for name, a, b in R.DIFFS:
        v = (d[a] - d[b]).dt.total_seconds().to_numpy("float64")
        out[f"d_{name}"] = np.clip(np.nan_to_num(v, nan=0.0), -R.CLIP, R.CLIP)
        out[f"miss_{name}"] = np.isnan(v).astype("float64")
    t = d["MVT_TIME_UTC_mvt"]
    sec = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy("float64")
    out["tod_sin"] = np.sin(2 * np.pi * sec / 86400.0)
    out["tod_cos"] = np.cos(2 * np.pi * sec / 86400.0)
    out["dow"] = t.dt.dayofweek.to_numpy("float64")
    for k in range(6):
        out[f"dow_{k}"] = (t.dt.dayofweek.to_numpy() == k).astype("float64")
    for col, lv in levels.items():
        s = _cat_series(d, col)
        for name in lv:
            out[f"{col}={name}"] = (s == name).to_numpy().astype("float64")
    cols = list(out)
    return np.column_stack([out[c] for c in cols]).astype("float32"), cols


def final_tag(month: str) -> str:
    """exp-186: "-final" for a 2026 month when $PRC_RANKING_MODE=final, else "".

    The cache file name rule of `exp-018-fresh/code/dataset.py` (`final_tag`): a 2026 table of
    the final file never reuses the table that `ranking.parquet` gave for the same month.
    """
    if not hasattr(D, "ranking_mode"):
        import os
        if os.environ.get("PRC_RANKING_MODE", "ranking") not in ("", "ranking"):
            raise SystemExit("PRC_RANKING_MODE is set, but this prc.data has no ranking modes "
                             "(prc/data.py VERSION >= 7 needed)")
        return ""
    return "-final" if month.startswith("2026") and D.ranking_mode() == "final" else ""


def shipped(month: str, drop_cache_vocab: bool = False) -> pd.DataFrame:
    """The shipped feature block for one month, keyed by movement id (float64).

    Sol round 1, blocking 5: the 13 `c_*` columns are integer codes from ONE
    persistent, append-only vocabulary shared across every cached month
    (`exp-018-fresh/code/dataset.py:_vocab_encode`).  LightGBM sees them as
    ordinary numeric columns, so their ORDER carries information the fitting
    months alone do not fix, and a cross-month arm that uses them is not
    certified fold-isolated.  `drop_cache_vocab=True` removes them, leaving the
    76 columns whose values are functions of the month's own frame.
    """
    f = W_CACHE / f"{month}{final_tag(month)}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing; build it with "
                                f"experiments/exp-018-fresh/code/dataset.py")
    w = pd.read_parquet(f)
    for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "TAXITIME_SEC", "y_true"):
        if c in w.columns:
            raise ValueError(f"{c} is present in the shipped feature block")
    w = w.set_index(w["MVT_ID_mvt"].to_numpy("float64"))
    keep = [c for c in w.columns if c not in W_DROP and pd.api.types.is_numeric_dtype(w[c])]
    if drop_cache_vocab:
        keep = [c for c in keep if not c.startswith("c_")]
    return w[keep].astype("float32")


# ------------------------------------------------------------------ cells ----
def cellcode(air, regime, t, bin_s: int):
    key = (pd.Series(np.asarray(air).astype(str)) + "|"
           + pd.Series(np.asarray(regime).astype(str)) + "|"
           + pd.Series(np.floor(np.asarray(t, "float64") / bin_s).astype("int64")).astype(str))
    code, uniq = pd.factorize(key)
    return code, len(uniq)


def cellmean(v, code, n):
    s = np.bincount(code, weights=v, minlength=n)
    c = np.bincount(code, minlength=n).astype("float64")
    return (s / np.maximum(c, 1))[code]


def project(v, code, n):
    return v - cellmean(v, code, n)


def blocks_of(days):
    u = np.unique(days)
    idx = {d: b for b, chunk in enumerate(np.array_split(u, BLOCKS)) for d in chunk}
    return np.array([idx[d] for d in days])


def traffic_rate(air: np.ndarray, t: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Eligible departures at the same airport within +/- 300 s, the row included.

    A fixed physical window, so the same number on every cell grid.
    """
    out = np.zeros(len(t), "int64")
    for a in np.unique(air):
        m = air == a
        te = np.sort(t[m & elig])
        lo = np.searchsorted(te, t[m] - RATE_HALF_WINDOW, side="left")
        hi = np.searchsorted(te, t[m] + RATE_HALF_WINDOW, side="right")
        out[m] = hi - lo
    return out


def pairs_in_cells(code, ids, n, min_cell, cap: int | None = None):
    """Fixed disjoint within-cell pairs, sorted on (cell, movement id).

    `cap` keeps only the first `cap` rows of each cell (the matched-subset rule
    of step 1); `None` is Q03's rule, which uses every row.
    """
    order = np.lexsort((ids, code))
    c = code[order]
    cnt = np.bincount(code, minlength=n)
    keep = cnt[c] >= min_cell
    order, c = order[keep], c[keep]
    start = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    size = np.diff(np.r_[start, len(c)])
    rank = np.arange(len(c)) - np.repeat(start, size)
    used = size if cap is None else np.minimum(size, cap)
    ok = rank < 2 * np.repeat(used // 2, size)
    order, rank = order[ok], rank[ok]
    return order[rank % 2 == 0], order[rank % 2 == 1]


# ------------------------------------------------------------------- fits ----
def ridge_h(Xfit, rc_fit, code_fit, ncf, Xscore) -> np.ndarray:
    """Q03's ridge: centred features, centred target, no intercept."""
    Xc = Xfit - np.column_stack([cellmean(Xfit[:, j], code_fit, ncf)
                                 for j in range(Xfit.shape[1])])
    mu, sd = Xc.mean(0), Xc.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (Xc - mu) / sd
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += ALPHA
    w = np.linalg.solve(A, Z.T @ rc_fit)
    return ((Xscore - mu) / sd) @ w


def lgb_h(Xfit, rc_fit, code_fit, ncf, Xscore, budget: str,
          hess_mode: str = "unit") -> np.ndarray:
    """Q03's objective in the tree class.

    With loss `1/2 sum (d_i - rc_i)^2`, `d = P h` and `P` the within-cell
    centring projector, the gradient is EXACT: `grad = P h - rc`.  The Hessian is
    `P'P = P` (a projector), whose diagonal is `1 - 1/n_c` and whose off-diagonal
    terms are `-1/n_c` inside each cell.  LightGBM accepts only a diagonal
    Hessian, so the exact object cannot be passed:

      `hess_mode="unit"`   hess = 1              the predeclared arm
      `hess_mode="diag"`   hess = 1 - 1/n_c      the EXACT diagonal of P

    Sol round 1, blocking 1: the first version called this "the objective
    implemented exactly", which is true of the gradient and false of the
    Hessian.  Both modes are now run and reported.
    """
    import lightgbm as lgb
    spec = dict(LGB_PARAMS[budget])
    spec.pop("note", None)
    n_est = int(spec.pop("n_estimators"))
    rc = np.asarray(rc_fit, "float64")

    if hess_mode == "unit":
        hv = np.ones(len(rc))
    elif hess_mode == "diag":
        n_c = np.bincount(code_fit, minlength=ncf).astype("float64")[code_fit]
        hv = 1.0 - 1.0 / np.maximum(n_c, 1.0)
    else:
        raise ValueError(hess_mode)

    def obj(preds, _ds):
        g = project(np.asarray(preds, "float64"), code_fit, ncf) - rc
        return g, hv

    params = {**spec, "objective": obj, "metric": "None", "verbose": -1,
              "num_threads": THREADS, "deterministic": True,
              "force_row_wise": True}
    ds = lgb.Dataset(Xfit, label=rc, free_raw_data=False)
    bst = lgb.train(params, ds, num_boost_round=n_est)
    return bst.predict(Xscore).astype("float64")


FITTERS = {"ridge": lambda *a: ridge_h(*a),
           "lgb_small": lambda *a: lgb_h(*a, "lgb_small"),
           "lgb_wide": lambda *a: lgb_h(*a, "lgb_wide")}


# --------------------------------------------------------------- bootstrap ---
def boot_days(num, den, days, rng):
    codes, uniq = pd.factorize(days)
    k = len(uniq)
    n = np.bincount(codes, weights=num, minlength=k)
    d = np.bincount(codes, weights=den, minlength=k)
    draws = rng.integers(0, k, size=(BOOT, k))
    b = n[draws].sum(1) / d[draws].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(n.sum() / d.sum()), float(lo), float(hi)
