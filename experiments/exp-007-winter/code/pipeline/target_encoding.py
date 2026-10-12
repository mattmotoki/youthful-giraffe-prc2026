"""Group statistics fitted on training months only (COMMON.md), with a fallback
chain so a stand with no 2025 history falls back to its runway and then to its
airport.  Munich opened an apron (stands 103-108) that has no history at all,
and CDG lost two runways, so the chain is not decoration (docs/shift-2026.md).

Two quantities are encoded:

* the taxi time itself, keyed on the physical route (airport, stand, runway) —
  an unimpeded taxi-time estimate that a categorical split over 2,303 stands
  approximates badly;
* the block-versus-AOBT gap `r = BLOCK - AOBT = (take-off - AOBT) - taxi`,
  keyed on operator and airport, which is the systematic part of the reporting
  difference between the two systems.

Training rows are encoded leave-one-month-out so a row never sees its own
month's statistic; validation and ranking rows use every training month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTH = 20.0

# (name, key columns, quantity) with the fallback chain implied by the key order
SPECS = [
    ("route", ["airport", "airport_stand", "airport_runway"], "taxi"),
    ("stand", ["airport", "airport_stand"], "taxi"),
    ("runway", ["airport", "airport_runway"], "taxi"),
    ("actype", ["airport", "actype_mvt"], "taxi"),
    ("operator", ["airport", "operator"], "gap"),
    ("op_stand", ["airport", "operator", "airport_stand"], "gap"),
]


def _quantity(X: pd.DataFrame, y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "taxi":
        return y
    gap = X["d_takeoff_aobt"].to_numpy(dtype="float64") - y  # BLOCK - AOBT
    return gap


def _key(X: pd.DataFrame, cols: list[str]) -> pd.Series:
    s = X[cols[0]].astype("string")
    for c in cols[1:]:
        s = s + "\x1f" + X[c].astype("string")
    return s.fillna("\x00NA")


def _stats(key: pd.Series, v: np.ndarray, months: np.ndarray) -> pd.DataFrame:
    """Per-key count and sum, overall and per month, for leave-one-month-out."""
    df = pd.DataFrame({"k": key.to_numpy(), "v": v, "m": months})
    tot = df.groupby("k")["v"].agg(["count", "sum"])
    per = df.groupby(["k", "m"])["v"].agg(["count", "sum"])
    return tot, per


class Encoder:
    """Fit on the training frame, apply to any frame."""

    def __init__(self, specs=SPECS, smooth: float = SMOOTH):
        self.specs, self.smooth = specs, smooth
        self.tot: dict = {}
        self.per: dict = {}
        self.prior: dict = {}
        self.prior_per: dict = {}
        self.prior_tot: dict = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Encoder":
        months = X["month_num"].to_numpy()
        for name, cols, kind in self.specs:
            v = _quantity(X, y, kind)
            ok = np.isfinite(v)
            self.prior[name] = float(np.mean(v[ok])) if ok.any() else 0.0
            # the prior is a group statistic too, so it is held out per month as
            # well; using the all-month mean leaked the held-out month into the
            # smoothing target (gpt-5.6-sol, 2026-09-05).
            pm = pd.DataFrame({"m": months[ok], "v": v[ok]}).groupby("m")["v"].agg(["count", "sum"])
            self.prior_per[name] = pm
            self.prior_tot[name] = (int(ok.sum()), float(v[ok].sum()))
            for depth in range(1, len(cols) + 1):
                k = _key(X[ok], cols[:depth])
                self.tot[(name, depth)], self.per[(name, depth)] = _stats(k, v[ok], months[ok])
        return self

    def _level(self, name, depth, X, months=None):
        """Smoothed mean for one key level; leave-one-month-out when months given."""
        cols = dict((n, c) for n, c, _ in self.specs)[name][:depth]
        k = _key(X, cols)
        tot = self.tot[(name, depth)]
        n = k.map(tot["count"]).to_numpy(dtype="float64")
        s = k.map(tot["sum"]).to_numpy(dtype="float64")
        if months is not None:
            per = self.per[(name, depth)]
            idx = pd.MultiIndex.from_arrays([k.to_numpy(), months])
            n = n - per["count"].reindex(idx).to_numpy(dtype="float64", na_value=0.0)
            s = s - per["sum"].reindex(idx).to_numpy(dtype="float64", na_value=0.0)
        n = np.nan_to_num(n)
        s = np.nan_to_num(s)
        return n, s

    def transform(self, X: pd.DataFrame, leave_one_month_out: bool = False) -> pd.DataFrame:
        months = X["month_num"].to_numpy() if leave_one_month_out else None
        out = {}
        for name, cols, _ in self.specs:
            if months is None:
                parent = np.full(len(X), self.prior[name], dtype="float64")
            else:
                n0, s0 = self.prior_tot[name]
                pm = self.prior_per[name]
                idx = pd.Index(months)
                n = n0 - idx.map(pm["count"]).to_numpy(dtype="float64", na_value=0.0)
                sm = s0 - idx.map(pm["sum"]).to_numpy(dtype="float64", na_value=0.0)
                parent = np.where(n > 0, sm / np.maximum(n, 1), self.prior[name])
            for depth in range(1, len(cols) + 1):
                n, s = self._level(name, depth, X, months)
                enc = (s + parent * self.smooth) / (n + self.smooth)
                parent = enc
                if depth == len(cols):
                    out[f"te_{name}"] = enc
                    out[f"te_{name}_n"] = n
        return pd.DataFrame(out, index=X.index)


#: the physical route encodings only, without the two that encode the
#: block-versus-AOBT gap. c9 showed the full set costs 43 s on the july fold and
#: all of it in the core, so the gap encodings are screened separately.
TAXI_ONLY = [sp for sp in SPECS if sp[2] == "taxi"]


def add(Xtr: pd.DataFrame, ytr: np.ndarray, others: list[pd.DataFrame], specs=SPECS):
    """Encode the training frame leave-one-month-out and every other frame fully."""
    e = Encoder(specs).fit(Xtr, ytr)
    Xtr = pd.concat([Xtr, e.transform(Xtr, leave_one_month_out=True)], axis=1)
    return Xtr, [pd.concat([X, e.transform(X)], axis=1) for X in others]
