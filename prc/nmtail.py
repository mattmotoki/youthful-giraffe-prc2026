"""The NM-joined tail: where the airport block clock sat on LOBT, not on AOBT.

0.26 % of departures have a Network Manager join *and* a taxi time above
3600 s. On 94 % of them the recorded block time sits on the last filed
off-block `LOBT_flt` while `AOBT_3_flt` is half an hour to two hours later:
the aircraft left the stand on time and Network Manager saw it move much
later. At about 1900 s RMSE in the best model those rows contribute roughly
100 s in quadrature to a fold RMSE -- but a contribution is not a prize:
predicting the segment *perfectly* is worth 16.2 s of july_loo fold RMSE, 8.9 s
on december and 5.2 s on january, and a perfect regime call a fifth of that.
The segment is understood and small (`experiments/exp-010-nmtail/RESULTS.md`).

**The parameterisation.** With `nm_taxi = MVT - AOBT_3`, `gap = LOBT - AOBT_3`
(both observable) and the lirf worker's coefficient
`phi = (BLOCK - AOBT) / (LOBT - AOBT)`,

    taxi = nm_taxi - phi * gap

is an identity. `phi = 0` is block-on-AOBT, `phi = 1` is block-on-LOBT. Under
squared error the optimal prediction is `nm_taxi - E[phi | x] * gap`, and
because the taxi residual is exactly `gap * (phi_hat - phi)`, **fitting `phi`
by least squares with sample weight `gap**2` is the same objective as fitting
the taxi time** — only reparameterised so that the target is scale-free and the
tree never has to reproduce `nm_taxi` as an input. That is what `NMTailModel`
does; `mode="p"` fits the two-point approximation `E[phi] ~ p(block on LOBT)`
instead, for comparison.

**The screen that is new here.** If another aircraft on-blocks at the stand
between `LOBT` and `AOBT_3`, this departure was gone before it, so the block
time sat near LOBT and not near the NM off-block. `reoccupation_signals`
computes that; it needs no arrival/departure alternation assumption (unlike
`prc.stand_features.displacement_bounds`), only "two aircraft are not on one
stand at once". Over all twelve 2025 months, on rows with `AOBT_3 - LOBT > 600 s`,
it fires on 1.80 % and lifts P(block-on-LOBT) from 12.2 % to 49.8 % and
P(taxi > 3600 s) from 0.41 % to 5.14 %; requiring the intervening arrival to be
a *different aircraft type* (so it is not this flight's own late inbound)
raises the first to 55.3 %. **Measured inside a model that already has
`AOBT_3 - LOBT` it is worth nothing** (+17 s on the july_loo NM tail, -5 s on
december); it is offered as a cheap column, not as a gain.

Nothing here reads a departure's own `BLOCK_TIME_UTC_mvt` or
`TAXITIME_SEC_mvt`. Arrival block times are legal inputs
(`docs/input-rule.md`); `NMTailModel.fit` reads departure block times from the
training months only, which is what training months are for.

Usage::

    import prc.nmtail as nt
    sig   = nt.reoccupation_signals(month_frame)      # per departure, one month
    model = nt.NMTailModel().fit(train_features, train_taxi)
    pred  = model.predict(valid_features)             # taxi seconds
"""

from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np
import pandas as pd

import prc.stand_features as sf

__all__ = ["VERSION", "SIGNALS", "reoccupation_signals", "NMTailModel"]

#: Module version. Desk rule: never change a default silently -- bump this, keep
#: the old behaviour reachable by argument, and name the consumers in the report.
#:
#: 4  2026-09-06  `deterministic=True` and `force_row_wise=True` added to the
#:    LightGBM parameters. Without them this model was the ONLY member of the
#:    standing blend that failed a bit-stability test at a fixed thread count:
#:    two runs at six threads on identical data gave 10f9671d… and 51e98c54…,
#:    while the four members whose builders already set the flag were all
#:    bit-stable (exp-015 separation, 2026-09-06). LightGBM documents
#:    `deterministic=True` as reproducible "with the same data and the same
#:    number of threads", and every observation in that separation fits the
#:    sentence exactly. The previous behaviour is
#:    `NMTailModel(params_extra={"deterministic": False})` — but note it was not
#:    a deliberate choice, it was an omission relative to retro-v1's
#:    `DEFAULT_PARAMS` and exp-008's `LgbLearner`, both of which set both flags.
#:    This CHANGES the model's output bytes; exp-010's published figures were
#:    produced without the flags.
#: 3  2026-09-06  `num_threads` now defaults to the box-wide compute allocation
#:    (`PRC_THREADS`, falling back to 8) instead of a hardcoded 8. A hardcoded
#:    count made a *declared* thread count untrue: exp-015's release manifest
#:    declared 6 for this model while 8 executed, and the release reproducibility
#:    contract rests on that field being the count the library actually used. The
#:    previous behaviour is `NMTailModel(num_threads=8)`, and passing any explicit
#:    value still wins over the environment. Consumers: `experiments/exp-010-nmtail/code/`
#:    (owner) and `experiments/exp-012-blend2/nmtail_submit.py`. Changed by the
#:    diversity lane on the desk's instruction (2026-09-06); noted for exp-010 in
#:    their RESULTS.md.
#: 2  2026-09-05  `NMTailModel` defaults changed to the configuration every
#:    number in exp-010's RESULTS.md is measured at: `n_estimators` 600 -> 900 and
#:    `learning_rate` 0.05 -> 0.04, so that the documented reproduction command
#:    reproduces the reported figures (review round 1, B3). Nothing else changed;
#:    the previous behaviour is `NMTailModel(n_estimators=600, learning_rate=0.05)`.
#:    Only consumer: `experiments/exp-010-nmtail/code/`.
#: 1  2026-09-05  first version (exp-010). Defaults pinned by the experiment:
#:    `NMTailModel(mode="phi", min_gap=60.0)`, `reoccupation_signals(key="key_raw")`.
VERSION = 4

#: columns returned by :func:`reoccupation_signals`
SIGNALS = (
    "reocc_lobt_aobt",       # arrivals parking at the stand between LOBT and AOBT
    "reocc_lobt_aobt_diff",  # ... of a different aircraft type from this departure
    "reocc_lobt_mvt",        # arrivals parking between LOBT and this take-off
    "reocc_aobt_mvt",        # arrivals parking between AOBT and this take-off
    "reocc_before_lobt",     # arrivals at the stand before LOBT (is there an inbound?)
    "reocc_first_gap_lobt",  # seconds from LOBT to the first such arrival
    "reocc_last_gap_mvt",    # seconds from the last such arrival to the take-off
)


def _runs(sorted_keys: np.ndarray):
    if len(sorted_keys) == 0:
        return np.array([]), np.array([], int), np.array([], int)
    ch = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    st = np.concatenate([[0], ch])
    en = np.concatenate([ch, [len(sorted_keys)]])
    return sorted_keys[st], st, en


def reoccupation_signals(mvt: pd.DataFrame, key: str = "key_raw") -> pd.DataFrame:
    """Stand re-occupation windows for every departure of one month's frame.

    `mvt` is a single month of movements as the ranking file gives it (arrivals
    complete, departures with block and taxi time blank). Returns a frame indexed
    like `mvt`'s departure rows with the columns in :data:`SIGNALS`. Call it once
    per calendar month: the ranking file holds January and July 2026 as two
    isolated months and a joint call would search across the six-month gap.
    """
    df = mvt if "key_raw" in mvt.columns else sf.add_stand_keys(mvt)
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    dep = df.loc[is_dep]
    arr = df.loc[~is_dep]
    arr = arr[arr["BLOCK_TIME_UTC_mvt"].notna()]
    ns = lambda s: s.dt.as_unit("ns").astype("int64").to_numpy()
    NAT = np.iinfo(np.int64).min

    a_key = arr[key].astype(object).fillna("?").astype(str).to_numpy()
    a_t = ns(arr["BLOCK_TIME_UTC_mvt"])
    a_ty = arr["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    o = np.argsort(a_key, kind="mergesort")
    a_key, a_t, a_ty = a_key[o], a_t[o], a_ty[o]
    spans = {k: (s, e) for k, s, e in zip(*_runs(a_key))}

    n = len(dep)
    cols = {c: np.full(n, np.nan) for c in SIGNALS}
    L, A, T = ns(dep["LOBT_flt"]), ns(dep["AOBT_3_flt"]), ns(dep["MVT_TIME_UTC_mvt"])
    d_ty = dep["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    lo_edge = np.where(L == NAT, T, L)      # no LOBT: the window collapses
    hi_edge = np.where(A == NAT, T, A)
    dk = dep[key].astype(object).fillna("?").astype(str)

    for k, idx in dk.groupby(dk).indices.items():
        se = spans.get(k)
        if se is None:
            for c in ("reocc_lobt_aobt", "reocc_lobt_aobt_diff", "reocc_lobt_mvt",
                      "reocc_aobt_mvt", "reocc_before_lobt"):
                cols[c][idx] = 0.0
            continue
        s, e = se
        order = np.argsort(a_t[s:e], kind="mergesort")
        t_s, ty_s = a_t[s:e][order], a_ty[s:e][order]
        lo = np.searchsorted(t_s, lo_edge[idx], "right")
        hi_a = np.searchsorted(t_s, hi_edge[idx], "left")
        hi_t = np.searchsorted(t_s, T[idx], "left")
        after_a = np.searchsorted(t_s, hi_edge[idx], "right")
        cols["reocc_before_lobt"][idx] = lo
        cols["reocc_lobt_aobt"][idx] = np.maximum(hi_a - lo, 0)
        cols["reocc_lobt_mvt"][idx] = np.maximum(hi_t - lo, 0)
        cols["reocc_aobt_mvt"][idx] = np.maximum(hi_t - after_a, 0)
        for j, i in enumerate(idx):
            if hi_a[j] > lo[j]:
                cols["reocc_lobt_aobt_diff"][i] = int((ty_s[lo[j]:hi_a[j]] != d_ty[i]).sum())
            else:
                cols["reocc_lobt_aobt_diff"][i] = 0.0
            if hi_t[j] > lo[j]:
                cols["reocc_first_gap_lobt"][i] = (t_s[lo[j]] - lo_edge[i]) / 1e9
                cols["reocc_last_gap_mvt"][i] = (T[i] - t_s[hi_t[j] - 1]) / 1e9
    return pd.DataFrame(cols, index=dep.index)[list(SIGNALS)]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

@dataclass
class NMTailModel:
    """`E[phi | x]` on NM-joined departures, and the taxi time it implies.

    `mode="phi"` (default, and what exp-010 measured) fits `phi` by least squares
    with sample weight `gap**2`, which is exactly least squares on the taxi time.
    `mode="p"` fits the two-point approximation, a binary classifier for
    `block-on-LOBT`, and uses its probability as `E[phi]`; it is kept because the
    desk asked for a calibrated `p`, and because a probability is auditable in a
    way a regression output is not.

    `min_gap` is the smallest `|LOBT - AOBT|` that identifies a coefficient at
    all; rows below it carry almost no weight anyway and are dropped from the fit.
    """

    mode: str = "phi"
    min_gap: float = 60.0
    num_leaves: int = 63
    learning_rate: float = 0.04
    n_estimators: int = 900
    min_data_in_leaf: int = 60
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    #: Defaults to the box-wide allocation so a declared thread count is the one
    #: that runs; an explicit argument still wins. See the VERSION 3 note.
    num_threads: int = field(default_factory=lambda: int(os.environ.get("PRC_THREADS", "8")))
    seed: int = 7
    clip: tuple[float, float] = (-2.0, 3.0)
    features: list[str] = field(default_factory=list)
    categoricals: list[str] = field(default_factory=list)
    model_: object = None

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _gap(X: pd.DataFrame) -> np.ndarray:
        return X["gap"].to_numpy(dtype=float)

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        M = X[self.features].copy()
        for c in self.categoricals:
            if c in M.columns:
                M[c] = M[c].astype("category")
        return M

    def _params(self) -> dict:
        p = dict(num_leaves=self.num_leaves, learning_rate=self.learning_rate,
                 min_data_in_leaf=self.min_data_in_leaf,
                 feature_fraction=self.feature_fraction,
                 bagging_fraction=self.bagging_fraction, bagging_freq=1,
                 lambda_l2=self.lambda_l2, num_threads=self.num_threads,
                 deterministic=True, force_row_wise=True,
                 verbose=-1, seed=self.seed, max_cat_threshold=64, cat_smooth=20.0)
        p["objective"] = "binary" if self.mode == "p" else "regression"
        if self.mode != "p":
            p["metric"] = "l2"
        return p

    # ---- fit / predict --------------------------------------------------
    def fit(self, X: pd.DataFrame, taxi: np.ndarray) -> "NMTailModel":
        """`X` must carry `nm_taxi`, `gap` and `self.features`; `taxi` is the label."""
        import lightgbm as lgb
        g = self._gap(X)
        y = np.asarray(taxi, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(np.abs(g) > 1e-9, (X["nm_taxi"].to_numpy(float) - y) / g, np.nan)
        keep = np.isfinite(phi) & (np.abs(g) >= self.min_gap) & np.isfinite(y)
        if self.mode == "p":
            label = ((np.abs(y - (X["nm_taxi"].to_numpy(float) - g))
                      < np.abs(y - X["nm_taxi"].to_numpy(float)))[keep]).astype(float)
            weight = np.ones(keep.sum())
        else:
            label = np.clip(phi[keep], *self.clip)
            weight = g[keep] ** 2
            weight = weight / weight.mean()
        ds = lgb.Dataset(self._matrix(X[keep]), label=label, weight=weight)
        self.model_ = lgb.train(self._params(), ds, num_boost_round=self.n_estimators)
        return self

    def predict_phi(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model_.predict(self._matrix(X))
        return np.clip(raw, *self.clip) if self.mode != "p" else raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Taxi seconds: `nm_taxi - E[phi] * gap`, non-negative."""
        phi = self.predict_phi(X)
        return np.maximum(X["nm_taxi"].to_numpy(float) - phi * self._gap(X), 0.0)
