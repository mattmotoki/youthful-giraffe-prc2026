"""Regime-routed taxi-out model for PRC 2026, with the extreme-tail sub-model.

Architecture (measured in `experiments/exp-002-tail/RESULTS.md`). A departure
falls in exactly one of three regimes, and each wants a different target,
because squared error is invariant to the choice of anchor but *learnability*
is not — a gradient-boosted tree cannot reproduce a continuous input feature,
so an anchor that leaves a large offset is a bad anchor:

===========  ================================  ==================================
regime       rows                              target
===========  ================================  ==================================
``nm``       Network Manager join present      ``BLOCK - AOBT_3``, small
``noac``     ``AIRCRAFT_TYPE_mvt`` null        ``BLOCK - SCHED``; half of these
                                               rows carry a placeholder block
                                               time equal to the schedule, so
                                               the offset really is 0 there
``nonm``     no NM join, aircraft type present  the taxi time itself
===========  ================================  ==================================

The ``noac`` regime is 0.07 % of training departures and 388 of the 344,841
ranking departures, but it carries 94 % of the rows above 15,000 s. Predicted
near the bulk mean it puts a floor of about 417 s on the total ranking RMSE;
routed here it contributes about 145 s. See RESULTS.md findings 1-3.

Under RMSE the prediction for a noac row is the mixture *expectation*
``mvt_sched - (1 - p) * offset_hat``, never a hard switch on ``argmax p``: the
placeholder classifier reaches AUC 0.832, which moves the expectation a long
way but does not justify committing to a mode. ``predict`` returns ``p`` so a
caller can see the routing.

Usage::

    from prc.tail_model import TailModel
    import prc.data as D

    train, valid_masked, y_valid = D.fold_frames("july")
    model = TailModel().fit(train)
    pred = model.predict(valid_masked)      # indexed like the departure rows
    rmse = D.rmse(y_valid.reindex(pred.index), pred["y_pred"])

Nothing here reads a departure's ``BLOCK_TIME_UTC_mvt`` or
``TAXITIME_SEC_mvt`` at prediction time; the features come from
``prc.tail_features.tail_signals`` (``docs/input-rule.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from prc.tail_features import SIGNALS, tail_signals

try:  # lirf's phi model (exp-006); optional so this module imports without it
    from prc.phi_features import PhiModel
except Exception:  # pragma: no cover
    PhiModel = None

#: How the phi coefficient is weighted. **Pinned deliberately**: `PhiModel`'s
#: own default changed from `gap2` to `rows` mid-experiment, which silently moved
#: this model's fold RMSEs (326.46/222.83/350.35 -> 327.31/223.12/350.55) and
#: invalidated a set of artifacts. Pinning means a future change to that module's
#: default cannot move this one without someone editing this line.
#:
#: `gap2` (least squares, `sum(r*g)/sum(g^2)`) is the estimator that minimises
#: squared error in exactly this module's NM target, and it measures ~0.85 s
#: better on july. It is *not* the one pinned here: its weight is `g^2`, so on the
#: july training rows a single long-gap flight can own 36 % of its group's
#: coefficient (measured by lirf and their reviewer). A 0.85 s edge on one fold is
#: not worth an estimator whose leverage is concentrated like that, because that
#: is the failure mode that does not transfer to an unseen month. See RESULTS
#: finding 17.
PHI_WEIGHT = "rows"

#: columns `prc.phi_features.PhiModel.transform` contributes to the NM head.
#: phi is defined as BLOCK = AOBT_3 + phi*(LOBT - AOBT_3), so these are exactly
#: this module's NM target written in lirf's parameterisation. Handing them over
#: as features (rather than moving the anchor onto `taxi_hat`) measured better on
#: all three folds: july 331.30 -> 327.31, december 226.39 -> 223.12, january
#: 352.53 -> 350.55, against 327.66 / 223.41 / 351.39 for the anchor form
#: (work/metrics_phi.csv, under the pinned PHI_WEIGHT below). The
#: shrunk phi is an estimate, and a feature lets the model weigh it against
#: everything else instead of committing to it.
PHI_COLUMNS = ("phi", "phi_support", "taxi_hat", "gap", "phi_airport",
               "taxi_hat_airport", "phi_airport_operator",
               "phi_support_airport_operator", "taxi_hat_airport_operator")

#: 2.6.0 — exp-032: an optional caller-supplied window block, `fit(window=W)`.
#: `W` is a frame indexed like the input frame's departure rows; its columns are
#: appended to the `nm` and `nonm` feature lists and to NEITHER the `noac` head
#: nor the band mixture, which is exp-027's measured `offload_noac` rule met by
#: the router's own architecture. OFF by default (`window=None` takes no branch),
#: so the delivered model is byte-unchanged. Nothing is built here: the block
#: lives in `experiments/exp-018-fresh/code/window_features.py` and the caller
#: builds it ONE MONTH AT A TIME, which is `signals_by_month`'s contract too.
#: 2.5.0 — exp-022: an optional schedule-placeholder regime over the non-noac
#: population, `fit(placeholder=True)`. OFF by default, so the delivered model is
#: byte-unchanged until the arms are measured.
#: 2.4.0 — exp-009 v4.1: the band mixture no longer overrides above BAND_TOP,
#: where its two points invert (+0.46 s of board, v4 review N2); a `keep` mask no
#: longer reaches the band head (exp-017 would otherwise collapse it). Bit-
#: identical to 2.3.0 on all four delivered folds: the only training row above
#: BAND_TOP is in 2025-02 and no fold validates February.
#: 2.3.0 — exp-017: `fit` accepts `seed`, a training-side `keep` mask and
#: `weights`, so label hygiene and recency weighting are testable without a fork.
#: 2.2.0 — exp-009: above 8 h of MVT-SCHED the noac rows are a clean two-point
#: mixture (placeholder or 86,400 + taxi, nothing else in 50 training rows), so
#: an explicit fitted mixture replaces the learned regressor there. v4.
#: 2.1.0 — exp-009 round 2: the noac head gains the desk's airport-day arrival
#: placeholder rate (v3). Worth 11-13 s of noac LOMO on the seed-averaged
#: mixture, which is the head that ships; it measured as redundant on the offset
#: head, which does not.
#: 2.0.0 — exp-009: seed-averaged noac head, `noac_head` selects v3 (default)
#: or v1 (exp-002 exactly); FEATURES frozen to `_EXP002_SIGNALS`.
VERSION = "2.6.0"

TARGET = "TAXITIME_SEC_mvt"

CATS = ("apt", "stand", "runway")
#: features every head may use
#: columns that are null on a row with no Network Manager join, so the no-NM
#: head must not see them. Airport-day aggregates of `mvt_aobt` stay: they are
#: computed from *other* departures' `AOBT_3_flt`, which is present for 98.5 %
#: of ranking rows, and are well defined for a row whose own join is missing.
_OWN_FLT = ("mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt", "aobt_lobt",
            "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
            "gap_dep_prev_aobt", "mvt_aobt_vs_day_med",
            "mvt_arvt1", "mvt_arvt3", "arvt3_arvt1")
#: FROZEN at the exp-002 delivered set. `SIGNALS` grows as later experiments add
#: observables (exp-009 added ten), and deriving FEATURES from it would silently
#: change the NM and no-NM heads and invalidate the delivered fold numbers every
#: time someone extends the feature module. New columns reach a head only by
#: being named here or in the head's own list.
_EXP002_SIGNALS = (
    "hour", "dow", "minute_of_day", "noac", "nonm", "vfr", "flight_len",
    "mvt_sched", "mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt",
    "aobt_lobt", "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
    "noac_day_count", "nonm_day_count",
    "apt_day_med_mvt_aobt", "apt_day_p90_mvt_aobt", "apt_day_long_frac",
    "apt_hour_med_mvt_aobt", "mvt_aobt_vs_day_med",
    "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
    "n_arr_sched_mvt", "gap_arr_after_sched", "gap_dep_prev_mvt",
    "gap_dep_prev_aobt", "gap_dep_next_mvt", "stand_busy_frac_6h",
    "mvt_arvt1", "mvt_arvt3", "arvt3_arvt1",
)
FEATURES = _EXP002_SIGNALS + CATS
FEATURES_NONM = tuple(c for c in FEATURES if c not in _OWN_FLT)

#: exp-009: the noac head gains the per-day noac counts, the flight-string
#: shape, and stand/flight-prefix categoricals. Leave-one-month-out over all
#: 1,507 training noac rows: 4,294.1 s -> 4,234.0 s, ranking contribution
#: 144.0 s -> 142.0 s. `noac_head="v1"` restores exp-002's noac FEATURE SET, not
#: its exact predictor: exp-009 made both heads seed-averaged (the delivery path
#: was single-seed, which is why the head that shipped was not the head that was
#: measured -- Opus round 1, B1). exp-002's exact delivered artifact is
#: reproducible from git 963fe96, whose OOF hashes to ffbc8fae.
NOAC_VERSION = "v4"
NOAC_FEATURES_V2 = ("mvt_sched", "sched_date_cross", "flight_len", "hour", "dow", "minute_of_day",
                    "noac_day_count", "nonm_day_count", "n_arr_sched_mvt",
                    "gap_arr_after_sched", "gap_arr1", "gap_arr2", "gap_arr3",
                    "gap_arr_next", "n_arr_6h", "gap_dep_prev_mvt", "gap_dep_next_mvt",
                    "vfr", "noac_day_n", "noac_day_share", "dep_day_n",
                    "arr_within_1h_of_sched", "arr_within_3h_of_sched",
                    "flight_n_digits", "flight_n_alpha", "flight_repeated_prefix",
                    "mvt_sched_log", "arr_day_ph_rate",
                    "apt", "runway", "stand", "flight_prefix")
NOAC_CATS_V2 = ("apt", "runway", "stand", "flight_prefix")

NOAC_FEATURES = ("mvt_sched", "flight_len", "hour", "dow", "minute_of_day",
                 "noac_day_count", "nonm_day_count", "n_arr_sched_mvt", "gap_arr_after_sched",
                 "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
                 "gap_dep_prev_mvt", "gap_dep_next_mvt", "vfr", "apt", "runway")
NOAC_CATS = ("apt", "runway")
PLACEHOLDER_TOL = 300.0
MIN_TAXI = 60.0

#: LightGBM thread count. Callers on a shared machine should pass `n_jobs` to
#: `fit`, or set this, rather than letting LightGBM take every core.
DEFAULT_N_JOBS = 16

#: Bit-for-bit reproducibility: LightGBM is only deterministic with
#: `deterministic` and a row-wise histogram build, and every sampling seed has
#: to be pinned, not just `seed`. Without these the fold RMSEs are stable to
#: about 0.01 s but the artifact SHA-256 is not.
_DETERMINISM = dict(deterministic=True, force_row_wise=True, bagging_seed=7,
                    feature_fraction_seed=7, data_random_seed=7, seed=7)
MAIN_PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=63,
                   min_child_samples=60, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, n_estimators=500, verbose=-1, max_cat_threshold=64,
                   cat_smooth=20, **_DETERMINISM)
NOAC_PARAMS = dict(objective="regression", learning_rate=0.04, num_leaves=8,
                   min_child_samples=30, feature_fraction=0.6, bagging_fraction=0.8,
                   bagging_freq=1, n_estimators=300, verbose=-1,
                   **dict(_DETERMINISM, seed=11, bagging_seed=11,
                          feature_fraction_seed=11, data_random_seed=11))
NOAC_SEEDS = (11, 23, 37, 51, 77)

# --- the high-MVT-SCHED two-point mixture (exp-009, desk 2026-09-05) ---------
#: Above this delay the noac population is a clean TWO-POINT mixture: every one
#: of the 50 training rows is either a placeholder (y = MVT - SCHED) or a
#: rollover (y = 86,400 + a normal taxi, measured mean 1,136 s, sd 400 s).
#: Nothing else occurs, so a learned regressor -- which must interpolate between
#: two widely separated atoms -- is the wrong instrument here and an explicit
#: mixture is the right one.
BAND_MIN = 8 * 3600.0
DAY = 86400.0
#: MVT-SCHED bands, in hours. p(rollover) is 0 below 13.6 h (0 of 29), about a
#: half in 14-19 h (7 of 15), 1 in 20-26 h (4 of 4) and 0 above 26 h (0 of 1).
BAND_EDGES = (8.0, 13.6, 19.0, 26.0, 1e6)
#: shrinkage toward the band's own rate, in pseudo-rows; the cells are tiny
BAND_SHRINK = 2.0
#: The mixture's two points are `mvt_sched` (placeholder) and `86,400 + offset`
#: (rollover). Above the top edge of BAND_EDGES the rollover point lies BELOW
#: the placeholder point, so any p > 0 drags the prediction downward, away from
#: the placeholder outcome -- and the whole annual evidence up there is ONE
#: 2025-02 row which is itself a placeholder (y 131,167 against mvt_sched
#: 131,163). Its shrunk p = 0.098 therefore de-hedges the most extreme ranking
#: row on one observation that points the other way; the v4 review measures the
#: cost at +0.46 s of board. Above this the learned head keeps the row.
#:
#: A general minimum-support gate was tried instead and MEASURED HARMFUL:
#: requiring 3 rows costs 3.05 s on july and 3.29 s on july_loo, because it also
#: removes the 19-26 h band, which is 4-for-4 rollover across the year
#: (`work/decompose_band.log`, R3). So the rule is the domain of the mixture,
#: not a row count.
BAND_TOP = BAND_EDGES[-2] * 3600.0

# ---------------------------------------------------------------- exp-022
#: The schedule-placeholder regime over the NON-noac population. `BLOCK` sits in
#: the same minute as `SCHED` on 3.93 % of july departures and 5.07 % of
#: january's, very unevenly by airport (LIRF 8.27 %, EDDF 2.39 %), and those rows
#: carry 5.98 % of the delivered model's july squared error at 3.90 % of the
#: rows. On them `y = MVT - SCHED - delta` with delta = `BLOCK - SCHED` in
#: [0, 60): mean 23.3 s, sd 24.7. So the branch is an ANCHOR with a ~25 s
#: residual, not the identity that exact-second equality would give (that holds
#: on only 0.32 % of rows -- exp-018's "to the second" does not reproduce).
#: An oracle detector is worth 9.64 s of july; off by default until measured.
PLACEHOLDER_VERSION = "p1"
PH_SEEDS = (11, 23, 37)
PH_FEATURES = ("mvt_sched", "mvt_aobt", "mvt_lobt", "mvt_iobt", "mvt_eobt",
               "aobt_lobt", "lobt_sched", "aobt_sched", "eobt_sched", "lobt_iobt",
               "hour", "dow", "minute_of_day", "flight_len", "vfr", "nonm",
               "apt_day_med_mvt_aobt", "apt_hour_med_mvt_aobt", "dep_day_n",
               "gap_dep_prev_mvt", "gap_dep_next_mvt",
               "apt", "runway", "stand", "flight_prefix")
PH_CATS = ("apt", "runway", "stand", "flight_prefix")
#: exact-anchor coincidences. A tree splits badly on a knife-edge at zero, and
#: these are the strongest single separators measured: AOBT == LOBT == SCHED to
#: the minute carries P(class) = 14.6 % against a 3.9 % base, a lift of 3.7.
#: None of them is decisive -- there is no deterministic subset -- which is why
#: the head is a calibrated mixture and not a rule.
PH_EQ = ("ph_eq_aobt_lobt", "ph_eq_lobt_sched", "ph_eq_aobt_sched")
PH_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                 n_estimators=400, min_child_samples=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                 deterministic=True, force_row_wise=True)


def placeholder_features(sig: pd.DataFrame, phi_cols=()) -> tuple:
    """`sig` plus the exact-anchor indicators, and the feature list to use."""
    out = sig.copy()
    for name, col in zip(PH_EQ, ("aobt_lobt", "lobt_sched", "aobt_sched")):
        v = out[col].to_numpy(float) if col in out.columns else np.full(len(out), np.nan)
        out[name] = np.where(np.isfinite(v), (np.abs(v) < 60.0).astype(float), np.nan)
    feats = PH_FEATURES + PH_EQ + tuple(c for c in phi_cols if c in out.columns)
    return out, feats


def placeholder_label(frame: pd.DataFrame, sig_index) -> pd.Series:
    """`BLOCK` in the same MINUTE as `SCHED`. Training-side only -- reads BLOCK.

    Minute resolution, not exact equality: exact equality holds on 0.32 % of
    2025-07 departures against the 4.67 % the fresh lineage reported, and it is
    the minute form that reproduces both that share and its airport ordering
    (`experiments/exp-022-placeholder/premise_check.py`).
    """
    b = frame["BLOCK_TIME_UTC_mvt"].reindex(sig_index)
    c = frame["SCHED_TIME_UTC_mvt"].reindex(sig_index)
    return (b.dt.floor("min") == c.dt.floor("min")) & b.notna() & c.notna()


def signals_by_month(frame: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """Tail signals for a possibly multi-month frame, one isolated month at a time.

    The ranking file holds January and July 2026 as two separate months; a
    neighbour or per-day feature must not reach across the gap between them.
    """
    if month_col not in frame.columns:
        return tail_signals(frame)
    parts = [tail_signals(g) for _, g in frame.groupby(month_col, sort=True)]
    return pd.concat(parts)


def with_window(sig: pd.DataFrame, window: pd.DataFrame | None) -> tuple[pd.DataFrame, tuple]:
    """Join a caller-supplied window block onto `sig`, and name its columns.

    `window` is indexed like the departure rows of the frame `sig` was built
    from, so a plain reindex both aligns it and drops the departures
    `tail_signals` filtered out (airports outside `RANKING_AIRPORTS`). A column
    that collides with an existing signal is refused rather than silently
    shadowing it.
    """
    if window is None:
        return sig, ()
    clash = [c for c in window.columns if c in sig.columns]
    if clash:
        raise ValueError(f"window block shadows existing signals: {clash[:5]}")
    w = window.reindex(sig.index)
    if not len(w.columns):
        raise ValueError("window block has no columns")
    return pd.concat([sig, w], axis=1), tuple(w.columns)


def regime_of(sig: pd.DataFrame) -> pd.Series:
    """`nm`, `nonm` or `noac` for every departure row of `sig`."""
    r = np.where(sig["noac"].to_numpy() == 1, "noac",
                 np.where(sig["nonm"].to_numpy() == 1, "nonm", "nm"))
    return pd.Series(r, index=sig.index, name="regime")


@dataclass
class TailModel:
    """Fit on training months, predict on one ranking-masked month frame."""

    categories: dict = field(default_factory=dict)
    nm_head: object = None
    nonm_head: object = None
    noac_models: list = field(default_factory=list)
    noac_clf: list = field(default_factory=list)
    noac_reg_np: list = field(default_factory=list)
    _w: object = None
    phi: object = None
    phi_cols: tuple = ()
    noac_head: str = NOAC_VERSION
    band: dict = field(default_factory=dict)
    placeholder: bool = False          # exp-022, off unless fit() asks
    window_cols: tuple = ()            # exp-032, empty unless fit(window=...) asks
    ph_clf: list = field(default_factory=list)
    ph_offset: float = 0.0
    ph_rate: float = 0.0
    _ph_feats: tuple = ()
    _noac_feats: tuple = ()
    _noac_cats: tuple = ()
    fitted: bool = False

    # ---------------------------------------------------------------- helpers
    def _with_phi(self, frame: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
        """Join the phi columns onto `sig`, or return it unchanged."""
        if self.phi is None:
            return sig
        t = self.phi.transform(frame)
        if "MVT_ID_mvt" in t.columns:
            t = t.set_index("MVT_ID_mvt").reindex(frame["MVT_ID_mvt"].reindex(sig.index).to_numpy())
            t.index = sig.index
        else:
            t = t.reindex(sig.index)
        cols = [c for c in PHI_COLUMNS if c in t.columns]
        return pd.concat([sig, t[cols]], axis=1)

    def _encode(self, sig: pd.DataFrame, feats, cats) -> pd.DataFrame:
        X = sig.loc[:, list(feats)].copy()
        for c in cats:
            X[c] = pd.Categorical(sig[c].astype(str), categories=self.categories[c])
        return X

    @staticmethod
    def _labels(frame: pd.DataFrame, sig: pd.DataFrame) -> pd.Series:
        y = frame[TARGET].reindex(sig.index).astype(float)
        if y.isna().all():
            raise ValueError("training frame has no departure taxi times; pass the unmasked months")
        return y

    # -------------------------------------------------------------------- fit
    def fit(self, train_frames, month_col: str = "month", n_jobs: int = DEFAULT_N_JOBS,
            use_phi: bool = True, noac_head: str | None = None,
            seed: int | None = None, keep: pd.Series | None = None,
            weights: pd.Series | None = None,
            placeholder: bool = False,
            window: pd.DataFrame | None = None) -> "TailModel":
        """`train_frames` is a movement frame (or a list of them) *with* labels.

        `keep` is a training-side row mask (exp-017 label hygiene). It reaches
        the learned heads only: the two bulk trees and the noac classifier and
        regressors. It deliberately does NOT reach the high-delay band head --
        see the note at the mask below.
        """
        import lightgbm as lgb

        main = dict(MAIN_PARAMS, n_jobs=n_jobs)
        noac = dict(NOAC_PARAMS, n_jobs=n_jobs)
        if seed is not None:      # exp-017 needs seed variation over the whole model
            main.update(seed=seed, bagging_seed=seed,
                        feature_fraction_seed=seed, data_random_seed=seed)
        self.noac_head = noac_head or self.noac_head
        if self.noac_head in ("v4", "v3", "v2"):
            # v2 and v3 share the code path; v3 is v2 plus `arr_day_ph_rate`
            drop = {"v2": ("arr_day_ph_rate", "sched_date_cross"),
                    "v3": ("sched_date_cross",), "v4": ()}[self.noac_head]
            f = tuple(c for c in NOAC_FEATURES_V2 if c not in drop)
            self._noac_feats, self._noac_cats = f, NOAC_CATS_V2
        elif self.noac_head == "v1":
            self._noac_feats, self._noac_cats = NOAC_FEATURES, NOAC_CATS
        else:
            raise ValueError(f"noac_head must be v1..v4, got {self.noac_head!r}")

        if isinstance(train_frames, (list, tuple)):
            train_frames = pd.concat(train_frames, ignore_index=True)
        sig = signals_by_month(train_frames, month_col)
        if use_phi and PhiModel is not None:
            # phi reads BLOCK_TIME_UTC_mvt, so it is fitted on training months only
            self.phi = PhiModel(weight=PHI_WEIGHT).fit(train_frames)
            sig = self._with_phi(train_frames, sig)
            self.phi_cols = tuple(c for c in PHI_COLUMNS if c in sig.columns)
        sig, self.window_cols = with_window(sig, window)
        y = self._labels(train_frames, sig)
        labelled = y.notna()
        keep_mask = labelled.copy()
        if keep is not None:      # exp-017 label hygiene: a training-side subset
            keep_mask &= keep.reindex(sig.index).fillna(True).astype(bool)
        # The band head is a counting estimator over ~50 rows and 11 of them are
        # the very rollovers a hygiene mask removes; filtering it would set every
        # cell to p = 0 and silently turn the mixture into a constant. So `keep`
        # reaches the LEARNED heads only, and the band keeps all labelled rows.
        sig_band, y_band = sig[labelled], y[labelled]
        self._w = (weights.reindex(sig.index) if weights is not None else None)
        sig, y = sig[keep_mask], y[keep_mask]
        if self._w is not None:
            self._w = self._w[keep_mask]
        self.categories = {c: pd.Index(sorted(set(sig[c].astype(str)))) for c in CATS}
        reg = regime_of(sig)

        nm = reg.to_numpy() == "nm"
        Xnm = self._encode(sig[nm], FEATURES + self.phi_cols + self.window_cols, CATS)
        o_nm = sig.loc[nm, "mvt_aobt"].to_numpy(float) - y[nm].to_numpy(float)
        self.nm_head = lgb.LGBMRegressor(**main).fit(
            Xnm, o_nm, categorical_feature=list(CATS),
            sample_weight=None if self._w is None else self._w[nm].to_numpy())

        # the no-NM head learns the taxi process itself, so it trains on every
        # non-noac departure: whether NM joined a flight is a property of the
        # data pipeline, not of the aircraft's run to the runway. Training it on
        # the no-NM rows alone loses to a per-airport median (RESULTS finding 11).
        ok = reg.to_numpy() != "noac"
        Xnn = self._encode(sig[ok], FEATURES_NONM + self.window_cols, CATS)
        self.nonm_head = lgb.LGBMRegressor(**main).fit(
            Xnn, y[ok].to_numpy(float), categorical_feature=list(CATS),
            sample_weight=None if self._w is None else self._w[ok].to_numpy())

        na = reg.to_numpy() == "noac"
        if na.sum() >= 30:
            for c in self._noac_cats:
                self.categories.setdefault(c, pd.Index(sorted(set(sig[c].astype(str)))))
            Xna = self._encode(sig[na], self._noac_feats, self._noac_cats)
            o_na = sig.loc[na, "mvt_sched"].to_numpy(float) - y[na].to_numpy(float)
            self.noac_models = [lgb.LGBMRegressor(**dict(noac, seed=s, bagging_seed=s,
                feature_fraction_seed=s, data_random_seed=s)).fit(
                Xna, o_na, categorical_feature=list(self._noac_cats)) for s in NOAC_SEEDS]
            ph = (np.abs(o_na) < PLACEHOLDER_TOL).astype(int)
            seeded = lambda s_: dict(noac, seed=s_, bagging_seed=s_,
                                     feature_fraction_seed=s_, data_random_seed=s_)
            self.noac_clf = [lgb.LGBMClassifier(**dict(seeded(s_), objective="binary")).fit(
                Xna, ph, categorical_feature=list(self._noac_cats)) for s_ in NOAC_SEEDS]
            if (ph == 0).sum() >= 20:
                self.noac_reg_np = [lgb.LGBMRegressor(**seeded(s_)).fit(
                    Xna[ph == 0], o_na[ph == 0], categorical_feature=list(self._noac_cats))
                    for s_ in NOAC_SEEDS]
        # --- the high-delay two-point mixture, fitted not hard-coded ----------
        # --- exp-022: the population-wide schedule-placeholder regime --------
        self.placeholder = placeholder
        if placeholder:
            lab = placeholder_label(train_frames, sig.index)
            non = reg.to_numpy() != "noac"          # noac has its own machinery
            ms_ph = sig["mvt_sched"].to_numpy(float)
            usable = non & np.isfinite(ms_ph)
            if usable.sum() and lab.to_numpy()[usable].sum() >= 50:
                for c in PH_CATS:   # `flight_prefix` is not in CATS
                    self.categories.setdefault(c, pd.Index(sorted(set(sig[c].astype(str)))))
                sph, self._ph_feats = placeholder_features(sig, self.phi_cols)
                Xph = self._encode(sph[usable], self._ph_feats, PH_CATS)
                yph = lab.to_numpy()[usable].astype(int)
                self.ph_clf = [lgb.LGBMClassifier(**dict(
                    PH_PARAMS, n_jobs=n_jobs, seed=s_, bagging_seed=s_,
                    feature_fraction_seed=s_, data_random_seed=s_)).fit(
                        Xph, yph, categorical_feature=list(PH_CATS))
                    for s_ in PH_SEEDS]
                # the anchor is MVT-SCHED minus a fitted offset, because the
                # same-minute class is not an identity: delta in [0, 60)
                cls = usable & lab.to_numpy()
                self.ph_offset = float(np.mean(ms_ph[cls] - y.to_numpy(float)[cls]))
                self.ph_rate = float(yph.mean())

        if self.noac_head == "v4":
            na_band = regime_of(sig_band).to_numpy() == "noac"
            if na_band.sum():
                self.band = self._fit_band(sig_band[na_band],
                                           y_band[na_band].to_numpy(float))
        self.fitted = True
        return self

    @staticmethod
    def _fit_band(sig_na: pd.DataFrame, y_na: np.ndarray) -> dict:
        """p(rollover) per (MVT-SCHED band, date-cross) cell, and the rollover offset.

        The class is defined by the OUTCOME -- `|y - 86,400| < 1 h` -- not by a
        threshold on y: a threshold sweeps in placeholders whose schedule happens
        to be more than a day stale, which is what produced a wrong reading of
        this band the first time round.
        """
        ms = sig_na["mvt_sched"].to_numpy(float)
        band = ms >= BAND_MIN
        if band.sum() < 5:
            return {}
        roll = np.abs(y_na - DAY) < 3600.0
        off = float(np.mean(y_na[roll] - DAY)) if roll.any() else 1136.0
        cross = sig_na["sched_date_cross"].to_numpy(float)
        b = np.digitize(ms / 3600.0, BAND_EDGES) - 1
        glob = float(roll[band].mean())
        out = {"offset": off, "global": glob, "cells": {}, "bands": {}}
        for bi in np.unique(b[band]):
            m = band & (b == bi)
            pb = (roll[m].sum() + BAND_SHRINK * glob) / (m.sum() + BAND_SHRINK)
            out["bands"][int(bi)] = float(pb)
            for cv in np.unique(cross[m][~np.isnan(cross[m])]):
                mc = m & (cross == cv)
                if mc.sum():
                    out["cells"][(int(bi), int(cv))] = float(
                        (roll[mc].sum() + BAND_SHRINK * pb) / (mc.sum() + BAND_SHRINK))
        return out

    def _band_predict(self, sig_na: pd.DataFrame) -> np.ndarray:
        """RMSE-optimal expectation over the two points, or NaN where out of band."""
        if not self.band:
            return np.full(len(sig_na), np.nan)
        ms = sig_na["mvt_sched"].to_numpy(float)
        cross = sig_na["sched_date_cross"].to_numpy(float)
        b = np.digitize(ms / 3600.0, BAND_EDGES) - 1
        p = np.full(len(ms), np.nan)
        for i in range(len(ms)):
            if not np.isfinite(ms[i]) or ms[i] < BAND_MIN or ms[i] >= BAND_TOP:
                continue          # BAND_TOP: the mixture inverts above it
            key = (int(b[i]), int(cross[i])) if np.isfinite(cross[i]) else None
            p[i] = self.band["cells"].get(key, self.band["bands"].get(int(b[i]),
                                                                     self.band["global"]))
        rollv = DAY + self.band["offset"]
        return np.where(np.isnan(p), np.nan, p * rollv + (1.0 - p) * ms)

    # ---------------------------------------------------------------- predict
    def predict(self, month_frame: pd.DataFrame, month_col: str = "month",
                window: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-departure predictions for one ranking-masked month frame.

        Columns
        -------
        ``MVT_ID_mvt``      the movement id, for joining back
        ``regime``          ``nm`` / ``nonm`` / ``noac``
        ``y_pred``          the RMSE-optimal prediction, clipped at 60 s
        ``p_placeholder``   P(the block time is a schedule placeholder); NaN
                            outside the ``noac`` regime
        ``y_pred_tail``     the value the row takes if it *is* a placeholder
                            (``MVT_TIME - SCHED_TIME``); NaN elsewhere
        ``y_pred_bulk``     the value the row takes if it is not; NaN elsewhere
        ``y_pred_direct_offset``  the alternative single-regressor noac
                            prediction (RESULTS finding 10); NaN elsewhere
        """
        if not self.fitted:
            raise RuntimeError("TailModel.fit must be called first")
        sig = signals_by_month(month_frame, month_col)
        if self.phi is not None:
            sig = self._with_phi(month_frame, sig)
        # A model fitted with the block cannot be served without it, and one
        # fitted without it cannot be served with it: either way the served
        # estimator would not be the estimator that was measured.
        if self.window_cols:
            sig, got = with_window(sig, window)
            if got != self.window_cols:
                raise ValueError(f"fitted with {len(self.window_cols)} window columns, "
                                 f"asked to predict with {len(got)}")
        elif window is not None:
            raise ValueError("this model was fitted without a window block")
        reg = regime_of(sig)
        out = pd.DataFrame(index=sig.index)
        out["MVT_ID_mvt"] = month_frame["MVT_ID_mvt"].reindex(sig.index)
        out["regime"] = reg
        out["p_placeholder"] = np.nan
        out["y_pred_tail"] = np.nan
        out["y_pred_bulk"] = np.nan
        out["y_pred_direct_offset"] = np.nan
        out["band_p_rollover"] = np.nan

        pred = np.full(len(sig), np.nan)
        nm = reg.to_numpy() == "nm"
        if nm.any():
            pred[nm] = (sig.loc[nm, "mvt_aobt"].to_numpy(float)
                        - self.nm_head.predict(self._encode(
                            sig[nm], FEATURES + self.phi_cols + self.window_cols, CATS)))
        nn = reg.to_numpy() == "nonm"
        if nn.any():
            pred[nn] = self.nonm_head.predict(self._encode(sig[nn], FEATURES_NONM + self.window_cols, CATS))
        na = reg.to_numpy() == "noac"
        if na.any():
            ms = sig.loc[na, "mvt_sched"].to_numpy(float)
            if self.noac_clf and self.noac_reg_np:
                Xna = self._encode(sig[na], self._noac_feats, self._noac_cats)
                # seed-averaged, matching the LOMO harness: a single-seed
                # classifier and regressor here is what made the delivered head
                # a different predictor from the measured one (Opus round 1, B1)
                p = np.mean([c.predict_proba(Xna)[:, 1] for c in self.noac_clf], axis=0)
                off_np = np.mean([r.predict(Xna) for r in self.noac_reg_np], axis=0)
                # RMSE-optimal expectation over the placeholder mixture:
                # E[y] = p*mvt_sched + (1-p)*(mvt_sched - off_np) = mvt_sched - (1-p)*off_np
                pred[na] = ms - (1.0 - p) * off_np
                out.loc[na, "p_placeholder"] = p
                out.loc[na, "y_pred_tail"] = ms
                out.loc[na, "y_pred_bulk"] = np.clip(ms - off_np, MIN_TAXI, None)
                if self.noac_models:
                    Xd = np.mean([m.predict(Xna) for m in self.noac_models], axis=0)
                    out.loc[na, "y_pred_direct_offset"] = np.clip(ms - Xd, MIN_TAXI, None)
            elif self.noac_models:
                Xna = self._encode(sig[na], self._noac_feats, self._noac_cats)
                pred[na] = ms - np.mean([m.predict(Xna) for m in self.noac_models], axis=0)
                out.loc[na, "y_pred_tail"] = ms
            else:  # no noac rows in training: fall back to the no-NM head
                pred[na] = self.nonm_head.predict(self._encode(sig[na], FEATURES_NONM + self.window_cols, CATS))
        # the high-delay band overrides the learned head, where it applies
        if na.any() and self.band and self.noac_head == "v4":
            bp = self._band_predict(sig[na])
            idx = np.where(na)[0]
            ok = ~np.isnan(bp)
            pred[idx[ok]] = bp[ok]
            ms = sig.loc[na, "mvt_sched"].to_numpy(float)
            rollv = DAY + self.band["offset"]
            with np.errstate(invalid="ignore"):
                pr = (bp - ms) / (rollv - ms)
            out.loc[sig.index[na], "band_p_rollover"] = pr
        # --- exp-022: the non-noac schedule-placeholder mixture ---------------
        out["p_placeholder_pop"] = np.nan
        out["ph_anchor"] = np.nan
        if self.placeholder and self.ph_clf:
            non = reg.to_numpy() != "noac"
            ms_ph = sig["mvt_sched"].to_numpy(float)
            usable = non & np.isfinite(ms_ph)
            if usable.any():
                sph, _ = placeholder_features(sig, self.phi_cols)
                Xph = self._encode(sph[usable], self._ph_feats, PH_CATS)
                p_ph = np.mean([c.predict_proba(Xph)[:, 1] for c in self.ph_clf], axis=0)
                anchor = np.clip(ms_ph[usable] - self.ph_offset, MIN_TAXI, None)
                # RMSE-optimal expectation over the two anchors, never the argmax:
                # a hard switch is exp-018's cliff, and the p-rule pays on the mean
                pred[usable] = p_ph * anchor + (1.0 - p_ph) * pred[usable]
                out.loc[sig.index[usable], "p_placeholder_pop"] = p_ph
                out.loc[sig.index[usable], "ph_anchor"] = anchor
        out["y_pred"] = np.clip(pred, MIN_TAXI, None)
        return out

    def fit_predict(self, train_frames, month_frame, month_col: str = "month",
                    n_jobs: int = DEFAULT_N_JOBS, use_phi: bool = True,
                    noac_head: str | None = None) -> pd.DataFrame:
        return self.fit(train_frames, month_col, n_jobs, use_phi, noac_head).predict(month_frame, month_col)
