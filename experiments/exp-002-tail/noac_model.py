"""The noac sub-model: RMSE-optimal expectation over the placeholder mixture.

A noac row (`AIRCRAFT_TYPE_mvt` null) carries, with probability p, a block time
that is a placeholder equal to `SCHED_TIME_UTC_mvt`, in which case the target is
exactly `MVT_TIME - SCHED_TIME`; otherwise it carries a real block time. Under
RMSE the optimal prediction is the mixture expectation

    E[y | x] = mvt_sched - (1 - p(x)) * E[o | not placeholder, x],   o = BLOCK - SCHED

never a hard switch on argmax p. Both are measured in `fit_predict`.
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb

PH_TOL = 300.0
FEATS = ["mvt_sched", "flight_len", "hour", "dow", "minute_of_day",
         "noac_day_count", "nonm_day_count", "n_arr_sched_mvt", "gap_arr_after_sched",
         "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
         "gap_dep_prev_mvt", "gap_dep_next_mvt", "vfr"]
CATS = ["apt", "runway"]
CLF = dict(objective="binary", learning_rate=0.04, num_leaves=8, n_estimators=300,
           min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
           n_jobs=16, verbose=-1, seed=11)
REG = dict(objective="regression", learning_rate=0.04, num_leaves=8, n_estimators=300,
           min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
           n_jobs=16, verbose=-1, seed=11)


def _X(df, cats):
    X = df[FEATS].copy()
    for c in CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cats[c])
    return X


def fit_predict(tr, te):
    """tr, te: noac rows only. Returns dict of predictor name -> prediction on te."""
    cats = {c: pd.Index(sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))) for c in CATS}
    Xtr, Xte = _X(tr, cats), _X(te, cats)
    ytr = tr.y.to_numpy(float)
    o_tr = tr.mvt_sched.to_numpy(float) - ytr            # BLOCK - SCHED
    ph_tr = (np.abs(o_tr) < PH_TOL).astype(int)
    ms_te = te.mvt_sched.to_numpy(float)

    out = {"const-mean": np.full(len(te), ytr.mean())}

    band = [-1e18, 0, 1800, 3600, 7200, 14400, 28800, 1e18]
    bt = pd.cut(tr.mvt_sched, band); be = pd.cut(te.mvt_sched, band)
    gm = pd.Series(ytr).groupby(bt.to_numpy(), observed=False).mean()
    out["band-mean"] = pd.Series(be.to_numpy()).map(gm).astype(float).fillna(ytr.mean()).to_numpy()

    reg_y = lgb.LGBMRegressor(**REG).fit(Xtr, ytr, categorical_feature=CATS)
    out["lgb-direct-y"] = reg_y.predict(Xte)

    reg_o = lgb.LGBMRegressor(**REG).fit(Xtr, o_tr, categorical_feature=CATS)
    out["lgb-direct-offset"] = ms_te - reg_o.predict(Xte)

    clf = lgb.LGBMClassifier(**CLF).fit(Xtr, ph_tr, categorical_feature=CATS)
    p = clf.predict_proba(Xte)[:, 1]
    npx = ph_tr == 0
    reg_np = lgb.LGBMRegressor(**REG).fit(Xtr[npx], o_tr[npx], categorical_feature=CATS)
    o_hat = reg_np.predict(Xte)
    out["mixture-expectation"] = ms_te - (1.0 - p) * o_hat
    out["mixture-hardswitch"] = np.where(p >= 0.5, ms_te, ms_te - o_hat)
    out["_p"] = p
    return out


BEST = dict(REG, num_leaves=8, n_estimators=300, learning_rate=0.04,
            min_child_samples=30, feature_fraction=0.6)
SEEDS = (11, 23, 37, 51, 77)


class NoacPredictor:
    """Fitted on the noac rows of every training month; predicts on any frame.

    Usage from another experiment (e.g. retro-v1)::

        from prc.tail_features import tail_signals
        sig = tail_signals(month_frame)              # has `noac`
        p = NoacPredictor.fit(train_signals_with_y)
        pred[sig.noac == 1] = p.predict(sig[sig.noac == 1])

    Target is the offset `BLOCK - SCHED`; the prediction is
    `mvt_sched - offset_hat`, which is the mixture expectation over the
    placeholder and non-placeholder modes without having to classify them
    (measured equal to the explicit two-part mixture, and better than a hard
    switch — see RESULTS.md finding 10).
    """

    def __init__(self, models, cats):
        self.models, self.cats = models, cats

    @classmethod
    def fit(cls, tr):
        tr = tr[(tr.noac == 1) & tr.y.notna()]
        cats = {c: pd.Index(sorted(set(tr[c].astype(str)))) for c in CATS}
        X = _X(tr, cats)
        o = tr.mvt_sched.to_numpy(float) - tr.y.to_numpy(float)
        models = [lgb.LGBMRegressor(**dict(BEST, seed=s)).fit(X, o, categorical_feature=CATS) for s in SEEDS]
        return cls(models, cats)

    def predict(self, te):
        X = _X(te, self.cats)
        o = np.mean([m.predict(X) for m in self.models], axis=0)
        return te.mvt_sched.to_numpy(float) - o
