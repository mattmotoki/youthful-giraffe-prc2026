"""Structurally different learners on the frozen v2 frame (exp-008).

Every learner sees the same columns and the same residual target, so the
disagreement between their predictions is the learner's, not the frame's.
Each one owns its categorical convention, because that is part of what makes it
a different learner: LightGBM takes integer codes with unseen -> -1 and splits
on them natively, CatBoost takes strings and builds ordered target statistics,
XGBoost takes a pandas category dtype, and the MLP takes an embedding index
with 0 reserved for unseen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

THREADS = int(os.environ.get("PRC_THREADS", "16"))
GPU = os.environ.get("PRC_GPU", "1")

#: Features whose physical relation to taxi-out time has a sign we are willing
#: to assert: more traffic ahead of you is never faster, and a longer gap to the
#: movement before you is never slower.  Used only by the monotone sibling, and
#: only when it fits the taxi time itself -- under the residual target the model
#: output is AOBT minus block time, a recording offset with no such physics
#: (see SPEC.md, "What the monotone constraint can and cannot be asserted on").
MONOTONE = {}
for _w in ("10m", "30m", "60m"):
    for _s in ("before", "after"):
        MONOTONE[f"dep_takeoffs_{_w}_{_s}"] = 1
        MONOTONE[f"rwy_takeoffs_{_w}_{_s}"] = 1
        MONOTONE[f"arr_landings_{_w}_{_s}"] = 1
MONOTONE["queue_ahead_aobt"] = 1
MONOTONE["prev_rwy_gap"] = -1
MONOTONE["prev_apt_gap"] = -1

NA = "\x00NA"
CAT_SENTINEL = "__none__"   # arrow (XGBoost) rejects a NUL byte in a category name


# ------------------------------------------------------------ encoders ----
class CodeEncoder:
    """Integer codes, vocabulary fixed on the fitting frame, unseen -> -1."""

    def __init__(self, cols):
        self.cols = list(cols)
        self.vocab: dict[str, dict] = {}

    def fit(self, X):
        for c in self.cols:
            vals = pd.unique(X[c].astype("object"))
            self.vocab[c] = {v: i for i, v in enumerate(v for v in vals if pd.notna(v))}
        return self

    def transform(self, X):
        out = {}
        for c in X.columns:
            if c in self.vocab:
                out[c] = X[c].astype("object").map(self.vocab[c]).fillna(-1).astype("int32")
            else:
                out[c] = X[c].astype("float32")
        return pd.DataFrame(out, index=X.index)


class StringEncoder:
    """Strings with an explicit missing token; what CatBoost's CTRs want."""

    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, X):
        return self

    def transform(self, X):
        out = {}
        for c in X.columns:
            if c in self.cols:
                out[c] = X[c].astype("object").where(X[c].notna(), NA).astype(str)
            else:
                out[c] = X[c].astype("float32")
        return pd.DataFrame(out, index=X.index)


class CategoryEncoder:
    """pandas category dtype with categories fixed on the fitting frame."""

    def __init__(self, cols):
        self.cols = list(cols)
        self.cats: dict[str, pd.Index] = {}

    def fit(self, X):
        for c in self.cols:
            v = list(pd.unique(X[c].dropna().astype("object")))
            # A column that is entirely missing on the fitting rows (the noac
            # head sees no NM-side aircraft type at all) would give XGBoost a
            # zero-category dtype, which its arrow path cannot build. One
            # sentinel category that no row ever takes keeps the dtype legal
            # and leaves every real row missing, which is the intended reading.
            self.cats[c] = pd.Index(v if v else [CAT_SENTINEL])
        return self

    def transform(self, X):
        out = {}
        for c in X.columns:
            if c in self.cats:
                out[c] = pd.Categorical(X[c].astype("object"), categories=self.cats[c])
            else:
                out[c] = X[c].astype("float32")
        return pd.DataFrame(out, index=X.index)


# ------------------------------------------------------------ learners ----
@dataclass
class Learner:
    """One fit of one algorithm on (features, residual target)."""

    name: str
    params: dict = field(default_factory=dict)

    def fit(self, X, y, cats, rounds, seed, w=None, Xes=None, yes=None, es_rounds=0):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    @property
    def best_iteration(self) -> int | None:
        return getattr(self, "_best", None)


class LgbLearner(Learner):
    """LightGBM.  `variant` picks the sibling: `lin` is retro-v1's linear-leaf
    reference, `mono` drops linear leaves and asserts the congestion signs,
    `huber` drops linear leaves and changes the loss."""

    BASE = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=255,
                min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, max_cat_threshold=32, cat_smooth=10.0, min_data_per_group=100,
                num_threads=THREADS, verbosity=-1, force_row_wise=True, deterministic=True)

    def __init__(self, name, variant="lin", monotone=False, huber_delta=None, params=None):
        super().__init__(name, params or {})
        self.variant, self.monotone, self.huber_delta = variant, monotone, huber_delta

    def _params(self, X, seed):
        p = dict(self.BASE, **self.params)
        p["linear_tree"] = self.variant == "lin"
        if self.variant == "lin":
            p.setdefault("linear_lambda", 1.0)
        if self.huber_delta is not None:
            p["objective"], p["alpha"] = "huber", float(self.huber_delta)
        if self.monotone:
            p["monotone_constraints"] = [MONOTONE.get(c, 0) for c in X.columns]
            p["monotone_constraints_method"] = "advanced"
        p.update(seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
                 data_random_seed=seed + 3, extra_seed=seed + 4, objective_seed=seed + 5,
                 drop_seed=seed + 6)
        return p

    def fit(self, X, y, cats, rounds, seed, w=None, Xes=None, yes=None, es_rounds=0):
        import lightgbm as lgb

        self.enc = CodeEncoder(cats).fit(X)
        T = self.enc.transform(X)
        p = self._params(T, seed)
        dtr = lgb.Dataset(T, label=y, weight=w, categorical_feature=cats)
        valid, cbs = [], [lgb.log_evaluation(200)]
        if Xes is not None and es_rounds:
            valid = [lgb.Dataset(self.enc.transform(Xes), label=yes,
                                 categorical_feature=cats, reference=dtr)]
            cbs.append(lgb.early_stopping(es_rounds, verbose=True))
        self.model = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=valid, callbacks=cbs)
        self._best = self.model.best_iteration or None
        self.feature_names = list(T.columns)
        return self

    def predict(self, X):
        return self.model.predict(self.enc.transform(X),
                                  num_iteration=self.model.best_iteration or None)

    def importance(self):
        return pd.Series(self.model.feature_importance("gain"), index=self.feature_names)


class CatLearner(Learner):
    """CatBoost with native categoricals.  Ordered target statistics are
    CatBoost's default categorical treatment and are kept; the boosting scheme
    is Plain because Ordered boosting is not supported on GPU at this size."""

    def __init__(self, name, depth=8, gpu=True, params=None):
        super().__init__(name, params or {})
        self.depth, self.gpu = depth, gpu

    def fit(self, X, y, cats, rounds, seed, w=None, Xes=None, yes=None, es_rounds=0):
        from catboost import CatBoostRegressor, Pool

        self.enc = StringEncoder(cats).fit(X)
        T = self.enc.transform(X)
        idx = [T.columns.get_loc(c) for c in cats]
        p = dict(loss_function="RMSE", eval_metric="RMSE", iterations=rounds,
                 learning_rate=0.05, depth=self.depth, l2_leaf_reg=3.0,
                 random_seed=seed, verbose=200, allow_writing_files=False,
                 boosting_type="Plain", thread_count=THREADS)
        if self.gpu:
            p.update(task_type="GPU", devices=GPU, max_ctr_complexity=2, border_count=128)
        else:
            p.update(max_ctr_complexity=2)
        p.update(self.params)
        self.model = CatBoostRegressor(**p)
        pool = Pool(T, label=y, weight=w, cat_features=idx)
        ev = None
        if Xes is not None and es_rounds:
            ev = Pool(self.enc.transform(Xes), label=yes, cat_features=idx)
        self.model.fit(pool, eval_set=ev, early_stopping_rounds=es_rounds or None, use_best_model=bool(ev))
        self._best = int(self.model.get_best_iteration()) + 1 if ev is not None else None
        self.feature_names = list(T.columns)
        return self

    def predict(self, X):
        return self.model.predict(self.enc.transform(X))

    def importance(self):
        return pd.Series(self.model.get_feature_importance(), index=self.feature_names)


class XgbLearner(Learner):
    """XGBoost hist with native categorical splits."""

    def __init__(self, name, max_depth=8, gpu=True, params=None):
        super().__init__(name, params or {})
        self.max_depth, self.gpu = max_depth, gpu

    def fit(self, X, y, cats, rounds, seed, w=None, Xes=None, yes=None, es_rounds=0):
        import xgboost as xgb

        self.enc = CategoryEncoder(cats).fit(X)
        T = self.enc.transform(X)
        p = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                 max_depth=self.max_depth, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=20.0, reg_lambda=1.0,
                 max_cat_to_onehot=1, max_cat_threshold=32, seed=seed, nthread=THREADS)
        p["device"] = f"cuda:{GPU}" if self.gpu else "cpu"
        p.update(self.params)
        dtr = xgb.DMatrix(T, label=y, weight=w, enable_categorical=True, nthread=THREADS)
        watch, es = [], None
        if Xes is not None and es_rounds:
            dva = xgb.DMatrix(self.enc.transform(Xes), label=yes, enable_categorical=True, nthread=THREADS)
            watch = [(dtr, "train"), (dva, "es")]
            es = es_rounds
        self.model = xgb.train(p, dtr, num_boost_round=rounds, evals=watch,
                               early_stopping_rounds=es, verbose_eval=200)
        self._best = int(self.model.best_iteration) + 1 if es else None
        self.feature_names = list(T.columns)
        return self

    def predict(self, X):
        import xgboost as xgb

        d = xgb.DMatrix(self.enc.transform(X), enable_categorical=True, nthread=THREADS)
        kw = {}
        if getattr(self.model, "best_iteration", None) is not None and self._best:
            kw["iteration_range"] = (0, self._best)
        return self.model.predict(d, **kw)

    def importance(self):
        g = self.model.get_score(importance_type="total_gain")
        return pd.Series({c: g.get(c, 0.0) for c in self.feature_names})


class MlpLearner(Learner):
    """Embedding MLP: categoricals as learned vectors, numerics standardised
    with an explicit missing indicator, target the residual in seconds / 1000."""

    def __init__(self, name, hidden=(512, 256, 128), emb_cap=32, epochs=8,
                 batch=4096, lr=1e-3, gpu=True, params=None):
        super().__init__(name, params or {})
        self.hidden, self.emb_cap, self.epochs = hidden, emb_cap, epochs
        self.batch, self.lr, self.gpu = batch, lr, gpu
        self.scale = 1000.0

    def _prep_fit(self, X, cats):
        self.cats = list(cats)
        self.nums = [c for c in X.columns if c not in self.cats]
        self.vocab = {}
        for c in self.cats:                       # 0 is reserved for unseen/missing
            vals = pd.unique(X[c].astype("object").dropna())
            self.vocab[c] = {v: i + 1 for i, v in enumerate(vals)}
        N = X[self.nums].to_numpy(dtype="float32")
        self.med = np.nanmedian(N, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0).astype("float32")
        F = np.where(np.isfinite(N), N, self.med)
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0)
        self.sd = np.where(self.sd > 1e-6, self.sd, 1.0).astype("float32")

    def _prep(self, X):
        C = np.stack([X[c].astype("object").map(self.vocab[c]).fillna(0).to_numpy("int64")
                      for c in self.cats], axis=1)
        N = X[self.nums].to_numpy(dtype="float32")
        miss = (~np.isfinite(N)).astype("float32")
        F = np.where(np.isfinite(N), N, self.med)
        F = np.clip((F - self.mu) / self.sd, -10.0, 10.0)
        return C, np.concatenate([F, miss], axis=1).astype("float32")

    def fit(self, X, y, cats, rounds, seed, w=None, Xes=None, yes=None, es_rounds=0):
        """`rounds` is the frozen per-family count, and for this learner the unit
        is an epoch; `self.epochs` is only the fallback when none is passed."""
        import torch
        from torch import nn

        epochs = max(1, int(rounds) if rounds else self.epochs)
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device(f"cuda:{GPU}" if (self.gpu and torch.cuda.is_available()) else "cpu")
        self._prep_fit(X, cats)
        C, F = self._prep(X)
        sizes = [len(self.vocab[c]) + 1 for c in self.cats]
        dims = [min(self.emb_cap, max(4, int(round(s ** 0.5)))) for s in sizes]

        class Net(nn.Module):
            def __init__(self, sizes, dims, n_num, hidden):
                super().__init__()
                self.emb = nn.ModuleList([nn.Embedding(s, d) for s, d in zip(sizes, dims)])
                d0 = sum(dims) + n_num
                layers, prev = [], d0
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
                    prev = h
                layers += [nn.Linear(prev, 1)]
                self.body = nn.Sequential(*layers)

            def forward(self, c, f):
                e = [emb(c[:, i]) for i, emb in enumerate(self.emb)]
                return self.body(torch.cat(e + [f], dim=1)).squeeze(1)

        self.net = Net(sizes, dims, F.shape[1], self.hidden).to(dev)
        self.dev = dev
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        n = len(F)
        steps = max(1, n // self.batch) * epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        Ct = torch.from_numpy(C)
        Ft = torch.from_numpy(F)
        yt = torch.from_numpy((np.asarray(y, dtype="float32") / self.scale))
        rng = np.random.default_rng(seed)
        done = 0
        best_rmse, best_state, self._best = float("inf"), None, None
        Ces = Fes = yes_np = None
        if Xes is not None and es_rounds:
            Ces, Fes = self._prep(Xes)
            yes_np = np.asarray(yes, dtype="float64")
        for ep in range(epochs):
            self.net.train()
            order = rng.permutation(n)
            tot, cnt = 0.0, 0
            for i in range(0, n - self.batch + 1, self.batch):
                idx = torch.from_numpy(order[i:i + self.batch])
                opt.zero_grad(set_to_none=True)
                out = self.net(Ct[idx].to(dev), Ft[idx].to(dev))
                loss = nn.functional.mse_loss(out, yt[idx].to(dev))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                if done < steps:
                    sched.step()
                    done += 1
                tot += float(loss.item()) * len(idx)
                cnt += len(idx)
            msg = f"[mlp] epoch {ep+1}/{epochs} train_rmse={self.scale*np.sqrt(tot/max(cnt,1)):.1f}s"
            if Ces is not None:
                # The early-stop set is the last TRAINING month, never a
                # validation month; an epoch here is what a round is elsewhere.
                pe = self._forward(Ces, Fes)
                r = float(np.sqrt(np.mean((yes_np - pe) ** 2)))
                msg += f" es_rmse={r:.1f}s"
                if r < best_rmse - 1e-9:
                    best_rmse, self._best = r, ep + 1
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                elif ep + 1 - (self._best or 0) >= max(1, es_rounds):
                    print(msg + "  (early stop)", flush=True)
                    break
            print(msg, flush=True)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _forward(self, C, F):
        import torch

        self.net.eval()
        out = np.empty(len(F), dtype="float64")
        with torch.no_grad():
            for i in range(0, len(F), 16384):
                c = torch.from_numpy(C[i:i + 16384]).to(self.dev)
                f = torch.from_numpy(F[i:i + 16384]).to(self.dev)
                out[i:i + 16384] = self.net(c, f).float().cpu().numpy()
        self.net.train()
        return out * self.scale

    def predict(self, X):
        C, F = self._prep(X)
        p = self._forward(C, F)
        self.net.eval()
        return p

    def importance(self):
        return pd.Series(dtype="float64")


def make(kind: str, **kw) -> Learner:
    if kind == "lgb":
        return LgbLearner(kw.pop("name", kind), **kw)
    if kind == "cat":
        return CatLearner(kw.pop("name", kind), **kw)
    if kind == "xgb":
        return XgbLearner(kw.pop("name", kind), **kw)
    if kind == "mlp":
        return MlpLearner(kw.pop("name", kind), **kw)
    raise ValueError(kind)
