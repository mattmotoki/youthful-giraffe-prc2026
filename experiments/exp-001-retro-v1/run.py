"""exp-001 retro-v1: LightGBM taxi-out model on the ranking file's actual columns.

  python run.py --name v1                      # both primary folds + candidate
  python run.py --name v1c --target residual   # one climb step

Each run appends one climb.log line per fold, writes oof/<name>-<fold>.parquet
and result-<name>.json, and (unless --no-submission) a candidate parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import lightgbm as lgb  # noqa: E402

import prc.data as D  # noqa: E402
import guard as guard_mod  # noqa: E402
import lirf_step  # noqa: E402
import phi_step  # noqa: E402
import daystate_step  # noqa: E402
import exot_step  # noqa: E402
from frame import SOURCE_SETS, FrameBuilder  # noqa: E402
import stand_step  # noqa: E402
import surface_step  # noqa: E402
import target_encoding  # noqa: E402
from features import CAT_FEATURES, build_features  # noqa: E402

TAIL_CUT = 3600.0


# ---------------------------------------------------------------- data ----
def frame_for_months(months, version="v1", keep_raw: dict | None = None):
    """Features, target and take-off epoch for training months, month by month."""
    xs, ys, ts = [], [], []
    for m in months:
        df = D.load_month(m)
        if keep_raw is not None:
            keep_raw[m] = df
        dep = (df["PHASE_mvt"] == "DEP").to_numpy()
        xs.append(build_features(df, version))
        ys.append(df.loc[dep, D.TARGET].astype("float64"))
        ts.append(df.loc[dep, "MVT_TIME_UTC_mvt"].astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9)
    return (
        pd.concat(xs, ignore_index=True),
        pd.concat(ys, ignore_index=True).reset_index(drop=True),
        np.concatenate(ts),
    )


def valid_frame(fold: D.Fold, version="v1") -> tuple[pd.DataFrame, pd.Series]:
    """Features from the ranking-masked validation month, plus the hidden labels."""
    _, masked, y_hidden = D.fold_frames(fold)
    X = build_features(masked, version)
    y = y_hidden.loc[X.index].astype("float64")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def ranking_frames(version="v1") -> pd.DataFrame:
    """Ranking features, built separately per ranking month (they are isolated)."""
    r = D.load_ranking()
    return pd.concat([build_features(r[r["month"] == m], version) for m in sorted(r["month"].unique())],
                     ignore_index=True)


# ------------------------------------------------------------ encoding ----
class CatEncoder:
    """Integer codes with a vocabulary fixed on the fitting frame; unseen -> -1."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.vocab: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame) -> "CatEncoder":
        for c in self.cols:
            vals = pd.unique(X[c].astype("object"))
            self.vocab[c] = {v: i for i, v in enumerate(v for v in vals if pd.notna(v))}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in X.columns:
            if c == "MVT_ID_mvt":
                continue
            if c in self.vocab:
                out[c] = X[c].astype("object").map(self.vocab[c]).fillna(-1).astype("int32")
            else:
                out[c] = X[c].astype("float32")
        return pd.DataFrame(out, index=X.index)


# ------------------------------------------------------- target algebra ----
def forward_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.log1p(np.clip(y, 0.0, None))
    if mode == "asinh":
        return np.arcsinh(y)
    return y


def inverse_target(z: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.expm1(z)
    if mode == "asinh":
        return np.sinh(z)
    return z


def base_score(X: pd.DataFrame, fallback: pd.Series | None, mode: str) -> np.ndarray | None:  # noqa: C901
    """init_score for the residual parameterisation.

    target = (take-off - AOBT) - (block - AOBT).  Offsetting by take-off minus
    AOBT leaves the model fitting only the block-versus-AOBT gap, which is what
    the task is really about.  Rows with no NM join have no AOBT, so they get the
    training median target of their airport and the model fits them from zero.
    """
    if mode not in ("residual", "phi"):
        return None
    col = "taxi_hat" if mode == "phi" else "d_takeoff_aobt"
    b = X[col].to_numpy(dtype="float64").copy()
    miss = ~np.isfinite(b)
    if miss.any():
        assert fallback is not None, "an offset mode needs an airport fallback"
        b[miss] = X.loc[miss, "airport"].astype("object").map(fallback).fillna(fallback.median()).to_numpy("float64")
    return b


def airport_fallback(X: pd.DataFrame, y: pd.Series, mode: str) -> pd.Series | None:
    if mode not in ("residual", "phi"):
        return None
    return y.groupby(X["airport"].astype("object").to_numpy()).median()


# --------------------------------------------------------------- model ----
def cat_columns(X: pd.DataFrame) -> list[str]:
    """Every non-numeric column is categorical, so an imported module's string
    columns register themselves instead of having to be listed here."""
    return [c for c in X.columns
            if c != "MVT_ID_mvt" and (X[c].dtype == object or pd.api.types.is_string_dtype(X[c]))]


def train_model(cfg, Xtr, ytr, wtr, base_tr, Xva=None, yva=None, base_va=None, num_round=1000):
    cats = cat_columns(Xtr)
    enc = CatEncoder(cats).fit(Xtr)
    Ttr = enc.transform(Xtr)
    dtr = lgb.Dataset(Ttr, label=ytr, weight=wtr, init_score=base_tr, categorical_feature=cats)
    valid_sets, callbacks = [], [lgb.log_evaluation(500)]
    hist: dict = {}
    if Xva is not None:
        callbacks.append(lgb.record_evaluation(hist))
        dva = lgb.Dataset(
            enc.transform(Xva), label=yva, init_score=base_va,
            categorical_feature=cats, reference=dtr,
        )
        valid_sets = [dva]
        callbacks.append(lgb.early_stopping(cfg["es_rounds"], verbose=True))
    # Every seed LightGBM exposes is pinned, not just the top-level one: the
    # dataset's bin-construction sample and the bagging and feature-fraction
    # draws each have their own generator, and leaving any of them to a default
    # made a fold's result depend on what had been trained earlier in the same
    # process (see the reproducibility section of RESULTS.md).
    params = dict(cfg["params"], seed=cfg["seed"], bagging_seed=cfg["seed"] + 1,
                  feature_fraction_seed=cfg["seed"] + 2, data_random_seed=cfg["seed"] + 3,
                  extra_seed=cfg["seed"] + 4, objective_seed=cfg["seed"] + 5,
                  drop_seed=cfg["seed"] + 6)
    model = lgb.train(params, dtr, num_boost_round=num_round, valid_sets=valid_sets, callbacks=callbacks)
    # The whole validation curve, so the RMSE at any round is a lookup rather than
    # another fit. exp-008 found patience 100 truncating a stop at 405 that
    # patience 900 puts at 4879, so every budget has to be read off a curve that
    # was allowed to run long enough.
    model._prc_curve = list(hist.get("valid_0", {}).get("rmse", []))
    return model, enc, list(Ttr.columns)


def predict_seconds(cfg, model, enc, X, base) -> np.ndarray:
    """Model output mapped back to taxi-out seconds, with the output guards applied."""
    p = model.predict(enc.transform(X), num_iteration=model.best_iteration or None)
    if base is not None:
        p = p + base
    p = inverse_target(p, cfg["target"])
    if cfg["clip_floor"] is not None:
        p = np.maximum(p, cfg["clip_floor"])
    if cfg["clip_ceiling"] is not None:
        p = np.minimum(p, cfg["clip_ceiling"])
    return p


def sample_weights(cfg, y: np.ndarray, X: pd.DataFrame):
    w = _scheme_weights(cfg, y, X)
    hl = float(cfg.get("recency_halflife", 0.0))
    if hl > 0:
        # Recency for a forward task: 2026 is predicted from 2025, so December is
        # the nearest month and January the furthest. Weight halves every `hl`
        # months of distance from the end of the training year. anatomy measures
        # about 4 s on july; it is an arm, not a default.
        age = 12.0 - X["month_num"].to_numpy()
        base = np.ones(len(X)) if w is None else np.asarray(w, dtype="float64")
        w = base * (0.5 ** (age / hl))
    ww = float(cfg.get("winter_weight", 1.0))
    if ww != 1.0:
        # Up-weight winter rather than fitting a winter-only model: exp-007
        # measures a x3 up-weighting at 1.7 s on january, and a winter-only model
        # 15 s worse, so the seasonal signal is worth emphasis and not separation.
        winter = np.isin(X["month_num"].to_numpy(), (12.0, 1.0, 2.0))
        base = np.ones(len(X)) if w is None else np.asarray(w, dtype="float64")
        w = np.where(winter, base * ww, base)
    return w


def _scheme_weights(cfg, y: np.ndarray, X: pd.DataFrame):
    scheme = cfg["weights"]
    if scheme in (None, "none"):
        return None
    if scheme == "gap":  # up-weight rows whose NM off-block sits far from the plan
        g = np.abs(X["d_aobt_lobt"].to_numpy(dtype="float64"))
        g = np.where(np.isfinite(g), g, 0.0)  # no NM join: neutral weight, so this
        #                                        mechanism stays separate from "no_nm"
        return (1.0 + cfg["weight_alpha"] * np.minimum(g, 3600.0) / 3600.0).astype("float64")
    if scheme == "no_nm":  # up-weight the rows with no NM join
        return np.where(X["nm_missing"].to_numpy().astype(bool), cfg["weight_alpha"], 1.0).astype("float64")
    raise ValueError(scheme)


def training_target(cfg, y: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
    """Training-side treatment of out-of-range targets; the scored population is untouched."""
    lo, hi = cfg["train_clip_lo"], cfg["train_clip_hi"]
    if lo is not None:
        y = np.maximum(y, lo)
    if hi is not None:
        y = np.minimum(y, hi)
    c = cfg.get("train_clip_r")
    if c and X is not None:
        # Winsorise the RESIDUAL, symmetrically, before fitting. r = BLOCK - AOBT
        # = (take-off - AOBT) - taxi, so clipping r and mapping back moves the
        # label rather than dropping the row. This is a different operation from
        # the prediction-side guard (which clips a finished prediction) and from
        # dropping extreme rows (which changes the training population): every row
        # still trains, with a label pulled to the edge of the plausible range.
        a_ = X["d_takeoff_aobt"].to_numpy(dtype="float64")
        ok = np.isfinite(a_)
        r = a_[ok] - y[ok]
        y = y.copy()
        y[ok] = a_[ok] - np.clip(r, -float(c), float(c))
    return y


# -------------------------------------------------------------- report ----
def decompose(y, p, no_nm, noac=None, reach=None) -> dict:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    se = (y - p) ** 2
    bulk = y <= TAIL_CUT
    return {
        "rmse": float(np.sqrt(se.mean())),
        "rmse_bulk": float(np.sqrt(se[bulk].mean())),
        "rmse_tail": float(np.sqrt(se[~bulk].mean())) if (~bulk).any() else float("nan"),
        "rmse_no_nm": float(np.sqrt(se[no_nm].mean())) if no_nm.any() else float("nan"),
        "n": int(len(y)),
        "n_tail": int((~bulk).sum()),
        "n_no_nm": int(no_nm.sum()),
        "sse_share_tail": float(se[~bulk].sum() / se.sum()),
        "sse_share_no_nm": float(se[no_nm].sum() / se.sum()),
        "bias": float((p - y).mean()),
        "rmse_clean": float(np.sqrt(se[reach].mean())) if reach is not None and reach.any() else float("nan"),
        "rmse_noac": float(np.sqrt(se[noac].mean())) if noac is not None and noac.any() else float("nan"),
        "n_noac": int(noac.sum()) if noac is not None else 0,
        "sse_share_noac": float(se[noac].sum() / se.sum()) if noac is not None and noac.any() else float("nan"),
    }


def _specs(cfg):
    return target_encoding.SPECS if cfg["encodings"] == "all" else target_encoding.TAXI_ONLY


def _guard_bounds(cfg, months):
    """Guard bounds, derived from the training months unless explicitly fixed."""
    if cfg["guard_bounds"] == "fixed":
        return guard_mod.LO, guard_mod.HI
    # Symmetric 0.01 % tails, chosen a priori rather than reverse-engineered from
    # the hand-picked [-10800, 3600]. The quantile IS a choice: on twelve training
    # months the observed range of r is [-20657, 9069], so `trainrange` clips
    # nothing at all on the ranking set (0 rows against 195), which makes the
    # guard literally correct and operationally inert. What the fold evidence
    # actually supports is shrinking the sparse tail, not rejecting the unobserved.
    q = {"trainrange": (0.0, 1.0), "trainq": (1e-4, 1 - 1e-4)}[cfg["guard_bounds"]]
    return guard_mod.bounds_from_months(months, *q)


def shared_module_provenance() -> dict:
    """Every prc/ default this experiment reads but cannot pass as an argument.

    Recorded in each result file so that a change upstream is visible in the
    record rather than showing up as an unexplained shift in a fold number.
    `surface_features.BUCKET` moved from 1800 to 1200 between two runs of this
    experiment with no signal that anything had changed, which is the reason the
    block exists.
    """
    out = {}
    try:
        import prc.surface_features as SF
        out["surface"] = {"UNIMPEDED_Q": SF.UNIMPEDED_Q, "WINDOWS": list(SF.WINDOWS),
                          "BUCKET": SF.BUCKET, "NBR_WINDOW": SF.NBR_WINDOW,
                          "SHRINK_K": getattr(SF, "SHRINK_K", None),
                          "n_families": len(SF.FAMILY_COLUMNS)}
    except Exception as e:  # pragma: no cover - provenance must never break a run
        out["surface"] = f"unavailable: {e}"
    try:
        import prc.phi_features as PF
        d = PF.PhiModel()
        out["phi"] = {"default_weight": d.weight, "default_shrink_rows": d.shrink_rows,
                      "default_min_rows": d.min_rows, "keys": [list(k) for k in d.keys]}
    except Exception as e:
        out["phi"] = f"unavailable: {e}"
    try:
        import prc.lirf_features as LF
        out["lirf"] = {"cat_features": list(LF.LIRF_CAT_FEATURES)}
    except Exception as e:
        out["lirf"] = f"unavailable: {e}"
    return out


def shared_module_tag() -> str:
    """Compact provenance for a climb line: the prc/ constants a run depended on.

    Until the shared modules take these as constructor arguments and carry a
    version, the only defence against a silent drift is to record what was
    imported. `surface_features.BUCKET` moved from 1800 to 1200 between two runs
    of this experiment with no signal, so a climb line that does not say which
    value it used cannot be compared with one that does.
    """
    bits = []
    for name, attrs in (("data", ("VERSION", "LIVE_RANKING_BYTES", "PRIMARY_FOLDS")),
                        ("surface_features", ("VERSION", "BUCKET", "UNIMPEDED_Q")),
                        ("phi_features", ("VERSION",)),
                        ("lirf_features", ("VERSION",)),
                        ("tail_features", ("VERSION",)),
                        ("tail_model", ("VERSION",)),
                        ("stand_features", ("VERSION",)),
                        ("weather_features", ("VERSION",)),
                        ("day_state_features", ("VERSION",))):
        try:
            mod = __import__(f"prc.{name}", fromlist=["x"])
        except Exception:
            continue
        got = [f"{a}={str(getattr(mod, a))[:12]}" for a in attrs if hasattr(mod, a)]
        if name == "phi_features":
            d = mod.PhiModel()
            got.append(f"phidef={d.weight}/{d.shrink_rows:g}/{d.min_rows}")
        if got:
            bits.append(f"{name.split('_')[0]}[{','.join(got)}]")
    return " ".join(bits) or "none"


#: this worker's share of the desk's 40-thread box-wide cap (docs/COMPUTE.md, compute allocation)
THREAD_ALLOCATION = 12
LOAD_HOLD = 140.0


def check_compute_budget() -> None:
    """Refuse to start a fit that would break the desk's compute contract.

    A hard cap is only hard if something enforces it, and this experiment has
    twice drifted over its allocation by leaving two queues running at once.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(min(int(os.environ.get(var, THREAD_ALLOCATION)), THREAD_ALLOCATION))
    n = int(os.environ.get("PRC_THREADS", THREAD_ALLOCATION))
    if n > THREAD_ALLOCATION:
        raise SystemExit(f"PRC_THREADS={n} exceeds this worker's allocation of {THREAD_ALLOCATION}")
    load1 = float(Path("/proc/loadavg").read_text().split()[0])
    if load1 > LOAD_HOLD:
        raise SystemExit(f"1-minute load {load1:.0f} is over the {LOAD_HOLD:.0f} hold threshold; not starting")
    print(f"[compute] {n} threads, 1-minute load {load1:.0f}", flush=True)


def drop_rollover(Xtr, ytr, ttr):
    """Remove the unreached-rollover rows from the MAIN model's training set.

    `prc.data.keep_reachable` (prc/data.py VERSION 6) is the published recipe and
    is used whenever the frame it is given is a movement frame. This experiment's
    training frame is a *feature* frame with no `PHASE_mvt` column, which
    `keep_reachable` cannot consume, so the fallback derives the same ids from the
    same published constant, `UNREACHED_ROLLOVER` — the same four rows by the same
    definition, not a hand-copied list. Their label is a rollover artefact while
    every timing column is ordinary, so no input reaches them; two are the LFPG
    pair this experiment found and handed to the tail worker.

    Three constraints, all of them from the tail worker's warnings:

    * **Training only.** The rows stay in the scored population; that is what
      `rmse_clean` in `decompose` is for.
    * **Not the noac head.** One of the four is a noac row, and the band head is
      fitted on rollover rows by design. `run_fold` keeps an unfiltered frame for
      the head and filters only what the main model sees.
    * **Not `clean_training_labels`.** Its window drops two rows and misses two
      rollover-class rows at 58,206 s and 31,268 s.
    """
    keep = D.keep_reachable(Xtr).to_numpy() if "PHASE_mvt" in Xtr.columns else \
        ~Xtr["MVT_ID_mvt"].isin({r["id"] for rows in D.UNREACHED_ROLLOVER.values() for r in rows}).to_numpy()
    print(f"[rollover] dropped {int((~keep).sum())} of {len(Xtr)} main-model training rows", flush=True)
    return Xtr[keep].reset_index(drop=True), ytr[keep].reset_index(drop=True), ttr[keep]


def _absurdity_check(name: str, fold: str, path, rmse: float) -> None:
    """Two cheap checks that only fire when something is obviously wrong.

    Both of the confounds this experiment has hit were caught by a number looking
    wrong rather than by a guard: two arms coming out identical to the decimal
    (a source list that had drifted, making one command a duplicate of another),
    and a fold RMSE far outside its plausible range. Guards only cover the
    failure modes already met, so these cover the shape of "obviously wrong"
    instead.
    """
    import hashlib

    if not (80.0 < rmse < 3000.0):
        print(f"[ABSURD] {name} {fold} RMSE {rmse:.1f} is outside 80-3000; check the row alignment",
              flush=True)
    h = hashlib.sha256(pd.read_parquet(path)["y_pred"].to_numpy().tobytes()).hexdigest()[:16]
    reg = path.parent / "_pred_hashes.json"
    seen = json.loads(reg.read_text()) if reg.exists() else {}
    key = f"{fold}:{h}"
    if key in seen and seen[key] != name:
        print(f"[ABSURD] {name} {fold} predictions are bit-identical to {seen[key]}; "
              f"two runs that should differ do not", flush=True)
    seen.setdefault(key, name)
    reg.write_text(json.dumps(seen, indent=0, sort_keys=True))


def _reseed(seed: int) -> None:
    """Put every process-global generator back to a known state at each fold."""
    random.seed(seed)
    np.random.seed(seed)


def _refit_only(cfg, Xtr, yt, w, base, targets, refit):
    """Pass two alone, at a budget chosen elsewhere."""
    n = max(1, int(cfg.get("seed_bag", 1)))
    acc, imp = None, None
    for i in range(n):
        c = dict(cfg, seed=cfg["seed"] + 100 * i)
        model, enc, feats = train_model(c, Xtr, yt, w, base, num_round=refit)
        ps = [predict_seconds(c, model, enc, X, b) for X, b in targets]
        acc = ps if acc is None else [x + y for x, y in zip(acc, ps)]
        g = pd.Series(model.feature_importance("gain"), index=feats)
        imp = g if imp is None else imp + g
    return [x / n for x in acc], refit, refit, imp / n


def _sel(a, m):
    """Mask an optional array."""
    return None if a is None else a[m]


def _es_slice(cfg, Xtr, ttr):
    """The early-stopping rows, and what fraction of training they are.

    `frac` holds out the last 15 % of training rows by take-off time. `lastmonth`
    holds out the final whole training month instead: the fold's own task is to
    generalise to an unseen *month*, and a slice that cuts mid-month shares its
    weather, its schedule and its congestion with the rows still in training,
    which flatters it. H1 stopping at 64 rounds on july where H0 stopped at 196
    is what put the question on the table.
    """
    if cfg["es_mode"] == "lastmonth":
        m = Xtr["month_num"].to_numpy()
        es = m == m.max()
    else:
        es = ttr >= np.quantile(ttr, 1.0 - cfg["es_frac"])
    return es, float(es.mean())


def two_pass(cfg, Xtr, yt, w, base, es, targets):
    if cfg["fixed_rounds"]:
        # With the budget already fixed, the early-stopping pass trains a whole
        # model only to throw its round count away. Skipping it halves the cost of
        # every fixed-rounds run, which is most of them now that the budget is
        # chosen once across three held-out months rather than per fold.
        return _refit_only(cfg, Xtr, yt, w, base, targets, cfg["fixed_rounds"])
    """Early-stop on the `es` rows, refit on all of them, predict each target.

    `targets` is a list of (frame, init_score) pairs; the return is one
    prediction array per target, in seconds.
    """
    model, enc, _ = train_model(
        cfg, Xtr[~es], yt[~es], _sel(w, ~es), _sel(base, ~es),
        Xva=Xtr[es], yva=yt[es], base_va=_sel(base, es), num_round=cfg["max_rounds"],
    )
    best = model.best_iteration or cfg["max_rounds"]
    curve = getattr(model, "_prc_curve", [])
    if curve:
        c = np.asarray(curve, dtype="float64")
        k = int(np.argmin(c))
        marks = [m for m in (100, 200, 300, 400, 472, 600, 883, 1000, 2000, 4000, 8000) if m <= len(c)]
        print(f"[curve] len={len(c)} argmin={k + 1} min={c[k]:.4f} | "
              + " ".join(f"{m}:{c[m - 1]:.2f}" for m in marks), flush=True)
        (HERE / f"curve-{cfg['name']}.json").write_text(json.dumps({"curve": curve}))
    held = float(np.mean(es))
    refit = cfg["fixed_rounds"] or max(50, int(round(best / (1.0 - held))))
    # Seed bagging: the refit is repeated at n seeds and the predictions averaged.
    # Round selection stays on the first seed, because the point of bagging is to
    # average away the seed's effect on the fitted trees, not to average away the
    # early-stopping draw as well.
    n = max(1, int(cfg.get("seed_bag", 1)))
    acc, imp = None, None
    for i in range(n):
        c = dict(cfg, seed=cfg["seed"] + 100 * i)
        model, enc, feats = train_model(c, Xtr, yt, w, base, num_round=refit)
        ps = [predict_seconds(c, model, enc, X, b) for X, b in targets]
        acc = ps if acc is None else [a + b for a, b in zip(acc, ps)]
        g = pd.Series(model.feature_importance("gain"), index=feats)
        imp = g if imp is None else imp + g
    preds = [a / n for a in acc]
    return preds, best, refit, imp / n


def noac_head(cfg, Xtr, ytr, targets):
    """A model for the placeholder-block-time rows alone, anchored on the schedule.

    A departure with no AIRCRAFT_TYPE_mvt ("noac") often has the scheduled
    off-block recorded as its block time instead of an actual one, so its target
    is take-off minus schedule.  Offsetting by take-off minus schedule turns the
    head's job into "is this row a placeholder, and if not, how late was the real
    off-block", which is a well-posed small problem.  These rows are 0.18 % of
    departures and 60 % of v1's squared error, so they get their own resolution
    instead of a few leaves in a model fitted on two million ordinary flights.
    """
    m = Xtr["noac"].to_numpy().astype(bool)
    if m.sum() < cfg["noac_min_rows"]:
        print(f"[noac] only {m.sum()} training rows, head skipped", flush=True)
        return [(np.zeros(len(X), dtype=bool), None) for X in targets]
    params = dict(cfg["params"], num_leaves=cfg["noac_leaves"], min_data_in_leaf=cfg["noac_min_data"],
                  learning_rate=cfg["noac_lr"], bagging_fraction=1.0, bagging_freq=0, feature_fraction=0.8,
                  max_cat_threshold=16, cat_smooth=5.0, min_data_per_group=5, linear_tree=False)
    cfg_h = dict(cfg, params=params, target="raw", clip_ceiling=None)
    base = Xtr.loc[m, "d_takeoff_sched"].to_numpy(dtype="float64")
    model, enc, _ = train_model(cfg_h, Xtr[m], ytr[m], None, base, num_round=cfg["noac_rounds"])
    out = []
    for X in targets:
        mm = X["noac"].to_numpy().astype(bool)
        if not mm.any():
            out.append((mm, np.empty(0)))
            continue
        out.append((mm, predict_seconds(cfg_h, model, enc, X[mm], X.loc[mm, "d_takeoff_sched"].to_numpy("float64"))))
    print(f"[noac] head on {m.sum()} training rows, {cfg['noac_rounds']} rounds", flush=True)
    return out


# ----------------------------------------------------------------- main ----
def budget_scan(cfg, fold_name: str) -> dict:
    """One pass: train on the fold's eleven months, watch the held-out month.

    This measures where THAT month's optimum lies and nothing else -- no refit, no
    out-of-fold predictions, no climb line. It exists because a round count chosen
    on a single early-stop month does not transfer: the blend owner measured a
    count selected on 2025-11 scoring 3.6 s worse on July at 8,725 rounds than at
    2,100, entirely in the bulk. The submission budget is the median of these
    per-month optima across seasons, with the spread reported, and the smaller
    count preferred when the spread is wide.
    """
    fold = D.FOLDS[fold_name]
    _reseed(cfg["seed"])
    raw = {} if (cfg["phi"] or cfg["surface"] or cfg["lirf"] or cfg["stand"] or cfg["deice"] or cfg["exot"]) else None
    Xtr, ytr, _ = frame_for_months(fold.train_months, cfg["features"], keep_raw=raw)
    Xva, yva = valid_frame(fold, cfg["features"])
    if cfg["deice"]:
        _, masked, _ = D.fold_frames(fold)
        Xtr, (Xva,) = daystate_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)])
    fb = airport_fallback(Xtr, ytr, cfg["target"])
    assert len(yva) == len(Xva) == len(pv), f"valid rows disagree: y {len(yva)}, X {len(Xva)}, pred {len(pv)}"
    keep = yva.notna().to_numpy()
    model, _, _ = train_model(
        cfg, Xtr, forward_target(training_target(cfg, ytr.to_numpy(), Xtr), cfg["target"]),
        sample_weights(cfg, ytr.to_numpy(), Xtr), base_score(Xtr, fb, cfg["target"]),
        Xva=Xva[keep], yva=forward_target(yva.to_numpy()[keep], cfg["target"]),
        base_va=_sel(base_score(Xva, fb, cfg["target"]), keep), num_round=cfg["max_rounds"],
    )
    c = np.asarray(getattr(model, "_prc_curve", []), dtype="float64")
    k = int(np.argmin(c)) + 1 if c.size else -1
    (HERE / f"curve-scan-{fold_name}.json").write_text(json.dumps({"fold": fold_name, "curve": c.tolist()}))
    print(f"[scan {fold_name}] optimum {k} at {c.min():.3f}; "
          + " ".join(f"{m}:{c[m - 1]:.2f}" for m in (300, 537, 800, 1200, 2000, 4000) if m <= len(c)), flush=True)
    return {"fold": fold_name, "optimum": k, "rmse_at_optimum": float(c.min()), "curve_len": int(len(c))}


def build_v3(cfg, months, targets):
    """The v3 frame for a fold or the submission.

    The kept-column list is frozen on the LAST training month, never on the month
    being predicted, and the same list is then imposed on every frame so training
    and serving carry identical columns in identical order.
    """
    raw = {m: D.load_month(m) for m in months}
    fb = FrameBuilder("v3", sources=SOURCE_SETS[cfg["frame_sources"]]).fit(raw)
    fb.learn_dedup(raw[months[-1]], months[-1])
    Xtr = pd.concat([fb.transform(raw[m], m) for m in months], ignore_index=True)
    ytr = pd.concat([raw[m].loc[raw[m]["PHASE_mvt"].eq("DEP"), D.TARGET].astype("float64")
                     for m in months], ignore_index=True)
    ttr = np.concatenate([raw[m].loc[raw[m]["PHASE_mvt"].eq("DEP"), "MVT_TIME_UTC_mvt"]
                          .astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9 for m in months])
    print(f"[v3] {Xtr.shape}, {len(fb.dropped)} columns deduplicated", flush=True)
    assert len(Xtr) == len(ytr) == len(ttr), f"train rows disagree: {len(Xtr)}, {len(ytr)}, {len(ttr)}"
    out = [fb.transform(t, None) for t in targets]
    for t, o in zip(targets, out):
        n = int(t["PHASE_mvt"].eq("DEP").sum())
        assert len(o) == n, f"target frame has {len(o)} rows for {n} departures"
        assert list(o.columns) == list(Xtr.columns), "train and target columns differ"
    return Xtr, ytr, ttr, out, fb


def run_fold(cfg, fold_name: str) -> dict:
    fold = D.FOLDS[fold_name]
    _reseed(cfg["seed"])
    t0 = time.time()
    if cfg["frame"] == "v3":
        _, masked, y_hidden = D.fold_frames(fold)
        Xtr, ytr, ttr, (Xva,), _ = build_v3(cfg, list(fold.train_months), [masked])
        # y_hidden is indexed like the whole month frame, arrivals included, and
        # the frame builder returns departure rows only. Taking it whole cost a
        # two-hour fold to an IndexError after the fit had finished.
        yva = y_hidden[masked["PHASE_mvt"].eq("DEP")].reset_index(drop=True)
        raw = None
    else:
        raw = {} if (cfg["phi"] or cfg["surface"] or cfg["lirf"] or cfg["stand"] or cfg["deice"] or cfg["exot"]) else None
        Xtr, ytr, ttr = frame_for_months(fold.train_months, cfg["features"], keep_raw=raw)
        Xva, yva = valid_frame(fold, cfg["features"])
    if raw is not None and (cfg["phi"] or cfg["surface"] or cfg["lirf"] or cfg["stand"] or cfg["deice"] or cfg["exot"]):
        _, masked, _ = D.fold_frames(fold)
        if cfg["exot"]:
            Xtr, (Xva,) = exot_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)])
        if cfg["deice"]:
            Xtr, (Xva,) = daystate_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)])
        if cfg["stand"]:
            Xtr, (Xva,) = stand_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)])
        if cfg["lirf"]:
            Xtr, (Xva,) = lirf_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)], gate=cfg["lirf_gate"])
        if cfg["phi"]:
            Xtr, (Xva,) = phi_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)],
                                                  mode=cfg["phi_fit"], shrink=cfg["phi_shrink"])
        if cfg["surface"]:
            Xtr, (Xva,) = surface_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)])
    if cfg["encodings"] != "none":
        # KNOWN DEFECT (gpt-5.6-sol, 2026-09-05): the encoder is fitted here,
        # before two_pass splits off the early-stopping rows, so a training row
        # can carry an aggregate built partly from early-stop rows and the round
        # count is chosen against a slightly optimistic metric. Validation and
        # ranking labels never enter it, so this is a round-selection bias, not
        # a leak. The repair is to fit the encoder inside two_pass, once per
        # pass, on exactly the rows that pass fits. Until then no encoded run is
        # promotable, which is why the flag has to be forced.
        if not cfg["allow_unisolated_encodings"]:
            raise SystemExit(
                "--encodings is not isolated from the early-stop split; pass "
                "--allow-unisolated-encodings to run it as a screen only, and do "
                "not promote the result"
            )
        Xtr, (Xva,) = target_encoding.add(Xtr, ytr.to_numpy(), [Xva], _specs(cfg))
    print(f"[{fold_name}] train {Xtr.shape} valid {Xva.shape} built in {time.time()-t0:.0f}s", flush=True)

    Xtr_head, ytr_head = Xtr, ytr          # the noac head is fitted on rollover rows by design
    if cfg["drop_rollover"]:
        Xtr, ytr, ttr = drop_rollover(Xtr, ytr, ttr)
    es, frac = _es_slice(cfg, Xtr, ttr)
    print(f"[{fold_name}] early-stop slice ({cfg['es_mode']}): {es.sum()} rows, {frac:.1%} of train", flush=True)

    fb = airport_fallback(Xtr, ytr, cfg["target"])
    base = base_score(Xtr, fb, cfg["target"])
    yt = forward_target(training_target(cfg, ytr.to_numpy(), Xtr), cfg["target"])
    w = sample_weights(cfg, ytr.to_numpy(), Xtr)

    base_va = base_score(Xva, fb, cfg["target"])
    if cfg["per_airport"]:
        pv = np.full(len(Xva), np.nan)
        bests, refits, imps = {}, {}, []
        for a in sorted(pd.unique(Xtr["airport"].to_numpy())):
            mtr = Xtr["airport"].to_numpy() == a
            mva = Xva["airport"].to_numpy() == a
            (p,), b, rf, imp = two_pass(
                cfg, Xtr[mtr], yt[mtr], _sel(w, mtr), _sel(base, mtr), es[mtr],
                [(Xva[mva], _sel(base_va, mva))],
            )
            pv[mva] = p
            bests[a], refits[a] = int(b), rf
            imps.append(imp)
        best, refit = int(np.mean(list(bests.values()))), int(np.mean(list(refits.values())))
        imp = pd.concat(imps, axis=1).sum(axis=1)
        print(f"[{fold_name}] per-airport rounds {refits}", flush=True)
    else:
        (pv,), best, refit, imp = two_pass(cfg, Xtr, yt, w, base, es, [(Xva, base_va)])
        print(f"[{fold_name}] best_iteration {best} -> refit {refit}", flush=True)

    if cfg["noac_head"]:
        (mm, ph), = noac_head(cfg, Xtr_head, ytr_head.to_numpy(), [Xva])
        if ph is not None and mm.any():
            pv[mm] = ph
    if cfg["residual_guard"]:
        lo, hi = _guard_bounds(cfg, fold.train_months)
        av = Xva["d_takeoff_aobt"].to_numpy("float64")
        n = guard_mod.count_touched(pv, av, lo, hi)
        pv = guard_mod.apply(pv, av, lo, hi, floor=cfg["clip_floor"])
        print(f"[{fold_name}] residual guard [{lo:.0f}, {hi:.0f}] touched {n} rows", flush=True)

    assert len(yva) == len(Xva) == len(pv), f"valid rows disagree: y {len(yva)}, X {len(Xva)}, pred {len(pv)}"
    keep = yva.notna().to_numpy()
    no_nm = Xva["nm_missing"].to_numpy().astype(bool)
    noac = Xva["noac"].to_numpy().astype(bool) if "noac" in Xva.columns else None
    ids = {r["id"] for rows in D.UNREACHED_ROLLOVER.values() for r in rows}
    reach = ~Xva["MVT_ID_mvt"].isin(ids).to_numpy()
    res = decompose(yva.to_numpy()[keep], pv[keep], no_nm[keep],
                    None if noac is None else noac[keep], reach[keep])
    res.update(fold=fold_name, best_iteration=int(best), refit_rounds=refit, seconds=round(time.time() - t0))

    out = HERE / "oof" / f"{cfg['name']}-{fold_name}.parquet"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        "MVT_ID_mvt": Xva["MVT_ID_mvt"].to_numpy()[keep], "fold": fold_name,
        "y_true": yva.to_numpy()[keep], "y_pred": pv[keep],
        "nm_missing": no_nm[keep].astype("int8"), "airport": Xva["airport"].to_numpy()[keep],
    }).to_parquet(out, index=False)
    res["oof"] = str(out.relative_to(ROOT))
    _absurdity_check(cfg["name"], fold_name, out, res["rmse"])
    (HERE / f"importance-{cfg['name']}-{fold_name}.csv").write_text(
        imp.sort_values(ascending=False).to_csv(header=["gain"])
    )
    return res


def build_submission(cfg, folds: dict) -> dict:
    if cfg["per_airport"]:
        raise NotImplementedError("per-airport submission needs per-airport round counts; screen on folds first")
    t0 = time.time()
    _reseed(cfg["seed"])
    raw = {} if (cfg["phi"] or cfg["surface"] or cfg["lirf"] or cfg["stand"] or cfg["deice"] or cfg["exot"]) else None
    # The ranking file holds January and July 2026 as two isolated months, so every
    # step that aggregates across rows must see them separately. Passing the whole
    # file let a nearest-event or day-level feature reach across the six-month gap
    # between them; gpt-5.6-sol caught it on the c8 review before either family
    # had shipped in a candidate.
    rr = D.load_ranking()
    r_months = sorted(rr["month"].unique())
    r_frames = [rr[rr["month"] == m] for m in r_months]
    if cfg["frame"] == "v3":
        # The submission must be built on the same frame as the folds. It was not:
        # build_submission called frame_for_months and produced a 50-column fit
        # while the folds ran on 171, which is why the first member attempt died
        # with KeyError: 'noac' instead of shipping a wrong-frame member silently.
        Xtr, ytr, ttr, ranked, _ = build_v3(cfg, list(D.MONTHS), r_frames)
        pairs = list(zip(ranked, r_frames))
    else:
        Xtr, ytr, ttr = frame_for_months(D.MONTHS, cfg["features"], keep_raw=raw)
        pairs = [(build_features(f, cfg["features"]).reset_index(drop=True), f) for f in r_frames]
    Xtr_head, ytr_head = Xtr, ytr          # the noac head is fitted on rollover rows by design
    if cfg["drop_rollover"]:
        Xtr, ytr, ttr = drop_rollover(Xtr, ytr, ttr)
    fold_rounds = int(round(np.mean([r["refit_rounds"] for r in folds.values()]) * cfg["full_round_scale"]))
    if cfg["frame"] != "v3" and (cfg["phi"] or cfg["surface"] or cfg["lirf"] or cfg["stand"]
                                 or cfg["deice"] or cfg["exot"]):
        if cfg["exot"]:
            Xtr, outs = exot_step.fit_transform(D.MONTHS, Xtr, raw, pairs)
            pairs = list(zip(outs, [r for _, r in pairs]))
        if cfg["deice"]:
            Xtr, outs = daystate_step.fit_transform(D.MONTHS, Xtr, raw, pairs)
            pairs = list(zip(outs, [r for _, r in pairs]))
        if cfg["stand"]:
            Xtr, outs = stand_step.fit_transform(D.MONTHS, Xtr, raw, pairs)
            pairs = list(zip(outs, [r for _, r in pairs]))
        if cfg["lirf"]:
            Xtr, outs = lirf_step.fit_transform(D.MONTHS, Xtr, raw, pairs, gate=cfg["lirf_gate"])
            pairs = list(zip(outs, [r for _, r in pairs]))
        if cfg["phi"]:
            Xtr, outs = phi_step.fit_transform(D.MONTHS, Xtr, raw, pairs,
                                               mode=cfg["phi_fit"], shrink=cfg["phi_shrink"])
            pairs = list(zip(outs, [r for _, r in pairs]))
        if cfg["surface"]:
            Xtr, outs = surface_step.fit_transform(D.MONTHS, Xtr, raw, pairs)
            pairs = list(zip(outs, [r for _, r in pairs]))
    Xr = pd.concat([X for X, _ in pairs], ignore_index=True)
    if cfg["encodings"] != "none":
        Xtr, (Xr,) = target_encoding.add(Xtr, ytr.to_numpy(), [Xr], _specs(cfg))
    fb = airport_fallback(Xtr, ytr, cfg["target"])
    yt = forward_target(training_target(cfg, ytr.to_numpy(), Xtr), cfg["target"])
    w = sample_weights(cfg, ytr.to_numpy(), Xtr)
    base = base_score(Xtr, fb, cfg["target"])
    base_r = base_score(Xr, fb, cfg["target"])
    if cfg["full_early_stop"]:
        # The folds disagree about the round count by up to a factor of five, so
        # averaging them is a guess. The full fit runs the same two-pass
        # procedure the folds run — early stop on the last 15 % of the twelve
        # training months by take-off time, refit on all of them — which uses no
        # validation month and no ranking data, and adapts to the data it is
        # actually fitted on.
        es, frac = _es_slice(cfg, Xtr, ttr)
        print(f"[full] {Xtr.shape}, early stop ({cfg['es_mode']}) on {es.sum()} rows, {frac:.1%}"
              f" (fold mean would have been {fold_rounds})", flush=True)
        (pr,), best, rounds, _ = two_pass(cfg, Xtr, yt, w, base, es, [(Xr, base_r)])
        print(f"[full] best_iteration {best} -> refit {rounds}", flush=True)
    else:
        rounds = fold_rounds
        print(f"[full] {Xtr.shape}, {rounds} rounds from the fold mean", flush=True)
        model, enc, _ = train_model(cfg, Xtr, yt, w, base, num_round=rounds)
        pr = predict_seconds(cfg, model, enc, Xr, base_r)
    if cfg["noac_head"]:
        (mm, ph), = noac_head(cfg, Xtr_head, ytr_head.to_numpy(), [Xr])
        if ph is not None and mm.any():
            pr[mm] = ph
    guard_touched = guard_bounds = None
    if cfg["residual_guard"]:
        lo, hi = _guard_bounds(cfg, D.MONTHS)
        guard_bounds = [lo, hi]
        ar = Xr["d_takeoff_aobt"].to_numpy("float64")
        guard_touched = guard_mod.count_touched(pr, ar, lo, hi)
        pr = guard_mod.apply(pr, ar, lo, hi, floor=cfg["clip_floor"])
        print(f"[full] residual guard [{lo:.0f}, {hi:.0f}] touched {guard_touched} of {len(pr)} rows", flush=True)
    blend = None
    if cfg["blend_tail"]:
        # Blend with the tail worker's regime-routed model. The weights come from
        # blend_weights.json, fitted on the folds by blend_eval.py; the honest
        # estimate of what this buys is that script's rotated number, not any
        # figure measured on a fold whose weights were fitted on itself.
        from prc.tail_model import TailModel

        if cfg["blend_weight"] is not None:
            # A flat weight, chosen without looking at any fold. The rotation in
            # blend_eval.py has it beating per-regime weights fitted on two folds
            # and applied to the third (july 321.3 against 325.4), which is the
            # usual verdict when the fitted weights are noisier than the prior.
            w = {"_global": float(cfg["blend_weight"])}
        else:
            w = json.loads((HERE / "blend_weights.json").read_text())
        tm = TailModel().fit(D.load_months(D.MONTHS))
        tp = tm.predict(D.load_ranking()).set_index("MVT_ID_mvt")
        ids = Xr["MVT_ID_mvt"].to_numpy()
        theirs = tp["y_pred"].reindex(ids).to_numpy(dtype="float64")
        reg = tp["regime"].reindex(ids).astype("object")
        wv = reg.map({k: v for k, v in w.items() if not k.startswith("_")}).fillna(w["_global"]).to_numpy(dtype="float64")
        ok = np.isfinite(theirs)
        blend = dict(weights=w, n_blended=int(ok.sum()), n_unmatched=int((~ok).sum()),
                     regimes={k: int(v) for k, v in reg.value_counts().items()})
        print(f"[blend] {ok.sum()} of {len(ids)} ranking rows blended, weights {w}", flush=True)
        pr = np.where(ok, wv * np.where(ok, theirs, 0.0) + (1 - wv) * pr, pr)
        if cfg["clip_floor"] is not None:
            pr = np.maximum(pr, cfg["clip_floor"])

    path = ROOT / "submissions" / "candidates" / f"exp-001-retro-{cfg['name']}.parquet"
    digest = D.write_submission(pd.Series(pr, index=Xr["MVT_ID_mvt"].to_numpy()), path)
    q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    return {
        "path": str(path.relative_to(ROOT)), "sha256": digest, "rounds": rounds,
        "fold_mean_rounds": fold_rounds, "full_early_stop": cfg["full_early_stop"], "blend": blend,
        "guard_touched": guard_touched, "guard_bounds": guard_bounds,
        "seconds": round(time.time() - t0),
        "pred_quantiles": {str(k): float(v) for k, v in zip(q, np.quantile(pr, q))},
        "pred_mean": float(pr.mean()), "pred_min": float(pr.min()), "pred_max": float(pr.max()),
        "train_quantiles": {str(k): float(v) for k, v in zip(q, np.quantile(ytr.to_numpy(), q))},
        "train_mean": float(ytr.mean()),
        "n_rows": int(len(pr)),
    }


DEFAULT_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.05, num_leaves=255,
    min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, max_cat_threshold=32, cat_smooth=10.0, min_data_per_group=100,
    num_threads=int(os.environ.get("PRC_THREADS", "12")), verbosity=-1, force_row_wise=True,
    deterministic=True,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--target", default="raw", choices=["raw", "log", "asinh", "residual", "phi"])
    ap.add_argument("--weights", default="none", choices=["none", "gap", "no_nm"])
    ap.add_argument("--weight-alpha", type=float, default=3.0)
    ap.add_argument("--clip-floor", type=float, default=None)
    ap.add_argument("--clip-ceiling", type=float, default=None)
    ap.add_argument("--train-clip-lo", type=float, default=None)
    ap.add_argument("--train-clip-hi", type=float, default=None)
    ap.add_argument("--drop-rollover", action="store_true",
                    help="drop prc.data.UNREACHED_ROLLOVER rows from training only")
    ap.add_argument("--train-clip-r", type=float, default=None,
                    help="winsorise the training residual r = BLOCK-AOBT symmetrically at +/- this")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=255)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--seed-bag", type=int, default=1,
                    help="average this many models, seeds seed, seed+100, seed+200, ...")
    ap.add_argument("--max-rounds", type=int, default=6000)
    ap.add_argument("--es-rounds", type=int, default=100)
    ap.add_argument("--es-frac", type=float, default=0.15)
    ap.add_argument("--es-mode", default="frac", choices=["frac", "lastmonth"])
    ap.add_argument("--fixed-rounds", type=int, default=0,
                    help="skip round selection and refit for exactly this many rounds")
    ap.add_argument("--full-round-scale", type=float, default=1.1)
    ap.add_argument("--fold-mean-rounds", action="store_true",
                    help="take the full fit's round count from the fold mean instead of its own early stop")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", default="july,december,january")
    ap.add_argument("--features", default="v1", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--frame-sources", default="v3", choices=["v3", "v3f"],
                    help="v3 is the frozen 171-column set; v3f adds exp-018's families")
    ap.add_argument("--frame", default="v2", choices=["v2", "v3"],
                    help="v3 is the all-in frame from frame.py (169 columns after dedup)")
    ap.add_argument("--encodings", default="none", choices=["none", "all", "taxi"])
    ap.add_argument("--allow-unisolated-encodings", action="store_true")
    ap.add_argument("--phi-fit", default="cv", choices=["cv", "full"])
    ap.add_argument("--phi-shrink", type=float, default=None,
                    help="override PhiModel shrink_rows (the row-count weighting's shrinkage)")
    ap.add_argument("--no-lirf-gate", action="store_true",
                    help="apply the lirf columns at every airport instead of only LIRF")
    ap.add_argument("--guard-bounds", default="trainrange", choices=["trainrange", "trainq", "fixed"])
    ap.add_argument("--residual-guard", action="store_true",
                    help="clip the implied BLOCK-minus-AOBT on rows with an NM off-block. RETIRED for "
                         "plain trees: it costs +0.36 s on july_loo and +0.27 on january and fires zero "
                         "times under the training-range rule. It is a linear-leaf instrument.")
    ap.add_argument("--exot", action="store_true",
                    help="add EXOT features: MVT-AOBT_3 is the airport taxi estimate (DPI 2.700)")
    ap.add_argument("--deice", action="store_true",
                    help="add prc.day_state_features ds_deice_hr (one column, a winter signal)")
    ap.add_argument("--recency-halflife", type=float, default=0.0,
                    help="halve the training weight every N months of distance from December")
    ap.add_argument("--winter-weight", type=float, default=1.0,
                    help="multiply the sample weight of December/January/February training rows")
    ap.add_argument("--stand", action="store_true",
                    help="add the non-duplicated part of prc.stand_features (9 of 24 columns)")
    ap.add_argument("--lirf", action="store_true",
                    help="add prc.lirf_features block-clock signals (label-free, no fit)")
    ap.add_argument("--surface", action="store_true",
                    help="add the push-anchored families of prc.surface_features")
    ap.add_argument("--phi", action="store_true",
                    help="add prc.phi_features columns, fitted leave-one-month-out on training months")
    ap.add_argument("--noac-head", action="store_true")
    ap.add_argument("--blend-weight", type=float, default=None,
                    help="flat weight on the tail model; omit to use the fitted per-regime weights")
    ap.add_argument("--blend-tail", action="store_true",
                    help="blend the candidate with prc.tail_model per regime, weights from blend_weights.json")
    ap.add_argument("--noac-rounds", type=int, default=60)
    ap.add_argument("--noac-leaves", type=int, default=63)
    ap.add_argument("--noac-min-data", type=int, default=5)
    ap.add_argument("--noac-lr", type=float, default=0.05)
    ap.add_argument("--noac-min-rows", type=int, default=50)
    ap.add_argument("--linear-tree", action="store_true")
    ap.add_argument("--linear-lambda", type=float, default=1.0)
    ap.add_argument("--per-airport", action="store_true")
    ap.add_argument("--no-submission", action="store_true")
    ap.add_argument("--budget-scan", action="store_true",
                    help="measure each fold's own round optimum and report the median and spread")
    ap.add_argument("--submission-only", default=None,
                    help="skip the folds and reuse result-<NAME>.json for the record")
    a = ap.parse_args()

    cfg = dict(
        name=a.name,
        params=dict(DEFAULT_PARAMS, learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                    min_data_in_leaf=a.min_data_in_leaf, lambda_l2=a.lambda_l2,
                    feature_fraction=a.feature_fraction, bagging_fraction=a.bagging_fraction,
                    linear_tree=a.linear_tree, linear_lambda=a.linear_lambda),
        target=a.target, weights=a.weights, weight_alpha=a.weight_alpha,
        clip_floor=a.clip_floor, clip_ceiling=a.clip_ceiling,
        train_clip_lo=a.train_clip_lo, train_clip_hi=a.train_clip_hi, train_clip_r=a.train_clip_r, drop_rollover=a.drop_rollover,
        max_rounds=a.max_rounds, es_rounds=a.es_rounds, es_frac=a.es_frac,
        es_mode=a.es_mode, fixed_rounds=a.fixed_rounds, seed_bag=a.seed_bag,
        seed=a.seed, full_round_scale=a.full_round_scale, per_airport=a.per_airport,
        full_early_stop=not a.fold_mean_rounds,
        frame=a.frame, frame_sources=a.frame_sources, features=a.features, encodings=a.encodings,
        allow_unisolated_encodings=a.allow_unisolated_encodings,
        lirf_gate=not a.no_lirf_gate, guard_bounds=a.guard_bounds, phi=a.phi, phi_fit=a.phi_fit, phi_shrink=a.phi_shrink, surface=a.surface, lirf=a.lirf, stand=a.stand, deice=a.deice, exot=a.exot, winter_weight=a.winter_weight, recency_halflife=a.recency_halflife, residual_guard=a.residual_guard, noac_head=a.noac_head, blend_tail=a.blend_tail, blend_weight=a.blend_weight, noac_rounds=a.noac_rounds,
        noac_leaves=a.noac_leaves, noac_min_data=a.noac_min_data, noac_lr=a.noac_lr,
        noac_min_rows=a.noac_min_rows,
    )
    if a.budget_scan:
        check_compute_budget()
        scans = [budget_scan(cfg, f) for f in a.folds.split(",")]
        opt = sorted(r["optimum"] for r in scans)
        med = int(np.median(opt))
        print(json.dumps({"scans": scans, "optima": opt, "median": med,
                          "spread": opt[-1] - opt[0]}, indent=2), flush=True)
        (HERE / f"budget-{a.name}.json").write_text(json.dumps(
            {"scans": scans, "optima": opt, "median": med, "spread": opt[-1] - opt[0]}, indent=2))
        return
    if a.submission_only:
        check_compute_budget()
        # Build a candidate from a configuration whose folds are already
        # measured: the full fit picks its own round count, so the fold numbers
        # are carried into the record rather than used.
        prev = json.loads((HERE / f"result-{a.submission_only}.json").read_text())
        results = prev["folds"]
        print(f"[reuse] fold numbers from result-{a.submission_only}.json: "
              + ", ".join(f"{k} {v['rmse']:.1f}" for k, v in results.items()), flush=True)
    else:
        check_compute_budget()
        results = {f: run_fold(cfg, f) for f in a.folds.split(",")}
    for f, r in results.items():
        print(f"[{f}] " + json.dumps(r), flush=True)

    record = {"name": a.name, "argv": " ".join(sys.argv[1:]), "folds": results,
              "phi_settings": phi_step.PHI_SETTINGS if a.phi else None,
              "lirf_gate": lirf_step.GATE_AIRPORT if (a.lirf and not a.no_lirf_gate) else None,
              "shared_modules": shared_module_provenance()}
    if not a.no_submission:
        record["submission"] = build_submission(cfg, results)
    (HERE / f"result-{a.name}.json").write_text(json.dumps(record, indent=2, default=float))

    if a.budget_scan:
        check_compute_budget()
        scans = [budget_scan(cfg, f) for f in a.folds.split(",")]
        opt = sorted(r["optimum"] for r in scans)
        med = int(np.median(opt))
        print(json.dumps({"scans": scans, "optima": opt, "median": med,
                          "spread": opt[-1] - opt[0]}, indent=2), flush=True)
        (HERE / f"budget-{a.name}.json").write_text(json.dumps(
            {"scans": scans, "optima": opt, "median": med, "spread": opt[-1] - opt[0]}, indent=2))
        return
    if a.submission_only:
        return
    tag = shared_module_tag()
    with open(HERE / "climb.log", "a") as fh:
        for f, r in results.items():
            fh.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{a.name}\t{f}\trmse={r['rmse']:.1f}\t"
                f"bulk={r['rmse_bulk']:.1f}\ttail={r['rmse_tail']:.1f}\tno_nm={r['rmse_no_nm']:.1f}\t"
                f"noac={r.get('rmse_noac', float('nan')):.1f}\tclean={r.get('rmse_clean', float('nan')):.1f}\t"
                f"rounds={r['refit_rounds']}\t"
                f"{' '.join(sys.argv[1:])}\tmods={tag}\n"
            )
    print(json.dumps(record, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
