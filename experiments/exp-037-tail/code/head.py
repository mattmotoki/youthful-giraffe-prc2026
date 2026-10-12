"""The noac head, with and without the chain, out of fold.

The shipped head (`prc.tail_model`, noac v4) is a two-class mixture:

    p        a LightGBM classifier for `ph = |mvt_sched - y| < 300 s`
    off_np   a LightGBM regressor for `mvt_sched - y` on the NON-placeholder rows
    y_hat    mvt_sched - (1 - p) * off_np          (the RMSE-optimal expectation)

and above `BAND_MIN` (8 h of `mvt_sched`) the exp-009 two-point band table
overrides it and every prediction is floored at `MIN_TAXI`.  This module refits
and re-serves exactly that path on the noac rows alone, twice: on
`NOAC_FEATURES_V2` (the `base` arm, which is the shipped feature set) and on
`NOAC_FEATURES_V2 + CHAIN_FEATS` (the `chain` arm).  Everything else -- the
parameters, the seed set, the seed averaging, the class definition -- is the
shipped model's.

The correction the §4 arm applies to a SHIPPED prediction is the difference of
the two, `pred_chain - pred_base`, so the shipped vector stays the anchor and
the only thing measured is what the chain adds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_model import (NOAC_FEATURES_V2, NOAC_CATS_V2, NOAC_PARAMS, NOAC_SEEDS,
                            BAND_MIN, MIN_TAXI, TailModel)  # noqa: E402
from frames import CHAIN_FEATS, ROT_FEATS  # noqa: E402

N_JOBS = int(__import__("os").environ.get("PRC_N_JOBS", "2"))
ARMS = {"base": tuple(NOAC_FEATURES_V2),
        "chain": tuple(NOAC_FEATURES_V2) + tuple(CHAIN_FEATS),
        "rot": tuple(NOAC_FEATURES_V2) + tuple(CHAIN_FEATS) + tuple(ROT_FEATS)}


def encode(d: pd.DataFrame, feats, cats, categories: dict | None = None) -> tuple:
    """`prc.tail_model._encode`: pandas Categoricals against a fixed vocabulary.

    LightGBM is then told which columns they are (`categorical_feature=`) and
    splits them natively.  Coding them as ordinals instead -- which this module
    did until the round-1 review -- makes a different model: the tree can only
    cut the sorted-string order.  Unseen levels are NaN either way.
    """
    if categories is None:
        categories = {c: pd.Index(sorted(set(d[c].astype(str)))) for c in cats}
    out = pd.DataFrame(index=d.index)
    for c in feats:
        if c in cats:
            out[c] = pd.Categorical(d[c].astype(str), categories=categories[c])
        else:
            out[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    return out, categories


def fit(train: pd.DataFrame, feats, cats=NOAC_CATS_V2, n_jobs: int = N_JOBS) -> dict:
    X, categories = encode(train, feats, cats)
    ph = train.ph.to_numpy("float64")
    off = (train.mvt_sched - train.y_true).to_numpy("float64")
    clf, reg = [], []
    for s in NOAC_SEEDS:
        p = dict(NOAC_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s,
                 data_random_seed=s, n_jobs=n_jobs)
        c = lgb.LGBMClassifier(**{**p, "objective": "binary"})
        c.fit(X, ph, categorical_feature=list(cats))
        clf.append(c)
        m = ph == 0
        r = lgb.LGBMRegressor(**p)
        r.fit(X[m], off[m], categorical_feature=list(cats))
        reg.append(r)
    # the shipped serving path, not only the shipped fit: above BAND_MIN the
    # exp-009 two-point band table overrides the mixture, and every prediction
    # is floored at MIN_TAXI.  Round 3 of the review found `predict` returning
    # the raw mixture, which is not what any shipped number is.
    band = TailModel._fit_band(train, train.y_true.to_numpy("float64"))
    return {"clf": clf, "reg": reg, "feats": feats, "cats": cats,
            "categories": categories, "band": band}


def predict(model: dict, d: pd.DataFrame) -> pd.DataFrame:
    X, _ = encode(d, model["feats"], model["cats"], model["categories"])
    p = np.mean([c.predict_proba(X)[:, 1] for c in model["clf"]], axis=0)
    off = np.mean([r.predict(X) for r in model["reg"]], axis=0)
    ms = d.mvt_sched.to_numpy("float64")
    y = ms - (1.0 - p) * off
    bp = TailModel(band=model.get("band") or {}, noac_head="v4")._band_predict(d)
    y = np.where(np.isnan(bp), y, bp)
    return pd.DataFrame({"p": p, "off_np": off,
                         "y_hat": np.clip(y, MIN_TAXI, None),
                         "y_raw": ms - (1.0 - p) * off,
                         "y_bulk": np.clip(ms - off, MIN_TAXI, None)}, index=d.index)


def lomo(a: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    """Leave one 2025 month out, twelve times, both arms."""
    out = []
    for m in D.MONTHS:
        tr, te = a[a.month != m], a[a.month == m]
        r = te[["MVT_ID_mvt", "month", "day", "band", "mvt_sched", "y_true", "ph"]].copy()
        for name, feats in arms.items():
            pr = predict(fit(tr, feats), te)
            r[f"p_{name}"] = pr.p.to_numpy()
            r[f"y_{name}"] = pr.y_hat.to_numpy()
            r[f"bulk_{name}"] = pr.y_bulk.to_numpy()
            r[f"raw_{name}"] = pr.y_raw.to_numpy()
        out.append(r)
        print(f"  lomo {m}: {len(te)} rows", flush=True)
    return pd.concat(out, ignore_index=True)


def fold_arms(a: pd.DataFrame, folds=("july_loo", "december", "january"), arms=ARMS) -> pd.DataFrame:
    """Both arms fitted on each fold's TRAINING months and predicted on its month."""
    out = []
    for f in folds:
        fold = D.FOLDS[f]
        tr = a[a.month.isin(fold.train_months)]
        te = a[a.month == fold.valid_month]
        r = te[["MVT_ID_mvt", "month", "day", "band", "mvt_sched", "y_true", "ph"]].copy()
        r["fold"] = f
        for name, feats in arms.items():
            pr = predict(fit(tr, feats), te)
            r[f"p_{name}"] = pr.p.to_numpy()
            r[f"y_{name}"] = pr.y_hat.to_numpy()
            r[f"bulk_{name}"] = pr.y_bulk.to_numpy()
            r[f"raw_{name}"] = pr.y_raw.to_numpy()
        out.append(r)
        print(f"  fold {f}: train {len(tr)}, test {len(te)}", flush=True)
    return pd.concat(out, ignore_index=True)


def ranking_head(a: pd.DataFrame, b: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    """Both arms fitted on all twelve 2025 months and predicted on the 388."""
    r = b[["MVT_ID_mvt", "month", "band", "mvt_sched"]].copy()
    for name, feats in arms.items():
        pr = predict(fit(a, feats), b)
        r[f"p_{name}"] = pr.p.to_numpy()
        r[f"y_{name}"] = pr.y_hat.to_numpy()
        r[f"bulk_{name}"] = pr.y_bulk.to_numpy()
        r[f"raw_{name}"] = pr.y_raw.to_numpy()
    return r


if __name__ == "__main__":
    a = pd.read_parquet(EXP / "work" / "noac_2025.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026.parquet")
    print("LOMO:", flush=True)
    lomo(a).to_parquet(EXP / "work" / "lomo.parquet")
    print("folds:", flush=True)
    fold_arms(a).to_parquet(EXP / "work" / "fold_arms.parquet")
    print("ranking head:", flush=True)
    ranking_head(a, b).to_parquet(EXP / "work" / "ranking_head.parquet")
    print("done; BAND_MIN =", BAND_MIN)
