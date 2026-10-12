"""Step 3e: the two-class placeholder odds, fitted on the matched rows only.

The richer of the two heads the assignment allows: a LightGBM classifier for
p(placeholder) in the separator's own feature space, and two anchors, predicting
the mixture EXPECTATION `mvt_sched - (p*o_ph + (1-p)*o_np)`.  Fit set `all` or
`matched`; leave-one-month-out over the twelve 2025 months; scored on the
held-out month's matched rows and on all its noac rows.

Also the question a revival would turn on: is the matched sub-population MORE
separable than the whole set?  If a classifier fitted and scored inside the
matched population beats the one fitted on everything, the lane has a head to
build; if not, the matched rows are the same problem with fewer rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_model import NOAC_FEATURES_V2, NOAC_CATS_V2, NOAC_PARAMS, NOAC_SEEDS  # noqa: E402
from direction import matched_mask, paired_rmse_diff, rmse  # noqa: E402

CATS = {}


def encode(d):
    X = d[list(NOAC_FEATURES_V2)].copy()
    for c in NOAC_CATS_V2:
        X[c] = pd.Categorical(d[c].astype(str), categories=CATS[c])
    return X


def fit_arm(fit: pd.DataFrame):
    X = encode(fit)
    ph = fit.ph.to_numpy(int)
    o = (fit.mvt_sched - fit.y_true).to_numpy(float)
    clf = [lgb.LGBMClassifier(**dict(NOAC_PARAMS, n_jobs=6, seed=s, bagging_seed=s,
                                     feature_fraction_seed=s, data_random_seed=s,
                                     objective="binary")).fit(
        X, ph, categorical_feature=list(NOAC_CATS_V2)) for s in NOAC_SEEDS]
    oph = float(o[ph == 1].mean()) if (ph == 1).any() else 0.0
    reg = [lgb.LGBMRegressor(**dict(NOAC_PARAMS, n_jobs=6, seed=s, bagging_seed=s,
                                    feature_fraction_seed=s, data_random_seed=s)).fit(
        X[ph == 0], o[ph == 0], categorical_feature=list(NOAC_CATS_V2)) for s in NOAC_SEEDS]
    return clf, reg, oph


def apply_arm(arm, d):
    clf, reg, oph = arm
    X = encode(d)
    p = np.mean([c.predict_proba(X)[:, 1] for c in clf], axis=0)
    onp = np.mean([r.predict(X) for r in reg], axis=0)
    return d.mvt_sched.to_numpy(float) - (p * oph + (1 - p) * onp), p


def main() -> None:
    global CATS
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = matched_mask(a, b)
    CATS = {c: pd.Index(sorted(set(a[c].astype(str)) | set(b[c].astype(str))))
            for c in NOAC_CATS_V2}

    res = {k: [] for k in ("y", "day", "m", "band", "p_all", "p_mat", "pr_all", "pr_mat", "ph")}
    for m in D.MONTHS:
        va, fit = a[a.month == m], a[a.month != m]
        pa, ppa = apply_arm(fit_arm(fit), va)
        pm, ppm = apply_arm(fit_arm(fit[fit._m]), va)
        res["y"].append(va.y_true.to_numpy(float)); res["day"].append(va.day.to_numpy())
        res["m"].append(va._m.to_numpy()); res["band"].append(va.band.to_numpy())
        res["p_all"].append(pa); res["p_mat"].append(pm)
        res["pr_all"].append(ppa); res["pr_mat"].append(ppm)
        res["ph"].append(va.ph.to_numpy())
        print(f"  {m} done", flush=True)
    r = {k: np.concatenate(v) for k, v in res.items()}
    np.savez(EXP / "work" / "clf_lomo.npz", **r)

    out = []
    for scope in ("matched", "all"):
        s = np.flatnonzero(r["m"]) if scope == "matched" else np.arange(len(r["y"]))
        lo, hi = paired_rmse_diff(r["y"][s], r["p_all"][s], r["p_mat"][s], r["day"][s])
        out.append({"scope": scope, "n": len(s),
                    "rmse_fit_all": rmse(r["y"][s], r["p_all"][s]),
                    "rmse_fit_matched": rmse(r["y"][s], r["p_mat"][s]),
                    "gain": rmse(r["y"][s], r["p_all"][s]) - rmse(r["y"][s], r["p_mat"][s]),
                    "lo": lo, "hi": hi,
                    "auc_fit_all": float(roc_auc_score(r["ph"][s], r["pr_all"][s])),
                    "auc_fit_matched": float(roc_auc_score(r["ph"][s], r["pr_mat"][s]))})
    t = pd.DataFrame(out)
    t.to_csv(EXP / "analysis" / "clf_arm.csv", index=False)
    print("\n=== two-class placeholder odds, LOMO over twelve months ===")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
