"""Step 1: is the 2026 noac population distinguishable from the 2025 one?

A label-free two-sample test: a classifier separating 2026 rows from 2025 rows
in the placeholder separator's own feature space (`NOAC_FEATURES_V2`), scored
out of fold.  Nothing here reads a target.

**Validation is grouped by calendar day** (Sol round 1, S3).  The first version
of this module used row-stratified folds, so rows of one day sat on both sides
of the split and the six day-level columns (`dep_day_n`, `noac_day_n`,
`noac_day_share`, `noac_day_count`, `nonm_day_count`, `arr_day_ph_rate`, which
are constant within a day and nearly unique to it) could be memorised inside the
fold.  That contaminated the null as well as the comparison and made "all
memorisation" look like the whole story.  With day-grouped validation a
classifier has to transfer between days, which is what the 2026 rows demand.

Three reference levels, all on the same instrument:

  real        a 2025 sample against a 2026 sample
  month       two DIFFERENT 2025 months of the same season -- day-disjoint and
              month-disjoint, same population, so it prices what a month costs
  permutation **the real comparison's own days, relabelled**: the two samples
              are pooled and the class-1 label is permuted over CALENDAR DAYS
              with the real number of class-1 days held fixed.  Under "same
              population" the day labels are exchangeable, so this is the null
              for the comparison actually made.  The first version substituted a
              split of a separate 2025 season pool, which is not the same
              structure (the exact-July comparison is 31 days against 31, and
              that substitute gave 31 against 59) -- Sol round 2, R2-2.

The p-value is **two-sided**: an AUC below 0.5 with day-grouped folds is a
learner that generalises in the wrong direction, which is a departure from the
null in its own right, not a "no signal" reading.

Seeds are FIXED per comparison and recorded in the output.  The first version
used Python's `hash(tag)`, which is randomised per process, so the cited command
did not reproduce the committed CSV (Sol round 2, R2-2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from prc.tail_model import NOAC_FEATURES_V2, NOAC_CATS_V2  # noqa: E402

DAY_BLOCK = ("dep_day_n", "noac_day_n", "noac_day_share", "noac_day_count",
             "nonm_day_count", "arr_day_ph_rate")
SUMMER = ("2025-06", "2025-07", "2025-08")
WINTER = ("2025-12", "2025-01", "2025-02")
NPERM = 100
NFOLD = 5
P = dict(objective="binary", learning_rate=0.05, num_leaves=8, n_estimators=200,
         min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=6)


def day_of(d: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(d.mvt_time, utc=True).dt.floor("D").to_numpy()


def auc(a: pd.DataFrame, b: pd.DataFrame, feats: tuple) -> float:
    """Day-grouped out-of-fold AUC separating `b` (class 1) from `a` (class 0)."""
    d = pd.concat([a.assign(_y=0), b.assign(_y=1)], ignore_index=True)
    g = day_of(d)
    cats = [c for c in NOAC_CATS_V2 if c in feats]
    X = d[list(feats)].copy()
    for c in cats:
        X[c] = pd.Categorical(d[c].astype(str),
                              categories=pd.Index(sorted(set(d[c].astype(str)))))
    y = d._y.to_numpy()
    ndays = len(np.unique(g))
    k = min(NFOLD, ndays)
    if k < 2 or y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    oof = np.full(len(d), np.nan)
    for tr, va in GroupKFold(k).split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = lgb.LGBMClassifier(**dict(P, seed=7, bagging_seed=7,
                                      feature_fraction_seed=7, data_random_seed=7)).fit(
            X.iloc[tr], y[tr], categorical_feature=cats)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    ok = np.isfinite(oof)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], oof[ok]))


def perm_null(x: pd.DataFrame, y: pd.DataFrame, feats: tuple, seed: int) -> tuple:
    """Relabel the real comparison's own days, holding the class-1 day count.

    `x` and `y` are pooled; the class-1 label is permuted over the pooled set of
    calendar days with `len(unique(days of y))` of them assigned to class 1.  Day
    clustering, the day counts on both sides and the covariate geometry are
    exactly the real ones; only which days are called 2026 changes.
    """
    rng = np.random.default_rng(seed)
    pool = pd.concat([x, y], ignore_index=True)
    d = day_of(pool)
    days = np.unique(d)
    k = len(np.unique(day_of(y)))
    out = []
    for _ in range(NPERM):
        pick = set(rng.permutation(days)[:k])
        m = pd.Series(d).isin(pick).to_numpy()
        out.append(auc(pool[~m], pool[m], feats))
    return out, k


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026.parquet")
    FULL = tuple(NOAC_FEATURES_V2)
    ROW = tuple(c for c in NOAC_FEATURES_V2 if c not in DAY_BLOCK)

    comparisons = [
        ("2025-07 vs 2026-07", a[a.month == "2025-07"], b[b.month == "2026-07"], 101),
        ("2025-01 vs 2026-01", a[a.month == "2025-01"], b[b.month == "2026-01"], 102),
        ("summer 2025 vs 2026-07", a[a.month.isin(SUMMER)], b[b.month == "2026-07"], 103),
        ("winter 2025 vs 2026-01", a[a.month.isin(WINTER)], b[b.month == "2026-01"], 104),
        ("all 2025 vs all 2026", a, b, 105),
    ]
    month_controls = [
        ("2025-07 vs 2025-08", a[a.month == "2025-07"], a[a.month == "2025-08"]),
        ("2025-06 vs 2025-07", a[a.month == "2025-06"], a[a.month == "2025-07"]),
        ("2025-08 vs 2025-09", a[a.month == "2025-08"], a[a.month == "2025-09"]),
        ("2025-12 vs 2025-01", a[a.month == "2025-12"], a[a.month == "2025-01"]),
        ("2025-01 vs 2025-02", a[a.month == "2025-01"], a[a.month == "2025-02"]),
    ]

    reps, rows = [], []
    for space, feats in (("full", FULL), ("rowlevel", ROW)):
        for tag, x, y, seed in comparisons:
            r = auc(x, y, feats)
            nd = len(np.unique(day_of(y)))
            nulls, nd_used = perm_null(x, y, feats, seed=seed)
            for i, v in enumerate(nulls):
                reps.append({"space": space, "comparison": tag, "kind": "permutation",
                             "rep": i, "seed": seed, "auc": v})
            reps.append({"space": space, "comparison": tag, "kind": "real", "rep": 0,
                         "seed": seed, "auc": r})
            n = np.array([v for v in nulls if np.isfinite(v)])
            p_hi = (1 + (n >= r).sum()) / (1 + len(n))
            p_lo = (1 + (n <= r).sum()) / (1 + len(n))
            rows.append({"space": space, "comparison": tag, "n2025": len(x), "n2026": len(y),
                         "days_2026": nd, "null_days": nd_used, "seed": seed,
                         "null_kind": "day relabel of the pooled real comparison",
                         "auc": r,
                         "null_mean": float(n.mean()), "null_sd": float(n.std()),
                         "null_min": float(n.min()), "null_max": float(n.max()),
                         "add_one_p_upper": float(p_hi),
                         "add_one_p_two_sided": float(min(1.0, 2 * min(p_hi, p_lo)))})
            print(f"{space:9s} {tag:26s} n {len(x):5d}/{len(y):4d} AUC {r:.3f}  "
                  f"null {n.mean():.3f}+/-{n.std():.3f} [{n.min():.3f},{n.max():.3f}]  "
                  f"two-sided p {rows[-1]['add_one_p_two_sided']:.3f}", flush=True)
        for tag, x, y in month_controls:
            r = auc(x, y, feats)
            reps.append({"space": space, "comparison": tag, "kind": "month_control",
                         "rep": 0, "seed": np.nan, "auc": r})
            rows.append({"space": space, "comparison": tag, "n2025": len(x), "n2026": len(y),
                         "days_2026": len(np.unique(day_of(y))),
                         "null_days": np.nan, "seed": np.nan,
                         "null_kind": "2025 month control", "auc": r,
                         "null_mean": np.nan, "null_sd": np.nan, "null_min": np.nan,
                         "null_max": np.nan, "add_one_p_upper": np.nan,
                         "add_one_p_two_sided": np.nan})
            print(f"{space:9s} {tag:26s} n {len(x):5d}/{len(y):4d} AUC {r:.3f}  "
                  f"(2025 month control)", flush=True)
    pd.DataFrame(reps).to_csv(EXP / "analysis" / "twosample_reps.csv", index=False)
    pd.DataFrame(rows).to_csv(EXP / "analysis" / "twosample.csv", index=False)


if __name__ == "__main__":
    main()
