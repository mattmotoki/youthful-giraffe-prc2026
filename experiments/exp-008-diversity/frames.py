"""Month-level cache of the frozen v2 feature frame, shared by every exp-008 learner.

One build per month, written once to artifacts/exp-008/frames/ (git-ignored),
so that CatBoost, XGBoost, LightGBM and the MLP all consume bit-identical
columns and the diversity measured between them is the learner's, not the
frame's.  The v2 contract is retro-v1's published interface
(`experiments/exp-001-retro-v1/frame_contract.py`): 54 columns, 12 of them
object-dtype categoricals, one month = one build, no cross-month state.

`build_features` blanks a departure's own BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt
itself, so an unmasked training month and the ranking-masked validation month
produce the same columns; `verify_mask_invariance` checks that rather than
trusting it, and the cache therefore stores one build per month.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RETRO = ROOT / "experiments" / "exp-001-retro-v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RETRO))

import prc.data as D  # noqa: E402
from features import build_features  # noqa: E402

CACHE = ROOT / "artifacts" / "exp-008" / "frames"
VERSION = "v2"
SCHEMA_FINGERPRINT = "cf956e1b26b3c11f"   # frame_contract.py v2, 2026-09-05
NUMERIC_FINGERPRINT = "605fe71f984628d5"  # on 2025-07

#: The four folds this experiment scores on.
FOLDS = ("july", "july_loo", "december", "january")

#: Fold families for the fixed round counts. The first cut is the number of
#: training months. The second cut is the SEASON of the early-stop month, which
#: the first version of this file missed: the protocol early-stops on a training
#: month, and "the last training month" is 2025-12 for `july_loo`, so an
#: eleven-month fold validating July was having its tree budget chosen on a calm
#: winter month. CatBoost read 425 rounds that way and 3,625 on the six-month
#: fold whose early-stop month is 2025-06, and its `july_loo` score came out
#: 13.5 s WORSE than its `july` score on the same validation rows with five more
#: months of training data -- the sign that the budget, not the data, was wrong.
#: `m11s` was a season-matched family added on that reasoning and then REMOVED:
#: finding 10 refuted season as the cause (three eleven-month readings of 450,
#: 400 and 600 across two winter early-stop months and one summer one) and
#: finding 13 identified the real one -- the early-stopping patience. `july_loo`
#: is back in the plain `m11` family. `--rounds-m11s` and `ES_MONTH` survive as
#: the instruments that produced those measurements.
FAMILY = {"july": "m6", "july_loo": "m11", "december": "m11", "january": "m11"}

#: The nested rotation folds of `prc.data` (exp-025-cloud, the clean instrument
#: of the nested validation rule) train on TEN months, one fewer than the `m11` family.
#: They are put in `m11` rather than given a family of their own on purpose: the
#: round count is part of the member's identity, and re-deriving a budget for a
#: ten-month set would make the nested member a different estimator from the one
#: whose channel is being measured. Additive; no existing fold changes.
for _nested in ("july_loo_nx_dec", "july_loo_nx_jan", "december_nx_jul",
                "december_nx_jan", "january_nx_jul", "january_nx_dec"):
    FAMILY.setdefault(_nested, "m11")

#: The training month each fold early-stops on when choosing its frozen count.
#: `december` and `january` already satisfy the season rule with the default
#: "last training month"; `july_loo` does not, which is why it is named here.
ES_MONTH = {"july_loo": "2025-08"}


def cache_path(month: str) -> Path:
    return CACHE / f"{VERSION}-{month}.parquet"


def build_month(month: str) -> pd.DataFrame:
    """Features + label + take-off epoch for one month, one row per departure."""
    df = D.load_month(month)
    dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    X = build_features(df, VERSION)
    X["y"] = df.loc[dep, D.TARGET].astype("float64").to_numpy()
    X["t"] = df.loc[dep, "MVT_TIME_UTC_mvt"].astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9
    X["month_str"] = month
    return X.reset_index(drop=True)


def ensure_month(month: str, force: bool = False) -> Path:
    p = cache_path(month)
    if force or not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        build_month(month).to_parquet(p, index=False)
    return p


def load_cached(month: str) -> pd.DataFrame:
    return pd.read_parquet(ensure_month(month))


def verify_mask_invariance(month: str = "2025-07") -> dict:
    """The masked validation frame must give the same feature columns as the
    unmasked training frame; only then is one cached build per month legitimate."""
    full = D.load_month(month)
    masked, y_hidden = D.mask_like_ranking(full)
    a = build_features(full, VERSION)
    b = build_features(masked, VERSION)
    assert list(a.columns) == list(b.columns), "column list differs under masking"
    diffs = {}
    for c in a.columns:
        if a[c].dtype == object or pd.api.types.is_string_dtype(a[c]):
            same = (a[c].fillna("\x00").to_numpy() == b[c].fillna("\x00").to_numpy()).all()
            if not same:
                diffs[c] = "object mismatch"
        else:
            u = a[c].to_numpy(dtype="float64")
            v = b[c].to_numpy(dtype="float64")
            bad = ~((u == v) | (np.isnan(u) & np.isnan(v)))
            if bad.any():
                diffs[c] = float(np.nanmax(np.abs(u[bad] - v[bad])))
    y_a = full.loc[(full["PHASE_mvt"] == "DEP").to_numpy(), D.TARGET].astype("float64").to_numpy()
    y_b = y_hidden.loc[b.index].astype("float64").to_numpy()
    label_ok = bool(np.allclose(y_a, y_b, equal_nan=True))
    return {"month": month, "n_cols": len(a.columns), "diffs": diffs, "labels_match": label_ok}


def fold_frames(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, valid) for one fold, assembled from the month cache.

    The validation frame carries the hidden label in `y`; it comes from the
    month file, which is exactly what `prc.data.fold_frames` returns as
    `y_valid`, and the features were built by a function that cannot see it.
    """
    f = D.FOLDS[fold]
    tr = pd.concat([load_cached(m) for m in f.train_months], ignore_index=True)
    va = load_cached(f.valid_month)
    return tr, va


def ranking_frame() -> pd.DataFrame:
    """v2 features over the ranking file, built per ranking month.

    The two ranking months are isolated in the file, so each is built from its
    own rows alone -- the same contract every training month gets, which is what
    makes month-boundary edge effects identical in training and at serve time.
    """
    p = CACHE / f"{VERSION}-ranking.parquet"
    if not p.exists():
        r = D.load_ranking()
        out = []
        for m in sorted(r["month"].unique()):
            X = build_features(r[r["month"] == m], VERSION)
            X["month_str"] = m
            out.append(X)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(out, ignore_index=True).to_parquet(p, index=False)
    return pd.read_parquet(p)


def all_training() -> pd.DataFrame:
    """The twelve 2025 months, which is what a submission model fits."""
    return pd.concat([load_cached(m) for m in D.MONTHS], ignore_index=True)


FEATURE_COLUMNS = None  # filled on first use


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("MVT_ID_mvt", "y", "t", "month_str")]


def cat_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(frame)
            if frame[c].dtype == object or pd.api.types.is_string_dtype(frame[c])]


if __name__ == "__main__":
    import json
    import time

    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    if what == "verify":
        print(json.dumps(verify_mask_invariance(sys.argv[2] if len(sys.argv) > 2 else "2025-07"), indent=2))
    else:
        for m in D.MONTHS:
            t0 = time.time()
            p = ensure_month(m, force="--force" in sys.argv)
            print(f"{m}: {p.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
