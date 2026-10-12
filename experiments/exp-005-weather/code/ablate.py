#!/usr/bin/env python3
"""Weather ablation over a frozen base model, on the july and december folds.

Base model: own-flight deltas + airport, stand, runway, aircraft type,
operator, hour (plus day-of-week, so weather cannot proxy weekday traffic).
Learning target is the residual `taxi - nm_taxi` with `nm_taxi = MVT - AOBT_3`
anchored to 0 where there is no NM join, so L2 on the residual is L2 on taxi
and one model covers both regimes. Config is frozen: every reported gain is a
feature-set difference, not a tuning difference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import prc.data as d  # noqa: E402
import prc.weather_features as wf  # noqa: E402

CACHE = ROOT / "artifacts" / "exp-005-weather" / "cache"
OUTDIR = ROOT / "experiments" / "exp-005-weather"

BASE_NUMERIC = [
    "nm_taxi", "mvt_sched", "mvt_lobt", "mvt_eobt1", "aobt_sched", "aobt_lobt",
    "aobt_eobt1", "lobt_iobt", "lobt_sched", "arvt3_aobt", "has_nm", "hour", "dow",
    # routed by the desk from the tail worker (exp-002): the placeholder-block-time
    # rows. They are 0.2 % of departures but set the floor on fold RMSE, and until
    # the base handles them the fold metric is too noisy to measure weather with.
    "noac", "noac_day_count", "flight_len",
]
CATEGORICALS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                "AIRCRAFT_OPERATOR_flt", "stand_runway"]

PARAMS = dict(
    objective="l2",
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=100,
    min_data_per_group=50,
    cat_smooth=20.0,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=1.0,
    max_cat_to_onehot=1,
    n_jobs=int(os.environ.get("PRC_N_JOBS", "12")),  # assignment cap 12; shrink when the box is loaded
    verbosity=-1,
)

# Columns that carry the Network Manager off-block time. The `noaobt_` configs
# drop them and set the anchor to 0, so the model has to predict the physical
# taxi duration instead of the BLOCK-minus-AOBT clock offset. That is not the
# scored task; it is the control that says whether a small weather gain means
# "weather does not matter" or "the actual columns already carry the weather".
AOBT_COLUMNS = ["nm_taxi", "aobt_sched", "aobt_lobt", "aobt_eobt1", "arvt3_aobt", "has_nm"]

CONFIGS = {
    "base": [],
    "basic": ["basic"],
    "precip": ["precip"],
    "runway_wind": ["runway_wind"],
    "visibility": ["visibility"],
    "change": ["change"],
    "aobt": ["aobt"],
    "all": list(wf.FAMILIES),
    "no_aobt": [f for f in wf.FAMILIES if f != "aobt"],
    "precip_vis": ["precip", "visibility"],
    "noaobt_base": [],
    "noaobt_all": list(wf.FAMILIES),
}

# Explicit feature lists, for configs that are not a union of families.
# `winter_core` is the repair for the one negative the v1 ablation produced: on
# the july fold the full 44-column weather block made the <=3600 slice WORSE by
# 11.5 s [3.6, 19.7], all of it on the 3,938 no-NM rows, where the model has no
# AOBT anchor and must regress absolute taxi time. Forty-four weak columns on a
# 2 % subgroup buy variance, not signal. This is the smallest set that carries
# the effect the december bootstrap actually confirmed (snow, freezing, cold).
# Single definition: the lists live in prc.weather_features so a consumer and
# this ablation cannot drift apart.
EXPLICIT = {
    "winter_core": list(wf.WINTER_CORE),
    "winter_core_plus": list(wf.WINTER_CORE_PLUS),
    # MATCHED control: exactly the WINTER_CORE columns with the AOBT anchor and
    # every AOBT-derived own-flight delta removed. `noaobt_all` vs `base ->
    # winter_core` confounded removing AOBT with adding 24 more weather columns
    # (Sol round 2, blocking finding 1); this pair differs only in the anchor.
    # Every WINTER_CORE column is a take-off-hour column, so the drop_aobt
    # filter leaves all seven intact.
    "noaobt_winter_core": list(wf.WINTER_CORE),
}
CONFIGS.update({name: [] for name in EXPLICIT})


# `january` (train Feb-Dec 2025, validate Jan 2025) is the shared backward fold
# prc.data added on 2026-09-05, for the reason this experiment measured:
# December 2025 carries 0.2 mm of snow at EDDF against 27.8 mm in the January
# 2026 ranking month, so the forward folds cannot see a winter effect at all.


def load_fold(fold_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold = d.FOLDS[fold_name]
    train = pd.concat([pd.read_parquet(CACHE / f"{m}.parquet") for m in fold.train_months],
                      ignore_index=True)
    valid = pd.read_parquet(CACHE / f"{fold.valid_month}.parquet")
    return train, valid


def encode(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    """Category codes from the training months; unseen levels become -1."""
    for column in CATEGORICALS:
        levels = pd.Index(pd.unique(train[column]))
        train[column] = pd.Categorical(train[column], categories=levels).codes
        valid[column] = pd.Categorical(valid[column], categories=levels).codes
        train[column] = train[column].astype("int32")
        valid[column] = valid[column].astype("int32")


def slices(y: np.ndarray, pred: np.ndarray, has_nm: np.ndarray) -> dict[str, float]:
    def r(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))) if mask.sum() else float("nan")

    nm = has_nm > 0.5
    return {
        "rmse": r(np.ones(len(y), bool)),
        "rmse_bulk": r((y <= 3600) & nm),
        "rmse_le3600": r(y <= 3600),
        "rmse_gt3600": r(y > 3600),
        "rmse_no_nm": r(~nm),
        "n": int(len(y)),
        "n_gt3600": int((y > 3600).sum()),
        "n_no_nm": int((~nm).sum()),
    }


def run(config: str, fold_name: str, seed: int, train: pd.DataFrame, valid: pd.DataFrame,
        save_oof: bool) -> dict:
    drop_aobt = config.startswith("noaobt_")
    numeric = [c for c in BASE_NUMERIC if not (drop_aobt and c in AOBT_COLUMNS)]
    families = [c for family in CONFIGS[config] for c in wf.FAMILIES[family]]
    families += EXPLICIT.get(config, [])
    if drop_aobt:
        # Every member of the `aobt` family comes from the AOBT-hour join,
        # including wx_taxi_d_temp_c / wx_taxi_d_wind_speed_ms / wx_taxi_precip_mm,
        # which carry no wxo_ prefix. Sol review 2026-09-05, blocking finding 1.
        banned = set(wf.FAMILIES["aobt"])
        families = [c for c in families if c not in banned and not c.startswith("wxo_")]
    features = numeric + CATEGORICALS + families
    if drop_aobt:
        anchor_tr = np.zeros(len(train))
        anchor_va = np.zeros(len(valid))
    else:
        anchor_tr = train["nm_taxi"].fillna(0.0).to_numpy(dtype="float64")
        anchor_va = valid["nm_taxi"].fillna(0.0).to_numpy(dtype="float64")
    y_tr = train["y"].to_numpy(dtype="float64")
    y_va = valid["y"].to_numpy(dtype="float64")

    params = dict(PARAMS, random_state=seed)
    model = lgb.LGBMRegressor(**params)
    started = time.time()
    model.fit(train[features], y_tr - anchor_tr, categorical_feature=CATEGORICALS)
    pred = model.predict(valid[features]) + anchor_va
    result = {"config": config, "fold": fold_name, "seed": seed,
              "n_features": len(features), "fit_seconds": round(time.time() - started, 1)}
    result.update(slices(y_va, pred, valid["has_nm"].to_numpy()))

    # December weather-day split: days with any snow or freezing precipitation
    # at the departure airport versus the rest.
    bad = (valid["wx_snow_6h_mm"].fillna(0) > 0.1) | (valid["wx_freezing_precip"].fillna(0) > 0.5)
    result["n_bad_wx"] = int(bad.sum())
    if bad.any():
        result["rmse_bad_wx"] = float(np.sqrt(np.mean((y_va[bad] - pred[bad]) ** 2)))
    if (~bad).any():
        result["rmse_good_wx"] = float(np.sqrt(np.mean((y_va[~bad] - pred[~bad]) ** 2)))

    if save_oof:
        oof = pd.DataFrame({
            "MVT_ID_mvt": valid["MVT_ID_mvt"].to_numpy(),
            "fold": fold_name,
            "y_true": y_va,
            "y_pred": pred,
        })
        (OUTDIR / "oof").mkdir(parents=True, exist_ok=True)
        oof.to_parquet(OUTDIR / "oof" / f"{config}_{fold_name}.parquet", index=False)
        imp = pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")})
        imp.sort_values("gain", ascending=False).to_csv(
            OUTDIR / "oof" / f"importance_{config}_{fold_name}.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="base,basic,precip,runway_wind,visibility,change,aobt,all")
    parser.add_argument("--folds", default="july,december")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--save-oof", default="")
    parser.add_argument("--out", default="results.jsonl")
    arguments = parser.parse_args()

    configs = arguments.configs.split(",")
    seeds = [int(s) for s in arguments.seeds.split(",")]
    save = set(arguments.save_oof.split(",")) - {""}
    out_path = OUTDIR / arguments.out
    for fold_name in arguments.folds.split(","):
        train, valid = load_fold(fold_name)
        encode(train, valid)
        print(f"[{fold_name}] train={len(train):,} valid={len(valid):,}", flush=True)
        for config in configs:
            for seed in seeds:
                record = run(config, fold_name, seed, train, valid, config in save)
                with out_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        del train, valid


if __name__ == "__main__":
    main()
