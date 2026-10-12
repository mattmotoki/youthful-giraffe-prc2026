#!/usr/bin/env python3
"""Paired comparison of two saved OOF prediction sets, clustered by airport-day.

Comparing two RMSE numbers on one fold is weak: a few hundred tail rows carry
most of the squared error, so the fold RMSE moves several seconds on them. The
same rows are scored by both models, so the paired difference removes that.

Resampling FLIGHTS, however, assumes flights are independent, and they are not:
weather and operational shocks are shared inside an airport-day, so a
flight-level interval is far too narrow for exactly the conditional slices this
experiment cares about. The unit of resampling here is therefore the
(airport, UTC day) cluster. Sol review 2026-09-05, blocking finding 2.

These intervals are POST-SELECTION: `winter_core` was chosen by looking at
these folds. They describe the spread of the measured difference on these rows,
not a generalization guarantee for a set chosen in advance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OOF = ROOT / "experiments" / "exp-005-weather" / "oof"
CACHE = ROOT / "artifacts" / "exp-005-weather" / "cache"
import prc.data as d  # noqa: E402


def paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, cluster: np.ndarray,
           rng: np.random.Generator, draws: int = 2000) -> tuple[float, float, float, float]:
    """RMSE(a), RMSE(b), and a cluster-bootstrap interval for RMSE(b) - RMSE(a).

    `cluster` holds an integer airport-day id per row. Whole clusters are drawn
    with replacement, so the interval respects the fact that every flight in an
    airport-day shares one weather realisation.
    """
    sa, sb = (y - a) ** 2, (y - b) ** 2
    rmse_a, rmse_b = np.sqrt(sa.mean()), np.sqrt(sb.mean())

    order = np.argsort(cluster, kind="stable")
    sa_sorted, sb_sorted = sa[order], sb[order]
    keys = cluster[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    sizes = np.diff(np.r_[starts, len(keys)])
    sum_a = np.add.reduceat(sa_sorted, starts)
    sum_b = np.add.reduceat(sb_sorted, starts)

    n_clusters = len(starts)
    pick = rng.integers(0, n_clusters, size=(draws, n_clusters))
    counts = sizes[pick].sum(axis=1)
    mean_a = sum_a[pick].sum(axis=1) / counts
    mean_b = sum_b[pick].sum(axis=1) / counts
    diffs = np.sqrt(mean_b) - np.sqrt(mean_a)
    return float(rmse_a), float(rmse_b), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="base")
    parser.add_argument("--b", default="all")
    parser.add_argument("--folds", default="july,december")
    arguments = parser.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for fold_name in arguments.folds.split(","):
        a = pd.read_parquet(OOF / f"{arguments.a}_{fold_name}.parquet")
        b = pd.read_parquet(OOF / f"{arguments.b}_{fold_name}.parquet")
        assert (a["MVT_ID_mvt"].to_numpy() == b["MVT_ID_mvt"].to_numpy()).all()
        valid_month = {"july": "2025-07", "december": "2025-12", "january": "2025-01"}[fold_name]
        cache = pd.read_parquet(CACHE / f"{valid_month}.parquet")
        assert (cache["MVT_ID_mvt"].to_numpy() == a["MVT_ID_mvt"].to_numpy()).all()

        # airport-day cluster id, from columns the ranking file carries
        day = pd.to_datetime(cache["takeoff_hour_utc"], utc=True).dt.floor("D")
        cluster = pd.factorize(
            cache["ADEP_mvt"].astype(str) + "|" + day.dt.strftime("%Y-%m-%d"))[0]

        y = a["y_true"].to_numpy()
        pa, pb = a["y_pred"].to_numpy(), b["y_pred"].to_numpy()
        nm = cache["has_nm"].to_numpy() > 0.5
        bad = ((cache["wx_snow_6h_mm"].fillna(0) > 0.1)
               | (cache["wx_freezing_precip"].fillna(0) > 0.5)).to_numpy()
        cold = (cache["wx_temp_c"] <= 3.0).to_numpy()
        windy = (cache["wx_wind_speed_ms"] >= 8.0).to_numpy()
        lowvis = (cache["wx_low_visibility"].fillna(0) > 0.5).to_numpy()
        ones = np.ones(len(y), bool)

        for label, mask in [
            ("overall", ones),
            ("bulk (<=3600 & NM)", (y <= 3600) & nm),
            ("<=3600", y <= 3600),
            (">3600", y > 3600),
            ("no-NM", ~nm),
            ("snow/freezing hours", bad),
            ("other hours", ~bad),
            ("temp<=3C", cold),
            ("wind>=8m/s", windy),
            ("low visibility", lowvis),
            ("bulk & snow/freezing", (y <= 3600) & nm & bad),
        ]:
            if mask.sum() < 50:
                rows.append((fold_name, label, int(mask.sum()), np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            ra, rb, lo, hi = paired(y[mask], pa[mask], pb[mask], cluster[mask], rng)
            rows.append((fold_name, label, int(mask.sum()), ra, rb, rb - ra, lo, hi))

    table = pd.DataFrame(rows, columns=["fold", "slice", "n", f"rmse_{arguments.a}",
                                        f"rmse_{arguments.b}", "delta", "ci_lo", "ci_hi"])
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
    print("\ndelta < 0 means the weather model is better. CI is a 95 % PAIRED CLUSTER"
          "\nbootstrap over (airport, UTC day); post-selection, not a generalization guarantee.")


if __name__ == "__main__":
    main()
