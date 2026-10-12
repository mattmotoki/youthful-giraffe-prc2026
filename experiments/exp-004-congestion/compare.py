"""Segment scoring and airport-day cluster bootstrap for two saved OOF files.

Movements are not independent: rows share an airport and a day, and the
surface features are built from shared neighbouring events, so resampling
individual rows understates the uncertainty of a difference. Every interval in
RESULTS.md is produced here by resampling **airport-days** with replacement.

Because a cluster is drawn whole, the RMSE of a resample is exact from the
per-cluster squared-error sums:

    RMSE_b = sqrt( sum_c w_c * sse_c / sum_c w_c * n_c )

so a draw costs O(number of clusters), not O(rows).

    python compare.py <ref-config> <config> <fold> [--weather]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import prc.data as D

BOOT = 2000
SEED = 11


def load_oof(cfg: str, fold: str) -> pd.DataFrame:
    return pd.read_parquet(HERE / "oof" / f"{cfg}_{fold}.parquet")


def segments(o: pd.DataFrame, bad_wx: np.ndarray | None = None) -> dict:
    y = o["y_true"].to_numpy()
    nm = o["seg_has_nm"].to_numpy() == 1
    noac = o["seg_noac"].to_numpy().astype(bool)
    bulk = y <= 3600
    segs = {
        "all": np.ones(len(y), bool),
        "bulk (y<=3600)": bulk,
        "tail (y>3600)": ~bulk,
        "no-NM": ~nm,
        "no-NM bulk": (~nm) & bulk,
        "noac": noac,
        "non-noac": ~noac,
        "NM bulk": nm & bulk,
    }
    if bad_wx is not None:
        segs["bad weather"] = bad_wx
        segs["bad weather, NM bulk"] = bad_wx & nm & bulk
        segs["fair weather, NM bulk"] = (~bad_wx) & nm & bulk
    return segs


def _cluster_stats(e2: np.ndarray, cluster: np.ndarray, mask: np.ndarray):
    """Per-cluster squared-error sum and row count, restricted to `mask`."""
    c = cluster[mask]
    codes, _ = pd.factorize(c)
    k = codes.max() + 1 if len(codes) else 0
    sse = np.bincount(codes, weights=e2[mask], minlength=k)
    n = np.bincount(codes, minlength=k).astype(float)
    return sse, n


def cluster_ci(e2a, e2b, cluster, mask, boot=BOOT, seed=SEED):
    """95 % CI for rmse(b) - rmse(a), resampling airport-days with replacement."""
    ssa, n = _cluster_stats(e2a, cluster, mask)
    ssb, _ = _cluster_stats(e2b, cluster, mask)
    k = len(n)
    if k < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=boot).astype(float)
    denom = w @ n
    good = denom > 0
    d = np.full(boot, np.nan)
    d[good] = np.sqrt((w @ ssb)[good] / denom[good]) - np.sqrt((w @ ssa)[good] / denom[good])
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def compare(ref: str, cfg: str, fold: str, bad_wx: np.ndarray | None = None,
            boot: int = BOOT) -> pd.DataFrame:
    a, b = load_oof(ref, fold), load_oof(cfg, fold)
    assert (a[D.ID].to_numpy() == b[D.ID].to_numpy()).all(), "OOF row order differs"
    e2a = ((a.y_true - a.y_pred) ** 2).to_numpy()
    e2b = ((b.y_true - b.y_pred) ** 2).to_numpy()
    cluster = (a["seg_apt"].astype(str) + ":" + a["seg_day"].astype(str)).to_numpy()
    rows = []
    for name, m in segments(a, bad_wx).items():
        if m.sum() < 30:
            continue
        ra = float(np.sqrt(e2a[m].mean()))
        rb = float(np.sqrt(e2b[m].mean()))
        lo, hi = cluster_ci(e2a, e2b, cluster, m, boot)
        rows.append({"segment": name, "n": int(m.sum()), "ref": round(ra, 1),
                     "cfg": round(rb, 1), "delta": round(rb - ra, 1),
                     "ci95": f"[{lo:+.1f}, {hi:+.1f}]",
                     "sig": "" if lo <= 0 <= hi else "*"})
    return pd.DataFrame(rows)


def weather_mask(fold: str, oof: pd.DataFrame) -> np.ndarray:
    from weather_slice import bad_weather_mask
    wx = bad_weather_mask(fold).set_index(D.ID).loc[oof[D.ID]]
    return wx["bad_wx"].to_numpy().astype(bool)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ref, cfg, fold = args[0], args[1], args[2]
    bad = weather_mask(fold, load_oof(ref, fold)) if "--weather" in sys.argv else None
    pd.set_option("display.width", 200)
    print(f"== {cfg} vs {ref} on {fold}  (airport-day cluster bootstrap, {BOOT} draws)")
    print(compare(ref, cfg, fold, bad).to_string(index=False))
