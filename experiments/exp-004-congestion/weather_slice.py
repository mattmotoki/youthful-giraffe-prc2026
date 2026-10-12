"""Score two saved OOF files on the disruption slice.

The `flow` family was built for disruption days. July has none and December
2025 has roughly a tenth of the snow of the scored January 2026 month
(exp-005 section 3), so a fold-average is the wrong instrument for it. This
scores a config on the rows that are actually in winter weather, using
exp-005's own definition of a snow-or-freezing hour:

    bad_wx = wx_snow_6h_mm > 0.1  or  wx_freezing_precip > 0.5
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import prc.data as D
import prc.weather_features as W
from run_ablation import CACHE


def bad_weather_mask(fold: str) -> pd.DataFrame:
    """Per-departure disruption flags for a fold's validation month."""
    month = D.FOLDS[fold].valid_month
    # mask first: the weather module reads neither hidden column today, but the
    # contract is that a validation feature is computed from the masked frame.
    frame, _ = D.mask_like_ranking(D.load_month(month))
    wx = W.transform(frame)
    dep = frame["PHASE_mvt"].to_numpy() == "DEP"
    out = pd.DataFrame({
        D.ID: frame.loc[dep, D.ID].to_numpy(),
        "snow6": wx.loc[dep, "wx_snow_6h_mm"].to_numpy(),
        "freez": wx.loc[dep, "wx_freezing_precip"].to_numpy(),
        "subzero": wx.loc[dep, "wx_subzero"].to_numpy(),
    })
    out["bad_wx"] = ((np.nan_to_num(out["snow6"]) > 0.1)
                     | (np.nan_to_num(out["freez"]) > 0.5))
    return out


def rmse(e2: np.ndarray) -> float:
    return float(np.sqrt(e2.mean())) if len(e2) else float("nan")


def paired_bootstrap(e2a: np.ndarray, e2b: np.ndarray, n: int = 2000, seed: int = 3):
    """CI for rmse(b) - rmse(a) on the same rows."""
    rng = np.random.default_rng(seed)
    m = len(e2a)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        d[i] = rmse(e2b[idx]) - rmse(e2a[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(ref: str, cfg: str, fold: str, boot: int = 2000) -> pd.DataFrame:
    a = pd.read_parquet(HERE / "oof" / f"{ref}_{fold}.parquet")
    b = pd.read_parquet(HERE / "oof" / f"{cfg}_{fold}.parquet")
    assert (a[D.ID].to_numpy() == b[D.ID].to_numpy()).all()
    wx = bad_weather_mask(fold).set_index(D.ID).loc[a[D.ID]].reset_index()
    feat = pd.read_parquet(CACHE / f"feat_{fold}_{D.FOLDS[fold].valid_month}.parquet")
    if not (feat[D.ID].to_numpy() == a[D.ID].to_numpy()).all():
        feat = feat.set_index(D.ID).loc[a[D.ID]].reset_index()
    e2a = ((a.y_true - a.y_pred) ** 2).to_numpy()
    e2b = ((b.y_true - b.y_pred) ** 2).to_numpy()
    y = a.y_true.to_numpy()
    bad = wx.bad_wx.to_numpy()
    nm = feat.has_nm.to_numpy() == 1
    segs = {
        "all": np.ones(len(y), bool),
        "bad weather": bad,
        "bad weather, bulk": bad & (y <= 3600),
        "bad weather, NM bulk": bad & nm & (y <= 3600),
        "other hours": ~bad,
        "other hours, NM bulk": (~bad) & nm & (y <= 3600),
    }
    rows = []
    for k, m in segs.items():
        if m.sum() < 30:
            continue
        lo, hi = paired_bootstrap(e2a[m], e2b[m], boot)
        rows.append({"segment": k, "n": int(m.sum()),
                     "ref": round(rmse(e2a[m]), 1), "cfg": round(rmse(e2b[m]), 1),
                     "delta": round(rmse(e2b[m]) - rmse(e2a[m]), 1),
                     "ci95": f"[{lo:+.1f}, {hi:+.1f}]"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ref, cfg, fold = sys.argv[1], sys.argv[2], sys.argv[3]
    pd.set_option("display.width", 200)
    print(f"== {cfg} vs {ref} on {fold}, exp-005 snow-or-freezing definition")
    print(compare(ref, cfg, fold).to_string(index=False))
