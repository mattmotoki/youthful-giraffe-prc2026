"""Local evaluation on the held January and July 2025 departures (v64's own protocol).

G = sum_i w_i [(y_i - base_i)^2 - (y_i - cand_i)^2], in s^2 per 2026 ranking row; w_i is the
saved exposure weight (2026 cell share / 2025 cell rows, cell = month x airport x class; zero
on unsupported cells). Positive G = the candidate is better. Interval: 2,000 draws of
airport-days with replacement within airport x month (seed 601), percentile 95 %.
Also: wo5 (G without the five airport-days that add most), and G by month, class, airport.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLD = ROOT / "work/chain/fold.parquet"
_CACHE = {}


def fold() -> pd.DataFrame:
    if "fold" not in _CACHE:
        d = pd.read_parquet(FOLD)
        raw = pd.concat([pd.read_parquet(ROOT / f"input-root/data/raw/training_2025-{m}-01_2025-{m2}-01.parquet",
                                         columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
                         for m, m2 in (("01", "02"), ("07", "08"))])
        raw = raw[raw.PHASE_mvt.eq("DEP")].set_index("MVT_ID_mvt")
        day = raw.MVT_TIME_UTC_mvt.reindex(d.MVT_ID_mvt.to_numpy()).dt.strftime("%m-%d").to_numpy()
        d["day"] = day
        d["aday"] = d.airport.astype(str) + "|" + d.day
        _CACHE["fold"] = d
    return _CACHE["fold"]


def _draws(d: pd.DataFrame, n_draws=2000, seed=601):
    key = ("draws", n_draws, seed)
    if key in _CACHE:
        return _CACHE[key]
    days = d[["airport", "month", "aday"]].drop_duplicates().sort_values("aday").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_draws, len(days)))
    for (_, _), grp in days.groupby(["airport", "month"]):
        idx = grp.index.to_numpy()
        pick = rng.integers(0, len(idx), size=(n_draws, len(idx)))
        for b in range(n_draws):
            counts[b, idx] += np.bincount(pick[b], minlength=len(idx))
    code = pd.Series(np.arange(len(days)), index=days.aday)
    _CACHE[key] = (counts, code)
    return _CACHE[key]


def score(cand, base=None, d: pd.DataFrame | None = None, n_draws=2000, views=True) -> dict:
    """G of cand against base (default v64) on the held months."""
    d = fold() if d is None else d
    base = d.v64.to_numpy(float) if base is None else np.asarray(base, float)
    cand = np.asarray(cand, float)
    assert cand.shape == base.shape and np.isfinite(cand).all()
    y = d.y.to_numpy(float); w = d.w.to_numpy(float)
    contrib = w * ((y - base) ** 2 - (y - cand) ** 2)
    G = float(contrib.sum())
    counts, code = _draws(d, n_draws)
    per_day = pd.Series(contrib).groupby(code.reindex(d.aday).to_numpy()).sum()
    S = np.zeros(len(code)); S[per_day.index.to_numpy()] = per_day.to_numpy()
    # A draw resamples airport-days; cell means are approximated by weighting each sampled day's
    # contribution (the weights already carry 1/n_cell); this is the usual day-block bootstrap.
    boot = counts @ S
    lo, hi = np.percentile(boot, [2.5, 97.5])
    top = per_day.sort_values(ascending=False)
    wo5 = G - float(top.head(5).sum())
    out = {"G": G, "lo": float(lo), "hi": float(hi), "p_le0": float((boot <= 0).mean()), "wo5": wo5,
           "changed": int((cand != base).sum())}
    if views:
        for col in ("month", "cls", "airport"):
            out[col] = {str(k): round(float(v), 2) for k, v in pd.Series(contrib).groupby(d[col].to_numpy()).sum().items()}
    return out


def rmse(pred, d: pd.DataFrame | None = None) -> float:
    """Ranking-weighted held RMSE (cells weighted by their 2026 share; supported cells only)."""
    d = fold() if d is None else d
    w = d.w.to_numpy(float); y = d.y.to_numpy(float)
    return float(np.sqrt((w * (y - np.asarray(pred, float)) ** 2).sum() / w.sum()))


def fmt(r: dict) -> str:
    return (f"G {r['G']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] P(G<=0) {r['p_le0']:.3f} wo5 {r['wo5']:+.2f} "
            f"changed {r['changed']}" + (f" month {r.get('month')} cls {r.get('cls')}" if 'month' in r else ""))
