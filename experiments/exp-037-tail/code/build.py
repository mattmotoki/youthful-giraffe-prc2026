"""Build the chained departure frames: 2025 (twelve months) and the 388.

One row per departure of the ten ranking airports, carrying the chain
(`code/chain.py`) and the identifying signals the tables group by.  The 2025
frame carries the label; the 2026 frame cannot, because there is none.

The chain is built on a RANKING-MASKED frame in every case, so a 2025 month and
a ranking month see exactly the same columns.  Months never touch: the chain,
like every other per-day feature in this repo, is built one month at a time
(`prc.tail_model.signals_by_month`'s contract).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_features import tail_signals  # noqa: E402
from prc.tail_model import PLACEHOLDER_TOL  # noqa: E402
from chain import stand_chain  # noqa: E402

BAND = [-1e18, 0, 1800, 3600, 7200, 14400, 28800, 1e18]
LAB = ["neg", "0-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8h+"]
CHAIN_COLS = ["g_any", "g_strict", "g_prev2", "n_dep_between", "self_aobt_pre", "n_arr_stand"]


def add_band(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["band"] = pd.cut(d.mvt_sched, BAND, labels=LAB, right=False).astype(str)
    return d


def _one(masked: pd.DataFrame, raw: pd.DataFrame, month: str) -> pd.DataFrame:
    sig = tail_signals(masked)
    ch = stand_chain(masked).reindex(sig.index)
    out = pd.DataFrame(index=sig.index)
    out["MVT_ID_mvt"] = raw["MVT_ID_mvt"].reindex(sig.index).to_numpy("float64")
    out["month"] = month
    out["apt"] = sig["apt"].astype(str).to_numpy()
    out["stand"] = sig["stand"].astype(str).to_numpy()
    out["noac"] = sig["noac"].to_numpy("int8")
    out["nonm"] = sig["nonm"].to_numpy("int8")
    out["mvt_sched"] = sig["mvt_sched"].to_numpy("float64")
    out["mvt_aobt"] = sig["mvt_aobt"].to_numpy("float64")
    out["t_off"] = raw["MVT_TIME_UTC_mvt"].reindex(sig.index).to_numpy()
    out["day"] = pd.to_datetime(out["t_off"]).dt.strftime("%Y-%m-%d")
    for c in CHAIN_COLS:
        out[c] = ch[c].to_numpy()
    return add_band(out).reset_index(drop=True)


def build_2025() -> pd.DataFrame:
    frames = []
    for m in D.MONTHS:
        raw = D.load_month(m)
        masked, y = D.mask_like_ranking(raw)
        f = _one(masked, raw, m)
        idx = tail_signals(masked).index
        f["y_true"] = y.reindex(idx).to_numpy("float64")
        f["ph"] = (np.abs(f.mvt_sched - f.y_true) < PLACEHOLDER_TOL).astype("float64")
        assert f.y_true.notna().all()
        frames.append(f)
        print(f"{m}: {len(f)} dep, {int(f.noac.sum())} noac, chained "
              f"{f.g_any.notna().mean():.4f}", flush=True)
    return pd.concat(frames, ignore_index=True)


def build_2026() -> pd.DataFrame:
    rank = D.load_ranking()
    frames = []
    for m, g in rank.groupby("month", sort=True):
        f = _one(g, g, m)
        frames.append(f)
        print(f"{m}: {len(f)} dep, {int(f.noac.sum())} noac, chained "
              f"{f.g_any.notna().mean():.4f}", flush=True)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    (EXP / "work").mkdir(parents=True, exist_ok=True)
    b = build_2026()
    assert "y_true" not in b.columns and "ph" not in b.columns
    b.to_parquet(EXP / "work" / "dep_2026.parquet")
    print("2026 departures:", len(b), "noac:", int(b.noac.sum()))
    a = build_2025()
    a.to_parquet(EXP / "work" / "dep_2025.parquet")
    print("2025 departures:", len(a), "noac:", int(a.noac.sum()))
