"""The exp-013 segment and the baselines measured on it.

Segment: departures with no Network Manager join (`LOBT_flt` null, which makes
every `_flt` column null) and a present `AIRCRAFT_TYPE_mvt`. The noac rows are
excluded — they are exp-002's and exp-009's lane and have a different
mechanism (a placeholder block time equal to the schedule).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D

FOLDS = ("july", "july_loo", "december", "january")


def segment_mask(dep: pd.DataFrame) -> np.ndarray:
    """no NM join and aircraft type present, on a departures-only frame."""
    return (dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].notna()).to_numpy()


def month_segment(month: str) -> pd.DataFrame:
    """Segment rows of one month with the quantities the lane turns on."""
    m = D.load_month(month)
    dep = m[m["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    seg = dep[segment_mask(dep)].reset_index(drop=True)
    out = pd.DataFrame({
        D.ID: seg[D.ID].to_numpy(),
        "y": seg[D.TARGET].to_numpy(float),
        "apt": seg["ADEP_mvt"].astype(str).to_numpy(),
        "mvt_sched": (seg["MVT_TIME_UTC_mvt"] - seg["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(),
        "block_sched": (seg["BLOCK_TIME_UTC_mvt"] - seg["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(),
    })
    return out


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def baseline(oof_path: Path, fold: str) -> dict:
    """RMSE of a saved OOF on the segment of that fold's validation month."""
    o = pd.read_parquet(oof_path)
    if "fold" in o.columns and o["fold"].nunique() > 1:
        o = o[o["fold"] == fold]
    seg = month_segment(D.FOLDS[fold].valid_month).set_index(D.ID)
    j = o.set_index(D.ID).join(seg[["y"]], how="inner", rsuffix="_seg")
    if not len(j):
        return {}
    return {"n": len(j), "rmse": rmse(j["y_true"], j["y_pred"])}
