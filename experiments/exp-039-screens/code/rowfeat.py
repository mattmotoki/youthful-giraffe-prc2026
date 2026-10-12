"""Row-only linear features for a validation month, from masked inputs only.

Every screen that needs departure-level inputs reads them here, so all four use
one masking story.  `prc.data.mask_like_ranking` blanks departure
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` -- the B and y of the measurement
contract -- and this module is called on the masked frame, so no feature can
descend from either.  The take-off clock `MVT_TIME_UTC_mvt` and every `_flt`
column survive the mask and are legal inputs (`docs/input-rule.md`);
`AOBT_3_flt` is NM's off-block estimate and is not the masked B.

The features are deliberately thin and linear: Q03 asks for "row-only linear
features" and a fixed ridge, not a model.  Categorical levels are frozen from
the month's own frame by frequency, which uses no label.

    F = features(month)      # index = MVT_ID_mvt, float32 columns
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

#: The clock differences.  Each is `left - right` in seconds; all four clocks on
#: the right are inputs in the ranking file.
DIFFS = [
    ("t_aobt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"),
    ("t_sched", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
    ("t_lobt", "MVT_TIME_UTC_mvt", "LOBT_flt"),
    ("t_eobt", "MVT_TIME_UTC_mvt", "EOBT_1_flt"),
    ("t_iobt", "MVT_TIME_UTC_mvt", "IOBT_flt"),
    ("aobt_lobt", "AOBT_3_flt", "LOBT_flt"),
    ("aobt_sched", "AOBT_3_flt", "SCHED_TIME_UTC_mvt"),
    ("aobt_eobt", "AOBT_3_flt", "EOBT_1_flt"),
    ("lobt_sched", "LOBT_flt", "SCHED_TIME_UTC_mvt"),
    ("iobt_lobt", "IOBT_flt", "LOBT_flt"),
]
#: Robust clip, in seconds: a linear learner must not be steered by one row.
CLIP = 7200.0
#: How many levels each categorical keeps; the rest fall into the implicit base.
TOPN = {"AIRCRAFT_TYPE_mvt": 30, "RUNWAY_mvt": 40, "pier": 40,
        "AIRCRAFT_OPERATOR_flt": 40, "MARKET_SEGMENT_flt": 12,
        "WK_TBL_CAT_flt": 8, "FLIGHT_TYPE_flt": 8, "FLIGHT_RULE_mvt": 4}


def departures(month: str) -> pd.DataFrame:
    """The month's DEP rows with B and y blanked exactly as the ranking file does."""
    raw = D.load_month(month)
    masked, _ = D.mask_like_ranking(raw)
    del raw
    dep = masked[masked["PHASE_mvt"] == "DEP"].copy()
    del masked
    if dep["BLOCK_TIME_UTC_mvt"].notna().any() or dep[D.TARGET].notna().any():
        raise SystemExit(f"{month}: departure B/y survived the ranking mask")
    return dep.reset_index(drop=True)


def features(dep: pd.DataFrame) -> pd.DataFrame:
    """Row-only linear features, indexed by `MVT_ID_mvt`."""
    out = {}
    for name, a, b in DIFFS:
        v = (dep[a] - dep[b]).dt.total_seconds().to_numpy("float64")
        out[f"d_{name}"] = np.clip(np.nan_to_num(v, nan=0.0), -CLIP, CLIP)
        out[f"miss_{name}"] = np.isnan(v).astype("float64")
    t = dep["MVT_TIME_UTC_mvt"]
    sec = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy("float64")
    out["tod_sin"] = np.sin(2 * np.pi * sec / 86400.0)
    out["tod_cos"] = np.cos(2 * np.pi * sec / 86400.0)
    out["dow"] = t.dt.dayofweek.to_numpy("float64")
    for d in range(6):
        out[f"dow_{d}"] = (t.dt.dayofweek.to_numpy() == d).astype("float64")
    pier = dep["STAND_mvt"].astype("string").fillna("").str.slice(0, 2)
    cats = {"AIRCRAFT_TYPE_mvt": dep["AIRCRAFT_TYPE_mvt"].astype("string"),
            "RUNWAY_mvt": dep["RUNWAY_mvt"].astype("string"),
            "pier": pier,
            "AIRCRAFT_OPERATOR_flt": dep["AIRCRAFT_OPERATOR_flt"].astype("string"),
            "MARKET_SEGMENT_flt": dep["MARKET_SEGMENT_flt"].astype("string"),
            "WK_TBL_CAT_flt": dep["WK_TBL_CAT_flt"].astype("string"),
            "FLIGHT_TYPE_flt": dep["FLIGHT_TYPE_flt"].astype("string"),
            "FLIGHT_RULE_mvt": dep["FLIGHT_RULE_mvt"].astype("string")}
    for col, s in cats.items():
        s = s.fillna("<na>")
        for lvl in s.value_counts().index[:TOPN[col]]:
            out[f"{col}={lvl}"] = (s == lvl).to_numpy().astype("float64")
    F = pd.DataFrame(out, index=dep["MVT_ID_mvt"].astype("float64").to_numpy())
    F.index.name = "MVT_ID_mvt"
    return F.astype("float32")


def month_of(fold: str) -> str:
    return D.FOLDS[fold].valid_month
