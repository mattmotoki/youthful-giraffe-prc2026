"""exp-168 unit 11 (rule hunt): the raw 2025 rows and the 2026 ranking rows with the keys of the scans.
Read-only looks, no fit. `pre` is the airline part of the airport's flight string (leading letters, at most 3;
a string that starts with a digit keeps its first 2 characters)."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import train, ranking, secs, ROOT  # noqa: E402
AP = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 600)


def keys(d):
    d = d.copy(); dep = d.PHASE_mvt == "DEP"
    d["ap"] = np.where(dep, d.ADEP_mvt, d.ADES_mvt).astype(str); d["mon"] = d.MVT_TIME_UTC_mvt.dt.month; d["yr"] = d.MVT_TIME_UTC_mvt.dt.year
    f = d.FLIGHT_mvt.fillna("").astype(str); p = f.str.extract(r"^([A-Z]{1,3})")[0]; d["pre"] = p.where(p.notna(), f.str[:2])
    d["y"] = d.TAXITIME_SEC_mvt.astype(float); d["plan"] = d.LOBT_flt.notna()
    return d


def raw2025(months=None):
    return keys(train(months))


def rank2026():
    return keys(ranking())


def near(x, c, w=6):
    """|x - c| <= w, False where x is missing."""
    return (x - c).abs() <= w


def spike(x, w=6, ks=(1, 2, 3)):
    """Rows with x within w of 0, and the mean number of rows within w of +-60 k s (the chance level of a whole-minute hit)."""
    hit = int(near(x, 0, w).sum()); ch = float(np.mean([near(x, s * 60 * k, w).sum() for k in ks for s in (-1, 1)]))
    return hit, ch
