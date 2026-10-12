"""Load the raw challenge files into one compact frame (exp-171 first pass).

Times become int64 epoch seconds (NaN -> -1 sentinel is NOT used; they stay as
float64 seconds with NaN). Strings become categoricals. The source file and the
row position in the file are kept, because the file order is one of the things
under test.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.environ.get("EXP171_CACHE", "/tmp/exp171-cache")
TIME_COLS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt",
    "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]
SHORT = {
    "MVT_TIME_UTC_mvt": "mvt", "BLOCK_TIME_UTC_mvt": "block", "SCHED_TIME_UTC_mvt": "sched",
    "LOBT_flt": "lobt", "IOBT_flt": "iobt", "EOBT_1_flt": "eobt", "ARVT_1_flt": "arvt1",
    "AOBT_3_flt": "aobt", "ARVT_3_flt": "arvt3",
}
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]


def _read(path, src):
    t = pq.read_table(path)
    d = t.to_pandas()
    d["src"] = src
    d["pos"] = np.arange(len(d), dtype=np.int64)
    for c in TIME_COLS:
        v = d[c].astype("int64").to_numpy()  # microseconds; NaT -> min int
        na = d[c].isna().to_numpy()
        sec = v // 10**6
        d[SHORT[c] + "_us"] = np.where(na, 0, v % 10**6).astype(np.int32)
        d[SHORT[c]] = np.where(na, np.nan, sec.astype("float64"))
        del d[c]
    return d


def load(refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "all.pkl")
    if os.path.exists(p) and not refresh:
        return pd.read_pickle(p)
    parts = []
    for f in sorted(glob.glob(os.path.join(RAW, "training_2025-*.parquet"))):
        parts.append(_read(f, os.path.basename(f)[9:16]))
    parts.append(_read(os.path.join(RAW, "ranking.parquet"), "ranking"))
    d = pd.concat(parts, ignore_index=True)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype("category")
    d["src"] = d["src"].astype("category")
    d["is_rank"] = (d["src"] == "ranking").to_numpy()
    d["dep"] = (d["PHASE_mvt"] == "DEP").to_numpy()
    # reporting airport
    d["apt"] = np.where(d["dep"], d["ADEP_mvt"].astype(object), d["ADES_mvt"].astype(object))
    d["apt"] = d["apt"].astype("category")
    d["taxi"] = d["TAXITIME_SEC_mvt"].astype("float64")
    d.to_pickle(p)
    return d


if __name__ == "__main__":
    d = load(refresh="--refresh" in sys.argv)
    print(d.shape)
    print(d.dtypes)
    print(d.memory_usage(deep=True).sum() / 1e9, "GB")
