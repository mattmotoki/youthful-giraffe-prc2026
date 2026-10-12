"""Cache per-month ERA5 weather features for the departure rows, keyed by MVT_ID."""
import time
import numpy as np, pandas as pd
import prc.data as D
import prc.weather_features as W
from prc.tail_features import RANKING_AIRPORTS

OUT = D.ROOT / "artifacts" / "exp-002-tail"
hourly = W.load_hourly()
for m in D.MONTHS:
    t0 = time.time()
    full = D.load_month(m)
    wx = W.transform(full, hourly)
    dep = full["PHASE_mvt"].to_numpy() == "DEP"
    keep = dep & full["ADEP_mvt"].isin(RANKING_AIRPORTS).to_numpy()
    out = wx[keep].copy()
    out.insert(0, "MVT_ID_mvt", full.loc[keep, "MVT_ID_mvt"].to_numpy())
    out.reset_index(drop=True).to_parquet(OUT / f"weather_{m}.parquet")
    print(f"{m}: {len(out)} rows, {out.shape[1]-1} weather columns, {time.time()-t0:.1f}s", flush=True)
