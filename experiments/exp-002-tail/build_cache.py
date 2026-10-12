"""Cache per-month tail signals + labels. Features are identical on masked and
unmasked frames (verified below), so one pass serves both roles."""
import sys, time
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_features import tail_signals

OUT = D.ROOT / "artifacts" / "exp-002-tail"
OUT.mkdir(parents=True, exist_ok=True)

def build(month):
    full = D.load_month(month)
    masked, labels = D.mask_like_ranking(full)
    s = tail_signals(masked)
    # leakage check: same signals from the unmasked frame
    s2 = tail_signals(full)
    num = [c for c in s.columns if s[c].dtype.kind in "fiu"]
    assert s.shape == s2.shape and np.allclose(s[num].to_numpy(dtype=float), s2[num].to_numpy(dtype=float), equal_nan=True), month
    s = s.copy()
    s["y"] = labels.reindex(s.index).astype(float)
    s["MVT_ID_mvt"] = full["MVT_ID_mvt"].reindex(s.index)
    s["month"] = month
    s["blk_sched_true"] = (full["BLOCK_TIME_UTC_mvt"] - full["SCHED_TIME_UTC_mvt"]).dt.total_seconds().reindex(s.index)
    s["blk_aobt_true"] = (full["BLOCK_TIME_UTC_mvt"] - full["AOBT_3_flt"]).dt.total_seconds().reindex(s.index)
    s.reset_index(drop=True).to_parquet(OUT / f"signals_{month}.parquet")
    return len(s)

if __name__ == "__main__":
    for m in D.MONTHS:
        t0 = time.time(); n = build(m)
        print(f"{m}: {n} departures, {time.time()-t0:.1f}s", flush=True)
