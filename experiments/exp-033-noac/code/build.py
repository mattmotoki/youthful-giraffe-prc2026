"""Build the noac population frames.

Sol S1 and round 2 R2-1: step 1 must be label-free, so this module NEVER READS A
LABEL.  It builds `noac_2025.parquet` and `noac_2026.parquet`, which carry no
target, no departure block time and no placeholder class, and it asserts that at
the end.  `join_labels.py` -- the §5 boundary -- is the only module in this
experiment that opens a label column at all; the first version of this file
extracted the labels here and merely stored them elsewhere, which is a storage
separation and not a boundary.

Everything in the label-free frame is a `prc.tail_features.tail_signals` column
plus identifying keys.  Signals are built one month at a time, so the ranking
file's January and July never touch (`prc.tail_model.signals_by_month`'s
contract).  `arr_day_ph_rate` is an ARRIVAL-side aggregate and a legal ranking
input (`docs/input-rule.md`); it is not a departure label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402
from prc.tail_features import tail_signals  # noqa: E402
# NO label import here: `join_labels.py` owns every label (Sol round 2, R2-1).

OUT = Path(__file__).resolve().parents[1] / "work"


def _noac_frame(masked: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Label-free: `tail_signals` on the ranking-masked frame, plus keys."""
    sig = tail_signals(masked)
    out = sig.loc[sig["noac"].to_numpy() == 1].copy()
    idx = out.index
    out["MVT_ID_mvt"] = raw["MVT_ID_mvt"].reindex(idx).to_numpy("float64")
    out["FLIGHT_mvt"] = raw["FLIGHT_mvt"].reindex(idx).to_numpy()
    out["month"] = raw["month"].reindex(idx).to_numpy()
    out["mvt_time"] = raw["MVT_TIME_UTC_mvt"].reindex(idx).to_numpy()
    out["sched_time"] = raw["SCHED_TIME_UTC_mvt"].reindex(idx).to_numpy()
    out["adep"] = raw["ADEP_mvt"].reindex(idx).to_numpy()
    out["ades"] = raw["ADES_mvt"].reindex(idx).to_numpy()
    out["operator"] = raw["AIRCRAFT_OPERATOR_flt"].reindex(idx).to_numpy()
    out["segment"] = raw["MARKET_SEGMENT_flt"].reindex(idx).to_numpy()
    return out


def build_2025() -> pd.DataFrame:
    frames = []
    for m in D.MONTHS:
        raw = D.load_month(m)
        masked, _ = D.mask_like_ranking(raw)
        frames.append(_noac_frame(masked, raw))
        print(f"{m}: {len(frames[-1])} noac", flush=True)
    return pd.concat(frames, ignore_index=True)


def build_2026() -> pd.DataFrame:
    rank = D.load_ranking()
    parts = [_noac_frame(g, g) for _, g in rank.groupby("month", sort=True)]
    for m, p in zip(sorted(rank.month.unique()), parts):
        print(f"{m}: {len(p)} noac", flush=True)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = build_2025()
    a.to_parquet(OUT / "noac_2025.parquet")
    print("2025 noac rows:", len(a))
    b = build_2026()
    b.to_parquet(OUT / "noac_2026.parquet")
    print("2026 noac rows:", len(b))
    for f in ("noac_2025", "noac_2026"):
        d = pd.read_parquet(OUT / f"{f}.parquet")
        bad = [c for c in d.columns if c in ("y_true", "ph", "ph_minute", "block_time",
                                             "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")]
        assert not bad, f"{f} carries a label column: {bad}"
    print("label-free check: both frames clean")
