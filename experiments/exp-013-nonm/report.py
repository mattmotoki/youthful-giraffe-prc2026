"""The exp-013 result table, from the saved OOF files.

January carries two LFPG date-rollover rows (`BLOCK_TIME` stamped about a day
before `SCHED_TIME`) whose targets are 58,206 s and 84,240 s against a segment
sd of 803 s without them. They are identifiable from `BLOCK − SCHED` alone, so
the trimmed column is a pre-declared segment definition rather than a trim
chosen after seeing errors; both are reported.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
from segment import month_segment, rmse

FOLDS = ["july_loo", "december", "january", "july"]
ORDER = ["base", "op", "stand", "surface", "bracket", "sb", "sbo", "all"]
BASELINE = {"july": 651.4, "july_loo": 569.8, "december": 493.0, "january": 2794.2}
ROLLOVER = -20000.0  # BLOCK - SCHED below this is a date rollover, not a taxi


def rollover_ids(fold: str) -> set:
    s = month_segment(D.FOLDS[fold].valid_month)
    return set(s.loc[s["block_sched"] < ROLLOVER, D.ID].to_numpy())


def table() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        try:
            bad = rollover_ids(fold)
        except FileNotFoundError:
            continue
        for cfg in ORDER:
            p = HERE / "oof" / f"{cfg}_{fold}.parquet"
            if not p.exists():
                continue
            o = pd.read_parquet(p)
            seg = o["seg"].to_numpy().astype(bool)
            s = o[seg]
            keep = ~s[D.ID].isin(bad).to_numpy()
            rows.append({
                "fold": fold, "config": cfg, "n": int(seg.sum()),
                "segment": round(rmse(s["y_true"], s["y_pred"]), 1),
                "segment_trim": round(rmse(s["y_true"][keep], s["y_pred"][keep]), 1),
                "n_trim": int(keep.sum()),
                "all_rows": round(rmse(o["y_true"], o["y_pred"]), 1),
                "vs_baseline": round(rmse(s["y_true"], s["y_pred"]) - BASELINE[fold], 1),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = table()
    print(t.to_string(index=False))
    print("\nbest per fold on the trimmed segment:")
    if len(t):
        print(t.loc[t.groupby("fold")["segment_trim"].idxmin()].to_string(index=False))
