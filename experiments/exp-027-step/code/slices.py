"""The four RMSE slices `assignments/COMMON.md` requires of every RESULTS table:
overall, y <= 3600, y > 3600, and the no-NM rows, for each fold.

Round-1 review, blocking finding 7: the head and K1 tables reported the board
and the no-noac board only. This file supplies the missing slices for both heads
and for the K1 arm, on the clean population the board estimate is built from.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from window_probe import load, FOLDS  # noqa: E402


def rms(e) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, "float64") ** 2)))


def table(head: str, adj: dict | None = None, label: str = "") -> str:
    rows = []
    for f in FOLDS:
        d = load(f, head)
        e = (d["y_pred"] - d["y_true"]).to_numpy()
        if adj is not None:
            e = e - adj[f]
        lo, hi = (d["y_true"] <= 3600).to_numpy(), (d["y_true"] > 3600).to_numpy()
        nm = (d["nm_missing"] == 1).to_numpy()
        rows.append(f"| {label or head} | {f} | {len(d):,} | {rms(e):.3f} | "
                    f"{rms(e[lo]):.3f} | {rms(e[hi]):.3f} | {rms(e[nm]):.3f} |")
    return "\n".join(rows)


HDR = ("| head | fold | n (clean) | RMSE all | RMSE y ≤ 3600 | RMSE y > 3600 | RMSE no-NM |\n"
       "|---|---|---|---|---|---|---|")

if __name__ == "__main__":
    print(HDR)
    for h in ("blend5", "stack5"):
        print(table(h))
