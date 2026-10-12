"""Land the K2 fold files under their canonical names, and price the member.

The Kaggle kernels write `kaggle-k2-window*-<fold>.parquet`; `code/candidate.py`
reads `oof/k2-window-<fold>.parquet`.  This renames them and prints the member
level A/B beside the committed F1-fresh, with the environment caveat attached to
the two folds that have no Kaggle baseline of their own.

    python land_k2.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
WS = "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet"
#: The like-for-like Kaggle baseline, where one exists.  exp-025 measured the
#: same code on the same bytes reading 293.9248 here and 293.6500 on the
#: workstation, a 0.275 s environment band; December and January have no Kaggle
#: baseline, so their A/B carries that band and the rotation is what decides.
KAGGLE_BASE = {"july_loo": 293.9248}


def land() -> None:
    oof = EXP / "oof"
    for f in FOLDS:
        dst = oof / f"k2-window-{f}.parquet"
        if dst.exists():
            continue
        src = [p for p in oof.glob(f"k2*{f}.parquet") if p != dst]
        if not src:
            print(f"[land] {f}: no pulled file yet")
            continue
        if len(src) > 1:
            raise SystemExit(f"{f}: {len(src)} candidate files: {[p.name for p in src]}")
        shutil.copy(src[0], dst)
        print(f"[land] {f}: {src[0].name} -> {dst.name}")


def price() -> None:
    print(f"\n{'fold':10s} {'n':>8s}  {'F1-fresh':>10s} {'K2':>10s} {'gain':>8s}   "
          f"{'clean F1':>10s} {'clean K2':>10s} {'gain':>8s}   baseline")
    for f in FOLDS:
        p = EXP / "oof" / f"k2-window-{f}.parquet"
        if not p.exists():
            print(f"{f:10s} pending")
            continue
        k = pd.read_parquet(p)
        w = pd.read_parquet(ROOT / WS.format(fold=f))
        m = w[["MVT_ID_mvt", "y_true", "y_pred"]].merge(
            k[["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": "k2"}),
            on="MVT_ID_mvt", how="inner")
        if len(m) != len(w):
            raise SystemExit(f"{f}: K2 covers {len(m)} of {len(w)} rows")
        y = m["y_true"].to_numpy("float64")
        a = m["y_pred"].to_numpy("float64")
        b = m["k2"].to_numpy("float64")
        ra = float(np.sqrt(((a - y) ** 2).mean()))
        rb = float(np.sqrt(((b - y) ** 2).mean()))
        ca = D.rmse_clean(y, a, m["MVT_ID_mvt"], f)
        cb = D.rmse_clean(y, b, m["MVT_ID_mvt"], f)
        base = KAGGLE_BASE.get(f)
        note = (f"Kaggle {base:.4f}, gain {base - rb:+.4f} like for like"
                if base else "no Kaggle baseline: this gain carries the 0.275 s band")
        print(f"{f:10s} {len(m):8d}  {ra:10.4f} {rb:10.4f} {ra - rb:+8.4f}   "
              f"{ca:10.4f} {cb:10.4f} {ca - cb:+8.4f}   {note}")


if __name__ == "__main__":
    land()
    price()
