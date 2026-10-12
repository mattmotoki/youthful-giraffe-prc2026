"""Cache the c5 values (held fold and 2026 ranking) with the chain columns and the row features.

    .venv/bin/python w2/base.py      -> work/w2/base_{fold,ranking}.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso"))
OUT = ROOT / "work/w2"
C5_STEPS = ["ext", "long", "lirf", "noac", "roll"]


def main():
    import candidate as C
    base, out, info = C.build(C5_STEPS)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("fold", "ranking"):
        ch = pd.read_parquet(ROOT / f"work/chain/{split}.parquet")
        ch["c5"] = out[split]["v64"]
        ch["c5_primary"] = out[split]["primary"]
        ch["c5_lower"] = out[split]["lower"]
        ch["c5_upper"] = out[split]["upper"]
        ch.to_parquet(OUT / f"base_{split}.parquet", index=False)
        print(split, len(ch), "changed vs v64", int((ch.c5 != ch.v64).sum()))


def load(split: str) -> pd.DataFrame:
    """Chain columns + c5 + row features, in chain row order."""
    b = pd.read_parquet(OUT / f"base_{split}.parquet")
    f = pd.read_parquet(ROOT / f"work/feat/{split}.parquet")
    assert np.array_equal(b.MVT_ID_mvt.to_numpy(), f.MVT_ID_mvt.to_numpy())
    f = f.drop(columns=["MVT_ID_mvt", "airport"])
    return pd.concat([b, f], axis=1)


if __name__ == "__main__":
    main()
