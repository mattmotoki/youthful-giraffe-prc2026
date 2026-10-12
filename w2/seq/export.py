"""Average the s1 seed files and align them to the chain rows.

    python3 w2/seq/export.py s1 [work/seq]         -> work/w2/s1_fold.parquet (held fits), work/w2/s1_ranking.parquet (full fits)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work/seq"
ID = "MVT_ID_mvt"


def main(name, data="work/seq"):
    global D
    D = ROOT / data
    ids = np.load(D / "ids.npy"); is26 = np.load(D / "is2026.npy")
    for mode, split, sel in (("held", "fold", ~is26), ("full", "ranking", is26)):
        files = sorted(glob.glob(str(D / f"{name}-{mode}-*.npy")))
        if not files:
            print(split, "no seed files"); continue
        v = np.mean([np.load(f).astype(float) for f in files], 0)
        s = pd.Series(v[sel], index=ids[sel])
        chain = pd.read_parquet(ROOT / f"work/w2/chain_ids_{split}.parquet")[ID].to_numpy()
        out = pd.DataFrame({ID: chain, name: s.reindex(chain).to_numpy()})
        out.to_parquet(ROOT / f"work/w2/{name}_{split}.parquet", index=False)
        print(split, len(files), "seeds", "finite", int(np.isfinite(out[name]).sum()), "of", len(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
