"""Add the segment keys to OOF files written before they were recorded.

Only used for the four configs whose feature columns are byte-identical
between feature generations (base, +queue, +pressure, +stand); everything else
was re-run. The keys are joined from the rebuilt month cache by MVT_ID.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
from run_ablation import CACHE

SEG = ["seg_noac", "seg_apt", "seg_day", "has_nm"]

for fold in ["july", "december", "january"]:
    feat = pd.read_parquet(CACHE / f"feat_{fold}_{D.FOLDS[fold].valid_month}.parquet",
                           columns=[D.ID] + SEG).set_index(D.ID)
    for p in sorted((HERE / "oof").glob(f"*_{fold}.parquet")):
        o = pd.read_parquet(p)
        if "seg_apt" in o.columns:
            continue
        s = feat.loc[o[D.ID]]
        o["seg_noac"] = s["seg_noac"].to_numpy()
        o["seg_apt"] = s["seg_apt"].to_numpy()
        o["seg_day"] = s["seg_day"].to_numpy()
        o["seg_has_nm"] = s["has_nm"].to_numpy()
        o.to_parquet(p, index=False)
        print("backfilled", p.name)
