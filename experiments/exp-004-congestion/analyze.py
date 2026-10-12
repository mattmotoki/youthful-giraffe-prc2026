"""Segment breakdown for one saved OOF prediction file."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import prc.data as D
from run_ablation import CACHE

def load(name: str, fold: str):
    oof = pd.read_parquet(HERE / "oof" / f"{name}_{fold}.parquet")
    f = D.FOLDS[fold]
    feat = pd.read_parquet(CACHE / f"feat_{fold}_{f.valid_month}.parquet")
    if not (feat[D.ID].to_numpy() == oof[D.ID].to_numpy()).all():
        feat = feat.set_index(D.ID).loc[oof[D.ID]].reset_index()
    raw = D.load_month(f.valid_month, columns=[D.ID, "PHASE_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt"])
    raw = raw[raw.PHASE_mvt == "DEP"].set_index(D.ID).loc[oof[D.ID]].reset_index()
    oof["apt"] = raw["ADEP_mvt"].to_numpy()
    oof["noac"] = raw["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    oof["has_nm"] = feat["has_nm"].to_numpy()
    return oof

def rmse(g):
    return float(np.sqrt(np.mean((g.y_true - g.y_pred) ** 2)))

def table(oof: pd.DataFrame, other: pd.DataFrame | None = None):
    segs = {
        "all": np.ones(len(oof), bool),
        "bulk (<=3600)": (oof.y_true <= 3600).to_numpy(),
        "tail (>3600)": (oof.y_true > 3600).to_numpy(),
        "no-NM": (oof.has_nm == 0).to_numpy(),
        "no-NM bulk": ((oof.has_nm == 0) & (oof.y_true <= 3600)).to_numpy(),
        "noac": oof.noac.to_numpy(),
        "NM bulk": ((oof.has_nm == 1) & (oof.y_true <= 3600)).to_numpy(),
    }
    rows = []
    for k, m in segs.items():
        r = {"segment": k, "n": int(m.sum()), "rmse": round(rmse(oof[m]), 1)}
        if other is not None:
            r["ref"] = round(rmse(other[m]), 1)
            r["delta"] = round(r["rmse"] - r["ref"], 1)
        rows.append(r)
    return pd.DataFrame(rows)

def by_airport(oof, other=None):
    rows = []
    for a, g in oof.groupby("apt"):
        r = {"apt": a, "n": len(g), "rmse": round(rmse(g), 1),
             "bulk": round(rmse(g[g.y_true <= 3600]), 1)}
        if other is not None:
            o = other[other.apt == a]
            r["ref"] = round(rmse(o), 1)
            r["ref_bulk"] = round(rmse(o[o.y_true <= 3600]), 1)
            r["delta_bulk"] = round(r["bulk"] - r["ref_bulk"], 1)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("rmse", ascending=False)

if __name__ == "__main__":
    name, fold = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    o = load(name, fold)
    r = load(ref, fold) if ref else None
    pd.set_option("display.width", 200)
    print(f"== {name} on {fold}" + (f" vs {ref}" if ref else ""))
    print(table(o, r).to_string(index=False))
    print(by_airport(o, r).to_string(index=False))
