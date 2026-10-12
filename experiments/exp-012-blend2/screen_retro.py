"""Would a better retro-lineage member earn weight, and is december worth asking for?

Candidate 1's retro member `R0-retro` is a LINEAR-TREE, single-seed,
patience-100 arm scoring 324.76 on july_loo. retro-v1 has plain-tree arms nine
seconds better (`A0-plain-jloo` 315.2, `Q2-plain-jloo` 315.9) and the de-icing
arm at 314.3 -- but NONE of them has a `december` fold, so none can enter the
three-fold rotation. Before asking retro-v1 for that fold, this screens whether
the better member would earn weight at all: X2-xgb held the lowest correlations
in the matrix and still earned exactly zero, so "better and different" is not
enough on its own.

The weights here are fitted IN SAMPLE on the scored fold. That is a screen for
"does this member attract weight", not a score: an in-sample fit is the most
weight a member can possibly earn, so a member that earns little here earns
less out of sample, and the negative reading is the trustworthy one.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import blend as B

E8, E1 = ROOT / "experiments/exp-008-diversity", ROOT / "experiments/exp-001-retro-v1"
BASE = {
    "T-tailv2": ROOT / "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":   E8 / "oof/C2-d8p-{fold}.parquet",
    "L2-hub":   E8 / "oof/L2-hub-{fold}.parquet",
}
RETRO = {                      # candidates for the retro slot, per fold
    "july_loo": {
        "R0-retro (linear, shipped)": ROOT / "experiments/exp-012-blend2/oof/R0-retro-july_loo.parquet",
        "A0-plain":                   E1 / "oof/A0-plain-jloo-july_loo.parquet",
        "Q2-plain":                   E1 / "oof/Q2-plain-jloo-july_loo.parquet",
        "X1-deice":                   E1 / "oof/X1-deice-july_loo.parquet",
        "X2-wint3":                   E1 / "oof/X2-wint3-july_loo.parquet",
    },
    "january": {
        "R0-retro (linear, shipped)": ROOT / "experiments/exp-012-blend2/oof/R0-retro-january.parquet",
        "Q2-plain":                   E1 / "oof/Q2-plain-jan-january.parquet",
        "X1-deice":                   E1 / "oof/X1-deice-january.parquet",
        "X2-wint3":                   E1 / "oof/X2-wint3-january.parquet",
    },
}

def load(p, fold, name):
    d = pd.read_parquet(p)
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    keep = ["MVT_ID_mvt", "y_true", "y_pred"] + (["d_takeoff_aobt"] if "d_takeoff_aobt" in d else [])
    return d[keep].rename(columns={"y_pred": name})

for fold in ("july_loo", "january"):
    lo, hi = json.loads((E8 / "guard_bounds.json").read_text())[fold]
    base = load(str(BASE["L2-hub"]).format(fold=fold), fold, "L2-hub")
    for n, p in BASE.items():
        if n == "L2-hub":
            continue
        base = base.merge(load(str(p).format(fold=fold), fold, n)[["MVT_ID_mvt", n]],
                          on="MVT_ID_mvt", how="inner")
    a = base["d_takeoff_aobt"].to_numpy("float64")
    y = base["y_true"].to_numpy("float64")
    prep = lambda v: B.apply_residual_guard(
        np.maximum(np.asarray(v, "float64"), B.FLOOR), a, float(lo), float(hi), B.FLOOR)
    names3 = ["T-tailv2", "C2-cat", "L2-hub"]
    P3 = np.stack([prep(base[n].to_numpy("float64")) for n in names3], axis=1)
    rmse = lambda p: float(np.sqrt(((y - p) ** 2).mean()))
    w3 = B.fit_weights(P3, y, "nnls")
    b3 = rmse(P3 @ w3)
    print(f"\n=== {fold}: 3-member in-sample floor {b3:.3f}   w={np.round(w3,3)} ({names3})")
    print(f"{'retro candidate':>28} {'alone':>8} {'weight':>7} {'blend':>8} {'gain':>7} {'corr':>6}")
    for n, p in RETRO[fold].items():
        try:
            d = load(p, fold, "R")
        except Exception as e:
            print(f"{n:>28}   -- {e}"); continue
        j = base[["MVT_ID_mvt"]].merge(d[["MVT_ID_mvt", "R"]], on="MVT_ID_mvt", how="left")
        if j["R"].isna().any():
            print(f"{n:>28}   -- covers {int(j['R'].notna().sum())}/{len(j)} rows"); continue
        r = prep(j["R"].to_numpy("float64"))
        P4 = np.concatenate([P3, r[:, None]], axis=1)
        w4 = B.fit_weights(P4, y, "nnls")
        b4 = rmse(P4 @ w4)
        # residual correlation against the blend the member would join
        e_b, e_r = y - (P3 @ w3), y - r
        c = float(np.corrcoef(e_b, e_r)[0, 1])
        print(f"{n:>28} {rmse(r):8.2f} {w4[-1]:7.3f} {b4:8.3f} {b3-b4:7.3f} {c:6.3f}")
