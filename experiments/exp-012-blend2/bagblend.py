"""What does a seed-bagged member buy the BLEND, not the member?

The seed curve says 3 seeds buy 1.66 s on the Huber member itself. A blend
already averages across members, and averaging is exactly what bagging does, so
part of that 1.66 s may be gain the blend has already taken. Measured here by
substituting the k-seed bagged Huber into july_loo with the weights UNCHANGED
(they were fitted on december+january, which have no bagged members, so they
cannot be refitted honestly). Holding the weights fixed understates the gain,
because a better member would earn a larger weight; it is the conservative read.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import blend as B

FOLD = "july_loo"
E8 = ROOT / "experiments/exp-008-diversity"
bounds = json.loads((E8 / "guard_bounds.json").read_text())[FOLD]
lo, hi = float(bounds[0]), float(bounds[1])

paths = {
    "R0-retro": ROOT / f"experiments/exp-012-blend2/oof/R0-retro-{FOLD}.parquet",
    "T-tailv2": ROOT / "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":   E8 / f"oof/C2-d8p-{FOLD}.parquet",
    "X2-xgb":   E8 / f"oof/X2-d8p-{FOLD}.parquet",
    "L2-hub":   E8 / f"oof/L2-hub-{FOLD}.parquet",
}
frames = {}
for k, p in paths.items():
    d = pd.read_parquet(p)
    if "fold" in d.columns:
        d = d[d["fold"] == FOLD]
    keep = ["MVT_ID_mvt", "y_true", "y_pred"] + (
        ["d_takeoff_aobt"] if "d_takeoff_aobt" in d.columns else [])
    frames[k] = d[keep].rename(columns={"y_pred": k})

# d_takeoff_aobt (the residual guard's anchor) lives only in the exp-008 files;
# it is a property of the row, not of the member, so any one of them serves.
m = frames["L2-hub"][["MVT_ID_mvt", "y_true", "d_takeoff_aobt", "L2-hub"]]
for k in ("R0-retro", "T-tailv2", "C2-cat", "X2-xgb"):
    m = m.merge(frames[k][["MVT_ID_mvt", k]], on="MVT_ID_mvt", how="inner")
print(f"joined {len(m)} rows on {FOLD}")

# the k-seed bagged Huber, post-processed exactly as the arm reports
sb = pd.read_parquet(E8 / f"oof/SB-L2hub-{FOLD}.parquet")
cols = sorted([c for c in sb.columns if c.startswith("y_pred_s")],
              key=lambda c: int(c.split("s")[-1]))
print("seed columns:", cols)

names = ["R0-retro", "T-tailv2", "C2-cat", "X2-xgb", "L2-hub"]
# the rotated july_loo vector from RESULTS.md (fitted on december + january)
W = np.array([0.089, 0.376, 0.278, 0.000, 0.257])

a = m["d_takeoff_aobt"].to_numpy("float64")
y = m["y_true"].to_numpy("float64")

def prep(v):
    v = np.maximum(np.asarray(v, "float64"), B.FLOOR)
    return B.apply_residual_guard(v, a, lo, hi, B.FLOOR)

P = np.stack([prep(m[n].to_numpy("float64")) for n in names], axis=1)

def rmse(p):
    return float(np.sqrt(((y - p) ** 2).mean()))

base = rmse(P @ W)
print(f"\nbaseline blend (1-seed Huber, fixed rotated weights): {base:.3f}")
print(f"  member L2-hub alone: {rmse(P[:,4]):.3f}")

sb_idx = sb.set_index("MVT_ID_mvt")
order = m["MVT_ID_mvt"].to_numpy()
S = sb_idx.loc[order, cols].to_numpy("float64")

print(f"\n{'k':>2} {'member':>9} {'blend':>9} {'d_member':>9} {'d_blend':>9} {'kept':>6}")
for k in range(1, len(cols) + 1):
    mv, bv = [], []
    for idx in itertools.combinations(range(len(cols)), k):
        pk = prep(S[:, list(idx)].mean(axis=1))
        mv.append(rmse(pk))
        Pk = P.copy(); Pk[:, 4] = pk
        bv.append(rmse(Pk @ W))
    mm, bb = float(np.mean(mv)), float(np.mean(bv))
    if k == 1:
        m1, b1 = mm, bb
    kept = (b1 - bb) / (m1 - mm) if k > 1 else float("nan")
    print(f"{k:>2} {mm:9.3f} {bb:9.3f} {m1-mm:9.3f} {b1-bb:9.3f} {kept:6.2f}")

# --- upper bound on what refitting the weights could add -------------------
# The fixed-weight number understates because a better member earns more weight.
# Fitting NNLS on july_loo ITSELF is in-sample and therefore an upper bound, not
# an estimate: it is the most a refit could possibly extract on this fold.
print("\nin-sample NNLS refit (an UPPER BOUND, not an estimate):")
w1 = B.fit_weights(P, y, "nnls")
b1s = rmse(P @ w1)
print(f"  1-seed Huber   in-sample {b1s:.3f}  w={np.round(w1,3)}")
for k in (3, 5):
    idx = list(range(k))
    Pk = P.copy(); Pk[:, 4] = prep(S[:, idx].mean(axis=1))
    wk = B.fit_weights(Pk, y, "nnls")
    bks = rmse(Pk @ wk)
    print(f"  {k}-seed Huber   in-sample {bks:.3f}  w={np.round(wk,3)}  "
          f"upper-bound gain {b1s-bks:.3f}")
