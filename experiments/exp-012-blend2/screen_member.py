"""Screen any candidate OOF set as a blend member, on all three folds.

Same instrument as `screen_retro.py`: NNLS fitted IN SAMPLE on the scored fold
with the candidate added to the current core. In-sample is the most weight a
member can earn, so a candidate that earns little here earns less out of sample
and the negative reading is the trustworthy one. A candidate that earns a lot
here is a reason to run the honest rotated fit, not a result on its own.

Usage:  python screen_member.py <name>=<path-with-{fold}> [...]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc import blend as B

E8 = ROOT / "experiments/exp-008-diversity"
CORE = {
    "T-tailv2": "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":   "experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet",
    "L2-hub":   "experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet",
}
FOLDS = ("july_loo", "december", "january")
SHARE = {"july_loo": 0.5571, "january": 0.4429}


def load(path, fold, name):
    d = pd.read_parquet(ROOT / str(path).format(fold=fold))
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    keep = ["MVT_ID_mvt", "y_true", "y_pred"] + (["d_takeoff_aobt"] if "d_takeoff_aobt" in d else [])
    return d[keep].rename(columns={"y_pred": name})


def main(cands):
    board = {}
    for fold in FOLDS:
        lo, hi = json.loads((E8 / "guard_bounds.json").read_text())[fold]
        base = load(CORE["L2-hub"], fold, "L2-hub")
        for n, p in CORE.items():
            if n != "L2-hub":
                base = base.merge(load(p, fold, n)[["MVT_ID_mvt", n]], on="MVT_ID_mvt", how="inner")
        a = base["d_takeoff_aobt"].to_numpy("float64")
        y = base["y_true"].to_numpy("float64")
        prep = lambda v: B.apply_residual_guard(
            np.maximum(np.asarray(v, "float64"), B.FLOOR), a, float(lo), float(hi), B.FLOOR)
        names = list(CORE)
        P = np.stack([prep(base[n].to_numpy("float64")) for n in names], axis=1)
        rmse = lambda p: float(np.sqrt(((y - p) ** 2).mean()))
        w0 = B.fit_weights(P, y, "nnls")
        b0 = rmse(P @ w0)
        print(f"\n=== {fold}: core in-sample {b0:.3f}  w={np.round(w0,3)} ({names})")
        print(f"{'candidate':>16} {'alone':>8} {'weight':>7} {'blend':>8} {'gain':>7} {'corr':>6}")
        for n, p in cands.items():
            try:
                d = load(p, fold, "R")
            except Exception as e:
                print(f"{n:>16}   -- {type(e).__name__}: {e}"); continue
            j = base[["MVT_ID_mvt"]].merge(d[["MVT_ID_mvt", "R"]], on="MVT_ID_mvt", how="left")
            if j["R"].isna().any():
                print(f"{n:>16}   -- covers {int(j['R'].notna().sum())}/{len(j)} rows"); continue
            r = prep(j["R"].to_numpy("float64"))
            P2 = np.concatenate([P, r[:, None]], axis=1)
            w2 = B.fit_weights(P2, y, "nnls")
            b2 = rmse(P2 @ w2)
            c = float(np.corrcoef(y - (P @ w0), y - r)[0, 1])
            print(f"{n:>16} {rmse(r):8.2f} {w2[-1]:7.3f} {b2:8.3f} {b0-b2:7.3f} {c:6.3f}")
            board.setdefault(n, {})[fold] = (b0, b2)
        board.setdefault("__core__", {})[fold] = (b0, b0)
    print("\n=== board estimate, in sample (quadrature of july_loo and january) ===")
    for n, d in board.items():
        if not all(f in d for f in SHARE):
            continue
        e0 = np.sqrt(sum(SHARE[f] * d[f][0] ** 2 for f in SHARE))
        e2 = np.sqrt(sum(SHARE[f] * d[f][1] ** 2 for f in SHARE))
        print(f"  {n:>16}  core {e0:7.2f} -> {e2:7.2f}   gain {e0-e2:6.3f}")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
