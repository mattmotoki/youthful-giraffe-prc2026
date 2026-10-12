"""Rebuild blend-5's out-of-fold head predictions from committed member OOF.

Why this file exists: `experiments/exp-012-blend2/blend.py::regime_frame` reads
`artifacts/exp-008/frames/v2-{month}.parquet`, which is git-ignored and absent
on the laptop. The five regime columns it supplies
(`noac`, `nm_missing`, `d_aobt_lobt`, `d_takeoff_aobt`, `airport`) are all
carried inside the committed `pinned/C2-cat--C2-d8p-{fold}.parquet`, so the
rotation runs with no competition data at all. Everything else -- the guard, the
floor, the per-regime NNLS, the hybrid nm_late split, the fold rotation -- is
`prc.blend` and `experiments/exp-012-blend2/nmtail_regime.py` unchanged.

Verification target (`blend-config-5.json`, "rotated"): july_loo 286.3274,
december 207.0083, january 339.0173, board(all) 310.7662; clean board 259.132
(`exp-012 RESULTS.md`, "blend-5 shipped, hybrid").
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prc.blend as B          # noqa: E402
import prc.data as D           # noqa: E402

FOLDS = ("july_loo", "december", "january")
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}

E12 = "experiments/exp-012-blend2"
#: blend-5's member set, in the order `blend-config-5.json` declares.
#: `N1-nmtail` MUST stay last: the hybrid rule gives it weight only inside
#: `nm_late` and slices the design matrix as `[:, :k-1]` everywhere else.
CORE = {
    "T-tailv2":  "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet",
    "C2-cat":    f"{E12}/pinned/C2-cat--C2-d8p-{{fold}}.parquet",
    "L2-hub":    f"{E12}/published/L2-hub8725-{{fold}}.parquet",
    "D1-winter": f"{E12}/published/D1-winter-deice-v1-{{fold}}.parquet",
    "F1-fresh":  f"{E12}/published/F1-fresh-s1-{{fold}}.parquet",
    "V3-allin":  f"{E12}/published/V3-allin-v1-{{fold}}.parquet",
}
NMTAIL = f"{E12}/published/N1-nmtail-v4-{{fold}}.parquet"
#: The regime/label columns come from one declared file, not from whichever
#: member sorts first, so a JSON key order can never change the regime split.
GATES = f"{E12}/pinned/C2-cat--C2-d8p-{{fold}}.parquet"
GATE_COLS = ["MVT_ID_mvt", "noac", "nm_missing", "d_aobt_lobt", "d_takeoff_aobt", "airport"]


def _one(path: str, fold: str, name: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / path.format(fold=fold))
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    return d[["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": name})


def fold_data(fold: str) -> dict:
    lo, hi = json.loads((ROOT / "experiments/exp-008-diversity/guard_bounds.json").read_text())[fold]
    g = pd.read_parquet(ROOT / GATES.format(fold=fold))
    g = g[g["fold"] == fold] if "fold" in g.columns else g
    m = g[GATE_COLS + ["y_true"]].copy()
    for n, p in CORE.items():
        m = m.merge(_one(p, fold, n), on="MVT_ID_mvt", how="inner")
    h = pd.read_parquet(ROOT / NMTAIL.format(fold=fold))
    h = h[h["fold"] == fold][["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": "N1-nmtail"})
    m = m.merge(h, on="MVT_ID_mvt", how="left")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    prep = lambda v: B.apply_residual_guard(
        np.maximum(np.asarray(v, "float64"), B.FLOOR), a, float(lo), float(hi), B.FLOOR)
    hv = m["N1-nmtail"].to_numpy("float64")
    ok = ~np.isnan(hv)
    P = np.stack([prep(m[n].to_numpy("float64")) for n in CORE]
                 + [prep(np.where(ok, hv, 0.0))], axis=1)
    return {"P": P, "y": m["y_true"].to_numpy("float64"), "ok": ok,
            "regime": B.regimes_from_features(m), "frame": m,
            "names": list(CORE) + ["N1-nmtail"]}


def rotate(mode: str = "nnls", outer: str = "nnls", target: str = "nm_late") -> dict:
    """Hybrid blend-5: k members inside `nm_late`, k-1 outside, weights from the
    other two folds only. Returns per-fold predictions aligned with `frame`."""
    data = {f: fold_data(f) for f in FOLDS}
    k = len(data[FOLDS[0]]["names"])
    out = {}
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        selg = {g: (data[g]["regime"] == target) & data[g]["ok"] for g in src}
        P5 = np.concatenate([data[g]["P"][selg[g]] for g in src])
        y5 = np.concatenate([data[g]["y"][selg[g]] for g in src])
        P4 = np.concatenate([data[g]["P"][~selg[g]][:, :k - 1] for g in src])
        y4 = np.concatenate([data[g]["y"][~selg[g]] for g in src])
        w5 = B.fit_weights(P5, y5, mode)
        w4 = B.fit_weights(P4, y4, outer)
        d = data[f]
        sel = (d["regime"] == target) & d["ok"]
        p = np.empty(len(d["y"]))
        p[sel] = d["P"][sel] @ w5
        p[~sel] = d["P"][~sel][:, :k - 1] @ w4
        p = np.maximum(p, B.FLOOR)
        out[f] = {"y_pred": p, "w_in": w5, "w_out": w4, **d}
    return out


def board(per_fold: dict) -> float:
    return float(np.sqrt(sum(SHARES[f] * per_fold[f] ** 2 for f in SHARES)))


def main():
    r = rotate()
    allr, cln = {}, {}
    for f in FOLDS:
        d = r[f]
        ids = d["frame"]["MVT_ID_mvt"].to_numpy()
        b = D.rmse_both(d["y"], d["y_pred"], ids, f)
        allr[f], cln[f] = b["rmse"], b["rmse_clean"]
        print(f"  {f:9s} n={len(d['y']):,}  rmse {b['rmse']:9.4f}  clean {b['rmse_clean']:9.4f}"
              f"  excluded {b['n_excluded']}")
        out = pd.DataFrame({"MVT_ID_mvt": ids, "fold": f, "y_true": d["y"],
                            "y_pred": d["y_pred"], "regime": d["regime"],
                            "airport": d["frame"]["airport"].to_numpy(),
                            "noac": d["frame"]["noac"].to_numpy(),
                            "nm_missing": d["frame"]["nm_missing"].to_numpy(),
                            "d_aobt_lobt": d["frame"]["d_aobt_lobt"].to_numpy(),
                            "d_takeoff_aobt": d["frame"]["d_takeoff_aobt"].to_numpy()})
        out.to_parquet(ROOT / f"experiments/exp-027-step/oof/blend5-{f}.parquet", index=False)
    print(f"  board all {board(allr):.4f}   board clean {board(cln):.4f}")
    print("  target    board all 310.7662  (blend-config-5.json 'rotated'/'board_estimate')")
    print("  target    board clean 259.132 (exp-012 RESULTS.md)")


if __name__ == "__main__":
    main()
