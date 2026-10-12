"""Why the mechanism is smaller than exp-014 priced it: the base moved.

exp-014 measured the under-prediction past the envelope on **candidate 1** (the
exp-012 blend-1 rotated blend, `members-cand1.json`).  The artifact this
correction would ship on is **stack-5** (`experiments/exp-026-rebuild`), two
member generations later.  This script puts every base on the same five days.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import envelope as E  # noqa: E402
import frames as F26  # noqa: E402
import prc.blend as B  # noqa: E402

#: `members-cand1.json`: the three members with non-zero NNLS weight, rotated
#: with plain global NNLS -- `exp-014/code/q2_days.py`'s `BL.rotate(..., "nnls")`.
#: exp-012's own `blend.py` cannot run here (it reads the lost
#: `artifacts/exp-008/frames/`), so the members come from exp-026's rebuilt
#: frames, which reproduce the published rotations to 1e-4.  The gate on the
#: rebuild is exp-014's own per-day bias table, checked in `main`.
CAND1 = ("T-tailv2", "C2-cat", "L2-hub")
B5 = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "V3-allin", "N1-nmtail")


def candidate1() -> dict:
    """Candidate 1's rotated blend, exp-014's own base, per fold."""
    data = {f: F26.fold_frame(f, CAND1) for f in E.FOLDS}
    out = {}
    for f in E.FOLDS:
        fit = pd.concat([data[g] for g in E.FOLDS if g != f], ignore_index=True)
        P = np.stack([fit[n].to_numpy("float64") for n in CAND1], axis=1)
        w = B.fit_weights(P, fit["y_true"].to_numpy("float64"), "nnls")
        Q = np.stack([data[f][n].to_numpy("float64") for n in CAND1], axis=1)
        out[f] = pd.DataFrame({"MVT_ID_mvt": data[f]["MVT_ID_mvt"].to_numpy("float64"),
                               "cand1": np.maximum(Q @ w, B.FLOOR)})
        print(f"[cand1] {f:9s} weights " +
              " ".join(f"{n}={v:.4f}" for n, v in zip(CAND1, w)), flush=True)
    return out


def gate_cand1(R: pd.DataFrame, c1: dict) -> None:
    """exp-014's own per-day bias table is the gate on the rebuild."""
    ref = pd.read_csv(ROOT / "experiments/exp-014-robustness/analysis/q2b_fold_days.csv",
                      parse_dates=["day"])
    ref = ref[ref["over_tr_max"] > 0]
    rows = []
    for f in E.FOLDS:
        g = R[R["fold"] == f].merge(c1[f], on="MVT_ID_mvt", how="left", validate="one_to_one")
        g = g[g["flagged"]]
        for (a, d), s in g.groupby(["apt", "day"]):
            e = s["cand1"].to_numpy("float64") - s["y_true"].to_numpy("float64")
            rows.append(dict(fold=f, apt=a, day=d, n=len(s), bias=float(e.mean()),
                             rmse=float(np.sqrt((e ** 2).mean()))))
    got = pd.DataFrame(rows).merge(ref[["fold", "apt", "day", "n", "bias", "rmse"]],
                                   on=["fold", "apt", "day"], suffixes=("", "_ref"))
    got["d_bias"] = got["bias"] - got["bias_ref"]
    got["d_rmse"] = got["rmse"] - got["rmse_ref"]
    print("\n=== gate: rebuilt candidate 1 against exp-014's published per-day table ===")
    print(got[["fold", "apt", "day", "n", "n_ref", "bias", "bias_ref", "d_bias",
               "rmse", "rmse_ref", "d_rmse"]].round(2).to_string(index=False), flush=True)
    if len(got) != 5:
        raise SystemExit(f"gate matched {len(got)} of 5 days")


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "fold_rows.parquet")
    c1 = candidate1()
    gate_cand1(R, c1)
    rec = []
    for f in E.FOLDS:
        g = R[R["fold"] == f].copy()
        cols = {"candidate 1 (exp-014's base)": None,
                "blend-5 (stack-5's anchor)": g["y_pred_anchor"].to_numpy("float64"),
                "stack-5 keep-noac": g["y_pred"].to_numpy("float64")}
        m = g.merge(c1[f], on="MVT_ID_mvt", how="left", validate="one_to_one")
        cols["candidate 1 (exp-014's base)"] = m["cand1"].to_numpy("float64")
        # every member of blend-5, on its own
        fr = F26.fold_frame(f, B5)[["MVT_ID_mvt"] + list(B5)]
        m2 = g.merge(fr, on="MVT_ID_mvt", how="left", validate="one_to_one")
        for n in B5:
            cols[f"member {n}"] = m2[n].to_numpy("float64")
        y = g["y_true"].to_numpy("float64")
        fl = g["flagged"].to_numpy()
        big = (g["exceedance"] > E.CLEAR).to_numpy()
        for lab, p in cols.items():
            ok = np.isfinite(p)
            for sl, mask in (("flagged", fl & ok), (">30 s", big & ok),
                             ("whole fold", ok)):
                if mask.sum() < 20:
                    continue
                e = p[mask] - y[mask]
                rec.append(dict(fold=f, base=lab, slice=sl, n=int(mask.sum()),
                                rmse=float(np.sqrt((e ** 2).mean())),
                                bias=float(e.mean())))
    T = pd.DataFrame(rec)
    T.to_csv(EXP / "analysis" / "base_moved.csv", index=False)
    piv = (T[T["slice"] != "whole fold"]
           .pivot_table(index="base", columns=["slice", "fold"], values="bias"))
    # pooled across folds, weighted by rows
    pool = []
    for lab, g in T[T["slice"] != "whole fold"].groupby(["base", "slice"]):
        pool.append(dict(base=lab[0], slice=lab[1], n=int(g["n"].sum()),
                         bias=float((g["bias"] * g["n"]).sum() / g["n"].sum())))
    P = pd.DataFrame(pool).pivot_table(index="base", columns="slice",
                                       values=["n", "bias"])
    print("=== bias (pred - truth) on the five envelope days, pooled over the three folds ===")
    print(P.round(1).to_string())
    print("\n=== per fold ===")
    print(piv.round(1).to_string())
    P.to_csv(EXP / "analysis" / "base_moved_pooled.csv")


if __name__ == "__main__":
    main()
