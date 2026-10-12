"""Rotated per-regime blend evaluation over exp-008 members.

Weights for a held-out fold are fitted on the OOF of the *other validation
months* and never on the fold being scored.  The rotation has one trap worth
naming: `july` and `july_loo` are the same validation month (2025-07) predicted
by two different training sets, so treating them as two independent folds would
fit the weights for one of them on the very rows it is scored on.  The fit set
is therefore built per validation *month*, each month represented once by its
ranking-analogue fold (`july_loo`, `december`, `january`), and the scored fold's
own month is always dropped.

  python blend_eval.py members.json --mode nnls
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = ("july", "july_loo", "december", "january")
#: One fold per validation month, used as the source of fitting rows.
MONTH_SOURCE = {"2025-07": "july_loo", "2025-12": "december", "2025-01": "january"}
TAIL_CUT = 3600.0


def load(files: dict[str, str], folds=FOLDS) -> dict[str, pd.DataFrame]:
    out = {}
    for f in folds:
        frame, names = B.stack_members(files, ROOT, fold=f)
        frame["regime"] = B.regimes_from_features(frame)
        frame["fold"] = f
        out[f] = frame
    return out, names


def decompose(y, p, frame) -> dict:
    y, p = np.asarray(y, "float64"), np.asarray(p, "float64")
    se = (y - p) ** 2
    bulk = y <= TAIL_CUT
    nonm = frame["nm_missing"].to_numpy().astype(bool)
    noac = frame["noac"].to_numpy().astype(bool)
    r = {"rmse": float(np.sqrt(se.mean())),
         "bulk": float(np.sqrt(se[bulk].mean())),
         "tail": float(np.sqrt(se[~bulk].mean())) if (~bulk).any() else float("nan"),
         "no_nm": float(np.sqrt(se[nonm].mean())) if nonm.any() else float("nan"),
         "noac": float(np.sqrt(se[noac].mean())) if noac.any() else float("nan")}
    for g in B.REGIMES:
        m = frame["regime"].to_numpy() == g
        r[f"reg_{g}"] = float(np.sqrt(se[m].mean())) if m.any() else float("nan")
    return r


def corr_matrix(frame: pd.DataFrame, names: list[str], subset=None) -> pd.DataFrame:
    d = frame if subset is None else frame[subset]
    R = np.stack([d["y_true"].to_numpy("float64") - d[n].to_numpy("float64") for n in names])
    return pd.DataFrame(np.corrcoef(R), index=names, columns=names)


#: Candidate shrinkage pseudo-counts for `--shrink auto`.
SHRINK_GRID = (0.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def _score(fit_frame, score_frame, names, mode, min_rows, n0):
    P = fit_frame[names].to_numpy("float64")
    W = B.fit_per_regime(P, fit_frame["y_true"].to_numpy("float64"),
                         fit_frame["regime"].to_numpy(), mode, min_rows=min_rows, shrink_rows=n0)
    Pv = score_frame[names].to_numpy("float64")
    p = np.maximum(B.apply_weights(Pv, score_frame["regime"].to_numpy(), W), B.FLOOR)
    y = score_frame["y_true"].to_numpy("float64")
    return float(np.sqrt(((y - p) ** 2).mean()))


def choose_shrink(sources, data, names, mode, min_rows, grid=SHRINK_GRID):
    """Pick the shrinkage pseudo-count inside the fitting months only.

    Choosing `n0` by looking at the rotated fold scores would be a second-level
    fit to the very folds being scored. Instead, for a scored fold whose fitting
    set is two other months, each of those two months takes a turn as the inner
    validation month; the `n0` with the lower mean inner RMSE is then used to
    refit on both. The scored fold never enters the choice.
    """
    best, best_rmse = grid[0], float("inf")
    for n0 in grid:
        rs = [_score(data[a], data[b], names, mode, min_rows, n0)
              for a in sources for b in sources if a != b]
        m = float(np.mean(rs)) if rs else float("inf")
        if m < best_rmse - 1e-9:
            best, best_rmse = n0, m
    return best, best_rmse


def rotate(data, names, mode="nnls", min_rows=200, per_regime=True, fit_source=MONTH_SOURCE,
           shrink_rows=0.0):
    rows, wrec = [], {}
    for f in FOLDS:
        vm = D.FOLDS[f].valid_month
        src = [g for m, g in fit_source.items() if m != vm]
        if shrink_rows == "auto":
            shrink_rows_f, inner = choose_shrink(src, data, names, mode, min_rows)
            print(f"  [{f}] inner rotation over {'+'.join(src)} chose n0={shrink_rows_f:.0f} "
                  f"(inner rmse {inner:.1f})", flush=True)
        else:
            shrink_rows_f = shrink_rows
        fit = pd.concat([data[g] for g in src], ignore_index=True)
        P = fit[names].to_numpy("float64")
        y = fit["y_true"].to_numpy("float64")
        W = (B.fit_per_regime(P, y, fit["regime"].to_numpy(), mode, min_rows=min_rows,
                              shrink_rows=shrink_rows_f)
             if per_regime else {"_all": B.fit_weights(P, y, mode)})
        d = data[f]
        Pv = d[names].to_numpy("float64")
        pred = (B.apply_weights(Pv, d["regime"].to_numpy(), W) if per_regime else Pv @ W["_all"])
        pred = np.maximum(pred, B.FLOOR)
        r = decompose(d["y_true"], pred, d)
        r.update(fold=f, fit_on="+".join(src), mode=mode, per_regime=per_regime,
                 shrink_rows=shrink_rows_f)
        rows.append(r)
        wrec[f] = {k: dict(zip(names, map(float, v))) for k, v in W.items()}
    return pd.DataFrame(rows), wrec


def singles(data, names):
    rows = []
    for f in FOLDS:
        d = data[f]
        for n in names:
            r = decompose(d["y_true"], np.maximum(d[n].to_numpy("float64"), B.FLOOR), d)
            r.update(fold=f, member=n)
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", default="nnls", choices=["nnls", "ols", "flat"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--members", default=None, help="comma-separated subset")
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--shrink", default="0", help="comma-separated shrink_rows values to sweep")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    files = cfg["members"]
    if a.members:
        files = {k: files[k] for k in a.members.split(",")}
    data, names = load(files)

    print("=== single members ===")
    S = singles(data, names)
    print(S.pivot(index="member", columns="fold", values="rmse")[list(FOLDS)].round(1).to_string())
    print("\n=== residual correlation (july) ===")
    print(corr_matrix(data["july"], names).round(3).to_string())

    out = {"members": names, "singles": S.to_dict("records"), "corr": {}}
    for f in FOLDS:
        out["corr"][f] = corr_matrix(data[f], names).round(4).to_dict()
    shrinks = [x if x == "auto" else float(x) for x in a.shrink.split(",")]
    for mode in (["nnls", "ols", "flat"] if a.mode == "nnls" else [a.mode]):
        for pr, sh in [(True, s) for s in shrinks] + [(False, 0.0)]:
            T, W = rotate(data, names, mode=mode, per_regime=pr, min_rows=a.min_rows, shrink_rows=sh)
            tag = (f"{mode}-perregime-n0={sh if sh == 'auto' else int(sh)}" if pr
                   else f"{mode}-global")
            print(f"\n=== blend {tag} ===")
            print(T[["fold", "rmse", "bulk", "tail", "no_nm", "noac"]].round(1).to_string(index=False))
            out.setdefault("blends", {})[tag] = {"table": T.to_dict("records"), "weights": W}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
