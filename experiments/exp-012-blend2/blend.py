"""The standing blend: rotated NNLS over three board-relevant folds.

Three folds, three distinct validation months, each used once: a fold's weights
are fitted on the other two and applied unseen. `--stability` prints the three
rotated weight vectors beside the all-month vector that ships, which is the
month-to-month stability check the desk asked for.

This file puts ITS OWN directory first on sys.path. exp-001 and exp-008 both
have a `blend_eval.py`, and importing across experiments without doing this
silently loads the wrong module (exp-008 lost an hour to it).
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
for p in (str(ROOT), str(HERE)):          # HERE last => HERE first
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
#: Ranking-month shares of the 344,841 scored departures (exp-001 board.py).
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
TAIL_CUT = 3600.0


def board_estimate(per_fold: dict) -> float:
    """RMSE combines in QUADRATURE, weighted by each ranking month's share."""
    return float(np.sqrt(sum(SHARES[f] * per_fold[f] ** 2 for f in SHARES)))


#: The frozen v2 month cache exp-008 built; the source of the regime columns.
FRAME_CACHE = ROOT / "artifacts" / "exp-008" / "frames"
REGIME_COLS = ["MVT_ID_mvt", "noac", "nm_missing", "d_aobt_lobt", "d_takeoff_aobt", "airport"]


def regime_frame(fold: str) -> pd.DataFrame:
    """Regime columns for a fold's validation month, from the frozen v2 cache.

    Taken from the cache rather than from whichever member happens to be listed
    first: exp-001's OOF files carry no `noac` or `d_aobt_lobt`, so ordering the
    member dict wrongly would break the regime split, and a blend must not
    depend on the order of a JSON object.
    """
    month = D.FOLDS[fold].valid_month
    return pd.read_parquet(FRAME_CACHE / f"v2-{month}.parquet", columns=REGIME_COLS)


def load(files: dict, folds=FOLDS):
    """Members joined per fold, with regime columns attached from the cache."""
    out, names = {}, list(files)
    for f in folds:
        frame, _ = B.stack_members(files, ROOT, fold=f)
        keep = [c for c in frame.columns if c not in REGIME_COLS or c == "MVT_ID_mvt"]
        frame = frame[keep].merge(regime_frame(f), on="MVT_ID_mvt", how="inner")
        if len(frame) == 0:
            raise SystemExit(f"{f}: no rows survive the member join")
        frame["regime"] = B.regimes_from_features(frame)
        out[f] = frame
    return out, names


def decompose(y, p, frame) -> dict:
    y, p = np.asarray(y, "float64"), np.asarray(p, "float64")
    se = (y - p) ** 2
    bulk = y <= TAIL_CUT
    nonm = frame["nm_missing"].to_numpy().astype(bool)
    noac = frame["noac"].to_numpy().astype(bool)
    return {"rmse": float(np.sqrt(se.mean())),
            "bulk": float(np.sqrt(se[bulk].mean())),
            "tail": float(np.sqrt(se[~bulk].mean())) if (~bulk).any() else float("nan"),
            "no_nm": float(np.sqrt(se[nonm].mean())) if nonm.any() else float("nan"),
            "noac": float(np.sqrt(se[noac].mean())) if noac.any() else float("nan")}


def fit_all(data, names, mode="nnls", folds=FOLDS):
    fit = pd.concat([data[f] for f in folds], ignore_index=True)
    w = B.fit_weights(fit[names].to_numpy("float64"), fit["y_true"].to_numpy("float64"), mode)
    return w, len(fit)


def rotate(data, names, mode="nnls"):
    """A fold's weights come only from the OTHER validation months."""
    rows, wrec = [], {}
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        fit = pd.concat([data[g] for g in src], ignore_index=True)
        w = B.fit_weights(fit[names].to_numpy("float64"), fit["y_true"].to_numpy("float64"), mode)
        d = data[f]
        p = np.maximum(d[names].to_numpy("float64") @ w, B.FLOOR)
        r = decompose(d["y_true"], p, d)
        r.update(fold=f, fit_on="+".join(src))
        rows.append(r)
        wrec[f] = dict(zip(names, map(float, w)))
    t = pd.DataFrame(rows)
    return t, wrec


def singles(data, names):
    rows = []
    for f in FOLDS:
        d = data[f]
        for n in names:
            r = decompose(d["y_true"], np.maximum(d[n].to_numpy("float64"), B.FLOOR), d)
            r.update(fold=f, member=n)
            rows.append(r)
    return pd.DataFrame(rows)


def corr(data, names, fold):
    d = data[fold]
    R = np.stack([d["y_true"].to_numpy("float64") - np.maximum(d[n].to_numpy("float64"), B.FLOOR)
                  for n in names])
    return pd.DataFrame(np.corrcoef(R), index=names, columns=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", default="nnls", choices=["nnls", "ols", "flat"])
    ap.add_argument("--members", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stability", action="store_true")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    files = cfg["members"]
    if a.members:
        files = {k: files[k] for k in a.members.split(",")}
    data, names = load(files)

    S = singles(data, names)
    piv = S.pivot(index="member", columns="fold", values="rmse")[list(FOLDS)]
    piv["board"] = [board_estimate(dict(r)) for _, r in piv.iterrows()]
    print("=== single members ===")
    print(piv.sort_values("board").round(2).to_string())

    print("\n=== residual correlation (july_loo) ===")
    print(corr(data, names, "july_loo").round(3).to_string())

    out = {"members": names, "folds": list(FOLDS),
           "singles": piv.round(4).to_dict(), "blends": {}}
    for mode in ("nnls", "ols", "flat"):
        T, W = rotate(data, names, mode)
        be = board_estimate({r.fold: r.rmse for r in T.itertuples()})
        print(f"\n=== rotated {mode} === board estimate {be:.2f}")
        print(T[["fold", "fit_on", "rmse", "bulk", "tail", "no_nm", "noac"]].round(1).to_string(index=False))
        out["blends"][mode] = {"table": T.to_dict("records"), "rotated_weights": W,
                               "board_estimate": be}
        if a.stability and mode == "nnls":
            w_all, n = fit_all(data, names, mode)
            wt = pd.DataFrame(W).T
            wt.loc["ALL (shipped)"] = dict(zip(names, w_all))
            print(f"\n=== weight stability: rotated vectors vs the shipped one ({n} rows) ===")
            print(wt.round(3).to_string())
            print("\nper-member spread across the three rotated vectors:")
            sp = (wt.loc[list(FOLDS)].max() - wt.loc[list(FOLDS)].min()).round(3)
            print(sp.to_string())
            out["shipped_weights"] = dict(zip(names, map(float, w_all)))
            out["weight_spread"] = sp.to_dict()
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
