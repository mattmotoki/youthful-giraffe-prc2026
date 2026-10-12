"""The hybrid blend rule blend-4 and blend-5 both ship, and its rotation.

The rule, from `blend-config-5.json` ("hybrid: nnls in nm_late, nnls
elsewhere") and from exp-016's `blend4.fit_blend4`:

  inside `nm_late`, where the NM-tail specialist is defined -- NNLS over ALL
  members including it;
  everywhere else -- NNLS over the general members only, the specialist at
  exactly 0.

`fit_blend4` is that rule with the member list hard-coded; this is the same
function with the list passed in, so a ladder rung is a member tuple.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

SPECIALIST = "N1-nmtail"


def _split(m: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(member matrix, the rows the `nm_late` vector is fitted and applied on).

    The specialist is undefined off the NM join, so an `nm_late` row where it is
    null belongs with the general rows -- that is `fit_blend4`'s `& okf`.  A
    rung that drops the specialist keeps the regime split and simply has no
    partial member, so the mask is the regime alone.
    """
    P = np.stack([m[n].to_numpy("float64") for n in names], axis=1)
    sel = m["regime"].to_numpy() == "nm_late"
    if SPECIALIST in names:
        sel = sel & np.isfinite(P[:, names.index(SPECIALIST)])
    return P, sel


def fit_hybrid(fit: pd.DataFrame, names) -> dict:
    """Weights per regime in `prc.blend` order, keyed as `blend-config-*.json`."""
    names = list(names)
    P, sel = _split(fit, names)
    gen = [n for n in names if n != SPECIALIST]
    y = fit["y_true"].to_numpy("float64")
    w_late = B.fit_weights(P[sel], y[sel], "nnls")
    Pg = np.stack([fit[n].to_numpy("float64") for n in gen], axis=1)
    w_out = B.fit_weights(Pg[~sel], y[~sel], "nnls")
    late = dict(zip(names, map(float, w_late)))
    out = {n: 0.0 for n in names}
    out.update(dict(zip(gen, map(float, w_out))))
    return {"nm_late": late, "nm_bulk": out, "nonm": out, "noac": out, "_all": out}


def apply_hybrid(m: pd.DataFrame, names, W: dict) -> np.ndarray:
    names = list(names)
    P, sel = _split(m, names)
    w_late = np.array([W["nm_late"][n] for n in names], dtype="float64")
    w_out = np.array([W["_all"][n] for n in names], dtype="float64")
    Pz = np.where(np.isfinite(P), P, 0.0)   # the specialist carries 0.0 outside
    out = np.where(sel, Pz @ w_late, Pz @ w_out)
    return np.maximum(out, B.FLOOR)


def rotate(data: dict, names, extra: dict | None = None) -> dict:
    """Fold-by-fold: fit on the other two folds, score the held-out one.

    Returns `{fold: {"rmse", "rmse_clean", "n_excluded", "pred"}}` plus the
    board estimates.  `extra` maps a fold to a prediction vector to score in
    place of the blend (used by the stacker, whose anchor is this rotation).
    """
    res = {}
    for f in F.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in F.FOLDS if g != f], ignore_index=True)
        W = fit_hybrid(fit, names)
        p = apply_hybrid(ap, names, W) if extra is None else extra[f]
        y = ap["y_true"].to_numpy("float64")
        r = D.rmse_both(y, p, ap["MVT_ID_mvt"].to_numpy("float64"), f)
        res[f] = dict(r, pred=p, weights=W)
    res["board_all"] = F.board_estimate({f: res[f]["rmse"] for f in F.SHARES})
    res["board_clean"] = F.board_estimate({f: res[f]["rmse_clean"] for f in F.SHARES})
    res["predict"] = res["board_clean"] + 4.0
    return res


def decomp(y, p, ap) -> dict:
    """The decomposition COMMON.md requires: bulk, tail, no-NM, noac."""
    e = (np.asarray(p, "float64") - np.asarray(y, "float64")) ** 2
    r = lambda m: float(np.sqrt(e[m].mean())) if m.any() else float("nan")
    bulk = y <= 3600.0
    return dict(rmse=float(np.sqrt(e.mean())), le3600=r(bulk), gt3600=r(~bulk),
                no_nm=r(ap["nm_missing"].to_numpy().astype(bool)),
                noac=r(ap["noac"].to_numpy().astype(bool)))
