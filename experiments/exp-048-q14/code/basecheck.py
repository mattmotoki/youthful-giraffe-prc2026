"""Does the verdict depend on how the frozen base was reproduced?

`code/base.py` reproduces the shipped v4 noac path but makes two choices the
served path does not force on a noac-only harness: the v4 band override is
included (exp-033 section 6a's harness omits it), and the categorical level sets
are the union of the 2025 and 2026 noac frames rather than the training frame's
own levels.  Neither can change the COMPARISON between the two objectives -- both
arms sit on the same base -- but either could move the level of the gate.

So the whole pipeline is rerun on two other bases and the gate re-read:

  B1  the primary base (band override, union levels)          -- base.py
  B2  no band override                                        -- exp-033 6a's harness
  B3  levels from the eleven training months' noac rows only  -- no 2026 strings

Nothing is re-registered: the basis, the fit months, the reading months, the
capacity rule and the gate are SPEC.md's.  This is a robustness read on a
verdict already taken, not a second attempt at it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "code"))
from prc.tail_model import NOAC_CATS_V2, MIN_TAXI  # noqa: E402
from base import MONTHS, fit_path, apply_path, band_predict, encode, SRC  # noqa: E402
from arms import (FIT_MONTHS, READ_MONTHS, GRID, basis, solve, simplex_grid,
                  worst_alpha, bisect)  # noqa: E402
from read import endpoint, boot  # noqa: E402


def lomo(a, b, band: bool, union: bool):
    out = pd.Series(np.nan, index=a.index, dtype=float)
    cats_u = {c: pd.Index(sorted(set(a[c].astype(str)) | set(b[c].astype(str))))
              for c in NOAC_CATS_V2}
    for m in MONTHS:
        va = (a.month == m).to_numpy()
        fit = a[~va]
        cats = cats_u if union else {
            c: pd.Index(sorted(set(fit[c].astype(str)) | set(a.loc[va, c].astype(str))))
            for c in NOAC_CATS_V2}
        path = fit_path(fit, cats)
        if band:
            out[va] = apply_path(path, a[va], cats)
        else:
            clf, reg, _ = path
            X = encode(a[va], cats)
            ms = a.loc[va, "mvt_sched"].to_numpy(float)
            p = np.mean([c.predict_proba(X)[:, 1] for c in clf], axis=0)
            off = np.mean([r.predict(X) for r in reg], axis=0)
            out[va] = np.clip(ms - (1.0 - p) * off, MIN_TAXI, None)
    return out.to_numpy()


def gate(a, base, X388, v7, mu, sd):
    a = a.assign(base=base)
    fit = a[a.month.isin(FIT_MONTHS)]
    Xf = basis(fit, mu, sd)
    rf = (fit.y_true - fit.base).to_numpy(float)
    Gs, cs, ns = [], [], []
    for m in FIT_MONTHS:
        s = (fit.month == m).to_numpy()
        Gs.append(Xf[s].T @ Xf[s] / s.sum())
        cs.append(Xf[s].T @ rf[s] / s.sum())
        ns.append(int(s.sum()))
    ap = np.array([n / sum(ns) for n in ns])
    Gp = sum(ap[i] * Gs[i] for i in range(3))
    cp = sum(ap[i] * cs[i] for i in range(3))
    al = simplex_grid(GRID)
    ws = {}
    _, ws["P"], _ = bisect(lambda l: solve(Gp, cp, l), X388, v7)
    lam, ws["W"], _ = bisect(lambda l: worst_alpha(Gs, cs, l, al)[1], X388, v7)
    alpha_w = np.asarray(worst_alpha(Gs, cs, lam, al)[0]).tolist()

    rd = a[a.month.isin(READ_MONTHS)]
    X9 = basis(rd, mu, sd)
    b9 = rd.base.to_numpy(float)
    r9 = (rd.y_true - b9).astype(float).to_numpy()
    ds = {k: np.clip(b9 + X9 @ ws[k], MIN_TAXI, None) - b9 for k in ws}
    days = rd.day.to_numpy().astype("datetime64[D]").astype("int64")
    bs = boot(r9, ds, days)
    out = {"rmse_base_nine": float(np.sqrt(np.mean(r9 ** 2))),
           "rmse_base_all": float(np.sqrt(np.mean((a.y_true - a.base) ** 2))),
           "alpha_W": alpha_w}
    for k in ws:
        e = endpoint(r9, ds[k])
        ef = endpoint(rf, np.clip(fit.base.to_numpy(float) + Xf @ ws[k], MIN_TAXI, None)
                      - fit.base.to_numpy(float))
        out[k] = {"gain_nine": e["gain_rmse"], "lo": bs[k]["lo"], "hi": bs[k]["hi"],
                  "rho_nine": e["rho"], "scale_nine": e["scale"],
                  "gain_fit_in_sample": ef["gain_rmse"]}
    pw = bs["P-W"]
    out["W-P"] = {"mean": -pw["mean"], "lo": -pw["hi"], "hi": -pw["lo"],
                  "p_pos": 1.0 - pw["p_pos"]}
    return out


def main() -> None:
    a = pd.read_parquet(SRC / "noac_2025_lab.parquet")
    b = pd.read_parquet(SRC / "noac_2026_d.parquet")
    cfg = json.loads((EXP / "analysis" / "arms.json").read_text())
    mu, sd = cfg["mu"], cfg["sd"]
    X388, v7 = basis(b, mu, sd), b.v7.to_numpy(float)

    res = {}
    prim = pd.read_parquet(EXP / "work" / "base_lomo.parquet")
    a = a.merge(prim[["MVT_ID_mvt", "base"]], on="MVT_ID_mvt", how="left")
    assert a.base.notna().all()
    res["B1 band override, union levels (primary)"] = gate(
        a.drop(columns="base"), a.base.to_numpy(), X388, v7, mu, sd)
    for tag, band, union in (("B2 no band override", False, True),
                             ("B3 training-month levels only", True, False)):
        base = lomo(a.drop(columns="base"), b, band, union)
        res[tag] = gate(a.drop(columns="base"), base, X388, v7, mu, sd)
        print(f"  {tag} done", flush=True)

    (EXP / "analysis" / "basecheck.json").write_text(json.dumps(res, indent=2, default=float))
    rows = []
    for tag, r in res.items():
        for k in ("P", "W"):
            rows.append({"base": tag, "arm": k, "rmse_base_nine": r["rmse_base_nine"],
                         **{x: r[k][x] for x in ("gain_nine", "lo", "hi", "rho_nine",
                                                 "scale_nine", "gain_fit_in_sample")}})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "basecheck.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== the gate under three reproductions of the frozen base ===")
    print(t.round(3).to_string(index=False))
    for tag, r in res.items():
        print(f"{tag}: alpha_W {[round(x, 3) for x in r['alpha_W']]}, "
              f"W-P {r['W-P']['mean']:+.2f} [{r['W-P']['lo']:+.2f}, {r['W-P']['hi']:+.2f}]")


if __name__ == "__main__":
    main()
