"""The repair QUEUE.md names, and the fit-side challenge that diagnoses it.

The primary reading (`read.py`) fails the gate for both objectives, so a FAIL
needs a repair attempt first.  QUEUE.md's Q14 failure clause names two:
"shrink month estimates OR interpolate toward pooled loss in a fixed repair".
SPEC.md registered the first with a trigger (`alpha*` mass > 0.6 on a fit month
under 100 rows) which did NOT fire -- `alpha* = (0.04, 0.475, 0.485)`.  Both are
run here anyway, unconditionally, because the protocol wants the repair tried
before the FAIL is written.

**These arms are post-hoc.**  Their nine-month numbers are diagnostic only: once
the nine-month reading has been seen, an arm chosen after it is no longer
"evidence that did not take part in the selection" and cannot pass the gate.
They are reported to answer *why*, not to advance.

Also here: the fit-side challenge.  Refit each objective with one of the three
fit months left out, and read the same nine months.  If the correction is one
month's noise, this moves it a lot.
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
from prc.tail_model import MIN_TAXI  # noqa: E402
from arms import (FIT_MONTHS, READ_MONTHS, CAPACITY, GRID, M0, COLS, basis,
                  solve, simplex_grid, worst_alpha, bisect, move_388)  # noqa: E402
from read import endpoint, boot  # noqa: E402

SRC = ROOT / "experiments" / "exp-033-noac" / "work"


def moments(fit: pd.DataFrame, Xf: np.ndarray, r: np.ndarray, months):
    Gs, cs, ns = [], [], []
    for m in months:
        s = (fit.month == m).to_numpy()
        Xm, rm = Xf[s], r[s]
        Gs.append(Xm.T @ Xm / len(rm))
        cs.append(Xm.T @ rm / len(rm))
        ns.append(int(s.sum()))
    return Gs, cs, ns


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "base_lomo.parquet")
    b = pd.read_parquet(SRC / "noac_2026_d.parquet")
    cfg = json.loads((EXP / "analysis" / "arms.json").read_text())
    mu, sd = cfg["mu"], cfg["sd"]
    X388, v7 = basis(b, mu, sd), b.v7.to_numpy(float)
    Xa = basis(a, mu, sd)

    fit = a[a.month.isin(FIT_MONTHS)].reset_index(drop=True)
    Xf = basis(fit, mu, sd)
    r_fit = (fit.y_true - fit.base).to_numpy(float)
    Gs, cs, ns = moments(fit, Xf, r_fit, FIT_MONTHS)
    N = sum(ns)
    a_pool = np.array([n / N for n in ns])
    Gp = sum(a_pool[i] * Gs[i] for i in range(3))
    cp = sum(a_pool[i] * cs[i] for i in range(3))
    alphas = simplex_grid(GRID)

    ws, meta = {}, {}

    # repair 1: shrink the month estimates toward the pooled loss (m0 = 30)
    Gt = [(ns[i] * Gs[i] + M0 * Gp) / (ns[i] + M0) for i in range(3)]
    ct = [(ns[i] * cs[i] + M0 * cp) / (ns[i] + M0) for i in range(3)]
    lam, w, st = bisect(lambda l: worst_alpha(Gt, ct, l, alphas)[1], X388, v7)
    ws["WR"] = w
    meta["WR"] = {"what": "worst month, month losses shrunk toward pooled, m0=30",
                  "lambda": lam, "bisect": st,
                  "alpha": np.asarray(worst_alpha(Gt, ct, lam, alphas)[0]).tolist()}

    # repair 2: interpolate the minimax weights half way toward pooled
    al_w = np.asarray(cfg["arms"]["W"]["alpha"], float)
    al_i = 0.5 * al_w + 0.5 * a_pool
    Gi = sum(al_i[i] * Gs[i] for i in range(3))
    ci = sum(al_i[i] * cs[i] for i in range(3))
    lam, w, st = bisect(lambda l: solve(Gi, ci, l), X388, v7)
    ws["WI"] = w
    meta["WI"] = {"what": "alpha = 0.5*alpha_minimax + 0.5*alpha_pooled",
                  "lambda": lam, "bisect": st, "alpha": al_i.tolist()}

    # the fit-side challenge: each objective refitted without one fit month
    for m in FIT_MONTHS:
        keep = [x for x in FIT_MONTHS if x != m]
        f2 = fit[fit.month.isin(keep)].reset_index(drop=True)
        X2 = basis(f2, mu, sd)
        r2 = (f2.y_true - f2.base).to_numpy(float)
        G2, c2, n2 = moments(f2, X2, r2, keep)
        ap2 = np.array([n / sum(n2) for n in n2])
        Gp2 = sum(ap2[i] * G2[i] for i in range(2))
        cp2 = sum(ap2[i] * c2[i] for i in range(2))
        al2 = simplex_grid(GRID)[:, :2]
        al2 = np.unique(np.column_stack([al2[:, 0], 1 - al2[:, 0]]), axis=0)
        lam, w, st = bisect(lambda l: solve(Gp2, cp2, l), X388, v7)
        ws[f"P_no{m[-2:]}"] = w
        meta[f"P_no{m[-2:]}"] = {"what": f"pooled, fit months minus {m}",
                                 "lambda": lam, "bisect": st}
        lam, w, st = bisect(lambda l: worst_alpha(G2, c2, l, al2)[1], X388, v7)
        ws[f"W_no{m[-2:]}"] = w
        meta[f"W_no{m[-2:]}"] = {"what": f"worst month, fit months minus {m}",
                                 "lambda": lam, "bisect": st}

    base = a.base.to_numpy(float)
    out = a[["MVT_ID_mvt", "month", "day", "band", "y_true", "base"]].copy()
    for k, w in ws.items():
        out[f"d_{k}"] = np.clip(base + Xa @ w, MIN_TAXI, None) - base
    out.to_parquet(EXP / "work" / "repair_2025.parquet", index=False)

    rd = out[out.month.isin(READ_MONTHS)].reset_index(drop=True)
    ft = out[out.month.isin(FIT_MONTHS)].reset_index(drop=True)
    r = (rd.y_true - rd.base).to_numpy(float)
    ds = {k: rd[f"d_{k}"].to_numpy(float) for k in ws}
    days = rd.day.to_numpy().astype("datetime64[D]").astype("int64")
    bs = boot(r, ds, days)

    rows = []
    for k in ws:
        e = endpoint(r, ds[k])
        ei = endpoint((ft.y_true - ft.base).to_numpy(float), ft[f"d_{k}"].to_numpy(float))
        rows.append({"arm": k, "what": meta[k]["what"], "n": e["n"],
                     "gain_rmse_nine": e["gain_rmse"], "lo": bs[k]["lo"], "hi": bs[k]["hi"],
                     "p_pos": bs[k]["p_pos"], "rho_nine": e["rho"], "rms_d_nine": e["rms_d"],
                     "gain_rmse_fit_insample": ei["gain_rmse"], "rho_fit": ei["rho"],
                     "move_388_rms": move_388(ws[k], X388, v7),
                     "lambda": meta[k]["lambda"], "bisect": meta[k]["bisect"]})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "repair.csv", index=False)
    (EXP / "analysis" / "repair.json").write_text(json.dumps(
        {"arms": {k: {**meta[k], "w": ws[k].tolist()} for k in ws},
         "cols": list(COLS), "table": t.to_dict("records"),
         "note": "post-hoc: read after the primary nine-month reading, diagnostic only"},
        indent=2, default=float))

    pd.set_option("display.width", 220)
    print("=== repairs and the fit-side challenge, read on the same nine months ===")
    print("(post-hoc: diagnostic only, not gate-eligible)")
    print(t[["arm", "what", "gain_rmse_nine", "lo", "hi", "p_pos", "rho_nine",
             "gain_rmse_fit_insample", "rho_fit", "move_388_rms"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
