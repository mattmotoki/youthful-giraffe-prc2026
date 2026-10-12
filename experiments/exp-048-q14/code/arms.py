"""The two fitting objectives, at the registered fixed capacity.

SPEC.md fixes everything here: the eight-column basis, the three fit months, the
250 s movement on the 388, and the minimax repair.  `d = X w` is the correction;
`L_m(w) = mean_m(d^2 - 2 r d)` is the incremental squared loss the contract
quotes as `-(2A - B)`.  P minimises the row-weighted pooled `L`, W minimises the
worst of the three fit months' `L`, both with the same ridge penalty and both
scaled to the same movement on the 388, so the arms differ in DIRECTION only.

The fitted objective uses the linear `d = X w` (convex, closed form).  The
reading uses the served update `clip(base + X w, 60) - base`; the two differ only
where the 60 s floor binds, which is reported.
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
from prc.tail_model import MIN_TAXI  # noqa: E402

SRC = ROOT / "experiments" / "exp-033-noac" / "work"
FIT_MONTHS = ("2025-01", "2025-07", "2025-12")
READ_MONTHS = ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
               "2025-08", "2025-09", "2025-10", "2025-11")
CAPACITY = 250.0          # s RMS of the served update on the 388, registered
GRID = 201                # barycentric grid over the 2-simplex
M0 = 30.0                 # the registered repair's pseudo-rows
COLS = ("one", "zs", "zs2", "cross", "sin_tod", "cos_tod", "z_stand", "weekend")


def basis(d: pd.DataFrame, mu: dict, sd: dict) -> np.ndarray:
    zs = (d.mvt_sched_log.to_numpy(float) - mu["s"]) / sd["s"]
    tod = 2.0 * np.pi * d.minute_of_day.to_numpy(float) / 1440.0
    return np.column_stack([
        np.ones(len(d)), zs, zs ** 2, d.sched_date_cross.to_numpy(float),
        np.sin(tod), np.cos(tod),
        (d.stand_busy_frac_6h.to_numpy(float) - mu["b"]) / sd["b"],
        np.isin(d.dow.to_numpy(int), (5, 6)).astype(float)])


def solve(G: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), c)


def simplex_grid(k: int) -> np.ndarray:
    """All barycentric weights on the 2-simplex at resolution `k-1`."""
    n = k - 1
    out = [(i / n, j / n, (n - i - j) / n)
           for i in range(k) for j in range(k - i)]
    return np.asarray(out, float)


def worst_alpha(Gs, cs, lam, alphas):
    """Dual of `min_w max_m L_m(w) + lam||w||^2`: `max_alpha -c(a)'w(a)`."""
    best, bw, ba = -np.inf, None, None
    for a in alphas:
        G = sum(a[i] * Gs[i] for i in range(len(Gs)))
        c = sum(a[i] * cs[i] for i in range(len(cs)))
        w = solve(G, c, lam)
        g = -float(c @ w)
        if g > best:
            best, bw, ba = g, w, a
    return ba, bw


def move_388(w, X388, v7):
    return float(np.sqrt(np.mean((np.clip(v7 + X388 @ w, MIN_TAXI, None) - v7) ** 2)))


def bisect(make_w, X388, v7, target=CAPACITY, lo=1e-8, hi=1e12):
    """`lam` with the registered movement on the 388; movement falls with `lam`."""
    if move_388(make_w(lo), X388, v7) < target:
        return lo, make_w(lo), "lower bound"
    if move_388(make_w(hi), X388, v7) > target:
        return hi, make_w(hi), "upper bound"
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if move_388(make_w(mid), X388, v7) > target:
            lo = mid
        else:
            hi = mid
    lam = np.sqrt(lo * hi)
    return lam, make_w(lam), "ok"


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "base_lomo.parquet")
    b = pd.read_parquet(SRC / "noac_2026_d.parquet")
    fit = a[a.month.isin(FIT_MONTHS)].reset_index(drop=True)
    mu = {"s": float(fit.mvt_sched_log.mean()), "b": float(fit.stand_busy_frac_6h.mean())}
    sd = {"s": float(fit.mvt_sched_log.std(ddof=0)),
          "b": float(fit.stand_busy_frac_6h.std(ddof=0))}

    Xa, X388 = basis(a, mu, sd), basis(b, mu, sd)
    v7 = b.v7.to_numpy(float)
    r_fit = (fit.y_true - fit.base).to_numpy(float)
    Xf = basis(fit, mu, sd)

    Gs, cs, ns = [], [], []
    for m in FIT_MONTHS:
        s = (fit.month == m).to_numpy()
        Xm, rm = Xf[s], r_fit[s]
        Gs.append(Xm.T @ Xm / len(rm))
        cs.append(Xm.T @ rm / len(rm))
        ns.append(int(s.sum()))
    N = sum(ns)
    a_pool = np.array([n / N for n in ns])
    Gp = sum(a_pool[i] * Gs[i] for i in range(3))
    cp = sum(a_pool[i] * cs[i] for i in range(3))
    alphas = simplex_grid(GRID)

    arms, meta = {}, {}
    lam_p, w_p, st_p = bisect(lambda l: solve(Gp, cp, l), X388, v7)
    arms["P"], meta["P"] = w_p, {"lambda": lam_p, "bisect": st_p, "alpha": a_pool.tolist()}

    lam_w, w_w, st_w = bisect(lambda l: worst_alpha(Gs, cs, l, alphas)[1], X388, v7)
    al_w = worst_alpha(Gs, cs, lam_w, alphas)[0]
    arms["W"], meta["W"] = w_w, {"lambda": lam_w, "bisect": st_w,
                                 "alpha": np.asarray(al_w).tolist()}

    # the registered repair, if a sparse fit month takes the mass
    sparse = [i for i, n in enumerate(ns) if n < 100]
    trig = any(al_w[i] > 0.6 for i in sparse)
    meta["repair_triggered"] = bool(trig)
    if trig:
        Gt = [(ns[i] * Gs[i] + M0 * Gp) / (ns[i] + M0) for i in range(3)]
        ct = [(ns[i] * cs[i] + M0 * cp) / (ns[i] + M0) for i in range(3)]
        lam_r, w_r, st_r = bisect(lambda l: worst_alpha(Gt, ct, l, alphas)[1], X388, v7)
        al_r = worst_alpha(Gt, ct, lam_r, alphas)[0]
        arms["WR"], meta["WR"] = w_r, {"lambda": lam_r, "bisect": st_r,
                                       "alpha": np.asarray(al_r).tolist()}

    out = a[["MVT_ID_mvt", "month", "day", "band", "y_true", "mvt_sched", "base"]].copy()
    for k, w in arms.items():
        cand = np.clip(a.base.to_numpy(float) + Xa @ w, MIN_TAXI, None)
        out[f"d_{k}"] = cand - a.base.to_numpy(float)
        out[f"lin_{k}"] = Xa @ w
    out.to_parquet(EXP / "work" / "arms_2025.parquet", index=False)

    s388 = b[["MVT_ID_mvt", "band", "mvt_sched", "v7"]].copy()
    for k, w in arms.items():
        s388[f"d_{k}"] = np.clip(v7 + X388 @ w, MIN_TAXI, None) - v7
    s388.to_parquet(EXP / "work" / "arms_388.parquet", index=False)

    def per_month_L(w):
        d = Xf @ w
        return {m: float(np.mean(d[(fit.month == m).to_numpy()] ** 2
                                 - 2 * r_fit[(fit.month == m).to_numpy()]
                                 * d[(fit.month == m).to_numpy()]))
                for m in FIT_MONTHS}

    rec = {"fit_months": list(FIT_MONTHS), "read_months": list(READ_MONTHS),
           "n_fit": ns, "capacity_s": CAPACITY, "mu": mu, "sd": sd,
           "cols": list(COLS),
           "arms": {k: {"w": arms[k].tolist(), **meta[k],
                        "move_388_rms": move_388(arms[k], X388, v7),
                        "move_fit_rms": float(np.sqrt(np.mean((Xf @ arms[k]) ** 2))),
                        "L_fit_pooled": float(np.mean((Xf @ arms[k]) ** 2
                                                      - 2 * r_fit * (Xf @ arms[k]))),
                        "L_fit_per_month": per_month_L(arms[k])}
                    for k in arms},
           "repair_triggered": bool(trig),
           "clip_binds_2025": {k: int((np.abs(out[f"d_{k}"] - out[f"lin_{k}"]) > 1e-9).sum())
                               for k in arms}}
    (EXP / "analysis" / "arms.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: {"lambda": rec["arms"][k]["lambda"],
                          "alpha": rec["arms"][k]["alpha"],
                          "move_388": round(rec["arms"][k]["move_388_rms"], 1),
                          "move_fit": round(rec["arms"][k]["move_fit_rms"], 1),
                          "L_fit": {m: round(v, 1) for m, v in
                                    rec["arms"][k]["L_fit_per_month"].items()},
                          "bisect": rec["arms"][k]["bisect"]} for k in arms}, indent=2))
    print("repair triggered:", trig, "| clip binds:", rec["clip_binds_2025"])
    for k in arms:
        print(f"  w[{k}] = " + ", ".join(f"{c}={v:+.1f}" for c, v in zip(COLS, arms[k])))


if __name__ == "__main__":
    main()
