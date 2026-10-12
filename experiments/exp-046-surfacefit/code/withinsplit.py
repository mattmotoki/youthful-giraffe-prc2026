"""RESULTS section 47.73: #4 of 47.72 -- the within/cross-month split on the head's OWN arm.

The shipped fold read already trains fold f's arm on months other than f (arms.arm_lgb: the
amplitude is cross-fitted inside the FITTING months and never touches the scored one), so the
face carries no structure specific to the fold month.  What it can carry that the board cannot
get is structure shared with its training months and not with 2026.  The part of that visible in
2025 data is TEMPORAL LOCALITY; the year-specific part (H2) is not (47.59.4).  This measures the
visible part, on the head's own arm, per edge fold:

    W      the fold month's OTHER days            cyclic 0, time ~0   (two fits: A->B, B->A)
    N1/N2  the month at (c1,t1) / (c2,t2)         near in season AND time
    F1/F2  the month at (c1,t11) / (c2,t10)       near in season, FAR in time
    P      N1's rows, target permuted within airport x day   (placebo; must read ~0)

Every training set carries the SAME per-airport row budget, the SAME target (y - base6, OOF,
from queue's oof/ -- the fold months included, so a fold-month source and an extension-month
source are not fitted to different bases: the 47.17 mismatch is 30-39 s), the SAME features and
the shipped LightGBM setting.  Every arm is scored on the SAME rows: the fold month's class rows
at the airports every source covers, less HOLD.  Gains use the record's statistic
2*mean(r*d) - mean(d^2) with a day-clustered paired bootstrap.

    .venv/bin/python experiments/exp-046-surfacefit/code/withinsplit.py --fold december --plan
    .venv/bin/python experiments/exp-046-surfacefit/code/withinsplit.py --fold december

--plan builds every set and prints the budgets, fits nothing and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402
import onemonth as OM                                                 # noqa: E402
import nooverwrite as NO                                              # noqa: E402

OOF = ROOT / "experiments/exp-047-queue/oof"
SEED = 4773
N_BOOT = 2000
# RESULTS 47.73.3, registered before any fit.  (source, cyclic, time) relative to the fold month;
# a bare fold name is that fold's panel, a YYYY-MM is an extension month.  Each far source is
# matched in CYCLIC distance to its near source and differs from it only in TIME.
DESIGN = {
    "december": {"N1": ("2025-11", 1, 1), "N2": ("2025-10", 2, 2),
                 "F1": ("january", 1, 11), "F2": ("2025-02", 2, 10)},
    "january": {"N1": ("2025-02", 1, 1), "N2": ("2025-03", 2, 2),
                "F1": ("december", 1, 11), "F2": ("2025-11", 2, 10)},
}


def base6_target(frame: pd.DataFrame, fold: str) -> pd.DataFrame:
    """Replace a fold panel's `rbase` (true price base) by y - base6, the extension target."""
    b6 = pd.read_parquet(OOF / f"base6--{fold}.parquet").set_index("MVT_ID_mvt")
    idx = frame.MVT_ID_mvt.to_numpy("float64")
    p6 = b6.y_pred.reindex(idx).to_numpy("float64")
    y6 = b6.y_true.reindex(idx).to_numpy("float64")
    y = frame.y_true.to_numpy("float64")
    if not np.isfinite(p6).all():
        raise SystemExit(f"{fold}: {int((~np.isfinite(p6)).sum())} rows did not join base6")
    if int((np.abs(y6 - y) > 1e-6).sum()):
        raise SystemExit(f"{fold}: base6's y_true differs from the panel's -- bad join")
    out = frame.copy()
    out["rbase"] = y - p6
    return out


def fold_rows(pan: dict, ship: dict, fold: str) -> pd.DataFrame:
    p = pan[fold][pan[fold].usable.to_numpy(bool)].reset_index(drop=True)
    p = p.join(ship[fold].reindex(p.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
    return base6_target(p, fold)


def source_rows(pan: dict, ship: dict, scols: list, src: str) -> pd.DataFrame:
    if src in A.FOLDS:
        return fold_rows(pan, ship, src)
    x = OM.nov_rows(scols, month=src)           # rbase is already y - base6 here
    return x[x.usable.to_numpy(bool)].reset_index(drop=True)


def halves(days: pd.Series) -> np.ndarray:
    """Four contiguous day blocks; half A = blocks 0 and 2, half B = blocks 1 and 3."""
    d = pd.to_datetime(days).dt.normalize()
    u = np.sort(d.unique())
    code = {v: min(int(i * 4 / len(u)), 3) for i, v in enumerate(u)}
    return (d.map(code).to_numpy() % 2).astype(int)          # 0 = A, 1 = B


def draw(frame: pd.DataFrame, budget: dict, seed: int) -> pd.DataFrame:
    """Exactly budget[a] rows at each airport a, without replacement, original order kept."""
    rng = np.random.default_rng(seed)
    ap = frame.airport.astype(str).to_numpy()
    keep = []
    for a in sorted(budget):
        pos = np.flatnonzero(ap == a)
        if len(pos) < budget[a]:
            raise SystemExit(f"airport {a}: {len(pos)} rows, budget {budget[a]}")
        keep.append(rng.choice(pos, size=budget[a], replace=False))
    return frame.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)


def gains_by_day(r: np.ndarray, d: np.ndarray, day: np.ndarray):
    """Per-day sums of r*d, d^2 and n, so a day bootstrap is a resample of three vectors."""
    u, inv = np.unique(day, return_inverse=True)
    s1 = np.bincount(inv, weights=r * d, minlength=len(u))
    s2 = np.bincount(inv, weights=d * d, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    return s1, s2, n



# RESULTS 47.73.2 -- the board history, READ ONLY.  These enter as a COMPARATOR for the measured
# rho and never as an input to it or to any candidate (score-feedback rule).  Per-fold s2/row of each uploaded
# step, the step's board gain, and the months each head added, from 47.21.1 / 47.40.1 / the
# seasonal-bag10 read; v21's per-month shares from the alone table.
BOARD_W = {"jul": 0.0009318464575330257, "jan": 0.00074096}
STEPS = {"v20": {"jul": (290.27, {11: 1.0}), "jan": (327.36, {11: 1.0}), "board": 0.4851},
         "v21": {"jul": (387.98, {10: 0.6824, 3: 0.3176}),
                 "jan": (264.17, {10: 0.1490, 3: 0.8510}), "board": 0.1961},
         "v22": {"jul": (307.41, {8: 1.0}), "jan": (148.24, {2: 1.0}), "board": 0.0733}}


def board_dt(m: int, served: str) -> int:
    """t_board - t_fold for 2025-m: cyclic distance is year-invariant, time distance mirrors."""
    return (19 - m) - abs(7 - m) if served == "jul" else (13 - m) - (m - 1)


def step_pred(step: str, rho: float) -> tuple[float, float]:
    d, num, den = STEPS[step], 0.0, 0.0
    for f, w in BOARD_W.items():
        g, sh = d[f]
        den += w * g
        num += w * g * sum(v * rho ** board_dt(m, f) for m, v in sh.items())
    return num, den


def pool(overwrite: bool) -> None:
    out_json = EXP / "analysis/withinsplit-pooled.json"
    NO.refuse_existing([out_json], overwrite=overwrite)
    fa = {f: json.loads((EXP / f"analysis/withinsplit-{f}.json").read_text()) for f in DESIGN}
    for f, j in fa.items():
        if j["placebo_clear_of_zero"]:
            raise SystemExit(f"{f}: the placebo CLEARS zero -- 47.73.4 voids this fold's read")
        weak = [k for k in ("W", "N1", "N2") if j["gain_s2"][k]["ci95"][0] <= 0]
        if weak:
            raise SystemExit(f"{f}: {weak} do not clear zero at the interval bottom -- 47.73.4's "
                             f"kill line: this fold's ratios are reported, never pooled")
    dts = sum(j["rho_fold_pooled"]["log_sum_tau_for_pooling"]["sum_delta_t"] for j in fa.values())
    pts = [j["rho_fold_pooled"]["log_sum_tau_for_pooling"]["sum_log_tau"] for j in fa.values()]
    rho = float(np.exp(sum(pts) / dts)) if all(v is not None for v in pts) else None
    a, b = (np.array([np.nan if v is None else v for v in fa[f]["draws_sum_log_tau"]])
            for f in DESIGN)
    ok = np.isfinite(a) & np.isfinite(b)
    rd = np.exp((a[ok] + b[ok]) / dts)
    ci = [float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))]
    band = ("ABSENT" if ci[0] >= 0.99 else "DOMINANT" if ci[1] < 0.90 else
            "PARTIAL" if ci[0] >= 0.90 and ci[1] < 0.99 else "STRADDLES")
    expl = {}
    for s in ("v21", "v22"):
        num, den = step_pred(s, rho) if rho else (None, None)
        lo = step_pred(s, ci[1])[0]
        hi = step_pred(s, ci[0])[0]
        loss = den - STEPS[s]["board"] if den else None
        expl[s] = {"predicted_board": num, "predicted_board_ci_from_rho": [hi, lo],
                   "board_read_only": STEPS[s]["board"], "face": den,
                   "share_of_transfer_loss_explained": (den - num) / loss if loss else None}
    v20 = step_pred("v20", rho)[0] if rho else None
    k = {f: fa[f]["kappa_N1_over_W"]["point"] for f in DESIGN}
    out = {"section": "RESULTS 47.73.5", "rho_pooled": rho, "rho_pooled_ci95": ci,
           "draws_usable": int(ok.sum()), "outcome_band": band,
           "by_fold": {f: {"rho": fa[f]["rho_fold_pooled"]["point"],
                           "rho_ci95": fa[f]["rho_fold_pooled"]["ci95_on_positive_draws"],
                           "tau_c1": fa[f]["tau_c1_F1_over_N1"]["point"],
                           "tau_c2": fa[f]["tau_c2_F2_over_N2"]["point"],
                           "kappa": k[f]} for f in DESIGN},
           "discount_by_month": {s: {m: (rho ** board_dt(m, s) if rho else None)
                                     for m in (2, 3, 6, 8, 10, 11)} for s in ("jul", "jan")},
           "explained_share_board_read_only": expl,
           "v20_predicted_board_read_only": v20, "v20_board_read_only": STEPS["v20"]["board"],
           "board_c0_t12_over_fold_c1_t1": {f: (rho ** 12 / k[f] if rho and k[f] else None)
                                            for f in DESIGN}}
    NO.refuse_existing([out_json], overwrite=overwrite)
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"written to {out_json.relative_to(ROOT)}")


def main() -> None:
    t0 = time.time()
    ps = argparse.ArgumentParser()
    ps.add_argument("--fold", choices=sorted(DESIGN))
    ps.add_argument("--pool", action="store_true",
                    help="47.73.5: pool the two fold artifacts into one rho; zero fit")
    ps.add_argument("--plan", action="store_true",
                    help="build every set and print the budgets; fit nothing, write nothing")
    ps.add_argument("--overwrite", action="store_true",
                    help="47.71.6: allow replacing an existing artifact")
    args = ps.parse_args()
    if args.pool:
        return pool(args.overwrite)
    if not args.fold:
        raise SystemExit("--fold is required unless --pool")
    fold = args.fold
    out_json = EXP / f"analysis/withinsplit-{fold}.json"
    if not args.plan:
        NO.refuse_existing([out_json], overwrite=args.overwrite)
        print(f"artifact {out_json.name}: no existing artifact", flush=True)

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        mth = A.FOLD_MONTH[f]
        b = CF.build_month(mth, A.prefix_months_for(mth), rule)
        R.assert_mode3(b, mth, "shipped")
        blk[mth] = b
    pan = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
    ship, scols = A.shipped_cols(pan)
    cols = [c for c in CF.BU.FEATURES if c not in ("stand_lat", "stand_lon")] + scols

    E_all = fold_rows(pan, ship, fold)
    src = {k: source_rows(pan, ship, scols, v[0]) for k, v in DESIGN[fold].items()}
    aps = set(E_all.airport.astype(str))
    for k in src:
        aps &= set(src[k].airport.astype(str))
    aps -= set(CF.HOLD)
    aps = sorted(aps)
    E = E_all[E_all.airport.astype(str).isin(aps)].reset_index(drop=True)
    h = halves(E.day)
    Ea, Eb = E[h == 0].reset_index(drop=True), E[h == 1].reset_index(drop=True)
    budget = {}
    for a in aps:
        c = [int((Ea.airport.astype(str) == a).sum()), int((Eb.airport.astype(str) == a).sum())]
        c += [int((s.airport.astype(str) == a).sum()) for s in src.values()]
        budget[a] = min(c)
    N = int(sum(budget.values()))
    plan = {"fold": fold, "airports": aps, "held_out": list(CF.HOLD),
            "budget_per_airport": budget, "fit_rows_per_set": N,
            "eval_rows": int(len(E)), "eval_rows_half_A": int(len(Ea)),
            "eval_rows_half_B": int(len(Eb)),
            "fold_class_rows_before_airport_cut": int(len(E_all)),
            "days_half_A": int(pd.to_datetime(Ea.day).dt.normalize().nunique()),
            "days_half_B": int(pd.to_datetime(Eb.day).dt.normalize().nunique()),
            "sources": {k: {"source": v[0], "cyclic": v[1], "time": v[2],
                            "class_rows": int(len(src[k]))} for k, v in DESIGN[fold].items()}}
    print(json.dumps(plan, indent=1), flush=True)
    if args.plan:
        print(f"PLAN ONLY -- nothing fitted, nothing written ({time.time()-t0:.0f}s)")
        return

    sets = {"W_A": draw(Ea, budget, SEED + 1), "W_B": draw(Eb, budget, SEED + 2)}
    for i, k in enumerate(DESIGN[fold]):
        sets[k] = draw(src[k], budget, SEED + 10 + i)
    pl = sets["N1"].copy()                                   # placebo: N1's rows, target shuffled
    rng = np.random.default_rng(SEED + 99)
    grp = pl.airport.astype(str) + "|" + pd.to_datetime(pl.day).dt.strftime("%Y-%m-%d")
    rb = pl.rbase.to_numpy("float64").copy()
    for _, pos in pl.groupby(grp).indices.items():
        rb[pos] = rb[rng.permutation(pos)]
    pl["rbase"] = rb
    sets["P"] = pl

    r = E.rbase.to_numpy("float64")
    day = pd.to_datetime(E.day).dt.strftime("%Y-%m-%d").to_numpy()
    d = {}
    for k in ("W_A", "W_B", "N1", "N2", "F1", "F2", "P"):
        t1 = time.time()
        sco = {"W_A": Eb, "W_B": Ea}.get(k, E)
        pred, _ = A.arm_lgb(sets[k], sco, cols, params=CF.SETTING)
        d[k] = pred
        print(f"  {k:4} fitted on {len(sets[k]):,} rows, scored {len(sco):,} ({time.time()-t1:.0f}s)",
              flush=True)
    dW = np.zeros(len(E))
    dW[h == 1] = d.pop("W_A")
    dW[h == 0] = d.pop("W_B")
    d = {"W": dW, **d}

    agg = {k: gains_by_day(r, v, day) for k, v in d.items()}
    nday = len(agg["W"][2])
    rng = np.random.default_rng(SEED + 7)
    idx = rng.integers(0, nday, size=(N_BOOT, nday))

    def G(k, sel=None):
        s1, s2, n = agg[k]
        if sel is None:
            return float((2 * s1.sum() - s2.sum()) / n.sum())
        return (2 * s1[sel].sum(1) - s2[sel].sum(1)) / n[sel].sum(1)

    point = {k: G(k) for k in d}
    boot = {k: G(k, idx) for k in d}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {"gain_s2": {k: {"point": point[k], "ci95": ci(boot[k])} for k in d}}
    ratios = {"kappa_N1_over_W": ("N1", "W", None),
              "tau_c1_F1_over_N1": ("F1", "N1", DESIGN[fold]["F1"][2] - DESIGN[fold]["N1"][2]),
              "tau_c2_F2_over_N2": ("F2", "N2", DESIGN[fold]["F2"][2] - DESIGN[fold]["N2"][2])}
    for name, (a, b, dt) in ratios.items():
        pv, bv = point[a] / point[b], boot[a] / boot[b]
        ent = {"point": pv, "ci95": ci(bv), "share_of_draws_with_nonpositive_ratio":
               float((bv <= 0).mean()), "delta_t": dt}
        if dt:
            ok = bv > 0
            ent["rho_point"] = pv ** (1 / dt) if pv > 0 else None
            ent["rho_ci95_on_positive_draws"] = ci(bv[ok] ** (1 / dt)) if ok.any() else None
        res[name] = ent
    t1v, t2v = boot["F1"] / boot["N1"], boot["F2"] / boot["N2"]
    dts = ratios["tau_c1_F1_over_N1"][2] + ratios["tau_c2_F2_over_N2"][2]
    okb = (t1v > 0) & (t2v > 0)
    lp = (np.log(point["F1"] / point["N1"]) + np.log(point["F2"] / point["N2"])) / dts \
        if point["F1"] > 0 and point["N1"] > 0 and point["F2"] > 0 and point["N2"] > 0 else None
    res["rho_fold_pooled"] = {
        "point": float(np.exp(lp)) if lp is not None else None,
        "ci95_on_positive_draws": ci(np.exp((np.log(t1v[okb]) + np.log(t2v[okb])) / dts))
        if okb.any() else None,
        "share_of_draws_usable": float(okb.mean()),
        "log_sum_tau_for_pooling": {"sum_log_tau": float(np.log(point["F1"] / point["N1"])
                                                         + np.log(point["F2"] / point["N2"]))
                                    if lp is not None else None, "sum_delta_t": dts}}
    slt = np.where(okb, np.log(np.where(t1v > 0, t1v, 1.0)) + np.log(np.where(t2v > 0, t2v, 1.0)),
                   np.nan)
    res["draws_sum_log_tau"] = [None if not np.isfinite(v) else float(v) for v in slt]
    res["placebo_clear_of_zero"] = not (res["gain_s2"]["P"]["ci95"][0] <= 0
                                        <= res["gain_s2"]["P"]["ci95"][1])
    out = {"section": "RESULTS 47.73", "plan": plan, "seed": SEED, "n_boot": N_BOOT,
           "n_days": int(nday), "statistic": "2*mean(r*d) - mean(d^2), r = y - base6",
           **res, "seconds": round(time.time() - t0, 1)}
    NO.refuse_existing([out_json], overwrite=args.overwrite)
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("gain_s2", "kappa_N1_over_W", "tau_c1_F1_over_N1",
                                          "tau_c2_F2_over_N2", "rho_fold_pooled",
                                          "placebo_clear_of_zero")}, indent=1))
    print(f"written to {out_json.relative_to(ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
