"""The TWO-month extension's prediction, assembled from the three per-fold curves.

Section 47.23 registers, BEFORE queue's 2025-10 base6 and block exist, what the curve says
the SECOND extra month buys as an increment over the 2025-11-extended member.  It refits
nothing: it reads the four cached level vectors section 47.15 fitted and prices them again,
exactly as `code/curveshape.py` does.

The fit is COPIED from `curveshape.py` rather than imported, because that file does its work
inside `main()`.  Copying is a divergence risk, so the copy is CHECKED: this file asserts its
own per-fold alpha and asymptote reproduce `analysis/curve-shape.json` to 1e-12 before it
predicts anything.  If the two ever drift apart, this stops instead of registering a band
from a second, quietly different curve.

Why the second month is not simply "the first month again": a fold's fitting rows are the
other two fold months, so each extra month is a different fraction of each fold's set, and
the curve saturates -- the second month is priced at the multiple the first one leaves behind.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/twomonth.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402
import curve as CV                                                    # noqa: E402

OUT = EXP / "analysis/twomonth-predict.json"

#: 2025-11's writable rows, MEASURED (section 47.18, and the count every fit since carried).
N11 = 99709
#: 2025-10 does not exist yet.  Queue's OOF carries 185,674 rows; 2025-11 gave 99,709
#: writable out of 162,332, a ratio of 0.61423, so the estimate below is that ratio applied.
#: It is an ESTIMATE and the registration is re-evaluated at the true count before the read.
N10_TOTAL_OOF = 185674
N11_TOTAL_OOF = 162332
N10_ESTIMATE = int(round(N10_TOTAL_OOF * (N11 / N11_TOTAL_OOF)))
#: the sensitivity the registration is read across, so the true count lands inside a stated row
N10_GRID = [85000, 95000, 105000, N10_ESTIMATE, 120000, 130000]
#: once `code/octcheck.py` has counted the month for real, the TRUE figure joins the grid and
#: the registration is re-evaluated on it -- section 47.23.4 caveat 3, pre-registered
_OC = EXP / "analysis/octcheck.json"
N10_TRUE = None
if _OC.exists():
    import json as _j
    N10_TRUE = int(_j.loads(_OC.read_text())["WRITABLE_ROWS"])
    if N10_TRUE not in N10_GRID:
        N10_GRID.append(N10_TRUE)

#: the K = 1 screen's MEASURED per-fold gain from 2025-11 alone (section 47.18), s^2 per row,
#: beside the curve's prediction for the same month -- the ratio is what section 47.23's
#: adjacency claim is registered on
MEASURED_MONTH_1 = {"july_loo": 312.90, "december": 957.11, "january": 405.94}


def main() -> None:
    t0 = time.time()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan_ship = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    levels = {}
    for frac in CV.FRACS:
        cache = EXP / f"work/d-curve{int(round(frac*100))}-255-free.npz"
        if not cache.exists():
            raise SystemExit(f"{cache.name} is missing -- section 47.15's levels are gone")
        levels[frac] = {f: v.astype("float64") for f, v in np.load(cache).items()}
    d100 = levels[1.00]
    ph = R.head_panel(pan_ship, d100)
    incs = {f"{int(round(fr*100))}pct": {f: levels[fr][f] - d100[f] for f in A.FOLDS}
            for fr in CV.FRACS}
    sub = [fr for fr in CV.FRACS if fr < 1.0]
    u = np.array(sub)
    grid = np.linspace(0.02, 4.0, 800)
    Hm = 1.0 - u[:, None] ** (-grid[None, :])
    hh = (Hm * Hm).sum(0)

    def power_fit(gv):
        gv = np.atleast_2d(gv.T).T if gv.ndim > 1 else gv[:, None]
        Cs = (Hm.T @ gv) / hh[:, None]
        sse = ((gv[None, :, :] - Cs[:, None, :] * Hm.T[:, :, None]) ** 2).sum(1)
        j = sse.argmin(0)
        return grid[j], Cs[j, np.arange(gv.shape[1])]

    CJ = json.loads((EXP / "analysis/curve.json").read_text())
    saved = json.loads((EXP / "analysis/curve-shape.json").read_text())["per_fold_bounded_fit"]
    alpha, C, a_draws, C_draws, mse0, nfit = {}, {}, {}, {}, {}, {}
    for f in A.FOLDS:
        pf = ph[f]
        yv = pf["y_true"].to_numpy("float64")
        pbv = pf["pbase"].to_numpy("float64")
        codes, uniq = np.unique(pf["day"].to_numpy(), return_inverse=True)[::-1]
        kd = len(uniq)
        r0 = (yv - pbv) ** 2
        s0 = np.bincount(codes, weights=r0, minlength=kd)
        cc = np.bincount(codes, minlength=kd).astype("float64")
        rf = np.random.default_rng(BA.SEED)
        drw = rf.integers(0, kd, size=(BA.BOOT, kd))
        n0, ct = s0[drw].sum(1), cc[drw].sum(1)
        gsub, gdraw = [], []
        for fr in sub:
            dd = np.asarray(incs[f"{int(round(fr*100))}pct"][f], dtype="float64")
            s1 = np.bincount(codes, weights=(yv - pbv - dd) ** 2, minlength=kd)
            gsub.append(float(r0.mean() - ((yv - pbv - dd) ** 2).mean()))
            gdraw.append(n0 / ct - s1[drw].sum(1) / ct)
        af, Cf = power_fit(np.array(gsub))
        ad, Cd = power_fit(np.stack(gdraw))
        alpha[f], C[f] = float(af[0]), float(Cf[0])
        a_draws[f], C_draws[f] = ad, Cd
        mse0[f] = float(r0.mean())
        nfit[f] = int(CJ["levels"]["100"]["folds"][f]["fit_rows"])
        # THE COPY IS CHECKED, not trusted
        for key, got in (("alpha", alpha[f]), ("asymptote_s2_per_row", C[f])):
            want = float(saved[f][key])
            if abs(got - want) > 1e-12:
                raise SystemExit(f"{f}: this file's {key} is {got!r}, curve-shape.json has "
                                 f"{want!r} -- the copied fit has DIVERGED, stopping")

    def g_at(n_extra):
        """per-fold gain over the CURRENT panel, s^2/row, point and draws."""
        mult = {f: (nfit[f] + n_extra) / nfit[f] for f in A.FOLDS}
        pt = {f: C[f] * (1.0 - mult[f] ** -alpha[f]) for f in A.FOLDS}
        dr = {f: C_draws[f] * (1.0 - mult[f] ** -a_draws[f]) for f in A.FOLDS}
        return mult, pt, dr

    def board_of(gp):
        return BA.board({f: max(mse0[f] - gp[f], 0.0) ** 0.5 for f in A.FOLDS})

    def board_draws(gd):
        return np.sqrt(sum(BA.SHARES[f] * np.maximum(mse0[f] - gd[f], 0.0)
                           for f in BA.SHARES))

    m1, g1, d1 = g_at(N11)
    b1_pt, b1_dr = board_of(g1), board_draws(d1)

    rows = {}
    for n10 in N10_GRID:
        m2, g2, d2 = g_at(N11 + n10)
        b2_pt, b2_dr = board_of(g2), board_draws(d2)
        inc_dr = b1_dr - b2_dr                       # paired inside each draw
        lo, hi = (float(v) for v in np.percentile(inc_dr, [2.5, 97.5]))
        per_fold = {}
        for f in A.FOLDS:
            fd = d2[f] - d1[f]
            flo, fhi = (float(v) for v in np.percentile(fd, [2.5, 97.5]))
            per_fold[f] = {
                "multiple_after_one_month": float(m1[f]),
                "multiple_after_two_months": float(m2[f]),
                "second_month_gain_s2_per_row": float(g2[f] - g1[f]),
                "ci95": [flo, fhi],
                "interval_clear": bool(flo > 0),
                "first_month_gain_s2_per_row_predicted": float(g1[f]),
                "first_month_gain_s2_per_row_MEASURED": MEASURED_MONTH_1[f],
                "month_1_measured_over_predicted": float(
                    MEASURED_MONTH_1[f] / g1[f]) if g1[f] > 0 else None}
        rows[f"n{n10}"] = {
            "n_2025_10_writable_rows": n10,
            "board_second_month_gain_s": float(b1_pt - b2_pt),
            "board_ci95": [lo, hi],
            "every_fold_interval_clear_predicted": all(
                v["interval_clear"] for v in per_fold.values()),
            "by_fold": per_fold}

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "refits_nothing": "reads work/d-curve{25,50,75,100}-255-free.npz",
        "copied_fit_checked_against": "analysis/curve-shape.json per_fold_bounded_fit, 1e-12",
        "n_2025_11_writable_rows": N11,
        "n_2025_10_TRUE": N10_TRUE,
        "n_2025_10_estimate": {
            "value": N10_ESTIMATE,
            "how": (f"queue's 2025-10 OOF carries {N10_TOTAL_OOF} rows; 2025-11 gave {N11} "
                    f"writable of {N11_TOTAL_OOF}, a ratio of "
                    f"{N11 / N11_TOTAL_OOF:.5f}, applied here"),
            "status": "ESTIMATE -- the registration is re-evaluated at the true count before the read"},
        "per_fold_curve": {f: {"alpha": alpha[f], "asymptote_s2_per_row": C[f],
                               "current_fitting_rows": nfit[f], "mse0_s2": mse0[f]}
                           for f in A.FOLDS},
        "board_gain_of_the_FIRST_month_predicted_s": float(
            BA.board({f: mse0[f] ** 0.5 for f in A.FOLDS}) - b1_pt),
        "by_2025_10_row_count": rows,
        "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    e = rows[f"n{N10_TRUE if N10_TRUE else N10_ESTIMATE}"]
    if N10_TRUE:
        print(f"TRUE 2025-10 writable rows {N10_TRUE:,} (estimate was {N10_ESTIMATE:,}, "
              f"{100*(N10_ESTIMATE/N10_TRUE-1):+.1f}%)")
    print(f"the copied fit reproduces curve-shape.json on all three folds")
    print(f"2025-10 estimated writable rows: {N10_ESTIMATE:,}")
    print(f"\n  SECOND month, board  {e['board_second_month_gain_s']:+.5f} "
          f"[{e['board_ci95'][0]:+.5f}, {e['board_ci95'][1]:+.5f}]  "
          f"every-fold-clear {e['every_fold_interval_clear_predicted']}")
    for f in A.FOLDS:
        v = e["by_fold"][f]
        print(f"    {f:<9} {v['second_month_gain_s2_per_row']:+8.2f} "
              f"[{v['ci95'][0]:+8.2f}, {v['ci95'][1]:+8.2f}]   "
              f"month 1: predicted {v['first_month_gain_s2_per_row_predicted']:+7.2f}, "
              f"measured {v['first_month_gain_s2_per_row_MEASURED']:+7.2f} "
              f"(x{v['month_1_measured_over_predicted']:.2f})")
    print(f"\n  board over the current panel, FIRST month predicted "
          f"{rec['board_gain_of_the_FIRST_month_predicted_s']:+.5f}")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
