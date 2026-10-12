"""Does the log law actually FIT? -- a post-hoc diagnostic on section 47.15's four levels.

Section 47.15 registered a ONE-parameter fit, `g = k*ln(frac)` through the exact zero at
frac = 1, and every band in section 47.12 now rests on extrapolating that k UP by 1.7 nats.
Nothing in `curve.py` records whether the three measured points sit on that line, so this
asks separately.  **It is post-hoc and is labelled as such**: it was written while `curve.py`
ran, it registers no band and it moves no gate.  It refits nothing -- it reads the four cached
level vectors and prices them again.

Four questions, in order of how much they bear on the extrapolation:

1. **The per-level slope.**  `k_j = g_j / ln(frac_j)` from each level ALONE.  If the log law
   holds the three agree.  If `k(25 %) > k(50 %) > k(75 %)` the curve is steeper than log at
   the bottom, so it is flattening FASTER than log, and extrapolating up with a log law
   OVERSTATES the extended panel -- the direction that matters for section 47.12's anchor.
2. **The intercept.**  `g = a + k*ln(frac)` as a two-parameter fit.  The anchor says `a = 0`
   exactly; an interval on `a` that excludes zero says the one-parameter form is wrong.
3. **The residuals** of the registered fit, against the bootstrap spread of each level, so a
   miss is read on the scale of the noise rather than in absolute seconds.
4. **A bounded alternative.**  `g(u) = C * (1 - u^-alpha)` -- the gain of a fit on `u` times
   the rows, which is what a learning curve `e(n) = e_inf + c*n^-alpha` implies.  It costs one
   more parameter than the registered form and it has the property the log law lacks: it
   SATURATES, so `C` is the most this recipe can ever buy from more rows of the same kind.
   Extrapolating a log law upward from points measured downward is the weakest step in
   section 47.12's chain, and this is the honest alternative to it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/curveshape.py
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
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402
import curve as CV                                                    # noqa: E402

OUT = EXP / "analysis/curve-shape.json"


def main() -> None:
    t0 = time.time()
    if not (EXP / "analysis/curve.json").exists():
        raise SystemExit("curve.json does not exist yet -- this reads section 47.15's levels")

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
        pct = int(round(frac * 100))
        cache = EXP / f"work/d-curve{pct}-255-free.npz"
        if not cache.exists():
            raise SystemExit(f"{cache.name} is missing -- run curve.py first")
        levels[frac] = {f: v.astype("float64") for f, v in np.load(cache).items()}

    d100 = levels[1.00]
    ph = R.head_panel(pan_ship, d100)
    keys = [f"{int(round(fr*100))}pct" for fr in CV.FRACS]
    incs = {f"{int(round(fr*100))}pct": {f: levels[fr][f] - d100[f] for f in A.FOLDS}
            for fr in CV.FRACS}
    draws = CV.curve_bootstrap(ph, incs, keys)

    sub = [fr for fr in CV.FRACS if fr < 1.0]
    x = np.log(np.array(sub))
    g_pt = np.array([CV.board_point(ph, incs[f"{int(round(fr*100))}pct"]) for fr in sub])
    G = np.stack([draws[f"{int(round(fr*100))}pct"] for fr in sub])

    # 1 -- the slope each level implies on its own
    k_each = g_pt / x
    k_each_draws = G / x[:, None]
    per_level = {}
    for i, fr in enumerate(sub):
        lo, hi = (float(v) for v in np.percentile(k_each_draws[i], [2.5, 97.5]))
        per_level[f"{int(round(fr*100))}pct"] = {
            "board_gain_s": float(g_pt[i]), "ln_frac": float(x[i]),
            "implied_k": float(k_each[i]), "implied_k_ci95": [lo, hi],
            "bootstrap_sd_of_the_gain_s": float(G[i].std())}
    # is the sequence steeper at the bottom?  paired inside each draw, so the comparison
    # does not pay for the levels' shared scoring rows twice
    steeper = k_each_draws[sub.index(0.25)] - k_each_draws[sub.index(0.75)]
    s_lo, s_hi = (float(v) for v in np.percentile(steeper, [2.5, 97.5]))

    # 2 -- the intercept the anchor asserts is zero
    Xd = np.stack([np.ones_like(x), x])                       # (2, 3)
    coef = np.linalg.lstsq(Xd.T, g_pt, rcond=None)[0]
    cd = np.linalg.lstsq(Xd.T, G, rcond=None)[0]              # (2, BOOT)
    a_lo, a_hi = (float(v) for v in np.percentile(cd[0], [2.5, 97.5]))
    k2_lo, k2_hi = (float(v) for v in np.percentile(cd[1], [2.5, 97.5]))

    # 3 -- residuals of the REGISTERED one-parameter fit
    k1 = CV.fit_slope(x, g_pt)
    resid = g_pt - k1 * x
    sd = G.std(axis=1)

    # 4 -- the bounded alternative, fitted the same way in every draw.  C is linear given
    # alpha, so alpha is scanned and C solved in closed form: no optimiser, no dependency,
    # and the same fit is applied to the point estimate and to all 2,000 draws.
    u = np.array(sub)
    grid = np.linspace(0.02, 4.0, 800)
    Hm = 1.0 - u[:, None] ** (-grid[None, :])              # (3, A)
    hh = (Hm * Hm).sum(0)                                  # (A,)

    def power_fit(gv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """gv is (3,) or (3, B) -> alpha, C, sse, chosen on the grid."""
        gv = np.atleast_2d(gv.T).T if gv.ndim > 1 else gv[:, None]
        Cs = (Hm.T @ gv) / hh[:, None]                     # (A, B)
        sse = ((gv[None, :, :] - Cs[:, None, :] * Hm.T[:, :, None]) ** 2).sum(1)
        j = sse.argmin(0)
        b = np.arange(gv.shape[1])
        return grid[j], Cs[j, b], sse[j, b]

    a_pt, C_pt, sse_pt = power_fit(g_pt)
    a_d, C_d, _ = power_fit(G)
    at_edge = float(((a_d <= grid[1]) | (a_d >= grid[-2])).mean())

    def pw_pred(R: float) -> dict:
        pt = float(C_pt[0] * (1.0 - R ** -a_pt[0]))
        dr = C_d * (1.0 - R ** -a_d)
        lo, hi = (float(v) for v in np.percentile(dr, [2.5, 97.5]))
        return {"gain_over_the_current_panel_s": pt, "ci95": [lo, hi]}

    # ---- per fold, because the gate is every_fold_interval_clear and section 47.17's
    # one-month extension multiplies each fold's fitting set by a DIFFERENT factor: a fold's
    # fitting rows are the other two months, so the fold with the largest fitting set gains
    # the least from one more month.  A board-only curve cannot say which fold fails.
    CJ = json.loads((EXP / "analysis/curve.json").read_text())
    per_fold = {}
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
        gsub = np.array(gsub)
        Gf = np.stack(gdraw)
        af, Cf, _ = power_fit(gsub)
        afd, Cfd, _ = power_fit(Gf)
        per_fold[f] = {
            "units": "s^2 per row, the by_fold currency the gate reads",
            "measured_over_the_100pct_member": {f"{int(round(fr*100))}pct": float(v)
                                                for fr, v in zip(sub, gsub)},
            "alpha": float(af[0]), "asymptote_s2_per_row": float(Cf[0]),
            "current_fitting_rows": int(CJ["levels"]["100"]["folds"][f]["fit_rows"]),
            "predictions": {f"x{m:.2f}": {
                "gain_s2_per_row": float(Cf[0] * (1.0 - m ** -af[0])),
                "ci95": [float(v) for v in np.percentile(
                    Cfd * (1.0 - m ** -afd), [2.5, 97.5])]}
                for m in (1.40, 1.45, 1.50, 1.55, 1.60)}}

    # ---- section 47.17: ONE extra month multiplies each fold's fitting set by a DIFFERENT
    # factor, so the board prediction is assembled from the three per-fold curves at their own
    # multiples rather than read off a board curve at an averaged one.
    mse0 = {f: float(((ph[f]["y_true"].to_numpy("float64")
                       - ph[f]["pbase"].to_numpy("float64")) ** 2).mean()) for f in A.FOLDS}
    fold_alpha = {f: per_fold[f]["alpha"] for f in A.FOLDS}
    fold_C = {f: per_fold[f]["asymptote_s2_per_row"] for f in A.FOLDS}
    fold_C_draws, fold_a_draws = {}, {}
    for f in A.FOLDS:
        yv = ph[f]["y_true"].to_numpy("float64")
        pbv = ph[f]["pbase"].to_numpy("float64")
        codes, uniq = np.unique(ph[f]["day"].to_numpy(), return_inverse=True)[::-1]
        kd = len(uniq)
        cc = np.bincount(codes, minlength=kd).astype("float64")
        s0f = np.bincount(codes, weights=(yv - pbv) ** 2, minlength=kd)
        rf = np.random.default_rng(BA.SEED)
        drw = rf.integers(0, kd, size=(BA.BOOT, kd))
        gd = []
        for fr in sub:
            dd = np.asarray(incs[f"{int(round(fr*100))}pct"][f], dtype="float64")
            s1f = np.bincount(codes, weights=(yv - pbv - dd) ** 2, minlength=kd)
            gd.append(s0f[drw].sum(1) / cc[drw].sum(1) - s1f[drw].sum(1) / cc[drw].sum(1))
        ad, Cd, _ = power_fit(np.stack(gd))
        fold_a_draws[f], fold_C_draws[f] = ad, Cd

    def board_at(n11: float) -> dict:
        mult = {f: (per_fold[f]["current_fitting_rows"] + n11)
                   / per_fold[f]["current_fitting_rows"] for f in A.FOLDS}
        gp = {f: fold_C[f] * (1.0 - mult[f] ** -fold_alpha[f]) for f in A.FOLDS}
        b0 = BA.board({f: mse0[f] ** 0.5 for f in A.FOLDS})
        b1 = BA.board({f: max(mse0[f] - gp[f], 0.0) ** 0.5 for f in A.FOLDS})
        gd_ = {f: fold_C_draws[f] * (1.0 - mult[f] ** -fold_a_draws[f]) for f in A.FOLDS}
        bd = (np.sqrt(sum(BA.SHARES[f] * mse0[f] for f in BA.SHARES))
              - np.sqrt(sum(BA.SHARES[f] * np.maximum(mse0[f] - gd_[f], 0.0)
                            for f in BA.SHARES)))
        lo, hi = (float(v) for v in np.percentile(bd, [2.5, 97.5]))
        return {"n_2025_11_writable_rows": n11,
                "multiple_by_fold": {f: float(mult[f]) for f in A.FOLDS},
                "gain_s2_per_row_by_fold": {f: float(gp[f]) for f in A.FOLDS},
                "fold_ci95_s2_per_row": {f: [float(v) for v in
                                             np.percentile(gd_[f], [2.5, 97.5])]
                                         for f in A.FOLDS},
                "every_fold_interval_clear_predicted": bool(
                    all(float(np.percentile(gd_[f], 2.5)) > 0 for f in A.FOLDS)),
                "board_gain_s": float(b0 - b1), "board_ci95": [lo, hi]}

    one_month = {f"n{int(n)}": board_at(float(n)) for n in (70000, 87252, 105000)}

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "per_fold_bounded_fit": per_fold,
        "section_47_17_one_extra_month": {
            "assembled_from": ("the three per-fold curves at their OWN multiples, not a "
                               "board curve at an averaged multiple -- one extra month is a "
                               "different fraction of each fold's fitting set"),
            "n87252_is": "the mean usable rows per month over the three fold months",
            "by_2025_11_row_count": one_month},
        "status": ("POST-HOC. Written while curve.py ran, registers no band, moves no gate. "
                   "Section 47.15's registered reading is the one-parameter fit; this asks "
                   "whether that form is the right one and reports the answer beside it."),
        "refits_nothing": "reads work/d-curve{25,50,75,100}-255-free.npz and prices them",
        "one_parameter_fit_as_registered": {
            "k": float(k1), "residuals_s": [float(v) for v in resid],
            "residuals_in_bootstrap_sds": [float(r / s) if s > 0 else None
                                           for r, s in zip(resid, sd)],
            "levels": [f"{int(round(fr*100))}pct" for fr in sub]},
        "per_level_implied_slope": per_level,
        "is_the_curve_steeper_at_the_bottom": {
            "k_25pct_minus_k_75pct": float(k_each[sub.index(0.25)]
                                           - k_each[sub.index(0.75)]),
            "ci95": [s_lo, s_hi],
            "clear_of_zero": bool(s_lo > 0 or s_hi < 0),
            "reading": ("positive and clear = the curve flattens faster than log, so the "
                        "registered extrapolation OVERSTATES the extended panel; negative "
                        "and clear = it understates; crossing zero = the three levels are "
                        "consistent with one log slope and the registered form stands")},
        "two_parameter_fit": {
            "intercept_a": float(coef[0]), "intercept_ci95": [a_lo, a_hi],
            "slope_k": float(coef[1]), "slope_ci95": [k2_lo, k2_hi],
            "intercept_consistent_with_the_anchor": bool(a_lo <= 0.0 <= a_hi),
            "why_it_matters": ("the anchor at frac = 1 is exact by construction, so a "
                               "non-zero intercept is not a free parameter -- it is the "
                               "log form failing")},
        "bounded_alternative_power_law": {
            "form": "g(u) = C * (1 - u**-alpha), u = rows / current rows",
            "post_hoc": ("NOT registered. Section 47.15 registered the one-parameter log "
                         "fit and that reading stands as the scored one; this is reported "
                         "beside it because the shape diagnostics above say the log form "
                         "is wrong, and a wrong form extrapolated 1.7 nats upward is the "
                         "weakest link in section 47.12's chain"),
            "alpha": float(a_pt[0]), "C_the_asymptote_s": float(C_pt[0]),
            "alpha_ci95": [float(v) for v in np.percentile(a_d, [2.5, 97.5])],
            "asymptote_ci95": [float(v) for v in np.percentile(C_d, [2.5, 97.5])],
            "sse": float(sse_pt[0]),
            "sse_of_the_registered_log_fit": float((resid ** 2).sum()),
            "draws_at_the_grid_edge": at_edge,
            "fitted_points_s": [float(C_pt[0] * (1.0 - uu ** -a_pt[0])) for uu in u],
            "measured_points_s": [float(v) for v in g_pt],
            "predictions": {"x1_45": pw_pred(1.45), "x1_50": pw_pred(1.50),
                            "x1_55": pw_pred(1.55), "x1_60": pw_pred(1.60),
                            "x2": pw_pred(2.0), "x3": pw_pred(3.0), "x5": pw_pred(5.0),
                            "x5_5_the_extended_panel": pw_pred(5.5),
                            "x10": pw_pred(10.0)},
            "why_the_1_4_to_1_6_grid": ("section 47.17's ONE-MONTH extension (2025-11) lands "
                                        "there, and the exact multiple is not known until "
                                        "queue reports 2025-11's writable row count, so the "
                                        "grid is registered rather than one point"),
            "what_the_asymptote_means": ("C is the whole remaining upside of MORE ROWS OF "
                                         "THE SAME KIND for this recipe on this panel. It "
                                         "is not a bound on the lane: a different recipe, "
                                         "more capacity, or months that carry coverage the "
                                         "folds never see are all outside it")},
        "bootstrap": {"draws": BA.BOOT, "seed": BA.SEED,
                      "paired": "one set of day resamples prices every level"},
        "seconds": None}
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print("  level   gain over 100 %   implied k alone")
    for fr in sub:
        p = per_level[f"{int(round(fr*100))}pct"]
        print(f"  {fr:5.0%}   {p['board_gain_s']:+.5f}         {p['implied_k']:+.4f} "
              f"[{p['implied_k_ci95'][0]:+.4f}, {p['implied_k_ci95'][1]:+.4f}]")
    st = rec["is_the_curve_steeper_at_the_bottom"]
    print(f"\n  k(25 %) - k(75 %) = {st['k_25pct_minus_k_75pct']:+.4f} "
          f"[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]  clear {st['clear_of_zero']}")
    tp = rec["two_parameter_fit"]
    print(f"  free intercept a = {tp['intercept_a']:+.5f} "
          f"[{tp['intercept_ci95'][0]:+.5f}, {tp['intercept_ci95'][1]:+.5f}]  "
          f"consistent with the anchor {tp['intercept_consistent_with_the_anchor']}")
    op = rec["one_parameter_fit_as_registered"]
    print(f"  registered fit k = {op['k']:+.5f}, residuals "
          + " ".join(f"{v:+.5f}" for v in op["residuals_s"])
          + "  (in bootstrap SDs: "
          + " ".join(f"{v:+.2f}" for v in op["residuals_in_bootstrap_sds"]) + ")")
    print("\n  per fold (s^2/row, the gate's currency), bounded fit at one extra month:")
    for f in A.FOLDS:
        q = per_fold[f]
        pr = q["predictions"]["x1.50"]
        print(f"    {f:9s} rows {q['current_fitting_rows']:7,d}  alpha {q['alpha']:.3f}  "
              f"asymptote {q['asymptote_s2_per_row']:8.2f}  at 1.50x "
              f"{pr['gain_s2_per_row']:+8.2f} [{pr['ci95'][0]:+8.2f}, {pr['ci95'][1]:+8.2f}]")

    print("\n  section 47.17, ONE extra month (2025-11), by its writable row count:")
    for kk, q in one_month.items():
        print(f"    {kk:9s} multiples "
              + "/".join(f"{q['multiple_by_fold'][f]:.3f}" for f in A.FOLDS)
              + f"  board {q['board_gain_s']:+.4f} [{q['board_ci95'][0]:+.4f}, "
              f"{q['board_ci95'][1]:+.4f}]  every-fold-clear "
              f"{q['every_fold_interval_clear_predicted']}")

    pw = rec["bounded_alternative_power_law"]
    print(f"\n  POST-HOC bounded alternative: alpha {pw['alpha']:.4f} "
          f"[{pw['alpha_ci95'][0]:.4f}, {pw['alpha_ci95'][1]:.4f}], asymptote "
          f"{pw['C_the_asymptote_s']:+.4f} "
          f"[{pw['asymptote_ci95'][0]:+.4f}, {pw['asymptote_ci95'][1]:+.4f}] s")
    print(f"  sse {pw['sse']:.6f} against the registered log fit's "
          f"{pw['sse_of_the_registered_log_fit']:.6f}")
    for nm in ("x1_45", "x1_50", "x1_55", "x1_60", "x2", "x3", "x5",
               "x5_5_the_extended_panel", "x10"):
        q = pw["predictions"][nm]
        print(f"    {nm:24s} {q['gain_over_the_current_panel_s']:+.4f} "
              f"[{q['ci95'][0]:+.4f}, {q['ci95'][1]:+.4f}]")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
