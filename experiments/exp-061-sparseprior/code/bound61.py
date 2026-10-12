"""exp-061-sparseprior: the exact hard sparse-key bound on V (SPEC 6).  In-sample, descriptive.

Runs only after the historical verdict, the prediction hashes and the loss receipts are frozen.
For every primary arm, (n*, lambda, c, beta-map) is chosen ON V ITSELF over the complete registered
grid, with exp-060's exact clipped SSE and both clip repairs.  At every grid point a free V airport
residual mean is removed before the SSE (the primary `adjusted_all_rows` kind); the unadjusted
`full` kind and the `adjusted_population_rows` variant come from the same sufficient statistics.
Levels: grid oracle (one beta everywhere), A (one beta per airport), B (one beta per airport and
family, the primary).

The paired three-day band uses the same joint date draws as the pricing, for ALL seventeen arms;
inside every draw the airport intercept is refitted and each arm's n*, lambda, c and airport-family
beta map are reselected on that draw's rows under the registered tie order.  It is a selected,
in-sample stability diagnostic, not a confidence interval and not an out-of-sample test.

    OMP_NUM_THREADS=1 .venv/bin/python experiments/exp-061-sparseprior/code/bound61.py jul
    OMP_NUM_THREADS=1 .venv/bin/python experiments/exp-061-sparseprior/code/bound61.py --combine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

GRIDS = {"B": E.grid_B(), "A": E.grid_A()}
PHI = {lev: E.monomials(GRIDS[lev]) for lev in GRIDS}
BAND_ARMS = E.ARMS
BAND_L = 3
BAND_LEVEL = "adjusted_all_rows|B"
#: (name, subtract a free airport intercept, take the mean over all V rows)
KINDS = (("full", False, True), ("adjusted_all_rows", True, True),
         ("adjusted_population_rows", True, False))
LEVELS = (("grid_oracle", "A", False), ("A", "A", True), ("B", "B", True))


def out_path(r: str | None) -> Path:
    return E.ANALYSIS / (f"bound61-{r}.json" if r else "bound61.json")


def select_from_curves(sse, rsum, n, per_airport: bool, adjust: bool):
    v = sse - (rsum ** 2) / np.maximum(n, 1.0)[:, None] if adjust else sse
    if per_airport:
        idx = v.argmin(1)
        return float(v[np.arange(len(v)), idx].sum()), idx
    j = int(v.sum(0).argmin())
    return float(v[:, j].sum()), np.full(len(v), j)


def one(r: str, overwrite: bool) -> None:
    out = E.out_path(out_path(r))
    draws_p = E.out_path(E.ART / f"bound61-draws-{r}.npz")
    grid_p = E.out_path(E.ART / f"bound61-grid-{r}.npz")
    NO.refuse_existing([out, draws_p, grid_p], overwrite)
    t0 = time.time()
    launch = E.hold_load(f"bound {r}")

    oos_p = E.ANALYSIS / "oos61.json"
    if not oos_p.exists():
        raise SystemExit("analysis/oos61.json is missing: the bound runs only after the V verdict")
    oos = E.load_json(oos_p)
    in_rec = E.load_json(E.ANALYSIS / f"inputs61-{r}.json")
    mk_rec = E.load_json(E.ANALYSIS / f"mask61-{r}.json")
    fit_rec = E.load_json(E.ANALYSIS / f"fit61-{r}.json")
    for rec_, key in ((in_rec, "npz"), (mk_rec, "npz"), (fit_rec, "pred_npz")):
        if E.sha_file(E.ROOT / rec_[key]) != rec_[f"{key}_sha256"]:
            raise SystemExit(f"{rec_[key]} is not the frozen file")
    if (oos["inputs"][r]["record_npz_sha256"] != in_rec["npz_sha256"]
            or oos["inputs"][r]["masks_npz_sha256"] != mk_rec["npz_sha256"]
            or oos["inputs"][r]["fit_pred_npz_sha256"] != fit_rec["pred_npz_sha256"]):
        raise SystemExit("the verdict was priced on other inputs, masks or corrections")
    for key in ("rows", "draws"):
        aa = oos["artifacts"][key]
        if E.sha_file(E.ROOT / aa["path"]) != aa["sha256"]:
            raise SystemExit(f"the frozen loss receipt {aa['path']} has changed")

    z = np.load(E.ROOT / in_rec["npz"])
    mk = np.load(E.ROOT / mk_rec["npz"])
    vocab = json.loads((E.ROOT / in_rec["vocab61_path"]).read_text())
    E.pin_months(in_rec)
    V, S = in_rec["months"]["V"], in_rec["months"]["S"]
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
    napt = nlev["apt"]
    off = E.fam_offsets(nlev)

    yS = E.dep_labels(S, z["ids_S"], "hard sparse-key bound: the S fit")
    yV = E.dep_labels(V, z["ids_V"], "hard sparse-key bound: scoring on V itself (in sample)")

    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
    p, P = z["p_V"], z["pop_V"]
    day = z["day_V"]
    ndates = int(day.max()) + 1
    apt = codes["V"]["apt"]
    fit_rows = z["pop_S"] & np.isfinite(yS) & (yS > 0)
    X = E.design(codes["S"], nlev, fit_rows)
    res_S = E.demean_by_airport(np.clip(yS[fit_rows] - z["p_S"][fit_rows], -E.CLIP, E.CLIP),
                                codes["S"]["apt"][fit_rows])
    XV = E.design(codes["V"], nlev)

    lu = {lam: splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
          for lam in E.LAMBDAS if np.isfinite(lam)}
    h0 = {lam: (XV @ lu[lam].solve(np.asarray(X.T @ res_S, dtype="float64"))) if np.isfinite(lam)
          else np.zeros(len(yV)) for lam in E.LAMBDAS}

    cellP = apt[P] * ndates + day[P]
    ncell = napt * ndates
    yP = yV[P]
    eo = yV[~P] - p[~P]
    cello = apt[~P] * ndates + day[~P]
    st_out = np.stack([np.bincount(cello, minlength=ncell),
                       np.bincount(cello, weights=eo, minlength=ncell),
                       np.bincount(cello, weights=eo * eo, minlength=ncell)], 1)
    sse_out_total = float(st_out[:, 2].sum())

    def by_airport(a2d):
        return a2d.reshape(napt, ndates, -1).sum(1)

    st_out_a = by_airport(st_out)
    reg_pick = {}
    for arm in E.ARMS:
        b = fit_rec["arms"][arm]["levels"]["B"]
        reg_pick[arm] = (int(b["n_star"]),
                         float("inf") if b["lam"] == "inf" else float(b["lam"]),
                         float("inf") if b["c"] == "inf" else float(b["c"]))

    def prior_at(arm, i_n, i_c):
        if arm in E.DEGENERATE_ARMS:
            return np.zeros(ncol)
        if arm in E.SHAM_ARMS:
            return mk["eff_sham"][E.SHAM_ARMS.index(arm), i_n, i_c]
        return mk[f"eff_{arm}"][i_n, i_c]

    KEYS = E.cell_order()
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "V": V, "S": S,
           "launch": launch, "code_sha256": E.code_shas(),
           "inputs": {"npz_sha256": in_rec["npz_sha256"], "masks_npz_sha256": mk_rec["npz_sha256"],
                      "fit_pred_npz_sha256": fit_rec["pred_npz_sha256"],
                      "oos_rows_sha256": oos["artifacts"]["rows"]["sha256"],
                      "oos_advance": oos["gate"]["advance"],
                      "matched_null_resolved": mk_rec["matched_null"]["resolved"]},
           "rows_all": int(len(yV)), "rows_P": int(P.sum()), "dates": ndates,
           "airports": [str(x) for x in vocab["apt"]],
           "grids": {"thresholds": list(E.THRESHOLDS), "registered_betas": list(E.BETAS),
                     "level_B_points": len(GRIDS["B"]), "level_A_points": len(GRIDS["A"]),
                     "lambdas": [E.jsonable(x) for x in E.LAMBDAS],
                     "c": [E.jsonable(x) for x in E.C_GRID], "cells": len(KEYS)},
           "clip_counts_summary": {}, "arms": {}, "checks": {}}
    check = {"points": 0, "max_abs_sse_difference": 0.0, "max_relative_difference": 0.0}

    # ---- the draws, shared by every arm --------------------------------------------------------
    dr = E.block_draws(ndates, BAND_L, E.draw_seed(r, BAND_L))
    W = np.stack([np.bincount(dr[d], minlength=ndates) for d in range(E.BOOT)]).astype("float64")
    O = np.tensordot(W, st_out.reshape(napt, ndates, 3), axes=([1], [1]))       # (BOOT, napt, 3)

    grid_maps, band, n_ref = {}, {}, None
    for arm in E.ARMS:
        cells = {}
        cc_max = {"mixed": 0, "always_clipped": 0}
        for i_n, ns in enumerate(E.THRESHOLDS):
            for i_c, c in enumerate(E.C_GRID):
                pv = prior_at(arm, i_n, i_c)
                for lam in E.LAMBDAS:
                    Sm = np.zeros((int(P.sum()), 3))
                    for i, f in enumerate(E.PRIOR_FAMS):
                        pf = np.zeros(ncol)
                        pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                        if np.any(pf):
                            th = pf.copy() if np.isinf(lam) else lu[lam].solve(lam * pf)
                            Sm[:, i] = (XV @ th)[P]
                    u = p[P] + h0[lam][P]
                    st, dB, _, cnt, mix = E.prepare_cells(yP, u, Sm, cellP, ncell, GRIDS["B"])
                    dA = (E.mixed_correction(yP[mix], u[mix], Sm[mix], GRIDS["A"], cellP[mix],
                                             ncell)[0] if cnt["mixed"] else None)
                    cells[(ns, lam, c)] = {"st": st, "dB": dB, "dA": dA}
                    cc_max["mixed"] = max(cc_max["mixed"], cnt["mixed"])
                    cc_max["always_clipped"] = max(cc_max["always_clipped"], cnt["always_clipped"])
                    if arm in ("true", "zero") and c in (E.C_GRID[0], E.C_GRID[-1]):
                        sse_a, _, _ = E.grid_curves(by_airport(st),
                                                    None if dA is None else by_airport(dA),
                                                    None, PHI["A"])
                        for j, b in enumerate(E.BETAS):
                            pred = np.maximum(E.MIN_PRED, u + Sm.sum(1) * b)
                            d_sse = float(np.sum((yP - pred) ** 2))
                            q = float(sse_a[:, j].sum())
                            check["points"] += 1
                            check["max_abs_sse_difference"] = max(check["max_abs_sse_difference"],
                                                                  abs(q - d_sse))
                            check["max_relative_difference"] = max(
                                check["max_relative_difference"], abs(q - d_sse) / max(d_sse, 1.0))
        rec["clip_counts_summary"][arm] = cc_max

        # ---- the point bounds of this arm ------------------------------------------------------
        a_rec = {}
        for kind, adjust, use_out in KINDS:
            for lev, gkey, per_a in LEVELS:
                totals, idxs = {}, {}
                for key, cc in cells.items():
                    sse, rsum, n = E.grid_curves(
                        by_airport(cc["st"]),
                        None if cc["d" + gkey] is None else by_airport(cc["d" + gkey]),
                        None, PHI[gkey], st_out=st_out_a if use_out else None)
                    tot, idx = select_from_curves(sse, rsum, n, per_a, adjust)
                    totals[key] = tot + (0.0 if use_out else sse_out_total)
                    idxs[key] = idx
                key = E.pick_cell3(totals)
                a_rec[f"{kind}|{lev}"] = {
                    "n_star": key[0], "lam": E.jsonable(key[1]), "c": E.jsonable(key[2]),
                    "cell": E.cell_name(key), "sse": totals[key],
                    "sse_at_the_registered_pick": totals[reg_pick[arm]],
                    "betas_by_airport": {str(vocab["apt"][aa]):
                                         [float(x) for x in GRIDS[gkey][idxs[key][aa]]]
                                         for aa in range(napt)},
                    "sse_by_cell": {E.cell_name(kk): v for kk, v in totals.items()}}
                grid_maps[(arm, kind, lev)] = np.stack(
                    [idxs[kk].astype("int32") for kk in KEYS]).astype("int32")
        rec["arms"][arm] = a_rec

        # ---- this arm's three-day band ----------------------------------------------------------
        best = np.full(E.BOOT, np.inf)
        best_key = np.zeros(E.BOOT, dtype="int16")
        betas_d = np.zeros((E.BOOT, napt, 3))
        n_d = None
        for ki, key in enumerate(KEYS):
            cc = cells[key]
            M_sse = E.sse_coeffs(cc["st"]).reshape(napt, ndates, 10)
            M_r = E.res_coeffs(cc["st"]).reshape(napt, ndates, 10)
            nn = cc["st"][:, 0].reshape(napt, ndates)
            A_sse = np.tensordot(W, M_sse, axes=([1], [1]))
            A_r = np.tensordot(W, M_r, axes=([1], [1]))
            sse = (A_sse.reshape(-1, 10) @ PHI["B"]).reshape(E.BOOT, napt, -1)
            rs = (A_r.reshape(-1, 10) @ PHI["B"]).reshape(E.BOOT, napt, -1)
            if cc["dB"] is not None:
                sse += np.tensordot(W, cc["dB"].reshape(napt, ndates, -1), axes=([1], [1]))
            sse += O[:, :, 2:3]
            rs += O[:, :, 1:2]
            nt = (W @ nn.T) + O[:, :, 0]
            v = sse - (rs ** 2) / np.maximum(nt, 1.0)[:, :, None]
            idx = v.argmin(2)
            tot = np.take_along_axis(v, idx[:, :, None], 2)[:, :, 0].sum(1)
            take = tot < best
            best = np.where(take, tot, best)
            best_key = np.where(take, ki, best_key)
            betas_d[take] = GRIDS["B"][idx[take]]
            if n_d is None:
                n_d = nt.sum(1)
            del sse, rs, v, A_sse, A_r
        if n_ref is None:
            n_ref = n_d
        elif not np.array_equal(n_ref, n_d):
            raise SystemExit("the draws hold different row counts for two arms")
        band[arm] = {"sse": best, "mse": best / n_d, "key": best_key.astype("int16"),
                     "betas": betas_d}
        print(f"  {r} {arm:13s} adj|B {a_rec['adjusted_all_rows|B']['sse']:.6e}  band mean "
              f"{band[arm]['mse'].mean():.3f} s2/row ({time.time() - t0:.0f}s)", flush=True)
        del cells

    rec["checks"]["direct_recomputation_on_the_one_beta_grid"] = check
    if check["max_relative_difference"] > 1e-10:
        raise SystemExit("the clipped grid machinery does not reproduce the served prediction: "
                         f"relative {check['max_relative_difference']:.3e}")
    rec["checks"]["level_A_over_grid_oracle_sse"] = {
        arm: {k2: rec["arms"][arm][f"{k2}|grid_oracle"]["sse"] - rec["arms"][arm][f"{k2}|A"]["sse"]
              for k2, _, _ in KINDS} for arm in E.ARMS}
    rec["band_level"] = BAND_LEVEL
    rec["band_keys"] = [E.cell_name(k) for k in KEYS]

    NO.refuse_existing([out, draws_p, grid_p], overwrite)
    draws_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(grid_p, cell_order=np.array([E.cell_name(k) for k in KEYS]),
             airports=np.array([str(x) for x in vocab["apt"]]),
             beta_grid_A=GRIDS["A"], beta_grid_B=GRIDS["B"],
             **{f"{arm}|{kind}|{lev}": m for (arm, kind, lev), m in grid_maps.items()})
    np.savez(draws_p, draw_dates=dr, n=n_ref,
             **{f"{arm}_{w}": band[arm][w] for arm in BAND_ARMS
                for w in ("sse", "mse", "key", "betas")})
    rec["grid_artifact"] = {
        "path": str(grid_p.relative_to(E.ROOT)), "sha256": E.sha_file(grid_p),
        "what": ("the per-airport grid index of every arm, kind, level and (n*, lambda, c) cell; "
                 "index into beta_grid_A for the grid_oracle and A levels and beta_grid_B for "
                 "level B; axis order (cell, airport), cells in cell_order"),
        "cells": len(KEYS), "entries": int(sum(m.size for m in grid_maps.values()))}
    rec["band_artifact"] = {"path": str(draws_p.relative_to(E.ROOT)), "sha256": E.sha_file(draws_p),
                            "block_length_days": BAND_L, "draws": E.BOOT,
                            "draw_seed": E.draw_seed(r, BAND_L)}
    rec["band_summary"] = {arm: {"mean_adjusted_mse_s2": float(band[arm]["mse"].mean()),
                                 "selected_cells": {rec["band_keys"][i]: int(n) for i, n in
                                                    enumerate(np.bincount(band[arm]["key"].astype("int64"),
                                                                          minlength=len(KEYS))) if n}}
                           for arm in BAND_ARMS}
    rec["label_reads"] = E.LABEL_READS
    rec["opened_paths"] = list(E.OPENED)
    rec["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    zb = rec["arms"]["zero"]
    for kind, _, _ in KINDS:
        tb = rec["arms"]["true"][f"{kind}|B"]
        print(f"  {r} {kind:26s} level B: true "
              f"{(zb[f'{kind}|B']['sse'] - tb['sse']) / rec['rows_all']:+8.4f} s2/row over zero "
              f"({tb['cell']})", flush=True)
    print(f"written {out.name} ({rec['seconds']:.0f}s; direct check {check['points']} points, "
          f"max relative {check['max_relative_difference']:.2e})", flush=True)


def combine(overwrite: bool) -> None:
    out = E.out_path(out_path(None))
    NO.refuse_existing([out], overwrite)
    d = {r: E.load_json(out_path(r)) for r in E.ORDER}
    bands = {}
    for r in E.ORDER:
        aa = d[r]["band_artifact"]
        if E.sha_file(E.ROOT / aa["path"]) != aa["sha256"]:
            raise SystemExit(f"{aa['path']} is not the band artifact its record names")
        bands[r] = np.load(E.ROOT / aa["path"])

    def board(a: dict, b: dict) -> float:
        ma = sum(s * a[r] / d[r]["rows_all"] for r, s in E.BOARD.items())
        mb = sum(s * b[r] / d[r]["rows_all"] for r, s in E.BOARD.items())
        return float(np.sqrt(ma) - np.sqrt(mb))

    res = {"what": "hard sparse-key exact bound on V (post-hoc, in-sample, descriptive; not a test)",
           "rotations": {r: {"V": d[r]["V"], "rows_all": d[r]["rows_all"], "rows_P": d[r]["rows_P"],
                             "checks": d[r]["checks"]["direct_recomputation_on_the_one_beta_grid"],
                             "clip_mixed_max": max(v["mixed"] for v in
                                                   d[r]["clip_counts_summary"].values()),
                             "always_clipped_max": max(v["always_clipped"] for v in
                                                       d[r]["clip_counts_summary"].values())}
                         for r in E.ORDER},
           "levels": {}}
    for kind, _, _ in KINDS:
        for lev, _, _ in LEVELS:
            key = f"{kind}|{lev}"
            zero = {r: d[r]["arms"]["zero"][key]["sse"] for r in E.ORDER}
            zreg = {r: d[r]["arms"]["zero"][key]["sse_at_the_registered_pick"] for r in E.ORDER}
            gain = {arm: board(zero, {r: d[r]["arms"][arm][key]["sse"] for r in E.ORDER})
                    for arm in E.ARMS}
            folds = {arm: {r: (zero[r] - d[r]["arms"][arm][key]["sse"]) / d[r]["rows_all"]
                           for r in E.ORDER} for arm in E.ARMS}
            v = np.array([gain[s] for s in E.SHAM_ARMS])
            res["levels"][key] = {
                "reference": "the zero arm's lambda-oracle at the same level and bound kind",
                "zero_lambda_oracle_over_zero_registered_pick_board_s": board(zreg, zero),
                "board_gain_s": gain, "fold_gain_s2_per_row": folds,
                "true_rank_of_11": int(1 + (v > gain["true"]).sum()),
                "sham_mean_s": float(v.mean()), "sham_max_s": float(v.max()),
                "true_beats_every_sham": bool(gain["true"] > v.max()),
                "picks": {r: {arm: {kk: d[r]["arms"][arm][key][kk] for kk in ("n_star", "lam", "c")}
                              for arm in ("true", "zero")} for r in E.ORDER}}

    rmse = {arm: np.sqrt(sum(s * bands[r][f"{arm}_mse"] for r, s in E.BOARD.items()))
            for arm in BAND_ARMS}
    mat = np.stack([rmse[s] for s in E.SHAM_ARMS])
    g = mat.min(0) - rmse["true"]
    key = BAND_LEVEL
    pt_rmse = {arm: float(np.sqrt(sum(s * d[r]["arms"][arm][key]["sse"] / d[r]["rows_all"]
                                      for r, s in E.BOARD.items()))) for arm in E.ARMS}
    best_s = min(E.SHAM_ARMS, key=lambda s: pt_rmse[s])
    res["winner_adjusted_three_day_band"] = {
        "level": key, "block_length_days": BAND_L, "draws": E.BOOT,
        "definition": ("min over the ten shams of the recombined board RMSE minus true's, with the "
                       "free airport intercept refitted and n*, lambda, c and the airport-family "
                       "beta map reselected inside every draw"),
        "point_s": pt_rmse[best_s] - pt_rmse["true"],
        "winning_sham_on_the_original_rows": best_s,
        "board_rmse_on_the_original_rows_s": pt_rmse,
        "ci_block_3d": E.ci(g),
        "gain_mean_s": float(g.mean()), "gain_positive_share": float((g > 0).mean()),
        "winning_sham_counts": {s: int(n) for s, n in
                                zip(E.SHAM_ARMS, np.bincount(mat.argmin(0),
                                                             minlength=len(E.SHAM_ARMS)))},
        "what_it_is_not": ("a selected, in-sample empirical stability diagnostic; not a confidence "
                           "interval, not an out-of-sample test, not a candidate rule")}
    w = res["winner_adjusted_three_day_band"]
    lb = res["levels"][key]
    res["reading_inputs"] = {
        "adjusted_level_B_true_beats_every_sham": lb["true_beats_every_sham"],
        "adjusted_level_B_true_board_gain_s": lb["board_gain_s"]["true"],
        "adjusted_level_B_sham_max_s": lb["sham_max_s"],
        "winner_adjusted_point_s": w["point_s"],
        "winner_adjusted_3d_lower_endpoint_s": w["ci_block_3d"][0],
        "bound_condition_met": bool(lb["true_beats_every_sham"] and w["point_s"] > 0
                                    and w["ci_block_3d"][0] > 0)}
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    for k2, x in res["levels"].items():
        print(f"{k2:38s} true {x['board_gain_s']['true']:+.4f} s, rank {x['true_rank_of_11']} of 11, "
              f"sham mean {x['sham_mean_s']:+.4f}, max {x['sham_max_s']:+.4f}")
    print(f"winner-adjusted ({key}, 3-day): point {w['point_s']:+.4f} s, band {w['ci_block_3d']}, "
          f"bound condition met {res['reading_inputs']['bound_condition_met']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", nargs="?", choices=E.ORDER)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.combine == (a.rotation is not None):
        raise SystemExit("give one rotation, or --combine")
    combine(a.overwrite) if a.combine else one(a.rotation, a.overwrite)
