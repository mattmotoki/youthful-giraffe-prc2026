"""exp-060-coldprior: the exact cold bound, with every registered arm in the band and the full grids.

This is `code/bound60_r1.py` (itself `bound60.py` with the registered tie rule in the draw loop)
with the two repairs review round 2 asked for, made after the last allowed round:

  * finding 1 - the three-day band now reselects and saves EVERY registered arm in every draw,
    not only `true`, the ten shams and `zero`.  The assignment says "use the same joint date draws
    for all arms" and "save every draw's selected settings, arm RMSEs, winning sham and gain".
  * finding 2 - the FULL bound grids are now saved: for every arm, kind, level and (lambda, c)
    cell, the per-airport grid index (and so the beta map), beside the totals the record already
    held.  They go to `artifacts/exp-060/bound-r2-grid-<r>.npz` as int16, with the axis order
    recorded in the JSON.

Outputs carry an `-r2` suffix, so nothing earlier is touched and every value can be compared.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-060-coldprior/code/bound60_r2.py jul
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
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

GRIDS = {"B": E.grid_B(), "A": E.grid_A()}
PHI = {lev: E.monomials(GRIDS[lev]) for lev in GRIDS}
#: the arms whose three-day band is resampled: true, the ten shams, and zero as the reference
#: review round 2 finding 1: every registered arm is in the band
BAND_ARMS = E.ARMS
BAND_L = 3
BAND_LEVEL = "adjusted_all_rows|B"
#: kinds of bound: (name, subtract a free airport intercept, take the mean over all V rows)
KINDS = (("full", False, True), ("adjusted_all_rows", True, True),
         ("adjusted_population_rows", True, False))
LEVELS = (("grid_oracle", "A", False), ("A", "A", True), ("B", "B", True))
MAX_STORE_BYTES = 3_000_000_000


def out_path(r: str | None) -> Path:
    return E.ANALYSIS / (f"bound-r2-{r}.json" if r else "bound-r2.json")


def select_from_curves(sse, rsum, n, per_airport: bool, adjust: bool):
    """(total, per-airport grid index) at one (lambda, c) cell.  `sse` and `rsum` are (napt, K),
    `n` is (napt,).  With `adjust` each grid point's free airport intercept is removed:
    SSE_adj = sum r^2 - (sum r)^2 / n.  With `per_airport` the argmin is taken inside each airport
    (levels A and B); otherwise one grid point is shared by every airport (the grid oracle)."""
    v = sse - (rsum ** 2) / np.maximum(n, 1.0)[:, None] if adjust else sse
    if per_airport:
        idx = v.argmin(1)
        return float(v[np.arange(len(v)), idx].sum()), idx
    j = int(v.sum(0).argmin())
    return float(v[:, j].sum()), np.full(len(v), j)


def one(r: str, overwrite: bool) -> None:
    out = out_path(r)
    draws_p = E.ART / f"bound-r2-draws-{r}.npz"
    grid_p = E.ART / f"bound-r2-grid-{r}.npz"
    NO.refuse_existing([out, draws_p, grid_p], overwrite)
    t0 = time.time()
    launch = E.hold_load(f"bound {r}")

    oos_p = E.ANALYSIS / "oos.json"
    if not oos_p.exists():
        raise SystemExit("analysis/oos.json is missing: the bound runs only after the V verdict")
    oos = E.load_json(oos_p)
    in_rec = E.load_json(E.ANALYSIS / f"inputs60-{r}.json")
    fit_rec = E.load_json(E.ANALYSIS / f"fit60-{r}.json")
    for rec_, key in ((in_rec, "npz"), (fit_rec, "pred_npz")):
        if E.sha_file(E.ROOT / rec_[key]) != rec_[f"{key}_sha256"]:
            raise SystemExit(f"{rec_[key]} is not the frozen file")
    if (oos["inputs"][r]["record_npz_sha256"] != in_rec["npz_sha256"]
            or oos["inputs"][r]["fit_pred_npz_sha256"] != fit_rec["pred_npz_sha256"]):
        raise SystemExit("the verdict was priced on other inputs or corrections")
    for key in ("rows", "draws"):
        aa = oos["artifacts"][key]
        if E.sha_file(E.ROOT / aa["path"]) != aa["sha256"]:
            raise SystemExit(f"the frozen loss receipt {aa['path']} has changed")

    z = np.load(E.ROOT / in_rec["npz"])
    vocab = json.loads((E.ROOT / in_rec["vocab_path"]).read_text())
    V, S, _, _ = E.ROT[r]
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
    napt = nlev["apt"]
    off = E.fam_offsets(nlev)

    yS = E.dep_labels(S, z["ids_S"], "cold bound: the S fit")
    yV = E.dep_labels(V, z["ids_V"], "cold bound: scoring on V itself (in-sample, descriptive)")

    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
    p, P = z["p_V"], z["pop_V"]
    day = z["day_V"]
    ndates = int(day.max()) + 1
    apt = codes["V"]["apt"]
    fit_rows = z["pop_S"] & np.isfinite(yS) & (yS > 0)
    X = E.design(codes["S"], nlev, fit_rows)
    res = E.demean_by_airport(np.clip(yS[fit_rows] - z["p_S"][fit_rows], -E.CLIP, E.CLIP),
                              codes["S"]["apt"][fit_rows])
    XV = E.design(codes["V"], nlev)
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}

    lu = {lam: splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
          for lam in E.LAMBDAS if np.isfinite(lam)}
    h0 = {lam: (XV @ lu[lam].solve(np.asarray(X.T @ res, dtype="float64"))) if np.isfinite(lam)
          else np.zeros(len(yV)) for lam in E.LAMBDAS}

    cellP = apt[P] * ndates + day[P]
    ncell = napt * ndates
    yP = yV[P]
    eo = yV[~P] - p[~P]                       # rows the correction never moves: a cell constant
    cello = apt[~P] * ndates + day[~P]
    st_out = np.stack([np.bincount(cello, minlength=ncell),
                       np.bincount(cello, weights=eo, minlength=ncell),
                       np.bincount(cello, weights=eo * eo, minlength=ncell)], 1)
    sse_out_total = float(st_out[:, 2].sum())

    def by_airport(a2d):
        return a2d.reshape(napt, ndates, -1).sum(1)

    st_out_a = by_airport(st_out)
    reg_pick = {arm: (float("inf") if fit_rec["arms"][arm]["levels"]["B"]["lam"] == "inf"
                      else float(fit_rec["arms"][arm]["levels"]["B"]["lam"]),
                      float("inf") if fit_rec["arms"][arm]["levels"]["B"]["c"] == "inf"
                      else float(fit_rec["arms"][arm]["levels"]["B"]["c"])) for arm in E.ARMS}

    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "V": V, "S": S,
           "launch": launch, "code_sha256": E.code_shas(),
           "inputs": {"npz_sha256": in_rec["npz_sha256"],
                      "cold_table_sha256": in_rec["cold_table"]["sha256"],
                      "fit_pred_npz_sha256": fit_rec["pred_npz_sha256"],
                      "oos_rows_sha256": oos["artifacts"]["rows"]["sha256"],
                      "oos_advance": oos["gate"]["advance"]},
           "rows_all": int(len(yV)), "rows_P": int(P.sum()), "dates": ndates,
           "airports": [str(x) for x in vocab["apt"]],
           "grids": {"registered_betas": list(E.BETAS), "level_B_points": len(GRIDS["B"]),
                     "level_A_points": len(GRIDS["A"]),
                     "lambdas": [x if np.isfinite(x) else "inf" for x in E.LAMBDAS],
                     "c": [x if np.isfinite(x) else "inf" for x in E.C_GRID]},
           "clip_counts": {}, "arms": {}, "checks": {}}
    check = {"points": 0, "max_abs_sse_difference": 0.0, "max_relative_difference": 0.0}

    store, store_bytes = {}, 0
    grid_maps = {}                       # (arm, kind, level) -> (cell, airport) grid index
    cell_order = [(lam, c) for c in E.C_GRID for lam in E.LAMBDAS]
    for arm in E.ARMS:
        pv0 = (np.zeros(ncol) if arm in ("zero", "literal_sham")
               else E.prior_vector({f: z[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS}))
        cells = {}
        for c in E.C_GRID:
            pv = pv0 * E.support_weights(nA, c)
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
                dA = (E.mixed_correction(yP[mix], u[mix], Sm[mix], GRIDS["A"], cellP[mix], ncell)[0]
                      if cnt["mixed"] else None)
                cells[(lam, c)] = {"st": st, "dB": dB, "dA": dA}
                rec["clip_counts"][f"{arm}|c={c}|lam={lam}"] = cnt
                # the independent exactness check: the served prediction vector at every one-beta
                # grid point, against the airport-summed level-A curve of the same cell
                if arm in ("true", "zero") and c in (E.C_GRID[0], E.C_GRID[-1]):
                    sse_a, _, _ = E.grid_curves(by_airport(st), None if dA is None else by_airport(dA),
                                                None, PHI["A"])
                    for j, b in enumerate(E.BETAS):
                        pred = np.maximum(E.MIN_PRED, u + Sm.sum(1) * b)
                        d_sse = float(np.sum((yP - pred) ** 2))
                        q = float(sse_a[:, j].sum())
                        check["points"] += 1
                        check["max_abs_sse_difference"] = max(check["max_abs_sse_difference"],
                                                              abs(q - d_sse))
                        check["max_relative_difference"] = max(check["max_relative_difference"],
                                                               abs(q - d_sse) / max(d_sse, 1.0))
        keep = arm in BAND_ARMS
        if keep:
            store[arm] = cells
            store_bytes += sum(v["st"].nbytes + (0 if v["dB"] is None else v["dB"].nbytes)
                               + (0 if v["dA"] is None else v["dA"].nbytes) for v in cells.values())
            if store_bytes > MAX_STORE_BYTES:
                raise SystemExit(f"the band would hold {store_bytes / 1e9:.1f} GB of exact clip "
                                 "corrections: stop and report the measured mixed-row counts")
        # ---- the point bounds of this arm ---------------------------------------------------
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
                lam, c = E.pick_cell(totals)
                a_rec[f"{kind}|{lev}"] = {
                    "lam": "inf" if np.isinf(lam) else lam, "c": "inf" if np.isinf(c) else c,
                    "sse": totals[(lam, c)],
                    "sse_at_the_registered_pick": totals[reg_pick[arm]],
                    "betas_by_airport": {str(vocab["apt"][aa]):
                                         [float(x) for x in GRIDS[gkey][idxs[(lam, c)][aa]]]
                                         for aa in range(napt)},
                    "sse_by_cell": {f"lam={kk[0]},c={kk[1]}": v for kk, v in totals.items()}}
                grid_maps[(arm, kind, lev)] = np.stack(
                    [idxs[kk].astype("int32") for kk in cell_order]).astype("int32")
        rec["arms"][arm] = a_rec
        if not keep:
            del cells
        print(f"  {r} {arm:13s} full|B {a_rec['full|B']['sse']:.6e}  "
              f"adj|B {a_rec['adjusted_all_rows|B']['sse']:.6e} ({time.time() - t0:.0f}s)", flush=True)

    rec["checks"]["direct_recomputation_on_the_one_beta_grid"] = check
    if check["max_relative_difference"] > 1e-10:
        raise SystemExit("the clipped grid machinery does not reproduce the served prediction: "
                         f"relative {check['max_relative_difference']:.3e}")
    # level A with one beta for all airports IS the grid-oracle level by construction; the check
    # above is what makes both exact, and this records the gap the shared argmin costs
    rec["checks"]["level_A_over_grid_oracle_sse"] = {
        arm: {k2: rec["arms"][arm][f"{k2}|grid_oracle"]["sse"] - rec["arms"][arm][f"{k2}|A"]["sse"]
              for k2, _, _ in KINDS} for arm in E.ARMS}
    rec["store_bytes"] = int(store_bytes)

    # ---- the paired three-day band (SPEC 6.3) --------------------------------------------------
    dr = E.block_draws(ndates, BAND_L, E.draw_seed(r, BAND_L))
    W = np.stack([np.bincount(dr[d], minlength=ndates) for d in range(E.BOOT)]).astype("float64")
    O = np.tensordot(W, st_out.reshape(napt, ndates, 3), axes=([1], [1]))       # (BOOT, napt, 3)
    # SPEC 3.3's tie rule, applied to the draw loop: on an exact tie keep the larger
    # lambda, then the larger c, so the cells are visited in that preference order and
    # a strict improvement is the only thing that can replace an earlier cell
    keys = sorted(store["true"], key=lambda k: (-k[0], -k[1]))
    band = {}
    for arm in BAND_ARMS:
        best = np.full(E.BOOT, np.inf)
        best_key = np.zeros(E.BOOT, dtype="int16")
        betas_d = np.zeros((E.BOOT, napt, 3))
        n_d = None
        for ki, key in enumerate(keys):
            cc = store[arm][key]
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
            del sse, rs, v
        band[arm] = {"sse": best, "mse": best / n_d, "key": best_key.astype("int16"),
                     "betas": betas_d, "n": n_d}
        print(f"  {r} band {arm:10s} mean adjusted MSE {band[arm]['mse'].mean():.2f} s2/row "
              f"({time.time() - t0:.0f}s)", flush=True)
    rec["band_level"] = BAND_LEVEL
    rec["band_keys"] = [f"lam={k[0]},c={k[1]}" for k in keys]

    NO.refuse_existing([out, draws_p, grid_p], overwrite)
    draws_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(grid_p, cell_order=np.array([f"lam={a},c={b}" for a, b in cell_order]),
             airports=np.array([str(x) for x in vocab["apt"]]),
             beta_grid_A=GRIDS["A"], beta_grid_B=GRIDS["B"],
             **{f"{arm}|{kind}|{lev}": m for (arm, kind, lev), m in grid_maps.items()})
    np.savez(draws_p, draw_dates=dr,
             **{f"{arm}_{w}": band[arm][w] for arm in BAND_ARMS
                for w in ("sse", "mse", "key", "betas", "n")})
    rec["grid_artifact"] = {
        "path": str(grid_p.relative_to(E.ROOT)), "sha256": E.sha_file(grid_p),
        "what": ("the per-airport grid index of every arm, kind, level and (lambda, c) cell; "
                 "index into beta_grid_A for the grid_oracle and A levels and beta_grid_B for "
                 "level B; axis order (cell, airport), cells in cell_order"),
        "cells": len(cell_order), "entries": int(sum(m.size for m in grid_maps.values()))}
    rec["band_artifact"] = {"path": str(draws_p.relative_to(E.ROOT)), "sha256": E.sha_file(draws_p),
                            "block_length_days": BAND_L, "draws": E.BOOT,
                            "draw_seed": E.draw_seed(r, BAND_L)}
    rec["band_summary"] = {arm: {"mean_adjusted_mse_s2": float(band[arm]["mse"].mean()),
                                 "selected_cells": {rec["band_keys"][i]: int(n) for i, n in
                                                    enumerate(np.bincount(band[arm]["key"].astype("int64"),
                                                                          minlength=len(keys))) if n}}
                           for arm in BAND_ARMS}
    rec["label_reads"] = E.LABEL_READS
    rec["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    zb = rec["arms"]["zero"]
    for kind, _, _ in KINDS:
        tb = rec["arms"]["true"][f"{kind}|B"]
        print(f"  {r} {kind:26s} level B: true {(zb[f'{kind}|B']['sse'] - tb['sse']) / rec['rows_all']:+8.3f} "
              f"s2/row over zero (lam {tb['lam']}, c {tb['c']})", flush=True)
    print(f"written {out.name} ({rec['seconds']:.0f}s; direct check {check['points']} points, "
          f"max relative {check['max_relative_difference']:.2e})", flush=True)


def combine(overwrite: bool) -> None:
    out = out_path(None)
    NO.refuse_existing([out], overwrite)
    d = {r: E.load_json(out_path(r)) for r in E.ORDER}
    bands = {r: np.load(E.ROOT / d[r]["band_artifact"]["path"]) for r in E.ORDER}

    def board(a: dict, b: dict) -> float:
        ma = sum(s * a[r] / d[r]["rows_all"] for r, s in E.BOARD.items())
        mb = sum(s * b[r] / d[r]["rows_all"] for r, s in E.BOARD.items())
        return float(np.sqrt(ma) - np.sqrt(mb))

    res = {"what": "cold-table exact bound on V (post-hoc, in-sample, descriptive; not a test)",
           "rotations": {r: {"V": d[r]["V"], "rows_all": d[r]["rows_all"], "rows_P": d[r]["rows_P"],
                             "checks": d[r]["checks"]["direct_recomputation_on_the_one_beta_grid"],
                             "clip_rows_max": max(v["mixed"] for v in d[r]["clip_counts"].values()),
                             "always_clipped_max": max(v["always_clipped"]
                                                       for v in d[r]["clip_counts"].values())}
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
                "picks": {r: {arm: {kk: d[r]["arms"][arm][key][kk] for kk in ("lam", "c")}
                              for arm in ("true", "zero")} for r in E.ORDER}}

    # ---- the winner-adjusted three-day band on the airport-adjusted level-B bound -------------
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
                       "free airport intercept refitted and lambda, c and the airport-family beta "
                       "map reselected inside every draw"),
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
