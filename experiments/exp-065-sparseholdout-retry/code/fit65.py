"""exp-065-sparseholdout-retry: fit, select and freeze the hard sparse-key corrections of one rotation
(SPEC 3).

Reads this experiment's frozen `inputs65` projection and `masks65` effective priors (both
hash-checked), then the departure labels of S and I only.  For every arm (zero, the literal sham,
true/const/gmean/donor/shuffle and the ten NEW primary matched shams) it evaluates the exact
clipped pooled squared error of BOTH inner directions over the registered grid --- four hard
thresholds n*, six support constants c (inf is weight 1), nine lambdas and, per airport, the full
7^3 grid of (beta_stand, beta_runway, beta_operator) --- then

  * stage 1: each airport takes the grid point with the smallest summed SSE (a tie takes the
    smaller beta in stand, runway, operator order);
  * stage 2: (n*, lambda, c) takes the smallest total over all airports and both directions; a tie
    takes the SMALLER n* first, then the larger lambda, then the larger c;

freezes that map, fits the final correction on S and writes the correction for V.  Level A is
selected the same way and is descriptive.  V's labels are not read here.  One rotation per process.

    OMP_NUM_THREADS=1 .venv/bin/python experiments/exp-065-sparseholdout-retry/code/fit65.py jul
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e65 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

LEVELS = ("B", "A")
GRIDS = {"B": E.grid_B(), "A": E.grid_A()}
PHI = {lev: E.monomials(GRIDS[lev]) for lev in LEVELS}
DIRS = (("S", "I"), ("I", "S"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", choices=E.ORDER)
    a = ap.parse_args()
    r = a.rotation
    out_npz = E.out_path(E.WORK / f"pred65-{r}.npz")
    out_grid = E.out_path(E.WORK / f"grid65-{r}.npz")
    out_par = E.out_path(E.ART / f"dirloss65-{r}.parquet")
    out_rec = E.out_path(E.ANALYSIS / f"fit65-{r}.json")
    NO.refuse_existing([out_npz, out_grid, out_par, out_rec], False)
    t0 = time.time()
    launch = E.hold_load(f"fit {r}")

    in_rec = E.load_json(E.ANALYSIS / f"inputs65-{r}.json")
    mk_rec = E.load_json(E.ANALYSIS / f"mask65-{r}.json")
    for rec_, key in ((in_rec, "npz"), (mk_rec, "npz")):
        if E.sha_file(E.ROOT / rec_[key]) != rec_[f"{key}_sha256"]:
            raise SystemExit(f"{rec_[key]} is not the frozen file")
    if mk_rec["inputs"]["npz_sha256"] != in_rec["npz_sha256"]:
        raise SystemExit("the masks were built on another projection")
    z = np.load(E.ROOT / in_rec["npz"])
    mk = np.load(E.ROOT / mk_rec["npz"])
    vocab = json.loads((E.ROOT / in_rec["vocab65_path"]).read_text())
    E.pin_months(in_rec)
    V, S, I = (in_rec["months"][k] for k in ("V", "S", "I"))
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
    napt = nlev["apt"]
    off = E.fam_offsets(nlev)
    col_apt = E.column_airport(vocab)

    # ---- the only label reads of this job: S and I ------------------------------------------
    y = {"S": E.dep_labels(S, z["ids_S"], "hard sparse-key correction fit and inner selection"),
         "I": E.dep_labels(I, z["ids_I"], "inner selection")}
    gate_unix = max(in_rec["freeze_unix"], mk_rec["freeze_unix"])
    for rd in E.LABEL_READS:
        if rd["unix"] <= gate_unix:
            raise SystemExit("a label was read before the inputs and masks were frozen")
    assert {rd["month"] for rd in E.LABEL_READS} == {S, I}

    p = {k: z[f"p_{k}"] for k in ("V", "S", "I")}
    pop = {k: z[f"pop_{k}"] for k in ("V", "S", "I")}
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("V", "S", "I")}
    Xall = {k: E.design(codes[k], nlev) for k in ("V", "S", "I")}

    fit_rows = {k: pop[k] & np.isfinite(y[k]) & (y[k] > 0) for k in ("S", "I")}
    score_rows = {k: pop[k] & np.isfinite(y[k]) for k in ("S", "I")}
    Xfit = {k: E.design(codes[k], nlev, fit_rows[k]) for k in ("S", "I")}
    rfit = {}
    for k in ("S", "I"):
        rr = np.clip(y[k][fit_rows[k]] - p[k][fit_rows[k]], -E.CLIP, E.CLIP)
        rfit[k] = E.demean_by_airport(rr, codes[k]["apt"][fit_rows[k]])

    lu_cache, t0_cache = {}, {}

    def lu(F, lam):
        if (F, lam) not in lu_cache:
            X = Xfit[F]
            lu_cache[(F, lam)] = splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
        return lu_cache[(F, lam)]

    def data_part(F, lam):
        if (F, lam) not in t0_cache:
            t0_cache[(F, lam)] = (np.zeros(ncol) if np.isinf(lam)
                                  else lu(F, lam).solve(np.asarray(Xfit[F].T @ rfit[F], dtype="float64")))
        return t0_cache[(F, lam)]

    def slope(F, lam, pv):
        if not np.any(pv):
            return np.zeros(ncol)
        return pv.copy() if np.isinf(lam) else lu(F, lam).solve(lam * pv)

    base = {}
    for F, G in DIRS:
        m = score_rows[G]
        for lam in E.LAMBDAS:
            h0 = Xall[G] @ data_part(F, lam)
            base[(F, G, lam)] = (p[G][m] + h0[m], y[G][m], codes[G]["apt"][m])

    def prior_at(arm, i_n, i_c):
        if arm in E.DEGENERATE_ARMS:
            return np.zeros(ncol)
        if arm in E.SHAM_ARMS:
            return mk["eff_sham"][E.SHAM_ARMS.index(arm), i_n, i_c]
        return mk[f"eff_{arm}"][i_n, i_c]

    KEYS = E.cell_order()
    out_arrays, rec_arms, clip_counts = {}, {}, {}
    grid_arrays = {}
    loss_cols, rows_ref = {}, None
    for arm in E.ARMS:
        cells = {}
        for i_n, ns in enumerate(E.THRESHOLDS):
            for i_c, c in enumerate(E.C_GRID):
                pv = prior_at(arm, i_n, i_c)
                for lam in E.LAMBDAS:
                    curves = {lev: np.zeros((napt, len(GRIDS[lev]))) for lev in LEVELS}
                    for F, G in DIRS:
                        u, yy, apt = base[(F, G, lam)]
                        m = score_rows[G]
                        Sm = np.zeros((len(yy), 3))
                        for i, f in enumerate(E.PRIOR_FAMS):
                            pf = np.zeros(ncol)
                            pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                            if np.any(pf):
                                Sm[:, i] = (Xall[G] @ slope(F, lam, pf))[m]
                        st, dsse, _, cnt, mix = E.prepare_cells(yy, u, Sm, apt, napt, GRIDS["B"])
                        dA = (E.mixed_correction(yy[mix], u[mix], Sm[mix], GRIDS["A"], apt[mix],
                                                 napt)[0] if cnt["mixed"] else None)
                        for lev, d_ in (("B", dsse), ("A", dA)):
                            s_, _, _ = E.grid_curves(st, d_, None, PHI[lev])
                            curves[lev] += s_
                        clip_counts[f"{arm}|{E.cell_name((ns, lam, c))}|{F}>{G}"] = cnt
                    cells[(ns, lam, c)] = curves

        picks = {}
        for lev in LEVELS:
            totals, idxs = {}, {}
            for key, cur in cells.items():
                idx = cur[lev].argmin(1)
                idxs[key] = idx
                totals[key] = float(cur[lev][np.arange(napt), idx].sum())
            key = E.pick_cell3(totals)
            ns, lam, c = key
            idx = idxs[key]
            betas = GRIDS[lev][idx]
            rows_a = np.array([int((codes[G]["apt"][score_rows[G]] == aa).sum())
                               for aa in range(napt) for G in ("I", "S")]).reshape(napt, 2).sum(1)
            betas = np.where((rows_a > 0)[:, None], betas, 0.0)
            picks[lev] = {"key": key, "n_star": ns, "lam": lam, "c": c, "beta_index": idx,
                          "betas": betas, "inner_sse": totals[key],
                          "inner_sse_by_cell": {E.cell_name(k): totals[k] for k in KEYS},
                          "per_airport_min": [float(x) for x in cells[key][lev][np.arange(napt), idx]],
                          "airport_rows_in_selection": [int(x) for x in rows_a],
                          "airports_without_selection_rows": [int(aa) for aa in range(napt)
                                                              if rows_a[aa] == 0]}
            grid_arrays[f"{arm}|{lev}|total"] = np.array([totals[k] for k in KEYS])
            grid_arrays[f"{arm}|{lev}|per_airport_min"] = np.stack(
                [cells[k][lev][np.arange(napt), idxs[k]] for k in KEYS])
            grid_arrays[f"{arm}|{lev}|beta_index"] = np.stack([idxs[k] for k in KEYS]).astype("int32")

        # ---- the registered degeneracy proof for the arms without a prior -------------------
        if arm in E.DEGENERATE_ARMS:
            const_in_beta = all(
                np.array_equal(cells[k]["B"], np.repeat(cells[k]["B"][:, :1], len(GRIDS["B"]), 1))
                for k in cells)
            same_over_c_and_n = all(
                np.array_equal(cells[(E.THRESHOLDS[0], lam, E.C_GRID[0])]["B"], cells[(nn, lam, cc)]["B"])
                for lam in E.LAMBDAS for cc in E.C_GRID for nn in E.THRESHOLDS)
            row_only = {lam: float(cells[(E.THRESHOLDS[0], lam, E.C_GRID[0])]["B"][:, 0].sum())
                        for lam in E.LAMBDAS}
            best_lam = min(sorted(row_only, key=lambda L: -L), key=lambda L: row_only[L])
            ok_pick = (picks["B"]["n_star"] == E.THRESHOLDS[0] and np.isinf(picks["B"]["c"])
                       and picks["B"]["lam"] == best_lam
                       and float(np.max(np.abs(picks["B"]["betas"]))) == 0.0)
            rec_deg = {"prior_max_abs": 0.0,
                       "sse_constant_over_the_beta_grid": bool(const_in_beta),
                       "sse_identical_over_the_c_and_threshold_grid": bool(same_over_c_and_n),
                       "row_only_sse_by_lambda": {str(E.jsonable(L)): v for L, v in row_only.items()},
                       "row_only_minimising_lambda": str(E.jsonable(best_lam)),
                       "selected": {"n_star": picks["B"]["n_star"], "c": E.jsonable(picks["B"]["c"]),
                                    "lam": E.jsonable(picks["B"]["lam"]),
                                    "beta_max_abs": float(np.max(np.abs(picks["B"]["betas"])))},
                       "selected_the_registered_degenerate_point": bool(ok_pick)}
            if not (const_in_beta and same_over_c_and_n and ok_pick):
                raise SystemExit(f"{r} {arm}: the registered zero identity fails: {rec_deg}")
        else:
            rec_deg = None

        # ---- the final correction on S, and the direct-recomputation check ------------------
        arm_rec = {"degeneracy": rec_deg, "levels": {}}
        for lev in LEVELS:
            ns, lam, c = picks[lev]["key"]
            i_n, i_c = E.THRESHOLDS.index(ns), E.C_GRID.index(c)
            pv = prior_at(arm, i_n, i_c)
            bmap = picks[lev]["betas"]
            pvb = pv.copy()
            for i, f in enumerate(E.PRIOR_FAMS):
                sl = slice(off[i], off[i + 1])
                b_of_col = np.where(col_apt[sl] >= 0, bmap[np.maximum(col_apt[sl], 0), i], 0.0)
                pvb[sl] = pv[sl] * b_of_col
            th = data_part("S", lam) + slope("S", lam, pvb)
            out_arrays[f"h_{arm}_{lev}"] = Xall["V"] @ th
            direct, by_dir, parts, ids_, dirn = 0.0, {}, [], [], []
            for F, G in DIRS:
                th_ = data_part(F, lam) + slope(F, lam, pvb)
                h_ = Xall[G] @ th_
                m = score_rows[G]
                pr = np.maximum(E.MIN_PRED, p[G][m] + h_[m])
                row = (y[G][m] - pr) ** 2
                by_dir[f"{F}>{G}"] = {"rows": int(m.sum()), "sse": float(row.sum()),
                                      "sse_by_airport": [float(x) for x in np.bincount(
                                          codes[G]["apt"][m], weights=row, minlength=napt)]}
                direct += float(row.sum())
                if lev == "B":
                    parts.append(row)
                    ids_.append(z[f"ids_{G}"][m])
                    dirn.append(np.full(int(m.sum()), f"{F}>{G}"))
            rel = abs(direct - picks[lev]["inner_sse"]) / max(direct, 1.0)
            if rel > 1e-11:
                raise SystemExit(f"{r} {arm} level {lev}: the grid SSE {picks[lev]['inner_sse']} "
                                 f"does not reproduce the direct recomputation {direct} (rel {rel:.2e})")
            if lev == "B":
                loss_cols[f"loss:{arm}|B"] = np.concatenate(parts)
                if rows_ref is None:
                    rows_ref = {"MVT_ID_mvt": np.concatenate(ids_),
                                "direction": np.concatenate(dirn)}
            arm_rec["levels"][lev] = {
                "n_star": ns, "lam": E.jsonable(lam), "c": E.jsonable(c),
                "cell": E.cell_name(picks[lev]["key"]),
                "betas_by_airport": {str(vocab["apt"][aa]): [float(x) for x in bmap[aa]]
                                     for aa in range(napt)},
                "beta_nonzero_cells": int((bmap > 0).sum()), "beta_mean": float(bmap.mean()),
                "inner_sse": picks[lev]["inner_sse"], "inner_sse_direct": direct,
                "inner_sse_relative_difference": rel,
                "inner_sse_by_cell": picks[lev]["inner_sse_by_cell"],
                "per_airport_min": picks[lev]["per_airport_min"],
                "airport_rows_in_selection": picks[lev]["airport_rows_in_selection"],
                "airports_without_selection_rows": picks[lev]["airports_without_selection_rows"],
                "directional": by_dir,
                "prior_nonzero_columns": int((pv != 0).sum()),
                "h_sha256": E.sha_arr(out_arrays[f"h_{arm}_{lev}"]),
                "h_nonzero_rows": int((out_arrays[f"h_{arm}_{lev}"] != 0).sum())}
        rec_arms[arm] = arm_rec
        b = arm_rec["levels"]["B"]
        print(f"  {r} {arm:13s} B {b['cell']:>24s} beta>0 {b['beta_nonzero_cells']:2d}/30 "
              f"prior cols {b['prior_nonzero_columns']:4d} ({time.time() - t0:.0f}s)", flush=True)
        del cells

    # ---- the byte identity of the three degenerate arms --------------------------------------
    byte_ident = {}
    for arm in ("literal_sham", "const"):
        ok = all(np.array_equal(out_arrays[f"h_{arm}_{lev}"], out_arrays[f"h_zero_{lev}"])
                 for lev in LEVELS)
        same_grid = all(np.array_equal(grid_arrays[f"{arm}|{lev}|{w}"],
                                       grid_arrays[f"zero|{lev}|{w}"])
                        for lev in LEVELS for w in ("total", "per_airport_min", "beta_index"))
        same_loss = np.array_equal(loss_cols[f"loss:{arm}|B"], loss_cols["loss:zero|B"])
        byte_ident[arm] = {"predictions": bool(ok), "selection_grid": bool(same_grid),
                           "row_losses": bool(same_loss)}
        if not (ok and same_grid and same_loss):
            raise SystemExit(f"{arm} differs from zero: the registered zero identity fails")

    NO.refuse_existing([out_npz, out_grid, out_par, out_rec], False)
    np.savez(out_npz, **out_arrays)
    np.savez(out_grid, cell_order=np.array([E.cell_name(k) for k in KEYS]),
             airports=np.array([str(x) for x in vocab["apt"]]),
             beta_grid_A=GRIDS["A"], beta_grid_B=GRIDS["B"], **grid_arrays)
    out_par.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_ref | {k: loss_cols[k] for k in sorted(loss_cols)}).to_parquet(out_par,
                                                                                     index=False)
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r,
           "months": in_rec["months"], "launch": launch, "code_sha256": E.code_shas(),
           "inputs": {"record": f"experiments/exp-065-sparseholdout-retry/analysis/inputs65-{r}.json",
                      "npz_sha256": in_rec["npz_sha256"], "freeze_unix": in_rec["freeze_unix"]},
           "masks": {"record": f"experiments/exp-065-sparseholdout-retry/analysis/mask65-{r}.json",
                     "npz_sha256": mk_rec["npz_sha256"], "freeze_unix": mk_rec["freeze_unix"],
                     "matched_null_resolved": mk_rec["matched_null"]["resolved"],
                     "failing_cells": mk_rec["matched_null"]["failing_cells"]},
           "grid": {"thresholds": list(E.THRESHOLDS),
                    "lambdas": [E.jsonable(x) for x in E.LAMBDAS], "betas": list(E.BETAS),
                    "c": [E.jsonable(x) for x in E.C_GRID],
                    "cells": len(KEYS), "level_B_points": len(GRIDS["B"]),
                    "level_A_points": len(GRIDS["A"]),
                    "tie_order": "stage 1 smaller beta in stand, rwy, op order; stage 2 smaller "
                                 "n*, then larger lambda, then larger c"},
           "rows": {"fit": {k: int(v.sum()) for k, v in fit_rows.items()},
                    "score": {k: int(v.sum()) for k, v in score_rows.items()}},
           "clip_counts": clip_counts, "byte_identity_with_zero": byte_ident,
           "label_reads": E.LABEL_READS, "opened_paths": list(E.OPENED), "arms": rec_arms,
           "pred_npz": str(out_npz.relative_to(E.ROOT)), "pred_npz_sha256": E.sha_file(out_npz),
           "grid_npz": str(out_grid.relative_to(E.ROOT)), "grid_npz_sha256": E.sha_file(out_grid),
           "directional_row_losses": {"path": str(out_par.relative_to(E.ROOT)),
                                      "sha256": E.sha_file(out_par),
                                      "level": "B, at each arm's chosen map",
                                      "column_sha256": {k: E.sha_arr(v)
                                                        for k, v in sorted(loss_cols.items())}},
           "array_sha256": {k: E.sha_arr(v) for k, v in sorted(out_arrays.items())},
           "freeze_unix": time.time(), "freeze_utc": E.utc(),
           "seconds": round(time.time() - t0, 1)}
    out_rec.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"frozen {r} corrections {out_npz.name} {rec['pred_npz_sha256'][:16]} "
          f"({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
