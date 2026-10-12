"""exp-060-coldprior: fit, select and freeze the cold corrections of one rotation (SPEC 3).

Reads the frozen inputs (hash-checked), then the departure labels of S and I only.  For every arm
(zero, the literal sham, the cold tables true/const/gmean/donor/shuffle and the ten permuted-prior
shams) it evaluates the exact clipped pooled squared error of BOTH inner directions (fit S, score
I; fit I, score S) over the registered grid --- nine lambdas, six support constants c (inf is the
unweighted prior) and, per airport, the full 7^3 grid of (beta_stand, beta_runway, beta_operator)
--- then

  * stage 1: each airport takes the grid point with the smallest summed SSE (a tie takes the
    smaller beta in stand, runway, operator order);
  * stage 2: (lambda, c) takes the smallest total over all airports and both directions (a tie
    takes the larger lambda, then the larger c);

freezes that map, fits the final correction on S and writes the correction for V.  Level A (one
beta per airport across the three families) is selected the same way and is descriptive.
V's labels are not read here.  One rotation per process.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-060-coldprior/code/fit60.py jul
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

LEVELS = ("B", "A")
GRIDS = {"B": E.grid_B(), "A": E.grid_A()}
PHI = {lev: E.monomials(GRIDS[lev]) for lev in LEVELS}


def column_airport(vocab: dict, nlev: dict) -> np.ndarray:
    """The airport code of every correction column (the key string's `APT|` prefix)."""
    apt_pos = {v: i for i, v in enumerate(vocab["apt"])}
    out = []
    for f in E.PRIOR_FAMS:
        out.append(np.array([apt_pos.get(str(v).split("|")[0], -1) for v in vocab[f]], dtype="int64"))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", choices=E.ORDER)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    r = a.rotation
    out_npz = E.WORK / f"pred-{r}.npz"
    out_rec = E.ANALYSIS / f"fit60-{r}.json"
    NO.refuse_existing([out_npz, out_rec], a.overwrite)
    t0 = time.time()
    launch = E.hold_load(f"fit {r}")

    in_rec = E.load_json(E.ANALYSIS / f"inputs60-{r}.json")
    npz = E.ROOT / in_rec["npz"]
    if E.sha_file(npz) != in_rec["npz_sha256"]:
        raise SystemExit(f"{npz.name} is not the frozen input")
    z = np.load(npz)
    vocab = json.loads((E.ROOT / in_rec["vocab_path"]).read_text())
    V, S, I, _ = E.ROT[r]
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
    napt = nlev["apt"]
    off = E.fam_offsets(nlev)
    col_apt = column_airport(vocab, nlev)

    # ---- the only label reads of this job: S and I ------------------------------------------
    y = {"S": E.dep_labels(S, z["ids_S"], "cold correction fit and inner selection"),
         "I": E.dep_labels(I, z["ids_I"], "inner selection")}
    for rd in E.LABEL_READS:
        if rd["unix"] <= in_rec["freeze_unix"]:
            raise SystemExit("a label was read before the inputs were frozen")
    assert {rd["month"] for rd in E.LABEL_READS} == {S, I}

    p = {k: z[f"p_{k}"] for k in ("V", "S", "I")}
    pop = {k: z[f"pop_{k}"] for k in ("V", "S", "I")}
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("V", "S", "I")}
    Xall = {k: E.design(codes[k], nlev) for k in ("V", "S", "I")}
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}

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
        """theta at beta = 0: the data part of the correction fitted on month F."""
        if (F, lam) not in t0_cache:
            t0_cache[(F, lam)] = (np.zeros(ncol) if np.isinf(lam)
                                  else lu(F, lam).solve(np.asarray(Xfit[F].T @ rfit[F], dtype="float64")))
        return t0_cache[(F, lam)]

    def slope(F, lam, pv):
        """theta's derivative in beta for one prior vector (the same exact sparse solve)."""
        if not np.any(pv):
            return np.zeros(ncol)
        return pv.copy() if np.isinf(lam) else lu(F, lam).solve(lam * pv)

    DIRS = (("S", "I"), ("I", "S"))            # (fit month, scored month)
    # the scored-month pieces that do not depend on the arm
    base = {}
    for F, G in DIRS:
        m = score_rows[G]
        for lam in E.LAMBDAS:
            h0 = Xall[G] @ data_part(F, lam)
            base[(F, G, lam)] = (p[G][m] + h0[m], y[G][m], codes[G]["apt"][m])

    out_arrays, rec_arms = {}, {}
    clip_counts = {}
    for arm in E.ARMS:
        pv0 = (np.zeros(ncol) if arm in ("zero", "literal_sham")
               else E.prior_vector({f: z[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS}))
        degenerate = not np.any(pv0)
        cells = {}                      # (lam, c) -> level -> (per-airport curve)
        for c in E.C_GRID:
            pv = pv0 * E.support_weights(nA, c)
            for lam in E.LAMBDAS:
                curves = {lev: np.zeros((napt, len(GRIDS[lev]))) for lev in LEVELS}
                for F, G in DIRS:
                    u, yy, apt = base[(F, G, lam)]
                    m = score_rows[G]
                    Sm = np.zeros((len(yy), 3))
                    for i, f in enumerate(E.PRIOR_FAMS):
                        pf = np.zeros(ncol)
                        pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                        Sm[:, i] = (Xall[G] @ slope(F, lam, pf))[m]
                    # level A's grid has the same bounding box as level B's, so the clip
                    # classification and the statistics are shared; only the mixed rows'
                    # exact per-grid-point correction differs between the two grids
                    st, dsse, _, cnt, mix = E.prepare_cells(yy, u, Sm, apt, napt, GRIDS["B"])
                    if cnt["mixed"]:
                        dA = E.mixed_correction(yy[mix], u[mix], Sm[mix], GRIDS["A"],
                                                apt[mix], napt)[0]
                    else:
                        dA = None
                    for lev, d_ in (("B", dsse), ("A", dA)):
                        s_, _, _ = E.grid_curves(st, d_, None, PHI[lev])
                        curves[lev] += s_
                    key = f"{arm}|c={c}|lam={lam}|{F}>{G}"
                    clip_counts[key] = cnt
                cells[(lam, c)] = curves
        picks = {}
        for lev in LEVELS:
            totals = {}
            for key, cur in cells.items():
                idx = cur[lev].argmin(1)
                totals[key] = float(cur[lev][np.arange(napt), idx].sum())
            lam, c = E.pick_cell(totals)
            idx = cells[(lam, c)][lev].argmin(1)
            betas = GRIDS[lev][idx]
            rows_a = np.array([int((codes[G]["apt"][score_rows[G]] == aa).sum())
                               for aa in range(napt) for G in ("I", "S")]).reshape(napt, 2).sum(1)
            betas = np.where((rows_a > 0)[:, None], betas, 0.0)
            picks[lev] = {"lam": lam, "c": c, "beta_index": idx, "betas": betas,
                          "inner_sse": totals[(lam, c)],
                          "inner_sse_by_cell": {f"lam={k[0]},c={k[1]}": v for k, v in totals.items()},
                          "per_airport_min": [float(x) for x in
                                              cells[(lam, c)][lev][np.arange(napt), idx]],
                          "airport_rows_in_selection": [int(x) for x in rows_a],
                          "airports_without_selection_rows": [int(aa) for aa in range(napt)
                                                              if rows_a[aa] == 0]}
        # ---- degeneracy proof for the arms without a prior ---------------------------------
        if degenerate:
            flat = np.concatenate([cells[k]["B"].ravel() for k in cells])
            per = cells[(E.LAMBDAS[0], E.C_GRID[0])]["B"]
            const_in_k = all(np.array_equal(cells[k]["B"], np.repeat(cells[k]["B"][:, :1],
                                                                     len(GRIDS["B"]), 1))
                             for k in cells)
            same_c = all(np.array_equal(cells[(lam, E.C_GRID[0])]["B"], cells[(lam, cc)]["B"])
                         for lam in E.LAMBDAS for cc in E.C_GRID)
            rec_deg = {"prior_max_abs": 0.0, "sse_constant_over_the_beta_grid": bool(const_in_k),
                       "sse_identical_over_the_c_grid": bool(same_c),
                       "selected_beta": [float(x) for x in picks["B"]["betas"].ravel()],
                       "selected_c": str(picks["B"]["c"]),
                       "flat_min": float(flat.min()), "unused": float(per.max())}
            if not (const_in_k and same_c):
                raise SystemExit(f"{r} {arm}: the degeneracy proof fails")
        else:
            rec_deg = None
        # ---- the final correction on S, and the direct-recomputation check ------------------
        arm_rec = {"degeneracy": rec_deg, "levels": {}}
        for lev in LEVELS:
            lam, c = picks[lev]["lam"], picks[lev]["c"]
            pv = pv0 * E.support_weights(nA, c)
            bmap = picks[lev]["betas"]
            pvb = pv.copy()
            for i, f in enumerate(E.PRIOR_FAMS):
                sl = slice(off[i], off[i + 1])
                b_of_col = np.where(col_apt[sl] >= 0, bmap[np.maximum(col_apt[sl], 0), i], 0.0)
                pvb[sl] = pv[sl] * b_of_col
            th = data_part("S", lam) + slope("S", lam, pvb)
            out_arrays[f"h_{arm}_{lev}"] = Xall["V"] @ th
            # the same map scored on the inner months, directly, without the quadratic form
            direct, by_dir = 0.0, {}
            for F, G in DIRS:
                pvb_F = pvb
                th_ = data_part(F, lam) + slope(F, lam, pvb_F)
                h_ = Xall[G] @ th_
                m = score_rows[G]
                pr = np.maximum(E.MIN_PRED, p[G][m] + h_[m])
                sse = float(np.sum((y[G][m] - pr) ** 2))
                by_dir[f"{F}>{G}"] = {"rows": int(m.sum()), "sse": sse,
                                      "sse_by_airport": [float(x) for x in np.bincount(
                                          codes[G]["apt"][m], weights=(y[G][m] - pr) ** 2,
                                          minlength=napt)]}
                direct += sse
            rel = abs(direct - picks[lev]["inner_sse"]) / max(direct, 1.0)
            if rel > 1e-11:
                raise SystemExit(f"{r} {arm} level {lev}: the grid SSE {picks[lev]['inner_sse']} "
                                 f"does not reproduce the direct recomputation {direct} (rel {rel:.2e})")
            arm_rec["levels"][lev] = {
                "lam": "inf" if np.isinf(lam) else lam, "c": "inf" if np.isinf(c) else c,
                "betas_by_airport": {str(vocab["apt"][aa]): [float(x) for x in bmap[aa]]
                                     for aa in range(napt)},
                "inner_sse": picks[lev]["inner_sse"], "inner_sse_direct": direct,
                "inner_sse_relative_difference": rel,
                "inner_sse_by_cell": picks[lev]["inner_sse_by_cell"],
                "per_airport_min": picks[lev]["per_airport_min"],
                "airport_rows_in_selection": picks[lev]["airport_rows_in_selection"],
                "airports_without_selection_rows": picks[lev]["airports_without_selection_rows"],
                "directional": by_dir,
                "h_sha256": E.sha_arr(out_arrays[f"h_{arm}_{lev}"]),
                "h_nonzero_rows": int((out_arrays[f"h_{arm}_{lev}"] != 0).sum())}
        rec_arms[arm] = arm_rec
        b = rec_arms[arm]["levels"]["B"]
        print(f"  {r} {arm:13s} B lam={b['lam']!s:>6} c={b['c']!s:>6} "
              f"beta range [{min(min(v) for v in b['betas_by_airport'].values()):.1f}, "
              f"{max(max(v) for v in b['betas_by_airport'].values()):.1f}] ({time.time() - t0:.0f}s)",
              flush=True)

    # the literal sham and const must be the zero arm on the bytes
    byte_ident = {}
    for arm in ("literal_sham", "const"):
        ok = all(np.array_equal(out_arrays[f"h_{arm}_{lev}"], out_arrays[f"h_zero_{lev}"])
                 for lev in LEVELS)
        byte_ident[arm] = bool(ok)
        if not ok:
            raise SystemExit(f"{arm} differs from zero: the degeneracy proof fails")

    NO.refuse_existing([out_npz, out_rec], a.overwrite)
    np.savez(out_npz, **out_arrays)
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r,
           "months": {"V": V, "S": S, "I": I, "A": E.cold_months(r)},
           "inputs": {"record": f"experiments/exp-060-coldprior/analysis/inputs60-{r}.json",
                      "npz_sha256": in_rec["npz_sha256"], "freeze_unix": in_rec["freeze_unix"]},
           "cold_table_sha256": in_rec["cold_table"]["sha256"],
           "launch": launch, "code_sha256": E.code_shas(),
           "grid": {"lambdas": [x if np.isfinite(x) else "inf" for x in E.LAMBDAS],
                    "betas": list(E.BETAS), "c": [x if np.isfinite(x) else "inf" for x in E.C_GRID],
                    "level_B_points": len(GRIDS["B"]), "level_A_points": len(GRIDS["A"])},
           "rows": {"fit": {k: int(v.sum()) for k, v in fit_rows.items()},
                    "score": {k: int(v.sum()) for k, v in score_rows.items()}},
           "clip_counts": clip_counts,
           "byte_identity_with_zero": byte_ident,
           "label_reads": E.LABEL_READS, "arms": rec_arms,
           "pred_npz": str(out_npz.relative_to(E.ROOT)), "pred_npz_sha256": E.sha_file(out_npz),
           "array_sha256": {k: E.sha_arr(v) for k, v in sorted(out_arrays.items())},
           "freeze_unix": time.time(), "freeze_utc": E.utc(),
           "seconds": round(time.time() - t0, 1)}
    out_rec.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"frozen {r} corrections {out_npz.name} {rec['pred_npz_sha256'][:16]} "
          f"({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
