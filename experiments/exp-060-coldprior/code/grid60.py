"""exp-060-coldprior: the FULL selection grid and the directional row losses (SPEC 3.6).

Review round 1 finding 3: `fit60.py` saved one total per (lambda, c) cell and the per-airport
minima and beta triples of the WINNING cell only, and it saved directional SSEs rather than row
losses.  SPEC 3.6 promised "the full selection grid (per arm: every (lambda, c) with its
per-airport minimum and chosen triple, and the total)" and "both directional losses".  This job
supplies both, by recomputing the selection independently of `fit60.py` and refusing if it does
not reproduce every recorded pick and total.

It writes
  * `analysis/grid60-<r>.json` - per arm, per level, per (lambda, c): the ten per-airport minima,
    the ten chosen beta triples and the total; plus the reproduction check against
    `analysis/fit60-<r>.json`;
  * `artifacts/exp-060/dirloss-<r>.parquet` - the per-row squared error of both inner directions
    at every arm's chosen level-B map (one column per arm), with the month and the row id.

It reads the departure labels of S and I, which `fit60.py` already read; it reads no V label and
it changes no choice.  One rotation per process.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-060-coldprior/code/grid60.py jul
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
import e60 as E                                                       # noqa: E402
from fit60 import column_airport                                      # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

LEVELS = ("B", "A")
GRIDS = {"B": E.grid_B(), "A": E.grid_A()}
PHI = {lev: E.monomials(GRIDS[lev]) for lev in LEVELS}
DIRS = (("S", "I"), ("I", "S"))
TOL = 1e-12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", choices=E.ORDER)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    r = a.rotation
    out_rec = E.ANALYSIS / f"grid60-{r}.json"
    out_par = E.ART / f"dirloss-{r}.parquet"
    NO.refuse_existing([out_rec, out_par], a.overwrite)
    t0 = time.time()
    launch = E.hold_load(f"grid {r}")

    in_rec = E.load_json(E.ANALYSIS / f"inputs60-{r}.json")
    fit_rec = E.load_json(E.ANALYSIS / f"fit60-{r}.json")
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

    y = {"S": E.dep_labels(S, z["ids_S"], "full selection grid (SPEC 3.6), already read by fit60"),
         "I": E.dep_labels(I, z["ids_I"], "full selection grid (SPEC 3.6), already read by fit60")}
    p = {k: z[f"p_{k}"] for k in ("S", "I")}
    pop = {k: z[f"pop_{k}"] for k in ("S", "I")}
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "I")}
    Xall = {k: E.design(codes[k], nlev) for k in ("S", "I")}
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
        if (F, lam) not in t0_cache:
            t0_cache[(F, lam)] = (np.zeros(ncol) if np.isinf(lam)
                                  else lu(F, lam).solve(np.asarray(Xfit[F].T @ rfit[F], "float64")))
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

    def cell_curves(pv, lam):
        cur = {lev: np.zeros((napt, len(GRIDS[lev]))) for lev in LEVELS}
        for F, G in DIRS:
            u, yy, apt = base[(F, G, lam)]
            Sm = np.zeros((len(yy), 3))
            for i, f in enumerate(E.PRIOR_FAMS):
                pf = np.zeros(ncol)
                pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                Sm[:, i] = (Xall[G] @ slope(F, lam, pf))[score_rows[G]]
            st, dsse, _, cnt, mix = E.prepare_cells(yy, u, Sm, apt, napt, GRIDS["B"])
            dA = (E.mixed_correction(yy[mix], u[mix], Sm[mix], GRIDS["A"], apt[mix], napt)[0]
                  if cnt["mixed"] else None)
            for lev, d_ in (("B", dsse), ("A", dA)):
                s_, _, _ = E.grid_curves(st, d_, None, PHI[lev])
                cur[lev] += s_
        return cur

    rec = {"what": ("the full selection grid and the directional row losses of one rotation "
                    "(SPEC 3.6; review round 1 finding 3)"),
           "rotation": r, "utc": E.utc(), "launch": launch, "arms": {}, "mismatches": [],
           "fit60_record": {"path": str((E.ANALYSIS / f'fit60-{r}.json').relative_to(E.ROOT)),
                            "sha256": E.sha_file(E.ANALYSIS / f"fit60-{r}.json")}}
    loss_cols, rows_ref = {}, None
    for arm in E.ARMS:
        pv0 = (np.zeros(ncol) if arm in ("zero", "literal_sham")
               else E.prior_vector({f: z[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS}))
        cells = {}
        for c in E.C_GRID:
            pv = pv0 * E.support_weights(nA, c)
            for lam in E.LAMBDAS:
                cells[(lam, c)] = cell_curves(pv, lam)
        a_rec = {}
        for lev in LEVELS:
            grid_rows, totals = {}, {}
            for (lam, c), cur in cells.items():
                idx = cur[lev].argmin(1)
                mins = cur[lev][np.arange(napt), idx]
                totals[(lam, c)] = float(mins.sum())
                grid_rows[f"lam={lam},c={c}"] = {
                    "total": float(mins.sum()),
                    "airport_minimum_sse": [float(x) for x in mins],
                    "airport_beta": [[float(b) for b in GRIDS[lev][j]] for j in idx]}
            lam, c = E.pick_cell(totals)
            got = {"lam": "inf" if np.isinf(lam) else lam, "c": "inf" if np.isinf(c) else c}
            want = {k: fit_rec["arms"][arm]["levels"][lev][k] for k in ("lam", "c")}
            if got != want:
                rec["mismatches"].append({"arm": arm, "level": lev, "mine": got, "fit60": want})
            d = abs(totals[(lam, c)] - fit_rec["arms"][arm]["levels"][lev]["inner_sse"])
            if d / max(abs(totals[(lam, c)]), 1.0) > TOL:
                rec["mismatches"].append({"arm": arm, "level": lev, "sse_relative_difference":
                                          d / abs(totals[(lam, c)])})
            idx = cells[(lam, c)][lev].argmin(1)
            bet = GRIDS[lev][idx]
            rows_a = np.array([int((codes[G]["apt"][score_rows[G]] == aa).sum())
                               for aa in range(napt) for G in ("I", "S")]).reshape(napt, 2).sum(1)
            bet = np.where((rows_a > 0)[:, None], bet, 0.0)
            want_b = fit_rec["arms"][arm]["levels"][lev]["betas_by_airport"]
            for aa in range(napt):
                if [float(x) for x in bet[aa]] != [float(x) for x in want_b[str(vocab["apt"][aa])]]:
                    rec["mismatches"].append({"arm": arm, "level": lev,
                                              "airport": str(vocab["apt"][aa])})
            a_rec[lev] = {"picked": got, "total_at_the_pick": totals[(lam, c)], "grid": grid_rows}
            if lev == "B":
                # the per-row squared error of both directions at this arm's chosen level-B map
                pv = pv0 * E.support_weights(nA, c)
                parts, ids, mons = [], [], []
                for F, G in DIRS:
                    u, yy, apt = base[(F, G, lam)]
                    Sm = np.zeros((len(yy), 3))
                    for i, f in enumerate(E.PRIOR_FAMS):
                        pf = np.zeros(ncol)
                        pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                        Sm[:, i] = (Xall[G] @ slope(F, lam, pf))[score_rows[G]]
                    pred = np.maximum(E.MIN_PRED, u + np.einsum("ij,ij->i", Sm, bet[apt]))
                    parts.append((yy - pred) ** 2)
                    ids.append(z[f"ids_{G}"][score_rows[G]])
                    mons.append(np.full(len(yy), f"{F}>{G}"))
                loss_cols[f"loss:{arm}|B"] = np.concatenate(parts)
                if rows_ref is None:
                    rows_ref = {"MVT_ID_mvt": np.concatenate(ids),
                                "direction": np.concatenate(mons)}
                s = float(np.concatenate(parts).sum())
                w = (fit_rec["arms"][arm]["levels"]["B"]["directional"]["S>I"]["sse"]
                     + fit_rec["arms"][arm]["levels"]["B"]["directional"]["I>S"]["sse"])
                if abs(s - w) / max(abs(w), 1.0) > 1e-11:
                    rec["mismatches"].append({"arm": arm, "row_loss_sum_relative_difference":
                                              abs(s - w) / abs(w)})
        rec["arms"][arm] = a_rec
        print(f"  {r} {arm:13s} B {a_rec['B']['picked']}  A {a_rec['A']['picked']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows_ref | {k: loss_cols[k] for k in sorted(loss_cols)})
    NO.refuse_existing([out_rec, out_par], a.overwrite)
    out_par.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_par, index=False)
    rec["row_losses"] = {"path": str(out_par.relative_to(E.ROOT)), "sha256": E.sha_file(out_par),
                         "rows": int(len(df)), "arms": len(loss_cols),
                         "column_sha256": {k: E.sha_arr(loss_cols[k]) for k in sorted(loss_cols)}}
    rec["reproduces_fit60"] = not rec["mismatches"]
    rec["label_reads"] = E.LABEL_READS
    rec["code_sha256"] = E.code_shas()
    rec["seconds"] = round(time.time() - t0, 1)
    out_rec.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({"rotation": r, "reproduces_fit60": rec["reproduces_fit60"],
                      "mismatches": len(rec["mismatches"])}))
    if rec["mismatches"]:
        raise SystemExit(f"REFUSING: the independent grid does not reproduce fit60-{r}")
    print(f"written {out_rec.relative_to(E.ROOT)} and {out_par.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
