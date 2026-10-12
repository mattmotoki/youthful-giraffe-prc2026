"""exp-060-coldprior: the registered level-A one-beta identity check (SPEC 6.1).

SPEC 6.1 promised: "A registered identity check proves that level A restricted to one beta for all
airports reproduces the cold grid oracle (maximum absolute SSE difference recorded; a relative
difference above 1e-6 refuses)."  `bound60.py` made the two identical by construction (it computes
the grid-oracle level as level A with the airport argmin shared) and recorded what the per-airport
freedom buys instead, so the promised number was never produced.  Review round 1 finding 3 is
right about that.

This job produces it, by an INDEPENDENT path.  For every arm and every (lambda, c) cell of every
rotation it computes the airport-summed SSE at each of the seven common betas twice:

  * through the level-A grid and monomials (7 points), the path `bound60.py` uses;
  * through the level-B grid and monomials (343 points), restricted to the seven diagonal triples
    (b, b, b), which is the same estimator reached by different machinery and different clip
    corrections (level B's mixed-row correction table, not level A's).

It records the maximum absolute and relative difference over all three bound kinds, and refuses
above 1e-6 relative.  It reads S and V, which `bound60.py` already read, scores nothing new and
changes no choice.  One rotation per process.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-060-coldprior/code/identity60.py jul
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
#: the positions of the diagonal triples (b, b, b) inside the level-B grid
DIAG = np.array([int(np.flatnonzero(np.all(GRIDS["B"] == np.array([b, b, b]), axis=1))[0])
                 for b in E.BETAS])
TOL = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", choices=E.ORDER)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    r = a.rotation
    out = E.ANALYSIS / f"identity60-{r}.json"
    NO.refuse_existing([out], a.overwrite)
    t0 = time.time()
    launch = E.hold_load(f"identity {r}")

    in_rec = E.load_json(E.ANALYSIS / f"inputs60-{r}.json")
    z = np.load(E.ROOT / in_rec["npz"])
    V, S, _, _ = E.ROT[r]
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
    napt = nlev["apt"]
    off = E.fam_offsets(nlev)

    yS = E.dep_labels(S, z["ids_S"], "identity check: the S fit (already read by bound60)")
    yV = E.dep_labels(V, z["ids_V"], "identity check: V in sample (already read by bound60)")
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
    p, P, day = z["p_V"], z["pop_V"], z["day_V"]
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
    eo = yV[~P] - p[~P]
    cello = apt[~P] * ndates + day[~P]
    st_out = np.stack([np.bincount(cello, minlength=ncell),
                       np.bincount(cello, weights=eo, minlength=ncell),
                       np.bincount(cello, weights=eo * eo, minlength=ncell)], 1)
    sse_out_total = float(st_out[:, 2].sum())

    def by_airport(x):
        return x.reshape(napt, ndates, -1).sum(1)

    st_out_a = by_airport(st_out)
    worst = {"points": 0, "max_abs_sse_difference": 0.0, "max_relative_difference": 0.0,
             "where": None}
    per_arm = {}
    for arm in E.ARMS:
        pv0 = (np.zeros(ncol) if arm in ("zero", "literal_sham")
               else E.prior_vector({f: z[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS}))
        amax = 0.0
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
                sA, rA, nA_ = E.grid_curves(by_airport(st), None if dA is None else by_airport(dA),
                                            None, PHI["A"], st_out=st_out_a)
                sB, rB, nB_ = E.grid_curves(by_airport(st), None if dB is None else by_airport(dB),
                                            None, PHI["B"], st_out=st_out_a)
                sB, rB = sB[:, DIAG], rB[:, DIAG]
                for kind, adjust in (("full", False), ("adjusted_all_rows", True)):
                    if adjust:
                        vA = sA - (rA ** 2) / np.maximum(nA_, 1.0)[:, None]
                        vB = sB - (rB ** 2) / np.maximum(nB_, 1.0)[:, None]
                    else:
                        vA, vB = sA, sB
                    tA = vA.sum(0)
                    tB = vB.sum(0)
                    d = float(np.max(np.abs(tA - tB)))
                    rel = d / max(float(np.max(np.abs(tA))), 1.0)
                    amax = max(amax, rel)
                    worst["points"] += len(tA)
                    if d > worst["max_abs_sse_difference"]:
                        worst["max_abs_sse_difference"] = d
                    if rel > worst["max_relative_difference"]:
                        worst["max_relative_difference"] = rel
                        worst["where"] = f"{arm}|{kind}|lam={lam}|c={c}"
        per_arm[arm] = amax
        print(f"  {r} {arm:13s} max relative {amax:.3e} ({time.time() - t0:.0f}s)", flush=True)

    rec = {"what": ("the registered level-A one-beta identity against the cold grid oracle, "
                    "computed through the independent level-B machinery (SPEC 6.1)"),
           "rotation": r, "V": V, "S": S, "utc": E.utc(), "launch": launch,
           "diagonal_positions_in_the_level_B_grid": [int(x) for x in DIAG],
           "per_arm_max_relative_difference": per_arm, "worst": worst, "tolerance": TOL,
           "sse_out_total_s2": sse_out_total,
           "inputs": {"npz_sha256": in_rec["npz_sha256"]},
           "label_reads": E.LABEL_READS, "code_sha256": E.code_shas(),
           "seconds": round(time.time() - t0, 1)}
    rec["pass"] = bool(worst["max_relative_difference"] <= TOL)
    NO.refuse_existing([out], a.overwrite)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({"rotation": r, "max_relative_difference": worst["max_relative_difference"],
                      "pass": rec["pass"]}))
    if not rec["pass"]:
        raise SystemExit(f"REFUSING: the one-beta identity fails at {worst['where']}")
    print(f"written {out.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
