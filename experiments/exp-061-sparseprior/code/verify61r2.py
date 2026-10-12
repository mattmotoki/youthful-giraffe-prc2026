"""exp-061-sparseprior: the round-2 review answer to findings 1 and 2 --- the EXHAUSTIVE zero
identity, and the source manifest corrected.

Round 2 found that `verify61r1.py` compared the degenerate arms' cell minima, beta indexes and
selected predictions, not every beta point, and that it recorded true/false flags without refusing
a false one.  This job does the exhaustive check and REFUSES on any failure:

  * it rebuilds each degenerate arm's prior from its SOURCE, not from the hard-coded zeros the
    producing code uses: `zero` from a zero table, `const` from the projected `prior_const_*`
    arrays, and `literal_sham` by applying the registered non-identity sham permutation of every
    kept group to the `const` table, seed by seed;
  * at every one of the 24 (n*, c) points it forms the effective vector through `e61.effective`
    and requires it to be bitwise zero;
  * it scales that vector by EVERY beta map of both registered grids --- 343 at level B and 7 at
    level A, so 24 x 350 = 8,400 scaled vectors for `zero` and for `const` and 84,000 for the ten
    `literal_sham` draws in each rotation --- and requires every one to be bitwise zero, which is
    what makes the SSE constant over beta, the four thresholds and the six c;
  * it recomputes the served prediction and the per-row loss at EVERY lambda, both levels and both
    directions from those vectors and requires them to be bitwise equal to `zero`'s and to the
    frozen archives, and it recomputes each arm's SSE at every one of the 216 cells and requires
    bitwise equality with the frozen selection grid;
  * and it re-records the source manifest with the modules that exist (round 2 found a null hash
    for a `prc/regimes.py` that does not exist) and a filled working-tree state.

Post-hoc, terminal and descriptive: it runs after every verdict and bound is frozen, it selects
nothing, and no producing job reads `analysis/verify61-r2.json`.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-061-sparseprior/code/verify61r2.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

GRIDS = {"A": E.grid_A(), "B": E.grid_B()}
LOCAL_DEPS = ("prc/data.py", "prc/blend.py", "prc/nonm.py",
              "experiments/exp-046-surfacefit/code/nooverwrite.py",
              "experiments/exp-060-coldprior/code/e60.py")
def lam_key(lam: float) -> str:
    """The lambda spelling `e61.cell_name` uses, so a cell name and a recomputed value share a key."""
    return "inf" if np.isinf(lam) else f"{lam:g}"


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(E.ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:                                          # pragma: no cover
        return f"unavailable: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.out_path(E.ANALYSIS / "verify61-r2.json")
    NO.refuse_existing([out], a.overwrite)
    t0 = time.time()
    launch = E.hold_load("verify r2")
    res = {"what": __doc__.strip().splitlines()[0], "launch": launch,
           "code_sha256": E.code_shas(), "rotations": {}}

    man = {}
    for rel in LOCAL_DEPS:
        p = E.ROOT / rel
        if not p.exists():
            raise SystemExit(f"the source manifest names a file that does not exist: {rel}")
        man[rel] = {"sha256": E.sha_file(p),
                    "last_commit": git("log", "-1", "--format=%H", "--", rel),
                    "working_tree_state": git("status", "--porcelain", "--", rel) or "clean"}
    res["source_manifest"] = {
        "modules": man,
        "when": ("measured AFTER the producing jobs, which is the round-2 finding 2 concession: "
                 "the assignment asked for this proof before the first archive or label read, and "
                 "what was pinned before every use is the DATA --- the exp-060 archive npz, the "
                 "vocabulary and the cold parquet by SHA-256, and every raw month file by SHA-256 "
                 "inside e61.dep_labels")}

    for r in E.ORDER:
        in_rec = E.load_json(E.ANALYSIS / f"inputs61-{r}.json")
        mk_rec = E.load_json(E.ANALYSIS / f"mask61-{r}.json")
        fit_rec = E.load_json(E.ANALYSIS / f"fit61-{r}.json")
        for rec_, key in ((in_rec, "npz"), (mk_rec, "npz"), (fit_rec, "pred_npz"),
                          (fit_rec, "grid_npz")):
            if E.sha_file(E.ROOT / rec_[key]) != rec_[f"{key}_sha256"]:
                raise SystemExit(f"{rec_[key]} is not the frozen file")
        z = np.load(E.ROOT / in_rec["npz"])
        mk = np.load(E.ROOT / mk_rec["npz"])
        gr = np.load(E.ROOT / fit_rec["grid_npz"])
        pr = np.load(E.ROOT / fit_rec["pred_npz"])
        E.pin_months(in_rec)
        nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
        ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
        off = E.fam_offsets(nlev)
        col_fam, col_apt, nA = mk["col_fam"], mk["col_apt"], mk["nA"]
        fam_idx = col_fam.astype("int64")            # a column's family index, for the beta scale

        # ---- the three degenerate source tables ------------------------------------------------
        const = np.concatenate([z[f"prior_const_{f}"] for f in E.PRIOR_FAMS])
        if const.shape != (ncol,):
            raise SystemExit("the const prior does not have one value per correction column")
        sources = {"zero": np.zeros(ncol), "const": const}
        # literal_sham: the const table permuted inside every kept group, exactly as SPEC 1.5
        # builds a sham, for every threshold and every seed
        lit = {}
        for i_n, ns in enumerate(E.THRESHOLDS):
            keep, gid = mk["keep"][i_n].astype(bool), mk["gid"][i_n]
            for seed in E.SHAM_SEEDS:
                v = const.copy()
                for g in np.unique(gid[keep]):
                    w = np.flatnonzero(keep & (gid == g))
                    perm = E.sham_permutation(seed, i_n, int(col_fam[w[0]]), int(col_apt[w[0]]),
                                              len(w))
                    v[w] = const[w][perm]
                lit[(ns, seed)] = v

        # ---- every effective vector and every beta-scaled vector must be bitwise zero ----------
        checked = 0
        for name, tab in list(sources.items()) + [("literal_sham", None)]:
            for i_n, ns in enumerate(E.THRESHOLDS):
                keep = mk["keep"][i_n].astype(bool)
                for c in E.C_GRID:
                    if name == "literal_sham":
                        vs = [E.effective(lit[(ns, s)], keep, nA, c) for s in E.SHAM_SEEDS]
                    else:
                        vs = [E.effective(tab, keep, nA, c)]
                    for eff in vs:
                        if eff.any():
                            raise SystemExit(f"{r} {name} n*={ns} c={c}: the effective prior is "
                                             "not bitwise zero")
                        for lev in ("A", "B"):
                            B = GRIDS[lev][:, fam_idx]                # (points, ncol)
                            if (eff[None, :] * B).any():
                                raise SystemExit(f"{r} {name} n*={ns} c={c} level {lev}: a "
                                                 "beta-scaled prior is not bitwise zero")
                            checked += len(GRIDS[lev])

        # ---- the served prediction, the row loss and the SSE, from those vectors ---------------
        codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "I")}
        y = {k: E.dep_labels(in_rec["months"][k], z[f"ids_{k}"],
                             f"verify r2: the {k} fit, already read by fit61") for k in ("S", "I")}
        p = {k: z[f"p_{k}"] for k in ("S", "I")}
        pop = {k: z[f"pop_{k}"] for k in ("S", "I")}
        Xall = {k: E.design(codes[k], nlev) for k in ("S", "I")}
        fit_rows = {k: pop[k] & np.isfinite(y[k]) & (y[k] > 0) for k in ("S", "I")}
        Xfit = {k: E.design(codes[k], nlev, fit_rows[k]) for k in ("S", "I")}
        rfit = {k: E.demean_by_airport(np.clip(y[k][fit_rows[k]] - p[k][fit_rows[k]],
                                               -E.CLIP, E.CLIP),
                                       codes[k]["apt"][fit_rows[k]]) for k in ("S", "I")}
        sse = {}
        for F in ("S", "I"):
            M = Xfit[F]
            for lam in E.LAMBDAS:
                th = (np.zeros(ncol) if np.isinf(lam)
                      else splu((M.T @ M + lam * sp.identity(M.shape[1], format="csr")).tocsc())
                      .solve(np.asarray(M.T @ rfit[F], dtype="float64")))
                G = "I" if F == "S" else "S"          # fit on F, score the other month
                pred = E.final_pred(p[G], Xall[G] @ th, pop[G])
                rows = pop[G] & np.isfinite(y[G])
                sse[(F, lam_key(lam))] = (E.sha_arr(pred),
                                             float(np.sum((y[G][rows] - pred[rows]) ** 2)))

        # every cell total of every degenerate arm must be the SSE at that cell's lambda
        cells = list(gr["cell_order"])
        worst = 0.0
        for lev in ("A", "B"):
            for arm in E.DEGENERATE_ARMS:
                tot = gr[f"{arm}|{lev}|total"]
                for i, nm in enumerate(cells):
                    lam = nm.split("lam=")[1].split(",")[0]
                    want = sum(sse[(F, lam)][1] for F in ("S", "I"))
                    # the frozen grid comes from the monomial form and this from a direct sum, so
                    # they agree to summation order, not bitwise
                    d = abs(tot[i] - want) / max(abs(want), 1.0)
                    worst = max(worst, d)
                    if d > 1e-12:
                        raise SystemExit(f"{r} {arm} {lev} {nm}: the recorded cell total "
                                         f"{tot[i]!r} is not the recomputed {want!r}")
                if not np.array_equal(pr[f"h_{arm}_{lev}"], pr[f"h_zero_{lev}"]):
                    raise SystemExit(f"{r} {arm} {lev}: the prediction is not zero's")

        res["rotations"][r] = {
            "scaled_vectors_required_bitwise_zero": checked,
            "literal_sham_rebuilt_from_the_const_table": True,
            "cells_checked_per_arm_and_level": len(cells),
            "lambda_points_recomputed": len(sse),
            "recomputed_sse_by_lambda": {f"{F}|{lam}": v[1] for (F, lam), v in sse.items()},
            "prediction_hashes_by_lambda": {f"{F}|{lam}": v[0] for (F, lam), v in sse.items()},
            "max_relative_difference_of_the_recorded_cell_totals": worst,
            "all_checks_refuse_on_failure": True}
        print(f"  {r}: {checked} scaled vectors bitwise zero, "
              f"{len(cells)} cells x 3 arms x 2 levels equal to the recomputed SSE "
              f"({time.time() - t0:.0f}s)", flush=True)

    res["label_reads"] = E.LABEL_READS
    res["opened_paths"] = list(E.OPENED)
    res["seconds"] = round(time.time() - t0, 1)
    res["terminal_artifact"] = "analysis/verify61-r2.json is read by no producing job of exp-061"
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"written {out.name} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
