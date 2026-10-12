"""exp-063-sparseholdout: the served-prediction half of SPEC 4.3, and the exact diagnosis of every
cell that fails the label-free matched-null criterion.

Post-hoc and descriptive.  It runs after the V verdict is frozen, selects nothing, and no number
it produces enters a choice, a gate, a bound or a registered reading.  It exists because SPEC
reading (e) registered a LABEL-FREE criterion --- a population row differs when it uses a
correction column whose effective value differs --- which is sufficient but not necessary: the
ridge solve couples columns of the same airport that share a fit row, so a served prediction can
move on a row that uses no moved column.  This job measures the served prediction directly.

Two measurements:

  (a) at each sham's own chosen level-B cell and beta map, the maximum absolute difference of the
      served prediction from `true`'s, on the S, I and V population rows;
  (b) for every cell the label-free criterion failed, the same difference at that exact (n*, c)
      cell with the rotation's `true` level-B lambda and beta = 1.0 in every airport and family
      (the largest registered scale, which maximises the difference).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-063-sparseholdout/code/nullcheck61.py
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
import e63 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

DIRS = {"I": ("S", "I"), "S": ("I", "S")}          # scored month -> (fit month, scored month)


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    out = E.out_path(E.ANALYSIS / "nullcheck61.json")
    NO.refuse_existing([out], False)
    t0 = time.time()
    launch = E.hold_load("nullcheck")
    oos = E.load_json(E.ANALYSIS / "oos63.json")

    res = {"what": __doc__.strip().splitlines()[0], "launch": launch,
           "code_sha256": E.code_shas(),
           "oos_gate_advance": oos["gate"]["advance"], "rotations": {}}
    for r in E.ORDER:
        in_rec = E.load_json(E.ANALYSIS / f"inputs63-{r}.json")
        mk_rec = E.load_json(E.ANALYSIS / f"mask63-{r}.json")
        fit_rec = E.load_json(E.ANALYSIS / f"fit61-{r}.json")
        for rec_, key in ((in_rec, "npz"), (mk_rec, "npz"), (fit_rec, "pred_npz")):
            if E.sha_file(E.ROOT / rec_[key]) != rec_[f"{key}_sha256"]:
                raise SystemExit(f"{rec_[key]} is not the frozen file")
        z = np.load(E.ROOT / in_rec["npz"])
        mk = np.load(E.ROOT / mk_rec["npz"])
        vocab = json.loads((E.ROOT / in_rec["vocab63_path"]).read_text())
        E.pin_months(in_rec)
        V, S, I = (in_rec["months"][k] for k in ("V", "S", "I"))
        nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
        ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
        napt = nlev["apt"]
        off = E.fam_offsets(nlev)
        col_apt = z["col_apt"]

        y = {"S": E.dep_labels(S, z["ids_S"], "nullcheck: the S fit, already read by fit61"),
             "I": E.dep_labels(I, z["ids_I"], "nullcheck: the I fit, already read by fit61")}
        p = {k: z[f"p_{k}"] for k in ("V", "S", "I")}
        pop = {k: z[f"pop_{k}"] for k in ("V", "S", "I")}
        codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("V", "S", "I")}
        Xall = {k: E.design(codes[k], nlev) for k in ("V", "S", "I")}
        fit_rows = {k: pop[k] & np.isfinite(y[k]) & (y[k] > 0) for k in ("S", "I")}
        Xfit = {k: E.design(codes[k], nlev, fit_rows[k]) for k in ("S", "I")}
        rfit = {k: E.demean_by_airport(np.clip(y[k][fit_rows[k]] - p[k][fit_rows[k]],
                                               -E.CLIP, E.CLIP),
                                       codes[k]["apt"][fit_rows[k]]) for k in ("S", "I")}
        lu_cache, t0_cache = {}, {}

        def lu(F, lam):
            if (F, lam) not in lu_cache:
                X = Xfit[F]
                lu_cache[(F, lam)] = splu((X.T @ X + lam * sp.identity(X.shape[1],
                                                                      format="csr")).tocsc())
            return lu_cache[(F, lam)]

        def theta(F, lam, pvb):
            if (F, lam) not in t0_cache:
                t0_cache[(F, lam)] = (np.zeros(ncol) if np.isinf(lam)
                                      else lu(F, lam).solve(np.asarray(Xfit[F].T @ rfit[F],
                                                                       dtype="float64")))
            d = t0_cache[(F, lam)]
            if not np.any(pvb):
                return d
            return d + (pvb.copy() if np.isinf(lam) else lu(F, lam).solve(lam * pvb))

        def scale(pv, bmap):
            out_ = pv.copy()
            for i in range(len(E.PRIOR_FAMS)):
                sl = slice(off[i], off[i + 1])
                b = np.where(col_apt[sl] >= 0, bmap[np.maximum(col_apt[sl], 0), i], 0.0)
                out_[sl] = pv[sl] * b
            return out_

        def served(month, lam, pvb):
            F = "S" if month in ("V", "I") else "I"
            th = theta(F, lam, pvb)
            return E.final_pred(p[month], Xall[month] @ th, pop[month])

        def prior_at(arm, i_n, i_c):
            if arm in E.DEGENERATE_ARMS:
                return np.zeros(ncol)
            if arm in E.SHAM_ARMS:
                return mk["eff_sham"][E.SHAM_ARMS.index(arm), i_n, i_c]
            return mk[f"eff_{arm}"][i_n, i_c]

        def pick(arm):
            b = fit_rec["arms"][arm]["levels"]["B"]
            lam = float("inf") if b["lam"] == "inf" else float(b["lam"])
            c = float("inf") if b["c"] == "inf" else float(b["c"])
            bmap = np.array([b["betas_by_airport"][str(vocab["apt"][aa])] for aa in range(napt)])
            return int(b["n_star"]), lam, c, bmap

        # ---- (a) at each arm's own chosen cell ---------------------------------------------
        ns, lam, c, bmap = pick("true")
        pv_true = scale(prior_at("true", E.THRESHOLDS.index(ns), E.C_GRID.index(c)), bmap)
        base_pred = {m: served(m, lam, pv_true) for m in ("S", "I", "V")}
        own = {}
        for s in E.SHAM_ARMS:
            ns2, lam2, c2, bmap2 = pick(s)
            pv2 = scale(prior_at(s, E.THRESHOLDS.index(ns2), E.C_GRID.index(c2)), bmap2)
            own[s] = {}
            for m in ("S", "I", "V"):
                d = np.abs(served(m, lam2, pv2) - base_pred[m])[pop[m]]
                own[s][m] = {"max_abs_s": float(d.max()), "rows_differing": int((d > 0).sum())}

        # ---- (b) the cells the label-free criterion failed ---------------------------------
        failing = {}
        for key in mk_rec["matched_null"]["failing_cells"]:
            parts = dict(kv.split("=") for kv in key.split(","))
            ns_f = int(parts["n"])
            c_f = float("inf") if parts["c"] == "inf" else float(parts["c"])
            seed = int(parts["seed"])
            i_n, i_c = E.THRESHOLDS.index(ns_f), E.C_GRID.index(c_f)
            one = np.ones((napt, 3))
            pv_t = scale(mk["eff_true"][i_n, i_c], one)
            pv_s = scale(mk["eff_sham"][seed - 1, i_n, i_c], one)
            ent = {"n_star": ns_f, "c": E.jsonable(c_f), "seed": seed,
                   "lambda": E.jsonable(lam), "beta": 1.0,
                   "label_free_moved_population_rows":
                       mk_rec["moved_by_cell"][key]["moved_population_rows"]}
            for m in ("S", "I", "V"):
                d = np.abs(served(m, lam, pv_s) - served(m, lam, pv_t))[pop[m]]
                ent[m] = {"max_abs_s": float(d.max()), "rows_differing": int((d > 0).sum())}
            failing[key] = ent

        res["rotations"][r] = {
            "months": {"V": V, "S": S, "I": I},
            "matched_null_resolved": mk_rec["matched_null"]["resolved"],
            "at_each_arms_own_chosen_cell": own,
            "cells_failing_the_label_free_criterion": failing,
            "served_prediction_moves_on_every_failing_cell":
                bool(failing) and all(min(v[m]["rows_differing"] for m in ("S", "I", "V")) > 0
                                      for v in failing.values())}
        print(f"  {r}: failing cells {len(failing)}; "
              f"served prediction moves on all of them "
              f"{res['rotations'][r]['served_prediction_moves_on_every_failing_cell']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    res["label_reads"] = E.LABEL_READS
    res["opened_paths"] = list(E.OPENED)
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"written {out.name} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
