"""exp-056-q12advocate: fit, select and freeze the corrections of one rotation (SPEC 3.3-3.6).

Reads the frozen inputs (sha-checked), then the departure labels of S and I only. For every arm
(zero, the literal sham, the exp-054 tables true/const/gmean/donor/shuffle, and the ten
permuted-prior shams) it picks (lam, beta) on the two inner directions (fit S -> score I and
fit I -> score S), fits the correction on S with the pick, and writes the correction for V
under five designs: main (S residuals of the isolated base), the matched-size exposure pair
(S residuals of the exposed base and of the isolated base at the same training size, same
pick), and the time-origin pair (matched-size S and I, same pick).
V's labels are not read here. One rotation per process. `--iteration 2` is route A (SPEC 3.9,
4): every arm's prior is weighted by c / (c + n_A(key)), with c chosen jointly with (lam, beta)
on the same inner directions.

    .venv/bin/python experiments/exp-056-q12advocate/code/fit.py jul
    .venv/bin/python experiments/exp-056-q12advocate/code/fit.py jul --iteration 2
    .venv/bin/python experiments/exp-056-q12advocate/code/fit.py jul --dry --scenario null
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

DESIGNS = ("main", "expo", "expo_ref", "originS", "originI")


def arm_names() -> list:
    return ["zero", "literal_sham"] + list(E.TABLE_ARMS) + [f"sham{s:02d}" for s in E.SHAM_SEEDS]


def prior_of(z, arm: str, nlev: dict) -> np.ndarray:
    if arm in ("zero", "literal_sham"):
        return np.zeros(sum(nlev[f] for f in E.PRIOR_FAMS))
    return E.prior_vector({f: z[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS}, nlev)


SCENARIOS = ("null", "planted", "planted10", "keys", "support")


def synthetic_labels(z, scenario: str, k: str, seed: int) -> np.ndarray:
    """Dry run only. y = base + airport-month offset + gamma * (true unit prior)
    [+ persistent stand effects unrelated to the prior, scenario `keys`] + t3 noise. Scenario
    `support` plants the prior only through route A's weights at c = 100 (sparse keys)."""
    gamma = {"null": 0.0, "planted": 0.3, "planted10": 0.1, "keys": 0.0, "support": 0.6}[scenario]
    rng = np.random.default_rng(seed)
    codes = {f: z[f"code_{k}_{f}"] for f in E.FAMS}
    q = np.zeros(len(codes["apt"]))
    for f in E.PRIOR_FAMS:
        c = codes[f]
        w = (100.0 / (100.0 + z[f"nA_{f}"].astype("float64"))) if scenario == "support" else 1.0
        q[c >= 0] += (z[f"prior_true_{f}"] * w)[c[c >= 0]]
    napt = int(codes["apt"].max()) + 1
    off = rng.normal(0, 30, napt)[np.maximum(codes["apt"], 0)]
    eps = 250.0 * rng.standard_t(3, len(q))
    extra = np.zeros(len(q))
    if scenario == "keys":
        eff = np.random.default_rng(777).normal(0, 40, len(z["prior_true_stand"]))
        c = codes["stand"]
        extra[c >= 0] = eff[c[c >= 0]]
    return np.maximum(1.0, z[f"p_{k}"] + off + gamma * q + extra + eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", choices=E.ORDER)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--scenario", choices=SCENARIOS, default=None)
    ap.add_argument("--iteration", type=int, choices=(1, 2), default=1,
                    help="2 = route A, the support-aware prior (SPEC 3.9 and 4)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.dry != (a.scenario is not None):
        raise SystemExit("--dry needs --scenario, and --scenario needs --dry")
    r = a.rotation
    tag = f"dry{E.DRY_VERSION}-{a.scenario}-" if a.dry else ""
    it = "it2-" if a.iteration == 2 else ""
    base = E.WORK / "dry" if a.dry else E.WORK
    in_rec_p = E.ANALYSIS / (f"dry{E.DRY_VERSION}-inputs-{r}.json" if a.dry else f"inputs-{r}.json")
    out_npz = base / f"{tag}{it}pred-{r}.npz"
    out_rec = E.ANALYSIS / f"{tag}{it}fit-{r}.json"
    NO.refuse_existing([out_npz, out_rec], a.overwrite)
    t0 = time.time()
    launch = E.hold_load(f"fit {r}")

    in_rec = E.load_json(in_rec_p)
    npz = E.ROOT / in_rec["npz"]
    if E.sha_file(npz) != in_rec["npz_sha256"]:
        raise SystemExit(f"{npz.name} is not the frozen input")
    z = np.load(npz)
    V, S, I, M = E.ROT[r]
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    ncol = sum(nlev[f] for f in E.PRIOR_FAMS)

    # ---- the only label reads of this job: S and I -----------------------------------------
    if a.dry:
        y = {"S": synthetic_labels(z, a.scenario, "S", 101 + E.ORDER.index(r)),
             "I": synthetic_labels(z, a.scenario, "I", 201 + E.ORDER.index(r))}
    else:
        y = {"S": E.dep_labels(S, z["ids_S"], "correction fit and inner selection"),
             "I": E.dep_labels(I, z["ids_I"], "inner selection and time-origin control")}
        for rd in E.LABEL_READS:
            if rd["unix"] <= in_rec["freeze_unix"]:
                raise SystemExit("a label was read before the inputs were frozen")
        assert {rd["month"] for rd in E.LABEL_READS} == {S, I}
    p = {k: z[f"p_{k}"] for k in ("V", "S", "I")}
    pop = {k: z[f"pop_{k}"] for k in ("V", "S", "I")}
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("V", "S", "I")}
    Xall = {k: E.design(codes[k], nlev) for k in ("V", "S", "I")}

    def fit_rows(k, extra=None):
        m = pop[k] & np.isfinite(y[k]) & (y[k] > 0)
        return m if extra is None else (m & extra)

    def resid(k, rows, pbase=None):
        """Clipped residual of the fit rows, demeaned by airport over those rows (SPEC 3.3)."""
        pb = p[k] if pbase is None else pbase
        rr = np.clip(y[k][rows] - pb[rows], -E.CLIP, E.CLIP)
        return E.demean_by_airport(rr, codes[k]["apt"][rows])

    # factorizations depend on (rows, lam) only, the data part t0 on the residual only
    rowsets = {"S": fit_rows("S"), "I": fit_rows("I"),
               "Sthin": fit_rows("S", z["thin_S"]), "Ithin": fit_rows("I", z["thin_I"])}
    Xfit = {"S": E.design(codes["S"], nlev, rowsets["S"]), "I": E.design(codes["I"], nlev, rowsets["I"]),
            "Sthin": E.design(codes["S"], nlev, rowsets["Sthin"]),
            "Ithin": E.design(codes["I"], nlev, rowsets["Ithin"])}
    rfit = {"S": resid("S", rowsets["S"]), "I": resid("I", rowsets["I"]),
            "Sexpo": resid("S", rowsets["S"], z["p_Sexpo"]),
            "Sref": resid("S", rowsets["S"], z["p_Sref"]),
            "Sthin": resid("S", rowsets["Sthin"]), "Ithin": resid("I", rowsets["Ithin"])}
    from scipy.sparse.linalg import splu
    import scipy.sparse as sp
    lu_cache = {}

    def lu(which, lam):
        key = (which, lam)
        if key not in lu_cache:
            X = Xfit[which]
            lu_cache[key] = splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
        return lu_cache[key]

    def theta(which, rkey, pv, lam):
        if np.isinf(lam):
            return np.zeros_like(pv), pv.copy()
        f = lu(which, lam)
        t0 = f.solve(np.asarray(Xfit[which].T @ rfit[rkey], dtype="float64"))
        t1 = f.solve(lam * pv) if np.any(pv) else np.zeros_like(pv)
        return t0, t1

    t0_cache = {}

    def data_part(which, rkey, lam):
        key = (which, rkey, lam)
        if key not in t0_cache:
            t0_cache[key] = theta(which, rkey, np.zeros(ncol), lam)[0]
        return t0_cache[key]

    def slope_part(which, pv, lam):
        if np.isinf(lam):
            return pv.copy()
        return lu(which, lam).solve(lam * pv) if np.any(pv) else np.zeros_like(pv)

    out_arrays, rec_arms = {}, {}
    label_free_check = {}
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
    for arm in arm_names():
        pv0 = prior_of(z, arm, nlev)
        # iteration 2 (route A): every arm with a prior gets the same support weights and the c
        # grid; an arm without a prior has nothing to weight and keeps iteration 1's grid
        cs = E.C_GRID if (a.iteration == 2 and np.any(pv0)) else (None,)
        pvc = {c: (pv0 if c is None else pv0 * E.support_weights(nA, c)) for c in cs}
        parts_c = {}
        for c in cs:
            parts_c[c] = {"S>I": {}, "I>S": {}}
            for lam in E.LAMBDAS:
                for d, fit_k, sc in (("S>I", "S", "I"), ("I>S", "I", "S")):
                    t0_ = data_part(fit_k, fit_k, lam)
                    t1_ = slope_part(fit_k, pvc[c], lam)
                    parts_c[c][d][lam] = (Xall[sc] @ t0_, Xall[sc] @ t1_)
        if cs == (None,):
            best, table = E.select(parts_c[None], y, p, pop)
            best["c"] = None
        else:
            best, table = E.select_c(parts_c, y, p, pop)
        lam, beta = best["lam"], best["beta"]
        pv, parts = pvc[best["c"]], parts_c[best["c"]]
        preds = {}
        th = data_part("S", "S", lam) + beta * slope_part("S", pv, lam)
        preds["main"] = Xall["V"] @ th
        for dsg, rkey in (("expo", "Sexpo"), ("expo_ref", "Sref")):
            preds[dsg] = Xall["V"] @ (data_part("S", rkey, lam) + beta * slope_part("S", pv, lam))
        tho_s = data_part("Sthin", "Sthin", lam) + beta * slope_part("Sthin", pv, lam)
        tho_i = data_part("Ithin", "Ithin", lam) + beta * slope_part("Ithin", pv, lam)
        preds["originS"] = Xall["V"] @ tho_s
        preds["originI"] = Xall["V"] @ tho_i
        # inner-month corrections at the pick (for the inner alignment diagnostic)
        h_inner_I = parts["S>I"][lam][0] + beta * parts["S>I"][lam][1]
        for dsg in DESIGNS:
            out_arrays[f"h_{arm}_{dsg}"] = preds[dsg]
        out_arrays[f"hI_{arm}"] = h_inner_I
        rec_arms[arm] = {"lam": lam if np.isfinite(lam) else "inf", "beta": beta,
                         "inner_sse_at_pick": best["sse"],
                         "inner_grid": [{"lam": (g["lam"] if np.isfinite(g["lam"]) else "inf"),
                                         "beta": g["beta"], "sse": g["sse"]} for g in table],
                         "h_sha256": {dsg: E.sha_arr(preds[dsg]) for dsg in DESIGNS}}
        if a.iteration == 2:
            rec_arms[arm]["c"] = best["c"]
            if best["c"] is not None:
                for g_rec, g in zip(rec_arms[arm]["inner_grid"], table):
                    g_rec["c"] = g["c"]
        c_txt = "" if a.iteration == 1 else f"c={best['c']} "
        print(f"  {r} {arm:13s} lam={rec_arms[arm]['lam']!s:>6} beta={beta:.1f} "
              f"{c_txt}({time.time() - t0:.0f}s)", flush=True)

    # the literal sham and const are the zero arm by construction: prove it on the bytes
    for arm in ("literal_sham", "const"):
        for dsg in DESIGNS:
            if not np.array_equal(out_arrays[f"h_{arm}_{dsg}"], out_arrays[f"h_zero_{dsg}"]):
                raise SystemExit(f"{arm} differs from zero in {dsg}: the degeneracy proof fails")
        label_free_check[arm] = "bitwise equal to zero in all four designs"

    # inner residual alignment of the unit true prior on I (fit S -> score I), zero arm residual
    q = {}
    for k in ("I", "V"):
        qq = np.zeros(len(codes[k]["apt"]))
        for f in E.PRIOR_FAMS:
            c = codes[k][f]
            qq[c >= 0] += z[f"prior_true_{f}"][c[c >= 0]]
        q[k] = qq
    mI = pop["I"] & np.isfinite(y["I"])
    eI = y["I"][mI] - np.maximum(E.MIN_PRED, p["I"][mI] + out_arrays["hI_zero"][mI])
    qI = q["I"][mI]
    inner_align = {"rows": int(mI.sum()), "alignment": float(np.sum(eI * qI) / np.sum(qI * qI)),
                   "corr": float(np.corrcoef(eI, qI)[0, 1])}

    NO.refuse_existing([out_npz, out_rec], a.overwrite)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out_arrays)
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "dry": a.dry,
           "scenario": a.scenario, "months": {"V": V, "S": S, "I": I, "M": M},
           "inputs": {"record": str(in_rec_p.relative_to(E.ROOT)), "npz_sha256": in_rec["npz_sha256"],
                      "freeze_unix": in_rec["freeze_unix"]},
           "iteration": a.iteration,
           "c_grid": list(E.C_GRID) if a.iteration == 2 else None,
           "launch": launch, "code_sha256": E.code_shas(),
           "grid": {"lambdas": [x if np.isfinite(x) else "inf" for x in E.LAMBDAS],
                    "betas": list(E.BETAS)},
           "fit_rows": {k: int(v.sum()) for k, v in rowsets.items()},
           "label_reads": E.LABEL_READS, "arms": rec_arms, "degeneracy": label_free_check,
           "inner_alignment_true_unit_prior_on_I": inner_align,
           "pred_npz": str(out_npz.relative_to(E.ROOT)), "pred_npz_sha256": E.sha_file(out_npz),
           "array_sha256": {k: E.sha_arr(v) for k, v in sorted(out_arrays.items())},
           "freeze_unix": time.time(), "freeze_utc": E.utc(),
           "seconds": round(time.time() - t0, 1)}
    out_rec.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"frozen {r} predictions {out_npz.name} {rec['pred_npz_sha256'][:16]} "
          f"({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
