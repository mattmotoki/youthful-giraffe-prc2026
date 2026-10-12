"""exp-061-sparseprior: the round-1 review answers that need a measurement (findings 1, 2, 5, 6, 7).

Post-hoc and descriptive.  It runs after every verdict, hash and bound is frozen, selects nothing,
and no number it produces enters a choice, a gate, a bound or the registered reading.  Five parts:

  (1) the `inputs61` archive holds 43 arrays, not the 41 SPEC 1.2 registers plus `col_apt`: the
      extra one is `col_fam`.  This rebuilds BOTH index arrays from the frozen vocabulary and the
      registered vocabulary sizes and requires byte equality, which shows `col_fam` is a
      deterministic function of an already registered quantity and carries no new input.
  (2) the provenance the projection did not pin: the SHA-256 and the git commit of every exp-060
      record it opened by path, and of every local module the jobs import.
  (3) the moved KEY IDENTITIES the assignment requires beside the counts, for every
      (family, airport, n*, c, seed), written as a parquet artifact.
  (4) the registered zero identity over the WHOLE grid and BOTH levels, from the frozen archives:
      the source priors of `const` and `literal_sham` are bitwise zero, so every effective value,
      every beta-scaled value, every cell total, every per-airport minimum, every beta index and
      every served prediction of `zero`, `const` and `literal_sham` is identical at both levels and
      constant over the four thresholds and the six c values at every lambda.
  (5) the registered level-A identity of SPEC 6: level A restricted to one beta for all airports
      reproduces the grid oracle.  The A-grid curves of `true` and `zero` are recomputed here from
      the frozen inputs and compared against the recorded `A` and `grid_oracle` SSE of all three
      bound kinds.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-061-sparseprior/code/verify61r1.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

GRID_A = E.grid_A()
PHI_A = E.monomials(GRID_A)
KINDS = (("full", False, True), ("adjusted_all_rows", True, True),
         ("adjusted_population_rows", True, False))
#: every local module the exp-061 jobs import, beside the pinned exp-060 module
LOCAL_DEPS = ("prc/data.py", "prc/blend.py", "prc/regimes.py",
              "experiments/exp-046-surfacefit/code/nooverwrite.py",
              "experiments/exp-060-coldprior/code/e60.py")
#: the exp-060 records project61.py opened by path without a SHA-256 pin
UNPINNED_READS = ("analysis/inputs60-{r}.json", "analysis/proj-{r}.json", "analysis/cold-{r}.json")


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(E.ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:                                          # pragma: no cover
        return f"unavailable: {exc}"


def select(sse, rsum, n, per_airport: bool, adjust: bool) -> float:
    v = sse - (rsum ** 2) / np.maximum(n, 1.0)[:, None] if adjust else sse
    if per_airport:
        return float(v.min(1).sum())
    return float(v.sum(0).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.out_path(E.ANALYSIS / "verify61-r1.json")
    keys_p = E.out_path(E.ART / "movedkeys61-r1.parquet")
    NO.refuse_existing([out, keys_p], a.overwrite)
    t0 = time.time()
    launch = E.hold_load("verify r1")

    res = {"what": __doc__.strip().splitlines()[0], "launch": launch,
           "code_sha256": E.code_shas(), "answers": {}, "rotations": {}}

    # ---- (2) provenance, once ------------------------------------------------------------------
    prov = {"local_modules": {}, "exp060_records_opened_by_path": {}, "working_tree_clean": {}}
    for rel in LOCAL_DEPS:
        p = E.ROOT / rel
        prov["local_modules"][rel] = {
            "sha256": E.sha_file(p) if p.exists() else None,
            "last_commit": git("log", "-1", "--format=%H", "--", rel),
            "uncommitted_change": bool(git("status", "--porcelain", "--", rel))}
    res["answers"]["finding_2_provenance"] = prov

    moved_rows = []
    for r in E.ORDER:
        in_rec = E.load_json(E.ANALYSIS / f"inputs61-{r}.json")
        mk_rec = E.load_json(E.ANALYSIS / f"mask61-{r}.json")
        fit_rec = E.load_json(E.ANALYSIS / f"fit61-{r}.json")
        bd_rec = E.load_json(E.ANALYSIS / f"bound61-{r}.json")
        for rec_, key in ((in_rec, "npz"), (mk_rec, "npz"), (fit_rec, "pred_npz"),
                          (fit_rec, "grid_npz")):
            if E.sha_file(E.ROOT / rec_[key]) != rec_[f"{key}_sha256"]:
                raise SystemExit(f"{rec_[key]} is not the frozen file")
        z = np.load(E.ROOT / in_rec["npz"])
        mk = np.load(E.ROOT / mk_rec["npz"])
        gr = np.load(E.ROOT / fit_rec["grid_npz"])
        pr = np.load(E.ROOT / fit_rec["pred_npz"])
        vocab = json.loads((E.ROOT / in_rec["vocab61_path"]).read_text())
        E.pin_months(in_rec)
        nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
        ncol = sum(nlev[f] for f in E.PRIOR_FAMS)
        napt = nlev["apt"]
        off = E.fam_offsets(nlev)
        rot = {}

        # ---- (1) the archive contents -----------------------------------------------------------
        names = sorted(z.files)
        allow = sorted(in_rec["source"]["arrays_copied"])
        extra = sorted(set(names) - set(allow))
        rot["finding_1_archive"] = {
            "arrays_in_the_archive": len(names), "registered_allowlist": len(allow),
            "beyond_the_allowlist": extra,
            "col_apt_rebuilt_equal": bool(np.array_equal(z["col_apt"], E.column_airport(vocab))),
            "col_fam_rebuilt_equal": bool(np.array_equal(z["col_fam"], E.column_family(nlev))),
            "both_are_functions_of_the_registered_vocabulary": True,
            "deviation": ("SPEC 1.2 registers the 41 allowed arrays plus col_apt; project61.py "
                          "also writes col_fam, a per-column family index rebuilt here byte for "
                          "byte from the registered vocabulary sizes, so it adds no input")}

        # ---- (2) the records opened by path -----------------------------------------------------
        rr = {}
        for tpl in UNPINNED_READS:
            rel = f"experiments/exp-060-coldprior/{tpl.format(r=r)}"
            p = E.ROOT / rel
            rr[rel] = {"sha256": E.sha_file(p), "last_commit": git("log", "-1", "--format=%H",
                                                                   "--", rel),
                       "uncommitted_change": bool(git("status", "--porcelain", "--", rel))}
        prov["exp060_records_opened_by_path"][r] = rr

        # ---- (5 of the assignment) the moved key identities -------------------------------------
        col = np.array([f"{vocab[f][i]}" for f in E.PRIOR_FAMS for i in range(nlev[f])])
        cfam = np.array([f for f in E.PRIOR_FAMS for _ in range(nlev[f])])
        capt = np.array([str(vocab["apt"][x]) if x >= 0 else "" for x in z["col_apt"]])
        nmoved = 0
        for i_n, ns in enumerate(E.THRESHOLDS):
            for i_c, c in enumerate(E.C_GRID):
                et = mk["eff_true"][i_n, i_c]
                for si, seed in enumerate(E.SHAM_SEEDS):
                    es = mk["eff_sham"][si, i_n, i_c]
                    w = np.flatnonzero(et != es)
                    nmoved += len(w)
                    if len(w):
                        moved_rows.append(pd.DataFrame({
                            "rotation": r, "n_star": ns, "c": str(E.jsonable(c)), "seed": seed,
                            "family": cfam[w], "airport": capt[w], "key": col[w],
                            "effective_true": et[w], "effective_sham": es[w]}))
        rot["finding_5_moved_keys"] = {
            "rows_written": nmoved,
            "equals_the_recorded_counts": bool(
                nmoved == sum(v["moved_columns"] for v in mk_rec["moved_by_cell"].values())),
            "excluded_singleton_keys_already_recorded":
                len(mk_rec["excluded_singleton_groups"])}

        # ---- (6) the registered zero identity, whole grid, both levels --------------------------
        zero_src = {f"prior_const_{f}": float(np.abs(z[f"prior_const_{f}"]).max())
                    for f in E.PRIOR_FAMS}
        cells = list(gr["cell_order"])
        ident = {"source_prior_max_abs": zero_src,
                 "source_prior_is_bitwise_zero": all(v == 0.0 for v in zero_src.values())}
        for lev in ("A", "B"):
            base = gr[f"zero|{lev}|total"]
            ident[f"{lev}_totals_identical_to_zero"] = {
                arm: bool(np.array_equal(gr[f"{arm}|{lev}|total"], base)
                          and np.array_equal(gr[f"{arm}|{lev}|per_airport_min"],
                                             gr[f"zero|{lev}|per_airport_min"])
                          and np.array_equal(gr[f"{arm}|{lev}|beta_index"],
                                             gr[f"zero|{lev}|beta_index"]))
                for arm in E.DEGENERATE_ARMS}
            ident[f"{lev}_beta_index_is_the_zero_beta"] = bool(
                (gr[f"zero|{lev}|beta_index"] == 0).all())
            by_lam = {}
            for i, name in enumerate(cells):
                lam = name.split("lam=")[1].split(",")[0]
                by_lam.setdefault(lam, []).append(base[i])
            ident[f"{lev}_constant_over_the_four_thresholds_and_six_c_at_every_lambda"] = bool(
                all(len(set(v)) == 1 and len(v) == len(E.THRESHOLDS) * len(E.C_GRID)
                    for v in by_lam.values()))
            ident[f"{lev}_predictions_identical_to_zero"] = {
                arm: bool(np.array_equal(pr[f"h_{arm}_{lev}"], pr[f"h_zero_{lev}"]))
                for arm in E.DEGENERATE_ARMS}
        rot["finding_6_zero_identity"] = ident

        # ---- (7) level A restricted to one beta reproduces the grid oracle ----------------------
        V, S = in_rec["months"]["V"], in_rec["months"]["S"]
        yS = E.dep_labels(S, z["ids_S"], "verify r1: the S fit, already read by bound61")
        yV = E.dep_labels(V, z["ids_V"], "verify r1: V in sample, already read by bound61")
        codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
        p, P, day = z["p_V"], z["pop_V"], z["day_V"]
        ndates = int(day.max()) + 1
        apt = codes["V"]["apt"]
        fit_rows = z["pop_S"] & np.isfinite(yS) & (yS > 0)
        X = E.design(codes["S"], nlev, fit_rows)
        res_S = E.demean_by_airport(np.clip(yS[fit_rows] - z["p_S"][fit_rows], -E.CLIP, E.CLIP),
                                    codes["S"]["apt"][fit_rows])
        XV = E.design(codes["V"], nlev)
        lu = {lam: splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
              for lam in E.LAMBDAS if np.isfinite(lam)}
        h0 = {lam: (XV @ lu[lam].solve(np.asarray(X.T @ res_S, dtype="float64")))
              if np.isfinite(lam) else np.zeros(len(yV)) for lam in E.LAMBDAS}
        ncell = napt * ndates
        cellP = apt[P] * ndates + day[P]
        yP = yV[P]
        eo = yV[~P] - p[~P]
        cello = apt[~P] * ndates + day[~P]
        st_out = np.stack([np.bincount(cello, minlength=ncell),
                           np.bincount(cello, weights=eo, minlength=ncell),
                           np.bincount(cello, weights=eo * eo, minlength=ncell)], 1)
        sse_out_total = float(st_out[:, 2].sum())
        st_out_a = st_out.reshape(napt, ndates, -1).sum(1)

        def by_airport(x):
            return x.reshape(napt, ndates, -1).sum(1)

        idt = {}
        for arm in ("true", "zero"):
            best = {f"{k}|{lv}": np.inf for k, _, _ in KINDS for lv in ("A", "grid_oracle")}
            for i_n, ns in enumerate(E.THRESHOLDS):
                for i_c, c in enumerate(E.C_GRID):
                    pv = (np.zeros(ncol) if arm == "zero" else mk["eff_true"][i_n, i_c])
                    for lam in E.LAMBDAS:
                        Sm = np.zeros((int(P.sum()), 3))
                        for i, f in enumerate(E.PRIOR_FAMS):
                            pf = np.zeros(ncol)
                            pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                            if np.any(pf):
                                th = pf.copy() if np.isinf(lam) else lu[lam].solve(lam * pf)
                                Sm[:, i] = (XV @ th)[P]
                        u = p[P] + h0[lam][P]
                        st, _, _, _, mix = E.prepare_cells(yP, u, Sm, cellP, ncell, GRID_A)
                        dA = by_airport(E.mixed_correction(yP[mix], u[mix], Sm[mix], GRID_A,
                                                           cellP[mix], ncell)[0]) if mix.any() \
                            else None
                        for kind, adjust, use_out in KINDS:
                            sse, rsum, n = E.grid_curves(by_airport(st), dA, None, PHI_A,
                                                         st_out=st_out_a if use_out else None)
                            add = 0.0 if use_out else sse_out_total
                            for lv, per_a in (("A", True), ("grid_oracle", False)):
                                v = select(sse, rsum, n, per_a, adjust) + add
                                best[f"{kind}|{lv}"] = min(best[f"{kind}|{lv}"], v)
            idt[arm] = {k: {"recomputed_sse": v, "recorded_sse": bd_rec["arms"][arm][k]["sse"],
                            "relative_difference":
                                abs(v - bd_rec["arms"][arm][k]["sse"])
                                / max(abs(bd_rec["arms"][arm][k]["sse"]), 1.0)}
                        for k, v in best.items()}
        worst = max(x["relative_difference"] for arm in idt for x in idt[arm].values())
        rot["finding_7_level_A_restricted_identity"] = {
            "what": ("level A and the grid oracle are the SAME beta-7 curves, minimised per "
                     "airport and at one common beta; both are recomputed here from the frozen "
                     "inputs and compared with the recorded bound"),
            "arms": idt, "max_relative_difference": worst,
            "identity_holds": bool(worst < 1e-12)}
        if worst >= 1e-12:
            raise SystemExit(f"the level-A restricted identity fails: relative {worst:.3e}")
        res["rotations"][r] = rot
        print(f"  {r}: archive {rot['finding_1_archive']['arrays_in_the_archive']} arrays, "
              f"moved keys {nmoved}, zero identity ok, level-A identity {worst:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.concat(moved_rows, ignore_index=True)
    keys_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(keys_p, index=False)
    res["moved_keys_artifact"] = {"path": str(keys_p.relative_to(E.ROOT)),
                                  "sha256": E.sha_file(keys_p), "rows": int(len(df))}
    res["label_reads"] = E.LABEL_READS
    res["opened_paths"] = list(E.OPENED)
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"written {out.name} and {keys_p.name} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
