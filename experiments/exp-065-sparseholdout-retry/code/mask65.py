"""exp-065-sparseholdout-retry: the hard support masks, the ten primary matched shams and the matched-null
proof (SPEC 1.5).  Label-free: it opens only this experiment's own `inputs65` projection.

For every threshold n* in {25, 50, 100, 200} it defines the primary test support separately for
each (family, airport): the columns with n_A(k) < n*.  A group with fewer than two columns is
EXCLUDED and zeroed in every arm that carries a prior; the test and any close exclude it.

The effective prior of an arm at cell (n*, c) is `table(k) * 1[k kept] * c/(c + n_A(k))` --- the
hard mask BEFORE the smooth weight, c = inf meaning weight 1.  Each of the ten primary matched
shams is a deterministic non-identity permutation, drawn once per (seed, family, airport, n*) and
applied to the EFFECTIVE masked-and-weighted true values of every cell, so `true` and every sham
have the identical test support and the identical effective-value multiset in every group and cell.

It proves those two identities, records the excluded singleton columns, the moved columns and the
moved S, I and V population rows by family, airport, threshold, c and seed, and decides the
matched-null row-difference requirement: every complete sham must move at least one S, one I and
one V population row at EVERY (n*, c) cell.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-065-sparseholdout-retry/code/mask65.py jul jan dec
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e65 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

MONTHS = ("S", "I", "V")


def one(r: str, overwrite: bool) -> dict:
    npz_p = E.out_path(E.WORK / f"masks65-{r}.npz")
    rec_p = E.out_path(E.ANALYSIS / f"mask65-{r}.json")
    NO.refuse_existing([npz_p, rec_p], overwrite)
    t0 = time.time()
    launch = E.hold_load(f"masks {r}")

    in_rec = E.load_json(E.ANALYSIS / f"inputs65-{r}.json")
    src = E.ROOT / in_rec["npz"]
    if E.sha_file(src) != in_rec["npz_sha256"]:
        raise SystemExit(f"{src.name} is not the frozen inputs65 archive")
    z = np.load(src)
    vocab = json.loads((E.ROOT / in_rec["vocab65_path"]).read_text())
    nlev = {f: int(in_rec["vocab_sizes"][f]) for f in E.FAMS}
    napt = nlev["apt"]
    off = E.fam_offsets(nlev)
    ncol = int(off[-1])
    col_apt = E.column_airport(vocab)
    col_fam = E.column_family(nlev)
    col_apt_direct = np.concatenate([
        np.array([vocab["apt"].index(str(v).split("|")[0])
                  if str(v).split("|")[0] in vocab["apt"] else -1 for v in vocab[f]],
                 dtype="int64") for f in E.PRIOR_FAMS])
    col_fam_direct = np.concatenate([np.full(nlev[f], i, dtype="int64")
                                     for i, f in enumerate(E.PRIOR_FAMS)])
    if not np.array_equal(col_apt, col_apt_direct) or not np.array_equal(col_fam, col_fam_direct):
        raise SystemExit("derived column indexes fail byte-for-byte proof")
    nA = np.concatenate([z[f"nA_{f}"] for f in E.PRIOR_FAMS]).astype("int64")
    tables = {a: np.concatenate([z[f"prior_{a}_{f}"] for f in E.PRIOR_FAMS])
              for a in ("true", "gmean", "donor", "shuffle")}

    nT, nC, nS = len(E.THRESHOLDS), len(E.C_GRID), len(E.SHAM_NAMES)
    keep = np.zeros((nT, ncol), dtype=bool)
    gid = np.zeros((nT, ncol), dtype="int64")
    eff = {a: np.zeros((nT, nC, ncol)) for a in tables}
    eff_sham = np.zeros((nS, nT, nC, ncol))
    perms: dict = {}
    seed_search: dict = {}
    excluded = []

    # Population column references are label-free and are needed to reject a seed before any
    # S, I, or V departure label is read.
    colg = {}
    rows_pop = {}
    for m in MONTHS:
        pop = z[f"pop_{m}"].astype(bool)
        rows_pop[m] = int(pop.sum())
        for i, f in enumerate(E.PRIOR_FAMS):
            c_ = z[f"code_{m}_{f}"][pop]
            colg[(m, f)] = np.where(c_ >= 0, c_ + off[i], -1)

    for i_n, ns in enumerate(E.THRESHOLDS):
        g, k = E.group_ids(col_fam, col_apt, nA, ns)
        gid[i_n], keep[i_n] = g, k
        below = nA < ns
        for gg in np.unique(g[g >= 0]):
            idx = np.flatnonzero(g == gg)
            if len(idx) < 2:
                excluded.append({"n_star": ns, "family": E.PRIOR_FAMS[int(gg // 10_000)],
                                 "airport": str(vocab["apt"][int(gg % 10_000)]),
                                 "columns": [str(vocab[E.PRIOR_FAMS[int(gg // 10_000)]]
                                                 [int(c - off[int(gg // 10_000)])]) for c in idx],
                                 "nonzero_true": int((tables["true"][idx] != 0).sum())})
        # columns below the threshold whose airport is missing (-1) are never in a group
        orphan = int((below & (col_apt < 0)).sum())
        if orphan:
            excluded.append({"n_star": ns, "family": "*", "airport": None,
                             "columns": [], "orphan_columns_without_an_airport": orphan})
        for i_c, c in enumerate(E.C_GRID):
            for a, tab in tables.items():
                eff[a][i_n, i_c] = E.effective(tab, k, nA, c)
        for gg in np.unique(g[g >= 0]):
            idx = np.flatnonzero(g == gg)
            if len(idx) < 2:
                continue
            fam, apt = int(gg // 10_000), int(gg % 10_000)
            chosen = set()
            for i_s, sham_name in enumerate(E.SHAM_NAMES):
                attempts = []
                p = None
                seed = None
                for candidate in E.SHAM_SEED_CANDIDATES:
                    if candidate in chosen:
                        attempts.append({"seed": candidate, "reason": "already chosen in group"})
                        continue
                    trial = E.sham_permutation(candidate, i_n, fam, apt, len(idx))
                    counts = {}
                    valid = True
                    for i_c, c in enumerate(E.C_GRID):
                        before = eff["true"][i_n, i_c, idx]
                        changed_cols = idx[before[trial] != before]
                        for m in MONTHS:
                            used = colg[(m, E.PRIOR_FAMS[fam])]
                            count = int(np.isin(used, changed_cols).sum())
                            counts[f"{m}|{'inf' if np.isinf(c) else int(c)}"] = count
                            valid &= count > 0
                    attempts.append({"seed": candidate,
                                     "reason": "valid" if valid else "no changed eligible row",
                                     "affected_rows": counts})
                    if valid:
                        seed, p = candidate, trial
                        break
                key = f"n={ns}|family={E.PRIOR_FAMS[fam]}|airport={vocab['apt'][apt]}|sham={sham_name:02d}"
                seed_search[key] = attempts
                if p is None:
                    raise SystemExit("matched null incomplete; open")
                chosen.add(seed)
                perms[(i_s, i_n, int(gg))] = p
                for i_c in range(nC):
                    eff_sham[i_s, i_n, i_c, idx] = eff["true"][i_n, i_c, idx][p]

    # ---- the two registered identities, in every group and every cell ------------------------
    ident = {"groups_checked": 0, "cells_checked": 0, "support_equal": True,
             "multiset_equal": True, "all_permutations_non_identity": True}
    for i_n in range(nT):
        for gg in np.unique(gid[i_n][gid[i_n] >= 0]):
            idx = np.flatnonzero(gid[i_n] == gg)
            if len(idx) < 2:
                continue
            ident["groups_checked"] += 1
            for i_s in range(nS):
                p = perms[(i_s, i_n, int(gg))]
                if np.array_equal(p, np.arange(len(idx))):
                    ident["all_permutations_non_identity"] = False
                for i_c in range(nC):
                    t = eff["true"][i_n, i_c, idx]
                    s = eff_sham[i_s, i_n, i_c, idx]
                    ident["cells_checked"] += 1
                    if not np.array_equal(np.sort(t), np.sort(s)):
                        ident["multiset_equal"] = False
                    if int((t != 0).sum()) != int((s != 0).sum()):
                        ident["support_equal"] = False
    # outside the kept groups every prior arm is exactly zero, so the support is identical there
    for i_n in range(nT):
        out = ~keep[i_n]
        for i_c in range(nC):
            if float(np.max(np.abs(eff["true"][i_n, i_c, out]), initial=0.0)) != 0.0:
                ident["support_equal"] = False
            if float(np.max(np.abs(eff_sham[:, i_n, i_c, out]), initial=0.0)) != 0.0:
                ident["support_equal"] = False
            for a in tables:
                if float(np.max(np.abs(eff[a][i_n, i_c, out]), initial=0.0)) != 0.0:
                    ident["support_equal"] = False
    if not (ident["support_equal"] and ident["multiset_equal"]
            and ident["all_permutations_non_identity"]):
        raise SystemExit(f"{r}: the matched-sham identities do not hold: {ident}")

    # ---- moved columns and moved population rows ----------------------------------------------
    aptrow = {}
    for m in MONTHS:
        pop = z[f"pop_{m}"].astype(bool)
        aptrow[m] = z[f"code_{m}_apt"][pop]

    moved_cols = np.zeros((3, napt, nT, nC, nS), dtype="int32")
    moved_rows = {m: np.zeros((3, napt, nT, nC, nS), dtype="int32") for m in MONTHS}
    cell_rows = {}
    for i_n, ns in enumerate(E.THRESHOLDS):
        for i_c, c in enumerate(E.C_GRID):
            for i_s, sham_name in enumerate(E.SHAM_NAMES):
                d = eff_sham[i_s, i_n, i_c] != eff["true"][i_n, i_c]
                for i, f in enumerate(E.PRIOR_FAMS):
                    sl = slice(off[i], off[i + 1])
                    a_ = np.where(col_apt[sl] >= 0, col_apt[sl], napt)
                    cnt = np.bincount(a_[d[sl]], minlength=napt + 1)[:napt]
                    moved_cols[i, :, i_n, i_c, i_s] = cnt
                anyrow = {}
                for m in MONTHS:
                    tot = np.zeros(rows_pop[m], dtype=bool)
                    for i, f in enumerate(E.PRIOR_FAMS):
                        g_ = colg[(m, f)]
                        hit = (g_ >= 0) & d[np.maximum(g_, 0)]
                        moved_rows[m][i, :, i_n, i_c, i_s] = np.bincount(
                            aptrow[m][hit], minlength=napt)
                        tot |= hit
                    anyrow[m] = int(tot.sum())
                key = f"n={ns},c={'inf' if np.isinf(c) else int(c)},sham={sham_name:02d}"
                cell_rows[key] = {"moved_columns": int(d.sum()),
                                  "moved_columns_by_family":
                                      {f: int(d[off[i]:off[i + 1]].sum())
                                       for i, f in enumerate(E.PRIOR_FAMS)},
                                  "moved_population_rows": {m: anyrow[m] for m in MONTHS}}

    failing = sorted(k for k, v in cell_rows.items()
                     if min(v["moved_population_rows"].values()) < 1)
    resolved = not failing

    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r,
           "months": in_rec["months"], "launch": launch, "code_sha256": E.code_shas(),
           "inputs": {"record": f"experiments/exp-065-sparseholdout-retry/analysis/inputs65-{r}.json",
                      "npz_sha256": in_rec["npz_sha256"], "freeze_unix": in_rec["freeze_unix"]},
           "grid": {"thresholds": list(E.THRESHOLDS),
                    "c": [E.jsonable(c) for c in E.C_GRID],
                    "sham_names": list(E.SHAM_NAMES), "seed_candidates": [1, 100000],
                    "sham_seed_base": E.SHAM_SEED_BASE,
                    "inequality": "n_A(k) < n_star, strict", "mask_before_weight": True},
           "columns": {"total": ncol, "by_family": {f: nlev[f] for f in E.PRIOR_FAMS},
                       "airports": napt},
           "support": {str(ns): {"columns_below": int((nA < ns).sum()),
                                 "columns_in_kept_groups": int(keep[i].sum()),
                                 "excluded_columns": int((nA < ns).sum()) - int(keep[i].sum()),
                                 "kept_groups": int(len(np.unique(gid[i][gid[i] >= 0]))
                                                    - sum(1 for e in excluded
                                                          if e["n_star"] == ns and e["family"] != "*")),
                                 "nonzero_true": {f: int((eff["true"][i, nC - 1,
                                                          off[j]:off[j + 1]] != 0).sum())
                                                  for j, f in enumerate(E.PRIOR_FAMS)}}
                       for i, ns in enumerate(E.THRESHOLDS)},
           "excluded_singleton_groups": excluded,
           "identities": ident,
           "matched_null": {"requirement": ("every complete sham moves at least one S, one I and "
                                            "one V population row at every (n*, c) cell"),
                            "cells": len(cell_rows), "failing_cells": failing,
                            "resolved": bool(resolved)},
           "seed_search": seed_search,
           "moved_by_cell": cell_rows,
           "population_rows": rows_pop,
           "detail_axes": ("family (stand, rwy, op), airport (vocab order), threshold "
                           "(25, 50, 100, 200), c (10, 30, 100, 300, 1000, inf), seed (1..10)")}

    arrays = {"keep": keep, "gid": gid, "eff_sham": eff_sham,
              "moved_cols": moved_cols, "nA": nA,
              "thresholds": np.array(E.THRESHOLDS),
              "c_grid": np.array([1e30 if np.isinf(c) else c for c in E.C_GRID])}
    arrays |= {f"eff_{a}": eff[a] for a in eff}
    arrays |= {f"moved_rows_{m}": moved_rows[m] for m in MONTHS}
    rec["array_sha256"] = {k: E.sha_arr(v) for k, v in sorted(arrays.items())}

    NO.refuse_existing([npz_p, rec_p], overwrite)
    np.savez(npz_p, **arrays)
    rec["npz"] = str(npz_p.relative_to(E.ROOT))
    rec["npz_sha256"] = E.sha_file(npz_p)
    rec["label_reads"] = E.LABEL_READS
    assert not E.LABEL_READS, "the mask job read a label"
    rec["opened_paths"] = list(E.OPENED)
    rec["freeze_unix"] = time.time()
    rec["freeze_utc"] = E.utc()
    rec["seconds"] = round(time.time() - t0, 1)
    rec_p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    mn = min(min(v["moved_population_rows"].values()) for v in cell_rows.values())
    print(f"masks {r}: groups {ident['groups_checked']}, cells {ident['cells_checked']}, "
          f"excluded {len(excluded)}, matched null resolved {resolved} "
          f"(smallest moved population rows over the 240 cells: {mn}) ({rec['seconds']:.0f}s)",
          flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotations", nargs="+", choices=E.ORDER)
    a = ap.parse_args()
    paths = []
    for r in a.rotations:
        paths += [E.WORK / f"masks65-{r}.npz", E.ANALYSIS / f"mask65-{r}.json"]
    NO.refuse_existing(paths, False)
    print("guard: no existing artifact at " + ", ".join(p.name for p in paths), flush=True)
    for r in a.rotations:
        one(r, False)


if __name__ == "__main__":
    main()
