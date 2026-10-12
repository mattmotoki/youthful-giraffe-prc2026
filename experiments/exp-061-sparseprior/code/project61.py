"""exp-061-sparseprior: the positive `inputs61` projection (SPEC 1.2).  Label-free, one process,
run before any S, I or V label read.

From each hash-pinned exp-060 INPUT archive it takes FORTY-ONE arrays and nothing else:

    ids_{V,S,I} pop_{V,S,I} p_{V,S,I} day_V nonm_V          (the eleven exp-060 projected)
    code_{V,S,I}_{apt,stand,rwy,op}                          (the twelve codes)
    nA_{stand,rwy,op}                                        (the base-month departure counts)
    prior_{true,const,gmean,donor,shuffle}_{stand,rwy,op}    (the five allowed priors)

plus the vocabulary and the airport mapping.  Every other array is refused by name, which is
exactly exp-060's thirty `prior_sham*` arrays: recorded rule forbids them in any choice, gate or bound here.

Before it writes anything it must pass, byte for byte:
  * the twelve codes, the ordered ids, the population and the UTC-date index rebuilt from the raw
    month files with exp-060's named label-free column list;
  * the fifteen allowed priors rebuilt from the frozen cold table;
  * the three nA arrays rebuilt from the nine A months' label-free key columns;
and it reproduces exp-060's projection allowlist and refusal list, its three rotations, common
base, population, cold-table hashes and support counts, and the label-free half of the
literal-sham identity.  Every raw month file is hashed and the nine A-month hashes must equal the
cold record's.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-061-sparseprior/code/project61.py jul jan dec
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

BASE11 = ("ids_V", "ids_S", "ids_I", "pop_V", "pop_S", "pop_I", "p_V", "p_S", "p_I",
          "day_V", "nonm_V")
CODES12 = tuple(f"code_{k}_{f}" for k in ("V", "S", "I") for f in E.FAMS)
NA3 = tuple(f"nA_{f}" for f in E.PRIOR_FAMS)
PRIORS15 = tuple(f"prior_{a}_{f}" for a in E.TABLE_ARMS for f in E.PRIOR_FAMS)
ALLOW = BASE11 + CODES12 + NA3 + PRIORS15

_COUNTS: dict = {}
_MONTH_SHA: dict = {}


def month_counts(m: str) -> dict:
    if m not in _COUNTS:
        _COUNTS[m] = E.month_key_counts(m)
    return _COUNTS[m]


def month_sha(m: str) -> str:
    if m not in _MONTH_SHA:
        _MONTH_SHA[m] = E.sha_file(E.D.month_file(m))
    return _MONTH_SHA[m]


def one(r: str, overwrite: bool) -> dict:
    npz_p = E.out_path(E.WORK / f"inputs61-{r}.npz")
    vocab_p = E.out_path(E.WORK / f"vocab61-{r}.json")
    rec_p = E.out_path(E.ANALYSIS / f"inputs61-{r}.json")
    NO.refuse_existing([npz_p, vocab_p, rec_p], overwrite)
    t0 = time.time()
    launch = E.hold_load(f"projection {r}")
    V, S, I, _ = E.ROT[r]
    A = E.cold_months(r)

    # ---- exp-060's label-free records (no result, row-loss, prediction, gate or bound) --------
    in60_p = E.guard_open(E.E60EXP / f"analysis/inputs60-{r}.json", "the pinned input archive record")
    in60 = E.load_json(in60_p)
    proj60 = E.load_json(E.guard_open(E.E60EXP / f"analysis/proj-{r}.json",
                                      "exp-060's projection allowlist and refusal list"))
    cold60 = E.load_json(E.guard_open(E.E60EXP / f"analysis/cold-{r}.json", "the cold-table record"))

    src_p = E.guard_open(E.ROOT / in60["npz"], "the exp-060 input archive")
    got = E.sha_file(src_p)
    if got != in60["npz_sha256"]:
        raise SystemExit(f"{src_p.name} is {got[:16]}, not exp-060's recorded {in60['npz_sha256'][:16]}")
    voc_p = E.guard_open(E.ROOT / in60["vocab_path"], "the exp-060 vocabulary")
    voc_sha = E.sha_file(voc_p)
    if voc_sha != in60["vocab_path_sha256"]:
        raise SystemExit(f"{voc_p.name} is not exp-060's recorded vocabulary")
    vocab = json.loads(voc_p.read_text())

    # ---- reproduce exp-060's registered facts, refuse a mismatch ------------------------------
    repro = {}
    if tuple(E.ROT[r][:3]) != (in60["months"]["V"], in60["months"]["S"], in60["months"]["I"]):
        raise SystemExit(f"{r}: the rotation differs from exp-060's")
    if sorted(in60["months"]["A"]) != sorted(A):
        raise SystemExit(f"{r}: the A months differ from exp-060's")
    repro["rotations"] = {"V": V, "S": S, "I": I, "A": A, "equal_to_exp060": True}
    got_allow = tuple(proj60["source"]["arrays_copied"])
    if got_allow != BASE11:
        raise SystemExit(f"{r}: exp-060's projection allowlist is {got_allow}, not the eleven")
    repro["exp060_projection"] = {
        "allowlist": list(got_allow), "allowlist_is_a_subset_of_inputs61": True,
        "refused_count": len(proj60["source"]["arrays_refused"]),
        "forbidden_patterns": proj60["source"]["forbidden_patterns"],
        "no_pred_or_loss_artifact_opened": proj60["source"]["no_pred_or_loss_artifact_opened"]}
    if cold60["table"]["sha256"] != in60["cold_table"]["sha256"]:
        raise SystemExit(f"{r}: the cold-table hash of the two exp-060 records differs")
    cold_p = E.guard_open(E.ROOT / cold60["table"]["path"], "the frozen cold factor table")
    cold_sha = E.sha_file(cold_p)
    if cold_sha != cold60["table"]["sha256"]:
        raise SystemExit(f"{cold_p.name} is {cold_sha[:16]}, not the recorded {cold60['table']['sha256'][:16]}")
    if (sorted(cold60["fitting_months"]) != sorted(A)
            or not cold60["no_S_I_V_arrival_outcome_loaded"]):
        raise SystemExit(f"{r}: the cold table is not the A-only table of this rotation")
    if cold60["supported_values"] != in60["cold_table"]["supported_values"]:
        raise SystemExit(f"{r}: the cold support counts of the two exp-060 records differ")
    repro["cold_table"] = {"path": cold60["table"]["path"], "sha256": cold_sha,
                           "fitting_months": cold60["fitting_months"],
                           "supported_values": cold60["supported_values"],
                           "no_S_I_V_arrival_outcome_loaded": True}

    # ---- the positive projection ---------------------------------------------------------------
    for a in ALLOW:
        if E.forbidden_array(a):
            raise SystemExit(f"the allowlist contains a forbidden name: {a}")
    z = np.load(src_p)
    present = sorted(z.files)
    missing = [a for a in ALLOW if a not in present]
    if missing:
        raise SystemExit(f"{src_p.name} does not hold {missing}")
    refused = [a for a in present if a not in ALLOW]
    not_forbidden = [a for a in refused if not E.forbidden_array(a)]
    if not_forbidden:
        raise SystemExit(f"refused arrays that no pattern catches: {not_forbidden}")
    arrays = {a: np.array(z[a]) for a in ALLOW}
    z.close()
    for a in arrays:
        if E.forbidden_array(a):
            raise SystemExit(f"a forbidden array reached the projection: {a}")
    for a in BASE11 + CODES12 + NA3 + PRIORS15:
        if E.sha_arr(arrays[a]) != in60["array_sha256"][a]:
            raise SystemExit(f"{r} {a}: the copied array is not the one exp-060 recorded")
    repro["common_base_p_hashes"] = {k: in60["array_sha256"][f"p_{k}"] for k in ("V", "S", "I")}
    repro["population_hashes"] = {k: in60["array_sha256"][f"pop_{k}"] for k in ("V", "S", "I")}

    nlev = {f: len(vocab[f]) for f in E.FAMS}
    if nlev != {f: int(in60["vocab_sizes"][f]) for f in E.FAMS}:
        raise SystemExit(f"{r}: the vocabulary sizes differ from exp-060's record")

    # ---- rebuild 1: the twelve codes, the ids, the population, the UTC-date index --------------
    rb_codes = {}
    pos = {f: {v: i for i, v in enumerate(vocab[f])} for f in E.FAMS}
    for k, m in (("V", V), ("S", S), ("I", I)):
        month_sha(m)
        d = E.dep_frame(m)
        ids = d["MVT_ID_mvt"].to_numpy("int64")
        if not np.array_equal(ids, arrays[f"ids_{k}"].astype("int64")):
            raise SystemExit(f"{r} {k}: the rebuilt ordered ids differ from the projection")
        popm = d["pop"].to_numpy(bool)
        if not np.array_equal(popm, arrays[f"pop_{k}"].astype(bool)):
            raise SystemExit(f"{r} {k}: the rebuilt population differs from the projection")
        di, uniq = E.day_index(d["day"].to_numpy())
        if k == "V" and not np.array_equal(di, arrays["day_V"].astype("int64")):
            raise SystemExit(f"{r} V: the rebuilt UTC-date index differs from the projection")
        for f in E.FAMS:
            c = np.array([pos[f][x] if (x is not None and not pd.isna(x)) else -1 for x in d[f]],
                         dtype="int64")
            if not np.array_equal(c, arrays[f"code_{k}_{f}"].astype("int64")):
                raise SystemExit(f"{r} {k} {f}: the rebuilt codes differ from the projection")
        rb_codes[k] = {"month": m, "departures": int(len(d)), "population_rows": int(popm.sum()),
                       "utc_dates": len(uniq), "ordered_ids_sha256": E.sha_arr(ids),
                       "codes_equal": True, "ids_equal": True, "population_equal": True,
                       "day_index_equal": bool(k == "V")}

    # ---- rebuild 2: the fifteen allowed priors, from the frozen cold table ----------------------
    tab = pd.read_parquet(cold_p)
    rb_priors = {}
    for arm in E.TABLE_ARMS:
        for f in E.PRIOR_FAMS:
            sub = tab["key"].astype(str).to_numpy() == f
            mp = dict(zip(tab.loc[sub, "value"].astype(str), tab.loc[sub, arm].to_numpy("float64")))
            v = np.array([mp.get(x, 0.0) for x in vocab[f]], dtype="float64")
            if not np.array_equal(v, arrays[f"prior_{arm}_{f}"]):
                raise SystemExit(f"{r} prior_{arm}_{f}: the rebuild from the cold table differs")
        rb_priors[arm] = {"rebuilt_from": cold60["table"]["path"], "byte_equal": True,
                          "nonzero": {f: int((arrays[f"prior_{arm}_{f}"] != 0).sum())
                                      for f in E.PRIOR_FAMS}}
    const_max = float(np.max(np.abs(np.concatenate([arrays[f"prior_const_{f}"]
                                                    for f in E.PRIOR_FAMS]))))
    if const_max != 0.0:
        raise SystemExit(f"{r}: the const prior is not exactly zero")

    # ---- rebuild 3: the three base-month departure counts ---------------------------------------
    nA_src = {f: {} for f in E.PRIOR_FAMS}
    for m in A:
        month_sha(m)
        c = month_counts(m)
        for f in E.PRIOR_FAMS:
            for key, n in c[f].items():
                nA_src[f][key] = nA_src[f].get(key, 0) + n
    for f in E.PRIOR_FAMS:
        v = np.array([nA_src[f].get(x, 0) for x in vocab[f]], dtype="int64")
        if not np.array_equal(v, arrays[f"nA_{f}"].astype("int64")):
            raise SystemExit(f"{r} nA_{f}: the rebuild from the nine A months differs")

    # ---- the raw month pins ---------------------------------------------------------------------
    for m in A:
        if cold60["month_files_sha256"][m] != month_sha(m):
            raise SystemExit(f"{m}: the raw file is not the one the cold table was fitted on")
    month_pins = {m: month_sha(m) for m in sorted(set(A) | {V, S, I})}

    # ---- the label-free half of the literal-sham identity ---------------------------------------
    nA_all = np.concatenate([arrays[f"nA_{f}"] for f in E.PRIOR_FAMS])
    zero_vec = np.zeros(len(nA_all))
    keep_all = np.ones(len(nA_all), dtype=bool)
    ident = all(float(np.max(np.abs(E.effective(zero_vec, keep_all, nA_all, c)))) == 0.0
                for c in E.C_GRID)
    if not ident:
        raise SystemExit(f"{r}: a zero prior is not zero after masking and weighting")
    repro["literal_sham_label_free_identity"] = {
        "const_prior_max_abs": const_max,
        "zero_const_literal_sham_prior_is_bitwise_zero_at_every_threshold_and_c": True}

    # ---- the airport and family map of every correction column -----------------------------------
    arrays["col_apt"] = E.column_airport(vocab)
    arrays["col_fam"] = E.column_family(nlev)

    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r,
           "months": {"V": V, "S": S, "I": I, "A": A},
           "launch": launch, "code_sha256": E.code_shas(),
           "source": {"record": str(in60_p.relative_to(E.ROOT)), "npz": in60["npz"],
                      "npz_sha256": got, "vocab_path": in60["vocab_path"],
                      "vocab_path_sha256": voc_sha,
                      "arrays_present": present, "arrays_copied": list(ALLOW),
                      "arrays_refused": refused,
                      "every_refused_array_matches_a_forbidden_pattern": True,
                      "forbidden_patterns": list(E.FORBID_ARRAY),
                      "no_exp056_archive_and_no_exp060_result_opened": True},
           "reproduced_from_exp060": repro,
           "rebuilds": {"codes": rb_codes, "priors": rb_priors,
                        "nA_from_the_nine_A_months": {f: {"columns": int(len(arrays[f'nA_{f}'])),
                                                          "byte_equal": True} for f in E.PRIOR_FAMS}},
           "month_files_sha256": month_pins,
           "vocab_sizes": nlev,
           "array_sha256": {k: E.sha_arr(v) for k, v in sorted(arrays.items())},
           "opened_paths": list(E.OPENED)}

    NO.refuse_existing([npz_p, vocab_p, rec_p], overwrite)
    npz_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_p, **arrays)
    vocab_p.write_text(json.dumps(vocab))
    rec["npz"] = str(npz_p.relative_to(E.ROOT))
    rec["npz_sha256"] = E.sha_file(npz_p)
    rec["vocab61_path"] = str(vocab_p.relative_to(E.ROOT))
    rec["vocab61_sha256"] = E.sha_file(vocab_p)
    rec["label_reads"] = E.LABEL_READS
    assert not E.LABEL_READS, "the projection read a label"
    rec["freeze_unix"] = time.time()
    rec["freeze_utc"] = E.utc()
    rec["seconds"] = round(time.time() - t0, 1)
    rec_p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"projected {r}: {len(ALLOW)} arrays copied, {len(refused)} refused; "
          f"{rec['npz_sha256'][:16]} ({rec['seconds']:.0f}s)", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotations", nargs="+", choices=E.ORDER)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    paths = []
    for r in a.rotations:
        paths += [E.WORK / f"inputs61-{r}.npz", E.WORK / f"vocab61-{r}.json",
                  E.ANALYSIS / f"inputs61-{r}.json"]
    NO.refuse_existing(paths, a.overwrite)
    print("guard: no existing artifact at " + ", ".join(p.name for p in paths), flush=True)
    for r in a.rotations:
        one(r, a.overwrite)


if __name__ == "__main__":
    main()
