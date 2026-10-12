"""exp-060-coldprior: the whitelisted input projection (SPEC 1.3).  Label-free, one process,
run before any S, I or V label read.

From each hash-pinned exp-056 input archive it takes ELEVEN arrays and nothing else:

    ids_V ids_S ids_I   pop_V pop_S pop_I   p_V p_S p_I   day_V   nonm_V

It refuses any requested or written name that matches a forbidden pattern (`prior_*`, `h_*`,
`member_*`, `code_*`, `nA_*`, `thin_*`, the exposure bases, or any name that carries `loss`,
`label`, `expo`, `sham` or `y_`), and it records every array of the source that it did not copy.
It opens no exp-056 `pred-*` archive and no row-loss artifact.

Everything else is rebuilt from the raw monthly files: the V, S and I departure key strings and
the population with exp-056's explicit label-free column list, whose ordered ids and population
must equal the projection exactly; and the base-month (A) departure count of every key, from
label-free A rows.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-060-coldprior/code/project60.py jul jan dec
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

#: the assignment's allowlist, frozen here
ALLOW = ("ids_V", "ids_S", "ids_I", "pop_V", "pop_S", "pop_I", "p_V", "p_S", "p_I",
         "day_V", "nonm_V")
#: nothing that matches any of these may be requested or written
FORBID = (r"^prior_", r"^h_", r"^hI_", r"^member_", r"^code_", r"^nA_", r"^thin_",
          r"^p_S(expo|ref)$", r"loss", r"label", r"expo", r"sham", r"^y_")


def forbidden(name: str) -> bool:
    return any(re.search(p, name) for p in FORBID)


_COUNT_CACHE: dict = {}


def month_counts(month: str) -> dict:
    if month not in _COUNT_CACHE:
        _COUNT_CACHE[month] = E.month_key_counts(month)
    return _COUNT_CACHE[month]


def one(r: str, overwrite: bool) -> dict:
    npz_p = E.WORK / f"proj-{r}.npz"
    keys_p = E.WORK / f"keys-{r}.parquet"
    na_p = E.WORK / f"nA-{r}.json"
    rec_p = E.ANALYSIS / f"proj-{r}.json"
    NO.refuse_existing([npz_p, keys_p, na_p, rec_p], overwrite)
    t0 = time.time()
    launch = E.hold_load(f"projection {r}")
    V, S, I, _ = E.ROT[r]
    A = E.cold_months(r)

    # ---- the pinned exp-056 source ---------------------------------------------------------
    src_rec_p = E.E56 / f"analysis/inputs-{r}.json"
    src_rec = E.load_json(src_rec_p)
    src_p = E.ROOT / src_rec["npz"]
    got = E.sha_file(src_p)
    if got != src_rec["npz_sha256"]:
        raise SystemExit(f"{src_p.name} is {got[:16]}, not exp-056's recorded {src_rec['npz_sha256'][:16]}")
    for a in ALLOW:
        if forbidden(a):
            raise SystemExit(f"the allowlist contains a forbidden name: {a}")
    z = np.load(src_p)
    present = sorted(z.files)
    missing = [a for a in ALLOW if a not in present]
    if missing:
        raise SystemExit(f"{src_p.name} does not hold {missing}")
    arrays = {a: np.array(z[a]) for a in ALLOW}
    z.close()
    refused = [a for a in present if a not in ALLOW]
    bad = [a for a in arrays if forbidden(a)]
    if bad:
        raise SystemExit(f"a forbidden array reached the projection: {bad}")

    # ---- the rebuild from the raw files (label-free) ----------------------------------------
    frames, months_detail = [], {}
    for k, m in (("V", V), ("S", S), ("I", I)):
        d = E.dep_frame(m)
        ids = d["MVT_ID_mvt"].to_numpy("int64")
        if not np.array_equal(ids, arrays[f"ids_{k}"].astype("int64")):
            raise SystemExit(f"{r} {k}: the rebuilt ordered ids differ from the projection")
        pop = d["pop"].to_numpy(bool)
        if not np.array_equal(pop, arrays[f"pop_{k}"].astype(bool)):
            raise SystemExit(f"{r} {k}: the rebuilt population differs from the projection")
        di, uniq = E.day_index(d["day"].to_numpy())
        if k == "V" and not np.array_equal(di, arrays["day_V"].astype("int64")):
            raise SystemExit(f"{r} V: the rebuilt UTC-date index differs from the projection")
        d = d.assign(month=m, side=k)
        frames.append(d[["month", "side", "MVT_ID_mvt", "apt", "stand", "rwy", "op", "pop", "day"]])
        months_detail[k] = {
            "month": m, "departures": int(len(d)), "population_rows": int(pop.sum()),
            "utc_dates": len(uniq),
            "ordered_ids_sha256": E.sha_arr(ids),
            "population_ids_sha256": E.sha_arr(ids[pop]),
            "keys_sha256": {f: E.sha_text(d[f].tolist()) for f in E.FAMS},
            "distinct_key_values": {f: int(pd.Series(d[f]).nunique()) for f in E.FAMS},
            "p_from_projection": {"sha256_ids_then_p": E.sha_arr(ids, arrays[f"p_{k}"]),
                                 "mean_s": float(np.mean(arrays[f"p_{k}"])),
                                 "min_s": float(np.min(arrays[f"p_{k}"]))}}
    keys = pd.concat(frames, ignore_index=True)

    # ---- base-month departure counts per key (label-free A rows) ---------------------------
    nA = {f: {} for f in E.PRIOR_FAMS}
    for m in A:
        c = month_counts(m)
        for f in E.PRIOR_FAMS:
            for key, n in c[f].items():
                nA[f][key] = nA[f].get(key, 0) + n

    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r,
           "months": {"V": V, "S": S, "I": I, "A": A},
           "launch": launch, "code_sha256": E.code_shas(),
           "source": {"record": str(src_rec_p.relative_to(E.ROOT)), "npz": str(src_p.relative_to(E.ROOT)),
                      "npz_sha256": got, "arrays_present": present,
                      "arrays_copied": list(ALLOW), "arrays_refused": refused,
                      "forbidden_patterns": list(FORBID),
                      "forbidden_names_in_output": bad,
                      "no_pred_or_loss_artifact_opened": True},
           "months_detail": months_detail,
           "A_key_counts": {f: {"keys": len(nA[f]), "departures": int(sum(nA[f].values()))}
                            for f in E.PRIOR_FAMS},
           "array_sha256": {k: E.sha_arr(v) for k, v in sorted(arrays.items())}}

    NO.refuse_existing([npz_p, keys_p, na_p, rec_p], overwrite)
    npz_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_p, **arrays)
    keys.to_parquet(keys_p, index=False)
    na_p.write_text(json.dumps(nA, sort_keys=True))
    rec["npz"] = str(npz_p.relative_to(E.ROOT))
    rec["npz_sha256"] = E.sha_file(npz_p)
    rec["keys_parquet"] = str(keys_p.relative_to(E.ROOT))
    rec["keys_parquet_sha256"] = E.sha_file(keys_p)
    rec["nA_json"] = str(na_p.relative_to(E.ROOT))
    rec["nA_json_sha256"] = E.sha_file(na_p)
    rec["label_reads"] = E.LABEL_READS
    assert not E.LABEL_READS, "the projection read a label"
    rec["projection_unix"] = time.time()
    rec["projection_utc"] = E.utc()
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
        paths += [E.WORK / f"proj-{r}.npz", E.WORK / f"keys-{r}.parquet",
                  E.WORK / f"nA-{r}.json", E.ANALYSIS / f"proj-{r}.json"]
    NO.refuse_existing(paths, a.overwrite)
    print("guard: no existing artifact at " + ", ".join(str(p.name) for p in paths), flush=True)
    for r in a.rotations:
        one(r, a.overwrite)


if __name__ == "__main__":
    main()
