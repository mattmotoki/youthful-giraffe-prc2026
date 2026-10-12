"""exp-063-sparseholdout: freeze the ordered correction inputs of one rotation (SPEC 1.4).  Label-free.

Reads the whitelisted projection (`work/proj-<r>.npz`, hash-checked against `analysis/proj-<r>.json`),
the rebuilt departure key strings (`work/keys-<r>.parquet`), the base-month key counts
(`work/nA-<r>.json`) and the COLD factor table of the rotation (`work/cold-<r>.parquet`,
hash-checked against `analysis/cold-<r>.json`).  It defines each vocabulary as the sorted union of
the V, S and I departure key values plus the cold table's supported values, writes the codes, the
prior vector of every arm aligned to that vocabulary (each carrying the cold table's hash, so the
prior reproduces from a named cold63-table hash), the base-month departure count per column, and
every array and ordered-id hash with the freeze time.  No label is read.

    .venv/bin/python experiments/exp-063-sparseholdout/code/freeze63.py jul
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
import e63 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def one(r: str, overwrite: bool) -> None:
    npz_p = E.WORK / f"inputs63-{r}.npz"
    vocab_p = E.WORK / f"vocab63-{r}.json"
    rec_p = E.ANALYSIS / f"inputs63-{r}.json"
    NO.refuse_existing([npz_p, vocab_p, rec_p], overwrite)
    t0 = time.time()
    launch = E.hold_load(f"freeze {r}")
    V, S, I, _ = E.ROT[r]
    A = E.cold_months(r)

    proj_rec = E.load_json(E.ANALYSIS / f"proj-{r}.json")
    for key, path in (("npz_sha256", E.ROOT / proj_rec["npz"]),
                      ("keys_parquet_sha256", E.ROOT / proj_rec["keys_parquet"]),
                      ("nA_json_sha256", E.ROOT / proj_rec["nA_json"])):
        got = E.sha_file(path)
        if got != proj_rec[key]:
            raise SystemExit(f"{path.name} is {got[:16]}, not the projection's {proj_rec[key][:16]}")
    cold_rec = E.load_json(E.ANALYSIS / f"cold63-{r}.json")
    cold_p = E.ROOT / cold_rec["table"]["path"]
    got = E.sha_file(cold_p)
    if got != cold_rec["table"]["sha256"]:
        raise SystemExit(f"{cold_p.name} is {got[:16]}, not the cold table's {cold_rec['table']['sha256'][:16]}")
    if set(cold_rec["fitting_months"]) != set(A) or not cold_rec["no_S_I_V_arrival_outcome_loaded"]:
        raise SystemExit(f"{r}: the cold table is not the A-only table of this rotation")

    z = np.load(E.ROOT / proj_rec["npz"])
    keys = pd.read_parquet(E.ROOT / proj_rec["keys_parquet"])
    nA_src = E.load_json(E.ROOT / proj_rec["nA_json"])
    tab = pd.read_parquet(cold_p)

    arrays = {a: np.array(z[a]) for a in z.files}
    z.close()

    # ---- the vocabulary: sorted union of V, S and I departure keys + cold supported values ----
    vocab, rec_vocab = {}, {}
    for f in E.FAMS:
        dep_vals = set(x for x in keys[f].tolist() if x is not None and not pd.isna(x))
        cold_vals = set()
        if f in E.PRIOR_FAMS:
            cold_vals = set(tab.loc[tab["key"] == f, "value"].astype(str))
        vocab[f] = sorted(dep_vals | cold_vals)
        rec_vocab[f] = {"size": len(vocab[f]), "departure_values": len(dep_vals),
                        "cold_supported_values": len(cold_vals),
                        "cold_only_values": len(cold_vals - dep_vals),
                        "departure_only_values": len(dep_vals - cold_vals),
                        "sha256": E.sha_text(vocab[f])}

    months = {"V": V, "S": S, "I": I}
    detail = {}
    for k, m in months.items():
        sub = keys[keys["side"] == k].reset_index(drop=True)
        ids = sub["MVT_ID_mvt"].to_numpy("int64")
        if not np.array_equal(ids, arrays[f"ids_{k}"].astype("int64")):
            raise SystemExit(f"{r} {k}: the key table's ids are not the projection's ids")
        for f in E.FAMS:
            pos = {v: i for i, v in enumerate(vocab[f])}
            arrays[f"code_{k}_{f}"] = np.array(
                [pos[x] if (x is not None and not pd.isna(x)) else -1 for x in sub[f]], dtype="int64")
        detail[k] = {"month": m, "departures": int(len(sub)),
                     "population_rows": int(arrays[f"pop_{k}"].sum()),
                     "ordered_ids_sha256": E.sha_arr(ids),
                     "coded_rows": {f: int((arrays[f"code_{k}_{f}"] >= 0).sum()) for f in E.FAMS}}

    # ---- the priors of every arm, aligned to the vocabulary ----------------------------------
    prior_src = {a: tab[a] for a in E.TABLE_ARMS}
    for s in E.SHAM_SEEDS:
        prior_src[f"sham{s:02d}"] = E.sham_table(tab, s)
    for arm, vals in prior_src.items():
        for f in E.PRIOR_FAMS:
            sub = tab["key"].astype(str).to_numpy() == f
            m = dict(zip(tab.loc[sub, "value"].astype(str), vals[sub].to_numpy("float64")))
            arrays[f"prior_{arm}_{f}"] = np.array([m.get(v, 0.0) for v in vocab[f]], dtype="float64")
    rec_priors = {arm: {"vector_sha256": E.sha_arr(*[arrays[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS]),
                        "nonzero": {f: int((arrays[f"prior_{arm}_{f}"] != 0).sum()) for f in E.PRIOR_FAMS},
                        "sd_s": {f: float(arrays[f"prior_{arm}_{f}"].std()) for f in E.PRIOR_FAMS},
                        "from_cold_table_sha256": cold_rec["table"]["sha256"]}
                  for arm in prior_src}
    if float(np.max(np.abs(np.concatenate([arrays[f"prior_const_{f}"] for f in E.PRIOR_FAMS])))) != 0.0:
        raise SystemExit(f"{r}: the const prior is not exactly zero")

    # ---- the base-month departure count of every column --------------------------------------
    for f in E.PRIOR_FAMS:
        arrays[f"nA_{f}"] = np.array([nA_src[f].get(v, 0) for v in vocab[f]], dtype="int64")

    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r,
           "months": {"V": V, "S": S, "I": I, "A": A},
           "launch": launch, "code_sha256": E.code_shas(),
           "projection": {"record": f"experiments/exp-063-sparseholdout/analysis/proj-{r}.json",
                          "npz_sha256": proj_rec["npz_sha256"],
                          "arrays_copied": proj_rec["source"]["arrays_copied"]},
           "cold_table": {"record": f"experiments/exp-063-sparseholdout/analysis/cold63-{r}.json",
                          "path": cold_rec["table"]["path"], "sha256": cold_rec["table"]["sha256"],
                          "fitting_months": cold_rec["fitting_months"],
                          "supported_values": cold_rec["supported_values"],
                          "uses_S_arrivals": False, "uses_I_arrivals": False,
                          "uses_V_arrivals": False},
           "vocab": rec_vocab, "months_detail": detail, "priors": rec_priors,
           "nA": {f: {"columns": int(len(arrays[f"nA_{f}"])),
                      "columns_with_A_departures": int((arrays[f"nA_{f}"] > 0).sum()),
                      "departures": int(arrays[f"nA_{f}"].sum())} for f in E.PRIOR_FAMS},
           "vocab_sizes": {f: len(vocab[f]) for f in E.FAMS},
           "array_sha256": {k: E.sha_arr(v) for k, v in sorted(arrays.items())}}

    NO.refuse_existing([npz_p, vocab_p, rec_p], overwrite)
    np.savez(npz_p, **arrays)
    vocab_p.write_text(json.dumps(vocab))
    rec["npz"] = str(npz_p.relative_to(E.ROOT))
    rec["npz_sha256"] = E.sha_file(npz_p)
    rec["vocab_path"] = str(vocab_p.relative_to(E.ROOT))
    rec["vocab_path_sha256"] = E.sha_file(vocab_p)
    rec["label_reads"] = E.LABEL_READS
    assert not E.LABEL_READS, "the freeze job read a label"
    rec["freeze_unix"] = time.time()
    rec["freeze_utc"] = E.utc()
    rec["seconds"] = round(time.time() - t0, 1)
    rec_p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"frozen {r}: {npz_p.name} {rec['npz_sha256'][:16]} "
          f"vocab {rec['vocab_sizes']} ({rec['seconds']:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotations", nargs="+", choices=E.ORDER)
    a = ap.parse_args()
    paths = []
    for r in a.rotations:
        paths += [E.WORK / f"inputs63-{r}.npz", E.WORK / f"vocab63-{r}.json",
                  E.ANALYSIS / f"inputs63-{r}.json"]
    NO.refuse_existing(paths, False)
    print("guard: no existing artifact at " + ", ".join(p.name for p in paths), flush=True)
    for r in a.rotations:
        one(r, False)


if __name__ == "__main__":
    main()
