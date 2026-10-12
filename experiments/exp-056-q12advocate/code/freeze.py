"""exp-056-q12advocate: freeze the ordered correction inputs of one rotation (SPEC 3.5).

Label-free. Reads, for the rotation's V, S and I months, the departure rows WITHOUT either label
column (keys, regime, UTC date), the three members' kernel predictions (sha-checked against the
kernel's own port.json), the exposure base on S, exp-054's factor table for V (sha-checked), and
the departure key counts of the nine base months A (key columns and phase only). Writes one npz
and one record with every ordered-id and vector hash and the freeze time. No label is read.

    .venv/bin/python experiments/exp-056-q12advocate/code/freeze.py jul
    .venv/bin/python experiments/exp-056-q12advocate/code/freeze.py jul --dry   # synthetic base
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
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def paths(r: str, dry: bool):
    base = E.WORK / "dry" if dry else E.WORK
    tag = f"dry{E.DRY_VERSION}-" if dry else ""
    return (base / f"{tag}inputs-{r}.npz", base / f"{tag}vocab-{r}.json",
            E.ANALYSIS / f"{tag}inputs-{r}.json")


def member_preds(r: str, which: str, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """The three members' predictions for fold iso_<r>_<which>, aligned to `ids`. The matched-
    size folds (`xm`, `sm`) come from the thin kernels (`port-<member>-thin.json`)."""
    fold = f"iso_{r}_{which}"
    group = "thin" if which in ("xm", "sm") else r
    cols, rec = [], {}
    for m in E.MEMBERS:
        p = E.KAGGLE / f"{m}--{fold}.parquet"
        port = E.load_json(E.KAGGLE / f"port-{m}-{group}.json")
        if group == "thin":
            th = port["thin"][fold]
            if th["kept_ids_sorted_sha256"] != E.size_keep(r)["kept_ids_sorted_sha256"]:
                raise SystemExit(f"{m} {fold}: the kernel kept other rows than the registered selection")
        want = port["written"][fold]["file_sha256"]
        got = E.sha_file(p)
        if got != want:
            raise SystemExit(f"{p.name} is {got[:16]}, not the kernel's recorded {want[:16]}")
        df = pd.read_parquet(p, columns=["MVT_ID_mvt", "fold", "y_pred"])
        assert set(df["fold"].unique()) == {fold}
        if list(df.columns) != ["MVT_ID_mvt", "fold", "y_pred"]:
            raise SystemExit(f"{p.name} carries more than predictions: {list(df.columns)}")
        s = pd.Series(df["y_pred"].to_numpy("float64"), index=df["MVT_ID_mvt"].to_numpy("int64"))
        if not s.index.is_unique or set(s.index) != set(ids.tolist()):
            raise SystemExit(f"{p.name}: its ids are not the month's departure ids "
                             f"({len(s)} vs {len(ids)})")
        v = s.reindex(ids).to_numpy("float64")
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"{p.name}: non-finite prediction")
        cols.append(v)
        rec[m] = {"file": str(p.relative_to(E.ROOT)), "file_sha256": got,
                  "vector_sha256_ids_then_pred": E.sha_arr(ids, v),
                  "kernel": port["cfg"], "git": port["git"][:1]}
        if group == "thin":
            rec[m]["thin"] = port["thin"][fold]
    base = np.maximum(E.MIN_PRED, np.mean(cols, axis=0))
    return base, {"members": rec, "base_vector_sha256_ids_then_pred": E.sha_arr(ids, base)}, cols


def dry_base(dep: pd.DataFrame) -> np.ndarray:
    """Dry run only: a label-free stand-in base (600 s plus a fixed per-airport offset)."""
    apts = sorted(set(a for a in dep["apt"] if a is not None))
    off = {a: 20.0 * (i - len(apts) / 2) for i, a in enumerate(apts)}
    return np.array([600.0 + off.get(a, 0.0) for a in dep["apt"]], dtype="float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", choices=E.ORDER)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    r = a.rotation
    npz, vocab_p, rec_p = paths(r, a.dry)
    NO.refuse_existing([npz, vocab_p, rec_p], a.overwrite)
    npz.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    launch = E.hold_load(f"freeze {r}")
    V, S, I, M = E.ROT[r]
    A = [m for m in E.D.MONTHS if m not in (V, S, I)]
    months = {"V": V, "S": S, "I": I}

    dep = {k: E.dep_frame(m) for k, m in months.items()}
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "dry": a.dry,
           "months": {"V": V, "S": S, "I": I, "M": M, "A": A},
           "exposure_base_months": list(E.D.FOLDS[f"iso_{r}_x"].train_months),
           "launch": launch, "code_sha256": E.code_shas(), "months_detail": {}}
    arrays = {}
    for k, d in dep.items():
        ids = d["MVT_ID_mvt"].to_numpy("int64")
        arrays[f"ids_{k}"] = ids
        arrays[f"pop_{k}"] = d["pop"].to_numpy(bool)
        di, uniq = E.day_index(d["day"].to_numpy())
        arrays[f"day_{k}"] = di
        arrays[f"nonm_{k}"] = d["regime"].astype(str).to_numpy() == "nonm"
        md = {"month": months[k], "departures": int(len(d)), "population_rows": int(d["pop"].sum()),
              "regime_rows": {str(x): int(n) for x, n in d["regime"].astype(str).value_counts().items()},
              "utc_dates": uniq, "ordered_ids_sha256": E.sha_arr(ids),
              "population_ids_sha256": E.sha_arr(ids[d["pop"].to_numpy(bool)]),
              "days_sha256": E.sha_text(d["day"].tolist())}
        if a.dry:
            arrays[f"p_{k}"] = dry_base(d)
            md["base"] = {"dry_stand_in": True, "vector_sha256": E.sha_arr(ids, arrays[f"p_{k}"])}
        else:
            base, brec, cols = member_preds(r, k.lower(), ids)
            arrays[f"p_{k}"] = base
            for m, c in zip(E.MEMBERS, cols):
                arrays[f"member_{m}_{k}"] = c
            md["base"] = brec
        rec["months_detail"][k] = md
    # the matched-size exposure pair on S (SPEC 3.5): exposed (A - M + S) and isolated (A)
    # bases at the same number of training departures; the larger side is the thinned one
    ex, ref = E.EXPOSURE_PAIR[r]
    rec["exposure"] = {"expo_fold": f"iso_{r}_{ex}", "ref_fold": f"iso_{r}_{ref}",
                       "size_selection": E.size_keep(r)}
    if a.dry:
        arrays["p_Sexpo"] = arrays["p_S"] + 5.0
        arrays["p_Sref"] = arrays["p_S"] - 3.0
        rec["exposure"]["dry_stand_in"] = True
    else:
        for key, which in (("Sexpo", ex), ("Sref", ref)):
            if which == "s":
                arrays[f"p_{key}"] = arrays["p_S"]
                rec["exposure"][key] = "the main isolated base on S (unthinned side of the pair)"
                continue
            b_, brec_, cols_ = member_preds(r, which, arrays["ids_S"])
            arrays[f"p_{key}"] = b_
            for m, c in zip(E.MEMBERS, cols_):
                arrays[f"member_{m}_{key}"] = c
            rec["exposure"][key] = brec_

    # vocabulary: the union of keys in V, S, I and the factor table, sorted per family
    tab = E.factor_table(r)
    rec["factor_table"] = {"fold": E.REP_FOLD[r], "sha256": E.REP_SHA[E.REP_FOLD[r]],
                           "rows": int(len(tab)),
                           "keys_by_family": {f: int(n) for f, n in tab["key"].value_counts().items()},
                           "arrival_months": [m for m in E.D.MONTHS if m != V],
                           "uses_S_arrivals": True, "uses_I_arrivals": True, "uses_V_arrivals": False}
    vocab = {}
    for f in E.FAMS:
        ks = set()
        for d in dep.values():
            ks |= set(x for x in d[f] if x is not None)
        if f in E.PRIOR_FAMS:
            ks |= set(tab.loc[tab["key"] == f, "value"].astype(str))
        vocab[f] = sorted(ks)
    for k, d in dep.items():
        for f in E.FAMS:
            pos = {v: i for i, v in enumerate(vocab[f])}
            arrays[f"code_{k}_{f}"] = np.array([pos[x] if x is not None else -1 for x in d[f]],
                                               dtype="int64")
    # priors per arm, aligned to the vocabulary
    arms = {t: tab[t] for t in E.TABLE_ARMS}
    for s in E.SHAM_SEEDS:
        arms[f"sham{s:02d}"] = E.sham_table(tab, s)
    for arm, vals in arms.items():
        for f in E.PRIOR_FAMS:
            sub = tab["key"].astype(str).to_numpy() == f
            m = dict(zip(tab.loc[sub, "value"].astype(str), vals[sub].to_numpy("float64")))
            arrays[f"prior_{arm}_{f}"] = np.array([m.get(v, 0.0) for v in vocab[f]], dtype="float64")
    rec["priors"] = {arm: E.sha_arr(*[arrays[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS])
                     for arm in arms}
    # base-month departure support per key (label-free counts over A)
    for f in E.PRIOR_FAMS:
        c = E.key_counts(A, f)
        arrays[f"nA_{f}"] = np.array([c.get(v, 0) for v in vocab[f]], dtype="int64")
    # time-origin thinning (fixed before any label read)
    n_keep = int(min(arrays["pop_S"].sum(), arrays["pop_I"].sum()))
    arrays["thin_S"] = E.thin_mask(arrays["ids_S"], arrays["pop_S"], n_keep)
    arrays["thin_I"] = E.thin_mask(arrays["ids_I"], arrays["pop_I"], n_keep)
    rec["time_origin"] = {"rows_each": n_keep, "salt": E.ORIGIN_SALT,
                          "thin_S_ids_sha256": E.sha_arr(arrays["ids_S"][arrays["thin_S"]]),
                          "thin_I_ids_sha256": E.sha_arr(arrays["ids_I"][arrays["thin_I"]])}
    rec["vocab_sizes"] = {f: len(vocab[f]) for f in E.FAMS}
    rec["array_sha256"] = {k: E.sha_arr(v) for k, v in sorted(arrays.items())}

    NO.refuse_existing([npz, vocab_p, rec_p], a.overwrite)
    np.savez(npz, **arrays)
    vocab_p.write_text(json.dumps(vocab))
    rec["npz"] = str(npz.relative_to(E.ROOT))
    rec["npz_sha256"] = E.sha_file(npz)
    rec["vocab"] = str(vocab_p.relative_to(E.ROOT))
    rec["vocab_sha256"] = E.sha_file(vocab_p)
    rec["label_reads"] = E.LABEL_READS
    assert not E.LABEL_READS, "the freeze job read a label"
    rec["freeze_unix"] = time.time()
    rec["freeze_utc"] = E.utc()
    rec["seconds"] = round(time.time() - t0, 1)
    rec_p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"frozen {r}: {npz.name} {rec['npz_sha256'][:16]} ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
