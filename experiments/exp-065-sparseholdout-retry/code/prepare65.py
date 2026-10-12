"""Build and freeze exp-065's sealed label-free archive from its three member predictions."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e65 as E

MEMBERS = ("F1-fresh", "T-tailv4", "L2-hub8725")
SIDES = {"V": ("sparse65_v", "2025-05"), "S": ("sparse65_s", "2025-06"),
         "I": ("sparse65_i", "2025-04")}


def main() -> None:
    outs = [E.WORK / "inputs65-may.npz", E.WORK / "vocab65-may.json",
            E.ANALYSIS / "inputs65-may.json"]
    if any(p.exists() for p in outs):
        raise SystemExit("write-once refusal: an output exists")
    manifest = json.loads((E.ANALYSIS / "manifest.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    detail = {}
    frames = []
    for side, (fold, month) in SIDES.items():
        d = E.dep_frame(month)
        ids = d.MVT_ID_mvt.to_numpy("int64")
        pred = []
        sources = {}
        for member in MEMBERS:
            p = E.WORK / f"{member}--{fold}.parquet"
            q = pd.read_parquet(p, columns=["MVT_ID_mvt", "fold", "y_pred"])
            if q["fold"].nunique() != 1 or q["fold"].iloc[0] != fold:
                raise SystemExit(f"{p}: wrong fold")
            if not np.array_equal(q.MVT_ID_mvt.to_numpy("int64"), ids):
                raise SystemExit(f"{p}: ordered ids differ from raw month")
            pred.append(q.y_pred.to_numpy("float64"))
            sources[member] = {"path": str(p.relative_to(E.ROOT)), "sha256": E.sha_file(p),
                               "vector_sha256": E.sha_arr(ids, pred[-1])}
        base = np.maximum(E.MIN_PRED, np.mean(pred, axis=0))
        arrays[f"ids_{side}"] = ids
        arrays[f"pop_{side}"] = d["pop"].to_numpy(bool)
        arrays[f"p_{side}"] = base
        if side == "V":
            arrays["day_V"] = E.day_index(d["day"].to_numpy())[0]
            arrays["nonm_V"] = np.zeros(len(d), dtype=bool)
        x = d.assign(side=side)
        frames.append(x[["side", "MVT_ID_mvt", "apt", "stand", "rwy", "op"]])
        detail[side] = {"month": month, "rows": len(d), "members": sources,
                        "base_sha256": E.sha_arr(ids, base), "population": int(arrays[f"pop_{side}"].sum())}

    cold_rec = json.loads((E.ANALYSIS / "cold65-may.json").read_text())
    cold_p = E.ROOT / cold_rec["table"]["path"]
    if E.sha_file(cold_p) != cold_rec["table"]["sha256"]:
        raise SystemExit("cold table hash mismatch")
    tab = pd.read_parquet(cold_p)
    keys = pd.concat(frames, ignore_index=True)
    vocab = {}
    for fam in E.FAMS:
        v = set(keys[fam].dropna().astype(str))
        if fam in E.PRIOR_FAMS:
            v |= set(tab.loc[tab.key == fam, "value"].astype(str))
        vocab[fam] = sorted(v)
        pos = {x: i for i, x in enumerate(vocab[fam])}
        for side in SIDES:
            s = keys.loc[keys.side == side, fam]
            arrays[f"code_{side}_{fam}"] = np.array([pos.get(str(x), -1) if pd.notna(x) else -1 for x in s],
                                                     dtype="int64")
    counts = {f: {} for f in E.PRIOR_FAMS}
    for month in E.cold_months("may"):
        for fam, vals in E.month_key_counts(month).items():
            for key, n in vals.items():
                counts[fam][key] = counts[fam].get(key, 0) + n
    for fam in E.PRIOR_FAMS:
        arrays[f"nA_{fam}"] = np.array([counts[fam].get(x, 0) for x in vocab[fam]], dtype="int64")
        sub = tab.key.to_numpy() == fam
        for arm in E.TABLE_ARMS:
            mp = dict(zip(tab.loc[sub, "value"].astype(str), tab.loc[sub, arm].to_numpy("float64")))
            arrays[f"prior_{arm}_{fam}"] = np.array([mp.get(x, 0.0) for x in vocab[fam]])

    allow = manifest["sealed_archive_allowlist"]
    if sorted(arrays) != sorted(allow):
        raise SystemExit(f"archive names differ: missing={set(allow)-set(arrays)}, extra={set(arrays)-set(allow)}")
    E.WORK.mkdir(parents=True, exist_ok=True)
    np.savez(outs[0], **arrays)
    outs[1].write_text(json.dumps(vocab, sort_keys=True))
    rec = {"rotation": "may", "months": manifest["rotation"], "base": detail,
           "cold_table": cold_rec["table"], "vocab_sizes": {k: len(v) for k, v in vocab.items()},
           "array_names": sorted(arrays), "array_sha256": {k: E.sha_arr(v) for k, v in arrays.items()},
           "npz": str(outs[0].relative_to(E.ROOT)), "npz_sha256": E.sha_file(outs[0]),
           "vocab65_path": str(outs[1].relative_to(E.ROOT)), "vocab65_sha256": E.sha_file(outs[1]),
           "month_files_sha256": {m: E.sha_file(E.D.month_file(m)) for m in E.D.MONTHS},
           "derived_indexes_absent": not ({"col_fam", "col_apt"} & set(arrays)),
           "freeze_unix": time.time(), "freeze_utc": E.utc(), "label_reads": []}
    outs[2].write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"frozen {outs[0]} {rec['npz_sha256']}")


if __name__ == "__main__":
    main()
