"""Reorder frozen input values by runway; fit no encoder and read no new label."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_runway import sha, windows

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
assert not a.out.exists()
a.out.mkdir(parents=True)
rec = json.loads((a.inputs / "MANIFEST.json").read_text())
for name, digest in rec["files"].items():
    assert sha(a.inputs / name) == digest, name
parts = []
for path, digest in rec["sources"].items():
    p = Path(path)
    if p.name.startswith("training_2025-") or p.name == "ranking.parquet":
        assert sha(p) == digest
        f = pd.read_parquet(p, columns=["MVT_ID_mvt", "RUNWAY_mvt", "PHASE_mvt"])
        parts.append(f.loc[f.PHASE_mvt == "DEP", ["MVT_ID_mvt", "RUNWAY_mvt"]])
rw = pd.concat(parts).set_index("MVT_ID_mvt")
assert not rw.index.duplicated().any()
meta = pd.read_parquet(a.inputs / "rows.parquet")
meta["RUNWAY_mvt"] = rw.RUNWAY_mvt.reindex(meta.MVT_ID_mvt).astype("string").fillna("__MISSING__").to_numpy()
meta["original_row"] = np.arange(len(meta))
meta = meta.sort_values(["ADEP_mvt", "date", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"], kind="stable").reset_index(drop=True)
order = meta.original_row.to_numpy()
assert np.array_equal(np.sort(order), np.arange(len(meta)))
meta["group"] = pd.factorize(pd.MultiIndex.from_frame(meta[["ADEP_mvt", "date", "RUNWAY_mvt"]]), sort=False)[0]
meta.to_parquet(a.out / "rows.parquet", index=False)
for name in ["numeric", "category", "anchor", "labels"]:
    values = np.load(a.inputs / (name + ".npy"), mmap_mode="r")
    np.save(a.out / (name + ".npy"), values[order], allow_pickle=False)
    print("reordered", name, flush=True)
train = (meta.year == 2025).to_numpy() & ~meta.month.isin([1, 7]).to_numpy()
held = (meta.year == 2025).to_numpy() & meta.month.isin([1, 7]).to_numpy()
rank = (meta.year == 2026).to_numpy()
checks = {}
for name, mask in [("train", train), ("held", held), ("rank", rank)]:
    chunks = windows(meta, mask, 4)
    visits = np.zeros(len(meta), dtype="int8")
    for lo, hi, central, end in chunks:
        assert meta.group.iloc[lo] == meta.group.iloc[hi - 1]
        assert mask[lo:hi].all()
        visits[central:end] += 1
        assert hi - lo <= 12
    assert np.array_equal(visits, mask.astype("int8"))
    checks[name] = {"rows": int(mask.sum()), "chunks": len(chunks), "max_window": 12}
rec.update({"parent_manifest_sha256": sha(a.inputs / "MANIFEST.json"),
            "source_code_sha256": sha(__file__), "group": "airport-day-runway",
            "array_rule": "unchanged values, reordered only; airport spacing inputs retained",
            "checks": checks, "files": {p.name: sha(p) for p in sorted(a.out.iterdir())}})
(a.out / "MANIFEST.json").write_text(json.dumps(rec, indent=1) + "\n")
print("ready", checks, flush=True)
