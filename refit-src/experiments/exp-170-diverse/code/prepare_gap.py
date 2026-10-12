"""Replace only airport spacing with train-normalized same-runway spacing."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_runway import sha

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
assert not a.out.exists()
a.out.mkdir(parents=True)
rec = json.loads((a.inputs / "MANIFEST.json").read_text())
for name, digest in rec["files"].items():
    assert sha(a.inputs / name) == digest
meta = pd.read_parquet(a.inputs / "rows.parquet")
assert rec["group"] == "airport-day-runway"
train = (meta.year == 2025).to_numpy() & ~meta.month.isin([1, 7]).to_numpy()
assert sorted(meta.loc[train, "month"].unique().tolist()) == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
times = meta.MVT_TIME_UTC_mvt
gaps = {"gap_prev": times.groupby(meta.group, sort=False).diff().dt.total_seconds().clip(0, 86400),
        "gap_next": -times.groupby(meta.group, sort=False).diff(-1).dt.total_seconds().clip(-86400, 0)}
num = np.load(a.inputs / "numeric.npy").copy()
old = np.load(a.inputs / "numeric.npy", mmap_mode="r")
width = len(rec["numeric"])
changed = []
details = {}
for name, value in gaps.items():
    col = rec["numeric"].index(name)
    v = np.sign(value.to_numpy(float)) * np.log1p(np.abs(value.to_numpy(float)))
    center = float(np.nanmedian(v[train]))
    scale = float(max(np.nanstd(v[train]), .1))
    assert np.isfinite(center) and np.isfinite(scale)
    num[:, col] = np.nan_to_num((v - center) / scale, nan=0.).clip(-12, 12).astype("float32")
    num[:, col + width] = np.isfinite(v).astype("float32")
    rec["center"][col], rec["scale"][col] = center, scale
    changed += [col, col + width]
    details[name] = {"center": center, "scale": scale, "train_present": int(np.isfinite(v[train]).sum()),
                     "missing_rows": int((~np.isfinite(v)).sum()),
                     "median_raw_seconds": float(np.nanmedian(value.to_numpy(float)[train]))}
same = sorted(set(range(num.shape[1])) - set(changed))
assert np.array_equal(num[:, same], old[:, same])
assert np.isfinite(num).all()
np.save(a.out / "numeric.npy", num, allow_pickle=False)
for name in ["rows.parquet", "category.npy", "anchor.npy", "labels.npy"]:
    (a.out / name).symlink_to((a.inputs / name).resolve())
rec.update({"parent_manifest_sha256": sha(a.inputs / "MANIFEST.json"), "source_code_sha256": sha(__file__),
            "array_rule": "replace gap_prev/gap_next only; train-only scaling; all other values unchanged",
            "gap_columns": details, "unchanged_columns_exact": True,
            "timestamp_unit": str(times.dtype), "gap_unit": "seconds via timedelta.total_seconds",
            "files": {p.name: sha(p) for p in sorted(a.out.iterdir())}})
(a.out / "MANIFEST.json").write_text(json.dumps(rec, indent=1) + "\n")
print("ready", details, flush=True)
