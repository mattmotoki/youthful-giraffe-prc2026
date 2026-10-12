"""Part s7c (and the held fold vector) from the s7 seed files of a run tree.

1. w2/seq/export.py s7 work/seq7: the mean over the seed files of each mode, aligned to work/w2/chain_ids_{split}
   (fold <- held fits, ranking <- full fits): work/w2/s7_{fold,ranking}.parquet (released code, unchanged).
2. Part <parts>/s7c.parquet: MVT_ID_mvt (float64, template order) and s7c = the seed mean as float32, as w2/ords.py
   reads it, stored as float64; s7_full_<seed> per seed; NaN where the departure has no NM off-block time.
   Fold vector <parts>/s7c-held-fold.parquet (344,419 rows in chain fold order, held fits) for the ORD-S beta.
3. Reads: seed spread (rms of seed minus seed mean, by month and airport), the held read of the fold vector against
   the 2025 labels next to the saved s7c fold values, and the January/July distance to the saved s7c values.

    python s7_part.py <run> <parts dir>        -> prints JSON, writes <parts>/s7c-reads.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
run = Path(sys.argv[1]).resolve(); parts = Path(sys.argv[2]).resolve(); parts.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(run / "iso"))
import finalmode as FM  # noqa: E402

subprocess.run([sys.executable, "w2/seq/export.py", "s7", "work/seq7"], cwd=run, check=True)
D = run / "work/seq7"
ids = np.load(D / "ids.npy"); is26 = np.load(D / "is2026.npy"); month = np.load(D / "month.npy")
nm = np.load(D / "nm_raw.npy"); y = np.load(D / "y.npy").astype(float)
seeds = {m: sorted(glob.glob(str(D / f"s7-{m}-*.npy"))) for m in ("full", "held")}
rec = {"seed_files": {m: [Path(p).name for p in v] for m, v in seeds.items()},
       "seed_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for v in seeds.values() for p in v}}

# part s7c (ranking split, full fits)
rk = pd.read_parquet(run / "work/w2/s7_ranking.parquet")
tmpl = pd.read_parquet(FM.submitting_path(run / "input-root/data/raw"), columns=[ID])
assert np.array_equal(rk[ID].to_numpy("int64"), tmpl[ID].to_numpy("float64").astype("int64")) and len(rk) == FM.n_rows()
part = pd.DataFrame({ID: tmpl[ID].to_numpy("float64"), "s7c": rk.s7.to_numpy(float).astype(np.float32).astype(np.float64)})
pos = pd.Series(np.arange(len(ids)), index=ids)
row = pos.reindex(rk[ID].to_numpy("int64")).to_numpy()
assert np.isfinite(row).all() and is26[row.astype(int)].all()
for p in seeds["full"]:
    part[Path(p).stem.replace("s7-", "s7_").replace("-", "_")] = np.load(p).astype(float)[row.astype(int)]
fin = np.isfinite(part.s7c.to_numpy())
has = np.isfinite(nm[row.astype(int)])
assert np.array_equal(fin, has), "s7c finite rows differ from the rows with an NM off-block time"
part.to_parquet(parts / "s7c.parquet", index=False)
rec["s7c"] = {"rows": int(len(part)), "finite": int(fin.sum()), "sha256": hashlib.sha256((parts / "s7c.parquet").read_bytes()).hexdigest()}

# spread of the seeds around their mean (2026 rows, full fits), by month and airport
cols = [c for c in part.columns if c.startswith("s7_full_")]
mon = month[row.astype(int)]
f = pd.read_parquet(run / "work/feat/ranking.parquet", columns=[ID, "airport"])
assert np.array_equal(f[ID].to_numpy("int64"), rk[ID].to_numpy("int64"))
if len(cols) > 1:
    S = part[cols].to_numpy(float); mu = S.mean(1)
    dev = pd.DataFrame({"m": mon, "ap": f.airport.to_numpy(), "d2": ((S - mu[:, None]) ** 2).mean(1)})[fin]
    rec["seed_rms_by_month"] = {int(k): round(float(np.sqrt(v)), 1) for k, v in dev.groupby("m").d2.mean().items()}
    rec["seed_rms_by_airport"] = {k: round(float(np.sqrt(v)), 1) for k, v in dev.groupby("ap").d2.mean().items()}

# held fold vector (held fits) and the held read against the 2025 labels
fo = pd.read_parquet(run / "work/w2/s7_fold.parquet")
fo.to_parquet(parts / "s7c-held-fold.parquet", index=False)
rec["s7c_held_fold"] = {"rows": int(len(fo)), "finite": int(np.isfinite(fo.s7).sum()),
                        "sha256": hashlib.sha256((parts / "s7c-held-fold.parquet").read_bytes()).hexdigest()}
release = (run / "release").resolve()
sys.path.insert(0, str(release))
import assembly_replay as R  # noqa: E402
yl = pd.Series(y, index=ids).reindex(fo[ID].to_numpy("int64")).to_numpy(float)


def saved_vec(split):
    spec = json.loads((release / "ENTRIES.json").read_text())["entries"]["v73"]["member"][split]
    return R.decode_array(release.parent / spec["public_vector"], spec["public_vector_sha256"], spec["rows"])


sf = saved_vec("fold")
v = fo.s7.to_numpy(float)
ok = np.isfinite(v) & np.isfinite(sf) & np.isfinite(yl) & (yl < 20000)
rec["held_read"] = {"rows": int(ok.sum()), "rmse_new": float(np.sqrt(np.mean((yl[ok] - v[ok]) ** 2))),
                    "rmse_saved": float(np.sqrt(np.mean((yl[ok] - sf[ok]) ** 2))),
                    "rms_new_minus_saved": float(np.sqrt(np.mean((v[ok] - sf[ok]) ** 2))),
                    "bias_new_minus_saved": float(np.mean(v[ok] - sf[ok]))}
# January/July 2026 against the saved s7c values (board rows, template order)
sr = saved_vec("ranking")
n = len(sr)
a = part.s7c.to_numpy()[:n]
okr = np.isfinite(a) & np.isfinite(sr)
rec["janjul_vs_saved"] = {"rows": int(okr.sum()), "rms": float(np.sqrt(np.mean((a[okr] - sr[okr]) ** 2))),
                          "bias": float(np.mean(a[okr] - sr[okr])), "corr": float(np.corrcoef(a[okr], sr[okr])[0, 1]),
                          "finite_pattern_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(sr)))}
(parts / "s7c-reads.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
