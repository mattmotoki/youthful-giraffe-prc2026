"""Part `v` and the V file from a run tree after ORD-S (final mode).

Part <parts>/v.parquet: MVT_ID_mvt (float64, template order) and the stage values in float64: v64 (the chain on the
pack, before NOAC and ROLL), c5, l2c, v (ORD-S, the V value). The V file is a byte copy of the ORD-S output
(outputs/w2-l2c-s7c.parquet, two columns, template order) under the final file name.

    python v_part.py <run> <parts dir> <V file>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, COL = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
run = Path(sys.argv[1]).resolve(); parts = Path(sys.argv[2]); vfile = Path(sys.argv[3])
sys.path.insert(0, str(run / "iso"))
import finalmode as FM  # noqa: E402

assert FM.is_final()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
tmpl = pd.read_parquet(FM.submitting_path(run / "input-root/data/raw"), columns=[ID])[ID].to_numpy("float64")
b = pd.read_parquet(run / "work/w2/base_ranking.parquet", columns=[ID, "v64", "c5"])
assert np.array_equal(b[ID].to_numpy("float64"), tmpl) and len(b) == FM.n_rows()
l2c = np.load(run / "work/w2/l2c_ranking.npy"); v = np.load(run / "work/w2/ords_ranking.npy")
src = run / "outputs/w2-l2c-s7c.parquet"
f = pd.read_parquet(src)
assert f.columns.tolist() == [ID, COL] and np.array_equal(f[ID].to_numpy("float64"), tmpl)
assert np.array_equal(f[COL].to_numpy("float64"), v), "the ORD-S file differs from work/w2/ords_ranking.npy"
assert np.isfinite(v).all() and v.min() >= 60
part = pd.DataFrame({ID: tmpl, "v64": b.v64.to_numpy("float64"), "c5": b.c5.to_numpy("float64"),
                     "l2c": l2c.astype("float64"), "v": v.astype("float64")})
parts.mkdir(parents=True, exist_ok=True)
out = parts / "v.parquet"
assert not out.exists() and not vfile.exists(), "part or file exists"
part.to_parquet(out, index=False)
shutil.copyfile(src, vfile)
assert sha(vfile) == sha(src)
rec ={"part": {"path": str(out), "sha256": sha(out), "rows": len(part), "columns": part.columns.tolist()},
       "file": {"path": str(vfile), "sha256": sha(vfile), "source": str(src)},
       "changed_rows": {"c5_vs_v64": int((part.c5 != part.v64).sum()), "l2c_vs_c5": int((part.l2c != part.c5).sum()),
                        "v_vs_l2c": int((part.v != part.l2c).sum())}}
(parts / "v-record.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
