"""exp-148 SPEC-r2 section 4: one feature table and one proxy table for the R2 stacker.

    WORK=<work root> python combine_r2.py

`inputs/features-r2c.parquet` = the committed `features.parquet` (2025 January and July, 2026
ranking) plus `features-r2.parquet` (2025-12/02/06/08, no `v23_*`, which stay NaN there).
`inputs/proxy-features-r2c.parquet` = the committed `proxy-features.parquet` plus the new
months of `kaggle/r2-la/out/proxy-features-r2.parquet`, after its January rows are checked
equal to the committed ones. Write-once, with a JSON record of the source hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    w = Path(os.environ["WORK"])
    src = {"features": w / "inputs/features.parquet", "features_r2": w / "inputs/features-r2.parquet",
           "proxy": w / "inputs/proxy-features.parquet",
           "proxy_r2": w / "kaggle/r2-la/out/proxy-features-r2.parquet"}
    outs = {"features": w / "inputs/features-r2c.parquet", "proxy": w / "inputs/proxy-features-r2c.parquet"}
    js = w / "inputs/r2c.json"
    if js.exists() or any(p.exists() for p in outs.values()):
        raise SystemExit("combined r2 tables exist; write-once")
    rec = {"what": "exp-148 SPEC-r2 combined stacker inputs", "sources": {k: str(p) for k, p in src.items()},
           "sources_sha256": {k: sha(p) for k, p in src.items()}, "checks": {}}

    f0, f1 = pd.read_parquet(src["features"]), pd.read_parquet(src["features_r2"])
    missing = [c for c in f0.columns if c not in f1.columns]
    if [c for c in f1.columns if c not in f0.columns] or any(not c.startswith("v23_") for c in missing):
        raise SystemExit(f"feature columns differ beyond v23_*: {missing}")
    f1 = f1.reindex(columns=f0.columns)
    f = pd.concat([f0, f1.astype(f0.dtypes.to_dict(), errors="ignore")], ignore_index=True)
    if f[ID].duplicated().any():
        raise SystemExit("a feature id repeats")
    rec["checks"]["features_rows"] = {"committed": int(len(f0)), "new": int(len(f1)), "combined": int(len(f))}
    rec["checks"]["new_months"] = f1.ctx_month.value_counts().sort_index().to_dict()

    p0, p1 = pd.read_parquet(src["proxy"]), pd.read_parquet(src["proxy_r2"])
    if list(p0.columns) != list(p1.columns):
        raise SystemExit("proxy columns differ")
    j0 = p0[p0.month == "2025-01"].set_index(ID).sort_index()
    j1 = p1[p1.month == "2025-01"].set_index(ID).sort_index()
    num = [c for c in j0.columns if c != "month"]
    if not j0.index.equals(j1.index):
        raise SystemExit("January proxy ids differ")
    diff = float(np.nanmax(np.abs(j0[num].to_numpy() - j1[num].to_numpy())))
    nan_mis = int((j0[num].isna().to_numpy() != j1[num].isna().to_numpy()).sum())
    if diff != 0.0 or nan_mis:
        raise SystemExit(f"January proxies differ: max {diff}, NaN pattern {nan_mis}")
    rec["checks"]["january_proxy_parity"] = {"rows": int(len(j0)), "max_abs_diff": diff, "nan_mismatch": nan_mis}
    p = pd.concat([p0, p1[p1.month != "2025-01"]], ignore_index=True)
    if p[ID].duplicated().any():
        raise SystemExit("a proxy id repeats")
    rec["checks"]["proxy_rows"] = {"committed": int(len(p0)), "combined": int(len(p))}
    new_ids = set(f1[ID].astype("int64"))
    got = set(p.loc[p.month.isin(["2025-12", "2025-02", "2025-06", "2025-08"]), ID].astype("int64"))
    rec["checks"]["new_month_ids_feature_vs_proxy"] = "equal" if new_ids == got else f"{len(new_ids ^ got)} differ"
    if new_ids != got:
        raise SystemExit("the new months' ids differ between the feature and proxy tables")

    f.to_parquet(outs["features"], index=False)
    p.to_parquet(outs["proxy"], index=False)
    rec["outputs_sha256"] = {k: sha(v) for k, v in outs.items()}
    rec["code_sha256"] = sha(Path(__file__))
    js.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(rec["checks"], indent=1, default=str))


if __name__ == "__main__":
    main()
