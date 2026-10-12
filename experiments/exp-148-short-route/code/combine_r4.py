"""exp-148 SPEC-r4 item M.3: one feature table and one proxy table for the R4 stacker (nine months).

    WORK=<work root> python combine_r4.py

`inputs/features-r4c.parquet` = `inputs/features-r2c.parquet` (2025 January, July, December,
February, June, August; 2026 ranking) plus `inputs-r4/features-r4.parquet` (2025-03/10/11, built on
the Mac by `features_r2.py --months`, no `v23_*`, which stay NaN there).
`inputs/proxy-features-r4c.parquet` = `inputs/proxy-features-r2c.parquet` plus the three months of
`kaggle/r4-la/out/proxy-features-r4.parquet`, after its January rows are checked equal to the
committed ones. Write-once, with a JSON record of the source hashes (as `combine_r2.py`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
NEW = ["2025-03", "2025-10", "2025-11"]


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r4_proxy_only_ids() -> set[int]:
    """Read the caller-derived March boundary exception."""
    path = os.environ.get("PRC_R2_NO_FEATURES")
    if not path:
        raise SystemExit("Set PRC_R2_NO_FEATURES from scripts/exp148_missing_features.py")
    values = json.loads(Path(path).read_text())["missing_feature_ids_by_file_month"]["2025-03"]
    if len(values) != 1 or type(values[0]) is not int:
        raise SystemExit("March needs exactly one caller-owned boundary ID")
    return set(values)


def main() -> None:
    w = Path(os.environ["WORK"])
    src = {"features_r2c": w / "inputs/features-r2c.parquet", "features_r4": w / "inputs-r4/features-r4.parquet",
           "proxy_r2c": w / "inputs/proxy-features-r2c.parquet",
           "proxy_r4": w / "kaggle/r4-la/out/proxy-features-r4.parquet"}
    outs = {"features": w / "inputs/features-r4c.parquet", "proxy": w / "inputs/proxy-features-r4c.parquet"}
    js = w / "inputs/r4c.json"
    if js.exists() or any(p.exists() for p in outs.values()):
        raise SystemExit("combined r4 tables exist; write-once")
    rec = {"what": "exp-148 SPEC-r4 combined stacker inputs (nine months)", "sources": {k: str(p) for k, p in src.items()},
           "sources_sha256": {k: sha(p) for k, p in src.items()}, "checks": {}}
    mac = json.loads((w / "inputs-r4/features-r4.json").read_text())
    if mac["output_sha256"] != rec["sources_sha256"]["features_r4"]:
        raise SystemExit("features-r4.parquet differs from its Mac build record")
    rec["checks"]["features_r4_equals_mac_record"] = True

    f0, f1 = pd.read_parquet(src["features_r2c"]), pd.read_parquet(src["features_r4"])
    missing = [c for c in f0.columns if c not in f1.columns]
    if [c for c in f1.columns if c not in f0.columns] or any(not c.startswith("v23_") for c in missing):
        raise SystemExit(f"feature columns differ beyond v23_*: {missing}")
    if sorted(f1.ctx_month.unique().tolist()) != [3, 10, 11] or f1.ctx_year.unique().tolist() != [2025]:
        raise SystemExit("features-r4 holds other months")
    f1 = f1.reindex(columns=f0.columns)
    f = pd.concat([f0, f1.astype(f0.dtypes.to_dict(), errors="ignore")], ignore_index=True)
    if f[ID].duplicated().any():
        raise SystemExit("a feature id repeats")
    rec["checks"]["features_rows"] = {"r2c": int(len(f0)), "new": int(len(f1)), "combined": int(len(f))}
    rec["checks"]["new_months"] = {str(k): int(v) for k, v in f1.ctx_month.value_counts().sort_index().items()}

    p0, p1 = pd.read_parquet(src["proxy_r2c"]), pd.read_parquet(src["proxy_r4"])
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
    p = pd.concat([p0, p1[p1.month.isin(NEW)]], ignore_index=True)
    if p[ID].duplicated().any():
        raise SystemExit("a proxy id repeats")
    rec["checks"]["proxy_rows"] = {"r2c": int(len(p0)), "combined": int(len(p))}
    new_ids = set(f1[ID].astype("int64"))
    got = set(p.loc[p.month.isin(NEW), ID].astype("int64"))
    only_proxy = sorted(int(i) for i in got - new_ids)
    rec["checks"]["new_month_ids_feature_vs_proxy"] = {"only_features": len(new_ids - got), "only_proxy": only_proxy}
    if new_ids - got or not set(only_proxy) <= r4_proxy_only_ids():
        raise SystemExit(f"the new months' ids differ between the feature and proxy tables: {rec['checks']}")

    f.to_parquet(outs["features"], index=False)
    p.to_parquet(outs["proxy"], index=False)
    rec["outputs_sha256"] = {k: sha(v) for k, v in outs.items()}
    rec["code_sha256"] = sha(Path(__file__))
    js.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(rec["checks"], indent=1, default=str))


if __name__ == "__main__":
    main()
