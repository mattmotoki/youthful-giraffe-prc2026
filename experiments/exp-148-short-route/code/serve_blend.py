"""exp-148 SPEC-r4 section V: the ranking vector of a fixed blend of two ranking vectors.

    PRC_SOURCE_ROOT=... python serve_blend.py --a ranking-R2b-w100-kno.parquet --b v32.parquet \
        --w 0.5 --name B50 --out DIR

Serves w × a + (1 − w) × b on the 344,841 ranking rows in the order of the template
`submitting.parquet`: the template's ids once each, both inputs cover them exactly, finite, at
least 60 s. The weight is an argument fixed by the SPEC, never fitted. Write-once
`<out>/ranking-<name>.parquet` and `.json` (input and output sha256; the change against each
input; mean((a − b)²) on the ranking rows, which the frozen forecast reads).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PRC_DATA_DIR", str(Path(os.environ["PRC_SOURCE_ROOT"]) / "data/raw"))
import stacker as ST                 # noqa: E402

ID = ST.ID
N_RANKING = 344_841


def vector(path: Path) -> pd.Series:
    v = pd.read_parquet(path)
    return pd.Series(v["TAXITIME_SEC_mvt"].to_numpy("float64"), index=v[ID].astype("int64").to_numpy())


def change(d: np.ndarray) -> dict:
    return {"rms_s": float(np.sqrt(np.mean(d ** 2))), "mean_s": float(d.mean()),
            "max_abs_s": float(np.abs(d).max()), "rows_changed": int((d != 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--w", type=float, required=True, help="weight on --a")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pq, js = a.out / f"ranking-{a.name}.parquet", a.out / f"ranking-{a.name}.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    if not 0.0 < a.w < 1.0:
        raise SystemExit("the weight must be inside (0, 1)")
    tpl = pd.read_parquet(Path(os.environ["PRC_DATA_DIR"]) / "submitting.parquet", columns=[ID])
    ids = tpl[ID].to_numpy("float64")
    if len(ids) != N_RANKING or not np.all(ids == np.round(ids)) or pd.Index(ids).duplicated().any():
        raise SystemExit("template ids are not 344,841 unique integers")
    ids = ids.astype("int64")
    va, vb = vector(a.a), vector(a.b)
    for name, v in (("a", va), ("b", vb)):
        if v.index.duplicated().any() or set(v.index) != set(ids):
            raise SystemExit(f"--{name} does not cover the template ids exactly once")
    pa, pb = va.reindex(ids).to_numpy(), vb.reindex(ids).to_numpy()
    out = a.w * pa + (1 - a.w) * pb
    if not np.isfinite(out).all() or out.min() < ST.FLOOR - 1e-9:
        raise SystemExit(f"blend not finite or below {ST.FLOOR} s")
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: tpl[ID].to_numpy(), "TAXITIME_SEC_mvt": out}).to_parquet(pq, index=False)
    d = pa - pb
    rec = {"what": f"exp-148 SPEC-r4 blend {a.name} = {a.w} x a + {1 - a.w} x b", "w_on_a": a.w,
           "rows": int(len(out)), "min": float(out.min()), "max": float(out.max()), "mean": float(out.mean()),
           "mean_sq_a_minus_b": float(np.mean(d ** 2)), "var_of_sq_a_minus_b": float(np.var(d ** 2)),
           "rows_a_equals_b": int((d == 0).sum()), "vs_a": change(out - pa), "vs_b": change(out - pb),
           "inputs_sha256": {"a": ST.sha(a.a), "b": ST.sha(a.b)},
           "code_sha256": {"serve_blend.py": ST.sha(Path(__file__))}, "output_sha256": ST.sha(pq)}
    js.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(js.read_text())


if __name__ == "__main__":
    main()
