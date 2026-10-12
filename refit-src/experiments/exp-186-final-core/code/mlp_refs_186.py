"""exp-186: reference table of the saved MLP008 and mix12 `final` vectors, in the layout of `v73-parts.parquet`.

v73-parts holds the released choice `primary_frame:mlp` (mix12 on member-scope clock rows, mlp008 on every
other row) and no column of either vector alone. This table gives parity.py a reference for each part
column: `saved:mlp008` (out/mlp-008/final, the mean of seeds 701/702, pinned by its FIT.json) and
`saved:mix12` (out/mlp-012/vectors/final-mixb12.parquet, pinned by BAG-mix.json), NaN on the rows without a
value, next to `MVT_ID_mvt`, `airport`, `month`, `bundle:cls` copied from v73-parts (344,841 board rows,
template order). Before it writes, it checks that the released rule on the two columns gives
`primary_frame:mlp` bit for bit (NaN at the same rows). Write-once.

    python experiments/exp-186-final-core/code/mlp_refs_186.py --v73-parts <refs/v73-parts.parquet> \
        --work <prc-exp164-work> --out <dir>
then
    python experiments/exp-186-final-core/code/parity.py --refs <dir>/mlp-refs.parquet --new <parts>/mlp.parquet \
        --pair mlp008=saved:mlp008 --pair mix12=saved:mix12 --kind refit --out <parts>/mlp-parity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
KEEP = [ID, "airport", "month", "bundle:cls"]


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def saved(path: Path, digest: str, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    assert sha(path) == digest, f"{path}: SHA-256 differs from its record"
    v = pd.read_parquet(path, columns=[ID, "prediction"])
    k = v[ID].to_numpy("int64")
    assert pd.Index(k).is_unique and np.isin(k, ids).all(), f"{path}: ids"
    x = v.prediction.to_numpy("float64")
    assert np.isfinite(x).all(), f"{path}: nonfinite value"
    return pd.Series(x, index=k).reindex(ids).to_numpy("float64"), {"path": str(path), "sha256": digest, "rows": int(len(v))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v73-parts", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="board exp-164 work tree (prc-exp164-work)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pq, js = a.out / "mlp-refs.parquet", a.out / "mlp-refs.json"
    assert not pq.exists() and not js.exists(), f"{pq}: write-once"
    want = json.loads(a.v73_parts.with_suffix(".json").read_text())["output_sha256"]
    assert sha(a.v73_parts) == want, "v73-parts SHA-256 differs from its record"
    v = pd.read_parquet(a.v73_parts, columns=KEEP + ["primary_frame:member_scope", "primary_frame:mlp"])
    k = v[ID].to_numpy("float64")
    ids = k.astype("int64")
    assert len(v) == 344_841 and np.array_equal(ids, k) and pd.Index(ids).is_unique
    fit = json.loads((a.work / "out/mlp-008/final/FIT.json").read_text())
    assert fit["fit"] == "final" and fit["seeds"] == [701, 702]
    m8, s8 = saved(Path(fit["prediction"]["path"]), fit["prediction"]["sha256"], ids)
    bag = json.loads((a.work / "out/mlp-012/vectors/BAG-mix.json").read_text())
    final = [x for x in bag["vectors"] if x["name"] == "final"]
    assert len(final) == 1
    m12, s12 = saved(Path(final[0]["path"]), final[0]["sha256"], ids)
    scope = v["primary_frame:member_scope"].to_numpy(bool)
    clock = v["bundle:cls"].astype(str).to_numpy() == "clock"
    rule = np.where(scope & clock, m12, m8)
    rel = v["primary_frame:mlp"].to_numpy("float64")
    same = (rule.view("u8") == rel.view("u8")) | (np.isnan(rule) & np.isnan(rel))
    assert same.all(), f"the released rule differs from primary_frame:mlp on {int((~same).sum())} rows"
    out = v[KEEP].copy()
    out["saved:mlp008"] = m8
    out["saved:mix12"] = m12
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    rec = {"what": "exp-186 saved MLP008 and mix12 final vectors on the board rows (v73-parts layout)",
           "v73_parts": {"path": str(a.v73_parts), "sha256": want},
           "columns": {"saved:mlp008": s8, "saved:mix12": s12},
           "rule_check": {"rows": int(len(v)), "bits_equal_primary_frame_mlp": int(same.sum()),
                          "scope_clock_rows": int((scope & clock).sum()), "nan_rows": int(np.isnan(rel).sum())},
           "output": str(pq), "output_sha256": sha(pq), "code_sha256": sha(__file__)}
    js.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"event": "refs-ready", "output_sha256": rec["output_sha256"], **rec["rule_check"]}))


if __name__ == "__main__":
    main()
