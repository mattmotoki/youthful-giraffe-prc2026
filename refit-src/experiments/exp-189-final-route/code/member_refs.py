"""exp-189: reference table of the saved v73-era route members, in the layout of finalcore's
`v73-parts.parquet` (344,841 board rows, template order), so that exp-186 `parity.py` reads a rebuilt
member part against its saved `final` vector.

    python member_refs.py --v73-parts <v73-parts.parquet> --work <prc-exp148-work> --out <dir>

Columns: `MVT_ID_mvt`, `airport`, `month`, `bundle:cls` (copied from v73-parts), and `saved:<name>` for
each member `final` vector of the R4mks-final record (`member_artifacts`, SHA-256 checked) and for the
saved level-2 output `R4mks-final.parquet` (`pred_Sb`, `anchor_served`). Write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
MEMBERS = {"T-tailv4": "y_pred", "C2-cat": "y_pred", "L2-hub8725": "y_pred", "D1-winter": "y_pred",
           "F1-fresh": "y_pred", "N1-nmtail": "y_pred", "L": "pred_L", "Lp": "pred_Lp"}
LEVEL2 = {"R4mks-Sb": "pred_Sb", "R4mks-anchor": "anchor_served"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v73-parts", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pq, js = a.out / "member-refs.parquet", a.out / "member-refs.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    base = pd.read_parquet(a.v73_parts, columns=[ID, "airport", "month", "bundle:cls"])
    ids = base[ID].to_numpy("float64")
    assert len(ids) == 344_841 and not pd.Index(ids).duplicated().any()
    ids = ids.astype("int64")
    lvl2 = a.work / "stack/R4mks-final.parquet"
    rec2 = json.loads(lvl2.with_suffix(".json").read_text())
    arts = rec2["member_artifacts"]          # {"<member>-<split>": sha256}
    sources = {}
    for m, col in MEMBERS.items():
        want = arts[f"{m}-final"]
        cands = sorted(a.work.glob(f"out/{m}-final.parquet")) + sorted(a.work.glob(f"kaggle/*/out/{m}-final.parquet"))
        path = next((c for c in cands if sha(c) == want), None)
        if path is None:
            raise SystemExit(f"{m}: no saved final vector with the record's SHA-256 {want} in {cands}")
        v = pd.read_parquet(path, columns=[ID, col])
        s = pd.Series(v[col].to_numpy("float64"), index=v[ID].astype("int64").to_numpy())
        assert s.index.is_unique and set(s.index) <= set(ids)
        base[f"saved:{m}"] = s.reindex(ids).to_numpy()
        sources[m] = {"path": str(path), "sha256": sha(path), "record_sha256": want,
                      "rows": int(len(s)), "nan": int(s.isna().sum())}
    v = pd.read_parquet(lvl2)
    v = v.set_index(v[ID].astype("int64"))
    for name, col in LEVEL2.items():
        base[f"saved:{name}"] = v[col].reindex(ids).to_numpy("float64")
    sources["R4mks-final"] = {"path": str(lvl2), "sha256": sha(lvl2), "rows": int(len(v))}
    a.out.mkdir(parents=True, exist_ok=True)
    base.to_parquet(pq, index=False)
    rec = {"what": "exp-189 saved v73-era route members and R4mks level-2 columns on the board rows",
           "v73_parts_sha256": sha(a.v73_parts), "sources": sources, "rows": int(len(base)),
           "columns": list(base.columns), "sha256": sha(pq), "code_sha256": sha(Path(__file__))}
    js.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
