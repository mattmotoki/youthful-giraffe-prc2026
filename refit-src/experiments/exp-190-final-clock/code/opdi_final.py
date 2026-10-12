"""Final-mode OPDI input of exp-190: the saved 2025 rows of exp-155 and finaldata's rows of the final file.

    $PY experiments/exp-190-final-clock/code/opdi_final.py --saved <exp-155 opdi-first.parquet>
        --final <finaldata opdi/first-final/opdi-first.parquet> --out <parquet>

frame.py and frame12.py join OPDI to every frame row, the 2025 training rows too. finaldata's file of the final
file holds only its 670,790 departures (no 2025 row), so the final mode reads the 2025 rows (utc_day in 2025) of
the saved exp-155 file, which the released frame used, and every row of the final file. The two files have the
same columns and types, and no id is in both. Reads no label. Write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

ID = "MVT_ID_mvt"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rec_path = a.out.with_suffix(".json")
    for p in (a.out, rec_path):
        if p.exists():
            raise SystemExit(f"{p} exists; write-once")
    t0 = time.time()
    saved, final = pd.read_parquet(a.saved), pd.read_parquet(a.final)
    if list(saved.columns) != list(final.columns) or not (saved.dtypes == final.dtypes).all():
        raise SystemExit("the saved and final OPDI files differ in columns or types")
    keep = saved.utc_day.astype(str).str.startswith("2025").to_numpy()
    out = pd.concat([saved[keep], final], ignore_index=True)
    if out[ID].duplicated().any():
        raise SystemExit("an id is in both files")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    rec = {"what": "exp-190 final-mode OPDI input: the saved 2025 rows and the final file's rows",
           "rows": int(len(out)), "columns": list(out.columns),
           "rows_by_month": out.utc_day.astype(str).str[:7].value_counts().sort_index().to_dict(),
           "sources": [{"path": str(a.saved), "sha256": sha(a.saved), "rows": int(len(saved)),
                        "rows_taken": int(keep.sum()), "rule": "utc_day in 2025"},
                       {"path": str(a.final), "sha256": sha(a.final), "rows": int(len(final)),
                        "rows_taken": int(len(final)), "rule": "all rows"}],
           "output_sha256": sha(a.out), "seconds": round(time.time() - t0, 1),
           "code_sha256": {"opdi_final.py": sha(Path(__file__))}}
    rec_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rec[k] for k in ("rows", "rows_by_month", "output_sha256")}), flush=True)


if __name__ == "__main__":
    main()
