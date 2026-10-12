"""The frozen type code table of the released HB3u frame (exp-190 SPEC, design row "type table").

    $PY experiments/exp-190-final-clock/code/actype_table.py <released frame12.parquet> <out.json>

Applies the released rule of exp-159 `common.derived` to the released twelve-month frame: the code table is
every aircraft type of the frame in sorted order, and the frequent types are those with `ACTYPE_MIN_ROWS` clock
rows or more. Reads no label. Write-once.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ACTYPE_MIN_ROWS = 5000            # exp-159 common.py (SPEC-push addendum 7)
RELEASED_FRAME12 = "250fb20b69ebadfd923a80da4759b14ce7135c71674e1d41370e4f031a664f64"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    frame, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists; write-once")
    digest = sha(frame)
    if digest != RELEASED_FRAME12:
        raise SystemExit(f"{frame}: SHA-256 {digest} is not the released frame12 {RELEASED_FRAME12}")
    typ = pd.read_parquet(frame, columns=["AIRCRAFT_TYPE_mvt"])["AIRCRAFT_TYPE_mvt"].astype("string")
    counts = typ.value_counts()
    types = sorted(typ.dropna().unique())
    common = sorted(t for t in types if counts[t] >= ACTYPE_MIN_ROWS)
    rec = {"what": "exp-190 frozen type code table of the released HB3u frame (exp-159 common.derived rule)",
           "frame": str(frame), "frame_sha256": digest, "rows": int(len(typ)), "floor": ACTYPE_MIN_ROWS,
           "types": types, "common": common, "n_types": len(types), "n_common": len(common),
           "common_rows_share": round(float(typ.isin(common).mean()), 4),
           "near_floor": {t: int(counts[t]) for t in types if 3000 <= counts[t] < ACTYPE_MIN_ROWS}}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("n_types", "n_common", "common_rows_share", "near_floor")}))


if __name__ == "__main__":
    main()
