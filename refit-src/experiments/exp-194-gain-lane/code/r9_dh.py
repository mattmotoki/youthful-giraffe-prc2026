"""The dh step (part dh-final1) on a V file of the R9 chain (SPEC-R9.md).

The consumer and the writer are those of release73's `experiments/exp-193-final-build/code/final_dh.py` (copied:
`table_bytes`, the scope checks, value = max(60, V + w_pair (pred_DH - V)) where w_pair > 0, V elsewhere). Its parity
with the part's diff record against V 121e33b9 is left out: R9 moves V on the scope rows. With --expect-sha256 the
output (written first) must have that SHA-256 (the code check: the run (iii) V file 6de02372 gives final-iii-dh c4511650).

    $PY -I r9_dh.py --v-file <V file> --part <dh-final1.parquet> --raw <dir with final_*.parquet> --out-dir <new dir>
        [--expect-sha256 <sha>]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
PART = "a89366d17a80ee751fb23b7331d9c6ff907896c8bf12a59f6bc7702d11d1ff4f"
ROWS, W, SCOPE_ROWS = 670_790, 0.164, 14_587
SCOPE_CELLS = {"1-EDDF": 1_557, "2-EDDF": 12_374, "6-EDDF": 656}
FLOOR = 60.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_bytes(ids: np.ndarray, values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({ID: ids, Y: values}).to_parquet(buffer, index=False)
    return buffer.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-file", type=Path, required=True)
    ap.add_argument("--part", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expect-sha256")
    a = ap.parse_args()
    out, rec_path = a.out_dir / "youthful-giraffe_final.parquet", a.out_dir / "dh-record.json"
    assert not a.out_dir.exists(), f"{a.out_dir} exists; write-once"
    assert sha256(a.part) == PART, f"{a.part} is not the part dh-final1"
    ids = pd.read_parquet(a.raw / "final_submitting.parquet", columns=[ID])[ID].to_numpy("float64")
    v = pd.read_parquet(a.v_file)
    part = pd.read_parquet(a.part, columns=[ID, "pred_DH", "has_head", "w_pair"])
    assert list(v.columns) == [ID, Y] and len(v) == len(part) == len(ids) == ROWS
    assert np.array_equal(v[ID].to_numpy("float64"), ids) and np.array_equal(part[ID].to_numpy("float64"), ids)
    vv = v[Y].to_numpy("float64")
    assert table_bytes(ids, vv) == a.v_file.read_bytes(), "the writer does not give the V file's bytes"
    w, dh = part["w_pair"].to_numpy("float64"), part["pred_DH"].to_numpy("float64")
    on = w > 0
    assert int(on.sum()) == SCOPE_ROWS and set(np.unique(w).tolist()) == {0.0, W}
    assert np.isfinite(vv).all() and np.isfinite(dh[on]).all() and not part["has_head"].to_numpy(bool)[on].any()
    raw = pd.read_parquet(a.raw / "final_ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"])
    raw = raw[raw.PHASE_mvt.eq("DEP")].set_index(ID).reindex(ids)
    month = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    airport = raw.ADEP_mvt.astype(str).to_numpy()
    cell = np.char.add(np.char.add(month.astype(str), "-"), airport.astype(str))
    got = {k: int(n) for k, n in zip(*np.unique(cell[on], return_counts=True))}
    assert got == SCOPE_CELLS, got
    raw_new = vv[on] + w[on] * (dh[on] - vv[on])
    new = vv.copy()
    new[on] = np.maximum(FLOOR, raw_new)
    assert np.array_equal(new[~on].view(np.uint64), vv[~on].view(np.uint64)), "a row outside the scope moved"
    assert np.isfinite(new).all() and new.min() >= FLOOR
    body = table_bytes(ids, new)
    got_sha = hashlib.sha256(body).hexdigest()
    a.out_dir.mkdir(parents=True)
    out.write_bytes(body)
    d = new - vv
    rec = {"what": "dh step (dh-final1, release73 final_dh.py consumer) on a V file of the R9 chain",
           "v_file": {"path": str(a.v_file), "sha256": sha256(a.v_file)}, "part": {"path": str(a.part), "sha256": PART},
           "output": {"path": str(out), "sha256": got_sha, "bytes": len(body)},
           "expect_sha256": a.expect_sha256, "expect_match": got_sha == a.expect_sha256 if a.expect_sha256 else None,
           "moved": int((d != 0).sum()),
           "floor_60_binds": int((raw_new < FLOOR).sum()), "min_s": float(new.min()),
           "code": {"r9_dh.py": sha256(Path(__file__))}}
    rec_path.write_text(json.dumps(rec, indent=1) + "\n")
    print("R9DH", json.dumps(rec))
    assert not a.expect_sha256 or got_sha == a.expect_sha256, (got_sha, a.expect_sha256)


if __name__ == "__main__":
    main()
