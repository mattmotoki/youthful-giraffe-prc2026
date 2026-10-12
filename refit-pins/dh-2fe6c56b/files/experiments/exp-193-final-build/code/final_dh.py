"""The dh stage after V: part dh-final1 on the V file (exp-193, PLAN-D166 step 1).

    .venv/bin/python final_dh.py --v-file <V file> --part <dh-final1.parquet> --raw <dir with final_ranking.parquet
        and final_submitting.parquet> --diff-ref <exp-194 out/part-final1/dh-final1-diff-v.json> --out-dir <dir>

value = max(60, V + w_pair (pred_DH - V)) where w_pair > 0, and V elsewhere: the consumer step of exp-194 dh_apply.py
and part_diff.py (desk 13:37 UTC 10-09). V is finalv's V file (121e33b9...), the part is the gainlane's MANIFEST
row dh-final1 (a89366d1...). Writes <out-dir>/youthful-giraffe_final.parquet (the candidate; V stays the final file
until the review verdict) in the byte form of the V file, and <out-dir>/final-dh-record.json.

Checks (inline, fail closed): input SHA-256; IDs of V and the part in final_submitting order; w_pair is 0.164 on
14,587 rows and 0 elsewhere; on those rows pred_DH is finite, has_head is false, the airport is EDDF and the months are
January, February and June; rows outside the scope are byte-equal to V; the writer gives the V file's bytes from V's
values; the diff against V per month and airport equals the part's own diff record (part_diff.py, rounded as there);
the H8 file checks (schema, rows, finite values, minimum 60 s).
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
V_FILE = "121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a"
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


def stats(d: np.ndarray, moved: np.ndarray, floor: np.ndarray) -> dict:
    # the fields and rounding of exp-194 part_diff.py
    dm = d[moved]
    return {"rows": int(len(d)), "moved": int(moved.sum()), "rms_all_s": round(float(np.sqrt(np.mean(d ** 2))), 3),
            "rms_moved_s": round(float(np.sqrt(np.mean(dm ** 2))), 3) if len(dm) else 0.0,
            "mean_moved_s": round(float(np.mean(dm)), 3) if len(dm) else 0.0,
            "max_abs_s": round(float(np.max(np.abs(d))), 3), "floor_60_rows": int(floor.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-file", type=Path, required=True)
    ap.add_argument("--part", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--diff-ref", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    out, rec_path = a.out_dir / "youthful-giraffe_final.parquet", a.out_dir / "final-dh-record.json"
    for p in (out, rec_path):
        if p.exists():
            raise SystemExit(f"{p} exists; write-once")

    assert sha256(a.v_file) == V_FILE, f"{a.v_file} is not finalv's V file"
    assert sha256(a.part) == PART, f"{a.part} is not the part dh-final1"
    ids = pd.read_parquet(a.raw / "final_submitting.parquet", columns=[ID])[ID].to_numpy("float64")
    v = pd.read_parquet(a.v_file)
    part = pd.read_parquet(a.part, columns=[ID, "pred_DH", "has_head", "w_pair"])
    assert list(v.columns) == [ID, Y] and len(v) == len(part) == len(ids) == ROWS, (list(v.columns), len(v), len(part))
    assert np.array_equal(v[ID].to_numpy("float64"), ids), "V is not in final_submitting order"
    assert np.array_equal(part[ID].to_numpy("float64"), ids), "the part is not in final_submitting order"
    vv = v[Y].to_numpy("float64")
    assert table_bytes(ids, vv) == a.v_file.read_bytes(), "the writer does not give the V file's bytes"

    w, dh = part["w_pair"].to_numpy("float64"), part["pred_DH"].to_numpy("float64")
    on = w > 0
    assert int(on.sum()) == SCOPE_ROWS and set(np.unique(w).tolist()) == {0.0, W}, (int(on.sum()), np.unique(w))
    assert np.isfinite(vv).all() and np.isfinite(dh[on]).all()
    assert not part["has_head"].to_numpy(bool)[on].any(), "a scoped row has a head value"

    raw = pd.read_parquet(a.raw / "final_ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"])
    raw = raw[raw.PHASE_mvt.eq("DEP")].set_index(ID).reindex(ids)
    assert raw.ADEP_mvt.notna().all()
    month = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    airport = raw.ADEP_mvt.astype(str).to_numpy()
    cell = np.char.add(np.char.add(month.astype(str), "-"), airport.astype(str))
    got_cells = dict(zip(*np.unique(cell[on], return_counts=True)))
    assert {k: int(n) for k, n in got_cells.items()} == SCOPE_CELLS, got_cells

    raw_new = vv[on] + w[on] * (dh[on] - vv[on])
    new = vv.copy()
    new[on] = np.maximum(FLOOR, raw_new)
    floor = np.zeros(ROWS, dtype=bool)
    floor[np.flatnonzero(on)[raw_new < FLOOR]] = True
    assert np.array_equal(new[~on].view(np.uint64), vv[~on].view(np.uint64)), "a row outside the scope moved"
    assert np.isfinite(new).all() and new.min() >= FLOOR

    d = new - vv
    moved = d != 0
    by_cell = {}
    for c in sorted(set(cell[on]) | set(cell[moved])):
        idx = cell == c
        by_cell[c] = stats(d[idx], moved[idx], floor[idx])
    total = stats(d, moved, floor)
    ref = json.loads(a.diff_ref.read_text())
    parity = {"all_rows": total == ref["all_rows"],
              "by_month_airport": by_cell == ref["by_month_airport"],
              "part": ref["part"]["sha256"] == PART}
    assert all(parity.values()), (parity, total, ref["all_rows"])

    body = table_bytes(ids, new)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    back = pd.read_parquet(out)
    assert list(back.columns) == [ID, Y] and np.array_equal(back[Y].to_numpy("float64"), new)
    janjul = np.isin(month, (1, 7))
    record = {
        "what": "D166 stage after V: max(60, V + w_pair (pred_DH - V)) where w_pair > 0 (part dh-final1), V elsewhere",
        "inputs": {"v_file": {"path": str(a.v_file), "sha256": V_FILE}, "part": {"path": str(a.part), "sha256": PART},
                   "final_submitting": sha256(a.raw / "final_submitting.parquet"),
                   "final_ranking": sha256(a.raw / "final_ranking.parquet"),
                   "diff_ref": {"path": str(a.diff_ref), "sha256": sha256(a.diff_ref)}},
        "output": {"path": str(out), "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)},
        "h8": {"columns": list(back.columns), "rows": len(back), "ids_in_template_order": True,
               "finite": bool(np.isfinite(new).all()), "min_s": float(new.min()), "max_s": float(new.max()),
               "floor_60_binds": int(floor.sum())},
        "scope": {"rows": SCOPE_ROWS, "w": W, "cells": SCOPE_CELLS},
        "diff_v": {"all_rows": total, "by_month_airport": by_cell,
                   "january_july_rows_moved": int((moved & janjul).sum())},
        "parity_with_part_diff": parity,
        "code": {"final_dh.py": sha256(Path(__file__))},
    }
    rec_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("output", "h8", "parity_with_part_diff")}, indent=1))
    print("diff_v", json.dumps(total), "january_july_rows_moved", record["diff_v"]["january_july_rows_moved"])


if __name__ == "__main__":
    main()
