"""Saved DH values and the ENTRIES.json body of the release entry `final-dh` (exp-193, PLAN-D166 step 2).

    .venv/bin/python dh_entry.py --release <release worktree> --part <dh-final1.parquet> --data-root <dir with
        data/raw> --candidate <final_dh.py output file> --out <entry json>

Input: the gainlane's part dh-final1 (MANIFEST row, a89366d1...; columns ID, pred_DH, has_head, w_pair in
final_submitting order). Writes release/saved-vectors/final-entries/dh/{pred_DH,has_head,w_pair}.json with the
release's encoder and checks that the decoded vectors with the final_submitting IDs give the part's bytes (the check
of release/dh_stage.py). The entry gets the two final official files by SHA-256, the stage code, the scope, and the
bytes of the candidate that final_dh.py wrote from the V file and the part. It writes the entry body to --out; the
caller adds it to ENTRIES.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
PART = "a89366d17a80ee751fb23b7331d9c6ff907896c8bf12a59f6bc7702d11d1ff4f"
V_FILE = "121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a"
ROWS, W, SCOPE_ROWS = 670_790, 0.164, 14_587
OUT = "release/saved-vectors/final-entries/dh/{col}.json"
FINAL_FILES = ("final_ranking.parquet", "final_submitting.parquet")
CODE = ("release/dh_stage.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--part", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import assembly_replay as R  # noqa: E402
    import build_v32 as legacy  # noqa: E402

    book = json.loads((a.release / "release/ENTRIES.json").read_text())
    assert "final-dh" not in book["entries"], "entry final-dh exists"
    assert book["entries"]["final"]["sha256"] == V_FILE
    assert sha256(a.part) == PART, f"{a.part} is not the part dh-final1"
    raw = a.data_root / "data/raw"
    ids = pd.read_parquet(raw / "final_submitting.parquet")[ID].to_numpy("float64")
    part = pd.read_parquet(a.part)
    assert list(part.columns) == [ID, "pred_DH", "has_head", "w_pair"] and len(part) == ROWS
    assert np.array_equal(part[ID].to_numpy("float64"), ids), "the part is not in final_submitting order"
    assert table_bytes(part) == a.part.read_bytes(), "the writer does not give the part's bytes"

    member = {"name": "DH", "part_sha256": PART, "w": W, "scope_rows": SCOPE_ROWS,
              "scope": "EDDF typed no-head rows with dark_b, January, February and June 2026"}
    decoded = {}
    for col in ("pred_DH", "has_head", "w_pair"):
        rel = OUT.format(col=col)
        out = a.release / rel
        if out.exists():
            raise SystemExit(f"refusing to replace {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(R.encode_array(part[col])) + "\n")
        member[col] = {"public_vector": rel, "public_vector_sha256": sha256(out), "rows": ROWS}
        decoded[col] = R.decode_array(out, member[col]["public_vector_sha256"], ROWS)
    rebuilt = pd.DataFrame({ID: ids, "pred_DH": decoded["pred_DH"], "has_head": decoded["has_head"].astype(bool),
                            "w_pair": decoded["w_pair"]})
    assert hashlib.sha256(table_bytes(rebuilt)).hexdigest() == PART, "saved DH vectors do not give the part bytes"
    w = decoded["w_pair"]
    assert int((w > 0).sum()) == SCOPE_ROWS and set(np.unique(w).tolist()) == {0.0, W}

    official = {name: sha256(raw / name) for name in FINAL_FILES}
    assert official["final_submitting.parquet"] == book["entries"]["final"]["official_sha256"]["final_submitting.parquet"]
    code = {rel: sha256(a.release / rel) for rel in CODE}
    body = a.candidate.read_bytes()
    frame = pd.read_parquet(io.BytesIO(body))
    assert np.array_equal(frame[ID].to_numpy("float64"), ids) and len(frame) == ROWS
    entry = {"recipe": "steps", "stage": "dh-step", "mode": "final", "base": "final", "base_sha256": V_FILE,
             "official_sha256": official, "code": code, "member": member, "rows": ROWS, "id_dtype": "float64",
             "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "value_sha256": legacy.value_sha256(frame)}
    a.out.write_text(json.dumps(entry, indent=2) + "\n")
    print(json.dumps({k: entry[k] for k in ("bytes", "sha256", "value_sha256")}, indent=1))
    print({c: member[c]["public_vector_sha256"][:16] for c in ("pred_DH", "has_head", "w_pair")})


if __name__ == "__main__":
    main()
