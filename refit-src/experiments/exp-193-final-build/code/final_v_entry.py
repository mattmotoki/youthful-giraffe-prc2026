"""Saved values and the ENTRIES.json body of the release's final V entry `final` (exp-193, release step 2c part 3).

    .venv/bin/python final_v_entry.py --release <release worktree> --v-file <youthful-giraffe_final.parquet> \
        --data-root <dir with data/raw> [--intermediate <json {name: {platform: sha256}}>] --out <entry json>

Inputs: finalv's V file (MANIFEST row v-file, 121e33b9…). Writes release/saved-vectors/final-entries/v.json with the
release's encoder (assembly_replay.encode_array) and checks it with the reader of release/v73_stage.py: the decoded
V values with final_submitting's IDs give the V file's bytes. The member values are the saved s7 values of release
step 2b (release/saved-vectors/s7c/final-{fold,ranking}.json, written by s7_final_vectors.py from the two member
files that the V build copied, 390733bc… and 8e3956fd…; v73_stage.member_files checks them against those files by
SHA-256 at each replay). The entry gets the 16 official files by
SHA-256 (the 14 of v73 and the two final files), the step code of the release tree (v73's map, all equal, plus
chain_ids.py), and the x86-64 Linux hashes as the reference platform. It writes the entry body to --out; the
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
V_FILE = "121e33b9880a7a4907ceeb50c8c6ba587529b502fd2c41d570df8925a72f098a"
S7_SAVED = "release/saved-vectors/s7c/final-{split}.json"
S7 = {"fold": "390733bc8634b9d898faea76f7786d6158fc8cde48200df9b0331a2b01556471",
      "ranking": "8e3956fd5c929b68e0f66155f5e16bad4e13d1c00b2310d04a75723fa8095deb"}
V64_FINAL = "8cfb8efb552eb0ee545a3d7ccf476c711157c44757287496b59099b5e70f4af1"
ROWS = {"fold": 344_419, "ranking": 670_790}
FINAL_FILES = ("final_ranking.parquet", "final_submitting.parquet")
NEW_CODE = ("chain_ids.py",)
OUT_V = "release/saved-vectors/final-entries/v.json"
CLOSURE = "66b11c579"  # finalv's code of the V run (exp-191 REPORT section 8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def write_vector(release: Path, rel: str, values: pd.Series, R, rows: int) -> dict:
    out = release / rel
    if out.exists():
        raise SystemExit(f"refusing to replace {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R.encode_array(values.astype("float64"))) + "\n")
    spec = {"public_vector": rel, "public_vector_sha256": sha256(out)}
    back = R.decode_array(release / rel, spec["public_vector_sha256"], rows)
    assert np.array_equal(back, values.to_numpy("float64"), equal_nan=True), rel
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--v-file", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--intermediate", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import assembly_replay as R  # noqa: E402
    import build_v32 as legacy  # noqa: E402

    book = json.loads((a.release / "release/ENTRIES.json").read_text())
    entries = book["entries"]
    assert "final" not in entries, "entry final exists"
    assert entries["v64-final"]["sha256"] == V64_FINAL
    v73 = entries["v73"]

    # V values
    assert sha256(a.v_file) == V_FILE, f"{a.v_file} is not finalv's V file"
    v = pd.read_parquet(a.v_file)
    assert list(v.columns) == [ID, Y] and len(v) == ROWS["ranking"], (list(v.columns), len(v))
    raw = a.data_root / "data/raw"
    template = pd.read_parquet(raw / "final_submitting.parquet")[ID].to_numpy("float64")
    assert np.array_equal(v[ID].to_numpy("float64"), template), "V is not in final_submitting order"
    v_spec = write_vector(a.release, OUT_V, v[Y], R, ROWS["ranking"])
    saved = legacy.saved_array(v_spec, ROWS["ranking"])
    body = table_bytes(pd.DataFrame({ID: template, Y: saved}))
    assert hashlib.sha256(body).hexdigest() == V_FILE, "saved V values with the template IDs do not give the V bytes"

    # s7 member values: the saved values of release step 2b
    member = {"name": "s7", "seeds_per_mode": 3}
    for split in ("fold", "ranking"):
        rel = S7_SAVED.format(split=split)
        spec = {"public_vector": rel, "public_vector_sha256": sha256(a.release / rel)}
        back = R.decode_array(a.release / rel, spec["public_vector_sha256"], ROWS[split])
        assert len(back) == ROWS[split]
        member[split] = {**spec, "rows": ROWS[split], "file_sha256": S7[split]}

    # official files and step code
    official = dict(v73["official_sha256"])
    for name in FINAL_FILES:
        official[name] = sha256(raw / name)
    assert official["final_submitting.parquet"] == R.sha256_file(raw / "final_submitting.parquet")
    for name, expected in official.items():
        assert sha256(raw / name) == expected, f"data/raw/{name} differs from the registered hash"
    code = {}
    for rel in sorted([*v73["code"], *NEW_CODE]):
        code[rel] = sha256(a.release / rel)
        if rel in v73["code"]:
            assert code[rel] == v73["code"][rel], f"{rel} differs from the v73 entry's step code"

    intermediate = {"c5": {}, "w2-l2c": {}, "v73": {"Linux-x86_64": V_FILE}}
    if a.intermediate:
        for name, by_platform in json.loads(a.intermediate.read_text()).items():
            intermediate[name].update(by_platform)
    frame = pd.read_parquet(io.BytesIO(body))
    entry = {"recipe": "steps", "stage": "v73-steps", "mode": "final", "base": "v64-final", "base_sha256": V64_FINAL,
             "closure_commit": CLOSURE, "official_sha256": official, "code": code, "member": member,
             "intermediate_sha256": intermediate, "reference_platform": "Linux-x86_64", "comparison_vector": v_spec,
             "rows": ROWS["ranking"], "id_dtype": "float64", "bytes": len(body), "sha256": V_FILE,
             "value_sha256": legacy.value_sha256(frame)}
    a.out.write_text(json.dumps(entry, indent=2) + "\n")
    print(json.dumps({k: entry[k] for k in ("comparison_vector", "bytes", "sha256", "value_sha256")}, indent=1))
    print("member", {s: member[s]["public_vector_sha256"][:16] for s in ("fold", "ranking")}, "code", len(code),
          "official", len(official))


if __name__ == "__main__":
    main()
