"""The release entries of R9 (gainlane's SPEC-R9, chain r9-f1) beside final-iii-dh (exp-193, PLAN-R9.md).

    PRC_RANKING_MODE=final OMP_NUM_THREADS=2 .venv/bin/python r9_entries.py --release <release worktree> \
        --inputs <folder of gainlane's r9-f1 files> --data-root <dir with data/raw> --closure <code commit> [--write]

Inputs, copied from $PRC_FINAL_ROOT/gainlane/r9-f1 with gget.sh, each checked by SHA-256: the R9 pack
(export-r9/release/ZABD-PACK.json) and its two new saved columns (the mlp column of fold/primary_frame and of
ranking/primary_frame), the zabd part (parts/r9/zabd.parquet: zabd and v64 in final_submitting order), the V file
(root-r9/finalv) and the dh file (dh-r9). The x86 intermediates c5 and L2C are the hashes of run-r9/outputs.

With --write it writes into the release tree:
- release/saved-vectors/final-r9/{fold,ranking}/primary_frame/mlp.json: the two new saved columns, bytes unchanged.
- release/ZABD-PACK-final-r9.json: gainlane's pack with these two paths and no other change. (In gainlane's pack the
  ranking file has the path of the final pack's own column file, release/saved-vectors/final/ranking/...)
- release/saved-vectors/final-entries/r9/{zabd,v-iii}.json: the saved zabd and V values.
- the entries zabd-final-r9, v64-final-r9, final-iii-r9 and final-iii-dh-r9 after final-iii-dh; each copies the
  form of its final-iii chain entry and changes only the R9 facts. default_target stays final-iii-dh.
Then the T11b-L stage on the saved R9 zabd must give the R9 v64 bytes (the stage reads the R9 pack's frames).
Without --write it checks the inputs and prints what it would write.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ROWS = {"fold": 344_419, "ranking": 670_790}
FLOOR = 60.0
MLP = "release/saved-vectors/final/{split}/primary_frame/mlp.json"
INPUTS = {  # path under --inputs: SHA-256 (guarded sha256sum on remote, 10-10 13:20 to 13:27 UTC)
    "export-r9/release/ZABD-PACK.json": "50091dbc5c5c5e104ba42222f0c325d59869ecad39335a53451c907427733f14",
    MLP.format(split="fold").replace("release/", "export-r9/release/", 1):
        "5e83a292b7ba5d7e7a4e0eacecc9b11ad2195c0e7731869c94016a9352f4b462",
    MLP.format(split="ranking").replace("release/", "export-r9/release/", 1):
        "20a7dc7253ad70bb2238229fff4ef649a0ec213af9b584a4baa68bf5653c3d6e",
    "parts/r9/zabd.parquet": "1124b91ca9b6db67bab00dd3dd23b3092e158d4ec3ac07b2be7c0a0c2eba60b2",
    "root-r9/finalv/youthful-giraffe_final.parquet": "1c1ed67bdd7e54182bafb9a3eb08b48e6e708e981af6bb607c0016ca1f010c4b",
    "dh-r9/youthful-giraffe_final.parquet": "4ac66707333eab1b32c43db0f412907e05d35e268d6a869065dba3fdc99b8139",
}
X86 = {"c5": "0f8280246f02162d4232577af04f9a23bdf78dd3bc29028536fdbbce18fcf56a",  # run-r9/outputs/iso-c5.parquet
       "w2-l2c": "d71244b4f41911bb4bcdd96d9bce2e5957aed665d54c44818a66d66268d76116"}  # run-r9/outputs/w2-l2c.parquet
FINAL_PACK = "29c0ace1a5c30d981b962d042db77fbf828a52b5de8d8c194a72a8b996ea3317"
PACK = "release/ZABD-PACK-final-r9.json"
MOVED = {split: f"release/saved-vectors/final-r9/{split}/primary_frame/mlp.json" for split in ROWS}
SAVED = "release/saved-vectors/final-entries/r9/{name}.json"
NAMES = ("zabd-final-r9", "v64-final-r9", "final-iii-r9", "final-iii-dh-r9")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_bytes(ids: np.ndarray, values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({ID: ids, Y: values}).to_parquet(buffer, index=False)
    return buffer.getvalue()


def write_new(path: Path, body: bytes, write: bool) -> None:
    if write:
        if path.exists():
            raise SystemExit(f"refusing to replace {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def vector(release: Path, rel: str, values: np.ndarray, R, rows: int, write: bool) -> dict:
    body = json.dumps(R.encode_array(pd.Series(values.astype("float64")))) + "\n"
    spec = {"public_vector": rel, "public_vector_sha256": hashlib.sha256(body.encode()).hexdigest()}
    write_new(release / rel, body.encode(), write)
    if write:
        back = R.decode_array(release / rel, spec["public_vector_sha256"], rows)
        assert np.array_equal(back, values.astype("float64"), equal_nan=True), rel
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--closure", required=True, help="code commit of gainlane's R9 chain (its MANIFEST row)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("PRC_RANKING_MODE") == "final", "run with PRC_RANKING_MODE=final"
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import assembly_replay as R  # noqa: E402
    import build_v32 as legacy  # noqa: E402
    import t11b_stage as T  # noqa: E402

    for rel, expected in INPUTS.items():
        got = sha256(a.inputs / rel)
        assert got == expected, f"{rel}: {got} is not gainlane's file"
    book_path = a.release / "release/ENTRIES.json"
    book_text = book_path.read_text()
    book = json.loads(book_text)
    assert json.dumps(book, indent=2) + "\n" == book_text, "ENTRIES.json is not in the writer's form"
    entries = book["entries"]
    assert not set(NAMES) & set(entries), "an R9 entry exists"
    assert book["default_target"] == "final-iii-dh"
    zf, vf, f3, f3dh = (entries[n] for n in ("zabd-final", "v64-final", "final-iii", "final-iii-dh"))
    report: dict = {}

    # 1. the R9 pack: against the final pack, only the mlp column of the two primary frames moves
    src_text = (a.inputs / "export-r9/release/ZABD-PACK.json").read_text()
    pack = json.loads(src_text)
    assert json.dumps(pack, indent=1) + "\n" == src_text, "the R9 pack is not in the writer's form"
    assert sha256(a.release / zf["pack"]) == zf["pack_sha256"] == FINAL_PACK
    final = json.loads((a.release / zf["pack"]).read_text())
    assert list(pack["frames"]) == list(final["frames"])
    moved = []
    for key, spec in pack["frames"].items():
        old = final["frames"][key]
        assert {k: v for k, v in spec.items() if k not in ("columns", "digest")} == \
            {k: v for k, v in old.items() if k not in ("columns", "digest")}, key
        assert [c["name"] for c in spec["columns"]] == [c["name"] for c in old["columns"]], key
        for col, was in zip(spec["columns"], old["columns"]):
            if col != was:
                moved.append((key, col["name"]))
        if key.endswith("/primary_frame"):
            assert spec["digest"] != old["digest"], key
        else:
            assert spec["digest"] == old["digest"], key
    assert moved == [("fold/primary_frame", "mlp"), ("ranking/primary_frame", "mlp")], moved
    for split in ROWS:
        col = next(c for c in pack["frames"][f"{split}/primary_frame"]["columns"] if c["name"] == "mlp")
        assert col["kind"] == "saved" and col["file"] == MLP.format(split=split)
        body = (a.inputs / "export-r9" / col["file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == col["sha256"]
        R.decode_array(a.inputs / "export-r9" / col["file"], col["sha256"], ROWS[split])
        col["file"] = MOVED[split]
        write_new(a.release / MOVED[split], body, a.write)
    for key in ("official_sha256", "rows", "target"):
        assert pack[key] == final[key], key
    out_text = json.dumps(pack, indent=1) + "\n"
    pack_sha = hashlib.sha256(out_text.encode()).hexdigest()
    write_new(a.release / PACK, out_text.encode(), a.write)
    t11b = pack["final_build"]["t11b"]
    report["pack"] = {"path": PACK, "sha256": pack_sha, "source_sha256": INPUTS["export-r9/release/ZABD-PACK.json"],
                      "moved_columns": [f"{k}:{c}" for k, c in moved], "candidate_sha256": pack["candidate_sha256"],
                      "v64_sha256": pack["final_build"]["v64_sha256"]}

    # 2. zabd-final-r9: the saved zabd gives the pack's candidate bytes; the part's v64 gives the pack's v64 bytes
    template = pd.read_parquet(a.data_root / "data/raw/final_submitting.parquet")[ID].to_numpy("float64")
    part = pd.read_parquet(a.inputs / "parts/r9/zabd.parquet")
    assert list(part.columns) == [ID, "zabd", "v64"] and len(part) == ROWS["ranking"]
    assert np.array_equal(part[ID].to_numpy("float64"), template), "zabd part IDs differ from final_submitting order"
    zabd, v64 = part["zabd"].to_numpy("float64"), part["v64"].to_numpy("float64")
    zabd_body, v64_body = table_bytes(template, zabd), table_bytes(template, v64)
    assert hashlib.sha256(zabd_body).hexdigest() == pack["candidate_sha256"]
    assert hashlib.sha256(v64_body).hexdigest() == pack["final_build"]["v64_sha256"]
    new = {}
    e = dict(zf)
    e.update({"pack": PACK, "pack_sha256": pack_sha, "parent_sha256": pack["parent_sha256"],
              "with_c65_sha256": pack["with_c65_sha256"],
              "comparison_vector": vector(a.release, SAVED.format(name="zabd"), zabd, R, ROWS["ranking"], a.write),
              "bytes": len(zabd_body), "sha256": pack["candidate_sha256"],
              "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(zabd_body)))})
    assert list(e) == list(zf)
    new["zabd-final-r9"] = e

    # 3. v64-final-r9: T11b-L on zabd-final-r9 (the facts of the R9 build's T11b-L record)
    e = dict(vf)
    e.update({"base": "zabd-final-r9", "base_sha256": pack["candidate_sha256"], "pack": PACK, "pack_sha256": pack_sha,
              "p2_rows": t11b["p2_rows"], "long_taxi_days": t11b["long_taxi_days"], "changed_rows": t11b["changed_rows"],
              "bytes": len(v64_body), "sha256": pack["final_build"]["v64_sha256"],
              "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(v64_body)))})
    assert list(e) == list(vf)
    new["v64-final-r9"] = e

    # 4. final-iii-r9: final-iii's steps, code and member on the R9 chain; x86 intermediates of run-r9
    src = a.inputs / "root-r9/finalv/youthful-giraffe_final.parquet"
    v = pd.read_parquet(src)
    assert list(v.columns) == [ID, Y] and np.array_equal(v[ID].to_numpy("float64"), template)
    values = v[Y].to_numpy("float64")
    body = table_bytes(template, values)
    assert hashlib.sha256(body).hexdigest() == sha256(src)
    e = dict(f3)
    e.update({"base": "v64-final-r9", "base_sha256": pack["final_build"]["v64_sha256"], "closure_commit": a.closure,
              "intermediate_sha256": {"c5": {"Linux-x86_64": X86["c5"]}, "w2-l2c": {"Linux-x86_64": X86["w2-l2c"]},
                                      "v73": {"Linux-x86_64": sha256(src)}},
              "comparison_vector": vector(a.release, SAVED.format(name="v-iii"), values, R, ROWS["ranking"], a.write),
              "bytes": len(body), "sha256": sha256(src),
              "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(body)))})
    assert list(e) == list(f3)
    new["final-iii-r9"] = e

    # 5. final-iii-dh-r9: final-iii-dh's DH member on the R9 V; release/dh_stage.py's formula gives gainlane's file
    m = f3dh["member"]
    w = R.decode_array(a.release / m["w_pair"]["public_vector"], m["w_pair"]["public_vector_sha256"], ROWS["ranking"])
    pred = R.decode_array(a.release / m["pred_DH"]["public_vector"], m["pred_DH"]["public_vector_sha256"],
                          ROWS["ranking"])
    on = w > 0
    out = values.copy()
    out[on] = np.maximum(FLOOR, values[on] + w[on] * (pred[on] - values[on]))
    dh_body = table_bytes(template, out)
    assert dh_body == (a.inputs / "dh-r9/youthful-giraffe_final.parquet").read_bytes(), "DH step differs from gainlane's"
    e = dict(f3dh)
    e.update({"base": "final-iii-r9", "base_sha256": sha256(src), "bytes": len(dh_body),
              "sha256": hashlib.sha256(dh_body).hexdigest(),
              "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(dh_body)))})
    assert list(e) == list(f3dh)
    new["final-iii-dh-r9"] = e
    old_v = legacy.saved_array(f3["comparison_vector"], f3["rows"])
    old_dh = old_v.copy()
    old_dh[on] = np.maximum(FLOOR, old_v[on] + w[on] * (pred[on] - old_v[on]))
    report["entries"] = {n: {"sha256": new[n]["sha256"], "bytes": new[n]["bytes"]} for n in NAMES}
    report["against_final_iii"] = {"zabd_rows_moved": int((zabd != legacy.saved_array(zf["comparison_vector"],
                                                                                      zf["rows"])).sum()),
                                   "v_rows_moved": int((values != old_v).sum()),
                                   "dh_rows_moved": int((out != old_dh).sum()),
                                   "dh_rows_changed_by_dh": int((out != values).sum()),
                                   "v_mean_move_s": float(np.mean(values - old_v)),
                                   "v_rms_move_s": float(np.sqrt(np.mean((values - old_v) ** 2)))}

    if a.write:
        ordered = {}
        for k, item in entries.items():
            ordered[k] = item
            if k == "final-iii-dh":
                ordered.update({n: new[n] for n in NAMES})
        assert len(ordered) == len(entries) + 4 and list(ordered)[-4:] == list(NAMES)
        book["entries"] = ordered
        book_path.write_text(json.dumps(book, indent=2) + "\n")
        # the T11b-L stage of the release on the saved R9 zabd, with the R9 pack's frames
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "zabd-r9.parquet"
            zpath.write_bytes(zabd_body)
            got = T.compose(new["v64-final-r9"], zpath, a.data_root)
        assert hashlib.sha256(got["body"]).hexdigest() == new["v64-final-r9"]["sha256"], "T11b-L differs"
        report["t11b_mac"] = {"sha256": hashlib.sha256(got["body"]).hexdigest(), "facts": got["facts"]}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
