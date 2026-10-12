"""The release entries of LD (isolated-5's ORD-S on long-taxi-day rows; gainlane exp-194 REPORT-LD.md, MANIFEST rows
v-ld, v-ld-file, ld-dh-file) beside final-iii-dh-r9 (exp-193, PLAN-LD.md).

    PRC_RANKING_MODE=final OMP_NUM_THREADS=2 .venv/bin/python ld_entries.py --release <release worktree> \
        --inputs <folder of gainlane's ld-b files> --ords-ld <exp-194 code/ords_ld.py> --data-root <dir with data/raw> \
        [--write]

Inputs, each checked by SHA-256: gainlane's LD V file (root-ld/finalv, copied from $PRC_FINAL_ROOT/gainlane/
ld-b with gget.sh), the LD dh file (dh-ld) and code/ords_ld.py (finalv's w2/ords.py with the flag --long-days; the
code of job ld-b, commit fa170fcd0). LD changes only ORD-S: every stage before it is the R9 chain's (MANIFEST row
v-ld: c5 0f828024, l2c d71244b4), and the 2026 β are equal to R9's (REPORT-LD section 3).

With --write it writes into the release tree:
- w2/ords_ld.py: gainlane's ords_ld.py, bytes unchanged; the LD entry copies it into the run folder as w2/ords.py
  (`code_sources`), as final-iii does with candidate A's L2C.
- release/saved-vectors/final-entries/r9-ld/v-iii.json: the saved LD V values.
- the entries final-iii-r9-ld and final-iii-dh-r9-ld after final-iii-dh-r9: copies of final-iii-r9 and
  final-iii-dh-r9 with the LD facts (code map w2/ords.py = the ords_ld.py hash, member.ords_flags ["--long-days"],
  closure commit fa170fcd0, x86 V hash). default_target stays final-iii-dh-r9.
- experiments/exp-149-release-zabd/imported-sources.json: w2/ords_ld.py pinned at the source of w2/ords.py.
release/v73_stage.py passes member.ords_flags to ORD-S in the final mode (a separate edit of the same commit).
Checks: the LD V file differs from the R9 V entry on the rows that REPORT-LD counts (6,753) and keeps every other
row bit for bit; release/dh_stage.py's formula on the saved LD V gives gainlane's dh file; no DH row moves.
Without --write it checks the inputs and prints what it would write.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ROWS = 670_790
FLOOR = 60.0
INPUTS = {  # path under --inputs: SHA-256 (MANIFEST rows v-ld-file and ld-dh-file of exp-194)
    "root-ld/finalv/youthful-giraffe_final.parquet": "260b427b83f99921b26ce372bad78fbcb3718b402e7a8bd0725a914ae3da114d",
    "dh-ld/youthful-giraffe_final.parquet": "07d4a4e6344f5c681761a00a851d23dd9dd6bc8675861179b0de80b4fe669865",
}
ORDS_LD = "1da4f3a09fd4b26b70413ec2df3a5fb6a0d96de84b50ec8b6c9f393684976551"  # MANIFEST row v-ld
ORDS = "d2b5c7cd1a6e52d9794e8833bfcf909cd2b6d852b02cda58cf8038534dc20acf"     # release w2/ords.py (finalv's)
CLOSURE = "fa170fcd0"  # code commit of job ld-b (MANIFEST rows v-ld to ld-dh-file)
LD_ROWS = 6_753        # REPORT-LD section 1: the 2026 rows that LD changes
CODE = "w2/ords_ld.py"
SAVED = "release/saved-vectors/final-entries/r9-ld/v-iii.json"
NAMES = ("final-iii-r9-ld", "final-iii-dh-r9-ld")
IMPORTS = "experiments/exp-149-release-zabd/imported-sources.json"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--ords-ld", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("PRC_RANKING_MODE") == "final", "run with PRC_RANKING_MODE=final"
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import assembly_replay as R  # noqa: E402
    import build_v32 as legacy  # noqa: E402

    for rel, expected in INPUTS.items():
        got = sha256(a.inputs / rel)
        assert got == expected, f"{rel}: {got} is not gainlane's file"
    code = a.ords_ld.read_bytes()
    assert hashlib.sha256(code).hexdigest() == ORDS_LD, "ords_ld.py is not the code of job ld-b"
    assert sha256(a.release / "w2/ords.py") == ORDS
    book_path = a.release / "release/ENTRIES.json"
    book_text = book_path.read_text()
    book = json.loads(book_text)
    assert json.dumps(book, indent=2) + "\n" == book_text, "ENTRIES.json is not in the writer's form"
    entries = book["entries"]
    assert not set(NAMES) & set(entries), "an LD entry exists"
    assert book["default_target"] == "final-iii-dh-r9"
    f3, f3dh = entries["final-iii-r9"], entries["final-iii-dh-r9"]
    assert f3["code"]["w2/ords.py"] == ORDS and "w2/ords.py" not in f3["code_sources"]
    report: dict = {}

    # 1. final-iii-r9-ld: final-iii-r9 with ords_ld.py as the run's w2/ords.py and the flag --long-days
    template = pd.read_parquet(a.data_root / "data/raw/final_submitting.parquet")[ID].to_numpy("float64")
    src = a.inputs / "root-ld/finalv/youthful-giraffe_final.parquet"
    v = pd.read_parquet(src)
    assert list(v.columns) == [ID, Y] and np.array_equal(v[ID].to_numpy("float64"), template)
    values = v[Y].to_numpy("float64")
    body = table_bytes(template, values)
    assert hashlib.sha256(body).hexdigest() == sha256(src)
    r9 = legacy.saved_array(f3["comparison_vector"], f3["rows"])
    moved = values != r9
    assert int(moved.sum()) == LD_ROWS, int(moved.sum())
    write_new(a.release / CODE, code, a.write)
    spec_body = json.dumps(R.encode_array(pd.Series(values))) + "\n"
    write_new(a.release / SAVED, spec_body.encode(), a.write)
    saved = {"public_vector": SAVED, "public_vector_sha256": hashlib.sha256(spec_body.encode()).hexdigest()}
    if a.write:
        back = R.decode_array(a.release / SAVED, saved["public_vector_sha256"], ROWS)
        assert np.array_equal(back, values), SAVED
    e = json.loads(json.dumps(f3))
    e["code"]["w2/ords.py"] = ORDS_LD
    e["code_sources"]["w2/ords.py"] = CODE
    e["member"]["ords_flags"] = ["--long-days"]
    e.update({"closure_commit": CLOSURE,
              "intermediate_sha256": {**f3["intermediate_sha256"], "v73": {"Linux-x86_64": sha256(src)}},
              "comparison_vector": saved, "bytes": len(body), "sha256": sha256(src),
              "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(body)))})
    assert list(e) == list(f3)
    new = {"final-iii-r9-ld": e}

    # 2. final-iii-dh-r9-ld: final-iii-dh-r9's DH member on the LD V; release/dh_stage.py's formula gives gainlane's file
    m = f3dh["member"]
    w = R.decode_array(a.release / m["w_pair"]["public_vector"], m["w_pair"]["public_vector_sha256"], ROWS)
    pred = R.decode_array(a.release / m["pred_DH"]["public_vector"], m["pred_DH"]["public_vector_sha256"], ROWS)
    on = w > 0
    out = values.copy()
    out[on] = np.maximum(FLOOR, values[on] + w[on] * (pred[on] - values[on]))
    dh_body = table_bytes(template, out)
    assert dh_body == (a.inputs / "dh-ld/youthful-giraffe_final.parquet").read_bytes(), "DH step differs from gainlane's"
    assert not (moved & on).any(), "LD moves a DH row"
    e = dict(f3dh)
    e.update({"base": "final-iii-r9-ld", "base_sha256": sha256(src), "bytes": len(dh_body),
              "sha256": hashlib.sha256(dh_body).hexdigest(),
              "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(dh_body)))})
    assert list(e) == list(f3dh)
    new["final-iii-dh-r9-ld"] = e
    r9_dh = r9.copy()
    r9_dh[on] = np.maximum(FLOOR, r9[on] + w[on] * (pred[on] - r9[on]))
    assert hashlib.sha256(table_bytes(template, r9_dh)).hexdigest() == f3dh["sha256"]
    d = out - r9_dh
    report["entries"] = {n: {"sha256": new[n]["sha256"], "bytes": new[n]["bytes"]} for n in NAMES}
    report["against_final_iii_dh_r9"] = {"rows_moved": int((d != 0).sum()), "dh_rows_moved": int(((d != 0) & on).sum()),
                                         "rms_on_moved_s": float(np.sqrt(np.mean(d[d != 0] ** 2))),
                                         "mean_on_moved_s": float(np.mean(d[d != 0])),
                                         "largest_s": float(np.abs(d).max())}

    # 3. the import record: w2/ords_ld.py pinned at the source of w2/ords.py (as w2/l2c_candidate_a.py at w2/l2c.py)
    imports_path = a.release / IMPORTS
    imports_text = imports_path.read_text()
    imports = json.loads(imports_text)
    paths = [item["path"] for item in imports["files"]]
    assert CODE not in paths
    ords_rec = next(item for item in imports["files"] if item["path"] == "w2/ords.py")
    rec = {"path": CODE, "source_revision": ords_rec["source_revision"], "source_path": "w2/ords.py",
           "original_sha256": ords_rec["original_sha256"]}
    files = []
    for item in imports["files"]:
        files.append(item)
        if item["path"] == "w2/l2c_candidate_a.py":
            files.append(rec)
    assert len(files) == len(imports["files"]) + 1
    imports["files"] = files
    report["import_record"] = rec

    if a.write:
        for form in (1, 2, None):
            if json.dumps(json.loads(imports_text), indent=form) + "\n" == imports_text:
                break
        else:
            raise SystemExit("imported-sources.json is not in a json.dumps form")
        imports_path.write_text(json.dumps(imports, indent=form) + "\n")
        ordered = {}
        for k, item in entries.items():
            ordered[k] = item
            if k == "final-iii-dh-r9":
                ordered.update({n: new[n] for n in NAMES})
        assert len(ordered) == len(entries) + 2 and list(ordered)[-2:] == list(NAMES)
        book["entries"] = ordered
        book_path.write_text(json.dumps(book, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
