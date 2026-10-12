"""The release entries of the 15-seed s7 set (s7x15 delta review PASS, docs/status/2026-10-11-s7x15-review/DESK-NOTE.md;
exp-193, PLAN-S15.md): run (v), R9-15 and LD-15, each with its DH step.

    PRC_RANKING_MODE=final OMP_NUM_THREADS=2 .venv/bin/python s15_entries.py --release <release worktree> \
        --inputs <folder of the copied files> --chain <folder with chain_ids_{fold,ranking}.parquet> \
        --data-root <dir with data/raw> [--write]

Inputs, each checked by SHA-256 (copied from $PRC_FINAL_ROOT with gget.sh): finalv's 15-seed member files
(run-s7x15 work/w2/s7_{fold,ranking}.parquet, MANIFEST row s7c15-held-fold and the chain log lines 468-469), the npz of
run (iv)'s ORD-S that run (v)'s L2C and the R9-15 tree read (92aa25c8, log lines 784 and 1628), finalv's run (v) V file
(vA15-file), gainlane's R9-15 and LD-15 V files and their dh files (exp-194 MANIFEST rows v-r9-15-file, r9-15-dh-file,
v-ld15-file, ld15-dh-file). The chain IDs are V's.

With --write it writes into the release tree:
- release/saved-vectors/s7c15/final-{fold,ranking}.json: the saved 15-seed member values (float64 seed means). Each
  decodes, with the chain IDs, to the member file's bytes, and its float32 cast equals the npz array of the split.
- release/saved-vectors/final-entries/s15/v-{v,r9,r9-ld}.json: the saved V values of the three V entries.
- the entries final-v, final-v-dh, final-v-r9, final-v-dh-r9, final-v-r9-ld and final-v-dh-r9-ld after
  final-iii-dh-r9-ld, and default_target final-v-dh-r9-ld. Each V entry is a copy of its 9-seed sibling (final-iii,
  final-iii-r9, final-iii-r9-ld) with the 15-seed member, the closure commit of the run and the x86 L2C and V hashes;
  c5 reads no s7 value, so c5 stays. Each DH entry is a copy of its sibling on the new base.
Checks: LD-15 moves 6,752 rows against R9-15 (exp-194 REPORT-R9 section 13.7) and no DH row; release/dh_stage.py's
formula on the saved V values gives finalv's run (v) dh file (833533ad, final_dh.py) and gainlane's R9-15 and LD-15 dh
files byte for byte; on the DH rows each dh file equals its 9-seed sibling's values.
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
ROWS = {"fold": 344_419, "ranking": 670_790}
FLOOR = 60.0
INPUTS = {  # path under --inputs: SHA-256 (finalv and gainlane MANIFEST rows, finalv chain log)
    "finalv/run-s7x15/work/w2/s7_fold.parquet": "40bdf34b6ebfcd573ecdef8ef1c46d6a7d28df44d9d3200d360ffbada8c11f66",
    "finalv/run-s7x15/work/w2/s7_ranking.parquet": "f2995bc391d8ba3d22fcc56ca3a9122abd14eb411ec9d8f20cec56caf6621e03",
    "finalv/run-port15-46a2f4ab7/work/w2/s7c-final.npz": "92aa25c815248bd20e57ecf707fdf2a3172275dea8b368a7e852566b644b7f68",
    "finalv/port15/root-A15/finalv/youthful-giraffe_final.parquet":
        "a9146e9d3c057e5fbad3c7acf7f8e448efbbfdfdbe02f9e073948fc6ab34e903",
    "gainlane/r9-15-a/root-r9/finalv/youthful-giraffe_final.parquet":
        "18a7bfb77bfdc5a68387cc9d96acbf3f276d441748fa7c2fd807a340e753ab82",
    "gainlane/r9-15-a/dh-r9/youthful-giraffe_final.parquet":
        "5ce37d698943fed7a1f512aacf9500140ea7ee0809592670fe3a73ec2039fb7e",
    "gainlane/r9-15-a/root-ld15/finalv/youthful-giraffe_final.parquet":
        "31b09989b2575af1b9438bd59ed876caa0e049d95c91a9f2a3832326a5bce759",
    "gainlane/r9-15-a/dh-ld15/youthful-giraffe_final.parquet":
        "eea300c01d53f4665071fbbd63084a1558a05af7fd52decb60363e1762491b02",
}
CHAIN = {"fold": "47a0128f337c04ff8622577f635ad1d9c37101840a99ac49997606c8cfeb4486",
         "ranking": "fd895855236c8931283695d631944e628ee1a6176e3f4e2319115496c08d26c3"}
RUN_V_DH = "833533adfc04"  # prefix of final_dh.py on vA15-file (results/s15-entries-mac/final-dh-vA15/)
# entry: 9-seed sibling, V file, x86 L2C (finalv chain log line 1664; gainlane MANIFEST row l2c-r9-15), closure
# commit (MANIFEST code_commit), dh file (None: the formula's bytes are checked against RUN_V_DH)
RUNS = {
    "final-v": ("final-iii", "finalv/port15/root-A15/finalv/youthful-giraffe_final.parquet",
                "a33dae4132f23a78cbad7ec0bad39661822684b7074d332e914d4546f99f4e2d", "46a2f4ab7", None),
    "final-v-r9": ("final-iii-r9", "gainlane/r9-15-a/root-r9/finalv/youthful-giraffe_final.parquet",
                   "b09ec11091a8b02054b8a730790c73678eb00950807911dd1df5cfcd98bca8b7", "757fbb271",
                   "gainlane/r9-15-a/dh-r9/youthful-giraffe_final.parquet"),
    "final-v-r9-ld": ("final-iii-r9-ld", "gainlane/r9-15-a/root-ld15/finalv/youthful-giraffe_final.parquet",
                      "b09ec11091a8b02054b8a730790c73678eb00950807911dd1df5cfcd98bca8b7", "757fbb271",
                      "gainlane/r9-15-a/dh-ld15/youthful-giraffe_final.parquet"),
}
LD_ROWS = 6_752  # exp-194 REPORT-R9 section 13.7: LD-15 against R9-15
S7_SAVED = "release/saved-vectors/s7c15/final-{split}.json"
V_SAVED = {"final-v": "v-v", "final-v-r9": "v-r9", "final-v-r9-ld": "v-r9-ld"}
NAMES = ("final-v", "final-v-dh", "final-v-r9", "final-v-dh-r9", "final-v-r9-ld", "final-v-dh-r9-ld")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def write_vector(release: Path, rel: str, values: np.ndarray, R, rows: int, write: bool) -> dict:
    body = json.dumps(R.encode_array(pd.Series(values.astype("float64")))) + "\n"
    spec = {"public_vector": rel, "public_vector_sha256": hashlib.sha256(body.encode()).hexdigest()}
    if write:
        out = release / rel
        if out.exists():
            raise SystemExit(f"refusing to replace {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        back = R.decode_array(out, spec["public_vector_sha256"], rows)
        assert np.array_equal(back, values.astype("float64"), equal_nan=True), rel
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--chain", type=Path, required=True)
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
        assert got == expected, f"{rel}: {got} is not the producer's file"
    book_path = a.release / "release/ENTRIES.json"
    book_text = book_path.read_text()
    book = json.loads(book_text)
    assert json.dumps(book, indent=2) + "\n" == book_text, "ENTRIES.json is not in the writer's form"
    entries = book["entries"]
    assert not set(NAMES) & set(entries), "a 15-seed entry exists"
    assert book["default_target"] in entries and list(entries)[-1] == "final-iii-dh-r9-ld"
    template = pd.read_parquet(a.data_root / "data/raw/final_submitting.parquet")[ID].to_numpy("float64")
    report: dict = {}

    # 1. the 15-seed member: saved values, member file bytes again, float32 arrays equal to the npz
    npz = np.load(a.inputs / "finalv/run-port15-46a2f4ab7/work/w2/s7c-final.npz")
    assert sorted(npz.files) == ["fold", "ranking"], npz.files
    member = {"name": "s7", "seeds_per_mode": 15}
    for split in ("fold", "ranking"):
        chain_path = a.chain / f"chain_ids_{split}.parquet"
        assert sha256(chain_path) == CHAIN[split], f"{chain_path} is not V's chain_ids file"
        chain = pd.read_parquet(chain_path)[ID].to_numpy()
        src = a.inputs / f"finalv/run-s7x15/work/w2/s7_{split}.parquet"
        frame = pd.read_parquet(src)
        assert list(frame.columns) == [ID, "s7"] and len(frame) == ROWS[split] == len(chain)
        assert np.array_equal(frame[ID].to_numpy(), chain), f"{split}: member IDs differ from the chain IDs"
        values = frame["s7"].to_numpy("float64")
        spec = write_vector(a.release, S7_SAVED.format(split=split), values, R, ROWS[split], a.write)
        again = hashlib.sha256(table_bytes(pd.DataFrame({ID: chain, "s7": values}))).hexdigest()
        assert again == sha256(src), f"{split}: the member file written from the saved values differs"
        narrow = values.astype("<f4")
        assert npz[split].dtype == np.float32 and np.array_equal(npz[split], narrow, equal_nan=True), \
            f"{split}: the float32 cast of the member values differs from the npz array"
        member[split] = {**spec, "rows": ROWS[split], "file_sha256": sha256(src),
                         "float32_sha256": hashlib.sha256(narrow.tobytes()).hexdigest(),
                         "finite_rows": int(np.isfinite(values).sum())}
    report["member"] = member

    # 2. V entries (copies of the 9-seed siblings) and DH entries (release/dh_stage.py's formula)
    sib_dh = entries["final-iii-dh-r9-ld"]["member"]
    w = R.decode_array(a.release / sib_dh["w_pair"]["public_vector"], sib_dh["w_pair"]["public_vector_sha256"],
                       ROWS["ranking"])
    pred = R.decode_array(a.release / sib_dh["pred_DH"]["public_vector"], sib_dh["pred_DH"]["public_vector_sha256"],
                          ROWS["ranking"])
    on = w > 0
    new, values_of = {}, {}
    for name, (sibling, v_rel, l2c, closure, dh_rel) in RUNS.items():
        sib = entries[sibling]
        assert sib["member"]["seeds_per_mode"] == 9 and sib["member"].get("l2c_npz") == "work/w2/s7c-l2c.npz"
        src = a.inputs / v_rel
        v = pd.read_parquet(src)
        assert list(v.columns) == [ID, Y] and np.array_equal(v[ID].to_numpy("float64"), template)
        values = v[Y].to_numpy("float64")
        values_of[name] = values
        body = table_bytes(pd.DataFrame({ID: template, Y: values}))
        assert hashlib.sha256(body).hexdigest() == sha256(src)
        v_spec = write_vector(a.release, f"release/saved-vectors/final-entries/s15/{V_SAVED[name]}.json", values, R,
                              ROWS["ranking"], a.write)
        e = json.loads(json.dumps(sib))
        e["member"] = {**member, **{k: sib["member"][k] for k in ("l2c_npz", "ords_flags") if k in sib["member"]}}
        e["closure_commit"] = closure
        e["intermediate_sha256"] = {"c5": sib["intermediate_sha256"]["c5"], "w2-l2c": {"Linux-x86_64": l2c},
                                    "v73": {"Linux-x86_64": sha256(src)}}
        e.update({"comparison_vector": v_spec, "bytes": len(body), "sha256": sha256(src),
                  "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(body)))})
        assert list(e) == list(sib) and list(e["member"]) == list(sib["member"])
        new[name] = e
        out = values.copy()
        out[on] = np.maximum(FLOOR, values[on] + w[on] * (pred[on] - values[on]))
        dh_body = table_bytes(pd.DataFrame({ID: template, Y: out}))
        dh_sha = hashlib.sha256(dh_body).hexdigest()
        if dh_rel:
            assert dh_body == (a.inputs / dh_rel).read_bytes(), f"{name}: DH step differs from gainlane's dh file"
        else:
            assert dh_sha.startswith(RUN_V_DH), f"{name}: DH step differs from final_dh.py's file"
        sib_v = legacy.saved_array(sib["comparison_vector"], ROWS["ranking"])
        sib_out = np.maximum(FLOOR, sib_v[on] + w[on] * (pred[on] - sib_v[on]))
        assert np.array_equal(out[on], sib_out), f"{name}: a DH row differs from the 9-seed sibling's"
        sib_d = entries[f"{sibling.replace('final-iii', 'final-iii-dh', 1)}"]
        d = json.loads(json.dumps(sib_d))
        d.update({"base": name, "base_sha256": e["sha256"], "bytes": len(dh_body), "sha256": dh_sha,
                  "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(dh_body)))})
        assert list(d) == list(sib_d)
        new[name.replace("final-v", "final-v-dh", 1)] = d
    moved = values_of["final-v-r9-ld"] != values_of["final-v-r9"]
    assert int(moved.sum()) == LD_ROWS and not (moved & on).any(), int(moved.sum())
    r9_moved = values_of["final-v-r9"] != values_of["final-v"]
    d15 = values_of["final-v-r9-ld"] - legacy.saved_array(entries["final-iii-r9-ld"]["comparison_vector"], ROWS["ranking"])
    report["entries"] = {n: {"sha256": new[n]["sha256"], "bytes": new[n]["bytes"]} for n in NAMES}
    report["r9_15_against_run_v"] = {"rows_moved": int(r9_moved.sum()), "dh_rows_moved": int((r9_moved & on).sum())}
    report["ld15_against_r9_15"] = {"rows_moved": int(moved.sum()), "dh_rows_moved": 0}
    report["ld15_v_against_ld9_v"] = {"rows_moved": int((d15 != 0).sum()), "dh_rows_moved": int(((d15 != 0) & on).sum()),
                                      "rms_on_moved_s": float(np.sqrt(np.mean(d15[d15 != 0] ** 2))),
                                      "mean_on_moved_s": float(np.mean(d15[d15 != 0])),
                                      "largest_s": float(np.abs(d15).max())}

    if a.write:
        book["entries"] = {**entries, **{n: new[n] for n in NAMES}}
        assert list(book["entries"])[-len(NAMES):] == list(NAMES)
        book["default_target"] = "final-v-dh-r9-ld"
        book_path.write_text(json.dumps(book, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
