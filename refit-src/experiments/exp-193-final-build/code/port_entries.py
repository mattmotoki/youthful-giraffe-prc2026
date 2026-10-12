"""The release entries of the port runs (ii) and (iii) and their DH steps (exp-193, PLAN-PORT.md; desk 05:37 UTC 10-10).

    .venv/bin/python port_entries.py --release <release worktree> --inputs <folder of finalv's run files> \
        --chain <folder with chain_ids_{fold,ranking}.parquet> --l2c-diff <isolated5-A/l2c.diff> \
        --data-root <dir with data/raw> [--write]

Inputs, each checked by SHA-256 against finalv's records (MANIFEST rows s7c9-held-fold, v9-file, vA9-file; the chain
log results/port/logs/finalv-port-chain89-4d8a6532a.log; review question 6): the 9-seed member files of run (ii)
(work/w2/s7_{fold,ranking}.parquet), the npz that run (ii)'s ORD-S saved and run (iii)'s L2C read (s7c-final.npz),
run (iii)'s w2/l2c.py (candidate A: the release w2/l2c.py plus isolated5-A/l2c.diff), the two V files and the
chain_ids files (the same as V's).

With --write it writes into the release tree:
- release/saved-vectors/s7c9/final-{fold,ranking}.json: the saved member values (float64 seed means). Each decodes,
  with the chain IDs, to the member file's bytes (the form of v73_stage.member_files), and its float32 cast equals
  the npz array of the same split (the form of w2/ords.py --save-npz).
- w2/l2c_candidate_a.py: the patched l2c.py; the stage copies it as the run's w2/l2c.py.
- release/saved-vectors/final-entries/v-{ii,iii}.json: the saved V values; with final_submitting's IDs they give the
  V file's bytes.
- the four entries final-ii, final-ii-dh, final-iii, final-iii-dh in release/ENTRIES.json, after final-dh, and
  default_target final-iii-dh. The V entries copy the official files, the step code and the chain of entry final;
  the DH entries copy entry final-dh's member (the same part a89366d1) on the new base.
Without --write it checks the inputs and prints what it would write.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ROWS = {"fold": 344_419, "ranking": 670_790}
FLOOR = 60.0
INPUTS = {  # path under --inputs: SHA-256 (finalv's records)
    "run-port9-a0c484bab/work/w2/s7_fold.parquet": "7060e1604900decd8493470595a0da150b8ace13077dd3a6fd6ab0c7d6217a1e",
    "run-port9-a0c484bab/work/w2/s7_ranking.parquet": "3614ed1e94c5331c313d7d098e0a5ddc1b2d62864a972dbb4520e1a61383ae46",
    "run-port9-a0c484bab/work/w2/s7c-final.npz": "505ceafd1059287e674777654dbe62269b1880770b3512b1abdb8a61cb59fc0b",
    "run-portA9-a0c484bab/w2/l2c.py": "c9931e0a6e2192b2391df6d4b4a74b67ab2592934ce5eb420dc177f59e82c9ae",
    "port/root-9/finalv/youthful-giraffe_final.parquet":
        "280db2058f10a5cc3d44acdcf20aed80aae143cfadf2c9c299ddf7212c4f7afc",
    "port/root-A9/finalv/youthful-giraffe_final.parquet":
        "6de02372de5d1aea062300a88af7af51606ac2d6aee645c384b09398c9c57378",
}
CHAIN = {"fold": "47a0128f337c04ff8622577f635ad1d9c37101840a99ac49997606c8cfeb4486",
         "ranking": "fd895855236c8931283695d631944e628ee1a6176e3f4e2319115496c08d26c3"}
L2C_RELEASE = "c9cd2869063d5534e81b37ed57675994ee71a2f57eed3a22ca7b78dc5f2358f9"
L2C_DIFF = "9e72607ad896dc09bc7c822c97767bfde2c5e5255bd9d92333a5bae5d086e743"
C5 = "2f0130a84e934e65f7445d538f79ffb4ae493e04638808942d125d631085ad8f"
RUNS = {  # entry: V file, L2C output (chain log lines 204 and 1329), the port's code commit (MANIFEST)
    "final-ii": {"v_file": "port/root-9/finalv/youthful-giraffe_final.parquet",
                 "l2c": "87972433f677a93f628e3a5207ba483214713901949c82dfb8198c21a3bc9023", "candidate_a": False},
    "final-iii": {"v_file": "port/root-A9/finalv/youthful-giraffe_final.parquet",
                  "l2c": "72c3e909c8788e086974fe115184798fdb1f9823284376daffbaeb02c28b2fbb", "candidate_a": True},
}
CLOSURE = "a0c484bab"
S7_SAVED = "release/saved-vectors/s7c9/final-{split}.json"
L2C_A = "w2/l2c_candidate_a.py"
L2C_NPZ = "work/w2/s7c-l2c.npz"


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
    ap.add_argument("--l2c-diff", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import assembly_replay as R  # noqa: E402
    import build_v32 as legacy  # noqa: E402

    for rel, expected in INPUTS.items():
        got = sha256(a.inputs / rel)
        assert got == expected, f"{rel}: {got} is not finalv's file"
    book = json.loads((a.release / "release/ENTRIES.json").read_text())
    entries = book["entries"]
    assert not {"final-ii", "final-ii-dh", "final-iii", "final-iii-dh"} & set(entries), "an entry exists"
    final, final_dh = entries["final"], entries["final-dh"]
    raw = a.data_root / "data/raw"
    template = pd.read_parquet(raw / "final_submitting.parquet")[ID].to_numpy("float64")
    report: dict = {}

    # 1. the 9-seed member: saved values, member file bytes again, float32 arrays equal to the npz
    npz = np.load(a.inputs / "run-port9-a0c484bab/work/w2/s7c-final.npz")
    assert sorted(npz.files) == ["fold", "ranking"], npz.files
    member = {"name": "s7", "seeds_per_mode": 9}
    for split in ("fold", "ranking"):
        chain_path = a.chain / f"chain_ids_{split}.parquet"
        assert sha256(chain_path) == CHAIN[split], f"{chain_path} is not V's chain_ids file"
        chain = pd.read_parquet(chain_path)[ID].to_numpy()
        src = a.inputs / f"run-port9-a0c484bab/work/w2/s7_{split}.parquet"
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

    # 2. candidate A's l2c.py: the release file plus the diff, equal to run (iii)'s file
    assert sha256(a.release / "w2/l2c.py") == L2C_RELEASE and sha256(a.l2c_diff) == L2C_DIFF
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "w2").mkdir()
        (Path(td) / "w2/l2c.py").write_bytes((a.release / "w2/l2c.py").read_bytes())
        subprocess.run(["patch", "-p1", "--no-backup-if-mismatch", "-s", "-i", str(a.l2c_diff.resolve())], cwd=td,
                       check=True)
        patched = (Path(td) / "w2/l2c.py").read_bytes()
    assert patched == (a.inputs / "run-portA9-a0c484bab/w2/l2c.py").read_bytes()
    l2c_a = hashlib.sha256(patched).hexdigest()
    if a.write:
        out = a.release / L2C_A
        if out.exists():
            raise SystemExit(f"refusing to replace {out}")
        out.write_bytes(patched)

    # 3. V entries and DH entries
    dh_member = final_dh["member"]
    w = R.decode_array(a.release / dh_member["w_pair"]["public_vector"], dh_member["w_pair"]["public_vector_sha256"],
                       ROWS["ranking"])
    pred = R.decode_array(a.release / dh_member["pred_DH"]["public_vector"],
                          dh_member["pred_DH"]["public_vector_sha256"], ROWS["ranking"])
    on = w > 0
    v_old = legacy.saved_array(final["comparison_vector"], final["rows"])
    new = {}
    for name, run in RUNS.items():
        src = a.inputs / run["v_file"]
        v = pd.read_parquet(src)
        assert list(v.columns) == [ID, Y] and np.array_equal(v[ID].to_numpy("float64"), template)
        values = v[Y].to_numpy("float64")
        tag = name.split("-")[1]
        v_spec = write_vector(a.release, f"release/saved-vectors/final-entries/v-{tag}.json", values, R,
                              ROWS["ranking"], a.write)
        body = table_bytes(pd.DataFrame({ID: template, Y: values}))
        assert hashlib.sha256(body).hexdigest() == sha256(src)
        entry = {k: final[k] for k in ("recipe", "stage", "mode", "base", "base_sha256", "official_sha256")}
        entry["closure_commit"] = CLOSURE
        entry["code"] = dict(final["code"])
        if run["candidate_a"]:
            entry["code"]["w2/l2c.py"] = l2c_a
            entry["code_sources"] = {"w2/l2c.py": L2C_A}
        entry["member"] = dict(member, **({"l2c_npz": L2C_NPZ} if run["candidate_a"] else {}))
        entry["l2c_seeds"] = 9 if run["candidate_a"] else 3
        entry["intermediate_sha256"] = {"c5": {"Linux-x86_64": C5}, "w2-l2c": {"Linux-x86_64": run["l2c"]},
                                        "v73": {"Linux-x86_64": sha256(src)}}
        entry.update({"reference_platform": "Linux-x86_64", "comparison_vector": v_spec, "rows": ROWS["ranking"],
                      "id_dtype": "float64", "bytes": len(body), "sha256": sha256(src),
                      "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(body)))})
        new[name] = entry
        # the DH step on this V (release/dh_stage.py's formula)
        out = values.copy()
        out[on] = np.maximum(FLOOR, values[on] + w[on] * (pred[on] - values[on]))
        dh_body = table_bytes(pd.DataFrame({ID: template, Y: out}))
        dh = {k: final_dh[k] for k in ("recipe", "stage", "mode", "official_sha256", "code", "member")}
        dh.update({"base": name, "base_sha256": entry["sha256"], "rows": ROWS["ranking"], "id_dtype": "float64",
                   "bytes": len(dh_body), "sha256": hashlib.sha256(dh_body).hexdigest(),
                   "value_sha256": legacy.value_sha256(pd.read_parquet(io.BytesIO(dh_body)))})
        new[f"{name}-dh"] = dh
        d = values - v_old
        report[name] = {"v_sha256": entry["sha256"], "dh_sha256": dh["sha256"],
                        "rows_moved_against_V": int((d != 0).sum()),
                        "dh_rows_moved_against_V": int((d[on] != 0).sum()),
                        "dh_rows_changed_by_dh": int((out != values).sum()),
                        "floor_rows": int((values[on] + w[on] * (pred[on] - values[on]) < FLOOR).sum())}
    report["l2c_candidate_a_sha256"] = l2c_a
    if a.write:
        ordered = {}
        for k, e in entries.items():
            ordered[k] = e
            if k == "final-dh":
                ordered.update({k2: new[k2] for k2 in ("final-ii", "final-ii-dh", "final-iii", "final-iii-dh")})
        assert len(ordered) == len(entries) + 4
        book["entries"] = ordered
        book["default_target"] = "final-iii-dh"
        (a.release / "release/ENTRIES.json").write_text(json.dumps(book, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
