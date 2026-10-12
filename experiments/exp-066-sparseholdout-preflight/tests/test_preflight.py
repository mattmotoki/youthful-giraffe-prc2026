from pathlib import Path
import importlib.util
import hashlib
import json
import subprocess
import sys
import tarfile
import unittest

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import cold_controls as C
import future_wrapper as W
import may_protocol as M
import receipts as R
import member_adapters as AD
import routes


class PreflightTests(unittest.TestCase):
 def test_cold_controls_and_seed_contract(self):
    assert list(M.SEED_SEARCH) == list(range(1, 100001))
    assert np.array_equal(C.sham_table([1, 2, 3, 4], [0, 0, 1, 1], [1, 0, 3, 2]), [2, 1, 4, 3])
    assert np.array_equal(C.zero_singletons([3, 4, 5], [0, 1, 1]), [0, 4, 5])
    with self.assertRaises(ValueError): C.sham_table([1, 2], [0, 0], [0, 1])


 def test_may_only_and_prelabel_stops(self):
    import tempfile
    tmp_path = Path(tempfile.mkdtemp())
    M.assert_one_fold(["2025-05"])
    with self.assertRaises(ValueError): M.assert_one_fold(["2025-05", "2025-07"])
    with self.assertRaises(ValueError): M.refuse_recombination(True)
    with self.assertRaisesRegex(RuntimeError, "before label read"): M.require_matched_null(tmp_path/"none", "x")
    with self.assertRaises(FileNotFoundError): M.require_draws(tmp_path / "missing.npz", "x", [], [])
    with self.assertRaises(ValueError): M.freeze_draws(pd.DataFrame({"MVT_ID_mvt":[1], "utc_date":["2025-05-01"], "TAXITIME_SEC_mvt":[9]}), tmp_path/"bad.npz")
    out = tmp_path / "draws.npz"
    M.freeze_draws(pd.DataFrame({"MVT_ID_mvt":[2,1], "utc_date":["2025-05-02","2025-05-01"]}), out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    M.require_draws(out, digest, [2,1], ["2025-05-02","2025-05-01"])
    z = np.load(out)
    assert np.all((z["draw_l2"][:, 1::2] - z["draw_l2"][:, 0::2]) % len(z["unique_dates"]) == 1)
    with self.assertRaises(ValueError): M.require_draws(out, "0"*64, [2,1], ["2025-05-02","2025-05-01"])
    with self.assertRaises(FileExistsError): M.freeze_draws(pd.DataFrame({"MVT_ID_mvt":[1], "utc_date":["2025-05-01"]}), out)


 def test_forbidden_roots_and_receipt_write_once(self):
    import tempfile
    tmp_path = Path(tempfile.mkdtemp())
    for root in W.FORBIDDEN_ROOTS:
        with self.assertRaises(ValueError): W.reject_forbidden([root + "/oof/x.parquet"])
    out = tmp_path / "out.txt"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(out)!r}).write_text('ok')"]
    R.run("valid", command, tmp_path/"receipts", tmp_path/"valid.log", [out])
    with self.assertRaises(FileExistsError): R.run("valid", command, tmp_path/"receipts", tmp_path/"valid.log", [out])
    start = json.loads((tmp_path/"receipts/valid.start.json").read_text())
    assert start["pid_or_remote_identity"] and start["command"] == command

 def test_routes_gate_before_sentinel_and_effective_threads(self):
    import tempfile
    tmp = Path(tempfile.mkdtemp()); called = []
    for member in ("F1-fresh", "L2-hub8725", "T-tailv4"):
        assert AD.assert_effective(AD.adapter(member))
    with self.assertRaises(ValueError): AD.assert_effective({"lightgbm":{"num_threads":6}})
    with self.assertRaises(ValueError):
        routes.guarded_route("produce", tmp, ["experiments/exp-061-sparseprior/oof/x"],
                             lambda settings: called.append(settings))
    assert called == []
    old_reject, old_raw = routes.gates.reject_forbidden, routes.gates.verify_raw
    events=[]
    try:
        routes.gates.reject_forbidden=lambda paths: events.append("forbidden")
        routes.gates.verify_raw=lambda path: events.append("raw")
        routes.guarded_route("produce", tmp, ["synthetic/input"],
                             lambda settings: events.append("fit"), member="F1-fresh")
    finally:
        routes.gates.reject_forbidden, routes.gates.verify_raw = old_reject, old_raw
    assert events == ["forbidden", "raw", "fit"]

 def test_matched_null_search_and_gate(self):
    import tempfile
    tmp = Path(tempfile.mkdtemp()); out = tmp/"null.json"
    values=np.array([1.,2.,3.,4.]); groups=np.array(["a","a","b","b"])
    rows={m:np.array([0,1,2,3]) for m in ("S","I","V")}
    digest=M.search_matched_null(values,groups,{"a":True,"b":True},rows,np.ones((6,4)),out,shams=2)
    M.require_matched_null(out,digest)
    rec=json.loads(out.read_text()); assert rec["resolved"] and rec["attempts"]
    with self.assertRaises(RuntimeError): M.require_matched_null(out,"0"*64)


 def test_clean_tree_import_closure(self):
    root = Path(__file__).resolve().parents[3]
    allow = (root/"experiments/exp-066-sparseholdout-preflight/source_allowlist.txt").read_text().splitlines()
    assert allow == sorted(allow) and len(allow) == len(set(allow))
    assert "experiments/exp-001-retro-v1/features.py" in allow
    assert "prc/phi_features.py" in allow
    for rel in allow: assert (root/rel).is_file(), rel
    forbidden = ("RESULTS", "/review/", "/oof/", "/work/", "/analysis/")
    assert not any(any(x in rel for x in forbidden) or Path(rel).name == "SPEC.md" for rel in allow)

 def test_built_transport_refusals_and_clean_imports(self):
    import tempfile
    root = Path(__file__).resolve().parents[3]
    source = root/"experiments/exp-066-sparseholdout-preflight/source/transport-data-r3"
    cfg = json.loads((source/"transport.json").read_text())
    expected = [x["name"] for x in cfg["chunks"]]

    def reconstruct(names, mutate=False):
        if names != expected: raise ValueError("missing, extra, or reordered")
        blob = bytearray()
        for item in cfg["chunks"]:
            data = bytearray((source/item["name"]).read_bytes())
            if mutate and item["index"] == 0: data[0] ^= 1
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("changed chunk")
            blob.extend(data)
        if hashlib.sha256(blob).hexdigest() != cfg["archive_sha256"]: raise ValueError("archive")
        return bytes(blob)

    with self.assertRaises(ValueError): reconstruct(expected[:-1])
    with self.assertRaises(ValueError): reconstruct(expected + ["payload.part999"])
    with self.assertRaises(ValueError): reconstruct(list(reversed(expected)))
    with self.assertRaises(ValueError): reconstruct(expected, mutate=True)
    temp = Path(tempfile.mkdtemp())
    archive = temp/"verified.archive"; archive.write_bytes(reconstruct(expected))
    with tarfile.open(archive) as tf: tf.extractall(temp/"repo")
    env = dict(__import__('os').environ); env["PYTHONPATH"] = str(temp/"repo")
    commands = [
      "import sys;sys.path.insert(0,'experiments/exp-018-fresh/code');import run,dataset,model,features",
      "import sys;sys.path.insert(0,'experiments/exp-008-diversity');import run,frames,learners;assert 'exp-001-retro-v1/features.py' in frames.build_features.__code__.co_filename",
      "import sys,inspect;sys.path.insert(0,'experiments/exp-009-noac2');import deliver;from prc.tail_model import PhiModel;assert PhiModel is not None;assert inspect.signature(deliver.TailModel.fit).parameters['use_phi'].default is True",
      "import sys;sys.path.insert(0,'experiments/exp-066-sparseholdout-preflight/code');import cold_controls",
      "import sys;sys.path.insert(0,'experiments/exp-066-sparseholdout-preflight/code');import stage_controller;assert stage_controller.STAGES == ('produce-a','freeze-features','freeze-null','read-labels')",
    ]
    for command in commands:
        subprocess.run([sys.executable, "-c", command], cwd=temp/"repo", env=env, check=True)

 def test_kernel_pins_and_rejects_unsafe_members(self):
    root = Path(__file__).resolve().parents[3]
    text = (root/"experiments/exp-066-sparseholdout-preflight/kernel/kernel.py").read_text()
    for token in ("CONFIG_SHA256", "TRUSTED", "duplicate archive member",
                  "link, device, or special archive member", "target.open(\"xb\")"):
        assert token in text
    cfg = root/"experiments/exp-066-sparseholdout-preflight/source/transport-data-r3/transport.json"
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() in text


if __name__ == "__main__": unittest.main()
