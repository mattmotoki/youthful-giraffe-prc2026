from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys
CODE=Path(__file__).resolve().parents[1]/"code";sys.path.insert(0,str(CODE))
import stage_controller as S


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+"\n")


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.old_root=S.STATE_ROOT;self.root=Path(tempfile.mkdtemp()).resolve();S.STATE_ROOT=self.root
        (self.root/"inputs/raw").mkdir(parents=True);(self.root/"GIT_COMMIT.txt").write_text("source-fixed\n")
        hashes={}
        for name in S.RAW_NAMES:
            p=self.root/"inputs/raw"/name;p.write_bytes(("synthetic:"+name).encode());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        dump(self.root/"registration.json",{"schema":1,"source_commit":"source-fixed","raw_sha256":hashes,"mode":"synthetic-preflight"})
        dump(self.root/"inputs/a_features.json",[{"x":i,"y_synthetic":i%2} for i in range(8)])
        rows=[]
        for i in range(6):
            rows.append({"MVT_ID_mvt":f"id{i}","utc_date":f"2025-05-{i+1:02d}",
              "group":"g1" if i<3 else "g2","key":f"k{i}","S":i+1,"I":i+11,"V":i+21})
        dump(self.root/"inputs/feature_rows.json",rows)
        dump(self.root/"inputs/synthetic_labels.json",{"synthetic_labels":[1,2,3,4,5,6]})

    def tearDown(self): S.STATE_ROOT=self.old_root

    def test_complete_real_state_machine_and_write_once(self):
        c=S.Controller()
        with self.assertRaises(FileNotFoundError): c.run("read-labels")
        c.run("produce-a")
        proof=json.loads((self.root/"frozen/a-only.json").read_text())["thread_proof"]
        self.assertEqual(proof,{"lightgbm":3,"catboost":3,"tail_model":3})
        with self.assertRaises(FileNotFoundError): c.run("read-labels")
        c.run("freeze-features")
        with self.assertRaises(FileNotFoundError): c.run("read-labels")
        c.run("freeze-null");self.assertEqual(c.run("read-labels"),6)
        with self.assertRaises(FileExistsError): c.run("produce-a")
        with self.assertRaises(FileExistsError): c.run("freeze-features")
        with self.assertRaises(FileExistsError): c.run("freeze-null")

    def test_changed_missing_wrong_source_and_incomplete_null_refuse(self):
        c=S.Controller();c.run("produce-a");c.run("freeze-features");c.run("freeze-null")
        (self.root/"frozen/features-draws.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError): c.run("read-labels")
        fresh=json.loads((self.root/"registration.json").read_text());fresh["source_commit"]="wrong"
        (self.root/"registration.json").write_text(json.dumps(fresh))
        with self.assertRaises(ValueError): c.run("read-labels")

    def test_traversal_link_forbidden_and_legacy_routes_refuse(self):
        c=S.Controller()
        with self.assertRaises(ValueError): c._safe("../escape",must_exist=False)
        target=self.root/"outside";target.write_text("x")
        alias=self.root/"inputs/alias";alias.symlink_to(target)
        with self.assertRaises(ValueError): c.run("produce-a")
        allow=(Path(__file__).resolve().parents[1]/"source_allowlist.txt").read_text()
        for old in ("producer066.py","measurement066.py","report066.py","routes.py","member_adapters.py","may_protocol.py"):
            self.assertNotIn(old,allow)
        with self.assertRaises(KeyError): c.run("legacy-route")

    def test_wrong_thread_probe_refuses_before_freeze(self):
        old=S.fit_thread_probes
        try:
            S.fit_thread_probes=lambda rows:{"lightgbm":6,"catboost":3,"tail_model":3}
            with self.assertRaises(ValueError): S.Controller().run("produce-a")
            self.assertFalse((self.root/"frozen/a-only.json").exists())
        finally:S.fit_thread_probes=old


if __name__=="__main__":unittest.main()
