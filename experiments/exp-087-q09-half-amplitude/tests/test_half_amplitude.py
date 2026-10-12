from __future__ import annotations
import importlib.util, json, math, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np

EXP = Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
h=load("h087",EXP/"code/half_amplitude.py")
p=load("p087",EXP/"code/phase_b.py")

class HalfAmplitudeTests(unittest.TestCase):
    def test_half_before_floor(self):
        np.testing.assert_array_equal(h.half_update([61,61],[ -4, 4]),[60,63])
    def test_exact_amplitude(self): self.assertEqual(h.AMPLITUDE,.5)
    def test_mask_identity(self):
        base=np.array([100.,100.]); np.testing.assert_array_equal(h.half_update(base,[0,2]),[100,101])
    def test_noac_hold(self):
        full=np.array([3.,0.]); self.assertEqual(h.half_update([100,100],full)[1],100)
    def test_dtype_failure(self):
        with self.assertRaises(TypeError): h.canonical_id_bytes(np.array([1],dtype="int64"))
    def test_duplicate_id_failure(self):
        with self.assertRaises(ValueError): h.canonical_id_bytes(np.array([1.,1.]))
    def test_order_hash_changes(self):
        a=h.canonical_id_bytes(np.array([1.,2.]))[1]; b=h.canonical_id_bytes(np.array([2.,1.]))[1]; self.assertNotEqual(a,b)
    def test_shape_failure(self):
        with self.assertRaises(ValueError): h.half_update([1],[1,2])
    def test_board_weighting(self):
        s={f:{"rmse_comparator":10.,"rmse_arm":9.,"rows":2,"changed_rows":1,
              "max_movement":1.,"A":1.,"B":1.} for f in h.FOLDS}
        self.assertAlmostEqual(h.board_record(s)["gain"],1)
    def test_bootstrap_deterministic(self):
        x=np.arange(8.); d=np.repeat(np.arange(4),2); self.assertEqual(h.day_bootstrap(x,x+1,x+.5,d,draws=30),h.day_bootstrap(x,x+1,x+.5,d,draws=30))
    def test_status_advance(self): self.assertEqual(h.status(.1,{f:.1 for f in h.FOLDS},{f:.01 for f in h.FOLDS}),"ADVANCE")
    def test_status_fail_board(self): self.assertEqual(h.status(0,{f:.1 for f in h.FOLDS},{f:.01 for f in h.FOLDS}),"FAIL")
    def test_status_fail_fold(self): self.assertEqual(h.status(.2,{"july_loo":.1,"december":0,"january":.1},{f:.01 for f in h.FOLDS}),"FAIL")
    def test_status_maybe(self): self.assertEqual(h.status(.09,{f:.1 for f in h.FOLDS},{f:.01 for f in h.FOLDS}),"MAYBE")
    def test_all_row_primary(self): self.assertEqual(json.loads((EXP/"source-manifest.json").read_text())["primary_population"],"all rows")
    def test_complement_report_only(self): self.assertIn("complement",json.loads((EXP/"source-manifest.json").read_text())["report_only_populations"][0])
    def test_no_label_registration(self): self.assertFalse((EXP/"measurement-state.json").exists())
    def test_phase_b_lock(self):
        with mock.patch.object(p,"ACCEPTANCE",Path("/no/such/file")):
            with self.assertRaises(PermissionError): p.require_acceptance()
    def test_state_transitions(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(p,"STATE",Path(td)/"state"):
            p.transition("unused","july_package_prepared"); p.transition("july_package_prepared","july_launch_started")
            p.transition("july_launch_started","july_landed"); p.transition("july_landed","july_verified")
            p.transition("july_verified","score_started"); p.transition("score_started","complete")
            self.assertEqual(p.state()["state"],"complete")
    def test_wrong_state_refused(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(p,"STATE",Path(td)/"state"):
            with self.assertRaises(RuntimeError): p.transition("july_verified","score_started")
    def test_private_offline_cpu(self):
        m=json.loads((EXP/"kaggle/kernel-metadata.json").read_text()); self.assertTrue(m["is_private"] and not m["enable_internet"] and not m["enable_gpu"] and not m["enable_tpu"])
    def test_output_schema(self): self.assertIn("A_over_B",h.OUTPUT_FIELDS)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): h.fold_stats([],[],[],[])
    def test_nonfinite_fold_rejected(self):
        with self.assertRaises(ValueError): h.fold_stats([math.nan],[1],[1],[1])
    def test_nonfinite_status_rejected(self):
        with self.assertRaises(ValueError): h.status(math.nan,{f:1 for f in h.FOLDS},{f:1 for f in h.FOLDS})
    def test_nonfinite_bootstrap_rejected(self):
        with self.assertRaises(ValueError): h.day_bootstrap([1],[1],[math.inf],[1])
    def test_zero_draws_rejected(self):
        with self.assertRaises(ValueError): h.day_bootstrap([1],[1],[1],[1],draws=0)
    def test_board_bootstrap(self):
        parts={f:(np.arange(4.),np.arange(4.)+1,np.arange(4.)+.5,np.arange(4.)) for f in h.FOLDS}
        self.assertEqual(h.board_bootstrap(parts,draws=20),h.board_bootstrap(parts,draws=20))
    def test_atomic_standard_json(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'x.json'; h.atomic_json(path,{"x":1}); self.assertEqual(json.loads(path.read_text()),{"x":1})
            with self.assertRaises(ValueError): h.atomic_json(path,{"x":math.nan})
    def test_source_trace_complete(self):
        m=json.loads((EXP/'source-manifest.json').read_text()); self.assertFalse(set(m['runtime_source_trace'])-set(m['sources']))
    def test_source_omission_detected(self):
        m=json.loads((EXP/'source-manifest.json').read_text()); omitted=dict(m['sources']); omitted.pop(next(iter(omitted)))
        self.assertTrue(set(m['runtime_source_trace'])-set(omitted))
    def test_source_mutation_detected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'x'; path.write_text('a'); before=h.sha256_file(path); path.write_text('b'); self.assertNotEqual(before,h.sha256_file(path))
    def test_native_dtype_not_coerced(self):
        with tempfile.TemporaryDirectory() as td:
            import pandas as pd
            path=Path(td)/'x.parquet'; pd.DataFrame({'MVT_ID_mvt':np.array([1,2],dtype='int64')}).to_parquet(path,index=False)
            self.assertEqual(p.native_id_record(path)[0]['id_dtype'],'int64')
    def test_systemexit_failure_recorded(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(p,'STATE',Path(td)/'state'), mock.patch.object(p,'INCOMPLETE',Path(td)/'bad'):
            with self.assertRaises(SystemExit): p.run_guarded('unused','july_launch_started',lambda: (_ for _ in ()).throw(SystemExit(2)))
            self.assertTrue(p.INCOMPLETE.exists())
    def test_serving_path_is_exact_q09(self):
        source=(EXP/'code/phase_b.py').read_text(); self.assertIn('Q.frames_for("chain")',source); self.assertIn('Q.serve',source); self.assertIn('noac',source)
    def test_acceptance_exact_keys(self): self.assertEqual(len(p.ACCEPTANCE_KEYS),9)
    def test_receipt_strict_keys(self): self.assertEqual(len(p.RECEIPT_KEYS),15)
    def test_output_receipt_strict_keys(self): self.assertEqual(len(p.OUTPUT_KEYS),8)
    def test_result_completeness_registered(self):
        source=(EXP/'code/phase_b.py').read_text()
        for field in ('cause','destination_value','path_feasibility','revival_condition','claim_boundary'): self.assertIn(field,source)
    def test_renderer_atomic(self): self.assertIn('os.replace', (EXP/'code/render.py').read_text())
    def test_renderer_checks_digest(self): self.assertIn('result_sha256', (EXP/'code/render.py').read_text())
    def test_deterministic_package_layout_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)/'src'; repo.mkdir(); subprocess.run(['git','init','-q'],cwd=repo,check=True)
            (repo/'x.txt').write_text('x'); subprocess.run(['git','add','.'],cwd=repo,check=True)
            subprocess.run(['git','-c','user.name=t','-c','user.email=t@t','commit','-qm','x'],cwd=repo,check=True)
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
            with mock.patch.object(p,'ROOT',repo):
                a=Path(td)/'a.tar'; b=Path(td)/'b.tar'; da=p.deterministic_package(commit,a); db=p.deterministic_package(commit,b)
            self.assertEqual(da,db)
            import tarfile
            with tarfile.open(a) as archive:
                self.assertIn('repo/GIT_COMMIT.txt',archive.getnames())

if __name__ == "__main__": unittest.main()
