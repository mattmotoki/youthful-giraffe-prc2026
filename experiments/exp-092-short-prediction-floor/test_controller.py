from __future__ import annotations
import importlib.util, json, tempfile
from pathlib import Path
P=Path(__file__).with_name('controller.py'); sp=importlib.util.spec_from_file_location('controller092',P); c=importlib.util.module_from_spec(sp); sp.loader.exec_module(c)
def test_atomic_standard_json_receipt():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'r.json'; c.atomic_json(p,{'exit_code':0}); assert json.loads(p.read_text())=={'exit_code':0}
def test_exactly_once_lock_refuses():
 with tempfile.TemporaryDirectory() as td:
  d=Path(td); lock=d/'lock'; started=d/'start'; exitp=d/'exit'; c.claim_launch(lock,started,exitp)
  try: c.claim_launch(lock,started,exitp); assert False
  except RuntimeError as e: assert 'already claimed' in str(e)
if __name__=='__main__':
 tests=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
 for t in tests:t()
 print(len(tests),'tests passed')
