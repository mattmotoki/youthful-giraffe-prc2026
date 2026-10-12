from __future__ import annotations
import importlib.util, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).with_name('score.py'); sp=importlib.util.spec_from_file_location('score092',P); s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
def frame(n,key=('AAA','01'),values=None):
 return pd.DataFrame({'airport':[key[0]]*n,'runway':[key[1]]*n,'y':np.arange(n,dtype=float) if values is None else values})
def test_hierarchy_and_support_edges():
 x=pd.concat([frame(499),frame(500,('AAA','02'))]); t=s.floors(x,['airport','runway'],500)
 assert not t[('AAA','01')]['eligible'] and t[('AAA','01')]['floor']==60
 assert t[('AAA','02')]['eligible']; a=s.floors(pd.concat([frame(4999),frame(1)]),['airport'],5000); assert a[('AAA',)]['eligible']
def test_quantile_linear_and_clip_edges():
 assert s.q01_linear([0,100])==1
 assert s.floors(frame(500,values=np.full(500,1.)),['airport','runway'],500)[('AAA','01')]['floor']==60
 assert s.floors(frame(500,values=np.full(500,999.)),['airport','runway'],500)[('AAA','01')]['floor']==240
def test_application_edges_and_preservation():
 t={('AAA','01'):{'support':500,'eligible':True,'floor':100.}}
 b=np.array([50.,100.,239.,240.,300.,np.nan]); p,m,_=s.apply_arm(b,['AAA']*6,['01']*6,t,'primary')
 assert np.array_equal(p[:5],[100,100,239,240,300]) and m.tolist()==[True,False,False,False,False,False] and np.isnan(p[5])
def test_missing_runway_and_rotation_control():
 assert s.norm(None)==s.MISS and s.norm(' ０1 ' )=='01'
 t={('A',s.MISS):{'support':500,'eligible':True,'floor':70.},('A','02'):{'support':600,'eligible':True,'floor':90.}}
 r=s.rotate(t); assert r[('A','02')]['floor']==70 and r[('A',s.MISS)]['floor']==90
def test_identity_order_and_fold_refusals():
 p=pd.DataFrame({'MVT_ID_mvt':[1.,2.],'fold':['x','x']}); r=pd.DataFrame({'MVT_ID_mvt':[1.,2.],'PHASE_mvt':['DEP','DEP']})
 assert len(s.validate_join(p,r,'x'))==2
 for bad in [pd.DataFrame({'MVT_ID_mvt':[2.,1.],'PHASE_mvt':['DEP','DEP']}),pd.DataFrame({'MVT_ID_mvt':[1.,1.],'PHASE_mvt':['DEP','DEP']})]:
  try: s.validate_join(p,bad,'x'); assert False
  except ValueError: pass
 p.fold=['x','y']
 try: s.validate_join(p,r,'x'); assert False
 except ValueError: pass
def test_fold_exclusion_exact():
 assert '2025-07' not in s.FOLDS['july_loo'][0] and '2025-12' not in s.FOLDS['december'][0] and '2025-01' not in s.FOLDS['january'][0]
def test_controls_and_bootstrap():
 b=np.array([70.,80.]); p,m,_=s.apply_arm(b,['A','A'],['R','R'],{},'global60'); assert np.array_equal(b,p) and not m.any()
 old=s.B; s.B=20
 try: assert s.bootstrap(np.array([100.,100.]),b,np.array([80.,90.]),np.array(['d1','d2']),3)==s.bootstrap(np.array([100.,100.]),b,np.array([80.,90.]),np.array(['d1','d2']),3)
 finally: s.B=old
def test_standard_json_and_atomic_writers():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'x.json'; s.atomic_json(p,{'x':1.}); assert json.loads(p.read_text())=={'x':1}
  try: s.atomic_json(p,{'x':float('nan')}); assert False
  except ValueError: pass
  q=Path(td)/'x.parquet'; d=pd.DataFrame({'a':[1]}); s.atomic_parquet(q,d); assert pd.read_parquet(q).equals(d)
if __name__=='__main__':
 tests=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
 for t in tests:t()
 print(len(tests),'tests passed')
