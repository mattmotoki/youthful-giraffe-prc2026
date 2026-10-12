from __future__ import annotations
import importlib.util, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).with_name('score.py'); sp=importlib.util.spec_from_file_location('score093',P); s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
def frame(times,service,radio,origin=None,ids=None):
 n=len(times); origin=origin or ['AAA']*n; ids=ids or list(range(1,n+1))
 raw=pd.DataFrame({'MVT_ID_mvt':ids,'PHASE_mvt':['DEP']*n,'MVT_TIME_UTC_mvt':pd.to_datetime(times,utc=True),'FLIGHT_mvt':service,'CALLSIGN_flt':radio,'ADEP_mvt':origin})
 return s.prepared(raw)[0]
def test_exact_gap_edges_and_midnight():
 d=frame(['2025-01-01T23:58:59Z','2025-01-02T00:00:00Z'],['A','A'],['R','R']); assert s.primary_pairs(d)==[]
 d=frame(['2025-01-01T00:00:00Z','2025-01-01T00:01:01Z'],['A','A'],['R','R']); assert s.primary_pairs(d)==[(0,1)]
 d=frame(['2025-01-01T00:00:00Z','2025-01-01T23:59:59Z'],['A','A'],['R','R']); assert s.primary_pairs(d)==[(0,1)]
def test_missing_and_unicode_collision():
 d=frame(['2025-01-01T00:00Z','2025-01-01T00:02Z'],['Ａ',None],['R','R']); assert s.primary_pairs(d)==[]
 d,_=s.prepared(pd.DataFrame({'MVT_ID_mvt':[1,2],'PHASE_mvt':['DEP']*2,'MVT_TIME_UTC_mvt':pd.to_datetime(['2025-01-01T00:00Z','2025-01-01T00:02Z']),'FLIGHT_mvt':['Ａ','A'],'CALLSIGN_flt':['R','R'],'ADEP_mvt':['AAA','AAA']})); assert s.primary_pairs(d)==[]
def test_ambiguous_overlap_and_different_lane_partner():
 d=frame(['2025-01-01T00:00Z','2025-01-01T00:02Z','2025-01-01T00:04Z'],['A']*3,['R']*3); assert s.primary_pairs(d)==[]
 d=frame(['2025-01-01T00:00Z','2025-01-01T00:02Z','2025-01-01T00:04Z','2025-01-01T00:06Z'],['A','A','B','B'],['R','S','R','S']); assert s.primary_pairs(d)==[]
 edges={(0,1),(1,2)}; assert s.disjoint(edges)==[]
def test_origin_date_and_previous_day():
 d=frame(['2025-01-01T23:58Z','2025-01-02T00:02Z'],['A','A'],['R','R']); assert s.primary_pairs(d)==[]; assert s.previous_day_pairs(d)==[(0,1)]
 d=frame(['2025-01-01T00:00Z','2025-01-01T00:02Z'],['A','A'],['R','R'],['AAA','BBB']); assert s.primary_pairs(d)==[]
def test_rotation_and_tie_digest_determinism():
 d=frame(['2025-01-01T00:00Z','2025-01-01T00:02Z','2025-01-01T00:04Z','2025-01-01T00:06Z'],['A','B','A','C'],['R','S','R','T']); assert s.rotation_pairs(d)==[(1,3)]
 assert s.tie_digest(1)==s.tie_digest(1) and s.tie_digest(1)!=s.tie_digest(2)
def test_unchanged_bits_and_averaging():
 b=np.array([1.0,3.0,7.0]); p,pm,ch=s.smooth(b,[(0,1)]); assert np.array_equal(p,np.array([2.,2.,7.])); assert pm.tolist()==[True,True,False] and ch.tolist()==[True,True,False]; assert p[2:].view('uint64')[0]==b[2:].view('uint64')[0]
def test_row_order_and_id_refusals():
 d=frame(['2025-01-01T00:00Z','2025-01-01T00:02Z'],['A','A'],['R','R'],ids=[1,2]); panel=pd.DataFrame({'MVT_ID_mvt':[2,1],'fold':['x','x'],'day':['2025-01-01']*2})
 try:s.join_panel(d,panel,'x');assert False
 except ValueError as e:assert 'identity/order' in str(e)
 panel.MVT_ID_mvt=[1,1]
 try:s.join_panel(d,panel,'x');assert False
 except ValueError as e:assert 'invalid IDs' in str(e)
def test_bootstrap_repeats_and_standard_json_atomic():
 old=s.B; s.B=20
 try:
  y=np.array([3.,4.]); b=np.array([1.,1.]); p=np.array([2.,2.]); day=np.array(['d1','d2']); assert s.bootstrap(y,b,p,day,9)==s.bootstrap(y,b,p,day,9)
 finally:s.B=old
 with tempfile.TemporaryDirectory() as td:
  q=Path(td)/'x.json'; s.atomic_json(q,{'x':1.}); assert json.loads(q.read_text())=={'x':1}
  try:s.atomic_json(q,{'x':float('nan')});assert False
  except ValueError:pass
def test_atomic_parquet():
 with tempfile.TemporaryDirectory() as td:
  q=Path(td)/'x.parquet'; d=pd.DataFrame({'a':[1]}); s.atomic_parquet(q,d); assert pd.read_parquet(q).equals(d)
if __name__=='__main__':
 tests=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
 for t in tests:t()
 print(len(tests),'tests passed')
