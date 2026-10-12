"""Rebuild target's held no-plan values from saved model predictions; never fit."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'experiments/exp-167-target/code'))
import common as C
import noplan_form as N
import noplan_form_v2 as V

parts=[]
raw_paths = sorted((ROOT/'data/raw').glob('training_2025-*.parquet'))
for path in raw_paths:
    t=pd.read_parquet(path,columns=N.P.COLS)
    # All LIRF departures, linked arrivals, and LIRF arrivals for the C39 table.
    t=t[((t.PHASE_mvt=='DEP') & (t.ADEP_mvt=='LIRF')) |
        ((t.PHASE_mvt=='ARR') & ((t.ADEP_flt=='LIRF') | (t.ADES_mvt=='LIRF')))]
    parts.append(t)
t=pd.concat(parts,ignore_index=True)
d=N.classes(N.noplan_rows(t))
d=V.add_v2(d,V.arrival_share(t))
src=d[d.month.isin(N.R1.TEN)]
tb=N.tables(src)
rt=V.rule_tables(src)
r=C.served_forms(C.fold_rows()).set_index(C.ID)
f=d[d[C.ID].isin(r.index)].reset_index(drop=True)
base=r.F.reindex(f[C.ID]).to_numpy(float)
a=pd.read_parquet(ROOT/'data/exp167/inbound-sign-001.parquet')
assert a[C.ID].is_unique and a.month.isin([1,7]).all()
p=a['p_sign+lead'].to_numpy(float)
mix=pd.Series(np.maximum(60,p*a.ms.to_numpy(float)+(1-p)*a.m_real.to_numpy(float)),index=a[C.ID].to_numpy('int64'))
v1,part=N.form(f,base,mix,tb,False)
v2,part2=V.form(f,base,mix,tb,rt,False)
bet,_=N.form(f,base,mix,tb,True)
out=pd.DataFrame({C.ID:f[C.ID], 'np_base':base,'np_v1':v1,'np_v2':v2,'np_bet':bet,
                  'part_v1':part,'part_v2':part2,'bet':(f.later & f.long).to_numpy(bool)})
out.to_parquet(EXP/'inputs/noplan-fold.parquet',index=False)
rec={'method':'Saved p_sign+lead and m_real, same ten-month tables; no fitting call.',
     'rows':len(out),'ten_months':list(N.R1.TEN),'training_tables':{k:v for k,v in tb.items() if k!='c32'},
     'raw_inputs':{str(p.relative_to(ROOT)):C.sha(p) for p in raw_paths},
     'saved_predictions_sha256':C.sha(ROOT/'data/exp167/inbound-sign-001.parquet'),
     'source_hashes':{p.name:C.sha(p) for p in (ROOT/'experiments/exp-167-target/code').glob('*.py')},
     'output_sha256':C.sha(EXP/'inputs/noplan-fold.parquet')}
(EXP/'results/noplan-replay-inputs.json').write_text(json.dumps(rec,indent=2)+'\n')
print('No-plan saved-vector replay prepared:',len(out),'rows',flush=True)
