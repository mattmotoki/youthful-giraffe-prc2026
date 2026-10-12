#!/usr/bin/env python3
import json,pathlib,sys
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parents[3];EXP=ROOT/'experiments/exp-089-movement-set';sys.path.insert(0,str(EXP/'code'));import exp089_controller as ctl
COLS=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt']
def run():
 m=json.loads((EXP/'source-manifest.json').read_text());assert set(COLS)<=set(m['allowed_columns']);ctl.reject_forbidden(COLS);totals={};degrees=Counter();support=Counter();schemas={}
 for rel,digest in m['sources'].items():
  if not rel.startswith('data/raw/') or 'submitting' in rel:continue
  path=ROOT/rel
  if ctl.sha256_path(path)!=digest:raise ValueError('source mutation')
  pf=pq.ParquetFile(path);s=pf.schema_arrow
  if any(c not in s.names for c in COLS):raise ValueError('schema omission')
  schemas[rel]={c:str(s.field(c).type) for c in COLS};rows=pf.read(columns=COLS).to_pylist();totals[rel]={'rows':len(rows),'departures':sum(x['PHASE_mvt']=='DEP' for x in rows),'arrivals':sum(x['PHASE_mvt']=='ARR' for x in rows)}
  queries=[x for x in rows if x['PHASE_mvt']=='DEP']
  for name,field,lo,hi in [('runway','RUNWAY_mvt',-1800,600),('stand','STAND_mvt',-7200,1800),('operator','AIRCRAFT_OPERATOR_flt',-1800,600)]:
   groups={}
   for x in rows:
    if x[field] is not None:groups.setdefault((x['ADEP_mvt'],x[field]),[]).append(x['MVT_TIME_UTC_mvt'].timestamp())
   groups={k:np.sort(np.asarray(v)) for k,v in groups.items()}
   for q in queries:
    values=groups.get((q['ADEP_mvt'],q[field])) if q[field] is not None else None
    if values is None:n=0
    else:
     stamp=q['MVT_TIME_UTC_mvt'].timestamp();n=int(np.searchsorted(values,stamp+hi,side='right')-np.searchsorted(values,stamp+lo,side='left')-1)
    degrees[(name,ctl.density_stratum(n))]+=1;support[name]+=n>0
 out={'schema':1,'columns':COLS,'totals':totals,'relation_support':dict(support),'degree_strata':{'|'.join(k):v for k,v in sorted(degrees.items())},'schemas':schemas,'label_values_read':False,'prediction_values_read':False};ctl.atomic_write(EXP/'analysis/label-free-census.json',out);return out
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
