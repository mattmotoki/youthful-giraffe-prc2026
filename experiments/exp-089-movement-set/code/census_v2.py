#!/usr/bin/env python3
"""One corrected label-free census. It never emits a movement ID."""
import json,pathlib,sys
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parents[3];EXP=ROOT/'experiments/exp-089-movement-set';sys.path.insert(0,str(EXP/'code'));import exp089_controller as ctl
PRE=json.loads((EXP/'census-prereg-v3.json').read_text());COLS=PRE['projection']
def airport(row):
 phase=row['PHASE_mvt'];adep=row['ADEP_mvt'];ades=row['ADES_mvt']
 if phase=='DEP' and adep:return adep
 if phase=='ARR' and ades:return ades
 raise ValueError('missing or conflicting direction and airport field')
def run():
 manifest=json.loads((EXP/'source-manifest.json').read_text());ctl.reject_forbidden(COLS);totals={};degrees=Counter();support=Counter();schemas={}
 for rel,digest in manifest['sources'].items():
  if not rel.startswith('data/raw/') or 'submitting' in rel:continue
  path=ROOT/rel
  if ctl.sha256_path(path)!=digest:raise ValueError('source mutation')
  pf=pq.ParquetFile(path);schema=pf.schema_arrow;actual={c:str(schema.field(c).type) for c in COLS}
  if actual!=PRE['native_types']:raise TypeError('native schema mismatch')
  rows=pf.read(columns=COLS).to_pylist()
  for row in rows:row['_airport']=airport(row)
  totals[rel]={'rows':len(rows),'departures':sum(x['PHASE_mvt']=='DEP' for x in rows),'arrivals':sum(x['PHASE_mvt']=='ARR' for x in rows)};schemas[rel]=actual;queries=[x for x in rows if x['PHASE_mvt']=='DEP']
  for name,definition in PRE['relations'].items():
   field=definition['field'];lo,hi=definition['window_seconds'];groups={}
   for x in rows:
    if x[field] is not None:groups.setdefault((x['_airport'],x[field]),[]).append(x['MVT_TIME_UTC_mvt'].timestamp())
   groups={k:np.sort(np.asarray(v)) for k,v in groups.items()}
   for query in queries:
    values=groups.get((query['_airport'],query[field])) if query[field] is not None else None
    if values is None:n=0
    else:
     stamp=query['MVT_TIME_UTC_mvt'].timestamp();n=int(np.searchsorted(values,stamp+hi,side='right')-np.searchsorted(values,stamp+lo,side='left')-1)
    degrees[(name,ctl.density_stratum(n))]+=1;support[name]+=n>0
 if sum(x['rows'] for x in totals.values())!=PRE['expected_total_rows']:raise ValueError('total count mismatch')
 out={'schema':3,'columns':COLS,'direction_rule':PRE['direction_rule'],'totals':totals,'relation_support':dict(support),'degree_strata':{'|'.join(k):v for k,v in sorted(degrees.items())},'schemas':schemas,'label_values_read':False,'prediction_values_read':False,'private_ids_saved':False};ctl.atomic_write(EXP/'analysis/label-free-census-v3.json',out);return out
if __name__=='__main__':run()
