from __future__ import annotations
import hashlib, json, math, os, tempfile, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]
MANIFEST=EXP/'source-manifest.json'; RESULT=EXP/'result.json'
FOLDS={'july_loo':(['2025-%02d'%m for m in range(1,13) if m!=7],'2025-07'),
       'december':(['2025-%02d'%m for m in range(1,12)],'2025-12'),
       'january':(['2025-%02d'%m for m in range(2,13)],'2025-01')}
FOLD_ORDER=['july_loo','december','january']; ARMS=['primary','global60','airport','runway_permutation']
SEED=92026; MISS='__MISSING__'; B=2000; SHARES={'july_loo':192122/344841,'january':152719/344841}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def atomic_json(path,obj):
 data=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode(); fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path); d=os.open(path.parent,os.O_RDONLY); os.fsync(d); os.close(d)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def atomic_parquet(path,frame):
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.parquet',dir=path.parent); os.close(fd)
 try:
  frame.to_parquet(tmp,index=False)
  with open(tmp,'rb') as f: os.fsync(f.fileno())
  os.replace(tmp,path); d=os.open(path.parent,os.O_RDONLY); os.fsync(d); os.close(d)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def norm(x):
 if pd.isna(x): return MISS
 y=unicodedata.normalize('NFKC',str(x)).strip().upper(); return y or MISS
def q01_linear(a): return float(np.quantile(np.asarray(a,dtype='float64'),.01,method='linear'))
def floors(frame,keys,min_support):
 out={}
 for key,g in frame.groupby(keys,dropna=False,sort=True):
  if not isinstance(key,tuple): key=(key,)
  n=len(g); out[key]={'support':int(n),'eligible':bool(n>=min_support),'floor':float(np.clip(q01_linear(g.y),60,240)) if n>=min_support else 60.0}
 return out
def rotate(table):
 out={}; aps=sorted({k[0] for k in table})
 for ap in aps:
  rs=sorted(k[1] for k in table if k[0]==ap)
  for i,r in enumerate(rs): out[(ap,r)]=dict(table[(ap,rs[(i+1)%len(rs)])])
 return out
def apply_arm(base,air,run,table,kind):
 base=np.asarray(base,dtype='float64'); out=base.copy(); eligible=np.zeros(len(base),bool); fl=np.full(len(base),60.0)
 if kind=='global60': return out,np.zeros(len(base),bool),fl
 for i,(a,r) in enumerate(zip(air,run)):
  rec=table.get((a,) if kind=='airport' else (a,r))
  if rec is not None and rec['eligible']: eligible[i]=True; fl[i]=rec['floor']
 mask=np.isfinite(base)&(base<240)&eligible&(base<fl); out[mask]=fl[mask]
 return out,mask,fl
def validate_ids(ids,name):
 a=np.asarray(ids,dtype='float64')
 if not np.isfinite(a).all() or not np.equal(a,np.floor(a)).all() or len(np.unique(a))!=len(a): raise ValueError(name+' invalid IDs')
def validate_join(panel,raw,fold):
 dep=raw.loc[raw.PHASE_mvt.eq('DEP')].reset_index(drop=True)
 validate_ids(panel.MVT_ID_mvt,fold+' panel'); validate_ids(dep.MVT_ID_mvt,fold+' raw')
 if len(dep)!=len(panel) or not np.array_equal(dep.MVT_ID_mvt.to_numpy(),panel.MVT_ID_mvt.to_numpy()): raise ValueError(fold+' identity/order refusal')
 if set(panel.fold.astype(str))!={fold}: raise ValueError(fold+' fold refusal')
 return dep
def bootstrap(y,base,pred,day,seed):
 d=pd.DataFrame({'day':day,'b':(y-base)**2,'p':(y-pred)**2}).groupby('day',sort=True).agg(n=('b','size'),b=('b','sum'),p=('p','sum'))
 a=d[['n','b','p']].to_numpy(float); rng=np.random.default_rng(seed); vals=np.empty(B)
 for i in range(B):
  z=a[rng.integers(0,len(a),len(a))].sum(axis=0); vals[i]=math.sqrt(z[1]/z[0])-math.sqrt(z[2]/z[0])
 return [float(x) for x in np.quantile(vals,[.025,.975],method='linear')]
def metrics(y,base,pred,mask,day,seed,nm):
 eb=(y-base)**2; ep=(y-pred)**2
 def rm(m,v): return float(np.sqrt(np.mean(v[m]))) if m.any() else None
 slices={'le3600':y<=3600,'gt3600':y>3600,'no_nm':nm}
 return {'rows':int(len(y)),'corrected_rows':int(mask.sum()),'base_rmse':rm(np.ones(len(y),bool),eb),'corrected_rmse':rm(np.ones(len(y),bool),ep),
  'sse_gain':float(eb.sum()-ep.sum()),'rmse_gain':float(np.sqrt(eb.mean())-np.sqrt(ep.mean())),
  'corrected_row_base_rmse':rm(mask,eb),'corrected_row_rmse':rm(mask,ep),'interval95':bootstrap(y,base,pred,day,seed),
  'slices':{k:{'rows':int(m.sum()),'base_rmse':rm(m,eb),'corrected_rmse':rm(m,ep)} for k,m in slices.items()}}
def table_json(t): return [{'key':list(k),'support':v['support'],'eligible':v['eligible'],'floor':v['floor']} for k,v in sorted(t.items())]
def contributions(y,b,p,air,run):
 d=pd.DataFrame({'airport':air,'runway':run,'gain':(y-b)**2-(y-p)**2})
 return [{'airport':a,'runway':r,'rows':int(len(g)),'sse_gain':float(g.gain.sum())} for (a,r),g in d.groupby(['airport','runway'],sort=True)]
def verify_sources():
 m=json.loads(MANIFEST.read_text())
 for rel,(digest,size,rows,skind) in m['sources'].items():
  p=ROOT/rel
  if p.stat().st_size!=size or sha(p)!=digest: raise ValueError(rel+' byte refusal')
  pf=pq.ParquetFile(p); fp=hashlib.sha256(str(pf.schema_arrow).encode()).hexdigest()
  if pf.metadata.num_rows!=rows or fp!=m['schema_fingerprints'][skind]: raise ValueError(rel+' metadata refusal')
 return m
def raw_path(month):
 y,mo=month.split('-'); import calendar
 end=f'{y}-{int(mo)+1:02d}-01' if mo!='12' else '2026-01-01'
 return ROOT/f'data/raw/training_{month}-01_{end}.parquet'
def load_raw(month): return pd.read_parquet(raw_path(month),columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','RUNWAY_mvt','TAXITIME_SEC_mvt'])
def dep_labels(months,cache):
 z=[]
 for m in months:
  d=cache[m]; d=d.loc[d.PHASE_mvt.eq('DEP'),['ADEP_mvt','RUNWAY_mvt','TAXITIME_SEC_mvt']].copy(); d=d.loc[np.isfinite(d.TAXITIME_SEC_mvt)]
  d['airport']=d.ADEP_mvt.map(norm); d['runway']=d.RUNWAY_mvt.map(norm); d['y']=d.TAXITIME_SEC_mvt.astype(float); z.append(d[['airport','runway','y']])
 return pd.concat(z,ignore_index=True)
def main():
 manifest=verify_sources(); cache={m:load_raw(m) for m in ['2025-%02d'%x for x in range(1,13)]}; result={'registration':{'source_manifest_sha256':sha(MANIFEST),'seed':SEED,'draws':B},'folds':{}}
 mses={a:{} for a in ARMS}; gains={a:{} for a in ARMS}
 for fi,fold in enumerate(FOLD_ORDER):
  months,valid=FOLDS[fold]; train=dep_labels(months,cache); pri=floors(train,['airport','runway'],500); apt=floors(train,['airport'],5000); perm=rotate(pri)
  panel=pd.read_parquet(ROOT/f'experiments/exp-050-nmblind/work/panel-v22-{fold}.parquet')
  dep=validate_join(panel,cache[valid],fold); y=panel.y_true.to_numpy(float); base=panel.p_v22.to_numpy(float)
  if not np.isfinite(y).all() or not np.isfinite(base).all(): raise ValueError(fold+' nonfinite label/base')
  air=dep.ADEP_mvt.map(norm).to_numpy(); run=dep.RUNWAY_mvt.map(norm).to_numpy(); day=panel.day.astype(str).to_numpy(); nm=panel.regime.astype(str).eq('nonm').to_numpy()
  fr={'train_months':months,'validation_month':valid,'support_tables':{'primary':table_json(pri),'airport':table_json(apt),'runway_permutation':table_json(perm)},'arms':{}}
  for ai,arm in enumerate(ARMS):
   tab={'primary':pri,'airport':apt,'runway_permutation':perm}.get(arm,{})
   pred,mask,_=apply_arm(base,air,run,tab,arm); mt=metrics(y,base,pred,mask,day,SEED+fi+100*ai,nm)
   mt['contributions']=contributions(y,base,pred,air,run); fr['arms'][arm]=mt; mses[arm][fold]=float(np.mean((y-pred)**2)); gains[arm][fold]=mt['rmse_gain']
  result['folds'][fold]=fr
 result['board_share']={a:float(math.sqrt(sum(SHARES[f]*mses['global60'][f] for f in SHARES))-math.sqrt(sum(SHARES[f]*mses[a][f] for f in SHARES))) for a in ARMS}
 # Candidate-side gate is measured in memory before any write.
 train_all=dep_labels(['2025-%02d'%x for x in range(1,13)],cache); allpri=floors(train_all,['airport','runway'],500)
 rank=pd.read_parquet(ROOT/'data/raw/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','RUNWAY_mvt']); rank=rank.loc[rank.PHASE_mvt.eq('DEP')].reset_index(drop=True)
 v22=pd.read_parquet(ROOT/'submissions/uploaded/youthful-giraffe_v22.parquet'); tmpl=pd.read_parquet(ROOT/'data/raw/submitting.parquet')
 validate_ids(rank.MVT_ID_mvt,'ranking'); validate_ids(v22.MVT_ID_mvt,'v22'); validate_ids(tmpl.MVT_ID_mvt,'template')
 if not np.array_equal(rank.MVT_ID_mvt,v22.MVT_ID_mvt) or not np.array_equal(tmpl.MVT_ID_mvt,v22.MVT_ID_mvt): raise ValueError('ranking/template/v22 identity-order refusal')
 vb=v22.TAXITIME_SEC_mvt.to_numpy(float); rp,rmask,_=apply_arm(vb,rank.ADEP_mvt.map(norm),rank.RUNWAY_mvt.map(norm),allpri,'primary')
 gates={'nonnegative_all_folds':all(gains['primary'][f]>=0 for f in FOLD_ORDER),
  'clear_july_january':all(result['folds'][f]['arms']['primary']['interval95'][0]>0 for f in ['july_loo','january']),
  'board_gain_at_least_0_10':result['board_share']['primary']>=.10,
  'beats_permutation_two_folds':sum(gains['primary'][f]>gains['runway_permutation'][f] for f in FOLD_ORDER)>=2,
  'beats_permutation_board':result['board_share']['primary']>result['board_share']['runway_permutation'],
  'ranking_corrections_at_least_100':int(rmask.sum())>=100}
 result['ranking']={'rows':len(vb),'corrected_rows':int(rmask.sum()),'all_2025_support_table':table_json(allpri)}; result['gates']=gates; result['decision']='ADVANCE' if all(gates.values()) else 'FAIL'
 if result['decision']=='ADVANCE':
  out=tmpl.copy(); out['TAXITIME_SEC_mvt']=rp
  cp=ROOT/'submissions/candidates/exp-092-short-prediction-floor.parquet'; atomic_parquet(cp,out)
  reread=pd.read_parquet(cp)
  if not np.array_equal(reread.MVT_ID_mvt,tmpl.MVT_ID_mvt) or not np.isfinite(reread.TAXITIME_SEC_mvt).all() or not np.array_equal(reread.TAXITIME_SEC_mvt.to_numpy()[~rmask],vb[~rmask]): raise ValueError('candidate verification refusal')
  result['candidate']={'path':str(cp.relative_to(ROOT)),'sha256':sha(cp),'rows':len(out),'changed_rows':int(rmask.sum()),'minimum':float(rp.min()),'maximum':float(rp.max())}
 atomic_json(RESULT,result)
if __name__=='__main__': main()
