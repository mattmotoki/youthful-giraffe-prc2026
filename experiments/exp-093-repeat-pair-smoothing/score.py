from __future__ import annotations
import hashlib, json, math, os, tempfile, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]
MANIFEST=EXP/'source-manifest.json'; RESULT=EXP/'result.json'
FOLD_MONTH={'july_loo':'2025-07','december':'2025-12','january':'2025-01'}
FOLDS=list(FOLD_MONTH); ARMS=['primary','rotation']; B=2000; SEED=93026
SHARES={'july_loo':192122/344841,'january':152719/344841}

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
 if pd.isna(x): return None
 y=unicodedata.normalize('NFKC',str(x)).strip().upper(); return y or None
def validate_ids(v,name):
 a=np.asarray(v,dtype='float64')
 if not np.isfinite(a).all() or not np.equal(a,np.floor(a)).all() or len(np.unique(a))!=len(a): raise ValueError(name+' invalid IDs')
 return a.astype('int64')
def schema_fp(path): return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode()).hexdigest()
def verify_sources():
 m=json.loads(MANIFEST.read_text())
 for rel,(digest,size,rows,kind) in m['sources'].items():
  p=ROOT/rel; pf=pq.ParquetFile(p)
  if p.stat().st_size!=size or sha(p)!=digest: raise ValueError(rel+' byte refusal')
  if pf.metadata.num_rows!=rows or schema_fp(p)!=m['schema_fingerprints'][kind]: raise ValueError(rel+' metadata refusal')
  names=set(pf.schema_arrow.names); proj=m['projections']['raw' if kind in ('training','ranking') else kind]
  if len(proj)!=len(set(proj)) or not set(proj)<=names: raise ValueError(rel+' projection refusal')
 return m
def raw_path(month):
 y,mo=month.split('-'); end=f'{y}-{int(mo)+1:02d}-01' if mo!='12' else '2026-01-01'
 return ROOT/f'data/raw/training_{month}-01_{end}.parquet'
def collision_bad(raw,origin,date):
 z=pd.DataFrame({'raw':raw,'origin':origin,'date':date}); z['key']=z.raw.map(norm); z=z[z.key.notna()]
 n=z.groupby(['origin','date','key'],dropna=False).raw.agg(lambda s:len({str(x) for x in s if not pd.isna(x)}))
 return set(n[n>1].index)
def reciprocal_pairs(time,key,min_gap=60,max_gap=86399,next_day=False):
 z=pd.DataFrame({'time':pd.to_datetime(time,utc=True),'key':key},index=time.index); z=z[z.time.notna()&z.key.notna()]
 edges=[]
 for _,g in z.groupby('key',sort=False):
  g=g.sort_values('time',kind='mergesort'); t=(g.time.astype('int64')//1_000_000_000).to_numpy(); ix=g.index.to_numpy()
  lo=np.searchsorted(t,t+min_gap,side='right'); hi=np.searchsorted(t,t+max_gap,side='right'); out=hi-lo
  in_lo=np.searchsorted(t,t-max_gap,side='left'); in_hi=np.searchsorted(t,t-min_gap,side='left'); inc=in_hi-in_lo
  for a in np.flatnonzero(out==1):
   b=lo[a]
   if b<len(t) and inc[b]==1:
    if not next_day or g.time.iloc[b].date()==(g.time.iloc[a]+pd.Timedelta(days=1)).date(): edges.append((int(ix[a]),int(ix[b])))
 return set(edges)
def disjoint(edges):
 degree=Counter(x for e in edges for x in e); return sorted(e for e in edges if degree[e[0]]==1 and degree[e[1]]==1)
def prepared(raw):
 d=raw.loc[raw.PHASE_mvt.eq('DEP')].reset_index(drop=True).copy(); d['time']=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True)
 d['date']=d.time.dt.strftime('%Y-%m-%d'); d['origin']=d.ADEP_mvt.map(norm); d['service']=d.FLIGHT_mvt.map(norm); d['radio']=d.CALLSIGN_flt.map(norm)
 zo=pd.DataFrame({'raw':d.ADEP_mvt,'date':d.date,'key':d.origin}); zo=zo[zo.key.notna()]; no=zo.groupby(['date','key'],dropna=False).raw.agg(lambda x:len({str(v) for v in x if not pd.isna(v)})); bado=set(no[no>1].index)
 d['origin_ok']=[o is not None and (dt,o) not in bado for dt,o in zip(d.date,d.origin)]
 bads=collision_bad(d.FLIGHT_mvt,d.origin,d.date); badr=collision_bad(d.CALLSIGN_flt,d.origin,d.date)
 d['service_ok']=[s is not None and (o,dt,s) not in bads for o,dt,s in zip(d.origin,d.date,d.service)]
 d['radio_ok']=[r is not None and (o,dt,r) not in badr for o,dt,r in zip(d.origin,d.date,d.radio)]
 return d,{'origin':len(bado),'service':len(bads),'radio':len(badr)}
def exact_pairs(d,service=None,radio=None,date_in_key=True,max_gap=86399,next_day=False):
 service=d.service if service is None else service; radio=d.radio if radio is None else radio
 complete=service.notna()&radio.notna()&d.origin.notna()
 if date_in_key:
  ks=pd.Series(list(zip(d.origin,d.date,service)),index=d.index).where(complete); kr=pd.Series(list(zip(d.origin,d.date,radio)),index=d.index).where(complete)
 else:
  ks=pd.Series(list(zip(d.origin,service)),index=d.index).where(complete); kr=pd.Series(list(zip(d.origin,radio)),index=d.index).where(complete)
 ps=reciprocal_pairs(d.time,ks,60,max_gap,next_day); pr=reciprocal_pairs(d.time,kr,60,max_gap,next_day)
 return disjoint(ps&pr)
def primary_pairs(d):
 valid=d.origin_ok&d.service_ok&d.radio_ok; return exact_pairs(d,d.service.where(valid),d.radio.where(valid))
def tie_digest(v): return hashlib.sha256(f'MVT_ID_mvt={int(v)}'.encode()).hexdigest()
def rotation_pairs(d):
 valid=d.origin_ok&d.service_ok&d.radio_ok; s=pd.Series(None,index=d.index,dtype=object); r=s.copy()
 z=d.loc[valid&d.origin.notna(),['origin','date','time','MVT_ID_mvt','service','radio']].copy(); z['tie']=z.MVT_ID_mvt.map(tie_digest)
 for _,g in z.groupby(['origin','date'],sort=False):
  ix=g.sort_values(['time','tie'],kind='mergesort').index; ss=d.loc[ix,'service'].tolist(); rr=d.loc[ix,'radio'].tolist()
  if len(ix): s.loc[ix]=ss[-1:]+ss[:-1]; r.loc[ix]=rr[-1:]+rr[:-1]
 return exact_pairs(d,s,r)
def previous_day_pairs(d):
 valid=d.origin_ok&d.service_ok&d.radio_ok; return exact_pairs(d,d.service.where(valid),d.radio.where(valid),False,172799,True)
def smooth(base,pairs):
 out=np.asarray(base,dtype='float64').copy(); mask=np.zeros(len(out),bool)
 for i,j in pairs: out[i]=out[j]=(out[i]+out[j])/2.0; mask[i]=mask[j]=True
 changed=mask&(out.view('uint64')!=np.asarray(base,dtype='float64').view('uint64'))
 if not np.array_equal(out[~mask].view('uint64'),np.asarray(base,dtype='float64')[~mask].view('uint64')): raise ValueError('unchanged-bit refusal')
 return out,mask,changed
def bootstrap(y,b,p,day,seed):
 q=pd.DataFrame({'day':day,'b':(y-b)**2,'p':(y-p)**2}).groupby('day',sort=True).agg(n=('b','size'),b=('b','sum'),p=('p','sum')).to_numpy(float)
 rng=np.random.default_rng(seed); vals=np.empty(B)
 for i in range(B):
  z=q[rng.integers(0,len(q),len(q))].sum(0); vals[i]=math.sqrt(z[1]/z[0])-math.sqrt(z[2]/z[0])
 return [float(x) for x in np.quantile(vals,[.025,.975],method='linear')]
def metrics(y,b,p,pairmask,changed,day,seed,nm,air,pairs):
 eb=(y-b)**2; ep=(y-p)**2
 def rm(m,v): return float(np.sqrt(np.mean(v[m]))) if np.any(m) else None
 slices={'le3600':y<=3600,'gt3600':y>3600,'no_nm':nm}; gain=eb-ep
 ap=pd.DataFrame({'airport':air,'gain':gain}).groupby('airport',dropna=False,sort=True).agg(rows=('gain','size'),sse_gain=('gain','sum')).reset_index()
 return {'pair_count':len(pairs),'paired_rows':int(pairmask.sum()),'changed_rows':int(changed.sum()),'base_rmse':rm(np.ones(len(y),bool),eb),'corrected_rmse':rm(np.ones(len(y),bool),ep),'sse_gain':float(gain.sum()),'rmse_gain':float(np.sqrt(eb.mean())-np.sqrt(ep.mean())),'paired_row_base_rmse':rm(pairmask,eb),'paired_row_corrected_rmse':rm(pairmask,ep),'interval95':bootstrap(y,b,p,day,seed),'airport_contributions':[{'airport':None if pd.isna(x.airport) else str(x.airport),'rows':int(x.rows),'sse_gain':float(x.sse_gain)} for x in ap.itertuples(index=False)],'slices':{k:{'rows':int(m.sum()),'base_rmse':rm(m,eb),'corrected_rmse':rm(m,ep)} for k,m in slices.items()}}
def join_panel(d,panel,fold):
 validate_ids(d.MVT_ID_mvt,fold+' raw'); validate_ids(panel.MVT_ID_mvt,fold+' panel')
 if len(d)!=len(panel) or not np.array_equal(d.MVT_ID_mvt.to_numpy(),panel.MVT_ID_mvt.to_numpy()): raise ValueError(fold+' identity/order refusal')
 if set(panel.fold.astype(str))!={fold}: raise ValueError(fold+' fold refusal')
 if not np.array_equal(d.date.astype(str).to_numpy(),panel.day.astype(str).to_numpy()): raise ValueError(fold+' day refusal')
def main():
 m=verify_sources(); out={'registration':{'source_manifest_sha256':sha(MANIFEST),'seed':SEED,'draws':B,'shares':SHARES},'folds':{}}; mses={a:{} for a in ARMS}; gains={a:{} for a in ARMS}
 for fi,fold in enumerate(FOLDS):
  raw=pd.read_parquet(raw_path(FOLD_MONTH[fold]),columns=m['projections']['raw']); d,coll=prepared(raw)
  panel=pd.read_parquet(ROOT/f'experiments/exp-050-nmblind/work/panel-v22-{fold}.parquet',columns=m['projections']['panel']); join_panel(d,panel,fold)
  y=panel.y_true.to_numpy('float64'); b=panel.p_v22.to_numpy('float64')
  if not np.isfinite(y).all() or not np.isfinite(b).all(): raise ValueError(fold+' nonfinite label/base')
  pp=primary_pairs(d); rp=rotation_pairs(d); prev=previous_day_pairs(d); fr={'rows':len(d),'collisions':coll,'previous_day_diagnostic':{'pair_count':len(prev),'row_count':2*len(prev)},'arms':{}}
  for ai,(arm,pairs) in enumerate([('primary',pp),('rotation',rp)]):
   pred,pm,changed=smooth(b,pairs); mt=metrics(y,b,pred,pm,changed,d.date.to_numpy(),SEED+fi+100*ai,panel.regime.astype(str).eq('nonm').to_numpy(),d.origin.to_numpy(),pairs); fr['arms'][arm]=mt; mses[arm][fold]=float(np.mean((y-pred)**2)); gains[arm][fold]=mt['rmse_gain']
  out['folds'][fold]=fr
 out['board_share']={a:float(math.sqrt(sum(SHARES[f]*np.mean((pd.read_parquet(ROOT/f'experiments/exp-050-nmblind/work/panel-v22-{f}.parquet',columns=['y_true','p_v22']).y_true-pd.read_parquet(ROOT/f'experiments/exp-050-nmblind/work/panel-v22-{f}.parquet',columns=['y_true','p_v22']).p_v22)**2) for f in SHARES))-math.sqrt(sum(SHARES[f]*mses[a][f] for f in SHARES))) for a in ARMS}
 raw=pd.read_parquet(ROOT/'data/raw/ranking.parquet',columns=m['projections']['raw']); rank,coll=prepared(raw); v22=pd.read_parquet(ROOT/'submissions/uploaded/youthful-giraffe_v22.parquet',columns=m['projections']['v22']); tmpl=pd.read_parquet(ROOT/'data/raw/submitting.parquet',columns=m['projections']['submitting'])
 validate_ids(rank.MVT_ID_mvt,'ranking'); validate_ids(v22.MVT_ID_mvt,'v22'); validate_ids(tmpl.MVT_ID_mvt,'template')
 if not np.array_equal(rank.MVT_ID_mvt,v22.MVT_ID_mvt) or not np.array_equal(tmpl.MVT_ID_mvt,v22.MVT_ID_mvt): raise ValueError('ranking/template/v22 identity-order refusal')
 vb=v22.TAXITIME_SEC_mvt.to_numpy('float64')
 if not np.isfinite(vb).all(): raise ValueError('v22 nonfinite prediction')
 pairs=primary_pairs(rank); pred,pm,changed=smooth(vb,pairs); out['ranking']={'rows':len(rank),'pair_count':len(pairs),'paired_rows':int(pm.sum()),'changed_rows':int(changed.sum()),'collisions':coll}
 gates={'nonnegative_all_folds':all(gains['primary'][f]>=0 for f in FOLDS),'clear_july_january':all(out['folds'][f]['arms']['primary']['interval95'][0]>0 for f in ['july_loo','january']),'beats_rotation_two_folds':sum(gains['primary'][f]>gains['rotation'][f] for f in FOLDS)>=2,'beats_rotation_board':out['board_share']['primary']>out['board_share']['rotation'],'board_gain_at_least_0_10':out['board_share']['primary']>=.10,'ranking_changed_at_least_100':int(changed.sum())>=100}; out['gates']=gates; out['decision']='ADVANCE' if all(gates.values()) else 'FAIL'
 if out['decision']=='ADVANCE':
  cand=tmpl.copy(); cand['TAXITIME_SEC_mvt']=pred; cp=ROOT/'submissions/candidates/exp-093-repeat-pair-smoothing.parquet'; atomic_parquet(cp,cand); reread=pd.read_parquet(cp)
  if not np.array_equal(reread.MVT_ID_mvt,tmpl.MVT_ID_mvt) or not np.isfinite(reread.TAXITIME_SEC_mvt).all() or not np.array_equal(reread.TAXITIME_SEC_mvt.to_numpy()[~pm].view('uint64'),vb[~pm].view('uint64')): raise ValueError('candidate refusal')
  out['candidate']={'path':str(cp.relative_to(ROOT)),'sha256':sha(cp),'rows':len(cand),'changed_rows':int(changed.sum()),'minimum':float(pred.min()),'maximum':float(pred.max())}
 atomic_json(RESULT,out)
if __name__=='__main__': main()
