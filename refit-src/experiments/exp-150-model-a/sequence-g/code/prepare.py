"""Lane G: input-only mixed runway events and separately owned loss labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP/'code'))
import r4_admission_runtime as R

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
TIMES = [TIME,'SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt',
         'ARVT_1_flt','AOBT_3_flt','ARVT_3_flt']
RAW_CATS = ['STAND_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_TYPE_flt','WK_TBL_CAT_flt',
            'AIRCRAFT_OPERATOR_flt','MARKET_SEGMENT_flt','FLIGHT_RULE_mvt','FLIGHT_TYPE_flt']
COLS = [ID,'PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt',*TIMES,*RAW_CATS]
CATS = ['airport','runway','PHASE_mvt',*RAW_CATS]
HALF, SECONDS = 8, 1800

def sha(path): return R.sha(path)

def events(raw):
    # Always select this whitelist before any transform, including poison tests.
    f = raw[COLS].copy()
    if not f.PHASE_mvt.isin(['DEP','ARR']).all(): raise ValueError('Unknown phase')
    ids = pd.to_numeric(f[ID], errors='raise').to_numpy(float)
    if not np.isfinite(ids).all() or not np.equal(ids,np.round(ids)).all(): raise ValueError('Noninteger key')
    f[ID] = ids.astype('int64')
    for c in TIMES: f[c] = pd.to_datetime(f[c], utc=True).astype('datetime64[ns, UTC]')
    if f[TIME].isna().any(): raise ValueError('Missing movement time')
    dep = f.PHASE_mvt.eq('DEP')
    f['airport'] = f.ADEP_mvt.where(dep, f.ADES_mvt)
    if f.airport.isna().any() or f.airport.astype(str).str.strip().eq('').any(): raise ValueError('Missing event airport')
    f['airport'] = f.airport.astype(str).str.strip().str.upper()
    f['runway'] = f.RUNWAY_mvt.astype('string').str.strip().str.upper()
    f['missing_runway'] = f.runway.isna() | f.runway.eq('').fillna(False)
    f['runway'] = f.runway.mask(f.missing_runway,'<MISSING>')
    f['day'] = f[TIME].dt.strftime('%Y-%m-%d')
    keys = f.airport+'|'+f.day+'|'+f.runway
    # Unknown runway rows do not form a combined airport context.
    keys = keys.where(~f.missing_runway,keys+'|singleton|'+f[ID].astype(str))
    f['group_key'] = keys
    f = f.sort_values(['group_key',TIME,ID],kind='stable').reset_index(drop=True)
    f['group'] = pd.factorize(f.group_key,sort=False)[0].astype('int32')
    return f

def bounds(f):
    g=f.group.to_numpy();st=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1]
    en=np.r_[st[1:],len(f)]
    return st,en,np.repeat(st,en-st),np.repeat(en,en-st)

def numerics(f):
    z=pd.DataFrame(index=f.index)
    for c in TIMES[1:]: z[c+'_minus_m']=(f[c]-f[TIME]).dt.total_seconds().clip(-10800,10800)
    for a,b in [('AOBT_3_flt','EOBT_1_flt'),('AOBT_3_flt','LOBT_flt'),
                ('ARVT_3_flt','ARVT_1_flt'),('ARVT_1_flt','EOBT_1_flt'),('SCHED_TIME_UTC_mvt','LOBT_flt')]:
        z[a+'_minus_'+b]=(f[a]-f[b]).dt.total_seconds().clip(-10800,10800)
    h=f[TIME].dt.hour+f[TIME].dt.minute/60+f[TIME].dt.second/3600
    for name,val,period in [('hour',h,24),('weekday',f[TIME].dt.dayofweek,7),('year_day',f[TIME].dt.dayofyear,365)]:
        z[name+'_sin']=np.sin(2*np.pi*val/period);z[name+'_cos']=np.cos(2*np.pi*val/period)
    t=f[TIME].astype('int64').to_numpy()/1e9
    st,en,start,end=bounds(f);ix=np.arange(len(f))
    z['gap_previous']=np.where(ix>start,t-t[np.maximum(ix-1,0)],np.nan)
    z['gap_next']=np.where(ix<end-1,t[np.minimum(ix+1,len(f)-1)]-t,np.nan)
    z['gap_previous']=z.gap_previous.clip(0,10800);z['gap_next']=z.gap_next.clip(0,10800)
    phase=f.PHASE_mvt.to_numpy();cnt={}
    for sec in (300,900,1800):
        for ph in ('ARR','DEP'): cnt[(sec,ph)]=np.zeros(len(f),dtype='float32')
    for a,b in zip(st,en):
        tt=t[a:b]
        for sec in (300,900,1800):
            lo=np.searchsorted(tt,tt-sec,'left');hi=np.searchsorted(tt,tt+sec,'right')
            for ph in ('ARR','DEP'):
                own=phase[a:b]==ph;cs=np.r_[0,np.cumsum(own)]
                cnt[(sec,ph)][a:b]=cs[hi]-cs[lo]-own
    for (sec,ph),v in cnt.items(): z[f'count_{ph}_{sec}s']=v
    z['is_arrival']=f.PHASE_mvt.eq('ARR').astype(float)
    nm=(f[TIME]-f.AOBT_3_flt).dt.total_seconds().to_numpy()
    good=f.PHASE_mvt.eq('DEP').to_numpy() & np.isfinite(nm) & (nm>0) & (nm<=10800)
    anchor=np.where(good,np.maximum(nm,60),600.).astype('float32')
    z['anchor_missing']=~good
    return z.astype('float32'),anchor

def windows(f, centres):
    _,_,start,end=bounds(f)
    n=centres[:,None]+np.arange(-HALF,HALF+1,dtype='int64')[None,:]
    valid=(n>=start[centres,None]) & (n<end[centres,None])
    clipped=np.clip(n,0,len(f)-1)
    t=f[TIME].astype('int64').to_numpy()/1e9
    dt=t[clipped]-t[centres,None]
    valid &= np.abs(dt)<=SECONDS
    n=np.where(valid,n,-1).astype('int32');dt=np.where(valid,dt,0).astype('float32')
    if not np.array_equal(n[:,HALF],centres): raise ValueError('Centre token changed')
    return centres,n,dt

def neighbours(f):
    return windows(f,np.flatnonzero(f.PHASE_mvt.eq('DEP')).astype('int64'))

def assert_loss_contract(cons, loss, target):
    cons.assert_training_frame(loss,target,what='lane G train-loss labels')

def coverage(f, dep_centres, dep_windows):
    records=[]
    phase=f.PHASE_mvt.to_numpy()
    for ph in ('ARR','DEP'):
        centres=dep_centres if ph=='DEP' else np.flatnonzero(phase=='ARR')
        nb=dep_windows if ph=='DEP' else windows(f,centres)[1]
        safe=np.maximum(nb,0);ok=nb>=0;ok[:,HALF]=False
        frame=f.iloc[centres][['airport','missing_runway']].reset_index(drop=True)
        frame['UTC_month']=f.iloc[centres][TIME].dt.month.to_numpy()
        frame['ARR_neighbours']=((phase[safe]=='ARR') & ok).sum(1)
        frame['DEP_neighbours']=((phase[safe]=='DEP') & ok).sum(1)
        frame['truncated_windows']=(nb<0).any(1)
        frame['rows']=1
        grouped=frame.groupby(['airport','UTC_month']).agg(rows=('rows','sum'),
            missing_runway=('missing_runway','sum'),ARR_neighbours=('ARR_neighbours','sum'),
            DEP_neighbours=('DEP_neighbours','sum'),truncated_windows=('truncated_windows','sum'))
        for (airport,month),r in grouped.iterrows():
            records.append({'phase':ph,'airport':airport,'UTC_month':int(month),
                **{k:int(v) for k,v in r.items()},
                'mean_ARR_neighbours':float(r.ARR_neighbours/r.rows),
                'mean_DEP_neighbours':float(r.DEP_neighbours/r.rows)})
    return {'truncated_definition':'fewer than 17 tokens after group and 30-minute bounds',
            'neighbour_counts_exclude_centre':True,'by_phase_UTC_month_airport':records}

def encode(f,z,mask):
    if not mask.any(): raise ValueError('No admitted encoder rows')
    # Fixed arithmetic dtype prevents held values changing a column's inferred
    # dtype and hence changing train-only quantile rounding.
    fit=z.loc[mask].astype('float64')
    med=fit.median().fillna(0)
    scale=(fit.quantile(.75)-fit.quantile(.25)).fillna(1)
    scale=scale.where(scale>0,1)
    num=((z.astype('float64').fillna(med)-med)/scale).to_numpy('float32')
    num=np.concatenate([num,z.isna().to_numpy('float32')],axis=1)
    cat=np.zeros((len(f),len(CATS)),dtype='int32');voc={}
    for j,c in enumerate(CATS):
        s=f[c].astype('string').fillna('<MISSING>');counts=s[mask].value_counts()
        vals=['<MISSING>']+sorted(v for v,n in counts.items() if n>=20 and v!='<MISSING>')
        codes={v:i+1 for i,v in enumerate(vals)}
        cat[:,j]=s.map(codes).fillna(0).to_numpy('int32');voc[c]=codes
    if not np.isfinite(num).all(): raise ValueError('Encoded features not finite')
    return num,cat,{'columns':list(z.columns),'missing_bit_columns':list(z.columns),
        'medians':med.to_dict(),'iqr_scales':scale.to_dict(),'vocab':voc,'fit_event_rows':int(mask.sum())}

def main():
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();started=time.time()
    if a.out.exists(): raise ValueError('Input root exists; write-once')
    a.out.mkdir(parents=True)
    st,_,_,_,_=R.load();cons=st.P.consumer('janjul')
    from prc import gscore as GS
    files=sorted(a.raw.glob('training_2025-*.parquet'))
    if len(files)!=12: raise ValueError('Need twelve 2025 raw sources')
    sources={};parts=[]
    for p in files:
        sources[str(p)]=sha(p);part=pd.read_parquet(p,columns=COLS,use_threads=False);parts.append(part)
        print(json.dumps({'source':p.name,'events':len(part)}),flush=True)
    raw=pd.concat(parts,ignore_index=True);del parts
    duplicates=raw[ID].duplicated(keep=False)
    if duplicates.any():
        unique=raw.loc[duplicates].drop_duplicates(COLS)
        if unique[ID].duplicated().any(): raise ValueError('Repeated key has different inputs')
    duplicate_count=int(raw[ID].duplicated().sum());raw=raw.drop_duplicates(ID)
    f=events(raw);del raw
    z,anchor=numerics(f);centres,nb,dt=neighbours(f)
    days=f.day.to_numpy();year=f[TIME].dt.year.to_numpy();month=f[TIME].dt.month.to_numpy()
    train_events=(year==2025) & ~np.isin(month,[1,7]) & ~np.isin(days,cons.excluded)
    # All runways of one airport-day have the same development owner.
    day_key=f.airport+'|'+f.day
    unique_days=day_key.unique()
    val_keys={v for v in unique_days if int(hashlib.sha256(('150071|'+v).encode()).hexdigest()[:16],16)%10==0}
    valid=day_key.isin(val_keys).to_numpy()
    dev_events=train_events & ~valid
    def save(name,v): np.save(a.out/(name+'.npy'),v,allow_pickle=False)
    for name,mask in [('development',dev_events),('full',train_events)]:
        num,cat,rec=encode(f,z,mask);save('numeric-'+name,num);save('category-'+name,cat)
        (a.out/('encoder-'+name+'.json')).write_text(json.dumps(rec,indent=2)+'\n');del num,cat
    loss=[]
    for p in files:
        lab=pd.read_parquet(p,columns=[ID,'PHASE_mvt',TIME,'TAXITIME_SEC_mvt'],use_threads=False)
        lab[TIME]=pd.to_datetime(lab[TIME],utc=True)
        d=lab[TIME].dt.strftime('%Y-%m-%d')
        take=lab.PHASE_mvt.eq('DEP') & lab[TIME].dt.year.eq(2025) & ~lab[TIME].dt.month.isin([1,7]) & ~d.isin(cons.excluded)
        loss.append(lab.loc[take])
    loss=pd.concat(loss,ignore_index=True)
    if loss[ID].duplicated().any():
        loss=loss.drop_duplicates()
        if loss[ID].duplicated().any(): raise ValueError('Loss key has different admitted labels')
    assert_loss_contract(cons,loss,'TAXITIME_SEC_mvt')
    indices=pd.Series(np.arange(len(f)),index=f[ID]).reindex(loss[ID]).to_numpy()
    if not np.isfinite(indices).all() or not np.isfinite(loss.TAXITIME_SEC_mvt.to_numpy()).all(): raise ValueError('Loss alignment/label missing')
    train_ix=indices.astype('int64');train_y=loss.TAXITIME_SEC_mvt.to_numpy('float32')
    expected=np.flatnonzero(train_events & f.PHASE_mvt.eq('DEP').to_numpy())
    if set(train_ix)!=set(expected): raise ValueError('Train-loss departure ownership differs')
    held=GS.load_rows();held=held[held.month.isin([1,7])]
    held_ids=held[ID].to_numpy('int64')
    held_ix=pd.Series(np.arange(len(f)),index=f[ID]).reindex(held_ids).to_numpy()
    if len(held_ids)!=344419 or not np.isfinite(held_ix).all(): raise ValueError('Held IDs incomplete')
    if not f.iloc[held_ix.astype('int64')].PHASE_mvt.eq('DEP').all(): raise ValueError('Held key is not a departure')
    if set(train_ix) & set(held_ix): raise ValueError('Held key is in loss')
    save('anchor',anchor);save('departure-centres',centres);save('neighbours',nb);save('elapsed-seconds',dt)
    save('train-indices',train_ix);save('train-labels',train_y);save('train-development-valid',valid[train_ix])
    save('held-indices',held_ix.astype('int64'));save('held-ids',held_ids)
    # This metadata has no target or BLOCK field. IDs are for alignment only.
    meta=f[[ID,'PHASE_mvt',TIME,'airport','runway','day','group','missing_runway']].copy()
    meta['encoder_train']=train_events;meta['development_valid']=valid
    meta.to_parquet(a.out/'events.parquet',index=False)
    phase=f.PHASE_mvt.to_numpy();safe=np.maximum(nb,0);arr=(phase[safe]=='ARR') & (nb>=0)
    dep=(phase[safe]=='DEP') & (nb>=0);dep[:,HALF]=False
    summary={'events':len(f),'departures':len(centres),'arrivals':int(f.PHASE_mvt.eq('ARR').sum()),
        'train_labels':len(train_ix),'development_valid_labels':int(valid[train_ix].sum()),
        'held_rows':len(held_ids),'duplicate_input_rows_removed':duplicate_count,
        'departure_missing_runway':int(f.iloc[centres].missing_runway.sum()),
        'neighbours_mean_ARR':float(arr.sum(1).mean()),'neighbours_mean_DEP':float(dep.sum(1).mean()),
        'windows_with_ARR':int(arr.any(1).sum()),'padded_tokens':int((nb<0).sum()),
        'train_rows_by_UTC_month':loss.groupby(loss[TIME].dt.month).size().astype(int).to_dict()}
    (a.out/'COVERAGE.json').write_text(json.dumps(coverage(f,centres,nb),indent=2)+'\n')
    # Inspect label-free real windows from every airport and held month.
    inspect=[];lookup=pd.Series(np.arange(len(centres)),index=f.iloc[centres][ID])
    for m in (1,7):
        for airport in sorted(f.airport.unique()):
            sel=held_ids[(held.month.to_numpy()==m) & (held.airport.to_numpy()==airport)]
            candidates=lookup.reindex(sel).dropna().to_numpy('int64')
            j=next((int(i) for i in candidates if arr[i].any()),int(candidates[0]) if len(candidates) else None)
            if j is None: continue
            window=nb[j];window=window[window>=0]
            inspect.append({'airport':airport,'month':m,'centre':int(f.iloc[centres[j]][ID]),
                'tokens':meta.iloc[window][[ID,'PHASE_mvt',TIME,'runway']].astype(str).to_dict('records')})
    (a.out/'INSPECT-WINDOWS.json').write_text(json.dumps(inspect,indent=2)+'\n')
    summary['seconds']=time.time()-started;summary['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    record={'status':'built; independent check required','sources_sha256':sources,'input_columns':COLS,
        'category_columns':CATS,'split':cons.manifest(),'summary':summary,
        'no_held_labels_in_payload':True,'label_fields_are_loss_only':['train-labels.npy'],
        'code_sha256':sha(Path(__file__)),'SPEC_sha256':sha(EXP/'sequence-g/SPEC.md'),
        'files':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}}
    (a.out/'MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
